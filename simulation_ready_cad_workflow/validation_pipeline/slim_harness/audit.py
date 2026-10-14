"""Check original solids at supplied poses or uniformly sampled saved motion."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import math
import time
import numpy as np
import trimesh
import fcl

from ..mechanism.io import write_json_atomic
from .implementation import source_sha256
from .validation_scope import validate_validation_scope, flexible_regions_for_pair


CLEARANCE_PROOF_MARGIN_M = 1e-8
AUDIT_IMPLEMENTATION_SHA256 = hashlib.sha256(
    (source_sha256("audit.py")+source_sha256("validation_scope.py")).encode("ascii")).hexdigest()
SOURCE_CLASSIFIER_TOLERANCE_MM = 1e-7


def rotation(q):
    return trimesh.transformations.quaternion_matrix(q)[:3,:3]


def _select_saved_pose_samples(samples):
    """Keep Slim's existing 10 Hz policy independent of legacy audit policy."""
    if not samples:
        raise RuntimeError("INTERPENETRATION_AUDIT_SAMPLES_MISSING: physical trace has no saved samples")
    times = [float(item["time_s"]) for item in samples]
    if any(not math.isfinite(item) for item in times) or any(
            right < left for left, right in zip(times, times[1:])):
        raise RuntimeError("INTERPENETRATION_AUDIT_SAMPLE_TIME_INVALID: sample times must be finite and monotonic")
    interval = 0.1
    selected_indices = {0, len(samples) - 1}
    next_time = times[0] + interval
    for index, time_s in enumerate(times[1:], start=1):
        if time_s + 1.0e-12 < next_time:
            continue
        selected_indices.add(index)
        while next_time <= time_s + 1.0e-12:
            next_time += interval
    indices = sorted(selected_indices)
    selected = []
    for index in indices:
        item = dict(samples[index])
        item["_audit_source_sample_index"] = int(item.get("_audit_source_sample_index", index))
        selected.append(item)
    return selected, {
        "schema_version": "0.1",
        "mode": "progressive_saved_trace_uniform_10hz",
        "nominal_interval_s": interval,
        "original_saved_sample_count": len(samples),
        "selected_sample_count": len(selected),
        "selected_source_indices": indices,
        "trace_endpoints_retained": True,
        "phase_boundaries_forced": False,
        "contact_transitions_forced": False,
        "progressive_pair_budget_enabled": False,
        "every_saved_pose_mesh_audited": len(selected) == len(samples),
    }


def audit_saved_motion(prepared: dict, samples, output: Path) -> dict:
    """Apply the approved 10 Hz saved-trace policy; retain exact endpoints.

    Cheap validation still sees every recorded state. The geometry kernel is
    deliberately separate so initial/prescribed pose callers keep all poses.
    """
    body_ids = {body["id"] for body in prepared["bodies"]}
    conditional = {row["active_element_id"] for row in (prepared.get("validation_scope") or {}).get("flexible_contact_regions", [])
                   if row.get("active_element_id") is not None}
    for sample in samples:
        if "time_s" not in sample:
            raise RuntimeError("Saved motion omitted its sample time")
        if set(sample.get("body_poses", {})) != body_ids:
            raise RuntimeError("Solver sample omitted or added an original body pose")
        for pose in sample["body_poses"].values():
            position = np.asarray(pose.get("position_abs_m"), dtype=float)
            quat = np.asarray(pose.get("rotation_abs_quat_wxyz"), dtype=float)
            if (position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all()
                    or not np.isfinite(quat).all() or np.linalg.norm(quat) == 0):
                raise RuntimeError("Saved motion contains an invalid body pose")
        states = sample.get("equivalent_elasticity", {}).get("elements", {})
        if any(not isinstance(states.get(key, {}).get("active"), bool) for key in conditional):
            raise RuntimeError("Saved motion omitted conditional support activity")
    selected, policy = _select_saved_pose_samples(samples)
    report = audit_motion(prepared, selected, output)
    policy.update(progressive_pair_budget_enabled=False, stop_rule="first_confirmed_violation")
    report.update(available_solver_samples=len(samples), candidate_solver_samples=len(selected),
                  sampling_policy=policy, audited_source_indices=policy["selected_source_indices"][:report["checked_solver_samples"]],
                  scope="Original geometry at uniform 10 Hz saved poses with exact endpoints; no claim for unsampled states or continuous time")
    write_json_atomic(output, report)
    return report


def audit_saved_request(request_path: Path, output: Path, *, trace_path=None) -> dict:
    """Audit an existing request/trace without rerunning physics or publishing it."""
    import json
    from .trace_storage import file_digest, open_trace, open_trace_prefix, release_trace_cache
    request_path, output = Path(request_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Choose a new audit output; existing evidence is preserved")
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    prepared = request["scene"]
    path = Path(trace_path or request["output"]).resolve()
    for body in prepared["bodies"]:
        for file_key, hash_key in (("source_file", "source_sha256"), ("mesh_file", "mesh_sha256")):
            if file_digest(Path(body[file_key])) != body[hash_key]:
                raise ValueError("Saved audit source binding changed: " + body[file_key])
        for component in body.get("source_components", []):
            if file_digest(Path(component["mesh_file"])) != component["sha256"]:
                raise ValueError("Saved audit component binding changed: " + component["mesh_file"])
    prefix = path.name == "prefix.json"
    trace = open_trace_prefix(path) if prefix else open_trace(path)
    try:
        report = audit_saved_motion(prepared, trace["samples"], output)
        report.update(request_file=str(request_path), trace_file=str(path), certified=False,
                      prefix_only=prefix, recorded_time_range_s=[trace["samples"][0]["time_s"], trace["samples"][-1]["time_s"]],
                      requested_duration_s=request["test"]["duration_s"], physics_rerun=False)
        write_json_atomic(output, report)
        return report
    finally:
        release_trace_cache(trace)


def audit_motion(prepared: dict, samples: list[dict], output: Path) -> dict:
    started = time.monotonic()
    scope = prepared.get("validation_scope")
    if scope is not None:
        validate_validation_scope(scope, prepared["bodies"])
    excluded = set(scope["elastic_reference_body_ids"]) if scope else set()
    all_body_ids = {body["id"] for body in prepared["bodies"]}
    if not samples:
        raise RuntimeError("Geometry audit requires actual solver samples")
    meshes, solids, witnesses, collision_objects = {}, {}, {}, {}
    local_bounds, source_radii, source_solid_corners = {}, {}, {}
    source_parts, source_centers, source_classifiers = {}, {}, {}
    records = {b["id"]:b for b in prepared["bodies"] if b["id"] not in excluded}
    manager = trimesh.collision.CollisionManager()
    for body in prepared["bodies"]:
        key = body["id"]
        if key in excluded:
            continue
        # This combined surface is suitable for a BVH, not for parity-based
        # material tests. Welding touching source solids can make it nonmanifold.
        mesh = trimesh.load_mesh(body["mesh_file"],process=False)
        meshes[key] = mesh
        manager.add_object(key,mesh)
        model = fcl.BVHModel()
        model.beginModel(len(mesh.vertices),len(mesh.faces))
        model.addSubModel(np.asarray(mesh.vertices),np.asarray(mesh.faces,dtype=np.int32))
        model.endModel()
        collision_objects[key] = fcl.CollisionObject(model)
        components = body.get("source_components") or []
        component_ids = [component["solid_id"] for component in components]
        if not components or len(set(component_ids)) != len(component_ids) or set(component_ids) != set(body["source_solid_ids"]):
            raise RuntimeError(f"Original source-part inventory is incomplete: {key}")
        parts, part_witnesses = [], []
        for component in components:
            part = trimesh.load_mesh(component["mesh_file"],process=False)
            if not isinstance(part,trimesh.Trimesh) or not part.is_watertight or not part.is_winding_consistent or part.volume <= 0:
                raise RuntimeError(f"Original source part is not a valid closed material: {key}/{component['solid_id']}")
            parts.append(part)
            # Split only within this ORIGINAL closed part, never across the
            # body union. Cavity shells stay together for target classification.
            part_witnesses.extend(shell.vertices[0] for shell in part.split(only_watertight=False))
        source_parts[key] = parts
        witnesses[key] = np.asarray(part_witnesses,dtype=float)
        source = Path(body["source_file"])
        if source.suffix.lower() in {".step",".stp"}:
            from ..intake.step_shape_cache import load_step_shape
            imported = load_step_shape(source)
            indices = [int(x.split("_")[-1])-1 for x in body["source_solid_ids"]]
            solids[key] = [imported.solids[index] for index in indices]
            center = np.asarray(body.get("source_position_m",body["position_m"]))
            source_centers[key] = center
            from OCP.BRepClass3d import BRepClass3d_SolidClassifier
            source_classifiers[key] = [
                (np.asarray([imported.inventory[index]["bounds_mm"]["min"],imported.inventory[index]["bounds_mm"]["max"]]),
                 BRepClass3d_SolidClassifier(imported.solids[index]))
                for index in indices
            ]
            source_solid_corners[key] = [
                trimesh.bounds.corners(part_bounds*.001-center)
                for part_bounds,_ in source_classifiers[key]
            ]
            minimum = np.min([np.asarray(imported.inventory[index]["bounds_mm"]["min"])*.001 for index in indices],axis=0)-center
            maximum = np.max([np.asarray(imported.inventory[index]["bounds_mm"]["max"])*.001 for index in indices],axis=0)-center
            # These bounds come from the original BRep, not an inscribed
            # tessellation. Their radius encloses every original source point.
            local_bounds[key] = np.asarray([minimum,maximum])
            source_radii[key] = float(np.linalg.norm(np.maximum(np.abs(minimum),np.abs(maximum))))
        else:
            local_bounds[key] = meshes[key].bounds
    violations, candidate_count, checked = [], 0, 0
    max_overlap = 0.
    transforms = {}
    source_transform_cache, source_bounds_cache = {}, {}
    clearance_proofs = {}
    proof_retry = {}
    cached_proof_samples, boolean_checks, clearance_queries = 0, 0, 0
    clearance_queries_deferred = 0
    cached_solver_samples, cached_pairs = set(), {}
    step_classification_queries, mesh_part_contains_calls = 0, 0
    excluded_region_volume, region_cut_checks = 0., 0

    def original_material_contains(key,points_local):
        nonlocal step_classification_queries, mesh_part_contains_calls
        if key in source_classifiers:
            from OCP.TopAbs import TopAbs_IN, TopAbs_UNKNOWN
            from OCP.gp import gp_Pnt
            points_mm = (points_local+source_centers[key])*1000.
            for part_bounds,classifier in source_classifiers[key]:
                inside_bounds = np.all(points_mm >= part_bounds[0]-SOURCE_CLASSIFIER_TOLERANCE_MM,axis=1)&np.all(points_mm <= part_bounds[1]+SOURCE_CLASSIFIER_TOLERANCE_MM,axis=1)
                for point in points_mm[inside_bounds]:
                    step_classification_queries += 1
                    classifier.Perform(gp_Pnt(*map(float,point)),SOURCE_CLASSIFIER_TOLERANCE_MM)
                    state = classifier.State()
                    if state == TopAbs_IN: return True
                    if state == TopAbs_UNKNOWN:
                        raise RuntimeError(f"Original STEP material classification is unresolved: {key}")
            return False
        for part in source_parts[key]:
            part_bounds = part.bounds
            inside_bounds = np.all(points_local >= part_bounds[0]-CLEARANCE_PROOF_MARGIN_M,axis=1)&np.all(points_local <= part_bounds[1]+CLEARANCE_PROOF_MARGIN_M,axis=1)
            if inside_bounds.any():
                mesh_part_contains_calls += 1
                # Material is the UNION of original parts. A cavity remains
                # inside its own part's parity test; overlapping parts do not
                # cancel each other as they would in a combined-body ray cast.
                if np.any(part.contains(points_local[inside_bounds])): return True
        return False

    def source_transform(key):
        if key in source_transform_cache:
            return source_transform_cache[key]
        from OCP.gp import gp_Quaternion, gp_Trsf, gp_Vec
        matrix = transforms[key]
        quat = trimesh.transformations.quaternion_from_matrix(matrix)
        trsf = gp_Trsf()
        trsf.SetRotation(gp_Quaternion(*quat[1:],quat[0]))
        shift = matrix[:3,3]-matrix[:3,:3]@np.asarray(records[key].get("source_position_m",records[key]["position_m"]))
        trsf.SetTranslationPart(gp_Vec(*(shift*1000)))
        source_transform_cache[key] = trsf
        return trsf

    def transformed_solid(key,index):
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        # Common/Cut may update argument sub-shapes in their default mode.
        # A fresh copy prevents one pair's Boolean from affecting another pair.
        return BRepBuilderAPI_Transform(
            solids[key][index],source_transform(key),True).Shape()

    def transformed_solids(key):
        return [transformed_solid(key,index) for index in range(len(solids[key]))]

    def source_solid_bounds(key):
        if key not in source_bounds_cache:
            bounds = []
            for corners in source_solid_corners[key]:
                points = trimesh.transform_points(corners,transforms[key])
                # Original BRep enclosures, expanded outward; touching remains
                # a Boolean candidate, including roundoff near a shared bound.
                bounds.append((points.min(axis=0)-CLEARANCE_PROOF_MARGIN_M,
                               points.max(axis=0)+CLEARANCE_PROOF_MARGIN_M))
            source_bounds_cache[key] = bounds
        return source_bounds_cache[key]

    region_shapes = {}
    if scope:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
        for row in scope["flexible_contact_regions"]:
            if any(key not in solids for key in row["body_pair"]):
                raise RuntimeError("Local flexible contact scope requires original STEP solids")
            start, end = row["axial_interval_m"]
            axis = np.asarray(row["axis_direction_source"])
            origin = np.asarray(row["axis_origin_source_m"])+axis*start
            region_shapes[row["id"]] = BRepPrimAPI_MakeCylinder(
                gp_Ax2(gp_Pnt(*(origin*1000)), gp_Dir(*axis)), row["radius_m"]*1000,
                (end-start)*1000).Shape()

    current_element_states = {}
    conditional_elements = {row["active_element_id"] for row in (scope or {}).get("flexible_contact_regions", [])
                            if row.get("active_element_id") is not None}

    def cut_scoped_regions(shape, a, b):
        nonlocal excluded_region_volume, region_cut_checks
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        for row in flexible_regions_for_pair(scope or {}, (a,b)):
            if row.get("active_element_id") is not None and not current_element_states[row["active_element_id"]]["active"]:
                continue
            properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(shape, properties)
            before = abs(properties.Mass())*1e-9
            region = BRepBuilderAPI_Transform(region_shapes[row["id"]],
                        source_transform(row["region_owner_body"]), True).Shape()
            cut = BRepAlgoAPI_Cut(shape, region)
            if not cut.IsDone(): cut.Build()
            if not cut.IsDone():
                raise RuntimeError("Flexible-region Boolean subtraction failed: " + row["id"])
            region_cut_checks += 1
            shape = cut.Shape()
            BRepGProp.VolumeProperties_s(shape, properties)
            excluded_region_volume += max(0., before-abs(properties.Mass())*1e-9)
        return shape

    def overlap_volume(a,b):
        nonlocal boolean_checks
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        result = 0.
        for left_index,left_bounds in enumerate(source_solid_bounds(a)):
            for right_index,right_bounds in enumerate(source_solid_bounds(b)):
                if np.any(left_bounds[1] < right_bounds[0]) or np.any(right_bounds[1] < left_bounds[0]):
                    continue
                left = transformed_solid(a,left_index)
                right = transformed_solid(b,right_index)
                operation = BRepAlgoAPI_Common(left,right)
                # The two-shape constructor already computes the Common.
                # Building again repeats the entire native Boolean operation.
                if not operation.IsDone(): operation.Build()
                if not operation.IsDone(): raise RuntimeError(f"Original STEP intersection failed: {a}, {b}")
                boolean_checks += 1
                properties = GProp_GProps()
                intersection = cut_scoped_regions(operation.Shape(), a, b)
                BRepGProp.VolumeProperties_s(intersection,properties)
                result += abs(properties.Mass())*1e-9
        return result

    def relative_transform(reference,moving):
        reference_pose,moving_pose = transforms[reference],transforms[moving]
        result = np.eye(4)
        result[:3,:3] = reference_pose[:3,:3].T@moving_pose[:3,:3]
        result[:3,3] = reference_pose[:3,:3].T@(moving_pose[:3,3]-reference_pose[:3,3])
        return result

    def proof_still_separates(proof):
        current = relative_transform(proof["reference"],proof["moving"])
        previous = proof["relative_pose"]
        translation = float(np.linalg.norm(current[:3,3]-previous[:3,3]))
        # For rotations, ||R1-R0||_F/sqrt(2) = 2*sin(delta_angle/2).
        # This form retains precision for the tiny measured constraint drift.
        rotation_displacement = float(np.linalg.norm(current[:3,:3]-previous[:3,:3],ord="fro")/np.sqrt(2))*proof["radius_m"]
        bound = translation+rotation_displacement+CLEARANCE_PROOF_MARGIN_M
        return bool(np.isfinite(bound) and bound < proof["clearance_m"])

    def establish_clearance_proof(a,b):
        nonlocal clearance_queries
        from OCP.BRepExtrema import BRepExtrema_DistShapeShape
        distances = []
        minimum = None
        try:
            for left_index,left_bounds in enumerate(source_solid_bounds(a)):
                for right_index,right_bounds in enumerate(source_solid_bounds(b)):
                    separation = np.maximum(np.maximum(left_bounds[0]-right_bounds[1],
                                                       right_bounds[0]-left_bounds[1]),0.)
                    lower_bound = float(np.linalg.norm(separation))
                    # Expanded original BRep enclosures give a distance lower
                    # bound, never a substitute for an actual BRep minimum.
                    if minimum is not None and lower_bound >= minimum:
                        continue
                    left = transformed_solid(a,left_index)
                    right = transformed_solid(b,right_index)
                    clearance_queries += 1
                    query = BRepExtrema_DistShapeShape(left,right)
                    if not query.IsDone(): query.Perform()
                    if not query.IsDone(): return None
                    value = float(query.Value())*.001
                    if value <= CLEARANCE_PROOF_MARGIN_M:
                        return None
                    distances.append(value)
                    minimum = value if minimum is None else min(minimum,value)
        except (RuntimeError,ValueError):
            # This is an optional acceleration proof. An unavailable distance
            # query leaves the original per-pose Boolean checks in place.
            return None
        distance = min(distances)
        if not np.isfinite(distance) or distance <= CLEARANCE_PROOF_MARGIN_M:
            return None
        moving = min((a,b),key=lambda key:source_radii[key])
        reference = b if moving==a else a
        return {"reference":reference,"moving":moving,"radius_m":source_radii[moving],
                "relative_pose":relative_transform(reference,moving),"clearance_m":distance,"reused_samples":0}

    def defer_new_proof(pair,sample_index,*,short_lived=False):
        # Failed/one-step certificates can cost more than the Boolean itself.
        # Back off ONLY this optional distance query; all uncached solver poses
        # still receive the original geometry checks below.
        delay = max(proof_retry.get(pair,{}).get("delay",1),8 if short_lived else 1)
        proof_retry[pair] = {"after":sample_index+delay,"delay":min(64,delay*2)}

    passed_pair_poses = {}
    unchanged_pair_checks = 0
    for sample_index,sample in enumerate(samples):
        current_element_states = sample.get("equivalent_elasticity", {}).get("elements", {})
        for identifier in conditional_elements:
            if not isinstance(current_element_states.get(identifier, {}).get("active"), bool):
                raise RuntimeError("Conditional original-geometry audit requires recorded support activity: " + identifier)
        source_transform_cache.clear()
        source_bounds_cache.clear()
        if set(sample["body_poses"]) != all_body_ids:
            raise RuntimeError("Solver sample omitted or added an original body pose")
        bounds = {}
        for key in records:
            pose = sample["body_poses"][key]
            matrix = trimesh.transformations.quaternion_matrix(pose["rotation_abs_quat_wxyz"])
            matrix[:3,3] = pose["position_abs_m"]
            transforms[key] = matrix
            manager.set_transform(key,matrix)
            collision_objects[key].setTransform(fcl.Transform(matrix[:3,:3],matrix[:3,3]))
            corners = trimesh.bounds.corners(local_bounds[key])
            corners = trimesh.transform_points(corners,matrix)
            padding = CLEARANCE_PROOF_MARGIN_M if key in solids else 0.
            bounds[key] = (corners.min(axis=0)-padding,corners.max(axis=0)+padding)
        collision, names = manager.in_collision_internal(return_names=True)
        names = {tuple(sorted(pair)) for pair in names}
        for a,b in combinations(records,2):
            pair = tuple(sorted((a,b)))
            relative = relative_transform(a,b)
            activity = tuple((row["active_element_id"], current_element_states[row["active_element_id"]]["active"])
                             for row in flexible_regions_for_pair(scope or {}, pair)
                             if row.get("active_element_id") is not None)
            previous = passed_pair_poses.get(pair)
            if previous is not None and previous[1] == activity and np.array_equal(previous[0], relative):
                unchanged_pair_checks += 1
                continue
            passed_pose = (relative, activity)
            proof = clearance_proofs.get(pair)
            if proof is not None:
                if proof_still_separates(proof):
                    cached_proof_samples += 1
                    proof["reused_samples"] += 1
                    proof_retry.pop(pair,None)
                    cached_solver_samples.add(sample_index)
                    cached_pairs["/".join(pair)] = cached_pairs.get("/".join(pair),0)+1
                    passed_pair_poses[pair] = passed_pose
                    continue
                if proof["reused_samples"] == 0:
                    defer_new_proof(pair,sample_index,short_lived=True)
                del clearance_proofs[pair]
            if np.any(bounds[a][1] < bounds[b][0]) or np.any(bounds[b][1] < bounds[a][0]):
                passed_pair_poses[pair] = passed_pose
                continue
            contained = False
            if pair not in names:
                for source,target in ((a,b),(b,a)):
                    points = trimesh.transform_points(witnesses[source],relative_transform(target,source))
                    if len(points) and original_material_contains(target,points):
                        contained = True
                        break
            near_original_surface = False
            if pair not in names and not contained and (a in solids or b in solids):
                # OCCT curved surfaces can extend outside tessellated chords.
                # Include the stated deflection of BOTH original surfaces.
                separation = fcl.distance(collision_objects[a],collision_objects[b],fcl.DistanceRequest(),fcl.DistanceResult())
                near_original_surface = separation <= .000105
            if pair not in names and not contained and not near_original_surface:
                passed_pair_poses[pair] = passed_pose
                continue
            candidate_count += 1
            if a in solids and b in solids:
                overlap = overlap_volume(a,b)
                max_overlap = max(max_overlap,overlap)
                # 0.001 mm^3 absorbs Boolean roundoff, not visible overlaps.
                if overlap > 1e-12:
                    violations.append({"time_s":sample["time_s"],"body_a":a,"body_b":b,
                                       "kind":"original_solid_interpenetration","overlap_m3":overlap})
                else:
                    passed_pair_poses[pair] = passed_pose
                if overlap == 0.:
                    if sample_index >= proof_retry.get(pair,{}).get("after",0):
                        proof = establish_clearance_proof(a,b)
                        if proof is not None: clearance_proofs[pair] = proof
                        else: defer_new_proof(pair,sample_index)
                    else:
                        clearance_queries_deferred += 1
            else:
                # Do not label an ambiguous intersecting triangle surface PASS.
                # The caller can request denser/local evidence without reauthoring.
                violations.append({"time_s":sample["time_s"],"body_a":a,"body_b":b,
                    "kind":"mesh_containment" if contained else "mesh_surface_intersection_needs_local_evidence"})
        checked += 1
        if violations: break
    equivalent = scope is not None and scope["mode"] == "equivalent_elastic"
    report = {"schema_version":"slim_geometry_audit_v2" if equivalent else "slim_geometry_audit_v1",
              "passed":not violations and checked==len(samples),
              "audit_implementation_sha256":AUDIT_IMPLEMENTATION_SHA256,
              "checked_solver_samples":checked,"available_solver_samples":len(samples),
              "intersecting_candidates":candidate_count,"max_overlap_m3":max_overlap,
              "boolean_checks":boolean_checks,"clearance_queries":clearance_queries,
              "clearance_queries_deferred":clearance_queries_deferred,
              "cached_proof_samples":cached_proof_samples,"cached_proof_solver_samples":len(cached_solver_samples),
              "cached_proof_pairs":cached_pairs,
              "unchanged_pair_checks_reused":unchanged_pair_checks,
              "unchanged_pair_policy":"Exact unchanged relative pose and identical active local exclusions within this source-bound audit; moving-to-static pairs remain checked",
              "containment":{"source_part_counts":{key:len(parts) for key,parts in source_parts.items()},
                  "witness_counts":{key:len(points) for key,points in witnesses.items()},
                  "step_classification_queries":step_classification_queries,"mesh_part_contains_calls":mesh_part_contains_calls,
                  "step_classifier_tolerance_mm":SOURCE_CLASSIFIER_TOLERANCE_MM,
                  "material_semantics":"union of original closed parts; cavities retained within each part"},
              "clearance_proof_policy":{"source":"original_step_brep","numerical_margin_m":CLEARANCE_PROOF_MARGIN_M,
                  "motion_bound":"relative translation + 2*source radius*sin(relative rotation angle/2)",
                  "uses_assumed_joint_rigidity":False,"counter_unit":"body_pair_at_solver_sample"},
              "boolean_roundoff_volume_m3":1e-12,"violations":violations,
              "scope":(scope["dynamic_geometry"] if scope else "original geometry at every solver state")
                      + "; no claim between solver steps",
              "elapsed_seconds":time.monotonic()-started}
    if scope is not None:
        report["validation_scope"] = scope
        report["equivalent_elastic_exclusions"] = {
            "elastic_reference_body_ids": sorted(excluded),
            "checked_rigid_body_ids": sorted(records),
            "region_cut_checks": region_cut_checks,
            "excluded_intersection_volume_m3_sum": excluded_region_volume,
            "contact_records_modified": False,
            "conditional_element_ids": sorted(conditional_elements),
        }
    write_json_atomic(output,report)
    return report
