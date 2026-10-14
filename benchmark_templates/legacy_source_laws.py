"""旧源已声明本构的忠实接入：只构造源载荷并更新内力，绝不推进时间。

来源：mechanism/compound_runtime.py 的exact pins、soft limits和compliant surfaces；
mechanism/detent_slider_runtime.py 的两bushing、有限gate、预载和行程stop。
整源CAD入口的质量/惯量来自OCCT，不声称与旧估算质量整体等价。
单独的plain两体组件入口忠实保留旧声明质量/包围盒惯量，且明确不代表完整源CAD。
"""

from pathlib import Path
import copy, math
import numpy as np


def prepare_laws(
    family_specs,
    geometry,
    output,
    source_root,
    *,
    capsules=None,
    contract=None,
    source_geometry_evidence=None,
    source_fixture=None,
):
    import trimesh
    from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import _write_obj
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_spec import (
        compound_linkage_from_dict,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_runtime import (
        _validated_compliant_surface_contacts,
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_spec import (
        prismatic_detent_latch_from_dict,
    )

    records = {r["id"]: r for r in geometry["bodies"]}
    laws = []
    frictions = []
    for reference, raw in family_specs.items():
        if (
            raw.get("compliant_surface_contacts")
            or raw.get("mechanism_kind") == "one_way_drive_clutch"
        ):
            spec = compound_linkage_from_dict(raw)
            if spec.analytical_transmission is not None:
                raise ValueError("UNSUPPORTED_ADDITIONAL_SOURCE_ANALYTICAL_TRANSMISSION")
            kind = "compound_source_pins_drag"
        elif raw.get("latch_press_joint", {}).get("spring_stiffness_n_m"):
            spec = prismatic_detent_latch_from_dict(raw)
            kind = "detent_source_bushings"
        elif raw.get("mechanism_kind") == "bicycle_chain_sprocket_drive":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
                bicycle_chain_from_dict,
            )

            spec = bicycle_chain_from_dict(raw)
            kind = "bicycle_source_chain_steering_fixture"
        elif raw.get("mechanism_kind") == "hydraulic_scissor_lift":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_spec import (
                scissor_lift_from_dict,
            )

            spec = scissor_lift_from_dict(raw)
            kind = "scissor_source_hydraulics"
        elif raw.get("mechanism_kind") == "axis_spring":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.axis_spring_spec import (
                axis_spring_from_dict,
            )

            spec = axis_spring_from_dict(raw, expected_family="revolute")
            kind = "plain_axis_source_spring"
        elif raw.get("mechanism_kind") == "click_pen_cam_latch":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.click_pen_spec import (
                click_pen_cam_latch_from_dict,
            )

            spec = click_pen_cam_latch_from_dict(raw)
            kind = "click_pen_source_bushings"
        elif raw.get("mechanism_kind") == "nail_clipper_compound_lever":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.nail_clipper_compound_lever import (
                NailClipperCompoundLeverAdapter,
            )

            spec = NailClipperCompoundLeverAdapter().parse_family_spec(raw)
            kind = "nail_source_bushings_spring"
        else:
            continue
        mapping = {}
        axis_source = {}
        if kind == "plain_axis_source_spring":
            matches = [
                cap
                for cap in (capsules or {}).values()
                if cap.get("reduced_model_claim", {}).get("family_spec_file") == reference
            ]
            if len(matches) != 1:
                raise ValueError("Axis spring needs its actual unique source capsule")
            claim = matches[0]["reduced_model_claim"]
            if len(claim["fixed_group_ids"]) != 1 or len(claim["moving_group_ids"]) != 1:
                raise ValueError("Axis spring pair must be explicit")
            mapping = {"fixed": claim["fixed_group_ids"][0], "moving": claim["moving_group_ids"][0]}
            frame = next(f for f in contract["frames"] if f["frame_id"] == claim["joint_frame_id"])
            axis_source = {"axis_abs": claim["axis_abs"], "anchor_abs_m": frame["origin_m"]}
        else:
            for body in spec.bodies:
                if len(body.group_ids) != 1 or body.group_ids[0] not in records:
                    raise ValueError(
                        "Source law body must bind one real prepared rigid group: " + body.body_id
                    )
                mapping[body.body_id] = body.group_ids[0]
        law = {
            "kind": kind,
            "family_spec": raw,
            "body_mapping": mapping,
            "source_file": str(Path(source_root) / reference),
            "source_runtime": {
                "compound_source_pins_drag": "mechanism/compound_runtime.py",
                "detent_source_bushings": "mechanism/detent_slider_runtime.py",
                "click_pen_source_bushings": "mechanism/click_pen_runtime.py",
                "nail_source_bushings_spring": "mechanism/nail_clipper_compound_lever.py",
                "plain_axis_source_spring": "mechanism/worker.py + family_plugins.PlainRevoluteAdapter",
                "scissor_source_hydraulics": "mechanism/scissor_lift_runtime.py",
                "bicycle_source_chain_steering_fixture": "mechanism/bicycle_chain_runtime.py + source steering capsule",
            }[kind],
            "scope": "Original source constitutive parameters and update equations; actual OCCT density-based mass/inertia retained, no legacy estimated-mass equivalence claim",
        }
        law.update(axis_source)
        if kind == "bicycle_source_chain_steering_fixture":
            import json, hashlib
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
                resolve_bicycle_contract_bindings,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_path import (
                build_moving_chain_path_contract,
            )

            if source_fixture != "bicycle_steering_at_rest" or source_geometry_evidence is None:
                raise ValueError(
                    "Bicycle source bench and original geometry preflight are required"
                )
            evidence_path = Path(source_geometry_evidence)
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))[
                "bicycle_chain_geometry"
            ]
            contract_file = Path(source_root) / "assembly_contract.json"
            contract_hash = hashlib.sha256(contract_file.read_bytes()).hexdigest()
            if (
                not evidence["passed"]
                or evidence["assembly_sha256"] != geometry["source_bindings"][0]["source_sha256"]
                or evidence["assembly_contract_sha256"] != contract_hash
            ):
                raise ValueError("BICYCLE_SOURCE_GEOMETRY_BINDING_MISMATCH")
            roles = resolve_bicycle_contract_bindings(
                spec, contract, assembly_contract_sha256=contract_hash
            )
            if roles["binding_sha256"] != evidence["contract_role_binding_sha256"]:
                raise ValueError("Bicycle role bindings differ from the original geometric proof")
            law["geometry_evidence"] = evidence
            law["validated_role_binding_sha256"] = roles["binding_sha256"]
            law["geometry_evidence_path"] = str(evidence_path)
            law["moving_chain_contract"] = build_moving_chain_path_contract(
                spec, evidence["visible_chain_geometry"]
            )
            steering = [
                x
                for x in capsules.values()
                if x.get("reduced_model_claim", {}).get("family") == "revolute"
            ]
            if len(steering) != 1:
                raise ValueError("Unique source steering capsule required")
            claim = steering[0]["reduced_model_claim"]
            frame = next(f for f in contract["frames"] if f["frame_id"] == claim["joint_frame_id"])
            law["steering"] = {
                "fixed": claim["fixed_group_ids"][0],
                "moving": claim["moving_group_ids"][0],
                "axis_abs": claim["axis_abs"],
                "anchor_abs_m": frame["origin_m"],
            }
            ground = next(
                x for x in family_specs.values() if x.get("mechanism_kind") == "bicycle_ground_roll"
            )
            front = next(
                b for b in ground["bodies"] if b["body_id"] == ground["front_wheel_body_id"]
            )["group_ids"]
            if len(front) != 1:
                raise ValueError("Front wheel group must be explicit")
            law["front_wheel"] = {
                "body": front[0],
                "parent": law["steering"]["moving"],
                "anchor": ground["front_axle_origin_abs_m"],
                "axis": ground["wheel_axis_abs"],
            }
            law["reference_only_body_ids"] = []
            for part in contract["parts"]:
                if part.get("role") == "context":
                    group = part["rigid_group_id"]
                    r = records[group]
                    r["fixed"] = True
                    r["source_reference_only"] = True
                    r["fidelity"]["original_mesh_retained_in_collider"] = False
                    r["source_reference_role"] = (
                        "Original finite-chain source reference geometry; moving chain path supplied by original source law, not rigid-chain dynamics"
                    )
                    law["reference_only_body_ids"].append(group)
            law[
                "scope"
            ] += "; fixed-body source steering bench only; original finite chain and source-reference chain CAD/path retained; not whole-vehicle riding, cargo or bell evidence"
        if kind == "scissor_source_hydraulics":
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.finite_linear_capture import (
                build_finite_linear_capture_report,
            )

            raw_meshes = {}
            ownership = {}
            for name, group in mapping.items():
                record = records[group]
                ownership[name] = [int(x.split("_")[1]) for x in record["source_solid_ids"]]
                for comp in record["source_components"]:
                    mesh = trimesh.load(comp["mesh_file"], force="mesh", process=False)
                    mesh.vertices += np.asarray(record["source_position_m"])
                    raw_meshes[int(comp["solid_id"].split("_")[1])] = mesh
            capture = build_finite_linear_capture_report(
                spec.finite_linear_captures,
                raw_meshes=raw_meshes,
                body_solid_indices=ownership,
                package_binding_sha256=geometry["source_bindings"][0]["source_sha256"],
                mesh_tolerance_m=1e-6,
            )
            capture["binding_scope"] = (
                "Loader original assembly geometry hash; not an original whole-package certification receipt"
            )
            if not capture["passed"]:
                raise ValueError("SOURCE_FINITE_CAPTURE_FAILED " + str(capture["failure_reasons"]))
            law["finite_capture_geometry"] = capture
            law["contact_profile"] = {"envelope_m": 12e-6, "margin_m": 2e-6}
            law[
                "scope"
            ] += "; source finite support, actual STEP capture, closed-valve oil lock/throttle and MINRES settings preserved"
        if kind == "compound_source_pins_drag":
            manifest = {
                "bodies": [
                    {
                        "body_id": name,
                        "components": [
                            {"solid_id": sid} for sid in records[group]["source_solid_ids"]
                        ],
                    }
                    for name, group in mapping.items()
                ]
            }
            contacts = _validated_compliant_surface_contacts(raw, manifest)
            suppressed = {}
            for contact in contacts:
                for ref in contact["moving_component_refs"]:
                    name, sid = ref.split("/", 1)
                    suppressed.setdefault(mapping[name], set()).add(sid)
            # 原旧builder只从native碰撞中移除指定柔顺胶条，完整源网格、质量、惯量仍保留。
            for group, excluded in suppressed.items():
                record = records[group]
                remaining = [
                    c for c in record["source_components"] if c["solid_id"] not in excluded
                ]
                if not remaining:
                    raise ValueError(
                        "A compliant source replacement may not suppress an entire source body"
                    )
                meshes = [
                    trimesh.load(c["mesh_file"], force="mesh", process=False) for c in remaining
                ]
                path = Path(output) / f"{group}_source_rigid_collision.obj"
                export = _write_obj(path, trimesh.util.concatenate(meshes))
                record["collision_mesh_file"] = str(path)
                record["collision_mesh_sha256"] = export["sha256"]
                record["source_collision_suppressed_solids"] = sorted(excluded)
                record["source_collision_replacement"] = (
                    "Original compound_runtime suppresses only declared moving compliant-surface components; full source audit mesh retained"
                )
            law["compliant_surface_contacts"] = list(contacts)
            law[
                "scope"
            ] += "; original exact pin constraints are preserved and cannot prove physical pin retention"
        if (
            kind == "plain_axis_source_spring"
            and spec.contact_fixture is not None
            and spec.contact_fixture.shape == "tabletop_pickup_box"
        ):
            frictions.append(spec.contact_fixture.friction)
        else:
            frictions.append(float(getattr(spec, "friction", 0.2)))
        laws.append(law)
    return laws, frictions


class LegacyLaws:
    def __init__(self, c, system, bodies, records, joints, declarations, axis_quat):
        self.c = c
        self.system = system
        self.bodies = bodies
        self.records = records
        self.joints = joints
        self.declarations = declarations
        self.pins = []
        self.drags = []
        self.detents = []
        self.pen_systems = []
        self.nail_systems = []
        self.axis_systems = []
        self.scissor_systems = []
        self.bicycle_systems = []
        self.loads = []
        self.states = {}
        self.last_forces = []
        self.acc = {key: body.AddAccumulator() for key, body in bodies.items()}
        for declaration in declarations:
            if declaration["kind"] == "compound_source_pins_drag":
                self._compound(declaration, axis_quat)
            elif declaration["kind"] == "detent_source_bushings":
                self._detent(declaration, axis_quat)
            elif declaration["kind"] == "bicycle_source_chain_steering_fixture":
                self._bicycle(declaration, axis_quat)
            elif declaration["kind"] == "scissor_source_hydraulics":
                self._scissor(declaration, axis_quat)
            elif declaration["kind"] in (
                "plain_axis_source_spring",
                "plain_revolute_source_support",
            ):
                self._axis_spring(declaration, axis_quat)
            elif declaration["kind"] == "click_pen_source_bushings":
                self._pen(declaration, axis_quat)
            elif declaration["kind"] == "nail_source_bushings_spring":
                self._nail(declaration, axis_quat)
            else:
                raise ValueError("Unsupported prepared legacy source law " + declaration["kind"])

    def vec(self, v):
        return np.array([float(v.x), float(v.y), float(v.z)])

    def quaternion(self, b):
        q = b.GetRot()
        return (q.e0, q.e1, q.e2, q.e3)

    def point_speed(self, b, point):
        return self.vec(b.GetPosDt()) + np.cross(
            self.vec(b.GetAngVelParent()), self.vec(point) - self.vec(b.GetPos())
        )

    def force(self, key, force, point, reason):
        self.bodies[key].AccumulateForce(self.acc[key], self.c.ChVector3d(*force), point, False)
        self.last_forces.append(
            {
                "body": key,
                "force_n": list(map(float, force)),
                "point_m": self.vec(point).tolist(),
                "source_law": reason,
            }
        )

    def torque(self, key, torque, reason):
        self.bodies[key].AccumulateTorque(self.acc[key], self.c.ChVector3d(*torque), False)
        self.last_forces.append(
            {"body": key, "torque_nm": list(map(float, torque)), "source_law": reason}
        )

    def _compound(self, d, axis_quat):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_spec import (
            compound_linkage_from_dict,
        )

        spec = compound_linkage_from_dict(d["family_spec"])
        mapping = d["body_mapping"]
        c = self.c
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_runtime import (
            _compound_physical_solver,
        )

        self.system.SetSolver(_compound_physical_solver(c))
        self.source_solver = {
            "type": "ChSolverADMM",
            "max_iterations": 500,
            "tolerance": 1e-10,
            "source": "compound_runtime._compound_physical_solver",
        }
        cap = max(spec.nominal_input_torque_nm, spec.return_input_torque_nm, 1e-6)
        for j in spec.revolute_joints:
            pk, ck = mapping[j.parent_body_id], mapping[j.child_body_id]
            parent, child = self.bodies[pk], self.bodies[ck]
            link = c.ChLinkLockRevolute()
            link.SetName(j.joint_id)
            link.Initialize(
                child, parent, c.ChFramed(c.ChVector3d(*j.origin_abs_m), axis_quat(c, j.axis_abs))
            )
            self.system.Add(link)
            native_spec = {
                "id": j.joint_id,
                "type": "revolute",
                "body_a": ck,
                "body_b": pk,
                "anchor": list(j.origin_abs_m),
                "axis": list(j.axis_abs),
                "source_declared_exact": True,
            }
            self.joints[j.joint_id] = {
                "spec": native_spec,
                "link": link,
                "active": True,
                "coordinate": 0.0,
                "rate": 0.0,
                "effort": 0.0,
                "initial_offset": self.vec(child.GetPos()) - self.vec(parent.GetPos()),
            }
            minimum = (
                spec.home_output_angle_rad
                if j.min_angle_rad is None and j.joint_id == spec.output_joint_id
                else j.min_angle_rad
            )
            maximum = (
                spec.outbound_output_angle_rad
                if j.max_angle_rad is None and j.joint_id == spec.output_joint_id
                else j.max_angle_rad
            )
            self.pins.append(
                {
                    "joint": j,
                    "parent": parent,
                    "child": child,
                    "child_key": ck,
                    "cap": cap,
                    "minimum": minimum,
                    "maximum": maximum,
                    "parent_q0": self.quaternion(parent),
                    "child_q0": self.quaternion(child),
                    "parent_axis": parent.TransformDirectionParentToLocal(
                        c.ChVector3d(*j.axis_abs)
                    ),
                    "child_axis": child.TransformDirectionParentToLocal(c.ChVector3d(*j.axis_abs)),
                }
            )
        for row in d["compliant_surface_contacts"]:
            key = mapping[row["moving_body_id"]]
            body = self.bodies[key]
            self.drags.append(
                {
                    "spec": row,
                    "body": body,
                    "key": key,
                    "point_local": body.TransformPointParentToLocal(
                        c.ChVector3d(*row["contact_point_abs_m"])
                    ),
                }
            )

    def _detent(self, d, axis_quat):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
            detent_slider_runtime as original,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_spec import (
            prismatic_detent_latch_from_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
            bounded_support_gate,
            BoundedSupportRuntime,
            TRANSVERSE_ANCHOR_DRIFT,
        )

        c = self.c
        spec = prismatic_detent_latch_from_dict(d["family_spec"])
        mapping = d["body_mapping"]
        # 原detent_slider_runtime物理支路指定ADMM处理接触和bushing矩阵，不能沿用PSOR。
        solver = c.ChSolverADMM()
        solver.SetMaxIterations(500)
        solver.SetTolerance(1.0e-10)
        self.system.SetSolver(solver)
        self.source_solver = {
            "type": "ChSolverADMM",
            "max_iterations": 500,
            "tolerance": 1.0e-10,
            "source": "detent_slider_runtime.py physical branch",
        }
        container = c.ChLoadContainer()
        self.system.Add(container)
        self.loads.append(container)
        entries = []
        for j, k, damper, template in [
            (
                spec.blade_prismatic_joint,
                0.0,
                original.GUIDE_FREE_AXIS_DAMPING_N_S_M,
                "finite_prismatic_guide_capture_v1",
            ),
            (
                spec.latch_press_joint,
                spec.latch_press_joint.spring_stiffness_n_m,
                spec.latch_press_joint.damping_n_s_m,
                "finite_spring_return_press_capture_v1",
            ),
        ]:
            pk, ck = mapping[j.parent_body_id], mapping[j.child_body_id]
            parent, child = self.bodies[pk], self.bodies[ck]
            stiffness = c.ChMatrix66d()
            damping = c.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            for i, v in enumerate(
                (
                    original.GUIDE_TRANSVERSE_STIFFNESS_N_M,
                    original.GUIDE_TRANSVERSE_STIFFNESS_N_M,
                    k,
                    *([original.GUIDE_ROTATIONAL_STIFFNESS_NM_RAD] * 3),
                )
            ):
                stiffness.SetItem(i, i, v)
            for i, v in enumerate(
                (
                    original.GUIDE_TRANSVERSE_DAMPING_N_S_M,
                    original.GUIDE_TRANSVERSE_DAMPING_N_S_M,
                    damper,
                    *([original.GUIDE_ROTATIONAL_DAMPING_NM_S_RAD] * 3),
                )
            ):
                damping.SetItem(i, i, v)
            frame = c.ChFramed(c.ChVector3d(*j.origin_abs_m), axis_quat(c, j.axis_abs))
            bushing = c.ChLoadBodyBodyBushingGeneric(child, parent, frame, stiffness, damping)
            container.Add(bushing)
            zero = c.ChMatrix66d()
            zero.SetZero()
            origin = c.ChVector3d(*j.origin_abs_m)
            gate = bounded_support_gate(
                body_a_id=j.parent_body_id,
                body_b_id=j.child_body_id,
                endpoint_a_component_refs=tuple(
                    j.parent_body_id + "/" + s for s in self.records[pk]["source_solid_ids"]
                ),
                endpoint_b_component_refs=tuple(
                    j.child_body_id + "/" + s for s in self.records[ck]["source_solid_ids"]
                ),
                maximum_drift_m=spec.max_support_drift_m,
                coordinate_mode=TRANSVERSE_ANCHOR_DRIFT,
                maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
                template_id=template,
                proof_source="validator_prismatic_detent_support_envelope_v1",
            )
            entries.append(
                {
                    "joint": j,
                    "parent": parent,
                    "child": child,
                    "parent_key": pk,
                    "child_key": ck,
                    "bushing": bushing,
                    "stiffness": stiffness,
                    "damping": damping,
                    "zero": zero,
                    "parent_point": parent.TransformPointParentToLocal(origin),
                    "child_point": child.TransformPointParentToLocal(origin),
                    "parent_axis": parent.TransformDirectionParentToLocal(
                        c.ChVector3d(*j.axis_abs)
                    ),
                    "child_axis": child.TransformDirectionParentToLocal(c.ChVector3d(*j.axis_abs)),
                    "gate": BoundedSupportRuntime(interface_id="support-" + j.joint_id, gate=gate),
                }
            )
        blade = self.bodies[mapping[spec.output_body_id]]
        latch = self.bodies[mapping[spec.input_body_id]]
        self.detents.append(
            {
                "spec": spec,
                "mapping": mapping,
                "entries": entries,
                "blade0": self.vec(blade.GetPos()),
                "latch0": self.vec(latch.GetPos()),
                "original": original,
            }
        )

    def before_step(self, now, dt):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_runtime import (
            _soft_limit_torque,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
            _signed_axis_angle,
        )

        for key, b in self.bodies.items():
            b.EmptyAccumulator(self.acc[key])
        self.last_forces = []
        for row in self.pins:
            j = row["joint"]
            parent, child = row["parent"], row["child"]
            pa = self.vec(parent.TransformDirectionLocalToParent(row["parent_axis"]))
            ca = self.vec(child.TransformDirectionLocalToParent(row["child_axis"]))
            axis = pa + ca
            axis /= max(np.linalg.norm(axis), 1e-12)
            speed = float(
                (self.vec(child.GetAngVelParent()) - self.vec(parent.GetAngVelParent())) @ axis
            )
            angle = (
                _signed_axis_angle(self.quaternion(child), j.axis_abs)
                - _signed_axis_angle(row["child_q0"], j.axis_abs)
                - _signed_axis_angle(self.quaternion(parent), j.axis_abs)
                + _signed_axis_angle(row["parent_q0"], j.axis_abs)
            )
            limit = _soft_limit_torque(
                joint_angle_rad=angle,
                free_speed_rad_s=speed,
                minimum_rad=row["minimum"],
                maximum_rad=row["maximum"],
                stiffness_n_m_rad=j.limit_stiffness_n_m_rad,
                damping_n_m_s_rad=j.limit_damping_n_m_s_rad,
                approach_zone_rad=j.limit_approach_zone_rad,
                approach_damping_n_m_s_rad=j.limit_approach_damping_n_m_s_rad,
            )
            effort = max(
                -row["cap"], min(row["cap"], limit - j.free_axis_damping_n_m_s_rad * speed)
            )
            # 原更新只给child轴阻尼/限位，parent_damping_torque为None；照源保留。
            self.torque(
                row["child_key"], axis * effort, "compound original bounded axis damping/soft limit"
            )
        for row in self.drags:
            s = row["spec"]
            body = row["body"]
            point = body.TransformPointLocalToParent(row["point_local"])
            velocity = self.point_speed(body, point)
            normal = np.array(s["normal_abs"])
            tangent = velocity - normal * float(velocity @ normal)
            speed = float(np.linalg.norm(tangent))
            magnitude = min(
                s["tangential_damping_n_s_m"] * speed,
                s["friction_coefficient"] * s["normal_load_n"],
            )
            force = -tangent * magnitude / speed if speed > 1e-12 else np.zeros(3)
            self.force(row["key"], force, point, "compound original compliant surface drag")
            self.states[s["contact_id"]] = {"drag_n": magnitude, "speed_m_s": speed}
        for system in self.detents:
            self._update_detent(system, now)
        for system in self.pen_systems:
            self._update_pen(system, now)
        for system in self.nail_systems:
            self._update_nail(system, now)
        for system in self.axis_systems:
            self._update_axis_spring(system, now)
        for system in self.scissor_systems:
            self._update_scissor(system, now)
        for system in self.bicycle_systems:
            self._update_bicycle(system, now)

    def _update_detent(self, system, now):
        spec = system["spec"]
        original = system["original"]
        mapping = system["mapping"]
        c = self.c
        for row in system["entries"]:
            parent, child = row["parent"], row["child"]
            a = parent.TransformPointLocalToParent(row["parent_point"])
            b = child.TransformPointLocalToParent(row["child_point"])
            pa = self.vec(parent.TransformDirectionLocalToParent(row["parent_axis"]))
            ca = self.vec(child.TransformDirectionLocalToParent(row["child_axis"]))
            angle = math.acos(
                max(-1.0, min(1.0, abs(float(pa @ ca) / (np.linalg.norm(pa) * np.linalg.norm(ca)))))
            )
            state = row["gate"].evaluate(
                time_s=now,
                anchor_a_abs_m=tuple(self.vec(a)),
                anchor_b_abs_m=tuple(self.vec(b)),
                velocity_a_abs_m_s=tuple(self.point_speed(parent, a)),
                velocity_b_abs_m_s=tuple(self.point_speed(child, b)),
                requested_effort_si=float(np.linalg.norm(self.vec(row["bushing"].GetForce()))),
                free_axis_abs=tuple(pa),
                axis_misalignment_rad=angle,
            )
            self.states[row["joint"].joint_id] = state
            active = bool(state["transmission_active"])
            row["bushing"].SetStiffnessMatrix(row["stiffness"] if active else row["zero"])
            row["bushing"].SetDampingMatrix(row["damping"] if active else row["zero"])
        blade = self.bodies[mapping[spec.output_body_id]]
        latch = self.bodies[mapping[spec.input_body_id]]
        bj = spec.blade_prismatic_joint
        lj = spec.latch_press_joint
        axis = np.array(bj.axis_abs)
        press_axis = np.array(lj.axis_abs)
        progress = float((self.vec(blade.GetPos()) - system["blade0"]) @ axis)
        speed = float(self.vec(blade.GetPosDt()) @ axis)
        scalar = 0.0
        if progress < bj.minimum_position_m:
            scalar = original.LIMIT_STIFFNESS_N_M * (
                bj.minimum_position_m - progress
            ) - original.LIMIT_DAMPING_N_S_M * min(speed, 0.0)
        elif progress > bj.maximum_position_m:
            scalar = -original.LIMIT_STIFFNESS_N_M * (
                progress - bj.maximum_position_m
            ) - original.LIMIT_DAMPING_N_S_M * max(speed, 0.0)
        brow = system["entries"][0]
        self.force(
            mapping[spec.output_body_id],
            axis * scalar,
            blade.TransformPointLocalToParent(brow["child_point"]),
            "detent original blade travel stop",
        )
        press = float((self.vec(latch.GetPos()) - system["latch0"] - axis * progress) @ press_axis)
        rate = float((self.vec(latch.GetPosDt()) - self.vec(blade.GetPosDt())) @ press_axis)
        scalar = -lj.spring_preload_n if self.states[lj.joint_id]["transmission_active"] else 0.0
        if press < lj.minimum_position_m:
            scalar += original.LIMIT_STIFFNESS_N_M * (
                lj.minimum_position_m - press
            ) - original.LIMIT_DAMPING_N_S_M * min(rate, 0.0)
        elif press > lj.maximum_position_m:
            scalar += -original.LIMIT_STIFFNESS_N_M * (
                press - lj.maximum_position_m
            ) - original.LIMIT_DAMPING_N_S_M * max(rate, 0.0)
        force = press_axis * scalar
        lrow = system["entries"][1]
        self.force(
            mapping[spec.input_body_id],
            force,
            latch.TransformPointLocalToParent(lrow["child_point"]),
            "detent original preload and press stop",
        )
        self.force(
            mapping[spec.output_body_id],
            -force,
            c.ChVector3d(*lj.origin_abs_m),
            "detent original carriage reaction",
        )

    def _pen(self, d, axis_quat):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
            click_pen_runtime as original,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.click_pen_spec import (
            click_pen_cam_latch_from_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
            bounded_support_gate,
            BoundedSupportRuntime,
            TRANSVERSE_ANCHOR_DRIFT,
        )

        c = self.c
        spec = click_pen_cam_latch_from_dict(d["family_spec"])
        mapping = d["body_mapping"]
        solver = c.ChSolverADMM()
        solver.SetMaxIterations(1000)
        solver.SetTolerance(1e-12)
        if hasattr(solver, "SetTolerancePrimal"):
            solver.SetTolerancePrimal(1e-10)
        if hasattr(solver, "SetToleranceDual"):
            solver.SetToleranceDual(1e-10)
        it = solver.AsIterative()
        if hasattr(it, "EnableWarmStart"):
            it.EnableWarmStart(True)
        self.system.SetSolver(solver)
        if hasattr(self.system, "SetMaxPenetrationRecoverySpeed"):
            self.system.SetMaxPenetrationRecoverySpeed(1.0)
        self.source_solver = {
            "type": "ChSolverADMM",
            "max_iterations": 1000,
            "tolerance": 1e-12,
            "source": "click_pen_runtime physical branch",
        }
        container = c.ChLoadContainer()
        self.system.Add(container)
        self.loads.append(container)
        entries = []
        definitions = [
            (spec.button_prismatic_joint, "finite_spring_return_prismatic_v1"),
            (spec.refill_prismatic_joint, "finite_compression_spring_prismatic_v1"),
            (spec.cam_cylindrical_joint, "finite_cylindrical_indexing_guide_v1"),
        ]
        for j, template in definitions:
            cam = j.joint_id == spec.cam_cylindrical_joint.joint_id
            pk, ck = mapping[j.parent_body_id], mapping[j.child_body_id]
            parent, child = self.bodies[pk], self.bodies[ck]
            k = 0.0 if cam else j.spring_stiffness_n_m
            damper = original.FREE_AXIS_DAMPING_N_S_M if cam else j.damping_n_s_m
            kt = 0.0 if cam else original.GUIDE_ROTATIONAL_STIFFNESS_NM_RAD
            ct = (
                original.FREE_TORSION_DAMPING_NM_S_RAD
                if cam
                else original.GUIDE_ROTATIONAL_DAMPING_NM_S_RAD
            )
            stiffness = c.ChMatrix66d()
            damping = c.ChMatrix66d()
            zero = c.ChMatrix66d()
            for matrix in (stiffness, damping, zero):
                matrix.SetZero()
            for i, v in enumerate(
                (
                    original.GUIDE_TRANSVERSE_STIFFNESS_N_M,
                    original.GUIDE_TRANSVERSE_STIFFNESS_N_M,
                    k,
                    original.GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                    original.GUIDE_ROTATIONAL_STIFFNESS_NM_RAD,
                    kt,
                )
            ):
                stiffness.SetItem(i, i, v)
            for i, v in enumerate(
                (
                    original.GUIDE_TRANSVERSE_DAMPING_N_S_M,
                    original.GUIDE_TRANSVERSE_DAMPING_N_S_M,
                    damper,
                    original.GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                    original.GUIDE_ROTATIONAL_DAMPING_NM_S_RAD,
                    ct,
                )
            ):
                damping.SetItem(i, i, v)
            origin = c.ChVector3d(*j.origin_abs_m)
            bushing = c.ChLoadBodyBodyBushingGeneric(
                child, parent, c.ChFramed(origin, axis_quat(c, j.axis_abs)), stiffness, damping
            )
            container.Add(bushing)
            gate = bounded_support_gate(
                body_a_id=j.parent_body_id,
                body_b_id=j.child_body_id,
                endpoint_a_component_refs=tuple(
                    j.parent_body_id + "/" + x for x in self.records[pk]["source_solid_ids"]
                ),
                endpoint_b_component_refs=tuple(
                    j.child_body_id + "/" + x for x in self.records[ck]["source_solid_ids"]
                ),
                maximum_drift_m=spec.max_support_drift_m,
                coordinate_mode=TRANSVERSE_ANCHOR_DRIFT,
                maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
                template_id=template,
                proof_source="validator_click_pen_support_envelope_v1",
            )
            entries.append(
                dict(
                    joint=j,
                    parent=parent,
                    child=child,
                    parent_key=pk,
                    child_key=ck,
                    bushing=bushing,
                    stiffness=stiffness,
                    damping=damping,
                    zero=zero,
                    parent_point=parent.TransformPointParentToLocal(origin),
                    child_point=child.TransformPointParentToLocal(origin),
                    parent_axis=parent.TransformDirectionParentToLocal(c.ChVector3d(*j.axis_abs)),
                    child_axis=child.TransformDirectionParentToLocal(c.ChVector3d(*j.axis_abs)),
                    gate=BoundedSupportRuntime(interface_id="support-" + j.joint_id, gate=gate),
                    initial_position=self.vec(child.GetPos()),
                    cam=cam,
                )
            )
        self.pen_systems.append(
            dict(spec=spec, mapping=mapping, entries=entries, original=original)
        )

    def _update_pen(self, system, now):
        original = system["original"]
        for row in system["entries"]:
            j = row["joint"]
            parent, child = row["parent"], row["child"]
            a = parent.TransformPointLocalToParent(row["parent_point"])
            b = child.TransformPointLocalToParent(row["child_point"])
            pa = self.vec(parent.TransformDirectionLocalToParent(row["parent_axis"]))
            ca = self.vec(child.TransformDirectionLocalToParent(row["child_axis"]))
            angle = math.acos(
                max(-1.0, min(1.0, abs(float(pa @ ca) / (np.linalg.norm(pa) * np.linalg.norm(ca)))))
            )
            state = row["gate"].evaluate(
                time_s=now,
                anchor_a_abs_m=tuple(self.vec(a)),
                anchor_b_abs_m=tuple(self.vec(b)),
                velocity_a_abs_m_s=tuple(self.point_speed(parent, a)),
                velocity_b_abs_m_s=tuple(self.point_speed(child, b)),
                requested_effort_si=float(np.linalg.norm(self.vec(row["bushing"].GetForce()))),
                free_axis_abs=tuple(pa),
                axis_misalignment_rad=angle,
            )
            self.states[j.joint_id] = state
            active = bool(state["transmission_active"])
            row["bushing"].SetStiffnessMatrix(row["stiffness"] if active else row["zero"])
            row["bushing"].SetDampingMatrix(row["damping"] if active else row["zero"])
            # 与原set_limit_forces一致：cam压下坐标是其轴的反向，凸轮转角从不强制。
            axis = np.array(j.axis_abs) * (-1 if row["cam"] else 1)
            position = float((self.vec(child.GetPos()) - row["initial_position"]) @ axis)
            speed = float(self.vec(child.GetPosDt()) @ axis)
            preload = 0.0 if row["cam"] else j.spring_preload_n
            force = -preload if active else 0.0
            if position < j.minimum_position_m:
                force += original.LIMIT_STIFFNESS_N_M * (
                    j.minimum_position_m - position
                ) - original.LIMIT_DAMPING_N_S_M * min(speed, 0.0)
            elif position > j.maximum_position_m:
                force += -original.LIMIT_STIFFNESS_N_M * (
                    position - j.maximum_position_m
                ) - original.LIMIT_DAMPING_N_S_M * max(speed, 0.0)
            self.force(
                row["child_key"], axis * force, b, "click pen original preload/finite guide stop"
            )

    def _nail(self, d, axis_quat):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.nail_clipper_compound_lever import (
            NailClipperCompoundLeverAdapter,
            _typed_flexible_analytical_enabled,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_runtime import (
            compound_bushing_parameters,
            _compound_physical_solver,
        )

        if _typed_flexible_analytical_enabled(d["family_spec"]):
            raise ValueError(
                "Typed no-contact nail variant requires its original analytical coupling; not this dense-contact adapter"
            )
        c = self.c
        spec = NailClipperCompoundLeverAdapter().parse_family_spec(d["family_spec"])
        mapping = d["body_mapping"]
        parameters = compound_bushing_parameters()
        self.system.SetSolver(_compound_physical_solver(c))
        container = c.ChLoadContainer()
        self.system.Add(container)
        self.loads.append(container)
        for j in spec.revolute_joints:
            stiffness = c.ChMatrix66d()
            damping = c.ChMatrix66d()
            stiffness.SetZero()
            damping.SetZero()
            for i, v in enumerate(
                (
                    [parameters["translational_stiffness_n_m"]] * 3
                    + [parameters["axis_stiffness_n_m_rad"]] * 2
                    + [0.0]
                )
            ):
                stiffness.SetItem(i, i, v)
            for i, v in enumerate(
                (
                    [parameters["translational_damping_n_s_m"]] * 3
                    + [parameters["axis_damping_n_m_s_rad"]] * 2
                    + [parameters["free_axis_damping_n_m_s_rad"]]
                )
            ):
                damping.SetItem(i, i, v)
            parent = self.bodies[mapping[j.parent_body_id]]
            child = self.bodies[mapping[j.child_body_id]]
            bushing = c.ChLoadBodyBodyBushingGeneric(
                child,
                parent,
                c.ChFramed(c.ChVector3d(*j.origin_abs_m), axis_quat(c, j.axis_abs)),
                stiffness,
                damping,
            )
            container.Add(bushing)
            self.loads.append((bushing, stiffness, damping))
        spring = spec.restoring_spring
        body = self.bodies[mapping[spring.body_b_id]]
        self.nail_systems.append(
            dict(spec=spec, mapping=mapping, initial_q=self.quaternion(body), parameters=parameters)
        )
        self.source_solver = {
            "type": "ChSolverADMM",
            "max_iterations": 500,
            "tolerance": 1e-10,
            "source": "nail_clipper_compound_lever physical topology",
        }

    def _update_nail(self, system, now):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
            _signed_axis_angle,
        )

        spring = system["spec"].restoring_spring
        key = system["mapping"][spring.body_b_id]
        body = self.bodies[key]
        axis = np.array(spring.axis_abs)
        angle = _signed_axis_angle(self.quaternion(body), spring.axis_abs) - _signed_axis_angle(
            system["initial_q"], spring.axis_abs
        )
        rate = float(self.vec(body.GetAngVelParent()) @ axis)
        torque = (
            -spring.stiffness_n_m_rad * (angle - spring.rest_angle_rad)
            - spring.damping_n_m_s_rad * rate
        )
        self.torque(key, axis * torque, "nail original passive restoring spring")

    def _axis_spring(self, d, axis_quat):
        from types import SimpleNamespace
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
            PlainRevoluteAdapter,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.axis_spring_spec import (
            axis_spring_from_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
            bounded_support_gate,
            BoundedSupportRuntime,
            RELATIVE_ANCHOR_DRIFT,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
            RevoluteKinematicThresholds,
        )

        c = self.c
        spec = (
            axis_spring_from_dict(d["family_spec"], expected_family="revolute")
            if d["kind"] == "plain_axis_source_spring"
            else None
        )
        adapter = PlainRevoluteAdapter()
        source = SimpleNamespace(family_spec=d["family_spec"])
        supports = adapter.physical_supports(None, source) + adapter.constitutive_supports(
            None, source
        )
        threshold = RevoluteKinematicThresholds()
        a = d["body_mapping"]["fixed"]
        b = d["body_mapping"]["moving"]
        parent = self.bodies[a]
        child = self.bodies[b]
        origin = c.ChVector3d(*d["anchor_abs_m"])
        axis = c.ChVector3d(*d["axis_abs"])
        gate = bounded_support_gate(
            body_a_id=a,
            body_b_id=b,
            endpoint_a_component_refs=tuple(
                a + "/" + x for x in self.records[a]["source_solid_ids"]
            ),
            endpoint_b_component_refs=tuple(
                b + "/" + x for x in self.records[b]["source_solid_ids"]
            ),
            maximum_drift_m=threshold.max_pivot_drift_m,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=threshold.max_off_axis_rotation_rad,
            template_id="finite_revolute_pin_capture_v1",
            proof_source="validator_plain_revolute_support_envelope_v1",
        )
        self.axis_systems.append(
            dict(
                spec=spec,
                supports=supports,
                parent=parent,
                child=child,
                parent_key=a,
                child_key=b,
                parent_point=parent.TransformPointParentToLocal(origin),
                child_point=child.TransformPointParentToLocal(origin),
                parent_axis=parent.TransformDirectionParentToLocal(axis),
                child_axis=child.TransformDirectionParentToLocal(axis),
                q0=parent.GetRot().GetConjugate() * child.GetRot(),
                gate=BoundedSupportRuntime(interface_id="axis_spring_source_capture", gate=gate),
            )
        )

    def _update_axis_spring(self, row, now):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
            _finite_compliant_support_wrench,
            _axis_spring_limit_response,
        )

        c = self.c
        parent, child = row["parent"], row["child"]
        a = parent.TransformPointLocalToParent(row["parent_point"])
        b = child.TransformPointLocalToParent(row["child_point"])
        axis = self.vec(parent.TransformDirectionLocalToParent(row["parent_axis"]))
        axis /= np.linalg.norm(axis)
        child_axis = self.vec(child.TransformDirectionLocalToParent(row["child_axis"]))
        child_axis /= np.linalg.norm(child_axis)
        alignment = math.acos(max(-1.0, min(1.0, abs(float(axis @ child_axis)))))
        state = row["gate"].evaluate(
            time_s=now,
            anchor_a_abs_m=tuple(self.vec(a)),
            anchor_b_abs_m=tuple(self.vec(b)),
            velocity_a_abs_m_s=tuple(self.point_speed(parent, a)),
            velocity_b_abs_m_s=tuple(self.point_speed(child, b)),
            requested_effort_si=row.get("last_force_n", 0.0),
            axis_misalignment_rad=alignment,
        )
        self.states["axis_spring_source_capture"] = state
        if not state["transmission_active"]:
            row["last_force_n"] = 0.0
            return
        q = (parent.GetRot().GetConjugate() * child.GetRot()) * row["q0"].GetConjugate()
        v = np.array([q.e1, q.e2, q.e3])
        length = float(np.linalg.norm(v))
        rv = v * (2 * math.atan2(length, q.e0) / length) if length > 1e-15 else np.zeros(3)
        rv_world = self.vec(parent.GetRot().Rotate(c.ChVector3d(*rv)))
        omega = self.vec(child.GetAngVelParent()) - self.vec(parent.GetAngVelParent())
        force, torque = _finite_compliant_support_wrench(
            row["supports"],
            axis_abs=tuple(axis),
            displacement_abs_m=tuple(self.vec(b) - self.vec(a)),
            relative_velocity_abs_m_s=tuple(
                self.point_speed(child, b) - self.point_speed(parent, a)
            ),
            rotation_vector_abs_rad=tuple(rv_world),
            angular_velocity_abs_rad_s=tuple(omega),
        )
        angle = float(rv_world @ axis)
        rate = float(omega @ axis)
        limit, penetration, side = (
            _axis_spring_limit_response(angle, rate, row["spec"].maximum_deflection_si)
            if row["spec"] is not None
            else (0.0, 0.0, None)
        )
        torque = np.array(torque) + axis * limit
        force = np.array(force)
        # 当整把夹钳被抬起时，下臂是运动参考；内部弹性在两端成对作用，而非暗自固定到世界。
        self.force(row["child_key"], force, b, "source plain bearing and axis spring")
        self.force(row["parent_key"], -force, a, "source bearing reaction")
        self.torque(row["child_key"], torque, "source axis spring and finite angle stop")
        self.torque(row["parent_key"], -torque, "source axis spring reaction")
        row["last_force_n"] = float(np.linalg.norm(force))
        self.states["axis_spring_constitutive"] = {
            "angle_rad": angle,
            "rate_rad_s": rate,
            "stop_penetration_rad": penetration,
            "stop_side": side,
            "torque_nm": torque.tolist(),
        }

    def _scissor(self, d, axis_quat):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
            scissor_lift_runtime as original,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_spec import (
            scissor_lift_from_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
            bounded_support_gate,
            BoundedSupportRuntime,
            RELATIVE_ANCHOR_DRIFT,
            TRANSVERSE_ANCHOR_DRIFT,
        )

        c = self.c
        spec = scissor_lift_from_dict(d["family_spec"])
        mapping = d["body_mapping"]
        solver = c.ChSolverMINRES()
        solver.SetMaxIterations(250)
        solver.SetTolerance(1e-10)
        self.system.SetSolver(solver)
        self.source_solver = {
            "type": "ChSolverMINRES",
            "max_iterations": 250,
            "tolerance": 1e-10,
            "source": "scissor_lift_runtime original physical branch",
        }
        container = c.ChLoadContainer()
        self.system.Add(container)
        self.loads.append(container)
        entries = []
        hinge = spec.revolute_joints[0].axis_abs
        for kind, j in (
            [("revolute", j) for j in spec.revolute_joints]
            + [("prismatic", j) for j in spec.prismatic_joints]
            + [("cylindrical", spec.cylindrical_joint)]
        ):
            pk, ck = mapping[j.parent_body_id], mapping[j.child_body_id]
            parent, child = self.bodies[pk], self.bodies[ck]
            stiffness = c.ChMatrix66d()
            damping = c.ChMatrix66d()
            zero = c.ChMatrix66d()
            for matrix in (stiffness, damping, zero):
                matrix.SetZero()
            k, c1, kr, cr = (
                original.BUSHING_STIFFNESS_N_M,
                original.BUSHING_DAMPING_N_S_M,
                original.AXIS_STIFFNESS_N_M_RAD,
                original.AXIS_DAMPING_N_M_S_RAD,
            )
            if kind == "revolute":
                ks = (k, k, k, kr, kr, 0.0)
                ds = (c1, c1, c1, cr, cr, original.FREE_AXIS_DAMPING_N_M_S_RAD)
            elif kind == "prismatic":
                ks = (0.0, k, k, kr, kr, 0.0)
                ds = (
                    original.FREE_TRANSLATION_DAMPING_N_S_M,
                    c1,
                    c1,
                    cr,
                    cr,
                    original.FREE_AXIS_DAMPING_N_M_S_RAD,
                )
            else:
                ks = (k, k, 0.0, kr, kr, 0.0)
                ds = (
                    c1,
                    c1,
                    original.FREE_TRANSLATION_DAMPING_N_S_M,
                    cr,
                    cr,
                    original.FREE_AXIS_DAMPING_N_M_S_RAD,
                )
            for i, v in enumerate(ks):
                stiffness.SetItem(i, i, v)
            for i, v in enumerate(ds):
                damping.SetItem(i, i, v)
            q = (
                original._quat_slide_x_hinge_z(c, j.axis_abs, hinge)
                if kind == "prismatic"
                else axis_quat(c, j.axis_abs)
            )
            origin = c.ChVector3d(*j.origin_abs_m)
            bushing = c.ChLoadBodyBodyBushingGeneric(
                child, parent, c.ChFramed(origin, q), stiffness, damping
            )
            container.Add(bushing)
            align = hinge if kind == "prismatic" else j.axis_abs
            gate = bounded_support_gate(
                body_a_id=j.parent_body_id,
                body_b_id=j.child_body_id,
                endpoint_a_component_refs=tuple(
                    j.parent_body_id + "/" + x for x in self.records[pk]["source_solid_ids"]
                ),
                endpoint_b_component_refs=tuple(
                    j.child_body_id + "/" + x for x in self.records[ck]["source_solid_ids"]
                ),
                maximum_drift_m=spec.max_joint_drift_m,
                coordinate_mode=(
                    RELATIVE_ANCHOR_DRIFT if kind == "revolute" else TRANSVERSE_ANCHOR_DRIFT
                ),
                maximum_axis_misalignment_rad=spec.max_off_plane_rotation_rad,
                template_id=(
                    "finite_revolute_pin_capture_v1"
                    if kind == "revolute"
                    else (
                        "finite_roller_track_retention_v1"
                        if kind == "prismatic"
                        else "finite_cylindrical_lateral_capture_v1"
                    )
                ),
                proof_source="validator_scissor_support_envelope_v1",
            )
            entries.append(
                dict(
                    joint=j,
                    kind=kind,
                    parent=parent,
                    child=child,
                    parent_key=pk,
                    child_key=ck,
                    bushing=bushing,
                    stiffness=stiffness,
                    damping=damping,
                    zero=zero,
                    parent_point=parent.TransformPointParentToLocal(origin),
                    child_point=child.TransformPointParentToLocal(origin),
                    parent_axis=parent.TransformDirectionParentToLocal(c.ChVector3d(*j.axis_abs)),
                    child_axis=child.TransformDirectionParentToLocal(c.ChVector3d(*j.axis_abs)),
                    parent_alignment=parent.TransformDirectionParentToLocal(c.ChVector3d(*align)),
                    child_alignment=child.TransformDirectionParentToLocal(c.ChVector3d(*align)),
                    parent_q0=self.quaternion(parent),
                    child_q0=self.quaternion(child),
                    gate=BoundedSupportRuntime(interface_id="support-" + j.joint_id, gate=gate),
                )
            )
        a = spec.actuator
        barrel = self.bodies[mapping[a.barrel_body_id]]
        rod = self.bodies[mapping[a.rod_body_id]]
        capture = d["finite_capture_geometry"]["captures"][0]
        row = dict(
            spec=spec,
            mapping=mapping,
            entries=entries,
            original=original,
            barrel=barrel,
            rod=rod,
            barrel_key=mapping[a.barrel_body_id],
            rod_key=mapping[a.rod_body_id],
            barrel_point=barrel.TransformPointParentToLocal(c.ChVector3d(*a.barrel_anchor_abs_m)),
            rod_point=rod.TransformPointParentToLocal(c.ChVector3d(*a.rod_anchor_abs_m)),
            capture=capture,
            port=None,
            previous_mode=None,
            hold_reference=None,
            last_length=None,
        )
        row["last_length"] = self._cylinder(row)[3]
        self.scissor_systems.append(row)
        self._update_scissor(row, 0.0)

    def _cylinder(self, row):
        a = row["barrel"].TransformPointLocalToParent(row["barrel_point"])
        b = row["rod"].TransformPointLocalToParent(row["rod_point"])
        delta = self.vec(b) - self.vec(a)
        length = max(float(np.linalg.norm(delta)), 1e-12)
        return delta / length, a, b, length

    def _update_scissor(self, system, now):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
            _signed_axis_angle,
        )

        original = system["original"]
        axis, a, b, length = self._cylinder(system)
        capture = system["capture"]
        overlap = capture["parent_capture_length_m"] + capture["child_capture_length_m"] - length
        self.states["hydraulic_capture"] = {
            "length_m": length,
            "overlap_m": overlap,
            "minimum_overlap_m": system["spec"].finite_linear_captures[0].minimum_overlap_m,
        }
        for row in system["entries"]:
            j = row["joint"]
            parent, child = row["parent"], row["child"]
            pa = parent.TransformPointLocalToParent(row["parent_point"])
            pb = child.TransformPointLocalToParent(row["child_point"])
            va = self.vec(parent.TransformDirectionLocalToParent(row["parent_axis"]))
            vb = self.vec(child.TransformDirectionLocalToParent(row["child_axis"]))
            mean = va + vb
            mean /= max(float(np.linalg.norm(mean)), 1e-12)
            aa = self.vec(parent.TransformDirectionLocalToParent(row["parent_alignment"]))
            bb = self.vec(child.TransformDirectionLocalToParent(row["child_alignment"]))
            angle = math.acos(
                max(-1.0, min(1.0, abs(float(aa @ bb) / (np.linalg.norm(aa) * np.linalg.norm(bb)))))
            )
            free = (
                None
                if row["kind"] == "revolute"
                else tuple(va if row["kind"] == "prismatic" else mean)
            )
            captured = (
                row["kind"] != "cylindrical"
                or overlap >= system["spec"].finite_linear_captures[0].minimum_overlap_m
            )
            state = row["gate"].evaluate(
                time_s=now,
                anchor_a_abs_m=tuple(self.vec(pa)),
                anchor_b_abs_m=tuple(self.vec(pb)),
                velocity_a_abs_m_s=tuple(self.point_speed(parent, pa)),
                velocity_b_abs_m_s=tuple(self.point_speed(child, pb)),
                requested_effort_si=float(np.linalg.norm(self.vec(row["bushing"].GetForce()))),
                free_axis_abs=free,
                axis_misalignment_rad=angle,
                additional_constraints_satisfied=captured,
                additional_metrics=(
                    {"finite_capture_overlap_m": overlap} if row["kind"] == "cylindrical" else {}
                ),
                additional_blockers=(
                    ("finite_capture_overlap_lost",) if row["kind"] == "cylindrical" else ()
                ),
            )
            self.states[j.joint_id] = state
            active = bool(state["transmission_active"])
            row["bushing"].SetStiffnessMatrix(row["stiffness"] if active else row["zero"])
            row["bushing"].SetDampingMatrix(row["damping"] if active else row["zero"])
            q = (
                _signed_axis_angle(self.quaternion(child), j.axis_abs)
                - _signed_axis_angle(row["child_q0"], j.axis_abs)
                - _signed_axis_angle(self.quaternion(parent), j.axis_abs)
                + _signed_axis_angle(row["parent_q0"], j.axis_abs)
            )
            rate = float(
                (self.vec(child.GetAngVelParent()) - self.vec(parent.GetAngVelParent())) @ mean
            )
            lo = getattr(j, "min_angle_rad", None)
            hi = getattr(j, "max_angle_rad", None)
            torque = 0.0
            if lo is not None and q < lo:
                torque = -original.LIMIT_STIFFNESS_N_M_RAD * (
                    q - lo
                ) - original.LIMIT_DAMPING_N_M_S_RAD * min(rate, 0.0)
            elif hi is not None and q > hi:
                torque = -original.LIMIT_STIFFNESS_N_M_RAD * (
                    q - hi
                ) - original.LIMIT_DAMPING_N_M_S_RAD * max(rate, 0.0)
            if active:
                self.torque(row["child_key"], mean * torque, "scissor original finite angular stop")

    def cylinder_port(self, index=0):
        row = self.scissor_systems[index]
        if row["port"] is None:
            row["port"] = CylinderForcePort(self, row)
        return row["port"]

    def commit_inputs(self, now, dt, model):
        # 原液压内力依赖当前实际驱动是否撤去；在公共输入齐备后计算，再统一Update。
        for row in self.scissor_systems:
            original = row["original"]
            axis, a, b, length = self._cylinder(row)
            port = row["port"]
            effort = port._last_input if port is not None else 0.0
            mode = "extend" if effort > 0 else "retract" if effort < 0 else "closed"
            if port is None and any(
                getattr(x, "obj", None) in (row["rod"], row["barrel"])
                for x in model.actuators.values()
            ):
                raise ValueError("Use source cylinder_port for the declared hydraulic force pair")
            if mode == "closed" and row["previous_mode"] != "closed":
                row["hold_reference"] = length
            rate = (length - row["last_length"]) / dt
            row["last_length"] = length
            if mode == "closed":
                scalar = max(
                    -original.HYDRAULIC_LOCK_RELIEF_N,
                    min(
                        original.HYDRAULIC_LOCK_RELIEF_N,
                        -original.HYDRAULIC_LOCK_STIFFNESS_N_M * (length - row["hold_reference"])
                        - original.HYDRAULIC_LOCK_DAMPING_N_S_M * rate,
                    ),
                )
            else:
                scalar = (
                    -(
                        original.CYLINDER_THROTTLE_EXTEND_DAMPING_N_S_M
                        if mode == "extend"
                        else original.CYLINDER_THROTTLE_RETRACT_DAMPING_N_S_M
                    )
                    * rate
                )
            limit = 0.0
            a_spec = row["spec"].actuator
            if length < a_spec.min_length_m:
                limit = original.CYLINDER_STOP_STIFFNESS_N_M * (
                    a_spec.min_length_m - length
                ) - original.CYLINDER_STOP_DAMPING_N_S_M * min(rate, 0.0)
            elif length > a_spec.max_length_m:
                limit = -original.CYLINDER_STOP_STIFFNESS_N_M * (
                    length - a_spec.max_length_m
                ) - original.CYLINDER_STOP_DAMPING_N_S_M * max(rate, 0.0)
            capture = row["capture"]
            overlap = (
                capture["parent_capture_length_m"] + capture["child_capture_length_m"] - length
            )
            if overlap < row["spec"].finite_linear_captures[0].minimum_overlap_m:
                scalar = 0.0
                limit = 0.0
            self.force(
                row["rod_key"],
                axis * (scalar + limit),
                b,
                "source hydraulic oil lock/throttle and cylinder stop",
            )
            self.force(
                row["barrel_key"],
                -axis * (scalar + limit),
                a,
                "source hydraulic constitutive reaction",
            )
            row["previous_mode"] = mode
            self.states["hydraulic_oil"] = {
                "mode": mode,
                "hold_reference_m": row["hold_reference"],
                "rate_m_s": rate,
                "constitutive_force_n": scalar,
                "stop_force_n": limit,
            }

    def _bicycle(self, d, axis_quat):
        from types import SimpleNamespace
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
            bicycle_chain_from_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
            PlainRevoluteAdapter,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
            bounded_support_gate,
            BoundedSupportRuntime,
            RELATIVE_ANCHOR_DRIFT,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
            RevoluteKinematicThresholds,
        )

        c = self.c
        spec = bicycle_chain_from_dict(d["family_spec"])
        mapping = d["body_mapping"]
        steer = d["steering"]
        parent = self.bodies[steer["fixed"]]
        child = self.bodies[steer["moving"]]
        if not parent.IsFixed() or not self.bodies[mapping[spec.fixed_body_id]].IsFixed():
            raise ValueError(
                "Source steering-at-rest and chain bench require their declared fixed reference bodies"
            )
        # 原steering胶囊是有限plain bearing；不加入ground_roll用于直行的转向锁死夹具。
        supports = PlainRevoluteAdapter().physical_supports(None, SimpleNamespace(family_spec=None))
        threshold = RevoluteKinematicThresholds()
        origin = c.ChVector3d(*steer["anchor_abs_m"])
        axis = c.ChVector3d(*steer["axis_abs"])
        gate = bounded_support_gate(
            body_a_id=steer["fixed"],
            body_b_id=steer["moving"],
            endpoint_a_component_refs=tuple(
                steer["fixed"] + "/" + x for x in self.records[steer["fixed"]]["source_solid_ids"]
            ),
            endpoint_b_component_refs=tuple(
                steer["moving"] + "/" + x for x in self.records[steer["moving"]]["source_solid_ids"]
            ),
            maximum_drift_m=threshold.max_pivot_drift_m,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=threshold.max_off_axis_rotation_rad,
            template_id="finite_revolute_pin_capture_v1",
            proof_source="validator_plain_revolute_support_envelope_v1",
        )
        self.axis_systems.append(
            dict(
                spec=None,
                supports=supports,
                parent=parent,
                child=child,
                parent_key=steer["fixed"],
                child_key=steer["moving"],
                parent_point=parent.TransformPointParentToLocal(origin),
                child_point=child.TransformPointParentToLocal(origin),
                parent_axis=parent.TransformDirectionParentToLocal(axis),
                child_axis=child.TransformDirectionParentToLocal(axis),
                q0=parent.GetRot().GetConjugate() * child.GetRot(),
                gate=BoundedSupportRuntime(interface_id="source_steering_capture", gate=gate),
            )
        )
        front = d["front_wheel"]
        a = self.bodies[front["body"]]
        b = self.bodies[front["parent"]]
        link = c.ChLinkLockRevolute()
        link.SetName("source_front_wheel_bearing")
        link.Initialize(
            a, b, c.ChFramed(c.ChVector3d(*front["anchor"]), axis_quat(c, front["axis"]))
        )
        self.system.Add(link)
        self.joints["source_front_wheel_bearing"] = {
            "spec": {
                "id": "source_front_wheel_bearing",
                "type": "revolute",
                "body_a": front["body"],
                "body_b": front["parent"],
                "anchor": front["anchor"],
                "axis": front["axis"],
                "source_declared_exact": True,
            },
            "link": link,
            "active": True,
            "coordinate": 0.0,
            "rate": 0.0,
            "effort": 0.0,
            "initial_offset": self.vec(a.GetPos()) - self.vec(b.GetPos()),
        }
        entries = []
        for name, bearing, anchor in [
            (spec.input_body_id, spec.input_bearing, spec.input_origin_abs_m),
            (spec.output_body_id, spec.output_bearing, spec.output_origin_abs_m),
        ]:
            key = mapping[name]
            body = self.bodies[key]
            gate = bounded_support_gate(
                body_a_id=spec.fixed_body_id,
                body_b_id=name,
                endpoint_a_component_refs=tuple(
                    spec.fixed_body_id + "/" + x
                    for x in self.records[mapping[spec.fixed_body_id]]["source_solid_ids"]
                ),
                endpoint_b_component_refs=tuple(
                    name + "/" + x for x in self.records[key]["source_solid_ids"]
                ),
                maximum_drift_m=spec.max_support_drift_m,
                coordinate_mode=RELATIVE_ANCHOR_DRIFT,
                maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
                template_id=(
                    "bicycle_input_finite_bearing_v2"
                    if name == spec.input_body_id
                    else "bicycle_output_finite_bearing_v2"
                ),
                proof_source="approved_bicycle_chain_sprocket_drive_v2",
            )
            sprocket = spec.input_sprocket if name == spec.input_body_id else spec.output_sprocket
            entries.append(
                dict(
                    name=name,
                    key=key,
                    body=body,
                    bearing=bearing,
                    anchor=anchor,
                    point_local=body.TransformPointParentToLocal(c.ChVector3d(*anchor)),
                    axis_local=body.TransformDirectionParentToLocal(c.ChVector3d(*spec.axis_abs)),
                    pitch_local=body.TransformPointParentToLocal(
                        c.ChVector3d(*sprocket.pitch_loop_center_abs_m)
                    ),
                    q0=self.quaternion(body),
                    gate=BoundedSupportRuntime(
                        interface_id="support-" + bearing.support_id, gate=gate
                    ),
                )
            )
        row = dict(
            spec=spec,
            mapping=mapping,
            entries=entries,
            evidence=d["geometry_evidence"],
            validated_role_binding_sha256=d["validated_role_binding_sha256"],
            moving_chain_contract=d["moving_chain_contract"],
        )
        self.bicycle_systems.append(row)
        self._update_bicycle(row, 0.0)

    def _update_bicycle(self, row, now):
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_runtime import (
            _bearing_response,
            _bearing_primary_torque_cap,
            _bearing_total_torque_cap,
            _clamp_vector,
            _clamp_scalar,
            _relative_quaternion,
            _axis_angle_components,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
            chain_engagement_active,
            chain_reaction_torques,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_path import (
            build_moving_chain_path_state,
        )

        spec = row["spec"]
        axis = np.array(spec.axis_abs)
        observed = []
        for entry in row["entries"]:
            b = entry["body"]
            point = b.TransformPointLocalToParent(entry["point_local"])
            actual_axis = self.vec(b.TransformDirectionLocalToParent(entry["axis_local"]))
            velocity = self.point_speed(b, point)
            omega = self.vec(b.GetAngVelParent())
            angle, _ = _axis_angle_components(
                _relative_quaternion(self.quaternion(b), entry["q0"]), spec.axis_abs
            )
            rate = float(omega @ axis)
            response = _bearing_response(
                entry["bearing"],
                anchor_abs=tuple(self.vec(point)),
                home_anchor_abs=entry["anchor"],
                anchor_velocity_abs=tuple(velocity),
                current_axis_abs=tuple(actual_axis),
                required_axis_abs=spec.axis_abs,
                angular_velocity_abs=tuple(omega),
            )
            bearing = entry["bearing"]
            primary = (
                -bearing.primary_axis_centering_stiffness_nm_rad
                * (angle - bearing.primary_axis_home_rad)
                - bearing.primary_axis_damping_nm_s_rad * rate
            )
            cap = _bearing_primary_torque_cap(bearing)
            if cap is not None:
                primary = _clamp_scalar(primary, cap)
            total = np.array(
                _clamp_vector(
                    np.array(response["alignment_torque_abs_nm"]) + axis * primary,
                    _bearing_total_torque_cap(bearing),
                )
            )
            state = entry["gate"].evaluate(
                time_s=now,
                anchor_a_abs_m=entry["anchor"],
                anchor_b_abs_m=tuple(self.vec(point)),
                velocity_a_abs_m_s=(0.0, 0.0, 0.0),
                velocity_b_abs_m_s=tuple(velocity),
                requested_effort_si=float(np.linalg.norm(response["force_abs_n"])),
                axis_misalignment_rad=response["axis_error_rad"],
            )
            self.states[entry["bearing"].support_id] = state
            if state["transmission_active"]:
                self.force(
                    entry["key"],
                    np.array(response["force_abs_n"]),
                    point,
                    "original bicycle finite bearing",
                )
                self.torque(entry["key"], total, "original bicycle primary/axis bearing torque")
            observed.append(
                dict(
                    angle=angle,
                    rate=rate,
                    axis=actual_axis,
                    pitch=self.vec(b.TransformPointLocalToParent(entry["pitch_local"])),
                )
            )
        a, b = observed
        axial = abs(float((a["pitch"] - b["pitch"]) @ axis))
        alignment = math.acos(max(-1.0, min(1.0, abs(float(a["axis"] @ b["axis"])))))
        evidence = row["evidence"]
        active = chain_engagement_active(
            spec,
            bindings_match=(
                evidence.get("passed") is True
                and evidence.get("contract_role_binding_sha256")
                == row["validated_role_binding_sha256"]
            ),
            input_pitch_radius_m=float(evidence["input_pitch_surface"]["major_radius_m"]),
            output_pitch_radius_m=float(evidence["output_pitch_surface"]["major_radius_m"]),
            axis_angle_error_rad=alignment,
            axial_alignment_error_m=axial,
        )
        error = spec.input_pitch_radius_m * a["angle"] - spec.output_pitch_radius_m * b["angle"]
        rate = spec.input_pitch_radius_m * a["rate"] - spec.output_pitch_radius_m * b["rate"]
        tension, ti, to = chain_reaction_torques(
            spec, coordinate_error_m=error, coordinate_rate_m_s=rate, gate_active=active
        )
        self.torque(
            row["mapping"][spec.input_body_id],
            axis * ti,
            "original geometry-gated finite chain input reaction",
        )
        self.torque(
            row["mapping"][spec.output_body_id],
            axis * to,
            "original geometry-gated finite chain output reaction",
        )
        self.states["bicycle_chain"] = {
            "active": active,
            "tension_n": tension,
            "coordinate_error_m": error,
            "rate_m_s": rate,
            "axis_error_rad": alignment,
            "axial_alignment_m": axial,
            "moving_chain_state": build_moving_chain_path_state(
                row["moving_chain_contract"],
                input_angle_rad=a["angle"],
                output_angle_rad=b["angle"],
                coordinate_error_m=error,
            ),
        }

    def after_step(self, now, dt):
        # 内力系数不随结果调整；状态供绑定/记录读取。
        pass


class CylinderForcePort:
    """原液压缸实际两锚点力对；仅此人工输入由公共Action控制。"""

    kind = "source_cylinder_force_pair"

    def __init__(self, law, row):
        self.law = law
        self.row = row
        self._last_input = 0.0
        self.rod_acc = row["rod"].AddAccumulator()
        self.barrel_acc = row["barrel"].AddAccumulator()

    def coordinate(self):
        return self.law._cylinder(self.row)[3]

    def speed(self):
        axis, a, b, _ = self.law._cylinder(self.row)
        return float(
            (self.law.point_speed(self.row["rod"], b) - self.law.point_speed(self.row["barrel"], a))
            @ axis
        )

    def clear_input(self):
        self.row["rod"].EmptyAccumulator(self.rod_acc)
        self.row["barrel"].EmptyAccumulator(self.barrel_acc)
        self._last_input = 0.0

    def add_input(self, effort):
        if not math.isfinite(effort):
            raise ValueError("Nonfinite cylinder input")
        axis, a, b, length = self.law._cylinder(self.row)
        capture = self.row["capture"]
        if (
            capture["parent_capture_length_m"] + capture["child_capture_length_m"] - length
            < self.row["spec"].finite_linear_captures[0].minimum_overlap_m
        ):
            raise ValueError("SOURCE_CYLINDER_CAPTURE_LOST: no hydraulic input can transmit")
        c = self.law.c
        self.row["rod"].AccumulateForce(self.rod_acc, c.ChVector3d(*(axis * effort)), b, False)
        self.row["barrel"].AccumulateForce(
            self.barrel_acc, c.ChVector3d(*(-axis * effort)), a, False
        )
        self._last_input += effort


def build_plain_revolute_component(source_dir, step_s):
    """忠实复用旧plain运行器的两体组件；明确不补未选中零件或未知材料。

    source_dir保存原plan/load/完整assembly与contract副本和原相对路径网格。
    原质量、包围盒惯量、有限支持和原碰撞范围都保持；不代表完整CAD通过。
    """
    import json, hashlib
    from types import SimpleNamespace
    from benchmark_templates.runtime import NativeModel
    from simulation_ready_cad_workflow.validation_pipeline.mechanism import worker as original
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import MeshAssetSpec
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.pychrono_core import (
        load_pychrono_core,
    )

    root = Path(source_dir).resolve()
    plan = json.loads((root / "plan.json").read_text(encoding="utf-8"))
    load = json.loads((root / "load.json").read_text(encoding="utf-8"))
    for name in ("assembly.step", "assembly_contract.json"):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != plan["artifact_hashes"][name]:
            raise ValueError("Original component source artifact changed: " + name)
    if (
        plan["adapter_id"] != "derived_plain_revolute_v1"
        or len(plan["fixed_group_ids"]) != 1
        or len(plan["moving_group_ids"]) != 1
    ):
        raise ValueError("This component adapter requires the original single fixed/moving pair")
    geometry = plan["adapter_spec"]["geometry"]
    c = load_pychrono_core()
    system = original._make_nsc_system(
        c,
        collision_envelope_m=geometry["collision_envelope_m"],
        collision_margin_m=geometry["collision_margin_m"],
    )
    system.SetGravitationalAcceleration(c.ChVector3d(*plan["gravity_abs_m_s2"]))
    material = original._make_material(c, geometry["friction"])
    bodies = {}
    records = {}
    output = root / "component_runtime_assets"
    output.mkdir(exist_ok=True)
    omitted = geometry.get("analytical_transmission_no_contact_colliders", [])
    for role, key, mass, fixed in [
        ("fixed", "fixed_boundary", geometry["fixed_mass_kg"], True),
        ("moving", "moving_group", load["moving_mass_kg"], False),
    ]:
        origin = geometry[role + "_origin_abs_m"]
        combined = MeshAssetSpec(**geometry[role + "_mesh"])
        original._write_body_local_mesh(
            combined,
            workspace=root,
            destination=output / (role + "_full_original.local.obj"),
            body_origin_abs_m=origin,
        )
        components = geometry[role + "_collision_meshes"]
        assets = (
            []
            if role == "moving" and omitted and not components
            else components or [geometry[role + "_mesh"]]
        )
        local = []
        for index, raw in enumerate(assets):
            asset = MeshAssetSpec(**raw)
            target = output / (role + "_" + str(index) + ".obj")
            original._write_body_local_mesh(
                asset, workspace=root, destination=target, body_origin_abs_m=origin
            )
            local.append((target, asset.convex))
        extents = (
            tuple(max(x, 0.01) for x in plan["moving_extents_m"])
            if fixed
            else plan["moving_extents_m"]
        )
        inertia = original._inertia_box(mass, extents)
        body = original._make_body(
            c,
            name=key,
            local_meshes=local,
            origin_abs_m=origin,
            fixed=fixed,
            mass_kg=mass,
            inertia_xx=inertia,
            material=material,
            collision_radius_m=geometry["collision_radius_m"],
            physical_contact_required=True,
        )
        system.Add(body)
        bodies[key] = body
        contract = json.loads((root / "assembly_contract.json").read_text(encoding="utf-8"))
        groups = plan["fixed_group_ids"] if fixed else plan["moving_group_ids"]
        solids = [
            sid
            for part in contract["parts"]
            if part["rigid_group_id"] in groups
            for sid in part["source_solid_ids"]
        ]
        records[key] = {
            "id": key,
            "source_group_ids": groups,
            "source_solid_ids": solids,
            "position_m": origin,
            "mass_kg": mass,
            "inertia_xx_kg_m2": inertia,
            "mass_authority": "original load/geometry declaration; box inertia from trusted worker, not density-derived",
        }
    declaration = {
        "kind": "plain_revolute_source_support",
        "family_spec": plan["adapter_spec"].get("family_spec") or {},
        "body_mapping": {"fixed": "fixed_boundary", "moving": "moving_group"},
        "axis_abs": plan["axis_abs"],
        "anchor_abs_m": plan["joint_origin_abs_m"],
    }
    law = LegacyLaws(c, system, bodies, records, {}, [declaration], original._quat_from_z_to_axis)
    api = {}

    def commit(now, dt):
        try:
            system.Update()
            api["call"] = "Update()"
        except TypeError as exc:
            if "ChSystem_Update" not in str(exc):
                raise
            system.Update(True)
            api["call"] = "Update(True)"

    used = set(plan["fixed_group_ids"] + plan["moving_group_ids"])
    all_groups = {x["rigid_group_id"] for x in contract["rigid_groups"]}
    model = NativeModel(
        system,
        {},
        {},
        before_step=law.before_step,
        after_step=law.after_step,
        commit_inputs=commit,
        keepalive=(law, material, bodies),
        scope="Historical plain-revolute TWO-BODY COMPONENT only; original declared masses and box inertia; finite compliant support; original typed no-contact moving geometry retained as audit mesh; omitted source groups "
        + str(sorted(all_groups - used))
        + "; NOT full BM51 retention/collision/function evidence",
        initial_state="Original assembled two-body component state, original gravity; no pose replay",
    )
    model.objects = {
        "bodies": bodies,
        "body_records": records,
        "legacy_source_laws": law,
        "source_plan": plan,
        "omitted_source_group_ids": sorted(all_groups - used),
        "source_collision_scope": geometry.get("omitted_contact_geometry", []),
        "system_update_api": api,
    }
    law.before_step(0.0, step_s)
    commit(0.0, step_s)
    return model
