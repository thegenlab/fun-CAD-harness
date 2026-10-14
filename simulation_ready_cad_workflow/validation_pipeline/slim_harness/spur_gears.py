"""Source-bound external spur gears with native finite-compliance transmission.

The original rigid geometry remains in the scene and in its motion audit. Only
contacts inside both declared tooth annuli are replaced while the pair remains
engaged. This is an equivalent tooth model, not a tooth-impact or stress model.
"""
from __future__ import annotations

import copy
import math
import numpy as np

SPUR_MODEL_SCOPE = "finite equivalent spur tooth forces and sampled friction; not resolved tooth contact, backlash, stress or deformation"


class SpurGearError(ValueError):
    pass


def _positive(row, name, *, zero=False):
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SpurGearError(f"Spur gear {name} must be finite")
    if value < 0 or (not zero and value == 0):
        raise SpurGearError(f"Spur gear {name} must be {'nonnegative' if zero else 'positive'}")
    return float(value)


def _vector(value, label):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise SpurGearError(f"Spur gear {label} must contain three finite coordinates")
    return value


def validate_spur_gears(rows):
    if not isinstance(rows, list):
        raise SpurGearError("spur_gears must be a list")
    ids = set()
    pairs = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise SpurGearError("Spur gear IDs must be unique nonempty strings")
        ids.add(row["id"])
        a, b = row.get("body_a"), row.get("body_b")
        if not isinstance(a, str) or not isinstance(b, str) or a == b:
            raise SpurGearError("A spur gear needs two different source bodies")
        pair = tuple(sorted((a, b)))
        if pair in pairs:
            raise SpurGearError("The current spur model supports one mesh per body pair")
        pairs.add(pair)
        for key in ("teeth_a", "teeth_b"):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 17:
                raise SpurGearError("Standard unshifted spur gears require integer tooth counts >= 17")
        for key in ("module_m", "face_width_m", "stiffness_n_m", "center_tolerance_m", "axis_tolerance_rad"):
            _positive(row, key)
        _positive(row, "damping_n_s_m", zero=True)
        alpha = _positive(row, "pressure_angle_rad")
        if not 0 < alpha < math.pi / 4:
            raise SpurGearError("Spur pressure angle must lie between 0 and 45 degrees")
        if row["center_tolerance_m"] >= row["module_m"] / 2 or row["axis_tolerance_rad"] > .02:
            raise SpurGearError("Spur engagement tolerance exceeds the supported narrow domain")
        _vector(row.get("center_a_m"), "center_a_m")
        _vector(row.get("center_b_m"), "center_b_m")
        axis = _vector(row.get("axis_abs"), "axis_abs")
        if abs(np.linalg.norm(axis) - 1) > 1e-8:
            raise SpurGearError("Spur axis_abs must be a unit direction")
        if not row.get("joint_a") or not row.get("joint_b") or not row.get("parameter_basis"):
            raise SpurGearError("Spur gears require their bearing joints and nominal parameter basis")


def _rotation(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _profile_evidence(record, center, axis, teeth, module, width):
    import trimesh
    mesh = trimesh.load(record["mesh_file"], force="mesh", process=False)
    world = np.asarray(mesh.vertices) @ _rotation(record["rotation_wxyz"]).T + record["position_m"]
    delta = world-center
    axial = delta @ axis
    radial_vectors = delta-np.outer(axial, axis)
    radius = np.linalg.norm(radial_vectors, axis=1)
    pitch, tip, root = module*teeth/2, module*(teeth/2+1), module*(teeth/2-1.25)
    selected = (abs(axial) <= width/2+1e-7) & (radius >= root-5e-5)
    if selected.sum() < teeth*4 or abs(float(radius[selected].max())-tip) > 5e-5:
        raise SpurGearError("Declared spur tip radius/face interval disagrees with source mesh")
    seed = np.array([1., 0., 0.]) if abs(axis[0]) < .8 else np.array([0., 1., 0.])
    x = seed-axis*float(seed@axis); x /= np.linalg.norm(x); y = np.cross(axis, x)
    # Intersect the actual mid-plane profile. Cap tessellation may contain
    # interior vertices, which are not points on the radial outer boundary.
    grid = np.linspace(0, 2*math.pi, max(1024, teeth*64), endpoint=False)
    directions = np.outer(np.cos(grid),x)+np.outer(np.sin(grid),y)
    rotation = _rotation(record["rotation_wxyz"])
    local_center = (center-record["position_m"])@rotation
    locations, ray_ids, _ = mesh.ray.intersects_location(
        np.tile(local_center,(len(grid),1)),directions@rotation,multiple_hits=True)
    sampled = np.zeros(len(grid))
    if len(locations):
        np.maximum.at(sampled,ray_ids,np.linalg.norm(locations-local_center,axis=1))
    if np.any(sampled < root-5e-5):
        raise SpurGearError("Source spur axial face interval has missing mid-plane material or an unresolved profile ray")
    high = sampled > (pitch+tip)/2
    count = int(np.count_nonzero(high & ~np.roll(high, 1)))
    if count != teeth:
        raise SpurGearError(f"Source radial profile has {count} teeth, declaration has {teeth}")
    # A radial silhouette can hide thin or separated tooth bands. Bind the
    # declared face width to a material section through every source tooth.
    # This samples straight tooth sections; it is not a full involute proof.
    starts = np.flatnonzero(high & ~np.roll(high, 1))
    middle = [(start+(np.flatnonzero(~np.roll(high, -start))[0]-1)//2) % len(grid)
              for start in starts]
    theta = grid[middle]
    reach = width+float(np.linalg.norm(delta, axis=1).max())
    origins = (center + (pitch+tip)/2*(np.outer(np.cos(theta), x)+np.outer(np.sin(theta), y))
               - reach*axis)
    source = trimesh.Trimesh(vertices=world, faces=mesh.faces, process=False)
    locations, ray_ids, _ = source.ray.intersects_location(origins, np.tile(axis, (teeth, 1)))
    sections = (locations-center)@axis if len(locations) else np.empty(0)
    for index in range(teeth):
        interval = np.unique(np.round(sections[ray_ids == index], 10))
        # A compound shaft can carry another gear outside this tooth band.
        # Starting outside the closed source, each crossing pair bounds one
        # material interval; one whole interval must match this declaration.
        if len(interval) % 2 or not any(np.allclose(interval[i:i+2], [-width/2, width/2], atol=1e-7, rtol=0)
                                        for i in range(0, len(interval), 2)):
            raise SpurGearError("Source tooth material does not span the declared axial face interval")
    return {"observed_teeth": count, "tip_radius_m": float(radius[selected].max()),
            "axial_face_interval_m": [-width/2, width/2], "checked_axial_tooth_sections": teeth,
            "pitch_radius_m": pitch, "root_radius_m": root, "mesh_file": record["mesh_file"],
            "source_solid_ids": record.get("source_solid_ids", []),
            "source_components": [{k: v for k, v in component.items() if k in
                                   {"solid_id", "sha256", "mesh_file", "source_sha256"}}
                                  for component in record.get("source_components", [])]}


def compile_spur_gears(scene, geometry):
    rows = copy.deepcopy(scene.get("spur_gears", [])); validate_spur_gears(rows)
    records = {b["id"]: b for b in geometry["bodies"]}
    declared_joints = {j["id"]: j for j in scene.get("joints", [])}
    supports = {row["id"]: row for row in scene.get("equivalent_elasticity", {}).get("elements", [])
                if row.get("kind") == "shaft_support"}
    scale = .001 if scene["units"] == "mm" else 1.
    for row in rows:
        axis = np.asarray(row["axis_abs"])
        if any(key not in records or records[key]["fixed"] for key in (row["body_a"], row["body_b"])):
            raise SpurGearError("Spur meshes require two actual dynamic bodies")
        ca, cb = (np.asarray(row[f"center_{side}_m"]) for side in ("a", "b"))
        delta = cb-ca; distance = np.linalg.norm(delta-axis*float(delta@axis))
        nominal = row["module_m"]*(row["teeth_a"]+row["teeth_b"])/2
        if abs(distance-nominal) > row["center_tolerance_m"] or abs(delta@axis) > 1e-7:
            raise SpurGearError("Spur centers must be in the same plane at their declared pitch distance")
        row["source_evidence"] = {}
        for side in ("a", "b"):
            body_id = row[f"body_{side}"]; center = np.asarray(row[f"center_{side}_m"])
            support_id = row[f"joint_{side}"]
            joint = declared_joints.get(support_id, supports.get(support_id, {}))
            compliant = joint.get("kind") == "shaft_support"
            other = joint.get("body_b") if joint.get("body_a") == body_id else joint.get("body_a")
            if (not compliant and joint.get("type") != "revolute") or body_id not in (joint.get("body_a"), joint.get("body_b")) or not records.get(other, {}).get("fixed"):
                raise SpurGearError("Spur support requires an actual revolute bearing or typed shaft support to fixed material")
            jaxis = np.asarray(joint["axis_abs"] if compliant else joint.get("axis", [0, 0, 1]), dtype=float)
            jaxis /= np.linalg.norm(jaxis)
            anchor = np.asarray(joint["anchor_abs_m"]) if compliant else np.asarray(joint["anchor"])*scale
            if abs(jaxis@axis) < 1-1e-8 or np.linalg.norm(np.cross(center-anchor, axis)) > 1e-7:
                raise SpurGearError("Spur center/axis disagrees with the actual bearing declaration")
            record = records[body_id]; rotation = _rotation(record["rotation_wxyz"])
            row[f"center_{side}_local_m"] = ((center-record["position_m"]) @ rotation).tolist()
            row[f"axis_{side}_local"] = (axis@rotation).tolist()
            row["source_evidence"][side] = _profile_evidence(record, center, axis,
                row[f"teeth_{side}"], row["module_m"], row["face_width_m"])
        row["representation"] = "native_compliant_external_spur_gear"
    return rows


def _xyz(v):
    return np.array([v.x, v.y, v.z], dtype=float)


def build_spur_gears(c, system, bodies, joints, rows, *, axis_quat, friction, previous_filter=None):
    """Create native finite gear loads and their finite tooth-contact selector."""
    class SpurGears(c.NarrowphaseCallback):
        def __init__(self):
            super().__init__()
            self.entries, self.events, self.last_exclusions, self.friction_records = [], [], [], []
            self.excluded_contacts = 0
            self.friction_energy_j = 0.
            self.minimum_friction_impulse_scale = 1.
            self.accumulators = {key: body.AddAccumulator() for key, body in bodies.items()}
            for spec in rows:
                a, b = bodies[spec["body_a"]], bodies[spec["body_b"]]
                link = c.ChLinkLockGear(); link.SetName(spec["id"])
                link.Initialize(a, b, c.ChFramed())
                for side, number in (("a", 1), ("b", 2)):
                    frame = c.ChFramed(c.ChVector3d(*spec[f"center_{side}_local_m"]), axis_quat(c, spec[f"axis_{side}_local"]))
                    getattr(link, f"SetFrameShaft{number}")(frame)
                link.SetTransmissionRatio(spec["teeth_a"]/spec["teeth_b"])
                link.SetPressureAngle(spec["pressure_angle_rad"])
                link.SetTeethCompliant(True, spec["stiffness_n_m"], spec["damping_n_s_m"])
                link.SetPhaseAutomatically()
                if not link.IsTeethCompliant():
                    raise RuntimeError("Native gear did not enable finite-compliance teeth")
                system.AddLink(link)
                self.entries.append({"spec": spec, "link": link, "active": True})
            self.begin_step()

        def placement(self, entry):
            s = entry["spec"]
            result = {}
            for side in ("a", "b"):
                body = bodies[s[f"body_{side}"]]
                result[side] = (_xyz(body.TransformPointLocalToParent(c.ChVector3d(*s[f"center_{side}_local_m"]))),
                                _xyz(body.TransformDirectionLocalToParent(c.ChVector3d(*s[f"axis_{side}_local"]))))
            return result

        def begin_step(self):
            self.last_exclusions = []
            if previous_filter is not None:
                previous_filter.begin_step()
            self.refresh_engagement()

        def refresh_engagement(self):
            for entry in self.entries:
                if not entry["active"]:
                    continue
                s = entry["spec"]; placement = self.placement(entry)
                ca, aa = placement["a"]; cb, ab = placement["b"]
                delta = cb-ca; axial = abs(float(delta@aa))
                radial = np.linalg.norm(delta-aa*float(delta@aa))
                nominal = s["module_m"]*(s["teeth_a"]+s["teeth_b"])/2
                reason = ("bearing_released" if not all(joints[s[f"joint_{side}"]]["active"] for side in ("a", "b")) else
                          "center_distance" if abs(radial-nominal) > s["center_tolerance_m"] else
                          "axis_alignment" if float(aa@ab) < math.cos(s["axis_tolerance_rad"]) else
                          "axial_disengagement" if axial >= s["face_width_m"] else None)
                if reason:
                    entry["active"] = False; entry["link"].SetDisabled(True)
                    self.events.append({"kind": "spur_gear_released", "gear": s["id"],
                        "time_s": float(system.GetChTime()), "reason": reason,
                        "center_distance_m": float(radial), "axial_offset_m": axial})

        def after_step(self):
            # Capture the just-completed interval before an endpoint release
            # changes availability. Never rewrite an already-applied reaction.
            for entry in self.entries:
                entry["solved_step_force_n"] = float(entry["link"].GetContactForce()) if entry["active"] else 0.
                entry["active_during_step"] = bool(entry["active"])
            self.refresh_engagement()

        def in_teeth(self, entry, side, point):
            s = entry["spec"]; body = bodies[s[f"body_{side}"]]
            p = _xyz(body.TransformPointParentToLocal(point))-s[f"center_{side}_local_m"]
            axis = np.asarray(s[f"axis_{side}_local"]); z = float(p@axis)
            r = float(np.linalg.norm(p-z*axis)); m = s["module_m"]; teeth = s[f"teeth_{side}"]
            return abs(z) <= s["face_width_m"]/2+5e-5 and m*(teeth/2-1.25)-5e-5 <= r <= m*(teeth/2+1)+5e-5

        def OnNarrowphase(self, info):
            if previous_filter is not None and not previous_filter.OnNarrowphase(info):
                return False
            keys = [str(model.GetContactable().GetPhysicsItem().GetName()) for model in (info.modelA, info.modelB)]
            for entry in self.entries:
                if not entry["active"]:
                    continue
                s = entry["spec"]
                if set(keys) != {s["body_a"], s["body_b"]}:
                    continue
                sides = ["a" if key == s["body_a"] else "b" for key in keys]
                if all(self.in_teeth(entry, side, point) for side, point in zip(sides, (info.vpA, info.vpB))):
                    self.excluded_contacts += 1
                    if len(self.last_exclusions) < 8:
                        self.last_exclusions.append({"gear": s["id"], "body_pair": keys,
                            "point_a_m": _xyz(info.vpA).tolist(), "point_b_m": _xyz(info.vpB).tolist()})
                    return False
            return True

        def apply_friction(self, dt):
            """Passive sampled Coulomb loss on nominal involute contact stations.

            Normal load is the preceding native compliant solve. Tooth load
            sharing is uniform; it is an explicit approximation, never a
            statement of resolved tooth impact or local stress.
            """
            self.friction_records = []
            for key, body in bodies.items():
                body.EmptyAccumulator(self.accumulators[key])
            pending = []
            body_wrenches = {}
            world_inertias = {}
            parents = {key:key for key in bodies}
            def component(key):
                while parents[key] != key:
                    parents[key] = parents[parents[key]]
                    key = parents[key]
                return key
            for entry in self.entries:
                if not entry["active"] or friction <= 0:
                    continue
                s, link = entry["spec"], entry["link"]
                normal_load = abs(float(link.GetContactForce()))
                if not math.isfinite(normal_load):
                    raise RuntimeError("Nonfinite native gear normal force")
                ca, axis = self.placement(entry)["a"]; cb, _ = self.placement(entry)["b"]
                e = cb-ca; e -= axis*float(e@axis); e /= np.linalg.norm(e)
                tangent = np.cross(axis, e); alpha = s["pressure_angle_rad"]
                normal = math.sin(alpha)*e+math.cos(alpha)*tangent
                sliding_axis = np.cross(axis, normal)
                ra, rb = (s["module_m"]*s[f"teeth_{side}"]/2 for side in ("a", "b"))
                base_pitch = math.pi*s["module_m"]*math.cos(alpha)
                approach = math.sqrt((rb+s["module_m"])**2-(rb*math.cos(alpha))**2)-rb*math.sin(alpha)
                recess = math.sqrt((ra+s["module_m"])**2-(ra*math.cos(alpha))**2)-ra*math.sin(alpha)
                phase = (ra*math.cos(alpha)*float(link.GetRotation1())) % base_pitch
                stations = [phase+i*base_pitch for i in range(-3, 4) if -approach <= phase+i*base_pitch <= recess]
                for station in stations:
                    point = ca+ra*e+station*normal
                    velocities, inverse_effective_mass = [], 0.
                    for side in ("a", "b"):
                        body = bodies[s[f"body_{side}"]]; arm = point-_xyz(body.GetPos())
                        velocities.append(_xyz(body.GetPosDt())+np.cross(_xyz(body.GetAngVelParent()), arm))
                        key = s[f"body_{side}"]
                        if key not in world_inertias:
                            q = body.GetRot(); rotation = _rotation([q.e0, q.e1, q.e2, q.e3])
                            xx, xy = _xyz(body.GetInertiaXX()), _xyz(body.GetInertiaXY())
                            world_inertias[key] = rotation@np.array([[xx[0],xy[0],xy[1]],[xy[0],xx[1],xy[2]],[xy[1],xy[2],xx[2]]])@rotation.T
                        inertia = world_inertias[key]
                        cross = np.cross(arm, sliding_axis)
                        inverse_effective_mass += 1/float(body.GetMass())+float(cross@np.linalg.solve(inertia,cross))
                    speed = float((velocities[0]-velocities[1])@sliding_axis)
                    limit = abs(speed)/(dt*inverse_effective_mass) if inverse_effective_mass > 0 else 0.
                    magnitude = min(friction*normal_load/len(stations), limit)
                    force = -math.copysign(magnitude, speed)*sliding_axis if speed else np.zeros(3)
                    power = float(force@(velocities[0]-velocities[1]))
                    if power > 1e-12:
                        raise RuntimeError("Gear friction cannot inject power")
                    if np.any(force):
                        parents[component(s["body_a"])] = component(s["body_b"])
                    for side, sign in (("a",1), ("b",-1)):
                        key=s[f"body_{side}"]; f=sign*force
                        pending.append((s["id"]+":"+side, key, point, f))
                        resultant, moment = body_wrenches.setdefault(key, (np.zeros(3), np.zeros(3)))
                        resultant += f
                        moment += np.cross(point-_xyz(bodies[key].GetPos()), f)
            # Individual station caps do not bound their combined impulse:
            # three near-stopped stations can reverse and accelerate a body.
            # Aggregate shared-body gear pairs in the source mass/inertia
            # metric. Disconnected groups get separate budgets: dissipation
            # in a fast group cannot fund acceleration of a slow group.
            # Scale along the same Coulomb wrench until
            # its free-body kinetic-energy derivative reaches zero. This only
            # reduces the requested friction; it never raises mu*N capacity.
            component_terms = {}
            for key, (force, torque) in body_wrenches.items():
                terms = component_terms.setdefault(component(key), [0.,0.])
                terms[0] += float(force@_xyz(bodies[key].GetPosDt())+torque@_xyz(bodies[key].GetAngVelParent()))
                terms[1] += (float(force@force)/float(bodies[key].GetMass())
                             + float(torque@np.linalg.solve(world_inertias[key], torque)))
            scales = {}
            for group, (power, inverse_mass_quadratic) in component_terms.items():
                scale = (min(1., max(0., -power/(dt*inverse_mass_quadratic)))
                         if inverse_mass_quadratic > 0 else 1.)
                scales[group] = scale
                self.minimum_friction_impulse_scale = min(self.minimum_friction_impulse_scale, scale)
                self.friction_energy_j -= scale*power*dt
            for identifier, key, point, force in pending:
                force = scales[component(key)]*force
                bodies[key].AccumulateForce(self.accumulators[key],c.ChVector3d(*force),c.ChVector3d(*point),False)
                self.friction_records.append({"id":identifier, "kind":"spur_gear_friction",
                    "body_id":key,"point_abs_m":point.tolist(),"force_abs_n":force.tolist(),"torque_abs_nm":[0.,0.,0.]})

        def sample(self):
            result = {}
            for entry in self.entries:
                s, link = entry["spec"], entry["link"]
                force = entry["solved_step_force_n"] if "solved_step_force_n" in entry else (float(link.GetContactForce()) if entry["active"] else 0.)
                result[s["id"]] = {"active":entry["active"], "normal_force_n":force,
                    "active_during_step":entry.get("active_during_step",False),
                    "rotation_a_rad":float(link.GetRotation1()),"rotation_b_rad":float(link.GetRotation2())}
            return result

        def metadata(self):
            return {"representation":"native_finite_compliance_spur_gears", "declaration":copy.deepcopy(rows),
                "original_geometry_audit":"unchanged", "contact_replacement":"only both tooth annuli of an active declared pair",
                "friction_coefficient":friction,"friction_model":"sampled Coulomb loss at nominal involute stations with uniform normal-load sharing",
                "friction_impulse_limit":"Shared across stations and active gear pairs within each connected body group using full body mass and world inertia; disconnected groups have separate budgets; does not establish combined solver energy balance",
                "minimum_friction_impulse_scale":self.minimum_friction_impulse_scale,
                "friction_dissipated_energy_j":self.friction_energy_j,"excluded_native_contacts":self.excluded_contacts,
                "backlash":"not represented by the native compliant law; source backlash remains in the rigid geometry audit",
                "normal_force_sample_kind":"reaction during the completed solver step; active is availability at its endpoint",
                "scope":SPUR_MODEL_SCOPE,
                "events":copy.deepcopy(self.events)}

    callback = SpurGears()
    system.GetCollisionSystem().RegisterNarrowphaseCallback(callback)
    return callback
