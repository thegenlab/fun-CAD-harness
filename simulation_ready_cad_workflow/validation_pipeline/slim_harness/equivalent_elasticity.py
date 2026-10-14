"""Passive equivalent elasticity ABI; no FE module or author-executed callback input."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from bisect import bisect_right
import math

import numpy as np
from .shaft_supports import ShaftSupport, validate_support


SCHEMA = "slim_equivalent_elasticity_v1"


class EquivalentElasticityError(ValueError):
    pass


def _number(value, label, *, positive=False, nonnegative=False):
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (nonnegative and result < 0):
        raise EquivalentElasticityError(f"Invalid {label}: {value!r}")
    return result


def _vector(value, label, *, unit=False):
    if len(value) != 3:
        raise EquivalentElasticityError(f"{label} needs three coordinates")
    values = [_number(v, label) for v in value]
    if unit and abs(sum(v*v for v in values) - 1) > 1e-8:
        raise EquivalentElasticityError(f"{label} must be a unit direction")
    return values


def validate_equivalent_elasticity(value, body_ids, joint_specs=()):
    if value.get("schema_version") != SCHEMA or value.get("units") != "m":
        raise EquivalentElasticityError("Expected metre-based slim_equivalent_elasticity_v1")
    bodies = set(body_ids)
    joint_rows = joint_specs.values() if isinstance(joint_specs, Mapping) else joint_specs
    joints = {row["id"]: row for row in joint_rows}
    for body_id in value["elastic_reference_body_ids"]:
        if body_id not in bodies:
            raise EquivalentElasticityError("Unknown elastic reference body: " + body_id)
    seen = set()
    for element in value["elements"]:
        identifier, kind = element["id"], element["kind"]
        if not identifier or identifier in seen:
            raise EquivalentElasticityError("Missing or duplicate equivalent element ID")
        seen.add(identifier)
        if element["body_a"] == element["body_b"] or any(element[key] not in bodies for key in ("body_a", "body_b")):
            raise EquivalentElasticityError("Equivalent elements require two distinct actual bodies")
        if not element["source_body_ids"] or any(key not in bodies for key in element["source_body_ids"]):
            raise EquivalentElasticityError("Equivalent element source body is missing")
        if not isinstance(element["parameter_basis"], Mapping):
            raise EquivalentElasticityError("Parameter provenance must be explicitly supplied")
        if kind == "shaft_support":
            if identifier in joints:
                raise EquivalentElasticityError("A shaft-support provider ID must not reuse an ideal joint ID")
            validate_support(element)
            continue
        if kind == "torsional_spring":
            _vector(element["anchor_abs_m"], "anchor")
            _vector(element["axis_abs"], "torsion axis", unit=True)
            _number(element["stiffness_nm_rad"], "torsional stiffness", positive=True)
            _number(element["rest_angle_rad"], "rest angle")
            _number(element["damping_nm_s_rad"], "torsional damping", nonnegative=True)
            domain = element["working_range_rad"]
        elif kind in {"linear_spring", "point_axis_spring", "profile_detent"}:
            _vector(element["point_a_body_m"], "point A")
            _vector(element["point_b_body_m"], "point B")
            _number(element["stiffness_n_m"], "linear stiffness", positive=True)
            domain = element["working_range_m"]
            if kind == "linear_spring":
                _number(element["rest_length_m"], "rest length", nonnegative=True)
                _number(element["damping_n_s_m"], "damping", nonnegative=True)
            else:
                _vector(element["axis_a_body"], "root axis", unit=True)
                if kind == "point_axis_spring":
                    _number(element["rest_displacement_m"], "rest displacement")
                    _number(element["damping_n_s_m"], "damping", nonnegative=True)
                    if element["response"] not in {"bilateral", "compression_only"}:
                        raise EquivalentElasticityError("Unsupported point-axis response")
                else:
                    _number(element["preload_displacement_m"], "preload", nonnegative=True)
                    _number(element["profile_reference_height_m"], "profile reference height")
                    _number(element["normal_damping_n_s_m"], "normal damping", nonnegative=True)
                    knots = element["profile_knots"]
                    if len(knots) < 2:
                        raise EquivalentElasticityError("Detent profile needs at least two source points")
                    coordinates = [_number(row["coordinate_m"], "profile coordinate") for row in knots]
                    for row in knots:
                        _number(row["height_m"], "profile height")
                    if any(a >= b for a, b in zip(coordinates, coordinates[1:])):
                        raise EquivalentElasticityError("Profile coordinates must be strictly increasing")
                    if domain[0] < coordinates[0] or domain[1] > coordinates[-1]:
                        raise EquivalentElasticityError("Working range must be inside the supplied profile")
        elif kind == "friction_clutch":
            _vector(element["axis_a_body"], "clutch axis", unit=True)
            _number(element["mu"], "Coulomb coefficient", nonnegative=True)
            _number(element["effective_radius_m"], "effective radius", positive=True)
            _number(element["normal_stiffness_n_m"], "normal stiffness", positive=True)
            _number(element["preload_displacement_m"], "preload", nonnegative=True)
            joint = joints.get(element["joint_id"])
            if joint is None or joint["type"] != "revolute" or {joint["body_a"], joint["body_b"]} != {element["body_a"], element["body_b"]}:
                raise EquivalentElasticityError("Clutch must refer to its actual existing revolute pair")
            continue
        else:
            raise EquivalentElasticityError("Unsupported equivalent kind: " + kind)
        if len(domain) != 2 or _number(domain[0], "range lower") >= _number(domain[1], "range upper"):
            raise EquivalentElasticityError("Working range must be increasing")


def scalar_response(spec, q, rate):
    """Return U, generalized force Q=-dU/dq plus passive damping, and loss rate."""
    kind = spec["kind"]
    if kind == "torsional_spring":
        k, damping = spec["stiffness_nm_rad"], spec["damping_nm_s_rad"]
        deformation = q - spec["rest_angle_rad"]
        return {"potential_energy_j": .5*k*deformation**2, "generalized_force": -k*deformation-damping*rate,
                "dissipation_rate_w": damping*rate**2, "deformation": deformation}
    k = spec["stiffness_n_m"]
    if kind == "profile_detent":
        knots = spec["profile_knots"]
        index = min(max(bisect_right([row["coordinate_m"] for row in knots], q)-1, 0), len(knots)-2)
        first, second = knots[index:index+2]
        slope = (second["height_m"]-first["height_m"])/(second["coordinate_m"]-first["coordinate_m"])
        height = first["height_m"] + slope*(q-first["coordinate_m"])
        compression = max(0., spec["preload_displacement_m"] + height-spec["profile_reference_height_m"])
        velocity = slope*rate
        normal = max(0., k*compression+spec["normal_damping_n_s_m"]*velocity) if compression > 0 else 0.
        return {"potential_energy_j": .5*k*compression**2, "generalized_force": -normal*slope,
                "dissipation_rate_w": max(0., (normal-k*compression)*velocity), "deformation": compression,
                "profile_height_m": height, "profile_slope": slope, "normal_force_n": normal}
    rest = spec["rest_length_m"] if kind == "linear_spring" else spec["rest_displacement_m"]
    deformation = q-rest
    damping = spec["damping_n_s_m"]
    if kind == "point_axis_spring" and spec["response"] == "compression_only":
        deformation = min(deformation, 0.)
        elastic_force = -k*deformation
        force = max(0., elastic_force-damping*rate) if deformation < 0 else 0.
        return {"potential_energy_j": .5*k*deformation**2, "generalized_force": force,
                "dissipation_rate_w": max(0., -(force-elastic_force)*rate), "deformation": deformation}
    return {"potential_energy_j": .5*k*deformation**2, "generalized_force": -k*deformation-damping*rate,
            "dissipation_rate_w": damping*rate**2, "deformation": deformation}


def _xyz(vector):
    return np.asarray([float(vector.x), float(vector.y), float(vector.z)])


def _direction(c, body, local):
    return _xyz(body.TransformDirectionLocalToParent(c.ChVector3d(*local)))


def _point(c, body, local):
    point = _xyz(body.TransformPointLocalToParent(c.ChVector3d(*local)))
    velocity = _xyz(body.GetPosDt()) + np.cross(_xyz(body.GetAngVelParent()), point-_xyz(body.GetPos()))
    return point, velocity


def _inertia(body):
    xx, xy = _xyz(body.GetInertiaXX()), _xyz(body.GetInertiaXY())
    return np.asarray([[xx[0], xy[0], xy[1]], [xy[0], xx[1], xy[2]], [xy[1], xy[2], xx[2]]])


def inertia_partition(matrix, direction):
    """An exact rank-one body/shaft representation split, within the SPD bound."""
    matrix, direction = np.asarray(matrix, dtype=float), np.asarray(direction, dtype=float)
    if min(np.linalg.eigvalsh(matrix)) <= 0:
        raise EquivalentElasticityError("Inertia partition requires a positive-definite source tensor")
    limit = 1/float(direction @ np.linalg.solve(matrix, direction))
    shaft_inertia = limit/8
    remainder = matrix-shaft_inertia*np.outer(direction, direction)
    if min(np.linalg.eigvalsh(remainder)) <= 0:
        raise EquivalentElasticityError("Inertia representation did not preserve positive definiteness")
    return shaft_inertia, remainder


class EquivalentElasticity:
    def __init__(self, c, system, bodies, value):
        self.c, self.system, self.bodies = c, system, bodies
        self.declaration = deepcopy(value)
        self.entries, self.inertia_allocations, self.native_objects = [], [], []
        self.support_records = {}
        self.accumulators = {key: body.AddAccumulator() for key, body in bodies.items()}
        self._pending = None
        self._applied_records = []
        self._native_step_loads = None
        self._completed_steps = 0

    def _add_inertia_port(self, body_id, direction):
        c, system, body = self.c, self.system, self.bodies[body_id]
        direction = np.asarray(direction, dtype=float)
        before = _inertia(body)
        inertia, remainder = inertia_partition(before, direction)
        body.SetInertiaXX(c.ChVector3d(*np.diag(remainder)))
        body.SetInertiaXY(c.ChVector3d(remainder[0, 1], remainder[0, 2], remainder[1, 2]))
        shaft = c.ChShaft()
        shaft.SetInertia(inertia)
        shaft.SetPosDt(float(_direction(c, body, direction)@_xyz(body.GetAngVelParent())))
        system.Add(shaft)
        coupling = c.ChShaftBodyRotation()
        if coupling.Initialize(shaft, body, c.ChVector3d(*direction)) is False:
            raise EquivalentElasticityError("Native shaft/body coupling initialization failed")
        system.Add(coupling)
        observed = _inertia(body)+float(shaft.GetInertia())*np.outer(direction, direction)
        if not np.allclose(before, observed, rtol=1e-12, atol=1e-18):
            raise EquivalentElasticityError("Native body+shaft inertia did not preserve the source tensor")
        self.inertia_allocations.append({"body_id": body_id, "axis_body": direction.tolist(),
            "body_inertia_before_kg_m2": before.tolist(), "body_inertia_after_kg_m2": _inertia(body).tolist(),
            "shaft_inertia_kg_m2": float(shaft.GetInertia()), "body_fixed": bool(body.IsFixed()),
            "representation": "Rank-one inertia redistribution with zero-power gyroscopic completion",
            "gyroscopic_completion_body_local": "-J*(n dot omega)*(omega cross n)"})
        self.native_objects.extend((shaft, coupling))
        return shaft

    def _state(self, entry):
        spec = entry["spec"]
        a, b = self.bodies[spec["body_a"]], self.bodies[spec["body_b"]]
        kind = spec["kind"]
        if kind == "torsional_spring":
            ax = _direction(self.c, a, entry["x_a"])
            bx, by = _direction(self.c, b, entry["x_b"]), _direction(self.c, b, entry["y_b"])
            axis = _direction(self.c, b, entry["axis_b"])
            phase = math.atan2(float(by@ax), float(bx@ax))
            delta = math.atan2(math.sin(phase-entry["phase"]), math.cos(phase-entry["phase"]))
            coordinate = entry["angle"]+delta
            rate = float(axis @ (_xyz(a.GetAngVelParent())-_xyz(b.GetAngVelParent())))
            return {"coordinate": coordinate, "rate": rate, "phase": phase, "axis": axis}
        if kind == "friction_clutch":
            first, second = entry["shafts"]
            return {"coordinate": float(first.GetPos()-second.GetPos()), "rate": float(first.GetPosDt()-second.GetPosDt())}
        pa, va = _point(self.c, a, spec["point_a_body_m"])
        pb, vb = _point(self.c, b, spec["point_b_body_m"])
        difference = pb-pa
        if kind == "linear_spring":
            coordinate = float(np.linalg.norm(difference))
            if coordinate <= 1e-12:
                raise EquivalentElasticityError("TSDA endpoints became coincident")
            axis = difference/coordinate
            return {"coordinate": coordinate, "rate": float(axis@(vb-va)), "point_a": pa, "point_b": pb, "axis": axis}
        axis = _direction(self.c, a, spec["axis_a_body"])
        coordinate = float(axis@difference)
        rate = float(axis@(vb-va)+np.cross(_xyz(a.GetAngVelParent()), axis)@difference)
        return {"coordinate": coordinate, "rate": rate, "point_a": pa, "point_b": pb, "axis": axis}

    def metadata(self):
        return {"schema_version": SCHEMA, "declaration": deepcopy(self.declaration),
                "torsional_coordinate": "Accepted-step unwrap of actual body-frame relative pose; stock RSDA angle is not used",
                "inertia_allocations": deepcopy(self.inertia_allocations),
                "energy_scope": "Passive equivalent potentials and dissipation; no coil shape, bind, stress or fatigue proof",
                "mass_scope": "Original source total inertia retained; any native shaft inertia is deducted from its body's tensor",
                "profile_extrapolation": "Terminal segment continues linearly; out-of-working-range state is recorded, not source-validated",
                "shaft_supports": [entry["support"].metadata() for entry in self.entries if "support" in entry]}

    def applied_load_records(self):
        return deepcopy(self._applied_records)

    def capture_native_step_loads(self):
        """Capture TSDA after system.Update and before the actual dynamics step."""
        if self._pending is None:
            raise EquivalentElasticityError("before_step is required before capturing native loads")
        if abs(float(self.system.GetChTime())-self._pending[0]) > 1e-10:
            raise EquivalentElasticityError("Native TSDA forces must be captured before dynamics advances")
        rows = []
        for entry in self.entries:
            spec = entry["spec"]
            if spec["kind"] == "linear_spring":
                pa, pb = _xyz(entry["link"].GetPoint1Abs()), _xyz(entry["link"].GetPoint2Abs())
                difference = pa-pb
                length = float(np.linalg.norm(difference))
                if length <= 1e-12:
                    raise EquivalentElasticityError("Native TSDA has coincident force application points")
                first_force = float(entry["link"].GetForce())*difference/length
                for key, point, force in ((spec["body_a"], pa, first_force), (spec["body_b"], pb, -first_force)):
                    rows.append({"id": spec["id"]+":"+key, "kind": "equivalent_elasticity", "body_id": key,
                                 "point_abs_m": point.tolist(), "force_abs_n": force.tolist(), "torque_abs_nm": [0., 0., 0.]})
        self._native_step_loads = rows
        return deepcopy(rows)

    def solved_load_records(self):
        """Accepted-step TSDA cache plus solved clutch reaction; never external input work."""
        if self._pending is not None:
            raise EquivalentElasticityError("Native step loads are available only after after_step")
        if self._completed_steps == 0:
            return []
        if self._native_step_loads is None and any(entry["spec"]["kind"] == "linear_spring" for entry in self.entries):
            raise EquivalentElasticityError("Native TSDA step force capture is missing")
        rows = deepcopy(self._native_step_loads or [])
        for entry in self.entries:
            spec = entry["spec"]
            if spec["kind"] == "friction_clutch":
                joint = entry["joint_record"]
                for side, method in (("a", "GetReaction1"), ("b", "GetReaction2")):
                    key = spec["body_"+side]
                    number = 1 if joint["spec"]["body_a"] == key else 2
                    joint_frame = getattr(joint["link"], f"GetFrame{number}Abs")()
                    point = _xyz(joint_frame.GetPos())
                    axis = _direction(self.c, self.bodies[key], entry["axis_"+side])
                    torque = axis*float(getattr(entry["link"], method)())
                    rows.append({"id": spec["id"]+":"+key, "kind": "equivalent_elasticity", "body_id": key,
                                 "point_abs_m": point.tolist(), "force_abs_n": [0., 0., 0.], "torque_abs_nm": torque.tolist(),
                                 "recipient_representation": "source body plus its allocated native shaft inertia"})
        return rows

    def before_step(self, dt):
        if self._pending is not None:
            raise EquivalentElasticityError("after_step must finish the previous accepted native step")
        dt = _number(dt, "step duration", positive=True)
        for entry in self.entries:
            if entry["spec"]["kind"] == "friction_clutch" and not entry["joint_record"].get("active", True):
                raise EquivalentElasticityError("MODEL_DOMAIN_EXIT: clutch "+entry["spec"]["id"]+
                    " requires its active revolute joint "+entry["spec"]["joint_id"])
        for key, accumulator in self.accumulators.items():
            self.bodies[key].EmptyAccumulator(accumulator)
        self._applied_records = []
        self._native_step_loads = None
        # A scalar shaft supplies J*n*n^T angular acceleration, but not the
        # transverse Euler term of that body-attached inertia. Restore that
        # missing term explicitly; omega dot this torque is identically zero.
        for index, allocation in enumerate(self.inertia_allocations):
            key = allocation["body_id"]
            body = self.bodies[key]
            omega = _xyz(body.TransformDirectionParentToLocal(body.GetAngVelParent()))
            axis = np.asarray(allocation["axis_body"])
            local = -allocation["shaft_inertia_kg_m2"]*float(axis@omega)*np.cross(omega, axis)
            torque = _direction(self.c, body, local)
            body.AccumulateTorque(self.accumulators[key], self.c.ChVector3d(*torque), False)
            self._applied_records.append({"id": f"inertia:{key}:{index}", "kind": "inertia_representation", "body_id": key,
                "point_abs_m": _xyz(body.GetPos()).tolist(), "force_abs_n": [0., 0., 0.], "torque_abs_nm": torque.tolist(),
                "instantaneous_power_w": float(local@omega)})
        pending = {}
        for entry in self.entries:
            spec = entry["spec"]
            if spec["kind"] == "shaft_support":
                entry["support"].before_step(dt, float(self.system.GetChTime()))
                self._applied_records.extend(entry["support"].applied_records)
                continue
            state = self._state(entry)
            if spec["kind"] == "friction_clutch":
                pending[spec["id"]] = {"state": state}
                continue
            law = scalar_response(spec, state["coordinate"], state["rate"])
            entry["last_applied_step_generalized_force"] = law["generalized_force"] if spec["kind"] != "linear_spring" else None
            pending[spec["id"]] = {"state": state, "law": law}
            force = law["generalized_force"]
            if spec["kind"] == "torsional_spring":
                if abs(state["rate"])*dt >= math.pi:
                    raise EquivalentElasticityError("Torsional motion is too fast for unambiguous pose unwrapping at this step")
                for key, sign in ((spec["body_a"], 1), (spec["body_b"], -1)):
                    torque = state["axis"]*(sign*force)
                    self.bodies[key].AccumulateTorque(self.accumulators[key], self.c.ChVector3d(*torque), False)
                    self._applied_records.append({"id": spec["id"]+":"+key, "kind": "equivalent_elasticity", "body_id": key,
                        "point_abs_m": _xyz(self.bodies[key].GetPos()).tolist(), "force_abs_n": [0., 0., 0.], "torque_abs_nm": torque.tolist()})
            elif spec["kind"] != "linear_spring":
                # Both forces act at the current output point. This supplies
                # the root-axis orientation moment required by -grad U(q).
                for key, sign in ((spec["body_b"], 1), (spec["body_a"], -1)):
                    value = state["axis"]*(sign*force)
                    self.bodies[key].AccumulateForce(self.accumulators[key], self.c.ChVector3d(*value), self.c.ChVector3d(*state["point_b"]), False)
                    self._applied_records.append({"id": spec["id"]+":"+key, "kind": "equivalent_elasticity", "body_id": key,
                        "point_abs_m": state["point_b"].tolist(), "force_abs_n": value.tolist(), "torque_abs_nm": [0., 0., 0.]})
        self._pending = (float(self.system.GetChTime()), dt, pending)

    def after_step(self):
        if self._pending is None:
            raise EquivalentElasticityError("before_step was not called")
        start, dt, pending = self._pending
        if abs(float(self.system.GetChTime())-start-dt) > max(1e-10, dt*1e-7):
            raise EquivalentElasticityError("Accepted native time does not match this recorded step")
        if self._native_step_loads is None and any(entry["spec"]["kind"] == "linear_spring" for entry in self.entries):
            raise EquivalentElasticityError("Missing capture_native_step_loads before the accepted TSDA step")
        for entry in self.entries:
            spec = entry["spec"]
            if spec["kind"] == "shaft_support":
                entry["support"].after_step()
                continue
            current = self._state(entry)
            previous = pending[spec["id"]]
            if spec["kind"] == "torsional_spring":
                entry["angle"], entry["phase"] = current["coordinate"], current["phase"]
            if spec["kind"] == "friction_clutch":
                torque = float(entry["link"].GetReaction1())
                work = torque*.5*(previous["state"]["rate"]+current["rate"])*dt
                entry["work_on_bodies_j"] += work
                entry["dissipated_j"] += max(0., -work)
                entry["positive_friction_work_j"] += max(0., work)
            else:
                law = scalar_response(spec, current["coordinate"], current["rate"])
                entry["work_on_bodies_j"] += previous["law"]["generalized_force"]*(current["coordinate"]-previous["state"]["coordinate"])
                entry["dissipated_j"] += .5*(previous["law"]["dissipation_rate_w"]+law["dissipation_rate_w"])*dt
        self._pending = None
        self._completed_steps += 1

    def sample(self):
        rows = {}
        for entry in self.entries:
            spec = entry["spec"]
            if spec["kind"] == "shaft_support":
                rows[spec["id"]] = entry["support"].sample()
                continue
            state = self._state(entry)
            if spec["kind"] == "friction_clutch":
                normal = spec["normal_stiffness_n_m"]*spec["preload_displacement_m"]
                axes = [_direction(self.c, self.bodies[spec["body_"+side]], entry["axis_"+side]) for side in ("a", "b")]
                velocity_errors = {side: float(entry["shafts"][index].GetPosDt())-
                    float(axes[index]@_xyz(self.bodies[spec["body_"+side]].GetAngVelParent())) for index, side in enumerate(("a", "b"))}
                row = {"kind": spec["kind"], "coordinate": state["coordinate"], "rate": state["rate"],
                       "normal_force_n": normal, "torque_limit_nm": spec["mu"]*normal*spec["effective_radius_m"],
                       "torque_on_a_nm": float(entry["link"].GetReaction1()), "torque_on_b_nm": float(entry["link"].GetReaction2()),
                       "potential_energy_j": .5*spec["normal_stiffness_n_m"]*spec["preload_displacement_m"]**2,
                       "positive_friction_work_j": entry["positive_friction_work_j"], "within_working_range": True,
                       "shaft_minus_body_speed_rad_s": velocity_errors,
                       "axis_alignment_error": float(np.linalg.norm(axes[0]-axes[1]))}
            else:
                law = scalar_response(spec, state["coordinate"], state["rate"])
                domain = spec["working_range_rad"] if spec["kind"] == "torsional_spring" else spec["working_range_m"]
                row = {"kind": spec["kind"], "coordinate": state["coordinate"], "rate": state["rate"], **law,
                       "within_working_range": domain[0]-1e-10 <= state["coordinate"] <= domain[1]+1e-10}
                if spec["kind"] == "linear_spring":
                    row["native_force_n"] = float(entry["link"].GetForce())
                    row["native_force_sample_kind"] = "current_state_constitutive"
            row.update(work_on_bodies_j=entry["work_on_bodies_j"], dissipated_energy_j=entry["dissipated_j"],
                       initial_potential_energy_j=entry["initial_potential_energy_j"])
            row["applied_step_generalized_force"] = entry.get("last_applied_step_generalized_force")
            row["discrete_energy_residual_j"] = row["potential_energy_j"]-entry["initial_potential_energy_j"]+entry["work_on_bodies_j"]+entry["dissipated_j"]
            rows[spec["id"]] = row
        return {"schema_version": "slim_equivalent_elastic_state_v1", "time_s": float(self.system.GetChTime()), "elements": rows,
                "potential_energy_j": sum(row["potential_energy_j"] for row in rows.values()),
                "dissipated_energy_j": sum(row["dissipated_energy_j"] for row in rows.values())}


def build_equivalent_elasticity(c, system, bodies, joints, value):
    specs = [row["spec"] for row in joints.values()]
    validate_equivalent_elasticity(value, bodies, specs)
    handle = EquivalentElasticity(c, system, bodies, value)

    for spec in handle.declaration["elements"]:
        a, b = bodies[spec["body_a"]], bodies[spec["body_b"]]
        entry = {"spec": spec, "work_on_bodies_j": 0., "dissipated_j": 0., "positive_friction_work_j": 0.}
        kind = spec["kind"]
        if kind == "shaft_support":
            support = ShaftSupport(c, bodies, spec, handle.accumulators)
            entry["support"] = support
            handle.support_records[spec["id"]] = support.record
            handle.entries.append(entry)
            continue
        if kind == "torsional_spring":
            axis = np.asarray(spec["axis_abs"], dtype=float)
            seed = np.asarray([1., 0., 0.] if abs(axis[0]) < .9 else [0., 1., 0.])
            x = seed-axis*(axis@seed); x /= np.linalg.norm(x)
            y = np.cross(axis, x)
            entry.update(x_a=_xyz(a.TransformDirectionParentToLocal(c.ChVector3d(*x))),
                         x_b=_xyz(b.TransformDirectionParentToLocal(c.ChVector3d(*x))),
                         y_b=_xyz(b.TransformDirectionParentToLocal(c.ChVector3d(*y))),
                         axis_b=_xyz(b.TransformDirectionParentToLocal(c.ChVector3d(*axis))), angle=0., phase=0.)
        elif kind == "linear_spring":
            link = c.ChLinkTSDA()
            link.Initialize(a, b, True, c.ChVector3d(*spec["point_a_body_m"]), c.ChVector3d(*spec["point_b_body_m"]))
            link.SetRestLength(spec["rest_length_m"])
            link.SetSpringCoefficient(spec["stiffness_n_m"])
            link.SetDampingCoefficient(spec["damping_n_s_m"])
            link.SetActuatorForce(0.)
            system.Add(link)
            entry["link"] = link
            handle.native_objects.append(link)
        elif kind == "friction_clutch":
            axis_a = np.asarray(spec["axis_a_body"], dtype=float)
            world = _direction(c, a, axis_a)
            axis_b = _xyz(b.TransformDirectionParentToLocal(c.ChVector3d(*world)))
            first, second = handle._add_inertia_port(spec["body_a"], axis_a), handle._add_inertia_port(spec["body_b"], axis_b)
            link = c.ChShaftsClutch()
            if link.Initialize(first, second) is False:
                raise EquivalentElasticityError("Native clutch initialization failed")
            link.SetTorqueLimit(spec["mu"]*spec["normal_stiffness_n_m"]*spec["preload_displacement_m"]*spec["effective_radius_m"])
            link.SetModulation(1.)
            system.Add(link)
            entry.update(link=link, shafts=(first, second), axis_a=axis_a, axis_b=axis_b,
                         joint_record=joints[spec["joint_id"]])
            handle.native_objects.append(link)
        handle.entries.append(entry)
        state = handle._state(entry)
        entry["initial_potential_energy_j"] = (.5*spec["normal_stiffness_n_m"]*spec["preload_displacement_m"]**2
            if kind == "friction_clutch" else scalar_response(spec, state["coordinate"], state["rate"])["potential_energy_j"])
    return handle
