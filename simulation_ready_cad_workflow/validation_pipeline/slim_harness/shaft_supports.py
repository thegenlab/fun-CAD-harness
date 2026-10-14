"""Finite, passive shaft supports on the original Chrono bodies.

Two separated radial bearings leave axial travel and shaft rotation free. Local
compression-only stops supply axial retention. This is a declared lumped model,
not an ideal joint or a replacement time integrator.
"""
from __future__ import annotations

import math
import numpy as np


KIND = "shaft_support"


def _positive(value, name, *, zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not zero and value == 0):
        raise ValueError(f"Invalid shaft-support {name}")
    return value


def _vector(value, name, *, unit=False):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"Invalid shaft-support {name}")
    if unit and not np.isclose(np.linalg.norm(result), 1., atol=1e-8, rtol=0):
        raise ValueError(f"Shaft-support {name} must be unit length")
    return result


def validate_support(spec):
    axis = _vector(spec["axis_abs"], "axis", unit=True)
    anchor = _vector(spec["anchor_abs_m"], "anchor")
    bearings, stops = spec["bearings"], spec["axial_stops"]
    if len(bearings) != 2 or not stops:
        raise ValueError("A shaft support requires two separated bearings and axial stops")
    identifiers = [row["id"] for row in bearings + stops]
    if any(not isinstance(key, str) or not key for key in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Shaft-support contact IDs must be unique")
    centers = []
    for bearing in bearings:
        center = _vector(bearing["center_abs_m"], "bearing center")
        if np.linalg.norm(np.cross(center-anchor, axis)) > 1e-7:
            raise ValueError("Both bearing centers must lie on the declared shaft axis")
        centers.append(center)
        _positive(bearing["clearance_m"], "radial clearance", zero=True)
        _positive(bearing["shaft_radius_m"], "shaft radius")
    if abs(float((centers[1]-centers[0])@axis)) < 1e-6:
        raise ValueError("Separated bearing stations are required to support shaft tilt")
    directions = []
    for stop in stops:
        _vector(stop["point_a_abs_m"], "stop material point")
        normal = _vector(stop["normal_abs"], "stop normal", unit=True)
        alignment = float(normal@axis)
        if abs(alignment) < 1-1e-8:
            raise ValueError("Axial stop normals must be parallel to the shaft axis")
        directions.append(alignment)
        _positive(stop["gap_m"], "axial gap", zero=True)
    if min(directions) >= 0 or max(directions) <= 0:
        raise ValueError("Axial retention requires stops in both directions")
    for row in bearings + stops:
        _positive(row["stiffness_n_m"], "stiffness")
        _positive(row["damping_n_s_m"], "damping", zero=True)
        _positive(row["friction"], "friction", zero=True)
    _positive(spec["friction_regularization_m_s"], "friction regularization")
    limits = spec["release"]
    for name in ("radial_offset_m", "axial_offset_m", "tilt_rad"):
        _positive(limits[name], name)
    if limits["radial_offset_m"] <= max(row["clearance_m"] for row in bearings):
        raise ValueError("Support radial release must be outside the bearing clearance")
    if limits["axial_offset_m"] <= max(row["gap_m"] for row in stops):
        raise ValueError("Support axial release must be outside the stop clearance")
    if limits["tilt_rad"] >= math.pi/2:
        raise ValueError("A shaft support needs a finite small-tilt operating domain")


def _rotation(record):
    w, x, y, z = _vector_quaternion(record["rotation_wxyz"])
    return np.asarray([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _vector_quaternion(value):
    q = np.asarray(value, dtype=float)
    return q/np.linalg.norm(q)


def compile_support(spec, records):
    """Bind world-space source points to actual original COM frames."""
    from .geometry import geometry_distance_to_point
    validate_support(spec)
    a, b = records[spec["body_a"]], records[spec["body_b"]]
    if a["fixed"] or not b["fixed"]:
        raise ValueError("Current shaft support requires a dynamic shaft and a fixed support body")
    rotations = {"a": _rotation(a), "b": _rotation(b)}
    bodies = {"a": a, "b": b}
    def local(point, side):
        return (rotations[side].T@(np.asarray(point)-bodies[side]["position_m"])).tolist()
    for side in ("a", "b"):
        spec["anchor_"+side+"_body_m"] = local(spec["anchor_abs_m"], side)
        spec["axis_"+side+"_body"] = (rotations[side].T@spec["axis_abs"]).tolist()
    source_checks = []
    for row in spec["bearings"]:
        point = row["center_abs_m"]
        row["point_a_body_m"], row["point_b_body_m"] = local(point, "a"), local(point, "b")
        distances = [geometry_distance_to_point(body, point) for body in (a, b)]
        if max(distances) > row["shaft_radius_m"]+row["clearance_m"]+1e-6:
            raise ValueError("Bearing station is not near its declared original shaft/support material")
        source_checks.append({"id": row["id"], "source_distance_m": distances})
    for row in spec["axial_stops"]:
        point_a = np.asarray(row["point_a_abs_m"])
        point_b = point_a-row["gap_m"]*np.asarray(row["normal_abs"])
        distances = [geometry_distance_to_point(body, point.tolist()) for body, point in ((a, point_a), (b, point_b))]
        if max(distances) > 1e-6:
            raise ValueError("Axial stop points must lie on the actual original retaining surfaces")
        row["point_a_body_m"], row["point_b_body_m"] = local(point_a, "a"), local(point_b, "b")
        row["normal_b_body"] = (rotations["b"].T@row["normal_abs"]).tolist()
        source_checks.append({"id": row["id"], "source_distance_m": distances})
    spec["source_support_checks"] = source_checks


def compression_response(penetration, rate, stiffness, damping):
    """Positive rate means increasing compression; never attracts across a gap."""
    if penetration <= 0:
        return 0., 0., 0.
    elastic = stiffness*penetration
    normal = max(0., elastic+damping*rate)
    return normal, .5*stiffness*penetration**2, max(0., (normal-elastic)*rate)


def _xyz(v):
    return np.asarray([float(v.x), float(v.y), float(v.z)])


class ShaftSupport:
    def __init__(self, c, bodies, spec, accumulators):
        self.c, self.spec, self.bodies, self.accumulators = c, spec, bodies, accumulators
        self.a, self.b = bodies[spec["body_a"]], bodies[spec["body_b"]]
        self.record = {"active": True, "kind": KIND, "spec": spec}
        self.events, self.applied_records = [], []
        self.work_j = self.dissipation_j = self.release_dissipation_j = 0.
        self.positive_friction_work_j = 0.
        self.applied_friction_scale = 1.
        self.step_s = None
        self.pending = None
        self.initial_potential_j = self._evaluate(False)["potential_energy_j"]

    def _point(self, body, local):
        return _xyz(body.TransformPointLocalToParent(self.c.ChVector3d(*local)))

    def _direction(self, body, local):
        return _xyz(body.TransformDirectionLocalToParent(self.c.ChVector3d(*local)))

    def _velocity(self, body, point):
        return _xyz(body.GetPosDt())+np.cross(_xyz(body.GetAngVelParent()), point-_xyz(body.GetPos()))

    def _metrics(self):
        axis = self._direction(self.b, self.spec["axis_b_body"])
        shaft_axis = self._direction(self.a, self.spec["axis_a_body"])
        offsets = [self._point(self.a, row["point_a_body_m"])-self._point(self.b, row["point_b_body_m"])
                   for row in self.spec["bearings"]]
        radial = max(np.linalg.norm(q-axis*float(q@axis)) for q in offsets)
        q = self._point(self.a, self.spec["anchor_a_body_m"])-self._point(self.b, self.spec["anchor_b_body_m"])
        return {"radial_offset_m": float(radial), "axial_offset_m": float(q@axis),
                "tilt_rad": math.acos(float(np.clip(axis@shaft_axis, -1., 1.)))}

    def _force_pair(self, identifier, force_a, point):
        for side, sign, body in (("a", 1., self.a), ("b", -1., self.b)):
            key = self.spec["body_"+side]
            force = sign*force_a
            body.AccumulateForce(self.accumulators[key], self.c.ChVector3d(*force), self.c.ChVector3d(*point), False)
            self.applied_records.append({"id": self.spec["id"]+":"+identifier+":"+side,
                "kind": "equivalent_shaft_support", "body_id": key,
                "point_abs_m": point.tolist(), "force_abs_n": force.tolist(), "torque_abs_nm": [0., 0., 0.]})

    def _friction(self, row, normal, normal_axis, point, apply):
        velocity = self._velocity(self.a, point)-self._velocity(self.b, point)
        tangent = velocity-normal_axis*float(velocity@normal_axis)
        regularization = self.spec["friction_regularization_m_s"]
        force = -row["friction"]*normal*tangent/math.sqrt(float(tangent@tangent)+regularization**2)
        if apply and np.any(force):
            self._friction_candidates.append((row["id"], force, point))
        return max(0., -float(force@tangent))

    def _friction_scale(self):
        # One shared impulse budget covers all bearing and stop friction on
        # this pair; independently clipping each point could over-stop a shaft.
        power, inverse_mass_quadratic = 0., 0.
        for body, sign in ((self.a, 1.), (self.b, -1.)):
            if body.IsFixed():
                continue
            center = _xyz(body.GetPos())
            force = sum((sign*f for _, f, _ in self._friction_candidates), np.zeros(3))
            torque = sum((np.cross(p-center, sign*f) for _, f, p in self._friction_candidates), np.zeros(3))
            local_torque = _xyz(body.TransformDirectionParentToLocal(self.c.ChVector3d(*torque)))
            xx, xy = _xyz(body.GetInertiaXX()), _xyz(body.GetInertiaXY())
            inertia = np.asarray([[xx[0], xy[0], xy[1]], [xy[0], xx[1], xy[2]], [xy[1], xy[2], xx[2]]])
            power += float(force@_xyz(body.GetPosDt())+torque@_xyz(body.GetAngVelParent()))
            inverse_mass_quadratic += float(force@force)/float(body.GetMass())
            inverse_mass_quadratic += float(local_torque@np.linalg.solve(inertia, local_torque))
        if inverse_mass_quadratic <= 0:
            return 1.
        # Stop at the minimum kinetic energy along the proposed friction
        # impulse, rather than carrying that impulse beyond a full stop.
        return min(1., max(0., -power/(self.step_s*inverse_mass_quadratic)))

    def _evaluate(self, apply):
        values = {"potential_energy_j": 0., "dissipation_rate_w": 0.,
                  "normal_dissipation_rate_w": 0., "max_normal_force_n": 0.}
        friction_loss = 0.
        if apply:
            self._friction_candidates = []
        if not self.record["active"]:
            return values
        axis = self._direction(self.b, self.spec["axis_b_body"])
        for row in self.spec["bearings"]:
            pa, pb = self._point(self.a, row["point_a_body_m"]), self._point(self.b, row["point_b_body_m"])
            q = pa-pb
            radial = q-axis*float(q@axis)
            distance = float(np.linalg.norm(radial))
            if distance <= row["clearance_m"]:
                continue
            normal_axis = radial/distance
            rate = float(normal_axis@(self._velocity(self.a, pa)-self._velocity(self.b, pa)))
            normal, potential, loss = compression_response(distance-row["clearance_m"], rate,
                row["stiffness_n_m"], row["damping_n_s_m"])
            # Applying the paired radial forces at pa also supplies the moment
            # required by a bearing plane attached to the support body.
            if apply and normal:
                self._force_pair(row["id"]+":normal", -normal*normal_axis, pa)
            surface_point = pa+row["shaft_radius_m"]*normal_axis
            friction_loss += self._friction(row, normal, normal_axis, surface_point, apply)
            values["potential_energy_j"] += potential
            values["dissipation_rate_w"] += loss
            values["max_normal_force_n"] = max(values["max_normal_force_n"], normal)
        for row in self.spec["axial_stops"]:
            pa, pb = self._point(self.a, row["point_a_body_m"]), self._point(self.b, row["point_b_body_m"])
            normal_axis = self._direction(self.b, row["normal_b_body"])
            gap = float(normal_axis@(pa-pb))
            if gap >= 0:
                continue
            rate = -float(normal_axis@(self._velocity(self.a, pa)-self._velocity(self.b, pa)))
            normal, potential, loss = compression_response(-gap, rate, row["stiffness_n_m"], row["damping_n_s_m"])
            if apply and normal:
                self._force_pair(row["id"]+":normal", normal*normal_axis, pa)
            friction_loss += self._friction(row, normal, normal_axis, pa, apply)
            values["potential_energy_j"] += potential
            values["dissipation_rate_w"] += loss
            values["max_normal_force_n"] = max(values["max_normal_force_n"], normal)
        if apply:
            self.applied_friction_scale = self._friction_scale()
            for identifier, force, point in self._friction_candidates:
                self._force_pair(identifier+":friction", self.applied_friction_scale*force, point)
        values["normal_dissipation_rate_w"] = values["dissipation_rate_w"]
        values["dissipation_rate_w"] += friction_loss*(self.applied_friction_scale if apply else 1.)
        return values

    def _refresh_activity(self, time_s):
        metrics = self._metrics()
        limits = self.spec["release"]
        outside = [name for name, value in metrics.items() if abs(value) > limits[name]]
        if self.record["active"] and outside:
            lost = self._evaluate(False)["potential_energy_j"]
            self.release_dissipation_j += lost
            self.dissipation_j += lost
            self.record["active"] = False
            self.events.append({"time_s": time_s, "kind": "shaft_support_released", "support": self.spec["id"],
                                "limits_exceeded": outside, "released_elastic_energy_j": lost, **metrics})

    def before_step(self, dt, time_s):
        self.applied_records = []
        self.step_s = dt
        self.step_end_s = time_s+dt
        self._refresh_activity(time_s)
        values = self._evaluate(True)
        before = {key: (_xyz(self.bodies[key].GetPosDt()), _xyz(self.bodies[key].GetAngVelParent()),
                        _xyz(self.bodies[key].GetPos())) for key in (self.spec["body_a"], self.spec["body_b"])}
        self.pending = (dt, values.get("normal_dissipation_rate_w", 0.), before)

    def after_step(self):
        dt, loss, before = self.pending
        friction_work = 0.
        for row in self.applied_records:
            key = row["body_id"]
            body = self.bodies[key]
            if body.IsFixed():
                continue
            velocity, omega, center = before[key]
            force = np.asarray(row["force_abs_n"])
            torque = np.cross(np.asarray(row["point_abs_m"])-center, force)
            work = dt*(float(force@(.5*(velocity+_xyz(body.GetPosDt()))))
                       +float(torque@(.5*(omega+_xyz(body.GetAngVelParent())))))
            self.work_j += work
            if ":friction:" in row["id"]:
                friction_work += work
        self.positive_friction_work_j += max(0., friction_work)
        self.dissipation_j += dt*loss+max(0., -friction_work)
        self.pending = None
        # The final integrated pose is also part of the public evidence. A
        # last-step release must not wait for another before_step that may
        # never happen. Keep the completed interval's force records intact.
        self._refresh_activity(self.step_end_s)

    def sample(self):
        values = self._evaluate(False)
        metrics = self._metrics()
        inside = all(abs(value) <= self.spec["release"][name] for name, value in metrics.items())
        return {"kind": KIND, "active": bool(self.record["active"]),
                "within_support_range": inside,
                "within_working_range": inside or not self.record["active"],
                **metrics, **values, "work_on_bodies_j": self.work_j,
                "dissipated_energy_j": self.dissipation_j, "release_dissipation_j": self.release_dissipation_j,
                "positive_friction_work_j": self.positive_friction_work_j,
                "applied_friction_scale": self.applied_friction_scale,
                "initial_potential_energy_j": self.initial_potential_j,
                "discrete_energy_residual_j": self.work_j+values["potential_energy_j"]-self.initial_potential_j+self.dissipation_j}

    def metadata(self):
        return {"id": self.spec["id"], "representation": "passive_clearance_bearings_and_unilateral_stops_on_original_bodies",
                "ideal_constraints_created": 0, "active": bool(self.record["active"]), "events": list(self.events),
                "integration": "original Chrono system; explicit constitutive loads",
                "friction": "regularized passive Coulomb; no static-stick accuracy claim",
                "release": "irreversible; stored elastic energy becomes recorded release dissipation"}
