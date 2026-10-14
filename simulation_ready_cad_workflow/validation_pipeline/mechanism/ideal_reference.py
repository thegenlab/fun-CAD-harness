from __future__ import annotations

"""Deterministic ideal mechanism references with no simulation dependency."""

from dataclasses import dataclass
import math
from typing import Any, Mapping

try:
    from typing import Protocol
except ImportError:  # Python 3.7 runtime compatibility.
    from typing_extensions import Protocol

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error


IDEAL_REFERENCE_SCHEMA = "ideal_reference_v1"


class IdealReferenceError(MechanismValidationError):
    pass


def _ideal_reference_failure(
    exc: BaseException,
    *,
    capsule_id: str,
    load_case_id: str,
) -> dict[str, Any]:
    carried = getattr(exc, "failure_envelope", None)
    if isinstance(carried, Mapping):
        return dict(carried)
    code = str(getattr(exc, "code", "IDEAL_REFERENCE_INCONCLUSIVE"))
    return make_failure_envelope(
        code,
        FailureOwner.HARNESS_CODE,
        source_component="mechanism.ideal_reference.aggregate",
        message=str(exc),
        evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        details={
            "capsule_id": capsule_id,
            "load_case_id": load_case_id,
        },
    )


class IdealMechanismModel(Protocol):
    model_id: str

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        ...


def _fail(code: str, message: str) -> IdealReferenceError:
    return IdealReferenceError(code, message)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _fail(
            "IDEAL_REFERENCE_INPUT_INVALID",
            f"{label} must be a finite number",
        )
    return float(value)


def _base_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    time_s = _finite(sample.get("time_s"), "sample.time_s")
    phase_id = sample.get("phase_id")
    if not isinstance(phase_id, str):
        raise _fail(
            "IDEAL_REFERENCE_INPUT_INVALID",
            "physical sample phase_id is missing",
        )
    return {
        "time_s": time_s,
        "phase_id": phase_id,
    }


def _speeds(
    samples: list[dict[str, Any]],
    position_field: str,
    velocity_field: str,
) -> None:
    previous_position: float | None = None
    previous_time: float | None = None
    for sample in samples:
        position = float(sample[position_field])
        time_s = float(sample["time_s"])
        if previous_position is None or previous_time is None:
            speed = 0.0
        else:
            elapsed = time_s - previous_time
            speed = (
                0.0
                if elapsed <= 1e-12
                else (position - previous_position) / elapsed
            )
        sample[velocity_field] = speed
        previous_position = position
        previous_time = time_s


@dataclass(frozen=True)
class PlainCoordinateModel:
    model_id: str
    position_field: str
    velocity_field: str
    unit: str

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for sample in samples:
            item = _base_sample(sample)
            coordinate = _finite(
                sample.get(self.position_field),
                self.position_field,
            )
            item.update(
                {
                    self.position_field: coordinate,
                    self.velocity_field: 0.0,
                    "input_coordinate": coordinate,
                    "input_coordinate_unit": self.unit,
                    "output_coordinate": coordinate,
                    "output_coordinate_unit": self.unit,
                }
            )
            output.append(item)
        _speeds(output, self.position_field, self.velocity_field)
        return output, {
            "kind": "single_ideal_coordinate",
            "coordinate_field": self.position_field,
            "domain_check": "finite_coordinate",
            "domain_status": "PASS",
        }


def _dot3(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _sub3(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def _add3(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))


def _scale3(
    value: tuple[float, ...],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(component * factor for component in value)


def _cross3(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm3(value: tuple[float, ...]) -> float:
    return math.sqrt(_dot3(value, value))


def _unit3(value: tuple[float, ...], label: str) -> tuple[float, float, float]:
    length = _norm3(value)
    if length <= 1e-12:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            f"{label} has zero length",
        )
    return _scale3(value, 1.0 / length)


def _unit_quaternion(
    value: tuple[float, ...],
    label: str,
) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1e-12:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            f"{label} has zero length",
        )
    return tuple(component / length for component in value)


def _quaternion_product(
    left: tuple[float, ...],
    right: tuple[float, ...],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quaternion_rotate(
    quaternion: tuple[float, ...],
    vector: tuple[float, ...],
) -> tuple[float, float, float]:
    _, qx, qy, qz = quaternion
    qw = quaternion[0]
    q_vector = (qx, qy, qz)
    twice_cross = _scale3(_cross3(q_vector, vector), 2.0)
    return _add3(
        vector,
        _add3(
            _scale3(twice_cross, qw),
            _cross3(q_vector, twice_cross),
        ),
    )


def _rotation_between(
    source: tuple[float, ...],
    target: tuple[float, ...],
    *,
    fallback_axis: tuple[float, ...],
) -> tuple[float, float, float, float]:
    source_unit = _unit3(source, "ideal pose source direction")
    target_unit = _unit3(target, "ideal pose target direction")
    cosine = max(-1.0, min(1.0, _dot3(source_unit, target_unit)))
    if cosine <= -1.0 + 1e-10:
        axis = _unit3(
            _cross3(source_unit, fallback_axis),
            "ideal pose half-turn axis",
        )
        return (0.0, *axis)
    cross = _cross3(source_unit, target_unit)
    return _unit_quaternion(
        (1.0 + cosine, *cross),
        "ideal pose rotation",
    )


def _rigid_pose_from_segment(
    body: Any,
    *,
    home_start: tuple[float, ...],
    home_end: tuple[float, ...],
    current_start: tuple[float, ...],
    current_end: tuple[float, ...],
    fallback_axis: tuple[float, ...],
) -> dict[str, list[float]]:
    rotation_delta = _rotation_between(
        _sub3(home_end, home_start),
        _sub3(current_end, current_start),
        fallback_axis=fallback_axis,
    )
    center_offset = _sub3(body.center_abs_m, home_start)
    center = _add3(
        current_start,
        _quaternion_rotate(rotation_delta, center_offset),
    )
    rotation = _unit_quaternion(
        _quaternion_product(
            rotation_delta,
            tuple(body.rotation_abs_quat_wxyz),
        ),
        f"{body.body_id} ideal rotation",
    )
    return {
        "position_abs_m": list(center),
        "rotation_abs_quat_wxyz": list(rotation),
    }


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _planar_four_bar_geometry(spec: Any) -> dict[str, Any]:
    if (
        len(spec.bodies) != 4
        or len(spec.revolute_joints) != 4
        or spec.loop_count != 1
    ):
        raise _fail(
            "IDEAL_REFERENCE_UNSUPPORTED",
            "generic compound ideal reference currently requires one four-bar loop",
        )
    joints = {item.joint_id: item for item in spec.revolute_joints}
    input_joint = joints[spec.input_joint_id]
    output_joint = joints[spec.output_joint_id]

    def other_joint(body_id: str, excluded_id: str) -> Any:
        candidates = [
            joint
            for joint in spec.revolute_joints
            if joint.joint_id != excluded_id
            and body_id in {joint.parent_body_id, joint.child_body_id}
        ]
        if len(candidates) != 1:
            raise _fail(
                "IDEAL_REFERENCE_UNSUPPORTED",
                "four-bar input/output bodies must each have one coupler pivot",
            )
        return candidates[0]

    input_coupler_joint = other_joint(
        spec.input_body_id,
        spec.input_joint_id,
    )
    output_coupler_joint = other_joint(
        spec.output_body_id,
        spec.output_joint_id,
    )

    def other_body(joint: Any, body_id: str) -> str:
        return (
            joint.child_body_id
            if joint.parent_body_id == body_id
            else joint.parent_body_id
        )

    connector_a = other_body(input_coupler_joint, spec.input_body_id)
    connector_b = other_body(output_coupler_joint, spec.output_body_id)
    if connector_a != connector_b:
        raise _fail(
            "IDEAL_REFERENCE_UNSUPPORTED",
            "four-bar input and output must share one coupler body",
        )
    axis = tuple(float(value) for value in input_joint.axis_abs)
    if any(_dot3(axis, tuple(joint.axis_abs)) < 0.999 for joint in spec.revolute_joints):
        raise _fail(
            "IDEAL_REFERENCE_UNSUPPORTED",
            "four-bar ideal reference requires consistently oriented parallel axes",
        )
    point_a = tuple(input_joint.origin_abs_m)
    point_b = tuple(input_coupler_joint.origin_abs_m)
    point_c = tuple(output_coupler_joint.origin_abs_m)
    point_d = tuple(output_joint.origin_abs_m)
    crank_home_raw = _sub3(point_b, point_a)
    crank_home = _sub3(
        crank_home_raw,
        _scale3(axis, _dot3(crank_home_raw, axis)),
    )
    crank_length = _norm3(crank_home)
    if crank_length <= 1e-9:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "four-bar input crank has zero projected length",
        )
    basis_x = _scale3(crank_home, 1.0 / crank_length)
    basis_y = _cross3(axis, basis_x)
    basis_y_norm = _norm3(basis_y)
    if basis_y_norm <= 1e-9:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "four-bar plane basis is degenerate",
        )
    basis_y = _scale3(basis_y, 1.0 / basis_y_norm)

    def coordinates(point: tuple[float, ...]) -> tuple[float, float]:
        relative = _sub3(point, point_a)
        axial = _dot3(relative, axis)
        if abs(axial) > 1e-4:
            raise _fail(
                "IDEAL_REFERENCE_UNSUPPORTED",
                "four-bar pivots are not coplanar",
            )
        return _dot3(relative, basis_x), _dot3(relative, basis_y)

    b_home = coordinates(point_b)
    c_home = coordinates(point_c)
    d = coordinates(point_d)
    coupler_length = math.dist(b_home, c_home)
    rocker_length = math.dist(c_home, d)
    if min(coupler_length, rocker_length) <= 1e-9:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "four-bar coupler or rocker length is zero",
        )
    output_min = (
        spec.home_output_angle_rad
        if output_joint.min_angle_rad is None
        else output_joint.min_angle_rad
    )
    output_max = (
        spec.outbound_output_angle_rad
        if output_joint.max_angle_rad is None
        else output_joint.max_angle_rad
    )
    return {
        "axis": axis,
        "origin": point_a,
        "basis_x": basis_x,
        "basis_y": basis_y,
        "point_d": point_d,
        "b_home": b_home,
        "c_home": c_home,
        "d": d,
        "crank_length": crank_length,
        "coupler_length": coupler_length,
        "rocker_length": rocker_length,
        "home_rocker_angle": math.atan2(
            c_home[1] - d[1],
            c_home[0] - d[0],
        ),
        "output_min": float(output_min),
        "output_max": float(output_max),
    }


@dataclass(frozen=True)
class PlanarFourBarModel:
    model_id: str = "planar_four_bar_circle_closure_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .compound_spec import compound_linkage_from_dict

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        spec = compound_linkage_from_dict(family_spec)
        geometry = _planar_four_bar_geometry(spec)
        output: list[dict[str, Any]] = []
        clamped_samples = 0
        for sample in samples:
            item = _base_sample(sample)
            input_angle = _finite(
                sample.get("compound_input_angle_rad"),
                "compound_input_angle_rad",
            )
            b = (
                geometry["crank_length"] * math.cos(input_angle),
                geometry["crank_length"] * math.sin(input_angle),
            )
            dx = geometry["d"][0] - b[0]
            dy = geometry["d"][1] - b[1]
            distance = math.hypot(dx, dy)
            radius_a = geometry["coupler_length"]
            radius_b = geometry["rocker_length"]
            if (
                distance <= 1e-12
                or distance > radius_a + radius_b + 1e-9
                or distance < abs(radius_a - radius_b) - 1e-9
            ):
                raise _fail(
                    "IDEAL_REFERENCE_OUTSIDE_DOMAIN",
                    "four-bar input angle has no real closed-loop solution",
                )
            along = (
                radius_a * radius_a
                - radius_b * radius_b
                + distance * distance
            ) / (2.0 * distance)
            height = math.sqrt(max(radius_a * radius_a - along * along, 0.0))
            base = (
                b[0] + along * dx / distance,
                b[1] + along * dy / distance,
            )
            offset = (-dy * height / distance, dx * height / distance)
            candidates = (
                (base[0] + offset[0], base[1] + offset[1]),
                (base[0] - offset[0], base[1] - offset[1]),
            )
            c = min(
                candidates,
                key=lambda value: (
                    (value[0] - geometry["c_home"][0]) ** 2
                    + (value[1] - geometry["c_home"][1]) ** 2
                ),
            )
            raw_delta = _wrap_angle(
                math.atan2(
                    c[1] - geometry["d"][1],
                    c[0] - geometry["d"][0],
                )
                - geometry["home_rocker_angle"]
            )
            raw_output = spec.home_output_angle_rad + raw_delta
            output_angle = max(
                geometry["output_min"],
                min(geometry["output_max"], raw_output),
            )
            if not math.isclose(output_angle, raw_output, abs_tol=1e-12):
                clamped_samples += 1
            used_delta = output_angle - spec.home_output_angle_rad
            observable_relative = _sub3(
                tuple(spec.output_observable_point_abs_m),
                geometry["point_d"],
            )
            axial = _dot3(observable_relative, geometry["axis"])
            planar_x = _dot3(observable_relative, geometry["basis_x"])
            planar_y = _dot3(observable_relative, geometry["basis_y"])
            rotated = _add3(
                _add3(
                    _scale3(
                        geometry["basis_x"],
                        planar_x * math.cos(used_delta)
                        - planar_y * math.sin(used_delta),
                    ),
                    _scale3(
                        geometry["basis_y"],
                        planar_x * math.sin(used_delta)
                        + planar_y * math.cos(used_delta),
                    ),
                ),
                _scale3(geometry["axis"], axial),
            )
            observable = _add3(geometry["point_d"], rotated)
            item.update(
                {
                    "compound_input_angle_rad": input_angle,
                    "compound_output_angle_rad": output_angle,
                    "generalized_angle_rad": output_angle,
                    "input_coordinate": input_angle,
                    "input_coordinate_unit": "rad",
                    "output_coordinate": output_angle,
                    "output_coordinate_unit": "rad",
                    "observable_point_abs_m": list(observable),
                    "closure_error_m": 0.0,
                    "hard_stop_clamped": output_angle != raw_output,
                }
            )
            output.append(item)
        _speeds(
            output,
            "compound_input_angle_rad",
            "compound_input_speed_rad_s",
        )
        _speeds(
            output,
            "compound_output_angle_rad",
            "compound_output_speed_rad_s",
        )
        for item in output:
            item["generalized_speed_rad_s"] = item[
                "compound_output_speed_rad_s"
            ]
        return output, {
            "kind": "declared_planar_four_bar_circle_closure",
            "link_lengths_m": {
                "crank": geometry["crank_length"],
                "coupler": geometry["coupler_length"],
                "rocker": geometry["rocker_length"],
            },
            "output_domain_rad": [
                geometry["output_min"],
                geometry["output_max"],
            ],
            "hard_stop_clamped_sample_count": clamped_samples,
            "domain_check": "real_circle_intersection_and_declared_output_limits",
            "domain_status": "PASS",
        }


@dataclass(frozen=True)
class RackPinionModel:
    model_id: str = "rack_pinion_closed_form_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .rack_pinion_spec import rack_pinion_from_dict

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        spec = rack_pinion_from_dict(family_spec)
        initial_angle = _finite(
            samples[0].get("pinion_angle_rad"),
            "pinion_angle_rad",
        )
        output: list[dict[str, Any]] = []
        tolerance = max(spec.output_progress_tolerance_m, 1e-9)
        for sample in samples:
            item = _base_sample(sample)
            angle = _finite(
                sample.get("pinion_angle_rad"),
                "pinion_angle_rad",
            )
            progress = (
                spec.home_progress_m
                - spec.pinion_pitch_radius_m * (angle - initial_angle)
            )
            if (
                progress < spec.home_progress_m - tolerance
                or progress > spec.outbound_progress_m + tolerance
            ):
                raise _fail(
                    "IDEAL_REFERENCE_OUTSIDE_DOMAIN",
                    "rack-pinion input angle maps outside declared rack travel",
                )
            progress = max(
                spec.home_progress_m,
                min(spec.outbound_progress_m, progress),
            )
            item.update(
                {
                    "pinion_angle_rad": angle,
                    "rack_progress_m": progress,
                    "generalized_progress_m": progress,
                    "input_coordinate": angle,
                    "input_coordinate_unit": "rad",
                    "output_coordinate": progress,
                    "output_coordinate_unit": "m",
                    "coupling_error_m": 0.0,
                }
            )
            # The ideal branch runs the same commanded-input schedule as the
            # physical branch (only output coordinates are recomputed), so
            # the saved crank-input wrench flows through with the input
            # coordinates for input-trace acceptance predicates.
            if "applied_torque_abs_nm" in sample:
                item["applied_torque_abs_nm"] = list(
                    sample["applied_torque_abs_nm"]
                )
            output.append(item)
        _speeds(output, "rack_progress_m", "rack_speed_m_s")
        for item in output:
            item["generalized_speed_m_s"] = item["rack_speed_m_s"]
        return output, {
            "kind": "closed_form_rack_pinion",
            "equation": "rack_progress = home - pitch_radius * pinion_angle_delta",
            "pitch_radius_m": spec.pinion_pitch_radius_m,
            "domain": {
                "rack_progress_m": [
                    spec.home_progress_m,
                    spec.outbound_progress_m,
                ]
            },
            "domain_status": "PASS",
        }


@dataclass(frozen=True)
class BicycleChainModel:
    model_id: str = "bicycle_chain_signed_pitch_mapping_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .bicycle_chain_spec import bicycle_chain_from_dict

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        if isinstance(family_spec, Mapping):
            package_spec = {
                key: value
                for key, value in family_spec.items()
                if key != "collision_geometry"
            }
        else:
            package_spec = family_spec
        spec = bicycle_chain_from_dict(package_spec)
        input_home = _finite(
            samples[0].get("input_crank_angle_rad"),
            "input_crank_angle_rad",
        )
        output_home = _finite(
            samples[0].get("rear_drive_wheel_angle_rad"),
            "rear_drive_wheel_angle_rad",
        )
        output: list[dict[str, Any]] = []
        for sample in samples:
            item = _base_sample(sample)
            input_angle = _finite(
                sample.get("input_crank_angle_rad"),
                "input_crank_angle_rad",
            )
            output_angle = output_home + spec.signed_output_per_input * (
                input_angle - input_home
            )
            item.update(
                {
                    "input_crank_angle_rad": input_angle,
                    "rear_drive_wheel_angle_rad": output_angle,
                    "generalized_angle_rad": output_angle,
                    "input_coordinate": input_angle,
                    "input_coordinate_unit": "rad",
                    "output_coordinate": output_angle,
                    "output_coordinate_unit": "rad",
                    "chain_coordinate_error_m": 0.0,
                    "chain_engagement_gate_active": True,
                }
            )
            output.append(item)
        _speeds(
            output,
            "input_crank_angle_rad",
            "input_crank_speed_rad_s",
        )
        _speeds(
            output,
            "rear_drive_wheel_angle_rad",
            "rear_drive_wheel_speed_rad_s",
        )
        for item in output:
            item["generalized_speed_rad_s"] = item[
                "rear_drive_wheel_speed_rad_s"
            ]
        return output, {
            "kind": "signed_open_chain_pitch_mapping",
            "equation": (
                "rear_angle - rear_home = "
                "input_pitch_radius / output_pitch_radius * "
                "(crank_angle - crank_home)"
            ),
            "input_pitch_radius_m": spec.input_pitch_radius_m,
            "output_pitch_radius_m": spec.output_pitch_radius_m,
            "signed_output_per_input": spec.signed_output_per_input,
            "contact_or_dynamic_response_claimed": False,
            "domain_status": "PASS",
        }


@dataclass(frozen=True)
class BicycleGroundRollModel:
    model_id: str = "bicycle_ground_roll_no_slip_reference_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .bicycle_ground_roll_spec import (
            bicycle_ground_roll_from_dict,
        )

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        if isinstance(family_spec, Mapping):
            family_spec = {
                key: value
                for key, value in family_spec.items()
                if key != "collision_geometry"
            }
        spec = bicycle_ground_roll_from_dict(family_spec)
        initial_rear = _finite(
            samples[0].get("rear_wheel_angle_rad"),
            "rear_wheel_angle_rad",
        )
        body_by_id = {body.body_id: body for body in spec.bodies}

        def axis_rotation(angle: float) -> tuple[float, float, float, float]:
            half = 0.5 * angle
            sine = math.sin(half)
            return (
                math.cos(half),
                spec.wheel_axis_abs[0] * sine,
                spec.wheel_axis_abs[1] * sine,
                spec.wheel_axis_abs[2] * sine,
            )

        output: list[dict[str, Any]] = []
        for sample in samples:
            item = _base_sample(sample)
            rear_angle = _finite(
                sample.get("rear_wheel_angle_rad"),
                "rear_wheel_angle_rad",
            )
            rear_delta = rear_angle - initial_rear
            travel = spec.rear_wheel_radius_m * rear_delta
            front_angle = travel / spec.front_wheel_radius_m
            translation = _scale3(spec.travel_axis_abs, travel)
            body_poses: dict[str, dict[str, list[float]]] = {}
            for body_id, body in body_by_id.items():
                position = tuple(body.center_abs_m)
                if body_id != spec.ground_body_id:
                    position = _add3(position, translation)
                rotation = tuple(body.rotation_abs_quat_wxyz)
                if body_id == spec.rear_wheel_body_id:
                    rotation = _unit_quaternion(
                        _quaternion_product(
                            axis_rotation(rear_delta), rotation
                        ),
                        "rear wheel ideal rotation",
                    )
                elif body_id == spec.front_wheel_body_id:
                    rotation = _unit_quaternion(
                        _quaternion_product(
                            axis_rotation(front_angle), rotation
                        ),
                        "front wheel ideal rotation",
                    )
                body_poses[body_id] = {
                    "position_abs_m": list(position),
                    "rotation_abs_quat_wxyz": list(rotation),
                }
            item.update(
                {
                    "rear_wheel_angle_rad": rear_angle,
                    "front_wheel_angle_rad": front_angle,
                    "chassis_position_m": travel,
                    "generalized_progress_m": travel,
                    "input_coordinate": rear_angle,
                    "input_coordinate_unit": "rad",
                    "output_coordinate": front_angle,
                    "output_coordinate_unit": "rad",
                    "rolling_slip_ratio": 0.0,
                    "rear_axle_drift_m": 0.0,
                    "front_axle_drift_m": 0.0,
                    "chassis_guide_transverse_drift_m": 0.0,
                    "max_off_axis_rotation_rad": 0.0,
                    "max_penetration_m": 0.0,
                    "max_ground_separation_m": 0.0,
                    "rear_ground_contact_count": 0,
                    "front_ground_contact_count": 0,
                    "expected_contact_count": 0,
                    "body_poses": body_poses,
                }
            )
            output.append(item)
        _speeds(output, "rear_wheel_angle_rad", "rear_wheel_speed_rad_s")
        _speeds(output, "front_wheel_angle_rad", "front_wheel_speed_rad_s")
        _speeds(output, "chassis_position_m", "chassis_speed_m_s")
        return output, {
            "kind": "straight_line_no_slip_kinematic_reference",
            "equations": [
                "travel = rear_radius * rear_angle_delta",
                "front_angle = travel / front_radius",
            ],
            "physical_contact_evidence_claimed": False,
            "physical_no_slip_constraint_claimed": False,
            "body_poses_from_ideal_geometry": True,
            "domain_status": "PASS",
        }


def _clock_affine_relations(
    spec: Any,
    source_body_id: str,
    *,
    setting_clutch_engaged: bool,
) -> dict[str, tuple[float, float]]:
    relations: dict[str, tuple[float, float]] = {
        source_body_id: (1.0, 0.0)
    }
    pending = [source_body_id]
    while pending:
        current = pending.pop(0)
        current_scale, current_offset = relations[current]
        for mesh in spec.gear_meshes:
            if (
                mesh.driver_body_id == current
                and mesh.driven_body_id not in relations
            ):
                relations[mesh.driven_body_id] = (
                    -mesh.ratio * current_scale,
                    mesh.phase_offset_rad - mesh.ratio * current_offset,
                )
                pending.append(mesh.driven_body_id)
            elif (
                mesh.driven_body_id == current
                and mesh.driver_body_id not in relations
            ):
                relations[mesh.driver_body_id] = (
                    -current_scale / mesh.ratio,
                    (mesh.phase_offset_rad - current_offset) / mesh.ratio,
                )
                pending.append(mesh.driver_body_id)
        for transfer in spec.shaft_transfers:
            if (
                transfer.driver_body_id == current
                and transfer.driven_body_id not in relations
            ):
                relations[transfer.driven_body_id] = (
                    transfer.ratio * current_scale,
                    transfer.ratio * current_offset,
                )
                pending.append(transfer.driven_body_id)
            elif (
                transfer.driven_body_id == current
                and transfer.driver_body_id not in relations
            ):
                relations[transfer.driver_body_id] = (
                    current_scale / transfer.ratio,
                    current_offset / transfer.ratio,
                )
                pending.append(transfer.driver_body_id)
        clutch = spec.setting_clutch
        if setting_clutch_engaged and clutch is not None:
            if (
                clutch.input_body_id == current
                and clutch.driven_body_id not in relations
            ):
                relations[clutch.driven_body_id] = (
                    clutch.ratio * current_scale,
                    clutch.ratio * current_offset,
                )
                pending.append(clutch.driven_body_id)
            elif (
                clutch.driven_body_id == current
                and clutch.input_body_id not in relations
            ):
                relations[clutch.input_body_id] = (
                    current_scale / clutch.ratio,
                    current_offset / clutch.ratio,
                )
                pending.append(clutch.input_body_id)
    return relations


def _clock_orientation(spec: Any) -> float:
    relations = _clock_affine_relations(
        spec,
        spec.input_body_id,
        setting_clutch_engaged=False,
    )
    output = relations.get(spec.output_body_id)
    if output is None or abs(output[0]) <= 1e-12:
        raise _fail(
            "IDEAL_REFERENCE_TOPOLOGY_UNRESOLVED",
            "clock input has no ideal path to its declared output",
        )
    return math.copysign(1.0, output[0])


@dataclass(frozen=True)
class ClockGearModel:
    model_id: str = "spur_gear_clock_affine_graph_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        mechanism_kind = (
            family_spec.get("mechanism_kind")
            if isinstance(family_spec, Mapping)
            else None
        )
        if mechanism_kind in {
            "crown_bevel_dual_output",
            "multi_output_shaft_coupling",
        }:
            from .multi_output_shaft_spec import (
                multi_output_shaft_from_dict,
            )

            spec = multi_output_shaft_from_dict(family_spec)
            is_multi_output = True
        else:
            from .clock_gear_train_spec import clock_gear_train_from_dict

            spec = clock_gear_train_from_dict(family_spec)
            is_multi_output = False
        orientation = _clock_orientation(spec)
        output: list[dict[str, Any]] = []
        modes: set[str] = set()
        for sample in samples:
            item = _base_sample(sample)
            source_body = sample.get(
                "active_input_body_id",
                spec.input_body_id,
            )
            if not isinstance(source_body, str):
                raise _fail(
                    "IDEAL_REFERENCE_INPUT_INVALID",
                    "clock active input body is missing",
                )
            shaft_angles = sample.get("shaft_angles_rad")
            if not isinstance(shaft_angles, Mapping):
                raise _fail(
                    "IDEAL_REFERENCE_INPUT_INVALID",
                    "clock trace does not expose input shaft angles",
                )
            input_angle = _finite(
                shaft_angles.get(source_body),
                f"shaft_angles_rad.{source_body}",
            )
            clutch_engaged = sample.get("setting_clutch_engaged") is True
            if spec.alarm_isolated:
                # The timekeeping and alarm trains intentionally have no
                # mechanical path between them.  Evaluate each declared
                # independent input coordinate through its own affine graph;
                # never invent a link merely to make every output reachable
                # from the currently active actuator.
                timekeeping_source = (
                    spec.setting_input_body_id
                    if clutch_engaged
                    else spec.input_body_id
                )
                independent_sources = (
                    timekeeping_source,
                    spec.effective_alarm_input_body_id,
                )
                ideal_angles: dict[str, float] = {}
                for independent_source in independent_sources:
                    independent_angle = _finite(
                        shaft_angles.get(independent_source),
                        f"shaft_angles_rad.{independent_source}",
                    )
                    relations = _clock_affine_relations(
                        spec,
                        independent_source,
                        setting_clutch_engaged=(
                            clutch_engaged
                            and independent_source == timekeeping_source
                        ),
                    )
                    component_angles = {
                        body_id: scale * independent_angle + offset
                        for body_id, (scale, offset) in relations.items()
                    }
                    overlap = set(ideal_angles) & set(component_angles)
                    if overlap:
                        raise _fail(
                            "IDEAL_REFERENCE_TOPOLOGY_UNRESOLVED",
                            "declared independent clock inputs overlap at: "
                            + ", ".join(sorted(overlap)),
                        )
                    ideal_angles.update(component_angles)
            else:
                relations = _clock_affine_relations(
                    spec,
                    source_body,
                    setting_clutch_engaged=clutch_engaged,
                )
                ideal_angles = {
                    body_id: scale * input_angle + offset
                    for body_id, (scale, offset) in relations.items()
                }
            for follower in spec.cam_followers:
                cam_angle = ideal_angles.get(follower.cam_body_id)
                if cam_angle is not None:
                    ideal_angles[follower.follower_body_id] = (
                        follower.center_angle_rad
                        + follower.amplitude_rad
                        * math.sin(follower.lobe_count * cam_angle)
                    )
            required = (
                tuple(spec.output_body_ids)
                if is_multi_output
                else (
                    spec.output_body_id,
                    spec.hour_body_id,
                    spec.second_body_id,
                    spec.alarm_body_id,
                )
            )
            missing = [body_id for body_id in required if body_id not in ideal_angles]
            if missing:
                raise _fail(
                    "IDEAL_REFERENCE_TOPOLOGY_UNRESOLVED",
                    "clock input mode cannot reach: " + ", ".join(missing),
                )
            clock_output = orientation * ideal_angles[spec.output_body_id]
            item.update(
                {
                    "active_input_body_id": source_body,
                    "setting_clutch_engaged": clutch_engaged,
                    "input_coordinate": input_angle,
                    "input_coordinate_unit": "rad",
                    "output_coordinate": clock_output,
                    "output_coordinate_unit": "rad",
                    "generalized_angle_rad": clock_output,
                    "multi_output_primary_angle_rad": clock_output,
                    "clock_output_angle_rad": clock_output,
                    "hour_hand_angle_rad": (
                        orientation * ideal_angles[spec.hour_body_id]
                    ),
                    "second_hand_angle_rad": (
                        orientation * ideal_angles[spec.second_body_id]
                    ),
                    "alarm_angle_rad": ideal_angles[spec.alarm_body_id],
                    "shaft_angles_rad": ideal_angles,
                    "mesh_errors_rad": {
                        mesh.mesh_id: 0.0 for mesh in spec.gear_meshes
                    },
                    "cam_follower_errors_rad": {
                        follower.follower_id: 0.0
                        for follower in spec.cam_followers
                    },
                    "setting_clutch_error_rad": 0.0 if clutch_engaged else None,
                }
            )
            if is_multi_output:
                item.update(
                    {
                        f"{shaft.body_id}_angle_rad": ideal_angles[
                            shaft.body_id
                        ]
                        for shaft in spec.shafts
                    }
                )
            output.append(item)
            modes.add(
                "manual_setting" if clutch_engaged else str(source_body)
            )
        for position, velocity in (
            ("generalized_angle_rad", "generalized_speed_rad_s"),
            (
                "multi_output_primary_angle_rad",
                "multi_output_primary_speed_rad_s",
            ),
            ("clock_output_angle_rad", "clock_output_speed_rad_s"),
            ("hour_hand_angle_rad", "hour_hand_speed_rad_s"),
            ("second_hand_angle_rad", "second_hand_speed_rad_s"),
            ("alarm_angle_rad", "alarm_speed_rad_s"),
        ):
            _speeds(output, position, velocity)
        if is_multi_output:
            for shaft in spec.shafts:
                _speeds(
                    output,
                    f"{shaft.body_id}_angle_rad",
                    f"{shaft.body_id}_speed_rad_s",
                )
        return output, {
            "kind": (
                "affine_multi_output_shaft_graph"
                if is_multi_output
                else "affine_gear_graph_with_analytic_cam"
            ),
            "gear_mesh_count": len(spec.gear_meshes),
            "shaft_transfer_count": len(spec.shaft_transfers),
            "cam_follower_count": len(spec.cam_followers),
            "working_modes": sorted(modes),
            "domain_check": (
                "all_required_outputs_reachable_from_declared_independent_inputs"
                if spec.alarm_isolated
                else "all_required_outputs_reachable_from_active_input"
            ),
            "domain_status": "PASS",
        }


@dataclass(frozen=True)
class MultiOutputShaftModel(ClockGearModel):
    model_id: str = "multi_output_shaft_affine_graph_v1"


@dataclass(frozen=True)
class DetentSliderModel:
    """Input-driven ideal latch with discrete released detent states."""

    model_id: str = "prismatic_detent_latch_discrete_state_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .detent_slider_spec import prismatic_detent_latch_from_dict

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        spec = prismatic_detent_latch_from_dict(family_spec)
        blade_joint = spec.blade_prismatic_joint
        latch_joint = spec.latch_press_joint
        blade_axis = blade_joint.axis_abs
        press_axis = latch_joint.axis_abs
        output: list[dict[str, Any]] = []

        for sample in samples:
            item = _base_sample(sample)
            input_position = _finite(
                sample.get("thumb_latch_position_m"),
                "thumb_latch_position_m",
            )
            press_depth = _finite(
                sample.get("thumb_press_depth_m"),
                "thumb_press_depth_m",
            )
            clamped_input = max(
                blade_joint.minimum_position_m,
                min(blade_joint.maximum_position_m, input_position),
            )
            released = press_depth < spec.detent_interface.release_press_m
            if released:
                ideal_position = float(
                    min(
                        spec.detent_interface.positions_m,
                        key=lambda value: abs(float(value) - clamped_input),
                    )
                )
                engaged = True
                locked_position: float | None = ideal_position
            else:
                ideal_position = clamped_input
                engaged = False
                locked_position = None

            body_poses: dict[str, dict[str, list[float]]] = {}
            for body in spec.bodies:
                translation = (0.0, 0.0, 0.0)
                if body.body_id in {
                    spec.output_body_id,
                    spec.input_body_id,
                }:
                    translation = _scale3(blade_axis, ideal_position)
                if body.body_id == spec.input_body_id:
                    translation = _add3(
                        translation,
                        _scale3(press_axis, press_depth),
                    )
                body_poses[body.body_id] = {
                    "position_abs_m": list(
                        _add3(body.center_abs_m, translation)
                    ),
                    "rotation_abs_quat_wxyz": list(
                        body.rotation_abs_quat_wxyz
                    ),
                }

            item.update(
                {
                    "thumb_latch_position_m": input_position,
                    "thumb_press_depth_m": press_depth,
                    "input_coordinate": input_position,
                    "input_coordinate_unit": "m",
                    "output_coordinate": ideal_position,
                    "output_coordinate_unit": "m",
                    "generalized_progress_m": ideal_position,
                    "lateral_shift_m": 0.0,
                    "blade_carriage_position_m": ideal_position,
                    "blade_carriage_speed_m_s": 0.0,
                    "detent_engaged": engaged,
                    "locked_detent_position_m": locked_position,
                    "body_poses": body_poses,
                    "ideal_pose_source": (
                        "declared_prismatic_axes_and_discrete_detent_positions"
                    ),
                }
            )
            output.append(item)
        _speeds(
            output,
            "blade_carriage_position_m",
            "blade_carriage_speed_m_s",
        )
        return output, {
            "kind": "declared_prismatic_detent_state_machine",
            "input_field": "thumb_latch_position_m",
            "press_field": "thumb_press_depth_m",
            "detent_positions_m": list(spec.detent_interface.positions_m),
            "release_press_m": spec.detent_interface.release_press_m,
            "body_pose_ids": [body.body_id for body in spec.bodies],
            "body_poses_from_ideal_geometry": True,
            "domain_status": "PASS",
        }


@dataclass(frozen=True)
class ClickPenModel:
    """Ideal two-state click cycle driven only by button displacement."""

    model_id: str = "click_pen_two_state_indexing_cam_v1"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .click_pen_spec import click_pen_cam_latch_from_dict

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        spec = click_pen_cam_latch_from_dict(family_spec)
        target = spec.click_state_machine.first_click_target_m
        button_max = spec.button_prismatic_joint.maximum_position_m
        cam_press_axis = tuple(
            -item for item in spec.cam_cylindrical_joint.axis_abs
        )
        output: list[dict[str, Any]] = []
        for sample in samples:
            item = _base_sample(sample)
            phase_id = str(sample.get("phase_id") or "")
            button_depth = max(
                spec.button_prismatic_joint.minimum_position_m,
                min(
                    button_max,
                    _finite(
                        sample.get("button_press_depth_m"),
                        "button_press_depth_m",
                    ),
                ),
            )
            click_index = 0
            if phase_id.startswith("click_"):
                try:
                    click_index = int(phase_id.split("_", 2)[1])
                except (ValueError, IndexError):
                    click_index = 0
            previous_extended = click_index > 1 and (click_index - 1) % 2 == 1
            release = phase_id.endswith("_release")
            if click_index == 0:
                ideal_refill = 0.0
                completed_clicks = 0
            elif release:
                ideal_refill = target if click_index % 2 == 1 else 0.0
                completed_clicks = click_index
            elif previous_extended:
                ideal_refill = target
                completed_clicks = click_index - 1
            else:
                ideal_refill = min(
                    target,
                    target * button_depth / max(button_max, 1.0e-12),
                )
                completed_clicks = click_index - 1
            press_fraction = min(
                1.0,
                max(0.0, button_depth / max(button_max, 1.0e-12)),
            )
            cam_index_angle = (
                completed_clicks
                + (0.0 if release or click_index == 0 else press_fraction)
            ) * spec.cam_cylindrical_joint.index_angle_rad
            half = 0.5 * cam_index_angle
            axis = spec.cam_cylindrical_joint.axis_abs
            cam_delta = (
                math.cos(half),
                axis[0] * math.sin(half),
                axis[1] * math.sin(half),
                axis[2] * math.sin(half),
            )
            body_poses: dict[str, dict[str, list[float]]] = {}
            for body in spec.bodies:
                translation = (0.0, 0.0, 0.0)
                rotation = tuple(body.rotation_abs_quat_wxyz)
                if body.body_id == spec.input_body_id:
                    translation = _scale3(
                        spec.button_prismatic_joint.axis_abs,
                        button_depth,
                    )
                elif body.body_id == spec.output_body_id:
                    translation = _scale3(
                        spec.refill_prismatic_joint.axis_abs,
                        ideal_refill,
                    )
                elif body.body_id == spec.cam_body_id:
                    translation = _scale3(cam_press_axis, ideal_refill)
                    rotation = _unit_quaternion(
                        _quaternion_product(cam_delta, rotation),
                        "click pen ideal cam rotation",
                    )
                body_poses[body.body_id] = {
                    "position_abs_m": list(
                        _add3(body.center_abs_m, translation)
                    ),
                    "rotation_abs_quat_wxyz": list(rotation),
                }
            extended = (
                ideal_refill
                >= target - spec.click_state_machine.latch_contact_tolerance_m
                and release
                and click_index % 2 == 1
            )
            item.update(
                {
                    "button_press_depth_m": button_depth,
                    "button_speed_m_s": 0.0,
                    "input_coordinate": button_depth,
                    "input_coordinate_unit": "m",
                    "output_coordinate": ideal_refill,
                    "output_coordinate_unit": "m",
                    "generalized_progress_m": ideal_refill,
                    "lateral_shift_m": 0.0,
                    "refill_extension_m": ideal_refill,
                    "refill_speed_m_s": 0.0,
                    "cam_axial_position_m": ideal_refill,
                    "cam_index_angle_rad": cam_index_angle,
                    "cam_index_count": (
                        cam_index_angle
                        / spec.cam_cylindrical_joint.index_angle_rad
                    ),
                    "extended_latched": extended,
                    "latch_contact_count": 1 if extended else 0,
                    "body_poses": body_poses,
                    "ideal_pose_source": (
                        "declared_button_refill_axes_and_two_state_cam_index"
                    ),
                }
            )
            output.append(item)
        _speeds(output, "button_press_depth_m", "button_speed_m_s")
        _speeds(output, "refill_extension_m", "refill_speed_m_s")
        return output, {
            "kind": "declared_two_state_click_pen_cam",
            "input_field": "button_press_depth_m",
            "output_field": "refill_extension_m",
            "first_click_target_m": target,
            "index_angle_rad": spec.cam_cylindrical_joint.index_angle_rad,
            "body_pose_ids": [body.body_id for body in spec.bodies],
            "body_poses_from_ideal_geometry": True,
            "domain_status": "PASS",
        }


def _scissor_lift_geometry(spec: Any) -> dict[str, Any]:
    """Resolve the driven scissor arm from declared joints and anchors."""

    fixed_body_id = spec.fixed_body_id
    output_body_id = spec.output_body_id
    rod_body_id = spec.actuator.rod_body_id

    def connects(joint: Any, left: str, right: str) -> bool:
        return {
            joint.parent_body_id,
            joint.child_body_id,
        } == {left, right}

    candidates: list[tuple[Any, Any, Any, str]] = []
    for track in spec.prismatic_joints:
        if fixed_body_id not in {
            track.parent_body_id,
            track.child_body_id,
        }:
            continue
        arm_body_id = (
            track.child_body_id
            if track.parent_body_id == fixed_body_id
            else track.parent_body_id
        )
        output_pins = [
            joint
            for joint in spec.revolute_joints
            if connects(joint, arm_body_id, output_body_id)
        ]
        rod_pins = [
            joint
            for joint in spec.revolute_joints
            if connects(joint, arm_body_id, rod_body_id)
        ]
        if len(output_pins) == 1 and len(rod_pins) == 1:
            candidates.append(
                (track, output_pins[0], rod_pins[0], arm_body_id)
            )
    if len(candidates) != 1:
        raise _fail(
            "IDEAL_REFERENCE_UNSUPPORTED",
            "scissor-lift ideal reference requires one driven arm connecting "
            "the fixed-base track, output platform, and cylinder rod",
        )

    track, output_pin, rod_pin, arm_body_id = candidates[0]
    bottom_point = tuple(float(value) for value in track.origin_abs_m)
    top_point = tuple(float(value) for value in output_pin.origin_abs_m)
    rod_point = tuple(float(value) for value in spec.actuator.rod_anchor_abs_m)
    rod_pin_point = tuple(float(value) for value in rod_pin.origin_abs_m)
    barrel_point = tuple(
        float(value) for value in spec.actuator.barrel_anchor_abs_m
    )
    slide_axis = tuple(float(value) for value in track.axis_abs)
    output_axis = tuple(float(value) for value in spec.output_axis_abs)
    arm_vector = _sub3(top_point, bottom_point)
    arm_length = _norm3(arm_vector)
    if arm_length <= 1e-9:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "scissor-lift driven arm has zero length",
        )
    if _norm3(_sub3(rod_point, rod_pin_point)) > 1e-4:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "scissor-lift actuator rod anchor does not match its declared clevis",
        )
    rod_fraction = _dot3(
        _sub3(rod_point, bottom_point),
        arm_vector,
    ) / (arm_length * arm_length)
    reconstructed_rod = _add3(
        bottom_point,
        _scale3(arm_vector, rod_fraction),
    )
    if (
        rod_fraction < -1e-9
        or rod_fraction > 1.0 + 1e-9
        or _norm3(_sub3(rod_point, reconstructed_rod)) > 1e-4
    ):
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "scissor-lift cylinder clevis must lie on the declared driven arm",
        )
    home_slide_projection = _dot3(arm_vector, slide_axis)
    if abs(home_slide_projection) <= 1e-9:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "scissor-lift home configuration cannot be perpendicular to its track",
        )
    return {
        "arm_body_id": arm_body_id,
        "bottom_point": bottom_point,
        "top_point": top_point,
        "barrel_point": barrel_point,
        "slide_axis": slide_axis,
        "output_axis": output_axis,
        "arm_vector": arm_vector,
        "arm_length": arm_length,
        "rod_fraction": rod_fraction,
        "home_progress_m": float(spec.home_progress_m),
        "slide_branch_sign": (
            1.0 if home_slide_projection > 0.0 else -1.0
        ),
    }


def _scissor_cylinder_length_m(
    geometry: Mapping[str, Any],
    progress_m: float,
) -> float:
    """Close the rigid scissor loop and return cylinder pin-to-pin length."""

    delta_progress = progress_m - float(geometry["home_progress_m"])
    top_point = _add3(
        geometry["top_point"],
        _scale3(geometry["output_axis"], delta_progress),
    )
    top_from_home_bottom = _sub3(top_point, geometry["bottom_point"])
    slide_projection = _dot3(
        top_from_home_bottom,
        geometry["slide_axis"],
    )
    perpendicular_length_sq = (
        _dot3(top_from_home_bottom, top_from_home_bottom)
        - slide_projection * slide_projection
    )
    closure_sq = (
        float(geometry["arm_length"]) ** 2 - perpendicular_length_sq
    )
    if closure_sq < -1e-9:
        raise _fail(
            "IDEAL_REFERENCE_OUTSIDE_DOMAIN",
            "scissor-lift platform progress has no real rigid-loop closure",
        )
    slide_distance = slide_projection - float(
        geometry["slide_branch_sign"]
    ) * math.sqrt(max(closure_sq, 0.0))
    bottom_point = _add3(
        geometry["bottom_point"],
        _scale3(geometry["slide_axis"], slide_distance),
    )
    rod_point = _add3(
        bottom_point,
        _scale3(
            _sub3(top_point, bottom_point),
            float(geometry["rod_fraction"]),
        ),
    )
    return _norm3(_sub3(rod_point, geometry["barrel_point"]))


def _scissor_progress_for_cylinder_length_m(
    geometry: Mapping[str, Any],
    *,
    cylinder_length_m: float,
    home_progress_m: float,
    outbound_progress_m: float,
    home_length_m: float,
    outbound_length_m: float,
) -> float:
    """Invert a monotonic declared scissor geometry map by bisection."""

    increasing = outbound_length_m > home_length_m
    lower = home_progress_m
    upper = outbound_progress_m
    for _iteration in range(64):
        midpoint = (lower + upper) * 0.5
        midpoint_length = _scissor_cylinder_length_m(geometry, midpoint)
        if (midpoint_length < cylinder_length_m) == increasing:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) * 0.5


def _scissor_body_poses(
    spec: Any,
    geometry: Mapping[str, Any],
    progress_m: float,
) -> dict[str, dict[str, list[float]]]:
    """Resolve every scissor-lift rigid body from the ideal loop closure."""

    bodies = {body.body_id: body for body in spec.bodies}
    output_delta = _scale3(
        tuple(spec.output_axis_abs),
        progress_m - float(spec.home_progress_m),
    )
    output_body = bodies[spec.output_body_id]
    fixed_body = bodies[spec.fixed_body_id]
    driven_arm_id = str(geometry["arm_body_id"])

    def connecting(
        joints: Any,
        left: str,
        right: str,
        label: str,
    ) -> Any:
        matches = [
            joint
            for joint in joints
            if {joint.parent_body_id, joint.child_body_id} == {left, right}
        ]
        if len(matches) != 1:
            raise _fail(
                "IDEAL_REFERENCE_UNSUPPORTED",
                f"scissor-lift ideal poses require exactly one {label}",
            )
        return matches[0]

    driven_track = connecting(
        spec.prismatic_joints,
        spec.fixed_body_id,
        driven_arm_id,
        "driven base track",
    )
    driven_top_pin = connecting(
        spec.revolute_joints,
        spec.output_body_id,
        driven_arm_id,
        "driven platform pin",
    )
    driven_home_bottom = tuple(driven_track.origin_abs_m)
    driven_home_top = tuple(driven_top_pin.origin_abs_m)
    driven_current_top = _add3(driven_home_top, output_delta)
    top_from_home_bottom = _sub3(
        driven_current_top,
        driven_home_bottom,
    )
    driven_slide_projection = _dot3(
        top_from_home_bottom,
        tuple(driven_track.axis_abs),
    )
    driven_perpendicular_sq = (
        _dot3(top_from_home_bottom, top_from_home_bottom)
        - driven_slide_projection * driven_slide_projection
    )
    driven_closure_sq = (
        float(geometry["arm_length"]) ** 2 - driven_perpendicular_sq
    )
    if driven_closure_sq < -1e-9:
        raise _fail(
            "IDEAL_REFERENCE_OUTSIDE_DOMAIN",
            "scissor-lift driven arm has no rigid pose at this progress",
        )
    driven_slide = driven_slide_projection - float(
        geometry["slide_branch_sign"]
    ) * math.sqrt(max(driven_closure_sq, 0.0))
    driven_current_bottom = _add3(
        driven_home_bottom,
        _scale3(tuple(driven_track.axis_abs), driven_slide),
    )

    other_arm_candidates = [
        body_id
        for body_id in bodies
        if body_id
        not in {
            spec.fixed_body_id,
            spec.output_body_id,
            driven_arm_id,
            spec.actuator.barrel_body_id,
            spec.actuator.rod_body_id,
        }
    ]
    if len(other_arm_candidates) != 1:
        raise _fail(
            "IDEAL_REFERENCE_UNSUPPORTED",
            "scissor-lift ideal poses require exactly two scissor arms",
        )
    other_arm_id = other_arm_candidates[0]
    other_bottom_pin = connecting(
        spec.revolute_joints,
        spec.fixed_body_id,
        other_arm_id,
        "opposite base pin",
    )
    other_top_track = connecting(
        spec.prismatic_joints,
        spec.output_body_id,
        other_arm_id,
        "opposite platform track",
    )
    other_home_bottom = tuple(other_bottom_pin.origin_abs_m)
    other_home_top = tuple(other_top_track.origin_abs_m)
    other_arm_vector = _sub3(other_home_top, other_home_bottom)
    other_arm_length = _norm3(other_arm_vector)
    other_current_track_origin = _add3(other_home_top, output_delta)
    other_track_axis = tuple(other_top_track.axis_abs)
    other_track_from_bottom = _sub3(
        other_current_track_origin,
        other_home_bottom,
    )
    other_projection = _dot3(
        other_track_from_bottom,
        other_track_axis,
    )
    other_perpendicular_sq = (
        _dot3(other_track_from_bottom, other_track_from_bottom)
        - other_projection * other_projection
    )
    other_closure_sq = other_arm_length**2 - other_perpendicular_sq
    if other_closure_sq < -1e-9:
        raise _fail(
            "IDEAL_REFERENCE_OUTSIDE_DOMAIN",
            "scissor-lift opposite arm has no rigid pose at this progress",
        )
    other_branch_sign = 1.0 if other_projection >= 0.0 else -1.0
    other_slide = (
        -other_projection
        + other_branch_sign * math.sqrt(max(other_closure_sq, 0.0))
    )
    other_current_top = _add3(
        other_current_track_origin,
        _scale3(other_track_axis, other_slide),
    )
    other_current_bottom = other_home_bottom

    hinge_axis = tuple(other_bottom_pin.axis_abs)
    poses = {
        spec.fixed_body_id: {
            "position_abs_m": list(fixed_body.center_abs_m),
            "rotation_abs_quat_wxyz": list(
                fixed_body.rotation_abs_quat_wxyz
            ),
        },
        spec.output_body_id: {
            "position_abs_m": list(
                _add3(tuple(output_body.center_abs_m), output_delta)
            ),
            "rotation_abs_quat_wxyz": list(
                output_body.rotation_abs_quat_wxyz
            ),
        },
        driven_arm_id: _rigid_pose_from_segment(
            bodies[driven_arm_id],
            home_start=driven_home_bottom,
            home_end=driven_home_top,
            current_start=driven_current_bottom,
            current_end=driven_current_top,
            fallback_axis=hinge_axis,
        ),
        other_arm_id: _rigid_pose_from_segment(
            bodies[other_arm_id],
            home_start=other_home_bottom,
            home_end=other_home_top,
            current_start=other_current_bottom,
            current_end=other_current_top,
            fallback_axis=hinge_axis,
        ),
    }

    center_pin = connecting(
        spec.revolute_joints,
        driven_arm_id,
        other_arm_id,
        "scissor center pin",
    )

    def transformed_point(
        body_id: str,
        *,
        home_start: tuple[float, ...],
        home_end: tuple[float, ...],
        current_start: tuple[float, ...],
        current_end: tuple[float, ...],
        point: tuple[float, ...],
    ) -> tuple[float, float, float]:
        rotation_delta = _rotation_between(
            _sub3(home_end, home_start),
            _sub3(current_end, current_start),
            fallback_axis=hinge_axis,
        )
        return _add3(
            current_start,
            _quaternion_rotate(
                rotation_delta,
                _sub3(point, home_start),
            ),
        )

    driven_center_pin = transformed_point(
        driven_arm_id,
        home_start=driven_home_bottom,
        home_end=driven_home_top,
        current_start=driven_current_bottom,
        current_end=driven_current_top,
        point=tuple(center_pin.origin_abs_m),
    )
    other_center_pin = transformed_point(
        other_arm_id,
        home_start=other_home_bottom,
        home_end=other_home_top,
        current_start=other_current_bottom,
        current_end=other_current_top,
        point=tuple(center_pin.origin_abs_m),
    )
    if _norm3(_sub3(driven_center_pin, other_center_pin)) > 1e-5:
        raise _fail(
            "IDEAL_REFERENCE_DOMAIN_INVALID",
            "scissor-lift arm poses do not close at the center pin",
        )

    rod_pin = connecting(
        spec.revolute_joints,
        driven_arm_id,
        spec.actuator.rod_body_id,
        "cylinder rod clevis",
    )
    current_rod_anchor = transformed_point(
        driven_arm_id,
        home_start=driven_home_bottom,
        home_end=driven_home_top,
        current_start=driven_current_bottom,
        current_end=driven_current_top,
        point=tuple(rod_pin.origin_abs_m),
    )
    home_barrel_anchor = tuple(spec.actuator.barrel_anchor_abs_m)
    home_rod_anchor = tuple(spec.actuator.rod_anchor_abs_m)
    cylinder_rotation = _rotation_between(
        _sub3(home_rod_anchor, home_barrel_anchor),
        _sub3(current_rod_anchor, home_barrel_anchor),
        fallback_axis=hinge_axis,
    )
    for body_id, home_reference, current_reference in (
        (
            spec.actuator.barrel_body_id,
            home_barrel_anchor,
            home_barrel_anchor,
        ),
        (
            spec.actuator.rod_body_id,
            home_rod_anchor,
            current_rod_anchor,
        ),
    ):
        body = bodies[body_id]
        center = _add3(
            current_reference,
            _quaternion_rotate(
                cylinder_rotation,
                _sub3(tuple(body.center_abs_m), home_reference),
            ),
        )
        rotation = _unit_quaternion(
            _quaternion_product(
                cylinder_rotation,
                tuple(body.rotation_abs_quat_wxyz),
            ),
            f"{body_id} ideal rotation",
        )
        poses[body_id] = {
            "position_abs_m": list(center),
            "rotation_abs_quat_wxyz": list(rotation),
        }
    return poses


@dataclass(frozen=True)
class ScissorLiftModel:
    model_id: str = "scissor_lift_declared_linkage_closure_v2"

    def evaluate_samples(
        self,
        plan: MechanismPlan,
        samples: list[Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        from .scissor_lift_spec import scissor_lift_from_dict

        family_spec = (
            plan.adapter_spec.get("family_spec")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        spec = scissor_lift_from_dict(family_spec)
        geometry = _scissor_lift_geometry(spec)
        home_length = _scissor_cylinder_length_m(
            geometry,
            spec.home_progress_m,
        )
        outbound_length = _scissor_cylinder_length_m(
            geometry,
            spec.outbound_progress_m,
        )
        span = outbound_length - home_length
        if abs(span) <= 1e-9:
            raise _fail(
                "IDEAL_REFERENCE_DOMAIN_INVALID",
                "scissor-lift cylinder reference span is zero",
            )
        progress_span = spec.outbound_progress_m - spec.home_progress_m
        previous_length = home_length
        increasing = span > 0.0
        for index in range(1, 33):
            progress = (
                spec.home_progress_m
                + progress_span * index / 32.0
            )
            current_length = _scissor_cylinder_length_m(geometry, progress)
            if (current_length <= previous_length + 1e-10) == increasing:
                raise _fail(
                    "IDEAL_REFERENCE_DOMAIN_INVALID",
                    "scissor-lift cylinder length is not monotonic across "
                    "the declared output range",
                )
            previous_length = current_length
        tolerance = max(spec.output_progress_tolerance_m, 1e-6)
        length_margin = max(abs(span) * 0.05, 1e-6)
        minimum_length = min(home_length, outbound_length)
        maximum_length = max(home_length, outbound_length)
        output: list[dict[str, Any]] = []
        initial_height = _finite(
            samples[0].get("platform_height_m"),
            "platform_height_m",
        )
        for sample in samples:
            item = _base_sample(sample)
            length = _finite(
                sample.get("cylinder_length_m"),
                "cylinder_length_m",
            )
            if (
                length < minimum_length - length_margin
                or length > maximum_length + length_margin
            ):
                raise _fail(
                    "IDEAL_REFERENCE_OUTSIDE_DOMAIN",
                    "scissor-lift cylinder length is outside declared geometry",
                )
            clamped_length = max(
                minimum_length,
                min(maximum_length, length),
            )
            progress = _scissor_progress_for_cylinder_length_m(
                geometry,
                cylinder_length_m=clamped_length,
                home_progress_m=spec.home_progress_m,
                outbound_progress_m=spec.outbound_progress_m,
                home_length_m=home_length,
                outbound_length_m=outbound_length,
            )
            item.update(
                {
                    "cylinder_length_m": length,
                    "input_coordinate": length,
                    "input_coordinate_unit": "m",
                    "output_coordinate": progress,
                    "output_coordinate_unit": "m",
                    "platform_progress_m": progress,
                    "generalized_progress_m": progress,
                    "platform_height_m": initial_height + progress,
                    "ideal_endpoint_tolerance_m": tolerance,
                    "body_poses": _scissor_body_poses(
                        spec,
                        geometry,
                        progress,
                    ),
                    "lateral_shift_m": 0.0,
                    "ideal_pose_source": "declared_scissor_rigid_loop_closure",
                }
            )
            output.append(item)
        _speeds(output, "platform_progress_m", "platform_speed_m_s")
        for item in output:
            item["generalized_speed_m_s"] = item["platform_speed_m_s"]
        return output, {
            "kind": "declared_scissor_linkage_closure",
            "driven_arm_body_id": geometry["arm_body_id"],
            "cylinder_domain_m": [
                min(home_length, outbound_length),
                max(home_length, outbound_length),
            ],
            "platform_progress_domain_m": [
                spec.home_progress_m,
                spec.outbound_progress_m,
            ],
            "body_pose_ids": [body.body_id for body in spec.bodies],
            "body_poses_from_ideal_geometry": True,
            "domain_status": "PASS",
        }


def _ideal_model_binding(
    plan: MechanismPlan,
) -> tuple[IdealMechanismModel, Any, Mapping[str, Any] | None]:
    """Resolve the ideal evaluator together with its registered author."""

    from .family_plugins import (
        mechanism_adapter_for_plan,
        plugin_registration_or_none,
    )

    family_spec = (
        plan.adapter_spec.get("family_spec")
        if isinstance(plan.adapter_spec, dict)
        else None
    )
    plugin, _adapter = mechanism_adapter_for_plan(plan)
    registration = plugin_registration_or_none(str(plugin.plugin_id))
    model = invoke_plugin_callback(
        lambda: plugin.ideal_model(plan, family_spec),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="ideal_model",
        source_component="mechanism.ideal_reference.ideal_model",
        unexpected_code="MECHANISM_ADAPTER_IDEAL_MODEL_FAILED",
        details={"capsule_id": plan.capsule_id},
    )
    if model is not None:
        if (
            not isinstance(getattr(model, "model_id", None), str)
            or not callable(getattr(model, "evaluate_samples", None))
        ):
            message = "mechanism adapter ideal_model() returned an invalid model"
            raise plugin_boundary_error(
                TypeError(message),
                registration=registration,
                plugin_id=str(plugin.plugin_id),
                plugin_kind="mechanism",
                callback="ideal_model",
                source_component="mechanism.ideal_reference.ideal_model",
                code="MECHANISM_ADAPTER_IDEAL_MODEL_INVALID",
                message=message,
                details={"capsule_id": plan.capsule_id},
                contract_violation=True,
            )
        return model, plugin, registration
    message = (
        "selected mechanism adapter returned no deterministic ideal model"
    )
    raise plugin_boundary_error(
        TypeError(message),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="ideal_model",
        source_component="mechanism.ideal_reference.ideal_model",
        code="MECHANISM_ADAPTER_IDEAL_MODEL_REQUIRED",
        message=message,
        details={"capsule_id": plan.capsule_id},
        contract_violation=True,
    )


def ideal_model_for_plan(plan: MechanismPlan) -> IdealMechanismModel:
    """Resolve the deterministic ideal evaluator from the plan's adapter."""

    model, _plugin, _registration = _ideal_model_binding(plan)
    return model


def generate_ideal_reference_trace(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: Mapping[str, Any],
) -> dict[str, Any]:
    expected = {
        "status": "COMPLETED",
        "branch": "physical",
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches = [
        key
        for key, value in expected.items()
        if physical_trace.get(key) != value
    ]
    if mismatches:
        raise _fail(
            "IDEAL_REFERENCE_PHYSICAL_TRACE_MISMATCH",
            "physical trace identity differs at: " + ", ".join(mismatches),
        )
    raw_samples = physical_trace.get("samples")
    if not isinstance(raw_samples, list) or len(raw_samples) < 2:
        raise _fail(
            "IDEAL_REFERENCE_SAMPLES_MISSING",
            "physical trace has too few saved samples",
        )
    samples = [
        sample for sample in raw_samples if isinstance(sample, Mapping)
    ]
    if len(samples) != len(raw_samples):
        raise _fail(
            "IDEAL_REFERENCE_SAMPLE_INVALID",
            "physical trace contains a non-object sample",
        )
    model, plugin, registration = _ideal_model_binding(plan)
    evaluated = invoke_plugin_callback(
        lambda: model.evaluate_samples(plan, samples),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="ideal_model.evaluate_samples",
        source_component="mechanism.ideal_reference.evaluate_samples",
        unexpected_code="MECHANISM_ADAPTER_IDEAL_EVALUATION_FAILED",
        details={
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
        },
    )
    if (
        not isinstance(evaluated, tuple)
        or len(evaluated) != 2
        or not isinstance(evaluated[0], list)
        or any(not isinstance(item, dict) for item in evaluated[0])
        or not isinstance(evaluated[1], dict)
    ):
        message = "ideal model evaluate_samples() returned an invalid result"
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="ideal_model.evaluate_samples",
            source_component="mechanism.ideal_reference.evaluate_samples",
            code="MECHANISM_ADAPTER_IDEAL_EVALUATION_INVALID",
            message=message,
            details={
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
            },
            contract_violation=True,
        )
    ideal_samples, domain = evaluated
    family_spec = (
        plan.adapter_spec.get("family_spec")
        if isinstance(plan.adapter_spec, Mapping)
        else None
    )
    required_reference_kind = (
        family_spec.get("ideal_reference_kind")
        if isinstance(family_spec, Mapping)
        else None
    )
    if (
        isinstance(required_reference_kind, str)
        and required_reference_kind
        and domain.get("kind") != required_reference_kind
    ):
        message = (
            "hash-bound family spec requires ideal reference kind "
            f"{required_reference_kind!r}, got {domain.get('kind')!r}"
        )
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="ideal_model.evaluate_samples",
            source_component="mechanism.ideal_reference.reference_contract",
            code="MECHANISM_ADAPTER_IDEAL_REFERENCE_KIND_MISMATCH",
            message=message,
            details={
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
                "required_kind": required_reference_kind,
                "actual_kind": domain.get("kind"),
            },
            contract_violation=True,
        )
    time_integration = domain.get("time_integration") is True
    internal_dynamics_steps = int(domain.get("internal_dynamics_steps") or 0)
    payload = {
        "schema_version": IDEAL_REFERENCE_SCHEMA,
        "status": "COMPLETED",
        "branch": "ideal_reference",
        "package_id": plan.package_id,
        "design_version": plan.design_version,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
        "physical_trace_sha256": canonical_sha256(physical_trace),
        "engine": {
            "name": "deterministic_template_evaluator",
            "model_id": model.model_id,
            "time_integration": time_integration,
            "collision_enabled": False,
            "collision_geometry_loaded": False,
            "internal_dynamics_steps": internal_dynamics_steps,
        },
        "branch_semantics": {
            "physical_samples_supply_input_coordinates_only": True,
            "physical_output_coordinates_copied": False,
            "physical_body_poses_copied": False,
            "ideal_body_poses_computed": bool(
                ideal_samples
                and isinstance(ideal_samples[0].get("body_poses"), dict)
            ),
            "contact_evidence_authority": False,
            "dynamic_response_authority": time_integration,
        },
        "domain": domain,
        "axis_abs": list(plan.axis_abs),
        "sample_count": len(ideal_samples),
        "samples": ideal_samples,
    }
    payload["ideal_reference_sha256"] = canonical_sha256(payload)
    return payload


def materialize_ideal_references(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Build light ideal traces from completed physical saved samples."""

    from pathlib import Path

    from .io import (
        ensure_inside,
        read_json,
        read_trace_json,
        relative_to_root,
        sha256_file,
        write_json_atomic,
    )

    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    summary_path = output / "results" / "ideal_reference_summary.json"
    dispatch = read_json(output / "dispatch_summary.json")
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for relative_plan in dispatch.get("plan_files", []):
        plan_path = ensure_inside(workspace, str(relative_plan))
        plan = MechanismPlan.from_dict(read_json(plan_path))
        load_root = output / "loads" / plan.capsule_id
        for load_path in sorted(load_root.glob("*.json")):
            load_case = LoadCase.from_dict(read_json(load_path))
            trace_root = (
                output
                / "traces"
                / plan.capsule_id
                / load_case.load_case_id
            )
            physical_path = trace_root / "physical.trace.json"
            ideal_path = trace_root / "ideal_reference.trace.json"
            try:
                physical = read_trace_json(physical_path)
                ideal = generate_ideal_reference_trace(
                    plan,
                    load_case,
                    physical,
                )
                write_json_atomic(ideal_path, ideal)
                generated.append(
                    {
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case.load_case_id,
                        "trace": relative_to_root(workspace, ideal_path),
                        "trace_sha256": sha256_file(ideal_path),
                        "model_id": ideal["engine"]["model_id"],
                        "sample_count": ideal["sample_count"],
                    }
                )
            except (OSError, ValueError, MechanismValidationError) as exc:
                failure = _ideal_reference_failure(
                    exc,
                    capsule_id=plan.capsule_id,
                    load_case_id=load_case.load_case_id,
                )
                failures.append(
                    {
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case.load_case_id,
                        "code": getattr(
                            exc,
                            "code",
                            "IDEAL_REFERENCE_INCONCLUSIVE",
                        ),
                        "detail": str(exc),
                        "failure": failure,
                    }
                )
    status = "PASS" if generated and not failures else "INCONCLUSIVE"
    summary = {
        "schema_version": IDEAL_REFERENCE_SCHEMA,
        "status": status,
        "certificate_authority": status == "PASS",
        "trace_count": len(generated),
        "failure_count": len(failures),
        "traces": generated,
        "failures": failures,
        "engine_policy": {
            "time_integration": False,
            "collision_geometry_loaded": False,
            "pychrono_queue_jobs": 0,
        },
    }
    write_json_atomic(summary_path, summary)
    summary["summary"] = relative_to_root(workspace, summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary


def summarize_materialized_ideal_references(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Aggregate receipt-bound per-load ideal traces without rewriting them."""

    from pathlib import Path

    from .io import (
        ensure_inside,
        read_json,
        read_trace_json,
        relative_to_root,
        sha256_file,
        write_json_atomic,
    )

    from .reuse import load_reuse_receipts

    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    summary_path = output / "results" / "ideal_reference_summary.json"
    dispatch = read_json(output / "dispatch_summary.json")
    if dispatch.get("execution") not in {
        "per_load_composite_lane_v1",
        "per_load_pipeline_v2",
    }:
        raise _fail(
            "IDEAL_REFERENCE_LANE_PROTOCOL_INVALID",
            "materialized ideal aggregation requires composite load lanes",
        )
    receipts_by_lane = {
        (entry.get("capsule_id"), entry.get("load_case_id")): entry
        for entry in load_reuse_receipts(output)
    }
    generated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for relative_plan in dispatch.get("plan_files", []):
        plan_path = ensure_inside(workspace, str(relative_plan))
        plan = MechanismPlan.from_dict(read_json(plan_path))
        load_root = output / "loads" / plan.capsule_id
        for load_path in sorted(load_root.glob("*.json")):
            load_case = LoadCase.from_dict(read_json(load_path))
            ideal_path = (
                output
                / "traces"
                / plan.capsule_id
                / load_case.load_case_id
                / "ideal_reference.trace.json"
            )
            lane_path = (
                output
                / "results"
                / plan.capsule_id
                / "lanes"
                / f"{load_case.load_case_id}.json"
            )
            try:
                lane = read_json(lane_path)
                artifact = (lane.get("artifacts") or {}).get(
                    "ideal_reference_trace"
                )
                digest = sha256_file(ideal_path)
                # A lane adopted through a sign-off receipt keeps its
                # origin-run paths and identity hashes; the chain-verified
                # receipt is the only authority allowed to map them onto
                # this run's byte-identical copies, and only when it
                # licenses exactly this run's plan/load inputs.
                receipt = receipts_by_lane.get(
                    (plan.capsule_id, load_case.load_case_id)
                )
                if receipt is not None and (
                    receipt.get("current_plan_sha256")
                    != canonical_sha256(plan.to_dict())
                    or receipt.get("current_load_case_sha256")
                    != canonical_sha256(load_case.to_dict())
                    or receipt.get("current_package_binding_sha256")
                    != plan.package_binding_sha256
                    or not isinstance(
                        receipt.get("origin_package_binding_sha256"), str
                    )
                ):
                    raise _fail(
                        "IDEAL_REFERENCE_LANE_BINDING_INVALID",
                        "sign-off receipt does not license this lane's "
                        "current inputs",
                    )
                if receipt is not None:
                    adopted = (receipt.get("artifacts") or {}).get(
                        "ideal_reference_trace"
                    )
                    binding_ok = (
                        isinstance(adopted, Mapping)
                        and isinstance(artifact, Mapping)
                        and artifact.get("file")
                        == adopted.get("origin_file")
                        and adopted.get("sha256") == digest
                    )
                    expected_plan_sha256 = receipt.get("origin_plan_sha256")
                    expected_load_case_sha256 = receipt.get(
                        "origin_load_case_sha256"
                    )
                    expected_package_binding_sha256 = receipt.get(
                        "origin_package_binding_sha256"
                    )
                else:
                    binding_ok = isinstance(artifact, Mapping) and artifact.get(
                        "file"
                    ) == relative_to_root(workspace, ideal_path)
                    expected_plan_sha256 = canonical_sha256(plan.to_dict())
                    expected_load_case_sha256 = canonical_sha256(
                        load_case.to_dict()
                    )
                    expected_package_binding_sha256 = (
                        plan.package_binding_sha256
                    )
                if (
                    lane.get("schema_version")
                    != "simulation_ready_cad_mechanism_load_lane_v1"
                    or lane.get("status") != "PASS"
                    or not isinstance(artifact, Mapping)
                    or not binding_ok
                    or artifact.get("sha256") != digest
                ):
                    raise _fail(
                        "IDEAL_REFERENCE_LANE_BINDING_INVALID",
                        "lane result does not bind its ideal reference",
                    )
                ideal = read_trace_json(ideal_path)
                core = dict(ideal)
                recorded = core.pop("ideal_reference_sha256", None)
                expected = {
                    "status": "COMPLETED",
                    "branch": "ideal_reference",
                    "package_binding_sha256": (
                        expected_package_binding_sha256
                    ),
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case.load_case_id,
                    "plan_sha256": expected_plan_sha256,
                    "load_case_sha256": expected_load_case_sha256,
                }
                if (
                    any(ideal.get(key) != value for key, value in expected.items())
                    or recorded != canonical_sha256(core)
                ):
                    raise _fail(
                        "IDEAL_REFERENCE_LANE_BINDING_INVALID",
                        "materialized ideal reference identity changed",
                    )
                engine = ideal.get("engine")
                generated.append(
                    {
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case.load_case_id,
                        "trace": relative_to_root(workspace, ideal_path),
                        "trace_sha256": digest,
                        "model_id": (
                            engine.get("model_id")
                            if isinstance(engine, Mapping)
                            else None
                        ),
                        "sample_count": ideal.get("sample_count"),
                    }
                )
            except (OSError, ValueError, MechanismValidationError) as exc:
                failure = _ideal_reference_failure(
                    exc,
                    capsule_id=plan.capsule_id,
                    load_case_id=load_case.load_case_id,
                )
                failures.append(
                    {
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case.load_case_id,
                        "code": getattr(
                            exc,
                            "code",
                            "IDEAL_REFERENCE_INCONCLUSIVE",
                        ),
                        "detail": str(exc),
                        "failure": failure,
                    }
                )
    status = "PASS" if generated and not failures else "INCONCLUSIVE"
    summary = {
        "schema_version": IDEAL_REFERENCE_SCHEMA,
        "status": status,
        "certificate_authority": status == "PASS",
        "trace_count": len(generated),
        "failure_count": len(failures),
        "traces": generated,
        "failures": failures,
        "engine_policy": {
            "time_integration": False,
            "collision_geometry_loaded": False,
            "pychrono_queue_jobs": 0,
        },
    }
    write_json_atomic(summary_path, summary)
    summary["summary"] = relative_to_root(workspace, summary_path)
    summary["summary_sha256"] = sha256_file(summary_path)
    return summary
