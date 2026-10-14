"""Deterministic 3D pose replay for the light ideal-reference branch.

The functions here never invoke a physics engine. They map trusted ideal
coordinates onto the intake-bound rigid-body geometry for presentation only.
The result has no contact, force, stability, or dynamic-response authority.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class IdealReplayUnsupported(ValueError):
    pass


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise IdealReplayUnsupported(f"{label} is not finite")
    return float(value)


def _vector(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise IdealReplayUnsupported(f"{label} is not a 3-vector")
    return tuple(_number(item, label) for item in value)


def _normalize(vector: Sequence[float], label: str) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(float(item) ** 2 for item in vector))
    if magnitude <= 1e-12:
        raise IdealReplayUnsupported(f"{label} has zero length")
    return tuple(float(item) / magnitude for item in vector)


def _quat(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise IdealReplayUnsupported(f"{label} is not a quaternion")
    values = tuple(_number(item, label) for item in value)
    magnitude = math.sqrt(sum(item * item for item in values))
    if magnitude <= 1e-12:
        raise IdealReplayUnsupported(f"{label} has zero length")
    return tuple(item / magnitude for item in values)


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _scale(vector: Sequence[float], factor: float) -> tuple[float, ...]:
    return tuple(float(item) * float(factor) for item in vector)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cross(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _axis_quaternion(
    axis: Sequence[float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    unit = _normalize(axis, "rotation axis")
    half = 0.5 * float(angle_rad)
    sine = math.sin(half)
    return (
        math.cos(half),
        unit[0] * sine,
        unit[1] * sine,
        unit[2] * sine,
    )


def _quat_multiply(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _rotate_vector(
    vector: Sequence[float],
    axis: Sequence[float],
    angle_rad: float,
) -> tuple[float, float, float]:
    unit = _normalize(axis, "rotation axis")
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return _add(
        _add(
            _scale(vector, cosine),
            _scale(_cross(unit, vector), sine),
        ),
        _scale(unit, _dot(unit, vector) * (1.0 - cosine)),
    )


def _pose(
    position: Sequence[float],
    quaternion: Sequence[float],
) -> dict[str, list[float]]:
    return {
        "position_abs_m": [float(item) for item in position],
        "rotation_abs_quat_wxyz": [float(item) for item in quaternion],
    }


def _rotate_pose_about(
    initial: Mapping[str, Any],
    pivot: Sequence[float],
    axis: Sequence[float],
    angle_rad: float,
) -> dict[str, list[float]]:
    position = _vector(initial.get("position_abs_m"), "body position")
    rotation = _quat(
        initial.get("rotation_abs_quat_wxyz"),
        "body rotation",
    )
    delta = _axis_quaternion(axis, angle_rad)
    moved = _add(
        pivot,
        _rotate_vector(_sub(position, pivot), axis, angle_rad),
    )
    return _pose(moved, _quat_multiply(delta, rotation))


def _translate_pose(
    initial: Mapping[str, Any],
    direction: Sequence[float],
    distance: float,
) -> dict[str, list[float]]:
    position = _vector(initial.get("position_abs_m"), "body position")
    rotation = _quat(
        initial.get("rotation_abs_quat_wxyz"),
        "body rotation",
    )
    moved = _add(position, _scale(_normalize(direction, "translation axis"), distance))
    return _pose(moved, rotation)


def _anchor_pose(
    initial: Mapping[str, Any],
    home_anchor: Sequence[float],
    current_anchor: Sequence[float],
    axis: Sequence[float],
    angle_rad: float,
) -> dict[str, list[float]]:
    position = _vector(initial.get("position_abs_m"), "body position")
    rotation = _quat(
        initial.get("rotation_abs_quat_wxyz"),
        "body rotation",
    )
    moved = _add(
        current_anchor,
        _rotate_vector(_sub(position, home_anchor), axis, angle_rad),
    )
    delta = _axis_quaternion(axis, angle_rad)
    return _pose(moved, _quat_multiply(delta, rotation))


def _family_spec(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    adapter = plan.get("adapter_spec")
    family = adapter.get("family_spec") if isinstance(adapter, Mapping) else None
    return family if isinstance(family, Mapping) else {}


def _initial_body_poses(
    plan: Mapping[str, Any],
    physical_trace: Mapping[str, Any],
) -> tuple[dict[str, dict[str, list[float]]], set[str]]:
    family = _family_spec(plan)
    result: dict[str, dict[str, list[float]]] = {}
    fixed: set[str] = set()
    for body in family.get("bodies") or []:
        if not isinstance(body, Mapping):
            continue
        body_id = str(body.get("body_id") or "")
        if not body_id:
            continue
        result[body_id] = _pose(
            _vector(body.get("center_abs_m"), f"{body_id}.center_abs_m"),
            _quat(
                body.get("rotation_abs_quat_wxyz", (1.0, 0.0, 0.0, 0.0)),
                f"{body_id}.rotation_abs_quat_wxyz",
            ),
        )
        if body.get("fixed") is True:
            fixed.add(body_id)

    samples = [
        sample
        for sample in physical_trace.get("samples") or []
        if isinstance(sample, Mapping)
    ]
    first_poses = (samples[0].get("body_poses") or {}) if samples else {}
    for body_id, value in first_poses.items():
        if body_id in result or not isinstance(value, Mapping):
            continue
        result[str(body_id)] = _pose(
            _vector(value.get("position_abs_m"), f"{body_id}.position_abs_m"),
            _quat(
                value.get("rotation_abs_quat_wxyz"),
                f"{body_id}.rotation_abs_quat_wxyz",
            ),
        )

    if not result and samples:
        first = samples[0]
        fixed_pose = first.get("fixed_pose")
        moving_pose = first.get("moving_pose")
        if isinstance(fixed_pose, Mapping) and isinstance(moving_pose, Mapping):
            result["fixed_boundary"] = _pose(
                _vector(
                    fixed_pose.get("position_abs_m"),
                    "fixed_pose.position_abs_m",
                ),
                _quat(
                    fixed_pose.get("rotation_abs_quat_wxyz"),
                    "fixed_pose.rotation_abs_quat_wxyz",
                ),
            )
            result["moving_group"] = _pose(
                _vector(
                    moving_pose.get("position_abs_m"),
                    "moving_pose.position_abs_m",
                ),
                _quat(
                    moving_pose.get("rotation_abs_quat_wxyz"),
                    "moving_pose.rotation_abs_quat_wxyz",
                ),
            )
            fixed.add("fixed_boundary")

    collision = physical_trace.get("collision_geometry") or {}
    for body in collision.get("bodies") or []:
        if isinstance(body, Mapping) and body.get("fixed") is True:
            fixed.add(str(body.get("body_id") or ""))

    if not result:
        raise IdealReplayUnsupported("no initial rigid-body poses are available")
    return result, fixed


def _copy_poses(
    initial: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    return {
        body_id: _pose(
            _vector(value.get("position_abs_m"), f"{body_id}.position_abs_m"),
            _quat(
                value.get("rotation_abs_quat_wxyz"),
                f"{body_id}.rotation_abs_quat_wxyz",
            ),
        )
        for body_id, value in initial.items()
    }


def _plain_samples(
    plan: Mapping[str, Any],
    samples: list[dict[str, Any]],
    initial: Mapping[str, Mapping[str, Any]],
    fixed: set[str],
) -> None:
    moving = [body_id for body_id in initial if body_id not in fixed]
    if len(moving) != 1:
        named = [
            body_id
            for body_id in initial
            if body_id.casefold() in {"moving", "moving_group"}
        ]
        moving = named
    if len(moving) != 1:
        raise IdealReplayUnsupported("plain replay requires one moving body")
    body_id = moving[0]
    axis = _vector(plan.get("axis_abs"), "plan.axis_abs")
    family = str(plan.get("family") or "")
    field = (
        "generalized_progress_m"
        if family == "prismatic"
        else "generalized_angle_rad"
    )
    home = _number(samples[0].get(field), field)
    for sample in samples:
        poses = _copy_poses(initial)
        coordinate = _number(sample.get(field), field)
        if family == "prismatic":
            poses[body_id] = _translate_pose(
                initial[body_id],
                axis,
                coordinate - home,
            )
        else:
            poses[body_id] = _rotate_pose_about(
                initial[body_id],
                _vector(
                    plan.get("joint_origin_abs_m"),
                    "plan.joint_origin_abs_m",
                ),
                axis,
                coordinate - home,
            )
        sample["body_poses"] = poses


def _rack_samples(
    family: Mapping[str, Any],
    samples: list[dict[str, Any]],
    initial: Mapping[str, Mapping[str, Any]],
) -> None:
    pinion = str(family.get("pinion_body_id") or "")
    rack = str(family.get("rack_body_id") or "")
    if pinion not in initial or rack not in initial:
        raise IdealReplayUnsupported("rack-pinion body poses are incomplete")
    pinion_axis = _vector(family.get("pinion_axis_abs"), "pinion_axis_abs")
    rack_axis = _vector(family.get("rack_axis_abs"), "rack_axis_abs")
    pivot = _vector(family.get("pinion_origin_abs_m"), "pinion_origin_abs_m")
    home_angle = _number(samples[0].get("pinion_angle_rad"), "pinion_angle_rad")
    home_progress = _number(samples[0].get("rack_progress_m"), "rack_progress_m")
    for sample in samples:
        poses = _copy_poses(initial)
        angle = _number(sample.get("pinion_angle_rad"), "pinion_angle_rad")
        progress = _number(sample.get("rack_progress_m"), "rack_progress_m")
        poses[pinion] = _rotate_pose_about(
            initial[pinion],
            pivot,
            pinion_axis,
            angle - home_angle,
        )
        poses[rack] = _translate_pose(
            initial[rack],
            rack_axis,
            progress - home_progress,
        )
        sample["body_poses"] = poses


def _clock_samples(
    family: Mapping[str, Any],
    samples: list[dict[str, Any]],
    initial: Mapping[str, Mapping[str, Any]],
) -> None:
    shafts = [
        shaft
        for shaft in family.get("shafts") or []
        if isinstance(shaft, Mapping)
    ]
    if not shafts:
        raise IdealReplayUnsupported("clock shaft geometry is missing")
    first_angles = samples[0].get("shaft_angles_rad")
    if not isinstance(first_angles, Mapping):
        raise IdealReplayUnsupported("clock ideal shaft angles are missing")
    for sample in samples:
        angles = sample.get("shaft_angles_rad")
        if not isinstance(angles, Mapping):
            raise IdealReplayUnsupported("clock ideal shaft angles are missing")
        poses = _copy_poses(initial)
        for shaft in shafts:
            body_id = str(shaft.get("body_id") or "")
            if (
                body_id not in initial
                or body_id not in angles
                or body_id not in first_angles
            ):
                continue
            poses[body_id] = _rotate_pose_about(
                initial[body_id],
                _vector(shaft.get("origin_abs_m"), f"{body_id}.origin_abs_m"),
                _vector(shaft.get("axis_abs"), f"{body_id}.axis_abs"),
                _number(angles[body_id], f"{body_id}.angle")
                - _number(first_angles[body_id], f"{body_id}.home_angle"),
            )
        sample["body_poses"] = poses


def _signed_angle(
    home: Sequence[float],
    current: Sequence[float],
    axis: Sequence[float],
) -> float:
    home_unit = _normalize(home, "home link direction")
    current_unit = _normalize(current, "current link direction")
    unit_axis = _normalize(axis, "linkage axis")
    return math.atan2(
        _dot(unit_axis, _cross(home_unit, current_unit)),
        max(-1.0, min(1.0, _dot(home_unit, current_unit))),
    )


def _four_bar_samples(
    plan: Mapping[str, Any],
    family: Mapping[str, Any],
    samples: list[dict[str, Any]],
    initial: Mapping[str, Mapping[str, Any]],
    fixed: set[str],
) -> None:
    from ..mechanism.compound_spec import compound_linkage_from_dict
    from ..mechanism.ideal_reference import _planar_four_bar_geometry
    from ..mechanism.models import MechanismPlan

    spec = compound_linkage_from_dict(family)
    typed_plan = MechanismPlan.from_dict(dict(plan))
    geometry = _planar_four_bar_geometry(spec)
    input_body = spec.input_body_id
    output_body = spec.output_body_id
    intermediate = [
        body_id
        for body_id in initial
        if body_id not in fixed and body_id not in {input_body, output_body}
    ]
    if (
        input_body not in initial
        or output_body not in initial
        or len(intermediate) != 1
    ):
        raise IdealReplayUnsupported(
            "planar four-bar replay requires one coupler body"
        )
    coupler = intermediate[0]
    axis = geometry["axis"]
    point_a = geometry["origin"]
    point_d = geometry["point_d"]
    basis_x = geometry["basis_x"]
    basis_y = geometry["basis_y"]

    def plane_point(value: Sequence[float]) -> tuple[float, float, float]:
        return _add(
            point_a,
            _add(
                _scale(basis_x, float(value[0])),
                _scale(basis_y, float(value[1])),
            ),
        )

    b_home = plane_point(geometry["b_home"])
    c_home = plane_point(geometry["c_home"])
    input_home = math.atan2(
        geometry["b_home"][1],
        geometry["b_home"][0],
    )
    coupler_home = _sub(c_home, b_home)
    for sample in samples:
        input_angle = _number(
            sample.get("compound_input_angle_rad"),
            "compound_input_angle_rad",
        )
        output_angle = _number(
            sample.get("compound_output_angle_rad"),
            "compound_output_angle_rad",
        )
        b_current = plane_point(
            (
                geometry["crank_length"] * math.cos(input_angle),
                geometry["crank_length"] * math.sin(input_angle),
            )
        )
        output_delta = output_angle - spec.home_output_angle_rad
        c_current = _add(
            point_d,
            _rotate_vector(_sub(c_home, point_d), axis, output_delta),
        )
        coupler_delta = _signed_angle(
            coupler_home,
            _sub(c_current, b_current),
            axis,
        )
        poses = _copy_poses(initial)
        poses[input_body] = _rotate_pose_about(
            initial[input_body],
            point_a,
            axis,
            input_angle - input_home,
        )
        poses[output_body] = _rotate_pose_about(
            initial[output_body],
            point_d,
            axis,
            output_delta,
        )
        poses[coupler] = _anchor_pose(
            initial[coupler],
            b_home,
            b_current,
            axis,
            coupler_delta,
        )
        sample["body_poses"] = poses


def synthesize_ideal_kinematic_trace(
    *,
    plan: Mapping[str, Any],
    physical_trace: Mapping[str, Any],
    ideal_trace: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a presentation copy of an ideal trace with deterministic poses."""

    if ideal_trace.get("branch") != "ideal_reference":
        raise IdealReplayUnsupported("trace is not an ideal reference")
    samples = [
        dict(sample)
        for sample in ideal_trace.get("samples") or []
        if isinstance(sample, Mapping)
    ]
    if not samples:
        raise IdealReplayUnsupported("ideal trace contains no samples")
    initial, fixed = _initial_body_poses(plan, physical_trace)
    family = _family_spec(plan)
    mechanism_kind = str(family.get("mechanism_kind") or "")
    plan_family = str(plan.get("family") or "")
    if mechanism_kind == "rack_pinion":
        _rack_samples(family, samples, initial)
        pose_model = "rack_pinion_closed_form_pose_v1"
    elif mechanism_kind == "spur_gear_clock":
        _clock_samples(family, samples, initial)
        pose_model = "clock_shaft_graph_pose_v1"
    elif mechanism_kind in {
        "crown_bevel_dual_output",
        "multi_output_shaft_coupling",
    }:
        _clock_samples(family, samples, initial)
        pose_model = "multi_output_shaft_graph_pose_v1"
    elif plan_family == "compound" and mechanism_kind in {
        "",
        "revolute_linkage",
    }:
        _four_bar_samples(plan, family, samples, initial, fixed)
        pose_model = "planar_four_bar_closure_pose_v1"
    elif plan_family in {"prismatic", "revolute", "continuous_revolute"}:
        _plain_samples(plan, samples, initial, fixed)
        pose_model = "plain_coordinate_pose_v1"
    else:
        raise IdealReplayUnsupported(
            f"no deterministic pose model for {mechanism_kind or plan_family}"
        )

    for sample in samples:
        poses = sample.get("body_poses")
        if not isinstance(poses, Mapping) or isinstance(sample.get("moving_pose"), Mapping):
            continue
        moving = poses.get("moving_group") or poses.get("moving")
        if isinstance(moving, Mapping):
            sample["moving_pose"] = dict(moving)

    payload = dict(ideal_trace)
    payload["samples"] = samples
    payload["visual_replay_semantics"] = {
        "status": "GENERATED",
        "pose_model": pose_model,
        "physics_engine_used": False,
        "time_integration_used": False,
        "collision_geometry_loaded_for_solving": False,
        "physical_output_poses_copied": False,
        "physical_home_pose_used_for_scene_registration": not bool(
            family.get("bodies")
        ),
        "contact_evidence_authority": False,
        "force_evidence_authority": False,
        "dynamic_response_authority": False,
        "purpose": "interactive ideal kinematic animation only",
    }
    return payload
