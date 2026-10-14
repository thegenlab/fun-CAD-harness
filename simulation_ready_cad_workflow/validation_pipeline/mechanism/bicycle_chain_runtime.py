from __future__ import annotations

import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .bicycle_chain_spec import (
    BICYCLE_CHAIN_PLUGIN_ID,
    BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
    BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
    BicycleBearingSpec,
    bicycle_chain_from_dict,
    chain_engagement_active,
    chain_reaction_torques,
    resolve_bicycle_contract_bindings,
    validate_runtime_bindings,
    validate_task_input_target,
)
from .bicycle_chain_path import (
    build_moving_chain_path_contract,
    build_moving_chain_path_state,
)
from .bounded_support import (
    RELATIVE_ANCHOR_DRIFT,
    BoundedSupportRuntime,
    body_component_refs,
    bounded_support_gate,
    conditional_runtime_summary,
)
from .collision_policy import (
    ANALYTICAL_TRANSMISSION,
    COMPLIANT_INTERFACE_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .conditional_interface import (
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
)
from .io import canonical_sha256, ensure_inside, read_json, sha256_file
from .models import MechanismValidationError
from .wrench_ledger import (
    add_ledgered_load,
    create_body_force,
    create_body_torque,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

if TYPE_CHECKING:
    from .worker import ExecutionRequest


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _scale(
    value: Sequence[float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(float(item) * factor for item in value)


def _cross(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _clamp_vector(
    value: Sequence[float],
    maximum: float,
) -> tuple[float, float, float]:
    magnitude = _norm(value)
    if magnitude <= maximum or magnitude <= 1.0e-30:
        return tuple(float(item) for item in value)
    return _scale(value, maximum / magnitude)


def _componentwise(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) * float(b) for a, b in zip(left, right))


def _clamp_scalar(value: float, maximum: float) -> float:
    return max(-float(maximum), min(float(maximum), float(value)))


def _bearing_translation_coefficients(
    bearing: BicycleBearingSpec,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return (
        tuple(
            float(value)
            for value in bearing.translation_stiffness_xyz_n_m
        ),
        tuple(
            float(value)
            for value in bearing.translation_damping_xyz_n_s_m
        ),
    )


def _bearing_off_axis_coefficients(
    bearing: BicycleBearingSpec,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return (
        tuple(
            float(value)
            for value in (
                bearing.off_axis_rotation_stiffness_xyz_nm_rad
            )
        ),
        tuple(
            float(value)
            for value in (
                bearing.off_axis_rotation_damping_xyz_nm_s_rad
            )
        ),
    )


def _bearing_total_torque_cap(bearing: BicycleBearingSpec) -> float:
    return float(bearing.maximum_total_support_torque_nm)


def _bearing_primary_torque_cap(
    bearing: BicycleBearingSpec,
) -> float | None:
    value = bearing.primary_axis_maximum_abs_torque_nm
    if value is None:
        return None
    return float(value)


def _body_inertia_diagonal(
    spec: Any,
    body_spec: Any,
) -> tuple[float, float, float]:
    dynamics = tuple(spec.body_dynamics)
    matches = [
        item
        for item in dynamics
        if item.body_id == body_spec.body_id
    ]
    if len(matches) != 1:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BODY_DYNAMICS_BINDING_INVALID",
            f"body {body_spec.body_id!r} does not have one dynamics record",
        )
    values = matches[0].inertia_diagonal_kg_m2
    inertia = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0.0 for value in inertia):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BODY_INERTIA_INVALID",
            f"body {body_spec.body_id!r} inertia must be finite and positive",
        )
    return inertia


def _quat_conjugate(
    value: Sequence[float],
) -> tuple[float, float, float, float]:
    return (
        float(value[0]),
        -float(value[1]),
        -float(value[2]),
        -float(value[3]),
    )


def _quat_product(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = (float(item) for item in left)
    rw, rx, ry, rz = (float(item) for item in right)
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _axis_angle_quaternion(
    axis: Sequence[float],
    angle_rad: float,
) -> tuple[float, float, float, float]:
    magnitude = _norm(axis)
    if magnitude <= 1.0e-12:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_SWEEP_AXIS_INVALID",
            "prescribed swept-envelope axis is zero",
        )
    normalized = tuple(float(value) / magnitude for value in axis)
    half = 0.5 * float(angle_rad)
    sine = math.sin(half)
    return (
        math.cos(half),
        normalized[0] * sine,
        normalized[1] * sine,
        normalized[2] * sine,
    )


def _rotate_about_axis(
    value: Sequence[float],
    axis: Sequence[float],
    angle_rad: float,
) -> tuple[float, float, float]:
    magnitude = _norm(axis)
    if magnitude <= 1.0e-12:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_SWEEP_AXIS_INVALID",
            "prescribed swept-envelope axis is zero",
        )
    unit = tuple(float(item) / magnitude for item in axis)
    cosine = math.cos(angle_rad)
    sine = math.sin(angle_rad)
    return _add(
        _add(
            _scale(value, cosine),
            _scale(_cross(unit, value), sine),
        ),
        _scale(unit, _dot(unit, value) * (1.0 - cosine)),
    )


def _prescribed_rotation_pose(
    *,
    center_abs_m: Sequence[float],
    initial_orientation_wxyz: Sequence[float],
    origin_abs_m: Sequence[float],
    axis_abs: Sequence[float],
    angle_rad: float,
) -> dict[str, list[float]]:
    relative_center = _sub(center_abs_m, origin_abs_m)
    rotated_center = _add(
        origin_abs_m,
        _rotate_about_axis(relative_center, axis_abs, angle_rad),
    )
    orientation = _quat_product(
        _axis_angle_quaternion(axis_abs, angle_rad),
        initial_orientation_wxyz,
    )
    return {
        "position_abs_m": list(rotated_center),
        "rotation_abs_quat_wxyz": list(orientation),
    }


def _bicycle_prescribed_swept_envelope_samples(
    spec: Any,
    *,
    maximum_step_rad: float = math.pi / 36.0,
) -> list[dict[str, Any]]:
    """Cover the frozen task angle out and back before long dynamics."""

    acceptance = spec.acceptance_thresholds
    if acceptance is None:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_TASK_CONTRACT_INVALID",
            "prescribed swept-envelope audit has no frozen acceptance angle",
        )
    required_output_angle = float(
        acceptance.powered_rotation_min_rad
    )
    if (
        not math.isfinite(maximum_step_rad)
        or maximum_step_rad <= 0.0
        or maximum_step_rad > math.pi / 18.0
    ):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_SWEEP_SAMPLING_INVALID",
            "prescribed swept-envelope sampling step is invalid",
        )
    subdivisions = max(
        1,
        math.ceil(required_output_angle / maximum_step_rad),
    )
    body_specs = {body.body_id: body for body in spec.bodies}
    fixed = body_specs[spec.fixed_body_id]
    crank = body_specs[spec.input_body_id]
    rear = body_specs[spec.output_body_id]
    ratio = float(spec.signed_output_per_input)
    if abs(ratio) <= 1.0e-12:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_SWEEP_RATIO_INVALID",
            "prescribed swept-envelope ratio is zero",
        )
    fixed_pose = {
        "position_abs_m": list(fixed.center_abs_m),
        "rotation_abs_quat_wxyz": list(fixed.rotation_abs_quat_wxyz),
    }
    outward_fractions = [
        index / subdivisions for index in range(subdivisions + 1)
    ]
    fractions = outward_fractions + list(
        reversed(outward_fractions[:-1])
    )
    samples: list[dict[str, Any]] = []
    for index, fraction in enumerate(fractions):
        output_angle = math.copysign(
            required_output_angle * fraction,
            ratio,
        )
        input_angle = output_angle / ratio
        samples.append(
            {
                "time_s": index * maximum_step_rad,
                "phase_id": "validator_prescribed_swept_envelope",
                "sweep_fraction": fraction,
                "sweep_direction": (
                    "outbound"
                    if index <= subdivisions
                    else "return"
                ),
                "contract_required_output_angle_rad": (
                    required_output_angle
                ),
                "body_poses": {
                    spec.fixed_body_id: fixed_pose,
                    spec.input_body_id: _prescribed_rotation_pose(
                        center_abs_m=crank.center_abs_m,
                        initial_orientation_wxyz=(
                            crank.rotation_abs_quat_wxyz
                        ),
                        origin_abs_m=spec.input_origin_abs_m,
                        axis_abs=spec.input_frame.axis_abs,
                        angle_rad=input_angle,
                    ),
                    spec.output_body_id: _prescribed_rotation_pose(
                        center_abs_m=rear.center_abs_m,
                        initial_orientation_wxyz=rear.rotation_abs_quat_wxyz,
                        origin_abs_m=spec.output_origin_abs_m,
                        axis_abs=spec.output_frame.axis_abs,
                        angle_rad=output_angle,
                    ),
                },
            }
        )
    return samples


def _relative_quaternion(
    current: Sequence[float],
    initial: Sequence[float],
) -> tuple[float, float, float, float]:
    result = _quat_product(current, _quat_conjugate(initial))
    if result[0] < 0.0:
        return tuple(-item for item in result)
    return result


def _axis_angle_components(
    relative_quaternion: Sequence[float],
    axis_abs: Sequence[float],
) -> tuple[float, float]:
    w_value = max(-1.0, min(1.0, float(relative_quaternion[0])))
    vector = tuple(float(item) for item in relative_quaternion[1:])
    projected = _dot(vector, axis_abs)
    signed_axis = 2.0 * math.atan2(projected, w_value)
    perpendicular = _sub(vector, _scale(axis_abs, projected))
    off_axis = 2.0 * math.asin(max(0.0, min(1.0, _norm(perpendicular))))
    return signed_axis, off_axis


def _point_velocity(
    body: Any,
    point_abs: Sequence[float],
    *,
    tuple3: Any,
    angular_velocity: Any,
) -> tuple[float, float, float]:
    center = tuple3(body.GetPos())
    linear = tuple3(body.GetPosDt())
    omega = angular_velocity(body)
    return _add(linear, _cross(omega, _sub(point_abs, center)))


def _bearing_response(
    bearing: BicycleBearingSpec,
    *,
    anchor_abs: Sequence[float],
    home_anchor_abs: Sequence[float],
    anchor_velocity_abs: Sequence[float],
    current_axis_abs: Sequence[float],
    required_axis_abs: Sequence[float],
    angular_velocity_abs: Sequence[float],
) -> dict[str, Any]:
    displacement = _sub(anchor_abs, home_anchor_abs)
    translation_stiffness, translation_damping = (
        _bearing_translation_coefficients(bearing)
    )
    force = _clamp_vector(
        _add(
            _scale(
                _componentwise(translation_stiffness, displacement),
                -1.0,
            ),
            _scale(
                _componentwise(
                    translation_damping,
                    anchor_velocity_abs,
                ),
                -1.0,
            ),
        ),
        bearing.maximum_force_n,
    )
    alignment_cross = _cross(current_axis_abs, required_axis_abs)
    perpendicular_speed = _sub(
        angular_velocity_abs,
        _scale(
            required_axis_abs,
            _dot(angular_velocity_abs, required_axis_abs),
        ),
    )
    tilt_stiffness, tilt_damping = _bearing_off_axis_coefficients(bearing)
    torque = _clamp_vector(
        _add(
            _componentwise(tilt_stiffness, alignment_cross),
            _scale(
                _componentwise(tilt_damping, perpendicular_speed),
                -1.0,
            ),
        ),
        _bearing_total_torque_cap(bearing),
    )
    axis_error = math.acos(
        max(
            -1.0,
            min(1.0, abs(_dot(current_axis_abs, required_axis_abs))),
        )
    )
    return {
        "force_abs_n": force,
        "alignment_torque_abs_nm": torque,
        "anchor_drift_m": _norm(displacement),
        "axis_error_rad": axis_error,
        "translation_stiffness_xyz_n_m": translation_stiffness,
        "translation_damping_xyz_n_s_m": translation_damping,
        "off_axis_rotation_stiffness_xyz_nm_rad": tilt_stiffness,
        "off_axis_rotation_damping_xyz_nm_s_rad": tilt_damping,
    }


def _family_spec_without_collision(
    value: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return value
    return {
        key: item
        for key, item in value.items()
        if key != "collision_geometry"
    }


def _validate_zero_energy_initial_state(spec: Any) -> None:
    if not spec.body_dynamics:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BODY_DYNAMICS_MISSING",
            "the physical runtime requires typed body initial states",
        )
    body_specs = {body.body_id: body for body in spec.bodies}
    for dynamics in spec.body_dynamics:
        body = body_specs.get(dynamics.body_id)
        if (
            body is None
            or dynamics.mass_kg != body.mass_kg
            or tuple(dynamics.initial_position_abs_m)
            != tuple(body.center_abs_m)
            or tuple(dynamics.initial_orientation_wxyz)
            != tuple(body.rotation_abs_quat_wxyz)
            or any(
                value != 0.0
                for value in (
                    *dynamics.initial_linear_velocity_abs_m_s,
                    *dynamics.initial_angular_velocity_abs_rad_s,
                )
            )
        ):
            raise MechanismValidationError(
                "BICYCLE_CHAIN_INITIAL_STATE_INVALID",
                f"body {dynamics.body_id!r} changed its exact zero state",
            )
    transmission = spec.transmission
    if (
        transmission.initial_chain_error_m != 0.0
        or transmission.initial_chain_error_rate_m_s != 0.0
        or transmission.initial_stored_energy_j != 0.0
        or transmission.preload_allowed is not False
        or transmission.output_feedforward_allowed is not False
        or transmission.ideal_constraint_in_physical_branch is not False
        or spec.input_bearing.zero_state_stored_energy_j != 0.0
        or spec.output_bearing.zero_state_stored_energy_j != 0.0
    ):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_ZERO_INPUT_ENERGY_CREATED",
            "the physical runtime requires zero displacement, velocity, "
            "preload and stored energy",
        )


def _validate_load_program(spec: Any, load_case: Any) -> None:
    program = spec.task_program
    domain = spec.held_out_validation_domain
    if program is None or domain is None:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_TASK_PROGRAM_MISSING",
            "the runtime requires the complete typed task program and domain",
        )
    sample_interval = (
        program.integration_step_s * program.saved_sample_stride_steps
    )
    if (
        load_case.step_s != program.integration_step_s
        or load_case.sample_interval_s != sample_interval
        or program.saved_sample_stride_steps != 1
    ):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_TRACE_CADENCE_INVALID",
            "the r04 physical program must save every 0.00025 second step",
        )
    if load_case.program not in {
        "task_bicycle_powered_roundtrip",
        "bicycle_passive_release_return",
        BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
    }:
        if (
            load_case.program != "bicycle_zero_input_home_drift"
            or len(load_case.phases) != 1
            or load_case.phases[0].phase_id
            != "zero_state_no_input_drift"
            or load_case.phases[0].wrench_kind != "none"
            or load_case.phases[0].direction_abs is not None
            or load_case.phases[0].magnitude_si != 0.0
            or load_case.phases[0].duration_s != 0.60
        ):
            raise MechanismValidationError(
                "BICYCLE_CHAIN_TASK_PROGRAM_CHANGED",
                "the runtime received an unapproved bicycle-chain program",
            )
        return
    if len(load_case.phases) != len(program.phases):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            "the physical load omits part of the fixed four-phase program",
        )
    torques: list[float] = []
    expected_axis = tuple(spec.axis_abs)
    for actual, expected in zip(load_case.phases, program.phases):
        if (
            actual.phase_id != expected.phase_id
            or actual.wrench_kind != expected.wrench_kind
            or actual.duration_s != expected.duration_s
            or actual.stop_condition is not None
            or actual.target_coordinate is not None
            or actual.end_magnitude_si is not None
            or actual.branch_termination_policy is not None
        ):
            raise MechanismValidationError(
                "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
                "the physical load changed, shortened or made adaptive one "
                "of the fixed r04 phases",
            )
        if expected.wrench_kind == "torque":
            expected_direction = (
                tuple(-value for value in expected_axis)
                if load_case.program
                == BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM
                else expected_axis
            )
            if tuple(actual.direction_abs or ()) != expected_direction:
                raise MechanismValidationError(
                    "BICYCLE_CHAIN_TASK_INPUT_DIRECTION_INVALID",
                    "the crank torque direction differs from its typed program",
                )
            torques.append(float(actual.magnitude_si))
        elif actual.direction_abs is not None or actual.magnitude_si != 0.0:
            raise MechanismValidationError(
                "BICYCLE_CHAIN_RETURN_TASK_INPUT_NONZERO",
                "return phases require an exact-zero task input",
            )
    if len(torques) != 2 or torques[0] != torques[1]:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_INVALID",
            "one coupled torque value must drive engagement and measurement",
        )
    torque = torques[0]
    if load_case.program == BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM:
        if torque != BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM:
            raise MechanismValidationError(
                "BICYCLE_CHAIN_CRANK_PERTURBATION_INVALID",
                "the input-only crank perturbation changed its approved "
                "bounded magnitude",
            )
        return
    if not domain.minimum_torque_nm <= torque <= domain.maximum_torque_nm:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_INVALID",
            "the selected crank torque lies outside the closed approved domain",
        )


def _read_runtime_assembly_contract(
    collision: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[dict[str, Any], str]:
    contract_path = collision.get("assembly_contract")
    if not isinstance(contract_path, str):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "materialized geometry has no assembly-contract binding",
        )
    if Path(contract_path).is_absolute():
        resolved_contract = Path(contract_path).resolve()
        try:
            resolved_contract.relative_to(workspace)
        except ValueError as exc:
            raise MechanismValidationError(
                "BICYCLE_CHAIN_BINDING_INVALID",
                "assembly contract escaped the runtime workspace",
            ) from exc
    else:
        resolved_contract = ensure_inside(workspace, contract_path)
    contract = read_json(resolved_contract)
    if not isinstance(contract, dict):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "assembly contract must be an object",
        )
    contract_sha256 = sha256_file(resolved_contract)
    if collision.get("assembly_contract_sha256") != contract_sha256:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "materialized assembly-contract hash changed before runtime",
        )
    return contract, contract_sha256


def _actual_runtime_bindings(
    execution: ExecutionRequest,
    collision: Mapping[str, Any],
    *,
    spec: Any,
    workspace: Path,
) -> dict[str, Any]:
    """Resolve runtime identities from independent, intake-bound inventories."""

    from .task_motion_evidence import task_channel_registry_for_plan

    registry = task_channel_registry_for_plan(execution.plan)
    input_channels = list(registry.inputs.values())
    input_pose_channels = [
        item
        for item in registry.observables.values()
        if item.body_id in registry.input_body_ids
        and item.quantity == "pose_6d"
    ]
    output_pose_channels = [
        item
        for item in registry.observables.values()
        if item.body_id in registry.output_body_ids
        and item.quantity == "pose_6d"
    ]
    output_log_channels = [
        item
        for item in registry.observables.values()
        if item.body_id in registry.output_body_ids
        and item.quantity == "actuation_log"
    ]
    if not (
        len(registry.input_body_ids) == 1
        and len(registry.output_body_ids) == 1
        and len(input_channels) == 1
        and len(input_pose_channels) == 1
        and len(output_pose_channels) == 1
        and len(output_log_channels) == 1
    ):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "runtime registry must resolve one crank input and one rear output",
        )
    collision_bodies = collision.get("bodies")
    if not isinstance(collision_bodies, list):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "materialized collision inventory has no bodies",
        )
    typed_collision_bodies = [
        item for item in collision_bodies if isinstance(item, Mapping)
    ]
    fixed_entries = [
        item for item in typed_collision_bodies if item.get("fixed") is True
    ]
    if len(fixed_entries) != 1:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "materialized geometry must have one fixed body",
        )
    fixed_body_id = str(fixed_entries[0].get("body_id") or "")
    input_channel = input_channels[0]
    input_pose = input_pose_channels[0]
    output_pose = output_pose_channels[0]
    output_log = output_log_channels[0]
    if (
        input_pose.body_id != input_channel.body_id
        or input_pose.rigid_group_id != input_channel.rigid_group_id
        or input_pose.frame_id != input_channel.frame_id
        or output_log.body_id != output_pose.body_id
        or output_log.rigid_group_id != output_pose.rigid_group_id
        or output_log.frame_id != output_pose.frame_id
    ):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "input pose and output actuation channels must share the exact "
            "body, rigid group, and frame of their physical role",
        )
    contract, contract_sha256 = _read_runtime_assembly_contract(
        collision,
        workspace=workspace,
    )
    contract_binding = resolve_bicycle_contract_bindings(
        spec,
        contract,
        assembly_contract_sha256=contract_sha256,
    )

    collision_by_body: dict[str, Mapping[str, Any]] = {}
    for item in typed_collision_bodies:
        body_id = item.get("body_id")
        if (
            not isinstance(body_id, str)
            or body_id in collision_by_body
        ):
            raise MechanismValidationError(
                "BICYCLE_CHAIN_BINDING_INVALID",
                "collision bodies must have unique canonical identities",
            )
        collision_by_body[body_id] = item

    def collision_binding(
        body_id: str,
        *,
        channel_group_id: str | None = None,
    ) -> tuple[str, list[str]]:
        item = collision_by_body.get(body_id)
        if item is None:
            raise MechanismValidationError(
                "BICYCLE_CHAIN_BINDING_INVALID",
                f"runtime body {body_id!r} is missing from collision inventory",
            )
        group_ids = item.get("group_ids")
        source_solid_ids = item.get("source_solid_ids")
        if (
            not isinstance(group_ids, list)
            or len(group_ids) != 1
            or not isinstance(group_ids[0], str)
            or (
                channel_group_id is not None
                and group_ids[0] != channel_group_id
            )
            or not isinstance(source_solid_ids, list)
            or not source_solid_ids
            or any(not isinstance(value, str) for value in source_solid_ids)
        ):
            raise MechanismValidationError(
                "BICYCLE_CHAIN_BINDING_INVALID",
                f"runtime body {body_id!r} has an invalid group/solid binding",
            )
        return group_ids[0], list(source_solid_ids)

    fixed_group_ids = fixed_entries[0].get("group_ids")
    if not isinstance(fixed_group_ids, list) or len(fixed_group_ids) != 1:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "fixed collision body does not resolve one rigid group",
        )
    fixed_group_id, fixed_solids = collision_binding(fixed_body_id)
    input_group_id, input_solids = collision_binding(
        input_channel.body_id,
        channel_group_id=input_channel.rigid_group_id,
    )
    output_group_id, output_solids = collision_binding(
        output_pose.body_id,
        channel_group_id=output_pose.rigid_group_id,
    )
    roles = contract_binding["roles"]
    return {
        "task_id": execution.plan.task_ids[0],
        "fixed_body_id": fixed_body_id,
        "input_body_id": input_channel.body_id,
        "output_body_id": output_pose.body_id,
        "fixed_rigid_group_id": fixed_group_id,
        "input_rigid_group_id": input_group_id,
        "output_rigid_group_id": output_group_id,
        "fixed_source_solid_ids": fixed_solids,
        "input_source_solid_ids": input_solids,
        "output_source_solid_ids": output_solids,
        "fixed_part_id": roles["fixed"]["part_id"],
        "input_part_id": roles["input"]["part_id"],
        "output_part_id": roles["output"]["part_id"],
        "fixed_occurrence_id": roles["fixed"]["occurrence_id"],
        "input_occurrence_id": roles["input"]["occurrence_id"],
        "output_occurrence_id": roles["output"]["occurrence_id"],
        "input_frame_id": input_channel.frame_id,
        "output_frame_id": output_pose.frame_id,
        "input_port_id": input_channel.semantic_id,
        "input_observable_id": input_pose.semantic_id,
        "output_observable_id": output_pose.semantic_id,
        "actuation_observable_id": output_log.semantic_id,
        "input_axis_abs": list(input_channel.axis_abs or ()),
        "output_axis_abs": list(output_pose.axis_abs or ()),
        "assembly_contract_sha256": contract_sha256,
        "contract_role_binding_sha256": contract_binding[
            "binding_sha256"
        ],
    }


def simulate_bicycle_chain(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
    """Run a free-body, finite-support, geometry-gated bicycle drivetrain."""

    if execution.mechanism_plugin.worker_backend != BICYCLE_CHAIN_PLUGIN_ID:
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            "bicycle-chain runtime received a different mechanism backend",
        )
    if execution.branch != "physical":
        raise MechanismValidationError(
            "BICYCLE_CHAIN_IDEAL_DYNAMICS_NOT_REQUIRED",
            "approved r04 uses a deterministic ideal-reference trace; the "
            "runtime backend is reserved for the physical branch",
        )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    from .family_collision import (
        family_collision_geometry,
        make_family_body,
        trace_collision_thresholds,
    )
    from .worker import (
        _angular_velocity,
        _collision_progress,
        _contact_count,
        _contact_pair_counts,
        _make_material,
        _make_nsc_system,
        _pose,
        _tuple3,
        _write_trace_checkpoint,
        lane_execution_source_sha256,
    )

    spec = bicycle_chain_from_dict(
        _family_spec_without_collision(execution.adapter.family_spec)
    )
    _validate_zero_energy_initial_state(spec)
    _validate_load_program(spec, execution.load_case)
    collision = family_collision_geometry(
        execution.adapter,
        expected_bodies=spec.bodies,
    )
    if (
        collision["package_binding_sha256"]
        != execution.plan.package_binding_sha256
    ):
        raise MechanismValidationError(
            "FAMILY_COLLIDER_PACKAGE_BINDING_CHANGED",
            "bicycle collision geometry differs from the executed package",
        )
    runtime_contract, runtime_contract_sha256 = (
        _read_runtime_assembly_contract(
            collision,
            workspace=workspace,
        )
    )
    runtime_bindings = validate_runtime_bindings(
        spec,
        _actual_runtime_bindings(
            execution,
            collision,
            spec=spec,
            workspace=workspace,
        ),
        task_id=execution.plan.task_ids[0],
        assembly_contract=runtime_contract,
        assembly_contract_sha256=runtime_contract_sha256,
    )
    runtime_binding_sha256 = canonical_sha256(runtime_bindings)
    collision_binding_sha256 = canonical_sha256(collision)
    geometry = execution.functional_geometry_evidence
    if not isinstance(geometry, dict):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_FUNCTIONAL_GEOMETRY_MISSING",
            "physical execution did not receive the hash-bound CAD preflight evidence",
        )
    if geometry.get("passed") is not True:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_FUNCTIONAL_GEOMETRY_INVALID",
            "; ".join(geometry.get("failure_reasons") or ()),
        )
    moving_chain_path_contract = build_moving_chain_path_contract(
        spec,
        geometry["visible_chain_geometry"],
    )
    runtime_geometry_bindings_match = (
        geometry.get("contract_role_binding_sha256")
        == runtime_bindings["contract_role_binding_sha256"]
    )
    if not runtime_geometry_bindings_match:
        raise MechanismValidationError(
            "BICYCLE_CHAIN_BINDING_INVALID",
            "geometry and runtime resolved different contract role bindings",
        )
    fixed_component_refs = body_component_refs(
        collision,
        spec.fixed_body_id,
    )
    input_component_refs = body_component_refs(
        collision,
        spec.input_body_id,
    )
    output_component_refs = body_component_refs(
        collision,
        spec.output_body_id,
    )
    input_support_interface_id = (
        f"support-{spec.input_bearing.support_id}"
    )
    output_support_interface_id = (
        f"support-{spec.output_bearing.support_id}"
    )
    chain_interface_id = (
        "transmission-input-sprocket-chain-output-sprocket"
    )
    input_support_gate = bounded_support_gate(
        body_a_id=spec.fixed_body_id,
        body_b_id=spec.input_body_id,
        endpoint_a_component_refs=fixed_component_refs,
        endpoint_b_component_refs=input_component_refs,
        maximum_drift_m=spec.max_support_drift_m,
        coordinate_mode=RELATIVE_ANCHOR_DRIFT,
        maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
        template_id="bicycle_input_finite_bearing_v2",
        proof_source="approved_bicycle_chain_sprocket_drive_v2",
    )
    output_support_gate = bounded_support_gate(
        body_a_id=spec.fixed_body_id,
        body_b_id=spec.output_body_id,
        endpoint_a_component_refs=fixed_component_refs,
        endpoint_b_component_refs=output_component_refs,
        maximum_drift_m=spec.max_support_drift_m,
        coordinate_mode=RELATIVE_ANCHOR_DRIFT,
        maximum_axis_misalignment_rad=spec.max_off_axis_rotation_rad,
        template_id="bicycle_output_finite_bearing_v2",
        proof_source="approved_bicycle_chain_sprocket_drive_v2",
    )
    chain_gate = conditional_transmission_gate(
        geometry_plugin_id="bicycle_chain_pitch_geometry_v1",
        body_a_id=spec.input_body_id,
        body_b_id=spec.output_body_id,
        endpoint_a_component_refs=input_component_refs,
        endpoint_b_component_refs=output_component_refs,
        # The sprockets do not overlap, and their full collision geometry is
        # still useful.  This analytical force law therefore requests no
        # validator-owned local collider suppression.
        endpoint_a_local_collision_component_refs=(),
        endpoint_b_local_collision_component_refs=(),
        activation_coordinate="normalized_geometry_margin",
        enter_threshold_si=0.0,
        release_threshold_si=-1.0e-12,
        proof_source="authoritative_step_and_live_sprocket_pose",
        geometry_parameters={
            "static_geometry_gate_sha256": geometry["geometry_gate"][
                "gate_sha256"
            ],
            "maximum_axis_angle_error_rad": (
                spec.maximum_axis_angle_error_rad
            ),
            "maximum_axial_alignment_error_m": (
                spec.maximum_alignment_error_m
            ),
            "visible_chain_geometry": geometry[
                "visible_chain_geometry"
            ],
            "inactive_transmission_effort": "EXACT_ZERO",
        },
    )
    physical_interfaces = (
        interface_declaration(
            interface_id=input_support_interface_id,
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=spec.fixed_body_id,
            body_b_id=spec.input_body_id,
            model="finite_bearing_force_and_alignment_torque",
            source="built_in_bicycle_chain_runtime_v2",
            free_coordinates=("rotation_about_shaft_axis",),
            transmission_gate=input_support_gate,
        ),
        interface_declaration(
            interface_id=output_support_interface_id,
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=spec.fixed_body_id,
            body_b_id=spec.output_body_id,
            model="finite_bearing_force_and_alignment_torque",
            source="built_in_bicycle_chain_runtime_v2",
            free_coordinates=("rotation_about_shaft_axis",),
            transmission_gate=output_support_gate,
        ),
        interface_declaration(
            interface_id=chain_interface_id,
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id=spec.input_body_id,
            body_b_id=spec.output_body_id,
            model=(
                "bounded_geometry_gated_compliant_differential_tension"
            ),
            source="built_in_bicycle_chain_runtime_v2",
            transmission_gate=chain_gate,
        ),
    )
    physical_interface_policy = collision_policy_disclosure(
        physical_interfaces,
        branch="physical",
    )
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            collision,
        )
    )
    input_support_runtime = BoundedSupportRuntime(
        input_support_interface_id,
        input_support_gate,
    )
    output_support_runtime = BoundedSupportRuntime(
        output_support_interface_id,
        output_support_gate,
    )
    swept_envelope_preflight = geometry.get(
        "prescribed_swept_envelope_preflight"
    )
    if not isinstance(swept_envelope_preflight, Mapping):
        raise MechanismValidationError(
            "BICYCLE_CHAIN_SWEPT_ENVELOPE_MISSING",
            "physical execution did not receive the hash-bound swept-envelope preflight",
        )
    if swept_envelope_preflight.get("passed") is not True:
        first_event = swept_envelope_preflight.get("first_illegal_event")
        detail = (
            f"; first event={first_event}"
            if isinstance(first_event, Mapping)
            else ""
        )
        raise MechanismValidationError(
            "BICYCLE_CHAIN_SWEPT_ENVELOPE_INVALID",
            "the required coupled crank/rear-wheel motion envelope "
            f"intersects another rigid body{detail}",
        )

    system = _make_nsc_system(
        chrono,
        collision_envelope_m=float(collision["collision_envelope_m"]),
        collision_margin_m=float(collision["collision_margin_m"]),
    )
    solver = chrono.ChSolverMINRES()
    if hasattr(solver, "SetMaxIterations"):
        solver.SetMaxIterations(400)
    if hasattr(solver, "SetTolerance"):
        solver.SetTolerance(1.0e-10)
    system.SetSolver(solver)
    system.SetGravitationalAcceleration(
        chrono.ChVector3d(*execution.plan.gravity_abs_m_s2)
    )
    material = _make_material(chrono, spec.friction)
    bodies: dict[str, Any] = {}
    inertias: dict[str, list[float]] = {}
    initial_poses: dict[str, dict[str, Any]] = {}
    collision_audit: list[dict[str, Any]] = []
    for index, body_spec in enumerate(spec.bodies):
        inertia = _body_inertia_diagonal(spec, body_spec)
        body, audit = make_family_body(
            chrono,
            body_spec=body_spec,
            geometry=collision,
            workspace=workspace,
            material=material,
            inertia_xx=inertia,
            branch="physical",
        )
        body.GetCollisionModel().SetFamily(index + 1)
        audit["collision_family"] = index + 1
        _collision_progress(
            "system_add_body_begin",
            body_id=body_spec.body_id,
        )
        system.AddBody(body)
        _collision_progress(
            "system_add_body_complete",
            body_id=body_spec.body_id,
        )
        bodies[body_spec.body_id] = body
        inertias[body_spec.body_id] = list(inertia)
        initial_poses[body_spec.body_id] = _pose(body)
        collision_audit.append(audit)

    fixed = bodies[spec.fixed_body_id]
    crank = bodies[spec.input_body_id]
    rear = bodies[spec.output_body_id]
    if hasattr(system, "GetNumLinks") and int(system.GetNumLinks()) != 0:
        raise RuntimeError(
            "bicycle physical branch unexpectedly contains an ideal link"
        )

    input_anchor_local = _tuple3(
        crank.TransformPointParentToLocal(
            chrono.ChVector3d(*spec.input_origin_abs_m)
        )
    )
    output_anchor_local = _tuple3(
        rear.TransformPointParentToLocal(
            chrono.ChVector3d(*spec.output_origin_abs_m)
        )
    )
    input_axis_local = _tuple3(
        crank.GetRot().GetConjugate().Rotate(
            chrono.ChVector3d(*spec.input_frame.axis_abs)
        )
    )
    output_axis_local = _tuple3(
        rear.GetRot().GetConjugate().Rotate(
            chrono.ChVector3d(*spec.output_frame.axis_abs)
        )
    )
    input_pitch_center = geometry["input_pitch_surface"]["center_abs_m"]
    output_pitch_center = geometry["output_pitch_surface"]["center_abs_m"]
    input_pitch_local = _tuple3(
        crank.TransformPointParentToLocal(
            chrono.ChVector3d(*input_pitch_center)
        )
    )
    output_pitch_local = _tuple3(
        rear.TransformPointParentToLocal(
            chrono.ChVector3d(*output_pitch_center)
        )
    )

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    plugin_id = execution.mechanism_plugin.plugin_id

    def declaration(
        profile: str,
        kind: str,
        load_id: str,
        target_body: str,
        source_role: str,
        target_role: str,
        geometry_gate: str,
        *,
        release_required: bool = False,
        owner: str | None = None,
    ) -> Any:
        return standard_wrench_declaration(
            plugin_id=plugin_id,
            profile=profile,
            wrench_kind=kind,
            load_id=load_id,
            owner=plugin_id if owner is None else owner,
            target_body=target_body,
            source_role=source_role,
            target_role=target_role,
            geometry_gate=geometry_gate,
            active_interval="declared_load_case_phases",
            release_required=release_required,
        )

    task_input = create_body_torque(
        chrono,
        crank,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "task_input",
            "torque",
            "validator_crank_task_torque",
            spec.input_body_id,
            "declared_input_port",
            "declared_input",
            "load_case_phase_and_input_crank_frame",
            release_required=True,
        ),
    )
    chain_input = create_body_torque(
        chrono,
        crank,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "bicycle_chain_transmission",
            "torque",
            "finite_chain_input_torque",
            spec.input_body_id,
            "registered_chain_state",
            "declared_input",
            spec.geometry_gate.gate_id,
            release_required=True,
        ),
    )
    chain_output = create_body_torque(
        chrono,
        rear,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "bicycle_chain_transmission",
            "torque",
            "finite_chain_output_torque",
            spec.output_body_id,
            "registered_chain_state",
            "task_output",
            spec.geometry_gate.gate_id,
            release_required=True,
        ),
    )
    input_support_force = create_body_force(
        chrono,
        crank,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*spec.input_origin_abs_m),
        False,
        declaration=declaration(
            "bicycle_finite_support",
            "force",
            "plugin_crank_bearing_retention_force",
            spec.input_body_id,
            "finite_bearing_support",
            "declared_input",
            "input_crank_finite_bearing_envelope",
            release_required=True,
        ),
    )
    input_support_torque = create_body_torque(
        chrono,
        crank,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "bicycle_finite_support",
            "torque",
            "plugin_crank_bearing_alignment_torque",
            spec.input_body_id,
            "finite_bearing_support",
            "declared_input",
            "input_crank_finite_bearing_envelope",
            release_required=True,
        ),
    )
    output_support_force = create_body_force(
        chrono,
        rear,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        chrono.ChVector3d(*spec.output_origin_abs_m),
        False,
        declaration=declaration(
            "bicycle_finite_support",
            "force",
            "plugin_rear_bearing_retention_force",
            spec.output_body_id,
            "finite_bearing_support",
            "task_output",
            "rear_wheel_finite_bearing_envelope",
            release_required=True,
        ),
    )
    output_support_torque = create_body_torque(
        chrono,
        rear,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "bicycle_finite_support",
            "torque",
            "plugin_rear_bearing_alignment_torque",
            spec.output_body_id,
            "finite_bearing_support",
            "task_output",
            "rear_wheel_finite_bearing_envelope",
            release_required=True,
        ),
    )
    centering_return = create_body_torque(
        chrono,
        crank,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "bicycle_finite_support",
            "torque",
            "plugin_crank_centering_return_torque",
            spec.input_body_id,
            "finite_passive_centering",
            "declared_input",
            "input_bearing_present",
            release_required=True,
        ),
    )
    rear_drag = create_body_torque(
        chrono,
        rear,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=declaration(
            "bicycle_finite_support",
            "torque",
            "plugin_rear_bearing_spin_drag_torque",
            spec.output_body_id,
            "bounded_bearing_spin_drag",
            "task_output",
            "output_bearing_present",
            release_required=True,
        ),
    )
    for load in (
        task_input,
        chain_input,
        chain_output,
        input_support_force,
        input_support_torque,
        output_support_force,
        output_support_torque,
        centering_return,
        rear_drag,
    ):
        add_ledgered_load(load_container, load)

    input_angle = 0.0
    output_angle = 0.0
    time_s = 0.0
    task_torque_nm = 0.0
    current_state: dict[str, Any] = {}
    dynamics_step_count = 0
    chain_interface_evaluation_count = 0
    chain_interface_state_path: list[str] = []
    chain_max_inactive_abs_transmitted_effort_si = 0.0

    def update_state_and_loads() -> None:
        nonlocal chain_interface_evaluation_count
        nonlocal chain_max_inactive_abs_transmitted_effort_si
        nonlocal current_state
        input_anchor = _tuple3(
            crank.TransformPointLocalToParent(
                chrono.ChVector3d(*input_anchor_local)
            )
        )
        output_anchor = _tuple3(
            rear.TransformPointLocalToParent(
                chrono.ChVector3d(*output_anchor_local)
            )
        )
        input_axis = _tuple3(
            crank.GetRot().Rotate(chrono.ChVector3d(*input_axis_local))
        )
        output_axis = _tuple3(
            rear.GetRot().Rotate(chrono.ChVector3d(*output_axis_local))
        )
        input_speed = _dot(_angular_velocity(crank), spec.axis_abs)
        output_speed = _dot(_angular_velocity(rear), spec.axis_abs)
        input_anchor_velocity = _point_velocity(
            crank,
            input_anchor,
            tuple3=_tuple3,
            angular_velocity=_angular_velocity,
        )
        output_anchor_velocity = _point_velocity(
            rear,
            output_anchor,
            tuple3=_tuple3,
            angular_velocity=_angular_velocity,
        )
        input_support = _bearing_response(
            spec.input_bearing,
            anchor_abs=input_anchor,
            home_anchor_abs=spec.input_origin_abs_m,
            anchor_velocity_abs=input_anchor_velocity,
            current_axis_abs=input_axis,
            required_axis_abs=spec.axis_abs,
            angular_velocity_abs=_angular_velocity(crank),
        )
        output_support = _bearing_response(
            spec.output_bearing,
            anchor_abs=output_anchor,
            home_anchor_abs=spec.output_origin_abs_m,
            anchor_velocity_abs=output_anchor_velocity,
            current_axis_abs=output_axis,
            required_axis_abs=spec.axis_abs,
            angular_velocity_abs=_angular_velocity(rear),
        )
        actual_input_pitch_center = _tuple3(
            crank.TransformPointLocalToParent(
                chrono.ChVector3d(*input_pitch_local)
            )
        )
        actual_output_pitch_center = _tuple3(
            rear.TransformPointLocalToParent(
                chrono.ChVector3d(*output_pitch_local)
            )
        )
        axial_alignment = abs(
            _dot(
                _sub(
                    actual_input_pitch_center,
                    actual_output_pitch_center,
                ),
                spec.axis_abs,
            )
        )
        axis_angle = math.acos(
            max(-1.0, min(1.0, abs(_dot(input_axis, output_axis))))
        )
        gate_active = chain_engagement_active(
            spec,
            bindings_match=runtime_geometry_bindings_match,
            input_pitch_radius_m=float(
                geometry["input_pitch_surface"]["major_radius_m"]
            ),
            output_pitch_radius_m=float(
                geometry["output_pitch_surface"]["major_radius_m"]
            ),
            axis_angle_error_rad=axis_angle,
            axial_alignment_error_m=axial_alignment,
        )
        coordinate_error = (
            spec.input_pitch_radius_m * input_angle
            - spec.output_pitch_radius_m * output_angle
        )
        coordinate_rate = (
            spec.input_pitch_radius_m * input_speed
            - spec.output_pitch_radius_m * output_speed
        )
        tension, input_reaction, output_reaction = (
            chain_reaction_torques(
                spec,
                coordinate_error_m=coordinate_error,
                coordinate_rate_m_s=coordinate_rate,
                gate_active=gate_active,
            )
        )
        return_torque = (
            -spec.input_bearing.primary_axis_centering_stiffness_nm_rad
            * (
                input_angle
                - spec.input_bearing.primary_axis_home_rad
            )
            - spec.input_bearing.primary_axis_damping_nm_s_rad
            * input_speed
        )
        primary_cap = _bearing_primary_torque_cap(spec.input_bearing)
        if primary_cap is None:
            raise MechanismValidationError(
                "BICYCLE_CHAIN_PRIMARY_AXIS_CAP_MISSING",
                "the input finite support needs its distinct primary-axis cap",
            )
        return_torque = _clamp_scalar(return_torque, primary_cap)
        drag_torque = (
            -spec.output_bearing.primary_axis_centering_stiffness_nm_rad
            * (
                output_angle
                - spec.output_bearing.primary_axis_home_rad
            )
            - spec.output_bearing.primary_axis_damping_nm_s_rad
            * output_speed
        )
        output_primary_cap = _bearing_primary_torque_cap(
            spec.output_bearing
        )
        if output_primary_cap is not None:
            drag_torque = _clamp_scalar(
                drag_torque,
                output_primary_cap,
            )
        input_total_torque = _clamp_vector(
            _add(
                input_support["alignment_torque_abs_nm"],
                _scale(spec.axis_abs, return_torque),
            ),
            _bearing_total_torque_cap(spec.input_bearing),
        )
        return_torque = _dot(input_total_torque, spec.axis_abs)
        input_support["alignment_torque_abs_nm"] = _sub(
            input_total_torque,
            _scale(spec.axis_abs, return_torque),
        )
        output_total_torque = _clamp_vector(
            _add(
                output_support["alignment_torque_abs_nm"],
                _scale(spec.axis_abs, drag_torque),
            ),
            _bearing_total_torque_cap(spec.output_bearing),
        )
        drag_torque = _dot(output_total_torque, spec.axis_abs)
        output_support["alignment_torque_abs_nm"] = _sub(
            output_total_torque,
            _scale(spec.axis_abs, drag_torque),
        )
        input_support_state = input_support_runtime.evaluate(
            time_s=time_s,
            anchor_a_abs_m=spec.input_origin_abs_m,
            anchor_b_abs_m=input_anchor,
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=input_anchor_velocity,
            # BoundedSupportRuntime is a compliant force-path gate. Keep its
            # scalar effort in newtons; alignment torque is recorded below as
            # a separate N*m channel instead of comparing unlike units.
            requested_effort_si=_norm(input_support["force_abs_n"]),
            axis_misalignment_rad=input_support["axis_error_rad"],
        )
        output_support_state = output_support_runtime.evaluate(
            time_s=time_s,
            anchor_a_abs_m=spec.output_origin_abs_m,
            anchor_b_abs_m=output_anchor,
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=output_anchor_velocity,
            requested_effort_si=_norm(output_support["force_abs_n"]),
            axis_misalignment_rad=output_support["axis_error_rad"],
        )
        for support_state in (input_support_state, output_support_state):
            observation = support_state.get("observation")
            support_state["constraints_satisfied"] = bool(
                isinstance(observation, Mapping)
                and observation.get("constraints_satisfied") is True
            )
        input_support_state["requested_support_torque_nm"] = _norm(
            input_total_torque
        )
        input_support_state["transmitted_support_torque_nm"] = (
            _norm(input_total_torque)
            if input_support_state["transmission_active"]
            else 0.0
        )
        output_support_state["requested_support_torque_nm"] = _norm(
            output_total_torque
        )
        output_support_state["transmitted_support_torque_nm"] = (
            _norm(output_total_torque)
            if output_support_state["transmission_active"]
            else 0.0
        )
        if not input_support_state["transmission_active"]:
            input_support["force_abs_n"] = (0.0, 0.0, 0.0)
            input_support["alignment_torque_abs_nm"] = (0.0, 0.0, 0.0)
            input_total_torque = (0.0, 0.0, 0.0)
            return_torque = 0.0
        if not output_support_state["transmission_active"]:
            output_support["force_abs_n"] = (0.0, 0.0, 0.0)
            output_support["alignment_torque_abs_nm"] = (0.0, 0.0, 0.0)
            output_total_torque = (0.0, 0.0, 0.0)
            drag_torque = 0.0
        normalized_geometry_margin = min(
            1.0
            - axis_angle
            / max(spec.maximum_axis_angle_error_rad, 1.0e-30),
            1.0
            - axial_alignment
            / max(spec.maximum_alignment_error_m, 1.0e-30),
        )
        chain_state_name = "ENGAGED" if gate_active else "INACTIVE"
        chain_transmitted_effort = abs(tension) if gate_active else 0.0
        chain_interface_evaluation_count += 1
        if (
            not chain_interface_state_path
            or chain_interface_state_path[-1] != chain_state_name
        ):
            chain_interface_state_path.append(chain_state_name)
        if not gate_active:
            chain_max_inactive_abs_transmitted_effort_si = max(
                chain_max_inactive_abs_transmitted_effort_si,
                chain_transmitted_effort,
            )
        chain_interface_state = {
            "interface_id": chain_interface_id,
            "state": chain_state_name,
            "transmission_active": gate_active,
            "requested_effort_si": abs(tension),
            "transmitted_effort_si": chain_transmitted_effort,
            "constraints_satisfied": gate_active,
            "activation_coordinate_si": normalized_geometry_margin,
            "geometry_gate_sha256": geometry["geometry_gate"][
                "gate_sha256"
            ],
        }
        input_support_force.SetApplicationPoint(
            chrono.ChVector3d(*input_anchor),
            False,
        )
        output_support_force.SetApplicationPoint(
            chrono.ChVector3d(*output_anchor),
            False,
        )
        task_input.SetTorque(
            chrono.ChVector3d(*_scale(spec.axis_abs, task_torque_nm)),
            False,
        )
        chain_input.SetTorque(
            chrono.ChVector3d(*_scale(spec.axis_abs, input_reaction)),
            False,
        )
        chain_output.SetTorque(
            chrono.ChVector3d(*_scale(spec.axis_abs, output_reaction)),
            False,
        )
        input_support_force.SetForce(
            chrono.ChVector3d(*input_support["force_abs_n"]),
            False,
        )
        input_support_torque.SetTorque(
            chrono.ChVector3d(
                *input_support["alignment_torque_abs_nm"]
            ),
            False,
        )
        output_support_force.SetForce(
            chrono.ChVector3d(*output_support["force_abs_n"]),
            False,
        )
        output_support_torque.SetTorque(
            chrono.ChVector3d(
                *output_support["alignment_torque_abs_nm"]
            ),
            False,
        )
        centering_return.SetTorque(
            chrono.ChVector3d(*_scale(spec.axis_abs, return_torque)),
            False,
        )
        rear_drag.SetTorque(
            chrono.ChVector3d(*_scale(spec.axis_abs, drag_torque)),
            False,
        )
        state_payload = {
            "time_s": time_s,
            "input_angle_rad": input_angle,
            "output_angle_rad": output_angle,
            "input_speed_rad_s": input_speed,
            "output_speed_rad_s": output_speed,
            "coordinate_error_m": coordinate_error,
            "coordinate_rate_m_s": coordinate_rate,
            "tension_n": tension,
            "gate_active": gate_active,
            "axis_angle_error_rad": axis_angle,
            "axial_alignment_error_m": axial_alignment,
        }
        state_id = canonical_sha256(state_payload)
        current_state = {
            **state_payload,
            "state_id": state_id,
            "input_reaction_nm": input_reaction,
            "output_reaction_nm": output_reaction,
            "return_torque_nm": return_torque,
            "drag_torque_nm": drag_torque,
            "input_support": input_support,
            "output_support": output_support,
            "input_total_support_torque_abs_nm": input_total_torque,
            "output_total_support_torque_abs_nm": output_total_torque,
            "conditional_interface_states": [
                input_support_state,
                output_support_state,
                chain_interface_state,
            ],
            "geometry_gate_sha256": geometry["geometry_gate"][
                "gate_sha256"
            ],
        }

    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []

    def record(phase_id: str) -> None:
        input_support = current_state["input_support"]
        output_support = current_state["output_support"]
        support_drifts = {
            spec.input_bearing.support_id: input_support[
                "anchor_drift_m"
            ],
            spec.output_bearing.support_id: output_support[
                "anchor_drift_m"
            ],
        }
        support_forces = {
            spec.input_bearing.support_id: _norm(
                input_support["force_abs_n"]
            ),
            spec.output_bearing.support_id: _norm(
                output_support["force_abs_n"]
            ),
        }
        support_torques = {
            spec.input_bearing.support_id: _norm(
                current_state["input_total_support_torque_abs_nm"]
            ),
            spec.output_bearing.support_id: _norm(
                current_state["output_total_support_torque_abs_nm"]
            ),
        }
        context = {
            "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
            "plugin_instance_id": (
                f"{BICYCLE_CHAIN_PLUGIN_ID}:single_instance"
            ),
            "state_id": current_state["state_id"],
            "solver_time_s": time_s,
            "geometry_gate_sha256": current_state[
                "geometry_gate_sha256"
            ],
            "geometry_gate_id": spec.geometry_gate.gate_id,
            "gate_active": current_state["gate_active"],
            "runtime_binding_sha256": runtime_binding_sha256,
            "contract_role_binding_sha256": runtime_bindings[
                "contract_role_binding_sha256"
            ],
            "collision_binding_sha256": collision_binding_sha256,
            "assembly_contract_sha256": collision[
                "assembly_contract_sha256"
            ],
            "assembly_sha256": collision["assembly_sha256"],
            "collision_schema": collision["schema"],
            "collision_component_count": collision["component_count"],
            "fixed_body_id": runtime_bindings["fixed_body_id"],
            "input_body_id": runtime_bindings["input_body_id"],
            "output_body_id": runtime_bindings["output_body_id"],
            "input_port_id": runtime_bindings["input_port_id"],
            "output_observable_id": runtime_bindings[
                "output_observable_id"
            ],
        }
        samples.append(
            {
                "time_s": time_s,
                "phase_id": phase_id,
                "input_crank_angle_rad": input_angle,
                "rear_drive_wheel_angle_rad": output_angle,
                "input_crank_speed_rad_s": current_state[
                    "input_speed_rad_s"
                ],
                "rear_drive_wheel_speed_rad_s": current_state[
                    "output_speed_rad_s"
                ],
                "chain_coordinate_error_m": current_state[
                    "coordinate_error_m"
                ],
                "chain_coordinate_rate_m_s": current_state[
                    "coordinate_rate_m_s"
                ],
                "chain_differential_tension_n": current_state[
                    "tension_n"
                ],
                "chain_input_reaction_torque_nm": current_state[
                    "input_reaction_nm"
                ],
                "chain_output_reaction_torque_nm": current_state[
                    "output_reaction_nm"
                ],
                "task_input_torque_nm": task_torque_nm,
                "crank_centering_return_torque_nm": current_state[
                    "return_torque_nm"
                ],
                "input_bearing_primary_axis_torque_nm": current_state[
                    "return_torque_nm"
                ],
                "input_bearing_primary_axis_torque_cap_nm": (
                    _bearing_primary_torque_cap(spec.input_bearing)
                ),
                "rear_bearing_drag_torque_nm": current_state[
                    "drag_torque_nm"
                ],
                "chain_engagement_gate_active": current_state[
                    "gate_active"
                ],
                "chain_axis_angle_error_rad": current_state[
                    "axis_angle_error_rad"
                ],
                "chain_axial_alignment_error_m": current_state[
                    "axial_alignment_error_m"
                ],
                "support_drifts_m": support_drifts,
                "support_forces_n": support_forces,
                "support_torques_nm": support_torques,
                "max_support_drift_m": max(support_drifts.values()),
                "max_off_axis_rotation_rad": max(
                    input_support["axis_error_rad"],
                    output_support["axis_error_rad"],
                ),
                "max_support_force_n": max(support_forces.values()),
                "max_support_torque_nm": max(
                    support_torques.values()
                ),
                "input_bearing_total_support_torque_nm": support_torques[
                    spec.input_bearing.support_id
                ],
                "rear_bearing_total_support_torque_nm": support_torques[
                    spec.output_bearing.support_id
                ],
                "chain_stored_energy_j": (
                    0.5
                    * spec.chain_stiffness_n_m
                    * current_state["coordinate_error_m"]
                    * current_state["coordinate_error_m"]
                ),
                "moving_chain_path_state": build_moving_chain_path_state(
                    moving_chain_path_contract,
                    input_angle_rad=input_angle,
                    output_angle_rad=output_angle,
                    coordinate_error_m=current_state[
                        "coordinate_error_m"
                    ],
                ),
                "input_bearing_primary_axis_stored_energy_j": (
                    0.5
                    * spec.input_bearing.primary_axis_centering_stiffness_nm_rad
                    * input_angle
                    * input_angle
                ),
                "chain_wrench_context": context,
                "conditional_interface_states": [
                    dict(item)
                    for item in current_state[
                        "conditional_interface_states"
                    ]
                ],
                "moving_pose": _pose(rear),
                "body_poses": {
                    body_id: _pose(body)
                    for body_id, body in bodies.items()
                },
                "contact_count": _contact_count(system),
                "contact_pair_counts": _contact_pair_counts(
                    chrono,
                    system,
                ),
            }
        )
        # This activation worker is intentionally non-resumable.  Keep the
        # full-cadence evidence in memory and publish checkpoints only at
        # completed phase boundaries below; repeatedly rewriting the growing
        # JSON trace is quadratic work.

    # Initial evidence is recorded before any externally powered phase.
    initial_state_payload = {
        "time_s": 0.0,
        "input_angle_rad": 0.0,
        "output_angle_rad": 0.0,
        "input_speed_rad_s": 0.0,
        "output_speed_rad_s": 0.0,
        "coordinate_error_m": 0.0,
        "coordinate_rate_m_s": 0.0,
        "tension_n": 0.0,
        "gate_active": bool(
            geometry["geometry_gate"]["initially_active"]
        ),
        "axis_angle_error_rad": 0.0,
        "axial_alignment_error_m": geometry["axial_alignment_error_m"],
    }
    current_state = {
        **initial_state_payload,
        "state_id": canonical_sha256(initial_state_payload),
        "input_reaction_nm": 0.0,
        "output_reaction_nm": 0.0,
        "return_torque_nm": 0.0,
        "drag_torque_nm": 0.0,
        "input_support": {
            "force_abs_n": (0.0, 0.0, 0.0),
            "alignment_torque_abs_nm": (0.0, 0.0, 0.0),
            "anchor_drift_m": 0.0,
            "axis_error_rad": 0.0,
        },
        "output_support": {
            "force_abs_n": (0.0, 0.0, 0.0),
            "alignment_torque_abs_nm": (0.0, 0.0, 0.0),
            "anchor_drift_m": 0.0,
            "axis_error_rad": 0.0,
        },
        "input_total_support_torque_abs_nm": (0.0, 0.0, 0.0),
        "output_total_support_torque_abs_nm": (0.0, 0.0, 0.0),
        "conditional_interface_states": [
            {
                "interface_id": input_support_interface_id,
                "state": "ENGAGED",
                "transmission_active": True,
                "requested_effort_si": 0.0,
                "transmitted_effort_si": 0.0,
                "requested_support_torque_nm": 0.0,
                "transmitted_support_torque_nm": 0.0,
                "constraints_satisfied": True,
            },
            {
                "interface_id": output_support_interface_id,
                "state": "ENGAGED",
                "transmission_active": True,
                "requested_effort_si": 0.0,
                "transmitted_effort_si": 0.0,
                "requested_support_torque_nm": 0.0,
                "transmitted_support_torque_nm": 0.0,
                "constraints_satisfied": True,
            },
            {
                "interface_id": chain_interface_id,
                "state": "ENGAGED",
                "transmission_active": True,
                "requested_effort_si": 0.0,
                "transmitted_effort_si": 0.0,
                "constraints_satisfied": True,
                "activation_coordinate_si": 1.0,
                "geometry_gate_sha256": geometry["geometry_gate"][
                    "gate_sha256"
                ],
            },
        ],
        "geometry_gate_sha256": geometry["geometry_gate"]["gate_sha256"],
    }
    record("initial")
    next_sample_s = execution.load_case.sample_interval_s
    first_step = True
    phases = tuple(execution.load_case.phases)
    for phase_index, phase in enumerate(phases):
        transition_current_wrench_phase(phase.phase_id)
        if phase.wrench_kind not in {"torque", "none"}:
            raise MechanismValidationError(
                "BICYCLE_CHAIN_LOAD_KIND_UNSUPPORTED",
                "bicycle drivetrain accepts crank torque or an exact zero-input phase",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        task_torque_nm = (
            _dot(direction, spec.axis_abs) * phase.magnitude_si
            if phase.wrench_kind == "torque"
            else 0.0
        )
        validate_task_input_target(
            spec,
            target_body_id=spec.input_body_id,
            phase_id=phase.phase_id,
            torque_nm=task_torque_nm,
        )
        # The previous step's post-update already installed feedback loads for
        # the next step.  Recompute once at a phase boundary because only the
        # phase-owned task torque changed; doing it again at every loop head
        # duplicated every state query and ledger mutation.
        update_state_and_loads()
        elapsed = 0.0
        while elapsed < phase.duration_s - 1.0e-12:
            step = min(
                execution.load_case.step_s,
                phase.duration_s - elapsed,
            )
            if first_step:
                _collision_progress(
                    "first_dynamics_step_begin",
                    capsule_id=execution.plan.capsule_id,
                )
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            if first_step:
                _collision_progress(
                    "first_dynamics_step_complete",
                    capsule_id=execution.plan.capsule_id,
                )
                first_step = False
            elapsed += step
            time_s += step
            input_angle += (
                _dot(_angular_velocity(crank), spec.axis_abs) * step
            )
            output_angle += (
                _dot(_angular_velocity(rear), spec.axis_abs) * step
            )
            update_state_and_loads()
            if time_s + 1.0e-12 >= next_sample_s:
                record(phase.phase_id)
                while next_sample_s <= time_s + 1.0e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if dynamics_step_count % 1000 == 0:
                _collision_progress(
                    "dynamics_progress",
                    capsule_id=execution.plan.capsule_id,
                    phase_id=phase.phase_id,
                    step_count=dynamics_step_count,
                    simulated_time_s=time_s,
                )
        if samples[-1]["time_s"] < time_s - 1.0e-12:
            record(phase.phase_id)
        task_torque_nm = 0.0
        task_input.SetTorque(
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
        )
        executed_phases.append(
            {
                "phase_id": phase.phase_id,
                "wrench_kind": phase.wrench_kind,
                "direction_abs": (
                    None
                    if phase.direction_abs is None
                    else list(phase.direction_abs)
                ),
                "magnitude_si": phase.magnitude_si,
                "duration_s": elapsed,
            }
        )
        # Checkpoints are non-authoritative and this activation worker is not
        # resumable.  Retain early phase-boundary evidence for timeout review,
        # but do not serialise the two large return phases before immediately
        # writing the complete trace.
        if phase_index < min(2, len(phases) - 1):
            _write_trace_checkpoint(
                execution,
                samples,
                executed_phases=executed_phases,
            )

    zero = chrono.ChVector3d(0.0, 0.0, 0.0)
    # Every release-required external wrench is explicitly released.
    task_input.SetTorque(zero, False)
    chain_input.SetTorque(zero, False)
    chain_output.SetTorque(zero, False)
    input_support_force.SetForce(zero, False)
    input_support_torque.SetTorque(zero, False)
    output_support_force.SetForce(zero, False)
    output_support_torque.SetTorque(zero, False)
    centering_return.SetTorque(zero, False)
    rear_drag.SetTorque(zero, False)
    set_current_wrench_phase(None)
    executed_load_sha256 = canonical_sha256(executed_phases)
    conditional_interface_runtime = conditional_runtime_summary(
        (input_support_runtime, output_support_runtime),
        physics_step_count=dynamics_step_count,
        additional_interface_summaries=(
            {
                "interface_id": chain_interface_id,
                "evaluation_count": chain_interface_evaluation_count,
                "state_path": list(chain_interface_state_path),
                "max_inactive_abs_transmitted_effort_si": (
                    chain_max_inactive_abs_transmitted_effort_si
                ),
                "loss_policy": "continue_with_exact_zero_effort",
            },
        ),
    )
    chrono_version = (
        str(chrono.GetChronoVersion())
        if hasattr(chrono, "GetChronoVersion")
        else None
    )
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": "physical",
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
        "package_id": execution.plan.package_id,
        "design_version": execution.plan.design_version,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(
            execution.load_case.to_dict()
        ),
        "engine": {
            "name": "PyChrono",
            "version": chrono_version,
            "system": "ChSystemNSC",
            "solver": "ChSolverMINRES",
            "mechanism_plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
            "worker_backend": BICYCLE_CHAIN_PLUGIN_ID,
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(
                execution.plan.gravity_abs_m_s2
            ),
            "chrono_link_count": 0,
            "ideal_links": [],
            "internal_ideal_joint_count": 0,
            "chain_model": (
                "bounded_geometry_gated_compliant_differential_tension"
            ),
        },
        "branch_semantics": {
            "same_mass_kg": sum(
                item.mass_kg for item in spec.bodies if not item.fixed
            ),
            "same_inertia_xx_kg_m2": inertias,
            "body_inertia_diagonal_kg_m2": inertias,
            "same_initial_pose": initial_poses,
            "replaced_contacts_disabled": False,
            "pose_setters_used": False,
            "direct_output_actuation_used": False,
            "input_only_crank_perturbation_used": (
                execution.load_case.program
                == BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM
            ),
            "direct_rear_diagnostic_wrench_used": False,
            "task_input_body_id": spec.input_body_id,
            "unique_task_output_body_id": spec.output_body_id,
            "physical_internal_ideal_joints_forbidden": True,
            "support_mode": (
                "finite_bearing_force_and_alignment_torque"
            ),
            "family_spec_sha256": canonical_sha256(spec.to_dict()),
            "collision_geometry_sha256": collision_binding_sha256,
            "runtime_bindings": runtime_bindings,
            "runtime_binding_sha256": runtime_binding_sha256,
            "contract_role_binding_sha256": runtime_bindings[
                "contract_role_binding_sha256"
            ],
            "assembly_contract_sha256": collision[
                "assembly_contract_sha256"
            ],
            "visible_chain_declared": geometry[
                "visible_chain_geometry"
            ]["declared"],
            "visible_chain_geometry_bound_to_finite_transmission_model": (
                geometry["visible_chain_geometry"]["passed"]
                and moving_chain_path_contract[
                    "source_visible_chain_geometry_sha256"
                ]
                == canonical_sha256(geometry["visible_chain_geometry"])
            ),
            "moving_chain_path_evidence_bound_to_force_coordinate": True,
            "fixed_spatial_sprocket_flex_zones_used": True,
            "individual_link_tooth_contact_simulated": False,
            "physical_interface_policy": physical_interface_policy,
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": (
                conditional_interface_runtime
            ),
            "body_inertias_are_typed_family_values": True,
            "initial_state_zero_energy_verified": True,
            "prescribed_swept_envelope_preflight": (
                swept_envelope_preflight
            ),
            "task_program_sha256": canonical_sha256(
                asdict(spec.task_program)
            ),
            "held_out_validation_domain_sha256": canonical_sha256(
                asdict(spec.held_out_validation_domain)
            ),
        },
        "functional_geometry": {
            "status": geometry["status"],
            "input_pitch_radius_m": spec.input_pitch_radius_m,
            "output_pitch_radius_m": spec.output_pitch_radius_m,
            "signed_output_per_input": spec.signed_output_per_input,
            "axial_alignment_error_m": geometry[
                "axial_alignment_error_m"
            ],
            "geometry_gate_sha256": geometry["geometry_gate"][
                "gate_sha256"
            ],
            "contract_role_binding_sha256": geometry[
                "contract_role_binding_sha256"
            ],
            "input_pitch_surface": geometry["input_pitch_surface"],
            "output_pitch_surface": geometry["output_pitch_surface"],
            "visible_chain_geometry": geometry[
                "visible_chain_geometry"
            ],
            "required_input_step_solid_id": (
                spec.input_sprocket.step_solid_id
            ),
            "required_output_step_solid_id": (
                spec.output_sprocket.step_solid_id
            ),
            "required_input_toroidal_minor_radius_m": (
                spec.input_sprocket.toroidal_minor_radius_m
            ),
            "required_output_toroidal_minor_radius_m": (
                spec.output_sprocket.toroidal_minor_radius_m
            ),
            "proof_source": "authoritative_step_toroidal_surfaces",
        },
        "moving_chain_path_contract": moving_chain_path_contract,
        "collision_geometry": {
            "mode": (
                "intake-bound STEP per-solid body-local collision components"
            ),
            **trace_collision_thresholds(collision),
            "schema": collision["schema"],
            "source": collision["source"],
            "assembly_file": collision["assembly_file"],
            "assembly_sha256": collision["assembly_sha256"],
            "assembly_contract_sha256": collision[
                "assembly_contract_sha256"
            ],
            "package_binding_sha256": collision[
                "package_binding_sha256"
            ],
            "component_count": collision["component_count"],
            "bodies": collision_audit,
            "contact_enabled": True,
            "physical_interface_policy": physical_interface_policy,
            "collision_filter_policy": (
                "all_inter_body_collision_pairs_enabled"
            ),
            "whole_body_box_proxy_used": False,
            "whole_body_convex_hull_used": False,
            "convex_hull_used": False,
            "joint_count": 0,
            "loop_count": 1,
            "input_pitch_radius_m": spec.input_pitch_radius_m,
            "output_pitch_radius_m": spec.output_pitch_radius_m,
            "friction_assumption": spec.friction,
        },
        "axis_abs": list(spec.axis_abs),
        "joint_origin_abs_m": list(spec.input_origin_abs_m),
        "input_point_abs_m": list(spec.input_origin_abs_m),
        "observable_point_initial_abs_m": list(
            spec.output_origin_abs_m
        ),
        "characteristic_length_m": (
            execution.plan.characteristic_length_m
        ),
        "executed_phases": executed_phases,
        "executed_load_sha256": executed_load_sha256,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "manufacturing_certification": False,
            "strength_fatigue_or_chain_life": False,
            "tooth_contact_certified": False,
            "individual_link_tooth_contact_simulated": False,
            "full_flexible_body_chain_physics_simulated": False,
            "startup_transient_hidden": False,
            "engagement_motion_credited": False,
            "engagement_ratio_credited": False,
            "preload_used": False,
            "output_feedforward_used": False,
            "dynamic_response_certified": False,
        },
    }
