from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Mapping, Sequence

from .adapters import adapter_from_dict, get_adapter
from .bicycle_chain_spec import (
    BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
    BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
    BICYCLE_CHAIN_PLUGIN_ID,
    BicycleChainSpec,
    bicycle_chain_from_dict,
)
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _fail("BICYCLE_CHAIN_TRACE_NUMBER_INVALID", f"{label} must be finite")
    return float(value)


def _package_family_spec(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return value
    return {
        key: item
        for key, item in value.items()
        if key != "collision_geometry"
    }


def _adapter_spec(plan: MechanismPlan) -> BicycleChainSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    return bicycle_chain_from_dict(_package_family_spec(adapter.family_spec))


def _task_thresholds(plan: MechanismPlan) -> dict[str, float]:
    contract = (
        plan.acceptance_contract
        if isinstance(plan.acceptance_contract, Mapping)
        else {}
    )
    requirements = [
        item
        for item in contract.get("requirements", ())
        if isinstance(item, Mapping)
    ]

    def target(metric_id: str) -> Mapping[str, Any]:
        matches = [
            item.get("target")
            for item in requirements
            if item.get("metric_id") == metric_id
            and isinstance(item.get("target"), Mapping)
        ]
        if len(matches) != 1:
            raise _fail(
                "BICYCLE_CHAIN_TASK_THRESHOLD_MISSING",
                f"frozen contract must expose exactly one {metric_id!r} target",
            )
        return matches[0]

    rotation = target("continuous_rotation_rad")
    ratio = target("angular_ratio")
    returned = target("return_to_start_rad")
    result = {
        "powered_rotation_min_rad": _finite(
            rotation.get("value"), "continuous_rotation_rad.value"
        ),
        "signed_output_per_input": _finite(
            ratio.get("value"), "angular_ratio.value"
        ),
        "angular_ratio_tolerance": _finite(
            ratio.get("tolerance"), "angular_ratio.tolerance"
        ),
        "return_to_home_max_rad": _finite(
            returned.get("value"), "return_to_start_rad.value"
        ),
    }
    if (
        result["powered_rotation_min_rad"] <= 0.0
        or result["signed_output_per_input"] <= 0.0
        or not 0.0 <= result["angular_ratio_tolerance"] < 1.0
        or result["return_to_home_max_rad"] < 0.0
    ):
        raise _fail(
            "BICYCLE_CHAIN_TASK_THRESHOLD_INVALID",
            "frozen bicycle task thresholds are outside their physical domain",
        )
    return result


def _task_program(spec: BicycleChainSpec) -> Any:
    program = spec.task_program
    if program is None:
        raise _fail(
            "BICYCLE_CHAIN_TASK_PROGRAM_MISSING",
            "runtime comparison requires the closed typed r04 task program",
        )
    return program


def _quaternion_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _pose_rotation_quaternion(
    pose: Any,
    *,
    label: str,
) -> tuple[float, float, float, float]:
    if not isinstance(pose, Mapping):
        raise _fail(
            "BICYCLE_CHAIN_TRACE_BODY_POSE_INVALID",
            f"{label} is not an object",
        )
    raw = pose.get("rotation_abs_quat_wxyz")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 4:
        raise _fail(
            "BICYCLE_CHAIN_TRACE_BODY_POSE_INVALID",
            f"{label}.rotation_abs_quat_wxyz is not a quaternion",
        )
    quaternion = tuple(
        _finite(value, f"{label}.rotation_abs_quat_wxyz") for value in raw
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1.0e-12 or abs(norm - 1.0) > 1.0e-6:
        raise _fail(
            "BICYCLE_CHAIN_TRACE_BODY_POSE_INVALID",
            f"{label}.rotation_abs_quat_wxyz is not normalized",
        )
    return tuple(value / norm for value in quaternion)


def _pose_angle_about_axis(
    initial: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    axis: tuple[float, float, float],
) -> float:
    relative = _quaternion_multiply(
        current,
        (initial[0], -initial[1], -initial[2], -initial[3]),
    )
    projected = sum(relative[index + 1] * axis[index] for index in range(3))
    return 2.0 * math.atan2(projected, relative[0])


def _wrapped_angle_error(left: float, right: float) -> float:
    difference = left - right
    return abs(math.atan2(math.sin(difference), math.cos(difference)))


def _held_out_domain(spec: BicycleChainSpec) -> Any:
    domain = spec.held_out_validation_domain
    if domain is None:
        raise _fail(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_MISSING",
            "runtime comparison requires the closed typed r04 validation domain",
        )
    return domain


def _is_roundtrip(load_case: LoadCase) -> bool:
    return load_case.program in {
        "task_bicycle_powered_roundtrip",
        BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
        "bicycle_passive_release_return",
    }


def _validate_case_against_typed_program(
    load_case: LoadCase,
    spec: BicycleChainSpec,
) -> None:
    program = _task_program(spec)
    if (
        load_case.step_s != program.integration_step_s
        or load_case.sample_interval_s
        != program.integration_step_s * program.saved_sample_stride_steps
    ):
        raise _fail(
            "BICYCLE_CHAIN_TRACE_CADENCE_INVALID",
            "runtime cadence differs from the approved typed r04 cadence",
        )
    if not _is_roundtrip(load_case):
        if (
            load_case.program != "bicycle_zero_input_home_drift"
            or len(load_case.phases) != 1
            or load_case.phases[0].phase_id != "zero_state_no_input_drift"
            or load_case.phases[0].wrench_kind != "none"
            or load_case.phases[0].magnitude_si != 0.0
            or load_case.phases[0].direction_abs is not None
            or load_case.phases[0].duration_s != 0.60
            or load_case.phases[0].stop_condition is not None
        ):
            raise _fail(
                "BICYCLE_CHAIN_TASK_PROGRAM_CHANGED",
                "runtime load case is not an approved r04 program",
            )
        return

    expected = tuple(
        (
            phase.phase_id,
            phase.wrench_kind,
            phase.duration_s,
        )
        for phase in program.phases
    )
    actual = tuple(
        (phase.phase_id, phase.wrench_kind, phase.duration_s)
        for phase in load_case.phases
    )
    if actual != expected:
        raise _fail(
            "BICYCLE_CHAIN_TASK_PROGRAM_CHANGED",
            "roundtrip phase identity, kind, order, or duration changed",
        )
    if (
        tuple(program.engagement_phase_ids)
        != ("finite_engagement_settle",)
        or tuple(program.powered_phase_ids)
        != ("measured_powered_drive",)
        or tuple(program.return_phase_ids)
        != ("passive_return", "return_settle")
        or program.adaptive_phase_ids
        or program.engagement_task_motion_credit is not False
        or program.engagement_ratio_metric_credit is not False
        or program.return_input_policy != "zero"
    ):
        raise _fail(
            "BICYCLE_CHAIN_TASK_PROGRAM_CHANGED",
            "r04 task-window credit or return policy changed",
        )
    torques = {
        phase.magnitude_si
        for phase in load_case.phases
        if phase.phase_id
        in set(program.engagement_phase_ids) | set(program.powered_phase_ids)
    }
    directions = {
        phase.direction_abs
        for phase in load_case.phases
        if phase.phase_id
        in set(program.engagement_phase_ids) | set(program.powered_phase_ids)
    }
    domain = _held_out_domain(spec)
    perturbation = (
        load_case.program == BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM
    )
    expected_direction = tuple(
        -value if perturbation else value for value in spec.axis_abs
    )
    torque = next(iter(torques)) if len(torques) == 1 else None
    torque_valid = (
        torque == BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM
        if perturbation
        else (
            torque is not None
            and domain.minimum_torque_nm
            <= torque
            <= domain.maximum_torque_nm
        )
    )
    if (
        len(torques) != 1
        or not torque_valid
        or len(directions) != 1
        or None in directions
        or next(iter(directions)) != expected_direction
    ):
        raise _fail(
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_INVALID",
            "engagement and measured phases must share the approved bounded "
            "crank torque and direction",
        )
    if any(
        phase.direction_abs is not None or phase.magnitude_si != 0.0
        for phase in load_case.phases
        if phase.phase_id in set(program.return_phase_ids)
    ):
        raise _fail(
            "BICYCLE_CHAIN_RETURN_TASK_INPUT_NONZERO",
            "both return phases must carry exact numeric zero task input",
        )


_REQUIRED_SAMPLE_NUMBERS = (
    "input_crank_angle_rad",
    "rear_drive_wheel_angle_rad",
    "input_crank_speed_rad_s",
    "rear_drive_wheel_speed_rad_s",
    "chain_coordinate_error_m",
    "chain_coordinate_rate_m_s",
    "chain_differential_tension_n",
    "chain_input_reaction_torque_nm",
    "chain_output_reaction_torque_nm",
    "task_input_torque_nm",
    "crank_centering_return_torque_nm",
    "input_bearing_primary_axis_torque_nm",
    "input_bearing_primary_axis_torque_cap_nm",
    "rear_bearing_drag_torque_nm",
    "max_support_drift_m",
    "max_off_axis_rotation_rad",
    "max_support_force_n",
    "max_support_torque_nm",
    "input_bearing_total_support_torque_nm",
    "rear_bearing_total_support_torque_nm",
    "chain_stored_energy_j",
    "input_bearing_primary_axis_stored_energy_j",
    "chain_axis_angle_error_rad",
    "chain_axial_alignment_error_m",
)

_REAR_DRIVE_WHEEL_WRENCH_ALLOWLIST = frozenset(
    {
        (
            "finite_chain_output_torque",
            "torque",
            "transmission",
            "finite_geometry_gated_chain_v2",
        ),
        (
            "plugin_rear_bearing_retention_force",
            "force",
            "finite_support",
            "finite_6d_linear_support_v2",
        ),
        (
            "plugin_rear_bearing_alignment_torque",
            "torque",
            "finite_support",
            "finite_6d_linear_support_v2",
        ),
        (
            "plugin_rear_bearing_spin_drag_torque",
            "torque",
            "finite_support",
            "finite_6d_linear_support_v2",
        ),
    }
)


def _validate_rear_drive_wheel_wrench_inventory(
    records: Any,
    spec: BicycleChainSpec,
) -> None:
    """Require the exact runtime-owned wrench inventory on the task output."""

    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise _fail(
            "WRENCH_POLICY_NOT_ALLOWED",
            "bicycle rear-drive-wheel wrench inventory is missing",
        )
    actual: list[tuple[Any, Any, Any, Any]] = []
    for item in records:
        if not isinstance(item, Mapping):
            raise _fail(
                "WRENCH_POLICY_NOT_ALLOWED",
                "bicycle wrench inventory contains a non-record entry",
            )
        if item.get("target_body") != spec.output_body_id:
            continue
        actual.append(
            (
                item.get("load_id"),
                item.get("wrench_kind"),
                item.get("category"),
                item.get("law_id"),
            )
        )
    if (
        len(actual) != len(_REAR_DRIVE_WHEEL_WRENCH_ALLOWLIST)
        or frozenset(actual) != _REAR_DRIVE_WHEEL_WRENCH_ALLOWLIST
    ):
        raise _fail(
            "WRENCH_POLICY_NOT_ALLOWED",
            "bicycle rear-drive-wheel wrench inventory differs from the "
            "closed runtime allowlist",
        )


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: BicycleChainSpec,
) -> list[dict[str, Any]]:
    _validate_case_against_typed_program(load_case, spec)
    expected = {
        "status": "COMPLETED",
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "mechanism_plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
    }
    mismatches = [
        field for field, value in expected.items()
        if trace.get(field) != value
    ]
    if mismatches:
        raise _fail(
            "BICYCLE_CHAIN_TRACE_IDENTITY_MISMATCH",
            f"{branch} trace differs at: {', '.join(mismatches)}",
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise _fail(
            "BICYCLE_CHAIN_TRACE_SAMPLES_MISSING",
            f"{branch} trace has too few samples",
        )
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    previous_time: float | None = None
    context_runtime_hashes: set[str] = set()
    context_contract_role_hashes: set[str] = set()
    context_collision_hashes: set[str] = set()
    context_assembly_hashes: set[str] = set()
    context_assembly_step_hashes: set[str] = set()
    context_collision_schemas: set[str] = set()
    context_collision_component_counts: set[int] = set()
    context_gate_hashes: set[str] = set()
    initial_pose_quaternions: dict[str, tuple[float, float, float, float]] = {}
    initial_reported_angles: dict[str, float] = {}
    for index, sample in enumerate(samples):
        label = f"{branch}.samples[{index}]"
        if not isinstance(sample, dict):
            raise _fail(
                "BICYCLE_CHAIN_TRACE_SAMPLE_INVALID",
                f"{label} is not an object",
            )
        if sample.get("phase_id") not in allowed_phases:
            raise _fail(
                "BICYCLE_CHAIN_TRACE_PHASE_INVALID",
                f"{label} has an unknown phase",
            )
        time_s = _finite(sample.get("time_s"), f"{label}.time_s")
        if previous_time is not None and time_s <= previous_time:
            raise _fail(
                "BICYCLE_CHAIN_TRACE_TIME_INVALID",
                f"{branch} sample time must be unique and increasing",
            )
        if previous_time is not None and abs(
            time_s - previous_time - load_case.sample_interval_s
        ) > 1.0e-9:
            raise _fail(
                "BICYCLE_CHAIN_TRACE_CADENCE_INVALID",
                f"{branch} trace is not the exact approved saved cadence",
            )
        previous_time = time_s
        for field in _REQUIRED_SAMPLE_NUMBERS:
            _finite(sample.get(field), f"{label}.{field}")
        if not isinstance(sample.get("chain_engagement_gate_active"), bool):
            raise _fail(
                "BICYCLE_CHAIN_TRACE_GATE_INVALID",
                f"{label} lacks a boolean chain gate",
            )
        body_poses = sample.get("body_poses")
        if (
            not isinstance(body_poses, Mapping)
            or not {
                spec.fixed_body_id,
                spec.input_body_id,
                spec.output_body_id,
            }.issubset(body_poses)
        ):
            raise _fail(
                "BICYCLE_CHAIN_TRACE_BODY_POSE_MISSING",
                f"{label} lacks the three bound body poses",
            )
        for body_id, angle_field in (
            (spec.input_body_id, "input_crank_angle_rad"),
            (spec.output_body_id, "rear_drive_wheel_angle_rad"),
        ):
            quaternion = _pose_rotation_quaternion(
                body_poses[body_id],
                label=f"{label}.body_poses[{body_id!r}]",
            )
            reported_angle = float(sample[angle_field])
            if body_id not in initial_pose_quaternions:
                initial_pose_quaternions[body_id] = quaternion
                initial_reported_angles[body_id] = reported_angle
                continue
            pose_angle = _pose_angle_about_axis(
                initial_pose_quaternions[body_id],
                quaternion,
                spec.axis_abs,
            )
            reported_delta = reported_angle - initial_reported_angles[body_id]
            if _wrapped_angle_error(reported_delta, pose_angle) > 1.0e-6:
                raise _fail(
                    "BICYCLE_CHAIN_TRACE_POSE_ANGLE_MISMATCH",
                    f"{label} reported angle disagrees with the body pose",
                )
        context = sample.get("chain_wrench_context")
        if not isinstance(context, Mapping):
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_CONTEXT_MISSING",
                f"{label} lacks the paired chain context",
            )
        context_time = _finite(
            context.get("solver_time_s"),
            f"{label}.chain_wrench_context.solver_time_s",
        )
        if (
            abs(context_time - time_s) > 1.0e-12
            or context.get("plugin_instance_id")
            != f"{BICYCLE_CHAIN_PLUGIN_ID}:single_instance"
            or context.get("plugin_id") != BICYCLE_CHAIN_PLUGIN_ID
            or not isinstance(context.get("state_id"), str)
            or not isinstance(context.get("geometry_gate_sha256"), str)
            or context.get("geometry_gate_id")
            != spec.geometry_gate.gate_id
            or not isinstance(context.get("runtime_binding_sha256"), str)
            or not isinstance(
                context.get("contract_role_binding_sha256"),
                str,
            )
            or not isinstance(context.get("collision_binding_sha256"), str)
            or not isinstance(context.get("assembly_contract_sha256"), str)
            or not isinstance(context.get("assembly_sha256"), str)
            or not isinstance(context.get("collision_schema"), str)
            or isinstance(context.get("collision_component_count"), bool)
            or not isinstance(context.get("collision_component_count"), int)
            or int(context["collision_component_count"]) <= 0
            or context.get("gate_active")
            is not sample["chain_engagement_gate_active"]
            or context.get("fixed_body_id") != spec.fixed_body_id
            or context.get("input_body_id") != spec.input_body_id
            or context.get("output_body_id") != spec.output_body_id
            or context.get("input_port_id") != spec.input_port_id
            or context.get("output_observable_id")
            != spec.output_observable_id
        ):
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_CONTEXT_MISMATCH",
                f"{label} does not bind one solver state and gate",
            )
        state_payload = {
            "time_s": time_s,
            "input_angle_rad": float(sample["input_crank_angle_rad"]),
            "output_angle_rad": float(
                sample["rear_drive_wheel_angle_rad"]
            ),
            "input_speed_rad_s": float(
                sample["input_crank_speed_rad_s"]
            ),
            "output_speed_rad_s": float(
                sample["rear_drive_wheel_speed_rad_s"]
            ),
            "coordinate_error_m": float(
                sample["chain_coordinate_error_m"]
            ),
            "coordinate_rate_m_s": float(
                sample["chain_coordinate_rate_m_s"]
            ),
            "tension_n": float(sample["chain_differential_tension_n"]),
            "gate_active": sample["chain_engagement_gate_active"],
            "axis_angle_error_rad": float(
                sample["chain_axis_angle_error_rad"]
            ),
            "axial_alignment_error_m": float(
                sample["chain_axial_alignment_error_m"]
            ),
        }
        if context["state_id"] != canonical_sha256(state_payload):
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_CONTEXT_MISMATCH",
                f"{label} chain context is not bound to its saved solver state",
            )
        context_runtime_hashes.add(str(context["runtime_binding_sha256"]))
        context_contract_role_hashes.add(
            str(context["contract_role_binding_sha256"])
        )
        context_collision_hashes.add(
            str(context["collision_binding_sha256"])
        )
        context_assembly_hashes.add(
            str(context["assembly_contract_sha256"])
        )
        context_assembly_step_hashes.add(str(context["assembly_sha256"]))
        context_collision_schemas.add(str(context["collision_schema"]))
        context_collision_component_counts.add(
            int(context["collision_component_count"])
        )
        context_gate_hashes.add(str(context["geometry_gate_sha256"]))
        input_reaction = float(sample["chain_input_reaction_torque_nm"])
        output_reaction = float(sample["chain_output_reaction_torque_nm"])
        paired_residual = (
            input_reaction / spec.input_pitch_radius_m
            + output_reaction / spec.output_pitch_radius_m
        )
        if abs(paired_residual) > 1.0e-9:
            raise _fail(
                "BICYCLE_CHAIN_WRENCH_PAIR_INVALID",
                f"{label} chain reactions do not share one tension state",
            )
        if not sample["chain_engagement_gate_active"] and (
            input_reaction != 0.0
            or output_reaction != 0.0
            or float(sample["chain_differential_tension_n"]) != 0.0
        ):
            raise _fail(
                "BICYCLE_CHAIN_GATE_OFF_WRENCH_NONZERO",
                f"{label} has a nonzero chain wrench while its gate is off",
            )
    if (
        samples[0]["phase_id"] != "initial"
        or float(samples[0]["time_s"]) != 0.0
        or any(
            float(samples[0][field]) != 0.0
            for field in (
                "input_crank_angle_rad",
                "rear_drive_wheel_angle_rad",
                "input_crank_speed_rad_s",
                "rear_drive_wheel_speed_rad_s",
                "chain_coordinate_error_m",
                "chain_coordinate_rate_m_s",
                "chain_differential_tension_n",
                "chain_input_reaction_torque_nm",
                "chain_output_reaction_torque_nm",
                "task_input_torque_nm",
                "chain_stored_energy_j",
                "input_bearing_primary_axis_stored_energy_j",
            )
        )
    ):
        raise _fail(
            "BICYCLE_CHAIN_ZERO_STATE_INVALID",
            "trace must begin at exact zero pose, velocity, wrench, and energy",
        )
    expected_order = tuple(phase.phase_id for phase in load_case.phases)
    actual_order: list[str] = []
    for sample in samples:
        phase = str(sample["phase_id"])
        if phase == "initial":
            continue
        if not actual_order or actual_order[-1] != phase:
            actual_order.append(phase)
    if tuple(actual_order) != expected_order:
        raise _fail(
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            "trace does not cover every phase boundary exactly in order",
        )
    cumulative_end_s = 0.0
    previous_end_s = 0.0
    for phase in load_case.phases:
        cumulative_end_s += phase.duration_s
        phase_samples = [
            sample for sample in samples
            if sample["phase_id"] == phase.phase_id
        ]
        if (
            not phase_samples
            or abs(
                float(phase_samples[-1]["time_s"]) - cumulative_end_s
            ) > 1.0e-9
            or abs(
                float(phase_samples[0]["time_s"])
                - (previous_end_s + load_case.sample_interval_s)
            ) > 1.0e-9
        ):
            raise _fail(
                "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
                f"{phase.phase_id} does not preserve its exact boundary samples",
            )
        previous_end_s = cumulative_end_s
    if abs(float(samples[-1]["time_s"]) - cumulative_end_s) > 1.0e-9:
        raise _fail(
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            "trace final time differs from the immutable phase program",
        )

    semantics = trace.get("branch_semantics")
    geometry = trace.get("functional_geometry")
    collision = trace.get("collision_geometry")
    if (
        len(context_runtime_hashes) != 1
        or len(context_contract_role_hashes) != 1
        or len(context_collision_hashes) != 1
        or len(context_assembly_hashes) != 1
        or len(context_assembly_step_hashes) != 1
        or len(context_collision_schemas) != 1
        or len(context_collision_component_counts) != 1
        or len(context_gate_hashes) != 1
        or not isinstance(semantics, Mapping)
        or semantics.get("runtime_binding_sha256")
        != next(iter(context_runtime_hashes))
        or semantics.get("contract_role_binding_sha256")
        != next(iter(context_contract_role_hashes))
        or semantics.get("collision_geometry_sha256")
        != next(iter(context_collision_hashes))
        or semantics.get("assembly_contract_sha256")
        != next(iter(context_assembly_hashes))
        or semantics.get("task_program_sha256")
        != canonical_sha256(asdict(spec.task_program))
        or semantics.get("held_out_validation_domain_sha256")
        != canonical_sha256(asdict(spec.held_out_validation_domain))
        or not isinstance(geometry, Mapping)
        or geometry.get("geometry_gate_sha256")
        != next(iter(context_gate_hashes))
        or geometry.get("contract_role_binding_sha256")
        != next(iter(context_contract_role_hashes))
        or not isinstance(collision, Mapping)
        or collision.get("assembly_contract_sha256")
        != next(iter(context_assembly_hashes))
        or collision.get("assembly_sha256")
        != next(iter(context_assembly_step_hashes))
        or collision.get("schema")
        != next(iter(context_collision_schemas))
        or collision.get("component_count")
        != next(iter(context_collision_component_counts))
    ):
        raise _fail(
            "BICYCLE_CHAIN_WRENCH_CONTEXT_MISMATCH",
            "saved states do not share the trace's runtime, collision, "
            "assembly, and geometry-gate bindings",
        )
    return samples


def _ratio_diagnostics(
    samples: Sequence[Mapping[str, Any]],
    *,
    powered_phase_ids: set[str],
    engagement_phase_ids: set[str],
    expected_ratio: float,
) -> dict[str, float]:
    indices = [
        index
        for index, sample in enumerate(samples)
        if sample.get("phase_id") in powered_phase_ids
    ]
    if not indices:
        raise _fail(
            "BICYCLE_CHAIN_POWERED_WINDOW_MISSING",
            "powered ratio window has no samples",
        )
    if (
        indices != list(range(indices[0], indices[-1] + 1))
        or indices[0] == 0
        or samples[indices[0] - 1].get("phase_id")
        not in engagement_phase_ids
    ):
        raise _fail(
            "BICYCLE_CHAIN_POWERED_BASELINE_INVALID",
            "measured window must start immediately after its saved engagement "
            "boundary",
        )
    baseline_index = indices[0] - 1
    input_home = float(samples[baseline_index]["input_crank_angle_rad"])
    output_home = float(
        samples[baseline_index]["rear_drive_wheel_angle_rad"]
    )
    input_excursions = [
        float(samples[index]["input_crank_angle_rad"]) - input_home
        for index in indices
    ]
    output_excursions = [
        float(samples[index]["rear_drive_wheel_angle_rad"]) - output_home
        for index in indices
    ]
    peak_input = max(input_excursions)

    def error(output_delta: float, input_delta: float) -> float:
        if abs(input_delta) <= 1.0e-12:
            if abs(output_delta) <= 1.0e-12:
                return 0.0
            return math.inf
        return abs(output_delta / input_delta - expected_ratio)

    pointwise = [
        error(output_delta, input_delta)
        for output_delta, input_delta in zip(
            output_excursions, input_excursions
        )
    ]
    all_indices = [baseline_index, *indices]
    incremental = [
        error(
            float(samples[right]["rear_drive_wheel_angle_rad"])
            - float(samples[left]["rear_drive_wheel_angle_rad"]),
            float(samples[right]["input_crank_angle_rad"])
            - float(samples[left]["input_crank_angle_rad"]),
        )
        for left, right in zip(all_indices, all_indices[1:])
    ]
    return {
        "powered_output_excursion_rad": max(output_excursions),
        "powered_input_excursion_rad": peak_input,
        "max_powered_ratio_pointwise_error": max(pointwise),
        "max_powered_ratio_incremental_error": max(incremental),
        "ratio_reference_floor_rad": 0.0,
        "measured_baseline_time_s": float(
            samples[baseline_index]["time_s"]
        ),
    }


def _metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    spec: BicycleChainSpec,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    input_angles = [
        float(sample["input_crank_angle_rad"]) for sample in samples
    ]
    output_angles = [
        float(sample["rear_drive_wheel_angle_rad"]) for sample in samples
    ]
    program = _task_program(spec)
    powered_phase_ids = (
        set(program.powered_phase_ids) if _is_roundtrip(load_case) else set()
    )
    engagement_phase_ids = (
        set(program.engagement_phase_ids) if _is_roundtrip(load_case) else set()
    )
    return_phase_ids = (
        set(program.return_phase_ids) if _is_roundtrip(load_case) else set()
    )
    ratio = _ratio_diagnostics(
        samples,
        powered_phase_ids=powered_phase_ids,
        engagement_phase_ids=engagement_phase_ids,
        expected_ratio=thresholds["signed_output_per_input"],
    ) if powered_phase_ids else {
        "powered_output_excursion_rad": 0.0,
        "powered_input_excursion_rad": 0.0,
        "max_powered_ratio_pointwise_error": 0.0,
        "max_powered_ratio_incremental_error": 0.0,
        "ratio_reference_floor_rad": 0.0,
        "measured_baseline_time_s": 0.0,
    }
    engagement_output_excursion = (
        max(
            abs(
                float(sample["rear_drive_wheel_angle_rad"])
                - output_angles[0]
            )
            for sample in samples
            if sample["phase_id"] in engagement_phase_ids
        )
        if engagement_phase_ids
        else 0.0
    )
    return {
        **ratio,
        "input_angle_min_rad": min(input_angles),
        "input_angle_max_rad": max(input_angles),
        "output_angle_min_rad": min(output_angles),
        "output_angle_max_rad": max(output_angles),
        "final_output_home_error_rad": abs(output_angles[-1] - output_angles[0]),
        "max_chain_coordinate_error_m": max(
            abs(float(sample["chain_coordinate_error_m"]))
            for sample in samples
        ),
        "max_chain_differential_tension_n": max(
            abs(float(sample["chain_differential_tension_n"]))
            for sample in samples
        ),
        "max_support_drift_m": max(
            float(sample["max_support_drift_m"]) for sample in samples
        ),
        "max_off_axis_rotation_rad": max(
            float(sample["max_off_axis_rotation_rad"]) for sample in samples
        ),
        "max_support_force_n": max(
            float(sample["max_support_force_n"]) for sample in samples
        ),
        "max_support_torque_nm": max(
            float(sample["max_support_torque_nm"]) for sample in samples
        ),
        "max_input_bearing_primary_axis_torque_nm": max(
            abs(float(sample["input_bearing_primary_axis_torque_nm"]))
            for sample in samples
        ),
        "max_input_bearing_total_support_torque_nm": max(
            float(sample["input_bearing_total_support_torque_nm"])
            for sample in samples
        ),
        "max_rear_bearing_total_support_torque_nm": max(
            float(sample["rear_bearing_total_support_torque_nm"])
            for sample in samples
        ),
        "max_chain_axis_angle_error_rad": max(
            float(sample["chain_axis_angle_error_rad"])
            for sample in samples
        ),
        "max_chain_axial_alignment_error_m": max(
            float(sample["chain_axial_alignment_error_m"])
            for sample in samples
        ),
        "initial_stored_energy_j": (
            float(samples[0]["chain_stored_energy_j"])
            + float(
                samples[0][
                    "input_bearing_primary_axis_stored_energy_j"
                ]
            )
        ),
        "engagement_output_excursion_rad_uncredited": (
            engagement_output_excursion
        ),
        "max_inactive_chain_wrench_nm": max(
            max(
                abs(float(sample["chain_input_reaction_torque_nm"])),
                abs(float(sample["chain_output_reaction_torque_nm"])),
            )
            for sample in samples
            if sample["chain_engagement_gate_active"] is False
        ) if any(
            sample["chain_engagement_gate_active"] is False
            for sample in samples
        ) else 0.0,
        "return_task_input_exact_zero": all(
            float(sample["task_input_torque_nm"]) == 0.0
            for sample in samples
            if sample["phase_id"] in return_phase_ids
        ),
        "all_chain_gate_active": all(
            bool(sample["chain_engagement_gate_active"]) for sample in samples
        ),
        "full_cadence_sample_count": len(samples),
        "input_output_body_distinct": spec.input_body_id != spec.output_body_id,
    }


def _visible_chain_geometry_failures(
    geometry: Any,
    semantics: Any,
    tolerance: Any,
) -> list[str]:
    """Reject missing, inconsistent, or threshold-tampered chain evidence."""

    invalid = ["bicycle_visible_chain_geometry_invalid"]
    if not isinstance(geometry, Mapping) or not isinstance(semantics, Mapping):
        return invalid
    visible_chain = geometry.get("visible_chain_geometry")
    if not isinstance(visible_chain, Mapping):
        return invalid
    visible_declared = visible_chain.get("declared") is True
    expected_thresholds = {
        "maximum_chain_sprocket_separation_m": (
            tolerance.maximum_chain_sprocket_separation_m
        ),
        "wrap_band_m": tolerance.visible_chain_wrap_band_m,
        "minimum_wrap_angle_rad": (
            tolerance.minimum_visible_chain_wrap_angle_rad
        ),
        "minimum_wrap_sample_count": (
            tolerance.minimum_visible_chain_wrap_sample_count
        ),
        "maximum_pitch_plane_offset_m": (
            tolerance.maximum_pitch_plane_offset_m
        ),
    }
    if (
        visible_chain.get("schema_version")
        != "bicycle_visible_chain_geometry_v1"
        or visible_chain.get("required_part_id")
        != tolerance.visible_chain_part_id
        or visible_chain.get("thresholds") != expected_thresholds
        or semantics.get("visible_chain_declared") is not visible_declared
    ):
        return invalid
    if not visible_declared:
        return invalid
    if (
        visible_chain.get("status") != "PASS"
        or visible_chain.get("passed") is not True
        or visible_chain.get("failure_reasons") not in ([], ())
        or visible_chain.get("source")
        != "authoritative_step_solid_measurement"
        or not isinstance(visible_chain.get("chain_solid_ids"), list)
        or not visible_chain["chain_solid_ids"]
        or not isinstance(visible_chain.get("surface_sample_count"), int)
        or visible_chain["surface_sample_count"] <= 0
        or semantics.get(
            "visible_chain_geometry_bound_to_finite_transmission_model"
        )
        is not True
    ):
        return invalid
    for role in ("input", "output"):
        metrics = visible_chain.get(role)
        try:
            invalid_role = (
                not isinstance(metrics, Mapping)
                or _finite(
                    metrics.get("minimum_sprocket_separation_m"),
                    "minimum_sprocket_separation_m",
                )
                > tolerance.maximum_chain_sprocket_separation_m
                or _finite(
                    metrics.get("wrap_angle_rad"),
                    "wrap_angle_rad",
                )
                < tolerance.minimum_visible_chain_wrap_angle_rad
                or int(metrics.get("near_pitch_band_sample_count", -1))
                < tolerance.minimum_visible_chain_wrap_sample_count
                or _finite(
                    metrics.get("pitch_plane_offset_m"),
                    "pitch_plane_offset_m",
                )
                > tolerance.maximum_pitch_plane_offset_m
            )
        except (MechanismValidationError, TypeError, ValueError):
            invalid_role = True
        if invalid_role:
            return invalid
    return []


def _physical_semantic_failures(
    trace: Mapping[str, Any],
    spec: BicycleChainSpec,
) -> list[str]:
    failures: list[str] = []
    engine = trace.get("engine")
    semantics = trace.get("branch_semantics")
    if (
        not isinstance(engine, Mapping)
        or engine.get("internal_ideal_joint_count") != 0
        or engine.get("ideal_links") not in ((), [])
    ):
        failures.append("physical_bicycle_contains_ideal_link")
    if (
        not isinstance(semantics, Mapping)
        or semantics.get("pose_setters_used") is not False
        or semantics.get("direct_output_actuation_used") is not False
        or semantics.get("replaced_contacts_disabled") is not False
        or semantics.get("physical_internal_ideal_joints_forbidden")
        is not True
        or semantics.get("individual_link_tooth_contact_simulated")
        is not False
        or semantics.get("body_inertias_are_typed_family_values")
        is not True
        or semantics.get("initial_state_zero_energy_verified") is not True
        or semantics.get("task_input_body_id") != spec.input_body_id
        or semantics.get("unique_task_output_body_id")
        != spec.output_body_id
    ):
        failures.append("physical_bicycle_branch_semantics_invalid")
    geometry = trace.get("functional_geometry")
    if (
        not isinstance(geometry, Mapping)
        or geometry.get("status") != "PASS"
        or geometry.get("input_pitch_radius_m")
        != spec.input_pitch_radius_m
        or geometry.get("output_pitch_radius_m")
        != spec.output_pitch_radius_m
    ):
        failures.append("bicycle_chain_functional_geometry_not_bound")
    elif spec.geometry_gate is None:
        failures.append("bicycle_chain_functional_geometry_not_bound")
    else:
        surface_pairs = (
            (geometry.get("input_pitch_surface"), spec.input_sprocket),
            (geometry.get("output_pitch_surface"), spec.output_sprocket),
        )
        tolerance = spec.geometry_gate
        for surface, sprocket in surface_pairs:
            if (
                not isinstance(surface, Mapping)
                or surface.get("topology_bound") is not True
                or surface.get("solid_id") != sprocket.step_solid_id
                or abs(
                    _finite(
                        surface.get("major_radius_m"),
                        "pitch_surface.major_radius_m",
                    )
                    - sprocket.pitch_radius_m
                )
                > tolerance.pitch_radius_tolerance_m
                or abs(
                    _finite(
                        surface.get("minor_radius_m"),
                        "pitch_surface.minor_radius_m",
                    )
                    - sprocket.toroidal_minor_radius_m
                )
                > tolerance.pitch_radius_tolerance_m
                or len(surface.get("center_abs_m") or ()) != 3
                or max(
                    abs(float(actual) - float(expected))
                    for actual, expected in zip(
                        surface.get("center_abs_m") or (),
                        sprocket.pitch_loop_center_abs_m,
                    )
                )
                > tolerance.pitch_center_tolerance_m
                or len(surface.get("axis_abs") or ()) != 3
                or abs(
                    sum(
                        float(actual) * float(expected)
                        for actual, expected in zip(
                            surface["axis_abs"],
                            sprocket.pitch_axis_abs,
                        )
                    )
                )
                < math.cos(tolerance.maximum_axis_parallel_error_rad)
            ):
                failures.append(
                    "bicycle_chain_functional_geometry_not_bound"
                )
                break
        failures.extend(
            _visible_chain_geometry_failures(
                geometry,
                semantics,
                tolerance,
            )
        )
    audit = trace.get("wrench_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("passed") is not True
        or audit.get("direct_output_drive_detected") is not False
        or set(audit.get("task_input_bodies_observed") or ())
        - {spec.input_body_id}
    ):
        failures.append("bicycle_chain_crank_only_wrench_provenance_invalid")
    records = audit.get("records") if isinstance(audit, Mapping) else None
    try:
        _validate_rear_drive_wheel_wrench_inventory(records, spec)
    except MechanismValidationError as exc:
        failures.append(exc.code)
    crank_task_records = (
        [
            item
            for item in records
            if isinstance(item, Mapping)
            and item.get("load_id")
            == "validator_crank_task_torque"
        ]
        if isinstance(records, Sequence)
        and not isinstance(records, (str, bytes))
        else []
    )
    if trace.get("load_case_id") == "reverse_or_resisting_crank":
        if (
            len(crank_task_records) != 1
            or crank_task_records[0].get("owner")
            != BICYCLE_CHAIN_PLUGIN_ID
            or crank_task_records[0].get("target_body")
            != spec.input_body_id
            or crank_task_records[0].get("category")
            != "task_input"
            or crank_task_records[0].get("law_id")
            != "validator_load_phase_v1"
            or crank_task_records[0].get("task_input") is not True
            or crank_task_records[0].get("released") is not True
        ):
            failures.append(
                "bicycle_input_crank_perturbation_provenance_invalid"
            )
            return failures
        record = crank_task_records[0]
        nonzero_phases = {
            phase_id
            for phase_id, count in (
                record.get("phase_nonzero_update_counts") or {}
            ).items()
            if int(count) > 0
        }
        if nonzero_phases != {
            "finite_engagement_settle",
            "measured_powered_drive",
        }:
            failures.append(
                "bicycle_input_crank_perturbation_phase_audit_invalid"
            )
    return failures


def _behavior_failures(
    metrics: Mapping[str, Any],
    *,
    load_case: LoadCase,
    spec: BicycleChainSpec,
    thresholds: Mapping[str, float],
) -> list[str]:
    failures: list[str] = []
    if metrics["max_support_drift_m"] > spec.max_support_drift_m:
        failures.append("bicycle_support_drift_exceeded")
    if metrics["max_off_axis_rotation_rad"] > spec.max_off_axis_rotation_rad:
        failures.append("bicycle_off_axis_rotation_exceeded")
    if (
        metrics["max_chain_axis_angle_error_rad"]
        > spec.maximum_axis_angle_error_rad
        or metrics["max_chain_axial_alignment_error_m"]
        > spec.maximum_alignment_error_m
    ):
        failures.append("bicycle_chain_geometry_gate_limit_exceeded")
    if (
        metrics["max_chain_differential_tension_n"]
        > spec.maximum_differential_tension_n + 1.0e-9
    ):
        failures.append("bicycle_chain_tension_cap_exceeded")
    if (
        metrics["max_chain_coordinate_error_m"]
        > spec.transmission.maximum_abs_chain_error_m + 1.0e-9
    ):
        failures.append("bicycle_chain_stretch_limit_exceeded")
    if metrics["max_inactive_chain_wrench_nm"] != 0.0:
        failures.append("bicycle_inactive_chain_wrench_nonzero")
    if metrics["initial_stored_energy_j"] != 0.0:
        failures.append("bicycle_initial_hidden_energy_nonzero")
    if (
        metrics["max_input_bearing_primary_axis_torque_nm"]
        > float(spec.input_bearing.primary_axis_maximum_abs_torque_nm)
        + 1.0e-9
    ):
        failures.append("bicycle_input_primary_axis_torque_cap_exceeded")
    if (
        metrics["max_input_bearing_total_support_torque_nm"]
        > spec.input_bearing.maximum_total_support_torque_nm + 1.0e-9
        or metrics["max_rear_bearing_total_support_torque_nm"]
        > spec.output_bearing.maximum_total_support_torque_nm + 1.0e-9
    ):
        failures.append("bicycle_total_support_torque_cap_exceeded")
    if metrics["max_support_force_n"] > min(
        spec.input_bearing.maximum_force_n,
        spec.output_bearing.maximum_force_n,
    ) + 1.0e-9:
        failures.append("bicycle_support_force_cap_exceeded")
    if metrics["all_chain_gate_active"] is not True:
        failures.append("bicycle_chain_geometry_gate_dropped")

    program = load_case.program
    ratio_limit = thresholds["angular_ratio_tolerance"]
    if program in {
        "task_bicycle_powered_roundtrip",
        "bicycle_passive_release_return",
    }:
        if (
            metrics["powered_output_excursion_rad"]
            < thresholds["powered_rotation_min_rad"]
        ):
            failures.append("bicycle_powered_rotation_not_reached")
        if (
            metrics["max_powered_ratio_pointwise_error"] > ratio_limit
            or metrics["max_powered_ratio_incremental_error"] > ratio_limit
        ):
            failures.append("bicycle_powered_all_sample_ratio_invalid")
        if (
            metrics["final_output_home_error_rad"]
            > thresholds["return_to_home_max_rad"]
        ):
            failures.append("bicycle_return_home_error_exceeded")
        if metrics["return_task_input_exact_zero"] is not True:
            failures.append("bicycle_return_task_input_nonzero")
    elif program == BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM:
        if (
            metrics["input_angle_min_rad"] >= 0.0
            or metrics["output_angle_min_rad"] >= 0.0
        ):
            failures.append(
                "bicycle_input_crank_reverse_response_not_observed"
            )
        if (
            metrics["max_powered_ratio_pointwise_error"] > ratio_limit
            or metrics["max_powered_ratio_incremental_error"] > ratio_limit
        ):
            failures.append("bicycle_powered_all_sample_ratio_invalid")
        if metrics["return_task_input_exact_zero"] is not True:
            failures.append("bicycle_return_task_input_nonzero")
    elif program == "bicycle_zero_input_home_drift":
        if (
            abs(metrics["input_angle_min_rad"]) > 1.0e-8
            or abs(metrics["input_angle_max_rad"]) > 1.0e-8
            or abs(metrics["output_angle_min_rad"]) > 1.0e-8
            or abs(metrics["output_angle_max_rad"]) > 1.0e-8
        ):
            failures.append("bicycle_zero_input_output_drift")
        if metrics["final_output_home_error_rad"] > 1.0e-8:
            failures.append("bicycle_zero_input_final_drift")
    return failures


def physical_bicycle_chain_behavior(
    plan: MechanismPlan,
    load_case: LoadCase,
    trace: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    spec = _adapter_spec(plan)
    thresholds = _task_thresholds(plan)
    samples = _validate_trace(
        trace,
        branch="physical",
        plan=plan,
        load_case=load_case,
        spec=spec,
    )
    metrics = _metrics(samples, load_case, spec, thresholds)
    failures = [
        *_physical_semantic_failures(trace, spec),
        *_behavior_failures(
            metrics,
            load_case=load_case,
            spec=spec,
            thresholds=thresholds,
        ),
    ]
    return metrics, {
        **thresholds,
        "max_support_drift_m": spec.max_support_drift_m,
        "max_off_axis_rotation_rad": spec.max_off_axis_rotation_rad,
        "maximum_differential_tension_n": (
            spec.maximum_differential_tension_n
        ),
        "input_primary_axis_maximum_abs_torque_nm": (
            spec.input_bearing.primary_axis_maximum_abs_torque_nm
        ),
        "input_maximum_total_support_torque_nm": (
            spec.input_bearing.maximum_total_support_torque_nm
        ),
        "output_maximum_total_support_torque_nm": (
            spec.output_bearing.maximum_total_support_torque_nm
        ),
        "source": "typed_bicycle_spec_and_frozen_task_contract",
    }, failures


def compare_bicycle_chain_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    try:
        spec = _adapter_spec(plan)
        thresholds = _task_thresholds(plan)
        physical_samples = _validate_trace(
            physical_trace,
            branch="physical",
            plan=plan,
            load_case=load_case,
            spec=spec,
        )
        reduced_samples = _validate_trace(
            reduced_trace,
            branch="reduced",
            plan=plan,
            load_case=load_case,
            spec=spec,
        )
    except MechanismValidationError as exc:
        return {
            "schema_version": "0.1",
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "purpose": load_case.purpose,
            "program": load_case.program,
            "required_for_certificate": load_case.required_for_certificate,
            "verdict": "PHYSICS_INCONCLUSIVE",
            "failure_reasons": [exc.code],
            "detail": str(exc),
        }
    physical_metrics = _metrics(
        physical_samples, load_case, spec, thresholds
    )
    reduced_metrics = _metrics(
        reduced_samples, load_case, spec, thresholds
    )
    physical_failures = [
        *_physical_semantic_failures(physical_trace, spec),
        *_behavior_failures(
            physical_metrics,
            load_case=load_case,
            spec=spec,
            thresholds=thresholds,
        ),
    ]
    reduced_failures = _behavior_failures(
        reduced_metrics,
        load_case=load_case,
        spec=spec,
        thresholds=thresholds,
    )
    verdict = (
        "PHYSICS_INCONCLUSIVE"
        if _physical_semantic_failures(physical_trace, spec)
        else "PHYSICAL_FAILURE"
        if physical_failures
        else "CLAIM_MISMATCH"
        if reduced_failures
        else "PASS"
    )
    return {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "verdict": verdict,
        "failure_reasons": [*physical_failures, *reduced_failures],
        "thresholds": {
            **thresholds,
            "source": "typed_bicycle_spec_and_frozen_task_contract",
        },
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "visible_chain_geometry": (
            physical_trace.get("functional_geometry", {}).get(
                "visible_chain_geometry"
            )
            if isinstance(
                physical_trace.get("functional_geometry"),
                Mapping,
            )
            else None
        ),
        "strict_non_claims": {
            "endpoint_only_ratio_used": False,
            "startup_transient_hidden": False,
            "engagement_motion_credited": False,
            "engagement_ratio_credited": False,
            "ideal_reference_is_physical_evidence": False,
            "tooth_contact_or_chain_fatigue_certified": False,
        },
    }
