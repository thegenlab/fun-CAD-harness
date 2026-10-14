from __future__ import annotations

import math
from typing import Any, Mapping

from .adapters import adapter_from_dict, get_adapter
from .detent_slider_spec import (
    PrismaticDetentLatchSpec,
    prismatic_detent_latch_from_dict,
)
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MechanismValidationError(
            "DETENT_SLIDER_TRACE_NUMBER_INVALID",
            f"{label} must be finite",
        )
    return float(value)


def _spec(plan: MechanismPlan) -> PrismaticDetentLatchSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    return prismatic_detent_latch_from_dict(adapter.family_spec)


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: PrismaticDetentLatchSpec,
) -> list[dict[str, Any]]:
    expected = {
        "status": "COMPLETED",
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches = [
        key for key, value in expected.items() if trace.get(key) != value
    ]
    if mismatches:
        raise MechanismValidationError(
            "TRACE_IDENTITY_MISMATCH",
            "detent-slider trace differs at: " + ", ".join(mismatches),
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise MechanismValidationError(
            "TRACE_SAMPLES_MISSING",
            f"{branch} detent-slider trace has too few samples",
        )
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    body_ids = {body.body_id for body in spec.bodies}
    support_ids = {
        spec.blade_prismatic_joint.joint_id,
        spec.latch_press_joint.joint_id,
    }
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        label = f"{branch}.samples[{index}]"
        if not isinstance(sample, dict) or sample.get("phase_id") not in allowed_phases:
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID",
                f"{label} has an invalid phase",
            )
        time_s = _finite(sample.get("time_s"), f"{label}.time_s")
        if time_s < previous_time - 1.0e-12:
            raise MechanismValidationError(
                "TRACE_TIME_REVERSED",
                f"{branch} detent-slider time is not monotonic",
            )
        previous_time = time_s
        for field in (
            "blade_carriage_position_m",
            "blade_carriage_speed_m_s",
            "thumb_latch_position_m",
            "thumb_press_depth_m",
            "detent_position_error_m",
            "max_support_drift_m",
            "max_off_axis_rotation_rad",
        ):
            _finite(sample.get(field), f"{label}.{field}")
        if not isinstance(sample.get("detent_engaged"), bool):
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID",
                f"{label}.detent_engaged must be boolean",
            )
        if not isinstance(sample.get("body_poses"), dict) or set(
            sample["body_poses"]
        ) != body_ids:
            raise MechanismValidationError(
                "DETENT_SLIDER_TRACE_INVENTORY_MISMATCH",
                f"{label}.body_poses changed",
            )
        support_drifts = sample.get("support_drifts_m")
        if branch == "physical" and (
            not isinstance(support_drifts, dict)
            or set(support_drifts) != support_ids
        ):
            raise MechanismValidationError(
                "DETENT_SLIDER_TRACE_INVENTORY_MISMATCH",
                f"{label}.support_drifts_m changed",
            )
    return samples


def _phase_values(
    samples: list[dict[str, Any]],
    phase_id: str,
) -> list[float]:
    return [
        float(sample["blade_carriage_position_m"])
        for sample in samples
        if sample["phase_id"] == phase_id
    ]


def _metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    spec: PrismaticDetentLatchSpec,
) -> dict[str, Any]:
    positions = [
        float(sample["blade_carriage_position_m"]) for sample in samples
    ]
    pair_key = "|".join(
        sorted(
            (
                spec.detent_interface.rack_body_id,
                spec.detent_interface.tab_body_id,
            )
        )
    )
    contact_phase_counts: dict[str, int] = {}
    for phase in load_case.phases:
        contact_phase_counts[phase.phase_id] = max(
            (
                int((sample.get("contact_pair_counts") or {}).get(pair_key, 0))
                for sample in samples
                if sample["phase_id"] == phase.phase_id
            ),
            default=0,
        )
    phase_bounds = {
        phase.phase_id: {
            "min_m": min(values),
            "max_m": max(values),
            "start_m": values[0],
            "end_m": values[-1],
            "slip_m": max(values) - min(values),
        }
        for phase in load_case.phases
        if (values := _phase_values(samples, phase.phase_id))
    }
    final = samples[-1]
    return {
        "min_position_m": min(positions),
        "max_position_m": max(positions),
        "final_position_m": positions[-1],
        "cumulative_travel_m": sum(
            abs(current - previous)
            for previous, current in zip(positions, positions[1:])
        ),
        "max_press_depth_m": max(
            float(sample["thumb_press_depth_m"]) for sample in samples
        ),
        "final_press_depth_m": float(final["thumb_press_depth_m"]),
        "final_detent_engaged": bool(final["detent_engaged"]),
        "final_locked_detent_position_m": final.get(
            "locked_detent_position_m"
        ),
        "max_support_drift_m": max(
            float(sample["max_support_drift_m"]) for sample in samples
        ),
        "max_off_axis_rotation_rad": max(
            float(sample["max_off_axis_rotation_rad"])
            for sample in samples
        ),
        "phase_position_bounds_m": phase_bounds,
        "detent_contact_pair_key": pair_key,
        "detent_contact_count_by_phase": contact_phase_counts,
    }


def _target_for_move(load_case: LoadCase) -> float | None:
    for phase in load_case.phases:
        if phase.stop_condition is not None:
            return float(phase.target_coordinate)
    return None


def _behavior_failures(
    metrics: Mapping[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    spec: PrismaticDetentLatchSpec,
) -> list[str]:
    failures: list[str] = []
    tolerance = spec.detent_interface.capture_tolerance_m
    if metrics["max_support_drift_m"] > spec.max_support_drift_m:
        failures.append(f"{branch}_detent_slider_support_drift_exceeded")
    if metrics["max_off_axis_rotation_rad"] > spec.max_off_axis_rotation_rad:
        failures.append(f"{branch}_detent_slider_off_axis_rotation_exceeded")

    if load_case.program == "zero_input_detent_slider":
        if metrics["max_position_m"] - metrics["min_position_m"] > tolerance:
            failures.append(f"{branch}_zero_input_detent_drift")
        return failures

    target = _target_for_move(load_case)
    if target is not None and metrics["max_position_m"] < target - tolerance:
        failures.append(f"{branch}_selected_detent_not_reached")
    if metrics["max_press_depth_m"] < spec.detent_interface.release_press_m:
        failures.append(f"{branch}_latch_was_not_pressed_far_enough")
    if not metrics["final_detent_engaged"]:
        failures.append(f"{branch}_final_detent_not_engaged")

    if load_case.program == "hold_detent_under_axial_load" or (
        load_case.program == "task_detent_hold_under_load"
    ):
        hold = metrics["phase_position_bounds_m"].get("axial_load_hold")
        if not isinstance(hold, Mapping) or hold["slip_m"] > tolerance:
            failures.append(f"{branch}_detent_slipped_under_axial_load")
        if abs(metrics["final_position_m"] - 0.045) > tolerance:
            failures.append(f"{branch}_working_detent_position_invalid")
        if (
            branch == "physical"
            and metrics["detent_contact_count_by_phase"].get(
                "axial_load_hold", 0
            )
            <= 0
        ):
            failures.append("physical_detent_hold_has_no_surface_contact")
    elif load_case.program == "task_detent_cycle":
        if metrics["max_position_m"] < 0.065 - tolerance:
            failures.append(f"{branch}_outer_detent_not_reached")
        if abs(metrics["final_position_m"]) > tolerance:
            failures.append(f"{branch}_blade_did_not_return_home")
        if metrics["cumulative_travel_m"] < 0.09:
            failures.append(f"{branch}_blade_cycle_travel_too_small")
        if (
            branch == "physical"
            and metrics["detent_contact_count_by_phase"].get(
                "release_home_detent", 0
            )
            <= 0
        ):
            failures.append("physical_home_detent_has_no_surface_contact")
    else:
        expected = 0.025 if (
            load_case.program == "task_detent_second_position"
            or load_case.load_case_id == "held_out_second_detent"
        ) else target
        if expected is not None and abs(
            metrics["final_position_m"] - expected
        ) > tolerance:
            failures.append(f"{branch}_selected_detent_position_invalid")
        release_phases = [
            phase.phase_id
            for phase in load_case.phases
            if phase.wrench_kind == "none"
        ]
        if branch == "physical" and not any(
            metrics["detent_contact_count_by_phase"].get(phase_id, 0) > 0
            for phase_id in release_phases
        ):
            failures.append("physical_released_detent_has_no_surface_contact")
    return failures


def _semantic_failures(
    plan: MechanismPlan,
    physical: Mapping[str, Any],
    reduced: Mapping[str, Any] | None = None,
) -> list[str]:
    from .family_plugins import mechanism_adapter_for_plan

    plugin, adapter = mechanism_adapter_for_plan(plan)
    expected_links = list(plugin.expected_reduced_ideal_links(plan, adapter))
    failures: list[str] = []
    engine = physical.get("engine") or {}
    semantics = physical.get("branch_semantics") or {}
    if engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_detent_slider_contains_ideal_joint")
    if semantics.get("direct_output_actuation") is not False:
        failures.append("physical_blade_carriage_actuated_directly")
    if semantics.get("detent_force_is_analytical") is not False:
        failures.append("physical_detent_lock_is_analytical")
    if reduced is not None:
        reduced_engine = reduced.get("engine") or {}
        if reduced_engine.get("ideal_links") != expected_links:
            failures.append("reduced_detent_slider_topology_mismatch")
    return failures


def physical_detent_slider_behavior(
    plan: MechanismPlan,
    load_case: LoadCase,
    trace: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    spec = _spec(plan)
    samples = _validate_trace(
        trace,
        branch="physical",
        plan=plan,
        load_case=load_case,
        spec=spec,
    )
    metrics = _metrics(samples, load_case, spec)
    failures = [
        *_semantic_failures(plan, trace),
        *_behavior_failures(
            metrics,
            branch="physical",
            load_case=load_case,
            spec=spec,
        ),
    ]
    thresholds = {
        "capture_tolerance_m": spec.detent_interface.capture_tolerance_m,
        "release_press_m": spec.detent_interface.release_press_m,
        "holding_force_n": spec.detent_interface.holding_force_n,
        "max_support_drift_m": spec.max_support_drift_m,
        "max_off_axis_rotation_rad": spec.max_off_axis_rotation_rad,
        "source": "validated_prismatic_detent_latch_family_spec",
    }
    return metrics, thresholds, failures


def compare_detent_slider_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    spec = _spec(plan)
    thresholds = physical_detent_slider_behavior(
        plan,
        load_case,
        physical_trace,
    )[1]
    try:
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
            "program": load_case.program,
            "verdict": "PHYSICS_INCONCLUSIVE",
            "failure_reasons": [exc.code],
            "detail": str(exc),
            "thresholds": thresholds,
        }
    physical_metrics = _metrics(physical_samples, load_case, spec)
    reduced_metrics = _metrics(reduced_samples, load_case, spec)
    semantics = _semantic_failures(plan, physical_trace, reduced_trace)
    physical_failures = _behavior_failures(
        physical_metrics,
        branch="physical",
        load_case=load_case,
        spec=spec,
    )
    reduced_failures = _behavior_failures(
        reduced_metrics,
        branch="reduced",
        load_case=load_case,
        spec=spec,
    )
    verdict = (
        "PHYSICS_INCONCLUSIVE"
        if semantics
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
        "failure_reasons": [
            *semantics,
            *physical_failures,
            *reduced_failures,
        ],
        "thresholds": thresholds,
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "strict_non_claims": {
            "wear_life_certified": False,
            "friction_or_efficiency_certified": False,
            "dynamic_response_certified": False,
        },
    }
