from __future__ import annotations

import math
from typing import Any, Mapping

from .adapters import adapter_from_dict, get_adapter
from .click_pen_spec import ClickPenCamLatchSpec, click_pen_cam_latch_from_dict
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MechanismValidationError(
            "CLICK_PEN_TRACE_NUMBER_INVALID",
            f"{label} must be finite",
        )
    return float(value)


def _spec(plan: MechanismPlan) -> ClickPenCamLatchSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    return click_pen_cam_latch_from_dict(adapter.family_spec)


def _validate_trace(
    trace: Mapping[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: ClickPenCamLatchSpec,
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
            "click-pen trace differs at: " + ", ".join(mismatches),
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise MechanismValidationError(
            "TRACE_SAMPLES_MISSING",
            f"{branch} click-pen trace has too few samples",
        )
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    body_ids = {body.body_id for body in spec.bodies}
    support_ids = {
        spec.button_prismatic_joint.joint_id,
        spec.refill_prismatic_joint.joint_id,
        spec.cam_cylindrical_joint.joint_id,
    }
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        label = f"{branch}.samples[{index}]"
        if (
            not isinstance(sample, dict)
            or sample.get("phase_id") not in allowed_phases
        ):
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID",
                f"{label} has an invalid phase",
            )
        time_s = _finite(sample.get("time_s"), f"{label}.time_s")
        if time_s < previous_time - 1.0e-12:
            raise MechanismValidationError(
                "TRACE_TIME_REVERSED",
                f"{branch} click-pen time is not monotonic",
            )
        previous_time = time_s
        for field in (
            "button_press_depth_m",
            "button_speed_m_s",
            "refill_extension_m",
            "refill_speed_m_s",
            "cam_axial_position_m",
            "cam_index_angle_rad",
            "max_support_drift_m",
            "max_off_axis_rotation_rad",
        ):
            _finite(sample.get(field), f"{label}.{field}")
        if not isinstance(sample.get("extended_latched"), bool):
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID",
                f"{label}.extended_latched must be boolean",
            )
        if (
            not isinstance(sample.get("body_poses"), dict)
            or set(sample["body_poses"]) != body_ids
        ):
            raise MechanismValidationError(
                "CLICK_PEN_TRACE_INVENTORY_MISMATCH",
                f"{label}.body_poses changed",
            )
        support_drifts = sample.get("support_drifts_m")
        if branch == "physical" and (
            not isinstance(support_drifts, dict)
            or set(support_drifts) != support_ids
        ):
            raise MechanismValidationError(
                "CLICK_PEN_TRACE_INVENTORY_MISMATCH",
                f"{label}.support_drifts_m changed",
            )
    return samples


def _phase_samples(
    samples: list[dict[str, Any]],
    phase_id: str,
) -> list[dict[str, Any]]:
    return [sample for sample in samples if sample["phase_id"] == phase_id]


def _metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    spec: ClickPenCamLatchSpec,
) -> dict[str, Any]:
    refill = [float(sample["refill_extension_m"]) for sample in samples]
    button = [float(sample["button_press_depth_m"]) for sample in samples]
    cam_angles = [float(sample["cam_index_angle_rad"]) for sample in samples]
    release_end_states: list[dict[str, Any]] = []
    for phase in load_case.phases:
        if phase.wrench_kind != "none":
            continue
        phase_values = _phase_samples(samples, phase.phase_id)
        if phase_values:
            final = phase_values[-1]
            release_end_states.append(
                {
                    "phase_id": phase.phase_id,
                    "refill_extension_m": float(final["refill_extension_m"]),
                    "button_press_depth_m": float(final["button_press_depth_m"]),
                    "cam_index_angle_rad": float(final["cam_index_angle_rad"]),
                    "extended_latched": bool(final["extended_latched"]),
                    "latch_contact_count": int(
                        final.get("latch_contact_count", 0)
                    ),
                }
            )
    final = samples[-1]
    return {
        "min_refill_extension_m": min(refill),
        "max_refill_extension_m": max(refill),
        "final_refill_extension_m": refill[-1],
        "cumulative_refill_travel_m": sum(
            abs(current - previous)
            for previous, current in zip(refill, refill[1:])
        ),
        "max_button_press_depth_m": max(button),
        "final_button_press_depth_m": button[-1],
        "max_abs_cam_index_angle_rad": max(abs(item) for item in cam_angles),
        "final_cam_index_angle_rad": cam_angles[-1],
        "final_extended_latched": bool(final["extended_latched"]),
        "final_latch_contact_count": int(
            final.get("latch_contact_count", 0)
        ),
        "max_support_drift_m": max(
            float(sample["max_support_drift_m"]) for sample in samples
        ),
        "max_off_axis_rotation_rad": max(
            float(sample["max_off_axis_rotation_rad"]) for sample in samples
        ),
        "release_end_states": release_end_states,
        "extended_release_count": sum(
            item["extended_latched"] for item in release_end_states
        ),
        "retracted_release_count": sum(
            abs(item["refill_extension_m"])
            <= spec.click_state_machine.latch_contact_tolerance_m
            for item in release_end_states
        ),
    }


def _behavior_failures(
    metrics: Mapping[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    spec: ClickPenCamLatchSpec,
) -> list[str]:
    failures: list[str] = []
    tolerance = spec.click_state_machine.latch_contact_tolerance_m
    index_angle = spec.cam_cylindrical_joint.index_angle_rad
    if metrics["max_support_drift_m"] > spec.max_support_drift_m:
        failures.append(f"{branch}_click_pen_support_drift_exceeded")
    if metrics["max_off_axis_rotation_rad"] > spec.max_off_axis_rotation_rad:
        failures.append(f"{branch}_click_pen_off_axis_rotation_exceeded")
    if load_case.program == "zero_input_click_pen":
        if (
            metrics["max_refill_extension_m"]
            - metrics["min_refill_extension_m"]
            > tolerance
        ):
            failures.append(f"{branch}_zero_input_refill_drift")
        if metrics["max_button_press_depth_m"] > tolerance:
            failures.append(f"{branch}_zero_input_button_drift")
        return failures

    if (
        metrics["max_button_press_depth_m"]
        < 0.75 * spec.button_prismatic_joint.maximum_position_m
    ):
        failures.append(f"{branch}_button_press_too_shallow")
    if (
        abs(metrics["final_button_press_depth_m"])
        > max(tolerance, 0.0008)
    ):
        failures.append(f"{branch}_button_did_not_spring_return")

    one_click = load_case.program in {
        "pen_first_click",
        "task_pen_first_click",
    }
    two_click = load_case.program in {
        "pen_two_click_cycle",
        "task_pen_second_click",
    }
    repeat = load_case.program == "task_pen_repeat_cycles"
    if one_click:
        if (
            metrics["max_refill_extension_m"]
            < spec.click_state_machine.first_click_target_m - tolerance
        ):
            failures.append(f"{branch}_first_click_tip_extension_too_small")
        if (
            abs(
                metrics["final_refill_extension_m"]
                - spec.click_state_machine.first_click_target_m
            )
            > tolerance
        ):
            failures.append(f"{branch}_first_click_did_not_latch_extended")
        if not metrics["final_extended_latched"]:
            failures.append(f"{branch}_first_click_latch_contact_missing")
        if (
            metrics["max_abs_cam_index_angle_rad"]
            < 0.6 * index_angle
        ):
            failures.append(f"{branch}_first_click_cam_did_not_index")
    elif two_click or repeat:
        required_cycles = 2 if repeat else 1
        required_travel = 0.014 * required_cycles
        if metrics["max_refill_extension_m"] < 0.007:
            failures.append(f"{branch}_click_cycle_never_extended_tip")
        if abs(metrics["final_refill_extension_m"]) > tolerance:
            failures.append(f"{branch}_click_cycle_did_not_retract_tip")
        if metrics["cumulative_refill_travel_m"] < required_travel:
            failures.append(f"{branch}_click_cycle_travel_too_small")
        if metrics["extended_release_count"] < required_cycles:
            failures.append(f"{branch}_extended_latch_state_count_too_small")
        if metrics["retracted_release_count"] < required_cycles:
            failures.append(f"{branch}_retracted_state_count_too_small")
        required_indexes = 1.4 if two_click else 3.2
        if (
            metrics["max_abs_cam_index_angle_rad"]
            < required_indexes * index_angle
        ):
            failures.append(f"{branch}_cam_index_count_too_small")
    if branch == "physical":
        for release in metrics["release_end_states"]:
            if (
                release["extended_latched"]
                and release["latch_contact_count"] <= 0
            ):
                failures.append("physical_extended_state_has_no_surface_contact")
                break
    return failures


def _semantic_failures(
    plan: MechanismPlan,
    physical: Mapping[str, Any],
    reduced: Mapping[str, Any] | None = None,
) -> list[str]:
    from .family_plugins import mechanism_adapter_for_plan

    plugin, adapter = mechanism_adapter_for_plan(plan)
    expected_links = list(plugin.expected_reduced_ideal_links(plan, adapter))
    engine = physical.get("engine") or {}
    semantics = physical.get("branch_semantics") or {}
    failures: list[str] = []
    if engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_click_pen_contains_ideal_joint")
    if semantics.get("direct_output_actuation") is not False:
        failures.append("physical_refill_actuated_directly")
    if semantics.get("button_is_only_actuated_body") is not True:
        failures.append("physical_click_pen_has_undeclared_actuation")
    if semantics.get("cam_index_torque_is_analytical") is not False:
        failures.append("physical_cam_index_is_analytical")
    if semantics.get("latch_force_is_analytical") is not False:
        failures.append("physical_click_latch_is_analytical")
    if reduced is not None:
        if (reduced.get("engine") or {}).get("ideal_links") != expected_links:
            failures.append("reduced_click_pen_topology_mismatch")
    return failures


def physical_click_pen_behavior(
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
        "first_click_target_m": spec.click_state_machine.first_click_target_m,
        "latch_contact_tolerance_m": (
            spec.click_state_machine.latch_contact_tolerance_m
        ),
        "index_angle_rad": spec.cam_cylindrical_joint.index_angle_rad,
        "max_support_drift_m": spec.max_support_drift_m,
        "max_off_axis_rotation_rad": spec.max_off_axis_rotation_rad,
        "source": "validated_click_pen_cam_latch_family_spec",
    }
    return metrics, thresholds, failures


def compare_click_pen_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    spec = _spec(plan)
    thresholds = physical_click_pen_behavior(
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
            "writing_quality_certified": False,
            "dynamic_response_certified": False,
        },
    }
