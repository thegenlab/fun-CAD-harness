from __future__ import annotations

import math
from typing import Any

from .adapters import adapter_from_dict, get_adapter
from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .multi_output_shaft_spec import (
    MultiOutputShaftSpec,
    multi_output_shaft_from_dict,
)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MechanismValidationError(
            "MULTI_OUTPUT_TRACE_NUMBER_INVALID",
            f"{label} must be finite",
        )
    return float(value)


def _spec(plan: MechanismPlan) -> MultiOutputShaftSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    return multi_output_shaft_from_dict(adapter.family_spec)


def _validate_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: MultiOutputShaftSpec,
) -> list[dict[str, Any]]:
    if trace.get("status") != "COMPLETED":
        raise MechanismValidationError(
            "TRACE_NOT_COMPLETED",
            f"{branch} multi-output trace did not complete",
        )
    expected = {
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
            "multi-output trace identity differs at: "
            + ", ".join(mismatches),
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise MechanismValidationError(
            "TRACE_SAMPLES_MISSING",
            f"{branch} multi-output trace has too few samples",
        )
    body_ids = {item.body_id for item in spec.bodies}
    shaft_body_ids = {item.body_id for item in spec.shafts}
    shaft_ids = {item.shaft_id for item in spec.shafts}
    mesh_ids = {item.mesh_id for item in spec.gear_meshes}
    phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        label = f"{branch}.samples[{index}]"
        if not isinstance(sample, dict) or sample.get("phase_id") not in phases:
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID",
                f"{label} has an invalid phase",
            )
        time_s = _finite(sample.get("time_s"), f"{label}.time_s")
        if time_s < previous_time - 1.0e-12:
            raise MechanismValidationError(
                "TRACE_TIME_REVERSED",
                f"{branch} trace time is not monotonic",
            )
        previous_time = time_s
        for shaft in spec.shafts:
            _finite(
                sample.get(f"{shaft.body_id}_angle_rad"),
                f"{label}.{shaft.body_id}_angle_rad",
            )
            _finite(
                sample.get(f"{shaft.body_id}_speed_rad_s"),
                f"{label}.{shaft.body_id}_speed_rad_s",
            )
        for field in (
            "max_mesh_error_rad",
            "max_support_drift_m",
            "max_off_axis_rotation_rad",
        ):
            _finite(sample.get(field), f"{label}.{field}")
        inventories = (
            ("shaft_angles_rad", shaft_body_ids),
            ("shaft_speeds_rad_s", shaft_body_ids),
            ("mesh_errors_rad", mesh_ids),
            ("support_drifts_m", shaft_ids),
            ("off_axis_rotations_rad", shaft_ids),
            ("body_poses", body_ids),
        )
        for field, expected_ids in inventories:
            value = sample.get(field)
            if not isinstance(value, dict) or set(value) != expected_ids:
                raise MechanismValidationError(
                    "MULTI_OUTPUT_TRACE_INVENTORY_MISMATCH",
                    f"{label}.{field} changed",
                )
    return samples


def _phase_travel(
    samples: list[dict[str, Any]],
    body_id: str,
    phase_id: str,
) -> float:
    values = [
        float(sample["shaft_angles_rad"][body_id])
        for sample in samples
        if sample["phase_id"] == phase_id
    ]
    return values[-1] - values[0] if values else 0.0


def _metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    spec: MultiOutputShaftSpec,
) -> dict[str, Any]:
    input_values = [
        float(sample["shaft_angles_rad"][spec.input_body_id])
        for sample in samples
    ]
    initial_input = input_values[0]
    furthest_index = max(
        range(len(samples)),
        key=lambda index: abs(input_values[index] - initial_input),
    )
    input_delta = input_values[furthest_index] - initial_input
    body_delta = {
        shaft.body_id: (
            float(
                samples[furthest_index]["shaft_angles_rad"][
                    shaft.body_id
                ]
            )
            - float(samples[0]["shaft_angles_rad"][shaft.body_id])
        )
        for shaft in spec.shafts
    }
    output_to_input_ratio = {
        body_id: (
            body_delta[body_id] / input_delta
            if abs(input_delta) > 1.0e-9
            else None
        )
        for body_id in spec.output_body_ids
    }
    first_output, second_output = spec.output_body_ids[:2]
    output_pair_ratio = (
        body_delta[first_output] / body_delta[second_output]
        if abs(body_delta[second_output]) > 1.0e-9
        else None
    )
    phase_travel = {
        shaft.body_id: {
            phase.phase_id: _phase_travel(
                samples,
                shaft.body_id,
                phase.phase_id,
            )
            for phase in load_case.phases
        }
        for shaft in spec.shafts
    }
    return {
        "input_span_rad": max(input_values) - min(input_values),
        "shaft_angle_ranges_rad": {
            shaft.body_id: {
                "min": min(
                    float(sample["shaft_angles_rad"][shaft.body_id])
                    for sample in samples
                ),
                "max": max(
                    float(sample["shaft_angles_rad"][shaft.body_id])
                    for sample in samples
                ),
            }
            for shaft in spec.shafts
        },
        "input_delta_at_ratio_sample_rad": input_delta,
        "body_delta_at_ratio_sample_rad": body_delta,
        "output_to_input_ratio": output_to_input_ratio,
        "output_pair_ratio": output_pair_ratio,
        "phase_travel_rad": phase_travel,
        "max_mesh_error_rad": max(
            float(sample["max_mesh_error_rad"]) for sample in samples
        ),
        "max_support_drift_m": max(
            float(sample["max_support_drift_m"]) for sample in samples
        ),
        "max_off_axis_rotation_rad": max(
            float(sample["max_off_axis_rotation_rad"])
            for sample in samples
        ),
        "max_bushing_reaction_force_n": max(
            float(sample.get("max_bushing_reaction_force_n", 0.0))
            for sample in samples
        ),
    }


def _behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    spec: MultiOutputShaftSpec,
) -> list[str]:
    failures: list[str] = []
    if metrics["max_support_drift_m"] > spec.max_support_drift_m:
        failures.append(f"{branch}_shaft_support_drift_exceeded")
    if (
        metrics["max_off_axis_rotation_rad"]
        > spec.max_off_axis_rotation_rad
    ):
        failures.append(f"{branch}_shaft_off_axis_rotation_exceeded")
    if metrics["max_mesh_error_rad"] > spec.max_mesh_error_rad:
        failures.append(f"{branch}_gear_mesh_error_exceeded")
    if load_case.program == "zero_input_multi_output_shaft":
        return failures
    if load_case.program == "transverse_multi_output_shaft":
        if any(
            abs(
                metrics["phase_travel_rad"][body_id].get(
                    "off_axis_torque", 0.0
                )
            )
            > 0.10
            for body_id in spec.output_body_ids
        ):
            failures.append(f"{branch}_transverse_probe_drove_output")
        return failures
    forward_input = metrics["phase_travel_rad"][
        spec.input_body_id
    ].get("crank_forward", 0.0)
    reverse_input = metrics["phase_travel_rad"][
        spec.input_body_id
    ].get("crank_reverse", 0.0)
    if forward_input <= 0.01:
        failures.append(f"{branch}_crank_did_not_turn_forward")
    if reverse_input >= -0.005:
        failures.append(f"{branch}_crank_did_not_turn_reverse")
    for body_id in spec.output_body_ids:
        declared = spec.declared_ratio_for_body(body_id)
        forward = metrics["phase_travel_rad"][body_id].get(
            "crank_forward", 0.0
        )
        reverse = metrics["phase_travel_rad"][body_id].get(
            "crank_reverse", 0.0
        )
        if forward * declared <= 0.0:
            failures.append(
                f"{branch}_{body_id}_forward_direction_invalid"
            )
        if reverse * declared >= 0.0:
            failures.append(
                f"{branch}_{body_id}_reverse_direction_invalid"
            )
    if load_case.program == "task_egg_beater_cycle":
        for body_id in spec.output_body_ids:
            if (
                abs(metrics["body_delta_at_ratio_sample_rad"][body_id])
                < spec.minimum_output_travel_rad
            ):
                failures.append(
                    f"{branch}_{body_id}_task_rotation_too_small"
                )
            measured = metrics["output_to_input_ratio"][body_id]
            declared = spec.declared_ratio_for_body(body_id)
            if (
                not isinstance(measured, (int, float))
                or not math.isfinite(float(measured))
                or abs(measured - declared)
                > spec.input_output_ratio_tolerance
            ):
                failures.append(
                    f"{branch}_{body_id}_crank_ratio_invalid"
                )
        expected_pair = (
            spec.declared_ratio_for_body(spec.output_body_ids[0])
            / spec.declared_ratio_for_body(spec.output_body_ids[1])
        )
        measured_pair = metrics["output_pair_ratio"]
        if (
            not isinstance(measured_pair, (int, float))
            or not math.isfinite(float(measured_pair))
            or abs(measured_pair - expected_pair)
            > spec.output_pair_ratio_tolerance
        ):
            failures.append(f"{branch}_output_pair_ratio_invalid")
    return failures


def _semantic_failures(
    plan: MechanismPlan,
    physical: dict[str, Any],
    reduced: dict[str, Any] | None = None,
) -> list[str]:
    from .family_plugins import mechanism_adapter_for_plan

    plugin, adapter = mechanism_adapter_for_plan(plan)
    expected_links = list(
        plugin.expected_reduced_ideal_links(plan, adapter)
    )
    failures: list[str] = []
    physical_engine = physical.get("engine") or {}
    if physical_engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_multi_output_contains_ideal_joint")
    semantics = physical.get("branch_semantics") or {}
    spec = _spec(plan)
    if semantics.get("active_input_body_ids") != [spec.input_body_id]:
        failures.append("physical_output_shaft_actuated_directly")
    if semantics.get("direct_output_shaft_actuation") is not False:
        failures.append("physical_direct_output_actuation_not_disclosed")
    if reduced is not None:
        reduced_engine = reduced.get("engine") or {}
        if reduced_engine.get("ideal_links") != expected_links:
            failures.append("reduced_multi_output_topology_mismatch")
        if reduced_engine.get("internal_ideal_joint_count") != len(
            expected_links
        ):
            failures.append("reduced_multi_output_link_count_mismatch")
    return failures


def physical_multi_output_behavior(
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
        "max_support_drift_m": spec.max_support_drift_m,
        "max_off_axis_rotation_rad": spec.max_off_axis_rotation_rad,
        "max_mesh_error_rad": spec.max_mesh_error_rad,
        "minimum_output_travel_rad": spec.minimum_output_travel_rad,
        "output_pair_ratio_tolerance": (
            spec.output_pair_ratio_tolerance
        ),
        "input_output_ratio_tolerance": (
            spec.input_output_ratio_tolerance
        ),
        "source": "validated_multi_output_shaft_family_spec",
    }
    return metrics, thresholds, failures


def compare_multi_output_shaft_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    spec = _spec(plan)
    thresholds = physical_multi_output_behavior(
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
    semantics = _semantic_failures(
        plan,
        physical_trace,
        reduced_trace,
    )
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
        "required_for_certificate": (
            load_case.required_for_certificate
        ),
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
            "exact_tooth_impact_or_backlash_certified": False,
            "friction_or_efficiency_certified": False,
            "dynamic_response_certified": False,
        },
    }
