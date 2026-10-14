from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .adapters import adapter_from_dict, get_adapter
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .rack_pinion_loads_v3 import RACK_PINION_ANTI_DROP_HOLD_S
from .rack_pinion_spec import RackPinionSpec, rack_pinion_from_dict

ANTI_DROP_PHASE_ID = "zero_input_hold"


@dataclass(frozen=True)
class RackPinionThresholds:
    home_progress_m: float
    outbound_progress_m: float
    output_progress_tolerance_m: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    max_physical_coupling_error_m: float
    max_reduced_coupling_error_m: float
    min_directional_progress_m: float
    max_zero_input_drift_m: float
    max_transverse_progress_m: float
    min_anti_drop_hold_s: float = RACK_PINION_ANTI_DROP_HOLD_S
    max_anti_drop_slip_m: float = 0.0
    source: str = "family_spec_rack_pinion_v1"

    @classmethod
    def for_spec(cls, spec: RackPinionSpec) -> "RackPinionThresholds":
        travel = spec.outbound_progress_m - spec.home_progress_m
        return cls(
            home_progress_m=spec.home_progress_m,
            outbound_progress_m=spec.outbound_progress_m,
            output_progress_tolerance_m=spec.output_progress_tolerance_m,
            max_support_drift_m=spec.max_support_drift_m,
            max_off_axis_rotation_rad=spec.max_off_axis_rotation_rad,
            max_physical_coupling_error_m=spec.max_coupling_error_m,
            max_reduced_coupling_error_m=min(spec.max_coupling_error_m, 5e-5),
            min_directional_progress_m=max(0.002, travel * 0.04),
            max_zero_input_drift_m=spec.output_progress_tolerance_m,
            max_transverse_progress_m=spec.output_progress_tolerance_m,
            min_anti_drop_hold_s=RACK_PINION_ANTI_DROP_HOLD_S,
            max_anti_drop_slip_m=spec.output_progress_tolerance_m,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MechanismValidationError("TRACE_SAMPLE_INVALID", f"{label} must be finite")
    return float(value)


def _adapter_spec(plan: MechanismPlan) -> RackPinionSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    return rack_pinion_from_dict(adapter.family_spec)


def _validate_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
) -> list[dict[str, Any]]:
    expected = {
        "status": "COMPLETED",
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "mechanism_plugin_id": "rack_pinion_v1",
    }
    for field, value in expected.items():
        if trace.get(field) != value:
            raise MechanismValidationError(
                "TRACE_IDENTITY_MISMATCH", f"{branch} trace field {field!r} is inconsistent"
            )
    samples = trace.get("samples")
    if not isinstance(samples, list) or not samples:
        raise MechanismValidationError("TRACE_SAMPLES_MISSING", f"{branch} trace has no samples")
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("phase_id") not in allowed_phases:
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID", f"{branch}.samples[{index}] has an invalid phase"
            )
        time_s = _finite(sample.get("time_s"), f"{branch}.samples[{index}].time_s")
        if time_s < previous_time - 1e-12:
            raise MechanismValidationError("TRACE_TIME_REVERSED", f"{branch} time is not monotonic")
        previous_time = time_s
        for field in (
            "rack_progress_m",
            "rack_speed_m_s",
            "pinion_angle_rad",
            "pinion_speed_rad_s",
            "coupling_error_m",
            "max_support_drift_m",
            "max_bushing_reaction_force_n",
            "off_axis_rotation_rad",
        ):
            _finite(sample.get(field), f"{branch}.samples[{index}].{field}")
        drifts = sample.get("support_drifts_m")
        if not isinstance(drifts, dict) or set(drifts) != {"pinion_bearing", "rack_guide"}:
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID", f"{branch}.samples[{index}] support drift inventory is invalid"
            )
    return samples


def _metrics(
    samples: list[dict[str, Any]], load_case: LoadCase, pinion_axis: tuple[float, float, float]
) -> dict[str, Any]:
    progress = [float(item["rack_progress_m"]) for item in samples]
    initial = progress[0]
    phase_bounds: dict[str, dict[str, float]] = {}
    phase_spans: dict[str, dict[str, float]] = {}
    phase_downward_slip: dict[str, float] = {}
    directional: dict[str, float] = {}
    for phase in load_case.phases:
        indices = [index for index, item in enumerate(samples) if item["phase_id"] == phase.phase_id]
        if indices:
            values = [progress[index] for index in indices]
            phase_bounds[phase.phase_id] = {"min_m": min(values), "max_m": max(values)}
            # Measure the phase span from the last sample before it (the phase
            # boundary in time), so a full-length phase reports its declared
            # duration instead of losing one sampling interval.
            start_index = max(0, indices[0] - 1)
            start_s = float(samples[start_index]["time_s"])
            end_s = float(samples[indices[-1]]["time_s"])
            phase_spans[phase.phase_id] = {
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": end_s - start_s,
            }
            # Worst peak-to-trough downward excursion inside the phase,
            # anchored at the value the phase started from.
            running_peak = progress[start_index]
            slip = 0.0
            for index in indices:
                running_peak = max(running_peak, progress[index])
                slip = max(slip, running_peak - progress[index])
            phase_downward_slip[phase.phase_id] = slip
        if phase.wrench_kind != "torque" or phase.direction_abs is None or not indices:
            continue
        baseline = progress[max(0, indices[0] - 1)]
        # Rack progress is opposite pinion-axis rotation: x = -r*theta.
        sign = -1.0 if sum(a * b for a, b in zip(phase.direction_abs, pinion_axis)) >= 0 else 1.0
        directional[phase.phase_id] = max(
            0.0, max(sign * (progress[index] - baseline) for index in indices)
        )
    return {
        "min_progress_m": min(progress),
        "max_progress_m": max(progress),
        "final_progress_m": progress[-1],
        "max_abs_progress_delta_m": max(abs(item - initial) for item in progress),
        "max_coupling_error_m": max(abs(float(item["coupling_error_m"])) for item in samples),
        "max_support_drift_m": max(float(item["max_support_drift_m"]) for item in samples),
        "max_off_axis_rotation_rad": max(float(item["off_axis_rotation_rad"]) for item in samples),
        "max_bushing_reaction_force_n": max(
            float(item["max_bushing_reaction_force_n"]) for item in samples
        ),
        "phase_progress_bounds_m": phase_bounds,
        "phase_time_spans_s": phase_spans,
        "phase_downward_slip_m": phase_downward_slip,
        "phase_directional_travel_m": directional,
    }


def _semantics_failures(
    spec: RackPinionSpec, physical: dict[str, Any], reduced: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    physical_engine = physical.get("engine", {})
    reduced_engine = reduced.get("engine", {})
    if physical_engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_rack_pinion_contains_ideal_links")
    if reduced_engine.get("ideal_links") != [
        "ChLinkLockRevolute",
        "ChLinkLockPrismatic",
        "ChLinkMateRackPinion",
    ]:
        failures.append("reduced_rack_pinion_ideal_topology_mismatch")
    physical_semantics = physical.get("branch_semantics", {})
    reduced_semantics = reduced.get("branch_semantics", {})
    for field in (
        "same_mass_kg",
        "same_inertia_xx_kg_m2",
        "same_initial_pose",
        "same_external_load_sha256",
        "family_spec_sha256",
        "holding_brake_model",
    ):
        if physical_semantics.get(field) != reduced_semantics.get(field):
            failures.append(f"branch_{field}_mismatch")
    for trace in (physical, reduced):
        geometry = trace.get("collision_geometry", {})
        if geometry.get("pinion_pitch_radius_m") != spec.pinion_pitch_radius_m:
            failures.append("rack_pinion_pitch_radius_mismatch")
            break
    return failures


def _behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    thresholds: RackPinionThresholds,
) -> list[str]:
    failures: list[str] = []
    tolerance = thresholds.output_progress_tolerance_m
    if metrics["min_progress_m"] < thresholds.home_progress_m - tolerance:
        failures.append(f"{branch}_rack_home_stop_exceeded")
    if metrics["max_progress_m"] > thresholds.outbound_progress_m + tolerance:
        failures.append(f"{branch}_rack_outbound_stop_exceeded")
    if metrics["max_support_drift_m"] > thresholds.max_support_drift_m:
        failures.append(f"{branch}_rack_support_drift_exceeded")
    if metrics["max_off_axis_rotation_rad"] > thresholds.max_off_axis_rotation_rad:
        failures.append(f"{branch}_rack_off_axis_rotation_exceeded")
    coupling_limit = (
        thresholds.max_physical_coupling_error_m
        if branch == "physical"
        else thresholds.max_reduced_coupling_error_m
    )
    if metrics["max_coupling_error_m"] > coupling_limit:
        failures.append(f"{branch}_rack_pinion_coupling_error_exceeded")
    if branch == "physical" and metrics["max_bushing_reaction_force_n"] <= 1e-9:
        failures.append("physical_rack_support_reaction_missing")
    if branch == "reduced" and metrics["max_bushing_reaction_force_n"] > 1e-9:
        failures.append("reduced_rack_has_finite_support_reaction")

    if load_case.program == "bidirectional_rack_pinion":
        for phase in load_case.phases:
            if phase.wrench_kind != "torque":
                continue
            if metrics["phase_directional_travel_m"].get(phase.phase_id, 0.0) < (
                thresholds.min_directional_progress_m
            ):
                failures.append(f"{branch}_{phase.phase_id}_did_not_move_rack_as_loaded")
        upper_settle = metrics["phase_progress_bounds_m"].get("upper_settle", {})
        if upper_settle.get("min_m", -math.inf) < (
            thresholds.outbound_progress_m - tolerance
        ):
            failures.append(f"{branch}_upper_dwell_not_held")
        if abs(metrics["final_progress_m"] - thresholds.home_progress_m) > tolerance:
            failures.append(f"{branch}_home_dwell_not_held")
    elif load_case.program == "zero_input_gravity":
        if metrics["max_abs_progress_delta_m"] > thresholds.max_zero_input_drift_m:
            failures.append(f"{branch}_rack_zero_input_drift")
    elif load_case.program == "transverse_rack_probe":
        if metrics["max_abs_progress_delta_m"] > thresholds.max_transverse_progress_m:
            failures.append(f"{branch}_rack_failed_transverse_rejection")
    elif load_case.program == "task_rack_roundtrip":
        lift = metrics["phase_progress_bounds_m"].get("crank_lift", {})
        if lift.get("max_m", -math.inf) < thresholds.outbound_progress_m - tolerance:
            failures.append(f"{branch}_rack_outbound_target_not_reached")
        if abs(metrics["final_progress_m"] - thresholds.home_progress_m) > tolerance:
            failures.append(f"{branch}_rack_did_not_return_home")
        # prevent_gate_drop_after_release: the raised gate must hold with zero
        # commanded input for at least the approved anti-drop interval.
        hold_phase = next(
            (item for item in load_case.phases if item.phase_id == ANTI_DROP_PHASE_ID),
            None,
        )
        hold_span = metrics["phase_time_spans_s"].get(ANTI_DROP_PHASE_ID)
        if hold_phase is None or hold_span is None:
            failures.append(f"{branch}_rack_anti_drop_interval_missing")
        else:
            if hold_phase.wrench_kind != "none":
                failures.append(f"{branch}_rack_anti_drop_input_not_zero")
            # Allow one sampling interval of slack: phase boundaries may fall
            # between fixed-rate samples.
            if (
                hold_span["duration_s"] + load_case.sample_interval_s + 1e-9
                < thresholds.min_anti_drop_hold_s
            ):
                failures.append(f"{branch}_rack_anti_drop_hold_too_short")
            hold = metrics["phase_progress_bounds_m"].get(ANTI_DROP_PHASE_ID, {})
            if hold.get("min_m", -math.inf) < thresholds.outbound_progress_m - tolerance:
                failures.append(f"{branch}_rack_anti_drop_not_held_at_top")
            slip = metrics["phase_downward_slip_m"].get(ANTI_DROP_PHASE_ID, math.inf)
            if slip > thresholds.max_anti_drop_slip_m:
                failures.append(f"{branch}_rack_gate_drop_exceeded")
    else:
        failures.append(f"{branch}_unknown_rack_pinion_load_program")
    return failures


def compare_rack_pinion_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    spec = _adapter_spec(plan)
    thresholds = RackPinionThresholds.for_spec(spec)
    try:
        physical_samples = _validate_trace(
            physical_trace, branch="physical", plan=plan, load_case=load_case
        )
        reduced_samples = _validate_trace(
            reduced_trace, branch="reduced", plan=plan, load_case=load_case
        )
    except MechanismValidationError as exc:
        return {
            "schema_version": "0.1",
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "purpose": load_case.purpose,
            "required_for_certificate": load_case.required_for_certificate,
            "verdict": "PHYSICS_INCONCLUSIVE",
            "failure_reasons": [exc.code],
            "detail": str(exc),
            "thresholds": thresholds.to_dict(),
        }
    physical_metrics = _metrics(physical_samples, load_case, spec.pinion_axis_abs)
    reduced_metrics = _metrics(reduced_samples, load_case, spec.pinion_axis_abs)
    semantics = _semantics_failures(spec, physical_trace, reduced_trace)
    physical_failures = _behavior_failures(
        physical_metrics, branch="physical", load_case=load_case, thresholds=thresholds
    )
    reduced_failures = _behavior_failures(
        reduced_metrics, branch="reduced", load_case=load_case, thresholds=thresholds
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
        "failure_reasons": [*semantics, *physical_failures, *reduced_failures],
        "thresholds": thresholds.to_dict(),
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "strict_non_claims": {
            "timing_or_velocity_match_used_as_gate": False,
            "physical_and_reduced_progress_required_to_match_pointwise": False,
            "gear_efficiency_or_tooth_stress_certified": False,
            "dynamic_response_certified": False,
            "exact_tooth_or_guide_contact_certified": False,
        },
    }
