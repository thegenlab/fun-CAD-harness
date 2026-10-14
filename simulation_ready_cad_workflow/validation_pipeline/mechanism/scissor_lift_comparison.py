"""Trace-pair comparison for the ``hydraulic_scissor_lift`` family.

Mirrors ``rack_pinion_comparison_v3.py``: validates both branch traces,
checks branch semantics (the physical branch must contain zero ideal links,
the reduced branch must contain exactly the declared ideal topology), and
gates behavior on platform progress, joint drift, off-plane rotation, and
platform level error using the thresholds owned by the family spec.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

from .adapters import adapter_from_dict, get_adapter
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .scissor_lift_loads import cylinder_home_axis
from .scissor_lift_spec import ScissorLiftSpec, expected_ideal_links, scissor_lift_from_dict


EXPECTED_REDUCED_IDEAL_LINKS = (
    "ChLinkLockRevolute",
    "ChLinkLockRevolute",
    "ChLinkLockRevolute",
    "ChLinkLockRevolute",
    "ChLinkLockRevolute",
    "ChLinkLockRevolutePrismatic",
    "ChLinkLockRevolutePrismatic",
    "ChLinkLockCylindrical",
)


@dataclass(frozen=True)
class ScissorLiftThresholds:
    home_progress_m: float
    outbound_progress_m: float
    output_progress_tolerance_m: float
    max_joint_drift_m: float
    max_off_plane_rotation_rad: float
    platform_level_tolerance_rad: float
    min_directional_progress_m: float
    max_zero_input_drift_m: float
    max_transverse_progress_m: float
    minimum_finite_capture_overlap_m: float
    source: str = "family_spec_hydraulic_scissor_lift_v1"

    @classmethod
    def for_spec(cls, spec: ScissorLiftSpec) -> "ScissorLiftThresholds":
        travel = spec.outbound_progress_m - spec.home_progress_m
        return cls(
            home_progress_m=spec.home_progress_m,
            outbound_progress_m=spec.outbound_progress_m,
            output_progress_tolerance_m=spec.output_progress_tolerance_m,
            max_joint_drift_m=spec.max_joint_drift_m,
            max_off_plane_rotation_rad=spec.max_off_plane_rotation_rad,
            platform_level_tolerance_rad=spec.platform_level_tolerance_rad,
            min_directional_progress_m=max(0.002, travel * 0.04),
            max_zero_input_drift_m=spec.output_progress_tolerance_m,
            max_transverse_progress_m=spec.output_progress_tolerance_m,
            minimum_finite_capture_overlap_m=min(
                item.minimum_overlap_m
                for item in spec.finite_linear_captures
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise MechanismValidationError("TRACE_SAMPLE_INVALID", f"{label} must be finite")
    return float(value)


def _adapter_spec(plan: MechanismPlan) -> ScissorLiftSpec:
    adapter = (
        adapter_from_dict(plan.adapter_spec)
        if plan.adapter_spec is not None
        else get_adapter(str(plan.adapter_id))
    )
    return scissor_lift_from_dict(adapter.family_spec)


def _validate_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    spec: ScissorLiftSpec,
) -> list[dict[str, Any]]:
    expected = {
        "status": "COMPLETED",
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "mechanism_plugin_id": "hydraulic_scissor_lift_v1",
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
    joint_ids = set(spec.joint_ids)
    capture_ids = {
        item.capture_id for item in spec.finite_linear_captures
    }
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
            "platform_height_m",
            "platform_speed_m_s",
            "platform_progress_m",
            "cylinder_length_m",
            "max_joint_drift_m",
            "max_bushing_reaction_force_n",
            "off_plane_rotation_rad",
            "platform_level_error_rad",
            "minimum_finite_capture_overlap_m",
        ):
            _finite(sample.get(field), f"{branch}.samples[{index}].{field}")
        drifts = sample.get("joint_drifts_m")
        if not isinstance(drifts, dict) or set(drifts) != joint_ids:
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID", f"{branch}.samples[{index}] joint drift inventory is invalid"
            )
        capture_overlaps = sample.get("finite_capture_overlaps_m")
        if (
            not isinstance(capture_overlaps, dict)
            or set(capture_overlaps) != capture_ids
        ):
            raise MechanismValidationError(
                "TRACE_SAMPLE_INVALID",
                f"{branch}.samples[{index}] finite capture inventory is invalid",
            )
        for capture_id, overlap in capture_overlaps.items():
            _finite(
                overlap,
                f"{branch}.samples[{index}].finite_capture_overlaps_m."
                f"{capture_id}",
            )
    return samples


def _metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    home_axis: tuple[float, float, float],
) -> dict[str, Any]:
    progress = [float(item["platform_progress_m"]) for item in samples]
    initial = progress[0]
    phase_bounds: dict[str, dict[str, float]] = {}
    directional: dict[str, float] = {}
    for phase in load_case.phases:
        indices = [index for index, item in enumerate(samples) if item["phase_id"] == phase.phase_id]
        if indices:
            values = [progress[index] for index in indices]
            phase_bounds[phase.phase_id] = {"min_m": min(values), "max_m": max(values)}
        if phase.wrench_kind != "force" or phase.direction_abs is None or not indices:
            continue
        baseline = progress[max(0, indices[0] - 1)]
        # Extension direction is along the home cylinder axis: positive scalar
        # raises the platform, negative retracts it.
        sign = 1.0 if sum(a * b for a, b in zip(phase.direction_abs, home_axis)) >= 0 else -1.0
        directional[phase.phase_id] = max(
            0.0, max(sign * (progress[index] - baseline) for index in indices)
        )
    return {
        "min_progress_m": min(progress),
        "max_progress_m": max(progress),
        "final_progress_m": progress[-1],
        "max_abs_progress_delta_m": max(abs(item - initial) for item in progress),
        "max_joint_drift_m": max(float(item["max_joint_drift_m"]) for item in samples),
        "max_off_plane_rotation_rad": max(float(item["off_plane_rotation_rad"]) for item in samples),
        "max_platform_level_error_rad": max(
            float(item["platform_level_error_rad"]) for item in samples
        ),
        "max_bushing_reaction_force_n": max(
            float(item["max_bushing_reaction_force_n"]) for item in samples
        ),
        "minimum_finite_capture_overlap_m": min(
            float(item["minimum_finite_capture_overlap_m"])
            for item in samples
        ),
        "phase_progress_bounds_m": phase_bounds,
        "phase_directional_travel_m": directional,
    }


def _semantics_failures(
    spec: ScissorLiftSpec, physical: dict[str, Any], reduced: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    physical_engine = physical.get("engine", {})
    reduced_engine = reduced.get("engine", {})
    if physical_engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_scissor_lift_contains_ideal_links")
    if reduced_engine.get("ideal_links") != list(expected_ideal_links(spec)):
        failures.append("reduced_scissor_lift_ideal_topology_mismatch")
    physical_semantics = physical.get("branch_semantics", {})
    reduced_semantics = reduced.get("branch_semantics", {})
    for field in (
        "same_mass_kg",
        "same_inertia_xx_kg_m2",
        "same_initial_pose",
        "same_external_load_sha256",
        "family_spec_sha256",
        "actuator_model",
    ):
        if physical_semantics.get(field) != reduced_semantics.get(field):
            failures.append(f"branch_{field}_mismatch")
    for trace in (physical, reduced):
        geometry = trace.get("collision_geometry", {})
        if geometry.get("joint_count") != spec.joint_count:
            failures.append("scissor_lift_joint_count_mismatch")
            break
        capture = geometry.get("finite_linear_capture")
        if (
            not isinstance(capture, dict)
            or capture.get("passed") is not True
        ):
            failures.append("scissor_lift_finite_capture_geometry_missing")
            break
    return failures


def _behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    thresholds: ScissorLiftThresholds,
) -> list[str]:
    failures: list[str] = []
    tolerance = thresholds.output_progress_tolerance_m
    if metrics["min_progress_m"] < thresholds.home_progress_m - tolerance:
        failures.append(f"{branch}_platform_home_stop_exceeded")
    if metrics["max_progress_m"] > thresholds.outbound_progress_m + tolerance:
        failures.append(f"{branch}_platform_outbound_stop_exceeded")
    if metrics["max_joint_drift_m"] > thresholds.max_joint_drift_m:
        failures.append(f"{branch}_scissor_joint_drift_exceeded")
    if metrics["max_off_plane_rotation_rad"] > thresholds.max_off_plane_rotation_rad:
        failures.append(f"{branch}_scissor_off_plane_rotation_exceeded")
    if metrics["max_platform_level_error_rad"] > thresholds.platform_level_tolerance_rad:
        failures.append(f"{branch}_platform_level_error_exceeded")
    if (
        metrics["minimum_finite_capture_overlap_m"]
        < thresholds.minimum_finite_capture_overlap_m - 1e-9
    ):
        failures.append(f"{branch}_finite_capture_overlap_lost")
    if branch == "physical" and metrics["max_bushing_reaction_force_n"] <= 1e-9:
        failures.append("physical_scissor_lift_support_reaction_missing")
    if branch == "reduced" and metrics["max_bushing_reaction_force_n"] > 1e-9:
        failures.append("reduced_scissor_lift_has_finite_support_reaction")

    if load_case.program == "bidirectional_scissor_lift":
        for phase in load_case.phases:
            if phase.wrench_kind != "force":
                continue
            if metrics["phase_directional_travel_m"].get(phase.phase_id, 0.0) < (
                thresholds.min_directional_progress_m
            ):
                failures.append(f"{branch}_{phase.phase_id}_did_not_move_platform_as_loaded")
        raised_settle = metrics["phase_progress_bounds_m"].get("raised_settle", {})
        if raised_settle.get("min_m", -math.inf) < (
            thresholds.outbound_progress_m - tolerance
        ):
            failures.append(f"{branch}_raised_dwell_not_held")
        if abs(metrics["final_progress_m"] - thresholds.home_progress_m) > tolerance:
            failures.append(f"{branch}_home_dwell_not_held")
    elif load_case.program == "zero_input_gravity":
        if metrics["max_abs_progress_delta_m"] > thresholds.max_zero_input_drift_m:
            failures.append(f"{branch}_platform_zero_input_drift")
    elif load_case.program == "transverse_scissor_probe":
        if metrics["max_abs_progress_delta_m"] > thresholds.max_transverse_progress_m:
            failures.append(f"{branch}_platform_failed_transverse_rejection")
    elif load_case.program == "task_platform_roundtrip":
        extend = metrics["phase_progress_bounds_m"].get("cylinder_extend", {})
        if extend.get("max_m", -math.inf) < thresholds.outbound_progress_m - tolerance:
            failures.append(f"{branch}_platform_outbound_target_not_reached")
        if abs(metrics["final_progress_m"] - thresholds.home_progress_m) > tolerance:
            failures.append(f"{branch}_platform_did_not_return_home")
    else:
        failures.append(f"{branch}_unknown_scissor_lift_load_program")
    return failures


def compare_scissor_lift_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    spec = _adapter_spec(plan)
    thresholds = ScissorLiftThresholds.for_spec(spec)
    try:
        physical_samples = _validate_trace(
            physical_trace, branch="physical", plan=plan, load_case=load_case, spec=spec
        )
        reduced_samples = _validate_trace(
            reduced_trace, branch="reduced", plan=plan, load_case=load_case, spec=spec
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
    home_axis = cylinder_home_axis(spec)
    physical_metrics = _metrics(physical_samples, load_case, home_axis)
    reduced_metrics = _metrics(reduced_samples, load_case, home_axis)
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
            "hydraulic_pressure_flow_or_seal_friction_certified": False,
            "dynamic_response_certified": False,
            "exact_joint_surface_contact_certified": False,
        },
    }
