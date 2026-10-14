from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .adapters import adapter_from_dict, get_adapter
from .io import canonical_sha256, read_json, read_trace_json
from .family_plugins import (
    invoke_mechanism_plugin_callback,
    mechanism_adapter_for,
    plugin_registration_or_none,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import plugin_boundary_error


@dataclass(frozen=True)
class KinematicThresholds:
    min_directional_travel_m: float
    max_lateral_shift_m: float
    max_rotation_rad: float
    max_zero_input_drift_m: float
    max_transverse_motion_m: float
    min_axis_purity: float
    source: str = "provisional_protocol_default_scaled_by_characteristic_length"

    @classmethod
    def for_plan(cls, plan: MechanismPlan) -> KinematicThresholds:
        length = plan.characteristic_length_m
        return cls(
            min_directional_travel_m=max(1e-4, length * 0.001),
            max_lateral_shift_m=max(5e-4, length * 0.005),
            max_rotation_rad=math.radians(3.0),
            max_zero_input_drift_m=max(1e-4, length * 0.0025),
            max_transverse_motion_m=max(5e-4, length * 0.005),
            min_axis_purity=0.95,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_directional_travel_m": self.min_directional_travel_m,
            "max_lateral_shift_m": self.max_lateral_shift_m,
            "max_rotation_rad": self.max_rotation_rad,
            "max_zero_input_drift_m": self.max_zero_input_drift_m,
            "max_transverse_motion_m": self.max_transverse_motion_m,
            "min_axis_purity": self.min_axis_purity,
            "source": self.source,
        }


@dataclass(frozen=True)
class RevoluteKinematicThresholds:
    min_directional_angle_rad: float = math.radians(0.5)
    max_pivot_drift_m: float = 8e-4
    max_off_axis_rotation_rad: float = math.radians(3.0)
    max_transverse_angle_rad: float = math.radians(2.0)
    source: str = "provisional_plain_revolute_protocol_default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_directional_angle_rad": self.min_directional_angle_rad,
            "max_pivot_drift_m": self.max_pivot_drift_m,
            "max_off_axis_rotation_rad": self.max_off_axis_rotation_rad,
            "max_transverse_angle_rad": self.max_transverse_angle_rad,
            "source": self.source,
        }


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _adapter_for_plan(plan: MechanismPlan):
    if plan.adapter_spec is not None:
        return adapter_from_dict(plan.adapter_spec)
    if plan.adapter_id is None:
        raise KeyError("mechanism plan has no adapter")
    return get_adapter(plan.adapter_id)


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("TRACE_NUMBER_REQUIRED", f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise _fail("TRACE_NUMBER_NOT_FINITE", f"{label} must be finite")
    return number


def _vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise _fail("TRACE_VECTOR_INVALID", f"{label} must contain three numbers")
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), abs_tol=tolerance, rel_tol=1e-9)
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return all(_close(a, b, tolerance=tolerance) for a, b in zip(left, right))
    if isinstance(left, dict) and isinstance(right, dict) and set(left) == set(right):
        return all(_close(left[key], right[key], tolerance=tolerance) for key in left)
    return left == right


def _validate_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
) -> list[dict[str, Any]]:
    if trace.get("status") != "COMPLETED":
        raise _fail(
            "TRACE_NOT_COMPLETED",
            f"{branch} trace status is {trace.get('status')!r}",
        )
    expected_identity = {
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches = [
        key for key, expected in expected_identity.items() if trace.get(key) != expected
    ]
    if mismatches:
        raise _fail(
            "TRACE_IDENTITY_MISMATCH",
            f"{branch} trace identity mismatch: {', '.join(mismatches)}",
        )
    axis = _vector3(trace.get("axis_abs"), f"{branch}.axis_abs")
    if not _close(list(axis), list(plan.axis_abs), tolerance=1e-9):
        raise _fail("TRACE_AXIS_MISMATCH", f"{branch} trace axis changed")
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise _fail("TRACE_SAMPLES_MISSING", f"{branch} trace has too few samples")
    previous_time = -math.inf
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise _fail("TRACE_SAMPLE_INVALID", f"{branch}.samples[{index}] is invalid")
        time_s = _finite(sample.get("time_s"), f"{branch}.samples[{index}].time_s")
        if time_s < previous_time - 1e-12:
            raise _fail("TRACE_TIME_REVERSED", f"{branch} sample time is not monotonic")
        previous_time = time_s
        if sample.get("phase_id") not in allowed_phases:
            raise _fail("TRACE_PHASE_UNKNOWN", f"{branch} trace contains an unknown phase")
        for field in (
            "generalized_progress_m",
            "axis_speed_m_s",
            "lateral_shift_m",
            "rotation_from_start_rad",
        ):
            _finite(sample.get(field), f"{branch}.samples[{index}].{field}")
        _vector3(
            sample.get("observable_displacement_abs_m"),
            f"{branch}.samples[{index}].observable_displacement_abs_m",
        )
    return samples


def _phase_directional_travel(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    axis: tuple[float, float, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for phase_index, phase in enumerate(load_case.phases):
        if phase.wrench_kind == "none" or phase.direction_abs is None:
            continue
        response_phase_ids = {phase.phase_id}
        for following in load_case.phases[phase_index + 1 :]:
            if following.wrench_kind != "none":
                break
            response_phase_ids.add(following.phase_id)
        indices = [
            index
            for index, sample in enumerate(samples)
            if sample["phase_id"] in response_phase_ids
        ]
        if not indices:
            result[phase.phase_id] = 0.0
            continue
        first = indices[0]
        baseline = float(samples[max(0, first - 1)]["generalized_progress_m"])
        sign = 1.0 if sum(a * b for a, b in zip(phase.direction_abs, axis)) >= 0 else -1.0
        result[phase.phase_id] = max(
            0.0,
            max(
                sign * (float(samples[index]["generalized_progress_m"]) - baseline)
                for index in indices
            ),
        )
    return result


def _trace_metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    axis: tuple[float, float, float],
) -> dict[str, Any]:
    progress = [float(sample["generalized_progress_m"]) for sample in samples]
    lateral = [float(sample["lateral_shift_m"]) for sample in samples]
    rotation = [float(sample["rotation_from_start_rad"]) for sample in samples]
    axis_speed = [
        float(sample["axis_speed_m_s"]) for sample in samples
    ]
    displacement_vectors = [
        _vector3(sample["observable_displacement_abs_m"], "sample.displacement")
        for sample in samples
    ]
    projected_energy = sum(value * value for value in progress)
    displacement_energy = sum(
        sum(component * component for component in displacement)
        for displacement in displacement_vectors
    )
    purity = (
        math.sqrt(projected_energy / displacement_energy)
        if displacement_energy > 1e-18
        else 1.0
    )
    contacts = [
        int(sample["contact_count"])
        for sample in samples
        if isinstance(sample.get("contact_count"), int)
        and not isinstance(sample.get("contact_count"), bool)
    ]
    phase_progress_bounds: dict[str, dict[str, float]] = {}
    for phase in load_case.phases:
        values = [
            float(sample["generalized_progress_m"])
            for sample in samples
            if sample["phase_id"] == phase.phase_id
        ]
        if values:
            phase_progress_bounds[phase.phase_id] = {
                "min_m": min(values),
                "max_m": max(values),
            }
    return {
        "max_abs_progress_m": max(abs(value) for value in progress),
        "min_progress_m": min(progress),
        "max_progress_m": max(progress),
        "progress_range_m": max(progress) - min(progress),
        "final_progress_m": progress[-1],
        "max_abs_axis_speed_m_s": max(abs(value) for value in axis_speed),
        "max_lateral_shift_m": max(lateral),
        "max_rotation_rad": max(rotation),
        "max_total_displacement_m": max(
            math.sqrt(sum(component * component for component in displacement))
            for displacement in displacement_vectors
        ),
        "axis_purity": purity,
        "contact_observed": any(value > 0 for value in contacts),
        "max_contact_count": max(contacts) if contacts else None,
        "phase_directional_travel_m": _phase_directional_travel(
            samples, load_case, axis
        ),
        "phase_progress_bounds_m": phase_progress_bounds,
    }


def _branch_semantics_failures(
    plan: MechanismPlan,
    physical: dict[str, Any],
    reduced: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    physical_engine = physical.get("engine", {})
    reduced_engine = reduced.get("engine", {})
    adapter = _adapter_for_plan(plan)
    plugin = mechanism_adapter_for(plan, adapter)
    callback_results = {
        callback: invoke_mechanism_plugin_callback(
            plugin,
            callback,
            plan,
            adapter,
            source_component=f"mechanism.comparison.{callback}",
            unexpected_code=f"MECHANISM_ADAPTER_{callback.upper()}_FAILED",
        )
        for callback in (
            "expected_reduced_ideal_links",
            "physical_supports",
            "constitutive_supports",
        )
    }
    for callback, result in callback_results.items():
        invalid = (
            not isinstance(result, tuple)
            or (
                callback == "expected_reduced_ideal_links"
                and any(not isinstance(item, str) or not item for item in result)
            )
            or (
                callback != "expected_reduced_ideal_links"
                and any(not callable(getattr(item, "to_dict", None)) for item in result)
            )
        )
        if invalid:
            message = (
                f"mechanism adapter {callback}() returned an invalid tuple"
            )
            raise plugin_boundary_error(
                TypeError(message),
                registration=plugin_registration_or_none(str(plugin.plugin_id)),
                plugin_id=str(plugin.plugin_id),
                plugin_kind="mechanism",
                callback=callback,
                source_component=f"mechanism.comparison.{callback}",
                code=f"MECHANISM_ADAPTER_{callback.upper()}_INVALID",
                message=message,
                details={"capsule_id": plan.capsule_id},
                contract_violation=True,
            )
    expected_links = list(callback_results["expected_reduced_ideal_links"])
    expected_physical_supports = [
        support.to_dict() for support in callback_results["physical_supports"]
    ]
    expected_constitutive_supports = [
        support.to_dict() for support in callback_results["constitutive_supports"]
    ]
    if physical_engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_branch_contains_internal_ideal_joint")
    if reduced_engine.get("internal_ideal_joint_count") != len(expected_links):
        failures.append("reduced_branch_claimed_joint_count_is_incorrect")
    if reduced_engine.get("ideal_links") != expected_links:
        failures.append(
            f"reduced_branch_does_not_use_claimed_{plan.family}_model"
        )
    if physical_engine.get("compliant_supports", []) != expected_physical_supports:
        failures.append("physical_branch_compliant_support_mismatch")
    if reduced_engine.get("compliant_supports", []):
        failures.append("reduced_branch_contains_physical_compliant_support")
    if (
        physical_engine.get("constitutive_supports", [])
        != expected_constitutive_supports
        or reduced_engine.get("constitutive_supports", [])
        != expected_constitutive_supports
    ):
        failures.append("branch_constitutive_support_mismatch")
    physical_limit = physical.get("collision_geometry", {}).get(
        "certifiable_progress_limit_m"
    )
    if physical_limit is not None:
        reduced_limit = reduced_engine.get("ideal_travel_limit")
        limit_matches = (
            isinstance(reduced_limit, dict)
            and reduced_limit.get("coordinate") == "joint_frame_z"
            and _close(reduced_limit.get("min_m"), 0.0)
            and _close(reduced_limit.get("max_m"), physical_limit)
            and reduced_limit.get("source")
            == "physical_contact_certifiable_progress_limit"
        )
        if not limit_matches:
            failures.append(
                "reduced_prismatic_limit_does_not_match_physical_envelope"
            )
    if plan.family == "revolute":
        physical_angle_limit = physical.get("collision_geometry", {}).get(
            "certifiable_angle_limit_rad"
        )
        if physical_angle_limit is not None:
            reduced_limit = reduced_engine.get("ideal_angle_limit")
            angle_limit_matches = (
                isinstance(reduced_limit, dict)
                and reduced_limit.get("coordinate") == "joint_frame_rz"
                and _close(reduced_limit.get("min_rad"), 0.0)
                and _close(reduced_limit.get("max_rad"), physical_angle_limit)
                and reduced_limit.get("source")
                == "physical_contact_certifiable_angle_limit"
            )
            if not angle_limit_matches:
                failures.append(
                    "reduced_revolute_limit_does_not_match_physical_envelope"
                )
    physical_semantics = physical.get("branch_semantics", {})
    reduced_semantics = reduced.get("branch_semantics", {})
    for field in (
        "same_mass_kg",
        "same_inertia_xx_kg_m2",
        "same_initial_pose",
        "same_external_load_sha256",
    ):
        if not _close(physical_semantics.get(field), reduced_semantics.get(field)):
            failures.append(f"branch_{field}_mismatch")
    if physical_semantics.get("replaced_contacts_disabled") is not False:
        failures.append("physical_contact_pair_is_disabled")
    if reduced_semantics.get("replaced_contacts_disabled") is not True:
        failures.append("reduced_replaced_contact_pair_is_not_disabled")
    return failures


def _rolling_model_failures(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical: dict[str, Any],
    reduced: dict[str, Any],
) -> tuple[list[str], list[str]]:
    if plan.adapter_id is None:
        return [], []
    try:
        adapter = _adapter_for_plan(plan)
    except KeyError:
        return [], []
    geometry = adapter.geometry
    rolling = None if geometry is None else geometry.rolling_elements
    if rolling is None:
        return [], []
    semantics: list[str] = []
    physical_failures: list[str] = []
    physical_geometry = physical.get("collision_geometry", {})
    reduced_geometry = reduced.get("collision_geometry", {})
    physical_rolling = physical_geometry.get("rolling_elements")
    reduced_rolling = reduced_geometry.get("rolling_elements")
    if not isinstance(physical_rolling, dict) or (
        physical_rolling.get("runtime_enabled") is not True
        or physical_rolling.get("count") != len(rolling.elements)
    ):
        semantics.append("physical_rolling_element_model_missing")
    if not isinstance(reduced_rolling, dict) or (
        reduced_rolling.get("runtime_enabled") is not False
        or reduced_rolling.get("count") != len(rolling.elements)
    ):
        semantics.append("reduced_rolling_element_replacement_not_declared")
    if load_case.program != "bidirectional_axis":
        return semantics, physical_failures
    validation = physical.get("rolling_element_validation")
    if not isinstance(validation, dict) or validation.get("required") is not True:
        semantics.append("physical_rolling_element_validation_missing")
    elif validation.get("passed") is not True:
        physical_failures.append("physical_rolling_element_validation_failed")
    return semantics, physical_failures


def _physical_scope_evaluation(
    plan: MechanismPlan,
    metrics: dict[str, Any],
    load_case: LoadCase,
) -> tuple[list[str], float | None]:
    if plan.adapter_id is None:
        return [], None
    try:
        adapter = _adapter_for_plan(plan)
    except KeyError:
        return [], None
    geometry = adapter.geometry
    limit = None if geometry is None else geometry.certifiable_progress_limit_m
    if limit is None:
        return [], None
    numerical_tolerance_m = max(
        1e-9,
        float(metrics["max_abs_axis_speed_m_s"]) * load_case.step_s
        + geometry.collision_envelope_m
        + geometry.collision_margin_m,
    )
    failures = (
        ["physical_contact_model_progress_scope_exceeded"]
        if float(metrics["max_abs_progress_m"]) > limit + numerical_tolerance_m
        else []
    )
    return failures, numerical_tolerance_m


def _behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    thresholds: KinematicThresholds,
) -> list[str]:
    failures: list[str] = []
    if metrics["max_lateral_shift_m"] > thresholds.max_lateral_shift_m:
        failures.append(f"{branch}_excess_lateral_motion")
    if metrics["max_rotation_rad"] > thresholds.max_rotation_rad:
        failures.append(f"{branch}_excess_rotation")
    if load_case.program == "bidirectional_axis":
        if metrics["axis_purity"] < thresholds.min_axis_purity:
            failures.append(f"{branch}_wrong_motion_subspace")
        for phase in load_case.phases:
            if phase.wrench_kind == "none":
                continue
            travel = metrics["phase_directional_travel_m"].get(phase.phase_id, 0.0)
            if travel < thresholds.min_directional_travel_m:
                failures.append(f"{branch}_{phase.phase_id}_did_not_move_as_loaded")
    elif load_case.program == "unidirectional_axis":
        observed_travel = max(
            metrics["phase_directional_travel_m"].values(),
            default=0.0,
        )
        if (
            observed_travel >= thresholds.min_directional_travel_m
            and metrics["axis_purity"] < thresholds.min_axis_purity
        ):
            failures.append(f"{branch}_wrong_motion_subspace")
    elif load_case.program == "zero_input_gravity":
        if metrics["max_total_displacement_m"] > thresholds.max_zero_input_drift_m:
            failures.append(f"{branch}_zero_input_drift")
    elif load_case.program == "transverse_probe":
        if metrics["max_total_displacement_m"] > thresholds.max_transverse_motion_m:
            failures.append(f"{branch}_failed_transverse_rejection")
    elif load_case.program == "task_range_roundtrip":
        if metrics["axis_purity"] < thresholds.min_axis_purity:
            failures.append(f"{branch}_wrong_motion_subspace")
        for phase in load_case.phases:
            if phase.wrench_kind == "none":
                continue
            travel = metrics["phase_directional_travel_m"].get(phase.phase_id, 0.0)
            if branch == "reduced" and travel < thresholds.min_directional_travel_m:
                failures.append(f"{branch}_{phase.phase_id}_did_not_move_as_loaded")
            if branch != "physical" or phase.stop_condition is None:
                continue
            bounds = metrics["phase_progress_bounds_m"].get(phase.phase_id, {})
            tolerance = thresholds.min_directional_travel_m
            if (
                phase.stop_condition == "progress_at_least"
                and bounds.get("max_m", -math.inf)
                < float(phase.target_progress_m) - tolerance
            ):
                failures.append(f"physical_{phase.phase_id}_target_not_reached")
            if (
                phase.stop_condition == "progress_at_most"
                and bounds.get("min_m", math.inf)
                > float(phase.target_progress_m) + tolerance
            ):
                failures.append(f"physical_{phase.phase_id}_target_not_reached")
    else:
        failures.append(f"{branch}_unknown_load_program")
    return failures


def _validate_revolute_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
) -> list[dict[str, Any]]:
    if trace.get("status") != "COMPLETED":
        raise _fail(
            "TRACE_NOT_COMPLETED",
            f"{branch} trace status is {trace.get('status')!r}",
        )
    expected_identity = {
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches = [
        key for key, expected in expected_identity.items()
        if trace.get(key) != expected
    ]
    if mismatches:
        raise _fail(
            "TRACE_IDENTITY_MISMATCH",
            f"{branch} trace identity mismatch: {', '.join(mismatches)}",
        )
    if not _close(trace.get("axis_abs"), list(plan.axis_abs), tolerance=1e-9):
        raise _fail("TRACE_AXIS_MISMATCH", f"{branch} trace axis changed")
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise _fail("TRACE_SAMPLES_MISSING", f"{branch} trace has too few samples")
    previous_time = -math.inf
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise _fail("TRACE_SAMPLE_INVALID", f"{branch}.samples[{index}] is invalid")
        time_s = _finite(sample.get("time_s"), f"{branch}.samples[{index}].time_s")
        if time_s < previous_time - 1e-12:
            raise _fail("TRACE_TIME_REVERSED", f"{branch} sample time is not monotonic")
        previous_time = time_s
        if sample.get("phase_id") not in allowed_phases:
            raise _fail("TRACE_PHASE_UNKNOWN", f"{branch} trace contains an unknown phase")
        for field in (
            "generalized_angle_rad",
            "generalized_speed_rad_s",
            "pivot_drift_m",
            "off_axis_rotation_rad",
            "rotation_from_start_rad",
        ):
            _finite(sample.get(field), f"{branch}.samples[{index}].{field}")
        _vector3(
            sample.get("observable_displacement_abs_m"),
            f"{branch}.samples[{index}].observable_displacement_abs_m",
        )
    return samples


def _revolute_phase_directional_travel(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    axis: tuple[float, float, float],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for phase_index, phase in enumerate(load_case.phases):
        if phase.wrench_kind == "none" or phase.direction_abs is None:
            continue
        response_phase_ids = {phase.phase_id}
        for following in load_case.phases[phase_index + 1:]:
            if following.wrench_kind != "none":
                break
            response_phase_ids.add(following.phase_id)
        indices = [
            index for index, sample in enumerate(samples)
            if sample["phase_id"] in response_phase_ids
        ]
        if not indices:
            result[phase.phase_id] = 0.0
            continue
        first = indices[0]
        baseline = float(samples[max(0, first - 1)]["generalized_angle_rad"])
        sign = 1.0 if sum(
            left * right for left, right in zip(phase.direction_abs, axis)
        ) >= 0 else -1.0
        result[phase.phase_id] = max(
            0.0,
            max(
                sign * (float(samples[index]["generalized_angle_rad"]) - baseline)
                for index in indices
            ),
        )
    return result


def _revolute_metrics(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    axis: tuple[float, float, float],
) -> dict[str, Any]:
    angles = [float(sample["generalized_angle_rad"]) for sample in samples]
    speeds = [float(sample["generalized_speed_rad_s"]) for sample in samples]
    pivot = [float(sample["pivot_drift_m"]) for sample in samples]
    off_axis = [float(sample["off_axis_rotation_rad"]) for sample in samples]
    contacts = [
        int(sample["contact_count"])
        for sample in samples
        if isinstance(sample.get("contact_count"), int)
        and not isinstance(sample.get("contact_count"), bool)
    ]
    stop_penetrations = [
        float(sample["axis_spring_limit_penetration_rad"])
        for sample in samples
        if isinstance(
            sample.get("axis_spring_limit_penetration_rad"), (int, float)
        )
        and not isinstance(
            sample.get("axis_spring_limit_penetration_rad"), bool
        )
    ]
    phase_bounds: dict[str, dict[str, float]] = {}
    for phase in load_case.phases:
        values = [
            float(sample["generalized_angle_rad"])
            for sample in samples if sample["phase_id"] == phase.phase_id
        ]
        if values:
            phase_bounds[phase.phase_id] = {
                "min_rad": min(values),
                "max_rad": max(values),
            }
    return {
        "min_angle_rad": min(angles),
        "max_angle_rad": max(angles),
        "angle_range_rad": max(angles) - min(angles),
        "max_abs_angle_rad": max(abs(value) for value in angles),
        "final_angle_rad": angles[-1],
        "max_abs_speed_rad_s": max(abs(value) for value in speeds),
        "max_pivot_drift_m": max(pivot),
        "max_off_axis_rotation_rad": max(off_axis),
        "max_axis_spring_limit_penetration_rad": (
            max(stop_penetrations) if stop_penetrations else None
        ),
        "phase_directional_travel_rad": _revolute_phase_directional_travel(
            samples, load_case, axis
        ),
        "phase_angle_bounds_rad": phase_bounds,
        "contact_observed": any(value > 0 for value in contacts),
        "max_contact_count": max(contacts) if contacts else None,
    }


def _revolute_behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    thresholds: RevoluteKinematicThresholds,
) -> list[str]:
    failures: list[str] = []
    if metrics["max_pivot_drift_m"] > thresholds.max_pivot_drift_m:
        failures.append(f"{branch}_pivot_drift_exceeded")
    if metrics["max_off_axis_rotation_rad"] > thresholds.max_off_axis_rotation_rad:
        failures.append(f"{branch}_off_axis_rotation_exceeded")
    if load_case.program == "bidirectional_rotation":
        for phase in load_case.phases:
            if phase.wrench_kind == "none":
                continue
            if (
                phase.end_magnitude_si is not None
                and phase.end_magnitude_si < phase.magnitude_si
            ):
                # A decreasing envelope is controlled unloading. Motion may
                # correctly oppose the still-positive residual wrench as a
                # spring or contact releases stored energy.
                continue
            travel = metrics["phase_directional_travel_rad"].get(phase.phase_id, 0.0)
            if travel < thresholds.min_directional_angle_rad:
                failures.append(f"{branch}_{phase.phase_id}_did_not_rotate_as_loaded")
    elif load_case.program == "zero_input_gravity":
        pass
    elif load_case.program == "transverse_torque_probe":
        if metrics["max_abs_angle_rad"] > thresholds.max_transverse_angle_rad:
            failures.append(f"{branch}_failed_transverse_torque_rejection")
    elif load_case.program in {
        "task_angle_roundtrip",
        "tong_grip_object",
        "tong_hold_object",
        "tong_release_object",
        "tong_tabletop_pickup",
        "tong_tabletop_hold",
        "tong_tabletop_release",
    }:
        for phase in load_case.phases:
            if phase.wrench_kind == "none":
                continue
            if (
                phase.end_magnitude_si is not None
                and phase.end_magnitude_si < phase.magnitude_si
            ):
                continue
            travel = metrics["phase_directional_travel_rad"].get(phase.phase_id, 0.0)
            if branch == "reduced" and travel < thresholds.min_directional_angle_rad:
                failures.append(f"{branch}_{phase.phase_id}_did_not_rotate_as_loaded")
            if branch != "physical" or phase.stop_condition is None:
                continue
            bounds = metrics["phase_angle_bounds_rad"].get(phase.phase_id, {})
            target = float(phase.target_coordinate)
            tolerance = thresholds.min_directional_angle_rad
            if (
                phase.stop_condition in {"position_at_least", "progress_at_least"}
                and bounds.get("max_rad", -math.inf) < target - tolerance
            ):
                failures.append(f"physical_{phase.phase_id}_target_not_reached")
            if (
                phase.stop_condition in {"position_at_most", "progress_at_most"}
                and bounds.get("min_rad", math.inf) > target + tolerance
            ):
                failures.append(f"physical_{phase.phase_id}_target_not_reached")
    else:
        failures.append(f"{branch}_unknown_load_program")
    return failures


def _physical_revolute_scope_evaluation(
    plan: MechanismPlan,
    metrics: dict[str, Any],
    load_case: LoadCase,
) -> tuple[list[str], float | None]:
    if plan.adapter_id is None:
        return [], None
    try:
        adapter = _adapter_for_plan(plan)
    except KeyError:
        return [], None
    geometry = adapter.geometry
    limit = None if geometry is None else geometry.certifiable_angle_limit_rad
    if limit is None:
        return [], None
    numerical_tolerance_rad = max(
        1e-9,
        float(metrics["max_abs_speed_rad_s"]) * load_case.step_s
        + geometry.collision_envelope_m
        + geometry.collision_margin_m,
    )
    failures: list[str] = []
    stop_allowance_rad = 0.0
    if (
        adapter.family_spec is not None
        and adapter.family_spec.get("mechanism_kind") == "axis_spring"
    ):
        from .axis_spring_spec import axis_spring_from_dict

        spring = axis_spring_from_dict(
            adapter.family_spec,
            expected_family="revolute",
        )
        stop_allowance_rad = spring.maximum_stop_penetration_si
        observed_penetration = metrics.get(
            "max_axis_spring_limit_penetration_rad"
        )
        if not isinstance(observed_penetration, (int, float)) or isinstance(
            observed_penetration, bool
        ):
            failures.append("physical_axis_spring_limit_evidence_missing")
        elif (
            float(observed_penetration)
            > stop_allowance_rad + numerical_tolerance_rad
        ):
            failures.append("physical_axis_spring_limit_penetration_exceeded")
    outside = (
        float(metrics["max_angle_rad"])
        > limit + numerical_tolerance_rad + stop_allowance_rad
        or float(metrics["min_angle_rad"])
        < -numerical_tolerance_rad - stop_allowance_rad
    )
    if outside:
        failures.append("physical_contact_model_angle_scope_exceeded")
    return failures, numerical_tolerance_rad


def _compare_revolute_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    thresholds = RevoluteKinematicThresholds()
    try:
        physical_samples = _validate_revolute_trace(
            physical_trace, branch="physical", plan=plan, load_case=load_case
        )
        reduced_samples = _validate_revolute_trace(
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
    physical_metrics = _revolute_metrics(physical_samples, load_case, plan.axis_abs)
    reduced_metrics = _revolute_metrics(reduced_samples, load_case, plan.axis_abs)
    semantics_failures = _branch_semantics_failures(
        plan, physical_trace, reduced_trace
    )
    physical_failures = _revolute_behavior_failures(
        physical_metrics,
        branch="physical",
        load_case=load_case,
        thresholds=thresholds,
    )
    scope_failures, scope_tolerance_rad = _physical_revolute_scope_evaluation(
        plan, physical_metrics, load_case
    )
    physical_failures.extend(scope_failures)
    if not physical_metrics["contact_observed"]:
        physical_failures.append("physical_required_contact_not_observed")
    reduced_failures = _revolute_behavior_failures(
        reduced_metrics,
        branch="reduced",
        load_case=load_case,
        thresholds=thresholds,
    )
    if semantics_failures:
        verdict = "PHYSICS_INCONCLUSIVE"
    elif physical_failures:
        verdict = "PHYSICAL_FAILURE"
    elif reduced_failures:
        verdict = "CLAIM_MISMATCH"
    else:
        verdict = "PASS"
    return {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "verdict": verdict,
        "failure_reasons": [
            *semantics_failures, *physical_failures, *reduced_failures
        ],
        "thresholds": thresholds.to_dict(),
        "physical_scope_numerical_tolerance_rad": scope_tolerance_rad,
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "strict_non_claims": {
            "timing_or_velocity_match_used_as_gate": False,
            "friction_or_efficiency_certified": False,
            "dynamic_response_certified": False,
        },
    }



@dataclass(frozen=True)
class CompoundKinematicThresholds:
    min_directional_angle_rad: float
    max_joint_drift_m: float
    max_off_plane_rotation_rad: float
    max_zero_input_output_drift_rad: float
    max_transverse_output_angle_rad: float
    output_angle_tolerance_rad: float
    home_output_angle_rad: float
    outbound_output_angle_rad: float
    source: str = "validated_compound_family_spec"

    @classmethod
    def for_plan(cls, plan: MechanismPlan) -> "CompoundKinematicThresholds":
        from .compound_spec import compound_linkage_from_dict

        linkage = compound_linkage_from_dict(_adapter_for_plan(plan).family_spec)
        return cls(
            min_directional_angle_rad=math.radians(0.5),
            max_joint_drift_m=linkage.max_joint_drift_m,
            max_off_plane_rotation_rad=linkage.max_off_plane_rotation_rad,
            max_zero_input_output_drift_rad=linkage.output_angle_tolerance_rad,
            max_transverse_output_angle_rad=max(
                linkage.output_angle_tolerance_rad, math.radians(2.0)
            ),
            output_angle_tolerance_rad=linkage.output_angle_tolerance_rad,
            home_output_angle_rad=linkage.home_output_angle_rad,
            outbound_output_angle_rad=linkage.outbound_output_angle_rad,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_directional_angle_rad": self.min_directional_angle_rad,
            "max_joint_drift_m": self.max_joint_drift_m,
            "max_off_plane_rotation_rad": self.max_off_plane_rotation_rad,
            "max_zero_input_output_drift_rad": self.max_zero_input_output_drift_rad,
            "max_transverse_output_angle_rad": self.max_transverse_output_angle_rad,
            "output_angle_tolerance_rad": self.output_angle_tolerance_rad,
            "home_output_angle_rad": self.home_output_angle_rad,
            "outbound_output_angle_rad": self.outbound_output_angle_rad,
            "source": self.source,
        }


def _validated_external_fixture_body_id(
    trace: Mapping[str, Any],
    *,
    branch: str,
) -> str | None:
    if branch != "physical":
        return None
    collision = trace.get("collision_geometry")
    semantics = trace.get("branch_semantics")
    fixture = (
        collision.get("external_contact_fixture")
        if isinstance(collision, Mapping)
        else None
    )
    semantic_fixture = (
        semantics.get("external_passive_fixture")
        if isinstance(semantics, Mapping)
        else None
    )
    if fixture is None and semantic_fixture is None:
        return None
    fixture_id = str(
        fixture.get("fixture_id") if isinstance(fixture, Mapping) else ""
    )
    phone_fixture = (
        isinstance(fixture, Mapping)
        and fixture.get("shape") == "passive_phone_box"
        and fixture.get("source") == "installed_validator_phone_fixture_v1"
        and isinstance(fixture.get("mass_kg"), (int, float))
        and not isinstance(fixture.get("mass_kg"), bool)
        and abs(float(fixture["mass_kg"]) - 0.200) <= 1.0e-12
    )
    wood_rod_mass = math.pi * 0.004 * 0.004 * 0.080 * 700.0
    wood_rod_fixture = (
        isinstance(fixture, Mapping)
        and fixture.get("shape") == "passive_wood_rod_cylinder"
        and fixture.get("source")
        == "installed_validator_wood_rod_fixture_v1"
        and abs(float(fixture.get("radius_m") or 0.0) - 0.004)
        <= 1.0e-12
        and abs(float(fixture.get("length_m") or 0.0) - 0.080)
        <= 1.0e-12
        and isinstance(fixture.get("mass_kg"), (int, float))
        and not isinstance(fixture.get("mass_kg"), bool)
        and abs(float(fixture["mass_kg"]) - wood_rod_mass) <= 1.0e-12
        and len(fixture.get("required_contact_body_ids") or ()) == 2
    )
    valid = (
        isinstance(fixture, Mapping)
        and isinstance(semantic_fixture, Mapping)
        and dict(fixture) == dict(semantic_fixture)
        and fixture.get("schema_version") == "validator_passive_fixture_v1"
        and (phone_fixture or wood_rod_fixture)
        and fixture.get("runtime_enabled") is True
        and fixture.get("supported") is True
        and fixture.get("direct_actuation") is False
        and bool(fixture_id)
        and isinstance(semantics, Mapping)
        and semantics.get("direct_passive_fixture_actuation") is False
    )
    if not valid:
        raise _fail(
            "TRACE_EXTERNAL_FIXTURE_INVALID",
            "physical trace contains an invalid external passive fixture declaration",
        )
    return fixture_id


def _validate_compound_trace(
    trace: dict[str, Any],
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
) -> list[dict[str, Any]]:
    from .compound_spec import compound_linkage_from_dict

    linkage = compound_linkage_from_dict(_adapter_for_plan(plan).family_spec)
    if trace.get("status") != "COMPLETED":
        raise _fail("TRACE_NOT_COMPLETED", f"{branch} trace did not complete")
    expected_identity = {
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches = [
        key for key, expected in expected_identity.items()
        if trace.get(key) != expected
    ]
    if mismatches:
        raise _fail(
            "TRACE_IDENTITY_MISMATCH",
            f"{branch} compound trace identity mismatch: {', '.join(mismatches)}",
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise _fail("TRACE_SAMPLES_MISSING", f"{branch} compound trace has too few samples")
    expected_bodies = {item.body_id for item in linkage.bodies}
    fixture_body_id = _validated_external_fixture_body_id(
        trace, branch=branch
    )
    if fixture_body_id is not None:
        expected_bodies.add(fixture_body_id)
    expected_joints = {item.joint_id for item in linkage.revolute_joints}
    allowed_phases = {"initial", *(phase.phase_id for phase in load_case.phases)}
    previous_time = -math.inf
    for index, sample in enumerate(samples):
        label = f"{branch}.samples[{index}]"
        if not isinstance(sample, dict):
            raise _fail("TRACE_SAMPLE_INVALID", f"{label} is invalid")
        time_s = _finite(sample.get("time_s"), f"{label}.time_s")
        if time_s < previous_time - 1e-12:
            raise _fail("TRACE_TIME_REVERSED", f"{branch} sample time is not monotonic")
        previous_time = time_s
        if sample.get("phase_id") not in allowed_phases:
            raise _fail("TRACE_PHASE_UNKNOWN", f"{branch} trace contains an unknown phase")
        for field in (
            "generalized_angle_rad",
            "generalized_speed_rad_s",
            "compound_input_angle_rad",
            "compound_input_speed_rad_s",
            "compound_output_angle_rad",
            "compound_output_speed_rad_s",
            "max_joint_drift_m",
            "max_bushing_reaction_force_n",
            "off_plane_rotation_rad",
        ):
            _finite(sample.get(field), f"{label}.{field}")
        joint_drifts = sample.get("joint_drifts_m")
        if not isinstance(joint_drifts, dict) or set(joint_drifts) != expected_joints:
            raise _fail("TRACE_COMPOUND_JOINT_SET_MISMATCH", f"{label} joint set changed")
        for joint_id, value in joint_drifts.items():
            _finite(value, f"{label}.joint_drifts_m.{joint_id}")
        poses = sample.get("body_poses")
        if not isinstance(poses, dict) or set(poses) != expected_bodies:
            raise _fail("TRACE_COMPOUND_BODY_SET_MISMATCH", f"{label} body set changed")
        if (
            fixture_body_id is not None
            and sample.get("direct_passive_fixture_actuation") is not False
        ):
            raise _fail(
                "TRACE_EXTERNAL_FIXTURE_ACTUATION_INVALID",
                f"{label} does not prove passive fixture non-actuation",
            )
        for body_id, pose in poses.items():
            if not isinstance(pose, dict):
                raise _fail("TRACE_COMPOUND_BODY_POSE_INVALID", f"{label}.{body_id} pose is invalid")
            _vector3(pose.get("position_abs_m"), f"{label}.{body_id}.position_abs_m")
            quaternion = pose.get("rotation_abs_quat_wxyz")
            if not isinstance(quaternion, list) or len(quaternion) != 4:
                raise _fail("TRACE_QUATERNION_INVALID", f"{label}.{body_id} quaternion is invalid")
            for component_index, value in enumerate(quaternion):
                _finite(value, f"{label}.{body_id}.quaternion[{component_index}]")
    return samples


def _compound_phase_directional_travel(
    samples: list[dict[str, Any]], load_case: LoadCase, axis: tuple[float, float, float]
) -> dict[str, float]:
    # A compound mechanism can be driven by a force whose direction has no
    # meaningful dot product with the angular output axis (for example, a
    # vertical pedal force opening a lid about a horizontal hinge). This gate
    # only proves that each loaded phase produces an observable output
    # response. The physical-vs-ideal-reference comparison separately checks
    # the response direction and coordinate accuracy.
    del axis
    result: dict[str, float] = {}
    for phase_index, phase in enumerate(load_case.phases):
        if phase.wrench_kind == "none" or phase.direction_abs is None:
            continue
        response_phases = {phase.phase_id}
        for following in load_case.phases[phase_index + 1:]:
            if following.wrench_kind != "none":
                break
            response_phases.add(following.phase_id)
        indices = [
            index for index, sample in enumerate(samples)
            if sample["phase_id"] in response_phases
        ]
        if not indices:
            result[phase.phase_id] = 0.0
            continue
        baseline = float(samples[max(0, indices[0] - 1)]["compound_output_angle_rad"])
        result[phase.phase_id] = max(
            abs(float(samples[index]["compound_output_angle_rad"]) - baseline)
            for index in indices
        )
    return result


def _compound_metrics(
    samples: list[dict[str, Any]], load_case: LoadCase, axis: tuple[float, float, float]
) -> dict[str, Any]:
    output = [float(sample["compound_output_angle_rad"]) for sample in samples]
    input_angles = [float(sample["compound_input_angle_rad"]) for sample in samples]
    output_speed = [float(sample["compound_output_speed_rad_s"]) for sample in samples]
    phase_bounds: dict[str, dict[str, float]] = {}
    for phase in load_case.phases:
        values = [
            float(sample["compound_output_angle_rad"])
            for sample in samples if sample["phase_id"] == phase.phase_id
        ]
        if values:
            phase_bounds[phase.phase_id] = {
                "min_rad": min(values), "max_rad": max(values)
            }
    initial = output[0]
    return {
        "min_output_angle_rad": min(output),
        "max_output_angle_rad": max(output),
        "output_angle_range_rad": max(output) - min(output),
        "max_abs_output_delta_rad": max(abs(value - initial) for value in output),
        "final_output_angle_rad": output[-1],
        "min_input_angle_rad": min(input_angles),
        "max_input_angle_rad": max(input_angles),
        "max_abs_output_speed_rad_s": max(abs(value) for value in output_speed),
        "max_joint_drift_m": max(float(sample["max_joint_drift_m"]) for sample in samples),
        "max_off_plane_rotation_rad": max(float(sample["off_plane_rotation_rad"]) for sample in samples),
        "max_bushing_reaction_force_n": max(
            float(sample["max_bushing_reaction_force_n"]) for sample in samples
        ),
        "phase_directional_travel_rad": _compound_phase_directional_travel(
            samples, load_case, axis
        ),
        "phase_output_bounds_rad": phase_bounds,
    }


def _compound_semantics_failures(
    plan: MechanismPlan, physical: dict[str, Any], reduced: dict[str, Any]
) -> list[str]:
    from .compound_spec import compound_linkage_from_dict

    linkage = compound_linkage_from_dict(_adapter_for_plan(plan).family_spec)
    failures: list[str] = []
    expected_links = ["ChLinkLockRevolute" for _ in linkage.revolute_joints]
    physical_engine = physical.get("engine", {})
    reduced_engine = reduced.get("engine", {})
    if physical_engine.get("internal_ideal_joint_count") != 0:
        failures.append("physical_branch_contains_internal_ideal_joint")
    if reduced_engine.get("internal_ideal_joint_count") != len(expected_links):
        failures.append("reduced_branch_claimed_joint_count_is_incorrect")
    if reduced_engine.get("ideal_links") != expected_links:
        failures.append("reduced_branch_does_not_use_declared_linkage_topology")
    physical_bushings = physical_engine.get("compound_bushings")
    if not isinstance(physical_bushings, list) or len(physical_bushings) != len(expected_links):
        failures.append("physical_compound_bushing_set_mismatch")
    elif any(item.get("ideal_constraint") is not False for item in physical_bushings):
        failures.append("physical_compound_bushing_claims_ideal_constraint")
    if reduced_engine.get("compound_bushings") not in ([], None):
        failures.append("reduced_branch_contains_physical_compound_bushing")
    physical_semantics = physical.get("branch_semantics", {})
    reduced_semantics = reduced.get("branch_semantics", {})
    if physical_semantics.get("compound_joint_support_mode") != "finite_bushings":
        failures.append("physical_compound_support_mode_missing")
    if reduced_semantics.get("compound_joint_support_mode") != "ideal_revolute_links":
        failures.append("reduced_compound_support_mode_missing")
    expected_topology = canonical_sha256(linkage.to_dict())
    if physical_semantics.get("compound_topology_sha256") != expected_topology:
        failures.append("physical_compound_topology_binding_mismatch")
    if reduced_semantics.get("compound_topology_sha256") != expected_topology:
        failures.append("reduced_compound_topology_binding_mismatch")
    for field in (
        "same_mass_kg",
        "same_inertia_xx_kg_m2",
        "same_initial_pose",
        "same_external_load_sha256",
    ):
        if not _close(physical_semantics.get(field), reduced_semantics.get(field)):
            failures.append(f"branch_{field}_mismatch")
    physical_geometry = physical.get("collision_geometry", {})
    reduced_geometry = reduced.get("collision_geometry", {})
    for field, expected in (
        ("joint_count", len(linkage.revolute_joints)),
        ("loop_count", linkage.loop_count),
    ):
        if physical_geometry.get(field) != expected or reduced_geometry.get(field) != expected:
            failures.append(f"compound_{field}_mismatch")
    if not _close(
        physical_geometry.get("joint_angle_limits_rad"),
        reduced_geometry.get("joint_angle_limits_rad"),
    ):
        failures.append("compound_joint_angle_limit_mismatch")
    return failures


def _compound_behavior_failures(
    metrics: dict[str, Any],
    *,
    branch: str,
    load_case: LoadCase,
    thresholds: CompoundKinematicThresholds,
    reviewer_task_program: bool | None = None,
) -> list[str]:
    if reviewer_task_program is None:
        from .task_plugins import claimed_reviewer_task_programs

        reviewer_task_program = (
            load_case.program in claimed_reviewer_task_programs()
        )
    failures: list[str] = []
    if metrics["max_joint_drift_m"] > thresholds.max_joint_drift_m:
        failures.append(f"{branch}_compound_joint_drift_exceeded")
    if metrics["max_off_plane_rotation_rad"] > thresholds.max_off_plane_rotation_rad:
        failures.append(f"{branch}_compound_off_plane_rotation_exceeded")
    if branch == "physical" and metrics["max_bushing_reaction_force_n"] <= 1e-9:
        failures.append("physical_compound_bushing_reaction_missing")
    if branch == "reduced" and metrics["max_bushing_reaction_force_n"] > 1e-9:
        failures.append("reduced_compound_has_bushing_reaction")
    tolerance = thresholds.output_angle_tolerance_rad
    if metrics["min_output_angle_rad"] < thresholds.home_output_angle_rad - tolerance:
        failures.append(f"{branch}_compound_home_limit_exceeded")
    if metrics["max_output_angle_rad"] > thresholds.outbound_output_angle_rad + tolerance:
        failures.append(f"{branch}_compound_outbound_limit_exceeded")
    if load_case.program == "bidirectional_linkage":
        for phase in load_case.phases:
            if phase.wrench_kind == "none":
                continue
            travel = metrics["phase_directional_travel_rad"].get(phase.phase_id, 0.0)
            if travel < thresholds.min_directional_angle_rad:
                failures.append(f"{branch}_{phase.phase_id}_did_not_move_output_as_loaded")
    elif load_case.program == "zero_input_gravity":
        if metrics["max_abs_output_delta_rad"] > thresholds.max_zero_input_output_drift_rad:
            failures.append(f"{branch}_compound_zero_input_output_drift")
    elif load_case.program == "transverse_linkage_probe":
        if metrics["max_abs_output_delta_rad"] > thresholds.max_transverse_output_angle_rad:
            failures.append(f"{branch}_compound_failed_transverse_rejection")
    elif load_case.program == "task_output_angle_roundtrip":
        target_phase = next(
            (phase for phase in load_case.phases if phase.stop_condition is not None), None
        )
        bounds = {} if target_phase is None else metrics["phase_output_bounds_rad"].get(
            target_phase.phase_id, {}
        )
        if target_phase is None or bounds.get("max_rad", -math.inf) < (
            thresholds.outbound_output_angle_rad - tolerance
        ):
            failures.append(f"{branch}_compound_outbound_target_not_reached")
        if metrics["final_output_angle_rad"] > thresholds.home_output_angle_rad + tolerance:
            failures.append(f"{branch}_compound_did_not_return_home")
    elif not reviewer_task_program:
        failures.append(f"{branch}_unknown_compound_load_program")
    return failures


def _compare_compound_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    thresholds = CompoundKinematicThresholds.for_plan(plan)
    try:
        physical_samples = _validate_compound_trace(
            physical_trace, branch="physical", plan=plan, load_case=load_case
        )
        reduced_samples = _validate_compound_trace(
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
    from .compound_spec import compound_linkage_from_dict

    linkage = compound_linkage_from_dict(_adapter_for_plan(plan).family_spec)
    physical_metrics = _compound_metrics(
        physical_samples, load_case, linkage.input_actuation_direction_abs
    )
    reduced_metrics = _compound_metrics(
        reduced_samples, load_case, linkage.input_actuation_direction_abs
    )
    semantics_failures = _compound_semantics_failures(plan, physical_trace, reduced_trace)
    from .task_plugins import reviewer_task_program_policy

    reviewer_task_program = (
        reviewer_task_program_policy(load_case, plan=plan) is not None
    )
    physical_failures = _compound_behavior_failures(
        physical_metrics,
        branch="physical",
        load_case=load_case,
        thresholds=thresholds,
        reviewer_task_program=reviewer_task_program,
    )
    reduced_failures = _compound_behavior_failures(
        reduced_metrics,
        branch="reduced",
        load_case=load_case,
        thresholds=thresholds,
        reviewer_task_program=reviewer_task_program,
    )
    if semantics_failures:
        verdict = "PHYSICS_INCONCLUSIVE"
    elif physical_failures:
        verdict = "PHYSICAL_FAILURE"
    elif reduced_failures:
        verdict = "CLAIM_MISMATCH"
    else:
        verdict = "PASS"
    return {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "verdict": verdict,
        "failure_reasons": [*semantics_failures, *physical_failures, *reduced_failures],
        "thresholds": thresholds.to_dict(),
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "strict_non_claims": {
            "timing_or_velocity_match_used_as_gate": False,
            "physical_and_reduced_angles_required_to_match_pointwise": False,
            "friction_or_efficiency_certified": False,
            "dynamic_response_certified": False,
            "exact_non_joint_surface_contact_certified": False,
        },
    }

def compare_trace_pair(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
    *,
    thresholds: KinematicThresholds | None = None,
) -> dict[str, Any]:
    if plan.family == "compound":
        from .family_plugins import mechanism_adapter_for_plan

        plugin, _adapter = mechanism_adapter_for_plan(plan)
        if plugin.plugin_id == "rack_pinion_v1":
            from .rack_pinion_comparison_v3 import compare_rack_pinion_trace_pair

            return compare_rack_pinion_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        if plugin.plugin_id == "bicycle_chain_sprocket_drive_v2":
            from .bicycle_chain_comparison import (
                compare_bicycle_chain_trace_pair,
            )

            return compare_bicycle_chain_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        if plugin.plugin_id == "bicycle_ground_roll_v1":
            from .bicycle_ground_roll_comparison import (
                compare_bicycle_ground_roll_trace_pair,
            )

            return compare_bicycle_ground_roll_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        if plugin.plugin_id == "spur_gear_clock_v1":
            from .clock_gear_comparison import compare_clock_gear_trace_pair

            return compare_clock_gear_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        if plugin.plugin_id == "multi_output_shaft_coupling_v1":
            from .multi_output_shaft_comparison import (
                compare_multi_output_shaft_trace_pair,
            )

            return compare_multi_output_shaft_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        if plugin.plugin_id == "prismatic_detent_latch_v1":
            from .detent_slider_comparison import (
                compare_detent_slider_trace_pair,
            )

            return compare_detent_slider_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        if plugin.plugin_id == "click_pen_cam_latch_v1":
            from .click_pen_comparison import compare_click_pen_trace_pair

            return compare_click_pen_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        # [kimi-scissor-lift]
        if plugin.plugin_id == "hydraulic_scissor_lift_v1":
            from .scissor_lift_comparison import compare_scissor_lift_trace_pair

            return compare_scissor_lift_trace_pair(
                plan, load_case, physical_trace, reduced_trace
            )
        return _compare_compound_trace_pair(
            plan, load_case, physical_trace, reduced_trace
        )
    if plan.family == "revolute":
        return _compare_revolute_trace_pair(
            plan, load_case, physical_trace, reduced_trace
        )
    limits = thresholds or KinematicThresholds.for_plan(plan)
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
            "thresholds": limits.to_dict(),
        }
    physical_metrics = _trace_metrics(physical_samples, load_case, plan.axis_abs)
    reduced_metrics = _trace_metrics(reduced_samples, load_case, plan.axis_abs)
    semantics_failures = _branch_semantics_failures(
        plan, physical_trace, reduced_trace
    )
    rolling_semantics, rolling_physical = _rolling_model_failures(
        plan,
        load_case,
        physical_trace,
        reduced_trace,
    )
    semantics_failures.extend(rolling_semantics)
    physical_failures = _behavior_failures(
        physical_metrics,
        branch="physical",
        load_case=load_case,
        thresholds=limits,
    )
    scope_failures, scope_tolerance_m = _physical_scope_evaluation(
        plan, physical_metrics, load_case
    )
    physical_failures.extend(scope_failures)
    physical_failures.extend(rolling_physical)
    if not physical_metrics["contact_observed"]:
        physical_failures.append("physical_required_contact_not_observed")
    reduced_failures = _behavior_failures(
        reduced_metrics,
        branch="reduced",
        load_case=load_case,
        thresholds=limits,
    )
    if semantics_failures:
        verdict = "PHYSICS_INCONCLUSIVE"
    elif physical_failures:
        verdict = "PHYSICAL_FAILURE"
    elif reduced_failures:
        verdict = "CLAIM_MISMATCH"
    else:
        verdict = "PASS"
    return {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "verdict": verdict,
        "failure_reasons": [
            *semantics_failures,
            *physical_failures,
            *reduced_failures,
        ],
        "thresholds": limits.to_dict(),
        "physical_scope_numerical_tolerance_m": scope_tolerance_m,
        "physical_metrics": physical_metrics,
        "reduced_metrics": reduced_metrics,
        "strict_non_claims": {
            "timing_or_velocity_match_used_as_gate": False,
            "friction_or_efficiency_certified": False,
            "dynamic_response_certified": False,
        },
    }


def compare_trace_files(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: str | Path,
    reduced_trace: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    physical = read_trace_json(physical_trace)
    reduced = read_trace_json(reduced_trace)
    comparison = compare_trace_pair(
        plan,
        load_case,
        physical,
        reduced,
    )
    comparison = _apply_executed_load_schedule_gate(
        comparison,
        physical,
        reduced,
    )
    if plan.family == "compound":
        from .physical_legality import apply_saved_physical_legality_gate

        comparison = apply_saved_physical_legality_gate(
            comparison,
            physical,
            workspace_root=(
                Path.cwd() if workspace_root is None else workspace_root
            ),
        )
    return comparison


def compare_ideal_dynamics_trace_files(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: str | Path,
    ideal_dynamics_trace: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read new ideal_dynamics evidence through the legacy solver adapter."""

    return _compare_ideal_dynamics_trace_files(
        plan,
        load_case,
        physical_trace,
        ideal_dynamics_trace,
        workspace_root=workspace_root,
        defer_saved_pose_geometry_audit=False,
    )


def compare_composite_lane_ideal_dynamics_trace_files(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: str | Path,
    ideal_dynamics_trace: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Compare lane dynamics while reserving geometry for the outer gate."""

    return _compare_ideal_dynamics_trace_files(
        plan,
        load_case,
        physical_trace,
        ideal_dynamics_trace,
        workspace_root=workspace_root,
        defer_saved_pose_geometry_audit=True,
    )


def _compare_ideal_dynamics_trace_files(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: str | Path,
    ideal_dynamics_trace: str | Path,
    *,
    workspace_root: str | Path | None,
    defer_saved_pose_geometry_audit: bool,
) -> dict[str, Any]:

    physical = read_trace_json(physical_trace)
    ideal_dynamics = read_trace_json(ideal_dynamics_trace)
    if ideal_dynamics.get("branch") != "ideal_dynamics":
        raise _fail(
            "TRACE_IDENTITY_MISMATCH",
            "optional ideal dynamics trace has the wrong branch identity",
        )
    normalized = dict(ideal_dynamics)
    normalized["branch"] = "reduced"
    comparison = compare_trace_pair(
        plan,
        load_case,
        physical,
        normalized,
    )
    comparison = _apply_executed_load_schedule_gate(
        comparison,
        physical,
        ideal_dynamics,
    )
    if defer_saved_pose_geometry_audit:
        comparison["saved_pose_geometry_audit"] = {
            "status": "DEFERRED",
            "required_environment": "outer_cad_after_queue_barrier",
            "worker_audit_authority": False,
        }
    elif plan.family == "compound":
        from .physical_legality import apply_saved_physical_legality_gate

        comparison = apply_saved_physical_legality_gate(
            comparison,
            physical,
            workspace_root=(
                Path.cwd() if workspace_root is None else workspace_root
            ),
        )
    comparison["comparison_mode"] = "physical_vs_ideal_dynamics"
    comparison["ideal_dynamics_trace_branch"] = "ideal_dynamics"
    comparison["contact_evidence_authority"] = False
    return comparison

def _apply_executed_load_schedule_gate(
    comparison: dict[str, Any],
    physical_trace: dict[str, Any],
    reduced_trace: dict[str, Any],
) -> dict[str, Any]:
    result = dict(comparison)
    failures = list(result.get("failure_reasons") or [])
    physical_schedule = physical_trace.get("executed_phases")
    reduced_schedule = reduced_trace.get("executed_phases")
    physical_semantics = physical_trace.get("branch_semantics")
    reduced_semantics = reduced_trace.get("branch_semantics")
    audit_failures: list[str] = []
    if not isinstance(physical_schedule, list) or not isinstance(reduced_schedule, list):
        audit_failures.append("executed_external_load_schedule_missing")
    else:
        physical_hash = canonical_sha256(physical_schedule)
        reduced_hash = canonical_sha256(reduced_schedule)
        if not isinstance(physical_semantics, dict) or (
            physical_semantics.get("same_external_load_sha256") != physical_hash
        ):
            audit_failures.append("physical_external_load_schedule_hash_invalid")
        if not isinstance(reduced_semantics, dict) or (
            reduced_semantics.get("same_external_load_sha256") != reduced_hash
        ):
            audit_failures.append("reduced_external_load_schedule_hash_invalid")
        if physical_schedule != reduced_schedule:
            audit_failures.append("branch_executed_external_load_schedule_mismatch")
    for code in audit_failures:
        if code not in failures:
            failures.append(code)
    result["external_load_schedule_audit"] = {
        "passed": not audit_failures,
        "physical_phase_count": (
            len(physical_schedule) if isinstance(physical_schedule, list) else None
        ),
        "reduced_phase_count": (
            len(reduced_schedule) if isinstance(reduced_schedule, list) else None
        ),
        "failure_reasons": audit_failures,
        "durations_included_in_hash": True,
        "reduced_must_replay_physical_executed_schedule": True,
    }
    result["failure_reasons"] = failures
    if audit_failures and result.get("verdict") != "PHYSICAL_FAILURE":
        result["verdict"] = "PHYSICS_INCONCLUSIVE"
    return result


def aggregate_verdict(comparisons: Iterable[dict[str, Any]]) -> str:
    required = [item for item in comparisons if item.get("required_for_certificate") is True]
    if not required:
        return "PHYSICS_INCONCLUSIVE"
    verdicts = {item.get("verdict") for item in required}
    if "PHYSICS_INCONCLUSIVE" in verdicts:
        return "PHYSICS_INCONCLUSIVE"
    if "PHYSICAL_FAILURE" in verdicts:
        return "PHYSICAL_FAILURE"
    if "CLAIM_MISMATCH" in verdicts:
        return "CLAIM_MISMATCH"
    if verdicts == {"PASS"}:
        return "KINEMATIC_CERTIFIED"
    return "PHYSICS_INCONCLUSIVE"
