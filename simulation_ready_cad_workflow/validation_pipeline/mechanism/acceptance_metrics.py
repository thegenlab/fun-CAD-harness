from __future__ import annotations

"""Trusted evaluators for typed task acceptance requirements."""

import hashlib
import json
import math
from typing import Any, Mapping

from ..intake.acceptance_contract import metric_definition_for_requirement
from .adapters import adapter_from_dict
from .axis_spring_spec import (
    AxisSpringContactFixtureSpec,
    axis_spring_from_dict,
)
from .detent_slider_spec import (
    PrismaticDetentLatchSpec,
    prismatic_detent_latch_from_dict,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .multi_output_shaft_spec import (
    MultiOutputShaftSpec,
    multi_output_shaft_from_dict,
)
from .rack_pinion_spec import RackPinionSpec, rack_pinion_from_dict
from .task_motion_evidence import (
    MOTION_WINDOW_METRICS,
    RATIO_METRICS,
    TASK_WRENCH_COMPONENTS_SCHEMA,
    TaskChannelRegistry,
    ValidatedTraceProgram,
    resolve_and_validate_task_channels,
    task_channel_registry_for_plan,
    task_program_spec_for_load_case,
    validate_aligned_trace_timeline,
    validate_trace_program,
)
from .wrench_ledger import verify_saved_wrench_audit


TASK_ACCEPTANCE_SCHEMA = "task_acceptance_v1"
_WRENCH_COMPONENT_KEYS = frozenset(
    {
        "load_id",
        "target_body",
        "wrench_kind",
        "vector_si",
        "axis_abs",
    }
)


class MetricEvaluationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _task_load_case_evidence(
    traces_by_load_case: Mapping[str, Mapping[str, Any]],
    *,
    test_load_case_id: str,
    test_program: str,
) -> tuple[str, Mapping[str, Any]] | None:
    """Resolve one execution case from the contract's load/program pair."""

    matches: list[tuple[str, Mapping[str, Any]]] = []
    for evidence_id, evidence in traces_by_load_case.items():
        if not isinstance(evidence_id, str) or not isinstance(evidence, Mapping):
            continue
        raw_load_case = evidence.get("load_case")
        if not isinstance(raw_load_case, Mapping):
            continue
        try:
            load_case = LoadCase.from_dict(dict(raw_load_case))
        except (MechanismValidationError, TypeError, ValueError):
            continue
        if (
            load_case.load_case_id == evidence_id
            and load_case.canonical_test_load_case_id == test_load_case_id
            and load_case.program == test_program
        ):
            matches.append((evidence_id, evidence))
    if len(matches) > 1:
        raise MechanismValidationError(
            "TASK_LOAD_CASE_EVIDENCE_AMBIGUOUS",
            "multiple executed load cases match one contract load/program pair",
        )
    if matches:
        return matches[0]
    legacy = traces_by_load_case.get(test_load_case_id)
    if isinstance(legacy, Mapping):
        return test_load_case_id, legacy
    return None


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise MetricEvaluationError(
            "TASK_METRIC_VALUE_INVALID",
            f"{label} is not a finite numeric trace value.",
        )
    return float(value)


def _field_candidates(observable_id: str, *, angular: bool) -> list[str]:
    candidates = [observable_id]
    if observable_id.endswith("_pose"):
        stem = observable_id[: -len("_pose")]
        candidates.append(
            f"{stem}_angle_rad" if angular else f"{stem}_position_m"
        )
    if observable_id.endswith("_position"):
        candidates.append(f"{observable_id}_m")
    if observable_id.endswith("_angle"):
        candidates.append(f"{observable_id}_rad")
    return candidates


def _series(
    samples: list[Mapping[str, Any]],
    bound_object: Mapping[str, Any],
    *,
    angular: bool,
    allow_generalized: bool,
    canonical_generalized_observable_id: str | None,
    trusted_canonical_field: str | None = None,
) -> tuple[list[float], str]:
    observable_id = bound_object.get("observable_id")
    if not isinstance(observable_id, str):
        raise MetricEvaluationError(
            "TASK_OBSERVABLE_BINDING_MISSING",
            "The task metric has no bound observable ID.",
        )
    fields = (
        [trusted_canonical_field]
        if trusted_canonical_field is not None
        else _field_candidates(observable_id, angular=angular)
    )
    for field in fields:
        if all(field in sample for sample in samples):
            return (
                [_finite(sample[field], field) for sample in samples],
                field,
            )
        if all(
            isinstance(sample.get("observables"), Mapping)
            and field in sample["observables"]
            for sample in samples
        ):
            return (
                [
                    _finite(sample["observables"][field], field)
                    for sample in samples
                ],
                f"observables.{field}",
            )
    generalized = (
        "generalized_angle_rad" if angular else "generalized_progress_m"
    )
    if (
        allow_generalized
        and canonical_generalized_observable_id == observable_id
        and all(generalized in sample for sample in samples)
    ):
        return (
            [_finite(sample[generalized], generalized) for sample in samples],
            generalized,
        )
    raise MetricEvaluationError(
        "TASK_OBSERVABLE_TRACE_MISSING",
        (
            f"Trace samples do not expose the bound observable "
            f"{observable_id!r} through a trusted field."
        ),
    )


def _directed_excursion(values: list[float], direction: str) -> float:
    initial = values[0]
    if direction == "positive":
        return max(values) - initial
    if direction == "negative":
        return initial - min(values)
    return max(abs(value - initial) for value in values)


def _finite_vector3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_COMPONENT_SCHEMA_INVALID",
            f"{label} must be an exact finite three-vector.",
        )
    return tuple(
        _finite(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    )  # type: ignore[return-value]


def _validated_passive_wood_rod_fixture(
    trace: Mapping[str, Any],
    phase_samples: list[Mapping[str, Any]],
) -> bool:
    semantics = trace.get("branch_semantics")
    fixture = (
        semantics.get("external_passive_fixture")
        if isinstance(semantics, Mapping)
        else None
    )
    if not isinstance(fixture, Mapping):
        return False
    required_body_ids = fixture.get("required_contact_body_ids")
    fractions = fixture.get("contact_sample_fractions_by_body")
    minimum_fraction = fixture.get("minimum_contact_sample_fraction")
    if (
        fixture.get("shape") != "passive_wood_rod_cylinder"
        or fixture.get("source")
        != "installed_validator_wood_rod_fixture_v1"
        or fixture.get("supported") is not True
        or fixture.get("direct_actuation") is not False
        or fixture.get("runtime_enabled") is not True
        or not math.isclose(float(fixture.get("radius_m") or 0.0), 0.004)
        or not math.isclose(float(fixture.get("length_m") or 0.0), 0.080)
        or not isinstance(required_body_ids, list)
        or len(set(str(item) for item in required_body_ids)) != 2
        or not isinstance(fractions, Mapping)
        or not isinstance(minimum_fraction, (int, float))
        or isinstance(minimum_fraction, bool)
        or not phase_samples
        or int(fixture.get("evidence_sample_count") or -1)
        != len(phase_samples)
    ):
        return False
    threshold = float(minimum_fraction)
    if any(
        not isinstance(fractions.get(str(body_id)), (int, float))
        or isinstance(fractions.get(str(body_id)), bool)
        or float(fractions[str(body_id)]) < threshold
        for body_id in required_body_ids
    ):
        return False
    return all(
        sample.get("direct_passive_fixture_actuation") is False
        and sample.get("passive_fixture_bilateral_contact_active") is True
        for sample in phase_samples
    )


def _unit_axis(value: Any, label: str) -> tuple[float, float, float]:
    axis = _finite_vector3(value, label)
    norm = math.sqrt(sum(component * component for component in axis))
    if abs(norm - 1.0) > 128.0 * math.ulp(1.0):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_AXIS_INVALID",
            f"{label} must be normalized.",
        )
    return axis


def _output_torque_nm(
    samples: list[Mapping[str, Any]],
    *,
    return_sample_indices: tuple[int, ...],
    resolved_channels: Mapping[str, Any] | None,
    verified_wrench_audit: Mapping[str, Any] | None,
    trace_schema_version: Any,
) -> tuple[float, list[str], tuple[float, float, float]]:
    """Measure only audited transmission-path torque in the return window."""

    if trace_schema_version != TASK_WRENCH_COMPONENTS_SCHEMA:
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_COMPONENT_SCHEMA_INVALID",
            "Output torque evidence uses an unsupported component schema.",
        )
    if not return_sample_indices:
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_RETURN_WINDOW_MISSING",
            "Output torque requires a registered return window.",
        )
    target = (
        resolved_channels.get("target")
        if isinstance(resolved_channels, Mapping)
        else None
    )
    resolved_digest = (
        resolved_channels.get("resolved_channels_sha256")
        if isinstance(resolved_channels, Mapping)
        else None
    )
    if (
        not isinstance(target, Mapping)
        or not isinstance(target.get("semantic_id"), str)
        or not target.get("semantic_id")
        or not isinstance(target.get("body_id"), str)
        or not target.get("body_id")
        or not isinstance(target.get("canonical_field"), str)
        or not target.get("canonical_field")
        or not isinstance(resolved_digest, str)
        or len(resolved_digest) != 64
        or resolved_digest
        != _canonical_sha256(
            {
                key: value
                for key, value in resolved_channels.items()
                if key != "resolved_channels_sha256"
            }
        )
    ):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_CHANNEL_UNRESOLVED",
            "Output torque requires the validator-resolved target channel and body.",
        )
    target_body = str(target["body_id"])
    target_axis = _unit_axis(
        target.get("axis_abs"),
        "resolved target axis_abs",
    )

    if not isinstance(verified_wrench_audit, Mapping):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_AUDIT_MISSING",
            "Output torque requires a verified wrench audit.",
        )
    try:
        audit = verify_saved_wrench_audit(
            verified_wrench_audit,
            allow_multiple_task_input_channels=True,
        )
    except (MechanismValidationError, TypeError, ValueError) as exc:
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_AUDIT_INVALID",
            "Output torque requires a policy-verified wrench audit.",
        ) from exc
    audit_digest = audit.get("ledger_sha256")
    if (
        audit.get("schema_version") != "simulation_ready_cad_wrench_audit_v2"
        or audit.get("passed") is not True
        or audit.get("violations") != []
        or not isinstance(audit_digest, str)
        or audit_digest
        != _canonical_sha256(
            {key: value for key, value in audit.items() if key != "ledger_sha256"}
        )
    ):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_AUDIT_INVALID",
            "Output torque requires an intact passing wrench audit.",
        )
    raw_records = audit.get("records")
    if not isinstance(raw_records, list) or audit.get("load_count") != len(
        raw_records
    ):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_AUDIT_INVALID",
            "Output torque audit has no exact load inventory.",
        )
    records: dict[str, Mapping[str, Any]] = {}
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise MetricEvaluationError(
                "TASK_OUTPUT_TORQUE_AUDIT_INVALID",
                "Output torque audit contains a non-object load record.",
            )
        load_id = raw_record.get("load_id")
        if (
            not isinstance(load_id, str)
            or not load_id
            or load_id in records
        ):
            raise MetricEvaluationError(
                "TASK_OUTPUT_TORQUE_AUDIT_INVALID",
                "Output torque audit load IDs must be non-empty and unique.",
            )
        records[load_id] = raw_record
    selected = {
        load_id: record
        for load_id, record in records.items()
        if record.get("transmission") is True
    }
    if (
        not selected
        or audit.get("transmission_load_count") != len(selected)
    ):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_TRANSMISSION_MISSING",
            "Output torque requires an exact audited transmission-load inventory.",
        )
    if any(
        record.get("target_body") != target_body
        or record.get("wrench_kind") != "torque"
        or record.get("task_input") is not False
        or record.get("support") is not False
        for record in selected.values()
    ):
        raise MetricEvaluationError(
            "TASK_OUTPUT_TORQUE_TRANSMISSION_TARGET_MISMATCH",
            "Every audited transmission load must be torque on the resolved output body.",
        )

    peaks: list[float] = []
    for sample_index in return_sample_indices:
        components = samples[sample_index].get("wrench_components")
        if not isinstance(components, list):
            raise MetricEvaluationError(
                "TASK_OUTPUT_TORQUE_COMPONENT_EVIDENCE_MISSING",
                "Every return-window sample requires exact wrench components.",
            )
        seen: set[str] = set()
        selected_axis_torque = 0.0
        selected_seen: set[str] = set()
        for raw_component in components:
            if (
                not isinstance(raw_component, Mapping)
                or set(raw_component) != _WRENCH_COMPONENT_KEYS
            ):
                raise MetricEvaluationError(
                    "TASK_OUTPUT_TORQUE_COMPONENT_SCHEMA_INVALID",
                    "Every wrench component must use the exact registered fields.",
                )
            load_id = raw_component.get("load_id")
            if (
                not isinstance(load_id, str)
                or not load_id
                or load_id in seen
                or load_id not in records
            ):
                raise MetricEvaluationError(
                    "TASK_OUTPUT_TORQUE_COMPONENT_INVENTORY_MISMATCH",
                    "Wrench component load IDs must be unique and audit-owned.",
                )
            seen.add(load_id)
            record = records[load_id]
            if (
                raw_component.get("target_body")
                != record.get("target_body")
                or raw_component.get("wrench_kind")
                != record.get("wrench_kind")
            ):
                raise MetricEvaluationError(
                    "TASK_OUTPUT_TORQUE_COMPONENT_PROVENANCE_MISMATCH",
                    "A wrench component changes its audited body or kind.",
                )
            vector = _finite_vector3(
                raw_component.get("vector_si"),
                f"wrench_components[{load_id}].vector_si",
            )
            component_axis = _unit_axis(
                raw_component.get("axis_abs"),
                f"wrench_components[{load_id}].axis_abs",
            )
            if record.get("transmission") is not True:
                continue
            if any(
                abs(actual - expected) > 128.0 * math.ulp(1.0)
                for actual, expected in zip(component_axis, target_axis)
            ):
                raise MetricEvaluationError(
                    "TASK_OUTPUT_TORQUE_AXIS_MISMATCH",
                    "Transmission torque axis differs from the resolved output axis.",
                )
            selected_seen.add(load_id)
            selected_axis_torque += sum(
                component * axis
                for component, axis in zip(vector, target_axis)
            )
        if selected_seen != set(selected):
            raise MetricEvaluationError(
                "TASK_OUTPUT_TORQUE_COMPONENT_INVENTORY_MISMATCH",
                "Each audited transmission load must appear exactly once in every return sample.",
            )
        peaks.append(abs(selected_axis_torque))
    return max(peaks), sorted(selected), target_axis


def _compare(measured: Any, target: Mapping[str, Any]) -> bool:
    comparator = target.get("comparator")
    expected = target.get("value")
    if isinstance(measured, (int, float)) and isinstance(expected, (int, float)):
        measured = float(measured)
        expected = float(expected)
        tolerance = float(target.get("tolerance") or 0.0)
        if comparator == "at_least":
            return measured >= expected
        if comparator == "at_most":
            return measured <= expected
        if comparator in {"equals", "within"}:
            return abs(measured - expected) <= tolerance
    if comparator == "equals":
        return measured == expected
    raise MetricEvaluationError(
        "TASK_COMPARATOR_UNSUPPORTED",
        "The compiled target cannot be evaluated by this metric.",
    )


_CLOCK_HAND_FIELDS = {
    "hour": "hour_hand_angle_rad",
    "minute": "clock_output_angle_rad",
    "second": "second_hand_angle_rad",
}
_CLOCK_SUITE_METRICS = frozenset(
    {
        "clock.display_direction_consistent",
        "clock.maximum_hand_drift_rad",
        "clock.alarm_target_latch_error_rad",
        "clock.alarm_target_latch_span_rad",
        "clock.selected_alarm_position_error_rad",
        "clock.alarm_target_drift_rad",
    }
)
_EGG_BEATER_SUITE_METRICS = frozenset(
    {
        "egg_beater_outputs_opposite_directions",
        "egg_beater_output_speed_magnitude_ratio",
        "egg_beater_crank_speed_magnitude_ratio",
        "egg_beater_axis_bound_direction_consistency",
    }
)
_UTILITY_KNIFE_SUITE_METRICS = frozenset(
    {
        "utility_knife_reverse_load_magnitude",
        "utility_knife_loaded_hold_duration",
        "utility_knife_maximum_loaded_slip",
    }
)
_TONGS_SUITE_METRICS = frozenset(
    {
        "tongs.hold_duration_s",
        "tongs.release_input_zero",
    }
)
_RACK_PINION_GATE_SUITE_METRICS = frozenset(
    {
        "gate_zero_input_hold",
        "gate_zero_input_hold_duration_s",
        "gate_maximum_downward_slip_m",
    }
)


def _egg_beater_shaft_layout(
    plan: MechanismPlan | None,
) -> tuple[MultiOutputShaftSpec, dict[str, str], dict[str, float]]:
    """Frozen shaft fields and axis-bound declared ratios for two beaters."""

    if plan is None or not isinstance(plan.adapter_spec, Mapping):
        raise MetricEvaluationError(
            "EGG_BEATER_FAMILY_SPEC_MISSING",
            "Egg-beater suite metrics require the frozen family spec.",
        )
    try:
        adapter = adapter_from_dict(plan.adapter_spec)
        spec = multi_output_shaft_from_dict(adapter.family_spec)
        declared = {
            body_id: spec.declared_ratio_for_body(body_id)
            for body_id in spec.output_body_ids
        }
    except (MechanismValidationError, TypeError, ValueError, KeyError) as exc:
        raise MetricEvaluationError(
            "EGG_BEATER_FAMILY_SPEC_INVALID",
            "Egg-beater suite metrics could not parse the frozen family spec.",
        ) from exc
    if len(spec.output_body_ids) != 2:
        raise MetricEvaluationError(
            "EGG_BEATER_FAMILY_SPEC_INVALID",
            "Egg-beater suite metrics require exactly two declared outputs.",
        )
    fields = {
        "crank": f"{spec.input_body_id}_angle_rad",
        "left": f"{spec.output_body_ids[0]}_angle_rad",
        "right": f"{spec.output_body_ids[1]}_angle_rad",
    }
    return spec, fields, declared


def _utility_knife_detent_layout(
    plan: MechanismPlan | None,
) -> tuple[PrismaticDetentLatchSpec, str]:
    """Frozen slide axis and blade displacement field for the knife suite."""

    if plan is None or not isinstance(plan.adapter_spec, Mapping):
        raise MetricEvaluationError(
            "UTILITY_KNIFE_FAMILY_SPEC_MISSING",
            "Utility-knife suite metrics require the frozen family spec.",
        )
    try:
        adapter = adapter_from_dict(plan.adapter_spec)
        spec = prismatic_detent_latch_from_dict(adapter.family_spec)
    except (MechanismValidationError, TypeError, ValueError, KeyError) as exc:
        raise MetricEvaluationError(
            "UTILITY_KNIFE_FAMILY_SPEC_INVALID",
            "Utility-knife suite metrics could not parse the frozen family spec.",
        ) from exc
    return spec, f"{spec.output_body_id}_position_m"


def _tongs_tabletop_fixture(
    plan: MechanismPlan | None,
) -> AxisSpringContactFixtureSpec:
    """Frozen tabletop contact fixture owned by the tongs task suite."""

    if plan is None or not isinstance(plan.adapter_spec, Mapping):
        raise MetricEvaluationError(
            "TONGS_FAMILY_SPEC_MISSING",
            "Tongs suite metrics require the frozen family spec.",
        )
    try:
        adapter = adapter_from_dict(plan.adapter_spec)
        spec = axis_spring_from_dict(
            adapter.family_spec,
            expected_family="revolute",
        )
    except (MechanismValidationError, TypeError, ValueError, KeyError) as exc:
        raise MetricEvaluationError(
            "TONGS_FAMILY_SPEC_INVALID",
            "Tongs suite metrics could not parse the frozen family spec.",
        ) from exc
    fixture = spec.contact_fixture
    if fixture is None or fixture.shape != "tabletop_pickup_box":
        raise MetricEvaluationError(
            "TONGS_FAMILY_SPEC_INVALID",
            "Tongs suite metrics require the tabletop pickup fixture.",
        )
    return fixture


def _rack_pinion_gate_layout(plan: MechanismPlan | None) -> RackPinionSpec:
    """Frozen rack-and-pinion family spec for the gate task suite."""

    if plan is None or not isinstance(plan.adapter_spec, Mapping):
        raise MetricEvaluationError(
            "RACK_PINION_GATE_FAMILY_SPEC_MISSING",
            "Rack-and-pinion gate suite metrics require the frozen family spec.",
        )
    try:
        adapter = adapter_from_dict(plan.adapter_spec)
        spec = rack_pinion_from_dict(adapter.family_spec)
    except (MechanismValidationError, TypeError, ValueError, KeyError) as exc:
        raise MetricEvaluationError(
            "RACK_PINION_GATE_FAMILY_SPEC_INVALID",
            "Rack-and-pinion gate suite metrics could not parse the frozen family spec.",
        ) from exc
    return spec


def _suite_phase_window(
    requirement: Mapping[str, Any],
    samples: list[Mapping[str, Any]],
) -> tuple[int, tuple[int, ...]] | None:
    binding = requirement.get("task_suite_binding")
    if not isinstance(binding, Mapping):
        return None
    phase_ids = binding.get("evaluation_phase_ids")
    if not isinstance(phase_ids, list) or not phase_ids:
        raise MetricEvaluationError(
            "TASK_SUITE_WINDOW_MISSING",
            "Task-suite criterion has no frozen evaluation window.",
        )
    selected = tuple(
        index
        for index, sample in enumerate(samples)
        if sample.get("phase_id") in set(str(item) for item in phase_ids)
    )
    if not selected:
        raise MetricEvaluationError(
            "TASK_SUITE_WINDOW_EMPTY",
            "Trace has no samples in the frozen task-suite window.",
        )
    if selected[0] <= 0:
        raise MetricEvaluationError(
            "TASK_SUITE_BASELINE_MISSING",
            "Task-suite window has no saved pre-window baseline.",
        )
    return selected[0] - 1, selected


def _clock_hand_values(
    samples: list[Mapping[str, Any]],
    indices: tuple[int, ...],
) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for hand, field in _CLOCK_HAND_FIELDS.items():
        if not all(field in samples[index] for index in indices):
            raise MetricEvaluationError(
                "CLOCK_TASK_HAND_TRACE_MISSING",
                f"Clock task trace is missing {field!r} in its frozen window.",
            )
        values[hand] = [
            _finite(samples[index][field], field) for index in indices
        ]
    return values


def _uses_hold_window_reference(requirement: Mapping[str, Any]) -> bool:
    reference = " ".join(
        str(requirement.get("reference_role") or "").lower().split()
    )
    return "start" in reference and "hold" in reference


def _uses_loaded_repeatability_reference(
    requirement: Mapping[str, Any],
) -> bool:
    reference = " ".join(
        str(requirement.get("reference_role") or "").lower().split()
    )
    return "first" in reference and "settled" in reference and "loaded" in reference


def evaluate_registered_metric(
    requirement: Mapping[str, Any],
    binding: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    plan: MechanismPlan | None = None,
    canonical_generalized_observable_id: str | None = None,
    validated_program: ValidatedTraceProgram | None = None,
    resolved_channels: Mapping[str, Any] | None = None,
    verified_wrench_audit: Mapping[str, Any] | None = None,
    verified_comparison: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metric_id = requirement.get("metric_id")
    registered = metric_definition_for_requirement(requirement) is not None
    if trace.get("status") != "COMPLETED":
        raise MetricEvaluationError(
            "TASK_TRACE_NOT_COMPLETED",
            "Task metrics require a completed trace.",
        )
    samples = trace.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise MetricEvaluationError(
            "TASK_TRACE_SAMPLES_MISSING",
            "Task metrics require at least two saved samples.",
        )
    typed_samples = [
        sample for sample in samples if isinstance(sample, Mapping)
    ]
    if len(typed_samples) != len(samples):
        raise MetricEvaluationError(
            "TASK_TRACE_SAMPLE_INVALID",
            "Task metric trace contains a non-object sample.",
        )
    target_binding = binding.get("target")
    if not isinstance(target_binding, Mapping):
        raise MetricEvaluationError(
            "TASK_TARGET_BINDING_MISSING",
            "Task metric target binding is missing.",
        )
    direction = str(requirement.get("direction") or "either")
    source_field: str | None = None
    reference_field: str | None = None
    ratio_sample_index: int | None = None
    ratio_reference_floor: float | None = None
    ratio_max_pointwise_error: float | None = None
    ratio_max_incremental_error: float | None = None
    ratio_consistency_passed: bool | None = None
    output_torque_load_ids: list[str] | None = None
    output_torque_axis_abs: tuple[float, float, float] | None = None
    output_torque_sample_indices: tuple[int, ...] | None = None
    measured: Any
    target_channel = (
        resolved_channels.get("target")
        if isinstance(resolved_channels, Mapping)
        else None
    )
    reference_channel = (
        resolved_channels.get("reference")
        if isinstance(resolved_channels, Mapping)
        else None
    )
    if not registered:
        from .task_plugins import evaluate_with_reviewer_task_adapter

        try:
            plugin_result = evaluate_with_reviewer_task_adapter(
                requirement,
                binding,
                trace,
                plan=plan,
                canonical_generalized_observable_id=(
                    canonical_generalized_observable_id
                ),
                validated_program=validated_program,
                resolved_channels=resolved_channels,
                verified_wrench_audit=verified_wrench_audit,
            )
        except MechanismValidationError:
            # Reviewer TaskAdapter failures already carry producer ownership.
            # Finalization must preserve that fact instead of converting it to
            # an ordinary inconclusive metric owned by the design package.
            raise
        if plugin_result is not None:
            return plugin_result
        raise MetricEvaluationError(
            "TASK_METRIC_UNREGISTERED",
            f"Metric {metric_id!r} is not registered.",
        )

    trusted_target_field = (
        str(target_channel.get("canonical_field"))
        if isinstance(target_channel, Mapping)
        and isinstance(target_channel.get("canonical_field"), str)
        else None
    )
    trusted_reference_field = (
        str(reference_channel.get("canonical_field"))
        if isinstance(reference_channel, Mapping)
        and isinstance(reference_channel.get("canonical_field"), str)
        else None
    )

    suite_window = _suite_phase_window(requirement, typed_samples)
    if suite_window is not None:
        powered_baseline_index, powered_sample_indices = suite_window
    elif validated_program is not None:
        powered_baseline_index = validated_program.powered_baseline_index
        powered_sample_indices = validated_program.powered_sample_indices
    else:
        powered_baseline_index = 0
        powered_sample_indices = tuple(range(1, len(typed_samples)))

    def motion_indices(kind: str) -> tuple[int, ...]:
        if kind == "powered":
            return (powered_baseline_index, *powered_sample_indices)
        if validated_program is None:
            return tuple(range(len(typed_samples)))
        if kind == "return":
            return (
                validated_program.home_sample_index,
                *validated_program.return_sample_indices,
            )
        raise AssertionError(kind)

    powered_indices = motion_indices("powered")
    return_indices = motion_indices("return")

    if metric_id == "continuous_rotation_rad":
        values, source_field = _series(
            typed_samples,
            target_binding,
            angular=True,
            allow_generalized=True,
            canonical_generalized_observable_id=(
                canonical_generalized_observable_id
            ),
            trusted_canonical_field=trusted_target_field,
        )
        measured = _directed_excursion(
            [values[index] for index in powered_indices],
            direction,
        )
    elif metric_id == "return_to_start_rad":
        values, source_field = _series(
            typed_samples,
            target_binding,
            angular=True,
            allow_generalized=True,
            canonical_generalized_observable_id=(
                canonical_generalized_observable_id
            ),
            trusted_canonical_field=trusted_target_field,
        )
        hold_window_reference = _uses_hold_window_reference(requirement)
        if validated_program is not None and not validated_program.return_sample_indices:
            raise MetricEvaluationError(
                (
                    "TASK_HOLD_WINDOW_MISSING"
                    if hold_window_reference
                    else "TASK_RETURN_WINDOW_MISSING"
                ),
                (
                    "Hold drift requires a designated zero-input hold window."
                    if hold_window_reference
                    else "Return-to-start requires a designated post-powered window."
                ),
            )
        if hold_window_reference and validated_program is not None:
            hold_indices = validated_program.return_sample_indices
            hold_start = hold_indices[0]
            measured = max(
                abs(values[index] - values[hold_start])
                for index in hold_indices
            )
            source_field = f"{source_field}@zero_input_hold_window"
            terminal_delta = None
        else:
            home_index = (
                validated_program.home_sample_index
                if validated_program is not None
                else 0
            )
            final_index = (
                validated_program.final_sample_index
                if validated_program is not None
                else len(values) - 1
            )
            terminal_delta = values[final_index] - values[home_index]
        # The original return-error semantics remain unchanged for upper
        # bounds and equality windows.  A lower bound instead certifies
        # retained terminal travel and must preserve the frozen direction;
        # an equal-magnitude move in the opposite direction cannot pass.
        raw_target = requirement.get("target")
        comparator = (
            raw_target.get("comparator")
            if isinstance(raw_target, Mapping)
            else None
        )
        if hold_window_reference:
            pass
        elif comparator == "at_least":
            measured = (
                terminal_delta
                if direction == "positive"
                else -terminal_delta
                if direction == "negative"
                else abs(terminal_delta)
            )
        else:
            measured = abs(terminal_delta)
    elif metric_id == "output_torque_nm":
        if validated_program is None:
            raise MetricEvaluationError(
                "TASK_OUTPUT_TORQUE_PROGRAM_MISSING",
                "Output torque requires a validator-owned motion program.",
            )
        output_torque_sample_indices = (
            validated_program.return_sample_indices
        )
        (
            measured,
            output_torque_load_ids,
            output_torque_axis_abs,
        ) = _output_torque_nm(
            typed_samples,
            return_sample_indices=output_torque_sample_indices,
            resolved_channels=resolved_channels,
            verified_wrench_audit=verified_wrench_audit,
            trace_schema_version=trace.get(
                "wrench_components_schema_version"
            ),
        )
        source_field = (
            "wrench_components[transmission=True]"
            "@validator_owned_return_phases"
        )
    elif metric_id == "return_to_start_m":
        # Linear counterpart of return_to_start_rad: the final sample must
        # come back to the program's home sample within tolerance.
        if validated_program is not None and not validated_program.return_sample_indices:
            raise MetricEvaluationError(
                "TASK_RETURN_WINDOW_MISSING",
                "Return-to-start requires a designated post-powered window.",
            )
        family_spec = (
            plan.adapter_spec.get("family_spec")
            if plan is not None
            and isinstance(plan.adapter_spec, Mapping)
            else None
        )
        target_body_id = (
            target_channel.get("body_id")
            if isinstance(target_channel, Mapping)
            else None
        )
        position_output = bool(
            requirement.get("observable_quantity") == "position_3d"
            and isinstance(family_spec, Mapping)
            and target_body_id == family_spec.get("output_body_id")
        )
        vector_return = bool(
            position_output
            and all(
                isinstance(sample.get("observable_point_abs_m"), (list, tuple))
                and len(sample["observable_point_abs_m"]) == 3
                for sample in typed_samples
            )
        )
        if vector_return:
            home_index = (
                validated_program.home_sample_index
                if validated_program is not None
                else 0
            )
            final_index = (
                validated_program.final_sample_index
                if validated_program is not None
                else len(typed_samples) - 1
            )
            home = _finite_vector3(
                typed_samples[home_index]["observable_point_abs_m"],
                "observable_point_abs_m",
            )
            final = _finite_vector3(
                typed_samples[final_index]["observable_point_abs_m"],
                "observable_point_abs_m",
            )
            measured = math.sqrt(
                sum((right - left) ** 2 for left, right in zip(home, final))
            )
            source_field = "observable_point_abs_m@home_and_final"
        elif position_output and trace.get("branch") == "ideal_reference":
            output_joint_id = family_spec.get("output_joint_id")
            joints = family_spec.get("revolute_joints")
            joint = next(
                (
                    item
                    for item in joints
                    if isinstance(item, Mapping)
                    and item.get("joint_id") == output_joint_id
                ),
                None,
            ) if isinstance(joints, list) else None
            point_raw = family_spec.get("output_observable_point_abs_m")
            if not isinstance(joint, Mapping):
                raise MetricEvaluationError(
                    "TASK_OBSERVABLE_TRACE_MISSING",
                    "Ideal position return requires the declared output joint.",
                )
            point = _finite_vector3(
                point_raw, "output_observable_point_abs_m"
            )
            origin = _finite_vector3(
                joint.get("origin_abs_m"), "output_joint.origin_abs_m"
            )
            axis = _unit_axis(
                joint.get("axis_abs"), "output_joint.axis_abs"
            )
            radial = tuple(
                coordinate - center
                for coordinate, center in zip(point, origin)
            )
            axial = sum(value * direction for value, direction in zip(radial, axis))
            radius_m = math.sqrt(
                sum(
                    (value - axial * direction) ** 2
                    for value, direction in zip(radial, axis)
                )
            )
            angle_field = trusted_target_field or "compound_output_angle_rad"
            if not all(angle_field in sample for sample in typed_samples):
                raise MetricEvaluationError(
                    "TASK_OBSERVABLE_TRACE_MISSING",
                    "Ideal position return requires the bound output angle.",
                )
            home_index = (
                validated_program.home_sample_index
                if validated_program is not None
                else 0
            )
            final_index = (
                validated_program.final_sample_index
                if validated_program is not None
                else len(typed_samples) - 1
            )
            angle_delta = abs(
                _finite(typed_samples[final_index][angle_field], angle_field)
                - _finite(typed_samples[home_index][angle_field], angle_field)
            )
            measured = 2.0 * radius_m * math.sin(angle_delta / 2.0)
            source_field = (
                f"{angle_field}+output_observable_radius_m@home_and_final"
            )
        else:
            values, source_field = _series(
                typed_samples,
                target_binding,
                angular=False,
                allow_generalized=True,
                canonical_generalized_observable_id=(
                    canonical_generalized_observable_id
                ),
                trusted_canonical_field=trusted_target_field,
            )
            if (
                validated_program is not None
                and _uses_loaded_repeatability_reference(requirement)
                and validated_program.powered_sample_indices
                and validated_program.return_sample_indices
            ):
                measured = abs(
                    values[validated_program.return_sample_indices[-1]]
                    - values[validated_program.powered_sample_indices[-1]]
                )
                source_field += "@terminal_powered_and_terminal_return"
            else:
                measured = abs(
                    values[validated_program.final_sample_index]
                    - values[validated_program.home_sample_index]
                ) if validated_program is not None else abs(values[-1] - values[0])
    elif metric_id == "signed_displacement_m":
        values, source_field = _series(
            typed_samples,
            target_binding,
            angular=False,
            allow_generalized=True,
            canonical_generalized_observable_id=(
                canonical_generalized_observable_id
            ),
            trusted_canonical_field=trusted_target_field,
        )
        excursion_indices = list(powered_indices)
        if validated_program is not None:
            baseline_index = validated_program.powered_baseline_index
            if (
                validated_program.return_sample_indices
                and validated_program.return_sample_indices[-1]
                < validated_program.powered_sample_indices[0]
            ):
                # Some programs expose an explicit settled reference window
                # before a later loading/settling sequence.  The intervening
                # loading phase must not silently become the displacement
                # zero merely because only the final hold is scored.
                baseline_index = validated_program.return_sample_indices[-1]
            excursion_indices.insert(
                0, baseline_index
            )
        measured = _directed_excursion(
            [values[index] for index in excursion_indices], direction
        )
    elif metric_id == "cumulative_travel_m":
        values, source_field = _series(
            typed_samples,
            target_binding,
            angular=False,
            allow_generalized=True,
            canonical_generalized_observable_id=(
                canonical_generalized_observable_id
            ),
            trusted_canonical_field=trusted_target_field,
        )
        powered_values = [values[index] for index in powered_indices]
        measured = sum(
            abs(right - left)
            for left, right in zip(powered_values, powered_values[1:])
        )
    elif metric_id == "rolling_slip_ratio":
        if not all(
            "rolling_slip_ratio" in sample
            and "chassis_position_m" in sample
            for sample in typed_samples
        ):
            raise MetricEvaluationError(
                "TASK_ROLLING_SLIP_EVIDENCE_MISSING",
                "Rolling-slip acceptance requires per-sample slip and chassis position evidence.",
            )
        baseline_index = powered_indices[0]
        baseline_position = _finite(
            typed_samples[baseline_index]["chassis_position_m"],
            "chassis_position_m",
        )
        significant = [
            index
            for index in powered_indices
            if abs(
                _finite(
                    typed_samples[index]["chassis_position_m"],
                    "chassis_position_m",
                )
                - baseline_position
            )
            >= 0.10
        ]
        if not significant:
            raise MetricEvaluationError(
                "TASK_ROLLING_SLIP_WINDOW_MISSING",
                "Rolling-slip acceptance requires at least 0.10 m of powered chassis travel.",
            )
        measured = max(
            _finite(
                typed_samples[index]["rolling_slip_ratio"],
                "rolling_slip_ratio",
            )
            for index in significant
        )
        source_field = "rolling_slip_ratio@powered_after_0.10m"
    elif metric_id == "powered_contact_coverage_ratio":
        if not all(
            "expected_contact_count" in typed_samples[index]
            for index in powered_indices
        ):
            raise MetricEvaluationError(
                "TASK_POWERED_CONTACT_COVERAGE_EVIDENCE_MISSING",
                "Powered contact coverage requires expected-contact counts in every powered sample.",
            )
        measured = sum(
            int(typed_samples[index]["expected_contact_count"] or 0) > 0
            for index in powered_indices
        ) / len(powered_indices)
        source_field = "expected_contact_count@powered_coverage"
    elif metric_id == "ground_separation_m":
        if not all(
            "max_ground_separation_m" in typed_samples[index]
            for index in powered_indices
        ):
            raise MetricEvaluationError(
                "TASK_GROUND_SEPARATION_EVIDENCE_MISSING",
                "Ground separation acceptance requires a saved finite-surface separation in every powered sample.",
            )
        measured = max(
            _finite(
                typed_samples[index]["max_ground_separation_m"],
                "max_ground_separation_m",
            )
            for index in powered_indices
        )
        source_field = "max_ground_separation_m@powered"
    elif metric_id in {"angular_ratio", "displacement_ratio"}:
        reference_binding = binding.get("reference")
        if not isinstance(reference_binding, Mapping):
            raise MetricEvaluationError(
                "TASK_REFERENCE_BINDING_MISSING",
                "Ratio metrics require a distinct reference observable.",
            )
        angular = metric_id == "angular_ratio"
        values, source_field = _series(
            typed_samples,
            target_binding,
            angular=angular,
            allow_generalized=False,
            canonical_generalized_observable_id=None,
            trusted_canonical_field=trusted_target_field,
        )
        reference, reference_field = _series(
            typed_samples,
            reference_binding,
            angular=angular,
            allow_generalized=False,
            canonical_generalized_observable_id=None,
            trusted_canonical_field=trusted_reference_field,
        )
        # A round-trip task deliberately returns near its initial pose, so
        # endpoint-only division can report a zero denominator even when the
        # powered stroke was valid.  The peak reference sample still supplies
        # the reported ratio, but it is not sufficient evidence by itself:
        # every saved coordinate and every saved increment in the
        # validator-owned powered window (start through peak reference
        # excursion) must also remain consistent with the requested
        # transmission ratio.  The return phase has its own explicit
        # acceptance metric and tolerance.  This prevents a trace from
        # manufacturing one correct peak sample around otherwise unrelated
        # powered input and output motion.
        ratio_baseline_index = powered_baseline_index
        authoritative_window = (
            validated_program is not None or suite_window is not None
        )
        diagnostic_peak = (
            None
            if authoritative_window
            else max(
                range(1, len(reference)),
                key=lambda index: abs(reference[index] - reference[0]),
            )
        )
        ratio_candidate_indices = (
            powered_sample_indices
            if authoritative_window
            else tuple(range(1, int(diagnostic_peak) + 1))
        )
        ratio_sample_index = max(
            ratio_candidate_indices,
            key=lambda index: abs(
                reference[index] - reference[ratio_baseline_index]
            ),
        )
        numerator = values[ratio_sample_index] - values[ratio_baseline_index]
        denominator = (
            reference[ratio_sample_index] - reference[ratio_baseline_index]
        )
        if abs(denominator) <= 1e-12:
            raise MetricEvaluationError(
                "TASK_RATIO_REFERENCE_ZERO",
                "Ratio reference motion has no non-zero excursion.",
            )
        measured = numerator / denominator
        ratio_target = requirement.get("target")
        if not isinstance(ratio_target, Mapping):
            raise MetricEvaluationError(
                "TASK_TARGET_MISSING",
                "Compiled ratio target is missing.",
            )
        expected_ratio = _finite(
            ratio_target.get("value"),
            "ratio target value",
        )
        ratio_tolerance = _finite(
            ratio_target.get("tolerance") or 0.0,
            "ratio target tolerance",
        )
        if ratio_tolerance < 0.0:
            raise MetricEvaluationError(
                "TASK_TARGET_INVALID",
                "Ratio tolerance cannot be negative.",
            )
        peak_reference_excursion = max(
            abs(reference[index] - reference[ratio_baseline_index])
            for index in ratio_candidate_indices
        )
        ratio_reference_floor = max(
            1e-12,
            peak_reference_excursion * 1e-4,
        )

        def normalized_ratio_residual(
            output_delta: float,
            reference_delta: float,
        ) -> float:
            return abs(
                output_delta - expected_ratio * reference_delta
            ) / max(abs(reference_delta), ratio_reference_floor)

        pointwise_errors = [
            normalized_ratio_residual(
                values[index] - values[ratio_baseline_index],
                reference[index] - reference[ratio_baseline_index],
            )
            for index in ratio_candidate_indices
        ]
        incremental_indices = (
            (ratio_baseline_index, *ratio_candidate_indices)
        )
        incremental_errors = [
            normalized_ratio_residual(
                values[right] - values[left],
                reference[right] - reference[left],
            )
            for left, right in zip(
                incremental_indices,
                incremental_indices[1:],
            )
        ]
        ratio_max_pointwise_error = max(pointwise_errors)
        ratio_max_incremental_error = max(incremental_errors)
        ratio_consistency_passed = (
            ratio_max_pointwise_error <= ratio_tolerance
            and ratio_max_incremental_error <= ratio_tolerance
        )
    elif metric_id in _CLOCK_SUITE_METRICS:
        clock_indices = (powered_baseline_index, *powered_sample_indices)
        if metric_id in {
            "clock.alarm_target_latch_error_rad",
            "clock.alarm_target_latch_span_rad",
            "clock.selected_alarm_position_error_rad",
            "clock.alarm_target_drift_rad",
        }:
            required_fields = {
                "hour_hand_angle_rad",
                "alarm_setting_target_angle_rad",
            }
            if not all(
                required_fields.issubset(typed_samples[index])
                for index in clock_indices
            ):
                raise MetricEvaluationError(
                    "CLOCK_TASK_ALARM_TARGET_TRACE_MISSING",
                    "Clock alarm-target acceptance requires saved hour-hand and dedicated target-cam angles.",
                )
            hour_values = [
                _finite(
                    typed_samples[index]["hour_hand_angle_rad"],
                    "hour_hand_angle_rad",
                )
                for index in clock_indices
            ]
            target_values = [
                _finite(
                    typed_samples[index]["alarm_setting_target_angle_rad"],
                    "alarm_setting_target_angle_rad",
                )
                for index in clock_indices
            ]
            if metric_id in {
                "clock.alarm_target_latch_error_rad",
                "clock.selected_alarm_position_error_rad",
            }:
                measured = abs(hour_values[-1] - target_values[-1])
                source_field = (
                    "hour_hand_angle_rad-alarm_setting_target_angle_rad"
                    "@window_end"
                )
            else:
                measured = max(
                    abs(value - target_values[0])
                    for value in target_values[1:]
                )
                source_field = "alarm_setting_target_angle_rad@frozen_window"
        else:
            hands = _clock_hand_values(typed_samples, clock_indices)
        if metric_id == "clock.display_direction_consistent":
            deltas = {
                hand: values[-1] - values[0]
                for hand, values in hands.items()
            }
            minute_delta = deltas["minute"]
            measured = (
                abs(minute_delta) > 1.0e-12
                and all(
                    abs(delta) > 1.0e-12
                    and delta * minute_delta > 0.0
                    for delta in deltas.values()
                )
            )
            source_field = "clock_hand_angles@display_direction_window"
        elif metric_id == "clock.maximum_hand_drift_rad":
            measured = max(
                abs(value - values[0])
                for values in hands.values()
                for value in values[1:]
            )
            source_field = "clock_hand_angles@independent_alarm_window"
    elif metric_id in _EGG_BEATER_SUITE_METRICS:
        egg_spec, egg_fields, egg_declared = _egg_beater_shaft_layout(plan)
        egg_indices = (powered_baseline_index, *powered_sample_indices)
        egg_values: dict[str, list[float]] = {}
        for shaft, field in egg_fields.items():
            if not all(field in typed_samples[index] for index in egg_indices):
                raise MetricEvaluationError(
                    "EGG_BEATER_SHAFT_TRACE_MISSING",
                    f"Egg-beater task trace is missing {field!r} in its frozen window.",
                )
            egg_values[shaft] = [
                _finite(typed_samples[index][field], field)
                for index in egg_indices
            ]
        egg_deltas = {
            shaft: values[-1] - values[0]
            for shaft, values in egg_values.items()
        }
        left_body, right_body = egg_spec.output_body_ids
        if metric_id == "egg_beater_outputs_opposite_directions":
            left_axis = egg_spec.shaft_by_body[left_body].axis_abs
            right_axis = egg_spec.shaft_by_body[right_body].axis_abs
            left_delta = egg_deltas["left"]
            right_delta = egg_deltas["right"]
            measured = (
                abs(left_delta) > 1.0e-12
                and abs(right_delta) > 1.0e-12
                and all(
                    abs(
                        math.copysign(1.0, left_delta) * left_component
                        + math.copysign(1.0, right_delta) * right_component
                    )
                    <= 1.0e-9
                    for left_component, right_component in zip(
                        left_axis,
                        right_axis,
                    )
                )
            )
            source_field = (
                f"{egg_fields['left']}+{egg_fields['right']}"
                "@counter_rotation_window"
            )
        elif metric_id == "egg_beater_output_speed_magnitude_ratio":
            if abs(egg_deltas["right"]) <= 1.0e-12:
                raise MetricEvaluationError(
                    "EGG_BEATER_OUTPUT_MOTION_MISSING",
                    "Egg-beater output speed ratio requires right-output motion in its frozen window.",
                )
            measured = abs(egg_deltas["left"]) / abs(egg_deltas["right"])
            source_field = (
                f"{egg_fields['left']}/{egg_fields['right']}"
                "@counter_rotation_window"
            )
        else:
            if abs(egg_deltas["crank"]) <= 1.0e-12:
                raise MetricEvaluationError(
                    "EGG_BEATER_CRANK_MOTION_MISSING",
                    "Egg-beater crank metrics require crank motion in the frozen window.",
                )
            if metric_id == "egg_beater_crank_speed_magnitude_ratio":
                raw_target = requirement.get("target")
                expected_ratio = (
                    raw_target.get("value")
                    if isinstance(raw_target, Mapping)
                    else None
                )
                if (
                    isinstance(expected_ratio, bool)
                    or not isinstance(expected_ratio, (int, float))
                    or not math.isfinite(float(expected_ratio))
                ):
                    raise MetricEvaluationError(
                        "TASK_TARGET_MISSING",
                        "Egg-beater speedup ratio requires a finite numeric target.",
                    )
                beater_ratios = {
                    shaft: abs(egg_deltas[shaft]) / abs(egg_deltas["crank"])
                    for shaft in ("left", "right")
                }
                measured = max(
                    beater_ratios.values(),
                    key=lambda ratio: abs(ratio - float(expected_ratio)),
                )
                source_field = (
                    f"{egg_fields['left']},{egg_fields['right']}"
                    f"/{egg_fields['crank']}@crank_speedup_window"
                )
            else:
                measured = all(
                    egg_deltas[shaft]
                    * egg_declared[body]
                    * egg_deltas["crank"]
                    > 0.0
                    for shaft, body in (
                        ("left", left_body),
                        ("right", right_body),
                    )
                )
                source_field = (
                    f"{egg_fields['crank']}->{egg_fields['left']},"
                    f"{egg_fields['right']}@axis_bound_direction_window"
                )
    elif metric_id in _UTILITY_KNIFE_SUITE_METRICS:
        knife_spec, knife_field = _utility_knife_detent_layout(plan)
        knife_indices = (powered_baseline_index, *powered_sample_indices)
        if metric_id == "utility_knife_reverse_load_magnitude":
            slide_axis = knife_spec.blade_prismatic_joint.axis_abs
            if not all(
                "external_load_abs_n" in typed_samples[index]
                for index in powered_sample_indices
            ):
                raise MetricEvaluationError(
                    "UTILITY_KNIFE_LOAD_TRACE_MISSING",
                    "Utility-knife reverse-load acceptance requires a saved "
                    "external load in every frozen-window sample.",
                )
            measured = max(
                -sum(
                    component * axis
                    for component, axis in zip(
                        _finite_vector3(
                            typed_samples[index]["external_load_abs_n"],
                            "external_load_abs_n",
                        ),
                        slide_axis,
                    )
                )
                for index in powered_sample_indices
            )
            source_field = "external_load_abs_n@reverse_load_window"
        elif metric_id == "utility_knife_loaded_hold_duration":
            # The baseline sample sits at the phase boundary in time, so the
            # span from it to the final window sample reports the full
            # certified hold interval instead of losing the leading sampling
            # interval.  Resolve at the validator's 1e-9 s phase-boundary
            # tolerance: integration time bases accumulate machine-scale dust
            # below it.
            hold_start_s = _finite(
                typed_samples[powered_baseline_index].get("time_s"),
                "time_s",
            )
            hold_end_s = _finite(
                typed_samples[powered_sample_indices[-1]].get("time_s"),
                "time_s",
            )
            measured = round(hold_end_s - hold_start_s, 9)
            source_field = "time_s@loaded_hold_window"
        else:
            if not all(
                knife_field in typed_samples[index] for index in knife_indices
            ):
                raise MetricEvaluationError(
                    "UTILITY_KNIFE_BLADE_TRACE_MISSING",
                    "Utility-knife loaded-slip acceptance requires a saved "
                    "blade position in every frozen-window sample.",
                )
            blade_positions = [
                _finite(typed_samples[index][knife_field], knife_field)
                for index in knife_indices
            ]
            measured = max(
                abs(value - blade_positions[0]) for value in blade_positions[1:]
            )
            source_field = f"{knife_field}@loaded_hold_window"
    elif metric_id in _TONGS_SUITE_METRICS:
        _tongs_tabletop_fixture(plan)
        if metric_id == "tongs.hold_duration_s":
            # The baseline sample sits at the phase boundary in time, so the
            # span from it to the final window sample reports the full
            # certified hold interval instead of losing the leading sampling
            # interval.  Resolve at the validator's 1e-9 s phase-boundary
            # tolerance: integration time bases accumulate machine-scale dust
            # below it.
            hold_start_s = _finite(
                typed_samples[powered_baseline_index].get("time_s"),
                "time_s",
            )
            hold_end_s = _finite(
                typed_samples[powered_sample_indices[-1]].get("time_s"),
                "time_s",
            )
            measured = round(hold_end_s - hold_start_s, 9)
            source_field = "time_s@tabletop_hold_window"
        else:
            if not all(
                "applied_torque_abs_nm" in typed_samples[index]
                for index in powered_sample_indices
            ):
                raise MetricEvaluationError(
                    "TONGS_RELEASE_INPUT_TRACE_MISSING",
                    "Tongs passive-release acceptance requires a saved "
                    "hand-input wrench in every frozen-window sample.",
                )
            measured = all(
                all(
                    component == 0.0
                    for component in _finite_vector3(
                        typed_samples[index]["applied_torque_abs_nm"],
                        "applied_torque_abs_nm",
                    )
                )
                for index in powered_sample_indices
            )
            source_field = "applied_torque_abs_nm@passive_release_window"
    elif metric_id in _RACK_PINION_GATE_SUITE_METRICS:
        _rack_pinion_gate_layout(plan)
        if metric_id == "gate_zero_input_hold":
            if not all(
                "applied_torque_abs_nm" in typed_samples[index]
                for index in powered_sample_indices
            ):
                raise MetricEvaluationError(
                    "RACK_PINION_GATE_INPUT_TRACE_MISSING",
                    "Rack-and-pinion gate anti-drop acceptance requires a saved "
                    "crank-input wrench in every frozen-window sample.",
                )
            measured = all(
                all(
                    component == 0.0
                    for component in _finite_vector3(
                        typed_samples[index]["applied_torque_abs_nm"],
                        "applied_torque_abs_nm",
                    )
                )
                for index in powered_sample_indices
            )
            source_field = "applied_torque_abs_nm@zero_input_hold_window"
        elif metric_id == "gate_zero_input_hold_duration_s":
            # The baseline sample sits at the phase boundary in time, so the
            # span from it to the final window sample reports the full
            # certified hold interval instead of losing the leading sampling
            # interval.  Resolve at the validator's 1e-9 s phase-boundary
            # tolerance: integration time bases accumulate machine-scale dust
            # below it.
            hold_start_s = _finite(
                typed_samples[powered_baseline_index].get("time_s"),
                "time_s",
            )
            hold_end_s = _finite(
                typed_samples[powered_sample_indices[-1]].get("time_s"),
                "time_s",
            )
            measured = round(hold_end_s - hold_start_s, 9)
            source_field = "time_s@zero_input_hold_window"
        else:
            # The plugin comparator already owns the anti-drop slip semantics
            # (worst peak-to-trough downward excursion in rack progress over
            # the frozen window, anchored at the phase boundary, with the
            # direction convention derived from the frozen family spec) and
            # gates it on the family-spec threshold.  Reuse its passing
            # physical-branch result instead of recomputing a second truth.
            suite_binding = requirement.get("task_suite_binding")
            window_phase_ids = (
                suite_binding.get("evaluation_phase_ids")
                if isinstance(suite_binding, Mapping)
                else None
            )
            if (
                not isinstance(window_phase_ids, list)
                or len(window_phase_ids) != 1
            ):
                raise MetricEvaluationError(
                    "RACK_PINION_GATE_SUITE_WINDOW_INVALID",
                    "Rack-and-pinion gate anti-drop slip requires exactly one "
                    "frozen evaluation phase.",
                )
            window_phase_id = str(window_phase_ids[0])
            physical_metrics = (
                verified_comparison.get("physical_metrics")
                if isinstance(verified_comparison, Mapping)
                else None
            )
            phase_downward_slip = (
                physical_metrics.get("phase_downward_slip_m")
                if isinstance(physical_metrics, Mapping)
                else None
            )
            if (
                not isinstance(phase_downward_slip, Mapping)
                or window_phase_id not in phase_downward_slip
            ):
                raise MetricEvaluationError(
                    "RACK_PINION_GATE_COMPARISON_EVIDENCE_MISSING",
                    "Rack-and-pinion gate anti-drop slip requires the physical "
                    "comparison's per-phase downward slip for its frozen window.",
                )
            measured = _finite(
                phase_downward_slip[window_phase_id],
                f"phase_downward_slip_m.{window_phase_id}",
            )
            source_field = (
                "comparison.physical_metrics."
                f"phase_downward_slip_m.{window_phase_id}"
            )
    elif metric_id == "expected_contact":
        fixture_counts = [
            sample.get("expected_contact_count")
            for sample in typed_samples
            if "expected_contact_count" in sample
        ]
        if fixture_counts:
            measured = any(int(value or 0) > 0 for value in fixture_counts)
            source_field = "expected_contact_count"
        else:
            measured = any(
                int(sample.get("contact_count") or 0) > 0
                for sample in typed_samples
            )
            source_field = "contact_count"
    elif metric_id == "tong_two_sided_grip":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") in {
                "tong_pickup_verify",
                "tong_grip_verify",
            }
        ]
        if not phase_samples or not all(
            "test_object_two_sided_contact" in sample
            for sample in phase_samples
        ):
            raise MetricEvaluationError(
                "TONG_GRIP_EVIDENCE_MISSING",
                "Kitchen-tong grip acceptance requires saved two-sided contact evidence.",
            )
        measured = phase_samples[-1]["test_object_two_sided_contact"] is True
        source_field = "test_object_two_sided_contact@final_grip_verify"
    elif metric_id == "tong_object_starts_on_table":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_pickup_initial_table_settle"
        ]
        required_fields = {
            "test_object_table_contact_count",
            "test_object_height_above_table_m",
        }
        if not phase_samples or not required_fields.issubset(phase_samples[-1]):
            raise MetricEvaluationError(
                "TONG_TABLE_START_EVIDENCE_MISSING",
                "Kitchen-tong pickup requires a saved settled tabletop state.",
            )
        measured = (
            int(phase_samples[-1]["test_object_table_contact_count"]) > 0
            and abs(
                _finite(
                    phase_samples[-1]["test_object_height_above_table_m"],
                    "test_object_height_above_table_m",
                )
            )
            <= 0.001
        )
        source_field = "test_object_table_contact_count+height@initial_table_settle"
    elif metric_id == "tong_pickup_lift_m":
        settled = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_pickup_initial_table_settle"
        ]
        lifted = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_pickup_verify"
        ]
        if (
            not settled
            or not lifted
            or "test_object_position_abs_m" not in settled[-1]
            or "test_object_position_abs_m" not in lifted[-1]
        ):
            raise MetricEvaluationError(
                "TONG_PICKUP_EVIDENCE_MISSING",
                "Kitchen-tong pickup requires settled and lifted object positions.",
            )
        start = settled[-1]["test_object_position_abs_m"]
        end = lifted[-1]["test_object_position_abs_m"]
        if not (
            isinstance(start, (list, tuple))
            and isinstance(end, (list, tuple))
            and len(start) == 3
            and len(end) == 3
        ):
            raise MetricEvaluationError(
                "TONG_PICKUP_EVIDENCE_MISSING",
                "Kitchen-tong pickup object positions must be 3D vectors.",
            )
        measured = _finite(end[2], "test_object_position_abs_m[2]") - _finite(
            start[2], "test_object_position_abs_m[2]"
        )
        source_field = "test_object_position_abs_m@settle->pickup_verify"
    elif metric_id == "tong_pickup_table_clear":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_pickup_verify"
        ]
        required_fields = {
            "test_object_table_contact_count",
            "test_object_height_above_table_m",
        }
        if not phase_samples or not required_fields.issubset(phase_samples[-1]):
            raise MetricEvaluationError(
                "TONG_PICKUP_EVIDENCE_MISSING",
                "Kitchen-tong pickup requires final tabletop-clearance evidence.",
            )
        measured = (
            int(phase_samples[-1]["test_object_table_contact_count"]) == 0
            and _finite(
                phase_samples[-1]["test_object_height_above_table_m"],
                "test_object_height_above_table_m",
            )
            > 0.0
        )
        source_field = "test_object_table_contact_count+height@tong_pickup_verify"
    elif metric_id == "tong_hold_slip_m":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_hold_verify"
        ]
        if len(phase_samples) < 2 or not all(
            "test_object_position_m" in sample for sample in phase_samples
        ):
            raise MetricEvaluationError(
                "TONG_HOLD_EVIDENCE_MISSING",
                "Kitchen-tong hold acceptance requires saved test-object positions.",
            )
        positions = [
            _finite(sample["test_object_position_m"], "test_object_position_m")
            for sample in phase_samples
        ]
        measured = max(abs(value - positions[0]) for value in positions)
        source_field = "test_object_position_m@tong_hold_verify"
    elif metric_id == "tong_hold_relative_slip_m":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_hold_verify"
        ]
        legacy_field = "test_object_relative_to_fixed_abs_m"
        fixture_field = "passive_fixture_target_relative_position_m"
        if len(phase_samples) < 2:
            raise MetricEvaluationError(
                "TONG_HOLD_EVIDENCE_MISSING",
                "Kitchen-tong hold requires object positions relative to the tool.",
            )
        if all(
            isinstance(sample.get(legacy_field), (list, tuple))
            and len(sample[legacy_field]) == 3
            for sample in phase_samples
        ):
            relative_field = legacy_field
        elif _validated_passive_wood_rod_fixture(trace, phase_samples) and all(
            isinstance(sample.get(fixture_field), (list, tuple))
            and len(sample[fixture_field]) == 3
            for sample in phase_samples
        ):
            relative_field = fixture_field
        else:
            raise MetricEvaluationError(
                "TONG_HOLD_EVIDENCE_MISSING",
                "Tong hold requires validated object positions relative to the tool.",
            )
        vectors = [
            tuple(
                _finite(value, relative_field)
                for value in sample[relative_field]
            )
            for sample in phase_samples
        ]
        measured = max(
            math.sqrt(
                sum(
                    (value - initial) ** 2
                    for value, initial in zip(vector, vectors[0])
                )
            )
            for vector in vectors
        )
        source_field = f"{relative_field}@tong_hold_verify"
    elif metric_id == "tong_release_drop_m":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") in {
                "tong_release_gradual_open",
                "tong_release_passive_return",
                "tong_release_observe",
            }
        ]
        if len(phase_samples) < 2 or not all(
            "test_object_position_m" in sample for sample in phase_samples
        ):
            raise MetricEvaluationError(
                "TONG_RELEASE_EVIDENCE_MISSING",
                "Kitchen-tong release acceptance requires saved test-object positions.",
            )
        positions = [
            _finite(sample["test_object_position_m"], "test_object_position_m")
            for sample in phase_samples
        ]
        measured = max(0.0, positions[0] - min(positions[1:]))
        source_field = "test_object_position_m@tong_release_*"
    elif metric_id == "tong_release_contact_clear":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_release_observe"
        ]
        required_fields = {
            "test_object_fixed_jaw_contact_count",
            "test_object_moving_jaw_contact_count",
        }
        if not phase_samples or not required_fields.issubset(phase_samples[-1]):
            raise MetricEvaluationError(
                "TONG_RELEASE_EVIDENCE_MISSING",
                "Kitchen-tong release acceptance requires final jaw-contact evidence.",
            )
        measured = (
            int(phase_samples[-1]["test_object_fixed_jaw_contact_count"]) == 0
            and int(phase_samples[-1]["test_object_moving_jaw_contact_count"]) == 0
        )
        source_field = "test_object_*_jaw_contact_count@tong_release_observe"
    elif metric_id == "tong_release_returns_to_table":
        phase_samples = [
            sample for sample in typed_samples
            if sample.get("phase_id") == "tong_release_observe"
        ]
        required_fields = {
            "test_object_table_contact_count",
            "test_object_height_above_table_m",
        }
        if not phase_samples or not required_fields.issubset(phase_samples[-1]):
            raise MetricEvaluationError(
                "TONG_RELEASE_EVIDENCE_MISSING",
                "Kitchen-tong release requires final tabletop contact evidence.",
            )
        measured = (
            int(phase_samples[-1]["test_object_table_contact_count"]) > 0
            and abs(
                _finite(
                    phase_samples[-1]["test_object_height_above_table_m"],
                    "test_object_height_above_table_m",
                )
            )
            <= 0.001
        )
        source_field = "test_object_table_contact_count+height@tong_release_observe"
    elif metric_id == "forbidden_penetration_m":
        if not all("max_penetration_m" in sample for sample in typed_samples):
            raise MetricEvaluationError(
                "TASK_PENETRATION_EVIDENCE_MISSING",
                "Penetration metrics require per-saved-pose evidence.",
            )
        measured = max(
            _finite(sample["max_penetration_m"], "max_penetration_m")
            for sample in typed_samples
        )
        source_field = "max_penetration_m"
    elif metric_id == "actuation_provenance":
        if not isinstance(verified_wrench_audit, Mapping):
            raise MetricEvaluationError(
                "TASK_ACTUATION_EVIDENCE_MISSING",
                "Actuation provenance requires a verified wrench audit.",
            )
        measured = (
            "indirect_only"
            if (
                verified_wrench_audit.get("passed") is True
                and verified_wrench_audit.get(
                    "direct_output_drive_detected"
                ) is False
                and verified_wrench_audit.get(
                    "task_input_bodies_observed"
                )
            )
            else "direct_or_unproven"
        )
        source_field = "wrench_audit.task_input_bodies_observed"
    else:
        raise MetricEvaluationError(
            "TASK_METRIC_NOT_IMPLEMENTED",
            f"Metric {metric_id!r} has no trusted evaluator.",
        )

    target = requirement.get("target")
    if not isinstance(target, Mapping):
        raise MetricEvaluationError(
            "TASK_TARGET_MISSING",
            "Compiled task target is missing.",
        )
    passed = _compare(measured, target)
    if ratio_consistency_passed is not None:
        passed = passed and ratio_consistency_passed
    return {
        "schema_version": TASK_ACCEPTANCE_SCHEMA,
        "requirement_id": requirement.get("requirement_id"),
        "task_id": requirement.get("task_id"),
        "branch": trace.get("branch"),
        "metric_id": metric_id,
        "measured": measured,
        "target": dict(target),
        "passed": passed,
        "verdict": "PASS" if passed else "FAIL",
        "evidence": {
            "source_field": source_field,
            "reference_field": reference_field,
            "sample_count": len(typed_samples),
            "ratio_sample_index": ratio_sample_index,
            "ratio_window": (
                (
                    "validator_owned_powered_phases_all_samples_and_increments"
                    if validated_program is not None
                    else "diagnostic_initial_to_max_reference_excursion"
                )
                if ratio_sample_index is not None
                else None
            ),
            "motion_window_certificate_authority": (
                validated_program is not None
                if metric_id in MOTION_WINDOW_METRICS
                else None
            ),
            "ratio_reference_floor": ratio_reference_floor,
            "ratio_max_pointwise_error": ratio_max_pointwise_error,
            "ratio_max_incremental_error": ratio_max_incremental_error,
            "ratio_consistency_passed": ratio_consistency_passed,
            "output_torque_load_ids": output_torque_load_ids,
            "output_torque_axis_abs": (
                list(output_torque_axis_abs)
                if output_torque_axis_abs is not None
                else None
            ),
            "output_torque_sample_indices": (
                list(output_torque_sample_indices)
                if output_torque_sample_indices is not None
                else None
            ),
            "continuous_coordinate_required": (
                metric_id == "continuous_rotation_rad"
            ),
            "endpoint_pose_equivalence_is_not_sufficient": (
                metric_id == "continuous_rotation_rad"
            ),
            "task_suite_binding": requirement.get("task_suite_binding"),
            "validated_program": (
                validated_program.to_dict()
                if validated_program is not None
                else None
            ),
            "resolved_channels_sha256": (
                resolved_channels.get("resolved_channels_sha256")
                if isinstance(resolved_channels, Mapping)
                else None
            ),
        },
    }


def evaluate_task_acceptance(
    plan: MechanismPlan,
    traces_by_load_case: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate each typed requirement on one exact trusted load program."""

    contract = plan.acceptance_contract
    semantic = plan.semantic_binding
    if not isinstance(contract, Mapping) or not isinstance(semantic, Mapping):
        payload = {
            "schema_version": TASK_ACCEPTANCE_SCHEMA,
            "status": "LEGACY_UNTYPED",
            "certificate_authority": False,
            "evaluations": [],
            "failure_reasons": ["typed_task_evidence_missing"],
        }
        payload["task_acceptance_sha256"] = _canonical_sha256(payload)
        return payload
    if (
        contract.get("status") != "PASS"
        or contract.get("certificate_authority") is not True
        or semantic.get("status") != "PASS"
        or semantic.get("certificate_authority") is not True
    ):
        payload = {
            "schema_version": TASK_ACCEPTANCE_SCHEMA,
            "status": "INCONCLUSIVE",
            "certificate_authority": False,
            "evaluations": [],
            "failure_reasons": ["typed_task_evidence_not_authoritative"],
        }
        payload["task_acceptance_sha256"] = _canonical_sha256(payload)
        return payload

    bindings = {
        item.get("requirement_id"): item
        for item in semantic.get("bindings", [])
        if isinstance(item, Mapping)
    }
    evaluations: list[dict[str, Any]] = []
    failures: list[str] = []
    measured_failures = False
    channel_registry: TaskChannelRegistry | None = None
    validated_programs: dict[str, ValidatedTraceProgram] = {}
    verified_wrench_audits: dict[str, Mapping[str, Any]] = {}
    for requirement in contract.get("requirements", []):
        if not isinstance(requirement, Mapping):
            continue
        requirement_id = str(requirement.get("requirement_id"))
        binding = bindings.get(requirement_id)
        if not isinstance(binding, Mapping):
            failures.append(f"{requirement_id}:binding_missing")
            continue
        test_program = requirement.get("test_program")
        test_load_case_id = requirement.get("test_load_case_id")
        if not isinstance(test_program, str):
            failures.append(f"{requirement_id}:test_program_missing")
            continue
        if not isinstance(test_load_case_id, str):
            failures.append(f"{requirement_id}:test_load_case_id_missing")
            continue
        try:
            resolved_evidence = _task_load_case_evidence(
                traces_by_load_case,
                test_load_case_id=test_load_case_id,
                test_program=test_program,
            )
        except MechanismValidationError as exc:
            if exc.failure_envelope is not None:
                raise
            failures.append(
                f"{requirement_id}:{test_load_case_id}:{exc.code}"
            )
            continue
        if resolved_evidence is None:
            failures.append(
                f"{requirement_id}:{test_load_case_id}:load_case_missing"
            )
            continue
        executed_load_case_id, load_case_evidence = resolved_evidence
        if load_case_evidence.get("program") != test_program:
            failures.append(
                f"{requirement_id}:{test_load_case_id}:program_mismatch"
            )
            continue
        if channel_registry is None:
            try:
                channel_registry = task_channel_registry_for_plan(plan)
            except (MechanismValidationError, TypeError, ValueError) as exc:
                if (
                    isinstance(exc, MechanismValidationError)
                    and exc.failure_envelope is not None
                ):
                    raise
                failures.append(
                    f"{requirement_id}:{test_load_case_id}:"
                    f"{getattr(exc, 'code', 'TASK_CHANNEL_REGISTRY_INVALID')}"
                )
                continue
        raw_load_case = load_case_evidence.get("load_case")
        try:
            if not isinstance(raw_load_case, Mapping):
                raise MechanismValidationError(
                    "TASK_LOAD_CASE_EVIDENCE_MISSING",
                    "task acceptance requires the immutable validator load case",
                )
            load_case = LoadCase.from_dict(dict(raw_load_case))
            if (
                load_case.load_case_id != executed_load_case_id
                or load_case.canonical_test_load_case_id != test_load_case_id
                or load_case.program != test_program
            ):
                raise MechanismValidationError(
                    "TASK_LOAD_CASE_EVIDENCE_MISMATCH",
                    "task load-case evidence identity does not match the contract",
                )
            physical = load_case_evidence.get("physical")
            if not isinstance(physical, Mapping):
                raise MechanismValidationError(
                    "PHYSICAL_TRACE_EVIDENCE_MISSING",
                    "task acceptance requires the physical trace",
                )
            physical_gate = load_case_evidence.get("physical_trace_audit")
            comparison_gate = load_case_evidence.get("comparison")
            if (
                not isinstance(physical_gate, Mapping)
                or physical_gate.get("passed") is not True
                or not isinstance(comparison_gate, Mapping)
                or comparison_gate.get("verdict") != "PASS"
            ):
                raise MechanismValidationError(
                    "TASK_PHYSICAL_GATE_NOT_PASSING",
                    "physical audit and physical comparison must pass before task metrics",
                )
            validated = validated_programs.get(executed_load_case_id)
            if validated is None:
                program_spec = task_program_spec_for_load_case(
                    load_case,
                    plugin_id=channel_registry.plugin_id,
                    plan=plan,
                )
                validated = validate_trace_program(
                    physical,
                    load_case,
                    program_spec=program_spec,
                )
                validated_programs[executed_load_case_id] = validated
            resolved_channels = resolve_and_validate_task_channels(
                requirement,
                binding,
                channel_registry,
            )
            selected_input_body = str(
                resolved_channels["input"]["body_id"]
            )
            task_suite_binding = requirement.get("task_suite_binding")
            allowed_port_ids = (
                task_suite_binding.get("allowed_input_port_ids")
                if isinstance(task_suite_binding, Mapping)
                else None
            )
            if isinstance(allowed_port_ids, list) and allowed_port_ids:
                if resolved_channels["input"].get("semantic_id") not in allowed_port_ids:
                    raise MechanismValidationError(
                        "TASK_SUITE_INPUT_MISMATCH",
                        "selected task input is outside the frozen task-suite input set",
                    )
                try:
                    selected_input_bodies = tuple(
                        channel_registry.inputs[str(port_id)].body_id
                        for port_id in allowed_port_ids
                    )
                except KeyError as exc:
                    raise MechanismValidationError(
                        "TASK_SUITE_INPUT_UNREGISTERED",
                        "task-suite input is absent from the runtime channel registry",
                    ) from exc
            else:
                selected_input_bodies = tuple(
                    str(item)
                    for item in (
                        resolved_channels["input"].get(
                            "actuation_body_ids"
                        )
                        or (selected_input_body,)
                    )
                )
            selected_output_body = str(
                resolved_channels["target"]["body_id"]
            )
            audit_key = (
                f"{executed_load_case_id}:{','.join(selected_input_bodies)}:"
                f"{selected_output_body}"
            )
            verified_audit = verified_wrench_audits.get(audit_key)
            if verified_audit is None:
                verified_audit = verify_saved_wrench_audit(
                    physical.get("wrench_audit"),
                    expected_plugin_id=channel_registry.plugin_id,
                    expected_branch="physical",
                    expected_load_case_id=executed_load_case_id,
                    expected_phase_ids=validated.executed_phase_ids,
                    expected_program_phases=(
                        phase.to_dict() for phase in load_case.phases
                    ),
                    expected_executed_load_sha256=(
                        validated.executed_load_sha256
                    ),
                    expected_input_body_ids=channel_registry.input_body_ids,
                    expected_output_body_ids=channel_registry.output_body_ids,
                    expected_task_input_body_ids=selected_input_bodies,
                    expected_task_output_body_id=selected_output_body,
                    powered_phase_ids=(
                        validated.program_spec.powered_phase_ids
                    ),
                    return_phase_ids=validated.program_spec.return_phase_ids,
                    return_input_policy=(
                        validated.program_spec.return_input_policy
                    ),
                    allow_multiple_task_input_channels=(
                        len(channel_registry.input_body_ids) > 1
                        or len(selected_input_bodies) > 1
                    ),
                    allow_direct_output_actuation=requirement.get(
                        "allow_direct_output_actuation"
                    ),
                )
                verified_wrench_audits[audit_key] = verified_audit
        except (MechanismValidationError, TypeError, ValueError) as exc:
            if (
                isinstance(exc, MechanismValidationError)
                and exc.failure_envelope is not None
            ):
                raise
            failures.append(
                f"{requirement_id}:{test_load_case_id}:"
                f"{getattr(exc, 'code', 'TASK_EVIDENCE_INVALID')}"
            )
            continue
        for branch in requirement.get("required_branches", []):
            trace = load_case_evidence.get(str(branch))
            if not isinstance(trace, Mapping):
                failures.append(
                    f"{requirement_id}:{test_load_case_id}:{branch}:trace_missing"
                )
                continue
            if str(branch) != "physical":
                try:
                    validate_aligned_trace_timeline(trace, physical)
                except MechanismValidationError as exc:
                    failures.append(
                        f"{requirement_id}:{test_load_case_id}:{branch}:{exc.code}"
                    )
                    continue
            try:
                result = evaluate_registered_metric(
                    requirement,
                    binding,
                    trace,
                    plan=plan,
                    canonical_generalized_observable_id=getattr(
                        plan,
                        "observable_id",
                        None,
                    ),
                    validated_program=validated,
                    resolved_channels=resolved_channels,
                    verified_wrench_audit=verified_audit,
                    verified_comparison=comparison_gate,
                )
            except MetricEvaluationError as exc:
                failures.append(
                    f"{requirement_id}:{test_load_case_id}:{branch}:{exc.code}"
                )
                continue
            result["test_program"] = test_program
            result["test_load_case_id"] = test_load_case_id
            evaluations.append(result)
            if result["passed"] is not True:
                measured_failures = True
                failures.append(
                    f"{requirement_id}:{test_load_case_id}:{branch}:target_not_met"
                )
    expected_count = sum(
        len(item.get("required_branches", []))
        for item in contract.get("requirements", [])
        if isinstance(item, Mapping)
    )
    if not failures and expected_count > 0 and len(evaluations) == expected_count:
        status = "PASS"
    elif measured_failures:
        status = "FAIL"
    else:
        status = "INCONCLUSIVE"
    payload = {
        "schema_version": TASK_ACCEPTANCE_SCHEMA,
        "status": status,
        "certificate_authority": status == "PASS",
        "contract_sha256": contract.get("contract_sha256"),
        "task_contract_sha256": contract.get("task_contract_sha256"),
        "task_binding_sha256": semantic.get("task_binding_sha256"),
        "evaluation_count": len(evaluations),
        "expected_evaluation_count": expected_count,
        "evaluations": evaluations,
        "failure_reasons": failures,
    }
    payload["task_acceptance_sha256"] = _canonical_sha256(payload)
    return payload
