from __future__ import annotations

"""Compare physical saved poses with the deterministic ideal reference."""

import math
from typing import Any, Mapping

from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error


IDEAL_COMPARISON_SCHEMA = "ideal_reference_comparison_v1"


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _finite(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise _fail("IDEAL_COMPARISON_NUMBER_INVALID", f"{label} must be finite")
    return float(value)


def _identity(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical: Mapping[str, Any],
    ideal: Mapping[str, Any],
) -> None:
    common = {
        "status": "COMPLETED",
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
    }
    mismatches: list[str] = []
    for key, expected in common.items():
        if physical.get(key) != expected:
            mismatches.append(f"physical.{key}")
        if ideal.get(key) != expected:
            mismatches.append(f"ideal_reference.{key}")
    if physical.get("branch") != "physical":
        mismatches.append("physical.branch")
    if ideal.get("branch") != "ideal_reference":
        mismatches.append("ideal_reference.branch")
    if mismatches:
        raise _fail(
            "IDEAL_COMPARISON_IDENTITY_MISMATCH",
            "trace identity differs at: " + ", ".join(mismatches),
        )


def _fields_and_threshold(
    plan: MechanismPlan,
) -> tuple[tuple[str, ...], float, str, str]:
    """Resolve the saved-pose protocol from the plan's adapter.

    No per-family branch may live here.
    """

    from .family_plugins import (
        mechanism_adapter_for_plan,
        plugin_registration_or_none,
    )

    family_spec = (
        plan.adapter_spec.get("family_spec")
        if isinstance(plan.adapter_spec, Mapping)
        else None
    )
    plugin, _adapter = mechanism_adapter_for_plan(plan)
    registration = plugin_registration_or_none(str(plugin.plugin_id))
    protocol = invoke_plugin_callback(
        lambda: plugin.ideal_comparison_protocol(plan, family_spec),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="ideal_comparison_protocol",
        source_component=(
            "mechanism.ideal_comparison.ideal_comparison_protocol"
        ),
        unexpected_code="MECHANISM_ADAPTER_IDEAL_COMPARISON_FAILED",
        details={"capsule_id": plan.capsule_id},
    )
    if protocol is not None:
        valid_protocol = (
            isinstance(protocol, tuple)
            and len(protocol) == 4
            and isinstance(protocol[0], tuple)
            and bool(protocol[0])
            and all(isinstance(item, str) and item for item in protocol[0])
            and not isinstance(protocol[1], bool)
            and isinstance(protocol[1], (int, float))
            and math.isfinite(float(protocol[1]))
            and float(protocol[1]) >= 0.0
            and isinstance(protocol[2], str)
            and bool(protocol[2])
            and isinstance(protocol[3], str)
            and bool(protocol[3])
        )
        if not valid_protocol:
            message = (
                "mechanism adapter ideal_comparison_protocol() returned "
                "an invalid protocol"
            )
            raise plugin_boundary_error(
                TypeError(message),
                registration=registration,
                plugin_id=str(plugin.plugin_id),
                plugin_kind="mechanism",
                callback="ideal_comparison_protocol",
                source_component=(
                    "mechanism.ideal_comparison.ideal_comparison_protocol"
                ),
                code="MECHANISM_ADAPTER_IDEAL_COMPARISON_INVALID",
                message=message,
                details={"capsule_id": plan.capsule_id},
                contract_violation=True,
            )
        return protocol
    message = "selected mechanism adapter returned no ideal comparison protocol"
    raise plugin_boundary_error(
        TypeError(message),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="ideal_comparison_protocol",
        source_component=(
            "mechanism.ideal_comparison.ideal_comparison_protocol"
        ),
        code="MECHANISM_ADAPTER_IDEAL_COMPARISON_REQUIRED",
        message=message,
        details={"capsule_id": plan.capsule_id},
        contract_violation=True,
    )


def compare_physical_to_ideal(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical: Mapping[str, Any],
    ideal: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate same-time physical/ideal output error; contact is out of scope."""

    _identity(plan, load_case, physical, ideal)
    physical_samples = physical.get("samples")
    ideal_samples = ideal.get("samples")
    if (
        not isinstance(physical_samples, list)
        or not isinstance(ideal_samples, list)
        or len(physical_samples) < 2
        or len(physical_samples) != len(ideal_samples)
    ):
        raise _fail(
            "IDEAL_COMPARISON_SAMPLE_MISMATCH",
            "physical and ideal traces must have the same saved samples",
        )
    fields, threshold, unit, source = _fields_and_threshold(plan)
    if threshold < 0.0 or not math.isfinite(threshold):
        raise _fail(
            "IDEAL_COMPARISON_THRESHOLD_INVALID",
            "ideal comparison tolerance must be finite and non-negative",
        )
    squared = 0.0
    count = 0
    field_maxima = {field: 0.0 for field in fields}
    worst: dict[str, Any] | None = None
    family_spec = (
        plan.adapter_spec.get("family_spec")
        if isinstance(plan.adapter_spec, Mapping)
        else None
    )
    required_sample_policy = (
        family_spec.get("ideal_comparison_sample_policy")
        if isinstance(family_spec, Mapping)
        else None
    )
    ideal_domain = ideal.get("domain")
    declared_sample_policy = (
        ideal_domain.get("comparison_sample_policy")
        if isinstance(ideal_domain, Mapping)
        else None
    )
    if isinstance(required_sample_policy, str) and required_sample_policy:
        if declared_sample_policy != required_sample_policy:
            raise _fail(
                "IDEAL_COMPARISON_SAMPLE_POLICY_MISMATCH",
                "ideal reference does not implement the hash-bound comparison sample policy",
            )
        sample_policy = required_sample_policy
    else:
        sample_policy = "all_saved_samples"
    if sample_policy not in {"all_saved_samples", "phase_terminal"}:
        raise _fail(
            "IDEAL_COMPARISON_SAMPLE_POLICY_INVALID",
            f"unsupported ideal comparison sample policy: {sample_policy!r}",
        )
    if sample_policy == "phase_terminal":
        compared_indices = {
            index
            for index, sample in enumerate(physical_samples)
            if index == len(physical_samples) - 1
            or not isinstance(sample, Mapping)
            or not isinstance(physical_samples[index + 1], Mapping)
            or physical_samples[index + 1].get("phase_id")
            != sample.get("phase_id")
        }
    else:
        compared_indices = set(range(len(physical_samples)))
    for index, (physical_sample, ideal_sample) in enumerate(
        zip(physical_samples, ideal_samples)
    ):
        if not isinstance(physical_sample, Mapping) or not isinstance(
            ideal_sample, Mapping
        ):
            raise _fail(
                "IDEAL_COMPARISON_SAMPLE_INVALID",
                f"sample {index} is not an object",
            )
        physical_time = _finite(
            physical_sample.get("time_s"),
            f"physical.samples[{index}].time_s",
        )
        ideal_time = _finite(
            ideal_sample.get("time_s"),
            f"ideal.samples[{index}].time_s",
        )
        if (
            not math.isclose(physical_time, ideal_time, abs_tol=1e-12)
            or physical_sample.get("phase_id") != ideal_sample.get("phase_id")
        ):
            raise _fail(
                "IDEAL_COMPARISON_SAMPLE_IDENTITY_MISMATCH",
                f"saved sample {index} time or phase differs",
            )
        if index not in compared_indices:
            continue
        for field in fields:
            physical_value = _finite(
                physical_sample.get(field),
                f"physical.samples[{index}].{field}",
            )
            ideal_value = _finite(
                ideal_sample.get(field),
                f"ideal.samples[{index}].{field}",
            )
            error = abs(physical_value - ideal_value)
            squared += error * error
            count += 1
            field_maxima[field] = max(field_maxima[field], error)
            if worst is None or error > worst["absolute_error"]:
                worst = {
                    "sample_index": index,
                    "time_s": physical_time,
                    "phase_id": physical_sample.get("phase_id"),
                    "field": field,
                    "physical": physical_value,
                    "ideal_reference": ideal_value,
                    "absolute_error": error,
                }
    max_error = max(field_maxima.values())
    passed = max_error <= threshold
    return {
        "schema_version": IDEAL_COMPARISON_SCHEMA,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "program": load_case.program,
        "status": "PASS" if passed else "FAIL",
        "verdict": "PASS" if passed else "CLAIM_MISMATCH",
        "certificate_authority": passed,
        "contact_evidence_authority": False,
        "dynamic_response_authority": False,
        "coordinate_fields": list(fields),
        "unit": unit,
        "threshold": {
            "max_absolute_error": threshold,
            "unit": unit,
            "source": source,
        },
        "metrics": {
            "sample_policy": sample_policy,
            "source_sample_count": len(physical_samples),
            "sample_count": len(compared_indices),
            "coordinate_comparison_count": count,
            "max_absolute_error": max_error,
            "rmse": math.sqrt(squared / count),
            "field_max_absolute_error": field_maxima,
            "worst_sample": worst,
        },
        "failure_reasons": [] if passed else ["physical_ideal_output_error_exceeded"],
    }
