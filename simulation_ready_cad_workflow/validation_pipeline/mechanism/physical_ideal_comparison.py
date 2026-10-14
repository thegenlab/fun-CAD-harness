from __future__ import annotations

"""Physical behavior gates paired with deterministic ideal references."""

from typing import Any

from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error


def _physical_semantic_failures(trace: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    engine = trace.get("engine")
    semantics = trace.get("branch_semantics")
    if not isinstance(engine, dict) or engine.get(
        "internal_ideal_joint_count"
    ) != 0:
        failures.append("physical_branch_contains_internal_ideal_joint")
    if not isinstance(semantics, dict) or semantics.get(
        "replaced_contacts_disabled"
    ) is not False:
        failures.append("physical_contact_pair_is_disabled")
    return failures


def compare_physical_behavior_to_ideal_reference(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    ideal_comparison: dict[str, Any],
    *,
    workspace_root: str,
) -> dict[str, Any]:
    """Gate physical response and then apply the saved-pose ideal error gate."""

    return _compare_physical_behavior_to_ideal_reference(
        plan,
        load_case,
        physical_trace,
        ideal_comparison,
        workspace_root=workspace_root,
        defer_saved_pose_geometry_audit=False,
    )


def compare_composite_lane_physical_behavior_to_ideal_reference(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    ideal_comparison: dict[str, Any],
    *,
    workspace_root: str,
) -> dict[str, Any]:
    """Compare behavior while reserving CAD/FCL legality for the outer gate."""

    return _compare_physical_behavior_to_ideal_reference(
        plan,
        load_case,
        physical_trace,
        ideal_comparison,
        workspace_root=workspace_root,
        defer_saved_pose_geometry_audit=True,
    )


def _compare_physical_behavior_to_ideal_reference(
    plan: MechanismPlan,
    load_case: LoadCase,
    physical_trace: dict[str, Any],
    ideal_comparison: dict[str, Any],
    *,
    workspace_root: str,
    defer_saved_pose_geometry_audit: bool,
) -> dict[str, Any]:

    try:
        physical_metrics, thresholds, physical_failures = _physical_behavior(
            plan,
            load_case,
            physical_trace,
        )
    except MechanismValidationError as exc:
        return {
            "schema_version": "physical_ideal_comparison_v1",
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "purpose": load_case.purpose,
            "program": load_case.program,
            "required_for_certificate": load_case.required_for_certificate,
            "comparison_mode": "physical_vs_ideal_reference",
            "verdict": "PHYSICS_INCONCLUSIVE",
            "failure_reasons": [exc.code],
            "detail": str(exc),
        }
    semantic_failures = _physical_semantic_failures(physical_trace)
    ideal_verdict = ideal_comparison.get("verdict")
    ideal_failures = list(ideal_comparison.get("failure_reasons") or [])
    if semantic_failures or ideal_verdict not in {"PASS", "CLAIM_MISMATCH"}:
        verdict = "PHYSICS_INCONCLUSIVE"
    elif physical_failures:
        verdict = "PHYSICAL_FAILURE"
    elif ideal_verdict == "CLAIM_MISMATCH":
        verdict = "CLAIM_MISMATCH"
    else:
        verdict = "PASS"
    failures = [
        *semantic_failures,
        *physical_failures,
        *([] if ideal_verdict == "PASS" else ideal_failures),
    ]
    result = {
        "schema_version": "physical_ideal_comparison_v1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "comparison_mode": "physical_vs_ideal_reference",
        "verdict": verdict,
        "failure_reasons": failures,
        "thresholds": thresholds,
        "physical_metrics": physical_metrics,
        "ideal_reference_metrics": ideal_comparison.get("metrics"),
        "ideal_reference_threshold": ideal_comparison.get("threshold"),
        "contact_evidence_authority": False,
        "strict_non_claims": {
            "ideal_reference_dynamic_response_certified": False,
            "ideal_reference_contact_response_certified": False,
            "timing_or_velocity_match_used_as_gate": False,
        },
    }
    if defer_saved_pose_geometry_audit:
        result["saved_pose_geometry_audit"] = {
            "status": "DEFERRED",
            "required_environment": "outer_cad_after_queue_barrier",
            "worker_audit_authority": False,
        }
    elif plan.family == "compound":
        from .physical_legality import apply_saved_physical_legality_gate

        result = apply_saved_physical_legality_gate(
            result,
            physical_trace,
            workspace_root=workspace_root,
        )
    return result


def _physical_behavior(
    plan: MechanismPlan,
    load_case: LoadCase,
    trace: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Resolve the physical evaluator from the plan's adapter.

    No per-family branch may live here.
    """

    from .family_plugins import (
        mechanism_adapter_for_plan,
        plugin_registration_or_none,
    )

    family_spec = (
        plan.adapter_spec.get("family_spec")
        if isinstance(plan.adapter_spec, dict)
        else None
    )
    plugin, _adapter = mechanism_adapter_for_plan(plan)
    registration = plugin_registration_or_none(str(plugin.plugin_id))
    behavior = invoke_plugin_callback(
        lambda: plugin.physical_behavior(
            plan,
            load_case,
            trace,
            family_spec,
        ),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="physical_behavior",
        source_component="mechanism.physical_ideal_comparison.physical_behavior",
        unexpected_code="MECHANISM_ADAPTER_PHYSICAL_BEHAVIOR_FAILED",
        details={
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
        },
    )
    if behavior is not None:
        if (
            not isinstance(behavior, tuple)
            or len(behavior) != 3
            or not isinstance(behavior[0], dict)
            or not isinstance(behavior[1], dict)
            or not isinstance(behavior[2], list)
            or any(not isinstance(item, str) for item in behavior[2])
        ):
            message = (
                "mechanism adapter physical_behavior() returned an invalid result"
            )
            raise plugin_boundary_error(
                TypeError(message),
                registration=registration,
                plugin_id=str(plugin.plugin_id),
                plugin_kind="mechanism",
                callback="physical_behavior",
                source_component=(
                    "mechanism.physical_ideal_comparison.physical_behavior"
                ),
                code="MECHANISM_ADAPTER_PHYSICAL_BEHAVIOR_INVALID",
                message=message,
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case.load_case_id,
                },
                contract_violation=True,
            )
        return behavior
    message = "selected mechanism adapter returned no physical behavior evaluator"
    raise plugin_boundary_error(
        TypeError(message),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="physical_behavior",
        source_component="mechanism.physical_ideal_comparison.physical_behavior",
        code="MECHANISM_ADAPTER_PHYSICAL_BEHAVIOR_REQUIRED",
        message=message,
        details={
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
        },
        contract_violation=True,
    )
