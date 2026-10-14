from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .adapters import adapter_from_dict, get_adapter
from .comparison import aggregate_verdict
from .io import canonical_sha256, relative_to_root, sha256_file, write_json_atomic
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error


def _mechanism_plugin(plan: MechanismPlan):
    """Resolve the family plugin without hiding a binding failure."""

    from .family_plugins import mechanism_adapter_for_plan

    plugin, _ = mechanism_adapter_for_plan(plan)
    return plugin


def _plugin_registration(plugin: Any) -> dict[str, Any]:
    from .family_plugins import plugin_registration_for

    return plugin_registration_for(str(plugin.plugin_id))


def _certificate_plugin_hook(
    plugin: Any,
    *arguments: Any,
    hook: str,
) -> dict[str, Any] | None:
    """Invoke one certificate hook through the common ownership boundary."""

    plugin_id = str(plugin.plugin_id)
    registration = _plugin_registration(plugin)
    result = invoke_plugin_callback(
        lambda: getattr(plugin, hook)(*arguments),
        registration=registration,
        plugin_id=plugin_id,
        plugin_kind="mechanism",
        callback=hook,
        source_component="mechanism.certificate",
        unexpected_code=f"MECHANISM_ADAPTER_{hook.upper()}_FAILED",
        details={"hook": hook, "fallback": False},
    )
    if result is not None and not isinstance(result, dict):
        code = f"MECHANISM_ADAPTER_{hook.upper()}_INVALID"
        error = TypeError(
            f"mechanism adapter {hook}() must return a mapping or None"
        )
        raise plugin_boundary_error(
            error,
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            callback=hook,
            source_component="mechanism.certificate",
            code=code,
            message=str(error),
            details={"hook": hook, "fallback": False},
            contract_violation=True,
        )
    return result


def _compound_plugin_hook(
    plugin: Any,
    adapter: Any,
    *,
    hook: str,
) -> dict[str, Any] | None:
    if plugin is None:
        return None
    return _certificate_plugin_hook(plugin, adapter, hook=hook)


def _compound_fallback_linkage(
    adapter: Any,
    plugin: Any,
    *,
    hook: str,
) -> Any:
    from .compound_spec import compound_linkage_from_dict

    try:
        return compound_linkage_from_dict(adapter.family_spec)
    except Exception as exc:
        if plugin is None:
            raise
        code = f"MECHANISM_ADAPTER_{hook.upper()}_FALLBACK_FAILED"
        boundary_error = plugin_boundary_error(
            exc,
            registration=_plugin_registration(plugin),
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback=hook,
            source_component="mechanism.certificate",
            code=code,
            message=(
                f"generic fallback after mechanism adapter {hook}() returned "
                f"None failed: {exc}"
            ),
            details={"hook": hook, "fallback": True},
            contract_violation=True,
        )
        if boundary_error is exc:
            raise
        raise boundary_error from exc


def _load_envelope(
    plan: MechanismPlan,
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    force_magnitudes = [
        phase.magnitude_si
        for load_case in load_cases
        for phase in load_case.phases
        if phase.wrench_kind == "force"
    ]
    torque_magnitudes = [
        phase.magnitude_si
        for load_case in load_cases
        for phase in load_case.phases
        if phase.wrench_kind == "torque"
    ]
    masses = [load_case.moving_mass_kg for load_case in load_cases]
    physical_metrics = [
        item["physical_metrics"]
        for item in comparisons
        if isinstance(item.get("physical_metrics"), dict)
    ]
    envelope: dict[str, Any] = {
        "moving_mass_kg": {"min": min(masses), "max": max(masses)},
        "load_case_ids": [load_case.load_case_id for load_case in load_cases],
        "purposes": sorted({load_case.purpose for load_case in load_cases}),
        "gravity_abs_m_s2": None,
    }
    plugin = _mechanism_plugin(plan)
    if plugin is not None:
        family_envelope = _certificate_plugin_hook(
            plugin,
            load_cases,
            comparisons,
            hook="load_envelope",
        )
        if family_envelope is not None:
            return family_envelope
    if plan.family == "compound":
        angle_metrics = [
            item
            for item in physical_metrics
            if isinstance(item.get("min_output_angle_rad"), (int, float))
            and not isinstance(item.get("min_output_angle_rad"), bool)
            and isinstance(item.get("max_output_angle_rad"), (int, float))
            and not isinstance(item.get("max_output_angle_rad"), bool)
        ]
        angle_envelope = (
            {
                "min": min(item["min_output_angle_rad"] for item in angle_metrics),
                "max": max(item["max_output_angle_rad"] for item in angle_metrics),
                "source": "observed_physical_branch_compound_output_angle",
            }
            if angle_metrics else None
        )
        joint_drifts = [
            float(item["max_joint_drift_m"])
            for item in physical_metrics
            if isinstance(item.get("max_joint_drift_m"), (int, float))
            and not isinstance(item.get("max_joint_drift_m"), bool)
        ]
        off_plane_rotations = [
            float(item["max_off_plane_rotation_rad"])
            for item in physical_metrics
            if isinstance(item.get("max_off_plane_rotation_rad"), (int, float))
            and not isinstance(item.get("max_off_plane_rotation_rad"), bool)
        ]
        envelope.update(
            {
                "applied_force_n": {
                    "min_nonzero": min(force_magnitudes) if force_magnitudes else 0.0,
                    "max": max(force_magnitudes) if force_magnitudes else 0.0,
                },
                "applied_torque_nm": {
                    "min_nonzero": min(torque_magnitudes) if torque_magnitudes else 0.0,
                    "max": max(torque_magnitudes) if torque_magnitudes else 0.0,
                },
                "generalized_output_angle_rad": angle_envelope,
                "max_observed_joint_drift_m": (
                    max(joint_drifts) if joint_drifts else None
                ),
                "max_observed_off_plane_rotation_rad": (
                    max(off_plane_rotations) if off_plane_rotations else None
                ),
            }
        )
    elif plan.family == "revolute":
        angle_envelope = (
            {
                "min": min(item["min_angle_rad"] for item in physical_metrics),
                "max": max(item["max_angle_rad"] for item in physical_metrics),
                "source": "observed_physical_branch_angle",
            }
            if physical_metrics else None
        )
        envelope.update(
            {
                "applied_torque_nm": {
                    "min_nonzero": min(torque_magnitudes) if torque_magnitudes else 0.0,
                    "max": max(torque_magnitudes) if torque_magnitudes else 0.0,
                },
                "generalized_angle_rad": angle_envelope,
            }
        )
    else:
        progress_envelope = (
            {
                "min": min(item["min_progress_m"] for item in physical_metrics),
                "max": max(item["max_progress_m"] for item in physical_metrics),
                "source": "observed_physical_branch_progress",
            }
            if physical_metrics else None
        )
        envelope.update(
            {
                "applied_force_n": {
                    "min_nonzero": min(force_magnitudes) if force_magnitudes else 0.0,
                    "max": max(force_magnitudes) if force_magnitudes else 0.0,
                },
                "generalized_progress_m": progress_envelope,
            }
        )
    return envelope


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _requires_moving_chain_audit(plan: MechanismPlan) -> bool:
    plugin = _mechanism_plugin(plan)
    return (
        plugin is not None
        and "moving_chain_path_audit" in plugin.required_trace_audit_sections
    )


def _outer_audit_passes_certificate_gate(
    audit: dict[str, Any],
    *,
    moving_chain_required: bool,
) -> bool:
    geometry = audit.get("geometry_legality")
    contact = audit.get("component_contact_intent")
    if not (
        audit.get("status") == "PHYSICAL_PREFLIGHT_PASSED"
        and audit.get("passed") is True
        and audit.get("failure_reasons") == []
        and audit.get("reduced_branch_authorized") is True
        and isinstance(audit.get("report"), str)
        and _is_sha256(audit.get("report_sha256"))
        and audit.get("report_binding_verified") is True
        and _is_sha256(audit.get("physical_trace_sha256"))
        and audit.get("physical_trace_binding_verified") is True
        and audit.get("audit_execution_environment")
        == "outer_cad_after_queue_barrier"
        and audit.get("worker_produced_audit_accepted") is False
        and isinstance(geometry, dict)
        and geometry.get("passed") is True
        and isinstance(contact, dict)
        and contact.get("passed") is True
    ):
        return False
    if not moving_chain_required:
        return True
    moving_chain = audit.get("moving_chain_path_audit")
    return (
        contact.get("authoritative") is True
        and isinstance(moving_chain, dict)
        and moving_chain.get("authoritative") is True
        and moving_chain.get("passed") is True
        and moving_chain.get("fixed_spatial_flex_zones_verified") is True
        and moving_chain.get("saved_state_clearance_verified") is True
        and moving_chain.get("original_step_components_used") is True
    )


def build_certificate(
    plan: MechanismPlan,
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
    *,
    trace_files: Iterable[str | Path] = (),
    workspace_root: str | Path | None = None,
    task_acceptance: dict[str, Any] | None = None,
    ideal_reference_comparisons: tuple[dict[str, Any], ...] = (),
    physical_trace_audits: tuple[dict[str, Any], ...] = (),
    replacement_authorized: bool = True,
    replacement_policy: str = "certified_mechanism",
) -> dict[str, Any]:
    expected = {load_case.load_case_id for load_case in load_cases}
    actual = {str(item.get("load_case_id")) for item in comparisons}
    if expected != actual or len(comparisons) != len(load_cases):
        raise MechanismValidationError(
            "CERTIFICATE_EVIDENCE_INCOMPLETE",
            "comparison evidence does not cover the complete validator load suite",
        )
    comparison_verdict = aggregate_verdict(comparisons)
    ordered_physical_audits: tuple[dict[str, Any], ...] = ()
    if physical_trace_audits:
        audits_by_load_case: dict[str, dict[str, Any]] = {}
        for item in physical_trace_audits:
            if not isinstance(item, dict):
                raise MechanismValidationError(
                    "CERTIFICATE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
                    "physical trace audit evidence must be a mapping",
                )
            capsule_id = item.get("capsule_id")
            load_case_id = item.get("load_case_id")
            if (
                capsule_id != plan.capsule_id
                or not isinstance(load_case_id, str)
                or load_case_id in audits_by_load_case
            ):
                raise MechanismValidationError(
                    "CERTIFICATE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
                    "physical trace audit identities are invalid or duplicated",
                )
            audits_by_load_case[load_case_id] = item
        if set(audits_by_load_case) != expected:
            raise MechanismValidationError(
                "CERTIFICATE_PHYSICAL_AUDIT_EVIDENCE_INCOMPLETE",
                "physical trace audits do not cover the complete validator load suite",
            )
        ordered_physical_audits = tuple(
            audits_by_load_case[load_case.load_case_id]
            for load_case in load_cases
        )
    moving_chain_audit_required = _requires_moving_chain_audit(plan)
    physical_audit_gate_results = tuple(
        _outer_audit_passes_certificate_gate(
            item,
            moving_chain_required=moving_chain_audit_required,
        )
        for item in ordered_physical_audits
    )
    physical_audit_gate_passed = (
        not ordered_physical_audits
        or all(physical_audit_gate_results)
    )
    if physical_audit_gate_passed:
        verdict = comparison_verdict
    elif any(
        item.get("status") == "PHYSICAL_PREFLIGHT_FAILED"
        for item in ordered_physical_audits
    ):
        verdict = "PHYSICAL_FAILURE"
    else:
        verdict = "PHYSICS_INCONCLUSIVE"
    envelope = _load_envelope(plan, load_cases, comparisons)
    envelope["gravity_abs_m_s2"] = list(plan.gravity_abs_m_s2)
    contact_model: dict[str, Any] | None = None
    compound_model: dict[str, Any] | None = None
    adapter = None
    if plan.adapter_id is not None:
        try:
            adapter = (
                adapter_from_dict(plan.adapter_spec)
                if plan.adapter_spec is not None
                else get_adapter(plan.adapter_id)
            )
        except (KeyError, TypeError, ValueError):
            adapter = None
        geometry = None if adapter is None else adapter.geometry
        if geometry is not None:
            rolling = geometry.rolling_elements
            contact_model = {
                "adapter_id": plan.adapter_id,
                "certifiable_progress_limit_m": (
                    geometry.certifiable_progress_limit_m
                ),
                "certifiable_angle_limit_rad": (
                    geometry.certifiable_angle_limit_rad
                ),
                "independent_rolling_element_count": (
                    len(rolling.elements) if rolling is not None else 0
                ),
                "return_path_resolved": (
                    rolling.return_path_resolved if rolling is not None else None
                ),
            }
        if plan.family == "compound" and adapter is not None:
            plugin = _mechanism_plugin(plan)
            compound_model = _compound_plugin_hook(
                plugin,
                adapter,
                hook="certificate_model",
            )
            if compound_model is None:
                linkage = _compound_fallback_linkage(
                    adapter,
                    plugin,
                    hook="certificate_model",
                )
                compound_model = {
                    "adapter_id": plan.adapter_id,
                    "linkage_id": linkage.linkage_id,
                    "topology_sha256": canonical_sha256(linkage.to_dict()),
                    "body_count": len(linkage.bodies),
                    "joint_count": len(linkage.revolute_joints),
                    "loop_count": linkage.loop_count,
                    "input_body_id": linkage.input_body_id,
                    "input_joint_id": linkage.input_joint_id,
                    "input_force_direction_abs": list(linkage.input_force_direction_abs),
                    "input_actuation_kind": linkage.input_actuation_kind,
                    "input_actuation_direction_abs": list(
                        linkage.input_actuation_direction_abs
                    ),
                    "output_body_id": linkage.output_body_id,
                    "output_joint_id": linkage.output_joint_id,
                    "physical_support": "finite_ChLoadBodyBodyBushingGeneric",
                    "reduced_support": "declared_ChLinkLockRevolute_topology",
                    "exact_non_joint_surface_contact_certified": False,
                }
    collider_fidelity_evidence: dict[str, Any] | None = None
    rigid_attachment_evidence: dict[str, Any] | None = None
    if isinstance(plan.adapter_spec, dict):
        binding = plan.adapter_spec.get("collider_fidelity")
        if isinstance(binding, dict):
            collider_fidelity_evidence = {
                key: binding[key]
                for key in (
                    "schema_version",
                    "status",
                    "passed",
                    "report",
                    "report_sha256",
                    "package_binding_sha256",
                )
                if key in binding
            }
            attachment_binding = binding.get("rigid_attachment_proof")
            if isinstance(attachment_binding, dict):
                rigid_attachment_evidence = {
                    key: attachment_binding[key]
                    for key in (
                        "schema_version",
                        "status",
                        "passed",
                        "report",
                        "report_sha256",
                        "package_binding_sha256",
                    )
                    if key in attachment_binding
                }
    workspace = Path(workspace_root).resolve() if workspace_root is not None else None
    evidence_hashes = {}
    for path in trace_files:
        resolved = Path(path).resolve()
        key = (
            relative_to_root(workspace, resolved)
            if workspace is not None
            else str(resolved)
        )
        evidence_hashes[key] = sha256_file(resolved)
    threshold_sources = sorted(
        {
            item.get("thresholds", {}).get("source")
            for item in comparisons
            if isinstance(item.get("thresholds"), dict)
        }
        - {None}
    )
    deferred_comparison_count = sum(
        isinstance(item.get("saved_pose_geometry_audit"), dict)
        and item["saved_pose_geometry_audit"].get("status") == "DEFERRED"
        for item in comparisons
    )
    if ordered_physical_audits:
        geometry_audits = [
            item.get("geometry_legality")
            for item in ordered_physical_audits
            if isinstance(item.get("geometry_legality"), dict)
        ]
        geometry_evidence_source = "outer_physical_trace_audit"
        geometry_expected_count = len(ordered_physical_audits)
        broad_exclusion_count = sum(
            len(item.get("broad_body_pair_collision_exclusions") or [])
            for item in ordered_physical_audits
        )
    else:
        geometry_audits = [
            item.get("physical_geometry_legality")
            for item in comparisons
            if isinstance(item.get("physical_geometry_legality"), dict)
        ]
        geometry_evidence_source = "comparison_legacy_compatibility"
        geometry_expected_count = len(comparisons)
        broad_exclusion_count = sum(
            len(item.get("broad_body_pair_collision_exclusions") or [])
            for item in comparisons
        )
    completed_geometry_audits = [
        item
        for item in geometry_audits
        if isinstance(item.get("passed"), bool)
        and (
            geometry_evidence_source == "outer_physical_trace_audit"
            or item.get("audit_completed") is True
        )
    ]
    geometry_legality = {
        "required_for_compound_family": plan.family == "compound",
        "comparison_count": len(comparisons),
        "evidence_source": geometry_evidence_source,
        "audit_record_count": len(geometry_audits),
        "deferred_comparison_count": deferred_comparison_count,
        "completed_audit_count": len(completed_geometry_audits),
        "all_saved_physical_poses_audited": (
            bool(geometry_audits)
            and len(completed_geometry_audits) == geometry_expected_count
        ),
        "all_audits_passed": (
            bool(geometry_audits)
            and len(geometry_audits) == geometry_expected_count
            and all(item.get("passed") is True for item in geometry_audits)
        ),
        "max_observed_interpenetration_m": max(
            (
                float(item.get("max_penetration_m") or 0.0)
                for item in completed_geometry_audits
            ),
            default=None,
        ),
        "broad_body_pair_collision_exclusion_count": broad_exclusion_count,
    }
    contact_source_items = (
        ordered_physical_audits
        if ordered_physical_audits
        else comparisons
    )
    contact_intent_audits = [
        item.get("component_contact_intent")
        for item in contact_source_items
        if isinstance(item.get("component_contact_intent"), dict)
    ]
    contact_intent_outcomes = [
        outcome
        for audit in contact_intent_audits
        for outcome in audit.get("outcomes") or ()
        if isinstance(outcome, dict)
    ]
    component_contact_evidence = {
        "schema_version": "component_contact_certificate_summary_v1",
        "comparison_count": len(comparisons),
        "evidence_source": (
            "outer_physical_trace_audit"
            if ordered_physical_audits
            else "comparison_legacy_compatibility"
        ),
        "deferred_comparison_count": deferred_comparison_count,
        "audit_count": len(contact_intent_audits),
        "authoritative_audit_count": sum(
            item.get("authoritative") is True for item in contact_intent_audits
        ),
        "all_authoritative_audits_passed": (
            bool(contact_intent_audits)
            and len(contact_intent_audits) == len(contact_source_items)
            and all(
                item.get("authoritative") is True
                and item.get("passed") is True
                for item in contact_intent_audits
            )
        ),
        "expected_contact_intent_count": sum(
            item.get("kind") == "EXPECTED_REAL_CONTACT"
            for item in contact_intent_outcomes
        ),
        "observed_expected_contact_count": sum(
            item.get("status") == "EXPECTED_CONTACT_OBSERVED"
            for item in contact_intent_outcomes
        ),
        "forbidden_contact_failure_count": sum(
            item.get("status") == "FORBIDDEN_CONTACT_OBSERVED"
            for item in contact_intent_outcomes
        ),
        "uncertain_attribution_event_count": sum(
            int(item.get("ambiguous_attribution_event_count") or 0)
            for item in contact_intent_audits
        ),
        "runtime_body_pair_contacts_used_as_component_evidence": False,
        "audits": contact_intent_audits,
    }
    compact_physical_audit_records = [
        {
            **{
                key: item.get(key)
                for key in (
                    "capsule_id",
                    "load_case_id",
                    "status",
                    "passed",
                    "reduced_branch_authorized",
                    "report",
                    "report_sha256",
                    "report_binding_verified",
                    "physical_trace_sha256",
                    "physical_trace_binding_verified",
                    "audit_execution_environment",
                    "worker_produced_audit_accepted",
                    "failure_reasons",
                )
            },
            "certificate_gate_passed": gate_passed,
        }
        for item, gate_passed in zip(
            ordered_physical_audits,
            physical_audit_gate_results,
            strict=True,
        )
    ]
    physical_audit_evidence = {
        "schema_version": "physical_trace_audit_certificate_binding_v1",
        "evidence_source": (
            "outer_physical_trace_audit"
            if ordered_physical_audits
            else "comparison_legacy_compatibility"
        ),
        "gate_required": bool(ordered_physical_audits),
        "gate_status": (
            "PASS"
            if ordered_physical_audits and physical_audit_gate_passed
            else "FAIL"
            if ordered_physical_audits
            else "LEGACY_NOT_PROVIDED"
        ),
        "gate_passed": physical_audit_gate_passed,
        "expected_record_count": len(load_cases),
        "record_count": len(ordered_physical_audits),
        "all_records_passed": (
            physical_audit_gate_passed
            if ordered_physical_audits
            else None
        ),
        "comparison_verdict": comparison_verdict,
        "moving_chain_audit_required": moving_chain_audit_required,
        "records_sha256": canonical_sha256(compact_physical_audit_records),
        "records": compact_physical_audit_records,
    }
    chain_path_audits = [
        item.get("moving_chain_path_audit")
        for item in ordered_physical_audits
        if isinstance(item.get("moving_chain_path_audit"), dict)
    ]
    chain_path_sample_counts = [
        int(item.get("audit_sample_count") or item.get("sample_count") or 0)
        for item in chain_path_audits
    ]
    moving_chain_path_evidence = {
        "schema_version": "moving_chain_path_certificate_summary_v1",
        "evidence_source": (
            "outer_physical_trace_audit"
            if ordered_physical_audits
            else "not_provided"
        ),
        "audit_record_count": len(chain_path_audits),
        "all_saved_path_states_audited": (
            bool(chain_path_audits)
            and len(chain_path_audits) == len(ordered_physical_audits)
        ),
        "all_audits_passed": (
            bool(chain_path_audits)
            and len(chain_path_audits) == len(ordered_physical_audits)
            and all(item.get("passed") is True for item in chain_path_audits)
        ),
        "fixed_spatial_flex_zones_verified": (
            bool(chain_path_audits)
            and all(
                item.get("fixed_spatial_flex_zones_verified") is True
                for item in chain_path_audits
            )
        ),
        "saved_state_clearance_verified": (
            bool(chain_path_audits)
            and all(
                item.get("saved_state_clearance_verified") is True
                for item in chain_path_audits
            )
        ),
        "original_step_components_used": (
            bool(chain_path_audits)
            and all(
                item.get("original_step_components_used") is True
                for item in chain_path_audits
            )
        ),
        "minimum_audit_sample_count": (
            min(chain_path_sample_counts)
            if chain_path_sample_counts
            else None
        ),
    }
    typed_contract_authoritative = (
        isinstance(plan.acceptance_contract, dict)
        and plan.acceptance_contract.get("certificate_authority") is True
        and plan.acceptance_contract.get("status") == "PASS"
    )
    task_evidence = (
        dict(task_acceptance)
        if isinstance(task_acceptance, dict)
        else {
            "status": "LEGACY_UNTYPED",
            "certificate_authority": False,
            "failure_reasons": ["typed_task_evidence_missing"],
        }
    )
    ideal_by_load_case = {
        str(item.get("load_case_id")): item
        for item in ideal_reference_comparisons
        if isinstance(item, dict)
    }
    required_ideal_cases = sorted(
        {
            str(item.get("test_load_case_id"))
            for item in (plan.acceptance_contract or {}).get(
                "requirements", []
            )
            if isinstance(item, dict)
            and "ideal_reference" in item.get("required_branches", [])
        }
    )
    missing_ideal_cases = [
        case_id for case_id in required_ideal_cases
        if case_id not in ideal_by_load_case
    ]
    failed_ideal_cases = [
        case_id for case_id in required_ideal_cases
        if case_id in ideal_by_load_case
        and ideal_by_load_case[case_id].get("verdict")
        == "CLAIM_MISMATCH"
    ]
    inconclusive_ideal_cases = [
        case_id for case_id in required_ideal_cases
        if case_id in ideal_by_load_case
        and ideal_by_load_case[case_id].get("verdict")
        not in {"PASS", "CLAIM_MISMATCH"}
    ]
    if not required_ideal_cases:
        ideal_gate_status = "NOT_REQUIRED"
    elif failed_ideal_cases:
        ideal_gate_status = "FAIL"
    elif missing_ideal_cases or inconclusive_ideal_cases:
        ideal_gate_status = "INCONCLUSIVE"
    else:
        ideal_gate_status = "PASS"
    task_status = str(task_evidence.get("status"))
    if not typed_contract_authoritative:
        typed_task_verdict = "LEGACY_UNTYPED"
    elif task_status == "FAIL" or ideal_gate_status == "FAIL":
        typed_task_verdict = "FAIL"
    elif task_status != "PASS" or ideal_gate_status == "INCONCLUSIVE":
        typed_task_verdict = "INCONCLUSIVE"
    else:
        typed_task_verdict = "PASS"
    physical_audit_failure_reasons = [
        str(reason)
        for item, gate_passed in zip(
            ordered_physical_audits,
            physical_audit_gate_results,
            strict=True,
        )
        if not gate_passed
        for reason in (
            item.get("failure_reasons")
            or [f"physical_trace_audit_not_passed:{item.get('load_case_id')}"]
        )
    ]
    replacement_eligible = (
        verdict == "KINEMATIC_CERTIFIED"
        and (
            not typed_contract_authoritative
            or typed_task_verdict == "PASS"
        )
        and replacement_authorized
    )
    certificate = {
        "schema_version": "0.1",
        "verdict": verdict,
        "comparison_verdict": comparison_verdict,
        "certificate_level": (
            "KINEMATIC_CERTIFIED" if verdict == "KINEMATIC_CERTIFIED" else None
        ),
        "mechanism_verdict": verdict,
        "typed_task_verdict": typed_task_verdict,
        "overall_verdict": (
            "TASK_CERTIFIED"
            if verdict == "KINEMATIC_CERTIFIED"
            and typed_task_verdict == "PASS"
            else verdict
            if not typed_contract_authoritative
            else "NOT_CERTIFIED"
        ),
        "replacement_eligible": replacement_eligible,
        "replacement_policy": replacement_policy,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "package": {
            "package_id": plan.package_id,
            "design_version": plan.design_version,
            "package_binding_sha256": plan.package_binding_sha256,
            "intake_report_sha256": plan.intake_report_sha256,
            "artifact_hashes": dict(sorted(plan.artifact_hashes.items())),
            "source_component_hashes": dict(
                sorted(plan.source_component_hashes.items())
            ),
        },
        "mechanism": {
            "capsule_id": plan.capsule_id,
            "claim_provenance": "ai_declared",
            "family": plan.family,
            "protocol_id": plan.protocol_id,
            "fixed_group_ids": list(plan.fixed_group_ids),
            "moving_group_ids": list(plan.moving_group_ids),
            "axis_abs": list(plan.axis_abs),
            "origin_abs_m": list(plan.joint_origin_abs_m),
            "observable_ids": [plan.observable_id],
            "task_ids": list(plan.task_ids),
        },
        "tested_envelope": envelope,
        "physical_contact_model": contact_model,
        "physical_trace_audit_evidence": physical_audit_evidence,
        "physical_geometry_legality": geometry_legality,
        "component_contact_evidence": component_contact_evidence,
        "moving_chain_path_evidence": moving_chain_path_evidence,
        "compound_mechanism_model": compound_model,
        "threshold_sources": threshold_sources,
        "comparison_sha256": canonical_sha256(list(comparisons)),
        "comparisons": list(comparisons),
        "trace_file_hashes": evidence_hashes,
        "rigid_attachment_evidence": rigid_attachment_evidence,
        "collider_fidelity_evidence": collider_fidelity_evidence,
        "ideal_reference_evidence": {
            "required_load_case_ids": required_ideal_cases,
            "status": ideal_gate_status,
            "missing_load_case_ids": missing_ideal_cases,
            "failed_load_case_ids": failed_ideal_cases,
            "inconclusive_load_case_ids": inconclusive_ideal_cases,
            "comparisons": list(ideal_reference_comparisons),
            "contact_evidence_authority": False,
        },
        "typed_task_evidence": {
            "acceptance_contract": plan.acceptance_contract,
            "semantic_binding": plan.semantic_binding,
            "evaluation": task_evidence,
            "verdict": typed_task_verdict,
            "certificate_authority": typed_task_verdict == "PASS",
        },
        "failures": sorted(
            {
                reason
                for reason in (
                    *(
                        reason
                        for item in comparisons
                        for reason in item.get("failure_reasons", [])
                    ),
                    *physical_audit_failure_reasons,
                )
            }
        ),
        "strict_non_claims": {
            "dynamic_response_certified": False,
            "real_world_manufacturing_certified": False,
            "strength_fatigue_or_lifetime_certified": False,
            "real_friction_efficiency_or_preload_certified": False,
            "valid_outside_tested_envelope": False,
        },
    }
    return certificate


def build_replacement(
    plan: MechanismPlan,
    certificate: dict[str, Any],
    *,
    certificate_sha256: str,
) -> dict[str, Any]:
    if certificate.get("verdict") != "KINEMATIC_CERTIFIED":
        raise MechanismValidationError(
            "REPLACEMENT_NOT_AUTHORIZED",
            "only KINEMATIC_CERTIFIED mechanisms can generate a replacement",
        )
    compound_replacement: dict[str, Any] | None = None
    if plan.family == "compound":
        adapter = (
            adapter_from_dict(plan.adapter_spec)
            if plan.adapter_spec is not None
            else get_adapter(plan.adapter_id)
        )
        plugin = _mechanism_plugin(plan)
        compound_replacement = _compound_plugin_hook(
            plugin,
            adapter,
            hook="replacement_topology",
        )
        if compound_replacement is None:
            linkage = _compound_fallback_linkage(
                adapter,
                plugin,
                hook="replacement_topology",
            )
            compound_replacement = {
                "linkage_id": linkage.linkage_id,
                "family_spec_sha256": canonical_sha256(linkage.to_dict()),
                "body_ids": [item.body_id for item in linkage.bodies],
                "joint_ids": [item.joint_id for item in linkage.revolute_joints],
                "loop_count": linkage.loop_count,
                "input_body_id": linkage.input_body_id,
                "output_body_id": linkage.output_body_id,
                "output_angle_range_rad": [
                    linkage.home_output_angle_rad,
                    linkage.outbound_output_angle_rad,
                ],
            }
    return {
        "schema_version": "0.1",
        "status": "ACTIVE",
        "certificate_sha256": certificate_sha256,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "replacement_model": {
            "family": plan.family,
            "fixed_group_ids": list(plan.fixed_group_ids),
            "moving_group_ids": list(plan.moving_group_ids),
            "axis_abs": list(plan.axis_abs),
            "origin_abs_m": list(plan.joint_origin_abs_m),
            "compound_topology": compound_replacement,
        },
        "observable_ids": [plan.observable_id],
        "validity_envelope": certificate["tested_envelope"],
        "fallback": "retain_high_fidelity_contact_or_request_recertification",
        "invalidation_rules": [
            "package_binding_sha256_changed",
            "source_or_collision_geometry_hash_changed",
            "load_or_mass_outside_validity_envelope",
            "required_dependency_certificate_invalid",
            "compound_family_spec_or_topology_changed",
        ],
    }


def write_certificate_artifacts(
    plan: MechanismPlan,
    load_cases: tuple[LoadCase, ...],
    comparisons: tuple[dict[str, Any], ...],
    *,
    certificate_path: str | Path,
    replacement_path: str | Path,
    trace_files: Iterable[str | Path] = (),
    workspace_root: str | Path | None = None,
    task_acceptance: dict[str, Any] | None = None,
    ideal_reference_comparisons: tuple[dict[str, Any], ...] = (),
    physical_trace_audits: tuple[dict[str, Any], ...] = (),
    replacement_authorized: bool = True,
    replacement_policy: str = "certified_mechanism",
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    certificate = build_certificate(
        plan,
        load_cases,
        comparisons,
        trace_files=trace_files,
        workspace_root=workspace_root,
        task_acceptance=task_acceptance,
        ideal_reference_comparisons=ideal_reference_comparisons,
        physical_trace_audits=physical_trace_audits,
        replacement_authorized=replacement_authorized,
        replacement_policy=replacement_policy,
    )
    write_json_atomic(certificate_path, certificate)
    replacement: dict[str, Any] | None = None
    if certificate["replacement_eligible"]:
        replacement = build_replacement(
            plan,
            certificate,
            certificate_sha256=sha256_file(certificate_path),
        )
        write_json_atomic(replacement_path, replacement)
    elif Path(replacement_path).exists() or not replacement_authorized:
        write_json_atomic(
            replacement_path,
            {
                "schema_version": "0.1",
                "status": "INACTIVE",
                "package_binding_sha256": plan.package_binding_sha256,
                "capsule_id": plan.capsule_id,
                "reason": (
                    "single-mechanism package keeps reduced branch as reference only"
                    if not replacement_authorized
                    else "current certificate does not authorize replacement"
                ),
                "replacement_policy": replacement_policy,
                "certificate_verdict": certificate["verdict"],
                "certificate_sha256": sha256_file(certificate_path),
            },
        )
    return certificate, replacement
