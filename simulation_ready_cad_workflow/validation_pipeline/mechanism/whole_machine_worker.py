from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    failure_envelope_from_exception,
    make_failure_envelope,
)
from .io import (
    MECHANISM_SOURCE_SHA256_ENV,
    canonical_sha256,
    ensure_inside,
    mechanism_source_sha256,
    read_json,
    read_trace_json,
    sha256_file,
    write_json_atomic,
)
from .models import LoadCase, LoadPhase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error
from .task_motion_evidence import (
    canonical_registered_stop_coordinate,
    registered_stop_comparator_reached,
)
from .wrench_ledger import (
    activate_wrench_ledger,
    add_ledgered_load,
    create_body_force,
    create_body_torque,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
    verify_saved_wrench_audit,
)
from .rack_pinion_whole_machine_v4 import (
    WholeMachineRackPinionExecutionPlan,
    simulate_whole_machine_rack_pinion,
)
from .bicycle_chain_whole_machine import (
    WholeMachineBicycleChainExecutionPlan,
    simulate_whole_machine_bicycle_chain,
)
from .bicycle_ground_roll_whole_machine import (
    WholeMachineBicycleGroundRollExecutionPlan,
    simulate_whole_machine_bicycle_ground_roll,
    summarize_whole_machine_bicycle_ground_roll,
)
# [kimi-scissor-lift]
from .scissor_lift_whole_machine import (
    WholeMachineScissorLiftExecutionPlan,
    simulate_whole_machine_scissor_lift,
)
from .whole_machine import (
    WholeMachineClockExecutionPlan,
    WholeMachineCompoundExecutionPlan,
    WholeMachineExecutionPlan,
    WholeMachineRevoluteExecutionPlan,
)
from .worker import (
    _configure_prismatic_travel_limit,
    _configure_revolute_angle_limit,
)

_WHOLE_MACHINE_WRENCH_AUDIT_ENTRY_SCHEMA = (
    "simulation_ready_cad_whole_machine_wrench_audit_entry_v1"
)
_WHOLE_MACHINE_WRENCH_AUDIT_SET_SCHEMA = (
    "simulation_ready_cad_whole_machine_wrench_audit_set_v1"
)
_EXECUTED_PHASE_BASE_KEYS = frozenset(
    {
        "phase_id",
        "wrench_kind",
        "direction_abs",
        "magnitude_si",
        "duration_s",
    }
)


def _whole_worker_harness_error(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> MechanismValidationError:
    return MechanismValidationError(
        code,
        message,
        details=details,
        failure=make_failure_envelope(
            code,
            FailureOwner.HARNESS_CODE,
            source_component="mechanism.whole_machine_worker.evidence",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            details=details,
        ),
    )


_CLOCK_EXECUTED_PHASE_EXTENSION_KEYS = frozenset(
    {
        "active_input_body_id",
        "active_input_point_abs_m",
        "termination_reason",
        "setting_clutch_engaged",
        "actuation_channels",
    }
)
_WHOLE_MACHINE_EXECUTED_PHASE_EXTENSION_KEYS = {
    "plain_prismatic_v1": frozenset(),
    "plain_revolute_v1": frozenset(),
    "revolute_linkage_v1": frozenset(),
    "rack_pinion_v1": frozenset(),
    "bicycle_chain_sprocket_drive_v2": frozenset(),
    "bicycle_ground_roll_v1": frozenset(),
    "hydraulic_scissor_lift_v1": frozenset(),
    "spur_gear_clock_v1": _CLOCK_EXECUTED_PHASE_EXTENSION_KEYS,
    "multi_output_shaft_coupling_v1": (
        _CLOCK_EXECUTED_PHASE_EXTENSION_KEYS
    ),
}
_WHOLE_MACHINE_WRENCH_CONTEXT_KEYS = frozenset(
    {
        "plugin_id",
        "branch",
        "load_case_id",
        "program",
        "phase_ids",
        "load_case",
        "load_case_sha256",
        "program_phases",
        "program_phases_sha256",
        "phase_count",
        "input_body_ids",
        "output_body_ids",
        "input_body_count",
        "output_body_count",
        "capability_registration",
        "wrench_policy_inventory",
        "wrench_policy_inventory_sha256",
        "task_channel_registry",
        "task_channel_registry_sha256",
        "context_sha256",
    }
)
_WHOLE_MACHINE_WRENCH_ENTRY_KEYS = frozenset(
    {
        "schema_version",
        *_WHOLE_MACHINE_WRENCH_CONTEXT_KEYS,
        "executed_phases",
        "executed_load_sha256",
        "wrench_audit",
        "entry_sha256",
    }
)


def _wrench_context_for_execution(execution: Any) -> dict[str, Any]:
    """Derive the exact original-plugin audit context for one load case."""

    from .worker import _wrench_ledger_for_execution

    ledger = _wrench_ledger_for_execution(execution)
    plugin_id = execution.mechanism_plugin.plugin_id
    load_case = execution.load_case
    if (
        ledger.plugin_id != plugin_id
        or ledger.load_case_id != load_case.load_case_id
        or ledger.branch != execution.artifact_branch
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_CONTEXT_MISMATCH",
            "whole-machine load case is not bound to its original plugin ledger",
        )
    from .family_plugins import (
        plugin_registration_for,
        wrench_policies_for_plugin,
    )

    registration = plugin_registration_for(plugin_id)
    if (
        registration.get("status") != "ACTIVE"
        or ledger.capability_registration != registration
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_CONTEXT_MISMATCH",
            "whole-machine load case does not use the current ACTIVE plugin",
        )
    policies = [
        item.to_dict()
        for item in sorted(
            wrench_policies_for_plugin(plugin_id),
            key=lambda item: item.policy_id,
        )
    ]
    load_case_payload = load_case.to_dict()
    program_phases = [
        phase.to_dict() for phase in load_case.phases
    ]
    raw_registry = getattr(execution.plan, "task_channel_registry", None)
    channel_registry = None
    channel_registry_sha256 = None
    if raw_registry is not None:
        from .task_motion_evidence import task_channel_registry_for_plan

        channel_registry = task_channel_registry_for_plan(
            execution.plan
        ).to_dict()
        channel_registry_sha256 = channel_registry[
            "channel_registry_sha256"
        ]
    context = {
        "plugin_id": plugin_id,
        "branch": execution.artifact_branch,
        "load_case_id": load_case.load_case_id,
        "program": load_case.program,
        "phase_ids": [phase.phase_id for phase in load_case.phases],
        "load_case": load_case_payload,
        "load_case_sha256": canonical_sha256(load_case_payload),
        "program_phases": program_phases,
        "program_phases_sha256": canonical_sha256(program_phases),
        "phase_count": len(program_phases),
        "input_body_ids": sorted(ledger.declared_input_body_ids),
        "output_body_ids": sorted(ledger.declared_output_body_ids),
        "input_body_count": len(ledger.declared_input_body_ids),
        "output_body_count": len(ledger.declared_output_body_ids),
        "capability_registration": registration,
        "wrench_policy_inventory": policies,
        "wrench_policy_inventory_sha256": canonical_sha256(policies),
        "task_channel_registry": channel_registry,
        "task_channel_registry_sha256": channel_registry_sha256,
    }
    context["context_sha256"] = canonical_sha256(context)
    if set(context) != _WHOLE_MACHINE_WRENCH_CONTEXT_KEYS:
        raise AssertionError("whole-machine wrench context schema drift")
    return context


def _validate_executed_phase_records(
    execution: Any,
    context: dict[str, Any],
    value: Any,
    digest: Any,
) -> list[dict[str, Any]]:
    """Validate the closed runtime phase schema for an ACTIVE backend."""

    plugin_id = str(context["plugin_id"])
    extension_keys = _WHOLE_MACHINE_EXECUTED_PHASE_EXTENSION_KEYS.get(
        plugin_id
    )
    if extension_keys is None:
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_PHASE_SCHEMA_UNREGISTERED",
            "whole-machine execution has no registered closed phase schema",
        )
    if not isinstance(value, list) or not all(
        isinstance(item, dict) for item in value
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
            "whole-machine load case must expose an object inventory "
            "of executed phases",
        )
    if not isinstance(digest, str) or not digest:
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
            "whole-machine load case must expose its executed schedule "
            "digest",
        )
    if digest != canonical_sha256(value):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_SCHEDULE_MISMATCH",
            "whole-machine executed schedule digest does not match its records",
        )
    expected_phases = tuple(execution.load_case.phases)
    if len(value) != len(expected_phases):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_SCHEDULE_MISMATCH",
            "whole-machine executed schedule does not cover every phase",
        )
    declared_inputs = set(context["input_body_ids"])
    normalized: list[dict[str, Any]] = []
    for expected, raw in zip(expected_phases, value):
        allowed_keys = set(_EXECUTED_PHASE_BASE_KEYS | extension_keys)
        if expected.end_magnitude_si is not None:
            allowed_keys.add("end_magnitude_si")
        if expected.stop_condition is not None:
            allowed_keys.add("termination")
        if set(raw) != allowed_keys:
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_PHASE_RECORD_INVALID",
                "whole-machine phase record is not the exact registered schema",
            )
        expected_direction = (
            None
            if expected.direction_abs is None
            else list(expected.direction_abs)
        )
        if (
            raw.get("phase_id") != expected.phase_id
            or raw.get("wrench_kind") != expected.wrench_kind
            or raw.get("direction_abs") != expected_direction
            or isinstance(raw.get("magnitude_si"), bool)
            or not isinstance(raw.get("magnitude_si"), (int, float))
            or not math.isfinite(float(raw["magnitude_si"]))
            or float(raw["magnitude_si"]) != float(expected.magnitude_si)
            or (
                expected.end_magnitude_si is not None
                and (
                    isinstance(raw.get("end_magnitude_si"), bool)
                    or not isinstance(
                        raw.get("end_magnitude_si"),
                        (int, float),
                    )
                    or not math.isfinite(
                        float(raw["end_magnitude_si"])
                    )
                    or float(raw["end_magnitude_si"])
                    != float(expected.end_magnitude_si)
                )
            )
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
                "whole-machine phase record changes its validator load",
            )
        duration = raw.get("duration_s")
        adaptive = expected.stop_condition is not None
        # Fixed phase duration is an immutable program field.  Permit only
        # machine-scale floating-point drift, never a whole solver step.
        duration_tolerance = max(
            1.0e-12,
            abs(float(expected.duration_s)) * 1.0e-12,
        )
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or float(duration) <= 0.0
            or float(duration)
            > float(expected.duration_s) + duration_tolerance
            or (
                not adaptive
                and abs(float(duration) - float(expected.duration_s))
                > duration_tolerance
            )
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
                "whole-machine phase duration is outside its validator limit",
            )
        if adaptive:
            termination = raw.get("termination")
            expected_termination_keys = {
                "schema_version",
                "stop_condition",
                "target_coordinate_si",
                "actual_coordinate_si",
                "reason",
                "branch_termination_policy",
            }
            actual_coordinate = (
                termination.get("actual_coordinate_si")
                if isinstance(termination, dict)
                else None
            )
            if (
                not isinstance(termination, dict)
                or set(termination) != expected_termination_keys
                or termination.get("schema_version")
                != "registered_adaptive_stop_v1"
                or termination.get("stop_condition")
                != expected.stop_condition
                or termination.get("target_coordinate_si")
                != expected.target_coordinate
                or termination.get("branch_termination_policy")
                != expected.branch_termination_policy
                or termination.get("reason")
                not in {
                    "registered_stop_condition_reached",
                    "duration_limit",
                }
                or isinstance(actual_coordinate, bool)
                or not isinstance(actual_coordinate, (int, float))
                or not math.isfinite(float(actual_coordinate))
            ):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_PHASE_RECORD_INVALID",
                    "adaptive whole-machine phase lacks exact registered "
                    "termination evidence",
                )
            shortened = (
                float(duration)
                < float(expected.duration_s) - duration_tolerance
            )
            target = float(expected.target_coordinate)
            actual = float(actual_coordinate)
            comparator_reached = (
                expected.stop_condition
                in {"position_at_least", "progress_at_least"}
                and actual >= target
            ) or (
                expected.stop_condition
                in {"position_at_most", "progress_at_most"}
                and actual <= target
            )
            if shortened and (
                termination["reason"]
                != "registered_stop_condition_reached"
                or not comparator_reached
            ):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
                    "shortened adaptive phase is not justified by its "
                    "registered stop comparator",
                )
        if extension_keys:
            active_body = raw.get("active_input_body_id")
            active_point = raw.get("active_input_point_abs_m")
            channels = raw.get("actuation_channels")
            if (
                not isinstance(active_body, str)
                or active_body not in declared_inputs
                or not isinstance(active_point, list)
                or len(active_point) != 3
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    for item in active_point
                )
                or not isinstance(raw.get("termination_reason"), str)
                or not raw["termination_reason"]
                or not isinstance(raw.get("setting_clutch_engaged"), bool)
                or not isinstance(channels, list)
                or not 1 <= len(channels) <= 2
                or not all(isinstance(item, dict) for item in channels)
            ):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_PHASE_RECORD_INVALID",
                    "clock phase record has invalid runtime channel evidence",
                )
            channel_bodies: list[str] = []
            for channel in channels:
                magnitude = channel.get("magnitude_si")
                if (
                    set(channel)
                    != {"body_id", "kind", "magnitude_si"}
                    or channel.get("body_id") not in declared_inputs
                    or channel.get("kind") not in {"none", "force", "torque"}
                    or isinstance(magnitude, bool)
                    or not isinstance(magnitude, (int, float))
                    or not math.isfinite(float(magnitude))
                    or float(magnitude) < 0.0
                ):
                    raise MechanismValidationError(
                        "WHOLE_MACHINE_WRENCH_PHASE_RECORD_INVALID",
                        "clock actuation channel is outside its closed schema",
                    )
                channel_bodies.append(str(channel["body_id"]))
            if (
                len(set(channel_bodies)) != len(channel_bodies)
                or channels[0]
                != {
                    "body_id": active_body,
                    "kind": expected.wrench_kind,
                    "magnitude_si": expected.magnitude_si,
                }
            ):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
                    "clock phase actuation channels change the selected input",
                )
        normalized.append(dict(raw))
    return normalized


def _run_registered_whole_machine_load_case(
    execution: Any,
    run_simulation: Any,
    *,
    executed_phases: list[dict[str, Any]] | None = None,
    executed_load_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one replay under a fresh ledger owned by its ACTIVE plugin."""

    from .worker import _wrench_ledger_for_execution

    context = _wrench_context_for_execution(execution)
    ledger = _wrench_ledger_for_execution(execution)
    with activate_wrench_ledger(ledger):
        try:
            trace = run_simulation()
            if not isinstance(trace, dict):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_TRACE_INVALID",
                    "whole-machine load-case simulator did not return an object",
                )
            actual_phases = (
                trace.get("executed_phases")
                if executed_phases is None
                else executed_phases
            )
            actual_hash = (
                trace.get("executed_load_sha256")
                if executed_phases is None
                else executed_load_sha256
            )
            actual_phases = _validate_executed_phase_records(
                execution,
                context,
                actual_phases,
                actual_hash,
            )
            ledger.bind_executed_schedule(
                actual_phases,
                executed_load_sha256=actual_hash,
            )
            audit = ledger.verify_for_certification()
        finally:
            set_current_wrench_phase(None)
    verified = verify_saved_wrench_audit(
        audit,
        expected_plugin_id=context["plugin_id"],
        expected_branch=context["branch"],
        expected_load_case_id=context["load_case_id"],
        expected_phase_ids=context["phase_ids"],
        expected_program_phases=context["program_phases"],
        expected_executed_load_sha256=audit["executed_load_sha256"],
        expected_input_body_ids=context["input_body_ids"],
        expected_output_body_ids=context["output_body_ids"],
    )
    trace["wrench_audit"] = verified
    entry = {
        "schema_version": _WHOLE_MACHINE_WRENCH_AUDIT_ENTRY_SCHEMA,
        **{
            key: value
            for key, value in context.items()
            if key != "context_sha256"
        },
        "context_sha256": context["context_sha256"],
        "executed_phases": [dict(item) for item in actual_phases],
        "executed_load_sha256": verified["executed_load_sha256"],
        "wrench_audit": verified,
    }
    entry["entry_sha256"] = canonical_sha256(entry)
    if set(entry) != _WHOLE_MACHINE_WRENCH_ENTRY_KEYS:
        raise AssertionError("whole-machine wrench audit entry schema drift")
    return trace, entry


def _validate_whole_machine_wrench_audits(
    expected_contexts: Sequence[dict[str, Any]],
    entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Revalidate an exact, non-replayable audit set for a whole-machine run."""

    expected_by_id = {
        str(item.get("load_case_id") or ""): dict(item)
        for item in expected_contexts
    }
    if (
        len(expected_by_id) != len(expected_contexts)
        or "" in expected_by_id
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_CONTEXT_INVALID",
            "expected whole-machine load-case identities are not unique",
        )
    for expected in expected_contexts:
        from .family_plugins import (
            plugin_registration_for,
            wrench_policies_for_plugin,
        )

        current_registration = plugin_registration_for(
            str(expected.get("plugin_id") or "")
        )
        current_policies = [
            item.to_dict()
            for item in sorted(
                wrench_policies_for_plugin(
                    str(expected.get("plugin_id") or "")
                ),
                key=lambda item: item.policy_id,
            )
        ]
        if (
            set(expected) != _WHOLE_MACHINE_WRENCH_CONTEXT_KEYS
            or expected.get("context_sha256")
            != canonical_sha256(
                {
                    key: value
                    for key, value in expected.items()
                    if key != "context_sha256"
                }
            )
            or expected.get("capability_registration")
            != current_registration
            or current_registration.get("status") != "ACTIVE"
            or expected.get("wrench_policy_inventory")
            != current_policies
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_CONTEXT_INVALID",
                "expected whole-machine context is open, incomplete, or "
                "not hash-bound",
            )
    if not isinstance(entries, (list, tuple)) or not all(
        isinstance(item, dict) for item in entries
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_AUDITS_INVALID",
            "whole-machine wrench audits must be an object inventory",
        )
    entry_ids = [str(item.get("load_case_id") or "") for item in entries]
    if (
        len(set(entry_ids)) != len(entry_ids)
        or set(entry_ids) != set(expected_by_id)
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_AUDIT_SET_MISMATCH",
            "whole-machine wrench audits are missing, duplicated, or unexpected",
        )
    verified_entries: list[dict[str, Any]] = []
    for raw in entries:
        if set(raw) != _WHOLE_MACHINE_WRENCH_ENTRY_KEYS:
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_AUDIT_ENTRY_INVALID",
                "whole-machine wrench audit entry is not the exact schema",
            )
        entry = dict(raw)
        entry_digest = entry.pop("entry_sha256", None)
        if (
            raw.get("schema_version")
            != _WHOLE_MACHINE_WRENCH_AUDIT_ENTRY_SCHEMA
            or entry_digest != canonical_sha256(entry)
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_AUDIT_ENTRY_INVALID",
                "whole-machine wrench audit entry is changed or unsupported",
            )
        load_case_id = str(raw["load_case_id"])
        expected = expected_by_id[load_case_id]
        for field in _WHOLE_MACHINE_WRENCH_CONTEXT_KEYS:
            if raw.get(field) != expected.get(field):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_CONTEXT_MISMATCH",
                    f"whole-machine wrench audit changes {field}",
                )
        schedule = raw.get("executed_phases")
        if (
            not isinstance(schedule, list)
            or not all(isinstance(item, dict) for item in schedule)
            or [
                str(item.get("phase_id") or "")
                for item in schedule
            ]
            != expected["phase_ids"]
            or raw.get("executed_load_sha256")
            != canonical_sha256(schedule)
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_SCHEDULE_MISMATCH",
                "whole-machine wrench audit is not bound to its executed schedule",
            )
        from types import SimpleNamespace

        from .models import LoadCase

        try:
            bound_load_case = LoadCase.from_dict(
                dict(expected["load_case"])
            )
        except Exception as exc:
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_CONTEXT_INVALID",
                f"bound whole-machine load case is invalid: {exc}",
            ) from exc
        if (
            bound_load_case.to_dict() != expected["load_case"]
            or [phase.to_dict() for phase in bound_load_case.phases]
            != expected["program_phases"]
            or expected["load_case_sha256"]
            != canonical_sha256(expected["load_case"])
            or expected["program_phases_sha256"]
            != canonical_sha256(expected["program_phases"])
            or expected["phase_count"] != len(bound_load_case.phases)
            or expected["input_body_count"]
            != len(expected["input_body_ids"])
            or expected["output_body_count"]
            != len(expected["output_body_ids"])
            or expected["wrench_policy_inventory_sha256"]
            != canonical_sha256(expected["wrench_policy_inventory"])
            or (
                expected["task_channel_registry"] is None
            )
            != (
                expected["task_channel_registry_sha256"] is None
            )
            or (
                expected["task_channel_registry"] is not None
                and expected["task_channel_registry_sha256"]
                != expected["task_channel_registry"].get(
                    "channel_registry_sha256"
                )
            )
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_CONTEXT_INVALID",
                "whole-machine context counts, policies, or bindings are "
                "internally inconsistent",
            )
        _validate_executed_phase_records(
            SimpleNamespace(
                load_case=bound_load_case,
                mechanism_plugin=SimpleNamespace(
                    plugin_id=expected["plugin_id"]
                ),
            ),
            expected,
            schedule,
            raw.get("executed_load_sha256"),
        )
        verify_saved_wrench_audit(
            raw.get("wrench_audit"),
            expected_plugin_id=expected["plugin_id"],
            expected_branch=expected["branch"],
            expected_load_case_id=load_case_id,
            expected_phase_ids=expected["phase_ids"],
            expected_program_phases=expected["program_phases"],
            expected_executed_load_sha256=raw["executed_load_sha256"],
            expected_input_body_ids=expected["input_body_ids"],
            expected_output_body_ids=expected["output_body_ids"],
        )
        verified_entries.append(dict(raw))
    summary = {
        "schema_version": _WHOLE_MACHINE_WRENCH_AUDIT_SET_SCHEMA,
        "passed": True,
        "audit_count": len(verified_entries),
        "load_case_ids": sorted(entry_ids),
        "plugin_ids": sorted(
            {str(item["plugin_id"]) for item in verified_entries}
        ),
        "entry_sha256s": sorted(
            str(item["entry_sha256"]) for item in verified_entries
        ),
    }
    summary["audit_set_sha256"] = canonical_sha256(summary)
    return summary


def _attach_whole_machine_wrench_audits(
    payload: dict[str, Any],
    *,
    expected_contexts: Sequence[dict[str, Any]],
    entries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summary = _validate_whole_machine_wrench_audits(
        expected_contexts,
        entries,
    )
    payload["wrench_audits"] = [dict(item) for item in entries]
    payload["wrench_audit_summary"] = summary
    return payload


def _verify_validator_source_binding() -> str:
    expected = os.environ.get(MECHANISM_SOURCE_SHA256_ENV)
    actual = mechanism_source_sha256()
    if expected is None:
        raise MechanismValidationError(
            "VALIDATOR_SOURCE_BINDING_MISSING",
            "whole-machine worker requires a source-bound queue",
        )
    if expected != actual:
        raise MechanismValidationError(
            "VALIDATOR_SOURCE_CHANGED",
            "validator source changed after whole-machine planning",
        )
    return actual

def _quat_from_z_to_axis(chrono: Any, axis: tuple[float, float, float]) -> Any:
    dot = max(-1.0, min(1.0, axis[2]))
    if dot < -1.0 + 1e-12:
        return chrono.ChQuaterniond(0.0, 1.0, 0.0, 0.0)
    cross = (-axis[1], axis[0], 0.0)
    scale = math.sqrt(2.0 * (1.0 + dot))
    return chrono.ChQuaterniond(
        0.5 * scale,
        cross[0] / scale,
        cross[1] / scale,
        cross[2] / scale,
    )


def _inertia_box(
    mass_kg: float, extents_m: tuple[float, float, float]
) -> tuple[float, float, float]:
    x_value, y_value, z_value = extents_m
    return (
        mass_kg * (y_value**2 + z_value**2) / 12.0,
        mass_kg * (x_value**2 + z_value**2) / 12.0,
        mass_kg * (x_value**2 + y_value**2) / 12.0,
    )


def _plain_whole_machine_load_case(
    plan: WholeMachineExecutionPlan | WholeMachineRevoluteExecutionPlan,
) -> LoadCase:
    axis = tuple(float(item) for item in plan.axis_abs)

    def directed(sign: float) -> tuple[float, float, float]:
        return tuple(sign * item for item in axis)

    if isinstance(plan, WholeMachineRevoluteExecutionPlan):
        outbound_sign = (
            1.0
            if plan.outbound_angle_rad >= plan.home_angle_rad
            else -1.0
        )
        return_sign = -outbound_sign
        outbound_stop = (
            "position_at_least"
            if outbound_sign > 0.0
            else "position_at_most"
        )
        return_stop = (
            "position_at_least"
            if return_sign > 0.0
            else "position_at_most"
        )
        return_phase = (
            LoadPhase(
                phase_id="return",
                wrench_kind="none",
                direction_abs=None,
                magnitude_si=0.0,
                duration_s=plan.max_stage_duration_s,
            )
            if plan.passive_return_required
            else LoadPhase(
                phase_id="return",
                wrench_kind="torque",
                direction_abs=directed(return_sign),
                magnitude_si=plan.max_control_torque_nm,
                duration_s=plan.max_stage_duration_s,
                stop_condition=return_stop,
                target_position_si=plan.home_angle_rad,
                branch_termination_policy="each_branch_target",
            )
        )
        phases = (
            LoadPhase(
                phase_id="outbound",
                wrench_kind="torque",
                direction_abs=directed(outbound_sign),
                magnitude_si=plan.max_control_torque_nm,
                duration_s=plan.max_stage_duration_s,
                stop_condition=outbound_stop,
                target_position_si=plan.outbound_angle_rad,
                branch_termination_policy="each_branch_target",
            ),
            return_phase,
        )
        family = "plain_revolute"
        program = "task_angle_roundtrip"
    else:
        outbound_sign = (
            1.0
            if plan.outbound_progress_m >= plan.home_progress_m
            else -1.0
        )
        return_sign = -outbound_sign
        phases = (
            LoadPhase(
                phase_id="outbound",
                wrench_kind="force",
                direction_abs=directed(outbound_sign),
                magnitude_si=plan.max_control_force_n,
                duration_s=plan.max_stage_duration_s,
                stop_condition=(
                    "progress_at_least"
                    if outbound_sign > 0.0
                    else "progress_at_most"
                ),
                target_progress_m=plan.outbound_progress_m,
                branch_termination_policy="each_branch_target",
            ),
            LoadPhase(
                phase_id="return",
                wrench_kind="force",
                direction_abs=directed(return_sign),
                magnitude_si=plan.max_control_force_n,
                duration_s=plan.max_stage_duration_s,
                stop_condition=(
                    "progress_at_least"
                    if return_sign > 0.0
                    else "progress_at_most"
                ),
                target_progress_m=plan.home_progress_m,
                branch_termination_policy="each_branch_target",
            ),
        )
        family = "plain_prismatic"
        program = "task_range_roundtrip"
    return LoadCase(
        load_case_id="whole-machine-task",
        family=family,
        program=program,
        purpose="nominal",
        phases=phases,
        moving_mass_kg=plan.moving_mass_kg,
        step_s=plan.step_s,
        sample_interval_s=plan.sample_interval_s,
    )


def _plain_whole_machine_execution(
    plan: WholeMachineExecutionPlan | WholeMachineRevoluteExecutionPlan,
) -> SimpleNamespace:
    from .family_plugins import registered_mechanism_adapters

    plugin = next(
        (
            item
            for item in registered_mechanism_adapters()
            if item.plugin_id == plan.mechanism_plugin_id
        ),
        None,
    )
    if plugin is None or plugin.worker_backend != plan.worker_backend:
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_PLUGIN_MISMATCH",
            "whole-machine plan names an unregistered mechanism plugin",
        )
    return SimpleNamespace(
        mechanism_plugin=plugin,
        load_case=_plain_whole_machine_load_case(plan),
        artifact_branch="certified_reduced_whole_machine",
        plan=SimpleNamespace(task_channel_registry=None),
        adapter=SimpleNamespace(family_spec={}),
    )


def _plain_whole_machine_executed_schedule(
    load_case: LoadCase,
    stages: Any,
) -> list[dict[str, Any]]:
    if (
        not isinstance(stages, list)
        or len(stages) != len(load_case.phases)
        or not all(isinstance(item, dict) for item in stages)
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
            "plain whole-machine runtime lacks its exact stage inventory",
        )
    schedule: list[dict[str, Any]] = []
    for phase, stage in zip(load_case.phases, stages):
        if stage.get("stage") != phase.phase_id:
            raise MechanismValidationError(
                "WHOLE_MACHINE_WRENCH_SCHEDULE_MISMATCH",
                "plain whole-machine stage order differs from its load case",
            )
        record: dict[str, Any] = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None
                if phase.direction_abs is None
                else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
            "duration_s": stage.get("duration_s"),
        }
        if phase.stop_condition is not None:
            actual_coordinate = stage.get("final_coordinate_si")
            reached = stage.get("reached")
            if (
                isinstance(actual_coordinate, bool)
                or not isinstance(actual_coordinate, (int, float))
                or not math.isfinite(float(actual_coordinate))
                or not isinstance(reached, bool)
            ):
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
                    "plain whole-machine adaptive stage lacks final state",
                )
            canonical_coordinate = canonical_registered_stop_coordinate(
                phase.stop_condition,
                target_coordinate_si=float(phase.target_coordinate),
                actual_coordinate_si=float(actual_coordinate),
            )
            comparator_reached = registered_stop_comparator_reached(
                phase.stop_condition,
                target_coordinate_si=float(phase.target_coordinate),
                actual_coordinate_si=canonical_coordinate,
            )
            if reached and not comparator_reached:
                raise MechanismValidationError(
                    "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
                    "plain whole-machine stage claims a registered stop "
                    "without satisfying its exact comparator",
                )
            record["termination"] = {
                "schema_version": "registered_adaptive_stop_v1",
                "stop_condition": phase.stop_condition,
                "target_coordinate_si": phase.target_coordinate,
                "actual_coordinate_si": canonical_coordinate,
                "reason": (
                    "registered_stop_condition_reached"
                    if reached
                    else "duration_limit"
                ),
                "branch_termination_policy": (
                    phase.branch_termination_policy
                ),
            }
        schedule.append(record)
    return schedule


def _snapshot_vector3(vector: Any) -> tuple[float, float, float]:
    return float(vector.x), float(vector.y), float(vector.z)


def _quat_angle(initial: tuple[float, ...], current: tuple[float, ...]) -> float:
    dot = abs(sum(left * right for left, right in zip(initial, current)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _axis_spring_revolute_control(
    *,
    angle_rad: float,
    speed_rad_s: float,
    target_rad: float,
    max_control_torque_nm: float,
    controller_kp_nm_rad: float,
    controller_kd_nm_s_rad: float,
    spring_stiffness_nm_rad: float,
    spring_damping_nm_s_rad: float,
    passive_stage: bool,
) -> tuple[float, float]:
    spring_torque = (
        -spring_stiffness_nm_rad * angle_rad
        - spring_damping_nm_s_rad * speed_rad_s
    )
    if passive_stage:
        return 0.0, spring_torque
    error = target_rad - angle_rad
    requested_control = (
        controller_kp_nm_rad * error
        - controller_kd_nm_s_rad * speed_rad_s
        - spring_torque
    )
    control_torque = max(
        -max_control_torque_nm,
        min(max_control_torque_nm, requested_control),
    )
    return control_torque, spring_torque


def _load_plan(path: Path, workspace: Path, plugin: Any) -> Any:
    value = read_json(path)
    from .family_plugins import plugin_registration_or_none

    registration = plugin_registration_or_none(str(plugin.plugin_id))
    try:
        plan = plugin.load_whole_machine_plan(value)
    except Exception as exc:
        boundary_error = plugin_boundary_error(
            exc,
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="load_whole_machine_plan",
            source_component="mechanism.whole_machine_worker.load_plan",
            code=(
                None
                if isinstance(exc, MechanismValidationError)
                else "MECHANISM_ADAPTER_WHOLE_MACHINE_PLAN_LOAD_FAILED"
            ),
            message=(
                None
                if isinstance(exc, MechanismValidationError)
                else f"mechanism plugin load_whole_machine_plan() failed: {exc}"
            ),
            contract_violation=True,
        )
        if boundary_error is exc:
            raise
        raise boundary_error from exc
    if plan is None:
        message = (
            f"mechanism plugin {plugin.plugin_id!r} does not claim "
            f"coordinate_kind {value.get('coordinate_kind')!r}"
        )
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="load_whole_machine_plan",
            source_component="mechanism.whole_machine_worker.load_plan",
            code="WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
            message=message,
            contract_violation=True,
        )
    try:
        certificate = ensure_inside(workspace, plan.certificate_file)
        replacement = ensure_inside(workspace, plan.replacement_file)
        certificate_sha256 = sha256_file(certificate)
        replacement_sha256 = sha256_file(replacement)
    except (OSError, TypeError, ValueError) as exc:
        raise _whole_worker_harness_error(
            "WHOLE_MACHINE_EVIDENCE_INVALID",
            f"whole-machine evidence cannot be resolved: {exc}",
            details={"capsule_id": plan.capsule_id},
        ) from exc
    if certificate_sha256 != plan.certificate_sha256:
        raise _whole_worker_harness_error(
            "WHOLE_MACHINE_CERTIFICATE_CHANGED",
            "certificate changed after whole-machine dispatch",
            details={"capsule_id": plan.capsule_id},
        )
    if replacement_sha256 != plan.replacement_sha256:
        raise _whole_worker_harness_error(
            "WHOLE_MACHINE_REPLACEMENT_CHANGED",
            "replacement changed after whole-machine dispatch",
            details={"capsule_id": plan.capsule_id},
        )
    reference_bindings: list[tuple[str, str]] = []
    single_reference = getattr(plan, "reference_trace_file", None)
    if isinstance(single_reference, str) and single_reference:
        reference_bindings.append(
            (single_reference, str(getattr(plan, "reference_trace_sha256", "")))
        )
    for reference_spec in getattr(plan, "reference_traces", ()) or ():
        reference_bindings.append(
            (str(reference_spec["file"]), str(reference_spec["sha256"]))
        )
    for reference_file, reference_sha256 in reference_bindings:
        try:
            reference = ensure_inside(workspace, reference_file)
            actual_reference_sha256 = sha256_file(reference)
        except (OSError, TypeError, ValueError) as exc:
            raise _whole_worker_harness_error(
                "WHOLE_MACHINE_REFERENCE_TRACE_INVALID",
                f"certified physical task trace cannot be resolved: {exc}",
                details={"capsule_id": plan.capsule_id},
            ) from exc
        if actual_reference_sha256 != reference_sha256:
            raise _whole_worker_harness_error(
                "WHOLE_MACHINE_REFERENCE_TRACE_CHANGED",
                "certified physical task trace changed after whole-machine dispatch",
                details={"capsule_id": plan.capsule_id},
            )
    try:
        certificate_document = read_json(certificate)
        replacement_document = read_json(replacement)
    except (OSError, TypeError, ValueError) as exc:
        raise _whole_worker_harness_error(
            "WHOLE_MACHINE_EVIDENCE_INVALID",
            f"whole-machine evidence cannot be read: {exc}",
            details={"capsule_id": plan.capsule_id},
        ) from exc
    if (
        certificate_document.get("verdict") != "KINEMATIC_CERTIFIED"
        or replacement_document.get("status") != "ACTIVE"
        or replacement_document.get("package_binding_sha256")
        != plan.package_binding_sha256
    ):
        raise _whole_worker_harness_error(
            "WHOLE_MACHINE_AUTHORIZATION_INVALID",
            "certificate or replacement authorization is no longer valid",
            details={"capsule_id": plan.capsule_id},
        )
    return plan


def _expected_reduced_ideal_links(
    plugin: Any,
    mechanism_plan: MechanismPlan,
    adapter: Any,
) -> tuple[str, ...]:
    from .family_plugins import plugin_registration_or_none

    registration = plugin_registration_or_none(str(plugin.plugin_id))
    links = invoke_plugin_callback(
        lambda: plugin.expected_reduced_ideal_links(mechanism_plan, adapter),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="expected_reduced_ideal_links",
        source_component=(
            "mechanism.whole_machine_worker.expected_reduced_ideal_links"
        ),
        unexpected_code="MECHANISM_ADAPTER_REDUCED_IDEAL_LINKS_FAILED",
        details={"capsule_id": mechanism_plan.capsule_id},
    )
    if not isinstance(links, tuple) or any(
        not isinstance(item, str) or not item for item in links
    ):
        message = (
            "mechanism adapter expected_reduced_ideal_links() must return "
            "a tuple of identifiers"
        )
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="expected_reduced_ideal_links",
            source_component=(
                "mechanism.whole_machine_worker.expected_reduced_ideal_links"
            ),
            code="MECHANISM_ADAPTER_REDUCED_IDEAL_LINKS_INVALID",
            message=message,
            details={"capsule_id": mechanism_plan.capsule_id},
            contract_violation=True,
        )
    return links


def _simulate_prismatic(plan: WholeMachineExecutionPlan) -> dict[str, Any]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(*plan.gravity_abs_m_s2))
    fixed = chrono.ChBody()
    fixed.SetName("certified_machine_frame")
    fixed.SetFixed(True)
    fixed.EnableCollision(False)
    fixed.SetPos(chrono.ChVector3d(*plan.joint_origin_abs_m))
    system.Add(fixed)

    moving = chrono.ChBodyAuxRef()
    moving.SetName("moving_group")
    moving.SetFixed(False)
    moving.EnableCollision(False)
    moving.SetPos(chrono.ChVector3d(*plan.body_origin_abs_m))
    moving.SetRot(chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0))
    moving.SetMass(plan.moving_mass_kg)
    moving.SetInertiaXX(
        chrono.ChVector3d(*_inertia_box(plan.moving_mass_kg, plan.moving_extents_m))
    )
    system.Add(moving)

    prismatic = chrono.ChLinkLockPrismatic()
    prismatic.Initialize(
        moving,
        fixed,
        chrono.ChFramed(
            chrono.ChVector3d(*plan.joint_origin_abs_m),
            _quat_from_z_to_axis(chrono, plan.axis_abs),
        ),
    )
    ideal_travel_limit = _configure_prismatic_travel_limit(
        prismatic,
        max(plan.home_progress_m, plan.outbound_progress_m),
        min_progress_m=min(
            plan.home_progress_m, plan.outbound_progress_m
        ),
    )
    system.AddLink(prismatic)
    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if link_count != 1:
        raise RuntimeError("whole-machine replacement did not create one prismatic link")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    application_start = chrono.ChVector3d(*plan.input_point_abs_m)
    application_local = moving.TransformPointParentToLocal(application_start)
    force_load = create_body_force(
        chrono,
        moving,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        application_start,
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plan.mechanism_plugin_id,
            profile="task_input_controller",
            wrench_kind="force",
            load_id="whole-machine-controller-force",
            owner=plan.mechanism_plugin_id,
            target_body="moving_group",
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="certified_replacement_stage_controller",
            active_interval="outbound_and_return_controller_stages",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, force_load)

    initial_position = _snapshot_vector3(moving.GetPos())
    initial_rotation = moving.GetRot()
    initial_quaternion = (
        float(initial_rotation.e0),
        float(initial_rotation.e1),
        float(initial_rotation.e2),
        float(initial_rotation.e3),
    )
    axis = plan.axis_abs
    time_s = 0.0
    next_sample_s = 0.0
    samples: list[dict[str, Any]] = []

    def state() -> tuple[float, float, float, float]:
        position = moving.GetPos()
        velocity = moving.GetPosDt()
        displacement = (
            float(position.x) - initial_position[0],
            float(position.y) - initial_position[1],
            float(position.z) - initial_position[2],
        )
        progress = sum(value * direction for value, direction in zip(displacement, axis))
        lateral = tuple(
            displacement[index] - progress * axis[index] for index in range(3)
        )
        speed = sum(
            value * direction
            for value, direction in zip(
                (float(velocity.x), float(velocity.y), float(velocity.z)),
                axis,
            )
        )
        rotation = moving.GetRot()
        angle = _quat_angle(
            initial_quaternion,
            (
                float(rotation.e0),
                float(rotation.e1),
                float(rotation.e2),
                float(rotation.e3),
            ),
        )
        return progress, speed, math.sqrt(sum(value * value for value in lateral)), angle

    def record(stage: str, force_scalar_n: float) -> None:
        progress, speed, lateral, rotation = state()
        samples.append(
            {
                "time_s": time_s,
                "stage": stage,
                "target_progress_m": (
                    plan.outbound_progress_m if stage == "outbound" else plan.home_progress_m
                ),
                "progress_m": progress,
                "axis_speed_m_s": speed,
                "lateral_shift_m": lateral,
                "rotation_from_start_rad": rotation,
                "control_force_n": force_scalar_n,
            }
        )

    record("initial", 0.0)
    next_sample_s = plan.sample_interval_s

    def run_stage(
        stage: str,
        target_m: float,
        stop_condition: str,
    ) -> dict[str, Any]:
        nonlocal time_s, next_sample_s
        transition_current_wrench_phase(stage)
        control_target_m = target_m + (
            0.5 * plan.position_tolerance_m
            if stop_condition == "progress_at_least"
            else -0.5 * plan.position_tolerance_m
        )
        elapsed = 0.0
        settled = 0.0
        max_progress = -math.inf
        min_progress = math.inf
        max_force = 0.0
        reached = False
        while elapsed < plan.max_stage_duration_s - 1e-12:
            progress, speed, _, _ = state()
            error = control_target_m - progress
            force_scalar = max(
                -plan.max_control_force_n,
                min(
                    plan.max_control_force_n,
                    plan.controller_kp_n_m * error
                    - plan.controller_kd_n_s_m * speed,
                ),
            )
            application = moving.TransformPointLocalToParent(application_local)
            force_load.SetApplicationPoint(application, False)
            force_load.SetForce(
                chrono.ChVector3d(*(force_scalar * value for value in axis)),
                False,
            )
            system.DoStepDynamics(plan.step_s)
            elapsed += plan.step_s
            time_s += plan.step_s
            progress, speed, _, _ = state()
            max_progress = max(max_progress, progress)
            min_progress = min(min_progress, progress)
            max_force = max(max_force, abs(force_scalar))
            if (
                abs(target_m - progress) <= plan.position_tolerance_m
                and abs(speed) <= plan.speed_tolerance_m_s
            ):
                settled += plan.step_s
            else:
                settled = 0.0
            if time_s + 1e-12 >= next_sample_s:
                record(stage, force_scalar)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += plan.sample_interval_s
            if settled >= plan.settle_hold_s:
                reached = registered_stop_comparator_reached(
                    stop_condition,
                    target_coordinate_si=target_m,
                    actual_coordinate_si=progress,
                )
                if reached:
                    break
        force_load.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        record(stage, 0.0)
        final_progress, _, _, _ = state()
        return {
            "stage": stage,
            "target_progress_m": target_m,
            "reached": reached,
            "duration_s": elapsed,
            "min_progress_m": min_progress,
            "max_progress_m": max_progress,
            "max_control_force_n": max_force,
            "final_coordinate_si": final_progress,
        }

    try:
        outbound_stop = (
            "progress_at_least"
            if plan.outbound_progress_m >= plan.home_progress_m
            else "progress_at_most"
        )
        return_stop = (
            "progress_at_most"
            if outbound_stop == "progress_at_least"
            else "progress_at_least"
        )
        outbound = run_stage(
            "outbound",
            plan.outbound_progress_m,
            outbound_stop,
        )
        returned = run_stage(
            "return",
            plan.home_progress_m,
            return_stop,
        )
    finally:
        set_current_wrench_phase(None)
    max_lateral = max(float(sample["lateral_shift_m"]) for sample in samples)
    max_rotation = max(float(sample["rotation_from_start_rad"]) for sample in samples)
    passed = (
        outbound["reached"]
        and returned["reached"]
        and outbound["max_progress_m"]
        >= plan.outbound_progress_m - plan.position_tolerance_m
        and abs(samples[-1]["progress_m"] - plan.home_progress_m)
        <= plan.position_tolerance_m
        and max_lateral <= 1e-6
        and max_rotation <= 1e-6
    )
    version = str(chrono.GetChronoVersion()) if hasattr(chrono, "GetChronoVersion") else None
    return {
        "schema_version": "0.1",
        "validator_source_sha256": mechanism_source_sha256(),
        "status": "WHOLE_MACHINE_TASK_PASSED" if passed else "WHOLE_MACHINE_TASK_FAILED",
        "passed": passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "mechanism_plugin_id": plan.mechanism_plugin_id,
        "worker_backend": plan.worker_backend,
        "certificate_sha256": plan.certificate_sha256,
        "replacement_sha256": plan.replacement_sha256,
        "task_ids": list(plan.task_ids),
        "engine": {
            "name": "PyChrono",
            "version": version,
            "system": "ChSystemNSC",
            "mechanism_plugin_id": plan.mechanism_plugin_id,
            "worker_backend": plan.worker_backend,
            "chrono_link_count": link_count,
            "replacement_link": "ChLinkLockPrismatic",
            "replacement_link_limit": ideal_travel_limit,
            "high_fidelity_internal_contact_replaced": True,
            "gravity_abs_m_s2": list(plan.gravity_abs_m_s2),
        },
        "controller": {
            "kind": "bounded_external_force_pd",
            "max_force_n": plan.max_control_force_n,
            "kp_n_m": plan.controller_kp_n_m,
            "kd_n_s_m": plan.controller_kd_n_s_m,
        },
        "stages": [outbound, returned],
        "max_lateral_shift_m": max_lateral,
        "max_rotation_rad": max_rotation,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "high_fidelity_whole_machine_contact_run": False,
            "dynamic_response_certified": False,
            "controller_represents_real_hardware": False,
        },
    }


def _signed_axis_angle(
    quaternion_wxyz: tuple[float, float, float, float],
    axis: tuple[float, float, float],
) -> float:
    projected = sum(
        value * direction
        for value, direction in zip(quaternion_wxyz[1:], axis)
    )
    return 2.0 * math.atan2(projected, quaternion_wxyz[0])


def _angular_velocity(body: Any) -> tuple[float, float, float]:
    for method_name in ("GetAngVelParent", "GetAngVelLocal"):
        method = getattr(body, method_name, None)
        if method is not None:
            return _snapshot_vector3(method())
    return (0.0, 0.0, 0.0)


def _simulate_revolute(
    plan: WholeMachineRevoluteExecutionPlan,
) -> dict[str, Any]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(*plan.gravity_abs_m_s2))
    fixed = chrono.ChBody()
    fixed.SetName("certified_machine_frame")
    fixed.SetFixed(True)
    fixed.EnableCollision(False)
    fixed.SetPos(chrono.ChVector3d(*plan.joint_origin_abs_m))
    system.Add(fixed)

    moving = chrono.ChBodyAuxRef()
    moving.SetName("moving_group")
    moving.SetFixed(False)
    moving.EnableCollision(False)
    moving.SetPos(chrono.ChVector3d(*plan.body_origin_abs_m))
    moving.SetRot(chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0))
    moving.SetMass(plan.moving_mass_kg)
    moving.SetInertiaXX(
        chrono.ChVector3d(*_inertia_box(plan.moving_mass_kg, plan.moving_extents_m))
    )
    system.Add(moving)

    revolute = chrono.ChLinkLockRevolute()
    revolute.Initialize(
        moving,
        fixed,
        chrono.ChFramed(
            chrono.ChVector3d(*plan.joint_origin_abs_m),
            _quat_from_z_to_axis(chrono, plan.axis_abs),
        ),
    )
    ideal_angle_limit = _configure_revolute_angle_limit(
        revolute,
        max(plan.home_angle_rad, plan.outbound_angle_rad),
        min_angle_rad=min(plan.home_angle_rad, plan.outbound_angle_rad),
    )
    system.AddLink(revolute)
    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if link_count != 1:
        raise RuntimeError("whole-machine replacement did not create one revolute link")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    control_torque_load = create_body_torque(
        chrono,
        moving,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plan.mechanism_plugin_id,
            profile="task_input_controller",
            wrench_kind="torque",
            load_id="whole-machine-controller-torque",
            owner=plan.mechanism_plugin_id,
            target_body="moving_group",
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="certified_replacement_stage_controller",
            active_interval="outbound_and_return_controller_stages",
            release_required=True,
        ),
    )
    spring_torque_load = create_body_torque(
        chrono,
        moving,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plan.mechanism_plugin_id,
            profile="constitutive",
            wrench_kind="torque",
            load_id="whole-machine-passive-return-spring",
            owner=plan.mechanism_plugin_id,
            target_body="moving_group",
            source_role="finite_passive_return_spring",
            target_role="declared_input",
            geometry_gate="certified_axis_spring_constitutive_law",
            active_interval="simulation_when_spring_response_is_nonzero",
        ),
    )
    add_ledgered_load(load_container, control_torque_load)
    add_ledgered_load(load_container, spring_torque_load)
    initial_rotation = moving.GetRot()
    initial_quaternion = (
        float(initial_rotation.e0),
        float(initial_rotation.e1),
        float(initial_rotation.e2),
        float(initial_rotation.e3),
    )
    joint_start = chrono.ChVector3d(*plan.joint_origin_abs_m)
    joint_local = moving.TransformPointParentToLocal(joint_start)
    axis = plan.axis_abs
    time_s = 0.0
    next_sample_s = 0.0
    samples: list[dict[str, Any]] = []

    def state() -> tuple[float, float, float, float]:
        rotation = moving.GetRot()
        quaternion = (
            float(rotation.e0), float(rotation.e1),
            float(rotation.e2), float(rotation.e3),
        )
        angle = _signed_axis_angle(quaternion, axis) - _signed_axis_angle(
            initial_quaternion, axis
        )
        angular_velocity = _angular_velocity(moving)
        speed = sum(
            value * direction for value, direction in zip(angular_velocity, axis)
        )
        projected = sum(
            value * direction for value, direction in zip(quaternion[1:], axis)
        )
        perpendicular = tuple(
            quaternion[index + 1] - projected * axis[index]
            for index in range(3)
        )
        off_axis = 2.0 * math.asin(
            max(0.0, min(1.0, math.sqrt(sum(v * v for v in perpendicular))))
        )
        joint = moving.TransformPointLocalToParent(joint_local)
        joint_position = _snapshot_vector3(joint)
        pivot_drift = math.sqrt(sum(
            (value - initial) ** 2
            for value, initial in zip(joint_position, plan.joint_origin_abs_m)
        ))
        return angle, speed, pivot_drift, off_axis

    def record(
        stage: str,
        torque_scalar_nm: float,
        spring_torque_nm: float,
    ) -> None:
        angle, speed, pivot_drift, off_axis = state()
        position = moving.GetPos()
        rotation = moving.GetRot()
        samples.append({
            "time_s": time_s,
            "stage": stage,
            "target_angle_rad": (
                plan.outbound_angle_rad if stage == "outbound"
                else plan.home_angle_rad
            ),
            "angle_rad": angle,
            "angular_speed_rad_s": speed,
            "pivot_drift_m": pivot_drift,
            "off_axis_rotation_rad": off_axis,
            "control_torque_nm": torque_scalar_nm,
            "spring_torque_nm": spring_torque_nm,
            "moving_pose": {
                "position_abs_m": [float(position.x), float(position.y), float(position.z)],
                "rotation_abs_quat_wxyz": [
                    float(rotation.e0), float(rotation.e1),
                    float(rotation.e2), float(rotation.e3),
                ],
            },
        })

    record("initial", 0.0, 0.0)
    next_sample_s = plan.sample_interval_s

    def run_stage(
        stage: str,
        target_rad: float,
        stop_condition: str | None,
    ) -> dict[str, Any]:
        nonlocal time_s, next_sample_s
        transition_current_wrench_phase(stage)
        elapsed = 0.0
        settled = 0.0
        min_angle = math.inf
        max_angle = -math.inf
        max_torque = 0.0
        reached = False
        passive_stage = stage == "return" and plan.passive_return_required
        control_target_rad = target_rad
        if not passive_stage:
            control_target_rad += (
                0.5 * plan.angle_tolerance_rad
                if stop_condition == "position_at_least"
                else -0.5 * plan.angle_tolerance_rad
            )
        while elapsed < plan.max_stage_duration_s - 1e-12:
            angle, speed, _, _ = state()
            control_torque, spring_torque = _axis_spring_revolute_control(
                angle_rad=angle,
                speed_rad_s=speed,
                target_rad=control_target_rad,
                max_control_torque_nm=plan.max_control_torque_nm,
                controller_kp_nm_rad=plan.controller_kp_nm_rad,
                controller_kd_nm_s_rad=plan.controller_kd_nm_s_rad,
                spring_stiffness_nm_rad=(
                    plan.passive_return_stiffness_nm_rad
                ),
                spring_damping_nm_s_rad=(
                    plan.passive_return_damping_nm_s_rad
                ),
                passive_stage=passive_stage,
            )
            control_torque_load.SetTorque(
                chrono.ChVector3d(*(control_torque * value for value in axis)),
                False,
            )
            spring_torque_load.SetTorque(
                chrono.ChVector3d(*(spring_torque * value for value in axis)),
                False,
            )
            system.DoStepDynamics(plan.step_s)
            elapsed += plan.step_s
            time_s += plan.step_s
            angle, speed, _, _ = state()
            min_angle = min(min_angle, angle)
            max_angle = max(max_angle, angle)
            max_torque = max(max_torque, abs(control_torque))
            if (
                abs(target_rad - angle) <= plan.angle_tolerance_rad
                and abs(speed) <= plan.angular_speed_tolerance_rad_s
            ):
                settled += plan.step_s
            else:
                settled = 0.0
            if time_s + 1e-12 >= next_sample_s:
                record(stage, control_torque, spring_torque)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += plan.sample_interval_s
            if settled >= plan.settle_hold_s:
                reached = passive_stage or (
                    stop_condition is not None
                    and registered_stop_comparator_reached(
                        stop_condition,
                        target_coordinate_si=target_rad,
                        actual_coordinate_si=angle,
                    )
                )
                if reached and not passive_stage:
                    break
        angle, speed, _, _ = state()
        spring_torque = (
            -plan.passive_return_stiffness_nm_rad * angle
            - plan.passive_return_damping_nm_s_rad * speed
        )
        control_torque_load.SetTorque(
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
        )
        spring_torque_load.SetTorque(
            chrono.ChVector3d(*(spring_torque * value for value in axis)),
            False,
        )
        record(stage, 0.0, spring_torque)
        final_angle, _, _, _ = state()
        return {
            "stage": stage,
            "target_angle_rad": target_rad,
            "reached": reached,
            "duration_s": elapsed,
            "min_angle_rad": min_angle,
            "max_angle_rad": max_angle,
            "max_control_torque_nm": max_torque,
            "passive_return": passive_stage,
            "final_coordinate_si": final_angle,
        }

    try:
        outbound_stop = (
            "position_at_least"
            if plan.outbound_angle_rad >= plan.home_angle_rad
            else "position_at_most"
        )
        return_stop = (
            "position_at_most"
            if outbound_stop == "position_at_least"
            else "position_at_least"
        )
        outbound = run_stage(
            "outbound",
            plan.outbound_angle_rad,
            outbound_stop,
        )
        returned = run_stage(
            "return",
            plan.home_angle_rad,
            None if plan.passive_return_required else return_stop,
        )
    finally:
        set_current_wrench_phase(None)
    max_pivot = max(float(sample["pivot_drift_m"]) for sample in samples)
    max_off_axis = max(float(sample["off_axis_rotation_rad"]) for sample in samples)
    failure_reasons: list[str] = []
    if not outbound["reached"]:
        failure_reasons.append("outbound_target_not_settled")
    if not returned["reached"]:
        failure_reasons.append("return_target_not_settled")
    if (
        outbound["max_angle_rad"]
        < plan.outbound_angle_rad - plan.angle_tolerance_rad
    ):
        failure_reasons.append("outbound_angle_not_reached")
    if (
        abs(samples[-1]["angle_rad"] - plan.home_angle_rad)
        > plan.angle_tolerance_rad
    ):
        failure_reasons.append("home_angle_not_reached")
    if max_pivot > plan.kinematic_pivot_tolerance_m:
        failure_reasons.append("ideal_joint_numerical_pivot_drift_exceeded")
    if max_off_axis > plan.kinematic_off_axis_tolerance_rad:
        failure_reasons.append("ideal_joint_numerical_off_axis_drift_exceeded")
    passed = not failure_reasons
    version = str(chrono.GetChronoVersion()) if hasattr(chrono, "GetChronoVersion") else None
    return {
        "schema_version": "0.1",
        "validator_source_sha256": mechanism_source_sha256(),
        "status": "WHOLE_MACHINE_TASK_PASSED" if passed else "WHOLE_MACHINE_TASK_FAILED",
        "passed": passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "mechanism_plugin_id": plan.mechanism_plugin_id,
        "worker_backend": plan.worker_backend,
        "certificate_sha256": plan.certificate_sha256,
        "replacement_sha256": plan.replacement_sha256,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "angle_tolerance_rad": plan.angle_tolerance_rad,
            "angular_speed_tolerance_rad_s": plan.angular_speed_tolerance_rad_s,
            "kinematic_pivot_tolerance_m": plan.kinematic_pivot_tolerance_m,
            "kinematic_off_axis_tolerance_rad": plan.kinematic_off_axis_tolerance_rad,
        },
        "task_ids": list(plan.task_ids),
        "engine": {
            "name": "PyChrono", "version": version, "system": "ChSystemNSC",
            "mechanism_plugin_id": plan.mechanism_plugin_id,
            "worker_backend": plan.worker_backend,
            "chrono_link_count": link_count,
            "replacement_link": "ChLinkLockRevolute",
            "replacement_link_limit": ideal_angle_limit,
            "high_fidelity_internal_contact_replaced": True,
            "gravity_abs_m_s2": list(plan.gravity_abs_m_s2),
        },
        "controller": {
            "kind": (
                "bounded_external_torque_pd_with_passive_spring_return"
                if plan.passive_return_required
                else "bounded_external_torque_pd"
            ),
            "max_torque_nm": plan.max_control_torque_nm,
            "kp_nm_rad": plan.controller_kp_nm_rad,
            "kd_nm_s_rad": plan.controller_kd_nm_s_rad,
            "passive_return_required": plan.passive_return_required,
            "spring_stiffness_nm_rad": plan.passive_return_stiffness_nm_rad,
            "spring_damping_nm_s_rad": plan.passive_return_damping_nm_s_rad,
        },
        "stages": [outbound, returned],
        "max_pivot_drift_m": max_pivot,
        "max_off_axis_rotation_rad": max_off_axis,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "high_fidelity_whole_machine_contact_run": False,
            "dynamic_response_certified": False,
            "controller_represents_real_hardware": False,
        },
    }



def _simulate_compound(
    plan: WholeMachineCompoundExecutionPlan,
) -> dict[str, Any]:
    from .adapters import adapter_from_dict
    from .compound_runtime import simulate_compound_linkage
    from .compound_spec import compound_linkage_from_dict
    from .family_plugins import mechanism_adapter_for
    from .models import LoadCase, MechanismPlan
    from .worker import ExecutionRequest

    mechanism_plan = MechanismPlan.from_dict(plan.mechanism_plan)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism_plan, adapter)
    load_case = LoadCase.from_dict(plan.task_load_case)
    linkage = compound_linkage_from_dict(adapter.family_spec)
    checkpoint_file = _whole_machine_trace_checkpoint_file(
        plan.replacement_file,
        load_case.load_case_id,
    )
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    execution = ExecutionRequest(
        request_file=Path(plan.certificate_file),
        output_file=checkpoint_file,
        branch="reduced",
        artifact_branch="reduced",
        adapter=adapter,
        mechanism_plugin=plugin,
        plan=mechanism_plan,
        load_case=load_case,
        reference_trace_file=ensure_inside(
            Path.cwd(), plan.reference_trace_file
        ),
    )
    expected_context = _wrench_context_for_execution(execution)
    trace, wrench_audit_entry = _run_registered_whole_machine_load_case(
        execution,
        lambda: simulate_compound_linkage(execution, Path.cwd()),
    )
    raw_samples = trace["samples"]
    first_loaded_index = next(
        index for index, phase in enumerate(load_case.phases)
        if phase.wrench_kind != "none"
    )
    second_loaded_index = next(
        index for index, phase in enumerate(load_case.phases[first_loaded_index + 1:], first_loaded_index + 1)
        if phase.wrench_kind != "none"
    )
    outbound_phase_ids = {
        phase.phase_id for phase in load_case.phases[:second_loaded_index]
    }
    samples: list[dict[str, Any]] = []
    for sample in raw_samples:
        phase_id = sample["phase_id"]
        stage = (
            "initial" if phase_id == "initial"
            else "outbound" if phase_id in outbound_phase_ids
            else "return"
        )
        force = sample.get("applied_force_abs_n", [0.0, 0.0, 0.0])
        torque = sample.get("applied_torque_abs_nm", [0.0, 0.0, 0.0])
        force_scalar = sum(
            value * direction
            for value, direction in zip(force, linkage.input_force_direction_abs)
        )
        torque_scalar = sum(
            value * direction
            for value, direction in zip(
                torque, linkage.input_actuation_direction_abs
            )
        )
        samples.append(
            {
                **sample,
                "stage": stage,
                "target_angle_rad": (
                    plan.outbound_angle_rad if stage == "outbound" else plan.home_angle_rad
                ),
                "angle_rad": sample["compound_output_angle_rad"],
                "angular_speed_rad_s": sample["compound_output_speed_rad_s"],
                "control_force_n": force_scalar,
                "control_torque_nm": torque_scalar,
            }
        )
    output_angles = [float(item["angle_rad"]) for item in samples]
    max_joint_drift = max(float(item["max_joint_drift_m"]) for item in samples)
    max_off_axis = max(float(item["off_plane_rotation_rad"]) for item in samples)
    outbound_values = [
        float(item["angle_rad"]) for item in samples if item["stage"] == "outbound"
    ]
    return_values = [
        float(item["angle_rad"]) for item in samples if item["stage"] == "return"
    ]
    outbound_reached = bool(outbound_values) and max(outbound_values) >= (
        plan.outbound_angle_rad - plan.angle_tolerance_rad
    )
    home_reached = abs(output_angles[-1] - plan.home_angle_rad) <= plan.angle_tolerance_rad
    failure_reasons: list[str] = []
    if not outbound_reached:
        failure_reasons.append("outbound_angle_not_reached")
    if not home_reached:
        failure_reasons.append("home_angle_not_reached")
    if max_joint_drift > plan.kinematic_joint_drift_tolerance_m:
        failure_reasons.append("ideal_linkage_numerical_joint_drift_exceeded")
    if max_off_axis > plan.kinematic_off_axis_tolerance_rad:
        failure_reasons.append("ideal_linkage_numerical_off_axis_drift_exceeded")
    expected_links = len(linkage.revolute_joints)
    if trace.get("engine", {}).get("internal_ideal_joint_count") != expected_links:
        failure_reasons.append("replacement_linkage_topology_count_mismatch")
    passed = not failure_reasons
    payload = {
        "schema_version": "0.1",
        "validator_source_sha256": mechanism_source_sha256(),
        "status": "WHOLE_MACHINE_TASK_PASSED" if passed else "WHOLE_MACHINE_TASK_FAILED",
        "passed": passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "mechanism_plugin_id": plan.mechanism_plugin_id,
        "worker_backend": plan.worker_backend,
        "certificate_sha256": plan.certificate_sha256,
        "replacement_sha256": plan.replacement_sha256,
        "failure_reasons": failure_reasons,
        "thresholds": {
            "angle_tolerance_rad": plan.angle_tolerance_rad,
            "kinematic_joint_drift_tolerance_m": plan.kinematic_joint_drift_tolerance_m,
            "kinematic_off_axis_tolerance_rad": plan.kinematic_off_axis_tolerance_rad,
        },
        "task_ids": list(plan.task_ids),
        "engine": {
            **trace["engine"],
            "replacement_links": ["ChLinkLockRevolute"] * expected_links,
            "replacement_link_count": expected_links,
            "high_fidelity_finite_bushings_replaced": True,
        },
        "controller": {
            "kind": (
                "certified_input_torque_schedule_replay"
                if plan.control_wrench_kind == "torque"
                else "certified_input_force_schedule_replay"
            ),
            "wrench_kind": plan.control_wrench_kind,
            "max_force_n": plan.max_control_force_n,
            "max_torque_nm": plan.max_control_torque_nm,
            "input_direction_abs": list(
                linkage.input_actuation_direction_abs
            ),
        },
        "stages": [
            {
                "stage": "outbound",
                "target_angle_rad": plan.outbound_angle_rad,
                "reached": outbound_reached,
                "min_angle_rad": min(outbound_values) if outbound_values else None,
                "max_angle_rad": max(outbound_values) if outbound_values else None,
            },
            {
                "stage": "return",
                "target_angle_rad": plan.home_angle_rad,
                "reached": home_reached,
                "min_angle_rad": min(return_values) if return_values else None,
                "max_angle_rad": max(return_values) if return_values else None,
            },
        ],
        "max_joint_drift_m": max_joint_drift,
        "max_off_axis_rotation_rad": max_off_axis,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "high_fidelity_whole_machine_bushing_run": False,
            "dynamic_response_certified": False,
            "force_schedule_represents_real_hardware_controller": False,
            "exact_non_joint_surface_contact_certified": False,
        },
    }
    return _attach_whole_machine_wrench_audits(
        payload,
        expected_contexts=(expected_context,),
        entries=(wrench_audit_entry,),
    )


def _whole_machine_trace_checkpoint_file(
    replacement_file: str,
    load_case_id: str,
) -> Path:
    """Keep resumable whole-machine trace state separate from authorization.

    The certified replacement is an immutable input to the worker.  It must
    never double as a mutable trace checkpoint. Each task also needs an
    independent checkpoint so one load case cannot resume another.
    """

    replacement = Path(replacement_file)
    return (
        replacement.parent
        / "whole_machine"
        / "checkpoints"
        / f"{load_case_id}.partial.trace.json"
    )


def _clock_task_checkpoint_file(
    replacement_file: str,
    load_case_id: str,
) -> Path:
    """Backward-compatible clock-specific name for the shared safe path."""

    return _whole_machine_trace_checkpoint_file(replacement_file, load_case_id)


def _simulate_clock(
    plan: WholeMachineClockExecutionPlan,
) -> dict[str, Any]:
    from .adapters import adapter_from_dict
    from .clock_gear_runtime import simulate_clock_gear_train
    from .clock_gear_train_spec import clock_gear_train_from_dict
    from .family_plugins import mechanism_adapter_for
    from .models import LoadCase, MechanismPlan
    from .worker import ExecutionRequest

    mechanism_plan = MechanismPlan.from_dict(plan.mechanism_plan)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism_plan, adapter)
    spec = clock_gear_train_from_dict(adapter.family_spec)
    load_cases = tuple(LoadCase.from_dict(item) for item in plan.task_load_cases)
    references = {
        item["load_case_id"]: ensure_inside(Path.cwd(), item["file"])
        for item in plan.reference_traces
    }
    traces: dict[str, dict[str, Any]] = {}
    expected_wrench_contexts: list[dict[str, Any]] = []
    wrench_audit_entries: list[dict[str, Any]] = []
    for load_case in load_cases:
        checkpoint_file = _clock_task_checkpoint_file(
            plan.replacement_file,
            load_case.load_case_id,
        )
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        execution = ExecutionRequest(
            request_file=Path(plan.certificate_file),
            output_file=checkpoint_file,
            branch="reduced",
            artifact_branch="reduced",
            adapter=adapter,
            mechanism_plugin=plugin,
            plan=mechanism_plan,
            load_case=load_case,
            reference_trace_file=references[load_case.load_case_id],
        )
        expected_wrench_contexts.append(
            _wrench_context_for_execution(execution)
        )
        trace, wrench_audit_entry = (
            _run_registered_whole_machine_load_case(
                execution,
                lambda execution=execution: simulate_clock_gear_train(
                    execution,
                    Path.cwd(),
                ),
            )
        )
        traces[load_case.program] = trace
        wrench_audit_entries.append(wrench_audit_entry)

    def span(samples: list[dict[str, Any]], field: str) -> float:
        values = [float(item[field]) for item in samples]
        return max(values) - min(values)

    def hand_metrics(trace: dict[str, Any]) -> dict[str, Any]:
        samples = trace["samples"]
        minute = span(samples, "clock_output_angle_rad")
        hour = span(samples, "hour_hand_angle_rad")
        second = span(samples, "second_hand_angle_rad")
        farthest = max(
            range(len(samples)),
            key=lambda index: abs(
                float(samples[index]["clock_output_angle_rad"])
                - float(samples[0]["clock_output_angle_rad"])
            ),
        )
        minute_delta = (
            float(samples[farthest]["clock_output_angle_rad"])
            - float(samples[0]["clock_output_angle_rad"])
        )
        hour_delta = (
            float(samples[farthest]["hour_hand_angle_rad"])
            - float(samples[0]["hour_hand_angle_rad"])
        )
        second_delta = (
            float(samples[farthest]["second_hand_angle_rad"])
            - float(samples[0]["second_hand_angle_rad"])
        )
        return {
            "minute_span_rad": minute,
            "hour_span_rad": hour,
            "second_span_rad": second,
            "hour_to_minute_ratio": hour / minute if minute > 1.0e-8 else None,
            "second_to_minute_ratio": second / minute if minute > 1.0e-8 else None,
            "same_display_direction": (
                abs(minute_delta) <= 1.0e-8
                or (minute_delta * hour_delta > 0 and minute_delta * second_delta > 0)
            ),
        }

    normal_trace = traces["task_clock_cycle"]
    manual_trace = traces["manual_time_setting_clock"]
    alarm_trace = traces["alarm_clock_cycle"]
    hour_alarm_setting_trace = traces.get("hour_hand_alarm_setting_clock")
    zero_trace = traces.get("manual_zero_and_resume_clock")
    scheduled_trace = traces.get("scheduled_alarm_clock")
    normal = hand_metrics(normal_trace)
    manual = hand_metrics(manual_trace)
    alarm_swing = span(alarm_trace["samples"], "alarm_angle_rad")

    def ratio_ok(value: float | None, target: float) -> bool:
        return value is not None and abs(value / target - 1.0) <= (
            plan.hand_ratio_relative_tolerance
        )

    ideal_links = _expected_reduced_ideal_links(
        plugin,
        mechanism_plan,
        adapter,
    )
    topology_count = len(ideal_links)
    topology_ok = all(
        trace.get("engine", {}).get("internal_ideal_joint_count") == topology_count
        for trace in traces.values()
    )
    direct_actuation_ok = all(
        trace.get("branch_semantics", {}).get(
            "direct_hand_or_hammer_actuation"
        )
        is False
        and trace.get("branch_semantics", {}).get(
            "direct_hour_second_or_hammer_actuation"
        )
        is (
            spec.alarm_setting_clutch is not None
            and program in {
                "hour_hand_alarm_setting_clock",
                "scheduled_alarm_clock",
            }
        )
        for program, trace in traces.items()
    )
    running_pass = (
        max(
            float(item["clock_output_angle_rad"])
            for item in normal_trace["samples"]
        )
        >= plan.outbound_angle_rad - plan.angle_tolerance_rad
    )
    ratio_pass = (
        ratio_ok(normal["hour_to_minute_ratio"], plan.expected_hour_to_minute_ratio)
        and ratio_ok(
            normal["second_to_minute_ratio"],
            plan.expected_second_to_minute_ratio,
        )
        and normal["same_display_direction"]
    )
    manual_setting_span = span(
        manual_trace["samples"], "setting_input_angle_rad"
    )
    manual_pass = (
        manual["minute_span_rad"] >= plan.minimum_manual_setting_travel_rad
        and manual_setting_span >= plan.minimum_manual_setting_travel_rad
        and ratio_ok(
            manual["hour_to_minute_ratio"], plan.expected_hour_to_minute_ratio
        )
        and ratio_ok(
            manual["second_to_minute_ratio"],
            plan.expected_second_to_minute_ratio,
        )
        and manual["same_display_direction"]
        and manual_trace.get("branch_semantics", {}).get("input_actuation_policy")
        == "external_torque_on_setting_crown_only"
    )
    alarm_pass = (
        alarm_swing >= plan.minimum_alarm_swing_rad
        and alarm_trace.get("branch_semantics", {}).get("input_actuation_policy")
        == (
            "external_torque_on_independent_alarm_barrel_only"
            if spec.alarm_isolated
            else "external_torque_on_barrel_only"
        )
    )
    task_results = [
        {
            "task_id": (
                "run_three_hand_clock_logic"
                if spec.alarm_setting_clutch is not None
                else "run_three_hand_clock"
            ),
            "passed": (
                running_pass and ratio_pass
                if spec.alarm_setting_clutch is not None
                else running_pass
            ),
            "evidence": {
                **(
                    normal
                    if spec.alarm_setting_clutch is not None
                    else {}
                ),
                "minute_span_rad": normal["minute_span_rad"],
                "required_angle_rad": (
                    plan.outbound_angle_rad
                    - plan.angle_tolerance_rad
                ),
            },
        },
        *(
            []
            if spec.alarm_setting_clutch is not None
            else [
                {
                    "task_id": "verify_three_hand_ratio",
                    "passed": ratio_pass,
                    "evidence": normal,
                }
            ]
        ),
        {
            "task_id": "set_time_manually",
            "passed": manual_pass,
            "evidence": {
                **manual,
                "setting_crown_span_rad": manual_setting_span,
                "input_policy": manual_trace["branch_semantics"]["input_actuation_policy"],
            },
        },
        {
            "task_id": (
                "ring_alarm_independently"
                if spec.alarm_setting_clutch is not None
                else "ring_twin_bell_alarm"
            ),
            "passed": alarm_pass,
            "evidence": {
                "alarm_hammer_swing_rad": alarm_swing,
                "required_swing_rad": plan.minimum_alarm_swing_rad,
                "input_policy": alarm_trace["branch_semantics"]["input_actuation_policy"],
            },
        },
    ]
    if spec.alarm_setting_clutch is not None:
        if (
            spec.scheduled_alarm is None
            or hour_alarm_setting_trace is None
            or scheduled_trace is None
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_CLOCK_HOUR_ALARM_TRACE_MISSING",
                "five-task alarm clock validation requires hour-hand "
                "setting and selected-position trigger traces",
            )

        def advanced_phase_values(
            trace: dict[str, Any],
            phase_id: str,
            field: str,
        ) -> list[float]:
            return [
                float(item[field])
                for item in trace["samples"]
                if item["phase_id"] == phase_id
            ]

        def advanced_phase_travel(
            trace: dict[str, Any],
            phase_id: str,
            field: str,
        ) -> float:
            values = advanced_phase_values(trace, phase_id, field)
            return values[-1] - values[0] if values else 0.0

        def advanced_phase_span(
            trace: dict[str, Any],
            phase_id: str,
            field: str,
        ) -> float:
            values = advanced_phase_values(trace, phase_id, field)
            return max(values) - min(values) if values else 0.0

        scheduled = spec.scheduled_alarm
        setting_target_values = advanced_phase_values(
            hour_alarm_setting_trace,
            "alarm_setting_latched",
            "alarm_setting_target_angle_rad",
        )
        setting_target_error = (
            setting_target_values[-1]
            - scheduled.setting_target_angle_rad
        )
        setting_target_latch_span = advanced_phase_span(
            hour_alarm_setting_trace,
            "alarm_setting_latched",
            "alarm_setting_target_angle_rad",
        )
        setting_hour_travel = abs(
            advanced_phase_travel(
                hour_alarm_setting_trace,
                "turn_hour_hand_to_alarm_target",
                "hour_hand_angle_rad",
            )
        )
        setting_target_travel = abs(
            advanced_phase_travel(
                hour_alarm_setting_trace,
                "turn_hour_hand_to_alarm_target",
                "alarm_setting_target_angle_rad",
            )
        )
        hour_setting_pass = (
            abs(setting_target_error)
            <= scheduled.setting_angle_tolerance_rad
            and setting_target_latch_span
            <= scheduled.setting_angle_tolerance_rad
            and setting_hour_travel
            >= 0.5 * scheduled.setting_target_angle_rad
            and setting_target_travel
            >= 0.5 * scheduled.setting_target_angle_rad
            and hour_alarm_setting_trace.get(
                "branch_semantics",
                {},
            ).get("input_actuation_policy")
            == "turn_hour_hand_to_latch_alarm_target_only"
        )
        task_results.append(
            {
                "task_id": "set_alarm_time_by_turning_hour_hand",
                "passed": hour_setting_pass,
                "evidence": {
                    "selected_alarm_angle_rad": (
                        scheduled.setting_target_angle_rad
                    ),
                    "target_latch_error_rad": setting_target_error,
                    "target_latch_span_rad": setting_target_latch_span,
                    "hour_hand_setting_travel_rad": setting_hour_travel,
                    "alarm_target_cam_travel_rad": setting_target_travel,
                    "input_policy": hour_alarm_setting_trace[
                        "branch_semantics"
                    ]["input_actuation_policy"],
                },
            }
        )

        trigger_target_values = advanced_phase_values(
            scheduled_trace,
            "alarm_setting_latched",
            "alarm_setting_target_angle_rad",
        )
        trigger_target_angle = trigger_target_values[-1]
        trigger_hour_values = advanced_phase_values(
            scheduled_trace,
            "advance_to_alarm_time",
            "hour_hand_angle_rad",
        )
        selected_position_error = (
            trigger_hour_values[-1] - trigger_target_angle
        )
        trigger_second = abs(
            advanced_phase_travel(
                scheduled_trace,
                "advance_to_alarm_time",
                "second_hand_angle_rad",
            )
        )
        trigger_minute = abs(
            advanced_phase_travel(
                scheduled_trace,
                "advance_to_alarm_time",
                "clock_output_angle_rad",
            )
        )
        trigger_hour = abs(
            advanced_phase_travel(
                scheduled_trace,
                "advance_to_alarm_time",
                "hour_hand_angle_rad",
            )
        )
        target_drift = abs(
            advanced_phase_travel(
                scheduled_trace,
                "advance_to_alarm_time",
                "alarm_setting_target_angle_rad",
            )
        )
        pretrigger_alarm_span = max(
            advanced_phase_span(
                scheduled_trace,
                phase_id,
                "alarm_angle_rad",
            )
            for phase_id in (
                "turn_hour_hand_to_alarm_target",
                "alarm_setting_latched",
                "restore_hour_hand_to_current_time",
                "alarm_target_ready",
                "advance_to_alarm_time",
            )
        )
        ring_alarm_span = advanced_phase_span(
            scheduled_trace,
            "ring_with_timekeeping",
            "alarm_angle_rad",
        )
        ring_time_travel = abs(
            advanced_phase_travel(
                scheduled_trace,
                "ring_with_timekeeping",
                "clock_output_angle_rad",
            )
        )
        scheduled_pass = (
            abs(
                trigger_target_angle
                - scheduled.setting_target_angle_rad
            )
            <= scheduled.setting_angle_tolerance_rad
            and abs(selected_position_error)
            <= scheduled.setting_angle_tolerance_rad
            and target_drift
            <= scheduled.setting_angle_tolerance_rad
            and abs(
                trigger_second
                - scheduled.target_second_hand_travel_rad
            )
            <= scheduled.target_second_hand_tolerance_rad
            and abs(
                trigger_minute
                - scheduled.target_minute_hand_travel_rad
            )
            <= plan.angle_tolerance_rad
            and abs(
                trigger_hour
                - scheduled.target_hour_hand_travel_rad
            )
            <= max(
                scheduled.setting_angle_tolerance_rad,
                plan.angle_tolerance_rad / 12.0,
            )
            and pretrigger_alarm_span <= plan.angle_tolerance_rad
            and ring_alarm_span >= plan.minimum_alarm_swing_rad
            and ring_time_travel
            >= scheduled.minimum_post_trigger_time_travel_rad
            and scheduled_trace.get(
                "branch_semantics",
                {},
            ).get("input_actuation_policy")
            == "turn_hour_hand_to_latch_target_then_run_time_barrel_and_release_independent_alarm_at_selected_position"
        )
        task_results.append(
            {
                "task_id": "trigger_alarm_at_selected_hand_position",
                "passed": scheduled_pass,
                "evidence": {
                    "selected_alarm_angle_rad": trigger_target_angle,
                    "selected_position_error_rad": (
                        selected_position_error
                    ),
                    "target_drift_rad": target_drift,
                    "trigger_second_hand_travel_rad": trigger_second,
                    "trigger_minute_hand_travel_rad": trigger_minute,
                    "trigger_hour_hand_travel_rad": trigger_hour,
                    "pretrigger_alarm_hammer_span_rad": (
                        pretrigger_alarm_span
                    ),
                    "ring_alarm_hammer_span_rad": ring_alarm_span,
                    "ring_timekeeping_travel_rad": ring_time_travel,
                    "input_policy": scheduled_trace[
                        "branch_semantics"
                    ]["input_actuation_policy"],
                },
            }
        )
    if (
        spec.scheduled_alarm is not None
        and spec.alarm_setting_clutch is None
    ):
        if zero_trace is None or scheduled_trace is None:
            raise MechanismValidationError(
                "WHOLE_MACHINE_CLOCK_ADVANCED_TRACE_MISSING",
                "scheduled clock whole-machine validation requires both advanced traces",
            )

        def phase_values(
            trace: dict[str, Any], phase_id: str, field: str
        ) -> list[float]:
            return [
                float(item[field])
                for item in trace["samples"]
                if item["phase_id"] == phase_id
            ]

        def phase_travel(
            trace: dict[str, Any], phase_id: str, field: str
        ) -> float:
            values = phase_values(trace, phase_id, field)
            return values[-1] - values[0] if values else 0.0

        def phase_span(
            trace: dict[str, Any], phase_id: str, field: str
        ) -> float:
            values = phase_values(trace, phase_id, field)
            return max(values) - min(values) if values else 0.0

        zero_start = phase_values(
            zero_trace, "calibration_offset", "clock_output_angle_rad"
        )[0]
        zero_return = phase_values(
            zero_trace, "calibration_return", "clock_output_angle_rad"
        )[-1]
        zero_setting_span = span(zero_trace["samples"], "setting_input_angle_rad")
        zero_offset_travel = phase_travel(
            zero_trace, "calibration_offset", "clock_output_angle_rad"
        )
        zero_release_drift = phase_travel(
            zero_trace, "release_setting", "clock_output_angle_rad"
        )
        zero_resume_travel = phase_travel(
            zero_trace, "post_zero_run", "clock_output_angle_rad"
        )
        zero_pass = (
            zero_setting_span >= plan.minimum_manual_setting_travel_rad
            and abs(zero_offset_travel) >= plan.minimum_manual_setting_travel_rad
            and abs(zero_return - zero_start) <= plan.angle_tolerance_rad
            and abs(zero_release_drift) <= plan.angle_tolerance_rad
            and zero_resume_travel >= plan.minimum_manual_setting_travel_rad
            and zero_trace.get("branch_semantics", {}).get("input_actuation_policy")
            == "calibrate_on_setting_crown_then_release_and_run_time_barrel"
        )
        task_results.append(
            {
                "task_id": "zero_time_and_resume",
                "passed": zero_pass,
                "evidence": {
                    "setting_crown_span_rad": zero_setting_span,
                    "calibration_offset_travel_rad": zero_offset_travel,
                    "zero_residual_rad": zero_return - zero_start,
                    "release_drift_rad": zero_release_drift,
                    "post_release_time_travel_rad": zero_resume_travel,
                    "input_policy": zero_trace["branch_semantics"][
                        "input_actuation_policy"
                    ],
                },
            }
        )

        scheduled = spec.scheduled_alarm
        setting_latch_values = phase_values(
            scheduled_trace, "alarm_setting_latched", "setting_input_angle_rad"
        )
        setting_latch_error = (
            setting_latch_values[-1] - scheduled.setting_target_angle_rad
        )
        trigger_second = abs(
            phase_travel(
                scheduled_trace, "advance_to_alarm_time", "second_hand_angle_rad"
            )
        )
        trigger_minute = abs(
            phase_travel(
                scheduled_trace, "advance_to_alarm_time", "clock_output_angle_rad"
            )
        )
        trigger_hour = abs(
            phase_travel(
                scheduled_trace, "advance_to_alarm_time", "hour_hand_angle_rad"
            )
        )
        pretrigger_alarm_span = max(
            phase_span(
                scheduled_trace, phase_id, "alarm_angle_rad"
            )
            for phase_id in (
                "set_alarm_for_one_minute",
                "alarm_setting_latched",
                "advance_to_alarm_time",
            )
        )
        ring_alarm_span = phase_span(
            scheduled_trace, "ring_with_timekeeping", "alarm_angle_rad"
        )
        ring_time_travel = abs(
            phase_travel(
                scheduled_trace, "ring_with_timekeeping", "clock_output_angle_rad"
            )
        )
        scheduled_pass = (
            abs(setting_latch_error) <= scheduled.setting_angle_tolerance_rad
            and abs(trigger_second - scheduled.target_second_hand_travel_rad)
            <= scheduled.target_second_hand_tolerance_rad
            and abs(trigger_minute - scheduled.target_minute_hand_travel_rad)
            <= plan.angle_tolerance_rad
            and abs(trigger_hour - scheduled.target_hour_hand_travel_rad)
            <= max(0.01, plan.angle_tolerance_rad / 12.0)
            and pretrigger_alarm_span <= plan.angle_tolerance_rad
            and ring_alarm_span >= plan.minimum_alarm_swing_rad
            and ring_time_travel >= scheduled.minimum_post_trigger_time_travel_rad
            and scheduled_trace.get("branch_semantics", {}).get(
                "input_actuation_policy"
            )
            == "set_alarm_on_setting_crown_then_run_time_barrel_and_trigger_independent_alarm_barrel"
        )
        task_results.append(
            {
                "task_id": "alarm_after_one_minute",
                "passed": scheduled_pass,
                "evidence": {
                    "target_time_seconds": scheduled.target_time_seconds,
                    "setting_latch_error_rad": setting_latch_error,
                    "trigger_second_hand_travel_rad": trigger_second,
                    "trigger_minute_hand_travel_rad": trigger_minute,
                    "trigger_hour_hand_travel_rad": trigger_hour,
                    "pretrigger_alarm_hammer_span_rad": pretrigger_alarm_span,
                    "ring_alarm_hammer_span_rad": ring_alarm_span,
                    "ring_timekeeping_travel_rad": ring_time_travel,
                    "input_policy": scheduled_trace["branch_semantics"][
                        "input_actuation_policy"
                    ],
                },
            }
        )
    failure_reasons = [
        f"task_failed:{item['task_id']}" for item in task_results if not item["passed"]
    ]
    if not topology_ok:
        failure_reasons.append("replacement_clock_topology_count_mismatch")
    if not direct_actuation_ok:
        failure_reasons.append("direct_hand_or_hammer_actuation_detected")
    passed = not failure_reasons
    combined_samples = [
        {**sample, "whole_machine_program": program}
        for program, trace in traces.items()
        for sample in trace["samples"]
    ]
    payload = {
        "schema_version": "0.1",
        "validator_source_sha256": mechanism_source_sha256(),
        "status": "WHOLE_MACHINE_TASK_PASSED" if passed else "WHOLE_MACHINE_TASK_FAILED",
        "passed": passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "mechanism_plugin_id": plan.mechanism_plugin_id,
        "worker_backend": plan.worker_backend,
        "certificate_sha256": plan.certificate_sha256,
        "replacement_sha256": plan.replacement_sha256,
        "failure_reasons": failure_reasons,
        "task_ids": list(plan.task_ids),
        "task_results": task_results,
        "engine": {
            "name": "PyChrono",
            "replacement_topology": "certified_reduced_clock_train",
            "replacement_link_count": topology_count,
            "ideal_links": list(ideal_links),
            "high_fidelity_finite_bushings_replaced": True,
        },
        "thresholds": {
            "outbound_angle_rad": plan.outbound_angle_rad,
            "angle_tolerance_rad": plan.angle_tolerance_rad,
            "expected_hour_to_minute_ratio": plan.expected_hour_to_minute_ratio,
            "expected_second_to_minute_ratio": plan.expected_second_to_minute_ratio,
            "hand_ratio_relative_tolerance": plan.hand_ratio_relative_tolerance,
            "minimum_alarm_swing_rad": plan.minimum_alarm_swing_rad,
            "minimum_manual_setting_travel_rad": plan.minimum_manual_setting_travel_rad,
        },
        "external_actuation_evidence": {
            "normal": "time barrel torque only",
            "alarm": (
                "independent alarm barrel torque only"
                if spec.alarm_isolated
                else "shared time barrel torque only"
            ),
            "manual_setting": "rear setting crown torque only",
            "scheduled_alarm": (
                "setting crown, then time barrel plus triggered alarm barrel"
                if spec.scheduled_alarm is not None
                else None
            ),
            "direct_hand_or_hammer_actuation": False,
        },
        "sample_count": len(combined_samples),
        "samples": combined_samples,
        "strict_non_claims": {
            "high_fidelity_whole_machine_contact_run": False,
            "acoustic_loudness_certified": False,
            "long_duration_timekeeping_accuracy_certified": False,
            "controller_represents_real_hardware": False,
        },
    }
    return _attach_whole_machine_wrench_audits(
        payload,
        expected_contexts=expected_wrench_contexts,
        entries=wrench_audit_entries,
    )


def _single_case_whole_machine_execution(plan: Any) -> Any:
    """Reconstruct the exact reduced replay request used by a one-case backend."""

    from .adapters import adapter_from_dict
    from .family_plugins import mechanism_adapter_for
    from .models import LoadCase, MechanismPlan
    from .worker import ExecutionRequest

    mechanism = MechanismPlan.from_dict(plan.mechanism_plan)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism, adapter)
    load_case = LoadCase.from_dict(plan.task_load_case)
    if plugin.plugin_id != plan.mechanism_plugin_id:
        raise MechanismValidationError(
            "WHOLE_MACHINE_WRENCH_PLUGIN_MISMATCH",
            "whole-machine plan changed the ACTIVE mechanism plugin",
        )
    reference = Path(plan.reference_trace_file).resolve()
    output_file = (
        Path(plan.output_trace_file).resolve()
        if isinstance(getattr(plan, "output_trace_file", None), str)
        else reference.with_name("whole_machine.trace.json")
    )
    branch = plugin.whole_machine_artifact_branch
    return ExecutionRequest(
        request_file=output_file.with_name("whole_machine.request.json"),
        output_file=output_file,
        branch=branch,
        artifact_branch=branch,
        adapter=adapter,
        mechanism_plugin=plugin,
        plan=mechanism,
        load_case=load_case,
        reference_trace_file=reference,
    )


def _certified_reference_schedule(execution: Any) -> tuple[list[dict[str, Any]], str]:
    """Load the exact physical schedule that the reduced replay consumes."""

    reference = execution.reference_trace_file
    if reference is None:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
            "whole-machine load-case audit requires a certified reference trace",
        )
    document = read_trace_json(Path(reference))
    schedule = document.get("executed_phases")
    digest = document.get("executed_load_sha256")
    if (
        document.get("branch") != "physical"
        or document.get("load_case_id") != execution.load_case.load_case_id
        or document.get("package_binding_sha256")
        != execution.plan.package_binding_sha256
        or not isinstance(schedule, list)
        or not all(isinstance(item, dict) for item in schedule)
        or digest != canonical_sha256(schedule)
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_SCHEDULE_INVALID",
            "certified physical reference has no exact replay schedule",
        )
    return [dict(item) for item in schedule], str(digest)


def run_family_single_case(
    plan: Any,
    run_simulation: Any,
    *,
    summarize: Any | None = None,
) -> dict[str, Any]:
    """Run one certified single-load-case whole-machine task.

    Family plugins call this from ``simulate_whole_machine``; the optional
    ``summarize`` callback maps the raw trace into the family's result
    document before the generic wrench audits are attached.
    """

    execution = _single_case_whole_machine_execution(plan)
    expected_context = _wrench_context_for_execution(execution)
    executed_phases, executed_hash = _certified_reference_schedule(execution)
    payload, entry = _run_registered_whole_machine_load_case(
        execution,
        run_simulation,
        executed_phases=executed_phases,
        executed_load_sha256=executed_hash,
    )
    if summarize is not None:
        payload = summarize(payload)
    return _attach_whole_machine_wrench_audits(
        payload,
        expected_contexts=(expected_context,),
        entries=(entry,),
    )


def run_plain_whole_machine(
    plan: Any,
    simulate_stage_trace: Any,
) -> dict[str, Any]:
    """Run a plain prismatic/revolute whole-machine task with its schedule.

    ``simulate_stage_trace`` is the family's staged simulation; the executed
    schedule is reconstructed from its stages before the wrench audits.
    """

    execution = _plain_whole_machine_execution(plan)
    expected_context = _wrench_context_for_execution(execution)

    def simulate_plain_with_schedule() -> dict[str, Any]:
        trace = simulate_stage_trace()
        executed_phases = _plain_whole_machine_executed_schedule(
            execution.load_case,
            trace.get("stages"),
        )
        trace["executed_phases"] = executed_phases
        trace["executed_load_sha256"] = canonical_sha256(executed_phases)
        return trace

    payload, entry = _run_registered_whole_machine_load_case(
        execution,
        simulate_plain_with_schedule,
    )
    return _attach_whole_machine_wrench_audits(
        payload,
        expected_contexts=(expected_context,),
        entries=(entry,),
    )


def run_clock_whole_machine(plan: Any) -> dict[str, Any]:
    return _simulate_clock(plan)


def run_compound_whole_machine(plan: Any) -> dict[str, Any]:
    return _simulate_compound(plan)


def simulate_plain_prismatic_stages(plan: Any) -> dict[str, Any]:
    return _simulate_prismatic(plan)


def simulate_plain_revolute_stages(plan: Any) -> dict[str, Any]:
    return _simulate_revolute(plan)


def _simulate(
    plan: Any,
    *,
    workspace_root: Path,
    plugin: Any,
) -> dict[str, Any]:
    """Run through the exact plugin selected by the mechanism plan."""

    from .family_plugins import plugin_registration_or_none

    registration = plugin_registration_or_none(str(plugin.plugin_id))
    if (
        plugin.plugin_id == getattr(plan, "mechanism_plugin_id", None)
        and plugin.whole_machine_backend == getattr(plan, "worker_backend", None)
    ):
        result = invoke_plugin_callback(
            lambda: plugin.simulate_whole_machine(
                plan,
                workspace_root=workspace_root,
            ),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="simulate_whole_machine",
            source_component="mechanism.whole_machine_worker.simulate",
            unexpected_code="MECHANISM_ADAPTER_WHOLE_MACHINE_SIMULATE_FAILED",
            details={"capsule_id": getattr(plan, "capsule_id", None)},
            evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
        )
        if result is not None:
            if not isinstance(result, dict):
                message = (
                    "mechanism adapter simulate_whole_machine() must return "
                    "a result object or None"
                )
                raise plugin_boundary_error(
                    TypeError(message),
                    registration=registration,
                    plugin_id=str(plugin.plugin_id),
                    plugin_kind="mechanism",
                    callback="simulate_whole_machine",
                    source_component="mechanism.whole_machine_worker.simulate",
                    code="MECHANISM_ADAPTER_WHOLE_MACHINE_SIMULATE_INVALID",
                    message=message,
                    details={"capsule_id": getattr(plan, "capsule_id", None)},
                    contract_violation=True,
                    evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
                )
            return result
    message = (
        "whole-machine plugin/backend is not implemented: "
        f"{getattr(plan, 'mechanism_plugin_id', None)!r}/"
        f"{getattr(plan, 'worker_backend', None)!r}"
    )
    raise plugin_boundary_error(
        TypeError(message),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="simulate_whole_machine",
        source_component="mechanism.whole_machine_worker.simulate",
        code="WHOLE_MACHINE_BACKEND_UNAVAILABLE",
        message=message,
        details={"capsule_id": getattr(plan, "capsule_id", None)},
        contract_violation=True,
    )


def run(
    plan_file: str | Path,
    mechanism_plan_file: str | Path,
    output_file: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    plan_path = Path(plan_file)
    mechanism_plan_path = Path(mechanism_plan_file)
    output_path = Path(output_file)
    if not plan_path.is_absolute():
        plan_path = ensure_inside(workspace, plan_path.as_posix())
    else:
        plan_path = plan_path.resolve()
        plan_path.relative_to(workspace)
    if not mechanism_plan_path.is_absolute():
        mechanism_plan_path = ensure_inside(
            workspace, mechanism_plan_path.as_posix()
        )
    else:
        mechanism_plan_path = mechanism_plan_path.resolve()
        mechanism_plan_path.relative_to(workspace)
    if not output_path.is_absolute():
        output_path = ensure_inside(workspace, output_path.as_posix())
    else:
        output_path = output_path.resolve()
        output_path.relative_to(workspace)
    mechanism_plan = MechanismPlan.from_dict(read_json(mechanism_plan_path))
    from .family_plugins import mechanism_adapter_for_plan

    plugin, _adapter = mechanism_adapter_for_plan(mechanism_plan)
    plan = _load_plan(plan_path, workspace, plugin)
    try:
        payload = _simulate(
            plan,
            workspace_root=workspace,
            plugin=plugin,
        )
    except Exception as exc:
        failure = failure_envelope_from_exception(
            exc,
            default_owner=(
                FailureOwner.UNCLASSIFIED
                if isinstance(exc, MechanismValidationError)
                else FailureOwner.HARNESS_CODE
            ),
            source_component="mechanism.whole_machine_worker",
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        )
        payload = {
            "schema_version": "0.1",
            "status": "WHOLE_MACHINE_PHYSICS_INCONCLUSIVE",
            "passed": False,
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "certificate_sha256": plan.certificate_sha256,
            "replacement_sha256": plan.replacement_sha256,
            "error": {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", "WHOLE_MACHINE_SIMULATION_EXCEPTION"),
                "message": str(exc),
                "traceback": traceback.format_exc(limit=12),
            },
            "failure": failure,
        }
    if (
        payload.get("status") == "WHOLE_MACHINE_TASK_FAILED"
        and payload.get("passed") is False
        and not isinstance(payload.get("failure"), dict)
    ):
        reasons = [
            str(item)
            for item in payload.get("failure_reasons") or ()
            if str(item)
        ]
        payload["failure"] = make_failure_envelope(
            "WHOLE_MACHINE_TASK_FAILED",
            FailureOwner.DESIGN_PACKAGE,
            source_component="mechanism.whole_machine_worker.measurement",
            message="the certified replacement failed its whole-machine task",
            evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            causes=[
                {"code": reason, "message": "whole-machine task check failed"}
                for reason in reasons
            ],
            details={"capsule_id": plan.capsule_id},
        )
    write_json_atomic(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a certified reduced whole-machine task.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--mechanism-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _verify_validator_source_binding()
        payload = run(
            args.plan,
            args.mechanism_plan,
            args.output,
            workspace_root=args.workspace_root,
        )
    except Exception as exc:
        failure = failure_envelope_from_exception(
            exc,
            default_owner=(
                FailureOwner.UNCLASSIFIED
                if isinstance(exc, MechanismValidationError)
                else FailureOwner.HARNESS_CODE
            ),
            source_component="mechanism.whole_machine_worker.request",
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        )
        print(
            json.dumps(
                {
                    "status": "INVALID_WHOLE_MACHINE_REQUEST",
                    "error_code": getattr(exc, "code", type(exc).__name__),
                    "message": str(exc),
                    "failure": failure,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps({"status": payload["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
