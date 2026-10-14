"""Non-certifying outcomes for candidate jobs that reach a timebox.

Only supervisor-owned facts are considered here.  A timed-out child cannot
turn its own logs, traces, checkpoints, or claimed progress into success.
The raw job remains ``TIMED_OUT`` and capability activation still requires a
fully successful queue.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


TIMEBOX_OUTCOME_SCHEMA = "simulation_ready_cad_timebox_outcome_v1"
TIMEBOX_POLICY_ENV = "SIMREADY_TIMEBOX_POLICY"
TIMEBOX_POLICY = "continue_after_supervisor_deadline_v1"
TIMEBOX_COMPLETED = "TIMEBOX_COMPLETED_NO_ANOMALY"
TIMEBOX_NOT_ACCEPTED = "TIMEBOX_NOT_ACCEPTED"
TIMEBOX_ANOMALY_SCOPE = "SUPERVISOR_OBSERVABLE_PROCESS_LIFECYCLE"
PHYSICAL_TIMEBOX_LIMIT_SECONDS = 300.0
NONPHYSICAL_TIMEBOX_LIMIT_SECONDS = 60.0


def _outcome(
    *,
    status: str,
    reasons: Sequence[str],
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    accepted = status == TIMEBOX_COMPLETED
    return {
        "schema_version": TIMEBOX_OUTCOME_SCHEMA,
        "status": status,
        "process_status": "TIMED_OUT",
        "policy": TIMEBOX_POLICY,
        "anomaly_scope": TIMEBOX_ANOMALY_SCOPE,
        "may_continue": accepted,
        "raw_job_success": False,
        "physical_pass": False,
        "contract_pass": False,
        "activation_authority": False,
        "terminal_authority": False,
        "certificate_authority": False,
        "semantic_progress_proven": False,
        "goal_alignment_evaluated": False,
        "reasons": sorted(
            {str(item) for item in reasons if str(item)}
        ),
        "evidence": dict(evidence),
    }


def derive_timebox_outcome(
    *,
    job_environment: Mapping[str, str],
    exact_candidate_worker_bound: bool,
    dispatch_context_published: bool,
    full_declared_timebox_allocated: bool,
    deadline_reached_with_process_alive: bool,
    process_tree_stopped_after_cleanup: bool,
    input_files_unchanged_after: bool,
    finalization_errors: Sequence[str],
    observed_duration_seconds: float,
    total_wall_limit_seconds: float,
) -> dict[str, Any]:
    """Classify a timeout from facts observed by the parent supervisor."""

    reasons: list[str] = []
    if job_environment.get(TIMEBOX_POLICY_ENV) != TIMEBOX_POLICY:
        reasons.append("trusted_timebox_policy_not_declared")
    execution_class = job_environment.get(
        "SIMREADY_CAPABILITY_EXECUTION_CLASS"
    )
    expected_limit = {
        "physical": PHYSICAL_TIMEBOX_LIMIT_SECONDS,
        "nonphysical": NONPHYSICAL_TIMEBOX_LIMIT_SECONDS,
    }.get(str(execution_class))
    if expected_limit is None:
        reasons.append("candidate_execution_class_invalid")
    if exact_candidate_worker_bound is not True:
        reasons.append("exact_candidate_worker_not_bound")
    if dispatch_context_published is not True:
        reasons.append("trusted_dispatch_context_not_published")
    if full_declared_timebox_allocated is not True:
        reasons.append("full_declared_timebox_not_allocated")
    if deadline_reached_with_process_alive is not True:
        reasons.append("process_not_proven_alive_at_execution_deadline")
    if process_tree_stopped_after_cleanup is not True:
        reasons.append("process_tree_not_proven_stopped_after_cleanup")
    if input_files_unchanged_after is not True:
        reasons.append("declared_inputs_changed")
    if finalization_errors:
        reasons.append("attempt_finalization_error")

    try:
        duration = float(observed_duration_seconds)
        limit = float(total_wall_limit_seconds)
    except (TypeError, ValueError):
        duration = math.nan
        limit = math.nan
    if (
        not math.isfinite(duration)
        or not math.isfinite(limit)
        or duration < 0.0
        or limit <= 0.0
    ):
        reasons.append("timebox_duration_invalid")
    elif not math.isclose(
        limit,
        float(expected_limit or 0.0),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        reasons.append("candidate_total_wall_limit_invalid")
    elif duration > limit:
        reasons.append("total_wall_limit_exceeded")

    evidence = {
        "source": "parent_process_supervisor",
        "execution_class": execution_class,
        "exact_candidate_worker_bound": (
            exact_candidate_worker_bound is True
        ),
        "dispatch_context_published": (
            dispatch_context_published is True
        ),
        "full_declared_timebox_allocated": (
            full_declared_timebox_allocated is True
        ),
        "deadline_reached_with_process_alive": (
            deadline_reached_with_process_alive is True
        ),
        "process_tree_stopped_after_cleanup": (
            process_tree_stopped_after_cleanup is True
        ),
        "declared_inputs_unchanged": (
            input_files_unchanged_after is True
        ),
        "finalization_error_count": len(finalization_errors),
        "observed_duration_seconds": duration,
        "total_wall_limit_seconds": limit,
        "child_reported_progress_used": False,
    }
    if reasons:
        return _outcome(
            status=TIMEBOX_NOT_ACCEPTED,
            reasons=reasons,
            evidence=evidence,
        )
    return _outcome(
        status=TIMEBOX_COMPLETED,
        reasons=[
            "supervisor_observed_process_alive_until_execution_deadline",
            "no_supervisor_observed_anomaly",
        ],
        evidence=evidence,
    )


def timebox_outcome_allows_continuation(value: Any) -> bool:
    """Accept only the exact non-certifying outcome shape."""

    expected_fields = {
        "schema_version",
        "status",
        "process_status",
        "policy",
        "anomaly_scope",
        "may_continue",
        "raw_job_success",
        "physical_pass",
        "contract_pass",
        "activation_authority",
        "terminal_authority",
        "certificate_authority",
        "semantic_progress_proven",
        "goal_alignment_evaluated",
        "reasons",
        "evidence",
    }
    expected_evidence_fields = {
        "source",
        "execution_class",
        "exact_candidate_worker_bound",
        "dispatch_context_published",
        "full_declared_timebox_allocated",
        "deadline_reached_with_process_alive",
        "process_tree_stopped_after_cleanup",
        "declared_inputs_unchanged",
        "finalization_error_count",
        "observed_duration_seconds",
        "total_wall_limit_seconds",
        "child_reported_progress_used",
    }
    return (
        isinstance(value, Mapping)
        and set(value) == expected_fields
        and value.get("schema_version") == TIMEBOX_OUTCOME_SCHEMA
        and value.get("status") == TIMEBOX_COMPLETED
        and value.get("process_status") == "TIMED_OUT"
        and value.get("policy") == TIMEBOX_POLICY
        and value.get("anomaly_scope") == TIMEBOX_ANOMALY_SCOPE
        and value.get("may_continue") is True
        and value.get("raw_job_success") is False
        and value.get("physical_pass") is False
        and value.get("contract_pass") is False
        and value.get("activation_authority") is False
        and value.get("terminal_authority") is False
        and value.get("certificate_authority") is False
        and value.get("semantic_progress_proven") is False
        and value.get("goal_alignment_evaluated") is False
        and value.get("reasons")
        == [
            "no_supervisor_observed_anomaly",
            "supervisor_observed_process_alive_until_execution_deadline",
        ]
        and isinstance(value.get("evidence"), Mapping)
        and set(value["evidence"]) == expected_evidence_fields
        and value["evidence"].get("source")
        == "parent_process_supervisor"
        and value["evidence"].get("execution_class")
        in {"physical", "nonphysical"}
        and value["evidence"].get("exact_candidate_worker_bound") is True
        and value["evidence"].get("dispatch_context_published") is True
        and value["evidence"].get("full_declared_timebox_allocated") is True
        and value["evidence"].get(
            "deadline_reached_with_process_alive"
        )
        is True
        and value["evidence"].get(
            "process_tree_stopped_after_cleanup"
        )
        is True
        and value["evidence"].get("declared_inputs_unchanged") is True
        and value["evidence"].get("finalization_error_count") == 0
        and value["evidence"].get("child_reported_progress_used") is False
        and isinstance(
            value["evidence"].get("observed_duration_seconds"),
            (int, float),
        )
        and not isinstance(
            value["evidence"].get("observed_duration_seconds"),
            bool,
        )
        and isinstance(
            value["evidence"].get("total_wall_limit_seconds"),
            (int, float),
        )
        and not isinstance(
            value["evidence"].get("total_wall_limit_seconds"),
            bool,
        )
        and math.isfinite(
            float(value["evidence"]["observed_duration_seconds"])
        )
        and math.isfinite(
            float(value["evidence"]["total_wall_limit_seconds"])
        )
        and (
            (
                value["evidence"]["execution_class"] == "physical"
                and math.isclose(
                    float(
                        value["evidence"][
                            "total_wall_limit_seconds"
                        ]
                    ),
                    PHYSICAL_TIMEBOX_LIMIT_SECONDS,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            )
            or (
                value["evidence"]["execution_class"] == "nonphysical"
                and math.isclose(
                    float(
                        value["evidence"][
                            "total_wall_limit_seconds"
                        ]
                    ),
                    NONPHYSICAL_TIMEBOX_LIMIT_SECONDS,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            )
        )
        and 0.0
        <= float(value["evidence"]["observed_duration_seconds"])
        <= float(value["evidence"]["total_wall_limit_seconds"])
    )


__all__ = [
    "TIMEBOX_COMPLETED",
    "TIMEBOX_ANOMALY_SCOPE",
    "NONPHYSICAL_TIMEBOX_LIMIT_SECONDS",
    "PHYSICAL_TIMEBOX_LIMIT_SECONDS",
    "TIMEBOX_NOT_ACCEPTED",
    "TIMEBOX_OUTCOME_SCHEMA",
    "TIMEBOX_POLICY",
    "TIMEBOX_POLICY_ENV",
    "derive_timebox_outcome",
    "timebox_outcome_allows_continuation",
]
