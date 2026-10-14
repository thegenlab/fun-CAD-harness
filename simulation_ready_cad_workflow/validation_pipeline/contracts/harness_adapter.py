"""The frozen calling contract between the Loop and any Harness.

The Loop schedules validation exclusively through ``HarnessAdapter`` and
consumes exactly one ``HarnessOutcome`` per attempt.  The production
implementation lives in ``validation_pipeline.harness``; the Loop never
imports Harness internals and the Harness never imports Loop code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

from .errors import ReactLoopError
from .failure_envelope import FailureEnvelopeError, validate_failure_envelope


class OutcomeKind(str, Enum):
    PASS = "PASS"
    FAIL_CONTINUE_LOOP = "FAIL_CONTINUE_LOOP"
    RESOURCE_BLOCKED = "RESOURCE_BLOCKED"
    VISUAL_REVIEW_REQUIRED = "VISUAL_REVIEW_REQUIRED"
    TERMINATE = "TERMINATE"


PASS_RECORDED_STATUSES = frozenset(
    {
        "KINEMATIC_CERTIFIED",
        "TASK_CERTIFIED",
        "WHOLE_MACHINE_TASK_PASSED",
        "ITERATION_SCOPE_PASSED",
    }
)
RESOURCE_BLOCKED_STATUS = "RESOURCE_BLOCKED"
VISUAL_REVIEW_STATUS = "REQUIREMENT_VISUAL_REVIEW_REQUIRED"
TERMINATE_STATUS = "TERMINATED"
INTAKE_REJECTED_STATUS = "REJECTED"

_FAIL_FORBIDDEN_STATUSES = (
    PASS_RECORDED_STATUSES
    | {
        RESOURCE_BLOCKED_STATUS,
        VISUAL_REVIEW_STATUS,
        TERMINATE_STATUS,
        OutcomeKind.PASS.value,
    }
)


@dataclass(frozen=True)
class HarnessOutcome:
    kind: OutcomeKind
    recorded_status: str
    result: dict[str, Any]
    agent_feedback: dict[str, Any] | None
    attempt_started: bool
    elapsed_seconds: float


def validate_harness_outcome(outcome: HarnessOutcome) -> None:
    """Reject kind/status/result envelopes the Loop must not consume.

    Intake rejection uses a non-harness envelope with status ``REJECTED``.
    Every other terminal kind is bound to the closed Harness status map.
    """

    if not isinstance(outcome, HarnessOutcome):
        raise ReactLoopError(
            "HARNESS_CONTRACT_VIOLATION",
            "HarnessOutcome is missing",
        )
    result = outcome.result
    if not isinstance(result, Mapping):
        raise ReactLoopError(
            "HARNESS_CONTRACT_VIOLATION",
            "HarnessOutcome.result must be an object",
        )
    recorded = str(outcome.recorded_status or "").strip()
    status = str(result.get("status") or "").strip()
    if not recorded or recorded != status:
        raise ReactLoopError(
            "HARNESS_CONTRACT_VIOLATION",
            "recorded_status must equal result.status",
        )
    # Persisted outcomes from before failure-envelope v1 intentionally remain
    # readable.  Every new production outcome carries ``failure``; whenever
    # that field is present the consumption boundary validates it strictly.
    if "failure" in result:
        try:
            validate_failure_envelope(result.get("failure"))
        except (FailureEnvelopeError, TypeError, ValueError) as exc:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                f"HarnessOutcome.failure is invalid: {exc}",
            ) from exc
    kind = outcome.kind
    if kind is OutcomeKind.PASS:
        if recorded not in PASS_RECORDED_STATUSES:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "PASS requires a certified Harness status",
            )
        evidence = result.get("evidence_index")
        if (
            not isinstance(evidence, Mapping)
            or not str(evidence.get("schema_version") or "")
            or not str(evidence.get("path") or "")
            or not str(evidence.get("sha256") or "")
        ):
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "PASS requires a published evidence_index",
            )
        return
    if kind is OutcomeKind.RESOURCE_BLOCKED:
        if recorded != RESOURCE_BLOCKED_STATUS:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "RESOURCE_BLOCKED kind requires RESOURCE_BLOCKED status",
            )
        if result.get("resource_blocked") is not True:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "RESOURCE_BLOCKED requires resource_blocked=true",
            )
        if not str(result.get("retry_mode") or "").strip():
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "RESOURCE_BLOCKED requires retry_mode",
            )
        if not str(result.get("last_completed_stage") or "").strip():
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "RESOURCE_BLOCKED requires last_completed_stage",
            )
        return
    if kind is OutcomeKind.VISUAL_REVIEW_REQUIRED:
        if recorded != VISUAL_REVIEW_STATUS:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "VISUAL_REVIEW_REQUIRED requires "
                "REQUIREMENT_VISUAL_REVIEW_REQUIRED",
            )
        return
    if kind is OutcomeKind.TERMINATE:
        if recorded != TERMINATE_STATUS:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "TERMINATE requires TERMINATED status",
            )
        return
    if kind is OutcomeKind.FAIL_CONTINUE_LOOP:
        if recorded in _FAIL_FORBIDDEN_STATUSES:
            raise ReactLoopError(
                "HARNESS_CONTRACT_VIOLATION",
                "FAIL_CONTINUE_LOOP cannot carry a terminal success or "
                "control-flow status",
            )
        return
    raise ReactLoopError(
        "HARNESS_CONTRACT_VIOLATION",
        f"unsupported HarnessOutcome kind {kind!r}",
    )


class HarnessAdapter(Protocol):
    def run(
        self,
        *,
        workspace_root: Path,
        run_dir: Path,
        package_dir: Path,
        package_inventory_sha256: str,
        requirements_path: Path,
        requirements_sha256: str,
        revision_hypothesis: Mapping[str, Any] | None,
    ) -> HarnessOutcome:
        ...

    def resume(
        self,
        *,
        workspace_root: Path,
        run_dir: Path,
    ) -> HarnessOutcome:
        ...
