"""HarnessOutcome consumption-boundary contract."""

from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.contracts.failure_envelope import (
    FailureOwner,
    make_failure_envelope,
)
from simulation_ready_cad_workflow.validation_pipeline.contracts.errors import (
    ReactLoopError,
)
from simulation_ready_cad_workflow.validation_pipeline.contracts.harness_adapter import (
    HarnessOutcome,
    OutcomeKind,
    validate_harness_outcome,
)


def _pass_result() -> dict:
    return {
        "schema_version": "simulation_ready_cad_harness_result_v2",
        "status": "KINEMATIC_CERTIFIED",
        "last_completed_stage": "finalize",
        "evidence_index": {
            "schema_version": "simulation_ready_cad_evidence_index_v1",
            "path": "evidence_index.json",
            "sha256": "0" * 64,
            "entry_count": 0,
        },
    }


class HarnessOutcomeContractTest(unittest.TestCase):
    def test_new_failure_envelope_is_validated_but_legacy_remains_readable(
        self,
    ) -> None:
        valid = HarnessOutcome(
            kind=OutcomeKind.FAIL_CONTINUE_LOOP,
            recorded_status="FAILED",
            result={
                "status": "FAILED",
                "failure": make_failure_envelope(
                    "CHECK_FAILED",
                    FailureOwner.HARNESS_CODE,
                    source_component="test.harness",
                ),
            },
            agent_feedback=None,
            attempt_started=True,
            elapsed_seconds=0.1,
        )
        validate_harness_outcome(valid)

        with self.assertRaises(ReactLoopError) as caught:
            validate_harness_outcome(
                HarnessOutcome(
                    kind=OutcomeKind.FAIL_CONTINUE_LOOP,
                    recorded_status="FAILED",
                    result={
                        "status": "FAILED",
                        "failure": {"schema_version": "broken"},
                    },
                    agent_feedback=None,
                    attempt_started=True,
                    elapsed_seconds=0.1,
                )
            )
        self.assertEqual(caught.exception.code, "HARNESS_CONTRACT_VIOLATION")

        # No ``failure`` means a persisted pre-v1 outcome, not a new inferred
        # owner.  The router's explicit legacy upgrader handles this record.
        validate_harness_outcome(
            HarnessOutcome(
                kind=OutcomeKind.FAIL_CONTINUE_LOOP,
                recorded_status="FAILED",
                result={"status": "FAILED", "failure_reason": "OLD_FAILURE"},
                agent_feedback=None,
                attempt_started=True,
                elapsed_seconds=0.1,
            )
        )

    def test_pass_requires_certified_status_and_evidence_index(self) -> None:
        validate_harness_outcome(
            HarnessOutcome(
                kind=OutcomeKind.PASS,
                recorded_status="KINEMATIC_CERTIFIED",
                result=_pass_result(),
                agent_feedback=None,
                attempt_started=True,
                elapsed_seconds=0.1,
            )
        )
        with self.assertRaises(ReactLoopError) as caught:
            validate_harness_outcome(
                HarnessOutcome(
                    kind=OutcomeKind.PASS,
                    recorded_status="PASS",
                    result={"status": "PASS"},
                    agent_feedback=None,
                    attempt_started=True,
                    elapsed_seconds=0.1,
                )
            )
        self.assertEqual(caught.exception.code, "HARNESS_CONTRACT_VIOLATION")

    def test_kind_status_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ReactLoopError) as caught:
            validate_harness_outcome(
                HarnessOutcome(
                    kind=OutcomeKind.PASS,
                    recorded_status="FAILED",
                    result={"status": "FAILED"},
                    agent_feedback=None,
                    attempt_started=True,
                    elapsed_seconds=0.1,
                )
            )
        self.assertEqual(caught.exception.code, "HARNESS_CONTRACT_VIOLATION")

    def test_resource_blocked_requires_retry_fields(self) -> None:
        with self.assertRaises(ReactLoopError):
            validate_harness_outcome(
                HarnessOutcome(
                    kind=OutcomeKind.RESOURCE_BLOCKED,
                    recorded_status="RESOURCE_BLOCKED",
                    result={"status": "RESOURCE_BLOCKED"},
                    agent_feedback=None,
                    attempt_started=False,
                    elapsed_seconds=0.1,
                )
            )
        validate_harness_outcome(
            HarnessOutcome(
                kind=OutcomeKind.RESOURCE_BLOCKED,
                recorded_status="RESOURCE_BLOCKED",
                result={
                    "status": "RESOURCE_BLOCKED",
                    "resource_blocked": True,
                    "retry_mode": "RERUN_SAME_SUBMISSION",
                    "last_completed_stage": "physical",
                },
                agent_feedback=None,
                attempt_started=False,
                elapsed_seconds=0.1,
            )
        )

    def test_intake_rejected_envelope_is_allowed_as_fail_continue(self) -> None:
        validate_harness_outcome(
            HarnessOutcome(
                kind=OutcomeKind.FAIL_CONTINUE_LOOP,
                recorded_status="REJECTED",
                result={
                    "schema_version": "simulation_ready_cad_intake_report_v1",
                    "status": "REJECTED",
                    "issues": [],
                },
                agent_feedback={"phase": "intake", "status": "REJECTED"},
                attempt_started=True,
                elapsed_seconds=0.01,
            )
        )


if __name__ == "__main__":
    unittest.main()
