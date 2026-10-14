"""The frozen Harness result contract and its validating publisher."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.contracts import (
    HARNESS_RESULT_SCHEMA_ID,
    ContractViolation,
    SchemaCheckError,
    check_against_schema,
    harness_result_schema,
    harness_result_statuses,
    publish_run_status,
    validate_harness_result,
)


def _result(**fields: object) -> dict[str, object]:
    return {"schema_version": HARNESS_RESULT_SCHEMA_ID, **fields}


_REQUIREMENT_EXECUTION_PASS = {
    "status": "PASS",
    "passed": True,
    "failed_obligation_ids": [],
    "review_required_obligation_ids": [],
    "report": "runs/r01/results/requirement_obligation_execution.json",
    "report_sha256": "0" * 64,
}


class HarnessResultContractTests(unittest.TestCase):
    def test_status_vocabulary_is_closed_and_stable(self) -> None:
        self.assertEqual(
            harness_result_statuses(),
            frozenset(
                {
                    "RUNNING",
                    "RESOURCE_BLOCKED",
                    "PHYSICAL_PREFLIGHT_FAILED",
                    "FAILED",
                    "TERMINATED",
                    "NOT_CERTIFIED",
                    "KINEMATIC_CERTIFIED",
                    "TASK_CERTIFIED",
                    "WHOLE_MACHINE_TASK_PASSED",
                    "WHOLE_MACHINE_TASK_FAILED",
                    "WHOLE_MACHINE_TASK_INCONCLUSIVE",
                    "ITERATION_SCOPE_PASSED",
                    "ITERATION_SCOPE_FAILED",
                    "REQUIREMENT_OBLIGATIONS_FAILED",
                    "REQUIREMENT_VISUAL_REVIEW_REQUIRED",
                    "REQUIREMENT_VISUAL_REVIEW_REJECTED",
                }
            ),
        )

    def test_every_branch_envelope_validates(self) -> None:
        envelopes = (
            _result(status="RUNNING", last_completed_stage="intake"),
            _result(
                status="RESOURCE_BLOCKED",
                resource_blocked=True,
                resource_kind="CPU_RAM_OR_DISK",
                retry_mode="RERUN_SAME_SUBMISSION",
                last_completed_stage="physical",
                queue_status=None,
                queue_summary={},
            ),
            _result(
                status="PHYSICAL_PREFLIGHT_FAILED",
                failure_reason="PHYSICAL_TEST_LANE_FAILED",
                last_completed_stage="physical_checks",
                physical_queue_status="FAILED",
                physical_queue_summary={"FAILED": 1},
                authored_test_summary={},
                physical_preflight={"passed": False},
                issues=[{"code": "LANE_FAILED", "severity": "error"}],
            ),
            _result(
                status="FAILED",
                failure_reason="IDEAL_REFERENCE_CHECK_FAILED",
                last_completed_stage="ideal_reference",
                ideal_reference={"status": "FAILED"},
            ),
            _result(
                status="TERMINATED",
                error_code="RuntimeError",
                message="internal harness error",
                last_completed_stage="internal_harness_error",
            ),
            _result(status="NOT_CERTIFIED", last_completed_stage="verdict"),
            _result(
                status="TASK_CERTIFIED",
                last_completed_stage="requirement_obligations_passed",
                mechanical_status_before_requirement_execution="TASK_CERTIFIED",
                requirement_execution=_REQUIREMENT_EXECUTION_PASS,
                mechanisms=[{"capsule_id": "drive"}],
                branch_queues={"physical": {"status": "SUCCEEDED"}},
            ),
            _result(
                status="REQUIREMENT_VISUAL_REVIEW_REQUIRED",
                last_completed_stage="requirement_visual_review",
                mechanical_status_before_requirement_execution="TASK_CERTIFIED",
                requirement_execution={
                    **_REQUIREMENT_EXECUTION_PASS,
                    "status": "REVIEW_REQUIRED",
                    "passed": False,
                    "review_required_obligation_ids": ["reviewer.visual.a"],
                },
            ),
            _result(
                status="REQUIREMENT_VISUAL_REVIEW_REJECTED",
                failure_reason="VISUAL_OBLIGATION_REJECTED",
                last_completed_stage="requirement_visual_review",
                issues=[{"code": "VISUAL_OBLIGATION_REJECTED"}],
                visual_review={
                    "decision": "REVISE",
                    "reviewer": "registered-worker",
                    "reviewer_type": "worker_ai",
                    "path": "runs/r01/visual_review_approval.json",
                    "sha256": "0" * 64,
                    "certification_authority": "visual_obligations_only",
                },
            ),
            _result(
                status="WHOLE_MACHINE_TASK_PASSED",
                last_completed_stage="requirement_obligations_passed",
                mechanical_status_before_requirement_execution=(
                    "WHOLE_MACHINE_TASK_PASSED"
                ),
                requirement_execution=_REQUIREMENT_EXECUTION_PASS,
                whole_machine_results=[{"capsule_id": "drive"}],
            ),
            _result(
                status="ITERATION_SCOPE_FAILED",
                last_completed_stage="iteration_checks",
                iteration_diagnostic_status="ITERATION_DIAGNOSTIC_ONLY",
                iteration_verdict="FAIL",
            ),
        )
        for envelope in envelopes:
            with self.subTest(status=envelope["status"]):
                validate_harness_result(envelope)

    def test_finalize_success_path_document_satisfies_contract(self) -> None:
        # DEFECT #5 regression (2026-08-09): the first-ever fully successful
        # run was rejected at publish because finalize emitted a hardcoded
        # "0.1" schema_version.  Pin the real producer against the frozen
        # contract so producer/validator drift fails here, not in a live run.
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.finalize import (
            build_run_status_document,
        )

        document = build_run_status_document(
            summaries=[
                {
                    "overall_verdict": "TASK_CERTIFIED",
                    "rigid_attachment": {},
                    "collider_fidelity": {},
                    "whole_machine_dispatch": {},
                }
            ],
            ideal_reference_summary={},
            iteration_diagnostic_only=False,
            replacement_authorized=False,
            replacement_policy="test_policy",
        )

        self.assertEqual(document["schema_version"], HARNESS_RESULT_SCHEMA_ID)
        self.assertEqual(document["status"], "TASK_CERTIFIED")

        # The Loop merges requirement-execution fields before publish; the
        # merged document is what publish_run_status validates.
        published = {
            **document,
            "requirement_execution": _REQUIREMENT_EXECUTION_PASS,
            "mechanical_status_before_requirement_execution": "TASK_CERTIFIED",
        }
        validate_harness_result(published)

    def test_unknown_status_and_schema_version_fail_closed(self) -> None:
        with self.assertRaises(ContractViolation):
            validate_harness_result(_result(status="CERTIFIED_BY_AGENT"))
        with self.assertRaises(ContractViolation):
            validate_harness_result(
                {"schema_version": "some_other_schema", "status": "RUNNING"}
            )
        with self.assertRaises(ContractViolation):
            validate_harness_result({"status": "RUNNING"})

    def test_per_status_requirements_are_enforced(self) -> None:
        cases = (
            _result(status="RESOURCE_BLOCKED"),
            _result(
                status="RESOURCE_BLOCKED",
                resource_blocked=False,
                resource_kind="CPU_RAM_OR_DISK",
                retry_mode="RERUN_SAME_SUBMISSION",
                last_completed_stage="physical",
            ),
            _result(status="FAILED", last_completed_stage="x"),
            _result(status="TERMINATED", last_completed_stage="x"),
            _result(status="TASK_CERTIFIED", last_completed_stage="x"),
            _result(
                status="TASK_CERTIFIED",
                last_completed_stage="x",
                mechanical_status_before_requirement_execution="TASK_CERTIFIED",
                requirement_execution={"status": "PASS"},
            ),
            _result(
                status="REQUIREMENT_VISUAL_REVIEW_REJECTED",
                failure_reason="VISUAL_OBLIGATION_REJECTED",
                last_completed_stage="x",
                issues=[],
                visual_review={
                    "decision": "MAYBE",
                    "reviewer": "operator",
                    "reviewer_type": "human",
                    "path": "p",
                    "sha256": "0" * 64,
                },
            ),
        )
        for envelope in cases:
            with self.subTest(envelope=envelope):
                with self.assertRaises(ContractViolation):
                    validate_harness_result(envelope)

    def test_publish_writes_only_validated_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            value = publish_run_status(
                run, _result(status="RUNNING", last_completed_stage="intake")
            )
            written = json.loads(
                (run / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(written, value)
            with self.assertRaises(ContractViolation):
                publish_run_status(run, _result(status="INVENTED"))
            unchanged = json.loads(
                (run / "run_status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(unchanged["status"], "RUNNING")

    def test_schema_file_uses_only_interpreted_keywords(self) -> None:
        # A schema keyword the interpreter does not implement must fail the
        # check instead of being silently skipped.
        with self.assertRaises(SchemaCheckError):
            check_against_schema({"a": 1}, {"patternProperties": {}})
        # The shipped schema must therefore validate a good document without
        # tripping the unknown-keyword guard anywhere along the way.
        validate_harness_result(
            _result(status="RUNNING", last_completed_stage="intake")
        )
        self.assertEqual(
            harness_result_schema()["$id"], HARNESS_RESULT_SCHEMA_ID
        )


if __name__ == "__main__":
    unittest.main()
