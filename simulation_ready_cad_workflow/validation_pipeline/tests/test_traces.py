"""Tests for the three sanctioned trace families."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.contracts.harness_result import (
    publish_run_status,
)
from simulation_ready_cad_workflow.validation_pipeline.contracts.traces import (
    HARNESS_EXECUTION_TRACE_SCHEMA,
    MAINTENANCE_TRACE_SCHEMA,
    TraceContractError,
    append_execution_trace,
    append_maintenance_record,
    execution_trace_path,
    load_execution_trace,
    load_maintenance_trace,
    load_worker_action_trace,
    maintenance_trace_path,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)


class ExecutionTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._scratch = tempfile.TemporaryDirectory()
        self.run = Path(self._scratch.name) / "runs" / "r01"
        self.run.mkdir(parents=True)

    def tearDown(self) -> None:
        self._scratch.cleanup()

    def test_appends_form_a_verifiable_chain(self) -> None:
        append_execution_trace(
            self.run, status="RUNNING", last_completed_stage="intake"
        )
        append_execution_trace(
            self.run,
            status="TASK_CERTIFIED",
            last_completed_stage="finalize",
            detail={"outcome_kind": "PASS"},
        )
        trace = load_execution_trace(self.run)
        self.assertEqual(
            trace["schema_version"], HARNESS_EXECUTION_TRACE_SCHEMA
        )
        self.assertEqual(
            [entry["status"] for entry in trace["entries"]],
            ["RUNNING", "TASK_CERTIFIED"],
        )
        self.assertEqual(
            trace["entries"][1]["previous_entry_sha256"],
            trace["entries"][0]["entry_sha256"],
        )

    def test_tampered_entry_fails_closed(self) -> None:
        append_execution_trace(
            self.run, status="RUNNING", last_completed_stage="intake"
        )
        path = execution_trace_path(self.run)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["entries"][0]["status"] = "TASK_CERTIFIED"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(TraceContractError) as ctx:
            load_execution_trace(self.run)
        self.assertEqual(ctx.exception.code, "TRACE_CHAIN_BROKEN")
        # Appending to a broken chain must refuse too.
        with self.assertRaises(TraceContractError):
            append_execution_trace(
                self.run, status="RUNNING", last_completed_stage=None
            )

    def test_publish_run_status_journals_every_publication(self) -> None:
        publish_run_status(
            self.run,
            {
                "schema_version": (
                    "simulation_ready_cad_harness_result_v2"
                ),
                "status": "RUNNING",
                "last_completed_stage": "intake",
            },
        )
        publish_run_status(
            self.run,
            {
                "schema_version": (
                    "simulation_ready_cad_harness_result_v2"
                ),
                "status": "FAILED",
                "failure_reason": "INTAKE_FAILED",
                "last_completed_stage": "intake",
            },
        )
        trace = load_execution_trace(self.run)
        self.assertEqual(len(trace["entries"]), 2)
        self.assertEqual(trace["entries"][1]["status"], "FAILED")
        self.assertEqual(
            trace["entries"][1]["detail"]["failure_reason"], "INTAKE_FAILED"
        )


class MaintenanceTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._scratch = tempfile.TemporaryDirectory()
        self.project = Path(self._scratch.name) / "projects" / "demo"
        self.project.mkdir(parents=True)

    def tearDown(self) -> None:
        self._scratch.cleanup()

    def test_records_require_kind_reason_and_visibility(self) -> None:
        entry = append_maintenance_record(
            self.project,
            actor="operator",
            intervention_kind="harness_defect_fix",
            reason="physical queue mis-classified OOM as design failure",
            worker_visible_change="none",
            evidence={"commit": "abc123"},
        )
        self.assertEqual(entry["sequence"], 1)
        with self.assertRaises(TraceContractError):
            append_maintenance_record(
                self.project,
                actor="operator",
                intervention_kind="not_a_kind",
                reason="x",
            )
        with self.assertRaises(TraceContractError):
            append_maintenance_record(
                self.project,
                actor="operator",
                intervention_kind="regression_run",
                reason="   ",
            )
        with self.assertRaises(TraceContractError):
            append_maintenance_record(
                self.project,
                actor="operator",
                intervention_kind="controlled_communication",
                reason="unblock frozen worker",
                worker_visible_change="",
            )

    def test_chain_and_empty_load(self) -> None:
        self.assertEqual(load_maintenance_trace(self.project)["entries"], [])
        append_maintenance_record(
            self.project,
            actor="operator",
            intervention_kind="regression_run",
            reason="re-run r34 after harness fix",
        )
        append_maintenance_record(
            self.project,
            actor="reviewer",
            intervention_kind="template_publication",
            reason="publish task_suite 1.1.0",
            worker_visible_change=(
                "new public template version visible via catalog"
            ),
        )
        trace = load_maintenance_trace(self.project)
        self.assertEqual(trace["schema_version"], MAINTENANCE_TRACE_SCHEMA)
        self.assertEqual(len(trace["entries"]), 2)
        path = maintenance_trace_path(self.project)
        document = json.loads(path.read_text(encoding="utf-8"))
        document["entries"][0]["reason"] = "rewritten"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(TraceContractError):
            load_maintenance_trace(self.project)


class WorkerActionTraceTest(unittest.TestCase):
    def setUp(self) -> None:
        self._scratch = tempfile.TemporaryDirectory()
        self.project = Path(self._scratch.name) / "projects" / "demo"
        self.events = self.project / "react_loop" / "events"
        self.events.mkdir(parents=True)

    def tearDown(self) -> None:
        self._scratch.cleanup()

    def _write_event(
        self, sequence: int, previous: str | None, event_type: str
    ) -> str:
        core = {
            "schema_version": "simulation_ready_cad_react_loop_event_v2",
            "sequence": sequence,
            "previous_event_sha256": previous,
            "event_type": event_type,
            "project_id": "demo",
            "revision": "r01",
            "loop_status": "TASK_ACTIVE",
            "created_at": "2026-08-11T00:00:00+00:00",
            "detail": {},
        }
        event = {**core, "event_sha256": canonical_sha256(core)}
        (self.events / f"e{sequence:06d}.json").write_text(
            json.dumps(event), encoding="utf-8"
        )
        return event["event_sha256"]

    def test_verified_view_over_the_loop_journal(self) -> None:
        head = self._write_event(1, None, "LOOP_CREATED")
        head = self._write_event(2, head, "TURN_STARTED")
        trace = load_worker_action_trace(self.project)
        self.assertEqual(trace["event_count"], 2)
        self.assertEqual(trace["head_sha256"], head)
        self.assertEqual(
            [event["event_type"] for event in trace["events"]],
            ["LOOP_CREATED", "TURN_STARTED"],
        )

    def test_real_frozen_project_journal_verifies(self) -> None:
        frozen = Path(
            ".runtime_runs/projects/"
            "mechanical_alarm_clock_visible_selfloop_20260809_codex_c6"
        )
        if not (frozen / "react_loop" / "events").is_dir():
            self.skipTest("frozen clock project not available")
        trace = load_worker_action_trace(frozen)
        self.assertGreater(trace["event_count"], 100)

    def test_tampered_event_fails_closed(self) -> None:
        head = self._write_event(1, None, "LOOP_CREATED")
        self._write_event(2, head, "TURN_STARTED")
        path = self.events / "e000002.json"
        event = json.loads(path.read_text(encoding="utf-8"))
        event["event_type"] = "TURN_COMPLETED"
        path.write_text(json.dumps(event), encoding="utf-8")
        with self.assertRaises(TraceContractError) as ctx:
            load_worker_action_trace(self.project)
        self.assertEqual(ctx.exception.code, "TRACE_CHAIN_BROKEN")

    def test_broken_link_fails_closed(self) -> None:
        self._write_event(1, None, "LOOP_CREATED")
        self._write_event(2, "0" * 64, "TURN_STARTED")
        with self.assertRaises(TraceContractError):
            load_worker_action_trace(self.project)


if __name__ == "__main__":
    unittest.main()
