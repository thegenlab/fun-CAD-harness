from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    JsonDocumentError,
    read_json,
)
from simulation_ready_cad_workflow.validation_pipeline.project_usage import (
    assert_project_usage_settled,
    automatic_usage_session_id,
    begin_project_usage,
    begin_run_usage,
    close_project_usage_wall_time,
    codex_project_token_counter,
    finish_project_usage,
    finish_run_usage,
    kimi_worker_token_counter,
    migrate_codex_visible_backfill_allocation,
    migrate_legacy_token_sessions,
    record_codex_turn_usage,
    rebuild_project_usage_summary,
)


class ProjectUsageTests(unittest.TestCase):
    def test_records_and_accumulates_project_wall_time_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            begin_project_usage(
                workspace_root=root,
                project_id="hinged_toolbox_v1",
                session_id="toolbox-r01-loop",
                run_id="toolbox-r01",
                token_counter=100,
                token_counter_source="codex_goal_counter",
                clock=lambda: 1_000.0,
            )
            first = finish_project_usage(
                workspace_root=root,
                session_id="toolbox-r01-loop",
                token_counter=145,
                clock=lambda: 1_012.5,
            )
            self.assertEqual(first["total_wall_time_seconds"], 12.5)
            self.assertEqual(first["total_tokens"], 45)

            begin_project_usage(
                workspace_root=root,
                project_id="hinged_toolbox_v1",
                session_id="toolbox-r02-loop",
                run_id="toolbox-r02",
                token_counter=200,
                token_counter_source="codex_goal_counter",
                clock=lambda: 2_000.0,
            )
            second = finish_project_usage(
                workspace_root=root,
                session_id="toolbox-r02-loop",
                token_counter=280,
                clock=lambda: 2_007.25,
            )
            self.assertEqual(second["completed_session_count"], 2)
            self.assertEqual(second["total_wall_time_seconds"], 19.75)
            self.assertEqual(second["total_tokens"], 125)
            self.assertEqual(second["run_ids"], ["toolbox-r01", "toolbox-r02"])

    def test_begin_and_finish_are_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            arguments = {
                "workspace_root": root,
                "project_id": "four_bar_wiper_v1",
                "session_id": "wiper-r01-loop",
                "run_id": "wiper-r01",
                "token_counter": 500,
                "token_counter_source": "codex_goal_counter",
            }
            first = begin_project_usage(**arguments, clock=lambda: 4_000.0)
            repeated = begin_project_usage(**arguments, clock=lambda: 9_000.0)
            self.assertEqual(first, repeated)
            finished = finish_project_usage(
                workspace_root=root,
                session_id="wiper-r01-loop",
                token_counter=550,
                clock=lambda: 4_003.0,
            )
            repeated_finish = finish_project_usage(
                workspace_root=root,
                session_id="wiper-r01-loop",
                token_counter=550,
                clock=lambda: 9_000.0,
            )
            self.assertEqual(finished, repeated_finish)

    def test_rejects_counter_or_clock_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            begin_project_usage(
                workspace_root=root,
                project_id="rack_pinion_gate_v1",
                session_id="gate-r01-loop",
                run_id="gate-r01",
                token_counter=100,
                token_counter_source="codex_goal_counter",
                clock=lambda: 100.0,
            )
            with self.assertRaisesRegex(JsonDocumentError, "moved backwards"):
                finish_project_usage(
                    workspace_root=root,
                    session_id="gate-r01-loop",
                    token_counter=99,
                    clock=lambda: 101.0,
                )
            with self.assertRaisesRegex(JsonDocumentError, "clock moved backwards"):
                finish_project_usage(
                    workspace_root=root,
                    session_id="gate-r01-loop",
                    token_counter=101,
                    clock=lambda: 99.0,
                )

    def test_running_session_is_visible_in_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            begin_project_usage(
                workspace_root=root,
                project_id="pedal_trash_can_v1",
                session_id="trash-r01-loop",
                run_id="trash-r01",
                token_counter=10,
                token_counter_source="codex_goal_counter",
                clock=lambda: 200.0,
            )
            summary = rebuild_project_usage_summary(
                workspace_root=root,
                project_id="pedal_trash_can_v1",
            )
            self.assertEqual(summary["status"], "RUNNING")
            self.assertEqual(summary["running_session_count"], 1)
            project_path = (
                root
                / ".runtime_runs"
                / "project_usage"
                / "projects"
                / "pedal_trash_can_v1.json"
            )
            self.assertEqual(read_json(project_path), summary)

    def test_rejects_session_identity_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            begin_project_usage(
                workspace_root=root,
                project_id="clock_v1",
                session_id="clock-loop",
                run_id="clock-r01",
                token_counter=1,
                token_counter_source="codex_goal_counter",
                clock=lambda: 100.0,
            )
            with self.assertRaisesRegex(JsonDocumentError, "different identity"):
                begin_project_usage(
                    workspace_root=root,
                    project_id="clock_v1",
                    session_id="clock-loop",
                    run_id="clock-r02",
                    token_counter=1,
                    token_counter_source="codex_goal_counter",
                    clock=lambda: 100.0,
                )

    def test_automatic_run_closes_wall_time_then_requires_token_settlement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            session_id = automatic_usage_session_id("clock_v1", "r03")
            begin_project_usage(
                workspace_root=root,
                project_id="clock_v1",
                session_id=session_id,
                run_id="clock_v1-r03",
                token_counter=1_000,
                token_counter_source="codex_goal_counter",
                clock=lambda: 100.0,
            )

            pending = close_project_usage_wall_time(
                workspace_root=root,
                project_id="clock_v1",
                revision="r03",
                run_status="FAILED",
                clock=lambda: 112.5,
            )

            self.assertEqual(pending["status"], "TOKEN_PENDING")
            self.assertEqual(pending["pending_token_session_count"], 1)
            self.assertEqual(
                pending["pending_token_sessions"][0]["wall_time_seconds"],
                12.5,
            )
            with self.assertRaisesRegex(JsonDocumentError, "must be settled"):
                assert_project_usage_settled(
                    workspace_root=root,
                    project_id="clock_v1",
                )

            completed = finish_project_usage(
                workspace_root=root,
                session_id=session_id,
                token_counter=1_075,
                clock=lambda: 999.0,
            )

            self.assertEqual(completed["status"], "CURRENT")
            self.assertEqual(completed["pending_token_session_count"], 0)
            self.assertEqual(completed["total_wall_time_seconds"], 12.5)
            self.assertEqual(completed["total_tokens"], 75)
            assert_project_usage_settled(
                workspace_root=root,
                project_id="clock_v1",
            )


if __name__ == "__main__":
    unittest.main()



class KimiWireTokenCounterTests(unittest.TestCase):
    def _fixture(self, sessions_root: Path) -> None:
        bucket = sessions_root / "wd_demo_project_ab12cd"
        worker = bucket / "session_worker1"
        (worker / "agents" / "a1").mkdir(parents=True)
        (worker / "state.json").write_text(
            json.dumps(
                {
                    "title": (
                        "You are the sole mechanical-design agent "
                        "and Harness operator"
                    )
                }
            ),
            encoding="utf-8",
        )
        records = [
            {
                "type": "usage.record",
                "usageScope": "turn",
                "usage": {
                    "inputOther": 10,
                    "output": 5,
                    "inputCacheRead": 100,
                    "inputCacheCreation": 3,
                },
            },
            {
                "type": "usage.record",
                "usageScope": "session",
                "usage": {
                    "inputOther": 999,
                    "output": 999,
                    "inputCacheRead": 999,
                    "inputCacheCreation": 999,
                },
            },
            {
                "type": "context.append_loop_event",
                "event": {
                    "type": "step.end",
                    "usage": {
                        "inputOther": 7,
                        "output": 7,
                        "inputCacheRead": 7,
                        "inputCacheCreation": 7,
                    },
                },
            },
        ]
        (worker / "agents" / "a1" / "wire.jsonl").write_text(
            "\n".join(json.dumps(item) for item in records),
            encoding="utf-8",
        )
        auditor = bucket / "session_auditor1"
        (auditor / "agents" / "a1").mkdir(parents=True)
        (auditor / "state.json").write_text(
            json.dumps({"title": "Read-only audit, no edits"}),
            encoding="utf-8",
        )
        (auditor / "agents" / "a1" / "wire.jsonl").write_text(
            json.dumps(
                {
                    "type": "usage.record",
                    "usageScope": "turn",
                    "usage": {
                        "inputOther": 50,
                        "output": 50,
                        "inputCacheRead": 50,
                        "inputCacheCreation": 50,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_sums_only_worker_turn_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sessions_root = Path(temporary)
            self._fixture(sessions_root)
            self.assertEqual(
                kimi_worker_token_counter(sessions_root=sessions_root),
                118,
            )

    def test_scopes_to_workspace_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            sessions_root = base / "sessions"
            self._fixture(sessions_root)
            workspace = base / "demo_project"
            workspace.mkdir()
            self.assertEqual(
                kimi_worker_token_counter(
                    workspace_root=workspace,
                    sessions_root=sessions_root,
                ),
                118,
            )
            other = base / "some_other_checkout"
            other.mkdir()
            self.assertIsNone(
                kimi_worker_token_counter(
                    workspace_root=other,
                    sessions_root=sessions_root,
                )
            )

    def test_returns_none_without_matching_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            self.assertIsNone(
                kimi_worker_token_counter(sessions_root=Path(temporary))
            )


class RunUsageWrapperTests(unittest.TestCase):
    def _canonical_run_dir(self, root: Path) -> Path:
        run_dir = (
            root
            / ".runtime_runs"
            / "projects"
            / "demo_project"
            / "runs"
            / "r01"
        )
        run_dir.mkdir(parents=True)
        return run_dir

    def test_begin_and_finish_record_wall_time_and_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._canonical_run_dir(root)
            begin_run_usage(
                workspace_root=root,
                project_id="demo_project",
                revision="r01",
                run_id="demo_project-r01",
                token_counter=1000,
            )
            finish_run_usage(
                run_dir,
                run_status="TASK_CERTIFIED",
                token_counter=1450,
            )
            summary = read_json(
                root
                / ".runtime_runs"
                / "project_usage"
                / "projects"
                / "demo_project.json"
            )
            self.assertEqual(summary["status"], "CURRENT")
            self.assertEqual(summary["completed_session_count"], 1)
            self.assertEqual(summary["total_tokens"], 450)
            self.assertEqual(summary["run_ids"], ["demo_project-r01"])
            self.assertGreaterEqual(summary["total_wall_time_seconds"], 0.0)

    def test_missing_counter_falls_back_to_wall_time_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._canonical_run_dir(root)
            begin_run_usage(
                workspace_root=root,
                project_id="demo_project",
                revision="r01",
                run_id="demo_project-r01",
                token_counter=None,
            )
            finish_run_usage(
                run_dir,
                run_status="TASK_FAILED",
                token_counter=None,
            )
            summary = read_json(
                root
                / ".runtime_runs"
                / "project_usage"
                / "projects"
                / "demo_project.json"
            )
            self.assertEqual(summary["status"], "TOKEN_PENDING")
            self.assertEqual(summary["pending_token_session_count"], 1)

    def test_finish_without_begin_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = self._canonical_run_dir(root)
            finish_run_usage(
                run_dir,
                run_status="TASK_CERTIFIED",
                token_counter=10,
            )
            self.assertFalse(
                (root / ".runtime_runs" / "project_usage").exists()
            )

    def test_wrappers_swallow_garbage_inputs(self) -> None:
        finish_run_usage("/nonexistent/run", run_status="X", token_counter=1)
        begin_run_usage(
            workspace_root="/nonexistent",
            project_id="bad id!!",
            revision="r01",
            run_id="x",
            token_counter=1,
        )


class CodexLedgerTests(unittest.TestCase):
    def test_records_cumulative_session_usage_as_monotonic_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = record_codex_turn_usage(
                workspace_root=root,
                project_id="ratchet",
                input_tokens=1_000,
                output_tokens=100,
                context="r01-turn-0001",
            )
            second = record_codex_turn_usage(
                workspace_root=root,
                project_id="ratchet",
                input_tokens=1_500,
                output_tokens=150,
                context="r01-turn-0002",
            )
            repeated = record_codex_turn_usage(
                workspace_root=root,
                project_id="ratchet",
                input_tokens=1_500,
                output_tokens=150,
                context="r01-turn-0002",
            )

            self.assertEqual(first["total_tokens"], 1_100)
            self.assertEqual(second["total_tokens"], 1_650)
            self.assertEqual(repeated["total_tokens"], 1_650)
            self.assertEqual(repeated["entries"][-1]["total_tokens"], 0)
            self.assertEqual(
                codex_project_token_counter(
                    workspace_root=root,
                    project_id="ratchet",
                ),
                1_650,
            )

    def test_rejects_a_cumulative_counter_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_codex_turn_usage(
                workspace_root=root,
                project_id="ratchet",
                input_tokens=1_000,
                output_tokens=100,
            )
            with self.assertRaisesRegex(JsonDocumentError, "moved backwards"):
                record_codex_turn_usage(
                    workspace_root=root,
                    project_id="ratchet",
                    input_tokens=999,
                    output_tokens=100,
                )

    def test_records_independent_visible_role_counters_without_cross_talk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record_codex_turn_usage(
                workspace_root=root,
                project_id="visible_roles",
                counter_id="worker-thread",
                input_tokens=1_000,
                output_tokens=100,
            )
            record_codex_turn_usage(
                workspace_root=root,
                project_id="visible_roles",
                counter_id="reviewer-thread",
                input_tokens=400,
                output_tokens=40,
            )
            final = record_codex_turn_usage(
                workspace_root=root,
                project_id="visible_roles",
                counter_id="worker-thread",
                input_tokens=1_500,
                output_tokens=150,
            )

            self.assertEqual(final["total_tokens"], 2_090)
            self.assertEqual(
                final["counters"]["worker-thread"]["total_tokens"], 1_650
            )
            self.assertEqual(
                final["counters"]["reviewer-thread"]["total_tokens"], 440
            )

    def test_initial_visible_history_is_unallocated_not_charged_to_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            begin_project_usage(
                workspace_root=root,
                project_id="historical_visible",
                session_id=automatic_usage_session_id("historical_visible", "r01"),
                run_id="historical_visible-r01",
                token_counter=0,
                token_counter_source="codex_turn_usage",
                clock=lambda: 10.0,
            )
            finish_project_usage(
                workspace_root=root,
                session_id=automatic_usage_session_id("historical_visible", "r01"),
                token_counter=0,
                clock=lambda: 20.0,
            )
            begin_project_usage(
                workspace_root=root,
                project_id="historical_visible",
                session_id=automatic_usage_session_id("historical_visible", "r02"),
                run_id="historical_visible-r02",
                token_counter=0,
                token_counter_source="codex_turn_usage",
                clock=lambda: 30.0,
            )
            record_codex_turn_usage(
                workspace_root=root,
                project_id="historical_visible",
                counter_id="worker-thread",
                input_tokens=1_000,
                output_tokens=100,
                allocate_delta=False,
            )
            record_codex_turn_usage(
                workspace_root=root,
                project_id="historical_visible",
                counter_id="reviewer-thread",
                input_tokens=400,
                output_tokens=40,
                allocate_delta=False,
            )
            record_codex_turn_usage(
                workspace_root=root,
                project_id="historical_visible",
                counter_id="worker-thread",
                input_tokens=1_500,
                output_tokens=150,
            )

            self.assertEqual(
                codex_project_token_counter(
                    workspace_root=root, project_id="historical_visible"
                ),
                550,
            )
            summary = rebuild_project_usage_summary(
                workspace_root=root, project_id="historical_visible"
            )
            self.assertEqual(summary["unallocated_backfill_tokens"], 1_540)
            self.assertEqual(summary["total_tokens"], 2_090)
            self.assertEqual(summary["running_sessions"][0]["total_tokens"], 550)

    def test_migrates_legacy_visible_backfill_and_session_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project_id = "legacy_visible"
            begin_project_usage(
                workspace_root=root,
                project_id=project_id,
                session_id=automatic_usage_session_id(project_id, "r01"),
                run_id=f"{project_id}-r01",
                token_counter=0,
                token_counter_source="codex_turn_usage",
                clock=lambda: 10.0,
            )
            finish_project_usage(
                workspace_root=root,
                session_id=automatic_usage_session_id(project_id, "r01"),
                token_counter=0,
                clock=lambda: 20.0,
            )
            begin_project_usage(
                workspace_root=root,
                project_id=project_id,
                session_id=automatic_usage_session_id(project_id, "r02"),
                run_id=f"{project_id}-r02",
                token_counter=0,
                token_counter_source="codex_turn_usage",
                clock=lambda: 30.0,
            )
            ledger_path = (
                root
                / ".runtime_runs"
                / "project_usage"
                / "codex_ledger"
                / f"{project_id}.json"
            )
            ledger_path.parent.mkdir(parents=True)
            ledger_path.write_text(
                json.dumps(
                    {
                        "schema_version": (
                            "simulation_ready_cad_codex_token_ledger_v1"
                        ),
                        "project_id": project_id,
                        "total_tokens": 1_650,
                        "counters": {
                            "codex-app-worker": {
                                "last_cumulative_input_tokens": 1_500,
                                "last_cumulative_output_tokens": 150,
                                "total_tokens": 1_650,
                            }
                        },
                        "entries": [
                            {
                                "recorded_at": "2026-08-24T00:00:00+00:00",
                                "cumulative_input_tokens": 1_000,
                                "cumulative_output_tokens": 100,
                                "input_tokens": 1_000,
                                "output_tokens": 100,
                                "total_tokens": 1_100,
                                "counter_id": "codex-app-worker",
                                "context": "r02-worker-visible-role",
                            },
                            {
                                "recorded_at": "2026-08-24T00:01:00+00:00",
                                "cumulative_input_tokens": 1_500,
                                "cumulative_output_tokens": 150,
                                "input_tokens": 500,
                                "output_tokens": 50,
                                "total_tokens": 550,
                                "counter_id": "codex-app-worker",
                                "context": "r02-worker-visible-role",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            running_path = (
                root
                / ".runtime_runs"
                / "project_usage"
                / "sessions"
                / f"{automatic_usage_session_id(project_id, 'r02')}.json"
            )

            report = migrate_codex_visible_backfill_allocation(
                workspace_root=root,
                project_id=project_id,
                apply=True,
            )

            self.assertEqual(report["status"], "MIGRATED")
            self.assertEqual(report["unallocated_backfill_tokens"], 1_100)
            self.assertEqual(report["allocated_total_tokens"], 550)
            migrated = read_json(running_path)
            self.assertEqual(migrated["token_counter_start"], 0)
            summary = rebuild_project_usage_summary(
                workspace_root=root, project_id=project_id
            )
            self.assertEqual(summary["running_sessions"][0]["total_tokens"], 550)
            self.assertEqual(summary["total_tokens"], 1_650)


class LegacyTokenMigrationTests(unittest.TestCase):
    def _seed_legacy_project(self, root: Path) -> None:
        # r01 was left RUNNING on the Kimi counter (no honest end time).
        begin_project_usage(
            workspace_root=root,
            project_id="legacy_project",
            session_id=automatic_usage_session_id("legacy_project", "r01"),
            run_id="legacy_project-r01",
            token_counter=100,
            token_counter_source="kimi_wire_worker_turn_usage",
            clock=lambda: 100.0,
        )
        # r02 closed its wall time but awaits a counter that no longer
        # exists.
        begin_project_usage(
            workspace_root=root,
            project_id="legacy_project",
            session_id=automatic_usage_session_id("legacy_project", "r02"),
            run_id="legacy_project-r02",
            token_counter=None,
            token_counter_source="kimi_wire_worker_turn_usage",
            clock=lambda: 200.0,
        )
        close_project_usage_wall_time(
            workspace_root=root,
            project_id="legacy_project",
            revision="r02",
            run_status="FAILED",
            clock=lambda: 260.0,
        )
        # r03 is a healthy Codex-sourced session and must not be touched.
        begin_project_usage(
            workspace_root=root,
            project_id="legacy_project",
            session_id=automatic_usage_session_id("legacy_project", "r03"),
            run_id="legacy_project-r03",
            token_counter=0,
            token_counter_source="codex_turn_usage",
            clock=lambda: 300.0,
        )
        finish_project_usage(
            workspace_root=root,
            session_id=automatic_usage_session_id("legacy_project", "r03"),
            token_counter=500,
            clock=lambda: 320.0,
        )

    def _session(self, root: Path, revision: str) -> dict:
        return read_json(
            root
            / ".runtime_runs"
            / "project_usage"
            / "sessions"
            / f"{automatic_usage_session_id('legacy_project', revision)}.json"
        )

    def test_dry_run_reports_without_changing_sessions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._seed_legacy_project(root)
            report = migrate_legacy_token_sessions(
                workspace_root=root,
                project_id="legacy_project",
                apply=False,
            )
            self.assertFalse(report["applied"])
            self.assertEqual(report["migrated_session_count"], 2)
            self.assertEqual(self._session(root, "r01")["status"], "RUNNING")
            self.assertEqual(
                self._session(root, "r02")["status"],
                "AWAITING_TOKEN_COUNTER",
            )

    def test_apply_closes_legacy_sessions_and_settles_the_project(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._seed_legacy_project(root)
            report = migrate_legacy_token_sessions(
                workspace_root=root,
                project_id="legacy_project",
                apply=True,
                clock=lambda: 999.0,
            )
            self.assertTrue(report["applied"])
            self.assertEqual(report["migrated_session_count"], 2)

            running = self._session(root, "r01")
            self.assertEqual(running["status"], "CLOSED_UNMEASURED")
            self.assertTrue(running["wall_time_measured"])
            self.assertEqual(running["wall_time_seconds"], 100.0)
            self.assertEqual(
                running["wall_time_basis"], "next_session_started_at"
            )

            awaiting = self._session(root, "r02")
            self.assertEqual(awaiting["status"], "CLOSED_UNMEASURED")
            self.assertTrue(awaiting["wall_time_measured"])
            self.assertEqual(awaiting["wall_time_seconds"], 60.0)

            codex = self._session(root, "r03")
            self.assertEqual(codex["status"], "COMPLETED")
            self.assertEqual(codex["total_tokens"], 500)

            summary = rebuild_project_usage_summary(
                workspace_root=root, project_id="legacy_project"
            )
            self.assertEqual(summary["unmeasured_session_count"], 2)
            self.assertEqual(summary["total_tokens"], 500)
            self.assertEqual(summary["unmeasured_wall_time_seconds"], 160.0)
            self.assertEqual(summary["total_wall_time_seconds"], 180.0)
            assert_project_usage_settled(
                workspace_root=root, project_id="legacy_project"
            )

    def test_second_apply_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._seed_legacy_project(root)
            migrate_legacy_token_sessions(
                workspace_root=root, project_id="legacy_project", apply=True
            )
            repeat = migrate_legacy_token_sessions(
                workspace_root=root, project_id="legacy_project", apply=True
            )
            self.assertEqual(repeat["migrated_session_count"], 0)

    def test_migration_ignores_other_legacy_counter_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            begin_project_usage(
                workspace_root=root,
                project_id="legacy_project",
                session_id=automatic_usage_session_id("legacy_project", "r01"),
                run_id="legacy_project-r01",
                token_counter=10,
                token_counter_source="codex_goal_counter",
                clock=lambda: 100.0,
            )

            report = migrate_legacy_token_sessions(
                workspace_root=root,
                project_id="legacy_project",
                apply=True,
            )

            self.assertEqual(report["migrated_session_count"], 0)
            self.assertEqual(self._session(root, "r01")["status"], "RUNNING")
