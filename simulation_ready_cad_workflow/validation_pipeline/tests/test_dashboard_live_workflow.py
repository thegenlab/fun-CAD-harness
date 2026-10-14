from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.dashboard.live_workflow import (
    is_cold_start_run, live_project_metadata_paths, read_live_project,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import (
    ReadContext, _bound_available_visuals, group_runs_by_project, normalize_run, scan_runs,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def project(root, revision=1, **updates):
    directory = root / ".runtime_runs" / "projects" / "fixture"
    run = directory / "runs" / f"r{revision:02d}"
    run.mkdir(parents=True)
    state = {"schema_version": "cold_start_workflow_v1", "request": {"project_id": "fixture", "brief": "A fixture"},
             "phase": "author", "revision": revision, "elapsed_seconds": 12.0,
             "started_at": "2026-09-06T01:00:00+00:00", "updated_at": "2026-09-06T01:00:12+00:00", "events": []}
    state.update(updates)
    write(directory / "workflow.json", state)
    return directory, run, state


def turn(directory, number, inputs, outputs, cached=0):
    path = directory / "worker_turns" / f"t{number:03d}_author" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached}}) + "\n", encoding="utf-8")
    return path


class DashboardLiveWorkflowTests(unittest.TestCase):
    def test_external_author_does_not_change_stopped_native_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root, schema_version="slim_harness_workflow_v1",
                                           phase="simulation_timeout", stop_reason="SIMULATION_TIMEOUT")
            original = (directory / "workflow.json").read_bytes()
            write(directory / "external_author_runtime.json", {"schema_version": "external_author_runtime_v1",
                  "status": "running", "heartbeat_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": 50})
            workflow = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]
            self.assertEqual("author", workflow["phase"])
            self.assertTrue(workflow["is_running"])
            self.assertEqual(original, (directory / "workflow.json").read_bytes())

    def test_external_author_clock_skew_uses_local_receipt_and_expires(self):
        import os
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, _ = project(root, schema_version="slim_harness_workflow_v1",
                                        phase="simulation_timeout", stop_reason="SIMULATION_TIMEOUT")
            now = datetime.now(timezone.utc)
            path = directory / "external_author_runtime.json"
            write(path, {"schema_version": "external_author_runtime_v1", "status": "running",
                         "heartbeat_at": (now + timedelta(minutes=6)).isoformat()})
            self.assertTrue(read_live_project(workspace_root=root, run_dir=run)["live_workflow"]["is_running"])
            old = (now - timedelta(seconds=30)).timestamp()
            os.utime(path, (old, old))
            self.assertFalse(read_live_project(workspace_root=root, run_dir=run)["live_workflow"]["is_running"])

    def test_external_author_usage_is_bound_and_cache_is_not_double_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root)
            record = {"schema_version": "external_author_usage_v1", "project_id": "fixture",
                      "session_id": "actual-author", "model": "gpt-6-astra", "reasoning_effort": "max",
                      "usage": {"input_tokens": 1000, "cached_input_tokens": 900, "output_tokens": 50}}
            path = directory / "external-author-usage.json"
            write(path, record)
            usage = read_live_project(workspace_root=root, run_dir=run)["live_usage"]
            self.assertEqual(1050, usage["total_tokens"])
            self.assertEqual(900, usage["cached_input_tokens"])
            self.assertEqual("actual-author", usage["external_author"]["session_id"])
            record["project_id"] = "a-different-project"
            write(path, record)
            self.assertIsNone(read_live_project(workspace_root=root, run_dir=run)["live_usage"]["total_tokens"])

    def test_slim_progress_tracks_current_attempt_and_persisted_outcomes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            now = datetime.now(timezone.utc).isoformat()
            directory, run, state = project(root, schema_version="slim_harness_workflow_v1",
                                           phase="physics", active_test="alarm", updated_at=now)
            write(run / "design/scene.json", {"tests": [{"id": "clock"}, {"id": "alarm"}, {"id": "stop"}]})
            write(run / "attempts/a001/tests/stop/test-outcome.json", {"execution_status": "completed", "physical_pass": True})
            outcome = run / "attempts/a002/tests/clock/test-outcome.json"
            write(outcome, {"execution_status": "SIMULATION_TIMEOUT", "physical_pass": False})
            (run / "attempts/a002/tests/alarm").mkdir()
            write(directory / "workflow_runtime.json", {"schema_version": "workflow_runtime_v1",
                  "status": "running", "heartbeat_at": now, "phase": "physics", "revision": 1})
            with mock.patch("simulation_ready_cad_workflow.validation_pipeline.dashboard.live_workflow._process_matches", return_value=True):
                data = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]["test_progress"]
            self.assertEqual(("a002", "alarm", 1, 3), (data["attempt"], data["active_test"], data["returned"], data["total"]))
            self.assertEqual(["SIMULATION_TIMEOUT", "running", "not_run"], [test["execution_status"] for test in data["tests"]])
            write(run / "attempts/a002/tests/alarm/test-outcome.json", {"execution_status": "completed", "physical_pass": True, "geometry_audit": {"status": "pending_audit", "passed": False}})
            data = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]["test_progress"]
            self.assertEqual(2, data["returned"])
            self.assertEqual("pending_audit", data["tests"][1]["geometry_audit"]["status"])

    def test_interrupted_test_is_not_reported_as_unrun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root, schema_version="slim_harness_workflow_v1", phase="physics", active_test="clock", stop_reason="interrupted")
            write(run / "design/scene.json", {"tests": [{"id": "clock"}]})
            (run / "attempts/a001/tests/clock").mkdir(parents=True)
            data = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]["test_progress"]
            self.assertIsNone(data["active_test"])
            self.assertEqual("pending_result", data["tests"][0]["execution_status"])

    def test_workflow_identity_and_revision_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root)
            state["request"]["project_id"] = "different_project"
            write(directory / "workflow.json", state)
            self.assertFalse(is_cold_start_run(workspace_root=root, run_dir=run))
            self.assertEqual({}, read_live_project(workspace_root=root, run_dir=run)["live_workflow"])
            state["request"]["project_id"] = "fixture"
            state["revision"] = "not_a_revision"
            write(directory / "workflow.json", state)
            self.assertFalse(is_cold_start_run(workspace_root=root, run_dir=run))

    def test_unsubmitted_revision_is_discovered_and_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, _ = project(root, stop_reason="paused")
            self.assertTrue(is_cold_start_run(workspace_root=root, run_dir=run))
            self.assertFalse((directory / "project.json").exists())
            rows = scan_runs(root, workspace_root=root)
            self.assertEqual(1, len(rows))
            self.assertEqual("fixture", rows[0]["project_id"])
            self.assertFalse(rows[0]["eligible_pass"])
            data = normalize_run(run, workspace_root=root)
            self.assertEqual("fixture", data["overview"]["package"]["id"])
            self.assertEqual("stopped", data["live_workflow"]["status"])
            self.assertNotEqual("passed", data["run"]["status"])

    def test_current_workflow_revision_wins_over_older_pass_and_higher_future_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, _ = project(root, revision=18, stop_reason="paused")
            old = directory / "runs" / "r17"
            write(old / "run_status.json", {"status": "WHOLE_MACHINE_TASK_PASSED"})
            (directory / "runs" / "r19").mkdir()
            write(directory / "project.json", {"project_id": "fixture", "current_revision": "r17"})
            rows = scan_runs(root, workspace_root=root, include_history=True)
            result = group_runs_by_project(rows)[0]
            self.assertEqual(run.relative_to(root).as_posix(), result["default_run_id"])
            self.assertEqual("workflow_current", result["default_reason"])
            self.assertEqual(3, result["version_count"])

    def test_usage_counts_nonmonotonic_calls_once_and_retains_partial_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root)
            first = turn(directory, 1, 200, 20, 100)
            write(first.with_name("response.json"), {"usage": {"input_tokens": 200, "output_tokens": 20, "cached_input_tokens": 100}})
            state["events"] = [{"result": {"turn_directory": str(first.parent), "usage": {
                "input_tokens": 200, "output_tokens": 20, "cached_input_tokens": 100}}}]
            write(directory / "workflow.json", state)
            def usage():
                return read_live_project(workspace_root=root, run_dir=run)["live_usage"]
            self.assertEqual(220, usage()["total_tokens"])
            self.assertEqual(220, usage()["total_tokens"])
            second = turn(directory, 2, 30, 5)
            self.assertEqual(255, usage()["total_tokens"])
            self.assertEqual(100, usage()["cached_input_tokens"])
            self.assertEqual(2, usage()["completed_turn_count"])
            third = directory / "worker_turns" / "t003_author" / "events.jsonl"
            third.parent.mkdir()
            payload = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 7, "output_tokens": 2, "cached_input_tokens": 0}})
            third.write_text(payload[:20], encoding="utf-8")
            self.assertEqual(255, usage()["total_tokens"])
            self.assertTrue(usage()["partial"])
            with third.open("a", encoding="utf-8") as stream:
                stream.write(payload[20:] + "\n")
            self.assertEqual(264, usage()["total_tokens"])
            self.assertFalse(usage()["partial"])
            self.assertIn(second, live_project_metadata_paths(workspace_root=root, run_dir=run))
            fourth = turn(directory, 4, 8, 1)
            fourth.write_text(fourth.read_text(encoding="utf-8").rstrip(), encoding="utf-8")
            self.assertEqual(273, usage()["total_tokens"])
            with fourth.open("a", encoding="utf-8") as stream:
                stream.write("\n")
            self.assertEqual(273, usage()["total_tokens"])

    def test_cursor_result_usage_is_counted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, _ = project(root)
            path = directory / "worker_turns" / "t002_author" / "events.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "usage": {"inputTokens": 120, "outputTokens": 80,
                          "cacheReadTokens": 50, "cacheWriteTokens": 10},
            }) + "\n", encoding="utf-8")
            usage = read_live_project(workspace_root=root, run_dir=run)["live_usage"]
            self.assertEqual(260, usage["total_tokens"])
            self.assertEqual(180, usage["input_tokens"])
            self.assertEqual(50, usage["cached_input_tokens"])
            self.assertEqual(1, usage["completed_turn_count"])

    def test_missing_tokens_remain_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root)
            active = directory / "worker_turns" / "t001_author"
            active.mkdir(parents=True)
            state["active_turn"] = {"directory": str(active), "phase": "author"}
            write(directory / "workflow.json", state)
            data = read_live_project(workspace_root=root, run_dir=run)
            self.assertIsNone(data["live_usage"]["total_tokens"])
            self.assertTrue(data["live_usage"]["active_tokens_pending"])
            self.assertEqual("activity_unknown", data["live_workflow"]["status"])
            self.assertEqual(12, data["live_workflow"]["elapsed_seconds"])

    def test_construction_budget_is_experiment_elapsed_not_overnight_wall(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            started = datetime(2026, 9, 17, 2, 0, tzinfo=timezone.utc)
            ended = started + timedelta(hours=15, minutes=35)
            directory, run, _ = project(
                root,
                schema_version="slim_harness_workflow_v1",
                phase="budget_exhausted",
                stop_reason="budget_exhausted",
                started_at=started.isoformat(),
                updated_at=ended.isoformat(),
            )
            write(directory / "construction_budget.json", {
                "schema_version": "slim_construction_budget_v1",
                "active_seconds": 10692.0,
                "config": {"hours": 3, "simulation_timeout_seconds": 300, "max_tasks": 5,
                           "max_tests": 5, "max_geometry_units": 2048,
                           "max_geometry_units_per_body": 128, "summary_reserve_seconds": 120,
                           "audit_timeout_seconds": 600},
            })
            live = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]
            self.assertEqual(10800.0, live["elapsed_seconds"])
            self.assertIn(directory / "construction_budget.json",
                          live_project_metadata_paths(workspace_root=root, run_dir=run))

    def test_stopped_clock_does_not_advance_and_pid_identity_controls_running(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            directory, run, state = project(root, stop_reason="paused")
            stopped = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]
            self.assertEqual(12.0, stopped["elapsed_seconds"])
            self.assertFalse(stopped["is_running"])
            now = datetime.now(timezone.utc)
            state.pop("stop_reason")
            state["started_at"] = (now - timedelta(seconds=30)).isoformat()
            state["updated_at"] = (now - timedelta(seconds=3)).isoformat()
            write(directory / "workflow.json", state)
            runtime = {"schema_version": "cold_start_runtime_v1", "status": "running", "pid": 1234,
                       "process_started_at_epoch": 1.0, "heartbeat_at": now.isoformat(), "elapsed_seconds": 29.0,
                       "phase": "author", "revision": 1}
            write(directory / "workflow_runtime.json", runtime)
            module = "simulation_ready_cad_workflow.validation_pipeline.dashboard.live_workflow._process_matches"
            with mock.patch(module, return_value=True):
                live = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]
                self.assertTrue(live["is_running"])
                self.assertGreaterEqual(live["elapsed_seconds"], 30)
            with mock.patch(module, return_value=False):
                dead = read_live_project(workspace_root=root, run_dir=run)["live_workflow"]
                self.assertFalse(dead["is_running"])
                self.assertEqual(29, dead["elapsed_seconds"])
            runtime.update(status="stopped", elapsed_seconds=31)
            write(directory / "workflow_runtime.json", runtime)
            self.assertEqual(31, read_live_project(workspace_root=root, run_dir=run)["live_workflow"]["elapsed_seconds"])

    def test_media_binding_is_per_item_despite_incomplete_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            _directory, run, _ = project(root)
            visual = run / "visual"
            visual.mkdir()
            good = visual / "stage5_physics_evidence.glb"
            good.write_bytes(b"good")
            stale = visual / "stage5_stale.glb"
            stale.write_bytes(b"stale")
            summary = {"status": "INCOMPLETE", "source_run": run.relative_to(root).as_posix(),
                       "scene": {"path": good.relative_to(root).as_posix(), "sha256": hashlib.sha256(b"good").hexdigest()},
                       "stale": {"path": stale.relative_to(root).as_posix(), "sha256": "0" * 64},
                       "missing": {"path": (visual / "missing.mp4").relative_to(root).as_posix()}}
            self.assertEqual({good}, _bound_available_visuals(ReadContext(root, []), run, summary))
            write(visual / "dashboard_visuals.summary.json", summary)
            write(run / "traces" / "capsule" / "physical.trace.json", {})
            evidence = normalize_run(run, workspace_root=root)["evidence"]
            paths = {item["path"] for item in evidence}
            self.assertIn(good.relative_to(root).as_posix(), paths)
            self.assertNotIn(stale.relative_to(root).as_posix(), paths)


if __name__ == "__main__":
    unittest.main()
