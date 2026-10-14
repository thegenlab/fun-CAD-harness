"""Run monitor overview and the serial launch queue."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.dashboard import launcher
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import write_json_atomic
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli import create_project

from .test_dashboard_launch import AUTH, _call, _launch_server, _popen


def _free(project: str) -> dict:
    return {"mode": "free", "project": project, "task": f"Build {project}.", "provider": "codex", "tier": "light"}


class LaunchQueueTest(unittest.TestCase):
    def test_queue_waits_for_running_dashboard_runs_and_starts_in_order(self) -> None:
        alive: set[str] = set()
        with _launch_server() as (root, server), _popen() as calls, \
                mock.patch.object(launcher, "_alive", lambda record: record.get("project") in alive):
            self.assertEqual(_call(server, "POST", "/api/launch", _free("first"), AUTH)[0], 200)
            alive.add("first")
            for project, position in (("second", 1), ("third", 2)):
                status, queued = _call(server, "POST", "/api/launch/queue", _free(project), AUTH)
                self.assertEqual((status, queued["position"], queued["status"]), (200, position, "queued"))
            self.assertIsNone(launcher.advance_queue(root, limit=server.launch_limit))
            alive.clear()
            started = launcher.advance_queue(root, limit=server.launch_limit)
            self.assertEqual((started["status"], started["run_id"]), ("started", ".runtime_runs/projects/second/runs/r01"))
            alive.add("second")
            self.assertIsNone(launcher.advance_queue(root, limit=server.launch_limit))
            self.assertEqual([item["project"] for item in launcher.queue(root)], ["third"])
            alive.clear()
            self.assertEqual(launcher.advance_queue(root, limit=server.launch_limit)["status"], "started")
            projects = [call.argv[call.argv.index("--project") + 1] for call in calls]
            self.assertEqual(projects, ["first", "second", "third"])
            self.assertEqual(launcher.queue(root), [])

    def test_duplicates_are_refused_and_waiting_entries_can_be_cancelled(self) -> None:
        with _launch_server() as (root, server):
            status, queued = _call(server, "POST", "/api/launch/queue", _free("later"), AUTH)
            self.assertEqual(status, 200)
            status, error = _call(server, "POST", "/api/launch/queue", _free("later"), AUTH)
            self.assertEqual((status, error["error"]), (409, "LAUNCH_PROJECT_QUEUED"))
            self.assertEqual(_call(server, "POST", "/api/launch/queue/cancel", {"id": queued["id"]}, AUTH)[0], 200)
            self.assertEqual(launcher.queue(root), [])
            status, error = _call(server, "POST", "/api/launch/queue/cancel", {"id": queued["id"]}, AUTH)
            self.assertEqual((status, error["error"]), (409, "LAUNCH_QUEUE_ITEM"))

    def test_an_entry_that_cannot_start_is_kept_with_its_reason(self) -> None:
        with _launch_server() as (root, server), _popen():
            _call(server, "POST", "/api/launch/queue", _free("taken"), AUTH)
            (root / ".runtime_runs/projects/taken").mkdir(parents=True)
            failed = launcher.advance_queue(root, limit=1)
            self.assertEqual(failed["status"], "failed")
            self.assertIn("LAUNCH_PROJECT_EXISTS", failed["error"])
            view = launcher.queue(root)
            self.assertEqual([(item["project"], item["status"], item["position"]) for item in view], [("taken", "failed", None)])
            self.assertEqual(_call(server, "POST", "/api/launch/queue/cancel", {"id": view[0]["id"]}, AUTH)[0], 200)

    def test_an_entry_left_starting_by_a_stopped_dashboard_is_settled(self) -> None:
        with _launch_server() as (root, server), _popen() as calls:
            for project in ("interrupted", "launched", "exited", "next"):
                _call(server, "POST", "/api/launch/queue", _free(project), AUTH)
            path = root / ".runtime_runs/dashboard_launches/queue.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            for item in saved["items"][:3]:
                item.update(status="starting", starting_at=launcher._now())
            write_json_atomic(path, saved)
            self.assertIsNone(launcher.advance_queue(root, limit=1))  # Not stale yet: the start may still finish.
            for item in saved["items"][:3]:
                item["starting_at"] = "2026-10-11T00:00:00+00:00"
            write_json_atomic(path, saved)
            # "launched" got as far as its project; "exited" had a runner that died before creating one.
            for project in ("launched", "exited"):
                write_json_atomic(root / f".runtime_runs/dashboard_launches/{project}-1/launch.json", {
                    "schema_version": "dashboard_launch_v1", "kind": "create", "project": project,
                    "pid": 1, "create_time": 0.0, "started_at": "2026-10-11T00:00:05+00:00"})
            create_project(root, "launched", "Build launched.")
            started = launcher.advance_queue(root, limit=1)
            self.assertEqual((started["request"]["project"], started["status"]), ("next", "started"))
            self.assertEqual([call.argv[call.argv.index("--project") + 1] for call in calls], ["next"])
            items = {item["request"]["project"]: item for item in json.loads(path.read_text(encoding="utf-8"))["items"]}
            self.assertEqual(items["launched"]["status"], "started")
            self.assertEqual(items["interrupted"]["status"], "failed")
            self.assertIn("LAUNCH_INTERRUPTED", items["interrupted"]["error"])
            self.assertEqual(items["exited"]["status"], "failed")
            self.assertIn(".runtime_runs/dashboard_launches/exited-1/launch.log", items["exited"]["error"])
            self.assertEqual([item["project"] for item in launcher.queue(root)], ["interrupted", "exited"])

    def test_queue_requests_need_the_capability_and_launch(self) -> None:
        with _launch_server() as (_root, server):
            status, _body = _call(server, "POST", "/api/launch/queue", _free("x"), {"Content-Type": "application/json"})
            self.assertEqual(status, 403)
            server.launch_enabled = False
            self.assertEqual(_call(server, "POST", "/api/launch/queue", _free("x"), AUTH)[0], 404)


class RunsOverviewTest(unittest.TestCase):
    def test_overview_reports_running_and_finished_projects(self) -> None:
        with _launch_server() as (root, server):
            running = create_project(root, "running", "Build a running machine.")
            write_json_atomic(root / ".runtime_runs/dashboard_launches/running-1/launch.json", {
                "schema_version": "dashboard_launch_v1", "kind": "create", "project": "running",
                "pid": os.getpid(), "create_time": 0.0, "started_at": "2026-10-12T00:00:00+00:00"})
            write_json_atomic(running / "construction_budget.json", {
                "config": {"hours": 3}, "active_seconds": 600.0, "updated_at_epoch": time.time()})
            finished = create_project(root, "finished", "Build a finished machine.")
            state = json.loads((finished / "workflow.json").read_text(encoding="utf-8"))
            state.update(stop_reason="BUDGET_EXHAUSTED", phase="budget_exhausted")
            write_json_atomic(finished / "workflow.json", state)
            write_json_atomic(finished / "runs/r01/result.json", {"status": "NEEDS_REVISION", "issues": ["too weak"],
                "tests": [{"physical_pass": True}, {"physical_pass": False}]})
            with mock.patch.object(launcher, "_alive", lambda record: record.get("project") == "running"):
                status, overview = _call(server, "GET", "/api/runs-overview")
            self.assertEqual(status, 200)
            self.assertEqual([row["project"] for row in overview["runs"]], ["running", "finished"])
            row = overview["runs"][0]
            self.assertEqual((row["status"], row["can_stop"], row["can_resume"], row["launched"]), ("running", True, False, True))
            self.assertEqual((row["budget"]["limit_seconds"], row["budget"]["active_seconds"]), (10800.0, 600.0))
            self.assertEqual(row["task"], "Build a running machine.")
            self.assertEqual((row["attempt_started_at"], row["resumed"]), ("2026-10-12T00:00:00+00:00", False))
            done = overview["runs"][1]
            self.assertEqual((done["status"], done["can_stop"], done["can_resume"]), ("budget_exhausted", False, True))
            self.assertEqual((done["attempt_started_at"], done["resumed"]), (None, False))
            self.assertEqual({k: done["latest_result"][k] for k in ("revision", "status", "tests_total", "tests_passed")},
                             {"revision": "r01", "status": "NEEDS_REVISION", "tests_total": 2, "tests_passed": 1})
            self.assertTrue(overview["launch"])

    def test_runs_that_ended_without_passing_say_how(self) -> None:
        with _launch_server() as (root, server):
            endings = {"final": {"stop_reason": "EVIDENCE_REQUIRED", "phase": "evidence_collection", "first_pass_handoff": True},
                       "partial": {"stop_reason": "agent_chose_partial_delivery", "phase": "incomplete"},
                       "waiting": {"stop_reason": "EVIDENCE_REQUIRED", "phase": "evidence_collection"},
                       "untestable": {"stop_reason": "no_executable_scene", "phase": "first_pass_blocked", "first_pass_handoff": True},
                       "crashed": {"stop_reason": "HARNESS_ERROR", "phase": "harness_repair", "first_pass_handoff": True},
                       # Construction runs (all dashboard runs) stop the reserved pass on NEEDS_REVISION.
                       "crashed_construction": {"stop_reason": "NEEDS_REVISION", "phase": "revision_needed", "first_pass_handoff": True},
                       "unfinished": {"stop_reason": "NEEDS_REVISION", "phase": "revision_needed", "first_pass_handoff": True}}
            for project, ending in endings.items():
                directory = create_project(root, project, "Build a machine.")
                state = json.loads((directory / "workflow.json").read_text(encoding="utf-8"))
                state.update(ending)
                write_json_atomic(directory / "workflow.json", state)
                write_json_atomic(directory / "workflow_runtime.json", {
                    "schema_version": "workflow_runtime_v1", "status": "stopped", "stop_reason": ending["stop_reason"],
                    "heartbeat_at": "2099-01-01T00:00:00+00:00"})
                write_json_atomic(directory / "runs/r01/result.json", {
                    "schema_version": "slim_harness_result_v1", "revision": 1, "issues": [],
                    "status": {"partial": "INCOMPLETE_EVIDENCE", "untestable": "INCOMPLETE_EVIDENCE", "crashed": "HARNESS_ERROR",
                               "crashed_construction": "NEEDS_REVISION", "unfinished": "NEEDS_REVISION"}.get(project, "EVIDENCE_REQUIRED"),
                    **({"reported_status": "HARNESS_ERROR"} if project == "crashed_construction" else {}),
                    **({"tests": [{"physical_pass": True}, {"physical_pass": True}]} if project == "unfinished" else {})})
            expected = {"final": "final_evaluation_failed", "partial": "partial_delivery", "waiting": "stopped",
                        "untestable": "scene_untestable", "crashed": "harness_error",
                        "crashed_construction": "harness_error", "unfinished": "author_unfinished"}
            statuses = {row["project"]: row["status"] for row in launcher.runs_overview(root)["runs"]}
            self.assertEqual(statuses, expected)
            # An imported snapshot shows "imported_snapshot" as its stop reason but keeps its outcome.
            with mock.patch("simulation_ready_cad_workflow.validation_pipeline.snapshot_paths.imported", return_value=True), \
                    mock.patch("simulation_ready_cad_workflow.validation_pipeline.snapshot_paths.workspace_for", return_value=root):
                imported = launcher.runs_overview(root)["runs"]
            self.assertEqual({row["stop_reason"] for row in imported}, {"imported_snapshot"})
            self.assertEqual({row["project"]: row["status"] for row in imported}, expected)
            server.import_parent = None
            status, listing = _call(server, "GET", "/api/runs?fresh=1")
            self.assertEqual(status, 200)
            rows = {row["id"].split("/")[2]: row["status"] for row in listing["runs"]}
            self.assertEqual({key: rows[key] for key in ("final", "partial", "waiting", "crashed")},
                             {"final": "failed", "partial": "failed", "waiting": "blocked", "crashed": "blocked"})

    def test_a_stop_only_shows_while_its_runner_is_alive(self) -> None:
        with _launch_server() as (root, server):
            create_project(root, "resumed", "Build a machine.")
            write_json_atomic(root / ".runtime_runs/dashboard_launches/resumed-1/launch.json", {
                "schema_version": "dashboard_launch_v1", "kind": "create", "project": "resumed", "pid": 1,
                "create_time": 0.0, "started_at": "2026-10-12T00:00:00+00:00", "stop_requested_at": "2026-10-12T01:00:00+00:00"})
            live = {"live_workflow": {"is_running": True, "phase": "author"}}
            with mock.patch("simulation_ready_cad_workflow.validation_pipeline.dashboard.live_workflow.read_live_project",
                            return_value=live):
                for alive, status in ((True, "stopping"), (False, "running")):
                    with mock.patch.object(launcher, "_alive", return_value=alive):
                        row = launcher.runs_overview(root)["runs"][0]
                    self.assertEqual(row["status"], status)

    def test_an_action_is_visible_on_the_next_refresh(self) -> None:
        with _launch_server() as (_root, server):
            self.assertEqual(_call(server, "GET", "/api/runs-overview")[1]["queue"], [])
            _call(server, "POST", "/api/launch/queue", _free("later"), AUTH)
            self.assertEqual([item["project"] for item in _call(server, "GET", "/api/runs-overview")[1]["queue"]], ["later"])

    def test_monitor_page_is_served_without_launch(self) -> None:
        with _launch_server() as (_root, server):
            server.launch_enabled = False
            status, page = _call(server, "GET", "/runs")
            self.assertEqual(status, 200)
            self.assertIn("/static/runs.js", page["raw"])
            self.assertEqual(_call(server, "GET", "/static/runs.js")[0], 200)
            status, overview = _call(server, "GET", "/api/runs-overview")
            self.assertEqual((status, overview["launch"], overview["runs"]), (200, False, []))


class LaunchOptionsModelsTest(unittest.TestCase):
    def test_codex_models_come_from_its_cache_and_examples_have_titles(self) -> None:
        with tempfile.TemporaryDirectory() as codex_home, _launch_server() as (root, server), \
                mock.patch.object(launcher, "_checks", return_value={"ready": False, "items": []}):
            write_json_atomic(Path(codex_home) / "models_cache.json", {"models": [
                {"slug": "gpt-6.1-sol", "display_name": "GPT-6.1-Sol", "visibility": "list",
                 "supported_reasoning_levels": [{"effort": "high"}, {"effort": "xhigh"}]},
                {"slug": "internal", "display_name": "Hidden", "visibility": "hide"}]})
            (root / "examples/tasks").mkdir(parents=True)
            (root / "examples/tasks/screw_jack.md").write_text("Build a jack.", encoding="utf-8")
            with mock.patch.dict(os.environ, {"CODEX_HOME": codex_home}):
                status, options = _call(server, "GET", "/api/launch/options")
            self.assertEqual(status, 200)
            self.assertEqual(options["models"]["codex"], [{"id": "gpt-6.1-sol", "label": "GPT-6.1-Sol", "reasoning": ["high", "xhigh"]}])
            self.assertIn("claude-opus-5-5", [item["id"] for item in options["models"]["claude_code"]])
            self.assertEqual(options["examples"][0]["title"], "Screw jack")
            self.assertEqual(options["queue"], [])


if __name__ == "__main__":
    unittest.main()
