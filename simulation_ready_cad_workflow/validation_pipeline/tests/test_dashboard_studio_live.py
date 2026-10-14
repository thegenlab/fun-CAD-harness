from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from urllib.parse import quote

from simulation_ready_cad_workflow.validation_pipeline.tests.test_dashboard_changes import _dashboard, _get, _write_json


def make_project(root: Path, name: str = "studio_fixture") -> tuple[Path, str]:
    project = root / ".runtime_runs" / "projects" / name
    run = project / "runs" / "r01"
    run.mkdir(parents=True)
    _write_json(project / "workflow.json", {
        "schema_version": "cold_start_workflow_v1", "phase": "author", "revision": 1,
        "started_at": "2026-09-06T01:00:00+00:00", "updated_at": "2026-09-06T01:00:12+00:00",
        "elapsed_seconds": 12.0, "stop_reason": "test_paused", "complete_instance_pass": False,
        "turn_count": 1, "events": [],
        "request": {"project_id": name, "workspace_root": str(root), "brief": "Fixture mechanism",
                    "source_step": str(root / "source.step")},
    })
    return run, run.relative_to(root).as_posix()


def add_usage(run: Path, turn: int, inputs: int, outputs: int, cached: int = 0) -> Path:
    directory = run.parent.parent / "worker_turns" / f"t{turn:03d}_author"
    directory.mkdir(parents=True)
    path = directory / "events.jsonl"
    path.write_text(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": inputs, "output_tokens": outputs, "cached_input_tokens": cached,
    }}) + "\n", encoding="utf-8")
    return path


class DashboardStudioLiveTests(unittest.TestCase):
    def test_unsubmitted_run_progress_updates_without_full_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = make_project(root)
            add_usage(run, 1, 200, 20, 100)
            with _dashboard(root) as server, mock.patch.object(server, "studio_preview", return_value={"status": "queued"}), \
                    mock.patch("simulation_ready_cad_workflow.validation_pipeline.dashboard.server.normalize_run",
                               side_effect=AssertionError("Progress must not normalize all evidence")):
                self.assertEqual(run.resolve(), server.resolve_run(run_id))
                first_token = server.dashboard_change_snapshot(run_id)
                status, body, _ = _get(server, "/api/progress?run=" + quote(run_id, safe=""))
                self.assertEqual(200, status)
                first = json.loads(body)
                self.assertFalse(first["live_workflow"]["is_running"])
                self.assertEqual(12.0, first["live_workflow"]["elapsed_seconds"])
                self.assertEqual(220, first["live_usage"]["total_tokens"])
                add_usage(run, 2, 30, 5)
                second_token = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(first_token["run_token"], second_token["run_token"])
                status, body, _ = _get(server, "/api/progress?run=" + quote(run_id, safe=""))
                second = json.loads(body)
                self.assertEqual(255, second["live_usage"]["total_tokens"])
                self.assertEqual(100, second["live_usage"]["cached_input_tokens"])

    def test_preview_file_uses_the_selected_runs_explicit_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = make_project(root)
            model = root / ".runtime_runs" / "dashboard_previews" / "studio_fixture" / "r01" / "model.glb"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"glTF-test-preview")
            module = "simulation_ready_cad_workflow.validation_pipeline.dashboard.preview_service"
            with _dashboard(root) as server, mock.patch(module + ".ensure_preview", return_value={
                    "status": "ready", "model_path": str(model), "input_version": "version-1",
                    "source_kind": "original_geometry"}), \
                    mock.patch(module + ".preview_media_paths", return_value={"model.glb": model}):
                preview = server.studio_preview(run_id, run)
                self.assertIn("/api/preview-file?", preview["model_url"])
                status, body, _ = _get(server, preview["model_url"])
                self.assertEqual((200, b"glTF-test-preview"), (status, body))
                status, _, _ = _get(server, "/api/preview-file?run=" + quote(run_id, safe="") + "&name=unlisted.glb")
                self.assertEqual(404, status)

    def test_new_project_is_announced_while_another_run_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run, run_id = make_project(root)
            with _dashboard(root) as server:
                before = server.dashboard_change_snapshot(run_id)
                make_project(root, "another_machine")
                after = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(before["catalog_token"], after["catalog_token"])
