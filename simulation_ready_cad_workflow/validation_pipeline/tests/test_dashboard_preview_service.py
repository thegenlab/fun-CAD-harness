import json
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.dashboard import preview_service as previews


class DashboardPreviewServiceTests(unittest.TestCase):
    def test_partial_motion_requires_current_source_and_keeps_formal_preview_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "r01"
            evidence = run / "slim_evidence"
            evidence.mkdir(parents=True)
            original = {"schema_version": "slim_studio_preview_v1", "input_version": "current", "motions": []}
            (evidence / "preview.json").write_text(json.dumps(original))
            descriptor = {"schema_version": "slim_partial_motion_preview_v1", "input_version": "old",
                          "motions": [{"id": "prefix", "load_case_id": "load", "status": "UNVERIFIED_PREFIX"}]}
            partial = evidence / "partial-motion-preview.json"
            partial.write_text(json.dumps(descriptor))
            self.assertEqual([], previews._slim_preview(run)["motions"])
            descriptor["input_version"] = "current"
            partial.write_text(json.dumps(descriptor))
            self.assertEqual("UNVERIFIED_PREFIX", previews._slim_preview(run)["motions"][0]["status"])
            self.assertEqual(original, json.loads((evidence / "preview.json").read_text()))
            original["motions"] = [{"id": "full", "load_case_id": "load"}]
            (evidence / "preview.json").write_text(json.dumps(original))
            self.assertEqual(["full"], [m["id"] for m in previews._slim_preview(run)["motions"]])

    def test_published_preview_stays_workspace_relative_and_still_opens(self):
        from types import SimpleNamespace
        from simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow import SlimWorkflow

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            run = workspace / ".runtime_runs/projects/example/runs/r01"
            design = run / "design"
            geometry_dir = run / "geometry"
            design.mkdir(parents=True)
            geometry_dir.mkdir()
            scene = design / "scene.json"
            scene.write_text("{}", encoding="utf-8")
            body = design / "body.step"
            body.write_text("step", encoding="utf-8")
            model = geometry_dir / "model.glb"
            model.write_bytes(b"glb")
            SlimWorkflow._preview(
                SimpleNamespace(workspace=workspace),
                run,
                {"scene_file": str(scene), "model_file": str(model),
                 "bodies": [{"source_file": str(body), "triangle_count": 2}]},
                [],
            )
            recorded = json.loads((run / "slim_evidence/preview.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["model_path"], model.relative_to(workspace).as_posix())
            self.assertEqual(recorded["source_path"], scene.relative_to(workspace).as_posix())
            self.assertTrue(all(not Path(row[0]).is_absolute() for row in recorded["input_stamps"]))
            slim = previews._slim_preview(run)
            self.assertEqual(Path(slim["model_path"]).resolve(), model.resolve())
            self.assertEqual("ready", slim["status"])

    def service(self, workspace):
        service = object.__new__(previews.PreviewService)
        service.workspace = workspace
        service.lock = threading.Lock()
        service.jobs = queue.Queue()
        service.records = {}
        return service

    def test_repeated_observation_queues_one_stable_source_without_a_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            run = workspace / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("source geometry", encoding="utf-8")
            service = self.service(workspace)
            with mock.patch.object(previews.time, "monotonic", return_value=0):
                self.assertEqual("queued", service.ensure(run)["status"])
            self.assertEqual(0, service.jobs.qsize())
            with mock.patch.object(previews.time, "monotonic", return_value=2):
                service.ensure(run)
                service.ensure(run)
                service.ensure(run)
            self.assertEqual(1, service.jobs.qsize())
            source.write_text("changed source geometry", encoding="utf-8")
            with mock.patch.object(previews.time, "monotonic", return_value=3):
                service.ensure(run)
            with mock.patch.object(previews.time, "monotonic", return_value=5):
                service.ensure(run)
            self.assertEqual(2, service.jobs.qsize())

    def test_published_source_is_reused_after_server_restart(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            run = workspace / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("source geometry", encoding="utf-8")
            _, inputs, _ = previews._inputs(workspace, run)
            cache = previews._cache_root(workspace, run)
            cache.mkdir(parents=True)
            model = cache / "model.glb"
            model.write_bytes(b"glTF-existing-preview")
            (cache / "preview.json").write_text(json.dumps({"status": "ready", "model_path": str(model),
                "inputs": [list(item) for item in inputs], "generator": previews.PREVIEW_VERSION}), encoding="utf-8")
            service = self.service(workspace)
            self.assertEqual("ready", service.ensure(run)["status"])
            self.assertEqual(0, service.jobs.qsize())

    def test_failed_video_is_not_treated_as_a_completed_cache(self):
        from simulation_ready_cad_workflow.validation_pipeline.dashboard.visuals import _render_atomic_asset, _asset_current

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trace.json"
            source.write_text("recorded trace", encoding="utf-8")
            output = Path(temporary) / "motion.mp4"
            output.write_bytes(b"old incomplete mp4")

            def fail(path):
                path.write_bytes(b"incomplete mp4")
                raise RuntimeError("encoder failed")

            with self.assertRaises(RuntimeError):
                _render_atomic_asset(output, [source], fail)
            self.assertFalse(_asset_current(output, [source]))
            rebuild = mock.Mock(side_effect=lambda path: path.write_bytes(b"completed video"))
            _render_atomic_asset(output, [source], rebuild)
            _render_atomic_asset(output, [source], rebuild)
            self.assertEqual(1, rebuild.call_count)
            self.assertEqual(b"completed video", output.read_bytes())

    def test_source_update_during_render_cannot_stamp_old_frames_as_current(self):
        from simulation_ready_cad_workflow.validation_pipeline.dashboard.visuals import _render_atomic_asset, _asset_current

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "trace.json"
            source.write_text("initial trace", encoding="utf-8")
            output = Path(temporary) / "motion.mp4"

            def render(path):
                path.write_bytes(b"frames of initial trace")
                source.write_text("updated trace with more frames", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "source changed"):
                _render_atomic_asset(output, [source], render)
            self.assertFalse(output.exists())
            self.assertFalse(_asset_current(output, [source]))
            self.assertFalse(output.with_name(".motion.pending.mp4").exists())
            render = mock.Mock(side_effect=lambda path: path.write_bytes(b"completed recording"))
            _render_atomic_asset(output, [source], render)
            _render_atomic_asset(output, [source], render)
            self.assertEqual(1, render.call_count)
            self.assertEqual(b"completed recording", output.read_bytes())
            source.write_text("a changed trace", encoding="utf-8")
            self.assertFalse(_asset_current(output, [source]))

    def test_video_companion_frames_and_summary_are_published_together(self):
        from simulation_ready_cad_workflow.validation_pipeline.dashboard.visuals import _render_atomic_asset, _asset_current

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "trace.json"
            source.write_text("trace", encoding="utf-8")
            output = root / "movie.mp4"

            def render(path):
                path.write_bytes(b"video")
                frame = path.with_suffix(".start.png")
                frame.write_bytes(b"frame")
                path.with_suffix(".render_summary.json").write_text(json.dumps({"video": str(path),
                    "frame": frame.relative_to(root).as_posix()}), encoding="utf-8")

            _render_atomic_asset(output, [source], render, workspace=root)
            summary = json.loads(output.with_suffix(".render_summary.json").read_text())
            self.assertEqual(str(output), summary["video"])
            self.assertEqual("movie.start.png", summary["frame"])
            self.assertTrue(output.with_suffix(".start.png").is_file())
            self.assertTrue(_asset_current(output, [source]))
            output.with_suffix(".start.png").unlink()
            self.assertFalse(_asset_current(output, [source]))

    def test_close_during_request_write_cannot_start_a_new_converter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("geometry", encoding="utf-8")
            service = previews.PreviewService(root)
            blocked, release = threading.Event(), threading.Event()
            original_write = previews.write_json_atomic

            def write(path, value):
                if path.name == "request.json":
                    blocked.set()
                    release.wait(2)
                original_write(path, value)

            try:
                with mock.patch.object(previews, "write_json_atomic", side_effect=write), \
                        mock.patch.object(previews.subprocess, "Popen") as popen:
                    service.ensure(run)
                    service.records[str(run)]["first_seen"] -= 1
                    service.ensure(run)
                    self.assertTrue(blocked.wait(2))
                    closer = threading.Thread(target=service.close)
                    closer.start()
                    self.assertTrue(service.closed.wait(1))
                    release.set()
                    closer.join(2)
                    self.assertFalse(service.thread.is_alive())
                    popen.assert_not_called()
            finally:
                release.set()
                service.close()

    def test_trace_changes_publish_in_a_new_directory_and_reuse_static_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("geometry", encoding="utf-8")
            contract = source.parent / "assembly_contract.json"
            contract.write_text("initial", encoding="utf-8")
            requests = []

            class Converter:
                def __init__(self, command, **_kwargs):
                    request = json.loads(Path(command[-1]).read_text())
                    requests.append(request)
                    out = Path(request["output"])
                    model = Path(request["model_cache"])
                    model.parent.mkdir(parents=True, exist_ok=True)
                    if not model.exists(): model.write_bytes(b"glTF-static")
                    scene = out / "motion.scene.json"
                    scene.write_text(json.dumps({"generation": len(requests)}))
                    (out / "result.json").write_text(json.dumps({"model_path": str(model), "scene_path": str(scene),
                        "source_path": str(source)}))
                def wait(self, **_kwargs): return 0
                def poll(self): return 0

            def ready(service):
                service.ensure(run)
                service.records[str(run)]["first_seen"] -= 1
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    value = service.ensure(run)
                    if value["status"] == "ready": return value
                    time.sleep(.005)
                self.fail("preview did not finish")

            with mock.patch.object(previews.subprocess, "Popen", Converter):
                service = previews.PreviewService(root)
                try:
                    first = ready(service)
                    first_bytes = Path(first["scene_path"]).read_bytes()
                    contract.write_text("updated mapping", encoding="utf-8")
                    second = ready(service)
                    self.assertEqual(first["model_path"], second["model_path"])
                    self.assertNotEqual(first["scene_path"], second["scene_path"])
                    self.assertEqual(first_bytes, Path(first["scene_path"]).read_bytes())
                    self.assertNotEqual(requests[0]["output"], requests[1]["output"])
                finally:
                    service.close()

    def converter_writing(self, requests):
        class Converter:
            def __init__(self, command, **_kwargs):
                request = json.loads(Path(command[-1]).read_text())
                requests.append(request)
                model = Path(request["model_cache"])
                model.parent.mkdir(parents=True, exist_ok=True)
                model.write_bytes(b"glTF-frame")
                (Path(request["output"]) / "result.json").write_text(json.dumps({"model_path": str(model)}))
            def wait(self, **_kwargs): return 0
            def poll(self): return 0
        return Converter

    def drive_until_ready(self, service, run):
        service.ensure(run)
        service.records[str(run)]["first_seen"] -= 1
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            value = service.ensure(run)
            if value["status"] == "ready": return value
            time.sleep(.005)
        self.fail("preview did not finish")

    def test_successful_publish_records_one_timeline_frame_per_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("geometry v1", encoding="utf-8")
            requests = []
            with mock.patch.object(previews.subprocess, "Popen", self.converter_writing(requests)):
                service = previews.PreviewService(root)
                try:
                    first = self.drive_until_ready(service, run)
                    cache = previews._cache_root(root, run)
                    frames = previews._read_timeline(cache)["frames"]
                    self.assertEqual(1, len(frames))
                    self.assertEqual(first["input_version"], frames[0]["id"])
                    self.assertEqual(f"{first['input_version']}/model.glb", frames[0]["model"])
                    self.assertTrue(frames[0]["created_at"])
                    source.write_text("geometry v2 changed", encoding="utf-8")
                    second = self.drive_until_ready(service, run)
                    self.assertNotEqual(first["input_version"], second["input_version"])
                    frames = previews._read_timeline(cache)["frames"]
                    self.assertEqual(2, len(frames))
                    service._record_timeline_frame(cache, second, second["inputs"])
                    self.assertEqual(2, len(previews._read_timeline(cache)["frames"]))
                    media = previews.preview_media_paths(workspace_root=root, run_dir=run)
                    self.assertIn(f"timeline/{first['input_version']}/model.glb", media)
                    self.assertIn(f"timeline/{second['input_version']}/model.glb", media)
                finally:
                    service.close()

    def write_timeline_fixture(self, root, run, count):
        cache = previews._cache_root(root, run)
        frames = []
        for index in range(count):
            version = f"v{index:02d}"
            (cache / version).mkdir(parents=True)
            (cache / version / "model.glb").write_bytes(b"glTF")
            frames.append({"id": version, "created_at": "2026-09-12T00:00:00+00:00",
                           "model": f"{version}/model.glb", "bytes": 4})
        (cache / "timeline.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")
        return cache, frames

    def test_terminal_run_preserves_all_history_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("geometry", encoding="utf-8")
            cache, _ = self.write_timeline_fixture(root, run, 20)
            (run / "result.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            service = self.service(root)
            service.ensure(run)
            timeline = previews._read_timeline(cache)
            kept = [frame["id"] for frame in timeline["frames"]]
            self.assertNotIn("pruned", timeline)
            self.assertEqual(20, len(kept))
            self.assertEqual("v00", kept[0])
            self.assertEqual("v19", kept[-1])
            self.assertTrue((cache / "v01").is_dir())
            self.assertTrue((cache / "v05").exists())

    def test_prune_keeps_versions_referenced_by_the_published_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("geometry", encoding="utf-8")
            cache, _ = self.write_timeline_fixture(root, run, 20)
            model = cache / "v05" / "model.glb"
            (cache / "preview.json").write_text(json.dumps({"input_version": "v05", "model_path": str(model)}), encoding="utf-8")
            (run / "run_status.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
            service = self.service(root)
            service.ensure(run)
            kept = [frame["id"] for frame in previews._read_timeline(cache)["frames"]]
            self.assertEqual(20, len(kept))
            self.assertIn("v05", kept)
            self.assertTrue((cache / "v05" / "model.glb").is_file())

    def test_active_run_keeps_the_full_timeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / ".runtime_runs/projects/example/runs/r01"
            source = run / "design/assembly.step"
            source.parent.mkdir(parents=True)
            source.write_text("geometry", encoding="utf-8")
            cache, _ = self.write_timeline_fixture(root, run, 20)
            service = self.service(root)
            service.ensure(run)
            timeline = previews._read_timeline(cache)
            self.assertNotIn("pruned", timeline)
            self.assertEqual(20, len(timeline["frames"]))
            self.assertTrue((cache / "v05").is_dir())

    def test_local_slim_preview_keeps_motions_after_inode_only_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary).resolve() / "runs/r01"
            scene = run / "design/scene.json"
            scene.parent.mkdir(parents=True)
            scene.write_text('{"gravity_m_s2":[0,-9.81,0]}', encoding="utf-8")
            model = run / "geometry/model.glb"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"glTF")
            replay = run / "replay.json"
            replay.write_text("{}", encoding="utf-8")
            stat = scene.stat()
            (run / "slim_evidence").mkdir()
            (run / "slim_evidence/preview.json").write_text(json.dumps({
                "schema_version": "slim_studio_preview_v1",
                "status": "ready",
                "source_path": str(scene),
                "model_path": str(model),
                "input_stamps": [[str(scene), stat.st_size, stat.st_mtime_ns, stat.st_ino]],
                "motions": [{"id": "move", "replay_path": str(replay), "motion_model_path": str(model)}],
            }), encoding="utf-8")
            payload = scene.read_bytes()
            sibling = scene.with_name("scene.resync.json")
            sibling.write_bytes(payload)
            os.utime(sibling, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            os.replace(sibling, scene)
            self.assertNotEqual(scene.stat().st_ino, stat.st_ino)
            loaded = previews._slim_preview(run)
            self.assertEqual(["move"], [item["id"] for item in loaded.get("motions") or []])
            scene.write_text('{"gravity_m_s2":[0,-9.81,0],"edited":true}', encoding="utf-8")
            self.assertEqual({}, previews._slim_preview(run))
