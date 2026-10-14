"""Motions people can watch: end-of-run motion preview, motion choice and monitor thumbnails."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.dashboard import launcher
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import write_json_atomic
from simulation_ready_cad_workflow.validation_pipeline.slim_harness import motion_preview

from .test_dashboard_launch import _call, _launch_server


def _motion(motion_id, frames, **extra):
    return {"id": motion_id, "label": motion_id, "load_case_id": motion_id, "motion_frame_count": frames, **extra}


def _revision(project: Path, name: str, motions=(), *, poses=None, prepared=True) -> Path:
    run = project / "runs" / name
    write_json_atomic(run / "slim_evidence" / "preview.json", {"schema_version": "slim_studio_preview_v1",
                                                               "input_version": "v-" + name, "motions": list(motions)})
    attempt = run / "template_runs" / "a001"
    if prepared:
        write_json_atomic(attempt / "prepared" / "prepared.json", {"bodies": []})
    for trial, count in (poses or {}).items():
        folder = attempt / "tests" / trial / trial
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "body_poses.jsonl").write_text("".join(json.dumps({"time_s": i * .02, "bodies": {}}) + "\n"
                                                         for i in range(count)), encoding="utf-8")
    return run


def _fake_compute(argv, workspace, project, inputs, outputs, timeout):
    """Stand-in for the replay and settle subprocesses: write what they would write."""
    request = json.loads(Path(argv[-1]).read_text(encoding="utf-8"))
    if "settle" in argv:
        run = Path(request["output"])
        run.mkdir(parents=True)
        (run / "body_poses.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
        return mock.Mock(returncode=0)
    output = Path(request["output"])
    samples = [{"time_s": 0.0}, {"time_s": 0.5}]
    write_json_atomic(output / "native-trace.json", {"samples": samples})
    write_json_atomic(output / "replay-assets.json", {"model_file": str(output / "replay/m.glb"), "sample_count": 2,
                                                      "scene_file": str(output / "replay/scene.json"),
                                                      "replay_file": str(output / "replay/replay.json")})
    return mock.Mock(returncode=0)


class MotionChoiceTest(unittest.TestCase):
    def test_the_motion_with_most_frames_is_shown_and_single_frames_never(self) -> None:
        self.assertEqual(motion_preview.best_motion([_motion("a", 1), _motion("b", 82), _motion("c", 40)])["id"], "b")
        self.assertIsNone(motion_preview.best_motion([_motion("a", 1)]))

    def test_a_complete_recording_is_shown_before_a_longer_preview(self) -> None:
        motions = [_motion("settle", 76, status="UNVERIFIED_PREFIX"), _motion("fold", 51)]
        self.assertEqual(motion_preview.best_motion(motions)["id"], "fold")

    def test_partial_motions_count_only_for_the_same_design(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = _revision(Path(temporary), "r01", [_motion("a", 1)])
            write_json_atomic(run / "slim_evidence/partial-motion-preview.json", {
                "schema_version": motion_preview.PARTIAL_SCHEMA, "input_version": "v-r01", "motions": [_motion("b", 9)]})
            self.assertEqual([item["id"] for item in motion_preview.revision_motions(run)], ["a", "b"])
            write_json_atomic(run / "slim_evidence/partial-motion-preview.json", {
                "schema_version": motion_preview.PARTIAL_SCHEMA, "input_version": "older", "motions": [_motion("b", 9)]})
            self.assertEqual([item["id"] for item in motion_preview.revision_motions(run)], ["a"])


class FinalMotionTest(unittest.TestCase):
    def _ensure(self, project: Path) -> dict:
        with mock.patch.object(motion_preview, "_compute", side_effect=_fake_compute) as compute:
            report = motion_preview.ensure_final_motion(project.parent, project, chrono_python="chrono")
        self.compute_calls = compute.call_args_list
        return report

    def test_a_project_that_has_a_motion_is_left_alone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "p"
            _revision(project, "r01", [_motion("old", 30)])
            _revision(project, "r02", [_motion("new", 1)], poses={"new": 1})
            self.assertEqual(self._ensure(project), {"status": "present", "revision": "r01", "motion": "old"})
            self.assertEqual(self.compute_calls, [])

    def test_recorded_poses_that_were_never_replayed_are_exported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "p"
            _revision(project, "r01", poses={"lift": 5})
            run = _revision(project, "r02", [_motion("press", 1)], poses={"press": 1, "hold": 12})
            report = self._ensure(project)
            self.assertEqual(report, {"status": "salvaged", "revision": "r02", "motions": ["hold"]})
            motion = motion_preview.best_motion(motion_preview.revision_motions(run))
            self.assertEqual((motion["id"], motion["status"], motion["preview_kind"], motion["recorded_duration_s"]),
                             ("hold", "UNVERIFIED_PREFIX", "recorded_test", 0.5))
            self.assertEqual(json.loads((project / "final-motion.json").read_text())["status"], "salvaged")

    def test_without_any_recording_the_newest_prepared_design_settles_under_gravity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "p"
            _revision(project, "r01")
            run = _revision(project, "r02", [_motion("press", 1)], poses={"press": 1})
            report = self._ensure(project)
            self.assertEqual(report, {"status": "settled", "revision": "r02", "motion": "gravity_settle"})
            settle = [call for call in self.compute_calls if "settle" in call.args[0]]
            self.assertEqual(settle[0].args[0][0], "chrono")
            # Written inside the attempt that prepared the design (where sandboxed runs may write).
            self.assertEqual(settle[0].args[4], [run / "template_runs/a001/motion_preview/gravity_settle"])
            motion = motion_preview.best_motion(motion_preview.revision_motions(run))
            self.assertEqual((motion["id"], motion["preview_kind"]), ("gravity_settle", "gravity_settle"))

    def test_nothing_to_show_is_recorded_without_failing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "p"
            (project / "runs" / "r01").mkdir(parents=True)
            report = self._ensure(project)
            self.assertEqual(report["status"], "unavailable")


class MonitorMotionTest(unittest.TestCase):
    def test_monitor_rows_carry_the_newest_motion_and_its_thumbnail(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli import create_project
        with _launch_server() as (root, server):
            project = create_project(root, "lifter", "Build a lifter.")
            _revision(project, "r01", [_motion("lift", 40, replay_version="abc", replay_path="x", scene_path="y",
                                               motion_model_path="z")])
            _revision(project, "r02", [_motion("lift", 1)])
            self.assertEqual(launcher.latest_motion(project)[0].name, "r01")
            server._motion_thumbnails = mock.Mock(status=mock.Mock(return_value="pending"))
            server._image_service_lock = __import__("threading").Lock()
            row = _call(server, "GET", "/api/runs-overview")[1]["runs"][0]
            self.assertEqual({key: row["motion"][key] for key in ("revision", "id", "frames")},
                             {"revision": "r01", "id": "lift", "frames": 40})
            self.assertEqual(row["motion"]["thumbnail"], {"status": "pending", "url": None})
            self.assertEqual(_call(server, "GET", "/api/motion-thumbnail?run=.runtime_runs/projects/lifter/runs/r01&motion=lift")[0], 404)


class ThumbnailRenderTest(unittest.TestCase):
    def test_a_recorded_motion_becomes_a_looping_animation(self) -> None:
        import trimesh
        from PIL import Image

        from simulation_ready_cad_workflow.validation_pipeline.dashboard.motion_thumbnail import render
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            scene = trimesh.Scene()
            scene.add_geometry(trimesh.creation.box((.1, .02, .02)), node_name="physics__arm__original",
                               geom_name="physics__arm__original")
            (folder / "model.glb").write_bytes(trimesh.exchange.gltf.export_glb(scene))
            frames = [{"time_s": t, "body_poses": {"arm": {"position_abs_m": [0, 0, 0],
                                                            "rotation_abs_quat_wxyz": [1, 0, 0, t]}}} for t in (0, .5, 1)]
            write_json_atomic(folder / "replay.json", {"branches": {"physical": {"frames": frames}}})
            write_json_atomic(folder / "scene.json", {"motion_bounds_abs_m": {"min": [-.06, -.06, -.02], "max": [.06, .06, .02]}})
            render({"model": str(folder / "model.glb"), "replay": str(folder / "replay.json"),
                    "scene": str(folder / "scene.json"), "up_axis": [0, 0, 1], "output": str(folder / "motion.gif")})
            with Image.open(folder / "motion.gif") as image:
                self.assertEqual((image.size, image.n_frames), ((320, 200), 3))


if __name__ == "__main__":
    unittest.main()
