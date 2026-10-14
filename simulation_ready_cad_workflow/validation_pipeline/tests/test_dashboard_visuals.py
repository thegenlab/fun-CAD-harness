from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib

from simulation_ready_cad_workflow.validation_pipeline.dashboard.legacy_io import (
    react_physics_evidence_media_allowlist,
    react_reference_pose_media_allowlist,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.visuals import (
    _whole_machine_payload_with_trace_samples,
    _accepted_physical_task_load_case_ids,
    _published_reference_pose_diagnostic,
    _select_visual_trace_paths,
    _trace_samples_renderable,
    _whole_machine_render_bodies,
    _whole_machine_target_series,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.trace_pair_video import (
    _motion_coordinate,
)


class DashboardVisualTargetTests(unittest.TestCase):
    def test_frontend_keeps_multi_capsule_whole_machine_media_scoped(self) -> None:
        app_source = (
            Path(__file__).parents[1] / "dashboard" / "static" / "app.js"
        ).read_text(encoding="utf-8")

        self.assertIn("data.whole_machines || []", app_source)
        self.assertIn("if (scopedCapsules.size > 1) return null", app_source)
        self.assertIn(
            'visual("whole-machine-video", { capsuleId: whole.capsule_id })',
            app_source,
        )

    def test_frontend_shows_reference_pose_video_on_certificate_stage(self) -> None:
        app_source = (
            Path(__file__).parents[1] / "dashboard" / "static" / "app.js"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'visual("reference-comparison-video", { capsuleId: item?.capsule_id })',
            app_source,
        )
        self.assertIn("hero-media reference-comparison-replay", app_source)
        self.assertIn(
            'certificate: ["interactive-model", "visual-video", "certificate", "comparison"]',
            app_source,
        )
        self.assertIn("${physicsEvidenceReplay(item)}", app_source)
        self.assertIn(
            '<div class="physics-load-workspace">${loadCards(data, "comparison")}${physicsEvidenceReplay(item)}</div>',
            app_source,
        )

    def test_react_allowlist_includes_encoded_reference_pose_video(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / ".runtime_runs" / "projects" / "fixture_proj" / "runs" / "r01"
            run.mkdir(parents=True)
            video = (
                root
                / ".runtime_runs"
                / "diagnostics"
                / "fixture_proj-r01-ideal-pose"
                / "fixture_slide"
                / "physical_vs_ideal_roundtrip.mp4"
            )
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fake-mp4")
            (video.with_suffix(".sidecar.json")).write_text(
                json.dumps(
                    {
                        "status": "ENCODED",
                        "trace_sha256": {
                            "physical": "a" * 64,
                            "reduced": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )

            allowed = react_reference_pose_media_allowlist(
                workspace_root=root,
                run_dir=run,
            )

            self.assertEqual(
                set(allowed),
                {video.relative_to(root).as_posix()},
            )

    def test_react_allowlist_includes_diagnostic_physics_glb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / ".runtime_runs" / "projects" / "fixture_proj" / "runs" / "r01"
            run.mkdir(parents=True)
            glb = (
                root
                / ".runtime_runs"
                / "diagnostics"
                / "fixture_proj-r01-physics-evidence"
                / "fixture_slide"
                / "stage5_physics_evidence.glb"
            )
            glb.parent.mkdir(parents=True)
            glb.write_bytes(b"fake-glb")
            digest = hashlib.sha256(b"fake-glb").hexdigest()
            (glb.with_name("stage5_physics_evidence.scene.json")).write_text(
                json.dumps({"output_sha256": digest}),
                encoding="utf-8",
            )

            allowed = react_physics_evidence_media_allowlist(
                workspace_root=root,
                run_dir=run,
            )

            self.assertEqual(
                set(allowed),
                {glb.relative_to(root).as_posix()},
            )

    def test_formal_summary_excludes_outside_run_diagnostic_paths(self) -> None:
        published = _published_reference_pose_diagnostic(
            {
                "status": "ENCODED",
                "load_case_id": "task",
                "video": ".runtime_runs/diagnostics/run/video.mp4",
                "sidecar": ".runtime_runs/diagnostics/run/video.sidecar.json",
                "ideal_pose_trace": ".runtime_runs/diagnostics/run/trace.json",
                "simulation_was_rerun": False,
            }
        )

        self.assertEqual(published["status"], "ENCODED")
        self.assertNotIn("video", published)
        self.assertNotIn("sidecar", published)
        self.assertNotIn("ideal_pose_trace", published)

    def test_static_visuals_include_the_available_noto_cjk_fallback(self) -> None:
        self.assertIn(
            "Noto Sans CJK SC",
            matplotlib.rcParams["font.sans-serif"],
        )

    def test_renderable_gate_accepts_two_body_plain_revolute_samples(self) -> None:
        self.assertTrue(_trace_samples_renderable([{"body_poses": {"a": {}}}]))
        self.assertTrue(_trace_samples_renderable([{"moving_pose": {}}]))
        self.assertTrue(
            _trace_samples_renderable([{"fixed_pose": {}, "moving_pose": {}}])
        )
        self.assertFalse(_trace_samples_renderable([{"fixed_pose": {}}]))
        self.assertFalse(_trace_samples_renderable([]))
        self.assertFalse(_trace_samples_renderable(None))

    def test_motion_coordinate_labels_known_angular_mechanisms(self) -> None:
        rear_wheel = _motion_coordinate(
            [{"rear_drive_wheel_angle_rad": 0.25, "max_joint_drift_m": 0.0}]
        )
        crank = _motion_coordinate(
            [{"input_crank_angle_rad": 0.5, "max_support_drift_m": 0.0}]
        )
        generic = _motion_coordinate(
            [{"generalized_angle_rad": 0.75, "pivot_drift_m": 0.0}]
        )

        self.assertEqual(rear_wheel["kind"], "angular")
        self.assertEqual(rear_wheel["primary_label_zh"], "后轮转角 / °")
        self.assertEqual(rear_wheel["secondary_quantity_zh"], "关节漂移")
        self.assertEqual(crank["primary_label_zh"], "曲柄转角 / °")
        self.assertEqual(crank["secondary_quantity_zh"], "轴承支撑漂移")
        self.assertEqual(generic["primary_label_zh"], "转角 / °")
        self.assertEqual(generic["secondary_quantity_zh"], "转轴漂移")

    def test_rack_pinion_targets_fall_back_to_stage_targets(self) -> None:
        whole = {
            "stages": [
                {"stage": "outbound", "target_progress_m": 0.18},
                {"stage": "return", "target_progress_m": 0.0},
            ]
        }
        samples = [
            {"stage": "initial", "progress_m": 0.0},
            {"stage": "outbound", "progress_m": 0.08},
            {"stage": "return", "progress_m": 0.12},
        ]

        targets = _whole_machine_target_series(
            whole,
            samples,
            coordinate_key="progress_m",
            target_key="target_progress_m",
        )

        self.assertEqual(targets.tolist(), [0.0, 0.18, 0.0])

    def test_sample_target_takes_precedence(self) -> None:
        targets = _whole_machine_target_series(
            {"stages": [{"stage": "outbound", "target_progress_m": 0.18}]},
            [
                {
                    "stage": "outbound",
                    "progress_m": 0.02,
                    "target_progress_m": 0.11,
                }
            ],
            coordinate_key="progress_m",
            target_key="target_progress_m",
        )

        self.assertEqual(targets.tolist(), [0.11])

    def test_physical_replay_does_not_require_reduced_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_root = Path(temporary)
            task_dir = trace_root / "task_output_angle_roundtrip"
            task_dir.mkdir()
            (task_dir / "physical.trace.json").write_text(
                json.dumps(
                    {
                        "branch": "physical",
                        "status": "COMPLETED",
                        "samples": [{"time_s": 0.0}],
                    }
                ),
                encoding="utf-8",
            )

            physical, reduced = _select_visual_trace_paths(trace_root)

            self.assertEqual(physical, task_dir / "physical.trace.json")
            self.assertEqual(reduced, task_dir / "reduced.trace.json")

    def test_pose_complete_ideal_reference_is_selected_for_pair_replay(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            trace_root = Path(temporary)
            task_dir = trace_root / "task_platform_roundtrip"
            task_dir.mkdir()
            for name, branch in (
                ("physical.trace.json", "physical"),
                ("ideal_reference.trace.json", "ideal_reference"),
            ):
                (task_dir / name).write_text(
                    json.dumps(
                        {
                            "branch": branch,
                            "status": "COMPLETED",
                            "samples": [
                                {
                                    "time_s": 0.0,
                                    "body_poses": {
                                        "body": {
                                            "position_abs_m": [0.0, 0.0, 0.0],
                                            "rotation_abs_quat_wxyz": [
                                                1.0,
                                                0.0,
                                                0.0,
                                                0.0,
                                            ],
                                        }
                                    },
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )

            physical, comparison = _select_visual_trace_paths(trace_root)

            self.assertEqual(physical, task_dir / "physical.trace.json")
            self.assertEqual(
                comparison,
                task_dir / "ideal_reference.trace.json",
            )

    def test_machine_specific_trace_is_selected_from_task_acceptance_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trace_root = run / "traces" / "fixture_capsule"
            load_dir = trace_root / "fixture_pickup_cycle"
            load_dir.mkdir(parents=True)
            (load_dir / "physical.trace.json").write_text(
                json.dumps(
                    {
                        "branch": "physical",
                        "status": "COMPLETED",
                        "samples": [{"time_s": 0.0}],
                    }
                ),
                encoding="utf-8",
            )
            acceptance = (
                run
                / "results"
                / "fixture_capsule"
                / "task_acceptance.json"
            )
            acceptance.parent.mkdir(parents=True)
            acceptance.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "evaluations": [
                            {
                                "branch": "physical",
                                "verdict": "PASS",
                                "test_load_case_id": "fixture_pickup_cycle",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            accepted = _accepted_physical_task_load_case_ids(
                run,
                "fixture_capsule",
            )
            physical, reduced = _select_visual_trace_paths(
                trace_root,
                accepted_task_load_case_ids=accepted,
            )

            self.assertEqual(accepted, ("fixture_pickup_cycle",))
            self.assertEqual(physical, load_dir / "physical.trace.json")
            self.assertEqual(reduced, load_dir / "reduced.trace.json")

    def test_unsafe_task_acceptance_load_case_id_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            acceptance = (
                run
                / "results"
                / "fixture_capsule"
                / "task_acceptance.json"
            )
            acceptance.parent.mkdir(parents=True)
            acceptance.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "evaluations": [
                            {
                                "branch": "physical",
                                "passed": True,
                                "test_load_case_id": "../outside",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                _accepted_physical_task_load_case_ids(
                    run,
                    "fixture_capsule",
                ),
                (),
            )

    def test_fixture_rich_trace_is_reduced_to_two_body_whole_machine_scene(
        self,
    ) -> None:
        bodies = [
            {"body_id": "fixed_boundary", "fixed": True, "mesh": object()},
            {"body_id": "moving_group", "fixed": False, "mesh": object()},
            {
                "body_id": "spring_grip_test_object",
                "fixed": False,
                "mesh": object(),
            },
            {"body_id": "spring_pickup_table", "fixed": True, "mesh": object()},
        ]

        retained = _whole_machine_render_bodies(
            bodies,
            physical_trace={
                "collision_geometry": {
                    "fixed": {"body_origin_abs_m": [0.1, 0.2, 0.3]}
                }
            },
            whole_samples=[
                {
                    "time_s": 0.0,
                    "moving_pose": {
                        "position_abs_m": [0.0, 0.0, 0.0],
                        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                }
            ],
        )

        self.assertEqual(len(retained), 2)
        self.assertEqual(retained[0]["body_id"], "fixed_boundary")
        self.assertEqual(retained[0]["constant_pose"][0].tolist(), [0.1, 0.2, 0.3])
        self.assertIsNone(retained[1]["body_id"])



class WholeMachineTraceFallbackTests(unittest.TestCase):
    def test_fallback_backfills_decimated_samples_from_completed_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            trace_dir = run / "traces" / "chain_capsule" / "task_x"
            trace_dir.mkdir(parents=True)
            samples = [{"t": i} for i in range(2000)]
            (trace_dir / "whole_machine.trace.json").write_text(
                json.dumps({"status": "COMPLETED", "samples": samples}),
                encoding="utf-8",
            )
            merged = _whole_machine_payload_with_trace_samples(
                run, "chain_capsule", {"status": "WHOLE_MACHINE_TASK_PASSED"}
            )
            self.assertEqual(merged["sample_count"], 2000)
            self.assertEqual(len(merged["samples"]), 1000)
            self.assertEqual(merged["samples"][0], {"t": 0})
            self.assertEqual(merged["samples"][1], {"t": 2})

    def test_fallback_ignores_partial_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            trace_dir = run / "traces" / "chain_capsule" / "task_x"
            trace_dir.mkdir(parents=True)
            (trace_dir / "whole_machine.trace.json").write_text(
                json.dumps(
                    {"status": "PHYSICS_PARTIAL", "samples": [{"t": 0}]}
                ),
                encoding="utf-8",
            )
            payload = {"status": "WHOLE_MACHINE_TASK_PASSED"}
            merged = _whole_machine_payload_with_trace_samples(
                run, "chain_capsule", payload
            )
            self.assertIs(merged, payload)


if __name__ == "__main__":
    unittest.main()
