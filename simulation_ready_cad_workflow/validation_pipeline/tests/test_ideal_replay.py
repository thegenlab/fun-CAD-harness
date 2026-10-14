from __future__ import annotations

import math
import unittest

from simulation_ready_cad_workflow.validation_pipeline.dashboard.ideal_replay import (
    synthesize_ideal_kinematic_trace,
)


def _pose(x: float, y: float, z: float) -> dict:
    return {
        "position_abs_m": [x, y, z],
        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


class IdealReplayTests(unittest.TestCase):
    def test_rack_pinion_pose_replay_uses_closed_form_coordinates(self) -> None:
        plan = {
            "family": "compound",
            "adapter_spec": {
                "family_spec": {
                    "mechanism_kind": "rack_pinion",
                    "pinion_body_id": "pinion",
                    "rack_body_id": "rack",
                    "pinion_origin_abs_m": [0.0, 0.0, 0.0],
                    "pinion_axis_abs": [0.0, 1.0, 0.0],
                    "rack_axis_abs": [0.0, 0.0, 1.0],
                    "bodies": [
                        {
                            "body_id": "ground",
                            "fixed": True,
                            "center_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "body_id": "pinion",
                            "fixed": False,
                            "center_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "body_id": "rack",
                            "fixed": False,
                            "center_abs_m": [0.1, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    ],
                }
            },
        }
        physical = {"samples": [{"body_poses": {}}]}
        ideal = {
            "branch": "ideal_reference",
            "samples": [
                {
                    "time_s": 0.0,
                    "phase_id": "home",
                    "pinion_angle_rad": 0.0,
                    "rack_progress_m": 0.0,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "outbound",
                    "pinion_angle_rad": math.pi / 2.0,
                    "rack_progress_m": 0.08,
                },
            ],
        }

        replay = synthesize_ideal_kinematic_trace(
            plan=plan,
            physical_trace=physical,
            ideal_trace=ideal,
        )

        final = replay["samples"][-1]["body_poses"]
        self.assertAlmostEqual(final["rack"]["position_abs_m"][2], 0.08)
        self.assertNotEqual(
            final["pinion"]["rotation_abs_quat_wxyz"],
            [1.0, 0.0, 0.0, 0.0],
        )
        self.assertFalse(
            replay["visual_replay_semantics"]["physics_engine_used"]
        )
        self.assertFalse(
            replay["visual_replay_semantics"]["contact_evidence_authority"]
        )

    def test_plain_revolute_uses_only_home_registration_from_physical(self) -> None:
        plan = {
            "family": "revolute",
            "axis_abs": [0.0, 0.0, 1.0],
            "joint_origin_abs_m": [0.0, 0.0, 0.0],
            "adapter_spec": {"family_spec": None},
        }
        physical = {
            "collision_geometry": {
                "bodies": [
                    {"body_id": "fixed", "fixed": True},
                    {"body_id": "moving", "fixed": False},
                ]
            },
            "samples": [
                {
                    "body_poses": {
                        "fixed": _pose(0.0, 0.0, 0.0),
                        "moving": _pose(1.0, 0.0, 0.0),
                    }
                }
            ],
        }
        ideal = {
            "branch": "ideal_reference",
            "samples": [
                {
                    "time_s": 0.0,
                    "phase_id": "home",
                    "generalized_angle_rad": 0.0,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "open",
                    "generalized_angle_rad": math.pi / 2.0,
                },
            ],
        }

        replay = synthesize_ideal_kinematic_trace(
            plan=plan,
            physical_trace=physical,
            ideal_trace=ideal,
        )

        final = replay["samples"][-1]["body_poses"]["moving"]
        self.assertAlmostEqual(final["position_abs_m"][0], 0.0, places=7)
        self.assertAlmostEqual(final["position_abs_m"][1], 1.0, places=7)
        self.assertTrue(
            replay["visual_replay_semantics"][
                "physical_home_pose_used_for_scene_registration"
            ]
        )
        self.assertFalse(
            replay["visual_replay_semantics"]["physical_output_poses_copied"]
        )

    def test_clock_rotates_each_reachable_shaft_without_physics(self) -> None:
        plan = {
            "family": "compound",
            "adapter_spec": {
                "family_spec": {
                    "mechanism_kind": "spur_gear_clock",
                    "bodies": [
                        {
                            "body_id": "ground",
                            "fixed": True,
                            "center_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "body_id": "minute",
                            "fixed": False,
                            "center_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    ],
                    "shafts": [
                        {
                            "body_id": "minute",
                            "origin_abs_m": [0.0, 0.0, 0.0],
                            "axis_abs": [0.0, 1.0, 0.0],
                        }
                    ],
                }
            },
        }
        ideal = {
            "branch": "ideal_reference",
            "samples": [
                {
                    "time_s": 0.0,
                    "phase_id": "home",
                    "shaft_angles_rad": {"minute": 0.0},
                },
                {
                    "time_s": 1.0,
                    "phase_id": "tick",
                    "shaft_angles_rad": {"minute": 0.5},
                },
            ],
        }

        replay = synthesize_ideal_kinematic_trace(
            plan=plan,
            physical_trace={"samples": [{"body_poses": {}}]},
            ideal_trace=ideal,
        )

        self.assertNotEqual(
            replay["samples"][-1]["body_poses"]["minute"][
                "rotation_abs_quat_wxyz"
            ],
            [1.0, 0.0, 0.0, 0.0],
        )
        self.assertEqual(
            replay["visual_replay_semantics"]["pose_model"],
            "clock_shaft_graph_pose_v1",
        )

    def test_multi_output_shaft_rotates_each_declared_body(self) -> None:
        plan = {
            "family": "compound",
            "adapter_spec": {
                "family_spec": {
                    "mechanism_kind": "crown_bevel_dual_output",
                    "bodies": [
                        {
                            "body_id": "housing",
                            "fixed": True,
                            "center_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "body_id": "crank",
                            "fixed": False,
                            "center_abs_m": [0.0, 0.0, 0.1],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "body_id": "left",
                            "fixed": False,
                            "center_abs_m": [-0.03, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        {
                            "body_id": "right",
                            "fixed": False,
                            "center_abs_m": [0.03, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    ],
                    "shafts": [
                        {
                            "body_id": "crank",
                            "origin_abs_m": [0.0, 0.0, 0.1],
                            "axis_abs": [0.0, 1.0, 0.0],
                        },
                        {
                            "body_id": "left",
                            "origin_abs_m": [-0.03, 0.0, 0.0],
                            "axis_abs": [0.0, 0.0, 1.0],
                        },
                        {
                            "body_id": "right",
                            "origin_abs_m": [0.03, 0.0, 0.0],
                            "axis_abs": [0.0, 0.0, 1.0],
                        },
                    ],
                }
            },
        }
        ideal = {
            "branch": "ideal_reference",
            "samples": [
                {
                    "time_s": 0.0,
                    "phase_id": "home",
                    "shaft_angles_rad": {
                        "crank": 0.0,
                        "left": 0.0,
                        "right": 0.0,
                    },
                },
                {
                    "time_s": 1.0,
                    "phase_id": "turn",
                    "shaft_angles_rad": {
                        "crank": 0.5,
                        "left": -1.5,
                        "right": 1.5,
                    },
                },
            ],
        }

        replay = synthesize_ideal_kinematic_trace(
            plan=plan,
            physical_trace={"samples": [{"body_poses": {}}]},
            ideal_trace=ideal,
        )

        final = replay["samples"][-1]["body_poses"]
        for body_id in ("crank", "left", "right"):
            self.assertNotEqual(
                final[body_id]["rotation_abs_quat_wxyz"],
                [1.0, 0.0, 0.0, 0.0],
            )
        self.assertEqual(
            replay["visual_replay_semantics"]["pose_model"],
            "multi_output_shaft_graph_pose_v1",
        )
        self.assertFalse(
            replay["visual_replay_semantics"]["physics_engine_used"]
        )


if __name__ == "__main__":
    unittest.main()
