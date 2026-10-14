from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.sampling import (
    AdaptiveSamplingTracker,
    legacy_sampling_disclosure,
    maximum_pose_step_sweep_m,
    select_saved_trace_samples,
)


def _poses(x: float = 0.0) -> dict:
    return {
        "ground": {
            "position_abs_m": [0.0, 0.0, 0.0],
            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "moving": {
            "position_abs_m": [x, 0.0, 0.0],
            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
    }


class SamplingTests(unittest.TestCase):
    def test_pose_step_bound_tracks_translation(self) -> None:
        self.assertAlmostEqual(maximum_pose_step_sweep_m(_poses(), _poses(0.02)), 0.02)

    def test_tracker_keeps_nominal_and_event_samples(self) -> None:
        tracker = AdaptiveSamplingTracker(
            step_s=0.001,
            nominal_interval_s=0.05,
            interpenetration_tolerance_m=0.001,
            initial_poses=_poses(),
        )
        quiet = tracker.observe(time_s=0.001, poses=_poses(0.0001), contact_count=0)
        contact = tracker.observe(time_s=0.002, poses=_poses(0.0002), contact_count=1)
        fast = tracker.observe(time_s=0.003, poses=_poses(0.0010), contact_count=1)
        nominal = tracker.observe(time_s=0.05, poses=_poses(0.0011), contact_count=1)
        self.assertFalse(quiet.sample)
        self.assertEqual(contact.reasons, ("contact_count_transition",))
        self.assertIn("fast_pose_step", fast.reasons)
        self.assertIn("nominal_interval", nominal.reasons)

    def test_legacy_selector_retains_boundaries_and_contact_transitions(self) -> None:
        samples = [
            {"time_s": 0.0, "phase_id": "a", "contact_count": 0},
            {"time_s": 0.01, "phase_id": "a", "contact_count": 0},
            {"time_s": 0.02, "phase_id": "b", "contact_count": 0},
            {"time_s": 0.03, "phase_id": "b", "contact_count": 1},
            {"time_s": 0.05, "phase_id": "b", "contact_count": 1},
            {"time_s": 0.10, "phase_id": "b", "contact_count": 1},
        ]
        selected, disclosure = select_saved_trace_samples(samples)
        self.assertEqual(
            [item["time_s"] for item in selected],
            [0.0, 0.01, 0.02, 0.03, 0.05, 0.10],
        )
        self.assertEqual(disclosure["phase_boundary_trigger_count"], 1)
        self.assertEqual(disclosure["contact_transition_trigger_count"], 1)
    def test_legacy_disclosure_reports_actual_saved_gap(self) -> None:
        disclosure = legacy_sampling_disclosure(
            ({"time_s": 0.0}, {"time_s": 0.005}, {"time_s": 0.055})
        )
        self.assertEqual(disclosure["saved_sample_count"], 3)
        self.assertAlmostEqual(disclosure["minimum_sample_gap_s"], 0.005)
        self.assertAlmostEqual(disclosure["maximum_sample_gap_s"], 0.05)
        self.assertFalse(disclosure["adaptive_policy_recorded"])


if __name__ == "__main__":
    unittest.main()

