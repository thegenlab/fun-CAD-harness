from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import (
    _clock_comparison_checks,
    _whole_task_rows,
)


class DashboardClockTests(unittest.TestCase):
    def test_whole_machine_clock_tasks_are_not_treated_as_open_close_stages(self) -> None:
        whole = {
            "task_ids": [
                "run_three_hand_clock",
                "verify_three_hand_ratio",
                "set_time_manually",
                "ring_twin_bell_alarm",
            ],
            "task_results": [
                {"task_id": "run_three_hand_clock", "passed": True, "evidence": {"minute_span_rad": 0.5}},
                {"task_id": "verify_three_hand_ratio", "passed": True, "evidence": {"hour_to_minute_ratio": 1 / 12, "second_to_minute_ratio": 60.0, "same_display_direction": True}},
                {"task_id": "set_time_manually", "passed": True, "evidence": {"setting_crown_span_rad": 0.1, "minute_span_rad": 0.1}},
                {"task_id": "ring_twin_bell_alarm", "passed": True, "evidence": {"alarm_hammer_swing_rad": 0.2}},
            ],
        }
        plan = {
            "coordinate_kind": "clock_multifunction",
            "expected_hour_to_minute_ratio": 1 / 12,
            "expected_second_to_minute_ratio": 60.0,
        }

        rows = _whole_task_rows(whole, plan)

        self.assertEqual(4, len(rows))
        self.assertTrue(all(item["status"] == "passed" for item in rows))
        self.assertTrue(all(item["coordinate_kind"] == "clock_multifunction" for item in rows))
        self.assertIn("秒针/分针", rows[1]["summary"])
        self.assertIn("后冠", rows[2]["summary"])

    def test_clock_gate_table_does_not_require_tooth_contact_count(self) -> None:
        load = {"program": "task_clock_cycle"}
        thresholds = {
            "source": "validated_clock_gear_family_spec",
            "max_support_drift_m": 0.0012,
            "max_off_axis_rotation_rad": 0.05,
            "max_mesh_error_rad": 0.1,
            "minimum_alarm_swing_rad": 0.08,
            "expected_hour_to_minute_ratio": 1 / 12,
            "expected_second_to_minute_ratio": 60.0,
            "hand_ratio_relative_tolerance": 0.02,
            "output_angle_tolerance_rad": 0.08,
            "outbound_output_angle_rad": 0.35,
        }
        metrics = {
            "max_support_drift_m": 1e-7,
            "max_off_axis_rotation_rad": 1e-7,
            "max_mesh_error_rad": 0.001,
            "hour_to_minute_ratio": 1 / 12,
            "second_to_minute_ratio": 60.01,
            "hand_direction_consistent": True,
            "phase_output_travel_rad": {"wind_and_advance": 0.4, "controlled_return": -0.2},
            "alarm_swing_rad": 0.2,
            "max_output_angle_rad": 0.5,
            "contact_observed": False,
        }

        rows = _clock_comparison_checks(load, metrics, metrics, thresholds)

        self.assertFalse(any(item["label"] == "真实接触证据" for item in rows))
        self.assertTrue(all(item["physical_pass"] is not False for item in rows))
        self.assertTrue(all(item["reduced_pass"] is not False for item in rows))


if __name__ == "__main__":
    unittest.main()
