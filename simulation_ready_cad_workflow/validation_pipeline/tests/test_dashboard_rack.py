from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import (
    _comparison_checks,
    _whole_task_rows,
)


class RackPinionDashboardTests(unittest.TestCase):
    def test_rack_checks_use_rack_specific_gates(self) -> None:
        load = {
            "program": "bidirectional_rack_pinion",
            "phases": [
                {"phase_id": "crank_lift", "wrench_kind": "torque"},
                {"phase_id": "crank_lower", "wrench_kind": "torque"},
            ],
        }
        physical = {
            "min_progress_m": -0.0024,
            "max_progress_m": 0.1823,
            "max_coupling_error_m": 0.00191,
            "max_support_drift_m": 0.000063,
            "max_off_axis_rotation_rad": 0.0,
            "max_bushing_reaction_force_n": 8.5,
            "phase_directional_travel_m": {
                "crank_lift": 0.18,
                "crank_lower": 0.17,
            },
        }
        reduced = {
            "min_progress_m": 0.0,
            "max_progress_m": 0.18,
            "max_coupling_error_m": 0.000035,
            "max_support_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "max_bushing_reaction_force_n": 0.0,
            "phase_directional_travel_m": {
                "crank_lift": 0.18,
                "crank_lower": 0.168,
            },
        }
        thresholds = {
            "home_progress_m": 0.0,
            "outbound_progress_m": 0.18,
            "output_progress_tolerance_m": 0.004,
            "max_support_drift_m": 0.0015,
            "max_off_axis_rotation_rad": 0.04,
            "max_physical_coupling_error_m": 0.002,
            "max_reduced_coupling_error_m": 0.00005,
            "min_directional_progress_m": 0.0072,
            "max_zero_input_drift_m": 0.004,
            "max_transverse_progress_m": 0.004,
        }

        rows = _comparison_checks(load, physical, reduced, thresholds)

        self.assertIn("真实啮合误差", {row["label"] for row in rows})
        self.assertIn("支撑模型确实不同", {row["label"] for row in rows})
        self.assertNotIn("真实接触证据", {row["label"] for row in rows})
        self.assertTrue(
            all(
                row["physical_pass"] is not False
                and row["reduced_pass"] is not False
                for row in rows
            )
        )

    def test_gate_tasks_are_named_and_matched_by_direction(self) -> None:
        whole = {
            "task_ids": ["lower_gate", "raise_gate"],
            "stages": [
                {
                    "stage": "outbound",
                    "target_progress_m": 0.18,
                    "max_progress_m": 0.18,
                    "reached": True,
                },
                {
                    "stage": "return",
                    "target_progress_m": 0.0,
                    "min_progress_m": 0.0,
                    "reached": True,
                },
            ],
            "samples": [
                {"control_torque_nm": 0.0},
                {"control_torque_nm": 3.2},
                {"control_torque_nm": 2.0},
            ],
        }
        plan = {
            "coordinate_kind": "rack_pinion_progress",
            "progress_tolerance_m": 0.004,
            "max_control_torque_nm": 3.2,
        }

        rows = _whole_task_rows(whole, plan)

        self.assertEqual([row["id"] for row in rows], ["raise_gate", "lower_gate"])
        self.assertEqual([row["label"] for row in rows], ["升起闸门", "降下闸门"])
        self.assertEqual(rows[0]["tolerance"], 4.0)
        self.assertEqual(rows[0]["effort_unit"], "N·m")
        self.assertEqual(rows[0]["max_effort"], 3.2)


if __name__ == "__main__":
    unittest.main()
