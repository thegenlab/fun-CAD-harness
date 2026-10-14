from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.acceptance_metrics import (
    MetricEvaluationError,
    evaluate_registered_metric,
    evaluate_task_acceptance,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    TaskProgramEvidenceSpec,
    ValidatedTraceProgram,
)


def _requirement(
    metric_id: str,
    target: dict,
    *,
    direction: str = "either",
) -> dict:
    return {
        "requirement_id": "metric_test",
        "task_id": "task",
        "metric_id": metric_id,
        "direction": direction,
        "target": target,
    }


def _binding(*, reference: bool = False) -> dict:
    value = {
        "target": {"observable_id": "output_wheel_pose"},
        "input": {"port_id": "input_crank_torque"},
    }
    if reference:
        value["reference"] = {"observable_id": "input_crank_pose"}
    return value


def _trace(samples: list[dict], *, branch: str = "physical") -> dict:
    return {
        "status": "COMPLETED",
        "branch": branch,
        "samples": samples,
    }


class AcceptanceMetricTests(unittest.TestCase):
    def test_signed_displacement_uses_saved_prepowered_baseline(self) -> None:
        validated = ValidatedTraceProgram(
            program_spec=TaskProgramEvidenceSpec(
                program="spring_scale_calibration",
                powered_phase_ids=("settled_loaded",),
                return_phase_ids=("unloaded_zero",),
                return_input_policy="zero",
                adaptive_phase_ids=(),
                minimum_powered_sample_count=3,
                maximum_sample_gap_cadences=1.0,
            ),
            powered_sample_indices=(4, 5, 6),
            return_sample_indices=(1, 2),
            powered_baseline_index=3,
            home_sample_index=0,
            final_sample_index=6,
            executed_phase_ids=(
                "unloaded_zero",
                "gravity_mass_loading",
                "settled_loaded",
            ),
            executed_load_sha256="0" * 64,
            max_sample_gap_s=0.02,
        )
        result = evaluate_registered_metric(
            _requirement(
                "signed_displacement_m",
                {
                    "comparator": "within",
                    "value": 0.01,
                    "tolerance": 0.002,
                    "unit": "m",
                },
                direction="positive",
            ),
            _binding(),
            _trace(
                [
                    {"generalized_progress_m": 0.0},
                    {"generalized_progress_m": -0.00001},
                    {"generalized_progress_m": 0.0},
                    {"generalized_progress_m": 0.009},
                    {"generalized_progress_m": 0.0095},
                    {"generalized_progress_m": 0.0099},
                    {"generalized_progress_m": 0.010},
                ]
            ),
            canonical_generalized_observable_id="output_wheel_pose",
            validated_program=validated,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.01)

    def test_loaded_repeatability_compares_terminal_loaded_windows(self) -> None:
        validated = ValidatedTraceProgram(
            program_spec=TaskProgramEvidenceSpec(
                program="spring_scale_repeatability",
                powered_phase_ids=("cycle_1_settled_loaded",),
                return_phase_ids=("cycle_3_settled_loaded",),
                return_input_policy="selected_input_only",
                adaptive_phase_ids=(),
                minimum_powered_sample_count=3,
                maximum_sample_gap_cadences=1.0,
            ),
            powered_sample_indices=(1, 2, 3),
            return_sample_indices=(4, 5, 6),
            powered_baseline_index=0,
            home_sample_index=0,
            final_sample_index=6,
            executed_phase_ids=(
                "cycle_1_settled_loaded",
                "cycle_3_settled_loaded",
            ),
            executed_load_sha256="0" * 64,
            max_sample_gap_s=0.02,
        )
        requirement = _requirement(
            "return_to_start_m",
            {"comparator": "at_most", "value": 0.002, "unit": "m"},
        )
        requirement["reference_role"] = (
            "first settled 1.0 kg loaded pointer pose"
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"generalized_progress_m": 0.0},
                    {"generalized_progress_m": 0.0198},
                    {"generalized_progress_m": 0.0200},
                    {"generalized_progress_m": 0.0201},
                    {"generalized_progress_m": 0.0199},
                    {"generalized_progress_m": 0.0200},
                    {"generalized_progress_m": 0.0202},
                ]
            ),
            canonical_generalized_observable_id="output_wheel_pose",
            validated_program=validated,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.0001)
        self.assertIn(
            "terminal_powered_and_terminal_return",
            result["evidence"]["source_field"],
        )

    def test_full_turn_uses_continuous_coordinate_not_endpoint_pose(self) -> None:
        requirement = _requirement(
            "continuous_rotation_rad",
            {
                "comparator": "at_least",
                "value": 2.0 * math.pi,
                "unit": "rad",
            },
            direction="positive",
        )
        trace = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": math.pi},
                {"generalized_angle_rad": 2.0 * math.pi},
            ]
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            canonical_generalized_observable_id="output_wheel_pose",
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 2.0 * math.pi)
        self.assertTrue(
            result["evidence"][
                "endpoint_pose_equivalence_is_not_sufficient"
            ]
        )

    def test_positive_direction_does_not_accept_negative_motion(self) -> None:
        requirement = _requirement(
            "continuous_rotation_rad",
            {
                "comparator": "at_least",
                "value": 1.0,
                "unit": "rad",
            },
            direction="positive",
        )
        trace = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": -2.0},
            ]
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            canonical_generalized_observable_id="output_wheel_pose",
        )

        self.assertFalse(result["passed"])
        self.assertEqual(result["measured"], 0.0)

    def test_return_to_start_uses_terminal_angle_error(self) -> None:
        requirement = _requirement(
            "return_to_start_rad",
            {
                "comparator": "at_most",
                "value": 0.05,
                "unit": "rad",
            },
        )
        trace = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": 0.5},
                {"generalized_angle_rad": 0.02},
            ]
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            canonical_generalized_observable_id="output_wheel_pose",
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.02)

    def test_hold_reference_measures_drift_from_hold_window_start(self) -> None:
        requirement = _requirement(
            "return_to_start_rad",
            {
                "comparator": "at_most",
                "value": 0.05235987755982989,
                "unit": "rad",
            },
        )
        requirement["reference_role"] = (
            "support-plate pose at the start of the loaded hold"
        )
        program_spec = TaskProgramEvidenceSpec(
            program="task_phone_stand_loaded_hold",
            powered_phase_ids=("reach_working_detent",),
            return_phase_ids=("loaded_phone_hold",),
            return_input_policy="zero",
            adaptive_phase_ids=("reach_working_detent",),
            minimum_powered_sample_count=3,
            maximum_sample_gap_cadences=1.0,
        )
        validated = ValidatedTraceProgram(
            program_spec=program_spec,
            powered_sample_indices=(1, 2),
            return_sample_indices=(3, 4, 5),
            powered_baseline_index=0,
            home_sample_index=0,
            final_sample_index=5,
            executed_phase_ids=(
                "reach_working_detent",
                "loaded_phone_hold",
            ),
            executed_load_sha256="0" * 64,
            max_sample_gap_s=0.05,
        )
        trace = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": 0.5},
                {"generalized_angle_rad": 1.0470},
                {"generalized_angle_rad": 1.0472},
                {"generalized_angle_rad": 1.0500},
                {"generalized_angle_rad": 1.0480},
            ]
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            canonical_generalized_observable_id="output_wheel_pose",
            validated_program=validated,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.0028)
        self.assertEqual(
            result["evidence"]["source_field"],
            "generalized_angle_rad@zero_input_hold_window",
        )

    def test_return_lower_bound_measures_positive_retained_terminal_travel(self) -> None:
        requirement = _requirement(
            "return_to_start_rad",
            {
                "comparator": "at_least",
                "value": 0.3,
                "unit": "rad",
            },
            direction="positive",
        )

        retained = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"generalized_angle_rad": 0.0},
                    {"generalized_angle_rad": 0.5},
                    {"generalized_angle_rad": 0.35},
                ]
            ),
            canonical_generalized_observable_id="output_wheel_pose",
        )

        self.assertTrue(retained["passed"])
        self.assertAlmostEqual(retained["measured"], 0.35)

    def test_return_lower_bound_rejects_opposite_terminal_direction(self) -> None:
        requirement = _requirement(
            "return_to_start_rad",
            {
                "comparator": "at_least",
                "value": 0.3,
                "unit": "rad",
            },
            direction="positive",
        )

        reversed_result = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"generalized_angle_rad": 0.0},
                    {"generalized_angle_rad": 0.5},
                    {"generalized_angle_rad": -0.35},
                ]
            ),
            canonical_generalized_observable_id="output_wheel_pose",
        )

        self.assertFalse(reversed_result["passed"])
        self.assertAlmostEqual(reversed_result["measured"], -0.35)

    def test_ratio_uses_two_distinct_named_observables(self) -> None:
        requirement = _requirement(
            "angular_ratio",
            {
                "comparator": "within",
                "value": 2.0,
                "tolerance": 0.01,
                "unit": "ratio",
            },
        )
        trace = _trace(
            [
                {
                    "output_wheel_angle_rad": 0.0,
                    "input_crank_angle_rad": 0.0,
                },
                {
                    "output_wheel_angle_rad": 4.0,
                    "input_crank_angle_rad": 2.0,
                },
            ]
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(reference=True),
            trace,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["measured"], 2.0)
        self.assertEqual(
            result["evidence"]["reference_field"],
            "input_crank_angle_rad",
        )

    def test_roundtrip_ratio_uses_powered_peak_not_returned_endpoint(self) -> None:
        requirement = _requirement(
            "angular_ratio",
            {
                "comparator": "within",
                "value": 2.0,
                "tolerance": 0.01,
                "unit": "ratio",
            },
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(reference=True),
            _trace(
                [
                    {
                        "output_wheel_angle_rad": 0.0,
                        "input_crank_angle_rad": 0.0,
                    },
                    {
                        "output_wheel_angle_rad": 4.0,
                        "input_crank_angle_rad": 2.0,
                    },
                    {
                        "output_wheel_angle_rad": 0.0,
                        "input_crank_angle_rad": 0.0,
                    },
                ]
            ),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["measured"], 2.0)
        self.assertEqual(result["evidence"]["ratio_sample_index"], 1)
        self.assertEqual(
            result["evidence"]["ratio_window"],
            "diagnostic_initial_to_max_reference_excursion",
        )
        self.assertTrue(
            result["evidence"]["ratio_consistency_passed"]
        )

    def test_ratio_rejects_one_correct_peak_amid_unrelated_motion(self) -> None:
        requirement = _requirement(
            "angular_ratio",
            {
                "comparator": "within",
                "value": 2.0,
                "tolerance": 0.01,
                "unit": "ratio",
            },
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(reference=True),
            _trace(
                [
                    {
                        "output_wheel_angle_rad": 0.0,
                        "input_crank_angle_rad": 0.0,
                    },
                    {
                        "output_wheel_angle_rad": 100.0,
                        "input_crank_angle_rad": 1.0,
                    },
                    {
                        "output_wheel_angle_rad": 4.0,
                        "input_crank_angle_rad": 2.0,
                    },
                    {
                        "output_wheel_angle_rad": -100.0,
                        "input_crank_angle_rad": 1.0,
                    },
                    {
                        "output_wheel_angle_rad": 0.0,
                        "input_crank_angle_rad": 0.0,
                    },
                ]
            ),
        )

        self.assertEqual(result["measured"], 2.0)
        self.assertFalse(result["passed"])
        self.assertFalse(
            result["evidence"]["ratio_consistency_passed"]
        )
        self.assertGreater(
            result["evidence"]["ratio_max_pointwise_error"],
            0.01,
        )
        self.assertGreater(
            result["evidence"]["ratio_max_incremental_error"],
            0.01,
        )

    def test_ratio_window_leaves_return_error_to_return_metric(self) -> None:
        requirement = _requirement(
            "angular_ratio",
            {
                "comparator": "within",
                "value": 2.0,
                "tolerance": 0.01,
                "unit": "ratio",
            },
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(reference=True),
            _trace(
                [
                    {
                        "output_wheel_angle_rad": 0.0,
                        "input_crank_angle_rad": 0.0,
                    },
                    {
                        "output_wheel_angle_rad": 2.0,
                        "input_crank_angle_rad": 1.0,
                    },
                    {
                        "output_wheel_angle_rad": 4.0,
                        "input_crank_angle_rad": 2.0,
                    },
                    {
                        "output_wheel_angle_rad": 0.2,
                        "input_crank_angle_rad": 0.0,
                    },
                ]
            ),
        )

        self.assertTrue(result["passed"])
        self.assertTrue(
            result["evidence"]["ratio_consistency_passed"]
        )

    def test_generalized_coordinate_cannot_substitute_another_target(self) -> None:
        requirement = _requirement(
            "continuous_rotation_rad",
            {
                "comparator": "at_least",
                "value": 1.0,
                "unit": "rad",
            },
        )
        with self.assertRaisesRegex(
            MetricEvaluationError,
            "bound observable",
        ):
            evaluate_registered_metric(
                requirement,
                _binding(),
                _trace(
                    [
                        {"generalized_angle_rad": 0.0},
                        {"generalized_angle_rad": 2.0},
                    ]
                ),
                canonical_generalized_observable_id="input_crank_pose",
            )

    def test_expected_contact_prefers_declared_fixture_pair_count(self) -> None:
        requirement = _requirement(
            "expected_contact",
            {"comparator": "equals", "value": True, "unit": "bool"},
        )
        unrelated_only = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"contact_count": 0, "expected_contact_count": 0},
                    {"contact_count": 4, "expected_contact_count": 0},
                ]
            ),
        )
        fixture_contact = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"contact_count": 0, "expected_contact_count": 0},
                    {"contact_count": 4, "expected_contact_count": 1},
                ]
            ),
        )
        self.assertFalse(unrelated_only["passed"])
        self.assertTrue(fixture_contact["passed"])
        self.assertEqual(
            fixture_contact["evidence"]["source_field"],
            "expected_contact_count",
        )

    def test_rolling_slip_uses_public_powered_travel_window(self) -> None:
        requirement = _requirement(
            "rolling_slip_ratio",
            {"comparator": "at_most", "value": 0.12, "unit": "ratio"},
        )
        passing = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"chassis_position_m": 0.0, "rolling_slip_ratio": 0.9},
                    {"chassis_position_m": 0.05, "rolling_slip_ratio": 0.8},
                    {"chassis_position_m": 0.20, "rolling_slip_ratio": 0.10},
                ]
            ),
        )
        failing = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"chassis_position_m": 0.0, "rolling_slip_ratio": 0.0},
                    {"chassis_position_m": 0.20, "rolling_slip_ratio": 0.13},
                ]
            ),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.10)
        self.assertFalse(failing["passed"])
        self.assertEqual(
            passing["evidence"]["source_field"],
            "rolling_slip_ratio@powered_after_0.10m",
        )

    def test_powered_contact_coverage_uses_every_powered_sample(self) -> None:
        requirement = _requirement(
            "powered_contact_coverage_ratio",
            {"comparator": "at_least", "value": 0.90, "unit": "ratio"},
        )
        passing = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    *({"expected_contact_count": 1} for _ in range(9)),
                    {"expected_contact_count": 0},
                ]
            ),
        )
        failing = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    *({"expected_contact_count": 1} for _ in range(8)),
                    {"expected_contact_count": 0},
                    {"expected_contact_count": 0},
                ]
            ),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.90)
        self.assertFalse(failing["passed"])

    def test_ground_separation_uses_worst_powered_sample(self) -> None:
        requirement = _requirement(
            "ground_separation_m",
            {"comparator": "at_most", "value": 0.0002, "unit": "m"},
        )
        passing = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"max_ground_separation_m": 0.0},
                    {"max_ground_separation_m": 0.00019},
                ]
            ),
        )
        failing = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"max_ground_separation_m": 0.0},
                    {"max_ground_separation_m": 0.00021},
                ]
            ),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.00019)
        self.assertFalse(failing["passed"])

    def test_tong_two_sided_grip_requires_same_saved_grip_phase(self) -> None:
        requirement = _requirement(
            "tong_two_sided_grip",
            {"comparator": "equals", "value": True, "unit": "bool"},
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {
                        "phase_id": "tong_grip_verify",
                        "test_object_two_sided_contact": False,
                    },
                    {
                        "phase_id": "tong_grip_verify",
                        "test_object_two_sided_contact": True,
                    },
                ]
            ),
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["measured"])

    def test_tong_tabletop_pickup_metrics_use_settled_and_lifted_states(self) -> None:
        samples = [
            {
                "phase_id": "tong_pickup_initial_table_settle",
                "test_object_table_contact_count": 2,
                "test_object_height_above_table_m": 0.0001,
                "test_object_position_abs_m": [0.158, 0.0, 0.0],
            },
            {
                "phase_id": "tong_pickup_verify",
                "test_object_two_sided_contact": True,
                "test_object_table_contact_count": 0,
                "test_object_height_above_table_m": 0.029,
                "test_object_position_abs_m": [0.158, 0.0, 0.030],
            },
        ]
        checks = (
            (
                "tong_object_starts_on_table",
                {"comparator": "equals", "value": True, "unit": "bool"},
            ),
            (
                "tong_two_sided_grip",
                {"comparator": "equals", "value": True, "unit": "bool"},
            ),
            (
                "tong_pickup_lift_m",
                {"comparator": "at_least", "value": 0.020, "unit": "m"},
            ),
            (
                "tong_pickup_table_clear",
                {"comparator": "equals", "value": True, "unit": "bool"},
            ),
        )
        for metric_id, target in checks:
            with self.subTest(metric_id=metric_id):
                result = evaluate_registered_metric(
                    _requirement(metric_id, target),
                    _binding(),
                    _trace(samples),
                )
                self.assertTrue(result["passed"], result)

    def test_tong_hold_relative_slip_tracks_object_against_tool(self) -> None:
        result = evaluate_registered_metric(
            _requirement(
                "tong_hold_relative_slip_m",
                {"comparator": "at_most", "value": 0.003, "unit": "m"},
            ),
            _binding(),
            _trace(
                [
                    {
                        "phase_id": "tong_hold_verify",
                        "test_object_relative_to_fixed_abs_m": [0.10, 0.01, 0.00],
                    },
                    {
                        "phase_id": "tong_hold_verify",
                        "test_object_relative_to_fixed_abs_m": [0.101, 0.011, 0.001],
                    },
                ]
            ),
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], math.sqrt(3.0) * 0.001)

    def test_tong_hold_relative_slip_accepts_validated_wood_rod_fixture(
        self,
    ) -> None:
        samples = [
            {
                "phase_id": "tong_hold_verify",
                "passive_fixture_target_relative_position_m": [
                    0.048,
                    0.0,
                    0.0,
                ],
                "passive_fixture_bilateral_contact_active": True,
                "direct_passive_fixture_actuation": False,
            },
            {
                "phase_id": "tong_hold_verify",
                "passive_fixture_target_relative_position_m": [
                    0.048,
                    -0.001,
                    0.0,
                ],
                "passive_fixture_bilateral_contact_active": True,
                "direct_passive_fixture_actuation": False,
            },
        ]
        trace = _trace(samples)
        trace["branch_semantics"] = {
            "external_passive_fixture": {
                "shape": "passive_wood_rod_cylinder",
                "source": "installed_validator_wood_rod_fixture_v1",
                "radius_m": 0.004,
                "length_m": 0.080,
                "required_contact_body_ids": ["left_jaw", "right_jaw"],
                "minimum_contact_sample_fraction": 0.8,
                "contact_sample_fractions_by_body": {
                    "left_jaw": 1.0,
                    "right_jaw": 1.0,
                },
                "evidence_sample_count": 2,
                "runtime_enabled": True,
                "supported": True,
                "direct_actuation": False,
            }
        }

        result = evaluate_registered_metric(
            _requirement(
                "tong_hold_relative_slip_m",
                {"comparator": "at_most", "value": 0.002, "unit": "m"},
            ),
            _binding(),
            trace,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.001)
        self.assertEqual(
            result["evidence"]["source_field"],
            "passive_fixture_target_relative_position_m@tong_hold_verify",
        )

    def test_return_to_start_m_uses_bound_output_point_vector(self) -> None:
        requirement = _requirement(
            "return_to_start_m",
            {"comparator": "at_most", "value": 0.001, "unit": "m"},
        )
        requirement["observable_quantity"] = "position_3d"
        plan = SimpleNamespace(
            adapter_spec={"family_spec": {"output_body_id": "right_jaw"}}
        )
        resolved_channels = {
            "target": {
                "body_id": "right_jaw",
                "canonical_field": "compound_output_angle_rad",
            }
        }
        trace = _trace(
            [
                {
                    "observable_point_abs_m": [0.053, 0.0, -0.006],
                    "compound_output_angle_rad": 0.0,
                },
                {
                    "observable_point_abs_m": [0.0527, 0.0, -0.0062],
                    "compound_output_angle_rad": 0.03,
                },
            ]
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            plan=plan,
            resolved_channels=resolved_channels,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(
            result["measured"], math.sqrt(0.0003**2 + 0.0002**2)
        )
        self.assertEqual(
            result["evidence"]["source_field"],
            "observable_point_abs_m@home_and_final",
        )

    def test_return_to_start_m_converts_ideal_angle_to_tip_displacement(
        self,
    ) -> None:
        requirement = _requirement(
            "return_to_start_m",
            {"comparator": "at_most", "value": 0.001, "unit": "m"},
        )
        requirement["observable_quantity"] = "position_3d"
        plan = SimpleNamespace(
            adapter_spec={
                "family_spec": {
                    "output_body_id": "right_jaw",
                    "output_joint_id": "right_joint",
                    "output_observable_point_abs_m": [0.050, 0.0, 0.0],
                    "revolute_joints": [
                        {
                            "joint_id": "right_joint",
                            "origin_abs_m": [0.0, 0.0, 0.0],
                            "axis_abs": [0.0, 1.0, 0.0],
                        }
                    ],
                }
            }
        )
        resolved_channels = {
            "target": {
                "body_id": "right_jaw",
                "canonical_field": "compound_output_angle_rad",
            }
        }
        trace = _trace(
            [
                {"compound_output_angle_rad": 0.0},
                {"compound_output_angle_rad": 0.01},
            ],
            branch="ideal_reference",
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            plan=plan,
            resolved_channels=resolved_channels,
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.1 * math.sin(0.005))
        self.assertEqual(
            result["evidence"]["source_field"],
            "compound_output_angle_rad+output_observable_radius_m@home_and_final",
        )

    def test_tong_release_requires_clear_jaws_and_table_settle(self) -> None:
        samples = [
            {
                "phase_id": "tong_release_passive_return",
                "test_object_fixed_jaw_contact_count": 1,
                "test_object_moving_jaw_contact_count": 1,
            },
            {
                "phase_id": "tong_release_observe",
                "test_object_fixed_jaw_contact_count": 0,
                "test_object_moving_jaw_contact_count": 0,
                "test_object_table_contact_count": 3,
                "test_object_height_above_table_m": 0.0002,
            },
        ]
        for metric_id in (
            "tong_release_contact_clear",
            "tong_release_returns_to_table",
        ):
            with self.subTest(metric_id=metric_id):
                result = evaluate_registered_metric(
                    _requirement(
                        metric_id,
                        {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    ),
                    _binding(),
                    _trace(samples),
                )
                self.assertTrue(result["passed"], result)

    def test_tong_hold_slip_uses_only_unsupported_hold_interval(self) -> None:
        requirement = _requirement(
            "tong_hold_slip_m",
            {"comparator": "at_most", "value": 0.003, "unit": "m"},
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(),
            _trace(
                [
                    {"phase_id": "tong_hold_close", "test_object_position_m": 0.020},
                    {"phase_id": "tong_hold_verify", "test_object_position_m": 0.0100},
                    {"phase_id": "tong_hold_verify", "test_object_position_m": 0.0085},
                ]
            ),
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.0015)

    def test_tong_release_requires_drop_and_final_contact_clear(self) -> None:
        samples = [
            {
                "phase_id": "tong_release_passive_return",
                "test_object_position_m": 0.010,
                "test_object_fixed_jaw_contact_count": 1,
                "test_object_moving_jaw_contact_count": 1,
            },
            {
                "phase_id": "tong_release_observe",
                "test_object_position_m": -0.002,
                "test_object_fixed_jaw_contact_count": 0,
                "test_object_moving_jaw_contact_count": 0,
            },
        ]
        drop = evaluate_registered_metric(
            _requirement(
                "tong_release_drop_m",
                {"comparator": "at_least", "value": 0.006, "unit": "m"},
            ),
            _binding(),
            _trace(samples),
        )
        clear = evaluate_registered_metric(
            _requirement(
                "tong_release_contact_clear",
                {"comparator": "equals", "value": True, "unit": "bool"},
            ),
            _binding(),
            _trace(samples),
        )
        self.assertTrue(drop["passed"])
        self.assertAlmostEqual(drop["measured"], 0.012)
        self.assertTrue(clear["passed"])

    def test_tong_metrics_fail_closed_without_task_specific_evidence(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            evaluate_registered_metric(
                _requirement(
                    "tong_hold_slip_m",
                    {"comparator": "at_most", "value": 0.003, "unit": "m"},
                ),
                _binding(),
                _trace([
                    {"phase_id": "unrelated", "test_object_position_m": 0.0},
                    {"phase_id": "unrelated", "test_object_position_m": -1.0},
                ]),
            )
        self.assertEqual(caught.exception.code, "TONG_HOLD_EVIDENCE_MISSING")

    def test_penetration_metric_fails_closed_without_pose_evidence(self) -> None:
        requirement = _requirement(
            "forbidden_penetration_m",
            {
                "comparator": "at_most",
                "value": 0.0001,
                "unit": "m",
            },
        )

        with self.assertRaises(MetricEvaluationError) as caught:
            evaluate_registered_metric(
                requirement,
                _binding(),
                _trace([{"time_s": 0.0}, {"time_s": 1.0}]),
            )

        self.assertEqual(
            caught.exception.code,
            "TASK_PENETRATION_EVIDENCE_MISSING",
        )

    def test_actuation_provenance_requires_explicit_indirect_evidence(self) -> None:
        requirement = _requirement(
            "actuation_provenance",
            {
                "comparator": "equals",
                "value": "indirect_only",
                "unit": "enum",
            },
        )
        trace = _trace([{"time_s": 0.0}, {"time_s": 1.0}])
        trace["branch_semantics"] = {
            "direct_hand_actuation": False,
            "direct_hammer_actuation": False,
        }

        with self.assertRaises(MetricEvaluationError) as raised:
            evaluate_registered_metric(
                requirement,
                _binding(),
                trace,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_ACTUATION_EVIDENCE_MISSING",
        )


class TaskAcceptanceProgramBindingTests(unittest.TestCase):
    def _plan(self):
        requirement = _requirement(
            "continuous_rotation_rad",
            {
                "comparator": "at_least",
                "value": 1.0,
                "unit": "rad",
            },
            direction="positive",
        )
        requirement.update(
            {
                "test_program": "task_clock_cycle",
                "test_load_case_id": "task_clock_case",
                "required_branches": ["physical", "ideal_reference"],
            }
        )
        return SimpleNamespace(
            observable_id="output_wheel_pose",
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "contract_sha256": "a" * 64,
                "task_contract_sha256": "b" * 64,
                "requirements": [requirement],
            },
            semantic_binding={
                "status": "PASS",
                "certificate_authority": True,
                "task_binding_sha256": "c" * 64,
                "bindings": [
                    {
                        "requirement_id": "metric_test",
                        **_binding(),
                    }
                ],
            },
        )

    def test_requirement_uses_only_bound_program(self) -> None:
        passing = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": 2.0},
            ]
        )
        ideal = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": 2.0},
            ],
            branch="ideal_reference",
        )
        unrelated = _trace(
            [
                {"generalized_angle_rad": 0.0},
                {"generalized_angle_rad": 0.0},
            ]
        )

        result = evaluate_task_acceptance(
            self._plan(),
            {
                "task_clock_case": {
                    "program": "task_clock_cycle",
                    "physical": passing,
                    "ideal_reference": ideal,
                },
                "zero_input_case": {
                    "program": "zero_input_clock_train",
                    "physical": unrelated,
                },
            },
        )

        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertTrue(
            all("zero_input_case" not in item for item in result["failure_reasons"])
        )

    def test_missing_bound_program_is_inconclusive(self) -> None:
        result = evaluate_task_acceptance(
            self._plan(),
            {"zero_input_case": {"program": "zero_input_clock_train", "physical": _trace([])}},
        )

        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertFalse(result["certificate_authority"])
        self.assertIn(
            "metric_test:task_clock_case:load_case_missing",
            result["failure_reasons"],
        )

if __name__ == "__main__":
    unittest.main()
