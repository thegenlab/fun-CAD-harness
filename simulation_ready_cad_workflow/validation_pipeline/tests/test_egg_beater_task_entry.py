from __future__ import annotations

import copy
import hashlib
import json
import math
import unittest
from dataclasses import replace
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.intake.acceptance_contract import (
    METRIC_REGISTRY,
    AcceptanceContractError,
    compile_acceptance_contract,
    verify_acceptance_contract,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.egg_beater_task_suite import (
    EGG_BEATER_TASK_METRIC_VERSION,
    EGG_BEATER_TASK_SUITE_SHA256,
    EggBeaterTaskSuiteError,
    verify_egg_beater_task_suite_binding,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.acceptance_metrics import (
    MetricEvaluationError,
    evaluate_registered_metric,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismPlan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    task_program_spec_for_load_case,
)


def _egg_beater_requirements() -> dict:
    clauses = (
        "left beater turns at least one full revolution (0.1 rad shortfall); "
        "both beaters counter-rotate in opposite frozen-axis directions; "
        "beater speed magnitudes match 1:1 within 5%; "
        "beater-to-crank speedup is 3:1 within 5%; "
        "measured crank-to-beater direction matches the frozen axis bindings"
    )
    common = {
        "input_role": "hand crank",
        "observable_quantity": "pose_6d",
        "test_program": "task_egg_beater_cycle",
        "test_load_case_id": "task_egg_beater_cycle",
        "required_branches": ["physical", "ideal_reference"],
        "allow_direct_output_actuation": False,
    }
    return {
        "schema_version": "0.2",
        "requirements_id": "egg_beater_task_entry_test_v1",
        "prompt": clauses,
        "tasks": [
            {
                "task_id": "task.rotate_left_beater",
                "description": "Rotate the left beater through one revolution.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "criterion.left_continuous_rotation",
                        "prompt_clause": "left beater turns at least one full revolution (0.1 rad shortfall)",
                        "target_role": "left beater",
                        "metric_id": "continuous_rotation_rad",
                        "direction": "positive",
                        "target": {
                            "comparator": "at_least",
                            "value": 6.283185307179586,
                            "unit": "rad",
                            "tolerance": 0.1,
                        },
                    },
                ],
            },
            {
                "task_id": "task.counter_rotate_beaters",
                "description": "Counter-rotate the two beaters.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "criterion.opposite_directions",
                        "prompt_clause": "both beaters counter-rotate in opposite frozen-axis directions",
                        "target_role": "left beater",
                        "metric_id": "egg_beater_outputs_opposite_directions",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **common,
                        "requirement_id": "criterion.output_speed_magnitude_ratio",
                        "prompt_clause": "beater speed magnitudes match 1:1 within 5%",
                        "target_role": "left beater",
                        "reference_role": "right beater",
                        "metric_id": "egg_beater_output_speed_magnitude_ratio",
                        "target": {
                            "comparator": "within",
                            "value": 1.0,
                            "unit": "ratio",
                            "tolerance": 0.05,
                        },
                    },
                ],
            },
            {
                "task_id": "task.speed_up_from_crank",
                "description": "Increase beater speed from the crank.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "criterion.beater_crank_speed_magnitude_ratio",
                        "prompt_clause": "beater-to-crank speedup is 3:1 within 5%",
                        "target_role": "left beater",
                        "reference_role": "hand crank",
                        "metric_id": "egg_beater_crank_speed_magnitude_ratio",
                        "target": {
                            "comparator": "within",
                            "value": 3.0,
                            "unit": "ratio",
                            "tolerance": 0.15,
                        },
                    },
                    {
                        **common,
                        "requirement_id": "criterion.axis_bound_direction_consistency",
                        "prompt_clause": "measured crank-to-beater direction matches the frozen axis bindings",
                        "target_role": "left beater",
                        "reference_role": "hand crank",
                        "metric_id": "egg_beater_axis_bound_direction_consistency",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                ],
            },
        ],
    }


def _rehash(contract: dict) -> None:
    payload = {
        key: value
        for key, value in contract.items()
        if key != "contract_sha256"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    contract["contract_sha256"] = hashlib.sha256(encoded).hexdigest()


def _family_spec(*, left_axis=None, left_ratio=-3.0) -> dict:
    return {
        "mechanism_kind": "crown_bevel_dual_output",
        "mechanism_id": "household_hand_crank_egg_beater",
        "bodies": [
            {
                "body_id": "housing",
                "group_ids": ["group_housing"],
                "fixed": True,
                "mass_kg": 0.4,
                "center_abs_m": [0.0, 0.0, 0.1],
                "size_m": [0.1, 0.1, 0.2],
            },
        ]
        + [
            {
                "body_id": body_id,
                "group_ids": [f"group_{body_id}"],
                "fixed": False,
                "mass_kg": 0.1,
                "center_abs_m": center,
                "size_m": [0.03, 0.03, 0.05],
            }
            for body_id, center in (
                ("hand_crank", [0.0, -0.035, 0.158]),
                ("left_beater", [-0.024, 0.011, 0.112]),
                ("right_beater", [0.024, -0.019, 0.112]),
            )
        ],
        "shafts": [
            {
                "shaft_id": "crank_shaft",
                "body_id": "hand_crank",
                "origin_abs_m": [0.0, -0.004, 0.15],
                "axis_abs": [0.0, 1.0, 0.0],
            },
            {
                "shaft_id": "left_beater_shaft",
                "body_id": "left_beater",
                "origin_abs_m": [-0.024, 0.011, 0.112],
                "axis_abs": (
                    [0.0, 0.0, 1.0] if left_axis is None else list(left_axis)
                ),
            },
            {
                "shaft_id": "right_beater_shaft",
                "body_id": "right_beater",
                "origin_abs_m": [0.024, -0.019, 0.112],
                "axis_abs": [0.0, 0.0, 1.0],
            },
        ],
        "crown_bevel_meshes": [
            {
                "mesh_id": "crown_to_left",
                "driver_body_id": "hand_crank",
                "driven_body_id": "left_beater",
                "driver_teeth": 30,
                "driven_teeth": 10,
                "ratio": left_ratio,
                "shaft_angle_rad": 1.5707963267948966,
                "driver_pitch_radius_m": 0.045,
                "driven_pitch_radius_m": 0.015,
                "driver_face_width_m": 0.01,
                "driven_face_width_m": 0.001,
                "geometry_contact_tolerance_m": 0.001,
                "axis_perpendicular_tolerance_rad": 0.02,
                "mesh_stiffness_nm_rad": 30.0,
                "mesh_damping_nm_s_rad": 0.04,
            },
            {
                "mesh_id": "crown_to_right",
                "driver_body_id": "hand_crank",
                "driven_body_id": "right_beater",
                "driver_teeth": 30,
                "driven_teeth": 10,
                "ratio": 3.0,
                "shaft_angle_rad": 1.5707963267948966,
                "driver_pitch_radius_m": 0.045,
                "driven_pitch_radius_m": 0.015,
                "driver_face_width_m": 0.01,
                "driven_face_width_m": 0.001,
                "geometry_contact_tolerance_m": 0.001,
                "axis_perpendicular_tolerance_rad": 0.02,
                "mesh_stiffness_nm_rad": 30.0,
                "mesh_damping_nm_s_rad": 0.04,
            },
        ],
        "fixed_body_id": "housing",
        "input_body_id": "hand_crank",
        "output_body_ids": ["left_beater", "right_beater"],
        "nominal_input_torque_nm": 0.18,
        "return_input_torque_nm": 0.3,
        "max_torque_phase_s": 0.8,
        "max_return_phase_s": 0.8,
        "settle_phase_s": 0.08,
        "max_support_drift_m": 0.001,
        "max_off_axis_rotation_rad": 0.05,
        "max_mesh_error_rad": 0.08,
        "minimum_output_travel_rad": 1.0,
        "output_pair_ratio_tolerance": 0.08,
        "input_output_ratio_tolerance": 0.2,
    }


def _plan(family_spec: dict | None = None) -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=family_spec if family_spec is not None else _family_spec(),
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="hand_crank_egg_beater_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="egg_beater_crown_drive",
        capsule_file="egg_beater_crown_drive.json",
        family="compound",
        protocol_id="crown_bevel_dual_output",
        fixed_group_ids=("group_housing",),
        moving_group_ids=(
            "group_hand_crank",
            "group_left_beater",
            "group_right_beater",
        ),
        axis_abs=(0.0, 1.0, 0.0),
        joint_origin_abs_m=(0.0, -0.004, 0.15),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_hand_crank",
        input_port_id="hand_crank_torque",
        input_frame_id="frame_hand_crank",
        input_point_abs_m=(0.0, -0.004, 0.15),
        observable_id="left_beater_pose",
        observable_frame_id="frame_left_beater",
        observable_point_abs_m=(-0.024, 0.011, 0.112),
        task_ids=("task.counter_rotate_beaters", "task.speed_up_from_crank"),
        characteristic_length_m=0.3,
        moving_extents_m=(0.2, 0.1, 0.2),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _cycle_trace(
    *,
    crank_end: float = 2.0,
    left_end: float = -6.0,
    right_end: float = 6.0,
    drop_window_field: str | None = None,
) -> dict:
    def sample(phase_id, crank, left, right):
        item = {
            "phase_id": phase_id,
            "hand_crank_angle_rad": crank,
            "left_beater_angle_rad": left,
            "right_beater_angle_rad": right,
        }
        if drop_window_field is not None and phase_id == "crank_forward":
            item.pop(drop_window_field, None)
        return item

    samples = [
        sample("initial", 0.0, 0.0, 0.0),
        sample("crank_forward", crank_end / 2.0, left_end / 2.0, right_end / 2.0),
        sample("crank_forward", crank_end, left_end, right_end),
        sample("coast_after_forward", crank_end, left_end, right_end),
        sample("crank_reverse", 0.0, 0.0, 0.0),
        sample("coast_after_reverse", 0.0, 0.0, 0.0),
    ]
    return {"status": "COMPLETED", "branch": "physical", "samples": samples}


def _compiled_requirement(requirement_id: str) -> dict:
    contract = compile_acceptance_contract(_egg_beater_requirements())
    return next(
        item
        for item in contract["requirements"]
        if item["requirement_id"] == requirement_id
    )


def _evaluate(
    requirement_id: str,
    trace: dict,
    *,
    plan: MechanismPlan | None = None,
) -> dict:
    return evaluate_registered_metric(
        _compiled_requirement(requirement_id),
        {"target": {"observable_id": "left_beater_pose"}},
        trace,
        plan=_plan() if plan is None else plan,
    )


class EggBeaterTaskEntryTests(unittest.TestCase):
    def test_active_egg_beater_tasks_compile_from_their_approved_suite(self) -> None:
        contract = compile_acceptance_contract(_egg_beater_requirements())
        verify_acceptance_contract(contract)

        self.assertEqual(len(contract["requirements"]), 5)
        self.assertTrue(
            all("task_suite_binding" in item for item in contract["requirements"])
        )
        rotation = contract["requirements"][0]
        self.assertAlmostEqual(
            rotation["target"]["value"],
            6.283185307179586 - 0.1,
        )
        self.assertEqual(
            rotation["task_suite_binding"]["declared_target"]["value"],
            6.283185307179586,
        )
        predicate = contract["requirements"][1]
        self.assertEqual(
            predicate["metric_id"],
            "egg_beater_outputs_opposite_directions",
        )
        self.assertNotIn(predicate["metric_id"], METRIC_REGISTRY)
        self.assertEqual(
            predicate["metric_version"],
            EGG_BEATER_TASK_METRIC_VERSION,
        )
        binding = predicate["task_suite_binding"]
        self.assertEqual(binding["sha256"], EGG_BEATER_TASK_SUITE_SHA256)
        self.assertEqual(binding["evaluation_phase_ids"], ["crank_forward"])
        self.assertEqual(
            binding["allowed_input_port_ids"],
            ["hand_crank_torque"],
        )

    def test_egg_beater_threshold_cannot_diverge_from_task_suite(self) -> None:
        value = _egg_beater_requirements()
        value["tasks"][1]["acceptance_requirements"][1]["target"]["value"] = 1.02

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "EGG_BEATER_TASK_REQUIREMENT_MISMATCH",
        )

    def test_egg_beater_suite_fingerprint_change_fails_closed(self) -> None:
        with mock.patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "intake.egg_beater_task_suite.EGG_BEATER_TASK_SUITE_SHA256",
            "0" * 64,
        ):
            with self.assertRaises(AcceptanceContractError) as caught:
                compile_acceptance_contract(_egg_beater_requirements())

        self.assertEqual(caught.exception.code, "EGG_BEATER_TASK_SUITE_CHANGED")

    def test_egg_beater_unknown_evaluator_fails_closed(self) -> None:
        value = _egg_beater_requirements()
        value["tasks"][1]["acceptance_requirements"][0][
            "metric_id"
        ] = "egg_beater_bogus_evaluator"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "EGG_BEATER_TASK_EVALUATOR_MISMATCH",
        )

    def test_off_family_task_cannot_cite_egg_beater_evaluators(self) -> None:
        value = _egg_beater_requirements()
        value["tasks"][1]["task_id"] = "task.spin_beater_pair"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_egg_beater_contract_cannot_drop_its_suite_binding(self) -> None:
        contract = compile_acceptance_contract(_egg_beater_requirements())
        del contract["requirements"][1]["task_suite_binding"]
        _rehash(contract)

        with self.assertRaises(AcceptanceContractError) as caught:
            verify_acceptance_contract(contract)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_egg_beater_binding_cannot_diverge_from_template(self) -> None:
        contract = compile_acceptance_contract(_egg_beater_requirements())
        requirement = copy.deepcopy(contract["requirements"][2])
        requirement["task_suite_binding"]["declared_target"]["value"] = 1.02

        with self.assertRaises(EggBeaterTaskSuiteError) as caught:
            verify_egg_beater_task_suite_binding(requirement)

        self.assertEqual(caught.exception.code, "EGG_BEATER_TASK_BINDING_CHANGED")

    def test_outputs_opposite_directions_pass_and_fail(self) -> None:
        passing = _evaluate(
            "criterion.opposite_directions",
            _cycle_trace(left_end=-6.0, right_end=6.0),
        )
        self.assertTrue(passing["passed"])
        self.assertIs(passing["measured"], True)

        same_sign = _evaluate(
            "criterion.opposite_directions",
            _cycle_trace(left_end=-6.0, right_end=-6.0),
        )
        self.assertFalse(same_sign["passed"])
        self.assertIs(same_sign["measured"], False)

        idle_output = _evaluate(
            "criterion.opposite_directions",
            _cycle_trace(left_end=0.0, right_end=6.0),
        )
        self.assertFalse(idle_output["passed"])

    def test_output_speed_magnitude_ratio_enforces_five_percent(self) -> None:
        passing = _evaluate(
            "criterion.output_speed_magnitude_ratio",
            _cycle_trace(left_end=-6.0, right_end=6.0),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 1.0)
        self.assertEqual(
            passing["evidence"]["task_suite_binding"]["evaluation_phase_ids"],
            ["crank_forward"],
        )

        mismatched = _evaluate(
            "criterion.output_speed_magnitude_ratio",
            _cycle_trace(left_end=-6.36, right_end=6.0),
        )
        self.assertAlmostEqual(mismatched["measured"], 1.06)
        self.assertFalse(mismatched["passed"])

    def test_crank_speed_magnitude_ratio_uses_worst_beater(self) -> None:
        passing = _evaluate(
            "criterion.beater_crank_speed_magnitude_ratio",
            _cycle_trace(crank_end=2.0, left_end=-6.0, right_end=6.0),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 3.0)

        # The left beater is exactly 3:1; the right one exceeds five percent.
        mismatched = _evaluate(
            "criterion.beater_crank_speed_magnitude_ratio",
            _cycle_trace(crank_end=2.0, left_end=-6.0, right_end=6.42),
        )
        self.assertAlmostEqual(mismatched["measured"], 3.21)
        self.assertFalse(mismatched["passed"])

    def test_axis_bound_direction_consistency_pass_and_fail(self) -> None:
        passing = _evaluate(
            "criterion.axis_bound_direction_consistency",
            _cycle_trace(crank_end=2.0, left_end=-6.0, right_end=6.0),
        )
        self.assertTrue(passing["passed"])
        self.assertIs(passing["measured"], True)

        flipped_left = _evaluate(
            "criterion.axis_bound_direction_consistency",
            _cycle_trace(crank_end=2.0, left_end=6.0, right_end=6.0),
        )
        self.assertFalse(flipped_left["passed"])
        self.assertIs(flipped_left["measured"], False)

    def test_direction_signs_derive_from_frozen_axis_bindings(self) -> None:
        # A flipped left-output axis turns same-sign traces into physically
        # opposed rotation, and the declared mesh sign flips with it.  Neither
        # predicate may treat a global negative sign as a family constant.
        plan = _plan(_family_spec(left_axis=[0.0, 0.0, -1.0], left_ratio=3.0))
        trace = _cycle_trace(crank_end=2.0, left_end=6.0, right_end=6.0)

        opposite = _evaluate("criterion.opposite_directions", trace, plan=plan)
        self.assertTrue(opposite["passed"])

        consistent = _evaluate(
            "criterion.axis_bound_direction_consistency",
            trace,
            plan=plan,
        )
        self.assertTrue(consistent["passed"])

    def test_egg_beater_metrics_fail_closed_without_window(self) -> None:
        trace = _cycle_trace()
        for sample in trace["samples"]:
            sample["phase_id"] = "coast_after_forward"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("criterion.opposite_directions", trace)

        self.assertEqual(caught.exception.code, "TASK_SUITE_WINDOW_EMPTY")

    def test_egg_beater_metrics_fail_closed_without_shaft_fields(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "criterion.opposite_directions",
                _cycle_trace(drop_window_field="left_beater_angle_rad"),
            )

        self.assertEqual(caught.exception.code, "EGG_BEATER_SHAFT_TRACE_MISSING")

    def test_egg_beater_metrics_fail_closed_without_family_spec(self) -> None:
        requirement = _compiled_requirement("criterion.opposite_directions")
        with self.assertRaises(MetricEvaluationError) as caught:
            evaluate_registered_metric(
                requirement,
                {"target": {"observable_id": "left_beater_pose"}},
                _cycle_trace(),
                plan=None,
            )

        self.assertEqual(caught.exception.code, "EGG_BEATER_FAMILY_SPEC_MISSING")

    def test_egg_beater_ratio_fails_closed_without_motion(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "criterion.output_speed_magnitude_ratio",
                _cycle_trace(left_end=-6.0, right_end=0.0),
            )
        self.assertEqual(caught.exception.code, "EGG_BEATER_OUTPUT_MOTION_MISSING")

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "criterion.beater_crank_speed_magnitude_ratio",
                _cycle_trace(crank_end=0.0, left_end=0.0, right_end=0.0),
            )
        self.assertEqual(caught.exception.code, "EGG_BEATER_CRANK_MOTION_MISSING")

    def test_active_egg_beater_program_has_exact_strict_policy(self) -> None:
        load_case = LoadCase(
            load_case_id="task_egg_beater_cycle",
            family="compound",
            program="task_egg_beater_cycle",
            purpose="nominal",
            moving_mass_kg=0.34,
            step_s=0.00005,
            sample_interval_s=0.01,
            phases=tuple(
                LoadPhase(name, "none", None, 0.0, 0.1)
                for name in (
                    "crank_forward",
                    "coast_after_forward",
                    "crank_reverse",
                    "coast_after_reverse",
                )
            ),
        )

        policy = task_program_spec_for_load_case(
            load_case,
            plugin_id="multi_output_shaft_coupling_v1",
        )

        self.assertEqual(policy.powered_phase_ids, ("crank_forward",))
        self.assertEqual(
            policy.return_phase_ids,
            ("coast_after_forward", "crank_reverse", "coast_after_reverse"),
        )


if __name__ == "__main__":
    unittest.main()
