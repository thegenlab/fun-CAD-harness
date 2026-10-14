from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.intake.acceptance_contract import (
    METRIC_REGISTRY,
    METRIC_REGISTRY_VERSION,
    AcceptanceContractError,
    compile_acceptance_contract,
    verify_acceptance_contract,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.utility_knife_task_suite import (
    UTILITY_KNIFE_TASK_METRIC_VERSION,
    UTILITY_KNIFE_TASK_SUITE_SHA256,
    UtilityKnifeTaskSuiteError,
    verify_utility_knife_task_suite_binding,
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
from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_loads import (
    default_detent_slider_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    task_program_spec_for_load_case,
)


def _utility_knife_requirements() -> dict:
    clauses = (
        "the blade carriage finishes at the 45 mm working detent within 0.0015 m; "
        "the test applies the approved 30 N reverse axial load; "
        "the complete 30 N load is held for at least 0.3 s; "
        "blade slip during the loaded hold stays within 0.0015 m; "
        "after the hold the blade remains at the 45 mm working detent within 0.0015 m; "
        "the blade carriage finishes at the 25 mm second detent within 0.0015 m; "
        "the blade returns to the home position within 0.0015 m"
    )
    common = {
        "input_role": "thumb slider",
        "target_role": "blade carriage",
        "required_branches": ["physical", "ideal_reference"],
        "allow_direct_output_actuation": False,
    }
    position = {
        **common,
        "observable_quantity": "pose_6d",
        "metric_id": "signed_displacement_m",
        "direction": "positive",
    }
    return {
        "schema_version": "0.2",
        "requirements_id": "utility_knife_task_entry_test_v1",
        "prompt": clauses,
        "tasks": [
            {
                "task_id": "task.extend_to_working_position",
                "description": "Latch the blade carriage at the 45 mm working detent.",
                "acceptance_requirements": [
                    {
                        **position,
                        "requirement_id": "criterion.working_position",
                        "prompt_clause": "the blade carriage finishes at the 45 mm working detent within 0.0015 m",
                        "test_program": "task_detent_extend",
                        "test_load_case_id": "task_detent_extend",
                        "target": {
                            "comparator": "within",
                            "value": 0.045,
                            "unit": "m",
                            "tolerance": 0.0015,
                        },
                    },
                ],
            },
            {
                "task_id": "task.hold_working_position_under_load",
                "description": "Hold the working detent under the approved reverse load.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "criterion.reverse_load_magnitude",
                        "prompt_clause": "the test applies the approved 30 N reverse axial load",
                        "observable_quantity": "actuation_log",
                        "metric_id": "utility_knife_reverse_load_magnitude",
                        "test_program": "task_detent_hold_under_load",
                        "test_load_case_id": "task_detent_hold_under_load",
                        "target": {
                            "comparator": "equals",
                            "value": 30.0,
                            "unit": "N",
                        },
                    },
                    {
                        **common,
                        "requirement_id": "criterion.loaded_hold_duration",
                        "prompt_clause": "the complete 30 N load is held for at least 0.3 s",
                        "observable_quantity": "actuation_log",
                        "metric_id": "utility_knife_loaded_hold_duration",
                        "test_program": "task_detent_hold_under_load",
                        "test_load_case_id": "task_detent_hold_under_load",
                        "target": {
                            "comparator": "at_least",
                            "value": 0.3,
                            "unit": "s",
                        },
                    },
                    {
                        **common,
                        "requirement_id": "criterion.maximum_loaded_slip",
                        "prompt_clause": "blade slip during the loaded hold stays within 0.0015 m",
                        "observable_quantity": "pose_6d",
                        "metric_id": "utility_knife_maximum_loaded_slip",
                        "test_program": "task_detent_hold_under_load",
                        "test_load_case_id": "task_detent_hold_under_load",
                        "target": {
                            "comparator": "at_most",
                            "value": 0.0015,
                            "unit": "m",
                        },
                    },
                    {
                        **position,
                        "requirement_id": "criterion.loaded_final_position",
                        "prompt_clause": "after the hold the blade remains at the 45 mm working detent within 0.0015 m",
                        "test_program": "task_detent_hold_under_load",
                        "test_load_case_id": "task_detent_hold_under_load",
                        "target": {
                            "comparator": "within",
                            "value": 0.045,
                            "unit": "m",
                            "tolerance": 0.0015,
                        },
                    },
                ],
            },
            {
                "task_id": "task.select_second_position",
                "description": "Latch the blade carriage at the 25 mm second detent.",
                "acceptance_requirements": [
                    {
                        **position,
                        "requirement_id": "criterion.second_position",
                        "prompt_clause": "the blade carriage finishes at the 25 mm second detent within 0.0015 m",
                        "test_program": "task_detent_second_position",
                        "test_load_case_id": "task_detent_second_position",
                        "target": {
                            "comparator": "within",
                            "value": 0.025,
                            "unit": "m",
                            "tolerance": 0.0015,
                        },
                    },
                ],
            },
            {
                "task_id": "task.retract_blade",
                "description": "Return the blade carriage to the home position.",
                "acceptance_requirements": [
                    {
                        **position,
                        "requirement_id": "criterion.retracted_home_position",
                        "prompt_clause": "the blade returns to the home position within 0.0015 m",
                        "test_program": "task_detent_cycle",
                        "test_load_case_id": "task_detent_cycle",
                        "target": {
                            "comparator": "within",
                            "value": 0.0,
                            "unit": "m",
                            "tolerance": 0.0015,
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


def _family_spec() -> dict:
    wall_x = (-0.016, -0.002, 0.009, 0.023, 0.029, 0.043, 0.049, 0.063)
    return {
        "mechanism_kind": "prismatic_detent_latch",
        "mechanism_id": "auto_lock_utility_knife_slider",
        "bodies": [
            {
                "body_id": "handle",
                "group_ids": ["group_knife_handle"],
                "fixed": True,
                "mass_kg": 0.16,
                "center_abs_m": [-0.018, 0.0, 0.0],
                "size_m": [0.192, 0.034, 0.044],
            },
            {
                "body_id": "blade_carriage",
                "group_ids": ["group_blade_carriage"],
                "fixed": False,
                "mass_kg": 0.045,
                "center_abs_m": [0.028, 0.0, 0.0],
                "size_m": [0.156, 0.02, 0.026],
            },
            {
                "body_id": "thumb_latch",
                "group_ids": ["group_thumb_latch"],
                "fixed": False,
                "mass_kg": 0.012,
                "center_abs_m": [-0.027, 0.0, 0.019],
                "size_m": [0.052, 0.024, 0.034],
            },
        ],
        "blade_prismatic_joint": {
            "joint_id": "blade_slide",
            "parent_body_id": "handle",
            "child_body_id": "blade_carriage",
            "origin_abs_m": [-0.03, 0.0, 0.0],
            "axis_abs": [1.0, 0.0, 0.0],
            "min_position_m": 0.0,
            "max_position_m": 0.065,
        },
        "latch_press_joint": {
            "joint_id": "thumb_press",
            "parent_body_id": "blade_carriage",
            "child_body_id": "thumb_latch",
            "origin_abs_m": [-0.034, 0.0, 0.017],
            "axis_abs": [0.0, 0.0, -1.0],
            "min_position_m": -0.00015,
            "travel_m": 0.004,
            "spring_stiffness_n_m": 900.0,
            "damping_n_s_m": 8.0,
            "spring_preload_n": 1.0,
        },
        "detent_interface": {
            "rack_body_id": "handle",
            "tab_body_id": "thumb_latch",
            "positions_m": [0.0, 0.025, 0.045, 0.065],
            "capture_tolerance_m": 0.0015,
            "release_press_m": 0.003,
            "holding_force_n": 45.0,
            "geometry_contract": {
                "rack_component_refs": [
                    f"handle/solid_{index}" for index in range(11, 19)
                ],
                "seat_component_refs": [
                    f"handle/solid_{index}" for index in range(19, 23)
                ],
                "tab_component_ref": "thumb_latch/solid_35",
                "rack_tooth_centers_abs_m": [
                    [x, 0.0081, 0.0155] for x in wall_x
                ],
                "rack_tooth_size_m": [0.007, 0.0022, 0.007],
                "seat_centers_abs_m": [
                    [x, 0.00815, 0.0156]
                    for x in (-0.009, 0.016, 0.036, 0.056)
                ],
                "seat_size_m": [0.0066, 0.0023, 0.001],
                "tab_home_center_abs_m": [-0.009, 0.008, 0.0125],
                "tab_size_m": [0.006, 0.002, 0.005],
                "pressed_offset_abs_m": [0.0, 0.0, -0.004],
                "contact_tolerance_m": 0.00055,
                "minimum_pressed_clearance_m": 0.001,
            },
        },
        "input_body_id": "thumb_latch",
        "output_body_id": "blade_carriage",
        "nominal_press_force_n": 8.0,
        "nominal_slide_force_n": 15.0,
        "adversarial_axial_force_n": 30.0,
        "max_support_drift_m": 0.001,
        "max_off_axis_rotation_rad": 0.04,
    }


def _mirrored_family_spec() -> dict:
    """The approved spec mirrored across the slide axis (axis flips sign)."""

    spec = copy.deepcopy(_family_spec())
    spec["blade_prismatic_joint"]["axis_abs"] = [-1.0, 0.0, 0.0]
    for body in spec["bodies"]:
        body["center_abs_m"][0] = -body["center_abs_m"][0]
    for joint in ("blade_prismatic_joint", "latch_press_joint"):
        spec[joint]["origin_abs_m"][0] = -spec[joint]["origin_abs_m"][0]
    geometry = spec["detent_interface"]["geometry_contract"]
    for key in ("rack_tooth_centers_abs_m", "seat_centers_abs_m"):
        for center in geometry[key]:
            center[0] = -center[0]
    geometry["tab_home_center_abs_m"][0] = -geometry["tab_home_center_abs_m"][0]
    return spec


def _plan(family_spec: dict | None = None) -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=family_spec if family_spec is not None else _family_spec(),
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="auto_lock_utility_knife_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="utility_knife_detent_slider",
        capsule_file="utility_knife_detent_slider.json",
        family="compound",
        protocol_id="prismatic_detent_latch",
        fixed_group_ids=("group_knife_handle",),
        moving_group_ids=("group_blade_carriage", "group_thumb_latch"),
        axis_abs=(1.0, 0.0, 0.0),
        joint_origin_abs_m=(-0.03, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_blade_carriage",
        input_port_id="thumb_press_and_slide",
        input_frame_id="frame_thumb_slider",
        input_point_abs_m=(-0.034, 0.0, 0.017),
        observable_id="blade_carriage_pose",
        observable_frame_id="frame_blade_carriage",
        observable_point_abs_m=(0.028, 0.0, 0.0),
        task_ids=(
            "task.extend_to_working_position",
            "task.hold_working_position_under_load",
            "task.select_second_position",
            "task.retract_blade",
        ),
        characteristic_length_m=0.2,
        moving_extents_m=(0.2, 0.04, 0.05),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _hold_trace(
    *,
    load_n: float = 30.0,
    hold_s: float = 0.3,
    slip_m: float = 0.0005,
    drop_window_field: str | None = None,
) -> dict:
    """Synthetic hold trace: a non-window baseline, then the 30 N hold window."""

    def sample(time_s, phase_id, blade_m, load):
        item = {"time_s": time_s, "phase_id": phase_id}
        if drop_window_field is None or phase_id != "axial_load_hold":
            item["blade_carriage_position_m"] = blade_m
            item["external_load_abs_n"] = list(load)
        elif drop_window_field != "blade_carriage_position_m":
            item["blade_carriage_position_m"] = blade_m
        elif drop_window_field != "external_load_abs_n":
            item["external_load_abs_n"] = list(load)
        return item

    count = int(round(hold_s / 0.01))
    samples = [
        sample(0.0, "initial", 0.0, (0.0, 0.0, 0.0)),
        sample(1.02, "release_into_detent", 0.0454, (0.0, 0.0, 0.0)),
    ]
    for index in range(1, count + 1):
        samples.append(
            sample(
                1.02 + index * 0.01,
                "axial_load_hold",
                0.0454 + slip_m * index / count,
                (-load_n, 0.0, 0.0),
            )
        )
    return {"status": "COMPLETED", "branch": "physical", "samples": samples}


def _compiled_requirement(requirement_id: str) -> dict:
    contract = compile_acceptance_contract(_utility_knife_requirements())
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
        {"target": {"observable_id": "blade_carriage_pose"}},
        trace,
        plan=_plan() if plan is None else plan,
    )


class UtilityKnifeTaskEntryTests(unittest.TestCase):
    def test_active_utility_knife_tasks_compile_from_their_approved_suite(self) -> None:
        contract = compile_acceptance_contract(_utility_knife_requirements())
        verify_acceptance_contract(contract)

        self.assertEqual(len(contract["requirements"]), 7)
        self.assertTrue(
            all("task_suite_binding" in item for item in contract["requirements"])
        )
        predicate = contract["requirements"][1]
        self.assertEqual(
            predicate["metric_id"],
            "utility_knife_reverse_load_magnitude",
        )
        self.assertNotIn(predicate["metric_id"], METRIC_REGISTRY)
        self.assertEqual(
            predicate["metric_version"],
            UTILITY_KNIFE_TASK_METRIC_VERSION,
        )
        # A suite-declared "none" tolerance compiles to an exact zero
        # tolerance for the numeric equality target.
        self.assertEqual(
            predicate["target"],
            {"comparator": "equals", "value": 30.0, "unit": "N", "tolerance": 0.0},
        )
        binding = predicate["task_suite_binding"]
        self.assertEqual(binding["sha256"], UTILITY_KNIFE_TASK_SUITE_SHA256)
        self.assertEqual(binding["evaluation_phase_ids"], ["axial_load_hold"])
        self.assertEqual(
            binding["allowed_input_port_ids"],
            ["thumb_press_and_slide"],
        )
        self.assertEqual(
            binding["declared_target"],
            {
                "comparator": "equals",
                "value": 30.0,
                "unit": "N",
                "tolerance_mode": "none",
            },
        )
        final_position = contract["requirements"][4]
        self.assertEqual(final_position["metric_id"], "signed_displacement_m")
        self.assertEqual(
            final_position["metric_version"],
            METRIC_REGISTRY_VERSION,
        )
        self.assertEqual(
            final_position["task_suite_binding"]["evaluation_phase_ids"],
            [
                "press_and_slide_to_detent",
                "release_into_detent",
                "axial_load_hold",
            ],
        )
        retract = contract["requirements"][6]
        self.assertEqual(
            retract["task_suite_binding"]["evaluation_phase_ids"],
            ["release_home_detent"],
        )

    def test_utility_knife_threshold_cannot_diverge_from_task_suite(self) -> None:
        value = _utility_knife_requirements()
        value["tasks"][1]["acceptance_requirements"][3]["target"]["value"] = 0.046

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "UTILITY_KNIFE_TASK_REQUIREMENT_MISMATCH",
        )

    def test_utility_knife_suite_fingerprint_change_fails_closed(self) -> None:
        with mock.patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "intake.utility_knife_task_suite.UTILITY_KNIFE_TASK_SUITE_SHA256",
            "0" * 64,
        ):
            with self.assertRaises(AcceptanceContractError) as caught:
                compile_acceptance_contract(_utility_knife_requirements())

        self.assertEqual(caught.exception.code, "UTILITY_KNIFE_TASK_SUITE_CHANGED")

    def test_utility_knife_unknown_evaluator_fails_closed(self) -> None:
        value = _utility_knife_requirements()
        value["tasks"][1]["acceptance_requirements"][0][
            "metric_id"
        ] = "utility_knife_bogus_evaluator"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "UTILITY_KNIFE_TASK_EVALUATOR_MISMATCH",
        )

    def test_off_family_task_cannot_cite_utility_knife_evaluators(self) -> None:
        value = _utility_knife_requirements()
        value["tasks"][1]["task_id"] = "task.shake_blade_detent"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_utility_knife_contract_cannot_drop_its_suite_binding(self) -> None:
        contract = compile_acceptance_contract(_utility_knife_requirements())
        del contract["requirements"][1]["task_suite_binding"]
        _rehash(contract)

        with self.assertRaises(AcceptanceContractError) as caught:
            verify_acceptance_contract(contract)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_utility_knife_binding_cannot_diverge_from_template(self) -> None:
        contract = compile_acceptance_contract(_utility_knife_requirements())
        requirement = copy.deepcopy(contract["requirements"][1])
        requirement["task_suite_binding"]["declared_target"]["value"] = 29.0

        with self.assertRaises(UtilityKnifeTaskSuiteError) as caught:
            verify_utility_knife_task_suite_binding(requirement)

        self.assertEqual(
            caught.exception.code,
            "UTILITY_KNIFE_TASK_BINDING_CHANGED",
        )

    def test_reverse_load_magnitude_pass_and_fail(self) -> None:
        passing = _evaluate(
            "criterion.reverse_load_magnitude",
            _hold_trace(load_n=30.0),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 30.0)
        self.assertEqual(
            passing["evidence"]["task_suite_binding"]["evaluation_phase_ids"],
            ["axial_load_hold"],
        )

        underpowered = _evaluate(
            "criterion.reverse_load_magnitude",
            _hold_trace(load_n=25.0),
        )
        self.assertFalse(underpowered["passed"])
        self.assertAlmostEqual(underpowered["measured"], 25.0)

    def test_reverse_load_direction_derives_from_frozen_family_spec(self) -> None:
        # Mirroring the approved spec across the slide axis flips the approved
        # reverse-load direction; the predicate must not treat a global
        # negative sign as a family constant.
        mirrored = _plan(_mirrored_family_spec())
        flipped = _evaluate(
            "criterion.reverse_load_magnitude",
            _hold_trace(load_n=-30.0),
            plan=mirrored,
        )
        self.assertTrue(flipped["passed"])
        self.assertAlmostEqual(flipped["measured"], 30.0)

        forward = _evaluate(
            "criterion.reverse_load_magnitude",
            _hold_trace(load_n=-30.0),
        )
        self.assertFalse(forward["passed"])
        self.assertAlmostEqual(forward["measured"], -30.0)

    def test_loaded_hold_duration_pass_fail_and_integration_dust(self) -> None:
        passing = _evaluate(
            "criterion.loaded_hold_duration",
            _hold_trace(hold_s=0.3),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.3)

        short = _evaluate(
            "criterion.loaded_hold_duration",
            _hold_trace(hold_s=0.2),
        )
        self.assertFalse(short["passed"])
        self.assertAlmostEqual(short["measured"], 0.2)

        # Integration time bases accumulate machine-scale dust below the
        # validator's 1e-9 s phase-boundary tolerance; a certified 0.3 s hold
        # must still report its full duration.
        dusty = _hold_trace(hold_s=0.3)
        dusty["samples"][-1]["time_s"] = 1.02 + 0.29999999999996696
        dust_passing = _evaluate("criterion.loaded_hold_duration", dusty)
        self.assertTrue(dust_passing["passed"])
        self.assertAlmostEqual(dust_passing["measured"], 0.3)

    def test_maximum_loaded_slip_pass_and_fail(self) -> None:
        passing = _evaluate(
            "criterion.maximum_loaded_slip",
            _hold_trace(slip_m=0.0005),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.0005)

        slipped = _evaluate(
            "criterion.maximum_loaded_slip",
            _hold_trace(slip_m=0.002),
        )
        self.assertFalse(slipped["passed"])
        self.assertAlmostEqual(slipped["measured"], 0.002)

    def test_utility_knife_metrics_fail_closed_without_window(self) -> None:
        trace = _hold_trace()
        for sample in trace["samples"]:
            sample["phase_id"] = "release_into_detent"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("criterion.maximum_loaded_slip", trace)

        self.assertEqual(caught.exception.code, "TASK_SUITE_WINDOW_EMPTY")

    def test_utility_knife_metrics_fail_closed_without_baseline(self) -> None:
        trace = _hold_trace()
        trace["samples"] = trace["samples"][2:]

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("criterion.maximum_loaded_slip", trace)

        self.assertEqual(caught.exception.code, "TASK_SUITE_BASELINE_MISSING")

    def test_utility_knife_metrics_fail_closed_without_load_field(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "criterion.reverse_load_magnitude",
                _hold_trace(drop_window_field="external_load_abs_n"),
            )

        self.assertEqual(caught.exception.code, "UTILITY_KNIFE_LOAD_TRACE_MISSING")

    def test_utility_knife_metrics_fail_closed_without_blade_field(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "criterion.maximum_loaded_slip",
                _hold_trace(drop_window_field="blade_carriage_position_m"),
            )

        self.assertEqual(caught.exception.code, "UTILITY_KNIFE_BLADE_TRACE_MISSING")

    def test_utility_knife_metrics_fail_closed_without_family_spec(self) -> None:
        requirement = _compiled_requirement("criterion.maximum_loaded_slip")
        with self.assertRaises(MetricEvaluationError) as caught:
            evaluate_registered_metric(
                requirement,
                {"target": {"observable_id": "blade_carriage_pose"}},
                _hold_trace(),
                plan=None,
            )

        self.assertEqual(caught.exception.code, "UTILITY_KNIFE_FAMILY_SPEC_MISSING")

    def test_utility_knife_metrics_fail_closed_on_invalid_family_spec(self) -> None:
        invalid = _family_spec()
        invalid["mechanism_kind"] = "not_a_detent_latch"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "criterion.maximum_loaded_slip",
                _hold_trace(),
                plan=_plan(invalid),
            )

        self.assertEqual(caught.exception.code, "UTILITY_KNIFE_FAMILY_SPEC_INVALID")

    def test_active_utility_knife_programs_have_exact_strict_policies(self) -> None:
        loads = {
            case.program: case
            for case in default_detent_slider_load_cases(_plan())
            if case.program.startswith("task_detent")
        }
        self.assertEqual(
            set(loads),
            {
                "task_detent_extend",
                "task_detent_hold_under_load",
                "task_detent_second_position",
                "task_detent_cycle",
            },
        )

        extend = task_program_spec_for_load_case(
            loads["task_detent_extend"],
            plugin_id="prismatic_detent_latch_v1",
        )
        self.assertEqual(
            extend.powered_phase_ids,
            (
                "prepress_latch_ramp_25",
                "prepress_latch_ramp_50",
                "prepress_latch_ramp_75",
                "prepress_latch_ramp_100",
                "press_and_slide_to_detent",
            ),
        )
        self.assertEqual(extend.return_phase_ids, ("release_into_detent",))
        self.assertEqual(extend.return_input_policy, "zero")
        self.assertEqual(extend.adaptive_phase_ids, ("press_and_slide_to_detent",))
        self.assertEqual(
            extend.executed_phase_extension_keys,
            ("actuation_channels", "termination_reason"),
        )

        hold = task_program_spec_for_load_case(
            loads["task_detent_hold_under_load"],
            plugin_id="prismatic_detent_latch_v1",
        )
        self.assertEqual(
            hold.return_phase_ids,
            ("release_into_detent", "axial_load_hold"),
        )
        self.assertEqual(hold.return_input_policy, "zero")

        second = task_program_spec_for_load_case(
            loads["task_detent_second_position"],
            plugin_id="prismatic_detent_latch_v1",
        )
        self.assertEqual(second.powered_phase_ids, extend.powered_phase_ids)
        self.assertEqual(second.return_phase_ids, ("release_into_detent",))

        cycle = task_program_spec_for_load_case(
            loads["task_detent_cycle"],
            plugin_id="prismatic_detent_latch_v1",
        )
        self.assertEqual(
            cycle.powered_phase_ids,
            (
                "prepress_latch_for_extension_ramp_25",
                "prepress_latch_for_extension_ramp_50",
                "prepress_latch_for_extension_ramp_75",
                "prepress_latch_for_extension_ramp_100",
                "press_and_slide_to_outer_detent",
            ),
        )
        self.assertEqual(
            cycle.return_phase_ids,
            (
                "release_outer_detent",
                "align_outer_detent_for_unlock",
                "prepress_latch_for_retraction_ramp_25",
                "prepress_latch_for_retraction_ramp_50",
                "prepress_latch_for_retraction_ramp_75",
                "prepress_latch_for_retraction_ramp_100",
                "press_and_retract_home",
                "release_home_detent",
            ),
        )
        self.assertEqual(cycle.return_input_policy, "selected_input_only")
        self.assertEqual(
            cycle.adaptive_phase_ids,
            (
                "press_and_slide_to_outer_detent",
                "align_outer_detent_for_unlock",
                "press_and_retract_home",
            ),
        )

    def test_detent_program_policy_fails_closed_on_unknown_layout(self) -> None:
        load_case = LoadCase(
            load_case_id="task_detent_extend",
            family="compound",
            program="task_detent_extend",
            purpose="nominal",
            moving_mass_kg=0.057,
            step_s=0.0001,
            sample_interval_s=0.01,
            phases=(
                LoadPhase(
                    "press_and_slide_to_detent",
                    "force",
                    (1.0, 0.0, 0.0),
                    8.0,
                    0.8,
                ),
            ),
        )

        with self.assertRaises(MechanismValidationError) as caught:
            task_program_spec_for_load_case(
                load_case,
                plugin_id="prismatic_detent_latch_v1",
            )

        self.assertEqual(caught.exception.code, "TASK_PROGRAM_POLICY_UNREGISTERED")


if __name__ == "__main__":
    unittest.main()
