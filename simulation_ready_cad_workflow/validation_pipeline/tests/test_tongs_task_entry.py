from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.contracts.requirement_obligations import (
    build_requirement_obligation_inventory,
    requirement_check_catalog,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.acceptance_contract import (
    METRIC_REGISTRY,
    METRIC_REGISTRY_VERSION,
    AcceptanceContractError,
    compile_acceptance_contract,
    verify_acceptance_contract,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.tongs_task_suite import (
    TONGS_TASK_METRIC_VERSION,
    TONGS_TASK_SUITE_SHA256,
    TongsTaskSuiteError,
    verify_tongs_task_suite_binding,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.acceptance_metrics import (
    MetricEvaluationError,
    evaluate_registered_metric,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import (
    default_revolute_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    MOTION_WINDOW_METRICS,
    task_program_spec_for_load_case,
)


def _tongs_requirements() -> dict:
    clauses = (
        "the standard block is settled on the table before pickup begins; "
        "the block has effective contact with both jaws in the verified pickup state; "
        "the block rises by at least 0.02 m from its settled tabletop position; "
        "the block no longer contacts the table at pickup verification; "
        "the block moves only through jaw contact and is never directly actuated; "
        "the verified hold window lasts at least 0.35 s; "
        "the block slips no more than 0.003 m relative to the tongs during the hold; "
        "the hand input wrench remains zero throughout the passive-release window; "
        "both jaw contacts are clear at release verification; "
        "the block returns to tabletop contact at release verification; "
        "the jaw joint returns to 0 rad within three degrees"
    )
    pickup = {
        "input_role": "upper tong arm",
        "required_branches": ["physical", "ideal_reference"],
        "allow_direct_output_actuation": True,
        "test_program": "tong_tabletop_pickup",
        "test_load_case_id": "tong_tabletop_pickup",
    }
    hold = {
        "input_role": "upper tong arm",
        "required_branches": ["physical"],
        "allow_direct_output_actuation": True,
        "test_program": "tong_tabletop_hold",
        "test_load_case_id": "tong_tabletop_hold",
    }
    release = {
        "input_role": "upper tong arm",
        "required_branches": ["physical", "ideal_reference"],
        "allow_direct_output_actuation": True,
        "test_program": "tong_tabletop_release",
        "test_load_case_id": "tong_tabletop_release",
    }
    return {
        "schema_version": "0.2",
        "requirements_id": "tongs_task_entry_test_v1",
        "prompt": clauses,
        "tasks": [
            {
                "task_id": "tongs.pickup_standard_block",
                "description": "Pick up the standard block from the table.",
                "acceptance_requirements": [
                    {
                        **pickup,
                        "requirement_id": "tongs.block_starts_on_table",
                        "prompt_clause": "the standard block is settled on the table before pickup begins",
                        "target_role": "jaw contact",
                        "observable_quantity": "contact_event",
                        "metric_id": "tong_object_starts_on_table",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **pickup,
                        "requirement_id": "tongs.bilateral_grip",
                        "prompt_clause": "the block has effective contact with both jaws in the verified pickup state",
                        "target_role": "jaw contact",
                        "observable_quantity": "contact_event",
                        "metric_id": "tong_two_sided_grip",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **pickup,
                        "requirement_id": "tongs.minimum_pickup_lift",
                        "prompt_clause": "the block rises by at least 0.02 m from its settled tabletop position",
                        "target_role": "jaw contact",
                        "observable_quantity": "pose_6d",
                        "metric_id": "tong_pickup_lift_m",
                        "target": {
                            "comparator": "at_least",
                            "value": 0.02,
                            "unit": "m",
                        },
                    },
                    {
                        **pickup,
                        "requirement_id": "tongs.block_clears_table",
                        "prompt_clause": "the block no longer contacts the table at pickup verification",
                        "target_role": "jaw contact",
                        "observable_quantity": "contact_event",
                        "metric_id": "tong_pickup_table_clear",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **pickup,
                        "requirement_id": "tongs.block_not_directly_actuated",
                        "prompt_clause": "the block moves only through jaw contact and is never directly actuated",
                        "target_role": "jaw contact",
                        "observable_quantity": "actuation_log",
                        "metric_id": "actuation_provenance",
                        "target": {
                            "comparator": "equals",
                            "value": "indirect_only",
                            "unit": "enum",
                        },
                    },
                ],
            },
            {
                "task_id": "tongs.hold_standard_block",
                "description": "Hold the standard block above the table.",
                "acceptance_requirements": [
                    {
                        **hold,
                        "requirement_id": "tongs.minimum_hold_duration",
                        "prompt_clause": "the verified hold window lasts at least 0.35 s",
                        "target_role": "upper tong arm",
                        "observable_quantity": "contact_event",
                        "metric_id": "tongs.hold_duration_s",
                        "target": {
                            "comparator": "at_least",
                            "value": 0.35,
                            "unit": "s",
                        },
                    },
                    {
                        **hold,
                        "requirement_id": "tongs.maximum_relative_slip",
                        "prompt_clause": "the block slips no more than 0.003 m relative to the tongs during the hold",
                        "target_role": "jaw contact",
                        "observable_quantity": "pose_6d",
                        "metric_id": "tong_hold_relative_slip_m",
                        "target": {
                            "comparator": "at_most",
                            "value": 0.003,
                            "unit": "m",
                        },
                    },
                ],
            },
            {
                "task_id": "tongs.release_standard_block",
                "description": "Release the standard block to the table.",
                "acceptance_requirements": [
                    {
                        **release,
                        "requirement_id": "tongs.hand_input_zero_during_release",
                        "prompt_clause": "the hand input wrench remains zero throughout the passive-release window",
                        "target_role": "upper tong arm",
                        "observable_quantity": "actuation_log",
                        "metric_id": "tongs.release_input_zero",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **release,
                        "requirement_id": "tongs.both_jaw_contacts_clear",
                        "prompt_clause": "both jaw contacts are clear at release verification",
                        "target_role": "jaw contact",
                        "observable_quantity": "contact_event",
                        "metric_id": "tong_release_contact_clear",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **release,
                        "requirement_id": "tongs.block_returns_to_table",
                        "prompt_clause": "the block returns to tabletop contact at release verification",
                        "target_role": "jaw contact",
                        "observable_quantity": "contact_event",
                        "metric_id": "tong_release_returns_to_table",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **release,
                        "requirement_id": "tongs.joint_returns_to_zero",
                        "prompt_clause": "the jaw joint returns to 0 rad within three degrees",
                        "target_role": "upper tong arm",
                        "observable_quantity": "pose_6d",
                        "metric_id": "return_to_start_rad",
                        "target": {
                            "comparator": "within",
                            "value": 0.0,
                            "unit": "rad",
                            "tolerance": 0.05235987755982989,
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
    return {
        "schema_version": "0.1",
        "mechanism_kind": "axis_spring",
        "joint_family": "revolute",
        "rest_position_si": 0.0,
        "stiffness_si": 0.8,
        "damping_si": 0.035,
        "maximum_deflection_si": 0.30194196059501904,
        "maximum_stop_penetration_si": 0.002617993877991494,
        "bearing_off_axis_stiffness_si": 14.0,
        "bearing_off_axis_damping_si": 0.25,
        "moving_mass_kg": 0.12,
        "contact_fixture": {
            "shape": "tabletop_pickup_box",
            "center_abs_m": [0.158, -0.000965, 0.0],
            "size_m": [0.012, 0.0067, 0.01],
            "friction": 1.5,
            "hold_duration_s": 0.35,
            "mass_kg": 0.01,
            "grip_torque_nm": 1.1,
            "table_center_abs_m": [0.158, 0.0, -0.007],
            "table_size_m": [0.12, 0.08, 0.004],
            "approach_offset_abs_m": [-0.05, 0.0, 0.003],
            "grip_offset_abs_m": [0.0, 0.0, 0.003],
            "lift_offset_abs_m": [0.0, 0.0, 0.033],
            "lift_duration_s": 0.8,
        },
    }


def _plan(family_spec: dict | None = None) -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_PLAIN_REVOLUTE_ADAPTER_ID),
        family_spec=family_spec if family_spec is not None else _family_spec(),
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="spring_kitchen_tongs_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="spring_tong_hinge",
        capsule_file="spring_tong_hinge.json",
        family="revolute",
        protocol_id="revolute_pivot_hinge",
        fixed_group_ids=("group_lower_tong",),
        moving_group_ids=("group_upper_tong",),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_upper_jaw",
        input_port_id="hand_squeeze_torque",
        input_frame_id="frame_squeeze_input",
        input_point_abs_m=(-0.05, 0.0, 0.0),
        observable_id="upper_jaw_pose",
        observable_frame_id="frame_upper_jaw",
        observable_point_abs_m=(0.1, 0.0, 0.0),
        task_ids=(
            "tongs.pickup_standard_block",
            "tongs.hold_standard_block",
            "tongs.release_standard_block",
        ),
        characteristic_length_m=0.2,
        moving_extents_m=(0.18, 0.06, 0.04),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _load_plan_namespace() -> SimpleNamespace:
    return SimpleNamespace(
        family="revolute",
        adapter_id=None,
        adapter_spec={
            "adapter_id": "derived_plain_revolute_v1",
            "family": "revolute",
            "family_spec": _family_spec(),
            "whole_machine_angular_task": {
                "outbound_task_id": "pickup_standard_test_block_from_table",
                "return_task_id": "release_standard_test_block_to_table",
                "outbound_angle_rad": 0.29670597283903605,
                "home_angle_rad": 0.0,
                "moving_mass_kg": 0.12,
                "outbound_torque_nm": 1.1,
                "return_torque_nm": 0.0,
                "max_torque_phase_s": 3.0,
                "settle_phase_s": 0.05,
            },
        },
        axis_abs=(0.0, 0.0, 1.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        moving_extents_m=(0.18, 0.06, 0.04),
    )


def _hold_trace(*, hold_s: float = 0.35, last_time_s: float | None = None) -> dict:
    """Synthetic hold trace: a non-window baseline, then the hold window."""

    samples = [
        {"time_s": 0.0, "phase_id": "initial"},
        {"time_s": 1.0, "phase_id": "tong_hold_lift"},
    ]
    count = int(round(hold_s / 0.05))
    for index in range(1, count + 1):
        samples.append(
            {
                "time_s": 1.0 + index * 0.05,
                "phase_id": "tong_hold_verify",
            }
        )
    if last_time_s is not None:
        samples[-1]["time_s"] = last_time_s
    return {"status": "COMPLETED", "branch": "physical", "samples": samples}


def _release_trace(
    *,
    window_torque_nm: float = 0.0,
    drop_window_field: bool = False,
) -> dict:
    """Synthetic release trace: a ramp baseline, then the zero-input window."""

    def sample(time_s, phase_id, torque_z):
        item = {"time_s": time_s, "phase_id": phase_id}
        item["applied_torque_abs_nm"] = [0.0, 0.0, torque_z]
        return item

    samples = [
        sample(0.0, "initial", 0.0),
        sample(7.7, "tong_release_gradual_open", 1.1),
        sample(7.75, "tong_release_passive_return", window_torque_nm),
        sample(8.15, "tong_release_final_table_settle", window_torque_nm),
        sample(8.55, "tong_release_observe", window_torque_nm),
    ]
    if drop_window_field:
        del samples[2]["applied_torque_abs_nm"]
    return {"status": "COMPLETED", "branch": "physical", "samples": samples}


def _compiled_requirement(requirement_id: str) -> dict:
    contract = compile_acceptance_contract(_tongs_requirements())
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
        {"target": {"observable_id": "upper_jaw_pose"}},
        trace,
        plan=_plan() if plan is None else plan,
    )


class TongsTaskEntryTests(unittest.TestCase):
    def test_active_tongs_tasks_compile_from_their_approved_suite(self) -> None:
        contract = compile_acceptance_contract(_tongs_requirements())
        verify_acceptance_contract(contract)

        self.assertEqual(len(contract["requirements"]), 11)
        self.assertTrue(
            all("task_suite_binding" in item for item in contract["requirements"])
        )
        by_id = {
            item["requirement_id"]: item for item in contract["requirements"]
        }
        duration = by_id["tongs.minimum_hold_duration"]
        self.assertEqual(duration["metric_id"], "tongs.hold_duration_s")
        self.assertNotIn(duration["metric_id"], METRIC_REGISTRY)
        self.assertEqual(duration["metric_version"], TONGS_TASK_METRIC_VERSION)
        self.assertEqual(
            duration["target"],
            {"comparator": "at_least", "value": 0.35, "unit": "s"},
        )
        binding = duration["task_suite_binding"]
        self.assertEqual(binding["sha256"], TONGS_TASK_SUITE_SHA256)
        self.assertEqual(binding["evaluation_phase_ids"], ["tong_hold_verify"])
        self.assertEqual(binding["allowed_input_port_ids"], ["hand_squeeze_torque"])
        self.assertEqual(
            binding["declared_target"],
            {
                "comparator": "at_least",
                "value": 0.35,
                "unit": "s",
                "tolerance_mode": "none",
            },
        )
        zero_input = by_id["tongs.hand_input_zero_during_release"]
        self.assertEqual(zero_input["metric_id"], "tongs.release_input_zero")
        self.assertEqual(
            zero_input["metric_version"],
            TONGS_TASK_METRIC_VERSION,
        )
        self.assertEqual(
            zero_input["task_suite_binding"]["evaluation_phase_ids"],
            [
                "tong_release_passive_return",
                "tong_release_final_table_settle",
                "tong_release_observe",
            ],
        )
        registered = by_id["tongs.block_not_directly_actuated"]
        self.assertEqual(registered["metric_id"], "actuation_provenance")
        self.assertEqual(registered["metric_version"], METRIC_REGISTRY_VERSION)
        self.assertEqual(
            registered["task_suite_binding"]["evaluation_phase_ids"],
            [
                "tong_pickup_close",
                "tong_pickup_grip_verify",
                "tong_pickup_lift",
                "tong_pickup_verify",
            ],
        )
        returns = by_id["tongs.joint_returns_to_zero"]
        self.assertEqual(returns["metric_version"], METRIC_REGISTRY_VERSION)
        self.assertEqual(
            returns["target"],
            {
                "comparator": "within",
                "value": 0.0,
                "unit": "rad",
                "tolerance": 0.05235987755982989,
            },
        )

    def test_tongs_threshold_cannot_diverge_from_task_suite(self) -> None:
        value = _tongs_requirements()
        value["tasks"][1]["acceptance_requirements"][0]["target"]["value"] = 0.3

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "TONGS_TASK_REQUIREMENT_MISMATCH")

    def test_tongs_suite_fingerprint_change_fails_closed(self) -> None:
        with mock.patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "intake.tongs_task_suite.TONGS_TASK_SUITE_SHA256",
            "0" * 64,
        ):
            with self.assertRaises(AcceptanceContractError) as caught:
                compile_acceptance_contract(_tongs_requirements())

        self.assertEqual(caught.exception.code, "TONGS_TASK_SUITE_CHANGED")

    def test_tongs_unknown_evaluator_fails_closed(self) -> None:
        value = _tongs_requirements()
        value["tasks"][1]["acceptance_requirements"][0][
            "metric_id"
        ] = "tongs.bogus_evaluator"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "TONGS_TASK_EVALUATOR_MISMATCH")

    def test_tongs_warning_criterion_is_not_a_pass_gate(self) -> None:
        # The historical 17 degree jaw-travel observation is WARNING
        # disposition; citing it as a requirement fails closed as unknown.
        value = _tongs_requirements()
        value["tasks"][0]["acceptance_requirements"][0][
            "requirement_id"
        ] = "tongs.historical_jaw_travel_reference"
        value["tasks"][0]["acceptance_requirements"][0][
            "metric_id"
        ] = "continuous_rotation_rad"
        value["tasks"][0]["acceptance_requirements"][0]["target"] = {
            "comparator": "at_least",
            "value": 0.29670597283903605,
            "unit": "rad",
        }

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "TONGS_TASK_CRITERION_UNKNOWN")

    def test_off_family_task_cannot_cite_tongs_evaluators(self) -> None:
        value = _tongs_requirements()
        value["tasks"][1]["task_id"] = "tongs.shake_the_block"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_tongs_contract_cannot_drop_its_suite_binding(self) -> None:
        contract = compile_acceptance_contract(_tongs_requirements())
        del contract["requirements"][5]["task_suite_binding"]
        _rehash(contract)

        with self.assertRaises(AcceptanceContractError) as caught:
            verify_acceptance_contract(contract)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_tongs_binding_cannot_diverge_from_template(self) -> None:
        contract = compile_acceptance_contract(_tongs_requirements())
        requirement = copy.deepcopy(contract["requirements"][5])
        requirement["task_suite_binding"]["declared_target"]["value"] = 0.3

        with self.assertRaises(TongsTaskSuiteError) as caught:
            verify_tongs_task_suite_binding(requirement)

        self.assertEqual(caught.exception.code, "TONGS_TASK_BINDING_CHANGED")

    def test_tongs_evaluator_ids_are_motion_window_metrics(self) -> None:
        self.assertIn("tongs.hold_duration_s", MOTION_WINDOW_METRICS)
        self.assertIn("tongs.release_input_zero", MOTION_WINDOW_METRICS)

    def test_tongs_evaluator_ids_are_registered_obligation_checks(self) -> None:
        inventory = build_requirement_obligation_inventory(_tongs_requirements())
        catalog = requirement_check_catalog(inventory)

        modes = {entry["check_id"]: entry["mode"] for entry in catalog}
        self.assertEqual(modes["validator.metric.tongs.hold_duration_s"], "validator")
        self.assertEqual(
            modes["validator.metric.tongs.release_input_zero"],
            "validator",
        )

    def test_hold_duration_pass_fail_and_integration_dust(self) -> None:
        passing = _evaluate(
            "tongs.minimum_hold_duration",
            _hold_trace(hold_s=0.35),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.35)

        short = _evaluate(
            "tongs.minimum_hold_duration",
            _hold_trace(hold_s=0.30),
        )
        self.assertFalse(short["passed"])
        self.assertAlmostEqual(short["measured"], 0.30)

        # Integration time bases accumulate machine-scale dust below the
        # validator's 1e-9 s phase-boundary tolerance; a certified 0.35 s hold
        # must still report its full duration.
        dusty = _evaluate(
            "tongs.minimum_hold_duration",
            _hold_trace(hold_s=0.35, last_time_s=1.0 + 0.34999999999996696),
        )
        self.assertTrue(dusty["passed"])
        self.assertAlmostEqual(dusty["measured"], 0.35)

        # The smallest resolvable shortfall below the declared hold still
        # fails: dust rounding never promotes a short window.
        minimal = _evaluate(
            "tongs.minimum_hold_duration",
            _hold_trace(hold_s=0.35, last_time_s=1.0 + 0.349999999),
        )
        self.assertFalse(minimal["passed"])
        self.assertAlmostEqual(minimal["measured"], 0.349999999)

    def test_release_input_zero_pass_and_fail(self) -> None:
        passing = _evaluate(
            "tongs.hand_input_zero_during_release",
            _release_trace(),
        )
        self.assertTrue(passing["passed"])
        self.assertIs(passing["measured"], True)

        driven = _evaluate(
            "tongs.hand_input_zero_during_release",
            _release_trace(window_torque_nm=0.4),
        )
        self.assertFalse(driven["passed"])
        self.assertIs(driven["measured"], False)

    def test_tongs_metrics_fail_closed_without_window(self) -> None:
        trace = _hold_trace()
        for sample in trace["samples"]:
            sample["phase_id"] = "tong_hold_lift"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("tongs.minimum_hold_duration", trace)

        self.assertEqual(caught.exception.code, "TASK_SUITE_WINDOW_EMPTY")

    def test_tongs_metrics_fail_closed_without_baseline(self) -> None:
        trace = _hold_trace()
        trace["samples"] = trace["samples"][2:]

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("tongs.minimum_hold_duration", trace)

        self.assertEqual(caught.exception.code, "TASK_SUITE_BASELINE_MISSING")

    def test_tongs_metrics_fail_closed_without_input_field(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "tongs.hand_input_zero_during_release",
                _release_trace(drop_window_field=True),
            )

        self.assertEqual(caught.exception.code, "TONGS_RELEASE_INPUT_TRACE_MISSING")

    def test_tongs_metrics_fail_closed_without_family_spec(self) -> None:
        requirement = _compiled_requirement("tongs.minimum_hold_duration")
        with self.assertRaises(MetricEvaluationError) as caught:
            evaluate_registered_metric(
                requirement,
                {"target": {"observable_id": "upper_jaw_pose"}},
                _hold_trace(),
                plan=None,
            )

        self.assertEqual(caught.exception.code, "TONGS_FAMILY_SPEC_MISSING")

    def test_tongs_metrics_fail_closed_on_invalid_family_spec(self) -> None:
        invalid = _family_spec()
        invalid["mechanism_kind"] = "not_an_axis_spring"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "tongs.minimum_hold_duration",
                _hold_trace(),
                plan=_plan(invalid),
            )

        self.assertEqual(caught.exception.code, "TONGS_FAMILY_SPEC_INVALID")

    def test_tongs_metrics_fail_closed_on_non_tabletop_fixture(self) -> None:
        non_tabletop = _family_spec()
        non_tabletop["contact_fixture"] = {
            "shape": "fixed_box",
            "center_abs_m": [0.15, 0.0, 0.01],
            "size_m": [0.02, 0.02, 0.01],
            "friction": 0.5,
            "hold_duration_s": 0.35,
        }

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "tongs.hand_input_zero_during_release",
                _release_trace(),
                plan=_plan(non_tabletop),
            )

        self.assertEqual(caught.exception.code, "TONGS_FAMILY_SPEC_INVALID")

    def test_active_tongs_programs_have_exact_strict_policies(self) -> None:
        loads = {
            case.program: case
            for case in default_revolute_load_cases(_load_plan_namespace())
            if case.program.startswith("tong_tabletop_")
        }
        self.assertEqual(
            set(loads),
            {
                "tong_tabletop_pickup",
                "tong_tabletop_hold",
                "tong_tabletop_release",
            },
        )

        pickup = task_program_spec_for_load_case(
            loads["tong_tabletop_pickup"],
            plugin_id="plain_revolute_v1",
        )
        self.assertEqual(
            pickup.powered_phase_ids,
            (
                "tong_pickup_close",
                "tong_pickup_grip_verify",
                "tong_pickup_lift",
                "tong_pickup_verify",
            ),
        )
        self.assertEqual(
            pickup.return_phase_ids,
            ("tong_pickup_initial_table_settle", "tong_pickup_approach"),
        )
        self.assertEqual(pickup.return_input_policy, "zero")
        self.assertEqual(pickup.adaptive_phase_ids, ("tong_pickup_close",))
        self.assertEqual(pickup.executed_phase_extension_keys, ())

        hold = task_program_spec_for_load_case(
            loads["tong_tabletop_hold"],
            plugin_id="plain_revolute_v1",
        )
        self.assertEqual(
            hold.powered_phase_ids,
            (
                "tong_hold_close",
                "tong_hold_grip_verify",
                "tong_hold_lift",
                "tong_hold_verify",
            ),
        )
        self.assertEqual(hold.adaptive_phase_ids, ("tong_hold_close",))

        release = task_program_spec_for_load_case(
            loads["tong_tabletop_release"],
            plugin_id="plain_revolute_v1",
        )
        self.assertEqual(
            release.powered_phase_ids,
            (
                "tong_release_close",
                "tong_release_grip_verify",
                "tong_release_lift",
                "tong_release_preload",
                "tong_release_gradual_open",
            ),
        )
        self.assertEqual(
            release.return_phase_ids,
            (
                "tong_release_initial_table_settle",
                "tong_release_approach",
                "tong_release_passive_return",
                "tong_release_final_table_settle",
                "tong_release_observe",
            ),
        )
        self.assertEqual(release.return_input_policy, "zero")
        self.assertEqual(release.adaptive_phase_ids, ("tong_release_close",))

    def test_tongs_program_policy_fails_closed_on_unknown_layout(self) -> None:
        load_case = LoadCase(
            load_case_id="tong_tabletop_pickup",
            family="revolute",
            program="tong_tabletop_pickup",
            purpose="nominal",
            moving_mass_kg=0.12,
            step_s=0.0001,
            sample_interval_s=0.05,
            phases=(
                LoadPhase(
                    "tong_pickup_close",
                    "torque",
                    (0.0, 0.0, 1.0),
                    1.1,
                    3.0,
                ),
            ),
        )

        with self.assertRaises(MechanismValidationError) as caught:
            task_program_spec_for_load_case(
                load_case,
                plugin_id="plain_revolute_v1",
            )

        self.assertEqual(caught.exception.code, "TASK_PROGRAM_POLICY_UNREGISTERED")


if __name__ == "__main__":
    unittest.main()
