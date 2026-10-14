from __future__ import annotations

import copy
import hashlib
import json
import unittest
from dataclasses import replace
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
from simulation_ready_cad_workflow.validation_pipeline.intake.rack_pinion_gate_task_suite import (
    RACK_PINION_GATE_TASK_METRIC_VERSION,
    RACK_PINION_GATE_TASK_SUITE_SHA256,
    RackPinionGateTaskSuiteError,
    verify_rack_pinion_gate_task_suite_binding,
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
    MechanismPlan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.rack_pinion_loads_v3 import (
    default_rack_pinion_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    MOTION_WINDOW_METRICS,
    task_program_spec_for_load_case,
)


def _rack_pinion_gate_requirements() -> dict:
    clauses = (
        "the gate reaches 180 mm above its starting position within 4 mm; "
        "the gate returns to its bottom position within 4 mm; "
        "all commanded gate-drive input remains zero during the anti-drop observation interval; "
        "the zero-input observation interval lasts at least 0.5 s; "
        "downward gate slip during the zero-input interval is at most 4 mm"
    )
    common = {
        "input_role": "manual crank input",
        "target_role": "rack and gate body",
        "required_branches": ["physical", "ideal_reference"],
        "allow_direct_output_actuation": False,
        "test_program": "task_rack_roundtrip",
        "test_load_case_id": "task_rack_roundtrip",
    }
    return {
        "schema_version": "0.2",
        "requirements_id": "rack_pinion_gate_task_entry_test_v1",
        "prompt": clauses,
        "tasks": [
            {
                "task_id": "raise_gate",
                "description": "Raise the gate to the approved upper position.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "raised_position_target",
                        "prompt_clause": "the gate reaches 180 mm above its starting position within 4 mm",
                        "observable_quantity": "pose_6d",
                        "metric_id": "signed_displacement_m",
                        "direction": "positive",
                        "target": {
                            "comparator": "equals",
                            "value": 0.18,
                            "unit": "m",
                            "tolerance": 0.004,
                        },
                    },
                ],
            },
            {
                "task_id": "lower_gate_to_bottom",
                "description": "Return the gate to the approved bottom position.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "bottom_position_target",
                        "prompt_clause": "the gate returns to its bottom position within 4 mm",
                        "observable_quantity": "pose_6d",
                        "metric_id": "signed_displacement_m",
                        "direction": "either",
                        "target": {
                            "comparator": "equals",
                            "value": 0.0,
                            "unit": "m",
                            "tolerance": 0.004,
                        },
                    },
                ],
            },
            {
                "task_id": "prevent_gate_drop_after_release",
                "description": "Hold the raised gate with zero commanded input.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "zero_input_hold",
                        "prompt_clause": "all commanded gate-drive input remains zero during the anti-drop observation interval",
                        "observable_quantity": "actuation_log",
                        "metric_id": "gate_zero_input_hold",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                    {
                        **common,
                        "requirement_id": "minimum_hold_duration",
                        "prompt_clause": "the zero-input observation interval lasts at least 0.5 s",
                        "observable_quantity": "actuation_log",
                        "metric_id": "gate_zero_input_hold_duration_s",
                        "target": {
                            "comparator": "at_least",
                            "value": 0.5,
                            "unit": "s",
                        },
                    },
                    {
                        **common,
                        "requirement_id": "maximum_downward_slip",
                        "prompt_clause": "downward gate slip during the zero-input interval is at most 4 mm",
                        "observable_quantity": "pose_6d",
                        "metric_id": "gate_maximum_downward_slip_m",
                        "target": {
                            "comparator": "at_most",
                            "value": 0.004,
                            "unit": "m",
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
        "mechanism_kind": "rack_pinion",
        "mechanism_id": "manual_gate_drive",
        "bodies": [
            {
                "body_id": "ground",
                "group_ids": ["frame"],
                "fixed": True,
                "mass_kg": 8.0,
                "center_abs_m": [0.0, 0.0, 0.5],
                "size_m": [0.4, 0.2, 1.0],
            },
            {
                "body_id": "pinion",
                "group_ids": ["crank", "pinion"],
                "fixed": False,
                "mass_kg": 0.8,
                "center_abs_m": [0.0, 0.0, 0.35],
                "size_m": [0.12, 0.08, 0.12],
            },
            {
                "body_id": "rack",
                "group_ids": ["rack", "gate"],
                "fixed": False,
                "mass_kg": 2.5,
                "center_abs_m": [0.04, 0.0, 0.42],
                "size_m": [0.12, 0.05, 0.5],
            },
        ],
        "fixed_body_id": "ground",
        "pinion_body_id": "pinion",
        "rack_body_id": "rack",
        "pinion_origin_abs_m": [0.0, 0.0, 0.35],
        "pinion_axis_abs": [0.0, 1.0, 0.0],
        "rack_axis_abs": [0.0, 0.0, 1.0],
        "pinion_pitch_radius_m": 0.04,
        "pressure_angle_rad": 0.3490658503988659,
        "home_progress_m": 0.0,
        "outbound_progress_m": 0.18,
        "nominal_input_torque_nm": 3.2,
        "return_input_torque_nm": 2.0,
        "max_torque_phase_s": 1.4,
        "max_return_phase_s": 1.2,
        "settle_phase_s": 0.12,
        "output_progress_tolerance_m": 0.004,
        "max_support_drift_m": 0.0015,
        "max_off_axis_rotation_rad": 0.04,
        "max_coupling_error_m": 0.0015,
        "friction": 0.08,
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
        package_id="manual_gate_package",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="rack_pinion_gate",
        capsule_file="capsule.json",
        family="compound",
        protocol_id="mixed_joint_mechanism",
        fixed_group_ids=("frame",),
        moving_group_ids=("crank", "pinion", "rack", "gate"),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.35),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="pinion_frame",
        input_port_id="manual_crank_torque",
        input_frame_id="pinion_frame",
        input_point_abs_m=(0.0, 0.0, 0.35),
        observable_id="gate_vertical_pose",
        observable_frame_id="rack_frame",
        observable_point_abs_m=(0.04, 0.0, 0.42),
        task_ids=("raise_gate", "lower_gate_to_bottom", "prevent_gate_drop_after_release"),
        characteristic_length_m=1.0,
        moving_extents_m=(0.3, 0.2, 0.7),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _anti_drop_trace(
    *,
    window_torque_nm: float = 0.0,
    hold_s: float = 0.5,
    last_time_s: float | None = None,
    drop_window_field: bool = False,
) -> dict:
    """Synthetic roundtrip trace: a settle baseline, then the zero-input window."""

    def sample(time_s, phase_id, torque_y):
        item = {"time_s": time_s, "phase_id": phase_id}
        item["applied_torque_abs_nm"] = [0.0, torque_y, 0.0]
        item["rack_progress_m"] = 0.17686 if phase_id != "initial" else 0.0
        return item

    samples = [
        sample(0.0, "initial", 0.0),
        sample(2.0, "crank_lift", -8.0),
        sample(2.4, "upper_settle", 0.0),
    ]
    count = int(round(hold_s / 0.05))
    for index in range(1, count + 1):
        samples.append(
            sample(2.4 + index * 0.05, "zero_input_hold", window_torque_nm)
        )
    if last_time_s is not None:
        samples[-1]["time_s"] = last_time_s
    if drop_window_field:
        del samples[3]["applied_torque_abs_nm"]
    return {"status": "COMPLETED", "branch": "physical", "samples": samples}


def _comparison(slip_m: float = 0.0035) -> dict:
    """Physical-branch comparison metrics as saved by the plugin comparator."""

    return {
        "verdict": "PASS",
        "physical_metrics": {
            "phase_downward_slip_m": {"zero_input_hold": slip_m},
        },
    }


def _compiled_requirement(requirement_id: str) -> dict:
    contract = compile_acceptance_contract(_rack_pinion_gate_requirements())
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
    comparison: dict | None = None,
) -> dict:
    return evaluate_registered_metric(
        _compiled_requirement(requirement_id),
        {"target": {"observable_id": "gate_vertical_pose"}},
        trace,
        plan=_plan() if plan is None else plan,
        verified_comparison=comparison,
    )


class RackPinionGateTaskEntryTests(unittest.TestCase):
    def test_active_rack_pinion_gate_tasks_compile_from_their_approved_suite(
        self,
    ) -> None:
        contract = compile_acceptance_contract(_rack_pinion_gate_requirements())
        verify_acceptance_contract(contract)

        self.assertEqual(len(contract["requirements"]), 5)
        self.assertTrue(
            all("task_suite_binding" in item for item in contract["requirements"])
        )
        by_id = {
            item["requirement_id"]: item for item in contract["requirements"]
        }
        zero_input = by_id["zero_input_hold"]
        self.assertEqual(zero_input["metric_id"], "gate_zero_input_hold")
        self.assertNotIn(zero_input["metric_id"], METRIC_REGISTRY)
        self.assertEqual(
            zero_input["metric_version"],
            RACK_PINION_GATE_TASK_METRIC_VERSION,
        )
        self.assertEqual(
            zero_input["target"],
            {"comparator": "equals", "value": True, "unit": "bool"},
        )
        binding = zero_input["task_suite_binding"]
        self.assertEqual(binding["sha256"], RACK_PINION_GATE_TASK_SUITE_SHA256)
        self.assertEqual(binding["evaluation_phase_ids"], ["zero_input_hold"])
        self.assertEqual(
            binding["allowed_input_port_ids"],
            ["manual_crank_torque"],
        )
        self.assertEqual(
            binding["declared_target"],
            {
                "comparator": "equals",
                "value": True,
                "unit": "bool",
                "tolerance_mode": "none",
            },
        )
        duration = by_id["minimum_hold_duration"]
        self.assertEqual(
            duration["metric_id"], "gate_zero_input_hold_duration_s"
        )
        self.assertEqual(
            duration["target"],
            {"comparator": "at_least", "value": 0.5, "unit": "s"},
        )
        slip = by_id["maximum_downward_slip"]
        self.assertEqual(slip["metric_id"], "gate_maximum_downward_slip_m")
        self.assertEqual(
            slip["target"],
            {"comparator": "at_most", "value": 0.004, "unit": "m"},
        )
        self.assertEqual(
            slip["task_suite_binding"]["evaluation_phase_ids"],
            ["zero_input_hold"],
        )
        raised = by_id["raised_position_target"]
        self.assertEqual(raised["metric_id"], "signed_displacement_m")
        self.assertEqual(
            raised["metric_version"],
            METRIC_REGISTRY_VERSION,
        )
        self.assertEqual(
            raised["task_suite_binding"]["evaluation_phase_ids"],
            ["crank_lift", "upper_settle"],
        )
        bottom = by_id["bottom_position_target"]
        self.assertEqual(
            bottom["task_suite_binding"]["evaluation_phase_ids"],
            ["home_settle"],
        )

    def test_rack_pinion_gate_threshold_cannot_diverge_from_task_suite(self) -> None:
        value = _rack_pinion_gate_requirements()
        value["tasks"][2]["acceptance_requirements"][2]["target"]["value"] = 0.005

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_TASK_REQUIREMENT_MISMATCH",
        )

    def test_rack_pinion_gate_suite_fingerprint_change_fails_closed(self) -> None:
        with mock.patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "intake.rack_pinion_gate_task_suite.RACK_PINION_GATE_TASK_SUITE_SHA256",
            "0" * 64,
        ):
            with self.assertRaises(AcceptanceContractError) as caught:
                compile_acceptance_contract(_rack_pinion_gate_requirements())

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_TASK_SUITE_CHANGED",
        )

    def test_rack_pinion_gate_unknown_evaluator_fails_closed(self) -> None:
        value = _rack_pinion_gate_requirements()
        value["tasks"][2]["acceptance_requirements"][0][
            "metric_id"
        ] = "gate_bogus_evaluator"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_TASK_EVALUATOR_MISMATCH",
        )

    def test_warning_criterion_reference_fails_closed(self) -> None:
        value = _rack_pinion_gate_requirements()
        value["tasks"][1]["acceptance_requirements"].append(
            {
                "input_role": "manual crank input",
                "target_role": "rack and gate body",
                "required_branches": ["physical", "ideal_reference"],
                "allow_direct_output_actuation": False,
                "test_program": "task_rack_roundtrip",
                "test_load_case_id": "task_rack_roundtrip",
                "requirement_id": "historical_roundtrip_travel",
                "prompt_clause": "the gate returns to its bottom position within 4 mm",
                "observable_quantity": "pose_6d",
                "metric_id": "cumulative_travel_m",
                "target": {
                    "comparator": "at_least",
                    "value": 0.34,
                    "unit": "m",
                },
            }
        )

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_TASK_CRITERION_UNKNOWN",
        )

    def test_off_family_task_cannot_cite_rack_pinion_gate_evaluators(self) -> None:
        value = _rack_pinion_gate_requirements()
        value["tasks"][2]["task_id"] = "shake_gate_until_it_drops"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_rack_pinion_gate_contract_cannot_drop_its_suite_binding(
        self,
    ) -> None:
        contract = compile_acceptance_contract(_rack_pinion_gate_requirements())
        del contract["requirements"][2]["task_suite_binding"]
        _rehash(contract)

        with self.assertRaises(AcceptanceContractError) as caught:
            verify_acceptance_contract(contract)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_saved_rack_pinion_gate_binding_cannot_diverge_from_template(
        self,
    ) -> None:
        contract = compile_acceptance_contract(_rack_pinion_gate_requirements())
        requirement = copy.deepcopy(contract["requirements"][2])
        requirement["task_suite_binding"]["declared_target"]["value"] = False

        with self.assertRaises(RackPinionGateTaskSuiteError) as caught:
            verify_rack_pinion_gate_task_suite_binding(requirement)

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_TASK_BINDING_CHANGED",
        )

    def test_suite_version_1_0_0_reference_fails_closed(self) -> None:
        # The 1.0.0 suite declares the anti-drop task IMPLEMENTATION_REQUIRED;
        # only the pinned 1.1.0 binding is accepted.
        contract = compile_acceptance_contract(_rack_pinion_gate_requirements())
        requirement = copy.deepcopy(contract["requirements"][2])
        requirement["task_suite_binding"]["version"] = "1.0.0"

        with self.assertRaises(RackPinionGateTaskSuiteError) as caught:
            verify_rack_pinion_gate_task_suite_binding(requirement)

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_TASK_BINDING_CHANGED",
        )

    def test_rack_pinion_gate_evaluator_ids_are_motion_window_metrics(self) -> None:
        self.assertIn("gate_zero_input_hold", MOTION_WINDOW_METRICS)
        self.assertIn("gate_zero_input_hold_duration_s", MOTION_WINDOW_METRICS)
        self.assertIn("gate_maximum_downward_slip_m", MOTION_WINDOW_METRICS)

    def test_rack_pinion_gate_evaluator_ids_are_registered_obligation_checks(
        self,
    ) -> None:
        inventory = build_requirement_obligation_inventory(
            _rack_pinion_gate_requirements()
        )
        catalog = requirement_check_catalog(inventory)

        modes = {entry["check_id"]: entry["mode"] for entry in catalog}
        self.assertEqual(modes["validator.metric.gate_zero_input_hold"], "validator")
        self.assertEqual(
            modes["validator.metric.gate_zero_input_hold_duration_s"],
            "validator",
        )
        self.assertEqual(
            modes["validator.metric.gate_maximum_downward_slip_m"],
            "validator",
        )

    def test_zero_input_hold_pass_and_fail(self) -> None:
        passing = _evaluate("zero_input_hold", _anti_drop_trace())
        self.assertTrue(passing["passed"])
        self.assertIs(passing["measured"], True)

        driven = _evaluate(
            "zero_input_hold",
            _anti_drop_trace(window_torque_nm=0.4),
        )
        self.assertFalse(driven["passed"])
        self.assertIs(driven["measured"], False)

    def test_zero_input_hold_duration_pass_fail_and_integration_dust(self) -> None:
        passing = _evaluate("minimum_hold_duration", _anti_drop_trace(hold_s=0.5))
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.5)

        short = _evaluate("minimum_hold_duration", _anti_drop_trace(hold_s=0.4))
        self.assertFalse(short["passed"])
        self.assertAlmostEqual(short["measured"], 0.4)

        # Integration time bases accumulate machine-scale dust below the
        # validator's 1e-9 s phase-boundary tolerance; a certified 0.5 s hold
        # must still report its full duration.
        dusty = _evaluate(
            "minimum_hold_duration",
            _anti_drop_trace(hold_s=0.5, last_time_s=2.4 + 0.4999999999999),
        )
        self.assertTrue(dusty["passed"])
        self.assertAlmostEqual(dusty["measured"], 0.5)

        # The smallest resolvable shortfall below the declared hold still
        # fails: dust rounding never promotes a short window.
        minimal = _evaluate(
            "minimum_hold_duration",
            _anti_drop_trace(hold_s=0.5, last_time_s=2.4 + 0.499999999),
        )
        self.assertFalse(minimal["passed"])
        self.assertAlmostEqual(minimal["measured"], 0.499999999)

    def test_maximum_downward_slip_pass_and_fail(self) -> None:
        passing = _evaluate(
            "maximum_downward_slip",
            _anti_drop_trace(),
            comparison=_comparison(slip_m=0.0035),
        )
        self.assertTrue(passing["passed"])
        self.assertAlmostEqual(passing["measured"], 0.0035)

        dropped = _evaluate(
            "maximum_downward_slip",
            _anti_drop_trace(),
            comparison=_comparison(slip_m=0.005),
        )
        self.assertFalse(dropped["passed"])
        self.assertAlmostEqual(dropped["measured"], 0.005)

    def test_rack_pinion_gate_metrics_fail_closed_without_window(self) -> None:
        trace = _anti_drop_trace()
        for sample in trace["samples"]:
            sample["phase_id"] = "upper_settle"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("maximum_downward_slip", trace, comparison=_comparison())

        self.assertEqual(caught.exception.code, "TASK_SUITE_WINDOW_EMPTY")

    def test_rack_pinion_gate_metrics_fail_closed_without_baseline(self) -> None:
        trace = _anti_drop_trace()
        trace["samples"] = trace["samples"][3:]

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("minimum_hold_duration", trace)

        self.assertEqual(caught.exception.code, "TASK_SUITE_BASELINE_MISSING")

    def test_rack_pinion_gate_metrics_fail_closed_without_input_field(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("zero_input_hold", _anti_drop_trace(drop_window_field=True))

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_INPUT_TRACE_MISSING",
        )

    def test_rack_pinion_gate_slip_fail_closed_without_comparison(self) -> None:
        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate("maximum_downward_slip", _anti_drop_trace())

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_COMPARISON_EVIDENCE_MISSING",
        )

    def test_rack_pinion_gate_slip_fail_closed_without_window_metric(self) -> None:
        comparison = _comparison()
        comparison["physical_metrics"]["phase_downward_slip_m"] = {}

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "maximum_downward_slip",
                _anti_drop_trace(),
                comparison=comparison,
            )

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_COMPARISON_EVIDENCE_MISSING",
        )

    def test_rack_pinion_gate_metrics_fail_closed_without_family_spec(self) -> None:
        requirement = _compiled_requirement("maximum_downward_slip")
        with self.assertRaises(MetricEvaluationError) as caught:
            evaluate_registered_metric(
                requirement,
                {"target": {"observable_id": "gate_vertical_pose"}},
                _anti_drop_trace(),
                plan=None,
                verified_comparison=_comparison(),
            )

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_FAMILY_SPEC_MISSING",
        )

    def test_rack_pinion_gate_metrics_fail_closed_on_invalid_family_spec(
        self,
    ) -> None:
        invalid = _family_spec()
        invalid["mechanism_kind"] = "not_a_rack_pinion"

        with self.assertRaises(MetricEvaluationError) as caught:
            _evaluate(
                "maximum_downward_slip",
                _anti_drop_trace(),
                plan=_plan(invalid),
                comparison=_comparison(),
            )

        self.assertEqual(
            caught.exception.code,
            "RACK_PINION_GATE_FAMILY_SPEC_INVALID",
        )

    def test_active_rack_pinion_program_has_exact_strict_policy(self) -> None:
        loads = {
            case.program: case
            for case in default_rack_pinion_load_cases(_plan())
        }
        self.assertIn("task_rack_roundtrip", loads)
        self.assertEqual(
            tuple(phase.phase_id for phase in loads["task_rack_roundtrip"].phases),
            (
                "crank_lift",
                "upper_settle",
                "zero_input_hold",
                "crank_lower",
                "home_settle",
            ),
        )

        roundtrip = task_program_spec_for_load_case(
            loads["task_rack_roundtrip"],
            plugin_id="rack_pinion_v1",
        )
        self.assertEqual(roundtrip.powered_phase_ids, ("crank_lift",))
        self.assertEqual(
            roundtrip.return_phase_ids,
            ("upper_settle", "zero_input_hold", "crank_lower", "home_settle"),
        )
        self.assertEqual(roundtrip.return_input_policy, "selected_input_only")
        self.assertEqual(
            roundtrip.adaptive_phase_ids,
            ("crank_lift", "crank_lower"),
        )


if __name__ == "__main__":
    unittest.main()
