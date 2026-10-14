from __future__ import annotations

import copy
import hashlib
import json
import unittest

from simulation_ready_cad_workflow.validation_pipeline.intake.acceptance_contract import (
    AcceptanceContractError,
    METRIC_REGISTRY,
    compile_acceptance_contract,
    verify_acceptance_contract,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.clock_task_suite import (
    CLOCK_TASK_METRIC_VERSION,
    CLOCK_TASK_SUITE_SHA256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.acceptance_metrics import (
    evaluate_registered_metric,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    task_program_spec_for_load_case,
)
from simulation_ready_cad_workflow.validation_pipeline.contracts.requirement_obligations import (
    build_requirement_obligation_inventory,
    requirement_check_catalog,
)


def _timekeeping_requirements() -> dict:
    clauses = (
        "minute travel >= 0.27 rad (0.02 rad shortfall); "
        "hour/minute ratio = 1/12 within 2%; "
        "second/minute ratio = 60 within 2%; "
        "all hands follow frozen display direction"
    )
    common = {
        "input_role": "timekeeping barrel",
        "observable_quantity": "pose_6d",
        "test_program": "task_clock_cycle",
        "test_load_case_id": "task_three_hand_timekeeping",
        "required_branches": ["physical", "ideal_reference"],
        "allow_direct_output_actuation": False,
    }
    return {
        "schema_version": "0.2",
        "requirements_id": "clock_timekeeping_entry_test_v1",
        "prompt": clauses,
        "tasks": [
            {
                "task_id": "run_three_hand_clock_logic",
                "description": "Verify the three clock hands.",
                "acceptance_requirements": [
                    {
                        **common,
                        "requirement_id": "clock.minute_hand_minimum_travel",
                        "prompt_clause": "minute travel >= 0.27 rad (0.02 rad shortfall)",
                        "target_role": "minute hand",
                        "metric_id": "continuous_rotation_rad",
                        "direction": "positive",
                        "target": {
                            "comparator": "at_least",
                            "value": 0.27,
                            "unit": "rad",
                            "tolerance": 0.02,
                        },
                    },
                    {
                        **common,
                        "requirement_id": "clock.hour_to_minute_ratio",
                        "prompt_clause": "hour/minute ratio = 1/12 within 2%",
                        "target_role": "hour hand",
                        "reference_role": "minute hand",
                        "metric_id": "angular_ratio",
                        "direction": "either",
                        "target": {
                            "comparator": "within",
                            "value": 1.0 / 12.0,
                            "unit": "ratio",
                            "tolerance": (1.0 / 12.0) * 0.02,
                        },
                    },
                    {
                        **common,
                        "requirement_id": "clock.second_to_minute_ratio",
                        "prompt_clause": "second/minute ratio = 60 within 2%",
                        "target_role": "second hand",
                        "reference_role": "minute hand",
                        "metric_id": "angular_ratio",
                        "direction": "either",
                        "target": {
                            "comparator": "within",
                            "value": 60.0,
                            "unit": "ratio",
                            "tolerance": 1.2,
                        },
                    },
                    {
                        **common,
                        "requirement_id": "clock.display_direction_consistency",
                        "prompt_clause": "all hands follow frozen display direction",
                        "target_role": "minute hand",
                        "metric_id": "clock.display_direction_consistent",
                        "direction": "either",
                        "target": {
                            "comparator": "equals",
                            "value": True,
                            "unit": "bool",
                        },
                    },
                ],
            }
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


class ClockTaskEntryTests(unittest.TestCase):
    def test_active_clock_task_compiles_from_its_approved_suite(self) -> None:
        contract = compile_acceptance_contract(_timekeeping_requirements())
        verify_acceptance_contract(contract)

        self.assertEqual(len(contract["requirements"]), 4)
        self.assertTrue(
            all("task_suite_binding" in item for item in contract["requirements"])
        )
        minute = contract["requirements"][0]
        self.assertEqual(minute["target"]["value"], 0.25)
        self.assertEqual(
            minute["task_suite_binding"]["declared_target"]["value"],
            0.27,
        )
        predicate = contract["requirements"][-1]
        self.assertNotIn(predicate["metric_id"], METRIC_REGISTRY)
        self.assertEqual(predicate["metric_version"], CLOCK_TASK_METRIC_VERSION)
        self.assertEqual(
            predicate["task_suite_binding"]["sha256"],
            CLOCK_TASK_SUITE_SHA256,
        )

    def test_clock_threshold_cannot_diverge_from_task_suite(self) -> None:
        value = _timekeeping_requirements()
        value["tasks"][0]["acceptance_requirements"][0]["target"]["value"] = 0.28

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "CLOCK_TASK_REQUIREMENT_MISMATCH")

    def test_manual_setting_uses_corrected_active_binding(self) -> None:
        clause = "manual minute travel >= 0.2 rad (0.02 rad shortfall)"
        value = {
            "schema_version": "0.2",
            "requirements_id": "clock_manual_setting_entry_test_v1",
            "prompt": clause,
            "tasks": [
                {
                    "task_id": "set_time_manually",
                    "description": "Set the clock through the rear input.",
                    "acceptance_requirements": [
                        {
                            "requirement_id": "clock.manual_setting_minute_travel",
                            "prompt_clause": clause,
                            "input_role": "rear setting input",
                            "target_role": "minute hand",
                            "metric_id": "continuous_rotation_rad",
                            "observable_quantity": "pose_6d",
                            "direction": "positive",
                            "target": {
                                "comparator": "at_least",
                                "value": 0.2,
                                "unit": "rad",
                                "tolerance": 0.02,
                            },
                            "test_program": "manual_time_setting_clock",
                            "test_load_case_id": "task_manual_time_setting",
                            "required_branches": ["physical", "ideal_reference"],
                            "allow_direct_output_actuation": False,
                        }
                    ],
                }
            ],
        }

        contract = compile_acceptance_contract(value)

        self.assertAlmostEqual(
            contract["requirements"][0]["target"]["value"],
            0.18,
        )
        self.assertEqual(
            contract["requirements"][0]["task_suite_binding"]["task_id"],
            "set_time_manually",
        )

    def test_clock_obligation_checks_are_task_specific_and_registered(self) -> None:
        requirements = _timekeeping_requirements()
        requirements["non_numeric_engineering_constraints"] = [
            {
                "obligation_id": "engineering.clock_topology",
                "statement": "Use isolated clock power paths.",
                "coverage_mode": "validator",
                "required_check_ids": [
                    "validator.clock.independent_alarm_power_paths"
                ],
                "required_capability_ids": [],
                "required_evidence_roles": ["validation.branch_semantics"],
            },
            {
                "obligation_id": "engineering.clock_saved_poses",
                "statement": "Audit every saved clock pose.",
                "coverage_mode": "saved_state_audit",
                "required_check_ids": [
                    "validator.mechanical.saved_pose_legality"
                ],
                "required_capability_ids": [],
                "required_evidence_roles": [
                    "validation.saved_pose_geometry_audit"
                ],
            },
            {
                "obligation_id": "engineering.clock_boundary",
                "statement": "Keep clock claims inside the guided tests.",
                "coverage_mode": "policy",
                "required_check_ids": [
                    "policy.clock.validation_boundaries"
                ],
                "required_capability_ids": [],
                "required_evidence_roles": ["validation.terminal_verdict"],
            },
        ]
        visual_ids = (
            "reviewer.visual.complete_twin_bell_clock",
            "reviewer.visual.clock_three_hand_motion",
            "reviewer.visual.clock_manual_setting",
            "reviewer.visual.clock_independent_alarm",
            "reviewer.visual.clock_alarm_target_setting",
            "reviewer.visual.clock_selected_position_alarm",
        )
        requirements["visual_evidence_checklist"] = [
            {
                "obligation_id": f"visual.clock_{index}",
                "statement": f"Review clock evidence item {index}.",
                "coverage_mode": "reviewer_visual",
                "required_check_ids": [check_id],
                "required_capability_ids": [],
                "required_evidence_roles": ["dashboard.physics_replay"],
            }
            for index, check_id in enumerate(visual_ids, start=1)
        ]

        catalog = requirement_check_catalog(
            build_requirement_obligation_inventory(requirements)
        )
        checks = {item["check_id"]: item for item in catalog}

        expected = {
            "validator.clock.independent_alarm_power_paths",
            "validator.mechanical.saved_pose_legality",
            "policy.clock.validation_boundaries",
            *visual_ids,
        }
        self.assertTrue(expected <= set(checks))
        for check_id in expected:
            self.assertTrue(checks[check_id]["examines"])
            self.assertTrue(checks[check_id]["pass_condition"])
            self.assertTrue(checks[check_id]["evidence"])

    def test_saved_clock_contract_cannot_drop_its_suite_binding(self) -> None:
        contract = compile_acceptance_contract(_timekeeping_requirements())
        del contract["requirements"][0]["task_suite_binding"]
        _rehash(contract)

        with self.assertRaises(AcceptanceContractError) as caught:
            verify_acceptance_contract(contract)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_METRIC_UNREGISTERED")

    def test_clock_direction_predicate_uses_only_its_frozen_phase(self) -> None:
        requirement = compile_acceptance_contract(
            _timekeeping_requirements()
        )["requirements"][-1]
        samples = [
            {
                "phase_id": "initial",
                "clock_output_angle_rad": 0.0,
                "hour_hand_angle_rad": 0.0,
                "second_hand_angle_rad": 0.0,
            },
            {
                "phase_id": "wind_and_advance",
                "clock_output_angle_rad": 0.3,
                "hour_hand_angle_rad": 0.025,
                "second_hand_angle_rad": 18.0,
            },
            {
                "phase_id": "controlled_return",
                "clock_output_angle_rad": -4.0,
                "hour_hand_angle_rad": 3.0,
                "second_hand_angle_rad": -2.0,
            },
        ]
        result = evaluate_registered_metric(
            requirement,
            {"target": {"observable_id": "minute_hand_pose"}},
            {"status": "COMPLETED", "branch": "physical", "samples": samples},
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["evidence"]["task_suite_binding"]["evaluation_phase_ids"],
            ["wind_and_advance"],
        )

    def test_active_clock_programs_have_exact_strict_policies(self) -> None:
        manual = LoadCase(
            load_case_id="task_manual_time_setting",
            family="compound",
            program="manual_time_setting_clock",
            purpose="nominal",
            moving_mass_kg=1.0,
            step_s=0.001,
            sample_interval_s=0.01,
            phases=tuple(
                LoadPhase(name, "none", None, 0.0, 0.1)
                for name in (
                    "engage_and_set_forward",
                    "setting_settle",
                    "set_reverse",
                    "setting_home_settle",
                )
            ),
        )
        alarm_setting = LoadCase(
            load_case_id="task_alarm_time_setting",
            family="compound",
            program="hour_hand_alarm_setting_clock",
            purpose="nominal",
            moving_mass_kg=1.0,
            step_s=0.001,
            sample_interval_s=0.01,
            phases=tuple(
                LoadPhase(name, "none", None, 0.0, 0.1)
                for name in (
                    "turn_hour_hand_to_alarm_target",
                    "alarm_setting_latched",
                )
            ),
        )
        scheduled = LoadCase(
            load_case_id="task_scheduled_alarm_trigger",
            family="compound",
            program="scheduled_alarm_clock",
            purpose="nominal",
            moving_mass_kg=1.0,
            step_s=0.001,
            sample_interval_s=0.01,
            phases=tuple(
                LoadPhase(name, "none", None, 0.0, 0.1)
                for name in (
                    "turn_hour_hand_to_alarm_target",
                    "alarm_setting_latched",
                    "restore_hour_hand_to_current_time",
                    "alarm_target_ready",
                    "advance_to_alarm_time",
                    "ring_with_timekeeping",
                    "scheduled_alarm_settle",
                )
            ),
        )

        manual_policy = task_program_spec_for_load_case(
            manual, plugin_id="spur_gear_clock_v1"
        )
        alarm_setting_policy = task_program_spec_for_load_case(
            alarm_setting, plugin_id="spur_gear_clock_v1"
        )
        scheduled_policy = task_program_spec_for_load_case(
            scheduled, plugin_id="spur_gear_clock_v1"
        )

        self.assertEqual(
            manual_policy.powered_phase_ids,
            ("engage_and_set_forward",),
        )
        self.assertEqual(
            manual_policy.return_phase_ids,
            ("setting_settle", "set_reverse", "setting_home_settle"),
        )
        self.assertEqual(
            alarm_setting_policy.powered_phase_ids,
            ("turn_hour_hand_to_alarm_target",),
        )
        self.assertEqual(
            scheduled_policy.powered_phase_ids,
            ("advance_to_alarm_time", "ring_with_timekeeping"),
        )
        self.assertEqual(
            scheduled_policy.adaptive_phase_ids,
            ("advance_to_alarm_time",),
        )


if __name__ == "__main__":
    unittest.main()
