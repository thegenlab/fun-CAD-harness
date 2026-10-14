from __future__ import annotations

import copy
import math
import unittest

from simulation_ready_cad_workflow.validation_pipeline.intake.acceptance_contract import (
    AcceptanceContractError,
    compile_acceptance_contract,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.semantic_binding import (
    SemanticBindingError,
    bind_acceptance_contract,
)


def _requirements() -> dict:
    prompt = (
        "Build a gearbox where the input crank drives the output wheel "
        "through a full rotation."
    )
    return {
        "schema_version": "0.1",
        "requirements_id": "gearbox_requirements_v2",
        "prompt": prompt,
        "tasks": [
            {
                "task_id": "drive_output",
                "description": "Drive the output wheel from the crank.",
                "success_criteria": [],
                "acceptance_requirements": [
                    {
                        "requirement_id": "output_full_turn",
                        "prompt_clause": "the input crank drives the output wheel through a full rotation",
                        "target_role": "output wheel",
                        "input_role": "input crank",
                        "metric_id": "continuous_rotation_rad",
                        "test_program": "task_output_angle_roundtrip",
                        "test_load_case_id": "task_output_angle_roundtrip",
                        "target": {
                            "comparator": "at_least",
                            "value": 2.0 * math.pi,
                            "unit": "rad",
                        },
                        "allow_direct_output_actuation": False,
                        "required_branches": [
                            "physical",
                            "ideal_reference",
                        ],
                    }
                ],
            }
        ],
    }


def _package() -> tuple[dict, dict, dict]:
    mapping = {
        "task_mappings": [
            {
                "task_id": "drive_output",
                "capsule_ids": ["gear_train"],
                "inputs": [
                    {
                        "capsule_id": "gear_train",
                        "port_id": "input_crank_torque",
                    }
                ],
                "observables": [
                    {
                        "capsule_id": "gear_train",
                        "observable_id": "output_wheel_pose",
                    },
                    {
                        "capsule_id": "gear_train",
                        "observable_id": "housing_pose",
                    },
                ],
            }
        ]
    }
    capsules = {
        "gear_train": {
            "capsule_id": "gear_train",
            "name": "Input crank to output wheel gear train",
            "input_ports": [
                {
                    "port_id": "input_crank_torque",
                    "rigid_group_id": "group_input_crank",
                    "frame_id": "frame_input",
                    "wrench_type": "torque",
                    "direction_abs": [0.0, 0.0, 1.0],
                }
            ],
            "observables": [
                {
                    "observable_id": "output_wheel_pose",
                    "rigid_group_id": "group_output_wheel",
                    "frame_id": "frame_output",
                    "quantity": "pose_6d",
                },
                {
                    "observable_id": "housing_pose",
                    "rigid_group_id": "group_housing",
                    "frame_id": "frame_housing",
                    "quantity": "pose_6d",
                },
            ],
        }
    }
    assembly = {
        "parts": [
            {
                "part_id": "steel_input_crank",
                "rigid_group_id": "group_input_crank",
                "source_solid_ids": ["solid_1"],
            },
            {
                "part_id": "output_wheel",
                "rigid_group_id": "group_output_wheel",
                "source_solid_ids": ["solid_2", "solid_3"],
            },
            {
                "part_id": "fixed_housing",
                "rigid_group_id": "group_housing",
                "source_solid_ids": ["solid_4"],
            },
        ]
    }
    return mapping, capsules, assembly


class AcceptanceContractTests(unittest.TestCase):
    def test_legacy_requirements_remain_readable_without_new_authority(self) -> None:
        contract = compile_acceptance_contract(
            {
                "schema_version": "0.1",
                "requirements_id": "legacy_v1",
                "tasks": [
                    {
                        "task_id": "move",
                        "description": "Move.",
                        "success_criteria": ["looks right"],
                    }
                ],
            }
        )

        self.assertEqual(contract["status"], "LEGACY_UNTYPED")
        self.assertFalse(contract["certificate_authority"])
        self.assertEqual(len(contract["contract_sha256"]), 64)

    def test_typed_contract_is_deterministic_and_keeps_continuous_turn(self) -> None:
        first = compile_acceptance_contract(_requirements())
        second = compile_acceptance_contract(_requirements())
        criterion = first["requirements"][0]

        self.assertEqual(first["contract_sha256"], second["contract_sha256"])
        self.assertTrue(first["certificate_authority"])
        self.assertEqual(criterion["metric_id"], "continuous_rotation_rad")
        self.assertNotIn("observable_quantity", criterion)
        self.assertAlmostEqual(criterion["target"]["value"], 2.0 * math.pi)
        self.assertEqual(
            criterion["required_branches"],
            ["physical", "ideal_reference"],
        )

    def test_physical_tong_contact_metric_does_not_require_ideal_dynamics(self) -> None:
        value = _requirements()
        criterion = value["tasks"][0]["acceptance_requirements"][0]
        criterion["metric_id"] = "tong_two_sided_grip"
        criterion["target"] = {
            "comparator": "equals",
            "value": True,
            "unit": "bool",
        }
        criterion["required_branches"] = ["physical"]

        contract = compile_acceptance_contract(value)

        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(contract["requirements"][0]["required_branches"], ["physical"])

    def test_typed_requirement_binds_unique_load_case(self) -> None:
        value = _requirements()
        del value["tasks"][0]["acceptance_requirements"][0][
            "test_load_case_id"
        ]

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(
            caught.exception.json_path,
            "$.tasks[0].acceptance_requirements[0].test_load_case_id",
        )
    def test_prompt_clause_must_come_from_immutable_prompt(self) -> None:
        value = _requirements()
        value["tasks"][0]["acceptance_requirements"][0][
            "prompt_clause"
        ] = "quietly rotate only ten degrees"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "PROMPT_CLAUSE_UNBOUND")

    def test_metric_unit_is_controlled_by_registry(self) -> None:
        value = _requirements()
        value["tasks"][0]["acceptance_requirements"][0]["target"][
            "unit"
        ] = "deg"

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "ACCEPTANCE_UNIT_INVALID")

    def test_output_torque_metric_has_exact_unit_comparator_and_quantity(
        self,
    ) -> None:
        value = _requirements()
        criterion = value["tasks"][0]["acceptance_requirements"][0]
        criterion["metric_id"] = "output_torque_nm"
        criterion["observable_quantity"] = "pose_6d"
        criterion["target"] = {
            "comparator": "at_most",
            "value": 0.05,
            "unit": "N·m",
        }

        contract = compile_acceptance_contract(value)

        self.assertEqual(contract["metric_registry_version"], "metric_registry_v8")
        self.assertEqual(
            contract["requirements"][0]["metric_version"],
            "metric_registry_v8",
        )
        self.assertEqual(contract["requirements"][0]["target"]["unit"], "N·m")

        for field, attacked in (
            ("unit", "Nm"),
            ("comparator", "at_least"),
        ):
            with self.subTest(field=field):
                attack = copy.deepcopy(value)
                attack["tasks"][0]["acceptance_requirements"][0]["target"][
                    field
                ] = attacked
                with self.assertRaises(AcceptanceContractError):
                    compile_acceptance_contract(attack)

    def test_return_to_start_accepts_retained_terminal_travel_lower_bound(self) -> None:
        value = _requirements()
        criterion = value["tasks"][0]["acceptance_requirements"][0]
        criterion["metric_id"] = "return_to_start_rad"
        criterion["target"] = {
            "comparator": "at_least",
            "value": 0.3,
            "unit": "rad",
        }

        contract = compile_acceptance_contract(value)

        self.assertEqual(
            contract["requirements"][0]["target"]["comparator"],
            "at_least",
        )

    def test_observable_quantity_is_frozen_and_registry_checked(self) -> None:
        value = _requirements()
        criterion = value["tasks"][0]["acceptance_requirements"][0]
        criterion["observable_quantity"] = "pose_6d"

        contract = compile_acceptance_contract(value)

        self.assertEqual(
            contract["requirements"][0]["observable_quantity"],
            "pose_6d",
        )

        criterion["observable_quantity"] = "contact_event"
        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)
        self.assertEqual(
            caught.exception.code,
            "ACCEPTANCE_OBSERVABLE_QUANTITY_INVALID",
        )

    def test_every_typed_task_requires_a_machine_criterion(self) -> None:
        value = _requirements()
        value["tasks"].append(
            {
                "task_id": "uncovered_task",
                "description": "Uncovered.",
                "success_criteria": ["free text only"],
            }
        )

        with self.assertRaises(AcceptanceContractError) as caught:
            compile_acceptance_contract(value)

        self.assertEqual(caught.exception.code, "TYPED_ACCEPTANCE_REQUIRED")

    def test_semantic_binding_selects_input_output_and_source_solids(self) -> None:
        contract = compile_acceptance_contract(_requirements())
        mapping, capsules, assembly = _package()
        binding = bind_acceptance_contract(
            contract,
            task_mapping=mapping,
            capsules=capsules,
            assembly=assembly,
        )
        selected = binding["bindings"][0]

        self.assertEqual(binding["status"], "PASS")
        self.assertEqual(
            selected["target"]["observable_id"],
            "output_wheel_pose",
        )
        self.assertEqual(
            selected["input"]["port_id"],
            "input_crank_torque",
        )
        self.assertEqual(
            selected["target"]["source_solid_ids"],
            ["solid_2", "solid_3"],
        )

    def test_ambiguous_target_fails_without_manual_approval(self) -> None:
        contract = compile_acceptance_contract(_requirements())
        mapping, capsules, assembly = _package()
        capsules["gear_train"]["observables"] = [
            {
                "observable_id": "left_output_wheel_pose",
                "rigid_group_id": "group_left_output_wheel",
                "frame_id": "frame_left",
                "quantity": "pose_6d",
            },
            {
                "observable_id": "right_output_wheel_pose",
                "rigid_group_id": "group_right_output_wheel",
                "frame_id": "frame_right",
                "quantity": "pose_6d",
            },
        ]
        mapping["task_mappings"][0]["observables"] = [
            {
                "capsule_id": "gear_train",
                "observable_id": "left_output_wheel_pose",
            },
            {
                "capsule_id": "gear_train",
                "observable_id": "right_output_wheel_pose",
            },
        ]
        assembly["parts"] = [
            {
                "part_id": "left_output_wheel",
                "rigid_group_id": "group_left_output_wheel",
                "source_solid_ids": ["solid_2"],
            },
            {
                "part_id": "right_output_wheel",
                "rigid_group_id": "group_right_output_wheel",
                "source_solid_ids": ["solid_3"],
            },
            assembly["parts"][0],
        ]

        with self.assertRaises(SemanticBindingError) as caught:
            bind_acceptance_contract(
                contract,
                task_mapping=mapping,
                capsules=capsules,
                assembly=assembly,
            )

        self.assertEqual(caught.exception.code, "SEMANTIC_BINDING_AMBIGUOUS")

    def test_frozen_quantity_resolves_same_group_pose_contact_tie(self) -> None:
        requirements = _requirements()
        criterion = requirements["tasks"][0]["acceptance_requirements"][0]
        criterion["metric_id"] = "ground_separation_m"
        criterion["observable_quantity"] = "contact_event"
        criterion["target"] = {
            "comparator": "at_most",
            "value": 0.0002,
            "unit": "m",
        }
        mapping, capsules, assembly = _package()
        capsules["gear_train"]["observables"] = [
            {
                "observable_id": "output_wheel_pose",
                "rigid_group_id": "group_output_wheel",
                "frame_id": "frame_output",
                "quantity": "pose_6d",
            },
            {
                "observable_id": "output_wheel_contact",
                "rigid_group_id": "group_output_wheel",
                "frame_id": "frame_output",
                "quantity": "contact_event",
            },
        ]
        mapping["task_mappings"][0]["observables"] = [
            {
                "capsule_id": "gear_train",
                "observable_id": "output_wheel_pose",
            },
            {
                "capsule_id": "gear_train",
                "observable_id": "output_wheel_contact",
            },
        ]

        binding = bind_acceptance_contract(
            compile_acceptance_contract(requirements),
            task_mapping=mapping,
            capsules=capsules,
            assembly=assembly,
        )

        self.assertEqual(
            binding["bindings"][0]["target"]["observable_id"],
            "output_wheel_contact",
        )
        self.assertEqual(
            binding["bindings"][0]["selection_evidence"][
                "frozen_observable_quantity"
            ],
            "contact_event",
        )

    def test_frozen_quantity_fails_when_package_omits_it(self) -> None:
        requirements = _requirements()
        criterion = requirements["tasks"][0]["acceptance_requirements"][0]
        criterion["observable_quantity"] = "pose_6d"
        mapping, capsules, assembly = _package()
        for observable in capsules["gear_train"]["observables"]:
            observable["quantity"] = "position_3d"

        with self.assertRaises(SemanticBindingError) as caught:
            bind_acceptance_contract(
                compile_acceptance_contract(requirements),
                task_mapping=mapping,
                capsules=capsules,
                assembly=assembly,
            )

        self.assertEqual(caught.exception.code, "SEMANTIC_BINDING_MISSING")
        self.assertEqual(caught.exception.evidence["required_quantity"], "pose_6d")

    def test_indirect_task_rejects_direct_output_actuation(self) -> None:
        contract = compile_acceptance_contract(_requirements())
        mapping, capsules, assembly = _package()
        capsules["gear_train"]["input_ports"][0][
            "rigid_group_id"
        ] = "group_output_wheel"

        with self.assertRaises(SemanticBindingError) as caught:
            bind_acceptance_contract(
                contract,
                task_mapping=mapping,
                capsules=capsules,
                assembly=assembly,
            )

        self.assertEqual(
            caught.exception.code,
            "DIRECT_OUTPUT_ACTUATION_FORBIDDEN",
        )

    def test_ratio_metric_binds_distinct_reference_observable(self) -> None:
        requirements = _requirements()
        criterion = requirements["tasks"][0]["acceptance_requirements"][0]
        criterion["target_role"] = "output wheel"
        criterion["reference_role"] = "input crank"
        criterion["metric_id"] = "angular_ratio"
        criterion["target"] = {
            "comparator": "within",
            "value": 2.0,
            "tolerance": 0.02,
            "unit": "ratio",
        }
        criterion["allow_direct_output_actuation"] = True
        mapping, capsules, assembly = _package()
        capsules["gear_train"]["observables"].append(
            {
                "observable_id": "input_crank_pose",
                "rigid_group_id": "group_input_crank",
                "frame_id": "frame_input",
                "quantity": "pose_6d",
            }
        )
        mapping["task_mappings"][0]["observables"].append(
            {
                "capsule_id": "gear_train",
                "observable_id": "input_crank_pose",
            }
        )
        contract = compile_acceptance_contract(requirements)
        binding = bind_acceptance_contract(
            contract,
            task_mapping=mapping,
            capsules=capsules,
            assembly=assembly,
        )

        self.assertEqual(
            binding["bindings"][0]["reference"]["observable_id"],
            "input_crank_pose",
        )
        self.assertNotEqual(
            binding["bindings"][0]["reference"]["observable_id"],
            binding["bindings"][0]["target"]["observable_id"],
        )


if __name__ == "__main__":
    unittest.main()
