from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.collision_policy import (
    ANALYTICAL_TRANSMISSION,
    COMPLIANT_INTERFACE_PHYSICAL,
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
    legacy_interface_declarations,
    validate_interface_declarations,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface import (
    conditional_transmission_gate,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
    RELATIVE_ANCHOR_DRIFT,
    bounded_support_gate,
)


class CollisionPolicyTests(unittest.TestCase):
    @staticmethod
    def _support_gate() -> dict:
        return bounded_support_gate(
            body_a_id="ground",
            body_b_id="shaft",
            endpoint_a_component_refs=("ground/solid_1",),
            endpoint_b_component_refs=("shaft/solid_2",),
            maximum_drift_m=0.001,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=0.05,
            template_id="finite_pin_test_v1",
            proof_source="unit_test",
        )

    @staticmethod
    def _analytical_gate() -> dict:
        return conditional_transmission_gate(
            geometry_plugin_id="unit_test_gate_v1",
            body_a_id="gear-a",
            body_b_id="gear-b",
            endpoint_a_component_refs=("gear-a/solid_1",),
            endpoint_b_component_refs=("gear-b/solid_2",),
            endpoint_a_local_collision_component_refs=(
                "gear-a/solid_root",
                "gear-a/solid_1",
            ),
            endpoint_b_local_collision_component_refs=(
                "gear-b/solid_root",
                "gear-b/solid_2",
            ),
            activation_coordinate="face_overlap_m",
            enter_threshold_si=0.001,
            release_threshold_si=0.0005,
            proof_source="unit_test",
        )

    def test_new_compliant_interface_keeps_body_pair_collision_enabled(self) -> None:
        declaration = interface_declaration(
            interface_id="bearing-a",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id="ground",
            body_b_id="shaft",
            model="finite_translational_rotational_bushing",
            source="validator_runtime",
            free_coordinates=("rotation_about_axis",),
            transmission_gate=self._support_gate(),
        )
        disclosure = collision_policy_disclosure((declaration,), branch="physical")
        self.assertTrue(declaration["whole_body_pair_collision_enabled"])
        self.assertTrue(declaration["full_step_geometry_legality_required"])
        self.assertFalse(disclosure["ai_authored_collision_masks_allowed"])

    def test_validator_owned_component_exception_is_narrow_and_disclosed(self) -> None:
        declaration = interface_declaration(
            interface_id="rubber-glass",
            classification=CONTACT_PHYSICAL,
            body_a_id="ground",
            body_b_id="wiper",
            model="validator_compliant_surface_drag_v1",
            source="validator_runtime",
            parameters={
                "validator_owned_suppressed_component_refs": [
                    "wiper/solid_rubber"
                ]
            },
        )
        disclosure = collision_policy_disclosure(
            (declaration,),
            branch="physical",
        )
        self.assertTrue(declaration["whole_body_pair_collision_enabled"])
        self.assertEqual(
            disclosure["validator_owned_local_exception_count"],
            1,
        )
        self.assertEqual(
            disclosure["validator_owned_local_exceptions"][0][
                "component_refs"
            ],
            ["wiper/solid_rubber"],
        )

    def test_compliant_bearing_cannot_request_component_suppression(self) -> None:
        with self.assertRaises(MechanismValidationError):
            interface_declaration(
                interface_id="bearing-a",
                classification=COMPLIANT_INTERFACE_PHYSICAL,
                body_a_id="ground",
                body_b_id="shaft",
                model="finite_bushing",
                source="validator_runtime",
                parameters={
                    "validator_owned_suppressed_component_refs": [
                        "shaft/solid_1"
                    ]
                },
            )

    def test_analytical_interface_gets_only_gate_bound_local_exception(
        self,
    ) -> None:
        declaration = interface_declaration(
            interface_id="gear-a-b",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id="gear-a",
            body_b_id="gear-b",
            model="finite_mesh_torque",
            source="validator_runtime",
            transmission_gate=self._analytical_gate(),
        )
        disclosure = collision_policy_disclosure(
            (declaration,),
            branch="physical",
        )
        exception = declaration["local_collision_exception"]
        self.assertEqual(
            exception["endpoint_a_component_refs"],
            ["gear-a/solid_1", "gear-a/solid_root"],
        )
        self.assertEqual(
            exception["endpoint_b_component_refs"],
            ["gear-b/solid_2", "gear-b/solid_root"],
        )
        self.assertTrue(
            exception["full_step_saved_pose_audit_required"]
        )
        self.assertEqual(
            disclosure["validator_owned_local_exception_count"],
            1,
        )

        declaration["local_collision_exception"][
            "endpoint_a_component_refs"
        ] = ["gear-a/solid_other"]
        with self.assertRaises(MechanismValidationError) as caught:
            validate_interface_declarations((declaration,))
        self.assertEqual(
            caught.exception.code,
            "PHYSICAL_LOCAL_COLLISION_EXCEPTION_INVALID",
        )

    def test_analytical_interface_can_keep_all_collision_components(
        self,
    ) -> None:
        gate = conditional_transmission_gate(
            geometry_plugin_id="unit_test_gate_v1",
            body_a_id="gear-a",
            body_b_id="gear-b",
            endpoint_a_component_refs=("gear-a/solid_1",),
            endpoint_b_component_refs=("gear-b/solid_2",),
            endpoint_a_local_collision_component_refs=(),
            endpoint_b_local_collision_component_refs=(),
            activation_coordinate="geometry_margin_m",
            enter_threshold_si=0.0,
            release_threshold_si=-1.0e-6,
            proof_source="unit_test",
        )
        declaration = interface_declaration(
            interface_id="gear-a-b",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id="gear-a",
            body_b_id="gear-b",
            model="finite_compliant_effort_pair",
            source="validator_runtime",
            transmission_gate=gate,
        )
        disclosure = collision_policy_disclosure(
            (declaration,),
            branch="physical",
        )
        self.assertIsNone(declaration["local_collision_exception"])
        self.assertEqual(
            disclosure["validator_owned_local_exception_count"],
            0,
        )

    def test_new_whole_pair_collision_exclusion_is_forbidden(self) -> None:
        declaration = interface_declaration(
            interface_id="bearing-a",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id="ground",
            body_b_id="shaft",
            model="finite_bushing",
            source="validator_runtime",
            transmission_gate=self._support_gate(),
        )
        declaration["historical_runtime_collision_excluded"] = True
        declaration["whole_body_pair_collision_enabled"] = False
        with self.assertRaises(MechanismValidationError) as caught:
            validate_interface_declarations((declaration,))
        self.assertEqual(
            caught.exception.code,
            "PHYSICAL_BODY_PAIR_COLLISION_EXCLUSION_FORBIDDEN",
        )

    def test_new_compliant_support_requires_finite_action_gate(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            interface_declaration(
                interface_id="bearing-a",
                classification=COMPLIANT_INTERFACE_PHYSICAL,
                body_a_id="ground",
                body_b_id="shaft",
                model="finite_bushing",
                source="validator_runtime",
            )
        self.assertEqual(
            caught.exception.code,
            "COMPLIANT_SUPPORT_GEOMETRY_GATE_REQUIRED",
        )

    def test_legacy_exclusion_becomes_explicit_migration_declaration(self) -> None:
        trace = {
            "branch_semantics": {
                "excluded_collision_pairs": [
                    {
                        "joint_id": "mesh-a",
                        "body_a_id": "gear-a",
                        "body_b_id": "gear-b",
                        "reason": "finite gear mesh torque owns transmission",
                    }
                ]
            }
        }
        declarations = legacy_interface_declarations(trace)
        self.assertEqual(len(declarations), 1)
        self.assertEqual(declarations[0]["classification"], ANALYTICAL_TRANSMISSION)
        self.assertTrue(declarations[0]["historical_runtime_collision_excluded"])
        self.assertFalse(declarations[0]["whole_body_pair_collision_enabled"])


if __name__ == "__main__":
    unittest.main()
