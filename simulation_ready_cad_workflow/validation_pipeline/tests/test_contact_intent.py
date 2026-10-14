from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
    RELATIVE_ANCHOR_DRIFT,
    bounded_support_gate,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.collision_policy import (
    ANALYTICAL_TRANSMISSION,
    COMPLIANT_INTERFACE_PHYSICAL,
    CONTACT_PHYSICAL,
    interface_declaration,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface import (
    conditional_transmission_gate,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.contact_intent import (
    COMPLIANT_SUPPORT,
    EXPECTED_REAL_CONTACT,
    FORBIDDEN_CONTACT,
    audit_component_contact_intents,
    build_component_contact_intent_policy,
    collision_component_inventory,
    contact_intent_declaration,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
    _plain_prismatic_interface_declarations,
)


class ComponentContactIntentTests(unittest.TestCase):
    @staticmethod
    def geometry() -> dict:
        return {
            "package_binding_sha256": "package-binding",
            "assembly_sha256": "assembly-hash",
            "interpenetration_tolerance_m": 0.00035,
            "bodies": [
                {
                    "body_id": "frame",
                    "fixed": True,
                    "components": [
                        {"solid_id": "bell", "sha256": "a" * 64},
                        {"solid_id": "bearing", "sha256": "b" * 64},
                    ],
                },
                {
                    "body_id": "hammer",
                    "fixed": False,
                    "components": [
                        {"solid_id": "head", "sha256": "c" * 64},
                        {"solid_id": "shaft", "sha256": "d" * 64},
                    ],
                },
            ],
        }

    @staticmethod
    def event(phase: str = "strike", depth: float = 0.0001) -> dict:
        return {
            "solid_pair": ["frame/bell", "hammer/head"],
            "body_pair": ["frame", "hammer"],
            "phase_id": phase,
            "time_s": 1.0,
            "penetration_depth_m": depth,
        }

    def policy(self, declaration: dict) -> dict:
        return build_component_contact_intent_policy(
            (),
            self.geometry(),
            phase_ids=("strike", "return"),
            explicit_declarations=(declaration,),
        )

    def test_convex_components_have_unique_refs_with_shared_source_solid(
        self,
    ) -> None:
        geometry = self.geometry()
        geometry["bodies"][1]["components"] = [
            {
                "component_id": "head.cap_a",
                "solid_id": "solid_2",
                "sha256": "c" * 64,
            },
            {
                "component_id": "head.cap_b",
                "solid_id": "solid_2",
                "sha256": "d" * 64,
            },
        ]

        inventory = collision_component_inventory(geometry)

        hammer = [row for row in inventory if row["body_id"] == "hammer"]
        self.assertEqual(
            {row["component_ref"] for row in hammer},
            {"hammer/head.cap_a", "hammer/head.cap_b"},
        )
        self.assertEqual(
            {row["source_solid_id"] for row in hammer},
            {"solid_2"},
        )

        geometry["bodies"][1]["components"][1]["component_id"] = (
            "head.cap_a"
        )
        with self.assertRaises(MechanismValidationError) as captured:
            collision_component_inventory(geometry)
        self.assertEqual(
            captured.exception.code,
            "COMPONENT_CONTACT_COMPONENT_DUPLICATE",
        )

    def test_expected_contact_passes_only_with_bound_event_in_active_phase(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-strike",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            active_phase_ids=("strike",),
            max_penetration_m=0.00035,
            source="test_contract",
        )
        result = audit_component_contact_intents(
            self.policy(declaration),
            {"contact_events": [self.event()]},
            collision_geometry=self.geometry(),
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["outcomes"][0]["status"],
            "EXPECTED_CONTACT_OBSERVED",
        )

    def test_missing_expected_contact_fails(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-strike",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            active_phase_ids=("strike",),
            source="test_contract",
        )
        result = audit_component_contact_intents(
            self.policy(declaration),
            {"contact_events": []},
            collision_geometry=self.geometry(),
        )
        self.assertFalse(result["passed"])
        self.assertIn("expected_component_contact_missing", result["failure_reasons"])

    def test_plain_prismatic_finite_guide_does_not_require_contact(self) -> None:
        geometry = self.geometry()
        interfaces = _plain_prismatic_interface_declarations(
            collision_geometry=geometry,
            finite_support_required=True,
            maximum_transverse_drift_m=0.0012,
            maximum_axis_misalignment_rad=0.052,
            body_a_id="frame",
            body_b_id="hammer",
        )
        policy = build_component_contact_intent_policy(
            interfaces,
            geometry,
            phase_ids=("loaded",),
        )

        result = audit_component_contact_intents(
            policy,
            {"contact_events": []},
            collision_geometry=geometry,
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["outcomes"][0]["kind"], COMPLIANT_SUPPORT)

    def test_plain_prismatic_captive_guide_has_a_finite_retention_envelope(self) -> None:
        geometry = self.geometry()
        interfaces = _plain_prismatic_interface_declarations(
            collision_geometry=geometry,
            finite_support_required=True,
            maximum_transverse_drift_m=0.05,
            maximum_axis_misalignment_rad=1.0,
            body_a_id="frame",
            body_b_id="hammer",
        )

        gate = interfaces[0]["transmission_gate"]
        self.assertEqual(
            gate["geometry_parameters"]["maximum_drift_m"],
            0.05,
        )
        self.assertEqual(
            gate["geometry_parameters"]["maximum_axis_misalignment_rad"],
            1.0,
        )
        policy = build_component_contact_intent_policy(
            interfaces,
            geometry,
            phase_ids=("loaded",),
        )
        result = audit_component_contact_intents(
            policy,
            {"contact_events": []},
            collision_geometry=geometry,
        )
        self.assertTrue(result["passed"])

    def test_contact_in_wrong_phase_fails(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-strike",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            active_phase_ids=("strike",),
            source="test_contract",
        )
        result = audit_component_contact_intents(
            self.policy(declaration),
            {"contact_events": [self.event("return")]},
            collision_geometry=self.geometry(),
        )
        self.assertIn(
            "component_contact_outside_declared_phase",
            result["failure_reasons"],
        )

    def test_forbidden_contact_is_checked_at_saved_pose(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-must-clear-hammer",
            kind=FORBIDDEN_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            source="test_contract",
        )
        result = audit_component_contact_intents(
            self.policy(declaration),
            {"contact_events": [self.event()]},
            collision_geometry=self.geometry(),
        )
        self.assertIn("forbidden_component_contact_observed", result["failure_reasons"])

    def test_support_contact_cannot_be_relabelled_as_work_contact(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="hammer-bearing",
            kind=COMPLIANT_SUPPORT,
            component_a_refs=("frame/bearing",),
            component_b_refs=("hammer/shaft",),
            source="test_contract",
        )
        policy = self.policy(declaration)
        result = audit_component_contact_intents(
            policy,
            {
                "contact_events": [
                    {
                        **self.event(),
                        "solid_pair": ["frame/bearing", "hammer/shaft"],
                    }
                ]
            },
            collision_geometry=self.geometry(),
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["outcomes"][0]["counts_as_expected_contact"])

    def test_excessive_penetration_fails_even_when_expected_contact_exists(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-strike",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            active_phase_ids=("strike",),
            max_penetration_m=0.0002,
            source="test_contract",
        )
        result = audit_component_contact_intents(
            self.policy(declaration),
            {"contact_events": [self.event(depth=0.0005)]},
            collision_geometry=self.geometry(),
        )
        self.assertIn(
            "component_contact_penetration_exceeds_limit",
            result["failure_reasons"],
        )

    def test_ambiguous_body_only_event_cannot_prove_expected_contact(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-strike",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            active_phase_ids=("strike",),
            source="test_contract",
        )
        result = audit_component_contact_intents(
            self.policy(declaration),
            {"contact_events": [{"body_pair": ["frame", "hammer"], "phase_id": "strike"}]},
            collision_geometry=self.geometry(),
        )
        self.assertIn("component_contact_attribution_uncertain", result["failure_reasons"])
        self.assertIn("expected_component_contact_missing", result["failure_reasons"])

    def test_automatic_binder_uses_exact_owned_component_subsets(self) -> None:
        interface = interface_declaration(
            interface_id="guide",
            classification=CONTACT_PHYSICAL,
            body_a_id="frame",
            body_b_id="hammer",
            model="step_contact",
            source="trusted_runtime",
        )
        policy = build_component_contact_intent_policy(
            (interface,),
            self.geometry(),
            phase_ids=("strike",),
        )
        declaration = policy["declarations"][0]
        self.assertEqual(declaration["kind"], EXPECTED_REAL_CONTACT)
        self.assertEqual(
            declaration["component_a_refs"],
            ["frame/bearing", "frame/bell"],
        )
        self.assertFalse(declaration["allows_whole_body_pair_collision_exclusion"])

    def test_explicit_contact_binder_preserves_component_and_gap_contract(
        self,
    ) -> None:
        interface = interface_declaration(
            interface_id="guide",
            classification=CONTACT_PHYSICAL,
            body_a_id="frame",
            body_b_id="hammer",
            model="step_contact",
            source="trusted_runtime",
            parameters={
                "contact_component_a_refs": ["frame/bell"],
                "contact_component_b_refs": ["hammer/head"],
                "contact_evidence_max_separation_m": 20.0e-6,
            },
        )
        policy = build_component_contact_intent_policy(
            (interface,),
            self.geometry(),
            phase_ids=("strike",),
        )
        declaration = policy["declarations"][0]

        self.assertEqual(declaration["component_a_refs"], ["frame/bell"])
        self.assertEqual(declaration["component_b_refs"], ["hammer/head"])
        self.assertEqual(declaration["max_separation_m"], 20.0e-6)
        self.assertTrue(
            declaration["requires_same_sample_runtime_body_pair_contact"]
        )

    def test_support_interface_binds_but_never_claims_expected_contact(self) -> None:
        gate = bounded_support_gate(
            body_a_id="frame",
            body_b_id="hammer",
            endpoint_a_component_refs=("frame/bearing",),
            endpoint_b_component_refs=("hammer/head",),
            maximum_drift_m=0.001,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            maximum_axis_misalignment_rad=0.05,
            template_id="finite_pin_test_v1",
            proof_source="unit_test",
        )
        interface = interface_declaration(
            interface_id="bearing",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id="frame",
            body_b_id="hammer",
            model="finite_bushing",
            source="trusted_runtime",
            transmission_gate=gate,
        )
        policy = build_component_contact_intent_policy(
            (interface,),
            self.geometry(),
            phase_ids=("strike",),
        )
        self.assertEqual(policy["declarations"][0]["kind"], COMPLIANT_SUPPORT)
        self.assertFalse(policy["declarations"][0]["counts_as_expected_contact"])

    def test_analytical_interface_without_suppression_binds_owned_components(
        self,
    ) -> None:
        gate = conditional_transmission_gate(
            geometry_plugin_id="unit_test_chain_geometry_v1",
            body_a_id="frame",
            body_b_id="hammer",
            endpoint_a_component_refs=("frame/bearing",),
            endpoint_b_component_refs=("hammer/head",),
            endpoint_a_local_collision_component_refs=(),
            endpoint_b_local_collision_component_refs=(),
            activation_coordinate="geometry_margin",
            enter_threshold_si=0.0,
            release_threshold_si=-1.0e-6,
            proof_source="unit_test",
        )
        interface = interface_declaration(
            interface_id="analytical-chain",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id="frame",
            body_b_id="hammer",
            model="finite_compliant_effort_pair",
            source="trusted_runtime",
            transmission_gate=gate,
        )
        policy = build_component_contact_intent_policy(
            (interface,),
            self.geometry(),
            phase_ids=("strike",),
        )
        declaration = policy["declarations"][0]
        self.assertEqual(declaration["kind"], ANALYTICAL_TRANSMISSION)
        self.assertEqual(
            declaration["component_a_refs"],
            ["frame/bearing", "frame/bell"],
        )
        self.assertEqual(
            declaration["component_b_refs"],
            ["hammer/head", "hammer/shaft"],
        )
        self.assertEqual(
            declaration["binding_scope"],
            "body_owned_exact_component_subset_no_suppression",
        )
        self.assertFalse(declaration["counts_as_expected_contact"])

    def test_asymmetric_suppression_is_labeled_partial(self) -> None:
        gate = conditional_transmission_gate(
            geometry_plugin_id="unit_test_chain_geometry_v1",
            body_a_id="frame",
            body_b_id="hammer",
            endpoint_a_component_refs=("frame/bearing",),
            endpoint_b_component_refs=("hammer/head",),
            endpoint_a_local_collision_component_refs=("frame/bearing",),
            endpoint_b_local_collision_component_refs=(),
            activation_coordinate="geometry_margin",
            enter_threshold_si=0.0,
            release_threshold_si=-1.0e-6,
            proof_source="unit_test",
        )
        interface = interface_declaration(
            interface_id="analytical-chain",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id="frame",
            body_b_id="hammer",
            model="finite_compliant_effort_pair",
            source="trusted_runtime",
            transmission_gate=gate,
        )

        declaration = build_component_contact_intent_policy(
            (interface,),
            self.geometry(),
            phase_ids=("strike",),
        )["declarations"][0]

        self.assertEqual(
            declaration["binding_scope"],
            "validator_step_bound_partial_collision_suppression",
        )
        self.assertEqual(
            declaration["component_a_refs"],
            ["frame/bearing"],
        )
        self.assertEqual(
            declaration["component_b_refs"],
            ["hammer/head", "hammer/shaft"],
        )

    def test_component_inventory_hash_change_invalidates_policy(self) -> None:
        declaration = contact_intent_declaration(
            intent_id="bell-strike",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("frame/bell",),
            component_b_refs=("hammer/head",),
            active_phase_ids=("strike",),
            source="test_contract",
        )
        policy = self.policy(declaration)
        changed = self.geometry()
        changed["bodies"][0]["components"][0]["sha256"] = "e" * 64
        with self.assertRaises(MechanismValidationError):
            audit_component_contact_intents(
                policy,
                {"contact_events": [self.event()]},
                collision_geometry=changed,
            )


if __name__ == "__main__":
    unittest.main()
