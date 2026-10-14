from __future__ import annotations

import unittest
from dataclasses import replace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    AdapterSpec,
    adapter_to_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_loads import (
    default_bicycle_chain_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
    BICYCLE_CHAIN_PLUGIN_ID,
    approved_bicycle_chain_family_spec_fixture,
    bicycle_chain_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.certificate import (
    build_certificate,
    build_replacement,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    MechanismPlan,
    MechanismValidationError,
)


def _plan() -> MechanismPlan:
    family_spec = approved_bicycle_chain_family_spec_fixture()
    spec = bicycle_chain_from_dict(family_spec)
    family_spec["collision_geometry"] = {
        "runtime_materialized_extension": True,
    }
    adapter = AdapterSpec(
        adapter_id=BICYCLE_CHAIN_PLUGIN_ID,
        family="compound",
        required_source_hashes=frozenset(),
        supported_programs=frozenset(),
        family_spec=family_spec,
    )
    fixed_groups = tuple(
        group_id
        for body in spec.bodies
        if body.fixed
        for group_id in body.group_ids
    )
    moving_groups = tuple(
        group_id
        for body in spec.bodies
        if not body.fixed
        for group_id in body.group_ids
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="intake_report.json",
        intake_report_sha256="0" * 64,
        package_id="bicycle_certificate_test",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="bicycle_drivetrain",
        capsule_file="bicycle_drivetrain.json",
        family="compound",
        protocol_id="bicycle_chain_sprocket_drive",
        fixed_group_ids=fixed_groups,
        moving_group_ids=moving_groups,
        axis_abs=spec.axis_abs,
        joint_origin_abs_m=spec.input_origin_abs_m,
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id=spec.input_frame_id,
        input_port_id=spec.input_port_id,
        input_frame_id=spec.input_frame_id,
        input_point_abs_m=spec.input_origin_abs_m,
        observable_id=spec.output_observable_id,
        observable_frame_id=spec.output_frame_id,
        observable_point_abs_m=spec.output_origin_abs_m,
        task_ids=(spec.task_id,),
        characteristic_length_m=1.0,
        moving_extents_m=(1.0, 1.0, 1.0),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _passing_comparisons(
    load_cases: tuple[LoadCase, ...],
) -> tuple[dict[str, object], ...]:
    deferred = {
        "status": "DEFERRED",
        "required_environment": "outer_cad_after_queue_barrier",
        "worker_audit_authority": False,
    }
    return tuple(
        {
            "load_case_id": load_case.load_case_id,
            "verdict": "PASS",
            "required_for_certificate": True,
            "failure_reasons": [],
            "saved_pose_geometry_audit": deferred,
            "thresholds": {
                "source": "typed_bicycle_spec_and_frozen_task_contract"
            },
            "physical_metrics": {
                "input_angle_min_rad": -0.2 - index * 0.01,
                "input_angle_max_rad": 2.0 + index * 0.01,
                "output_angle_min_rad": -0.4 - index * 0.02,
                "output_angle_max_rad": 4.0 + index * 0.02,
                "max_support_drift_m": 1.0e-4 + index * 1.0e-6,
                "max_off_axis_rotation_rad": 1.0e-6,
                "max_chain_coordinate_error_m": 4.0e-4,
                "max_chain_differential_tension_n": 51.0,
                "max_chain_axis_angle_error_rad": 4.0e-7,
                "max_chain_axial_alignment_error_m": 2.0e-14,
                "max_support_force_n": 54.0,
                "max_support_torque_nm": 3.8,
                "powered_output_excursion_rad": 3.25,
                "max_powered_ratio_pointwise_error": 0.01,
                "max_powered_ratio_incremental_error": 0.01,
                "final_output_home_error_rad": 0.05,
            },
        }
        for index, load_case in enumerate(load_cases)
    )


def _passing_outer_audits(
    plan: MechanismPlan,
    load_cases: tuple[LoadCase, ...],
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "status": "PHYSICAL_PREFLIGHT_PASSED",
            "passed": True,
            "failure_reasons": [],
            "reduced_branch_authorized": True,
            "report": (
                "results/bicycle_drivetrain/physical_trace_audit/"
                f"{load_case.load_case_id}.json"
            ),
            "report_sha256": f"{index + 1:064x}",
            "report_binding_verified": True,
            "physical_trace_sha256": f"{index + 101:064x}",
            "physical_trace_binding_verified": True,
            "audit_execution_environment": "outer_cad_after_queue_barrier",
            "worker_produced_audit_accepted": False,
            "broad_body_pair_collision_exclusions": [],
            "geometry_legality": {
                "passed": True,
                "sample_count": 48_801,
                "max_penetration_m": 0.0,
            },
            "component_contact_intent": {
                "authoritative": True,
                "passed": True,
                "outcomes": [
                    {
                        "kind": "ANALYTICAL_TRANSMISSION",
                        "status": "ANALYTICAL_TRANSMISSION_DECLARED",
                    }
                ],
                "ambiguous_attribution_event_count": 0,
            },
            "moving_chain_path_audit": {
                "authoritative": True,
                "passed": True,
                "audit_sample_count": 48_801,
                "fixed_spatial_flex_zones_verified": True,
                "saved_state_clearance_verified": True,
                "original_step_components_used": True,
            },
        }
        for index, load_case in enumerate(load_cases)
    )


class BicycleChainCertificateTests(unittest.TestCase):
    def test_certificate_and_replacement_use_bicycle_evidence_schema(
        self,
    ) -> None:
        plan = _plan()
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        load_cases = default_bicycle_chain_load_cases(plan)
        comparisons = _passing_comparisons(load_cases)

        certificate = build_certificate(
            plan,
            load_cases,
            comparisons,
        )

        self.assertEqual(certificate["verdict"], "KINEMATIC_CERTIFIED")
        envelope = certificate["tested_envelope"]
        self.assertEqual(envelope["applied_torque_nm"]["max"], 3.22)
        self.assertEqual(
            envelope["input_crank_angle_rad"]["min"],
            -0.23,
        )
        self.assertEqual(
            envelope["rear_drive_wheel_angle_rad"]["max"],
            4.06,
        )
        self.assertEqual(
            envelope["max_observed_chain_differential_tension_n"],
            51.0,
        )
        model = certificate["compound_mechanism_model"]
        self.assertEqual(
            model["mechanism_kind"],
            "bicycle_chain_sprocket_drive",
        )
        self.assertEqual(
            model["topology_sha256"],
            canonical_sha256(spec.to_dict()),
        )
        self.assertFalse(model["direct_output_actuation"])
        self.assertFalse(model["individual_link_tooth_contact_simulated"])

        replacement = build_replacement(
            plan,
            certificate,
            certificate_sha256="f" * 64,
        )
        topology = replacement["replacement_model"]["compound_topology"]
        self.assertEqual(
            topology["mechanism_kind"],
            "bicycle_chain_sprocket_drive",
        )
        self.assertEqual(
            topology["family_spec_sha256"],
            canonical_sha256(spec.to_dict()),
        )
        self.assertFalse(topology["contact_or_dynamic_response_claimed"])

    def test_outer_physical_audits_are_bound_into_certificate(self) -> None:
        base_plan = _plan()
        plan = replace(
            base_plan,
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "requirements": [],
            },
        )
        load_cases = default_bicycle_chain_load_cases(plan)
        comparisons = _passing_comparisons(load_cases)
        audits = _passing_outer_audits(plan, load_cases)

        certificate = build_certificate(
            plan,
            load_cases,
            comparisons,
            task_acceptance={
                "status": "PASS",
                "certificate_authority": True,
            },
            physical_trace_audits=audits,
        )

        self.assertEqual(certificate["overall_verdict"], "TASK_CERTIFIED")
        bound = certificate["physical_trace_audit_evidence"]
        self.assertEqual(bound["gate_status"], "PASS")
        self.assertEqual(bound["record_count"], len(load_cases))
        self.assertTrue(bound["all_records_passed"])
        self.assertEqual(
            bound["records"][0]["report_sha256"],
            audits[0]["report_sha256"],
        )
        geometry = certificate["physical_geometry_legality"]
        self.assertEqual(
            geometry["evidence_source"], "outer_physical_trace_audit"
        )
        self.assertEqual(
            geometry["deferred_comparison_count"], len(load_cases)
        )
        self.assertTrue(geometry["all_saved_physical_poses_audited"])
        self.assertTrue(geometry["all_audits_passed"])
        contact = certificate["component_contact_evidence"]
        self.assertEqual(
            contact["evidence_source"], "outer_physical_trace_audit"
        )
        self.assertTrue(contact["all_authoritative_audits_passed"])
        chain = certificate["moving_chain_path_evidence"]
        self.assertTrue(chain["all_audits_passed"])
        self.assertTrue(chain["fixed_spatial_flex_zones_verified"])
        self.assertTrue(chain["saved_state_clearance_verified"])
        self.assertTrue(chain["original_step_components_used"])
        self.assertEqual(chain["minimum_audit_sample_count"], 48_801)

    def test_failed_outer_physical_audit_revokes_certificate(self) -> None:
        plan = _plan()
        load_cases = default_bicycle_chain_load_cases(plan)
        comparisons = _passing_comparisons(load_cases)
        audits = list(_passing_outer_audits(plan, load_cases))
        audits[0] = {
            **audits[0],
            "status": "PHYSICAL_PREFLIGHT_FAILED",
            "passed": False,
            "failure_reasons": ["physical_illegal_interpenetration"],
        }

        certificate = build_certificate(
            plan,
            load_cases,
            comparisons,
            physical_trace_audits=tuple(audits),
        )

        self.assertEqual(certificate["comparison_verdict"], "KINEMATIC_CERTIFIED")
        self.assertEqual(certificate["verdict"], "PHYSICAL_FAILURE")
        self.assertFalse(certificate["replacement_eligible"])
        self.assertEqual(
            certificate["physical_trace_audit_evidence"]["gate_status"],
            "FAIL",
        )
        self.assertIn(
            "physical_illegal_interpenetration", certificate["failures"]
        )

    def test_chain_subaudit_cannot_hide_behind_top_level_pass(self) -> None:
        plan = _plan()
        load_cases = default_bicycle_chain_load_cases(plan)
        audits = list(_passing_outer_audits(plan, load_cases))
        audits[0] = {
            **audits[0],
            "moving_chain_path_audit": {
                **audits[0]["moving_chain_path_audit"],
                "fixed_spatial_flex_zones_verified": False,
            },
        }

        certificate = build_certificate(
            plan,
            load_cases,
            _passing_comparisons(load_cases),
            physical_trace_audits=tuple(audits),
        )

        self.assertEqual(
            certificate["comparison_verdict"], "KINEMATIC_CERTIFIED"
        )
        self.assertEqual(certificate["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertFalse(certificate["replacement_eligible"])
        binding = certificate["physical_trace_audit_evidence"]
        self.assertTrue(binding["moving_chain_audit_required"])
        self.assertFalse(binding["records"][0]["certificate_gate_passed"])

    def test_outer_physical_audits_must_cover_every_load(self) -> None:
        plan = _plan()
        load_cases = default_bicycle_chain_load_cases(plan)

        with self.assertRaises(MechanismValidationError) as caught:
            build_certificate(
                plan,
                load_cases,
                _passing_comparisons(load_cases),
                physical_trace_audits=_passing_outer_audits(
                    plan, load_cases
                )[:-1],
            )

        self.assertEqual(
            caught.exception.code,
            "CERTIFICATE_PHYSICAL_AUDIT_EVIDENCE_INCOMPLETE",
        )


if __name__ == "__main__":
    unittest.main()
