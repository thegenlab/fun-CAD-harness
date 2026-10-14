from __future__ import annotations

import math
import importlib.util
import unittest
from copy import deepcopy

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
    RELATIVE_ANCHOR_DRIFT,
    bounded_support_gate,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.collision_policy import (
    ANALYTICAL_TRANSMISSION,
    COMPLIANT_INTERFACE_PHYSICAL,
    interface_declaration,
    validate_interface_declarations,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface import (
    APPROACHING,
    ENGAGED,
    INACTIVE,
    RELEASING,
    AxialClutchGeometry,
    AxialClutchPose,
    AxialGearPose,
    AxialSlidingSpurGearGeometry,
    ConditionalCompliantInterface,
    CoplanarPulleyTransferGeometry,
    InterfaceGeometryObservation,
    RadialReachGeometry,
    RadialReachPose,
    RackPinionGeometry,
    RackPinionPose,
    RightAngleBevelGearGeometry,
    assess_initial_functional_geometry,
    audit_conditional_interface_runtime_evidence,
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
    validate_functional_interface_geometry_binding,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface_testcase import (
    run_pychrono_sliding_gear_engagement_testcase,
    run_sliding_gear_engagement_testcase,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


def _gate() -> dict:
    return conditional_transmission_gate(
        geometry_plugin_id="generic_overlap_gate_v1",
        body_a_id="body_a",
        body_b_id="body_b",
        endpoint_a_component_refs=("body_a/solid_1",),
        endpoint_b_component_refs=("body_b/solid_2",),
        activation_coordinate="overlap_m",
        enter_threshold_si=0.0010,
        release_threshold_si=0.0005,
        proof_source="unit_test_geometry",
    )


def _observation(
    time_s: float,
    coordinate: float,
    *,
    rate: float = 0.0,
    constraints: bool = True,
    can_engage_later: bool = True,
) -> InterfaceGeometryObservation:
    return InterfaceGeometryObservation(
        time_s=time_s,
        activation_coordinate_si=coordinate,
        activation_rate_si_s=rate,
        constraints_satisfied=constraints,
        can_engage_later=can_engage_later,
        blockers=() if constraints else ("ALIGNMENT_INVALID",),
    )


def _collision_geometry() -> dict:
    return {
        "package_binding_sha256": "a" * 64,
        "assembly_sha256": "b" * 64,
        "bodies": [
            {
                "body_id": "body_a",
                "fixed": False,
                "components": [
                    {"solid_id": "solid_1", "sha256": "c" * 64}
                ],
            },
            {
                "body_id": "body_b",
                "fixed": False,
                "components": [
                    {"solid_id": "solid_2", "sha256": "d" * 64}
                ],
            },
        ],
    }


class ConditionalInterfaceTests(unittest.TestCase):
    def test_inactive_and_approaching_states_force_exact_zero(self) -> None:
        interface = ConditionalCompliantInterface(_gate())
        inactive = interface.step(_observation(0.0, -0.003))
        approaching = interface.step(
            _observation(0.1, 0.0008, rate=0.01)
        )

        self.assertEqual(inactive.state, INACTIVE)
        self.assertEqual(approaching.state, APPROACHING)
        self.assertEqual(inactive.transmit_scalar(125.0), 0.0)
        self.assertEqual(approaching.transmit_scalar(-125.0), 0.0)

    def test_late_engagement_hysteresis_and_release(self) -> None:
        interface = ConditionalCompliantInterface(_gate())
        decisions = [
            interface.step(_observation(0.0, 0.0)),
            interface.step(_observation(0.1, 0.0008, rate=0.01)),
            interface.step(_observation(0.2, 0.0010, rate=0.01)),
            interface.step(_observation(0.3, 0.0007, rate=-0.01)),
            interface.step(_observation(0.4, 0.0005, rate=-0.01)),
            interface.step(_observation(0.5, 0.0004, rate=-0.01)),
        ]

        self.assertEqual(
            [item.state for item in decisions],
            [INACTIVE, APPROACHING, ENGAGED, ENGAGED, RELEASING, INACTIVE],
        )
        self.assertTrue(decisions[2].capture_reference)
        self.assertFalse(decisions[3].capture_reference)
        self.assertEqual(decisions[2].transmit_scalar(2.5), 2.5)
        self.assertEqual(decisions[4].transmit_scalar(2.5), 0.0)

    def test_lost_geometry_stops_force_in_same_step(self) -> None:
        interface = ConditionalCompliantInterface(_gate())
        interface.step(_observation(0.0, 0.002))
        lost = interface.step(
            _observation(
                0.1,
                0.002,
                constraints=False,
                can_engage_later=False,
            )
        )

        self.assertEqual(lost.state, RELEASING)
        self.assertFalse(lost.transmission_active)
        self.assertEqual(lost.transmit_scalar(999.0), 0.0)

    def test_initial_non_contact_is_warning_not_error(self) -> None:
        assessment = assess_initial_functional_geometry(
            _gate(),
            _observation(0.0, -0.002, can_engage_later=True),
        )

        self.assertEqual(assessment["status"], "WARNING")
        self.assertFalse(assessment["hard_error"])
        self.assertEqual(
            assessment["code"],
            "FUNCTIONAL_INTERFACE_INITIALLY_INACTIVE",
        )

    def test_malformed_or_misbound_declaration_is_hard_error(self) -> None:
        with self.assertRaises(MechanismValidationError) as missing_binding:
            conditional_transmission_gate(
                geometry_plugin_id="generic_gate_v1",
                body_a_id="body_a",
                body_b_id="body_b",
                endpoint_a_component_refs=(),
                endpoint_b_component_refs=("body_b/solid_2",),
                activation_coordinate="overlap_m",
                enter_threshold_si=0.001,
                release_threshold_si=0.0,
                proof_source="test",
            )
        self.assertEqual(
            missing_binding.exception.code,
            "CONDITIONAL_INTERFACE_COMPONENT_BINDING_REQUIRED",
        )

        declaration = interface_declaration(
            interface_id="conditional",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id="body_a",
            body_b_id="body_b",
            model="finite_penalty",
            source="test",
            transmission_gate=_gate(),
        )
        declaration["transmission_gate"]["endpoint_a"]["body_id"] = "body_b"
        with self.assertRaises(MechanismValidationError) as mismatch:
            validate_interface_declarations((declaration,))
        self.assertEqual(
            mismatch.exception.code,
            "CONDITIONAL_INTERFACE_ENDPOINT_BODY_MISMATCH",
        )

    def test_step_component_hash_change_invalidates_geometry_binding(self) -> None:
        declaration = interface_declaration(
            interface_id="conditional",
            classification=ANALYTICAL_TRANSMISSION,
            body_a_id="body_a",
            body_b_id="body_b",
            model="finite_penalty",
            source="test",
            transmission_gate=_gate(),
        )
        geometry = _collision_geometry()
        binding = build_functional_interface_geometry_binding(
            (declaration,), geometry
        )
        validate_functional_interface_geometry_binding(
            binding,
            interfaces=(declaration,),
            collision_geometry=geometry,
        )

        changed = deepcopy(geometry)
        changed["bodies"][0]["components"][0]["sha256"] = "e" * 64
        with self.assertRaises(MechanismValidationError) as caught:
            validate_functional_interface_geometry_binding(
                binding,
                interfaces=(declaration,),
                collision_geometry=changed,
            )
        self.assertEqual(
            caught.exception.code,
            "CONDITIONAL_INTERFACE_STEP_BINDING_CHANGED",
        )

    def test_axial_gear_geometry_detects_late_face_overlap(self) -> None:
        plugin = AxialSlidingSpurGearGeometry(
            driver_pitch_radius_m=0.02,
            driven_pitch_radius_m=0.03,
            radial_tolerance_m=0.0002,
            axis_parallel_tolerance_rad=math.radians(0.25),
        )
        driven = AxialGearPose(
            center_abs_m=(0.0, 0.0, 0.0),
            axis_abs=(0.0, 0.0, 1.0),
            face_width_m=0.008,
        )
        separated = plugin.evaluate(
            time_s=0.0,
            driver=AxialGearPose(
                center_abs_m=(0.05, 0.0, 0.018),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
                linear_velocity_abs_m_s=(0.0, 0.0, -0.03),
            ),
            driven=driven,
        )
        aligned = plugin.evaluate(
            time_s=1.0,
            driver=AxialGearPose(
                center_abs_m=(0.05, 0.0, 0.0),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
            driven=driven,
        )

        self.assertTrue(separated.constraints_satisfied)
        self.assertLess(separated.activation_coordinate_si, 0.0)
        self.assertGreater(separated.activation_rate_si_s, 0.0)
        self.assertGreater(aligned.activation_coordinate_si, 0.0)

    def test_axial_gear_rejects_air_gap_in_pitch_plane(self) -> None:
        plugin = AxialSlidingSpurGearGeometry(
            driver_pitch_radius_m=0.02,
            driven_pitch_radius_m=0.03,
            radial_tolerance_m=0.0002,
            axis_parallel_tolerance_rad=math.radians(0.25),
        )
        observation = plugin.evaluate(
            time_s=0.0,
            driver=AxialGearPose(
                center_abs_m=(0.06, 0.0, 0.0),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
            driven=AxialGearPose(
                center_abs_m=(0.0, 0.0, 0.0),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
        )

        self.assertFalse(observation.constraints_satisfied)

    def test_pulley_gate_checks_coplanarity_without_requiring_contact(self) -> None:
        plugin = CoplanarPulleyTransferGeometry(
            required_center_distance_m=0.060,
            center_distance_tolerance_m=0.0005,
            axis_parallel_tolerance_rad=math.radians(0.25),
        )
        observation = plugin.evaluate(
            time_s=0.0,
            driver=AxialGearPose(
                center_abs_m=(0.0, 0.0, 0.0),
                axis_abs=(0.0, 1.0, 0.0),
                face_width_m=0.004,
            ),
            driven=AxialGearPose(
                center_abs_m=(0.060, 0.001, 0.0),
                axis_abs=(0.0, 1.0, 0.0),
                face_width_m=0.004,
            ),
        )

        self.assertTrue(observation.constraints_satisfied)
        self.assertAlmostEqual(
            observation.activation_coordinate_si,
            0.003,
        )

    def test_right_angle_gear_uses_pitch_gap_and_contact_side(self) -> None:
        plugin = RightAngleBevelGearGeometry(
            driver_pitch_radius_m=0.045,
            driven_pitch_radius_m=0.015,
            pitch_surface_tolerance_m=0.001,
            required_shaft_angle_rad=0.5 * math.pi,
            shaft_angle_tolerance_rad=0.02,
            driver_face_width_m=0.010,
            driven_face_width_m=0.008,
        )
        driver = AxialGearPose(
            center_abs_m=(0.0, -0.004, 0.150),
            axis_abs=(0.0, 1.0, 0.0),
            face_width_m=0.010,
        )
        contact_height = 0.150 - math.sqrt(
            0.045 * 0.045 - 0.024 * 0.024
        )
        front_face = plugin.evaluate(
            time_s=0.0,
            driver=driver,
            driven=AxialGearPose(
                center_abs_m=(-0.024, 0.011, contact_height),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
        )
        rear_face = plugin.evaluate(
            time_s=0.0,
            driver=driver,
            driven=AxialGearPose(
                center_abs_m=(0.024, -0.019, contact_height),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
        )

        self.assertTrue(front_face.constraints_satisfied)
        self.assertTrue(rear_face.constraints_satisfied)
        self.assertLess(front_face.activation_coordinate_si, 1.0e-8)
        self.assertLess(rear_face.activation_coordinate_si, 1.0e-8)
        self.assertEqual(
            front_face.metrics["mesh_direction_sign"],
            -1.0,
        )
        self.assertEqual(
            rear_face.metrics["mesh_direction_sign"],
            1.0,
        )

        gate = conditional_transmission_gate(
            geometry_plugin_id="right_angle_bevel_gear_mesh_v1",
            body_a_id="driver",
            body_b_id="driven",
            endpoint_a_component_refs=("driver/solid_1",),
            endpoint_b_component_refs=("driven/solid_2",),
            activation_coordinate="pitch_surface_gap_m",
            activation_direction="AT_OR_BELOW",
            enter_threshold_si=0.001,
            release_threshold_si=0.0015,
            proof_source="unit_test_geometry",
        )
        interface = ConditionalCompliantInterface(gate)
        engaged = interface.step(front_face)
        separated = plugin.evaluate(
            time_s=0.1,
            driver=driver,
            driven=AxialGearPose(
                center_abs_m=(
                    -0.024,
                    0.011,
                    contact_height - 0.008,
                ),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
        )
        released = interface.step(separated)
        self.assertTrue(engaged.transmission_active)
        self.assertFalse(released.transmission_active)
        self.assertEqual(released.transmit_scalar(10.0), 0.0)

    def test_right_angle_gear_rejects_same_face_air_gap(self) -> None:
        plugin = RightAngleBevelGearGeometry(
            driver_pitch_radius_m=0.045,
            driven_pitch_radius_m=0.015,
            pitch_surface_tolerance_m=0.001,
            required_shaft_angle_rad=0.5 * math.pi,
            shaft_angle_tolerance_rad=0.02,
            driver_face_width_m=0.010,
            driven_face_width_m=0.008,
        )
        driver = AxialGearPose(
            center_abs_m=(0.0, -0.004, 0.150),
            axis_abs=(0.0, 1.0, 0.0),
            face_width_m=0.010,
        )
        left = plugin.evaluate(
            time_s=0.0,
            driver=driver,
            driven=AxialGearPose(
                center_abs_m=(-0.024, 0.0, 0.09665),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
        )
        right = plugin.evaluate(
            time_s=0.0,
            driver=driver,
            driven=AxialGearPose(
                center_abs_m=(0.024, 0.0, 0.09665),
                axis_abs=(0.0, 0.0, 1.0),
                face_width_m=0.008,
            ),
        )

        self.assertGreater(left.activation_coordinate_si, 0.009)
        self.assertGreater(right.activation_coordinate_si, 0.009)
        self.assertEqual(left.metrics["mesh_direction_sign"], -1.0)
        self.assertEqual(right.metrics["mesh_direction_sign"], -1.0)

    def test_cam_reach_gap_can_start_inactive_without_hard_error(self) -> None:
        plugin = RadialReachGeometry(
            radial_gap_tolerance_m=0.0015,
            axis_parallel_tolerance_rad=math.radians(0.25),
        )
        observation = plugin.evaluate(
            time_s=0.0,
            driver=RadialReachPose(
                center_abs_m=(0.0, 0.0, 0.0),
                axis_abs=(0.0, 1.0, 0.0),
                radial_reach_m=0.010,
                face_width_m=0.004,
            ),
            follower=RadialReachPose(
                center_abs_m=(0.025, 0.0, 0.0),
                axis_abs=(0.0, 1.0, 0.0),
                radial_reach_m=0.010,
                face_width_m=0.004,
            ),
        )
        assessment = assess_initial_functional_geometry(
            conditional_transmission_gate(
                geometry_plugin_id="radial_cam_follower_reach_v1",
                body_a_id="body_a",
                body_b_id="body_b",
                endpoint_a_component_refs=("body_a/solid_1",),
                endpoint_b_component_refs=("body_b/solid_2",),
                activation_coordinate="surface_gap_m",
                enter_threshold_si=0.0015,
                release_threshold_si=0.003,
                activation_direction="AT_OR_BELOW",
                proof_source="unit_test_geometry",
            ),
            observation,
        )

        self.assertEqual(assessment["status"], "WARNING")
        self.assertFalse(assessment["hard_error"])
        self.assertFalse(assessment["transmission_active"])

    def test_clutch_gate_uses_face_gap_and_coaxiality(self) -> None:
        plugin = AxialClutchGeometry(
            radial_alignment_tolerance_m=0.0005,
            axis_parallel_tolerance_rad=math.radians(0.25),
        )
        observation = plugin.evaluate(
            time_s=0.0,
            input_face=AxialClutchPose(
                face_center_abs_m=(0.0, 0.001, 0.0),
                axis_abs=(0.0, 1.0, 0.0),
                radial_extent_m=0.003,
            ),
            driven_face=AxialClutchPose(
                face_center_abs_m=(0.0, 0.0, 0.0),
                axis_abs=(0.0, 1.0, 0.0),
                radial_extent_m=0.003,
            ),
        )

        self.assertTrue(observation.constraints_satisfied)
        self.assertAlmostEqual(
            observation.activation_coordinate_si,
            0.001,
        )

    def test_rack_pinion_geometry_accepts_aligned_pitch_line(self) -> None:
        plugin = RackPinionGeometry(
            required_pitch_line_offset_m=0.0408,
            pitch_line_offset_tolerance_m=0.001,
            axis_orthogonality_tolerance_rad=math.radians(0.5),
            face_axis_parallel_tolerance_rad=math.radians(0.5),
        )
        observation = plugin.evaluate(
            time_s=0.0,
            pose=RackPinionPose(
                pinion_center_abs_m=(0.39, -0.105, 0.53),
                pinion_axis_abs=(0.0, 1.0, 0.0),
                pinion_face_width_m=0.028,
                rack_pitch_line_point_abs_m=(
                    0.4308,
                    -0.105,
                    0.53,
                ),
                rack_axis_abs=(0.0, 0.0, 1.0),
                rack_face_axis_abs=(0.0, 1.0, 0.0),
                rack_face_width_m=0.028,
            ),
        )

        self.assertTrue(observation.constraints_satisfied)
        self.assertAlmostEqual(
            observation.activation_coordinate_si,
            0.028,
        )
        self.assertAlmostEqual(
            observation.metrics["pitch_line_offset_error_m"],
            0.0,
        )

    def test_rack_pinion_geometry_rejects_lateral_air_gap(self) -> None:
        plugin = RackPinionGeometry(
            required_pitch_line_offset_m=0.0408,
            pitch_line_offset_tolerance_m=0.001,
            axis_orthogonality_tolerance_rad=math.radians(0.5),
            face_axis_parallel_tolerance_rad=math.radians(0.5),
        )
        observation = plugin.evaluate(
            time_s=0.0,
            pose=RackPinionPose(
                pinion_center_abs_m=(0.39, -0.105, 0.53),
                pinion_axis_abs=(0.0, 1.0, 0.0),
                pinion_face_width_m=0.028,
                rack_pitch_line_point_abs_m=(
                    0.4408,
                    -0.105,
                    0.53,
                ),
                rack_axis_abs=(0.0, 0.0, 1.0),
                rack_face_axis_abs=(0.0, 1.0, 0.0),
                rack_face_width_m=0.028,
            ),
        )

        self.assertFalse(observation.constraints_satisfied)
        self.assertIn(
            "PINION_NOT_AT_RACK_PITCH_LINE",
            observation.blockers,
        )
        self.assertFalse(observation.can_engage_later)
        self.assertIn("PITCH_CIRCLES_NOT_TANGENT", observation.blockers)

    def test_sliding_gear_project_transmits_only_after_engagement(self) -> None:
        result = run_sliding_gear_engagement_testcase()

        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["max_inactive_transmitted_torque_nm"], 0.0)
        self.assertGreater(result["post_engagement_driven_span_rad"], 0.01)
        self.assertIn("APPROACHING", result["state_path"])
        self.assertIn("ENGAGED", result["state_path"])
        self.assertTrue(
            result["physical_interface_policy"]
            ["functional_interface_geometry_binding_required"]
        )
        self.assertEqual(
            result["functional_interface_geometry_binding"]
            ["conditional_interface_count"],
            1,
        )
        self.assertTrue(result["conditional_interface_runtime_audit"]["passed"])

    def test_any_inactive_effort_leakage_fails_runtime_audit(self) -> None:
        result = run_sliding_gear_engagement_testcase()
        samples = deepcopy(result["samples"])
        inactive = next(
            sample for sample in samples if not sample["transmission_active"]
        )
        inactive["conditional_interface_states"][0][
            "transmitted_effort_si"
        ] = 1e-15
        audit = audit_conditional_interface_runtime_evidence(
            result["physical_interface_policy"]["interface_declarations"],
            samples,
            result["conditional_interface_runtime"],
        )

        self.assertFalse(audit["passed"])
        self.assertIn(
            "conditional_interface_inactive_effort_nonzero",
            audit["failure_reasons"],
        )

    def test_compliant_support_torque_evidence_is_required_every_sample(
        self,
    ) -> None:
        gate = bounded_support_gate(
            body_a_id="body_a",
            body_b_id="body_b",
            endpoint_a_component_refs=("body_a/solid_1",),
            endpoint_b_component_refs=("body_b/solid_2",),
            maximum_drift_m=0.001,
            coordinate_mode=RELATIVE_ANCHOR_DRIFT,
            proof_source="unit_test",
            template_id="unit_test_support",
        )
        declaration = interface_declaration(
            interface_id="support",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id="body_a",
            body_b_id="body_b",
            model="finite_support",
            source="unit_test",
            free_coordinates=("rotation_about_axis",),
            transmission_gate=gate,
        )
        sample = {
            "conditional_interface_states": [
                {
                    "interface_id": "support",
                    "state": ENGAGED,
                    "transmission_active": True,
                    "transmitted_effort_si": 0.0,
                    "constraints_satisfied": True,
                }
            ]
        }
        summary = {
            "schema_version": "conditional_interface_runtime_v1",
            "evaluated_every_physics_step": True,
            "physics_step_count": 1,
            "interfaces": [
                {
                    "interface_id": "support",
                    "evaluation_count": 1,
                    "state_path": [ENGAGED],
                    "max_inactive_abs_transmitted_effort_si": 0.0,
                }
            ],
        }

        audit = audit_conditional_interface_runtime_evidence(
            (declaration,),
            (sample,),
            summary,
        )

        self.assertFalse(audit["passed"])
        self.assertIn(
            "conditional_interface_support_torque_invalid",
            audit["failure_reasons"],
        )

        sample["conditional_interface_states"][0].update(
            requested_support_torque_nm=0.25,
            transmitted_support_torque_nm=0.25,
        )
        repaired = audit_conditional_interface_runtime_evidence(
            (declaration,),
            (sample,),
            summary,
        )
        self.assertTrue(repaired["passed"], repaired["failure_reasons"])

    @unittest.skipUnless(
        importlib.util.find_spec("pychrono") is not None,
        "PyChrono is tested in its isolated environment",
    )
    def test_live_pychrono_backend_has_no_ideal_links(self) -> None:
        result = run_pychrono_sliding_gear_engagement_testcase()

        self.assertTrue(result["passed"], result["checks"])
        self.assertEqual(result["ideal_link_count"], 0)
        self.assertEqual(result["state_path"], [INACTIVE, APPROACHING, ENGAGED])
        self.assertEqual(result["max_inactive_transmitted_torque_nm"], 0.0)


if __name__ == "__main__":
    unittest.main()
