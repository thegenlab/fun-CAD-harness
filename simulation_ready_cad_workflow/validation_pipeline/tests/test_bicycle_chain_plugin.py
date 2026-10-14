from __future__ import annotations

import math
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_comparison import (
    _behavior_failures,
    _validate_rear_drive_wheel_wrench_inventory,
    _visible_chain_geometry_failures,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
    BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
    BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
    BICYCLE_CHAIN_PLUGIN_ID,
    BICYCLE_CHAIN_SCHEMA_VERSION,
    BICYCLE_CHAIN_TEMPLATE_PATH,
    MAX_SPROCKET_CONTACT_COLLISION_COMPONENT_IDS,
    BicycleAcceptanceThresholds,
    approved_bicycle_chain_family_spec_fixture,
    bicycle_chain_from_dict,
    evaluate_bicycle_roundtrip_trace,
    task_success_authorized,
    validate_powered_ratio,
    validate_roundtrip_phase_coverage,
    validate_runtime_bindings,
    validate_task_input_target,
    validate_wrench_provenance,
    validate_zero_input_trace,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_from_dict,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_runtime import (
    _bicycle_prescribed_swept_envelope_samples,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    candidate_mechanism_adapter_inventory,
    plugin_registration_for,
    registration_allows_runtime_execution,
    registered_candidate_definition,
    wrench_policies_for_plugin,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
    sha256_file,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


_THRESHOLDS = BicycleAcceptanceThresholds(
    powered_rotation_min_rad=2.8,
    signed_output_per_input=2.0,
    angular_ratio_tolerance=0.02,
    return_to_home_max_rad=0.25,
)
_EXPECTED_NEGATIVE_BINDINGS = {
    "reject_direct_rear_task_drive": "DIRECT_OUTPUT_DRIVE_REJECTED",
    "reject_inactive_gate_wrench": "GEOMETRY_GATE_OFF_ZERO_WRENCH",
    "reject_observable_axis_mismatch": (
        "OBSERVABLE_CHANNEL_BODY_FRAME_AXIS_MISMATCH_REJECTED"
    ),
    "reject_phase_gap_or_boundary_shift": (
        "PHASE_COVERAGE_AND_BOUNDARIES_REQUIRED"
    ),
    "reject_task_result_after_physical_audit_failure": (
        "PHYSICAL_AUDIT_FAIL_CANNOT_TASK_PASS"
    ),
    "reject_post_release_credit": "POST_RELEASE_EXCURSION_NOT_POWERED",
    "reject_partial_ratio_sampling": "POWERED_RATIO_ALL_SAMPLES",
    "reject_nonzero_return_input": (
        "RETURN_PHASE_TASK_INPUT_NONZERO_REJECTED"
    ),
    "reject_same_body_ratio_alias": "SAME_BODY_RATIO_ALIAS_REJECTED",
    "reject_runtime_body_swap": "SEMANTIC_RUNTIME_BODY_SWAP_REJECTED",
    "reject_sparse_or_duplicate_trace": (
        "SPARSE_OR_DUPLICATE_TIME_TRACE_INCONCLUSIVE"
    ),
    "reject_threshold_tamper": "THRESHOLD_TAMPER_REJECTED",
    "reject_unknown_typed_field": "UNKNOWN_FIELD_REJECTED",
    "reject_unrelated_direct_false": (
        "UNRELATED_DIRECT_FALSE_NOT_PROVENANCE"
    ),
    "reject_wrench_audit_context_swap": (
        "WRENCH_AUDIT_CONTEXT_MISMATCH_REJECTED"
    ),
    "reject_wrong_direction_motion": "WRONG_DIRECTION_FAILS",
    "reject_zero_input_energy": "ZERO_INPUT_NO_ENERGY",
}


def _assert_code(
    testcase: unittest.TestCase,
    expected: str,
    callback,
) -> None:
    with testcase.assertRaises(MechanismValidationError) as caught:
        callback()
    testcase.assertEqual(caught.exception.code, expected)


def _runtime_contract(spec) -> dict:
    role_bodies = {
        item.role: item
        for item in spec.body_dynamics
    }
    roles = (
        ("fixed", role_bodies["fixed"], "fixed"),
        ("input", role_bodies["input"], "dynamic"),
        ("output", role_bodies["unique_task_target"], "dynamic"),
    )
    groups = []
    parts = []
    for role, body, contract_role in roles:
        part_id = f"contract_{role}_part"
        groups.append(
            {
                "rigid_group_id": body.rigid_group_id,
                "part_ids": [part_id],
            }
        )
        parts.append(
            {
                "part_id": part_id,
                "occurrence_id": f"contract_{role}_occurrence",
                "rigid_group_id": body.rigid_group_id,
                "role": contract_role,
                "source_solid_ids": list(body.source_solid_ids),
            }
        )
    return {
        "rigid_groups": groups,
        "parts": parts,
        "frames": [
            {
                "frame_id": spec.input_frame_id,
                "rigid_group_id": role_bodies["input"].rigid_group_id,
                "origin_m": list(spec.input_origin_abs_m),
                "orientation_xyzw": [
                    -0.7071067811865475,
                    0.0,
                    0.0,
                    0.7071067811865476,
                ],
            },
            {
                "frame_id": spec.output_frame_id,
                "rigid_group_id": role_bodies[
                    "unique_task_target"
                ].rigid_group_id,
                "origin_m": list(spec.output_origin_abs_m),
                "orientation_xyzw": [
                    -0.7071067811865475,
                    0.0,
                    0.0,
                    0.7071067811865476,
                ],
            },
        ],
    }


def _parameterized_family() -> dict:
    value = approved_bicycle_chain_family_spec_fixture()
    value["mechanism_id"] = "legal_other_bicycle"
    body_ids = ("fixed_A", "crank_A", "wheel_A")
    group_ids = ("fixed_group_A", "crank_group_A", "wheel_group_A")
    solid_ids = ("part_A", "part_B", "part_C")
    for index in range(3):
        body = value["bodies"][index]
        body["body_id"] = body_ids[index]
        body["rigid_group_id"] = group_ids[index]
        body["source_solid_ids"] = [solid_ids[index]]
    frame_ids = ("frame_A", "frame_B")
    support_ids = ("support_A", "support_B")
    for index, body_index in enumerate((1, 2)):
        dof = value["degrees_of_freedom"][index]
        dof["dof_id"] = f"dof_{index}"
        dof["body_id"] = body_ids[body_index]
        support = value["finite_bearing_supports"][index]
        support["support_id"] = support_ids[index]
        support["fixed_body_id"] = body_ids[0]
        support["moving_body_id"] = body_ids[body_index]
        support["frame_id"] = frame_ids[index]
        sprocket = value["sprockets"][index]
        sprocket["sprocket_id"] = f"sprocket_{index}"
        sprocket["body_id"] = body_ids[body_index]
        sprocket["shaft_frame_id"] = frame_ids[index]
        sprocket["step_solid_id"] = solid_ids[body_index]
    transmission = value["chain_transmission"]
    transmission["transmission_id"] = "transmission_A"
    transmission["driver_body_id"] = body_ids[1]
    transmission["driven_body_id"] = body_ids[2]
    transmission["geometry_gate_id"] = "gate_A"
    gate = value["geometry_gates"][0]
    gate["gate_id"] = "gate_A"
    for index, body_index in enumerate((1, 2)):
        entity = gate["required_step_entities"][index]
        entity["body_id"] = body_ids[body_index]
        entity["step_solid_id"] = solid_ids[body_index]
    port = value["input_ports"][0]
    port["port_id"] = "port_A"
    port["target_body_id"] = body_ids[1]
    port["frame_id"] = frame_ids[0]
    target = value["unique_task_targets"][0]
    target["target_body_id"] = body_ids[2]
    target["pose_observable_id"] = "pose_A"
    target["actuation_observable_id"] = "actuation_A"
    fixed_parameters = value["held_out_validation_domain"][
        "fixed_parameters"
    ]
    fixed_parameters[0]["field_binding"] = (
        "bodies[crank_A].inertia_diagonal_kg_m2[1]"
    )
    fixed_parameters[1]["field_binding"] = (
        "bodies[wheel_A].inertia_diagonal_kg_m2[1]"
    )
    fixed_parameters[4]["field_binding"] = (
        "finite_bearing_supports[support_A]."
        "primary_axis_damping_nm_s_rad"
    )
    fixed_parameters[5]["field_binding"] = (
        "finite_bearing_supports[support_B]."
        "primary_axis_damping_nm_s_rad"
    )
    targets = (
        body_ids[1],
        body_ids[1],
        body_ids[2],
        body_ids[1],
        body_ids[2],
    )
    for index, target_body_id in enumerate(targets):
        value["wrench_policy"]["allowed"][index][
            "target_body_id"
        ] = target_body_id
    for index in (1, 2):
        value["wrench_policy"]["allowed"][index][
            "geometry_gate_id"
        ] = "gate_A"
    displacement = (0.10, 0.02, 0.03)
    for body_index in (1, 2):
        for key in ("center_of_mass_abs_m", "initial_position_abs_m"):
            value["bodies"][body_index][key] = [
                item + delta
                for item, delta in zip(
                    value["bodies"][body_index][key],
                    displacement,
                )
            ]
    for index in (0, 1):
        for container, key in (
            (value["degrees_of_freedom"], "origin_abs_m"),
            (value["sprockets"], "pitch_loop_center_abs_m"),
        ):
            container[index][key] = [
                item + delta
                for item, delta in zip(
                    container[index][key],
                    displacement,
                )
            ]
        entity = value["geometry_gates"][0]["required_step_entities"][
            index
        ]
        entity["center_abs_m"] = [
            item + delta
            for item, delta in zip(
                entity["center_abs_m"],
                displacement,
            )
        ]
    return value


def _ratio_trace(*, wrong_middle: bool = False, wrong_sign: bool = False):
    output_middle = -1.0 if wrong_sign else (0.7 if wrong_middle else 1.0)
    return [
        {
            "time_s": 0.0,
            "phase_id": "finite_engagement_settle",
            "input_crank_angle_rad": 0.0,
            "rear_drive_wheel_angle_rad": 0.0,
            "task_input_torque_nm": 3.2,
        },
        {
            "time_s": 0.00025,
            "phase_id": "finite_engagement_settle",
            "input_crank_angle_rad": 0.2,
            "rear_drive_wheel_angle_rad": 0.4,
            "task_input_torque_nm": 3.2,
        },
        {
            "time_s": 0.00050,
            "phase_id": "measured_powered_drive",
            "input_crank_angle_rad": 0.7,
            "rear_drive_wheel_angle_rad": 0.4 + output_middle,
            "task_input_torque_nm": 3.2,
        },
        {
            "time_s": 0.00075,
            "phase_id": "measured_powered_drive",
            "input_crank_angle_rad": 1.2,
            "rear_drive_wheel_angle_rad": 2.4,
            "task_input_torque_nm": 3.2,
        },
    ]


def _full_roundtrip_trace(*, measured_output_rad: float, post_peak_rad: float):
    step = 0.00025
    tick = 0
    result = []

    def add(phase_id, input_angle, output_angle, torque):
        nonlocal tick
        result.append(
            {
                "time_s": tick * step,
                "phase_id": phase_id,
                "input_crank_angle_rad": input_angle,
                "rear_drive_wheel_angle_rad": output_angle,
                "task_input_torque_nm": torque,
            }
        )
        tick += 1

    engagement_count = int(round(0.6 / step))
    for index in range(engagement_count + 1):
        fraction = index / engagement_count
        add(
            "finite_engagement_settle",
            0.2 * fraction,
            0.4 * fraction,
            3.2,
        )
    powered_count = int(round(1.6 / step))
    for index in range(powered_count + 1):
        fraction = index / powered_count
        add(
            "measured_powered_drive",
            0.2 + 0.5 * measured_output_rad * fraction,
            0.4 + measured_output_rad * fraction,
            3.2,
        )
    return_count = int(round(8.0 / step))
    for index in range(return_count + 1):
        fraction = index / return_count
        if fraction <= 0.5:
            output = (
                0.4
                + measured_output_rad
                + (post_peak_rad - 0.4 - measured_output_rad)
                * fraction
                * 2.0
            )
        else:
            output = post_peak_rad * (2.0 - 2.0 * fraction) + 0.1 * (
                2.0 * fraction - 1.0
            )
        add("passive_return", 0.5 * output, output, 0.0)
    settle_count = int(round(2.0 / step))
    for index in range(settle_count + 1):
        fraction = index / settle_count
        output = 0.1 * (1.0 - fraction)
        add("return_settle", 0.5 * output, output, 0.0)
    return result


def _chain_pair(spec, *, gate_active=True, mismatched_context=False):
    records = []
    for index, (wrench_id, body_id, torque) in enumerate(
        (
            ("finite_chain_input_torque", spec.input_body_id, -2.0),
            ("finite_chain_output_torque", spec.output_body_id, 1.0),
        )
    ):
        records.append(
            {
                "time_s": 0.00025,
                "phase_id": "measured_powered_drive",
                "wrench_id": wrench_id,
                "owner": "registered_plugin",
                "category": "transmission",
                "target_body_id": body_id,
                "torque_nm": torque,
                "axis_abs": list(spec.axis_abs),
                "geometry_gate_id": spec.transmission.geometry_gate_id,
                "gate_active": gate_active,
                "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
                "plugin_instance_id": "chain_instance",
                "chain_state_id": (
                    f"state_{index}"
                    if mismatched_context
                    else "shared_state"
                ),
            }
        )
    return records


class BicycleChainR04GovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = approved_bicycle_chain_family_spec_fixture()
        self.spec = bicycle_chain_from_dict(self.payload)
        self.contract = _runtime_contract(self.spec)
        self.contract_sha256 = canonical_sha256(self.contract)

    def _runtime_bindings(self) -> dict:
        return self.spec.expected_runtime_bindings(
            task_id="powered_bicycle_drive_roundtrip",
            assembly_contract=self.contract,
            assembly_contract_sha256=self.contract_sha256,
        )

    def _validate_runtime_bindings(self, bindings: dict) -> dict:
        return validate_runtime_bindings(
            self.spec,
            bindings,
            task_id="powered_bicycle_drive_roundtrip",
            assembly_contract=self.contract,
            assembly_contract_sha256=self.contract_sha256,
        )

    def test_r04_fixed_program_and_closed_domain_are_typed(self) -> None:
        self.assertEqual(self.spec.schema_version, BICYCLE_CHAIN_SCHEMA_VERSION)
        self.assertEqual(
            self.spec.capability_state,
            "INSTALLED_BUILTIN_ACTIVE",
        )
        self.assertEqual(
            [item.duration_s for item in self.spec.task_program.phases],
            [0.6, 1.6, 8.0, 2.0],
        )
        self.assertFalse(
            self.spec.task_program.engagement_task_motion_credit
        )
        self.assertFalse(
            self.spec.task_program.engagement_ratio_metric_credit
        )
        domain = self.spec.held_out_validation_domain
        self.assertEqual(
            domain.required_evidence_torque_nm,
            (3.22,),
        )
        self.assertTrue(domain.no_other_parameter_may_vary)
        self.assertTrue(domain.no_hidden_defaults)
        self.assertEqual(
            BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
            "bicycle_reverse_crank_and_resistance",
        )
        self.assertEqual(
            BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
            0.25,
        )
        compound_adapter = get_adapter(DERIVED_COMPOUND_ADAPTER_ID)
        materialized_adapter = adapter_from_dict(
            adapter_to_dict(compound_adapter)
        )
        self.assertIn(
            BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM,
            materialized_adapter.supported_programs,
        )
        self.assertEqual(
            dict(domain.fixed_parameters),
            {
                "input_crank_primary_axis_inertia": 0.03,
                "rear_drive_wheel_primary_axis_inertia": 0.45,
                "chain_stiffness": 100000.0,
                "chain_damping": 700.0,
                "input_primary_axis_bearing_damping": 1.6,
                "rear_primary_axis_bearing_damping": 0.03,
            },
        )

    def test_input_crank_perturbation_keeps_coupling_without_nominal_gate(
        self,
    ) -> None:
        metrics = {
            "max_support_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "max_chain_axis_angle_error_rad": 0.0,
            "max_chain_axial_alignment_error_m": 0.0,
            "max_chain_differential_tension_n": 1.0,
            "max_chain_coordinate_error_m": 0.0,
            "max_inactive_chain_wrench_nm": 0.0,
            "initial_stored_energy_j": 0.0,
            "max_input_bearing_primary_axis_torque_nm": 0.0,
            "max_input_bearing_total_support_torque_nm": 0.0,
            "max_rear_bearing_total_support_torque_nm": 0.0,
            "max_support_force_n": 0.0,
            "all_chain_gate_active": True,
            "powered_output_excursion_rad": 2.741719121624485,
            "input_angle_min_rad": -1.0,
            "output_angle_min_rad": -2.0,
            "max_powered_ratio_pointwise_error": 0.0,
            "max_powered_ratio_incremental_error": 0.0,
            "final_output_home_error_rad": 0.0,
            "return_task_input_exact_zero": True,
        }
        thresholds = {
            "powered_rotation_min_rad": 2.8,
            "angular_ratio_tolerance": 0.02,
            "return_to_home_max_rad": 0.25,
        }
        failures = _behavior_failures(
            metrics,
            load_case=SimpleNamespace(
                program=BICYCLE_CHAIN_CRANK_PERTURBATION_PROGRAM
            ),
            spec=self.spec,
            thresholds=thresholds,
        )
        self.assertEqual(failures, [])
        nominal_failures = _behavior_failures(
            metrics,
            load_case=SimpleNamespace(
                program="task_bicycle_powered_roundtrip"
            ),
            spec=self.spec,
            thresholds=thresholds,
        )
        self.assertIn(
            "bicycle_powered_rotation_not_reached",
            nominal_failures,
        )

    def test_prescribed_sweep_uses_frozen_task_angle_and_returns_home(
        self,
    ) -> None:
        samples = _bicycle_prescribed_swept_envelope_samples(self.spec)
        output_quaternions = [
            item["body_poses"][self.spec.output_body_id][
                "rotation_abs_quat_wxyz"
            ]
            for item in samples
        ]
        output_angles = [
            2.0 * math.acos(
                max(-1.0, min(1.0, abs(float(quaternion[0]))))
            )
            for quaternion in output_quaternions
        ]

        self.assertAlmostEqual(
            max(output_angles),
            self.spec.acceptance_thresholds.powered_rotation_min_rad,
            places=12,
        )
        self.assertLess(max(output_angles), 3.141592653589793)
        self.assertEqual(samples[0]["sweep_fraction"], 0.0)
        self.assertEqual(samples[-1]["sweep_fraction"], 0.0)
        self.assertEqual(
            output_quaternions[0],
            output_quaternions[-1],
        )

    def test_support_caps_and_step_tori_are_not_aliased(self) -> None:
        self.assertEqual(self.spec.chain_stiffness_n_m, 100000.0)
        self.assertEqual(self.spec.chain_damping_n_s_m, 700.0)
        self.assertEqual(
            self.spec.input_bearing.primary_axis_damping_nm_s_rad,
            1.6,
        )
        self.assertEqual(
            self.spec.output_bearing.primary_axis_damping_nm_s_rad,
            0.03,
        )
        self.assertEqual(
            self.spec.input_bearing.primary_axis_maximum_abs_torque_nm,
            8.0,
        )
        self.assertEqual(
            self.spec.input_bearing.maximum_total_support_torque_nm,
            80.0,
        )
        self.assertEqual(
            self.spec.output_bearing.maximum_total_support_torque_nm,
            80.0,
        )
        entities = self.spec.geometry_gate.required_step_entities
        self.assertEqual(
            self.spec.geometry_gate.pitch_radius_tolerance_m,
            0.0002,
        )
        self.assertEqual(
            self.spec.geometry_gate.pitch_center_tolerance_m,
            0.0002,
        )
        self.assertEqual(
            self.spec.geometry_gate.maximum_axis_parallel_error_rad,
            0.002,
        )
        self.assertEqual(
            self.spec.geometry_gate.maximum_pitch_plane_offset_m,
            0.0005,
        )
        self.assertTrue(
            self.spec.geometry_gate.zero_transmission_wrench_when_inactive
        )
        self.assertTrue(
            all(item.surface_exposure_required for item in entities)
        )
        self.assertEqual(
            (
                entities[0].step_solid_id,
                entities[0].major_radius_m,
                entities[0].minor_radius_m,
            ),
            ("solid_2", 0.08, 0.008),
        )
        self.assertEqual(
            (
                entities[1].step_solid_id,
                entities[1].major_radius_m,
                entities[1].minor_radius_m,
            ),
            ("solid_3", 0.04, 0.005),
        )

    def test_visible_chain_comparison_rejects_tampered_evidence(self) -> None:
        gate = self.spec.geometry_gate
        thresholds = {
            "maximum_chain_sprocket_separation_m": (
                gate.maximum_chain_sprocket_separation_m
            ),
            "wrap_band_m": gate.visible_chain_wrap_band_m,
            "minimum_wrap_angle_rad": (
                gate.minimum_visible_chain_wrap_angle_rad
            ),
            "minimum_wrap_sample_count": (
                gate.minimum_visible_chain_wrap_sample_count
            ),
            "maximum_pitch_plane_offset_m": (
                gate.maximum_pitch_plane_offset_m
            ),
        }
        role = {
            "minimum_sprocket_separation_m": 0.0,
            "wrap_angle_rad": 1.6,
            "near_pitch_band_sample_count": 20,
            "pitch_plane_offset_m": 0.0,
            "maximum_pitch_plane_deviation_m": 0.0,
        }
        visible = {
            "schema_version": "bicycle_visible_chain_geometry_v1",
            "source": "authoritative_step_solid_measurement",
            "declared": True,
            "required_part_id": gate.visible_chain_part_id,
            "passed": True,
            "status": "PASS",
            "failure_reasons": [],
            "thresholds": thresholds,
            "chain_solid_ids": ["solid_4"],
            "surface_sample_count": 40,
            "input": deepcopy(role),
            "output": deepcopy(role),
        }
        semantics = {
            "visible_chain_declared": True,
            "visible_chain_geometry_bound_to_finite_transmission_model": True,
        }
        self.assertEqual(
            _visible_chain_geometry_failures(
                {"visible_chain_geometry": visible},
                semantics,
                gate,
            ),
            [],
        )

        attacks = []
        missing = {}
        attacks.append(missing)
        threshold_tamper = deepcopy(visible)
        threshold_tamper["thresholds"][
            "maximum_chain_sprocket_separation_m"
        ] = 1.0
        attacks.append({"visible_chain_geometry": threshold_tamper})
        floating = deepcopy(visible)
        floating["input"]["minimum_sprocket_separation_m"] = 0.021
        attacks.append({"visible_chain_geometry": floating})
        plane_tamper = deepcopy(visible)
        plane_tamper["output"]["pitch_plane_offset_m"] = 0.03
        attacks.append({"visible_chain_geometry": plane_tamper})
        for attacked in attacks:
            with self.subTest(attacked=attacked):
                self.assertEqual(
                    _visible_chain_geometry_failures(
                        attacked,
                        semantics,
                        gate,
                    ),
                    ["bicycle_visible_chain_geometry_invalid"],
                )

    def test_omitted_required_visible_chain_is_rejected(self) -> None:
        gate = self.spec.geometry_gate
        geometry = {
            "visible_chain_geometry": {
                "schema_version": "bicycle_visible_chain_geometry_v1",
                "declared": False,
                "required_part_id": gate.visible_chain_part_id,
                "passed": True,
                "status": "NOT_DECLARED",
                "failure_reasons": [],
                "thresholds": {
                    "maximum_chain_sprocket_separation_m": (
                        gate.maximum_chain_sprocket_separation_m
                    ),
                    "wrap_band_m": gate.visible_chain_wrap_band_m,
                    "minimum_wrap_angle_rad": (
                        gate.minimum_visible_chain_wrap_angle_rad
                    ),
                    "minimum_wrap_sample_count": (
                        gate.minimum_visible_chain_wrap_sample_count
                    ),
                    "maximum_pitch_plane_offset_m": (
                        gate.maximum_pitch_plane_offset_m
                    ),
                },
            }
        }
        semantics = {
            "visible_chain_declared": False,
            "visible_chain_geometry_bound_to_finite_transmission_model": False,
        }

        self.assertEqual(
            _visible_chain_geometry_failures(geometry, semantics, gate),
            ["bicycle_visible_chain_geometry_invalid"],
        )

    def test_template_roundtrips_and_parameterized_bindings_are_legal(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[3]
        template = root / BICYCLE_CHAIN_TEMPLATE_PATH
        self.assertEqual(
            sha256_file(template),
            "2ff7fed2431647e4b141de349853036e532177d715a0748d1bc651c60681613d",
        )
        self.assertEqual(self.spec.to_dict(), self.payload)
        parameterized = bicycle_chain_from_dict(_parameterized_family())
        self.assertEqual(parameterized.input_body_id, "crank_A")
        self.assertEqual(parameterized.output_body_id, "wheel_A")
        self.assertEqual(parameterized.input_sprocket.step_solid_id, "part_B")
        self.assertEqual(parameterized.input_origin_abs_m, (-0.15, 0.02, 0.37))

    def test_contact_components_preserve_approved_seed_and_allow_segments(
        self,
    ) -> None:
        extended = deepcopy(self.payload)
        segment_ids = [
            f"rear.sprocket_torus_segment_{index:03d}"
            for index in range(1, 33)
        ]
        extended["sprockets"][1][
            "contact_collision_component_ids"
        ].extend(segment_ids)

        parsed = bicycle_chain_from_dict(extended)

        self.assertEqual(
            parsed.output_sprocket.approved_contact_collision_component_ids,
            ("rear.sprocket_backing",),
        )
        self.assertEqual(
            parsed.output_sprocket.contact_collision_component_ids,
            ("rear.sprocket_backing", *segment_ids),
        )
        self.assertEqual(parsed.to_dict(), extended)

    def test_contact_component_extensions_are_bounded_and_keep_seed_prefix(
        self,
    ) -> None:
        replaced_seed = deepcopy(self.payload)
        replaced_seed["sprockets"][1][
            "contact_collision_component_ids"
        ] = ["rear.unreviewed_contact"]
        _assert_code(
            self,
            "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
            lambda: bicycle_chain_from_dict(replaced_seed),
        )

        too_many = deepcopy(self.payload)
        too_many["sprockets"][1][
            "contact_collision_component_ids"
        ].extend(
            f"rear.contact_{index:03d}"
            for index in range(
                MAX_SPROCKET_CONTACT_COLLISION_COMPONENT_IDS
            )
        )
        _assert_code(
            self,
            "BICYCLE_CHAIN_GEOMETRY_BINDING_INVALID",
            lambda: bicycle_chain_from_dict(too_many),
        )

    def test_closed_schema_rejects_missing_wrong_type_and_extra_negative(
        self,
    ) -> None:
        mutations = []
        missing = deepcopy(self.payload)
        del missing["chain_transmission"]["chain_damping_n_s_m"]
        mutations.append(missing)
        wrong_type = deepcopy(self.payload)
        wrong_type["task_program_policy"]["phases"][0][
            "duration_s"
        ] = "0.6"
        mutations.append(wrong_type)
        extra_negative = deepcopy(self.payload)
        extra_negative["required_negative_tests"].append(
            "UNAPPROVED_EXTRA_NEGATIVE"
        )
        mutations.append(extra_negative)
        for attacked in mutations:
            with self.subTest(attacked=attacked):
                _assert_code(
                    self,
                    "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
                    lambda attacked=attacked: bicycle_chain_from_dict(attacked),
                )

    def test_unbound_domain_and_torque_cap_confusion_are_rejected(
        self,
    ) -> None:
        unbound = deepcopy(self.payload)
        unbound["held_out_validation_domain"]["fixed_parameters"][4][
            "field_binding"
        ] = "finite_bearing_supports[wrong].primary_axis_damping_nm_s_rad"
        _assert_code(
            self,
            "BICYCLE_CHAIN_HELD_OUT_DOMAIN_INVALID",
            lambda: bicycle_chain_from_dict(unbound),
        )
        for primary_cap, total_cap in ((80.0, 80.0), (8.0, 8.0)):
            attacked = deepcopy(self.payload)
            support = attacked["finite_bearing_supports"][0]
            support["primary_axis_maximum_abs_torque_nm"] = primary_cap
            support["maximum_total_support_torque_nm"] = total_cap
            _assert_code(
                self,
                "BICYCLE_CHAIN_PARAMETER_NOT_APPROVED",
                lambda attacked=attacked: bicycle_chain_from_dict(attacked),
            )

    def test_registration_is_honest_installed_builtin(self) -> None:
        registration = plugin_registration_for(BICYCLE_CHAIN_PLUGIN_ID)
        self.assertEqual(
            registration,
            {
                "schema_version": "simulation_ready_cad_installed_plugin_v2",
                "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
                "status": "ACTIVE",
                "registration_source": "installed_harness_code",
            },
        )
        self.assertEqual(candidate_mechanism_adapter_inventory(), [])
        self.assertTrue(wrench_policies_for_plugin(BICYCLE_CHAIN_PLUGIN_ID))
        _assert_code(
            self,
            "MECHANISM_PLUGIN_NOT_REGISTERED",
            lambda: registered_candidate_definition(BICYCLE_CHAIN_PLUGIN_ID),
        )

    def test_forged_test_dispatch_status_is_not_runtime_authority(self) -> None:
        forged = {
            **plugin_registration_for(BICYCLE_CHAIN_PLUGIN_ID),
            "status": "TEST_DISPATCH_AUTHORIZED",
            "activation_status": "NOT_ACTIVE",
            "test_dispatch_authorization": {
                "status": "TEST_DISPATCH_AUTHORIZED",
                "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
            },
        }
        self.assertFalse(registration_allows_runtime_execution(forged))

    def test_v1_registration_is_retained_only_as_read_only_history(
        self,
    ) -> None:
        _assert_code(
            self,
            "MECHANISM_PLUGIN_NOT_REGISTERED",
            lambda: plugin_registration_for(
                "bicycle_chain_sprocket_drive_v1"
            ),
        )
        self.assertEqual(candidate_mechanism_adapter_inventory(), [])


class BicycleChainApprovedContractTests(unittest.TestCase):
    """Canonical r04 negative-selector methods bound by the candidate registry."""

    def setUp(self) -> None:
        self.payload = approved_bicycle_chain_family_spec_fixture()
        self.spec = bicycle_chain_from_dict(self.payload)
        self.contract = _runtime_contract(self.spec)
        self.contract_sha256 = canonical_sha256(self.contract)

    def _runtime_bindings(self) -> dict:
        return self.spec.expected_runtime_bindings(
            task_id="powered_bicycle_drive_roundtrip",
            assembly_contract=self.contract,
            assembly_contract_sha256=self.contract_sha256,
        )

    def _validate_runtime_bindings(self, bindings: dict) -> dict:
        return validate_runtime_bindings(
            self.spec,
            bindings,
            task_id="powered_bicycle_drive_roundtrip",
            assembly_contract=self.contract,
            assembly_contract_sha256=self.contract_sha256,
        )

    def test_negative_reject_direct_rear_task_torque(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_DIRECT_OUTPUT_DRIVE",
            lambda: validate_task_input_target(
                self.spec,
                target_body_id=self.spec.output_body_id,
                phase_id="measured_powered_drive",
                torque_nm=3.2,
            ),
        )
        _assert_code(
            self,
            "WRENCH_POLICY_NOT_ALLOWED",
            lambda: _validate_rear_drive_wheel_wrench_inventory(
                [
                    {
                        "load_id": (
                            "validator_rear_shaft_diagnostic_resistance"
                        ),
                        "target_body": self.spec.output_body_id,
                        "wrench_kind": "torque",
                        "category": "validator_resistance",
                        "law_id": "validator_resistance_v1",
                    }
                ],
                self.spec,
            ),
        )

    def test_negative_reject_inactive_chain_nonzero_wrench(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_GATE_OFF_WRENCH_NONZERO",
            lambda: validate_wrench_provenance(
                self.spec,
                _chain_pair(self.spec, gate_active=False),
            ),
        )

    def test_negative_reject_observable_binding_mismatch(self) -> None:
        bindings = self._runtime_bindings()
        bindings["output_observable_id"] = "wrong_pose"
        _assert_code(
            self,
            "BICYCLE_CHAIN_BINDING_INVALID",
            lambda: self._validate_runtime_bindings(bindings),
        )

    def test_negative_reject_runtime_contract_identity_tamper(self) -> None:
        attacks = {
            "part": ("input_part_id", "attacker_part"),
            "occurrence": (
                "input_occurrence_id",
                "attacker_occurrence",
            ),
            "group": (
                "input_rigid_group_id",
                "attacker_group",
            ),
            "solid": (
                "input_source_solid_ids",
                ["solid_999"],
            ),
            "contract_hash": (
                "assembly_contract_sha256",
                "0" * 64,
            ),
            "role_binding_hash": (
                "contract_role_binding_sha256",
                "0" * 64,
            ),
        }
        for label, (field, replacement) in attacks.items():
            with self.subTest(label=label):
                bindings = self._runtime_bindings()
                bindings[field] = replacement
                _assert_code(
                    self,
                    "BICYCLE_CHAIN_BINDING_INVALID",
                    lambda bindings=bindings: (
                        self._validate_runtime_bindings(bindings)
                    ),
                )

    def test_negative_reject_contract_role_inventory_tamper(self) -> None:
        baseline = self._runtime_bindings()
        attacks = {}

        changed_part = deepcopy(self.contract)
        changed_part["parts"][1]["part_id"] = "renamed_input_part"
        changed_part["rigid_groups"][1]["part_ids"] = [
            "renamed_input_part"
        ]
        attacks["part"] = changed_part

        changed_occurrence = deepcopy(self.contract)
        changed_occurrence["parts"][1][
            "occurrence_id"
        ] = "renamed_input_occurrence"
        attacks["occurrence"] = changed_occurrence

        changed_group = deepcopy(self.contract)
        changed_group["parts"][1]["rigid_group_id"] = (
            self.spec.output_body_id + "_group_attack"
        )
        attacks["group"] = changed_group

        changed_solid = deepcopy(self.contract)
        changed_solid["parts"][1]["source_solid_ids"] = ["solid_999"]
        attacks["solid"] = changed_solid

        extra_part = deepcopy(self.contract)
        extra_part["rigid_groups"][1]["part_ids"].append("extra_part")
        extra_part["parts"].append(
            {
                "part_id": "extra_part",
                "occurrence_id": "extra_occurrence",
                "rigid_group_id": self.spec.body_dynamics[1].rigid_group_id,
                "role": "dynamic",
                "source_solid_ids": ["solid_998"],
            }
        )
        attacks["extra_part"] = extra_part

        swapped_frame = deepcopy(self.contract)
        swapped_frame["frames"][0]["rigid_group_id"] = (
            self.spec.body_dynamics[2].rigid_group_id
        )
        attacks["frame"] = swapped_frame

        for label, attacked in attacks.items():
            with self.subTest(label=label):
                with self.assertRaises(MechanismValidationError):
                    validate_runtime_bindings(
                        self.spec,
                        baseline,
                        task_id="powered_bicycle_drive_roundtrip",
                        assembly_contract=attacked,
                        assembly_contract_sha256=canonical_sha256(
                            attacked
                        ),
                    )

    def test_negative_reject_missing_phase_boundaries(self) -> None:
        samples = [
            {"phase_id": "measured_powered_drive"},
            {"phase_id": "passive_return"},
            {"phase_id": "return_settle"},
        ]
        _assert_code(
            self,
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            lambda: validate_roundtrip_phase_coverage(samples),
        )

    def test_phase_coverage_tolerates_only_runtime_float_roundoff(
        self,
    ) -> None:
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        samples = [
            {
                "phase_id": "finite_engagement_settle",
                "time_s": 0.00025,
            },
            {
                "phase_id": "finite_engagement_settle",
                "time_s": 0.5999999999999617,
            },
            {
                "phase_id": "measured_powered_drive",
                "time_s": 0.6002499999999616,
            },
            {
                "phase_id": "measured_powered_drive",
                "time_s": 2.200000000000023,
            },
            {
                "phase_id": "passive_return",
                "time_s": 2.2002500000000227,
            },
            {
                "phase_id": "passive_return",
                "time_s": 10.199999999994755,
            },
            {
                "phase_id": "return_settle",
                "time_s": 10.200249999994755,
            },
            {
                "phase_id": "return_settle",
                "time_s": 12.199999999989844,
            },
        ]
        validate_roundtrip_phase_coverage(samples, spec=spec)

        missing_one_cadence = deepcopy(samples)
        missing_one_cadence[4]["time_s"] += (
            spec.task_program.integration_step_s
        )
        _assert_code(
            self,
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
            lambda: validate_roundtrip_phase_coverage(
                missing_one_cadence,
                spec=spec,
            ),
        )

    def test_negative_reject_metric_success_after_physical_audit_failure(
        self,
    ) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_PHYSICAL_AUDIT_FAILED",
            lambda: task_success_authorized(
                metrics_pass=True,
                physical_audit_pass=False,
            ),
        )

    def test_negative_reject_post_release_powered_credit(self) -> None:
        trace = _full_roundtrip_trace(
            measured_output_rad=2.0,
            post_peak_rad=3.5,
        )
        _assert_code(
            self,
            "BICYCLE_CHAIN_POWERED_ROTATION_INSUFFICIENT",
            lambda: evaluate_bicycle_roundtrip_trace(
                self.spec,
                trace,
                _THRESHOLDS,
            ),
        )

    def test_negative_reject_return_task_input(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_RETURN_TASK_INPUT_NONZERO",
            lambda: validate_task_input_target(
                self.spec,
                target_body_id=self.spec.input_body_id,
                phase_id="passive_return",
                torque_nm=1.0e-12,
            ),
        )

    def test_negative_reject_runtime_body_swap(self) -> None:
        bindings = self._runtime_bindings()
        bindings["input_body_id"], bindings["output_body_id"] = (
            bindings["output_body_id"],
            bindings["input_body_id"],
        )
        _assert_code(
            self,
            "BICYCLE_CHAIN_BINDING_INVALID",
            lambda: self._validate_runtime_bindings(bindings),
        )

    def test_negative_reject_same_body_ratio_alias(self) -> None:
        bindings = self._runtime_bindings()
        bindings["input_body_id"] = bindings["output_body_id"]
        _assert_code(
            self,
            "BICYCLE_CHAIN_BINDING_INVALID",
            lambda: self._validate_runtime_bindings(bindings),
        )

    def test_negative_reject_endpoint_only_ratio(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_SIGNED_RATIO_INVALID",
            lambda: validate_powered_ratio(
                _ratio_trace(wrong_middle=True),
                _THRESHOLDS,
            ),
        )

    def test_negative_reject_frozen_threshold_change(self) -> None:
        changed = BicycleAcceptanceThresholds(
            powered_rotation_min_rad=2.7,
            signed_output_per_input=2.0,
            angular_ratio_tolerance=0.02,
            return_to_home_max_rad=0.25,
        )
        _assert_code(
            self,
            "BICYCLE_CHAIN_THRESHOLD_TAMPER",
            lambda: evaluate_bicycle_roundtrip_trace(
                self.spec,
                (),
                changed,
            ),
        )

    def test_negative_reject_sparse_duplicate_trace(self) -> None:
        samples = [
            {
                "time_s": time_s,
                "phase_id": "zero_state_no_input_drift",
                "input_crank_angle_rad": 0.0,
                "rear_drive_wheel_angle_rad": 0.0,
                "task_input_torque_nm": 0.0,
            }
            for time_s in (0.0, 0.00025, 0.00025, 0.00050)
        ]
        _assert_code(
            self,
            "BICYCLE_CHAIN_TRACE_INCONCLUSIVE",
            lambda: validate_zero_input_trace(samples),
        )

    def test_negative_reject_unknown_chain_field(self) -> None:
        attacked = deepcopy(self.payload)
        attacked["unapproved_hidden_field"] = 1
        _assert_code(
            self,
            "BICYCLE_CHAIN_SPEC_SCHEMA_INVALID",
            lambda: bicycle_chain_from_dict(attacked),
        )

    def test_negative_reject_direct_false_shortcut(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_WRENCH_PROVENANCE_INVALID",
            lambda: validate_wrench_provenance(
                self.spec,
                (),
                direct_actuation_flag=False,
            ),
        )

    def test_negative_reject_wrench_context_mismatch(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_WRENCH_CONTEXT_INVALID",
            lambda: validate_wrench_provenance(
                self.spec,
                _chain_pair(self.spec, mismatched_context=True),
            ),
        )

    def test_negative_reject_wrong_signed_ratio(self) -> None:
        _assert_code(
            self,
            "BICYCLE_CHAIN_SIGNED_RATIO_INVALID",
            lambda: validate_powered_ratio(
                _ratio_trace(wrong_sign=True),
                _THRESHOLDS,
            ),
        )
        double_reverse = _full_roundtrip_trace(
            measured_output_rad=-3.0,
            post_peak_rad=-3.5,
        )
        validate_powered_ratio(double_reverse, _THRESHOLDS)
        _assert_code(
            self,
            "BICYCLE_CHAIN_POWERED_ROTATION_INSUFFICIENT",
            lambda: evaluate_bicycle_roundtrip_trace(
                self.spec,
                double_reverse,
                _THRESHOLDS,
            ),
        )

    def test_negative_reject_zero_input_created_energy(self) -> None:
        attacked = deepcopy(self.payload)
        attacked["bodies"][1]["initial_angular_velocity_abs_rad_s"][1] = (
            0.001
        )
        _assert_code(
            self,
            "BICYCLE_CHAIN_PARAMETER_NOT_APPROVED",
            lambda: bicycle_chain_from_dict(attacked),
        )


if __name__ == "__main__":
    unittest.main()
