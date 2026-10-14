from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_ground_roll_spec import (
    MAXIMUM_OFF_AXIS_ROTATION_RAD,
    bicycle_ground_roll_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.certificate import (
    build_replacement,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
    compare_trace_pair,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.discovery import (
    _serialized_family_spec,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    mechanism_adapter_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    BicycleGroundRollModel,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    _body_inventory,
    build_task_channel_registry,
)


def _family_spec() -> dict:
    return {
        "mechanism_kind": "bicycle_ground_roll",
        "mechanism_id": "guided_bicycle_ground_roll_v1",
        "bodies": [
            {
                "body_id": "ground",
                "group_ids": ["ground_surface", "upright_guide"],
                "fixed": True,
                "mass_kg": 100.0,
                "center_abs_m": [0.6, 0.0, -0.05],
                "size_m": [7.0, 1.0, 0.1],
            },
            {
                "body_id": "chassis",
                "group_ids": ["frame", "saddle", "rear_sprocket"],
                "fixed": False,
                "mass_kg": 12.0,
                "center_abs_m": [0.55, 0.0, 0.62],
                "size_m": [1.2, 0.24, 0.7],
            },
            {
                "body_id": "steering",
                "group_ids": ["fork", "handlebar"],
                "fixed": False,
                "mass_kg": 2.5,
                "center_abs_m": [1.0, 0.0, 0.70],
                "size_m": [0.25, 0.55, 0.8],
            },
            {
                "body_id": "rear_wheel",
                "group_ids": ["rear_wheel"],
                "fixed": False,
                "mass_kg": 2.0,
                "center_abs_m": [0.0, 0.0, 0.33],
                "size_m": [0.66, 0.07, 0.66],
            },
            {
                "body_id": "front_wheel",
                "group_ids": ["front_wheel"],
                "fixed": False,
                "mass_kg": 2.0,
                "center_abs_m": [1.1, 0.0, 0.33],
                "size_m": [0.66, 0.07, 0.66],
            },
        ],
        "ground_body_id": "ground",
        "chassis_body_id": "chassis",
        "steering_body_id": "steering",
        "rear_wheel_body_id": "rear_wheel",
        "front_wheel_body_id": "front_wheel",
        "rear_axle_origin_abs_m": [0.0, 0.0, 0.33],
        "front_axle_origin_abs_m": [1.1, 0.0, 0.33],
        "steering_origin_abs_m": [0.96, 0.0, 0.78],
        "wheel_axis_abs": [0.0, 1.0, 0.0],
        "steering_axis_abs": [0.0, 0.0, 1.0],
        "travel_axis_abs": [1.0, 0.0, 0.0],
        "ground_normal_abs": [0.0, 0.0, 1.0],
        "ground_plane_height_m": 0.0,
        "rear_wheel_radius_m": 0.33,
        "front_wheel_radius_m": 0.33,
        "nominal_rear_drive_torque_nm": 15.0,
        "ground_friction": 0.8,
        "maximum_axle_drift_m": 0.004,
        "maximum_guide_drift_m": 0.006,
        "maximum_off_axis_rotation_rad": MAXIMUM_OFF_AXIS_ROTATION_RAD,
        "maximum_rolling_slip_ratio": 0.12,
        "maximum_ground_penetration_m": 0.004,
        "maximum_ground_separation_m": 0.0002,
        "minimum_powered_contact_coverage": 0.90,
        "minimum_front_rotation_rad": 5.0,
        "minimum_chassis_travel_m": 1.5,
    }


def _plan() -> MechanismPlan:
    family = _family_spec()
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=family,
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="bicycle_ground_roll_fixture",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="bicycle_ground_roll",
        capsule_file="bicycle_ground_roll.json",
        family="compound",
        protocol_id="bicycle_ground_roll",
        fixed_group_ids=("ground_surface", "upright_guide"),
        moving_group_ids=(
            "frame",
            "saddle",
            "rear_sprocket",
            "fork",
            "handlebar",
            "rear_wheel",
            "front_wheel",
        ),
        axis_abs=(1.0, 0.0, 0.0),
        joint_origin_abs_m=(0.0, 0.0, 0.33),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="rear_axle_frame",
        input_port_id="rear_wheel_torque",
        input_frame_id="rear_axle_frame",
        input_point_abs_m=(0.0, 0.0, 0.33),
        observable_id="front_wheel_and_chassis",
        observable_frame_id="front_axle_frame",
        observable_point_abs_m=(1.1, 0.0, 0.33),
        task_ids=("task.rear_driven_ground_roll",),
        characteristic_length_m=1.1,
        moving_extents_m=(1.5, 0.6, 1.1),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _traces(plan: MechanismPlan, load_case) -> tuple[dict, dict]:
    spec = bicycle_ground_roll_from_dict(_family_spec())
    phases = [
        (0.0, "initial", 0.0, 0.0, 0.0),
        (0.5, "ground_contact_settle", 0.0, 0.0, 0.0),
        (0.52, "rear_wheel_powered_drive", 0.3, 0.3, 0.10),
        (0.80, "rear_wheel_powered_drive", 2.4, 2.4, 0.80),
        (1.20, "rear_wheel_powered_drive", 6.0, 6.0, 1.98),
        (1.30, "passive_roll_observe", 6.2, 6.2, 2.046),
    ]

    def body_poses(travel: float) -> dict:
        return {
            body.body_id: {
                "position_abs_m": [
                    body.center_abs_m[0] + (0.0 if body.fixed else travel),
                    body.center_abs_m[1],
                    body.center_abs_m[2],
                ],
                "rotation_abs_quat_wxyz": list(body.rotation_abs_quat_wxyz),
            }
            for body in spec.bodies
        }

    def trace(branch: str) -> dict:
        physical = branch == "physical"
        samples = []
        for time_s, phase, rear_angle, front_angle, travel in phases:
            samples.append(
                {
                    "time_s": time_s,
                    "phase_id": phase,
                    "rear_wheel_angle_rad": rear_angle,
                    "front_wheel_angle_rad": front_angle,
                    "chassis_position_m": travel,
                    "rear_wheel_speed_rad_s": 1.0,
                    "front_wheel_speed_rad_s": 1.0,
                    "chassis_speed_m_s": 0.33,
                    "rolling_slip_ratio": 0.0,
                    "rear_axle_drift_m": 0.001,
                    "front_axle_drift_m": 0.001,
                    "chassis_guide_transverse_drift_m": 0.001,
                    "max_off_axis_rotation_rad": 0.001,
                    "max_penetration_m": 0.001,
                    "max_ground_separation_m": 0.0,
                    "rear_ground_contact_count": 1 if physical else 0,
                    "front_ground_contact_count": 1 if physical else 0,
                    "expected_contact_count": 1 if physical else 0,
                    "body_poses": body_poses(travel),
                }
            )
        payload = {
            "status": "COMPLETED",
            "branch": branch,
            "mechanism_plugin_id": "bicycle_ground_roll_v1",
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "load_case_sha256": canonical_sha256(load_case.to_dict()),
            "samples": samples,
        }
        if physical:
            payload.update(
                {
                    "engine": {
                        "internal_ideal_joint_count": 0,
                        "declared_support_links": [
                            "ChLinkLockPrismatic",
                            "ChLinkLockRevolute",
                            "ChLinkLockRevolute",
                            "ChLinkLockLock",
                        ],
                        "declared_axle_revolute_joint_count": 2,
                        "declared_upright_guide_count": 1,
                        "declared_straight_steering_lock_count": 1,
                        "contact_bypass_link_count": 0,
                        "contact_model": "native_finite_coulomb_wheel_ground_contact",
                    },
                    "branch_semantics": {
                        "replaced_contacts_disabled": False,
                        "pose_setters_used": False,
                        "direct_front_wheel_actuation_used": False,
                        "front_wheel_passive": True,
                        "chassis_fixed": False,
                        "chassis_fixed_during_contact_settle": True,
                        "chassis_released_and_velocities_zeroed_before_power": True,
                        "rolling_constraint_used": False,
                        "declared_axle_revolute_joints_used": True,
                        "contact_bypass_ideal_links_forbidden": True,
                        "upright_fixture_mode": "visible_declared_prismatic_test_stand_guide_free_in_travel_axis",
                        "steering_fixture_mode": "declared_ground_roll_task_lock_straight_ahead",
                        "task_input_body_id": "rear_wheel",
                        "unique_task_output_body_id": "front_wheel",
                    },
                    "functional_geometry": {
                        "passed": True,
                        "mechanism_kind": "bicycle_ground_roll",
                        "package_binding_sha256": plan.package_binding_sha256,
                        "assembly_sha256": "a" * 64,
                        "wheel_geometry_bound_to_step": True,
                        "ground_plane_bound_to_step": True,
                        "visible_upright_fixture_bound": True,
                    },
                    "collision_geometry": {
                        "package_binding_sha256": plan.package_binding_sha256,
                        "assembly_sha256": "a" * 64,
                        "assembly_contract_sha256": "b" * 64,
                        "bodies": [
                            {
                                "body_id": body.body_id,
                                "components": [
                                    {"solid_id": f"{body.body_id}_solid"}
                                ],
                            }
                            for body in spec.bodies
                        ],
                    },
                    "wrench_audit": {
                        "passed": True,
                        "direct_output_drive_detected": False,
                        "task_input_bodies_observed": ["rear_wheel"],
                        "records": [
                            {"target_body": "rear_wheel", "task_input": True}
                        ],
                    },
                }
            )
        return payload

    return trace("physical"), trace("reduced")


class BicycleGroundRollPluginTests(unittest.TestCase):
    def test_discovery_serialization_builds_exact_task_channels(self) -> None:
        raw = _family_spec()
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
            family_spec_plugin_for_kind,
        )

        serialized = _serialized_family_spec(
            raw,
            bicycle_ground_roll_from_dict(raw),
            family_spec_plugin_for_kind(raw.get("mechanism_kind")),
        )
        self.assertTrue(
            all(
                isinstance(body["group_ids"], list)
                for body in serialized["bodies"]
            )
        )

        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=serialized,
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        registry = build_task_channel_registry(
            plan,
            {
                "input_ports": [
                    {
                        "port_id": "rear_wheel_torque",
                        "rigid_group_id": "rear_wheel",
                        "frame_id": "rear_axle_frame",
                        "direction_abs": [0.0, 1.0, 0.0],
                        "wrench_type": "torque",
                    }
                ],
                "observables": [
                    {
                        "observable_id": "chassis_pose",
                        "rigid_group_id": "frame",
                        "frame_id": "chassis_frame",
                        "quantity": "pose_6d",
                    },
                    {
                        "observable_id": "front_wheel_pose",
                        "rigid_group_id": "front_wheel",
                        "frame_id": "front_axle_frame",
                        "quantity": "pose_6d",
                    },
                ],
            },
        )

        self.assertEqual(registry["plugin_id"], "bicycle_ground_roll_v1")
        self.assertEqual(
            registry["inputs"]["rear_wheel_torque"]["body_id"],
            "rear_wheel",
        )
        self.assertEqual(
            registry["observables"]["chassis_pose"]["body_id"],
            "chassis",
        )
        self.assertEqual(
            registry["observables"]["front_wheel_pose"]["body_id"],
            "front_wheel",
        )

    def test_runtime_body_inventory_rejects_non_json_group_arrays(self) -> None:
        invalid = _family_spec()
        invalid["bodies"][0]["group_ids"] = (
            "ground_surface",
            "upright_guide",
        )

        with self.assertRaises(MechanismValidationError) as caught:
            _body_inventory(invalid)

        self.assertEqual(caught.exception.code, "TASK_BODY_INVENTORY_INVALID")
        self.assertIn("bodies[0].group_ids", caught.exception.message)

    def test_runtime_body_inventory_allows_omitted_bodies(self) -> None:
        self.assertEqual(
            _body_inventory({"mechanism_kind": "axis_spring"}),
            ({}, ()),
        )

    def test_plugin_load_and_trace_pair_pass(self) -> None:
        plan = _plan()
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        self.assertEqual(plugin.plugin_id, "bicycle_ground_roll_v1")
        load_case = plugin.load_cases(plan)[0]
        self.assertEqual(load_case.program, "task_bicycle_ground_roll")
        physical, reduced = _traces(plan, load_case)
        result = compare_trace_pair(plan, load_case, physical, reduced)
        self.assertEqual(result["verdict"], "PASS")
        self.assertGreaterEqual(
            result["physical_metrics"]["powered_front_rotation_rad"], 5.0
        )
        self.assertGreaterEqual(
            result["physical_metrics"]["powered_chassis_travel_m"], 1.5
        )
        self.assertEqual(len(result["requirement_checks"]), 7)
        self.assertTrue(
            all(
                check["passed"] is True
                for check in result["requirement_checks"].values()
            )
        )

    def test_direct_front_actuation_is_inconclusive(self) -> None:
        plan = _plan()
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        load_case = plugin.load_cases(plan)[0]
        physical, reduced = _traces(plan, load_case)
        physical["branch_semantics"]["direct_front_wheel_actuation_used"] = True
        physical["wrench_audit"]["direct_output_drive_detected"] = True
        physical["wrench_audit"]["records"].append(
            {"target_body": "front_wheel", "task_input": True}
        )
        result = compare_trace_pair(plan, load_case, physical, reduced)
        self.assertEqual(result["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "bicycle_ground_roll_front_wheel_direct_actuation_detected",
            result["failure_reasons"],
        )
        self.assertFalse(
            result["requirement_checks"][
                "validator.bicycle.front_wheel_passive"
            ]["passed"]
        )

    def test_visible_fixture_step_binding_is_required(self) -> None:
        plan = _plan()
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        load_case = plugin.load_cases(plan)[0]
        physical, reduced = _traces(plan, load_case)
        physical["functional_geometry"]["visible_upright_fixture_bound"] = False

        result = compare_trace_pair(plan, load_case, physical, reduced)

        self.assertEqual(result["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertFalse(
            result["requirement_checks"][
                "validator.bicycle.ground_roll_step_geometry"
            ]["passed"]
        )

    def test_public_threshold_and_visible_fixture_are_closed(self) -> None:
        for field, changed_value in (
            ("maximum_rolling_slip_ratio", 0.2),
            ("maximum_ground_separation_m", 0.001),
            ("minimum_powered_contact_coverage", 0.5),
        ):
            with self.subTest(field=field):
                changed = _family_spec()
                changed[field] = changed_value
                with self.assertRaisesRegex(
                    MechanismValidationError,
                    "BICYCLE_GROUND_ROLL_THRESHOLD_CHANGED",
                ):
                    bicycle_ground_roll_from_dict(changed)
        missing_fixture = _family_spec()
        missing_fixture["bodies"][0]["group_ids"] = ["ground_surface"]
        with self.assertRaisesRegex(
            MechanismValidationError,
            "BICYCLE_GROUND_ROLL_UPRIGHT_FIXTURE_MISSING",
        ):
            bicycle_ground_roll_from_dict(missing_fixture)

        short_ground = _family_spec()
        short_ground["bodies"][0]["size_m"] = [3.0, 1.0, 0.1]
        with self.assertRaisesRegex(
            MechanismValidationError,
            "BICYCLE_GROUND_ROLL_SURFACE_TOO_SHORT",
        ):
            bicycle_ground_roll_from_dict(short_ground)

    def test_trace_phase_labels_must_match_public_time_windows(self) -> None:
        plan = _plan()
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        load_case = plugin.load_cases(plan)[0]
        physical, reduced = _traces(plan, load_case)
        physical["samples"][2]["phase_id"] = "passive_roll_observe"
        result = compare_trace_pair(plan, load_case, physical, reduced)
        self.assertEqual(result["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "BICYCLE_GROUND_ROLL_TRACE_PHASE_WINDOW_INVALID",
            result["failure_reasons"],
        )

    def test_paired_contact_coverage_and_finite_ground_separation_gate(self) -> None:
        plan = _plan()
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        load_case = plugin.load_cases(plan)[0]
        physical, reduced = _traces(plan, load_case)
        powered = [
            sample
            for sample in physical["samples"]
            if sample["phase_id"] == "rear_wheel_powered_drive"
        ]
        powered[0]["front_ground_contact_count"] = 0
        powered[0]["expected_contact_count"] = 0
        result = compare_trace_pair(plan, load_case, physical, reduced)
        self.assertEqual(result["verdict"], "PHYSICAL_FAILURE")
        self.assertIn(
            "bicycle_ground_roll_contact_coverage_incomplete",
            result["failure_reasons"],
        )

        physical, reduced = _traces(plan, load_case)
        physical["samples"][3]["max_ground_separation_m"] = 0.001
        result = compare_trace_pair(plan, load_case, physical, reduced)
        self.assertEqual(result["verdict"], "PHYSICAL_FAILURE")
        self.assertIn(
            "bicycle_ground_roll_ground_separation_exceeded",
            result["failure_reasons"],
        )

    def test_ideal_reference_moves_full_machine_without_claiming_contact(self) -> None:
        plan = _plan()
        samples = [
            {"time_s": 0.0, "phase_id": "initial", "rear_wheel_angle_rad": 0.0},
            {
                "time_s": 1.0,
                "phase_id": "rear_wheel_powered_drive",
                "rear_wheel_angle_rad": 6.0,
            },
        ]
        output, semantics = BicycleGroundRollModel().evaluate_samples(plan, samples)
        self.assertAlmostEqual(output[-1]["chassis_position_m"], 1.98)
        self.assertAlmostEqual(output[-1]["front_wheel_angle_rad"], 6.0)
        self.assertEqual(output[-1]["front_ground_contact_count"], 0)
        self.assertFalse(semantics["physical_contact_evidence_claimed"])
        self.assertAlmostEqual(
            output[-1]["body_poses"]["front_wheel"]["position_abs_m"][0],
            3.08,
        )

    def test_replacement_uses_ground_roll_topology(self) -> None:
        plan = _plan()
        replacement = build_replacement(
            plan,
            {
                "verdict": "KINEMATIC_CERTIFIED",
                "tested_envelope": {"load_case_ids": ["task_bicycle_ground_roll"]},
            },
            certificate_sha256="a" * 64,
        )
        topology = replacement["replacement_model"]["compound_topology"]
        self.assertEqual(topology["mechanism_kind"], "bicycle_ground_roll")
        self.assertEqual(topology["front_wheel_body_id"], "front_wheel")
        self.assertFalse(topology["contact_or_dynamic_response_claimed"])

    def test_live_runtime_rear_torque_moves_chassis_and_passive_front_wheel(
        self,
    ) -> None:
        try:
            import pychrono as chrono
        except ImportError:
            self.skipTest("PyChrono is not installed in this test environment")

        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_ground_roll_runtime import (
            simulate_bicycle_ground_roll,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
            ExecutionRequest,
            _wrench_ledger_for_execution,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
            activate_wrench_ledger,
        )

        plan = _plan()
        spec = bicycle_ground_roll_from_dict(_family_spec())
        plugin, adapter = mechanism_adapter_for_plan(plan)
        load_case = plugin.load_cases(plan)[0]
        execution = ExecutionRequest(
            request_file=Path("ground-roll-smoke.request.json"),
            output_file=Path("ground-roll-smoke.trace.json"),
            branch="physical",
            artifact_branch="physical",
            adapter=adapter,
            mechanism_plugin=plugin,
            plan=plan,
            load_case=load_case,
            reference_trace_file=None,
            functional_geometry_evidence={
                "passed": True,
                "mechanism_kind": "bicycle_ground_roll",
                "package_binding_sha256": plan.package_binding_sha256,
                "wheel_geometry_bound_to_step": True,
                "ground_plane_bound_to_step": True,
                "visible_upright_fixture_bound": True,
            },
        )
        collision = {
            "package_binding_sha256": plan.package_binding_sha256,
            "collision_envelope_m": 12.0e-6,
            "collision_margin_m": 2.0e-6,
            "schema": "ground_roll_smoke_v1",
            "source": "test_owned_primitives",
            "assembly_file": "assembly.step",
            "assembly_sha256": "a" * 64,
            "assembly_contract_sha256": "b" * 64,
            "component_count": 5,
            "bodies": [],
        }

        def make_test_body(
            chrono_module,
            *,
            body_spec,
            material,
            inertia_xx,
            **_kwargs,
        ):
            body = chrono_module.ChBodyAuxRef()
            if body_spec.body_id == "ground":
                body.AddCollisionShape(
                    chrono_module.ChCollisionShapeBox(
                        material, *body_spec.size_m
                    )
                )
            elif body_spec.body_id in {"rear_wheel", "front_wheel"}:
                body.AddCollisionShape(
                    chrono_module.ChCollisionShapeCylinder(
                        material, 0.33, 0.06
                    ),
                    chrono_module.ChFramed(
                        chrono_module.ChVector3d(0.0, 0.0, 0.0),
                        chrono_module.QuatFromAngleX(-0.5 * 3.141592653589793),
                    ),
                )
            else:
                body.AddCollisionShape(
                    chrono_module.ChCollisionShapeSphere(material, 0.02)
                )
            body.SetName(body_spec.body_id)
            body.SetFixed(body_spec.fixed)
            body.SetPos(chrono_module.ChVector3d(*body_spec.center_abs_m))
            body.SetRot(
                chrono_module.ChQuaterniond(*body_spec.rotation_abs_quat_wxyz)
            )
            body.SetMass(body_spec.mass_kg)
            body.SetInertiaXX(chrono_module.ChVector3d(*inertia_xx))
            body.EnableCollision(
                body_spec.body_id
                in {"ground", "rear_wheel", "front_wheel"}
            )
            return body, {
                "body_id": body_spec.body_id,
                "contact_enabled": body.IsCollisionEnabled(),
                "component_count": 1,
            }

        ledger = _wrench_ledger_for_execution(execution)
        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision.family_collision_geometry",
                return_value=collision,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision.make_family_body",
                side_effect=make_test_body,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision.trace_collision_thresholds",
                return_value={
                    "collision_envelope_m": 12.0e-6,
                    "collision_margin_m": 2.0e-6,
                },
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.worker._write_trace_checkpoint",
            ),
            activate_wrench_ledger(ledger),
        ):
            trace = simulate_bicycle_ground_roll(execution, Path.cwd())
            ledger.bind_executed_schedule(
                trace["executed_phases"],
                executed_load_sha256=trace["executed_load_sha256"],
            )
            audit = ledger.verify_for_certification()

        powered = [
            sample
            for sample in trace["samples"]
            if sample["phase_id"] == "rear_wheel_powered_drive"
        ]
        self.assertTrue(powered)
        self.assertEqual(audit["task_input_bodies_observed"], ["rear_wheel"])
        self.assertFalse(audit["direct_output_drive_detected"])
        first_powered_index = trace["samples"].index(powered[0])
        baseline = trace["samples"][first_powered_index - 1]
        significant = [
            sample
            for sample in powered
            if abs(
                sample["chassis_position_m"]
                - baseline["chassis_position_m"]
            )
            >= 0.10
        ]
        diagnostics = {
            "maximum_chassis_travel_m": max(
                abs(
                    sample["chassis_position_m"]
                    - baseline["chassis_position_m"]
                )
                for sample in powered
            ),
            "maximum_front_rotation_rad": max(
                abs(
                    sample["front_wheel_angle_rad"]
                    - baseline["front_wheel_angle_rad"]
                )
                for sample in powered
            ),
            "maximum_rear_rotation_rad": max(
                abs(
                    sample["rear_wheel_angle_rad"]
                    - baseline["rear_wheel_angle_rad"]
                )
                for sample in powered
            ),
            "maximum_rolling_slip_ratio": max(
                (sample["rolling_slip_ratio"] for sample in significant),
                default=float("inf"),
            ),
            "maximum_chassis_speed_m_s": max(
                abs(sample["chassis_speed_m_s"]) for sample in powered
            ),
            "maximum_rear_speed_rad_s": max(
                abs(sample["rear_wheel_speed_rad_s"]) for sample in powered
            ),
            "maximum_front_speed_rad_s": max(
                abs(sample["front_wheel_speed_rad_s"]) for sample in powered
            ),
            "maximum_rear_axle_drift_m": max(
                sample["rear_axle_drift_m"] for sample in trace["samples"]
            ),
            "maximum_front_axle_drift_m": max(
                sample["front_axle_drift_m"] for sample in trace["samples"]
            ),
            "maximum_guide_drift_m": max(
                sample["chassis_guide_transverse_drift_m"]
                for sample in trace["samples"]
            ),
            "maximum_off_axis_rotation_rad": max(
                sample["max_off_axis_rotation_rad"]
                for sample in trace["samples"]
            ),
            "maximum_ground_penetration_m": max(
                sample["max_penetration_m"] for sample in trace["samples"]
            ),
            "maximum_ground_separation_m": max(
                sample["max_ground_separation_m"] for sample in powered
            ),
            "rear_contact_samples": sum(
                sample["rear_ground_contact_count"] > 0
                for sample in powered
            ),
            "front_contact_samples": sum(
                sample["front_ground_contact_count"] > 0
                for sample in powered
            ),
            "powered_samples": len(powered),
            "motion_series": [
                (
                    round(sample["time_s"], 3),
                    round(sample["chassis_position_m"], 6),
                    round(sample["chassis_speed_m_s"], 6),
                    round(sample["rear_wheel_angle_rad"], 6),
                    round(sample["rear_wheel_speed_rad_s"], 6),
                    round(sample["front_wheel_angle_rad"], 6),
                    round(sample["front_wheel_speed_rad_s"], 6),
                )
                for sample in powered[::10]
            ],
            "final_body_poses": powered[-1]["body_poses"],
            "final_contact_pairs": powered[-1]["contact_pair_counts"],
            "final_support_reactions": powered[-1]["support_reactions"],
        }
        self.assertGreaterEqual(
            diagnostics["maximum_chassis_travel_m"],
            spec.minimum_chassis_travel_m,
            diagnostics,
        )
        self.assertGreaterEqual(
            diagnostics["maximum_front_rotation_rad"],
            spec.minimum_front_rotation_rad,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_rolling_slip_ratio"],
            spec.maximum_rolling_slip_ratio,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_rear_axle_drift_m"],
            spec.maximum_axle_drift_m,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_front_axle_drift_m"],
            spec.maximum_axle_drift_m,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_guide_drift_m"],
            spec.maximum_guide_drift_m,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_off_axis_rotation_rad"],
            spec.maximum_off_axis_rotation_rad,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_ground_penetration_m"],
            spec.maximum_ground_penetration_m,
            diagnostics,
        )
        self.assertLessEqual(
            diagnostics["maximum_ground_separation_m"],
            spec.maximum_ground_separation_m,
            diagnostics,
        )
        self.assertGreaterEqual(
            diagnostics["rear_contact_samples"] / diagnostics["powered_samples"],
            spec.minimum_powered_contact_coverage,
            diagnostics,
        )
        self.assertGreaterEqual(
            diagnostics["front_contact_samples"] / diagnostics["powered_samples"],
            spec.minimum_powered_contact_coverage,
            diagnostics,
        )


if __name__ == "__main__":
    unittest.main()
