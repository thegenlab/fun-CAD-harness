from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import replace
import json
import math
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    IdealReferenceError,
    generate_ideal_reference_trace,
    materialize_ideal_references,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismPlan,
)


_HASH = "0" * 64
_FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _plan(
    *,
    family: str,
    adapter_id: str,
    adapter_spec: dict,
) -> MechanismPlan:
    return MechanismPlan(
        package_dir="package",
        intake_report="intake.json",
        intake_report_sha256=_HASH,
        package_id="ideal_reference_test",
        design_version="1.0.0",
        requirements_sha256=_HASH,
        package_binding_sha256=_HASH,
        capsule_id="test_capsule",
        capsule_file="capsule.json",
        family=family,
        protocol_id="test_protocol",
        fixed_group_ids=("fixed",),
        moving_group_ids=("moving",),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="joint",
        input_port_id="input",
        input_frame_id="input_frame",
        input_point_abs_m=(0.0, 0.0, 0.0),
        observable_id="output",
        observable_frame_id="output_frame",
        observable_point_abs_m=(0.0, 0.0, 0.0),
        task_ids=("task",),
        characteristic_length_m=0.2,
        moving_extents_m=(0.1, 0.1, 0.1),
        adapter_id=adapter_id,
        adapter_spec=adapter_spec,
    )


def _load_case(family: str) -> LoadCase:
    return LoadCase(
        load_case_id="ideal_case",
        family=family,
        program="test_program",
        purpose="nominal",
        phases=(
            LoadPhase(
                phase_id="drive",
                wrench_kind="torque" if family != "prismatic" else "force",
                direction_abs=(0.0, 0.0, 1.0),
                magnitude_si=1.0,
                duration_s=1.0,
            ),
        ),
        moving_mass_kg=1.0,
        step_s=0.001,
        sample_interval_s=0.05,
    )


def _physical(
    plan: MechanismPlan,
    load_case: LoadCase,
    samples: list[dict],
) -> dict:
    return {
        "status": "COMPLETED",
        "branch": "physical",
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": canonical_sha256(plan.to_dict()),
        "load_case_sha256": canonical_sha256(load_case.to_dict()),
        "samples": samples,
    }


def _compound_plan(spec_path: Path) -> tuple[MechanismPlan, dict]:
    family_spec = json.loads(spec_path.read_text(encoding="utf-8"))
    return _compound_plan_value(family_spec), family_spec


def _compound_plan_value(family_spec: dict) -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=family_spec,
    )
    return _plan(
        family="compound",
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _isolated_clock_spec(family_spec: dict) -> dict:
    value = deepcopy(family_spec)
    if value.get("alarm_input_body_id") is not None:
        return value
    alarm_origin = [-0.072, 0.038, 0.140]
    value["bodies"].append(
        {
            "body_id": "alarm_barrel",
            "group_ids": ["group_alarm_barrel"],
            "fixed": False,
            "mass_kg": 0.065,
            "center_abs_m": alarm_origin,
            "size_m": [0.038, 0.018, 0.038],
        }
    )
    value["shafts"].append(
        {
            "shaft_id": "shaft_alarm_barrel",
            "body_id": "alarm_barrel",
            "origin_abs_m": alarm_origin,
            "axis_abs": [0.0, 1.0, 0.0],
        }
    )
    value["gear_meshes"] = [
        mesh
        for mesh in value["gear_meshes"]
        if mesh["mesh_id"] != "minute_alarm"
    ]
    value["gear_meshes"].append(
        {
            "mesh_id": "alarm_barrel_cam",
            "driver_body_id": "alarm_barrel",
            "driven_body_id": "alarm_cam",
            "driver_teeth": 12,
            "driven_teeth": 68,
            "mesh_stiffness_nm_rad": 280.0,
            "mesh_damping_nm_s_rad": 0.05,
        }
    )
    value["alarm_input_body_id"] = "alarm_barrel"
    value["alarm_input_point_abs_m"] = alarm_origin
    value["alarm_input_axis_abs"] = [0.0, 1.0, 0.0]
    value["nominal_alarm_input_torque_nm"] = 0.09
    return value


class IdealReferenceTests(unittest.TestCase):
    def test_plain_reference_has_no_simulator_or_collision_dependency(self) -> None:
        adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
        plan = _plan(
            family="prismatic",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        load_case = _load_case("prismatic")
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "generalized_progress_m": 0.0,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "generalized_progress_m": 0.04,
                },
            ],
        )
        real_import = builtins.__import__

        def guarded_import(name: str, *args: object, **kwargs: object):
            if name == "pychrono" or name.startswith("pychrono."):
                raise AssertionError("ideal reference imported PyChrono")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import):
            trace = generate_ideal_reference_trace(
                plan,
                load_case,
                physical,
            )

        self.assertEqual(trace["branch"], "ideal_reference")
        self.assertFalse(trace["engine"]["time_integration"])
        self.assertFalse(trace["engine"]["collision_geometry_loaded"])
        self.assertEqual(trace["engine"]["internal_dynamics_steps"], 0)
        self.assertEqual(
            trace["samples"][-1]["generalized_progress_m"],
            0.04,
        )

    def test_rack_pinion_uses_pitch_radius_relation(self) -> None:
        spec_path = _FIXTURES / "ideal_reference_rack_pinion.family.json"
        plan, spec = _compound_plan(spec_path)
        load_case = _load_case("compound")
        radius = float(spec["pinion_pitch_radius_m"])
        travel = float(spec["outbound_progress_m"])
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "pinion_angle_rad": 0.0,
                    "rack_progress_m": 0.0,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "pinion_angle_rad": -travel / radius,
                    "rack_progress_m": travel * 0.97,
                },
            ],
        )

        trace = generate_ideal_reference_trace(
            plan,
            load_case,
            physical,
        )

        self.assertAlmostEqual(
            trace["samples"][-1]["rack_progress_m"],
            travel,
        )
        self.assertEqual(
            trace["samples"][-1]["coupling_error_m"],
            0.0,
        )
        self.assertEqual(
            trace["domain"]["kind"],
            "closed_form_rack_pinion",
        )

    def test_rack_pinion_carries_the_commanded_input_record(self) -> None:
        spec_path = _FIXTURES / "ideal_reference_rack_pinion.family.json"
        plan, spec = _compound_plan(spec_path)
        load_case = _load_case("compound")
        radius = float(spec["pinion_pitch_radius_m"])
        travel = float(spec["outbound_progress_m"])
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "pinion_angle_rad": 0.0,
                    "rack_progress_m": 0.0,
                    "applied_torque_abs_nm": [0.0, 0.0, 0.0],
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "pinion_angle_rad": -travel / radius,
                    "rack_progress_m": travel * 0.97,
                    "applied_torque_abs_nm": [0.0, -3.2, 0.0],
                },
            ],
        )

        trace = generate_ideal_reference_trace(
            plan,
            load_case,
            physical,
        )

        self.assertEqual(
            trace["samples"][0]["applied_torque_abs_nm"],
            [0.0, 0.0, 0.0],
        )
        self.assertEqual(
            trace["samples"][-1]["applied_torque_abs_nm"],
            [0.0, -3.2, 0.0],
        )

    def test_planar_four_bar_uses_declared_pivot_circle_closure(self) -> None:
        spec_path = _FIXTURES / "ideal_reference_four_bar.family.json"
        plan, spec = _compound_plan(spec_path)
        load_case = _load_case("compound")
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "compound_input_angle_rad": 0.0,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "compound_input_angle_rad": 0.5,
                },
            ],
        )

        trace = generate_ideal_reference_trace(
            plan,
            load_case,
            physical,
        )

        self.assertAlmostEqual(
            trace["samples"][0]["compound_output_angle_rad"],
            float(spec["home_output_angle_rad"]),
            places=9,
        )
        self.assertGreater(
            trace["samples"][1]["compound_output_angle_rad"],
            trace["samples"][0]["compound_output_angle_rad"],
        )
        self.assertEqual(
            trace["domain"]["kind"],
            "declared_planar_four_bar_circle_closure",
        )
        self.assertFalse(trace["engine"]["time_integration"])
        self.assertFalse(trace["engine"]["collision_enabled"])

    def test_clock_reference_uses_declared_gear_graph(self) -> None:
        spec_path = _FIXTURES / "ideal_reference_clock.family.json"
        plan, spec = _compound_plan(spec_path)
        load_case = _load_case("compound")
        input_body = str(spec["input_body_id"])
        alarm_input_body = str(spec["alarm_input_body_id"])
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "active_input_body_id": input_body,
                    "setting_clutch_engaged": False,
                    "shaft_angles_rad": {
                        input_body: 0.0,
                        alarm_input_body: 0.0,
                    },
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "active_input_body_id": input_body,
                    "setting_clutch_engaged": False,
                    "shaft_angles_rad": {
                        input_body: 6.0,
                        alarm_input_body: 0.0,
                    },
                },
            ],
        )

        trace = generate_ideal_reference_trace(
            plan,
            load_case,
            physical,
        )
        initial = trace["samples"][0]
        final = trace["samples"][-1]
        minute = (
            final["clock_output_angle_rad"]
            - initial["clock_output_angle_rad"]
        )
        hour = (
            final["hour_hand_angle_rad"]
            - initial["hour_hand_angle_rad"]
        )
        second = (
            final["second_hand_angle_rad"]
            - initial["second_hand_angle_rad"]
        )

        self.assertNotEqual(minute, 0.0)
        tolerance = float(spec["hand_ratio_relative_tolerance"])
        self.assertLessEqual(
            abs(
                (hour / minute)
                / float(spec["expected_hour_to_minute_ratio"])
                - 1.0
            ),
            tolerance,
        )
        self.assertLessEqual(
            abs(
                (second / minute)
                / float(spec["expected_second_to_minute_ratio"])
                - 1.0
            ),
            tolerance,
        )
        self.assertTrue(
            all(value == 0.0 for value in final["mesh_errors_rad"].values())
        )

    def test_isolated_clock_reference_evaluates_both_declared_input_graphs(self) -> None:
        spec_path = _FIXTURES / "ideal_reference_clock.family.json"
        spec = _isolated_clock_spec(
            json.loads(spec_path.read_text(encoding="utf-8"))
        )
        plan = _compound_plan_value(spec)
        load_case = _load_case("compound")
        main_input = str(spec["input_body_id"])
        alarm_input = str(spec["alarm_input_body_id"])
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "active_input_body_id": main_input,
                    "setting_clutch_engaged": False,
                    "shaft_angles_rad": {main_input: 0.0, alarm_input: 0.0},
                },
                {
                    "time_s": 1.0,
                    "phase_id": "concurrent_timekeeping_and_alarm",
                    "active_input_body_id": main_input,
                    "setting_clutch_engaged": False,
                    "shaft_angles_rad": {main_input: 6.0, alarm_input: 1.2},
                },
            ],
        )

        trace = generate_ideal_reference_trace(plan, load_case, physical)
        initial, final = trace["samples"]

        self.assertNotEqual(
            final["clock_output_angle_rad"],
            initial["clock_output_angle_rad"],
        )
        self.assertNotEqual(final["alarm_angle_rad"], initial["alarm_angle_rad"])
        self.assertEqual(
            trace["domain"]["domain_check"],
            "all_required_outputs_reachable_from_declared_independent_inputs",
        )

    def test_isolated_clock_reference_fails_if_alarm_input_coordinate_is_missing(self) -> None:
        spec_path = _FIXTURES / "ideal_reference_clock.family.json"
        spec = _isolated_clock_spec(
            json.loads(spec_path.read_text(encoding="utf-8"))
        )
        plan = _compound_plan_value(spec)
        load_case = _load_case("compound")
        main_input = str(spec["input_body_id"])
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "active_input_body_id": main_input,
                    "setting_clutch_engaged": False,
                    "shaft_angles_rad": {main_input: 0.0},
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "active_input_body_id": main_input,
                    "setting_clutch_engaged": False,
                    "shaft_angles_rad": {main_input: 1.0},
                },
            ],
        )

        with self.assertRaises(IdealReferenceError) as caught:
            generate_ideal_reference_trace(plan, load_case, physical)

        self.assertEqual(caught.exception.code, "IDEAL_REFERENCE_INPUT_INVALID")

    def test_scissor_reference_maps_declared_cylinder_domain(self) -> None:
        spec_path = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "scissor_lift_drive.family.json"
        )
        plan, spec = _compound_plan(spec_path)
        load_case = _load_case("compound")
        reference = spec["geometry_reference"]
        home = float(reference["cylinder_home_length_m"])
        outbound = float(reference["cylinder_outbound_length_m"])
        midpoint = (home + outbound) * 0.5
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "cylinder_length_m": home,
                    "platform_height_m": 0.2,
                },
                {
                    "time_s": 0.5,
                    "phase_id": "drive",
                    "cylinder_length_m": midpoint,
                    "platform_height_m": 0.3,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "cylinder_length_m": outbound,
                    "platform_height_m": 0.38,
                },
            ],
        )

        trace = generate_ideal_reference_trace(
            plan,
            load_case,
            physical,
        )

        self.assertAlmostEqual(
            trace["samples"][-1]["platform_progress_m"],
            float(spec["outbound_progress_m"]),
        )
        linear_midpoint = float(spec["outbound_progress_m"]) * 0.5
        self.assertGreater(
            trace["samples"][1]["platform_progress_m"],
            linear_midpoint + 0.005,
        )
        self.assertEqual(
            trace["domain"]["kind"],
            "declared_scissor_linkage_closure",
        )
        self.assertEqual(
            trace["domain"]["domain_status"],
            "PASS",
        )
        self.assertTrue(
            trace["branch_semantics"]["ideal_body_poses_computed"]
        )
        self.assertFalse(
            trace["branch_semantics"]["physical_body_poses_copied"]
        )
        body_ids = {body["body_id"] for body in spec["bodies"]}
        self.assertEqual(
            set(trace["samples"][0]["body_poses"]),
            body_ids,
        )
        bodies = {body["body_id"]: body for body in spec["bodies"]}
        output_id = str(spec["output_body_id"])
        output_axis = spec["output_axis_abs"]
        initial_output = trace["samples"][0]["body_poses"][output_id][
            "position_abs_m"
        ]
        final_output = trace["samples"][-1]["body_poses"][output_id][
            "position_abs_m"
        ]
        for index in range(3):
            self.assertAlmostEqual(
                final_output[index] - initial_output[index],
                float(output_axis[index])
                * float(spec["outbound_progress_m"]),
                places=9,
            )
        for body_id, body in bodies.items():
            self.assertEqual(
                trace["samples"][0]["body_poses"][body_id]["position_abs_m"],
                body["center_abs_m"],
            )

    def test_materializer_writes_light_trace_without_queue_job(self) -> None:
        adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
        plan = _plan(
            family="prismatic",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        load_case = _load_case("prismatic")
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            output = workspace / "run"
            plan_path = output / "plans" / "test_capsule.json"
            load_path = output / "loads" / "test_capsule" / "ideal_case.json"
            physical_path = (
                output
                / "traces"
                / "test_capsule"
                / "ideal_case"
                / "physical.trace.json"
            )
            write_json_atomic(plan_path, plan.to_dict())
            write_json_atomic(load_path, load_case.to_dict())
            write_json_atomic(
                physical_path,
                _physical(
                    plan,
                    load_case,
                    [
                        {
                            "time_s": 0.0,
                            "phase_id": "initial",
                            "generalized_progress_m": 0.0,
                        },
                        {
                            "time_s": 1.0,
                            "phase_id": "drive",
                            "generalized_progress_m": 0.02,
                        },
                    ],
                ),
            )
            write_json_atomic(
                output / "dispatch_summary.json",
                {"plan_files": ["run/plans/test_capsule.json"]},
            )

            summary = materialize_ideal_references(
                workspace_root=workspace,
                output_dir=output,
            )
            ideal_path = physical_path.with_name(
                "ideal_reference.trace.json"
            )

            self.assertTrue(ideal_path.is_file())
            self.assertEqual(summary["status"], "PASS")
            self.assertEqual(
                summary["engine_policy"]["pychrono_queue_jobs"],
                0,
            )
    def test_unsupported_compound_fails_instead_of_falling_back(self) -> None:
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec={
                "mechanism_kind": "revolute_linkage",
            },
        )
        plan = _plan(
            family="compound",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        load_case = _load_case("compound")
        physical = _physical(
            plan,
            load_case,
            [
                {"time_s": 0.0, "phase_id": "initial"},
                {"time_s": 1.0, "phase_id": "drive"},
            ],
        )

        with self.assertRaises(IdealReferenceError) as caught:
            generate_ideal_reference_trace(
                plan,
                load_case,
                physical,
            )

        self.assertEqual(
            caught.exception.code,
            "IDEAL_REFERENCE_UNSUPPORTED",
        )

    def test_physical_trace_identity_is_hash_bound(self) -> None:
        adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
        plan = _plan(
            family="prismatic",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        load_case = _load_case("prismatic")
        physical = _physical(
            plan,
            load_case,
            [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "generalized_progress_m": 0.0,
                },
                {
                    "time_s": 1.0,
                    "phase_id": "drive",
                    "generalized_progress_m": 0.01,
                },
            ],
        )
        physical["plan_sha256"] = "f" * 64

        with self.assertRaises(IdealReferenceError) as caught:
            generate_ideal_reference_trace(
                plan,
                load_case,
                physical,
            )

        self.assertEqual(
            caught.exception.code,
            "IDEAL_REFERENCE_PHYSICAL_TRACE_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
