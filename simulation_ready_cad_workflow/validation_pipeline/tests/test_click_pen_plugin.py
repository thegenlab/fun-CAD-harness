from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.click_pen_comparison import (
    physical_click_pen_behavior,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.click_pen_spec import (
    click_pen_cam_latch_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    mechanism_adapter_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    generate_ideal_reference_trace,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import (
    default_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)


def _family_spec() -> dict:
    return {
        "mechanism_kind": "click_pen_cam_latch",
        "mechanism_id": "test_click_pen",
        "bodies": [
            {
                "body_id": "barrel",
                "group_ids": ["group_barrel"],
                "fixed": True,
                "mass_kg": 0.018,
                "center_abs_m": [0.0, 0.0, 0.0],
                "size_m": [0.03, 0.03, 0.18],
            },
            {
                "body_id": "refill",
                "group_ids": ["group_refill"],
                "fixed": False,
                "mass_kg": 0.006,
                "center_abs_m": [0.0, 0.0, -0.01],
                "size_m": [0.01, 0.01, 0.16],
            },
            {
                "body_id": "button",
                "group_ids": ["group_button"],
                "fixed": False,
                "mass_kg": 0.003,
                "center_abs_m": [0.0, 0.0, 0.078],
                "size_m": [0.015, 0.015, 0.03],
            },
            {
                "body_id": "cam",
                "group_ids": ["group_cam"],
                "fixed": False,
                "mass_kg": 0.003,
                "center_abs_m": [0.0, 0.0, 0.053],
                "size_m": [0.018, 0.018, 0.026],
            },
        ],
        "button_prismatic_joint": {
            "joint_id": "button_press",
            "parent_body_id": "barrel",
            "child_body_id": "button",
            "origin_abs_m": [0.0, 0.0, 0.068],
            "axis_abs": [0.0, 0.0, -1.0],
            "travel_m": 0.012,
            "spring_stiffness_n_m": 180.0,
            "damping_n_s_m": 2.0,
            "spring_preload_n": 0.6,
        },
        "refill_prismatic_joint": {
            "joint_id": "refill_slide",
            "parent_body_id": "barrel",
            "child_body_id": "refill",
            "origin_abs_m": [0.0, 0.0, -0.01],
            "axis_abs": [0.0, 0.0, -1.0],
            "retracted_position_m": 0.0,
            "extended_position_m": 0.008,
            "spring_stiffness_n_m": 520.0,
            "damping_n_s_m": 1.2,
            "spring_preload_n": 1.8,
        },
        "cam_cylindrical_joint": {
            "joint_id": "cam_index",
            "parent_body_id": "barrel",
            "child_body_id": "cam",
            "origin_abs_m": [0.0, 0.0, 0.055],
            "axis_abs": [0.0, 0.0, 1.0],
            "axial_travel_m": 0.010,
            "index_angle_rad": 1.0471975511965976,
        },
        "click_state_machine": {
            "states": ["retracted", "extended_latched"],
            "initial_state": "retracted",
            "first_click_target_m": 0.008,
            "second_click_target_m": 0.0,
            "latch_contact_tolerance_m": 0.0008,
            "release_button_return_m": 0.010,
            "geometry_contract": {
                "button_contact_component_refs": [
                    "button/solid_20",
                    "button/solid_21",
                ],
                "cam_button_face_component_refs": [
                    "cam/solid_30",
                    "cam/solid_31",
                ],
                "cam_follower_component_ref": "cam/solid_29",
                "refill_carrier_component_ref": "refill/solid_18",
                "barrel_guide_component_refs": [
                    "barrel/solid_10",
                    "barrel/solid_11",
                ],
                "cam_lobe_component_refs": [
                    "cam/solid_32",
                    "cam/solid_33",
                ],
                "home_contact_tolerance_m": 0.0002,
                "full_press_offset_abs_m": [0.0, 0.0, -0.012],
                "extended_latched_cam_offset_abs_m": [0.0, 0.0, -0.008],
                "extended_latched_refill_offset_abs_m": [0.0, 0.0, -0.008],
            },
        },
        "input_body_id": "button",
        "output_body_id": "refill",
        "nominal_press_force_n": 8.0,
        "max_support_drift_m": 0.0008,
        "max_off_axis_rotation_rad": 0.04,
        "friction": 0.24,
    }


def _plan() -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=_family_spec(),
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="click_pen_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="click_pen",
        capsule_file="click_pen.json",
        family="compound",
        protocol_id="click_pen_cam_latch",
        fixed_group_ids=("group_barrel",),
        moving_group_ids=("group_refill", "group_button", "group_cam"),
        axis_abs=(0.0, 0.0, -1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.068),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_button",
        input_port_id="top_button_force",
        input_frame_id="frame_button",
        input_point_abs_m=(0.0, 0.0, 0.084),
        observable_id="refill_tip_pose",
        observable_frame_id="frame_refill",
        observable_point_abs_m=(0.0, 0.0, -0.09),
        task_ids=("first_click", "second_click", "repeat"),
        characteristic_length_m=0.19,
        moving_extents_m=(0.018, 0.018, 0.17),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


class ClickPenPluginTests(unittest.TestCase):
    def test_plugin_owns_six_machine_specific_loads(self) -> None:
        plan = _plan()
        plugin, adapter = mechanism_adapter_for_plan(plan)
        loads = default_load_cases(plan)

        self.assertEqual(plugin.plugin_id, "click_pen_cam_latch_v1")
        self.assertEqual(
            plugin.expected_reduced_ideal_links(plan, adapter),
            (
                "ChLinkLockPrismatic",
                "ChLinkLockPrismatic",
                "ChLinkLockCylindrical",
            ),
        )
        self.assertEqual(len(loads), 6)
        self.assertEqual(
            {case.program for case in loads},
            {
                "pen_first_click",
                "pen_two_click_cycle",
                "zero_input_click_pen",
                "task_pen_first_click",
                "task_pen_second_click",
                "task_pen_repeat_cycles",
            },
        )
        repeat = next(
            case for case in loads if case.program == "task_pen_repeat_cycles"
        )
        self.assertEqual(
            [phase.phase_id for phase in repeat.phases if phase.wrench_kind == "none"],
            [
                "click_1_release",
                "click_2_release",
                "click_3_release",
                "click_4_release",
            ],
        )
        self.assertTrue(
            all(
                phase.direction_abs in {(0.0, 0.0, -1.0), None}
                for phase in repeat.phases
            )
        )

    def test_parser_binds_contact_surfaces_to_correct_bodies(self) -> None:
        spec = click_pen_cam_latch_from_dict(_family_spec())
        self.assertEqual(spec.input_body_id, "button")
        self.assertEqual(spec.output_body_id, "refill")
        self.assertEqual(click_pen_cam_latch_from_dict(spec.to_dict()), spec)

        wrong_owner = deepcopy(_family_spec())
        wrong_owner["click_state_machine"]["geometry_contract"][
            "cam_follower_component_ref"
        ] = "button/solid_29"
        with self.assertRaisesRegex(
            MechanismValidationError,
            "CLICK_PEN_GEOMETRY_BODY_MISMATCH",
        ):
            click_pen_cam_latch_from_dict(wrong_owner)

        out_of_range = deepcopy(_family_spec())
        out_of_range["click_state_machine"]["first_click_target_m"] = 0.02
        with self.assertRaisesRegex(
            MechanismValidationError,
            "CLICK_PEN_STATE_RANGE_INVALID",
        ):
            click_pen_cam_latch_from_dict(out_of_range)

    def test_ideal_reference_keeps_four_poses_and_alternates_state(self) -> None:
        plan = _plan()
        load_case = next(
            case
            for case in default_load_cases(plan)
            if case.program == "task_pen_second_click"
        )
        physical = {
            "status": "COMPLETED",
            "branch": "physical",
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "load_case_sha256": canonical_sha256(load_case.to_dict()),
            "samples": [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "button_press_depth_m": 0.0,
                },
                {
                    "time_s": 0.4,
                    "phase_id": "click_1_release",
                    "button_press_depth_m": 0.0,
                },
                {
                    "time_s": 0.8,
                    "phase_id": "click_2_release",
                    "button_press_depth_m": 0.0,
                },
            ],
        }
        ideal = generate_ideal_reference_trace(plan, load_case, physical)

        self.assertAlmostEqual(ideal["samples"][1]["refill_extension_m"], 0.008)
        self.assertTrue(ideal["samples"][1]["extended_latched"])
        self.assertAlmostEqual(ideal["samples"][2]["refill_extension_m"], 0.0)
        self.assertFalse(ideal["samples"][2]["extended_latched"])
        self.assertEqual(
            set(ideal["samples"][1]["body_poses"]),
            {"barrel", "refill", "button", "cam"},
        )
        self.assertAlmostEqual(
            ideal["samples"][2]["cam_index_angle_rad"],
            2.0 * 1.0471975511965976,
        )

    def test_zero_input_physical_behavior_rejects_ideal_joints(self) -> None:
        plan = _plan()
        load_case = next(
            case
            for case in default_load_cases(plan)
            if case.program == "zero_input_click_pen"
        )
        poses = {
            body_id: {
                "position_abs_m": [0.0, 0.0, 0.0],
                "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
            for body_id in ("barrel", "refill", "button", "cam")
        }
        support_drifts = {
            "button_press": 0.0,
            "refill_slide": 0.0,
            "cam_index": 0.0,
        }
        sample = {
            "time_s": 0.0,
            "phase_id": "initial",
            "button_press_depth_m": 0.0,
            "button_speed_m_s": 0.0,
            "refill_extension_m": 0.0,
            "refill_speed_m_s": 0.0,
            "cam_axial_position_m": 0.0,
            "cam_index_angle_rad": 0.0,
            "extended_latched": False,
            "latch_contact_count": 0,
            "max_support_drift_m": 0.0,
            "support_drifts_m": support_drifts,
            "max_off_axis_rotation_rad": 0.0,
            "body_poses": poses,
        }
        trace = {
            "status": "COMPLETED",
            "branch": "physical",
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "plan_sha256": canonical_sha256(plan.to_dict()),
            "load_case_sha256": canonical_sha256(load_case.to_dict()),
            "engine": {"internal_ideal_joint_count": 0},
            "branch_semantics": {
                "direct_output_actuation": False,
                "button_is_only_actuated_body": True,
                "cam_index_torque_is_analytical": False,
                "latch_force_is_analytical": False,
            },
            "samples": [
                sample,
                dict(sample, time_s=0.35, phase_id="zero_input"),
            ],
        }
        metrics, _thresholds, failures = physical_click_pen_behavior(
            plan,
            load_case,
            trace,
        )
        self.assertEqual(failures, [])
        self.assertEqual(metrics["final_refill_extension_m"], 0.0)

        trace["engine"]["internal_ideal_joint_count"] = 1
        _metrics, _thresholds, failures = physical_click_pen_behavior(
            plan,
            load_case,
            trace,
        )
        self.assertIn("physical_click_pen_contains_ideal_joint", failures)


if __name__ == "__main__":
    unittest.main()
