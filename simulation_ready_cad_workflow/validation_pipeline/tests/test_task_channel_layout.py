from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    PlainRevoluteAdapter,
    PrismaticDetentLatchAdapter,
    mechanism_adapter_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_spec import (
    declared_input_wrench_direction_abs,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    build_task_channel_registry,
)


_PROVENANCE = (
    Path(__file__).resolve().parents[1] / "planning_templates" / "provenance"
)


def _load_json(*parts: str) -> dict:
    return json.loads((_PROVENANCE.joinpath(*parts)).read_text(encoding="utf-8"))


def _compound_plan(family_spec: dict, *, capsule_id: str) -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=family_spec,
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="auto_lock_utility_knife_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id=capsule_id,
        capsule_file=f"{capsule_id}.json",
        family="compound",
        protocol_id="prismatic_detent_latch",
        fixed_group_ids=("group_knife_handle",),
        moving_group_ids=("group_blade_carriage", "group_thumb_latch"),
        axis_abs=(1.0, 0.0, 0.0),
        joint_origin_abs_m=(-0.03, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_blade_carriage",
        input_port_id="thumb_press_and_slide",
        input_frame_id="frame_thumb_slider",
        input_point_abs_m=(-0.034, 0.0, 0.027),
        observable_id="blade_carriage_pose",
        observable_frame_id="frame_blade_carriage",
        observable_point_abs_m=(-0.03, 0.0, 0.0),
        task_ids=("extend_blade_to_working_position",),
        characteristic_length_m=0.2,
        moving_extents_m=(0.18, 0.03, 0.05),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _revolute_plan(
    family_spec: dict | None,
    *,
    package_id: str,
    capsule_id: str,
    fixed_group_ids: tuple[str, ...],
    moving_group_ids: tuple[str, ...],
    axis_abs: tuple[float, float, float],
    input_port_id: str,
    input_frame_id: str,
    observable_id: str,
    observable_frame_id: str,
) -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_PLAIN_REVOLUTE_ADAPTER_ID),
        family_spec=family_spec,
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id=package_id,
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id=capsule_id,
        capsule_file=f"{capsule_id}.json",
        family="revolute",
        protocol_id="revolute_pivot_hinge",
        fixed_group_ids=fixed_group_ids,
        moving_group_ids=moving_group_ids,
        axis_abs=axis_abs,
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id=observable_frame_id,
        input_port_id=input_port_id,
        input_frame_id=input_frame_id,
        input_point_abs_m=(0.0, 0.0, 0.0),
        observable_id=observable_id,
        observable_frame_id=observable_frame_id,
        observable_point_abs_m=(0.0, 0.0, 0.0),
        task_ids=("task.hinge",),
        characteristic_length_m=0.2,
        moving_extents_m=(0.18, 0.06, 0.04),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _knife_slide_capsule(
    family_spec: dict,
    *,
    direction_abs: list[float] | None = None,
) -> dict:
    wrench = declared_input_wrench_direction_abs(family_spec)
    return {
        "input_ports": [
            {
                "port_id": "thumb_press_and_slide",
                "rigid_group_id": "group_thumb_latch",
                "frame_id": "frame_thumb_slider",
                "wrench_type": "force",
                "direction_abs": (
                    list(direction_abs)
                    if direction_abs is not None
                    else list(wrench or [])
                ),
            }
        ],
        "observables": [
            {
                "observable_id": "blade_carriage_pose",
                "rigid_group_id": "group_blade_carriage",
                "frame_id": "frame_blade_carriage",
                "quantity": "pose_6d",
            }
        ],
    }


class DetentTaskChannelLayoutTests(unittest.TestCase):
    def test_demo_family_spec_binds_wrench_and_slide_axis(self) -> None:
        family_spec = _load_json("auto_lock_utility_knife_v1", "family_spec.json")
        plan = _compound_plan(
            family_spec,
            capsule_id="utility_knife_detent_slider",
        )
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        self.assertIsInstance(plugin, PrismaticDetentLatchAdapter)

        registry = build_task_channel_registry(
            plan,
            _load_json("auto_lock_utility_knife_v1", "capsule.json"),
        )
        bound = registry["inputs"]["thumb_press_and_slide"]
        slide_axis = [
            float(item)
            for item in family_spec["blade_prismatic_joint"]["axis_abs"]
        ]
        wrench = list(declared_input_wrench_direction_abs(family_spec))

        self.assertEqual(registry["plugin_id"], "prismatic_detent_latch_v1")
        self.assertEqual(bound["body_id"], "thumb_latch")
        self.assertEqual(bound["axis_abs"], wrench)
        self.assertNotEqual(bound["axis_abs"], slide_axis)
        self.assertEqual(
            registry["observables"]["blade_carriage_pose"]["body_id"],
            "blade_carriage",
        )
        self.assertEqual(
            registry["observables"]["blade_carriage_pose"]["canonical_field"],
            "blade_carriage_position_m",
        )
        self.assertEqual(
            registry["observables"]["blade_carriage_pose"]["axis_abs"],
            slide_axis,
        )
        self.assertEqual(registry["input_body_ids"], ["thumb_latch"])
        self.assertEqual(registry["output_body_ids"], ["blade_carriage"])

        inputs, outputs, fields, axes = plugin.task_channel_layout(
            plan,
            family_spec,
            ("handle", "blade_carriage", "thumb_latch"),
        )
        self.assertEqual(axes["thumb_latch"], tuple(wrench))
        self.assertEqual(axes["blade_carriage"], tuple(slide_axis))
        self.assertEqual(inputs, {"thumb_latch"})
        self.assertEqual(outputs, {"blade_carriage"})
        self.assertEqual(fields["blade_carriage"], "blade_carriage_position_m")

    def test_missing_blade_axis_fail_closes_binding(self) -> None:
        family_spec = _load_json("auto_lock_utility_knife_v1", "family_spec.json")
        del family_spec["blade_prismatic_joint"]["axis_abs"]
        plan = _compound_plan(
            family_spec,
            capsule_id="utility_knife_detent_slider",
        )
        capsule = _knife_slide_capsule(
            _load_json("auto_lock_utility_knife_v1", "family_spec.json")
        )

        with self.assertRaises(MechanismValidationError) as caught:
            build_task_channel_registry(plan, capsule)

        self.assertEqual(
            caught.exception.code,
            "TASK_CHANNEL_BINDING_INCOMPLETE",
        )

    def test_port_direction_mismatch_fail_closes_binding(self) -> None:
        family_spec = _load_json("auto_lock_utility_knife_v1", "family_spec.json")
        plan = _compound_plan(
            family_spec,
            capsule_id="utility_knife_detent_slider",
        )
        capsule = deepcopy(
            _load_json("auto_lock_utility_knife_v1", "capsule.json")
        )
        slide_axis = [
            float(item)
            for item in family_spec["blade_prismatic_joint"]["axis_abs"]
        ]
        wrench = list(declared_input_wrench_direction_abs(family_spec))
        self.assertNotEqual(slide_axis, wrench)
        capsule["input_ports"][0]["direction_abs"] = slide_axis

        with self.assertRaises(MechanismValidationError) as caught:
            build_task_channel_registry(plan, capsule)

        self.assertEqual(
            caught.exception.code,
            "TASK_CHANNEL_BINDING_INCOMPLETE",
        )

    def test_force_split_mismatch_fail_closes_binding(self) -> None:
        family_spec = _load_json("auto_lock_utility_knife_v1", "family_spec.json")
        family_spec["nominal_slide_force_n"] = (
            2.0 * family_spec["nominal_slide_force_n"]
        )
        plan = _compound_plan(
            family_spec,
            capsule_id="utility_knife_detent_slider",
        )
        capsule = _load_json("auto_lock_utility_knife_v1", "capsule.json")

        with self.assertRaises(MechanismValidationError) as caught:
            build_task_channel_registry(plan, capsule)

        self.assertEqual(
            caught.exception.code,
            "TASK_CHANNEL_BINDING_INCOMPLETE",
        )
        self.assertNotEqual(
            list(declared_input_wrench_direction_abs(family_spec)),
            capsule["input_ports"][0]["direction_abs"],
        )


class PlainRevoluteTaskChannelLayoutTests(unittest.TestCase):
    def test_tongs_demo_family_spec_binds_hand_squeeze_channel(self) -> None:
        family_spec = _load_json("spring_kitchen_tongs_v1", "family_spec.json")
        capsule = _load_json("spring_kitchen_tongs_v1", "capsule.json")
        claim = capsule["reduced_model_claim"]
        plan = _revolute_plan(
            family_spec,
            package_id="spring_kitchen_tongs_v1",
            capsule_id=capsule["capsule_id"],
            fixed_group_ids=tuple(claim["fixed_group_ids"]),
            moving_group_ids=tuple(claim["moving_group_ids"]),
            axis_abs=tuple(claim["axis_abs"]),
            input_port_id="hand_squeeze_torque",
            input_frame_id="frame_squeeze_input",
            observable_id="upper_jaw_pose",
            observable_frame_id="frame_upper_jaw",
        )
        plugin, _adapter = mechanism_adapter_for_plan(plan)
        self.assertIsInstance(plugin, PlainRevoluteAdapter)

        registry = build_task_channel_registry(plan, capsule)
        bound = registry["inputs"]["hand_squeeze_torque"]

        self.assertEqual(registry["plugin_id"], "plain_revolute_v1")
        self.assertEqual(bound["body_id"], "moving_group")
        self.assertEqual(bound["canonical_field"], "generalized_angle_rad")
        self.assertEqual(bound["axis_abs"], [0.0, 0.0, 1.0])
        self.assertEqual(
            registry["observables"]["upper_jaw_pose"]["body_id"],
            "moving_group",
        )
        self.assertEqual(registry["input_body_ids"], ["moving_group"])
        self.assertEqual(registry["output_body_ids"], ["moving_group"])

    def test_toolbox_lid_binding_stays_single_hinge_moving_group(self) -> None:
        capsule = _load_json("hinged_toolbox_v1", "capsule.json")
        claim = capsule["reduced_model_claim"]
        plan = _revolute_plan(
            None,
            package_id="hinged_toolbox_v1",
            capsule_id=capsule["capsule_id"],
            fixed_group_ids=tuple(claim["fixed_group_ids"]),
            moving_group_ids=tuple(claim["moving_group_ids"]),
            axis_abs=tuple(claim["axis_abs"]),
            input_port_id="lid_hinge_torque",
            input_frame_id="frame_lid_torque_input",
            observable_id="lid_pose",
            observable_frame_id="frame_lid_front_observable",
        )

        registry = build_task_channel_registry(plan, capsule)
        bound = registry["inputs"]["lid_hinge_torque"]

        self.assertEqual(registry["plugin_id"], "plain_revolute_v1")
        self.assertEqual(bound["body_id"], "moving_group")
        self.assertEqual(bound["canonical_field"], "generalized_angle_rad")
        self.assertEqual(bound["axis_abs"], [0.0, -1.0, 0.0])
        self.assertEqual(
            registry["observables"]["lid_pose"]["canonical_field"],
            "generalized_angle_rad",
        )

    def test_plain_revolute_port_direction_mismatch_fail_closes(self) -> None:
        family_spec = _load_json("spring_kitchen_tongs_v1", "family_spec.json")
        capsule = deepcopy(
            _load_json("spring_kitchen_tongs_v1", "capsule.json")
        )
        capsule["input_ports"][0]["direction_abs"] = [1.0, 0.0, 0.0]
        claim = capsule["reduced_model_claim"]
        plan = _revolute_plan(
            family_spec,
            package_id="spring_kitchen_tongs_v1",
            capsule_id=capsule["capsule_id"],
            fixed_group_ids=tuple(claim["fixed_group_ids"]),
            moving_group_ids=tuple(claim["moving_group_ids"]),
            axis_abs=tuple(claim["axis_abs"]),
            input_port_id="hand_squeeze_torque",
            input_frame_id="frame_squeeze_input",
            observable_id="upper_jaw_pose",
            observable_frame_id="frame_upper_jaw",
        )

        with self.assertRaises(MechanismValidationError) as caught:
            build_task_channel_registry(plan, capsule)

        self.assertEqual(
            caught.exception.code,
            "TASK_CHANNEL_BINDING_INCOMPLETE",
        )

    def test_typed_layout_matches_single_hinge_fallback(self) -> None:
        family_spec = _load_json("spring_kitchen_tongs_v1", "family_spec.json")
        capsule = _load_json("spring_kitchen_tongs_v1", "capsule.json")
        claim = capsule["reduced_model_claim"]
        plan = _revolute_plan(
            family_spec,
            package_id="spring_kitchen_tongs_v1",
            capsule_id=capsule["capsule_id"],
            fixed_group_ids=tuple(claim["fixed_group_ids"]),
            moving_group_ids=tuple(claim["moving_group_ids"]),
            axis_abs=tuple(claim["axis_abs"]),
            input_port_id="hand_squeeze_torque",
            input_frame_id="frame_squeeze_input",
            observable_id="upper_jaw_pose",
            observable_frame_id="frame_upper_jaw",
        )
        plugin = PlainRevoluteAdapter()
        inputs, outputs, fields, axes = plugin.task_channel_layout(
            plan,
            family_spec,
            (),
        )

        self.assertEqual(inputs, {"moving_group"})
        self.assertEqual(outputs, {"moving_group"})
        self.assertEqual(fields["moving_group"], "generalized_angle_rad")
        self.assertEqual(axes["moving_group"], tuple(claim["axis_abs"]))


if __name__ == "__main__":
    unittest.main()
