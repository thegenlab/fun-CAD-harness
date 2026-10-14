from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_comparison import (
    physical_detent_slider_behavior,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_runtime import (
    executed_detent_slider_phase,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.detent_slider_spec import (
    prismatic_detent_latch_from_dict,
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
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    task_program_spec_for_load_case,
    validate_trace_program,
)


def _family_spec() -> dict:
    wall_x = (-0.016, -0.002, 0.009, 0.023, 0.029, 0.043, 0.049, 0.063)
    return {
        "mechanism_kind": "prismatic_detent_latch",
        "mechanism_id": "test_detent_slider",
        "bodies": [
            {
                "body_id": "handle",
                "group_ids": ["group_handle"],
                "fixed": True,
                "mass_kg": 0.16,
                "center_abs_m": [0.0, 0.0, 0.0],
                "size_m": [0.2, 0.04, 0.04],
            },
            {
                "body_id": "blade",
                "group_ids": ["group_blade"],
                "fixed": False,
                "mass_kg": 0.05,
                "center_abs_m": [0.02, 0.0, 0.0],
                "size_m": [0.15, 0.02, 0.02],
            },
            {
                "body_id": "latch",
                "group_ids": ["group_latch"],
                "fixed": False,
                "mass_kg": 0.01,
                "center_abs_m": [-0.009, 0.008, 0.0125],
                "size_m": [0.05, 0.02, 0.03],
            },
        ],
        "blade_prismatic_joint": {
            "joint_id": "blade_slide",
            "parent_body_id": "handle",
            "child_body_id": "blade",
            "origin_abs_m": [0.0, 0.0, 0.0],
            "axis_abs": [1.0, 0.0, 0.0],
            "min_position_m": 0.0,
            "max_position_m": 0.065,
        },
        "latch_press_joint": {
            "joint_id": "latch_press",
            "parent_body_id": "blade",
            "child_body_id": "latch",
            "origin_abs_m": [-0.009, 0.0, 0.0125],
            "axis_abs": [0.0, 0.0, -1.0],
            "min_position_m": -0.00015,
            "travel_m": 0.004,
            "spring_stiffness_n_m": 900.0,
            "damping_n_s_m": 1.8,
            "spring_preload_n": 1.0,
        },
        "detent_interface": {
            "rack_body_id": "handle",
            "tab_body_id": "latch",
            "positions_m": [0.0, 0.025, 0.045, 0.065],
            "capture_tolerance_m": 0.0015,
            "release_press_m": 0.003,
            "holding_force_n": 45.0,
            "geometry_contract": {
                "rack_component_refs": [
                    f"handle/solid_{index}" for index in range(1, 9)
                ],
                "seat_component_refs": [
                    f"handle/solid_{index}" for index in range(9, 13)
                ],
                "tab_component_ref": "latch/solid_13",
                "rack_tooth_centers_abs_m": [
                    [x, 0.008, 0.0155] for x in wall_x
                ],
                "rack_tooth_size_m": [0.007, 0.002, 0.007],
                "seat_centers_abs_m": [
                    [x, 0.008, 0.0156]
                    for x in (-0.009, 0.016, 0.036, 0.056)
                ],
                "seat_size_m": [0.007, 0.0023, 0.001],
                "tab_home_center_abs_m": [-0.009, 0.008, 0.0125],
                "tab_size_m": [0.007, 0.002, 0.005],
                "pressed_offset_abs_m": [0.0, 0.0, -0.004],
                "contact_tolerance_m": 0.00015,
                "minimum_pressed_clearance_m": 0.001,
            },
        },
        "input_body_id": "latch",
        "output_body_id": "blade",
        "nominal_press_force_n": 8.0,
        "nominal_slide_force_n": 15.0,
        "adversarial_axial_force_n": 30.0,
        "max_support_drift_m": 0.001,
        "max_off_axis_rotation_rad": 0.04,
        "friction": 0.32,
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
        package_id="detent_slider_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="detent_slider",
        capsule_file="detent_slider.json",
        family="compound",
        protocol_id="prismatic_detent_latch",
        fixed_group_ids=("group_handle",),
        moving_group_ids=("group_blade", "group_latch"),
        axis_abs=(1.0, 0.0, 0.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_blade",
        input_port_id="press_and_slide",
        input_frame_id="frame_latch",
        input_point_abs_m=(-0.009, 0.0, 0.02),
        observable_id="blade_pose",
        observable_frame_id="frame_blade",
        observable_point_abs_m=(0.0, 0.0, 0.0),
        task_ids=("extend_blade",),
        characteristic_length_m=0.2,
        moving_extents_m=(0.2, 0.04, 0.04),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


class DetentSliderPluginTests(unittest.TestCase):
    def test_plugin_owns_eight_machine_specific_loads(self) -> None:
        plan = _plan()
        plugin, adapter = mechanism_adapter_for_plan(plan)
        loads = default_load_cases(plan)

        self.assertEqual(plugin.plugin_id, "prismatic_detent_latch_v1")
        self.assertEqual(
            plugin.expected_reduced_ideal_links(plan, adapter),
            ("ChLinkLockPrismatic", "ChLinkLockPrismatic"),
        )
        self.assertEqual(len(loads), 8)
        self.assertEqual(
            {case.program for case in loads},
            {
                "select_detent_position",
                "hold_detent_under_axial_load",
                "zero_input_detent_slider",
                "task_detent_extend",
                "task_detent_hold_under_load",
                "task_detent_second_position",
                "task_detent_cycle",
            },
        )
        self.assertEqual(
            [phase.phase_id for phase in loads[0].phases[:4]],
            [
                "prepress_latch_ramp_25",
                "prepress_latch_ramp_50",
                "prepress_latch_ramp_75",
                "prepress_latch_ramp_100",
            ],
        )
        self.assertEqual(loads[0].phases[0].direction_abs, (0.0, 0.0, -1.0))
        self.assertEqual(
            [phase.magnitude_si for phase in loads[0].phases[:4]],
            [2.0, 4.0, 6.0, 8.0],
        )
        controlled_slide_force_n = (
            0.0015 * 10.0**2 / (4.0 * (0.05 + 0.01))
        )
        controlled_drive_force_n = (
            controlled_slide_force_n**2 + 8.0**2
        ) ** 0.5
        direction = loads[0].phases[4].direction_abs
        self.assertAlmostEqual(
            direction[0],
            controlled_slide_force_n / controlled_drive_force_n,
        )
        self.assertAlmostEqual(
            direction[2],
            -8.0 / controlled_drive_force_n,
        )
        self.assertGreater(loads[0].phases[4].duration_s, 0.8)
        cycle = next(case for case in loads if case.program == "task_detent_cycle")
        unlock_alignment = next(
            phase
            for phase in cycle.phases
            if phase.phase_id == "align_outer_detent_for_unlock"
        )
        self.assertEqual(unlock_alignment.direction_abs, (1.0, 0.0, 0.0))
        self.assertEqual(unlock_alignment.magnitude_si, 15.0)
        self.assertEqual(unlock_alignment.stop_condition, "position_at_least")
        self.assertEqual(unlock_alignment.target_position_si, 0.065)
        retraction_prepress = next(
            phase
            for phase in cycle.phases
            if phase.phase_id == "prepress_latch_for_retraction_ramp_25"
        )
        self.assertEqual(retraction_prepress.direction_abs[0], 0.0)
        self.assertLess(retraction_prepress.direction_abs[2], 0.0)

    def test_parser_requires_two_sided_finite_detent_pockets(self) -> None:
        spec = prismatic_detent_latch_from_dict(_family_spec())
        self.assertEqual(spec.detent_interface.positions_m[-1], 0.065)
        self.assertEqual(
            prismatic_detent_latch_from_dict(spec.to_dict()),
            spec,
        )

        one_sided = deepcopy(_family_spec())
        contract = one_sided["detent_interface"]["geometry_contract"]
        contract["rack_component_refs"] = contract["rack_component_refs"][1::2]
        contract["rack_tooth_centers_abs_m"] = contract[
            "rack_tooth_centers_abs_m"
        ][1::2]
        with self.assertRaisesRegex(
            MechanismValidationError,
            "DETENT_SLIDER_BIDIRECTIONAL_CAPTURE_REQUIRED",
        ):
            prismatic_detent_latch_from_dict(one_sided)

    def test_ideal_reference_uses_latch_input_and_snaps_on_release(self) -> None:
        plan = _plan()
        load_case = default_load_cases(plan)[0]
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
                    "thumb_latch_position_m": 0.0,
                    "thumb_press_depth_m": 0.0,
                },
                {
                    "time_s": 0.1,
                    "phase_id": "press_and_slide_to_detent",
                    "thumb_latch_position_m": 0.044,
                    "thumb_press_depth_m": 0.004,
                },
                {
                    "time_s": 0.2,
                    "phase_id": "release_into_detent",
                    "thumb_latch_position_m": 0.044,
                    "thumb_press_depth_m": 0.0,
                },
            ],
        }
        ideal = generate_ideal_reference_trace(plan, load_case, physical)

        self.assertAlmostEqual(
            ideal["samples"][1]["blade_carriage_position_m"],
            0.044,
        )
        self.assertAlmostEqual(
            ideal["samples"][2]["blade_carriage_position_m"],
            0.045,
        )
        self.assertTrue(ideal["samples"][2]["detent_engaged"])
        self.assertEqual(
            set(ideal["samples"][2]["body_poses"]),
            {"handle", "blade", "latch"},
        )

    def test_zero_input_physical_behavior_has_no_ideal_joint(self) -> None:
        plan = _plan()
        load_case = next(
            case
            for case in default_load_cases(plan)
            if case.program == "zero_input_detent_slider"
        )
        sample = {
            "time_s": 0.0,
            "phase_id": "initial",
            "blade_carriage_position_m": 0.0,
            "blade_carriage_speed_m_s": 0.0,
            "thumb_latch_position_m": 0.0,
            "thumb_press_depth_m": 0.0,
            "detent_engaged": True,
            "locked_detent_position_m": 0.0,
            "detent_position_error_m": 0.0,
            "max_support_drift_m": 0.0,
            "support_drifts_m": {
                "blade_slide": 0.0,
                "latch_press": 0.0,
            },
            "max_off_axis_rotation_rad": 0.0,
            "body_poses": {
                body_id: {
                    "position_abs_m": [0.0, 0.0, 0.0],
                    "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
                for body_id in ("handle", "blade", "latch")
            },
            "contact_pair_counts": {"handle|latch": 2},
        }
        second = dict(sample, time_s=0.25, phase_id="zero_input")
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
                "detent_force_is_analytical": False,
            },
            "samples": [sample, second],
        }

        metrics, _thresholds, failures = physical_detent_slider_behavior(
            plan,
            load_case,
            trace,
        )
        self.assertEqual(failures, [])
        self.assertEqual(metrics["final_position_m"], 0.0)


def _strict_detent_schedule(load_case, spec):
    return [
        executed_detent_slider_phase(
            phase,
            elapsed_s=phase.duration_s,
            spec=spec,
            stop_reached=False,
            actual_coordinate_si=(
                0.0
                if phase.target_coordinate is None
                else float(phase.target_coordinate) - 0.001
            ),
        )
        for phase in load_case.phases
    ]


def _strict_program_samples(load_case):
    samples = [{"time_s": 0.0, "phase_id": "initial"}]
    cumulative = 0.0
    interval = load_case.sample_interval_s
    for phase in load_case.phases:
        t = cumulative + interval
        end = cumulative + phase.duration_s
        while t < end - 1.0e-12:
            samples.append({"time_s": t, "phase_id": phase.phase_id})
            t += interval
        samples.append({"time_s": end, "phase_id": phase.phase_id})
        cumulative = end
    return samples


def _behavior_sample(spec, *, time_s, phase_id, position_m=0.0, press_m=0.0):
    body_ids = [body.body_id for body in spec.bodies]
    engaged = press_m < spec.detent_interface.release_press_m
    return {
        "time_s": time_s,
        "phase_id": phase_id,
        "blade_carriage_position_m": position_m,
        "blade_carriage_speed_m_s": 0.0,
        "thumb_latch_position_m": position_m,
        "thumb_press_depth_m": press_m,
        "detent_engaged": engaged,
        "locked_detent_position_m": position_m if engaged else None,
        "detent_position_error_m": 0.0,
        "max_support_drift_m": 0.0,
        "support_drifts_m": {
            spec.blade_prismatic_joint.joint_id: 0.0,
            spec.latch_press_joint.joint_id: 0.0,
        },
        "max_off_axis_rotation_rad": 0.0,
        "body_poses": {
            body_id: {
                "position_abs_m": [position_m, 0.0, 0.0],
                "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
            for body_id in body_ids
        },
        "contact_pair_counts": {
            "|".join(
                sorted(
                    (
                        spec.detent_interface.rack_body_id,
                        spec.detent_interface.tab_body_id,
                    )
                )
            ): 2
        },
    }


class DetentStrictTraceProgramTests(unittest.TestCase):
    def test_handbook_programs_accept_runtime_executed_schedule(self) -> None:
        plan = _plan()
        spec = prismatic_detent_latch_from_dict(_family_spec())
        programs = {
            "task_detent_extend",
            "task_detent_hold_under_load",
            "task_detent_second_position",
            "task_detent_cycle",
        }
        seen = set()
        for load_case in default_load_cases(plan):
            if load_case.program not in programs:
                continue
            seen.add(load_case.program)
            schedule = _strict_detent_schedule(load_case, spec)
            samples = _strict_program_samples(load_case)
            program_spec = task_program_spec_for_load_case(
                load_case,
                plugin_id="prismatic_detent_latch_v1",
            )
            self.assertEqual(
                program_spec.executed_phase_extension_keys,
                ("actuation_channels", "termination_reason"),
            )
            adaptive_ids = set(program_spec.adaptive_phase_ids)
            for phase, recorded in zip(load_case.phases, schedule):
                self.assertIn(
                    recorded["termination_reason"],
                    {"duration_limit", "target_reached"},
                )
                self.assertIn("actuation_channels", recorded)
                if phase.phase_id in adaptive_ids:
                    self.assertIn("termination", recorded)
                    self.assertEqual(
                        recorded["termination"]["reason"],
                        "duration_limit",
                    )
                else:
                    self.assertNotIn("termination", recorded)
            trace = {
                "status": "COMPLETED",
                "sample_count": len(samples),
                "executed_phases": schedule,
                "executed_load_sha256": canonical_sha256(schedule),
                "samples": samples,
            }
            validated = validate_trace_program(
                trace,
                load_case,
                program_spec=program_spec,
            )
            self.assertEqual(
                validated.executed_phase_ids,
                tuple(phase.phase_id for phase in load_case.phases),
            )
        self.assertEqual(seen, programs)

    def test_old_comparator_keeps_verdict_on_strict_schedule(self) -> None:
        plan = _plan()
        spec = prismatic_detent_latch_from_dict(_family_spec())
        load_case = next(
            case
            for case in default_load_cases(plan)
            if case.program == "zero_input_detent_slider"
        )
        samples = [
            _behavior_sample(spec, time_s=0.0, phase_id="initial"),
            _behavior_sample(spec, time_s=0.25, phase_id="zero_input"),
        ]
        identity = {
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
                "detent_force_is_analytical": False,
            },
            "samples": samples,
        }
        schedule = _strict_detent_schedule(load_case, spec)
        new_trace = {
            **identity,
            "executed_phases": schedule,
            "executed_load_sha256": canonical_sha256(schedule),
            "sample_count": len(samples),
        }
        old_metrics, _, old_failures = physical_detent_slider_behavior(
            plan,
            load_case,
            identity,
        )
        new_metrics, _, new_failures = physical_detent_slider_behavior(
            plan,
            load_case,
            new_trace,
        )
        self.assertEqual(old_failures, [])
        self.assertEqual(new_failures, old_failures)
        self.assertEqual(new_metrics, old_metrics)

    def test_old_comparator_accepts_strict_task_extend_trace(self) -> None:
        plan = _plan()
        spec = prismatic_detent_latch_from_dict(_family_spec())
        load_case = next(
            case
            for case in default_load_cases(plan)
            if case.program == "task_detent_extend"
        )
        target = 0.045
        samples = [
            _behavior_sample(spec, time_s=0.0, phase_id="initial"),
        ]
        cumulative = 0.0
        interval = load_case.sample_interval_s
        for phase in load_case.phases:
            end = cumulative + phase.duration_s
            press = (
                spec.detent_interface.release_press_m + 0.001
                if phase.wrench_kind == "force"
                else 0.0
            )
            position = (
                target
                if phase.phase_id != "prepress_latch_ramp_25"
                else 0.0
            )
            t = cumulative + interval
            while t < end - 1.0e-12:
                samples.append(
                    _behavior_sample(
                        spec,
                        time_s=t,
                        phase_id=phase.phase_id,
                        position_m=position,
                        press_m=press,
                    )
                )
                t += interval
            samples.append(
                _behavior_sample(
                    spec,
                    time_s=end,
                    phase_id=phase.phase_id,
                    position_m=target,
                    press_m=press,
                )
            )
            cumulative = end
        schedule = _strict_detent_schedule(load_case, spec)
        new_trace = {
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
                "detent_force_is_analytical": False,
            },
            "executed_phases": schedule,
            "executed_load_sha256": canonical_sha256(schedule),
            "sample_count": len(samples),
            "samples": samples,
        }
        old_trace = dict(new_trace)
        old_trace.pop("executed_phases")
        old_trace.pop("executed_load_sha256")
        old_trace.pop("sample_count")
        old_metrics, _, old_failures = physical_detent_slider_behavior(
            plan,
            load_case,
            old_trace,
        )
        new_metrics, _, new_failures = physical_detent_slider_behavior(
            plan,
            load_case,
            new_trace,
        )
        self.assertEqual(old_failures, [])
        self.assertEqual(new_failures, old_failures)
        self.assertEqual(new_metrics, old_metrics)
        validate_trace_program(
            new_trace,
            load_case,
            program_spec=task_program_spec_for_load_case(
                load_case,
                plugin_id="prismatic_detent_latch_v1",
            ),
        )

    def test_adaptive_phase_without_termination_object_is_rejected(self) -> None:
        plan = _plan()
        spec = prismatic_detent_latch_from_dict(_family_spec())
        load_case = next(
            case
            for case in default_load_cases(plan)
            if case.program == "task_detent_extend"
        )
        schedule = _strict_detent_schedule(load_case, spec)
        adaptive = next(
            item
            for item, phase in zip(schedule, load_case.phases)
            if phase.stop_condition is not None
        )
        adaptive.pop("termination")
        samples = _strict_program_samples(load_case)
        with self.assertRaises(MechanismValidationError) as caught:
            validate_trace_program(
                {
                    "status": "COMPLETED",
                    "sample_count": len(samples),
                    "executed_phases": schedule,
                    "executed_load_sha256": canonical_sha256(schedule),
                    "samples": samples,
                },
                load_case,
                program_spec=task_program_spec_for_load_case(
                    load_case,
                    plugin_id="prismatic_detent_latch_v1",
                ),
            )
        self.assertEqual(
            caught.exception.code,
            "TASK_EXECUTED_PHASE_SCHEMA_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
