from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.clock_gear_train_spec import (
    clock_gear_train_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.clock_gear_comparison import (
    _behavior_failures,
    _clock_requirement_checks,
    _semantic_failures,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.clock_gear_runtime import (
    _clock_physical_solver,
    _phase_limit_seconds,
    _support_drift_from_gate_state,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.certificate import (
    build_replacement,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
    compare_trace_pair,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    mechanism_adapter_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import (
    default_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine_worker import (
    _clock_task_checkpoint_file,
    _whole_machine_trace_checkpoint_file,
)


def _family_spec() -> dict:
    moving = ("barrel", "center", "third", "fourth", "minute", "hammer", "setting")
    bodies = [
        {
            "body_id": "ground",
            "group_ids": ["group_case"],
            "fixed": True,
            "mass_kg": 1.8,
            "center_abs_m": [0.0, 0.0, 0.1],
            "size_m": [0.24, 0.08, 0.24],
        }
    ]
    origins = {
        "barrel": [-0.06, 0.0, 0.08],
        "center": [-0.02, 0.0, 0.11],
        "third": [0.02, 0.0, 0.08],
        "fourth": [0.05, 0.0, 0.12],
        "minute": [0.0, -0.03, 0.13],
        "hammer": [0.07, 0.0, 0.17],
        "setting": [0.08, 0.0, 0.15],
    }
    for body_id in moving:
        bodies.append(
            {
                "body_id": body_id,
                "group_ids": [f"group_{body_id}"],
                "fixed": False,
                "mass_kg": 0.08,
                "center_abs_m": origins[body_id],
                "size_m": [0.05, 0.015, 0.05],
            }
        )
    meshes = []
    chain = ("barrel", "center", "third", "fourth", "minute")
    for index, (driver, driven) in enumerate(zip(chain, chain[1:]), start=1):
        meshes.append(
            {
                "mesh_id": f"mesh_{index}",
                "driver_body_id": driver,
                "driven_body_id": driven,
                "driver_teeth": 12 + index,
                "driven_teeth": 30 + index,
                "mesh_stiffness_nm_rad": 12.0,
                "mesh_damping_nm_s_rad": 0.08,
            }
        )
    return {
        "mechanism_kind": "spur_gear_clock",
        "train_id": "table_alarm_clock_train",
        "bodies": bodies,
        "shafts": [
            {
                "shaft_id": f"shaft_{body_id}",
                "body_id": body_id,
                "origin_abs_m": origins[body_id],
                "axis_abs": [0.0, 1.0, 0.0],
            }
            for body_id in moving
        ],
        "gear_meshes": meshes,
        "cam_followers": [
            {
                "follower_id": "alarm_cam_hammer",
                "cam_body_id": "minute",
                "follower_body_id": "hammer",
                "lobe_count": 2,
                "center_angle_rad": 0.0,
                "amplitude_rad": 0.25,
                "stiffness_nm_rad": 8.0,
                "damping_nm_s_rad": 0.12,
            }
        ],
        "setting_clutch": {
            "clutch_id": "setting_clutch",
            "input_body_id": "setting",
            "driven_body_id": "minute",
            "ratio": 1.0,
            "stiffness_nm_rad": 8.0,
            "damping_nm_s_rad": 0.05,
            "max_torque_nm": 0.25,
        },
        "fixed_body_id": "ground",
        "input_body_id": "barrel",
        "setting_input_body_id": "setting",
        "output_body_id": "minute",
        "hour_body_id": "minute",
        "second_body_id": "minute",
        "alarm_body_id": "hammer",
        "input_point_abs_m": origins["barrel"],
        "input_axis_abs": [0.0, 1.0, 0.0],
        "setting_input_point_abs_m": origins["setting"],
        "setting_input_axis_abs": [0.0, 1.0, 0.0],
        "expected_hour_to_minute_ratio": 1.0,
        "expected_second_to_minute_ratio": 1.0,
        "hand_ratio_relative_tolerance": 0.05,
        "home_output_angle_rad": 0.0,
        "outbound_output_angle_rad": 0.5,
        "nominal_input_torque_nm": 0.18,
        "return_input_torque_nm": 0.08,
        "max_torque_phase_s": 0.6,
        "max_return_phase_s": 0.5,
        "settle_phase_s": 0.08,
        "output_angle_tolerance_rad": 0.08,
        "max_support_drift_m": 0.001,
        "max_off_axis_rotation_rad": 0.05,
        "max_mesh_error_rad": 0.08,
        "minimum_alarm_swing_rad": 0.05,
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
        package_id="mechanical_alarm_clock_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="clock_train",
        capsule_file="clock_train.json",
        family="compound",
        protocol_id="multi_shaft_gear_train",
        fixed_group_ids=("group_case",),
        moving_group_ids=tuple(f"group_{item}" for item in (
            "barrel", "center", "third", "fourth", "minute", "hammer", "setting"
        )),
        axis_abs=(0.0, 1.0, 0.0),
        joint_origin_abs_m=(-0.06, 0.0, 0.08),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_barrel",
        input_port_id="winding_torque",
        input_frame_id="frame_barrel",
        input_point_abs_m=(-0.06, 0.0, 0.08),
        observable_id="minute_hand_angle",
        observable_frame_id="frame_minute",
        observable_point_abs_m=(0.0, -0.03, 0.13),
        task_ids=("advance_display", "ring_alarm"),
        characteristic_length_m=0.24,
        moving_extents_m=(0.18, 0.05, 0.18),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _isolated_family_spec() -> dict:
    value = deepcopy(_family_spec())
    origins = {
        "alarm_barrel": [-0.07, 0.0, 0.14],
        "alarm_cam": [-0.035, 0.0, 0.14],
    }
    for body_id, origin in origins.items():
        value["bodies"].append(
            {
                "body_id": body_id,
                "group_ids": [f"group_{body_id}"],
                "fixed": False,
                "mass_kg": 0.06,
                "center_abs_m": origin,
                "size_m": [0.04, 0.015, 0.04],
            }
        )
        value["shafts"].append(
            {
                "shaft_id": f"shaft_{body_id}",
                "body_id": body_id,
                "origin_abs_m": origin,
                "axis_abs": [0.0, 1.0, 0.0],
            }
        )
    value["gear_meshes"].append(
        {
            "mesh_id": "alarm_barrel_cam",
            "driver_body_id": "alarm_barrel",
            "driven_body_id": "alarm_cam",
            "driver_teeth": 12,
            "driven_teeth": 68,
            "mesh_stiffness_nm_rad": 12.0,
            "mesh_damping_nm_s_rad": 0.08,
        }
    )
    value["cam_followers"][0]["cam_body_id"] = "alarm_cam"
    value["alarm_input_body_id"] = "alarm_barrel"
    value["alarm_input_point_abs_m"] = origins["alarm_barrel"]
    value["alarm_input_axis_abs"] = [0.0, 1.0, 0.0]
    value["nominal_alarm_input_torque_nm"] = 0.09
    return value


def _scheduled_family_spec() -> dict:
    value = _isolated_family_spec()
    value["scheduled_alarm"] = {
        "setting_body_id": "setting",
        "target_time_seconds": 60.0,
        "setting_target_angle_rad": 0.10,
        "setting_angle_tolerance_rad": 0.02,
        "target_second_hand_travel_rad": 0.10,
        "target_second_hand_tolerance_rad": 0.02,
        "target_minute_hand_travel_rad": 0.10,
        "target_hour_hand_travel_rad": 0.10,
        "minimum_post_trigger_time_travel_rad": 0.03,
    }
    return value


def _hour_hand_alarm_family_spec() -> dict:
    value = _scheduled_family_spec()
    value["bodies"].append(
        {
            "body_id": "alarm_target",
            "group_ids": ["group_alarm_target"],
            "fixed": False,
            "mass_kg": 0.02,
            "center_abs_m": [0.0, 0.0, 0.0],
            "size_m": [0.02, 0.004, 0.02],
        }
    )
    value["shafts"].append(
        {
            "shaft_id": "shaft_alarm_target",
            "body_id": "alarm_target",
            "origin_abs_m": [0.0, 0.0, 0.0],
            "axis_abs": [0.0, 1.0, 0.0],
        }
    )
    value["scheduled_alarm"]["setting_body_id"] = "alarm_target"
    value["alarm_setting_clutch"] = {
        "clutch_id": "hour_hand_alarm_target_clutch",
        "input_body_id": value["hour_body_id"],
        "driven_body_id": "alarm_target",
        "ratio": 1.0,
        "stiffness_nm_rad": 16.0,
        "damping_nm_s_rad": 0.05,
        "max_torque_nm": 0.25,
    }
    return value


class ClockGearPluginTests(unittest.TestCase):
    def test_support_drift_uses_bounded_anchor_state(self) -> None:
        state = {
            "support_drift_m": 2.5e-7,
            "body_center_to_axis_offset_m": 0.127,
        }

        self.assertEqual(
            _support_drift_from_gate_state(state),
            2.5e-7,
        )

    def test_whole_machine_clock_uses_immutable_per_task_checkpoints(self) -> None:
        replacement = ".runtime_runs/results/clock_gear_train/replacement.json"
        normal = _clock_task_checkpoint_file(
            replacement,
            "task_three_hand_timekeeping",
        )
        scheduled = _clock_task_checkpoint_file(
            replacement,
            "task_scheduled_alarm_trigger",
        )

        self.assertNotEqual(normal, scheduled)
        self.assertNotEqual(normal, Path(replacement))
        self.assertEqual(normal.parent.name, "checkpoints")
        self.assertEqual(normal.parent.parent.name, "whole_machine")
        self.assertEqual(
            normal.name,
            "task_three_hand_timekeeping.partial.trace.json",
        )

    def test_compound_whole_machine_checkpoint_never_overwrites_replacement(
        self,
    ) -> None:
        replacement = ".runtime_runs/results/linkage/replacement.json"
        checkpoint = _whole_machine_trace_checkpoint_file(
            replacement,
            "task_output_angle_roundtrip",
        )

        self.assertNotEqual(checkpoint, Path(replacement))
        self.assertEqual(checkpoint.parent.name, "checkpoints")
        self.assertEqual(checkpoint.parent.parent.name, "whole_machine")
        self.assertEqual(
            checkpoint.name,
            "task_output_angle_roundtrip.partial.trace.json",
        )

    def test_clock_physical_solver_uses_preconditioned_gmres(self) -> None:
        events: list[tuple[str, object]] = []

        class Solver:
            def SetMaxIterations(self, value: int) -> None:
                events.append(("iterations", value))

            def SetTolerance(self, value: float) -> None:
                events.append(("tolerance", value))

            def EnableDiagonalPreconditioner(self, value: bool) -> None:
                events.append(("diagonal_preconditioner", value))

        solver = Solver()

        class Chrono:
            @staticmethod
            def ChSolverGMRES() -> Solver:
                events.append(("solver", "GMRES"))
                return solver

        self.assertIs(_clock_physical_solver(Chrono), solver)
        self.assertEqual(
            events,
            [
                ("solver", "GMRES"),
                ("iterations", 350),
                ("tolerance", 1.0e-10),
                ("diagonal_preconditioner", True),
            ],
        )

    def test_reduced_target_phases_keep_declared_timeout(self) -> None:
        self.assertAlmostEqual(
            _phase_limit_seconds(
                branch="reduced",
                program="manual_zero_and_resume_clock",
                phase_id="calibration_return",
                declared_duration_s=2.50,
                reference_duration_s=1.8252,
            ),
            2.50,
        )
        self.assertAlmostEqual(
            _phase_limit_seconds(
                branch="reduced",
                program="scheduled_alarm_clock",
                phase_id="advance_to_alarm_time",
                declared_duration_s=0.85,
                reference_duration_s=0.1615,
            ),
            0.85,
        )

    def test_non_target_phase_replays_reference_duration(self) -> None:
        self.assertAlmostEqual(
            _phase_limit_seconds(
                branch="reduced",
                program="task_clock_cycle",
                phase_id="wind_and_advance",
                declared_duration_s=0.85,
                reference_duration_s=0.73,
            ),
            0.73,
        )

    def test_scheduled_clock_adds_zero_resume_and_one_minute_alarm_cases(self) -> None:
        family_spec = _scheduled_family_spec()
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=family_spec,
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        cases = default_load_cases(plan)

        self.assertEqual(len(cases), 10)
        self.assertIn("manual_zero_and_resume_clock", adapter.supported_programs)
        self.assertIn("scheduled_alarm_clock", adapter.supported_programs)
        self.assertEqual(cases[-2].program, "manual_zero_and_resume_clock")
        self.assertEqual(cases[-1].program, "scheduled_alarm_clock")
        zero_return = next(
            phase
            for phase in cases[-2].phases
            if phase.phase_id == "calibration_return"
        )
        self.assertAlmostEqual(zero_return.magnitude_si, 0.16)
        self.assertAlmostEqual(zero_return.duration_s, 2.50)
        self.assertEqual(
            [phase.phase_id for phase in cases[-1].phases],
            [
                "set_alarm_for_one_minute",
                "alarm_setting_latched",
                "advance_to_alarm_time",
                "ring_with_timekeeping",
                "scheduled_alarm_settle",
            ],
        )

    def test_hour_hand_alarm_profile_owns_exactly_five_user_task_cases(self) -> None:
        family_spec = _hour_hand_alarm_family_spec()
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=family_spec,
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        cases = default_load_cases(plan)

        self.assertEqual(
            [item.load_case_id for item in cases],
            [
                "task_three_hand_timekeeping",
                "task_manual_time_setting",
                "task_alarm_cycle",
                "task_alarm_time_setting",
                "task_scheduled_alarm_trigger",
            ],
        )
        self.assertEqual(
            [item.program for item in cases],
            [
                "task_clock_cycle",
                "manual_time_setting_clock",
                "alarm_clock_cycle",
                "hour_hand_alarm_setting_clock",
                "scheduled_alarm_clock",
            ],
        )
        self.assertEqual(
            [
                phase.phase_id
                for phase in cases[-1].phases
            ],
            [
                "turn_hour_hand_to_alarm_target",
                "alarm_setting_latched",
                "restore_hour_hand_to_current_time",
                "alarm_target_ready",
                "advance_to_alarm_time",
                "ring_with_timekeeping",
                "scheduled_alarm_settle",
            ],
        )

    def test_scheduled_alarm_behavior_requires_trigger_then_concurrent_timekeeping(self) -> None:
        spec = clock_gear_train_from_dict(_scheduled_family_spec())
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=_scheduled_family_spec(),
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "scheduled_alarm_clock"
        )
        metrics = {
            "max_support_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "max_mesh_error_rad": 0.0,
            "output_span_rad": 0.16,
            "hour_to_minute_ratio": 1.0,
            "second_to_minute_ratio": 1.0,
            "hand_direction_consistent": True,
            "setting_input_span_rad": 0.10,
            "phase_setting_end_rad": {"alarm_setting_latched": 0.10},
            "phase_second_travel_rad": {"advance_to_alarm_time": 0.10},
            "phase_output_travel_rad": {
                "advance_to_alarm_time": 0.10,
                "ring_with_timekeeping": 0.04,
            },
            "phase_hour_travel_rad": {"advance_to_alarm_time": 0.10},
            "phase_alarm_span_rad": {
                "set_alarm_for_one_minute": 0.0,
                "alarm_setting_latched": 0.0,
                "advance_to_alarm_time": 0.0,
                "ring_with_timekeeping": 0.08,
            },
            "alarm_swing_rad": 0.08,
            "max_output_angle_rad": 0.16,
        }

        self.assertEqual(
            _behavior_failures(
                metrics,
                branch="physical",
                load_case=load_case,
                spec=spec,
            ),
            [],
        )

    def test_zero_resume_behavior_checks_return_release_and_restart(self) -> None:
        spec = clock_gear_train_from_dict(_scheduled_family_spec())
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=_scheduled_family_spec(),
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "manual_zero_and_resume_clock"
        )
        metrics = {
            "max_support_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "max_mesh_error_rad": 0.0,
            "output_span_rad": 0.20,
            "hour_to_minute_ratio": 1.0,
            "second_to_minute_ratio": 1.0,
            "hand_direction_consistent": True,
            "setting_input_span_rad": 0.20,
            "phase_output_travel_rad": {
                "calibration_offset": 0.12,
                "release_setting": 0.0,
                "post_zero_run": 0.05,
            },
            "phase_output_start_rad": {"calibration_offset": 0.0},
            "phase_output_end_rad": {"calibration_return": 0.01},
            "alarm_swing_rad": 0.0,
            "max_output_angle_rad": 0.20,
        }

        self.assertEqual(
            _behavior_failures(
                metrics,
                branch="physical",
                load_case=load_case,
                spec=spec,
            ),
            [],
        )

    def test_legacy_clock_serialization_does_not_gain_alarm_fields(self) -> None:
        spec = clock_gear_train_from_dict(_family_spec())
        payload = spec.to_dict()

        self.assertFalse(spec.alarm_isolated)
        self.assertNotIn("alarm_input_body_id", payload)
        self.assertEqual(spec.effective_alarm_input_body_id, spec.input_body_id)

    def test_isolated_alarm_declares_a_separate_input_and_load_axis(self) -> None:
        family_spec = _isolated_family_spec()
        spec = clock_gear_train_from_dict(family_spec)
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=family_spec,
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        alarm_case = next(
            item for item in default_load_cases(plan)
            if item.program == "alarm_clock_cycle"
        )

        self.assertTrue(spec.alarm_isolated)
        self.assertEqual(spec.effective_alarm_input_body_id, "alarm_barrel")
        self.assertEqual(alarm_case.phases[0].direction_abs, (0.0, 1.0, 0.0))
        self.assertAlmostEqual(alarm_case.phases[0].magnitude_si, 0.099)

    def test_isolated_alarm_rejects_a_hidden_hand_train_connection(self) -> None:
        family_spec = _isolated_family_spec()
        family_spec["gear_meshes"].append(
            {
                "mesh_id": "forbidden_minute_alarm_link",
                "driver_body_id": "minute",
                "driven_body_id": "alarm_cam",
                "driver_teeth": 10,
                "driven_teeth": 68,
                "mesh_stiffness_nm_rad": 12.0,
                "mesh_damping_nm_s_rad": 0.08,
            }
        )

        with self.assertRaisesRegex(
            MechanismValidationError,
            "independent alarm train must not connect",
        ):
            clock_gear_train_from_dict(family_spec)

    def test_isolated_alarm_semantics_require_the_alarm_barrel_policy(self) -> None:
        family_spec = _isolated_family_spec()
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=family_spec,
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        plugin, _ = mechanism_adapter_for_plan(plan)
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "alarm_clock_cycle"
        )
        links = list(plugin.expected_reduced_ideal_links(plan, adapter))
        common_semantics = {
            "same_mass_kg": 1.0,
            "same_inertia_xx_kg_m2": {},
            "same_initial_pose": {},
            "same_external_load_sha256": "a" * 64,
            "family_spec_sha256": canonical_sha256(
                clock_gear_train_from_dict(family_spec).to_dict()
            ),
            "direct_hand_or_hammer_actuation": False,
            "direct_hour_second_or_hammer_actuation": False,
            "input_actuation_policy": "external_torque_on_independent_alarm_barrel_only",
        }
        physical = {
            "engine": {"chrono_link_count": 0, "internal_ideal_joint_count": 0},
            "collision_geometry": {
                "contact_enabled": True,
                "whole_body_box_proxy_used": False,
                "whole_body_convex_hull_used": False,
                "convex_hull_used": False,
            },
            "branch_semantics": common_semantics,
        }
        reduced = {
            "engine": {
                "ideal_links": links,
                "internal_ideal_joint_count": len(links),
            },
            "collision_geometry": {
                "contact_enabled": False,
                "whole_body_box_proxy_used": False,
                "whole_body_convex_hull_used": False,
                "convex_hull_used": False,
            },
            "branch_semantics": common_semantics,
        }

        self.assertEqual(
            _semantic_failures(plan, load_case, physical, reduced),
            [],
        )

    def test_isolated_normal_clock_task_does_not_require_alarm_motion(self) -> None:
        spec = clock_gear_train_from_dict(_isolated_family_spec())
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=_isolated_family_spec(),
        )
        plan = replace(_plan(), adapter_spec=adapter_to_dict(adapter))
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "task_clock_cycle"
        )
        metrics = {
            "max_support_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "max_mesh_error_rad": 0.0,
            "output_span_rad": 0.60,
            "hour_to_minute_ratio": spec.expected_hour_to_minute_ratio,
            "second_to_minute_ratio": spec.expected_second_to_minute_ratio,
            "hand_direction_consistent": True,
            "setting_input_span_rad": 0.0,
            "phase_output_travel_rad": {
                "wind_and_advance": 0.50,
                "controlled_return": -0.20,
            },
            "alarm_swing_rad": 0.0,
            "max_output_angle_rad": 0.50,
        }

        self.assertEqual(
            _behavior_failures(
                metrics,
                branch="physical",
                load_case=load_case,
                spec=spec,
            ),
            [],
        )

    def test_clock_plugin_owns_topology_and_eight_load_cases(self) -> None:
        plan = _plan()
        plugin, adapter = mechanism_adapter_for_plan(plan)
        spec = clock_gear_train_from_dict(adapter.family_spec)
        cases = default_load_cases(plan)

        self.assertEqual(plugin.plugin_id, "spur_gear_clock_v1")
        self.assertEqual(plugin.worker_backend, "spur_gear_clock_v1")
        self.assertEqual(
            plugin.implementation_capability_ids,
            frozenset(
                {
                    "clock.finite_gear_train",
                    "clock.independent_alarm_power_paths",
                }
            ),
        )
        self.assertEqual(len(spec.shafts), 7)
        self.assertEqual(len(spec.gear_meshes), 4)
        self.assertEqual(len(cases), 8)
        self.assertEqual(cases[-2].program, "manual_time_setting_clock")
        self.assertEqual(cases[-1].program, "alarm_clock_cycle")
        self.assertEqual(
            plugin.expected_reduced_ideal_links(plan, adapter).count(
                "ChLinkLockRevolute"
            ),
            7,
        )
        self.assertEqual(
            plugin.expected_reduced_ideal_links(plan, adapter).count(
                "ChLinkLockGear"
            ),
            4,
        )

    def test_clock_comparison_routes_around_generic_compound_parser(self) -> None:
        plan = _plan()
        plugin, adapter = mechanism_adapter_for_plan(plan)
        spec = clock_gear_train_from_dict(adapter.family_spec)
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "zero_input_clock_train"
        )
        body_ids = [item.body_id for item in spec.bodies]
        shaft_body_ids = [item.body_id for item in spec.shafts]
        shaft_ids = [item.shaft_id for item in spec.shafts]
        mesh_ids = [item.mesh_id for item in spec.gear_meshes]
        follower_ids = [item.follower_id for item in spec.cam_followers]
        pose = {
            "position_abs_m": [0.0, 0.0, 0.0],
            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }

        def trace(branch: str) -> dict:
            sample = {
                "time_s": 0.0,
                "phase_id": "initial",
                "clock_output_angle_rad": 0.0,
                "clock_output_speed_rad_s": 0.0,
                "clock_input_angle_rad": 0.0,
                "clock_input_speed_rad_s": 0.0,
                "barrel_input_angle_rad": 0.0,
                "setting_input_angle_rad": 0.0,
                "hour_hand_angle_rad": 0.0,
                "hour_hand_speed_rad_s": 0.0,
                "second_hand_angle_rad": 0.0,
                "second_hand_speed_rad_s": 0.0,
                "alarm_angle_rad": 0.0,
                "alarm_speed_rad_s": 0.0,
                "max_mesh_error_rad": 0.0,
                "max_support_drift_m": 0.0,
                "max_off_axis_rotation_rad": 0.0,
                "setting_clutch_error_rad": 0.0,
                "shaft_angles_rad": {item: 0.0 for item in shaft_body_ids},
                "shaft_speeds_rad_s": {item: 0.0 for item in shaft_body_ids},
                "mesh_errors_rad": {item: 0.0 for item in mesh_ids},
                "support_drifts_m": {item: 0.0 for item in shaft_ids},
                "off_axis_rotations_rad": {item: 0.0 for item in shaft_ids},
                "cam_follower_errors_rad": {item: 0.0 for item in follower_ids},
                "body_poses": {item: pose for item in body_ids},
                "contact_count": 0,
            }
            final = dict(sample)
            final["time_s"] = load_case.duration_s
            final["phase_id"] = load_case.phases[0].phase_id
            expected_links = list(plugin.expected_reduced_ideal_links(plan, adapter))
            return {
                "status": "COMPLETED",
                "branch": branch,
                "package_binding_sha256": plan.package_binding_sha256,
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
                "plan_sha256": canonical_sha256(plan.to_dict()),
                "load_case_sha256": canonical_sha256(load_case.to_dict()),
                "engine": {
                    "chrono_link_count": 0 if branch == "physical" else 11,
                    "ideal_links": [] if branch == "physical" else expected_links,
                    "internal_ideal_joint_count": (
                        0 if branch == "physical" else len(expected_links)
                    ),
                },
                "collision_geometry": {
                    "contact_enabled": branch == "physical",
                    "whole_body_box_proxy_used": False,
                    "whole_body_convex_hull_used": False,
                    "convex_hull_used": False,
                },
                "branch_semantics": {
                    "same_mass_kg": 1.0,
                    "same_inertia_xx_kg_m2": {},
                    "same_initial_pose": {},
                    "same_external_load_sha256": "a" * 64,
                    "family_spec_sha256": canonical_sha256(spec.to_dict()),
                    "direct_hand_or_hammer_actuation": False,
                    "direct_hour_second_or_hammer_actuation": False,
                    "input_actuation_policy": "external_torque_on_barrel_only",
                },
                "samples": [sample, final],
            }

        comparison = compare_trace_pair(
            plan,
            load_case,
            trace("physical"),
            trace("reduced"),
        )
        self.assertEqual(comparison["verdict"], "PASS")
        self.assertEqual(comparison["thresholds"]["source"], "validated_clock_gear_family_spec")

    def test_clock_requirement_checks_are_backed_by_trace_evidence(self) -> None:
        spec = clock_gear_train_from_dict(_isolated_family_spec())
        pose = {
            "position_abs_m": [0.0, 0.0, 0.0],
            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        strict_non_claims = {
            "real_time_chronometric_accuracy_certified": False,
            "spring_endurance_certified": False,
            "wear_or_fatigue_certified": False,
            "manufacturability_certified": False,
            "consumer_safety_certified": False,
        }
        trace = {
            "package_binding_sha256": "2" * 64,
            "branch_semantics": {
                "alarm_train_isolated": True,
                "direct_hand_or_hammer_actuation": False,
                "direct_output_shaft_actuation": False,
            },
            "wrench_audit": {
                "passed": True,
                "direct_output_drive_detected": False,
            },
            "collision_geometry": {
                "mode": (
                    "intake-bound STEP per-solid body-local "
                    "collision components"
                ),
                "package_binding_sha256": "2" * 64,
                "assembly_sha256": "3" * 64,
                "assembly_contract_sha256": "4" * 64,
                "whole_body_box_proxy_used": False,
                "whole_body_convex_hull_used": False,
                "convex_hull_used": False,
                "bodies": [
                    {
                        "body_id": body.body_id,
                        "components": [{"solid_id": body.group_ids[0]}],
                    }
                    for body in spec.bodies
                ],
            },
        }
        samples = [
            {"body_poses": {body.body_id: pose for body in spec.bodies}}
        ]

        checks = _clock_requirement_checks(
            trace,
            spec,
            samples,
            strict_non_claims=strict_non_claims,
        )

        self.assertEqual(
            set(checks),
            {
                "validator.mechanical.single_source_cad",
                "validator.mechanical.rigid_group_ownership",
                "validator.clock.independent_alarm_power_paths",
                "policy.clock.validation_boundaries",
            },
        )
        self.assertTrue(all(item["passed"] for item in checks.values()))
        trace["branch_semantics"]["alarm_train_isolated"] = False
        failed = _clock_requirement_checks(
            trace,
            spec,
            samples,
            strict_non_claims=strict_non_claims,
        )
        self.assertFalse(
            failed["validator.clock.independent_alarm_power_paths"]["passed"]
        )

    def test_clock_replacement_preserves_full_declared_topology(self) -> None:
        plan = _plan()
        replacement = build_replacement(
            plan,
            {"verdict": "KINEMATIC_CERTIFIED", "tested_envelope": {}},
            certificate_sha256="f" * 64,
        )
        topology = replacement["replacement_model"]["compound_topology"]
        self.assertEqual(topology["mechanism_kind"], "spur_gear_clock")
        self.assertEqual(len(topology["revolute_links"]), 7)
        self.assertEqual(len(topology["gear_links"]), 4)
        self.assertEqual(len(topology["cam_followers"]), 1)
        self.assertEqual(topology["setting_clutch"]["clutch_id"], "setting_clutch")


if __name__ == "__main__":
    unittest.main()
