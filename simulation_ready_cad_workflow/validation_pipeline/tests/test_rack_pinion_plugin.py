from __future__ import annotations

import unittest
from dataclasses import replace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_from_dict,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    mechanism_adapter_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import default_load_cases
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_ideal_comparison import (
    _physical_behavior,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.rack_pinion_spec import (
    rack_pinion_from_dict,
)


def _family_spec() -> dict:
    return {
        "mechanism_kind": "rack_pinion",
        "mechanism_id": "manual_gate_drive",
        "bodies": [
            {
                "body_id": "ground",
                "group_ids": ["frame"],
                "fixed": True,
                "mass_kg": 8.0,
                "center_abs_m": [0.0, 0.0, 0.5],
                "size_m": [0.4, 0.2, 1.0],
            },
            {
                "body_id": "pinion",
                "group_ids": ["crank", "pinion"],
                "fixed": False,
                "mass_kg": 0.8,
                "center_abs_m": [0.0, 0.0, 0.35],
                "size_m": [0.12, 0.08, 0.12],
            },
            {
                "body_id": "rack",
                "group_ids": ["rack", "gate"],
                "fixed": False,
                "mass_kg": 2.5,
                "center_abs_m": [0.04, 0.0, 0.42],
                "size_m": [0.12, 0.05, 0.5],
            },
        ],
        "fixed_body_id": "ground",
        "pinion_body_id": "pinion",
        "rack_body_id": "rack",
        "pinion_origin_abs_m": [0.0, 0.0, 0.35],
        "pinion_axis_abs": [0.0, 1.0, 0.0],
        "rack_axis_abs": [0.0, 0.0, 1.0],
        "pinion_pitch_radius_m": 0.04,
        "pressure_angle_rad": 0.3490658503988659,
        "home_progress_m": 0.0,
        "outbound_progress_m": 0.18,
        "nominal_input_torque_nm": 3.2,
        "return_input_torque_nm": 2.0,
        "max_torque_phase_s": 1.4,
        "max_return_phase_s": 1.2,
        "settle_phase_s": 0.12,
        "output_progress_tolerance_m": 0.004,
        "max_support_drift_m": 0.0015,
        "max_off_axis_rotation_rad": 0.04,
        "max_coupling_error_m": 0.0015,
        "friction": 0.08,
    }


def _plan() -> MechanismPlan:
    adapter = replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID), family_spec=_family_spec()
    )
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="manual_gate_package",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="rack_pinion_gate",
        capsule_file="capsule.json",
        family="compound",
        protocol_id="mixed_joint_mechanism",
        fixed_group_ids=("frame",),
        moving_group_ids=("crank", "pinion", "rack", "gate"),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.35),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="pinion_frame",
        input_port_id="crank_torque",
        input_frame_id="pinion_frame",
        input_point_abs_m=(0.0, 0.0, 0.35),
        observable_id="gate_height",
        observable_frame_id="rack_frame",
        observable_point_abs_m=(0.04, 0.0, 0.42),
        task_ids=("raise_gate", "lower_gate"),
        characteristic_length_m=1.0,
        moving_extents_m=(0.3, 0.2, 0.7),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


class RackPinionPluginTests(unittest.TestCase):
    def test_plugin_selects_mixed_joint_backend_and_six_loads(self) -> None:
        plan = _plan()
        plugin, adapter = mechanism_adapter_for_plan(plan)
        spec = rack_pinion_from_dict(adapter.family_spec)
        cases = default_load_cases(plan)

        self.assertEqual(plugin.plugin_id, "rack_pinion_v1")
        self.assertEqual(plugin.coordinate.position_unit, "m")
        self.assertEqual(plugin.coordinate.input_wrench_type, "torque")
        self.assertEqual(spec.pinion_pitch_radius_m, 0.04)
        self.assertEqual(
            plugin.expected_reduced_ideal_links(plan, adapter),
            (
                "ChLinkLockRevolute",
                "ChLinkLockPrismatic",
                "ChLinkMateRackPinion",
            ),
        )
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[-1].program, "task_rack_roundtrip")
        self.assertEqual(cases[-1].phases[0].target_position_si, 0.177)
        self.assertTrue(
            all(
                phase.wrench_kind in {"none", "torque"}
                for case in cases
                for phase in case.phases
            )
        )
        for case in (cases[0], cases[1], cases[4], cases[5]):
            self.assertEqual(case.phases[0].stop_condition, "position_at_least")
            self.assertEqual(case.phases[0].target_position_si, 0.177)
            self.assertEqual(case.phases[1].duration_s, 0.25)
            lower = next(
                phase for phase in case.phases if phase.phase_id == "crank_lower"
            )
            self.assertEqual(lower.stop_condition, "position_at_most")
            self.assertEqual(lower.target_position_si, 0.003)
            self.assertEqual(case.phases[-1].duration_s, 0.25)
        for case in (cases[0], cases[1], cases[4]):
            self.assertEqual(
                tuple(phase.phase_id for phase in case.phases),
                ("crank_lift", "upper_settle", "crank_lower", "home_settle"),
            )
        task_case = cases[-1]
        self.assertEqual(
            tuple(phase.phase_id for phase in task_case.phases),
            (
                "crank_lift",
                "upper_settle",
                "zero_input_hold",
                "crank_lower",
                "home_settle",
            ),
        )
        hold_phase = task_case.phases[2]
        self.assertEqual(hold_phase.wrench_kind, "none")
        self.assertEqual(hold_phase.duration_s, 0.5)

    def test_physical_ideal_gate_uses_rack_direction_convention(self) -> None:
        plan = _plan()
        load_case = default_load_cases(plan)[0]

        def sample(time_s: float, phase_id: str, progress_m: float) -> dict:
            return {
                "time_s": time_s,
                "phase_id": phase_id,
                "rack_progress_m": progress_m,
                "rack_speed_m_s": 0.0,
                "pinion_angle_rad": -progress_m / 0.04,
                "pinion_speed_rad_s": 0.0,
                "coupling_error_m": 0.0,
                "max_support_drift_m": 0.0,
                "max_bushing_reaction_force_n": 1.0,
                "off_axis_rotation_rad": 0.0,
                "support_drifts_m": {
                    "pinion_bearing": 0.0,
                    "rack_guide": 0.0,
                },
            }

        trace = {
            "status": "COMPLETED",
            "branch": "physical",
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "mechanism_plugin_id": "rack_pinion_v1",
            "samples": [
                sample(0.0, "initial", 0.0),
                sample(0.1, "crank_lift", 0.18),
                sample(0.2, "upper_settle", 0.18),
                sample(0.3, "crank_lower", 0.0),
                sample(0.4, "home_settle", 0.0),
            ],
        }

        metrics, _thresholds, failures = _physical_behavior(
            plan,
            load_case,
            trace,
        )

        self.assertEqual(failures, [])
        self.assertAlmostEqual(
            metrics["phase_directional_travel_m"]["crank_lift"],
            0.18,
        )
        self.assertAlmostEqual(
            metrics["phase_directional_travel_m"]["crank_lower"],
            0.18,
        )

    def _task_trace(self, plan, load_case, hold_samples: list[tuple[float, float]]) -> dict:
        def sample(time_s: float, phase_id: str, progress_m: float) -> dict:
            return {
                "time_s": time_s,
                "phase_id": phase_id,
                "rack_progress_m": progress_m,
                "rack_speed_m_s": 0.0,
                "pinion_angle_rad": -progress_m / 0.04,
                "pinion_speed_rad_s": 0.0,
                "coupling_error_m": 0.0,
                "max_support_drift_m": 0.0,
                "max_bushing_reaction_force_n": 1.0,
                "off_axis_rotation_rad": 0.0,
                "support_drifts_m": {
                    "pinion_bearing": 0.0,
                    "rack_guide": 0.0,
                },
            }

        hold = [
            sample(time_s, "zero_input_hold", progress_m)
            for time_s, progress_m in hold_samples
        ]
        tail_start = (hold_samples[-1][0] if hold_samples else 0.3) + 0.05
        return {
            "status": "COMPLETED",
            "branch": "physical",
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "mechanism_plugin_id": "rack_pinion_v1",
            "samples": [
                sample(0.0, "initial", 0.0),
                sample(0.1, "crank_lift", 0.18),
                sample(0.3, "upper_settle", 0.18),
                *hold,
                sample(tail_start, "crank_lower", 0.0),
                sample(tail_start + 0.1, "home_settle", 0.0),
            ],
        }

    def test_task_anti_drop_hold_passes_when_gate_holds(self) -> None:
        plan = _plan()
        load_case = default_load_cases(plan)[-1]
        self.assertEqual(load_case.program, "task_rack_roundtrip")
        hold = [(0.3 + 0.05 * step, 0.18) for step in range(1, 11)]

        metrics, thresholds, failures = _physical_behavior(
            plan, load_case, self._task_trace(plan, load_case, hold)
        )

        self.assertEqual(failures, [])
        self.assertEqual(thresholds["min_anti_drop_hold_s"], 0.5)
        self.assertEqual(thresholds["max_anti_drop_slip_m"], 0.004)
        span = metrics["phase_time_spans_s"]["zero_input_hold"]
        self.assertAlmostEqual(span["duration_s"], 0.5)
        self.assertAlmostEqual(metrics["phase_downward_slip_m"]["zero_input_hold"], 0.0)

    def test_task_anti_drop_detects_gate_drop(self) -> None:
        plan = _plan()
        load_case = default_load_cases(plan)[-1]
        hold = [
            (0.3 + 0.05 * step, 0.18 - 0.0008 * step) for step in range(1, 11)
        ]

        metrics, _thresholds, failures = _physical_behavior(
            plan, load_case, self._task_trace(plan, load_case, hold)
        )

        self.assertIn("physical_rack_gate_drop_exceeded", failures)
        self.assertIn("physical_rack_anti_drop_not_held_at_top", failures)
        self.assertAlmostEqual(
            metrics["phase_downward_slip_m"]["zero_input_hold"], 0.008
        )

    def test_task_anti_drop_requires_minimum_hold_duration(self) -> None:
        plan = _plan()
        load_case = default_load_cases(plan)[-1]
        hold = [(0.35, 0.18), (0.40, 0.18)]

        _metrics, _thresholds, failures = _physical_behavior(
            plan, load_case, self._task_trace(plan, load_case, hold)
        )

        self.assertIn("physical_rack_anti_drop_hold_too_short", failures)

    def test_task_anti_drop_interval_missing_fails_closed(self) -> None:
        plan = _plan()
        load_case = default_load_cases(plan)[-1]

        _metrics, _thresholds, failures = _physical_behavior(
            plan, load_case, self._task_trace(plan, load_case, [])
        )

        self.assertIn("physical_rack_anti_drop_interval_missing", failures)

    def test_rejects_gate_that_cannot_hold_itself_under_gravity(self) -> None:
        plan = _plan()
        adapter = adapter_from_dict(plan.adapter_spec)
        weak_family_spec = dict(adapter.family_spec)
        weak_family_spec["nominal_input_torque_nm"] = 0.1
        weak_adapter = replace(adapter, family_spec=weak_family_spec)
        weak_plan = replace(
            plan,
            adapter_spec=adapter_to_dict(weak_adapter),
        )

        with self.assertRaises(MechanismValidationError) as caught:
            default_load_cases(weak_plan)

        self.assertEqual(
            caught.exception.code, "RACK_PINION_HOLDING_TORQUE_INSUFFICIENT"
        )


if __name__ == "__main__":
    unittest.main()
