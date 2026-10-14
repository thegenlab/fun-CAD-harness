from __future__ import annotations

from types import SimpleNamespace
import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    AdapterSpec,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.axis_spring_spec import (
    axis_spring_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
    RevoluteKinematicThresholds,
    _revolute_behavior_failures,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    PlainRevoluteAdapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import (
    default_revolute_load_cases,
    _revolute_execution_step_s,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine_worker import (
    _axis_spring_revolute_control,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
    _axis_spring_limit_response,
)


def _spec(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "0.1",
        "mechanism_kind": "axis_spring",
        "joint_family": "revolute",
        "rest_position_si": 0.0,
        "stiffness_si": 0.18,
        "damping_si": 0.015,
        "maximum_deflection_si": 0.7,
    }
    value.update(updates)
    return value


class AxisSpringSpecTests(unittest.TestCase):
    def test_revolute_spring_is_typed_and_maps_to_local_axis_torque(self) -> None:
        parsed = axis_spring_from_dict(_spec(), expected_family="revolute")
        adapter = AdapterSpec(
            adapter_id="derived_plain_revolute_v1",
            family="revolute",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
            family_spec=parsed.to_dict(),
        )
        supports = PlainRevoluteAdapter().constitutive_supports(None, adapter)  # type: ignore[arg-type]
        self.assertEqual(len(supports), 1)
        self.assertEqual(
            supports[0].rotational_stiffness_n_m_rad,
            (0.0, 0.0, 0.18),
        )
        self.assertEqual(
            supports[0].rotational_damping_n_m_s_rad,
            (0.0, 0.0, 0.015),
        )

    def test_axis_spring_revolute_uses_finer_physics_step(self) -> None:
        spring_adapter = AdapterSpec(
            adapter_id="derived_plain_revolute_v1",
            family="revolute",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
            family_spec=_spec(),
        )
        plain_adapter = AdapterSpec(
            adapter_id="derived_plain_revolute_v1",
            family="revolute",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
        )
        self.assertEqual(_revolute_execution_step_s(spring_adapter), 0.0001)
        self.assertEqual(_revolute_execution_step_s(plain_adapter), 0.0005)

    def test_whole_machine_control_compensates_spring_only_when_driven(self) -> None:
        driven = _axis_spring_revolute_control(
            angle_rad=0.3,
            speed_rad_s=0.0,
            target_rad=0.3,
            max_control_torque_nm=0.6,
            controller_kp_nm_rad=2.5,
            controller_kd_nm_s_rad=0.35,
            spring_stiffness_nm_rad=0.32,
            spring_damping_nm_s_rad=0.035,
            passive_stage=False,
        )
        passive = _axis_spring_revolute_control(
            angle_rad=0.3,
            speed_rad_s=0.0,
            target_rad=0.0,
            max_control_torque_nm=0.6,
            controller_kp_nm_rad=2.5,
            controller_kd_nm_s_rad=0.35,
            spring_stiffness_nm_rad=0.32,
            spring_damping_nm_s_rad=0.035,
            passive_stage=True,
        )
        self.assertAlmostEqual(driven[0], 0.096)
        self.assertAlmostEqual(driven[1], -0.096)
        self.assertEqual(passive[0], 0.0)
        self.assertAlmostEqual(passive[1], -0.096)

    def test_hash_bound_bearing_and_stop_tolerance_are_bounded(self) -> None:
        parsed = axis_spring_from_dict(
            _spec(
                maximum_stop_penetration_si=0.008,
                bearing_off_axis_stiffness_si=14.0,
                bearing_off_axis_damping_si=0.18,
                moving_mass_kg=0.12,
            ),
            expected_family="revolute",
        )
        adapter = AdapterSpec(
            adapter_id="derived_plain_revolute_v1",
            family="revolute",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
            family_spec=parsed.to_dict(),
        )
        support = PlainRevoluteAdapter().physical_supports(
            None, adapter  # type: ignore[arg-type]
        )[0]
        self.assertEqual(parsed.maximum_stop_penetration_si, 0.008)
        self.assertEqual(parsed.moving_mass_kg, 0.12)
        self.assertEqual(
            support.rotational_stiffness_n_m_rad, (14.0, 14.0, 0.0)
        )
        self.assertEqual(
            support.rotational_damping_n_m_s_rad, (0.18, 0.18, 0.35)
        )
        self.assertEqual(support.source, "hash_bound_axis_spring_bearing_v1")

    def test_bounded_fixed_box_contact_fixture_is_typed(self) -> None:
        parsed = axis_spring_from_dict(
            _spec(
                contact_fixture={
                    "shape": "fixed_box",
                    "center_abs_m": [0.15, 0.0, 0.01],
                    "size_m": [0.02, 0.02, 0.01],
                    "friction": 0.5,
                    "hold_duration_s": 0.15,
                }
            ),
            expected_family="revolute",
        )
        self.assertIsNotNone(parsed.contact_fixture)
        assert parsed.contact_fixture is not None
        self.assertEqual(parsed.contact_fixture.shape, "fixed_box")
        self.assertEqual(parsed.contact_fixture.size_m, (0.02, 0.02, 0.01))
        self.assertEqual(
            parsed.to_dict()["contact_fixture"]["center_abs_m"],
            [0.15, 0.0, 0.01],
        )

    def test_dynamic_grip_box_and_loading_support_are_typed(self) -> None:
        parsed = axis_spring_from_dict(
            _spec(
                contact_fixture={
                    "shape": "dynamic_grip_box",
                    "center_abs_m": [0.158, 0.0, 0.0],
                    "size_m": [0.018, 0.010, 0.018],
                    "friction": 0.8,
                    "hold_duration_s": 0.35,
                    "mass_kg": 0.010,
                    "loading_support_center_abs_m": [0.158, 0.0, -0.010],
                    "loading_support_size_m": [0.024, 0.016, 0.002],
                }
            ),
            expected_family="revolute",
        )
        assert parsed.contact_fixture is not None
        self.assertEqual(parsed.contact_fixture.shape, "dynamic_grip_box")
        self.assertEqual(parsed.contact_fixture.mass_kg, 0.010)
        self.assertEqual(
            parsed.contact_fixture.loading_support_size_m,
            (0.024, 0.016, 0.002),
        )
        fixture = parsed.to_dict()["contact_fixture"]
        self.assertEqual(fixture["loading_support_center_abs_m"], [0.158, 0.0, -0.010])

    def test_dynamic_grip_box_without_loading_support_fails_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as raised:
            axis_spring_from_dict(
                _spec(
                    contact_fixture={
                        "shape": "dynamic_grip_box",
                        "center_abs_m": [0.158, 0.0, 0.0],
                        "size_m": [0.018, 0.010, 0.018],
                        "friction": 0.8,
                        "hold_duration_s": 0.35,
                        "mass_kg": 0.010,
                    }
                )
            )
        self.assertEqual(raised.exception.code, "AXIS_SPRING_FIXTURE_INVALID")

    def test_tabletop_pickup_box_and_carrier_offsets_are_typed(self) -> None:
        parsed = axis_spring_from_dict(
            _spec(
                contact_fixture={
                    "shape": "tabletop_pickup_box",
                    "center_abs_m": [0.158, 0.0, 0.0],
                    "size_m": [0.018, 0.004, 0.010],
                    "friction": 0.8,
                    "hold_duration_s": 0.35,
                    "mass_kg": 0.010,
                    "table_center_abs_m": [0.158, 0.0, -0.007],
                    "table_size_m": [0.120, 0.080, 0.004],
                    "approach_offset_abs_m": [-0.050, 0.0, 0.0],
                    "grip_offset_abs_m": [0.0, 0.0, 0.0],
                    "lift_offset_abs_m": [0.0, 0.0, 0.030],
                }
            )
        )
        assert parsed.contact_fixture is not None
        self.assertEqual(parsed.contact_fixture.shape, "tabletop_pickup_box")
        self.assertEqual(
            parsed.contact_fixture.table_size_m,
            (0.120, 0.080, 0.004),
        )
        self.assertEqual(
            parsed.to_dict()["contact_fixture"]["lift_offset_abs_m"],
            [0.0, 0.0, 0.030],
        )

    def test_tabletop_pickup_box_without_table_fails_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as raised:
            axis_spring_from_dict(
                _spec(
                    contact_fixture={
                        "shape": "tabletop_pickup_box",
                        "center_abs_m": [0.158, 0.0, 0.0],
                        "size_m": [0.018, 0.004, 0.010],
                        "friction": 0.8,
                        "hold_duration_s": 0.35,
                        "mass_kg": 0.010,
                        "approach_offset_abs_m": [-0.050, 0.0, 0.0],
                        "grip_offset_abs_m": [0.0, 0.0, 0.0],
                        "lift_offset_abs_m": [0.0, 0.0, 0.030],
                    }
                )
            )
        self.assertEqual(raised.exception.code, "AXIS_SPRING_FIXTURE_INVALID")

    def test_dynamic_grip_box_gets_three_machine_specific_load_cases(self) -> None:
        family_spec = _spec(
            contact_fixture={
                "shape": "dynamic_grip_box",
                "center_abs_m": [0.158, 0.0, 0.0],
                "size_m": [0.018, 0.010, 0.018],
                "friction": 0.8,
                "hold_duration_s": 0.35,
                "mass_kg": 0.010,
                "loading_support_center_abs_m": [0.158, 0.0, -0.010],
                "loading_support_size_m": [0.024, 0.016, 0.002],
            }
        )
        plan = SimpleNamespace(
            family="revolute",
            adapter_id=None,
            adapter_spec={
                "adapter_id": "derived_plain_revolute_v1",
                "family": "revolute",
                "family_spec": family_spec,
                "whole_machine_angular_task": {
                    "outbound_task_id": "grip_standard_test_block",
                    "return_task_id": "release_standard_test_block",
                    "outbound_angle_rad": 0.31,
                    "home_angle_rad": 0.0,
                    "moving_mass_kg": 0.30,
                    "outbound_torque_nm": 0.40,
                    "return_torque_nm": 0.0,
                    "max_torque_phase_s": 0.50,
                    "settle_phase_s": 0.05,
                },
            },
            axis_abs=(0.0, 0.0, 1.0),
            gravity_abs_m_s2=(0.0, 0.0, -9.81),
            moving_extents_m=(0.18, 0.06, 0.04),
        )
        cases = default_revolute_load_cases(plan)  # type: ignore[arg-type]
        task_cases = {case.load_case_id: case for case in cases if case.load_case_id.startswith("tong_")}
        self.assertEqual(
            set(task_cases),
            {"tong_grip_object", "tong_hold_object", "tong_release_object"},
        )
        self.assertEqual(task_cases["tong_hold_object"].program, "tong_hold_object")
        self.assertEqual(
            [phase.phase_id for phase in task_cases["tong_release_object"].phases],
            [
                "tong_release_close",
                "tong_release_preload",
                "tong_release_passive_return",
                "tong_release_observe",
            ],
        )
        self.assertEqual(task_cases["tong_release_object"].phases[-1].wrench_kind, "none")

    def test_tabletop_pickup_box_gets_three_tabletop_task_programs(self) -> None:
        family_spec = _spec(
            contact_fixture={
                "shape": "tabletop_pickup_box",
                "center_abs_m": [0.158, 0.0, 0.0],
                "size_m": [0.018, 0.004, 0.010],
                "friction": 0.8,
                "hold_duration_s": 0.35,
                "mass_kg": 0.010,
                "grip_torque_nm": 0.80,
                "table_center_abs_m": [0.158, 0.0, -0.007],
                "table_size_m": [0.120, 0.080, 0.004],
                "approach_offset_abs_m": [-0.050, 0.0, 0.0],
                "grip_offset_abs_m": [0.0, 0.0, 0.0],
                "lift_offset_abs_m": [0.0, 0.0, 0.030],
                "lift_duration_s": 0.80,
            }
        )
        plan = SimpleNamespace(
            family="revolute",
            adapter_id=None,
            adapter_spec={
                "adapter_id": "derived_plain_revolute_v1",
                "family": "revolute",
                "family_spec": family_spec,
                "whole_machine_angular_task": {
                    "outbound_task_id": "pickup_standard_test_block_from_table",
                    "return_task_id": "release_standard_test_block_to_table",
                    "outbound_angle_rad": 0.31,
                    "home_angle_rad": 0.0,
                    "moving_mass_kg": 0.30,
                    "outbound_torque_nm": 0.40,
                    "return_torque_nm": 0.0,
                    "max_torque_phase_s": 0.50,
                    "settle_phase_s": 0.05,
                },
            },
            axis_abs=(0.0, 0.0, 1.0),
            gravity_abs_m_s2=(0.0, 0.0, -9.81),
            moving_extents_m=(0.18, 0.06, 0.04),
        )
        cases = default_revolute_load_cases(plan)  # type: ignore[arg-type]
        task_cases = {
            case.load_case_id: case
            for case in cases
            if case.load_case_id.startswith("tong_tabletop_")
        }
        self.assertEqual(
            set(task_cases),
            {
                "tong_tabletop_pickup",
                "tong_tabletop_hold",
                "tong_tabletop_release",
            },
        )
        self.assertEqual(
            [
                phase.phase_id
                for phase in task_cases["tong_tabletop_pickup"].phases
            ],
            [
                "tong_pickup_initial_table_settle",
                "tong_pickup_approach",
                "tong_pickup_close",
                "tong_pickup_grip_verify",
                "tong_pickup_lift",
                "tong_pickup_verify",
            ],
        )
        release_phases = [
            phase.phase_id
            for phase in task_cases["tong_tabletop_release"].phases
        ]
        self.assertIn("tong_release_final_table_settle", release_phases)
        self.assertEqual(release_phases[-1], "tong_release_observe")
        for load_case in task_cases.values():
            lift_phase = next(
                phase for phase in load_case.phases if phase.phase_id.endswith("_lift")
            )
            self.assertAlmostEqual(lift_phase.duration_s, 0.80)
            for phase in load_case.phases:
                if phase.wrench_kind == "torque":
                    self.assertAlmostEqual(phase.magnitude_si, 0.80)

    def test_tabletop_pickup_lift_duration_defaults_and_is_shape_scoped(self) -> None:
        fixture = {
            "shape": "tabletop_pickup_box",
            "center_abs_m": [0.158, 0.0, 0.0],
            "size_m": [0.018, 0.004, 0.010],
            "friction": 0.8,
            "hold_duration_s": 0.35,
            "mass_kg": 0.010,
            "table_center_abs_m": [0.158, 0.0, -0.007],
            "table_size_m": [0.120, 0.080, 0.004],
            "approach_offset_abs_m": [-0.050, 0.0, 0.0],
            "grip_offset_abs_m": [0.0, 0.0, 0.0],
            "lift_offset_abs_m": [0.0, 0.0, 0.030],
        }
        parsed = axis_spring_from_dict(_spec(contact_fixture=fixture))
        self.assertIsNotNone(parsed.contact_fixture)
        self.assertAlmostEqual(parsed.contact_fixture.lift_duration_s, 0.40)
        self.assertIsNone(parsed.contact_fixture.grip_torque_nm)

        invalid = {
            "shape": "fixed_box",
            "center_abs_m": [0.15, 0.0, 0.01],
            "size_m": [0.02, 0.02, 0.01],
            "friction": 0.5,
            "hold_duration_s": 0.15,
            "lift_duration_s": 0.80,
        }
        with self.assertRaises(MechanismValidationError) as raised:
            axis_spring_from_dict(_spec(contact_fixture=invalid))
        self.assertEqual(raised.exception.code, "AXIS_SPRING_FIXTURE_INVALID")

        invalid_grip_torque = {
            "shape": "fixed_box",
            "center_abs_m": [0.15, 0.0, 0.01],
            "size_m": [0.02, 0.02, 0.01],
            "friction": 0.5,
            "hold_duration_s": 0.15,
            "grip_torque_nm": 0.80,
        }
        with self.assertRaises(MechanismValidationError) as raised:
            axis_spring_from_dict(
                _spec(contact_fixture=invalid_grip_torque)
            )
        self.assertEqual(raised.exception.code, "AXIS_SPRING_FIXTURE_INVALID")

    def test_tong_task_programs_are_known_to_revolute_comparison(self) -> None:
        metrics = {
            "max_pivot_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "phase_directional_travel_rad": {},
            "phase_angle_bounds_rad": {},
        }
        for program in (
            "tong_grip_object",
            "tong_hold_object",
            "tong_release_object",
            "tong_tabletop_pickup",
            "tong_tabletop_hold",
            "tong_tabletop_release",
        ):
            case = LoadCase(
                load_case_id=program,
                family="revolute",
                program=program,
                purpose="nominal",
                moving_mass_kg=0.1,
                step_s=0.00025,
                phases=(LoadPhase("observe", "none", None, 0.0, 0.05),),
            )
            self.assertEqual(
                _revolute_behavior_failures(
                    metrics,
                    branch="physical",
                    load_case=case,
                    thresholds=RevoluteKinematicThresholds(),
                ),
                [],
            )

    def test_ramped_torque_phase_serializes_and_interpolates(self) -> None:
        phase = LoadPhase(
            "controlled_unload",
            "torque",
            (0.0, 0.0, 1.0),
            0.8,
            0.4,
            end_magnitude_si=0.0,
        )
        self.assertAlmostEqual(phase.magnitude_at_fraction(0.0), 0.8)
        self.assertAlmostEqual(phase.magnitude_at_fraction(0.5), 0.4)
        self.assertAlmostEqual(phase.magnitude_at_fraction(1.0), 0.0)
        self.assertEqual(
            LoadPhase.from_dict(phase.to_dict()),
            phase,
        )

    def test_ramped_unload_is_not_required_to_move_with_residual_wrench(self) -> None:
        phase = LoadPhase(
            "controlled_unload",
            "torque",
            (0.0, 0.0, 1.0),
            0.8,
            0.4,
            end_magnitude_si=0.0,
        )
        case = LoadCase(
            load_case_id="controlled_release",
            family="revolute",
            program="tong_tabletop_release",
            purpose="nominal",
            moving_mass_kg=0.1,
            phases=(phase,),
        )
        failures = _revolute_behavior_failures(
            {
                "max_pivot_drift_m": 0.0,
                "max_off_axis_rotation_rad": 0.0,
                "phase_directional_travel_rad": {"controlled_unload": 0.0},
                "phase_angle_bounds_rad": {},
            },
            branch="reduced",
            load_case=case,
            thresholds=RevoluteKinematicThresholds(),
        )
        self.assertEqual(failures, [])

    def test_contact_fixture_unknown_fields_fail_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as raised:
            axis_spring_from_dict(
                _spec(
                    contact_fixture={
                        "shape": "fixed_box",
                        "center_abs_m": [0.15, 0.0, 0.01],
                        "size_m": [0.02, 0.02, 0.01],
                        "friction": 0.5,
                        "hold_duration_s": 0.15,
                        "product_role": "jaw_target",
                    }
                )
            )
        self.assertEqual(
            raised.exception.code,
            "AXIS_SPRING_FIXTURE_FIELD_UNKNOWN",
        )

    def test_physical_limit_response_is_finite_and_restoring(self) -> None:
        lower = _axis_spring_limit_response(-0.1, -0.2, 0.5)
        upper = _axis_spring_limit_response(0.6, 0.2, 0.5)
        inside = _axis_spring_limit_response(0.2, 10.0, 0.5)
        exiting_lower = _axis_spring_limit_response(-0.1, 0.2, 0.5)

        self.assertEqual(lower[2], "lower")
        self.assertGreater(lower[0], 0.0)
        self.assertAlmostEqual(lower[1], 0.1)
        self.assertEqual(upper[2], "upper")
        self.assertLess(upper[0], 0.0)
        self.assertAlmostEqual(upper[1], 0.1)
        self.assertEqual(inside, (0.0, 0.0, None))
        self.assertAlmostEqual(exiting_lower[0], 30.0)

    def test_family_mismatch_fails_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as raised:
            axis_spring_from_dict(_spec(), expected_family="prismatic")
        self.assertEqual(raised.exception.code, "AXIS_SPRING_FAMILY_MISMATCH")

    def test_unknown_field_and_out_of_range_values_fail_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as unknown:
            axis_spring_from_dict(_spec(product_id="kitchen_tongs"))
        self.assertEqual(unknown.exception.code, "AXIS_SPRING_FIELD_UNKNOWN")
        with self.assertRaises(MechanismValidationError) as excessive:
            axis_spring_from_dict(_spec(stiffness_si=1001.0))
        self.assertEqual(excessive.exception.code, "AXIS_SPRING_VALUE_OUT_OF_RANGE")
        with self.assertRaises(MechanismValidationError) as excessive_stop:
            axis_spring_from_dict(_spec(maximum_stop_penetration_si=1.0))
        self.assertEqual(
            excessive_stop.exception.code, "AXIS_SPRING_VALUE_OUT_OF_RANGE"
        )


if __name__ == "__main__":
    unittest.main()
