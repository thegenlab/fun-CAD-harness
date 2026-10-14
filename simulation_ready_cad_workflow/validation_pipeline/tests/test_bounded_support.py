from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bounded_support import (
    BoundedSupportRuntime,
    CONTINUE_WITH_ZERO_FORCE,
    RELATIVE_ANCHOR_DRIFT,
    TRANSVERSE_ANCHOR_DRIFT,
    bounded_support_gate,
)


class BoundedSupportTests(unittest.TestCase):
    @staticmethod
    def gate(*, coordinate_mode: str = RELATIVE_ANCHOR_DRIFT) -> dict:
        return bounded_support_gate(
            body_a_id="frame",
            body_b_id="member",
            endpoint_a_component_refs=("frame/solid_1",),
            endpoint_b_component_refs=("member/solid_2",),
            maximum_drift_m=0.01,
            coordinate_mode=coordinate_mode,
            maximum_axis_misalignment_rad=0.1,
            template_id="unit_test_capture_v1",
            proof_source="unit_test",
        )

    def test_separated_support_transmits_zero_without_terminating(self) -> None:
        runtime = BoundedSupportRuntime(
            interface_id="support-pin",
            gate=self.gate(),
        )
        engaged = runtime.evaluate(
            time_s=0.0,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.001, 0.0, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.0, 0.0, 0.0),
            requested_effort_si=25.0,
            axis_misalignment_rad=0.0,
        )
        self.assertTrue(engaged["transmission_active"])
        self.assertEqual(engaged["transmitted_effort_si"], 25.0)

        separated = runtime.evaluate(
            time_s=0.01,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.02, 0.0, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(1.0, 0.0, 0.0),
            requested_effort_si=500.0,
            axis_misalignment_rad=0.0,
        )
        self.assertFalse(separated["transmission_active"])
        self.assertEqual(separated["transmitted_effort_si"], 0.0)
        self.assertEqual(
            separated["loss_policy"],
            CONTINUE_WITH_ZERO_FORCE,
        )
        self.assertTrue(runtime.disengaged())

    def test_prismatic_gate_ignores_legitimate_slide_but_not_escape(self) -> None:
        runtime = BoundedSupportRuntime(
            interface_id="support-track",
            gate=self.gate(coordinate_mode=TRANSVERSE_ANCHOR_DRIFT),
        )
        sliding = runtime.evaluate(
            time_s=0.0,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.25, 0.001, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.5, 0.0, 0.0),
            requested_effort_si=40.0,
            free_axis_abs=(1.0, 0.0, 0.0),
            axis_misalignment_rad=0.0,
        )
        self.assertTrue(sliding["transmission_active"])
        self.assertAlmostEqual(sliding["support_drift_m"], 0.001)

        escaped = runtime.evaluate(
            time_s=0.01,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.25, 0.02, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.0, 1.0, 0.0),
            requested_effort_si=400.0,
            free_axis_abs=(1.0, 0.0, 0.0),
            axis_misalignment_rad=0.0,
        )
        self.assertFalse(escaped["transmission_active"])
        self.assertEqual(escaped["transmitted_effort_si"], 0.0)

    def test_axis_skew_disengages_and_zeroes_force(self) -> None:
        runtime = BoundedSupportRuntime(
            interface_id="support-track",
            gate=self.gate(coordinate_mode=TRANSVERSE_ANCHOR_DRIFT),
        )
        runtime.evaluate(
            time_s=0.0,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.2, 0.001, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.5, 0.0, 0.0),
            requested_effort_si=40.0,
            free_axis_abs=(1.0, 0.0, 0.0),
            axis_misalignment_rad=0.0,
        )
        skewed = runtime.evaluate(
            time_s=0.01,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.2, 0.001, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.5, 0.0, 0.0),
            requested_effort_si=400.0,
            free_axis_abs=(1.0, 0.0, 0.0),
            axis_misalignment_rad=0.11,
        )

        self.assertFalse(skewed["transmission_active"])
        self.assertEqual(skewed["transmitted_effort_si"], 0.0)

    def test_specific_capture_loss_disables_force_without_exception(self) -> None:
        runtime = BoundedSupportRuntime(
            interface_id="support-sleeve",
            gate=self.gate(coordinate_mode=TRANSVERSE_ANCHOR_DRIFT),
        )
        runtime.evaluate(
            time_s=0.0,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.0, 0.0, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.0, 0.0, 0.0),
            requested_effort_si=5.0,
            free_axis_abs=(1.0, 0.0, 0.0),
            axis_misalignment_rad=0.0,
            additional_constraints_satisfied=True,
        )
        separated = runtime.evaluate(
            time_s=0.01,
            anchor_a_abs_m=(0.0, 0.0, 0.0),
            anchor_b_abs_m=(0.0, 0.0, 0.0),
            velocity_a_abs_m_s=(0.0, 0.0, 0.0),
            velocity_b_abs_m_s=(0.0, 0.0, 0.0),
            requested_effort_si=100.0,
            free_axis_abs=(1.0, 0.0, 0.0),
            axis_misalignment_rad=0.0,
            additional_constraints_satisfied=False,
            additional_metrics={"finite_capture_overlap_m": 0.0},
            additional_blockers=("finite_capture_overlap_lost",),
        )
        self.assertFalse(separated["transmission_active"])
        self.assertEqual(separated["transmitted_effort_si"], 0.0)


if __name__ == "__main__":
    unittest.main()
