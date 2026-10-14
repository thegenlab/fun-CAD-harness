from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.rack_pinion_runtime_v10 import (
    _component_ids_for_body_refs,
    _bounded_operator_torque_axis_nm,
)


class RackPinionOperatorTorqueTests(unittest.TestCase):
    def test_speed_damping_cannot_reverse_declared_operator_direction(self) -> None:
        self.assertLessEqual(
            _bounded_operator_torque_axis_nm(
                commanded_torque_axis_nm=-1.0,
                pinion_speed_rad_s=-4.0,
                nominal_input_torque_nm=1.0,
            ),
            0.0,
        )
        self.assertGreaterEqual(
            _bounded_operator_torque_axis_nm(
                commanded_torque_axis_nm=1.0,
                pinion_speed_rad_s=4.0,
                nominal_input_torque_nm=1.0,
            ),
            0.0,
        )

    def test_bound_endpoint_refs_resolve_to_component_ids(self) -> None:
        self.assertEqual(
            _component_ids_for_body_refs(
                "knob_body",
                (
                    "knob_body/knob_body_component_001",
                    "knob_body/knob_body_component_003",
                ),
            ),
            (
                "knob_body_component_001",
                "knob_body_component_003",
            ),
        )


if __name__ == "__main__":
    unittest.main()
