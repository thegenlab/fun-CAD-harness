from __future__ import annotations

import unittest
from unittest.mock import patch

from ..mechanism.io import canonical_sha256
from ..mechanism.models import MechanismValidationError
from ..mechanism.rotary_test_programs import (
    ROTARY_TEST_PROGRAM_REGISTRY_SHA256,
    materialize_rotary_test_program,
    rotary_program_view,
    rotary_test_program,
    rotary_test_program_registry_document,
)


class RotaryTestProgramTests(unittest.TestCase):
    def test_multi_cycle_is_exactly_twenty_complete_roundtrips(self) -> None:
        program = rotary_test_program("task_output_angle_multi_cycle")
        phases = materialize_rotary_test_program(
            program.program_id,
            axis_abs=(0.0, 0.0, 1.0),
            drive_torque_nm=5.0,
            return_torque_nm=2.0,
            max_torque_phase_s=0.2,
            settle_phase_s=0.05,
        )

        self.assertEqual(program.cycle_count, 20)
        self.assertEqual(len(phases), 80)
        self.assertEqual(tuple(item.phase_id for item in phases), program.phase_ids)
        self.assertEqual(len(program.powered_phase_ids), 20)
        self.assertEqual(len(program.adaptive_phase_ids), 40)
        for offset in range(0, len(phases), 4):
            drive, drive_settle, returned, return_settle = phases[offset : offset + 4]
            self.assertEqual(drive.stop_condition, "position_at_least")
            self.assertEqual(drive.target_position_si, 0.4)
            self.assertEqual(drive.direction_abs, (0.0, 0.0, 1.0))
            self.assertEqual(drive_settle.wrench_kind, "none")
            self.assertEqual(returned.stop_condition, "position_at_most")
            self.assertEqual(returned.target_position_si, 0.0)
            self.assertEqual(returned.direction_abs, (-0.0, -0.0, -1.0))
            self.assertEqual(return_settle.wrench_kind, "none")

    def test_roundtrip_view_is_exact_complete_first_cycle(self) -> None:
        view = rotary_program_view(
            "task_output_angle_multi_cycle",
            "task_output_angle_roundtrip",
        )

        self.assertEqual(
            view.phase_ids,
            (
                "cycle_01_drive",
                "cycle_01_drive_settle",
                "cycle_01_return",
                "cycle_01_return_settle",
            ),
        )
        self.assertEqual(view.powered_phase_ids, ("cycle_01_drive",))
        self.assertEqual(view.return_phase_ids, view.phase_ids[1:])

    def test_registry_document_is_pinned(self) -> None:
        document = rotary_test_program_registry_document()
        core = dict(document)
        recorded = core.pop("registry_sha256")
        self.assertEqual(recorded, ROTARY_TEST_PROGRAM_REGISTRY_SHA256)
        self.assertEqual(canonical_sha256(core), recorded)

    def test_registry_tamper_fails_closed(self) -> None:
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "rotary_test_programs.ROTARY_TEST_PROGRAM_REGISTRY_SHA256",
            "0" * 64,
        ):
            with self.assertRaises(MechanismValidationError) as caught:
                rotary_test_program_registry_document()
        self.assertEqual(
            caught.exception.code,
            "ROTARY_TEST_PROGRAM_REGISTRY_TAMPERED",
        )

    def test_unknown_program_and_view_fail_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as program_error:
            rotary_test_program("invented_program")
        self.assertEqual(
            program_error.exception.code,
            "ROTARY_TEST_PROGRAM_UNREGISTERED",
        )
        with self.assertRaises(MechanismValidationError) as view_error:
            rotary_program_view(
                "task_output_angle_multi_cycle",
                "invented_view",
            )
        self.assertEqual(
            view_error.exception.code,
            "ROTARY_PROGRAM_VIEW_UNREGISTERED",
        )

    def test_materializer_rejects_unbounded_parameters(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            materialize_rotary_test_program(
                "task_output_angle_multi_cycle",
                axis_abs=(0.0, 0.0, 2.0),
                drive_torque_nm=5.0,
                return_torque_nm=2.0,
                max_torque_phase_s=0.2,
                settle_phase_s=0.05,
            )
        self.assertEqual(
            caught.exception.code,
            "ROTARY_TEST_PROGRAM_AXIS_INVALID",
        )


if __name__ == "__main__":
    unittest.main()
