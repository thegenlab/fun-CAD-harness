from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCP.gp import gp_Pnt

from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    read_json,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.rigid_attachment import (
    build_rigid_attachment_proof,
    materialize_rigid_attachment_proof,
    verify_rigid_attachment_binding,
)


_HASH = "a" * 64


def _box(x: float, length: float = 10.0):
    return BRepPrimAPI_MakeBox(
        gp_Pnt(float(x), 0.0, 0.0),
        float(length),
        10.0,
        10.0,
    ).Shape()


class RigidAttachmentProofTests(unittest.TestCase):
    def test_one_solid_body_passes_without_authored_edges(self) -> None:
        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "mechanism.rigid_attachment._shape_volume_mm3",
                side_effect=AssertionError(
                    "singleton body must not evaluate solid volume"
                ),
            ) as volume,
            patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "mechanism.rigid_attachment._shape_bounds_mm",
                side_effect=AssertionError(
                    "singleton body must not evaluate solid bounds"
                ),
            ) as bounds,
        ):
            report = build_rigid_attachment_proof(
                [_box(0.0)],
                {"moving": (1,)},
                package_binding_sha256=_HASH,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["bodies"][0]["connected_components"], [["solid_1"]])
        self.assertEqual(report["bodies"][0]["proven_edge_count"], 0)
        volume.assert_not_called()
        bounds.assert_not_called()

    def test_positive_volume_chain_proves_one_fixed_body(self) -> None:
        report = build_rigid_attachment_proof(
            [_box(0.0), _box(9.0), _box(18.0)],
            {"fixed": (1, 2, 3)},
            package_binding_sha256=_HASH,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["bodies"][0]["proven_edge_count"], 2)
        for edge in report["bodies"][0]["proven_fixed_edges"]:
            self.assertEqual(edge["constraint_rank"], 6)
            self.assertEqual(edge["remaining_relative_dof"], 0)

    def test_surface_contact_does_not_create_six_dof_edge(self) -> None:
        report = build_rigid_attachment_proof(
            [_box(0.0), _box(10.0)],
            {"fixed": (1, 2)},
            package_binding_sha256=_HASH,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(report["bodies"][0]["proven_edge_count"], 0)
        self.assertEqual(
            report["bodies"][0]["connected_components"],
            [["solid_1"], ["solid_2"]],
        )
        self.assertFalse(
            report["policy"]["surface_contact_creates_fixed_edge"]
        )

    def test_floating_solid_fails_closed(self) -> None:
        report = build_rigid_attachment_proof(
            [_box(0.0), _box(25.0)],
            {"moving": (1, 2)},
            package_binding_sha256=_HASH,
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["failure_reasons"])
        self.assertTrue(report["policy"]["unknown_attachment_fails_closed"])

    def test_report_is_written_before_failed_materialization_returns(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "proof.json"
            with self.assertRaises(MechanismValidationError) as caught:
                materialize_rigid_attachment_proof(
                    [_box(0.0), _box(25.0)],
                    {"moving": (1, 2)},
                    package_binding_sha256=_HASH,
                    workspace_root=root,
                    destination=report_path,
                )

            self.assertEqual(
                caught.exception.code,
                "RIGID_ATTACHMENT_PROOF_FAILED",
            )
            self.assertTrue(report_path.is_file())
            self.assertEqual(read_json(report_path)["status"], "FAIL")

    def test_binding_is_hash_and_package_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "proof.json"
            binding = materialize_rigid_attachment_proof(
                [_box(0.0)],
                {"moving": (1,)},
                package_binding_sha256=_HASH,
                workspace_root=root,
                destination=report_path,
            )
            report = verify_rigid_attachment_binding(
                binding,
                workspace_root=root,
                package_binding_sha256=_HASH,
            )
            self.assertTrue(report["passed"])

            report["status"] = "FAIL"
            write_json_atomic(report_path, report)
            with self.assertRaises(MechanismValidationError) as caught:
                verify_rigid_attachment_binding(
                    binding,
                    workspace_root=root,
                    package_binding_sha256=_HASH,
                )
            self.assertEqual(
                caught.exception.code,
                "RIGID_ATTACHMENT_REPORT_CHANGED",
            )


if __name__ == "__main__":
    unittest.main()
