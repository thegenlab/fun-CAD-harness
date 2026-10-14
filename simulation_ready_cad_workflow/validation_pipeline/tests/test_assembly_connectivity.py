from __future__ import annotations

import unittest

import trimesh

from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    physical_support_geometry,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.assembly_connectivity import (
    build_assembly_connectivity_report,
)


_HASH = "a" * 64
_DEFLECTION_M = 0.0002
_ENVELOPE_M = 0.000012
_PENETRATION_M = 0.00035


def _box(x_m: float, solid_id: str) -> tuple[str, trimesh.Trimesh]:
    mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
    mesh.apply_translation((x_m, 0.0, 0.0))
    return solid_id, mesh


def _report(
    bodies: dict[str, tuple[tuple[str, trimesh.Trimesh], ...]],
) -> dict:
    return build_assembly_connectivity_report(
        bodies,
        package_binding_sha256=_HASH,
        mesh_linear_deflection_m=_DEFLECTION_M,
        collision_envelope_m=_ENVELOPE_M,
        interpenetration_tolerance_m=_PENETRATION_M,
    )


class AssemblyConnectivityTests(unittest.TestCase):
    def test_rejected_axis_template_design_is_a_guarded_tombstone(self) -> None:
        self.assertTrue(physical_support_geometry.DEPRECATED)
        self.assertIn(
            "assembly_connectivity",
            physical_support_geometry.REPLACEMENT_MODULE,
        )
        with self.assertRaises(RuntimeError):
            physical_support_geometry.build_physical_support_geometry_report

    def test_one_body_passes_without_an_external_support(self) -> None:
        report = _report({"mechanism": (_box(0.0, "solid_1"),)})

        self.assertTrue(report["passed"])
        self.assertEqual(report["assembly_islands"], [["mechanism"]])
        self.assertFalse(report["joint_declarations_used"])

    def test_normal_small_assembly_clearance_creates_an_edge(self) -> None:
        report = _report(
            {
                "left": (_box(0.0, "solid_1"),),
                "right": (_box(0.0105, "solid_2"),),
            }
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["connected_edge_count"], 1)
        self.assertLess(
            report["pair_measurements"][0]["surface_gap_m"],
            report["clearance_policy"]["maximum_surface_gap_m"],
        )

    def test_floating_body_fails_closed_and_reports_gap(self) -> None:
        report = _report(
            {
                "main": (_box(0.0, "solid_1"),),
                "floating": (_box(0.030, "solid_2"),),
            }
        )

        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["assembly_island_count"], 2)
        self.assertAlmostEqual(
            report["pair_measurements"][0]["surface_gap_m"],
            0.020,
            places=8,
        )
        self.assertIn("20.000 mm gap", report["failure_reasons"][1])

    def test_task_environment_cannot_hide_a_machine_island(self) -> None:
        report = _report(
            {
                "main": (_box(0.0, "solid_1"),),
                "floating": (_box(0.030, "solid_2"),),
            }
        )

        self.assertFalse(report["task_objects_can_create_edges"])
        self.assertFalse(report["environment_can_create_edges"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
