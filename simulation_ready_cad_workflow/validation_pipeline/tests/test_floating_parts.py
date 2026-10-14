"""Advisory floating-part check on as-built geometry."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.slim_harness import execution_profile
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.floating_parts import advisory_report, check_assembly


def _bodies(folder: Path) -> list[dict]:
    import trimesh
    shapes = {
        "pin": (trimesh.creation.cylinder(radius=.002, height=.02), True),  # fixed shaft along z
        "wheel": (trimesh.creation.annulus(r_min=.0022, r_max=.008, height=.006), False),  # bore around the shaft
        "flag": (trimesh.creation.box((.01, .01, .004)), False),
    }
    positions = {"pin": [0, 0, 0], "wheel": [0, 0, 0], "flag": [.02, 0, .02]}
    bodies = []
    for name, (mesh, fixed) in shapes.items():
        path = folder / f"{name}.obj"
        mesh.export(path)
        bodies.append({"id": name, "fixed": fixed, "mesh_file": str(path), "position_m": positions[name],
                       "rotation_wxyz": [1, 0, 0, 0]})
    return bodies


JOINTS = [
    {"id": "axle", "type": "revolute", "body_a": "pin", "body_b": "wheel", "anchor": [0, 0, 0], "axis": [0, 0, 1],
     "region_a": {"radius": .002, "points": [[0, 0, -.003], [0, 0, .003]]}, "region_b": {"radius": .003, "points": []}},
    # The flag hangs 5 mm away from the wheel; this joint holds it only in the simulation.
    {"id": "flag_hinge", "type": "revolute", "body_a": "wheel", "body_b": "flag", "anchor": [.0125, 0, .015],
     "axis": [0, 1, 0], "region_a": {"radius": .004, "points": []}, "region_b": {"radius": .004, "points": []}},
]


class FloatingPartsTest(unittest.TestCase):
    def test_engaged_joints_hold_parts_and_floating_ones_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            report = check_assembly(_bodies(Path(temporary)), JOINTS)
        engaged = {row["joint"]: row["physically_engaged"] for row in report["joints"]}
        self.assertEqual(engaged, {"axle": True, "flag_hinge": False})
        self.assertEqual(report["components"], [["flag"], ["pin", "wheel"]])
        kinds = {(finding["kind"], tuple(finding.get("bodies"))) for finding in report["findings"]}
        self.assertEqual(kinds, {("JOINT_NOT_ENGAGED", ("wheel", "flag")), ("FLOATING_PART", ("flag",))})
        self.assertTrue(report["advisory"])

    def test_unreadable_geometry_gives_an_empty_advisory_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(execution_profile, "FLOATING_PART_ADVICE", True):
            output = Path(temporary) / "floating-parts.json"
            report = advisory_report([{"id": "x", "mesh_file": str(Path(temporary) / "missing.obj"),
                                       "position_m": [0, 0, 0], "rotation_wxyz": [1, 0, 0, 0]}], [], output)
            self.assertEqual(report["findings"], [])
            self.assertIn("error", json.loads(output.read_text()))

    def test_the_advice_is_a_distribution_setting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(execution_profile, "FLOATING_PART_ADVICE", False):
            output = Path(temporary) / "floating-parts.json"
            self.assertIsNone(advisory_report(_bodies(Path(temporary)), JOINTS, output))
            self.assertFalse(output.exists())


class MonitorFloatingPartsTest(unittest.TestCase):
    def test_the_newest_checked_revision_reaches_the_monitor(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.dashboard import launcher
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import write_json_atomic
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            write_json_atomic(project / "runs/r01/template_runs/a001/floating-parts.json",
                              {"findings": [{"message": "old"}, {"message": "older"}]})
            write_json_atomic(project / "runs/r02/template_runs/a001/floating-parts.json",
                              {"findings": [{"message": "lever is not physically attached"}]})
            self.assertEqual(launcher.floating_parts(project),
                             {"revision": "r02", "count": 1, "messages": ["lever is not physically attached"]})
            self.assertIsNone(launcher.floating_parts(project / "missing"))


if __name__ == "__main__":
    unittest.main()
