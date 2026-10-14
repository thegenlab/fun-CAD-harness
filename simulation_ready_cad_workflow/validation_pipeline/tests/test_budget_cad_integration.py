"""End-to-end budget integration with real CAD geometry.

Generates actual STEP files using build123d, creates valid slim scenes,
and validates that tier enforcement works through geometry preparation.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..mechanism.io import write_json_atomic
from ..slim_harness.cli import create_project
from ..slim_harness.workflow import SlimWorkflow
from ..slim_harness.budget import (
    TierConfig,
    ActiveBudgetTracker,
    get_tier,
)


def _make_box_step(path: Path, size_mm: float = 20.0) -> Path:
    """Generate a real STEP box using build123d."""
    from build123d import Box, Part, export_step
    box = Box(size_mm, size_mm, size_mm)
    part = Part() + box
    export_step(part, str(path))
    return path


def _make_cylinder_step(path: Path, radius_mm: float = 5.0, height_mm: float = 30.0) -> Path:
    """Generate a real STEP cylinder."""
    from build123d import Cylinder, Part, export_step
    cyl = Cylinder(radius_mm, height_mm)
    part = Part() + cyl
    export_step(part, str(path))
    return path


class TestBudgetWithRealCAD(unittest.TestCase):
    """Budget enforcement with real STEP geometry through geometry preparation."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = create_project(
            self.workspace, "cad_budget_test",
            "A box must slide along a rail."
        )
        self.run = self.project / "runs" / "r01"
        self.design = self.run / "design"
        criteria = [
            {"id": "travel", "kind": "function",
             "statement": "Box travels at least 5 mm",
             "source_quote": "slide along a rail",
             "verification": "Physical test"},
        ]
        frozen = {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(self.project / "task.md"),
            "task_text": "A box must slide along a rail.",
            "criteria": criteria,
            "reviewer_summary": "Source-bound slider criteria",
        }
        write_json_atomic(self.project / "criteria.json", frozen)
        state = json.loads((self.project / "workflow.json").read_text())
        state["events"] = [{"kind": "criteria", "result": {
            "status": "ready", "summary": frozen["reviewer_summary"],
            "criteria": criteria,
        }}]
        write_json_atomic(self.project / "workflow.json", state)

    def _write_real_scene(self, num_dynamic_bodies=1):
        """Write a scene with real STEP geometry."""
        # Generate STEP files
        _make_box_step(self.design / "base.step", 50.0)
        bodies = [
            {"id": "base", "source": "base.step", "fixed": True},
        ]
        for i in range(num_dynamic_bodies):
            name = f"slider_{i}.step"
            _make_box_step(self.design / name, 10.0)
            bodies.append({
                "id": f"slider_{i}", "source": name, "fixed": False,
                "density_kg_m3": 7800,
            })
        scene = {
            "schema_version": "slim_scene_v1",
            "units": "mm",
            "bodies": bodies,
            "joints": [
                {
                    "id": f"slide_{i}",
                    "type": "prismatic",
                    "body_a": f"slider_{i}",
                    "body_b": "base",
                    "anchor": [0, 0, 15.0 * (i + 1)],
                    "axis": [1, 0, 0],
                    "region_a": {"points": [[0, 0, 15.0 * (i + 1)], [0, 0, 15.0 * (i + 1)]], "radius": 6.0},
                    "region_b": {"points": [[0, 0, 15.0 * (i + 1)], [0, 0, 15.0 * (i + 1)]], "radius": 6.0},
                }
                for i in range(num_dynamic_bodies)
            ],
            "tests": [
                {
                    "id": "travel_test",
                    "purpose": "Measure slider travel under applied force",
                    "duration_s": 0.5,
                    "inputs": [
                        {"joint": "slide_0", "mode": "effort", "value": 1.0, "max_effort": 1.0}
                    ],
                    "assertions": [
                        {"id": "slider_moves", "criterion_ids": ["travel"],
                         "purpose": "Slider travels",
                         "expression": "span(signal('slider_0.position.x_m')) >= 0.001"}
                    ],
                },
            ],
        }
        write_json_atomic(self.design / "scene.json", scene)
        return scene

    def test_real_geometry_passes_tier_part_limit(self):
        """A scene with 2 bodies (1 fixed + 1 dynamic) passes light tier limits."""
        self._write_real_scene(num_dynamic_bodies=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        # Should NOT fail due to part count (2 bodies <= 10)
        self.assertNotEqual(result.get("budget_violation"), "part_count")
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_real_geometry_exceeds_light_tier_part_limit(self):
        """11 dynamic bodies + 1 fixed = 12 bodies, exceeding light tier limit of 10."""
        self._write_real_scene(num_dynamic_bodies=11)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "part_count")
        self.assertTrue(any("12 parts" in str(i) for i in result["issues"]))

    def test_real_geometry_within_medium_tier(self):
        """12 bodies fits within medium tier limit of 15."""
        self._write_real_scene(num_dynamic_bodies=11)
        wf = SlimWorkflow(self.workspace, self.project, tier="medium")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "part_count")

    def test_no_tier_allows_any_count(self):
        """Without a tier, no part/task limits are enforced."""
        self._write_real_scene(num_dynamic_bodies=11)
        wf = SlimWorkflow(self.workspace, self.project)
        result = wf.run_revision()
        self.assertNotIn("budget_violation", result)

    def test_budget_timing_recorded_after_geometry_prep(self):
        """Budget tracker records nonzero active time after geometry preparation."""
        self._write_real_scene(num_dynamic_bodies=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertIsNotNone(wf.budget_tracker)
        self.assertGreater(wf.budget_tracker.elapsed_active, 0)
        self.assertGreater(wf.budget_tracker.elapsed_wall, 0)


class TestBudgetWithMultipleTests(unittest.TestCase):
    """Task count enforcement with real geometry and multiple tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = create_project(
            self.workspace, "multi_test",
            "A box must slide and then return."
        )
        self.run = self.project / "runs" / "r01"
        self.design = self.run / "design"
        criteria = [
            {"id": "travel", "kind": "function",
             "statement": "Slides and returns",
             "source_quote": "slide and then return",
             "verification": "Physical test"},
        ]
        frozen = {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(self.project / "task.md"),
            "task_text": "A box must slide and then return.",
            "criteria": criteria,
            "reviewer_summary": "Slider return criteria",
        }
        write_json_atomic(self.project / "criteria.json", frozen)
        state = json.loads((self.project / "workflow.json").read_text())
        state["events"] = [{"kind": "criteria", "result": {
            "status": "ready", "summary": frozen["reviewer_summary"],
            "criteria": criteria,
        }}]
        write_json_atomic(self.project / "workflow.json", state)

    def _write_scene_with_tests(self, num_tests):
        _make_box_step(self.design / "base.step", 50.0)
        _make_box_step(self.design / "slider.step", 10.0)
        tests = [
            {
                "id": f"test_{i}",
                "purpose": f"Test phase {i}",
                "duration_s": 0.5,
                "inputs": [{"joint": "slide", "mode": "effort", "value": 1.0, "max_effort": 1.0}],
                "assertions": [
                    {"id": f"a_{i}", "criterion_ids": ["travel"],
                     "purpose": "Check",
                     "expression": "last(signal('time_s')) >= 0.5"}
                ],
            }
            for i in range(num_tests)
        ]
        scene = {
            "schema_version": "slim_scene_v1", "units": "mm",
            "bodies": [
                {"id": "base", "source": "base.step", "fixed": True},
                {"id": "slider", "source": "slider.step", "fixed": False},
            ],
            "joints": [{
                "id": "slide", "type": "prismatic",
                "body_a": "slider", "body_b": "base",
                "anchor": [0, 0, 0], "axis": [1, 0, 0],
                "region_a": {"points": [[0, 0, 0], [0, 0, 0]], "radius": 6.0},
                "region_b": {"points": [[0, 0, 0], [0, 0, 0]], "radius": 6.0},
            }],
            "tests": tests,
        }
        write_json_atomic(self.design / "scene.json", scene)

    def test_3_tests_rejected_by_light_tier(self):
        self._write_scene_with_tests(3)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "task_count")

    def test_2_tests_accepted_by_light_tier(self):
        self._write_scene_with_tests(2)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_5_tests_accepted_by_high_tier(self):
        self._write_scene_with_tests(5)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_6_tests_rejected_by_high_tier(self):
        self._write_scene_with_tests(6)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "task_count")


if __name__ == "__main__":
    unittest.main()
