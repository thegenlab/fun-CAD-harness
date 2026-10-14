"""Full physics integration tests for budget tier enforcement.

Creates real trimesh geometry, runs actual Chrono simulations through
the slim_harness runtime subprocess, and validates that budget constraints
and timeout handling work correctly with live physics.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

CHRONO_PYTHON = Path(os.environ.get(
    "SIMREADY_TEST_CHRONO_PYTHON",
    str(Path.home() / ".conda/envs/chrono_py312_projectchrono/python.exe"),
))
HAS_CHRONO = CHRONO_PYTHON.is_file()

from ..slim_harness.contracts import compile_scene
from ..slim_harness.geometry import prepare_geometry
from ..slim_harness.trace_storage import open_trace
from ..slim_harness.budget import (
    ActiveBudgetTracker,
    get_tier,
    enforce_simulation_timeout,
    check_part_count,
    check_task_count,
)

WORKSPACE = Path(__file__).resolve().parents[3]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestBudgetPhysicsExecution(unittest.TestCase):
    """Run real Chrono simulations within tier-enforced timeout limits."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _create_box_mesh(self, name, extents, center=(0, 0, 0)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def _prepare_scene(self, scene):
        scene_path = self.root / "scene.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        geometry = prepare_geometry(scene_path, self.root / "geometry")
        if geometry["errors"]:
            return None, geometry
        prepared = compile_scene(scene, geometry)
        return prepared, geometry

    def _run_physics(self, prepared, test, timeout_s=30, max_step_s=0.001):
        output = self.root / "physical.trace.json"
        request_path = self.root / "request.json"
        request = {
            "scene": prepared, "test": test,
            "output": str(output),
            "execution_limits": {"timeout_seconds": timeout_s, "max_rss_bytes": 3 * 1024**3},
            "numerical_settings": {"max_step_s": max_step_s},
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run(
            [str(CHRONO_PYTHON), "-X", "utf8", "-m",
             "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",
             str(request_path)],
            cwd=str(WORKSPACE),
            capture_output=True, timeout=timeout_s + 30,
            creationflags=_NO_WINDOW,
        )
        if result.returncode != 0:
            return None, result.stderr.decode("utf-8", errors="replace")
        return open_trace(output), None

    def test_simple_gravity_drop_within_light_tier(self):
        """A single box dropping under gravity completes within light tier timeout."""
        self._create_box_mesh("base", [0.1, 0.01, 0.1], [0, -0.005, 0])
        self._create_box_mesh("ball", [0.01, 0.01, 0.01], [0, 0.02, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "base", "source": "base.obj", "fixed": True, "density_kg_m3": 7800},
                {"id": "ball", "source": "ball.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "drop", "purpose": "Ball falls onto base",
                "duration_s": 0.2, "step_s": 0.001,
                "assertions": [{"id": "fell", "purpose": "Ball descended",
                                "expression": "last(signal('ball.position.y_m')) < 0.02"}],
            }],
        }
        tier = get_tier("light")
        self.assertEqual(check_part_count(scene["bodies"], tier), [])
        self.assertEqual(check_task_count(scene["tests"], tier), [])

        prepared, geometry = self._prepare_scene(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        timeout = enforce_simulation_timeout(tier)
        trace, error = self._run_physics(prepared, prepared["tests"][0], timeout_s=timeout)
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertIsNone(trace["failure"])
        self.assertTrue(trace["physical_pass"])
        self.assertLess(trace["elapsed_seconds"], tier.simulation_timeout_seconds)

    def test_slider_with_drive_within_medium_tier(self):
        """A driven slider completes within medium tier constraints."""
        self._create_box_mesh("rail", [0.2, 0.02, 0.02], [0, 0, 0])
        self._create_box_mesh("slider", [0.02, 0.015, 0.015], [0.05, 0.0175, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "rail", "source": "rail.obj", "fixed": True, "density_kg_m3": 7800},
                {"id": "slider", "source": "slider.obj", "fixed": False, "density_kg_m3": 2700},
            ],
            "joints": [{
                "id": "slide", "type": "prismatic",
                "body_a": "slider", "body_b": "rail",
                "anchor": [0.05, 0.0175, 0], "axis": [1, 0, 0],
                "region_a": {"points": [[0.05, 0.0175, 0], [0.05, 0.0175, 0]], "radius": 0.015},
                "region_b": {"points": [[0.05, 0.0175, 0], [0.05, 0.0175, 0]], "radius": 0.015},
            }],
            "tests": [{
                "id": "travel", "purpose": "Slider travels under force",
                "duration_s": 0.3, "step_s": 0.001,
                "inputs": [{"joint": "slide", "mode": "effort", "value": 0.5, "max_effort": 0.5}],
                "assertions": [
                    {"id": "moved", "purpose": "Slider moved forward",
                     "expression": "span(signal('slider.position.x_m')) >= 0.001"},
                ],
            }],
        }
        tier = get_tier("medium")
        self.assertEqual(check_part_count(scene["bodies"], tier), [])
        self.assertEqual(check_task_count(scene["tests"], tier), [])

        prepared, geometry = self._prepare_scene(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        timeout = enforce_simulation_timeout(tier)
        trace, error = self._run_physics(prepared, prepared["tests"][0], timeout_s=timeout)
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertIsNone(trace["failure"])
        self.assertTrue(trace["physical_pass"])

    def test_budget_tracker_measures_physics_time(self):
        """ActiveBudgetTracker records elapsed time during physics execution."""
        self._create_box_mesh("box", [0.02, 0.02, 0.02], [0, 0.02, 0])
        self._create_box_mesh("floor", [0.1, 0.01, 0.1], [0, -0.005, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "floor", "source": "floor.obj", "fixed": True},
                {"id": "box", "source": "box.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "settle", "purpose": "Box settles on floor",
                "duration_s": 0.1, "step_s": 0.001,
                "assertions": [{"id": "time", "purpose": "Completed",
                                "expression": "last(signal('time_s')) >= 0.1"}],
            }],
        }
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        before_active = tracker.elapsed_active

        prepared, geometry = self._prepare_scene(scene)
        self.assertIsNotNone(prepared)

        trace, error = self._run_physics(prepared, prepared["tests"][0],
                                          timeout_s=enforce_simulation_timeout(tier))
        after_active = tracker.elapsed_active

        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertGreater(after_active, before_active)
        self.assertFalse(tracker.is_exhausted)

    def test_timeout_enforcement_on_very_short_budget(self):
        """A very short timeout actually stops the simulation."""
        self._create_box_mesh("box", [0.02, 0.02, 0.02], [0, 0.1, 0])
        self._create_box_mesh("floor", [0.1, 0.01, 0.1], [0, -0.005, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "floor", "source": "floor.obj", "fixed": True},
                {"id": "box", "source": "box.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "long", "purpose": "Very long test",
                "duration_s": 600, "step_s": 0.0001,
                "assertions": [{"id": "impossible", "purpose": "Never completes",
                                "expression": "last(signal('time_s')) >= 600"}],
            }],
        }
        prepared, geometry = self._prepare_scene(scene)
        self.assertIsNotNone(prepared)

        output = self.root / "timeout_trace.json"
        request_path = self.root / "timeout_request.json"
        request = {
            "scene": prepared, "test": prepared["tests"][0],
            "output": str(output),
            "execution_limits": {"timeout_seconds": 2.0, "max_rss_bytes": 3 * 1024**3},
            "numerical_settings": {"max_step_s": 0.0001},
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")

        from ..runtime.bounded_subprocess import run_bounded_subprocess, BoundedSubprocessError
        with self.assertRaises(BoundedSubprocessError) as ctx:
            run_bounded_subprocess(
                [str(CHRONO_PYTHON), "-X", "utf8", "-m",
                 "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",
                 str(request_path)],
                cwd=WORKSPACE, timeout_seconds=2.0, max_rss_bytes=3 * 1024**3,
            )
        self.assertEqual(ctx.exception.code, "BOUNDED_SUBPROCESS_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
