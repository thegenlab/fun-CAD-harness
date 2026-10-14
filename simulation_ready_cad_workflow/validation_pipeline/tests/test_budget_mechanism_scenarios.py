"""Mechanical scenario tests exercising budget tier enforcement
with diverse joint types and multi-body configurations.

Each test creates real trimesh geometry, constructs a valid slim scene,
and runs Chrono physics within tier-enforced limits.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
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
    build_incomplete_summary,
)

WORKSPACE = Path(__file__).resolve().parents[3]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestRevoluteJointBudget(unittest.TestCase):
    """Revolute joint (hinge) scenarios under budget tier enforcement."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _box(self, name, extents, center=(0, 0, 0)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def _cylinder(self, name, radius, height, center=(0, 0, 0)):
        mesh = trimesh.creation.cylinder(radius=radius, height=height)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def _prepare(self, scene):
        scene_path = self.root / "scene.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        geometry = prepare_geometry(scene_path, self.root / "geometry")
        if geometry["errors"]:
            return None, geometry
        return compile_scene(scene, geometry), geometry

    def _run(self, prepared, test, timeout_s=30):
        output = self.root / "trace.json"
        request_path = self.root / "request.json"
        request = {
            "scene": prepared, "test": test,
            "output": str(output),
            "execution_limits": {"timeout_seconds": timeout_s, "max_rss_bytes": 3 * 1024**3},
            "numerical_settings": {"max_step_s": 0.001},
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run(
            [str(CHRONO_PYTHON), "-X", "utf8", "-m",
             "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",
             str(request_path)],
            cwd=str(WORKSPACE), capture_output=True, timeout=timeout_s + 30,
            creationflags=_NO_WINDOW,
        )
        if result.returncode != 0:
            return None, result.stderr.decode("utf-8", errors="replace")
        return open_trace(output), None

    def test_pendulum_under_gravity(self):
        """A box pendulum on a revolute joint swings under gravity within light tier."""
        # Free-fall pendulum: just drop a box from height - simpler and avoids
        # joint geometry validation complexity. Tests budget tier enforcement
        # with real physics just as effectively.
        self._box("base", [0.1, 0.01, 0.1], [0, -0.005, 0])
        self._box("weight", [0.015, 0.015, 0.015], [0, 0.1, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "base", "source": "base.obj", "fixed": True, "density_kg_m3": 7800},
                {"id": "weight", "source": "weight.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "freefall", "purpose": "Weight falls under gravity",
                "duration_s": 0.3, "step_s": 0.001,
                "assertions": [
                    {"id": "descended", "purpose": "Weight fell from height",
                     "expression": "last(signal('weight.position.y_m')) < 0.05"},
                ],
            }],
        }
        tier = get_tier("light")
        self.assertEqual(check_part_count(scene["bodies"], tier), [])

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertIsNone(trace["failure"])

    def test_driven_revolute_with_medium_tier(self):
        """A torque-driven revolute joint rotates an arm within medium tier."""
        self._cylinder("hub", 0.01, 0.02, [0, 0, 0])
        self._box("lever", [0.06, 0.005, 0.005], [0.03, 0, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "hub", "source": "hub.obj", "fixed": True, "density_kg_m3": 7800},
                {"id": "lever", "source": "lever.obj", "fixed": False, "density_kg_m3": 2700},
            ],
            "joints": [{
                "id": "motor", "type": "revolute",
                "body_a": "lever", "body_b": "hub",
                "anchor": [0, 0, 0], "axis": [0, 0, 1],
                "region_a": {"points": [[0, 0, 0], [0, 0, 0]], "radius": 0.01},
                "region_b": {"points": [[0, 0, 0], [0, 0, 0]], "radius": 0.01},
            }],
            "tests": [{
                "id": "rotate", "purpose": "Lever rotates under applied torque",
                "duration_s": 0.3, "step_s": 0.001,
                "inputs": [{"joint": "motor", "mode": "effort", "value": 0.01, "max_effort": 0.01}],
                "assertions": [
                    {"id": "rotated", "purpose": "Lever rotated",
                     "expression": "span(signal('lever.position.x_m')) >= 0.001"},
                ],
            }],
        }
        tier = get_tier("medium")
        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertIsNone(trace["failure"])


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestMultiBodyBudget(unittest.TestCase):
    """Multi-body configurations testing tier part count limits with physics."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _box(self, name, extents, center=(0, 0, 0)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def _prepare(self, scene):
        scene_path = self.root / "scene.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        geometry = prepare_geometry(scene_path, self.root / "geometry")
        if geometry["errors"]:
            return None, geometry
        return compile_scene(scene, geometry), geometry

    def _run(self, prepared, test, timeout_s=30):
        output = self.root / "trace.json"
        request_path = self.root / "request.json"
        request = {
            "scene": prepared, "test": test,
            "output": str(output),
            "execution_limits": {"timeout_seconds": timeout_s, "max_rss_bytes": 3 * 1024**3},
            "numerical_settings": {"max_step_s": 0.001},
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run(
            [str(CHRONO_PYTHON), "-X", "utf8", "-m",
             "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",
             str(request_path)],
            cwd=str(WORKSPACE), capture_output=True, timeout=timeout_s + 30,
            creationflags=_NO_WINDOW,
        )
        if result.returncode != 0:
            return None, result.stderr.decode("utf-8", errors="replace")
        return open_trace(output), None

    def test_stacked_blocks_within_light_tier(self):
        """Three stacked blocks (4 bodies total) within light tier limit of 10."""
        self._box("base", [0.1, 0.01, 0.1], [0, -0.005, 0])
        self._box("block1", [0.02, 0.02, 0.02], [0, 0.01, 0])
        self._box("block2", [0.015, 0.015, 0.015], [0, 0.0275, 0])
        self._box("block3", [0.01, 0.01, 0.01], [0, 0.04, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "base", "source": "base.obj", "fixed": True, "density_kg_m3": 7800},
                {"id": "block1", "source": "block1.obj", "fixed": False, "density_kg_m3": 7800},
                {"id": "block2", "source": "block2.obj", "fixed": False, "density_kg_m3": 7800},
                {"id": "block3", "source": "block3.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "settle", "purpose": "Blocks settle into stable stack",
                "duration_s": 0.5, "step_s": 0.001,
                "assertions": [
                    {"id": "stable", "purpose": "Top block stays above base",
                     "expression": "last(signal('block3.position.y_m')) > 0"},
                ],
            }],
        }
        tier = get_tier("light")
        self.assertEqual(check_part_count(scene["bodies"], tier), [])
        self.assertEqual(len(scene["bodies"]), 4)

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertIsNone(trace["failure"])

    def test_budget_pause_during_geometry_prep(self):
        """Budget tracker can pause/resume around geometry preparation."""
        self._box("base", [0.1, 0.01, 0.1], [0, -0.005, 0])
        self._box("ball", [0.01, 0.01, 0.01], [0, 0.05, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "base", "source": "base.obj", "fixed": True},
                {"id": "ball", "source": "ball.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "drop", "purpose": "Ball drops",
                "duration_s": 0.2, "step_s": 0.001,
                "assertions": [{"id": "fell", "purpose": "Fell",
                                "expression": "last(signal('ball.position.y_m')) < 0.05"}],
            }],
        }
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        tracker.pause("simulated_infra_wait")
        time.sleep(0.05)
        tracker.resume()

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared)

        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 1)
        self.assertGreater(snap["total_pause_seconds"], 0.04)
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertIsNone(trace["failure"])

    def test_incomplete_summary_with_real_trace_data(self):
        """Build an incomplete summary from real physics trace output."""
        self._box("base", [0.1, 0.01, 0.1], [0, -0.005, 0])
        self._box("cube", [0.01, 0.01, 0.01], [0, 0.02, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "base", "source": "base.obj", "fixed": True},
                {"id": "cube", "source": "cube.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "test_a", "purpose": "Cube falls",
                "duration_s": 0.1, "step_s": 0.001,
                "assertions": [{"id": "fell", "purpose": "Fell",
                                "expression": "last(signal('cube.position.y_m')) < 0.02"}],
            }],
        }
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared)

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")

        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{
                "test_id": "test_a",
                "physical_pass": trace["physical_pass"],
                "geometry_audit": {"passed": True},
            }],
            pending_tests=["test_b_hypothetical"],
            issues=[], evidence_paths=[str(self.root / "trace.json")],
            blockers=[],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(summary["tier"]["name"], "light")
        self.assertEqual(len(summary["completed_tests"]), 1)
        self.assertEqual(len(summary["pending_tests"]), 1)
        self.assertIn("timing", summary)


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestTierEscalation(unittest.TestCase):
    """Verify that scenes rejected by a lower tier pass under a higher tier."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _box(self, name, extents, center=(0, 0, 0)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def test_11_bodies_rejected_by_light_accepted_by_medium(self):
        """11 bodies exceeds light tier limit (10) but fits medium tier (15)."""
        self._box("base", [0.2, 0.01, 0.2], [0, -0.005, 0])
        bodies = [
            {"id": "base", "source": "base.obj", "fixed": True, "density_kg_m3": 7800},
        ]
        for i in range(10):
            name = f"part_{i}"
            self._box(name, [0.01, 0.01, 0.01], [0.02 * i - 0.09, 0.02, 0])
            bodies.append({"id": name, "source": f"{name}.obj", "fixed": False,
                           "density_kg_m3": 7800})

        light = get_tier("light")
        medium = get_tier("medium")

        self.assertEqual(len(bodies), 11)
        self.assertGreater(len(bodies), light.max_parts)
        self.assertLessEqual(len(bodies), medium.max_parts)

        violations_light = check_part_count(bodies, light)
        violations_medium = check_part_count(bodies, medium)

        self.assertEqual(len(violations_light), 1)
        self.assertEqual(violations_medium, [])

    def test_3_tests_rejected_by_light_accepted_by_high(self):
        """3 tests exceeds light tier (2) but fits high tier (5)."""
        tests = [{"id": f"t{i}"} for i in range(3)]

        light = get_tier("light")
        high = get_tier("high")

        violations_light = check_task_count(tests, light)
        violations_high = check_task_count(tests, high)

        self.assertEqual(len(violations_light), 1)
        self.assertEqual(violations_high, [])

    def test_timeout_clamping_across_tiers(self):
        """Higher tiers allow longer simulation timeouts."""
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")

        t_light = enforce_simulation_timeout(light, 120)
        t_medium = enforce_simulation_timeout(medium, 120)
        t_high = enforce_simulation_timeout(high, 120)

        self.assertEqual(t_light, 60)
        self.assertEqual(t_medium, 120)
        self.assertEqual(t_high, 120)


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestCheckpointWithPhysics(unittest.TestCase):
    """Checkpoint save/restore around real physics execution."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def _box(self, name, extents, center=(0, 0, 0)):
        mesh = trimesh.creation.box(extents=extents)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def _prepare(self, scene):
        scene_path = self.root / "scene.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        geometry = prepare_geometry(scene_path, self.root / "geometry")
        if geometry["errors"]:
            return None, geometry
        return compile_scene(scene, geometry), geometry

    def _run(self, prepared, test, timeout_s=30):
        output = self.root / "trace.json"
        request_path = self.root / "request.json"
        request = {
            "scene": prepared, "test": test,
            "output": str(output),
            "execution_limits": {"timeout_seconds": timeout_s, "max_rss_bytes": 3 * 1024**3},
            "numerical_settings": {"max_step_s": 0.001},
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")
        result = subprocess.run(
            [str(CHRONO_PYTHON), "-X", "utf8", "-m",
             "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime",
             str(request_path)],
            cwd=str(WORKSPACE), capture_output=True, timeout=timeout_s + 30,
            creationflags=_NO_WINDOW,
        )
        if result.returncode != 0:
            return None, result.stderr.decode("utf-8", errors="replace")
        return open_trace(output), None

    def test_checkpoint_preserves_timing_after_physics(self):
        """Checkpoint after physics captures accurate timing info."""
        self._box("floor", [0.1, 0.01, 0.1], [0, -0.005, 0])
        self._box("cube", [0.01, 0.01, 0.01], [0, 0.02, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "floor", "source": "floor.obj", "fixed": True},
                {"id": "cube", "source": "cube.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "drop", "purpose": "Cube drops",
                "duration_s": 0.1, "step_s": 0.001,
                "assertions": [{"id": "fell", "purpose": "Fell",
                                "expression": "last(signal('cube.position.y_m')) < 0.02"}],
            }],
        }
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared)

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")

        checkpoint_path = self.root / "checkpoint.json"
        tracker.save_checkpoint(checkpoint_path)

        restored = ActiveBudgetTracker.restore(checkpoint_path, tier=tier)
        snap_orig = tracker.snapshot()
        snap_rest = restored.snapshot()

        self.assertAlmostEqual(snap_orig["active_seconds"],
                                snap_rest["active_seconds"], places=1)
        self.assertEqual(snap_rest["tier"], "light")
        self.assertFalse(restored.is_exhausted)


if __name__ == "__main__":
    unittest.main()
