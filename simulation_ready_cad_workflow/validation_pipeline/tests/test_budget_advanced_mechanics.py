"""Advanced mechanical generation tests combining budget tier enforcement
with diverse mechanism configurations, multi-tier workflows, and
end-to-end budget lifecycle validation.

Covers: multi-body constraint systems, tier escalation decision-making,
budget checkpoint round-trips through physics, and simulation timeout
enforcement on real Chrono runs.
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

from ..slim_harness.budget import (
    TIERS,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    TierConfig,
    build_incomplete_summary,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    get_tier,
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
)

WORKSPACE = Path(__file__).resolve().parents[3]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class TestTierSelectionWorkflow(unittest.TestCase):
    """Test the full tier selection workflow for different mechanism sizes."""

    def test_simple_two_body_selects_light(self):
        """A 2-body mechanism with 1 test fits light tier."""
        bodies = [{"id": f"b{i}"} for i in range(2)]
        tests = [{"id": "t0"}]

        for name, tier in TIERS.items():
            part_ok = check_part_count(bodies, tier) == []
            task_ok = check_task_count(tests, tier) == []
            if name == "light":
                self.assertTrue(part_ok and task_ok)
                break

    def test_medium_mechanism_needs_medium_tier(self):
        """12 bodies fits medium (15) but not light (10)."""
        bodies = [{"id": f"b{i}"} for i in range(12)]
        tests = [{"id": "t0"}]

        light = get_tier("light")
        medium = get_tier("medium")

        self.assertNotEqual(check_part_count(bodies, light), [])
        self.assertEqual(check_part_count(bodies, medium), [])
        self.assertEqual(check_task_count(tests, medium), [])

    def test_complex_mechanism_needs_high_tier(self):
        """18 bodies + 4 tests requires high tier."""
        bodies = [{"id": f"b{i}"} for i in range(18)]
        tests = [{"id": f"t{i}"} for i in range(4)]

        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")

        self.assertNotEqual(check_part_count(bodies, light), [])
        self.assertNotEqual(check_part_count(bodies, medium), [])
        self.assertEqual(check_part_count(bodies, high), [])
        self.assertNotEqual(check_task_count(tests, light), [])
        self.assertNotEqual(check_task_count(tests, medium), [])
        self.assertEqual(check_task_count(tests, high), [])

    def test_auto_tier_selection_algorithm(self):
        """Simulate automatic tier selection: pick the smallest tier that fits."""
        test_cases = [
            (3, 1, "light"),
            (10, 2, "light"),
            (11, 1, "medium"),
            (15, 2, "medium"),
            (16, 1, "high"),
            (20, 5, "high"),
        ]
        tier_order = ["light", "medium", "high"]

        for n_bodies, n_tests, expected_tier in test_cases:
            bodies = [{"id": f"b{i}"} for i in range(n_bodies)]
            tests = [{"id": f"t{i}"} for i in range(n_tests)]

            selected = None
            for name in tier_order:
                tier = get_tier(name)
                if not check_part_count(bodies, tier) and not check_task_count(tests, tier):
                    selected = name
                    break

            self.assertEqual(selected, expected_tier,
                             f"{n_bodies} bodies + {n_tests} tests should select {expected_tier}")


class TestBudgetLifecycleEndToEnd(unittest.TestCase):
    """End-to-end budget lifecycle: create, track, pause, resume, checkpoint, summarize."""

    def test_full_lifecycle_light_tier(self):
        """Full lifecycle for a light-tier mechanism."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        self.assertFalse(tracker.is_exhausted)
        self.assertGreater(tracker.snapshot()["remaining_active_seconds"], 0)

        tracker.pause("geometry_prep")
        time.sleep(0.02)
        tracker.resume()

        tracker.pause("review_wait")
        time.sleep(0.02)
        tracker.resume()

        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 2)
        self.assertGreater(snap["total_pause_seconds"], 0.03)

        with tempfile.TemporaryDirectory() as tmp:
            cp_path = Path(tmp) / "checkpoint.json"
            tracker.save_checkpoint(cp_path)

            restored = ActiveBudgetTracker.restore(cp_path, tier=tier)
            restored.resume()

            self.assertAlmostEqual(
                tracker.elapsed_active,
                restored.elapsed_active,
                places=1,
            )

        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[
                {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
            ],
            pending_tests=[],
            issues=[],
            evidence_paths=["/path/to/trace.json"],
            blockers=[],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(summary["tier"]["name"], "light")
        self.assertIn("timing", summary)

    def test_exhaustion_lifecycle(self):
        """Budget exhaustion triggers correct status transitions."""
        tiny = TierConfig("tiny", 0.03, 5, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tiny)

        self.assertFalse(tracker.is_exhausted)
        time.sleep(0.04)
        self.assertTrue(tracker.is_exhausted)

        self.assertTrue(tracker.is_exhausted)
        snap = tracker.snapshot()
        self.assertLessEqual(snap["remaining_active_seconds"], 0)

        summary = build_incomplete_summary(
            tier=tiny, tracker=tracker,
            completed_tests=[],
            pending_tests=["pending_1"],
            issues=["Budget exhausted before physics"],
            evidence_paths=[],
            blockers=[],
        )
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(summary["severity"], "functional")

    def test_multi_checkpoint_lifecycle(self):
        """Multiple checkpoints track progressive work."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                tracker.pause(f"phase_{i}")
                time.sleep(0.01)
                tracker.resume()
                cp = Path(tmp) / f"cp_{i}.json"
                tracker.save_checkpoint(cp)

            cp_first = Path(tmp) / "cp_0.json"
            cp_last = Path(tmp) / "cp_4.json"

            first = json.loads(cp_first.read_text())
            last = json.loads(cp_last.read_text())

            self.assertLess(first["active_seconds"], last["active_seconds"])
            self.assertLess(first["pause_count"], last["pause_count"])


class TestBudgetRecoveryWorkflow(unittest.TestCase):
    """Test recovery workflow: retry decisions, delay calculations, summary generation."""

    def test_recovery_retry_cycle(self):
        """Full retry cycle: attempt -> fail -> delay -> retry -> succeed."""
        for attempt in range(MAX_RECOVERY_RETRIES):
            self.assertTrue(should_retry(attempt, "SIMULATION_TIMEOUT"))
            delay = recovery_delay(attempt)
            self.assertGreater(delay, 0)
            self.assertLessEqual(delay, 60)

        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "SIMULATION_TIMEOUT"))

    def test_retryable_error_types(self):
        """Infrastructure errors are retryable; budget exhaustion is not."""
        retryable = ["SIMULATION_TIMEOUT", "INFRASTRUCTURE_FAILURE", "HARNESS_ERROR"]
        non_retryable = ["ACTIVE_BUDGET_EXHAUSTED", "TERMINATED", "PASS"]

        for error in retryable:
            self.assertTrue(should_retry(0, error), f"{error} should be retryable")
            self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, error))

        for error in non_retryable:
            self.assertFalse(should_retry(0, error), f"{error} should NOT be retryable")

    def test_recovery_delay_exponential_backoff(self):
        """Recovery delays increase with attempt number."""
        delays = [recovery_delay(i) for i in range(MAX_RECOVERY_RETRIES)]
        for i in range(1, len(delays)):
            self.assertGreaterEqual(delays[i], delays[i - 1])


class TestAdjustmentLogWorkflow(unittest.TestCase):
    """Criteria adjustment logging for tier-aware calibration tracking."""

    def test_adjustment_log_round_trip(self):
        """Record adjustments, save, load, verify integrity."""
        log = AdjustmentLog()

        log.record(CriteriaAdjustment(
            "tolerance_mm", {"value": 0.1}, {"value": 0.15},
            "Relaxed tolerance for mesh contact",
        ))
        log.record(CriteriaAdjustment(
            "step_size", {"value": 0.001}, {"value": 0.0005},
            "Decreased step for stability",
        ))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)

        self.assertEqual(len(loaded.entries), 2)
        self.assertEqual(loaded.entries[0].criterion_id, "tolerance_mm")
        self.assertEqual(loaded.entries[1].criterion_id, "step_size")

    def test_frozen_criteria_always_rejected(self):
        """Frozen criteria cannot be adjusted regardless of reason."""
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                "frozen_criterion", {"v": 1}, {"v": 2},
                "Attempted change", frozen=True,
            )


class TestTierTimeoutBehavior(unittest.TestCase):
    """Detailed timeout enforcement behavior across tiers."""

    def test_timeout_clamp_within_tier(self):
        """Requested timeout is clamped to tier maximum."""
        for name, tier in TIERS.items():
            clamped = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds + 100)
            self.assertEqual(clamped, tier.simulation_timeout_seconds)

    def test_timeout_passthrough_within_tier(self):
        """Requested timeout passes through when within tier limit."""
        for name, tier in TIERS.items():
            requested = tier.simulation_timeout_seconds // 2
            if requested > 0:
                actual = enforce_simulation_timeout(tier, requested)
                self.assertEqual(actual, requested)

    def test_default_timeout_is_tier_max(self):
        """No requested timeout defaults to tier maximum."""
        for name, tier in TIERS.items():
            default = enforce_simulation_timeout(tier)
            self.assertEqual(default, tier.simulation_timeout_seconds)

    def test_tier_timeout_ordering(self):
        """Higher tiers always allow longer simulation timeouts."""
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")

        self.assertLess(light.simulation_timeout_seconds, medium.simulation_timeout_seconds)
        self.assertLess(medium.simulation_timeout_seconds, high.simulation_timeout_seconds)


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestMultiJointMechanism(unittest.TestCase):
    """Multi-joint mechanisms with diverse body types and budget tracking."""

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

    def _sphere(self, name, radius, center=(0, 0, 0)):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
        mesh.apply_translation(center)
        path = self.root / f"{name}.obj"
        mesh.export(path)
        return path

    def _prepare(self, scene):
        scene_path = self.root / "scene.json"
        scene_path.write_text(json.dumps(scene), encoding="utf-8")
        from ..slim_harness.geometry import prepare_geometry
        from ..slim_harness.contracts import compile_scene
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
        from ..slim_harness.trace_storage import open_trace
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

    def test_diverse_body_shapes_freefall(self):
        """Box, cylinder, sphere fall independently within light tier."""
        self._box("floor", [0.2, 0.01, 0.2], [0, -0.005, 0])
        self._box("cube", [0.015, 0.015, 0.015], [0, 0.1, 0])
        self._cylinder("rod", 0.005, 0.03, [-0.06, 0.15, 0])
        self._sphere("ball", 0.008, [0.06, 0.2, 0])

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": [
                {"id": "floor", "source": "floor.obj", "fixed": True, "density_kg_m3": 7800},
                {"id": "cube", "source": "cube.obj", "fixed": False, "density_kg_m3": 7800},
                {"id": "rod", "source": "rod.obj", "fixed": False, "density_kg_m3": 2700},
                {"id": "ball", "source": "ball.obj", "fixed": False, "density_kg_m3": 7800},
            ],
            "tests": [{
                "id": "fall", "purpose": "Bodies fall under gravity",
                "duration_s": 0.15, "step_s": 0.001,
                "assertions": [
                    {"id": "ball_fell", "purpose": "Ball descended",
                     "expression": "last(signal('ball.position.y_m')) < 0.15"},
                ],
            }],
        }
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        self.assertEqual(check_part_count(scene["bodies"], tier), [])

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        timeout = enforce_simulation_timeout(tier)
        trace, error = self._run(prepared, prepared["tests"][0], timeout_s=timeout)
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertTrue(trace["physical_pass"])
        self.assertFalse(tracker.is_exhausted)

    def test_prismatic_with_budget_tracking(self):
        """A driven prismatic joint with full budget tracking cycle."""
        self._box("rail", [0.2, 0.02, 0.02], [0, 0, 0])
        self._box("slider", [0.02, 0.015, 0.015], [0.05, 0.0175, 0])

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
                "id": "push", "purpose": "Slider moves under force",
                "duration_s": 0.3, "step_s": 0.001,
                "inputs": [{"joint": "slide", "mode": "effort", "value": 0.5, "max_effort": 0.5}],
                "assertions": [
                    {"id": "slid", "purpose": "Slider traveled",
                     "expression": "span(signal('slider.position.x_m')) >= 0.001"},
                ],
            }],
        }
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)

        tracker.pause("pre_geometry")
        time.sleep(0.01)
        tracker.resume()

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        timeout = enforce_simulation_timeout(tier)
        trace, error = self._run(prepared, prepared["tests"][0], timeout_s=timeout)
        self.assertIsNotNone(trace, f"Physics failed: {error}")

        snap_before_cp = tracker.snapshot()

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "checkpoint.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()

            snap_after = restored.snapshot()
            self.assertAlmostEqual(
                snap_before_cp["active_seconds"],
                snap_after["active_seconds"],
                places=1,
            )

        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{
                "test_id": "push",
                "physical_pass": trace["physical_pass"],
                "geometry_audit": {"passed": True},
            }],
            pending_tests=[],
            issues=[],
            evidence_paths=[str(self.root / "trace.json")],
            blockers=[],
        )
        self.assertIn("timing", summary)
        self.assertEqual(summary["tier"]["name"], "medium")


@unittest.skipUnless(HAS_TRIMESH, "Requires trimesh")
@unittest.skipUnless(HAS_CHRONO, "The verified Chrono runtime is unavailable")
class TestBodyCountScaling(unittest.TestCase):
    """Test that scenes with increasing body counts scale properly within tiers."""

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
        from ..slim_harness.geometry import prepare_geometry
        from ..slim_harness.contracts import compile_scene
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
        from ..slim_harness.trace_storage import open_trace
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

    def test_six_body_freefall(self):
        """6 bodies fall independently within medium tier — no mutual contact."""
        self._box("floor", [0.3, 0.01, 0.3], [0, -0.005, 0])
        bodies = [
            {"id": "floor", "source": "floor.obj", "fixed": True, "density_kg_m3": 7800},
        ]
        for i in range(5):
            name = f"block_{i}"
            x = -0.1 + i * 0.05
            self._box(name, [0.012, 0.012, 0.012], [x, 0.1 + i * 0.04, 0])
            bodies.append({
                "id": name, "source": f"{name}.obj",
                "fixed": False, "density_kg_m3": 7800,
            })

        scene = {
            "schema_version": "slim_scene_v1", "units": "m",
            "bodies": bodies,
            "tests": [{
                "id": "freefall", "purpose": "Blocks fall independently",
                "duration_s": 0.12, "step_s": 0.001,
                "assertions": [
                    {"id": "top_fell", "purpose": "Top block descended",
                     "expression": "last(signal('block_4.position.y_m')) < 0.28"},
                ],
            }],
        }
        tier = get_tier("medium")
        self.assertEqual(check_part_count(bodies, tier), [])
        self.assertEqual(len(bodies), 6)

        prepared, geometry = self._prepare(scene)
        self.assertIsNotNone(prepared, f"Geometry errors: {geometry.get('errors')}")

        trace, error = self._run(prepared, prepared["tests"][0],
                                  timeout_s=enforce_simulation_timeout(tier))
        self.assertIsNotNone(trace, f"Physics failed: {error}")
        self.assertTrue(trace["physical_pass"])


if __name__ == "__main__":
    unittest.main()
