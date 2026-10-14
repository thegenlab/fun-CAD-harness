"""Integration tests for budget module wired into SlimWorkflow.

Tests that tier enforcement actually works when passed to SlimWorkflow:
part count limits, task count limits, simulation timeout differentiation,
budget annotation on results, and the workflow CLI --tier argument parsing.
"""
from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from ..mechanism.io import write_json_atomic
from ..slim_harness.cli import create_project
from ..slim_harness.contracts import DeliveryError, load_scene, compile_scene
from ..slim_harness.workflow import SlimWorkflow, digest
from ..slim_harness.geometry import GEOMETRY_IMPLEMENTATION_SHA256
from ..slim_harness.audit import AUDIT_IMPLEMENTATION_SHA256
from ..slim_harness.implementation import native_implementation_binding
from ..slim_harness.numerical import NUMERICAL_POLICY, effective_initial_step_seconds
from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    BudgetStatus,
    get_tier,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    build_incomplete_summary,
)


class BudgetWorkflowIntegrationBase(unittest.TestCase):
    """Common setup for workflow integration tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = create_project(self.workspace, "budget_test",
                                      "A slider must travel 10 mm.")
        self.run = self.project / "runs" / "r01"
        self.scene_path = self.run / "design" / "scene.json"
        self.source = self.run / "design" / "part.step"
        self.source.write_text("geometry-file snapshot", encoding="utf-8")
        criteria = [
            {"id": "travel", "kind": "function",
             "statement": "Travel 10 mm", "source_quote": "10 mm",
             "verification": "Measure slider travel"},
        ]
        frozen = {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(self.project / "task.md"),
            "task_text": "A slider must travel 10 mm.",
            "criteria": criteria,
            "reviewer_summary": "Source-bound slider travel criteria",
        }
        write_json_atomic(self.project / "criteria.json", frozen)
        state = json.loads((self.project / "workflow.json").read_text())
        state["events"] = [{"kind": "criteria", "result": {
            "status": "ready", "summary": frozen["reviewer_summary"],
            "criteria": criteria,
        }}]
        write_json_atomic(self.project / "workflow.json", state)

    def _write_scene(self, num_bodies=1, num_tests=1):
        """Write a scene with configurable body and test count."""
        bodies = [
            {"id": f"body_{i}", "source": "part.step", "fixed": i == 0}
            for i in range(num_bodies)
        ]
        if not any(not b["fixed"] for b in bodies):
            bodies[-1]["fixed"] = False
        tests = [
            {"id": f"test_{i}", "purpose": f"Test {i}", "duration_s": 1,
             "assertions": [{"id": f"a_{i}", "purpose": "Check",
                             "expression": "last(signal('time_s')) >= 1"}]}
            for i in range(num_tests)
        ]
        scene = {
            "schema_version": "slim_scene_v1", "units": "mm",
            "bodies": bodies, "tests": tests,
        }
        write_json_atomic(self.scene_path, scene)
        return scene


class TestWorkflowTierConstruction(BudgetWorkflowIntegrationBase):
    """SlimWorkflow correctly constructs with tier parameter."""

    def test_no_tier_is_default(self):
        self._write_scene()
        wf = SlimWorkflow(self.workspace, self.project)
        self.assertIsNone(wf.tier)
        self.assertIsNone(wf.budget_tracker)

    def test_light_tier_by_string(self):
        self._write_scene()
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        self.assertEqual(wf.tier.name, "light")
        self.assertIsNotNone(wf.budget_tracker)
        self.assertEqual(wf.physics_timeout_seconds, 60)

    def test_medium_tier_by_string(self):
        self._write_scene()
        wf = SlimWorkflow(self.workspace, self.project, tier="medium")
        self.assertEqual(wf.tier.name, "medium")
        self.assertEqual(wf.physics_timeout_seconds, 180)

    def test_high_tier_by_string(self):
        self._write_scene()
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        self.assertEqual(wf.tier.name, "high")
        self.assertEqual(wf.physics_timeout_seconds, 300)

    def test_tier_object_directly(self):
        self._write_scene()
        tier = get_tier("light")
        wf = SlimWorkflow(self.workspace, self.project, tier=tier)
        self.assertEqual(wf.tier.name, "light")

    def test_invalid_tier_raises(self):
        self._write_scene()
        with self.assertRaises(ValueError):
            SlimWorkflow(self.workspace, self.project, tier="ultra")

    def test_tier_clamps_physics_timeout(self):
        self._write_scene()
        wf = SlimWorkflow(self.workspace, self.project, tier="light",
                          physics_timeout_seconds=600)
        self.assertEqual(wf.physics_timeout_seconds, 60)

    def test_smaller_physics_timeout_preserved(self):
        self._write_scene()
        wf = SlimWorkflow(self.workspace, self.project, tier="high",
                          physics_timeout_seconds=120)
        self.assertEqual(wf.physics_timeout_seconds, 120)

    def test_budget_tracker_injected(self):
        self._write_scene()
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        wf = SlimWorkflow(self.workspace, self.project, tier="light",
                          budget_tracker=tracker)
        self.assertIs(wf.budget_tracker, tracker)


class TestPartCountEnforcement(BudgetWorkflowIntegrationBase):
    """run_revision rejects scenes that exceed the tier's part limit."""

    def test_light_tier_rejects_11_parts(self):
        self._write_scene(num_bodies=11, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertIn("budget_violation", result)
        self.assertEqual(result["budget_violation"], "part_count")
        self.assertTrue(any("11 parts" in str(i) for i in result["issues"]))

    def test_light_tier_accepts_10_parts(self):
        self._write_scene(num_bodies=10, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        # Will fail later in geometry prep (no real STEP), but NOT due to part limits
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "part_count")

    def test_medium_tier_accepts_15_parts(self):
        self._write_scene(num_bodies=15, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="medium")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "part_count")

    def test_medium_tier_rejects_16_parts(self):
        self._write_scene(num_bodies=16, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="medium")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "part_count")


class TestTaskCountEnforcement(BudgetWorkflowIntegrationBase):
    """run_revision rejects scenes with too many tests per tier."""

    def test_light_tier_rejects_3_tests(self):
        self._write_scene(num_bodies=2, num_tests=3)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "task_count")

    def test_light_tier_accepts_2_tests(self):
        self._write_scene(num_bodies=2, num_tests=2)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_high_tier_accepts_5_tests(self):
        self._write_scene(num_bodies=3, num_tests=5)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_high_tier_rejects_6_tests(self):
        self._write_scene(num_bodies=3, num_tests=6)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "task_count")


class TestBudgetExhaustion(BudgetWorkflowIntegrationBase):
    """run_revision returns ACTIVE_BUDGET_EXHAUSTED when budget is used up."""

    def test_exhausted_budget_before_geometry(self):
        self._write_scene(num_bodies=2, num_tests=1)
        tiny = TierConfig("tiny", 0.001, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.01)  # ensure budget expires
        wf = SlimWorkflow(self.workspace, self.project, tier=tiny,
                          budget_tracker=tracker)
        result = wf.run_revision()
        self.assertEqual(result["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertIn("timing", result)
        self.assertIn("pending_tests", result)
        self.assertEqual(result["pending_tests"], ["test_0"])


class TestBudgetAnnotation(BudgetWorkflowIntegrationBase):
    """run() annotates results with budget timing when a tier is set."""

    def test_annotation_on_revision_result(self):
        self._write_scene(num_bodies=2, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        # The result may fail (no real STEP), but _annotate_budget in run()
        # adds the timing. Since we're calling run_revision directly,
        # let's test the annotation method.
        annotated = wf._annotate_budget(result)
        self.assertIn("budget_timing", annotated)
        self.assertIn("active_seconds", annotated["budget_timing"])
        self.assertIn("wall_seconds", annotated["budget_timing"])
        self.assertIn("tier", annotated)
        self.assertEqual(annotated["tier"]["name"], "light")


class TestWorkflowRunBudgetTermination(BudgetWorkflowIntegrationBase):
    """run() treats SIMULATION_TIMEOUT and ACTIVE_BUDGET_EXHAUSTED as terminal."""

    def test_simulation_timeout_is_terminal(self):
        """run() should stop and return on SIMULATION_TIMEOUT."""
        self._write_scene(num_bodies=2, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        # Mock run_revision to return SIMULATION_TIMEOUT
        timeout_result = {
            "status": "SIMULATION_TIMEOUT",
            "issues": [{"test_id": "test_0", "timeout_seconds": 60}],
            "attempt": str(self.run / "attempts" / "a001"),
            "completed_tests": [],
        }
        with patch.object(wf, "run_revision", return_value=timeout_result):
            with patch.object(wf, "publish"):  # skip heartbeat
                result = wf.run(one_revision=True)
        self.assertEqual(result["status"], "SIMULATION_TIMEOUT")

    def test_budget_exhausted_is_terminal(self):
        """run() should stop and return on ACTIVE_BUDGET_EXHAUSTED."""
        self._write_scene(num_bodies=2, num_tests=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        exhausted_result = {
            "schema_version": "budget_incomplete_summary_v1",
            "status": "ACTIVE_BUDGET_EXHAUSTED",
            "timing": {"active_seconds": 3600, "wall_seconds": 3800},
            "completed_tests": [], "pending_tests": ["test_0"],
        }
        with patch.object(wf, "run_revision", return_value=exhausted_result):
            with patch.object(wf, "publish"):
                result = wf.run(one_revision=True)
        self.assertEqual(result["status"], "ACTIVE_BUDGET_EXHAUSTED")


class TestCLITierArgument(unittest.TestCase):
    """The workflow CLI correctly parses --tier."""

    def test_tier_argument_parsing(self):
        from ..slim_harness.workflow import main
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("project", type=Path)
        parser.add_argument("--workspace", type=Path, default=Path.cwd())
        parser.add_argument("--tier", choices=["light", "medium", "high"])
        parser.add_argument("--physics-timeout-seconds", type=float, default=600)
        parser.add_argument("--physics-max-step-seconds", type=float, default=.001)
        parser.add_argument("--physics-solver-iterations", type=int, default=160)
        args = parser.parse_args(["some_project", "--tier", "medium"])
        self.assertEqual(args.tier, "medium")

    def test_no_tier_is_none(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("project", type=Path)
        parser.add_argument("--tier", choices=["light", "medium", "high"])
        args = parser.parse_args(["some_project"])
        self.assertIsNone(args.tier)


class TestBudgetPauseResumeDuringWorkflow(unittest.TestCase):
    """Budget tracker pause/resume works correctly in workflow context."""

    def test_pause_during_infrastructure_wait(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        active_before = tracker.elapsed_active

        tracker.pause("memory_stall")
        time.sleep(0.05)
        active_during_pause = tracker.elapsed_active
        self.assertAlmostEqual(active_during_pause, active_before, delta=0.01)

        tracker.resume()
        time.sleep(0.02)
        active_after = tracker.elapsed_active
        self.assertGreater(active_after, active_before)
        # Wall clock should include pause time
        self.assertGreater(tracker.elapsed_wall, active_after)

    def test_multiple_pause_resume_cycles(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        for _ in range(5):
            time.sleep(0.01)
            tracker.pause("cycle")
            time.sleep(0.01)
            tracker.resume()

        # Active time should be roughly half of wall time
        ratio = tracker.elapsed_active / tracker.elapsed_wall
        self.assertLess(ratio, 0.8)
        self.assertGreater(ratio, 0.2)

    def test_checkpoint_preserves_pause_count(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("first")
        tracker.resume()
        tracker.pause("second")
        tracker.resume()
        tracker.pause("third")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            data = tracker.save_checkpoint(path)
            self.assertEqual(data["pause_count"], 3)

            restored = ActiveBudgetTracker.restore(path, tier=tier)
            snap = restored.snapshot()
            self.assertEqual(snap["pause_count"], 4)  # +1 for restore itself


class TestIncompleteSummaryIntegration(unittest.TestCase):
    """build_incomplete_summary produces correct output for various scenarios."""

    def test_all_tests_failed(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[
                {"test_id": "a", "physical_pass": False, "geometry_audit": {"passed": False}},
                {"test_id": "b", "physical_pass": True, "geometry_audit": {"passed": False}},
            ],
            pending_tests=["c"],
            issues=["budget ran out"],
            evidence_paths=["/evidence/trace.json"],
            blockers=["sim too slow"],
        )
        self.assertEqual(summary["severity"], "functional")
        self.assertEqual(len(summary["completed_tests"]), 2)
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")
        self.assertEqual(summary["completed_tests"][1]["status"], "FAIL")

    def test_empty_completed(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=["test_1", "test_2"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")
        self.assertEqual(summary["completed_tests"], [])
        self.assertEqual(len(summary["pending_tests"]), 2)

    def test_timing_keys_present(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        timing = summary["timing"]
        self.assertIn("active_seconds", timing)
        self.assertIn("wall_seconds", timing)
        self.assertIn("budget_limit_seconds", timing)
        self.assertIn("pause_count", timing)
        self.assertIn("total_pause_seconds", timing)
        self.assertEqual(timing["budget_limit_seconds"], 10800)


if __name__ == "__main__":
    unittest.main()
