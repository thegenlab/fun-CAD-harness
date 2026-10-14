"""End-to-end workflow scenario tests for budget tier enforcement.

Realistic scenarios exercising the full SlimWorkflow + budget integration:
- Slider mechanism with tier enforcement through workflow
- Multi-body assemblies hitting tier limits
- Budget exhaustion producing correct terminal state
- Tier escalation workflow (light -> medium -> high)
- Checkpoint recovery mid-workflow
- Adjustment log alongside workflow runs
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ..mechanism.io import write_json_atomic
from ..slim_harness.cli import create_project
from ..slim_harness.workflow import SlimWorkflow
from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    get_tier,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    build_incomplete_summary,
)


class SliderMechanismBase(unittest.TestCase):
    """Setup a slider mechanism project for realistic testing."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = create_project(self.workspace, "slider_mech",
                                      "A slider must travel 10 mm along a prismatic joint.")
        self.run = self.project / "runs" / "r01"
        self.scene_path = self.run / "design" / "scene.json"
        self.source = self.run / "design" / "part.step"
        self.source.write_text("geometry-file snapshot", encoding="utf-8")
        criteria = [
            {"id": "travel", "kind": "function",
             "statement": "Slider travels 10mm", "source_quote": "travel 10 mm",
             "verification": "Measure slider displacement along prismatic axis"},
        ]
        frozen = {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(self.project / "task.md"),
            "task_text": "A slider must travel 10 mm along a prismatic joint.",
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

    def _write_slider_scene(self, body_count=3, test_count=1):
        """Write a slider mechanism scene with base + slider + guide bodies."""
        bodies = [{"id": "base", "source": "part.step", "fixed": True}]
        for i in range(1, body_count):
            bodies.append({
                "id": f"slider_{i}",
                "source": "part.step",
                "fixed": False,
            })
        tests = [
            {"id": f"travel_test_{i}", "purpose": f"Verify slider {i} travel",
             "duration_s": 2,
             "assertions": [{"id": f"a_{i}", "purpose": "Displacement check",
                             "expression": "last(signal('time_s')) >= 2"}]}
            for i in range(test_count)
        ]
        scene = {
            "schema_version": "slim_scene_v1", "units": "mm",
            "bodies": bodies, "tests": tests,
        }
        write_json_atomic(self.scene_path, scene)
        return scene


class TestSliderTierEnforcement(SliderMechanismBase):
    """Slider mechanism enforces tier limits correctly."""

    def test_light_tier_3_body_slider_accepted(self):
        """A 3-body slider (base + 2 sliders) fits in light tier."""
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "part_count")
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_light_tier_11_body_slider_rejected(self):
        """An 11-body slider exceeds light tier's 10-part limit."""
        self._write_slider_scene(body_count=11, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "part_count")

    def test_medium_tier_15_body_accepted(self):
        """15-body assembly fits in medium tier."""
        self._write_slider_scene(body_count=15, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="medium")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "part_count")

    def test_high_tier_20_body_accepted(self):
        """20-body assembly fits in high tier."""
        self._write_slider_scene(body_count=20, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "part_count")

    def test_high_tier_21_body_rejected(self):
        """21-body assembly exceeds high tier's 20-part limit."""
        self._write_slider_scene(body_count=21, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "part_count")


class TestSliderTaskLimits(SliderMechanismBase):
    """Slider mechanism task limits enforce correctly."""

    def test_light_tier_2_tests_accepted(self):
        self._write_slider_scene(body_count=3, test_count=2)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_light_tier_3_tests_rejected(self):
        self._write_slider_scene(body_count=3, test_count=3)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "task_count")

    def test_high_tier_5_tests_accepted(self):
        self._write_slider_scene(body_count=3, test_count=5)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertNotEqual(result.get("budget_violation"), "task_count")

    def test_high_tier_6_tests_rejected(self):
        self._write_slider_scene(body_count=3, test_count=6)
        wf = SlimWorkflow(self.workspace, self.project, tier="high")
        result = wf.run_revision()
        self.assertEqual(result["status"], "NEEDS_REVISION")
        self.assertEqual(result["budget_violation"], "task_count")


class TestSliderBudgetExhaustion(SliderMechanismBase):
    """Slider mechanism produces correct terminal state on budget exhaustion."""

    def test_tiny_budget_exhausts_before_physics(self):
        self._write_slider_scene(body_count=3, test_count=1)
        tiny = TierConfig("tiny", 0.005, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.01)
        wf = SlimWorkflow(self.workspace, self.project, tier=tiny,
                          budget_tracker=tracker)
        result = wf.run_revision()
        self.assertEqual(result["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertIn("timing", result)
        self.assertIn("pending_tests", result)

    def test_exhaustion_timing_is_plausible(self):
        self._write_slider_scene(body_count=3, test_count=1)
        tiny = TierConfig("tiny", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
        wf = SlimWorkflow(self.workspace, self.project, tier=tiny,
                          budget_tracker=tracker)
        result = wf.run_revision()
        self.assertEqual(result["status"], "ACTIVE_BUDGET_EXHAUSTED")
        timing = result["timing"]
        self.assertGreater(timing["active_seconds"], 0)
        self.assertGreater(timing["wall_seconds"], 0)


class TestTierEscalationWorkflow(SliderMechanismBase):
    """Simulating tier escalation: light -> medium -> high."""

    def test_escalation_increases_budget(self):
        """Each tier upgrade increases the budget tracker's limit."""
        self._write_slider_scene(body_count=3, test_count=1)

        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")

        t_light = ActiveBudgetTracker(tier=light)
        t_medium = ActiveBudgetTracker(tier=medium)
        t_high = ActiveBudgetTracker(tier=high)

        self.assertEqual(t_light.tier.active_budget_seconds, 3600)
        self.assertEqual(t_medium.tier.active_budget_seconds, 7200)
        self.assertEqual(t_high.tier.active_budget_seconds, 10800)

    def test_escalation_widens_part_limits(self):
        """Higher tiers accept more parts."""
        parts_11 = [{} for _ in range(11)]
        parts_16 = [{} for _ in range(16)]
        parts_21 = [{} for _ in range(21)]

        self.assertGreater(len(check_part_count(parts_11, get_tier("light"))), 0)
        self.assertEqual(check_part_count(parts_11, get_tier("medium")), [])

        self.assertGreater(len(check_part_count(parts_16, get_tier("medium"))), 0)
        self.assertEqual(check_part_count(parts_16, get_tier("high")), [])

        self.assertGreater(len(check_part_count(parts_21, get_tier("high"))), 0)

    def test_escalation_with_checkpoint_transfer(self):
        """Active time from lower tier carries to higher tier via checkpoint."""
        light = get_tier("light")
        tracker = ActiveBudgetTracker(tier=light)
        time.sleep(0.03)
        active_so_far = tracker.elapsed_active

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "escalate.json"
            tracker.save_checkpoint(cp)

            high = get_tier("high")
            restored = ActiveBudgetTracker.restore(cp, tier=high)
            restored.resume()

            self.assertEqual(restored.tier.name, "high")
            self.assertAlmostEqual(restored.elapsed_active, active_so_far, delta=0.02)
            self.assertGreater(restored.remaining_active, 10700)


class TestCheckpointMidWorkflow(SliderMechanismBase):
    """Checkpoint during a workflow run."""

    def test_checkpoint_captures_pre_revision_state(self):
        """Checkpoint before run_revision captures clean tracker state."""
        self._write_slider_scene(body_count=3, test_count=1)
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "pre.json"
            tracker.save_checkpoint(cp)

            wf = SlimWorkflow(self.workspace, self.project, tier=tier,
                              budget_tracker=tracker)
            wf.run_revision()

            post_cp = Path(tmp) / "post.json"
            tracker.save_checkpoint(post_cp)

            pre = json.loads(cp.read_text())
            post = json.loads(post_cp.read_text())

            self.assertGreater(post["active_seconds"], pre["active_seconds"])
            self.assertGreater(post["wall_seconds"], pre["wall_seconds"])

    def test_restored_tracker_continues_counting(self):
        """Tracker restored from checkpoint continues accumulating time."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "mid.json"
            tracker.save_checkpoint(cp)
            active_at_save = tracker.elapsed_active

            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
            time.sleep(0.02)

            self.assertGreater(restored.elapsed_active, active_at_save)


class TestAdjustmentLogWithWorkflow(SliderMechanismBase):
    """Adjustment log persisted alongside workflow runs."""

    def test_log_saved_in_project_directory(self):
        """Adjustment log can be saved in the project's run directory."""
        self._write_slider_scene(body_count=3, test_count=1)

        log = AdjustmentLog()
        log.record(CriteriaAdjustment(
            "slider_tolerance", {"mm": 0.1}, {"mm": 0.5},
            "Relaxed tolerance for coarse mesh",
        ))

        log_path = self.run / "adjustment_log.json"
        log.save(log_path)
        self.assertTrue(log_path.exists())

        loaded = AdjustmentLog.load(log_path)
        self.assertEqual(len(loaded.entries), 1)
        self.assertEqual(loaded.entries[0].criterion_id, "slider_tolerance")

    def test_frozen_benchmark_cannot_be_adjusted(self):
        """Frozen benchmark criteria raise before any log entry is created."""
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                "benchmark_travel_mm",
                {"threshold": 10.0},
                {"threshold": 12.0},
                "Tried to relax benchmark",
                frozen=True,
            )


class TestTimeoutEnforcementInWorkflow(SliderMechanismBase):
    """Workflow correctly enforces tier timeout limits."""

    def test_light_tier_clamps_to_60s(self):
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light",
                          physics_timeout_seconds=600)
        self.assertEqual(wf.physics_timeout_seconds, 60)

    def test_medium_tier_clamps_to_180s(self):
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="medium",
                          physics_timeout_seconds=600)
        self.assertEqual(wf.physics_timeout_seconds, 180)

    def test_high_tier_clamps_to_300s(self):
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="high",
                          physics_timeout_seconds=600)
        self.assertEqual(wf.physics_timeout_seconds, 300)

    def test_smaller_timeout_preserved(self):
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="high",
                          physics_timeout_seconds=30)
        self.assertEqual(wf.physics_timeout_seconds, 30)


class TestBudgetAnnotation(SliderMechanismBase):
    """Workflow annotates results with budget metadata."""

    def test_annotated_result_has_budget_timing(self):
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project, tier="light")
        result = wf.run_revision()
        annotated = wf._annotate_budget(result)
        self.assertIn("budget_timing", annotated)
        self.assertIn("active_seconds", annotated["budget_timing"])
        self.assertIn("tier", annotated)
        self.assertEqual(annotated["tier"]["name"], "light")

    def test_no_tier_no_annotation(self):
        self._write_slider_scene(body_count=3, test_count=1)
        wf = SlimWorkflow(self.workspace, self.project)
        result = wf.run_revision()
        annotated = wf._annotate_budget(result)
        self.assertNotIn("budget_timing", annotated)


if __name__ == "__main__":
    unittest.main()
