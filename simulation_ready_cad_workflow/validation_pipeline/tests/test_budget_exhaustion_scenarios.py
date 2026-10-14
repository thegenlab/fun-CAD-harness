"""Budget exhaustion scenario tests.

Exercises every path around budget exhaustion: near-boundary detection,
post-exhaustion behavior, interaction with pause/resume, checkpoint
persistence of exhausted state, and summary generation after exhaustion.
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    BudgetStatus,
    build_incomplete_summary,
    get_tier,
    should_retry,
)


class TestExhaustionDetection(unittest.TestCase):
    """is_exhausted transitions correctly at the boundary."""

    def test_fresh_tracker_not_exhausted(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertFalse(tracker.is_exhausted)

    def test_tiny_budget_exhausts_quickly(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)

    def test_exhaustion_is_permanent(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        for _ in range(10):
            self.assertTrue(tracker.is_exhausted)
            time.sleep(0.001)

    def test_paused_before_exhaustion_delays_it(self):
        tier = TierConfig("tiny", 0.01, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.003)
        tracker.pause("save budget")
        time.sleep(0.05)
        self.assertFalse(tracker.is_exhausted)
        tracker.resume()
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)

    def test_various_tiny_budgets(self):
        for budget_ms in (5, 10, 20, 50):
            tier = TierConfig(f"t{budget_ms}", budget_ms / 1000, 60, 1, 0, 5)
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(max(budget_ms / 1000 * 3, 0.02))
            self.assertTrue(tracker.is_exhausted, f"budget={budget_ms}ms")


class TestPostExhaustionBehavior(unittest.TestCase):
    """Tracker remains functional after exhaustion."""

    def _make_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        return tier, tracker

    def test_snapshot_works_after_exhaustion(self):
        _, tracker = self._make_exhausted()
        snap = tracker.snapshot()
        self.assertIn("schema_version", snap)
        self.assertIn("active_seconds", snap)

    def test_remaining_zero_or_negative(self):
        _, tracker = self._make_exhausted()
        self.assertLessEqual(tracker.remaining_active, 0)

    def test_elapsed_exceeds_budget(self):
        tier, tracker = self._make_exhausted()
        self.assertGreaterEqual(tracker.elapsed_active, tier.active_budget_seconds)

    def test_pause_after_exhaustion_ok(self):
        _, tracker = self._make_exhausted()
        tracker.pause("post-exhaust")
        self.assertTrue(tracker.is_paused)
        self.assertTrue(tracker.is_exhausted)

    def test_resume_after_exhaustion_ok(self):
        _, tracker = self._make_exhausted()
        tracker.pause("p")
        tracker.resume()
        self.assertFalse(tracker.is_paused)
        self.assertTrue(tracker.is_exhausted)

    def test_checkpoint_after_exhaustion(self):
        _, tracker = self._make_exhausted()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            data = tracker.save_checkpoint(cp)
        self.assertGreater(data["active_seconds"], 0)

    def test_multiple_snapshots_after_exhaustion(self):
        _, tracker = self._make_exhausted()
        snaps = [tracker.snapshot() for _ in range(5)]
        for i in range(1, len(snaps)):
            self.assertGreaterEqual(
                snaps[i]["active_seconds"],
                snaps[i-1]["active_seconds"],
            )


class TestExhaustionCheckpointRestore(unittest.TestCase):
    """Exhausted state survives checkpoint/restore."""

    def test_restored_tracker_still_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        restored.resume()
        self.assertTrue(restored.is_exhausted)

    def test_restored_remaining_still_depleted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertLessEqual(restored.remaining_active, 0)

    def test_checkpoint_preserves_active_above_budget(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        self.assertGreaterEqual(data["active_seconds"], tier.active_budget_seconds)

    def test_near_boundary_checkpoint_restore(self):
        tier = TierConfig("near", 0.05, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertFalse(tracker.is_exhausted)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        restored.resume()
        time.sleep(0.04)
        self.assertTrue(restored.is_exhausted)


class TestExhaustionSummary(unittest.TestCase):
    """build_incomplete_summary after budget exhaustion."""

    def _make_exhausted_summary(self, pending=None, completed=None):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed or [],
            pending_tests=pending or [],
            issues=["budget exhausted"],
            evidence_paths=[], blockers=["ACTIVE_BUDGET_EXHAUSTED"],
        )

    def test_exhausted_status(self):
        summary = self._make_exhausted_summary()
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")

    def test_exhausted_severity_no_pending(self):
        summary = self._make_exhausted_summary()
        self.assertEqual(summary["severity"], "minor")

    def test_exhausted_severity_with_pending(self):
        summary = self._make_exhausted_summary(pending=["t1", "t2"])
        self.assertEqual(summary["severity"], "functional")

    def test_exhausted_severity_with_failures(self):
        completed = [{"test_id": "t1", "physical_pass": False,
                      "geometry_audit": {"passed": True}}]
        summary = self._make_exhausted_summary(completed=completed)
        self.assertEqual(summary["severity"], "functional")

    def test_exhausted_timing_active_exceeds_budget(self):
        summary = self._make_exhausted_summary()
        t = summary["timing"]
        self.assertGreaterEqual(t["active_seconds"], 0.005)

    def test_exhausted_not_retryable(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))
        self.assertFalse(should_retry(1, "ACTIVE_BUDGET_EXHAUSTED"))
        self.assertFalse(should_retry(2, "ACTIVE_BUDGET_EXHAUSTED"))


class TestExhaustionRetryInteraction(unittest.TestCase):
    """Exhaustion interacts correctly with retry logic."""

    def test_exhausted_status_is_terminal(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())

    def test_exhausted_blocks_all_retries(self):
        for attempt in range(10):
            self.assertFalse(should_retry(attempt, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_pass_also_blocks_retries(self):
        for attempt in range(10):
            self.assertFalse(should_retry(attempt, "PASS"))

    def test_infrastructure_failure_allows_retry(self):
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE"))
        self.assertTrue(should_retry(1, "INFRASTRUCTURE_FAILURE"))
        self.assertTrue(should_retry(2, "INFRASTRUCTURE_FAILURE"))
        self.assertFalse(should_retry(3, "INFRASTRUCTURE_FAILURE"))


class TestNearBoundaryBehavior(unittest.TestCase):
    """Behavior very close to the exhaustion boundary."""

    def test_just_under_budget_not_exhausted(self):
        tier = TierConfig("near", 0.1, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.05)
        self.assertFalse(tracker.is_exhausted)
        self.assertGreater(tracker.remaining_active, 0)

    def test_pause_preserves_remaining_at_boundary(self):
        tier = TierConfig("near", 0.05, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        tracker.pause("near boundary")
        remaining = tracker.remaining_active
        time.sleep(0.05)
        self.assertEqual(tracker.remaining_active, remaining)

    def test_resume_at_boundary_can_exhaust(self):
        tier = TierConfig("near", 0.04, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        tracker.pause("near")
        self.assertFalse(tracker.is_exhausted)
        tracker.resume()
        time.sleep(0.04)
        self.assertTrue(tracker.is_exhausted)

    def test_rapid_pause_resume_near_boundary(self):
        tier = TierConfig("near", 0.08, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        for _ in range(5):
            tracker.pause("p")
            tracker.resume()
        self.assertFalse(tracker.is_exhausted)
        time.sleep(0.08)
        self.assertTrue(tracker.is_exhausted)

    def test_snapshot_near_boundary_has_small_remaining(self):
        tier = TierConfig("near", 0.05, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.04)
        snap = tracker.snapshot()
        self.assertGreater(snap["remaining_active_seconds"], 0)
        self.assertLess(snap["remaining_active_seconds"], 0.02)


class TestAllTierExhaustion(unittest.TestCase):
    """Each production tier can exhaust with a custom tiny-budget tier."""

    def test_light_equivalent_exhaustion(self):
        tier = TierConfig("light-like", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertTrue(tracker.is_exhausted)

    def test_medium_equivalent_exhaustion(self):
        tier = TierConfig("medium-like", 0.01, 180, 2, 0, 15)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertTrue(tracker.is_exhausted)

    def test_high_equivalent_exhaustion(self):
        tier = TierConfig("high-like", 0.01, 300, 5, 0, 20)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertTrue(tracker.is_exhausted)


if __name__ == "__main__":
    unittest.main()
