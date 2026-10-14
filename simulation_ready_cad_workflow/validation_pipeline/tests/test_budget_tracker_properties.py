"""ActiveBudgetTracker property interaction tests.

Tests elapsed_active, elapsed_wall, remaining_active, is_exhausted,
and is_paused across various tracker states and ensures they are
mutually consistent.
"""
from __future__ import annotations

import time
import unittest

from ..slim_harness.budget import (
    ActiveBudgetTracker,
    TierConfig,
    get_tier,
)


class TestFreshTrackerProperties(unittest.TestCase):
    """Properties on a freshly created tracker."""

    def test_elapsed_active_near_zero(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertLess(tracker.elapsed_active, 1.0)

    def test_elapsed_wall_near_zero(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        self.assertLess(tracker.elapsed_wall, 1.0)

    def test_remaining_near_budget(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        self.assertAlmostEqual(
            tracker.remaining_active,
            tier.active_budget_seconds,
            delta=1.0,
        )

    def test_not_exhausted(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertFalse(tracker.is_exhausted)

    def test_not_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        self.assertFalse(tracker.is_paused)


class TestActiveWallRelationship(unittest.TestCase):
    """active <= wall always holds."""

    def test_without_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.02)
        snap = tracker.snapshot()
        self.assertLessEqual(
            snap["active_seconds"],
            snap["wall_seconds"] + 0.001,
        )

    def test_with_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("test")
        time.sleep(0.02)
        tracker.resume()
        snap = tracker.snapshot()
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])

    def test_multiple_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for _ in range(5):
            time.sleep(0.005)
            tracker.pause("p")
            time.sleep(0.005)
            tracker.resume()
        snap = tracker.snapshot()
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])


class TestRemainingBudgetMath(unittest.TestCase):
    """remaining = budget_limit - active."""

    def test_active_plus_remaining_equals_budget(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        snap = tracker.snapshot()
        total = snap["active_seconds"] + snap["remaining_active_seconds"]
        self.assertAlmostEqual(total, tier.active_budget_seconds, delta=0.01)

    def test_remaining_decreases_over_time(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        r1 = tracker.remaining_active
        time.sleep(0.02)
        r2 = tracker.remaining_active
        self.assertLess(r2, r1)

    def test_remaining_frozen_during_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        tracker.pause("freeze")
        r1 = tracker.remaining_active
        time.sleep(0.02)
        r2 = tracker.remaining_active
        self.assertAlmostEqual(r1, r2, places=2)


class TestExhaustionProperty(unittest.TestCase):
    """is_exhausted becomes True and stays True."""

    def test_exhaustion_triggered(self):
        tier = TierConfig("micro", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)

    def test_exhaustion_permanent(self):
        tier = TierConfig("micro", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)

    def test_pause_does_not_unexhaust(self):
        tier = TierConfig("micro", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        tracker.pause("too late")
        self.assertTrue(tracker.is_exhausted)

    def test_remaining_zero_when_exhausted(self):
        tier = TierConfig("micro", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertLessEqual(tracker.remaining_active, 0)


class TestPropertyConsistencyWithSnapshot(unittest.TestCase):
    """Properties match snapshot values."""

    def test_elapsed_active_matches_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertAlmostEqual(
            tracker.elapsed_active,
            snap["active_seconds"],
            delta=0.01,
        )

    def test_elapsed_wall_matches_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertAlmostEqual(
            tracker.elapsed_wall,
            snap["wall_seconds"],
            delta=0.01,
        )

    def test_remaining_matches_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertAlmostEqual(
            tracker.remaining_active,
            snap["remaining_active_seconds"],
            delta=0.01,
        )

    def test_paused_matches_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("check")
        snap = tracker.snapshot()
        self.assertEqual(tracker.is_paused, snap["paused"])


if __name__ == "__main__":
    unittest.main()
