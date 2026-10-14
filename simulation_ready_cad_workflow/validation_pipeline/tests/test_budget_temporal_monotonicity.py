"""Budget temporal monotonicity tests.

Validates that time-related properties obey strict monotonicity:
active_seconds and wall_seconds never decrease, remaining_active
never increases, and pause/resume transitions preserve ordering.
"""
from __future__ import annotations

import time
import unittest

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    get_tier,
)


class TestActiveSecondsMonotonicity(unittest.TestCase):
    """elapsed_active never decreases between observations."""

    def test_running_tracker_active_increases(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            readings = []
            for _ in range(10):
                readings.append(tracker.elapsed_active)
                time.sleep(0.002)
            for i in range(1, len(readings)):
                self.assertGreaterEqual(readings[i], readings[i-1],
                    f"Tier {name}: active decreased at step {i}")

    def test_paused_tracker_active_stable(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("freeze")
        a1 = tracker.elapsed_active
        time.sleep(0.02)
        a2 = tracker.elapsed_active
        self.assertEqual(a1, a2)

    def test_resume_continues_from_paused_value(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.005)
        tracker.pause("p")
        paused_val = tracker.elapsed_active
        time.sleep(0.01)
        tracker.resume()
        time.sleep(0.005)
        resumed_val = tracker.elapsed_active
        self.assertGreater(resumed_val, paused_val)

    def test_multiple_pause_resume_monotonic(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        prev = 0.0
        for _ in range(8):
            time.sleep(0.003)
            curr = tracker.elapsed_active
            self.assertGreaterEqual(curr, prev)
            prev = curr
            tracker.pause("p")
            paused = tracker.elapsed_active
            self.assertGreaterEqual(paused, prev)
            time.sleep(0.003)
            still_paused = tracker.elapsed_active
            self.assertEqual(still_paused, paused)
            tracker.resume()
            prev = tracker.elapsed_active
            self.assertGreaterEqual(prev, paused)


class TestWallSecondsMonotonicity(unittest.TestCase):
    """elapsed_wall never decreases, even across pause/resume."""

    def test_wall_always_increases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        readings = []
        for _ in range(10):
            readings.append(tracker.elapsed_wall)
            time.sleep(0.002)
        for i in range(1, len(readings)):
            self.assertGreaterEqual(readings[i], readings[i-1])

    def test_wall_increases_while_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("p")
        w1 = tracker.elapsed_wall
        time.sleep(0.02)
        w2 = tracker.elapsed_wall
        self.assertGreater(w2, w1)

    def test_wall_exceeds_active_when_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.005)
        tracker.pause("p")
        time.sleep(0.02)
        self.assertGreater(tracker.elapsed_wall, tracker.elapsed_active)


class TestRemainingActiveMonotonicity(unittest.TestCase):
    """remaining_active never increases while running."""

    def test_running_remaining_decreases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        readings = []
        for _ in range(10):
            readings.append(tracker.remaining_active)
            time.sleep(0.002)
        for i in range(1, len(readings)):
            self.assertLessEqual(readings[i], readings[i-1],
                f"remaining increased at step {i}")

    def test_paused_remaining_stable(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("p")
        r1 = tracker.remaining_active
        time.sleep(0.02)
        r2 = tracker.remaining_active
        self.assertEqual(r1, r2)

    def test_remaining_plus_elapsed_equals_budget(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            total = tracker.elapsed_active + tracker.remaining_active
            self.assertAlmostEqual(total, tier.active_budget_seconds, delta=0.1)

    def test_remaining_reaches_zero_on_exhaustion(self):
        tier = TierConfig("tiny", 0.01, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertTrue(tracker.is_exhausted)
        self.assertLessEqual(tracker.remaining_active, 0)

    def test_remaining_never_negative_for_fresh(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertGreater(tracker.remaining_active, 0)


class TestSnapshotTemporalConsistency(unittest.TestCase):
    """Snapshots taken at successive times maintain temporal ordering."""

    def test_successive_snapshots_active_nondecreasing(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snaps = []
        for _ in range(5):
            snaps.append(tracker.snapshot())
            time.sleep(0.005)
        for i in range(1, len(snaps)):
            self.assertGreaterEqual(
                snaps[i]["active_seconds"],
                snaps[i-1]["active_seconds"],
            )

    def test_successive_snapshots_wall_nondecreasing(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        snaps = []
        for _ in range(5):
            snaps.append(tracker.snapshot())
            time.sleep(0.005)
        for i in range(1, len(snaps)):
            self.assertGreaterEqual(
                snaps[i]["wall_seconds"],
                snaps[i-1]["wall_seconds"],
            )

    def test_successive_snapshots_remaining_nonincreasing(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snaps = []
        for _ in range(5):
            snaps.append(tracker.snapshot())
            time.sleep(0.005)
        for i in range(1, len(snaps)):
            self.assertLessEqual(
                snaps[i]["remaining_active_seconds"],
                snaps[i-1]["remaining_active_seconds"],
            )

    def test_snapshot_checkpoint_at_monotonic(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        stamps = []
        for _ in range(5):
            stamps.append(tracker.snapshot()["checkpoint_at"])
            time.sleep(0.005)
        for i in range(1, len(stamps)):
            self.assertGreaterEqual(stamps[i], stamps[i-1])


class TestPauseResumeOrdering(unittest.TestCase):
    """Pause/resume transitions maintain temporal invariants."""

    def test_pause_freezes_active_not_wall(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.005)
        tracker.pause("p")
        a_paused = tracker.elapsed_active
        w_paused = tracker.elapsed_wall
        time.sleep(0.02)
        self.assertEqual(tracker.elapsed_active, a_paused)
        self.assertGreater(tracker.elapsed_wall, w_paused)

    def test_pause_count_monotonic(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        counts = []
        for i in range(5):
            tracker.pause(f"p{i}")
            counts.append(tracker.snapshot()["pause_count"])
            tracker.resume()
        for i in range(1, len(counts)):
            self.assertGreater(counts[i], counts[i-1])

    def test_total_pause_seconds_monotonic(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        pause_secs = []
        for _ in range(4):
            tracker.pause("p")
            time.sleep(0.005)
            tracker.resume()
            pause_secs.append(tracker.snapshot()["total_pause_seconds"])
        for i in range(1, len(pause_secs)):
            self.assertGreaterEqual(pause_secs[i], pause_secs[i-1])

    def test_idempotent_pause_no_double_count(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("first")
        c1 = tracker.snapshot()["pause_count"]
        tracker.pause("second")
        c2 = tracker.snapshot()["pause_count"]
        self.assertEqual(c1, c2)

    def test_idempotent_resume_no_effect(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        a1 = tracker.elapsed_active
        tracker.resume()
        a2 = tracker.elapsed_active
        self.assertGreaterEqual(a2, a1)


class TestExhaustionTiming(unittest.TestCase):
    """Budget exhaustion timing is consistent across observations."""

    def test_exhaustion_detected_reliably(self):
        tier = TierConfig("micro", 0.008, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.025)
        self.assertTrue(tracker.is_exhausted)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)

    def test_exhaustion_not_reversed_by_pause(self):
        tier = TierConfig("micro", 0.008, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.025)
        self.assertTrue(tracker.is_exhausted)
        tracker.pause("p")
        self.assertTrue(tracker.is_exhausted)

    def test_near_boundary_transitions(self):
        tier = TierConfig("near", 0.05, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        self.assertFalse(tracker.is_exhausted)
        time.sleep(0.08)
        self.assertTrue(tracker.is_exhausted)

    def test_elapsed_exceeds_budget_when_exhausted(self):
        tier = TierConfig("short", 0.01, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertTrue(tracker.is_exhausted)
        self.assertGreaterEqual(tracker.elapsed_active, tier.active_budget_seconds)


class TestWallActiveRelationship(unittest.TestCase):
    """Wall time >= active time, gap = total pause duration."""

    def test_wall_ge_active_always(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            snap = tracker.snapshot()
            self.assertGreaterEqual(snap["wall_seconds"], snap["active_seconds"])

    def test_no_pause_wall_approx_active(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        diff = tracker.elapsed_wall - tracker.elapsed_active
        self.assertAlmostEqual(diff, 0.0, delta=0.05)

    def test_pause_widens_gap(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.005)
        gap_before = tracker.elapsed_wall - tracker.elapsed_active
        tracker.pause("p")
        time.sleep(0.02)
        tracker.resume()
        gap_after = tracker.elapsed_wall - tracker.elapsed_active
        self.assertGreater(gap_after, gap_before)

    def test_gap_approximates_pause_duration(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("p")
        time.sleep(0.03)
        tracker.resume()
        gap = tracker.elapsed_wall - tracker.elapsed_active
        self.assertAlmostEqual(gap, 0.03, delta=0.02)

    def test_multiple_pauses_accumulate_gap(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        for _ in range(3):
            tracker.pause("p")
            time.sleep(0.01)
            tracker.resume()
        gap = tracker.elapsed_wall - tracker.elapsed_active
        self.assertAlmostEqual(gap, 0.03, delta=0.02)


if __name__ == "__main__":
    unittest.main()
