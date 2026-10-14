"""ActiveBudgetTracker pause/resume deep testing.

Validates idempotent pause/resume, pause reason tracking, multiple
pause/resume cycles, interaction with exhaustion, and snapshot
consistency during paused states.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    ActiveBudgetTracker,
    BudgetStatus,
    TierConfig,
    get_tier,
)


class TestPauseIdempotent(unittest.TestCase):
    """Pausing when already paused is a no-op."""

    def test_double_pause_no_error(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("first")
        tracker.pause("second")
        self.assertTrue(tracker.is_paused)

    def test_double_pause_count_stays_one(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("a")
        snap1 = tracker.snapshot()
        tracker.pause("b")
        snap2 = tracker.snapshot()
        self.assertEqual(snap1["pause_count"], snap2["pause_count"])

    def test_triple_pause_same_count(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("x")
        tracker.pause("y")
        tracker.pause("z")
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 1)


class TestResumeIdempotent(unittest.TestCase):
    """Resuming when not paused is a no-op."""

    def test_resume_without_pause_no_error(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.resume()
        self.assertFalse(tracker.is_paused)

    def test_double_resume_no_error(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("p")
        tracker.resume()
        tracker.resume()
        self.assertFalse(tracker.is_paused)

    def test_resume_before_any_pause_no_count(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 0)


class TestPauseResumeCycles(unittest.TestCase):
    """Multiple pause/resume cycles accumulate correctly."""

    def test_pause_count_increments_per_cycle(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for i in range(5):
            tracker.pause(f"cycle_{i}")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 5)

    def test_total_pause_seconds_accumulates(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        for _ in range(3):
            tracker.pause("accum")
            time.sleep(0.01)
            tracker.resume()
        snap = tracker.snapshot()
        self.assertGreater(snap["total_pause_seconds"], 0)

    def test_active_time_excludes_pauses(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("p")
        time.sleep(0.03)
        tracker.resume()
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])

    def test_wall_time_includes_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("w")
        time.sleep(0.02)
        tracker.resume()
        snap = tracker.snapshot()
        self.assertGreater(snap["wall_seconds"], 0)

    def test_many_rapid_cycles(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        for i in range(50):
            tracker.pause(f"rapid_{i}")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 50)


class TestPausedSnapshotConsistency(unittest.TestCase):
    """Snapshots taken while paused are consistent."""

    def test_paused_flag_true_in_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("snap check")
        snap = tracker.snapshot()
        self.assertTrue(snap["paused"])

    def test_active_time_frozen_during_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("freeze")
        snap1 = tracker.snapshot()
        time.sleep(0.02)
        snap2 = tracker.snapshot()
        self.assertAlmostEqual(
            snap1["active_seconds"],
            snap2["active_seconds"],
            places=2,
        )

    def test_wall_time_advances_during_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("wall advance")
        snap1 = tracker.snapshot()
        time.sleep(0.02)
        snap2 = tracker.snapshot()
        self.assertGreater(snap2["wall_seconds"], snap1["wall_seconds"])

    def test_remaining_frozen_during_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        tracker.pause("remaining")
        snap1 = tracker.snapshot()
        time.sleep(0.02)
        snap2 = tracker.snapshot()
        self.assertAlmostEqual(
            snap1["remaining_active_seconds"],
            snap2["remaining_active_seconds"],
            places=2,
        )


class TestPauseResumeWithExhaustion(unittest.TestCase):
    """Pause/resume behavior near budget exhaustion."""

    def test_pause_does_not_prevent_exhaustion_check(self):
        tier = TierConfig("tiny", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)

    def test_paused_tracker_not_exhausted_if_active_below_budget(self):
        tier = TierConfig("quick", 0.5, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("hold")
        time.sleep(0.5)
        self.assertFalse(tracker.is_exhausted)

    def test_resume_after_long_pause_not_exhausted(self):
        tier = TierConfig("short", 0.1, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("long hold")
        time.sleep(0.2)
        tracker.resume()
        self.assertFalse(tracker.is_exhausted)

    def test_active_time_accumulation_after_resume(self):
        tier = TierConfig("budget", 0.05, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("early")
        time.sleep(0.1)
        tracker.resume()
        time.sleep(0.06)
        self.assertTrue(tracker.is_exhausted)


class TestPauseReasonIgnoredForLogic(unittest.TestCase):
    """The pause reason is metadata only; it doesn't affect behavior."""

    def test_different_reasons_same_effect(self):
        t1 = ActiveBudgetTracker(tier=get_tier("light"))
        t2 = ActiveBudgetTracker(tier=get_tier("light"))
        t1.pause("reason A")
        t2.pause("reason B")
        self.assertEqual(t1.is_paused, t2.is_paused)

    def test_empty_reason_accepted(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("")
        self.assertTrue(tracker.is_paused)

    def test_long_reason_accepted(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("x" * 10000)
        self.assertTrue(tracker.is_paused)


class TestRestoreStartsPaused(unittest.TestCase):
    """ActiveBudgetTracker.restore() starts in paused state."""

    def test_restored_tracker_is_paused(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            tracker.save_checkpoint(p)
            restored = ActiveBudgetTracker.restore(p, tier)
            self.assertTrue(restored.is_paused)

    def test_restored_then_resumed(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            snap = tracker.save_checkpoint(p)
            restored = ActiveBudgetTracker.restore(p, tier)
            restored.resume()
            self.assertFalse(restored.is_paused)
            time.sleep(0.01)
            snap2 = restored.snapshot()
            self.assertGreater(snap2["active_seconds"], snap["active_seconds"])

    def test_restored_active_time_preserved(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            snap = tracker.save_checkpoint(p)
            restored = ActiveBudgetTracker.restore(p, tier)
            rsnap = restored.snapshot()
            self.assertAlmostEqual(
                snap["active_seconds"],
                rsnap["active_seconds"],
                places=2,
            )


if __name__ == "__main__":
    unittest.main()
