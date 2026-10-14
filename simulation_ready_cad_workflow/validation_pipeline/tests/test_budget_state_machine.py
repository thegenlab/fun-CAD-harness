"""Budget state machine and transition tests.

Tests that the budget tracker follows correct state machine semantics:
- Active/Paused/Exhausted state transitions
- Multi-tracker independence
- Checkpoint state preservation through all transitions
- Summary generation at every state
- Tier enforcement at state boundaries
- Status enum exhaustive coverage
- Integration between tracker state and enforcement functions
"""
from __future__ import annotations

import json
import tempfile
import time
import threading
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    SimulationTimeoutResult,
    get_tier,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    build_incomplete_summary,
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
)


class TestTrackerStateActive(unittest.TestCase):
    """Tracker in the active (running) state."""

    def test_starts_active(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        self.assertFalse(tracker.is_paused)
        self.assertFalse(tracker.is_exhausted)

    def test_active_time_increases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        self.assertGreater(tracker.elapsed_active, 0)

    def test_remaining_decreases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        initial = tracker.remaining_active
        time.sleep(0.01)
        self.assertLess(tracker.remaining_active, initial)

    def test_snapshot_not_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertFalse(snap["paused"])
        self.assertEqual(snap["pause_count"], 0)

    def test_wall_time_increases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertGreater(snap["wall_seconds"], 0)


class TestTrackerStatePaused(unittest.TestCase):
    """Tracker in the paused state."""

    def test_pause_stops_active_counting(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("infra_wait")
        active_at_pause = tracker.elapsed_active
        time.sleep(0.02)
        self.assertAlmostEqual(tracker.elapsed_active, active_at_pause, delta=0.005)

    def test_pause_counts_pause_time(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("test")
        time.sleep(0.02)
        snap = tracker.snapshot()
        self.assertGreater(snap["total_pause_seconds"], 0.01)

    def test_resume_restarts_active(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("p")
        time.sleep(0.01)
        active_paused = tracker.elapsed_active
        tracker.resume()
        time.sleep(0.01)
        self.assertGreater(tracker.elapsed_active, active_paused)

    def test_double_pause_is_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("first")
        tracker.pause("second")
        self.assertEqual(tracker.snapshot()["pause_count"], 1)

    def test_double_resume_is_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("p")
        tracker.resume()
        tracker.resume()
        self.assertEqual(tracker.snapshot()["pause_count"], 1)


class TestTrackerStateExhausted(unittest.TestCase):
    """Tracker in the exhausted state."""

    def test_exhausted_after_budget(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)

    def test_remaining_zero_when_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_still_trackable_after_exhaustion(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertGreater(snap["active_seconds"], 0)
        self.assertEqual(snap["remaining_active_seconds"], 0.0)

    def test_can_still_pause_after_exhaustion(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("post_exhaust")
        self.assertTrue(tracker.is_paused)
        self.assertTrue(tracker.is_exhausted)

    def test_can_checkpoint_after_exhaustion(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "exhausted.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            self.assertEqual(data["remaining_active_seconds"], 0.0)


class TestMultiTrackerIndependence(unittest.TestCase):
    """Multiple trackers don't interfere with each other."""

    def test_different_tiers_independent(self):
        t1 = ActiveBudgetTracker(tier=get_tier("light"))
        t2 = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        t1.pause("p1")
        self.assertTrue(t1.is_paused)
        self.assertFalse(t2.is_paused)

    def test_pause_one_doesnt_affect_other(self):
        t1 = ActiveBudgetTracker(tier=get_tier("medium"))
        t2 = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        t1.pause("p")
        a1 = t1.elapsed_active
        time.sleep(0.01)
        self.assertAlmostEqual(t1.elapsed_active, a1, delta=0.005)
        self.assertGreater(t2.elapsed_active, a1)

    def test_exhaustion_independent(self):
        tiny = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        big = get_tier("high")
        t1 = ActiveBudgetTracker(tier=tiny)
        t2 = ActiveBudgetTracker(tier=big)
        time.sleep(0.01)
        self.assertTrue(t1.is_exhausted)
        self.assertFalse(t2.is_exhausted)

    def test_checkpoint_independent(self):
        t1 = ActiveBudgetTracker(tier=get_tier("light"))
        t2 = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "cp1.json"
            cp2 = Path(tmp) / "cp2.json"
            t1.save_checkpoint(cp1)
            t2.save_checkpoint(cp2)
            d1 = json.loads(cp1.read_text(encoding="utf-8"))
            d2 = json.loads(cp2.read_text(encoding="utf-8"))
            self.assertEqual(d1["tier"], "light")
            self.assertEqual(d2["tier"], "high")


class TestCheckpointStatePreservation(unittest.TestCase):
    """Checkpoint round-trips preserve all state dimensions."""

    def test_active_preserved(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        active = tracker.elapsed_active
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
            self.assertAlmostEqual(restored.elapsed_active, active, delta=0.01)

    def test_pause_count_preserved(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        for i in range(5):
            tracker.pause(f"p{i}")
            tracker.resume()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            self.assertEqual(restored.snapshot()["pause_count"], 6)

    def test_paused_state_preserved(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("save_paused")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            self.assertTrue(restored.is_paused)

    def test_tier_from_file_matches(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            self.assertEqual(restored.tier.name, "medium")


class TestSummaryAtEveryState(unittest.TestCase):
    """build_incomplete_summary works regardless of tracker state."""

    def _summary(self, tracker, tier_name):
        return build_incomplete_summary(
            tier=get_tier(tier_name), tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )

    def test_summary_from_fresh_tracker(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        summary = self._summary(tracker, "light")
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_summary_from_paused_tracker(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("p")
        summary = self._summary(tracker, "medium")
        self.assertGreaterEqual(summary["timing"]["pause_count"], 1)

    def test_summary_from_exhausted_tracker(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertGreater(summary["timing"]["active_seconds"], tier.active_budget_seconds)

    def test_summary_from_restored_tracker(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
        summary = self._summary(restored, "high")
        self.assertIn("timing", summary)


class TestTierEnforcementAtBoundaries(unittest.TestCase):
    """Tier enforcement interacts correctly with tracker state."""

    def test_parts_within_limit_all_tiers(self):
        for name, tier in TIERS.items():
            bodies = [{}] * tier.max_parts
            self.assertEqual(check_part_count(bodies, tier), [], f"Tier {name}")

    def test_parts_over_limit_all_tiers(self):
        for name, tier in TIERS.items():
            bodies = [{}] * (tier.max_parts + 1)
            violations = check_part_count(bodies, tier)
            self.assertGreater(len(violations), 0, f"Tier {name}")

    def test_tasks_within_limit_all_tiers(self):
        for name, tier in TIERS.items():
            tasks = [{}] * tier.max_tasks_per_design
            self.assertEqual(check_task_count(tasks, tier), [], f"Tier {name}")

    def test_tasks_over_limit_all_tiers(self):
        for name, tier in TIERS.items():
            tasks = [{}] * (tier.max_tasks_per_design + 1)
            violations = check_task_count(tasks, tier)
            self.assertGreater(len(violations), 0, f"Tier {name}")

    def test_timeout_clamping_all_tiers(self):
        for name, tier in TIERS.items():
            clamped = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds * 2)
            self.assertEqual(clamped, tier.simulation_timeout_seconds, f"Tier {name}")

    def test_timeout_passthrough_all_tiers(self):
        for name, tier in TIERS.items():
            small = tier.simulation_timeout_seconds / 2
            result = enforce_simulation_timeout(tier, small)
            self.assertEqual(result, small, f"Tier {name}")


class TestStatusEnumCoverage(unittest.TestCase):
    """Exhaustive coverage of BudgetStatus enum values."""

    def test_all_values_unique(self):
        values = [s.value for s in BudgetStatus]
        self.assertEqual(len(values), len(set(values)))

    def test_is_terminal_defined_for_all(self):
        for status in BudgetStatus:
            result = status.is_terminal()
            self.assertIsInstance(result, bool)

    def test_pass_in_non_retryable(self):
        self.assertFalse(should_retry(0, "PASS"))

    def test_exhausted_in_non_retryable(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_terminated_in_non_retryable(self):
        self.assertFalse(should_retry(0, "TERMINATED"))

    def test_infrastructure_failure_retryable(self):
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE"))

    def test_simulation_timeout_retryable(self):
        self.assertTrue(should_retry(0, "SIMULATION_TIMEOUT"))


class TestTrackerLifecycleSequences(unittest.TestCase):
    """Complete lifecycle sequences mimicking real workflows."""

    def test_basic_lifecycle(self):
        """Active → Pause → Resume → Checkpoint → Restore → Resume"""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("infra_wait")
        time.sleep(0.01)
        tracker.resume()
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
            time.sleep(0.01)

        snap = restored.snapshot()
        self.assertGreater(snap["active_seconds"], 0)
        self.assertGreater(snap["pause_count"], 0)
        self.assertGreater(snap["total_pause_seconds"], 0)

    def test_error_recovery_lifecycle(self):
        """Active → Error → Pause → Checkpoint → Restore → Resume"""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        try:
            raise RuntimeError("simulated error")
        except RuntimeError:
            tracker.pause("error_recovery")

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "recovery.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()

        self.assertFalse(restored.is_exhausted)
        self.assertGreater(restored.remaining_active, 0)

    def test_multi_task_lifecycle(self):
        """Multiple tasks with pauses between them."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)

        for task_num in range(3):
            time.sleep(0.005)
            tracker.pause(f"between_task_{task_num}")
            time.sleep(0.005)
            tracker.resume()

        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 3)
        self.assertGreater(snap["active_seconds"], 0)
        self.assertGreater(snap["total_pause_seconds"], 0)

    def test_tier_escalation_lifecycle(self):
        """Start light, checkpoint, restore to high."""
        light = get_tier("light")
        high = get_tier("high")
        tracker = ActiveBudgetTracker(tier=light)
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "escalate.json"
            tracker.save_checkpoint(cp)
            active_before = tracker.elapsed_active
            restored = ActiveBudgetTracker.restore(cp, tier=high)
            restored.resume()

        self.assertEqual(restored.tier.name, "high")
        self.assertAlmostEqual(restored.elapsed_active, active_before, delta=0.01)
        self.assertAlmostEqual(
            restored.remaining_active,
            high.active_budget_seconds - active_before,
            delta=0.05,
        )


class TestConcurrentStateTransitions(unittest.TestCase):
    """Concurrent operations on tracker state."""

    def test_parallel_snapshot_reads(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        errors = []
        snapshots = []

        def reader():
            try:
                for _ in range(50):
                    snap = tracker.snapshot()
                    snapshots.append(snap)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(snapshots), 250)
        for snap in snapshots:
            self.assertEqual(snap["tier"], "medium")

    def test_parallel_pause_resume(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        errors = []

        def toggler(tid):
            try:
                for _ in range(20):
                    tracker.pause(f"t{tid}")
                    tracker.resume()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggler, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        snap = tracker.snapshot()
        self.assertGreater(snap["pause_count"], 0)


if __name__ == "__main__":
    unittest.main()
