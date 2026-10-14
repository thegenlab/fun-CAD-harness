"""Stress tests for budget module under sustained load conditions.

Tests rapid workflow cycling, large part counts near tier boundaries,
concurrent tracker operations, and checkpoint reliability under load.
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.budget import (
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


class TestConcurrentTrackerAccess(unittest.TestCase):
    """Thread-safety stress tests for ActiveBudgetTracker."""

    def test_concurrent_pause_resume_50_threads(self):
        """50 threads doing rapid pause/resume must not corrupt state."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        errors = []

        def worker(tid):
            try:
                for _ in range(20):
                    tracker.pause(f"thread-{tid}")
                    time.sleep(0.001)
                    tracker.resume()
            except Exception as e:
                errors.append((tid, e))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["pause_count"], 0)
        self.assertGreaterEqual(snap["active_seconds"], 0)
        self.assertLessEqual(snap["active_seconds"], snap["wall_seconds"])

    def test_concurrent_snapshot_during_pause(self):
        """Snapshots taken during active pause/resume cycles are consistent."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        snapshots = []

        def pauser():
            for _ in range(100):
                tracker.pause("cycle")
                time.sleep(0.0005)
                tracker.resume()

        def snapper():
            for _ in range(100):
                snap = tracker.snapshot()
                snapshots.append(snap)
                time.sleep(0.0005)

        t1 = threading.Thread(target=pauser)
        t2 = threading.Thread(target=snapper)
        t1.start()
        t2.start()
        t1.join(timeout=30)
        t2.join(timeout=30)

        for snap in snapshots:
            self.assertIn("active_seconds", snap)
            self.assertIn("wall_seconds", snap)
            self.assertGreaterEqual(snap["active_seconds"], 0)
            self.assertLessEqual(snap["active_seconds"], snap["wall_seconds"] + 0.1)


class TestCheckpointReliability(unittest.TestCase):
    """Checkpoint save/restore reliability under various conditions."""

    def test_rapid_save_restore_cycle(self):
        """50 save/restore cycles preserve data integrity."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            for i in range(50):
                tracker.pause(f"cycle-{i}")
                time.sleep(0.001)
                tracker.resume()
                tracker.save_checkpoint(path)
                restored = ActiveBudgetTracker.restore(path, tier=tier)
                self.assertEqual(restored.snapshot()["tier"], "light")
                self.assertAlmostEqual(
                    tracker.elapsed_active,
                    restored.elapsed_active,
                    places=1,
                )

    def test_checkpoint_with_large_pause_count(self):
        """Checkpoint handles tracker with 1000 pauses."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        for i in range(1000):
            tracker.pause(f"p{i}")
            tracker.resume()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.json"
            tracker.save_checkpoint(path)
            restored = ActiveBudgetTracker.restore(path, tier=tier)
            # restore() starts paused, adding +1 to the count
            self.assertEqual(restored.snapshot()["pause_count"], 1001)

    def test_checkpoint_size_stays_bounded(self):
        """Checkpoint file size doesn't grow unboundedly with pause count."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        for i in range(500):
            tracker.pause(f"reason_{i}")
            tracker.resume()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sized.json"
            tracker.save_checkpoint(path)
            size = path.stat().st_size
            self.assertLess(size, 4096, "Checkpoint should stay compact")


class TestAdjustmentLogStress(unittest.TestCase):
    """Stress tests for criteria adjustment logging."""

    def test_1000_adjustments_round_trip(self):
        """Save and load 1000 adjustments."""
        log = AdjustmentLog()
        for i in range(1000):
            log.record(CriteriaAdjustment(
                f"criterion_{i}", {"threshold": i}, {"threshold": i + 1},
                f"Adjusted from {i} to {i+1}",
            ))
        self.assertEqual(len(log.entries), 1000)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big_log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
            self.assertEqual(len(loaded.entries), 1000)
            self.assertEqual(loaded.entries[0].criterion_id, "criterion_0")
            self.assertEqual(loaded.entries[999].criterion_id, "criterion_999")

    def test_frozen_criteria_never_adjustable(self):
        """Frozen criteria always raise ValueError on adjustment."""
        for i in range(100):
            with self.assertRaises(ValueError):
                CriteriaAdjustment(
                    f"criterion_{i}", {"v": 1}, {"v": 2}, "test",
                    frozen=True,
                )


class TestTierLimitBoundaryStress(unittest.TestCase):
    """Stress boundary limits across all tiers."""

    def test_part_counts_at_every_tier_boundary(self):
        """Check part count at exact limit, -1, and +1 for all tiers."""
        for name, tier in TIERS.items():
            exact = [{} for _ in range(tier.max_parts)]
            under = [{} for _ in range(tier.max_parts - 1)] if tier.max_parts > 0 else []
            over = [{} for _ in range(tier.max_parts + 1)]

            self.assertEqual(check_part_count(exact, tier), [],
                             f"{name}: exact limit should pass")
            self.assertEqual(check_part_count(under, tier), [],
                             f"{name}: under limit should pass")
            self.assertEqual(len(check_part_count(over, tier)), 1,
                             f"{name}: over limit should fail")

    def test_task_counts_at_every_tier_boundary(self):
        """Check task count at exact limit, -1, and +1 for all tiers."""
        for name, tier in TIERS.items():
            exact = [{} for _ in range(tier.max_tasks_per_design)]
            under = [{} for _ in range(tier.max_tasks_per_design - 1)] if tier.max_tasks_per_design > 0 else []
            over = [{} for _ in range(tier.max_tasks_per_design + 1)]

            self.assertEqual(check_task_count(exact, tier), [],
                             f"{name}: exact limit should pass")
            self.assertEqual(check_task_count(under, tier), [],
                             f"{name}: under limit should pass")
            self.assertEqual(len(check_task_count(over, tier)), 1,
                             f"{name}: over limit should fail")

    def test_large_body_count_performance(self):
        """check_part_count with 10000 bodies returns quickly."""
        tier = get_tier("light")
        bodies = [{} for _ in range(10000)]
        start = time.monotonic()
        violations = check_part_count(bodies, tier)
        elapsed = time.monotonic() - start
        self.assertEqual(len(violations), 1)
        self.assertLess(elapsed, 0.1, "Part count check should be O(1)")


class TestBudgetExhaustionRecoveryFlow(unittest.TestCase):
    """Full budget exhaustion -> checkpoint -> restore -> summary flow."""

    def test_exhaustion_checkpoint_restore_summary(self):
        """Complete flow: exhaust budget, save checkpoint, restore, build summary."""
        tier = TierConfig("tiny", 0.05, 5, 2, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)

        time.sleep(0.06)
        self.assertTrue(tracker.is_exhausted)

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            tracker.save_checkpoint(checkpoint)

            restored = ActiveBudgetTracker.restore(checkpoint, tier=tier)
            self.assertTrue(restored.is_exhausted)

            summary = build_incomplete_summary(
                tier=tier, tracker=restored,
                completed_tests=[{"test_id": "done", "physical_pass": True,
                                  "geometry_audit": {"passed": True}}],
                pending_tests=["remaining_1", "remaining_2"],
                issues=["Budget exhausted"],
                evidence_paths=["/evidence/trace.json"],
                blockers=[],
            )
            self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
            self.assertEqual(len(summary["completed_tests"]), 1)
            self.assertEqual(len(summary["pending_tests"]), 2)
            self.assertIn("timing", summary)

    def test_recovery_delay_never_exceeds_bound(self):
        """Recovery delay stays bounded even at high attempt counts."""
        for attempt in range(100):
            delay = recovery_delay(attempt)
            self.assertGreater(delay, 0)
            self.assertLess(delay, 3600)

    def test_should_retry_respects_max_retries(self):
        """should_retry returns False at and beyond MAX_RECOVERY_RETRIES."""
        for i in range(MAX_RECOVERY_RETRIES + 5):
            if i < MAX_RECOVERY_RETRIES:
                self.assertTrue(should_retry(i, "some_error"))
            else:
                self.assertFalse(should_retry(i, "some_error"))


class TestIncompleteSummaryVariations(unittest.TestCase):
    """Various incomplete summary generation scenarios."""

    def test_empty_state(self):
        """Summary with no completed tests and no pending tests."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(summary["severity"], "minor")

    def test_all_tests_completed(self):
        """Summary with all tests completed (partial success scenario)."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[
                {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
                {"test_id": "t2", "physical_pass": False, "geometry_audit": {"passed": True}},
            ],
            pending_tests=[],
            issues=["One test failed physics"],
            evidence_paths=["/a.json", "/b.json"],
            blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")
        self.assertEqual(len(summary["evidence_paths"]), 2)

    def test_summary_with_blockers(self):
        """Summary includes blockers when present."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[],
            pending_tests=["t1"],
            issues=[],
            evidence_paths=[],
            blockers=["Chrono unavailable", "Memory limit reached"],
        )
        self.assertEqual(len(summary["blockers"]), 2)

    def test_summary_json_round_trip(self):
        """Summary serializes to JSON and back identically."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "x", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=["y"],
            issues=["timeout"],
            evidence_paths=["/path"],
            blockers=["memory"],
        )
        encoded = json.dumps(summary)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema_version"], summary["schema_version"])
        self.assertEqual(decoded["status"], summary["status"])
        self.assertEqual(decoded["completed_tests"], summary["completed_tests"])


class TestCustomTierCreation(unittest.TestCase):
    """Tests for creating custom TierConfig instances."""

    def test_custom_tier_is_frozen(self):
        """Custom TierConfig instances are frozen dataclasses."""
        custom = TierConfig("custom", 600, 30, 3, 1, 8)
        with self.assertRaises(AttributeError):
            custom.active_budget_seconds = 9999

    def test_custom_tier_enforce_timeout(self):
        """enforce_simulation_timeout works with custom tiers."""
        custom = TierConfig("custom", 600, 45, 3, 1, 8)
        self.assertEqual(enforce_simulation_timeout(custom, 30), 30)
        self.assertEqual(enforce_simulation_timeout(custom, 60), 45)
        self.assertEqual(enforce_simulation_timeout(custom), 45)

    def test_custom_tier_budget_tracker(self):
        """ActiveBudgetTracker works with custom tiers."""
        custom = TierConfig("custom", 1.0, 10, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=custom)
        self.assertFalse(tracker.is_exhausted)
        time.sleep(1.1)
        self.assertTrue(tracker.is_exhausted)

    def test_custom_tier_check_limits(self):
        """check_part_count and check_task_count work with custom tiers."""
        custom = TierConfig("custom", 600, 30, 3, 1, 8)
        self.assertEqual(check_part_count([{} for _ in range(8)], custom), [])
        self.assertEqual(len(check_part_count([{} for _ in range(9)], custom)), 1)
        self.assertEqual(check_task_count([{} for _ in range(3)], custom), [])
        self.assertEqual(len(check_task_count([{} for _ in range(4)], custom)), 1)


if __name__ == "__main__":
    unittest.main()
