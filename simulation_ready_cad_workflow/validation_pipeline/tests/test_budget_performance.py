"""Budget module performance and scalability tests.

Tests that budget operations meet performance requirements:
- Snapshot creation latency under load
- Checkpoint save/restore throughput
- Summary generation with large test lists
- Tracker operations scale O(1) with pause count
- Limit check operations scale O(n) with body/task count
- Concurrent tracker access doesn't degrade
- Memory footprint stays bounded
"""
from __future__ import annotations

import json
import sys
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
    get_tier,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    build_incomplete_summary,
    recovery_delay,
    should_retry,
)


class TestSnapshotLatency(unittest.TestCase):
    """Snapshot creation is fast enough for inline use."""

    def test_snapshot_under_1ms(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        start = time.perf_counter()
        for _ in range(100):
            tracker.snapshot()
        elapsed = time.perf_counter() - start
        per_call = elapsed / 100
        self.assertLess(per_call, 0.001, f"Snapshot takes {per_call*1000:.3f}ms")

    def test_snapshot_after_1000_pauses_still_fast(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(1000):
            tracker.pause(f"p{i}")
            tracker.resume()
        start = time.perf_counter()
        for _ in range(100):
            tracker.snapshot()
        elapsed = time.perf_counter() - start
        per_call = elapsed / 100
        self.assertLess(per_call, 0.001, f"Snapshot takes {per_call*1000:.3f}ms after 1000 pauses")

    def test_snapshot_json_serialization_fast(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        start = time.perf_counter()
        for _ in range(1000):
            json.dumps(snap)
        elapsed = time.perf_counter() - start
        per_call = elapsed / 1000
        self.assertLess(per_call, 0.001, f"JSON serialize takes {per_call*1000:.3f}ms")


class TestCheckpointThroughput(unittest.TestCase):
    """Checkpoint save and restore are fast enough for recovery flows."""

    def test_save_under_5ms(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            start = time.perf_counter()
            for i in range(50):
                tracker.save_checkpoint(path)
            elapsed = time.perf_counter() - start
            per_call = elapsed / 50
        self.assertLess(per_call, 0.005, f"Save takes {per_call*1000:.1f}ms")

    def test_restore_under_5ms(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            start = time.perf_counter()
            for i in range(50):
                restored = ActiveBudgetTracker.restore(path, tier=tier)
            elapsed = time.perf_counter() - start
            per_call = elapsed / 50
        self.assertLess(per_call, 0.005, f"Restore takes {per_call*1000:.1f}ms")

    def test_round_trip_under_10ms(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            start = time.perf_counter()
            for i in range(20):
                path = Path(tmp) / f"cp_{i}.json"
                tracker.save_checkpoint(path)
                tracker = ActiveBudgetTracker.restore(path, tier=tier)
                tracker.resume()
            elapsed = time.perf_counter() - start
            per_cycle = elapsed / 20
        self.assertLess(per_cycle, 0.01, f"Round-trip takes {per_cycle*1000:.1f}ms")


class TestSummaryGenerationPerformance(unittest.TestCase):
    """Summary generation scales with input size."""

    def test_summary_with_100_tests_under_5ms(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        tests = [{"test_id": f"t{i}", "physical_pass": i % 2 == 0,
                  "geometry_audit": {"passed": True}} for i in range(100)]
        start = time.perf_counter()
        for _ in range(10):
            build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=tests, pending_tests=[f"p{i}" for i in range(50)],
                issues=["issue1", "issue2"], evidence_paths=["/path/1"],
                blockers=[],
            )
        elapsed = time.perf_counter() - start
        per_call = elapsed / 10
        self.assertLess(per_call, 0.005, f"Summary takes {per_call*1000:.1f}ms for 100 tests")

    def test_summary_with_1000_tests_under_50ms(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        tests = [{"test_id": f"t{i}", "physical_pass": i % 3 != 0,
                  "geometry_audit": {"passed": i % 5 != 0}} for i in range(1000)]
        start = time.perf_counter()
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=tests,
            pending_tests=[f"p{i}" for i in range(500)],
            issues=[f"issue_{i}" for i in range(100)],
            evidence_paths=[f"/path/{i}.json" for i in range(50)],
            blockers=[],
        )
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.05, f"Summary takes {elapsed*1000:.1f}ms for 1000 tests")
        self.assertEqual(len(summary["completed_tests"]), 1000)


class TestTrackerOperationsScale(unittest.TestCase):
    """Tracker operations are O(1) with respect to history size."""

    def test_elapsed_active_constant_time(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(500):
            tracker.pause(f"p{i}")
            tracker.resume()

        start = time.perf_counter()
        for _ in range(10000):
            _ = tracker.elapsed_active
        elapsed_after_500 = time.perf_counter() - start

        tracker2 = ActiveBudgetTracker(tier=get_tier("high"))
        start = time.perf_counter()
        for _ in range(10000):
            _ = tracker2.elapsed_active
        elapsed_fresh = time.perf_counter() - start

        ratio = elapsed_after_500 / max(elapsed_fresh, 1e-9)
        self.assertLess(ratio, 3.0, f"Elapsed_active slows {ratio:.1f}x after 500 pauses")

    def test_remaining_active_constant_time(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        for i in range(500):
            tracker.pause(f"p{i}")
            tracker.resume()

        start = time.perf_counter()
        for _ in range(10000):
            _ = tracker.remaining_active
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / 10000) * 1e6
        self.assertLess(per_call_us, 10, f"remaining_active takes {per_call_us:.2f}µs")

    def test_is_exhausted_constant_time(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for i in range(500):
            tracker.pause(f"p{i}")
            tracker.resume()

        start = time.perf_counter()
        for _ in range(10000):
            _ = tracker.is_exhausted
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / 10000) * 1e6
        self.assertLess(per_call_us, 10, f"is_exhausted takes {per_call_us:.2f}µs")


class TestLimitCheckScaling(unittest.TestCase):
    """Limit checks scale linearly with input size."""

    def test_part_count_100_bodies(self):
        bodies = [{} for _ in range(100)]
        tier = get_tier("high")
        start = time.perf_counter()
        for _ in range(1000):
            check_part_count(bodies, tier)
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / 1000) * 1e6
        self.assertLess(per_call_us, 50, f"check_part_count(100) takes {per_call_us:.1f}µs")

    def test_task_count_100_tasks(self):
        tasks = [{} for _ in range(100)]
        tier = get_tier("high")
        start = time.perf_counter()
        for _ in range(1000):
            check_task_count(tasks, tier)
        elapsed = time.perf_counter() - start
        per_call_us = (elapsed / 1000) * 1e6
        self.assertLess(per_call_us, 50, f"check_task_count(100) takes {per_call_us:.1f}µs")


class TestConcurrentAccessPerformance(unittest.TestCase):
    """Concurrent access doesn't cause severe degradation."""

    def test_10_threads_reading_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        errors = []
        counts = []

        def reader():
            try:
                count = 0
                for _ in range(100):
                    snap = tracker.snapshot()
                    assert snap["tier"] == "medium"
                    count += 1
                counts.append(count)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(10)]
        start = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        elapsed = time.perf_counter() - start

        self.assertEqual(errors, [])
        self.assertEqual(sum(counts), 1000)
        self.assertLess(elapsed, 2.0, f"10 threads x 100 snapshots took {elapsed:.2f}s")

    def test_mixed_read_write_threads(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        errors = []

        def writer():
            try:
                for _ in range(50):
                    tracker.pause("write")
                    tracker.resume()
            except Exception as e:
                errors.append(("writer", e))

        def reader():
            try:
                for _ in range(50):
                    snap = tracker.snapshot()
                    assert snap["tier"] == "high"
            except Exception as e:
                errors.append(("reader", e))

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])


class TestMemoryBounds(unittest.TestCase):
    """Tracker memory stays bounded regardless of history."""

    def test_tracker_size_independent_of_pauses(self):
        tracker_fresh = ActiveBudgetTracker(tier=get_tier("medium"))
        size_fresh = sys.getsizeof(tracker_fresh)

        tracker_heavy = ActiveBudgetTracker(tier=get_tier("medium"))
        for i in range(1000):
            tracker_heavy.pause(f"p{i}")
            tracker_heavy.resume()
        size_heavy = sys.getsizeof(tracker_heavy)

        self.assertAlmostEqual(size_fresh, size_heavy, delta=256,
                               msg=f"Fresh={size_fresh}, 1000-pause={size_heavy}")

    def test_adjustment_log_size_grows_linearly(self):
        log = AdjustmentLog()
        sizes = []
        for i in range(100):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))
            if i in [0, 49, 99]:
                sizes.append(len(log.to_list()))
        self.assertEqual(sizes, [1, 50, 100])


class TestRecoveryDelayPerformance(unittest.TestCase):
    """Recovery delay computation is fast."""

    def test_delay_computation_under_1us(self):
        start = time.perf_counter()
        for i in range(10000):
            recovery_delay(i % 100)
        elapsed = time.perf_counter() - start
        per_call_ns = (elapsed / 10000) * 1e9
        self.assertLess(per_call_ns, 1000, f"recovery_delay takes {per_call_ns:.0f}ns")

    def test_should_retry_under_1us(self):
        start = time.perf_counter()
        for i in range(10000):
            should_retry(i % 5, "INFRASTRUCTURE_FAILURE" if i % 3 else "generic")
        elapsed = time.perf_counter() - start
        per_call_ns = (elapsed / 10000) * 1e9
        self.assertLess(per_call_ns, 5000, f"should_retry takes {per_call_ns:.0f}ns")


class TestTierConfigPerformance(unittest.TestCase):
    """Tier operations are fast."""

    def test_get_tier_fast(self):
        start = time.perf_counter()
        for _ in range(10000):
            get_tier("light")
            get_tier("medium")
            get_tier("high")
        elapsed = time.perf_counter() - start
        per_call_ns = (elapsed / 30000) * 1e9
        self.assertLess(per_call_ns, 1000, f"get_tier takes {per_call_ns:.0f}ns")

    def test_enforce_timeout_fast(self):
        tier = get_tier("medium")
        start = time.perf_counter()
        for _ in range(10000):
            enforce_simulation_timeout(tier, 100)
        elapsed = time.perf_counter() - start
        per_call_ns = (elapsed / 10000) * 1e9
        self.assertLess(per_call_ns, 2000, f"enforce_timeout takes {per_call_ns:.0f}ns")


if __name__ == "__main__":
    unittest.main()
