"""Budget concurrency pattern tests.

Validates thread-safety and isolation of budget primitives under
concurrent access patterns that mirror real multi-design validation:
- Independent trackers on separate threads
- Concurrent checkpoint save/restore
- Concurrent limit checks
- Concurrent adjustment log operations
- Race conditions in pause/resume sequences
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
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
)


class TestConcurrentTrackerIsolation(unittest.TestCase):
    """Trackers on different threads don't interfere."""

    def test_three_tiers_parallel(self):
        results = {}
        errors = []

        def run_tier(name):
            try:
                tier = get_tier(name)
                tracker = ActiveBudgetTracker(tier=tier)
                for _ in range(10):
                    time.sleep(0.002)
                    tracker.pause(f"p_{name}")
                    time.sleep(0.001)
                    tracker.resume()
                snap = tracker.snapshot()
                results[name] = snap
            except Exception as e:
                errors.append((name, e))

        threads = [threading.Thread(target=run_tier, args=(n,)) for n in ("light", "medium", "high")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(errors, [])
        for name in ("light", "medium", "high"):
            self.assertIn(name, results)
            self.assertEqual(results[name]["tier"], name)
            self.assertEqual(results[name]["pause_count"], 10)

    def test_twenty_trackers_same_tier(self):
        tier = get_tier("medium")
        results = [None] * 20
        errors = []

        def run_tracker(idx):
            try:
                tracker = ActiveBudgetTracker(tier=tier)
                time.sleep(0.005)
                for _ in range(5):
                    tracker.pause(f"p_{idx}")
                    time.sleep(0.001)
                    tracker.resume()
                results[idx] = tracker.snapshot()
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=run_tracker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        for i in range(20):
            self.assertIsNotNone(results[i], f"Tracker {i}")
            self.assertEqual(results[i]["pause_count"], 5, f"Tracker {i}")
            self.assertGreater(results[i]["active_seconds"], 0, f"Tracker {i}")


class TestConcurrentCheckpointSaveRestore(unittest.TestCase):
    """Concurrent checkpoint operations to different files."""

    def test_parallel_save_different_files(self):
        errors = []

        def save_checkpoint(idx, tmp_dir):
            try:
                tier = get_tier(["light", "medium", "high"][idx % 3])
                tracker = ActiveBudgetTracker(tier=tier)
                time.sleep(0.005)
                cp = Path(tmp_dir) / f"cp_{idx}.json"
                tracker.save_checkpoint(cp)
            except Exception as e:
                errors.append((idx, e))

        with tempfile.TemporaryDirectory() as tmp:
            threads = [threading.Thread(target=save_checkpoint, args=(i, tmp)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

            self.assertEqual(errors, [])
            for i in range(10):
                cp = Path(tmp) / f"cp_{i}.json"
                self.assertTrue(cp.exists(), f"Checkpoint {i} missing")
                data = json.loads(cp.read_text(encoding="utf-8"))
                self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_parallel_save_and_restore(self):
        tier = get_tier("medium")
        errors = []
        restored_snaps = [None] * 5

        with tempfile.TemporaryDirectory() as tmp:
            trackers = []
            for i in range(5):
                t = ActiveBudgetTracker(tier=tier)
                time.sleep(0.005)
                cp = Path(tmp) / f"cp_{i}.json"
                t.save_checkpoint(cp)
                trackers.append(cp)

            def restore_one(idx):
                try:
                    restored = ActiveBudgetTracker.restore(trackers[idx], tier=tier)
                    restored.resume()
                    time.sleep(0.003)
                    restored_snaps[idx] = restored.snapshot()
                except Exception as e:
                    errors.append((idx, e))

            threads = [threading.Thread(target=restore_one, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

        self.assertEqual(errors, [])
        for i in range(5):
            self.assertIsNotNone(restored_snaps[i], f"Restore {i}")
            self.assertEqual(restored_snaps[i]["tier"], "medium")


class TestConcurrentLimitChecks(unittest.TestCase):
    """Part/task/timeout checks from multiple threads."""

    def test_parallel_part_checks(self):
        results = [None] * 10
        errors = []

        def check_parts(idx):
            try:
                tier = get_tier(["light", "medium", "high"][idx % 3])
                count = tier.max_parts + (1 if idx % 2 else 0)
                violations = check_part_count([{}] * count, tier)
                results[idx] = len(violations)
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=check_parts, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i in range(10):
            expected = 1 if i % 2 else 0
            self.assertEqual(results[i], expected, f"Check {i}")

    def test_parallel_task_checks(self):
        results = [None] * 10
        errors = []

        def check_tasks(idx):
            try:
                tier = get_tier(["light", "medium", "high"][idx % 3])
                count = tier.max_tasks_per_design + (1 if idx % 2 else 0)
                violations = check_task_count([{}] * count, tier)
                results[idx] = len(violations)
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=check_tasks, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i in range(10):
            expected = 1 if i % 2 else 0
            self.assertEqual(results[i], expected, f"Check {i}")

    def test_parallel_timeout_enforcement(self):
        results = [None] * 9
        errors = []

        def enforce_timeout(idx):
            try:
                tier = get_tier(["light", "medium", "high"][idx % 3])
                requested = tier.simulation_timeout_seconds * (0.5 + idx * 0.2)
                result = enforce_simulation_timeout(tier, max(0.001, requested))
                results[idx] = result
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=enforce_timeout, args=(i,)) for i in range(9)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i in range(9):
            tier = get_tier(["light", "medium", "high"][i % 3])
            self.assertIsNotNone(results[i], f"Timeout {i}")
            self.assertLessEqual(results[i], tier.simulation_timeout_seconds, f"Timeout {i}")


class TestConcurrentAdjustmentLog(unittest.TestCase):
    """Adjustment log operations from threads (single log, sequential record)."""

    def test_sequential_records_then_parallel_read(self):
        log = AdjustmentLog()
        for i in range(10):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))

        results = [None] * 5
        errors = []

        def read_log(idx):
            try:
                results[idx] = log.to_list()
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=read_log, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i in range(5):
            self.assertEqual(len(results[i]), 10, f"Read {i}")

    def test_parallel_save_to_different_files(self):
        log = AdjustmentLog()
        for i in range(5):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))

        errors = []

        with tempfile.TemporaryDirectory() as tmp:
            def save_log(idx):
                try:
                    path = Path(tmp) / f"log_{idx}.json"
                    log.save(path)
                except Exception as e:
                    errors.append((idx, e))

            threads = [threading.Thread(target=save_log, args=(i,)) for i in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

            self.assertEqual(errors, [])
            for i in range(5):
                path = Path(tmp) / f"log_{i}.json"
                self.assertTrue(path.exists(), f"Log {i}")
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(len(data["adjustments"]), 5)


class TestConcurrentSimulationTimeout(unittest.TestCase):
    """SimulationTimeoutResult creation from threads."""

    def test_parallel_result_creation(self):
        results = [None] * 10
        errors = []

        def create_result(idx):
            try:
                timed_out = idx % 2 == 0
                r = SimulationTimeoutResult(
                    timed_out=timed_out,
                    elapsed_seconds=float(idx * 10),
                )
                results[idx] = r.status
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=create_result, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i in range(10):
            expected = BudgetStatus.SIMULATION_TIMEOUT if i % 2 == 0 else BudgetStatus.PASS
            self.assertEqual(results[i], expected, f"Result {i}")


class TestConcurrentRecoveryDecisions(unittest.TestCase):
    """Recovery delay and retry decisions from threads."""

    def test_parallel_recovery_delay(self):
        results = [None] * 10
        errors = []

        def compute_delay(idx):
            try:
                results[idx] = recovery_delay(idx)
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=compute_delay, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i in range(10):
            self.assertEqual(results[i], recovery_delay(i), f"Delay {i}")

    def test_parallel_should_retry(self):
        cases = [
            (0, "INFRASTRUCTURE_FAILURE", True),
            (1, "INFRASTRUCTURE_FAILURE", True),
            (2, "INFRASTRUCTURE_FAILURE", True),
            (3, "INFRASTRUCTURE_FAILURE", False),
            (0, "ACTIVE_BUDGET_EXHAUSTED", False),
            (0, "TERMINATED", False),
            (0, "PASS", False),
            (0, "unknown error", True),
        ]
        results = [None] * len(cases)
        errors = []

        def check_retry(idx, attempt, error, expected):
            try:
                results[idx] = should_retry(attempt, error)
            except Exception as e:
                errors.append((idx, e))

        threads = [
            threading.Thread(target=check_retry, args=(i, *c))
            for i, c in enumerate(cases)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i, (attempt, error, expected) in enumerate(cases):
            self.assertEqual(results[i], expected, f"Case {i}: attempt={attempt}, error={error}")


class TestConcurrentSummaryGeneration(unittest.TestCase):
    """build_incomplete_summary from threads with different trackers."""

    def test_parallel_summaries(self):
        results = [None] * 6
        errors = []

        def generate_summary(idx):
            try:
                tier = get_tier(["light", "medium", "high"][idx % 3])
                tracker = ActiveBudgetTracker(tier=tier)
                time.sleep(0.005)
                summary = build_incomplete_summary(
                    tier=tier, tracker=tracker,
                    completed_tests=[{"test_id": f"t{idx}", "physical_pass": True,
                                     "geometry_audit": {"passed": True}}],
                    pending_tests=[f"pending_{idx}"],
                    issues=[], evidence_paths=[], blockers=[],
                )
                results[idx] = summary
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=generate_summary, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        self.assertEqual(errors, [])
        for i in range(6):
            self.assertIsNotNone(results[i], f"Summary {i}")
            self.assertEqual(results[i]["schema_version"], "budget_incomplete_summary_v1")
            self.assertEqual(len(results[i]["completed_tests"]), 1)
            self.assertEqual(results[i]["severity"], "functional")


class TestConcurrentCheckpointChain(unittest.TestCase):
    """Checkpoint chain operations interleaved between threads."""

    def test_two_chains_independent(self):
        errors = []
        final_snaps = [None, None]

        def run_chain(idx, tier_name):
            try:
                tier = get_tier(tier_name)
                tracker = ActiveBudgetTracker(tier=tier)
                with tempfile.TemporaryDirectory() as tmp:
                    for step in range(3):
                        time.sleep(0.005)
                        tracker.pause(f"step_{step}")
                        cp = Path(tmp) / f"chain_{idx}_{step}.json"
                        tracker.save_checkpoint(cp)
                        tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                        tracker.resume()
                    final_snaps[idx] = tracker.snapshot()
            except Exception as e:
                errors.append((idx, e))

        t1 = threading.Thread(target=run_chain, args=(0, "light"))
        t2 = threading.Thread(target=run_chain, args=(1, "high"))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        self.assertEqual(errors, [])
        self.assertEqual(final_snaps[0]["tier"], "light")
        self.assertEqual(final_snaps[1]["tier"], "high")
        self.assertGreater(final_snaps[0]["active_seconds"], 0)
        self.assertGreater(final_snaps[1]["active_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
