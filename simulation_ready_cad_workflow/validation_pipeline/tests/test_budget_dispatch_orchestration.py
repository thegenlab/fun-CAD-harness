"""Budget dispatch orchestration tests.

Tests the budget module's behavior in realistic dispatch/retry scenarios:
- Retry loops with budget awareness
- Tier escalation from light to high under budget pressure
- Multi-stage dispatch with checkpoint handoff
- Concurrent dispatch with independent trackers
- Recovery delay integration with should_retry
- Budget-aware task scheduling under exhaustion
- Adjustment log persistence through dispatch cycles
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
    BACKOFF_BASE_SECONDS,
)


class TestRetryLoopWithBudget(unittest.TestCase):
    """Retry loops that respect budget exhaustion."""

    def test_retry_stops_on_exhaustion(self):
        tier = TierConfig("micro", 0.02, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        attempts = 0
        while not tracker.is_exhausted and attempts < 100:
            time.sleep(0.005)
            attempts += 1
        self.assertTrue(tracker.is_exhausted)
        self.assertGreater(attempts, 0)
        self.assertLess(attempts, 100)

    def test_retry_respects_should_retry_on_pass(self):
        results = []
        for attempt in range(5):
            if attempt == 2:
                error = "PASS"
            else:
                error = "INFRASTRUCTURE_FAILURE: connection reset"
            if not should_retry(attempt, error):
                results.append(("stopped", attempt, error))
                break
            results.append(("retried", attempt, error))
        self.assertEqual(results[-1][0], "stopped")
        self.assertEqual(results[-1][1], 2)

    def test_retry_respects_max_retries(self):
        attempts = []
        for attempt in range(10):
            error = "INFRASTRUCTURE_FAILURE"
            if not should_retry(attempt, error):
                break
            attempts.append(attempt)
        self.assertEqual(len(attempts), MAX_RECOVERY_RETRIES)

    def test_retry_with_checkpoint_between_attempts(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            for attempt in range(3):
                time.sleep(0.005)
                cp = Path(tmp) / f"retry_{attempt}.json"
                tracker.save_checkpoint(cp)
                if attempt < 2:
                    tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                    tracker.resume()
            final_snap = tracker.snapshot()
            self.assertGreater(final_snap["active_seconds"], 0.01)
            self.assertEqual(final_snap["tier"], "medium")

    def test_retry_budget_aware_stops_before_full_exhaustion(self):
        tier = TierConfig("small", 0.1, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        dispatched = 0
        while tracker.remaining_active > 0.02:
            time.sleep(0.01)
            dispatched += 1
        self.assertGreater(dispatched, 0)
        self.assertGreater(tracker.remaining_active, 0)

    def test_pause_during_retry_wait_excludes_time(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        active_before = tracker.elapsed_active
        tracker.pause("retry_backoff")
        time.sleep(0.03)
        tracker.resume()
        active_after = tracker.elapsed_active
        growth = active_after - active_before
        self.assertLess(growth, 0.02)


class TestTierEscalationFlow(unittest.TestCase):
    """Tier escalation from light to medium to high."""

    def test_escalate_preserves_active_time(self):
        light = get_tier("light")
        medium = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=light)
        time.sleep(0.02)
        active_at_light = tracker.elapsed_active

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "escalate.json"
            tracker.save_checkpoint(cp)
            escalated = ActiveBudgetTracker.restore(cp, tier=medium)
            escalated.resume()

        self.assertEqual(escalated.tier.name, "medium")
        self.assertAlmostEqual(escalated.elapsed_active, active_at_light, delta=0.01)
        self.assertGreater(escalated.remaining_active, light.active_budget_seconds - active_at_light)

    def test_full_escalation_chain(self):
        tiers = [get_tier("light"), get_tier("medium"), get_tier("high")]
        tracker = ActiveBudgetTracker(tier=tiers[0])
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            for i, next_tier in enumerate(tiers[1:], 1):
                cp = Path(tmp) / f"step_{i}.json"
                tracker.save_checkpoint(cp)
                tracker = ActiveBudgetTracker.restore(cp, tier=next_tier)
                tracker.resume()
                time.sleep(0.005)

        self.assertEqual(tracker.tier.name, "high")
        self.assertGreater(tracker.elapsed_active, 0.015)

    def test_escalation_unlocks_more_parts(self):
        light = get_tier("light")
        high = get_tier("high")
        bodies_11 = [{}] * 11
        self.assertGreater(len(check_part_count(bodies_11, light)), 0)
        self.assertEqual(len(check_part_count(bodies_11, high)), 0)

    def test_escalation_unlocks_more_tasks(self):
        light = get_tier("light")
        high = get_tier("high")
        tasks_3 = [{}] * 3
        self.assertGreater(len(check_task_count(tasks_3, light)), 0)
        self.assertEqual(len(check_task_count(tasks_3, high)), 0)

    def test_escalation_unlocks_longer_timeout(self):
        light = get_tier("light")
        high = get_tier("high")
        requested = 500.0
        light_timeout = enforce_simulation_timeout(light, requested)
        high_timeout = enforce_simulation_timeout(high, requested)
        self.assertEqual(light_timeout, light.simulation_timeout_seconds)
        self.assertEqual(high_timeout, high.simulation_timeout_seconds)
        self.assertGreater(high_timeout, light_timeout)

    def test_downgrade_reduces_remaining(self):
        high = get_tier("high")
        light = get_tier("light")
        tracker = ActiveBudgetTracker(tier=high)
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "downgrade.json"
            tracker.save_checkpoint(cp)
            downgraded = ActiveBudgetTracker.restore(cp, tier=light)
            downgraded.resume()

        self.assertEqual(downgraded.tier.name, "light")
        self.assertLess(downgraded.remaining_active, high.active_budget_seconds)


class TestMultiStageDispatch(unittest.TestCase):
    """Multi-stage dispatch with checkpoint handoff between stages."""

    def test_geometry_then_physics_stages(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)

        time.sleep(0.01)
        bodies = [{}] * 5
        violations = check_part_count(bodies, tier)
        self.assertEqual(violations, [])

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "stage1.json"
            tracker.save_checkpoint(cp)
            tracker = ActiveBudgetTracker.restore(cp, tier=tier)
            tracker.resume()

        time.sleep(0.01)
        timeout = enforce_simulation_timeout(tier, 120)
        self.assertEqual(timeout, 120)

        snap = tracker.snapshot()
        self.assertGreater(snap["active_seconds"], 0.015)
        self.assertGreater(snap["pause_count"], 0)

    def test_three_stage_pipeline(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        stage_times = []

        with tempfile.TemporaryDirectory() as tmp:
            for stage in range(3):
                time.sleep(0.01)
                stage_times.append(tracker.elapsed_active)
                cp = Path(tmp) / f"stage_{stage}.json"
                tracker.save_checkpoint(cp)
                if stage < 2:
                    tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                    tracker.resume()

        for i in range(1, len(stage_times)):
            self.assertGreater(stage_times[i], stage_times[i-1])

    def test_checkpoint_chain_recovers_from_any_point(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        checkpoints = []

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                time.sleep(0.005)
                cp = Path(tmp) / f"cp_{i}.json"
                tracker.save_checkpoint(cp)
                checkpoints.append(cp)
                if i < 4:
                    tracker.pause(f"stage_gap_{i}")
                    tracker.resume()

            mid_restore = ActiveBudgetTracker.restore(checkpoints[2], tier=tier)
            mid_restore.resume()
            self.assertGreater(mid_restore.elapsed_active, 0.01)
            self.assertLess(mid_restore.elapsed_active, tracker.elapsed_active)


class TestConcurrentDispatch(unittest.TestCase):
    """Concurrent dispatch with independent budget trackers."""

    def test_parallel_tasks_independent_budgets(self):
        tiers = [get_tier("light"), get_tier("medium"), get_tier("high")]
        trackers = [ActiveBudgetTracker(tier=t) for t in tiers]
        errors = []
        results = [None, None, None]

        def run_task(idx):
            try:
                time.sleep(0.01)
                trackers[idx].pause(f"task_{idx}")
                time.sleep(0.005)
                trackers[idx].resume()
                results[idx] = trackers[idx].snapshot()
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=run_task, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        for i, result in enumerate(results):
            self.assertIsNotNone(result)
            self.assertEqual(result["tier"], tiers[i].name)
            self.assertEqual(result["pause_count"], 1)

    def test_concurrent_checkpoints_no_interference(self):
        tier = get_tier("medium")
        trackers = [ActiveBudgetTracker(tier=tier) for _ in range(4)]
        errors = []

        def checkpoint_task(idx, tmp_dir):
            try:
                time.sleep(0.005 * (idx + 1))
                cp = Path(tmp_dir) / f"worker_{idx}.json"
                trackers[idx].save_checkpoint(cp)
                restored = ActiveBudgetTracker.restore(cp, tier=tier)
                restored.resume()
                assert restored.tier.name == "medium"
            except Exception as e:
                errors.append((idx, e))

        with tempfile.TemporaryDirectory() as tmp:
            threads = [threading.Thread(target=checkpoint_task, args=(i, tmp)) for i in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        self.assertEqual(errors, [])

    def test_shared_adjustment_log_threadsafe_appends(self):
        log = AdjustmentLog()
        errors = []

        def append_adjustments(thread_id):
            try:
                for i in range(10):
                    adj = CriteriaAdjustment(
                        criterion_id=f"c_{thread_id}_{i}",
                        before={"v": i},
                        after={"v": i + 1},
                        reason=f"thread {thread_id} step {i}",
                    )
                    log.record(adj)
            except Exception as e:
                errors.append((thread_id, e))

        threads = [threading.Thread(target=append_adjustments, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(errors, [])
        self.assertEqual(len(log.to_list()), 50)


class TestRecoveryDelayIntegration(unittest.TestCase):
    """Recovery delay computation with retry decisions."""

    def test_delay_increases_with_attempts(self):
        delays = [recovery_delay(i) for i in range(5)]
        for i in range(1, len(delays)):
            self.assertGreaterEqual(delays[i], delays[i-1])

    def test_delay_capped_at_60(self):
        for attempt in range(20):
            self.assertLessEqual(recovery_delay(attempt), 60.0)

    def test_first_delay_is_base(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_delay_schedule(self):
        expected = [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
        for i, exp in enumerate(expected):
            self.assertEqual(recovery_delay(i), exp)

    def test_retry_stops_after_max_attempts(self):
        retried = 0
        for attempt in range(10):
            if should_retry(attempt, "INFRASTRUCTURE_FAILURE"):
                retried += 1
            else:
                break
        self.assertEqual(retried, MAX_RECOVERY_RETRIES)

    def test_retry_integrates_with_pause_for_delay(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        for attempt in range(MAX_RECOVERY_RETRIES):
            if not should_retry(attempt, "INFRASTRUCTURE_FAILURE"):
                break
            tracker.pause(f"recovery_wait_{attempt}")
            time.sleep(0.005)
            tracker.resume()

        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], MAX_RECOVERY_RETRIES)
        self.assertGreater(snap["total_pause_seconds"], 0)


class TestBudgetAwareScheduling(unittest.TestCase):
    """Scheduling decisions based on budget state."""

    def test_schedule_small_tasks_when_low_budget(self):
        tier = TierConfig("tiny", 0.05, 60, 5, 0, 20)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        remaining = tracker.remaining_active
        self.assertLess(remaining, tier.active_budget_seconds * 0.5)
        self.assertGreater(remaining, 0)

    def test_summary_at_exhaustion_captures_pending(self):
        tier = TierConfig("micro", 0.01, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "done1", "physical_pass": True, "geometry_audit": {"passed": True}}],
            pending_tests=["pending1", "pending2"],
            issues=["exhaustion"],
            evidence_paths=["/evidence/1.json"],
            blockers=[],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(len(summary["completed_tests"]), 1)
        self.assertEqual(len(summary["pending_tests"]), 2)
        self.assertEqual(summary["severity"], "functional")

    def test_summary_minor_when_all_done(self):
        tier = TierConfig("micro", 0.01, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}}],
            pending_tests=[],
            issues=[],
            evidence_paths=[],
            blockers=[],
        )
        self.assertEqual(summary["severity"], "minor")

    def test_summary_functional_when_tests_failed(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "t1", "physical_pass": False, "geometry_audit": {"passed": True}}],
            pending_tests=[],
            issues=[],
            evidence_paths=[],
            blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")


class TestAdjustmentLogDispatch(unittest.TestCase):
    """Adjustment log persistence through dispatch cycles."""

    def test_log_round_trip_through_file(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "tolerance"))
        log.record(CriteriaAdjustment("c2", {"b": 10}, {"b": 20}, "precision"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adj_log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)

        entries = loaded.to_list()
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0]["criterion_id"], "c1")
        self.assertEqual(entries[1]["criterion_id"], "c2")

    def test_log_grows_through_stages(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            for stage in range(3):
                log.record(CriteriaAdjustment(
                    f"c_stage{stage}", {"v": stage}, {"v": stage + 1},
                    f"stage {stage} adjustment",
                ))
                path = Path(tmp) / f"stage_{stage}_log.json"
                log.save(path)

            final = AdjustmentLog.load(path)
            self.assertEqual(len(final.to_list()), 3)

    def test_log_entries_have_criterion_id_key(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("metric_x", {"old": 0}, {"old": 1}, "update"))
        entries = log.to_list()
        self.assertIn("criterion_id", entries[0])
        self.assertNotIn("criterion", entries[0])

    def test_empty_log_round_trip(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(len(loaded.to_list()), 0)

    def test_log_schema_version(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "versioned.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")

    def test_bad_schema_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"schema_version": "wrong_v99", "adjustments": []}))
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)


class TestSimulationTimeoutDispatch(unittest.TestCase):
    """Timeout enforcement in dispatch scenarios."""

    def test_timeout_result_pass(self):
        result = SimulationTimeoutResult(
            timed_out=False,
            elapsed_seconds=55.0,
        )
        self.assertEqual(result.status, BudgetStatus.PASS)

    def test_timeout_result_timeout(self):
        result = SimulationTimeoutResult(
            timed_out=True,
            elapsed_seconds=60.0,
        )
        self.assertEqual(result.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_enforce_clamps_to_tier(self):
        for name, tier in TIERS.items():
            over = tier.simulation_timeout_seconds * 3
            clamped = enforce_simulation_timeout(tier, over)
            self.assertEqual(clamped, tier.simulation_timeout_seconds)

    def test_enforce_passes_through_smaller(self):
        for name, tier in TIERS.items():
            under = tier.simulation_timeout_seconds * 0.5
            result = enforce_simulation_timeout(tier, under)
            self.assertEqual(result, under)

    def test_enforce_no_override_returns_tier_max(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier)
            self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_enforce_rejects_zero(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("medium"), 0)

    def test_enforce_rejects_negative(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("light"), -1)

    def test_enforce_rejects_inf(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("high"), float("inf"))

    def test_enforce_rejects_nan(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("medium"), float("nan"))


class TestBudgetStatusDispatch(unittest.TestCase):
    """Budget status integration with dispatch decisions."""

    def test_all_statuses_have_terminal_flag(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.is_terminal(), bool)

    def test_pass_is_terminal(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_exhausted_is_terminal(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())

    def test_infrastructure_failure_is_terminal(self):
        self.assertTrue(BudgetStatus.INFRASTRUCTURE_FAILURE.is_terminal())

    def test_non_terminal_statuses_exist(self):
        non_terminal = [s for s in BudgetStatus if not s.is_terminal()]
        self.assertGreater(len(non_terminal), 0)

    def test_pass_not_retryable(self):
        self.assertFalse(should_retry(0, BudgetStatus.PASS.value))

    def test_exhausted_not_retryable(self):
        self.assertFalse(should_retry(0, BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.value))

    def test_infra_failure_retryable_at_zero(self):
        self.assertTrue(should_retry(0, BudgetStatus.INFRASTRUCTURE_FAILURE.value))

    def test_simulation_timeout_retryable(self):
        self.assertTrue(should_retry(0, BudgetStatus.SIMULATION_TIMEOUT.value))


class TestEndToEndDispatchScenario(unittest.TestCase):
    """Full dispatch scenario: validate, simulate, retry, checkpoint, summary."""

    def test_happy_path_dispatch(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)

        bodies = [{}] * 5
        tasks = [{}] * 2
        self.assertEqual(check_part_count(bodies, tier), [])
        self.assertEqual(check_task_count(tasks, tier), [])

        timeout = enforce_simulation_timeout(tier, 100)
        self.assertEqual(timeout, 100)

        time.sleep(0.01)
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=50.0)
        self.assertEqual(result.status, BudgetStatus.PASS)

        snap = tracker.snapshot()
        self.assertGreater(snap["active_seconds"], 0)
        self.assertFalse(tracker.is_exhausted)

    def test_failure_retry_checkpoint_dispatch(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = []

        with tempfile.TemporaryDirectory() as tmp:
            for attempt in range(MAX_RECOVERY_RETRIES + 1):
                time.sleep(0.005)
                error = "INFRASTRUCTURE_FAILURE: timeout"
                if attempt == 2:
                    completed.append({"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}})
                    break
                if not should_retry(attempt, error):
                    break
                tracker.pause(f"retry_{attempt}")
                cp = Path(tmp) / f"cp_{attempt}.json"
                tracker.save_checkpoint(cp)
                tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                tracker.resume()

        self.assertEqual(len(completed), 1)
        self.assertFalse(tracker.is_exhausted)

    def test_exhaustion_summary_dispatch(self):
        tier = TierConfig("tiny", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)

        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[],
            pending_tests=["task_a", "task_b"],
            issues=["budget exhausted before dispatch"],
            evidence_paths=[],
            blockers=["ACTIVE_BUDGET_EXHAUSTED"],
        )
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(len(summary["pending_tests"]), 2)
        self.assertEqual(len(summary["blockers"]), 1)


if __name__ == "__main__":
    unittest.main()
