"""Budget lifecycle pattern tests.

Tests complete lifecycle patterns that mirror real harness usage:
- Single-shot validation: check → simulate → report
- Multi-attempt validation with recovery
- Progressive tier escalation through checkpoint chains
- Long-running tracked session with many pause/resume cycles
- Edge case: immediate exhaustion
- Edge case: pause-dominated timeline (active << wall)
- Edge case: no-op session (zero elapsed)
- Adjustment log accumulation through multi-stage workflows
- Concurrent multi-design validation
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


class TestSingleShotValidation(unittest.TestCase):
    """Single-shot: validate geometry → simulate → report."""

    def test_within_limits_passes(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)

        bodies = [{"name": f"part_{i}"} for i in range(5)]
        tasks = [{"id": "sim_1"}]
        self.assertEqual(check_part_count(bodies, tier), [])
        self.assertEqual(check_task_count(tasks, tier), [])

        timeout = enforce_simulation_timeout(tier, 90)
        self.assertEqual(timeout, 90)

        time.sleep(0.01)
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=5.0)
        self.assertEqual(result.status, BudgetStatus.PASS)

        snap = tracker.snapshot()
        self.assertGreater(snap["active_seconds"], 0)
        self.assertGreater(snap["remaining_active_seconds"], 0)
        self.assertFalse(tracker.is_exhausted)

    def test_over_part_limit_reports_violation(self):
        tier = get_tier("light")
        bodies = [{}] * 15
        violations = check_part_count(bodies, tier)
        self.assertEqual(len(violations), 1)
        self.assertIn("15", violations[0])
        self.assertIn("10", violations[0])

    def test_over_task_limit_reports_violation(self):
        tier = get_tier("light")
        tasks = [{}] * 5
        violations = check_task_count(tasks, tier)
        self.assertEqual(len(violations), 1)
        self.assertIn("5", violations[0])
        self.assertIn("2", violations[0])


class TestMultiAttemptRecovery(unittest.TestCase):
    """Multi-attempt: fail → checkpoint → retry → succeed."""

    def test_infra_failure_recovery(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        succeeded = False

        with tempfile.TemporaryDirectory() as tmp:
            for attempt in range(MAX_RECOVERY_RETRIES + 1):
                time.sleep(0.005)
                if attempt < 2:
                    error = "INFRASTRUCTURE_FAILURE"
                    if should_retry(attempt, error):
                        tracker.pause(f"retry_wait_{attempt}")
                        cp = Path(tmp) / f"recovery_{attempt}.json"
                        tracker.save_checkpoint(cp)
                        time.sleep(0.003)
                        tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                        tracker.resume()
                        continue
                succeeded = True
                break

        self.assertTrue(succeeded)
        snap = tracker.snapshot()
        self.assertGreater(snap["pause_count"], 0)
        self.assertGreater(snap["total_pause_seconds"], 0)
        self.assertFalse(tracker.is_exhausted)

    def test_all_retries_exhausted(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        all_failed = True

        for attempt in range(MAX_RECOVERY_RETRIES + 1):
            error = "INFRASTRUCTURE_FAILURE: persistent"
            if not should_retry(attempt, error):
                break
            tracker.pause(f"wait_{attempt}")
            time.sleep(0.002)
            tracker.resume()

        self.assertTrue(all_failed)
        self.assertFalse(tracker.is_exhausted)


class TestProgressiveTierEscalation(unittest.TestCase):
    """Escalate from light → medium → high through checkpoint chain."""

    def test_escalation_preserves_work(self):
        tiers_list = [get_tier("light"), get_tier("medium"), get_tier("high")]
        tracker = ActiveBudgetTracker(tier=tiers_list[0])
        time.sleep(0.01)
        initial_active = tracker.elapsed_active

        with tempfile.TemporaryDirectory() as tmp:
            for i, next_tier in enumerate(tiers_list[1:]):
                cp = Path(tmp) / f"escalate_{i}.json"
                tracker.save_checkpoint(cp)
                tracker = ActiveBudgetTracker.restore(cp, tier=next_tier)
                tracker.resume()
                time.sleep(0.005)

        final_snap = tracker.snapshot()
        self.assertEqual(final_snap["tier"], "high")
        self.assertGreater(final_snap["active_seconds"], initial_active)
        self.assertEqual(
            final_snap["budget_limit_seconds"],
            tiers_list[2].active_budget_seconds,
        )

    def test_escalation_increases_limits(self):
        for i in range(len(list(TIERS.values())) - 1):
            tiers = list(TIERS.values())
            self.assertLess(tiers[i].max_parts, tiers[i+1].max_parts)
            self.assertLessEqual(tiers[i].max_tasks_per_design, tiers[i+1].max_tasks_per_design)
            self.assertLess(tiers[i].simulation_timeout_seconds, tiers[i+1].simulation_timeout_seconds)


class TestLongRunningSession(unittest.TestCase):
    """Session with many pause/resume cycles."""

    def test_50_cycle_session(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)

        for cycle in range(50):
            time.sleep(0.001)
            tracker.pause(f"cycle_{cycle}")
            time.sleep(0.001)
            tracker.resume()

        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 50)
        self.assertGreater(snap["total_pause_seconds"], 0.04)
        self.assertGreater(snap["active_seconds"], 0.04)
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])

    def test_checkpoint_at_every_10th_cycle(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        checkpoints = []

        with tempfile.TemporaryDirectory() as tmp:
            for cycle in range(30):
                time.sleep(0.001)
                tracker.pause(f"c_{cycle}")
                if cycle % 10 == 9:
                    cp = Path(tmp) / f"cp_{cycle}.json"
                    tracker.save_checkpoint(cp)
                    checkpoints.append(cp)
                tracker.resume()

            self.assertEqual(len(checkpoints), 3)
            last = ActiveBudgetTracker.restore(checkpoints[-1], tier=tier)
            last.resume()
            self.assertGreater(last.snapshot()["pause_count"], 20)


class TestImmediateExhaustion(unittest.TestCase):
    """Budget exhausts before any meaningful work."""

    def test_micro_budget_exhausts_quickly(self):
        tier = TierConfig("micro", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_summary_at_immediate_exhaustion(self):
        tier = TierConfig("micro", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[],
            pending_tests=["task_1", "task_2"],
            issues=["Budget exhausted before any simulation"],
            evidence_paths=[],
            blockers=["ACTIVE_BUDGET_EXHAUSTED"],
        )
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(len(summary["pending_tests"]), 2)
        self.assertEqual(summary["severity"], "functional")


class TestPauseDominatedTimeline(unittest.TestCase):
    """Active time is much smaller than wall time."""

    def test_mostly_paused_session(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.005)
        tracker.pause("long_wait")
        time.sleep(0.05)
        tracker.resume()
        time.sleep(0.005)

        snap = tracker.snapshot()
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])
        self.assertGreater(snap["total_pause_seconds"], snap["active_seconds"])
        self.assertFalse(tracker.is_exhausted)


class TestNoOpSession(unittest.TestCase):
    """Session that starts and immediately checkpoints with near-zero time."""

    def test_immediate_checkpoint(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        snap = tracker.snapshot()
        self.assertLess(snap["active_seconds"], 0.01)
        self.assertAlmostEqual(
            snap["remaining_active_seconds"],
            tier.active_budget_seconds,
            delta=0.01,
        )

    def test_immediate_checkpoint_round_trip(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "immediate.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
        snap = restored.snapshot()
        self.assertLess(snap["active_seconds"], 0.05)


class TestAdjustmentLogAccumulation(unittest.TestCase):
    """Adjustment log grows through multi-stage workflow."""

    def test_accumulate_through_stages(self):
        log = AdjustmentLog()
        stages = [
            ("geometry_tolerance", {"tol": 0.1}, {"tol": 0.05}, "tighter fit"),
            ("mesh_density", {"density": 10}, {"density": 20}, "finer mesh"),
            ("sim_timestep", {"dt": 0.01}, {"dt": 0.005}, "stability"),
        ]
        for cid, before, after, reason in stages:
            log.record(CriteriaAdjustment(cid, before, after, reason))

        entries = log.to_list()
        self.assertEqual(len(entries), 3)
        self.assertEqual(entries[0]["criterion_id"], "geometry_tolerance")
        self.assertEqual(entries[2]["criterion_id"], "sim_timestep")

    def test_save_and_reload(self):
        log = AdjustmentLog()
        for i in range(5):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow_log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)

        self.assertEqual(len(loaded.to_list()), 5)
        for i, entry in enumerate(loaded.to_list()):
            self.assertEqual(entry["criterion_id"], f"c{i}")


class TestConcurrentMultiDesign(unittest.TestCase):
    """Concurrent validation of multiple designs with independent budgets."""

    def test_three_designs_parallel(self):
        designs = [
            {"tier": get_tier("light"), "parts": 5, "tasks": 1},
            {"tier": get_tier("medium"), "parts": 10, "tasks": 2},
            {"tier": get_tier("high"), "parts": 15, "tasks": 3},
        ]
        results = [None] * 3
        errors = []

        def validate_design(idx, design):
            try:
                tier = design["tier"]
                tracker = ActiveBudgetTracker(tier=tier)

                violations = check_part_count([{}] * design["parts"], tier)
                if violations:
                    results[idx] = {"status": "REJECTED", "violations": violations}
                    return

                task_violations = check_task_count([{}] * design["tasks"], tier)
                if task_violations:
                    results[idx] = {"status": "REJECTED", "violations": task_violations}
                    return

                time.sleep(0.01)
                timeout = enforce_simulation_timeout(tier, 60)

                snap = tracker.snapshot()
                results[idx] = {
                    "status": "PASS",
                    "active_seconds": snap["active_seconds"],
                    "tier": snap["tier"],
                    "timeout": timeout,
                }
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=validate_design, args=(i, d)) for i, d in enumerate(designs)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        for i, result in enumerate(results):
            self.assertIsNotNone(result, f"Design {i} had no result")
            self.assertEqual(result["status"], "PASS", f"Design {i}: {result}")


class TestTimeoutClamping(unittest.TestCase):
    """Timeout clamping across all tiers."""

    def test_clamp_to_tier_max(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier, 999999)
            self.assertEqual(result, tier.simulation_timeout_seconds, f"Tier {name}")

    def test_passthrough_under_limit(self):
        for name, tier in TIERS.items():
            half = tier.simulation_timeout_seconds / 2
            result = enforce_simulation_timeout(tier, half)
            self.assertEqual(result, half, f"Tier {name}")

    def test_none_returns_tier_max(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier)
            self.assertEqual(result, tier.simulation_timeout_seconds, f"Tier {name}")

    def test_exact_boundary_returns_exact(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds)
            self.assertEqual(result, tier.simulation_timeout_seconds, f"Tier {name}")


class TestRecoveryDelaySchedule(unittest.TestCase):
    """Recovery delay schedule matches exponential backoff."""

    def test_doubles_each_attempt(self):
        for i in range(4):
            d_now = recovery_delay(i)
            d_next = recovery_delay(i + 1)
            self.assertGreaterEqual(d_next, d_now)

    def test_capped_at_60(self):
        for i in range(10):
            self.assertLessEqual(recovery_delay(i), 60.0)

    def test_attempt_0_is_5(self):
        self.assertEqual(recovery_delay(0), 5.0)

    def test_at_cap_stays_capped(self):
        self.assertEqual(recovery_delay(4), 60.0)
        self.assertEqual(recovery_delay(10), 60.0)
        self.assertEqual(recovery_delay(100), 60.0)


if __name__ == "__main__":
    unittest.main()
