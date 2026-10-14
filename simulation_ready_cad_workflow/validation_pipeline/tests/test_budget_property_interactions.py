"""Budget property interaction tests.

Validates that computed properties (elapsed_active, elapsed_wall,
remaining_active, is_exhausted, is_paused, check_budget) interact
correctly across all tracker states, and that build_incomplete_summary
severity classification logic is correct.
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from dataclasses import asdict
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


class TestPropertyCoherenceFresh(unittest.TestCase):
    """Properties on a fresh tracker are mutually consistent."""

    def test_fresh_not_paused(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertFalse(tracker.is_paused, f"Tier {name}")

    def test_fresh_not_exhausted(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertFalse(tracker.is_exhausted, f"Tier {name}")

    def test_fresh_check_budget_true(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertTrue(tracker.check_budget(), f"Tier {name}")

    def test_fresh_remaining_near_budget(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertAlmostEqual(
                tracker.remaining_active,
                tier.active_budget_seconds,
                delta=0.1,
                msg=f"Tier {name}",
            )

    def test_fresh_active_near_zero(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertLess(tracker.elapsed_active, 0.1, f"Tier {name}")

    def test_fresh_wall_near_zero(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertLess(tracker.elapsed_wall, 0.1, f"Tier {name}")


class TestPropertyCoherencePaused(unittest.TestCase):
    """Properties while paused are consistent."""

    def test_paused_is_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        self.assertTrue(tracker.is_paused)

    def test_paused_active_frozen(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("test")
        frozen = tracker.elapsed_active
        time.sleep(0.02)
        self.assertAlmostEqual(tracker.elapsed_active, frozen, delta=0.001)

    def test_paused_wall_advances(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        w1 = tracker.elapsed_wall
        time.sleep(0.02)
        w2 = tracker.elapsed_wall
        self.assertGreater(w2, w1)

    def test_paused_remaining_frozen(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        tracker.pause("test")
        r1 = tracker.remaining_active
        time.sleep(0.02)
        r2 = tracker.remaining_active
        self.assertAlmostEqual(r1, r2, delta=0.001)

    def test_paused_check_budget_still_true(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("test")
        self.assertTrue(tracker.check_budget())

    def test_paused_not_exhausted(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        self.assertFalse(tracker.is_exhausted)


class TestPropertyCoherenceResumed(unittest.TestCase):
    """Properties after resume return to running state."""

    def test_resumed_not_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        tracker.resume()
        self.assertFalse(tracker.is_paused)

    def test_resumed_active_advances(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        tracker.resume()
        a1 = tracker.elapsed_active
        time.sleep(0.02)
        a2 = tracker.elapsed_active
        self.assertGreater(a2, a1)

    def test_resumed_remaining_decreases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("test")
        tracker.resume()
        r1 = tracker.remaining_active
        time.sleep(0.02)
        r2 = tracker.remaining_active
        self.assertLess(r2, r1 + 0.001)


class TestPropertyCoherenceExhausted(unittest.TestCase):
    """Properties when budget is exhausted."""

    def test_exhausted_is_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)

    def test_exhausted_check_budget_false(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertFalse(tracker.check_budget())

    def test_exhausted_remaining_zero(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_exhausted_active_exceeds_budget(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertGreaterEqual(tracker.elapsed_active, tier.active_budget_seconds)

    def test_exhausted_while_paused_stays_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        tracker.pause("too late")
        self.assertTrue(tracker.is_exhausted)
        self.assertFalse(tracker.check_budget())


class TestPauseResumeIdempotency(unittest.TestCase):
    """Double pause/resume are idempotent with no side effects."""

    def test_double_pause_no_extra_count(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("first")
        snap1 = tracker.snapshot()
        tracker.pause("second")
        snap2 = tracker.snapshot()
        self.assertEqual(snap1["pause_count"], snap2["pause_count"])

    def test_double_resume_no_side_effect(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        tracker.resume()
        a1 = tracker.elapsed_active
        tracker.resume()
        a2 = tracker.elapsed_active
        self.assertAlmostEqual(a1, a2, delta=0.005)

    def test_resume_without_pause_noop(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        a1 = tracker.elapsed_active
        tracker.resume()
        a2 = tracker.elapsed_active
        self.assertAlmostEqual(a1, a2, delta=0.005)

    def test_pause_resume_cycle_count(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(10):
            tracker.pause(f"p{i}")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 10)


class TestSeverityClassification(unittest.TestCase):
    """build_incomplete_summary severity logic."""

    def _build(self, completed_tests, pending_tests):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        return build_incomplete_summary(
            tier=tier,
            tracker=tracker,
            completed_tests=completed_tests,
            pending_tests=pending_tests,
            issues=[],
            evidence_paths=[],
            blockers=[],
        )

    def test_no_tests_no_pending_minor(self):
        summary = self._build([], [])
        self.assertEqual(summary["severity"], "minor")

    def test_all_pass_no_pending_minor(self):
        tests = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
            {"test_id": "t2", "physical_pass": True, "geometry_audit": {"passed": True}},
        ]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "minor")

    def test_pending_tests_functional(self):
        tests = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
        ]
        summary = self._build(tests, ["t2"])
        self.assertEqual(summary["severity"], "functional")

    def test_failed_physics_functional(self):
        tests = [
            {"test_id": "t1", "physical_pass": False, "geometry_audit": {"passed": True}},
        ]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "functional")

    def test_failed_geometry_functional(self):
        tests = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": False}},
        ]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "functional")

    def test_both_failures_functional(self):
        tests = [
            {"test_id": "t1", "physical_pass": False, "geometry_audit": {"passed": False}},
        ]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "functional")

    def test_mixed_pass_fail_functional(self):
        tests = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
            {"test_id": "t2", "physical_pass": False, "geometry_audit": {"passed": True}},
        ]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "functional")

    def test_only_pending_functional(self):
        summary = self._build([], ["t1", "t2"])
        self.assertEqual(summary["severity"], "functional")

    def test_missing_geometry_audit_functional(self):
        tests = [{"test_id": "t1", "physical_pass": True}]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "functional")

    def test_empty_geometry_audit_functional(self):
        tests = [{"test_id": "t1", "physical_pass": True, "geometry_audit": {}}]
        summary = self._build(tests, [])
        self.assertEqual(summary["severity"], "functional")


class TestSummaryCompletedTestStatus(unittest.TestCase):
    """Completed test entries in summary get correct PASS/FAIL."""

    def _build(self, tests):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=tests, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )

    def test_passing_test_status(self):
        tests = [{"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}}]
        summary = self._build(tests)
        self.assertEqual(summary["completed_tests"][0]["status"], "PASS")

    def test_failing_physics_status(self):
        tests = [{"test_id": "t1", "physical_pass": False, "geometry_audit": {"passed": True}}]
        summary = self._build(tests)
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_failing_geometry_status(self):
        tests = [{"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": False}}]
        summary = self._build(tests)
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_mixed_statuses(self):
        tests = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
            {"test_id": "t2", "physical_pass": False, "geometry_audit": {"passed": True}},
            {"test_id": "t3", "physical_pass": True, "geometry_audit": {"passed": False}},
        ]
        summary = self._build(tests)
        statuses = [t["status"] for t in summary["completed_tests"]]
        self.assertEqual(statuses, ["PASS", "FAIL", "FAIL"])

    def test_missing_test_id_defaults(self):
        tests = [{"physical_pass": True, "geometry_audit": {"passed": True}}]
        summary = self._build(tests)
        self.assertEqual(summary["completed_tests"][0]["id"], "unknown")


class TestShouldRetrySubstringEdgeCases(unittest.TestCase):
    """should_retry uses substring matching — test edge cases."""

    def test_pass_in_longer_string(self):
        self.assertFalse(should_retry(0, "PASS: all good"))

    def test_pass_as_substring(self):
        self.assertFalse(should_retry(0, "test result: PASS"))

    def test_terminated_in_longer_string(self):
        self.assertFalse(should_retry(0, "Process TERMINATED unexpectedly"))

    def test_exhausted_in_longer_string(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED at 3600s"))

    def test_partial_keyword_no_match(self):
        self.assertTrue(should_retry(0, "PAS"))
        self.assertTrue(should_retry(0, "TERMINAT"))

    def test_case_sensitive(self):
        self.assertTrue(should_retry(0, "pass"))
        self.assertTrue(should_retry(0, "terminated"))
        self.assertTrue(should_retry(0, "active_budget_exhausted"))

    def test_empty_error_retryable(self):
        self.assertTrue(should_retry(0, ""))

    def test_whitespace_error_retryable(self):
        self.assertTrue(should_retry(0, "   "))

    def test_keyword_at_boundary(self):
        self.assertFalse(should_retry(0, "PASS"))
        self.assertFalse(should_retry(0, "TERMINATED"))
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))


class TestRestoreFallbackTier(unittest.TestCase):
    """Restore from checkpoint with unknown tier triggers fallback."""

    def test_restore_unknown_tier_name(self):
        tier = TierConfig("custom_test_tier", 5000, 120, 3, 1, 12)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)

        self.assertEqual(restored.tier.name, "custom_test_tier")
        self.assertEqual(restored.tier.active_budget_seconds, 5000.0)
        self.assertTrue(restored.is_paused)

    def test_restore_with_explicit_tier_overrides_name(self):
        tier_original = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier_original)
        time.sleep(0.01)

        tier_override = get_tier("high")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier_override)

        self.assertEqual(restored.tier.name, "high")
        self.assertEqual(restored.tier.active_budget_seconds, 10800)

    def test_restore_preserves_active_seconds(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        active_before = tracker.elapsed_active

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)

        self.assertAlmostEqual(restored.elapsed_active, active_before, delta=0.01)

    def test_restore_starts_paused(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.005)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)

        self.assertTrue(restored.is_paused)

    def test_restore_increments_pause_count(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("p1")
        tracker.resume()
        tracker.pause("p2")
        tracker.resume()

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)

        self.assertEqual(restored.snapshot()["pause_count"], 3)


class TestCheckBudgetTransition(unittest.TestCase):
    """check_budget() transitions from True to False exactly at exhaustion."""

    def test_true_to_false_at_exhaustion(self):
        tier = TierConfig("short", 0.03, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        self.assertTrue(tracker.check_budget())
        time.sleep(0.05)
        self.assertFalse(tracker.check_budget())

    def test_stays_false_after_pause_resume(self):
        tier = TierConfig("short", 0.01, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        self.assertFalse(tracker.check_budget())
        tracker.pause("late")
        self.assertFalse(tracker.check_budget())
        tracker.resume()
        self.assertFalse(tracker.check_budget())


class TestEnforceTimeoutEdgeCases(unittest.TestCase):
    """enforce_simulation_timeout edge cases."""

    def test_none_uses_tier_limit(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier)
            self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_smaller_than_tier(self):
        tier = get_tier("high")
        result = enforce_simulation_timeout(tier, 10.0)
        self.assertEqual(result, 10.0)

    def test_larger_than_tier_capped(self):
        tier = get_tier("light")
        result = enforce_simulation_timeout(tier, 1000.0)
        self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_exact_tier_limit(self):
        tier = get_tier("medium")
        result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds)
        self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_tiny_positive(self):
        tier = get_tier("light")
        result = enforce_simulation_timeout(tier, 0.001)
        self.assertEqual(result, 0.001)

    def test_zero_raises(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, 0.0)

    def test_negative_raises(self):
        tier = get_tier("high")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -1.0)

    def test_inf_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("inf"))

    def test_nan_raises(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("nan"))


class TestCheckPartCountEdgeCases(unittest.TestCase):
    """check_part_count boundary behavior."""

    def test_zero_parts_ok(self):
        for name, tier in TIERS.items():
            violations = check_part_count([], tier)
            self.assertEqual(violations, [], f"Tier {name}")

    def test_at_max_ok(self):
        for name, tier in TIERS.items():
            violations = check_part_count([{}] * tier.max_parts, tier)
            self.assertEqual(violations, [], f"Tier {name}")

    def test_one_over_max_fails(self):
        for name, tier in TIERS.items():
            violations = check_part_count([{}] * (tier.max_parts + 1), tier)
            self.assertEqual(len(violations), 1, f"Tier {name}")

    def test_way_over_max(self):
        tier = get_tier("light")
        violations = check_part_count([{}] * 100, tier)
        self.assertEqual(len(violations), 1)
        self.assertIn("100", violations[0])
        self.assertIn("10", violations[0])


class TestCheckTaskCountEdgeCases(unittest.TestCase):
    """check_task_count boundary behavior."""

    def test_zero_tasks_ok(self):
        for name, tier in TIERS.items():
            violations = check_task_count([], tier)
            self.assertEqual(violations, [], f"Tier {name}")

    def test_at_max_ok(self):
        for name, tier in TIERS.items():
            violations = check_task_count([{}] * tier.max_tasks_per_design, tier)
            self.assertEqual(violations, [], f"Tier {name}")

    def test_one_over_max_fails(self):
        for name, tier in TIERS.items():
            violations = check_task_count([{}] * (tier.max_tasks_per_design + 1), tier)
            self.assertEqual(len(violations), 1, f"Tier {name}")

    def test_single_task_ok(self):
        for name, tier in TIERS.items():
            violations = check_task_count([{}], tier)
            self.assertEqual(violations, [], f"Tier {name}")


class TestSimulationTimeoutResultProperties(unittest.TestCase):
    """SimulationTimeoutResult property interactions."""

    def test_not_timed_out_status_pass(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=5.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_timed_out_status_timeout(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_with_partial_evidence(self):
        r = SimulationTimeoutResult(
            timed_out=True,
            elapsed_seconds=30.0,
            partial_evidence={"frames": 100, "coverage": 0.5},
        )
        self.assertEqual(r.partial_evidence["frames"], 100)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_with_error_string(self):
        r = SimulationTimeoutResult(
            timed_out=False,
            elapsed_seconds=2.0,
            error="Simulation completed with warnings",
        )
        self.assertEqual(r.status, BudgetStatus.PASS)
        self.assertIn("warnings", r.error)

    def test_with_log_path(self):
        r = SimulationTimeoutResult(
            timed_out=True,
            elapsed_seconds=60.0,
            log_path=Path("/tmp/sim.log"),
        )
        self.assertEqual(r.log_path, Path("/tmp/sim.log"))

    def test_zero_elapsed(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=0.0)
        self.assertEqual(r.status, BudgetStatus.PASS)
        self.assertEqual(r.elapsed_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
