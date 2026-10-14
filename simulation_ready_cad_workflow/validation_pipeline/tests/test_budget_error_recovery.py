"""Budget error recovery and fault tolerance tests.

Tests budget behavior under error conditions that occur in real workflows:
- Budget tracker state after unexpected exceptions
- Checkpoint save during error recovery
- Budget exhaustion detection races
- Recovery delay computation with edge-case attempt counts
- should_retry with compound and partial error strings
- Budget status encoding through exception propagation
- Incomplete summary generation with malformed inputs
- Tracker pause/resume idempotency under error conditions
"""
from __future__ import annotations

import json
import math
import tempfile
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
    MAX_RECOVERY_RETRIES,
    BACKOFF_BASE_SECONDS,
)


class TestTrackerStateAfterException(unittest.TestCase):
    """Tracker state is consistent even when operations are interrupted."""

    def test_pause_after_exception_is_safe(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        try:
            raise RuntimeError("simulated geometry error")
        except RuntimeError:
            tracker.pause("error_recovery")
        self.assertTrue(tracker.is_paused)
        snap = tracker.snapshot()
        self.assertGreater(snap["active_seconds"], 0)
        self.assertEqual(snap["pause_count"], 1)

    def test_resume_after_exception_recovery(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("pre_error")
        try:
            raise ValueError("bad scene format")
        except ValueError:
            pass
        tracker.resume()
        self.assertFalse(tracker.is_paused)
        time.sleep(0.01)
        self.assertGreater(tracker.elapsed_active, 0)

    def test_checkpoint_during_error_recovery(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        try:
            raise IOError("disk full")
        except IOError:
            tracker.pause("disk_error")

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "recovery.json"
            tracker.save_checkpoint(cp)
            self.assertTrue(cp.exists())
            data = json.loads(cp.read_text(encoding="utf-8"))
            self.assertGreater(data["active_seconds"], 0)
            self.assertEqual(data["pause_count"], 1)

    def test_multiple_pause_calls_are_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("first")
        tracker.pause("second")
        tracker.pause("third")
        self.assertEqual(tracker.snapshot()["pause_count"], 1)

    def test_multiple_resume_calls_are_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        tracker.resume()
        tracker.resume()
        tracker.resume()
        self.assertFalse(tracker.is_paused)
        self.assertEqual(tracker.snapshot()["pause_count"], 1)

    def test_pause_resume_pause_resume_sequence(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for i in range(5):
            tracker.pause(f"cycle_{i}")
            time.sleep(0.002)
            tracker.resume()
            time.sleep(0.002)
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 5)
        self.assertGreater(snap["active_seconds"], 0)
        self.assertGreater(snap["total_pause_seconds"], 0)


class TestExhaustionDetectionRaces(unittest.TestCase):
    """Budget exhaustion is detected correctly at timing boundaries."""

    def test_exactly_at_boundary(self):
        """A tracker with budget = active should be exhausted."""
        tiny = TierConfig("boundary", 0.01, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.015)
        self.assertTrue(tracker.is_exhausted)
        self.assertTrue(tracker.elapsed_active >= tiny.active_budget_seconds)

    def test_just_under_boundary(self):
        tracker = ActiveBudgetTracker(tier=TierConfig("big", 999999, 60, 1, 0, 10))
        time.sleep(0.01)
        self.assertFalse(tracker.is_exhausted)
        self.assertGreater(tracker.remaining_active, 0)

    def test_exhaustion_during_pause(self):
        """Budget can be exceeded while active; pausing doesn't undo it."""
        tiny = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.01)
        tracker.pause("late_pause")
        self.assertTrue(tracker.is_exhausted)

    def test_not_exhausted_after_pause_if_budget_remains(self):
        """Pause doesn't cause exhaustion if budget wasn't exceeded."""
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        tracker.pause("check")
        self.assertFalse(tracker.is_exhausted)


class TestRecoveryDelayEdgeCases(unittest.TestCase):
    """Recovery delay with extreme attempt counts."""

    def test_attempt_zero(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_large_attempt_capped(self):
        self.assertEqual(recovery_delay(1000), 60.0)

    def test_delay_is_float(self):
        for i in range(10):
            self.assertIsInstance(recovery_delay(i), float)

    def test_delay_never_infinite(self):
        for i in range(100):
            d = recovery_delay(i)
            self.assertTrue(math.isfinite(d))
            self.assertGreater(d, 0)

    def test_delay_progression_exact(self):
        expected = [5.0, 10.0, 20.0, 40.0, 60.0]
        for i, exp in enumerate(expected):
            self.assertEqual(recovery_delay(i), exp)


class TestShouldRetryCompoundErrors(unittest.TestCase):
    """should_retry with compound error strings and partial matches."""

    def test_error_containing_pass_keyword(self):
        self.assertFalse(should_retry(0, "Result: PASS"))

    def test_error_containing_exhausted(self):
        self.assertFalse(should_retry(0, "Error: ACTIVE_BUDGET_EXHAUSTED after 3600s"))

    def test_error_containing_terminated(self):
        self.assertFalse(should_retry(0, "Process TERMINATED by user"))

    def test_error_with_only_lowercase_keywords(self):
        self.assertTrue(should_retry(0, "result: pass"))

    def test_generic_infrastructure_error(self):
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE: memory pressure"))

    def test_empty_error_string(self):
        self.assertTrue(should_retry(0, ""))

    def test_whitespace_only_error(self):
        self.assertTrue(should_retry(0, "   "))

    def test_at_max_retries_always_false(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "generic error"))

    def test_just_under_max_retries(self):
        self.assertTrue(should_retry(MAX_RECOVERY_RETRIES - 1, "generic error"))

    def test_multiple_non_retryable_in_one_string(self):
        self.assertFalse(should_retry(0, "PASS then TERMINATED then ACTIVE_BUDGET_EXHAUSTED"))


class TestIncompleteSummaryWithMalformedInputs(unittest.TestCase):
    """build_incomplete_summary handles edge cases in input data."""

    def _make_tracker(self, tier_name="light"):
        return ActiveBudgetTracker(tier=get_tier(tier_name))

    def test_empty_completed_tests(self):
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._make_tracker(),
            completed_tests=[], pending_tests=["t1"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(len(summary["completed_tests"]), 0)
        self.assertEqual(summary["severity"], "functional")

    def test_empty_pending_tests(self):
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._make_tracker(),
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "minor")

    def test_missing_physical_pass_key(self):
        summary = build_incomplete_summary(
            tier=get_tier("medium"), tracker=self._make_tracker("medium"),
            completed_tests=[{"test_id": "t1"}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_missing_geometry_audit_key(self):
        summary = build_incomplete_summary(
            tier=get_tier("medium"), tracker=self._make_tracker("medium"),
            completed_tests=[{"test_id": "t1", "physical_pass": True}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_missing_test_id_defaults_to_unknown(self):
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._make_tracker(),
            completed_tests=[{"physical_pass": True, "geometry_audit": {"passed": True}}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["completed_tests"][0]["id"], "unknown")

    def test_large_issues_list(self):
        issues = [f"issue_{i}" for i in range(1000)]
        summary = build_incomplete_summary(
            tier=get_tier("high"), tracker=self._make_tracker("high"),
            completed_tests=[], pending_tests=[],
            issues=issues, evidence_paths=[], blockers=[],
        )
        self.assertEqual(len(summary["issues"]), 1000)

    def test_unicode_in_all_fields(self):
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._make_tracker(),
            completed_tests=[{"test_id": "测试_1", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=["待定_2"],
            issues=["超时"],
            evidence_paths=["/路径/证据.json"],
            blockers=["阻塞"],
        )
        encoded = json.dumps(summary, ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertIn("测试_1", decoded["completed_tests"][0]["id"])
        self.assertIn("待定_2", decoded["pending_tests"])


class TestSimulationTimeoutResultEdgeCases(unittest.TestCase):
    """SimulationTimeoutResult edge cases."""

    def test_zero_elapsed(self):
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=0)
        self.assertEqual(result.status, BudgetStatus.PASS)

    def test_negative_elapsed_doesnt_crash(self):
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=-1)
        self.assertEqual(result.status, BudgetStatus.PASS)

    def test_very_large_elapsed(self):
        result = SimulationTimeoutResult(timed_out=True, elapsed_seconds=1e9)
        self.assertEqual(result.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_partial_evidence_none_by_default(self):
        result = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60)
        self.assertIsNone(result.partial_evidence)
        self.assertIsNone(result.log_path)
        self.assertIsNone(result.error)

    def test_all_optional_fields_set(self):
        result = SimulationTimeoutResult(
            timed_out=True,
            elapsed_seconds=120,
            partial_evidence={"steps": 100, "time_reached": 1.5},
            log_path=Path("/tmp/sim.log"),
            error="Process killed after timeout",
        )
        self.assertEqual(result.partial_evidence["steps"], 100)
        self.assertEqual(result.log_path, Path("/tmp/sim.log"))
        self.assertIn("timeout", result.error)


class TestEnforceSimulationTimeoutEdgeCases(unittest.TestCase):
    """enforce_simulation_timeout with edge-case values."""

    def test_actual_equals_tier_limit(self):
        tier = get_tier("light")
        self.assertEqual(
            enforce_simulation_timeout(tier, tier.simulation_timeout_seconds),
            tier.simulation_timeout_seconds,
        )

    def test_actual_slightly_below_limit(self):
        tier = get_tier("medium")
        result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds - 0.001)
        self.assertAlmostEqual(result, tier.simulation_timeout_seconds - 0.001)

    def test_actual_slightly_above_limit(self):
        tier = get_tier("high")
        result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds + 0.001)
        self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_very_small_actual(self):
        tier = get_tier("light")
        result = enforce_simulation_timeout(tier, 0.001)
        self.assertEqual(result, 0.001)

    def test_infinity_rejected(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("light"), float("inf"))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("light"), float("nan"))


class TestCheckLimitsEdgeCases(unittest.TestCase):
    """Limit check functions with edge-case inputs."""

    def test_part_count_with_none_elements(self):
        """Bodies can be any objects; limit checks only count."""
        bodies = [None, None, None]
        tier = get_tier("light")
        self.assertEqual(check_part_count(bodies, tier), [])

    def test_task_count_with_string_elements(self):
        tasks = ["a", "b"]
        tier = get_tier("light")
        self.assertEqual(check_task_count(tasks, tier), [])

    def test_very_large_count(self):
        bodies = [None] * 10000
        tier = get_tier("high")
        violations = check_part_count(bodies, tier)
        self.assertGreater(len(violations), 0)
        self.assertIn("10000", violations[0])

    def test_custom_tier_with_max_1(self):
        tier = TierConfig("strict", 3600, 60, 1, 0, 1)
        self.assertEqual(check_part_count([{}], tier), [])
        self.assertGreater(len(check_part_count([{}, {}], tier)), 0)
        self.assertEqual(check_task_count([{}], tier), [])
        self.assertGreater(len(check_task_count([{}, {}], tier)), 0)


class TestAdjustmentValidation(unittest.TestCase):
    """CriteriaAdjustment validation edge cases."""

    def test_nested_dict_change_accepted(self):
        adj = CriteriaAdjustment(
            "c1",
            {"threshold": {"min": 0.1, "max": 0.5}},
            {"threshold": {"min": 0.1, "max": 0.8}},
            "Relaxed upper bound",
        )
        self.assertEqual(adj.after["threshold"]["max"], 0.8)

    def test_type_change_accepted(self):
        adj = CriteriaAdjustment(
            "c1",
            {"value": 10},
            {"value": "ten"},
            "Changed to string representation",
        )
        self.assertEqual(adj.after["value"], "ten")

    def test_custom_timestamp_preserved(self):
        adj = CriteriaAdjustment(
            "c1", {"a": 1}, {"a": 2}, "reason",
            adjusted_at="2026-01-01T00:00:00Z",
        )
        self.assertEqual(adj.adjusted_at, "2026-01-01T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
