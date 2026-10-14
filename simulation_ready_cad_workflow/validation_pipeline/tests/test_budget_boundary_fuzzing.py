"""Budget boundary and fuzzing tests.

Tests budget behavior at extreme boundary conditions and with fuzzed inputs:
- Tier config creation with boundary values (epsilon, very large, zero-adjacent)
- Tracker behavior at exact budget exhaustion boundaries
- Checkpoint round-trips with very large/small timing values
- Summary generation with adversarial inputs
- Status transitions at every BudgetStatus boundary
- Recovery delay with extreme attempt counts
- Concurrent pause/resume at budget boundary
- Incomplete summary with mixed pass/fail/missing patterns
"""
from __future__ import annotations

import json
import math
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


class TestTierConfigBoundaries(unittest.TestCase):
    """Tier creation with boundary values."""

    def test_minimal_valid_config(self):
        tier = TierConfig("min", 0.001, 0.001, 1, 0, 1)
        self.assertEqual(tier.name, "min")
        self.assertAlmostEqual(tier.active_budget_seconds, 0.001)

    def test_very_large_budget(self):
        tier = TierConfig("huge", 1e8, 1e6, 100, 0, 10000)
        self.assertEqual(tier.max_parts, 10000)

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("zero", 0, 60, 1, 0, 10)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("neg", -1, 60, 1, 0, 10)

    def test_infinity_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("inf", float("inf"), 60, 1, 0, 10)

    def test_nan_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("nan", float("nan"), 60, 1, 0, 10)

    def test_zero_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("zt", 3600, 0, 1, 0, 10)

    def test_negative_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("nt", 3600, -0.001, 1, 0, 10)

    def test_zero_tasks_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("zk", 3600, 60, 0, 0, 10)

    def test_min_exceeds_max_parts_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 11, 10)

    def test_negative_min_parts_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, -1, 10)

    def test_zero_max_parts_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 0, 0)

    def test_equal_min_max_parts_accepted(self):
        tier = TierConfig("eq", 3600, 60, 1, 5, 5)
        self.assertEqual(tier.min_parts, 5)
        self.assertEqual(tier.max_parts, 5)


class TestTrackerExhaustionBoundary(unittest.TestCase):
    """Precise exhaustion detection at the active budget edge."""

    def test_just_below_budget_not_exhausted(self):
        tier = TierConfig("edge", 0.1, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.05)
        self.assertFalse(tracker.is_exhausted)
        self.assertGreater(tracker.remaining_active, 0)

    def test_well_past_budget_exhausted(self):
        tier = TierConfig("edge", 0.01, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_exhaustion_flag_is_monotonic(self):
        """Once exhausted, tracker stays exhausted."""
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)
        tracker.pause("try_pause")
        self.assertTrue(tracker.is_exhausted)
        tracker.resume()
        self.assertTrue(tracker.is_exhausted)

    def test_remaining_never_negative(self):
        tier = TierConfig("tiny", 0.001, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertEqual(tracker.remaining_active, 0.0)
        self.assertGreaterEqual(tracker.remaining_active, 0.0)

    def test_elapsed_always_growing_while_active(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        prev = tracker.elapsed_active
        for _ in range(5):
            time.sleep(0.005)
            curr = tracker.elapsed_active
            self.assertGreaterEqual(curr, prev)
            prev = curr

    def test_elapsed_frozen_while_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        tracker.pause("frozen_check")
        frozen = tracker.elapsed_active
        time.sleep(0.02)
        self.assertAlmostEqual(tracker.elapsed_active, frozen, delta=0.002)


class TestCheckpointBoundaryRoundTrips(unittest.TestCase):
    """Checkpoint save/restore with extreme timing values."""

    def test_immediate_checkpoint_preserves_near_zero_active(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            self.assertGreaterEqual(data["active_seconds"], 0)
            self.assertLess(data["active_seconds"], 1.0)

    def test_checkpoint_after_long_pause(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("long_pause")
        time.sleep(0.05)

        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            self.assertTrue(data["paused"])
            self.assertGreater(data["total_pause_seconds"], 0.04)
            self.assertAlmostEqual(data["active_seconds"], 0.01, delta=0.015)

    def test_restore_from_freshly_created_tracker(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
            self.assertFalse(restored.is_exhausted)
            self.assertAlmostEqual(
                restored.remaining_active,
                tier.active_budget_seconds,
                delta=0.5,
            )

    def test_checkpoint_all_fields_present(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("p1")
        tracker.resume()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            required = {
                "schema_version", "tier", "active_seconds", "wall_seconds",
                "paused", "pause_count", "total_pause_seconds",
                "budget_limit_seconds", "remaining_active_seconds",
                "started_at", "checkpoint_at",
            }
            self.assertTrue(required.issubset(data.keys()), f"Missing: {required - data.keys()}")


class TestSnapshotBoundaryValues(unittest.TestCase):
    """Snapshot accuracy at timing boundaries."""

    def test_snapshot_active_leq_wall(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("p")
        time.sleep(0.02)
        tracker.resume()
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertLessEqual(snap["active_seconds"], snap["wall_seconds"] + 0.01)

    def test_snapshot_remaining_plus_active_equals_budget(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.02)
        snap = tracker.snapshot()
        total = snap["active_seconds"] + snap["remaining_active_seconds"]
        self.assertAlmostEqual(total, snap["budget_limit_seconds"], delta=0.05)

    def test_snapshot_pause_count_matches_cycles(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for i in range(7):
            tracker.pause(f"p{i}")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 7)

    def test_snapshot_tier_matches(self):
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            snap = tracker.snapshot()
            self.assertEqual(snap["tier"], name)

    def test_snapshot_schema_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertEqual(snap["schema_version"], "budget_checkpoint_v1")


class TestSummaryAdversarialInputs(unittest.TestCase):
    """build_incomplete_summary with adversarial data."""

    def _tracker(self, tier="light"):
        return ActiveBudgetTracker(tier=get_tier(tier))

    def test_very_long_test_id(self):
        test_id = "x" * 10000
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._tracker(),
            completed_tests=[{"test_id": test_id, "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["completed_tests"][0]["id"], test_id)

    def test_empty_string_fields(self):
        summary = build_incomplete_summary(
            tier=get_tier("medium"), tracker=self._tracker("medium"),
            completed_tests=[{"test_id": "", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=[""], issues=[""], evidence_paths=[""], blockers=[""],
        )
        self.assertEqual(len(summary["completed_tests"]), 1)

    def test_null_physical_pass_treated_as_fail(self):
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._tracker(),
            completed_tests=[{"test_id": "t1", "physical_pass": None,
                              "geometry_audit": {"passed": True}}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_mixed_pass_fail_severity(self):
        tests = [
            {"test_id": "pass1", "physical_pass": True, "geometry_audit": {"passed": True}},
            {"test_id": "fail1", "physical_pass": False, "geometry_audit": {"passed": True}},
            {"test_id": "pass2", "physical_pass": True, "geometry_audit": {"passed": False}},
        ]
        summary = build_incomplete_summary(
            tier=get_tier("medium"), tracker=self._tracker("medium"),
            completed_tests=tests, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_all_passing_no_pending_is_minor(self):
        tests = [
            {"test_id": f"t{i}", "physical_pass": True, "geometry_audit": {"passed": True}}
            for i in range(10)
        ]
        summary = build_incomplete_summary(
            tier=get_tier("high"), tracker=self._tracker("high"),
            completed_tests=tests, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "minor")

    def test_all_passing_with_pending_is_functional(self):
        tests = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
        ]
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._tracker(),
            completed_tests=tests, pending_tests=["pending1"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_summary_schema_version(self):
        summary = build_incomplete_summary(
            tier=get_tier("medium"), tracker=self._tracker("medium"),
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_summary_tier_info_present(self):
        summary = build_incomplete_summary(
            tier=get_tier("high"), tracker=self._tracker("high"),
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["tier"]["name"], "high")
        self.assertEqual(summary["tier"]["active_budget_seconds"], 10800)
        self.assertEqual(summary["tier"]["simulation_timeout_seconds"], 300)

    def test_summary_json_serializable(self):
        summary = build_incomplete_summary(
            tier=get_tier("light"), tracker=self._tracker(),
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=["p1"], issues=["i1"],
            evidence_paths=["/p/1.json"], blockers=["b1"],
        )
        encoded = json.dumps(summary)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["schema_version"], "budget_incomplete_summary_v1")


class TestBudgetStatusTransitions(unittest.TestCase):
    """BudgetStatus enum properties at every value."""

    def test_all_statuses_are_strings(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.value, str)

    def test_pass_is_terminal(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_active_budget_exhausted_is_terminal(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())

    def test_infrastructure_failure_is_terminal(self):
        self.assertTrue(BudgetStatus.INFRASTRUCTURE_FAILURE.is_terminal())

    def test_simulation_timeout_not_terminal(self):
        self.assertFalse(BudgetStatus.SIMULATION_TIMEOUT.is_terminal())

    def test_needs_revision_not_terminal(self):
        self.assertFalse(BudgetStatus.NEEDS_REVISION.is_terminal())

    def test_harness_error_not_terminal(self):
        self.assertFalse(BudgetStatus.HARNESS_ERROR.is_terminal())

    def test_terminal_set_is_exactly_three(self):
        terminal = [s for s in BudgetStatus if s.is_terminal()]
        self.assertEqual(len(terminal), 3)

    def test_simulation_timeout_result_pass(self):
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=10)
        self.assertEqual(result.status, BudgetStatus.PASS)

    def test_simulation_timeout_result_timeout(self):
        result = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60)
        self.assertEqual(result.status, BudgetStatus.SIMULATION_TIMEOUT)


class TestRecoveryDelayBoundaries(unittest.TestCase):
    """Recovery delay at extreme values."""

    def test_attempt_0_is_base(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_attempt_1_is_double(self):
        self.assertEqual(recovery_delay(1), BACKOFF_BASE_SECONDS * 2)

    def test_cap_at_60(self):
        for attempt in [4, 5, 10, 50, 100, 1000]:
            self.assertEqual(recovery_delay(attempt), 60.0)

    def test_delay_progression(self):
        delays = [recovery_delay(i) for i in range(10)]
        for i in range(1, len(delays)):
            self.assertGreaterEqual(delays[i], delays[i-1])

    def test_all_delays_positive(self):
        for i in range(100):
            self.assertGreater(recovery_delay(i), 0)

    def test_all_delays_finite(self):
        for i in range(100):
            self.assertTrue(math.isfinite(recovery_delay(i)))


class TestShouldRetryBoundaries(unittest.TestCase):
    """should_retry at retry count boundaries."""

    def test_attempt_0_generic_is_retryable(self):
        self.assertTrue(should_retry(0, "generic error"))

    def test_at_max_minus_1_generic_is_retryable(self):
        self.assertTrue(should_retry(MAX_RECOVERY_RETRIES - 1, "generic"))

    def test_at_max_always_false(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "generic"))

    def test_beyond_max_always_false(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES + 100, "generic"))

    def test_pass_at_attempt_0_not_retryable(self):
        self.assertFalse(should_retry(0, "PASS"))

    def test_exhausted_at_attempt_0_not_retryable(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_terminated_at_attempt_0_not_retryable(self):
        self.assertFalse(should_retry(0, "TERMINATED"))

    def test_infrastructure_failure_is_retryable(self):
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE"))

    def test_case_sensitivity_lowercase_pass_retryable(self):
        self.assertTrue(should_retry(0, "pass"))

    def test_embedded_keyword_in_longer_string(self):
        self.assertFalse(should_retry(0, "status: PASS - all tests complete"))


class TestEnforceTimeoutBoundaries(unittest.TestCase):
    """enforce_simulation_timeout at boundary values."""

    def test_zero_actual_rejected(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, 0)

    def test_negative_actual_rejected(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -5)

    def test_at_limit_returns_limit(self):
        tier = get_tier("high")
        self.assertEqual(
            enforce_simulation_timeout(tier, tier.simulation_timeout_seconds),
            tier.simulation_timeout_seconds,
        )

    def test_over_limit_clamped(self):
        tier = get_tier("light")
        self.assertEqual(
            enforce_simulation_timeout(tier, tier.simulation_timeout_seconds + 100),
            tier.simulation_timeout_seconds,
        )

    def test_inf_rejected(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("medium"), float("inf"))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("high"), float("nan"))

    def test_all_tiers_clamp_correctly(self):
        for name, tier in TIERS.items():
            clamped = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds * 2)
            self.assertEqual(clamped, tier.simulation_timeout_seconds, f"Tier {name} failed")


class TestLimitCheckBoundaries(unittest.TestCase):
    """Part and task count limits at boundary values."""

    def test_empty_list_always_passes(self):
        for name in TIERS:
            tier = get_tier(name)
            self.assertEqual(check_part_count([], tier), [])
            self.assertEqual(check_task_count([], tier), [])

    def test_at_max_passes(self):
        tier = get_tier("light")
        bodies = [{}] * tier.max_parts
        self.assertEqual(check_part_count(bodies, tier), [])

    def test_one_over_max_fails(self):
        tier = get_tier("light")
        bodies = [{}] * (tier.max_parts + 1)
        violations = check_part_count(bodies, tier)
        self.assertGreater(len(violations), 0)

    def test_tasks_at_max_passes(self):
        tier = get_tier("high")
        tasks = [{}] * tier.max_tasks_per_design
        self.assertEqual(check_task_count(tasks, tier), [])

    def test_tasks_one_over_max_fails(self):
        tier = get_tier("high")
        tasks = [{}] * (tier.max_tasks_per_design + 1)
        violations = check_task_count(tasks, tier)
        self.assertGreater(len(violations), 0)

    def test_violation_message_contains_counts(self):
        tier = get_tier("light")
        bodies = [{}] * 15
        violations = check_part_count(bodies, tier)
        msg = violations[0]
        self.assertIn("15", msg)
        self.assertIn(str(tier.max_parts), msg)


class TestConcurrentBoundaryAccess(unittest.TestCase):
    """Concurrent operations near budget exhaustion."""

    def test_concurrent_reads_near_exhaustion(self):
        tier = TierConfig("near", 0.02, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.015)
        errors = []

        def reader():
            try:
                for _ in range(50):
                    _ = tracker.is_exhausted
                    _ = tracker.remaining_active
                    _ = tracker.elapsed_active
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(errors, [])

    def test_pause_during_exhaustion(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)
        tracker.pause("post_exhaust")
        self.assertTrue(tracker.is_exhausted)
        snap = tracker.snapshot()
        self.assertEqual(snap["remaining_active_seconds"], 0.0)


class TestGetTierBoundaries(unittest.TestCase):
    """get_tier with boundary inputs."""

    def test_all_known_tiers(self):
        for name in TIERS:
            tier = get_tier(name)
            self.assertEqual(tier.name, name)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            get_tier("nonexistent")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            get_tier("")

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_tier("Light")

    def test_error_message_lists_available(self):
        try:
            get_tier("bad")
            self.fail("Should have raised ValueError")
        except ValueError as e:
            msg = str(e)
            for name in TIERS:
                self.assertIn(name, msg)


class TestAdjustmentLogBoundaries(unittest.TestCase):
    """AdjustmentLog with boundary inputs."""

    def test_empty_log(self):
        log = AdjustmentLog()
        self.assertEqual(log.to_list(), [])

    def test_single_entry(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"v": 1}, {"v": 2}, "test"))
        entries = log.to_list()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["criterion_id"], "c1")

    def test_100_entries(self):
        log = AdjustmentLog()
        for i in range(100):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))
        self.assertEqual(len(log.to_list()), 100)

    def test_entry_has_timestamp(self):
        adj = CriteriaAdjustment("c1", {"before": 0}, {"after": 1}, "reason")
        self.assertIsNotNone(adj.adjusted_at)
        self.assertIsInstance(adj.adjusted_at, str)

    def test_custom_timestamp_preserved(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "reason", adjusted_at="custom-time")
        self.assertEqual(adj.adjusted_at, "custom-time")

    def test_log_entries_are_dicts(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "change"))
        for entry in log.to_list():
            self.assertIsInstance(entry, dict)
            self.assertIn("criterion_id", entry)
            self.assertIn("before", entry)
            self.assertIn("after", entry)
            self.assertIn("reason", entry)


if __name__ == "__main__":
    unittest.main()
