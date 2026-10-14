"""Edge case and stress tests for the budget module.

Covers: budget exhaustion timing edge cases, checkpoint corruption recovery,
boundary conditions on tier limits, concurrent pause operations, rapid
pause/resume cycling, serialization round-trips, and cross-tier comparisons.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.budget import (
    TIERS,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    SimulationTimeoutResult,
    TierConfig,
    build_incomplete_summary,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    get_tier,
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
    BACKOFF_BASE_SECONDS,
)


class TestTierBoundaryValues(unittest.TestCase):
    """Boundary value testing on tier limits."""

    def test_exact_part_limit_accepted(self):
        for name, tier in TIERS.items():
            bodies = [{} for _ in range(tier.max_parts)]
            self.assertEqual(check_part_count(bodies, tier), [],
                             f"Exact limit {tier.max_parts} should be accepted for {name}")

    def test_one_over_part_limit_rejected(self):
        for name, tier in TIERS.items():
            bodies = [{} for _ in range(tier.max_parts + 1)]
            violations = check_part_count(bodies, tier)
            self.assertEqual(len(violations), 1,
                             f"One over limit should produce exactly 1 violation for {name}")

    def test_zero_parts_accepted(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([], tier), [])

    def test_one_part_accepted(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([{}], tier), [])

    def test_exact_task_limit_accepted(self):
        for name, tier in TIERS.items():
            tests = [{} for _ in range(tier.max_tasks_per_design)]
            self.assertEqual(check_task_count(tests, tier), [],
                             f"Exact limit {tier.max_tasks_per_design} for {name}")

    def test_one_over_task_limit_rejected(self):
        for name, tier in TIERS.items():
            tests = [{} for _ in range(tier.max_tasks_per_design + 1)]
            violations = check_task_count(tests, tier)
            self.assertEqual(len(violations), 1)


class TestBudgetExhaustionEdgeCases(unittest.TestCase):
    """Edge cases around budget exhaustion timing."""

    def test_exactly_at_budget_is_exhausted(self):
        tier = TierConfig("exact", 0.05, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.06)
        self.assertTrue(tracker.is_exhausted)

    def test_just_under_budget_is_not_exhausted(self):
        tier = TierConfig("near", 1.0, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        # Should not be exhausted immediately
        self.assertFalse(tracker.is_exhausted)

    def test_remaining_never_negative(self):
        tier = TierConfig("tiny", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.05)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_pause_prevents_exhaustion(self):
        tier = TierConfig("paused", 0.03, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("test")
        time.sleep(0.05)
        self.assertFalse(tracker.is_exhausted)


class TestCheckpointCorruption(unittest.TestCase):
    """Recovery from corrupted or invalid checkpoint files."""

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("{}")
            with self.assertRaises(ValueError):
                ActiveBudgetTracker.restore(path)

    def test_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.json"
            path.write_text('{"schema_version": "budget_checkpoint_v1"}')
            with self.assertRaises((ValueError, KeyError)):
                ActiveBudgetTracker.restore(path)

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json")
            with self.assertRaises(json.JSONDecodeError):
                ActiveBudgetTracker.restore(path)

    def test_nonexistent_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            ActiveBudgetTracker.restore(Path("/nonexistent/checkpoint.json"))


class TestRapidPauseResumeCycling(unittest.TestCase):
    """Stress test: rapid pause/resume cycles must not accumulate errors."""

    def test_100_cycles(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        for _ in range(100):
            tracker.pause("cycle")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 100)
        self.assertGreaterEqual(snap["active_seconds"], 0)
        self.assertGreaterEqual(snap["wall_seconds"], 0)
        self.assertLessEqual(snap["active_seconds"], snap["wall_seconds"])

    def test_active_always_less_than_wall_after_pauses(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        for _ in range(10):
            time.sleep(0.002)
            tracker.pause("p")
            time.sleep(0.005)
            tracker.resume()
        self.assertLess(tracker.elapsed_active, tracker.elapsed_wall)


class TestSnapshotSerializationRoundTrip(unittest.TestCase):
    """Snapshot must serialize to JSON and back correctly."""

    def test_round_trip_fidelity(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("stall")
        time.sleep(0.01)
        tracker.resume()
        snap = tracker.snapshot()

        serialized = json.dumps(snap)
        deserialized = json.loads(serialized)

        self.assertEqual(deserialized["schema_version"], "budget_checkpoint_v1")
        self.assertEqual(deserialized["tier"], "medium")
        self.assertIsInstance(deserialized["active_seconds"], float)
        self.assertIsInstance(deserialized["wall_seconds"], float)
        self.assertEqual(deserialized["pause_count"], 1)

    def test_all_snapshot_values_are_finite(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        snap = tracker.snapshot()
        for key in ("active_seconds", "wall_seconds", "budget_limit_seconds",
                     "remaining_active_seconds", "total_pause_seconds"):
            self.assertTrue(math.isfinite(snap[key]), f"{key} must be finite")


class TestCrossTierComparison(unittest.TestCase):
    """Tier configurations are strictly ordered."""

    def test_budgets_increase_with_tier(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLess(light.active_budget_seconds, medium.active_budget_seconds)
        self.assertLess(medium.active_budget_seconds, high.active_budget_seconds)

    def test_timeouts_increase_with_tier(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLess(light.simulation_timeout_seconds, medium.simulation_timeout_seconds)
        self.assertLess(medium.simulation_timeout_seconds, high.simulation_timeout_seconds)

    def test_part_limits_increase_with_tier(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLessEqual(light.max_parts, medium.max_parts)
        self.assertLessEqual(medium.max_parts, high.max_parts)


class TestRecoveryEdgeCases(unittest.TestCase):
    """Recovery helpers edge cases."""

    def test_backoff_at_zero_attempts(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_backoff_monotonically_increases(self):
        delays = [recovery_delay(i) for i in range(10)]
        for i in range(1, len(delays)):
            self.assertGreaterEqual(delays[i], delays[i - 1])

    def test_should_retry_boundary(self):
        self.assertTrue(should_retry(MAX_RECOVERY_RETRIES - 1, "some_error"))
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "some_error"))

    def test_should_retry_partial_match(self):
        self.assertFalse(should_retry(0, "error during ACTIVE_BUDGET_EXHAUSTED state"))


class TestSimulationTimeoutEdgeCases(unittest.TestCase):
    """Edge cases for simulation timeout enforcement."""

    def test_zero_timeout_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, 0)

    def test_negative_timeout_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -1)

    def test_nan_timeout_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("nan"))

    def test_very_small_timeout(self):
        tier = get_tier("high")
        result = enforce_simulation_timeout(tier, 0.001)
        self.assertEqual(result, 0.001)

    def test_exactly_equal_timeout(self):
        tier = get_tier("light")
        result = enforce_simulation_timeout(tier, 60)
        self.assertEqual(result, 60)


class TestBudgetStatusProperties(unittest.TestCase):
    """Exhaustive status enum checks."""

    def test_all_statuses_have_string_values(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.value, str)
            self.assertTrue(len(status.value) > 0)

    def test_statuses_are_unique(self):
        values = [s.value for s in BudgetStatus]
        self.assertEqual(len(values), len(set(values)))

    def test_terminal_count(self):
        terminal = [s for s in BudgetStatus if s.is_terminal()]
        self.assertEqual(len(terminal), 3)

    def test_non_terminal_count(self):
        non_terminal = [s for s in BudgetStatus if not s.is_terminal()]
        self.assertEqual(len(non_terminal), 5)


class TestAdjustmentLogEdgeCases(unittest.TestCase):
    """Edge cases for criteria adjustment logging."""

    def test_many_adjustments(self):
        log = AdjustmentLog()
        for i in range(100):
            log.record(CriteriaAdjustment(f"c_{i}", {"v": i}, {"v": i + 1}, f"reason {i}"))
        self.assertEqual(len(log.entries), 100)

    def test_round_trip_preserves_unicode(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment(
            "travel_距离",
            {"threshold": "10 毫米"},
            {"threshold": "8 毫米"},
            "降低门槛以匹配几何限制",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adj.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
            self.assertEqual(loaded.entries[0].criterion_id, "travel_距离")
            self.assertEqual(loaded.entries[0].reason, "降低门槛以匹配几何限制")


class TestIncompleteSummaryEdgeCases(unittest.TestCase):
    """Edge cases for incomplete summary generation."""

    def test_all_fields_present(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        required_keys = {"schema_version", "status", "tier", "timing",
                         "completed_tests", "pending_tests", "severity",
                         "issues", "evidence_paths", "blockers", "generated_at"}
        self.assertTrue(required_keys <= set(summary.keys()))

    def test_summary_is_json_serializable(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "x", "physical_pass": True, "geometry_audit": {"passed": True}}],
            pending_tests=["y", "z"],
            issues=["timeout"],
            evidence_paths=["/a/b"],
            blockers=["slow sim"],
        )
        serialized = json.dumps(summary)
        self.assertIsInstance(serialized, str)
        parsed = json.loads(serialized)
        self.assertEqual(parsed["status"], "ACTIVE_BUDGET_EXHAUSTED")


if __name__ == "__main__":
    unittest.main()
