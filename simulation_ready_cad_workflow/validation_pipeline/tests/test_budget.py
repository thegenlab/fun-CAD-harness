"""Focused regression tests for the budget/orchestration module.

Covers: tier configuration contract, active budget accounting with pause/resume,
simulation timeout enforcement, task/part count limits, checkpoint recovery,
and incomplete summary generation.
"""
from __future__ import annotations

import json
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
)


class TestTierContract(unittest.TestCase):
    """The three tiers must have the exact values specified in the contract."""

    def test_light_tier_values(self):
        t = get_tier("light")
        self.assertEqual(t.name, "light")
        self.assertEqual(t.active_budget_seconds, 3600)
        self.assertEqual(t.simulation_timeout_seconds, 60)
        self.assertEqual(t.max_tasks_per_design, 2)
        self.assertEqual(t.max_parts, 10)

    def test_medium_tier_values(self):
        t = get_tier("medium")
        self.assertEqual(t.name, "medium")
        self.assertEqual(t.active_budget_seconds, 7200)
        self.assertEqual(t.simulation_timeout_seconds, 180)
        self.assertEqual(t.max_tasks_per_design, 2)
        self.assertEqual(t.max_parts, 15)

    def test_high_tier_values(self):
        t = get_tier("high")
        self.assertEqual(t.name, "high")
        self.assertEqual(t.active_budget_seconds, 10800)
        self.assertEqual(t.simulation_timeout_seconds, 300)
        self.assertEqual(t.max_tasks_per_design, 5)
        self.assertEqual(t.max_parts, 20)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            get_tier("extreme")

    def test_tier_is_frozen(self):
        t = get_tier("light")
        with self.assertRaises(AttributeError):
            t.active_budget_seconds = 9999

    def test_invalid_tier_construction(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -1, 60, 2, 0, 10)
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 0, 2, 0, 10)
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 0, 0, 10)
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, 5, 3)

    def test_all_tiers_present(self):
        self.assertEqual(set(TIERS.keys()), {"light", "medium", "high"})

    def test_tiers_serializable(self):
        for name, tier in TIERS.items():
            data = asdict(tier)
            self.assertEqual(data["name"], name)
            self.assertIsInstance(json.dumps(data), str)


class TestActiveBudgetTracker(unittest.TestCase):
    """Active budget tracking with pause/resume."""

    def _make_tracker(self, budget=100):
        tier = TierConfig("test", budget, 60, 2, 0, 10)
        return ActiveBudgetTracker(tier=tier)

    def test_initial_state(self):
        tracker = self._make_tracker()
        self.assertFalse(tracker.is_paused)
        self.assertFalse(tracker.is_exhausted)
        self.assertAlmostEqual(tracker.elapsed_active, 0, delta=0.1)
        self.assertAlmostEqual(tracker.elapsed_wall, 0, delta=0.1)

    def test_active_time_advances_when_not_paused(self):
        tracker = self._make_tracker()
        time.sleep(0.05)
        self.assertGreater(tracker.elapsed_active, 0)
        self.assertGreater(tracker.elapsed_wall, 0)

    def test_pause_stops_active_time(self):
        tracker = self._make_tracker()
        time.sleep(0.02)
        tracker.pause("test")
        active_at_pause = tracker.elapsed_active
        self.assertTrue(tracker.is_paused)
        time.sleep(0.05)
        self.assertAlmostEqual(tracker.elapsed_active, active_at_pause, delta=0.01)
        self.assertGreater(tracker.elapsed_wall, active_at_pause + 0.04)

    def test_resume_restarts_active_time(self):
        tracker = self._make_tracker()
        tracker.pause("test")
        time.sleep(0.02)
        active_at_pause = tracker.elapsed_active
        tracker.resume()
        self.assertFalse(tracker.is_paused)
        time.sleep(0.02)
        self.assertGreater(tracker.elapsed_active, active_at_pause)

    def test_double_pause_is_idempotent(self):
        tracker = self._make_tracker()
        tracker.pause("first")
        active = tracker.elapsed_active
        tracker.pause("second")
        self.assertAlmostEqual(tracker.elapsed_active, active, delta=0.01)

    def test_double_resume_is_idempotent(self):
        tracker = self._make_tracker()
        tracker.resume()
        self.assertFalse(tracker.is_paused)

    def test_exhaustion(self):
        tier = TierConfig("tiny", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        self.assertFalse(tracker.check_budget())

    def test_remaining_active(self):
        tracker = self._make_tracker(budget=1.0)
        self.assertAlmostEqual(tracker.remaining_active, 1.0, delta=0.1)
        time.sleep(0.05)
        self.assertLess(tracker.remaining_active, 1.0)

    def test_snapshot_schema(self):
        tracker = self._make_tracker()
        snap = tracker.snapshot()
        self.assertEqual(snap["schema_version"], "budget_checkpoint_v1")
        self.assertIn("active_seconds", snap)
        self.assertIn("wall_seconds", snap)
        self.assertIn("paused", snap)
        self.assertIn("budget_limit_seconds", snap)
        self.assertIn("remaining_active_seconds", snap)
        self.assertIn("started_at", snap)
        self.assertIn("checkpoint_at", snap)


class TestCheckpointRecovery(unittest.TestCase):
    """Save and restore budget tracker from checkpoint."""

    def test_save_and_restore(self):
        tier = TierConfig("test", 100, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        tracker.pause("stall")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            tracker.save_checkpoint(path)
            self.assertTrue(path.exists())

            restored = ActiveBudgetTracker.restore(path)
            self.assertTrue(restored.is_paused)
            self.assertAlmostEqual(restored.elapsed_active, tracker.elapsed_active, delta=0.01)
            self.assertEqual(restored.tier.name, "test")

    def test_restore_pauses_by_default(self):
        tier = TierConfig("test", 100, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.json"
            tracker.save_checkpoint(path)
            restored = ActiveBudgetTracker.restore(path)
            self.assertTrue(restored.is_paused)
            active = restored.elapsed_active
            time.sleep(0.02)
            self.assertAlmostEqual(restored.elapsed_active, active, delta=0.01)

    def test_restore_invalid_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema_version": "wrong"}')
            with self.assertRaises(ValueError):
                ActiveBudgetTracker.restore(path)


class TestSimulationTimeout(unittest.TestCase):
    """Simulation timeout enforcement per tier."""

    def test_tier_limit_applied(self):
        tier = get_tier("light")
        self.assertEqual(enforce_simulation_timeout(tier), 60)

    def test_smaller_timeout_wins(self):
        tier = get_tier("high")
        self.assertEqual(enforce_simulation_timeout(tier, 120), 120)

    def test_larger_timeout_clamped(self):
        tier = get_tier("light")
        self.assertEqual(enforce_simulation_timeout(tier, 300), 60)

    def test_invalid_timeout_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -1)
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("inf"))

    def test_timeout_result_status(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)
        r2 = SimulationTimeoutResult(timed_out=False, elapsed_seconds=30.0)
        self.assertEqual(r2.status, BudgetStatus.PASS)


class TestTaskCountLimit(unittest.TestCase):
    """Task count validation against tier limit."""

    def test_within_limit(self):
        tier = get_tier("light")
        violations = check_task_count([{"id": "a"}, {"id": "b"}], tier)
        self.assertEqual(violations, [])

    def test_over_limit(self):
        tier = get_tier("light")
        violations = check_task_count([{"id": "a"}, {"id": "b"}, {"id": "c"}], tier)
        self.assertEqual(len(violations), 1)
        self.assertIn("3 tests", violations[0])
        self.assertIn("at most 2", violations[0])

    def test_high_tier_allows_five(self):
        tier = get_tier("high")
        violations = check_task_count([{} for _ in range(5)], tier)
        self.assertEqual(violations, [])


class TestPartCountLimit(unittest.TestCase):
    """Part count validation against tier limit."""

    def test_within_limit(self):
        tier = get_tier("light")
        violations = check_part_count([{} for _ in range(10)], tier)
        self.assertEqual(violations, [])

    def test_over_limit(self):
        tier = get_tier("light")
        violations = check_part_count([{} for _ in range(11)], tier)
        self.assertEqual(len(violations), 1)
        self.assertIn("11 parts", violations[0])

    def test_medium_tier_limit(self):
        tier = get_tier("medium")
        self.assertEqual(check_part_count([{} for _ in range(15)], tier), [])
        self.assertNotEqual(check_part_count([{} for _ in range(16)], tier), [])


class TestIncompleteSummary(unittest.TestCase):
    """Summary generation when budget is exhausted."""

    def test_summary_schema(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier,
            tracker=tracker,
            completed_tests=[
                {"test_id": "a", "physical_pass": True, "geometry_audit": {"passed": True}},
            ],
            pending_tests=["b"],
            issues=["ran out of time"],
            evidence_paths=["/path/to/evidence.png"],
            blockers=["simulation too slow"],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(summary["tier"]["name"], "light")
        self.assertIn("timing", summary)
        self.assertEqual(len(summary["completed_tests"]), 1)
        self.assertEqual(summary["completed_tests"][0]["status"], "PASS")
        self.assertEqual(summary["pending_tests"], ["b"])
        self.assertEqual(summary["severity"], "functional")

    def test_severity_minor_when_all_passed(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "a", "physical_pass": True, "geometry_audit": {"passed": True}}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "minor")


class TestBudgetStatus(unittest.TestCase):
    """Status vocabulary and terminal checks."""

    def test_terminal_statuses(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())
        self.assertTrue(BudgetStatus.INFRASTRUCTURE_FAILURE.is_terminal())

    def test_non_terminal_statuses(self):
        self.assertFalse(BudgetStatus.NEEDS_REVISION.is_terminal())
        self.assertFalse(BudgetStatus.SIMULATION_TIMEOUT.is_terminal())
        self.assertFalse(BudgetStatus.HARNESS_ERROR.is_terminal())

    def test_status_values_are_strings(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.value, str)


class TestRecovery(unittest.TestCase):
    """Recovery helpers: retry logic and backoff."""

    def test_backoff_increases(self):
        d1 = recovery_delay(0)
        d2 = recovery_delay(1)
        d3 = recovery_delay(2)
        self.assertLess(d1, d2)
        self.assertLess(d2, d3)
        self.assertLessEqual(d3, 60.0)

    def test_backoff_capped_at_60(self):
        self.assertEqual(recovery_delay(100), 60.0)

    def test_should_retry_within_limit(self):
        self.assertTrue(should_retry(0, "connection_timeout"))
        self.assertTrue(should_retry(2, "memory_stall"))

    def test_should_not_retry_past_limit(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "any_error"))

    def test_should_not_retry_terminal(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))
        self.assertFalse(should_retry(0, "TERMINATED"))
        self.assertFalse(should_retry(0, "PASS"))


class TestCriteriaAdjustment(unittest.TestCase):
    """Adjustable criteria tracking and frozen-criteria protection."""

    def test_frozen_criterion_cannot_be_adjusted(self):
        with self.assertRaises(ValueError) as ctx:
            CriteriaAdjustment(
                criterion_id="stroke",
                before={"threshold": 10},
                after={"threshold": 5},
                reason="too strict",
                frozen=True,
            )
        self.assertIn("frozen", str(ctx.exception).lower())

    def test_adjustment_requires_reason(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="stroke",
                before={"threshold": 10},
                after={"threshold": 5},
                reason="",
            )

    def test_adjustment_requires_actual_change(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment(
                criterion_id="stroke",
                before={"threshold": 10},
                after={"threshold": 10},
                reason="no real change",
            )

    def test_valid_adjustment_records_timestamp(self):
        adj = CriteriaAdjustment(
            criterion_id="travel",
            before={"min_mm": 10},
            after={"min_mm": 8},
            reason="Simulation shows 8 mm is achievable with current geometry",
        )
        self.assertTrue(adj.adjusted_at)
        self.assertEqual(adj.criterion_id, "travel")

    def test_adjustment_to_dict(self):
        adj = CriteriaAdjustment(
            criterion_id="speed",
            before={"min_rpm": 100},
            after={"min_rpm": 80},
            reason="Motor torque limits",
        )
        d = adj.to_dict()
        self.assertEqual(d["criterion_id"], "speed")
        self.assertEqual(d["before"]["min_rpm"], 100)
        self.assertEqual(d["after"]["min_rpm"], 80)
        self.assertIn("reason", d)


class TestAdjustmentLog(unittest.TestCase):
    """Append-only log for criteria adjustments."""

    def test_empty_log(self):
        log = AdjustmentLog()
        self.assertEqual(log.entries, [])
        self.assertEqual(log.to_list(), [])

    def test_record_and_retrieve(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "reason 1"))
        log.record(CriteriaAdjustment("c2", {"b": 3}, {"b": 4}, "reason 2"))
        self.assertEqual(len(log.entries), 2)
        self.assertEqual(log.entries[0].criterion_id, "c1")

    def test_save_and_load(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"x": 10}, {"x": 5}, "reduced threshold"))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adjustments.json"
            log.save(path)
            self.assertTrue(path.exists())

            loaded = AdjustmentLog.load(path)
            self.assertEqual(len(loaded.entries), 1)
            self.assertEqual(loaded.entries[0].criterion_id, "c1")
            self.assertEqual(loaded.entries[0].before, {"x": 10})
            self.assertEqual(loaded.entries[0].after, {"x": 5})

    def test_load_invalid_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text('{"schema_version": "wrong"}')
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)


if __name__ == "__main__":
    unittest.main()
