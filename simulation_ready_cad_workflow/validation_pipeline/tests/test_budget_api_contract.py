"""Budget public API contract tests.

Validates every exported symbol from budget.py: function signatures,
return types, exception types, and docstring-promised behavior.
Guards against accidental API breakage.
"""
from __future__ import annotations

import inspect
import json
import tempfile
import time
import unittest
from dataclasses import fields
from pathlib import Path

from ..slim_harness.budget import (
    BACKOFF_BASE_SECONDS,
    MAX_RECOVERY_RETRIES,
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    SimulationTimeoutResult,
    build_incomplete_summary,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    get_tier,
    recovery_delay,
    should_retry,
)


class TestExportedConstants(unittest.TestCase):
    """Module-level constants have expected values."""

    def test_backoff_base(self):
        self.assertEqual(BACKOFF_BASE_SECONDS, 5.0)
        self.assertIsInstance(BACKOFF_BASE_SECONDS, float)

    def test_max_recovery_retries(self):
        self.assertEqual(MAX_RECOVERY_RETRIES, 3)
        self.assertIsInstance(MAX_RECOVERY_RETRIES, int)

    def test_tiers_has_three_entries(self):
        self.assertEqual(set(TIERS.keys()), {"light", "medium", "high"})

    def test_tiers_values_are_tierconfig(self):
        for name, tier in TIERS.items():
            self.assertIsInstance(tier, TierConfig)
            self.assertEqual(tier.name, name)


class TestTierConfigContract(unittest.TestCase):
    """TierConfig is a frozen dataclass with expected fields."""

    def test_is_frozen(self):
        tier = get_tier("light")
        with self.assertRaises(AttributeError):
            tier.name = "changed"

    def test_expected_fields(self):
        expected = {"name", "active_budget_seconds", "simulation_timeout_seconds",
                    "max_tasks_per_design", "min_parts", "max_parts"}
        actual = {f.name for f in fields(TierConfig)}
        self.assertEqual(actual, expected)

    def test_field_types(self):
        tier = get_tier("medium")
        self.assertIsInstance(tier.name, str)
        self.assertIsInstance(tier.active_budget_seconds, (int, float))
        self.assertIsInstance(tier.simulation_timeout_seconds, (int, float))
        self.assertIsInstance(tier.max_tasks_per_design, int)
        self.assertIsInstance(tier.min_parts, int)
        self.assertIsInstance(tier.max_parts, int)

    def test_production_tier_values(self):
        light = get_tier("light")
        self.assertEqual(light.active_budget_seconds, 3600)
        self.assertEqual(light.simulation_timeout_seconds, 60)
        self.assertEqual(light.max_tasks_per_design, 2)
        self.assertEqual(light.min_parts, 0)
        self.assertEqual(light.max_parts, 10)

        medium = get_tier("medium")
        self.assertEqual(medium.active_budget_seconds, 7200)
        self.assertEqual(medium.simulation_timeout_seconds, 180)
        self.assertEqual(medium.max_tasks_per_design, 2)
        self.assertEqual(medium.min_parts, 0)
        self.assertEqual(medium.max_parts, 15)

        high = get_tier("high")
        self.assertEqual(high.active_budget_seconds, 10800)
        self.assertEqual(high.simulation_timeout_seconds, 300)
        self.assertEqual(high.max_tasks_per_design, 5)
        self.assertEqual(high.min_parts, 0)
        self.assertEqual(high.max_parts, 20)


class TestGetTierContract(unittest.TestCase):
    """get_tier returns TierConfig or raises ValueError."""

    def test_returns_tierconfig(self):
        for name in ("light", "medium", "high"):
            result = get_tier(name)
            self.assertIsInstance(result, TierConfig)

    def test_raises_valueerror_for_unknown(self):
        with self.assertRaises(ValueError):
            get_tier("ultra")

    def test_error_message_contains_name(self):
        try:
            get_tier("nonexistent")
        except ValueError as e:
            self.assertIn("nonexistent", str(e))


class TestBudgetStatusContract(unittest.TestCase):
    """BudgetStatus enum has expected members and methods."""

    EXPECTED_MEMBERS = {
        "PASS", "ACTIVE_BUDGET_EXHAUSTED", "INFRASTRUCTURE_FAILURE",
        "SIMULATION_TIMEOUT", "EVIDENCE_REQUIRED", "INCOMPLETE_EVIDENCE",
        "HARNESS_ERROR", "NEEDS_REVISION", "RESOURCE_POLICY_REQUIRED",
    }

    def test_all_members_present(self):
        actual = {s.name for s in BudgetStatus}
        self.assertEqual(actual, self.EXPECTED_MEMBERS)

    def test_is_terminal_exists(self):
        for status in BudgetStatus:
            self.assertTrue(hasattr(status, "is_terminal"))
            self.assertIsInstance(status.is_terminal(), bool)

    def test_terminal_statuses(self):
        terminal = {s for s in BudgetStatus if s.is_terminal()}
        expected_terminal = {
            BudgetStatus.PASS,
            BudgetStatus.ACTIVE_BUDGET_EXHAUSTED,
            BudgetStatus.INFRASTRUCTURE_FAILURE,
        }
        self.assertEqual(terminal, expected_terminal)

    def test_value_equals_name(self):
        for status in BudgetStatus:
            self.assertEqual(status.value, status.name)


class TestActiveTrackerContract(unittest.TestCase):
    """ActiveBudgetTracker has expected properties and methods."""

    def test_constructor_accepts_tier(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertIsNotNone(tracker)

    def test_properties_exist(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        self.assertIsInstance(tracker.elapsed_active, float)
        self.assertIsInstance(tracker.elapsed_wall, float)
        self.assertIsInstance(tracker.remaining_active, float)
        self.assertIsInstance(tracker.is_exhausted, bool)
        self.assertIsInstance(tracker.is_paused, bool)

    def test_methods_exist(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        self.assertTrue(callable(tracker.pause))
        self.assertTrue(callable(tracker.resume))
        self.assertTrue(callable(tracker.snapshot))
        self.assertTrue(callable(tracker.save_checkpoint))
        self.assertTrue(hasattr(ActiveBudgetTracker, "restore"))

    def test_snapshot_returns_dict(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap = tracker.snapshot()
        self.assertIsInstance(snap, dict)

    def test_snapshot_required_keys(self):
        expected = {"schema_version", "tier", "tier_config", "active_seconds", "wall_seconds",
                    "paused", "pause_count", "total_pause_seconds",
                    "budget_limit_seconds", "remaining_active_seconds",
                    "started_at", "checkpoint_at"}
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        self.assertEqual(set(tracker.snapshot().keys()), expected)

    def test_save_checkpoint_returns_dict(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            result = tracker.save_checkpoint(Path(tmp) / "cp.json")
        self.assertIsInstance(result, dict)

    def test_restore_returns_tracker(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertIsInstance(restored, ActiveBudgetTracker)

    def test_restore_starts_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertTrue(restored.is_paused)


class TestSimulationTimeoutResultContract(unittest.TestCase):
    """SimulationTimeoutResult dataclass fields and status property."""

    def test_required_fields(self):
        expected = {"timed_out", "elapsed_seconds", "partial_evidence",
                    "log_path", "error"}
        actual = {f.name for f in fields(SimulationTimeoutResult)}
        self.assertEqual(actual, expected)

    def test_status_property(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=1.0)
        self.assertIsInstance(r.status, BudgetStatus)

    def test_timed_out_returns_timeout_status(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=1.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_not_timed_out_returns_pass(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_defaults_are_none(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0)
        self.assertIsNone(r.partial_evidence)
        self.assertIsNone(r.log_path)
        self.assertIsNone(r.error)


class TestEnforceTimeoutContract(unittest.TestCase):
    """enforce_simulation_timeout signature and return type."""

    def test_returns_float_or_int(self):
        result = enforce_simulation_timeout(get_tier("medium"))
        self.assertIsInstance(result, (int, float))

    def test_none_uses_tier_default(self):
        tier = get_tier("light")
        self.assertEqual(enforce_simulation_timeout(tier, None), tier.simulation_timeout_seconds)

    def test_clamped_to_tier_max(self):
        tier = get_tier("medium")
        result = enforce_simulation_timeout(tier, 999999)
        self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_raises_valueerror_for_zero(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("light"), 0)

    def test_raises_valueerror_for_negative(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("medium"), -1)


class TestCheckFunctionsContract(unittest.TestCase):
    """check_task_count and check_part_count return list of strings."""

    def test_task_count_returns_list(self):
        result = check_task_count(["t1"], get_tier("light"))
        self.assertIsInstance(result, list)

    def test_part_count_returns_list(self):
        result = check_part_count(["p1"], get_tier("medium"))
        self.assertIsInstance(result, list)

    def test_empty_for_valid(self):
        self.assertEqual(check_task_count([], get_tier("high")), [])
        self.assertEqual(check_part_count([], get_tier("high")), [])

    def test_nonempty_for_over_limit(self):
        tier = get_tier("light")
        over = ["t"] * (tier.max_tasks_per_design + 1)
        violations = check_task_count(over, tier)
        self.assertGreater(len(violations), 0)
        self.assertIsInstance(violations[0], str)


class TestSummaryContract(unittest.TestCase):
    """build_incomplete_summary returns dict with expected schema."""

    def test_returns_dict(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        result = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIsInstance(result, dict)

    def test_schema_version(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        result = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(result["schema_version"], "budget_incomplete_summary_v1")

    def test_severity_values(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        minor = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                             "geometry_audit": {"passed": True}}],
            pending_tests=[], issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(minor["severity"], "minor")
        functional = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=["t1"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(functional["severity"], "functional")


class TestCriteriaAdjustmentContract(unittest.TestCase):
    """CriteriaAdjustment to_dict keys and validation."""

    def test_to_dict_keys(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        d = adj.to_dict()
        expected = {"criterion_id", "before", "after", "reason", "adjusted_at"}
        self.assertEqual(set(d.keys()), expected)

    def test_frozen_raises_valueerror(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "r", frozen=True)

    def test_same_raises_valueerror(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 1}, "r")

    def test_empty_reason_raises_valueerror(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "")


class TestAdjustmentLogContract(unittest.TestCase):
    """AdjustmentLog methods and persistence format."""

    def test_record_and_list(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "r"))
        self.assertEqual(len(log.to_list()), 1)

    def test_entries_returns_list(self):
        log = AdjustmentLog()
        self.assertIsInstance(log.entries, list)

    def test_save_load_schema(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")
        self.assertIn("adjustments", data)

    def test_load_returns_adjustmentlog(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertIsInstance(loaded, AdjustmentLog)


class TestRetryHelperContract(unittest.TestCase):
    """should_retry and recovery_delay signatures and return types."""

    def test_should_retry_returns_bool(self):
        self.assertIsInstance(should_retry(0, "error"), bool)

    def test_recovery_delay_returns_float(self):
        self.assertIsInstance(recovery_delay(0), float)

    def test_delay_always_positive(self):
        for i in range(20):
            self.assertGreater(recovery_delay(i), 0)

    def test_delay_capped_at_60(self):
        for i in range(20):
            self.assertLessEqual(recovery_delay(i), 60.0)

    def test_retry_respects_max_retries(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "transient"))
        if MAX_RECOVERY_RETRIES > 0:
            self.assertTrue(should_retry(MAX_RECOVERY_RETRIES - 1, "transient"))


if __name__ == "__main__":
    unittest.main()
