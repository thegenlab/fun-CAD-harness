"""Budget defensive input tests.

Tests the boundary defense of every budget API against unusual,
extreme, or adversarial inputs: malformed JSON, extreme numeric
values, special characters, and unexpected type combinations.
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
    build_incomplete_summary,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    get_tier,
    recovery_delay,
    should_retry,
)


# ── TierConfig construction defense ──────────────────────────────────

class TestTierConfigDefense(unittest.TestCase):
    """TierConfig rejects invalid construction arguments."""

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 0, 60, 2, 0, 10)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -100, 60, 2, 0, 10)

    def test_zero_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 0, 2, 0, 10)

    def test_negative_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, -5, 2, 0, 10)

    def test_inf_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("inf"), 60, 2, 0, 10)

    def test_nan_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("nan"), 60, 2, 0, 10)

    def test_inf_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, float("inf"), 2, 0, 10)

    def test_nan_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, float("nan"), 2, 0, 10)

    def test_max_tasks_zero_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 0, 0, 10)

    def test_max_parts_zero_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, 0, 0)

    def test_min_parts_negative_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, -1, 10)

    def test_min_exceeds_max_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, 15, 10)

    def test_minimal_valid_tier(self):
        tier = TierConfig("min", 0.001, 0.001, 1, 0, 1)
        self.assertEqual(tier.name, "min")

    def test_very_large_budget(self):
        tier = TierConfig("huge", 1e9, 1e9, 1000, 0, 10000)
        self.assertEqual(tier.active_budget_seconds, 1e9)

    def test_min_equals_max_ok(self):
        tier = TierConfig("equal", 3600, 60, 2, 5, 5)
        self.assertEqual(tier.min_parts, 5)
        self.assertEqual(tier.max_parts, 5)


# ── enforce_simulation_timeout defense ────────────────────────────────

class TestEnforceTimeoutDefense(unittest.TestCase):
    """enforce_simulation_timeout rejects invalid actual_timeout values."""

    def test_nan_rejected(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("nan"))

    def test_negative_inf_rejected(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("-inf"))

    def test_very_small_positive_accepted(self):
        tier = get_tier("high")
        result = enforce_simulation_timeout(tier, 0.0001)
        self.assertAlmostEqual(result, 0.0001, places=6)

    def test_none_uses_tier_default(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier, None)
            self.assertEqual(result, tier.simulation_timeout_seconds)


# ── should_retry defense ──────────────────────────────────────────────

class TestShouldRetryDefense(unittest.TestCase):
    """should_retry handles unusual error strings."""

    def test_empty_error_retryable(self):
        self.assertTrue(should_retry(0, ""))

    def test_very_long_error_string(self):
        error = "x" * 100000
        self.assertTrue(should_retry(0, error))

    def test_unicode_error(self):
        self.assertTrue(should_retry(0, "错误信息：连接超时"))

    def test_newlines_in_error(self):
        self.assertTrue(should_retry(0, "line1\nline2\nline3"))

    def test_error_with_substring_pass(self):
        self.assertFalse(should_retry(0, "this contains PASS somewhere"))

    def test_error_with_compass(self):
        self.assertFalse(should_retry(0, "COMPASS navigation error"))

    def test_negative_attempt(self):
        self.assertTrue(should_retry(-1, "error"))

    def test_very_high_attempt(self):
        self.assertFalse(should_retry(1000, "error"))


# ── recovery_delay defense ────────────────────────────────────────────

class TestRecoveryDelayDefense(unittest.TestCase):
    """recovery_delay with extreme attempt numbers."""

    def test_zero_attempt(self):
        self.assertEqual(recovery_delay(0), 5.0)

    def test_large_attempt_capped(self):
        self.assertEqual(recovery_delay(100), 60.0)

    def test_negative_attempt(self):
        d = recovery_delay(-1)
        self.assertIsInstance(d, float)
        self.assertGreater(d, 0)

    def test_all_results_positive(self):
        for i in range(-5, 50):
            self.assertGreater(recovery_delay(i), 0)


# ── CriteriaAdjustment defense ────────────────────────────────────────

class TestCriteriaAdjustmentDefense(unittest.TestCase):
    """CriteriaAdjustment with unusual but valid inputs."""

    def test_unicode_criterion_id(self):
        adj = CriteriaAdjustment("齿轮_公差", {"x": 1}, {"x": 2}, "中文原因")
        self.assertEqual(adj.to_dict()["criterion_id"], "齿轮_公差")

    def test_very_long_criterion_id(self):
        cid = "a" * 10000
        adj = CriteriaAdjustment(cid, {"x": 1}, {"x": 2}, "long id")
        self.assertEqual(adj.to_dict()["criterion_id"], cid)

    def test_numeric_dict_keys(self):
        adj = CriteriaAdjustment("c1", {"1": "a"}, {"1": "b"}, "num keys")
        d = adj.to_dict()
        self.assertIn("1", d["before"])

    def test_boolean_values_in_dicts(self):
        adj = CriteriaAdjustment("c1", {"flag": True}, {"flag": False}, "flip flag")
        d = adj.to_dict()
        self.assertTrue(d["before"]["flag"])
        self.assertFalse(d["after"]["flag"])

    def test_list_values_in_dicts(self):
        adj = CriteriaAdjustment("c1", {"items": [1, 2]}, {"items": [3, 4]}, "change list")
        d = adj.to_dict()
        self.assertEqual(d["after"]["items"], [3, 4])

    def test_none_value_transitions(self):
        adj = CriteriaAdjustment("c1", {"val": None}, {"val": 42}, "set value")
        d = adj.to_dict()
        self.assertIsNone(d["before"]["val"])
        self.assertEqual(d["after"]["val"], 42)


# ── AdjustmentLog defense ─────────────────────────────────────────────

class TestAdjustmentLogDefense(unittest.TestCase):
    """AdjustmentLog save/load with malformed files."""

    def test_load_non_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json at all", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                AdjustmentLog.load(path)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            AdjustmentLog.load(Path("/nonexistent/path/log.json"))

    def test_load_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                AdjustmentLog.load(path)

    def test_load_array_instead_of_object(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arr.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises((ValueError, AttributeError, TypeError)):
                AdjustmentLog.load(path)

    def test_save_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "deep" / "nested" / "log.json"
            log = AdjustmentLog()
            log.save(path)
            self.assertTrue(path.exists())
            loaded = AdjustmentLog.load(path)
            self.assertEqual(loaded.to_list(), [])


# ── Checkpoint defense ────────────────────────────────────────────────

class TestCheckpointDefense(unittest.TestCase):
    """ActiveBudgetTracker.restore with malformed checkpoint files."""

    def test_restore_non_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "bad.json"
            cp.write_text("not json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                ActiveBudgetTracker.restore(cp)

    def test_restore_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "empty.json"
            cp.write_text("", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                ActiveBudgetTracker.restore(cp)

    def test_restore_missing_active_seconds(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            del data["active_seconds"]
            cp.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                ActiveBudgetTracker.restore(cp)

    def test_checkpoint_overwrites_existing(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text("old data", encoding="utf-8")
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "budget_checkpoint_v1")


# ── build_incomplete_summary defense ──────────────────────────────────

class TestSummaryDefense(unittest.TestCase):
    """build_incomplete_summary with various test list shapes."""

    def test_empty_everything(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "minor")

    def test_only_passing_tests(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": f"t{i}", "physical_pass": True, "geometry_audit": {"passed": True}}
            for i in range(5)
        ]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "minor")
        self.assertEqual(len(summary["completed_tests"]), 5)

    def test_mixed_pass_fail(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
            {"test_id": "t2", "physical_pass": False, "geometry_audit": {"passed": True}},
        ]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_geometry_fail_is_functional(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": False}},
        ]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_pending_alone_is_functional(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=["pending1"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_missing_geometry_audit_key(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [{"test_id": "t1", "physical_pass": True}]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_missing_physical_pass_key(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [{"test_id": "t1", "geometry_audit": {"passed": True}}]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_large_test_lists(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": f"t{i}", "physical_pass": True, "geometry_audit": {"passed": True}}
            for i in range(200)
        ]
        pending = [f"p{i}" for i in range(100)]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=pending,
            issues=["many tests"], evidence_paths=["/e"],
            blockers=["time"],
        )
        self.assertEqual(len(summary["completed_tests"]), 200)
        self.assertEqual(len(summary["pending_tests"]), 100)
        self.assertEqual(summary["severity"], "functional")


# ── SimulationTimeoutResult defense ───────────────────────────────────

class TestSimulationTimeoutResultDefense(unittest.TestCase):
    """SimulationTimeoutResult with edge-case field values."""

    def test_zero_elapsed(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=0.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_very_large_elapsed(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=1e12)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_none_optional_fields(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0)
        self.assertIsNone(r.partial_evidence)
        self.assertIsNone(r.log_path)
        self.assertIsNone(r.error)

    def test_all_optional_fields_set(self):
        r = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=10.0,
            partial_evidence={"key": "val"}, log_path=Path("/tmp/log"),
            error="timed out",
        )
        self.assertIsNotNone(r.partial_evidence)
        self.assertIsNotNone(r.log_path)
        self.assertIsNotNone(r.error)

    def test_empty_partial_evidence(self):
        r = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=5.0, partial_evidence={},
        )
        self.assertEqual(r.partial_evidence, {})


if __name__ == "__main__":
    unittest.main()
