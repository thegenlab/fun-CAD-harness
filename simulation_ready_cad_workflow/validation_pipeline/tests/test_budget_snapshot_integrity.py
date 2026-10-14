"""Budget snapshot and checkpoint data integrity tests.

Tests that snapshots, checkpoints, and summaries maintain structural
integrity across all operations:
- Snapshot schema completeness at every tracker state
- Checkpoint JSON structural validity
- Summary schema adherence
- Cross-reference consistency between snapshot and checkpoint
- Schema version stability
- Field type invariants
- Monotonicity of timing fields
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
    get_tier,
    build_incomplete_summary,
)

SNAPSHOT_REQUIRED_KEYS = {
    "schema_version", "tier", "active_seconds", "wall_seconds",
    "paused", "pause_count", "total_pause_seconds",
    "budget_limit_seconds", "remaining_active_seconds",
    "started_at", "checkpoint_at",
}

SUMMARY_REQUIRED_KEYS = {
    "schema_version", "status", "tier", "timing",
    "completed_tests", "pending_tests", "severity",
    "issues", "evidence_paths", "blockers", "generated_at",
}

SUMMARY_TIMING_KEYS = {
    "active_seconds", "wall_seconds", "budget_limit_seconds",
    "pause_count", "total_pause_seconds",
}


class TestSnapshotSchemaCompleteness(unittest.TestCase):
    """Every snapshot has all required keys."""

    def test_fresh_tracker_has_all_keys(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertEqual(SNAPSHOT_REQUIRED_KEYS - set(snap.keys()), set())

    def test_paused_tracker_has_all_keys(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        tracker.pause("test")
        snap = tracker.snapshot()
        self.assertEqual(SNAPSHOT_REQUIRED_KEYS - set(snap.keys()), set())

    def test_exhausted_tracker_has_all_keys(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertEqual(SNAPSHOT_REQUIRED_KEYS - set(snap.keys()), set())

    def test_restored_tracker_has_all_keys(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
        snap = restored.snapshot()
        self.assertEqual(SNAPSHOT_REQUIRED_KEYS - set(snap.keys()), set())

    def test_all_tiers_have_complete_snapshots(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            snap = tracker.snapshot()
            missing = SNAPSHOT_REQUIRED_KEYS - set(snap.keys())
            self.assertEqual(missing, set(), f"Tier {name} missing: {missing}")


class TestSnapshotFieldTypes(unittest.TestCase):
    """Snapshot field types are correct."""

    def _assert_types(self, snap):
        self.assertIsInstance(snap["schema_version"], str)
        self.assertIsInstance(snap["tier"], str)
        self.assertIsInstance(snap["active_seconds"], (int, float))
        self.assertIsInstance(snap["wall_seconds"], (int, float))
        self.assertIsInstance(snap["paused"], bool)
        self.assertIsInstance(snap["pause_count"], int)
        self.assertIsInstance(snap["total_pause_seconds"], (int, float))
        self.assertIsInstance(snap["budget_limit_seconds"], (int, float))
        self.assertIsInstance(snap["remaining_active_seconds"], (int, float))
        self.assertIsInstance(snap["started_at"], str)
        self.assertIsInstance(snap["checkpoint_at"], str)

    def test_fresh_snapshot_types(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        self._assert_types(tracker.snapshot())

    def test_after_pauses_snapshot_types(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for i in range(3):
            tracker.pause(f"p{i}")
            tracker.resume()
        self._assert_types(tracker.snapshot())

    def test_restored_snapshot_types(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
        self._assert_types(restored.snapshot())


class TestSnapshotTimingInvariants(unittest.TestCase):
    """Timing fields obey mathematical invariants."""

    def test_active_le_wall(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        self.assertLessEqual(snap["active_seconds"], snap["wall_seconds"] + 0.001)

    def test_remaining_plus_active_eq_budget(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        total = snap["remaining_active_seconds"] + snap["active_seconds"]
        self.assertAlmostEqual(total, snap["budget_limit_seconds"], delta=0.01)

    def test_active_nonnegative(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["active_seconds"], 0)

    def test_wall_nonnegative(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["wall_seconds"], 0)

    def test_remaining_nonnegative(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["remaining_active_seconds"], 0)

    def test_pause_seconds_nonnegative(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("p")
        time.sleep(0.005)
        tracker.resume()
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["total_pause_seconds"], 0)

    def test_pause_count_nonnegative(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["pause_count"], 0)

    def test_all_numeric_are_finite(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        for key in ["active_seconds", "wall_seconds", "budget_limit_seconds",
                     "remaining_active_seconds", "total_pause_seconds"]:
            self.assertTrue(math.isfinite(snap[key]), f"{key} not finite: {snap[key]}")

    def test_active_monotonic_over_time(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        prev = 0
        for _ in range(5):
            time.sleep(0.005)
            snap = tracker.snapshot()
            self.assertGreaterEqual(snap["active_seconds"], prev)
            prev = snap["active_seconds"]


class TestSnapshotSchemaVersion(unittest.TestCase):
    """Schema version is always correct."""

    def test_schema_is_budget_checkpoint_v1(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertEqual(snap["schema_version"], "budget_checkpoint_v1")

    def test_schema_stable_across_operations(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertEqual(tracker.snapshot()["schema_version"], "budget_checkpoint_v1")
        tracker.pause("p")
        self.assertEqual(tracker.snapshot()["schema_version"], "budget_checkpoint_v1")
        tracker.resume()
        self.assertEqual(tracker.snapshot()["schema_version"], "budget_checkpoint_v1")


class TestCheckpointJsonIntegrity(unittest.TestCase):
    """Checkpoint files are valid JSON with correct structure."""

    def test_checkpoint_is_valid_json(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)

    def test_checkpoint_has_tier(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        self.assertIn("tier", data)
        self.assertEqual(data["tier"], "light")

    def test_checkpoint_has_timing_fields(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        for key in ["active_seconds", "wall_seconds", "pause_count", "total_pause_seconds"]:
            self.assertIn(key, data, f"Missing key: {key}")

    def test_checkpoint_round_trip_preserves_tier(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / f"cp_{name}.json"
                tracker.save_checkpoint(cp)
                restored = ActiveBudgetTracker.restore(cp, tier=tier)
                restored.resume()
            self.assertEqual(restored.tier.name, name)

    def test_multiple_checkpoints_independent(self):
        tier = get_tier("medium")
        t1 = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        t2 = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)

        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "cp1.json"
            cp2 = Path(tmp) / "cp2.json"
            t1.save_checkpoint(cp1)
            t2.save_checkpoint(cp2)
            d1 = json.loads(cp1.read_text(encoding="utf-8"))
            d2 = json.loads(cp2.read_text(encoding="utf-8"))
        self.assertNotEqual(d1["active_seconds"], d2["active_seconds"])


class TestSummarySchemaCompleteness(unittest.TestCase):
    """Incomplete summary has all required keys and correct structure."""

    def _make_summary(self, tier_name="medium", **kwargs):
        tier = get_tier(tier_name)
        tracker = ActiveBudgetTracker(tier=tier)
        defaults = dict(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        defaults.update(kwargs)
        return build_incomplete_summary(**defaults)

    def test_has_all_required_keys(self):
        summary = self._make_summary()
        missing = SUMMARY_REQUIRED_KEYS - set(summary.keys())
        self.assertEqual(missing, set(), f"Missing: {missing}")

    def test_timing_has_all_required_keys(self):
        summary = self._make_summary()
        missing = SUMMARY_TIMING_KEYS - set(summary["timing"].keys())
        self.assertEqual(missing, set(), f"Missing timing: {missing}")

    def test_timing_has_no_remaining_active(self):
        summary = self._make_summary()
        self.assertNotIn("remaining_active_seconds", summary["timing"])

    def test_schema_version_is_correct(self):
        summary = self._make_summary()
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_status_is_exhausted(self):
        summary = self._make_summary()
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")

    def test_tier_is_dict(self):
        summary = self._make_summary()
        self.assertIsInstance(summary["tier"], dict)
        self.assertIn("name", summary["tier"])

    def test_completed_tests_structure(self):
        summary = self._make_summary(
            completed_tests=[
                {"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}},
                {"test_id": "t2", "physical_pass": False, "geometry_audit": {"passed": True}},
            ]
        )
        for entry in summary["completed_tests"]:
            self.assertIn("id", entry)
            self.assertIn("status", entry)
            self.assertIn(entry["status"], ("PASS", "FAIL"))

    def test_severity_functional_with_pending(self):
        summary = self._make_summary(pending_tests=["t1"])
        self.assertEqual(summary["severity"], "functional")

    def test_severity_minor_all_pass(self):
        summary = self._make_summary(
            completed_tests=[{"test_id": "t1", "physical_pass": True, "geometry_audit": {"passed": True}}],
            pending_tests=[],
        )
        self.assertEqual(summary["severity"], "minor")

    def test_generated_at_is_iso_string(self):
        summary = self._make_summary()
        self.assertIsInstance(summary["generated_at"], str)
        self.assertIn("T", summary["generated_at"])


class TestAdjustmentLogSchemaIntegrity(unittest.TestCase):
    """Adjustment log schema stays correct through operations."""

    def test_to_dict_has_criterion_id(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        d = adj.to_dict()
        self.assertIn("criterion_id", d)
        self.assertNotIn("criterion", d)

    def test_to_dict_has_all_fields(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        d = adj.to_dict()
        required = {"criterion_id", "before", "after", "reason", "adjusted_at"}
        self.assertEqual(required - set(d.keys()), set())

    def test_log_save_schema(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"x": 0}, {"x": 1}, "update"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")
        self.assertIn("adjustments", data)
        self.assertIn("saved_at", data)

    def test_log_entries_match_to_dict(self):
        log = AdjustmentLog()
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "reason")
        log.record(adj)
        entries = log.to_list()
        d = adj.to_dict()
        self.assertEqual(entries[0], d)


if __name__ == "__main__":
    unittest.main()
