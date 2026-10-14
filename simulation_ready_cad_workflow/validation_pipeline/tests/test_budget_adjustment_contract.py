"""Budget adjustment and snapshot field contract tests.

Deep validation of CriteriaAdjustment edge cases, AdjustmentLog
append-only invariant, snapshot field completeness, and the
relationship between snapshot and checkpoint on disk.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from datetime import datetime, timezone
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


class TestCriteriaAdjustmentValidation(unittest.TestCase):
    """CriteriaAdjustment rejects invalid inputs."""

    def test_frozen_raises(self):
        with self.assertRaises(ValueError) as ctx:
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test", frozen=True)
        self.assertIn("frozen", str(ctx.exception))

    def test_empty_reason_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "")

    def test_same_before_after_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 1}, "test")

    def test_whitespace_reason_ok(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "   ")
        self.assertEqual(adj.reason, "   ")

    def test_long_reason_ok(self):
        reason = "x" * 10000
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, reason)
        self.assertEqual(adj.reason, reason)

    def test_complex_nested_dicts(self):
        before = {"level1": {"level2": {"level3": [1, 2, {"key": "val"}]}}}
        after = {"level1": {"level2": {"level3": [1, 2, {"key": "changed"}]}}}
        adj = CriteriaAdjustment("nested", before, after, "deep change")
        self.assertEqual(adj.to_dict()["before"], before)
        self.assertEqual(adj.to_dict()["after"], after)

    def test_custom_adjusted_at(self):
        ts = "2025-01-15T12:00:00+00:00"
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test", adjusted_at=ts)
        self.assertEqual(adj.adjusted_at, ts)
        self.assertEqual(adj.to_dict()["adjusted_at"], ts)

    def test_auto_adjusted_at_is_recent(self):
        before = datetime.now(timezone.utc).isoformat()
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        after = datetime.now(timezone.utc).isoformat()
        self.assertGreaterEqual(adj.adjusted_at, before)
        self.assertLessEqual(adj.adjusted_at, after)

    def test_empty_dicts_different(self):
        adj = CriteriaAdjustment("c1", {}, {"a": 1}, "add field")
        self.assertEqual(adj.to_dict()["before"], {})
        self.assertEqual(adj.to_dict()["after"], {"a": 1})

    def test_none_values_in_dicts(self):
        adj = CriteriaAdjustment("c1", {"a": None}, {"a": 1}, "set value")
        self.assertIsNone(adj.to_dict()["before"]["a"])


class TestAdjustmentLogAppendOnly(unittest.TestCase):
    """AdjustmentLog is append-only and ordered."""

    def test_entries_ordered(self):
        log = AdjustmentLog()
        for i in range(5):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))
        entries = log.to_list()
        for i, entry in enumerate(entries):
            self.assertEqual(entry["criterion_id"], f"c{i}")

    def test_entries_immutable_copy(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test"))
        entries = log.entries
        self.assertEqual(len(entries), 1)
        entries.clear()
        self.assertEqual(len(log.entries), 1)

    def test_to_list_stable(self):
        log = AdjustmentLog()
        for i in range(3):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))
        list1 = log.to_list()
        list2 = log.to_list()
        self.assertEqual(list1, list2)

    def test_empty_log_to_list(self):
        log = AdjustmentLog()
        self.assertEqual(log.to_list(), [])

    def test_empty_log_entries(self):
        log = AdjustmentLog()
        self.assertEqual(log.entries, [])

    def test_record_after_save(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "first"))
        with tempfile.TemporaryDirectory() as tmp:
            log.save(Path(tmp) / "log.json")
        log.record(CriteriaAdjustment("c2", {"b": 1}, {"b": 2}, "second"))
        self.assertEqual(len(log.to_list()), 2)

    def test_same_criterion_multiple_adjustments(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"v": 1}, {"v": 2}, "first"))
        log.record(CriteriaAdjustment("c1", {"v": 2}, {"v": 3}, "second"))
        log.record(CriteriaAdjustment("c1", {"v": 3}, {"v": 4}, "third"))
        entries = log.to_list()
        self.assertEqual(len(entries), 3)
        self.assertTrue(all(e["criterion_id"] == "c1" for e in entries))


class TestAdjustmentLogPersistence(unittest.TestCase):
    """Save/load round-trip preserves all data."""

    def test_save_load_empty(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(loaded.to_list(), [])

    def test_save_load_preserves_order(self):
        log = AdjustmentLog()
        for i in range(10):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i+1}, f"step {i}"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        for i, entry in enumerate(loaded.to_list()):
            self.assertEqual(entry["criterion_id"], f"c{i}")

    def test_save_load_preserves_reason(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "special reason 🔧"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(loaded.to_list()[0]["reason"], "special reason 🔧")

    def test_load_bad_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            path.write_text(json.dumps({"schema_version": "wrong_v1", "adjustments": []}))
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)

    def test_load_missing_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            path.write_text(json.dumps({"adjustments": []}))
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)


class TestSnapshotFieldCompleteness(unittest.TestCase):
    """Snapshot dict has all required fields and correct types."""

    REQUIRED_FIELDS = {
        "schema_version": str,
        "tier": str,
        "active_seconds": float,
        "wall_seconds": float,
        "paused": bool,
        "pause_count": int,
        "total_pause_seconds": float,
        "budget_limit_seconds": (int, float),
        "remaining_active_seconds": float,
        "started_at": str,
        "checkpoint_at": str,
    }

    def _check_fields(self, snap):
        for field, expected_type in self.REQUIRED_FIELDS.items():
            self.assertIn(field, snap, f"Missing field: {field}")
            self.assertIsInstance(snap[field], expected_type, f"Wrong type for {field}: {type(snap[field])}")

    def test_fresh_tracker(self):
        for name, tier in TIERS.items():
            self._check_fields(ActiveBudgetTracker(tier=tier).snapshot())

    def test_after_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        self._check_fields(tracker.snapshot())

    def test_after_resume(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("p")
        tracker.resume()
        self._check_fields(tracker.snapshot())

    def test_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self._check_fields(tracker.snapshot())

    def test_no_extra_fields(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap = tracker.snapshot()
        extra = set(snap.keys()) - set(self.REQUIRED_FIELDS.keys())
        self.assertEqual(extra, set(), f"Unexpected fields: {extra}")


class TestSnapshotCheckpointEquivalence(unittest.TestCase):
    """save_checkpoint writes exactly the snapshot data."""

    def test_checkpoint_matches_snapshot(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / f"cp_{name}.json"
                returned = tracker.save_checkpoint(cp)
                on_disk = json.loads(cp.read_text(encoding="utf-8"))
            self.assertEqual(returned, on_disk, f"Tier {name}")

    def test_checkpoint_has_schema_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            data = tracker.save_checkpoint(cp)
        self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_checkpoint_overwrites(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            time.sleep(0.01)
            data2 = tracker.save_checkpoint(cp)
            on_disk = json.loads(cp.read_text(encoding="utf-8"))
        self.assertEqual(data2, on_disk)


class TestSummaryFieldContract(unittest.TestCase):
    """build_incomplete_summary has all required fields."""

    REQUIRED_FIELDS = {
        "schema_version", "status", "tier", "timing",
        "completed_tests", "pending_tests", "severity",
        "issues", "evidence_paths", "blockers", "generated_at",
    }

    def test_all_fields_present(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            missing = self.REQUIRED_FIELDS - set(summary.keys())
            self.assertEqual(missing, set(), f"Tier {name}: missing {missing}")

    def test_status_is_exhausted(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")

    def test_tier_is_dict(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIsInstance(summary["tier"], dict)
        self.assertEqual(summary["tier"]["name"], "high")

    def test_timing_keys(self):
        expected_timing = {
            "active_seconds", "wall_seconds", "budget_limit_seconds",
            "pause_count", "total_pause_seconds",
        }
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(set(summary["timing"].keys()), expected_timing)

    def test_generated_at_is_iso(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIn("T", summary["generated_at"])

    def test_issues_and_blockers_passthrough(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=["issue1", "issue2"],
            evidence_paths=["/path1"],
            blockers=["blocker1"],
        )
        self.assertEqual(summary["issues"], ["issue1", "issue2"])
        self.assertEqual(summary["evidence_paths"], ["/path1"])
        self.assertEqual(summary["blockers"], ["blocker1"])


class TestGetTierValidation(unittest.TestCase):
    """get_tier edge cases."""

    def test_all_known_tiers(self):
        for name in ("light", "medium", "high"):
            tier = get_tier(name)
            self.assertEqual(tier.name, name)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_tier("nonexistent")
        self.assertIn("nonexistent", str(ctx.exception))

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_tier("Light")
        with self.assertRaises(ValueError):
            get_tier("MEDIUM")
        with self.assertRaises(ValueError):
            get_tier("HIGH")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            get_tier("")


if __name__ == "__main__":
    unittest.main()
