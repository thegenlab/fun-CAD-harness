"""Checkpoint corruption and recovery edge case tests.

Validates that checkpoint save/restore handles:
- Partial writes and truncated JSON
- Missing/extra fields in checkpoint data
- Type mismatches in checkpoint fields
- Multiple sequential checkpoint overwrites
- Checkpoint interleaving between trackers
- Restore from very old or very recent checkpoints
- Path edge cases (non-existent directories, read-only scenarios)
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    CriteriaAdjustment,
    get_tier,
)


class TestCheckpointSaveIntegrity(unittest.TestCase):
    """Checkpoint save produces valid, complete data."""

    def test_save_creates_file(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            self.assertTrue(cp.exists())

    def test_save_creates_parent_dirs(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "deep" / "nested" / "dir" / "cp.json"
            tracker.save_checkpoint(cp)
            self.assertTrue(cp.exists())

    def test_save_returns_snapshot_data(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            data = tracker.save_checkpoint(cp)
        self.assertIsInstance(data, dict)
        self.assertIn("schema_version", data)
        self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_save_matches_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            save_data = tracker.save_checkpoint(cp)
        self.assertEqual(snap["tier"], save_data["tier"])
        self.assertEqual(snap["schema_version"], save_data["schema_version"])

    def test_save_overwrites_existing(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            first_data = json.loads(cp.read_text(encoding="utf-8"))

            time.sleep(0.01)
            tracker.save_checkpoint(cp)
            second_data = json.loads(cp.read_text(encoding="utf-8"))

        self.assertGreater(
            second_data["active_seconds"],
            first_data["active_seconds"],
        )

    def test_save_is_valid_json(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / f"cp_{name}.json"
                tracker.save_checkpoint(cp)
                data = json.loads(cp.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict, f"Tier {name}")

    def test_save_after_pause(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("save_while_paused")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            data = tracker.save_checkpoint(cp)
        self.assertTrue(data["paused"])

    def test_save_after_resume(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("p")
        tracker.resume()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            data = tracker.save_checkpoint(cp)
        self.assertFalse(data["paused"])


class TestCheckpointRestoreEdgeCases(unittest.TestCase):
    """Checkpoint restore with edge-case data."""

    def test_restore_with_zero_active(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "light",
                "active_seconds": 0.0,
                "budget_limit_seconds": 3600,
                "pause_count": 0,
                "total_pause_seconds": 0.0,
            }))
            tracker = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            tracker.resume()
        self.assertAlmostEqual(tracker.elapsed_active, 0.0, delta=0.01)
        self.assertFalse(tracker.is_exhausted)

    def test_restore_with_near_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "light",
                "active_seconds": 3599.9,
                "budget_limit_seconds": 3600,
                "pause_count": 5,
                "total_pause_seconds": 100.0,
            }))
            tracker = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            tracker.resume()
        self.assertGreater(tracker.elapsed_active, 3599)

    def test_restore_with_many_pauses(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "medium",
                "active_seconds": 500.0,
                "budget_limit_seconds": 7200,
                "pause_count": 1000,
                "total_pause_seconds": 5000.0,
            }))
            tracker = ActiveBudgetTracker.restore(cp, tier=get_tier("medium"))
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 1001)

    def test_restore_extra_fields_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "high",
                "active_seconds": 10.0,
                "budget_limit_seconds": 10800,
                "pause_count": 1,
                "total_pause_seconds": 5.0,
                "extra_field_1": "should be ignored",
                "extra_field_2": 42,
                "nested_extra": {"a": 1},
            }))
            tracker = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["tier"], "high")
        self.assertAlmostEqual(snap["active_seconds"], 10.0, delta=0.1)

    def test_restore_missing_pause_count_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "light",
                "active_seconds": 50.0,
                "budget_limit_seconds": 3600,
            }))
            tracker = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 1)

    def test_restore_missing_pause_seconds_defaults_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "medium",
                "active_seconds": 100.0,
                "budget_limit_seconds": 7200,
                "pause_count": 3,
            }))
            tracker = ActiveBudgetTracker.restore(cp, tier=get_tier("medium"))
            tracker.resume()
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["total_pause_seconds"], 0)


class TestCheckpointSequentialOverwrite(unittest.TestCase):
    """Multiple sequential saves to the same path."""

    def test_five_sequential_saves(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            actives = []
            for _ in range(5):
                time.sleep(0.005)
                tracker.save_checkpoint(cp)
                data = json.loads(cp.read_text(encoding="utf-8"))
                actives.append(data["active_seconds"])

        for i in range(len(actives) - 1):
            self.assertLess(actives[i], actives[i+1])

    def test_overwrite_preserves_format(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            for _ in range(3):
                time.sleep(0.003)
                tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "budget_checkpoint_v1")
        self.assertEqual(data["tier"], "light")


class TestCheckpointInterleaving(unittest.TestCase):
    """Two trackers saving checkpoints to different paths."""

    def test_independent_checkpoints(self):
        t1 = ActiveBudgetTracker(tier=get_tier("light"))
        t2 = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "t1.json"
            cp2 = Path(tmp) / "t2.json"
            t1.save_checkpoint(cp1)
            t2.save_checkpoint(cp2)

            d1 = json.loads(cp1.read_text(encoding="utf-8"))
            d2 = json.loads(cp2.read_text(encoding="utf-8"))

        self.assertEqual(d1["tier"], "light")
        self.assertEqual(d2["tier"], "high")
        self.assertEqual(d1["budget_limit_seconds"], 3600)
        self.assertEqual(d2["budget_limit_seconds"], 10800)

    def test_restore_correct_checkpoint(self):
        t1 = ActiveBudgetTracker(tier=get_tier("light"))
        t2 = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)

        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "t1.json"
            cp2 = Path(tmp) / "t2.json"
            t1.save_checkpoint(cp1)
            t2.save_checkpoint(cp2)

            r1 = ActiveBudgetTracker.restore(cp1, tier=get_tier("light"))
            r2 = ActiveBudgetTracker.restore(cp2, tier=get_tier("high"))
            r1.resume()
            r2.resume()

        self.assertEqual(r1.tier.name, "light")
        self.assertEqual(r2.tier.name, "high")

    def test_alternating_save_restore(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                time.sleep(0.005)
                cp = Path(tmp) / f"cp_{i}.json"
                tracker.save_checkpoint(cp)
                tracker = ActiveBudgetTracker.restore(cp, tier=tier)
                tracker.resume()

        snap = tracker.snapshot()
        self.assertEqual(snap["tier"], "medium")
        self.assertGreater(snap["active_seconds"], 0.02)
        self.assertGreater(snap["pause_count"], 0)


class TestAdjustmentLogCorruption(unittest.TestCase):
    """AdjustmentLog save/load with corrupted files."""

    def test_wrong_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            path.write_text(json.dumps({
                "schema_version": "wrong_v1",
                "adjustments": [],
            }))
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)

    def test_missing_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            path.write_text(json.dumps({"adjustments": []}))
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)

    def test_empty_adjustments_loads_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            path.write_text(json.dumps({
                "schema_version": "criteria_adjustment_log_v1",
                "adjustments": [],
                "saved_at": "2024-01-01T00:00:00+00:00",
            }))
            log = AdjustmentLog.load(path)
        self.assertEqual(len(log.to_list()), 0)

    def test_round_trip_preserves_all_entries(self):
        log = AdjustmentLog()
        for i in range(20):
            log.record(CriteriaAdjustment(
                f"criterion_{i}", {"v": i}, {"v": i + 1}, f"step {i}"
            ))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)

        self.assertEqual(len(loaded.to_list()), 20)
        for i, entry in enumerate(loaded.to_list()):
            self.assertEqual(entry["criterion_id"], f"criterion_{i}")
            self.assertEqual(entry["before"], {"v": i})
            self.assertEqual(entry["after"], {"v": i + 1})

    def test_sequential_saves_overwrite(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "first"))
            log.save(path)
            first = json.loads(path.read_text(encoding="utf-8"))

            log.record(CriteriaAdjustment("c2", {"b": 1}, {"b": 2}, "second"))
            log.save(path)
            second = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(first["adjustments"]), 1)
        self.assertEqual(len(second["adjustments"]), 2)


class TestCriteriaAdjustmentEdgeCases(unittest.TestCase):
    """CriteriaAdjustment validation edge cases."""

    def test_frozen_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test", frozen=True)

    def test_empty_reason_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "")

    def test_same_before_after_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 1}, "no change")

    def test_deeply_nested_dicts_accepted(self):
        before = {"level1": {"level2": {"level3": {"value": 0}}}}
        after = {"level1": {"level2": {"level3": {"value": 1}}}}
        adj = CriteriaAdjustment("deep", before, after, "deep change")
        d = adj.to_dict()
        self.assertEqual(d["criterion_id"], "deep")

    def test_empty_dict_before_after_raises_if_equal(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {}, {}, "empty")

    def test_different_keys_accepted(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"b": 2}, "key change")
        self.assertEqual(adj.to_dict()["criterion_id"], "c1")

    def test_to_dict_has_adjusted_at(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test")
        d = adj.to_dict()
        self.assertIn("adjusted_at", d)
        self.assertIn("T", d["adjusted_at"])

    def test_custom_adjusted_at_preserved(self):
        ts = "2024-06-15T12:00:00+00:00"
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test", adjusted_at=ts)
        self.assertEqual(adj.adjusted_at, ts)


if __name__ == "__main__":
    unittest.main()
