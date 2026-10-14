"""Budget checkpoint versioning and forward-compatibility tests.

Validates that checkpoint files are schema-versioned correctly, survive
round-trip save/restore, tolerate unknown extra fields for forward
compatibility, and correctly reject invalid or missing required fields.
"""
from __future__ import annotations

import json
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
    CriteriaAdjustment,
    get_tier,
)


class TestCheckpointSchemaVersion(unittest.TestCase):
    """Checkpoint files carry the correct schema version."""

    def test_snapshot_has_correct_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap = tracker.snapshot()
        self.assertEqual(snap["schema_version"], "budget_checkpoint_v1")

    def test_saved_file_has_correct_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_restore_rejects_wrong_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            data["schema_version"] = "budget_checkpoint_v999"
            cp.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                ActiveBudgetTracker.restore(cp)

    def test_restore_rejects_missing_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            del data["schema_version"]
            cp.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                ActiveBudgetTracker.restore(cp)

    def test_version_string_is_exact(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        self.assertNotEqual(snap["schema_version"], "budget_snapshot_v1")
        self.assertNotEqual(snap["schema_version"], "checkpoint_v1")


class TestCheckpointForwardCompatibility(unittest.TestCase):
    """Restore tolerates extra unknown fields for forward compatibility."""

    def test_extra_fields_tolerated(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            data["future_field"] = "some_value"
            data["another_future"] = 42
            data["nested_future"] = {"a": [1, 2, 3]}
            cp.write_text(json.dumps(data), encoding="utf-8")
            restored = ActiveBudgetTracker.restore(cp)
        self.assertIsNotNone(restored)
        self.assertTrue(restored.is_paused)

    def test_extra_fields_dont_corrupt_state(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            orig_data = tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            data["future_feature_enabled"] = True
            data["compatibility_level"] = 99
            cp.write_text(json.dumps(data), encoding="utf-8")
            restored = ActiveBudgetTracker.restore(cp)
        self.assertAlmostEqual(
            restored.elapsed_active,
            orig_data["active_seconds"],
            delta=0.01,
        )

    def test_null_extra_fields_tolerated(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            data["optional_future"] = None
            cp.write_text(json.dumps(data), encoding="utf-8")
            restored = ActiveBudgetTracker.restore(cp)
        self.assertIsNotNone(restored)


class TestCheckpointRequiredFields(unittest.TestCase):
    """Restore requires specific fields in the checkpoint."""

    def _save_and_modify(self, tier_name, modifier_fn):
        tracker = ActiveBudgetTracker(tier=get_tier(tier_name))
        time.sleep(0.005)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            modifier_fn(data)
            cp.write_text(json.dumps(data), encoding="utf-8")
            return cp, tmp

    def test_missing_tier_uses_budget_limit(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            data["tier"] = "unknown_custom_tier"
            cp.write_text(json.dumps(data), encoding="utf-8")
            restored = ActiveBudgetTracker.restore(cp)
        self.assertEqual(restored.tier.name, "unknown_custom_tier")
        self.assertEqual(
            restored.tier.active_budget_seconds,
            get_tier("light").active_budget_seconds,
        )

    def test_active_seconds_preserved(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            orig = tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertAlmostEqual(
            restored.elapsed_active,
            orig["active_seconds"],
            delta=0.005,
        )

    def test_pause_count_incremented_on_restore(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("p")
        tracker.resume()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            orig = tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertEqual(
            restored.snapshot()["pause_count"],
            orig["pause_count"] + 1,
        )

    def test_budget_limit_preserved(self):
        for name in ("light", "medium", "high"):
            tier = get_tier(name)
            tracker = ActiveBudgetTracker(tier=tier)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / "cp.json"
                tracker.save_checkpoint(cp)
                restored = ActiveBudgetTracker.restore(cp)
            self.assertEqual(
                restored.tier.active_budget_seconds,
                tier.active_budget_seconds,
            )


class TestCheckpointRoundTrip(unittest.TestCase):
    """Save then restore then save again preserves semantics."""

    def test_double_round_trip_active_seconds(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "cp1.json"
            tracker.save_checkpoint(cp1)
            r1 = ActiveBudgetTracker.restore(cp1)
            cp2 = Path(tmp) / "cp2.json"
            r1.save_checkpoint(cp2)
            r2 = ActiveBudgetTracker.restore(cp2)
        self.assertAlmostEqual(
            r1.elapsed_active, r2.elapsed_active, delta=0.01,
        )

    def test_round_trip_preserves_tier_name(self):
        for name in ("light", "medium", "high"):
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / "cp.json"
                tracker.save_checkpoint(cp)
                restored = ActiveBudgetTracker.restore(cp)
            self.assertEqual(restored.tier.name, name)

    def test_round_trip_preserves_budget_limit(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
            snap = restored.snapshot()
        self.assertEqual(snap["budget_limit_seconds"], 10800)

    def test_round_trip_paused_state(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("before save")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertTrue(restored.is_paused)

    def test_round_trip_running_state_restores_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertFalse(tracker.is_paused)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertTrue(restored.is_paused)

    def test_round_trip_with_accumulated_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(5):
            tracker.pause(f"p{i}")
            time.sleep(0.003)
            tracker.resume()
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            orig = tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
            snap = restored.snapshot()
        self.assertGreater(snap["total_pause_seconds"], 0)


class TestCheckpointTimestamps(unittest.TestCase):
    """Checkpoint timestamps are valid ISO 8601 UTC strings."""

    def test_started_at_is_iso(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = tracker.snapshot()
        dt = datetime.fromisoformat(snap["started_at"])
        self.assertIsNotNone(dt)

    def test_checkpoint_at_is_iso(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap = tracker.snapshot()
        dt = datetime.fromisoformat(snap["checkpoint_at"])
        self.assertIsNotNone(dt)

    def test_checkpoint_at_after_started_at(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.005)
        snap = tracker.snapshot()
        started = datetime.fromisoformat(snap["started_at"])
        checkpointed = datetime.fromisoformat(snap["checkpoint_at"])
        self.assertGreaterEqual(checkpointed, started)

    def test_successive_checkpoints_at_nondecreasing(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        stamps = []
        for _ in range(5):
            snap = tracker.snapshot()
            stamps.append(datetime.fromisoformat(snap["checkpoint_at"]))
            time.sleep(0.002)
        for i in range(1, len(stamps)):
            self.assertGreaterEqual(stamps[i], stamps[i - 1])

    def test_saved_file_timestamps_are_iso(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        datetime.fromisoformat(data["started_at"])
        datetime.fromisoformat(data["checkpoint_at"])


class TestCheckpointCrossTierRestore(unittest.TestCase):
    """Restore with explicit tier overrides the saved tier name."""

    def test_restore_with_explicit_tier(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
        self.assertEqual(restored.tier.name, "high")
        self.assertEqual(restored.tier.active_budget_seconds, 10800)

    def test_restore_with_custom_tier(self):
        custom = TierConfig("custom", 999, 30, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=custom)
        self.assertEqual(restored.tier.name, "custom")
        self.assertEqual(restored.tier.active_budget_seconds, 999)

    def test_cross_tier_restore_preserves_active_seconds(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            orig = tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
        self.assertAlmostEqual(
            restored.elapsed_active, orig["active_seconds"], delta=0.01,
        )

    def test_cross_tier_changes_remaining(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored_light = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            restored_high = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
        self.assertGreater(
            restored_high.remaining_active,
            restored_light.remaining_active,
        )


class TestAdjustmentLogVersioning(unittest.TestCase):
    """AdjustmentLog schema version checks."""

    def test_saved_log_has_correct_version(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "criteria_adjustment_log_v1")

    def test_load_rejects_wrong_version(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["schema_version"] = "criteria_adjustment_log_v999"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)

    def test_load_rejects_missing_version(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            del data["schema_version"]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)

    def test_round_trip_preserves_entries(self):
        log = AdjustmentLog()
        for i in range(3):
            log.record(CriteriaAdjustment(
                f"c{i}", {"v": i}, {"v": i + 1}, f"reason {i}",
            ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(len(loaded.entries), 3)
        for i, entry in enumerate(loaded.entries):
            self.assertEqual(entry.criterion_id, f"c{i}")

    def test_empty_log_round_trip(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(len(loaded.entries), 0)

    def test_saved_at_is_iso_timestamp(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        datetime.fromisoformat(data["saved_at"])


if __name__ == "__main__":
    unittest.main()
