"""Checkpoint chain integrity tests.

Validates multi-step checkpoint save/restore/modify/save chains,
ensuring active time accumulates correctly across restores, pause
counts carry forward, and no data loss through multiple cycles.
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    ActiveBudgetTracker,
    TierConfig,
    get_tier,
)


class TestCheckpointSaveRestoreSave(unittest.TestCase):
    """Save → restore → work → save cycle preserves state."""

    def test_active_time_accumulates_across_restore(self):
        tier = get_tier("medium")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t1 = ActiveBudgetTracker(tier=tier)
            time.sleep(0.02)
            s1 = t1.save_checkpoint(p)

            t2 = ActiveBudgetTracker.restore(p, tier)
            t2.resume()
            time.sleep(0.02)
            s2 = t2.save_checkpoint(p)

            self.assertGreater(s2["active_seconds"], s1["active_seconds"])

    def test_pause_count_carries_forward(self):
        tier = get_tier("light")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t1 = ActiveBudgetTracker(tier=tier)
            t1.pause("p1")
            t1.resume()
            t1.pause("p2")
            t1.resume()
            s1 = t1.save_checkpoint(p)
            self.assertEqual(s1["pause_count"], 2)

            t2 = ActiveBudgetTracker.restore(p, tier)
            s2 = t2.snapshot()
            self.assertGreaterEqual(s2["pause_count"], 2)

    def test_three_step_chain(self):
        tier = get_tier("high")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            active_times = []
            tracker = ActiveBudgetTracker(tier=tier)
            for i in range(3):
                time.sleep(0.01)
                snap = tracker.save_checkpoint(p)
                active_times.append(snap["active_seconds"])
                if i < 2:
                    tracker = ActiveBudgetTracker.restore(p, tier)
                    tracker.resume()

            for j in range(1, len(active_times)):
                self.assertGreater(active_times[j], active_times[j - 1])

    def test_checkpoint_file_overwritten_each_save(self):
        tier = get_tier("light")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t = ActiveBudgetTracker(tier=tier)
            t.save_checkpoint(p)
            size1 = p.stat().st_size
            time.sleep(0.01)
            t.save_checkpoint(p)
            size2 = p.stat().st_size
            self.assertGreater(size2, 0)
            self.assertAlmostEqual(size1, size2, delta=50)


class TestCheckpointCrossTierChain(unittest.TestCase):
    """Checkpoint restored with a different tier."""

    def test_light_to_medium(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t1 = ActiveBudgetTracker(tier=get_tier("light"))
            time.sleep(0.01)
            t1.save_checkpoint(p)

            t2 = ActiveBudgetTracker.restore(p, get_tier("medium"))
            s = t2.snapshot()
            self.assertEqual(s["tier"], "medium")
            self.assertGreater(s["active_seconds"], 0)

    def test_medium_to_high(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t1 = ActiveBudgetTracker(tier=get_tier("medium"))
            time.sleep(0.01)
            t1.save_checkpoint(p)

            t2 = ActiveBudgetTracker.restore(p, get_tier("high"))
            self.assertTrue(t2.is_paused)
            t2.resume()
            s = t2.snapshot()
            self.assertEqual(s["budget_limit_seconds"], 10800)

    def test_high_to_light_remaining_adjusts(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t1 = ActiveBudgetTracker(tier=get_tier("high"))
            time.sleep(0.01)
            s1 = t1.save_checkpoint(p)

            t2 = ActiveBudgetTracker.restore(p, get_tier("light"))
            s2 = t2.snapshot()
            self.assertEqual(s2["budget_limit_seconds"], 3600)
            self.assertLess(
                s2["remaining_active_seconds"],
                s1["remaining_active_seconds"],
            )

    def test_custom_tier_restore(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t1 = ActiveBudgetTracker(tier=get_tier("light"))
            time.sleep(0.01)
            t1.save_checkpoint(p)

            custom = TierConfig("custom", 500, 30, 1, 0, 5)
            t2 = ActiveBudgetTracker.restore(p, custom)
            s = t2.snapshot()
            self.assertEqual(s["tier"], "custom")
            self.assertEqual(s["budget_limit_seconds"], 500)


class TestCheckpointSchemaRoundTrip(unittest.TestCase):
    """Checkpoint file has required fields and valid JSON."""

    REQUIRED_FIELDS = {
        "schema_version", "tier", "active_seconds", "wall_seconds",
        "paused", "pause_count", "total_pause_seconds",
        "budget_limit_seconds", "remaining_active_seconds",
        "started_at", "checkpoint_at",
    }

    def test_all_required_fields_in_file(self):
        tier = get_tier("medium")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t = ActiveBudgetTracker(tier=tier)
            t.save_checkpoint(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(set(data.keys()), self.REQUIRED_FIELDS)

    def test_schema_version_correct(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t = ActiveBudgetTracker(tier=get_tier("high"))
            t.save_checkpoint(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_numeric_fields_finite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t = ActiveBudgetTracker(tier=get_tier("light"))
            time.sleep(0.01)
            t.save_checkpoint(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            for key in ("active_seconds", "wall_seconds",
                        "budget_limit_seconds", "remaining_active_seconds",
                        "total_pause_seconds"):
                self.assertIsInstance(data[key], (int, float))
                self.assertTrue(
                    data[key] == data[key] and data[key] != float("inf"),
                    f"{key} should be finite",
                )

    def test_boolean_paused_field(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t = ActiveBudgetTracker(tier=get_tier("medium"))
            t.save_checkpoint(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertIsInstance(data["paused"], bool)

    def test_timestamps_are_strings(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "ckpt.json"
            t = ActiveBudgetTracker(tier=get_tier("high"))
            t.save_checkpoint(p)
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertIsInstance(data["started_at"], str)
            self.assertIsInstance(data["checkpoint_at"], str)
            self.assertTrue(len(data["started_at"]) > 10)
            self.assertTrue(len(data["checkpoint_at"]) > 10)


class TestCheckpointMultipleTrackers(unittest.TestCase):
    """Multiple independent trackers saving to separate files."""

    def test_independent_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            trackers = {}
            for name in ("light", "medium", "high"):
                p = Path(td) / f"{name}.json"
                t = ActiveBudgetTracker(tier=get_tier(name))
                time.sleep(0.01)
                t.save_checkpoint(p)
                trackers[name] = p

            for name, path in trackers.items():
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["tier"], name)

    def test_restore_correct_tracker(self):
        with tempfile.TemporaryDirectory() as td:
            for name in ("light", "medium", "high"):
                p = Path(td) / f"{name}.json"
                t = ActiveBudgetTracker(tier=get_tier(name))
                t.save_checkpoint(p)

            restored = ActiveBudgetTracker.restore(
                Path(td) / "medium.json", get_tier("medium"),
            )
            self.assertEqual(restored.snapshot()["tier"], "medium")

    def test_interleaved_saves(self):
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.json"
            p2 = Path(td) / "b.json"
            t1 = ActiveBudgetTracker(tier=get_tier("light"))
            t2 = ActiveBudgetTracker(tier=get_tier("high"))
            time.sleep(0.01)
            t1.save_checkpoint(p1)
            time.sleep(0.01)
            t2.save_checkpoint(p2)
            t1.save_checkpoint(p1)

            d1 = json.loads(p1.read_text(encoding="utf-8"))
            d2 = json.loads(p2.read_text(encoding="utf-8"))
            self.assertEqual(d1["tier"], "light")
            self.assertEqual(d2["tier"], "high")


if __name__ == "__main__":
    unittest.main()
