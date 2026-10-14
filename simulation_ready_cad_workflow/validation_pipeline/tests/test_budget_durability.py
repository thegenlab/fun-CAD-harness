"""Durability and robustness tests for budget checkpoint and tracking.

Tests failure modes that occur in real deployment:
- Corrupted checkpoint files
- Partial writes and interrupted saves
- Filesystem edge cases
- Concurrent checkpoint writes from threads
- Rapid tracker lifecycle churn
- Adjustment log corruption recovery
- Budget tracker under sustained load
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
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
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
)


class TestCorruptedCheckpointRecovery(unittest.TestCase):
    """Budget tracker handles corrupted checkpoint files gracefully."""

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("", encoding="utf-8")
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                ActiveBudgetTracker.restore(path, tier=get_tier("light"))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                ActiveBudgetTracker.restore(path, tier=get_tier("light"))

    def test_wrong_schema_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong_schema.json"
            data = {
                "schema_version": "unknown_v99",
                "tier": "light",
                "active_seconds": 10.0,
                "budget_limit_seconds": 3600,
            }
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                ActiveBudgetTracker.restore(path, tier=get_tier("light"))
            self.assertIn("schema", str(ctx.exception).lower())

    def test_missing_active_seconds_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_key.json"
            data = {
                "schema_version": "budget_checkpoint_v1",
                "tier": "light",
                "budget_limit_seconds": 3600,
            }
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(KeyError):
                ActiveBudgetTracker.restore(path, tier=get_tier("light"))

    def test_negative_active_seconds_still_loads(self):
        """Restore doesn't validate value ranges — just loads data."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "negative.json"
            data = {
                "schema_version": "budget_checkpoint_v1",
                "tier": "light",
                "active_seconds": -5.0,
                "budget_limit_seconds": 3600,
                "pause_count": 0,
                "total_pause_seconds": 0.0,
            }
            path.write_text(json.dumps(data), encoding="utf-8")
            tracker = ActiveBudgetTracker.restore(path, tier=get_tier("light"))
            self.assertLess(tracker._active_seconds, 0)

    def test_truncated_json_raises(self):
        """A checkpoint file cut mid-write is detected."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truncated.json"
            tracker.save_checkpoint(path)
            content = path.read_text(encoding="utf-8")
            path.write_text(content[:len(content) // 2], encoding="utf-8")
            with self.assertRaises((json.JSONDecodeError, ValueError, KeyError)):
                ActiveBudgetTracker.restore(path, tier=tier)


class TestCheckpointFilesystemEdgeCases(unittest.TestCase):
    """Checkpoint save/restore with filesystem edge cases."""

    def test_save_creates_parent_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep_path = Path(tmp) / "a" / "b" / "c" / "checkpoint.json"
            tier = get_tier("medium")
            tracker = ActiveBudgetTracker(tier=tier)
            tracker.save_checkpoint(deep_path)
            self.assertTrue(deep_path.exists())
            restored = ActiveBudgetTracker.restore(deep_path, tier=tier)
            self.assertAlmostEqual(restored.elapsed_active,
                                   tracker.elapsed_active, delta=0.05)

    def test_save_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tier = get_tier("light")

            t1 = ActiveBudgetTracker(tier=tier)
            time.sleep(0.02)
            t1.save_checkpoint(path)
            first_active = t1.elapsed_active

            t2 = ActiveBudgetTracker(tier=tier)
            time.sleep(0.05)
            t2.save_checkpoint(path)
            second_active = t2.elapsed_active

            restored = ActiveBudgetTracker.restore(path, tier=tier)
            self.assertAlmostEqual(restored.elapsed_active, second_active,
                                   delta=0.02)
            self.assertGreater(second_active, first_active)

    def test_nonexistent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            ActiveBudgetTracker.restore(
                Path("/nonexistent/path/checkpoint.json"),
                tier=get_tier("light"),
            )

    def test_checkpoint_file_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "valid.json"
            tier = get_tier("high")
            tracker = ActiveBudgetTracker(tier=tier)
            tracker.pause("test")
            time.sleep(0.01)
            tracker.resume()
            tracker.save_checkpoint(path)

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIsInstance(data, dict)
            self.assertEqual(data["schema_version"], "budget_checkpoint_v1")
            self.assertIsInstance(data["active_seconds"], float)

    def test_checkpoint_is_human_readable(self):
        """Checkpoint uses indented JSON for debugging."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "readable.json"
            tracker = ActiveBudgetTracker(tier=get_tier("light"))
            tracker.save_checkpoint(path)
            content = path.read_text(encoding="utf-8")
            self.assertIn("\n", content)
            self.assertIn("  ", content)


class TestConcurrentCheckpointWrites(unittest.TestCase):
    """Multiple threads writing checkpoints concurrently."""

    def test_concurrent_saves_dont_corrupt(self):
        """10 threads saving to different files produce valid checkpoints."""
        tier = get_tier("medium")
        errors = []

        def save_worker(thread_id, tmp_dir):
            try:
                tracker = ActiveBudgetTracker(tier=tier)
                time.sleep(0.01 * (thread_id % 5))
                path = Path(tmp_dir) / f"cp_{thread_id}.json"
                tracker.save_checkpoint(path)
                restored = ActiveBudgetTracker.restore(path, tier=tier)
                assert restored.tier.name == "medium"
            except Exception as e:
                errors.append((thread_id, e))

        with tempfile.TemporaryDirectory() as tmp:
            threads = [
                threading.Thread(target=save_worker, args=(i, tmp))
                for i in range(10)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_saves_to_same_file(self):
        """Multiple threads saving to the same file — last write wins."""
        tier = get_tier("high")
        errors = []

        def save_worker(thread_id, path):
            try:
                tracker = ActiveBudgetTracker(tier=tier)
                time.sleep(0.005 * thread_id)
                tracker.save_checkpoint(path)
            except Exception as e:
                errors.append((thread_id, e))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.json"
            threads = [
                threading.Thread(target=save_worker, args=(i, path))
                for i in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=30)

            self.assertEqual(errors, [])
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "budget_checkpoint_v1")


class TestRapidTrackerLifecycle(unittest.TestCase):
    """Rapid creation and destruction of trackers."""

    def test_100_trackers_created_and_discarded(self):
        """Creating 100 trackers in sequence doesn't leak resources."""
        for i in range(100):
            tier_name = ["light", "medium", "high"][i % 3]
            tracker = ActiveBudgetTracker(tier=get_tier(tier_name))
            snap = tracker.snapshot()
            self.assertEqual(snap["tier"], tier_name)

    def test_rapid_pause_resume_1000_cycles(self):
        """1000 pause/resume cycles on a single tracker."""
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(1000):
            tracker.pause(f"cycle_{i}")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 1000)
        self.assertGreater(snap["active_seconds"], 0)

    def test_alternating_tier_trackers(self):
        """Create trackers alternating between tiers rapidly."""
        trackers = []
        for i in range(30):
            tier_name = ["light", "medium", "high"][i % 3]
            t = ActiveBudgetTracker(tier=get_tier(tier_name))
            t.pause(f"init_{i}")
            t.resume()
            trackers.append(t)

        for t in trackers:
            snap = t.snapshot()
            self.assertGreaterEqual(snap["active_seconds"], 0)
            self.assertGreater(snap["wall_seconds"], 0)


class TestAdjustmentLogCorruption(unittest.TestCase):
    """Adjustment log handles corrupted files gracefully."""

    def test_empty_log_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("", encoding="utf-8")
            with self.assertRaises((json.JSONDecodeError, ValueError)):
                AdjustmentLog.load(path)

    def test_wrong_schema_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrong.json"
            path.write_text(json.dumps({
                "schema_version": "wrong_v1",
                "adjustments": [],
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                AdjustmentLog.load(path)

    def test_empty_adjustments_list_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty_list.json"
            path.write_text(json.dumps({
                "schema_version": "criteria_adjustment_log_v1",
                "adjustments": [],
                "saved_at": "2026-01-01T00:00:00Z",
            }), encoding="utf-8")
            log = AdjustmentLog.load(path)
            self.assertEqual(len(log.entries), 0)

    def test_log_save_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / "x" / "y" / "log.json"
            log = AdjustmentLog()
            log.record(CriteriaAdjustment("c", {"a": 1}, {"a": 2}, "test"))
            log.save(deep)
            self.assertTrue(deep.exists())
            loaded = AdjustmentLog.load(deep)
            self.assertEqual(len(loaded.entries), 1)


class TestBudgetStatusEnumDurability(unittest.TestCase):
    """BudgetStatus enum is durable for serialization."""

    def test_all_values_are_uppercase(self):
        for status in BudgetStatus:
            self.assertEqual(status.value, status.value.upper())

    def test_status_from_string(self):
        for status in BudgetStatus:
            reconstructed = BudgetStatus(status.value)
            self.assertEqual(reconstructed, status)

    def test_unknown_status_raises(self):
        with self.assertRaises(ValueError):
            BudgetStatus("UNKNOWN_STATUS")

    def test_status_json_round_trip(self):
        for status in BudgetStatus:
            encoded = json.dumps({"status": status.value})
            decoded = json.loads(encoded)
            self.assertEqual(BudgetStatus(decoded["status"]), status)


class TestSummaryDurability(unittest.TestCase):
    """build_incomplete_summary produces durable, serializable output."""

    def test_summary_survives_json_round_trip(self):
        for tier_name in TIERS:
            tier = get_tier(tier_name)
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[{"test_id": "t1", "physical_pass": True,
                                  "geometry_audit": {"passed": True}}],
                pending_tests=["t2", "t3"],
                issues=["timeout"],
                evidence_paths=["/path/to/evidence.json"],
                blockers=["memory"],
            )
            encoded = json.dumps(summary, indent=2)
            decoded = json.loads(encoded)

            self.assertEqual(decoded["schema_version"], "budget_incomplete_summary_v1")
            self.assertEqual(decoded["status"], "ACTIVE_BUDGET_EXHAUSTED")
            self.assertEqual(len(decoded["completed_tests"]), 1)
            self.assertEqual(len(decoded["pending_tests"]), 2)
            self.assertEqual(decoded["tier"]["name"], tier_name)

    def test_summary_with_unicode_issues(self):
        """Summary handles Unicode strings in issues and evidence paths."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[],
            pending_tests=[],
            issues=["超时 — simulation timed out", "エラー: memory exceeded"],
            evidence_paths=["/path/目录/evidence.json"],
            blockers=["内存不足"],
        )
        encoded = json.dumps(summary, ensure_ascii=False)
        decoded = json.loads(encoded)
        self.assertIn("超时", decoded["issues"][0])
        self.assertIn("目录", decoded["evidence_paths"][0])

    def test_summary_with_very_long_pending_list(self):
        """Summary handles large numbers of pending tests."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        pending = [f"test_{i}" for i in range(1000)]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[],
            pending_tests=pending,
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(len(summary["pending_tests"]), 1000)
        self.assertEqual(summary["severity"], "functional")


class TestRecoveryMechanismDurability(unittest.TestCase):
    """Recovery helpers work under edge conditions."""

    def test_recovery_delay_with_zero_attempt(self):
        self.assertEqual(recovery_delay(0), 5.0)

    def test_should_retry_with_empty_error(self):
        self.assertTrue(should_retry(0, ""))

    def test_should_retry_with_compound_error(self):
        """Error string containing multiple keywords."""
        self.assertFalse(should_retry(0, "PASS after ACTIVE_BUDGET_EXHAUSTED"))

    def test_should_retry_case_sensitive(self):
        """Non-retryable check is case-sensitive."""
        self.assertTrue(should_retry(0, "pass"))
        self.assertTrue(should_retry(0, "active_budget_exhausted"))
        self.assertFalse(should_retry(0, "PASS"))
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_max_recovery_retries_is_positive(self):
        self.assertGreater(MAX_RECOVERY_RETRIES, 0)
        self.assertIsInstance(MAX_RECOVERY_RETRIES, int)


if __name__ == "__main__":
    unittest.main()
