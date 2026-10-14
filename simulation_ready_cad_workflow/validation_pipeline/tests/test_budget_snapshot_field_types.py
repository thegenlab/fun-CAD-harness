"""Snapshot field type and value invariant tests.

Verifies that every snapshot() call returns a dict with exactly
the expected 11 fields, correct types for each, and mathematically
consistent values across all tracker states.
"""
from __future__ import annotations

import math
import time
import unittest

from ..slim_harness.budget import (
    ActiveBudgetTracker,
    TierConfig,
    get_tier,
)


SNAPSHOT_FIELDS = {
    "schema_version", "tier", "active_seconds", "wall_seconds",
    "paused", "pause_count", "total_pause_seconds",
    "budget_limit_seconds", "remaining_active_seconds",
    "started_at", "checkpoint_at",
}


class TestSnapshotFieldPresence(unittest.TestCase):
    """Snapshot always has exactly 11 fields."""

    def test_fresh_tracker(self):
        t = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertEqual(set(t.snapshot().keys()), SNAPSHOT_FIELDS)

    def test_after_work(self):
        t = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        self.assertEqual(set(t.snapshot().keys()), SNAPSHOT_FIELDS)

    def test_paused(self):
        t = ActiveBudgetTracker(tier=get_tier("high"))
        t.pause("p")
        self.assertEqual(set(t.snapshot().keys()), SNAPSHOT_FIELDS)

    def test_after_resume(self):
        t = ActiveBudgetTracker(tier=get_tier("light"))
        t.pause("p")
        t.resume()
        self.assertEqual(set(t.snapshot().keys()), SNAPSHOT_FIELDS)

    def test_exhausted(self):
        t = ActiveBudgetTracker(tier=TierConfig("x", 0.01, 5, 1, 0, 3))
        time.sleep(0.02)
        self.assertEqual(set(t.snapshot().keys()), SNAPSHOT_FIELDS)


class TestSnapshotFieldTypes(unittest.TestCase):
    """Each field has the expected Python type."""

    def _snap(self):
        t = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        return t.snapshot()

    def test_schema_version_str(self):
        self.assertIsInstance(self._snap()["schema_version"], str)

    def test_tier_str(self):
        self.assertIsInstance(self._snap()["tier"], str)

    def test_active_seconds_float(self):
        self.assertIsInstance(self._snap()["active_seconds"], float)

    def test_wall_seconds_float(self):
        self.assertIsInstance(self._snap()["wall_seconds"], float)

    def test_paused_bool(self):
        self.assertIsInstance(self._snap()["paused"], bool)

    def test_pause_count_int(self):
        self.assertIsInstance(self._snap()["pause_count"], int)

    def test_total_pause_seconds_float(self):
        self.assertIsInstance(self._snap()["total_pause_seconds"], (int, float))

    def test_budget_limit_float(self):
        self.assertIsInstance(self._snap()["budget_limit_seconds"], (int, float))

    def test_remaining_active_float(self):
        self.assertIsInstance(self._snap()["remaining_active_seconds"], float)

    def test_started_at_str(self):
        self.assertIsInstance(self._snap()["started_at"], str)

    def test_checkpoint_at_str(self):
        self.assertIsInstance(self._snap()["checkpoint_at"], str)


class TestSnapshotSchemaVersion(unittest.TestCase):
    """Schema version is always budget_checkpoint_v1."""

    def test_fresh(self):
        t = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertEqual(t.snapshot()["schema_version"], "budget_checkpoint_v1")

    def test_after_operations(self):
        t = ActiveBudgetTracker(tier=get_tier("high"))
        t.pause("p")
        t.resume()
        time.sleep(0.01)
        self.assertEqual(t.snapshot()["schema_version"], "budget_checkpoint_v1")


class TestSnapshotNumericInvariants(unittest.TestCase):
    """Numeric fields are finite and non-negative."""

    def _snap(self):
        t = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        t.pause("p")
        time.sleep(0.01)
        t.resume()
        return t.snapshot()

    def test_active_non_negative(self):
        self.assertGreaterEqual(self._snap()["active_seconds"], 0)

    def test_wall_non_negative(self):
        self.assertGreaterEqual(self._snap()["wall_seconds"], 0)

    def test_active_finite(self):
        self.assertTrue(math.isfinite(self._snap()["active_seconds"]))

    def test_wall_finite(self):
        self.assertTrue(math.isfinite(self._snap()["wall_seconds"]))

    def test_remaining_non_negative(self):
        self.assertGreaterEqual(self._snap()["remaining_active_seconds"], 0)

    def test_remaining_finite(self):
        self.assertTrue(math.isfinite(self._snap()["remaining_active_seconds"]))

    def test_pause_count_non_negative(self):
        self.assertGreaterEqual(self._snap()["pause_count"], 0)

    def test_total_pause_non_negative(self):
        self.assertGreaterEqual(self._snap()["total_pause_seconds"], 0)

    def test_budget_limit_positive(self):
        self.assertGreater(self._snap()["budget_limit_seconds"], 0)


class TestSnapshotTimestamps(unittest.TestCase):
    """Timestamp strings look like ISO format."""

    def test_started_at_has_t(self):
        t = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertIn("T", t.snapshot()["started_at"])

    def test_checkpoint_at_has_t(self):
        t = ActiveBudgetTracker(tier=get_tier("medium"))
        self.assertIn("T", t.snapshot()["checkpoint_at"])

    def test_started_at_length(self):
        t = ActiveBudgetTracker(tier=get_tier("high"))
        self.assertGreater(len(t.snapshot()["started_at"]), 15)

    def test_checkpoint_at_length(self):
        t = ActiveBudgetTracker(tier=get_tier("light"))
        self.assertGreater(len(t.snapshot()["checkpoint_at"]), 15)


class TestSnapshotTierReflection(unittest.TestCase):
    """Tier name and budget limit match the tier used for construction."""

    def test_light(self):
        t = ActiveBudgetTracker(tier=get_tier("light"))
        snap = t.snapshot()
        self.assertEqual(snap["tier"], "light")
        self.assertEqual(snap["budget_limit_seconds"], 3600)

    def test_medium(self):
        t = ActiveBudgetTracker(tier=get_tier("medium"))
        snap = t.snapshot()
        self.assertEqual(snap["tier"], "medium")
        self.assertEqual(snap["budget_limit_seconds"], 7200)

    def test_high(self):
        t = ActiveBudgetTracker(tier=get_tier("high"))
        snap = t.snapshot()
        self.assertEqual(snap["tier"], "high")
        self.assertEqual(snap["budget_limit_seconds"], 10800)

    def test_custom(self):
        t = ActiveBudgetTracker(tier=TierConfig("c", 42, 7, 3, 1, 9))
        snap = t.snapshot()
        self.assertEqual(snap["tier"], "c")
        self.assertEqual(snap["budget_limit_seconds"], 42)


if __name__ == "__main__":
    unittest.main()
