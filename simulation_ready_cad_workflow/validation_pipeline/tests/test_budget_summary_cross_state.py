"""Summary generation across tracker states and unusual inputs.

Tests build_incomplete_summary output consistency when the tracker
is in various states (fresh, active, paused, exhausted) and with
edge-case completed_tests and pending_tests data.
"""
from __future__ import annotations

import time
import unittest

from ..slim_harness.budget import (
    ActiveBudgetTracker,
    BudgetStatus,
    TierConfig,
    build_incomplete_summary,
    get_tier,
)


def _make_summary(tier_name="medium", *, pause=False, exhaust=False,
                  completed=None, pending=None, issues=None,
                  evidence=None, blockers=None):
    tier = get_tier(tier_name) if isinstance(tier_name, str) else tier_name
    tracker = ActiveBudgetTracker(tier=tier)
    if exhaust:
        tiny = TierConfig("ex", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
    elif pause:
        time.sleep(0.01)
        tracker.pause("test")
    return build_incomplete_summary(
        tier=tier,
        tracker=tracker,
        completed_tests=completed or [],
        pending_tests=pending or [],
        issues=issues or [],
        evidence_paths=evidence or [],
        blockers=blockers or [],
    )


class TestSummarySchemaStability(unittest.TestCase):
    """Every summary has the same top-level keys."""

    REQUIRED_KEYS = {
        "schema_version", "status", "tier", "timing",
        "completed_tests", "pending_tests", "severity",
        "issues", "evidence_paths", "blockers", "generated_at",
    }

    def test_fresh_tracker(self):
        s = _make_summary()
        self.assertEqual(set(s.keys()), self.REQUIRED_KEYS)

    def test_paused_tracker(self):
        s = _make_summary(pause=True)
        self.assertEqual(set(s.keys()), self.REQUIRED_KEYS)

    def test_exhausted_tracker(self):
        s = _make_summary(exhaust=True)
        self.assertEqual(set(s.keys()), self.REQUIRED_KEYS)

    def test_with_all_inputs(self):
        completed = [{"test_id": "t1", "physical_pass": True,
                       "geometry_audit": {"passed": True}}]
        s = _make_summary(
            completed=completed, pending=["p1"],
            issues=["i1"], evidence=["/e"], blockers=["b1"],
        )
        self.assertEqual(set(s.keys()), self.REQUIRED_KEYS)


class TestSummarySeverityLogic(unittest.TestCase):
    """severity = 'functional' if any failed or pending, else 'minor'."""

    def test_no_tests_minor(self):
        s = _make_summary()
        self.assertEqual(s["severity"], "minor")

    def test_all_pass_no_pending_minor(self):
        completed = [
            {"test_id": f"t{i}", "physical_pass": True,
             "geometry_audit": {"passed": True}}
            for i in range(5)
        ]
        s = _make_summary(completed=completed)
        self.assertEqual(s["severity"], "minor")

    def test_one_fail_functional(self):
        completed = [
            {"test_id": "fail", "physical_pass": False,
             "geometry_audit": {"passed": True}},
        ]
        s = _make_summary(completed=completed)
        self.assertEqual(s["severity"], "functional")

    def test_geometry_fail_functional(self):
        completed = [
            {"test_id": "gfail", "physical_pass": True,
             "geometry_audit": {"passed": False}},
        ]
        s = _make_summary(completed=completed)
        self.assertEqual(s["severity"], "functional")

    def test_pending_only_functional(self):
        s = _make_summary(pending=["todo"])
        self.assertEqual(s["severity"], "functional")

    def test_pass_plus_pending_functional(self):
        completed = [
            {"test_id": "ok", "physical_pass": True,
             "geometry_audit": {"passed": True}},
        ]
        s = _make_summary(completed=completed, pending=["more"])
        self.assertEqual(s["severity"], "functional")


class TestSummaryTimingKeys(unittest.TestCase):
    """Timing section has exactly the expected keys."""

    TIMING_KEYS = {
        "active_seconds", "wall_seconds", "budget_limit_seconds",
        "pause_count", "total_pause_seconds",
    }

    def test_timing_keys_present(self):
        s = _make_summary()
        self.assertEqual(set(s["timing"].keys()), self.TIMING_KEYS)

    def test_paused_timing_keys(self):
        s = _make_summary(pause=True)
        self.assertEqual(set(s["timing"].keys()), self.TIMING_KEYS)

    def test_timing_values_numeric(self):
        s = _make_summary()
        for k in self.TIMING_KEYS:
            self.assertIsInstance(s["timing"][k], (int, float))


class TestSummaryCompletedTestFormat(unittest.TestCase):
    """completed_tests entries have id/physical_pass/geometry_pass/status."""

    def test_pass_entry(self):
        completed = [{"test_id": "t1", "physical_pass": True,
                       "geometry_audit": {"passed": True}}]
        s = _make_summary(completed=completed)
        entry = s["completed_tests"][0]
        self.assertEqual(entry["id"], "t1")
        self.assertTrue(entry["physical_pass"])
        self.assertTrue(entry["geometry_pass"])
        self.assertEqual(entry["status"], "PASS")

    def test_fail_entry(self):
        completed = [{"test_id": "t2", "physical_pass": False,
                       "geometry_audit": {"passed": True}}]
        s = _make_summary(completed=completed)
        entry = s["completed_tests"][0]
        self.assertEqual(entry["status"], "FAIL")

    def test_missing_test_id_defaults(self):
        completed = [{"physical_pass": True,
                       "geometry_audit": {"passed": True}}]
        s = _make_summary(completed=completed)
        self.assertEqual(s["completed_tests"][0]["id"], "unknown")

    def test_missing_geometry_audit(self):
        completed = [{"test_id": "t3", "physical_pass": True}]
        s = _make_summary(completed=completed)
        entry = s["completed_tests"][0]
        self.assertFalse(entry["geometry_pass"])

    def test_multiple_entries_order_preserved(self):
        completed = [
            {"test_id": f"t{i}", "physical_pass": True,
             "geometry_audit": {"passed": True}}
            for i in range(5)
        ]
        s = _make_summary(completed=completed)
        ids = [e["id"] for e in s["completed_tests"]]
        self.assertEqual(ids, [f"t{i}" for i in range(5)])


class TestSummaryTierInfo(unittest.TestCase):
    """Summary tier section reflects the provided tier."""

    def test_tier_name_light(self):
        s = _make_summary("light")
        self.assertEqual(s["tier"]["name"], "light")

    def test_tier_name_high(self):
        s = _make_summary("high")
        self.assertEqual(s["tier"]["name"], "high")

    def test_tier_budget_matches(self):
        tier = get_tier("medium")
        s = _make_summary("medium")
        self.assertEqual(
            s["timing"]["budget_limit_seconds"],
            tier.active_budget_seconds,
        )

    def test_custom_tier(self):
        custom = TierConfig("special", 999, 42, 7, 2, 50)
        s = _make_summary(custom)
        self.assertEqual(s["tier"]["name"], "special")
        self.assertEqual(s["timing"]["budget_limit_seconds"], 999)


class TestSummaryStatusField(unittest.TestCase):
    """Status field is always a valid BudgetStatus value string."""

    def test_fresh_status(self):
        s = _make_summary()
        valid = {st.value for st in BudgetStatus}
        self.assertIn(s["status"], valid)

    def test_exhausted_status(self):
        s = _make_summary(exhaust=True)
        self.assertEqual(s["status"], BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.value)

    def test_paused_status(self):
        s = _make_summary(pause=True)
        valid = {st.value for st in BudgetStatus}
        self.assertIn(s["status"], valid)


if __name__ == "__main__":
    unittest.main()
