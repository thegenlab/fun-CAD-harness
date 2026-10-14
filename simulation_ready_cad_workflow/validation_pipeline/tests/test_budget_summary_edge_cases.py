"""Budget summary generation edge case tests.

Validates build_incomplete_summary across boundary conditions: empty inputs,
large test lists, mixed pass/fail combinations, timing edge cases, and
output schema consistency.
"""
from __future__ import annotations

import time
import unittest

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    BudgetStatus,
    build_incomplete_summary,
    get_tier,
)


class TestSummarySchemaConsistency(unittest.TestCase):
    """Every summary has the required schema fields."""

    REQUIRED_KEYS = {
        "schema_version", "status", "tier", "timing", "completed_tests",
        "pending_tests", "severity", "issues", "evidence_paths",
        "blockers", "generated_at",
    }
    TIMING_KEYS = {
        "active_seconds", "wall_seconds", "budget_limit_seconds",
        "pause_count", "total_pause_seconds",
    }

    def _make_summary(self, tier_name="light", **kwargs):
        tier = get_tier(tier_name)
        tracker = ActiveBudgetTracker(tier=tier)
        defaults = dict(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        defaults.update(kwargs)
        return build_incomplete_summary(**defaults)

    def test_all_required_keys_present(self):
        for name in ("light", "medium", "high"):
            summary = self._make_summary(name)
            self.assertEqual(set(summary.keys()), self.REQUIRED_KEYS)

    def test_timing_keys_present(self):
        summary = self._make_summary()
        self.assertEqual(set(summary["timing"].keys()), self.TIMING_KEYS)

    def test_schema_version_correct(self):
        summary = self._make_summary()
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_status_is_valid_budget_status(self):
        summary = self._make_summary()
        self.assertIn(summary["status"], {s.value for s in BudgetStatus})


class TestSummaryEmptyInputs(unittest.TestCase):
    """Summary handles all-empty inputs correctly."""

    def _make_summary(self, **kwargs):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        defaults = dict(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        defaults.update(kwargs)
        return build_incomplete_summary(**defaults)

    def test_no_tests_severity_minor(self):
        summary = self._make_summary()
        self.assertEqual(summary["severity"], "minor")

    def test_no_tests_completed_list_empty(self):
        summary = self._make_summary()
        self.assertEqual(summary["completed_tests"], [])

    def test_no_tests_pending_list_empty(self):
        summary = self._make_summary()
        self.assertEqual(summary["pending_tests"], [])

    def test_empty_issues_and_blockers(self):
        summary = self._make_summary()
        self.assertEqual(summary["issues"], [])
        self.assertEqual(summary["blockers"], [])

    def test_empty_evidence_paths(self):
        summary = self._make_summary()
        self.assertEqual(summary["evidence_paths"], [])


class TestSummarySeverityLogic(unittest.TestCase):
    """Severity classification across pass/fail/pending combinations."""

    def _make_summary(self, completed=None, pending=None):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed or [],
            pending_tests=pending or [],
            issues=[], evidence_paths=[], blockers=[],
        )

    def test_all_passing_no_pending_is_minor(self):
        completed = [
            {"test_id": "t1", "physical_pass": True,
             "geometry_audit": {"passed": True}},
            {"test_id": "t2", "physical_pass": True,
             "geometry_audit": {"passed": True}},
        ]
        summary = self._make_summary(completed=completed)
        self.assertEqual(summary["severity"], "minor")

    def test_one_failure_is_functional(self):
        completed = [
            {"test_id": "t1", "physical_pass": True,
             "geometry_audit": {"passed": True}},
            {"test_id": "t2", "physical_pass": False,
             "geometry_audit": {"passed": True}},
        ]
        summary = self._make_summary(completed=completed)
        self.assertEqual(summary["severity"], "functional")

    def test_geometry_failure_is_functional(self):
        completed = [
            {"test_id": "t1", "physical_pass": True,
             "geometry_audit": {"passed": False}},
        ]
        summary = self._make_summary(completed=completed)
        self.assertEqual(summary["severity"], "functional")

    def test_pending_only_is_functional(self):
        summary = self._make_summary(pending=["t1"])
        self.assertEqual(summary["severity"], "functional")

    def test_mixed_pass_fail_pending_is_functional(self):
        completed = [
            {"test_id": "t1", "physical_pass": True,
             "geometry_audit": {"passed": True}},
        ]
        summary = self._make_summary(completed=completed, pending=["t2"])
        self.assertEqual(summary["severity"], "functional")

    def test_all_failed_is_functional(self):
        completed = [
            {"test_id": f"t{i}", "physical_pass": False,
             "geometry_audit": {"passed": False}}
            for i in range(5)
        ]
        summary = self._make_summary(completed=completed)
        self.assertEqual(summary["severity"], "functional")


class TestSummaryCompletedTestFormat(unittest.TestCase):
    """Completed test entries have the expected structure."""

    def _make_summary(self, completed):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )

    def test_passing_test_marked_pass(self):
        completed = [
            {"test_id": "pass1", "physical_pass": True,
             "geometry_audit": {"passed": True}},
        ]
        summary = self._make_summary(completed)
        self.assertEqual(summary["completed_tests"][0]["status"], "PASS")
        self.assertEqual(summary["completed_tests"][0]["id"], "pass1")

    def test_failing_test_marked_fail(self):
        completed = [
            {"test_id": "fail1", "physical_pass": False,
             "geometry_audit": {"passed": True}},
        ]
        summary = self._make_summary(completed)
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_geometry_fail_marked_fail(self):
        completed = [
            {"test_id": "gfail", "physical_pass": True,
             "geometry_audit": {"passed": False}},
        ]
        summary = self._make_summary(completed)
        self.assertEqual(summary["completed_tests"][0]["status"], "FAIL")

    def test_multiple_tests_order_preserved(self):
        completed = [
            {"test_id": f"t{i}", "physical_pass": True,
             "geometry_audit": {"passed": True}}
            for i in range(10)
        ]
        summary = self._make_summary(completed)
        for i, entry in enumerate(summary["completed_tests"]):
            self.assertEqual(entry["id"], f"t{i}")

    def test_missing_test_id_defaults_unknown(self):
        completed = [
            {"physical_pass": True, "geometry_audit": {"passed": True}},
        ]
        summary = self._make_summary(completed)
        self.assertEqual(summary["completed_tests"][0]["id"], "unknown")


class TestSummaryTimingValues(unittest.TestCase):
    """Timing section values are non-negative and consistent."""

    def _make_summary_with_delay(self, delay=0.01):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(delay)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )

    def test_active_seconds_positive_after_delay(self):
        summary = self._make_summary_with_delay()
        self.assertGreater(summary["timing"]["active_seconds"], 0)

    def test_wall_seconds_positive_after_delay(self):
        summary = self._make_summary_with_delay()
        self.assertGreater(summary["timing"]["wall_seconds"], 0)

    def test_budget_limit_matches_tier(self):
        summary = self._make_summary_with_delay()
        self.assertEqual(summary["timing"]["budget_limit_seconds"], 7200)

    def test_pause_count_zero_when_no_pauses(self):
        summary = self._make_summary_with_delay()
        self.assertEqual(summary["timing"]["pause_count"], 0)

    def test_total_pause_seconds_zero_when_no_pauses(self):
        summary = self._make_summary_with_delay()
        self.assertAlmostEqual(summary["timing"]["total_pause_seconds"], 0.0, places=3)

    def test_timing_with_pauses(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("p")
        time.sleep(0.01)
        tracker.resume()
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertGreater(summary["timing"]["total_pause_seconds"], 0)
        self.assertEqual(summary["timing"]["pause_count"], 1)


class TestSummaryLargeInputs(unittest.TestCase):
    """Summary handles large test lists without errors."""

    def test_many_completed_tests(self):
        completed = [
            {"test_id": f"t{i}", "physical_pass": i % 2 == 0,
             "geometry_audit": {"passed": i % 3 != 0}}
            for i in range(100)
        ]
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(len(summary["completed_tests"]), 100)
        self.assertEqual(summary["severity"], "functional")

    def test_many_pending_tests(self):
        pending = [f"pending_{i}" for i in range(200)]
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=pending,
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(len(summary["pending_tests"]), 200)
        self.assertEqual(summary["severity"], "functional")

    def test_many_issues_and_blockers(self):
        issues = [f"issue_{i}" for i in range(50)]
        blockers = [f"blocker_{i}" for i in range(30)]
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=issues, evidence_paths=[],
            blockers=blockers,
        )
        self.assertEqual(len(summary["issues"]), 50)
        self.assertEqual(len(summary["blockers"]), 30)


class TestSummaryTierInfo(unittest.TestCase):
    """Summary tier section matches the input tier."""

    def test_tier_name_in_summary(self):
        for name in ("light", "medium", "high"):
            tier = get_tier(name)
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            self.assertEqual(summary["tier"]["name"], name)

    def test_tier_budget_in_summary(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(
            summary["tier"]["active_budget_seconds"], 10800,
        )

    def test_custom_tier_in_summary(self):
        tier = TierConfig("custom", 999, 30, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["tier"]["name"], "custom")
        self.assertEqual(summary["tier"]["active_budget_seconds"], 999)


if __name__ == "__main__":
    unittest.main()
