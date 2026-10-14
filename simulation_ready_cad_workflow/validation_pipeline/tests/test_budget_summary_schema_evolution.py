"""Summary schema evolution and edge case tests.

Validates build_incomplete_summary output across different tracker
states, tier combinations, and unusual input patterns to ensure
the schema stays consistent and fields are always present.
"""
from __future__ import annotations

import time
import unittest

from ..slim_harness.budget import (
    TIERS,
    ActiveBudgetTracker,
    BudgetStatus,
    TierConfig,
    build_incomplete_summary,
    get_tier,
)


class TestSummaryWithPausedTracker(unittest.TestCase):
    """Summary generated while tracker is paused."""

    def test_paused_tracker_summary_valid(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("generate summary")
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIn("schema_version", summary)
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_paused_timing_reflects_frozen_active(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        tracker.pause("freeze")
        time.sleep(0.02)
        s1 = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        time.sleep(0.02)
        s2 = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertAlmostEqual(
            s1["timing"]["active_seconds"],
            s2["timing"]["active_seconds"],
            places=2,
        )


class TestSummaryWithExhaustedTracker(unittest.TestCase):
    """Summary from an exhausted tracker."""

    def test_exhausted_status_in_summary(self):
        tier = TierConfig("tiny", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=["remaining"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIn(summary["status"], {s.value for s in BudgetStatus})

    def test_exhausted_timing_shows_overbudget(self):
        tier = TierConfig("tiny", 0.01, 5, 1, 0, 3)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.03)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertGreater(
            summary["timing"]["active_seconds"],
            summary["timing"]["budget_limit_seconds"],
        )


class TestSummaryMixedResults(unittest.TestCase):
    """Summary with mixed pass/fail/pending combinations."""

    def _make_summary(self, completed=None, pending=None, issues=None,
                      blockers=None, evidence=None):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed or [],
            pending_tests=pending or [],
            issues=issues or [],
            evidence_paths=evidence or [],
            blockers=blockers or [],
        )

    def test_all_pass_severity_minor(self):
        completed = [
            {"test_id": f"t{i}", "physical_pass": True,
             "geometry_audit": {"passed": True}}
            for i in range(10)
        ]
        s = self._make_summary(completed=completed)
        self.assertEqual(s["severity"], "minor")

    def test_one_fail_severity_functional(self):
        completed = [
            {"test_id": "pass", "physical_pass": True,
             "geometry_audit": {"passed": True}},
            {"test_id": "fail", "physical_pass": False,
             "geometry_audit": {"passed": True}},
        ]
        s = self._make_summary(completed=completed)
        self.assertEqual(s["severity"], "functional")

    def test_with_issues_and_blockers(self):
        s = self._make_summary(
            issues=["issue1", "issue2"],
            blockers=["blocker1"],
        )
        self.assertEqual(len(s["issues"]), 2)
        self.assertEqual(len(s["blockers"]), 1)

    def test_with_evidence_paths(self):
        s = self._make_summary(
            evidence=["/path/a.json", "/path/b.json"],
        )
        self.assertEqual(len(s["evidence_paths"]), 2)

    def test_completed_and_pending(self):
        completed = [
            {"test_id": "done", "physical_pass": True,
             "geometry_audit": {"passed": True}},
        ]
        s = self._make_summary(completed=completed, pending=["todo"])
        self.assertEqual(len(s["completed_tests"]), 1)
        self.assertEqual(len(s["pending_tests"]), 1)


class TestSummaryAllTiers(unittest.TestCase):
    """Summary generates correctly for all production tiers."""

    def test_each_tier_produces_valid_summary(self):
        for name in ("light", "medium", "high"):
            tier = get_tier(name)
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            self.assertEqual(summary["tier"]["name"], name)
            self.assertEqual(
                summary["timing"]["budget_limit_seconds"],
                tier.active_budget_seconds,
            )

    def test_custom_tier_summary(self):
        tier = TierConfig("exotic", 42, 7, 3, 1, 9)
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["tier"]["name"], "exotic")
        self.assertEqual(summary["timing"]["budget_limit_seconds"], 42)


class TestSummaryTimingInvariants(unittest.TestCase):
    """Timing section mathematical invariants."""

    def test_active_leq_wall(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertLessEqual(
            summary["timing"]["active_seconds"],
            summary["timing"]["wall_seconds"] + 0.001,
        )

    def test_pause_seconds_non_negative(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertGreaterEqual(summary["timing"]["total_pause_seconds"], 0)

    def test_pause_count_non_negative(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertGreaterEqual(summary["timing"]["pause_count"], 0)

    def test_budget_limit_matches_tier(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            self.assertEqual(
                summary["timing"]["budget_limit_seconds"],
                tier.active_budget_seconds,
            )


class TestSummaryGeneratedAt(unittest.TestCase):
    """generated_at timestamp is present and ISO-formatted."""

    def test_generated_at_present(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIn("generated_at", summary)
        self.assertIsInstance(summary["generated_at"], str)
        self.assertTrue(len(summary["generated_at"]) > 10)

    def test_generated_at_contains_t(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertIn("T", summary["generated_at"])


if __name__ == "__main__":
    unittest.main()
