"""BudgetStatus enum completeness and classification tests.

Validates that every BudgetStatus member is properly classified by
is_terminal(), and cross-checks against should_retry and recovery
logic to ensure consistent semantics.
"""
from __future__ import annotations

import unittest

from ..slim_harness.budget import (
    BudgetStatus,
    should_retry,
)


ALL_STATUSES = list(BudgetStatus)
TERMINAL = {BudgetStatus.PASS, BudgetStatus.ACTIVE_BUDGET_EXHAUSTED,
            BudgetStatus.INFRASTRUCTURE_FAILURE}
NON_TERMINAL = set(ALL_STATUSES) - TERMINAL


class TestBudgetStatusMembers(unittest.TestCase):
    """Enum membership and count."""

    def test_exactly_eight_members(self):
        self.assertEqual(len(ALL_STATUSES), 8)

    def test_expected_members(self):
        expected = {
            "PASS", "NEEDS_REVISION", "HARNESS_ERROR",
            "SIMULATION_TIMEOUT", "INFRASTRUCTURE_FAILURE",
            "ACTIVE_BUDGET_EXHAUSTED", "INCOMPLETE_EVIDENCE",
            "EVIDENCE_REQUIRED",
        }
        self.assertEqual({s.name for s in ALL_STATUSES}, expected)

    def test_all_are_str(self):
        for s in ALL_STATUSES:
            self.assertIsInstance(s.value, str)

    def test_values_are_unique(self):
        values = [s.value for s in ALL_STATUSES]
        self.assertEqual(len(values), len(set(values)))

    def test_name_equals_value(self):
        for s in ALL_STATUSES:
            self.assertEqual(s.name, s.value)


class TestIsTerminal(unittest.TestCase):
    """is_terminal() classification of every member."""

    def test_pass_is_terminal(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_exhausted_is_terminal(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())

    def test_infrastructure_failure_is_terminal(self):
        self.assertTrue(BudgetStatus.INFRASTRUCTURE_FAILURE.is_terminal())

    def test_needs_revision_not_terminal(self):
        self.assertFalse(BudgetStatus.NEEDS_REVISION.is_terminal())

    def test_harness_error_not_terminal(self):
        self.assertFalse(BudgetStatus.HARNESS_ERROR.is_terminal())

    def test_simulation_timeout_not_terminal(self):
        self.assertFalse(BudgetStatus.SIMULATION_TIMEOUT.is_terminal())

    def test_incomplete_evidence_not_terminal(self):
        self.assertFalse(BudgetStatus.INCOMPLETE_EVIDENCE.is_terminal())

    def test_evidence_required_not_terminal(self):
        self.assertFalse(BudgetStatus.EVIDENCE_REQUIRED.is_terminal())

    def test_terminal_set_size(self):
        terminals = [s for s in ALL_STATUSES if s.is_terminal()]
        self.assertEqual(len(terminals), 3)

    def test_non_terminal_set_size(self):
        non_terminals = [s for s in ALL_STATUSES if not s.is_terminal()]
        self.assertEqual(len(non_terminals), 5)


class TestTerminalRetrySemantics(unittest.TestCase):
    """Terminal vs. retryable status cross-check.

    PASS and ACTIVE_BUDGET_EXHAUSTED are terminal AND non-retryable.
    INFRASTRUCTURE_FAILURE is terminal but IS retryable (transient infra).
    """

    def test_pass_terminal_and_non_retryable(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())
        self.assertFalse(should_retry(0, BudgetStatus.PASS.value))

    def test_exhausted_terminal_and_non_retryable(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())
        self.assertFalse(should_retry(0, BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.value))

    def test_infra_failure_terminal_but_retryable(self):
        self.assertTrue(BudgetStatus.INFRASTRUCTURE_FAILURE.is_terminal())
        self.assertTrue(should_retry(0, BudgetStatus.INFRASTRUCTURE_FAILURE.value))

    def test_non_terminal_statuses_all_retryable(self):
        for s in NON_TERMINAL:
            self.assertTrue(should_retry(0, s.value),
                            f"{s.name} should be retryable")


class TestStatusStringRepresentation(unittest.TestCase):
    """String conversion and lookup."""

    def test_str_contains_name(self):
        for s in ALL_STATUSES:
            self.assertIn(s.name, str(s))

    def test_lookup_by_value(self):
        for s in ALL_STATUSES:
            self.assertEqual(BudgetStatus(s.value), s)

    def test_invalid_value_raises(self):
        with self.assertRaises(ValueError):
            BudgetStatus("NOT_A_STATUS")

    def test_lookup_case_sensitive(self):
        with self.assertRaises(ValueError):
            BudgetStatus("pass")
        with self.assertRaises(ValueError):
            BudgetStatus("Pass")


class TestStatusPartitioning(unittest.TestCase):
    """Terminal and non-terminal form a complete partition."""

    def test_every_status_classified(self):
        for s in ALL_STATUSES:
            classified = s in TERMINAL or s in NON_TERMINAL
            self.assertTrue(classified, f"{s.name} not in either set")

    def test_no_overlap(self):
        self.assertEqual(len(TERMINAL & NON_TERMINAL), 0)

    def test_union_is_all(self):
        self.assertEqual(TERMINAL | NON_TERMINAL, set(ALL_STATUSES))

    def test_terminal_count_three(self):
        actual_terminal = {s for s in ALL_STATUSES if s.is_terminal()}
        self.assertEqual(actual_terminal, TERMINAL)


if __name__ == "__main__":
    unittest.main()
