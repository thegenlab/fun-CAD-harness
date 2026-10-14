"""Budget retry and recovery interaction tests.

Tests should_retry + recovery_delay across all BudgetStatus values,
combined scenarios with exhaustion, and delay schedule properties.
"""
from __future__ import annotations

import unittest

from ..slim_harness.budget import (
    BACKOFF_BASE_SECONDS,
    MAX_RECOVERY_RETRIES,
    BudgetStatus,
    recovery_delay,
    should_retry,
)


class TestShouldRetryBasic(unittest.TestCase):
    """Basic should_retry behavior."""

    def test_first_attempt_transient_retries(self):
        self.assertTrue(should_retry(0, "some transient error"))

    def test_max_retries_never_retries(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "transient"))

    def test_above_max_retries_never_retries(self):
        for i in range(MAX_RECOVERY_RETRIES, MAX_RECOVERY_RETRIES + 10):
            self.assertFalse(should_retry(i, "transient"))

    def test_empty_error_retries(self):
        self.assertTrue(should_retry(0, ""))

    def test_long_error_message_checked(self):
        msg = "x" * 10000 + " ACTIVE_BUDGET_EXHAUSTED " + "y" * 10000
        self.assertFalse(should_retry(0, msg))


class TestShouldRetryNonRetryableKeywords(unittest.TestCase):
    """Non-retryable keywords block retries regardless of attempt."""

    NON_RETRYABLE = {"ACTIVE_BUDGET_EXHAUSTED", "TERMINATED", "PASS"}

    def test_each_keyword_blocks_retry(self):
        for keyword in self.NON_RETRYABLE:
            self.assertFalse(should_retry(0, keyword),
                             f"{keyword} should block retry")

    def test_keywords_in_longer_message(self):
        for keyword in self.NON_RETRYABLE:
            msg = f"Error occurred: {keyword} at step 5"
            self.assertFalse(should_retry(0, msg))

    def test_partial_keyword_does_not_block(self):
        self.assertTrue(should_retry(0, "ACTIVE_BUDGET"))
        self.assertTrue(should_retry(0, "TERMINAT"))

    def test_case_sensitive(self):
        self.assertTrue(should_retry(0, "pass"))
        self.assertTrue(should_retry(0, "terminated"))
        self.assertTrue(should_retry(0, "active_budget_exhausted"))


class TestShouldRetryWithBudgetStatuses(unittest.TestCase):
    """should_retry with each BudgetStatus value as the error string."""

    def test_pass_blocks_retry(self):
        self.assertFalse(should_retry(0, BudgetStatus.PASS.value))

    def test_exhausted_blocks_retry(self):
        self.assertFalse(should_retry(0, BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.value))

    def test_infrastructure_failure_allows_retry(self):
        self.assertTrue(should_retry(0, BudgetStatus.INFRASTRUCTURE_FAILURE.value))

    def test_simulation_timeout_allows_retry(self):
        self.assertTrue(should_retry(0, BudgetStatus.SIMULATION_TIMEOUT.value))

    def test_harness_error_allows_retry(self):
        self.assertTrue(should_retry(0, BudgetStatus.HARNESS_ERROR.value))

    def test_needs_revision_allows_retry(self):
        self.assertTrue(should_retry(0, BudgetStatus.NEEDS_REVISION.value))

    def test_evidence_required_allows_retry(self):
        self.assertTrue(should_retry(0, BudgetStatus.EVIDENCE_REQUIRED.value))

    def test_incomplete_evidence_allows_retry(self):
        self.assertTrue(should_retry(0, BudgetStatus.INCOMPLETE_EVIDENCE.value))


class TestShouldRetryAttemptProgression(unittest.TestCase):
    """Retry behavior across attempt progression."""

    def test_all_attempts_up_to_max(self):
        for i in range(MAX_RECOVERY_RETRIES):
            self.assertTrue(should_retry(i, "transient"))
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "transient"))

    def test_non_retryable_blocks_at_any_attempt(self):
        for i in range(MAX_RECOVERY_RETRIES + 5):
            self.assertFalse(should_retry(i, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_retryable_allows_below_max(self):
        for i in range(MAX_RECOVERY_RETRIES):
            self.assertTrue(should_retry(i, "INFRASTRUCTURE_FAILURE"))

    def test_retryable_blocks_at_max(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "INFRASTRUCTURE_FAILURE"))


class TestRecoveryDelaySchedule(unittest.TestCase):
    """recovery_delay produces the correct exponential backoff schedule."""

    def test_first_delay(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_second_delay(self):
        self.assertEqual(recovery_delay(1), BACKOFF_BASE_SECONDS * 2)

    def test_third_delay(self):
        self.assertEqual(recovery_delay(2), BACKOFF_BASE_SECONDS * 4)

    def test_fourth_delay(self):
        self.assertEqual(recovery_delay(3), min(BACKOFF_BASE_SECONDS * 8, 60.0))

    def test_exact_schedule(self):
        expected = [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
        for i, exp in enumerate(expected):
            self.assertEqual(recovery_delay(i), exp,
                             f"Attempt {i}: expected {exp}")

    def test_capped_at_60(self):
        for i in range(20):
            self.assertLessEqual(recovery_delay(i), 60.0)

    def test_always_positive(self):
        for i in range(50):
            self.assertGreater(recovery_delay(i), 0)


class TestRecoveryDelayMonotonicity(unittest.TestCase):
    """Delays are non-decreasing across attempts."""

    def test_nondecreasing(self):
        prev = 0.0
        for i in range(20):
            d = recovery_delay(i)
            self.assertGreaterEqual(d, prev)
            prev = d

    def test_strictly_increasing_until_cap(self):
        prev = 0.0
        for i in range(20):
            d = recovery_delay(i)
            if d < 60.0:
                self.assertGreater(d, prev)
            prev = d

    def test_all_at_cap_after_threshold(self):
        for i in range(4, 20):
            self.assertEqual(recovery_delay(i), 60.0)


class TestRetryRecoveryCombined(unittest.TestCase):
    """Combined retry + delay scenarios."""

    def test_total_delay_for_max_retries(self):
        total = sum(recovery_delay(i) for i in range(MAX_RECOVERY_RETRIES))
        self.assertEqual(total, 5.0 + 10.0 + 20.0)

    def test_retry_then_delay_sequence(self):
        delays = []
        for i in range(10):
            if should_retry(i, "INFRASTRUCTURE_FAILURE"):
                delays.append(recovery_delay(i))
        self.assertEqual(len(delays), MAX_RECOVERY_RETRIES)
        self.assertEqual(delays, [5.0, 10.0, 20.0])

    def test_non_retryable_gets_no_delays(self):
        delays = []
        for i in range(10):
            if should_retry(i, "ACTIVE_BUDGET_EXHAUSTED"):
                delays.append(recovery_delay(i))
        self.assertEqual(delays, [])

    def test_pass_gets_no_delays(self):
        delays = []
        for i in range(10):
            if should_retry(i, "PASS"):
                delays.append(recovery_delay(i))
        self.assertEqual(delays, [])

    def test_terminated_gets_no_delays(self):
        delays = []
        for i in range(10):
            if should_retry(i, "TERMINATED"):
                delays.append(recovery_delay(i))
        self.assertEqual(delays, [])


if __name__ == "__main__":
    unittest.main()
