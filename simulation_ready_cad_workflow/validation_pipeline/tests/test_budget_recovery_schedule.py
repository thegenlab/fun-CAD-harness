"""Recovery delay schedule and retry decision boundary tests.

Tests recovery_delay() formula edge cases, should_retry() keyword
matching edge cases, and their combined behavior in multi-step
retry sequences.
"""
from __future__ import annotations

import unittest

from ..slim_harness.budget import (
    BACKOFF_BASE_SECONDS,
    MAX_RECOVERY_RETRIES,
    recovery_delay,
    should_retry,
)


class TestRecoveryDelayFormula(unittest.TestCase):
    """recovery_delay = min(5.0 * 2**attempt, 60.0)."""

    def test_attempt_zero(self):
        self.assertAlmostEqual(recovery_delay(0), 5.0)

    def test_attempt_one(self):
        self.assertAlmostEqual(recovery_delay(1), 10.0)

    def test_attempt_two(self):
        self.assertAlmostEqual(recovery_delay(2), 20.0)

    def test_attempt_three(self):
        self.assertAlmostEqual(recovery_delay(3), 40.0)

    def test_attempt_four_capped(self):
        self.assertAlmostEqual(recovery_delay(4), 60.0)

    def test_attempt_five_still_capped(self):
        self.assertAlmostEqual(recovery_delay(5), 60.0)

    def test_large_attempt_capped(self):
        self.assertAlmostEqual(recovery_delay(100), 60.0)

    def test_all_delays_positive(self):
        for a in range(20):
            self.assertGreater(recovery_delay(a), 0.0)

    def test_monotonic_until_cap(self):
        prev = 0.0
        for a in range(5):
            d = recovery_delay(a)
            self.assertGreaterEqual(d, prev)
            prev = d

    def test_all_delays_at_most_60(self):
        for a in range(50):
            self.assertLessEqual(recovery_delay(a), 60.0)


class TestRecoveryDelayConstants(unittest.TestCase):
    """Verify exported constants."""

    def test_backoff_base(self):
        self.assertEqual(BACKOFF_BASE_SECONDS, 5.0)

    def test_max_retries(self):
        self.assertEqual(MAX_RECOVERY_RETRIES, 3)

    def test_base_matches_delay_zero(self):
        self.assertAlmostEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)


class TestShouldRetryAttemptBounds(unittest.TestCase):
    """should_retry respects MAX_RECOVERY_RETRIES."""

    def test_attempt_0_retryable(self):
        self.assertTrue(should_retry(0, "some infra error"))

    def test_attempt_1_retryable(self):
        self.assertTrue(should_retry(1, "connection timeout"))

    def test_attempt_2_retryable(self):
        self.assertTrue(should_retry(2, "server unavailable"))

    def test_attempt_3_not_retryable(self):
        self.assertFalse(should_retry(3, "some error"))

    def test_attempt_4_not_retryable(self):
        self.assertFalse(should_retry(4, "some error"))

    def test_large_attempt_not_retryable(self):
        self.assertFalse(should_retry(100, "transient"))


class TestShouldRetryKeywords(unittest.TestCase):
    """Non-retryable keywords in error string."""

    def test_active_budget_exhausted_exact(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_terminated_exact(self):
        self.assertFalse(should_retry(0, "TERMINATED"))

    def test_pass_exact(self):
        self.assertFalse(should_retry(0, "PASS"))

    def test_exhausted_as_substring(self):
        self.assertFalse(
            should_retry(0, "Error: ACTIVE_BUDGET_EXHAUSTED during run")
        )

    def test_terminated_as_substring(self):
        self.assertFalse(
            should_retry(0, "Process was TERMINATED by signal")
        )

    def test_pass_as_substring(self):
        self.assertFalse(should_retry(0, "Result: PASS"))

    def test_lowercase_not_matched(self):
        self.assertTrue(should_retry(0, "active_budget_exhausted"))

    def test_lowercase_pass_not_matched(self):
        self.assertTrue(should_retry(0, "pass"))

    def test_lowercase_terminated_not_matched(self):
        self.assertTrue(should_retry(0, "terminated"))

    def test_empty_error_retryable(self):
        self.assertTrue(should_retry(0, ""))

    def test_unrelated_error_retryable(self):
        self.assertTrue(should_retry(0, "network timeout"))


class TestShouldRetryKeywordPriority(unittest.TestCase):
    """Keyword check happens even at attempt 0."""

    def test_exhausted_at_attempt_0(self):
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_terminated_at_attempt_0(self):
        self.assertFalse(should_retry(0, "TERMINATED"))

    def test_pass_at_attempt_0(self):
        self.assertFalse(should_retry(0, "PASS"))

    def test_attempt_limit_overrides_retryable_error(self):
        self.assertFalse(should_retry(3, "normal infra error"))

    def test_both_non_retryable_keyword_and_over_limit(self):
        self.assertFalse(should_retry(5, "ACTIVE_BUDGET_EXHAUSTED"))


class TestRetryDelayIntegration(unittest.TestCase):
    """Combined retry decisions with delay schedule."""

    def test_full_retry_sequence(self):
        delays = []
        for attempt in range(10):
            if should_retry(attempt, "infra timeout"):
                delays.append(recovery_delay(attempt))
            else:
                break
        self.assertEqual(len(delays), MAX_RECOVERY_RETRIES)
        self.assertEqual(delays, [5.0, 10.0, 20.0])

    def test_immediate_reject_no_delays(self):
        delays = []
        for attempt in range(5):
            if should_retry(attempt, "ACTIVE_BUDGET_EXHAUSTED"):
                delays.append(recovery_delay(attempt))
            else:
                break
        self.assertEqual(delays, [])

    def test_total_backoff_time(self):
        total = sum(
            recovery_delay(a) for a in range(MAX_RECOVERY_RETRIES)
        )
        self.assertAlmostEqual(total, 35.0)

    def test_delay_schedule_matches_formula(self):
        for a in range(10):
            expected = min(BACKOFF_BASE_SECONDS * (2 ** a), 60.0)
            self.assertAlmostEqual(recovery_delay(a), expected)


if __name__ == "__main__":
    unittest.main()
