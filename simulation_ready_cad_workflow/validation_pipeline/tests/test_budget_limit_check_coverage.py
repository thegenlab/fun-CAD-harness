"""Task and part limit check comprehensive coverage tests.

Tests check_task_count and check_part_count with all production
tiers, exact boundary values, empty lists, single items, and
violation message formatting.
"""
from __future__ import annotations

import unittest

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    check_part_count,
    check_task_count,
    get_tier,
)


class TestTaskCountEmpty(unittest.TestCase):
    """Empty task list always passes."""

    def test_empty_light(self):
        self.assertEqual(check_task_count([], get_tier("light")), [])

    def test_empty_medium(self):
        self.assertEqual(check_task_count([], get_tier("medium")), [])

    def test_empty_high(self):
        self.assertEqual(check_task_count([], get_tier("high")), [])


class TestTaskCountExactBoundary(unittest.TestCase):
    """At exact limit = pass, at limit+1 = violation."""

    def test_light_at_limit(self):
        tier = get_tier("light")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
        self.assertEqual(check_task_count(tasks, tier), [])

    def test_light_over_limit(self):
        tier = get_tier("light")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design + 1)]
        self.assertGreater(len(check_task_count(tasks, tier)), 0)

    def test_medium_at_limit(self):
        tier = get_tier("medium")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
        self.assertEqual(check_task_count(tasks, tier), [])

    def test_medium_over_limit(self):
        tier = get_tier("medium")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design + 1)]
        self.assertGreater(len(check_task_count(tasks, tier)), 0)

    def test_high_at_limit(self):
        tier = get_tier("high")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
        self.assertEqual(check_task_count(tasks, tier), [])

    def test_high_over_limit(self):
        tier = get_tier("high")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design + 1)]
        self.assertGreater(len(check_task_count(tasks, tier)), 0)


class TestTaskCountViolationMessage(unittest.TestCase):
    """Violation message contains count and tier name."""

    def test_message_contains_count(self):
        tier = get_tier("light")
        tasks = ["t"] * 5
        violations = check_task_count(tasks, tier)
        self.assertTrue(any("5" in v for v in violations))

    def test_message_contains_tier_name(self):
        tier = get_tier("light")
        tasks = ["t"] * 5
        violations = check_task_count(tasks, tier)
        self.assertTrue(any("light" in v for v in violations))

    def test_single_violation_returned(self):
        tier = get_tier("light")
        tasks = ["t"] * 100
        violations = check_task_count(tasks, tier)
        self.assertEqual(len(violations), 1)


class TestPartCountEmpty(unittest.TestCase):
    """Empty part list always passes."""

    def test_empty_light(self):
        self.assertEqual(check_part_count([], get_tier("light")), [])

    def test_empty_medium(self):
        self.assertEqual(check_part_count([], get_tier("medium")), [])

    def test_empty_high(self):
        self.assertEqual(check_part_count([], get_tier("high")), [])


class TestPartCountExactBoundary(unittest.TestCase):
    """At exact limit = pass, at limit+1 = violation."""

    def test_light_at_limit(self):
        tier = get_tier("light")
        parts = [f"p{i}" for i in range(tier.max_parts)]
        self.assertEqual(check_part_count(parts, tier), [])

    def test_light_over_limit(self):
        tier = get_tier("light")
        parts = [f"p{i}" for i in range(tier.max_parts + 1)]
        self.assertGreater(len(check_part_count(parts, tier)), 0)

    def test_medium_at_limit(self):
        tier = get_tier("medium")
        parts = [f"p{i}" for i in range(tier.max_parts)]
        self.assertEqual(check_part_count(parts, tier), [])

    def test_medium_over_limit(self):
        tier = get_tier("medium")
        parts = [f"p{i}" for i in range(tier.max_parts + 1)]
        self.assertGreater(len(check_part_count(parts, tier)), 0)

    def test_high_at_limit(self):
        tier = get_tier("high")
        parts = [f"p{i}" for i in range(tier.max_parts)]
        self.assertEqual(check_part_count(parts, tier), [])

    def test_high_over_limit(self):
        tier = get_tier("high")
        parts = [f"p{i}" for i in range(tier.max_parts + 1)]
        self.assertGreater(len(check_part_count(parts, tier)), 0)


class TestPartCountViolationMessage(unittest.TestCase):
    """Violation message includes count and tier name."""

    def test_message_contains_count(self):
        tier = get_tier("light")
        parts = ["p"] * 15
        violations = check_part_count(parts, tier)
        self.assertTrue(any("15" in v for v in violations))

    def test_message_contains_tier_name(self):
        tier = get_tier("medium")
        parts = ["p"] * 20
        violations = check_part_count(parts, tier)
        self.assertTrue(any("medium" in v for v in violations))


class TestCustomTierLimits(unittest.TestCase):
    """Limit checks work with custom (non-production) tiers."""

    def test_strict_task_limit(self):
        tier = TierConfig("strict", 100, 10, 1, 0, 5)
        self.assertEqual(check_task_count(["t1"], tier), [])
        self.assertGreater(len(check_task_count(["t1", "t2"], tier)), 0)

    def test_strict_part_limit(self):
        tier = TierConfig("strict", 100, 10, 1, 0, 2)
        self.assertEqual(check_part_count(["p1", "p2"], tier), [])
        self.assertGreater(len(check_part_count(["a", "b", "c"], tier)), 0)

    def test_generous_limits(self):
        tier = TierConfig("gen", 86400, 3600, 100, 0, 1000)
        tasks = [f"t{i}" for i in range(100)]
        parts = [f"p{i}" for i in range(1000)]
        self.assertEqual(check_task_count(tasks, tier), [])
        self.assertEqual(check_part_count(parts, tier), [])


class TestLimitCheckReturnType(unittest.TestCase):
    """Return type is always list of strings."""

    def test_pass_returns_empty_list(self):
        result = check_task_count(["t"], get_tier("light"))
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_violation_returns_string_list(self):
        result = check_task_count(["t"] * 10, get_tier("light"))
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

    def test_part_pass_returns_empty_list(self):
        result = check_part_count(["p"], get_tier("high"))
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_part_violation_returns_string_list(self):
        result = check_part_count(["p"] * 50, get_tier("light"))
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)


if __name__ == "__main__":
    unittest.main()
