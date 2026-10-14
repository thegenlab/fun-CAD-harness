"""Budget tier validation and counting function edge case tests.

Tests TierConfig construction validation, production tier ordering
invariants, and check_task_count/check_part_count boundary behavior.
"""
from __future__ import annotations

import math
import unittest

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    check_part_count,
    check_task_count,
    get_tier,
)


class TestTierConfigValidation(unittest.TestCase):
    """TierConfig __post_init__ rejects invalid configurations."""

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 0, 60, 1, 0, 5)

    def test_negative_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -1, 60, 1, 0, 5)

    def test_zero_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 0, 1, 0, 5)

    def test_negative_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, -10, 1, 0, 5)

    def test_inf_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("inf"), 60, 1, 0, 5)

    def test_nan_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("nan"), 60, 1, 0, 5)

    def test_inf_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, float("inf"), 1, 0, 5)

    def test_nan_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, float("nan"), 1, 0, 5)

    def test_min_greater_than_max_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 10, 5)

    def test_max_parts_zero_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 0, 0)

    def test_negative_min_parts_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, -1, 5)

    def test_valid_minimal_tier(self):
        tier = TierConfig("minimal", 0.001, 0.001, 1, 0, 1)
        self.assertEqual(tier.name, "minimal")


class TestTierConfigFrozen(unittest.TestCase):
    """TierConfig is a frozen dataclass."""

    def test_cannot_change_name(self):
        tier = get_tier("light")
        with self.assertRaises(AttributeError):
            tier.name = "changed"

    def test_cannot_change_budget(self):
        tier = get_tier("medium")
        with self.assertRaises(AttributeError):
            tier.active_budget_seconds = 999

    def test_cannot_change_timeout(self):
        tier = get_tier("high")
        with self.assertRaises(AttributeError):
            tier.simulation_timeout_seconds = 1

    def test_cannot_change_max_tasks(self):
        tier = get_tier("light")
        with self.assertRaises(AttributeError):
            tier.max_tasks_per_design = 99

    def test_cannot_change_parts_range(self):
        tier = get_tier("medium")
        with self.assertRaises(AttributeError):
            tier.min_parts = 5
        with self.assertRaises(AttributeError):
            tier.max_parts = 100


class TestProductionTierOrdering(unittest.TestCase):
    """Production tiers have strictly ordered budgets and timeouts."""

    def test_budget_ordering_light_lt_medium_lt_high(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLess(light.active_budget_seconds, medium.active_budget_seconds)
        self.assertLess(medium.active_budget_seconds, high.active_budget_seconds)

    def test_timeout_ordering(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLess(light.simulation_timeout_seconds, medium.simulation_timeout_seconds)
        self.assertLess(medium.simulation_timeout_seconds, high.simulation_timeout_seconds)

    def test_max_parts_ordering(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLessEqual(light.max_parts, medium.max_parts)
        self.assertLessEqual(medium.max_parts, high.max_parts)

    def test_max_tasks_ordering(self):
        light = get_tier("light")
        medium = get_tier("medium")
        high = get_tier("high")
        self.assertLessEqual(light.max_tasks_per_design, medium.max_tasks_per_design)
        self.assertLessEqual(medium.max_tasks_per_design, high.max_tasks_per_design)


class TestGetTierFunction(unittest.TestCase):
    """get_tier returns correct tiers or raises ValueError."""

    def test_all_production_tiers(self):
        for name in ("light", "medium", "high"):
            tier = get_tier(name)
            self.assertEqual(tier.name, name)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            get_tier("ultra")

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_tier("Light")
        with self.assertRaises(ValueError):
            get_tier("MEDIUM")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            get_tier("")

    def test_returns_same_object_as_tiers_dict(self):
        for name in TIERS:
            self.assertIs(get_tier(name), TIERS[name])


class TestCheckTaskCountBoundary(unittest.TestCase):
    """check_task_count at and around the limit."""

    def test_zero_tasks_ok(self):
        for name in ("light", "medium", "high"):
            self.assertEqual(check_task_count([], get_tier(name)), [])

    def test_one_task_ok(self):
        self.assertEqual(check_task_count(["t1"], get_tier("light")), [])

    def test_at_limit_ok(self):
        for name, tier in TIERS.items():
            tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
            self.assertEqual(check_task_count(tasks, tier), [],
                             f"Tier {name}")

    def test_one_over_limit_violates(self):
        for name, tier in TIERS.items():
            tasks = [f"t{i}" for i in range(tier.max_tasks_per_design + 1)]
            violations = check_task_count(tasks, tier)
            self.assertGreater(len(violations), 0, f"Tier {name}")

    def test_far_over_limit_violates(self):
        tier = get_tier("light")
        tasks = [f"t{i}" for i in range(100)]
        violations = check_task_count(tasks, tier)
        self.assertGreater(len(violations), 0)

    def test_violation_message_is_string(self):
        tier = get_tier("light")
        tasks = [f"t{i}" for i in range(tier.max_tasks_per_design + 1)]
        for v in check_task_count(tasks, tier):
            self.assertIsInstance(v, str)


class TestCheckPartCountBoundary(unittest.TestCase):
    """check_part_count at and around min/max limits."""

    def test_zero_parts_ok_when_min_zero(self):
        for name, tier in TIERS.items():
            if tier.min_parts == 0:
                self.assertEqual(check_part_count([], tier), [])

    def test_at_max_ok(self):
        for name, tier in TIERS.items():
            parts = [f"p{i}" for i in range(tier.max_parts)]
            self.assertEqual(check_part_count(parts, tier), [],
                             f"Tier {name}")

    def test_one_over_max_violates(self):
        for name, tier in TIERS.items():
            parts = [f"p{i}" for i in range(tier.max_parts + 1)]
            violations = check_part_count(parts, tier)
            self.assertGreater(len(violations), 0, f"Tier {name}")

    def test_far_over_max_violates(self):
        tier = get_tier("light")
        parts = [f"p{i}" for i in range(200)]
        violations = check_part_count(parts, tier)
        self.assertGreater(len(violations), 0)

    def test_violation_message_is_string(self):
        tier = get_tier("medium")
        parts = [f"p{i}" for i in range(tier.max_parts + 1)]
        for v in check_part_count(parts, tier):
            self.assertIsInstance(v, str)

    def test_within_range_no_violations(self):
        for name, tier in TIERS.items():
            mid = (tier.min_parts + tier.max_parts) // 2
            parts = [f"p{i}" for i in range(mid)]
            self.assertEqual(check_part_count(parts, tier), [],
                             f"Tier {name} with {mid} parts")


class TestCustomTierWithCounting(unittest.TestCase):
    """Counting functions work with custom (non-production) tiers."""

    def test_custom_task_limit(self):
        tier = TierConfig("custom", 100, 10, 3, 0, 5)
        self.assertEqual(check_task_count(["a", "b", "c"], tier), [])
        violations = check_task_count(["a", "b", "c", "d"], tier)
        self.assertGreater(len(violations), 0)

    def test_custom_part_limit(self):
        tier = TierConfig("custom", 100, 10, 1, 2, 8)
        self.assertEqual(check_part_count(["p"] * 5, tier), [])
        violations = check_part_count(["p"] * 9, tier)
        self.assertGreater(len(violations), 0)

    def test_single_task_single_part_tier(self):
        tier = TierConfig("minimal", 10, 5, 1, 0, 1)
        self.assertEqual(check_task_count(["t"], tier), [])
        self.assertEqual(check_part_count(["p"], tier), [])
        self.assertGreater(len(check_task_count(["t", "t2"], tier)), 0)
        self.assertGreater(len(check_part_count(["p", "p2"], tier)), 0)


if __name__ == "__main__":
    unittest.main()
