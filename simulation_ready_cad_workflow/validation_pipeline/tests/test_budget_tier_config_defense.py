"""TierConfig construction defense and frozen enforcement tests.

Comprehensive boundary testing for TierConfig dataclass validation:
__post_init__ rejection of invalid values, frozen attribute mutation,
and asdict round-trip fidelity.
"""
from __future__ import annotations

import math
import unittest
from dataclasses import asdict, FrozenInstanceError

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    get_tier,
)


class TestTierConfigValidBoundaries(unittest.TestCase):
    """Valid construction at boundary values."""

    def test_minimal_valid(self):
        t = TierConfig("min", 1, 1, 1, 0, 1)
        self.assertEqual(t.name, "min")
        self.assertEqual(t.active_budget_seconds, 1)

    def test_large_budget(self):
        t = TierConfig("big", 86400, 3600, 100, 0, 1000)
        self.assertEqual(t.active_budget_seconds, 86400)

    def test_min_parts_zero(self):
        t = TierConfig("zero_min", 100, 10, 1, 0, 5)
        self.assertEqual(t.min_parts, 0)

    def test_min_equals_max_parts(self):
        t = TierConfig("eq", 100, 10, 1, 5, 5)
        self.assertEqual(t.min_parts, 5)
        self.assertEqual(t.max_parts, 5)

    def test_one_task(self):
        t = TierConfig("one", 100, 10, 1, 0, 5)
        self.assertEqual(t.max_tasks_per_design, 1)


class TestTierConfigInvalidBudget(unittest.TestCase):
    """Budget must be positive and finite."""

    def test_zero_budget(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 0, 10, 1, 0, 5)

    def test_negative_budget(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -1, 10, 1, 0, 5)

    def test_inf_budget(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("inf"), 10, 1, 0, 5)

    def test_nan_budget(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("nan"), 10, 1, 0, 5)


class TestTierConfigInvalidTimeout(unittest.TestCase):
    """Timeout must be positive and finite."""

    def test_zero_timeout(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 0, 1, 0, 5)

    def test_negative_timeout(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, -5, 1, 0, 5)

    def test_inf_timeout(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, float("inf"), 1, 0, 5)

    def test_nan_timeout(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, float("nan"), 1, 0, 5)


class TestTierConfigInvalidPartBounds(unittest.TestCase):
    """min_parts <= max_parts, both non-negative, max >= 1."""

    def test_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 10, 1, 6, 5)

    def test_negative_min_parts(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 10, 1, -1, 5)

    def test_max_parts_zero(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 10, 1, 0, 0)

    def test_negative_max_parts(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 10, 1, 0, -1)


class TestTierConfigInvalidTasks(unittest.TestCase):
    """max_tasks_per_design must be >= 1."""

    def test_zero_tasks(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 10, 0, 0, 5)

    def test_negative_tasks(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 100, 10, -1, 0, 5)


class TestTierConfigFrozen(unittest.TestCase):
    """Frozen dataclass prevents mutation."""

    def test_cannot_set_name(self):
        t = get_tier("light")
        with self.assertRaises(FrozenInstanceError):
            t.name = "hacked"

    def test_cannot_set_budget(self):
        t = get_tier("medium")
        with self.assertRaises(FrozenInstanceError):
            t.active_budget_seconds = 999999

    def test_cannot_set_timeout(self):
        t = get_tier("high")
        with self.assertRaises(FrozenInstanceError):
            t.simulation_timeout_seconds = 0

    def test_cannot_set_tasks(self):
        t = get_tier("light")
        with self.assertRaises(FrozenInstanceError):
            t.max_tasks_per_design = 100

    def test_cannot_set_parts(self):
        t = get_tier("medium")
        with self.assertRaises(FrozenInstanceError):
            t.max_parts = 9999


class TestTierConfigAsDict(unittest.TestCase):
    """asdict round-trip preserves all fields."""

    def test_round_trip_light(self):
        t = get_tier("light")
        d = asdict(t)
        self.assertEqual(d["name"], "light")
        self.assertEqual(d["active_budget_seconds"], 3600)
        self.assertEqual(d["simulation_timeout_seconds"], 60)
        self.assertEqual(d["max_tasks_per_design"], 2)
        self.assertEqual(d["min_parts"], 0)
        self.assertEqual(d["max_parts"], 10)

    def test_round_trip_high(self):
        t = get_tier("high")
        d = asdict(t)
        self.assertEqual(d["active_budget_seconds"], 10800)
        self.assertEqual(d["max_parts"], 20)

    def test_all_tiers_have_six_fields(self):
        for name, tier in TIERS.items():
            d = asdict(tier)
            self.assertEqual(len(d), 6, f"{name} should have 6 fields")

    def test_custom_tier_round_trip(self):
        t = TierConfig("custom", 42, 7, 3, 1, 9)
        d = asdict(t)
        t2 = TierConfig(**d)
        self.assertEqual(asdict(t2), d)


class TestGetTierLookup(unittest.TestCase):
    """get_tier exact name lookup."""

    def test_light(self):
        t = get_tier("light")
        self.assertEqual(t.name, "light")

    def test_medium(self):
        t = get_tier("medium")
        self.assertEqual(t.name, "medium")

    def test_high(self):
        t = get_tier("high")
        self.assertEqual(t.name, "high")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_tier("ultra")

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_tier("Light")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            get_tier("")


class TestProductionTierOrdering(unittest.TestCase):
    """Production tiers have increasing budgets and limits."""

    def test_budget_ordering(self):
        l, m, h = get_tier("light"), get_tier("medium"), get_tier("high")
        self.assertLess(l.active_budget_seconds, m.active_budget_seconds)
        self.assertLess(m.active_budget_seconds, h.active_budget_seconds)

    def test_timeout_ordering(self):
        l, m, h = get_tier("light"), get_tier("medium"), get_tier("high")
        self.assertLess(l.simulation_timeout_seconds, m.simulation_timeout_seconds)
        self.assertLess(m.simulation_timeout_seconds, h.simulation_timeout_seconds)

    def test_parts_ordering(self):
        l, m, h = get_tier("light"), get_tier("medium"), get_tier("high")
        self.assertLess(l.max_parts, m.max_parts)
        self.assertLess(m.max_parts, h.max_parts)

    def test_exactly_three_production_tiers(self):
        self.assertEqual(len(TIERS), 3)
        self.assertEqual(set(TIERS.keys()), {"light", "medium", "high"})


if __name__ == "__main__":
    unittest.main()
