"""Enforce simulation timeout and SimulationTimeoutResult edge cases.

Covers enforce_simulation_timeout input validation (positive, finite,
None default), tier capping, and SimulationTimeoutResult status mapping.
"""
from __future__ import annotations

import math
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    BudgetStatus,
    SimulationTimeoutResult,
    TierConfig,
    enforce_simulation_timeout,
    get_tier,
)


class TestEnforceTimeoutDefault(unittest.TestCase):
    """When actual_timeout is None, tier limit is returned."""

    def test_none_returns_tier_limit(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier)
            self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_explicit_none_same_as_omitted(self):
        tier = get_tier("medium")
        self.assertEqual(
            enforce_simulation_timeout(tier, None),
            enforce_simulation_timeout(tier),
        )

    def test_each_tier_returns_own_limit(self):
        self.assertEqual(enforce_simulation_timeout(get_tier("light")), 60)
        self.assertEqual(enforce_simulation_timeout(get_tier("medium")), 180)
        self.assertEqual(enforce_simulation_timeout(get_tier("high")), 300)


class TestEnforceTimeoutCapping(unittest.TestCase):
    """actual_timeout is capped at tier limit."""

    def test_below_limit_passes_through(self):
        tier = get_tier("high")  # 300s
        self.assertEqual(enforce_simulation_timeout(tier, 100.0), 100.0)

    def test_at_limit_returns_limit(self):
        tier = get_tier("medium")  # 180s
        self.assertEqual(enforce_simulation_timeout(tier, 180.0), 180.0)

    def test_above_limit_capped(self):
        tier = get_tier("light")  # 60s
        self.assertEqual(enforce_simulation_timeout(tier, 999.0), 60.0)

    def test_just_below_limit(self):
        tier = get_tier("medium")
        self.assertAlmostEqual(
            enforce_simulation_timeout(tier, 179.999), 179.999,
        )

    def test_just_above_limit(self):
        tier = get_tier("medium")
        self.assertEqual(enforce_simulation_timeout(tier, 180.001), 180.0)

    def test_tiny_timeout_passes_through(self):
        tier = get_tier("high")
        self.assertEqual(enforce_simulation_timeout(tier, 0.001), 0.001)

    def test_custom_tier_capping(self):
        tier = TierConfig("fast", 100, 5, 1, 0, 3)
        self.assertEqual(enforce_simulation_timeout(tier, 10), 5.0)
        self.assertEqual(enforce_simulation_timeout(tier, 3), 3.0)


class TestEnforceTimeoutInvalidInputs(unittest.TestCase):
    """Invalid actual_timeout values raise ValueError."""

    def test_zero_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, 0)

    def test_negative_raises(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -1.0)

    def test_large_negative_raises(self):
        tier = get_tier("high")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -1e10)

    def test_inf_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("inf"))

    def test_negative_inf_raises(self):
        tier = get_tier("medium")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("-inf"))

    def test_nan_raises(self):
        tier = get_tier("high")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("nan"))

    def test_negative_zero_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -0.0)


class TestEnforceTimeoutReturnType(unittest.TestCase):
    """Return type is always float."""

    def test_returns_float_with_none(self):
        result = enforce_simulation_timeout(get_tier("light"))
        self.assertIsInstance(result, (int, float))

    def test_returns_numeric_with_value(self):
        result = enforce_simulation_timeout(get_tier("high"), 50.0)
        self.assertIsInstance(result, (int, float))

    def test_result_always_positive(self):
        for name, tier in TIERS.items():
            self.assertGreater(enforce_simulation_timeout(tier), 0)
            self.assertGreater(enforce_simulation_timeout(tier, 1.0), 0)

    def test_result_never_exceeds_tier(self):
        for name, tier in TIERS.items():
            for val in [0.1, 1.0, 50.0, 500.0, 10000.0]:
                result = enforce_simulation_timeout(tier, val)
                self.assertLessEqual(result, tier.simulation_timeout_seconds)


class TestSimulationTimeoutResultStatus(unittest.TestCase):
    """SimulationTimeoutResult.status property mapping."""

    def test_timed_out_true_gives_timeout_status(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_timed_out_false_gives_pass_status(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=30.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_zero_elapsed_pass(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=0.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_zero_elapsed_timeout(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=0.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)


class TestSimulationTimeoutResultFields(unittest.TestCase):
    """SimulationTimeoutResult field defaults and access."""

    def test_defaults_none_fields(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0)
        self.assertIsNone(r.partial_evidence)
        self.assertIsNone(r.log_path)
        self.assertIsNone(r.error)

    def test_partial_evidence_stored(self):
        ev = {"metric": 42}
        r = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=60.0, partial_evidence=ev,
        )
        self.assertEqual(r.partial_evidence, ev)

    def test_log_path_stored(self):
        p = Path("/tmp/test.log")
        r = SimulationTimeoutResult(
            timed_out=False, elapsed_seconds=5.0, log_path=p,
        )
        self.assertEqual(r.log_path, p)

    def test_error_stored(self):
        r = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=120.0, error="killed",
        )
        self.assertEqual(r.error, "killed")

    def test_all_fields_populated(self):
        r = SimulationTimeoutResult(
            timed_out=True,
            elapsed_seconds=99.5,
            partial_evidence={"a": 1},
            log_path=Path("x.log"),
            error="oom",
        )
        self.assertTrue(r.timed_out)
        self.assertAlmostEqual(r.elapsed_seconds, 99.5)
        self.assertEqual(r.partial_evidence, {"a": 1})
        self.assertEqual(r.log_path, Path("x.log"))
        self.assertEqual(r.error, "oom")


class TestSimulationTimeoutResultStatusConsistency(unittest.TestCase):
    """Status depends only on timed_out, not other fields."""

    def test_error_does_not_affect_status(self):
        r1 = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0, error="err")
        r2 = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0)
        self.assertEqual(r1.status, r2.status)

    def test_evidence_does_not_affect_status(self):
        r1 = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=1.0,
            partial_evidence={"data": True},
        )
        r2 = SimulationTimeoutResult(timed_out=True, elapsed_seconds=1.0)
        self.assertEqual(r1.status, r2.status)

    def test_elapsed_does_not_affect_status(self):
        r1 = SimulationTimeoutResult(timed_out=True, elapsed_seconds=0.0)
        r2 = SimulationTimeoutResult(timed_out=True, elapsed_seconds=9999.0)
        self.assertEqual(r1.status, r2.status)

    def test_status_is_enum_member(self):
        for timed_out in (True, False):
            r = SimulationTimeoutResult(timed_out=timed_out, elapsed_seconds=1.0)
            self.assertIsInstance(r.status, BudgetStatus)


if __name__ == "__main__":
    unittest.main()
