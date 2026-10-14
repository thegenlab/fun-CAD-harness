"""Combined enforcement scenario tests.

Tests interactions between check_task_count, check_part_count,
enforce_simulation_timeout, and BudgetStatus when applied together
in realistic validation scenarios.
"""
from __future__ import annotations

import unittest

from ..slim_harness.budget import (
    TIERS,
    BudgetStatus,
    SimulationTimeoutResult,
    TierConfig,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    get_tier,
)


class TestCombinedLimitChecks(unittest.TestCase):
    """Task + part checks applied together."""

    def test_both_within_limits(self):
        tier = get_tier("medium")
        tasks = ["t1"]
        parts = ["p1", "p2", "p3"]
        self.assertEqual(check_task_count(tasks, tier), [])
        self.assertEqual(check_part_count(parts, tier), [])

    def test_tasks_over_parts_ok(self):
        tier = get_tier("light")
        tasks = ["t"] * 10
        parts = ["p"] * 5
        task_v = check_task_count(tasks, tier)
        part_v = check_part_count(parts, tier)
        self.assertGreater(len(task_v), 0)
        self.assertEqual(part_v, [])

    def test_parts_over_tasks_ok(self):
        tier = get_tier("light")
        tasks = ["t1"]
        parts = ["p"] * 20
        task_v = check_task_count(tasks, tier)
        part_v = check_part_count(parts, tier)
        self.assertEqual(task_v, [])
        self.assertGreater(len(part_v), 0)

    def test_both_over_limits(self):
        tier = get_tier("light")
        tasks = ["t"] * 10
        parts = ["p"] * 50
        task_v = check_task_count(tasks, tier)
        part_v = check_part_count(parts, tier)
        self.assertGreater(len(task_v), 0)
        self.assertGreater(len(part_v), 0)

    def test_at_exact_limits(self):
        for name, tier in TIERS.items():
            tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
            parts = [f"p{i}" for i in range(tier.max_parts)]
            self.assertEqual(check_task_count(tasks, tier), [])
            self.assertEqual(check_part_count(parts, tier), [])


class TestTimeoutWithLimitChecks(unittest.TestCase):
    """Timeout enforcement combined with part/task checks."""

    def test_valid_full_scenario(self):
        tier = get_tier("high")
        timeout = enforce_simulation_timeout(tier, 100.0)
        tasks = ["t1", "t2"]
        parts = [f"p{i}" for i in range(10)]
        self.assertEqual(timeout, 100.0)
        self.assertEqual(check_task_count(tasks, tier), [])
        self.assertEqual(check_part_count(parts, tier), [])

    def test_timeout_capped_but_limits_ok(self):
        tier = get_tier("light")  # 60s timeout
        timeout = enforce_simulation_timeout(tier, 120.0)
        self.assertEqual(timeout, 60.0)
        self.assertEqual(check_task_count(["t1"], tier), [])
        self.assertEqual(check_part_count(["p1", "p2"], tier), [])

    def test_all_enforcement_at_limits(self):
        for name, tier in TIERS.items():
            timeout = enforce_simulation_timeout(tier)
            self.assertEqual(timeout, tier.simulation_timeout_seconds)
            tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
            parts = [f"p{i}" for i in range(tier.max_parts)]
            self.assertEqual(check_task_count(tasks, tier), [])
            self.assertEqual(check_part_count(parts, tier), [])


class TestSimulationResultWithEnforcement(unittest.TestCase):
    """SimulationTimeoutResult status with enforcement context."""

    def test_pass_result_with_valid_timeout(self):
        tier = get_tier("medium")
        timeout = enforce_simulation_timeout(tier, 60.0)
        result = SimulationTimeoutResult(
            timed_out=False, elapsed_seconds=45.0,
        )
        self.assertEqual(result.status, BudgetStatus.PASS)
        self.assertEqual(timeout, 60.0)

    def test_timeout_result_at_tier_limit(self):
        tier = get_tier("light")
        timeout = enforce_simulation_timeout(tier)
        result = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=timeout,
        )
        self.assertEqual(result.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_pass_at_boundary(self):
        tier = get_tier("high")
        timeout = enforce_simulation_timeout(tier, 299.99)
        result = SimulationTimeoutResult(
            timed_out=False, elapsed_seconds=299.99,
        )
        self.assertEqual(result.status, BudgetStatus.PASS)
        self.assertAlmostEqual(timeout, 299.99)


class TestStatusInteraction(unittest.TestCase):
    """BudgetStatus semantics with enforcement violations."""

    def test_pass_is_terminal_no_violations(self):
        tier = get_tier("medium")
        self.assertEqual(check_task_count(["t1"], tier), [])
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_violations_independent_of_status(self):
        tier = get_tier("light")
        violations = check_task_count(["t"] * 100, tier)
        self.assertGreater(len(violations), 0)
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_all_terminal_statuses_with_clean_enforcement(self):
        tier = get_tier("high")
        self.assertEqual(check_task_count(["t1"], tier), [])
        for s in BudgetStatus:
            if s.is_terminal():
                self.assertIn(s, {
                    BudgetStatus.PASS,
                    BudgetStatus.ACTIVE_BUDGET_EXHAUSTED,
                    BudgetStatus.INFRASTRUCTURE_FAILURE,
                })


class TestCustomTierEnforcement(unittest.TestCase):
    """Enforcement with custom (non-production) tiers."""

    def test_strict_tier(self):
        tier = TierConfig("strict", 60, 10, 1, 1, 2)
        self.assertEqual(check_task_count(["t1"], tier), [])
        self.assertGreater(len(check_task_count(["t1", "t2"], tier)), 0)
        self.assertEqual(check_part_count(["p1", "p2"], tier), [])
        self.assertGreater(len(check_part_count(["p1", "p2", "p3"], tier)), 0)
        self.assertEqual(enforce_simulation_timeout(tier), 10.0)

    def test_generous_tier(self):
        tier = TierConfig("generous", 86400, 3600, 100, 0, 1000)
        tasks = [f"t{i}" for i in range(100)]
        parts = [f"p{i}" for i in range(1000)]
        self.assertEqual(check_task_count(tasks, tier), [])
        self.assertEqual(check_part_count(parts, tier), [])
        self.assertEqual(enforce_simulation_timeout(tier, 1800.0), 1800.0)

    def test_minimal_tier(self):
        tier = TierConfig("min", 1, 1, 1, 0, 1)
        self.assertEqual(check_task_count(["t"], tier), [])
        self.assertEqual(check_part_count(["p"], tier), [])
        self.assertEqual(enforce_simulation_timeout(tier), 1.0)
        self.assertGreater(len(check_task_count(["t1", "t2"], tier)), 0)
        self.assertGreater(len(check_part_count(["p1", "p2"], tier)), 0)


if __name__ == "__main__":
    unittest.main()
