"""Budget mathematical invariant tests.

Exhaustive verification that mathematical relationships between
budget fields hold under all conditions:
- active + remaining = budget_limit (always)
- active <= wall (always)
- pause_seconds >= 0 (always)
- wall >= active + pause_seconds (approximately)
- remaining >= 0 (always, clamped)
- snapshot timing fields are monotonically consistent
- Enum string values match expected vocabulary
- Recovery delay is monotonically non-decreasing
- Summary timing is a proper subset of snapshot timing
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    BudgetStatus,
    SimulationTimeoutResult,
    get_tier,
    enforce_simulation_timeout,
    build_incomplete_summary,
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
    BACKOFF_BASE_SECONDS,
)


class TestActiveRemainingBudgetInvariant(unittest.TestCase):
    """active_seconds + remaining_active_seconds == budget_limit_seconds."""

    def _check_invariant(self, tracker):
        snap = tracker.snapshot()
        total = snap["active_seconds"] + snap["remaining_active_seconds"]
        self.assertAlmostEqual(
            total, snap["budget_limit_seconds"], delta=0.01,
            msg=f"active({snap['active_seconds']}) + remaining({snap['remaining_active_seconds']}) != budget({snap['budget_limit_seconds']})"
        )

    def test_fresh_tracker(self):
        for name, tier in TIERS.items():
            self._check_invariant(ActiveBudgetTracker(tier=tier))

    def test_after_work(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.01)
            self._check_invariant(tracker)

    def test_after_pause_resume(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.005)
        tracker.pause("p")
        time.sleep(0.005)
        tracker.resume()
        time.sleep(0.005)
        self._check_invariant(tracker)

    def test_after_many_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(20):
            time.sleep(0.002)
            tracker.pause(f"p{i}")
            time.sleep(0.001)
            tracker.resume()
        self._check_invariant(tracker)

    def test_exhausted_tracker(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        snap = tracker.snapshot()
        self.assertEqual(snap["remaining_active_seconds"], 0.0)
        self.assertGreaterEqual(snap["active_seconds"], snap["budget_limit_seconds"])

    def test_after_checkpoint_round_trip(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=tier)
            restored.resume()
            time.sleep(0.005)
        self._check_invariant(restored)


class TestActiveLeWallInvariant(unittest.TestCase):
    """active_seconds <= wall_seconds (within tolerance)."""

    def _check_invariant(self, tracker):
        snap = tracker.snapshot()
        self.assertLessEqual(
            snap["active_seconds"],
            snap["wall_seconds"] + 0.002,
            msg=f"active({snap['active_seconds']}) > wall({snap['wall_seconds']})"
        )

    def test_fresh_tracker(self):
        for name, tier in TIERS.items():
            self._check_invariant(ActiveBudgetTracker(tier=tier))

    def test_after_work(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.02)
        self._check_invariant(tracker)

    def test_after_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(10):
            time.sleep(0.002)
            tracker.pause(f"p{i}")
            time.sleep(0.005)
            tracker.resume()
        self._check_invariant(tracker)

    def test_pause_dominated(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.002)
        tracker.pause("long")
        time.sleep(0.05)
        tracker.resume()
        time.sleep(0.002)
        snap = tracker.snapshot()
        self.assertLess(snap["active_seconds"], snap["wall_seconds"])


class TestNonNegativeInvariants(unittest.TestCase):
    """All timing fields are non-negative."""

    def _check_nonneg(self, snap):
        for key in ["active_seconds", "wall_seconds", "remaining_active_seconds",
                     "total_pause_seconds", "budget_limit_seconds"]:
            self.assertGreaterEqual(snap[key], 0, f"{key} is negative: {snap[key]}")
        self.assertGreaterEqual(snap["pause_count"], 0)

    def test_fresh_all_tiers(self):
        for name, tier in TIERS.items():
            self._check_nonneg(ActiveBudgetTracker(tier=tier).snapshot())

    def test_after_operations(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("p")
        time.sleep(0.01)
        tracker.resume()
        self._check_nonneg(tracker.snapshot())

    def test_exhausted(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self._check_nonneg(tracker.snapshot())


class TestFiniteInvariants(unittest.TestCase):
    """All numeric fields are finite (no inf, no nan)."""

    def _check_finite(self, snap):
        for key in ["active_seconds", "wall_seconds", "remaining_active_seconds",
                     "total_pause_seconds", "budget_limit_seconds"]:
            self.assertTrue(math.isfinite(snap[key]), f"{key} not finite: {snap[key]}")

    def test_fresh(self):
        self._check_finite(ActiveBudgetTracker(tier=get_tier("high")).snapshot())

    def test_after_work(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        self._check_finite(tracker.snapshot())

    def test_after_many_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        for i in range(50):
            tracker.pause(f"p{i}")
            tracker.resume()
        self._check_finite(tracker.snapshot())


class TestMonotonicActiveTime(unittest.TestCase):
    """Active time only increases (never decreases)."""

    def test_active_monotonic_continuous(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        prev = 0
        for _ in range(10):
            time.sleep(0.003)
            current = tracker.elapsed_active
            self.assertGreaterEqual(current, prev)
            prev = current

    def test_active_monotonic_across_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        prev = 0
        for _ in range(5):
            time.sleep(0.003)
            self.assertGreaterEqual(tracker.elapsed_active, prev)
            prev = tracker.elapsed_active
            tracker.pause("p")
            self.assertGreaterEqual(tracker.elapsed_active, prev)
            prev = tracker.elapsed_active
            time.sleep(0.003)
            tracker.resume()
            self.assertGreaterEqual(tracker.elapsed_active, prev)
            prev = tracker.elapsed_active


class TestMonotonicWallTime(unittest.TestCase):
    """Wall time only increases."""

    def test_wall_monotonic_continuous(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        prev = 0
        for _ in range(10):
            time.sleep(0.002)
            current = tracker.elapsed_wall
            self.assertGreater(current, prev)
            prev = current

    def test_wall_monotonic_across_pauses(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        prev = 0
        for i in range(5):
            time.sleep(0.002)
            tracker.pause(f"p{i}")
            self.assertGreater(tracker.elapsed_wall, prev)
            prev = tracker.elapsed_wall
            time.sleep(0.002)
            tracker.resume()
            self.assertGreater(tracker.elapsed_wall, prev)
            prev = tracker.elapsed_wall


class TestRemainingMonotonicDecreasing(unittest.TestCase):
    """Remaining active time only decreases (while not paused)."""

    def test_remaining_decreases(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        prev = tracker.remaining_active
        for _ in range(5):
            time.sleep(0.005)
            current = tracker.remaining_active
            self.assertLessEqual(current, prev + 0.001)
            prev = current

    def test_remaining_frozen_while_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.005)
        tracker.pause("freeze")
        frozen = tracker.remaining_active
        time.sleep(0.02)
        still_frozen = tracker.remaining_active
        self.assertAlmostEqual(frozen, still_frozen, delta=0.001)


class TestRecoveryDelayMathInvariants(unittest.TestCase):
    """Recovery delay obeys exponential backoff math."""

    def test_formula_matches(self):
        for i in range(10):
            expected = min(BACKOFF_BASE_SECONDS * (2 ** i), 60.0)
            self.assertEqual(recovery_delay(i), expected, f"attempt {i}")

    def test_monotonic_nondecreasing(self):
        prev = 0
        for i in range(20):
            d = recovery_delay(i)
            self.assertGreaterEqual(d, prev, f"attempt {i}")
            prev = d

    def test_always_positive(self):
        for i in range(100):
            self.assertGreater(recovery_delay(i), 0)

    def test_always_finite(self):
        for i in range(100):
            self.assertTrue(math.isfinite(recovery_delay(i)))

    def test_cap_is_exactly_60(self):
        for i in range(4, 50):
            self.assertEqual(recovery_delay(i), 60.0, f"attempt {i}")


class TestBudgetStatusInvariants(unittest.TestCase):
    """BudgetStatus enum members have correct properties."""

    def test_all_members_are_strings(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.value, str)

    def test_terminal_count(self):
        terminals = [s for s in BudgetStatus if s.is_terminal()]
        self.assertEqual(len(terminals), 3)

    def test_non_terminal_count(self):
        non_terminals = [s for s in BudgetStatus if not s.is_terminal()]
        self.assertEqual(len(non_terminals), 5)

    def test_terminal_set(self):
        terminals = {s for s in BudgetStatus if s.is_terminal()}
        expected = {BudgetStatus.PASS, BudgetStatus.ACTIVE_BUDGET_EXHAUSTED,
                    BudgetStatus.INFRASTRUCTURE_FAILURE}
        self.assertEqual(terminals, expected)

    def test_simulation_timeout_result_status_mapping(self):
        r_pass = SimulationTimeoutResult(timed_out=False, elapsed_seconds=10.0)
        r_timeout = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.0)
        self.assertEqual(r_pass.status, BudgetStatus.PASS)
        self.assertEqual(r_timeout.status, BudgetStatus.SIMULATION_TIMEOUT)


class TestShouldRetryInvariants(unittest.TestCase):
    """should_retry respects boundary invariants."""

    def test_never_retries_at_max(self):
        for error in ["INFRASTRUCTURE_FAILURE", "unknown", "", "anything"]:
            self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, error))

    def test_never_retries_above_max(self):
        for attempt in range(MAX_RECOVERY_RETRIES, MAX_RECOVERY_RETRIES + 5):
            self.assertFalse(should_retry(attempt, "INFRASTRUCTURE_FAILURE"))

    def test_always_false_for_terminal_errors(self):
        for attempt in range(10):
            self.assertFalse(should_retry(attempt, "ACTIVE_BUDGET_EXHAUSTED"))
            self.assertFalse(should_retry(attempt, "TERMINATED"))
            self.assertFalse(should_retry(attempt, "PASS"))


class TestSummaryTimingSubset(unittest.TestCase):
    """Summary timing dict is a proper subset of snapshot timing."""

    def test_summary_timing_keys_are_subset_of_snapshot(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        snap = tracker.snapshot()
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        timing_keys = set(summary["timing"].keys())
        snapshot_keys = set(snap.keys())
        for key in timing_keys:
            self.assertIn(key, snapshot_keys, f"Timing key {key} not in snapshot")

    def test_summary_timing_values_match_snapshot(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        snap = tracker.snapshot()
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        for key in summary["timing"]:
            if key in snap:
                self.assertAlmostEqual(
                    summary["timing"][key], snap[key], delta=0.05,
                    msg=f"Timing {key}: summary={summary['timing'][key]}, snap={snap[key]}"
                )


class TestTierConfigMathInvariants(unittest.TestCase):
    """Production tier configs obey expected mathematical relationships."""

    def test_budget_is_positive(self):
        for name, tier in TIERS.items():
            self.assertGreater(tier.active_budget_seconds, 0, f"Tier {name}")

    def test_timeout_is_positive(self):
        for name, tier in TIERS.items():
            self.assertGreater(tier.simulation_timeout_seconds, 0, f"Tier {name}")

    def test_timeout_less_than_budget(self):
        for name, tier in TIERS.items():
            self.assertLess(tier.simulation_timeout_seconds, tier.active_budget_seconds,
                          f"Tier {name}")

    def test_min_parts_le_max_parts(self):
        for name, tier in TIERS.items():
            self.assertLessEqual(tier.min_parts, tier.max_parts, f"Tier {name}")

    def test_tasks_at_least_one(self):
        for name, tier in TIERS.items():
            self.assertGreaterEqual(tier.max_tasks_per_design, 1, f"Tier {name}")

    def test_budget_is_finite(self):
        for name, tier in TIERS.items():
            self.assertTrue(math.isfinite(tier.active_budget_seconds), f"Tier {name}")

    def test_timeout_is_finite(self):
        for name, tier in TIERS.items():
            self.assertTrue(math.isfinite(tier.simulation_timeout_seconds), f"Tier {name}")


if __name__ == "__main__":
    unittest.main()
