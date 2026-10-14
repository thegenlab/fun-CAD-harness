"""Property-based tests for budget module invariants.

Verifies mathematical properties that must hold regardless of input:
- Active + pause ≤ wall time
- Snapshot always JSON-serializable
- Checkpoint round-trip preserves active seconds within tolerance
- Tier limits are monotonically increasing (light < medium < high)
- Recovery delay is monotonically non-decreasing up to cap
- Incomplete summary schema is always valid
- TierConfig validation rejects all invalid inputs
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
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    get_tier,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    build_incomplete_summary,
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
    BACKOFF_BASE_SECONDS,
)


class TestTimingInvariants(unittest.TestCase):
    """Mathematical invariants of budget time tracking."""

    def test_active_never_exceeds_wall(self):
        """For any sequence of pause/resume, active ≤ wall."""
        for tier_name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(tier_name))
            for i in range(10):
                time.sleep(0.005)
                if i % 2 == 0:
                    tracker.pause(f"p{i}")
                else:
                    tracker.resume()
            if tracker.is_paused:
                tracker.resume()
            self.assertLessEqual(
                tracker.elapsed_active, tracker.elapsed_wall + 0.01,
                f"Tier {tier_name}: active must never exceed wall",
            )

    def test_remaining_plus_active_equals_budget(self):
        """remaining_active + elapsed_active ≈ budget_limit for non-exhausted."""
        for tier_name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(tier_name))
            time.sleep(0.01)
            total = tracker.elapsed_active + tracker.remaining_active
            self.assertAlmostEqual(
                total, tracker.tier.active_budget_seconds, delta=0.05,
                msg=f"Tier {tier_name}: remaining + active must equal budget limit",
            )

    def test_remaining_is_zero_when_exhausted(self):
        """Once exhausted, remaining_active is exactly 0."""
        tiny = TierConfig("tiny", 0.01, 5, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_pause_count_never_decreases(self):
        """pause_count is monotonically non-decreasing."""
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        counts = []
        for i in range(20):
            tracker.pause(f"p{i}")
            counts.append(tracker.snapshot()["pause_count"])
            tracker.resume()
        for i in range(1, len(counts)):
            self.assertGreaterEqual(counts[i], counts[i - 1])

    def test_wall_time_always_increases(self):
        """wall_seconds strictly increases over time."""
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        snaps = []
        for _ in range(5):
            time.sleep(0.01)
            snaps.append(tracker.snapshot()["wall_seconds"])
        for i in range(1, len(snaps)):
            self.assertGreater(snaps[i], snaps[i - 1])

    def test_active_time_frozen_during_pause(self):
        """Active seconds do not change while paused."""
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        tracker.pause("freeze")
        frozen = tracker.elapsed_active
        time.sleep(0.03)
        self.assertAlmostEqual(tracker.elapsed_active, frozen, delta=0.001)

    def test_snapshot_pause_seconds_consistency(self):
        """total_pause_seconds + active_seconds ≈ wall_seconds."""
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for i in range(5):
            time.sleep(0.01)
            tracker.pause(f"p{i}")
            time.sleep(0.01)
            tracker.resume()
        snap = tracker.snapshot()
        computed_wall = snap["active_seconds"] + snap["total_pause_seconds"]
        self.assertAlmostEqual(computed_wall, snap["wall_seconds"], delta=0.02)


class TestCheckpointRoundTripFidelity(unittest.TestCase):
    """Checkpoint save/restore preserves all meaningful state."""

    def _round_trip(self, tier_name, pause_cycles=3):
        tier = get_tier(tier_name)
        tracker = ActiveBudgetTracker(tier=tier)
        for i in range(pause_cycles):
            time.sleep(0.005)
            tracker.pause(f"p{i}")
            time.sleep(0.005)
            tracker.resume()

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rt.json"
            before_snap = tracker.snapshot()
            tracker.save_checkpoint(path)
            restored = ActiveBudgetTracker.restore(path, tier=tier)
            after_snap = restored.snapshot()
            return before_snap, after_snap

    def test_all_tiers_round_trip(self):
        """Round-trip preserves active_seconds across all tiers."""
        for name in TIERS:
            before, after = self._round_trip(name)
            self.assertAlmostEqual(
                before["active_seconds"], after["active_seconds"], delta=0.02,
                msg=f"Tier {name}: active_seconds should survive round-trip",
            )

    def test_round_trip_increments_pause_count(self):
        """Restore adds +1 to pause_count (starts paused)."""
        for name in TIERS:
            before, after = self._round_trip(name, pause_cycles=2)
            self.assertEqual(after["pause_count"], before["pause_count"] + 1)

    def test_round_trip_schema_version_preserved(self):
        """Schema version is preserved through round-trip."""
        for name in TIERS:
            before, after = self._round_trip(name, pause_cycles=0)
            self.assertEqual(before["schema_version"], after["schema_version"])
            self.assertEqual(after["schema_version"], "budget_checkpoint_v1")

    def test_round_trip_budget_limit_preserved(self):
        """Budget limit is preserved through round-trip."""
        for name in TIERS:
            before, after = self._round_trip(name)
            self.assertEqual(before["budget_limit_seconds"],
                             after["budget_limit_seconds"])


class TestTierMonotonicity(unittest.TestCase):
    """Tier values increase monotonically: light < medium < high."""

    def test_active_budget_increases(self):
        self.assertLess(TIERS["light"].active_budget_seconds,
                        TIERS["medium"].active_budget_seconds)
        self.assertLess(TIERS["medium"].active_budget_seconds,
                        TIERS["high"].active_budget_seconds)

    def test_simulation_timeout_increases(self):
        self.assertLess(TIERS["light"].simulation_timeout_seconds,
                        TIERS["medium"].simulation_timeout_seconds)
        self.assertLess(TIERS["medium"].simulation_timeout_seconds,
                        TIERS["high"].simulation_timeout_seconds)

    def test_max_parts_increases(self):
        self.assertLess(TIERS["light"].max_parts,
                        TIERS["medium"].max_parts)
        self.assertLess(TIERS["medium"].max_parts,
                        TIERS["high"].max_parts)

    def test_max_tasks_non_decreasing(self):
        self.assertLessEqual(TIERS["light"].max_tasks_per_design,
                             TIERS["medium"].max_tasks_per_design)
        self.assertLessEqual(TIERS["medium"].max_tasks_per_design,
                             TIERS["high"].max_tasks_per_design)

    def test_tier_names_match_keys(self):
        for name, tier in TIERS.items():
            self.assertEqual(tier.name, name)


class TestRecoveryDelayMathProperties(unittest.TestCase):
    """Mathematical properties of recovery_delay."""

    def test_formula_matches_expectation(self):
        """recovery_delay(n) = min(BACKOFF_BASE * 2^n, 60)."""
        for n in range(20):
            expected = min(BACKOFF_BASE_SECONDS * (2 ** n), 60.0)
            self.assertEqual(recovery_delay(n), expected)

    def test_always_positive(self):
        for n in range(100):
            self.assertGreater(recovery_delay(n), 0)

    def test_always_finite(self):
        for n in range(100):
            self.assertTrue(math.isfinite(recovery_delay(n)))

    def test_cap_is_exactly_60(self):
        self.assertEqual(recovery_delay(1000), 60.0)


class TestShouldRetryProperties(unittest.TestCase):
    """Properties of should_retry decision function."""

    def test_always_false_at_max_retries(self):
        for err in ["SIMULATION_TIMEOUT", "HARNESS_ERROR", "random"]:
            self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, err))

    def test_always_false_for_non_retryable(self):
        for attempt in range(MAX_RECOVERY_RETRIES):
            self.assertFalse(should_retry(attempt, "ACTIVE_BUDGET_EXHAUSTED"))
            self.assertFalse(should_retry(attempt, "TERMINATED"))
            self.assertFalse(should_retry(attempt, "PASS"))

    def test_always_true_for_retryable_under_max(self):
        retryable = ["SIMULATION_TIMEOUT", "INFRASTRUCTURE_FAILURE",
                     "HARNESS_ERROR", "unknown_error"]
        for err in retryable:
            for attempt in range(MAX_RECOVERY_RETRIES):
                self.assertTrue(should_retry(attempt, err),
                                f"should_retry({attempt}, {err!r}) must be True")

    def test_substring_matching_behavior(self):
        """Non-retryable check uses substring matching."""
        self.assertFalse(should_retry(0, "Error: ACTIVE_BUDGET_EXHAUSTED in step 3"))
        self.assertFalse(should_retry(0, "Process TERMINATED unexpectedly"))
        self.assertFalse(should_retry(0, "Status: PASS - all criteria met"))


class TestTierConfigValidation(unittest.TestCase):
    """TierConfig rejects all invalid inputs."""

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 0, 60, 2, 0, 10)

    def test_negative_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -100, 60, 2, 0, 10)

    def test_infinite_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("inf"), 60, 2, 0, 10)

    def test_nan_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("nan"), 60, 2, 0, 10)

    def test_zero_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 0, 2, 0, 10)

    def test_negative_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, -10, 2, 0, 10)

    def test_zero_tasks_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 0, 0, 10)

    def test_zero_max_parts_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, 0, 0)

    def test_min_exceeds_max_parts_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, 11, 10)

    def test_negative_min_parts_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 2, -1, 10)

    def test_valid_minimal_config(self):
        tier = TierConfig("minimal", 1, 1, 1, 0, 1)
        self.assertEqual(tier.name, "minimal")

    def test_frozen_attribute_raises(self):
        tier = get_tier("light")
        with self.assertRaises(AttributeError):
            tier.active_budget_seconds = 9999


class TestEnforceTimeoutProperties(unittest.TestCase):
    """enforce_simulation_timeout properties."""

    def test_never_exceeds_tier_limit(self):
        for name, tier in TIERS.items():
            for requested in [1, 10, 100, 1000, 99999]:
                result = enforce_simulation_timeout(tier, requested)
                self.assertLessEqual(result, tier.simulation_timeout_seconds)

    def test_smaller_request_preserved(self):
        for name, tier in TIERS.items():
            small = tier.simulation_timeout_seconds / 2
            self.assertEqual(enforce_simulation_timeout(tier, small), small)

    def test_none_returns_tier_limit(self):
        for name, tier in TIERS.items():
            self.assertEqual(enforce_simulation_timeout(tier),
                             tier.simulation_timeout_seconds)

    def test_invalid_timeout_raises(self):
        tier = get_tier("light")
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, 0)
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, -1)
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("inf"))
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(tier, float("nan"))


class TestLimitCheckProperties(unittest.TestCase):
    """Properties of check_part_count and check_task_count."""

    def test_empty_always_passes(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([], tier), [])
            self.assertEqual(check_task_count([], tier), [])

    def test_at_limit_always_passes(self):
        for name, tier in TIERS.items():
            parts = [{} for _ in range(tier.max_parts)]
            tasks = [{} for _ in range(tier.max_tasks_per_design)]
            self.assertEqual(check_part_count(parts, tier), [])
            self.assertEqual(check_task_count(tasks, tier), [])

    def test_over_limit_always_fails(self):
        for name, tier in TIERS.items():
            parts = [{} for _ in range(tier.max_parts + 1)]
            tasks = [{} for _ in range(tier.max_tasks_per_design + 1)]
            self.assertGreater(len(check_part_count(parts, tier)), 0)
            self.assertGreater(len(check_task_count(tasks, tier)), 0)

    def test_violation_message_contains_counts(self):
        tier = get_tier("light")
        parts = [{} for _ in range(11)]
        violations = check_part_count(parts, tier)
        self.assertIn("11", violations[0])
        self.assertIn("10", violations[0])


class TestIncompleteSummarySchema(unittest.TestCase):
    """build_incomplete_summary always produces a valid schema."""

    REQUIRED_KEYS = {
        "schema_version", "status", "tier", "timing",
        "completed_tests", "pending_tests", "severity",
        "issues", "evidence_paths", "blockers", "generated_at",
    }

    def _make_summary(self, **kwargs):
        tier = kwargs.pop("tier", get_tier("light"))
        tracker = kwargs.pop("tracker", ActiveBudgetTracker(tier=tier))
        defaults = {
            "completed_tests": [],
            "pending_tests": [],
            "issues": [],
            "evidence_paths": [],
            "blockers": [],
        }
        defaults.update(kwargs)
        return build_incomplete_summary(tier=tier, tracker=tracker, **defaults)

    def test_always_has_required_keys(self):
        summary = self._make_summary()
        self.assertEqual(set(summary.keys()), self.REQUIRED_KEYS)

    def test_always_json_serializable(self):
        for tier_name in TIERS:
            tier = get_tier(tier_name)
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[{"test_id": "t", "physical_pass": True,
                                  "geometry_audit": {"passed": True}}],
                pending_tests=["p"],
                issues=["i"],
                evidence_paths=["/e"],
                blockers=["b"],
            )
            encoded = json.dumps(summary)
            decoded = json.loads(encoded)
            self.assertEqual(set(decoded.keys()), self.REQUIRED_KEYS)

    def test_status_is_always_budget_exhausted(self):
        summary = self._make_summary()
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")

    def test_schema_version_is_correct(self):
        summary = self._make_summary()
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_timing_has_required_subkeys(self):
        summary = self._make_summary()
        timing = summary["timing"]
        for key in ["active_seconds", "wall_seconds", "budget_limit_seconds",
                     "pause_count", "total_pause_seconds"]:
            self.assertIn(key, timing, f"timing missing key: {key}")

    def test_severity_binary(self):
        """Severity is always 'functional' or 'minor'."""
        s1 = self._make_summary(pending_tests=["t1"])
        self.assertEqual(s1["severity"], "functional")
        s2 = self._make_summary()
        self.assertEqual(s2["severity"], "minor")


class TestGetTierValidation(unittest.TestCase):
    """get_tier validates inputs correctly."""

    def test_known_tiers(self):
        for name in ["light", "medium", "high"]:
            tier = get_tier(name)
            self.assertEqual(tier.name, name)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError) as ctx:
            get_tier("ultra")
        self.assertIn("ultra", str(ctx.exception))

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_tier("Light")
        with self.assertRaises(ValueError):
            get_tier("MEDIUM")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            get_tier("")


class TestCriteriaAdjustmentProperties(unittest.TestCase):
    """CriteriaAdjustment invariant properties."""

    def test_frozen_always_raises(self):
        for i in range(10):
            with self.assertRaises(ValueError):
                CriteriaAdjustment(f"c{i}", {"a": 1}, {"a": 2}, "r", frozen=True)

    def test_no_change_always_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c", {"x": 1}, {"x": 1}, "no-op")

    def test_empty_reason_always_raises(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c", {"x": 1}, {"x": 2}, "")

    def test_valid_adjustment_has_timestamp(self):
        adj = CriteriaAdjustment("c", {"x": 1}, {"x": 2}, "test reason")
        self.assertTrue(len(adj.adjusted_at) > 0)

    def test_to_dict_round_trip(self):
        adj = CriteriaAdjustment("c", {"x": 1}, {"x": 2}, "test")
        d = adj.to_dict()
        self.assertEqual(d["criterion_id"], "c")
        self.assertEqual(d["before"], {"x": 1})
        self.assertEqual(d["after"], {"x": 2})
        self.assertEqual(d["reason"], "test")
        self.assertIn("adjusted_at", d)


if __name__ == "__main__":
    unittest.main()
