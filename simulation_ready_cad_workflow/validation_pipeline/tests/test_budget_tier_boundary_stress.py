"""Tier boundary stress tests.

Validates behavior at exact tier boundaries, cross-tier checkpoint
migration under stress, TierConfig validation edge cases, and
part/task/timeout limits at exact boundary values.
"""
from __future__ import annotations

import json
import math
import tempfile
import threading
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
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    build_incomplete_summary,
    recovery_delay,
    should_retry,
    MAX_RECOVERY_RETRIES,
    BACKOFF_BASE_SECONDS,
)


class TestTierConfigValidation(unittest.TestCase):
    """TierConfig rejects invalid configurations."""

    def test_zero_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 0, 60, 1, 0, 10)

    def test_negative_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -1, 60, 1, 0, 10)

    def test_inf_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("inf"), 60, 1, 0, 10)

    def test_nan_budget_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("nan"), 60, 1, 0, 10)

    def test_zero_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 0, 1, 0, 10)

    def test_negative_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, -1, 1, 0, 10)

    def test_inf_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, float("inf"), 1, 0, 10)

    def test_nan_timeout_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, float("nan"), 1, 0, 10)

    def test_zero_tasks_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 0, 0, 10)

    def test_negative_tasks_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, -1, 0, 10)

    def test_zero_max_parts_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 0, 0)

    def test_min_greater_than_max_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 11, 10)

    def test_negative_min_parts_raises(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, -1, 10)

    def test_minimal_valid_config(self):
        t = TierConfig("minimal", 0.001, 0.001, 1, 0, 1)
        self.assertEqual(t.name, "minimal")
        self.assertEqual(t.active_budget_seconds, 0.001)
        self.assertEqual(t.simulation_timeout_seconds, 0.001)

    def test_large_valid_config(self):
        t = TierConfig("huge", 1e9, 1e6, 1000, 0, 100000)
        self.assertEqual(t.active_budget_seconds, 1e9)

    def test_min_equals_max_parts_valid(self):
        t = TierConfig("exact", 3600, 60, 1, 5, 5)
        self.assertEqual(t.min_parts, 5)
        self.assertEqual(t.max_parts, 5)

    def test_frozen_dataclass(self):
        t = get_tier("light")
        with self.assertRaises(AttributeError):
            t.name = "hacked"
        with self.assertRaises(AttributeError):
            t.active_budget_seconds = 999999


class TestGetTierBoundary(unittest.TestCase):
    """get_tier lookup edge cases."""

    def test_all_known_tiers_resolve(self):
        for name in ("light", "medium", "high"):
            tier = get_tier(name)
            self.assertEqual(tier.name, name)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            get_tier("ultra")

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            get_tier("")

    def test_case_sensitive(self):
        with self.assertRaises(ValueError):
            get_tier("Light")
        with self.assertRaises(ValueError):
            get_tier("MEDIUM")

    def test_tier_order_is_ascending_budget(self):
        tiers = [get_tier(n) for n in ("light", "medium", "high")]
        for i in range(len(tiers) - 1):
            self.assertLess(tiers[i].active_budget_seconds, tiers[i+1].active_budget_seconds)

    def test_tier_order_is_ascending_timeout(self):
        tiers = [get_tier(n) for n in ("light", "medium", "high")]
        for i in range(len(tiers) - 1):
            self.assertLess(tiers[i].simulation_timeout_seconds, tiers[i+1].simulation_timeout_seconds)

    def test_tier_order_is_ascending_parts(self):
        tiers = [get_tier(n) for n in ("light", "medium", "high")]
        for i in range(len(tiers) - 1):
            self.assertLess(tiers[i].max_parts, tiers[i+1].max_parts)


class TestPartCountBoundary(unittest.TestCase):
    """Part count checks at exact boundary values."""

    def test_at_max_no_violation(self):
        for name, tier in TIERS.items():
            bodies = [{}] * tier.max_parts
            self.assertEqual(check_part_count(bodies, tier), [], f"Tier {name}")

    def test_one_over_max_has_violation(self):
        for name, tier in TIERS.items():
            bodies = [{}] * (tier.max_parts + 1)
            violations = check_part_count(bodies, tier)
            self.assertEqual(len(violations), 1, f"Tier {name}")

    def test_one_under_max_no_violation(self):
        for name, tier in TIERS.items():
            if tier.max_parts > 0:
                bodies = [{}] * (tier.max_parts - 1)
                self.assertEqual(check_part_count(bodies, tier), [], f"Tier {name}")

    def test_zero_parts_no_violation(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([], tier), [], f"Tier {name}")

    def test_single_part_no_violation(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([{}], tier), [], f"Tier {name}")

    def test_double_max_has_violation(self):
        for name, tier in TIERS.items():
            bodies = [{}] * (tier.max_parts * 2)
            violations = check_part_count(bodies, tier)
            self.assertEqual(len(violations), 1, f"Tier {name}")
            self.assertIn(str(tier.max_parts * 2), violations[0])


class TestTaskCountBoundary(unittest.TestCase):
    """Task count checks at exact boundary values."""

    def test_at_max_no_violation(self):
        for name, tier in TIERS.items():
            tasks = [{}] * tier.max_tasks_per_design
            self.assertEqual(check_task_count(tasks, tier), [], f"Tier {name}")

    def test_one_over_max_has_violation(self):
        for name, tier in TIERS.items():
            tasks = [{}] * (tier.max_tasks_per_design + 1)
            violations = check_task_count(tasks, tier)
            self.assertEqual(len(violations), 1, f"Tier {name}")

    def test_one_task_no_violation(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_task_count([{}], tier), [], f"Tier {name}")

    def test_zero_tasks_no_violation(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_task_count([], tier), [], f"Tier {name}")

    def test_large_overshoot_has_exactly_one_violation(self):
        for name, tier in TIERS.items():
            tasks = [{}] * 100
            violations = check_task_count(tasks, tier)
            self.assertEqual(len(violations), 1, f"Tier {name}")


class TestSimulationTimeoutBoundary(unittest.TestCase):
    """Simulation timeout enforcement at exact boundary values."""

    def test_at_tier_limit_returns_limit(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds)
            self.assertEqual(result, tier.simulation_timeout_seconds, f"Tier {name}")

    def test_one_over_limit_clamps(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds + 1)
            self.assertEqual(result, tier.simulation_timeout_seconds, f"Tier {name}")

    def test_one_under_limit_passes_through(self):
        for name, tier in TIERS.items():
            requested = tier.simulation_timeout_seconds - 1
            if requested > 0:
                result = enforce_simulation_timeout(tier, requested)
                self.assertEqual(result, requested, f"Tier {name}")

    def test_epsilon_above_zero_accepted(self):
        tier = get_tier("light")
        result = enforce_simulation_timeout(tier, 0.001)
        self.assertEqual(result, 0.001)

    def test_zero_raises(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError, msg=f"Tier {name}"):
                enforce_simulation_timeout(tier, 0)

    def test_negative_raises(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError, msg=f"Tier {name}"):
                enforce_simulation_timeout(tier, -1)

    def test_inf_raises(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError, msg=f"Tier {name}"):
                enforce_simulation_timeout(tier, float("inf"))

    def test_nan_raises(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError, msg=f"Tier {name}"):
                enforce_simulation_timeout(tier, float("nan"))

    def test_none_returns_tier_max(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier)
            self.assertEqual(result, tier.simulation_timeout_seconds, f"Tier {name}")

    def test_very_small_timeout_passes(self):
        tier = get_tier("high")
        result = enforce_simulation_timeout(tier, 0.0001)
        self.assertAlmostEqual(result, 0.0001)


class TestBudgetExhaustionBoundary(unittest.TestCase):
    """Behavior at the exact moment of budget exhaustion."""

    def test_micro_budget_exhaustion(self):
        tier = TierConfig("micro", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertTrue(tracker.is_exhausted)
        self.assertEqual(tracker.remaining_active, 0.0)
        self.assertFalse(tracker.check_budget())

    def test_remaining_never_negative(self):
        tier = TierConfig("micro", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertEqual(tracker.remaining_active, 0.0)
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["remaining_active_seconds"], 0.0)

    def test_exhausted_snapshot_active_exceeds_budget(self):
        tier = TierConfig("micro", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.015)
        snap = tracker.snapshot()
        self.assertGreaterEqual(snap["active_seconds"], tier.active_budget_seconds)

    def test_not_exhausted_just_before(self):
        tier = TierConfig("generous", 100, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        self.assertFalse(tracker.is_exhausted)
        self.assertTrue(tracker.check_budget())
        self.assertGreater(tracker.remaining_active, 0)


class TestCrossTierCheckpointMigration(unittest.TestCase):
    """Checkpoint migration between all tier pairs."""

    def test_light_to_medium(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("medium"))
            restored.resume()
        snap = restored.snapshot()
        self.assertEqual(snap["tier"], "medium")
        self.assertEqual(snap["budget_limit_seconds"], 7200)

    def test_light_to_high(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            restored.resume()
        snap = restored.snapshot()
        self.assertEqual(snap["tier"], "high")
        self.assertEqual(snap["budget_limit_seconds"], 10800)

    def test_medium_to_high(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            restored.resume()
        snap = restored.snapshot()
        self.assertEqual(snap["tier"], "high")

    def test_high_to_light_downgrade(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            restored.resume()
        snap = restored.snapshot()
        self.assertEqual(snap["tier"], "light")
        self.assertEqual(snap["budget_limit_seconds"], 3600)

    def test_medium_to_light_downgrade(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("light"))
            restored.resume()
        snap = restored.snapshot()
        self.assertEqual(snap["tier"], "light")

    def test_same_tier_round_trip(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / f"cp_{name}.json"
                tracker.save_checkpoint(cp)
                restored = ActiveBudgetTracker.restore(cp, tier=tier)
                restored.resume()
            snap = restored.snapshot()
            self.assertEqual(snap["tier"], name, f"Tier {name}")

    def test_migration_preserves_active_time(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.02)
        active_before = tracker.elapsed_active
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp, tier=get_tier("high"))
            restored.resume()
        self.assertAlmostEqual(restored.elapsed_active, active_before, delta=0.01)

    def test_chain_through_all_tiers(self):
        tier_names = ["light", "medium", "high"]
        tracker = ActiveBudgetTracker(tier=get_tier(tier_names[0]))
        time.sleep(0.005)

        with tempfile.TemporaryDirectory() as tmp:
            for i, name in enumerate(tier_names[1:]):
                cp = Path(tmp) / f"chain_{i}.json"
                tracker.save_checkpoint(cp)
                tracker = ActiveBudgetTracker.restore(cp, tier=get_tier(name))
                tracker.resume()
                time.sleep(0.005)

        snap = tracker.snapshot()
        self.assertEqual(snap["tier"], "high")
        self.assertGreater(snap["active_seconds"], 0.01)
        self.assertGreater(snap["pause_count"], 0)


class TestRecoveryDelayBoundary(unittest.TestCase):
    """Recovery delay schedule at boundary values."""

    def test_attempt_zero(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_attempt_one(self):
        self.assertEqual(recovery_delay(1), BACKOFF_BASE_SECONDS * 2)

    def test_attempt_two(self):
        self.assertEqual(recovery_delay(2), BACKOFF_BASE_SECONDS * 4)

    def test_attempt_three(self):
        self.assertEqual(recovery_delay(3), BACKOFF_BASE_SECONDS * 8)

    def test_cap_at_60(self):
        self.assertEqual(recovery_delay(4), 60.0)
        self.assertEqual(recovery_delay(5), 60.0)
        self.assertEqual(recovery_delay(100), 60.0)
        self.assertEqual(recovery_delay(1000), 60.0)

    def test_monotonic_up_to_cap(self):
        prev = 0
        for i in range(10):
            d = recovery_delay(i)
            self.assertGreaterEqual(d, prev)
            prev = d

    def test_exact_schedule(self):
        expected = [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
        for i, exp in enumerate(expected):
            self.assertEqual(recovery_delay(i), exp, f"attempt {i}")


class TestShouldRetryBoundary(unittest.TestCase):
    """Retry decision at boundary attempt counts."""

    def test_attempt_zero_infra_retries(self):
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE"))

    def test_at_max_retries_stops(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "INFRASTRUCTURE_FAILURE"))

    def test_one_before_max_retries(self):
        self.assertTrue(should_retry(MAX_RECOVERY_RETRIES - 1, "INFRASTRUCTURE_FAILURE"))

    def test_exhausted_never_retries(self):
        for attempt in range(MAX_RECOVERY_RETRIES + 2):
            self.assertFalse(should_retry(attempt, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_terminated_never_retries(self):
        for attempt in range(MAX_RECOVERY_RETRIES + 2):
            self.assertFalse(should_retry(attempt, "TERMINATED"))

    def test_pass_never_retries(self):
        for attempt in range(MAX_RECOVERY_RETRIES + 2):
            self.assertFalse(should_retry(attempt, "PASS"))

    def test_partial_match_stops_retry(self):
        self.assertFalse(should_retry(0, "Error: ACTIVE_BUDGET_EXHAUSTED after 3600s"))
        self.assertFalse(should_retry(0, "Process TERMINATED unexpectedly"))

    def test_unknown_error_retries(self):
        self.assertTrue(should_retry(0, "something unexpected"))
        self.assertTrue(should_retry(1, "random error"))

    def test_empty_error_retries(self):
        self.assertTrue(should_retry(0, ""))


class TestBudgetStatusTerminality(unittest.TestCase):
    """BudgetStatus.is_terminal() at every enum member."""

    def test_pass_is_terminal(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_exhausted_is_terminal(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())

    def test_infra_failure_is_terminal(self):
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


class TestSimulationTimeoutResultBoundary(unittest.TestCase):
    """SimulationTimeoutResult at boundary conditions."""

    def test_zero_elapsed(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=0.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_timed_out_status(self):
        r = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_not_timed_out_status(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=30.0)
        self.assertEqual(r.status, BudgetStatus.PASS)

    def test_optional_fields_default_none(self):
        r = SimulationTimeoutResult(timed_out=False, elapsed_seconds=1.0)
        self.assertIsNone(r.partial_evidence)
        self.assertIsNone(r.log_path)
        self.assertIsNone(r.error)

    def test_with_all_optional_fields(self):
        r = SimulationTimeoutResult(
            timed_out=True,
            elapsed_seconds=60.0,
            partial_evidence={"frames": 100},
            log_path=Path("/tmp/sim.log"),
            error="timeout exceeded",
        )
        self.assertEqual(r.partial_evidence, {"frames": 100})
        self.assertEqual(r.error, "timeout exceeded")


class TestPauseResumeBoundary(unittest.TestCase):
    """Pause/resume at boundary conditions."""

    def test_double_pause_is_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("first")
        snap1 = tracker.snapshot()
        tracker.pause("second")
        snap2 = tracker.snapshot()
        self.assertEqual(snap1["pause_count"], snap2["pause_count"])

    def test_double_resume_is_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("p")
        tracker.resume()
        snap1 = tracker.snapshot()
        tracker.resume()
        snap2 = tracker.snapshot()
        self.assertEqual(snap1["pause_count"], snap2["pause_count"])

    def test_resume_without_pause_is_noop(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        snap_before = tracker.snapshot()
        tracker.resume()
        snap_after = tracker.snapshot()
        self.assertEqual(snap_before["pause_count"], snap_after["pause_count"])

    def test_rapid_pause_resume_cycles(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        for i in range(100):
            tracker.pause(f"rapid_{i}")
            tracker.resume()
        snap = tracker.snapshot()
        self.assertEqual(snap["pause_count"], 100)
        self.assertFalse(tracker.is_exhausted)

    def test_paused_active_time_frozen(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("freeze")
        active_at_pause = tracker.elapsed_active
        time.sleep(0.02)
        active_during_pause = tracker.elapsed_active
        self.assertAlmostEqual(active_at_pause, active_during_pause, delta=0.001)

    def test_paused_wall_time_advances(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("wait")
        wall_at_pause = tracker.elapsed_wall
        time.sleep(0.02)
        wall_during_pause = tracker.elapsed_wall
        self.assertGreater(wall_during_pause, wall_at_pause)


class TestConcurrentTrackerStress(unittest.TestCase):
    """Multiple trackers under concurrent access."""

    def test_ten_independent_trackers(self):
        results = [None] * 10
        errors = []

        def run_tracker(idx):
            try:
                tier = get_tier(["light", "medium", "high"][idx % 3])
                tracker = ActiveBudgetTracker(tier=tier)
                for _ in range(20):
                    time.sleep(0.001)
                    tracker.pause(f"p_{idx}")
                    time.sleep(0.001)
                    tracker.resume()
                snap = tracker.snapshot()
                results[idx] = snap
            except Exception as e:
                errors.append((idx, e))

        threads = [threading.Thread(target=run_tracker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        self.assertEqual(errors, [])
        for i, snap in enumerate(results):
            self.assertIsNotNone(snap, f"Tracker {i} no result")
            self.assertEqual(snap["pause_count"], 20, f"Tracker {i}")
            self.assertGreater(snap["active_seconds"], 0, f"Tracker {i}")


class TestCheckpointCorruptionRecovery(unittest.TestCase):
    """Checkpoint restore rejects corrupted data."""

    def test_missing_schema_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "bad.json"
            cp.write_text(json.dumps({"tier": "light", "active_seconds": 10}))
            with self.assertRaises(ValueError):
                ActiveBudgetTracker.restore(cp, tier=get_tier("light"))

    def test_wrong_schema_version_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "bad.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v999",
                "tier": "light", "active_seconds": 10,
            }))
            with self.assertRaises(ValueError):
                ActiveBudgetTracker.restore(cp, tier=get_tier("light"))

    def test_invalid_json_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "bad.json"
            cp.write_text("not json at all")
            with self.assertRaises(json.JSONDecodeError):
                ActiveBudgetTracker.restore(cp, tier=get_tier("light"))

    def test_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "empty.json"
            cp.write_text("")
            with self.assertRaises(json.JSONDecodeError):
                ActiveBudgetTracker.restore(cp, tier=get_tier("light"))

    def test_unknown_tier_in_checkpoint_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "custom.json"
            cp.write_text(json.dumps({
                "schema_version": "budget_checkpoint_v1",
                "tier": "custom_unknown_tier",
                "active_seconds": 100,
                "budget_limit_seconds": 5000,
                "pause_count": 2,
                "total_pause_seconds": 50,
            }))
            restored = ActiveBudgetTracker.restore(cp)
            restored.resume()
            snap = restored.snapshot()
            self.assertEqual(snap["tier"], "custom_unknown_tier")
            self.assertAlmostEqual(snap["active_seconds"], 100, delta=0.1)


class TestIncompleteSummaryBoundary(unittest.TestCase):
    """build_incomplete_summary at boundary conditions."""

    def test_empty_everything(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(summary["completed_tests"], [])
        self.assertEqual(summary["pending_tests"], [])

    def test_many_completed_tests(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": f"t{i}", "physical_pass": True, "geometry_audit": {"passed": True}}
            for i in range(50)
        ]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(len(summary["completed_tests"]), 50)
        self.assertEqual(summary["severity"], "minor")

    def test_all_failed_tests(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": f"t{i}", "physical_pass": False, "geometry_audit": {"passed": True}}
            for i in range(10)
        ]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=["remaining"],
            issues=["all failed"], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")
        for entry in summary["completed_tests"]:
            self.assertEqual(entry["status"], "FAIL")

    def test_mixed_pass_fail(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        completed = [
            {"test_id": "pass1", "physical_pass": True, "geometry_audit": {"passed": True}},
            {"test_id": "fail1", "physical_pass": False, "geometry_audit": {"passed": True}},
            {"test_id": "pass2", "physical_pass": True, "geometry_audit": {"passed": True}},
        ]
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=completed, pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        statuses = [e["status"] for e in summary["completed_tests"]]
        self.assertEqual(statuses, ["PASS", "FAIL", "PASS"])


if __name__ == "__main__":
    unittest.main()
