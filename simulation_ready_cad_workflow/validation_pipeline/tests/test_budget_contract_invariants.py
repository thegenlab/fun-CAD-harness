"""Budget module contract invariants.

Tests the guarantees that budget.py makes to all consumers:
- Tier immutability: frozen dataclass fields cannot be mutated
- Schema stability: checkpoint and summary schemas are forward-compatible
- State machine properties: tracker state transitions are monotonic
- Adjustment guard: frozen criteria always rejected regardless of arguments
- Recovery contract: backoff is deterministic and bounded
- Limit enforcement: never silently drops or inflates counts
- TierConfig validation: constructor rejects invalid configurations
- Budget exhaustion: is_exhausted transitions once and stays true
- Snapshot algebra: snapshot values satisfy timing invariants at every point
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import threading
import unittest
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path

from ..slim_harness.budget import (
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
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


class TestTierImmutability(unittest.TestCase):
    """TierConfig is frozen — fields cannot be mutated after construction."""

    def test_cannot_mutate_name(self):
        tier = get_tier("light")
        with self.assertRaises(FrozenInstanceError):
            tier.name = "hacked"

    def test_cannot_mutate_active_budget(self):
        tier = get_tier("medium")
        with self.assertRaises(FrozenInstanceError):
            tier.active_budget_seconds = 999999

    def test_cannot_mutate_timeout(self):
        tier = get_tier("high")
        with self.assertRaises(FrozenInstanceError):
            tier.simulation_timeout_seconds = 0

    def test_cannot_mutate_max_tasks(self):
        tier = get_tier("light")
        with self.assertRaises(FrozenInstanceError):
            tier.max_tasks_per_design = 100

    def test_cannot_mutate_max_parts(self):
        tier = get_tier("light")
        with self.assertRaises(FrozenInstanceError):
            tier.max_parts = 9999

    def test_cannot_mutate_min_parts(self):
        tier = get_tier("light")
        with self.assertRaises(FrozenInstanceError):
            tier.min_parts = -1

    def test_tiers_dict_values_are_frozen(self):
        """Every value in the TIERS registry is a frozen dataclass."""
        for name, tier in TIERS.items():
            with self.assertRaises(FrozenInstanceError, msg=f"{name} is mutable"):
                tier.name = "mutated"

    def test_tier_equality_is_structural(self):
        """Two TierConfig instances with the same values are equal."""
        a = TierConfig("test", 100, 10, 1, 0, 5)
        b = TierConfig("test", 100, 10, 1, 0, 5)
        self.assertEqual(a, b)

    def test_tier_hash_is_consistent_with_equality(self):
        """Equal TierConfig instances hash the same (frozen dataclass contract)."""
        a = TierConfig("test", 100, 10, 1, 0, 5)
        b = TierConfig("test", 100, 10, 1, 0, 5)
        self.assertEqual(hash(a), hash(b))


class TestTierValidation(unittest.TestCase):
    """TierConfig constructor rejects invalid configurations."""

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 0, 60, 1, 0, 10)

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", -1, 60, 1, 0, 10)

    def test_infinite_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("inf"), 60, 1, 0, 10)

    def test_nan_budget_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", float("nan"), 60, 1, 0, 10)

    def test_zero_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 0, 1, 0, 10)

    def test_negative_timeout_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, -5, 1, 0, 10)

    def test_zero_tasks_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 0, 0, 10)

    def test_min_greater_than_max_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 20, 10)

    def test_negative_min_parts_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, -1, 10)

    def test_zero_max_parts_rejected(self):
        with self.assertRaises(ValueError):
            TierConfig("bad", 3600, 60, 1, 0, 0)

    def test_minimal_valid_tier(self):
        tier = TierConfig("minimal", 0.001, 0.001, 1, 0, 1)
        self.assertEqual(tier.name, "minimal")


class TestExhaustionMonotonicity(unittest.TestCase):
    """Once is_exhausted becomes True it never returns to False."""

    def test_exhaustion_is_permanent(self):
        tiny = TierConfig("tiny", 0.01, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        tracker.pause("post-exhaustion")
        self.assertTrue(tracker.is_exhausted)
        tracker.resume()
        self.assertTrue(tracker.is_exhausted)

    def test_remaining_active_floors_at_zero(self):
        tiny = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
        self.assertEqual(tracker.remaining_active, 0.0)

    def test_elapsed_active_exceeds_budget_after_exhaustion(self):
        tiny = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.015)
        self.assertGreater(tracker.elapsed_active, tiny.active_budget_seconds)


class TestSnapshotTimingInvariants(unittest.TestCase):
    """Snapshot values always satisfy timing invariants."""

    def _check_invariants(self, snap):
        self.assertGreaterEqual(snap["active_seconds"], 0)
        self.assertGreaterEqual(snap["wall_seconds"], 0)
        self.assertGreaterEqual(snap["total_pause_seconds"], 0)
        self.assertGreaterEqual(snap["pause_count"], 0)
        self.assertLessEqual(snap["active_seconds"], snap["wall_seconds"] + 0.01)
        expected_remaining = max(0, snap["budget_limit_seconds"] - snap["active_seconds"])
        self.assertAlmostEqual(snap["remaining_active_seconds"], expected_remaining, delta=0.05)

    def test_fresh_tracker_invariants(self):
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            self._check_invariants(tracker.snapshot())

    def test_after_pause_invariants(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        tracker.pause("test")
        self._check_invariants(tracker.snapshot())

    def test_after_resume_invariants(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("test")
        time.sleep(0.01)
        tracker.resume()
        self._check_invariants(tracker.snapshot())

    def test_after_multiple_cycles_invariants(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        for _ in range(10):
            time.sleep(0.002)
            tracker.pause("cycle")
            time.sleep(0.002)
            tracker.resume()
        self._check_invariants(tracker.snapshot())

    def test_exhausted_tracker_invariants(self):
        tiny = TierConfig("tiny", 0.005, 60, 1, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
        snap = tracker.snapshot()
        self._check_invariants(snap)
        self.assertEqual(snap["remaining_active_seconds"], 0)


class TestCheckpointSchemaStability(unittest.TestCase):
    """Checkpoint schema is stable across save/restore cycles."""

    REQUIRED_CHECKPOINT_KEYS = frozenset({
        "schema_version", "tier", "active_seconds",
        "budget_limit_seconds", "pause_count", "total_pause_seconds",
        "started_at", "checkpoint_at",
    })

    def test_checkpoint_file_has_required_keys(self):
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{name}.json"
                tracker.save_checkpoint(path)
                data = json.loads(path.read_text(encoding="utf-8"))
            missing = self.REQUIRED_CHECKPOINT_KEYS - set(data.keys())
            self.assertEqual(missing, set(), f"Tier {name} checkpoint missing: {missing}")

    def test_checkpoint_schema_version_is_v1(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_save_restore_preserves_active_seconds(self):
        for name in TIERS:
            tier = get_tier(name)
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.02)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "cp.json"
                tracker.save_checkpoint(path)
                saved_active = tracker.elapsed_active
                restored = ActiveBudgetTracker.restore(path, tier=tier)
                self.assertAlmostEqual(restored.elapsed_active, saved_active, delta=0.01,
                                       msg=f"Tier {name}: active seconds not preserved")

    def test_restore_starts_paused(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            restored = ActiveBudgetTracker.restore(path, tier=tier)
        self.assertTrue(restored.is_paused)

    def test_restore_increments_pause_count(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            snap_before = json.loads(path.read_text(encoding="utf-8"))
            restored = ActiveBudgetTracker.restore(path, tier=tier)
            snap_after = restored.snapshot()
        self.assertEqual(snap_after["pause_count"], snap_before["pause_count"] + 1)


class TestSummarySchemaStability(unittest.TestCase):
    """Incomplete summary schema is stable and forward-compatible."""

    REQUIRED_SUMMARY_KEYS = frozenset({
        "schema_version", "status", "tier", "timing",
        "completed_tests", "pending_tests", "severity",
        "issues", "evidence_paths", "blockers", "generated_at",
    })

    def _make_summary(self, tier_name="light"):
        tier = get_tier(tier_name)
        tracker = ActiveBudgetTracker(tier=tier)
        return build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=["t2"],
            issues=["test issue"],
            evidence_paths=["/path/evidence.json"],
            blockers=[],
        )

    def test_summary_has_required_keys(self):
        for name in TIERS:
            summary = self._make_summary(name)
            missing = self.REQUIRED_SUMMARY_KEYS - set(summary.keys())
            self.assertEqual(missing, set(), f"Tier {name} summary missing: {missing}")

    def test_summary_schema_version(self):
        summary = self._make_summary()
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")

    def test_summary_status_is_always_exhausted(self):
        for name in TIERS:
            summary = self._make_summary(name)
            self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")

    def test_summary_tier_has_all_fields(self):
        for name in TIERS:
            summary = self._make_summary(name)
            tier_dict = summary["tier"]
            self.assertIn("name", tier_dict)
            self.assertIn("active_budget_seconds", tier_dict)
            self.assertIn("simulation_timeout_seconds", tier_dict)
            self.assertIn("max_tasks_per_design", tier_dict)
            self.assertIn("max_parts", tier_dict)

    def test_summary_timing_keys(self):
        summary = self._make_summary()
        expected = {"active_seconds", "wall_seconds", "budget_limit_seconds",
                    "pause_count", "total_pause_seconds"}
        self.assertTrue(expected.issubset(set(summary["timing"].keys())))


class TestFrozenCriteriaGuard(unittest.TestCase):
    """CriteriaAdjustment always rejects frozen criteria."""

    def test_frozen_true_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            CriteriaAdjustment("bench_1", {"a": 1}, {"a": 2}, "relax", frozen=True)
        self.assertIn("frozen", str(ctx.exception).lower())

    def test_frozen_false_accepted(self):
        adj = CriteriaAdjustment("bench_1", {"a": 1}, {"a": 2}, "relax", frozen=False)
        self.assertEqual(adj.criterion_id, "bench_1")

    def test_frozen_default_is_false(self):
        adj = CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "reason")
        self.assertFalse(adj.frozen)

    def test_empty_reason_rejected(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "")

    def test_no_change_rejected(self):
        with self.assertRaises(ValueError):
            CriteriaAdjustment("c1", {"a": 1}, {"a": 1}, "same values")

    def test_adjustment_has_timestamp(self):
        adj = CriteriaAdjustment("c1", {"x": 0}, {"x": 1}, "changed")
        self.assertTrue(len(adj.adjusted_at) > 0)
        self.assertIn("T", adj.adjusted_at)

    def test_adjustment_to_dict_complete(self):
        adj = CriteriaAdjustment("c1", {"x": 0}, {"x": 1}, "reason")
        d = adj.to_dict()
        self.assertEqual(set(d.keys()), {"criterion_id", "before", "after", "reason", "adjusted_at"})


class TestRecoveryContract(unittest.TestCase):
    """Recovery delay is deterministic and bounded."""

    def test_delay_sequence_exact(self):
        expected = [5.0, 10.0, 20.0, 40.0, 60.0, 60.0]
        for i, exp in enumerate(expected):
            self.assertEqual(recovery_delay(i), exp)

    def test_delay_is_deterministic(self):
        for attempt in range(10):
            a = recovery_delay(attempt)
            b = recovery_delay(attempt)
            self.assertEqual(a, b)

    def test_delay_never_exceeds_60(self):
        for attempt in range(100):
            self.assertLessEqual(recovery_delay(attempt), 60.0)

    def test_delay_never_negative(self):
        for attempt in range(100):
            self.assertGreater(recovery_delay(attempt), 0)

    def test_should_retry_respects_max_retries(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "some error"))
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES + 1, "some error"))

    def test_should_retry_at_max_minus_one(self):
        self.assertTrue(should_retry(MAX_RECOVERY_RETRIES - 1, "some_error"))

    def test_non_retryable_keywords_exact(self):
        for keyword in ["ACTIVE_BUDGET_EXHAUSTED", "TERMINATED", "PASS"]:
            self.assertFalse(should_retry(0, keyword), f"{keyword} should not retry")

    def test_case_sensitivity_of_non_retryable(self):
        self.assertTrue(should_retry(0, "pass"))
        self.assertTrue(should_retry(0, "terminated"))
        self.assertTrue(should_retry(0, "active_budget_exhausted"))


class TestLimitEnforcementContract(unittest.TestCase):
    """Limit checks never silently drop or inflate counts."""

    def test_empty_list_always_passes(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([], tier), [])
            self.assertEqual(check_task_count([], tier), [])

    def test_single_item_always_passes(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_part_count([{}], tier), [])
            self.assertEqual(check_task_count([{}], tier), [])

    def test_at_limit_passes(self):
        for name, tier in TIERS.items():
            parts = [{} for _ in range(tier.max_parts)]
            tasks = [{} for _ in range(tier.max_tasks_per_design)]
            self.assertEqual(check_part_count(parts, tier), [], f"{name} parts at limit")
            self.assertEqual(check_task_count(tasks, tier), [], f"{name} tasks at limit")

    def test_one_over_limit_fails(self):
        for name, tier in TIERS.items():
            parts = [{} for _ in range(tier.max_parts + 1)]
            tasks = [{} for _ in range(tier.max_tasks_per_design + 1)]
            self.assertGreater(len(check_part_count(parts, tier)), 0, f"{name} parts over")
            self.assertGreater(len(check_task_count(tasks, tier)), 0, f"{name} tasks over")

    def test_violation_messages_include_counts(self):
        tier = get_tier("light")
        violations = check_part_count([{} for _ in range(11)], tier)
        self.assertIn("11", violations[0])
        self.assertIn("10", violations[0])

    def test_enforce_timeout_returns_tier_limit(self):
        for name, tier in TIERS.items():
            self.assertEqual(enforce_simulation_timeout(tier),
                             tier.simulation_timeout_seconds)

    def test_enforce_timeout_clamps_oversized(self):
        for name, tier in TIERS.items():
            self.assertEqual(enforce_simulation_timeout(tier, 1e6),
                             tier.simulation_timeout_seconds)

    def test_enforce_timeout_allows_smaller(self):
        tier = get_tier("high")
        self.assertEqual(enforce_simulation_timeout(tier, 10), 10)

    def test_enforce_timeout_rejects_zero(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("light"), 0)

    def test_enforce_timeout_rejects_negative(self):
        with self.assertRaises(ValueError):
            enforce_simulation_timeout(get_tier("light"), -5)


class TestBudgetStatusContract(unittest.TestCase):
    """BudgetStatus enum values and is_terminal() are stable."""

    EXPECTED_STATUSES = {
        "PASS", "NEEDS_REVISION", "HARNESS_ERROR", "SIMULATION_TIMEOUT",
        "INFRASTRUCTURE_FAILURE", "ACTIVE_BUDGET_EXHAUSTED",
        "INCOMPLETE_EVIDENCE", "EVIDENCE_REQUIRED",
    }

    TERMINAL_STATUSES = {"PASS", "ACTIVE_BUDGET_EXHAUSTED", "INFRASTRUCTURE_FAILURE"}

    def test_all_expected_statuses_exist(self):
        actual = {s.value for s in BudgetStatus}
        self.assertEqual(actual, self.EXPECTED_STATUSES)

    def test_terminal_statuses_correct(self):
        terminal = {s.value for s in BudgetStatus if s.is_terminal()}
        self.assertEqual(terminal, self.TERMINAL_STATUSES)

    def test_non_terminal_statuses(self):
        non_terminal = {s.value for s in BudgetStatus if not s.is_terminal()}
        expected_non_terminal = self.EXPECTED_STATUSES - self.TERMINAL_STATUSES
        self.assertEqual(non_terminal, expected_non_terminal)

    def test_status_is_string_enum(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.value, str)
            self.assertEqual(str(status), f"BudgetStatus.{status.name}")

    def test_simulation_timeout_result_status_derivation(self):
        timed_out = SimulationTimeoutResult(timed_out=True, elapsed_seconds=100)
        success = SimulationTimeoutResult(timed_out=False, elapsed_seconds=50)
        self.assertEqual(timed_out.status, BudgetStatus.SIMULATION_TIMEOUT)
        self.assertEqual(success.status, BudgetStatus.PASS)


class TestGetTierContract(unittest.TestCase):
    """get_tier() is the canonical lookup and rejects unknown names."""

    def test_all_known_tiers_resolve(self):
        for name in TIERS:
            tier = get_tier(name)
            self.assertEqual(tier.name, name)

    def test_unknown_tier_raises(self):
        with self.assertRaises(ValueError):
            get_tier("nonexistent")

    def test_error_message_lists_available(self):
        with self.assertRaises(ValueError) as ctx:
            get_tier("unknown_tier")
        msg = str(ctx.exception)
        for name in TIERS:
            self.assertIn(name, msg)


class TestTierOrdering(unittest.TestCase):
    """Tier limits are monotonically non-decreasing: light ≤ medium ≤ high."""

    def test_budget_monotonic(self):
        self.assertLessEqual(
            get_tier("light").active_budget_seconds,
            get_tier("medium").active_budget_seconds,
        )
        self.assertLessEqual(
            get_tier("medium").active_budget_seconds,
            get_tier("high").active_budget_seconds,
        )

    def test_timeout_monotonic(self):
        self.assertLessEqual(
            get_tier("light").simulation_timeout_seconds,
            get_tier("medium").simulation_timeout_seconds,
        )
        self.assertLessEqual(
            get_tier("medium").simulation_timeout_seconds,
            get_tier("high").simulation_timeout_seconds,
        )

    def test_max_parts_monotonic(self):
        self.assertLessEqual(
            get_tier("light").max_parts,
            get_tier("medium").max_parts,
        )
        self.assertLessEqual(
            get_tier("medium").max_parts,
            get_tier("high").max_parts,
        )

    def test_max_tasks_monotonic(self):
        self.assertLessEqual(
            get_tier("light").max_tasks_per_design,
            get_tier("medium").max_tasks_per_design,
        )
        self.assertLessEqual(
            get_tier("medium").max_tasks_per_design,
            get_tier("high").max_tasks_per_design,
        )


class TestAdjustmentLogContract(unittest.TestCase):
    """AdjustmentLog maintains ordering and schema through save/load."""

    def test_entries_preserve_insertion_order(self):
        log = AdjustmentLog()
        for i in range(5):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i + 1}, f"step {i}"))
        for i, entry in enumerate(log.entries):
            self.assertEqual(entry.criterion_id, f"c{i}")

    def test_save_load_preserves_order(self):
        log = AdjustmentLog()
        for i in range(3):
            log.record(CriteriaAdjustment(f"c{i}", {"v": i}, {"v": i + 1}, f"step {i}"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        for i, entry in enumerate(loaded.entries):
            self.assertEqual(entry.criterion_id, f"c{i}")

    def test_entries_is_a_copy(self):
        """Modifying the returned list does not affect the log."""
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test"))
        entries = log.entries
        entries.clear()
        self.assertEqual(len(log.entries), 1)

    def test_to_list_produces_json_safe_dicts(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "reason"))
        lst = log.to_list()
        encoded = json.dumps(lst)
        decoded = json.loads(encoded)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["criterion_id"], "c1")


class TestConcurrentTrackerIsolation(unittest.TestCase):
    """Multiple trackers created concurrently are fully independent."""

    def test_concurrent_creation_no_interference(self):
        results = {}
        errors = []

        def worker(tier_name, sleep_s):
            try:
                tracker = ActiveBudgetTracker(tier=get_tier(tier_name))
                time.sleep(sleep_s)
                results[tier_name] = tracker.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("light", 0.02)),
            threading.Thread(target=worker, args=("medium", 0.03)),
            threading.Thread(target=worker, args=("high", 0.01)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(results["light"]["tier"], "light")
        self.assertEqual(results["medium"]["tier"], "medium")
        self.assertEqual(results["high"]["tier"], "high")
        self.assertEqual(
            results["light"]["budget_limit_seconds"],
            get_tier("light").active_budget_seconds,
        )


if __name__ == "__main__":
    unittest.main()
