"""Budget composite invariant tests.

Validates cross-component contracts: the coherence between tracker,
summary, checkpoint, simulation timeout, tier limits, recovery helpers,
and adjustment log when they interact in realistic sequences.
"""
from __future__ import annotations

import json
import math
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path

from ..slim_harness.budget import (
    BACKOFF_BASE_SECONDS,
    MAX_RECOVERY_RETRIES,
    TIERS,
    TierConfig,
    ActiveBudgetTracker,
    AdjustmentLog,
    BudgetStatus,
    CriteriaAdjustment,
    SimulationTimeoutResult,
    build_incomplete_summary,
    check_part_count,
    check_task_count,
    enforce_simulation_timeout,
    get_tier,
    recovery_delay,
    should_retry,
)


# ── Summary ↔ Tracker coherence ──────────────────────────────────────

class TestSummaryTrackerCoherence(unittest.TestCase):
    """Summary timing must reflect the tracker's actual state."""

    def test_summary_timing_matches_snapshot(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.005)
            snap = tracker.snapshot()
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            t = summary["timing"]
            self.assertAlmostEqual(t["active_seconds"], snap["active_seconds"], delta=0.05)
            self.assertAlmostEqual(t["wall_seconds"], snap["wall_seconds"], delta=0.05)
            self.assertEqual(t["budget_limit_seconds"], snap["budget_limit_seconds"])
            self.assertEqual(t["pause_count"], snap["pause_count"])
            self.assertEqual(t["total_pause_seconds"], snap["total_pause_seconds"])

    def test_paused_tracker_summary_has_pause(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        tracker.pause("test")
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["timing"]["pause_count"], 1)

    def test_multiple_pause_resume_reflected(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        for _ in range(5):
            tracker.pause("p")
            tracker.resume()
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["timing"]["pause_count"], 5)

    def test_summary_tier_matches_tracker_tier(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=[],
                issues=[], evidence_paths=[], blockers=[],
            )
            self.assertEqual(summary["tier"]["name"], name)
            self.assertEqual(summary["tier"]["active_budget_seconds"], tier.active_budget_seconds)


# ── SimulationTimeout ↔ Tier limits ──────────────────────────────────

class TestTimeoutTierCoherence(unittest.TestCase):
    """enforce_simulation_timeout always respects tier upper bound."""

    def test_timeout_never_exceeds_tier(self):
        for name, tier in TIERS.items():
            effective = enforce_simulation_timeout(tier)
            self.assertEqual(effective, tier.simulation_timeout_seconds)

    def test_smaller_actual_wins(self):
        for name, tier in TIERS.items():
            small = tier.simulation_timeout_seconds / 2
            effective = enforce_simulation_timeout(tier, small)
            self.assertEqual(effective, small)

    def test_larger_actual_clamped(self):
        for name, tier in TIERS.items():
            big = tier.simulation_timeout_seconds * 10
            effective = enforce_simulation_timeout(tier, big)
            self.assertEqual(effective, tier.simulation_timeout_seconds)

    def test_result_status_coherent(self):
        r_timeout = SimulationTimeoutResult(timed_out=True, elapsed_seconds=10.0)
        self.assertEqual(r_timeout.status, BudgetStatus.SIMULATION_TIMEOUT)
        r_pass = SimulationTimeoutResult(timed_out=False, elapsed_seconds=5.0)
        self.assertEqual(r_pass.status, BudgetStatus.PASS)

    def test_result_with_partial_evidence(self):
        r = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=30.0,
            partial_evidence={"force_max": 42.0}, log_path=Path("/tmp/log"),
        )
        self.assertTrue(r.timed_out)
        self.assertEqual(r.partial_evidence["force_max"], 42.0)
        self.assertEqual(r.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_result_with_error(self):
        r = SimulationTimeoutResult(
            timed_out=False, elapsed_seconds=1.0, error="solver diverged",
        )
        self.assertEqual(r.status, BudgetStatus.PASS)
        self.assertEqual(r.error, "solver diverged")


# ── Task/Part count ↔ Tier limits ────────────────────────────────────

class TestLimitEnforcementAcrossTiers(unittest.TestCase):
    """check_task_count and check_part_count enforce per-tier limits."""

    def test_at_limit_ok(self):
        for name, tier in TIERS.items():
            tasks = [f"t{i}" for i in range(tier.max_tasks_per_design)]
            parts = [f"p{i}" for i in range(tier.max_parts)]
            self.assertEqual(check_task_count(tasks, tier), [])
            self.assertEqual(check_part_count(parts, tier), [])

    def test_one_over_limit_fails(self):
        for name, tier in TIERS.items():
            tasks = [f"t{i}" for i in range(tier.max_tasks_per_design + 1)]
            parts = [f"p{i}" for i in range(tier.max_parts + 1)]
            self.assertEqual(len(check_task_count(tasks, tier)), 1)
            self.assertEqual(len(check_part_count(parts, tier)), 1)

    def test_empty_always_ok(self):
        for name, tier in TIERS.items():
            self.assertEqual(check_task_count([], tier), [])
            self.assertEqual(check_part_count([], tier), [])

    def test_cross_tier_escalation(self):
        light = get_tier("light")
        high = get_tier("high")
        tasks = [f"t{i}" for i in range(light.max_tasks_per_design + 1)]
        self.assertGreater(len(check_task_count(tasks, light)), 0)
        self.assertEqual(check_task_count(tasks, high), [])

    def test_violation_message_contains_tier_name(self):
        tier = get_tier("medium")
        tasks = [f"t{i}" for i in range(100)]
        violations = check_task_count(tasks, tier)
        if violations:
            self.assertIn("medium", violations[0])


# ── Recovery ↔ Retry coordination ────────────────────────────────────

class TestRecoveryRetryCoordination(unittest.TestCase):
    """recovery_delay only meaningful when should_retry is True."""

    def test_delay_increases_over_allowed_attempts(self):
        delays = []
        for attempt in range(MAX_RECOVERY_RETRIES):
            self.assertTrue(should_retry(attempt, "transient error"))
            delays.append(recovery_delay(attempt))
        for i in range(1, len(delays)):
            self.assertGreater(delays[i], delays[i-1])

    def test_delay_at_max_attempt_is_irrelevant(self):
        self.assertFalse(should_retry(MAX_RECOVERY_RETRIES, "transient error"))
        d = recovery_delay(MAX_RECOVERY_RETRIES)
        self.assertIsInstance(d, float)

    def test_non_retryable_stops_before_delay_matters(self):
        for keyword in ("ACTIVE_BUDGET_EXHAUSTED", "TERMINATED", "PASS"):
            self.assertFalse(should_retry(0, keyword))
            self.assertFalse(should_retry(1, f"error: {keyword} occurred"))

    def test_delay_cap_at_60(self):
        for attempt in range(20):
            self.assertLessEqual(recovery_delay(attempt), 60.0)

    def test_delay_formula_exact(self):
        for attempt in range(6):
            expected = min(BACKOFF_BASE_SECONDS * (2 ** attempt), 60.0)
            self.assertEqual(recovery_delay(attempt), expected)

    def test_retry_and_delay_full_sequence(self):
        attempt = 0
        error = "temporary network error"
        while should_retry(attempt, error):
            d = recovery_delay(attempt)
            self.assertGreater(d, 0)
            attempt += 1
        self.assertEqual(attempt, MAX_RECOVERY_RETRIES)


# ── Checkpoint ↔ Restore ↔ Re-snapshot chain ─────────────────────────

class TestCheckpointRestoreChain(unittest.TestCase):
    """Checkpoint → Restore → Re-snapshot preserves essential state."""

    def test_restore_preserves_tier(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            time.sleep(0.01)
            with tempfile.TemporaryDirectory() as tmp:
                cp = Path(tmp) / "cp.json"
                tracker.save_checkpoint(cp)
                restored = ActiveBudgetTracker.restore(cp)
            self.assertEqual(restored.snapshot()["tier"], name)

    def test_restore_starts_paused(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        self.assertTrue(restored.is_paused)

    def test_double_restore_idempotent(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            r1 = ActiveBudgetTracker.restore(cp)
            r2 = ActiveBudgetTracker.restore(cp)
        s1 = r1.snapshot()
        s2 = r2.snapshot()
        self.assertEqual(s1["tier"], s2["tier"])
        self.assertAlmostEqual(s1["active_seconds"], s2["active_seconds"], delta=0.01)

    def test_restore_resume_accumulates(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.01)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            snap_before = tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        restored.resume()
        time.sleep(0.01)
        snap_after = restored.snapshot()
        self.assertGreater(snap_after["active_seconds"], snap_before["active_seconds"])

    def test_checkpoint_chain_three_deep(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.005)
        with tempfile.TemporaryDirectory() as tmp:
            cp1 = Path(tmp) / "cp1.json"
            tracker.save_checkpoint(cp1)
            r1 = ActiveBudgetTracker.restore(cp1)
            r1.resume()
            time.sleep(0.005)
            cp2 = Path(tmp) / "cp2.json"
            r1.save_checkpoint(cp2)
            r2 = ActiveBudgetTracker.restore(cp2)
            r2.resume()
            time.sleep(0.005)
            cp3 = Path(tmp) / "cp3.json"
            r2.save_checkpoint(cp3)
            r3 = ActiveBudgetTracker.restore(cp3)
        s = r3.snapshot()
        self.assertEqual(s["tier"], "medium")
        self.assertGreater(s["active_seconds"], 0)


# ── Full lifecycle: create → operate → exhaust → summarize ───────────

class TestFullLifecycle(unittest.TestCase):
    """End-to-end lifecycle exercising multiple components."""

    def test_create_pause_checkpoint_restore_resume_summary(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.005)
        tracker.pause("checkpoint")
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            restored = ActiveBudgetTracker.restore(cp)
        restored.resume()
        time.sleep(0.005)
        summary = build_incomplete_summary(
            tier=tier, tracker=restored,
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                             "geometry_audit": {"passed": True}}],
            pending_tests=["t2", "t3"],
            issues=["budget running low"],
            evidence_paths=["/evidence/t1"],
            blockers=[],
        )
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(summary["severity"], "functional")
        self.assertEqual(len(summary["completed_tests"]), 1)
        self.assertEqual(len(summary["pending_tests"]), 2)

    def test_lifecycle_with_adjustment_log(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"tol": 0.1}, {"tol": 0.05}, "tighter"))
        time.sleep(0.005)
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            lp = Path(tmp) / "log.json"
            tracker.save_checkpoint(cp)
            log.save(lp)
            restored_tracker = ActiveBudgetTracker.restore(cp)
            restored_log = AdjustmentLog.load(lp)
        restored_tracker.resume()
        restored_log.record(CriteriaAdjustment("c2", {"gap": 1}, {"gap": 2}, "wider"))
        self.assertEqual(len(restored_log.to_list()), 2)
        summary = build_incomplete_summary(
            tier=tier, tracker=restored_tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["tier"]["name"], "medium")

    def test_tiny_budget_exhaustion_lifecycle(self):
        tier = TierConfig("tiny", 0.005, 60, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.02)
        self.assertTrue(tracker.is_exhausted)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=["t1"],
            issues=["budget exhausted"], evidence_paths=[], blockers=["ACTIVE_BUDGET_EXHAUSTED"],
        )
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(summary["severity"], "functional")
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_all_checks_pass_for_valid_design(self):
        tier = get_tier("high")
        tasks = [f"task_{i}" for i in range(tier.max_tasks_per_design)]
        parts = [f"part_{i}" for i in range(tier.max_parts)]
        self.assertEqual(check_task_count(tasks, tier), [])
        self.assertEqual(check_part_count(parts, tier), [])
        timeout = enforce_simulation_timeout(tier, 100)
        self.assertEqual(timeout, 100)
        tracker = ActiveBudgetTracker(tier=tier)
        self.assertFalse(tracker.is_exhausted)
        self.assertGreater(tracker.remaining_active, 0)


# ── BudgetStatus ↔ Component interaction ─────────────────────────────

class TestBudgetStatusInteraction(unittest.TestCase):
    """BudgetStatus terminal states interact correctly with retry logic."""

    def test_terminal_pass_and_exhausted_block_retry(self):
        for status in (BudgetStatus.PASS, BudgetStatus.ACTIVE_BUDGET_EXHAUSTED):
            self.assertTrue(status.is_terminal())
            self.assertFalse(
                should_retry(0, status.value),
                f"{status.value} is terminal but should_retry returned True",
            )

    def test_non_terminal_allow_retry(self):
        non_terminal = [s for s in BudgetStatus if not s.is_terminal()]
        for status in non_terminal:
            if status.value not in ("TERMINATED",):
                retryable = should_retry(0, status.value)
                if "PASS" not in status.value and "EXHAUSTED" not in status.value:
                    self.assertTrue(retryable, f"{status.value}")

    def test_simulation_timeout_result_uses_correct_statuses(self):
        self.assertEqual(
            SimulationTimeoutResult(timed_out=True, elapsed_seconds=1).status,
            BudgetStatus.SIMULATION_TIMEOUT,
        )
        self.assertEqual(
            SimulationTimeoutResult(timed_out=False, elapsed_seconds=1).status,
            BudgetStatus.PASS,
        )

    def test_pass_is_terminal_and_non_retryable(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())
        self.assertFalse(should_retry(0, "PASS"))

    def test_exhausted_is_terminal_and_non_retryable(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())
        self.assertFalse(should_retry(0, "ACTIVE_BUDGET_EXHAUSTED"))

    def test_infrastructure_failure_is_terminal_but_retryable_by_keyword(self):
        self.assertTrue(BudgetStatus.INFRASTRUCTURE_FAILURE.is_terminal())
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE"))


# ── Snapshot schema ↔ Restore acceptance ──────────────────────────────

class TestSnapshotRestoreSchemaContract(unittest.TestCase):
    """Snapshot schema_version must match what restore expects."""

    def test_snapshot_version_is_checkpoint_v1(self):
        for name, tier in TIERS.items():
            tracker = ActiveBudgetTracker(tier=tier)
            self.assertEqual(tracker.snapshot()["schema_version"], "budget_checkpoint_v1")

    def test_checkpoint_file_has_same_version(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "budget_checkpoint_v1")

    def test_corrupted_version_in_checkpoint(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            data["schema_version"] = "budget_checkpoint_v999"
            cp.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                ActiveBudgetTracker.restore(cp)

    def test_missing_tier_in_checkpoint_raises(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        with tempfile.TemporaryDirectory() as tmp:
            cp = Path(tmp) / "cp.json"
            tracker.save_checkpoint(cp)
            data = json.loads(cp.read_text(encoding="utf-8"))
            del data["tier"]
            cp.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises((ValueError, KeyError)):
                ActiveBudgetTracker.restore(cp)


# ── TierConfig ↔ enforce_simulation_timeout ───────────────────────────

class TestTierTimeoutEnforcement(unittest.TestCase):
    """Every production tier's timeout is enforced correctly."""

    def test_default_timeout_equals_tier_limit(self):
        for name, tier in TIERS.items():
            self.assertEqual(enforce_simulation_timeout(tier), tier.simulation_timeout_seconds)

    def test_exact_tier_limit_accepted(self):
        for name, tier in TIERS.items():
            result = enforce_simulation_timeout(tier, tier.simulation_timeout_seconds)
            self.assertEqual(result, tier.simulation_timeout_seconds)

    def test_one_below_tier_limit(self):
        for name, tier in TIERS.items():
            below = tier.simulation_timeout_seconds - 1
            if below > 0:
                result = enforce_simulation_timeout(tier, below)
                self.assertEqual(result, below)

    def test_zero_rejected(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError):
                enforce_simulation_timeout(tier, 0)

    def test_negative_rejected(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError):
                enforce_simulation_timeout(tier, -1)

    def test_inf_rejected(self):
        for name, tier in TIERS.items():
            with self.assertRaises(ValueError):
                enforce_simulation_timeout(tier, float("inf"))


if __name__ == "__main__":
    unittest.main()
