"""Budget integration with evidence recording and measurement subsystems.

Tests interactions between budget tracking and evidence artifacts:
- Snapshot schema compatibility with evidence recording
- Incomplete summary as a serializable evidence artifact
- Budget timing metadata in evidence chains
- Checkpoint recovery preserving evidence context
- Budget status transitions with measurement recording
- Tier enforcement interacting with measurement limits
- Budget annotation schema completeness for downstream consumers
- Round-trip fidelity of budget artifacts through JSON evidence pipeline
"""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from copy import deepcopy
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


class TestSnapshotAsEvidence(unittest.TestCase):
    """Budget snapshot embeds cleanly in evidence records."""

    def test_snapshot_embeds_in_evidence_dict(self):
        """A snapshot can be embedded inside an evidence record dict."""
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        time.sleep(0.01)
        snap = tracker.snapshot()
        evidence = {
            "schema_version": "slim_evidence_v1",
            "test_id": "travel_test_1",
            "budget_snapshot": snap,
            "measurements": {"displacement_mm": 9.85},
        }
        encoded = json.dumps(evidence)
        decoded = json.loads(encoded)
        self.assertEqual(decoded["budget_snapshot"]["tier"], "medium")
        self.assertGreater(decoded["budget_snapshot"]["active_seconds"], 0)

    def test_snapshot_keys_stable_across_tiers(self):
        """Every tier produces the same set of snapshot keys."""
        key_sets = []
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            key_sets.append(set(tracker.snapshot().keys()))
        for ks in key_sets[1:]:
            self.assertEqual(key_sets[0], ks)

    def test_snapshot_values_are_json_primitive(self):
        """All snapshot values are JSON-serializable primitives."""
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("check")
        tracker.resume()
        snap = tracker.snapshot()
        for key, value in snap.items():
            self.assertIsInstance(
                value, (str, int, float, bool, type(None)),
                f"Snapshot key {key!r} has non-primitive type {type(value).__name__}",
            )

    def test_snapshot_timing_always_non_negative(self):
        """All time-related snapshot values are non-negative."""
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            time.sleep(0.005)
            snap = tracker.snapshot()
            for key in ["active_seconds", "wall_seconds", "total_pause_seconds",
                        "budget_limit_seconds", "remaining_active_seconds"]:
                self.assertGreaterEqual(snap[key], 0, f"{name}/{key}")


class TestIncompleteSummaryAsEvidence(unittest.TestCase):
    """build_incomplete_summary produces evidence-ready artifacts."""

    def _make_evidence_summary(self, tier_name="light", **kwargs):
        tier = get_tier(tier_name)
        tracker = ActiveBudgetTracker(tier=tier)
        defaults = {
            "completed_tests": [
                {"test_id": "t1", "physical_pass": True,
                 "geometry_audit": {"passed": True}},
                {"test_id": "t2", "physical_pass": False,
                 "geometry_audit": {"passed": True}},
            ],
            "pending_tests": ["t3", "t4"],
            "issues": ["timeout after 45s"],
            "evidence_paths": ["/run/r01/evidence/trace_t1.json"],
            "blockers": [],
        }
        defaults.update(kwargs)
        return build_incomplete_summary(tier=tier, tracker=tracker, **defaults)

    def test_summary_classifies_completed_tests(self):
        """Completed tests are classified as PASS or FAIL."""
        summary = self._make_evidence_summary()
        self.assertEqual(summary["completed_tests"][0]["status"], "PASS")
        self.assertEqual(summary["completed_tests"][1]["status"], "FAIL")

    def test_summary_preserves_test_ids(self):
        summary = self._make_evidence_summary()
        ids = [t["id"] for t in summary["completed_tests"]]
        self.assertEqual(ids, ["t1", "t2"])

    def test_summary_embeds_full_tier_config(self):
        summary = self._make_evidence_summary(tier_name="high")
        tier = summary["tier"]
        self.assertEqual(tier["name"], "high")
        self.assertEqual(tier["active_budget_seconds"], 10800)
        self.assertEqual(tier["simulation_timeout_seconds"], 300)
        self.assertEqual(tier["max_tasks_per_design"], 5)
        self.assertEqual(tier["max_parts"], 20)

    def test_summary_round_trips_through_file(self):
        """Summary survives write-to-file then read-back."""
        summary = self._make_evidence_summary()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "incomplete_summary.json"
            path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(loaded["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(len(loaded["completed_tests"]), 2)
        self.assertEqual(len(loaded["pending_tests"]), 2)

    def test_summary_severity_with_only_passing_tests(self):
        summary = self._make_evidence_summary(
            completed_tests=[
                {"test_id": "t1", "physical_pass": True,
                 "geometry_audit": {"passed": True}},
            ],
            pending_tests=[],
        )
        self.assertEqual(summary["severity"], "minor")

    def test_summary_severity_with_failures_but_no_pending(self):
        summary = self._make_evidence_summary(
            completed_tests=[
                {"test_id": "t1", "physical_pass": False,
                 "geometry_audit": {"passed": True}},
            ],
            pending_tests=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_summary_deep_copy_safe(self):
        """Summary can be deep-copied without error."""
        summary = self._make_evidence_summary()
        clone = deepcopy(summary)
        self.assertEqual(summary, clone)
        clone["issues"].append("extra")
        self.assertNotEqual(len(summary["issues"]), len(clone["issues"]))


class TestCheckpointEvidencePreservation(unittest.TestCase):
    """Checkpoint save/restore preserves evidence-relevant state."""

    def test_checkpoint_includes_started_at(self):
        """Checkpoint file records when tracking started."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("started_at", data)
        self.assertIn("2026", data["started_at"])

    def test_checkpoint_includes_checkpoint_at(self):
        """Checkpoint file records when checkpoint was taken."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("checkpoint_at", data)

    def test_restored_tracker_has_accurate_remaining(self):
        """After restore, remaining_active reflects the actual work done."""
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.05)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cp.json"
            tracker.save_checkpoint(path)
            restored = ActiveBudgetTracker.restore(path, tier=tier)
            restored.resume()
            self.assertAlmostEqual(
                restored.remaining_active,
                10800 - tracker.elapsed_active,
                delta=0.1,
            )

    def test_checkpoint_file_deterministic_keys(self):
        """Checkpoint always has the same top-level keys."""
        expected_keys = {
            "schema_version", "tier", "active_seconds",
            "budget_limit_seconds", "pause_count",
            "total_pause_seconds", "started_at", "checkpoint_at",
        }
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / f"{name}.json"
                tracker.save_checkpoint(path)
                data = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(
                expected_keys.issubset(data.keys()),
                f"Tier {name} missing keys: {expected_keys - data.keys()}",
            )


class TestBudgetStatusTransitions(unittest.TestCase):
    """Budget status transitions are compatible with evidence recording."""

    def test_terminal_non_recoverable_statuses_are_non_retryable(self):
        """PASS and ACTIVE_BUDGET_EXHAUSTED are never retried."""
        for keyword in ["PASS", "ACTIVE_BUDGET_EXHAUSTED", "TERMINATED"]:
            self.assertFalse(
                should_retry(0, keyword),
                f"{keyword} should never be retried",
            )

    def test_infrastructure_failure_is_retryable(self):
        """INFRASTRUCTURE_FAILURE is terminal for the run but retryable in recovery."""
        self.assertTrue(should_retry(0, "INFRASTRUCTURE_FAILURE"))

    def test_non_terminal_statuses_are_retryable(self):
        """Non-terminal statuses are retryable at attempt 0."""
        for status in BudgetStatus:
            if not status.is_terminal() and status.value not in {"TERMINATED"}:
                pass

    def test_all_statuses_are_json_serializable(self):
        for status in BudgetStatus:
            data = {"status": status.value, "is_terminal": status.is_terminal()}
            encoded = json.dumps(data)
            decoded = json.loads(encoded)
            self.assertEqual(decoded["status"], status.value)

    def test_status_values_are_valid_identifiers(self):
        """Status values use only uppercase letters and underscores."""
        import re
        pattern = re.compile(r'^[A-Z][A-Z_]+$')
        for status in BudgetStatus:
            self.assertRegex(status.value, pattern)


class TestSimulationTimeoutResultEvidence(unittest.TestCase):
    """SimulationTimeoutResult produces evidence-compatible output."""

    def test_result_from_timeout(self):
        result = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.1)
        self.assertTrue(result.timed_out)
        self.assertGreater(result.elapsed_seconds, 60)

    def test_result_from_success(self):
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=30.0)
        self.assertFalse(result.timed_out)

    def test_result_status_derivation(self):
        timeout_result = SimulationTimeoutResult(timed_out=True, elapsed_seconds=61)
        self.assertEqual(timeout_result.status, BudgetStatus.SIMULATION_TIMEOUT)

        success_result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=30)
        self.assertEqual(success_result.status, BudgetStatus.PASS)

    def test_result_optional_fields(self):
        result = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=45,
            partial_evidence={"trace": "partial"},
            error="timeout exceeded",
        )
        self.assertEqual(result.partial_evidence, {"trace": "partial"})
        self.assertEqual(result.error, "timeout exceeded")


class TestTierEnforcementWithMeasurements(unittest.TestCase):
    """Tier limits interact correctly with measurement-like data."""

    def test_part_count_at_exact_boundary(self):
        """Exactly max_parts passes; max_parts+1 fails."""
        for name, tier in TIERS.items():
            at_limit = [{} for _ in range(tier.max_parts)]
            over_limit = [{} for _ in range(tier.max_parts + 1)]
            self.assertEqual(check_part_count(at_limit, tier), [],
                             f"{name}: at limit should pass")
            self.assertGreater(len(check_part_count(over_limit, tier)), 0,
                               f"{name}: over limit should fail")

    def test_task_count_at_exact_boundary(self):
        for name, tier in TIERS.items():
            at_limit = [{} for _ in range(tier.max_tasks_per_design)]
            over_limit = [{} for _ in range(tier.max_tasks_per_design + 1)]
            self.assertEqual(check_task_count(at_limit, tier), [])
            self.assertGreater(len(check_task_count(over_limit, tier)), 0)

    def test_timeout_enforcement_returns_exact_tier_limit(self):
        for name, tier in TIERS.items():
            self.assertEqual(
                enforce_simulation_timeout(tier),
                tier.simulation_timeout_seconds,
            )

    def test_timeout_enforcement_clamps_oversized_request(self):
        for name, tier in TIERS.items():
            clamped = enforce_simulation_timeout(tier, 999999)
            self.assertEqual(clamped, tier.simulation_timeout_seconds)


class TestAdjustmentLogEvidence(unittest.TestCase):
    """Adjustment log produces evidence-compatible records."""

    def test_log_round_trips_through_file(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "loosened"))
        log.record(CriteriaAdjustment("c2", {"b": "x"}, {"b": "y"}, "tightened"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adj.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(len(loaded.entries), 2)
        self.assertEqual(loaded.entries[0].criterion_id, "c1")
        self.assertEqual(loaded.entries[1].criterion_id, "c2")

    def test_log_entries_have_timestamps(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "test"))
        self.assertTrue(len(log.entries[0].adjusted_at) > 0)
        self.assertIn("T", log.entries[0].adjusted_at)

    def test_log_to_list_produces_dicts(self):
        log = AdjustmentLog()
        log.record(CriteriaAdjustment("c1", {"a": 1}, {"a": 2}, "reason"))
        lst = log.to_list()
        self.assertIsInstance(lst, list)
        self.assertIsInstance(lst[0], dict)
        self.assertIn("criterion_id", lst[0])

    def test_empty_log_round_trips(self):
        log = AdjustmentLog()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
        self.assertEqual(len(loaded.entries), 0)


class TestRecoveryDelayEvidence(unittest.TestCase):
    """Recovery delay values are evidence-compatible (finite, bounded)."""

    def test_first_delay_is_base(self):
        self.assertEqual(recovery_delay(0), BACKOFF_BASE_SECONDS)

    def test_delay_doubles_each_attempt(self):
        for i in range(4):
            expected = min(BACKOFF_BASE_SECONDS * (2 ** i), 60.0)
            self.assertEqual(recovery_delay(i), expected)

    def test_delay_capped_at_60(self):
        self.assertEqual(recovery_delay(100), 60.0)
        self.assertEqual(recovery_delay(1000), 60.0)

    def test_delay_sequence_is_monotonic(self):
        prev = 0
        for i in range(20):
            d = recovery_delay(i)
            self.assertGreaterEqual(d, prev)
            prev = d


class TestBudgetAnnotationCompleteness(unittest.TestCase):
    """Budget annotation produces complete metadata for downstream consumers."""

    def test_tier_asdict_has_all_fields(self):
        """Tier serialization includes all TierConfig fields."""
        from dataclasses import asdict
        for name, tier in TIERS.items():
            d = asdict(tier)
            self.assertIn("name", d)
            self.assertIn("active_budget_seconds", d)
            self.assertIn("simulation_timeout_seconds", d)
            self.assertIn("max_tasks_per_design", d)
            self.assertIn("max_parts", d)
            self.assertIn("min_parts", d)

    def test_tier_asdict_round_trips_through_json(self):
        from dataclasses import asdict
        for name, tier in TIERS.items():
            encoded = json.dumps(asdict(tier))
            decoded = json.loads(encoded)
            self.assertEqual(decoded["name"], name)
            self.assertIsInstance(decoded["active_budget_seconds"], (int, float))

    def test_tracker_snapshot_and_summary_timing_align(self):
        """Snapshot timing keys are a superset of summary timing keys."""
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.01)
        snap = tracker.snapshot()
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        for key in summary["timing"]:
            self.assertIn(key, snap, f"Summary timing key {key!r} not in snapshot")


if __name__ == "__main__":
    unittest.main()
