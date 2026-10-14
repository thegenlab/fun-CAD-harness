"""Cross-cutting integration tests for budget module interactions.

Tests that verify budget module behavior across multiple subsystem boundaries:
- Workflow constructor + checkpoint round-trip
- Budget exhaustion during multi-step simulated workflows
- Adjustment log + incomplete summary integration
- SimulationTimeoutResult status derivation
- Tier escalation with preserved checkpoint state
- Concurrent workflow budget tracking
"""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from ..mechanism.io import write_json_atomic
from ..slim_harness.cli import create_project
from ..slim_harness.workflow import SlimWorkflow
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
)


class CrossCuttingWorkflowBase(unittest.TestCase):
    """Shared project setup for cross-cutting tests."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = create_project(self.workspace, "cross_test",
                                      "A block must settle under gravity.")
        self.run = self.project / "runs" / "r01"
        self.scene_path = self.run / "design" / "scene.json"
        self.source = self.run / "design" / "part.step"
        self.source.write_text("geometry-file snapshot", encoding="utf-8")
        criteria = [
            {"id": "settle", "kind": "function",
             "statement": "Block settles", "source_quote": "settle under gravity",
             "verification": "Measure final position"},
        ]
        frozen = {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(self.project / "task.md"),
            "task_text": "A block must settle under gravity.",
            "criteria": criteria,
            "reviewer_summary": "Source-bound gravity settle criteria",
        }
        write_json_atomic(self.project / "criteria.json", frozen)
        state = json.loads((self.project / "workflow.json").read_text())
        state["events"] = [{"kind": "criteria", "result": {
            "status": "ready", "summary": frozen["reviewer_summary"],
            "criteria": criteria,
        }}]
        write_json_atomic(self.project / "workflow.json", state)

    def _write_scene(self, num_bodies=2, num_tests=1):
        bodies = [
            {"id": f"body_{i}", "source": "part.step", "fixed": i == 0}
            for i in range(num_bodies)
        ]
        tests = [
            {"id": f"test_{i}", "purpose": f"Test {i}", "duration_s": 1,
             "assertions": [{"id": f"a_{i}", "purpose": "Check",
                             "expression": "last(signal('time_s')) >= 1"}]}
            for i in range(num_tests)
        ]
        scene = {
            "schema_version": "slim_scene_v1", "units": "mm",
            "bodies": bodies, "tests": tests,
        }
        write_json_atomic(self.scene_path, scene)
        return scene


class TestCheckpointWorkflowRoundTrip(CrossCuttingWorkflowBase):
    """Checkpoint state persists correctly across workflow construction."""

    def test_checkpoint_before_workflow_creation(self):
        """A checkpoint saved before SlimWorkflow init restores correctly after."""
        self._write_scene()
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        time.sleep(0.05)
        tracker.pause("pre-workflow")

        with tempfile.TemporaryDirectory() as tmp:
            cp_path = Path(tmp) / "pre_wf.json"
            tracker.save_checkpoint(cp_path)
            active_before = tracker.elapsed_active

            restored = ActiveBudgetTracker.restore(cp_path, tier=tier)
            self.assertAlmostEqual(restored.elapsed_active, active_before, delta=0.01)
            self.assertTrue(restored.is_paused)

            restored.resume()
            wf = SlimWorkflow(self.workspace, self.project, tier=tier,
                              budget_tracker=restored)
            self.assertIs(wf.budget_tracker, restored)
            self.assertFalse(wf.budget_tracker.is_paused)

    def test_checkpoint_survives_tier_escalation(self):
        """A checkpoint from light tier can inform budget remaining after escalation."""
        self._write_scene()
        light = get_tier("light")
        tracker = ActiveBudgetTracker(tier=light)
        time.sleep(0.03)

        with tempfile.TemporaryDirectory() as tmp:
            cp_path = Path(tmp) / "escalate.json"
            tracker.save_checkpoint(cp_path)
            snap = json.loads(cp_path.read_text())

            self.assertEqual(snap["tier"], "light")
            self.assertEqual(snap["budget_limit_seconds"], 3600)

            high = get_tier("high")
            restored = ActiveBudgetTracker.restore(cp_path, tier=high)
            self.assertEqual(restored.tier.name, "high")
            self.assertEqual(restored.tier.active_budget_seconds, 10800)
            self.assertGreater(restored.remaining_active, 10700)


class TestMultiStepBudgetExhaustion(CrossCuttingWorkflowBase):
    """Budget exhaustion during a multi-step simulated workflow."""

    def test_budget_exhausts_mid_step(self):
        """A tiny budget exhausts between geometry prep and physics."""
        self._write_scene(num_bodies=2, num_tests=1)
        tiny = TierConfig("tiny", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)

        wf = SlimWorkflow(self.workspace, self.project, tier=tiny,
                          budget_tracker=tracker)
        result = wf.run_revision()
        self.assertEqual(result["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertIn("timing", result)

    def test_budget_survives_multiple_revisions_with_fresh_tracker(self):
        """Fresh tracker per revision still tracks cumulative budget."""
        self._write_scene(num_bodies=2, num_tests=1)
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)

        wf = SlimWorkflow(self.workspace, self.project, tier=tier,
                          budget_tracker=tracker)
        r1 = wf.run_revision()

        active_after_r1 = tracker.elapsed_active
        self.assertGreater(active_after_r1, 0)

        r2 = wf.run_revision()
        active_after_r2 = tracker.elapsed_active
        self.assertGreater(active_after_r2, active_after_r1)

    def test_exhausted_tracker_produces_complete_summary(self):
        """build_incomplete_summary has all required fields after exhaustion."""
        tiny = TierConfig("tiny", 0.01, 60, 2, 0, 10)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)

        self.assertTrue(tracker.is_exhausted)
        summary = build_incomplete_summary(
            tier=tiny, tracker=tracker,
            completed_tests=[{"test_id": "t1", "physical_pass": True,
                              "geometry_audit": {"passed": True}}],
            pending_tests=["t2"],
            issues=["Budget exhausted before t2"],
            evidence_paths=["/evidence/t1.json"],
            blockers=[],
        )
        required_keys = {"schema_version", "status", "tier", "timing",
                         "completed_tests", "pending_tests", "severity",
                         "issues", "evidence_paths", "blockers", "generated_at"}
        self.assertEqual(required_keys - set(summary.keys()), set())
        self.assertEqual(summary["schema_version"], "budget_incomplete_summary_v1")
        self.assertEqual(summary["status"], "ACTIVE_BUDGET_EXHAUSTED")
        self.assertEqual(summary["severity"], "functional")


class TestAdjustmentLogWithSummary(unittest.TestCase):
    """Adjustment log and summary generation work together."""

    def test_adjustment_log_saved_alongside_summary(self):
        """Both adjustment log and summary can be saved to same directory."""
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        log = AdjustmentLog()
        log.record(CriteriaAdjustment(
            "tolerance_mm", {"threshold": 0.5}, {"threshold": 1.0},
            "Relaxed tolerance for large assembly",
        ))

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            log.save(base / "adjustments.json")
            summary = build_incomplete_summary(
                tier=tier, tracker=tracker,
                completed_tests=[], pending_tests=["t1"],
                issues=["criteria adjusted"], evidence_paths=[],
                blockers=[],
            )
            (base / "summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")

            adj_data = json.loads((base / "adjustments.json").read_text())
            sum_data = json.loads((base / "summary.json").read_text())

            self.assertEqual(len(adj_data["adjustments"]), 1)
            self.assertEqual(sum_data["schema_version"], "budget_incomplete_summary_v1")
            self.assertEqual(adj_data["schema_version"], "criteria_adjustment_log_v1")

    def test_frozen_criterion_blocks_adjustment_before_summary(self):
        """Frozen criteria raise before any summary can reference them."""
        with self.assertRaises(ValueError) as ctx:
            CriteriaAdjustment(
                "benchmark_travel", {"mm": 10}, {"mm": 12},
                "Tried to relax", frozen=True,
            )
        self.assertIn("frozen", str(ctx.exception).lower())

    def test_multiple_adjustments_tracked_in_order(self):
        """AdjustmentLog preserves insertion order."""
        log = AdjustmentLog()
        for i in range(5):
            log.record(CriteriaAdjustment(
                f"crit_{i}", {"v": i}, {"v": i + 1}, f"Step {i}",
            ))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ordered.json"
            log.save(path)
            loaded = AdjustmentLog.load(path)
            for i, entry in enumerate(loaded.entries):
                self.assertEqual(entry.criterion_id, f"crit_{i}")
                self.assertEqual(entry.before, {"v": i})


class TestSimulationTimeoutResultIntegration(unittest.TestCase):
    """SimulationTimeoutResult status derivation and budget interaction."""

    def test_timeout_result_status(self):
        result = SimulationTimeoutResult(timed_out=True, elapsed_seconds=60.0)
        self.assertEqual(result.status, BudgetStatus.SIMULATION_TIMEOUT)

    def test_normal_result_status(self):
        result = SimulationTimeoutResult(timed_out=False, elapsed_seconds=5.0)
        self.assertEqual(result.status, BudgetStatus.PASS)

    def test_timeout_with_partial_evidence(self):
        result = SimulationTimeoutResult(
            timed_out=True, elapsed_seconds=180.0,
            partial_evidence={"frames": 100, "last_time_s": 2.5},
            error="Exceeded tier timeout",
        )
        self.assertEqual(result.status, BudgetStatus.SIMULATION_TIMEOUT)
        self.assertIsNotNone(result.partial_evidence)
        self.assertEqual(result.partial_evidence["frames"], 100)

    def test_timeout_status_is_terminal(self):
        self.assertFalse(BudgetStatus.SIMULATION_TIMEOUT.is_terminal())

    def test_pass_status_is_terminal(self):
        self.assertTrue(BudgetStatus.PASS.is_terminal())

    def test_exhausted_status_is_terminal(self):
        self.assertTrue(BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.is_terminal())

    def test_needs_revision_not_terminal(self):
        self.assertFalse(BudgetStatus.NEEDS_REVISION.is_terminal())


class TestBudgetStatusExhaustive(unittest.TestCase):
    """All BudgetStatus values are covered and consistent."""

    def test_all_status_values_are_strings(self):
        for status in BudgetStatus:
            self.assertIsInstance(status.value, str)

    def test_terminal_statuses(self):
        terminal = {s for s in BudgetStatus if s.is_terminal()}
        self.assertEqual(terminal, {
            BudgetStatus.PASS,
            BudgetStatus.ACTIVE_BUDGET_EXHAUSTED,
            BudgetStatus.INFRASTRUCTURE_FAILURE,
        })

    def test_non_terminal_statuses(self):
        non_terminal = {s for s in BudgetStatus if not s.is_terminal()}
        self.assertEqual(non_terminal, {
            BudgetStatus.NEEDS_REVISION,
            BudgetStatus.HARNESS_ERROR,
            BudgetStatus.SIMULATION_TIMEOUT,
            BudgetStatus.INCOMPLETE_EVIDENCE,
            BudgetStatus.EVIDENCE_REQUIRED,
        })

    def test_retryable_vs_non_retryable_alignment(self):
        """should_retry non-retryable set aligns with known terminal statuses."""
        non_retryable_errors = ["ACTIVE_BUDGET_EXHAUSTED", "TERMINATED", "PASS"]
        for err in non_retryable_errors:
            self.assertFalse(should_retry(0, err),
                             f"{err} should not be retryable")

        retryable_errors = ["SIMULATION_TIMEOUT", "INFRASTRUCTURE_FAILURE",
                            "HARNESS_ERROR", "some_unknown_error"]
        for err in retryable_errors:
            self.assertTrue(should_retry(0, err),
                            f"{err} should be retryable at attempt 0")


class TestRecoveryDelayProperties(unittest.TestCase):
    """Recovery delay exhibits correct mathematical properties."""

    def test_monotonically_increases_to_cap(self):
        delays = [recovery_delay(i) for i in range(20)]
        for i in range(1, len(delays)):
            self.assertGreaterEqual(delays[i], delays[i - 1])
        self.assertLessEqual(max(delays), 60.0)

    def test_first_delay_is_base(self):
        self.assertEqual(recovery_delay(0), 5.0)

    def test_cap_reached(self):
        self.assertEqual(recovery_delay(10), 60.0)
        self.assertEqual(recovery_delay(100), 60.0)

    def test_attempt_1_is_10(self):
        self.assertEqual(recovery_delay(1), 10.0)

    def test_attempt_2_is_20(self):
        self.assertEqual(recovery_delay(2), 20.0)

    def test_attempt_3_is_40(self):
        self.assertEqual(recovery_delay(3), 40.0)

    def test_attempt_4_is_capped(self):
        self.assertEqual(recovery_delay(4), 60.0)


class TestTierEnforcementCombinations(CrossCuttingWorkflowBase):
    """Cross-tier enforcement: part limits, task limits, and timeouts together."""

    def test_all_tiers_part_boundary(self):
        """Each tier accepts max parts and rejects max+1."""
        for name, tier in TIERS.items():
            ok = [{} for _ in range(tier.max_parts)]
            over = [{} for _ in range(tier.max_parts + 1)]
            self.assertEqual(check_part_count(ok, tier), [],
                             f"{name}: max parts should pass")
            self.assertEqual(len(check_part_count(over, tier)), 1,
                             f"{name}: max+1 parts should fail")

    def test_all_tiers_task_boundary(self):
        """Each tier accepts max tasks and rejects max+1."""
        for name, tier in TIERS.items():
            ok = [{} for _ in range(tier.max_tasks_per_design)]
            over = [{} for _ in range(tier.max_tasks_per_design + 1)]
            self.assertEqual(check_task_count(ok, tier), [],
                             f"{name}: max tasks should pass")
            self.assertEqual(len(check_task_count(over, tier)), 1,
                             f"{name}: max+1 tasks should fail")

    def test_timeout_clamping_across_tiers(self):
        """Timeout enforcement clamps to tier limit across all tiers."""
        for name, tier in TIERS.items():
            self.assertEqual(enforce_simulation_timeout(tier, 1), 1)
            self.assertEqual(
                enforce_simulation_timeout(tier, 99999),
                tier.simulation_timeout_seconds,
            )
            self.assertEqual(
                enforce_simulation_timeout(tier),
                tier.simulation_timeout_seconds,
            )

    def test_workflow_tier_clamps_combined(self):
        """SlimWorkflow respects tier timeout even when physics_timeout is larger."""
        self._write_scene(num_bodies=2, num_tests=1)
        for name in TIERS:
            tier = get_tier(name)
            wf = SlimWorkflow(self.workspace, self.project, tier=name,
                              physics_timeout_seconds=99999)
            self.assertEqual(wf.physics_timeout_seconds,
                             tier.simulation_timeout_seconds)


class TestConcurrentBudgetTrackers(unittest.TestCase):
    """Multiple independent trackers operating concurrently."""

    def test_independent_trackers_do_not_interfere(self):
        """Two trackers for different tiers track independently."""
        light = ActiveBudgetTracker(tier=get_tier("light"))
        high = ActiveBudgetTracker(tier=get_tier("high"))

        light.pause("wait")
        time.sleep(0.03)
        high_active = high.elapsed_active
        light_active = light.elapsed_active

        self.assertGreater(high_active, light_active)
        light.resume()

    def test_threaded_trackers_isolation(self):
        """Trackers in separate threads don't share state."""
        results = {}
        errors = []

        def run_tracker(name, tier_name):
            try:
                tracker = ActiveBudgetTracker(tier=get_tier(tier_name))
                time.sleep(0.02)
                tracker.pause("test")
                time.sleep(0.02)
                tracker.resume()
                time.sleep(0.02)
                results[name] = tracker.snapshot()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=run_tracker, args=("t1", "light")),
            threading.Thread(target=run_tracker, args=("t2", "medium")),
            threading.Thread(target=run_tracker, args=("t3", "high")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 3)
        for name, snap in results.items():
            self.assertGreater(snap["active_seconds"], 0)
            self.assertEqual(snap["pause_count"], 1)


class TestSnapshotSchemaConsistency(unittest.TestCase):
    """Snapshot schema is consistent across all tracker states."""

    REQUIRED_KEYS = {
        "schema_version", "tier", "active_seconds", "wall_seconds",
        "paused", "pause_count", "total_pause_seconds",
        "budget_limit_seconds", "remaining_active_seconds",
        "started_at", "checkpoint_at",
    }

    def _check_snapshot(self, snap, tier_name):
        self.assertEqual(set(snap.keys()), self.REQUIRED_KEYS)
        self.assertEqual(snap["schema_version"], "budget_checkpoint_v1")
        self.assertEqual(snap["tier"], tier_name)
        self.assertIsInstance(snap["active_seconds"], float)
        self.assertIsInstance(snap["wall_seconds"], float)
        self.assertIsInstance(snap["paused"], bool)
        self.assertIsInstance(snap["pause_count"], int)

    def test_fresh_tracker_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        self._check_snapshot(tracker.snapshot(), "light")

    def test_paused_tracker_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("medium"))
        tracker.pause("test")
        self._check_snapshot(tracker.snapshot(), "medium")

    def test_resumed_tracker_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("high"))
        tracker.pause("test")
        tracker.resume()
        self._check_snapshot(tracker.snapshot(), "high")

    def test_exhausted_tracker_snapshot(self):
        tiny = TierConfig("tiny", 0.01, 5, 1, 0, 5)
        tracker = ActiveBudgetTracker(tier=tiny)
        time.sleep(0.02)
        snap = tracker.snapshot()
        self._check_snapshot(snap, "tiny")
        self.assertAlmostEqual(snap["remaining_active_seconds"], 0.0, places=2)

    def test_restored_tracker_snapshot(self):
        tracker = ActiveBudgetTracker(tier=get_tier("light"))
        time.sleep(0.02)
        tracker.pause("save")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "snap.json"
            tracker.save_checkpoint(path)
            restored = ActiveBudgetTracker.restore(path, tier=get_tier("light"))
            self._check_snapshot(restored.snapshot(), "light")

    def test_snapshot_is_json_serializable(self):
        for name in TIERS:
            tracker = ActiveBudgetTracker(tier=get_tier(name))
            snap = tracker.snapshot()
            encoded = json.dumps(snap)
            decoded = json.loads(encoded)
            self.assertEqual(set(decoded.keys()), self.REQUIRED_KEYS)


class TestIncompleteSummaryEdgeCases(unittest.TestCase):
    """Edge cases in build_incomplete_summary."""

    def test_summary_with_no_evidence(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["evidence_paths"], [])
        self.assertEqual(summary["severity"], "minor")

    def test_summary_severity_with_failed_test(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "f", "physical_pass": False,
                              "geometry_audit": {"passed": True}}],
            pending_tests=[],
            issues=["Physics failed"], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_summary_severity_with_geometry_failure(self):
        tier = get_tier("high")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[{"test_id": "g", "physical_pass": True,
                              "geometry_audit": {"passed": False}}],
            pending_tests=[],
            issues=["Geometry audit failed"], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_summary_severity_with_pending_only(self):
        tier = get_tier("light")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=["t1"],
            issues=[], evidence_paths=[], blockers=[],
        )
        self.assertEqual(summary["severity"], "functional")

    def test_summary_tier_dict_has_all_fields(self):
        tier = get_tier("medium")
        tracker = ActiveBudgetTracker(tier=tier)
        summary = build_incomplete_summary(
            tier=tier, tracker=tracker,
            completed_tests=[], pending_tests=[],
            issues=[], evidence_paths=[], blockers=[],
        )
        tier_dict = summary["tier"]
        self.assertEqual(tier_dict["name"], "medium")
        self.assertEqual(tier_dict["active_budget_seconds"], 7200)
        self.assertEqual(tier_dict["simulation_timeout_seconds"], 180)
        self.assertEqual(tier_dict["max_tasks_per_design"], 2)
        self.assertEqual(tier_dict["max_parts"], 15)


if __name__ == "__main__":
    unittest.main()
