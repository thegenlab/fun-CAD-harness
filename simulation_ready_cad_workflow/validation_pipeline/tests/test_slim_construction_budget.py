"""Accounting tests use a controlled clock, plus a real overlapping-process test."""
from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from ..slim_harness.construction_budget import BudgetConfig, BudgetExhausted, ProjectBudget, initialize_budget


MODULE = "simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_budget."


def _parallel_activity(project, ready, release, timing):
    budget = ProjectBudget(Path(project))
    with budget.activity("agent", heartbeat=False):
        start = time.monotonic()
        ready.put(start)
        release.wait(5)
        end = time.monotonic()
    timing.put((start, end))


class ConstructionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.now = 1000.0
        self.clock = patch(MODULE + "time.monotonic", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.boot = patch(MODULE + "_boot_id", return_value="test-boot")
        self.boot.start()
        self.addCleanup(self.boot.stop)
        initialize_budget(self.project, BudgetConfig(hours=1))
        self.budget = ProjectBudget(self.project, lease_seconds=10000)

    def advance(self, seconds):
        self.now += seconds

    def test_simulation_does_not_mask_an_author_who_keeps_working(self):
        with self.budget.activity("agent", heartbeat=False):
            self.advance(10)
            with ProjectBudget(self.project, lease_seconds=10000).activity("simulation", heartbeat=False):
                self.advance(20)
            self.advance(5)
        snapshot = self.budget.snapshot()
        self.assertEqual(snapshot["active_seconds"], 35)
        self.assertEqual(snapshot["excluded_seconds"], 0)

    def test_simulation_overrides_harness_side_waiting(self):
        with self.budget.activity("workflow", heartbeat=False):
            self.advance(10)
            with ProjectBudget(self.project, lease_seconds=10000).activity("simulation", heartbeat=False):
                self.advance(20)
            self.advance(5)
        snapshot = self.budget.snapshot()
        self.assertEqual((15, 20), (snapshot["active_seconds"], snapshot["excluded_by_reason"]["simulation"]))

    def test_simulation_wait_overrides_a_blocked_agent_turn(self):
        with self.budget.activity("agent", heartbeat=False):
            self.advance(10)
            with ProjectBudget(self.project, lease_seconds=10000).activity("simulation_wait", heartbeat=False):
                self.advance(20)
            self.advance(5)
        snapshot = self.budget.snapshot()
        self.assertEqual((15, 20), (snapshot["active_seconds"], snapshot["excluded_by_reason"]["simulation_wait"]))

    def test_nested_and_overlapping_process_handles_are_a_union_and_idle_is_free(self):
        self.advance(100)
        self.assertEqual(self.budget.snapshot()["active_seconds"], 0)
        other = ProjectBudget(self.project, lease_seconds=10000)
        with self.budget.activity("agent", heartbeat=False):
            self.advance(10)
            with other.activity("probe", heartbeat=False):
                self.advance(20)
            self.advance(5)
        self.advance(200)
        snapshot = other.snapshot()
        self.assertEqual(snapshot["active_seconds"], 35)
        self.assertEqual(snapshot["excluded_seconds"], 0)
        self.assertEqual(snapshot["active_sessions"], 0)

    def test_setup_and_intervention_override_agent_time_and_remain_visible(self):
        with self.budget.activity("agent", heartbeat=False):
            self.advance(10)
            with self.budget.activity("environment_setup", heartbeat=False):
                self.advance(60)
            self.budget.pause("human_intervention")
            self.advance(500)
            self.budget.resume()
            self.advance(20)
            with self.budget.activity("provider_wait", heartbeat=False):
                self.advance(30)
        snapshot = self.budget.snapshot()
        self.assertEqual(snapshot["active_seconds"], 30)
        self.assertEqual(snapshot["excluded_seconds"], 590)
        self.assertEqual(snapshot["excluded_by_reason"], {
            "environment_setup": 60, "human_intervention": 500, "provider_wait": 30})

    def test_reclassify_moves_charged_work_into_provider_wait(self):
        with self.budget.activity("agent", heartbeat=False):
            self.advance(400)
        snapshot = self.budget.reclassify_active("provider_wait", 120)
        self.assertEqual(snapshot["active_seconds"], 280)
        self.assertEqual(snapshot["excluded_seconds"], 120)
        self.assertEqual(snapshot["excluded_by_reason"], {"provider_wait": 120})
        self.assertEqual(snapshot["remaining_seconds"], 3320)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            self.budget.reclassify_active("provider_wait", 281)
        with self.assertRaisesRegex(ValueError, "existing exclusion"):
            self.budget.reclassify_active("sandbox_unavailable", 1)

    def test_reopen_preserves_budget_and_never_charges_between_invocations(self):
        with self.budget.activity("agent", heartbeat=False):
            self.advance(60)
        self.advance(10000)
        reopened = initialize_budget(self.project)
        self.assertEqual(reopened.snapshot()["active_seconds"], 60)
        with self.assertRaisesRegex(ValueError, "different configuration"):
            initialize_budget(self.project, BudgetConfig(hours=2))
        self.assertEqual(reopened.remaining_seconds, 3540)

    def test_older_ledger_without_audit_cap_uses_default_without_reset_or_rewrite(self):
        self.budget._transaction(lambda state,now:(state['config'].pop('audit_timeout_seconds'),state.update(active_seconds=123)))
        reopened=initialize_budget(self.project,BudgetConfig(hours=1))
        snapshot=reopened.snapshot()
        self.assertEqual(snapshot['config']['audit_timeout_seconds'],600)
        self.assertEqual(snapshot['config']['simulation_timeout_seconds'],180)
        self.assertEqual(snapshot['active_seconds'],123)
        self.assertEqual(snapshot['remaining_seconds'],3477)
        recorded=json.loads(reopened.path.read_text())
        self.assertNotIn('audit_timeout_seconds',recorded['config'])
        with self.assertRaisesRegex(ValueError,'different configuration'):
            initialize_budget(self.project,BudgetConfig(hours=1,audit_timeout_seconds=180))

    def test_dead_owner_recovery_uses_last_heartbeat_not_restart_time(self):
        activity = self.budget.activity("agent", heartbeat=False)
        handle = activity.__enter__()
        self.advance(4)
        handle.heartbeat()
        self.advance(600)
        with patch(MODULE + "_owner_alive", return_value=False):
            snapshot = ProjectBudget(self.project).snapshot()
        self.assertEqual(snapshot["active_seconds"], 4)
        self.assertEqual(snapshot["active_sessions"], 0)
        self.assertEqual(len(snapshot["recovered_sessions"]), 1)
        activity.__exit__(None, None, None)
        self.advance(1000)
        self.assertEqual(self.budget.snapshot()["active_seconds"], 4)

    def test_lost_heartbeat_lease_bounds_stalled_owner_accounting(self):
        budget = ProjectBudget(self.project, heartbeat_seconds=1, lease_seconds=5)
        with budget.activity("agent", heartbeat=False):
            self.advance(100)
            snapshot = budget.snapshot()
            self.assertEqual(snapshot["active_seconds"], 5)
            self.assertEqual(snapshot["active_sessions"], 0)

    def test_live_owner_heartbeat_resumes_after_a_ledger_stall_expired_it(self):
        budget = ProjectBudget(self.project)
        with budget.activity("agent", heartbeat=False) as handle:
            self.advance(40)
            handle.heartbeat()
            self.advance(1)
            snapshot = budget.snapshot()
        self.assertEqual(snapshot["active_seconds"], budget.lease_seconds + 1)
        self.assertEqual(snapshot["active_sessions"], 1)
        handle.heartbeat()
        self.assertEqual(budget.snapshot()["active_sessions"], 0)

    def test_sandbox_observer_does_not_declare_host_owner_dead(self):
        budget=ProjectBudget(self.project,lease_seconds=5)
        with patch(MODULE+'_pid_namespace',return_value='host-pid-namespace'):
            activity=budget.activity('agent',heartbeat=False)
            handle=activity.__enter__()
            self.advance(1)
            handle.heartbeat()
        self.advance(1)
        with patch(MODULE+'_pid_namespace',return_value='tool-pid-namespace'), \
             patch(MODULE+'os.kill',side_effect=ProcessLookupError) as kill:
            snapshot=budget.snapshot()
            kill.assert_not_called()
            self.assertEqual(snapshot['active_sessions'],1)
            self.assertEqual(snapshot['active_seconds'],2)
            self.advance(10)
            self.assertEqual(budget.snapshot()['active_sessions'],0)
        activity.__exit__(None,None,None)

    def test_reboot_does_not_count_unobserved_wall_time(self):
        with self.budget.activity("agent", heartbeat=False) as activity:
            self.advance(8)
            activity.heartbeat()
            self.now = 1
            with patch(MODULE + "_boot_id", return_value="next-boot"):
                snapshot = self.budget.snapshot()
                self.assertEqual(snapshot["active_seconds"], 8)
                self.assertEqual(snapshot["active_sessions"], 0)

    def test_summary_reserve_and_exhaustion_are_structured(self):
        with self.budget.activity("agent", heartbeat=False):
            self.advance(3000)
        self.assertEqual(self.budget.snapshot()["strategy_tier"], "time_tight")
        with self.budget.activity("agent", heartbeat=False):
            self.advance(480)
        self.assertEqual(self.budget.snapshot()["strategy_tier"], "finalizing")
        with self.assertRaises(BudgetExhausted) as caught:
            self.budget.check_available()
        self.assertEqual(caught.exception.details["code"], "CONSTRUCTION_SUMMARY_RESERVE")
        with self.assertRaises(BudgetExhausted):
            with self.budget.activity("simulation", heartbeat=False):
                self.fail("No new simulation may consume the completion-summary reserve")
        with self.budget.activity("completion_summary", heartbeat=False):
            self.advance(120)
        snapshot = self.budget.snapshot()
        self.assertEqual(snapshot["strategy_tier"], "exhausted")
        self.assertTrue(snapshot["exhausted"])
        with self.assertRaises(BudgetExhausted) as caught:
            self.budget.check_available(for_summary=True)
        self.assertEqual(caught.exception.details["required_action"], "write_completion_summary")
        self.assertEqual(caught.exception.details["code"], "CONSTRUCTION_BUDGET_EXHAUSTED")

    def test_uninitialized_evaluation_project_does_not_get_a_budget(self):
        project = self.project / "frozen-evaluation"
        with self.assertRaises(FileNotFoundError):
            ProjectBudget(project).snapshot()
        self.assertFalse((project / "construction_budget.json").exists())

    def test_configuration_rejects_invalid_presets_and_limits(self):
        self.assertEqual(BudgetConfig(hours=3).total_seconds,10800)
        for values in ({"hours": 4}, {"hours": True}, {"simulation_timeout_seconds": float("nan")},
                       {'audit_timeout_seconds':float('inf')},{'audit_timeout_seconds':0},{'audit_timeout_seconds':True},
                       {"max_tests": 0}, {"summary_reserve_seconds": 7200}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                BudgetConfig(**values)


@unittest.skipIf(os.name == "nt", "POSIX process overlap integration")
class ConstructionBudgetConcurrencyTests(unittest.TestCase):
    def test_concurrent_writers_preserve_overlapping_intervals_without_double_counting(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = initialize_budget(Path(directory))
            ctx = multiprocessing.get_context("fork")
            ready, timing = ctx.Queue(), ctx.Queue()
            release = ctx.Event()
            children = [ctx.Process(target=_parallel_activity, args=(directory, ready, release, timing)) for _ in range(2)]
            try:
                for child in children:
                    child.start()
                starts = [ready.get(timeout=5) for _ in children]
                self.assertEqual(budget.snapshot()["active_sessions"], 2)
                time.sleep(.15)
                release.set()
                intervals = [timing.get(timeout=5) for _ in children]
                for child in children:
                    child.join(timeout=5)
                    self.assertEqual(child.exitcode, 0)
                snapshot = budget.snapshot()
                union = max(end for _, end in intervals) - min(starts)
                self.assertAlmostEqual(snapshot["active_seconds"], union, delta=.05)
                self.assertEqual(snapshot["active_sessions"], 0)
                self.assertLess(snapshot["active_seconds"], sum(end-start for start, end in intervals) * .8)
            finally:
                release.set()
                for child in children:
                    child.join(timeout=5)
                    if child.is_alive():
                        child.terminate()
                        child.join()


if __name__ == "__main__":
    unittest.main()
