"""Short entrypoint checks for observed timing, named parts, and resource wait."""
from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import run_bounded_subprocess
from ..slim_harness.budget import (
    MAX_RECOVERY_RETRIES,
    PROVIDER_BACKOFF_CAP_SECONDS,
    PROVIDER_RECOVERY_RETRIES,
    ProjectBudget,
    TierConfig,
    provider_needs_external_action,
    recovery_delay,
)
from ..slim_harness.cli import create_project
from ..slim_harness.physical_parts import (
    PART_INVENTORY_REQUIRED,
    check_physical_part_inventory,
    is_time_only_continuation,
    resolve_physical_part_inventory,
)
from ..slim_harness.resource_wait import (
    RESOURCE_POLICY_REQUIRED,
    load_declared_resource_requirement,
    wait_for_observed_resources,
)
from ..slim_harness.workflow import SlimWorkflow


PREFIX = "simulation_ready_cad_workflow.validation_pipeline.slim_harness."


class FakeProbe:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def sample(self, interval_seconds=0.0):
        index = min(self.calls, len(self.values) - 1)
        self.calls += 1
        available = self.values[index]
        return _Snapshot(available)


class _Snapshot:
    def __init__(self, available_gib):
        self.available_memory_gib = available_gib

    def to_dict(self):
        return {"available_memory_gib": self.available_memory_gib}


class RemainingBudgetGapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.project = create_project(self.workspace, "gaps_fixture", "A slider must move.")
        self.tier = TierConfig("fixture", 5, 0.3, 2, 0, 10)

    def test_observed_segments_persist_across_cli_and_do_not_rewrite_active(self):
        with ProjectBudget(self.project, self.tier) as budget:
            budget.record_observed_work(0.04)
            budget.record_observed_wait(0.11)
            budget.record_unclassified_provider(0.08)
            with budget.active("physical_simulation"):
                time.sleep(0.02)
        first = json.loads((self.project / "budget_checkpoint.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(first["observed_work_seconds"], 0.05)
        self.assertGreaterEqual(first["observed_wait_seconds"], 0.11)
        self.assertGreaterEqual(first["unclassified_provider_seconds"], 0.08)
        self.assertTrue(first["timing_overlap"]["unclassified_is_not_net_thinking"])
        self.assertIn("upper_estimate", first["accounting_basis"])
        self.assertGreater(first["active_seconds"], 0.015)
        self.assertLess(first["active_seconds"], first["wall_seconds"])
        with ProjectBudget(self.project) as budget:
            budget.record_observed_wait(0.05)
        second = json.loads((self.project / "budget_checkpoint.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(second["observed_wait_seconds"], 0.16)
        self.assertGreaterEqual(second["unclassified_provider_seconds"], 0.08)
        self.assertEqual(first["started_at"], second["started_at"])

    def test_named_parts_count_three_on_one_body_and_reject_missing(self):
        design = self.project / "runs/r01/design"
        for filename in ("body-001.step", "body-002.step"):
            (design/filename).write_text("Path-binding fixture; geometry validity is tested separately.", encoding="utf-8")
        write_json_atomic(design / "parts.json", {
            "description": "fixture",
            "parts": [
                {"name": "base", "body": "frame", "role": "fixed", "source_solid_count": 1},
                {"name": "foot", "body": "frame", "role": "fixed", "source_solid_count": 1},
                {"name": "weight", "body": "moving", "role": "dynamic", "source_solid_count": 2,
                 "source_members": ["weight:solid_1", "weight:solid_2"]},
            ],
            "body_sources": {"frame": "body-001.step", "moving": "body-002.step"},
        })
        scene = {
            "bodies": [{"id": "frame", "source": "body-001.step"}, {"id": "moving", "source": "body-002.step"}],
            "authoring_sources": ["parts.json"],
        }
        inventory = resolve_physical_part_inventory(design / "scene.json", scene, project=self.project)
        self.assertEqual("ready", inventory["status"])
        self.assertEqual(3, inventory["machine_part_count"])
        self.assertEqual(2, len(scene["bodies"]))
        self.assertEqual([], check_physical_part_inventory(inventory, self.tier, project=self.project))
        empty = Path(self.temp.name) / "no_inventory"
        empty.mkdir()
        missing = resolve_physical_part_inventory(empty / "scene.json", {"bodies": scene["bodies"]})
        self.assertEqual(PART_INVENTORY_REQUIRED, missing["status"])
        self.assertTrue(any(PART_INVENTORY_REQUIRED in issue for issue in missing["issues"]))

    def test_explicit_coupon_category_controls_count_not_task_prose(self):
        design = self.project / "runs/r01/design"
        for filename in ("base.step", "bolt.step"):
            (design/filename).write_text("Path-binding fixture", encoding="utf-8")
        write_json_atomic(design / "all_print_inventory.json", {
            "parts": [
                {"id": "base", "category": "machine", "body_id":"base", "source":"base.step"},
                {"id": "bolt", "category": "machine", "body_id":"bolt", "source":"bolt.step"},
                {"id": "fit_coupon", "category": "coupon"},
            ]
        })
        scene = {"bodies": [{"id": "base", "source":"base.step"}, {"id": "bolt", "source":"bolt.step"}], "artifacts": ["all_print_inventory.json"]}
        inventory = resolve_physical_part_inventory(
            design / "scene.json", scene, task_text="A slider must move.")
        self.assertEqual(2, inventory["machine_part_count"])
        self.assertEqual(1, inventory["excluded_coupon_count"])
        counted = resolve_physical_part_inventory(
            design / "scene.json", scene, task_text="Count the coupon as a machine part.")
        self.assertEqual(2, counted["machine_part_count"])

    def test_time_only_continuation_keeps_original_tests_and_parts(self):
        write_json_atomic(self.project / "scope.json", {
            "medium_time_only": True,
            "additional_active_seconds": 7200,
            "simulation_timeout_seconds": 180,
            "max_tests": 33,
            "machine_parts": 16,
            "coupons": 3,
        })
        custom = TierConfig("lock_original_scope_medium_time", 7200, 180, 33, 0, 17)
        medium = TierConfig("medium", 7200, 180, 2, 0, 15)
        self.assertTrue(is_time_only_continuation(self.project, custom))
        design = self.project / "runs/r01/design"
        (design/"body-001.step").write_text("Path-binding fixture", encoding="utf-8")
        parts = [{"name": f"p{i}", "body": "frame", "role": "dynamic", "source_solid_count": 1} for i in range(16)]
        write_json_atomic(design / "parts.json", {
            "parts": parts, "body_sources": {"frame": "body-001.step"}})
        scene = {"bodies": [{"id": "frame", "source":"body-001.step"}], "authoring_sources": ["parts.json"],
                 "tests": [{"id": f"t{i}"} for i in range(8)]}
        inventory = resolve_physical_part_inventory(
            design / "scene.json", scene, project=self.project)
        self.assertEqual(16, inventory["machine_part_count"])
        self.assertEqual([], check_physical_part_inventory(inventory, custom, project=self.project))
        self.assertGreater(len(check_physical_part_inventory(inventory, medium)), 0)

    def test_resource_wait_is_out_of_active_and_missing_policy_is_explicit(self):
        with ProjectBudget(self.project, self.tier) as budget:
            before = budget.tracker.elapsed_active
            probe = FakeProbe([0.2, 0.2, 2.0])
            waited = wait_for_observed_resources(
                probe, 1.0, max_wait_seconds=0.4, poll_seconds=0.02)
            budget.record_observed_wait(waited["waited_seconds"])
            after = budget.tracker.elapsed_active
        self.assertEqual("ready", waited["status"])
        self.assertGreaterEqual(waited["waited_seconds"], 0.02)
        self.assertLess(after - before, 0.01)
        checkpoint = json.loads((self.project / "budget_checkpoint.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(checkpoint["observed_wait_seconds"], 0.02)
        self.assertIsNone(load_declared_resource_requirement(self.project))
        write_json_atomic(self.project / "scope.json", {"medium_time_only": True, "machine_parts": 16})
        self.assertIsNone(load_declared_resource_requirement(self.project))
        tiny = wait_for_observed_resources(FakeProbe([1.0]), 0.01, max_wait_seconds=0.1)
        self.assertEqual("ready", tiny["status"])
        write_json_atomic(self.project / "resource_policy.json", {
            "required_available_memory_gib": 1.5, "max_resource_wait_seconds": 0.2,
            "resource_poll_seconds": 0.02})
        declared = load_declared_resource_requirement(self.project)
        self.assertEqual(1.5, declared["required_available_memory_gib"])
        exhausted = wait_for_observed_resources(
            FakeProbe([0.1]), 1.5, max_wait_seconds=0.08, poll_seconds=0.02)
        self.assertEqual("wait_exhausted", exhausted["status"])

    def test_quota_without_reset_needs_external_action(self):
        self.assertTrue(provider_needs_external_action({"kind": "quota", "message": "usage limit"}))
        self.assertTrue(provider_needs_external_action({"kind": "auth"}))
        self.assertFalse(provider_needs_external_action({"kind": "quota", "retry_after_seconds": 2}))
        self.assertFalse(provider_needs_external_action({"kind": "timeout"}))
        self.assertTrue(provider_needs_external_action({"kind":"quota","retry_after_seconds":float("nan")}))

    def test_explicit_provider_reset_waits_and_retries_without_active_charge(self):
        class Worker:
            session_id="fixture"
            def __init__(self):self.calls=0
            def run_turn(self,*args,**kwargs):
                self.calls+=1
                time.sleep(.01)
                if self.calls==1:
                    return {"status":"provider_error","provider_error":{"kind":"quota","retry_after_seconds":.06,"message":"explicit reset"}}
                return {"status":"ready"}
        raw=Worker()
        with ProjectBudget(self.project,self.tier) as budget:
            result=budget.worker(raw).run_turn(self.workspace,"fixture",self.project/"worker_turns/reset",1)
            snap=budget.annotate({})["budget_timing"]
        self.assertEqual("ready",result["status"])
        self.assertEqual(2,raw.calls)
        self.assertGreaterEqual(snap["observed_wait_seconds"],.055)
        self.assertGreater(snap["unclassified_provider_seconds"],.015)
        self.assertGreater(snap["wall_seconds"]-snap["active_seconds"],.05)

    def test_provider_outages_retry_longer_than_other_failures(self):
        class Worker:
            session_id="fixture"
            def __init__(self,error):self.error,self.calls=error,0
            def run_turn(self,*args,**kwargs):
                self.calls+=1
                return {"status":"provider_error","provider_error":self.error}
        cases=(
            # Recorded: Codex CLI 0.162.1 and Claude Code 2.1.296 behind an API gateway.
            ({"kind":"exit","message":'{"type": "error", "message": "Selected model is at capacity. Please try a different model."}'},
             PROVIDER_RECOVERY_RETRIES),
            ({"kind":"exit","message":'{"api_error_status": 403, "result": "Failed to authenticate. API Error: 403 '
              'insufficient user balance (request id: 01M4TDNX57DWTTZ0DD5MQDP1Q5)"}'},PROVIDER_RECOVERY_RETRIES),
            ({"kind":"timeout","message":"Codex turn exceeded its time budget."},MAX_RECOVERY_RETRIES),
            ({"kind":"exit","message":"API Error: 400 invalid model"},0),
            # Durations and parse positions are not status codes; a local parse error is not an outage.
            ({"kind":"exit","message":'{"result": "API Error: 400 invalid model", "duration_ms": 15031}'},0),
            ({"kind":"response","message":"Expecting value: line 1 column 504 (char 503)"},0),
        )
        for index,(error,retries) in enumerate(cases):
            with self.subTest(message=error["message"]):
                raw=Worker(error)
                with ProjectBudget(self.project,self.tier) as budget, \
                        patch(PREFIX+"budget.recovery_delay",return_value=0) as delay:
                    result=budget.worker(raw).run_turn(self.workspace,"fixture",self.project/f"worker_turns/outage-{index}",1)
                self.assertEqual((retries+1,retries),(raw.calls,result["recovery_attempts"]))
                self.assertEqual([PROVIDER_BACKOFF_CAP_SECONDS]*retries,[call.args[1] for call in delay.call_args_list])
        self.assertEqual((60.0,PROVIDER_BACKOFF_CAP_SECONDS),(recovery_delay(9),recovery_delay(9,PROVIDER_BACKOFF_CAP_SECONDS)))

    def test_admission_then_real_child_subprocess(self):
        probe = FakeProbe([4.0])
        admission = wait_for_observed_resources(
            probe, 1.0, max_wait_seconds=0.2, poll_seconds=0.01)
        self.assertEqual("ready", admission["status"])
        result = run_bounded_subprocess(
            ["C:\\Tools\\miniconda3\\python.exe", "-c", "print('admitted')"],
            timeout_seconds=5)
        self.assertEqual(0, result.returncode)
        self.assertIn(b"admitted", result.stdout)

    def test_time_only_workflow_does_not_apply_ordinary_medium_caps(self):
        (self.project / "task.md").write_text("A slider must move.", encoding="utf-8")
        write_json_atomic(self.project / "scope.json", {
            "medium_time_only": True, "max_tests": 33, "machine_parts": 16})
        write_json_atomic(self.project / "criteria.json", {
            "schema_version": "slim_frozen_criteria_v1",
            "task_text": "A slider must move.",
            "criteria": [{"id": "move", "kind": "function", "statement": "Move",
                          "source_quote": "move", "verification": "Observe"}],
            "reviewer_summary": "fixture"})
        state = json.loads((self.project / "workflow.json").read_text(encoding="utf-8"))
        state["events"] = [{"kind": "criteria", "result": {
            "status": "ready", "summary": "fixture",
            "criteria": json.loads((self.project / "criteria.json").read_text(encoding="utf-8"))["criteria"]}}]
        write_json_atomic(self.project / "workflow.json", state)
        design = self.project / "runs/r01/design"
        (design / "part.step").write_text("fixture", encoding="utf-8")
        parts = [{"name": f"p{i}", "body": "frame", "role": "dynamic", "source_solid_count": 1} for i in range(16)]
        write_json_atomic(design / "parts.json", {
            "parts": parts, "body_sources": {"frame": "part.step"}})
        write_json_atomic(design / "scene.json", {
            "schema_version": "slim_scene_v1", "units": "mm",
            "bodies": [{"id": "frame", "source": "part.step", "fixed": False}],
            "authoring_sources": ["parts.json"],
            "tests": [{"id": f"t{i}", "purpose": "Fixture", "duration_s": 1,
                       "assertions": [{"id": "a", "purpose": "Fixture",
                                       "expression": "last(signal('time_s')) >= 1"}]}
                      for i in range(8)]})
        custom = TierConfig("lock_original_scope_medium_time", 7200, 180, 33, 0, 17)
        class FakeWorker:
            session_id = "fixture"
            def run_turn(self, *args, **kwargs):
                return {"status": "ready", "summary": "fixture", "issues": [], "evidence_paths": []}
        with patch(PREFIX + "workflow.assert_inactive"), patch(PREFIX + "workflow.create_worker", return_value=FakeWorker()):
            workflow = SlimWorkflow(self.workspace, self.project, tier=custom)
        result = workflow.run_revision()
        self.assertNotEqual("part_count", result.get("budget_violation"))
        self.assertNotEqual("task_count", result.get("budget_violation"))
        self.assertNotEqual("physical_part_inventory", result.get("budget_violation"))
        self.assertTrue((self.project / "runs/r01/attempts/a001/physical-part-inventory.json").is_file())
        recorded = json.loads(
            (self.project / "runs/r01/attempts/a001/physical-part-inventory.json").read_text(encoding="utf-8"))
        self.assertEqual(16, recorded["machine_part_count"])


if __name__ == "__main__":
    unittest.main()
