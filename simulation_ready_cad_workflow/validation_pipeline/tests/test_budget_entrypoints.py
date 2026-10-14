"""Short entrypoint acceptance; synthetic workers are not mechanical evidence."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from ..mechanism.io import write_json_atomic
from ..slim_harness import cli
from ..slim_harness.budget import PROVIDER_RECOVERY_RETRIES, ProjectBudget, TierConfig, draft_simulation_budget, ActiveBudgetExhausted
from ..slim_harness.workflow import SlimWorkflow, digest
from ..runtime.bounded_subprocess import run_bounded_subprocess, BoundedSubprocessError

PREFIX="simulation_ready_cad_workflow.validation_pipeline.slim_harness."


class FakeWorker:
    session_id="fixture-session"
    def __init__(self, *, fail_criteria=0, fail_author=0, author_status="needs_revision"):
        self.fail_criteria=fail_criteria
        self.fail_author=fail_author
        self.author_status=author_status
        self.calls=[]

    def run_turn(self, workspace, prompt, directory, timeout_seconds, **kwargs):
        directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
        self.calls.append((directory,timeout_seconds))
        time.sleep(min(.025,timeout_seconds))
        if "Create stable acceptance criteria" in prompt:
            if self.fail_criteria:
                self.fail_criteria-=1
                result={"status":"provider_error","summary":"connection temporarily lost",
                        "provider_error":{"kind":"exit","message":"connection temporarily lost"}}
            else:
                result={"status":"ready","summary":"fixture criteria","criteria":[{
                    "id":"move","kind":"function","statement":"Move","source_quote":"move",
                    "verification":"Observe motion"}]}
        else:
            if self.fail_author:
                self.fail_author-=1
                result={"status":"provider_error","summary":"connection temporarily lost",
                        "provider_error":{"kind":"exit","message":"connection temporarily lost"}}
            else:
                result={"status":self.author_status,"summary":"fixture author","issues":[],"evidence_paths":[]}
        write_json_atomic(directory/"response.json",result)
        return result


class BudgetEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name)
        self.project=cli.create_project(self.workspace,"budget_fixture","A slider must move.")
        self.tier=TierConfig("fixture",5,.3,2,0,10)

    def checkpoint(self):
        return json.loads((self.project/"budget_checkpoint.json").read_text())

    def initialize(self,worker):
        with patch(PREFIX+"cli.create_worker",return_value=worker):
            return cli.initialize_project(self.workspace,self.project,tier=self.tier)

    def test_initial_worker_retry_and_cli_resume_accumulate_without_wait_charge(self):
        worker=FakeWorker(fail_author=1)
        with patch(PREFIX+"budget.recovery_delay",return_value=.12):
            first=self.initialize(worker)
        checkpoint=self.checkpoint()
        self.assertEqual("needs_revision",first["status"])
        self.assertGreater(checkpoint["active_seconds"],.04)
        self.assertGreater(checkpoint["wall_seconds"]-checkpoint["active_seconds"],.1)
        self.assertEqual(2,len(worker.calls))
        self.assertTrue((worker.calls[0][0]/"response.json").is_file())
        self.assertTrue((worker.calls[1][0]/"response.json").is_file())
        time.sleep(.08)
        with patch(PREFIX+"cli.create_worker",return_value=FakeWorker()),redirect_stdout(io.StringIO()) as output:
            exit_code=cli.main(["--workspace",str(self.workspace),"resume","budget_fixture"])
        resumed=json.loads(output.getvalue())
        second=self.checkpoint()
        self.assertEqual(2,exit_code)
        self.assertGreater(second["active_seconds"],checkpoint["active_seconds"]+.015)
        self.assertGreater(second["wall_seconds"],checkpoint["wall_seconds"]+.09)
        self.assertEqual(checkpoint["started_at"],second["started_at"])
        self.assertIn("upper_estimate",resumed["budget_timing"]["accounting_basis"])
        self.assertEqual("stopped",json.loads((self.project/"workflow_runtime.json").read_text())["status"])
        with self.assertRaisesRegex(ValueError,"tier is frozen"):
            ProjectBudget(self.project,"high")

    def test_recovery_is_finite_and_later_cli_can_retry(self):
        with patch(PREFIX+"budget.recovery_delay",return_value=.001):
            failed=FakeWorker(fail_author=PROVIDER_RECOVERY_RETRIES+5)
            result=self.initialize(failed)
        self.assertEqual(PROVIDER_RECOVERY_RETRIES+1,len(failed.calls))
        self.assertEqual("provider_error",result["status"])
        self.assertEqual(PROVIDER_RECOVERY_RETRIES,result["recovery_attempts"])
        self.assertTrue(result["recoverable"])
        with patch(PREFIX+"cli.create_worker",return_value=FakeWorker()):
            result=cli.resume_project(self.workspace,self.project)
        self.assertEqual("needs_revision",result["status"])

    def test_exhausted_initialization_normal_cli_closeout_and_no_worker(self):
        budget=ProjectBudget(self.project,self.tier)
        budget.tracker._active_seconds=self.tier.active_budget_seconds
        budget.save()
        with patch(PREFIX+"cli.create_worker") as worker,redirect_stdout(io.StringIO()) as output:
            exit_code=cli.main(["--workspace",str(self.workspace),"resume","budget_fixture"])
        self.assertEqual(0,exit_code)
        self.assertEqual("ACTIVE_BUDGET_EXHAUSTED",json.loads(output.getvalue())["status"])
        worker.assert_not_called()
        result=json.loads((self.project/"runs/r01/result.json").read_text())
        self.assertFalse(result["completion"]["whole_machine_pass"])
        self.assertTrue(result["blockers"])

    def seed_scene(self):
        self.initialize(FakeWorker())
        design=self.project/"runs/r01/design"
        (design/"part.step").write_text("synthetic fixture; no geometry claim")
        write_json_atomic(design/"parts.json",{"description":"fixture",
            "parts":[{"name":"body","body":"body","role":"dynamic","source_solid_count":1,
                      "source_members":["body:solid_1"]}],
            "body_sources":{"body":"part.step"}})
        scene={"schema_version":"slim_scene_v1","units":"mm",
               "bodies":[{"id":"body","source":"part.step","fixed":False}],
               "authoring_sources":["parts.json"],
               "tests":[{"id":"motion","purpose":"Fixture","duration_s":1,
                         "assertions":[{"id":"a","purpose":"Fixture","expression":"last(signal('time_s')) >= 1"}]}]}
        write_json_atomic(design/"scene.json",scene)
        state=json.loads((self.project/"workflow.json").read_text())
        state.update(phase="geometry")
        state["request"]["delivery_mode"]="submit"
        write_json_atomic(self.project/"workflow.json",state)
        return scene

    def test_formal_timeout_kills_only_child_preserves_prefix_and_pending_test(self):
        scene=self.seed_scene()
        source=self.project/"runs/r01/design/part.step"
        geometry={"errors":[],"bodies":[{"id":"body","source_file":str(source),"source_sha256":digest(source),
                    "position_m":[0,0,0],"rotation_wxyz":[1,0,0,0]}]}
        prefix=self.workspace/"prefix.txt"
        unrelated=subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"])
        def cleanup():
            if unrelated.poll() is None:unrelated.kill()
            unrelated.wait(timeout=2)
        self.addCleanup(cleanup)
        def bounded(command,**kwargs):
            code="import pathlib,time; pathlib.Path("+repr(str(prefix))+").write_text('prefix'); time.sleep(20)"
            return run_bounded_subprocess([sys.executable,"-c",code],**kwargs)
        with ExitStack() as stack:
            for name,value in {"prepare_geometry":geometry,"compile_scene":scene,"audit_motion":{"passed":True,"violations":[]},
                               "render_views":self.workspace/"fixture.png"}.items():
                stack.enter_context(patch(PREFIX+"workflow."+name,return_value=value))
            stack.enter_context(patch.object(SlimWorkflow,"_preview"))
            stack.enter_context(patch(PREFIX+"workflow.run_bounded_subprocess",side_effect=bounded))
            workflow=SlimWorkflow(self.workspace,self.project)
            result=workflow.run(one_revision=True)
        self.assertEqual("SIMULATION_TIMEOUT",result["status"],result)
        self.assertEqual(["motion"],result["pending_tests"])
        self.assertEqual("prefix",prefix.read_text())
        self.assertIsNone(unrelated.poll())
        self.assertGreater(self.checkpoint()["active_seconds"],.3)
        self.assertEqual("stopped",json.loads((self.project/"workflow_runtime.json").read_text())["status"])

    def test_project_remaining_budget_clamps_formal_simulation(self):
        self.seed_scene()
        workflow=SlimWorkflow(self.workspace,self.project)
        workflow.budget_tracker._active_seconds=4.99
        seen=[]
        def revision():
            seen.append(workflow.budget.limit(60))
            with workflow.budget.active("physical_simulation"):
                time.sleep(.02)
            return {"status":"ACTIVE_BUDGET_EXHAUSTED","issues":[],"completed_tests":[],"pending_tests":["motion"]}
        with patch.object(workflow,"run_revision",side_effect=revision):
            result=workflow.run(one_revision=True)
        self.assertLess(seen[0],.011)
        self.assertEqual("ACTIVE_BUDGET_EXHAUSTED",result["status"])
        self.assertEqual(["motion"],result["pending_tests"])

    def test_interruption_records_lifecycle_and_checkpoint(self):
        self.seed_scene()
        workflow=SlimWorkflow(self.workspace,self.project)
        with patch.object(workflow,"run_revision",side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):workflow.run()
        self.assertTrue(self.checkpoint()["paused"])
        self.assertEqual("interrupted",json.loads((self.project/"workflow.json").read_text())["phase"])
        self.assertEqual("stopped",json.loads((self.project/"workflow_runtime.json").read_text())["status"])

    def test_exhaustion_after_physics_keeps_validated_test_summary(self):
        self.seed_scene()
        run=self.project/"runs/r01"
        scene_path=run/"design/scene.json"
        bundle={"criteria_sha256":digest(self.project/"criteria.json"),
                "source_bindings":{str(scene_path):digest(scene_path)},
                "tests":[{"test_id":"motion","physical_pass":True,"geometry_audit":{"passed":True}}]}
        write_json_atomic(run/"attempts/a001/validated-physics.json",bundle)
        budget=ProjectBudget(self.project)
        result=budget.incomplete(issues=["Project budget exhausted after physics"])
        self.assertEqual("motion",result["completed_tests"][0]["id"])
        self.assertEqual([],result["pending_tests"])
        self.assertFalse(result["completion"]["whole_machine_pass"])
        self.assertTrue((run/"budget-summary.md").is_file())

    def test_draft_simulation_standalone_counts_once_and_nested_shares_parent(self):
        self.initialize(FakeWorker())
        before=self.checkpoint()["active_seconds"]
        with draft_simulation_budget(self.project) as limit:
            self.assertEqual(.3,limit)
            time.sleep(.02)
        after=self.checkpoint()["active_seconds"]
        self.assertGreater(after,before+.015)
        import psutil
        write_json_atomic(self.project/"workflow_runtime.json",{"status":"running","pid":os.getpid(),
                          "process_started_at_epoch":psutil.Process().create_time()})
        checkpoint=(self.project/"budget_checkpoint.json").read_bytes()
        with draft_simulation_budget(self.project) as limit:
            time.sleep(.02)
            self.assertEqual(.3,limit)
        self.assertEqual(checkpoint,(self.project/"budget_checkpoint.json").read_bytes())

    def test_standalone_draft_timeout_at_remaining_budget_closes_project(self):
        self.seed_scene()
        budget=ProjectBudget(self.project)
        budget.tracker._active_seconds=4.99
        budget.save()
        with self.assertRaises(ActiveBudgetExhausted):
            with draft_simulation_budget(self.project) as limit:
                run_bounded_subprocess([sys.executable,"-c","import time; time.sleep(20)"],timeout_seconds=limit)
        result=json.loads((self.project/"runs/r01/result.json").read_text())
        self.assertEqual("ACTIVE_BUDGET_EXHAUSTED",result["status"])
        self.assertTrue((self.project/"runs/r01/budget-summary.md").is_file())

    def test_author_backoff_interruption_resumes_author_before_validation(self):
        self.seed_scene()
        state=json.loads((self.project/"workflow.json").read_text())
        state.update(phase="author_followup")
        write_json_atomic(self.project/"workflow.json",state)
        worker=FakeWorker()
        sleep=time.sleep
        def interrupt_wait(seconds):
            if workflow.state.get("phase")=="recovery_wait":raise KeyboardInterrupt
            sleep(seconds)
        with patch(PREFIX+"workflow.create_worker",return_value=worker):
            workflow=SlimWorkflow(self.workspace,self.project)
        error={"status":"provider_error","provider_error":{"kind":"timeout","message":"timed out"}}
        with patch.object(worker,"run_turn",return_value=error),patch(PREFIX+"budget.recovery_delay",return_value=.12),patch(PREFIX+"budget.time.sleep",side_effect=interrupt_wait):
            with self.assertRaises(KeyboardInterrupt):workflow.run()
        state=json.loads((self.project/"workflow.json").read_text())
        self.assertEqual("author",state["recovery_from_phase"])
        self.assertEqual("recovery_wait",state["interrupted_phase"])
        ready=FakeWorker(author_status="ready")
        with patch(PREFIX+"workflow.create_worker",return_value=ready):
            resumed=SlimWorkflow(self.workspace,self.project)
        with patch.object(resumed,"run_revision",return_value={"status":"SIMULATION_TIMEOUT","issues":[]}):
            resumed.run(one_revision=True)
        self.assertEqual(1,len(ready.calls))


if __name__=="__main__":unittest.main()
