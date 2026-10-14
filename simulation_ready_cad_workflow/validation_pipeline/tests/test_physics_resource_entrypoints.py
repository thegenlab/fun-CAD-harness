"""Actual entrypoint plumbing with synthetic geometry and scoped child fixtures."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from . import test_budget_entrypoints as fixtures
from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError, BoundedSubprocessResult
from ..cold_start.worker import CodexCliWorker
from ..slim_harness.author_tools import run_tool
from ..slim_harness.budget import ProjectBudget
from ..slim_harness.workflow import SlimWorkflow, digest
from ..slim_harness.physics_invocation import run_physics_attempts
from ..slim_harness.trace_storage import TraceWriter


PREFIX=fixtures.PREFIX


class PhysicsResourceEntrypointTests(unittest.TestCase):
    setUp=fixtures.BudgetEntrypointTests.setUp
    initialize=fixtures.BudgetEntrypointTests.initialize
    seed_scene=fixtures.BudgetEntrypointTests.seed_scene
    checkpoint=fixtures.BudgetEntrypointTests.checkpoint
    def geometry(self):
        source=self.project/"runs/r01/design/part.step"
        return {"errors":[],"bodies":[{"id":"body","source_file":str(source),"source_sha256":digest(source),
                "position_m":[0,0,0],"rotation_wxyz":[1,0,0,0]}]}

    def declare_resources(self):
        write_json_atomic(self.project/"resource_policy.json",{"required_available_memory_gib":.1,
            "max_resource_wait_seconds":.4,"resource_poll_seconds":.01,"max_process_tree_memory_gib":5})

    def test_actual_draft_admission_wait_is_not_active_or_work(self):
        scene=self.seed_scene();self.declare_resources()
        before=json.loads((self.project/"budget_checkpoint.json").read_text())["active_seconds"]
        def wait(*a,**kw):
            time.sleep(.06)
            return {"status":"wait_exhausted","waited_seconds":.06}
        with patch(PREFIX+"author_tools.prepare_geometry",return_value=self.geometry()),patch(PREFIX+"author_tools.compile_scene",return_value=scene),patch(PREFIX+"author_tools.audit_motion",return_value={"passed":True}),patch(PREFIX+"physics_invocation.wait_for_observed_resources",side_effect=wait):
            result=run_tool("simulate",self.project/"runs/r01/design/scene.json",workspace=self.workspace)
        checkpoint=json.loads((self.project/"budget_checkpoint.json").read_text())
        self.assertEqual("RESOURCE_POLICY_REQUIRED",result["status"],result)
        self.assertLess(checkpoint["active_seconds"]-before,.015)
        self.assertEqual(0,checkpoint["observed_work_seconds"])
        self.assertGreater(checkpoint["observed_wait_seconds"],.05)

    def test_formal_rss_retry_recomputes_deadline_and_keeps_audit_next_to_trace(self):
        scene=self.seed_scene();self.declare_resources()
        geometry=self.geometry();seen=[]
        def runner(command,**kwargs):
            request=json.loads(Path(command[-1]).read_text())
            seen.append({"limit":kwargs["timeout_seconds"],"remaining":workflow.budget.tracker.remaining_active,"rss":kwargs["max_rss_bytes"]})
            if len(seen)==1:
                time.sleep(.26)
                raise BoundedSubprocessError("BOUNDED_SUBPROCESS_RSS","fixture RSS")
            trace={"test_id":"motion","physical_pass":True,"samples":[{"time_s":0,"body_poses":{}}],"signals":{"time_s":[0]}}
            write_json_atomic(Path(request["output"]),trace)
            return BoundedSubprocessResult(0,b"completed",b"")
        def audit(scene,samples,path):
            result={"passed":True,"violations":[]}
            write_json_atomic(path,result)
            return result
        with ExitStack() as stack:
            for name,value in {"prepare_geometry":geometry,"compile_scene":scene,"render_views":self.workspace/"fixture.png"}.items():stack.enter_context(patch(PREFIX+"workflow."+name,return_value=value))
            stack.enter_context(patch(PREFIX+"workflow.audit_motion",side_effect=audit))
            stack.enter_context(patch(PREFIX+"workflow.audit_saved_motion",side_effect=audit))
            stack.enter_context(patch(PREFIX+"physical_parts.resolve_physical_part_inventory",return_value={"status":"ready","count":1,"machine_part_count":1}))
            stack.enter_context(patch.object(SlimWorkflow,"_preview"))
            stack.enter_context(patch.object(SlimWorkflow,"_motion",return_value=({},{})))
            finalizer=stack.enter_context(patch.object(SlimWorkflow,"_finish_validation",return_value={"status":"NEEDS_REVISION","issues":["Synthetic driver fixture; no mechanical acceptance"]}))
            stack.enter_context(patch(PREFIX+"physics_invocation.wait_for_observed_resources",return_value={"status":"ready","waited_seconds":0}))
            stack.enter_context(patch(PREFIX+"workflow.run_bounded_subprocess",side_effect=runner))
            workflow=SlimWorkflow(self.workspace,self.project);workflow.budget_tracker._active_seconds=4.5
            result=workflow.run(one_revision=True)
        self.assertEqual("NEEDS_REVISION",result["status"],result)
        finalizer.assert_called_once()
        self.assertEqual(2,len(seen));self.assertLess(seen[1]["limit"],seen[0]["limit"])
        self.assertLessEqual(seen[1]["limit"],seen[1]["remaining"]+.02)
        self.assertEqual(5*1024**3,seen[1]["rss"])
        folder=self.project/"runs/r01/attempts/a001/tests/motion"
        self.assertTrue((folder/"execution-failure.json").is_file())
        self.assertTrue((folder/"retry-1/geometry-audit.json").is_file())
        self.assertTrue((folder.parent.parent/"validated-physics.json").is_file())

    def test_draft_uses_same_finite_rss_retries(self):
        scene=self.seed_scene();self.declare_resources();calls=[]
        def runner(command,**kwargs):
            calls.append(command[-1])
            raise BoundedSubprocessError("BOUNDED_SUBPROCESS_RSS","fixture RSS")
        with patch(PREFIX+"author_tools.prepare_geometry",return_value=self.geometry()),patch(PREFIX+"author_tools.compile_scene",return_value=scene),patch(PREFIX+"author_tools.audit_motion",return_value={"passed":True}),patch(PREFIX+"physics_invocation.wait_for_observed_resources",return_value={"status":"ready","waited_seconds":0}),patch(PREFIX+"author_tools.run_bounded_subprocess",side_effect=runner):
            result=run_tool("simulate",self.project/"runs/r01/design/scene.json",workspace=self.workspace)
        self.assertEqual("RESOURCE_POLICY_REQUIRED",result["status"],result)
        self.assertEqual(4,len(calls));self.assertEqual(4,len(set(calls)))
        self.assertTrue(all((Path(path).parent/"execution-failure.json").is_file() for path in calls))

    def test_nested_real_child_wait_does_not_trigger_old_fixed_wall_deadline(self):
        budget=ProjectBudget(self.project,self.tier)
        worker=budget.worker(CodexCliWorker())
        child=self.workspace/"child.py"
        child.write_text('''import json,sys,time
from pathlib import Path
project,response=map(Path,sys.argv[1:])
directory=Path(json.loads((project/'budget_checkpoint.json').read_text())['worker_wait_directory'])
sys.stdin.read()
path=directory/'nested.json'
row={'activity':'resource_admission','started_monotonic':time.monotonic()}
def save():
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(row));temporary.replace(path)
save();time.sleep(.45);row['ended_monotonic']=time.monotonic();save()
response.write_text(json.dumps({'status':'ready','summary':'fixture','issues':[],'evidence_paths':[]}))
print(json.dumps({'type':'thread.started','thread_id':'fixture'}))
print(json.dumps({'type':'turn.completed','usage':{}}))
''',encoding="utf-8")
        popen=subprocess.Popen
        def start(command,**kwargs):
            response=command[command.index("-o")+1]
            return popen([sys.executable,str(child),str(self.project),response],**kwargs)
        with budget,patch("simulation_ready_cad_workflow.validation_pipeline.cold_start.worker.subprocess.Popen",side_effect=start):
            result=worker.run_turn(self.workspace,"fixture",self.workspace/"turn",.3)
        self.assertEqual("ready",result["status"],result)
        checkpoint=json.loads((self.project/"budget_checkpoint.json").read_text())
        self.assertGreater(checkpoint["observed_wait_seconds"],.4)
        self.assertLess(checkpoint["active_seconds"],.3)
        self.assertGreater(checkpoint["unclassified_provider_seconds"],0)
        self.assertLess(checkpoint["unclassified_provider_seconds"],.3)

    def test_timeout_result_exposes_only_reader_verified_prefix(self):
        scene=self.seed_scene();budget=ProjectBudget(self.project)
        folder=self.workspace/"prefix-call";folder.mkdir()
        def runner(command,**kwargs):
            request=json.loads(Path(command[-1]).read_text());trace=Path(request["output"])
            writer=TraceWriter(trace,10,chunk_rows=1)
            writer.append({"time_s":0},{"time_s":0})
            writer.handles.close()
            raise BoundedSubprocessError("BOUNDED_SUBPROCESS_TIMEOUT","fixture")
        with budget:
            result=run_physics_attempts(workspace=self.workspace,project=self.project,folder=folder,scene=scene,test=scene["tests"][0],chrono_python=sys.executable,budget=budget,timeout_seconds=.3,runner=runner)
        self.assertEqual("SIMULATION_TIMEOUT",result["status"])
        self.assertEqual(1,result["readable_prefixes"][0]["confirmed_samples"])
        self.assertEqual("UNVERIFIED_PREFIX",result["readable_prefixes"][0]["status"])
