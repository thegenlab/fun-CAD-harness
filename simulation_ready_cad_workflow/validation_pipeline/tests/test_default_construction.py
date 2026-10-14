"""Small integration checks for default construction and diagnostic cost probes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ..slim_harness import cli
from ..slim_harness.budget import ProjectBudget as OrdinaryBudget
from ..slim_harness.construction import budget_for, author_turn_timeout, policy_prompt
from ..slim_harness.cost_probe import measure_cost
from ..mechanism.io import write_json_atomic

PREFIX="simulation_ready_cad_workflow.validation_pipeline.slim_harness."

class DefaultConstructionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name)

    def create(self):
        with patch(PREFIX+"construction_supervisor.supervise_project",return_value={"status":"RUNNING"}) as supervisor:
            cli.start(self.workspace,"new","A moving machine.")
        supervisor.assert_called_once()
        return self.workspace/".runtime_runs/projects/new"

    def test_default_create_uses_one_high_budget_and_continuous_author(self):
        project=self.create();budget=budget_for(project)
        snap=budget.snapshot()
        self.assertEqual((10800,300,5),(snap["total_seconds"],snap["config"]["simulation_timeout_seconds"],snap["config"]["max_tests"]))
        self.assertEqual(9180,author_turn_timeout(project))
        with OrdinaryBudget(project) as ordinary:
            self.assertIsNone(ordinary.tracker)
            self.assertEqual("high",ordinary.tier.name)
            self.assertEqual(20,ordinary.tier.max_parts)
            result=ordinary.annotate({})
        self.assertFalse((project/"budget_checkpoint.json").exists())
        self.assertEqual(10800,result["budget_timing"]["budget_limit_seconds"])
        prompt=policy_prompt(project)
        for fragment in ("mandatory core", "probe --test", "no relative motion", "unchanged timed-out", "300"):
            self.assertIn(fragment,prompt)

    def test_resume_restores_high_without_restarting_clock(self):
        project=self.create();budget=budget_for(project)
        budget._transaction(lambda state,now:state.update(active_seconds=123))
        with patch(PREFIX+"construction_supervisor.supervise_project",return_value={"status":"RUNNING"}) as supervisor:
            cli.resume_project(self.workspace,project)
        supervisor.assert_called_once()
        self.assertEqual(123,budget.snapshot()["active_seconds"])
        with self.assertRaisesRegex(ValueError,"frozen"):OrdinaryBudget(project,"light")

    def test_probe_records_cost_without_changing_test_or_acceptance(self):
        test={"id":"motion","duration_s":5,"assertions":[{"id":"whole","expression":"last(signal('time_s')) >= 5"}]}
        original=copy.deepcopy(test);folder=self.workspace/"probe"
        def execute(command,**kwargs):
            request=json.loads(Path(command[-1]).read_text())
            self.assertEqual(.1,request["test"]["duration_s"])
            self.assertEqual(test["assertions"],request["test"]["assertions"])
            self.assertEqual(15,kwargs["timeout_seconds"])
            write_json_atomic(Path(request["output"]),{"samples":[{"time_s":0},{"time_s":.1}],"signals":{"time_s":[0,.1]}})
            return SimpleNamespace(returncode=0,stdout=b"",stderr=b"")
        with patch(PREFIX+"cost_probe.run_bounded_subprocess",side_effect=execute):
            result=measure_cost({"tests":[test]},test,folder,python="python",workspace=self.workspace,remaining_seconds=1000,simulation_timeout=300)
        self.assertEqual(original,test)
        self.assertFalse(result["certified"])
        self.assertFalse(result["counts_as_test_execution"])
        self.assertGreater(result["estimated_full_wall_seconds"],0)
        self.assertFalse((self.workspace/"first-pass-coverage.json").exists())
        self.assertFalse((folder/"geometry-audit.json").exists())

    def test_registered_probe_keeps_exact_test_and_passes_actual_binding_check(self):
        from .test_slim_measurement_registry import MeasurementRegistryTests
        from ..slim_harness.measurement_registry import compiled_test_spec,validate_runtime_registration
        fixture=MeasurementRegistryTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        test=compiled_test_spec(fixture.test,"mm")
        prepared={"registered_measurements":{test["id"]:fixture.binding}}
        def execute(command,**kwargs):
            request=json.loads(Path(command[-1]).read_text())
            validate_runtime_registration(request["scene"]["registered_measurements"][test["id"]],request["test"])
            self.assertEqual(test,request["test"])
            self.assertEqual(15,kwargs["timeout_seconds"])
            write_json_atomic(Path(request["output"]),{"samples":[{"time_s":0},{"time_s":.05}],"signals":{"time_s":[0,.05]}})
            return SimpleNamespace(returncode=0,stdout=b"",stderr=b"")
        with patch(PREFIX+"cost_probe.run_bounded_subprocess",side_effect=execute):
            result=measure_cost(prepared,test,self.workspace/"registered",python="python",workspace=self.workspace,remaining_seconds=1000,simulation_timeout=300)
        self.assertTrue(result["registered_test_unchanged"])
        self.assertEqual(.05,result["observed_simulated_seconds"])
        self.assertFalse(result["counts_as_test_execution"])

    def test_author_monitor_reserves_newly_added_tests_before_next_poll(self):
        import subprocess
        from unittest.mock import Mock
        from ..slim_harness.construction_worker import communicate_budgeted
        project=self.create();budget=budget_for(project);scene=project/"runs/r01/design/scene.json"
        write_json_atomic(scene,{"tests":[{"id":"a"}]})
        budget._transaction(lambda state,now:state.update(active_seconds=8800))
        events=project/"worker_turns/t001_author/events.jsonl";events.parent.mkdir(parents=True)
        process=Mock(args=["fixture"])
        def communicate(**kwargs):
            write_json_atomic(scene,{"tests":[{"id":str(i)} for i in range(5)]})
            budget._transaction(lambda state,now:state.update(active_seconds=9300))
            raise subprocess.TimeoutExpired(process.args,.5)
        process.communicate.side_effect=communicate
        with self.assertRaises(subprocess.TimeoutExpired):communicate_budgeted(process,"prompt",1580,project,events)
        self.assertEqual(1,process.communicate.call_count)

    def test_formal_timeout_continues_into_author_repair_with_remaining_budget(self):
        from .test_slim_workflow import SlimWorkflowTests
        from ..slim_harness.construction_budget import initialize_budget,BudgetConfig
        fixture=SlimWorkflowTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        state=json.loads((fixture.project/"workflow.json").read_text());state["request"].update(construction=True,tier="high")
        write_json_atomic(fixture.project/"workflow.json",state)
        initialize_budget(fixture.project,BudgetConfig(hours=3,simulation_timeout_seconds=300,max_tests=5,max_tasks=5))
        workflow=fixture.workflow()
        with patch.object(workflow,"run_revision",side_effect=[{"status":"SIMULATION_TIMEOUT","issues":["fixture timeout"]},{"status":"PASS"}]) as run, \
             patch.object(workflow.author,"run_turn",return_value={"status":"ready"}) as author:
            result=workflow.run()
        self.assertEqual("PASS",result["status"])
        self.assertEqual(2,run.call_count)
        self.assertGreater(author.call_args.args[3],1800)
        self.assertIn("simplify simulation",author.call_args.args[1])
        self.assertTrue((fixture.project/"runs/r01/result.json").is_file())


    def test_adjustable_author_turn_uses_construction_time_not_1800(self):
        from .test_slim_workflow import SlimWorkflowTests
        from ..slim_harness.construction_budget import initialize_budget,BudgetConfig
        fixture=SlimWorkflowTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        state=json.loads((fixture.project/"workflow.json").read_text());state["request"].update(construction=True,tier="high",task_adjustable=True)
        write_json_atomic(fixture.project/"workflow.json",state)
        initialize_budget(fixture.project,BudgetConfig(hours=3,simulation_timeout_seconds=300,max_tests=5,max_tasks=5))
        workflow=fixture.workflow()
        response={"status":"ready","budget_decision":{"action":"finish_partial","reason":"Preserve unfinished fixture.","tradeoff":"No function acceptance claim."}}
        with patch.object(workflow.author,"run_turn",return_value=response) as author:
            result=workflow._budget_revision(fixture.scene.parent,{"status":"SIMULATION_TIMEOUT","issues":["fixture"]})
        self.assertEqual("INCOMPLETE_EVIDENCE",result["status"])
        self.assertGreater(author.call_args.args[3],1800)
        self.assertIn("probe --test",author.call_args.args[1])


    def test_exhausted_construction_retains_per_test_evidence_without_tier_tracker(self):
        from .test_slim_workflow import SlimWorkflowTests
        from ..slim_harness.construction_budget import initialize_budget,BudgetConfig
        fixture=SlimWorkflowTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        state=json.loads((fixture.project/"workflow.json").read_text());state["request"].update(construction=True,tier="high")
        write_json_atomic(fixture.project/"workflow.json",state)
        initialize_budget(fixture.project,BudgetConfig(hours=3))
        workflow=fixture.workflow()
        failed={"test_id":"move","execution_started":True,"execution_status":"SIMULATION_TIMEOUT","physical_pass":False}
        with patch.object(workflow,"run_revision",return_value={"status":"ACTIVE_BUDGET_EXHAUSTED","completed_tests":[failed]}):
            result=workflow.run()
        self.assertEqual("BUDGET_EXHAUSTED",result["status"])
        self.assertEqual([failed],result["completed_tests"])
        self.assertEqual([],result["unrun_tests"])
        self.assertEqual("budget_exhausted",workflow.state["phase"])
        self.assertEqual(result["status"],json.loads((fixture.run/"result.json").read_text())["status"])


    def test_construction_closeout_accepts_saved_public_id_rows(self):
        project=self.create()
        write_json_atomic(project/"runs/r01/design/scene.json",{"tests":[{"id":"old"}]})
        write_json_atomic(project/"runs/r01/result.json",{"completed_tests":[{"id":"old","execution_started":True,"execution_status":"SIMULATION_TIMEOUT","physical_pass":False}]})
        result=OrdinaryBudget(project).incomplete(issues=["exhausted"])
        self.assertEqual("old",result["completed_tests"][0]["test_id"])
        self.assertEqual([],result["unrun_tests"])
        self.assertFalse(result["eligible_pass"])


if __name__=="__main__":unittest.main()
