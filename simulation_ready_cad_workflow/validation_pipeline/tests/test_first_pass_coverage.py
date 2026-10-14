"""Scheduling tests use synthetic geometry/traces, not mechanical acceptance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError
from ..slim_harness import cli
from ..slim_harness.budget import ProjectBudget, get_tier
from ..slim_harness.test_coverage import FirstPassReserveReached, record_visit, reserve_seconds, ordered_tests
from . import test_slim_workflow as fixtures
AUDIT_IMPLEMENTATION_SHA256=fixtures.AUDIT_IMPLEMENTATION_SHA256
digest=fixtures.digest

PREFIX="simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow."


class FirstPassCoverageTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.SlimWorkflowTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def suite(self,count=3):
        f=self.fixture
        scene=json.loads(f.scene.read_text())
        first=scene['tests'][0]
        scene['tests']=[{**copy.deepcopy(first),'id':f'test_{i}'} for i in range(count)]
        write_json_atomic(f.scene,scene)
        return scene

    def run_suite(self, outcomes, *, cached_failure=False):
        f=self.fixture;scene=self.suite(len(outcomes))
        workflow,old,geometry,request,template=f._partial_physics_fixture(physics_timeout_seconds=300)
        template["test_id"]="test_0"
        request["output"]=str(old/"tests/test_0/physical.trace.json")
        write_json_atomic(old/"tests/test_0/request.json",request)
        write_json_atomic(old/"tests/test_0/physical.trace.json",template)
        if cached_failure:
            template.update(physical_pass=False,failure={'kind':'assertion'})
            write_json_atomic(old/'tests/test_0/physical.trace.json',template)
        else:
            request['test']['purpose']='stale request'
            write_json_atomic(old/'tests/test_0/request.json',request)
        order=[]
        def runner(command,**kwargs):
            request=json.loads(Path(command[-1]).read_text())
            name=request['test']['id'];order.append(('physics',name))
            self.assertEqual(300,kwargs['timeout_seconds'])
            value=outcomes[int(name.split('_')[-1])]
            if value=='timeout':raise BoundedSubprocessError('BOUNDED_SUBPROCESS_TIMEOUT','synthetic timeout')
            if value in {'missing_trace','malformed_trace'}:
                if value=='malformed_trace':Path(request['output']).write_text('invalid json')
                return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
            if value=='error':return SimpleNamespace(returncode=1,stdout=b'',stderr=b'synthetic runtime error')
            trace=copy.deepcopy(template);trace.update(test_id=name,physical_pass=value=='pass',failure=None if value=='pass' else {'kind':'assertion'})
            trace['assertions']=[{'id':'moved','passed':value=='pass'}]
            write_json_atomic(Path(request['output']),trace)
            return SimpleNamespace(returncode=0,stdout=b'',stderr=b'')
        def audit(_geometry,samples,path):
            if path.name=='geometry-audit.json':order.append(('audit',path.parent.name))
            result={'passed':True,'violations':[],'checked_solver_samples':len(samples),
                    'audit_implementation_sha256':AUDIT_IMPLEMENTATION_SHA256}
            write_json_atomic(path,result);return result
        def render(_geometry,path,*args,**kwargs):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text('synthetic preview');return path
        with patch(PREFIX+'prepare_geometry',side_effect=AssertionError('No CAD in scheduling test')),patch(PREFIX+'run_bounded_subprocess',side_effect=runner),              patch(PREFIX+'audit_motion',side_effect=audit),patch(PREFIX+'audit_saved_motion',side_effect=audit),patch(PREFIX+'render_views',side_effect=render),              patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.trial_cache.reuse_trial',return_value=None):
            result=workflow.run_revision()
        return result,order,workflow

    def test_timeout_and_functional_failure_do_not_starve_later_tests(self):
        result,order,workflow=self.run_suite(['timeout','fail','pass'])
        self.assertEqual([('physics','test_0'),('physics','test_1'),('physics','test_2')],order[:3])
        self.assertEqual(('audit','test_2'),order[3])
        self.assertEqual('SIMULATION_TIMEOUT',result['status'])
        self.assertEqual([],result['unrun_tests'])
        self.assertTrue(result['test_coverage']['all_tests_visited'])
        self.assertEqual(3,len(result['tests']))
        self.assertFalse(result['tests'][1]['geometry_audit']['passed'])
        workflow._finish_validation.assert_not_called()

    def test_runtime_error_is_recorded_and_remaining_tests_run(self):
        result,order,_=self.run_suite(['error','pass','pass'])
        self.assertEqual(['test_0','test_1','test_2'],[name for stage,name in order if stage=='physics'])
        self.assertEqual('HARNESS_ERROR',result['status'])
        self.assertTrue(result['tests'][1]['physical_pass'])
        self.assertEqual([],result['unrun_tests'])

    def test_started_runtime_with_invalid_evidence_is_error_and_later_tests_run(self):
        result,order,_=self.run_suite(['missing_trace','malformed_trace','pass'])
        self.assertEqual(['test_0','test_1','test_2'],[name for kind,name in order if kind=='physics'])
        self.assertEqual([],result['unrun_tests'])
        self.assertTrue(result['test_coverage']['all_tests_visited'])
        self.assertTrue(all(row['execution_started'] for row in result['tests']))
        self.assertEqual(['HARNESS_ERROR','HARNESS_ERROR'],[row['execution_status'] for row in result['tests'][:2]])

    def test_recovered_failed_trace_does_not_stop_unvisited_test(self):
        result,order,_=self.run_suite(['fail','pass'],cached_failure=True)
        self.assertEqual([('physics','test_1')],[row for row in order if row[0]=='physics'])
        self.assertEqual('NEEDS_REVISION',result['status'])
        self.assertEqual('reused',result['tests'][0]['execution_status'])
        self.assertEqual([],result['unrun_tests'])

    def test_five_task_round_attempts_every_task_before_any_motion_audit(self):
        result,order,_=self.run_suite(['timeout','error','pass','fail','pass'])
        self.assertEqual([f'test_{i}' for i in range(5)],[name for kind,name in order[:5]])
        self.assertTrue(all(kind=='physics' for kind,_ in order[:5]))
        self.assertEqual([],result['unrun_tests'])
        self.assertEqual(5,len(result['tests']))

    def test_total_budget_stop_retains_exact_unrun_task_list(self):
        from ..slim_harness.budget import ActiveBudgetExhausted
        f=self.fixture;self.suite(2)
        workflow,*_=f._partial_physics_fixture()
        first={'test_id':'test_0','execution_status':'SIMULATION_TIMEOUT','execution_started':True,
               'physical_pass':False,'failure_status':'SIMULATION_TIMEOUT','geometry_audit':{'passed':False}}
        with patch.object(workflow,'_execute_physical_test',side_effect=[first,ActiveBudgetExhausted('total exhausted')]):
            result,_,_=f._exercise_partial_fixture(workflow)
        self.assertEqual('ACTIVE_BUDGET_EXHAUSTED',result['status'])
        self.assertEqual(['test_1'],result['unrun_tests'])
        self.assertEqual(['test_0'],result['test_coverage']['started_test_ids'])
        self.assertFalse(result['test_coverage']['all_tests_visited'])

    def test_reserve_is_for_untested_definitions_and_never_grants_pass(self):
        f=self.fixture;scene=self.suite(5);tier=get_tier('high')
        self.assertEqual(1500,reserve_seconds(f.project,tier))
        record_visit(f.project,scene['tests'][0],'completed',f.run,True)
        record_visit(f.project,scene['tests'][1],'not_run',f.run,False)
        self.assertEqual(1200,reserve_seconds(f.project,tier))
        self.assertEqual('test_1',ordered_tests(f.project,scene['tests'])[0]['id'])
        self.assertFalse((f.run/'result.json').exists())
        scene['tests'][0]['duration_s']=2;write_json_atomic(f.scene,scene)
        self.assertEqual(1500,reserve_seconds(f.project,tier))

    def test_author_cannot_spend_physics_reserve_but_physics_can(self):
        f=self.fixture;scene=self.suite(5)
        budget=ProjectBudget(f.project,'high')
        budget.tracker._active_seconds=9300
        with self.assertRaises(FirstPassReserveReached):budget.author_limit(1800)
        self.assertEqual(300,budget.limit(300))
        self.assertFalse(budget.tracker.is_exhausted)
        for test in scene['tests']:record_visit(f.project,test,'SIMULATION_TIMEOUT',f.run,True)
        self.assertEqual(1500,budget.author_limit(1800))

    def test_development_retry_yields_before_geometry_until_other_tests_are_attempted(self):
        from ..slim_harness.author_tools import run_tool
        f=self.fixture;scene=self.suite(2)
        record_visit(f.project,scene['tests'][0],'SIMULATION_TIMEOUT',f.run,True)
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.author_tools.prepare_geometry',side_effect=AssertionError('Do not redo geometry for a blocked retry')):
            result=run_tool('simulate',f.scene,workspace=f.workspace,test_id='test_0')
        self.assertEqual('FIRST_PASS_PENDING',result['status'])
        self.assertEqual(['test_1'],result['available_tests'])
        self.assertFalse(result['certified'])

    def test_partial_author_json_keeps_reserve_without_claiming_empty_coverage_complete(self):
        from ..slim_harness.test_coverage import coverage_summary
        f=self.fixture
        for value in ([],{'tests':['partial']},{'tests':[]}):
            write_json_atomic(f.scene,value)
            self.assertEqual(1500,reserve_seconds(f.project,get_tier('high')))
        self.assertFalse(coverage_summary([],[],[])['all_tests_visited'])

    def test_partial_author_handoff_cannot_directly_publish_machine_pass(self):
        f=self.fixture;workflow=f.workflow()
        scene=json.loads(f.scene.read_text())
        summary=f._implemented_test_summary();summary.update(assertions=[{'id':'moved','passed':True}],failure=None)
        bindings={str(f.scene):digest(f.scene),str(f.source):digest(f.source)}
        workflow.state['first_pass_handoff']=True
        result=workflow._finish_validation(f.run,scene,{},[],[summary],bindings,f.run,0)
        self.assertEqual('NEEDS_REVISION',result['status'])
        self.assertTrue(result['first_pass_only'])
        workflow.turn('author',lambda directory:{'status':'ready'})
        result=workflow._finish_validation(f.run,scene,{},[],[summary],bindings,f.run,0)
        self.assertEqual('PASS',result['status'])

    def test_repair_reserve_handoff_sets_and_persists_partial_delivery_guard(self):
        f=self.fixture;workflow=f.workflow()
        scene=json.loads(f.scene.read_text())
        summary=f._implemented_test_summary();summary.update(assertions=[{'id':'moved','passed':True}],failure=None)
        bindings={str(f.scene):digest(f.scene),str(f.source):digest(f.source)}
        def execute(*,one_revision):
            if not one_revision:raise FirstPassReserveReached('reserved',session_id='repair-session')
            self.assertTrue(json.loads((f.project/'workflow.json').read_text())['first_pass_handoff'])
            return workflow._finish_validation(f.run,scene,{},[],[summary],bindings,f.run,0)
        with patch.object(workflow,'_run',side_effect=execute):result=workflow.run()
        self.assertEqual('NEEDS_REVISION',result['status'])
        self.assertTrue(result['first_pass_only'])
        self.assertEqual('repair-session',workflow.state['author_session_id'])

    def test_budget_exhaustion_during_handoff_uses_normal_result_handler(self):
        from ..slim_harness.budget import ActiveBudgetExhausted
        from ..slim_harness.construction_budget import BudgetExhausted
        for exception,status in ((ActiveBudgetExhausted('exhausted'),'ACTIVE_BUDGET_EXHAUSTED'),(BudgetExhausted({'exhausted':True,'remaining_seconds':0,'active_seconds':10800}),'BUDGET_EXHAUSTED')):
            with self.subTest(status=status):
                workflow=self.fixture.workflow(tier="high")
                with patch.object(workflow,'_run',side_effect=[FirstPassReserveReached('reserved'),exception]):
                    result=workflow.run()
                self.assertEqual(status,result['status'])
                self.assertTrue(workflow.state['first_pass_handoff'])

    def test_reserved_handoff_evaluates_partial_scene_without_forging_ready(self):
        f=self.fixture
        write_json_atomic(f.project/'criteria.lock.json',{'criteria_sha256':digest(f.project/'criteria.json'),'task_sha256':digest(f.project/'task.md')})
        state=json.loads((f.project/'workflow.json').read_text());state['request']['model']='custom-model';write_json_atomic(f.project/'workflow.json',state)
        with patch.object(cli,'_initialize_project',side_effect=FirstPassReserveReached('reserved')),patch.object(cli,'SlimWorkflow') as workflow:
            workflow.return_value.run.return_value={'status':'NEEDS_REVISION'}
            result=cli.initialize_project(f.workspace,f.project)
        self.assertEqual('NEEDS_REVISION',result['status'])
        self.assertEqual('custom-model',workflow.call_args.kwargs['model'])
        state=json.loads((f.project/'workflow.json').read_text())
        self.assertTrue(state['first_pass_handoff'])
        self.assertFalse(any(row.get('kind')=='author' and row.get('result',{}).get('status')=='ready' for row in state['events']))


if __name__=='__main__':unittest.main()
