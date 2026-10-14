"""Current code-only acceptance, with no model preflight or final review."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock,patch

from ..mechanism.io import write_json_atomic
from ..slim_harness.contracts import DeliveryError
from ..slim_harness.publication import CODE_ACCEPTANCE_POLICY,LEGACY_ACCEPTANCE_POLICY,code_gate_failures,read_result
from ..slim_harness.workflow import SlimWorkflow,digest


class CodePolicyTests(unittest.TestCase):
    def fixture(self):
        from .test_slim_workflow import SlimWorkflowTests
        helper=SlimWorkflowTests('test_missing_or_partial_scene_is_delivery_feedback')
        helper.setUp();self.addCleanup(helper.doCleanups)
        workflow,old,geometry,request,trace=helper._partial_physics_fixture()
        workflow.author.run_turn=Mock(side_effect=AssertionError('No author model in unchanged input fixture'))
        workflow.reviewer.run_turn=Mock(side_effect=AssertionError('No model reviewer in current run'))
        workflow._finish_validation=SlimWorkflow._finish_validation.__get__(workflow,SlimWorkflow)
        return helper,workflow,old,geometry,request,trace

    def test_code_success_runs_without_model_reviewer_or_fake_review_reports(self):
        f,workflow,*_=self.fixture()
        result,native,auditor=f._exercise_partial_fixture(workflow)
        self.assertEqual('PASS',result['status'])
        self.assertEqual(CODE_ACCEPTANCE_POLICY,result['acceptance_policy'])
        self.assertNotIn('review',result)
        workflow.reviewer.run_turn.assert_not_called();workflow.author.run_turn.assert_not_called()
        self.assertEqual(2,auditor.call_count);native.assert_not_called()
        attempt=Path(result['attempt'])
        self.assertFalse((attempt/'preflight_review.json').exists())
        self.assertFalse((attempt/'final_review.json').exists())
        bundle=json.loads((attempt/'validated-physics.json').read_text())
        self.assertNotIn('preflight',bundle)
        self.assertEqual(CODE_ACCEPTANCE_POLICY,json.loads((attempt/'acceptance-policy.json').read_text())['policy'])

    def test_original_motion_audit_failure_still_cannot_pass(self):
        f,workflow,*_=self.fixture()
        result,native,_=f._exercise_partial_fixture(workflow,reject_motion=True)
        self.assertEqual('NEEDS_REVISION',result['status'])
        self.assertFalse(result['issues'][0]['geometry_audit']['passed'])
        workflow.reviewer.run_turn.assert_not_called();native.assert_not_called()

    def test_task_baseline_hook_precedes_compile_failure_without_overriding_it(self):
        f,workflow,*_=self.fixture()
        workflow._task_change_advice=Mock(return_value=None)
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow.compile_scene',side_effect=DeliveryError('Invalid source connection')):
            result,native,_=f._exercise_partial_fixture(workflow)
        self.assertEqual('NEEDS_REVISION',result['status'])
        workflow._task_change_advice.assert_called_once()
        native.assert_not_called();workflow.reviewer.run_turn.assert_not_called()

    def test_task_change_response_goes_back_without_entering_physics(self):
        f,workflow,*_=self.fixture()
        advice={'status':'TASK_CHANGE_REVISION_NEEDED','issues':['Author elected to revise changed tasks']}
        workflow._task_change_advice=Mock(return_value=advice)
        result,native,auditor=f._exercise_partial_fixture(workflow)
        self.assertEqual(advice,result);native.assert_not_called();auditor.assert_not_called()

    def test_initial_tasks_are_remembered_even_when_a_quantity_gate_rejects_design(self):
        from ..slim_harness.budget import get_tier
        f,workflow,*_=self.fixture()
        workflow.tier=get_tier('light')
        workflow._task_change_advice=Mock(side_effect=AssertionError('Hard count failure needs no model advice'))
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.budget.check_task_count',return_value=['Too many declared tasks']):
            result,_,_=f._exercise_partial_fixture(workflow)
        self.assertEqual('task_count',result['budget_violation'])
        baseline=json.loads((workflow.project/'simulation-task-baseline.json').read_text())
        self.assertEqual(json.loads((f.run/'design/scene.json').read_text())['tests'],baseline['tests'])
        workflow._task_change_advice.assert_not_called()

    def test_code_gate_requires_all_tests_and_preserves_physical_assertion_failures(self):
        scene={'tests':[{'id':'first'},{'id':'second'}]}
        row={'test_id':'first','physical_pass':True,'assertions':[{'passed':True}],'geometry_audit':{'passed':True}}
        self.assertTrue(code_gate_failures(scene,[row]))
        second={**copy.deepcopy(row),'test_id':'second'}
        self.assertEqual([],code_gate_failures(scene,[row,second]))
        second['assertions'][0]['passed']=False
        self.assertTrue(code_gate_failures(scene,[row,second]))

    def test_new_policy_publication_needs_code_gates_but_no_model_review(self):
        f,workflow,*_=self.fixture();result,_,_=f._exercise_partial_fixture(workflow)
        payload={**result,'schema_version':'slim_harness_result_v1','revision':1}
        path=f.run/'result.json';write_json_atomic(path,payload)
        self.assertTrue(read_result(f.run)['eligible_pass'])
        payload['tests'][0]['assertions'][0]['passed']=False;write_json_atomic(path,payload)
        self.assertFalse(read_result(f.run)['eligible_pass'])

    def test_legacy_pass_keeps_legacy_meaning_and_file_bytes(self):
        f,workflow,*_=self.fixture();result,_,_=f._exercise_partial_fixture(workflow)
        result.pop('acceptance_policy')
        result.update(schema_version='slim_harness_result_v1',revision=1,review={
            'status':'ready','observations':[
                {'criterion_id':key,'verdict':'met','observed_fact':'Historical review fixture','evidence_paths':['historical.png']}
                for key in ('appearance','function')],
            'test_coverage':[{'criterion_id':'function','appropriate':True,'test_ids':['move']}]})
        path=f.run/'result.json';write_json_atomic(path,result);before=path.read_bytes()
        old=read_result(f.run)
        self.assertTrue(old['eligible_pass']);self.assertEqual(LEGACY_ACCEPTANCE_POLICY,old['acceptance_policy'])
        self.assertEqual(before,path.read_bytes())
        result.pop('review');write_json_atomic(path,result)
        self.assertFalse(read_result(f.run)['eligible_pass'])


if __name__=='__main__':unittest.main()
