from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..mechanism.io import write_json_atomic
from ..slim_harness.construction_budget import initialize_budget
from ..slim_harness.construction_completion import write_completion_summary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ConstructionCompletionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.project=Path(self.temp.name)
        initialize_budget(self.project)
        write_json_atomic(self.project/'workflow.json',{'revision':1})
        (self.project/'task.md').write_text('Build a folding cart and verify load support.')
        self.run=self.project/'runs/r01';self.design=self.run/'design';self.design.mkdir(parents=True)
        self.scene_path=self.design/'scene.json';self.source=self.design/'cart.step'
        self.source.write_text('immutable CAD fixture bytes')
        self.scene={'schema_version':'slim_scene_v1','units':'mm','bodies':[{'id':'cart','source':'cart.step'}],
                    'tests':[{'id':'fold','purpose':'Folds','duration_s':1,'assertions':[{'id':'moves','criterion_ids':['F1']}]}]}
        write_json_atomic(self.scene_path,self.scene)
        write_json_atomic(self.project/'criteria.json',{'criteria':[{'id':'F1','kind':'function','statement':'Cart folds'}]})

    def attempt(self,*,formal=True,passed=True,trace=True,scope=None,snapshot=False):
        attempt=self.run/'attempts/a001' if formal else self.project/'tool-attempt'
        geometry_path=attempt/'geometry/geometry.json'
        scene_file=self.scene_path;source_file=self.source
        if snapshot:
            scene_file=attempt/'design/scene.json';scene_file.parent.mkdir(parents=True)
            scene_file.write_text(json.dumps(self.scene,separators=(',',':')))
            source_file=scene_file.parent/'cart.step';source_file.write_bytes(self.source.read_bytes())
        geometry={'scene_file':str(scene_file),'scene_sha256':digest(scene_file),'bodies':[
            {'id':'cart','source_file':str(source_file),'source_sha256':digest(source_file)}]}
        if scope:geometry['validation_scope']=scope
        write_json_atomic(geometry_path,geometry)
        folder=attempt/'tests/fold'
        request={'test':self.scene['tests'][0],'scene':{'bodies':geometry['bodies']}}
        if scope:request['scene']['validation_scope']=scope
        write_json_atomic(folder/'request.json',request)
        if trace:
            write_json_atomic(folder/'physical.trace.json',{'test_id':'fold','physical_pass':passed,
                'failure':None if passed else 'hinge displacement exceeds limit','assertions':[]})
        else:
            write_json_atomic(folder/'construction-cost.json',{'status':'hard_timeout','timeout_seconds':180})
        if formal:write_json_atomic(folder/'geometry-audit.json',{'passed':True})
        else:
            initial_path=attempt/'initial-audit.json';write_json_atomic(initial_path,{'passed':True,'violations':[]})
            report={'status':'DEVELOPMENT_PHYSICS_PASSED','geometry':str(geometry_path),'initial_audit':str(initial_path)}
            write_json_atomic(attempt/'result.json',report)
            write_json_atomic(self.project/'construction-evidence/tool.json',{'result_file':str(attempt/'result.json'),
                'source_scene_sha256':digest(scene_file)})
        return attempt

    def passed_result(self):
        return {'status':'PASS','criteria_sha256':digest(self.project/'criteria.json'),
                'source_bindings':{str(self.scene_path):digest(self.scene_path),str(self.source):digest(self.source)},
                'review':{'observations':[{'criterion_id':'F1','verdict':'met'}],
                          'test_coverage':[{'criterion_id':'F1','test_ids':['fold']}]}}

    def test_current_formal_evidence_and_review_completes_with_disclosed_scope(self):
        scope={'mode':'equivalent_elastic','excluded_claims':['Local stress is not established.']}
        self.attempt(scope=scope)
        summary=write_completion_summary(self.project,result=self.passed_result())
        self.assertTrue(summary['complete_instance_pass'])
        self.assertEqual(summary['verified_completed_tasks'],1)
        self.assertEqual(summary['tasks'][0]['limitations'],scope['excluded_claims'])
        self.assertEqual(summary['tasks'][0]['test_status'],'verified_within_declared_scope')

    def test_changed_cad_invalidates_old_pass_even_when_scene_unchanged(self):
        self.attempt();result=self.passed_result();self.source.write_text('changed joint and collision geometry')
        summary=write_completion_summary(self.project,result=result)
        self.assertFalse(summary['complete_instance_pass'])
        self.assertEqual(summary['tasks_with_executed_tests'],0)
        self.assertTrue(summary['rejected_stale_or_unbound_evidence'])
        self.assertEqual(summary['tasks'][0]['design_status'],'geometry_present_unreviewed')

    def test_changed_test_definition_with_same_id_does_not_reuse_pass(self):
        self.attempt();result=self.passed_result()
        self.scene['tests'][0]['duration_s']=10;write_json_atomic(self.scene_path,self.scene)
        summary=write_completion_summary(self.project,result=result)
        self.assertEqual(summary['tasks_with_executed_tests'],0)
        self.assertFalse(summary['complete_instance_pass'])

    def compiled_load_attempt(self,*,units='mm',formal=True):
        # Exercise the actual compiler's unit conversion. There are no joints;
        # only material-support lookup is stubbed, so no CAD is created/read.
        from ..slim_harness.contracts import compile_scene
        self.scene['units']=units
        self.scene['tests'][0]['loads']=[{'body':'cart','force_n':[0,-147.15,0],
                                        'point':[123.4,-250,80] if units=='mm' else [.1234,-.25,.08]}]
        write_json_atomic(self.scene_path,self.scene)
        attempt=self.attempt(formal=formal)
        geometry=json.loads((attempt/'geometry/geometry.json').read_text())
        with patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.source_support.check_supports',return_value=[]):
            compiled=compile_scene(self.scene,geometry)
        request=attempt/'tests/fold/request.json'
        write_json_atomic(request,{'scene':compiled,'test':compiled['tests'][0]})
        return request

    def test_compiler_mm_load_point_matches_current_test_without_source_mutation(self):
        original_source=self.source.read_bytes()
        request=self.compiled_load_attempt()
        original_scene=self.scene_path.read_bytes()
        summary=write_completion_summary(self.project,result=self.passed_result())
        self.assertTrue(summary['complete_instance_pass'])
        self.assertEqual(summary['tasks_with_executed_tests'],1)
        self.assertEqual(self.scene_path.read_bytes(),original_scene)
        self.assertEqual(self.source.read_bytes(),original_source)
        compiled=json.loads(request.read_text())['test']['loads'][0]
        self.assertEqual(compiled['point'],[123.4*.001,-.25,.08])
        self.assertEqual(compiled['force_n'],[0,-147.15,0])

    def test_compiler_mm_development_load_point_is_reported_as_partial(self):
        self.compiled_load_attempt(formal=False)
        summary=write_completion_summary(self.project)
        self.assertEqual(summary['tasks_with_executed_tests'],1)
        self.assertEqual(summary['tasks'][0]['completion_status'],'partial')

    def test_metre_load_point_is_not_rescaled_again(self):
        self.compiled_load_attempt(units='m')
        summary=write_completion_summary(self.project,result=self.passed_result())
        self.assertTrue(summary['complete_instance_pass'])

    def test_changed_force_or_application_point_still_rejects_request(self):
        request=self.compiled_load_attempt()
        original=json.loads(request.read_text())
        for field,replacement in [('force_n',[0,-14.715,0]),('point',[.2234,-.25,.08])]:
            with self.subTest(field=field):
                changed=copy.deepcopy(original);changed['test']['loads'][0][field]=replacement
                write_json_atomic(request,changed)
                summary=write_completion_summary(self.project,result=self.passed_result())
                self.assertFalse(summary['complete_instance_pass'])
                self.assertEqual(summary['tasks_with_executed_tests'],0)

    def test_development_snapshot_whitespace_is_valid_but_only_partial(self):
        self.attempt(formal=False,snapshot=True)
        summary=write_completion_summary(self.project)
        task=summary['tasks'][0]
        self.assertEqual(task['completion_status'],'partial')
        self.assertEqual(task['executed_tests'][0]['evidence_level'],'development_physics')
        self.assertEqual(summary['static_geometry_evidence'][0]['scope'],'original_geometry_initial_pose_only')
        self.assertFalse(summary['complete_instance_pass'])

    def test_editing_another_test_keeps_current_trial_evidence(self):
        self.scene['tests'].append({'id':'support','duration_s':1,'purpose':'Separate load case','assertions':[]})
        write_json_atomic(self.scene_path,self.scene)
        self.attempt(formal=False,snapshot=True)
        self.scene['tests'][1]['duration_s']=2
        write_json_atomic(self.scene_path,self.scene)
        summary=write_completion_summary(self.project)
        self.assertEqual([row['test_id'] for row in summary['test_results']],['fold'])
        self.assertTrue(summary['test_results'][0]['physical_pass'])
        self.assertFalse(summary['complete_instance_pass'])

    def test_shared_physics_change_still_invalidates_independent_trial(self):
        self.attempt(formal=False,snapshot=True)
        self.scene['gravity_m_s2']=[0,0,0]
        write_json_atomic(self.scene_path,self.scene)
        summary=write_completion_summary(self.project)
        self.assertEqual(summary['test_results'],[])

    def test_interrupted_test_reports_timeout_without_fabricating_execution_result(self):
        self.attempt(trace=False)
        summary=write_completion_summary(self.project,reason='budget_exhausted')
        task=summary['tasks'][0]
        self.assertEqual(task['completion_status'],'partial')
        self.assertEqual(task['test_status'],'partially_executed')
        self.assertIn('hard_timeout',task['reason'])
        self.assertIsNone(task['executed_tests'][0]['trace_file'])

    def test_actual_failure_is_explained_not_only_generic_failed(self):
        self.attempt(passed=False)
        summary=write_completion_summary(self.project)
        self.assertEqual(summary['tasks'][0]['completion_status'],'failed')
        self.assertIn('hinge displacement exceeds limit',summary['tasks'][0]['reason'])

    def test_finished_solver_with_failed_task_assertion_has_specific_reason(self):
        attempt=self.attempt(passed=False)
        write_json_atomic(attempt/'tests/fold/physical.trace.json',{
            'test_id':'fold','physical_pass':False,'failure':None,
            'assertions':[{'id':'x_hand_seated','purpose':'Carriage must reach the selected layer',
                           'passed':False,'value':False}]})
        summary=write_completion_summary(self.project)
        self.assertEqual(summary['test_results'][0]['failed_assertion_count'],1)
        self.assertIn('x_hand_seated',summary['tasks'][0]['reason'])
        self.assertIn('Carriage must reach',summary['tasks'][0]['reason'])

    def test_native_pass_with_audit_timeout_remains_partial_with_separate_failure_and_cost(self):
        attempt=self.attempt()
        folder=attempt/'tests/fold'
        trace_before=(folder/'physical.trace.json').read_bytes()
        failure={'code':'CONSTRUCTION_AUDIT_HARD_TIMEOUT','message':'Source-motion audit exhausted its separate 180 second cap.'}
        write_json_atomic(folder/'geometry-audit.json',{'passed':False,'status':'evidence_incomplete','failure':failure})
        write_json_atomic(folder/'construction-audit-cost.json',{'status':'timeout','elapsed_seconds':180.2,'effective_timeout_seconds':180})
        write_json_atomic(folder/'construction-simulation.json',{'status':'finished','elapsed_seconds':14.5})
        summary=write_completion_summary(self.project,result=self.passed_result())
        row=summary['test_results'][0];task=summary['tasks'][0]
        self.assertTrue(row['physical_pass'])
        self.assertIsNone(row['failure'])
        self.assertEqual(row['geometry_audit_status'],'evidence_incomplete')
        self.assertEqual(row['geometry_audit_failure'],failure)
        self.assertEqual(row['geometry_audit_cost']['elapsed_seconds'],180.2)
        self.assertEqual(row['native_simulation_cost']['elapsed_seconds'],14.5)
        self.assertIn('CONSTRUCTION_AUDIT_HARD_TIMEOUT',task['reason'])
        self.assertIn(str(folder/'construction-audit-cost.json'),task['evidence_paths'])
        self.assertEqual(task['completion_status'],'partial')
        self.assertFalse(summary['complete_instance_pass'])
        self.assertEqual((folder/'physical.trace.json').read_bytes(),trace_before)

    def test_no_criteria_uses_delivered_test_definitions_without_claiming_core_review(self):
        (self.project/'criteria.json').unlink()
        self.attempt(formal=False,snapshot=True)
        summary=write_completion_summary(self.project)
        self.assertEqual(summary['task_denominator_basis'],'unreviewed_scene_tests')
        self.assertEqual(summary['tasks'][0]['task_id'],'fold')
        self.assertEqual(summary['tasks'][0]['completion_status'],'partial')

    def test_formal_exhaust_does_not_drop_finished_development_pass(self):
        self.attempt(formal=False,snapshot=True)
        later=self.run/'attempts/a001'
        geometry_path=later/'geometry/geometry.json'
        write_json_atomic(geometry_path,{'scene_file':str(self.scene_path),'scene_sha256':digest(self.scene_path),
            'bodies':[{'id':'cart','source_file':str(self.source),'source_sha256':digest(self.source)}]})
        folder=later/'tests/fold'
        write_json_atomic(folder/'request.json',{'test':self.scene['tests'][0],'scene':{'bodies':[{'id':'cart'}]}})
        write_json_atomic(folder/'construction-cost.json',{'status':'hard_timeout','timeout_seconds':180})
        summary=write_completion_summary(self.project)
        row=summary['test_results'][0]
        self.assertTrue(row['physical_pass'])
        self.assertEqual(row['evidence_level'],'development_physics')
        self.assertEqual(row['execution_status'],'finished')
        self.assertFalse(summary['complete_instance_pass'])

    def test_budget_exhaustion_before_scene_or_criteria_still_writes_useful_summary(self):
        (self.project/'criteria.json').unlink();self.scene_path.write_text('{incomplete');self.source.unlink()
        (self.design/'model.py').write_text('# source drafting began')
        from ..slim_harness.construction_budget import ProjectBudget
        snapshot=ProjectBudget(self.project).snapshot();snapshot.update(work_remaining_seconds=0,remaining_seconds=120)
        with patch.object(ProjectBudget,'snapshot',return_value=snapshot):
            summary=write_completion_summary(self.project,reason='summary_reserve')
        self.assertEqual(summary['status'],'BUDGET_EXHAUSTED_PARTIAL')
        self.assertEqual(summary['task_count'],1)
        self.assertEqual(summary['task_denominator_basis'],'core_task_not_yet_decomposed')
        self.assertEqual(summary['tasks'][0]['design_status'],'source_only')
        self.assertTrue((self.project/'completion_summary.md').is_file())
        self.assertTrue((self.project/'completion_summary.json').is_file())

    def test_cursor_result_usage_is_recorded_in_summary(self):
        turn=self.project/'worker_turns/t002_author'
        turn.mkdir(parents=True)
        write_json_atomic(turn/'usage.live.json',{
            'schema_version':'cold_start_live_token_usage_v1','call_id':'author-1',
            'status':'completed','pending':False,'usage':None})
        (turn/'events.jsonl').write_text(json.dumps({
            'type':'result','subtype':'success','is_error':False,
            'usage':{'inputTokens':120,'outputTokens':80,
                     'cacheReadTokens':50,'cacheWriteTokens':10},
        })+'\n',encoding='utf-8')
        summary=write_completion_summary(self.project)
        usage=summary['model_usage']
        self.assertEqual(usage['reported_totals'],{
            'input_tokens':180,'cached_input_tokens':50,'output_tokens':80})
        self.assertEqual(usage['total_tokens'],260)
        self.assertEqual(usage['calls'],1)
        self.assertEqual(usage['calls_without_reported_usage'],0)
        text=(self.project/'completion_summary.md').read_text(encoding='utf-8')
        self.assertIn('模型用量：input 180 / output 80 / 合计 260',text)

    def test_live_snake_case_usage_is_summed(self):
        turn=self.project/'worker_turns/t001_author'
        turn.mkdir(parents=True)
        write_json_atomic(turn/'usage.live.json',{
            'call_id':'a','status':'completed',
            'usage':{'input_tokens':10,'output_tokens':4,'cached_input_tokens':3}})
        summary=write_completion_summary(self.project)
        self.assertEqual(summary['model_usage']['reported_totals']['input_tokens'],10)
        self.assertEqual(summary['model_usage']['reported_totals']['cached_input_tokens'],3)
        self.assertEqual(summary['model_usage']['total_tokens'],14)


if __name__=='__main__':unittest.main()
