import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from ..dashboard.project_summary import project_summary,short_reason
from ..dashboard.run_config import configuration_view
from ..trajectory.capture import record_turn,write_json,launch_publisher
from ..trajectory.store import HistoryStore


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.project=self.root/'.runtime_runs/projects/example'
        self.run=self.project/'runs/r01';self.design=self.run/'design';self.design.mkdir(parents=True)
        write_json(self.project/'workflow.json',{'revision':1,'phase':'author','request':{'project_id':'example'}})

    def test_configuration_uses_recorded_turn_not_new_project_defaults(self):
        class Worker:
            model='recorded-model';reasoning_effort='high';executable='local-worker'
            @record_turn
            def run_turn(self,workspace,prompt,output_dir,timeout_seconds):return {'status':'ready'}
        turn=self.project/'worker_turns/t001_author'
        Worker().run_turn(self.design,'test',turn,42)
        config=next(turn.glob('runtime_config.*.json'))
        write_json(self.design/'scene.json',{'units':'mm','parameters':{'width':12,'api_key':'do-not-show'},'bodies':[]})
        store=HistoryStore(self.project)
        store.put_snapshot({'id':'saved','revision':'r01','capture':'build','status':'BUILT','config_path':str(config),'files':{'scene.json':str(self.design/'scene.json')}})
        write_json(self.project/'workflow.json',{'revision':2,'request':{'model':'new-default'}})
        value=configuration_view(self.run,snapshot_id='saved')
        self.assertEqual('recorded-model',value['generation']['worker']['model'])
        self.assertEqual(42,value['generation']['worker']['turn_timeout_seconds'])
        self.assertEqual('[redacted]',value['mechanical']['parameters']['api_key'])
        geometry=self.project/'author_tools/tool-a/geometry/geometry.json'
        write_json(geometry,{'units':'m','bodies':[{'id':'base','mass_kg':2.5,'inertia_kg_m2':[[1,0,0],[0,1,0],[0,0,1]]}]})
        store.put_snapshot({'id':'geometry','revision':'r01','capture':'build','status':'BUILT','attempt':str(geometry.parent.parent)})
        self.assertEqual(2.5,configuration_view(self.run,snapshot_id='geometry')['mechanical']['recorded_body_properties'][0]['mass_kg'])
        with self.assertRaises(ValueError):configuration_view(self.project/'runs/r02',snapshot_id='saved')

    def test_old_configuration_is_unknown_not_filled_with_defaults(self):
        write_json(self.project/'workflow.json',{'revision':1,'request':{'model':'today'}})
        self.assertIsNone(configuration_view(self.run)['generation'])

    def test_simulation_settings_bind_to_model_not_snapshot_time(self):
        from ..trajectory.configuration import save_simulation_configuration
        store=HistoryStore(self.project)
        store.put_snapshot({'id':'model','revision':'r01','capture':'build','status':'BUILT','at_ms':1,'model_binding':{'scene_sha256':'same-scene','sources':[['part.step','source-a']]}})
        save_simulation_configuration(self.run/'attempts/a001/tests/load',interpreter='chrono',test={'id':'load'},
            numerical={'max_step_s':.0005},history_binding={'scene_sha256':'same-scene','sources':[['part.step','source-a']]})
        save_simulation_configuration(self.run/'attempts/a002/tests/load',interpreter='chrono',test={'id':'load'},
            numerical={'max_step_s':.001},history_binding={'scene_sha256':'same-scene','sources':[['part.step','source-b']]})
        value=configuration_view(self.run,snapshot_id='model',test_id='load')
        self.assertEqual(1,len(value['simulations']))
        self.assertEqual(.0005,value['simulations'][0]['numerical']['max_step_s'])

    def test_recovery_turn_and_normal_budget_are_recorded(self):
        write_json(self.project/'budget_checkpoint.json',{'tier':'high','tier_config':{'active_budget_seconds':100},'budget_limit_seconds':150,'granted_active_seconds':50})
        class Worker:
            model='new-model';reasoning_effort='high'
            @record_turn
            def run_turn(self,workspace,prompt,output_dir,timeout_seconds):return {}
        processes=[]
        def launch(*args):
            process=launch_publisher(*args);processes.append(process);return process
        self.addCleanup(lambda:[p.wait(timeout=15) for p in processes])
        with patch('simulation_ready_cad_workflow.validation_pipeline.trajectory.capture.launch_publisher',side_effect=launch):
            Worker().run_turn(self.design,'test',self.project/'worker_turns/t001_author/recovery-1',12)
        processes[-1].wait(timeout=15)
        value=configuration_view(self.run)
        self.assertEqual('new-model',value['generation']['worker']['model'])
        self.assertEqual(150,value['generation']['budget']['budget_limit_seconds'])
        self.assertEqual('t001_author',HistoryStore(self.project).state('capture')['turn'])

    def test_failure_summary_uses_concrete_reason(self):
        write_json(self.run/'result.json',{'status':'PHYSICAL_PREFLIGHT_FAILED','issues':[
            {'code':'LOAD_LANE_FAILED','message':'lane failed'},
            {'code':'LOAD_COMPARISON_FAILED','message':'physical_task_return_torque_target_not_reached'}]})
        self.assertEqual('回程转矩未达到目标',short_reason(self.run,{'status':'failed'}))
        write_json(self.run/'result.json',{'tests':[{'test_id':'load','assertions':[{'passed':False,'purpose':'30 mm 行程'}]}]})
        self.assertEqual('load：30 mm 行程未通过',short_reason(self.run,{'status':'failed'}))

    def test_notifications_are_suppressed_while_project_is_running(self):
        run_id=str(self.run.relative_to(self.root))
        row={'id':run_id,'name':'r01','revision':'r01','project_id':'example','package_id':'example','status':'failed',
             'effective_status':'failed','updated_at':'2026-09-15T12:00:00Z','healthy_running':True,
             'live_workflow':{'is_running':True,'stop_reason':'','revision':1}}
        server=types.SimpleNamespace(workspace_root=self.root,runs=lambda:[row],catalog_state=lambda:{'ready':True})
        result=project_summary(server)
        self.assertFalse(result['projects'][0]['notify'])
        self.assertEqual('running',result['projects'][0]['status'])
        self.assertTrue(result['projects'][0]['running'])
        row['healthy_running']=False;row['live_workflow']['is_running']=False
        self.assertTrue(project_summary(server)['projects'][0]['notify'])
