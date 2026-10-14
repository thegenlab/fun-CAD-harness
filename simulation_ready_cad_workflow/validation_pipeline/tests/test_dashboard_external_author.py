from datetime import datetime,timezone,timedelta
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import psutil
from ..dashboard.live_workflow import _workflow,_slim_test_progress
from ..mechanism.io import write_json_atomic


class ExternalAuthorDashboardTests(unittest.TestCase):
    def test_fresh_public_bridge_is_live_without_creating_a_workflow_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);identity=SimpleNamespace(project_dir=p,project_id='fresh')
            now=datetime.now(timezone.utc)
            state={'schema_version':'slim_harness_workflow_v1','request':{},'revision':1,'phase':'criteria','started_at':now.isoformat()}
            marker={'schema_version':'external_author_runtime_v1','status':'running','heartbeat_at':now.isoformat(),
                    'elapsed_seconds':10,'active_turn':{'id':'t001_external_author'}}
            write_json_atomic(p/'external_author_runtime.json',marker)
            self.assertTrue(_workflow(identity,state)['is_running'])
            self.assertFalse((p/'workflow_runtime.json').exists())
            marker['heartbeat_at']=(now-timedelta(seconds=30)).isoformat()
            write_json_atomic(p/'external_author_runtime.json',marker)
            old=(now-timedelta(seconds=30)).timestamp()
            os.utime(p/'external_author_runtime.json',(old,old))
            self.assertFalse(_workflow(identity,state)['is_running'])
            marker['heartbeat_at']=now.isoformat();write_json_atomic(p/'external_author_runtime.json',marker)
            state['complete_instance_pass']=True
            self.assertFalse(_workflow(identity,state)['is_running'])
            self.assertEqual(_workflow(identity,state)['status'],'complete')

    def test_pause_is_not_a_running_heartbeat_or_a_whole_machine_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);identity=SimpleNamespace(project_dir=p,project_id='fresh')
            now=datetime.now(timezone.utc).isoformat()
            state={'schema_version':'slim_harness_workflow_v1','request':{},'revision':1,'phase':'author','started_at':now}
            write_json_atomic(p/'external_author_runtime.json',{'schema_version':'external_author_runtime_v1',
                'status':'paused','reason':'human_intervention','heartbeat_at':now,'elapsed_seconds':20})
            result=_workflow(identity,state)
            self.assertFalse(result['is_running']);self.assertEqual(result['status'],'paused')
            self.assertEqual(result['phase'],'paused');self.assertEqual(result['stop_reason'],'human_intervention')
            self.assertFalse((p/'workflow_runtime.json').exists())

    def test_actual_harness_phase_wins_over_external_author(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);identity=SimpleNamespace(project_dir=p,project_id='fresh')
            now=datetime.now(timezone.utc).isoformat()
            state={'schema_version':'slim_harness_workflow_v1','request':{},'revision':1,'phase':'author','started_at':now}
            (p/'external_author_runtime.json').write_text(json.dumps({'schema_version':'external_author_runtime_v1','status':'running','heartbeat_at':now}))
            (p/'workflow_runtime.json').write_text(json.dumps({'schema_version':'workflow_runtime_v1','status':'running','heartbeat_at':now,
                'pid':os.getpid(),'process_started_at_epoch':psutil.Process().create_time(),'phase':'physics','revision':1}))
            result=_workflow(identity,state)
            self.assertTrue(result['is_running']);self.assertEqual(result['phase'],'physics')

    def test_template_tasks_are_shown_without_author_scene_tests_and_not_in_later_revision(self):
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder);attempt=project/'runs/r01/template_runs/a001'
            write_json_atomic(project/'template-progress.json',{'attempt':str(attempt),
                'plan':{'trials':[{'id':'light','duration_s':2},{'id':'heavy','duration_s':2}]},
                'tests':[{'test_id':'light','execution_status':'completed','physical_pass':False}]})
            state={'schema_version':'slim_harness_workflow_v1','request':{'mode':'free'},'revision':1,'phase':'physics','active_test':'heavy'}
            result=_slim_test_progress(project,state,True)
            self.assertEqual(result['total'],2);self.assertEqual(result['returned'],1)
            self.assertEqual(result['tests'][1]['execution_status'],'running')
            state['revision']=2
            self.assertEqual(_slim_test_progress(project,state,True),{})


if __name__=='__main__':unittest.main()
