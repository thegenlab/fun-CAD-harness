"""Publication binds drafts at start and never waits for simulation or indexing."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..trajectory.capture import (HarnessProgramCapture, TurnCapture, _publish_attempt, read_json,
    turn_environment, write_json, _CURRENT)
from ..trajectory.publisher import publish_pending, run_turn
from ..trajectory.store import HistoryStore


class TrajectoryPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name).resolve()
        self.project=self.workspace/'.runtime_runs/projects/example'
        (self.project/'runs/r01/design').mkdir(parents=True)
        write_json(self.project/'workflow.json',{'revision':1,'active_turn':{'id':'t001_author'}})
        self.attempt=self.workspace/'.runtime_runs/author_tools/simulate-example'
        self.attempt.mkdir(parents=True)
        self.result={'command':'simulate','source':str(self.workspace/'draft/scene.json')}
        self.launch=patch('simulation_ready_cad_workflow.validation_pipeline.trajectory.capture.launch_publisher',return_value=Mock(poll=lambda:None))
        self.launcher=self.launch.start();self.addCleanup(self.launch.stop)
        self.environment=patch.dict(os.environ,{},clear=True)
        self.environment.start();self.addCleanup(self.environment.stop)

    def test_explicit_project_binds_draft_once_and_publishes_geometry_before_completion(self):
        store=HistoryStore(self.project)
        with patch.object(HistoryStore,'put_snapshot',side_effect=AssertionError('indexing in author process')):
            _publish_attempt(self.workspace,self.attempt,self.result,project=self.project)
        write_json(self.project/'workflow.json',{'revision':2,'active_turn':{'id':'t002_review'}})
        model=self.attempt/'geometry/model.glb';model.parent.mkdir();model.write_bytes(b'glTF fixture')
        write_json(model.parent/'geometry.json',{'model_file':str(model)})
        _publish_attempt(self.workspace,self.attempt,self.result,project=self.project,stage='geometry_ready')
        publish_pending(self.project,{})
        snapshot=store.snapshot(self.attempt.name)
        self.assertEqual('r01',snapshot['revision'])
        self.assertEqual('t001_author',snapshot['turn'])
        self.assertEqual(str(model),snapshot['model'])
        self.assertIsNone(snapshot['completed_ms'])
        self.assertFalse((self.attempt/'result.json').exists())
        stamp=snapshot['geometry_ready_ms']
        _publish_attempt(self.workspace,self.attempt,{**self.result,'status':'DEVELOPMENT_FAILURE'},project=self.project,stage='finished')
        publish_pending(self.project,{})
        snapshot=store.snapshot(self.attempt.name)
        self.assertEqual(stamp,snapshot['geometry_ready_ms'])
        self.assertEqual('DEVELOPMENT_FAILURE',snapshot['status'])
        self.assertEqual('r01',snapshot['revision'])

    def test_turn_binds_environment_and_returns_without_waiting_for_publisher(self):
        output=self.project/'worker_turns/t001_template_review_fill'
        capture=TurnCapture(self.workspace/'reviewer_binding',output)
        with capture:
            token=_CURRENT.set(capture)
            try:environment=turn_environment()
            finally:_CURRENT.reset(token)
            self.assertEqual('r01',environment['SIMREADY_TRAJECTORY_REVISION'])
            self.assertEqual(str(capture.control_path),environment['SIMREADY_TRAJECTORY_PUBLISHER'])
            (output/'events.jsonl').write_text(json.dumps({'type':'item.completed','timestamp_ms':1000,
                'item':{'type':'agent_message','text':'Reviewer filling binding.py'}})+'\n')
        self.launcher.return_value.wait.assert_not_called()
        run_turn(capture.control_path)
        items=HistoryStore(self.project).event_page(turn=output.name)['items']
        self.assertEqual(['Reviewer filling binding.py'],[row['text'] for row in items])

    def test_child_uses_parent_publisher_queue(self):
        control=self.project/'worker_turns/t001_author/publication.json'
        write_json(control,{'status':'recording','project':str(self.project)})
        with patch.dict(os.environ,{'SIMREADY_TRAJECTORY_PUBLISHER':str(control)}):
            _publish_attempt(self.workspace,self.attempt,self.result,project=self.project)
        self.launcher.assert_not_called()
        self.assertEqual('r01',read_json(self.attempt/'capture.json')['revision'])
        self.assertTrue((self.project/'trajectory/pending/simulate-example.json').is_file())

    def test_benchmark_free_and_custom_keep_distinct_revision_attempts(self):
        identities=[]
        for revision,mode in enumerate(('benchmark','free','custom'),1):
            write_json(self.project/'workflow.json',{'revision':revision,'mode':mode})
            attempt=self.project/f'runs/r{revision:02d}/template_runs/a001/prepared'
            attempt.mkdir(parents=True)
            write_json(attempt/'prepared.json',{'bodies':[]})
            _publish_attempt(self.workspace,attempt,self.result,project=self.project,stage='geometry_ready')
            identities.append(read_json(attempt/'capture.json')['snapshot_id'])
        publish_pending(self.project,{})
        self.assertEqual(3,len(set(identities)))
        for revision,identity in enumerate(identities,1):
            snapshot=HistoryStore(self.project).snapshot(identity)
            self.assertEqual(f'r{revision:02d}',snapshot['revision'])
            self.assertIn('prepared.json',snapshot['files'])
            self.assertFalse(snapshot['source_complete'])

    def test_harness_commands_have_real_started_completed_events_without_model_usage(self):
        output=self.project/'worker_turns/t003_harness_tasks'
        with HarnessProgramCapture(self.project/'runs/r01/design',output) as capture:
            with capture.command('prepare request.json') as event:
                started=json.loads((output/'events.jsonl').read_text().strip())
                self.assertEqual('running',started['item']['status'])
                self.assertEqual('harness_program',started['source'])
                event.update(exit_code=0,aggregated_output='geometry ready')
            with self.assertRaisesRegex(RuntimeError,'solver stopped'):
                with capture.command('run trial.json'):
                    raise RuntimeError('solver stopped')
        run_turn(capture.control_path)
        items=HistoryStore(self.project).event_page(turn=output.name)['items']
        self.assertEqual(['completed','failed'],[row['status'] for row in items])
        self.assertTrue(all(row['source']=='harness_program' for row in items))
        self.assertEqual('geometry ready',items[0]['output'])
        self.assertIn('RuntimeError: solver stopped',items[1]['output'])
        self.assertFalse(list(output.glob('usage*')))
        self.assertFalse(list(output.glob('runtime_config*')))
        self.assertEqual('harness_program_capture_v1',read_json(output/'program.json')['schema_version'])
        self.launcher.return_value.wait.assert_not_called()


if __name__=='__main__':unittest.main()
