"""History survives real workflow boundaries without requiring a browser."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

from ..trajectory.store import HistoryStore
from ..trajectory.capture import FileObserver, TurnCapture, snapshot_attempt, write_json, record_turn, emit_public_event, launch_publisher
from ..dashboard import history_preview, preview_service
from ..dashboard.design_history import history_view


class DesignHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name).resolve()
        self.project=self.workspace/'.runtime_runs/projects/machine'
        self.design=self.project/'runs/r01/design';self.design.mkdir(parents=True)
        write_json(self.project/'workflow.json',{'schema_version':'slim_harness_workflow_v1',
            'request':{'project_id':'machine','brief':'fixture'},'revision':1,'phase':'author'})
        self.turn=self.project/'worker_turns/t001_author';self.turn.mkdir(parents=True)
        self.log=self.turn/'events.jsonl';self.store=HistoryStore(self.project)

    def events(self,events):
        self.log.write_text(''.join(json.dumps(e)+'\n' for e in events))

    def message(self,text,stamp=None):
        value={'type':'item.completed','item':{'type':'agent_message','id':'item_'+text[:12],'text':text}}
        if stamp is not None:value['timestamp_ms']=stamp
        return value

    def test_legacy_attempt_reuses_saved_assets_and_exact_call_association(self):
        attempt=self.workspace/'.runtime_runs/author_tools/build-example'
        (attempt/'design').mkdir(parents=True);(attempt/'geometry').mkdir()
        (attempt/'design/model.py').write_text('width = 10')
        (attempt/'design/part.step').write_text('retained geometry')
        write_json(attempt/'design/scene.json',{'bodies':[{'id':'body','source':'part.step'}]})
        model=attempt/'geometry/model.glb';model.write_bytes(b'glTF-fixture')
        write_json(attempt/'geometry/geometry.json',{'model_file':str(model)})
        write_json(attempt/'result.json',{'source':str(self.design/'model.py'),'command':'build','status':'BUILT'})
        self.events([{'type':'item.completed','timestamp_ms':1200,'item':{
            'type':'command_execution','id':'cmd','command':'build model.py','aggregated_output':json.dumps({'attempt':str(attempt),'status':'BUILT'})}}])
        self.store.index_log(self.log,self.turn.name,drain=True)
        snapshot_attempt(self.workspace,attempt,recovered=True)
        saved=self.store.snapshot(attempt.name)
        self.assertEqual(str(model),saved['model']);self.assertEqual('t001_author',saved['turn'])
        self.assertEqual(1200,saved['at_ms']);self.assertEqual('recovered',saved['capture'])
        (self.design/'model.py').write_text('width = 99')
        self.assertEqual('width = 10',Path(saved['files']['model.py']).read_text())
        page=self.store.event_page(snapshot=attempt.name)
        self.assertEqual('linked',page['association']);self.assertIn('build model.py',page['items'][0]['text'])

    def test_partial_event_is_retried_and_missing_time_gets_receive_time_only_when_live(self):
        raw=json.dumps(self.message('hello'))
        self.log.write_text(raw[:20]);self.store.index_log(self.log,self.turn.name,observed=True)
        self.assertEqual([],self.store.event_page()['items'])
        self.log.write_text(raw+'\n');self.store.index_log(self.log,self.turn.name,observed=True)
        page=self.store.event_page(from_ms=0)
        self.assertEqual(['hello'],[r['text'] for r in page['items']])
        self.assertIsNotNone(page['items'][0]['at_ms'])
        self.store.index_log(self.log,self.turn.name,observed=True)
        self.assertEqual(1,len(self.store.event_page()['items']))

    def test_attempt_link_uses_full_result_after_long_build_output(self):
        path=str(self.workspace/'.runtime_runs/author_tools/build-long')
        self.events([{'type':'item.completed','timestamp_ms':1200,'item':{
            'type':'command_execution','id':'cmd','command':'build model.py',
            'aggregated_output':'x'*15000+'\n'+json.dumps({'attempt':path,'status':'BUILT'})}}])
        self.store.index_log(self.log,self.turn.name,drain=True)
        self.store.put_snapshot({'id':'build-long','revision':'r01','capture':'build','status':'BUILT'})
        saved=self.store.snapshot('build-long')
        self.assertIsNotNone(saved['event_seq']);self.assertEqual(self.turn.name,saved['turn'])
        self.assertNotIn(path,self.store.event_page()['items'][0]['output'])

    def test_forward_pages_keep_all_large_events_and_query_never_reads_raw_log(self):
        self.events([{'type':'item.completed','timestamp_ms':i,'item':{'id':str(i),
            'type':'command_execution','command':str(i),'aggregated_output':'x'*11000}} for i in range(140)])
        self.store.index_log(self.log,self.turn.name,drain=True)
        seen=[];after=0
        with patch.object(Path,'read_bytes',side_effect=AssertionError('query reread raw log')):
            while True:
                page=self.store.event_page(after=after)
                if not page['items']:break
                seen.extend(int(x['text']) for x in page['items']);after=page['next_after']
        self.assertEqual(list(range(140)),seen)

    def test_retry_segment_does_not_merge_reused_backend_item_ids(self):
        class Worker:
            @record_turn
            def run_turn(self,workspace,prompt,output_dir,timeout_seconds):
                emit_public_event({'type':'item.completed','item':{
                    'id':'item_0','type':'command_execution','command':prompt,'aggregated_output':prompt}})
        worker=Worker()
        processes=[]
        def launch(*args):
            process=launch_publisher(*args);processes.append(process);return process
        self.addCleanup(lambda:[p.wait(timeout=15) for p in processes])
        with patch('simulation_ready_cad_workflow.validation_pipeline.trajectory.capture.launch_publisher',side_effect=launch):
            worker.run_turn(self.design,'first',self.turn,5)
            processes[-1].wait(timeout=15)
            worker.run_turn(self.design,'second',self.turn,5)
            processes[-1].wait(timeout=15)
        self.assertEqual(['first','second'],[r['text'] for r in self.store.event_page(turn=self.turn.name)['items']])
        self.assertEqual(2,len(list(self.turn.glob('events*.jsonl'))))

    def test_file_observation_reuses_unchanged_sources_and_records_deletions(self):
        (self.design/'model.py').write_text('width = 10')
        (self.design/'assembly.step').write_text('retained STEP')
        observer=FileObserver(self.store,self.design,self.turn.name);observer.tick(force=True)
        first=self.store.history('r01')['items'][-1]
        (self.design/'model.py').write_text('width = 20');observer.tick(force=True)
        second=self.store.history('r01')['items'][-1]
        self.assertEqual(first['files']['assembly.step'],second['files']['assembly.step'])
        self.assertNotEqual(first['files']['model.py'],second['files']['model.py'])
        diff=self.store.source_view(second['id'])
        self.assertIn('-width = 10',diff['diff']);self.assertIn('+width = 20',diff['diff'])
        self.assertEqual(first['id'],diff['comparison'])
        resumed=FileObserver(self.store,self.design,'t002_author');resumed.tick(force=True)
        self.assertEqual(2,self.store.history('r01')['total'])
        (self.design/'assembly.step').unlink();observer.tick(force=True)
        last=self.store.history('r01')['items'][-1]
        self.assertEqual('incomplete',last['status']);self.assertIn('assembly.step',last['removed_files'])
        self.assertTrue(Path(first['files']['assembly.step']).is_file())

    def test_snapshot_chooses_own_turn_after_later_worker_exists(self):
        self.events([self.message('first turn',1000)]);self.store.index_log(self.log,self.turn.name,drain=True)
        second=self.project/'worker_turns/t002_author/events.jsonl';second.parent.mkdir()
        second.write_text(json.dumps(self.message('other turn',2000))+'\n')
        self.store.index_log(second,'t002_author',drain=True)
        self.store.put_snapshot({'id':'old','revision':'r01','capture':'build','status':'BUILT','turn':'t001_author','event_seq':1,'at_ms':1000})
        self.assertEqual(['first turn'],[r['text'] for r in self.store.event_page(snapshot='old')['items']])

    def test_invalid_scene_still_records_failed_attempt_and_nested_sources(self):
        helper=self.design/'lib/shape.py';helper.parent.mkdir();helper.write_text('width=12')
        write_json(self.design/'scene.json',{'bodies':42,'authoring_sources':42})
        observer=FileObserver(self.store,self.design,self.turn.name);observer.tick(force=True)
        saved=self.store.history('r01')['items'][-1]
        self.assertEqual('incomplete',saved['status']);self.assertIn('lib/shape.py',saved['files'])
        attempt=self.workspace/'.runtime_runs/author_tools/build-invalid';(attempt/'design').mkdir(parents=True)
        write_json(attempt/'design/scene.json',{'gravity_m_s2':'invalid'})
        saved=snapshot_attempt(self.workspace,attempt,{'source':str(self.design/'model.py'),'status':'DEVELOPMENT_FAILURE'})
        self.assertEqual('DEVELOPMENT_FAILURE',saved['status'])

    def test_slim_publication_keeps_history_and_more_than_fourteen_models(self):
        from ..dashboard.server import DashboardServer
        model=self.design/'model.glb';model.write_bytes(b'glTF-fixture')
        for i in range(20):self.store.put_snapshot({'id':f'build-{i:02}','revision':'r01','capture':'build','status':'BUILT','model':str(model),'at_ms':i})
        write_json(self.design.parent/'slim_evidence/preview.json',{'schema_version':'slim_studio_preview_v1','status':'ready','model_path':str(model),'input_version':'formal'})
        write_json(self.design.parent/'result.json',{'status':'PASS'})
        server=types.SimpleNamespace(workspace_root=self.workspace)
        self.addCleanup(preview_service.shutdown_previews,self.workspace)
        payload=DashboardServer.studio_preview(server,str(self.design.parent.relative_to(self.workspace)),self.design.parent)
        self.assertEqual(20,len(payload['timeline']))
        self.assertTrue(all('/preview-file?' in r['model_url'] for r in payload['timeline']))

    def test_cache_eviction_only_removes_disposable_conversions(self):
        service=types.SimpleNamespace(workspace=self.workspace,lock=threading.Lock(),image_inputs={})
        cached=history_preview.cache_root(service)/'machine/old';cached.mkdir(parents=True)
        (cached/'model.glb').write_bytes(b'x'*50);os.utime(cached/'model.glb',(1,1))
        original=self.design/'model.glb';original.write_bytes(b'glTF-evidence')
        service.image_inputs['pending']=[str(cached/'model.glb')]
        history_preview.prune(service,limit=10)
        self.assertTrue(cached.exists())
        service.image_inputs.clear()
        history_preview.prune(service,limit=10)
        self.assertFalse(cached.exists());self.assertEqual(b'glTF-evidence',original.read_bytes())

    def test_observer_runs_without_dashboard(self):
        (self.design/'model.py').write_text('v1')
        with TurnCapture(self.design,self.turn) as capture:
            self.addCleanup(lambda:capture.process.wait(timeout=15))
            deadline=time.monotonic()+10
            while time.monotonic()<deadline and self.store.history('r01')['total']==0:time.sleep(.05)
            self.assertGreater(self.store.history('r01')['total'],0)
            (self.design/'model.py').write_text('v2')
        capture.process.wait(timeout=15)
        self.assertEqual(2,self.store.history('r01')['total'])


class HistoryGeometryTests(unittest.TestCase):
    def test_instanced_glb_keeps_world_transform_and_solid_selection(self):
        import numpy as np
        import trimesh
        from ..dashboard.preview_worker import _slim_scene
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);scene=trimesh.Scene()
            scene.add_geometry(trimesh.creation.box(),node_name='a',geom_name='shared')
            pose=np.eye(4);pose[0,3]=10
            scene.graph.update(frame_to='b',matrix=pose,geometry='shared')
            mesh=root/'parts.glb';mesh.write_bytes(trimesh.exchange.gltf.export_glb(scene))
            source=root/'scene.json'
            write_json(source,{'units':'m','bodies':[{'id':'body','source':'parts.glb','color':[255,0,0]}]})
            shown=_slim_scene(source,{})
            self.assertEqual(2,len(shown.geometry));self.assertAlmostEqual(11,shown.bounds[1,0]-shown.bounds[0,0])
            write_json(source,{'units':'m','bodies':[{'id':'body','source':'parts.glb','solid_ids':['solid_2']}]})
            shown=_slim_scene(source,{})
            self.assertEqual(1,len(shown.geometry));self.assertAlmostEqual(10,shown.centroid[0])
