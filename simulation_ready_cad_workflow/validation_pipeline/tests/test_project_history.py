import tempfile
from pathlib import Path
import unittest
from urllib.parse import parse_qs,urlparse

from ..dashboard.design_history import history_view
from ..dashboard.preview_service import shutdown_previews,_cache_root
from ..trajectory.capture import write_json
from ..trajectory.store import HistoryStore


class ProjectHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.workspace=Path(self.temp.name);self.project=self.workspace/'.runtime_runs/projects/example'
        for name in ('r01','r02','r10'):(self.project/'runs'/name).mkdir(parents=True)
        self.store=HistoryStore(self.project)
        self.addCleanup(shutdown_previews,self.workspace)

    def snapshot(self,revision,name,at,*,ready=True):
        path=self.project/'runs'/revision/'design'/name;path.mkdir(parents=True)
        source=path/'model.py';source.write_text('width = '+str(at)+'\n')
        model=path/'model.glb';model.write_bytes(b'glTF-saved')
        record={'id':name,'revision':revision,'capture':'build','status':'BUILT' if ready else 'BUILD_FAILED',
                'command':'build','at_ms':at,'completed_ms':at+1,'turn':revision+'-author','source_complete':True,
                'files':{'model.py':str(source)},'model':str(model) if ready else None}
        self.store.put_snapshot(record);return record

    def history(self,**kwargs):
        run=self.project/'runs/r01'
        return history_view(self.workspace,str(run.relative_to(self.workspace)),run,**kwargs)

    def test_project_scope_is_opt_in_and_keeps_each_models_own_run(self):
        self.snapshot('r01','one',900);self.snapshot('r02','two',500);self.snapshot('r10','ten',100)
        self.assertEqual(['one'],[item['id'] for item in self.history()['items']])
        data=self.history(scope='project')
        self.assertEqual(['one','two','ten'],[item['id'] for item in data['items']])
        self.assertEqual([0,1,2],[item['offset'] for item in data['revisions']])
        for item in data['items']:
            query=parse_qs(urlparse(item['model_url']).query)
            self.assertTrue(query['run'][0].endswith('/'+item['revision']))
        with self.assertRaises(ValueError):self.history(scope='project',selected='not-in-this-project')
        with self.assertRaises(ValueError):self.history(selected='two')

    def test_paging_and_opening_an_old_snapshot_keep_global_position(self):
        for index in range(205):self.snapshot('r01' if index<103 else 'r02','item-'+str(index),index)
        latest=self.history(scope='project');self.assertEqual(205,latest['total']);self.assertEqual(105,latest['offset']);self.assertEqual(100,len(latest['items']))
        selected=self.history(scope='project',selected='item-4')
        self.assertEqual(0,selected['offset']);self.assertEqual(4,selected['selected']['ordinal'])
        crossing=self.history(scope='project',around=103,selected='item-102')
        self.assertEqual(['r01','r02'],list(dict.fromkeys(item['revision'] for item in crossing['items'])))
        self.assertEqual(103,next(item['ordinal'] for item in crossing['items'] if item['id']=='item-103'))

    def test_failed_first_record_uses_prior_revision_and_source_diff(self):
        self.snapshot('r01','good',100);self.snapshot('r02','failed',200,ready=False)
        data=self.history(scope='project',selected='failed')
        self.assertEqual('failed',data['selected']['id']);self.assertIsNone(data['selected']['model_url'])
        self.assertEqual('r01',data['display_fallback']['revision'])
        self.assertEqual(1,next(marker['ordinal'] for marker in data['markers'] if marker['kind']=='build_failed'))
        normal=self.store.source_view('failed');self.assertIsNone(normal['comparison'])
        project=self.store.source_view('failed',across_revisions=True)
        self.assertEqual('r01',project['comparison_revision']);self.assertIn('-width = 100',project['diff'])

    def test_old_cached_previews_and_empty_versions_have_explicit_coverage(self):
        self.snapshot('r01','one',1)
        cache=_cache_root(self.workspace,self.project/'runs/r02');(cache/'old').mkdir(parents=True)
        (cache/'old/model.glb').write_bytes(b'glTF-old')
        write_json(cache/'timeline.json',{'frames':[{'id':'saved','model':'old/model.glb','created_at':None}]})
        data=self.history(scope='project',selected='legacy:r02:saved')
        self.assertEqual(2,data['total']);self.assertEqual([1,1,0],[row['count'] for row in data['revisions']])
        self.assertEqual('legacy',data['selected']['capture']);self.assertIsNone(data['selected']['created_at'])
        self.assertFalse(data['selected']['source_complete']);self.assertNotIn('turn',data['selected'])
        self.assertIn('name=timeline%2Fold%2Fmodel.glb',data['selected']['model_url'])

    def test_indexed_and_old_cached_models_in_same_revision_are_both_retained(self):
        self.snapshot('r01','one',1)
        cache=_cache_root(self.workspace,self.project/'runs/r01');(cache/'old').mkdir(parents=True)
        (cache/'old/model.glb').write_bytes(b'glTF-old')
        write_json(cache/'timeline.json',{'frames':[{'id':'only-in-old-cache','model':'old/model.glb','created_at':None}]})
        data=self.history(scope='project')
        self.assertEqual(2,data['total']);self.assertEqual({'one','legacy:r01:only-in-old-cache'},{item['id'] for item in data['items']})
        self.assertEqual(1,data['revisions'][0]['legacy_count'])

    def test_copied_index_and_legacy_preview_same_file_are_not_duplicated(self):
        cache=_cache_root(self.workspace,self.project/'runs/r01');(cache/'old').mkdir(parents=True)
        model=cache/'old/model.glb';model.write_bytes(b'glTF-old')
        write_json(cache/'timeline.json',{'frames':[{'id':'old','model':'old/model.glb','created_at':None}]})
        self.store.put_snapshot({'id':'indexed','revision':'r01','capture':'recovered','status':'BUILT',
            'model':str(Path('/previous-host')/model.relative_to(self.workspace))})
        data=self.history(scope='project')
        self.assertEqual(['indexed'],[item['id'] for item in data['items']])
