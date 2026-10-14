import json
from pathlib import Path
import tempfile
import types
import unittest

from ..dashboard.image_worker import render
from ..dashboard.history_markers import marker_view
from ..trajectory.capture import write_json
from ..trajectory.store import HistoryStore


class WorkbenchImageTests(unittest.TestCase):
    def test_pair_uses_shared_bounds_and_keeps_instances(self):
        import numpy as np
        import trimesh
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);paths=[]
            for width in (1,2):
                scene=trimesh.Scene();scene.add_geometry(trimesh.creation.box(extents=[width,1,1]),node_name='base',geom_name='shared')
                transform=np.eye(4);transform[0,3]=4
                scene.graph.update(frame_to='second',geometry='shared',matrix=transform)
                path=root/f'{width}.glb';path.write_bytes(trimesh.exchange.gltf.export_glb(scene));paths.append(str(path))
            original=[Path(p).read_bytes() for p in paths]
            result=render({'models':paths,'output':str(root/'images'),'up_axis':[0,0,1],'view':'front','width':240,'height':160})
            self.assertEqual('ready',result['status']);self.assertEqual(2,len(result['images']))
            self.assertAlmostEqual(6,result['camera']['span_m'][0])
            for path in result['images']:
                with Image.open(path) as image:self.assertEqual((240,160),image.size)
            self.assertEqual(original,[Path(p).read_bytes() for p in paths])

    def test_markers_keep_unbound_results_at_revision_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)/'.runtime_runs/projects/example';run=project/'runs/r01';run.mkdir(parents=True)
            store=HistoryStore(project)
            store.put_snapshot({'id':'build-a','revision':'r01','capture':'build','command':'build','status':'BUILT','at_ms':10})
            store.put_snapshot({'id':'check-a','revision':'r01','capture':'build','command':'check','status':'DEVELOPMENT_FAILURE','at_ms':20,'completed_ms':25})
            write_json(run/'result.json',{'status':'NEEDS_REVISION'})
            markers=marker_view(run)
            check=next(m for m in markers if m['kind']=='check_failed')
            self.assertEqual(1,check['ordinal']);self.assertEqual('check-a',check['snapshot_id'])
            final=next(m for m in markers if m['kind']=='validation_result')
            self.assertIsNone(final['ordinal']);self.assertEqual('revision',final['scope'])

    def test_timeout_after_recorded_request_has_terminal_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)/'.runtime_runs/projects/example';run=project/'runs/r01';run.mkdir(parents=True)
            HistoryStore(project).put_snapshot({'id':'simulate-a','revision':'r01','capture':'build','command':'simulate','status':'PHYSICS_TIMEOUT','at_ms':10,'completed_ms':30})
            config=project/'author_tools/simulate-a/runtime_config.json'
            write_json(config,{'attempt':'simulate-a','kind':'request','snapshot_id':'simulate-a','test_id':'loaded','captured_ms':20})
            write_json(project/'trajectory/configurations/a.json',{'file':str(config),'revision':'r01'})
            markers=marker_view(run)
            failed=next(m for m in markers if m['kind']=='simulation_failed')
            self.assertEqual(['loaded'],failed['test_ids']);self.assertEqual(0,failed['ordinal'])
