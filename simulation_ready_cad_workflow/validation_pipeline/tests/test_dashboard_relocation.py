import json
import shutil
import sys
from unittest.mock import patch
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ..dashboard.history_preview import model_path
from ..trajectory.store import HistoryStore


class CopiedDashboardHistoryTests(unittest.TestCase):
    def test_copied_snapshot_models_and_source_are_read_without_rewriting_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "receiving-workspace"
            old_workspace = Path(temporary) / "original-machine"
            project = workspace / ".runtime_runs/projects/example"
            relative = Path(".runtime_runs/author_tools/build-example")
            model = workspace / relative / "geometry/model.glb"
            source = workspace / relative / "design/model.py"
            model.parent.mkdir(parents=True)
            source.parent.mkdir(parents=True)
            model.write_bytes(b"model fixture")
            source.write_text("print('saved source')\n")
            recorded = {
                "id": "build-example", "revision": "r01", "capture": "build",
                "status": "BUILT", "source_complete": True,
                "model": str(old_workspace / relative / "geometry/model.glb"),
                "source": str(old_workspace / relative / "design/model.py"),
                "files": {"model.py": str(old_workspace / relative / "design/model.py")},
            }
            store = HistoryStore(project)
            store.put_snapshot(recorded)
            snapshot = store.snapshot("build-example")
            self.assertEqual(snapshot["model"], str(model))
            self.assertEqual(snapshot["source"], str(source))
            self.assertEqual(store.source_view("build-example")["content"], source.read_text())
            service = SimpleNamespace(workspace=workspace)
            self.assertEqual(model_path(service, project, recorded), model)
            with store.connect() as db:
                original = json.loads(db.execute("SELECT data FROM snapshots").fetchone()[0])
            self.assertEqual(original, recorded)

    def test_copied_configuration_and_simulation_markers_use_receiving_files(self):
        from ..dashboard.run_config import configuration_view
        from ..dashboard.history_markers import marker_view
        from ..trajectory.capture import write_json
        with tempfile.TemporaryDirectory() as temporary:
            workspace=Path(temporary)/'receiving';project=workspace/'.runtime_runs/projects/example'
            run=project/'runs/r01';run.mkdir(parents=True)
            worker=project/'worker_turns/t001_author/runtime_config.saved.json'
            simulation=run/'attempts/a001/tests/load/runtime_config.json'
            def recorded(path):return str(Path('/previous-machine')/path.relative_to(workspace))
            write_json(worker,{'worker':{'model':'recorded-model'}})
            write_json(simulation,{'kind':'request','attempt':'saved-attempt','snapshot_id':'saved','test_id':'load','captured_ms':25})
            write_json(project/'trajectory/configurations/load.json',{'file':recorded(simulation),'revision':'r01','test_id':'load'})
            store=HistoryStore(project);store.put_snapshot({'id':'saved','revision':'r01','capture':'build','status':'BUILT','at_ms':10,'config_path':recorded(worker)})
            before=worker.read_bytes(),simulation.read_bytes()
            value=configuration_view(run,snapshot_id='saved',test_id='load')
            self.assertEqual('recorded-model',value['generation']['worker']['model'])
            self.assertEqual('load',value['simulations'][0]['test_id'])
            self.assertEqual('saved',next(m for m in marker_view(run) if m['kind']=='simulation_requested')['snapshot_id'])
            self.assertEqual(before,(worker.read_bytes(),simulation.read_bytes()))

    def test_copying_index_reuses_log_cursor_and_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            old=Path(temporary)/'old';new=Path(temporary)/'new';relative=Path('.runtime_runs/projects/example')
            project=old/relative;log=project/'worker_turns/t001_author/events.jsonl';log.parent.mkdir(parents=True)
            first={'type':'item.completed','item':{'type':'agent_message','id':'one','text':'first'}}
            log.write_text(json.dumps(first)+'\n')
            store=HistoryStore(project);self.assertEqual(1,store.index_log(log,'t001_author'))
            with store.connect() as db:segment=db.execute('SELECT id FROM segments').fetchone()[0]
            shutil.copytree(old,new)
            copied=HistoryStore(new/relative);copy_log=new/log.relative_to(old)
            self.assertEqual(0,copied.index_log(copy_log,'t001_author'))
            with copy_log.open('a') as stream:stream.write(json.dumps({**first,'item':{**first['item'],'id':'two','text':'second'}})+'\n')
            self.assertEqual(1,copied.index_log(copy_log,'t001_author'))
            with copied.connect() as db:
                self.assertEqual(2,db.execute('SELECT COUNT(*) FROM events').fetchone()[0])
                self.assertEqual(segment,db.execute('SELECT id FROM segments').fetchone()[0])
                self.assertEqual(str(copy_log),db.execute('SELECT raw_path FROM events LIMIT 1').fetchone()[0])
            self.assertEqual(1,len(log.read_text().splitlines()))

    def test_recovered_windows_attempt_uses_existing_copied_artifacts(self):
        from ..trajectory.capture import snapshot_attempt,write_json
        from ..trajectory.configuration import model_binding
        from ..dashboard.image_service import model_reference
        from ..dashboard.preview_service import shutdown_previews,resolve_cad_python
        with tempfile.TemporaryDirectory() as temporary:
            workspace=Path(temporary);project=workspace/'.runtime_runs/projects/example';run=project/'runs/r01'
            (run/'design').mkdir(parents=True)
            write_json(project/'workflow.json',{'revision':1})
            attempt=workspace/'.runtime_runs/author_tools/build-copied'
            (attempt/'geometry').mkdir(parents=True);(attempt/'geometry/model.glb').write_bytes(b'glTF fixture')
            write_json(attempt/'design/scene.json',{'bodies':[]})
            old_root='F:/old-workspace/'
            old=lambda path:(old_root+str(path.relative_to(workspace))).replace('/','\\')
            write_json(attempt/'result.json',{'source':old(run/'design/model.py'),'command':'build','status':'BUILT'})
            write_json(attempt/'capture.json',{'project':old(project),'source':old(run/'design/model.py'),'revision':'r01','cad_python':'F:/missing/python.exe'})
            write_json(attempt/'geometry/geometry.json',{'model_file':old(attempt/'geometry/model.glb'),'scene_file':old(attempt/'design/scene.json'),
                'scene_sha256':'scene','source_bindings':[{'source_file':old(attempt/'design/part.step'),'source_sha256':'part'}]})
            value=snapshot_attempt(workspace,attempt,recovered=True)
            self.assertEqual(str(attempt/'geometry/model.glb'),value['model'])
            self.assertEqual([['part.step','part']],value['model_binding']['sources'])
            try:
                with patch.dict('os.environ',{'SIMREADY_CAD_PYTHON':sys.executable}):
                    model=model_reference(workspace,run,attempt.name)
                    self.assertEqual(str(Path(sys.executable).absolute()),model['python'])
                    self.assertEqual(str(Path(sys.executable).absolute()),resolve_cad_python('F:/missing/python.exe'))
            finally:shutdown_previews(workspace)


if __name__ == "__main__":
    unittest.main()
