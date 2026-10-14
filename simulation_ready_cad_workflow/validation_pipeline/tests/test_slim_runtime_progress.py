"""A stalled import still leaves a truthful stage record, without fake poses."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from ..slim_harness.runtime_progress import RuntimeProgress


class RuntimeProgressTests(unittest.TestCase):
    def test_stage_snapshot_and_completion_do_not_claim_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            progress=RuntimeProgress(Path(directory)/'physical.trace.json')
            progress.mark('loading_native')
            first=json.loads(progress.path.read_text(encoding='utf-8'))
            self.assertEqual(first['recorded_samples'],0)
            self.assertIsNone(first['last_recorded_time_s'])
            progress.mark('finished',recorded_samples=3,last_recorded_time_s=.002)
            result=json.loads(progress.path.read_text(encoding='utf-8'))
            self.assertEqual(result['stage'],'finished')
            self.assertFalse(result['certified'])
            self.assertEqual(len(progress.log_path.read_text(encoding='utf-8').splitlines()),2)

    def test_hard_kill_during_import_retains_stage_and_no_invented_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'physical.trace.json'
            request=Path(directory)/'request.json'
            request.write_text(json.dumps({'output':str(output),'scene':{},'test':{'id':'startup','duration_s':1}}),encoding='utf-8')
            code='''import builtins,runpy,sys,time
original=builtins.__import__
def delayed(name,*args,**kwargs):
    if name=='numpy':time.sleep(30)
    return original(name,*args,**kwargs)
builtins.__import__=delayed
sys.argv=['runtime',sys.argv[1]]
runpy.run_module('simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime',run_name='__main__')
'''
            child=subprocess.Popen([sys.executable,'-c',code,str(request)],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            try:
                progress=output.with_suffix('.progress.json')
                deadline=time.monotonic()+10
                while not progress.is_file() and child.poll() is None and time.monotonic()<deadline:time.sleep(.02)
                self.assertTrue(progress.is_file())
                child.kill();child.communicate(timeout=3)
                result=json.loads(progress.read_text(encoding='utf-8'))
                self.assertEqual(result['stage'],'importing_runtime')
                self.assertEqual(result['recorded_samples'],0)
                self.assertIsNone(result['last_recorded_time_s'])
                self.assertFalse(result['certified'])
                self.assertFalse(output.exists())
                self.assertEqual(list(Path(directory).rglob('prefix.json')),[])
            finally:
                if child.poll() is None:child.kill();child.communicate(timeout=3)


if __name__=='__main__':unittest.main()
