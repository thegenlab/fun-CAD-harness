"""Bounded real hard-kill recovery of durable rows; no solver execution."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from ..slim_harness.trace_storage import (
    TraceStorageError, TraceWriter, open_trace, open_trace_prefix, trace_prefix_path,
)


CHILD = r'''
import json,os,struct,sys,time
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.trace_storage import TraceWriter,open_trace_prefix
root=Path(sys.argv[1]);mode=sys.argv[2];path=root/'trace.json'
writer=(TraceWriter(path,20000) if mode=='timed_subchunk' else
        TraceWriter(path,20000,flush_interval_seconds=60) if mode=='first_sample' else
        TraceWriter(path,11,chunk_rows=3,flush_interval_seconds=60))
def append(i):
    writer.append({'time_s':float(i),'body_poses':{},'value':100.+2*i},
                  {'time_s':float(i),'value':100.+2*i})
if mode in ('first_sample','timed_subchunk'):
    append(0)
    assert len(open_trace_prefix(path)['samples'])==1
    if mode=='timed_subchunk':
        append(1)
        # No further append or explicit flush: the timer must publish the
        # pending row while the simulated next native step is still waiting.
        deadline=time.monotonic()+5
        while len(open_trace_prefix(path)['samples'])<2:
            assert time.monotonic()<deadline,'Timed flush never published the second row'
            time.sleep(.02)
else:
    for i in range(3):append(i)
if mode=='partial_tail':
    for i in (3,4):append(i)
    # Simulate interruption inside the next block: sample/offset tail and
    # only its first signal column reach disk, without publishing a boundary.
    for stream in (writer.samples_file,writer.offsets_file):
        stream.flush();os.fsync(stream.fileno())
    writer.signals_file.seek(3*8)
    writer.signals_file.write(struct.pack('<2d',3.,4.))
    writer.signals_file.flush();os.fsync(writer.signals_file.fileno())
elif mode=='during_compaction':
    original_open=Path.open
    class PausedCompact:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def write(self,data):
            self.stream.write(data[:8]);self.stream.flush();os.fsync(self.stream.fileno())
            (root/'ready.json').write_text(json.dumps({'pid':os.getpid(),'stage':mode}))
            time.sleep(20)
        def __getattr__(self,name):return getattr(self.stream,name)
    def open_file(path,*args,**kwargs):
        stream=original_open(path,*args,**kwargs)
        return PausedCompact(stream) if path.name=='signals.compact.f64' else stream
    Path.open=open_file
    writer.finish_data()
    raise AssertionError('Parent should hard-kill during compact copy')
(root/'ready.json').write_text(json.dumps({'pid':os.getpid(),'stage':mode}))
time.sleep(20)
'''


class SlimTracePrefixTests(unittest.TestCase):
    def setUp(self):
        self.workspace=Path(__file__).resolve().parents[3]
        base=self.workspace/'.runtime_runs/four_gaps_pair_20260912/prefix_codex'
        base.mkdir(parents=True,exist_ok=True)
        # Keep the stopped writer's real files and execution evidence for review.
        self.root=Path(tempfile.mkdtemp(prefix=self._testMethodName+'_',dir=base))

    def kill_writer(self,mode):
        child=self.root/'writer.py';child.write_text(CHILD,encoding='utf-8')
        env=os.environ.copy();env['PYTHONPATH']=str(self.workspace)
        flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
        started=time.perf_counter()
        with (self.root/'child.log').open('w',encoding='utf-8') as log:
            sentinel=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],
                                      creationflags=flags,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            process=subprocess.Popen([sys.executable,str(child),str(self.root),mode],
                cwd=self.workspace,env=env,creationflags=flags,stdout=log,stderr=subprocess.STDOUT)
            try:
                deadline=time.monotonic()+8
                while not (self.root/'ready.json').is_file():
                    if process.poll() is not None:
                        self.fail('Writer exited before hard kill: '+(self.root/'child.log').read_text())
                    if time.monotonic()>=deadline:self.fail('Writer did not reach its bounded stop point')
                    time.sleep(.02)
                process.kill()  # This owned subprocess only; __exit__ cannot run.
                process.wait(timeout=2)
                evidence={'mode':mode,'writer_pid':process.pid,'writer_returncode':process.returncode,
                          'unrelated_pid':sentinel.pid,'unrelated_alive_after_kill':sentinel.poll() is None,
                          'wall_seconds':time.perf_counter()-started}
                (self.root/'kill_evidence.json').write_text(json.dumps(evidence,indent=2))
                self.assertNotEqual(0,process.returncode)
                self.assertTrue(evidence['unrelated_alive_after_kill'])
            finally:
                for owned in (process,sentinel):
                    if owned.poll() is None:owned.kill()
                    owned.wait(timeout=2)
        return self.root/'trace.json'

    def check_three_rows(self,path):
        view=open_trace_prefix(path)
        self.assertEqual('UNVERIFIED_PREFIX',view['status'])
        self.assertFalse(view['physical_pass'])
        self.assertEqual(3,len(view['samples']))
        self.assertEqual([0.,1.,2.],[sample['time_s'] for sample in view['samples']])
        self.assertEqual([102.,104.],[sample['value'] for sample in view['samples'][1:]])
        self.assertEqual([104.,102.,100.],[sample['value'] for sample in view['samples'][::-1]])
        self.assertEqual(104.,view['samples'][-1]['value'])
        np.testing.assert_array_equal([0.,1.,2.],view['signals']['time_s'])
        np.testing.assert_array_equal([100.,102.,104.],view['signals']['value'])
        self.assertEqual(11,view['storage']['column_stride_samples'])
        self.assertFalse(path.exists())
        self.assertFalse((path.with_suffix('.data')/'data.json').exists())
        with self.assertRaises(TraceStorageError):open_trace(trace_prefix_path(path))
        return view

    def test_hard_kill_after_complete_block_preserves_confirmed_rows(self):
        self.check_three_rows(self.kill_writer('complete_block'))

    def test_hard_kill_ignores_partially_written_next_block_and_corrupt_tail(self):
        path=self.kill_writer('partial_tail');view=self.check_three_rows(path)
        sample_path=path.with_suffix('.data')/'samples.jsonl'
        self.assertGreater(sample_path.stat().st_size,view['storage']['sample_byte_end'])
        with sample_path.open('ab') as stream:stream.write(b'{broken unconfirmed tail')
        self.check_three_rows(path)

    def test_hard_kill_during_final_compaction_keeps_original_prefix_layout(self):
        path=self.kill_writer('during_compaction');self.check_three_rows(path)
        self.assertEqual(8,(path.with_suffix('.data')/'signals.compact.f64').stat().st_size)

    def test_no_recorded_state_has_no_prefix_and_first_append_is_immediately_readable(self):
        path=self.root/'empty.json'
        with TraceWriter(path,9,chunk_rows=3) as writer:
            with self.assertRaisesRegex(TraceStorageError,'No readable confirmed'):open_trace_prefix(path)
            writer.append({'time_s':0.},{'time_s':0.,'value':100.})
            first=open_trace_prefix(path)
            self.assertEqual(1,len(first['samples']))
            self.assertFalse(first['physical_pass'])
            np.testing.assert_array_equal([100.],first['signals']['value'])
        self.assertEqual(1,len(open_trace_prefix(path)['samples']))

    def test_hard_kill_after_first_sample_before_any_full_chunk(self):
        path=self.kill_writer('first_sample')
        prefix=open_trace_prefix(path)
        self.assertEqual('UNVERIFIED_PREFIX',prefix['status'])
        self.assertFalse(prefix['physical_pass'])
        self.assertEqual(1,len(prefix['samples']))
        self.assertEqual(0.,prefix['samples'][0]['time_s'])
        np.testing.assert_array_equal([100.],prefix['signals']['value'])
        self.assertEqual(20000,prefix['storage']['column_stride_samples'])
        self.assertFalse(path.exists())

    def test_hard_kill_after_timer_flush_with_no_new_append_or_full_chunk(self):
        path=self.kill_writer('timed_subchunk')
        prefix=open_trace_prefix(path)
        self.assertEqual(2,len(prefix['samples']))
        self.assertEqual([0.,1.],[row['time_s'] for row in prefix['samples']])
        np.testing.assert_array_equal([0.,1.],prefix['signals']['time_s'])
        np.testing.assert_array_equal([100.,102.],prefix['signals']['value'])
        self.assertEqual('UNVERIFIED_PREFIX',prefix['status'])
        self.assertFalse(prefix['physical_pass'])
        self.assertEqual(20000,prefix['storage']['column_stride_samples'])
        self.assertFalse(path.exists())
        self.assertFalse((path.with_suffix('.data')/'data.json').exists())

    def test_normal_full_and_short_completion_still_use_verified_manifests(self):
        for count in (4,9):
            path=self.root/f'normal-{count}.json'
            with TraceWriter(path,9,chunk_rows=3,flush_interval_seconds=60) as writer:
                for i in range(count):writer.append({'time_s':float(i)},{'time_s':float(i),'value':100.+2*i})
                writer.commit({'physical_pass':False,'test_id':'data-only-fixture'})
                self.assertFalse(writer._flush_thread.is_alive())
            complete=open_trace(path);prefix=open_trace_prefix(trace_prefix_path(path))
            self.assertEqual(count,len(complete['samples']))
            self.assertEqual(list(complete['samples']),list(prefix['samples']))
            for name in ('time_s','value'):
                np.testing.assert_array_equal(complete['signals'][name],prefix['signals'][name])
            self.assertEqual(count*2*8,complete['storage']['files']['signals']['bytes'])
            self.assertEqual(count<9,(path.with_suffix('.data')/'signals.compact.f64').exists())


if __name__=='__main__':unittest.main()
