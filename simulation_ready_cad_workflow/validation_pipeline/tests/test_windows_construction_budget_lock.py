"""Real Windows two-process lock contention on an owned temporary budget only."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest

from ..slim_harness.construction_budget import initialize_budget


@unittest.skipUnless(os.name == 'nt', 'Windows mandatory byte-range locking')
class WindowsBudgetLockTests(unittest.TestCase):
    def line(self, process):
        found=queue.Queue()
        threading.Thread(target=lambda:found.put(process.stdout.readline()),daemon=True).start()
        return found.get(timeout=10).strip()

    def exercise(self, *, empty):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)
            lock=project/'construction_budget.json.lock'
            before=None
            if not empty:
                budget=initialize_budget(project)
                budget._transaction(lambda state,now:state.update(active_seconds=17.125,excluded_seconds=3.0))
                before=json.loads(budget.path.read_text())
            holder_source='''import msvcrt,os,sys
from pathlib import Path
with Path(sys.argv[1]).open('a+b') as stream:
 stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_LOCK,1)
 if os.fstat(stream.fileno()).st_size==0:stream.write(b'0');stream.flush()
 print('LOCKED',flush=True)
 sys.stdin.readline()
 stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
'''
            waiter_source='''import json,sys
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_budget import _locked,ProjectBudget
path=Path(sys.argv[1])
try:
 with path.open('a+b') as stream:stream.seek(0);stream.read(1)
except PermissionError as error:print('LEGACY_READ_BLOCKED='+str(error.errno),flush=True)
else:raise AssertionError('The original mandatory-lock failure was not reproduced')
if sys.argv[2]=='empty':
 with _locked(path):print('ACQUIRED',flush=True)
else:
 result=ProjectBudget(path.parent).snapshot()
 print('ACQUIRED',flush=True);print(json.dumps(result),flush=True)
'''
            cwd=Path(__file__).resolve().parents[3]
            holder=waiter=None
            try:
                holder=subprocess.Popen([sys.executable,'-c',holder_source,str(lock)],cwd=cwd,
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                self.assertEqual(self.line(holder),'LOCKED')
                waiter=subprocess.Popen([sys.executable,'-c',waiter_source,str(lock),'empty' if empty else 'ledger'],
                    cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                denied=self.line(waiter);self.assertEqual(denied,'LEGACY_READ_BLOCKED=13')
                ready=queue.Queue()
                threading.Thread(target=lambda:ready.put(waiter.stdout.readline()),daemon=True).start()
                with self.assertRaises(queue.Empty):ready.get(timeout=.2)
                self.assertIsNone(waiter.poll())
                holder.stdin.write('release\n');holder.stdin.flush()
                self.assertEqual(ready.get(timeout=10).strip(),'ACQUIRED')
                if not empty:
                    snapshot=json.loads(self.line(waiter))
                    self.assertEqual(snapshot['active_seconds'],17.125)
                    self.assertEqual(snapshot['excluded_seconds'],3.0)
                self.assertEqual(waiter.wait(timeout=10),0,waiter.stderr.read())
                self.assertEqual(holder.wait(timeout=10),0,holder.stderr.read())
                self.assertEqual(lock.read_bytes(),b'0')
                if not empty:
                    after=json.loads((project/'construction_budget.json').read_text())
                    for key in ('config','active_seconds','excluded_seconds','excluded_by_reason','pause','created_at_epoch'):
                        self.assertEqual(after[key],before[key],key)
                print('WINDOWS_LOCK_PROOF='+json.dumps({'empty_initial_file':empty,'legacy_read_errno':13,
                    'new_waiter_waited_for_owned_holder':True,'lock_byte_unchanged':True,'ledger_not_reset':not empty}))
            finally:
                for process in (holder,waiter):
                    if process is None:continue
                    if process.poll() is None:process.kill();process.wait(timeout=5)
                    for stream in (process.stdin,process.stdout,process.stderr):
                        if stream is not None:stream.close()

    def test_empty_lock_can_be_acquired_before_initialization(self):
        self.exercise(empty=True)

    def test_contending_budget_snapshot_waits_without_reading_locked_byte_or_resetting_ledger(self):
        self.exercise(empty=False)


if __name__=='__main__':unittest.main()
