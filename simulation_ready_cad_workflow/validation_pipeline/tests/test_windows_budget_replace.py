"""Real Windows reader sharing conflicts must not corrupt budget snapshots."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ..slim_harness import construction_budget as budget


@unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics')
class WindowsBudgetReplaceTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3] / '.runtime_runs/windows_budget_replace_tests'
        root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'construction_budget.json'
        self.path.write_bytes(b'{"active_seconds":12.5}\n')
        self.before = self.path.read_bytes()
        self.reader = subprocess.Popen([sys.executable, '-c',
            'import sys; f=open(sys.argv[1],"rb"); print("READY",flush=True); sys.stdin.read(1); f.close()',
            str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.addCleanup(self.close_reader)
        self.assertEqual(self.reader.stdout.readline().strip(), 'READY')

    def close_reader(self):
        if self.reader.poll() is None:
            self.reader.communicate('x', timeout=5)
        else:
            for stream in (self.reader.stdin, self.reader.stdout, self.reader.stderr):
                if stream is not None and not stream.closed:
                    stream.close()

    def test_transient_reader_allows_one_complete_replacement_without_resetting_values(self):
        replace = os.replace
        calls = []
        release = []

        def attempt(source, target):
            calls.append(1)
            try:
                return replace(source, target)
            except OSError:
                if not release:
                    self.assertEqual(self.path.read_bytes(), self.before)
                    timer = threading.Timer(.15, self.close_reader)
                    release.append(timer)
                    timer.start()
                raise

        state = {'active_seconds': 12.5, 'limit_seconds': 10800, 'sessions': {}}
        try:
            with patch.object(budget.os, 'replace', side_effect=attempt):
                budget._write(self.path, state)
        finally:
            for timer in release:
                timer.join(5)
        self.assertGreater(len(calls), 1)
        self.assertEqual(json.loads(self.path.read_text()), state)
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])

    def test_persistent_denial_is_bounded_and_preserves_old_complete_ledger(self):
        started = time.monotonic()
        with self.assertRaises(PermissionError):
            budget._write(self.path, {'active_seconds': 999})
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 1.8)
        self.assertLess(elapsed, 5)
        self.assertEqual(self.path.read_bytes(), self.before)
        self.assertEqual(list(self.path.parent.glob('*.tmp')), [])


if __name__ == '__main__':
    unittest.main()
