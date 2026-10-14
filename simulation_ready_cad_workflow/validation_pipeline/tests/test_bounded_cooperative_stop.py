"""Owned synthetic children only: latched limits and bounded controller closeout."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ..runtime.bounded_subprocess import BoundedSubprocessError, run_bounded_subprocess


class BoundedCooperativeStopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stop = self.root / 'owned-stop'
        self.partial = self.root / 'partial.txt'
        self.processes = []

    def request(self, process):
        self.processes.append(process)
        self.stop.touch()

    def cooperative_command(self):
        script = ('import pathlib,sys,time\nstop=pathlib.Path(sys.argv[1]);partial=pathlib.Path(sys.argv[2])\n'
                  'while not stop.exists():time.sleep(.01)\npartial.write_text("PRESERVED_PARTIAL")\nprint("CLEANUP_COMPLETE",flush=True)\n')
        return [sys.executable, '-I', '-c', script, str(self.stop), str(self.partial)]

    def test_zero_exit_during_grace_still_raises_latched_timeout_and_preserves_partial(self):
        with self.assertRaises(BoundedSubprocessError) as caught:
            run_bounded_subprocess(self.cooperative_command(), timeout_seconds=.15, max_rss_bytes=None,
                cooperative_stop=self.request, cleanup_grace_seconds=3)
        error = caught.exception
        self.assertEqual(error.code, 'BOUNDED_SUBPROCESS_TIMEOUT')
        self.assertEqual(self.partial.read_text(), 'PRESERVED_PARTIAL')
        self.assertEqual(self.processes[0].returncode, 0)
        self.assertTrue(error.cooperative_cleanup['root_and_pipes_closed_during_grace'])
        self.assertFalse(error.cooperative_cleanup['fallback_used'])
        self.assertGreaterEqual(error.wall_seconds_including_cleanup, error.cooperative_cleanup['decision_wall_seconds'])

    def test_rss_limit_is_latched_before_a_successful_cooperative_exit(self):
        with patch('simulation_ready_cad_workflow.validation_pipeline.runtime.bounded_subprocess.SystemProbe.process_tree_rss_bytes', return_value=1000), \
                self.assertRaises(BoundedSubprocessError) as caught:
            run_bounded_subprocess(self.cooperative_command(), timeout_seconds=5, max_rss_bytes=1,
                cooperative_stop=self.request, cleanup_grace_seconds=3)
        self.assertEqual(caught.exception.code, 'BOUNDED_SUBPROCESS_RSS')
        self.assertEqual(caught.exception.cooperative_cleanup['latched_limit'], 'BOUNDED_SUBPROCESS_RSS')
        self.assertEqual(self.processes[0].returncode, 0)
        self.assertEqual(self.partial.read_text(), 'PRESERVED_PARTIAL')

    def test_ignored_stop_uses_existing_scoped_fallback_after_finite_grace(self):
        with self.assertRaises(BoundedSubprocessError) as caught:
            run_bounded_subprocess([sys.executable, '-I', '-c', 'import time;time.sleep(60)'],
                timeout_seconds=.15, max_rss_bytes=None, cooperative_stop=self.request, cleanup_grace_seconds=.1)
        self.assertEqual(caught.exception.code, 'BOUNDED_SUBPROCESS_TIMEOUT')
        self.assertTrue(caught.exception.cooperative_cleanup['fallback_used'])
        self.assertIsNotNone(self.processes[0].poll())
        self.assertFalse(self.partial.exists())

    def test_callback_failure_cannot_suppress_the_original_limit(self):
        def rejected(process):
            self.processes.append(process)
            raise OSError('Synthetic stop-marker failure')
        with self.assertRaises(BoundedSubprocessError) as caught:
            run_bounded_subprocess([sys.executable, '-I', '-c', 'import time;time.sleep(60)'],
                timeout_seconds=.1, max_rss_bytes=None, cooperative_stop=rejected, cleanup_grace_seconds=.1)
        self.assertEqual(caught.exception.code, 'BOUNDED_SUBPROCESS_TIMEOUT')
        self.assertEqual(caught.exception.cooperative_cleanup['callback_error_type'], 'OSError')
        self.assertIsNotNone(self.processes[0].poll())

    def test_no_callback_retains_existing_failure_contract(self):
        with self.assertRaises(BoundedSubprocessError) as caught:
            run_bounded_subprocess([sys.executable, '-I', '-c', 'import time;time.sleep(60)'],
                timeout_seconds=.1, max_rss_bytes=None)
        self.assertEqual(caught.exception.code, 'BOUNDED_SUBPROCESS_TIMEOUT')
        self.assertFalse(hasattr(caught.exception, 'cooperative_cleanup'))


if __name__ == '__main__':
    unittest.main()
