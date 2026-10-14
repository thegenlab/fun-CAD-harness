"""Formal sandbox selection, fail-closed absence and the local distribution mode."""
from importlib.util import find_spec
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..cold_start.worker import CodexCliWorker
from ..runtime.bounded_subprocess import BoundedSubprocessError
from ..slim_harness import cli, execution, execution_profile, modes, run_hooks
from ..slim_harness.sandbox_binding import bind_worker

FIND_SPEC = 'simulation_ready_cad_workflow.validation_pipeline.slim_harness.execution.find_spec'
HOOK_SPEC = 'simulation_ready_cad_workflow.validation_pipeline.slim_harness.run_hooks.find_spec'


FORMAL_INSTALLED = find_spec(
    'simulation_ready_cad_workflow.validation_pipeline.benchmark_sandbox.formal_compute') is not None


class ExecutionSeamTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.project = cli.create_project(self.root, 'machine', 'A slider must move.')

    @unittest.skipUnless(FORMAL_INSTALLED, 'This distribution has no formal sandbox')
    def test_installed_sandbox_is_always_selected(self):
        self.assertTrue(execution.formal_sandbox_required())
        self.assertEqual(execution.mode(), 'formal')
        with patch('simulation_ready_cad_workflow.validation_pipeline.benchmark_sandbox.formal_compute.run_compute',
                   return_value='formal') as formal:
            self.assertEqual(execution.run_compute(['python'], workspace=self.root, project=self.project,
                                                   inputs=[], outputs=[], timeout_seconds=5), 'formal')
        self.assertEqual(formal.call_args.kwargs['project'], self.project)

    def test_missing_sandbox_fails_closed_without_local_permission(self):
        runner = Mock()
        with patch(FIND_SPEC, return_value=None), patch.object(execution_profile, 'LOCAL_EXECUTION_ALLOWED', False):
            with self.assertRaises(BoundedSubprocessError) as raised:
                execution.run_compute(['python'], workspace=self.root, project=self.project,
                                      inputs=[], outputs=[], timeout_seconds=5, runner=runner)
            self.assertEqual(raised.exception.code, 'FORMAL_SANDBOX_CONFIGURATION')
            with self.assertRaises(BoundedSubprocessError):
                bind_worker(CodexCliWorker(), self.root, self.project, 'author')
        runner.assert_not_called()

    def test_local_distribution_runs_plain_bounded_commands_and_original_turns(self):
        runner = Mock(return_value='local')
        with patch(FIND_SPEC, return_value=None), patch.object(execution_profile, 'LOCAL_EXECUTION_ALLOWED', True):
            self.assertEqual(execution.mode(), 'local')
            with patch.dict('os.environ', {execution.CONTAINER_ENV: '1'}):
                self.assertEqual(execution.mode(), 'container')
            result = execution.run_compute(['python', '-V'], workspace=self.root, project=self.project,
                                           inputs=[self.project], outputs=[self.project], timeout_seconds=5,
                                           runner=runner)
            self.assertEqual(execution.geometry_inputs({'bodies': [{'source_file': str(self.project / 'a.step')}]}), [])
            worker = CodexCliWorker()
            self.assertIs(bind_worker(worker, self.root, self.project, 'author'), worker)
        self.assertEqual(result, 'local')
        self.assertEqual(runner.call_args.kwargs['cwd'], self.root)
        self.assertNotIn('inputs', runner.call_args.kwargs)
        self.assertIsNone(getattr(worker, 'benchmark_sandbox_context', None))

    def test_run_hook_skips_archive_when_package_is_absent(self):
        archive = self.root / '.runtime_runs/hf_sync'
        archive.mkdir(parents=True)
        (archive / 'config.json').write_text(json.dumps({'auto_upload': True}), encoding='utf-8')
        @run_hooks.after_run
        def run(workspace, project):
            return 'done'
        # Any import of the archive package would now fail the call.
        with patch(HOOK_SPEC, return_value=None), patch.dict(sys.modules, {run_hooks._ARCHIVE: None}):
            self.assertEqual(run(self.root, 'machine'), 'done')

    def test_local_runs_reject_the_no_visuals_ablation_they_cannot_enforce(self):
        with patch(FIND_SPEC, return_value=None), patch.object(execution_profile, 'LOCAL_EXECUTION_ALLOWED', True):
            with self.assertRaisesRegex(ValueError, 'requires the formal sandbox'):
                cli.start(self.root, 'ablation', 'A slider must move.', worker_provider='codex',
                          author_visual_policy='worker_no_visuals_v1')
        self.assertFalse((self.root / '.runtime_runs/projects/ablation').exists())

    def test_free_creation_is_the_only_mode(self):
        self.assertEqual(modes.available_modes(self.root), ('free',))
        with self.assertRaises(ValueError):
            modes.configure('benchmark')
        self.assertEqual(modes.configure('free'), {'mode': 'free'})


if __name__ == '__main__':
    unittest.main()
