"""Final CLI completion must distinguish recovered transport errors from failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..cold_start.worker import CodexCliWorker

MODULE = 'simulation_ready_cad_workflow.validation_pipeline.cold_start.worker.'


class CodexCompletionTests(unittest.TestCase):
    def run_fixture(self, *, completed=True, exit_code=0, valid_response=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(command, **kwargs):
                events = [
                    {'type': 'thread.started', 'thread_id': 'same-author-session'},
                    {'type': 'error', 'message': 'Reconnecting... 2/5 (request timed out)'},
                    {'type': 'turn.completed', 'usage': {'input_tokens': 1, 'output_tokens': 1}}
                    if completed else {'type': 'turn.failed', 'error': {'message': 'Disconnected'}},
                ]
                for event in events:
                    kwargs['stdout'].write(json.dumps(event) + '\n')
                response = {'status': 'ready', 'summary': 'Delivery ready', 'evidence_paths': [], 'issues': []}
                Path(command[command.index('-o') + 1]).write_text(json.dumps(response if valid_response else {}))
                return Mock(returncode=exit_code, stdin=None, poll=Mock(return_value=exit_code))
            watch = Mock(arrivals=[])
            with patch(MODULE + 'subprocess.Popen', side_effect=launch), patch(MODULE + 'TokenWatch', return_value=watch):
                return CodexCliWorker.run_turn.__wrapped__(CodexCliWorker(), root, 'fixture', root/'turn', 5)

    def test_completed_valid_response_survives_recovered_connection_error(self):
        result = self.run_fixture()
        self.assertEqual('ready', result['status'])
        self.assertEqual('same-author-session', result['session_id'])
        self.assertNotIn('provider_error', result)

    def test_unrecovered_turn_failure_is_preserved(self):
        self.assertEqual('provider_error', self.run_fixture(completed=False)['status'])

    def test_nonzero_exit_is_not_success(self):
        self.assertEqual('provider_error', self.run_fixture(exit_code=1)['status'])

    def test_completed_turn_still_requires_valid_response(self):
        result = self.run_fixture(valid_response=False)
        self.assertEqual('provider_error', result['status'])
        self.assertEqual('response', result['provider_error']['kind'])
