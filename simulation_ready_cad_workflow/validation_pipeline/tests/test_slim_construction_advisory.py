"""Optional review advice reaches live author tools without a new review gate."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness.construction import policy_prompt, review_notes
from ..slim_harness.construction_budget import initialize_budget


PREFIX = 'simulation_ready_cad_workflow.validation_pipeline.slim_harness.'


class ConstructionAdvisoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.project = self.workspace / '.runtime_runs/projects/advisory'
        self.project.mkdir(parents=True)
        (self.project / 'workflow.json').write_text('{"request":{}}')
        self.notes = self.project / 'construction_review_notes.md'

    def test_missing_and_frozen_notes_are_excluded(self):
        self.assertIsNone(review_notes(None))
        self.notes.write_text('Consider the linkage clearance.')
        self.assertIsNone(review_notes(self.project))  # Evaluation has no ledger.
        self.assertEqual('', policy_prompt(self.project))
        initialize_budget(self.project)
        (self.project / 'workflow.json').write_text('{"request":{"frozen_experiment_sha256":"frozen"}}')
        self.assertIsNone(review_notes(self.project))
        (self.project / 'workflow.json').write_text('{"request":{}}')
        self.notes.unlink()
        self.assertIsNone(review_notes(self.project))
        self.assertNotIn('OPTIONAL CONSTRUCTION REVIEW ADVICE', policy_prompt(self.project))

    def test_note_is_bounded_advisory_and_prompt_preserves_policy(self):
        initialize_budget(self.project)
        text = '先检查被动传动。' * 1000
        self.notes.write_text(text, encoding='utf-8')
        value = review_notes(self.project)
        self.assertEqual(4000, len(value['text']))
        self.assertTrue(value['truncated'])
        self.assertTrue(value['advisory_only'])
        self.assertIn('not a verdict', value['scope'])
        prompt = policy_prompt(self.project)
        self.assertIn(value['text'], prompt)
        self.assertIn('Do not silently remove core functions', prompt)
        self.assertEqual(text, self.notes.read_text(encoding='utf-8'))

    def test_tool_response_includes_latest_note_only_for_construction(self):
        from ..slim_harness.author_tools import run_tool

        source = self.project / 'runs/r01/design/scene.json'
        source.parent.mkdir(parents=True)
        source.write_text('{"bodies":[],"tests":[]}')
        self.notes.write_text('Check the passive output before adding cosmetic detail.')
        geometry = {'errors': [], 'bodies': [], 'model_file': 'fixture.glb', 'source_bindings': []}
        for construction in (False, True):
            if construction:
                initialize_budget(self.project)
                (self.project / 'workflow.json').write_text('{"request":{"construction":true}}')
            with patch.dict(os.environ, {'SIMREADY_CONSTRUCTION_PROJECT': ''}):
                with patch(PREFIX + 'author_tools.prepare_geometry', return_value=geometry):
                    with patch(PREFIX + 'author_tools.render_views', return_value=self.workspace / 'fixture.png'):
                        result = run_tool('preview', source, workspace=self.workspace)
            self.assertEqual('PREVIEW_READY', result['status'], result)
            self.assertFalse(result['certified'])
            self.assertEqual(construction, 'construction_review_notes' in result)
            if construction:
                self.assertIn('passive output', result['construction_review_notes']['text'])

    def test_cli_compact_output_keeps_advisory(self):
        from ..slim_harness.cli import main

        note = {'text': 'Inspect the axle clearance.', 'advisory_only': True}
        result = {'status': 'CHECKED', 'certified': False, 'construction_review_notes': note}
        with patch(PREFIX + 'author_tools.run_tool', return_value=result):
            with patch('builtins.print') as printed:
                code = main(['--workspace', str(self.workspace), 'check', 'scene.json'])
        self.assertEqual(0, code)
        self.assertEqual(note, json.loads(printed.call_args.args[0])['construction_review_notes'])

    def test_cli_forwards_geometry_caps_and_retains_defaults(self):
        from ..slim_harness.cli import main

        for overrides, expected in (([], (2048, 128)),
                                    (['--max-geometry-units', '512', '--max-geometry-units-per-body', '32'], (512, 32))):
            with patch(PREFIX + 'cli.start', return_value={'status': 'RUNNING'}) as start:
                with patch('builtins.print'):
                    code = main(['--workspace', str(self.workspace), 'create', '--project', 'example',
                                 '--brief', 'A folding fixture', *overrides])
            self.assertEqual(0, code)
            config = start.call_args.kwargs['budget_config']
            self.assertEqual(expected, (config.max_geometry_units, config.max_geometry_units_per_body))


if __name__ == '__main__':
    unittest.main()
