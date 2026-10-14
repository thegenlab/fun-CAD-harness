"""Expression interface errors must be found before CAD or physical execution."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..slim_harness.contracts import DeliveryError, load_scene
from ..slim_harness.measurements import evaluate_assertions, validate_expression
from ..slim_harness.workflow import SlimWorkflow, digest


# Exact D r04 manual_winding_stores_energy expression; no design is imported.
D_R04_WINDING = 'span(before(signal("time_wind.coordinate"), 8.1)) > 22 and last(signal("input.time_wind.work_j")) > 1.0'


class SlimExpressionPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def scene(self, expression, path=None):
        value = {"schema_version": "slim_scene_v1", "units": "mm",
                 "bodies": [{"id": "clock", "source": "clock.step", "fixed": False}],
                 "tests": [{"id": "wind_and_run", "purpose": "Check the submitted operation", "duration_s": 100,
                            "assertions": [{"id": "manual_winding_stores_energy", "purpose": "Measure the operation",
                                            "expression": expression}]}]}
        target = path or self.root / "scene.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(value), encoding="utf-8")
        return target, value

    def test_actual_r04_expression_rejected_before_geometry_and_physics(self):
        project = self.root / "project"
        path, _ = self.scene(D_R04_WINDING, project / "runs/r01/design/scene.json")
        criteria = project / "criteria.json"
        criteria.write_text("{}", encoding="utf-8")
        workflow = SlimWorkflow.__new__(SlimWorkflow)
        workflow.project = project
        workflow.state = {"revision": 1}
        workflow.criteria_sha256 = digest(criteria)
        workflow.publish = Mock()
        prefix = "simulation_ready_cad_workflow.validation_pipeline.slim_harness.workflow."
        with patch(prefix + "prepare_geometry", side_effect=AssertionError("Geometry must not run")) as geometry, \
             patch(prefix + "run_bounded_subprocess", side_effect=AssertionError("Physics must not run")) as physics:
            with self.assertRaisesRegex(DeliveryError, "test 'wind_and_run', assertion 'manual_winding_stores_energy'.*BoolOp"):
                workflow.run_revision()
        geometry.assert_not_called()
        physics.assert_not_called()
        self.assertEqual(D_R04_WINDING, json.loads(path.read_text())["tests"][0]["assertions"][0]["expression"])

    def test_documented_expression_is_valid_without_future_physical_signals(self):
        expression = "integral(window(abs(signal('future_speed')), 10, 20)) > 1"
        path, original = self.scene(expression)
        self.assertEqual(original, load_scene(path))

    def test_valid_but_false_assertion_remains_a_runtime_result(self):
        path, _ = self.scene("last(signal('measured')) > 10")
        loaded = load_scene(path)
        rows = evaluate_assertions(loaded["tests"][0], {"time_s": [0, 1], "measured": [0, 2]})
        self.assertFalse(rows[0]["passed"])
        self.assertNotIn("error", rows[0])

    def test_unknown_function_and_malformed_syntax_are_delivery_errors(self):
        for expression in ("unprovided_measure(signal('x')) > 0", "last(signal('x')) >"):
            with self.subTest(expression=expression):
                path, _ = self.scene(expression)
                with self.assertRaisesRegex(DeliveryError, "Invalid measurement expression in test 'wind_and_run'"):
                    load_scene(path)

    def test_runtime_keeps_its_defensive_language_check(self):
        rows = evaluate_assertions({"assertions": [{"id": "original", "expression": D_R04_WINDING}]}, {"time_s": [0, 1]})
        self.assertFalse(rows[0]["passed"])
        self.assertIn("Unsupported measurement syntax: BoolOp", rows[0]["error"])

    def test_existing_language_is_not_extended_to_boolean_operators(self):
        with self.assertRaisesRegex(ValueError, "Unsupported measurement syntax: BoolOp"):
            validate_expression("1 < 2 and 3 < 4")


if __name__ == "__main__":
    unittest.main()
