"""Construction resource failures replan work; they cannot certify geometry."""
from __future__ import annotations

import json
import copy
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import trimesh

from ..slim_harness.construction_budget import BudgetConfig, initialize_budget
from ..slim_harness.construction_limits import (
    ConstructionLimitError, ConstructionLimits, check_geometry_counts,
    scene_preflight, simulation_extension, simulation_preflight, simulation_time_steps, run_simulation_bounded, workload_key,
)
from ..slim_harness.geometry import prepare_geometry


PREFIX = "simulation_ready_cad_workflow.validation_pipeline.slim_harness."


class ConstructionLimitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.policy = ConstructionLimits(simulation_timeout_seconds=.2)
        # Most cases need the configured limit to be the hard limit.
        ceiling = patch(PREFIX + "construction_limits.SIMULATION_EXTENSION_CEILING_SECONDS", 0.)
        ceiling.start()
        self.addCleanup(ceiling.stop)

    def scene(self, root=None, count=2):
        root = root or self.root
        root.mkdir(parents=True, exist_ok=True)
        mesh_scene = trimesh.Scene()
        for index in range(count):
            mesh = trimesh.creation.box()
            mesh.apply_translation([index * 2, 0, 0])
            mesh_scene.add_geometry(mesh, node_name=f"box{index}", geom_name=f"box{index}")
        (root / "source.glb").write_bytes(trimesh.exchange.gltf.export_glb(mesh_scene))
        scene = {"schema_version": "slim_scene_v1", "units": "m", "tests": [],
                 "bodies": [{"id": "body", "source": "source.glb", "fixed": False}], "joints": []}
        path = root / "scene.json"
        path.write_text(json.dumps(scene))
        return path

    def test_count_and_optional_triangle_caps_do_not_invent_runtime_estimates(self):
        with self.assertRaises(ConstructionLimitError) as raised:
            check_geometry_counts({"tube": 129}, self.policy)
        self.assertEqual("CONSTRUCTION_GEOMETRY_LIMIT", raised.exception.code)
        self.assertEqual(3, len(raised.exception.details["revision_strategies"]))
        with self.assertRaises(ConstructionLimitError):
            check_geometry_counts({"tube": 1}, {"max_geometry_triangles": 20}, triangle_counts={"tube": 21})
        result = check_geometry_counts({"tube": 1}, self.policy, triangle_counts={"tube": 200000})
        self.assertEqual(200000, result["triangle_count"])

    def test_task_count_checked_before_source_import(self):
        path = self.scene()
        scene = json.loads(path.read_text())
        scene["tests"] = [{"id": str(i)} for i in range(4)]
        path.write_text(json.dumps(scene))
        with patch(PREFIX + "geometry._mesh_source") as importer:
            result = prepare_geometry(path, self.root / "out", construction_limits=self.policy)
        importer.assert_not_called()
        self.assertEqual("CONSTRUCTION_TASK_TEST_LIMIT", result["errors"][0]["code"])
        with self.assertRaises(ConstructionLimitError):
            scene_preflight({"tasks": [1, 2, 3, 4]}, self.policy)

    def test_actual_inventory_checked_before_surface_processing_and_export(self):
        path = self.scene(count=2)
        with patch(PREFIX + "geometry._closed_mesh") as surfaces:
            result = prepare_geometry(path, self.root / "limited", construction_limits={"max_geometry_units_per_body": 1})
        surfaces.assert_not_called()
        self.assertEqual("CONSTRUCTION_GEOMETRY_LIMIT", result["errors"][0]["code"])
        self.assertEqual([], result["bodies"])
        self.assertFalse((self.root / "limited/source_components").exists())
        # No policy: the very same original geometry still follows the frozen
        # path, with no cap, replacement, hull or altered fidelity tolerance.
        legacy = prepare_geometry(path, self.root / "original")
        self.assertEqual([], legacy["errors"])
        self.assertEqual(2, len(legacy["bodies"][0]["source_components"]))
        self.assertNotIn("construction_geometry", legacy)

    def test_estimate_separates_measurement_from_proven_bound(self):
        scene, geometry, test = {"tests": []}, {"bodies": []}, {"id": "move", "duration_s": 2}
        key = workload_key(scene, geometry, test)
        measurement = {"workload_key": key, "elapsed_seconds": 20}
        result = simulation_preflight(scene, geometry, test, self.policy, measurements=[measurement])
        self.assertTrue(result["simplification_recommended"])
        self.assertEqual(2000, result["estimated_step_count"])
        with self.assertRaises(ConstructionLimitError) as raised:
            simulation_preflight(scene, geometry, test, self.policy, measurements=[{
                **measurement, "proven_minimum_seconds": 1, "lower_bound_basis": "measured mandatory fixed setup lower bound"}])
        self.assertEqual("SIMULATION_COMPLEXITY_EXCEEDS_TIMEOUT", raised.exception.code)
        changed = {**measurement, "workload_key": "different-scene", "proven_minimum_seconds": 1,
                   "lower_bound_basis": "does not apply"}
        # Simulation time is not charged, so a nearly spent budget does not shorten the run.
        result = simulation_preflight(scene, geometry, test, self.policy, measurements=[changed], remaining_seconds=.05)
        self.assertEqual(.2, result["effective_timeout_seconds"])
        self.assertNotIn("estimated_wall_seconds", result)
        finer = simulation_preflight(scene, geometry, {**test, "step_s": .0001}, self.policy)
        self.assertEqual(20000, finer["estimated_step_count"])

    def test_workload_key_survives_copy_but_not_changed_mechanics_or_tests(self):
        body = {"id": "a", "source_sha256": "source-content", "mesh_sha256": "mesh-content",
                "mesh_file": "/first/mesh.obj", "source_file": "/first/asset.step", "mass_kg": 1}
        scene = {"bodies": [body], "joints": [{"stiffness": 10}]}
        copied = copy.deepcopy(scene)
        copied["bodies"][0].update(mesh_file="/second/mesh.obj", source_file="/second/asset.step")
        test = {"id": "move", "duration_s": 1}
        self.assertEqual(workload_key(scene, scene, test), workload_key(copied, copied, test))
        copied["bodies"][0]["mass_kg"] = 2
        self.assertNotEqual(workload_key(scene, scene, test), workload_key(copied, copied, test))
        self.assertNotEqual(workload_key(scene, scene, test), workload_key(scene, scene, {**test, "duration_s": .5}))

    def test_matching_timeout_history_replans_without_running_again(self):
        history = self.root / "history"
        with self.assertRaises(ConstructionLimitError):
            run_simulation_bounded([sys.executable, "-c", "import time; time.sleep(10)"], scene={}, geometry={},
                                   test={"id": "repeat"}, policy=self.policy, history_dir=history)
        with patch(PREFIX + "construction_limits.run_bounded_subprocess") as runner:
            with self.assertRaises(ConstructionLimitError) as raised:
                run_simulation_bounded([sys.executable], scene={}, geometry={}, test={"id": "repeat"},
                                       policy=self.policy, history_dir=history)
        runner.assert_not_called()
        self.assertEqual("SIMULATION_UNCHANGED_TIMEOUT", raised.exception.code)

    def test_runtime_entrypoint_enforces_direct_request_and_avoids_recursive_guard(self):
        from ..slim_harness.runtime import main

        request = self.root / "request.json"
        request.write_text(json.dumps({"scene": {}, "test": {}, "output": str(self.root / "trace.json")}))
        with patch.object(sys, "argv", ["runtime", str(request)]):
            with patch.dict(os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": str(self.root),
                                         "SIMREADY_CONSTRUCTION_GUARDED_REQUEST": ""}):
                with patch(PREFIX + "construction_limits.run_runtime_entrypoint", return_value=2) as guard:
                    with self.assertRaises(SystemExit) as raised:
                        main()
                self.assertEqual(2, raised.exception.code)
                guard.assert_called_once_with(request.resolve(), self.root)
            with patch.dict(os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": str(self.root),
                                         "SIMREADY_CONSTRUCTION_GUARDED_REQUEST": str(request.resolve())}):
                with patch(PREFIX + "runtime.simulate", return_value={"fixture": "native"}) as native:
                    with patch("builtins.print"):
                        main()
                native.assert_called_once()

    def test_imported_simulate_is_bounded_even_with_a_cli_marker_and_keeps_legacy_path(self):
        from ..slim_harness.runtime import simulate

        scene, test, output = {}, {"duration_s": .001}, self.root / "trace.json"
        with patch.dict(os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": str(self.root),
                                     "SIMREADY_CONSTRUCTION_GUARDED_REQUEST": "some-other-request.json"}):
            with patch(PREFIX + "construction_limits.run_imported_simulation", return_value={"bounded": True}) as bounded:
                with patch(PREFIX + "runtime.run_fixed_attempts") as native:
                    self.assertEqual({"bounded": True}, simulate(scene, test, output))
            bounded.assert_called_once_with(scene, test, output, self.root)
            native.assert_not_called()
        with patch.dict(os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": ""}):
            with patch(PREFIX + "construction_limits.run_imported_simulation") as bounded:
                with patch(PREFIX + "runtime.run_fixed_attempts", return_value={"original": True}) as native:
                    self.assertEqual({"original": True}, simulate(scene, test, output))
            bounded.assert_not_called()
            native.assert_called_once()

    def test_real_imported_simulation_returns_native_trace_without_recursion(self):
        executable = os.environ.get("SIMREADY_PYCHRONO_EXECUTABLE")
        if not executable or not Path(executable).is_file():
            self.skipTest("Set SIMREADY_PYCHRONO_EXECUTABLE for the actual native child probe")
        project, _budget = self.project(simulation_timeout_seconds=10)
        source = self.scene(self.root / "native-source", count=1)
        geometry = prepare_geometry(source, self.root / "native-geometry")
        self.assertEqual([], geometry["errors"])
        geometry["bodies"][0]["fixed"] = True
        test = {"id": "native-bound", "duration_s": .001, "assertions": [
            {"id": "duration", "purpose": "Native child executed", "expression": "last(signal('time_s')) >= 0.001"}]}
        output = self.root / "native.trace.json"
        request = self.root / "native-request.json"
        request.write_text(json.dumps({"scene": {"bodies": geometry["bodies"], "joints": [],
                                               "tests": [test], "gravity_m_s2": [0, 0, 0]},
                                       "test": test, "output": str(output)}))
        script = """import json,sys
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime import simulate
request=json.loads(Path(sys.argv[1]).read_text())
result=simulate(request['scene'],request['test'],Path(request['output']))
assert isinstance(result,dict) and result['physical_pass'], result
print(result['test_id'])
"""
        workspace = Path(__file__).resolve().parents[3]
        environment = dict(os.environ, SIMREADY_CONSTRUCTION_PROJECT=str(project), PYTHONPATH=str(workspace))
        environment.pop("SIMREADY_CONSTRUCTION_GUARDED_REQUEST", None)
        child = subprocess.run([executable, "-c", script, str(request)], cwd=workspace,
                               env=environment, capture_output=True, text=True, timeout=20)
        self.assertEqual(0, child.returncode, child.stderr)
        self.assertIn("native-bound", child.stdout)
        self.assertTrue(output.is_file())
        self.assertEqual(1, len(list(self.root.glob("native.trace.construction-*/request.json"))))
        self.assertEqual(1, len(list((project / "construction-simulations").glob("*.json"))))

    def test_exhausted_budget_does_not_start_a_process(self):
        with patch(PREFIX + "construction_limits.run_bounded_subprocess") as runner:
            with self.assertRaises(ConstructionLimitError):
                run_simulation_bounded([sys.executable], scene={}, geometry={}, test={},
                                       policy=self.policy, remaining_seconds=0)
        runner.assert_not_called()

    def test_hard_timeout_kills_descendants_and_records_actual_cost(self):
        marker = self.root / "child-survived"
        child = f"import time,pathlib; time.sleep(1); pathlib.Path({str(marker)!r}).write_text('bad')"
        parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(20)"
        report = self.root / "cost.json"
        with self.assertRaises(ConstructionLimitError) as raised:
            run_simulation_bounded([sys.executable, "-c", parent], scene={}, geometry={}, test={"id": "hang"},
                                   policy=self.policy, report_path=report)
        self.assertEqual("SIMULATION_HARD_TIMEOUT", raised.exception.code)
        value = json.loads(report.read_text())
        self.assertEqual("timeout", value["status"])
        self.assertGreaterEqual(value["elapsed_seconds"], .2)
        self.assertLess(value["elapsed_seconds"], 3)
        time.sleep(1.1)
        self.assertFalse(marker.exists())

    def test_limit_extends_to_twice_then_thirty_minutes(self):
        with patch(PREFIX + "construction_limits.SIMULATION_EXTENSION_CEILING_SECONDS", 1800.):
            self.assertEqual([(300., 600., 1800.), (600., 1200., 1800.), (1200., 1800.), (1800.,), (3600.,)],
                             [simulation_time_steps(seconds) for seconds in (300, 600, 1200, 1800, 3600)])
            self.assertEqual([None, 1200., 1800., 1800.],
                             [simulation_extension(600, elapsed)["extended_to_seconds"] for elapsed in (599, 900, 1500, 1800.5)])

    def test_unfinished_simulation_keeps_running_until_the_extended_limit(self):
        policy = ConstructionLimits(simulation_timeout_seconds=.3)
        with patch(PREFIX + "construction_limits.SIMULATION_EXTENSION_CEILING_SECONDS", 3.):
            report = self.root / "extended.json"
            result = run_simulation_bounded([sys.executable, "-c", "import time; time.sleep(1)"], scene={}, geometry={},
                                            test={"id": "slow"}, policy=policy, report_path=report)
            self.assertEqual(0, result.returncode)
            value = json.loads(report.read_text())
            self.assertEqual(("finished", .3, [.3, .6, 3.], 3., 3.),
                             (value["status"], value["configured_timeout_seconds"], value["timeout_steps_seconds"],
                              value["extended_to_seconds"], value["effective_timeout_seconds"]))
            started = time.monotonic()
            with self.assertRaises(ConstructionLimitError) as raised:
                run_simulation_bounded([sys.executable, "-c", "import time; time.sleep(20)"], scene={}, geometry={},
                                       test={"id": "hang"}, policy=policy)
            self.assertEqual(("SIMULATION_HARD_TIMEOUT", 3.), (raised.exception.code, raised.exception.details["effective_timeout_seconds"]))
            self.assertGreaterEqual(time.monotonic() - started, 3)

    def test_simulation_wall_time_is_excluded_from_the_construction_budget(self):
        project = self.root / ".runtime_runs/projects/excluded"
        project.mkdir(parents=True)
        budget = initialize_budget(project, BudgetConfig(hours=1))
        with budget.activity("workflow"):
            run_simulation_bounded([sys.executable, "-c", "import time; time.sleep(1.5)"], scene={}, geometry={},
                                   test={"id": "free"}, policy=ConstructionLimits(simulation_timeout_seconds=10),
                                   history_dir=project / "construction-simulations")
        snapshot = budget.snapshot()
        self.assertGreater(snapshot["excluded_by_reason"]["simulation"], 1.2)
        self.assertLess(snapshot["active_seconds"], snapshot["wall_seconds"] - 1.2)

    def test_successful_bounded_process_preserves_outputs(self):
        result = run_simulation_bounded([sys.executable, "-c", "print('native evidence')"],
            scene={}, geometry={}, test={"id": "short"}, policy={"simulation_timeout_seconds": 2})
        self.assertEqual(0, result.returncode)
        self.assertIn(b"native evidence", result.stdout)

    def project(self, **kwargs):
        project = self.root / ".runtime_runs/projects/construction"
        project.mkdir(parents=True)
        (project / "workflow.json").write_text('{"request":{"construction":true}}')
        budget = initialize_budget(project, BudgetConfig(**kwargs))
        return project, budget

    def test_author_tool_inherits_project_charges_work_and_records_replan_evidence(self):
        from ..slim_harness.author_tools import run_tool

        project, budget = self.project(max_geometry_units_per_body=1)
        path = self.scene(self.root / "draft")
        with patch.dict(os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": str(project)}):
            result = run_tool("preview", path, workspace=self.root)
        self.assertEqual("NEEDS_SIMPLIFICATION", result["status"], result)
        self.assertGreater(budget.snapshot()["active_seconds"], 0)
        manifests = list((project / "construction-evidence").glob("*.json"))
        self.assertEqual(1, len(manifests))
        evidence = json.loads(manifests[0].read_text())
        self.assertEqual(str(path), evidence["source"])
        self.assertTrue(Path(evidence["result_file"]).is_file())
        self.assertFalse(result["certified"])

    def test_author_tool_budget_expiry_requires_summary_and_records_partial_outcome(self):
        from ..slim_harness.author_tools import run_tool

        project, budget = self.project()
        budget._transaction(lambda state, now: state.update(active_seconds=7200))
        path = self.scene(project / "runs/r01/design")
        with patch.dict(os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": ""}):
            with patch(PREFIX + "author_tools.prepare_geometry") as prepare:
                result = run_tool("check", path, workspace=self.root)
        self.assertEqual("BUDGET_EXHAUSTED", result["status"], result)
        self.assertEqual("write_completion_summary", result["required_action"])
        prepare.assert_not_called()
        self.assertEqual(1, len(list((project / "construction-evidence").glob("*.json"))))


if __name__ == "__main__":
    unittest.main()
