"""Small regressions for real scene, measurement, and audit failures."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from ..slim_harness.audit import audit_motion
from ..slim_harness.contracts import DeliveryError, compile_scene, load_scene
from ..slim_harness.geometry import prepare_geometry
from ..slim_harness.measurements import evaluate, evaluate_assertions
from ..slim_harness.trace_storage import open_trace


CHRONO_PYTHON = Path(os.environ.get("SIMREADY_TEST_CHRONO_PYTHON", str(Path.home() / ".conda/envs/chrono_py312_projectchrono/python.exe")))


def _pose(position, quaternion=(1, 0, 0, 0)):
    return {"position_abs_m": list(position), "rotation_abs_quat_wxyz": list(quaternion)}


class SlimHarnessTests(unittest.TestCase):
    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[3]
        runtime = self.workspace / ".runtime_runs" / "slim_harness_20260907" / "test_fixtures"
        runtime.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        environment = mock.patch.dict(os.environ, {"SIMREADY_STEP_SHAPE_CACHE_DIR": str(self.root / "step_cache")})
        environment.start()
        self.addCleanup(environment.stop)

    def _prepare(self, scene, name="case"):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps(scene), encoding="utf-8")
        result = prepare_geometry(path, self.root / name)
        self.assertEqual([], result["errors"])
        return path, result

    def test_nonfinite_ratio_cannot_pass_a_measurement(self):
        signals = {"time_s": [0, 1], "distance": [1, 1], "speed": [0, 0]}
        for expression in (
            "max(signal('distance') / signal('speed')) > 1",
            "mean(signal('speed') / signal('speed')) != 1",
        ):
            with self.subTest(expression=expression):
                rows = evaluate_assertions({"assertions": [{"id": "ratio", "purpose": "real ratio", "expression": expression}]}, signals)
                self.assertFalse(rows[0]["passed"])
                self.assertIn("nonfinite", rows[0]["error"])
        self.assertTrue(evaluate("max(signal('distance') / signal('speed')) > .4",
                                 {**signals, "speed": [1, 2]})["passed"])

    def test_original_curved_surface_overlap_survives_mesh_candidate_filter(self):
        import xml.parsers.expat
        from build123d import Compound, Cylinder, Pos, Rot, export_step

        # Rotate a 200 mm cylinder away from its tessellation vertices.
        # Choose separation from the actual chords: mesh resolution can change
        # while this still isolates original-BRep overlap missed by the meshes.
        theta = 2 * np.pi / 5
        for baked_rotation in (False, True):
            with self.subTest(baked_rotation=baked_rotation):
                name = "baked" if baked_rotation else "pose_rotation"
                rotation = Rot(0, 0, np.degrees(theta) if baked_rotation else 0)
                source = self.root / f"{name}.step"
                export_step(Compound(children=[
                    Pos(-150, 0, 0) * rotation * Cylinder(100, 100),
                    Pos(150, 0, 0) * rotation * Cylinder(100, 100),
                ]), source)
                _, geometry = self._prepare({"schema_version": "slim_scene_v1", "units": "mm", "bodies": [
                    {"id": "a", "source": source.name, "solid_ids": ["solid_1"], "fixed": False},
                    {"id": "b", "source": source.name, "solid_ids": ["solid_2"], "fixed": False},
                ]}, name)
                q = [1, 0, 0, 0] if baked_rotation else [np.cos(theta / 2), 0, 0, np.sin(theta / 2)]
                matrix = trimesh.transformations.quaternion_matrix(q)[:3, :3]
                surfaces = [trimesh.load_mesh(body["mesh_file"], process=False).vertices @ matrix.T
                            for body in geometry["bodies"]]
                mesh_contact_separation = float(surfaces[0][:, 0].max() - surfaces[1][:, 0].min())
                self.assertLess(mesh_contact_separation, .2, "The fixture must expose curved-surface chord clearance")
                separation = (.2 + mesh_contact_separation) / 2
                sample = {"time_s": 0, "body_poses": {
                    "a": _pose([-separation / 2, 0, 0], q),
                    "b": _pose([separation / 2, 0, 0], q),
                }}
                manager = trimesh.collision.CollisionManager()
                for body in geometry["bodies"]:
                    pose = sample["body_poses"][body["id"]]
                    matrix = trimesh.transformations.quaternion_matrix(q)
                    matrix[:3, 3] = pose["position_abs_m"]
                    manager.add_object(body["id"], trimesh.load_mesh(body["mesh_file"]), transform=matrix)
                self.assertFalse(manager.in_collision_internal(), "This fixture must expose the curved-source gap, not an ordinary mesh intersection")
                report = audit_motion(geometry, [sample], self.root / f"audit-{name}.json")
                self.assertFalse(report["passed"], "Original STEP overlap cannot pass because its chord meshes do not intersect")
                self.assertGreater(report["max_overlap_m3"], 1e-10)
                self.assertEqual("original_solid_interpenetration", report["violations"][0]["kind"])

    def test_full_containment_only_at_an_intermediate_solver_state_fails(self):
        inner = trimesh.creation.box(extents=[.01, .01, .01])
        inner.apply_translation([.1, 0, 0])
        inner.export(self.root / "inner.obj")
        trimesh.creation.box(extents=[.03, .03, .03]).export(self.root / "outer.obj")
        _, geometry = self._prepare({"schema_version": "slim_scene_v1", "units": "m", "bodies": [
            {"id": "inner", "source": "inner.obj", "fixed": False},
            {"id": "outer", "source": "outer.obj", "fixed": True},
        ]})
        samples = [{"time_s": index * .001, "body_poses": {
            "inner": _pose([x, 0, 0]), "outer": _pose([0, 0, 0]),
        }} for index, x in enumerate((.1, 0, .1))]
        self.assertTrue(audit_motion(geometry, [samples[0], samples[-1]], self.root / "endpoints.json")["passed"])
        report = audit_motion(geometry, samples, self.root / "all-steps.json")
        self.assertFalse(report["passed"])
        self.assertEqual(.001, report["violations"][0]["time_s"])
        self.assertEqual(2, report["checked_solver_samples"])
        self.assertEqual(3, report["available_solver_samples"])

    def test_exact_clearance_proof_reuses_small_drift_and_expires_before_overlap(self):
        import xml.parsers.expat
        from build123d import Box, Cylinder, export_step

        export_step(Box(20,20,5)-Cylinder(5,6),self.root/"plate.step")
        export_step(Cylinder(4.95,5),self.root/"pin.step")
        _,geometry=self._prepare({"schema_version":"slim_scene_v1","units":"mm","bodies":[
            {"id":"plate","source":"plate.step","fixed":False},
            {"id":"pin","source":"pin.step","fixed":False},
        ]},"clearance")
        samples=[]
        for index,offset in enumerate((0,1e-6,3e-6,5e-6,1e-4)):
            # Both bodies also move and rotate together. The proof must use
            # relative motion, not assume that either world pose is stationary.
            q=[np.cos(index*.05/2),0,0,np.sin(index*.05/2)]
            rotation=trimesh.transformations.quaternion_matrix(q)[:3,:3]
            base=np.array([index*.1,index*.02,0.])
            samples.append({"time_s":index*.001,"body_poses":{
                "plate":_pose(base,q),"pin":_pose(base+rotation@np.array([offset,0,0]),q),
            }})
        separate=audit_motion(geometry,samples[:4],self.root/"clearance-pass.json")
        self.assertTrue(separate["passed"])
        self.assertEqual(4,separate["checked_solver_samples"])
        self.assertEqual(3,separate["cached_proof_samples"])
        self.assertEqual(1,separate["boolean_checks"])
        crossed=audit_motion(geometry,samples,self.root/"clearance-crossed.json")
        self.assertFalse(crossed["passed"])
        self.assertEqual(.004,crossed["violations"][0]["time_s"])
        self.assertEqual(5,crossed["checked_solver_samples"])
        self.assertEqual(3,crossed["cached_proof_samples"])
        self.assertGreaterEqual(crossed["boolean_checks"],2)

    def test_native_axis_must_cross_real_regions_but_its_longitudinal_origin_is_free(self):
        for name, x in (("a", -6), ("b", 6)):
            mesh = trimesh.creation.box(extents=[10, 10, 10])
            mesh.apply_translation([x, 0, 0])
            mesh.export(self.root / f"{name}.obj")
        scene = {"schema_version": "slim_scene_v1", "units": "mm", "bodies": [
            {"id": "a", "source": "a.obj", "fixed": False}, {"id": "b", "source": "b.obj", "fixed": True},
        ], "joints": [{"id": "connection", "type": "revolute", "body_a": "a", "body_b": "b",
                        "anchor": [0, 0, 1000], "axis": [0, 0, 1],
                        "region_a": {"points": [[-1, 0, 0], [-1, 0, 0]], "radius": 1.1},
                        "region_b": {"points": [[1, 0, 0], [1, 0, 0]], "radius": 1.1}}],
                 "tests": [{"id": "fixture", "purpose": "fixture validation", "duration_s": .01,
                            "assertions": [{"id": "duration", "purpose": "record duration", "expression": "last(signal('time_s')) >= .01"}]}]}
        path, geometry = self._prepare(scene)
        scene = load_scene(path)
        for kind in ("revolute", "prismatic"):
            with self.subTest(kind=kind):
                valid = copy.deepcopy(scene)
                valid["joints"][0]["type"] = kind
                self.assertEqual([0, 0, 1], compile_scene(valid, geometry)["joints"][0]["anchor"])
                invalid = copy.deepcopy(valid)
                invalid["joints"][0]["anchor"] = [1000, 0, 0]
                with self.assertRaisesRegex(DeliveryError, "constraint frame misses"):
                    compile_scene(invalid, geometry)
        for kind in ("spherical", "fixed"):
            invalid = copy.deepcopy(scene)
            invalid["joints"][0]["type"] = kind
            with self.assertRaisesRegex(DeliveryError, "constraint frame misses"):
                compile_scene(invalid, geometry)

    @unittest.skipUnless(CHRONO_PYTHON.is_file(), "The verified Chrono runtime is unavailable")
    def test_prismatic_coordinate_uses_the_rotating_reference_body_frame(self):
        for name, center in (("a", [.03, .015, 0]), ("b", [-.03, -.005, 0])):
            mesh = trimesh.creation.box(extents=[.01, .01, .01])
            mesh.apply_translation(center)
            mesh.export(self.root / f"{name}.obj")
        _, geometry = self._prepare({"schema_version": "slim_scene_v1", "units": "m", "bodies": [
            {"id": "a", "source": "a.obj", "fixed": False, "density_kg_m3": 1000},
            {"id": "b", "source": "b.obj", "fixed": False, "density_kg_m3": 1000},
        ]})
        # A local-measurement regression: torque rotates the reference body;
        # the prismatic joint remains passive. No fixture result is certified.
        prepared = {"bodies": geometry["bodies"], "gravity_m_s2": [0, 0, 0], "joints": [
            {"id": "slide", "type": "prismatic", "body_a": "a", "body_b": "b", "axis": [1, 0, 0],
             "anchor": [0, 0, 0], "region_a": {"points": [[0, 0, 0], [0, 0, 0]], "radius": .2},
             "region_b": {"points": [[0, 0, 0], [0, 0, 0]], "radius": .2}},
        ]}
        test = {"id": "reference_frame", "duration_s": .03, "step_s": .0001,
                "loads": [{"body": "b", "torque_nm": [0, 0, .001]}],
                "assertions": [{"id": "duration", "purpose": "record duration", "expression": "last(signal('time_s')) >= .03"}]}
        output = self.root / "physical.trace.json"
        request = self.root / "request.json"
        request.write_text(json.dumps({"scene": prepared, "test": test, "output": str(output)}), encoding="utf-8")
        process = subprocess.run([str(CHRONO_PYTHON), "-X", "utf8", "-m",
                                  "simulation_ready_cad_workflow.validation_pipeline.slim_harness.runtime", str(request)],
                                 cwd=self.workspace, capture_output=True, timeout=15, text=True, encoding="utf-8",
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(0, process.returncode, process.stderr)
        trace = open_trace(output)
        self.assertIsNone(trace["failure"])
        expected = []
        initial = np.asarray(geometry["bodies"][0]["position_m"]) - np.asarray(geometry["bodies"][1]["position_m"])
        for sample in trace["samples"]:
            a, b = sample["body_poses"]["a"], sample["body_poses"]["b"]
            rotation = trimesh.transformations.quaternion_matrix(b["rotation_abs_quat_wxyz"])[:3, :3]
            offset_in_reference = rotation.T @ (np.asarray(a["position_abs_m"]) - np.asarray(b["position_abs_m"]))
            expected.append(offset_in_reference[0] - initial[0])
        self.assertGreater(abs(trace["samples"][-1]["body_poses"]["b"]["rotation_abs_quat_wxyz"][3]), .001)
        np.testing.assert_allclose(trace["signals"]["slide.coordinate"], expected, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
