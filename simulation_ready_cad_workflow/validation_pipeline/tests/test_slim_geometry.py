from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import trimesh

from ..slim_harness.geometry import GeometryError, geometry_distance_to_point, prepare_geometry


class SlimGeometryTests(unittest.TestCase):
    def setUp(self):
        runtime = Path(__file__).resolve().parents[3] / ".runtime_runs" / "slim_geometry_tests"
        runtime.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=runtime)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        cache = mock.patch.dict(os.environ, {"SIMREADY_STEP_SHAPE_CACHE_DIR": str(self.root / "step_cache")})
        cache.start()
        self.addCleanup(cache.stop)

    def scene(self, bodies, *, units="mm", name="scene"):
        path = self.root / f"{name}.json"
        path.write_text(json.dumps({"schema_version": "slim_scene_v1", "units": units,
                                   "bodies": bodies, "joints": [], "tests": []}), encoding="utf-8")
        return path

    def step_boxes(self):
        import xml.parsers.expat
        from build123d import Box, Compound, Pos, Rot, export_step

        first = Pos(10, 20, 30) * Box(10, 20, 30)
        second = Pos(80, 50, 5) * Rot(0, 0, 30) * Box(20, 10, 10)
        first.label, second.label = "first", "second"
        source = self.root / "source.step"
        export_step(Compound(children=[first, second]), source)
        return source

    def test_relative_parent_source_is_rejected_before_host_cad_read(self):
        peer = self.root / 'peer.step'
        peer.write_text('CONTROLLED_PEER_ARTIFACT')
        design = self.root / 'selected-design'
        design.mkdir()
        source = design / 'scene.json'
        source.write_text(json.dumps({'schema_version': 'slim_scene_v1', 'units': 'mm',
            'bodies': [{'id': 'body', 'source': '../peer.step', 'fixed': False}], 'joints': [], 'tests': []}))
        with mock.patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry._step_source',
                        side_effect=RuntimeError('OUTSIDE_SOURCE_WOULD_BE_OPENED')) as loader:
            result = prepare_geometry(source, self.root / 'prepared')
        loader.assert_not_called()
        self.assertEqual(result['errors'][0]['code'], 'SOURCE_OUTSIDE_DESIGN')
        self.assertEqual(peer.read_text(), 'CONTROLLED_PEER_ARTIFACT')

    def test_no_visual_sdk_retains_numeric_geometry_without_display_sidecar(self):
        from ..benchmark_sandbox import visual_policy
        source = self.root / 'body.obj'
        source.write_text(trimesh.exchange.obj.export_obj(trimesh.creation.box(extents=[2, 3, 4])),
                          encoding='utf-8')
        scene = self.scene([{'id': 'body', 'source': source.name, 'fixed': False,
                             'density_kg_m3': 1000, 'color': [1.0, 0.25, 0.5]}])
        marker = self.root / 'author-visual-policy.json'
        with mock.patch.object(visual_policy, 'MARKER', marker):
            ordinary = prepare_geometry(scene, self.root / 'ordinary')
            self.assertEqual([], ordinary['errors'])
            self.assertTrue(Path(ordinary['model_file']).is_file())
            marker.write_text(json.dumps({'schema_version': 1,
                'author_visual_policy': visual_policy.WORKER_NO_VISUALS}), encoding='utf-8')
            with mock.patch.object(trimesh.exchange.gltf, 'export_glb',
                                   side_effect=AssertionError('No visual export may run')):
                restricted = prepare_geometry(scene, self.root / 'restricted')
            self.assertEqual([], restricted['errors'])
            self.assertIsNone(restricted['model_file'])
            self.assertFalse(list((self.root / 'restricted').glob('*.glb*')))
            ordinary_body, restricted_body = ordinary['bodies'][0], restricted['bodies'][0]
            for key in ('mass_kg', 'volume_m3', 'position_m', 'inertia_kg_m2',
                        'source_sha256', 'source_solid_ids', 'fidelity', 'color'):
                self.assertEqual(ordinary_body[key], restricted_body[key], key)
            self.assertEqual(Path(ordinary_body['mesh_file']).read_bytes(),
                             Path(restricted_body['mesh_file']).read_bytes())
            for before, after in zip(ordinary_body['source_components'], restricted_body['source_components']):
                self.assertEqual(Path(before['mesh_file']).read_bytes(), Path(after['mesh_file']).read_bytes())
            self.assertEqual(ordinary['source_bindings'], restricted['source_bindings'])
            marker.unlink()
            reviewer = prepare_geometry(scene, self.root / 'reviewer')
            self.assertEqual([], reviewer['errors'])
            self.assertTrue(Path(reviewer['model_file']).is_file())

    def step_pair(self, name, second_x, second_size=(10, 10, 10), third=None):
        import xml.parsers.expat
        from build123d import Box, Compound, Pos, export_step

        first = Pos(0, 0, 0) * Box(20, 20, 20)
        second = Pos(second_x, 0, 0) * Box(*second_size)
        children = [first, second]
        if third is not None:
            children.append(third)
        source = self.root / f"{name}.step"
        export_step(Compound(children=children), source)
        return source

    def test_rigid_body_solids_must_share_material_or_touch(self):
        from ..slim_harness.geometry import RIGID_BODY_CONTACT_GAP_MM
        self.assertEqual(0.1, RIGID_BODY_CONTACT_GAP_MM)
        cases = {
            "touching": (15.0, None),          # 10 mm box face-to-face on the 20 mm box (gap 0)
            "overlapping": (12.0, None),       # 3 mm of shared material
            "within_gap": (15.05, None),       # 0.05 mm gap: modelling tolerance
            "contained": (0.0, (4, 4, 4)),     # boundaries 8 mm apart, solid inside the material
        }
        for name, (x, size) in cases.items():
            source = self.step_pair(name, x, size or (10, 10, 10))
            result = prepare_geometry(self.scene([{"id": "welded", "source": source.name, "fixed": False}], name=name), self.root / name)
            self.assertEqual([], result["errors"], name)
            self.assertEqual(["solid_1", "solid_2"], result["bodies"][0]["source_solid_ids"], name)
        # A stop pad 1 mm away from the body is floating, not welded; fixed bodies get no exemption.
        source = self.step_pair("floating", 16.0)
        result = prepare_geometry(self.scene([{"id": "frame", "source": source.name, "fixed": True}], name="floating"), self.root / "floating")
        self.assertEqual("RIGID_BODY_MATERIAL_DISCONNECTED", result["errors"][0]["code"])
        self.assertIn("frame: its solids form 2 material-disconnected groups", result["errors"][0]["message"])
        self.assertIn("[solid_1] | [solid_2]", result["errors"][0]["message"])
        self.assertEqual([], result["bodies"])
        # Declaring the separated solid as its own body is the accepted form.
        result = prepare_geometry(self.scene([
            {"id": "frame", "source": source.name, "solid_ids": ["solid_1"], "fixed": True},
            {"id": "pad", "source": source.name, "solid_ids": ["solid_2"], "fixed": False},
        ], name="split"), self.root / "split")
        self.assertEqual([], result["errors"])
        # Chains count: a bridge solid touching both otherwise separated solids joins them.
        from build123d import Box, Pos
        bridge = Pos(12.5, 0, 0) * Box(5, 4, 4)  # spans x in [10, 15]
        source = self.step_pair("bridged", 16.0, third=bridge)  # second box starts at x=11 -> overlaps bridge
        result = prepare_geometry(self.scene([{"id": "welded", "source": source.name, "fixed": False}], name="bridged"), self.root / "bridged")
        self.assertEqual([], result["errors"])
        self.assertEqual(3, len(result["bodies"][0]["source_solid_ids"]))

    def test_step_solid_coverage_world_placement_and_full_source_inertia(self):
        self.step_boxes()
        scene = self.scene([
            {"id": "first", "source": "source.step", "solid_ids": ["solid_1"], "fixed": True, "density_kg_m3": 1000},
            {"id": "second", "source": "source.step", "solid_ids": ["solid_2"], "fixed": False, "density_kg_m3": 1000},
        ], units="m")  # STEP's embedded units must not be scaled a second time.
        result = prepare_geometry(scene, self.root / "prepared")
        self.assertEqual([], result["errors"])
        first, second = result["bodies"]
        np.testing.assert_allclose(first["position_m"], [.01, .02, .03], atol=1e-12)
        np.testing.assert_allclose(second["position_m"], [.08, .05, .005], atol=1e-12)
        self.assertAlmostEqual(.006, first["mass_kg"], places=12)
        expected_first = .006 / 12 * np.diag([.02**2 + .03**2, .01**2 + .03**2, .01**2 + .02**2])
        np.testing.assert_allclose(first["inertia_kg_m2"], expected_first, atol=1e-15)
        rotation = trimesh.transformations.rotation_matrix(np.pi / 6, [0, 0, 1])[:3, :3]
        source_inertia = .002 / 12 * np.diag([.01**2 + .01**2, .02**2 + .01**2, .02**2 + .01**2])
        np.testing.assert_allclose(second["inertia_kg_m2"], rotation @ source_inertia @ rotation.T, atol=1e-15)
        collider = trimesh.load(first["mesh_file"], force="mesh", process=False)
        np.testing.assert_allclose(collider.bounds, [[-.005, -.01, -.015], [.005, .01, .015]], atol=1e-12)
        shown = trimesh.load(result["model_file"], force="scene", process=False)
        self.assertEqual(2, len(shown.graph.nodes_geometry))
        self.assertEqual({"solid_1": "first", "solid_2": "second"}, result["source_bindings"][0]["solid_owners"])
        self.assertFalse(first["fidelity"]["convex_hull_used"])
        self.assertFalse(first["fidelity"]["brep_surface_distance_independently_measured"])
        self.assertLessEqual(first["fidelity"]["export_vertex_error_max_m"], 1e-12)

    def test_spherical_step_poles_keep_closed_geometry_and_brep_mass(self):
        import xml.parsers.expat
        from build123d import Pos, Sphere, export_step

        source = self.root / "sphere.step"
        export_step(Pos(-37, 160.2, 0) * Sphere(4.1), source)
        result = prepare_geometry(self.scene([
            {"id": "sphere", "source": source.name, "fixed": False, "density_kg_m3": 1000}
        ]), self.root / "sphere")
        self.assertEqual([], result["errors"])
        body = result["bodies"][0]
        self.assertEqual(["solid_1"], body["source_solid_ids"])
        self.assertAlmostEqual(1000 * 4 * np.pi * .0041**3 / 3, body["mass_kg"], places=12)
        np.testing.assert_allclose(body["position_m"], [-.037, .1602, 0], atol=1e-12)
        mesh = trimesh.load(body["mesh_file"], force="mesh", process=False)
        self.assertTrue(mesh.is_watertight and mesh.is_winding_consistent)
        self.assertGreater(mesh.volume, 0)

    def test_shared_step_cannot_duplicate_drop_or_guess_a_solid(self):
        self.step_boxes()
        cases = [
            ([{"id": "a", "source": "source.step", "solid_ids": ["solid_1"], "fixed": False}], "SOURCE_SOLIDS_UNASSIGNED"),
            ([{"id": "a", "source": "source.step", "solid_ids": ["solid_1", "solid_1", "solid_2"], "fixed": False}], "SOURCE_SOLID_ASSIGNED_TWICE"),
            ([{"id": "a", "source": "source.step", "solid_ids": ["solid_7"], "fixed": False}], "UNKNOWN_SOURCE_SOLID"),
            ([{"id": "a", "source": "source.step", "fixed": False}, {"id": "b", "source": "source.step", "solid_ids": ["solid_2"], "fixed": True}], "SOURCE_SOLID_ASSIGNMENT_REQUIRED"),
        ]
        for index, (bodies, code) in enumerate(cases):
            with self.subTest(code=code):
                result = prepare_geometry(self.scene(bodies, name=f"bad-{index}"), self.root / f"bad-{index}")
                self.assertEqual(code, result["errors"][0]["code"])
                self.assertEqual([], result["bodies"])
                self.assertIsNone(result["model_file"])

    def test_mesh_mm_and_m_have_identical_mass_and_distance(self):
        results = []
        for units, scale in (("mm", 1), ("m", .001)):
            mesh = trimesh.creation.box(extents=np.array([10, 20, 30]) * scale)
            mesh.apply_translation(np.array([100, 0, 0]) * scale)
            source = self.root / f"source-{units}.stl"
            mesh.export(source)
            result = prepare_geometry(self.scene([
                {"id": "body", "source": source.name, "fixed": False, "density_kg_m3": 1000}
            ], units=units, name=f"scene-{units}"), self.root / f"out-{units}")
            self.assertEqual([], result["errors"])
            body = result["bodies"][0]
            self.assertAlmostEqual(.006, body["mass_kg"], places=8)
            np.testing.assert_allclose(body["position_m"], [.1, 0, 0], atol=2e-8)
            self.assertEqual(0, geometry_distance_to_point(body, [.1, 0, 0]))
            self.assertAlmostEqual(.002, geometry_distance_to_point(body, [.107, 0, 0]), places=7)
            results.append(body)
        np.testing.assert_allclose(results[0]["inertia_kg_m2"], results[1]["inertia_kg_m2"], rtol=1e-6, atol=1e-18)

    def test_glb_instances_keep_transforms_and_parallel_axis_inertia(self):
        source_scene = trimesh.Scene()
        box = trimesh.creation.box(extents=[1, 1, 1])
        source_scene.add_geometry(box, geom_name="shared", node_name="left",
                                  transform=trimesh.transformations.translation_matrix([-1, 0, 0]))
        source_scene.graph.update(frame_to="right", frame_from=source_scene.graph.base_frame,
                                  matrix=trimesh.transformations.translation_matrix([1, 0, 0]), geometry="shared")
        source = self.root / "instances.glb"
        source.write_bytes(trimesh.exchange.gltf.export_glb(source_scene))
        result = prepare_geometry(self.scene([
            {"id": "one-rigid-assembly", "source": source.name, "fixed": False, "density_kg_m3": 1}
        ], units="m"), self.root / "instances")
        self.assertEqual([], result["errors"])
        body = result["bodies"][0]
        self.assertEqual(2, result["source_bindings"][0]["solid_count"])
        self.assertEqual(["solid_1", "solid_2"], body["source_solid_ids"])
        self.assertAlmostEqual(2, body["mass_kg"])
        np.testing.assert_allclose(body["position_m"], [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(body["inertia_kg_m2"], np.diag([1 / 3, 7 / 3, 7 / 3]), atol=1e-12)
        self.assertAlmostEqual(.5, geometry_distance_to_point(body, [0, 0, 0]))
        self.assertEqual(0, geometry_distance_to_point(body, [1, 0, 0]))
        shown = trimesh.load(result["model_file"], force="scene", process=False)
        self.assertEqual(2, len(shown.graph.nodes_geometry))

    def test_step_distance_uses_material_and_follows_updated_body_pose(self):
        import xml.parsers.expat
        from build123d import Cylinder, export_step

        source = self.root / "ring.step"
        export_step(Cylinder(10, 10) - Cylinder(5, 12), source)
        result = prepare_geometry(self.scene([{"id": "ring", "source": source.name, "fixed": False}]), self.root / "ring")
        self.assertEqual([], result["errors"])
        body = result["bodies"][0]
        self.assertAlmostEqual(.005, geometry_distance_to_point(body, [0, 0, 0]), places=10)
        self.assertEqual(0, geometry_distance_to_point(body, [.007, 0, 0]))
        moved = copy.deepcopy(body)
        moved["position_m"] = [1, 0, 0]
        moved["rotation_wxyz"] = [np.sqrt(.5), 0, np.sqrt(.5), 0]
        self.assertAlmostEqual(.005, geometry_distance_to_point(moved, [1, 0, 0]), places=10)
        self.assertEqual(0, geometry_distance_to_point(moved, [1, 0, -.007]))
        source.write_bytes(source.read_bytes() + b"\n")
        with self.assertRaisesRegex(GeometryError, "ring.step"):
            geometry_distance_to_point(body, [0, 0, 0])

    def test_open_or_negative_volume_mesh_is_rejected_without_repair(self):
        cases = [("open", trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]], process=False), "SOURCE_MESH_NOT_CLOSED")]
        inverted = trimesh.creation.box()
        inverted.invert()
        cases.append(("inverted", inverted, "SOURCE_VOLUME_NOT_POSITIVE"))
        for name, mesh, code in cases:
            with self.subTest(name=name):
                path = self.root / f"{name}.obj"
                mesh.export(path)
                result = prepare_geometry(self.scene([{"id": name, "source": path.name, "fixed": False}], units="m", name=name), self.root / name)
                self.assertEqual(code, result["errors"][0]["code"])
                self.assertEqual([], result["bodies"])

    def test_trimmed_cylinder_internal_diagonals_respect_absolute_deflection(self):
        import xml.parsers.expat
        from build123d import Align, Box, Cylinder, Plane, Pos, Rot, Solid, export_step, import_step
        from OCP.BRep import BRep_Tool
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Cylinder
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
        from ..slim_harness import geometry

        def cylinder(radius, y0, y1):
            return Pos(27, y1, 115) * Rot(X=90) * Cylinder(
                radius, y1 - y0, align=(Align.CENTER, Align.CENTER, Align.MIN)
            )

        pin = cylinder(3, -55, 12) + cylinder(5, -57, -55)
        pin -= Pos(27, -50.5, 115) * Box(10, 1.6, 2)
        pin -= Pos(27, 10.4, 115) * Box(20, 1.6, 2)
        pin = pin.mirror(Plane.XZ)
        # The supported STEP round trip matters: the in-memory parameterization
        # does not reproduce the long diagonal between the two transverse slots.
        source = self.root / "slotted_pin.step"
        export_step(Solid(pin.wrapped), source)
        raw = import_step(source).wrapped
        with mock.patch.object(geometry, "STEP_ANGULAR_TOLERANCE_RAD", .12):
            geometry._tessellate_step_mm(raw)

        # Measure the exact radial deficit of all original cylindrical-facet
        # edge midpoints; do not confuse the planar cut faces with the cylinder.
        errors = []
        explorer = TopExp_Explorer(raw, TopAbs_FACE)
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() == GeomAbs_Cylinder and abs(surface.Cylinder().Radius() - 3) < 1e-7:
                location = TopLoc_Location()
                mesh = BRep_Tool.Triangulation_s(face, location)
                self.assertIsNotNone(mesh)
                transform = location.Transformation()
                points = []
                for index in range(1, mesh.NbNodes() + 1):
                    point = mesh.Node(index).Transformed(transform)
                    points.append((point.X(), point.Y(), point.Z()))
                faces = np.asarray([
                    mesh.Triangle(index).Get() for index in range(1, mesh.NbTriangles() + 1)
                ]) - 1
                triangles = np.asarray(points)[faces]
                midpoints = np.concatenate([
                    (triangles[:, a] + triangles[:, b]) / 2 for a, b in ((0, 1), (1, 2), (2, 0))
                ])
                radial = np.linalg.norm(midpoints[:, [0, 2]] - [27, 115], axis=1)
                errors.append(float(np.max(3 - radial)))
            explorer.Next()
        self.assertTrue(errors, "Fixture has no original radius-3 cylinder faces")
        self.assertEqual(geometry.STEP_DEFLECTION_MM, 0.05)
        self.assertLessEqual(max(errors), geometry.STEP_DEFLECTION_MM + 1e-8)

    def test_incomplete_step_returns_a_failed_geometry_record(self):
        source = self.root / "unfinished.step"
        source.write_text("ISO-10303-21;\nHEADER;\n", encoding="ascii")
        result = prepare_geometry(self.scene([
            {"id": "unfinished", "source": source.name, "fixed": False}
        ]), self.root / "unfinished")
        self.assertTrue(result["errors"])
        self.assertEqual([], result["bodies"])
        self.assertIsNone(result["model_file"])

    def test_source_capsule_is_local_to_full_body_mass_center(self):
        import xml.parsers.expat
        from build123d import Box, Cylinder, Pos, Rot, export_step

        rod = Pos(20, 30, 40) * Rot(90, 0, 0) * Cylinder(3, 68)
        source = rod + Pos(24, 55, 40) * Box(10, 10, 10)
        path = self.root / "offset_shaft.step"
        export_step(source, path)
        result = prepare_geometry(self.scene([
            {"id": "shaft_with_boss", "source": path.name, "density_kg_m3": 1000}
        ]), self.root / "offset_shaft_geometry")
        self.assertEqual([], result["errors"])
        body = result["bodies"][0]
        self.assertAlmostEqual(source.volume * 1e-6, body["mass_kg"], places=12)
        self.assertEqual(1, len(body["source_analytic_material"]))
        capsule = body["source_analytic_material"][0]
        np.testing.assert_allclose(capsule["center_source_m"], [.02, .03, .04], atol=1e-12)
        np.testing.assert_allclose(np.asarray(capsule["center_local_m"])+body["position_m"], [.02, .03, .04], atol=1e-12)
        self.assertGreater(np.linalg.norm(capsule["center_local_m"]), .001)
        np.testing.assert_allclose(capsule["axis_source"], [0, 1, 0], atol=1e-12)
        self.assertAlmostEqual(.003, capsule["radius_m"])
        self.assertAlmostEqual(.062, capsule["straight_length_m"])
        self.assertEqual(0., capsule["proof"]["missing_volume_mm3"])
        self.assertTrue(Path(body["mesh_file"]).is_file())
        self.assertTrue(trimesh.load_mesh(body["mesh_file"], process=False).is_watertight)


if __name__ == "__main__":
    unittest.main()
