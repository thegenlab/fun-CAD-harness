from __future__ import annotations

import copy
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.contracts.failure_envelope import (
    FailureOwner,
    RepairScope,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    authored_collision_components,
    family_plugins,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.authored_collision_components import (
    COLLISION_COMPONENT_MANIFEST_PATH,
    _fuse_components,
    _validate_component_convexity,
    authored_collision_execution_identity,
    bind_exported_component_meshes,
    load_authored_collision_components,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
    _authored_collision_component_plugin_id,
    _fail,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    sha256_file,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.occt_collision_fidelity import (
    mesh_shape_m,
    occupancy_iou_brep_vs_brep_union,
    shape_bounds_mm,
)


def _write_step(path: Path, shape: object) -> None:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    if writer.Transfer(shape, STEPControl_AsIs) != IFSelect_RetDone:
        raise RuntimeError(f"could not transfer fixture STEP {path}")
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise RuntimeError(f"could not write fixture STEP {path}")


def _indented_box_mesh(*, side_m: float, depth_m: float) -> object:
    import numpy as np
    import trimesh

    side = float(side_m)
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [side, 0.0, 0.0],
            [side, side, 0.0],
            [0.0, side, 0.0],
            [0.0, 0.0, side],
            [side, 0.0, side],
            [side, side, side],
            [0.0, side, side],
            [0.5 * side, 0.5 * side, side - depth_m],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [4, 5, 8],
            [5, 6, 8],
            [6, 7, 8],
            [7, 4, 8],
        ],
        dtype=int,
    )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _wrongly_connected_twisted_prism_mesh() -> object:
    import math
    import numpy as np
    import trimesh

    angle = 2.0e-5
    bottom = np.asarray(
        [
            (
                math.cos(2.0 * math.pi * index / 3.0),
                math.sin(2.0 * math.pi * index / 3.0),
                0.0,
            )
            for index in range(3)
        ],
        dtype=float,
    )
    top = np.asarray(
        [
            (
                math.cos(2.0 * math.pi * index / 3.0 + angle),
                math.sin(2.0 * math.pi * index / 3.0 + angle),
                1.0,
            )
            for index in range(3)
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [3, 4, 5],
            [0, 1, 4],
            [0, 4, 3],
            [1, 2, 5],
            [1, 5, 4],
            [2, 0, 3],
            [2, 3, 5],
        ],
        dtype=int,
    )
    return trimesh.Trimesh(
        vertices=np.vstack((bottom, top)),
        faces=faces,
        process=False,
    )


class AuthoredCollisionComponentTests(unittest.TestCase):
    def test_large_component_union_uses_bounded_occupancy(self) -> None:
        components = tuple(
            SimpleNamespace(shape=object()) for _index in range(42)
        )
        expected = (
            0.975,
            {"method": "multiresolution_voxel_component_union"},
        )
        source_mesh = object()
        component_union_mesh = object()
        with (
            patch.object(
                authored_collision_components,
                "occupancy_iou_brep_vs_brep_union",
                side_effect=AssertionError(
                    "large component sets must not enter unbounded General Fuse"
                ),
            ),
            patch.object(
                authored_collision_components,
                "multiresolution_occupancy_iou",
                return_value=expected,
            ) as bounded,
        ):
            result = authored_collision_components._authored_union_occupancy(
                source_shape=object(),
                component_union_shape=object(),
                components=components,
                source_mesh=source_mesh,
                component_union_mesh=component_union_mesh,
            )

        self.assertEqual(result[0], expected[0])
        self.assertEqual(result[1]["method"], expected[1]["method"])
        self.assertEqual(result[1]["component_count"], 42)
        self.assertEqual(result[1]["exact_component_limit"], 8)
        self.assertIs(result[1]["exact_general_fuse_skipped"], True)
        bounded.assert_called_once_with(
            (source_mesh,),
            (component_union_mesh,),
        )

    def test_large_component_union_uses_bounded_mesh_distance(self) -> None:
        components = tuple(
            SimpleNamespace(shape=object()) for _index in range(88)
        )
        with patch.object(
            authored_collision_components,
            "hybrid_shape_distance_evaluator",
            side_effect=AssertionError(
                "large component sets must not run per-point OCCT distance"
            ),
        ):
            source, collider, method = (
                authored_collision_components._authored_surface_evaluators(
                    source_shape=object(),
                    source_mesh=object(),
                    component_union_shape=object(),
                    component_union_mesh=object(),
                    components=components,
                    exact_query_trigger_m=0.0002,
                )
            )

        self.assertIsNone(source)
        self.assertIsNone(collider)
        self.assertEqual(method, "bounded_mesh_nearest_surface_v1")

    def test_dynamic_geometry_legality_failure_is_design_package_owned(self) -> None:
        error = _fail(
            "FAMILY_DYNAMIC_COLLIDER_REPRESENTATION_UNSUPPORTED",
            "dynamic solid is not an exact primitive or verified convex component",
            owner=FailureOwner.DESIGN_PACKAGE,
            repair_scope=RepairScope.DERIVED_ARTIFACT,
            details={
                "unsupported_dynamic_components": [
                    {"body_id": "support_plate", "solid_id": "solid_5"}
                ]
            },
        )

        self.assertEqual(error.failure_envelope["owner"], "design_package")
        self.assertEqual(
            error.failure_envelope["details"]["unsupported_dynamic_components"],
            [{"body_id": "support_plate", "solid_id": "solid_5"}],
        )
        self.assertEqual(
            error.failure_envelope["repair"]["scope"],
            "derived_artifact",
        )

    def setUp(self) -> None:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.gp import gp_Pnt

        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.package = self.workspace / "package"
        self.package.mkdir()

        first = BRepPrimAPI_MakeBox(
            gp_Pnt(0.0, 0.0, 0.0),
            100.0,
            30.0,
            30.0,
        ).Shape()
        second = BRepPrimAPI_MakeBox(
            gp_Pnt(0.0, 30.0, 0.0),
            30.0,
            70.0,
            30.0,
        ).Shape()
        union = BRepAlgoAPI_Fuse(first, second)
        union.Build()
        self.assertTrue(union.IsDone())
        self.first_shape = first
        self.source_shape = union.Shape()
        self.source_mesh = mesh_shape_m(
            self.source_shape,
            self.workspace / "source.stl",
        )

        self.component_paths = (
            self.package / "collision_components" / "solid_2" / "arm.step",
            self.package / "collision_components" / "solid_2" / "leg.step",
        )
        _write_step(self.component_paths[0], first)
        _write_step(self.component_paths[1], second)
        self.manifest = {
            "schema_version": "simulation_ready_cad_collision_components_v1",
            "sources": [
                {
                    "body_id": "input_crank",
                    "source_solid_id": "solid_2",
                    "components": [
                        {
                            "component_id": "input_arm",
                            "step_file": (
                                "collision_components/solid_2/arm.step"
                            ),
                            "sha256": sha256_file(self.component_paths[0]),
                        },
                        {
                            "component_id": "input_leg",
                            "step_file": (
                                "collision_components/solid_2/leg.step"
                            ),
                            "sha256": sha256_file(self.component_paths[1]),
                        },
                    ],
                }
            ],
        }
        self._write_manifest(self.manifest)

    def test_execution_identity_ignores_component_step_header(self) -> None:
        before = authored_collision_execution_identity(self.package)
        payload = self.component_paths[0].read_bytes()
        data_index = payload.index(b"DATA;")
        self.component_paths[0].write_bytes(
            payload[:data_index] + b"\n" + payload[data_index:]
        )
        self.manifest["sources"][0]["components"][0]["sha256"] = sha256_file(
            self.component_paths[0]
        )
        self._write_manifest(self.manifest)

        header_only = authored_collision_execution_identity(self.package)
        self.assertEqual(header_only, before)

        renamed = self.component_paths[0].with_name("renamed_arm.step")
        self.component_paths[0].rename(renamed)
        source = self.manifest["sources"][0]
        source["body_id"] = "equivalent_runtime_alias"
        source["components"][0]["component_id"] = "renamed_component"
        source["components"][0]["step_file"] = (
            "collision_components/solid_2/renamed_arm.step"
        )
        source["components"].reverse()
        self._write_manifest(self.manifest)
        self.assertEqual(
            authored_collision_execution_identity(self.package), before
        )

        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

        _write_step(
            renamed,
            BRepPrimAPI_MakeBox(101.0, 30.0, 30.0).Shape(),
        )
        changed = next(
            item
            for item in source["components"]
            if item["component_id"] == "renamed_component"
        )
        changed["sha256"] = sha256_file(renamed)
        self._write_manifest(self.manifest)
        self.assertNotEqual(
            authored_collision_execution_identity(self.package), before
        )

    def test_brep_bounds_do_not_change_after_meshing(self) -> None:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeTorus

        shape = BRepPrimAPI_MakeTorus(80.0, 8.0).Shape()
        before = shape_bounds_mm(shape)

        mesh_shape_m(shape, self.workspace / "bounds_order_probe.stl")

        after = shape_bounds_mm(shape)
        for before_corner, after_corner in zip(before, after, strict=True):
            for before_value, after_value in zip(
                before_corner,
                after_corner,
                strict=True,
            ):
                self.assertAlmostEqual(before_value, after_value, places=9)
        self.assertAlmostEqual(before[0][0], -88.0, places=5)
        self.assertAlmostEqual(before[1][0], 88.0, places=5)

    def test_general_fuse_preserves_overlap_for_torus_component_partition(
        self,
    ) -> None:
        import math

        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.BRepPrimAPI import (
            BRepPrimAPI_MakeCylinder,
            BRepPrimAPI_MakeTorus,
        )
        from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt, gp_Vec

        axis = gp_Ax2(gp_Pnt(0.0, -78.0, 0.0), gp_Dir(0.0, 1.0, 0.0))
        torus = BRepPrimAPI_MakeTorus(axis, 40.0, 5.0).Shape()
        backing = BRepPrimAPI_MakeCylinder(
            gp_Ax2(
                gp_Pnt(0.0, -83.7, 0.0),
                gp_Dir(0.0, 1.0, 0.0),
            ),
            36.0,
            11.4,
        ).Shape()
        source_operation = BRepAlgoAPI_Fuse(torus, backing)
        source_operation.Build()
        self.assertTrue(source_operation.IsDone())
        source = source_operation.Shape()

        shapes = [backing]
        for index in range(32):
            start_angle = 2.0 * math.pi * index / 32.0
            end_angle = 2.0 * math.pi * (index + 1) / 32.0
            start = gp_Pnt(
                40.0 * math.cos(start_angle),
                -78.0,
                40.0 * math.sin(start_angle),
            )
            end = gp_Pnt(
                40.0 * math.cos(end_angle),
                -78.0,
                40.0 * math.sin(end_angle),
            )
            direction = gp_Vec(start, end)
            shapes.append(
                BRepPrimAPI_MakeCylinder(
                    gp_Ax2(start, gp_Dir(direction)),
                    5.0,
                    direction.Magnitude(),
                ).Shape()
            )
        components = tuple(SimpleNamespace(shape=shape) for shape in shapes)
        pairwise_union = _fuse_components(components)

        iou, evidence = occupancy_iou_brep_vs_brep_union(
            source,
            pairwise_union,
            component_shapes=shapes,
        )

        self.assertGreater(iou, 0.80)
        self.assertEqual(
            evidence["method"],
            "occt_brep_general_fuse_intersection_over_union",
        )
        self.assertEqual(evidence["component_shape_count"], 33)
        self.assertGreater(evidence["component_partition_solid_count"], 33)
        self.assertLessEqual(
            abs(evidence["component_partition_volume_delta_m3"]),
            evidence["numerical_tolerance_m3"],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self, value: dict) -> None:
        manifest_path = self.package / COLLISION_COMPONENT_MANIFEST_PATH
        write_json_atomic(manifest_path, value)
        artifact_hashes = {
            COLLISION_COMPONENT_MANIFEST_PATH: sha256_file(manifest_path),
        }
        for path in self.package.glob("collision_components/**/*.step"):
            artifact_hashes[path.relative_to(self.package).as_posix()] = (
                sha256_file(path)
            )
        self.plan = SimpleNamespace(
            artifact_hashes=artifact_hashes,
            package_binding_sha256="a" * 64,
        )

    def _load(
        self,
        *,
        expected_sources: dict | None = None,
    ) -> tuple:
        return load_authored_collision_components(
            self.plan,
            package=self.package,
            workspace=self.workspace,
            expected_sources=expected_sources
            or {
                ("input_crank", "solid_2"): (
                    self.source_shape,
                    self.source_mesh,
                )
            },
            scratch_root=self.workspace / "derived",
            characteristic_length_m=0.13,
            mesh_linear_deflection_m=0.0002,
            collision_envelope_m=12e-6,
        )

    def _round_trip_exports(
        self,
        source: object,
    ) -> tuple[dict[str, object], dict[str, Path], dict[str, str]]:
        import trimesh

        meshes: dict[str, object] = {}
        paths: dict[str, Path] = {}
        hashes: dict[str, str] = {}
        for item in source.components:
            path = self.workspace / "runtime_obj" / f"{item.component_id}.obj"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                trimesh.exchange.obj.export_obj(
                    item.mesh_world_m,
                    include_normals=False,
                    include_color=False,
                    include_texture=False,
                    digits=17,
                ),
                encoding="ascii",
                newline="\n",
            )
            meshes[item.component_id] = trimesh.load_mesh(
                path,
                process=False,
                force="mesh",
            )
            paths[item.component_id] = path
            hashes[item.component_id] = sha256_file(path)
        return meshes, paths, hashes

    def assert_code(self, code: str, callback: object) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            FailureOwner.DESIGN_PACKAGE.value,
        )
        self.assertEqual(
            caught.exception.failure_envelope["repair"]["scope"],
            "derived_artifact",
        )

    def test_skinny_triangle_convex_mesh_passes_lp_certificate(self) -> None:
        import trimesh

        mesh = trimesh.creation.cylinder(
            radius=0.005,
            height=0.05,
            sections=126,
        )
        certificate = _validate_component_convexity(
            mesh,
            component_id="skinny_convex",
            collision_envelope_m=12.0e-6,
        )
        self.assertTrue(certificate["passed"])
        self.assertEqual(
            certificate["method"],
            "normalized_convex_hull_halfspaces_and_triangle_interior_lp_v1",
        )
        self.assertFalse(
            certificate["trimesh_is_convex_used_as_evidence"]
        )
        self.assertGreater(certificate["lp_count"], 400)

    def test_near_coplanar_noise_below_collision_envelope_passes(self) -> None:
        mesh = _indented_box_mesh(
            side_m=0.01,
            depth_m=2.0e-6,
        )
        certificate = _validate_component_convexity(
            mesh,
            component_id="near_coplanar_noise",
            collision_envelope_m=12.0e-6,
        )
        depth = certificate[
            "maximum_triangle_interior_inward_depth_m"
        ]
        self.assertGreater(
            depth,
            1.0e-5 * certificate["characteristic_length_m"],
        )
        self.assertLess(depth, 12.0e-6)
        self.assertLess(
            certificate["relative_convex_hull_volume_gap"],
            1.0e-4,
        )

    def test_deep_local_concavity_fails_despite_small_volume_gap(self) -> None:
        from scipy.spatial import ConvexHull

        mesh = _indented_box_mesh(
            side_m=1.0,
            depth_m=24.0e-6,
        )
        gap = (
            float(ConvexHull(mesh.vertices).volume) - float(mesh.volume)
        ) / float(mesh.volume)
        self.assertLess(gap, 1.0e-4)
        self.assert_code(
            "COLLISION_COMPONENT_NOT_CONVEX",
            lambda: _validate_component_convexity(
                mesh,
                component_id="deep_local_concavity",
                collision_envelope_m=12.0e-6,
            ),
        )

    def test_wrong_face_connectivity_fails_with_all_vertices_on_hull(
        self,
    ) -> None:
        import numpy as np
        from scipy.spatial import ConvexHull

        mesh = _wrongly_connected_twisted_prism_mesh()
        hull = ConvexHull(mesh.vertices)
        equations = np.asarray(hull.equations, dtype=float)
        equations /= np.linalg.norm(
            equations[:, :3],
            axis=1,
        )[:, np.newaxis]
        slacks = -(
            np.asarray(mesh.vertices) @ equations[:, :3].T
            + equations[:, 3][np.newaxis, :]
        )
        self.assertLessEqual(
            float(np.max(np.min(slacks, axis=1))),
            1.0e-12,
        )
        gap = (
            float(hull.volume) - float(mesh.volume)
        ) / float(mesh.volume)
        self.assertLess(gap, 1.0e-4)
        self.assert_code(
            "COLLISION_COMPONENT_NOT_CONVEX",
            lambda: _validate_component_convexity(
                mesh,
                component_id="wrong_face_connectivity",
                collision_envelope_m=12.0e-6,
            ),
        )

    def test_convexity_lp_solver_failure_is_fail_closed(self) -> None:
        from unittest.mock import patch
        import trimesh

        mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
        failed = SimpleNamespace(
            status=4,
            success=False,
            x=None,
        )
        with patch("scipy.optimize.linprog", return_value=failed):
            self.assert_code(
                "COLLISION_COMPONENT_CONVEXITY_INCONCLUSIVE",
                lambda: _validate_component_convexity(
                    mesh,
                    component_id="solver_failure",
                    collision_envelope_m=12.0e-6,
                ),
            )

    def test_convexity_manifest_work_budget_is_fail_closed(self) -> None:
        import trimesh

        mesh = trimesh.creation.box(extents=(0.01, 0.01, 0.01))
        self.assert_code(
            "COLLISION_COMPONENT_CONVEXITY_MANIFEST_BUDGET_EXCEEDED",
            lambda: _validate_component_convexity(
                mesh,
                component_id="over_budget",
                collision_envelope_m=12.0e-6,
                manifest_lp_rows_remaining=1,
            ),
        )

    def test_positive_manifest_certifies_union_and_runtime_obj_bytes(self) -> None:
        sources = self._load()
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertTrue(source.report["passed"])
        self.assertTrue(source.report["unique_source_ownership"])
        exported, paths, hashes = self._round_trip_exports(source)
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "authored_collision_components."
            "occupancy_iou_brep_vs_brep_union",
            side_effect=AssertionError(
                "exact runtime mesh binding must transfer trusted occupancy"
            ),
        ):
            fidelity_input, report = bind_exported_component_meshes(
                source,
                exported,
                exported_asset_paths=paths,
                exported_asset_sha256=hashes,
            )
        self.assertTrue(report["runtime_component_mesh_bytes_fidelity_verified"])
        self.assertEqual(
            report["runtime_union_surface_authority"],
            "occt_boolean_union_of_round_trip_bound_component_steps",
        )
        self.assertIs(
            fidelity_input["collider_surface_points"],
            source.fidelity_input["collider_surface_points"],
        )
        self.assertGreater(
            source.fidelity_input["occupancy_iou_override"][0],
            0.999999,
        )
        occupancy_evidence = source.fidelity_input[
            "occupancy_iou_override"
        ][1]
        self.assertGreaterEqual(
            occupancy_evidence["numerical_tolerance_m3"],
            min(
                occupancy_evidence["source_volume_m3"],
                occupancy_evidence["component_union_volume_m3"],
            )
            * 1.0e-6,
        )
        self.assertEqual(
            occupancy_evidence["method"],
            "occt_brep_general_fuse_intersection_over_union",
        )
        self.assertFalse(report["runtime_union_occupancy_recomputed"])
        self.assertTrue(report["runtime_fidelity_result_transfer_ready"])
        self.assertIn(
            "_exact_runtime_mesh_fidelity_transfer",
            fidelity_input,
        )

    def test_missing_manifest_fails_closed_with_template_hint(self) -> None:
        (self.package / COLLISION_COMPONENT_MANIFEST_PATH).unlink()
        self.assert_code(
            "DYNAMIC_COLLISION_COMPONENT_MANIFEST_REQUIRED",
            self._load,
        )

    def test_global_box_or_hull_single_component_is_rejected(self) -> None:
        malicious = copy.deepcopy(self.manifest)
        malicious["sources"][0]["components"] = [
            malicious["sources"][0]["components"][0]
        ]
        self._write_manifest(malicious)
        self.assert_code("COLLISION_COMPONENT_COUNT_INVALID", self._load)

    def test_more_than_component_limit_is_rejected_before_asset_loading(
        self,
    ) -> None:
        malicious = copy.deepcopy(self.manifest)
        malicious["sources"][0]["components"] = [
            copy.deepcopy(malicious["sources"][0]["components"][0])
            for _index in range(513)
        ]
        self._write_manifest(malicious)
        self.assert_code("COLLISION_COMPONENT_COUNT_INVALID", self._load)

    def test_component_overlap_above_five_percent_is_rejected(self) -> None:
        duplicate_path = (
            self.package
            / "collision_components"
            / "solid_2"
            / "duplicate_arm.step"
        )
        _write_step(duplicate_path, self.first_shape)
        malicious = copy.deepcopy(self.manifest)
        malicious["sources"][0]["components"].append(
            {
                "component_id": "duplicate_input_arm",
                "step_file": (
                    "collision_components/solid_2/duplicate_arm.step"
                ),
                "sha256": sha256_file(duplicate_path),
            }
        )
        self._write_manifest(malicious)
        with self.assertRaises(MechanismValidationError) as caught:
            self._load()
        self.assertEqual(
            caught.exception.code,
            "COLLISION_COMPONENT_OVERLAP_EXCESSIVE",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            FailureOwner.DESIGN_PACKAGE.value,
        )
        self.assertEqual(
            caught.exception.failure_envelope["evidence_validity"],
            "VALID_MEASUREMENT",
        )
        self.assertEqual(
            caught.exception.failure_envelope["details"][
                "source_solid_id"
            ],
            "solid_2",
        )
        self.assertEqual(
            caught.exception.failure_envelope["repair"]["scope"],
            "derived_artifact",
        )
        self.assertEqual(
            caught.exception.failure_envelope["repair"]["resume_stage"],
            "preflight",
        )

    def test_global_convex_hull_plus_decoy_component_fails_fidelity(self) -> None:
        from OCP.BRepBuilderAPI import (
            BRepBuilderAPI_MakeFace,
            BRepBuilderAPI_MakePolygon,
        )
        from OCP.BRepPrimAPI import (
            BRepPrimAPI_MakeBox,
            BRepPrimAPI_MakePrism,
        )
        from OCP.gp import gp_Pnt, gp_Vec

        polygon = BRepBuilderAPI_MakePolygon()
        for x_value, y_value in (
            (0.0, 0.0),
            (100.0, 0.0),
            (100.0, 30.0),
            (30.0, 100.0),
            (0.0, 100.0),
        ):
            polygon.Add(gp_Pnt(x_value, y_value, 0.0))
        polygon.Close()
        hull = BRepPrimAPI_MakePrism(
            BRepBuilderAPI_MakeFace(polygon.Wire()).Face(),
            gp_Vec(0.0, 0.0, 30.0),
        ).Shape()
        decoy = BRepPrimAPI_MakeBox(
            gp_Pnt(5.0, 5.0, 5.0),
            2.0,
            2.0,
            2.0,
        ).Shape()
        hull_path = (
            self.package
            / "collision_components"
            / "solid_2"
            / "global_hull.step"
        )
        decoy_path = (
            self.package
            / "collision_components"
            / "solid_2"
            / "decoy.step"
        )
        _write_step(hull_path, hull)
        _write_step(decoy_path, decoy)
        malicious = copy.deepcopy(self.manifest)
        malicious["sources"][0]["components"] = [
            {
                "component_id": "global_hull",
                "step_file": (
                    "collision_components/solid_2/global_hull.step"
                ),
                "sha256": sha256_file(hull_path),
            },
            {
                "component_id": "decoy",
                "step_file": "collision_components/solid_2/decoy.step",
                "sha256": sha256_file(decoy_path),
            },
        ]
        self._write_manifest(malicious)
        self.assert_code(
            "COLLISION_COMPONENT_UNION_FIDELITY_FAILED",
            self._load,
        )

    def test_source_solid_swap_is_rejected(self) -> None:
        for body_id, solid_id in (
            ("visible_chain_path", "solid_4"),
            ("input_crank", "solid_3"),
        ):
            with self.subTest(body_id=body_id, solid_id=solid_id):
                malicious = copy.deepcopy(self.manifest)
                malicious["sources"][0]["body_id"] = body_id
                malicious["sources"][0]["source_solid_id"] = solid_id
                self._write_manifest(malicious)
                self.assert_code(
                    "COLLISION_COMPONENT_SOURCE_SET_MISMATCH",
                    self._load,
                )
                self._write_manifest(self.manifest)

    def test_body_alias_reuses_one_authoritative_source_declaration(
        self,
    ) -> None:
        aliased = copy.deepcopy(self.manifest)
        aliased["sources"][0]["body_id"] = "chain_task_crank"
        self._write_manifest(aliased)

        source = self._load(
            expected_sources={
                ("ground_task_chassis", "solid_2"): (
                    self.source_shape,
                    self.source_mesh,
                )
            }
        )[0]

        self.assertEqual(source.body_id, "ground_task_chassis")
        self.assertEqual(source.report["declared_body_id"], "chain_task_crank")
        self.assertTrue(source.report["runtime_body_alias_applied"])
        self.assertTrue(source.report["passed"])

    def test_manifest_superset_is_available_to_multiple_task_capsules(
        self,
    ) -> None:
        extra_paths = (
            self.package
            / "collision_components"
            / "solid_3"
            / "unused_a.step",
            self.package
            / "collision_components"
            / "solid_3"
            / "unused_b.step",
        )
        for source, target in zip(
            self.component_paths,
            extra_paths,
            strict=True,
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        superset = copy.deepcopy(self.manifest)
        superset["sources"].append(
            {
                "body_id": "rear_wheel",
                "source_solid_id": "solid_3",
                "components": [
                    {
                        "component_id": "unused_rear_a",
                        "step_file": (
                            "collision_components/solid_3/unused_a.step"
                        ),
                        "sha256": sha256_file(extra_paths[0]),
                    },
                    {
                        "component_id": "unused_rear_b",
                        "step_file": (
                            "collision_components/solid_3/unused_b.step"
                        ),
                        "sha256": sha256_file(extra_paths[1]),
                    },
                ],
            }
        )
        self._write_manifest(superset)

        source = self._load()[0]

        self.assertEqual(source.source_solid_id, "solid_2")
        self.assertEqual(
            source.report["ownership_verification"][
                "manifest_component_count"
            ],
            4,
        )

    def test_unused_manifest_source_is_still_hash_validated(self) -> None:
        malformed = copy.deepcopy(self.manifest)
        malformed["sources"].append(
            {
                "body_id": "front_wheel",
                "source_solid_id": "solid_20",
                "components": [
                    {
                        "component_id": "missing_front_a",
                        "step_file": (
                            "collision_components/solid_20/missing_a.step"
                        ),
                        "sha256": "a" * 64,
                    },
                    {
                        "component_id": "missing_front_b",
                        "step_file": (
                            "collision_components/solid_20/missing_b.step"
                        ),
                        "sha256": "b" * 64,
                    },
                ],
            }
        )
        self._write_manifest(malformed)

        self.assert_code(
            "COLLISION_COMPONENT_ASSET_CHANGED",
            self._load,
        )

    def test_duplicate_source_solid_under_another_body_is_rejected(
        self,
    ) -> None:
        duplicate = copy.deepcopy(self.manifest)
        duplicate_source = copy.deepcopy(duplicate["sources"][0])
        duplicate_source["body_id"] = "ground_task_chassis"
        duplicate["sources"].append(duplicate_source)
        self._write_manifest(duplicate)

        self.assert_code(
            "COLLISION_COMPONENT_SOURCE_DUPLICATE",
            self._load,
        )

    def test_runtime_cannot_assign_one_source_to_two_bodies(self) -> None:
        self.assert_code(
            "COLLISION_COMPONENT_RUNTIME_SOURCE_AMBIGUOUS",
            lambda: self._load(
                expected_sources={
                    ("first_body", "solid_2"): (
                        self.source_shape,
                        self.source_mesh,
                    ),
                    ("second_body", "solid_2"): (
                        self.source_shape,
                        self.source_mesh,
                    ),
                }
            ),
        )

    def test_explicit_mechanism_plugins_share_component_protocol(self) -> None:
        for mechanism_kind, plugin_id, field in (
            (
                "bicycle_chain_sprocket_drive",
                "bicycle_chain_sprocket_drive_v2",
                "mechanism_kind",
            ),
            ("bicycle_ground_roll", "bicycle_ground_roll_v1", "mechanism_kind"),
            (
                "deadbolt_door_lock",
                "reviewer.deadbolt_door_lock_v1",
                "mechanism_kind",
            ),
            (
                "clothespin_dual_jaw_torsion",
                "reviewer.clothespin_dual_jaw_torsion_v1",
                "mechanism_kind",
            ),
            (
                "rotary_detent_phone_stand",
                "reviewer.rotary_detent_phone_stand_v1",
                "mechanism_kind",
            ),
            (
                "spring_scale_prismatic_spring",
                "reviewer.spring_scale_prismatic_spring_v1",
                "mechanism_kind",
            ),
            (
                "foldable_phone_stand_rotary_detent",
                "revolute_linkage_v1",
                "linkage_id",
            ),
        ):
            with self.subTest(mechanism_kind=mechanism_kind):
                adapter = SimpleNamespace(
                    family_spec={field: mechanism_kind}
                )
                if plugin_id.startswith("reviewer."):
                    with (
                        patch.object(
                            family_plugins,
                            "plugin_registration_for",
                            return_value={"plugin_id": plugin_id},
                        ),
                        patch.object(
                            family_plugins,
                            "registration_allows_runtime_execution",
                            return_value=True,
                        ),
                    ):
                        actual = _authored_collision_component_plugin_id(adapter)
                else:
                    actual = _authored_collision_component_plugin_id(adapter)
                self.assertEqual(actual, plugin_id)
        self.assertIsNone(
            _authored_collision_component_plugin_id(
                SimpleNamespace(
                    family_spec={"mechanism_kind": "rack_pinion"}
                )
            )
        )
        self.assertIsNone(
            _authored_collision_component_plugin_id(
                SimpleNamespace(family_spec=None)
            )
        )

    def test_cross_source_asset_and_id_reuse_is_rejected(self) -> None:
        malicious = copy.deepcopy(self.manifest)
        duplicate = copy.deepcopy(malicious["sources"][0])
        duplicate["body_id"] = "rear_drive_wheel"
        duplicate["source_solid_id"] = "solid_3"
        malicious["sources"].append(duplicate)
        self._write_manifest(malicious)
        self.assert_code(
            "COLLISION_COMPONENT_OWNERSHIP_NOT_UNIQUE",
            lambda: self._load(
                expected_sources={
                    ("input_crank", "solid_2"): (
                        self.source_shape,
                        self.source_mesh,
                    ),
                    ("rear_drive_wheel", "solid_3"): (
                        self.source_shape,
                        self.source_mesh,
                    ),
                }
            ),
        )

    def test_noncanonical_solid_id_cannot_escape_scratch_root(self) -> None:
        malicious = copy.deepcopy(self.manifest)
        malicious["sources"][0]["source_solid_id"] = "../solid_2"
        self._write_manifest(malicious)
        self.assert_code(
            "COLLISION_COMPONENT_SOURCE_BINDING_INVALID",
            self._load,
        )

    def test_runtime_export_tamper_is_rejected(self) -> None:
        source = self._load()[0]
        exported, paths, hashes = self._round_trip_exports(source)
        exported["input_arm"].apply_translation((0.003, 0.0, 0.0))
        self.assert_code(
            "COLLISION_COMPONENT_RUNTIME_EXPORT_BINDING_FAILED",
            lambda: bind_exported_component_meshes(
                source,
                exported,
                exported_asset_paths=paths,
                exported_asset_sha256=hashes,
            ),
        )

    def test_runtime_obj_hash_tamper_is_rejected(self) -> None:
        source = self._load()[0]
        exported, paths, hashes = self._round_trip_exports(source)
        with paths["input_arm"].open("a", encoding="ascii", newline="\n") as stream:
            stream.write("# changed after certification\n")
        self.assert_code(
            "COLLISION_COMPONENT_RUNTIME_EXPORT_HASH_CHANGED",
            lambda: bind_exported_component_meshes(
                source,
                exported,
                exported_asset_paths=paths,
                exported_asset_sha256=hashes,
            ),
        )

    def test_runtime_obj_topology_tamper_is_rejected(self) -> None:
        import numpy as np

        source = self._load()[0]
        exported, paths, hashes = self._round_trip_exports(source)
        faces = np.asarray(exported["input_arm"].faces, dtype=int).copy()
        faces[[0, 1]] = faces[[1, 0]]
        exported["input_arm"].faces = faces
        self.assert_code(
            "COLLISION_COMPONENT_RUNTIME_EXPORT_BINDING_FAILED",
            lambda: bind_exported_component_meshes(
                source,
                exported,
                exported_asset_paths=paths,
                exported_asset_sha256=hashes,
            ),
        )


if __name__ == "__main__":
    unittest.main()
