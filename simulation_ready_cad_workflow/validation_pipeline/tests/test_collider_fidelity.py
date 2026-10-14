from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import trimesh

from simulation_ready_cad_workflow.validation_pipeline.mechanism.collider_fidelity import (
    ExactRuntimeMeshFidelityTransfer,
    _bounded_nearest_surface_distances,
    _promote_componentwise_body_union_result,
    build_collider_fidelity_report,
    derive_fidelity_thresholds,
    evaluate_mesh_fidelity,
    multiresolution_occupancy_iou,
    verify_collider_fidelity_binding,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    sha256_file,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class ColliderFidelityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = derive_fidelity_thresholds(
            characteristic_length_m=2.0,
            mesh_linear_deflection_m=0.001,
            collision_envelope_m=0.00001,
        )

    def evaluate(self, source: trimesh.Trimesh, collider: trimesh.Trimesh) -> dict:
        return evaluate_mesh_fidelity(
            source,
            collider,
            source_id="source-solid",
            collider_id="loaded-collider",
            representation="test_mesh",
            thresholds=self.thresholds,
            sample_count=192,
            voxel_resolution=28,
        )

    def test_identical_mesh_passes_with_bidirectional_evidence(self) -> None:
        box = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
        result = self.evaluate(box, box.copy())

        self.assertEqual(result["status"], "PASS")
        self.assertLess(result["surface_distance"]["bidirectional"]["max_m"], 1e-9)
        self.assertGreater(result["occupancy_iou"], 0.99)
        self.assertFalse(result["iou_is_sufficient_by_itself"])
        self.assertIsInstance(
            result["surface_distance"]["sample_count_each_direction"],
            int,
        )
        self.assertEqual(
            set(
                result["surface_distance"][
                    "actual_sample_count_by_direction"
                ]
            ),
            {"source", "collider"},
        )

    def test_multiresolution_occupancy_uses_bounded_median(self) -> None:
        source = object()
        collider = object()
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "collider_fidelity._occupancy_iou",
            side_effect=(
                (0.91, {"pitch_m": 0.1}),
                (0.97, {"pitch_m": 0.08}),
                (0.94, {"pitch_m": 0.06}),
            ),
        ) as occupancy:
            value, evidence = multiresolution_occupancy_iou(
                (source,),
                (collider,),
            )

        self.assertEqual(value, 0.94)
        self.assertEqual(
            evidence["method"],
            "multiresolution_voxel_component_union",
        )
        self.assertEqual(evidence["selected_statistic"], "median")
        self.assertEqual(
            [call.kwargs["resolution"] for call in occupancy.call_args_list],
            [48, 64, 80],
        )

    def test_bounded_surface_fallback_matches_brute_force(self) -> None:
        import numpy as np

        mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
        random = np.random.default_rng(1729)
        points = np.vstack(
            (
                np.asarray(
                    [
                        (0.0, 0.0, 0.0),
                        (0.8, 0.1, -0.2),
                        (-0.3, 0.6, 0.4),
                        (0.2, -0.1, 0.49),
                    ],
                    dtype=float,
                ),
                random.uniform(-0.9, 0.9, size=(64, 3)),
            )
        )
        _closest, expected, _triangles = (
            trimesh.proximity.closest_point_naive(mesh, points)
        )
        actual = _bounded_nearest_surface_distances(mesh, points)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_bounded_surface_fallback_rejects_all_degenerate_mesh(self) -> None:
        import numpy as np

        mesh = trimesh.Trimesh(
            vertices=np.asarray(
                [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
            ),
            faces=np.asarray([(0, 1, 2)]),
            process=False,
        )
        with self.assertRaisesRegex(
            ValueError,
            "no finite nondegenerate triangles",
        ):
            _bounded_nearest_surface_distances(
                mesh,
                np.asarray([(0.0, 1.0, 0.0)]),
            )

    def test_exact_box_uses_analytic_iou_instead_of_voxel_discretization(self) -> None:
        thin_box = trimesh.creation.box(extents=(0.0008, 0.008, 0.002))
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "collider_fidelity._occupancy_iou",
            return_value=(0.857142, {"pitch_m": 0.0006}),
        ):
            result = evaluate_mesh_fidelity(
                thin_box,
                thin_box.copy(),
                source_id="thin-spoke",
                collider_id="thin-spoke-collider",
                representation="ChCollisionShapeBox",
                thresholds=self.thresholds,
                sample_count=192,
                voxel_resolution=28,
            )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["occupancy_iou"], 1.0)
        self.assertEqual(result["voxel_occupancy_iou"], 0.857142)
        self.assertEqual(
            result["iou_evaluation_method"],
            "analytic_axis_aligned_box_intersection_union",
        )
        self.assertTrue(result["exact_box_invariant_match"])
        self.assertFalse(result["occupancy_iou_discretization_override"])
        self.assertTrue(result["checks"]["occupancy_iou"])

    def test_machine_precision_triangle_mesh_overrides_only_voxel_noise(self) -> None:
        source = trimesh.creation.box(extents=(0.036, 0.020, 0.010))
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "collider_fidelity._occupancy_iou",
            return_value=(0.837956, {"pitch_m": 0.001}),
        ):
            exact = evaluate_mesh_fidelity(
                source,
                source.copy(),
                source_id="slotted-static-source",
                collider_id="slotted-static-collider",
                representation="ChCollisionShapeTriangleMeshStaticNonconvex",
                thresholds=self.thresholds,
                sample_count=192,
                voxel_resolution=36,
            )

            shifted = source.copy()
            shifted.apply_translation((2.0e-6, 0.0, 0.0))
            real_error = evaluate_mesh_fidelity(
                source,
                shifted,
                source_id="slotted-static-source",
                collider_id="shifted-static-collider",
                representation="ChCollisionShapeTriangleMeshStaticNonconvex",
                thresholds=self.thresholds,
                sample_count=192,
                voxel_resolution=36,
            )

        self.assertEqual(exact["status"], "PASS")
        self.assertEqual(exact["occupancy_iou"], 1.0)
        self.assertEqual(exact["voxel_occupancy_iou"], 0.837956)
        self.assertTrue(exact["occupancy_iou_discretization_override"])
        self.assertEqual(
            exact["iou_evaluation_method"],
            "machine_precision_geometric_invariants_override_v1",
        )
        self.assertTrue(exact["checks"]["occupancy_iou"])

        self.assertEqual(real_error["status"], "FAIL")
        self.assertFalse(
            real_error["occupancy_iou_discretization_override"]
        )
        self.assertIn("occupancy_iou", real_error["failure_reasons"])

    def test_analytic_box_iou_does_not_hide_real_geometry_error(self) -> None:
        source = trimesh.creation.box(extents=(0.0008, 0.008, 0.002))
        collider = source.copy()
        collider.apply_translation((0.0001, 0.0, 0.0))
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "collider_fidelity._occupancy_iou",
            return_value=(0.857142, {"pitch_m": 0.000222}),
        ):
            result = evaluate_mesh_fidelity(
                source,
                collider,
                source_id="thin-spoke",
                collider_id="shifted-thin-spoke-collider",
                representation="ChCollisionShapeBox",
                thresholds=self.thresholds,
                sample_count=192,
                voxel_resolution=28,
            )

        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(
            result["iou_evaluation_method"],
            "analytic_axis_aligned_box_intersection_union",
        )
        self.assertLess(result["occupancy_iou"], self.thresholds.occupancy_iou_min)
        self.assertFalse(result["exact_box_invariant_match"])
        self.assertFalse(result["occupancy_iou_discretization_override"])
        self.assertIn("occupancy_iou", result["failure_reasons"])

    def test_non_axis_aligned_box_stays_on_voxel_iou(self) -> None:
        rotated = trimesh.creation.box(extents=(0.0008, 0.008, 0.002))
        rotated.apply_transform(
            trimesh.transformations.rotation_matrix(0.2, (0.0, 0.0, 1.0))
        )
        result = evaluate_mesh_fidelity(
            rotated,
            rotated.copy(),
            source_id="rotated-thin-spoke",
            collider_id="rotated-thin-spoke-collider",
            representation="ChCollisionShapeBox",
            thresholds=self.thresholds,
            sample_count=192,
            voxel_resolution=28,
        )

        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["iou_evaluation_method"], "voxelized_component_union")
        self.assertFalse(result["exact_box_invariant_match"])

    def test_translation_and_scale_faults_fail(self) -> None:
        source = trimesh.creation.box(extents=(1.0, 0.8, 0.6))
        translated = source.copy()
        translated.apply_translation((0.02, 0.0, 0.0))
        scaled = source.copy()
        scaled.apply_scale(0.96)

        translated_result = self.evaluate(source, translated)
        scaled_result = self.evaluate(source, scaled)

        self.assertEqual(translated_result["status"], "FAIL")
        self.assertIn("centroid_offset", translated_result["failure_reasons"])
        self.assertEqual(scaled_result["status"], "FAIL")
        self.assertIn("relative_volume_error", scaled_result["failure_reasons"])

    def test_decimation_and_convex_hull_substitution_fail(self) -> None:
        source = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        decimated = trimesh.creation.icosphere(subdivisions=1, radius=0.5)

        arm_x = trimesh.creation.box(extents=(1.2, 0.3, 0.3))
        arm_y = trimesh.creation.box(extents=(0.3, 1.2, 0.3))
        arm_y.apply_translation((0.45, 0.45, 0.0))
        concave = trimesh.util.concatenate((arm_x, arm_y))

        self.assertEqual(self.evaluate(source, decimated)["status"], "FAIL")
        hull_result = self.evaluate(concave, concave.convex_hull)
        self.assertEqual(hull_result["status"], "FAIL")
        self.assertTrue(
            {"surface_max", "relative_volume_error", "occupancy_iou"}
            & set(hull_result["failure_reasons"])
        )

    def test_filled_cavity_is_detected_by_bidirectional_surface_distance(self) -> None:
        outer = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        inner = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
        inner.invert()
        hollow = trimesh.util.concatenate((outer, inner))

        result = self.evaluate(hollow, outer)

        self.assertEqual(result["status"], "FAIL")
        self.assertIn("surface_max", result["failure_reasons"])

    def test_high_iou_missing_feature_cannot_pass_on_iou_alone(self) -> None:
        body = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        tab = trimesh.creation.box(extents=(0.12, 0.12, 0.16))
        tab.apply_translation((0.0, 0.0, 0.58))
        source = trimesh.util.concatenate((body, tab))

        result = self.evaluate(source, body)

        self.assertGreater(result["occupancy_iou"], 0.90)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("surface_max", result["failure_reasons"])

    def test_report_assesses_each_component_and_body_union(self) -> None:
        left = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
        right = left.copy()
        right.apply_translation((0.6, 0.0, 0.0))
        report = build_collider_fidelity_report(
            (
                {
                    "body_id": "ground",
                    "solid_id": "left",
                    "representation": "exact_triangle_mesh",
                    "source_mesh": left,
                    "collider_mesh": left.copy(),
                },
                {
                    "body_id": "ground",
                    "solid_id": "right",
                    "representation": "exact_triangle_mesh",
                    "source_mesh": right,
                    "collider_mesh": right.copy(),
                },
            ),
            package_binding_sha256="a" * 64,
            characteristic_length_m=1.2,
            mesh_linear_deflection_m=0.0002,
            collision_envelope_m=0.000012,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["rigid_body_count"], 1)
        self.assertTrue(report["compound_body_assessed_as_component_union"])
        self.assertFalse(report["enclosing_convex_hull_used_for_assessment"])

    def test_intersecting_exact_components_do_not_fail_on_open_concatenation(
        self,
    ) -> None:
        body = _promote_componentwise_body_union_result(
            {
                "status": "INCONCLUSIVE",
                "passed": False,
                "failure_reasons": [],
                "inconclusive_metrics": [
                    "centroid_offset:not_closed",
                    "inertia_relative_error:not_closed_or_degenerate",
                    "relative_volume_error:not_closed",
                ],
                "checks": {
                    "surface_rms": True,
                    "surface_p95": True,
                    "surface_max": True,
                    "relative_volume_error": None,
                    "centroid_offset": None,
                    "inertia_relative_error": None,
                    "occupancy_iou": True,
                },
            },
            component_results=(
                {"status": "PASS", "passed": True},
                {"status": "PASS", "passed": True},
            ),
        )

        self.assertTrue(body["passed"])
        self.assertEqual(body["status"], "PASS")
        self.assertEqual(
            body["body_union_metric_transfer"]["method"],
            "componentwise_pass_plus_union_surface_occupancy_v1",
        )
        self.assertIsNone(body["checks"]["relative_volume_error"])
        self.assertIsNone(body["checks"]["centroid_offset"])
        self.assertIsNone(body["checks"]["inertia_relative_error"])

    def test_exact_runtime_transfer_avoids_duplicate_fidelity_evaluation(
        self,
    ) -> None:
        source = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
        collider = source.copy()
        package_binding = "a" * 64
        representation = (
            "ChCollisionShapeTriangleMeshDynamicConvexComponents"
        )
        thresholds = derive_fidelity_thresholds(
            characteristic_length_m=1.2,
            mesh_linear_deflection_m=0.0002,
            collision_envelope_m=0.000012,
        )
        evaluated = evaluate_mesh_fidelity(
            source,
            collider,
            source_id="solid_2",
            collider_id="solid_2:authored_convex_component_union",
            representation=representation,
            thresholds=thresholds,
        )
        transfer = ExactRuntimeMeshFidelityTransfer(
            result=evaluated,
            package_binding_sha256=package_binding,
            source_id="solid_2",
            collider_id="solid_2:authored_convex_component_union",
            representation=representation,
            component_ids=("arm", "pedal"),
            binding_method=(
                "deterministic_obj_round_trip_topology_and_vertices_v1"
            ),
        )

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "collider_fidelity.evaluate_mesh_fidelity",
            side_effect=AssertionError(
                "transferred component and one-source body must not rerun"
            ),
        ):
            report = build_collider_fidelity_report(
                (
                    {
                        "body_id": "input_crank",
                        "solid_id": "solid_2",
                        "collider_id": (
                            "solid_2:authored_convex_component_union"
                        ),
                        "representation": representation,
                        "source_mesh": source,
                        "collider_mesh": collider,
                        "_exact_runtime_mesh_fidelity_transfer": transfer,
                    },
                ),
                package_binding_sha256=package_binding,
                characteristic_length_m=1.2,
                mesh_linear_deflection_m=0.0002,
                collision_envelope_m=0.000012,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["component_count"], 1)
        self.assertEqual(report["rigid_body_count"], 1)
        self.assertTrue(
            report["components"][0]["evaluation_transfer"][
                "runtime_mesh_topology_and_vertices_identical"
            ]
        )
        self.assertEqual(
            report["rigid_bodies"][0]["body_union_metric_transfer"][
                "method"
            ],
            "single_source_body_union_identity_v1",
        )

    def test_report_binding_is_hash_and_package_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report_path = root / "collider_fidelity.json"
            report = {
                "schema_version": "collider_fidelity_v1",
                "status": "PASS",
                "passed": True,
                "package_binding_sha256": "a" * 64,
            }
            write_json_atomic(report_path, report)
            binding = {
                "report": "collider_fidelity.json",
                "report_sha256": sha256_file(report_path),
                "package_binding_sha256": "a" * 64,
            }

            loaded = verify_collider_fidelity_binding(
                binding,
                workspace_root=root,
                package_binding_sha256="a" * 64,
            )
            self.assertTrue(loaded["passed"])

            report["status"] = "FAIL"
            write_json_atomic(report_path, report)
            with self.assertRaises(MechanismValidationError) as caught:
                verify_collider_fidelity_binding(
                    binding,
                    workspace_root=root,
                    package_binding_sha256="a" * 64,
                )
            self.assertEqual(
                caught.exception.code,
                "COLLIDER_FIDELITY_REPORT_CHANGED",
            )

    def test_missing_report_binding_fails_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            verify_collider_fidelity_binding(
                None,
                workspace_root=Path("."),
                package_binding_sha256="a" * 64,
            )
        self.assertEqual(
            caught.exception.code,
            "COLLIDER_FIDELITY_EVIDENCE_REQUIRED",
        )


if __name__ == "__main__":
    unittest.main()
