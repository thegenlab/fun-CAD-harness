from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
    approved_bicycle_chain_family_spec_fixture,
    bicycle_chain_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.initial_functional_geometry import (
    _bicycle_chain_preflight,
    _bicycle_role_solid_ids,
    _bind_bicycle_pitch_surface,
    _circular_coverage_rad,
    _declared_visible_chain_rows,
    _step_bicycle_torus_inventory,
    _visible_chain_measurement_failures,
    _visible_chain_wrap_metrics,
)


_INPUT_CENTER = (-0.25, -0.078, 0.34)
_OUTPUT_CENTER = (-0.85, -0.078, 0.38)
_CONTEXT_INPUT_CENTER = (-0.25, -0.098, 0.34)
_CONTEXT_OUTPUT_CENTER = (-0.85, -0.098, 0.38)
_AXIS = (0.0, 1.0, 0.0)


def _surface_records(
    start: int,
    *,
    center_mm: tuple[float, float, float],
    axis: tuple[float, float, float],
    major_mm: float,
    minor_mm: float,
) -> tuple[list[str], int, int]:
    point, direction, reference, placement, surface, face = range(
        start,
        start + 6,
    )
    records = [
        f"#{point} = CARTESIAN_POINT('',({center_mm[0]},{center_mm[1]},{center_mm[2]}));",
        f"#{direction} = DIRECTION('',({axis[0]},{axis[1]},{axis[2]}));",
        f"#{reference} = DIRECTION('',(1.,0.,0.));",
        f"#{placement} = AXIS2_PLACEMENT_3D('',#{point},#{direction},#{reference});",
        f"#{surface} = TOROIDAL_SURFACE('',#{placement},{major_mm},{minor_mm});",
        f"#{face} = ADVANCED_FACE('',(),#{surface},.T.);",
    ]
    return records, start + 6, face


def _solid_records(
    start: int,
    *,
    face_ids: tuple[int, ...],
) -> tuple[list[str], int]:
    shell, solid = range(start, start + 2)
    faces = ",".join(f"#{face_id}" for face_id in face_ids)
    return [
        f"#{shell} = CLOSED_SHELL('',({faces}));",
        f"#{solid} = MANIFOLD_SOLID_BREP('',#{shell});",
    ], start + 2


def _inventory(
    *,
    input_major_mm: float = 80.0,
    input_minor_mm: float = 8.0,
    output_minor_mm: float = 5.0,
    input_center_mm: tuple[float, float, float] = (-250.0, -78.0, 340.0),
    output_center_mm: tuple[float, float, float] = (-850.0, -78.0, 380.0),
    context_y_mm: float = -98.0,
    input_axis: tuple[float, float, float] = _AXIS,
    output_axis: tuple[float, float, float] = _AXIS,
) -> tuple[dict[str, object], ...]:
    records = ["ISO-10303-21;", "SI_UNIT(.MILLI.,.METRE.);"]
    next_id = 10
    surfaces = (
        # solid_1: fixed decoy with the correct input dimensions and pose.
        ((-250.0, -78.0, 340.0), _AXIS, 80.0, 8.0),
        (input_center_mm, input_axis, input_major_mm, input_minor_mm),
        (output_center_mm, output_axis, 40.0, output_minor_mm),
    )
    for center, axis, major, minor in surfaces:
        surface, next_id, face_id = _surface_records(
            next_id,
            center_mm=center,
            axis=axis,
            major_mm=major,
            minor_mm=minor,
        )
        solid, next_id = _solid_records(next_id, face_ids=(face_id,))
        records.extend(surface)
        records.extend(solid)

    # solid_4: the context-only chain visualization owns both 80/3 and 40/3
    # toroidal wrap surfaces at the approved -98 mm visual plane.
    context_input, next_id, context_input_face = _surface_records(
        next_id,
        center_mm=(-250.0, context_y_mm, 340.0),
        axis=_AXIS,
        major_mm=80.0,
        minor_mm=3.0,
    )
    context_output, next_id, context_output_face = _surface_records(
        next_id,
        center_mm=(-850.0, context_y_mm, 380.0),
        axis=_AXIS,
        major_mm=40.0,
        minor_mm=3.0,
    )
    context_solid, next_id = _solid_records(
        next_id,
        face_ids=(context_input_face, context_output_face),
    )
    records.extend(context_input)
    records.extend(context_output)
    records.extend(context_solid)
    records.append("END-ISO-10303-21;")
    with tempfile.TemporaryDirectory() as temporary:
        step = Path(temporary) / "assembly.step"
        step.write_text("\n".join(records), encoding="ascii")
        return _step_bicycle_torus_inventory(step)


def _contract() -> dict[str, object]:
    return {
        "rigid_groups": [
            {"rigid_group_id": "fixed_group", "part_ids": ["fixed_part"]},
            {"rigid_group_id": "input_group", "part_ids": ["input_part"]},
            {"rigid_group_id": "output_group", "part_ids": ["output_part"]},
            {"rigid_group_id": "context_group", "part_ids": ["context_part"]},
        ],
        "parts": [
            {
                "part_id": "fixed_part",
                "occurrence_id": "fixed_occurrence",
                "rigid_group_id": "fixed_group",
                "role": "fixed",
                "source_solid_ids": ["solid_1"],
            },
            {
                "part_id": "input_part",
                "occurrence_id": "input_occurrence",
                "rigid_group_id": "input_group",
                "role": "dynamic",
                "source_solid_ids": ["solid_2"],
            },
            {
                "part_id": "output_part",
                "occurrence_id": "output_occurrence",
                "rigid_group_id": "output_group",
                "role": "dynamic",
                "source_solid_ids": ["solid_3"],
            },
            {
                "part_id": "context_part",
                "occurrence_id": "context_occurrence",
                "rigid_group_id": "context_group",
                "role": "context",
                "source_solid_ids": ["solid_4"],
            },
        ],
    }


def _role_solids(contract: dict[str, object], role: str) -> frozenset[str]:
    return _bicycle_role_solid_ids(
        contract,
        role=role,
        part_id=f"{role}_part",
        occurrence_id=f"{role}_occurrence",
        allowed_group_ids=frozenset({f"{role}_group"}),
    )


def _bind(
    inventory: tuple[dict[str, object], ...],
    *,
    role: str,
    allowed_solid_ids: frozenset[str],
    center: tuple[float, float, float] | None = None,
    major: float | None = None,
    minor: float | None = None,
    axis: tuple[float, float, float] = _AXIS,
    minimum_faces: int = 1,
) -> tuple[dict[str, object] | None, list[float]]:
    is_input = role in {"input", "context_input_wrap"}
    return _bind_bicycle_pitch_surface(
        inventory,
        role=role,
        allowed_solid_ids=allowed_solid_ids,
        origin_abs_m=center
        or (
            _CONTEXT_INPUT_CENTER
            if role == "context_input_wrap"
            else _CONTEXT_OUTPUT_CENTER
            if role == "context_output_wrap"
            else _INPUT_CENTER
            if is_input
            else _OUTPUT_CENTER
        ),
        required_radius_m=major if major is not None else (0.08 if is_input else 0.04),
        required_minor_radius_m=minor
        if minor is not None
        else (0.003 if role.startswith("context_") else 0.008 if is_input else 0.005),
        required_axis_abs=axis,
        radius_tolerance_m=0.0002,
        center_tolerance_m=0.0002,
        axis_tolerance_rad=0.002,
        minimum_exposed_face_count=minimum_faces,
    )


class BicycleChainGeometryBindingTests(unittest.TestCase):
    def test_preflight_reads_collision_geometry_without_execution_gate(
        self,
    ) -> None:
        plan = SimpleNamespace(
            capsule_id="bicycle_drivetrain",
            package_binding_sha256="b" * 64,
        )
        family_spec = approved_bicycle_chain_family_spec_fixture()
        expected_bodies = bicycle_chain_from_dict(family_spec).bodies
        family_spec["collision_geometry"] = {
            "static_geometry_preflight": {
                "status": "FAIL",
                "passed": False,
            }
        }
        adapter = SimpleNamespace(family_spec=family_spec)
        collision = {"bodies": []}
        geometry = {
            "status": "PASS",
            "passed": True,
            "failure_reasons": [],
            "assembly_sha256": "a" * 64,
            "geometry_gate": {"gate_sha256": "g" * 64},
            "visible_chain_geometry": {
                "audited_collision_radius_m": 0.001,
            },
        }
        family_module = (
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "family_collision"
        )
        with (
            patch(
                family_module
                + ".family_collision_geometry_for_static_preflight",
                return_value=collision,
            ) as preflight_geometry,
            patch(
                family_module + ".family_collision_geometry",
                side_effect=AssertionError(
                    "execution gate used during preflight"
                ),
            ) as execution_geometry,
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "bicycle_chain_runtime._bicycle_prescribed_swept_envelope_samples",
                return_value=({"body_poses": {}},),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "physical_legality.audit_bicycle_sprocket_contact_component_geometry",
                return_value={"status": "PASS", "passed": True},
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "physical_legality.audit_prescribed_swept_envelope",
                return_value={"status": "PASS", "passed": True},
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "initial_functional_geometry._bicycle_chain_geometry_evidence",
                return_value=geometry,
            ),
        ):
            result = _bicycle_chain_preflight(
                plan,
                adapter,
                workspace=Path("/tmp"),
            )

        self.assertIs(result["passed"], True)
        preflight_geometry.assert_called_once_with(
            adapter,
            expected_bodies=expected_bodies,
        )
        execution_geometry.assert_not_called()

    def test_non_chain_context_part_does_not_declare_a_visible_chain(self) -> None:
        rows = [
            {
                "part_id": "rear_reflector",
                "role": "context",
                "source_solid_ids": ["solid_9"],
            }
        ]

        self.assertEqual(
            _declared_visible_chain_rows(
                rows,
                visible_chain_part_id="visible_chain",
            ),
            [],
        )

    def test_circular_coverage_uses_shortest_arc_across_angle_seam(
        self,
    ) -> None:
        coverage = _circular_coverage_rad(
            (-math.pi + 0.1, math.pi - 0.1)
        )
        self.assertAlmostEqual(coverage, 0.2, places=12)

    def test_visible_chain_wrap_reports_true_axial_plane_offset(self) -> None:
        points = [
            (
                0.08 * math.cos(-1.0 + index / 12.0),
                -0.03,
                0.08 * math.sin(-1.0 + index / 12.0),
            )
            for index in range(25)
        ]
        metrics = _visible_chain_wrap_metrics(
            points,
            center_abs_m=(0.0, 0.0, 0.0),
            axis_abs=(0.0, 1.0, 0.0),
            pitch_radius_m=0.08,
            wrap_band_m=0.001,
        )
        self.assertEqual(metrics["near_pitch_band_sample_count"], 25)
        self.assertAlmostEqual(metrics["wrap_angle_rad"], 2.0, places=12)
        self.assertAlmostEqual(
            metrics["pitch_plane_offset_m"],
            0.03,
            places=12,
        )
        self.assertAlmostEqual(
            metrics["maximum_pitch_plane_deviation_m"],
            0.03,
            places=12,
        )

    def test_visible_chain_plane_offset_ignores_symmetric_thickness(
        self,
    ) -> None:
        points = [
            (
                0.08 * math.cos(-1.0 + index / 12.0),
                -0.004 if index % 2 else 0.004,
                0.08 * math.sin(-1.0 + index / 12.0),
            )
            for index in range(25)
        ]
        metrics = _visible_chain_wrap_metrics(
            points,
            center_abs_m=(0.0, 0.0, 0.0),
            axis_abs=(0.0, 1.0, 0.0),
            pitch_radius_m=0.08,
            wrap_band_m=0.001,
        )

        self.assertAlmostEqual(metrics["pitch_plane_offset_m"], 0.0)
        self.assertAlmostEqual(
            metrics["maximum_pitch_plane_deviation_m"],
            0.004,
        )
        self.assertEqual(
            _visible_chain_measurement_failures(
                {
                    "audited_collision_radius_m": 0.003,
                    "input": {
                        **metrics,
                        "minimum_sprocket_separation_m": 0.0,
                    },
                    "output": {
                        **metrics,
                        "minimum_sprocket_separation_m": 0.0,
                    },
                },
                maximum_chain_sprocket_separation_m=0.002,
                minimum_wrap_angle_rad=1.5,
                minimum_wrap_sample_count=12,
                maximum_pitch_plane_offset_m=0.0005,
                wrap_band_m=0.012,
            ),
            [],
        )

    def test_visible_chain_public_measurements_reject_floating_path(
        self,
    ) -> None:
        role = {
            "minimum_sprocket_separation_m": 0.021,
            "wrap_angle_rad": 2.0,
            "near_pitch_band_sample_count": 25,
            "pitch_plane_offset_m": 0.03,
            "maximum_pitch_plane_deviation_m": 0.03,
        }
        failures = _visible_chain_measurement_failures(
            {
                "audited_collision_radius_m": 0.003,
                "input": role,
                "output": role,
            },
            maximum_chain_sprocket_separation_m=0.002,
            minimum_wrap_angle_rad=1.5,
            minimum_wrap_sample_count=12,
            maximum_pitch_plane_offset_m=0.0005,
            wrap_band_m=0.012,
        )

        self.assertTrue(
            any("input_separation_exceeded" in item for item in failures)
        )
        self.assertTrue(
            any(
                "output_pitch_plane_offset_exceeded" in item
                for item in failures
            )
        )

    def test_positive_r04_drive_and_context_surfaces_bind_exact_solids(self) -> None:
        inventory = _inventory()
        contract = _contract()
        input_surface, _ = _bind(
            inventory,
            role="input",
            allowed_solid_ids=_role_solids(contract, "input"),
        )
        output_surface, _ = _bind(
            inventory,
            role="output",
            allowed_solid_ids=_role_solids(contract, "output"),
        )
        context_input, _ = _bind(
            inventory,
            role="context_input_wrap",
            allowed_solid_ids=frozenset({"solid_4"}),
        )
        context_output, _ = _bind(
            inventory,
            role="context_output_wrap",
            allowed_solid_ids=frozenset({"solid_4"}),
        )

        self.assertEqual(input_surface["solid_id"], "solid_2")
        self.assertEqual(input_surface["minor_radius_m"], 0.008)
        self.assertEqual(output_surface["solid_id"], "solid_3")
        self.assertEqual(output_surface["minor_radius_m"], 0.005)
        self.assertEqual(context_input["solid_id"], "solid_4")
        self.assertEqual(context_input["minor_radius_m"], 0.003)
        self.assertEqual(context_output["solid_id"], "solid_4")
        self.assertEqual(context_output["center_abs_m"][1], -0.098)
        self.assertEqual(input_surface["exposed_face_count"], 1)
        self.assertEqual(
            input_surface["surface_axis_semantics"],
            "unoriented_surface_axis_parallel",
        )
        self.assertFalse(
            input_surface["provides_transmission_direction_evidence"]
        )

    def test_negative_wrong_minor_radius_is_rejected(self) -> None:
        surface, _ = _bind(
            _inventory(input_minor_mm=7.5),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        self.assertIsNone(surface)

    def test_negative_center_beyond_point_two_mm_is_rejected(self) -> None:
        surface, _ = _bind(
            _inventory(input_center_mm=(-250.0, -77.75, 340.0)),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        self.assertIsNone(surface)

    def test_negative_axis_over_point_zero_zero_two_rad_is_rejected(self) -> None:
        angle = 0.0025
        surface, _ = _bind(
            _inventory(
                input_axis=(math.sin(angle), math.cos(angle), 0.0),
            ),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        self.assertIsNone(surface)

    def test_reversed_surface_parameterisation_is_same_axis_line(self) -> None:
        surface, _ = _bind(
            _inventory(input_axis=(0.0, -1.0, 0.0)),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        self.assertIsNotNone(surface)
        self.assertEqual(
            surface["unoriented_surface_axis_parallel_error_rad"],
            0.0,
        )
        self.assertFalse(
            surface["provides_transmission_direction_evidence"]
        )

    def test_negative_fixed_decoy_and_wrong_moving_solid_cannot_swap(self) -> None:
        inventory = _inventory(input_major_mm=72.0)
        input_surface, found = _bind(
            inventory,
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        swapped_surface, _ = _bind(
            inventory,
            role="input",
            allowed_solid_ids=frozenset({"solid_3"}),
        )
        self.assertIsNone(input_surface)
        self.assertIsNone(swapped_surface)
        self.assertEqual(found, [0.072])
        self.assertTrue(
            any(
                item["solid_id"] == "solid_1"
                and item["major_radius_m"] == 0.08
                and item["minor_radius_m"] == 0.008
                for item in inventory
            )
        )

    def test_negative_context_solid_cannot_substitute_drive_solid(self) -> None:
        drive_from_context, _ = _bind(
            _inventory(),
            role="input",
            allowed_solid_ids=frozenset({"solid_4"}),
        )
        self.assertIsNone(drive_from_context)

    def test_matching_context_torus_cannot_replace_missing_physical_torus(
        self,
    ) -> None:
        inventory = list(_inventory())
        physical_input = next(
            item
            for item in inventory
            if item["solid_id"] == "solid_2"
            and item["major_radius_m"] == 0.08
        )
        inventory.remove(physical_input)
        context_decoy = dict(physical_input)
        context_decoy["solid_id"] = "solid_4"
        inventory.append(context_decoy)
        physical_surface, _ = _bind(
            tuple(inventory),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        diagnostic_context_surface, _ = _bind(
            tuple(inventory),
            role="input",
            allowed_solid_ids=frozenset({"solid_4"}),
        )
        self.assertIsNone(physical_surface)
        self.assertIsNotNone(diagnostic_context_surface)

    def test_context_torus_pose_is_only_an_exact_optional_diagnostic(self) -> None:
        context_surface, _ = _bind(
            _inventory(context_y_mm=-97.7),
            role="context_input_wrap",
            allowed_solid_ids=frozenset({"solid_4"}),
        )
        self.assertIsNone(context_surface)

    def test_negative_minimum_exposed_face_count_is_enforced(self) -> None:
        surface, _ = _bind(
            _inventory(),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
            minimum_faces=2,
        )
        self.assertIsNone(surface)

    def test_negative_legacy_72_mm_major_radius_remains_rejected(self) -> None:
        surface, found = _bind(
            _inventory(input_major_mm=72.0),
            role="input",
            allowed_solid_ids=frozenset({"solid_2"}),
        )
        self.assertIsNone(surface)
        self.assertEqual(found, [0.072])


if __name__ == "__main__":
    unittest.main()
