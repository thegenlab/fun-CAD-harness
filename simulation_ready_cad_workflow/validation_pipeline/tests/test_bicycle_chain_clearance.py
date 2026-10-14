from __future__ import annotations

import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import trimesh

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_path import (
    build_moving_chain_path_contract,
    build_moving_chain_path_state,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    sha256_file,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_legality import (
    audit_bicycle_moving_chain_path_clearance,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


def _spec(
    *,
    input_contact_component_ids: tuple[str, ...] = ("chainring",),
) -> SimpleNamespace:
    return SimpleNamespace(
        input_body_id="crank",
        output_body_id="rear_wheel",
        axis_abs=(0.0, 1.0, 0.0),
        input_pitch_radius_m=0.04,
        output_pitch_radius_m=0.02,
        input_sprocket=SimpleNamespace(
            pitch_loop_center_abs_m=(0.0, 0.0, 0.0),
            step_solid_id="chainring",
            toroidal_minor_radius_m=0.004,
            contact_collision_component_ids=input_contact_component_ids,
            approved_contact_collision_component_ids=("chainring",),
        ),
        output_sprocket=SimpleNamespace(
            pitch_loop_center_abs_m=(0.2, 0.0, 0.0),
            step_solid_id="rear_sprocket",
            toroidal_minor_radius_m=0.002,
            contact_collision_component_ids=("rear_sprocket",),
            approved_contact_collision_component_ids=("rear_sprocket",),
        ),
        geometry_gate=SimpleNamespace(
            visible_chain_wrap_band_m=0.003,
            maximum_chain_sprocket_separation_m=0.001,
        ),
    )


def _contract(
    *,
    input_contact_component_ids: tuple[str, ...] = ("chainring",),
) -> dict:
    return build_moving_chain_path_contract(
        _spec(
            input_contact_component_ids=input_contact_component_ids,
        ),
        {
            "declared": True,
            "passed": True,
            "audited_collision_radius_m": 0.001,
        },
    )


def _pose() -> dict:
    return {
        "position_abs_m": [0.0, 0.0, 0.0],
        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }


class BicycleChainClearanceTests(unittest.TestCase):
    def _component(
        self,
        workspace: Path,
        name: str,
        mesh: trimesh.Trimesh,
        *,
        source_solid_id: str | None = None,
    ) -> dict:
        path = workspace / "collision" / f"{name}.stl"
        path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(path)
        return {
            "component_id": name,
            "solid_id": source_solid_id or name,
            "mesh": path.relative_to(workspace).as_posix(),
            "sha256": sha256_file(path),
        }

    def _geometry(
        self,
        workspace: Path,
        *,
        blocked: bool,
        deeply_embedded: bool = False,
        same_solid_sibling_blocked: bool = False,
        contact_extension_position: tuple[float, float, float] | None = None,
    ) -> dict:
        input_sprocket = trimesh.creation.cylinder(
            sections=32,
            radius=0.05 if deeply_embedded else 0.038,
            height=0.004,
        )
        input_sprocket.apply_transform(
            trimesh.transformations.rotation_matrix(
                math.pi / 2.0,
                [1.0, 0.0, 0.0],
            )
        )
        output_sprocket = trimesh.creation.cylinder(
            sections=32,
            radius=0.018,
            height=0.002,
        )
        output_sprocket.apply_transform(
            trimesh.transformations.rotation_matrix(
                math.pi / 2.0,
                [1.0, 0.0, 0.0],
            )
        )
        output_sprocket.apply_translation([0.2, 0.0, 0.0])
        blocker = trimesh.creation.box(extents=[0.018, 0.018, 0.018])
        blocker.apply_translation(
            [0.1, 0.0 if blocked else 0.2, -0.03]
        )
        sibling = trimesh.creation.icosphere(subdivisions=2, radius=0.01)
        sibling.apply_translation(
            [0.0, 0.0, -0.04 if same_solid_sibling_blocked else -0.2]
        )
        input_components = [
            self._component(
                workspace,
                "chainring",
                input_sprocket,
            ),
            self._component(
                workspace,
                "crank_arm",
                sibling,
                source_solid_id="chainring",
            ),
        ]
        if contact_extension_position is not None:
            contact_extension = trimesh.creation.icosphere(
                subdivisions=2,
                radius=0.001,
            )
            contact_extension.apply_translation(
                contact_extension_position
            )
            input_components.append(
                self._component(
                    workspace,
                    "chainring_contact_segment",
                    contact_extension,
                    source_solid_id="chainring",
                )
            )
        return {
            "mesh_linear_deflection_m": 0.0002,
            "runtime_interpenetration_tolerance_m": 0.00005,
            "interpenetration_tolerance_m": 0.00035,
            "collision_envelope_m": 0.0,
            "bodies": [
                {
                    "body_id": "crank",
                    "fixed": False,
                    "components": input_components,
                },
                {
                    "body_id": "rear_wheel",
                    "fixed": False,
                    "components": [
                        self._component(
                            workspace,
                            "rear_sprocket",
                            output_sprocket,
                        )
                    ],
                },
                {
                    "body_id": "frame",
                    "fixed": True,
                    "components": [
                        self._component(workspace, "blocker", blocker)
                    ],
                },
            ],
        }

    def _samples(self, contract: dict) -> list[dict]:
        samples = []
        for time_s, input_angle, output_angle in (
            (0.0, 0.0, 0.0),
            (0.1, 0.2, 0.4),
        ):
            error = 0.04 * input_angle - 0.02 * output_angle
            samples.append(
                {
                    "time_s": time_s,
                    "phase_id": "powered",
                    "input_crank_angle_rad": input_angle,
                    "rear_drive_wheel_angle_rad": output_angle,
                    "chain_coordinate_error_m": error,
                    "moving_chain_path_state": build_moving_chain_path_state(
                        contract,
                        input_angle_rad=input_angle,
                        output_angle_rad=output_angle,
                        coordinate_error_m=error,
                    ),
                    "body_poses": {
                        "crank": _pose(),
                        "rear_wheel": _pose(),
                        "frame": _pose(),
                    },
                }
            )
        return samples

    def test_matching_sprockets_are_allowed_only_in_fixed_flex_zones(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract()
            audit = audit_bicycle_moving_chain_path_clearance(
                self._geometry(workspace, blocked=False),
                self._samples(contract),
                contract,
                workspace_root=workspace,
            )

            self.assertTrue(audit["passed"])
            self.assertTrue(audit["saved_state_clearance_verified"])
            self.assertEqual(audit["audit_sample_count"], 2)
            self.assertIsNone(
                audit[
                    "maximum_observed_matching_sprocket_penetration_m"
                ]
            )
            self.assertFalse(
                audit["matching_sprocket_runtime_epa_depth_observed"]
            )
            self.assertAlmostEqual(
                audit["maximum_matching_sprocket_penetration_m"],
                audit[
                    "maximum_mechanical_matching_sprocket_penetration_m"
                ]
                + audit["collision_mesh_linear_deflection_m"],
            )

    def test_matching_sprocket_contact_cannot_bury_the_whole_chain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract()
            with self.assertRaises(MechanismValidationError) as caught:
                audit_bicycle_moving_chain_path_clearance(
                    self._geometry(
                        workspace,
                        blocked=False,
                        deeply_embedded=True,
                    ),
                    self._samples(contract),
                    contract,
                    workspace_root=workspace,
                )
            self.assertEqual(
                caught.exception.code,
                "BICYCLE_CHAIN_CONTACT_COMPONENT_OUTSIDE_SPROCKET",
            )

    def test_component_crossing_tangent_span_fails_saved_state_audit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract()
            audit = audit_bicycle_moving_chain_path_clearance(
                self._geometry(workspace, blocked=True),
                self._samples(contract),
                contract,
                workspace_root=workspace,
            )

            self.assertFalse(audit["passed"])
            self.assertGreater(audit["illegal_event_count"], 0)
            self.assertTrue(
                any(
                    event["solid_id"] == "blocker"
                    and event["chain_region_id"] == "rigid.tangent_span"
                    for event in audit["worst_events"]
                )
            )

    def test_unapproved_sibling_from_same_source_solid_is_not_exempt(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract()
            audit = audit_bicycle_moving_chain_path_clearance(
                self._geometry(
                    workspace,
                    blocked=False,
                    same_solid_sibling_blocked=True,
                ),
                self._samples(contract),
                contract,
                workspace_root=workspace,
            )

            self.assertFalse(audit["passed"])
            self.assertTrue(
                any(
                    event["component_id"] == "crank_arm"
                    and event["solid_id"] == "chainring"
                    and event["reason"] == "forbidden_component_contact"
                    for event in audit["worst_events"]
                )
            )

    def test_design_bound_extension_on_public_torus_uses_local_certification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract(
                input_contact_component_ids=(
                    "chainring",
                    "chainring_contact_segment",
                )
            )
            audit = audit_bicycle_moving_chain_path_clearance(
                self._geometry(
                    workspace,
                    blocked=False,
                    contact_extension_position=(0.04, 0.0, 0.0),
                ),
                self._samples(contract),
                contract,
                workspace_root=workspace,
            )

            self.assertTrue(audit["passed"])
            certification = audit[
                "sprocket_contact_component_geometry_audit"
            ]
            self.assertTrue(certification["passed"])
            self.assertEqual(certification["extension_component_count"], 1)
            self.assertEqual(
                audit[
                    "geometry_certified_approved_seed_component_ids_by_body"
                ]["crank"],
                ["chainring"],
            )
            self.assertEqual(
                audit[
                    "geometry_certified_local_extension_component_ids_by_body"
                ]["crank"],
                ["chainring_contact_segment"],
            )
            self.assertIn(
                "local contact without unstable EPA depth",
                audit["flex_zone_policy"],
            )

    def test_design_cannot_relabel_off_torus_sibling_as_contact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract(
                input_contact_component_ids=(
                    "chainring",
                    "chainring_contact_segment",
                )
            )

            with self.assertRaises(MechanismValidationError) as caught:
                audit_bicycle_moving_chain_path_clearance(
                    self._geometry(
                        workspace,
                        blocked=False,
                        contact_extension_position=(0.0, 0.0, 0.0),
                    ),
                    self._samples(contract),
                    contract,
                    workspace_root=workspace,
                )
            self.assertEqual(
                caught.exception.code,
                "BICYCLE_CHAIN_CONTACT_COMPONENT_OUTSIDE_SPROCKET",
            )
            self.assertIn(
                "outside their public geometry envelope",
                str(caught.exception),
            )

    def test_certified_contact_cannot_move_out_of_fixed_zone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            contract = _contract()
            samples = self._samples(contract)
            samples[1]["body_poses"]["crank"] = {
                **_pose(),
                "position_abs_m": [0.01, 0.0, 0.0],
            }

            audit = audit_bicycle_moving_chain_path_clearance(
                self._geometry(workspace, blocked=False),
                samples,
                contract,
                workspace_root=workspace,
            )

            self.assertFalse(audit["passed"])
            self.assertTrue(
                any(
                    event["reason"]
                    == "sprocket_contact_envelope_pose_drift"
                    for event in audit["worst_events"]
                )
            )


if __name__ == "__main__":
    unittest.main()
