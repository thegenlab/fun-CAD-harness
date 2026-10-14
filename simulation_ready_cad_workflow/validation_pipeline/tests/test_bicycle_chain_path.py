from __future__ import annotations

from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_path import (
    audit_moving_chain_path_states,
    build_external_tangent_centerline,
    build_moving_chain_path_contract,
    build_moving_chain_path_state,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


def _spec() -> SimpleNamespace:
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
            contact_collision_component_ids=("chainring",),
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


def _contract() -> dict:
    return build_moving_chain_path_contract(
        _spec(),
        {
            "declared": True,
            "passed": True,
            "audited_collision_radius_m": 0.001,
        },
    )


def _samples(contract: dict) -> list[dict]:
    rows = []
    for time_s, input_angle, output_angle in (
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 2.0),
    ):
        coordinate_error = 0.04 * input_angle - 0.02 * output_angle
        rows.append(
            {
                "time_s": time_s,
                "input_crank_angle_rad": input_angle,
                "rear_drive_wheel_angle_rad": output_angle,
                "chain_coordinate_error_m": coordinate_error,
                "moving_chain_path_state": build_moving_chain_path_state(
                    contract,
                    input_angle_rad=input_angle,
                    output_angle_rad=output_angle,
                    coordinate_error_m=coordinate_error,
                ),
            }
        )
    return rows


class BicycleChainPathTests(unittest.TestCase):
    def test_unequal_radius_path_wraps_far_side_without_hidden_chord(self) -> None:
        path = build_external_tangent_centerline(
            input_center_abs_m=(0.0, 0.0, 0.0),
            output_center_abs_m=(0.2, 0.0, 0.0),
            axis_abs=(0.0, 1.0, 0.0),
            input_pitch_radius_m=0.04,
            output_pitch_radius_m=0.02,
        )
        points = path["centerline_points_abs_m"]

        input_arc = points[128:]
        output_arc = points[32:97]
        self.assertTrue(
            all(math.isclose(math.dist(point, (0.0, 0.0, 0.0)), 0.04, abs_tol=1.0e-12) for point in input_arc)
        )
        self.assertTrue(
            all(math.isclose(math.dist(point, (0.2, 0.0, 0.0)), 0.02, abs_tol=1.0e-12) for point in output_arc)
        )
        self.assertLess(input_arc[len(input_arc) // 2][0], -0.039)
        self.assertLess(max(path["edge_lengths_m"]), 0.01)
        self.assertLess(math.dist(points[0], points[-1]), 1.0e-12)

    def test_external_tangent_path_and_saved_states_are_rederived(self) -> None:
        contract = _contract()
        audit = audit_moving_chain_path_states(contract, _samples(contract))

        self.assertTrue(audit["passed"])
        self.assertTrue(audit["external_tangent_geometry_rederived"])
        self.assertTrue(audit["wrap_angles_rederived"])
        self.assertLess(
            math.dist(
                contract["centerline_points_abs_m"][0],
                contract["centerline_points_abs_m"][-1],
            ),
            1.0e-12,
        )
        self.assertEqual(
            {zone["stationary_frame"] for zone in contract["flex_zones"]},
            {"authored_world"},
        )
        zones = {
            zone["zone_id"]: zone
            for zone in contract["flex_zones"]
        }
        for left, right, region in zip(
            contract["centerline_points_abs_m"],
            contract["centerline_points_abs_m"][1:],
            contract["edge_region_ids"],
        ):
            if not region.startswith("flex."):
                continue
            zone = zones[region]
            midpoint = [
                0.5 * (left[index] + right[index])
                for index in range(3)
            ]
            self.assertTrue(
                all(
                    math.dist(point, zone["center_abs_m"])
                    <= zone["radius_m"]
                    for point in (left, midpoint, right)
                )
            )

    def test_rehashed_but_geometrically_changed_path_is_rejected(self) -> None:
        contract = deepcopy(_contract())
        contract["centerline_points_abs_m"][10][2] += 0.001
        core = {
            key: value
            for key, value in contract.items()
            if key != "contract_sha256"
        }
        contract["contract_sha256"] = canonical_sha256(core)

        with self.assertRaises(MechanismValidationError) as caught:
            audit_moving_chain_path_states(contract, _samples(contract))
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_GEOMETRY_MISMATCH",
        )

    def test_rehashed_flex_zone_center_change_is_rejected(self) -> None:
        contract = deepcopy(_contract())
        contract["flex_zones"][0]["center_abs_m"][2] += 0.001
        core = {
            key: value
            for key, value in contract.items()
            if key != "contract_sha256"
        }
        contract["contract_sha256"] = canonical_sha256(core)

        with self.assertRaises(MechanismValidationError) as caught:
            audit_moving_chain_path_states(contract, _samples(contract))
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_REGION_INVALID",
        )

    def test_rehashed_open_path_is_rejected_as_topology_error(self) -> None:
        contract = deepcopy(_contract())
        contract["centerline_points_abs_m"][-1][2] += 0.001
        core = {
            key: value
            for key, value in contract.items()
            if key != "contract_sha256"
        }
        contract["contract_sha256"] = canonical_sha256(core)

        with self.assertRaises(MechanismValidationError) as caught:
            audit_moving_chain_path_states(contract, _samples(contract))
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_TOPOLOGY_INVALID",
        )

    def test_changed_saved_phase_digest_is_rejected(self) -> None:
        contract = _contract()
        samples = _samples(contract)
        samples[1]["moving_chain_path_state"]["phase_modulo_m"] += 0.001

        with self.assertRaises(MechanismValidationError) as caught:
            audit_moving_chain_path_states(contract, samples)
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_STATE_CHANGED",
        )

    def test_every_saved_sample_requires_a_chain_path_state(self) -> None:
        contract = _contract()
        samples = _samples(contract)
        samples[1].pop("moving_chain_path_state")

        with self.assertRaises(MechanismValidationError) as caught:
            audit_moving_chain_path_states(contract, samples)
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_STATE_MISSING",
        )

    def test_saved_sample_time_must_be_finite_and_strictly_increasing(
        self,
    ) -> None:
        for invalid_times in ((0.0, 0.0), (math.nan, 1.0)):
            with self.subTest(invalid_times=invalid_times):
                contract = _contract()
                samples = _samples(contract)
                for sample, time_s in zip(samples, invalid_times):
                    sample["time_s"] = time_s

                with self.assertRaises(MechanismValidationError) as caught:
                    audit_moving_chain_path_states(contract, samples)
                self.assertEqual(
                    caught.exception.code,
                    "BICYCLE_CHAIN_PATH_PHASE_DISCONTINUITY",
                )

    def test_rehashed_contract_cannot_omit_material_motion_direction(
        self,
    ) -> None:
        contract = _contract()
        contract.pop("material_motion_per_phase_sign")
        core = {
            key: value
            for key, value in contract.items()
            if key != "contract_sha256"
        }
        contract["contract_sha256"] = canonical_sha256(core)

        with self.assertRaises(MechanismValidationError) as caught:
            audit_moving_chain_path_states(contract, _samples(contract))
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_EVIDENCE_INVALID",
        )

    def test_state_must_use_exact_force_coordinate(self) -> None:
        contract = _contract()
        with self.assertRaises(MechanismValidationError) as caught:
            build_moving_chain_path_state(
                contract,
                input_angle_rad=1.0,
                output_angle_rad=2.0,
                coordinate_error_m=0.001,
            )
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PATH_FORCE_COORDINATE_MISMATCH",
        )


if __name__ == "__main__":
    unittest.main()
