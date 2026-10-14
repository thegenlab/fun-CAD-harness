from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from types import MappingProxyType
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    test_load_profiles,
)


def _phase() -> LoadPhase:
    return LoadPhase(
        phase_id="drive",
        wrench_kind="torque",
        direction_abs=(0.0, 0.0, 1.0),
        magnitude_si=1.0,
        duration_s=0.5,
    )


def _legacy_load_case() -> LoadCase:
    return LoadCase(
        load_case_id="legacy_roundtrip",
        family="revolute",
        program="task_output_angle_roundtrip",
        purpose="nominal",
        phases=(_phase(),),
        moving_mass_kg=0.2,
    )


class TestLoadProfileRegistryTest(unittest.TestCase):
    def test_registered_profile_has_exact_serialization_and_hashes(self) -> None:
        expected_profile = {
            "profile_id": "rated_resistive_torque_3nm",
            "profile_kind": "passive_test_load",
            "wrench_kind": "torque",
            "wrench_law_id": "validator_dissipative_test_load_v1",
            "target_role": "task_output",
            "magnitude_si": 3.0,
            "unit": "N*m",
            "direction_policy": "opposes_target_angular_velocity",
            "rest_policy": "zero_at_rest",
            "schema_version": "validator_test_load_profile_v1",
        }
        profile = test_load_profiles.test_load_profile_for(
            "rated_resistive_torque_3nm"
        )
        self.assertEqual(profile.to_dict(), expected_profile)
        self.assertEqual(
            profile.profile_sha256,
            "e97dd16feec714e128a6cfa49e92a8bb7ec2af8df2c530b9aba85c3542d1e25e",
        )
        expected_core = {
            "schema_version": "validator_test_load_profile_registry_v1",
            "registry_version": "test_load_profile_registry_v1",
            "profiles": [expected_profile],
        }
        self.assertEqual(
            canonical_sha256(expected_core),
            "a0e535d69af80dd6044de75c3bedb0e8571e554adcf28aee472940c6b9a02cea",
        )
        self.assertEqual(
            test_load_profiles.test_load_profile_registry_document(),
            {
                **expected_core,
                "registry_sha256": (
                    "a0e535d69af80dd6044de75c3bedb0e8571e554adcf28aee472940c6b9a02cea"
                ),
            },
        )

    def test_registry_and_profile_are_immutable(self) -> None:
        profile = test_load_profiles.registered_test_load_profiles()[0]
        with self.assertRaises(FrozenInstanceError):
            profile.magnitude_si = 4.0  # type: ignore[misc]
        with self.assertRaises(TypeError):
            test_load_profiles._TEST_LOAD_PROFILE_BY_ID["invented"] = profile

    def test_invalid_and_tampered_profiles_fail_closed(self) -> None:
        profile = test_load_profiles.registered_test_load_profiles()[0]
        with self.assertRaises(MechanismValidationError) as ctx:
            replace(profile, magnitude_si=-1.0)
        self.assertEqual(
            ctx.exception.code, "TEST_LOAD_PROFILE_MAGNITUDE_INVALID"
        )

        tampered = replace(profile, magnitude_si=4.0)
        with self.assertRaises(MechanismValidationError) as ctx:
            test_load_profiles.dissipative_test_load_torque_nm(tampered, 2.0)
        self.assertEqual(ctx.exception.code, "TEST_LOAD_PROFILE_TAMPERED")

        tampered_inventory = (tampered,)
        tampered_index = MappingProxyType({tampered.profile_id: tampered})
        with mock.patch.object(
            test_load_profiles,
            "_REGISTERED_TEST_LOAD_PROFILES",
            tampered_inventory,
        ), mock.patch.object(
            test_load_profiles,
            "_TEST_LOAD_PROFILE_BY_ID",
            tampered_index,
        ):
            with self.assertRaises(MechanismValidationError) as ctx:
                test_load_profiles.test_load_profile_registry_document()
        self.assertEqual(
            ctx.exception.code, "TEST_LOAD_PROFILE_REGISTRY_TAMPERED"
        )

    def test_dissipative_torque_opposes_velocity_and_is_zero_at_rest(
        self,
    ) -> None:
        binding = test_load_profiles.binding_for_test_load_profile(
            "rated_resistive_torque_3nm"
        )
        self.assertEqual(
            test_load_profiles.dissipative_test_load_torque_nm(binding, 2.5),
            -3.0,
        )
        self.assertEqual(
            test_load_profiles.dissipative_test_load_torque_nm(binding, -2.5),
            3.0,
        )
        self.assertEqual(
            test_load_profiles.dissipative_test_load_torque_nm(binding, 0.0),
            0.0,
        )
        for velocity in (-2.5, 2.5):
            torque = test_load_profiles.dissipative_test_load_torque_nm(
                binding, velocity
            )
            self.assertLessEqual(torque * velocity, 0.0)

    def test_unknown_profile_and_tampered_binding_fail_closed(self) -> None:
        with self.assertRaises(MechanismValidationError) as ctx:
            test_load_profiles.test_load_profile_for("not_registered")
        self.assertEqual(ctx.exception.code, "TEST_LOAD_PROFILE_NOT_REGISTERED")

        binding = test_load_profiles.binding_for_test_load_profile(
            "rated_resistive_torque_3nm"
        )
        tampered = replace(binding, profile_sha256="0" * 64)
        with self.assertRaises(MechanismValidationError) as ctx:
            test_load_profiles.validate_test_load_profile_binding(tampered)
        self.assertEqual(
            ctx.exception.code, "TEST_LOAD_PROFILE_BINDING_MISMATCH"
        )


class LoadCaseProfileBindingTest(unittest.TestCase):
    def test_legacy_serialization_shape_is_exactly_unchanged(self) -> None:
        load_case = _legacy_load_case()
        expected = {
            "load_case_id": "legacy_roundtrip",
            "family": "revolute",
            "program": "task_output_angle_roundtrip",
            "purpose": "nominal",
            "phases": [
                {
                    "phase_id": "drive",
                    "wrench_kind": "torque",
                    "direction_abs": [0.0, 0.0, 1.0],
                    "magnitude_si": 1.0,
                    "duration_s": 0.5,
                    "stop_condition": None,
                    "target_progress_m": None,
                    "target_position_si": None,
                }
            ],
            "moving_mass_kg": 0.2,
            "step_s": 0.001,
            "sample_interval_s": 0.05,
            "required_for_certificate": True,
        }
        self.assertEqual(load_case.to_dict(), expected)
        self.assertNotIn("test_load_case_id", load_case.to_dict())
        self.assertNotIn("load_profile_binding", load_case.to_dict())
        self.assertEqual(
            json.dumps(load_case.to_dict(), separators=(",", ":")),
            json.dumps(expected, separators=(",", ":")),
        )
        self.assertEqual(
            load_case.canonical_test_load_case_id,
            load_case.load_case_id,
        )
        self.assertEqual(LoadCase.from_dict(expected), load_case)

    def test_external_selector_and_frozen_binding_round_trip(self) -> None:
        binding = test_load_profiles.binding_for_test_load_profile(
            "rated_resistive_torque_3nm"
        )
        load_case = LoadCase(
            load_case_id="roundtrip_under_rated_resistance",
            family="revolute",
            program="task_output_angle_roundtrip",
            purpose="nominal",
            phases=(_phase(),),
            moving_mass_kg=0.2,
            test_load_case_id="rated_resistive_torque_3nm",
            load_profile_binding=binding,
        )
        self.assertEqual(
            load_case.canonical_test_load_case_id,
            "rated_resistive_torque_3nm",
        )
        self.assertEqual(
            load_case.to_dict()["load_profile_binding"], binding.to_dict()
        )
        self.assertEqual(LoadCase.from_dict(load_case.to_dict()), load_case)

    def test_selector_and_binding_are_all_or_none_and_must_match(self) -> None:
        binding = test_load_profiles.binding_for_test_load_profile(
            "rated_resistive_torque_3nm"
        )
        legacy = _legacy_load_case().to_dict()
        with self.assertRaises(MechanismValidationError) as ctx:
            LoadCase.from_dict(
                {**legacy, "test_load_case_id": binding.profile_id}
            )
        self.assertEqual(
            ctx.exception.code, "TEST_LOAD_PROFILE_BINDING_INCOMPLETE"
        )
        with self.assertRaises(MechanismValidationError) as ctx:
            LoadCase.from_dict(
                {
                    **legacy,
                    "test_load_case_id": "other_profile",
                    "load_profile_binding": binding.to_dict(),
                }
            )
        self.assertEqual(
            ctx.exception.code, "TEST_LOAD_PROFILE_SELECTOR_MISMATCH"
        )


if __name__ == "__main__":
    unittest.main()
