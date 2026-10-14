from __future__ import annotations

"""Validator-owned passive loads used by output-focused task tests.

The requirements document may select a registered profile, but it cannot
author a wrench law.  Every profile in this module is immutable, versioned,
and covered by both a per-profile digest and an exact registry digest.
"""

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from .io import canonical_sha256


TEST_LOAD_PROFILE_SCHEMA = "validator_test_load_profile_v1"
TEST_LOAD_PROFILE_REGISTRY_SCHEMA = "validator_test_load_profile_registry_v1"
TEST_LOAD_PROFILE_REGISTRY_VERSION = "test_load_profile_registry_v1"


def _validation_error(code: str, message: str) -> ValueError:
    # Imported lazily so ``models.LoadCase`` can refer to the frozen binding
    # type without creating an import cycle while the modules initialize.
    from .models import MechanismValidationError

    return MechanismValidationError(code, message)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _validation_error(
            "TEST_LOAD_PROFILE_NUMBER_REQUIRED",
            f"{label} must be numeric",
        )
    number = float(value)
    if not math.isfinite(number):
        raise _validation_error(
            "TEST_LOAD_PROFILE_NUMBER_INVALID",
            f"{label} must be finite",
        )
    return number


@dataclass(frozen=True)
class TestLoadProfile:
    """One exact validator-authored passive test-load law."""

    profile_id: str
    profile_kind: str
    wrench_kind: str
    wrench_law_id: str
    target_role: str
    magnitude_si: float
    unit: str
    direction_policy: str
    rest_policy: str

    def __post_init__(self) -> None:
        if not self.profile_id:
            raise _validation_error(
                "TEST_LOAD_PROFILE_ID_INVALID",
                "profile_id must be a non-empty identifier",
            )
        if self.profile_kind != "passive_test_load":
            raise _validation_error(
                "TEST_LOAD_PROFILE_KIND_INVALID",
                "profile_kind must be passive_test_load",
            )
        if self.wrench_kind != "torque" or self.unit != "N*m":
            raise _validation_error(
                "TEST_LOAD_PROFILE_WRENCH_INVALID",
                "only torque profiles measured in N*m are supported",
            )
        if self.wrench_law_id != "validator_dissipative_test_load_v1":
            raise _validation_error(
                "TEST_LOAD_PROFILE_LAW_INVALID",
                "the profile must use validator_dissipative_test_load_v1",
            )
        if self.target_role != "task_output":
            raise _validation_error(
                "TEST_LOAD_PROFILE_TARGET_INVALID",
                "passive output loads must target task_output",
            )
        if _finite_number(self.magnitude_si, "magnitude_si") <= 0.0:
            raise _validation_error(
                "TEST_LOAD_PROFILE_MAGNITUDE_INVALID",
                "magnitude_si must be positive",
            )
        if self.direction_policy != "opposes_target_angular_velocity":
            raise _validation_error(
                "TEST_LOAD_PROFILE_DIRECTION_INVALID",
                "passive torque must oppose target angular velocity",
            )
        if self.rest_policy != "zero_at_rest":
            raise _validation_error(
                "TEST_LOAD_PROFILE_REST_POLICY_INVALID",
                "passive torque must be zero at rest",
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TEST_LOAD_PROFILE_SCHEMA
        return payload

    @property
    def profile_sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class TestLoadProfileBinding:
    """Immutable proof that a load case selected an installed profile."""

    profile_id: str
    registry_version: str
    registry_sha256: str
    profile_sha256: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TestLoadProfileBinding:
        if not isinstance(value, Mapping) or set(value) != {
            "profile_id",
            "registry_version",
            "registry_sha256",
            "profile_sha256",
        }:
            raise _validation_error(
                "TEST_LOAD_PROFILE_BINDING_INVALID",
                "load profile binding fields are incomplete or unexpected",
            )
        if not all(isinstance(item, str) for item in value.values()):
            raise _validation_error(
                "TEST_LOAD_PROFILE_BINDING_INVALID",
                "load profile binding values must be strings",
            )
        return cls(
            profile_id=value["profile_id"],
            registry_version=value["registry_version"],
            registry_sha256=value["registry_sha256"],
            profile_sha256=value["profile_sha256"],
        )


_REGISTERED_TEST_LOAD_PROFILES: tuple[TestLoadProfile, ...] = (
    TestLoadProfile(
        profile_id="rated_resistive_torque_3nm",
        profile_kind="passive_test_load",
        wrench_kind="torque",
        wrench_law_id="validator_dissipative_test_load_v1",
        target_role="task_output",
        magnitude_si=3.0,
        unit="N*m",
        direction_policy="opposes_target_angular_velocity",
        rest_policy="zero_at_rest",
    ),
)

_TEST_LOAD_PROFILE_BY_ID: Mapping[str, TestLoadProfile] = MappingProxyType(
    {profile.profile_id: profile for profile in _REGISTERED_TEST_LOAD_PROFILES}
)

# This digest deliberately is not derived at import time.  An edit to any
# trusted registration must make the registry fail closed until its reviewed
# version and pinned digest are updated together.
TEST_LOAD_PROFILE_REGISTRY_SHA256 = (
    "a0e535d69af80dd6044de75c3bedb0e8571e554adcf28aee472940c6b9a02cea"
)


def _registry_core() -> dict[str, Any]:
    return {
        "schema_version": TEST_LOAD_PROFILE_REGISTRY_SCHEMA,
        "registry_version": TEST_LOAD_PROFILE_REGISTRY_VERSION,
        "profiles": [
            profile.to_dict()
            for profile in sorted(
                _REGISTERED_TEST_LOAD_PROFILES,
                key=lambda item: item.profile_id,
            )
        ],
    }


def _validate_registry() -> None:
    ids = [profile.profile_id for profile in _REGISTERED_TEST_LOAD_PROFILES]
    if not ids or len(ids) != len(set(ids)):
        raise _validation_error(
            "TEST_LOAD_PROFILE_REGISTRY_INVALID",
            "registered test-load profile IDs must be non-empty and unique",
        )
    expected_index = {
        profile.profile_id: profile
        for profile in _REGISTERED_TEST_LOAD_PROFILES
    }
    if dict(_TEST_LOAD_PROFILE_BY_ID) != expected_index:
        raise _validation_error(
            "TEST_LOAD_PROFILE_REGISTRY_INVALID",
            "test-load profile index does not match the registered inventory",
        )
    actual = canonical_sha256(_registry_core())
    if actual != TEST_LOAD_PROFILE_REGISTRY_SHA256:
        raise _validation_error(
            "TEST_LOAD_PROFILE_REGISTRY_TAMPERED",
            "installed test-load profile registry does not match its pinned hash",
        )


def test_load_profile_registry_document() -> dict[str, Any]:
    _validate_registry()
    return {
        **_registry_core(),
        "registry_sha256": TEST_LOAD_PROFILE_REGISTRY_SHA256,
    }


def registered_test_load_profiles() -> tuple[TestLoadProfile, ...]:
    _validate_registry()
    return _REGISTERED_TEST_LOAD_PROFILES


def test_load_profile_for(profile_id: str) -> TestLoadProfile:
    _validate_registry()
    profile = _TEST_LOAD_PROFILE_BY_ID.get(profile_id)
    if profile is None:
        raise _validation_error(
            "TEST_LOAD_PROFILE_NOT_REGISTERED",
            f"test-load profile {profile_id!r} is not registered",
        )
    return profile


def binding_for_test_load_profile(profile_id: str) -> TestLoadProfileBinding:
    profile = test_load_profile_for(profile_id)
    return TestLoadProfileBinding(
        profile_id=profile.profile_id,
        registry_version=TEST_LOAD_PROFILE_REGISTRY_VERSION,
        registry_sha256=TEST_LOAD_PROFILE_REGISTRY_SHA256,
        profile_sha256=profile.profile_sha256,
    )


def validate_test_load_profile_binding(
    binding: TestLoadProfileBinding,
) -> TestLoadProfile:
    if not isinstance(binding, TestLoadProfileBinding):
        raise _validation_error(
            "TEST_LOAD_PROFILE_BINDING_INVALID",
            "load_profile_binding must be a frozen TestLoadProfileBinding",
        )
    profile = test_load_profile_for(binding.profile_id)
    expected = binding_for_test_load_profile(profile.profile_id)
    if binding != expected:
        raise _validation_error(
            "TEST_LOAD_PROFILE_BINDING_MISMATCH",
            "load profile binding does not match the installed registry",
        )
    return profile


def dissipative_test_load_torque_nm(
    profile_or_binding: TestLoadProfile | TestLoadProfileBinding,
    target_angular_velocity_rad_s: float,
) -> float:
    """Return the signed passive torque about the target's positive axis."""

    if isinstance(profile_or_binding, TestLoadProfileBinding):
        profile = validate_test_load_profile_binding(profile_or_binding)
    elif isinstance(profile_or_binding, TestLoadProfile):
        profile = test_load_profile_for(profile_or_binding.profile_id)
        if profile_or_binding != profile:
            raise _validation_error(
                "TEST_LOAD_PROFILE_TAMPERED",
                "test-load profile does not match the installed registration",
            )
    else:
        raise _validation_error(
            "TEST_LOAD_PROFILE_INVALID",
            "a registered test-load profile or binding is required",
        )
    angular_velocity = _finite_number(
        target_angular_velocity_rad_s,
        "target_angular_velocity_rad_s",
    )
    if angular_velocity == 0.0:
        return 0.0
    return -math.copysign(profile.magnitude_si, angular_velocity)
