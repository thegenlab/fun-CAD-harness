from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

from ..contracts.failure_envelope import validate_failure_envelope

if TYPE_CHECKING:
    from .test_load_profiles import TestLoadProfileBinding


_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class MechanismValidationError(ValueError):
    """A package cannot safely advance through mechanism validation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        failure: Mapping[str, Any] | None = None,
    ):
        self.code = code
        self.message = message
        self.details = dict(details or {})
        self.failure_envelope = (
            None if failure is None else validate_failure_envelope(failure)
        )
        super().__init__(f"{code}: {message}")


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MechanismValidationError("NUMBER_REQUIRED", f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MechanismValidationError("NUMBER_NOT_FINITE", f"{label} must be finite")
    return number


def identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise MechanismValidationError(
            "ID_INVALID",
            f"{label} must match {_ID_RE.pattern}",
        )
    return value


def vector3(
    value: Any,
    label: str,
    *,
    unit: bool = False,
    allow_zero: bool = False,
) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise MechanismValidationError(
            "VECTOR3_REQUIRED",
            f"{label} must contain three numbers",
        )
    result = tuple(finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    length = math.sqrt(sum(item * item for item in result))
    if not allow_zero and length <= 1e-12:
        raise MechanismValidationError("VECTOR3_ZERO", f"{label} cannot be zero")
    if unit and abs(length - 1.0) > 1e-6:
        raise MechanismValidationError(
            "VECTOR3_NOT_UNIT",
            f"{label} must be normalized",
        )
    return result


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MechanismValidationError(
            "SHA256_INVALID",
            f"{label} must be a lowercase SHA-256 digest",
        )
    return value


@dataclass(frozen=True)
class LoadPhase:
    phase_id: str
    wrench_kind: str
    direction_abs: tuple[float, float, float] | None
    magnitude_si: float
    duration_s: float
    stop_condition: str | None = None
    target_progress_m: float | None = None
    target_position_si: float | None = None
    end_magnitude_si: float | None = None
    branch_termination_policy: str | None = None

    def __post_init__(self) -> None:
        identifier(self.phase_id, "phase_id")
        if self.wrench_kind not in {"none", "force", "torque"}:
            raise MechanismValidationError(
                "WRENCH_KIND_INVALID",
                "wrench_kind must be none, force, or torque",
            )
        magnitude = finite(self.magnitude_si, "magnitude_si")
        if finite(self.duration_s, "duration_s") <= 0:
            raise MechanismValidationError(
                "DURATION_INVALID",
                "duration_s must be positive",
            )
        if self.wrench_kind == "none":
            if (
                self.direction_abs is not None
                or abs(magnitude) > 1e-12
                or self.end_magnitude_si is not None
            ):
                raise MechanismValidationError(
                    "ZERO_WRENCH_INVALID",
                    "none phases require no direction, zero magnitude, and no ramp",
                )
        else:
            vector3(self.direction_abs, "direction_abs", unit=True)
            if magnitude <= 0:
                raise MechanismValidationError(
                    "WRENCH_MAGNITUDE_INVALID",
                    "force and torque magnitudes must be positive",
                )
            if self.end_magnitude_si is not None:
                end_magnitude = finite(
                    self.end_magnitude_si,
                    "end_magnitude_si",
                )
                if end_magnitude < 0:
                    raise MechanismValidationError(
                        "WRENCH_MAGNITUDE_INVALID",
                        "ramped force and torque end magnitudes cannot be negative",
                    )
        targets = [
            value
            for value in (self.target_progress_m, self.target_position_si)
            if value is not None
        ]
        if (self.stop_condition is None) != (len(targets) == 0) or len(targets) > 1:
            raise MechanismValidationError(
                "PHASE_STOP_CONDITION_INCOMPLETE",
                "a stop condition requires exactly one target coordinate",
            )
        if self.stop_condition is not None:
            if self.wrench_kind == "none":
                raise MechanismValidationError(
                    "PHASE_STOP_CONDITION_INVALID",
                    "adaptive stop conditions require an active wrench",
                )
            if self.stop_condition not in {
                "progress_at_least",
                "progress_at_most",
                "position_at_least",
                "position_at_most",
            }:
                raise MechanismValidationError(
                    "PHASE_STOP_CONDITION_INVALID",
                    "unsupported adaptive stop condition",
                )
            finite(targets[0], "target_coordinate")
        if self.branch_termination_policy not in {
            None,
            "each_branch_target",
        }:
            raise MechanismValidationError(
                "PHASE_BRANCH_TERMINATION_POLICY_INVALID",
                "branch_termination_policy must be each_branch_target when supplied",
            )
        if (
            self.branch_termination_policy == "each_branch_target"
            and self.stop_condition is None
        ):
            raise MechanismValidationError(
                "PHASE_BRANCH_TERMINATION_POLICY_INVALID",
                "each_branch_target requires a target stop condition",
            )

    @property
    def target_coordinate(self) -> float | None:
        if self.target_position_si is not None:
            return self.target_position_si
        return self.target_progress_m

    def magnitude_at_fraction(self, fraction: float) -> float:
        """Return a zero-end-slope wrench envelope at normalized phase time."""

        if self.end_magnitude_si is None:
            return self.magnitude_si
        clamped = min(1.0, max(0.0, finite(fraction, "phase_fraction")))
        smooth = clamped * clamped * (3.0 - 2.0 * clamped)
        return self.magnitude_si + smooth * (
            self.end_magnitude_si - self.magnitude_si
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.direction_abs is not None:
            payload["direction_abs"] = list(self.direction_abs)
        if self.end_magnitude_si is None:
            payload.pop("end_magnitude_si")
        if self.branch_termination_policy is None:
            payload.pop("branch_termination_policy")
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LoadPhase:
        data = dict(value)
        if data.get("direction_abs") is not None:
            data["direction_abs"] = tuple(data["direction_abs"])
        return cls(**data)


@dataclass(frozen=True)
class LoadCase:
    load_case_id: str
    family: str
    program: str
    purpose: str
    phases: tuple[LoadPhase, ...]
    moving_mass_kg: float
    step_s: float = 0.001
    sample_interval_s: float = 0.05
    required_for_certificate: bool = True
    test_load_case_id: str | None = None
    load_profile_binding: TestLoadProfileBinding | None = None

    def __post_init__(self) -> None:
        identifier(self.load_case_id, "load_case_id")
        identifier(self.family, "family")
        identifier(self.program, "program")
        if self.purpose not in {"nominal", "adversarial", "held_out"}:
            raise MechanismValidationError(
                "LOAD_PURPOSE_INVALID",
                "purpose must be nominal, adversarial, or held_out",
            )
        if not isinstance(self.phases, tuple) or not self.phases:
            raise MechanismValidationError(
                "LOAD_PHASES_REQUIRED",
                "a load case requires at least one phase",
            )
        if len({phase.phase_id for phase in self.phases}) != len(self.phases):
            raise MechanismValidationError(
                "LOAD_PHASE_ID_DUPLICATE",
                "phase IDs must be unique within a load case",
            )
        if finite(self.moving_mass_kg, "moving_mass_kg") <= 0:
            raise MechanismValidationError(
                "MOVING_MASS_INVALID",
                "moving_mass_kg must be positive",
            )
        step = finite(self.step_s, "step_s")
        sample = finite(self.sample_interval_s, "sample_interval_s")
        if not 1e-6 <= step <= 0.01:
            raise MechanismValidationError(
                "STEP_SIZE_INVALID",
                "step_s must be in [1e-6, 0.01]",
            )
        if sample < step:
            raise MechanismValidationError(
                "SAMPLE_INTERVAL_INVALID",
                "sample_interval_s cannot be lower than step_s",
            )
        if not isinstance(self.required_for_certificate, bool):
            raise MechanismValidationError(
                "BOOLEAN_REQUIRED",
                "required_for_certificate must be boolean",
            )
        has_external_selector = self.test_load_case_id is not None
        has_profile_binding = self.load_profile_binding is not None
        if has_external_selector != has_profile_binding:
            raise MechanismValidationError(
                "TEST_LOAD_PROFILE_BINDING_INCOMPLETE",
                "test_load_case_id and load_profile_binding must be supplied together",
            )
        if has_external_selector:
            assert self.test_load_case_id is not None
            identifier(self.test_load_case_id, "test_load_case_id")
            from .test_load_profiles import (
                TestLoadProfileBinding,
                validate_test_load_profile_binding,
            )

            if not isinstance(self.load_profile_binding, TestLoadProfileBinding):
                raise MechanismValidationError(
                    "TEST_LOAD_PROFILE_BINDING_INVALID",
                    "load_profile_binding must be a frozen TestLoadProfileBinding",
                )
            profile = validate_test_load_profile_binding(
                self.load_profile_binding
            )
            if self.test_load_case_id != profile.profile_id:
                raise MechanismValidationError(
                    "TEST_LOAD_PROFILE_SELECTOR_MISMATCH",
                    "test_load_case_id must name the bound test-load profile",
                )

    @property
    def held_out(self) -> bool:
        return self.purpose == "held_out"

    @property
    def duration_s(self) -> float:
        return sum(phase.duration_s for phase in self.phases)

    @property
    def estimated_steps(self) -> int:
        return max(1, int(math.ceil(self.duration_s / self.step_s)))

    @property
    def canonical_test_load_case_id(self) -> str:
        """Return the immutable external selector, or the legacy case ID."""

        return self.test_load_case_id or self.load_case_id

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phases"] = [phase.to_dict() for phase in self.phases]
        if self.test_load_case_id is None:
            payload.pop("test_load_case_id")
            payload.pop("load_profile_binding")
        else:
            assert self.load_profile_binding is not None
            payload["load_profile_binding"] = self.load_profile_binding.to_dict()
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LoadCase:
        data = dict(value)
        data["phases"] = tuple(
            LoadPhase.from_dict(item) for item in data.get("phases", ())
        )
        binding = data.get("load_profile_binding")
        if binding is not None:
            from .test_load_profiles import TestLoadProfileBinding

            data["load_profile_binding"] = TestLoadProfileBinding.from_dict(
                binding
            )
        return cls(**data)


@dataclass(frozen=True)
class MechanismPlan:
    package_dir: str
    intake_report: str
    intake_report_sha256: str
    package_id: str
    design_version: str
    requirements_sha256: str
    package_binding_sha256: str
    capsule_id: str
    capsule_file: str
    family: str
    protocol_id: str
    fixed_group_ids: tuple[str, ...]
    moving_group_ids: tuple[str, ...]
    axis_abs: tuple[float, float, float]
    joint_origin_abs_m: tuple[float, float, float]
    gravity_abs_m_s2: tuple[float, float, float]
    joint_frame_id: str
    input_port_id: str
    input_frame_id: str
    input_point_abs_m: tuple[float, float, float]
    observable_id: str
    observable_frame_id: str
    observable_point_abs_m: tuple[float, float, float]
    task_ids: tuple[str, ...]
    characteristic_length_m: float
    moving_extents_m: tuple[float, float, float]
    adapter_id: str | None
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    source_component_hashes: dict[str, str] = field(default_factory=dict)
    adapter_spec: dict[str, Any] | None = None
    acceptance_contract: dict[str, Any] | None = None
    semantic_binding: dict[str, Any] | None = None
    task_channel_registry: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        for label in (
            "package_id",
            "capsule_id",
            "family",
            "protocol_id",
            "joint_frame_id",
            "input_port_id",
            "input_frame_id",
            "observable_id",
            "observable_frame_id",
        ):
            identifier(getattr(self, label), label)
        _sha256(self.intake_report_sha256, "intake_report_sha256")
        _sha256(self.requirements_sha256, "requirements_sha256")
        _sha256(self.package_binding_sha256, "package_binding_sha256")
        vector3(self.axis_abs, "axis_abs", unit=True)
        vector3(self.joint_origin_abs_m, "joint_origin_abs_m", allow_zero=True)
        vector3(self.gravity_abs_m_s2, "gravity_abs_m_s2", allow_zero=True)
        vector3(self.input_point_abs_m, "input_point_abs_m", allow_zero=True)
        vector3(
            self.observable_point_abs_m,
            "observable_point_abs_m",
            allow_zero=True,
        )
        moving_extents = vector3(
            self.moving_extents_m,
            "moving_extents_m",
        )
        if any(item <= 0 for item in moving_extents):
            raise MechanismValidationError(
                "MOVING_EXTENTS_INVALID",
                "moving_extents_m must be positive on every axis",
            )
        if finite(self.characteristic_length_m, "characteristic_length_m") <= 0:
            raise MechanismValidationError(
                "CHARACTERISTIC_LENGTH_INVALID",
                "characteristic_length_m must be positive",
            )
        if not self.fixed_group_ids or not self.moving_group_ids:
            raise MechanismValidationError(
                "RIGID_GROUPS_REQUIRED",
                "fixed and moving groups must be non-empty",
            )
        if set(self.fixed_group_ids) & set(self.moving_group_ids):
            raise MechanismValidationError(
                "RIGID_GROUP_OVERLAP",
                "fixed and moving groups must not overlap",
            )
        for label, values in (
            ("fixed_group_ids", self.fixed_group_ids),
            ("moving_group_ids", self.moving_group_ids),
            ("task_ids", self.task_ids),
        ):
            if not isinstance(values, tuple):
                raise MechanismValidationError(
                    "TUPLE_REQUIRED",
                    f"{label} must be a tuple",
                )
            for index, item in enumerate(values):
                identifier(item, f"{label}[{index}]")
        if self.adapter_id is not None:
            identifier(self.adapter_id, "adapter_id")
        for label, inventory in (
            ("artifact_hashes", self.artifact_hashes),
            ("source_component_hashes", self.source_component_hashes),
        ):
            if not isinstance(inventory, dict):
                raise MechanismValidationError(
                    "HASH_INVENTORY_INVALID",
                    f"{label} must be an object",
                )
            for path, digest in inventory.items():
                if not isinstance(path, str) or not path:
                    raise MechanismValidationError(
                        "HASH_PATH_INVALID",
                        f"{label} contains an invalid path",
                    )
                _sha256(digest, f"{label}.{path}")
        if self.adapter_spec is not None:
            if not isinstance(self.adapter_spec, dict):
                raise MechanismValidationError(
                    "ADAPTER_SPEC_INVALID",
                    "adapter_spec must be an object when provided",
                )
            if (
                self.adapter_spec.get("adapter_id") != self.adapter_id
                or self.adapter_spec.get("family") != self.family
            ):
                raise MechanismValidationError(
                    "ADAPTER_SPEC_MISMATCH",
                    "adapter_spec identity must match the mechanism plan",
                )

        for label, evidence in (
            ("acceptance_contract", self.acceptance_contract),
            ("semantic_binding", self.semantic_binding),
            ("task_channel_registry", self.task_channel_registry),
        ):
            if evidence is not None and not isinstance(evidence, dict):
                raise MechanismValidationError(
                    "TASK_EVIDENCE_INVALID",
                    f"{label} must be an object when provided",
                )
        if self.acceptance_contract is not None:
            digest = self.acceptance_contract.get("task_contract_sha256")
            if digest is not None:
                _sha256(digest, "acceptance_contract.task_contract_sha256")
        if self.semantic_binding is not None:
            digest = self.semantic_binding.get("task_binding_sha256")
            if digest is not None:
                _sha256(digest, "semantic_binding.task_binding_sha256")

    @property
    def dispatchable(self) -> bool:
        return self.adapter_id is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if payload["adapter_spec"] is None:
            payload.pop("adapter_spec")
        if payload["task_channel_registry"] is None:
            payload.pop("task_channel_registry")
        for name in (
            "fixed_group_ids",
            "moving_group_ids",
            "axis_abs",
            "joint_origin_abs_m",
            "gravity_abs_m_s2",
            "input_point_abs_m",
            "observable_point_abs_m",
            "task_ids",
            "moving_extents_m",
        ):
            payload[name] = list(payload[name])
        return payload

    def execution_identity(self) -> dict[str, Any]:
        """Return the compiled fields that can change mechanism execution."""

        payload = self.to_dict()
        for name in (
            "package_dir",
            "intake_report",
            "intake_report_sha256",
            "package_id",
            "design_version",
            "package_binding_sha256",
            "capsule_file",
            "artifact_hashes",
            "source_component_hashes",
        ):
            payload.pop(name)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MechanismPlan:
        data = dict(value)
        for name in (
            "fixed_group_ids",
            "moving_group_ids",
            "axis_abs",
            "joint_origin_abs_m",
            "gravity_abs_m_s2",
            "input_point_abs_m",
            "observable_point_abs_m",
            "task_ids",
            "moving_extents_m",
        ):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)
