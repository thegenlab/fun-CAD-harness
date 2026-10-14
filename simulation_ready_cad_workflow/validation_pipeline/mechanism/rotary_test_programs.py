from __future__ import annotations

"""Trusted rotary input programs and exact sub-program views.

Programs describe validator-owned motion schedules.  A mechanism plugin may
provide bounded actuator magnitudes and an axis, but it cannot change the
registered cycle count, phase order, target positions, or view windows.
"""

from dataclasses import asdict, dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

from .io import canonical_sha256
from .models import LoadPhase, MechanismValidationError


ROTARY_TEST_PROGRAM_SCHEMA = "validator_rotary_test_program_v1"
ROTARY_TEST_PROGRAM_VIEW_SCHEMA = "validator_rotary_program_view_v1"
ROTARY_TEST_PROGRAM_REGISTRY_SCHEMA = "validator_rotary_program_registry_v1"
ROTARY_TEST_PROGRAM_REGISTRY_VERSION = "rotary_test_program_registry_v1"


@dataclass(frozen=True)
class RotaryTestProgram:
    program_id: str
    cycle_count: int
    drive_target_angle_rad: float
    home_target_angle_rad: float
    phase_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.program_id != "task_output_angle_multi_cycle":
            raise MechanismValidationError(
                "ROTARY_TEST_PROGRAM_ID_INVALID",
                "only the installed multi-cycle rotary program is supported",
            )
        if self.cycle_count != 20:
            raise MechanismValidationError(
                "ROTARY_TEST_PROGRAM_CYCLE_COUNT_INVALID",
                "task_output_angle_multi_cycle requires exactly 20 cycles",
            )
        if (
            isinstance(self.drive_target_angle_rad, bool)
            or not isinstance(self.drive_target_angle_rad, (int, float))
            or not math.isfinite(float(self.drive_target_angle_rad))
            or float(self.drive_target_angle_rad) <= 0.0
        ):
            raise MechanismValidationError(
                "ROTARY_TEST_PROGRAM_TARGET_INVALID",
                "drive target angle must be finite and positive",
            )
        if (
            isinstance(self.home_target_angle_rad, bool)
            or not isinstance(self.home_target_angle_rad, (int, float))
            or not math.isfinite(float(self.home_target_angle_rad))
        ):
            raise MechanismValidationError(
                "ROTARY_TEST_PROGRAM_TARGET_INVALID",
                "home target angle must be finite",
            )
        if self.phase_roles != (
            "drive",
            "drive_settle",
            "return",
            "return_settle",
        ):
            raise MechanismValidationError(
                "ROTARY_TEST_PROGRAM_PHASE_ROLES_INVALID",
                "multi-cycle rotary phase roles are fixed",
            )

    @property
    def phase_ids(self) -> tuple[str, ...]:
        return tuple(
            f"cycle_{cycle:02d}_{role}"
            for cycle in range(1, self.cycle_count + 1)
            for role in self.phase_roles
        )

    @property
    def powered_phase_ids(self) -> tuple[str, ...]:
        return tuple(
            f"cycle_{cycle:02d}_drive"
            for cycle in range(1, self.cycle_count + 1)
        )

    @property
    def return_phase_ids(self) -> tuple[str, ...]:
        return tuple(
            phase_id
            for phase_id in self.phase_ids
            if phase_id not in set(self.powered_phase_ids)
        )

    @property
    def adaptive_phase_ids(self) -> tuple[str, ...]:
        return tuple(
            phase_id
            for phase_id in self.phase_ids
            if phase_id.endswith("_drive") or phase_id.endswith("_return")
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = ROTARY_TEST_PROGRAM_SCHEMA
        payload["phase_roles"] = list(self.phase_roles)
        payload["phase_ids"] = list(self.phase_ids)
        payload["powered_phase_ids"] = list(self.powered_phase_ids)
        payload["return_phase_ids"] = list(self.return_phase_ids)
        payload["adaptive_phase_ids"] = list(self.adaptive_phase_ids)
        return payload


@dataclass(frozen=True)
class RotaryProgramView:
    source_program_id: str
    view_program_id: str
    phase_ids: tuple[str, ...]
    powered_phase_ids: tuple[str, ...]
    return_phase_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.source_program_id != "task_output_angle_multi_cycle"
            or self.view_program_id != "task_output_angle_roundtrip"
        ):
            raise MechanismValidationError(
                "ROTARY_PROGRAM_VIEW_ID_INVALID",
                "the installed view is the first roundtrip of multi-cycle",
            )
        expected = (
            "cycle_01_drive",
            "cycle_01_drive_settle",
            "cycle_01_return",
            "cycle_01_return_settle",
        )
        if self.phase_ids != expected:
            raise MechanismValidationError(
                "ROTARY_PROGRAM_VIEW_PHASES_INVALID",
                "the roundtrip view must contain the complete first cycle",
            )
        if self.powered_phase_ids != ("cycle_01_drive",) or (
            self.return_phase_ids != expected[1:]
        ):
            raise MechanismValidationError(
                "ROTARY_PROGRAM_VIEW_WINDOWS_INVALID",
                "the roundtrip view has fixed powered and return windows",
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = ROTARY_TEST_PROGRAM_VIEW_SCHEMA
        for key in ("phase_ids", "powered_phase_ids", "return_phase_ids"):
            payload[key] = list(payload[key])
        return payload


_MULTI_CYCLE = RotaryTestProgram(
    program_id="task_output_angle_multi_cycle",
    cycle_count=20,
    drive_target_angle_rad=0.4,
    home_target_angle_rad=0.0,
    phase_roles=("drive", "drive_settle", "return", "return_settle"),
)
_FIRST_CYCLE_VIEW = RotaryProgramView(
    source_program_id=_MULTI_CYCLE.program_id,
    view_program_id="task_output_angle_roundtrip",
    phase_ids=_MULTI_CYCLE.phase_ids[:4],
    powered_phase_ids=("cycle_01_drive",),
    return_phase_ids=_MULTI_CYCLE.phase_ids[1:4],
)
_PROGRAMS: Mapping[str, RotaryTestProgram] = MappingProxyType(
    {_MULTI_CYCLE.program_id: _MULTI_CYCLE}
)
_VIEWS: Mapping[tuple[str, str], RotaryProgramView] = MappingProxyType(
    {
        (
            _FIRST_CYCLE_VIEW.source_program_id,
            _FIRST_CYCLE_VIEW.view_program_id,
        ): _FIRST_CYCLE_VIEW
    }
)


def _registry_core() -> dict[str, Any]:
    return {
        "schema_version": ROTARY_TEST_PROGRAM_REGISTRY_SCHEMA,
        "registry_version": ROTARY_TEST_PROGRAM_REGISTRY_VERSION,
        "programs": [
            _PROGRAMS[key].to_dict() for key in sorted(_PROGRAMS)
        ],
        "views": [
            _VIEWS[key].to_dict() for key in sorted(_VIEWS)
        ],
    }


# Registration edits must update both the versioned inventory and this pin.
ROTARY_TEST_PROGRAM_REGISTRY_SHA256 = (
    "a8ed7d7976570629bf0ff354d29acbe39918a3c01a81e88024059504b25d4ca4"
)


def _validate_registry() -> None:
    if canonical_sha256(_registry_core()) != ROTARY_TEST_PROGRAM_REGISTRY_SHA256:
        raise MechanismValidationError(
            "ROTARY_TEST_PROGRAM_REGISTRY_TAMPERED",
            "installed rotary test-program registry does not match its pin",
        )


def rotary_test_program_registry_document() -> dict[str, Any]:
    _validate_registry()
    return {
        **_registry_core(),
        "registry_sha256": ROTARY_TEST_PROGRAM_REGISTRY_SHA256,
    }


def rotary_test_program(program_id: str) -> RotaryTestProgram:
    _validate_registry()
    program = _PROGRAMS.get(program_id)
    if program is None:
        raise MechanismValidationError(
            "ROTARY_TEST_PROGRAM_UNREGISTERED",
            f"rotary test program {program_id!r} is not registered",
        )
    return program


def rotary_program_view(
    source_program_id: str,
    view_program_id: str,
) -> RotaryProgramView:
    _validate_registry()
    view = _VIEWS.get((source_program_id, view_program_id))
    if view is None:
        raise MechanismValidationError(
            "ROTARY_PROGRAM_VIEW_UNREGISTERED",
            "the requested rotary program view is not registered",
        )
    return view


def materialize_rotary_test_program(
    program_id: str,
    *,
    axis_abs: tuple[float, float, float],
    drive_torque_nm: float,
    return_torque_nm: float,
    max_torque_phase_s: float,
    settle_phase_s: float,
) -> tuple[LoadPhase, ...]:
    """Materialize the exact schedule with plugin-bounded wrench magnitudes."""

    program = rotary_test_program(program_id)
    if (
        len(axis_abs) != 3
        or any(not math.isfinite(float(item)) for item in axis_abs)
        or abs(math.sqrt(sum(float(item) ** 2 for item in axis_abs)) - 1.0)
        > 1.0e-6
    ):
        raise MechanismValidationError(
            "ROTARY_TEST_PROGRAM_AXIS_INVALID",
            "rotary test-program axis must be a finite unit vector",
        )
    numeric = {
        "drive_torque_nm": drive_torque_nm,
        "return_torque_nm": return_torque_nm,
        "max_torque_phase_s": max_torque_phase_s,
        "settle_phase_s": settle_phase_s,
    }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in numeric.values()
    ):
        raise MechanismValidationError(
            "ROTARY_TEST_PROGRAM_PARAMETER_INVALID",
            "actuator magnitudes and phase durations must be finite and positive",
        )
    negative_axis = tuple(-float(item) for item in axis_abs)
    phases: list[LoadPhase] = []
    for cycle in range(1, program.cycle_count + 1):
        stem = f"cycle_{cycle:02d}"
        phases.extend(
            (
                LoadPhase(
                    f"{stem}_drive",
                    "torque",
                    axis_abs,
                    float(drive_torque_nm),
                    float(max_torque_phase_s),
                    stop_condition="position_at_least",
                    target_position_si=float(program.drive_target_angle_rad),
                ),
                LoadPhase(
                    f"{stem}_drive_settle",
                    "none",
                    None,
                    0.0,
                    float(settle_phase_s),
                ),
                LoadPhase(
                    f"{stem}_return",
                    "torque",
                    negative_axis,
                    float(return_torque_nm),
                    float(max_torque_phase_s),
                    stop_condition="position_at_most",
                    target_position_si=float(program.home_target_angle_rad),
                ),
                LoadPhase(
                    f"{stem}_return_settle",
                    "none",
                    None,
                    0.0,
                    float(settle_phase_s),
                ),
            )
        )
    if tuple(phase.phase_id for phase in phases) != program.phase_ids:
        raise AssertionError("materialized rotary test program changed phase order")
    return tuple(phases)
