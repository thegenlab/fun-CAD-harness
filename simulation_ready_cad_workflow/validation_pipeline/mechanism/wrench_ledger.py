from __future__ import annotations

import contextlib
import contextvars
import math
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
    mechanism_owner_for_registration_source,
)
from .io import canonical_sha256
from .models import MechanismValidationError


WRENCH_LEDGER_SCHEMA = "simulation_ready_cad_wrench_ledger_v2"
WRENCH_AUDIT_SCHEMA = "simulation_ready_cad_wrench_audit_v2"
WRENCH_CHECKPOINT_SCHEMA = "simulation_ready_cad_wrench_checkpoint_v1"
_POLICY_KEYS = frozenset(
    {
        "policy_id",
        "wrench_kind",
        "category",
        "law_id",
        "target_roles",
        "task_input",
        "transmission",
        "support",
        "capability_required",
        "component_provenance_required",
    }
)
_RECORD_KEYS = frozenset(
    {
        "load_id",
        "owner",
        "target_body",
        "category",
        "law_id",
        "source_role",
        "target_role",
        "geometry_gate",
        "active_interval",
        "policy_id",
        "task_input",
        "transmission",
        "support",
        "release_required",
        "wrench_kind",
        "policy",
        "initial_vector_si",
        "last_vector_si",
        "update_count",
        "nonzero_update_count",
        "zero_update_count",
        "released",
        "component_event_count",
        "component_sources",
        "phase_update_counts",
        "phase_nonzero_update_counts",
        "phase_zero_update_counts",
    }
)
_TASK_INPUT_STATE_KEYS = frozenset(
    {"load_id", "target_body", "policy_id", "vector_si"}
)
_PHASE_BARRIER_KEYS = frozenset(
    {
        "phase_id",
        "wrench_kind",
        "entry_task_input_state",
        "exit_task_input_state",
        "entry_all_zero",
        "exit_all_zero",
        "task_input_update_count",
        "task_input_nonzero_update_count",
        "task_input_zero_update_count",
        "aggregated_task_input_event_count",
        "barrier_sha256",
    }
)
_TASK_INPUT_EVENT_KEYS = frozenset(
    {
        "target_body",
        "source_role",
        "active_interval",
        "phase_id",
        "law_id",
    }
)
_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "ledger_schema_version",
        "plugin_id",
        "capability_registration",
        "branch",
        "load_case_id",
        "program_phase_ids",
        "program_phases",
        "phase_context_ids",
        "phase_barriers",
        "executed_phase_ids",
        "executed_load_sha256",
        "declared_input_body_ids",
        "declared_output_body_ids",
        "load_count",
        "task_input_load_count",
        "transmission_load_count",
        "support_load_count",
        "task_input_bodies_observed",
        "task_input_sources_observed",
        "task_input_nonzero_phase_ids",
        "task_input_zero_phase_ids",
        "aggregated_task_input_events",
        "direct_output_drive_detected",
        "release_residual_count",
        "violations",
        "passed",
        "records",
        "ledger_sha256",
    }
)


@dataclass(frozen=True)
class WrenchPolicy:
    """One plugin-owned, exact permission to create an external body wrench."""

    policy_id: str
    wrench_kind: str
    category: str
    law_id: str
    target_roles: tuple[str, ...]
    task_input: bool = False
    transmission: bool = False
    support: bool = False
    capability_required: bool = False
    component_provenance_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_roles"] = list(self.target_roles)
        return value


@dataclass(frozen=True)
class WrenchDeclaration:
    """Immutable provenance attached before a native load can be constructed."""

    load_id: str
    owner: str
    target_body: str
    category: str
    law_id: str
    source_role: str
    target_role: str
    geometry_gate: str
    active_interval: str
    policy_id: str
    task_input: bool = False
    transmission: bool = False
    support: bool = False
    release_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_STANDARD_DECLARATION_PROFILES: dict[
    str, tuple[str, str, bool, bool, bool]
] = {
    "task_input": (
        "task_input",
        "validator_load_phase_v1",
        True,
        False,
        False,
    ),
    "task_input_controller": (
        "task_input",
        "validator_bounded_controller_phase_v1",
        True,
        False,
        False,
    ),
    "task_input_controller_150": (
        "task_input",
        "validator_bounded_controller_phase_150_v1",
        True,
        False,
        False,
    ),
    "task_input_reaction": (
        "task_input",
        "validator_opposed_load_phase_v1",
        True,
        False,
        False,
    ),
    "task_input_dynamic_axis": (
        "task_input",
        "validator_dynamic_actuator_phase_v1",
        True,
        False,
        False,
    ),
    "task_input_dynamic_axis_reaction": (
        "task_input",
        "validator_dynamic_actuator_reaction_phase_v1",
        True,
        False,
        False,
    ),
    "task_input_testcase": (
        "task_input",
        "validator_testcase_multichannel_v1",
        True,
        False,
        False,
    ),
    "support": (
        "physical_support",
        "finite_compliant_support_v1",
        False,
        False,
        True,
    ),
    "constitutive": (
        "constitutive_support",
        "finite_constitutive_response_v1",
        False,
        False,
        True,
    ),
    "limit": (
        "finite_limit",
        "geometry_gated_limit_v1",
        False,
        False,
        True,
    ),
    "transmission": (
        "analytical_transmission",
        "geometry_gated_transmission_v1",
        False,
        True,
        False,
    ),
    "bicycle_chain_transmission": (
        "transmission",
        "finite_geometry_gated_chain_v2",
        False,
        True,
        False,
    ),
    "bicycle_finite_support": (
        "finite_support",
        "finite_6d_linear_support_v2",
        False,
        False,
        True,
    ),
    "passive_storage": (
        "passive_energy_storage",
        "finite_passive_centering_v1",
        False,
        False,
        True,
    ),
    "passive_dissipation": (
        "passive_dissipation",
        "bounded_dissipative_drag_v1",
        False,
        False,
        False,
    ),
    "passive_test_load": (
        "passive_test_load",
        "validator_dissipative_test_load_v1",
        False,
        False,
        False,
    ),
    "resistance": (
        "validator_resistance",
        "validator_resistance_v1",
        False,
        False,
        False,
    ),
    "probe": (
        "validator_probe",
        "validator_probe_v1",
        False,
        False,
        False,
    ),
    "accumulator": (
        "audited_wrench_accumulator",
        "audited_component_sum_v1",
        False,
        False,
        False,
    ),
}


def standard_wrench_declaration(
    *,
    plugin_id: str,
    profile: str,
    wrench_kind: str,
    load_id: str,
    owner: str,
    target_body: str,
    source_role: str,
    target_role: str,
    geometry_gate: str,
    active_interval: str,
    release_required: bool = False,
) -> WrenchDeclaration:
    try:
        category, law_id, task_input, transmission, support = (
            _STANDARD_DECLARATION_PROFILES[profile]
        )
    except KeyError as exc:
        raise MechanismValidationError(
            "WRENCH_PROFILE_UNKNOWN",
            f"unknown standard wrench profile {profile!r}",
        ) from exc
    return WrenchDeclaration(
        load_id=load_id,
        owner=owner,
        target_body=target_body,
        category=category,
        law_id=law_id,
        source_role=source_role,
        target_role=target_role,
        geometry_gate=geometry_gate,
        active_interval=active_interval,
        policy_id=f"{plugin_id}.{profile}.{wrench_kind}",
        task_input=task_input,
        transmission=transmission,
        support=support,
        release_required=release_required,
    )


def _finite_vector(value: Any) -> tuple[float, float, float]:
    try:
        result = (float(value.x), float(value.y), float(value.z))
    except (AttributeError, TypeError, ValueError):
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise MechanismValidationError(
                "WRENCH_VECTOR_INVALID",
                "a wrench update must contain three finite components",
            )
        try:
            result = tuple(float(item) for item in value)
        except (TypeError, ValueError) as exc:
            raise MechanismValidationError(
                "WRENCH_VECTOR_INVALID",
                "a wrench update must contain three finite components",
            ) from exc
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise MechanismValidationError(
            "WRENCH_VECTOR_INVALID",
            "a wrench update must contain three finite components",
        )
    return result


def _is_zero(vector: Sequence[float]) -> bool:
    # A phase declared with ``wrench_kind == "none"`` is a protocol barrier,
    # not a numerical convergence test.  Only an explicitly released vector
    # is zero; a tiny nonzero value must not be relabelled as no input.
    return all(float(item) == 0.0 for item in vector)


_EXACT_TASK_INPUT_LAW = "validator_load_phase_v1"
_BOUNDED_TASK_INPUT_LAWS = {
    "validator_bounded_controller_phase_v1": 1.0,
    "validator_bounded_controller_phase_150_v1": 1.5,
}
_OPPOSED_TASK_INPUT_LAW = "validator_opposed_load_phase_v1"
_DYNAMIC_AXIS_TASK_INPUT_LAWS = frozenset(
    {
        "validator_dynamic_actuator_phase_v1",
        "validator_dynamic_actuator_reaction_phase_v1",
    }
)
_TESTCASE_TASK_INPUT_LAW = "validator_testcase_multichannel_v1"
_REGISTERED_TASK_INPUT_LAWS = frozenset(
    {
        _EXACT_TASK_INPUT_LAW,
        *_BOUNDED_TASK_INPUT_LAWS,
        _OPPOSED_TASK_INPUT_LAW,
        *_DYNAMIC_AXIS_TASK_INPUT_LAWS,
        _TESTCASE_TASK_INPUT_LAW,
    }
)
_AGGREGATED_TASK_INPUT_LAWS = frozenset(
    {_EXACT_TASK_INPUT_LAW, *_BOUNDED_TASK_INPUT_LAWS}
)


def _task_input_law_error(message: str) -> MechanismValidationError:
    return MechanismValidationError("WRENCH_TASK_INPUT_LAW_MISMATCH", message)


class LedgeredBodyWrench:
    """Small proxy that records every mutation and exposes the native load explicitly."""

    __slots__ = ("native_load", "_ledger", "_load_id", "_wrench_kind")

    def __init__(
        self,
        native_load: Any,
        ledger: WrenchLedger,
        load_id: str,
        wrench_kind: str,
    ) -> None:
        self.native_load = native_load
        self._ledger = ledger
        self._load_id = load_id
        self._wrench_kind = wrench_kind

    def SetForce(self, value: Any, local: bool) -> Any:
        if self._wrench_kind != "force":
            raise MechanismValidationError(
                "WRENCH_KIND_MISMATCH",
                f"{self._load_id!r} is not a force load",
            )
        vector = _finite_vector(value)
        self._ledger.record_update(self._load_id, vector)
        result = self.native_load.SetForce(value, local)
        return result

    def SetTorque(self, value: Any, local: bool) -> Any:
        if self._wrench_kind != "torque":
            raise MechanismValidationError(
                "WRENCH_KIND_MISMATCH",
                f"{self._load_id!r} is not a torque load",
            )
        vector = _finite_vector(value)
        self._ledger.record_update(self._load_id, vector)
        result = self.native_load.SetTorque(value, local)
        return result

    def SetApplicationPoint(self, value: Any, local: bool) -> Any:
        return self.native_load.SetApplicationPoint(value, local)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.native_load, name)


class WrenchLedger:
    """Validator-owned registry and semantic audit for all non-engine wrenches."""

    def __init__(
        self,
        *,
        plugin_id: str,
        capability_registration: Mapping[str, Any],
        policies: Iterable[WrenchPolicy],
        declared_input_body_ids: Iterable[str],
        declared_output_body_ids: Iterable[str],
        branch: str,
        load_case_id: str,
        program_phases: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        self.plugin_id = str(plugin_id)
        self.capability_registration = dict(capability_registration)
        self.policies = {item.policy_id: item for item in policies}
        self.declared_input_body_ids = frozenset(
            str(item) for item in declared_input_body_ids if str(item)
        )
        self.declared_output_body_ids = frozenset(
            str(item) for item in declared_output_body_ids if str(item)
        )
        self.branch = str(branch)
        self.load_case_id = str(load_case_id)
        self.program_phases = tuple(dict(item) for item in program_phases)
        self.program_phase_ids = tuple(
            str(item.get("phase_id") or "") for item in self.program_phases
        )
        if (
            any(not item for item in self.program_phase_ids)
            or len(set(self.program_phase_ids)) != len(self.program_phase_ids)
        ):
            raise MechanismValidationError(
                "WRENCH_PROGRAM_PHASES_INVALID",
                "wrench ledger program phase IDs must be non-empty and unique",
            )
        self._executed_phases: list[dict[str, Any]] | None = None
        self._executed_load_sha256: str | None = None
        self._phase_context_ids: list[str] = []
        self._records: dict[str, dict[str, Any]] = {}
        self._task_input_events: list[dict[str, str]] = []
        self._phase_barriers: list[dict[str, Any]] = []
        self._open_phase: dict[str, Any] | None = None

    def checkpoint_state(self) -> dict[str, Any]:
        """Return the complete mutable ledger prefix needed by a lane resume."""

        return {
            "schema_version": WRENCH_CHECKPOINT_SCHEMA,
            "plugin_id": self.plugin_id,
            "branch": self.branch,
            "load_case_id": self.load_case_id,
            "program_phase_ids": list(self.program_phase_ids),
            "phase_context_ids": list(self._phase_context_ids),
            "records": deepcopy(self._records),
            "task_input_events": deepcopy(self._task_input_events),
            "phase_barriers": deepcopy(self._phase_barriers),
            "open_phase": deepcopy(self._open_phase),
            "executed_phases": deepcopy(self._executed_phases),
            "executed_load_sha256": self._executed_load_sha256,
        }

    def restore_checkpoint_state(self, value: Mapping[str, Any]) -> str | None:
        """Restore a checkpoint after this execution recreated the same loads."""

        if value.get("schema_version") != WRENCH_CHECKPOINT_SCHEMA:
            raise MechanismValidationError(
                "WRENCH_CHECKPOINT_SCHEMA_INVALID",
                "wrench checkpoint uses an unsupported schema",
            )
        identity = (
            str(value.get("plugin_id") or ""),
            str(value.get("branch") or ""),
            str(value.get("load_case_id") or ""),
            tuple(str(item) for item in value.get("program_phase_ids") or ()),
        )
        expected_identity = (
            self.plugin_id,
            self.branch,
            self.load_case_id,
            self.program_phase_ids,
        )
        if identity != expected_identity:
            raise MechanismValidationError(
                "WRENCH_CHECKPOINT_BINDING_MISMATCH",
                "wrench checkpoint does not belong to the active load case",
            )
        records = value.get("records")
        if not isinstance(records, Mapping) or set(records) != set(self._records):
            raise MechanismValidationError(
                "WRENCH_CHECKPOINT_RECORDS_INVALID",
                "wrench checkpoint does not contain the active load inventory",
            )
        immutable_record_fields = {
            "load_id",
            "owner",
            "target_body",
            "category",
            "law_id",
            "source_role",
            "target_role",
            "geometry_gate",
            "active_interval",
            "policy_id",
            "task_input",
            "transmission",
            "support",
            "release_required",
            "wrench_kind",
            "policy",
            "initial_vector_si",
        }
        restored_records: dict[str, dict[str, Any]] = {}
        for load_id, current in self._records.items():
            restored = records.get(load_id)
            if (
                not isinstance(restored, Mapping)
                or set(restored) != _RECORD_KEYS
                or any(
                    restored.get(field) != current.get(field)
                    for field in immutable_record_fields
                )
            ):
                raise MechanismValidationError(
                    "WRENCH_CHECKPOINT_RECORDS_INVALID",
                    f"wrench checkpoint changed load {load_id!r}",
                )
            _finite_vector(restored.get("last_vector_si"))
            for counter_name in (
                "update_count",
                "nonzero_update_count",
                "zero_update_count",
                "component_event_count",
            ):
                counter = restored.get(counter_name)
                if (
                    isinstance(counter, bool)
                    or not isinstance(counter, int)
                    or counter < 0
                ):
                    raise MechanismValidationError(
                        "WRENCH_CHECKPOINT_RECORDS_INVALID",
                        f"wrench checkpoint has an invalid {counter_name}",
                    )
            restored_records[load_id] = deepcopy(dict(restored))

        phase_context_ids = value.get("phase_context_ids")
        task_input_events = value.get("task_input_events")
        phase_barriers = value.get("phase_barriers")
        open_phase = value.get("open_phase")
        executed_phases = value.get("executed_phases")
        if (
            not isinstance(phase_context_ids, list)
            or not isinstance(task_input_events, list)
            or not isinstance(phase_barriers, list)
            or (open_phase is not None and not isinstance(open_phase, Mapping))
            or (
                executed_phases is not None
                and not isinstance(executed_phases, list)
            )
        ):
            raise MechanismValidationError(
                "WRENCH_CHECKPOINT_STATE_INVALID",
                "wrench checkpoint contains malformed mutable state",
            )
        open_phase_id = (
            None
            if open_phase is None
            else str(open_phase.get("phase_id") or "")
        )
        if open_phase_id is not None and (
            not open_phase_id
            or open_phase_id not in self.program_phase_ids
        ):
            raise MechanismValidationError(
                "WRENCH_CHECKPOINT_STATE_INVALID",
                "wrench checkpoint has an invalid open phase",
            )

        self._records = restored_records
        self._phase_context_ids = [
            str(item) for item in phase_context_ids
        ]
        self._task_input_events = deepcopy(task_input_events)
        self._phase_barriers = deepcopy(phase_barriers)
        self._open_phase = (
            None if open_phase is None else deepcopy(dict(open_phase))
        )
        self._executed_phases = (
            None if executed_phases is None else deepcopy(executed_phases)
        )
        executed_load_sha256 = value.get("executed_load_sha256")
        self._executed_load_sha256 = (
            None
            if executed_load_sha256 is None
            else str(executed_load_sha256)
        )
        return open_phase_id

    def _phase_wrench_kind(self, phase_id: str) -> str:
        for phase in self.program_phases:
            if str(phase.get("phase_id") or "") == phase_id:
                return str(phase.get("wrench_kind") or "")
        return ""

    def _phase_record(self, phase_id: str) -> Mapping[str, Any]:
        for phase in self.program_phases:
            if str(phase.get("phase_id") or "") == phase_id:
                return phase
        raise _task_input_law_error(
            f"task input uses phase {phase_id!r} without a registered load law"
        )

    def _validate_task_input_law(
        self,
        *,
        phase_id: str,
        wrench_kind: str,
        vector: Sequence[float],
        law_id: str,
    ) -> None:
        value = _finite_vector(vector)
        if _is_zero(value):
            return
        if law_id == _TESTCASE_TASK_INPUT_LAW:
            # This law exists only for the validator-owned conditional-interface
            # physics testcase, whose one interval intentionally combines an
            # axial force and a shaft torque. It is never registered by a
            # production mechanism plugin.
            return
        if law_id not in _REGISTERED_TASK_INPUT_LAWS:
            raise _task_input_law_error(
                f"task input uses unregistered law {law_id!r}"
            )
        phase = self._phase_record(phase_id)
        phase_kind = str(phase.get("wrench_kind") or "")
        if phase_kind == "none":
            raise _task_input_law_error(
                f"zero-input phase {phase_id!r} received a nonzero vector"
            )
        if phase_kind not in {"force", "torque"} or wrench_kind != phase_kind:
            raise _task_input_law_error(
                f"task input kind does not match phase {phase_id!r}"
            )
        direction = phase.get("direction_abs")
        if (
            not isinstance(direction, (list, tuple))
            or len(direction) != 3
        ):
            raise _task_input_law_error(
                f"phase {phase_id!r} has no registered input direction"
            )
        axis = _finite_vector(direction)
        axis_norm = math.sqrt(sum(item * item for item in axis))
        if abs(axis_norm - 1.0) > 128.0 * math.ulp(1.0):
            raise _task_input_law_error(
                f"phase {phase_id!r} input direction is not a unit vector"
            )
        magnitude_raw = phase.get("magnitude_si")
        if (
            isinstance(magnitude_raw, bool)
            or not isinstance(magnitude_raw, (int, float))
            or not math.isfinite(float(magnitude_raw))
            or float(magnitude_raw) <= 0.0
        ):
            raise _task_input_law_error(
                f"phase {phase_id!r} has no positive registered magnitude"
            )
        magnitude = float(magnitude_raw)
        numeric_scale = max(
            1.0,
            magnitude,
            math.sqrt(sum(item * item for item in value)),
        )
        numeric_tolerance = 256.0 * math.ulp(1.0) * numeric_scale
        if law_id in _DYNAMIC_AXIS_TASK_INPUT_LAWS:
            if (
                math.sqrt(sum(item * item for item in value))
                > magnitude + numeric_tolerance
            ):
                raise _task_input_law_error(
                    f"dynamic actuator exceeds its registered phase envelope "
                    f"in {phase_id!r}"
                )
            return
        projection = sum(item * direction_item for item, direction_item in zip(value, axis))
        residual = tuple(
            item - projection * direction_item
            for item, direction_item in zip(value, axis)
        )
        if math.sqrt(sum(item * item for item in residual)) > numeric_tolerance:
            raise _task_input_law_error(
                f"task input vector is off the registered axis in phase {phase_id!r}"
            )
        if law_id == _OPPOSED_TASK_INPUT_LAW:
            expected_projection = -magnitude
            if abs(projection - expected_projection) > numeric_tolerance:
                raise _task_input_law_error(
                    f"paired reaction does not oppose the registered phase "
                    f"vector in {phase_id!r}"
                )
            return
        bound_scale = _BOUNDED_TASK_INPUT_LAWS.get(law_id)
        if bound_scale is not None:
            if abs(projection) > bound_scale * magnitude + numeric_tolerance:
                raise _task_input_law_error(
                    f"controller input exceeds its registered phase envelope "
                    f"in {phase_id!r}"
                )
            return
        end_raw = phase.get("end_magnitude_si")
        if end_raw is None:
            minimum = maximum = magnitude
        elif (
            isinstance(end_raw, bool)
            or not isinstance(end_raw, (int, float))
            or not math.isfinite(float(end_raw))
            or float(end_raw) < 0.0
        ):
            raise _task_input_law_error(
                f"phase {phase_id!r} has an invalid ramp endpoint"
            )
        else:
            minimum, maximum = sorted((magnitude, float(end_raw)))
        if (
            projection < minimum - numeric_tolerance
            or projection > maximum + numeric_tolerance
        ):
            raise _task_input_law_error(
                f"task input vector differs from the registered phase law "
                f"in {phase_id!r}"
            )

    def _task_input_state(self) -> list[dict[str, Any]]:
        return [
            {
                "load_id": str(record["load_id"]),
                "target_body": str(record["target_body"]),
                "policy_id": str(record["policy_id"]),
                "vector_si": list(_finite_vector(record["last_vector_si"])),
            }
            for _, record in sorted(self._records.items())
            if record["task_input"] is True
        ]

    @staticmethod
    def _state_is_zero(state: Sequence[Mapping[str, Any]]) -> bool:
        return all(_is_zero(item["vector_si"]) for item in state)

    def begin_phase(self, phase_id: str) -> None:
        if self._open_phase is not None:
            raise MechanismValidationError(
                "WRENCH_PHASE_CONTEXT_STALE",
                "the previous wrench phase must be closed before transition",
            )
        if self.program_phase_ids:
            expected_index = len(self._phase_barriers)
            if (
                expected_index >= len(self.program_phase_ids)
                or phase_id != self.program_phase_ids[expected_index]
            ):
                raise MechanismValidationError(
                    "WRENCH_PHASE_ORDER_INVALID",
                    "wrench phase traversal differs from the validator program",
                )
        entry_state = self._task_input_state()
        wrench_kind = self._phase_wrench_kind(phase_id)
        entry_all_zero = self._state_is_zero(entry_state)
        if wrench_kind == "none" and not entry_all_zero:
            raise MechanismValidationError(
                "WRENCH_PHASE_ENTRY_TASK_INPUT_NONZERO",
                "a zero-input phase begins with persistent task-input wrench state",
            )
        self._open_phase = {
            "phase_id": phase_id,
            "wrench_kind": wrench_kind,
            "entry_task_input_state": entry_state,
            "entry_all_zero": entry_all_zero,
            "task_input_update_count": 0,
            "task_input_nonzero_update_count": 0,
            "task_input_zero_update_count": 0,
            "aggregated_task_input_event_count": 0,
        }
        self.record_phase_context(phase_id)

    def end_phase(self, phase_id: str) -> None:
        opened = self._open_phase
        if opened is None or opened["phase_id"] != phase_id:
            raise MechanismValidationError(
                "WRENCH_PHASE_CONTEXT_INVALID",
                "wrench phase close does not match the open validator phase",
            )
        exit_state = self._task_input_state()
        exit_all_zero = self._state_is_zero(exit_state)
        barrier = {
            **opened,
            "exit_task_input_state": exit_state,
            "exit_all_zero": exit_all_zero,
        }
        barrier["barrier_sha256"] = canonical_sha256(barrier)
        self._phase_barriers.append(barrier)
        self._open_phase = None
        if opened["wrench_kind"] == "none" and not exit_all_zero:
            raise MechanismValidationError(
                "WRENCH_NONE_PHASE_TASK_INPUT_NONZERO",
                "a zero-input phase ended with a nonzero task-input wrench",
            )

    def close_context(self) -> None:
        if self._open_phase is not None:
            self.end_phase(str(self._open_phase["phase_id"]))
        residuals = [
            item["load_id"]
            for item in self._task_input_state()
            if not _is_zero(item["vector_si"])
        ]
        if residuals:
            raise MechanismValidationError(
                "WRENCH_LEDGER_CLEANUP_TASK_INPUT_NONZERO",
                "task-input wrench state was not zeroed before ledger cleanup: "
                + ", ".join(residuals),
            )

    def register(
        self,
        declaration: WrenchDeclaration,
        *,
        wrench_kind: str,
        initial_vector: Any,
    ) -> None:
        if declaration.load_id in self._records:
            raise MechanismValidationError(
                "WRENCH_LOAD_ID_DUPLICATE",
                f"duplicate wrench load ID {declaration.load_id!r}",
            )
        if declaration.task_input and self._open_phase is not None:
            raise MechanismValidationError(
                "WRENCH_TASK_INPUT_REGISTERED_DURING_PHASE",
                "task-input loads must be registered before phase traversal",
            )
        if not all(
            (
                declaration.owner,
                declaration.target_body,
                declaration.category,
                declaration.law_id,
                declaration.source_role,
                declaration.target_role,
                declaration.geometry_gate,
                declaration.active_interval,
                declaration.policy_id,
            )
        ):
            raise MechanismValidationError(
                "WRENCH_PROVENANCE_INCOMPLETE",
                f"wrench {declaration.load_id!r} has incomplete provenance",
            )
        policy = self.policies.get(declaration.policy_id)
        if policy is None:
            raise MechanismValidationError(
                "WRENCH_POLICY_NOT_ALLOWED",
                f"plugin {self.plugin_id!r} did not declare policy "
                f"{declaration.policy_id!r}",
            )
        actual = (
            wrench_kind,
            declaration.category,
            declaration.law_id,
            declaration.target_role,
            declaration.task_input,
            declaration.transmission,
            declaration.support,
        )
        expected_prefix = (
            policy.wrench_kind,
            policy.category,
            policy.law_id,
        )
        if actual[:3] != expected_prefix:
            raise MechanismValidationError(
                "WRENCH_POLICY_MISMATCH",
                f"wrench {declaration.load_id!r} does not match policy "
                f"{policy.policy_id!r}",
            )
        if declaration.target_role not in policy.target_roles:
            raise MechanismValidationError(
                "WRENCH_TARGET_ROLE_FORBIDDEN",
                f"policy {policy.policy_id!r} does not allow target role "
                f"{declaration.target_role!r}",
            )
        if (
            declaration.task_input
            and declaration.target_body in self.declared_output_body_ids
            and declaration.target_body not in self.declared_input_body_ids
        ):
            raise MechanismValidationError(
                "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
                f"task input {declaration.load_id!r} directly drives task "
                f"output body {declaration.target_body!r}",
            )
        if (
            declaration.target_role == "declared_input"
            and self.declared_input_body_ids
            and declaration.target_body not in self.declared_input_body_ids
        ):
            raise MechanismValidationError(
                "WRENCH_INPUT_BODY_INVALID",
                f"wrench {declaration.load_id!r} claims the input role on "
                f"undeclared body {declaration.target_body!r}",
            )
        if (
            declaration.target_role == "task_output"
            and self.declared_output_body_ids
            and declaration.target_body not in self.declared_output_body_ids
        ):
            raise MechanismValidationError(
                "WRENCH_OUTPUT_BODY_INVALID",
                f"wrench {declaration.load_id!r} claims the output role on "
                f"undeclared body {declaration.target_body!r}",
            )
        if actual[4:] != (
            policy.task_input,
            policy.transmission,
            policy.support,
        ):
            raise MechanismValidationError(
                "WRENCH_POLICY_FLAGS_MISMATCH",
                f"wrench {declaration.load_id!r} changes the semantic flags "
                f"declared by policy {policy.policy_id!r}",
            )
        if policy.capability_required:
            from .family_plugins import (
                registration_allows_runtime_execution,
            )

            capability_allowed = (
                self.capability_registration.get("plugin_id")
                == self.plugin_id
                and registration_allows_runtime_execution(
                    self.capability_registration
                )
            )
        else:
            capability_allowed = True
        if not capability_allowed:
            raise MechanismValidationError(
                "WRENCH_CAPABILITY_NOT_ACTIVE",
                f"analytical wrench {declaration.load_id!r} requires an "
                "ACTIVE capability or its live exact-one-test dispatch",
            )
        if declaration.task_input:
            if declaration.target_role != "declared_input":
                raise MechanismValidationError(
                    "WRENCH_INPUT_TARGET_INVALID",
                    f"task input {declaration.load_id!r} does not target the "
                    "declared input role",
                )
            if (
                declaration.target_body in self.declared_output_body_ids
                and declaration.target_body not in self.declared_input_body_ids
            ):
                raise MechanismValidationError(
                    "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
                    f"task input {declaration.load_id!r} directly drives task "
                    f"output body {declaration.target_body!r}",
                )
            if (
                self.declared_input_body_ids
                and declaration.target_body not in self.declared_input_body_ids
            ):
                raise MechanismValidationError(
                    "WRENCH_INPUT_BODY_INVALID",
                    f"task input {declaration.load_id!r} targets undeclared body "
                    f"{declaration.target_body!r}",
                )
        if (
            declaration.target_body in self.declared_output_body_ids
            and declaration.target_body not in self.declared_input_body_ids
        ):
            if declaration.category in {"task_input", "controller_drive"}:
                raise MechanismValidationError(
                    "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
                    f"drive wrench {declaration.load_id!r} directly targets the "
                    "task output",
                )
        initial = _finite_vector(initial_vector)
        if declaration.task_input and not _is_zero(initial):
            raise MechanismValidationError(
                "WRENCH_TASK_INPUT_PHASE_UNSET",
                "task-input loads must be registered at zero before a phase begins",
            )
        self._records[declaration.load_id] = {
            **declaration.to_dict(),
            "wrench_kind": wrench_kind,
            "policy": policy.to_dict(),
            "initial_vector_si": list(initial),
            "last_vector_si": list(initial),
            "update_count": 0,
            "nonzero_update_count": 0,
            "zero_update_count": 0,
            "released": _is_zero(initial),
            "component_event_count": 0,
            "component_sources": [],
            "phase_update_counts": {},
            "phase_nonzero_update_counts": {},
            "phase_zero_update_counts": {},
        }

    def record_update(
        self,
        load_id: str,
        vector: Sequence[float],
        *,
        component_source: str | None = None,
    ) -> None:
        if load_id not in self._records:
            raise MechanismValidationError(
                "WRENCH_NOT_REGISTERED",
                f"wrench update bypassed registration: {load_id!r}",
            )
        value = _finite_vector(vector)
        record = self._records[load_id]
        phase_id = current_wrench_phase()
        if phase_id is not None:
            if (
                self.program_phase_ids
                and phase_id not in self.program_phase_ids
            ):
                raise MechanismValidationError(
                    "WRENCH_TASK_INPUT_PHASE_UNKNOWN",
                    f"wrench update uses unknown phase {phase_id!r}",
                )
            updates = record["phase_update_counts"]
            updates[phase_id] = int(updates.get(phase_id, 0)) + 1
            bucket_name = (
                "phase_zero_update_counts"
                if _is_zero(value)
                else "phase_nonzero_update_counts"
            )
            bucket = record[bucket_name]
            bucket[phase_id] = int(bucket.get(phase_id, 0)) + 1
        if record["task_input"]:
            if phase_id is None:
                raise MechanismValidationError(
                    "WRENCH_TASK_INPUT_PHASE_UNSET",
                    "a task-input mutation has no validator-owned phase barrier",
                )
            phase_kind = self._phase_wrench_kind(phase_id)
            if phase_kind == "none" and not _is_zero(value):
                raise MechanismValidationError(
                    "WRENCH_NONE_PHASE_TASK_INPUT_NONZERO",
                    "a zero-input phase cannot apply a nonzero task-input wrench",
                )
            self._validate_task_input_law(
                phase_id=phase_id,
                wrench_kind=str(record["wrench_kind"]),
                vector=value,
                law_id=str(record["law_id"]),
            )
            if self._open_phase is None:
                raise MechanismValidationError(
                    "WRENCH_PHASE_CONTEXT_INVALID",
                    "task-input mutation has no open phase barrier",
                )
            self._open_phase["task_input_update_count"] = (
                int(self._open_phase["task_input_update_count"]) + 1
            )
            counter = (
                "task_input_zero_update_count"
                if _is_zero(value)
                else "task_input_nonzero_update_count"
            )
            self._open_phase[counter] = int(self._open_phase[counter]) + 1
        record["last_vector_si"] = list(value)
        record["update_count"] = int(record["update_count"]) + 1
        if not _is_zero(value):
            record["nonzero_update_count"] = (
                int(record["nonzero_update_count"]) + 1
            )
            record["released"] = False
        else:
            record["zero_update_count"] = int(record["zero_update_count"]) + 1
        if _is_zero(value):
            record["released"] = True
        if component_source is not None:
            source = str(component_source)
            record["component_event_count"] = (
                int(record["component_event_count"]) + 1
            )
            sources = set(record["component_sources"])
            sources.add(source)
            record["component_sources"] = sorted(sources)

    def record_component(
        self,
        load_id: str,
        *,
        component_source: str,
    ) -> None:
        if load_id not in self._records:
            raise MechanismValidationError(
                "WRENCH_NOT_REGISTERED",
                f"wrench component bypassed registration: {load_id!r}",
            )
        record = self._records[load_id]
        record["component_event_count"] = (
            int(record["component_event_count"]) + 1
        )
        sources = set(record["component_sources"])
        sources.add(str(component_source))
        record["component_sources"] = sorted(sources)

    def record_task_input_event(
        self,
        *,
        target_body: str,
        source_role: str,
        active_interval: str,
        vector_si: Sequence[float],
        law_id: str = _EXACT_TASK_INPUT_LAW,
    ) -> None:
        """Audit a task-input component before it enters an aggregated load."""

        body_id = str(target_body)
        source = str(source_role)
        interval = str(active_interval)
        selected_law = str(law_id)
        if (
            not body_id
            or not source
            or not interval
            or selected_law not in _AGGREGATED_TASK_INPUT_LAWS
        ):
            raise MechanismValidationError(
                "WRENCH_INPUT_EVENT_INCOMPLETE",
                "aggregated task-input event requires body, source, interval, "
                "and a registered component law",
            )
        if source != "declared_input_port":
            raise MechanismValidationError(
                "WRENCH_INPUT_SOURCE_INVALID",
                "aggregated task input must use the selected declared-input port",
            )
        if (
            body_id in self.declared_output_body_ids
            and body_id not in self.declared_input_body_ids
        ):
            raise MechanismValidationError(
                "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
                f"aggregated task input directly drives task output body {body_id!r}",
            )
        if self.declared_input_body_ids and body_id not in self.declared_input_body_ids:
            raise MechanismValidationError(
                "WRENCH_INPUT_BODY_INVALID",
                f"aggregated task input targets undeclared body {body_id!r}",
            )
        phase_id = current_wrench_phase()
        if phase_id is None:
            raise MechanismValidationError(
                "WRENCH_TASK_INPUT_PHASE_UNSET",
                "aggregated task-input events require a validator-owned phase",
            )
        if self.program_phase_ids and phase_id not in self.program_phase_ids:
            raise MechanismValidationError(
                "WRENCH_TASK_INPUT_PHASE_UNKNOWN",
                f"aggregated task input uses unknown phase {phase_id!r}",
            )
        if interval != f"phase:{phase_id}":
            raise MechanismValidationError(
                "WRENCH_INPUT_INTERVAL_INVALID",
                "aggregated task input interval must exactly name the active phase",
            )
        if self._phase_wrench_kind(phase_id) == "none":
            raise MechanismValidationError(
                "WRENCH_NONE_PHASE_TASK_INPUT_NONZERO",
                "a zero-input phase cannot contain an aggregated task-input event",
            )
        self._validate_task_input_law(
            phase_id=phase_id,
            wrench_kind=self._phase_wrench_kind(phase_id),
            vector=vector_si,
            law_id=selected_law,
        )
        if self._open_phase is None:
            raise MechanismValidationError(
                "WRENCH_PHASE_CONTEXT_INVALID",
                "aggregated task-input event has no open phase barrier",
            )
        event = {
            "target_body": body_id,
            "source_role": source,
            "active_interval": interval,
            "phase_id": phase_id,
            "law_id": selected_law,
        }
        if event not in self._task_input_events:
            self._task_input_events.append(event)
            self._open_phase["aggregated_task_input_event_count"] = (
                int(self._open_phase["aggregated_task_input_event_count"]) + 1
            )

    def bind_executed_schedule(
        self,
        executed_phases: Any,
        *,
        executed_load_sha256: str | None = None,
    ) -> None:
        """Bind the final runtime schedule before a wrench audit can pass."""

        if not isinstance(executed_phases, list) or not all(
            isinstance(item, Mapping) for item in executed_phases
        ):
            raise MechanismValidationError(
                "WRENCH_EXECUTED_SCHEDULE_INVALID",
                "wrench audit requires an object inventory of executed phases",
            )
        normalized = [dict(item) for item in executed_phases]
        phase_ids = tuple(
            str(item.get("phase_id") or "") for item in normalized
        )
        if self.program_phase_ids and phase_ids != self.program_phase_ids:
            raise MechanismValidationError(
                "WRENCH_EXECUTED_SCHEDULE_MISMATCH",
                "executed phase order differs from the validator load program",
            )
        digest = canonical_sha256(normalized)
        if executed_load_sha256 is not None and executed_load_sha256 != digest:
            raise MechanismValidationError(
                "WRENCH_EXECUTED_SCHEDULE_HASH_MISMATCH",
                "executed schedule digest differs from the trace",
            )
        self._executed_phases = normalized
        self._executed_load_sha256 = digest

    def record_phase_context(self, phase_id: str) -> None:
        if (
            self.program_phase_ids
            and phase_id not in self.program_phase_ids
        ):
            raise MechanismValidationError(
                "WRENCH_TASK_INPUT_PHASE_UNKNOWN",
                f"wrench phase {phase_id!r} is not in the validator program",
            )
        if not self._phase_context_ids or self._phase_context_ids[-1] != phase_id:
            self._phase_context_ids.append(phase_id)

    def audit(self) -> dict[str, Any]:
        violations: list[dict[str, str]] = []
        task_input_bodies: set[str] = set()
        task_input_sources: set[str] = set()
        task_input_nonzero_phases: set[str] = set()
        task_input_zero_phases: set[str] = set()
        for load_id, record in sorted(self._records.items()):
            policy = record["policy"]
            if record["task_input"] and int(record["nonzero_update_count"]) > 0:
                task_input_bodies.add(str(record["target_body"]))
                task_input_sources.add(str(record["source_role"]))
                task_input_nonzero_phases.update(
                    str(item)
                    for item in record["phase_nonzero_update_counts"]
                )
            if record["task_input"]:
                task_input_zero_phases.update(
                    str(item) for item in record["phase_zero_update_counts"]
                )
            if (
                record["release_required"]
                and not _is_zero(record["last_vector_si"])
            ):
                violations.append(
                    {
                        "code": "WRENCH_RELEASE_RESIDUAL",
                        "load_id": load_id,
                        "message": "release-required wrench retains a nonzero value",
                    }
                )
            if (
                policy.get("component_provenance_required")
                and (
                    int(record["nonzero_update_count"]) > 0
                    or not _is_zero(record["initial_vector_si"])
                )
                and int(record["component_event_count"]) == 0
            ):
                violations.append(
                    {
                        "code": "WRENCH_COMPONENT_PROVENANCE_MISSING",
                        "load_id": load_id,
                        "message": "aggregated wrench has no audited component source",
                    }
                )
        task_input_bodies.update(
            item["target_body"] for item in self._task_input_events
        )
        task_input_sources.update(
            item["source_role"] for item in self._task_input_events
        )
        task_input_nonzero_phases.update(
            item["phase_id"] for item in self._task_input_events
        )
        if self.program_phase_ids and self._executed_phases is None:
            violations.append(
                {
                    "code": "WRENCH_EXECUTED_SCHEDULE_UNBOUND",
                    "load_id": "",
                    "message": "wrench audit is not bound to the executed schedule",
                }
            )
        if self._open_phase is not None:
            violations.append(
                {
                    "code": "WRENCH_PHASE_CONTEXT_STALE",
                    "load_id": "",
                    "message": "wrench audit was requested with an open phase",
                }
            )
        if self.program_phase_ids and tuple(
            str(item["phase_id"]) for item in self._phase_barriers
        ) != self.program_phase_ids:
            violations.append(
                {
                    "code": "WRENCH_PHASE_BARRIER_COVERAGE_INCOMPLETE",
                    "load_id": "",
                    "message": "wrench phase barriers do not cover the complete program",
                }
            )
        if len(task_input_sources) > 1:
            violations.append(
                {
                    "code": "WRENCH_MULTIPLE_TASK_INPUT_CHANNELS",
                    "load_id": "",
                    "message": "one load case applied more than one task input channel",
                }
            )
        summary = {
            "schema_version": WRENCH_AUDIT_SCHEMA,
            "ledger_schema_version": WRENCH_LEDGER_SCHEMA,
            "plugin_id": self.plugin_id,
            "capability_registration": self.capability_registration,
            "branch": self.branch,
            "load_case_id": self.load_case_id,
            "program_phase_ids": list(self.program_phase_ids),
            "program_phases": [dict(item) for item in self.program_phases],
            "phase_context_ids": list(self._phase_context_ids),
            "phase_barriers": [dict(item) for item in self._phase_barriers],
            "executed_phase_ids": (
                []
                if self._executed_phases is None
                else [
                    str(item.get("phase_id") or "")
                    for item in self._executed_phases
                ]
            ),
            "executed_load_sha256": self._executed_load_sha256,
            "declared_input_body_ids": sorted(self.declared_input_body_ids),
            "declared_output_body_ids": sorted(self.declared_output_body_ids),
            "load_count": len(self._records),
            "task_input_load_count": sum(
                1 for item in self._records.values() if item["task_input"]
            ),
            "transmission_load_count": sum(
                1 for item in self._records.values() if item["transmission"]
            ),
            "support_load_count": sum(
                1 for item in self._records.values() if item["support"]
            ),
            "task_input_bodies_observed": sorted(task_input_bodies),
            "task_input_sources_observed": sorted(task_input_sources),
            "task_input_nonzero_phase_ids": sorted(
                task_input_nonzero_phases
            ),
            "task_input_zero_phase_ids": sorted(task_input_zero_phases),
            "aggregated_task_input_events": sorted(
                self._task_input_events,
                key=lambda item: (
                    item["active_interval"],
                    item["target_body"],
                    item["source_role"],
                ),
            ),
            "direct_output_drive_detected": any(
                item["code"] == "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN"
                for item in violations
            ),
            "release_residual_count": sum(
                1 for item in violations if item["code"] == "WRENCH_RELEASE_RESIDUAL"
            ),
            "violations": violations,
            "passed": not violations,
            "records": [self._records[key] for key in sorted(self._records)],
        }
        summary["ledger_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in summary.items()
                if key != "ledger_sha256"
            }
        )
        return summary

    def verify_for_certification(self) -> dict[str, Any]:
        summary = self.audit()
        if not summary["passed"]:
            codes = sorted(
                {str(item["code"]) for item in summary["violations"]}
            )
            registration_source = str(
                self.capability_registration.get("registration_source") or ""
            )
            owner = mechanism_owner_for_registration_source(registration_source)
            reviewer_owned = owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
            raise MechanismValidationError(
                "WRENCH_AUDIT_FAILED",
                "wrench provenance audit failed: " + ", ".join(codes),
                details={
                    "plugin_id": self.plugin_id,
                    "branch": self.branch,
                    "load_case_id": self.load_case_id,
                },
                failure=make_failure_envelope(
                    "WRENCH_AUDIT_FAILED",
                    owner,
                    source_component="mechanism.wrench_ledger",
                    message=(
                        "wrench provenance audit failed: " + ", ".join(codes)
                    ),
                    evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
                    plugin_id=(self.plugin_id if reviewer_owned else None),
                    plugin_kind=("mechanism" if reviewer_owned else None),
                    causes=[
                        {
                            "code": str(item.get("code") or "WRENCH_AUDIT_VIOLATION"),
                            "message": str(
                                item.get("message") or "wrench audit violation"
                            ),
                            "details": {
                                key: value
                                for key, value in item.items()
                                if key not in {"code", "message"}
                            },
                        }
                        for item in summary["violations"]
                    ],
                    details={
                        "branch": self.branch,
                        "load_case_id": self.load_case_id,
                    },
                ),
            )
        return summary


_CURRENT_LEDGER: contextvars.ContextVar[WrenchLedger | None] = (
    contextvars.ContextVar("simulation_ready_cad_wrench_ledger", default=None)
)
_CURRENT_PHASE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "simulation_ready_cad_wrench_phase",
    default=None,
)


@contextlib.contextmanager
def activate_wrench_ledger(ledger: WrenchLedger) -> Iterator[WrenchLedger]:
    ledger_token = _CURRENT_LEDGER.set(ledger)
    phase_token = _CURRENT_PHASE.set(None)
    failed = False
    try:
        yield ledger
    except BaseException:
        failed = True
        raise
    finally:
        try:
            if not failed:
                ledger.close_context()
        finally:
            _CURRENT_PHASE.reset(phase_token)
            _CURRENT_LEDGER.reset(ledger_token)


def current_wrench_ledger() -> WrenchLedger:
    ledger = _CURRENT_LEDGER.get()
    if ledger is None:
        raise MechanismValidationError(
            "WRENCH_LEDGER_REQUIRED",
            "native body wrenches require an active validator-owned ledger",
        )
    return ledger


def current_wrench_phase() -> str | None:
    return _CURRENT_PHASE.get()


def restore_current_wrench_checkpoint(value: Mapping[str, Any]) -> str | None:
    """Restore the active ledger and its already-open phase in one operation."""

    if _CURRENT_PHASE.get() is not None:
        raise MechanismValidationError(
            "WRENCH_CHECKPOINT_PHASE_STALE",
            "wrench checkpoint restore requires a fresh phase context",
        )
    phase_id = current_wrench_ledger().restore_checkpoint_state(value)
    _CURRENT_PHASE.set(phase_id)
    return phase_id


def set_current_wrench_phase(phase_id: str | None) -> None:
    """Set or clear the phase context for subsequent task-input mutations.

    A caller must explicitly clear the previous phase before selecting a new
    one.  This catches stale context at a phase transition instead of silently
    attributing a later input to the wrong window.
    """

    ledger = current_wrench_ledger()
    current = _CURRENT_PHASE.get()
    if phase_id is None:
        if current is not None:
            try:
                ledger.end_phase(current)
            finally:
                _CURRENT_PHASE.set(None)
        return
    selected = str(phase_id)
    if not selected:
        raise MechanismValidationError(
            "WRENCH_TASK_INPUT_PHASE_INVALID",
            "wrench phase ID cannot be empty",
        )
    if current is not None and current != selected:
        raise MechanismValidationError(
            "WRENCH_PHASE_CONTEXT_STALE",
            "the previous wrench phase must be cleared before transition",
        )
    if ledger.program_phase_ids and selected not in ledger.program_phase_ids:
        raise MechanismValidationError(
            "WRENCH_TASK_INPUT_PHASE_UNKNOWN",
            f"wrench phase {selected!r} is not in the validator load program",
        )
    ledger.begin_phase(selected)
    _CURRENT_PHASE.set(selected)


def transition_current_wrench_phase(phase_id: str) -> None:
    """Clear the prior phase and select the next validator load phase."""

    set_current_wrench_phase(None)
    set_current_wrench_phase(phase_id)


def current_wrench_audit_or_none() -> dict[str, Any] | None:
    ledger = _CURRENT_LEDGER.get()
    return None if ledger is None else ledger.audit()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    code: str,
    label: str,
) -> None:
    actual = frozenset(str(item) for item in value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise MechanismValidationError(
            code,
            f"{label} keys are not canonical; missing={missing}, extra={extra}",
        )


def _require_counter(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MechanismValidationError(
            "WRENCH_AUDIT_COUNTER_INVALID",
            f"{label} must be a nonnegative integer",
        )
    return value


def _validated_state_inventory(
    value: Any,
    *,
    records_by_id: Mapping[str, Mapping[str, Any]],
    label: str,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise MechanismValidationError(
            "WRENCH_PHASE_BARRIER_INVALID",
            f"{label} must be an ordered object inventory",
        )
    normalized: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "WRENCH_PHASE_BARRIER_INVALID",
                f"{label} contains a non-object state record",
            )
        item = dict(raw)
        _require_exact_keys(
            item,
            _TASK_INPUT_STATE_KEYS,
            code="WRENCH_PHASE_BARRIER_INVALID",
            label=f"{label} state",
        )
        load_id = str(item["load_id"])
        record = records_by_id.get(load_id)
        if (
            record is None
            or record.get("task_input") is not True
            or item["target_body"] != record.get("target_body")
            or item["policy_id"] != record.get("policy_id")
        ):
            raise MechanismValidationError(
                "WRENCH_PHASE_BARRIER_INVALID",
                f"{label} does not match the task-input load inventory",
            )
        item["vector_si"] = list(_finite_vector(item["vector_si"]))
        normalized.append(item)
    expected_order = sorted(
        load_id
        for load_id, record in records_by_id.items()
        if record.get("task_input") is True
    )
    if [item["load_id"] for item in normalized] != expected_order:
        raise MechanismValidationError(
            "WRENCH_PHASE_BARRIER_INVALID",
            f"{label} is incomplete, duplicated, or not canonically ordered",
        )
    return normalized


def verify_saved_wrench_audit(
    value: Mapping[str, Any] | None,
    *,
    expected_plugin_id: str | None = None,
    expected_branch: str | None = None,
    expected_load_case_id: str | None = None,
    expected_phase_ids: Sequence[str] | None = None,
    expected_program_phases: Sequence[Mapping[str, Any]] | None = None,
    expected_executed_load_sha256: str | None = None,
    expected_input_body_ids: Sequence[str] | None = None,
    expected_output_body_ids: Sequence[str] | None = None,
    expected_task_input_body_id: str | None = None,
    expected_task_input_body_ids: Sequence[str] | None = None,
    expected_task_output_body_id: str | None = None,
    powered_phase_ids: Sequence[str] = (),
    return_phase_ids: Sequence[str] = (),
    return_input_policy: str | None = None,
    allow_multiple_task_input_channels: bool = False,
    allow_direct_output_actuation: bool | None = None,
) -> dict[str, Any]:
    """Revalidate a saved worker audit before physical evidence can certify."""

    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "WRENCH_AUDIT_MISSING",
            "physical trace has no validator-owned wrench audit",
        )
    audit = dict(value)
    _require_exact_keys(
        audit,
        _AUDIT_KEYS,
        code="WRENCH_AUDIT_SCHEMA_INVALID",
        label="saved wrench audit",
    )
    if (
        audit.get("schema_version") != WRENCH_AUDIT_SCHEMA
        or audit.get("ledger_schema_version") != WRENCH_LEDGER_SCHEMA
    ):
        raise MechanismValidationError(
            "WRENCH_AUDIT_SCHEMA_INVALID",
            "physical trace wrench audit uses an unsupported schema",
        )
    expected_digest = audit.get("ledger_sha256")
    actual_digest = canonical_sha256(
        {
            key: item
            for key, item in audit.items()
            if key != "ledger_sha256"
        }
    )
    if expected_digest != actual_digest:
        raise MechanismValidationError(
            "WRENCH_AUDIT_HASH_INVALID",
            "physical trace wrench audit digest does not match its contents",
        )
    if audit.get("passed") is not True or audit.get("violations") != []:
        raise MechanismValidationError(
            "WRENCH_AUDIT_FAILED",
            "physical trace wrench audit is not passing",
        )
    plugin_id = str(audit.get("plugin_id") or "")
    expected_context = {
        "plugin_id": expected_plugin_id,
        "branch": expected_branch,
        "load_case_id": expected_load_case_id,
    }
    for field, expected_value in expected_context.items():
        if (
            expected_value is not None
            and audit.get(field) != expected_value
        ):
            raise MechanismValidationError(
                "WRENCH_AUDIT_CONTEXT_MISMATCH",
                f"saved wrench audit {field} differs from the task trace",
            )
    from .family_plugins import (
        plugin_registration_for,
        wrench_policies_for_plugin,
    )

    registration = plugin_registration_for(plugin_id)
    if (
        registration.get("status") != "ACTIVE"
        or audit.get("capability_registration") != registration
    ):
        raise MechanismValidationError(
            "WRENCH_CAPABILITY_NOT_ACTIVE",
            "saved wrench audit is not bound to the current ACTIVE capability",
        )
    allowed = {
        item.policy_id: item for item in wrench_policies_for_plugin(plugin_id)
    }
    records = audit.get("records")
    if (
        not isinstance(records, list)
        or _require_counter(audit.get("load_count"), label="load_count")
        != len(records)
        or len(
            {
                str(item.get("load_id") or "")
                for item in records
                if isinstance(item, Mapping)
            }
        )
        != len(records)
    ):
        raise MechanismValidationError(
            "WRENCH_AUDIT_RECORDS_INVALID",
            "saved wrench audit has an invalid load inventory",
        )
    declared_inputs = {
        str(item) for item in audit.get("declared_input_body_ids") or ()
    }
    declared_outputs = {
        str(item) for item in audit.get("declared_output_body_ids") or ()
    }
    if expected_input_body_ids is not None and declared_inputs != {
        str(item) for item in expected_input_body_ids
    }:
        raise MechanismValidationError(
            "WRENCH_AUDIT_CONTEXT_MISMATCH",
            "saved wrench input-body inventory differs from the active adapter",
        )
    if expected_output_body_ids is not None and declared_outputs != {
        str(item) for item in expected_output_body_ids
    }:
        raise MechanismValidationError(
            "WRENCH_AUDIT_CONTEXT_MISMATCH",
            "saved wrench output-body inventory differs from the active adapter",
        )
    program_phase_ids = audit.get("program_phase_ids")
    program_phases = audit.get("program_phases")
    phase_context_ids = audit.get("phase_context_ids")
    executed_phase_ids = audit.get("executed_phase_ids")
    if (
        not isinstance(program_phase_ids, list)
        or not isinstance(program_phases, list)
        or not isinstance(phase_context_ids, list)
        or not isinstance(executed_phase_ids, list)
        or not all(isinstance(item, Mapping) for item in program_phases)
        or [str(item.get("phase_id") or "") for item in program_phases]
        != program_phase_ids
        or any(not isinstance(item, str) or not item for item in program_phase_ids)
        or len(set(program_phase_ids)) != len(program_phase_ids)
        or phase_context_ids != program_phase_ids
        or executed_phase_ids != program_phase_ids
    ):
        raise MechanismValidationError(
            "WRENCH_EXECUTED_SCHEDULE_INVALID",
            "saved wrench audit lacks its phase inventories",
        )
    if expected_phase_ids is not None:
        expected_phases = [str(item) for item in expected_phase_ids]
        if (
            program_phase_ids != expected_phases
            or executed_phase_ids != expected_phases
        ):
            raise MechanismValidationError(
                "WRENCH_AUDIT_CONTEXT_MISMATCH",
                "saved wrench phases differ from the executed task schedule",
            )
        if not isinstance(audit.get("executed_load_sha256"), str):
            raise MechanismValidationError(
                "WRENCH_EXECUTED_SCHEDULE_INVALID",
                "saved wrench audit lacks the executed schedule digest",
            )
    if expected_program_phases is not None:
        expected_program = [
            dict(item) for item in expected_program_phases
        ]
        if program_phases != expected_program:
            raise MechanismValidationError(
                "WRENCH_AUDIT_PROGRAM_LAW_MISMATCH",
                "saved wrench phase laws differ from the registered load program",
            )
    if (
        expected_executed_load_sha256 is not None
        and audit.get("executed_load_sha256")
        != expected_executed_load_sha256
    ):
        raise MechanismValidationError(
            "WRENCH_AUDIT_CONTEXT_MISMATCH",
            "wrench audit schedule digest differs from the task trace",
        )
    observed_nonzero_phases: set[str] = set()
    observed_zero_phases: set[str] = set()
    records_by_id: dict[str, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "WRENCH_AUDIT_RECORDS_INVALID",
                "saved wrench audit contains a non-object record",
            )
        record = dict(raw)
        _require_exact_keys(
            record,
            _RECORD_KEYS,
            code="WRENCH_AUDIT_RECORDS_INVALID",
            label="saved wrench record",
        )
        load_id = str(record["load_id"])
        if not load_id or load_id in records_by_id:
            raise MechanismValidationError(
                "WRENCH_AUDIT_RECORDS_INVALID",
                "saved wrench record IDs must be non-empty and unique",
            )
        records_by_id[load_id] = record
        policy_id = str(record.get("policy_id") or "")
        policy = allowed.get(policy_id)
        raw_policy = record.get("policy")
        if not isinstance(raw_policy, Mapping):
            raise MechanismValidationError(
                "WRENCH_POLICY_NOT_ALLOWED",
                f"saved wrench record uses undeclared policy {policy_id!r}",
            )
        _require_exact_keys(
            raw_policy,
            _POLICY_KEYS,
            code="WRENCH_POLICY_NOT_ALLOWED",
            label="saved wrench policy",
        )
        if policy is None or dict(raw_policy) != policy.to_dict():
            raise MechanismValidationError(
                "WRENCH_POLICY_NOT_ALLOWED",
                f"saved wrench record uses undeclared policy {policy_id!r}",
            )
        if (
            record.get("wrench_kind") != policy.wrench_kind
            or record.get("category") != policy.category
            or record.get("law_id") != policy.law_id
            or record.get("target_role") not in policy.target_roles
            or record.get("task_input") is not policy.task_input
            or record.get("transmission") is not policy.transmission
            or record.get("support") is not policy.support
        ):
            raise MechanismValidationError(
                "WRENCH_POLICY_MISMATCH",
                f"saved wrench record does not match policy {policy_id!r}",
            )
        required_text = (
            "load_id",
            "owner",
            "target_body",
            "category",
            "law_id",
            "source_role",
            "target_role",
            "geometry_gate",
            "active_interval",
        )
        if any(not str(record.get(field) or "") for field in required_text):
            raise MechanismValidationError(
                "WRENCH_PROVENANCE_INCOMPLETE",
                "saved wrench record has incomplete provenance",
            )
        if record.get("owner") != plugin_id:
            raise MechanismValidationError(
                "WRENCH_OWNER_INVALID",
                "saved wrench owner is not the registered mechanism plugin",
            )
        if record.get("task_input") is True and record.get(
            "source_role"
        ) != "declared_input_port":
            raise MechanismValidationError(
                "WRENCH_INPUT_SOURCE_INVALID",
                "saved task input does not use the selected declared-input port",
            )
        target = str(record["target_body"])
        if (
            record.get("target_role") == "declared_input"
            and declared_inputs
            and target not in declared_inputs
        ):
            raise MechanismValidationError(
                "WRENCH_INPUT_BODY_INVALID",
                "saved wrench claims the input role on an undeclared body",
            )
        if (
            record.get("target_role") == "task_output"
            and declared_outputs
            and target not in declared_outputs
        ):
            raise MechanismValidationError(
                "WRENCH_OUTPUT_BODY_INVALID",
                "saved wrench claims the output role on an undeclared body",
            )
        if record.get("task_input") is True:
            if (
                record.get("target_role") != "declared_input"
                or (declared_inputs and target not in declared_inputs)
            ):
                raise MechanismValidationError(
                    "WRENCH_INPUT_BODY_INVALID",
                    "saved task input does not target a declared input body",
                )
            if (
                allow_direct_output_actuation is not True
                and target in declared_outputs
                and target not in declared_inputs
            ):
                raise MechanismValidationError(
                    "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
                    "saved task input directly targets the task output",
                )
        initial_vector = _finite_vector(record.get("initial_vector_si"))
        last_vector = _finite_vector(record.get("last_vector_si"))
        update_count = _require_counter(
            record.get("update_count"),
            label=f"{load_id}.update_count",
        )
        nonzero_update_count = _require_counter(
            record.get("nonzero_update_count"),
            label=f"{load_id}.nonzero_update_count",
        )
        zero_update_count = _require_counter(
            record.get("zero_update_count"),
            label=f"{load_id}.zero_update_count",
        )
        if nonzero_update_count + zero_update_count != update_count:
            raise MechanismValidationError(
                "WRENCH_AUDIT_COUNTER_INVALID",
                "saved wrench zero/nonzero counters do not total update_count",
            )
        if record.get("released") is not _is_zero(last_vector):
            raise MechanismValidationError(
                "WRENCH_AUDIT_RECORDS_INVALID",
                "saved wrench released state differs from its final vector",
            )
        if (
            record.get("release_required") is True
            and not _is_zero(last_vector)
        ):
            raise MechanismValidationError(
                "WRENCH_RELEASE_RESIDUAL",
                "saved release-required wrench retains a nonzero value",
            )
        if policy.capability_required and registration.get("status") != "ACTIVE":
            raise MechanismValidationError(
                "WRENCH_CAPABILITY_NOT_ACTIVE",
                "saved analytical transmission is not ACTIVE",
            )
        if (
            policy.component_provenance_required
            and (
                nonzero_update_count > 0
                or not _is_zero(initial_vector)
            )
            and _require_counter(
                record.get("component_event_count"),
                label=f"{load_id}.component_event_count",
            )
            == 0
        ):
            raise MechanismValidationError(
                "WRENCH_COMPONENT_PROVENANCE_MISSING",
                "saved aggregated wrench lacks component provenance",
            )
        _require_counter(
            record.get("component_event_count"),
            label=f"{load_id}.component_event_count",
        )
        sources = record.get("component_sources")
        if (
            not isinstance(sources, list)
            or any(not isinstance(item, str) or not item for item in sources)
            or sources != sorted(set(sources))
        ):
            raise MechanismValidationError(
                "WRENCH_AUDIT_RECORDS_INVALID",
                "saved wrench component sources are not a canonical inventory",
            )
        phase_updates = record.get("phase_update_counts")
        phase_nonzero = record.get("phase_nonzero_update_counts")
        phase_zero = record.get("phase_zero_update_counts")
        if not all(
            isinstance(item, Mapping)
            for item in (phase_updates, phase_nonzero, phase_zero)
        ):
            raise MechanismValidationError(
                "WRENCH_PHASE_AUDIT_INVALID",
                "saved wrench record lacks phase-aware update counts",
            )
        phase_update_keys = {str(item) for item in phase_updates}
        phase_nonzero_keys = {str(item) for item in phase_nonzero}
        phase_zero_keys = {str(item) for item in phase_zero}
        if phase_nonzero_keys | phase_zero_keys != phase_update_keys:
            raise MechanismValidationError(
                "WRENCH_PHASE_AUDIT_INVALID",
                "saved wrench phase buckets do not exactly match update buckets",
            )
        for phase_id, count in phase_updates.items():
            if (
                str(phase_id) not in program_phase_ids
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count <= 0
                or _require_counter(
                    phase_nonzero.get(phase_id, 0),
                    label=f"{load_id}.{phase_id}.nonzero",
                )
                + _require_counter(
                    phase_zero.get(phase_id, 0),
                    label=f"{load_id}.{phase_id}.zero",
                )
                != count
            ):
                raise MechanismValidationError(
                    "WRENCH_PHASE_AUDIT_INVALID",
                    "saved wrench phase counts are inconsistent",
                )
        phase_update_total = sum(int(item) for item in phase_updates.values())
        phase_nonzero_total = sum(
            int(item) for item in phase_nonzero.values()
        )
        phase_zero_total = sum(int(item) for item in phase_zero.values())
        if record.get("task_input") is True:
            if (
                phase_nonzero_total != nonzero_update_count
                or phase_update_total != update_count
                or phase_zero_total != zero_update_count
            ):
                raise MechanismValidationError(
                    "WRENCH_AUDIT_COUNTER_INVALID",
                    "saved task-input global counters do not exactly reconcile "
                    "with their phase buckets",
                )
            observed_nonzero_phases.update(str(item) for item in phase_nonzero)
            observed_zero_phases.update(str(item) for item in phase_zero)
        elif (
            phase_update_total > update_count
            or phase_nonzero_total > nonzero_update_count
            or phase_zero_total > zero_update_count
        ):
            raise MechanismValidationError(
                "WRENCH_AUDIT_COUNTER_INVALID",
                "saved non-task-input phase counters exceed their global "
                "update counters",
            )
    input_events = audit.get("aggregated_task_input_events") or []
    if not isinstance(input_events, list):
        raise MechanismValidationError(
            "WRENCH_INPUT_EVENT_INCOMPLETE",
            "saved aggregated task-input events are not a list",
        )
    input_sources: set[str] = set()
    input_event_bodies: set[str] = set()
    for raw in input_events:
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "WRENCH_INPUT_EVENT_INCOMPLETE",
                "saved aggregated task-input event is not an object",
            )
        _require_exact_keys(
            raw,
            _TASK_INPUT_EVENT_KEYS,
            code="WRENCH_INPUT_EVENT_INCOMPLETE",
            label="saved aggregated task-input event",
        )
        target = str(raw.get("target_body") or "")
        source = str(raw.get("source_role") or "")
        interval = str(raw.get("active_interval") or "")
        phase_id = str(raw.get("phase_id") or "")
        law_id = str(raw.get("law_id") or "")
        if (
            not target
            or not source
            or not interval
            or not phase_id
            or law_id not in _AGGREGATED_TASK_INPUT_LAWS
        ):
            raise MechanismValidationError(
                "WRENCH_INPUT_EVENT_INCOMPLETE",
                "saved aggregated task-input event is incomplete",
            )
        if (
            source != "declared_input_port"
            or interval != f"phase:{phase_id}"
        ):
            raise MechanismValidationError(
                "WRENCH_INPUT_EVENT_INCOMPLETE",
                "saved aggregated task-input event is not bound to the selected "
                "input port and exact phase interval",
            )
        if (
            allow_direct_output_actuation is not True
            and target in declared_outputs
            and target not in declared_inputs
        ):
            raise MechanismValidationError(
                "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
                "saved aggregated task input directly targets the task output",
            )
        if declared_inputs and target not in declared_inputs:
            raise MechanismValidationError(
                "WRENCH_INPUT_BODY_INVALID",
                "saved aggregated task input targets an undeclared body",
            )
        input_event_bodies.add(target)
        input_sources.add(source)
        if phase_id not in program_phase_ids:
            raise MechanismValidationError(
                "WRENCH_PHASE_AUDIT_INVALID",
                "saved task-input event uses an unknown phase",
            )
        observed_nonzero_phases.add(phase_id)
    canonical_events = sorted(
        [dict(item) for item in input_events],
        key=lambda item: (
            item["active_interval"],
            item["target_body"],
            item["source_role"],
            item["law_id"],
        ),
    )
    if input_events != canonical_events or len(
        {
            (
                item["target_body"],
                item["source_role"],
                item["active_interval"],
                item["phase_id"],
                item["law_id"],
            )
            for item in canonical_events
        }
    ) != len(canonical_events):
        raise MechanismValidationError(
            "WRENCH_INPUT_EVENT_INCOMPLETE",
            "saved aggregated task-input events are duplicated or unsorted",
        )
    raw_barriers = audit.get("phase_barriers")
    if (
        not isinstance(raw_barriers, list)
        or len(raw_barriers) != len(program_phase_ids)
    ):
        raise MechanismValidationError(
            "WRENCH_PHASE_BARRIER_COVERAGE_INCOMPLETE",
            "saved wrench phase barriers do not cover the complete program",
        )
    previous_exit: list[dict[str, Any]] | None = None
    first_entry: list[dict[str, Any]] | None = None
    last_exit: list[dict[str, Any]] | None = None
    program_kind = {
        str(item["phase_id"]): str(item.get("wrench_kind") or "")
        for item in program_phases
    }
    for index, raw in enumerate(raw_barriers):
        if not isinstance(raw, Mapping):
            raise MechanismValidationError(
                "WRENCH_PHASE_BARRIER_INVALID",
                "saved wrench phase barrier is not an object",
            )
        barrier = dict(raw)
        _require_exact_keys(
            barrier,
            _PHASE_BARRIER_KEYS,
            code="WRENCH_PHASE_BARRIER_INVALID",
            label="saved wrench phase barrier",
        )
        expected_barrier_digest = barrier.get("barrier_sha256")
        actual_barrier_digest = canonical_sha256(
            {
                key: item
                for key, item in barrier.items()
                if key != "barrier_sha256"
            }
        )
        phase_id = str(barrier.get("phase_id") or "")
        if (
            expected_barrier_digest != actual_barrier_digest
            or phase_id != program_phase_ids[index]
            or barrier.get("wrench_kind") != program_kind[phase_id]
        ):
            raise MechanismValidationError(
                "WRENCH_PHASE_BARRIER_INVALID",
                "saved wrench phase barrier hash or phase binding is invalid",
            )
        entry_state = _validated_state_inventory(
            barrier.get("entry_task_input_state"),
            records_by_id=records_by_id,
            label=f"{phase_id} entry",
        )
        exit_state = _validated_state_inventory(
            barrier.get("exit_task_input_state"),
            records_by_id=records_by_id,
            label=f"{phase_id} exit",
        )
        entry_all_zero = WrenchLedger._state_is_zero(entry_state)
        exit_all_zero = WrenchLedger._state_is_zero(exit_state)
        if (
            barrier.get("entry_all_zero") is not entry_all_zero
            or barrier.get("exit_all_zero") is not exit_all_zero
            or (previous_exit is not None and entry_state != previous_exit)
        ):
            raise MechanismValidationError(
                "WRENCH_PHASE_BARRIER_INVALID",
                "saved wrench phase state is discontinuous or mislabelled",
            )
        if first_entry is None:
            first_entry = entry_state
        last_exit = exit_state
        previous_exit = exit_state
        update_count = _require_counter(
            barrier.get("task_input_update_count"),
            label=f"{phase_id}.task_input_update_count",
        )
        nonzero_count = _require_counter(
            barrier.get("task_input_nonzero_update_count"),
            label=f"{phase_id}.task_input_nonzero_update_count",
        )
        zero_count = _require_counter(
            barrier.get("task_input_zero_update_count"),
            label=f"{phase_id}.task_input_zero_update_count",
        )
        event_count = _require_counter(
            barrier.get("aggregated_task_input_event_count"),
            label=f"{phase_id}.aggregated_task_input_event_count",
        )
        expected_update_count = sum(
            int(record["phase_update_counts"].get(phase_id, 0))
            for record in records_by_id.values()
            if record["task_input"] is True
        )
        expected_nonzero_count = sum(
            int(record["phase_nonzero_update_counts"].get(phase_id, 0))
            for record in records_by_id.values()
            if record["task_input"] is True
        )
        expected_zero_count = sum(
            int(record["phase_zero_update_counts"].get(phase_id, 0))
            for record in records_by_id.values()
            if record["task_input"] is True
        )
        expected_event_count = sum(
            1 for item in canonical_events if item["phase_id"] == phase_id
        )
        if (
            update_count != nonzero_count + zero_count
            or update_count != expected_update_count
            or nonzero_count != expected_nonzero_count
            or zero_count != expected_zero_count
            or event_count != expected_event_count
            or (
                program_kind[phase_id] == "none"
                and (
                    not entry_all_zero
                    or not exit_all_zero
                    or nonzero_count != 0
                    or event_count != 0
                )
            )
        ):
            raise MechanismValidationError(
                "WRENCH_PHASE_BARRIER_INVALID",
                "saved wrench phase barrier counters or zero policy are invalid",
            )
    task_records = [
        record
        for record in records_by_id.values()
        if record["task_input"] is True
    ]
    expected_initial_state = [
        {
            "load_id": record["load_id"],
            "target_body": record["target_body"],
            "policy_id": record["policy_id"],
            "vector_si": list(_finite_vector(record["initial_vector_si"])),
        }
        for record in sorted(task_records, key=lambda item: item["load_id"])
    ]
    expected_final_state = [
        {
            "load_id": record["load_id"],
            "target_body": record["target_body"],
            "policy_id": record["policy_id"],
            "vector_si": list(_finite_vector(record["last_vector_si"])),
        }
        for record in sorted(task_records, key=lambda item: item["load_id"])
    ]
    if first_entry != expected_initial_state or last_exit != expected_final_state:
        raise MechanismValidationError(
            "WRENCH_PHASE_BARRIER_INVALID",
            "saved wrench phase barriers do not span the complete persistent state",
        )
    if len(input_sources) > 1 and not allow_multiple_task_input_channels:
        raise MechanismValidationError(
            "WRENCH_MULTIPLE_TASK_INPUT_CHANNELS",
            "saved wrench audit contains multiple task-input channels",
        )
    observed_bodies = {
        str(item) for item in audit.get("task_input_bodies_observed") or ()
    }
    if observed_bodies != input_event_bodies | {
        str(record.get("target_body"))
        for record in records
        if isinstance(record, Mapping)
        and record.get("task_input") is True
        and int(record.get("nonzero_update_count") or 0) > 0
    }:
        raise MechanismValidationError(
            "WRENCH_PHASE_AUDIT_INVALID",
            "saved task-input body summary differs from its records",
        )
    selected_input_bodies = (
        {str(item) for item in expected_task_input_body_ids}
        if expected_task_input_body_ids is not None
        else (
            {expected_task_input_body_id}
            if expected_task_input_body_id is not None
            else None
        )
    )
    input_body_mismatch = False
    if selected_input_bodies is not None:
        if allow_multiple_task_input_channels:
            input_body_mismatch = (
                not selected_input_bodies.issubset(observed_bodies)
                or not observed_bodies.issubset(declared_inputs)
            )
        else:
            input_body_mismatch = observed_bodies != selected_input_bodies
    if input_body_mismatch:
        raise MechanismValidationError(
            "WRENCH_TASK_INPUT_BODY_MISMATCH",
            "observed task input does not include the semantically selected "
            "body within the registered composite input inventory",
        )
    # 2026-08-09 ATTENTION AI EDITORS: direct output actuation policy is
    # declared per requirement in the acceptance contract (the REQUIRED
    # allow_direct_output_actuation field) and consumed here. Enforcement
    # means "anything except explicit True". If the contract semantics
    # change, update this gate, the two gates above, AND
    # task_motion_evidence.resolve_and_validate_task_channels together.
    if (
        allow_direct_output_actuation is not True
        and expected_task_output_body_id is not None
        and expected_task_output_body_id in observed_bodies
    ):
        raise MechanismValidationError(
            "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
            "observed task input directly targets the selected task output",
        )
    if set(audit.get("task_input_nonzero_phase_ids") or ()) != (
        observed_nonzero_phases
    ):
        raise MechanismValidationError(
            "WRENCH_PHASE_AUDIT_INVALID",
            "saved nonzero task-input phase summary differs from its records",
        )
    if set(audit.get("task_input_zero_phase_ids") or ()) != observed_zero_phases:
        raise MechanismValidationError(
            "WRENCH_PHASE_AUDIT_INVALID",
            "saved zero task-input phase summary differs from its records",
        )
    observed_sources = {
        str(record.get("source_role"))
        for record in records
        if isinstance(record, Mapping)
        and record.get("task_input") is True
        and int(record.get("nonzero_update_count") or 0) > 0
    } | input_sources
    if (
        audit.get("task_input_sources_observed")
        != sorted(observed_sources)
        or audit.get("task_input_bodies_observed")
        != sorted(observed_bodies)
        or audit.get("task_input_nonzero_phase_ids")
        != sorted(observed_nonzero_phases)
        or audit.get("task_input_zero_phase_ids")
        != sorted(observed_zero_phases)
    ):
        raise MechanismValidationError(
            "WRENCH_PHASE_AUDIT_INVALID",
            "saved wrench summaries are not canonical exact inventories",
        )
    exact_counts = {
        "task_input_load_count": sum(
            1 for item in records if item.get("task_input") is True
        ),
        "transmission_load_count": sum(
            1 for item in records if item.get("transmission") is True
        ),
        "support_load_count": sum(
            1 for item in records if item.get("support") is True
        ),
    }
    for field, expected_count in exact_counts.items():
        if _require_counter(audit.get(field), label=field) != expected_count:
            raise MechanismValidationError(
                "WRENCH_AUDIT_COUNTER_INVALID",
                f"saved {field} differs from the record inventory",
            )
    if return_input_policy == "zero" and observed_nonzero_phases & {
        str(item) for item in return_phase_ids
    }:
        raise MechanismValidationError(
            "WRENCH_RETURN_PHASE_TASK_INPUT_NONZERO",
            "passive return contains a nonzero task-input update",
        )
    if not set(str(item) for item in powered_phase_ids).issubset(
        observed_nonzero_phases
    ):
        raise MechanismValidationError(
            "WRENCH_POWERED_PHASE_INPUT_COVERAGE_MISSING",
            "a designated powered phase has no audited nonzero task input",
        )
    phase_kind = {
        str(item.get("phase_id") or ""): str(item.get("wrench_kind") or "")
        for item in program_phases
        if isinstance(item, Mapping)
    }
    if return_input_policy == "selected_input_only":
        active_return = {
            str(item)
            for item in return_phase_ids
            if phase_kind.get(str(item)) != "none"
        }
        if not active_return.issubset(observed_nonzero_phases):
            raise MechanismValidationError(
                "WRENCH_RETURN_PHASE_INPUT_COVERAGE_MISSING",
                "an active return phase has no audited selected-input update",
            )
    if return_input_policy not in {None, "zero", "selected_input_only"}:
        raise MechanismValidationError(
            "WRENCH_RETURN_INPUT_POLICY_INVALID",
            "task acceptance supplied an invalid return input policy",
        )
    if (
        audit.get("direct_output_drive_detected") is not False
        or _require_counter(
            audit.get("release_residual_count"),
            label="release_residual_count",
        )
        != 0
    ):
        raise MechanismValidationError(
            "WRENCH_AUDIT_FAILED",
            "saved wrench audit reports an unsafe drive or release residual",
        )
    return audit


def create_body_force(
    chrono: Any,
    body: Any,
    force: Any,
    force_local: bool,
    application_point: Any,
    application_point_local: bool,
    *,
    declaration: WrenchDeclaration,
) -> LedgeredBodyWrench:
    ledger = current_wrench_ledger()
    ledger.register(
        declaration,
        wrench_kind="force",
        initial_vector=force,
    )
    native_load = chrono.ChLoadBodyForce(
        body,
        force,
        force_local,
        application_point,
        application_point_local,
    )
    return LedgeredBodyWrench(
        native_load,
        ledger,
        declaration.load_id,
        "force",
    )


def create_body_torque(
    chrono: Any,
    body: Any,
    torque: Any,
    torque_local: bool,
    *,
    declaration: WrenchDeclaration,
) -> LedgeredBodyWrench:
    ledger = current_wrench_ledger()
    ledger.register(
        declaration,
        wrench_kind="torque",
        initial_vector=torque,
    )
    native_load = chrono.ChLoadBodyTorque(
        body,
        torque,
        torque_local,
    )
    return LedgeredBodyWrench(
        native_load,
        ledger,
        declaration.load_id,
        "torque",
    )


def add_ledgered_load(container: Any, load: LedgeredBodyWrench) -> None:
    if not isinstance(load, LedgeredBodyWrench):
        raise MechanismValidationError(
            "WRENCH_LOAD_NOT_LEDGERED",
            "body force/torque must be added through a ledgered handle",
        )
    container.Add(load.native_load)
