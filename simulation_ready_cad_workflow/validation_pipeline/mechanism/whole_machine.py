from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from ..runtime import JobKind, JobSpec, QueueManifest, QueuePolicy, ResourcePolicy
from .adapters import adapter_from_dict, adapter_to_dict, get_adapter
from .family_plugins import mechanism_adapter_for, plugin_registration_or_none
from .io import (
    canonical_sha256,
    ensure_inside,
    MECHANISM_SOURCE_SHA256_ENV,
    mechanism_source_sha256,
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    validator_import_root,
    write_json_atomic,
)
from .launcher import PyChronoLauncher, select_pychrono_launcher
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import invoke_plugin_callback, plugin_boundary_error


def _whole_machine_harness_error(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
) -> MechanismValidationError:
    return MechanismValidationError(
        code,
        message,
        details=details,
        failure=make_failure_envelope(
            code,
            FailureOwner.HARNESS_CODE,
            source_component="mechanism.whole_machine.evidence",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            details=details,
        ),
    )


def mechanism_load_cases_for_whole_machine(
    plugin: Any,
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    """Read a plugin's certified load inventory through the owner boundary."""

    registration = plugin_registration_or_none(str(plugin.plugin_id))
    load_cases = invoke_plugin_callback(
        lambda: plugin.load_cases(plan),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="load_cases",
        source_component="mechanism.whole_machine.load_cases",
        unexpected_code="MECHANISM_ADAPTER_LOAD_CASE_FAILED",
        details={"capsule_id": plan.capsule_id},
    )
    if not isinstance(load_cases, tuple) or any(
        not isinstance(item, LoadCase) for item in load_cases
    ):
        message = "mechanism adapter load_cases() must return LoadCase values"
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="load_cases",
            source_component="mechanism.whole_machine.load_cases",
            code="MECHANISM_ADAPTER_LOAD_CASE_INVALID",
            message=message,
            details={"capsule_id": plan.capsule_id},
            contract_violation=True,
        )
    return load_cases


@dataclass(frozen=True)
class WholeMachineExecutionPlan:
    schema_version: str
    package_binding_sha256: str
    capsule_id: str
    certificate_file: str
    certificate_sha256: str
    replacement_file: str
    replacement_sha256: str
    adapter_id: str
    mechanism_plugin_id: str
    worker_backend: str
    task_ids: tuple[str, str]
    axis_abs: tuple[float, float, float]
    joint_origin_abs_m: tuple[float, float, float]
    gravity_abs_m_s2: tuple[float, float, float]
    body_origin_abs_m: tuple[float, float, float]
    input_point_abs_m: tuple[float, float, float]
    moving_extents_m: tuple[float, float, float]
    moving_mass_kg: float
    outbound_progress_m: float
    home_progress_m: float
    max_control_force_n: float
    position_tolerance_m: float = 5e-4
    speed_tolerance_m_s: float = 5e-3
    settle_hold_s: float = 0.05
    max_stage_duration_s: float = 3.0
    step_s: float = 0.001
    sample_interval_s: float = 0.01
    controller_kp_n_m: float = 12.0
    controller_kd_n_s_m: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "task_ids",
            "axis_abs",
            "joint_origin_abs_m",
            "gravity_abs_m_s2",
            "body_origin_abs_m",
            "input_point_abs_m",
            "moving_extents_m",
        ):
            value[field] = list(value[field])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WholeMachineExecutionPlan:
        data = dict(value)
        for field in (
            "task_ids",
            "axis_abs",
            "joint_origin_abs_m",
            "gravity_abs_m_s2",
            "body_origin_abs_m",
            "input_point_abs_m",
            "moving_extents_m",
        ):
            data[field] = tuple(data[field])
        plan = cls(**data)
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.schema_version != "0.1":
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID", "schema_version must be 0.1"
            )
        positive = {
            "moving_mass_kg": self.moving_mass_kg,
            "outbound_progress_m": self.outbound_progress_m,
            "max_control_force_n": self.max_control_force_n,
            "position_tolerance_m": self.position_tolerance_m,
            "speed_tolerance_m_s": self.speed_tolerance_m_s,
            "settle_hold_s": self.settle_hold_s,
            "max_stage_duration_s": self.max_stage_duration_s,
            "step_s": self.step_s,
            "sample_interval_s": self.sample_interval_s,
            "controller_kp_n_m": self.controller_kp_n_m,
            "controller_kd_n_s_m": self.controller_kd_n_s_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise MechanismValidationError(
                    "WHOLE_MACHINE_PLAN_VALUE_INVALID", f"{name} must be positive"
                )
        if self.sample_interval_s < self.step_s:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "sample_interval_s cannot be lower than step_s",
            )


@dataclass(frozen=True)
class WholeMachineRevoluteExecutionPlan:
    schema_version: str
    coordinate_kind: str
    package_binding_sha256: str
    capsule_id: str
    certificate_file: str
    certificate_sha256: str
    replacement_file: str
    replacement_sha256: str
    adapter_id: str
    mechanism_plugin_id: str
    worker_backend: str
    task_ids: tuple[str, str]
    axis_abs: tuple[float, float, float]
    joint_origin_abs_m: tuple[float, float, float]
    gravity_abs_m_s2: tuple[float, float, float]
    body_origin_abs_m: tuple[float, float, float]
    input_point_abs_m: tuple[float, float, float]
    moving_extents_m: tuple[float, float, float]
    moving_mass_kg: float
    outbound_angle_rad: float
    home_angle_rad: float
    max_control_torque_nm: float
    passive_return_stiffness_nm_rad: float = 0.0
    passive_return_damping_nm_s_rad: float = 0.0
    passive_return_required: bool = False
    angle_tolerance_rad: float = math.radians(0.5)
    angular_speed_tolerance_rad_s: float = 0.02
    settle_hold_s: float = 0.05
    max_stage_duration_s: float = 4.0
    step_s: float = 0.001
    kinematic_pivot_tolerance_m: float = 1e-5
    kinematic_off_axis_tolerance_rad: float = 1e-6
    sample_interval_s: float = 0.01
    controller_kp_nm_rad: float = 2.5
    controller_kd_nm_s_rad: float = 0.35

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "task_ids",
            "axis_abs",
            "joint_origin_abs_m",
            "gravity_abs_m_s2",
            "body_origin_abs_m",
            "input_point_abs_m",
            "moving_extents_m",
        ):
            value[field] = list(value[field])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WholeMachineRevoluteExecutionPlan:
        data = dict(value)
        for field in (
            "task_ids",
            "axis_abs",
            "joint_origin_abs_m",
            "gravity_abs_m_s2",
            "body_origin_abs_m",
            "input_point_abs_m",
            "moving_extents_m",
        ):
            data[field] = tuple(data[field])
        plan = cls(**data)
        plan.validate()
        return plan

    def validate(self) -> None:
        if self.schema_version != "0.1" or self.coordinate_kind != "rotation":
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "revolute whole-machine plan must use schema 0.1 and rotation",
            )
        positive = {
            "moving_mass_kg": self.moving_mass_kg,
            "outbound_angle_rad": self.outbound_angle_rad,
            "max_control_torque_nm": self.max_control_torque_nm,
            "angle_tolerance_rad": self.angle_tolerance_rad,
            "angular_speed_tolerance_rad_s": self.angular_speed_tolerance_rad_s,
            "settle_hold_s": self.settle_hold_s,
            "max_stage_duration_s": self.max_stage_duration_s,
            "step_s": self.step_s,
            "kinematic_pivot_tolerance_m": self.kinematic_pivot_tolerance_m,
            "kinematic_off_axis_tolerance_rad": self.kinematic_off_axis_tolerance_rad,
            "sample_interval_s": self.sample_interval_s,
            "controller_kp_nm_rad": self.controller_kp_nm_rad,
            "controller_kd_nm_s_rad": self.controller_kd_nm_s_rad,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise MechanismValidationError(
                    "WHOLE_MACHINE_PLAN_VALUE_INVALID", f"{name} must be positive"
                )
        if self.sample_interval_s < self.step_s:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "sample_interval_s cannot be lower than step_s",
            )
        if not math.isfinite(self.home_angle_rad):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID", "home_angle_rad must be finite"
            )
        for name, value in {
            "passive_return_stiffness_nm_rad": self.passive_return_stiffness_nm_rad,
            "passive_return_damping_nm_s_rad": self.passive_return_damping_nm_s_rad,
        }.items():
            if not math.isfinite(value) or value < 0.0:
                raise MechanismValidationError(
                    "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                    f"{name} must be finite and non-negative",
                )
        if self.passive_return_required and self.passive_return_stiffness_nm_rad <= 0.0:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "passive spring return requires positive stiffness",
            )



@dataclass(frozen=True)
class WholeMachineCompoundExecutionPlan:
    schema_version: str
    coordinate_kind: str
    package_binding_sha256: str
    capsule_id: str
    certificate_file: str
    certificate_sha256: str
    replacement_file: str
    replacement_sha256: str
    adapter_id: str
    mechanism_plugin_id: str
    worker_backend: str
    task_ids: tuple[str, str]
    mechanism_plan: dict[str, Any]
    adapter_spec: dict[str, Any]
    task_load_case: dict[str, Any]
    reference_trace_file: str
    reference_trace_sha256: str
    outbound_angle_rad: float
    home_angle_rad: float
    max_control_force_n: float
    angle_tolerance_rad: float
    kinematic_joint_drift_tolerance_m: float = 1e-5
    kinematic_off_axis_tolerance_rad: float = 1e-6
    max_stage_duration_s: float = 1.0
    step_s: float = 0.0001
    sample_interval_s: float = 0.005
    control_wrench_kind: str = "force"
    max_control_torque_nm: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_ids"] = list(self.task_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WholeMachineCompoundExecutionPlan":
        data = dict(value)
        data["task_ids"] = tuple(data["task_ids"])
        plan = cls(**data)
        plan.validate()
        return plan

    def validate(self) -> None:
        from .models import LoadCase

        if self.schema_version != "0.1" or self.coordinate_kind != "compound_output_rotation":
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "compound whole-machine plan must use schema 0.1 and compound output rotation",
            )
        mechanism = MechanismPlan.from_dict(self.mechanism_plan)
        load_case = LoadCase.from_dict(self.task_load_case)
        if (
            mechanism.family != "compound"
            or mechanism.package_binding_sha256 != self.package_binding_sha256
            or mechanism.capsule_id != self.capsule_id
            or self.worker_backend != "revolute_linkage_v1"
            or load_case.family != "compound"
            or load_case.program != "task_output_angle_roundtrip"
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "compound mechanism identity or task load case is inconsistent",
            )
        if self.control_wrench_kind not in {"force", "torque"}:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "compound control wrench must be force or torque",
            )
        positive = {
            "outbound_angle_rad": self.outbound_angle_rad,
            (
                "max_control_torque_nm"
                if self.control_wrench_kind == "torque"
                else "max_control_force_n"
            ): (
                self.max_control_torque_nm
                if self.control_wrench_kind == "torque"
                else self.max_control_force_n
            ),
            "angle_tolerance_rad": self.angle_tolerance_rad,
            "kinematic_joint_drift_tolerance_m": self.kinematic_joint_drift_tolerance_m,
            "kinematic_off_axis_tolerance_rad": self.kinematic_off_axis_tolerance_rad,
            "max_stage_duration_s": self.max_stage_duration_s,
            "step_s": self.step_s,
            "sample_interval_s": self.sample_interval_s,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise MechanismValidationError(
                    "WHOLE_MACHINE_PLAN_VALUE_INVALID", f"{name} must be positive"
                )
        if not math.isfinite(self.home_angle_rad) or self.sample_interval_s < self.step_s:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "compound home angle or sample interval is invalid",
            )
        if not set(self.task_ids).issubset(mechanism.task_ids):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
                "compound whole-machine tasks are not covered by the mechanism plan",
            )


@dataclass(frozen=True)
class WholeMachineClockExecutionPlan:
    schema_version: str
    coordinate_kind: str
    package_binding_sha256: str
    capsule_id: str
    certificate_file: str
    certificate_sha256: str
    replacement_file: str
    replacement_sha256: str
    adapter_id: str
    mechanism_plugin_id: str
    worker_backend: str
    task_ids: tuple[str, ...]
    mechanism_plan: dict[str, Any]
    adapter_spec: dict[str, Any]
    task_load_cases: tuple[dict[str, Any], ...]
    reference_traces: tuple[dict[str, str], ...]
    outbound_angle_rad: float
    angle_tolerance_rad: float
    expected_hour_to_minute_ratio: float
    expected_second_to_minute_ratio: float
    hand_ratio_relative_tolerance: float
    minimum_alarm_swing_rad: float
    max_stage_duration_s: float
    step_s: float
    minimum_manual_setting_travel_rad: float = 0.02

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_ids"] = list(self.task_ids)
        value["task_load_cases"] = list(self.task_load_cases)
        value["reference_traces"] = list(self.reference_traces)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WholeMachineClockExecutionPlan":
        data = dict(value)
        data["task_ids"] = tuple(data["task_ids"])
        data["task_load_cases"] = tuple(data["task_load_cases"])
        data["reference_traces"] = tuple(data["reference_traces"])
        plan = cls(**data)
        plan.validate()
        return plan

    def validate(self) -> None:
        from .adapters import adapter_from_dict
        from .clock_gear_train_spec import clock_gear_train_from_dict
        from .models import LoadCase

        if self.schema_version != "0.1" or self.coordinate_kind != "clock_multifunction":
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "clock whole-machine plan must use schema 0.1 and clock_multifunction",
            )
        mechanism = MechanismPlan.from_dict(self.mechanism_plan)
        cases = tuple(LoadCase.from_dict(item) for item in self.task_load_cases)
        programs = {item.program for item in cases}
        clock = clock_gear_train_from_dict(
            adapter_from_dict(self.adapter_spec).family_spec
        )
        required_programs = {
            "task_clock_cycle",
            "manual_time_setting_clock",
            "alarm_clock_cycle",
        }
        required_tasks = {
            "run_three_hand_clock",
            "verify_three_hand_ratio",
            "set_time_manually",
            "ring_twin_bell_alarm",
        }
        if clock.alarm_setting_clutch is not None:
            required_programs.update(
                {
                    "hour_hand_alarm_setting_clock",
                    "scheduled_alarm_clock",
                }
            )
            required_tasks = {
                "run_three_hand_clock_logic",
                "set_time_manually",
                "ring_alarm_independently",
                "set_alarm_time_by_turning_hour_hand",
                "trigger_alarm_at_selected_hand_position",
            }
        elif clock.scheduled_alarm is not None:
            required_programs.update(
                {"manual_zero_and_resume_clock", "scheduled_alarm_clock"}
            )
            required_tasks.update(
                {"zero_time_and_resume", "alarm_after_one_minute"}
            )
        if (
            mechanism.family != "compound"
            or mechanism.package_binding_sha256 != self.package_binding_sha256
            or mechanism.capsule_id != self.capsule_id
            or self.worker_backend != "spur_gear_clock_v1"
            or programs != required_programs
            or len(self.reference_traces) != len(cases)
            or not required_tasks.issubset(self.task_ids)
            or not set(self.task_ids).issubset(mechanism.task_ids)
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "clock mechanism identity, task set, or task load cases are inconsistent",
            )
        positive = {
            "outbound_angle_rad": self.outbound_angle_rad,
            "angle_tolerance_rad": self.angle_tolerance_rad,
            "expected_hour_to_minute_ratio": self.expected_hour_to_minute_ratio,
            "expected_second_to_minute_ratio": self.expected_second_to_minute_ratio,
            "hand_ratio_relative_tolerance": self.hand_ratio_relative_tolerance,
            "minimum_alarm_swing_rad": self.minimum_alarm_swing_rad,
            "max_stage_duration_s": self.max_stage_duration_s,
            "step_s": self.step_s,
            "minimum_manual_setting_travel_rad": self.minimum_manual_setting_travel_rad,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "clock whole-machine thresholds must be positive and finite",
            )


def build_execution_plan(
    plan: MechanismPlan,
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    certificate_file: str | Path,
    replacement_file: str | Path,
) -> WholeMachineExecutionPlan | WholeMachineRevoluteExecutionPlan | WholeMachineCompoundExecutionPlan | WholeMachineClockExecutionPlan:
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    certificate_path = Path(certificate_file).resolve()
    replacement_path = Path(replacement_file).resolve()
    try:
        certificate_path.relative_to(workspace)
        replacement_path.relative_to(workspace)
        output.relative_to(workspace)
        certificate = read_json(certificate_path)
        replacement = read_json(replacement_path)
        certificate_sha256 = sha256_file(certificate_path)
        replacement_sha256 = sha256_file(replacement_path)
    except (OSError, TypeError, ValueError) as exc:
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_EVIDENCE_INVALID",
            f"whole-machine evidence cannot be read or bound: {exc}",
            details={"capsule_id": plan.capsule_id},
        ) from exc
    if certificate.get("verdict") != "KINEMATIC_CERTIFIED":
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_CERTIFICATE_REQUIRED",
            "whole-machine replacement requires KINEMATIC_CERTIFIED evidence",
            details={"capsule_id": plan.capsule_id},
        )
    if certificate.get("package", {}).get("package_binding_sha256") != plan.package_binding_sha256:
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_CERTIFICATE_MISMATCH",
            "certificate belongs to a different design package",
            details={"capsule_id": plan.capsule_id},
        )
    if replacement.get("status") != "ACTIVE":
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_REPLACEMENT_INACTIVE",
            "replacement is not active",
            details={"capsule_id": plan.capsule_id},
        )
    expected_replacement = {
        "certificate_sha256": certificate_sha256,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
    }
    mismatches = [
        key for key, value in expected_replacement.items() if replacement.get(key) != value
    ]
    if mismatches:
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_REPLACEMENT_MISMATCH",
            "replacement binding mismatch: " + ", ".join(mismatches),
            details={
                "capsule_id": plan.capsule_id,
                "mismatches": mismatches,
            },
        )
    model = replacement.get("replacement_model", {})
    if (
        model.get("family") != plan.family
        or model.get("fixed_group_ids") != list(plan.fixed_group_ids)
        or model.get("moving_group_ids") != list(plan.moving_group_ids)
        or model.get("axis_abs") != list(plan.axis_abs)
        or model.get("origin_abs_m") != list(plan.joint_origin_abs_m)
    ):
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "replacement model differs from the certified mechanism plan",
            details={"capsule_id": plan.capsule_id},
        )
    if plan.adapter_id is None:
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_ADAPTER_REQUIRED",
            "mechanism has no whole-machine adapter",
            details={"capsule_id": plan.capsule_id},
        )
    try:
        adapter = (
            adapter_from_dict(plan.adapter_spec)
            if plan.adapter_spec is not None
            else get_adapter(plan.adapter_id)
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _whole_machine_harness_error(
            "WHOLE_MACHINE_ADAPTER_BINDING_INVALID",
            f"whole-machine adapter binding is invalid: {exc}",
            details={"capsule_id": plan.capsule_id},
        ) from exc
    plugin = mechanism_adapter_for(plan, adapter)
    registration = plugin_registration_or_none(str(plugin.plugin_id))
    family_plan = invoke_plugin_callback(
        lambda: plugin.build_whole_machine_execution_plan(
            plan=plan,
            workspace=workspace,
            output_dir=output,
            certificate_path=certificate_path,
            certificate_sha256=certificate_sha256,
            replacement_path=replacement_path,
            replacement_sha256=replacement_sha256,
            certificate=certificate,
            replacement_model=model,
            adapter=adapter,
        ),
        registration=registration,
        plugin_id=str(plugin.plugin_id),
        plugin_kind="mechanism",
        callback="build_whole_machine_execution_plan",
        source_component="mechanism.whole_machine.build_execution_plan",
        unexpected_code="MECHANISM_ADAPTER_WHOLE_MACHINE_PLAN_FAILED",
        details={"capsule_id": plan.capsule_id},
    )
    if family_plan is None:
        message = (
            f"whole-machine backend {plugin.whole_machine_backend!r} "
            "is not implemented"
        )
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="build_whole_machine_execution_plan",
            source_component="mechanism.whole_machine.build_execution_plan",
            code="WHOLE_MACHINE_BACKEND_UNAVAILABLE",
            message=message,
            details={"capsule_id": plan.capsule_id},
            contract_violation=True,
        )
    return family_plan


def build_compound_execution_plan(
    *,
    plan: MechanismPlan,
    workspace: Path,
    certificate_path: Path,
    certificate_sha256: str,
    replacement_path: Path,
    replacement_sha256: str,
    certificate: dict[str, Any],
    replacement_model: dict[str, Any],
    adapter: Any,
    plugin: Any,
) -> WholeMachineCompoundExecutionPlan:
    """Generic compound-linkage whole-machine plan (revolute_linkage family)."""

    from .compound_spec import compound_linkage_from_dict

    linkage = compound_linkage_from_dict(adapter.family_spec)
    topology = replacement_model.get("compound_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("family_spec_sha256") != canonical_sha256(linkage.to_dict())
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "compound replacement topology differs from the certified family spec",
        )
    if len(plan.task_ids) < 2:
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
            "compound mechanism must map outbound and return tasks",
        )
    envelope = certificate.get("tested_envelope", {})
    mass = envelope.get("moving_mass_kg", {})
    force = envelope.get("applied_force_n", {})
    torque = envelope.get("applied_torque_nm", {})
    angle = envelope.get("generalized_output_angle_rad")
    actuation_envelope = (
        torque if linkage.input_wrench_kind == "torque" else force
    )
    required_actuation = max(
        linkage.nominal_input_magnitude_si,
        linkage.return_input_magnitude_si,
    )
    moving_mass = sum(body.mass_kg for body in linkage.bodies if not body.fixed)
    tolerance = linkage.output_angle_tolerance_rad
    if (
        not isinstance(mass, dict)
        or not float(mass.get("min", math.inf)) <= moving_mass
        <= float(mass.get("max", -math.inf))
        or not isinstance(actuation_envelope, dict)
        or float(actuation_envelope.get("max", -math.inf))
        < required_actuation
        or not isinstance(angle, dict)
        or float(angle.get("min", math.inf)) > linkage.home_output_angle_rad + tolerance
        or float(angle.get("max", -math.inf)) < linkage.outbound_output_angle_rad - tolerance
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
            "compound machine task mass, actuation, or output angle exceeds the certified envelope",
        )
    task_load = next(
        case for case in mechanism_load_cases_for_whole_machine(plugin, plan)
        if case.program == "task_output_angle_roundtrip"
    )
    reference_path = None
    trace_hashes = certificate.get("trace_file_hashes", {})
    if isinstance(trace_hashes, dict):
        for candidate, expected_hash in trace_hashes.items():
            candidate_path = Path(candidate).resolve()
            try:
                candidate_path.relative_to(workspace)
            except ValueError:
                continue
            if not candidate_path.is_file() or sha256_file(candidate_path) != expected_hash:
                continue
            document = read_trace_json(candidate_path)
            if (
                document.get("branch") == "physical"
                and document.get("load_case_id") == task_load.load_case_id
                and document.get("package_binding_sha256") == plan.package_binding_sha256
            ):
                reference_path = candidate_path
                break
    if reference_path is None:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
            "compound whole-machine replay requires the certified physical task trace",
        )
    result = WholeMachineCompoundExecutionPlan(
        schema_version="0.1",
        coordinate_kind="compound_output_rotation",
        package_binding_sha256=plan.package_binding_sha256,
        capsule_id=plan.capsule_id,
        certificate_file=relative_to_root(workspace, certificate_path),
        certificate_sha256=certificate_sha256,
        replacement_file=relative_to_root(workspace, replacement_path),
        replacement_sha256=replacement_sha256,
        adapter_id=adapter.adapter_id,
        mechanism_plugin_id=plugin.plugin_id,
        worker_backend=plugin.whole_machine_backend,
        task_ids=(plan.task_ids[0], plan.task_ids[1]),
        mechanism_plan=plan.to_dict(),
        adapter_spec=adapter_to_dict(adapter),
        task_load_case=task_load.to_dict(),
        reference_trace_file=relative_to_root(workspace, reference_path),
        reference_trace_sha256=sha256_file(reference_path),
        outbound_angle_rad=linkage.outbound_output_angle_rad,
        home_angle_rad=linkage.home_output_angle_rad,
        max_control_force_n=max(
            linkage.nominal_input_force_n, linkage.return_input_force_n
        ),
        angle_tolerance_rad=linkage.output_angle_tolerance_rad,
        control_wrench_kind=linkage.input_wrench_kind,
        max_control_torque_nm=max(
            linkage.nominal_input_torque_nm,
            linkage.return_input_torque_nm,
        ),
        max_stage_duration_s=max(
            linkage.max_force_phase_s, linkage.max_return_phase_s
        ) + linkage.settle_phase_s,
        step_s=task_load.step_s,
        sample_interval_s=task_load.sample_interval_s,
    )
    result.validate()
    return result



def build_plain_execution_plan(
    *,
    plan: MechanismPlan,
    workspace: Path,
    certificate_path: Path,
    certificate_sha256: str,
    replacement_path: Path,
    replacement_sha256: str,
    certificate: dict[str, Any],
    replacement_model: dict[str, Any],
    adapter: Any,
    plugin: Any,
    kind: str,
) -> WholeMachineExecutionPlan | WholeMachineRevoluteExecutionPlan:
    """Plain prismatic/revolute whole-machine plan; ``kind`` is plugin-owned."""

    geometry = adapter.geometry
    if geometry is None:
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_UNAVAILABLE",
            "adapter has no trusted whole-machine task definition",
        )
    if adapter.required_requirements_sha256 != plan.requirements_sha256:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REQUIREMENTS_MISMATCH",
            "trusted task definition targets different external requirements",
        )
    envelope = certificate.get("tested_envelope", {})
    mass = envelope.get("moving_mass_kg", {})
    common = {
        "schema_version": "0.1",
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "certificate_file": relative_to_root(workspace, certificate_path),
        "certificate_sha256": certificate_sha256,
        "replacement_file": relative_to_root(workspace, replacement_path),
        "replacement_sha256": replacement_sha256,
        "adapter_id": adapter.adapter_id,
        "mechanism_plugin_id": plugin.plugin_id,
        "worker_backend": plugin.whole_machine_backend,
        "axis_abs": plan.axis_abs,
        "joint_origin_abs_m": plan.joint_origin_abs_m,
        "gravity_abs_m_s2": plan.gravity_abs_m_s2,
        "body_origin_abs_m": geometry.moving_origin_abs_m,
        "input_point_abs_m": plan.input_point_abs_m,
        "moving_extents_m": plan.moving_extents_m,
    }
    if kind == "prismatic":
        task = adapter.whole_machine_task
        if task is None:
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_UNAVAILABLE",
                "prismatic adapter has no trusted machine task",
            )
        if not {task.outbound_task_id, task.return_task_id}.issubset(plan.task_ids):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
                "certified capsule does not cover both required machine tasks",
            )
        force = envelope.get("applied_force_n", {})
        progress = envelope.get("generalized_progress_m")
        if (
            not isinstance(mass, dict)
            or not float(mass.get("min", math.inf)) <= task.moving_mass_kg
            <= float(mass.get("max", -math.inf))
            or not isinstance(force, dict)
            or float(force.get("max", -math.inf))
            < max(task.outbound_force_n, task.return_force_n)
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
                "machine task mass or force exceeds the certified envelope",
            )
        tolerance = 5e-4
        if (
            not isinstance(progress, dict)
            or float(progress.get("min", math.inf)) > task.home_progress_m + tolerance
            or float(progress.get("max", -math.inf))
            < task.outbound_progress_m - tolerance
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
                "machine task travel exceeds observed physical-branch progress",
            )
        result: WholeMachineExecutionPlan | WholeMachineRevoluteExecutionPlan = (
            WholeMachineExecutionPlan(
                **common,
                task_ids=(task.outbound_task_id, task.return_task_id),
                moving_mass_kg=task.moving_mass_kg,
                outbound_progress_m=task.outbound_progress_m,
                home_progress_m=task.home_progress_m,
                max_control_force_n=max(
                    task.outbound_force_n, task.return_force_n
                ),
            )
        )
    else:
        task = adapter.whole_machine_angular_task
        if task is None:
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_UNAVAILABLE",
                "revolute adapter has no trusted machine task",
            )
        if not {task.outbound_task_id, task.return_task_id}.issubset(plan.task_ids):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
                "certified capsule does not cover both required machine tasks",
            )
        torque = envelope.get("applied_torque_nm", {})
        angle = envelope.get("generalized_angle_rad")
        tolerance = math.radians(0.5)
        if (
            not isinstance(mass, dict)
            or not float(mass.get("min", math.inf)) <= task.moving_mass_kg
            <= float(mass.get("max", -math.inf))
            or not isinstance(torque, dict)
            or float(torque.get("max", -math.inf))
            < max(task.outbound_torque_nm, task.return_torque_nm)
            or not isinstance(angle, dict)
            or float(angle.get("min", math.inf)) > task.home_angle_rad + tolerance
            or float(angle.get("max", -math.inf))
            < task.outbound_angle_rad - tolerance
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
                "machine task mass, torque, or angle exceeds the certified envelope",
            )
        spring_stiffness = 0.0
        spring_damping = 0.0
        passive_return = False
        if adapter.family_spec is not None:
            from .axis_spring_spec import axis_spring_from_dict

            axis_spring = axis_spring_from_dict(
                adapter.family_spec,
                expected_family="revolute",
            )
            spring_stiffness = axis_spring.stiffness_si
            spring_damping = axis_spring.damping_si
            passive_return = task.return_torque_nm == 0.0
        result = WholeMachineRevoluteExecutionPlan(
            **common,
            coordinate_kind="rotation",
            task_ids=(task.outbound_task_id, task.return_task_id),
            moving_mass_kg=task.moving_mass_kg,
            outbound_angle_rad=task.outbound_angle_rad,
            home_angle_rad=task.home_angle_rad,
            max_control_torque_nm=max(
                task.outbound_torque_nm, task.return_torque_nm
            ),
            passive_return_stiffness_nm_rad=spring_stiffness,
            passive_return_damping_nm_s_rad=spring_damping,
            passive_return_required=passive_return,
        )
    result.validate()
    return result



def _whole_machine_max_memory_gib() -> float:
    """Adaptive per-job memory ceiling for whole-machine jobs.

    2026-08-10 ATTENTION AI EDITORS: do NOT hard-code this value again.
    The old fixed 1.0 GiB cap false-RESOURCE_BLOCKED the r29 bicycle
    chain whole-machine job whose measured peak RSS is ~1.12 GiB, on a
    machine with plenty of free RAM. Machine-wide memory pressure is
    already guarded by the queue SystemProbe (ResourcePolicy); this cap
    only stops a single runaway job, so it scales with total RAM
    (25%, floor 1.0 GiB) and works on machines of any size.
    Set SIMREADY_WHOLE_MACHINE_MAX_MEM_GIB to pin an explicit value.
    """
    import os

    override = os.environ.get("SIMREADY_WHOLE_MACHINE_MAX_MEM_GIB")
    if override:
        try:
            value = float(override)
        except ValueError:
            value = 0.0
        if value > 0:
            return value
    try:
        import psutil

        total_gib = psutil.virtual_memory().total / (1024**3)
    except Exception:
        return 1.0
    return max(1.0, round(0.25 * total_gib, 2))


def build_whole_machine_dispatch(
    plan: MechanismPlan,
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    certificate_file: str | Path,
    replacement_file: str | Path,
    additional_input_files: Sequence[str | Path] = (),
    pychrono_launcher: PyChronoLauncher | None = None,
    conda_executable: str | None = None,
    conda_environment: str | None = None,
) -> dict[str, Any]:
    launcher = select_pychrono_launcher(
        pychrono_launcher,
        legacy_conda_executable=conda_executable,
        legacy_conda_environment=conda_environment,
    )
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    output.relative_to(workspace)
    validator_source_sha256 = mechanism_source_sha256()
    execution = build_execution_plan(
        plan,
        workspace_root=workspace,
        output_dir=output,
        certificate_file=certificate_file,
        replacement_file=replacement_file,
    )
    execution_path = output / "whole_machine" / "execution_plan.json"
    mechanism_plan_path = output / "whole_machine" / "mechanism_plan.json"
    result_path = output / "whole_machine" / "result.json"
    write_json_atomic(execution_path, execution.to_dict())
    write_json_atomic(mechanism_plan_path, plan.to_dict())
    execution_relative = relative_to_root(workspace, execution_path)
    mechanism_plan_relative = relative_to_root(workspace, mechanism_plan_path)
    result_relative = relative_to_root(workspace, result_path)
    package = Path(plan.package_dir).resolve()
    input_candidates = {
        execution_path.resolve(),
        mechanism_plan_path.resolve(),
        Path(certificate_file).resolve(),
        Path(replacement_file).resolve(),
        Path(plan.intake_report).resolve(),
        (package / "package.json").resolve(),
        *{
            (package / relative).resolve()
            for relative in plan.artifact_hashes
        },
    }
    for additional in additional_input_files:
        additional_path = Path(additional).resolve()
        try:
            additional_path.relative_to(workspace)
        except ValueError as exc:
            raise MechanismValidationError(
                "WHOLE_MACHINE_ADDITIONAL_INPUT_OUTSIDE_WORKSPACE",
                "whole-machine additional input must stay inside the "
                "workspace",
            ) from exc
        if (
            not additional_path.is_file()
            or additional_path.is_symlink()
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_ADDITIONAL_INPUT_INVALID",
                "whole-machine additional input must be a regular file",
            )
        input_candidates.add(additional_path)
    input_files = {
        relative_to_root(workspace, path): sha256_file(path)
        for path in sorted(input_candidates, key=lambda item: item.as_posix())
    }
    expected_outputs = [result_relative]
    output_trace_file = getattr(execution, "output_trace_file", None)
    if isinstance(output_trace_file, str) and output_trace_file:
        expected_outputs.append(output_trace_file)
    job = JobSpec(
        job_id=f"whole-machine-{plan.capsule_id}"[:64],
        kind=JobKind.PYCHRONO,
        command=launcher.python_command(
            "-m",
            (
                "simulation_ready_cad_workflow.validation_pipeline."
                "mechanism.whole_machine_worker"
            ),
            "--plan",
            execution_relative,
            "--mechanism-plan",
            mechanism_plan_relative,
            "--output",
            result_relative,
        ),
        working_directory=".",
        timeout_seconds=180.0,
        expected_outputs=tuple(expected_outputs),
        input_files=dict(sorted(input_files.items())),
        environment={
            "PYTHONUNBUFFERED": "1",
            # The queue may execute from a clean benchmark scratch workspace
            # with no importable code. Bind the installed validator package
            # root explicitly instead of depending on the shell that
            # happened to launch the validator.
            "PYTHONPATH": str(validator_import_root()),
            MECHANISM_SOURCE_SHA256_ENV: validator_source_sha256,
        },
        # 2026-08-10 ATTENTION AI EDITORS: measured peak RSS of the
        # bicycle chain whole-machine job is ~1.12 GiB (r29). The old
        # estimate 0.45 / fixed cap 1.0 caused a false RESOURCE_BLOCKED.
        estimated_peak_memory_gib=1.25,
        max_process_tree_memory_gib=_whole_machine_max_memory_gib(),
        estimated_wall_time_seconds=30.0,
        estimated_simulation_steps=int(
            2.0 * execution.max_stage_duration_s / execution.step_s
        ),
        cpu_threads=2,
    )
    manifest = QueueManifest(
        run_name=f"whole-machine-{plan.package_binding_sha256[:12]}",
        jobs=(job,),
        resource_policy=ResourcePolicy(
            preflight_memory_percent=86.0,
            min_available_memory_gib=1.5,
            min_free_disk_gib=10.0,
            critical_available_memory_gib=0.75,
            critical_free_disk_gib=5.0,
            preflight_sample_seconds=10.0,
            max_resource_wait_seconds=30.0,
            cooldown_seconds=2.0,
            pychrono_cpu_threads=2,
            generic_cpu_threads=2,
            max_log_bytes_per_stream=8 * 1024 * 1024,
            lower_process_priority=True,
        ),
        queue_policy=QueuePolicy(
            max_wall_time_seconds=900.0,
            stop_on_failure=True,
            reuse_successful=True,
        ),
    )
    manifest_path = output / "whole_machine" / "queue_manifest.json"
    write_json_atomic(manifest_path, manifest.to_dict())
    summary = {
        "schema_version": "0.1",
        "execution": "serial",
        "max_parallel_jobs": 1,
        "queue_manifest": relative_to_root(workspace, manifest_path),
        "execution_plan": execution_relative,
        "expected_result": result_relative,
    }
    write_json_atomic(output / "whole_machine" / "dispatch_summary.json", summary)
    return summary
