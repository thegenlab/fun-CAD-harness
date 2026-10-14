"""Whole-machine execution plan for the spur-gear clock family.

Family-owned logic extracted from the generic whole-machine dispatcher;
only :class:`..family_plugins.SpurGearClockAdapter` calls this.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .adapters import adapter_to_dict
from .io import (
    canonical_sha256,
    ensure_inside,
    read_trace_json,
    relative_to_root,
    sha256_file,
)
from .models import MechanismPlan, MechanismValidationError
from .whole_machine import WholeMachineClockExecutionPlan


def build_clock_execution_plan(
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
) -> WholeMachineClockExecutionPlan:
    from .clock_gear_train_spec import clock_gear_train_from_dict

    clock = clock_gear_train_from_dict(adapter.family_spec)
    topology = replacement_model.get("compound_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("family_spec_sha256") != canonical_sha256(clock.to_dict())
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "clock replacement topology differs from the certified family spec",
        )
    required_tasks = {
        "run_three_hand_clock",
        "verify_three_hand_ratio",
        "set_time_manually",
        "ring_twin_bell_alarm",
    }
    if clock.alarm_setting_clutch is not None:
        required_tasks = {
            "run_three_hand_clock_logic",
            "set_time_manually",
            "ring_alarm_independently",
            "set_alarm_time_by_turning_hour_hand",
            "trigger_alarm_at_selected_hand_position",
        }
    elif clock.scheduled_alarm is not None:
        required_tasks.update(
            {"zero_time_and_resume", "alarm_after_one_minute"}
        )
    if not required_tasks.issubset(plan.task_ids):
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
            "clock mechanism must map running, hand-ratio, manual-setting, and alarm tasks",
        )
    envelope = certificate.get("tested_envelope", {})
    torque = envelope.get("applied_torque_nm", {})
    angle = envelope.get("generalized_output_angle_rad")
    if (
        not isinstance(torque, dict)
        or float(torque.get("max", -math.inf))
        < max(clock.nominal_input_torque_nm, clock.return_input_torque_nm)
        or not isinstance(angle, dict)
        or float(angle.get("max", -math.inf))
        < clock.outbound_output_angle_rad - clock.output_angle_tolerance_rad
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
            "clock task torque or display travel exceeds the certified envelope",
        )
    task_cases = tuple(
        case for case in plugin.load_cases(plan)
        if case.program in {
            "task_clock_cycle",
            "manual_time_setting_clock",
            "alarm_clock_cycle",
            *(
                {
                    "manual_zero_and_resume_clock",
                    "scheduled_alarm_clock",
                }
                if (
                    clock.scheduled_alarm is not None
                    and clock.alarm_setting_clutch is None
                )
                else set()
            ),
            *(
                {
                    "hour_hand_alarm_setting_clock",
                    "scheduled_alarm_clock",
                }
                if clock.alarm_setting_clutch is not None
                else set()
            ),
        }
    )
    trace_hashes = certificate.get("trace_file_hashes", {})
    references: list[dict[str, str]] = []
    for task_case in task_cases:
        match: dict[str, str] | None = None
        if isinstance(trace_hashes, dict):
            for candidate, expected_hash in trace_hashes.items():
                candidate_input = Path(candidate)
                candidate_path = (
                    candidate_input.resolve()
                    if candidate_input.is_absolute()
                    else ensure_inside(workspace, candidate)
                )
                try:
                    candidate_path.relative_to(workspace)
                except ValueError:
                    continue
                if (
                    not candidate_path.is_file()
                    or sha256_file(candidate_path) != expected_hash
                ):
                    continue
                document = read_trace_json(candidate_path)
                if (
                    document.get("branch") == "physical"
                    and document.get("load_case_id") == task_case.load_case_id
                    and document.get("package_binding_sha256")
                    == plan.package_binding_sha256
                ):
                    match = {
                        "load_case_id": task_case.load_case_id,
                        "file": relative_to_root(workspace, candidate_path),
                        "sha256": expected_hash,
                    }
                    break
        if match is None:
            raise MechanismValidationError(
                "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
                f"clock task {task_case.load_case_id!r} requires its certified physical trace",
            )
        references.append(match)
    result = WholeMachineClockExecutionPlan(
        schema_version="0.1",
        coordinate_kind="clock_multifunction",
        package_binding_sha256=plan.package_binding_sha256,
        capsule_id=plan.capsule_id,
        certificate_file=relative_to_root(workspace, certificate_path),
        certificate_sha256=certificate_sha256,
        replacement_file=relative_to_root(workspace, replacement_path),
        replacement_sha256=replacement_sha256,
        adapter_id=adapter.adapter_id,
        mechanism_plugin_id=plugin.plugin_id,
        worker_backend=plugin.whole_machine_backend,
        task_ids=tuple(task_id for task_id in plan.task_ids if task_id in required_tasks),
        mechanism_plan=plan.to_dict(),
        adapter_spec=adapter_to_dict(adapter),
        task_load_cases=tuple(item.to_dict() for item in task_cases),
        reference_traces=tuple(references),
        outbound_angle_rad=clock.outbound_output_angle_rad,
        angle_tolerance_rad=clock.output_angle_tolerance_rad,
        expected_hour_to_minute_ratio=clock.expected_hour_to_minute_ratio,
        expected_second_to_minute_ratio=clock.expected_second_to_minute_ratio,
        hand_ratio_relative_tolerance=clock.hand_ratio_relative_tolerance,
        minimum_alarm_swing_rad=clock.minimum_alarm_swing_rad,
        max_stage_duration_s=max(item.duration_s for item in task_cases),
        step_s=min(item.step_s for item in task_cases),
    )
    result.validate()
    return result

