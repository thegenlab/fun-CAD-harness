from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters import AdapterSpec, adapter_from_dict, adapter_to_dict
from .family_plugins import MechanismAdapter, mechanism_adapter_for
from .io import (
    canonical_sha256,
    read_trace_json,
    relative_to_root,
    sha256_file,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .rack_pinion_spec import rack_pinion_from_dict
from .rack_pinion_loads_v3 import (
    RACK_PINION_ACTUATOR_LIMIT_SCALE,
    RACK_PINION_ANTI_DROP_HOLD_S,
)


@dataclass(frozen=True)
class WholeMachineRackPinionExecutionPlan:
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
    task_load_case: dict[str, Any]
    reference_trace_file: str
    reference_trace_sha256: str
    outbound_progress_m: float
    home_progress_m: float
    max_control_torque_nm: float
    progress_tolerance_m: float
    coupling_tolerance_m: float = 5e-5
    max_stage_duration_s: float = 2.0
    step_s: float = 0.0001
    sample_interval_s: float = 0.005

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_ids"] = list(self.task_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WholeMachineRackPinionExecutionPlan":
        data = dict(value)
        data["task_ids"] = tuple(data["task_ids"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema_version != "0.1" or self.coordinate_kind != "rack_pinion_progress":
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "rack-pinion whole-machine plan must use schema 0.1 and rack progress",
            )
        mechanism = MechanismPlan.from_dict(self.mechanism_plan)
        load_case = LoadCase.from_dict(self.task_load_case)
        adapter = adapter_from_dict(self.adapter_spec)
        plugin = mechanism_adapter_for(mechanism, adapter)
        rack_pinion_from_dict(adapter.family_spec)
        if (
            mechanism.family != "compound"
            or mechanism.package_binding_sha256 != self.package_binding_sha256
            or mechanism.capsule_id != self.capsule_id
            or plugin.plugin_id != self.mechanism_plugin_id
            or self.worker_backend != "rack_pinion_v1"
            or load_case.family != "compound"
            or load_case.program != "task_rack_roundtrip"
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "rack-pinion mechanism identity or task load case is inconsistent",
            )
        positive = {
            "outbound_progress_m": self.outbound_progress_m,
            "max_control_torque_nm": self.max_control_torque_nm,
            "progress_tolerance_m": self.progress_tolerance_m,
            "coupling_tolerance_m": self.coupling_tolerance_m,
            "max_stage_duration_s": self.max_stage_duration_s,
            "step_s": self.step_s,
            "sample_interval_s": self.sample_interval_s,
        }
        if any(not math.isfinite(item) or item <= 0 for item in positive.values()):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "rack-pinion whole-machine limits must be positive",
            )
        if not math.isfinite(self.home_progress_m) or self.sample_interval_s < self.step_s:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "rack-pinion home progress or sample interval is invalid",
            )
        if not set(self.task_ids).issubset(mechanism.task_ids):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
                "rack-pinion whole-machine tasks are not covered by the mechanism plan",
            )


def _physical_task_reference(
    *,
    workspace: Path,
    certificate: dict[str, Any],
    plan: MechanismPlan,
    task_load: LoadCase,
) -> Path:
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
                return candidate_path
    raise MechanismValidationError(
        "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
        "rack-pinion whole-machine replay requires the certified physical task trace",
    )


def build_rack_pinion_execution_plan(
    *,
    plan: MechanismPlan,
    workspace: Path,
    certificate_path: Path,
    certificate_sha256: str,
    replacement_path: Path,
    replacement_sha256: str,
    certificate: dict[str, Any],
    replacement_model: dict[str, Any],
    adapter: AdapterSpec,
    plugin: MechanismAdapter,
) -> WholeMachineRackPinionExecutionPlan:
    spec = rack_pinion_from_dict(adapter.family_spec)
    topology = replacement_model.get("compound_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("mechanism_kind") != "rack_pinion"
        or topology.get("family_spec_sha256") != canonical_sha256(spec.to_dict())
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "rack-pinion replacement topology differs from the certified family spec",
        )
    if len(plan.task_ids) < 2:
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
            "rack-pinion mechanism must map raise and lower tasks",
        )
    envelope = certificate.get("tested_envelope", {})
    mass = envelope.get("moving_mass_kg", {})
    torque = envelope.get("applied_torque_nm", {})
    progress = envelope.get("generalized_rack_progress_m")
    moving_mass = sum(item.mass_kg for item in spec.bodies if not item.fixed)
    tolerance = spec.output_progress_tolerance_m
    if (
        not isinstance(mass, dict)
        or not float(mass.get("min", math.inf)) <= moving_mass <= float(mass.get("max", -math.inf))
        or not isinstance(torque, dict)
        or float(torque.get("max", -math.inf))
        < max(spec.nominal_input_torque_nm, spec.return_input_torque_nm)
        or not isinstance(progress, dict)
        or float(progress.get("min", math.inf)) > spec.home_progress_m + tolerance
        or float(progress.get("max", -math.inf)) < spec.outbound_progress_m - tolerance
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
            "rack-pinion machine task mass, torque, or travel exceeds the certified envelope",
        )
    task_load = next(
        item for item in plugin.load_cases(plan) if item.program == "task_rack_roundtrip"
    )
    reference = _physical_task_reference(
        workspace=workspace, certificate=certificate, plan=plan, task_load=task_load
    )
    result = WholeMachineRackPinionExecutionPlan(
        schema_version="0.1",
        coordinate_kind="rack_pinion_progress",
        package_binding_sha256=plan.package_binding_sha256,
        capsule_id=plan.capsule_id,
        certificate_file=relative_to_root(workspace, certificate_path),
        certificate_sha256=certificate_sha256,
        replacement_file=relative_to_root(workspace, replacement_path),
        replacement_sha256=replacement_sha256,
        adapter_id=adapter.adapter_id,
        mechanism_plugin_id=plugin.plugin_id,
        worker_backend=plugin.whole_machine_backend,
        task_ids=tuple(plan.task_ids),
        mechanism_plan=plan.to_dict(),
        adapter_spec=adapter_to_dict(adapter),
        task_load_case=task_load.to_dict(),
        reference_trace_file=relative_to_root(workspace, reference),
        reference_trace_sha256=sha256_file(reference),
        outbound_progress_m=spec.outbound_progress_m,
        home_progress_m=spec.home_progress_m,
        max_control_torque_nm=max(
            RACK_PINION_ACTUATOR_LIMIT_SCALE * spec.nominal_input_torque_nm,
            spec.return_input_torque_nm,
        ),
        progress_tolerance_m=spec.output_progress_tolerance_m,
        coupling_tolerance_m=5e-5,
        max_stage_duration_s=max(
            spec.max_torque_phase_s, spec.max_return_phase_s
        ) + spec.settle_phase_s + RACK_PINION_ANTI_DROP_HOLD_S,
        step_s=task_load.step_s,
        sample_interval_s=task_load.sample_interval_s,
    )
    result.validate()
    return result


def simulate_whole_machine_rack_pinion(
    plan: WholeMachineRackPinionExecutionPlan,
) -> dict[str, Any]:
    from .rack_pinion_runtime_v10 import simulate_rack_pinion
    from .worker import ExecutionRequest, mechanism_source_sha256

    mechanism = MechanismPlan.from_dict(plan.mechanism_plan)
    load_case = LoadCase.from_dict(plan.task_load_case)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism, adapter)
    reference = Path(plan.reference_trace_file).resolve()
    trace = simulate_rack_pinion(
        ExecutionRequest(
            request_file=reference.with_name("whole_machine.request.json"),
            output_file=reference.with_name("whole_machine.trace.json"),
            branch="reduced",
            artifact_branch="reduced",
            adapter=adapter,
            mechanism_plugin=plugin,
            plan=mechanism,
            load_case=load_case,
            reference_trace_file=reference,
        ),
        Path.cwd().resolve(),
    )
    samples = [
        {
            **item,
            "stage": (
                "outbound"
                if item["phase_id"] in {"crank_lift", "upper_settle"}
                else "anti_drop"
                if item["phase_id"] == "zero_input_hold"
                else "return"
                if item["phase_id"] in {"crank_lower", "home_settle"}
                else "initial"
            ),
            "progress_m": item["rack_progress_m"],
            "control_torque_nm": math.sqrt(
                sum(value * value for value in item["applied_torque_abs_nm"])
            ),
        }
        for item in trace["samples"]
    ]
    outbound_values = [item["progress_m"] for item in samples if item["stage"] == "outbound"]
    hold_samples = [item for item in samples if item["stage"] == "anti_drop"]
    hold_values = [item["progress_m"] for item in hold_samples]
    return_values = [item["progress_m"] for item in samples if item["stage"] == "return"]
    outbound_reached = bool(outbound_values) and max(outbound_values) >= (
        plan.outbound_progress_m - plan.progress_tolerance_m
    )
    home_reached = abs(samples[-1]["progress_m"] - plan.home_progress_m) <= (
        plan.progress_tolerance_m
    )
    # prevent_gate_drop_after_release: the hold interval must exist, keep the
    # commanded input at zero, last the approved duration, and bound the
    # downward slip of the raised gate.
    hold_first_index = samples.index(hold_samples[0]) if hold_samples else -1
    hold_start_s = float(
        samples[max(0, hold_first_index - 1)]["time_s"] if hold_samples else 0.0
    )
    hold_duration_s = (
        float(hold_samples[-1]["time_s"]) - hold_start_s if hold_samples else 0.0
    )
    hold_input_zero = bool(hold_samples) and all(
        item["control_torque_nm"] <= 1e-9 for item in hold_samples
    )
    anti_drop_slip_m = (max(hold_values) - min(hold_values)) if hold_values else math.inf
    anti_drop_held = (
        bool(hold_samples)
        and hold_input_zero
        and hold_duration_s + plan.sample_interval_s + 1e-9 >= RACK_PINION_ANTI_DROP_HOLD_S
        and anti_drop_slip_m <= plan.progress_tolerance_m
        and min(hold_values) >= plan.outbound_progress_m - plan.progress_tolerance_m
    )
    max_coupling_error = max(abs(item["coupling_error_m"]) for item in samples)
    max_support_drift = max(abs(item["max_support_drift_m"]) for item in samples)
    failures: list[str] = []
    if not outbound_reached:
        failures.append("outbound_progress_not_reached")
    if not home_reached:
        failures.append("home_progress_not_reached")
    if not hold_samples:
        failures.append("anti_drop_interval_missing")
    else:
        if not hold_input_zero:
            failures.append("nonzero_input_during_anti_drop_check")
        if hold_duration_s + plan.sample_interval_s + 1e-9 < RACK_PINION_ANTI_DROP_HOLD_S:
            failures.append("anti_drop_interval_too_short")
        if anti_drop_slip_m > plan.progress_tolerance_m:
            failures.append("gate_drop_exceeded")
        if min(hold_values) < plan.outbound_progress_m - plan.progress_tolerance_m:
            failures.append("anti_drop_not_held_at_top")
    if max_coupling_error > plan.coupling_tolerance_m:
        failures.append("ideal_rack_pinion_coupling_error_exceeded")
    if max_support_drift > 1e-5:
        failures.append("ideal_rack_guide_drift_exceeded")
    if trace.get("engine", {}).get("internal_ideal_joint_count") != 3:
        failures.append("replacement_rack_pinion_topology_count_mismatch")
    passed = not failures
    return {
        "schema_version": "0.1",
        "validator_source_sha256": mechanism_source_sha256(),
        "status": "WHOLE_MACHINE_TASK_PASSED" if passed else "WHOLE_MACHINE_TASK_FAILED",
        "passed": passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "mechanism_plugin_id": plan.mechanism_plugin_id,
        "worker_backend": plan.worker_backend,
        "certificate_sha256": plan.certificate_sha256,
        "replacement_sha256": plan.replacement_sha256,
        "failure_reasons": failures,
        "thresholds": {
            "progress_tolerance_m": plan.progress_tolerance_m,
            "coupling_tolerance_m": plan.coupling_tolerance_m,
            "kinematic_support_drift_tolerance_m": 1e-5,
            "min_anti_drop_hold_s": RACK_PINION_ANTI_DROP_HOLD_S,
            "max_anti_drop_slip_m": plan.progress_tolerance_m,
        },
        "task_ids": list(plan.task_ids),
        "engine": {
            **trace["engine"],
            "replacement_links": [
                "ChLinkLockRevolute",
                "ChLinkLockPrismatic",
                "ChLinkMateRackPinion",
            ],
            "replacement_link_count": 3,
            "finite_supports_and_gear_mesh_replaced": True,
        },
        "controller": {
            "kind": "certified_crank_torque_schedule_replay",
            "max_torque_nm": plan.max_control_torque_nm,
        },
        "stages": [
            {
                "stage": "outbound",
                "target_progress_m": plan.outbound_progress_m,
                "reached": outbound_reached,
                "min_progress_m": min(outbound_values) if outbound_values else None,
                "max_progress_m": max(outbound_values) if outbound_values else None,
            },
            {
                "stage": "anti_drop",
                "target_progress_m": plan.outbound_progress_m,
                "reached": anti_drop_held,
                "min_progress_m": min(hold_values) if hold_values else None,
                "max_progress_m": max(hold_values) if hold_values else None,
                "duration_s": hold_duration_s if hold_samples else None,
                "hold_duration_s": hold_duration_s if hold_samples else None,
                "downward_slip_m": anti_drop_slip_m if hold_values else None,
                "input_held_at_zero": hold_input_zero if hold_samples else None,
            },
            {
                "stage": "return",
                "target_progress_m": plan.home_progress_m,
                "reached": home_reached,
                "min_progress_m": min(return_values) if return_values else None,
                "max_progress_m": max(return_values) if return_values else None,
            },
        ],
        "max_coupling_error_m": max_coupling_error,
        "max_support_drift_m": max_support_drift,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "high_fidelity_whole_machine_bushing_run": False,
            "dynamic_response_certified": False,
            "torque_schedule_represents_real_hardware_controller": False,
            "exact_tooth_or_guide_contact_certified": False,
        },
    }
