"""Whole-machine execution plan and replay for the ``hydraulic_scissor_lift`` family.

Mirrors ``rack_pinion_whole_machine_v4.py``: the execution plan binds the
certificate, the replacement topology, and the certified physical task trace;
the worker-side simulation replays the certified physical cylinder-force
schedule on the reduced (ideal-link) model and reports outbound/return stages
against the platform-progress contract.  No physics is launched here directly
— this module is consumed by ``whole_machine_worker.py`` inside the harness
whole-machine queue.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters import AdapterSpec, adapter_from_dict
from .family_plugins import MechanismAdapter, mechanism_adapter_for
from .io import (
    canonical_sha256,
    read_trace_json,
    relative_to_root,
    sha256_file,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .scissor_lift_spec import expected_ideal_links, scissor_lift_from_dict


@dataclass(frozen=True)
class WholeMachineScissorLiftExecutionPlan:
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
    output_trace_file: str
    outbound_progress_m: float
    home_progress_m: float
    max_control_force_n: float
    progress_tolerance_m: float
    kinematic_joint_drift_tolerance_m: float = 1e-5
    kinematic_off_axis_tolerance_rad: float = 1e-6
    max_stage_duration_s: float = 2.0
    step_s: float = 0.0001
    sample_interval_s: float = 0.005

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_ids"] = list(self.task_ids)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WholeMachineScissorLiftExecutionPlan":
        data = dict(value)
        data["task_ids"] = tuple(data["task_ids"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != "0.1"
            or self.coordinate_kind != "scissor_lift_platform_progress"
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "scissor-lift whole-machine plan must use schema 0.1 and platform progress",
            )
        mechanism = MechanismPlan.from_dict(self.mechanism_plan)
        load_case = LoadCase.from_dict(self.task_load_case)
        adapter = adapter_from_dict(self.adapter_spec)
        plugin = mechanism_adapter_for(mechanism, adapter)
        scissor_lift_from_dict(adapter.family_spec)
        if (
            mechanism.family != "compound"
            or mechanism.package_binding_sha256 != self.package_binding_sha256
            or mechanism.capsule_id != self.capsule_id
            or plugin.plugin_id != self.mechanism_plugin_id
            or self.worker_backend != "hydraulic_scissor_lift_v1"
            or load_case.family != "compound"
            or load_case.program != "task_platform_roundtrip"
            or not self.output_trace_file
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "scissor-lift mechanism identity or task load case is inconsistent",
            )
        positive = {
            "outbound_progress_m": self.outbound_progress_m,
            "max_control_force_n": self.max_control_force_n,
            "progress_tolerance_m": self.progress_tolerance_m,
            "kinematic_joint_drift_tolerance_m": self.kinematic_joint_drift_tolerance_m,
            "kinematic_off_axis_tolerance_rad": self.kinematic_off_axis_tolerance_rad,
            "max_stage_duration_s": self.max_stage_duration_s,
            "step_s": self.step_s,
            "sample_interval_s": self.sample_interval_s,
        }
        if any(not math.isfinite(item) or item <= 0 for item in positive.values()):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "scissor-lift whole-machine limits must be positive",
            )
        if not math.isfinite(self.home_progress_m) or self.sample_interval_s < self.step_s:
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "scissor-lift home progress or sample interval is invalid",
            )
        if not set(self.task_ids).issubset(mechanism.task_ids):
            raise MechanismValidationError(
                "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
                "scissor-lift whole-machine tasks are not covered by the mechanism plan",
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
        "scissor-lift whole-machine replay requires the certified physical task trace",
    )


def build_scissor_lift_execution_plan(
    *,
    plan: MechanismPlan,
    workspace: Path,
    output_dir: Path,
    certificate_path: Path,
    certificate_sha256: str,
    replacement_path: Path,
    replacement_sha256: str,
    certificate: dict[str, Any],
    replacement_model: dict[str, Any],
    adapter: AdapterSpec,
    plugin: MechanismAdapter,
) -> WholeMachineScissorLiftExecutionPlan:
    spec = scissor_lift_from_dict(adapter.family_spec)
    topology = replacement_model.get("compound_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("mechanism_kind") != "hydraulic_scissor_lift"
        or topology.get("family_spec_sha256") != canonical_sha256(spec.to_dict())
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "scissor-lift replacement topology differs from the certified family spec",
        )
    if len(plan.task_ids) < 2:
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_MAPPING_MISMATCH",
            "scissor-lift mechanism must map raise and lower tasks",
        )
    envelope = certificate.get("tested_envelope", {})
    mass = envelope.get("moving_mass_kg", {})
    force = envelope.get("applied_force_n", {})
    progress = envelope.get("generalized_platform_progress_m")
    moving_mass = sum(item.mass_kg for item in spec.bodies if not item.fixed)
    tolerance = spec.output_progress_tolerance_m
    if (
        not isinstance(mass, dict)
        or not float(mass.get("min", math.inf)) <= moving_mass <= float(mass.get("max", -math.inf))
        or not isinstance(force, dict)
        or float(force.get("max", -math.inf))
        < max(spec.nominal_input_force_n, spec.return_input_force_n)
        or not isinstance(progress, dict)
        or float(progress.get("min", math.inf)) > spec.home_progress_m + tolerance
        or float(progress.get("max", -math.inf)) < spec.outbound_progress_m - tolerance
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
            "scissor-lift machine task mass, force, or travel exceeds the certified envelope",
        )
    task_load = next(
        item for item in plugin.load_cases(plan) if item.program == "task_platform_roundtrip"
    )
    reference = _physical_task_reference(
        workspace=workspace, certificate=certificate, plan=plan, task_load=task_load
    )
    result = WholeMachineScissorLiftExecutionPlan(
        schema_version="0.1",
        coordinate_kind="scissor_lift_platform_progress",
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
        adapter_spec=__import__(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters",
            fromlist=["adapter_to_dict"],
        ).adapter_to_dict(adapter),
        task_load_case=task_load.to_dict(),
        reference_trace_file=relative_to_root(workspace, reference),
        reference_trace_sha256=sha256_file(reference),
        output_trace_file=relative_to_root(
            workspace,
            output_dir / "whole_machine" / "whole_machine.trace.json",
        ),
        outbound_progress_m=spec.outbound_progress_m,
        home_progress_m=spec.home_progress_m,
        max_control_force_n=max(
            spec.nominal_input_force_n, spec.return_input_force_n
        ),
        progress_tolerance_m=spec.output_progress_tolerance_m,
        kinematic_joint_drift_tolerance_m=1e-5,
        kinematic_off_axis_tolerance_rad=1e-6,
        max_stage_duration_s=max(
            spec.max_force_phase_s, spec.max_return_phase_s
        ) + spec.settle_phase_s,
        step_s=task_load.step_s,
        sample_interval_s=task_load.sample_interval_s,
    )
    result.validate()
    return result


def simulate_whole_machine_scissor_lift(
    plan: WholeMachineScissorLiftExecutionPlan,
) -> dict[str, Any]:
    from .scissor_lift_runtime import simulate_scissor_lift
    from .worker import ExecutionRequest, mechanism_source_sha256

    mechanism = MechanismPlan.from_dict(plan.mechanism_plan)
    load_case = LoadCase.from_dict(plan.task_load_case)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism, adapter)
    spec = scissor_lift_from_dict(adapter.family_spec)
    reference = Path(plan.reference_trace_file).resolve()
    output_trace = Path(plan.output_trace_file).resolve()
    trace = simulate_scissor_lift(
        ExecutionRequest(
            request_file=output_trace.with_name("whole_machine.request.json"),
            output_file=output_trace,
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
    raw_samples = trace["samples"]
    first_loaded_index = next(
        index for index, phase in enumerate(load_case.phases)
        if phase.wrench_kind != "none"
    )
    second_loaded_index = next(
        index for index, phase in enumerate(load_case.phases[first_loaded_index + 1:], first_loaded_index + 1)
        if phase.wrench_kind != "none"
    )
    outbound_phase_ids = {
        phase.phase_id for phase in load_case.phases[:second_loaded_index]
    }
    samples: list[dict[str, Any]] = []
    for sample in raw_samples:
        phase_id = sample["phase_id"]
        stage = (
            "initial" if phase_id == "initial"
            else "outbound" if phase_id in outbound_phase_ids
            else "return"
        )
        force = sample.get("applied_force_abs_n", [0.0, 0.0, 0.0])
        samples.append(
            {
                **sample,
                "stage": stage,
                "target_progress_m": (
                    plan.outbound_progress_m if stage == "outbound" else plan.home_progress_m
                ),
                "progress_m": sample["platform_progress_m"],
                "control_force_n": math.sqrt(sum(value * value for value in force)),
            }
        )
    progress_values = [float(item["progress_m"]) for item in samples]
    max_joint_drift = max(float(item["max_joint_drift_m"]) for item in samples)
    max_off_axis = max(float(item["off_plane_rotation_rad"]) for item in samples)
    outbound_values = [
        float(item["progress_m"]) for item in samples if item["stage"] == "outbound"
    ]
    return_values = [
        float(item["progress_m"]) for item in samples if item["stage"] == "return"
    ]
    outbound_reached = bool(outbound_values) and max(outbound_values) >= (
        plan.outbound_progress_m - plan.progress_tolerance_m
    )
    home_reached = abs(progress_values[-1] - plan.home_progress_m) <= (
        plan.progress_tolerance_m
    )
    failures: list[str] = []
    if not outbound_reached:
        failures.append("outbound_progress_not_reached")
    if not home_reached:
        failures.append("home_progress_not_reached")
    if max_joint_drift > plan.kinematic_joint_drift_tolerance_m:
        failures.append("ideal_linkage_numerical_joint_drift_exceeded")
    if max_off_axis > plan.kinematic_off_axis_tolerance_rad:
        failures.append("ideal_linkage_numerical_off_axis_drift_exceeded")
    expected_links = list(expected_ideal_links(spec))
    if trace.get("engine", {}).get("internal_ideal_joint_count") != len(expected_links):
        failures.append("replacement_scissor_lift_topology_count_mismatch")
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
            "kinematic_joint_drift_tolerance_m": plan.kinematic_joint_drift_tolerance_m,
            "kinematic_off_axis_tolerance_rad": plan.kinematic_off_axis_tolerance_rad,
        },
        "task_ids": list(plan.task_ids),
        "engine": {
            **trace["engine"],
            "replacement_links": expected_links,
            "replacement_link_count": len(expected_links),
            "finite_bushings_and_stops_replaced": True,
        },
        "controller": {
            "kind": "certified_cylinder_force_schedule_replay",
            "max_force_n": plan.max_control_force_n,
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
                "stage": "return",
                "target_progress_m": plan.home_progress_m,
                "reached": home_reached,
                "min_progress_m": min(return_values) if return_values else None,
                "max_progress_m": max(return_values) if return_values else None,
            },
        ],
        "max_joint_drift_m": max_joint_drift,
        "max_off_axis_rotation_rad": max_off_axis,
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "high_fidelity_whole_machine_bushing_run": False,
            "dynamic_response_certified": False,
            "force_schedule_represents_real_hardware_controller": False,
            "hydraulic_pressure_or_flow_certified": False,
        },
    }
