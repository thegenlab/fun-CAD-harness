from __future__ import annotations

"""Whole-machine replay binding for the certified bicycle ground-roll task."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from .adapters import AdapterSpec, adapter_from_dict, adapter_to_dict
from .bicycle_ground_roll_comparison import (
    _behavior_failures,
    _metrics,
    _semantic_failures,
    _spec,
    _thresholds,
    _validate_trace,
)
from .bicycle_ground_roll_spec import (
    BICYCLE_GROUND_ROLL_PLUGIN_ID,
    BICYCLE_GROUND_ROLL_PROGRAM,
    bicycle_ground_roll_from_dict,
)
from .family_plugins import MechanismAdapter, mechanism_adapter_for
from .io import (
    canonical_sha256,
    read_trace_json,
    relative_to_root,
    sha256_file,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError


@dataclass(frozen=True)
class WholeMachineBicycleGroundRollExecutionPlan:
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
    minimum_front_rotation_rad: float
    minimum_chassis_travel_m: float
    maximum_rolling_slip_ratio: float
    maximum_axle_drift_m: float
    maximum_guide_drift_m: float
    maximum_off_axis_rotation_rad: float
    maximum_ground_penetration_m: float
    maximum_ground_separation_m: float
    minimum_powered_contact_coverage: float
    max_stage_duration_s: float
    step_s: float
    sample_interval_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["task_ids"] = list(self.task_ids)
        return payload

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "WholeMachineBicycleGroundRollExecutionPlan":
        payload = dict(value)
        payload["task_ids"] = tuple(payload["task_ids"])
        result = cls(**payload)
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != "0.1"
            or self.coordinate_kind != "bicycle_ground_roll"
            or self.worker_backend != BICYCLE_GROUND_ROLL_PLUGIN_ID
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "bicycle ground-roll whole-machine identity is invalid",
            )
        mechanism = MechanismPlan.from_dict(self.mechanism_plan)
        load_case = LoadCase.from_dict(self.task_load_case)
        adapter = adapter_from_dict(self.adapter_spec)
        plugin = mechanism_adapter_for(mechanism, adapter)
        bicycle_ground_roll_from_dict(
            {
                key: item
                for key, item in (adapter.family_spec or {}).items()
                if key != "collision_geometry"
            }
        )
        if (
            mechanism.package_binding_sha256
            != self.package_binding_sha256
            or mechanism.capsule_id != self.capsule_id
            or plugin.plugin_id != self.mechanism_plugin_id
            or load_case.program != BICYCLE_GROUND_ROLL_PROGRAM
            or load_case.load_case_id != BICYCLE_GROUND_ROLL_PROGRAM
            or not set(self.task_ids).issubset(mechanism.task_ids)
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "ground-roll mechanism, task, or load identity changed",
            )
        positive = (
            self.minimum_front_rotation_rad,
            self.minimum_chassis_travel_m,
            self.maximum_rolling_slip_ratio,
            self.maximum_axle_drift_m,
            self.maximum_guide_drift_m,
            self.maximum_off_axis_rotation_rad,
            self.maximum_ground_penetration_m,
            self.maximum_ground_separation_m,
            self.minimum_powered_contact_coverage,
            self.max_stage_duration_s,
            self.step_s,
            self.sample_interval_s,
        )
        if (
            any(not math.isfinite(item) or item <= 0.0 for item in positive)
            or self.sample_interval_s < self.step_s
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "ground-roll whole-machine thresholds are invalid",
            )


def _physical_reference(
    *,
    workspace: Path,
    certificate: dict[str, Any],
    plan: MechanismPlan,
    load_case: LoadCase,
) -> Path:
    hashes = certificate.get("trace_file_hashes")
    if isinstance(hashes, dict):
        for candidate, expected_hash in hashes.items():
            candidate_path = Path(candidate)
            path = (
                candidate_path.resolve()
                if candidate_path.is_absolute()
                else (workspace / candidate_path).resolve()
            )
            try:
                path.relative_to(workspace)
            except ValueError:
                continue
            if path.is_file() and sha256_file(path) == expected_hash:
                trace = read_trace_json(path)
                if (
                    trace.get("branch") == "physical"
                    and trace.get("load_case_id") == load_case.load_case_id
                    and trace.get("package_binding_sha256")
                    == plan.package_binding_sha256
                ):
                    return path
    raise MechanismValidationError(
        "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
        "ground-roll whole-machine execution requires its certified physical trace",
    )


def build_bicycle_ground_roll_execution_plan(
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
) -> WholeMachineBicycleGroundRollExecutionPlan:
    spec = bicycle_ground_roll_from_dict(
        {
            key: item
            for key, item in (adapter.family_spec or {}).items()
            if key != "collision_geometry"
        }
    )
    topology = replacement_model.get("compound_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("mechanism_kind") != "bicycle_ground_roll"
        or topology.get("family_spec_sha256")
        != canonical_sha256(spec.to_dict())
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "ground-roll replacement topology differs from the certified family spec",
        )
    task_load = next(
        (
            item
            for item in plugin.load_cases(plan)
            if item.load_case_id == BICYCLE_GROUND_ROLL_PROGRAM
            and item.program == BICYCLE_GROUND_ROLL_PROGRAM
        ),
        None,
    )
    if task_load is None:
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_LOAD_MISSING",
            "ground-roll task load case is unavailable",
        )
    reference = _physical_reference(
        workspace=workspace,
        certificate=certificate,
        plan=plan,
        load_case=task_load,
    )
    thresholds = _thresholds(spec)
    result = WholeMachineBicycleGroundRollExecutionPlan(
        schema_version="0.1",
        coordinate_kind="bicycle_ground_roll",
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
        minimum_front_rotation_rad=thresholds["minimum_front_rotation_rad"],
        minimum_chassis_travel_m=thresholds["minimum_chassis_travel_m"],
        maximum_rolling_slip_ratio=thresholds["maximum_rolling_slip_ratio"],
        maximum_axle_drift_m=thresholds["maximum_axle_drift_m"],
        maximum_guide_drift_m=thresholds["maximum_guide_drift_m"],
        maximum_off_axis_rotation_rad=thresholds["maximum_off_axis_rotation_rad"],
        maximum_ground_penetration_m=thresholds["maximum_ground_penetration_m"],
        maximum_ground_separation_m=thresholds["maximum_ground_separation_m"],
        minimum_powered_contact_coverage=thresholds["minimum_powered_contact_coverage"],
        max_stage_duration_s=sum(phase.duration_s for phase in task_load.phases),
        step_s=task_load.step_s,
        sample_interval_s=task_load.sample_interval_s,
    )
    result.validate()
    return result


def simulate_whole_machine_bicycle_ground_roll(
    plan: WholeMachineBicycleGroundRollExecutionPlan,
    workspace_root: str | Path,
) -> dict[str, Any]:
    from .bicycle_ground_roll_runtime import simulate_bicycle_ground_roll
    from .worker import ExecutionRequest, mechanism_source_sha256

    mechanism = MechanismPlan.from_dict(plan.mechanism_plan)
    load_case = LoadCase.from_dict(plan.task_load_case)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism, adapter)
    spec = _spec(mechanism)
    workspace = Path(workspace_root).resolve()
    reference = (workspace / plan.reference_trace_file).resolve()
    try:
        reference.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
            "ground-roll reference trace escapes the workspace",
        ) from exc
    if sha256_file(reference) != plan.reference_trace_sha256:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_TRACE_CHANGED",
            "certified ground-roll reference trace changed after planning",
        )
    reference_trace = read_trace_json(reference)
    functional_geometry = reference_trace.get("functional_geometry")
    if (
        not isinstance(functional_geometry, dict)
        or functional_geometry.get("passed") is not True
        or reference_trace.get("package_binding_sha256")
        != mechanism.package_binding_sha256
        or reference_trace.get("capsule_id") != mechanism.capsule_id
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_FUNCTIONAL_GEOMETRY_INVALID",
            "certified ground-roll trace has no matching PASS geometry evidence",
        )
    execution = ExecutionRequest(
        request_file=reference.with_name("whole_machine.request.json"),
        output_file=reference.with_name("whole_machine.trace.json"),
        branch="physical",
        artifact_branch="physical",
        adapter=adapter,
        mechanism_plugin=plugin,
        plan=mechanism,
        load_case=load_case,
        reference_trace_file=None,
        functional_geometry_evidence=functional_geometry,
    )
    return simulate_bicycle_ground_roll(execution, workspace)


def summarize_whole_machine_bicycle_ground_roll(
    plan: WholeMachineBicycleGroundRollExecutionPlan,
    trace: dict[str, Any],
) -> dict[str, Any]:
    from .worker import mechanism_source_sha256

    mechanism = MechanismPlan.from_dict(plan.mechanism_plan)
    load_case = LoadCase.from_dict(plan.task_load_case)
    spec = _spec(mechanism)
    samples = _validate_trace(
        trace,
        branch="physical",
        plan=mechanism,
        load_case=load_case,
        spec=spec,
    )
    metrics = _metrics(samples, spec)
    failures = [
        *_semantic_failures(trace, spec),
        *_behavior_failures(metrics, spec),
    ]
    passed = not failures
    return {
        "schema_version": "0.1",
        "validator_source_sha256": mechanism_source_sha256(),
        "status": (
            "WHOLE_MACHINE_TASK_PASSED"
            if passed
            else "WHOLE_MACHINE_TASK_FAILED"
        ),
        "passed": passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "mechanism_plugin_id": plan.mechanism_plugin_id,
        "worker_backend": plan.worker_backend,
        "certificate_sha256": plan.certificate_sha256,
        "replacement_sha256": plan.replacement_sha256,
        "failure_reasons": failures,
        "thresholds": _thresholds(spec),
        "physical_metrics": metrics,
        "engine": trace["engine"],
        "execution_method": "certified_rear_wheel_torque_schedule_replay",
        "executed_phases": trace["executed_phases"],
        "executed_load_sha256": trace["executed_load_sha256"],
        "sample_count": len(samples),
        "samples": samples,
        "strict_non_claims": {
            "visual_pose_replay_used": False,
            "contact_bypass_ideal_link_used": False,
            "declared_support_links_used": True,
            "front_wheel_direct_actuation_used": False,
            "unassisted_balance_certified": False,
        },
    }
