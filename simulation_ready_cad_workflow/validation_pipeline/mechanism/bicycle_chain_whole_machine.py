from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .adapters import AdapterSpec, adapter_from_dict, adapter_to_dict
from .bicycle_chain_comparison import (
    _behavior_failures,
    _metrics,
    _physical_semantic_failures,
    _task_thresholds,
    _validate_trace,
)
from .bicycle_chain_spec import (
    BICYCLE_CHAIN_PLUGIN_ID,
    bicycle_chain_from_dict,
)
from .family_plugins import MechanismAdapter, mechanism_adapter_for
from .io import (
    canonical_sha256,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError


@dataclass(frozen=True)
class WholeMachineBicycleChainExecutionPlan:
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
    powered_rotation_min_rad: float
    angular_ratio: float
    angular_ratio_tolerance: float
    return_to_home_max_rad: float
    max_stage_duration_s: float
    step_s: float
    sample_interval_s: float

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["task_ids"] = list(self.task_ids)
        return value

    @classmethod
    def from_dict(
        cls,
        value: dict[str, Any],
    ) -> "WholeMachineBicycleChainExecutionPlan":
        data = dict(value)
        data["task_ids"] = tuple(data["task_ids"])
        result = cls(**data)
        result.validate()
        return result

    def validate(self) -> None:
        if (
            self.schema_version != "0.1"
            or self.coordinate_kind != "bicycle_chain_rotation"
            or self.worker_backend != BICYCLE_CHAIN_PLUGIN_ID
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_SCHEMA_INVALID",
                "bicycle whole-machine plan identity is invalid",
            )
        mechanism = MechanismPlan.from_dict(self.mechanism_plan)
        load_case = LoadCase.from_dict(self.task_load_case)
        adapter = adapter_from_dict(self.adapter_spec)
        plugin = mechanism_adapter_for(mechanism, adapter)
        family_spec = {
            key: item
            for key, item in (adapter.family_spec or {}).items()
            if key != "collision_geometry"
        }
        bicycle_chain_from_dict(family_spec)
        if (
            mechanism.package_binding_sha256
            != self.package_binding_sha256
            or mechanism.capsule_id != self.capsule_id
            or plugin.plugin_id != self.mechanism_plugin_id
            or load_case.program != "task_bicycle_powered_roundtrip"
            or load_case.load_case_id != "task_bicycle_powered_roundtrip"
            or not set(self.task_ids).issubset(mechanism.task_ids)
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "bicycle mechanism, task, or load identity changed",
            )
        positive = (
            self.powered_rotation_min_rad,
            self.angular_ratio,
            self.return_to_home_max_rad,
            self.max_stage_duration_s,
            self.step_s,
            self.sample_interval_s,
        )
        if (
            any(not math.isfinite(item) or item <= 0.0 for item in positive)
            or not math.isfinite(self.angular_ratio_tolerance)
            or self.angular_ratio_tolerance < 0.0
            or self.sample_interval_s < self.step_s
        ):
            raise MechanismValidationError(
                "WHOLE_MACHINE_PLAN_VALUE_INVALID",
                "bicycle whole-machine thresholds are invalid",
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
            if (
                path.is_file()
                and sha256_file(path) == expected_hash
            ):
                trace = read_trace_json(path)
                if (
                    trace.get("branch") == "physical"
                    and trace.get("load_case_id")
                    == load_case.load_case_id
                    and trace.get("package_binding_sha256")
                    == plan.package_binding_sha256
                ):
                    return path
    raise MechanismValidationError(
        "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
        "bicycle whole-machine execution requires its certified physical trace",
    )


def build_bicycle_chain_execution_plan(
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
) -> WholeMachineBicycleChainExecutionPlan:
    family_spec = {
        key: item
        for key, item in (adapter.family_spec or {}).items()
        if key != "collision_geometry"
    }
    spec = bicycle_chain_from_dict(family_spec)
    topology = replacement_model.get("compound_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("mechanism_kind")
        != "bicycle_chain_sprocket_drive"
        or topology.get("family_spec_sha256")
        != canonical_sha256(spec.to_dict())
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH",
            "bicycle replacement topology differs from the certified family spec",
        )
    task_load = next(
        (
            item
            for item in plugin.load_cases(plan)
            if item.load_case_id == "task_bicycle_powered_roundtrip"
            and item.program == "task_bicycle_powered_roundtrip"
        ),
        None,
    )
    if task_load is None:
        raise MechanismValidationError(
            "WHOLE_MACHINE_TASK_LOAD_MISSING",
            "bicycle task load case is unavailable",
        )
    reference = _physical_reference(
        workspace=workspace,
        certificate=certificate,
        plan=plan,
        load_case=task_load,
    )
    thresholds = _task_thresholds(plan)
    result = WholeMachineBicycleChainExecutionPlan(
        schema_version="0.1",
        coordinate_kind="bicycle_chain_rotation",
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
        powered_rotation_min_rad=thresholds[
            "powered_rotation_min_rad"
        ],
        angular_ratio=thresholds["signed_output_per_input"],
        angular_ratio_tolerance=thresholds[
            "angular_ratio_tolerance"
        ],
        return_to_home_max_rad=thresholds[
            "return_to_home_max_rad"
        ],
        max_stage_duration_s=sum(
            phase.duration_s for phase in task_load.phases
        ),
        step_s=task_load.step_s,
        sample_interval_s=task_load.sample_interval_s,
    )
    result.validate()
    return result


def simulate_whole_machine_bicycle_chain(
    plan: WholeMachineBicycleChainExecutionPlan,
    workspace_root: str | Path,
) -> dict[str, Any]:
    from .bicycle_chain_runtime import simulate_bicycle_chain
    from .worker import (
        ExecutionRequest,
        mechanism_source_sha256,
    )
    from .wrench_ledger import current_wrench_ledger

    mechanism = MechanismPlan.from_dict(plan.mechanism_plan)
    load_case = LoadCase.from_dict(plan.task_load_case)
    adapter = adapter_from_dict(plan.adapter_spec)
    plugin = mechanism_adapter_for(mechanism, adapter)
    spec = bicycle_chain_from_dict(
        {
            key: item
            for key, item in (adapter.family_spec or {}).items()
            if key != "collision_geometry"
        }
    )
    workspace = Path(workspace_root).resolve()
    reference = (workspace / plan.reference_trace_file).resolve()
    try:
        reference.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_TRACE_REQUIRED",
            "bicycle whole-machine reference trace escapes the workspace",
        ) from exc
    if sha256_file(reference) != plan.reference_trace_sha256:
        raise MechanismValidationError(
            "WHOLE_MACHINE_REFERENCE_TRACE_CHANGED",
            "certified bicycle reference trace changed after planning",
        )
    reference_trace = read_trace_json(reference)
    functional_geometry = reference_trace.get("functional_geometry")
    # 2026-08-10 ATTENTION AI EDITORS: the certified trace stores the
    # swept-envelope preflight under "branch_semantics", NOT
    # "semantics" (whole_machine_worker.py reads branch_semantics
    # too). The wrong key made swept_envelope None and tripped
    # WHOLE_MACHINE_FUNCTIONAL_GEOMETRY_INVALID. Keep ends in sync.
    reference_semantics = reference_trace.get("branch_semantics")
    swept_envelope = (
        reference_semantics.get("prescribed_swept_envelope_preflight")
        if isinstance(reference_semantics, dict)
        else None
    )
    if (
        not isinstance(functional_geometry, dict)
        or functional_geometry.get("status") != "PASS"
        or not isinstance(swept_envelope, dict)
        or swept_envelope.get("passed") is not True
        or reference_trace.get("package_binding_sha256")
        != mechanism.package_binding_sha256
        or reference_trace.get("capsule_id") != mechanism.capsule_id
    ):
        raise MechanismValidationError(
            "WHOLE_MACHINE_FUNCTIONAL_GEOMETRY_INVALID",
            "certified bicycle trace has no matching PASS geometry evidence",
        )
    runtime_geometry = {
        **functional_geometry,
        "passed": True,
        "geometry_gate": {
            "gate_sha256": functional_geometry.get(
                "geometry_gate_sha256"
            ),
            "initially_active": True,
        },
        "prescribed_swept_envelope_preflight": swept_envelope,
    }
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
        functional_geometry_evidence=runtime_geometry,
    )
    trace = simulate_bicycle_chain(
        execution,
        workspace,
    )
    # 2026-08-10 ATTENTION AI EDITORS: use the AMBIENT ledger owned by
    # the caller (whole_machine_worker._run_registered_whole_machine_-
    # load_case). Never create a private WrenchLedger here -- the
    # pre-2026-08-10 shadow ledger swallowed every phase transition
    # and tripped WRENCH_PHASE_BARRIER_COVERAGE_INCOMPLETE on the
    # caller's ledger (r30). Bind+verify here so the semantic checks
    # below see the final audit; the caller re-binds and re-verifies
    # the same schedule afterwards (bind_executed_schedule is
    # idempotent, verify is pure). This is the current binding
    # design, not legacy cruft.
    ledger = current_wrench_ledger()
    ledger.bind_executed_schedule(
        trace.get("executed_phases"),
        executed_load_sha256=trace.get("executed_load_sha256"),
    )
    trace["wrench_audit"] = ledger.verify_for_certification()
    # 2026-08-11 ATTENTION AI EDITORS: the final authoritative trace
    # MUST be written here. The runtime only leaves PHYSICS_PARTIAL
    # phase checkpoints on disk; until 2026-08-11 nobody finalized the
    # file, so full-fidelity samples existed only inside result.json.
    # This is the current binding design, not legacy cruft.
    write_json_atomic(execution.output_file, trace)
    samples = _validate_trace(
        trace,
        branch="physical",
        plan=mechanism,
        load_case=load_case,
        spec=spec,
    )
    thresholds = _task_thresholds(mechanism)
    metrics = _metrics(samples, load_case, spec, thresholds)
    failures = [
        *_physical_semantic_failures(trace, spec),
        *_behavior_failures(
            metrics,
            load_case=load_case,
            spec=spec,
            thresholds=thresholds,
        ),
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
        "thresholds": {
            "powered_rotation_min_rad": plan.powered_rotation_min_rad,
            "angular_ratio": plan.angular_ratio,
            "angular_ratio_tolerance": plan.angular_ratio_tolerance,
            "return_to_home_max_rad": plan.return_to_home_max_rad,
        },
        "physical_metrics": metrics,
        "engine": trace["engine"],
        "execution_method": "certified_crank_torque_schedule_replay",
        "direct_output_actuation_used": False,
        "executed_phases": trace["executed_phases"],
        "executed_load_sha256": trace["executed_load_sha256"],
        "sample_count": len(samples),
        # 2026-08-11 ATTENTION AI EDITORS: do NOT embed raw samples in
        # this result payload. A full chain run carries 48,801 frames
        # (~300 MB), which blew the run assembly's 16 MiB JSON fuse on
        # r31-r33. Full-fidelity samples live in whole_machine.trace.json
        # (finalized above); this verdict document stays slim on purpose.
        "strict_non_claims": {
            "visual_pose_replay_used": False,
            "ideal_link_used_in_physical_branch": False,
            "startup_transient_hidden": False,
            "engagement_motion_credited": False,
            "engagement_ratio_credited": False,
        },
    }
