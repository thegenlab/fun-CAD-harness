from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    failure_envelope_from_exception,
    make_failure_envelope,
    mechanism_owner_for_registration_source,
)
from .bounded_support import (
    BoundedSupportRuntime,
    RELATIVE_ANCHOR_DRIFT,
    TRANSVERSE_ANCHOR_DRIFT,
    body_component_refs,
    bounded_support_gate,
    conditional_runtime_summary,
)
from .adapters import (
    AdapterSpec,
    MeshAssetSpec,
    RollingElementSetSpec,
    adapter_from_dict,
    get_adapter,
)
from .io import (
    MECHANISM_SOURCE_SHA256_ENV,
    canonical_sha256,
    ensure_inside,
    lane_execution_source_sha256,
    mechanism_source_sha256,
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
    write_trace_json_atomic,
)
from .collision_policy import (
    COMPLIANT_INTERFACE_PHYSICAL,
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from .conditional_interface import (
    build_functional_interface_geometry_binding,
)
from .family_plugins import (
    MechanismAdapter,
    invoke_mechanism_plugin_callback,
    mechanism_adapter_for,
    plugin_registration_for,
    wrench_policies_for_plugin,
)
from .task_plugins import (
    claimed_reviewer_task_programs,
    load_reviewer_task_adapter,
    task_adapter_for_program,
)
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import plugin_boundary_error
from .task_motion_evidence import (
    registered_adaptive_termination_evidence,
    registered_stop_comparator_reached,
)
from .wrench_ledger import (
    WrenchLedger,
    activate_wrench_ledger,
    add_ledgered_load,
    create_body_force,
    create_body_torque,
    current_wrench_audit_or_none,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)

_NSC_SOLVER_MAX_ITERATIONS = 200
_NSC_SOLVER_WARM_START = True
_AXIS_SPRING_LIMIT_STIFFNESS_N_M_RAD = 300.0
_AXIS_SPRING_LIMIT_DAMPING_N_M_S_RAD = 2.5


def _axis_spring_limit_response(
    angle_rad: float,
    angular_speed_rad_s: float,
    maximum_deflection_rad: float,
) -> tuple[float, float, str | None]:
    """Return finite stop torque, stop penetration, and active stop side."""

    if angle_rad < 0.0:
        penetration = -angle_rad
        damping = -_AXIS_SPRING_LIMIT_DAMPING_N_M_S_RAD * min(
            angular_speed_rad_s,
            0.0,
        )
        return (
            _AXIS_SPRING_LIMIT_STIFFNESS_N_M_RAD * penetration + damping,
            penetration,
            "lower",
        )
    if angle_rad > maximum_deflection_rad:
        penetration = angle_rad - maximum_deflection_rad
        damping = -_AXIS_SPRING_LIMIT_DAMPING_N_M_S_RAD * max(
            angular_speed_rad_s,
            0.0,
        )
        return (
            -_AXIS_SPRING_LIMIT_STIFFNESS_N_M_RAD * penetration + damping,
            penetration,
            "upper",
        )
    return 0.0, 0.0, None


def _collision_progress(event: str, **fields: Any) -> None:
    """Emit a durable, machine-readable breadcrumb before native collision calls.

    Native collision backends can occasionally block without raising a Python
    exception.  The queue runner retains stderr on timeout, so the final emitted
    breadcrumb identifies the exact body/component/native phase that stalled.
    """

    payload = {"kind": "collision_progress", "event": event, **fields}
    print("SIMREADY_COLLISION_PROGRESS " + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)


@dataclass(frozen=True)
class ExecutionRequest:
    request_file: Path
    output_file: Path
    branch: str
    artifact_branch: str
    adapter: AdapterSpec
    mechanism_plugin: MechanismAdapter
    plan: MechanismPlan
    load_case: LoadCase
    reference_trace_file: Path | None
    functional_geometry_evidence: dict[str, Any] | None = None

    def require_geometry_proof(
        self,
        joint_id: str,
        proof_kind: str,
    ) -> dict[str, Any]:
        evidence = self.functional_geometry_evidence
        joint_report = (
            evidence.get("joint_realization")
            if isinstance(evidence, dict)
            else None
        )
        proofs = (
            joint_report.get("proofs")
            if isinstance(joint_report, dict)
            else None
        )
        match = next(
            (
                item
                for item in proofs or ()
                if isinstance(item, dict)
                and item.get("joint_id") == joint_id
                and item.get("proof_kind") == proof_kind
            ),
            None,
        )
        artifact_hashes = getattr(self.plan, "artifact_hashes", {})
        expected_step_sha256 = (
            artifact_hashes.get("assembly.step")
            if isinstance(artifact_hashes, dict)
            else None
        )
        if (
            not isinstance(joint_report, dict)
            or joint_report.get("passed") is not True
            or not isinstance(expected_step_sha256, str)
            or joint_report.get("assembly_step_sha256")
            != expected_step_sha256
            or not isinstance(match, dict)
            or match.get("status") != "PASS"
        ):
            raise MechanismValidationError(
                "JOINT_REALIZATION_PROOF_REQUIRED",
                f"passing {proof_kind} proof is required for {joint_id}",
            )
        return dict(match)


def _trace_checkpoint_payload(
    execution: ExecutionRequest,
    samples: Sequence[dict[str, Any]],
    *,
    executed_phases: Sequence[dict[str, Any]] = (),
    resume_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative trace snapshot safe to retain on interruption."""

    if not samples:
        raise ValueError("trace checkpoint requires at least one sample")
    family_spec = (
        ((execution.plan.adapter_spec or {}).get("family_spec") or {})
        if execution.plan.adapter_spec is not None
        else execution.adapter.family_spec
    )
    collision_geometry = dict(family_spec.get("collision_geometry") or {})
    collision_geometry.setdefault(
        "contact_enabled",
        execution.branch == "physical",
    )
    declared_duration_s = sum(
        float(phase.duration_s) for phase in execution.load_case.phases
    )
    last_time_s = float(samples[-1].get("time_s") or 0.0)
    validator_source_sha256 = (
        os.environ.get(MECHANISM_SOURCE_SHA256_ENV)
        or lane_execution_source_sha256()
    )
    payload = {
        "schema_version": "0.1",
        "status": "PHYSICS_PARTIAL",
        "validator_source_sha256": validator_source_sha256,
        "branch": execution.artifact_branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "package_id": execution.plan.package_id,
        "design_version": execution.plan.design_version,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
        "engine": {
            "name": "PyChrono",
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "step_s": execution.load_case.step_s,
        },
        "collision_geometry": collision_geometry,
        "axis_abs": list(execution.plan.axis_abs),
        "joint_origin_abs_m": list(execution.plan.joint_origin_abs_m),
        "input_point_abs_m": list(execution.plan.input_point_abs_m),
        "observable_point_initial_abs_m": list(
            execution.plan.observable_point_abs_m
        ),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "executed_phases": list(executed_phases),
        "sample_count": len(samples),
        "samples": list(samples),
        "checkpoint": {
            "schema_version": "mechanism_trace_checkpoint_v1",
            "status": "IN_PROGRESS_AT_LAST_ATOMIC_WRITE",
            "atomic_write": True,
            "sample_count": len(samples),
            "last_sample_time_s": last_time_s,
            "last_phase_id": str(samples[-1].get("phase_id") or ""),
            "declared_duration_s": declared_duration_s,
            "completion_fraction": (
                min(1.0, max(0.0, last_time_s / declared_duration_s))
                if declared_duration_s > 0.0
                else 1.0
            ),
            "sample_interval_s": execution.load_case.sample_interval_s,
            "final_termination_reason": "UNKNOWN_UNTIL_WORKER_EXIT",
            "resumable": resume_state is not None,
            "resume_state": resume_state,
        },
        "visual_replay_semantics": {
            "trace_complete": False,
            "partial_checkpoint": True,
            "frames_are_saved_simulation_states": True,
            "certification_authority": False,
        },
        "strict_non_claims": {
            "complete_load_case": False,
            "certificate_authority": False,
            "manufacturing_certification": False,
            "strength_fatigue_or_wear": False,
            "dynamic_response_certified": False,
        },
    }
    wrench_audit = current_wrench_audit_or_none()
    if wrench_audit is not None:
        payload["wrench_audit"] = wrench_audit
    return payload


def _write_trace_checkpoint(
    execution: ExecutionRequest,
    samples: Sequence[dict[str, Any]],
    *,
    executed_phases: Sequence[dict[str, Any]] = (),
    resume_state: dict[str, Any] | None = None,
) -> None:
    """Atomically retain the latest sampled states before the worker can time out."""

    write_trace_json_atomic(
        execution.output_file,
        _trace_checkpoint_payload(
            execution,
            samples,
            executed_phases=executed_phases,
            resume_state=resume_state,
        ),
    )


def _load_trace_checkpoint_for_resume(
    execution: ExecutionRequest,
) -> dict[str, Any] | None:
    """Load a resumable partial trace only when every execution binding matches."""

    if not execution.output_file.is_file():
        return None
    payload = read_json(execution.output_file)
    if payload.get("status") != "PHYSICS_PARTIAL":
        return None
    expected = {
        "validator_source_sha256": (
            os.environ.get(MECHANISM_SOURCE_SHA256_ENV)
            or lane_execution_source_sha256()
        ),
        "branch": execution.artifact_branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
    }
    mismatches = [
        field_name
        for field_name, expected_value in expected.items()
        if payload.get(field_name) != expected_value
    ]
    if mismatches:
        raise MechanismValidationError(
            "TRACE_CHECKPOINT_BINDING_MISMATCH",
            "partial trace cannot be resumed because these bindings changed: "
            + ", ".join(sorted(mismatches)),
        )
    checkpoint = payload.get("checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != "mechanism_trace_checkpoint_v1"
        or checkpoint.get("resumable") is not True
        or not isinstance(checkpoint.get("resume_state"), dict)
        or not isinstance(payload.get("samples"), list)
        or not payload["samples"]
    ):
        raise MechanismValidationError(
            "TRACE_CHECKPOINT_RESUME_STATE_INVALID",
            "partial trace exists but does not contain a valid dynamics resume state",
        )
    return payload


def _verify_validator_source_binding() -> str:
    expected = os.environ.get(MECHANISM_SOURCE_SHA256_ENV)
    actual = lane_execution_source_sha256()
    if expected is None:
        raise MechanismValidationError(
            "VALIDATOR_SOURCE_BINDING_MISSING",
            "worker must run from a queue bound to the validator source tree",
        )
    if expected != actual:
        raise MechanismValidationError(
            "VALIDATOR_SOURCE_CHANGED",
            "validator source changed after queue planning; replan before running",
        )
    return actual

def _resolve_workspace_path(workspace: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str):
        raise MechanismValidationError(
            "REQUEST_PATH_INVALID", f"{label} must be a workspace-relative path"
        )
    try:
        return ensure_inside(workspace, relative)
    except ValueError as exc:
        raise MechanismValidationError(
            "REQUEST_PATH_INVALID", f"{label} is unsafe: {exc}"
        ) from exc


def _verify_plan_binding(plan: MechanismPlan, workspace: Path) -> None:
    package = Path(plan.package_dir).resolve()
    try:
        package.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "PACKAGE_OUTSIDE_WORKSPACE",
            "mechanism package is outside the queue workspace",
        ) from exc
    report = Path(plan.intake_report).resolve()
    try:
        report.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "INTAKE_REPORT_OUTSIDE_WORKSPACE",
            "intake report is outside the queue workspace",
        ) from exc
    if sha256_file(report) != plan.intake_report_sha256:
        raise MechanismValidationError(
            "INTAKE_REPORT_CHANGED",
            "intake report changed after mechanism discovery",
        )
    manifest_path = package / "package.json"
    manifest = read_json(manifest_path)
    actual_hashes: dict[str, str] = {}
    for relative, expected in sorted(plan.artifact_hashes.items()):
        path = ensure_inside(package, relative)
        if not path.is_file() or path.is_symlink():
            raise MechanismValidationError(
                "BOUND_ARTIFACT_MISSING", f"bound artifact is missing: {relative}"
            )
        digest = sha256_file(path)
        if digest != expected:
            raise MechanismValidationError(
                "BOUND_ARTIFACT_CHANGED",
                f"bound artifact changed before simulation: {relative}",
            )
        actual_hashes[relative] = digest
    binding = canonical_sha256(
        {
            "manifest_sha256": sha256_file(manifest_path),
            "artifact_hashes": actual_hashes,
            "requirements_ref": manifest.get("requirements_ref"),
        }
    )
    if binding != plan.package_binding_sha256:
        raise MechanismValidationError(
            "PACKAGE_BINDING_CHANGED",
            "package binding no longer matches the discovered mechanism plan",
        )


def _bind_request_reviewer_task_adapter(
    request: dict[str, Any],
    *,
    workspace: Path,
    plan: MechanismPlan,
    load_case: LoadCase,
) -> None:
    binding = request.get("reviewer_task_adapter")
    if binding is None:
        return
    if not isinstance(binding, dict) or set(binding) != {
        "plugin_id",
        "path",
        "sha256",
    }:
        raise MechanismValidationError(
            "REQUEST_REVIEWER_TASK_ADAPTER_INVALID",
            "reviewer task adapter request binding is malformed",
            failure=make_failure_envelope(
                "REQUEST_REVIEWER_TASK_ADAPTER_INVALID",
                FailureOwner.HARNESS_CODE,
                source_component="mechanism.worker.request",
                message="dispatch produced a malformed reviewer task binding",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            ),
        )
    plugin_id = str(binding.get("plugin_id") or "").strip()
    digest = str(binding.get("sha256") or "").strip()
    source_text = str(binding.get("path") or "").strip()
    if not plugin_id.startswith("reviewer.") or len(digest) != 64 or not source_text:
        raise MechanismValidationError(
            "REQUEST_REVIEWER_TASK_ADAPTER_INVALID",
            "reviewer task adapter request binding is incomplete",
            failure=make_failure_envelope(
                "REQUEST_REVIEWER_TASK_ADAPTER_INVALID",
                FailureOwner.HARNESS_CODE,
                source_component="mechanism.worker.request",
                message="dispatch produced an incomplete reviewer task binding",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            ),
        )
    source = _resolve_workspace_path(
        workspace, source_text, "reviewer_task_adapter.path"
    )
    try:
        plugin = load_reviewer_task_adapter(source, digest)
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise MechanismValidationError(
            exc.code,
            exc.message,
            details=exc.details,
            failure=make_failure_envelope(
                exc.code,
                FailureOwner.REVIEWER_TASK_PLUGIN,
                source_component="mechanism.worker.reviewer_task_adapter",
                message=exc.message,
                plugin_id=plugin_id,
                plugin_kind="task",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
                details=exc.details,
            ),
        ) from exc
    selected = task_adapter_for_program(load_case.program, plan=plan)
    if plugin.plugin_id != plugin_id or selected is not plugin:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_PROGRAM_UNSUPPORTED",
            f"reviewer task adapter {plugin_id!r} does not support "
            f"program {load_case.program!r} for this plan",
            details={"plugin_id": plugin_id, "program": load_case.program},
            failure=make_failure_envelope(
                "REVIEWER_TASK_ADAPTER_PROGRAM_UNSUPPORTED",
                FailureOwner.REVIEWER_TASK_PLUGIN,
                source_component="mechanism.worker.reviewer_task_adapter",
                message="bound reviewer task adapter does not support this program",
                plugin_id=plugin_id,
                plugin_kind="task",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
                details={"plugin_id": plugin_id, "program": load_case.program},
            ),
        )


def load_execution_request(
    request_file: str | Path,
    *,
    workspace_root: str | Path,
) -> ExecutionRequest:
    workspace = Path(workspace_root).resolve()
    raw_request_path = Path(request_file)
    if raw_request_path.is_absolute():
        request_path = raw_request_path.resolve()
        try:
            request_path.relative_to(workspace)
        except ValueError as exc:
            raise MechanismValidationError(
                "REQUEST_OUTSIDE_WORKSPACE",
                "request file must stay inside the queue workspace",
            ) from exc
    else:
        request_path = _resolve_workspace_path(
            workspace, raw_request_path.as_posix(), "request"
        )
    request = read_json(request_path)
    allowed = {
        "schema_version",
        "branch",
        "adapter_id",
        "mechanism_plugin_id",
        "package_binding_sha256",
        "plan_file",
        "plan_sha256",
        "load_case_file",
        "load_case_sha256",
        "initial_functional_geometry_file",
        "initial_functional_geometry_sha256",
        "output_file",
        "reference_trace_file",
        "reviewer_task_adapter",
    }
    unknown = sorted(set(request) - allowed)
    if unknown:
        raise MechanismValidationError(
            "REQUEST_FIELD_UNKNOWN",
            "request contains unknown fields: " + ", ".join(unknown),
        )
    if request.get("schema_version") != "0.1":
        raise MechanismValidationError(
            "REQUEST_SCHEMA_UNSUPPORTED", "request schema_version must be 0.1"
        )
    artifact_branch = request.get("branch")
    if artifact_branch not in {
        "physical",
        "reduced",
        "ideal_dynamics",
    }:
        raise MechanismValidationError(
            "REQUEST_BRANCH_INVALID",
            "branch must be physical, ideal_dynamics, or legacy reduced",
        )
    branch = (
        "reduced" if artifact_branch == "ideal_dynamics"
        else artifact_branch
    )
    plan_path = _resolve_workspace_path(workspace, request.get("plan_file"), "plan")
    load_path = _resolve_workspace_path(
        workspace, request.get("load_case_file"), "load_case"
    )
    output_path = _resolve_workspace_path(
        workspace, request.get("output_file"), "output"
    )
    reference_value = request.get("reference_trace_file")
    reference_path = (
        _resolve_workspace_path(workspace, reference_value, "reference_trace")
        if reference_value is not None
        else None
    )
    initial_geometry_value = request.get(
        "initial_functional_geometry_file"
    )
    initial_geometry_sha256 = request.get(
        "initial_functional_geometry_sha256"
    )
    if not isinstance(initial_geometry_value, str) or not isinstance(
        initial_geometry_sha256, str
    ):
        raise MechanismValidationError(
            "REQUEST_FUNCTIONAL_GEOMETRY_REQUIRED",
            "request must bind one initial functional-geometry report",
        )
    initial_geometry_path = _resolve_workspace_path(
        workspace,
        initial_geometry_value,
        "initial_functional_geometry",
    )
    if sha256_file(initial_geometry_path) != initial_geometry_sha256:
        raise MechanismValidationError(
            "REQUEST_FUNCTIONAL_GEOMETRY_HASH_MISMATCH",
            "initial functional-geometry report changed after dispatch",
        )
    initial_geometry = read_json(initial_geometry_path)
    if branch == "reduced" and reference_path is None:
        raise MechanismValidationError(
            "REFERENCE_TRACE_REQUIRED",
            "reduced branch must replay the physical branch load schedule",
        )
    if branch == "physical" and reference_path is not None:
        raise MechanismValidationError(
            "REFERENCE_TRACE_FORBIDDEN",
            "physical branch cannot consume a branch reference trace",
        )
    plan = MechanismPlan.from_dict(read_json(plan_path))
    load_case = LoadCase.from_dict(read_json(load_path))
    if request.get("plan_sha256") != canonical_sha256(plan.to_dict()):
        raise MechanismValidationError(
            "REQUEST_PLAN_HASH_MISMATCH",
            "request plan content changed after dispatch",
        )
    if request.get("load_case_sha256") != canonical_sha256(
        load_case.to_dict()
    ):
        raise MechanismValidationError(
            "REQUEST_LOAD_CASE_HASH_MISMATCH",
            "request load-case content changed after dispatch",
        )
    if request.get("package_binding_sha256") != plan.package_binding_sha256:
        raise MechanismValidationError(
            "REQUEST_BINDING_MISMATCH",
            "request and mechanism plan package bindings differ",
        )
    base_geometry_report_invalid = (
        initial_geometry.get("schema_version")
        != "initial_functional_geometry_preflight_v1"
        or initial_geometry.get("capsule_id") != plan.capsule_id
        or initial_geometry.get("package_binding_sha256")
        != plan.package_binding_sha256
        or initial_geometry.get("passed") is not True
    )
    if base_geometry_report_invalid:
        raise MechanismValidationError(
            "REQUEST_FUNCTIONAL_GEOMETRY_INVALID",
            "initial functional-geometry report does not match the mechanism plan",
        )
    adapter_id = request.get("adapter_id")
    if adapter_id != plan.adapter_id:
        raise MechanismValidationError(
            "REQUEST_ADAPTER_MISMATCH", "request adapter differs from the plan"
        )
    try:
        adapter = (
            adapter_from_dict(plan.adapter_spec)
            if plan.adapter_spec is not None
            else get_adapter(str(adapter_id))
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise MechanismValidationError(
            "REQUEST_ADAPTER_UNKNOWN", str(exc)
        ) from exc
    if adapter.family != plan.family or load_case.family != plan.family:
        raise MechanismValidationError(
            "REQUEST_FAMILY_MISMATCH",
            "adapter, plan, and load case families must match",
        )
    plugin = mechanism_adapter_for(plan, adapter)
    if request.get("mechanism_plugin_id") != plugin.plugin_id:
        raise MechanismValidationError(
            "REQUEST_MECHANISM_PLUGIN_MISMATCH",
            "request mechanism plugin differs from the trusted registry",
        )
    if type(plugin).simulate is MechanismAdapter.simulate:
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            f"worker backend {plugin.worker_backend!r} is not implemented",
        )
    missing_geometry_evidence = [
        key
        for key in sorted(plugin.initial_geometry_evidence_keys)
        if not isinstance(initial_geometry.get(key), dict)
    ]
    if missing_geometry_evidence:
        raise MechanismValidationError(
            "REQUEST_FUNCTIONAL_GEOMETRY_INVALID",
            "initial functional-geometry report does not match the mechanism plan",
        )
    _bind_request_reviewer_task_adapter(
        request,
        workspace=workspace,
        plan=plan,
        load_case=load_case,
    )
    if (
        load_case.program not in adapter.supported_programs
        and load_case.program not in claimed_reviewer_task_programs(plan)
    ):
        raise MechanismValidationError(
            "REQUEST_PROGRAM_UNSUPPORTED",
            f"adapter does not support load program {load_case.program!r}",
            details={
                "mechanism_plugin_id": plugin.plugin_id,
                "program": load_case.program,
            },
            failure=make_failure_envelope(
                "REQUEST_PROGRAM_UNSUPPORTED",
                FailureOwner.HARNESS_CODE,
                source_component="mechanism.worker.request",
                message="dispatch admitted a program without a bound runtime owner",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
                details={
                    "mechanism_plugin_id": plugin.plugin_id,
                    "program": load_case.program,
                },
            ),
        )
    if adapter.geometry is None and adapter.family_spec is None:
        raise MechanismValidationError(
            "ADAPTER_GEOMETRY_MISSING",
            "adapter has neither two-body geometry nor family-owned execution data",
        )
    try:
        plugin_registration = plugin_registration_for(str(plugin.plugin_id))
    except MechanismValidationError:
        plugin_registration = None
    try:
        plugin.validate_family_spec(adapter)
    except MechanismValidationError as exc:
        boundary_error = plugin_boundary_error(
            exc,
            registration=plugin_registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="validate_family_spec",
            source_component="mechanism.worker.validate_family_spec",
        )
        if boundary_error is exc:
            raise
        raise boundary_error from exc
    except Exception as exc:
        code = "MECHANISM_ADAPTER_FAMILY_SPEC_VALIDATION_FAILED"
        message = f"mechanism adapter validate_family_spec() failed: {exc}"
        raise plugin_boundary_error(
            exc,
            registration=plugin_registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="validate_family_spec",
            source_component="mechanism.worker.validate_family_spec",
            code=code,
            message=message,
            evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
        ) from exc
    _verify_plan_binding(plan, workspace)
    if branch == "physical":
        from .collider_fidelity import verify_collider_fidelity_binding
        from .rigid_attachment import verify_rigid_attachment_binding

        verify_collider_fidelity_binding(
            adapter.collider_fidelity,
            workspace_root=workspace,
            package_binding_sha256=plan.package_binding_sha256,
        )
        verify_rigid_attachment_binding(
            (adapter.collider_fidelity or {}).get("rigid_attachment_proof"),
            workspace_root=workspace,
            package_binding_sha256=plan.package_binding_sha256,
        )
    required_hashes = {value.casefold() for value in plan.source_component_hashes.values()}
    if not adapter.required_source_hashes.issubset(required_hashes):
        raise MechanismValidationError(
            "ADAPTER_SOURCE_MISMATCH",
            "package source-component hashes do not satisfy the adapter binding",
        )
    return ExecutionRequest(
        request_file=request_path,
        output_file=output_path,
        branch=branch,
        artifact_branch=artifact_branch,
        adapter=adapter,
        mechanism_plugin=plugin,
        plan=plan,
        load_case=load_case,
        reference_trace_file=reference_path,
        functional_geometry_evidence=(
            dict(initial_geometry["bicycle_chain_geometry"])
            if isinstance(
                initial_geometry.get("bicycle_chain_geometry"),
                dict,
            )
            # ATTENTION AI EDITORS (added 2026-08-07, CURRENT binding rule,
            # NOT a legacy note — do not ignore): shape protocol with the
            # emitter in initial_functional_geometry.py. Chain reports carry
            # the nested key; other backends get the whole flat report.
            # Update BOTH ends in the same change when touching the layout.
            # Origin: commit f1cd3a8.
            else dict(initial_geometry)
            if initial_geometry.get("passed") is True
            else None
        ),
    )


def _matrix_vector(
    matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )


def _read_obj(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
    ):
        if line.startswith("v "):
            fields = line.split()
            if len(fields) < 4:
                raise MechanismValidationError(
                    "COLLIDER_VERTEX_INVALID", f"invalid vertex at {path}:{line_number}"
                )
            vertex = tuple(float(value) for value in fields[1:4])
            if any(not math.isfinite(value) for value in vertex):
                raise MechanismValidationError(
                    "COLLIDER_VERTEX_INVALID", f"non-finite vertex at {path}:{line_number}"
                )
            vertices.append(vertex)
        elif line.startswith("f "):
            fields = line.split()[1:]
            if len(fields) < 3:
                raise MechanismValidationError(
                    "COLLIDER_FACE_INVALID", f"invalid face at {path}:{line_number}"
                )
            indices = [int(value.split("/")[0]) - 1 for value in fields]
            if any(index < 0 for index in indices):
                raise MechanismValidationError(
                    "COLLIDER_FACE_INVALID", "negative OBJ indices are not accepted"
                )
            for offset in range(1, len(indices) - 1):
                faces.append((indices[0], indices[offset], indices[offset + 1]))
    if not vertices or not faces:
        raise MechanismValidationError(
            "COLLIDER_EMPTY", f"collider has no triangle geometry: {path}"
        )
    if max(index for face in faces for index in face) >= len(vertices):
        raise MechanismValidationError(
            "COLLIDER_INDEX_INVALID", f"collider face index is out of range: {path}"
        )
    return vertices, faces


def _axis_aligned_box_spec(
    path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Recognize a complete axis-aligned box mesh without approximating it."""
    vertices, faces = _read_obj(path)
    if len(faces) != 12:
        return None
    minimum = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maximum = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    lengths = tuple(maximum[axis] - minimum[axis] for axis in range(3))
    if any(length <= 0.0 for length in lengths):
        return None
    tolerance = max(1e-10, max(lengths) * 1e-8)

    corner_keys: set[tuple[int, int, int]] = set()
    for vertex in vertices:
        key: list[int] = []
        for axis, value in enumerate(vertex):
            if abs(value - minimum[axis]) <= tolerance:
                key.append(0)
            elif abs(value - maximum[axis]) <= tolerance:
                key.append(1)
            else:
                return None
        corner_keys.add(tuple(key))
    if corner_keys != {
        (x_side, y_side, z_side)
        for x_side in (0, 1)
        for y_side in (0, 1)
        for z_side in (0, 1)
    }:
        return None

    face_counts = {(axis, side): 0 for axis in range(3) for side in (0, 1)}
    for face in faces:
        points = [vertices[index] for index in face]
        first_edge = tuple(points[1][axis] - points[0][axis] for axis in range(3))
        second_edge = tuple(points[2][axis] - points[0][axis] for axis in range(3))
        cross = (
            first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
            first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
        )
        if sum(value * value for value in cross) <= tolerance**4:
            return None
        boundaries: list[tuple[int, int]] = []
        for axis in range(3):
            if all(
                abs(point[axis] - minimum[axis]) <= tolerance for point in points
            ):
                boundaries.append((axis, 0))
            elif all(
                abs(point[axis] - maximum[axis]) <= tolerance for point in points
            ):
                boundaries.append((axis, 1))
        if len(boundaries) != 1:
            return None
        face_counts[boundaries[0]] += 1
    if any(count != 2 for count in face_counts.values()):
        return None

    center = tuple((minimum[axis] + maximum[axis]) * 0.5 for axis in range(3))
    return lengths, center


def _axis_aligned_y_cylinder_spec(
    path: Path,
) -> tuple[float, float, tuple[float, float, float]] | None:
    vertices, faces = _read_obj(path)
    if len(vertices) < 16 or len(faces) < 24:
        return None
    y_values = sorted({round(vertex[1], 10) for vertex in vertices})
    if len(y_values) != 2:
        return None
    height = y_values[1] - y_values[0]
    if height <= 0.0:
        return None
    min_x = min(vertex[0] for vertex in vertices)
    max_x = max(vertex[0] for vertex in vertices)
    min_z = min(vertex[2] for vertex in vertices)
    max_z = max(vertex[2] for vertex in vertices)
    center_x = 0.5 * (min_x + max_x)
    center_z = 0.5 * (min_z + max_z)
    radii = [
        math.hypot(vertex[0] - center_x, vertex[2] - center_z)
        for vertex in vertices
    ]
    radius = max(radii)
    tolerance = max(1e-9, radius * 2e-5)
    if radius <= 0.0 or any(
        value > tolerance and abs(value - radius) > tolerance
        for value in radii
    ):
        return None
    outer_vertices = [
        vertex for vertex, value in zip(vertices, radii)
        if abs(value - radius) <= tolerance
    ]
    angles = sorted(
        {
            math.atan2(vertex[2] - center_z, vertex[0] - center_x)
            % (2.0 * math.pi)
            for vertex in outer_vertices
        }
    )
    cyclic_gaps = [
        angles[(index + 1) % len(angles)]
        - angles[index]
        if index + 1 < len(angles)
        else angles[0] + 2.0 * math.pi - angles[index]
        for index in range(len(angles))
    ]
    if len(angles) < 8 or max(cyclic_gaps) > math.pi / 2.0:
        return None
    cap_levels = {y_values[0], y_values[1]}
    if any(round(vertex[1], 10) not in cap_levels for vertex in vertices):
        return None
    return radius, height, (center_x, 0.5 * sum(y_values), center_z)


def _sphere_spec(
    path: Path,
) -> tuple[float, tuple[float, float, float]] | None:
    vertices, faces = _read_obj(path)
    if len(vertices) < 12 or len(faces) < 20:
        return None
    minimum = tuple(min(vertex[index] for vertex in vertices) for index in range(3))
    maximum = tuple(max(vertex[index] for vertex in vertices) for index in range(3))
    origin = tuple(0.5 * (minimum[index] + maximum[index]) for index in range(3))

    # Fit the analytic sphere instead of treating the tessellation bounds as
    # exact.  OCCT meshes do not promise samples at every axial extremum, so a
    # bounds midpoint can be tens of micrometres away from the true centre.
    normal = [[0.0 for _ in range(4)] for _ in range(4)]
    right = [0.0 for _ in range(4)]
    for vertex in vertices:
        offset = [vertex[index] - origin[index] for index in range(3)]
        row = [2.0 * offset[0], 2.0 * offset[1], 2.0 * offset[2], 1.0]
        value = sum(component * component for component in offset)
        for row_index in range(4):
            right[row_index] += row[row_index] * value
            for column_index in range(4):
                normal[row_index][column_index] += (
                    row[row_index] * row[column_index]
                )

    augmented = [normal[index] + [right[index]] for index in range(4)]
    for column in range(4):
        pivot = max(range(column, 4), key=lambda index: abs(augmented[index][column]))
        if abs(augmented[pivot][column]) <= 1e-18:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row_index in range(4):
            if row_index == column:
                continue
            factor = augmented[row_index][column]
            augmented[row_index] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row_index], augmented[column]
                )
            ]
    fitted = [augmented[index][4] for index in range(4)]
    center = tuple(origin[index] + fitted[index] for index in range(3))
    radii = [
        math.sqrt(sum((vertex[index] - center[index]) ** 2 for index in range(3)))
        for vertex in vertices
    ]
    radius = sum(radii) / len(radii)
    tolerance = max(1e-9, radius * 3e-5)
    if radius <= 0.0 or any(abs(value - radius) > tolerance for value in radii):
        return None
    axis_reach = [
        center[index] - minimum[index]
        for index in range(3)
    ] + [
        maximum[index] - center[index]
        for index in range(3)
    ]
    if any(value < radius * 0.85 or value > radius * 1.02 for value in axis_reach):
        return None
    return radius, center


def _oriented_box_spec(
    path: Path,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
] | None:
    axis_aligned = _axis_aligned_box_spec(path)
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    if axis_aligned is not None:
        lengths, center = axis_aligned
        return lengths, center, identity

    from itertools import combinations, product

    vertices, faces = _read_obj(path)
    if len(vertices) != 8 or not faces:
        return None

    def subtract(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return tuple(a - b for a, b in zip(left, right))

    def add(*vectors: tuple[float, float, float]) -> tuple[float, float, float]:
        return tuple(sum(vector[index] for vector in vectors) for index in range(3))

    def scale(
        vector: tuple[float, float, float], value: float
    ) -> tuple[float, float, float]:
        return tuple(component * value for component in vector)

    def dot(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    def norm(vector: tuple[float, float, float]) -> float:
        return math.sqrt(dot(vector, vector))

    scale_hint = max(
        norm(subtract(left, right))
        for left in vertices for right in vertices
    )
    tolerance = max(1e-9, scale_hint * 2e-6)

    def close_point(
        left: tuple[float, float, float],
        right: tuple[float, float, float],
    ) -> bool:
        return norm(subtract(left, right)) <= tolerance

    for origin in vertices:
        candidates = [
            subtract(vertex, origin)
            for vertex in vertices
            if not close_point(vertex, origin)
        ]
        for edge_a, edge_b, edge_c in combinations(candidates, 3):
            lengths = [norm(edge_a), norm(edge_b), norm(edge_c)]
            if min(lengths) <= tolerance:
                continue
            orthogonal = all(
                abs(dot(left, right))
                <= tolerance * max(norm(left), norm(right))
                for left, right in (
                    (edge_a, edge_b),
                    (edge_a, edge_c),
                    (edge_b, edge_c),
                )
            )
            if not orthogonal:
                continue
            expected = [
                add(
                    origin,
                    scale(edge_a, float(bit_a)),
                    scale(edge_b, float(bit_b)),
                    scale(edge_c, float(bit_c)),
                )
                for bit_a, bit_b, bit_c in product((0, 1), repeat=3)
            ]
            if not all(
                any(close_point(point, vertex) for vertex in vertices)
                for point in expected
            ):
                continue
            axes = [
                scale(edge_a, 1.0 / lengths[0]),
                scale(edge_b, 1.0 / lengths[1]),
                scale(edge_c, 1.0 / lengths[2]),
            ]
            if dot(cross(axes[0], axes[1]), axes[2]) < 0.0:
                axes[1], axes[2] = axes[2], axes[1]
                lengths[1], lengths[2] = lengths[2], lengths[1]
            triangle_area = 0.0
            faces_valid = True
            for face in faces:
                points = [vertices[index] for index in face]
                triangle_area += 0.5 * norm(
                    cross(
                        subtract(points[1], points[0]),
                        subtract(points[2], points[0]),
                    )
                )
                local = [
                    [dot(subtract(point, origin), axis) for axis in axes]
                    for point in points
                ]
                if not any(
                    all(abs(item[axis_index]) <= tolerance for item in local)
                    or all(
                        abs(item[axis_index] - lengths[axis_index]) <= tolerance
                        for item in local
                    )
                    for axis_index in range(3)
                ):
                    faces_valid = False
                    break
            expected_area = 2.0 * (
                lengths[0] * lengths[1]
                + lengths[0] * lengths[2]
                + lengths[1] * lengths[2]
            )
            if (
                not faces_valid
                or abs(triangle_area - expected_area)
                > max(1e-12, expected_area * 2e-5)
            ):
                continue
            center = add(
                origin,
                scale(edge_a, 0.5),
                scale(edge_b, 0.5),
                scale(edge_c, 0.5),
            )
            rotation = (
                (axes[0][0], axes[1][0], axes[2][0]),
                (axes[0][1], axes[1][1], axes[2][1]),
                (axes[0][2], axes[1][2], axes[2][2]),
            )
            return tuple(lengths), center, rotation
    return None


def _quat_from_rotation_matrix(chrono: Any, matrix: Sequence[Sequence[float]]) -> Any:
    m00, m01, m02 = (float(value) for value in matrix[0])
    m10, m11, m12 = (float(value) for value in matrix[1])
    m20, m21, m22 = (float(value) for value in matrix[2])
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w_value = 0.25 * scale
        x_value = (m21 - m12) / scale
        y_value = (m02 - m20) / scale
        z_value = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w_value = (m21 - m12) / scale
        x_value = 0.25 * scale
        y_value = (m01 + m10) / scale
        z_value = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w_value = (m02 - m20) / scale
        x_value = (m01 + m10) / scale
        y_value = 0.25 * scale
        z_value = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w_value = (m10 - m01) / scale
        x_value = (m02 + m20) / scale
        y_value = (m12 + m21) / scale
        z_value = 0.25 * scale
    return chrono.ChQuaterniond(w_value, x_value, y_value, z_value)


def _write_body_local_mesh(
    asset: MeshAssetSpec,
    *,
    workspace: Path,
    destination: Path,
    body_origin_abs_m: tuple[float, float, float],
) -> dict[str, Any]:
    source = _resolve_workspace_path(
        workspace, asset.workspace_relative_path, "collider_asset"
    )
    if not source.is_file() or source.is_symlink():
        raise MechanismValidationError(
            "COLLIDER_ASSET_MISSING", f"trusted collider is missing: {source}"
        )
    if sha256_file(source) != asset.sha256:
        raise MechanismValidationError(
            "COLLIDER_ASSET_CHANGED", f"trusted collider hash changed: {source}"
        )
    vertices, faces = _read_obj(source)
    transformed: list[tuple[float, float, float]] = []
    for vertex in vertices:
        scaled = tuple(value * asset.units_to_m for value in vertex)
        world = _matrix_vector(asset.rotation_abs_from_source, scaled)
        transformed.append(
            tuple(world[index] - body_origin_abs_m[index] for index in range(3))
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# trusted meter-scale body-local nonconvex triangle mesh\n")
        for x_value, y_value, z_value in transformed:
            stream.write(f"v {x_value:.12g} {y_value:.12g} {z_value:.12g}\n")
        for first, second, third in faces:
            stream.write(f"f {first + 1} {second + 1} {third + 1}\n")
    return {
        "source": relative_to_root(workspace, source),
        "source_sha256": asset.sha256,
        "derived": relative_to_root(workspace, destination),
        "derived_sha256": sha256_file(destination),
        "vertex_count": len(vertices),
        "triangle_count": len(faces),
        "units_to_m": asset.units_to_m,
        "rotation_abs_from_source": [list(row) for row in asset.rotation_abs_from_source],
        "body_origin_abs_m": list(body_origin_abs_m),
        "convex_hull_used": False,
    }


def _rolling_element_assets(
    rolling: RollingElementSetSpec,
    *,
    workspace: Path,
) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    for element in rolling.elements:
        source = _resolve_workspace_path(
            workspace, element.evidence_mesh_path, "rolling_element_asset"
        )
        if not source.is_file() or source.is_symlink():
            raise MechanismValidationError(
                "ROLLING_ELEMENT_ASSET_MISSING",
                f"trusted rolling-element evidence is missing: {source}",
            )
        if sha256_file(source) != element.evidence_mesh_sha256:
            raise MechanismValidationError(
                "ROLLING_ELEMENT_ASSET_CHANGED",
                f"trusted rolling-element evidence hash changed: {source}",
            )
        vertices, faces = _read_obj(source)
        radial_distances_m = [
            math.sqrt(
                sum(
                    (
                        vertex[index] - element.center_source_units[index]
                    )
                    ** 2
                    for index in range(3)
                )
            )
            * rolling.units_to_m
            for vertex in vertices
        ]
        measured_radius_m = sum(radial_distances_m) / len(radial_distances_m)
        if abs(measured_radius_m - rolling.radius_m) > 1e-6:
            raise MechanismValidationError(
                "ROLLING_ELEMENT_RADIUS_MISMATCH",
                f"{element.element_id} evidence does not match the bound radius",
            )
        source_center_m = tuple(
            value * rolling.units_to_m for value in element.center_source_units
        )
        center_abs_m = _matrix_vector(
            rolling.rotation_abs_from_source, source_center_m
        )
        assets.append(
            {
                "element_id": element.element_id,
                "source": relative_to_root(workspace, source),
                "source_sha256": element.evidence_mesh_sha256,
                "source_vertex_count": len(vertices),
                "source_triangle_count": len(faces),
                "center_abs_m": list(center_abs_m),
                "radius_m": rolling.radius_m,
                "measured_mean_radius_m": measured_radius_m,
                "runtime_collision_shape": "ChCollisionShapeSphere",
                "convex_hull_used": False,
                "primitive_proxy_used": False,
            }
        )
    return assets


def _inertia_box(
    mass_kg: float, extents_m: tuple[float, float, float]
) -> tuple[float, float, float]:
    x_value, y_value, z_value = extents_m
    return (
        mass_kg * (y_value**2 + z_value**2) / 12.0,
        mass_kg * (x_value**2 + z_value**2) / 12.0,
        mass_kg * (x_value**2 + y_value**2) / 12.0,
    )


def _quat_from_z_to_axis(chrono: Any, axis: tuple[float, float, float]) -> Any:
    dot = max(-1.0, min(1.0, axis[2]))
    if dot < -1.0 + 1e-12:
        return chrono.ChQuaterniond(0.0, 1.0, 0.0, 0.0)
    cross = (-axis[1], axis[0], 0.0)
    scale = math.sqrt(2.0 * (1.0 + dot))
    return chrono.ChQuaterniond(
        0.5 * scale,
        cross[0] / scale,
        cross[1] / scale,
        cross[2] / scale,
    )


def _tuple3(vector: Any) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _pose(body: Any) -> dict[str, list[float]]:
    position = body.GetPos()
    rotation = body.GetRot()
    return {
        "position_abs_m": _tuple3(position),
        "rotation_abs_quat_wxyz": [
            float(rotation.e0),
            float(rotation.e1),
            float(rotation.e2),
            float(rotation.e3),
        ],
    }


def _quat_angle(
    initial: tuple[float, float, float, float], current: list[float]
) -> float:
    dot = abs(sum(left * right for left, right in zip(initial, current)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _make_material(chrono: Any, friction: float) -> Any:
    material = chrono.ChContactMaterialNSC()
    if not hasattr(material, "SetFriction"):
        raise MechanismValidationError(
            "CONTACT_MATERIAL_CONFIGURATION_FAILED",
            "PyChrono contact material does not expose SetFriction",
        )
    for method_name, value in (
        ("SetFriction", friction),
        ("SetRestitution", 0.0),
        ("SetDampingF", 0.2),
        ("SetCompliance", 0.0),
        ("SetComplianceT", 0.0),
    ):
        if hasattr(material, method_name):
            try:
                getattr(material, method_name)(value)
            except Exception as exc:
                raise MechanismValidationError(
                    "CONTACT_MATERIAL_CONFIGURATION_FAILED",
                    f"PyChrono contact material rejected {method_name}",
                ) from exc
    return material


def _make_nsc_system(
    chrono: Any, *, collision_envelope_m: float, collision_margin_m: float
) -> Any:
    # ChSystemNSC construction resets these process-wide defaults.
    system = chrono.ChSystemNSC()
    chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(collision_envelope_m)
    chrono.ChCollisionModel.SetDefaultSuggestedMargin(collision_margin_m)
    if hasattr(system, "SetCollisionSystemType") and hasattr(
        chrono, "ChCollisionSystem"
    ):
        system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    solver = system.GetSolver() if hasattr(system, "GetSolver") else None
    iterative = (
        solver.AsIterative()
        if solver is not None and hasattr(solver, "AsIterative")
        else None
    )
    if iterative is not None:
        if hasattr(iterative, "SetMaxIterations"):
            iterative.SetMaxIterations(_NSC_SOLVER_MAX_ITERATIONS)
        if hasattr(iterative, "EnableWarmStart"):
            iterative.EnableWarmStart(_NSC_SOLVER_WARM_START)
    return system


def _configure_prismatic_travel_limit(
    prismatic: Any,
    max_progress_m: float | None,
    *,
    min_progress_m: float = 0.0,
) -> dict[str, Any] | None:
    if max_progress_m is None:
        return None
    if (
        not math.isfinite(min_progress_m)
        or not math.isfinite(max_progress_m)
        or max_progress_m <= min_progress_m
    ):
        raise RuntimeError("prismatic travel limits must be finite and ordered")
    if not hasattr(prismatic, "LimitZ"):
        raise RuntimeError("PyChrono prismatic link does not expose its free-axis limit")
    limit = prismatic.LimitZ()
    limit.SetActive(True)
    limit.SetMin(float(min_progress_m))
    limit.SetMax(float(max_progress_m))
    return {
        "coordinate": "joint_frame_z",
        "min_m": float(min_progress_m),
        "max_m": float(max_progress_m),
        "source": "physical_contact_certifiable_progress_limit",
    }


def _configure_revolute_angle_limit(
    revolute: Any,
    max_angle_rad: float | None,
    *,
    min_angle_rad: float = 0.0,
) -> dict[str, Any] | None:
    if max_angle_rad is None:
        return None
    if (
        not math.isfinite(min_angle_rad)
        or not math.isfinite(max_angle_rad)
        or max_angle_rad <= min_angle_rad
    ):
        raise RuntimeError("revolute angle limits must be finite and ordered")
    if not hasattr(revolute, "LimitRz"):
        raise RuntimeError("PyChrono revolute link does not expose its free-axis limit")
    limit = revolute.LimitRz()
    limit.SetActive(True)
    limit.SetMin(float(min_angle_rad))
    limit.SetMax(float(max_angle_rad))
    return {
        "coordinate": "joint_frame_rz",
        "min_rad": float(min_angle_rad),
        "max_rad": float(max_angle_rad),
        "source": "physical_contact_certifiable_angle_limit",
    }


def _signed_axis_angle(
    quaternion_wxyz: Sequence[float],
    axis: tuple[float, float, float],
) -> float:
    w_value = float(quaternion_wxyz[0])
    vector = tuple(float(value) for value in quaternion_wxyz[1:4])
    projected = sum(value * direction for value, direction in zip(vector, axis))
    return 2.0 * math.atan2(projected, w_value)


def _off_axis_angle(
    quaternion_wxyz: Sequence[float],
    axis: tuple[float, float, float],
) -> float:
    vector = tuple(float(value) for value in quaternion_wxyz[1:4])
    projected = sum(value * direction for value, direction in zip(vector, axis))
    perpendicular = tuple(
        vector[index] - projected * axis[index] for index in range(3)
    )
    magnitude = math.sqrt(sum(value * value for value in perpendicular))
    return 2.0 * math.asin(max(0.0, min(1.0, magnitude)))


def _make_body(
    chrono: Any,
    *,
    name: str,
    local_meshes: Sequence[tuple[Path, bool]],
    origin_abs_m: tuple[float, float, float],
    fixed: bool,
    mass_kg: float,
    inertia_xx: tuple[float, float, float],
    material: Any,
    collision_radius_m: float,
    physical_contact_required: bool,
) -> Any:
    body = chrono.ChBodyAuxRef()
    body.SetName(name)
    body.SetFixed(fixed)
    body.SetPos(chrono.ChVector3d(*origin_abs_m))
    body.SetRot(chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0))
    body.SetMass(mass_kg)
    body.SetInertiaXX(chrono.ChVector3d(*inertia_xx))
    if not local_meshes:
        raise RuntimeError(f"body {name!r} has no collision meshes")
    _collision_progress("body_build_begin", body_id=name, component_count=len(local_meshes))
    for component_index, (local_mesh, convex) in enumerate(local_meshes):
        progress = {
            "body_id": name,
            "component_index": component_index,
            "component_count": len(local_meshes),
            "mesh": local_mesh.name,
            "convex": bool(convex),
        }
        _collision_progress("component_classify_begin", **progress)
        sphere_spec = _sphere_spec(local_mesh)
        if sphere_spec is not None:
            radius, center = sphere_spec
            _collision_progress("component_attach_sphere_begin", **progress)
            body.AddCollisionShape(
                chrono.ChCollisionShapeSphere(material, radius),
                chrono.ChFramed(
                    chrono.ChVector3d(*center),
                    chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0),
                ),
            )
            _collision_progress("component_attach_complete", representation="sphere", **progress)
            continue
        cylinder_spec = _axis_aligned_y_cylinder_spec(local_mesh)
        if cylinder_spec is not None:
            radius, height, center = cylinder_spec
            _collision_progress("component_attach_cylinder_begin", **progress)
            body.AddCollisionShape(
                chrono.ChCollisionShapeCylinder(material, radius, height),
                chrono.ChFramed(
                    chrono.ChVector3d(*center),
                    _quat_from_z_to_axis(chrono, (0.0, 1.0, 0.0)),
                ),
            )
            _collision_progress("component_attach_complete", representation="cylinder", **progress)
            continue
        box_spec = _oriented_box_spec(local_mesh)
        if box_spec is not None:
            lengths, center, rotation = box_spec
            _collision_progress("component_attach_box_begin", **progress)
            body.AddCollisionShape(
                chrono.ChCollisionShapeBox(material, *lengths),
                chrono.ChFramed(
                    chrono.ChVector3d(*center),
                    _quat_from_rotation_matrix(chrono, rotation),
                ),
            )
            _collision_progress("component_attach_complete", representation="box", **progress)
            continue
        if physical_contact_required and not fixed and not convex:
            raise MechanismValidationError(
                "DYNAMIC_COLLIDER_REPRESENTATION_UNSUPPORTED",
                "dynamic contact components must be verified boxes or a bounded "
                f"convex decomposition; {local_mesh} is not convex",
                details={
                    "body_id": name,
                    "repair_action": (
                        "generate_certified_collision_components"
                    ),
                    "repair_tool_module": (
                        "simulation_ready_cad_workflow.validation_pipeline."
                        "tools.compile_design_package"
                    ),
                    "authoritative_geometry_policy": (
                        "preserve_assembly_step_bytes"
                    ),
                },
            )
        _collision_progress("component_mesh_load_begin", **progress)
        mesh = chrono.ChTriangleMeshConnected()
        if not mesh.LoadWavefrontMesh(str(local_mesh), True, False):
            raise RuntimeError(f"Chrono failed to load collider {local_mesh}")
        _collision_progress("component_mesh_load_complete", **progress)
        _collision_progress("component_attach_triangle_mesh_begin", **progress)
        body.AddCollisionShape(
            chrono.ChCollisionShapeTriangleMesh(
                material,
                mesh,
                fixed,
                False if fixed else convex,
                collision_radius_m,
            )
        )
        _collision_progress("component_attach_complete", representation="triangle_mesh", **progress)
    _collision_progress("body_enable_collision_begin", body_id=name)
    body.EnableCollision(True)
    _collision_progress("body_build_complete", body_id=name)
    return body

def _make_rolling_element_body(
    chrono: Any,
    *,
    asset: dict[str, Any],
    rolling: RollingElementSetSpec,
    material: Any,
) -> Any:
    radius_m = float(asset["radius_m"])
    mass_kg = rolling.density_kg_m3 * 4.0 * math.pi * radius_m**3 / 3.0
    inertia = 0.4 * mass_kg * radius_m**2
    body = chrono.ChBodyAuxRef()
    body.SetName(str(asset["element_id"]))
    body.SetFixed(False)
    body.SetPos(chrono.ChVector3d(*asset["center_abs_m"]))
    body.SetRot(chrono.ChQuaterniond(1.0, 0.0, 0.0, 0.0))
    body.SetMass(mass_kg)
    body.SetInertiaXX(chrono.ChVector3d(inertia, inertia, inertia))
    body.AddCollisionShape(chrono.ChCollisionShapeSphere(material, radius_m))
    body.EnableCollision(True)
    return body


def _angular_velocity(body: Any) -> list[float]:
    for method_name in ("GetAngVelParent", "GetAngVelLocal"):
        method = getattr(body, method_name, None)
        if method is not None:
            return _tuple3(method())
    return [0.0, 0.0, 0.0]


def _reset_body_velocity(chrono: Any, body: Any) -> None:
    zero = chrono.ChVector3d(0.0, 0.0, 0.0)
    body.SetPosDt(zero)
    body.SetPosDt2(zero)
    if hasattr(body, "SetAngVelParent"):
        body.SetAngVelParent(zero)


def _contact_count(system: Any) -> int | None:
    container = system.GetContactContainer()
    if hasattr(container, "GetNumContacts"):
        return int(container.GetNumContacts())
    return None


def _contact_pair_counts(chrono: Any, system: Any) -> dict[str, int]:
    container = system.GetContactContainer()
    if container is None or int(container.GetNumContacts()) <= 0:
        return {}

    class PairReporter(chrono.ReportContactCallback):
        def __init__(self) -> None:
            super().__init__()
            self.counts: dict[str, int] = {}

        def OnReportContact(
            self,
            _point_a: Any,
            _point_b: Any,
            _plane: Any,
            _distance: float,
            _effective_radius: float,
            _reaction_force: Any,
            _reaction_torque: Any,
            contact_a: Any,
            contact_b: Any,
            _constraint_offset: int,
        ) -> bool:
            name_a = str(contact_a.GetPhysicsItem().GetName())
            name_b = str(contact_b.GetPhysicsItem().GetName())
            key = "|".join(sorted((name_a, name_b)))
            self.counts[key] = self.counts.get(key, 0) + 1
            return True

    reporter = PairReporter()
    container.ReportAllContacts(reporter)
    return dict(sorted(reporter.counts.items()))


def _reference_schedule(execution: ExecutionRequest) -> list[dict[str, Any]] | None:
    if execution.branch == "physical":
        return None
    if execution.reference_trace_file is None:
        raise MechanismValidationError(
            "REFERENCE_TRACE_REQUIRED", "reduced branch reference trace is missing"
        )
    reference = read_trace_json(execution.reference_trace_file)
    expected = {
        "status": "COMPLETED",
        "branch": "physical",
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
    }
    # The execution plan and certificate already bind the reference trace by
    # its full file SHA-256.  Requiring its historical validator-tree hash to
    # equal today's tree would invalidate a certified trace after unrelated
    # bookkeeping or another mechanism-family implementation changes.
    mismatches = [
        key for key, value in expected.items() if reference.get(key) != value
    ]
    if mismatches:
        raise MechanismValidationError(
            "REFERENCE_TRACE_MISMATCH",
            "physical reference trace mismatch: " + ", ".join(mismatches),
        )
    schedule = reference.get("executed_phases")
    if not isinstance(schedule, list) or len(schedule) != len(execution.load_case.phases):
        raise MechanismValidationError(
            "REFERENCE_SCHEDULE_INVALID",
            "physical reference trace has an invalid executed phase schedule",
        )
    normalized: list[dict[str, Any]] = []
    for index, (item, phase) in enumerate(zip(schedule, execution.load_case.phases)):
        if not isinstance(item, dict):
            raise MechanismValidationError(
                "REFERENCE_SCHEDULE_INVALID",
                f"executed_phases[{index}] must be an object",
            )
        duration = item.get("duration_s")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or not math.isfinite(float(duration))
            or not 0 < float(duration) <= phase.duration_s + 1e-9
        ):
            raise MechanismValidationError(
                "REFERENCE_SCHEDULE_INVALID",
                f"executed_phases[{index}] has an invalid duration",
            )
        expected_phase = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None if phase.direction_abs is None else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
        }
        if phase.end_magnitude_si is not None:
            expected_phase["end_magnitude_si"] = phase.end_magnitude_si
        for key, value in expected_phase.items():
            if item.get(key) != value:
                raise MechanismValidationError(
                    "REFERENCE_SCHEDULE_INVALID",
                    f"executed_phases[{index}].{key} differs from the load case",
                )
        normalized_phase = expected_phase | {"duration_s": float(duration)}
        if phase.stop_condition is not None:
            termination = item.get("termination")
            if not isinstance(termination, dict):
                raise MechanismValidationError(
                    "REFERENCE_SCHEDULE_INVALID",
                    f"executed_phases[{index}] lacks adaptive termination evidence",
                )
            normalized_phase["termination"] = dict(termination)
        normalized.append(normalized_phase)
    return normalized


def _rolling_element_validation(
    samples: list[dict[str, Any]],
    load_case: LoadCase,
    axis: tuple[float, float, float],
    rolling: RollingElementSetSpec,
) -> dict[str, Any]:
    required = load_case.program == "bidirectional_axis"
    if not required:
        return {
            "required": False,
            "passed": None,
            "reason": "rolling response is gated by bidirectional axis cases",
        }
    initial_states = samples[0].get("rolling_element_states")
    if not isinstance(initial_states, list) or len(initial_states) != len(
        rolling.elements
    ):
        return {
            "required": True,
            "passed": False,
            "reason": "rolling-element states are missing from the physical trace",
        }
    initial_by_id = {str(item["element_id"]): item for item in initial_states}
    force_windows: list[tuple[Any, set[str]]] = []
    for phase_index, phase in enumerate(load_case.phases):
        if phase.wrench_kind == "none":
            continue
        phase_ids = {phase.phase_id}
        for following in load_case.phases[phase_index + 1 :]:
            if following.wrench_kind != "none":
                break
            phase_ids.add(following.phase_id)
        force_windows.append((phase, phase_ids))
    minimum_center_travel_m = 10e-6
    metrics: list[dict[str, Any]] = []
    for element in rolling.elements:
        initial = initial_by_id[element.element_id]
        initial_position = tuple(float(value) for value in initial["position_abs_m"])
        max_lateral_m = 0.0
        for sample in samples:
            states = sample.get("rolling_element_states", [])
            state = next(
                (
                    item
                    for item in states
                    if item.get("element_id") == element.element_id
                ),
                None,
            )
            if state is None:
                continue
            displacement = tuple(
                float(value) - initial_position[index]
                for index, value in enumerate(state["position_abs_m"])
            )
            progress = sum(
                value * direction for value, direction in zip(displacement, axis)
            )
            lateral = math.sqrt(
                sum(
                    (displacement[index] - progress * axis[index]) ** 2
                    for index in range(3)
                )
            )
            max_lateral_m = max(max_lateral_m, lateral)
        phase_metrics: list[dict[str, Any]] = []
        for phase, phase_ids in force_windows:
            states = [
                next(
                    item
                    for item in sample.get("rolling_element_states", [])
                    if item.get("element_id") == element.element_id
                )
                for sample in samples
                if sample.get("phase_id") in phase_ids
            ]
            if len(states) < 2:
                phase_metrics.append(
                    {
                        "phase_id": phase.phase_id,
                        "passed": False,
                        "reason": "insufficient rolling-element samples",
                    }
                )
                continue
            first_position = tuple(float(value) for value in states[0]["position_abs_m"])
            last_position = tuple(float(value) for value in states[-1]["position_abs_m"])
            direction_sign = 1.0 if sum(
                value * direction
                for value, direction in zip(phase.direction_abs or axis, axis)
            ) >= 0 else -1.0
            signed_center_travel_m = direction_sign * sum(
                (last_position[index] - first_position[index]) * axis[index]
                for index in range(3)
            )
            rotation_rad = _quat_angle(
                tuple(float(value) for value in states[0]["rotation_abs_quat_wxyz"]),
                [float(value) for value in states[-1]["rotation_abs_quat_wxyz"]],
            )
            phase_metrics.append(
                {
                    "phase_id": phase.phase_id,
                    "signed_center_travel_m": signed_center_travel_m,
                    "rotation_rad": rotation_rad,
                    "passed": (
                        signed_center_travel_m >= minimum_center_travel_m
                        and rotation_rad >= rolling.minimum_phase_rotation_rad
                    ),
                }
            )
        retained = max_lateral_m <= rolling.max_lateral_shift_m
        response_passed = retained and all(
            bool(item.get("passed")) for item in phase_metrics
        )
        metrics.append(
            {
                "element_id": element.element_id,
                "max_lateral_shift_m": max_lateral_m,
                "retained": retained,
                "phase_metrics": phase_metrics,
                "response_passed": response_passed,
            }
        )
    rolling_count = sum(bool(item["response_passed"]) for item in metrics)
    retained_count = sum(bool(item["retained"]) for item in metrics)
    passed = (
        rolling_count >= rolling.minimum_rolling_count
        and retained_count == len(rolling.elements)
    )
    return {
        "required": True,
        "passed": passed,
        "rolling_response_count": rolling_count,
        "minimum_rolling_count": rolling.minimum_rolling_count,
        "retained_count": retained_count,
        "total_count": len(rolling.elements),
        "minimum_phase_rotation_rad": rolling.minimum_phase_rotation_rad,
        "minimum_center_travel_m": minimum_center_travel_m,
        "max_allowed_lateral_shift_m": rolling.max_lateral_shift_m,
        "elements": metrics,
    }


def _plain_prismatic_interface_declarations(
    *,
    collision_geometry: dict[str, Any],
    finite_support_required: bool,
    maximum_transverse_drift_m: float,
    maximum_axis_misalignment_rad: float,
    body_a_id: str = "fixed_boundary",
    body_b_id: str = "moving_group",
) -> tuple[dict[str, Any], ...]:
    """Bind a plain guide to the force path it actually uses.

    A clearance guide driven by an explicit finite support law is a compliant
    physical interface; lack of hard contact is not evidence that the guide
    vanished.  Plain prismatic mechanisms without that support contract keep
    the historical contact-bearing guide declaration.
    """

    if not finite_support_required:
        return (
            interface_declaration(
                interface_id="contact-fixed-moving-linear-guide",
                classification=CONTACT_PHYSICAL,
                body_a_id=body_a_id,
                body_b_id=body_b_id,
                model="step_derived_contact_linear_guide",
                source="trusted_plain_prismatic_runtime_v1",
                free_coordinates=("translation_along_axis",),
            ),
        )
    support_gate = bounded_support_gate(
        body_a_id=body_a_id,
        body_b_id=body_b_id,
        endpoint_a_component_refs=body_component_refs(
            collision_geometry, body_a_id
        ),
        endpoint_b_component_refs=body_component_refs(
            collision_geometry, body_b_id
        ),
        maximum_drift_m=maximum_transverse_drift_m,
        coordinate_mode=TRANSVERSE_ANCHOR_DRIFT,
        maximum_axis_misalignment_rad=maximum_axis_misalignment_rad,
        template_id="finite_prismatic_guide_retention_v1",
        proof_source="validator_plain_prismatic_support_envelope_v1",
    )
    return (
        interface_declaration(
            interface_id="support-fixed-moving-linear-guide",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id=body_a_id,
            body_b_id=body_b_id,
            model="finite_compliant_prismatic_guide",
            source="trusted_plain_prismatic_runtime_v1",
            free_coordinates=("translation_along_axis",),
            transmission_gate=support_gate,
        ),
    )


def _validate_plain_runtime_body_origins(
    execution: Any,
    geometry: Any,
) -> None:
    """Keep runtime body frames identical to materialized collision frames."""

    family_spec = getattr(execution.adapter, "family_spec", None)
    collision = (
        family_spec.get("collision_geometry")
        if isinstance(family_spec, Mapping)
        else None
    )
    bodies = collision.get("bodies") if isinstance(collision, Mapping) else None
    if not isinstance(bodies, list):
        return
    expected = {
        str(item.get("body_id") or ""): tuple(
            float(value) for value in item.get("initial_center_abs_m") or ()
        )
        for item in bodies
        if isinstance(item, Mapping)
    }
    actual = {
        "fixed_group": tuple(float(value) for value in geometry.fixed_origin_abs_m),
        "moving_group": tuple(float(value) for value in geometry.moving_origin_abs_m),
    }
    mismatches = {
        body_id: {"expected": list(expected.get(body_id, ())), "actual": list(origin)}
        for body_id, origin in actual.items()
        if len(expected.get(body_id, ())) != 3
        or any(
            abs(left - right) > 1.0e-9
            for left, right in zip(expected.get(body_id, ()), origin)
        )
    }
    if mismatches:
        raise MechanismValidationError(
            "MECHANISM_RUNTIME_BODY_ORIGIN_MISMATCH",
            "runtime body origins differ from the materialized collision manifest",
            details={"body_origin_mismatches": mismatches},
        )


def _support_response_is_active(
    support_spec: Any,
    *,
    physical_support_specs: Sequence[Any],
    guide_gate_active: bool,
) -> bool:
    """Gate only the guide represented by the conditional interface.

    Constitutive elements such as springs remain attached when a finite guide
    loses capture.  Turning both off converts one guide failure into a false
    loss of the mechanism's independent restoring path.
    """

    is_physical_guide = any(
        support_spec is candidate for candidate in physical_support_specs
    )
    return guide_gate_active if is_physical_guide else True


def simulate_plain_prismatic(
    execution: ExecutionRequest, workspace: Path
) -> dict[str, Any]:
    if execution.mechanism_plugin.worker_backend != "plain_prismatic_v1":
        raise MechanismValidationError(
            "MECHANISM_WORKER_BACKEND_UNAVAILABLE",
            f"worker backend {execution.mechanism_plugin.worker_backend!r} is not implemented",
        )
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    reference_schedule = _reference_schedule(execution)
    geometry = execution.adapter.geometry
    if geometry is None:
        raise RuntimeError("adapter geometry disappeared after request validation")
    _validate_plain_runtime_body_origins(execution, geometry)
    rolling = geometry.rolling_elements
    rolling_assets = (
        _rolling_element_assets(rolling, workspace=workspace)
        if rolling is not None
        else []
    )
    derived_dir = execution.output_file.parent / "derived"

    def prepare_collision_meshes(
        role: str,
        combined_asset: MeshAssetSpec,
        component_assets: tuple[MeshAssetSpec, ...],
        body_origin_abs_m: tuple[float, float, float],
    ) -> tuple[dict[str, Any], list[tuple[Path, bool]], list[dict[str, Any]]]:
        combined_local = derived_dir / f"{execution.branch}.{role}.local.obj"
        combined_metadata = _write_body_local_mesh(
            combined_asset,
            workspace=workspace,
            destination=combined_local,
            body_origin_abs_m=body_origin_abs_m,
        )

        def add_representation_metadata(
            metadata: dict[str, Any], local_path: Path, convex: bool
        ) -> None:
            box_verified = _oriented_box_spec(local_path) is not None
            metadata["convex"] = convex
            metadata["exact_primitive_equivalence_verified"] = box_verified
            if box_verified:
                metadata["collision_representation"] = "ChCollisionShapeBox"
            elif role == "fixed":
                metadata["collision_representation"] = (
                    "ChCollisionShapeTriangleMesh(static)"
                )
            elif execution.branch == "physical":
                metadata["collision_representation"] = (
                    "UNSUPPORTED_DYNAMIC_TRIANGLE_MESH"
                )
            else:
                metadata["collision_representation"] = (
                    "ChCollisionShapeTriangleMesh(disabled_reduced_branch)"
                )

        if not component_assets:
            metadata = dict(combined_metadata)
            add_representation_metadata(
                metadata, combined_local, combined_asset.convex
            )
            return (
                combined_metadata,
                [(combined_local, combined_asset.convex)],
                [metadata],
            )

        collision_inputs: list[tuple[Path, bool]] = []
        component_metadata: list[dict[str, Any]] = []
        for index, asset in enumerate(component_assets, start=1):
            local_path = (
                derived_dir
                / f"{execution.branch}.{role}.component-{index:03d}.local.obj"
            )
            metadata = _write_body_local_mesh(
                asset,
                workspace=workspace,
                destination=local_path,
                body_origin_abs_m=body_origin_abs_m,
            )
            add_representation_metadata(metadata, local_path, asset.convex)
            collision_inputs.append((local_path, asset.convex))
            component_metadata.append(metadata)
        return combined_metadata, collision_inputs, component_metadata

    fixed_asset, fixed_collision_meshes, fixed_components = (
        prepare_collision_meshes(
            "fixed",
            geometry.fixed_mesh,
            geometry.fixed_collision_meshes,
            geometry.fixed_origin_abs_m,
        )
    )
    moving_asset, moving_collision_meshes, moving_components = (
        prepare_collision_meshes(
            "moving",
            geometry.moving_mesh,
            geometry.moving_collision_meshes,
            geometry.moving_origin_abs_m,
        )
    )
    system = _make_nsc_system(
        chrono,
        collision_envelope_m=geometry.collision_envelope_m,
        collision_margin_m=geometry.collision_margin_m,
    )
    gravity = execution.plan.gravity_abs_m_s2
    system.SetGravitationalAcceleration(chrono.ChVector3d(*gravity))
    material = _make_material(chrono, geometry.friction)
    fixed_inertia = _inertia_box(
        geometry.fixed_mass_kg,
        (
            execution.plan.characteristic_length_m,
            max(execution.plan.moving_extents_m[1], 0.01),
            max(execution.plan.moving_extents_m[2], 0.01),
        ),
    )
    moving_inertia = _inertia_box(
        execution.load_case.moving_mass_kg,
        execution.plan.moving_extents_m,
    )
    fixed_body = _make_body(
        chrono,
        name="fixed_boundary",
        local_meshes=fixed_collision_meshes,
        origin_abs_m=geometry.fixed_origin_abs_m,
        fixed=True,
        mass_kg=geometry.fixed_mass_kg,
        inertia_xx=fixed_inertia,
        material=material,
        collision_radius_m=geometry.collision_radius_m,
        physical_contact_required=execution.branch == "physical",
    )
    moving_body = _make_body(
        chrono,
        name="moving_group",
        local_meshes=moving_collision_meshes,
        origin_abs_m=geometry.moving_origin_abs_m,
        fixed=False,
        mass_kg=execution.load_case.moving_mass_kg,
        inertia_xx=moving_inertia,
        material=material,
        collision_radius_m=geometry.collision_radius_m,
        physical_contact_required=execution.branch == "physical",
    )
    system.Add(fixed_body)
    system.Add(moving_body)

    rolling_bodies: list[Any] = []
    pre_settle_elapsed_s = 0.0
    if execution.branch == "physical" and rolling is not None:
        for asset in rolling_assets:
            body = _make_rolling_element_body(
                chrono,
                asset=asset,
                rolling=rolling,
                material=material,
            )
            system.Add(body)
            rolling_bodies.append(body)
        moving_body.SetFixed(True)
        while pre_settle_elapsed_s < rolling.pre_settle_s - 1e-12:
            step = min(
                execution.load_case.step_s,
                rolling.pre_settle_s - pre_settle_elapsed_s,
            )
            system.DoStepDynamics(step)
            pre_settle_elapsed_s += step
        moving_body.SetFixed(False)
        for body in [moving_body, *rolling_bodies]:
            _reset_body_velocity(chrono, body)

    ideal_links: list[str] = []
    ideal_travel_limit: dict[str, Any] | None = None
    replaced_contacts_disabled = False
    if execution.branch == "reduced":
        fixed_body.EnableCollision(False)
        moving_body.EnableCollision(False)
        prismatic = chrono.ChLinkLockPrismatic()
        prismatic.Initialize(
            moving_body,
            fixed_body,
            chrono.ChFramed(
                chrono.ChVector3d(*execution.plan.joint_origin_abs_m),
                _quat_from_z_to_axis(chrono, execution.plan.axis_abs),
            ),
        )
        ideal_travel_limit = _configure_prismatic_travel_limit(
            prismatic, geometry.certifiable_progress_limit_m
        )
        system.AddLink(prismatic)
        ideal_links.append("ChLinkLockPrismatic")
        replaced_contacts_disabled = True

    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if execution.branch == "physical" and link_count not in {0, None}:
        raise RuntimeError("physical branch unexpectedly contains an ideal link")
    if execution.branch == "reduced" and link_count != 1:
        raise RuntimeError("reduced branch did not create exactly one claimed link")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    physical_support_specs = (
        _mechanism_tuple_callback(
            execution,
            "physical_supports",
            execution.plan,
            execution.adapter,
        )
        if execution.branch == "physical"
        else ()
    )
    constitutive_support_specs = _mechanism_tuple_callback(
        execution,
        "constitutive_supports",
        execution.plan,
        execution.adapter,
    )
    support_contract = (
        execution.adapter.family_spec
        if isinstance(execution.adapter.family_spec, dict)
        else {}
    )
    missing_required_supports = []
    if (
        execution.branch == "physical"
        and support_contract.get("physical_supports_required") is True
        and not physical_support_specs
    ):
        missing_required_supports.append("physical_supports")
    if (
        support_contract.get("constitutive_supports_required") is True
        and not constitutive_support_specs
    ):
        missing_required_supports.append("constitutive_supports")
    if missing_required_supports:
        registration = plugin_registration_for(
            execution.mechanism_plugin.plugin_id
        )
        message = (
            "hash-bound family spec requires non-empty Reviewer support "
            "callbacks: " + ", ".join(missing_required_supports)
        )
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=execution.mechanism_plugin.plugin_id,
            plugin_kind="mechanism",
            callback=missing_required_supports[0],
            source_component="mechanism.worker.prismatic_support_contract",
            code="MECHANISM_ADAPTER_REQUIRED_SUPPORTS_MISSING",
            message=message,
            details={
                "capsule_id": execution.plan.capsule_id,
                "missing_callbacks": missing_required_supports,
            },
            contract_violation=True,
        )
    support_specs = (*physical_support_specs, *constitutive_support_specs)
    compliant_support_documents = [
        item.to_dict() for item in physical_support_specs
    ]
    constitutive_support_documents = [
        item.to_dict() for item in constitutive_support_specs
    ]

    def support_components(
        items: Sequence[dict[str, Any]], role: str
    ) -> list[dict[str, Any]]:
        retained: list[dict[str, Any]] = []
        for index, raw in enumerate(items, start=1):
            item = dict(raw)
            source = str(item.get("source") or item.get("derived") or "")
            item["solid_id"] = Path(source).stem or f"{role}_{index}"
            retained.append(item)
        return retained

    support_collision_geometry = {
        "schema": "plain_two_body_local_v1",
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "assembly_sha256": None,
        "bodies": [
            {
                "body_id": "fixed_boundary",
                "fixed": True,
                "components": support_components(fixed_components, "fixed"),
            },
            {
                "body_id": "moving_group",
                "fixed": False,
                "components": support_components(moving_components, "moving"),
            },
        ],
    }
    from .comparison import KinematicThresholds

    prismatic_thresholds = KinematicThresholds.for_plan(execution.plan)
    finite_guide_support = bool(
        support_contract.get("physical_supports_required") is True
    )
    captive_guide_support = bool(
        support_contract.get("physical_support_gate_mode") == "captive"
    )
    physical_interfaces = _plain_prismatic_interface_declarations(
        collision_geometry=support_collision_geometry,
        finite_support_required=finite_guide_support,
        maximum_transverse_drift_m=(
            max(prismatic_thresholds.max_transverse_motion_m, 0.05)
            if captive_guide_support
            else prismatic_thresholds.max_transverse_motion_m
        ),
        maximum_axis_misalignment_rad=(
            max(prismatic_thresholds.max_rotation_rad, 1.0)
            if captive_guide_support
            else prismatic_thresholds.max_rotation_rad
        ),
    )
    support_gate = physical_interfaces[0].get("transmission_gate")
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            support_collision_geometry,
        )
        if support_gate is not None
        else {
            "status": "NOT_REQUIRED",
            "passed": True,
            "conditional_interface_count": 0,
        }
    )
    runtime_supports: list[tuple[Any, Any, Any]] = []
    for support_spec in support_specs:
        if support_spec.chrono_type != "ExplicitChLoadBodyForceTorque":
            raise RuntimeError(
                "plain prismatic plugin requested an unsupported physical support"
            )
        initial_support_point = (
            moving_body.GetPos()
            if support_spec.force_application_point == "body_center"
            else chrono.ChVector3d(*execution.plan.joint_origin_abs_m)
        )
        support_profile = (
            "constitutive"
            if support_spec in constitutive_support_specs
            else "support"
        )
        support_force = create_body_force(
            chrono,
            moving_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            initial_support_point,
            False,
            declaration=standard_wrench_declaration(
                plugin_id=execution.mechanism_plugin.plugin_id,
                profile=support_profile,
                wrench_kind="force",
                load_id=f"{support_spec.support_id}-force",
                owner=execution.mechanism_plugin.plugin_id,
                target_body="moving_group",
                source_role="finite_joint_support",
                target_role="declared_input",
                geometry_gate=support_spec.support_id,
                active_interval="simulation_when_support_response_is_nonzero",
            ),
        )
        support_torque = create_body_torque(
            chrono,
            moving_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=execution.mechanism_plugin.plugin_id,
                profile=support_profile,
                wrench_kind="torque",
                load_id=f"{support_spec.support_id}-torque",
                owner=execution.mechanism_plugin.plugin_id,
                target_body="moving_group",
                source_role="finite_joint_support",
                target_role="declared_input",
                geometry_gate=support_spec.support_id,
                active_interval="simulation_when_support_response_is_nonzero",
            ),
        )
        add_ledgered_load(load_container, support_force)
        add_ledgered_load(load_container, support_torque)
        runtime_supports.append((support_spec, support_force, support_torque))
    input_start = chrono.ChVector3d(*execution.plan.input_point_abs_m)
    input_local = moving_body.TransformPointParentToLocal(input_start)
    plugin_id = execution.mechanism_plugin.plugin_id
    force_load = create_body_force(
        chrono,
        moving_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        input_start,
        False,
        declaration=standard_wrench_declaration(
            plugin_id=plugin_id,
            profile="task_input",
            wrench_kind="force",
            load_id="declared-input-force",
            owner=plugin_id,
            target_body="moving_group",
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, force_load)
    observable_start = chrono.ChVector3d(*execution.plan.observable_point_abs_m)
    observable_local = moving_body.TransformPointParentToLocal(observable_start)
    initial_pose = _pose(moving_body)
    initial_quaternion = tuple(initial_pose["rotation_abs_quat_wxyz"])
    axis = execution.plan.axis_abs
    joint_point_start = chrono.ChVector3d(*execution.plan.joint_origin_abs_m)
    joint_point_local = moving_body.TransformPointParentToLocal(joint_point_start)
    fixed_joint_point_local = fixed_body.TransformPointParentToLocal(
        joint_point_start
    )
    support_gate_runtime = (
        BoundedSupportRuntime(
            interface_id="support-fixed-moving-linear-guide",
            gate=support_gate,
        )
        if execution.branch == "physical" and support_gate is not None
        else None
    )
    fixed_axis_local = fixed_body.TransformDirectionParentToLocal(
        chrono.ChVector3d(*axis)
    )
    moving_axis_local = moving_body.TransformDirectionParentToLocal(
        chrono.ChVector3d(*axis)
    )

    def dot(
        left: Sequence[float], right: Sequence[float]
    ) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(
        left: Sequence[float], right: Sequence[float]
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    reference = (
        (1.0, 0.0, 0.0)
        if abs(axis[0]) < 0.8
        else (0.0, 1.0, 0.0)
    )
    frame_x_raw = cross(reference, axis)
    frame_x_norm = math.sqrt(dot(frame_x_raw, frame_x_raw))
    frame_x = tuple(value / frame_x_norm for value in frame_x_raw)
    frame_y = cross(axis, frame_x)
    joint_frame = (frame_x, frame_y, axis)

    def to_local(vector_abs: Sequence[float]) -> tuple[float, float, float]:
        return tuple(dot(vector_abs, basis) for basis in joint_frame)

    def to_abs(vector_local: Sequence[float]) -> tuple[float, float, float]:
        return tuple(
            sum(
                vector_local[index] * joint_frame[index][component]
                for index in range(3)
            )
            for component in range(3)
        )

    def relative_rotation_vector_abs(
        current: Sequence[float],
    ) -> tuple[float, float, float]:
        cw, cx, cy, cz = current
        iw, ix, iy, iz = initial_quaternion
        relative = (
            cw * iw + cx * ix + cy * iy + cz * iz,
            -cw * ix + cx * iw - cy * iz + cz * iy,
            -cw * iy + cx * iz + cy * iw - cz * ix,
            -cw * iz - cx * iy + cy * ix + cz * iw,
        )
        if relative[0] < 0.0:
            relative = tuple(-value for value in relative)
        vector_norm = math.sqrt(sum(value * value for value in relative[1:]))
        if vector_norm <= 1e-15:
            return (0.0, 0.0, 0.0)
        angle = 2.0 * math.atan2(vector_norm, relative[0])
        return tuple(value * angle / vector_norm for value in relative[1:])

    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    next_sample_s = 0.0
    time_s = 0.0
    current_support_force_abs_n = [0.0, 0.0, 0.0]
    current_support_torque_abs_nm = [0.0, 0.0, 0.0]
    current_conditional_interface_state: dict[str, Any] | None = None
    dynamics_step_count = 0

    def update_compliant_supports() -> None:
        nonlocal current_conditional_interface_state
        joint_point = _tuple3(
            moving_body.TransformPointLocalToParent(joint_point_local)
        )
        target_joint_point = _tuple3(
            fixed_body.TransformPointLocalToParent(fixed_joint_point_local)
        )
        moving_center = _tuple3(moving_body.GetPos())
        moving_velocity = _tuple3(moving_body.GetPosDt())
        angular_velocity = _angular_velocity(moving_body)
        lever = tuple(
            value - center
            for value, center in zip(joint_point, moving_center)
        )
        point_velocity = tuple(
            linear + rotational
            for linear, rotational in zip(
                moving_velocity,
                cross(angular_velocity, lever),
            )
        )
        displacement_local = to_local(
            tuple(
                value - target
                for value, target in zip(joint_point, target_joint_point)
            )
        )
        velocity_local = to_local(point_velocity)
        rotation_local = to_local(
            relative_rotation_vector_abs(
                _pose(moving_body)["rotation_abs_quat_wxyz"]
            )
        )
        angular_velocity_local = to_local(angular_velocity)
        total_force = [0.0, 0.0, 0.0]
        total_torque = [0.0, 0.0, 0.0]
        support_active = True
        if support_gate_runtime is not None:
            fixed_axis_abs = _tuple3(
                fixed_body.TransformDirectionLocalToParent(fixed_axis_local)
            )
            moving_axis_abs = _tuple3(
                moving_body.TransformDirectionLocalToParent(moving_axis_local)
            )
            fixed_axis_norm = math.sqrt(dot(fixed_axis_abs, fixed_axis_abs))
            moving_axis_norm = math.sqrt(dot(moving_axis_abs, moving_axis_abs))
            axis_alignment = dot(fixed_axis_abs, moving_axis_abs) / (
                fixed_axis_norm * moving_axis_norm
            )
            axis_misalignment = math.acos(
                max(-1.0, min(1.0, abs(axis_alignment)))
            )
            current_conditional_interface_state = support_gate_runtime.evaluate(
                time_s=time_s,
                anchor_a_abs_m=target_joint_point,
                anchor_b_abs_m=joint_point,
                velocity_a_abs_m_s=(0.0, 0.0, 0.0),
                velocity_b_abs_m_s=point_velocity,
                requested_effort_si=math.sqrt(
                    sum(value * value for value in current_support_force_abs_n)
                ),
                free_axis_abs=axis,
                axis_misalignment_rad=axis_misalignment,
            )
            support_active = bool(
                current_conditional_interface_state["transmission_active"]
            )
        for support_spec, support_force, support_torque in runtime_supports:
            response_active = _support_response_is_active(
                support_spec,
                physical_support_specs=physical_support_specs,
                guide_gate_active=support_active,
            )
            force_local = (
                tuple(
                    preload - stiffness * shift - damping * speed
                    for preload, stiffness, damping, shift, speed in zip(
                        support_spec.preload_force_local_n,
                        support_spec.translational_stiffness_n_m,
                        support_spec.translational_damping_n_s_m,
                        displacement_local,
                        velocity_local,
                    )
                )
                if response_active
                else (0.0, 0.0, 0.0)
            )
            torque_local = (
                tuple(
                    -stiffness * rotation - damping * speed
                    for stiffness, damping, rotation, speed in zip(
                        support_spec.rotational_stiffness_n_m_rad,
                        support_spec.rotational_damping_n_m_s_rad,
                        rotation_local,
                        angular_velocity_local,
                    )
                )
                if response_active
                else (0.0, 0.0, 0.0)
            )
            force_abs = to_abs(force_local)
            torque_abs = to_abs(torque_local)
            support_force.SetApplicationPoint(
                (
                    moving_body.GetPos()
                    if support_spec.force_application_point == "body_center"
                    else chrono.ChVector3d(*joint_point)
                ),
                False,
            )
            support_force.SetForce(chrono.ChVector3d(*force_abs), False)
            support_torque.SetTorque(chrono.ChVector3d(*torque_abs), False)
            for index in range(3):
                total_force[index] += force_abs[index]
                total_torque[index] += torque_abs[index]
        current_support_force_abs_n[:] = total_force
        current_support_torque_abs_nm[:] = total_torque

    def current_progress_m() -> float:
        observable = moving_body.TransformPointLocalToParent(observable_local)
        displacement = tuple(
            value - initial
            for value, initial in zip(
                _tuple3(observable),
                execution.plan.observable_point_abs_m,
            )
        )
        return sum(value * direction for value, direction in zip(displacement, axis))

    def record(phase_id: str, force_abs_n: tuple[float, float, float]) -> None:
        current_pose = _pose(moving_body)
        observable = moving_body.TransformPointLocalToParent(observable_local)
        displacement = tuple(
            value - initial
            for value, initial in zip(_tuple3(observable), execution.plan.observable_point_abs_m)
        )
        progress = sum(value * direction for value, direction in zip(displacement, axis))
        lateral_vector = tuple(
            displacement[index] - progress * axis[index] for index in range(3)
        )
        velocity = moving_body.GetPosDt()
        item = {
            "time_s": time_s,
            "phase_id": phase_id,
            "applied_force_abs_n": list(force_abs_n),
            "support_force_abs_n": list(current_support_force_abs_n),
            "support_torque_abs_nm": list(current_support_torque_abs_nm),
            "force_application_point_abs_m": _tuple3(
                moving_body.TransformPointLocalToParent(input_local)
            ),
            "moving_pose": current_pose,
            "moving_linear_velocity_abs_m_s": _tuple3(velocity),
            "observable_point_abs_m": _tuple3(observable),
            "observable_displacement_abs_m": list(displacement),
            "generalized_progress_m": progress,
            "axis_speed_m_s": sum(
                value * direction for value, direction in zip(_tuple3(velocity), axis)
            ),
            "lateral_shift_m": math.sqrt(
                sum(value * value for value in lateral_vector)
            ),
            "rotation_from_start_rad": _quat_angle(
                initial_quaternion, current_pose["rotation_abs_quat_wxyz"]
            ),
            "contact_count": _contact_count(system),
        }
        if support_gate_runtime is not None:
            if current_conditional_interface_state is None:
                raise RuntimeError(
                    "plain prismatic support gate was not evaluated"
                )
            conditional_state = dict(current_conditional_interface_state)
            support_torque_norm_nm = math.sqrt(
                sum(value * value for value in current_support_torque_abs_nm)
            )
            conditional_state["requested_support_torque_nm"] = (
                support_torque_norm_nm
            )
            conditional_state["transmitted_support_torque_nm"] = (
                support_torque_norm_nm
                if conditional_state["transmission_active"]
                else 0.0
            )
            item["conditional_interface_states"] = [conditional_state]
        if rolling_bodies:
            item["rolling_element_states"] = [
                {
                    "element_id": body.GetName(),
                    **_pose(body),
                    "angular_velocity_abs_rad_s": _angular_velocity(body),
                }
                for body in rolling_bodies
            ]
        samples.append(item)
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
        )

    if support_gate_runtime is not None:
        update_compliant_supports()
    record("initial", (0.0, 0.0, 0.0))
    next_sample_s = execution.load_case.sample_interval_s
    for phase_index, phase in enumerate(execution.load_case.phases):
        transition_current_wrench_phase(phase.phase_id)
        if phase.wrench_kind == "torque":
            raise MechanismValidationError(
                "TORQUE_LOAD_UNSUPPORTED",
                "the first prismatic worker supports force loads only",
            )
        if phase.end_magnitude_si is not None:
            raise MechanismValidationError(
                "RAMPED_WRENCH_UNSUPPORTED",
                "the plain prismatic worker does not yet support ramped force phases",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        force = tuple(value * phase.magnitude_si for value in direction)
        phase_limit_s = (
            reference_schedule[phase_index]["duration_s"]
            if reference_schedule is not None
            else phase.duration_s
        )
        elapsed = 0.0
        stop_reached = False
        while elapsed < phase_limit_s - 1e-12:
            application_point = moving_body.TransformPointLocalToParent(input_local)
            force_load.SetApplicationPoint(application_point, False)
            force_load.SetForce(chrono.ChVector3d(*force), False)
            update_compliant_supports()
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            elapsed += step
            time_s += step
            if time_s + 1e-12 >= next_sample_s:
                record(phase.phase_id, force)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if execution.branch == "physical" and phase.stop_condition is not None:
                progress = current_progress_m()
                reached = registered_stop_comparator_reached(
                    phase.stop_condition,
                    target_coordinate_si=float(phase.target_progress_m),
                    actual_coordinate_si=progress,
                )
                if reached:
                    stop_reached = True
                    break
        if not samples or samples[-1]["time_s"] < time_s - 1e-12:
            record(phase.phase_id, force)
        executed = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None if phase.direction_abs is None else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
            "duration_s": (
                reference_schedule[phase_index]["duration_s"]
                if reference_schedule is not None
                else elapsed
            ),
        }
        if phase.stop_condition is not None:
            if (
                reference_schedule is not None
                and isinstance(
                    reference_schedule[phase_index].get("termination"),
                    dict,
                )
            ):
                executed["termination"] = dict(
                    reference_schedule[phase_index]["termination"]
                )
            else:
                executed["termination"] = (
                    registered_adaptive_termination_evidence(
                        phase,
                        actual_coordinate_si=current_progress_m(),
                        reached=stop_reached,
                    )
                )
        executed_phases.append(executed)
        force_load.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    force_load.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    set_current_wrench_phase(None)
    for _, support_force, support_torque in runtime_supports:
        support_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        support_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    executed_load_sha256 = canonical_sha256(executed_phases)
    conditional_interface_runtime = (
        conditional_runtime_summary(
            [support_gate_runtime],
            physics_step_count=dynamics_step_count,
        )
        if support_gate_runtime is not None
        else None
    )
    rolling_validation = (
        _rolling_element_validation(
            samples,
            execution.load_case,
            axis,
            rolling,
        )
        if execution.branch == "physical" and rolling is not None
        else None
    )
    rolling_element_mass_kg = (
        rolling.density_kg_m3
        * 4.0
        * math.pi
        * rolling.radius_m**3
        / 3.0
        if rolling is not None
        else 0.0
    )

    chrono_version = None
    if hasattr(chrono, "GetChronoVersion"):
        chrono_version = str(chrono.GetChronoVersion())
    collision_representations = sorted(
        {
            item["collision_representation"]
            for item in [*fixed_components, *moving_components]
        }
    )
    if rolling is not None:
        collision_mode = (
            "static exact meshes, verified moving primitives, and independent "
            "STEP-derived rolling elements"
        )
    elif collision_representations == ["ChCollisionShapeBox"]:
        collision_mode = (
            "verified exact box primitives derived from per-solid STEP meshes"
        )
    else:
        collision_mode = (
            "static exact meshes plus verified force-bearing moving representations"
        )
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": execution.branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "package_id": execution.plan.package_id,
        "design_version": execution.plan.design_version,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
        "engine": {
            "name": "PyChrono",
            "version": chrono_version,
            "system": "ChSystemNSC",
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "collision_backend": "BULLET",
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(gravity),
            "chrono_link_count": link_count,
            "ideal_links": ideal_links,
            "ideal_travel_limit": ideal_travel_limit,
            "internal_ideal_joint_count": len(ideal_links),
            "collision_shape_counts": {
                "fixed": int(fixed_body.GetCollisionModel().GetNumShapes()),
            "contact_solver_max_iterations": _NSC_SOLVER_MAX_ITERATIONS,
            "contact_solver_warm_start": _NSC_SOLVER_WARM_START,
                "moving": int(moving_body.GetCollisionModel().GetNumShapes()),
            },
        },
        "branch_semantics": {
            "same_mass_kg": execution.load_case.moving_mass_kg,
            "same_inertia_xx_kg_m2": list(moving_inertia),
            "same_initial_pose": initial_pose,
            "same_external_load_sha256": executed_load_sha256,
            "replaced_contacts_disabled": replaced_contacts_disabled,
            "physical_compliant_supports": compliant_support_documents,
            "constitutive_supports": constitutive_support_documents,
            "physical_internal_ideal_joints_forbidden": True,
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": conditional_interface_runtime,
            "mass_scope": "primary moving group",
            "physical_auxiliary_contact_mass_kg": (
                rolling_element_mass_kg * len(rolling_bodies)
            ),
        },
        "collision_geometry": {
            "package_binding_sha256": execution.plan.package_binding_sha256,
            "assembly_sha256": None,
            "fixed": fixed_asset,
            "moving": moving_asset,
            "fixed_components": fixed_components,
            "moving_components": moving_components,
            "mode": collision_mode,
            "representations": collision_representations,
            "collision_scope": list(geometry.collision_scope),
            "omitted_contact_geometry": list(geometry.omitted_contact_geometry),
            "friction_assumption": geometry.friction,
            "collision_radius_m": geometry.collision_radius_m,
            "collision_envelope_m": geometry.collision_envelope_m,
            "collision_margin_m": geometry.collision_margin_m,
            "certifiable_progress_limit_m": geometry.certifiable_progress_limit_m,
            "rolling_elements": (
                {
                    "runtime_enabled": execution.branch == "physical",
                    "count": len(rolling_assets),
                    "assets": rolling_assets,
                    "radius_m": rolling.radius_m,
                    "density_kg_m3": rolling.density_kg_m3,
                    "per_element_mass_kg": rolling_element_mass_kg,
                    "pre_settle_s": pre_settle_elapsed_s,
                    "return_path_resolved": rolling.return_path_resolved,
                    "analytic_shape_is_step_derived": True,
                }
                if rolling is not None
                else None
            ),
        },
        "axis_abs": list(axis),
        "joint_origin_abs_m": list(execution.plan.joint_origin_abs_m),
        "input_point_abs_m": list(execution.plan.input_point_abs_m),
        "observable_point_initial_abs_m": list(execution.plan.observable_point_abs_m),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "executed_phases": executed_phases,
        "executed_load_sha256": executed_load_sha256,
        "sample_count": len(samples),
        "samples": samples,
        "rolling_element_validation": rolling_validation,
        "strict_non_claims": {
            "manufacturing_certification": False,
            "strength_or_fatigue": False,
            "measured_real_friction": False,
            "dynamic_response_certified": False,
            "independent_rolling_element_dynamics_resolved": bool(
                rolling_bodies
            ),
            "return_channel_circulation_resolved": (
                rolling.return_path_resolved if rolling is not None else None
            ),
        },
    }


def simulate_plain_revolute(
    execution: ExecutionRequest,
    workspace: Path,
) -> dict[str, Any]:
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(os.path.join(conda_prefix, "Library", "bin"))
    import pychrono as chrono

    reference_schedule = _reference_schedule(execution)
    geometry = execution.adapter.geometry
    if geometry is None:
        raise RuntimeError("adapter geometry disappeared after request validation")
    derived_dir = execution.output_file.parent / "derived"

    def prepare_collision_meshes(
        role: str,
        combined_asset: MeshAssetSpec,
        component_assets: tuple[MeshAssetSpec, ...],
        body_origin_abs_m: tuple[float, float, float],
    ) -> tuple[dict[str, Any], list[tuple[Path, bool]], list[dict[str, Any]]]:
        combined_local = derived_dir / f"{execution.branch}.{role}.local.obj"
        combined_metadata = _write_body_local_mesh(
            combined_asset,
            workspace=workspace,
            destination=combined_local,
            body_origin_abs_m=body_origin_abs_m,
        )
        assets = component_assets or (combined_asset,)
        collision_inputs: list[tuple[Path, bool]] = []
        component_metadata: list[dict[str, Any]] = []
        for index, asset in enumerate(assets, start=1):
            local_path = (
                derived_dir
                / f"{execution.branch}.{role}.component-{index:03d}.local.obj"
            )
            metadata = _write_body_local_mesh(
                asset,
                workspace=workspace,
                destination=local_path,
                body_origin_abs_m=body_origin_abs_m,
            )
            sphere_verified = _sphere_spec(local_path) is not None
            cylinder_verified = _axis_aligned_y_cylinder_spec(local_path) is not None
            box_verified = _oriented_box_spec(local_path) is not None
            metadata["convex"] = asset.convex
            metadata["exact_primitive_equivalence_verified"] = (
                box_verified or cylinder_verified or sphere_verified
            )
            if sphere_verified:
                representation = "ChCollisionShapeSphere"
            elif cylinder_verified:
                representation = "ChCollisionShapeCylinder"
            elif box_verified:
                representation = "ChCollisionShapeBox"
            elif role == "fixed":
                representation = "ChCollisionShapeTriangleMesh(static)"
            elif asset.convex:
                representation = "ChCollisionShapeTriangleMesh(dynamic_convex)"
            elif execution.branch == "physical":
                representation = "UNSUPPORTED_DYNAMIC_TRIANGLE_MESH"
            else:
                representation = "ChCollisionShapeTriangleMesh(disabled_reduced_branch)"
            metadata["collision_representation"] = representation
            collision_inputs.append((local_path, asset.convex))
            component_metadata.append(metadata)
        return combined_metadata, collision_inputs, component_metadata

    fixed_asset, fixed_collision_meshes, fixed_components = prepare_collision_meshes(
        "fixed",
        geometry.fixed_mesh,
        geometry.fixed_collision_meshes,
        geometry.fixed_origin_abs_m,
    )
    moving_asset, moving_collision_meshes, moving_components = prepare_collision_meshes(
        "moving",
        geometry.moving_mesh,
        geometry.moving_collision_meshes,
        geometry.moving_origin_abs_m,
    )
    family_spec = execution.adapter.family_spec or {}
    axis_spring_spec = None
    contact_fixture_spec = None
    if family_spec.get("mechanism_kind") == "axis_spring":
        from .axis_spring_spec import axis_spring_from_dict

        axis_spring_spec = axis_spring_from_dict(
            family_spec,
            expected_family="revolute",
        )
        contact_fixture_spec = axis_spring_spec.contact_fixture
    tabletop_programs = {
        "tong_tabletop_pickup",
        "tong_tabletop_hold",
        "tong_tabletop_release",
    }
    tabletop_task = (
        contact_fixture_spec is not None
        and contact_fixture_spec.shape == "tabletop_pickup_box"
        and execution.load_case.program in tabletop_programs
    )
    system = _make_nsc_system(
        chrono,
        collision_envelope_m=geometry.collision_envelope_m,
        collision_margin_m=geometry.collision_margin_m,
    )
    gravity = execution.plan.gravity_abs_m_s2
    system.SetGravitationalAcceleration(chrono.ChVector3d(*gravity))
    runtime_product_friction = (
        contact_fixture_spec.friction
        if tabletop_task and contact_fixture_spec is not None
        else geometry.friction
    )
    material = _make_material(chrono, runtime_product_friction)
    fixed_inertia = _inertia_box(
        geometry.fixed_mass_kg,
        (
            max(execution.plan.moving_extents_m[0], 0.01),
            max(execution.plan.moving_extents_m[1], 0.01),
            max(execution.plan.moving_extents_m[2], 0.01),
        ),
    )
    moving_inertia = _inertia_box(
        execution.load_case.moving_mass_kg,
        execution.plan.moving_extents_m,
    )
    fixed_body = _make_body(
        chrono,
        name="fixed_boundary",
        local_meshes=fixed_collision_meshes,
        origin_abs_m=geometry.fixed_origin_abs_m,
        fixed=True,
        mass_kg=geometry.fixed_mass_kg,
        inertia_xx=fixed_inertia,
        material=material,
        collision_radius_m=geometry.collision_radius_m,
        physical_contact_required=execution.branch == "physical",
    )
    moving_body = _make_body(
        chrono,
        name="moving_group",
        local_meshes=moving_collision_meshes,
        origin_abs_m=geometry.moving_origin_abs_m,
        fixed=False,
        mass_kg=execution.load_case.moving_mass_kg,
        inertia_xx=moving_inertia,
        material=material,
        collision_radius_m=geometry.collision_radius_m,
        physical_contact_required=execution.branch == "physical",
    )
    system.Add(fixed_body)
    system.Add(moving_body)

    contact_fixture_body = None
    loading_support_body = None
    loading_support_active = False
    pickup_table_body = None
    fixture_runtime_enabled = (
        execution.branch == "physical"
        and contact_fixture_spec is not None
        and (
            contact_fixture_spec.shape != "tabletop_pickup_box"
            or tabletop_task
        )
    )
    if fixture_runtime_enabled and contact_fixture_spec is not None:
        fixture_material = _make_material(chrono, contact_fixture_spec.friction)
        dynamic_grip_box = contact_fixture_spec.shape in {
            "dynamic_grip_box",
            "tabletop_pickup_box",
        }
        density_kg_m3 = 1000.0
        if dynamic_grip_box:
            assert contact_fixture_spec.mass_kg is not None
            volume_m3 = math.prod(contact_fixture_spec.size_m)
            density_kg_m3 = contact_fixture_spec.mass_kg / volume_m3
        contact_fixture_body = chrono.ChBodyEasyBox(
            *contact_fixture_spec.size_m,
            density_kg_m3,
            False,
            True,
            fixture_material,
        )
        contact_fixture_body.SetName(
            "spring_grip_test_object"
            if dynamic_grip_box
            else "spring_contact_fixture"
        )
        contact_fixture_body.SetFixed(not dynamic_grip_box)
        contact_fixture_body.SetPos(
            chrono.ChVector3d(*contact_fixture_spec.center_abs_m)
        )
        system.Add(contact_fixture_body)
        if contact_fixture_spec.shape == "dynamic_grip_box":
            assert contact_fixture_spec.loading_support_center_abs_m is not None
            assert contact_fixture_spec.loading_support_size_m is not None
            loading_support_body = chrono.ChBodyEasyBox(
                *contact_fixture_spec.loading_support_size_m,
                1000.0,
                False,
                True,
                fixture_material,
            )
            loading_support_body.SetName("spring_grip_loading_support")
            loading_support_body.SetFixed(True)
            loading_support_body.SetPos(
                chrono.ChVector3d(
                    *contact_fixture_spec.loading_support_center_abs_m
                )
            )
            system.Add(loading_support_body)
            loading_support_active = True
        elif contact_fixture_spec.shape == "tabletop_pickup_box":
            assert contact_fixture_spec.table_center_abs_m is not None
            assert contact_fixture_spec.table_size_m is not None
            pickup_table_body = chrono.ChBodyEasyBox(
                *contact_fixture_spec.table_size_m,
                1000.0,
                False,
                True,
                fixture_material,
            )
            pickup_table_body.SetName("spring_pickup_table")
            pickup_table_body.SetFixed(True)
            pickup_table_body.SetPos(
                chrono.ChVector3d(*contact_fixture_spec.table_center_abs_m)
            )
            system.Add(pickup_table_body)

    carrier_offset_abs_m = [0.0, 0.0, 0.0]
    if tabletop_task:
        assert contact_fixture_spec is not None
        assert contact_fixture_spec.approach_offset_abs_m is not None
        carrier_offset_abs_m[:] = contact_fixture_spec.approach_offset_abs_m
        fixed_body.SetPos(
            chrono.ChVector3d(
                *(
                    origin + offset
                    for origin, offset in zip(
                        geometry.fixed_origin_abs_m,
                        carrier_offset_abs_m,
                    )
                )
            )
        )
        moving_body.SetPos(
            chrono.ChVector3d(
                *(
                    origin + offset
                    for origin, offset in zip(
                        geometry.moving_origin_abs_m,
                        carrier_offset_abs_m,
                    )
                )
            )
        )
        _reset_body_velocity(chrono, fixed_body)
        _reset_body_velocity(chrono, moving_body)

    def support_components(
        items: Sequence[dict[str, Any]],
        role: str,
    ) -> list[dict[str, Any]]:
        retained: list[dict[str, Any]] = []
        for index, raw in enumerate(items, start=1):
            item = dict(raw)
            source = str(item.get("source") or item.get("derived") or "")
            item["solid_id"] = Path(source).stem or f"{role}_{index}"
            retained.append(item)
        return retained

    support_collision_geometry = {
        "schema": "plain_two_body_local_v1",
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "assembly_sha256": None,
        "bodies": [
            {
                "body_id": "fixed_boundary",
                "fixed": True,
                "components": support_components(
                    fixed_components,
                    "fixed",
                ),
            },
            {
                "body_id": "moving_group",
                "fixed": False,
                "components": support_components(
                    moving_components,
                    "moving",
                ),
            },
        ],
    }
    from .comparison import RevoluteKinematicThresholds

    revolute_thresholds = RevoluteKinematicThresholds()
    support_gate = bounded_support_gate(
        body_a_id="fixed_boundary",
        body_b_id="moving_group",
        endpoint_a_component_refs=[
            f"fixed_boundary/{item['solid_id']}"
            for item in support_collision_geometry["bodies"][0][
                "components"
            ]
        ],
        endpoint_b_component_refs=[
            f"moving_group/{item['solid_id']}"
            for item in support_collision_geometry["bodies"][1][
                "components"
            ]
        ],
        maximum_drift_m=revolute_thresholds.max_pivot_drift_m,
        coordinate_mode=RELATIVE_ANCHOR_DRIFT,
        maximum_axis_misalignment_rad=(
            revolute_thresholds.max_off_axis_rotation_rad
        ),
        template_id="finite_revolute_pin_capture_v1",
        proof_source="validator_plain_revolute_support_envelope_v1",
    )
    ideal_links: list[str] = []
    physical_interfaces = [
        interface_declaration(
            interface_id="support-fixed-moving-revolute",
            classification=COMPLIANT_INTERFACE_PHYSICAL,
            body_a_id="fixed_boundary",
            body_b_id="moving_group",
            model="finite_revolute_support",
            source="trusted_plain_revolute_runtime_v1",
            free_coordinates=("rotation_about_axis",),
            transmission_gate=support_gate,
        )
    ]
    functional_interface_geometry_binding = (
        build_functional_interface_geometry_binding(
            physical_interfaces,
            support_collision_geometry,
        )
    )

    ideal_angle_limit: dict[str, Any] | None = None
    replaced_contacts_disabled = False
    if execution.branch == "reduced":
        fixed_body.EnableCollision(False)
        moving_body.EnableCollision(False)
        revolute = chrono.ChLinkLockRevolute()
        revolute.Initialize(
            moving_body,
            fixed_body,
            chrono.ChFramed(
                chrono.ChVector3d(
                    *(
                        origin + offset
                        for origin, offset in zip(
                            execution.plan.joint_origin_abs_m,
                            carrier_offset_abs_m,
                        )
                    )
                ),
                _quat_from_z_to_axis(chrono, execution.plan.axis_abs),
            ),
        )
        ideal_angle_limit = _configure_revolute_angle_limit(
            revolute,
            geometry.certifiable_angle_limit_rad,
        )
        system.AddLink(revolute)
        ideal_links.append("ChLinkLockRevolute")
        replaced_contacts_disabled = True

    link_count = int(system.GetNumLinks()) if hasattr(system, "GetNumLinks") else None
    if execution.branch == "physical" and link_count not in {0, None}:
        raise RuntimeError("physical branch unexpectedly contains an ideal link")
    if execution.branch == "reduced" and link_count != 1:
        raise RuntimeError("reduced branch did not create exactly one claimed link")

    load_container = chrono.ChLoadContainer()
    system.Add(load_container)
    physical_support_specs = (
        _mechanism_tuple_callback(
            execution,
            "physical_supports",
            execution.plan,
            execution.adapter,
        )
        if execution.branch == "physical"
        else ()
    )
    constitutive_support_specs = _mechanism_tuple_callback(
        execution,
        "constitutive_supports",
        execution.plan,
        execution.adapter,
    )
    support_specs = (*physical_support_specs, *constitutive_support_specs)
    compliant_supports = [item.to_dict() for item in physical_support_specs]
    constitutive_supports = [
        item.to_dict() for item in constitutive_support_specs
    ]
    runtime_supports: list[tuple[Any, Any, Any]] = []
    for support_spec in support_specs:
        if support_spec.chrono_type != "ExplicitChLoadBodyForceTorque":
            raise RuntimeError(
                "plain revolute plugin requested an unsupported physical support"
            )
        support_profile = (
            "constitutive"
            if support_spec in constitutive_support_specs
            else "support"
        )
        support_force = create_body_force(
            chrono,
            moving_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            chrono.ChVector3d(*execution.plan.joint_origin_abs_m),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=execution.mechanism_plugin.plugin_id,
                profile=support_profile,
                wrench_kind="force",
                load_id=f"{support_spec.support_id}-force",
                owner=execution.mechanism_plugin.plugin_id,
                target_body="moving_group",
                source_role="finite_joint_support",
                target_role="declared_input",
                geometry_gate=support_spec.support_id,
                active_interval="simulation_when_support_response_is_nonzero",
            ),
        )
        support_torque = create_body_torque(
            chrono,
            moving_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=execution.mechanism_plugin.plugin_id,
                profile=support_profile,
                wrench_kind="torque",
                load_id=f"{support_spec.support_id}-torque",
                owner=execution.mechanism_plugin.plugin_id,
                target_body="moving_group",
                source_role="finite_joint_support",
                target_role="declared_input",
                geometry_gate=support_spec.support_id,
                active_interval="simulation_when_support_response_is_nonzero",
            ),
        )
        add_ledgered_load(load_container, support_force)
        add_ledgered_load(load_container, support_torque)
        runtime_supports.append((support_spec, support_force, support_torque))
    axis_spring_limit_torque_load = None
    if execution.branch == "physical" and axis_spring_spec is not None:
        axis_spring_limit_torque_load = create_body_torque(
            chrono,
            moving_body,
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
            declaration=standard_wrench_declaration(
                plugin_id=execution.mechanism_plugin.plugin_id,
                profile="limit",
                wrench_kind="torque",
                load_id="axis-spring-finite-limit",
                owner=execution.mechanism_plugin.plugin_id,
                target_body="moving_group",
                source_role="finite_joint_stop",
                target_role="declared_input",
                geometry_gate="axis_spring_deflection_outside_declared_range",
                active_interval="geometry_gated_simulation_interval",
                release_required=True,
            ),
        )
        add_ledgered_load(load_container, axis_spring_limit_torque_load)
    torque_load = create_body_torque(
        chrono,
        moving_body,
        chrono.ChVector3d(0.0, 0.0, 0.0),
        False,
        declaration=standard_wrench_declaration(
            plugin_id=execution.mechanism_plugin.plugin_id,
            profile="task_input",
            wrench_kind="torque",
            load_id="declared-input-torque",
            owner=execution.mechanism_plugin.plugin_id,
            target_body="moving_group",
            source_role="declared_input_port",
            target_role="declared_input",
            geometry_gate="load_case_phase_and_input_frame",
            active_interval="declared_load_case_phases",
            release_required=True,
        ),
    )
    add_ledgered_load(load_container, torque_load)
    observable_initial_abs_m = tuple(
        origin + offset
        for origin, offset in zip(
            execution.plan.observable_point_abs_m,
            carrier_offset_abs_m,
        )
    )
    joint_initial_abs_m = tuple(
        origin + offset
        for origin, offset in zip(
            execution.plan.joint_origin_abs_m,
            carrier_offset_abs_m,
        )
    )
    observable_start = chrono.ChVector3d(*observable_initial_abs_m)
    observable_local = moving_body.TransformPointParentToLocal(observable_start)
    joint_point_start = chrono.ChVector3d(*joint_initial_abs_m)
    joint_point_local = moving_body.TransformPointParentToLocal(joint_point_start)
    fixed_joint_point_local = fixed_body.TransformPointParentToLocal(
        joint_point_start
    )
    initial_pose = _pose(moving_body)
    initial_fixed_pose = _pose(fixed_body)
    initial_quaternion = tuple(initial_pose["rotation_abs_quat_wxyz"])
    axis = execution.plan.axis_abs
    support_gate_runtime = (
        BoundedSupportRuntime(
            interface_id="support-fixed-moving-revolute",
            gate=support_gate,
        )
        if execution.branch == "physical"
        else None
    )
    fixed_axis_local = fixed_body.TransformDirectionParentToLocal(
        chrono.ChVector3d(*axis)
    )
    moving_axis_local = moving_body.TransformDirectionParentToLocal(
        chrono.ChVector3d(*axis)
    )
    samples: list[dict[str, Any]] = []
    executed_phases: list[dict[str, Any]] = []
    next_sample_s = 0.0
    time_s = 0.0
    current_support_force_abs_n = [0.0, 0.0, 0.0]
    current_support_torque_abs_nm = [0.0, 0.0, 0.0]
    current_axis_spring_limit_torque_abs_nm = [0.0, 0.0, 0.0]
    current_axis_spring_limit_penetration_rad = 0.0
    current_axis_spring_limit_side: str | None = None
    current_conditional_interface_state: dict[str, Any] | None = None
    dynamics_step_count = 0

    def dot(
        left: Sequence[float], right: Sequence[float]
    ) -> float:
        return sum(a * b for a, b in zip(left, right))

    def cross(
        left: Sequence[float], right: Sequence[float]
    ) -> tuple[float, float, float]:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    reference = (1.0, 0.0, 0.0) if abs(axis[0]) < 0.8 else (0.0, 1.0, 0.0)
    frame_x_raw = cross(reference, axis)
    frame_x_norm = math.sqrt(dot(frame_x_raw, frame_x_raw))
    frame_x = tuple(value / frame_x_norm for value in frame_x_raw)
    frame_y = cross(axis, frame_x)
    joint_frame = (frame_x, frame_y, axis)

    def to_local(vector_abs: Sequence[float]) -> tuple[float, float, float]:
        return tuple(dot(vector_abs, basis) for basis in joint_frame)

    def to_abs(vector_local: Sequence[float]) -> tuple[float, float, float]:
        return tuple(
            sum(vector_local[index] * joint_frame[index][component]
                for index in range(3))
            for component in range(3)
        )

    def set_carrier_pose(
        offset_abs_m: Sequence[float],
        velocity_abs_m_s: Sequence[float],
    ) -> None:
        if not tabletop_task:
            return
        carrier_offset_abs_m[:] = [float(value) for value in offset_abs_m]
        fixed_body.SetPos(
            chrono.ChVector3d(
                *(
                    origin + offset
                    for origin, offset in zip(
                        geometry.fixed_origin_abs_m,
                        carrier_offset_abs_m,
                    )
                )
            )
        )
        fixed_body.SetPosDt(chrono.ChVector3d(*velocity_abs_m_s))

    def carrier_motion(
        phase_id: str,
        elapsed_s: float,
        duration_s: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not tabletop_task:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        assert contact_fixture_spec is not None
        assert contact_fixture_spec.approach_offset_abs_m is not None
        assert contact_fixture_spec.grip_offset_abs_m is not None
        assert contact_fixture_spec.lift_offset_abs_m is not None
        approach = contact_fixture_spec.approach_offset_abs_m
        grip = contact_fixture_spec.grip_offset_abs_m
        lift = contact_fixture_spec.lift_offset_abs_m
        if "initial_table_settle" in phase_id:
            return approach, (0.0, 0.0, 0.0)
        if phase_id.endswith("_approach"):
            start, end = approach, grip
        elif phase_id.endswith("_lift"):
            start, end = grip, lift
        elif phase_id in {
            "tong_pickup_verify",
            "tong_hold_verify",
            "tong_release_preload",
            "tong_release_gradual_open",
            "tong_release_passive_return",
            "tong_release_final_table_settle",
            "tong_release_observe",
        }:
            return lift, (0.0, 0.0, 0.0)
        else:
            return grip, (0.0, 0.0, 0.0)
        fraction = min(1.0, max(0.0, elapsed_s / max(duration_s, 1e-12)))
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        rate = (
            6.0 * fraction * (1.0 - fraction) / max(duration_s, 1e-12)
        )
        offset = tuple(
            left + smooth * (right - left)
            for left, right in zip(start, end)
        )
        velocity = tuple(
            rate * (right - left)
            for left, right in zip(start, end)
        )
        return offset, velocity

    def relative_rotation_vector_abs(
        current: Sequence[float],
    ) -> tuple[float, float, float]:
        cw, cx, cy, cz = current
        iw, ix, iy, iz = initial_quaternion
        relative = (
            cw * iw + cx * ix + cy * iy + cz * iz,
            -cw * ix + cx * iw - cy * iz + cz * iy,
            -cw * iy + cx * iz + cy * iw - cz * ix,
            -cw * iz - cx * iy + cy * ix + cz * iw,
        )
        if relative[0] < 0.0:
            relative = tuple(-value for value in relative)
        vector_norm = math.sqrt(sum(value * value for value in relative[1:]))
        if vector_norm <= 1e-15:
            return (0.0, 0.0, 0.0)
        angle = 2.0 * math.atan2(vector_norm, relative[0])
        return tuple(value * angle / vector_norm for value in relative[1:])

    def update_compliant_supports() -> None:
        nonlocal current_axis_spring_limit_penetration_rad
        nonlocal current_axis_spring_limit_side
        nonlocal current_conditional_interface_state
        total_force = [0.0, 0.0, 0.0]
        total_torque = [0.0, 0.0, 0.0]
        joint_point = _tuple3(
            moving_body.TransformPointLocalToParent(joint_point_local)
        )
        center = _tuple3(moving_body.GetPos())
        center_velocity = _tuple3(moving_body.GetPosDt())
        angular_velocity = _angular_velocity(moving_body)
        lever = tuple(value - origin for value, origin in zip(joint_point, center))
        point_velocity = tuple(
            linear + rotational
            for linear, rotational in zip(
                center_velocity, cross(angular_velocity, lever)
            )
        )
        target_joint_point = _tuple3(
            fixed_body.TransformPointLocalToParent(fixed_joint_point_local)
        )
        fixed_center = _tuple3(fixed_body.GetPos())
        fixed_center_velocity = _tuple3(fixed_body.GetPosDt())
        fixed_angular_velocity = _angular_velocity(fixed_body)
        fixed_lever = tuple(
            value - origin
            for value, origin in zip(target_joint_point, fixed_center)
        )
        target_point_velocity = tuple(
            linear + rotational
            for linear, rotational in zip(
                fixed_center_velocity,
                cross(fixed_angular_velocity, fixed_lever),
            )
        )
        displacement = tuple(
            value - target
            for value, target in zip(joint_point, target_joint_point)
        )
        displacement_local = to_local(displacement)
        velocity_local = to_local(
            tuple(
                value - target
                for value, target in zip(
                    point_velocity,
                    target_point_velocity,
                )
            )
        )
        rotation_local = to_local(
            relative_rotation_vector_abs(_pose(moving_body)["rotation_abs_quat_wxyz"])
        )
        angular_velocity_local = to_local(angular_velocity)
        support_active = True
        if support_gate_runtime is not None:
            fixed_axis_abs = _tuple3(
                fixed_body.TransformDirectionLocalToParent(
                    fixed_axis_local
                )
            )
            moving_axis_abs = _tuple3(
                moving_body.TransformDirectionLocalToParent(
                    moving_axis_local
                )
            )
            axis_misalignment = math.acos(
                max(
                    -1.0,
                    min(
                        1.0,
                        abs(dot(fixed_axis_abs, moving_axis_abs)),
                    ),
                )
            )
            current_conditional_interface_state = (
                support_gate_runtime.evaluate(
                    time_s=time_s,
                    anchor_a_abs_m=target_joint_point,
                    anchor_b_abs_m=joint_point,
                    velocity_a_abs_m_s=target_point_velocity,
                    velocity_b_abs_m_s=point_velocity,
                    requested_effort_si=math.sqrt(
                        sum(
                            value * value
                            for value in current_support_force_abs_n
                        )
                    ),
                    axis_misalignment_rad=axis_misalignment,
                )
            )
            support_active = bool(
                current_conditional_interface_state[
                    "transmission_active"
                ]
            )
        for support_spec, support_force, support_torque in runtime_supports:
            force_local = (
                tuple(
                    -stiffness * shift - damping * speed
                    for stiffness, damping, shift, speed in zip(
                        support_spec.translational_stiffness_n_m,
                        support_spec.translational_damping_n_s_m,
                        displacement_local,
                        velocity_local,
                    )
                )
                if support_active
                else (0.0, 0.0, 0.0)
            )
            torque_local = (
                tuple(
                    -stiffness * rotation - damping * speed
                    for stiffness, damping, rotation, speed in zip(
                        support_spec.rotational_stiffness_n_m_rad,
                        support_spec.rotational_damping_n_m_s_rad,
                        rotation_local,
                        angular_velocity_local,
                    )
                )
                if support_active
                else (0.0, 0.0, 0.0)
            )
            force_abs = to_abs(force_local)
            torque_abs = to_abs(torque_local)
            support_force.SetApplicationPoint(chrono.ChVector3d(*joint_point), False)
            support_force.SetForce(chrono.ChVector3d(*force_abs), False)
            support_torque.SetTorque(chrono.ChVector3d(*torque_abs), False)
            for index in range(3):
                total_force[index] += force_abs[index]
                total_torque[index] += torque_abs[index]
        limit_torque_local_z = 0.0
        limit_penetration_rad = 0.0
        limit_side = None
        if axis_spring_limit_torque_load is not None:
            assert axis_spring_spec is not None
            (
                limit_torque_local_z,
                limit_penetration_rad,
                limit_side,
            ) = _axis_spring_limit_response(
                rotation_local[2],
                angular_velocity_local[2],
                axis_spring_spec.maximum_deflection_si,
            )
            if not support_active:
                limit_torque_local_z = 0.0
            limit_torque_abs = to_abs((0.0, 0.0, limit_torque_local_z))
            axis_spring_limit_torque_load.SetTorque(
                chrono.ChVector3d(*limit_torque_abs),
                False,
            )
            for index in range(3):
                total_torque[index] += limit_torque_abs[index]
            current_axis_spring_limit_torque_abs_nm[:] = limit_torque_abs
        else:
            current_axis_spring_limit_torque_abs_nm[:] = [0.0, 0.0, 0.0]
        current_axis_spring_limit_penetration_rad = limit_penetration_rad
        current_axis_spring_limit_side = limit_side
        current_support_force_abs_n[:] = total_force
        current_support_torque_abs_nm[:] = total_torque

    def state() -> tuple[float, float, float, float, list[float], list[float]]:
        pose = _pose(moving_body)
        quaternion = tuple(pose["rotation_abs_quat_wxyz"])
        angle = _signed_axis_angle(quaternion, axis) - _signed_axis_angle(
            initial_quaternion, axis
        )
        angular_velocity = _angular_velocity(moving_body)
        speed = sum(
            value * direction for value, direction in zip(angular_velocity, axis)
        )
        joint_point = _tuple3(
            moving_body.TransformPointLocalToParent(joint_point_local)
        )
        target_joint_point = _tuple3(
            fixed_body.TransformPointLocalToParent(fixed_joint_point_local)
        )
        pivot_drift = math.sqrt(
            sum((value - target) ** 2 for value, target in zip(
                joint_point, target_joint_point
            ))
        )
        off_axis = _off_axis_angle(quaternion, axis)
        observable = _tuple3(moving_body.TransformPointLocalToParent(observable_local))
        displacement = [
            value - initial
            for value, initial in zip(observable, observable_initial_abs_m)
        ]
        return angle, speed, pivot_drift, off_axis, observable, displacement

    def record(phase_id: str, torque_abs_nm: tuple[float, float, float]) -> None:
        angle, speed, pivot_drift, off_axis, observable, displacement = state()
        contact_pair_counts = _contact_pair_counts(chrono, system)
        fixed_pose = _pose(fixed_body)
        moving_pose = _pose(moving_body)
        sample = {
            "time_s": time_s,
            "phase_id": phase_id,
            "applied_torque_abs_nm": list(torque_abs_nm),
            "compliant_support_force_abs_n": list(current_support_force_abs_n),
            "compliant_support_torque_abs_nm": list(current_support_torque_abs_nm),
            "axis_spring_limit_torque_abs_nm": list(
                current_axis_spring_limit_torque_abs_nm
            ),
            "axis_spring_limit_penetration_rad": (
                current_axis_spring_limit_penetration_rad
            ),
            "axis_spring_limit_side": current_axis_spring_limit_side,
            "fixed_pose": fixed_pose,
            "moving_pose": moving_pose,
            "moving_angular_velocity_abs_rad_s": _angular_velocity(moving_body),
            "observable_point_abs_m": observable,
            "observable_displacement_abs_m": displacement,
            "generalized_angle_rad": angle,
            "generalized_speed_rad_s": speed,
            "pivot_drift_m": pivot_drift,
            "off_axis_rotation_rad": off_axis,
            "rotation_from_start_rad": abs(angle),
            "contact_count": _contact_count(system),
            "carrier_offset_abs_m": list(carrier_offset_abs_m),
        }
        if execution.branch == "physical":
            if current_conditional_interface_state is None:
                raise RuntimeError(
                    "plain revolute support gate was not evaluated"
                )
            conditional_state = dict(current_conditional_interface_state)
            # ATTENTION AI EDITORS (2026-08-08): the conditional-interface
            # audit (conditional_interface.py) requires every saved
            # COMPLIANT support state to carry finite requested/transmitted
            # support torque (a separate N*m channel; the newton effort
            # channel stays effort-only).  The chain runtime emits the same
            # two fields from its bearing reaction torque.  Keep both
            # emitters and the audit aligned in the same commit. (DEFECT #3)
            support_torque_norm_nm = math.sqrt(
                sum(
                    value * value
                    for value in current_support_torque_abs_nm
                )
            )
            conditional_state["requested_support_torque_nm"] = (
                support_torque_norm_nm
            )
            conditional_state["transmitted_support_torque_nm"] = (
                support_torque_norm_nm
                if conditional_state["transmission_active"]
                else 0.0
            )
            sample["conditional_interface_states"] = [conditional_state]
        if contact_fixture_spec is not None:
            sample["contact_pair_counts"] = contact_pair_counts
            if (
                contact_fixture_spec.shape in {
                    "dynamic_grip_box",
                    "tabletop_pickup_box",
                }
                and contact_fixture_body is not None
            ):
                fixed_contacts = contact_pair_counts.get(
                    "fixed_boundary|spring_grip_test_object", 0
                )
                moving_contacts = contact_pair_counts.get(
                    "moving_group|spring_grip_test_object", 0
                )
                object_position = _tuple3(contact_fixture_body.GetPos())
                object_pose = _pose(contact_fixture_body)
                sample.update(
                    {
                        "expected_contact_count": min(
                            fixed_contacts, moving_contacts
                        ),
                        "test_object_fixed_jaw_contact_count": fixed_contacts,
                        "test_object_moving_jaw_contact_count": moving_contacts,
                        "test_object_two_sided_contact": (
                            fixed_contacts > 0 and moving_contacts > 0
                        ),
                        "test_object_position_abs_m": object_position,
                        "test_object_position_m": object_position[2],
                        "test_object_pose": object_pose,
                        "test_object_relative_to_fixed_abs_m": [
                            value - origin
                            for value, origin in zip(
                                object_position,
                                fixed_pose["position_abs_m"],
                            )
                        ],
                        "test_object_loading_support_contact_count": (
                            contact_pair_counts.get(
                                "spring_grip_loading_support|spring_grip_test_object",
                                0,
                            )
                        ),
                        "test_object_loading_support_active": (
                            loading_support_active
                        ),
                    }
                )
                if (
                    contact_fixture_spec.shape == "tabletop_pickup_box"
                    and pickup_table_body is not None
                    and contact_fixture_spec.table_center_abs_m is not None
                    and contact_fixture_spec.table_size_m is not None
                ):
                    table_contacts = contact_pair_counts.get(
                        "spring_grip_test_object|spring_pickup_table",
                        0,
                    )
                    quaternion = object_pose["rotation_abs_quat_wxyz"]
                    w, x, y, z = quaternion
                    vertical_row = (
                        2.0 * (x * z - y * w),
                        2.0 * (y * z + x * w),
                        1.0 - 2.0 * (x * x + y * y),
                    )
                    half_extents = tuple(
                        0.5 * value for value in contact_fixture_spec.size_m
                    )
                    vertical_half_extent = sum(
                        abs(component) * half
                        for component, half in zip(
                            vertical_row,
                            half_extents,
                        )
                    )
                    table_top_z = (
                        contact_fixture_spec.table_center_abs_m[2]
                        + 0.5 * contact_fixture_spec.table_size_m[2]
                    )
                    sample.update(
                        {
                            "test_object_table_contact_count": table_contacts,
                            "test_object_height_above_table_m": (
                                object_position[2]
                                - vertical_half_extent
                                - table_top_z
                            ),
                            "pickup_table_pose": _pose(pickup_table_body),
                            "direct_test_object_actuation": False,
                        }
                    )
                    sample["body_poses"] = {
                        "fixed_boundary": fixed_pose,
                        "moving_group": moving_pose,
                        "spring_grip_test_object": object_pose,
                        "spring_pickup_table": _pose(pickup_table_body),
                    }
            else:
                sample["expected_contact_count"] = contact_pair_counts.get(
                    "moving_group|spring_contact_fixture",
                    0,
                )
        samples.append(sample)
        _write_trace_checkpoint(
            execution,
            samples,
            executed_phases=executed_phases,
        )

    update_compliant_supports()
    record("initial", (0.0, 0.0, 0.0))
    next_sample_s = execution.load_case.sample_interval_s
    for phase_index, phase in enumerate(execution.load_case.phases):
        transition_current_wrench_phase(phase.phase_id)
        if (
            phase_index > 0
            and loading_support_body is not None
            and loading_support_active
        ):
            loading_support_body.EnableCollision(False)
            loading_support_active = False
        if phase.wrench_kind == "force":
            raise MechanismValidationError(
                "FORCE_LOAD_UNSUPPORTED",
                "the plain revolute worker accepts torque loads only",
            )
        direction = phase.direction_abs or (0.0, 0.0, 0.0)
        torque = tuple(value * phase.magnitude_si for value in direction)
        phase_limit_s = (
            reference_schedule[phase_index]["duration_s"]
            if reference_schedule is not None
            else phase.duration_s
        )
        elapsed = 0.0
        stop_reached = False
        while elapsed < phase_limit_s - 1e-12:
            step = min(execution.load_case.step_s, phase_limit_s - elapsed)
            magnitude = phase.magnitude_at_fraction(
                (elapsed + step) / max(phase_limit_s, 1e-12)
            )
            torque = tuple(value * magnitude for value in direction)
            carrier_offset, carrier_velocity = carrier_motion(
                phase.phase_id,
                elapsed + step,
                phase_limit_s,
            )
            set_carrier_pose(carrier_offset, carrier_velocity)
            torque_load.SetTorque(chrono.ChVector3d(*torque), False)
            update_compliant_supports()
            system.DoStepDynamics(step)
            dynamics_step_count += 1
            elapsed += step
            time_s += step
            if time_s + 1e-12 >= next_sample_s:
                record(phase.phase_id, torque)
                while next_sample_s <= time_s + 1e-12:
                    next_sample_s += execution.load_case.sample_interval_s
            if execution.branch == "physical" and phase.stop_condition is not None:
                angle = state()[0]
                target = float(phase.target_coordinate)
                reached = registered_stop_comparator_reached(
                    phase.stop_condition,
                    target_coordinate_si=target,
                    actual_coordinate_si=angle,
                )
                if reached:
                    stop_reached = True
                    break
        if not samples or samples[-1]["time_s"] < time_s - 1e-12:
            record(phase.phase_id, torque)
        executed_phase = {
                "phase_id": phase.phase_id,
                "wrench_kind": phase.wrench_kind,
                "direction_abs": (
                    None if phase.direction_abs is None else list(phase.direction_abs)
                ),
                "magnitude_si": phase.magnitude_si,
                "duration_s": (
                    reference_schedule[phase_index]["duration_s"]
                    if reference_schedule is not None
                    else elapsed
                ),
            }
        if phase.end_magnitude_si is not None:
            executed_phase["end_magnitude_si"] = phase.end_magnitude_si
        if phase.stop_condition is not None:
            if (
                reference_schedule is not None
                and isinstance(
                    reference_schedule[phase_index].get("termination"),
                    dict,
                )
            ):
                executed_phase["termination"] = dict(
                    reference_schedule[phase_index]["termination"]
                )
            else:
                executed_phase["termination"] = (
                    registered_adaptive_termination_evidence(
                        phase,
                        actual_coordinate_si=state()[0],
                        reached=stop_reached,
                    )
                )
        executed_phases.append(executed_phase)
        torque_load.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    torque_load.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    set_current_wrench_phase(None)
    for _, support_force, support_torque in runtime_supports:
        support_force.SetForce(chrono.ChVector3d(0.0, 0.0, 0.0), False)
        support_torque.SetTorque(chrono.ChVector3d(0.0, 0.0, 0.0), False)
    if axis_spring_limit_torque_load is not None:
        axis_spring_limit_torque_load.SetTorque(
            chrono.ChVector3d(0.0, 0.0, 0.0),
            False,
        )
    executed_load_sha256 = canonical_sha256(executed_phases)
    conditional_interface_runtime = (
        conditional_runtime_summary(
            [support_gate_runtime],
            physics_step_count=dynamics_step_count,
        )
        if support_gate_runtime is not None
        else None
    )
    chrono_version = (
        str(chrono.GetChronoVersion()) if hasattr(chrono, "GetChronoVersion") else None
    )
    collision_representations = sorted(
        {
            item["collision_representation"]
            for item in [*fixed_components, *moving_components]
        }
    )
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": execution.branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "package_id": execution.plan.package_id,
        "design_version": execution.plan.design_version,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
        "engine": {
            "name": "PyChrono",
            "version": chrono_version,
            "system": "ChSystemNSC",
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "worker_backend": execution.mechanism_plugin.worker_backend,
            "collision_backend": "BULLET",
            "step_s": execution.load_case.step_s,
            "gravity_abs_m_s2": list(gravity),
            "chrono_link_count": link_count,
            "ideal_links": ideal_links,
            "ideal_angle_limit": ideal_angle_limit,
            "internal_ideal_joint_count": len(ideal_links),
            "compliant_supports": compliant_supports,
            "constitutive_supports": constitutive_supports,
            "axis_spring_deflection_limit": (
                None
                if axis_spring_spec is None
                else {
                    "minimum_deflection_rad": 0.0,
                    "maximum_deflection_rad": (
                        axis_spring_spec.maximum_deflection_si
                    ),
                    "physical_model": (
                        "finite_penalty_stop"
                        if execution.branch == "physical"
                        else "ideal_revolute_limit"
                    ),
                    "physical_stiffness_n_m_rad": (
                        _AXIS_SPRING_LIMIT_STIFFNESS_N_M_RAD
                    ),
                    "physical_damping_n_m_s_rad": (
                        _AXIS_SPRING_LIMIT_DAMPING_N_M_S_RAD
                    ),
                    "ideal_constraint_in_physical_branch": False,
                }
            ),
            "collision_shape_counts": {
                "fixed": int(fixed_body.GetCollisionModel().GetNumShapes()),
                "moving": int(moving_body.GetCollisionModel().GetNumShapes()),
                "external_contact_fixture": (
                    int(contact_fixture_body.GetCollisionModel().GetNumShapes())
                    if contact_fixture_body is not None
                    else 0
                ),
                "external_loading_support": (
                    int(loading_support_body.GetCollisionModel().GetNumShapes())
                    if loading_support_body is not None
                    else 0
                ),
                "external_pickup_table": (
                    int(pickup_table_body.GetCollisionModel().GetNumShapes())
                    if pickup_table_body is not None
                    else 0
                ),
            },
            "contact_solver_max_iterations": _NSC_SOLVER_MAX_ITERATIONS,
            "contact_solver_warm_start": _NSC_SOLVER_WARM_START,
        },
        "branch_semantics": {
            "same_mass_kg": execution.load_case.moving_mass_kg,
            "same_inertia_xx_kg_m2": list(moving_inertia),
            "same_initial_pose": initial_pose,
            "same_external_load_sha256": executed_load_sha256,
            "replaced_contacts_disabled": replaced_contacts_disabled,
            "physical_internal_ideal_joints_forbidden": True,
            "physical_interface_policy": collision_policy_disclosure(
                physical_interfaces, branch=execution.branch
            ),
            "functional_interface_geometry_binding": (
                functional_interface_geometry_binding
            ),
            "conditional_interface_runtime": (
                conditional_interface_runtime
            ),
            "mass_scope": "primary moving group",
            "physical_auxiliary_contact_mass_kg": (
                float(contact_fixture_spec.mass_kg)
                if contact_fixture_body is not None
                and contact_fixture_spec is not None
                and contact_fixture_spec.shape in {
                    "dynamic_grip_box",
                    "tabletop_pickup_box",
                }
                and contact_fixture_spec.mass_kg is not None
                else 0.0
            ),
            "direct_test_object_actuation": False,
            "hand_guided_body_id": (
                "fixed_boundary" if tabletop_task else None
            ),
            "whole_tool_carrier_program": tabletop_task,
            "moving_group_follows_finite_revolute_support": tabletop_task,
            "loading_support_removed_after_setup_phase": (
                contact_fixture_spec is not None
                and contact_fixture_spec.shape == "dynamic_grip_box"
            ),
            "external_contact_fixture_declared": contact_fixture_spec is not None,
            "external_contact_fixture_runtime_enabled": (
                contact_fixture_body is not None
            ),
            "external_pickup_table_runtime_enabled": (
                pickup_table_body is not None
            ),
            "expected_contact_pair": (
                (
                    "fixed_boundary+moving_group|spring_grip_test_object"
                    if contact_fixture_spec.shape in {
                        "dynamic_grip_box",
                        "tabletop_pickup_box",
                    }
                    else "moving_group|spring_contact_fixture"
                )
                if contact_fixture_body is not None
                and contact_fixture_spec is not None
                else None
            ),
            "expected_table_contact_pair": (
                "spring_grip_test_object|spring_pickup_table"
                if pickup_table_body is not None
                and contact_fixture_spec is not None
                and contact_fixture_spec.shape == "tabletop_pickup_box"
                else None
            ),
            "physical_compliant_supports": compliant_supports,
            "constitutive_supports": constitutive_supports,
        },
        "collision_geometry": {
            "package_binding_sha256": execution.plan.package_binding_sha256,
            "assembly_sha256": None,
            "fixed": fixed_asset,
            "moving": moving_asset,
            "fixed_components": fixed_components,
            "moving_components": moving_components,
            "mode": "static exact mesh plus verified convex moving decomposition",
            "representations": collision_representations,
            "collision_scope": list(geometry.collision_scope),
            "omitted_contact_geometry": list(geometry.omitted_contact_geometry),
            "friction_assumption": geometry.friction,
            "runtime_product_friction_assumption": runtime_product_friction,
            "collision_radius_m": geometry.collision_radius_m,
            "collision_envelope_m": geometry.collision_envelope_m,
            "collision_margin_m": geometry.collision_margin_m,
            "certifiable_angle_limit_rad": geometry.certifiable_angle_limit_rad,
            "external_contact_fixture": (
                None
                if contact_fixture_spec is None
                else {
                    **contact_fixture_spec.to_dict(),
                    "runtime_enabled": contact_fixture_body is not None,
                    "source": "hash_bound_axis_spring_family_spec",
                    "product_geometry": False,
                }
            ),
            "rolling_elements": None,
        },
        "axis_abs": list(axis),
        "joint_origin_abs_m": list(execution.plan.joint_origin_abs_m),
        "input_point_abs_m": list(execution.plan.input_point_abs_m),
        "observable_point_initial_abs_m": list(execution.plan.observable_point_abs_m),
        "characteristic_length_m": execution.plan.characteristic_length_m,
        "executed_phases": executed_phases,
        "executed_load_sha256": executed_load_sha256,
        "sample_count": len(samples),
        "samples": samples,
        "rolling_element_validation": None,
        "strict_non_claims": {
            "manufacturing_certification": False,
            "strength_or_fatigue": False,
            "measured_real_friction": False,
            "bearing_stiffness_or_damping_certified": False,
            "dynamic_response_certified": False,
        },
    }


def _mechanism_tuple_callback(
    execution: ExecutionRequest,
    callback: str,
    *arguments: Any,
) -> tuple[Any, ...]:
    plugin = execution.mechanism_plugin
    result = invoke_mechanism_plugin_callback(
        plugin,
        callback,
        *arguments,
        source_component=f"mechanism.worker.{callback}",
        unexpected_code=f"MECHANISM_ADAPTER_{callback.upper()}_FAILED",
    )
    invalid = not isinstance(result, tuple) or (
        callback in {"physical_supports", "constitutive_supports"}
        and isinstance(result, tuple)
        and any(not callable(getattr(item, "to_dict", None)) for item in result)
    )
    if invalid:
        message = f"mechanism adapter {callback}() returned an invalid tuple"
        try:
            registration = plugin_registration_for(str(plugin.plugin_id))
        except MechanismValidationError:
            registration = None
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback=callback,
            source_component=f"mechanism.worker.{callback}",
            code=f"MECHANISM_ADAPTER_{callback.upper()}_INVALID",
            message=message,
            details={"capsule_id": execution.plan.capsule_id},
            contract_violation=True,
        )
    return result


def _mechanism_plugin_failure_owner(
    execution: ExecutionRequest,
) -> FailureOwner:
    plugin_id = str(execution.mechanism_plugin.plugin_id)
    try:
        registration = plugin_registration_for(plugin_id)
    except MechanismValidationError:
        return FailureOwner.UNCLASSIFIED
    return mechanism_owner_for_registration_source(
        str(registration.get("registration_source") or "")
    )


def _plugin_callback_failure(
    execution: ExecutionRequest,
    exc: BaseException,
    *,
    code: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    plugin_id = execution.mechanism_plugin.plugin_id
    try:
        registration = plugin_registration_for(str(plugin_id))
    except MechanismValidationError:
        registration = None
    boundary_error = plugin_boundary_error(
        exc,
        registration=registration,
        plugin_id=str(plugin_id),
        plugin_kind="mechanism",
        callback="simulate",
        source_component="mechanism.plugin.simulate",
        code=code,
        message=message,
        evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
    )
    return dict(boundary_error.failure_envelope or {})


def _simulate(execution: ExecutionRequest, workspace: Path) -> dict[str, Any]:
    MechanismAdapter.validate_execution_geometry_proofs(
        execution.mechanism_plugin,
        execution,
    )
    try:
        payload = execution.mechanism_plugin.simulate(execution, workspace)
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise MechanismValidationError(
            exc.code,
            exc.message,
            details=exc.details,
            failure=_plugin_callback_failure(execution, exc),
        ) from exc
    except Exception as exc:
        plugin_id = str(getattr(execution.mechanism_plugin, "plugin_id", ""))
        code = "MECHANISM_ADAPTER_SIMULATE_FAILED"
        message = f"mechanism adapter simulate() failed: {exc}"
        raise MechanismValidationError(
            code,
            message,
            details={"plugin_id": plugin_id},
            failure=_plugin_callback_failure(
                execution,
                exc,
                code=code,
                message=message,
            ),
        ) from exc
    expected_identity = {
        "status": "COMPLETED",
        "validator_source_sha256": lane_execution_source_sha256(),
        "branch": execution.artifact_branch,
        "adapter_id": execution.adapter.adapter_id,
        "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(execution.load_case.to_dict()),
    }
    mismatches = (
        ["result_not_object"]
        if not isinstance(payload, dict)
        else [
            key
            for key, expected in expected_identity.items()
            if payload.get(key) != expected
        ]
    )
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list) or not samples or any(
        not isinstance(item, dict) for item in samples
    ):
        mismatches.append("samples")
    if mismatches:
        message = (
            "mechanism adapter simulate() returned an invalid trace: "
            + ", ".join(mismatches)
        )
        plugin = execution.mechanism_plugin
        try:
            registration = plugin_registration_for(str(plugin.plugin_id))
        except MechanismValidationError:
            registration = None
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="simulate",
            source_component="mechanism.plugin.simulate",
            code="MECHANISM_ADAPTER_SIMULATE_RESULT_INVALID",
            message=message,
            details={
                "capsule_id": execution.plan.capsule_id,
                "load_case_id": execution.load_case.load_case_id,
                "mismatches": mismatches,
            },
            contract_violation=True,
            evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
        )
    return payload


def _execution_wrench_body_roles(
    execution: ExecutionRequest,
) -> tuple[frozenset[str], frozenset[str]]:
    if execution.plan.task_channel_registry is not None:
        from .task_motion_evidence import task_channel_registry_for_plan

        registry = task_channel_registry_for_plan(execution.plan)
        return (
            frozenset(registry.input_body_ids),
            frozenset(registry.output_body_ids),
        )
    family_spec = execution.adapter.family_spec or {}
    roles = _mechanism_tuple_callback(
        execution,
        "wrench_body_roles",
        family_spec,
    )
    if (
        len(roles) != 2
        or any(not isinstance(item, frozenset) for item in roles)
    ):
        message = "mechanism adapter wrench_body_roles() returned invalid roles"
        plugin = execution.mechanism_plugin
        try:
            registration = plugin_registration_for(str(plugin.plugin_id))
        except MechanismValidationError:
            registration = None
        raise plugin_boundary_error(
            TypeError(message),
            registration=registration,
            plugin_id=str(plugin.plugin_id),
            plugin_kind="mechanism",
            callback="wrench_body_roles",
            source_component="mechanism.worker.wrench_body_roles",
            code="MECHANISM_ADAPTER_WRENCH_BODY_ROLES_INVALID",
            message=message,
            details={"capsule_id": execution.plan.capsule_id},
            contract_violation=True,
        )
    return roles[0], roles[1]


def _wrench_ledger_for_execution(execution: ExecutionRequest) -> WrenchLedger:
    plugin_id = execution.mechanism_plugin.plugin_id
    input_bodies, output_bodies = _execution_wrench_body_roles(execution)
    return WrenchLedger(
        plugin_id=plugin_id,
        capability_registration=plugin_registration_for(plugin_id),
        policies=wrench_policies_for_plugin(plugin_id),
        declared_input_body_ids=input_bodies,
        declared_output_body_ids=output_bodies,
        branch=execution.artifact_branch,
        load_case_id=execution.load_case.load_case_id,
        program_phases=(
            phase.to_dict() for phase in execution.load_case.phases
        ),
    )


def run_request(
    request_file: str | Path,
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    execution = load_execution_request(request_file, workspace_root=workspace)
    ledger = _wrench_ledger_for_execution(execution)
    try:
        with activate_wrench_ledger(ledger):
            payload = _simulate(execution, workspace)
            ledger.bind_executed_schedule(
                payload.get("executed_phases"),
                executed_load_sha256=payload.get("executed_load_sha256"),
            )
            payload["wrench_audit"] = ledger.verify_for_certification()
    except Exception as exc:
        failure = failure_envelope_from_exception(
            exc,
            default_owner=FailureOwner.HARNESS_CODE,
            source_component="mechanism.worker.run_request",
            evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
        )
        measured_physical_failure = bool(
            execution.artifact_branch == "physical"
            and failure.get("owner") == FailureOwner.DESIGN_PACKAGE.value
            and failure.get("evidence_validity")
            == EvidenceValidity.VALID_MEASUREMENT.value
        )
        error = {
            "type": type(exc).__name__,
            "code": getattr(exc, "code", "SIMULATION_EXCEPTION"),
            "message": str(exc),
            "traceback": traceback.format_exc(limit=12),
        }
        checkpoint: dict[str, Any] = {}
        if execution.output_file.is_file():
            try:
                candidate = read_json(execution.output_file)
            except Exception:
                candidate = {}
            if (
                candidate.get("status") == "PHYSICS_PARTIAL"
                and candidate.get("samples")
            ):
                checkpoint = candidate
        payload = {
            **checkpoint,
            "schema_version": "0.1",
            "status": (
                "PHYSICAL_FAILURE"
                if checkpoint and measured_physical_failure
                else "PHYSICS_INCONCLUSIVE"
            ),
            "branch": execution.artifact_branch,
            "adapter_id": execution.adapter.adapter_id,
            "mechanism_plugin_id": execution.mechanism_plugin.plugin_id,
            "package_binding_sha256": execution.plan.package_binding_sha256,
            "capsule_id": execution.plan.capsule_id,
            "load_case_id": execution.load_case.load_case_id,
            "error": error,
            "failure": failure,
            "wrench_audit": ledger.audit(),
        }
        if checkpoint:
            payload["checkpoint"] = {
                **(checkpoint.get("checkpoint") or {}),
                "status": (
                    "TERMINATED_BY_CERTIFICATION_FAILURE"
                    if measured_physical_failure
                    else "TERMINATED_BY_SIMULATION_EXCEPTION"
                ),
                "final_termination_reason": error["code"],
            }
            payload["visual_replay_semantics"] = {
                **(checkpoint.get("visual_replay_semantics") or {}),
                "trace_complete": False,
                "partial_checkpoint": True,
                "certification_authority": False,
                "failure_evidence_authority": measured_physical_failure,
            }
    if execution.artifact_branch == "ideal_dynamics":
        payload["branch"] = "ideal_dynamics"
        payload["runtime_compatibility"] = {
            "internal_branch_name": "reduced",
            "meaning": "ideal_joint_dynamics_with_collision_disabled",
            "contact_evidence_authority": False,
        }
    write_trace_json_atomic(execution.output_file, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run physical or optional ideal-dynamics mechanism evidence."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _verify_validator_source_binding()
        payload = run_request(args.request, workspace_root=args.workspace_root)
    except Exception as exc:
        error = {
            "status": "INVALID_REQUEST",
            "error_code": getattr(exc, "code", type(exc).__name__),
            "message": str(exc),
        }
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "branch": payload["branch"],
                "capsule_id": payload["capsule_id"],
                "load_case_id": payload["load_case_id"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
