from __future__ import annotations

from pathlib import Path
from typing import Any

from ..contracts import HARNESS_RESULT_SCHEMA_ID
from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from .acceptance_metrics import evaluate_task_acceptance
from .certificate import write_certificate_artifacts
from .collider_fidelity import verify_collider_fidelity_binding
from .comparison import (
    compare_ideal_dynamics_trace_files,
    compare_trace_files,
)
from .failure_evidence import build_trace_failure_evidence
from .ideal_comparison import compare_physical_to_ideal
from .io import (
    canonical_sha256,
    ensure_inside,
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .loads import default_load_cases
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .reuse import load_reuse_receipts
from .rigid_attachment import verify_rigid_attachment_binding
from .physical_ideal_comparison import (
    compare_physical_behavior_to_ideal_reference,
)
from .physical_legality import audit_physical_trace_evidence
from .whole_machine import build_whole_machine_dispatch
from .validation_scope import (
    VALIDATION_MODE_FINAL_FULL,
    VALIDATION_MODE_ITERATION_TASK,
    VALIDATION_SCOPE_SCHEMA,
)


_DEFERRED_SAVED_POSE_AUDIT = {
    "status": "DEFERRED",
    "required_environment": "outer_cad_after_queue_barrier",
    "worker_audit_authority": False,
}

_REVIEWER_TASK_POLICY_FAILURE_CODES = frozenset(
    {
        "WRENCH_POWERED_PHASE_INPUT_COVERAGE_MISSING",
        "WRENCH_RETURN_PHASE_INPUT_COVERAGE_MISSING",
        "REVIEWER_TASK_ADAPTER_POLICY_MISSING",
    }
)


def _reviewer_task_policy_failures(
    summaries: list[dict[str, Any]],
) -> tuple[str, ...]:
    """Return adapter-owned typed-task failures, or empty for mixed failures."""

    failures: list[str] = []
    for item in summaries:
        if item.get("overall_verdict") in {
            "KINEMATIC_CERTIFIED",
            "TASK_CERTIFIED",
        }:
            continue
        if item.get("verdict") not in {
            "KINEMATIC_CERTIFIED",
            "TASK_CERTIFIED",
        }:
            return ()
        raw = item.get("task_acceptance_failure_reasons")
        if not isinstance(raw, list) or not raw:
            return ()
        normalized = [str(value or "").strip() for value in raw]
        if any(
            not value
            or value.rsplit(":", 1)[-1]
            not in _REVIEWER_TASK_POLICY_FAILURE_CODES
            for value in normalized
        ):
            return ()
        failures.extend(normalized)
    return tuple(failures)


def _finalize_harness_error(
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
            source_component="mechanism.finalize",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            details=details,
        ),
    )


def _raise_finalize_evidence_error(
    exc: BaseException,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Preserve producer ownership; otherwise mark Harness evidence failure."""

    if (
        isinstance(exc, MechanismValidationError)
        and exc.failure_envelope is not None
    ):
        raise exc
    failure_details = dict(details or {})
    failure_details.setdefault(
        "cause_code", str(getattr(exc, "code", type(exc).__name__))
    )
    raise _finalize_harness_error(
        code,
        message,
        details=failure_details,
    ) from exc


def _dispatch_validation_scope(
    *,
    workspace: Path,
    output: Path,
    dispatch: dict[str, Any],
) -> dict[str, Any] | None:
    reference = dispatch.get("validation_scope_manifest")
    claim = dispatch.get("validation_scope")
    if reference is None and claim is None:
        # Read-only compatibility for dispatches created before scoped runs.
        return None
    try:
        path = (
            ensure_inside(workspace, reference)
            if isinstance(reference, str)
            else None
        )
    except (OSError, TypeError, ValueError) as exc:
        _raise_finalize_evidence_error(
            exc,
            code="FINALIZE_VALIDATION_SCOPE_INVALID",
            message=f"dispatch validation scope path is invalid: {exc}",
        )
    if (
        path != (output / "validation_scope_manifest.json").resolve()
        or not isinstance(claim, dict)
    ):
        raise _finalize_harness_error(
            "FINALIZE_VALIDATION_SCOPE_INVALID",
            "dispatch validation scope is missing or inconsistent",
        )
    try:
        persisted_claim = read_json(path)
    except (OSError, TypeError, ValueError) as exc:
        _raise_finalize_evidence_error(
            exc,
            code="FINALIZE_VALIDATION_SCOPE_INVALID",
            message=f"dispatch validation scope cannot be read: {exc}",
        )
    if persisted_claim != claim:
        raise _finalize_harness_error(
            "FINALIZE_VALIDATION_SCOPE_INVALID",
            "dispatch validation scope is missing or inconsistent",
        )
    core = dict(claim)
    recorded_sha256 = core.pop("scope_sha256", None)
    if (
        claim.get("schema") != VALIDATION_SCOPE_SCHEMA
        or claim.get("mode")
        not in {
            VALIDATION_MODE_FINAL_FULL,
            VALIDATION_MODE_ITERATION_TASK,
        }
        or claim.get("certificate_authority") is not False
        or recorded_sha256 != canonical_sha256(core)
    ):
        raise _finalize_harness_error(
            "FINALIZE_VALIDATION_SCOPE_INVALID",
            "dispatch validation scope identity or self-hash is invalid",
        )
    return claim


def _dispatched_load_cases(
    plan: MechanismPlan,
    *,
    output: Path,
    scope: dict[str, Any] | None,
) -> tuple[LoadCase, ...]:
    """Read the exact load inventory materialized by modern dispatch.

    TaskAdapter cases are merged and persisted during dispatch. Re-running
    family defaults here can therefore silently drop task-owned lanes or
    produce a different inventory. Dispatches from before validation scopes
    have no persisted selector authority and retain their legacy behavior.
    """

    if scope is None:
        return default_load_cases(plan)

    selected = scope.get("selected")
    if not isinstance(selected, list):
        raise _finalize_harness_error(
            "FINALIZE_LOAD_INVENTORY_INVALID",
            "dispatch validation scope has no selected load inventory",
            details={"capsule_id": plan.capsule_id},
        )
    selected_ids: list[str] = []
    for index, selector in enumerate(selected):
        if not isinstance(selector, dict):
            raise _finalize_harness_error(
                "FINALIZE_LOAD_INVENTORY_INVALID",
                "dispatch selected load inventory contains a non-object selector",
                details={"selector_index": index},
            )
        capsule_id = selector.get("capsule_id")
        load_case_id = selector.get("load_case_id")
        if not isinstance(capsule_id, str) or not isinstance(load_case_id, str):
            raise _finalize_harness_error(
                "FINALIZE_LOAD_INVENTORY_INVALID",
                "dispatch selected load selector has an invalid identity",
                details={"selector_index": index},
            )
        if capsule_id == plan.capsule_id:
            selected_ids.append(load_case_id)
    if not selected_ids or len(selected_ids) != len(set(selected_ids)):
        raise _finalize_harness_error(
            "FINALIZE_LOAD_INVENTORY_INVALID",
            "dispatch selected load inventory is missing or duplicated for "
            f"{plan.capsule_id!r}",
            details={
                "capsule_id": plan.capsule_id,
                "selected_load_case_ids": selected_ids,
            },
        )

    loaded: list[LoadCase] = []
    for load_case_id in selected_ids:
        load_path = (
            output
            / "loads"
            / plan.capsule_id
            / f"{load_case_id}.json"
        )
        try:
            load_case = LoadCase.from_dict(read_json(load_path))
        except (KeyError, OSError, TypeError, ValueError) as exc:
            _raise_finalize_evidence_error(
                exc,
                code="FINALIZE_LOAD_INVENTORY_INVALID",
                message=(
                    "dispatch load file is missing or malformed for "
                    f"{plan.capsule_id}/{load_case_id}: {exc}"
                ),
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                },
            )
        if (
            load_case.load_case_id != load_case_id
            or load_case.family != plan.family
        ):
            raise _finalize_harness_error(
                "FINALIZE_LOAD_INVENTORY_INVALID",
                "dispatch load file identity does not match its selector for "
                f"{plan.capsule_id}/{load_case_id}",
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                    "stored_load_case_id": load_case.load_case_id,
                    "stored_family": load_case.family,
                    "plan_family": plan.family,
                },
            )
        loaded.append(load_case)
    return tuple(loaded)


def _scoped_load_cases(
    plan: MechanismPlan,
    all_load_cases: tuple[LoadCase, ...],
    scope: dict[str, Any] | None,
) -> tuple[LoadCase, ...]:
    if (
        scope is None
        or scope.get("mode") == VALIDATION_MODE_FINAL_FULL
    ):
        return all_load_cases
    selected = scope.get("selected")
    selected_ids = [
        item.get("load_case_id")
        for item in selected
        if isinstance(item, dict)
        and item.get("capsule_id") == plan.capsule_id
    ] if isinstance(selected, list) else []
    inventory = {
        load_case.load_case_id: load_case
        for load_case in all_load_cases
    }
    if (
        not selected_ids
        or len(selected_ids) != len(set(selected_ids))
        or any(load_case_id not in inventory for load_case_id in selected_ids)
    ):
        raise MechanismValidationError(
            "FINALIZE_VALIDATION_SCOPE_INVALID",
            f"iteration scope selectors are invalid for {plan.capsule_id!r}",
        )
    return tuple(inventory[load_case_id] for load_case_id in selected_ids)


def _inconclusive_comparison(
    plan: MechanismPlan,
    load_case: LoadCase,
    *,
    code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "purpose": load_case.purpose,
        "program": load_case.program,
        "required_for_certificate": load_case.required_for_certificate,
        "verdict": "PHYSICS_INCONCLUSIVE",
        "failure_reasons": [code],
        "detail": detail,
    }


def _inconclusive_ideal_comparison(
    plan: MechanismPlan,
    load_case: LoadCase,
    *,
    code: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "schema_version": "ideal_reference_comparison_v1",
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "program": load_case.program,
        "status": "INCONCLUSIVE",
        "verdict": "PHYSICS_INCONCLUSIVE",
        "certificate_authority": False,
        "contact_evidence_authority": False,
        "failure_reasons": [code],
        "detail": detail,
    }


def _read_bound_trace(
    path: Path,
    *,
    branch: str,
    plan: MechanismPlan,
    load_case: LoadCase,
    expected_validator_source_sha256: str | None = None,
    expected_plan_sha256: str | None = None,
    expected_load_case_sha256: str | None = None,
    expected_package_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a trace's embedded identity.

    The expected plan/load/package hashes default to this run's
    identities.  A signed-off lane passes the *origin* identities from its
    chain-verified receipt instead — the trace bytes still name the run
    that executed them, and that claim is checked, never rewritten.
    """

    trace = read_trace_json(path)
    expected = {
        "status": "COMPLETED",
        "branch": branch,
        "package_binding_sha256": (
            expected_package_binding_sha256
            if expected_package_binding_sha256 is not None
            else plan.package_binding_sha256
        ),
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": (
            expected_plan_sha256
            if expected_plan_sha256 is not None
            else canonical_sha256(plan.to_dict())
        ),
        "load_case_sha256": (
            expected_load_case_sha256
            if expected_load_case_sha256 is not None
            else canonical_sha256(load_case.to_dict())
        ),
    }
    if expected_validator_source_sha256 is not None:
        expected["validator_source_sha256"] = (
            expected_validator_source_sha256
        )
    mismatches = [
        key for key, value in expected.items()
        if trace.get(key) != value
    ]
    samples = trace.get("samples")
    if mismatches:
        raise MechanismValidationError(
            "TRACE_IDENTITY_MISMATCH",
            f"{branch} trace differs at: {', '.join(mismatches)}",
        )
    if not isinstance(samples, list) or len(samples) < 2:
        raise MechanismValidationError(
            "TRACE_SAMPLES_MISSING",
            f"{branch} trace has too few saved samples",
        )
    return trace


def _verified_reuse_receipt(
    receipt: dict[str, Any],
    *,
    plan: MechanismPlan,
    load_case: LoadCase,
    expected_branches: tuple[str, ...],
) -> dict[str, Any]:
    """Independently re-verify a sign-off receipt's semantic claims.

    The receipt already survived its hash-chain check; here finalize
    recomputes the content fingerprint from *this run's* inputs and the
    current validator source, so a receipt can never license evidence for
    inputs that differ from what it claims.  Fail-closed on any mismatch.
    """

    from .family_plugins import mechanism_adapter_for_plan
    from .io import lane_execution_source_sha256
    from .reuse import lane_fingerprint
    from .task_plugins import (
        reviewer_task_adapter_source,
        task_adapter_for_program,
    )

    plugin, _adapter = mechanism_adapter_for_plan(plan)
    task_plugin = task_adapter_for_program(load_case.program, plan=plan)
    task_source = (
        reviewer_task_adapter_source(task_plugin.plugin_id)
        if task_plugin is not None
        else None
    )
    task_plugin_binding = (
        {
            "plugin_id": task_plugin.plugin_id,
            "sha256": str(task_source["sha256"]),
        }
        if task_plugin is not None and isinstance(task_source, Mapping)
        else None
    )
    expected_fingerprint = lane_fingerprint(
        plan_document=plan.to_dict(),
        load_case_document=load_case.to_dict(),
        validator_source_sha256=lane_execution_source_sha256(),
        plugin_id=plugin.plugin_id,
        branches=expected_branches,
        task_plugin_binding=task_plugin_binding,
        extra_dropped_keys=plugin.lane_fingerprint_dropped_keys,
    )
    artifacts = receipt.get("artifacts")
    # The origin package binding may legitimately differ from the current
    # one: the fingerprint binds everything the lane physics consumes
    # (per-solid geometry hashes, family spec, load case), so a provably
    # physics-neutral package edit still licenses adoption.  The origin
    # binding is treated exactly like the origin plan hash — an origin
    # identity the adopted documents must carry, checked below.
    if (
        receipt.get("capsule_id") != plan.capsule_id
        or receipt.get("load_case_id") != load_case.load_case_id
        or receipt.get("plugin_id") != plugin.plugin_id
        or receipt.get("branches") != sorted(expected_branches)
        or receipt.get("validator_source_sha256")
        != lane_execution_source_sha256()
        or receipt.get("fingerprint") != expected_fingerprint
        or receipt.get("current_plan_sha256")
        != canonical_sha256(plan.to_dict())
        or receipt.get("current_load_case_sha256")
        != canonical_sha256(load_case.to_dict())
        or receipt.get("current_package_binding_sha256")
        != plan.package_binding_sha256
        or not isinstance(artifacts, dict)
        or not isinstance(receipt.get("origin_plan_sha256"), str)
        or not isinstance(receipt.get("origin_load_case_sha256"), str)
        or not isinstance(
            receipt.get("origin_package_binding_sha256"), str
        )
    ):
        raise MechanismValidationError(
            "REUSE_RECEIPT_INVALID",
            "sign-off receipt does not license this lane's current inputs "
            f"for {plan.capsule_id}/{load_case.load_case_id}",
        )
    return receipt


def _load_composite_lane_evidence(
    *,
    workspace: Path,
    output: Path,
    plan: MechanismPlan,
    load_case: LoadCase,
    load_path: Path,
    physical_path: Path,
    ideal_reference_path: Path,
    reuse_receipt: dict[str, Any] | None = None,
    expected_branches: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Verify one lane's receipt-bound files without rewriting any of them."""

    stored_load = LoadCase.from_dict(read_json(load_path))
    if stored_load.to_dict() != load_case.to_dict():
        raise MechanismValidationError(
            "LOAD_SUITE_CHANGED",
            f"stored validator load changed: {load_case.load_case_id}",
        )
    if reuse_receipt is not None:
        reuse_receipt = _verified_reuse_receipt(
            reuse_receipt,
            plan=plan,
            load_case=load_case,
            expected_branches=expected_branches,
        )
    expected_plan_sha256 = (
        str(reuse_receipt["origin_plan_sha256"])
        if reuse_receipt is not None
        else canonical_sha256(plan.to_dict())
    )
    expected_load_case_sha256 = (
        str(reuse_receipt["origin_load_case_sha256"])
        if reuse_receipt is not None
        else canonical_sha256(load_case.to_dict())
    )
    expected_package_binding_sha256 = (
        str(reuse_receipt["origin_package_binding_sha256"])
        if reuse_receipt is not None
        else plan.package_binding_sha256
    )
    receipt_artifacts = (
        reuse_receipt["artifacts"] if reuse_receipt is not None else {}
    )
    result_root = output / "results" / plan.capsule_id
    load_id = load_case.load_case_id
    paths = {
        "physical_trace": physical_path,
        "physical_trace_audit": (
            result_root / "physical_trace_audit" / f"{load_id}.json"
        ),
        "ideal_reference_trace": ideal_reference_path,
        "ideal_reference_comparison": (
            result_root
            / "ideal_reference_comparison"
            / f"{load_id}.json"
        ),
        "comparison": result_root / "comparison" / f"{load_id}.json",
    }
    ideal_dynamics_paths = {
        "ideal_dynamics_trace": (
            output
            / "traces"
            / plan.capsule_id
            / load_id
            / "ideal_dynamics.trace.json"
        ),
        "ideal_dynamics_comparison": (
            result_root
            / "ideal_dynamics_comparison"
            / f"{load_id}.json"
        ),
    }
    lane_path = result_root / "lanes" / f"{load_id}.json"
    lane = read_json(lane_path)
    lane_validator_source_sha256 = lane.get("validator_source_sha256")
    if (
        lane.get("schema_version")
        != "simulation_ready_cad_mechanism_load_lane_v1"
        or lane.get("status") != "PASS"
        or lane.get("package_binding_sha256")
        != expected_package_binding_sha256
        or lane.get("capsule_id") != plan.capsule_id
        or lane.get("load_case_id") != load_id
        or lane.get("plan_sha256") != expected_plan_sha256
        or lane.get("load_case_sha256") != expected_load_case_sha256
        or not isinstance(lane_validator_source_sha256, str)
        or not lane_validator_source_sha256
        or (
            reuse_receipt is not None
            and lane_validator_source_sha256
            != reuse_receipt.get("validator_source_sha256")
        )
    ):
        raise MechanismValidationError(
            "COMPOSITE_LANE_RESULT_INVALID",
            f"composite lane did not pass for {plan.capsule_id}/{load_id}",
        )
    if reuse_receipt is not None:
        lane_binding = receipt_artifacts.get("lane_result")
        if (
            not isinstance(lane_binding, dict)
            or sha256_file(lane_path) != lane_binding.get("sha256")
        ):
            raise MechanismValidationError(
                "REUSE_RECEIPT_INVALID",
                "adopted lane result does not match its sign-off receipt "
                f"for {plan.capsule_id}/{load_id}",
            )
    artifacts = lane.get("artifacts")
    if not isinstance(artifacts, dict):
        raise MechanismValidationError(
            "COMPOSITE_LANE_RESULT_INVALID",
            "composite lane has no artifact bindings",
        )
    artifact_hashes: dict[str, str] = {}
    lane_bound_paths = {
        label: path
        for label, path in paths.items()
        if label != "physical_trace_audit"
    }
    if "physical_trace_audit" in artifacts:
        raise MechanismValidationError(
            "COMPOSITE_LANE_ARTIFACT_INVALID",
            "the physics lane must not produce or bind a CAD/FCL audit",
        )
    ideal_dynamics_labels = set(ideal_dynamics_paths).intersection(artifacts)
    if ideal_dynamics_labels not in (set(), set(ideal_dynamics_paths)):
        raise MechanismValidationError(
            "COMPOSITE_LANE_ARTIFACT_INVALID",
            "ideal-dynamics trace and comparison must be bound together",
        )
    if ideal_dynamics_labels:
        lane_bound_paths.update(ideal_dynamics_paths)
    if set(artifacts) != set(lane_bound_paths):
        raise MechanismValidationError(
            "COMPOSITE_LANE_ARTIFACT_INVALID",
            "composite lane artifact inventory is open or incomplete",
        )
    for label, path in lane_bound_paths.items():
        binding = artifacts.get(label)
        digest = sha256_file(path)
        if reuse_receipt is not None:
            # Adopted artifacts keep their origin-run paths inside the
            # lane result; the receipt is the only authority allowed to
            # map those paths onto this run's byte-identical copies.
            adopted = receipt_artifacts.get(label)
            expected_file = (
                adopted.get("origin_file")
                if isinstance(adopted, dict)
                else None
            )
            expected_digest_ok = (
                isinstance(adopted, dict)
                and adopted.get("sha256") == digest
            )
        else:
            expected_file = relative_to_root(workspace, path)
            expected_digest_ok = True
        if (
            not isinstance(binding, dict)
            or binding.get("file") != expected_file
            or binding.get("sha256") != digest
            or not expected_digest_ok
        ):
            raise MechanismValidationError(
                "COMPOSITE_LANE_ARTIFACT_CHANGED",
                f"composite lane artifact changed: {label}",
            )
        artifact_hashes[label] = digest

    physical = _read_bound_trace(
        physical_path,
        branch="physical",
        plan=plan,
        load_case=load_case,
        expected_validator_source_sha256=lane_validator_source_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_load_case_sha256=expected_load_case_sha256,
        expected_package_binding_sha256=expected_package_binding_sha256,
    )
    ideal = _read_bound_trace(
        ideal_reference_path,
        branch="ideal_reference",
        plan=plan,
        load_case=load_case,
        expected_plan_sha256=expected_plan_sha256,
        expected_load_case_sha256=expected_load_case_sha256,
        expected_package_binding_sha256=expected_package_binding_sha256,
    )
    physical_audit = read_json(paths["physical_trace_audit"])
    physical_audit_digest = sha256_file(paths["physical_trace_audit"])
    physical_preflight_path = output / "results" / "physical_preflight.json"
    physical_preflight = read_json(physical_preflight_path)
    preflight_rows = physical_preflight.get("traces")
    matching_preflight_rows = []
    if isinstance(preflight_rows, list):
        matching_preflight_rows = [
            row
            for row in preflight_rows
            if (
                isinstance(row, dict)
                and row.get("capsule_id") == plan.capsule_id
                and row.get("load_case_id") == load_id
                and row.get("trace")
                == relative_to_root(workspace, physical_path)
            )
        ]
    if len(matching_preflight_rows) != 1:
        raise MechanismValidationError(
            "COMPOSITE_OUTER_AUDIT_BINDING_INVALID",
            "outer physical preflight does not identify exactly one load audit",
        )
    outer_audit_binding = matching_preflight_rows[0]
    ideal_comparison = read_json(paths["ideal_reference_comparison"])
    comparison = read_json(paths["comparison"])
    expected_evidence_bindings = {
        "physical_trace_sha256": artifact_hashes["physical_trace"],
        "ideal_reference_trace_sha256": artifact_hashes[
            "ideal_reference_trace"
        ],
    }
    if (
        physical_audit.get("passed") is not True
        or physical_audit.get("physical_trace_sha256")
        != artifact_hashes["physical_trace"]
        or physical_audit.get("audit_execution_environment")
        != "outer_cad_after_queue_barrier"
        or physical_audit.get("worker_produced_audit_accepted") is not False
        or physical_preflight.get("passed") is not True
        or outer_audit_binding.get("passed") is not True
        or outer_audit_binding.get("trace_sha256")
        != artifact_hashes["physical_trace"]
        or outer_audit_binding.get("physical_trace_audit")
        != relative_to_root(workspace, paths["physical_trace_audit"])
        or outer_audit_binding.get("physical_trace_audit_sha256")
        != physical_audit_digest
        or ideal.get("physical_trace_sha256")
        != canonical_sha256(physical)
        or ideal_comparison.get("capsule_id") != plan.capsule_id
        or ideal_comparison.get("load_case_id") != load_id
        or ideal_comparison.get("verdict") != "PASS"
        or ideal_comparison.get("evidence_bindings")
        != expected_evidence_bindings
        or comparison.get("capsule_id") != plan.capsule_id
        or comparison.get("load_case_id") != load_id
        or comparison.get("verdict") != "PASS"
        or comparison.get("saved_pose_geometry_audit")
        != _DEFERRED_SAVED_POSE_AUDIT
        or comparison.get("evidence_bindings")
        != expected_evidence_bindings
    ):
        raise MechanismValidationError(
            "COMPOSITE_LANE_SEMANTIC_RESULT_INVALID",
            f"composite lane evidence is not a PASS for {plan.capsule_id}/{load_id}",
        )
    optional_trace_files: tuple[Path, ...] = ()
    optional_generated_files: tuple[Path, ...] = ()
    optional_load_evidence: dict[str, Any] = {}
    if ideal_dynamics_labels:
        ideal_dynamics = _read_bound_trace(
            ideal_dynamics_paths["ideal_dynamics_trace"],
            branch="ideal_dynamics",
            plan=plan,
            load_case=load_case,
            expected_validator_source_sha256=(
                lane_validator_source_sha256
            ),
            expected_plan_sha256=expected_plan_sha256,
            expected_load_case_sha256=expected_load_case_sha256,
            expected_package_binding_sha256=(
                expected_package_binding_sha256
            ),
        )
        ideal_dynamics_comparison = read_json(
            ideal_dynamics_paths["ideal_dynamics_comparison"]
        )
        expected_dynamics_bindings = {
            "physical_trace_sha256": artifact_hashes["physical_trace"],
            "ideal_dynamics_trace_sha256": artifact_hashes[
                "ideal_dynamics_trace"
            ],
        }
        if (
            ideal_dynamics_comparison.get("capsule_id") != plan.capsule_id
            or ideal_dynamics_comparison.get("load_case_id") != load_id
            or ideal_dynamics_comparison.get("verdict") != "PASS"
            or ideal_dynamics_comparison.get("saved_pose_geometry_audit")
            != _DEFERRED_SAVED_POSE_AUDIT
            or ideal_dynamics_comparison.get("evidence_bindings")
            != expected_dynamics_bindings
        ):
            raise MechanismValidationError(
                "COMPOSITE_LANE_SEMANTIC_RESULT_INVALID",
                "optional ideal-dynamics lane evidence is not a bound PASS "
                f"for {plan.capsule_id}/{load_id}",
            )
        optional_trace_files = (
            ideal_dynamics_paths["ideal_dynamics_trace"],
        )
        optional_generated_files = (
            ideal_dynamics_paths["ideal_dynamics_comparison"],
        )
        optional_load_evidence = {
            "ideal_dynamics": ideal_dynamics,
            "ideal_dynamics_comparison": ideal_dynamics_comparison,
        }
    return {
        "comparison": comparison,
        "ideal_comparison": ideal_comparison,
        "trace_files": (
            physical_path,
            ideal_reference_path,
            *optional_trace_files,
        ),
        "generated_files": (
            lane_path,
            paths["physical_trace_audit"],
            paths["ideal_reference_comparison"],
            paths["comparison"],
            *optional_generated_files,
        ),
        "load_evidence": {
            "program": load_case.program,
            "load_case": load_case.to_dict(),
            "physical": physical,
            "physical_trace_audit": physical_audit,
            "ideal_reference": ideal,
            "comparison": comparison,
            "ideal_reference_comparison": ideal_comparison,
            "composite_lane": lane,
            **optional_load_evidence,
        },
    }


def _certificate_physical_trace_audits(
    *,
    workspace: Path,
    output: Path,
    result_root: Path,
    plan: MechanismPlan,
    load_cases: tuple[LoadCase, ...],
    traces_by_load_case: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Bind each outer CAD audit and physical trace into certificate evidence."""

    records: list[dict[str, Any]] = []
    for load_case in load_cases:
        load_case_id = load_case.load_case_id
        report_path = (
            result_root / "physical_trace_audit" / f"{load_case_id}.json"
        )
        trace_path = (
            output
            / "traces"
            / plan.capsule_id
            / load_case_id
            / "physical.trace.json"
        )
        load_evidence = traces_by_load_case.get(load_case_id) or {}
        materialized_audit = load_evidence.get("physical_trace_audit")
        failures: list[str] = []
        report: dict[str, Any] = {}
        report_sha256: str | None = None
        if not report_path.is_file() or report_path.is_symlink():
            raise _finalize_harness_error(
                "FINALIZE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
                "the persisted physical trace audit report is missing",
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                },
            )
        else:
            try:
                report = read_json(report_path)
                report_sha256 = sha256_file(report_path)
            except (OSError, ValueError) as exc:
                _raise_finalize_evidence_error(
                    exc,
                    code="FINALIZE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
                    message="the persisted physical trace audit is unreadable",
                    details={
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case_id,
                    },
                )
        report_binding_verified = (
            isinstance(materialized_audit, dict)
            and bool(report)
            and materialized_audit == report
        )
        if not report_binding_verified:
            raise _finalize_harness_error(
                "FINALIZE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
                "the persisted physical trace audit binding changed",
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                },
            )

        physical_trace_sha256: str | None = None
        if not trace_path.is_file() or trace_path.is_symlink():
            raise _finalize_harness_error(
                "FINALIZE_PHYSICAL_TRACE_EVIDENCE_INVALID",
                "the persisted physical trace is missing",
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                },
            )
        else:
            try:
                physical_trace_sha256 = sha256_file(trace_path)
            except OSError as exc:
                _raise_finalize_evidence_error(
                    exc,
                    code="FINALIZE_PHYSICAL_TRACE_EVIDENCE_INVALID",
                    message="the persisted physical trace is unreadable",
                    details={
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case_id,
                    },
                )
        physical_trace_binding_verified = (
            physical_trace_sha256 is not None
            and report.get("physical_trace_sha256")
            == physical_trace_sha256
        )
        if not physical_trace_binding_verified:
            raise _finalize_harness_error(
                "FINALIZE_PHYSICAL_TRACE_EVIDENCE_INVALID",
                "the physical trace no longer matches its outer audit",
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                },
            )
        if (
            report.get("audit_execution_environment")
            != "outer_cad_after_queue_barrier"
            or report.get("worker_produced_audit_accepted") is not False
        ):
            raise _finalize_harness_error(
                "FINALIZE_PHYSICAL_AUDIT_AUTHORITY_INVALID",
                "the persisted physical trace audit has invalid authority",
                details={
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case_id,
                },
            )
        failures.extend(
            str(reason) for reason in report.get("failure_reasons") or ()
        )
        failures = list(dict.fromkeys(failures))
        passed = (
            report.get("passed") is True
            and report_binding_verified
            and physical_trace_binding_verified
            and "physical_trace_audit_authority_invalid" not in failures
        )
        status = report.get("status") or "PHYSICAL_AUDIT_MISSING"
        if report.get("passed") is True and not passed:
            status = "PHYSICAL_AUDIT_BINDING_FAILED"
        records.append(
            {
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case_id,
                "status": status,
                "passed": passed,
                "failure_reasons": failures,
                "reduced_branch_authorized": report.get(
                    "reduced_branch_authorized"
                ),
                "report": (
                    relative_to_root(workspace, report_path)
                    if report_path.is_file()
                    else None
                ),
                "report_sha256": report_sha256,
                "report_binding_verified": report_binding_verified,
                "physical_trace_sha256": physical_trace_sha256,
                "physical_trace_binding_verified": (
                    physical_trace_binding_verified
                ),
                "audit_execution_environment": report.get(
                    "audit_execution_environment"
                ),
                "worker_produced_audit_accepted": report.get(
                    "worker_produced_audit_accepted"
                ),
                "broad_body_pair_collision_exclusions": report.get(
                    "broad_body_pair_collision_exclusions"
                ),
                "geometry_legality": report.get("geometry_legality"),
                "component_contact_intent": report.get(
                    "component_contact_intent"
                ),
                "moving_chain_path_audit": report.get(
                    "moving_chain_path_audit"
                ),
            }
        )
    return tuple(records)


def build_run_status_document(
    *,
    summaries: list[dict[str, Any]],
    ideal_reference_summary: Any,
    iteration_diagnostic_only: bool,
    replacement_authorized: bool,
    replacement_policy: str,
) -> dict[str, Any]:
    """Assemble the terminal run_status document for every finalize path.

    ATTENTION AI EDITORS (2026-08-09): schema_version MUST stay
    HARNESS_RESULT_SCHEMA_ID — never a string literal.  The publish gate
    (contracts.publish_run_status) validates this exact document against
    the frozen result contract before it reaches disk; a hardcoded version
    literal here is how DEFECT #5 shipped.  Producer and contract move
    together through the constant, or not at all.
    """
    if iteration_diagnostic_only:
        status = "ITERATION_DIAGNOSTIC_ONLY"
    elif summaries and all(
        item["overall_verdict"] == "TASK_CERTIFIED"
        for item in summaries
    ):
        status = "TASK_CERTIFIED"
    elif summaries and all(
        item["overall_verdict"] == "KINEMATIC_CERTIFIED"
        for item in summaries
    ):
        status = "KINEMATIC_CERTIFIED"
    else:
        status = "NOT_CERTIFIED"
    run_status = {
        "schema_version": HARNESS_RESULT_SCHEMA_ID,
        "status": status,
        "last_completed_stage": (
            "iteration_diagnostics"
            if iteration_diagnostic_only
            else "verdict_and_certificate"
        ),
        "mechanisms": summaries,
        "ideal_reference": ideal_reference_summary,
        "rigid_attachment_reports": [
            item["rigid_attachment"] for item in summaries
        ],
        "collider_fidelity_reports": [
            item["collider_fidelity"] for item in summaries
        ],
        "whole_machine_replacement_ready": (
            not iteration_diagnostic_only
            and
            replacement_authorized
            and
            status in {"KINEMATIC_CERTIFIED", "TASK_CERTIFIED"}
            and all(
                item["whole_machine_dispatch"] is not None for item in summaries
            )
        ),
        "whole_machine_replacement_required": (
            False if iteration_diagnostic_only else replacement_authorized
        ),
        "replacement_policy": (
            "iteration_diagnostic_no_replacement"
            if iteration_diagnostic_only
            else replacement_policy
        ),
        "reduced_branch_decision_authority": False,
    }
    if iteration_diagnostic_only:
        run_status["certificate_authority"] = False
        run_status["terminal_authority"] = False
    elif status == "NOT_CERTIFIED" and summaries:
        reviewer_task_failures = _reviewer_task_policy_failures(summaries)
        failure_owner = (
            FailureOwner.REVIEWER_TASK_PLUGIN
            if reviewer_task_failures
            else FailureOwner.DESIGN_PACKAGE
        )
        run_status["failure_reason"] = "CERTIFICATE_NOT_CERTIFIED"
        run_status["failure"] = make_failure_envelope(
            "CERTIFICATE_NOT_CERTIFIED",
            failure_owner,
            source_component="mechanism.finalize.certificate_verdict",
            message="one or more measured mechanism certificates did not pass",
            evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            plugin_kind=(
                "task"
                if failure_owner is FailureOwner.REVIEWER_TASK_PLUGIN
                else None
            ),
            registration_source=(
                "reviewer_task_adapter"
                if failure_owner is FailureOwner.REVIEWER_TASK_PLUGIN
                else None
            ),
            details=(
                {
                    "task_acceptance_failure_reasons": list(
                        reviewer_task_failures
                    )
                }
                if reviewer_task_failures
                else None
            ),
            causes=[
                {
                    "code": str(
                        item.get("overall_verdict") or "NOT_CERTIFIED"
                    ),
                    "message": "mechanism certificate did not pass",
                    "details": {
                        "capsule_id": item.get("capsule_id"),
                        "mechanism_verdict": item.get("verdict"),
                        "typed_task_verdict": item.get(
                            "typed_task_verdict"
                        ),
                        "task_acceptance_failure_reasons": item.get(
                            "task_acceptance_failure_reasons"
                        ),
                    },
                }
                for item in summaries
                if item.get("overall_verdict")
                not in {"KINEMATIC_CERTIFIED", "TASK_CERTIFIED"}
            ],
        )
    return run_status


def finalize_dispatch(
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(workspace)
    except ValueError as exc:
        raise _finalize_harness_error(
            "FINALIZE_OUTPUT_OUTSIDE_WORKSPACE",
            "finalize output must stay inside the workspace",
        ) from exc
    dispatch = read_json(output / "dispatch_summary.json")
    validation_scope = _dispatch_validation_scope(
        workspace=workspace,
        output=output,
        dispatch=dispatch,
    )
    composite_materialized = (
        dispatch.get("execution")
        in {"per_load_composite_lane_v1", "per_load_pipeline_v2"}
    )
    # Sign-off receipts license adopted lane evidence. Loading verifies the
    # hash chain; a damaged chain aborts finalize instead of degrading.
    receipts_by_lane = {
        (str(entry.get("capsule_id")), str(entry.get("load_case_id"))): entry
        for entry in load_reuse_receipts(output)
    }
    scope_branches_by_lane: dict[tuple[str, str], tuple[str, ...]] = {}
    if isinstance(validation_scope, dict):
        for record in validation_scope.get("required_branches") or []:
            if isinstance(record, dict):
                scope_branches_by_lane[
                    (
                        str(record.get("capsule_id")),
                        str(record.get("load_case_id")),
                    )
                ] = tuple(record.get("branches") or ())
    iteration_diagnostic_only = (
        validation_scope is not None
        and validation_scope.get("mode") == VALIDATION_MODE_ITERATION_TASK
    )
    plan_files = dispatch.get("plan_files")
    if not isinstance(plan_files, list) or not plan_files:
        raise MechanismValidationError(
            "FINALIZE_PLANS_MISSING", "dispatch summary has no mechanism plans"
        )
    initial_geometry_reports = dispatch.get(
        "initial_functional_geometry_reports"
    )
    if (
        not isinstance(initial_geometry_reports, list)
        or len(initial_geometry_reports) != len(plan_files)
    ):
        raise MechanismValidationError(
            "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_REQUIRED",
            "finalize requires one passing initial functional-geometry "
            "preflight for every mechanism plan",
        )
    initial_geometry_by_capsule: dict[str, dict[str, Any]] = {}
    initial_geometry_paths: list[Path] = []
    for raw in initial_geometry_reports:
        if not isinstance(raw, dict):
            raise MechanismValidationError(
                "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_INVALID",
                "dispatch contains an invalid functional-geometry report",
            )
        capsule_id = str(raw.get("capsule_id") or "")
        report_relative = raw.get("report")
        expected_sha256 = raw.get("report_sha256")
        if (
            not capsule_id
            or capsule_id in initial_geometry_by_capsule
            or not isinstance(report_relative, str)
            or not isinstance(expected_sha256, str)
        ):
            raise MechanismValidationError(
                "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_INVALID",
                "functional-geometry report binding is incomplete or duplicated",
            )
        report_path = ensure_inside(workspace, report_relative)
        if (
            not report_path.is_file()
            or report_path.is_symlink()
            or sha256_file(report_path) != expected_sha256
        ):
            raise MechanismValidationError(
                "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_CHANGED",
                "functional-geometry report is missing or changed",
            )
        report = read_json(report_path)
        if (
            report.get("capsule_id") != capsule_id
            or report.get("passed") is not True
            or report.get("physical_simulation_started") is not False
        ):
            raise MechanismValidationError(
                "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_FAILED",
                f"functional-geometry preflight did not pass for {capsule_id!r}",
            )
        initial_geometry_by_capsule[capsule_id] = raw
        initial_geometry_paths.append(report_path)
    summaries: list[dict[str, Any]] = []
    generated_files: list[Path] = [
        output / "dispatch_summary.json",
        *initial_geometry_paths,
    ]
    physical_preflight_path = output / "results" / "physical_preflight.json"
    if physical_preflight_path.is_file():
        generated_files.append(physical_preflight_path)
    ideal_reference_summary_path = (
        output / "results" / "ideal_reference_summary.json"
    )
    ideal_reference_summary = None
    if ideal_reference_summary_path.is_file():
        generated_files.append(ideal_reference_summary_path)
        ideal_reference_summary = read_json(ideal_reference_summary_path)
    replacement_authorized = len(plan_files) > 1
    replacement_policy = (
        "multi_mechanism_composition_after_physical_certification"
        if replacement_authorized
        else "reference_only_single_mechanism"
    )

    for relative_plan in plan_files:
        plan_path = ensure_inside(workspace, relative_plan)
        plan = MechanismPlan.from_dict(read_json(plan_path))
        if plan.capsule_id not in initial_geometry_by_capsule:
            raise MechanismValidationError(
                "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_REQUIRED",
                f"no functional-geometry report matches {plan.capsule_id!r}",
            )
        generated_files.append(plan_path)
        fidelity_binding = (
            plan.adapter_spec.get("collider_fidelity")
            if isinstance(plan.adapter_spec, dict)
            else None
        )
        fidelity_report = verify_collider_fidelity_binding(
            fidelity_binding,
            workspace_root=workspace,
            package_binding_sha256=plan.package_binding_sha256,
        )
        if not isinstance(fidelity_binding, dict):
            raise MechanismValidationError(
                "COLLIDER_FIDELITY_BINDING_INVALID",
                "verified collider-fidelity binding is not a mapping",
            )
        attachment_binding = fidelity_binding.get("rigid_attachment_proof")
        attachment_report = verify_rigid_attachment_binding(
            attachment_binding,
            workspace_root=workspace,
            package_binding_sha256=plan.package_binding_sha256,
        )
        if not isinstance(attachment_binding, dict):
            raise MechanismValidationError(
                "RIGID_ATTACHMENT_BINDING_INVALID",
                "verified rigid-attachment binding is not a mapping",
            )
        attachment_report_path = ensure_inside(
            workspace, str(attachment_binding["report"])
        )
        generated_files.append(attachment_report_path)
        attachment_summary = {
            "report": relative_to_root(workspace, attachment_report_path),
            "report_sha256": str(attachment_binding["report_sha256"]),
            "schema_version": attachment_report["schema_version"],
            "status": attachment_report["status"],
            "body_count": attachment_report.get("body_count"),
            "solid_count": attachment_report.get("solid_count"),
        }
        fidelity_report_path = ensure_inside(
            workspace, str(fidelity_binding["report"])
        )
        generated_files.append(fidelity_report_path)
        fidelity_summary = {
            "report": relative_to_root(workspace, fidelity_report_path),
            "report_sha256": str(fidelity_binding["report_sha256"]),
            "schema_version": fidelity_report["schema_version"],
            "status": fidelity_report["status"],
            "component_count": fidelity_report.get("component_count"),
            "rigid_body_count": fidelity_report.get("rigid_body_count"),
        }
        expected_loads = _dispatched_load_cases(
            plan,
            output=output,
            scope=validation_scope,
        )
        comparisons: list[dict[str, Any]] = []
        trace_files: list[Path] = []
        result_root = output / "results" / plan.capsule_id
        comparison_root = result_root / "comparison"
        ideal_comparisons: list[dict[str, Any]] = []
        traces_by_load_case: dict[str, dict[str, Any]] = {}
        ideal_comparison_root = result_root / "ideal_reference_comparison"
        for expected in expected_loads:
            load_path = (
                output / "loads" / plan.capsule_id
                / f"{expected.load_case_id}.json"
            )
            trace_root = (
                output / "traces" / plan.capsule_id
                / expected.load_case_id
            )
            physical_path = trace_root / "physical.trace.json"
            ideal_dynamics_path = (
                trace_root / "ideal_dynamics.trace.json"
            )
            reduced_path = trace_root / "reduced.trace.json"
            ideal_reference_path = trace_root / "ideal_reference.trace.json"
            generated_files.extend(
                (
                    load_path,
                    physical_path,
                    ideal_dynamics_path,
                    reduced_path,
                    ideal_reference_path,
                )
            )
            if composite_materialized:
                lane_key = (plan.capsule_id, expected.load_case_id)
                try:
                    materialized = _load_composite_lane_evidence(
                        workspace=workspace,
                        output=output,
                        plan=plan,
                        load_case=expected,
                        load_path=load_path,
                        physical_path=physical_path,
                        ideal_reference_path=ideal_reference_path,
                        reuse_receipt=receipts_by_lane.get(lane_key),
                        expected_branches=scope_branches_by_lane.get(
                            lane_key, ()
                        ),
                    )
                except (OSError, ValueError, MechanismValidationError) as exc:
                    _raise_finalize_evidence_error(
                        exc,
                        code="FINALIZE_COMPOSITE_LANE_EVIDENCE_INVALID",
                        message=(
                            "the persisted composite lane evidence is missing "
                            "or does not match its dispatch bindings"
                        ),
                        details={
                            "capsule_id": plan.capsule_id,
                            "load_case_id": expected.load_case_id,
                        },
                    )
                comparison = materialized["comparison"]
                ideal_comparison = materialized["ideal_comparison"]
                comparisons.append(comparison)
                ideal_comparisons.append(ideal_comparison)
                trace_files.extend(materialized["trace_files"])
                generated_files.extend(materialized["generated_files"])
                traces_by_load_case[expected.load_case_id] = materialized[
                    "load_evidence"
                ]
                continue
            try:
                stored_load = LoadCase.from_dict(read_json(load_path))
                if stored_load.to_dict() != expected.to_dict():
                    raise MechanismValidationError(
                        "LOAD_SUITE_CHANGED",
                        f"stored validator load changed: {expected.load_case_id}",
                    )
            except (OSError, ValueError, MechanismValidationError) as exc:
                _raise_finalize_evidence_error(
                    exc,
                    code="FINALIZE_LOAD_EVIDENCE_INVALID",
                    message=(
                        "the persisted validator load is missing or no longer "
                        "matches the dispatched load"
                    ),
                    details={
                        "capsule_id": plan.capsule_id,
                        "load_case_id": expected.load_case_id,
                    },
                )
            else:
                load_evidence: dict[str, Any] = {
                    "program": expected.program,
                    "load_case": expected.to_dict(),
                }
                physical_audit_error: MechanismValidationError | None = None
                try:
                    physical_trace = _read_bound_trace(
                        physical_path,
                        branch="physical",
                        plan=plan,
                        load_case=expected,
                    )
                    load_evidence["physical"] = physical_trace
                    trace_files.append(physical_path)
                    physical_audit = {
                        **audit_physical_trace_evidence(
                            physical_trace,
                            workspace_root=workspace,
                        ),
                        "physical_trace_sha256": sha256_file(
                            physical_path
                        ),
                        "audit_execution_environment": (
                            "outer_cad_after_queue_barrier"
                        ),
                        "worker_produced_audit_accepted": False,
                    }
                    physical_audit_path = (
                        result_root
                        / "physical_trace_audit"
                        / f"{expected.load_case_id}.json"
                    )
                    write_json_atomic(
                        physical_audit_path,
                        physical_audit,
                    )
                    generated_files.append(physical_audit_path)
                    load_evidence["physical_trace_audit"] = (
                        physical_audit
                    )
                    if physical_audit.get("passed") is not True:
                        physical_audit_error = MechanismValidationError(
                            "PHYSICAL_TRACE_AUDIT_FAILED",
                            ", ".join(
                                str(item)
                                for item in physical_audit.get(
                                    "failure_reasons"
                                ) or ()
                            ),
                        )
                except (
                    OSError,
                    ValueError,
                    MechanismValidationError,
                ) as exc:
                    _raise_finalize_evidence_error(
                        exc,
                        code="FINALIZE_PHYSICAL_TRACE_EVIDENCE_INVALID",
                        message=(
                            "the persisted physical trace or its outer audit "
                            "could not be validated"
                        ),
                        details={
                            "capsule_id": plan.capsule_id,
                            "load_case_id": expected.load_case_id,
                        },
                    )
                comparison = None
                if physical_audit_error is not None:
                    comparison = _inconclusive_comparison(
                        plan,
                        expected,
                        code=physical_audit_error.code,
                        detail=str(physical_audit_error),
                    )
                elif ideal_dynamics_path.is_file():
                    try:
                        comparison = compare_ideal_dynamics_trace_files(
                            plan,
                            expected,
                            physical_path,
                            ideal_dynamics_path,
                            workspace_root=workspace,
                        )
                        ideal_dynamics_trace = _read_bound_trace(
                            ideal_dynamics_path,
                            branch="ideal_dynamics",
                            plan=plan,
                            load_case=expected,
                        )
                        load_evidence["ideal_dynamics"] = (
                            ideal_dynamics_trace
                        )
                        trace_files.append(ideal_dynamics_path)
                    except (
                        OSError,
                        ValueError,
                        MechanismValidationError,
                    ) as exc:
                        if isinstance(exc, MechanismValidationError):
                            raise
                        _raise_finalize_evidence_error(
                            exc,
                            code="FINALIZE_IDEAL_DYNAMICS_EVIDENCE_INVALID",
                            message=(
                                "persisted ideal-dynamics evidence could not "
                                "be read or compared"
                            ),
                            details={
                                "capsule_id": plan.capsule_id,
                                "load_case_id": expected.load_case_id,
                            },
                        )
                elif reduced_path.is_file():
                    try:
                        comparison = compare_trace_files(
                            plan,
                            expected,
                            physical_path,
                            reduced_path,
                            workspace_root=workspace,
                        )
                        reduced_trace = _read_bound_trace(
                            reduced_path,
                            branch="reduced",
                            plan=plan,
                            load_case=expected,
                        )
                        load_evidence["legacy_ideal_dynamics"] = (
                            reduced_trace
                        )
                        trace_files.append(reduced_path)
                    except (
                        OSError,
                        ValueError,
                        MechanismValidationError,
                    ) as exc:
                        if isinstance(exc, MechanismValidationError):
                            raise
                        _raise_finalize_evidence_error(
                            exc,
                            code="FINALIZE_REDUCED_TRACE_EVIDENCE_INVALID",
                            message=(
                                "persisted reduced-branch evidence could not "
                                "be read or compared"
                            ),
                            details={
                                "capsule_id": plan.capsule_id,
                                "load_case_id": expected.load_case_id,
                            },
                        )
                if physical_audit_error is not None:
                    ideal_comparison = _inconclusive_ideal_comparison(
                        plan,
                        expected,
                        code=physical_audit_error.code,
                        detail=(
                            "ideal-reference comparison has no certificate "
                            "authority because the physical trace failed its "
                            f"audit: {physical_audit_error}"
                        ),
                    )
                else:
                    try:
                        if not isinstance(physical_trace, dict):
                            raise MechanismValidationError(
                                "PHYSICAL_TRACE_EVIDENCE_MISSING",
                                "physical trace is unavailable for ideal comparison",
                            )
                        ideal_trace = _read_bound_trace(
                            ideal_reference_path,
                            branch="ideal_reference",
                            plan=plan,
                            load_case=expected,
                        )
                        ideal_comparison = compare_physical_to_ideal(
                            plan,
                            expected,
                            physical_trace,
                            ideal_trace,
                        )
                        load_evidence["ideal_reference"] = ideal_trace
                        trace_files.append(ideal_reference_path)
                    except (
                        OSError,
                        ValueError,
                        MechanismValidationError,
                    ) as exc:
                        if isinstance(exc, MechanismValidationError):
                            raise
                        _raise_finalize_evidence_error(
                            exc,
                            code="FINALIZE_IDEAL_REFERENCE_EVIDENCE_INVALID",
                            message=(
                                "persisted ideal-reference evidence could not "
                                "be read or compared"
                            ),
                            details={
                                "capsule_id": plan.capsule_id,
                                "load_case_id": expected.load_case_id,
                            },
                        )
                if comparison is None:
                    if not isinstance(physical_trace, dict):
                        comparison = _inconclusive_comparison(
                            plan,
                            expected,
                            code="PHYSICAL_TRACE_EVIDENCE_MISSING",
                            detail=(
                                "physical trace is unavailable for normal "
                                "physical-versus-ideal comparison"
                            ),
                        )
                    else:
                        comparison = (
                            compare_physical_behavior_to_ideal_reference(
                                plan,
                                expected,
                                physical_trace,
                                ideal_comparison,
                                workspace_root=str(workspace),
                            )
                        )
                load_evidence["comparison"] = comparison
                load_evidence["ideal_reference_comparison"] = (
                    ideal_comparison
                )
                traces_by_load_case[expected.load_case_id] = load_evidence
            comparison_path = comparison_root / f"{expected.load_case_id}.json"
            write_json_atomic(comparison_path, comparison)
            generated_files.append(comparison_path)
            comparisons.append(comparison)
            ideal_comparison_path = (
                ideal_comparison_root / f"{expected.load_case_id}.json"
            )
            write_json_atomic(ideal_comparison_path, ideal_comparison)
            generated_files.append(ideal_comparison_path)
            ideal_comparisons.append(ideal_comparison)
        aggregate_path = comparison_root / "aggregate.json"
        write_json_atomic(
            aggregate_path,
            {
                "schema_version": "0.1",
                "capsule_id": plan.capsule_id,
                "comparison_count": len(comparisons),
                "verdict_counts": {
                    verdict: sum(
                        item.get("verdict") == verdict for item in comparisons
                    )
                    for verdict in (
                        "PASS",
                        "PHYSICAL_FAILURE",
                        "CLAIM_MISMATCH",
                        "PHYSICS_INCONCLUSIVE",
                    )
                },
                "comparison_files": [
                    relative_to_root(
                        workspace,
                        comparison_root / f"{item['load_case_id']}.json",
                    )
                    for item in comparisons
                ],
            },
        )
        generated_files.append(aggregate_path)
        ideal_aggregate_path = ideal_comparison_root / "aggregate.json"
        write_json_atomic(
            ideal_aggregate_path,
            {
                "schema_version": "ideal_reference_comparison_v1",
                "capsule_id": plan.capsule_id,
                "comparison_count": len(ideal_comparisons),
                "verdict_counts": {
                    verdict: sum(
                        item.get("verdict") == verdict
                        for item in ideal_comparisons
                    )
                    for verdict in (
                        "PASS",
                        "CLAIM_MISMATCH",
                        "PHYSICS_INCONCLUSIVE",
                    )
                },
                "comparison_files": [
                    relative_to_root(
                        workspace,
                        ideal_comparison_root
                        / f"{item['load_case_id']}.json",
                    )
                    for item in ideal_comparisons
                ],
            },
        )
        generated_files.append(ideal_aggregate_path)
        task_acceptance = evaluate_task_acceptance(
            plan,
            traces_by_load_case,
        )
        task_acceptance_path = result_root / "task_acceptance.json"
        write_json_atomic(task_acceptance_path, task_acceptance)
        generated_files.append(task_acceptance_path)
        failure_evidence, failure_files = build_trace_failure_evidence(
            plan,
            tuple(comparisons),
            workspace_root=workspace,
            output_dir=output,
        )
        generated_files.extend(failure_files)
        runtime_failure_path = output / "runtime_failure_evidence.json"
        if runtime_failure_path.is_file():
            runtime_failure = read_json(runtime_failure_path)
            runtime_cases = [
                item
                for item in runtime_failure.get("cases") or []
                if item.get("capsule_id") == plan.capsule_id
            ]
            if runtime_cases:
                failure_evidence["runtime_timeout_cases"] = runtime_cases
                failure_evidence["case_count"] = int(failure_evidence.get("case_count") or 0) + len(runtime_cases)
                generated_files.append(runtime_failure_path)
                for item in runtime_cases:
                    for field in ("snapshot", "sidecar"):
                        if item.get(field):
                            generated_files.append(ensure_inside(workspace, str(item[field])))
        certificate_path = result_root / "certificate.json"
        replacement_path = result_root / "replacement.json"
        if iteration_diagnostic_only:
            summaries.append(
                {
                    "capsule_id": plan.capsule_id,
                    "verdict": "ITERATION_DIAGNOSTIC_ONLY",
                    "overall_verdict": "ITERATION_DIAGNOSTIC_ONLY",
                    "typed_task_verdict": task_acceptance.get("verdict"),
                    "task_acceptance_failure_reasons": list(
                        task_acceptance.get("failure_reasons") or ()
                    ),
                    "task_acceptance": relative_to_root(
                        workspace, task_acceptance_path
                    ),
                    "ideal_reference_comparison": relative_to_root(
                        workspace, ideal_aggregate_path
                    ),
                    "certificate": None,
                    "certificate_sha256": None,
                    "replacement": None,
                    "replacement_policy": (
                        "iteration_diagnostic_no_replacement"
                    ),
                    "whole_machine_dispatch": None,
                    "whole_machine_dispatch_error": None,
                    "failure_evidence": failure_evidence,
                    "rigid_attachment": attachment_summary,
                    "collider_fidelity": fidelity_summary,
                    "certificate_authority": False,
                    "terminal_authority": False,
                }
            )
            continue
        physical_trace_audits = _certificate_physical_trace_audits(
            workspace=workspace,
            output=output,
            result_root=result_root,
            plan=plan,
            load_cases=expected_loads,
            traces_by_load_case=traces_by_load_case,
        )
        certificate, replacement = write_certificate_artifacts(
            plan,
            expected_loads,
            tuple(comparisons),
            certificate_path=certificate_path,
            replacement_path=replacement_path,
            trace_files=(path for path in trace_files if path.is_file()),
            workspace_root=workspace,
            task_acceptance=task_acceptance,
            ideal_reference_comparisons=tuple(ideal_comparisons),
            physical_trace_audits=physical_trace_audits,
            replacement_authorized=replacement_authorized,
            replacement_policy=replacement_policy,
        )
        generated_files.append(certificate_path)
        whole_machine_summary: dict[str, Any] | None = None
        whole_machine_error: dict[str, str] | None = None
        if replacement is not None:
            generated_files.append(replacement_path)
            try:
                whole_machine_summary = build_whole_machine_dispatch(
                    plan,
                    workspace_root=workspace,
                    output_dir=result_root,
                    certificate_file=certificate_path,
                    replacement_file=replacement_path,
                )
                generated_files.extend(
                    (
                        workspace / whole_machine_summary["queue_manifest"],
                        workspace / whole_machine_summary["execution_plan"],
                        result_root / "whole_machine" / "dispatch_summary.json",
                    )
                )
            except (KeyError, OSError, ValueError, MechanismValidationError) as exc:
                if isinstance(exc, MechanismValidationError):
                    raise
                _raise_finalize_evidence_error(
                    exc,
                    code="FINALIZE_WHOLE_MACHINE_DISPATCH_INVALID",
                    message="whole-machine dispatch could not be materialized",
                    details={"capsule_id": plan.capsule_id},
                )
        elif replacement_path.is_file():
            generated_files.append(replacement_path)
        summaries.append(
            {
                "capsule_id": plan.capsule_id,
                "verdict": certificate["verdict"],
                "overall_verdict": certificate["overall_verdict"],
                "typed_task_verdict": certificate["typed_task_verdict"],
                "task_acceptance_failure_reasons": list(
                    task_acceptance.get("failure_reasons") or ()
                ),
                "task_acceptance": relative_to_root(
                    workspace, task_acceptance_path
                ),
                "ideal_reference_comparison": relative_to_root(
                    workspace, ideal_aggregate_path
                ),
                "certificate": relative_to_root(workspace, certificate_path),
                "certificate_sha256": sha256_file(certificate_path),
                "replacement": (
                    relative_to_root(workspace, replacement_path)
                    if replacement is not None or replacement_path.is_file()
                    else None
                ),
                "replacement_policy": replacement_policy,
                "whole_machine_dispatch": whole_machine_summary,
                "whole_machine_dispatch_error": whole_machine_error,
                "failure_evidence": failure_evidence,
                "rigid_attachment": attachment_summary,
                "collider_fidelity": fidelity_summary,
            }
        )

    run_status = build_run_status_document(
        summaries=summaries,
        ideal_reference_summary=ideal_reference_summary,
        iteration_diagnostic_only=iteration_diagnostic_only,
        replacement_authorized=replacement_authorized,
        replacement_policy=replacement_policy,
    )
    return run_status
