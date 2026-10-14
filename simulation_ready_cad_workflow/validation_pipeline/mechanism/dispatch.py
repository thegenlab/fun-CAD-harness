from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, NoReturn

from ..design_quality import snapshot_design_quality
from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
    mechanism_owner_for_registration_source,
)
from ..runtime import (
    JobKind,
    JobSpec,
    QueueManifest,
    QueuePolicy,
    ResourcePolicy,
)
from .adapters import AdapterSpec, adapter_from_dict, adapter_to_dict, get_adapter
from .family_plugins import (
    mechanism_adapter_for,
    plugin_registration_for,
    reviewer_adapter_source,
)
from .task_plugins import (
    claimed_reviewer_task_programs,
    merge_mechanism_and_task_load_cases,
    reviewer_task_adapter_source,
    task_adapter_for_program,
)
from .io import (
    MECHANISM_SOURCE_SHA256_ENV,
    canonical_sha256,
    ensure_inside,
    lane_execution_source_sha256,
    read_json,
    relative_to_root,
    sha256_file,
    validator_import_root,
    write_json_atomic,
)
from .initial_functional_geometry import (
    build_initial_functional_geometry_preflight,
)
from .launcher import PyChronoLauncher, select_pychrono_launcher
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import plugin_boundary_error
from .physical_legality import audit_dispatch_home_interpenetration
from .promotion import (
    initial_functional_geometry_identity_sha256,
    load_lane_promotion_candidate,
)
from .reuse import (
    append_reuse_receipt,
    find_lane_sign_off_source,
    lane_fingerprint,
    reset_reuse_receipts,
    verify_lane_execution_root,
    write_reuse_manifest,
)
from .validation_scope import (
    VALIDATION_MODE_FINAL_FULL,
    resolve_validation_scope,
    validate_validation_mode,
)


_MAX_PHYSICAL_TASKS = 8
_MAX_NONPHYSICAL_TASKS = 8
AUDITED_PHYSICAL_TASK_BUDGETS: Mapping[str, int] = {
    "deadbolt_door_lock": 9,
    "spring_scale_prismatic_spring": 10,
    "spring_scale_prismatic_spring_v1": 10,
    "spring_scale_guided_carriage": 10,
    "spring_scale_calibrated_prismatic": 10,
    "folding_knife_liner_lock": 9,
}


def audited_physical_task_budget_for_kinds(
    mechanism_kinds: Sequence[str] | set[str] | frozenset[str],
) -> tuple[str, int] | None:
    """Return one exact reviewed task-rich capacity, if present."""

    available = {str(item) for item in mechanism_kinds}
    return next(
        (
            (kind, budget)
            for kind, budget in AUDITED_PHYSICAL_TASK_BUDGETS.items()
            if kind in available
        ),
        None,
    )


def _dispatch_physical_task_budget(plans: Sequence[MechanismPlan]) -> int:
    """Allow audited task-rich suites without widening ordinary dispatches."""

    mechanism_kinds: set[str] = set()
    for plan in plans:
        adapter_spec = plan.adapter_spec
        family_spec = (
            adapter_spec.get("family_spec")
            if isinstance(adapter_spec, Mapping)
            else None
        )
        if isinstance(family_spec, Mapping):
            kind = str(family_spec.get("mechanism_kind") or "")
            if kind:
                mechanism_kinds.add(kind)
    audited = audited_physical_task_budget_for_kinds(mechanism_kinds)
    if audited is not None:
        return audited[1]
    return _MAX_PHYSICAL_TASKS
_PHYSICAL_TASK_TIMEOUT_SECONDS = 300.0
_HIGH_COMPONENT_PHYSICAL_TASK_TIMEOUT_SECONDS = 900.0
_NONPHYSICAL_TASK_TIMEOUT_SECONDS = 60.0
# A load lane owns the live physics engine, a full-cadence physical trace, and
# its derived ideal evidence. Real bicycle lanes reached 0.758 GiB before
# final trace persistence under the former 0.75 GiB hard limit. These values
# reserve the observed working set plus serialization/comparison headroom.
_LOAD_LANE_ESTIMATED_PEAK_MEMORY_GIB = 1.0
_LOAD_LANE_MAX_PROCESS_TREE_MEMORY_GIB = 1.5


def _mechanism_callback_error(
    plugin: Any,
    registration: Mapping[str, Any] | None,
    callback: str,
    exc: BaseException,
    *,
    code: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    contract_violation: bool = False,
) -> MechanismValidationError:
    plugin_id = str(getattr(plugin, "plugin_id", ""))
    return plugin_boundary_error(
        exc,
        registration=registration,
        plugin_id=plugin_id,
        plugin_kind="mechanism",
        callback=callback,
        source_component=f"mechanism.dispatch.{callback}",
        code=code,
        message=message,
        details=details,
        contract_violation=contract_violation,
        evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
    )


def _reviewer_task_request_binding(
    plan: MechanismPlan,
    load_case: LoadCase,
    *,
    workspace: Path,
) -> dict[str, str] | None:
    """Bind the applicable Reviewer TaskAdapter into a fresh lane process."""

    plugin = task_adapter_for_program(load_case.program, plan=plan)
    if plugin is None:
        return None
    source = reviewer_task_adapter_source(plugin.plugin_id)
    if not isinstance(source, Mapping):
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_SOURCE_MISSING",
            f"reviewer task adapter {plugin.plugin_id!r} has no bound source",
        )
    path = Path(str(source.get("path") or "")).resolve()
    digest = str(source.get("sha256") or "")
    try:
        path.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_SOURCE_INVALID",
            "reviewer task adapter source is outside the workspace",
        ) from exc
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_SOURCE_CHANGED",
            "reviewer task adapter source changed before dispatch",
        )
    return {
        "plugin_id": plugin.plugin_id,
        "path": relative_to_root(workspace, path),
        "sha256": digest,
    }


def _reviewer_mechanism_job_input(
    plan: MechanismPlan,
    plugin: Any,
    *,
    workspace: Path,
) -> Path | None:
    """Resolve the selected Reviewer MechanismAdapter's immutable source.

    The request's plan hash already binds ``family_spec.reviewer_adapter``
    and ``mechanism_plugin_id`` binds the selected implementation identity.
    This helper closes the remaining queue boundary: the same source file
    must also be declared in ``JobSpec.input_files`` so the runner verifies
    it immediately before and after the worker process.
    """

    adapter_spec = plan.adapter_spec
    family_spec = (
        adapter_spec.get("family_spec")
        if isinstance(adapter_spec, Mapping)
        else None
    )
    plan_binding = (
        family_spec.get("reviewer_adapter")
        if isinstance(family_spec, Mapping)
        else None
    )
    registry_binding = reviewer_adapter_source(str(plugin.plugin_id))
    if plan_binding is None and registry_binding is None:
        return None

    plugin_id = str(getattr(plugin, "plugin_id", ""))
    try:
        registration: Mapping[str, Any] | None = plugin_registration_for(
            plugin_id
        )
    except MechanismValidationError:
        registration = None

    def source_error(code: str, message: str) -> MechanismValidationError:
        return plugin_boundary_error(
            MechanismValidationError(code, message),
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            callback="reviewer_source_binding",
            source_component="mechanism.dispatch.reviewer_source_binding",
            code=code,
            message=message,
            details={"capsule_id": plan.capsule_id},
            contract_violation=True,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        )

    if not isinstance(plan_binding, Mapping) or not isinstance(
        registry_binding, Mapping
    ):
        raise source_error(
            "REVIEWER_MECHANISM_ADAPTER_SOURCE_MISSING",
            "selected reviewer mechanism adapter has no complete plan and "
            "registry source binding",
        )
    plan_path_text = plan_binding.get("path")
    plan_digest = plan_binding.get("sha256")
    registry_path_text = registry_binding.get("path")
    registry_digest = registry_binding.get("sha256")
    if (
        not isinstance(plan_path_text, str)
        or not plan_path_text
        or not isinstance(plan_digest, str)
        or not isinstance(registry_path_text, str)
        or not registry_path_text
        or not isinstance(registry_digest, str)
    ):
        raise source_error(
            "REVIEWER_MECHANISM_ADAPTER_SOURCE_INVALID",
            "reviewer mechanism adapter source binding is incomplete",
        )

    def source_path(value: str) -> tuple[Path, Path]:
        lexical = Path(value)
        if not lexical.is_absolute():
            lexical = workspace / lexical
        return lexical, lexical.resolve()

    plan_lexical, path = source_path(plan_path_text)
    _registry_lexical, registry_path = source_path(registry_path_text)
    try:
        path.relative_to(workspace)
        registry_path.relative_to(workspace)
    except ValueError as exc:
        raise source_error(
            "REVIEWER_MECHANISM_ADAPTER_SOURCE_INVALID",
            "reviewer mechanism adapter source is outside the workspace",
        ) from exc
    if path != registry_path or plan_digest != registry_digest:
        raise source_error(
            "REVIEWER_MECHANISM_ADAPTER_SOURCE_MISMATCH",
            "final mechanism plan and selected reviewer adapter name "
            "different source bytes",
        )
    if (
        plan_lexical.is_symlink()
        or not path.is_file()
        or sha256_file(path) != plan_digest
    ):
        raise source_error(
            "REVIEWER_MECHANISM_ADAPTER_SOURCE_CHANGED",
            "reviewer mechanism adapter source changed before queue creation",
        )
    return path


@dataclass(frozen=True)
class DispatchBundle:
    output_dir: str
    queue_manifest: str
    physical_queue_manifest: str
    ideal_dynamics_queue_manifest: str | None
    validation_scope_manifest: str
    validation_scope: dict[str, Any]
    plan_files: tuple[str, ...]
    request_files: tuple[str, ...]
    expected_trace_files: tuple[str, ...]
    collider_fidelity_reports: tuple[dict[str, Any], ...]
    initial_functional_geometry_reports: tuple[dict[str, Any], ...]
    job_count: int
    execution: str = "serial"
    max_parallel_jobs: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": self.output_dir,
            "queue_manifest": self.queue_manifest,
            "physical_queue_manifest": self.physical_queue_manifest,
            "ideal_dynamics_queue_manifest": (
                self.ideal_dynamics_queue_manifest
            ),
            "validation_scope_manifest": self.validation_scope_manifest,
            "validation_scope": dict(self.validation_scope),
            "plan_files": list(self.plan_files),
            "request_files": list(self.request_files),
            "expected_trace_files": list(self.expected_trace_files),
            "collider_fidelity_reports": [
                dict(item) for item in self.collider_fidelity_reports
            ],
            "initial_functional_geometry_reports": [
                dict(item)
                for item in self.initial_functional_geometry_reports
            ],
            "job_count": self.job_count,
            "execution": self.execution,
            "max_parallel_jobs": self.max_parallel_jobs,
        }


def _requires_ideal_dynamics(plan: MechanismPlan) -> bool:
    """Preserved policy helper for callers that inspect full-run branches."""

    contract = plan.acceptance_contract
    if (
        not isinstance(contract, dict)
        or contract.get("status") != "PASS"
        or contract.get("certificate_authority") is not True
    ):
        return False
    return any(
        isinstance(requirement, dict)
        and "ideal_dynamics" in requirement.get("required_branches", [])
        for requirement in contract.get("requirements", [])
    )


def _job_id(capsule_id: str, load_case_id: str, branch: str) -> str:
    label = f"{capsule_id}-{load_case_id}-{branch}".lower()
    cleaned = "".join(
        character if character.isalnum() or character in "._-" else "-"
        for character in label
    )
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[:50].rstrip('-')}-{digest}"


def _load_pipeline_outputs(
    *,
    workspace: Path,
    destination: Path,
    capsule_id: str,
    load_case_id: str,
    include_ideal_dynamics: bool,
) -> tuple[str, ...]:
    trace_root = destination / "traces" / capsule_id / load_case_id
    result_root = destination / "results" / capsule_id
    paths = [
        trace_root / "physical.trace.json",
        trace_root / "ideal_reference.trace.json",
        result_root
        / "ideal_reference_comparison"
        / f"{load_case_id}.json",
        result_root / "comparison" / f"{load_case_id}.json",
    ]
    if include_ideal_dynamics:
        paths.extend(
            (
                trace_root / "ideal_dynamics.trace.json",
                result_root
                / "ideal_dynamics_comparison"
                / f"{load_case_id}.json",
            )
        )
    paths.append(result_root / "lanes" / f"{load_case_id}.json")
    return tuple(
        relative_to_root(workspace, path)
        for path in paths
    )


def _lane_sign_off_destinations(
    destination: Path,
    capsule_id: str,
    load_case_id: str,
) -> dict[str, Path]:
    """Canonical current-run paths for every adoptable lane artifact."""

    trace_root = destination / "traces" / capsule_id / load_case_id
    result_root = destination / "results" / capsule_id
    return {
        "physical_trace": trace_root / "physical.trace.json",
        "ideal_reference_trace": trace_root / "ideal_reference.trace.json",
        "ideal_dynamics_trace": trace_root / "ideal_dynamics.trace.json",
        "ideal_reference_comparison": (
            result_root / "ideal_reference_comparison" / f"{load_case_id}.json"
        ),
        "comparison": result_root / "comparison" / f"{load_case_id}.json",
        "ideal_dynamics_comparison": (
            result_root / "ideal_dynamics_comparison" / f"{load_case_id}.json"
        ),
        "lane_result": result_root / "lanes" / f"{load_case_id}.json",
    }


def _materialize_lane_sign_off(
    *,
    workspace: Path,
    destination: Path,
    capsule_id: str,
    load_case_id: str,
    verified_source: dict[str, Any],
) -> None:
    """Copy the verified origin artifacts byte-for-byte into this run."""

    destinations = _lane_sign_off_destinations(
        destination, capsule_id, load_case_id
    )
    for label, binding in verified_source["artifacts"].items():
        source = (workspace / str(binding["origin_file"])).resolve()
        target = destinations[label]
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = source.read_bytes()
        target.write_bytes(payload)
        if sha256_file(target) != binding["sha256"]:
            raise MechanismValidationError(
                "REUSE_SIGN_OFF_COPY_CORRUPTED",
                f"adopted artifact changed while copying: {label}",
            )


def _verified_lane_promotion_source(
    *,
    workspace: Path,
    candidates: Sequence[Mapping[str, Any]],
    job_id: str,
    capsule_id: str,
    load_case_id: str,
    fingerprint: str,
    plugin_id: str,
    branches: tuple[str, ...],
    validator_source_sha256: str,
    initial_functional_geometry_identity: str,
    include_ideal_dynamics: bool,
    task_plugin_binding: Mapping[str, str] | None,
    extra_dropped_keys: tuple[str, ...],
) -> dict[str, Any] | None:
    """Re-verify one explicitly bound diagnostic execution root.

    Candidate documents are routing hints only.  The same plan/load,
    artifact, trace-identity, and outer-audit checks used for canonical
    sibling reuse run again against the diagnostic execution root.  A miss
    leaves the normal lane job untouched.
    """

    for candidate in candidates:
        source_identity = candidate.get("source_identity")
        candidate_bindings = candidate.get("bindings")
        initial_geometry_binding = (
            candidate_bindings.get("initial_functional_geometry")
            if isinstance(candidate_bindings, Mapping)
            else None
        )
        if (
            candidate.get("job_id") != job_id
            or candidate.get("capsule_id") != capsule_id
            or candidate.get("load_case_id") != load_case_id
            or not isinstance(source_identity, Mapping)
            or source_identity.get("kind")
            != "lane_execution_source_sha256"
            or source_identity.get("sha256") != validator_source_sha256
            or not isinstance(initial_geometry_binding, Mapping)
            or not isinstance(candidate.get("execution_root"), str)
        ):
            continue
        try:
            candidate_initial_geometry = read_json(
                ensure_inside(
                    workspace,
                    str(initial_geometry_binding.get("file") or ""),
                )
            )
            if (
                initial_functional_geometry_identity_sha256(
                    candidate_initial_geometry
                )
                != initial_functional_geometry_identity
            ):
                continue
            execution_root = ensure_inside(
                workspace, str(candidate["execution_root"])
            )
        except (OSError, ValueError):
            continue
        verified = verify_lane_execution_root(
            workspace_root=workspace,
            origin_run_dir=execution_root,
            capsule_id=capsule_id,
            load_case_id=load_case_id,
            fingerprint=fingerprint,
            plugin_id=plugin_id,
            branches=branches,
            validator_source_sha256=validator_source_sha256,
            include_ideal_dynamics=include_ideal_dynamics,
            task_plugin_binding=task_plugin_binding,
            extra_dropped_keys=extra_dropped_keys,
        )
        if verified is not None:
            return verified
    return None


def _content_bound_job_inputs(
    plan: MechanismPlan,
    *,
    workspace_root: Path,
    direct_files: tuple[Path, ...],
) -> dict[str, str]:
    """Bind every immutable request/package input consumed by a worker."""

    package = Path(plan.package_dir).resolve()
    candidates = {
        path.resolve()
        for path in (
            *direct_files,
            Path(plan.intake_report),
            package / "package.json",
            *(package / relative for relative in plan.artifact_hashes),
        )
    }
    bindings: dict[str, str] = {}
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        relative = relative_to_root(workspace_root, path)
        digest = sha256_file(path)
        expected = (
            plan.intake_report_sha256
            if path == Path(plan.intake_report).resolve()
            else plan.artifact_hashes.get(
                path.relative_to(package).as_posix()
            )
            if path.is_relative_to(package)
            else None
        )
        if expected is not None and digest != expected:
            raise MechanismValidationError(
                "JOB_INPUT_BINDING_CHANGED",
                f"declared worker input changed before queue creation: {relative}",
            )
        bindings[relative] = digest
    return dict(sorted(bindings.items()))


def _job_runtime_budget(
    adapter: Any,
    *,
    branch: str,
    estimated_steps: int,
) -> tuple[float, float]:
    """Apply the global per-task wall-clock ceiling.

    Complexity estimates remain useful for a conservative ETA, but they may
    never enlarge a task beyond the operator-approved five-minute physical or
    one-minute non-physical budget.
    """

    if branch != "physical":
        estimate = max(1.0, 5.0 + estimated_steps * 0.004)
        return (
            _NONPHYSICAL_TASK_TIMEOUT_SECONDS,
            min(_NONPHYSICAL_TASK_TIMEOUT_SECONDS, estimate),
        )
    family_spec = adapter.family_spec if isinstance(adapter.family_spec, dict) else {}
    collision_geometry = family_spec.get("collision_geometry")
    if not isinstance(collision_geometry, dict):
        return _PHYSICAL_TASK_TIMEOUT_SECONDS, 90.0
    component_count = max(1, int(collision_geometry.get("component_count") or 1))
    complexity_scale = max(1.0, component_count / 120.0) ** 0.35
    timeout = (
        _HIGH_COMPONENT_PHYSICAL_TASK_TIMEOUT_SECONDS
        if component_count > 80
        else _PHYSICAL_TASK_TIMEOUT_SECONDS
    )
    estimate = 30.0 + estimated_steps * 0.035 * complexity_scale
    return (
        timeout,
        min(timeout, max(30.0, estimate)),
    )


def _family_static_geometry_checks(
    adapter: Any,
    *,
    workspace_root: Path,
    capsule_id: str,
) -> list[dict[str, Any]]:
    family_spec = (
        adapter.family_spec
        if isinstance(adapter.family_spec, dict)
        else {}
    )
    collision_geometry = family_spec.get("collision_geometry")
    if not isinstance(collision_geometry, dict):
        return []
    binding = collision_geometry.get("static_geometry_preflight")
    if not isinstance(binding, dict):
        return [
            {
                "capsule_id": capsule_id,
                "check_id": "family_static_geometry",
                "status": "FAIL",
                "passed": False,
                "failure_reason": "aggregate_evidence_missing",
            }
        ]
    report_ref = binding.get("report")
    report_path = (
        (workspace_root / str(report_ref)).resolve()
        if isinstance(report_ref, str) and report_ref
        else None
    )
    if (
        report_path is None
        or not report_path.is_file()
        or not report_path.is_relative_to(workspace_root)
        or sha256_file(report_path) != binding.get("report_sha256")
    ):
        return [
            {
                "capsule_id": capsule_id,
                "check_id": "family_static_geometry",
                "status": "FAIL",
                "passed": False,
                "failure_reason": "aggregate_evidence_binding_invalid",
            }
        ]
    report = read_json(report_path)
    checks = report.get("checks")
    if not isinstance(checks, list) or not checks:
        return [
            {
                "capsule_id": capsule_id,
                "check_id": "family_static_geometry",
                "status": "FAIL",
                "passed": False,
                "failure_reason": "aggregate_evidence_checks_missing",
            }
        ]
    return [
        {
            "capsule_id": capsule_id,
            **dict(item),
        }
        for item in checks
        if isinstance(item, dict)
    ]


def _static_preflight_failure(
    checks: list[dict[str, Any]],
    *,
    capsule_id: str,
    collision_geometry: dict[str, Any] | None,
    plugin_owner: FailureOwner = FailureOwner.DESIGN_PACKAGE,
    plugin_id: str | None = None,
) -> NoReturn:
    def fail(
        code: str,
        message: str,
        *,
        owner: FailureOwner = FailureOwner.DESIGN_PACKAGE,
        evidence_validity: EvidenceValidity = EvidenceValidity.VALID_MEASUREMENT,
        details: Mapping[str, Any] | None = None,
        plugin_id: str | None = None,
    ) -> MechanismValidationError:
        return MechanismValidationError(
            code,
            message,
            details=details,
            failure=make_failure_envelope(
                code,
                owner,
                source_component="mechanism.dispatch.static_preflight",
                message=message,
                evidence_validity=evidence_validity,
                plugin_id=plugin_id,
                plugin_kind=(
                    "mechanism"
                    if owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
                    else None
                ),
                details=details,
            ),
        )

    failed_ids = [
        str(item.get("check_id") or "unknown")
        for item in checks
        if item.get("passed") is not True
    ]
    reviewer_missing_fidelity = next(
        (
            item
            for item in checks
            if item.get("check_id") == "collider_fidelity"
            and item.get("passed") is not True
            and item.get("failure_reason")
            == "reviewer_mechanism_adapter_collider_evidence_missing"
        ),
        None,
    )
    if reviewer_missing_fidelity is not None:
        raise fail(
            "REVIEWER_MECHANISM_ADAPTER_COLLIDER_EVIDENCE_MISSING",
            "the selected reviewer MechanismAdapter did not produce a "
            "package-bound STEP-to-collider fidelity report",
            owner=FailureOwner.REVIEWER_MECHANISM_PLUGIN,
            evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
            details={
                "plugin_id": reviewer_missing_fidelity.get("plugin_id"),
                "capsule_id": capsule_id,
            },
            plugin_id=(
                str(reviewer_missing_fidelity.get("plugin_id"))
                if reviewer_missing_fidelity.get("plugin_id")
                else None
            ),
        )
    if failed_ids == ["rigid_attachment"]:
        raise fail(
            "RIGID_ATTACHMENT_PROOF_FAILED",
            f"capsule {capsule_id!r} has no passing rigid-attachment proof",
        )
    if failed_ids == ["assembly_connectivity"]:
        assembly_check = next(
            (
                item
                for item in checks
                if item.get("check_id") == "assembly_connectivity"
            ),
            {},
        )
        reasons = assembly_check.get("failure_reasons") or []
        detail = "; ".join(str(item) for item in reasons[:3])
        raise fail(
            "ASSEMBLY_CONNECTIVITY_PREFLIGHT_FAILED",
            "STEP-derived mechanism bodies form disconnected spatial "
            f"assembly islands for {capsule_id!r}"
            + (f": {detail}" if detail else ""),
        )
    if failed_ids == ["collider_fidelity"]:
        collider_check = next(
            (
                item
                for item in checks
                if item.get("check_id") == "collider_fidelity"
            ),
            {},
        )
        if (
            collider_check.get("failure_reason")
            == "collider_fidelity_evidence_missing"
            or not collider_check.get("report")
        ):
            raise fail(
                "COLLIDER_FIDELITY_PREFLIGHT_FAILED",
                "STEP-to-collider fidelity evidence is missing; "
                "no trusted budget report was produced",
                owner=plugin_owner,
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
                plugin_id=(
                    plugin_id
                    if plugin_owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
                    else None
                ),
            )
        raise fail(
            "COLLIDER_FIDELITY_PREFLIGHT_FAILED",
            "STEP-to-collider fidelity evidence did not pass trusted budgets",
        )
    if failed_ids == ["home_interpenetration"]:
        home_check = next(
            (
                item
                for item in checks
                if item.get("check_id") == "home_interpenetration"
            ),
            {},
        )
        worst = home_check.get("worst_events") or []
        detail = ", ".join(
            f"{' vs '.join(item['solid_pair'])}: "
            f"{float(item['penetration_depth_m']) * 1000.0:.3f} mm"
            for item in worst[:5]
        )
        raise fail(
            "FAMILY_HOME_INTERPENETRATION",
            "STEP-derived rigid bodies overlap at the declared home pose"
            + (f": {detail}" if detail else ""),
        )
    if failed_ids == ["initial_functional_geometry"]:
        raise fail(
            "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_FAILED",
            "physical dispatch blocked before simulation by initial "
            f"functional geometry for {capsule_id!r}",
        )
    aggregate_contract_check = next(
        (
            item
            for item in checks
            if item.get("check_id") == "family_static_geometry"
            and item.get("failure_reason")
            in {
                "aggregate_evidence_missing",
                "aggregate_evidence_binding_invalid",
                "aggregate_evidence_checks_missing",
            }
        ),
        None,
    )
    if aggregate_contract_check is not None:
        raise fail(
            "MECHANISM_STATIC_PREFLIGHT_FAILED",
            "family static-geometry evidence is missing or invalid for "
            f"{capsule_id!r}",
            owner=plugin_owner,
            evidence_validity=(
                EvidenceValidity.INVALID_MEASUREMENT
                if plugin_owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
                else EvidenceValidity.NOT_APPLICABLE
            ),
            plugin_id=(
                plugin_id
                if plugin_owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
                else None
            ),
        )
    known_design_checks = {
        "rigid_attachment",
        "assembly_connectivity",
        "collider_fidelity",
        "home_interpenetration",
        "initial_functional_geometry",
        "finite_linear_capture",
    }
    fallback_owner = (
        FailureOwner.DESIGN_PACKAGE
        if failed_ids
        and all(item in known_design_checks for item in failed_ids)
        else FailureOwner.UNCLASSIFIED
    )
    raise fail(
        "MECHANISM_STATIC_PREFLIGHT_FAILED",
        "physical dispatch blocked before simulation after completing all "
        f"independent static checks for {capsule_id!r}; failed: "
        + ", ".join(failed_ids),
        owner=fallback_owner,
        evidence_validity=(
            EvidenceValidity.VALID_MEASUREMENT
            if fallback_owner is FailureOwner.DESIGN_PACKAGE
            else EvidenceValidity.NOT_APPLICABLE
        ),
    )


def _reviewer_collider_fidelity(
    plugin: Any,
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace_root: Path,
    plugin_owner: FailureOwner,
) -> dict[str, Any] | None:
    """Validate the reviewer materialize() postcondition on the real package."""

    if plugin_owner is not FailureOwner.REVIEWER_MECHANISM_PLUGIN:
        fidelity = adapter.collider_fidelity
        return dict(fidelity) if isinstance(fidelity, dict) else None
    fidelity = adapter.collider_fidelity
    if not isinstance(fidelity, Mapping):
        return None
    report = fidelity.get("report")
    digest = fidelity.get("report_sha256")
    if (
        not isinstance(report, str)
        or not report
        or not isinstance(digest, str)
        or len(digest) != 64
        or not isinstance(fidelity.get("passed"), bool)
        or fidelity.get("package_binding_sha256") != plan.package_binding_sha256
    ):
        return None
    report_path = (workspace_root / report).resolve()
    try:
        report_path.relative_to(workspace_root)
    except ValueError:
        return None
    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or sha256_file(report_path) != digest
    ):
        return None
    try:
        report_document = read_json(report_path)
    except (OSError, ValueError):
        return None
    from .collider_fidelity import COLLIDER_FIDELITY_SCHEMA

    report_status = report_document.get("status")
    report_passed = report_document.get("passed")
    components = report_document.get("components")
    rigid_bodies = report_document.get("rigid_bodies")
    component_count = report_document.get("component_count")
    rigid_body_count = report_document.get("rigid_body_count")
    if (
        report_document.get("schema_version") != COLLIDER_FIDELITY_SCHEMA
        or report_status not in {"PASS", "FAIL", "INCONCLUSIVE"}
        or not isinstance(report_passed, bool)
        or report_passed is not (report_status == "PASS")
        or report_document.get("package_binding_sha256")
        != plan.package_binding_sha256
        or fidelity.get("status") != report_status
        or fidelity.get("passed") is not report_passed
        or not isinstance(report_document.get("thresholds"), Mapping)
        or not isinstance(components, list)
        or not isinstance(rigid_bodies, list)
        or not isinstance(component_count, int)
        or isinstance(component_count, bool)
        or component_count != len(components)
        or not isinstance(rigid_body_count, int)
        or isinstance(rigid_body_count, bool)
        or rigid_body_count != len(rigid_bodies)
        or not isinstance(report_document.get("failure_reasons"), list)
        or any(not isinstance(item, Mapping) for item in components)
        or any(not isinstance(item, Mapping) for item in rigid_bodies)
        or any(
            item.get("status") not in {"PASS", "FAIL", "INCONCLUSIVE"}
            or not isinstance(item.get("passed"), bool)
            for item in (*components, *rigid_bodies)
        )
    ):
        return None
    return dict(fidelity)


def _mechanism_load_cases(
    plugin: Any,
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    plugin_registration: Mapping[str, Any] | None,
) -> tuple[LoadCase, ...]:
    """Validate load cases at the selected reviewer mechanism-plugin boundary."""

    plugin_owner = mechanism_owner_for_registration_source(
        str((plugin_registration or {}).get("registration_source") or "")
    )
    reviewer_plugin = plugin_owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
    try:
        load_cases = plugin.load_cases(plan)
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise _mechanism_callback_error(
            plugin,
            plugin_registration,
            "load_cases",
            exc,
            details={
                "plugin_id": getattr(plugin, "plugin_id", None),
                "capsule_id": plan.capsule_id,
            },
        ) from exc
    except Exception as exc:
        code = "MECHANISM_ADAPTER_LOAD_CASE_FAILED"
        message = f"mechanism adapter load_cases() failed: {exc}"
        raise _mechanism_callback_error(
            plugin,
            plugin_registration,
            "load_cases",
            exc,
            code=code,
            message=message,
            details={
                "plugin_id": getattr(plugin, "plugin_id", None),
                "capsule_id": plan.capsule_id,
            },
        ) from exc
    if (
        not isinstance(load_cases, tuple)
        or any(not isinstance(item, LoadCase) for item in load_cases)
    ):
        code = "MECHANISM_ADAPTER_LOAD_CASE_INVALID"
        message = "mechanism adapter load_cases() must return a tuple of LoadCase values"
        raise _mechanism_callback_error(
            plugin,
            plugin_registration,
            "load_cases",
            TypeError(message),
            code=code,
            message=message,
            details={
                "plugin_id": getattr(plugin, "plugin_id", None),
                "capsule_id": plan.capsule_id,
            },
            contract_violation=True,
        )
    result = tuple(load_cases)
    if reviewer_plugin:
        claimed_task_programs = claimed_reviewer_task_programs(plan)
        unsupported = [
            item.program
            for item in result
            if item.program not in adapter.supported_programs
            and item.program not in claimed_task_programs
        ]
        if unsupported:
            code = "MECHANISM_ADAPTER_LOAD_CASE_PROGRAM_UNSUPPORTED"
            message = (
                "mechanism adapter emitted load cases without adapter or "
                "TaskAdapter program claims: " + ", ".join(sorted(set(unsupported)))
            )
            raise _mechanism_callback_error(
                plugin,
                plugin_registration,
                "load_cases",
                ValueError(message),
                code=code,
                message=message,
                details={
                    "plugin_id": getattr(plugin, "plugin_id", None),
                    "capsule_id": plan.capsule_id,
                    "programs": sorted(set(unsupported)),
                },
                contract_violation=True,
            )
        registry = getattr(plan, "task_channel_registry", None)
        input_body_ids = {
            str(item)
            for item in (
                registry.get("input_body_ids")
                if isinstance(registry, Mapping)
                else ()
            )
            or ()
            if str(item)
        }
        reviewer_task_programs = {
            item.program
            for item in result
            if item.program in claimed_task_programs
        }
        plugin_id = str(getattr(plugin, "plugin_id", ""))
        reviewer_owned_compound_backend = bool(
            getattr(plan, "family", None) == "compound"
            and str(getattr(plugin, "worker_backend", "")) == plugin_id
        )
        actuation_backend = str(
            getattr(plugin, "multi_input_task_actuation_backend", "") or ""
        ).strip()
        if (
            reviewer_owned_compound_backend
            and len(input_body_ids) > 1
            and reviewer_task_programs
            and not actuation_backend
        ):
            code = (
                "REVIEWER_MECHANISM_ADAPTER_MULTI_INPUT_TASK_RUNTIME_REQUIRED"
            )
            message = (
                "reviewer compound task programs select multiple input bodies "
                "without an explicit per-phase actuation backend"
            )
            raise _mechanism_callback_error(
                plugin,
                plugin_registration,
                "load_cases",
                ValueError(message),
                code=code,
                message=message,
                details={
                    "plugin_id": plugin_id,
                    "capsule_id": plan.capsule_id,
                    "programs": sorted(reviewer_task_programs),
                    "input_body_ids": sorted(input_body_ids),
                    "required_repair": (
                        "provide a reviewer-owned simulator with explicit "
                        "per-phase actuation targets"
                    ),
                },
                contract_violation=True,
            )
    return result


def _bind_final_static_preflight(
    adapter: Any,
    *,
    workspace_root: Path,
    report_path: Path,
    report: dict[str, Any],
    package_binding_sha256: str,
) -> Any:
    """Replace provisional family evidence with the final dispatch report."""

    family_spec = (
        adapter.family_spec
        if isinstance(adapter.family_spec, dict)
        else {}
    )
    collision_geometry = family_spec.get("collision_geometry")
    if not isinstance(collision_geometry, dict):
        return adapter
    if report.get("passed") is not True or report.get("status") != "PASS":
        raise MechanismValidationError(
            "MECHANISM_STATIC_PREFLIGHT_BINDING_INVALID",
            "only a passing final static preflight may authorize physical work",
        )
    collision_geometry = dict(collision_geometry)
    collision_geometry["static_geometry_preflight"] = {
        "schema_version": report["schema_version"],
        "status": report["status"],
        "passed": report["passed"],
        "report": relative_to_root(workspace_root, report_path),
        "report_sha256": sha256_file(report_path),
        "package_binding_sha256": package_binding_sha256,
    }
    family_spec = dict(family_spec)
    family_spec["collision_geometry"] = collision_geometry
    return replace(adapter, family_spec=family_spec)


def _enforce_single_source_cad_policy(
    package_dir: str | Path,
    evaluation: Any,
) -> None:
    """Require the one-authoritative-CAD contract before execution."""
    package = Path(package_dir).resolve()
    manifest = read_json(package / "package.json")
    policy = manifest.get("single_source_cad")
    if policy is None:
        raise MechanismValidationError(
            "SINGLE_SOURCE_CAD_POLICY_REQUIRED",
            "execution requires the toolbox-standard single-source CAD policy",
        )
    if not isinstance(policy, dict) or policy.get("policy") != "required_v1":
        raise MechanismValidationError(
            "SINGLE_SOURCE_CAD_POLICY_INVALID",
            "single_source_cad must declare the supported required_v1 policy",
        )
    authoritative = str(policy.get("authoritative_artifact") or "")
    if not authoritative or authoritative != manifest.get("assembly_file"):
        raise MechanismValidationError(
            "SINGLE_SOURCE_CAD_AUTHORITY_INVALID",
            "the single-source policy must bind the declared package assembly",
        )
    artifact_hashes = {
        str(item.get("path")): str(item.get("sha256") or "").lower()
        for item in manifest.get("artifacts", [])
        if isinstance(item, dict)
    }
    if artifact_hashes.get(authoritative) != evaluation.assembly_step_sha256.lower():
        raise MechanismValidationError(
            "SINGLE_SOURCE_CAD_HASH_MISMATCH",
            "the visual authority and package manifest do not bind the same STEP hash",
        )
    if not evaluation.evidence_complete:
        issue_codes = ", ".join(
            sorted({str(item.get("code") or "UNKNOWN") for item in evaluation.issues})
        )
        detail = issue_codes or evaluation.status
        raise MechanismValidationError(
            "SINGLE_SOURCE_CAD_REVIEW_BLOCKED",
            "the authoritative CAD cannot enter execution until its hash-bound "
            f"visual evidence is complete: {detail}",
        )


def build_dispatch_bundle(
    plans: tuple[MechanismPlan, ...],
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    validation_mode: str = VALIDATION_MODE_FINAL_FULL,
    pychrono_launcher: PyChronoLauncher | None = None,
    conda_executable: str | None = None,
    conda_environment: str | None = None,
    lane_replay_promotions: Sequence[Mapping[str, Any]] = (),
) -> DispatchBundle:
    if not plans:
        raise MechanismValidationError(
            "MECHANISM_PLAN_REQUIRED",
            "at least one mechanism plan is required",
        )
    resolved_validation_mode = validate_validation_mode(validation_mode)
    launcher = select_pychrono_launcher(
        pychrono_launcher,
        legacy_conda_executable=conda_executable,
        legacy_conda_environment=conda_environment,
    )
    workspace = Path(workspace_root).resolve()
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "DISPATCH_OUTPUT_OUTSIDE_WORKSPACE",
            f"dispatch output must stay inside {workspace}",
        ) from exc
    destination.mkdir(parents=True, exist_ok=True)
    reset_reuse_receipts(destination)
    design_quality = snapshot_design_quality(plans[0].package_dir, destination)
    _enforce_single_source_cad_policy(plans[0].package_dir, design_quality)
    validator_source_sha256 = lane_execution_source_sha256()
    promotion_candidates: list[dict[str, Any]] = []
    for binding in lane_replay_promotions:
        if not isinstance(binding, Mapping):
            continue
        candidate = load_lane_promotion_candidate(workspace, binding)
        if candidate is not None:
            promotion_candidates.append(candidate)

    jobs: list[JobSpec] = []
    reuse_entries: list[dict[str, Any]] = []
    total_lane_count = 0
    ideal_dynamics_task_count = 0
    plan_files: list[str] = []
    request_files: list[str] = []
    expected_trace_files: list[str] = []
    collider_fidelity_reports: list[dict[str, Any]] = []
    initial_functional_geometry_reports: list[dict[str, Any]] = []
    seen_job_ids: set[str] = set()
    prepared_dispatches: list[
        tuple[
            MechanismPlan,
            Any,
            Any,
            tuple[LoadCase, ...],
            Path,
            str,
            Path,
            str,
            str,
            Path | None,
        ]
    ] = []

    for plan in plans:
        if not plan.dispatchable or plan.adapter_id is None:
            raise MechanismValidationError(
                "MECHANISM_ADAPTER_UNAVAILABLE",
                f"no execution adapter matches capsule {plan.capsule_id!r}",
            )
        adapter = (
            adapter_from_dict(plan.adapter_spec)
            if plan.adapter_spec is not None
            else get_adapter(plan.adapter_id)
        )
        plugin = mechanism_adapter_for(plan, adapter)
        plugin_registration: Mapping[str, Any] | None = None
        try:
            plugin_registration = plugin_registration_for(plugin.plugin_id)
            plugin_owner = mechanism_owner_for_registration_source(
                str(plugin_registration.get("registration_source") or "")
            )
        except MechanismValidationError:
            plugin_owner = FailureOwner.UNCLASSIFIED
        reviewer_plugin = (
            plugin_owner is FailureOwner.REVIEWER_MECHANISM_PLUGIN
        )
        reviewer_source_binding = (
            dict(adapter.family_spec["reviewer_adapter"])
            if reviewer_plugin
            and isinstance(adapter.family_spec, Mapping)
            and isinstance(
                adapter.family_spec.get("reviewer_adapter"), Mapping
            )
            else None
        )
        try:
            materialized_adapter = plugin.materialize(
                plan,
                adapter,
                workspace_root=workspace,
                output_dir=destination,
            )
        except MechanismValidationError as exc:
            if exc.failure_envelope is not None:
                raise
            raise _mechanism_callback_error(
                plugin,
                plugin_registration,
                "materialize",
                exc,
                details={"capsule_id": plan.capsule_id},
            ) from exc
        except Exception as exc:
            code = "MECHANISM_ADAPTER_MATERIALIZE_FAILED"
            message = f"mechanism adapter materialize() failed: {exc}"
            raise _mechanism_callback_error(
                plugin,
                plugin_registration,
                "materialize",
                exc,
                code=code,
                message=message,
                details={"capsule_id": plan.capsule_id},
            ) from exc
        if not isinstance(materialized_adapter, AdapterSpec):
            code = "MECHANISM_ADAPTER_MATERIALIZE_INVALID"
            message = "mechanism adapter materialize() must return AdapterSpec"
            raise _mechanism_callback_error(
                plugin,
                plugin_registration,
                "materialize",
                TypeError(message),
                code=code,
                message=message,
                details={"capsule_id": plan.capsule_id},
                contract_violation=True,
            )
        if reviewer_source_binding is not None:
            materialized_family_spec = materialized_adapter.family_spec
            materialized_source_binding = (
                materialized_family_spec.get("reviewer_adapter")
                if isinstance(materialized_family_spec, Mapping)
                else None
            )
            if (
                materialized_source_binding is not None
                and materialized_source_binding != reviewer_source_binding
            ):
                code = "REVIEWER_MECHANISM_ADAPTER_SOURCE_CHANGED"
                message = (
                    "reviewer mechanism adapter materialize() changed its "
                    "Harness-owned source binding"
                )
                raise _mechanism_callback_error(
                    plugin,
                    plugin_registration,
                    "materialize",
                    MechanismValidationError(code, message),
                    code=code,
                    message=message,
                    details={"capsule_id": plan.capsule_id},
                    contract_violation=True,
                )
            if materialized_source_binding is None:
                materialized_adapter = replace(
                    materialized_adapter,
                    family_spec={
                        **dict(materialized_family_spec or {}),
                        "reviewer_adapter": reviewer_source_binding,
                    },
                )
        if materialized_adapter != adapter:
            adapter = materialized_adapter
            plan = replace(plan, adapter_spec=adapter_to_dict(adapter))
        fidelity = _reviewer_collider_fidelity(
            plugin,
            plan,
            adapter,
            workspace_root=workspace,
            plugin_owner=plugin_owner,
        )
        collider_fidelity_reports.append(
            {
                "capsule_id": plan.capsule_id,
                **(
                    dict(fidelity)
                    if isinstance(fidelity, dict)
                    else {
                        "status": "FAIL",
                        "passed": False,
                        "failure_reason": (
                            "reviewer_mechanism_adapter_collider_evidence_missing"
                            if reviewer_plugin
                            else "collider_fidelity_evidence_missing"
                        ),
                        **({"plugin_id": plugin.plugin_id} if reviewer_plugin else {}),
                    }
                ),
            }
        )
        plan_path = destination / "plans" / f"{plan.capsule_id}.json"
        write_json_atomic(plan_path, plan.to_dict())
        plan_relative = relative_to_root(workspace, plan_path)
        plan_files.append(plan_relative)

        initial_geometry = build_initial_functional_geometry_preflight(
            plan,
            adapter,
            worker_backend=plugin.worker_backend,
            workspace_root=workspace,
        )
        initial_geometry_path = (
            destination
            / "results"
            / "initial_functional_geometry"
            / f"{plan.capsule_id}.json"
        )
        write_json_atomic(initial_geometry_path, initial_geometry)
        initial_geometry_summary = {
            **initial_geometry,
            "report": relative_to_root(workspace, initial_geometry_path),
            "report_sha256": sha256_file(initial_geometry_path),
        }
        initial_functional_geometry_reports.append(
            initial_geometry_summary
        )
        aggregate_initial_geometry = {
            "schema_version": "initial_functional_geometry_preflight_v1",
            "status": (
                "PASS"
                if all(
                    item.get("passed") is True
                    for item in initial_functional_geometry_reports
                )
                else "FAIL"
            ),
            "passed": all(
                item.get("passed") is True
                for item in initial_functional_geometry_reports
            ),
            "mechanism_count": len(
                initial_functional_geometry_reports
            ),
            "analytical_interface_count": sum(
                int(item.get("analytical_interface_count") or 0)
                for item in initial_functional_geometry_reports
            ),
            "ungated_analytical_interface_count": sum(
                int(
                    item.get(
                        "ungated_analytical_interface_count"
                    ) or 0
                )
                for item in initial_functional_geometry_reports
            ),
            "initially_nontransmitting_interface_count": sum(
                int(
                    item.get(
                        "initially_nontransmitting_interface_count"
                    ) or 0
                )
                for item in initial_functional_geometry_reports
            ),
            "physical_simulation_started": False,
            "mechanisms": initial_functional_geometry_reports,
        }
        write_json_atomic(
            destination
            / "results"
            / "initial_functional_geometry_preflight.json",
            aggregate_initial_geometry,
        )
        family_spec = (
            adapter.family_spec
            if isinstance(adapter.family_spec, dict)
            else {}
        )
        collision_geometry = family_spec.get("collision_geometry")
        static_checks = _family_static_geometry_checks(
            adapter,
            workspace_root=workspace,
            capsule_id=plan.capsule_id,
        )
        if reviewer_plugin and fidelity is None:
            static_checks = [
                item
                for item in static_checks
                if item.get("check_id") != "collider_fidelity"
            ]
            static_checks.append(
                {
                    "capsule_id": plan.capsule_id,
                    "check_id": "collider_fidelity",
                    "status": "FAIL",
                    "passed": False,
                    "failure_reason": (
                        "reviewer_mechanism_adapter_collider_evidence_missing"
                    ),
                    "plugin_id": plugin.plugin_id,
                }
            )
        if isinstance(collision_geometry, dict):
            dispatch_home_audit = audit_dispatch_home_interpenetration(
                collision_geometry,
                initial_geometry,
            )
            dispatch_home_audit_path = (
                destination
                / "results"
                / "dispatch_home_interpenetration"
                / f"{plan.capsule_id}.json"
            )
            write_json_atomic(
                dispatch_home_audit_path,
                dispatch_home_audit,
            )
            static_checks = [
                item
                for item in static_checks
                if item.get("check_id") != "home_interpenetration"
            ]
            static_checks.append(
                {
                    "capsule_id": plan.capsule_id,
                    "check_id": "home_interpenetration",
                    "status": dispatch_home_audit["status"],
                    "passed": dispatch_home_audit["passed"],
                    "report": relative_to_root(
                        workspace,
                        dispatch_home_audit_path,
                    ),
                    "report_sha256": sha256_file(
                        dispatch_home_audit_path
                    ),
                    "runtime_tolerance_m": dispatch_home_audit[
                        "runtime_tolerance_m"
                    ],
                    "worst_events": dispatch_home_audit[
                        "worst_events"
                    ],
                }
            )
        if not static_checks:
            if isinstance(fidelity, dict):
                collider_check = {
                    "capsule_id": plan.capsule_id,
                    "check_id": "collider_fidelity",
                    "status": str(fidelity.get("status") or "FAIL"),
                    "passed": fidelity.get("passed") is True,
                }
                for key in ("report", "report_sha256", "failure_reason"):
                    if fidelity.get(key) not in {None, ""}:
                        collider_check[key] = fidelity[key]
            else:
                collider_check = {
                    "capsule_id": plan.capsule_id,
                    "check_id": "collider_fidelity",
                    "status": "FAIL",
                    "passed": False,
                    "failure_reason": (
                        "reviewer_mechanism_adapter_collider_evidence_missing"
                        if reviewer_plugin
                        else "collider_fidelity_evidence_missing"
                    ),
                    **({"plugin_id": plugin.plugin_id} if reviewer_plugin else {}),
                }
            static_checks.append(collider_check)
        static_checks.append(
            {
                "capsule_id": plan.capsule_id,
                "check_id": "initial_functional_geometry",
                "status": str(initial_geometry.get("status") or "FAIL"),
                "passed": initial_geometry.get("passed") is True,
                "report": initial_geometry_summary["report"],
                "report_sha256": initial_geometry_summary[
                    "report_sha256"
                ],
                "failure_reasons": list(
                    initial_geometry.get("failure_reasons") or ()
                ),
                "warning_count": int(
                    initial_geometry.get("warning_count") or 0
                ),
            }
        )
        static_passed = all(
            item.get("passed") is True for item in static_checks
        )
        aggregate_static_preflight = {
            "schema_version": "mechanism_static_preflight_v1",
            "status": "PASS" if static_passed else "FAIL",
            "passed": static_passed,
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "all_independent_checks_executed": True,
            "physical_simulation_started": False,
            "checks": static_checks,
            "failure_reasons": [
                f"{item.get('check_id') or 'unknown'}_failed"
                for item in static_checks
                if item.get("passed") is not True
            ],
        }
        aggregate_static_preflight_path = (
            destination
            / "results"
            / "mechanism_static_preflight.json"
        )
        write_json_atomic(
            aggregate_static_preflight_path,
            aggregate_static_preflight,
        )
        if not static_passed:
            raise _static_preflight_failure(
                static_checks,
                capsule_id=plan.capsule_id,
                collision_geometry=(
                    collision_geometry
                    if isinstance(collision_geometry, dict)
                    else None
                ),
                plugin_owner=plugin_owner,
                plugin_id=plugin.plugin_id,
            )
        if isinstance(collision_geometry, dict):
            # Family collision materialization performs an early, conservative
            # home-pose audit before analytical interfaces are geometry-bound.
            # Physical workers must consume the final aggregate report, which
            # includes the STEP-bound interface-aware audit and initial
            # functional geometry gate, rather than that provisional report.
            adapter = _bind_final_static_preflight(
                adapter,
                workspace_root=workspace,
                report_path=aggregate_static_preflight_path,
                report=aggregate_static_preflight,
                package_binding_sha256=plan.package_binding_sha256,
            )
            plan = replace(plan, adapter_spec=adapter_to_dict(adapter))
            write_json_atomic(plan_path, plan.to_dict())

        prepared_dispatches.append(
            (
                plan,
                adapter,
                plugin,
                merge_mechanism_and_task_load_cases(
                    _mechanism_load_cases(
                        plugin,
                        plan,
                        adapter,
                        plugin_registration=plugin_registration,
                    ),
                    plan,
                ),
                plan_path,
                plan_relative,
                initial_geometry_path,
                initial_geometry_summary["report"],
                initial_geometry_summary["report_sha256"],
                _reviewer_mechanism_job_input(
                    plan,
                    plugin,
                    workspace=workspace,
                ),
            )
        )

    resolved_scope = resolve_validation_scope(
        tuple(item[0] for item in prepared_dispatches),
        {
            item[0].capsule_id: item[3]
            for item in prepared_dispatches
        },
        mode=resolved_validation_mode,
    )
    validation_scope_path = destination / "validation_scope_manifest.json"
    write_json_atomic(validation_scope_path, resolved_scope.document)
    for (
        plan,
        adapter,
        plugin,
        load_cases,
        plan_path,
        plan_relative,
        initial_geometry_path,
        initial_geometry_relative,
        initial_geometry_sha256,
        reviewer_mechanism_input,
    ) in prepared_dispatches:
        initial_geometry_identity = (
            initial_functional_geometry_identity_sha256(
                read_json(initial_geometry_path)
            )
        )
        cases_by_id = {
            load_case.load_case_id: load_case
            for load_case in load_cases
        }
        for load_case_id in resolved_scope.selected_case_ids(
            plan.capsule_id
        ):
            load_case = cases_by_id[load_case_id]
            if (
                load_case.program not in adapter.supported_programs
                and load_case.program not in claimed_reviewer_task_programs(plan)
            ):
                raise MechanismValidationError(
                    "ADAPTER_PROGRAM_UNSUPPORTED",
                    f"adapter {adapter.adapter_id!r} cannot run "
                    f"{load_case.program!r}",
                )
            reviewer_task_binding = _reviewer_task_request_binding(
                plan,
                load_case,
                workspace=workspace,
            )
            load_path = (
                destination
                / "loads"
                / plan.capsule_id
                / f"{load_case.load_case_id}.json"
            )
            write_json_atomic(load_path, load_case.to_dict())
            load_relative = relative_to_root(workspace, load_path)

            branches = resolved_scope.required_branches(
                plan.capsule_id,
                load_case.load_case_id,
            )
            include_ideal_dynamics = "ideal_dynamics" in branches
            physical_timeout, physical_estimate = _job_runtime_budget(
                adapter,
                branch="physical",
                estimated_steps=load_case.estimated_steps,
            )
            if include_ideal_dynamics:
                ideal_timeout, ideal_estimate = _job_runtime_budget(
                    adapter,
                    branch="ideal_dynamics",
                    estimated_steps=load_case.estimated_steps,
                )
                ideal_dynamics_task_count += 1
            else:
                ideal_timeout = 0.0
                ideal_estimate = 0.0

            request_paths: dict[str, Path] = {}
            request_relatives: dict[str, str] = {}
            for branch in (
                ("physical", "ideal_dynamics")
                if include_ideal_dynamics
                else ("physical",)
            ):
                request_path = (
                    destination
                    / "requests"
                    / plan.capsule_id
                    / load_case.load_case_id
                    / f"{branch}.json"
                )
                trace_path = (
                    destination
                    / "traces"
                    / plan.capsule_id
                    / load_case.load_case_id
                    / f"{branch}.trace.json"
                )
                request_relative = relative_to_root(workspace, request_path)
                trace_relative = relative_to_root(workspace, trace_path)
                request_payload = {
                    "schema_version": "0.1",
                    "branch": branch,
                    "adapter_id": plan.adapter_id,
                    "mechanism_plugin_id": plugin.plugin_id,
                    "package_binding_sha256": plan.package_binding_sha256,
                    "plan_file": plan_relative,
                    "plan_sha256": canonical_sha256(plan.to_dict()),
                    "load_case_file": load_relative,
                    "load_case_sha256": canonical_sha256(
                        load_case.to_dict()
                    ),
                    "initial_functional_geometry_file": (
                        initial_geometry_relative
                    ),
                    "initial_functional_geometry_sha256": (
                        initial_geometry_sha256
                    ),
                    "output_file": trace_relative,
                }
                if reviewer_task_binding is not None:
                    request_payload["reviewer_task_adapter"] = (
                        reviewer_task_binding
                    )
                if branch == "ideal_dynamics":
                    request_payload["reference_trace_file"] = relative_to_root(
                        workspace,
                        trace_path.with_name("physical.trace.json"),
                    )
                write_json_atomic(request_path, request_payload)
                request_paths[branch] = request_path
                request_relatives[branch] = request_relative
                request_files.append(request_relative)
                expected_trace_files.append(trace_relative)
            expected_trace_files.append(
                relative_to_root(
                    workspace,
                    destination
                    / "traces"
                    / plan.capsule_id
                    / load_case.load_case_id
                    / "ideal_reference.trace.json",
                )
            )

            job_id = _job_id(
                plan.capsule_id,
                load_case.load_case_id,
                "load-pipeline",
            )
            if job_id in seen_job_ids:
                raise MechanismValidationError(
                    "DISPATCH_JOB_ID_COLLISION",
                    f"duplicate dispatch job ID: {job_id}",
                )
            seen_job_ids.add(job_id)
            command_args = [
                "-m",
                (
                    "simulation_ready_cad_workflow."
                    "validation_pipeline.mechanism.lane_worker"
                ),
                "--request",
                request_relatives["physical"],
            ]
            if include_ideal_dynamics:
                command_args.extend(
                    (
                        "--ideal-dynamics-request",
                        request_relatives["ideal_dynamics"],
                    )
                )
            lane_timeout = (
                physical_timeout
                + ideal_timeout
                + _NONPHYSICAL_TASK_TIMEOUT_SECONDS
            )
            total_lane_count += 1
            current_plan_sha256 = canonical_sha256(plan.to_dict())
            current_load_case_sha256 = canonical_sha256(load_case.to_dict())
            task_fingerprint_binding = (
                {
                    "plugin_id": reviewer_task_binding["plugin_id"],
                    "sha256": reviewer_task_binding["sha256"],
                }
                if reviewer_task_binding is not None
                else None
            )
            fingerprint = lane_fingerprint(
                plan_document=plan.to_dict(),
                load_case_document=load_case.to_dict(),
                validator_source_sha256=validator_source_sha256,
                plugin_id=plugin.plugin_id,
                branches=tuple(branches),
                task_plugin_binding=task_fingerprint_binding,
                extra_dropped_keys=plugin.lane_fingerprint_dropped_keys,
            )
            reuse_entry = {
                "job_id": job_id,
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
                "branches": sorted(branches),
                "fingerprint": fingerprint,
                "action": "EXECUTED",
                "package_binding_sha256": plan.package_binding_sha256,
            }
            verified_source = find_lane_sign_off_source(
                workspace_root=workspace,
                output_dir=destination,
                capsule_id=plan.capsule_id,
                load_case_id=load_case.load_case_id,
                fingerprint=fingerprint,
                plugin_id=plugin.plugin_id,
                branches=tuple(branches),
                validator_source_sha256=validator_source_sha256,
                include_ideal_dynamics=include_ideal_dynamics,
                task_plugin_binding=task_fingerprint_binding,
                extra_dropped_keys=plugin.lane_fingerprint_dropped_keys,
            )
            promoted = False
            if verified_source is None and promotion_candidates:
                verified_source = _verified_lane_promotion_source(
                    workspace=workspace,
                    candidates=promotion_candidates,
                    job_id=job_id,
                    capsule_id=plan.capsule_id,
                    load_case_id=load_case.load_case_id,
                    fingerprint=fingerprint,
                    plugin_id=plugin.plugin_id,
                    branches=tuple(branches),
                    validator_source_sha256=validator_source_sha256,
                    initial_functional_geometry_identity=(
                        initial_geometry_identity
                    ),
                    include_ideal_dynamics=include_ideal_dynamics,
                    task_plugin_binding=task_fingerprint_binding,
                    extra_dropped_keys=(
                        plugin.lane_fingerprint_dropped_keys
                    ),
                )
                promoted = verified_source is not None
            if verified_source is not None:
                _materialize_lane_sign_off(
                    workspace=workspace,
                    destination=destination,
                    capsule_id=plan.capsule_id,
                    load_case_id=load_case.load_case_id,
                    verified_source=verified_source,
                )
                receipt = append_reuse_receipt(
                    destination,
                    capsule_id=plan.capsule_id,
                    load_case_id=load_case.load_case_id,
                    fingerprint=fingerprint,
                    plugin_id=plugin.plugin_id,
                    branches=tuple(branches),
                    validator_source_sha256=validator_source_sha256,
                    origin_run=verified_source["origin_run"],
                    origin_plan_sha256=verified_source["origin_plan_sha256"],
                    origin_load_case_sha256=(
                        verified_source["origin_load_case_sha256"]
                    ),
                    origin_package_binding_sha256=(
                        verified_source["origin_package_binding_sha256"]
                    ),
                    current_plan_sha256=current_plan_sha256,
                    current_load_case_sha256=current_load_case_sha256,
                    current_package_binding_sha256=(
                        plan.package_binding_sha256
                    ),
                    artifacts=verified_source["artifacts"],
                    origin_outer_audit=verified_source["origin_outer_audit"],
                    signed_by="harness.build_dispatch_bundle",
                    reason=(
                        "current-source diagnostic replay promotion whose "
                        "plan/load identities, artifact hashes, and outer "
                        "CAD/FCL audit were re-verified before adoption"
                        if promoted
                        else (
                            "lane content fingerprint matches an origin lane "
                            "whose plan/load identities, artifact hashes, and "
                            "outer CAD/FCL audit were re-verified in the origin "
                            "run before adoption"
                        )
                    ),
                )
                adopted_entry = {
                    **reuse_entry,
                    "action": "REUSED",
                    "origin_run": verified_source["origin_run"],
                    "receipt_sha256": receipt["entry_sha256"],
                }
                if promoted:
                    adopted_entry["origin_kind"] = "diagnostic_lane_replay"
                reuse_entries.append(adopted_entry)
                continue
            reuse_entries.append(reuse_entry)
            job = JobSpec(
                job_id=job_id,
                kind=JobKind.PYCHRONO,
                command=launcher.python_command(*command_args),
                working_directory=".",
                timeout_seconds=lane_timeout,
                expected_outputs=_load_pipeline_outputs(
                    workspace=workspace,
                    destination=destination,
                    capsule_id=plan.capsule_id,
                    load_case_id=load_case.load_case_id,
                    include_ideal_dynamics=include_ideal_dynamics,
                ),
                input_files=_content_bound_job_inputs(
                    plan,
                    workspace_root=workspace,
                    direct_files=(
                        *request_paths.values(),
                        plan_path,
                        load_path,
                        initial_geometry_path,
                        validation_scope_path,
                        *(
                            (reviewer_mechanism_input,)
                            if reviewer_mechanism_input
                            else ()
                        ),
                        *(
                            (workspace / reviewer_task_binding["path"],)
                            if reviewer_task_binding is not None
                            else ()
                        ),
                    ),
                ),
                environment={
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    # Jobs execute from workspace_root, which may be a clean
                    # benchmark scratch directory with no importable code.
                    # Bind the installed validator package root explicitly
                    # instead of inheriting a caller-specific shell PYTHONPATH.
                    "PYTHONPATH": str(validator_import_root()),
                    MECHANISM_SOURCE_SHA256_ENV: validator_source_sha256,
                },
                estimated_peak_memory_gib=(
                    _LOAD_LANE_ESTIMATED_PEAK_MEMORY_GIB
                ),
                max_process_tree_memory_gib=(
                    _LOAD_LANE_MAX_PROCESS_TREE_MEMORY_GIB
                ),
                estimated_wall_time_seconds=min(
                    lane_timeout,
                    physical_estimate + ideal_estimate + 10.0,
                ),
                estimated_simulation_steps=(
                    load_case.estimated_steps
                    * (2 if include_ideal_dynamics else 1)
                ),
                cpu_threads=2,
            )
            jobs.append(job)

    # The dispatch budget covers every selected lane whether it executes
    # or adopts a signed-off result: sign-off must not widen a dispatch.
    physical_job_count = total_lane_count
    nonphysical_job_count = ideal_dynamics_task_count
    physical_task_budget = _dispatch_physical_task_budget(plans)
    if (
        physical_job_count > physical_task_budget
        or nonphysical_job_count > _MAX_NONPHYSICAL_TASKS
    ):
        raise MechanismValidationError(
            "DISPATCH_TASK_BUDGET_EXCEEDED",
            "one mechanism dispatch may contain at most "
            f"{physical_task_budget} physical tasks and "
            f"{_MAX_NONPHYSICAL_TASKS} necessary non-physical tasks; "
            f"requested {physical_job_count} physical and "
            f"{nonphysical_job_count} non-physical",
        )

    manifest_path = destination / "queue_manifest.json"
    physical_manifest_path = destination / "physical_queue_manifest.json"
    run_name = f"mechanism-{plans[0].package_binding_sha256[:12]}"
    if jobs:
        manifest = QueueManifest(
            run_name=run_name,
            jobs=tuple(jobs),
            resource_policy=ResourcePolicy(
                preflight_memory_percent=86.0,
                min_available_memory_gib=1.5,
                min_free_disk_gib=10.0,
                critical_available_memory_gib=0.75,
                critical_free_disk_gib=5.0,
                preflight_sample_seconds=1.0,
                max_resource_wait_seconds=30.0,
                cooldown_seconds=2.0,
                pychrono_cpu_threads=2,
                generic_cpu_threads=2,
                terminate_grace_seconds=5.0,
                max_log_bytes_per_stream=8 * 1024 * 1024,
                lower_process_priority=True,
            ),
            queue_policy=QueuePolicy(
                max_wall_time_seconds=(
                    _MAX_PHYSICAL_TASKS
                    * (
                        _PHYSICAL_TASK_TIMEOUT_SECONDS
                        + _NONPHYSICAL_TASK_TIMEOUT_SECONDS
                    )
                    + _MAX_NONPHYSICAL_TASKS
                    * _NONPHYSICAL_TASK_TIMEOUT_SECONDS
                ),
                stop_on_failure=True,
                reuse_successful=True,
                max_timeout_resume_attempts=3,
                max_parallel_jobs=min(
                    _MAX_PHYSICAL_TASKS, max(1, len(jobs))
                ),
            ),
        )
        write_json_atomic(manifest_path, manifest.to_dict())
        write_json_atomic(
            physical_manifest_path,
            replace(
                manifest,
                run_name=f"{manifest.run_name}-load-pipelines",
            ).to_dict(),
        )
    else:
        # Every selected lane adopted a signed-off result. A queue manifest
        # contract requires at least one job, so record an explicit empty
        # marker instead; the engine skips the queue when job_count is 0.
        empty_marker = {
            "schema_version": "0.1",
            "run_name": run_name,
            "jobs": [],
            "all_lanes_signed_off": True,
            "note": (
                "no executable physical lane remains: every selected load "
                "case adopted a hash-chained sign-off receipt "
                "(results/reuse_receipts.json)"
            ),
        }
        write_json_atomic(manifest_path, empty_marker)
        write_json_atomic(physical_manifest_path, empty_marker)
    bundle = DispatchBundle(
        output_dir=relative_to_root(workspace, destination),
        queue_manifest=relative_to_root(workspace, manifest_path),
        physical_queue_manifest=relative_to_root(workspace, physical_manifest_path),
        ideal_dynamics_queue_manifest=None,
        validation_scope_manifest=relative_to_root(
            workspace,
            validation_scope_path,
        ),
        validation_scope=resolved_scope.document,
        plan_files=tuple(plan_files),
        request_files=tuple(request_files),
        expected_trace_files=tuple(expected_trace_files),
        collider_fidelity_reports=tuple(collider_fidelity_reports),
        initial_functional_geometry_reports=tuple(
            initial_functional_geometry_reports
        ),
        job_count=len(jobs),
        execution="per_load_pipeline_v2",
        max_parallel_jobs=min(_MAX_PHYSICAL_TASKS, max(1, len(jobs))),
    )
    mesh_cache_stats: dict[str, Any] = {}
    for prepared in prepared_dispatches:
        family_spec = (
            prepared[1].family_spec
            if isinstance(prepared[1].family_spec, dict)
            else {}
        )
        collision_geometry = family_spec.get("collision_geometry")
        if isinstance(collision_geometry, dict):
            stats = collision_geometry.get("mesh_cache")
            if isinstance(stats, dict):
                mesh_cache_stats[prepared[0].capsule_id] = stats
    write_reuse_manifest(
        destination / "results" / "reuse_manifest.json",
        entries=reuse_entries,
        mesh_cache_stats=mesh_cache_stats,
    )
    write_json_atomic(destination / "dispatch_summary.json", bundle.to_dict())
    return bundle
