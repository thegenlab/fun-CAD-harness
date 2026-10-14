"""Single harness-owned publisher for canonical dashboard evidence."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from ..mechanism.io import (
    canonical_sha256,
    read_json,
    relative_to_root,
    sha256_file,
)
from ..run_layout import (
    CANONICAL_TERMINAL_STATUSES,
    TASK_PLANNING_POLICY,
    parse_canonical_run,
)
from .legacy_planning import (
    TASK_PLANNING_RELATIVE,
    LegacyPlanningError,
    load_legacy_task_planning,
)
from .stage_manifest import (
    STAGE_MANIFEST_RELATIVE,
    WorkflowStageManifestError,
    assert_run_open_for_dashboard_publication,
    load_workflow_stage_manifest,
    materialize_workflow_stage_manifest,
    read_stable_json_object,
    stable_regular_file_record,
    write_json_exclusive,
)


PUBLICATION_SCHEMA = "simulation_ready_cad_dashboard_publication_v2"
PUBLISHER_ID = "simulation_ready_cad_dashboard_publisher_v2"
_MEDIA_SUFFIXES = frozenset({".png", ".mp4", ".glb"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PUBLICATION_FIELDS = frozenset(
    {
        "schema_version",
        "publisher_id",
        "certification_authority",
        "run_layout",
        "recorded_terminal_status",
        "stage_count",
        "stages",
        "stage_manifest_sha256",
        "workflow_stage_manifest",
        "media",
        "media_manifest_sha256",
        "visual_summary",
    }
)
_MANIFEST_BINDING_FIELDS = frozenset(
    {
        "path",
        "sha256",
        "size_bytes",
        "canonical_payload_sha256",
    }
)


class DashboardPublicationError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        {
            path.resolve()
            for path in paths
            if path.is_file() and not path.is_symlink()
        },
        key=lambda item: item.as_posix(),
    )


def _stage(
    *,
    stage_id: str,
    kind: str,
    label: str,
    order: int,
    evidence: list[Path],
    workspace: Path,
    required: bool = True,
    scope: dict[str, Any] | None = None,
    status: str = "EVIDENCE_COMPLETE",
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "kind": kind,
        "label": label,
        "order": order,
        "required": required,
        "depends_on": [],
        "scope": scope or {},
        "status": status,
        "evidence": [
            {
                "path": relative_to_root(workspace, path),
                "sha256": sha256_file(path),
            }
            for path in evidence
        ],
    }


def _walk_media_references(
    value: Any,
    *,
    role: str = "",
) -> list[tuple[str, str, str | None]]:
    """Return ``(role, path, declared_sha256)`` media leaves."""

    found: list[tuple[str, str, str | None]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_role = f"{role}.{key}" if role else str(key)
            if (
                isinstance(child, str)
                and Path(child).suffix.lower() in _MEDIA_SUFFIXES
            ):
                declared = value.get(f"{key}_sha256")
                if declared is None and key in {"path", "file", "scene", "video"}:
                    declared = value.get("sha256")
                found.append(
                    (
                        child_role,
                        child,
                        str(declared) if isinstance(declared, str) else None,
                    )
                )
            else:
                found.extend(
                    _walk_media_references(child, role=child_role)
                )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(
                _walk_media_references(
                    child,
                    role=f"{role}[{index}]",
                )
            )
    return found


def _media_manifest(
    *,
    workspace: Path,
    run: Path,
    visual_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    by_hash: dict[str, dict[str, Any]] = {}
    if visual_summary is not None:
        for role, reference, declared_sha256 in _walk_media_references(
            visual_summary
        ):
            path = (workspace / reference).resolve()
            try:
                path.relative_to(run)
            except ValueError as exc:
                raise DashboardPublicationError(
                    "DASHBOARD_MEDIA_OUTSIDE_RUN",
                    f"visual media is outside the canonical run: {reference}",
                ) from exc
            if not path.is_file() or path.is_symlink():
                raise DashboardPublicationError(
                    "DASHBOARD_MEDIA_MISSING",
                    f"visual media is missing: {reference}",
                )
            try:
                record, _raw = stable_regular_file_record(
                    path,
                    capture_bytes=False,
                    code="DASHBOARD_MEDIA_UNSTABLE",
                )
            except WorkflowStageManifestError as exc:
                raise DashboardPublicationError(
                    exc.code,
                    exc.message,
                ) from exc
            digest = str(record["sha256"])
            if declared_sha256 is not None and declared_sha256 != digest:
                raise DashboardPublicationError(
                    "DASHBOARD_MEDIA_HASH_MISMATCH",
                    f"declared visual hash does not match: {reference}",
                )
            entry = by_hash.setdefault(
                digest,
                {
                    "asset_id": f"sha256:{digest}",
                    "sha256": digest,
                    "media_type": path.suffix.lower().lstrip("."),
                    "primary_path": relative_to_root(workspace, path),
                    "alias_paths": [],
                    "roles": [],
                },
            )
            normalized = relative_to_root(workspace, path)
            if normalized != entry["primary_path"] and normalized not in entry[
                "alias_paths"
            ]:
                entry["alias_paths"].append(normalized)
            if role not in entry["roles"]:
                entry["roles"].append(role)
    assets = sorted(by_hash.values(), key=lambda item: item["asset_id"])
    return {
        "schema_version": "simulation_ready_cad_media_manifest_v1",
        "unique_asset_count": len(assets),
        "role_count": sum(len(item["roles"]) for item in assets),
        "deduplicated_role_count": sum(
            max(0, len(item["roles"]) - 1) for item in assets
        ),
        "assets": assets,
    }


def _dynamic_stages(
    *,
    workspace: Path,
    run: Path,
    terminal_status: dict[str, Any],
) -> list[dict[str, Any]]:
    stages: list[dict[str, Any]] = []

    def add(
        kind: str,
        label: str,
        evidence: Iterable[Path],
        *,
        stage_id: str | None = None,
        scope: dict[str, Any] | None = None,
        status: str = "EVIDENCE_COMPLETE",
    ) -> None:
        files = _paths(evidence)
        if not files:
            return
        stages.append(
            _stage(
                stage_id=stage_id or kind,
                kind=kind,
                label=label,
                order=len(stages) + 1,
                evidence=files,
                workspace=workspace,
                scope=scope,
                status=status,
            )
        )

    planning_path = run / TASK_PLANNING_RELATIVE
    run_manifest = _read_status(run / "run_manifest.json")
    planning_required = (
        run_manifest.get("task_planning_policy") == TASK_PLANNING_POLICY
    )
    if planning_required and not (
        planning_path.exists() or planning_path.is_symlink()
    ):
        raise DashboardPublicationError(
            "DASHBOARD_TASK_PLANNING_REQUIRED",
            "this revision requires task planning before terminal publication",
        )
    if planning_path.exists() or planning_path.is_symlink():
        try:
            planning, planning_evidence = load_legacy_task_planning(
                workspace_root=workspace,
                run_dir=run,
            )
        except LegacyPlanningError as exc:
            raise DashboardPublicationError(
                "DASHBOARD_TASK_PLANNING_INVALID",
                exc.message,
            ) from exc
        add(
            "task-planning",
            "接收任务与测试规划",
            planning_evidence,
            scope={
                "joint_plan_count": len(planning["joint_plan"]),
                "test_plan_count": len(planning["test_plan"]),
                "authored_test_task_count": planning[
                    "authored_test_task_count"
                ],
                "requirements_document_sha256": str(
                    planning["task_charter_binding"].get(
                        "requirements_document_sha256"
                    )
                    or ""
                ),
            },
        )

    design_files = _paths(
        [
            *run.glob("inputs/**/*.json"),
            *run.glob("inputs/**/*.step"),
            *run.glob("design_quality/**/*.json"),
        ]
    )
    add("design-package", "设计包", design_files)

    intake_files = _paths(
        [run / "intake_report.json", *run.glob("intake*.json")]
    )
    intake_status = next(
        (
            str(_read_status(path).get("status") or "EVIDENCE_COMPLETE")
            for path in intake_files
            if path.suffix.lower() == ".json"
        ),
        "EVIDENCE_COMPLETE",
    )
    add(
        "intake",
        "接入与任务语义",
        intake_files,
        status=intake_status,
    )

    capability_files = _paths(
        [*run.glob("capability_review/**/*"), *run.glob("results/capability*")]
    )
    add(
        "capability-review",
        "缺失能力审批",
        capability_files,
        status=(
            str(terminal_status.get("status"))
            if str(terminal_status.get("status") or "")
            in {
                "CAPABILITY_PROPOSAL_DRAFT_REQUIRED",
                "CAPABILITY_REVIEW_REQUIRED",
            }
            else "CAPABILITY_REVIEW_REQUIRED"
        ),
    )

    plan_scope_by_capsule: dict[str, dict[str, object]] = {}
    for plan in _paths((run / "plans").glob("*.json")):
        plan_value = _read_status(plan)
        capsule = str(plan_value.get("capsule_id") or plan.stem)
        plan_scope_by_capsule[capsule] = {
            "capsule_id": capsule,
            "task_ids": [
                str(item)
                for item in (plan_value.get("task_ids") or [])
                if str(item)
            ],
        }
        add(
            "mechanism-plan",
            f"机构计划 · {capsule}",
            [plan],
            stage_id=f"mechanism-plan:{capsule}",
            scope=plan_scope_by_capsule[capsule],
        )

    global_geometry_files = _paths(
        run.glob("results/*geometry*preflight*.json")
    )
    add(
        "geometry-preflight",
        "几何与碰撞预检",
        global_geometry_files,
    )
    adapters_root = run / "derived_adapters"
    if adapters_root.is_dir():
        for adapter in sorted(adapters_root.iterdir(), key=lambda item: item.name):
            if not adapter.is_dir() or adapter.is_symlink():
                continue
            geometry_files = _paths(
                [
                    *adapter.glob("**/fidelity*.json"),
                    *adapter.glob("**/rigid_attachment*.json"),
                    *adapter.glob("**/provenance.json"),
                ]
            )
            add(
                "geometry-preflight",
                f"几何与碰撞预检 · {adapter.name}",
                geometry_files,
                stage_id=f"geometry-preflight:{adapter.name}",
                scope={"capsule_id": adapter.name},
            )

    traces_by_capsule: dict[str, list[Path]] = {}
    for trace in _paths((run / "traces").glob("**/*.trace.json")):
        try:
            relative = trace.relative_to(run / "traces")
            capsule = relative.parts[0]
        except (ValueError, IndexError):
            capsule = "mechanism"
        traces_by_capsule.setdefault(capsule, []).append(trace)
    for capsule, traces in sorted(traces_by_capsule.items()):
        load_case_ids = sorted(
            {
                trace.relative_to(run / "traces" / capsule).parts[0]
                for trace in traces
            }
        )
        scope = dict(
            plan_scope_by_capsule.get(capsule) or {"capsule_id": capsule}
        )
        scope["load_case_ids"] = load_case_ids
        add(
            "branch-simulation",
            f"真实物理与参考 · {capsule}",
            traces,
            stage_id=f"branch-simulation:{capsule}",
            scope=scope,
        )

    for certificate in _paths(
        (run / "results").glob("*/certificate.json")
    ):
        capsule = certificate.parent.name
        comparison = _paths(
            (certificate.parent / "comparison").glob("*.json")
        )
        scope = dict(
            plan_scope_by_capsule.get(capsule) or {"capsule_id": capsule}
        )
        scope["load_case_ids"] = [path.stem for path in comparison]
        add(
            "certificate",
            f"机构认证 · {capsule}",
            [certificate, *comparison],
            stage_id=f"certificate:{capsule}",
            scope=scope,
            status=str(
                _read_status(certificate).get("overall_verdict")
                or _read_status(certificate).get("verdict")
                or "NOT_CERTIFIED"
            ),
        )

    for replacement in _paths(
        (run / "results").glob("*/replacement.json")
    ):
        capsule = replacement.parent.name
        add(
            "replacement",
            f"简化替代 · {capsule}",
            [replacement],
            stage_id=f"replacement:{capsule}",
            scope={"capsule_id": capsule},
            status=str(_read_status(replacement).get("status") or "INACTIVE"),
        )

    for result in _paths(
        (run / "results").glob("*/whole_machine/result.json")
    ):
        capsule = result.parent.parent.name
        add(
            "whole-machine",
            f"整机任务 · {capsule}",
            [result],
            stage_id=f"whole-machine:{capsule}",
            scope={"capsule_id": capsule},
            status=str(_read_status(result).get("status") or "NOT_CERTIFIED"),
        )

    if not stages:
        # A terminal infrastructure failure still gets one truthful stage.
        failure = _paths([run / "failure.json"])
        stages.append(
            _stage(
                stage_id="terminal",
                kind="terminal",
                label="运行终态",
                order=1,
                evidence=failure,
                workspace=workspace,
                status=str(terminal_status.get("status") or "FAILED"),
            )
        )
    for index, stage in enumerate(stages):
        stage["order"] = index + 1
        if index:
            stage["depends_on"] = [stages[index - 1]["id"]]
    return stages


def _read_status(path: Path) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, ValueError):
        return {}


def load_bound_dashboard_publication(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    terminal_status: dict[str, Any],
    require_inventory: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Load the exact publication bound by terminal status.

    This is the common read boundary for the catalog, detail model, and media
    server.  It does not grant certificate authority; historical publications
    are displayed read-only exactly as their terminal status recorded them.
    """

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    identity = parse_canonical_run(workspace, run)
    publication_ref = terminal_status.get("dashboard_publication")
    if not isinstance(publication_ref, dict):
        return None, "dashboard-publication-missing"
    reference = publication_ref.get("path")
    expected_sha256 = publication_ref.get("sha256")
    if (
        not isinstance(reference, str)
        or not reference
        or not isinstance(expected_sha256, str)
        or not _SHA256.fullmatch(expected_sha256)
    ):
        return None, "dashboard-publication-binding-invalid"
    publication_path = (workspace / reference).resolve()
    try:
        publication_path.relative_to(run)
    except ValueError:
        return None, "dashboard-publication-outside-run"
    if relative_to_root(workspace, publication_path) != reference:
        return None, "dashboard-publication-binding-invalid"
    try:
        publication, publication_record = read_stable_json_object(
            publication_path,
            code="DASHBOARD_PUBLICATION_JSON_INVALID",
        )
    except (OSError, ValueError):
        return None, "dashboard-publication-invalid"
    if publication_record["sha256"] != expected_sha256:
        return None, "dashboard-publication-binding-invalid"

    stages = publication.get("stages")
    media = publication.get("media")
    manifest_ref = publication.get("workflow_stage_manifest")
    if (
        set(publication) != _PUBLICATION_FIELDS
        or publication.get("schema_version") != PUBLICATION_SCHEMA
        or publication.get("publisher_id") != PUBLISHER_ID
        or publication.get("certification_authority") is not False
        or identity is None
        or publication.get("run_layout") != identity.to_dict()
        or publication.get("recorded_terminal_status")
        != str(terminal_status.get("status") or "NOT_AVAILABLE")
        or not isinstance(stages, list)
        or publication.get("stage_count") != len(stages)
        or publication.get("stage_manifest_sha256")
        != canonical_sha256(stages)
        or not isinstance(manifest_ref, dict)
        or set(manifest_ref) != _MANIFEST_BINDING_FIELDS
        or not isinstance(media, dict)
        or publication.get("media_manifest_sha256")
        != canonical_sha256(media)
    ):
        return None, "dashboard-publication-identity-invalid"
    try:
        manifest, verified_manifest_ref = load_workflow_stage_manifest(
            workspace_root=workspace,
            run_dir=run,
        )
    except (OSError, ValueError):
        return None, "dashboard-stage-manifest-invalid"
    if (
        manifest_ref != verified_manifest_ref
        or manifest_ref.get("path")
        != relative_to_root(workspace, run / STAGE_MANIFEST_RELATIVE)
        or manifest.get("stages") != stages
        or manifest.get("stage_count") != len(stages)
        or manifest.get("stages_canonical_sha256")
        != publication.get("stage_manifest_sha256")
    ):
        return None, "dashboard-stage-manifest-binding-invalid"

    stage_ids = [
        str(stage.get("id") or "")
        for stage in stages
        if isinstance(stage, dict)
    ]
    if (
        len(stage_ids) != len(stages)
        or not all(stage_ids)
        or len(set(stage_ids)) != len(stage_ids)
    ):
        return None, "dashboard-publication-stage-invalid"

    inventory_artifacts: dict[str, Any] | None = None
    if require_inventory:
        inventory_ref = terminal_status.get("artifact_inventory")
        inventory_sha256 = terminal_status.get(
            "artifact_inventory_sha256"
        )
        if (
            not isinstance(inventory_ref, str)
            or not inventory_ref
            or not isinstance(inventory_sha256, str)
            or not _SHA256.fullmatch(inventory_sha256)
        ):
            return None, "terminal-inventory-binding-missing"
        inventory_path = (workspace / inventory_ref).resolve()
        try:
            inventory_path.relative_to(run)
        except ValueError:
            return None, "terminal-inventory-outside-run"
        if relative_to_root(workspace, inventory_path) != inventory_ref:
            return None, "terminal-inventory-binding-invalid"
        try:
            inventory, inventory_record = read_stable_json_object(
                inventory_path,
                code="DASHBOARD_TERMINAL_INVENTORY_INVALID",
            )
        except (OSError, ValueError):
            return None, "terminal-inventory-invalid"
        if inventory_record["sha256"] != inventory_sha256:
            return None, "terminal-inventory-binding-invalid"
        inventory_artifacts = inventory.get("artifacts")
        required_surface_bindings = {
            reference: expected_sha256,
            str(manifest_ref["path"]): str(manifest_ref["sha256"]),
            **{
                str(item["path"]): str(item["sha256"])
                for item in manifest.get("declarations") or []
            },
        }
        if (
            inventory.get("schema_version")
            != "simulation_ready_cad_artifact_index_v1"
            or inventory.get("path_base") != "workspace_root"
            or not isinstance(inventory_artifacts, dict)
            or any(
                inventory_artifacts.get(path_value) != digest
                for path_value, digest in required_surface_bindings.items()
            )
        ):
            return None, "terminal-inventory-schema-invalid"

    earlier_stage_ids: set[str] = set()
    for stage_index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            return None, "dashboard-publication-stage-invalid"
        dependencies = stage.get("depends_on")
        if (
            stage.get("order") != stage_index
            or not isinstance(dependencies, list)
            or not all(isinstance(dependency, str) for dependency in dependencies)
            or len(set(dependencies)) != len(dependencies)
            or not all(
                isinstance(dependency, str)
                and dependency in earlier_stage_ids
                for dependency in dependencies
            )
        ):
            return None, "dashboard-publication-stage-invalid"
        references = stage.get("evidence")
        if not isinstance(references, list):
            return None, "dashboard-publication-stage-evidence-invalid"
        for reference_value in references:
            if (
                not isinstance(reference_value, dict)
                or set(reference_value) != {"path", "sha256"}
            ):
                return None, "dashboard-publication-stage-evidence-invalid"
            path_value = reference_value.get("path")
            digest = reference_value.get("sha256")
            if (
                not isinstance(path_value, str)
                or not path_value
                or not isinstance(digest, str)
                or not _SHA256.fullmatch(digest)
            ):
                return None, "dashboard-publication-stage-evidence-invalid"
            evidence_path = (workspace / path_value).resolve()
            try:
                evidence_path.relative_to(run)
            except ValueError:
                return None, "dashboard-publication-stage-evidence-outside-run"
            if relative_to_root(workspace, evidence_path) != path_value:
                return None, "dashboard-publication-stage-evidence-binding-invalid"
            try:
                evidence_record, _raw = stable_regular_file_record(
                    evidence_path,
                    capture_bytes=False,
                    code="DASHBOARD_STAGE_EVIDENCE_INVALID",
                )
            except (OSError, ValueError):
                return None, "dashboard-publication-stage-evidence-binding-invalid"
            if (
                evidence_record["sha256"] != digest
                or (
                    inventory_artifacts is not None
                    and inventory_artifacts.get(path_value) != digest
                )
            ):
                return None, "dashboard-publication-stage-evidence-binding-invalid"
        earlier_stage_ids.add(str(stage["id"]))

    assets = media.get("assets")
    if (
        not isinstance(assets, list)
        or media.get("unique_asset_count") != len(assets)
    ):
        return None, "dashboard-publication-media-invalid"
    seen_hashes: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict):
            return None, "dashboard-publication-media-invalid"
        digest = asset.get("sha256")
        alias_paths = asset.get("alias_paths")
        roles = asset.get("roles")
        if not isinstance(alias_paths, list) or not isinstance(roles, list):
            return None, "dashboard-publication-media-invalid"
        paths = [asset.get("primary_path"), *alias_paths]
        if (
            not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or digest in seen_hashes
            or not paths
            or not all(isinstance(path, str) and path for path in paths)
            or not all(isinstance(role, str) and role for role in roles)
            or len(set(paths)) != len(paths)
            or len(set(roles)) != len(roles)
        ):
            return None, "dashboard-publication-media-invalid"
        seen_hashes.add(digest)
        for path_value in paths:
            media_path = (workspace / str(path_value)).resolve()
            try:
                media_path.relative_to(run)
            except ValueError:
                return None, "dashboard-publication-media-outside-run"
            if (
                relative_to_root(workspace, media_path) != str(path_value)
                or media_path.suffix.lower() not in _MEDIA_SUFFIXES
            ):
                return None, "dashboard-publication-media-binding-invalid"
            try:
                media_record, _raw = stable_regular_file_record(
                    media_path,
                    capture_bytes=False,
                    code="DASHBOARD_MEDIA_INVALID",
                )
            except (OSError, ValueError):
                return None, "dashboard-publication-media-binding-invalid"
            if (
                media_record["sha256"] != digest
                or (
                    inventory_artifacts is not None
                    and inventory_artifacts.get(str(path_value)) != digest
                )
            ):
                return None, "dashboard-publication-media-binding-invalid"
    role_count = sum(
        len(asset.get("roles") or [])
        for asset in assets
        if isinstance(asset, dict)
    )
    if (
        media.get("role_count") != role_count
        or media.get("deduplicated_role_count")
        != sum(
            max(0, len(asset.get("roles") or []) - 1)
            for asset in assets
            if isinstance(asset, dict)
        )
    ):
        return None, "dashboard-publication-media-count-invalid"
    return publication, "ready"


def published_media_allowlist(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    terminal_status: dict[str, Any],
) -> dict[str, str]:
    """Return publication+inventory-bound media paths keyed by workspace path."""

    publication, _reason = load_bound_dashboard_publication(
        workspace_root=workspace_root,
        run_dir=run_dir,
        terminal_status=terminal_status,
        require_inventory=True,
    )
    if publication is None:
        return {}
    allowed: dict[str, str] = {}
    for asset in publication["media"]["assets"]:
        digest = str(asset["sha256"])
        for path_value in [
            asset["primary_path"],
            *(asset.get("alias_paths") or []),
        ]:
            allowed[str(path_value)] = digest
    return allowed


def finalize_dashboard_publication(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    terminal_status: dict[str, Any],
) -> dict[str, Any]:
    recorded_terminal_status = str(
        terminal_status.get("status") or ""
    ).strip().upper()
    if recorded_terminal_status not in CANONICAL_TERMINAL_STATUSES:
        raise DashboardPublicationError(
            "DASHBOARD_PUBLICATION_TERMINAL_STATUS_INVALID",
            "formal publication accepts only a closed Harness-owned typed "
            "terminal status; PASS, FAIL, PASSED, and RESOLVED are forbidden",
        )
    try:
        workspace, run, identity = (
            assert_run_open_for_dashboard_publication(
                workspace_root,
                run_dir,
            )
        )
    except WorkflowStageManifestError as exc:
        raise DashboardPublicationError(exc.code, exc.message) from exc
    visual_path = run / "visual" / "dashboard_visuals.summary.json"
    visual_summary: dict[str, Any] | None = None
    visual_record: dict[str, Any] | None = None
    if visual_path.exists() or visual_path.is_symlink():
        try:
            visual_summary, visual_record = read_stable_json_object(
                visual_path,
                code="DASHBOARD_VISUAL_SUMMARY_INVALID",
            )
        except WorkflowStageManifestError as exc:
            raise DashboardPublicationError(
                exc.code,
                exc.message,
            ) from exc
    if visual_summary is not None and visual_summary.get("status") != "READY":
        raise DashboardPublicationError(
            "DASHBOARD_VISUALS_NOT_READY",
            "visual summary is not READY",
        )
    stages = _dynamic_stages(
        workspace=workspace,
        run=run,
        terminal_status=terminal_status,
    )
    try:
        stage_manifest, stage_manifest_binding = (
            materialize_workflow_stage_manifest(
                workspace_root=workspace,
                run_dir=run,
                automatic_stages=stages,
            )
        )
    except WorkflowStageManifestError as exc:
        raise DashboardPublicationError(exc.code, exc.message) from exc
    stages = stage_manifest["stages"]
    media = _media_manifest(
        workspace=workspace,
        run=run,
        visual_summary=visual_summary,
    )
    publication = {
        "schema_version": PUBLICATION_SCHEMA,
        "publisher_id": PUBLISHER_ID,
        "certification_authority": False,
        "run_layout": identity.to_dict(),
        "recorded_terminal_status": recorded_terminal_status,
        "stage_count": len(stages),
        "stages": stages,
        "stage_manifest_sha256": stage_manifest[
            "stages_canonical_sha256"
        ],
        "workflow_stage_manifest": stage_manifest_binding,
        "media": media,
        "media_manifest_sha256": canonical_sha256(media),
        "visual_summary": (
            {
                "path": relative_to_root(workspace, visual_path),
                "sha256": visual_record["sha256"],
            }
            if visual_summary is not None and visual_record is not None
            else None
        ),
    }
    destination = run / "dashboard" / "publication.json"
    try:
        publication_record = write_json_exclusive(
            destination,
            publication,
            code="DASHBOARD_PUBLICATION_WRITE_FAILED",
        )
    except FileExistsError:
        try:
            existing, publication_record = read_stable_json_object(
                destination,
                code="DASHBOARD_PUBLICATION_JSON_INVALID",
            )
        except WorkflowStageManifestError as exc:
            raise DashboardPublicationError(
                exc.code,
                exc.message,
            ) from exc
        if existing != publication:
            raise DashboardPublicationError(
                "DASHBOARD_PUBLICATION_REBUILD_MISMATCH",
                "existing publication does not equal freshly rebuilt truth",
            )
    except WorkflowStageManifestError as exc:
        raise DashboardPublicationError(exc.code, exc.message) from exc
    return {
        "path": relative_to_root(workspace, destination),
        "sha256": publication_record["sha256"],
        "publisher_id": PUBLISHER_ID,
        "stage_count": len(stages),
        "unique_media_count": publication["media"]["unique_asset_count"],
        "media_role_count": publication["media"]["role_count"],
        "workflow_stage_manifest": stage_manifest_binding,
        "stage_manifest": [
            {
                "id": stage["id"],
                "kind": stage["kind"],
                "required": stage["required"],
                "scope": stage["scope"],
            }
            for stage in stages
        ],
    }
