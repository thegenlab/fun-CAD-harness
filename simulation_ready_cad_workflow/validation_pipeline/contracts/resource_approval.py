"""Integrity and approval bindings for AI-authored planning resources."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from ..mechanism.io import (
    canonical_sha256,
    read_json,
    relative_to_root,
    sha256_file,
)
from ..planning_templates.loader import PlanningTemplateError
from ..planning_templates.project_proposals import (
    validate_project_planning_template_bundle,
)
from .errors import ReactLoopError
from .exclusive_io import write_new_json as _write_new_json
from .payloads import strict_json_copy


RESOURCE_REQUEST_SCHEMA = "simulation_ready_cad_resource_approval_request_v1"
RESOURCE_DECISION_SCHEMA = "simulation_ready_cad_resource_approval_decision_v1"
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_MAX_RESOURCE_BYTES = 16 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_dependencies(value: Any) -> list[dict[str, Any]]:
    """Normalize the dependency declaration frozen with a resource bundle."""

    if not isinstance(value, list) or len(value) > 128:
        raise ReactLoopError(
            "REACT_RESOURCE_DEPENDENCIES_INVALID",
            "resource_dependencies must be a bounded list",
        )
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, Mapping) or set(item) != {
            "name",
            "version",
            "sha256",
        }:
            raise ReactLoopError(
                "REACT_RESOURCE_DEPENDENCIES_INVALID",
                f"resource_dependencies[{index}] fields are invalid",
            )
        name = item.get("name")
        version = item.get("version")
        digest = item.get("sha256")
        normalized_name = name.strip() if isinstance(name, str) else ""
        normalized_version = version.strip() if isinstance(version, str) else ""
        if (
            not isinstance(name, str)
            or not normalized_name
            or len(name) > 256
            or normalized_name in seen
            or not isinstance(version, str)
            or not normalized_version
            or len(version) > 128
            or (
                digest is not None
                and (not isinstance(digest, str) or _HEX_64.fullmatch(digest) is None)
            )
        ):
            raise ReactLoopError(
                "REACT_RESOURCE_DEPENDENCIES_INVALID",
                f"resource_dependencies[{index}] values are invalid",
            )
        seen.add(normalized_name)
        output.append(
            {
                "name": normalized_name,
                "version": normalized_version,
                "sha256": digest,
            }
        )
    output.sort(key=lambda item: item["name"])
    return output


def _verify_bound_resource_record(
    *,
    workspace: Path,
    record: Mapping[str, Any],
    kind: str,
    task_id: str | None = None,
) -> dict[str, Any]:
    path_text = record.get("path")
    digest = record.get("sha256")
    size_bytes = record.get("size_bytes")
    if (
        kind not in {"authored_test", "mechanism_template"}
        or not isinstance(path_text, str)
        or not isinstance(digest, str)
        or _HEX_64.fullmatch(digest) is None
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
        or size_bytes > _MAX_RESOURCE_BYTES
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_INTEGRITY_INVALID", "resource binding is malformed"
        )
    relative = PurePosixPath(path_text)
    if (
        relative.is_absolute()
        or "\\" in path_text
        or "\x00" in path_text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_INTEGRITY_INVALID", "resource escaped the workspace"
        )
    source = workspace.joinpath(*relative.parts)
    current = workspace
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ReactLoopError(
                "REACT_RESOURCE_INTEGRITY_INVALID",
                f"resource path contains a symlink: {path_text}",
            )
    if (
        not source.is_file()
        or source.stat().st_size != size_bytes
        or sha256_file(source) != digest
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_INTEGRITY_CHANGED",
            f"resource changed after submission: {path_text}",
        )
    if kind == "authored_test" and (
        not isinstance(task_id, str)
        or not task_id.strip()
        or len(task_id) > 128
        or source.suffix != ".py"
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_INTEGRITY_INVALID",
            "authored test identity or file type is invalid",
        )
    logical_root = (
        "authored_tests" if kind == "authored_test" else "mechanism_templates"
    )
    logical_path = f"{logical_root}/{source.name}"
    if source.suffix == ".py":
        try:
            ast.parse(source.read_text(encoding="utf-8"), filename=logical_path)
        except (OSError, UnicodeError, SyntaxError) as exc:
            raise ReactLoopError(
                "REACT_RESOURCE_PYTHON_INVALID",
                f"Python resource is not syntactically valid: {logical_path}",
            ) from exc
    elif source.suffix == ".json":
        try:
            read_json(source, max_bytes=_MAX_RESOURCE_BYTES)
        except (OSError, ValueError) as exc:
            raise ReactLoopError(
                "REACT_RESOURCE_JSON_INVALID",
                f"JSON resource is invalid: {logical_path}",
            ) from exc
    else:
        raise ReactLoopError(
            "REACT_RESOURCE_TYPE_INVALID",
            f"resource has an unsupported extension: {logical_path}",
        )
    value: dict[str, Any] = {
        "kind": kind,
        "logical_path": logical_path,
        "path": relative_to_root(workspace, source),
        "sha256": digest,
        "size_bytes": size_bytes,
    }
    if kind == "authored_test":
        value["task_id"] = task_id.strip()
    return value


def _admit_planning_resource_record(
    *,
    workspace: Path,
    record: Mapping[str, Any],
    kind: str,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Apply today's planning-resource policy after byte verification."""

    checked = _verify_bound_resource_record(
        workspace=workspace,
        record=record,
        kind=kind,
        task_id=task_id,
    )
    if kind == "mechanism_template" and Path(checked["path"]).suffix == ".py":
        raise ReactLoopError(
            "REACT_RESOURCE_NOT_PLANNING_TEMPLATE",
            "mechanism_templates must be joint or task-suite JSON; "
            "Python MechanismAdapter files are not a reviewer surface",
        )
    return checked


def create_resource_request(
    *,
    workspace_root: str | Path,
    loop_dir: str | Path,
    project_id: str,
    revision: str,
    requirements_binding: Mapping[str, Any],
    joint_plan: Sequence[Mapping[str, Any]],
    test_plan: Sequence[Mapping[str, Any]],
    authored_tests: Sequence[Mapping[str, Any]],
    mechanism_templates: Sequence[Mapping[str, Any]],
    dependencies: Sequence[Mapping[str, Any]],
    turn: int,
) -> tuple[dict[str, Any], Path]:
    """Check submitted files and persist the exact approval request."""

    workspace = Path(workspace_root).resolve()
    resources: list[dict[str, Any]] = []
    for item in authored_tests:
        resources.append(
            _admit_planning_resource_record(
                workspace=workspace,
                record=item,
                kind="authored_test",
                task_id=str(item.get("task_id") or ""),
            )
        )
    for item in mechanism_templates:
        resources.append(
            _admit_planning_resource_record(
                workspace=workspace,
                record=item,
                kind="mechanism_template",
            )
        )
    template_resources = [
        item for item in resources if item["kind"] == "mechanism_template"
    ]
    if template_resources:
        try:
            validate_project_planning_template_bundle(
                template_resources, workspace=workspace
            )
        except PlanningTemplateError as exc:
            raise ReactLoopError(
                "REACT_RESOURCE_TEMPLATE_FORMAT_INVALID",
                f"{exc.code}: {exc.message}",
            ) from exc
    resources.sort(key=lambda item: (item["kind"], item["logical_path"]))
    logical_resources = [
        {key: value for key, value in item.items() if key != "path"}
        for item in resources
    ]
    normalized_dependencies = normalize_dependencies(list(dependencies))
    bundle_core = {
        "project_id": project_id,
        "requirements_binding": strict_json_copy(
            requirements_binding, label="resource requirements binding"
        ),
        "joint_plan": [dict(item) for item in joint_plan],
        "test_plan": [dict(item) for item in test_plan],
        "resources": logical_resources,
        "dependencies": normalized_dependencies,
    }
    bundle_sha256 = canonical_sha256(bundle_core)
    request_id = f"resource-t{turn:04d}-{bundle_sha256[:12]}"
    core = {
        "schema_version": RESOURCE_REQUEST_SCHEMA,
        "request_id": request_id,
        "project_id": project_id,
        "revision": revision,
        "status": "INTEGRITY_VALIDATED",
        "requirements_binding": bundle_core["requirements_binding"],
        "joint_plan": bundle_core["joint_plan"],
        "test_plan": bundle_core["test_plan"],
        "resources": resources,
        "dependencies": normalized_dependencies,
        "bundle_sha256": bundle_sha256,
        "integrity": {
            "status": "PASS",
            "checks": [
                "bounded_regular_files",
                "safe_paths",
                "json_or_python_syntax",
                "complete_sha256_manifest",
                "frozen_dependencies",
            ],
        },
        "created_at": _utc_now(),
    }
    request = {**core, "request_sha256": canonical_sha256(core)}
    path = (
        Path(loop_dir).resolve()
        / "resource_approvals"
        / "requests"
        / f"{request_id}.json"
    )
    try:
        _write_new_json(path, request)
    except FileExistsError:
        request = _reuse_resource_record(
            workspace_root=workspace,
            path=path,
            candidate=request,
            volatile_field="created_at",
            digest_field="request_sha256",
            validator=validate_resource_request,
            conflict_code="REACT_RESOURCE_REQUEST_CONFLICT",
        )
    return request, path


def validate_resource_request(
    workspace_root: str | Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute every frozen binding before review or execution."""

    workspace = Path(workspace_root).resolve()
    request = strict_json_copy(value, label="resource approval request")
    required = {
        "schema_version",
        "request_id",
        "project_id",
        "revision",
        "status",
        "requirements_binding",
        "joint_plan",
        "test_plan",
        "resources",
        "dependencies",
        "bundle_sha256",
        "integrity",
        "created_at",
        "request_sha256",
    }
    integrity = request.get("integrity")
    if (
        set(request) != required
        or request.get("schema_version") != RESOURCE_REQUEST_SCHEMA
        or request.get("status") != "INTEGRITY_VALIDATED"
        or not isinstance(request.get("project_id"), str)
        or not request["project_id"].strip()
        or not isinstance(request.get("request_id"), str)
        or re.fullmatch(r"resource-t[0-9]{4,}-[0-9a-f]{12}", request["request_id"])
        is None
        or not isinstance(request.get("revision"), str)
        or re.fullmatch(r"r[0-9]{2,}", request["revision"]) is None
        or not isinstance(request.get("created_at"), str)
        or not isinstance(request.get("requirements_binding"), Mapping)
        or set(request["requirements_binding"]) != {"path", "sha256", "size_bytes"}
        or not isinstance(request["requirements_binding"].get("path"), str)
        or _HEX_64.fullmatch(
            str(request["requirements_binding"].get("sha256") or "")
        )
        is None
        or not isinstance(request["requirements_binding"].get("size_bytes"), int)
        or isinstance(request["requirements_binding"].get("size_bytes"), bool)
        or request["requirements_binding"]["size_bytes"] < 0
        or _HEX_64.fullmatch(str(request.get("bundle_sha256") or "")) is None
        or not isinstance(request.get("joint_plan"), list)
        or not isinstance(request.get("test_plan"), list)
        or not request["test_plan"]
        or not isinstance(integrity, Mapping)
        or integrity.get("status") != "PASS"
        or integrity.get("checks")
        != [
            "bounded_regular_files",
            "safe_paths",
            "json_or_python_syntax",
            "complete_sha256_manifest",
            "frozen_dependencies",
        ]
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_REQUEST_INVALID", "resource request fields are invalid"
        )
    unsigned = {key: item for key, item in request.items() if key != "request_sha256"}
    if request.get("request_sha256") != canonical_sha256(unsigned):
        raise ReactLoopError(
            "REACT_RESOURCE_REQUEST_CHANGED", "resource request hash changed"
        )
    resources = request.get("resources")
    if not isinstance(resources, list) or not resources or len(resources) > 256:
        raise ReactLoopError(
            "REACT_RESOURCE_REQUEST_INVALID", "resource request has no bounded resources"
        )
    logical: list[dict[str, Any]] = []
    resource_keys: set[tuple[str, str]] = set()
    for item in resources:
        if not isinstance(item, Mapping):
            raise ReactLoopError(
                "REACT_RESOURCE_REQUEST_INVALID", "resource entry is invalid"
            )
        checked = _verify_bound_resource_record(
            workspace=workspace,
            record=item,
            kind=str(item.get("kind") or ""),
            task_id=(str(item["task_id"]) if "task_id" in item else None),
        )
        if checked != item:
            raise ReactLoopError(
                "REACT_RESOURCE_REQUEST_CHANGED", "resource entry changed"
            )
        key = (checked["kind"], checked["logical_path"])
        if key in resource_keys:
            raise ReactLoopError(
                "REACT_RESOURCE_REQUEST_INVALID", "resource entries are duplicated"
            )
        resource_keys.add(key)
        logical.append({key: child for key, child in checked.items() if key != "path"})
    if resources != sorted(
        resources, key=lambda item: (item["kind"], item["logical_path"])
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_REQUEST_CHANGED", "resource entries are not canonical"
        )
    dependencies = normalize_dependencies(request.get("dependencies"))
    if dependencies != request.get("dependencies"):
        raise ReactLoopError(
            "REACT_RESOURCE_REQUEST_CHANGED",
            "resource dependencies are not canonical",
        )
    bundle_core = {
        "project_id": request["project_id"],
        "requirements_binding": request["requirements_binding"],
        "joint_plan": request["joint_plan"],
        "test_plan": request["test_plan"],
        "resources": logical,
        "dependencies": dependencies,
    }
    if request.get("bundle_sha256") != canonical_sha256(bundle_core):
        raise ReactLoopError(
            "REACT_RESOURCE_BUNDLE_CHANGED", "resource bundle hash changed"
        )
    return request


def _reuse_resource_record(
    *,
    workspace_root: Path,
    path: Path,
    candidate: Mapping[str, Any],
    volatile_field: str,
    digest_field: str,
    validator: Any,
    conflict_code: str,
) -> dict[str, Any]:
    """Reuse a complete crash-window artifact only when stable content matches."""

    if path.is_symlink() or not path.is_file():
        raise ReactLoopError(conflict_code, f"resource artifact is unsafe: {path}")
    try:
        existing = read_json(path, max_bytes=2 * 1024 * 1024)
        checked = validator(workspace_root, existing)
    except (OSError, ValueError, ReactLoopError) as exc:
        raise ReactLoopError(
            conflict_code,
            f"existing resource artifact is not reusable: {path}",
        ) from exc
    ignored = {volatile_field, digest_field}
    existing_stable = {
        key: value for key, value in checked.items() if key not in ignored
    }
    candidate_stable = {
        key: value for key, value in candidate.items() if key not in ignored
    }
    if existing_stable != candidate_stable:
        raise ReactLoopError(
            conflict_code,
            f"existing resource artifact has different bound content: {path}",
        )
    return checked


def _validate_decision_record(
    _workspace_root: Path, value: Mapping[str, Any]
) -> dict[str, Any]:
    marker = strict_json_copy(value, label="resource decision marker")
    digest = marker.get("decision_sha256")
    unsigned = {
        key: child for key, child in marker.items() if key != "decision_sha256"
    }
    if not isinstance(digest, str) or digest != canonical_sha256(unsigned):
        raise ReactLoopError(
            "REACT_RESOURCE_DECISION_INVALID",
            "resource decision marker hash changed",
        )
    return marker


def validate_resource_decision(
    *,
    workspace_root: str | Path,
    project_id: str,
    requirements_sha256: str,
    request: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate any bound reviewer decision, including revise and reject."""

    checked_request = validate_resource_request(workspace_root, request)
    checked_marker = _validate_decision_record(
        Path(workspace_root).resolve(), marker
    )
    marker_fields = {
        "schema_version",
        "request_id",
        "request_sha256",
        "bundle_sha256",
        "project_id",
        "requirements_sha256",
        "approver",
        "approver_type",
        "decision",
        "effect",
        "note",
        "decided_at",
        "decision_sha256",
    }
    decision = checked_marker.get("decision")
    approver_type = checked_marker.get("approver_type")
    expected_effect = (
        "PROJECT_APPROVED"
        if decision == "APPROVE" and approver_type == "reviewer_ai"
        else "CATALOG_ADMISSION_APPROVED"
        if decision == "APPROVE" and approver_type == "human"
        else decision
    )
    if (
        set(checked_marker) != marker_fields
        or checked_marker.get("schema_version") != RESOURCE_DECISION_SCHEMA
        or decision not in {"APPROVE", "REVISE", "REJECT"}
        or approver_type not in {"reviewer_ai", "human"}
        or checked_marker.get("effect") != expected_effect
        or checked_marker.get("request_id") != checked_request["request_id"]
        or checked_marker.get("request_sha256")
        != checked_request["request_sha256"]
        or checked_marker.get("bundle_sha256")
        != checked_request["bundle_sha256"]
        or checked_marker.get("requirements_sha256") != requirements_sha256
        or checked_marker.get("requirements_sha256")
        != checked_request["requirements_binding"]["sha256"]
        or checked_marker.get("project_id") != checked_request["project_id"]
        or checked_marker.get("project_id") != project_id
        or not isinstance(checked_marker.get("approver"), str)
        or not checked_marker["approver"].strip()
        or not isinstance(checked_marker.get("note"), str)
        or not checked_marker["note"].strip()
        or not isinstance(checked_marker.get("decided_at"), str)
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_DECISION_INVALID",
            "resource decision marker is missing, changed, or out of scope",
        )
    return checked_marker


def write_resource_decision(
    *,
    workspace_root: str | Path,
    loop_dir: str | Path,
    request: Mapping[str, Any],
    approver: str,
    approver_type: str,
    decision: str,
    note: str,
) -> tuple[dict[str, Any], Path]:
    """Persist the approval marker generated by the reviewer or human."""

    checked = validate_resource_request(workspace_root, request)
    if approver_type not in {"reviewer_ai", "human"}:
        raise ReactLoopError(
            "REACT_RESOURCE_APPROVER_INVALID",
            "approver_type must be reviewer_ai or human",
        )
    if decision not in {"APPROVE", "REVISE", "REJECT"}:
        raise ReactLoopError(
            "REACT_RESOURCE_DECISION_INVALID",
            "decision must be APPROVE, REVISE, or REJECT",
        )
    if not isinstance(approver, str) or not approver.strip() or len(approver) > 256:
        raise ReactLoopError(
            "REACT_RESOURCE_APPROVER_INVALID", "approver identity is invalid"
        )
    if not isinstance(note, str) or not note.strip() or len(note) > 8192:
        raise ReactLoopError(
            "REACT_RESOURCE_DECISION_INVALID", "decision note is invalid"
        )
    effect = (
        "PROJECT_APPROVED"
        if decision == "APPROVE" and approver_type == "reviewer_ai"
        else "CATALOG_ADMISSION_APPROVED"
        if decision == "APPROVE"
        else decision
    )
    core = {
        "schema_version": RESOURCE_DECISION_SCHEMA,
        "request_id": checked["request_id"],
        "request_sha256": checked["request_sha256"],
        "bundle_sha256": checked["bundle_sha256"],
        "project_id": checked["project_id"],
        "requirements_sha256": checked["requirements_binding"]["sha256"],
        "approver": approver.strip(),
        "approver_type": approver_type,
        "decision": decision,
        "effect": effect,
        "note": note.strip(),
        "decided_at": _utc_now(),
    }
    marker = {**core, "decision_sha256": canonical_sha256(core)}
    path = (
        Path(loop_dir).resolve()
        / "resource_approvals"
        / "decisions"
        / f"{checked['request_id']}.json"
    )
    try:
        _write_new_json(path, marker)
    except FileExistsError:
        marker = _reuse_resource_record(
            workspace_root=Path(workspace_root).resolve(),
            path=path,
            candidate=marker,
            volatile_field="decided_at",
            digest_field="decision_sha256",
            validator=_validate_decision_record,
            conflict_code="REACT_RESOURCE_DECISION_CONFLICT",
        )
    return marker, path


def validate_resource_approval(
    *,
    workspace_root: str | Path,
    project_id: str,
    requirements_sha256: str,
    request: Mapping[str, Any],
    marker: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        checked_marker = validate_resource_decision(
            workspace_root=workspace_root,
            project_id=project_id,
            requirements_sha256=requirements_sha256,
            request=request,
            marker=marker,
        )
    except ReactLoopError as exc:
        if exc.code == "REACT_RESOURCE_DECISION_INVALID":
            raise ReactLoopError(
                "REACT_RESOURCE_APPROVAL_INVALID",
                "resource approval marker is missing, changed, or out of scope",
            ) from exc
        raise
    if (
        checked_marker.get("decision") != "APPROVE"
        or checked_marker.get("effect")
        not in {"PROJECT_APPROVED", "CATALOG_ADMISSION_APPROVED"}
    ):
        raise ReactLoopError(
            "REACT_RESOURCE_APPROVAL_INVALID",
            "resource approval marker is missing, changed, or out of scope",
        )
    return checked_marker


__all__ = [
    "RESOURCE_DECISION_SCHEMA",
    "RESOURCE_REQUEST_SCHEMA",
    "create_resource_request",
    "normalize_dependencies",
    "validate_resource_approval",
    "validate_resource_decision",
    "validate_resource_request",
    "write_resource_decision",
]
