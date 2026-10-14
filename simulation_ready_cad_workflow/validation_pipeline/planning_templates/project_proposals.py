"""Format gate for Worker-authored joint and task-suite proposals.

Catalog JSON is the only reviewer surface. Python MechanismAdapter files and
family specs that name ``project_adapter_file`` never reach the reviewer.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from typing import Any, Sequence

from ..mechanism.family_plugins import family_spec_plugin_for_kind
from ..mechanism.io import read_json
from ..mechanism.task_motion_evidence import installed_task_programs
from .loader import (
    JOINT_TEMPLATE_SCHEMA,
    TASK_SUITE_TEMPLATE_SCHEMA,
    PlanningTemplateError,
    _REPOSITORY_ROOT,
    _fail,
    _validate_joint_template,
    _validate_task_suite_template,
    load_builtin_template_catalog,
)


PROPOSAL_APPROVAL_STATUS = "PROPOSED"
_MAX_TEMPLATE_BYTES = 2 * 1024 * 1024


def _raise(code: str, message: str) -> None:
    raise _fail(code, message)


def _proposal_task_inventory(
    catalog,
) -> tuple[frozenset[str], frozenset[str]]:
    programs = set(installed_task_programs())
    predicates: set[str] = set()
    for record in catalog.records_for("task_suite"):
        payload = record.payload()
        for scenario in payload.get("scenarios") or []:
            if (
                isinstance(scenario, Mapping)
                and scenario.get("runtime_status") == "ACTIVE"
            ):
                program = scenario.get("test_program")
                if isinstance(program, str) and program:
                    programs.add(program)
        for task in payload.get("tasks") or []:
            if not isinstance(task, Mapping):
                continue
            if task.get("runtime_status") != "ACTIVE":
                continue
            for criterion in task.get("criteria") or []:
                if (
                    isinstance(criterion, Mapping)
                    and criterion.get("evaluator_kind") == "suite_predicate"
                ):
                    evaluator_id = criterion.get("evaluator_id")
                    if isinstance(evaluator_id, str) and evaluator_id:
                        predicates.add(evaluator_id)
    return frozenset(programs), frozenset(predicates)


def validate_project_planning_template_bundle(
    resources: Sequence[Mapping[str, Any]],
    *,
    workspace: Path,
    repository_root: Path | None = None,
) -> None:
    """Validate every submitted ``mechanism_templates`` JSON as a catalog-shaped proposal."""

    root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root).resolve()
    workspace = Path(workspace).resolve()
    catalog = load_builtin_template_catalog()
    installed_programs, installed_predicates = _proposal_task_inventory(catalog)
    catalog_ids = {
        (record.kind, record.template_id): record
        for record in catalog.records
    }
    joints: dict[tuple[str, str], dict[str, Any]] = {}
    suites: list[tuple[str, dict[str, Any]]] = []
    for resource in resources:
        logical_path = str(resource.get("logical_path") or "")
        source = workspace.joinpath(*Path(str(resource["path"])).parts)
        if source.suffix != ".json":
            _raise(
                "TEMPLATE_KIND_INVALID",
                f"{logical_path} is not a joint or task-suite JSON template",
            )
        payload = read_json(source, max_bytes=_MAX_TEMPLATE_BYTES)
        if not isinstance(payload, dict):
            _raise("TEMPLATE_OBJECT_REQUIRED", f"{logical_path} must be a JSON object")
        kind = payload.get("kind")
        schema = payload.get("schema_version")
        template_id = payload.get("template_id")
        version = payload.get("version")
        if kind == "joint" and schema == JOINT_TEMPLATE_SCHEMA:
            _validate_joint_template(
                payload,
                repository_root=root,
                approval_status=PROPOSAL_APPROVAL_STATUS,
                workspace_root=workspace,
                forbid_project_adapter=True,
            )
            identity = (str(template_id), str(version))
            if identity in joints:
                _raise(
                    "TEMPLATE_CATALOG_DUPLICATE",
                    f"{logical_path} duplicates another proposed joint identity",
                )
            if ("joint", str(template_id)) in catalog_ids:
                _raise(
                    "TEMPLATE_CATALOG_DUPLICATE",
                    f"{logical_path} duplicates built-in joint {template_id}; "
                    "reuse the catalog entry or report HARNESS_TEMPLATE_DEFECT",
                )
            joints[identity] = {
                "payload": payload,
                "sha256": resource["sha256"],
                "logical_path": logical_path,
            }
            for variant in payload["capability_variants"]:
                if variant.get("runtime_status") != "IMPLEMENTATION_REQUIRED":
                    continue
                kind = variant.get("mechanism_kind")
                if not isinstance(kind, str) or not kind.strip():
                    _raise(
                        "TEMPLATE_PLUGIN_KIND_REQUIRED",
                        f"{logical_path} IMPLEMENTATION_REQUIRED variants must "
                        "name a mechanism_kind for the later reviewer plugin",
                    )
                owner = family_spec_plugin_for_kind(kind)
                if owner is not None:
                    _raise(
                        "TEMPLATE_PLUGIN_KIND_INSTALLED",
                        f"{logical_path} mechanism_kind {kind!r} is already "
                        f"implemented by {owner.plugin_id}; set runtime_status "
                        "ACTIVE and plugin_id to that backend",
                    )
            continue
        if kind == "task_suite" and schema == TASK_SUITE_TEMPLATE_SCHEMA:
            _validate_task_suite_template(
                payload,
                approval_status=PROPOSAL_APPROVAL_STATUS,
                installed_programs=installed_programs,
                installed_predicates=installed_predicates,
            )
            if ("task_suite", str(template_id)) in catalog_ids:
                _raise(
                    "TEMPLATE_CATALOG_DUPLICATE",
                    f"{logical_path} duplicates built-in task suite {template_id}; "
                    "reuse the catalog entry or report HARNESS_TEMPLATE_DEFECT",
                )
            suites.append((logical_path, payload))
            continue
        _raise(
            "TEMPLATE_KIND_INVALID",
            f"{logical_path} is not a joint template or task suite",
        )

    catalog_joints = {
        (record.template_id, record.version): record
        for record in catalog.records
        if record.kind == "joint"
    }
    for logical_path, payload in suites:
        for index, ref in enumerate(payload["joint_template_refs"]):
            label = f"{logical_path} joint_template_refs[{index}]"
            identity = (ref["template_id"], ref["version"])
            catalog_record = catalog_joints.get(identity)
            bundled = joints.get(identity)
            if catalog_record is not None:
                target_payload = catalog_record.payload()
                target_hash = catalog_record.sha256
            elif bundled is not None:
                target_payload = bundled["payload"]
                target_hash = bundled["sha256"]
            else:
                _raise(
                    "TEMPLATE_REFERENCE_INVALID",
                    f"{label} does not resolve to a catalog joint or a joint in this bundle",
                )
            if target_hash != ref["sha256"]:
                _raise(
                    "TEMPLATE_REFERENCE_CHANGED",
                    f"{label} sha256 does not match the referenced joint",
                )
            variants = {
                item["variant_id"]: item
                for item in target_payload["capability_variants"]
            }
            if ref["variant_id"] not in variants:
                _raise(
                    "TEMPLATE_REFERENCE_INVALID",
                    f"{label} selects an unknown joint variant",
                )
