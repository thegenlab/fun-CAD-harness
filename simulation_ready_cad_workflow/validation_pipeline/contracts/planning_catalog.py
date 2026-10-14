"""Bind ReAct planning items to the fixed built-in template catalog."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from ..planning_templates import (
    PlanningTemplateError,
    load_builtin_template_catalog,
    resolve_builtin_template,
)
from .errors import ReactLoopError
from .payloads import strict_json_copy


MAX_TEMPLATE_SELECTIONS = 512
_SELECTION_FIELDS = frozenset(
    {
        "kind",
        "plan_item_id",
        "template_id",
        "version",
        "sha256",
        "selection_id",
    }
)


_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def validate_catalog_snapshot(value: Any) -> dict[str, Any]:
    """Validate and detach the public catalog frozen for one task."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema_version",
        "catalog_id",
        "catalog_version",
        "catalog_sha256",
        "entries",
    }:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_INVALID",
            "planning template snapshot has invalid top-level fields",
        )
    if value.get("schema_version") != "simulation_ready_cad_public_template_catalog_v1":
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_INVALID",
            "planning template snapshot schema is unsupported",
        )
    _nonblank(value.get("catalog_id"), label="catalog_id")
    _nonblank(value.get("catalog_version"), label="catalog_version", limit=64)
    digest = _nonblank(
        value.get("catalog_sha256"), label="catalog_sha256", limit=64
    )
    if _HEX_64.fullmatch(digest) is None:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_INVALID",
            "planning template snapshot catalog_sha256 is invalid",
        )
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > 512:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_INVALID",
            "planning template snapshot entries must be non-empty and bounded",
        )
    seen: set[tuple[str, str, str]] = set()
    for index, entry in enumerate(entries):
        label = f"planning template snapshot entries[{index}]"
        if not isinstance(entry, Mapping):
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID", f"{label} must be an object"
            )
        kind = entry.get("kind")
        template_id = entry.get("template_id")
        version = entry.get("version")
        sha256 = entry.get("sha256")
        if kind not in {"joint", "task_suite"}:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID", f"{label}.kind is invalid"
            )
        identity = (
            kind,
            _nonblank(template_id, label=f"{label}.template_id"),
            _nonblank(version, label=f"{label}.version", limit=64),
        )
        digest = _nonblank(sha256, label=f"{label}.sha256", limit=64)
        if _HEX_64.fullmatch(digest) is None or identity in seen:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID",
                f"{label} has an invalid hash or duplicate identity",
            )
        seen.add(identity)
        required_list = "capability_variants" if kind == "joint" else "tasks"
        if not isinstance(entry.get(required_list), list):
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID",
                f"{label}.{required_list} must be a list",
            )
        if kind == "task_suite" and not isinstance(entry.get("scenarios"), list):
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID",
                f"{label}.scenarios must be a list",
            )
    return strict_json_copy(value, label="planning template snapshot")


def catalog_binding(snapshot: Mapping[str, Any] | None = None) -> dict[str, str]:
    """Return the exact public catalog identity frozen into a plan."""

    if snapshot is not None:
        frozen = validate_catalog_snapshot(snapshot)
        return {
            "catalog_id": frozen["catalog_id"],
            "catalog_version": frozen["catalog_version"],
            "catalog_sha256": frozen["catalog_sha256"],
        }
    catalog = load_builtin_template_catalog()
    return {
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "catalog_sha256": catalog.catalog_sha256,
    }


def _nonblank(value: Any, *, label: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_INVALID",
            f"{label} must be a non-empty string of at most {limit} characters",
        )
    return value.strip()


def _snapshot_record(
    snapshot: Mapping[str, Any], selection: Mapping[str, str]
) -> dict[str, Any]:
    frozen = validate_catalog_snapshot(snapshot)
    matches = [
        entry
        for entry in frozen["entries"]
        if entry.get("kind") == selection["kind"]
        and entry.get("template_id") == selection["template_id"]
        and entry.get("version") == selection["version"]
    ]
    if len(matches) != 1:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_REFERENCE_UNKNOWN",
            "template selection does not resolve in the task-frozen catalog",
        )
    record = matches[0]
    if record.get("sha256") != selection["sha256"]:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_REFERENCE_CHANGED",
            "template selection hash differs from the task-frozen catalog",
        )
    return record


def resolve_template_selection(
    snapshot: Mapping[str, Any], selection: Mapping[str, str]
) -> dict[str, Any]:
    """Resolve one already-normalized selection in a frozen public catalog."""

    return strict_json_copy(
        _snapshot_record(snapshot, selection), label="selected template record"
    )


def normalize_template_selections(
    value: Any, *, catalog_snapshot: Mapping[str, Any] | None = None
) -> list[dict[str, str]]:
    """Resolve exact catalog references and normalize their selected variant/task."""

    if not isinstance(value, list) or len(value) > MAX_TEMPLATE_SELECTIONS:
        raise ReactLoopError(
            "REACT_PLANNING_TEMPLATE_INVALID",
            "template_selections must be a bounded list",
        )
    output: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw in enumerate(value):
        label = f"template_selections[{index}]"
        if not isinstance(raw, Mapping) or set(raw) != _SELECTION_FIELDS:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID",
                f"{label} must contain exactly {sorted(_SELECTION_FIELDS)}",
            )
        kind = _nonblank(raw.get("kind"), label=f"{label}.kind", limit=32)
        if kind not in {"joint", "task_suite"}:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_INVALID",
                f"{label}.kind must be joint or task_suite",
            )
        normalized = {
            "kind": kind,
            "plan_item_id": _nonblank(
                raw.get("plan_item_id"), label=f"{label}.plan_item_id"
            ),
            "template_id": _nonblank(
                raw.get("template_id"), label=f"{label}.template_id"
            ),
            "version": _nonblank(
                raw.get("version"), label=f"{label}.version", limit=64
            ),
            "sha256": _nonblank(
                raw.get("sha256"), label=f"{label}.sha256", limit=64
            ),
            "selection_id": _nonblank(
                raw.get("selection_id"), label=f"{label}.selection_id"
            ),
        }
        if catalog_snapshot is None:
            try:
                record = resolve_builtin_template(
                    {
                        "template_id": normalized["template_id"],
                        "version": normalized["version"],
                        "sha256": normalized["sha256"],
                    },
                    expected_kind=kind,
                )
            except PlanningTemplateError as exc:
                raise ReactLoopError(exc.code, exc.message) from exc
            payload = record.payload()
        else:
            payload = _snapshot_record(catalog_snapshot, normalized)
        if kind == "joint":
            available = {
                str(item.get("variant_id"))
                for item in payload.get("capability_variants", [])
                if isinstance(item, Mapping)
            }
            selection_label = "variant"
        else:
            available = {
                str(item.get("task_id"))
                for item in payload.get("tasks", [])
                if isinstance(item, Mapping)
            }
            selection_label = "task"
        if normalized["selection_id"] not in available:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_SELECTION_UNKNOWN",
                f"{label} selects unknown {selection_label} "
                f"{normalized['selection_id']!r}",
            )
        identity = (kind, normalized["plan_item_id"])
        if identity in seen:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_DUPLICATE",
                f"{label} repeats the {kind} plan item "
                f"{normalized['plan_item_id']!r}",
            )
        seen.add(identity)
        output.append(normalized)
    return output


def review_template_selections(
    *,
    joint_plan: Sequence[Mapping[str, Any]],
    test_plan: Sequence[Mapping[str, Any]],
    template_selections: Any,
    catalog_snapshot: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Validate executable selections and return non-blocking binding warnings."""

    selections = normalize_template_selections(
        template_selections, catalog_snapshot=catalog_snapshot
    )
    joints = {
        str(item.get("joint_id")): item
        for item in joint_plan
        if isinstance(item, Mapping)
    }
    tests = {
        str(item.get("test_id")): item
        for item in test_plan
        if isinstance(item, Mapping)
    }
    warnings: list[dict[str, str]] = []
    selected_joint_ids: set[str] = set()
    selected_test_ids: set[str] = set()
    for selection in selections:
        kind = selection["kind"]
        plan_item_id = selection["plan_item_id"]
        items = joints if kind == "joint" else tests
        if plan_item_id not in items:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_PLAN_ITEM_UNKNOWN",
                f"{kind} template selection references unknown plan item "
                f"{plan_item_id!r}",
            )
        if catalog_snapshot is None:
            try:
                record = resolve_builtin_template(
                    {
                        "template_id": selection["template_id"],
                        "version": selection["version"],
                        "sha256": selection["sha256"],
                    },
                    expected_kind=kind,
                )
            except PlanningTemplateError as exc:
                raise ReactLoopError(exc.code, exc.message) from exc
            payload = record.payload()
        else:
            payload = _snapshot_record(catalog_snapshot, selection)
        item = items[plan_item_id]
        target = item.get("target")
        target = target if isinstance(target, Mapping) else {}
        if kind == "joint":
            selected_joint_ids.add(plan_item_id)
            planned_family = target.get("family")
            template_family = payload.get("family")
            if (
                isinstance(planned_family, str)
                and planned_family
                and planned_family != template_family
            ):
                warnings.append(
                    {
                        "code": "JOINT_TEMPLATE_BINDING_DIFFERS",
                        "path": f"joint_plan.{plan_item_id}.target.family",
                        "message": (
                            f"planned family {planned_family!r} differs from "
                            f"template family {template_family!r}"
                        ),
                    }
                )
            continue

        selected_test_ids.add(plan_item_id)
        task = next(
            (
                candidate
                for candidate in payload["tasks"]
                if isinstance(candidate, Mapping)
                and candidate.get("task_id") == selection["selection_id"]
            ),
            None,
        )
        if not isinstance(task, Mapping):
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_SELECTION_UNKNOWN",
                f"selected task {selection['selection_id']!r} is missing",
            )
        scenario = next(
            (
                candidate
                for candidate in payload["scenarios"]
                if isinstance(candidate, Mapping)
                and candidate.get("scenario_id") == task.get("scenario_id")
            ),
            None,
        )
        if not isinstance(scenario, Mapping):
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_SCENARIO_MISSING",
                f"selected task {selection['selection_id']!r} has no scenario",
            )
        authored = item.get("authored_task_id")
        if task.get("runtime_status") != "ACTIVE" and not authored:
            raise ReactLoopError(
                "REACT_PLANNING_TEMPLATE_NOT_EXECUTABLE",
                f"selected task {selection['selection_id']!r} is not ACTIVE "
                "and has no authored test implementation",
            )
        planned_case = target.get("load_case_id")
        template_case = scenario.get("test_load_case_id")
        if planned_case and planned_case != template_case:
            warnings.append(
                {
                    "code": "TASK_TEMPLATE_LOAD_CASE_DIFFERS",
                    "path": f"test_plan.{plan_item_id}.target.load_case_id",
                    "message": (
                        f"planned load case {planned_case!r} differs from "
                        f"template load case {template_case!r}"
                    ),
                }
            )
        evaluator_ids = {
            str(criterion.get("evaluator_id"))
            for criterion in task.get("criteria", [])
            if isinstance(criterion, Mapping)
        }
        planned_metric = target.get("metric_id")
        if planned_metric and planned_metric not in evaluator_ids:
            warnings.append(
                {
                    "code": "TASK_TEMPLATE_METRIC_DIFFERS",
                    "path": f"test_plan.{plan_item_id}.target.metric_id",
                    "message": (
                        f"planned metric {planned_metric!r} is not one of the "
                        f"selected template task evaluators {sorted(evaluator_ids)!r}"
                    ),
                }
            )
        required = set(item.get("required_branches") or [])
        template_required = set(scenario.get("required_branches") or [])
        if required and required != template_required:
            warnings.append(
                {
                    "code": "TASK_TEMPLATE_BRANCHES_DIFFER",
                    "path": f"test_plan.{plan_item_id}.required_branches",
                    "message": (
                        f"planned branches {sorted(required)!r} differ from "
                        f"template branches {sorted(template_required)!r}"
                    ),
                }
            )

    for joint_id in sorted(set(joints) - selected_joint_ids):
        warnings.append(
            {
                "code": "JOINT_TEMPLATE_NOT_SELECTED",
                "path": f"joint_plan.{joint_id}",
                "message": "no built-in joint template is bound; review the project template if one is supplied",
            }
        )
    for test_id in sorted(set(tests) - selected_test_ids):
        if tests[test_id].get("authored_task_id"):
            continue
        warnings.append(
            {
                "code": "TASK_TEMPLATE_NOT_SELECTED",
                "path": f"test_plan.{test_id}",
                "message": "no built-in task template is bound; the frozen task contract remains authoritative",
            }
        )
    return selections, strict_json_copy(warnings, label="planning warnings")


def _active_task_candidates(
    *,
    test_plan: Sequence[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    frozen = validate_catalog_snapshot(catalog_snapshot)
    suites = [
        entry
        for entry in frozen["entries"]
        if isinstance(entry, Mapping) and entry.get("kind") == "task_suite"
    ]
    output: dict[str, list[dict[str, str]]] = {}
    for item in test_plan:
        test_id = str(item.get("test_id") or "")
        target = item.get("target")
        target = target if isinstance(target, Mapping) else {}
        load_case_id = target.get("load_case_id")
        metric_id = target.get("metric_id")
        required_branches = set(item.get("required_branches") or [])
        if not isinstance(load_case_id, str) or not load_case_id:
            continue
        candidates: list[dict[str, str]] = []
        for suite in suites:
            scenarios = {
                str(scenario.get("scenario_id")): scenario
                for scenario in suite.get("scenarios", [])
                if isinstance(scenario, Mapping)
            }
            for task in suite.get("tasks", []):
                if not isinstance(task, Mapping) or task.get("runtime_status") != "ACTIVE":
                    continue
                scenario = scenarios.get(str(task.get("scenario_id") or ""))
                if (
                    not isinstance(scenario, Mapping)
                    or scenario.get("runtime_status") != "ACTIVE"
                    or scenario.get("test_load_case_id") != load_case_id
                ):
                    continue
                evaluator_ids = {
                    str(criterion.get("evaluator_id"))
                    for criterion in task.get("criteria", [])
                    if isinstance(criterion, Mapping)
                }
                if isinstance(metric_id, str) and metric_id not in evaluator_ids:
                    continue
                template_branches = set(scenario.get("required_branches") or [])
                if required_branches and required_branches != template_branches:
                    continue
                candidates.append(
                    {
                        "kind": "task_suite",
                        "plan_item_id": test_id,
                        "template_id": str(suite["template_id"]),
                        "version": str(suite["version"]),
                        "sha256": str(suite["sha256"]),
                        "selection_id": str(task["task_id"]),
                    }
                )
        if candidates:
            output[test_id] = candidates
    return output


def _active_joint_candidates(
    *,
    joint_plan: Sequence[Mapping[str, Any]],
    relevant_suites: Sequence[Mapping[str, Any]],
    catalog_snapshot: Mapping[str, Any],
) -> dict[str, list[dict[str, str]]]:
    frozen = validate_catalog_snapshot(catalog_snapshot)
    joint_entries = {
        (
            str(entry.get("template_id")),
            str(entry.get("version")),
            str(entry.get("sha256")),
        ): entry
        for entry in frozen["entries"]
        if isinstance(entry, Mapping) and entry.get("kind") == "joint"
    }
    refs: list[Mapping[str, Any]] = []
    seen_refs: set[tuple[str, str, str, str]] = set()
    for suite in relevant_suites:
        for ref in suite.get("joint_template_refs", []):
            if not isinstance(ref, Mapping):
                continue
            identity = (
                str(ref.get("template_id") or ""),
                str(ref.get("version") or ""),
                str(ref.get("sha256") or ""),
                str(ref.get("variant_id") or ""),
            )
            if identity in seen_refs:
                continue
            seen_refs.add(identity)
            refs.append(ref)

    output: dict[str, list[dict[str, str]]] = {}
    for item in joint_plan:
        joint_id = str(item.get("joint_id") or "")
        target = item.get("target")
        target = target if isinstance(target, Mapping) else {}
        family = target.get("family")
        mechanism_kind = target.get("mechanism_kind")
        candidates: list[dict[str, str]] = []
        for ref in refs:
            identity = (
                str(ref.get("template_id") or ""),
                str(ref.get("version") or ""),
                str(ref.get("sha256") or ""),
            )
            entry = joint_entries.get(identity)
            variant_id = str(ref.get("variant_id") or "")
            if not isinstance(entry, Mapping) or (
                isinstance(family, str)
                and family
                and entry.get("family") != family
            ):
                continue
            variants = [
                variant
                for variant in entry.get("capability_variants", [])
                if isinstance(variant, Mapping)
                and variant.get("variant_id") == variant_id
                and (
                    not isinstance(mechanism_kind, str)
                    or not mechanism_kind
                    or variant.get("mechanism_kind") == mechanism_kind
                )
            ]
            if not variants:
                continue
            if variants[0].get("runtime_status") != "ACTIVE":
                continue
            candidates.append(
                {
                    "kind": "joint",
                    "plan_item_id": joint_id,
                    "template_id": identity[0],
                    "version": identity[1],
                    "sha256": identity[2],
                    "selection_id": variant_id,
                }
            )
        # Non-active variants remain eligible for project implementation and
        # must not make an independently active built-in variant disappear.
        # More than one active match is ambiguous and therefore stays
        # fail-closed instead of choosing by catalog order.
        if len(candidates) == 1:
            output[joint_id] = candidates
    return output


def reject_redundant_authored_resources(
    *,
    joint_plan: Sequence[Mapping[str, Any]],
    test_plan: Sequence[Mapping[str, Any]],
    authored_test_tasks: Sequence[Mapping[str, Any]],
    mechanism_template_paths: Sequence[str],
    template_selections: Sequence[Mapping[str, str]],
    catalog_snapshot: Mapping[str, Any],
) -> None:
    """Reject project resources when the frozen Harness catalog already covers them."""

    if not authored_test_tasks and not mechanism_template_paths:
        return
    frozen = validate_catalog_snapshot(catalog_snapshot)
    task_candidates = _active_task_candidates(
        test_plan=test_plan,
        catalog_snapshot=frozen,
    )
    conflicts: list[dict[str, str]] = []
    for item in test_plan:
        if item.get("authored_task_id") is None:
            continue
        conflicts.extend(task_candidates.get(str(item.get("test_id") or ""), []))

    suites_by_identity = {
        (
            str(entry.get("template_id")),
            str(entry.get("version")),
            str(entry.get("sha256")),
        ): entry
        for entry in frozen["entries"]
        if isinstance(entry, Mapping) and entry.get("kind") == "task_suite"
    }
    relevant_suites: list[Mapping[str, Any]] = []
    seen_suites: set[tuple[str, str, str]] = set()
    for candidates in task_candidates.values():
        for candidate in candidates:
            identity = (
                candidate["template_id"],
                candidate["version"],
                candidate["sha256"],
            )
            suite = suites_by_identity.get(identity)
            if isinstance(suite, Mapping) and identity not in seen_suites:
                seen_suites.add(identity)
                relevant_suites.append(suite)
    for selection in template_selections:
        if selection.get("kind") != "task_suite":
            continue
        identity = (
            str(selection.get("template_id")),
            str(selection.get("version")),
            str(selection.get("sha256")),
        )
        suite = suites_by_identity.get(identity)
        if isinstance(suite, Mapping) and identity not in seen_suites:
            seen_suites.add(identity)
            relevant_suites.append(suite)

    selected_joints: dict[str, dict[str, str]] = {}
    selected_joint_ids = {
        str(selection.get("plan_item_id"))
        for selection in template_selections
        if selection.get("kind") == "joint"
    }
    for selection in template_selections:
        if selection.get("kind") != "joint":
            continue
        entry = _snapshot_record(frozen, selection)
        selection_id = str(selection.get("selection_id") or "")
        if any(
            isinstance(variant, Mapping)
            and variant.get("variant_id") == selection_id
            and variant.get("runtime_status") == "ACTIVE"
            for variant in entry.get("capability_variants", [])
        ):
            selected_joints[str(selection.get("plan_item_id"))] = dict(
                selection
            )
    linked_joint_candidates = _active_joint_candidates(
        joint_plan=joint_plan,
        relevant_suites=relevant_suites,
        catalog_snapshot=frozen,
    )
    if mechanism_template_paths and joint_plan:
        joint_coverage: dict[str, list[dict[str, str]]] = {}
        for item in joint_plan:
            joint_id = str(item.get("joint_id") or "")
            if joint_id in selected_joints:
                joint_coverage[joint_id] = [selected_joints[joint_id]]
            elif (
                joint_id not in selected_joint_ids
                and joint_id in linked_joint_candidates
            ):
                joint_coverage[joint_id] = linked_joint_candidates[joint_id]
        if len(joint_coverage) == len(joint_plan):
            for candidates in joint_coverage.values():
                conflicts.extend(candidates)

    if not conflicts:
        return
    unique: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for conflict in conflicts:
        identity = (
            conflict["kind"],
            conflict["plan_item_id"],
            conflict["selection_id"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(conflict)
    bindings = "; ".join(
        (
            f"{item['kind']} {item['plan_item_id']} -> "
            f"{item['template_id']}@{item['version']}:{item['selection_id']} "
            f"sha256={item['sha256']}"
        )
        for item in unique
    )
    raise ReactLoopError(
        "REACT_BUILTIN_TEMPLATE_REUSE_REQUIRED",
        "The Harness already provides approved ACTIVE task/joint templates "
        "for the submitted project resources. Remove the redundant authored "
        f"files and bind these catalog entries: {bindings}. If a built-in "
        "template cannot cover the requirement as written, stop and "
        "report the exact typed mismatch through the Harness so it can route "
        "an Operator interrupt; do not contact the dormant Operator directly "
        "or replace it inside this project.",
    )


__all__ = [
    "catalog_binding",
    "normalize_template_selections",
    "reject_redundant_authored_resources",
    "resolve_template_selection",
    "review_template_selections",
    "validate_catalog_snapshot",
]
