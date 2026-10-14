"""Shared payload validators for Loop-produced planning documents.

These helpers define the *shape* contract of worker deliverables
(task understanding, plan, design submission).  The Loop uses them when
accepting an Action; the Harness (via ``contracts.planning``) uses the
same code when re-validating a frozen planning contract before
execution, so both sides agree byte-for-byte on what canonical form is.
"""

from __future__ import annotations

import json
from pathlib import PurePosixPath
import re
from typing import Any

from .errors import ReactLoopError


MAX_STAGE_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_TEXT = 8_192
MAX_LIST_ITEMS = 512
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")


def strict_json_copy(value: Any, *, label: str) -> Any:
    """Detach finite JSON and enforce the common turn-document bound."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReactLoopError(
            "REACT_JSON_INVALID",
            f"{label} must be finite JSON: {exc}",
        ) from exc
    if len(encoded) > MAX_STAGE_DOCUMENT_BYTES:
        raise ReactLoopError(
            "REACT_DOCUMENT_TOO_LARGE",
            f"{label} exceeds {MAX_STAGE_DOCUMENT_BYTES} bytes",
        )
    return json.loads(encoded.decode("utf-8"))


def _string(value: Any, *, label: str, limit: int = MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} must be a non-empty string of at most {limit} characters",
        )
    return value.strip()


def _string_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or len(value) > MAX_LIST_ITEMS
    ):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} must be a bounded JSON list",
        )
    output = [_string(item, label=f"{label}[{index}]", limit=1024) for index, item in enumerate(value)]
    if len(set(output)) != len(output):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} must not contain duplicates",
        )
    return output


def _closed_object(
    value: Any,
    *,
    label: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} must be a JSON object",
        )
    allowed = required | (optional or set())
    if not required.issubset(value) or not set(value).issubset(allowed):
        missing = sorted(required - set(value))
        unknown = sorted(set(value) - allowed)
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} has missing fields {missing} or unknown fields {unknown}",
        )
    return dict(value)


def _understanding_payload(value: Any) -> dict[str, Any]:
    payload = _closed_object(
        value,
        label="payload",
        required={"task_understanding"},
    )
    understanding = _closed_object(
        payload["task_understanding"],
        label="task_understanding",
        required={
            "summary",
            "functional_goals",
            "required_components",
            "assumptions",
        },
        optional={"no_joint_goal"},
    )
    normalized: dict[str, Any] = {
        "summary": _string(
            understanding["summary"],
            label="task_understanding.summary",
        ),
        "functional_goals": _string_list(
            understanding["functional_goals"],
            label="task_understanding.functional_goals",
        ),
        "required_components": _string_list(
            understanding["required_components"],
            label="task_understanding.required_components",
        ),
        "assumptions": _string_list(
            understanding["assumptions"],
            label="task_understanding.assumptions",
            allow_empty=True,
        ),
    }
    if understanding.get("no_joint_goal") is not None:
        normalized["no_joint_goal"] = _string(
            understanding["no_joint_goal"],
            label="task_understanding.no_joint_goal",
            limit=1024,
        )
    return {"task_understanding": normalized}


_JOINT_TARGET_FIELDS = {
    "capsule_id",
    "family",
    "mechanism_kind",
    "input_port_id",
    "observable_id",
}
_TEST_TARGET_FIELDS = {
    "capsule_id",
    "load_case_id",
    "metric_id",
    "phase_id",
    "observable_id",
}
_TEST_BRANCHES = {"physical", "ideal_reference", "ideal_dynamics"}
_TEMPLATE_SELECTION_FIELDS = {
    "kind",
    "plan_item_id",
    "template_id",
    "version",
    "sha256",
    "selection_id",
}


def _target(
    value: Any,
    *,
    label: str,
    allowed: set[str],
    identity: set[str],
) -> dict[str, Any]:
    if not isinstance(value, dict) or not value or not set(value).issubset(allowed):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} must be a non-empty closed target object",
        )
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        normalized[key] = _string(item, label=f"{label}.{key}", limit=256)
    if not any(key in normalized for key in identity):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            f"{label} must identify one of {sorted(identity)}",
        )
    return normalized


def _plan_payload(value: Any) -> dict[str, Any]:
    payload = _closed_object(
        value,
        label="payload",
        required={
            "joint_plan",
            "test_plan",
            "authored_test_tasks",
            "template_selections",
        },
    )
    joints = payload["joint_plan"]
    tests = payload["test_plan"]
    authored = payload["authored_test_tasks"]
    template_selections = payload["template_selections"]
    if not isinstance(joints, list) or len(joints) > MAX_LIST_ITEMS:
        raise ReactLoopError("REACT_RESPONSE_INVALID", "joint_plan must be a bounded list")
    if not isinstance(tests, list) or not tests or len(tests) > MAX_LIST_ITEMS:
        raise ReactLoopError("REACT_RESPONSE_INVALID", "test_plan must be a non-empty bounded list")
    if not isinstance(authored, list) or len(authored) > 128:
        raise ReactLoopError("REACT_RESPONSE_INVALID", "authored_test_tasks must be a bounded list")
    if (
        not isinstance(template_selections, list)
        or len(template_selections) > MAX_LIST_ITEMS
    ):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            "template_selections must be a bounded list",
        )

    normalized_joints: list[dict[str, Any]] = []
    seen_joint_ids: set[str] = set()
    for index, raw in enumerate(joints):
        item = _closed_object(
            raw,
            label=f"joint_plan[{index}]",
            required={"joint_id", "goal", "target"},
        )
        joint_id = _string(item["joint_id"], label=f"joint_plan[{index}].joint_id", limit=128)
        if _IDENTIFIER.fullmatch(joint_id) is None or joint_id in seen_joint_ids:
            raise ReactLoopError("REACT_RESPONSE_INVALID", "joint ids must be unique identifiers")
        seen_joint_ids.add(joint_id)
        normalized_joints.append(
            {
                "joint_id": joint_id,
                "goal": _string(item["goal"], label=f"joint_plan[{index}].goal"),
                "target": _target(
                    item["target"],
                    label=f"joint_plan[{index}].target",
                    allowed=_JOINT_TARGET_FIELDS,
                    identity={"capsule_id", "family"},
                ),
            }
        )

    normalized_tests: list[dict[str, Any]] = []
    seen_test_ids: set[str] = set()
    for index, raw in enumerate(tests):
        item = _closed_object(
            raw,
            label=f"test_plan[{index}]",
            required={"test_id", "goal", "target", "required_branches"},
            optional={"authored_task_id"},
        )
        test_id = _string(item["test_id"], label=f"test_plan[{index}].test_id", limit=128)
        if _IDENTIFIER.fullmatch(test_id) is None or test_id in seen_test_ids:
            raise ReactLoopError("REACT_RESPONSE_INVALID", "test ids must be unique identifiers")
        seen_test_ids.add(test_id)
        branches = _string_list(
            item["required_branches"],
            label=f"test_plan[{index}].required_branches",
        )
        if not set(branches).issubset(_TEST_BRANCHES):
            raise ReactLoopError("REACT_RESPONSE_INVALID", "test plan uses an unsupported branch")
        normalized = {
            "test_id": test_id,
            "goal": _string(item["goal"], label=f"test_plan[{index}].goal"),
            "target": _target(
                item["target"],
                label=f"test_plan[{index}].target",
                allowed=_TEST_TARGET_FIELDS,
                identity={"load_case_id", "metric_id"},
            ),
            "required_branches": branches,
        }
        if item.get("authored_task_id") is not None:
            normalized["authored_task_id"] = _string(
                item["authored_task_id"],
                label=f"test_plan[{index}].authored_task_id",
                limit=128,
            )
        normalized_tests.append(normalized)

    normalized_authored: list[dict[str, str]] = []
    seen_authored_ids: set[str] = set()
    seen_authored_paths: set[str] = set()
    for index, raw in enumerate(authored):
        item = _closed_object(
            raw,
            label=f"authored_test_tasks[{index}]",
            required={"task_id", "path"},
        )
        task_id = _string(item["task_id"], label=f"authored_test_tasks[{index}].task_id", limit=128)
        path = safe_relative_path(
            item["path"],
            label=f"authored_test_tasks[{index}].path",
            required_root="authored_tests",
        )
        if not path.endswith(".py"):
            raise ReactLoopError(
                "REACT_RESPONSE_INVALID",
                "authored test tasks must be Python files ending in .py",
            )
        if task_id in seen_authored_ids or path in seen_authored_paths:
            raise ReactLoopError("REACT_RESPONSE_INVALID", "authored task ids and paths must be unique")
        seen_authored_ids.add(task_id)
        seen_authored_paths.add(path)
        normalized_authored.append({"task_id": task_id, "path": path})
    referenced = {
        item["authored_task_id"]
        for item in normalized_tests
        if "authored_task_id" in item
    }
    if not referenced.issubset(seen_authored_ids):
        raise ReactLoopError(
            "REACT_RESPONSE_INVALID",
            "test_plan references an authored task that was not submitted",
        )
    normalized_selections: list[dict[str, str]] = []
    seen_selection_items: set[tuple[str, str]] = set()
    for index, raw in enumerate(template_selections):
        item = _closed_object(
            raw,
            label=f"template_selections[{index}]",
            required=set(_TEMPLATE_SELECTION_FIELDS),
        )
        kind = _string(
            item["kind"], label=f"template_selections[{index}].kind", limit=32
        )
        normalized = {
            "kind": kind,
            "plan_item_id": _string(
                item["plan_item_id"],
                label=f"template_selections[{index}].plan_item_id",
                limit=128,
            ),
            "template_id": _string(
                item["template_id"],
                label=f"template_selections[{index}].template_id",
                limit=256,
            ),
            "version": _string(
                item["version"],
                label=f"template_selections[{index}].version",
                limit=64,
            ),
            "sha256": _string(
                item["sha256"],
                label=f"template_selections[{index}].sha256",
                limit=64,
            ),
            "selection_id": _string(
                item["selection_id"],
                label=f"template_selections[{index}].selection_id",
                limit=256,
            ),
        }
        if kind not in {"joint", "task_suite"}:
            raise ReactLoopError(
                "REACT_RESPONSE_INVALID",
                "template selection kind must be joint or task_suite",
            )
        for name in ("plan_item_id", "template_id", "selection_id"):
            if _IDENTIFIER.fullmatch(normalized[name]) is None:
                raise ReactLoopError(
                    "REACT_RESPONSE_INVALID",
                    f"template selection {name} must be an identifier",
                )
        if re.fullmatch(
            r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
            normalized["version"],
        ) is None or _HEX_64.fullmatch(normalized["sha256"]) is None:
            raise ReactLoopError(
                "REACT_RESPONSE_INVALID",
                "template selection version or sha256 is invalid",
            )
        identity = (kind, normalized["plan_item_id"])
        if identity in seen_selection_items:
            raise ReactLoopError(
                "REACT_RESPONSE_INVALID",
                "template selections must bind each plan item at most once",
            )
        seen_selection_items.add(identity)
        normalized_selections.append(normalized)
    return {
        "joint_plan": normalized_joints,
        "test_plan": normalized_tests,
        "authored_test_tasks": normalized_authored,
        "template_selections": normalized_selections,
    }


def safe_relative_path(value: Any, *, label: str, required_root: str) -> str:
    path_text = _string(value, label=label, limit=1024)
    if "\\" in path_text or "\x00" in path_text:
        raise ReactLoopError("REACT_OUTPUT_PATH_INVALID", f"{label} is not a portable relative path")
    path = PurePosixPath(path_text)
    if path.is_absolute() or not path.parts or path.parts[0] != required_root:
        raise ReactLoopError(
            "REACT_OUTPUT_PATH_INVALID",
            f"{label} must remain below {required_root}/",
        )
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ReactLoopError("REACT_OUTPUT_PATH_INVALID", f"{label} contains traversal components")
    if required_root == "authored_tests" and len(path.parts) != 2:
        raise ReactLoopError(
            "REACT_OUTPUT_PATH_INVALID",
            f"{label} must name one direct child of authored_tests/",
        )
    return path.as_posix()


def _design_payload(value: Any) -> dict[str, Any]:
    payload = _closed_object(
        value,
        label="payload",
        required={"package_path", "summary"},
        optional={"revision_hypothesis"},
    )
    normalized: dict[str, Any] = {
        "package_path": safe_relative_path(
            payload["package_path"],
            label="payload.package_path",
            required_root="design",
        ),
        "summary": _string(payload["summary"], label="payload.summary"),
    }
    if payload.get("revision_hypothesis") is not None:
        hypothesis = _closed_object(
            payload["revision_hypothesis"],
            label="payload.revision_hypothesis",
            required={"hypothesis", "expected_metric_changes"},
        )
        changes = hypothesis["expected_metric_changes"]
        # An empty list is a valid honest prediction for a pure
        # document/format fix; the Loop separately requires at least one
        # entry when the previous failure was a measured mechanical gate.
        if not isinstance(changes, list) or len(changes) > 128:
            raise ReactLoopError(
                "REACT_RESPONSE_INVALID",
                "revision_hypothesis.expected_metric_changes must be a list "
                "of at most 128 entries",
            )
        normalized_changes: list[dict[str, Any]] = []
        seen_metrics: set[str] = set()
        for index, raw in enumerate(changes):
            item = _closed_object(
                raw,
                label=f"expected_metric_changes[{index}]",
                required={
                    "metric_id",
                    "direction",
                    "minimum_abs_delta",
                    "tolerance",
                },
            )
            metric_id = _string(
                item["metric_id"],
                label=f"expected_metric_changes[{index}].metric_id",
                limit=256,
            )
            direction = _string(
                item["direction"],
                label=f"expected_metric_changes[{index}].direction",
                limit=32,
            )
            minimum = item["minimum_abs_delta"]
            tolerance = item["tolerance"]
            if (
                not metric_id.startswith("validator.")
                or metric_id in seen_metrics
                or direction not in {"increase", "decrease", "change", "stable"}
                or isinstance(minimum, bool)
                or not isinstance(minimum, (int, float))
                or float(minimum) < 0
                or isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float))
                or float(tolerance) < 0
            ):
                raise ReactLoopError(
                    "REACT_RESPONSE_INVALID",
                    "revision expected changes must name unique validator.* metrics, "
                    "valid directions, and non-negative numeric bounds",
                )
            seen_metrics.add(metric_id)
            normalized_changes.append(
                {
                    "metric_id": metric_id,
                    "direction": direction,
                    "minimum_abs_delta": float(minimum),
                    "tolerance": float(tolerance),
                }
            )
        normalized["revision_hypothesis"] = {
            "hypothesis": _string(
                hypothesis["hypothesis"],
                label="payload.revision_hypothesis.hypothesis",
            ),
            "expected_metric_changes": normalized_changes,
        }
    return normalized


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise ReactLoopError("REACT_STATE_INVALID", f"{label} must be a SHA-256 digest")
    return value
