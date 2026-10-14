"""Strict fixed-root loader for Harness-owned planning templates.

Project ``mechanism_templates/`` remain proposal attachments.  They are never
searched or allowed to shadow this built-in catalog.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping

from ..intake.acceptance_contract import BRANCHES, METRIC_REGISTRY
from ..mechanism.family_plugins import registered_mechanism_adapters
from .models import TemplateCatalog, TemplateRecord


CATALOG_SCHEMA = "simulation_ready_cad_planning_template_catalog_v1"
JOINT_TEMPLATE_SCHEMA = "simulation_ready_cad_joint_template_v1"
TASK_SUITE_TEMPLATE_SCHEMA = "simulation_ready_cad_machine_task_suite_v1"

_BUILTIN_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _BUILTIN_ROOT.parents[2]
_INDEX_NAME = "index.v1.json"
_MAX_INDEX_BYTES = 2 * 1024 * 1024
_MAX_TEMPLATE_BYTES = 2 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$")
_SEMVER = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_STATUSES = frozenset({"ACTIVE", "IMPLEMENTATION_REQUIRED"})
_KINDS = frozenset({"joint", "task_suite"})
_JOINT_CATEGORIES = frozenset({"primitive_joint", "compound_joint"})
_WRENCH_KINDS = frozenset({"force", "torque"})
_CRITERION_DISPOSITIONS = frozenset({"PASS", "WARNING"})
_EVALUATOR_KINDS = frozenset({"registered_metric", "suite_predicate"})
_COMPARATORS = frozenset(
    {"at_least", "at_most", "equals", "within", "sequence_equals"}
)
_TOLERANCE_MODES = frozenset(
    {"none", "absolute", "shortfall", "relative_percent"}
)
_EVIDENCE_STATUSES = frozenset(
    {
        "TASK_CERTIFIED",
        "LEGACY_UNTYPED",
        "CERTIFICATE_SEMANTIC_GAP",
        "DEMO_DIAGNOSTIC_ONLY",
        "PHYSICAL_PREFLIGHT_FAILED",
        "CAPABILITY_CERTIFIED",
    }
)


class PlanningTemplateError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> PlanningTemplateError:
    return PlanningTemplateError(code, message)


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _read_json(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    if path.is_symlink():
        raise _fail("TEMPLATE_PATH_UNSAFE", f"{label} is a symlink: {path}")
    try:
        node = path.stat()
    except OSError as exc:
        raise _fail("TEMPLATE_FILE_MISSING", f"{label} is unavailable: {path}") from exc
    if not stat.S_ISREG(node.st_mode) or node.st_size > max_bytes:
        raise _fail("TEMPLATE_FILE_INVALID", f"{label} is not a bounded regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise _fail("TEMPLATE_JSON_INVALID", f"{label} is not strict JSON: {path}") from exc
    if not isinstance(value, dict):
        raise _fail("TEMPLATE_OBJECT_REQUIRED", f"{label} must be a JSON object")
    return value


def _closed(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    label: str,
) -> None:
    fields = frozenset(value)
    if fields != required:
        missing = sorted(required - fields)
        extra = sorted(fields - required)
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            f"{label} fields differ; missing={missing}, extra={extra}",
        )


def _text(value: Any, *, label: str, limit: int = 8_192) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be a non-empty bounded string")
    return value.strip()


def _identifier(value: Any, *, label: str) -> str:
    text = _text(value, label=label, limit=128)
    if _IDENTIFIER.fullmatch(text) is None:
        raise _fail("TEMPLATE_ID_INVALID", f"{label} is not a namespaced lowercase identifier")
    return text


def _version(value: Any, *, label: str) -> str:
    text = _text(value, label=label, limit=32)
    if _SEMVER.fullmatch(text) is None:
        raise _fail("TEMPLATE_VERSION_INVALID", f"{label} is not strict semantic versioning")
    return text


def _hex(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise _fail("TEMPLATE_HASH_INVALID", f"{label} must be a lowercase SHA-256")
    return value


def _string_list(value: Any, *, label: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value) or len(value) > 256:
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be a bounded JSON list")
    output = [_text(item, label=f"{label}[{index}]", limit=1_024) for index, item in enumerate(value)]
    if len(set(output)) != len(output):
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must not contain duplicates")
    return output


def _finite_json(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded, parse_constant=_reject_constant)
    except (TypeError, ValueError) as exc:
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be finite JSON") from exc
    if isinstance(decoded, dict):
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            f"{label} must be a scalar or a JSON list, not an open object",
        )
    return decoded


def _diagnostic_code(value: Any, *, label: str) -> str:
    text = _text(value, label=label, limit=128)
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", text) is None:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            f"{label} must be an uppercase diagnostic code",
        )
    return text


def _safe_catalog_path(root: Path, relative: str, *, label: str) -> Path:
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in relative
        or "\x00" in relative
    ):
        raise _fail("TEMPLATE_PATH_UNSAFE", f"{label} is not a safe relative path")
    candidate = root.joinpath(*path.parts)
    current = root
    for part in path.parts[:-1]:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise _fail("TEMPLATE_PATH_UNSAFE", f"{label} contains an unsafe directory")
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_approval(
    value: Any,
    *,
    label: str,
    required_status: str = "APPROVED",
) -> None:
    if not isinstance(value, dict):
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
    _closed(
        value,
        required=frozenset({"status", "approved_by", "approved_on", "basis"}),
        label=label,
    )
    if value.get("status") != required_status:
        code = (
            "TEMPLATE_NOT_APPROVED"
            if required_status == "APPROVED"
            else "TEMPLATE_NOT_PROPOSED"
        )
        raise _fail(code, f"{label}.status must be {required_status}")
    _text(value.get("approved_by"), label=f"{label}.approved_by", limit=128)
    approved_on = _text(value.get("approved_on"), label=f"{label}.approved_on", limit=10)
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", approved_on) is None:
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label}.approved_on must be YYYY-MM-DD")
    _text(value.get("basis"), label=f"{label}.basis")


def _resolve_source_file(
    raw_path: str,
    *,
    label: str,
    repository_root: Path,
    workspace_root: Path | None = None,
) -> Path:
    roots = (repository_root,) if workspace_root is None else (repository_root, workspace_root)
    for root in roots:
        try:
            candidate = _safe_catalog_path(root, raw_path, label=label)
        except PlanningTemplateError as exc:
            if exc.code == "TEMPLATE_PATH_UNSAFE":
                continue
            raise
        try:
            if candidate.is_file() and not candidate.is_symlink():
                return candidate
        except OSError:
            continue
    raise _fail("TEMPLATE_SOURCE_INVALID", f"{label} is unavailable")


def _reject_project_adapter_family_spec(path: Path, *, label: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return
    if isinstance(payload, dict) and "project_adapter_file" in payload:
        raise _fail(
            "TEMPLATE_ADAPTER_NOT_PLANNING_TEMPLATE",
            f"{label} names a project MechanismAdapter; joint/task templates "
            "must bind a registered plugin instance, not a project .py adapter",
        )


def _validate_source_binding(
    value: Any,
    *,
    label: str,
    repository_root: Path,
    workspace_root: Path | None = None,
    forbid_project_adapter: bool = False,
) -> None:
    if not isinstance(value, dict):
        raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
    _closed(
        value,
        required=frozenset(
            {
                "project_id",
                "capsule_id",
                "source_status",
                "capsule_path",
                "capsule_sha256",
                "family_spec_path",
                "family_spec_sha256",
            }
        ),
        label=label,
    )
    _identifier(value.get("project_id"), label=f"{label}.project_id")
    _text(value.get("capsule_id"), label=f"{label}.capsule_id", limit=128)
    if value.get("source_status") not in {
        "PROJECT_CERTIFIED",
        "CAPABILITY_CERTIFIED",
        "DEMO_HISTORICAL_PASS",
        "UNRESOLVED_EXEMPLAR",
    }:
        raise _fail("TEMPLATE_SOURCE_INVALID", f"{label}.source_status is unsupported")
    for path_field, hash_field in (
        ("capsule_path", "capsule_sha256"),
        ("family_spec_path", "family_spec_sha256"),
    ):
        raw_path = value.get(path_field)
        raw_hash = value.get(hash_field)
        if raw_path is None and raw_hash is None:
            continue
        if not isinstance(raw_path, str):
            raise _fail("TEMPLATE_SOURCE_INVALID", f"{label}.{path_field} must be a path or null")
        expected = _hex(raw_hash, label=f"{label}.{hash_field}")
        source = _resolve_source_file(
            raw_path,
            label=f"{label}.{path_field}",
            repository_root=repository_root,
            workspace_root=workspace_root,
        )
        try:
            source_stat = source.stat()
        except OSError as exc:
            raise _fail(
                "TEMPLATE_SOURCE_INVALID",
                f"{label}.{path_field} is unavailable",
            ) from exc
        if (
            source.is_symlink()
            or not stat.S_ISREG(source_stat.st_mode)
            or source_stat.st_size > _MAX_TEMPLATE_BYTES
        ):
            raise _fail(
                "TEMPLATE_SOURCE_INVALID",
                f"{label}.{path_field} must be a bounded regular file",
            )
        if _sha256_file(source) != expected:
            raise _fail("TEMPLATE_SOURCE_CHANGED", f"{label}.{path_field} no longer matches its approved hash")
        if forbid_project_adapter and path_field == "family_spec_path":
            _reject_project_adapter_family_spec(
                source, label=f"{label}.{path_field}"
            )


def _active_plugins_by_id() -> dict[str, Any]:
    return {item.plugin_id: item for item in registered_mechanism_adapters()}


def _validate_joint_template(
    payload: dict[str, Any],
    *,
    repository_root: Path,
    approval_status: str = "APPROVED",
    workspace_root: Path | None = None,
    forbid_project_adapter: bool = False,
) -> None:
    _closed(
        payload,
        required=frozenset(
            {
                "schema_version",
                "kind",
                "template_id",
                "version",
                "title",
                "category",
                "goal",
                "family",
                "capability_variants",
                "input_wrench_kinds",
                "observable_quantities",
                "components",
                "required_instance_fields",
                "source_bindings",
                "constraints",
                "approval",
            }
        ),
        label="joint template",
    )
    if payload.get("schema_version") != JOINT_TEMPLATE_SCHEMA or payload.get("kind") != "joint":
        raise _fail("TEMPLATE_SCHEMA_INVALID", "joint template identity fields are invalid")
    _identifier(payload.get("template_id"), label="joint template_id")
    _version(payload.get("version"), label="joint version")
    _text(payload.get("title"), label="joint title", limit=256)
    if payload.get("category") not in _JOINT_CATEGORIES:
        raise _fail("TEMPLATE_SCHEMA_INVALID", "joint category is unsupported")
    _text(payload.get("goal"), label="joint goal")
    family = _text(payload.get("family"), label="joint family", limit=64)
    if family not in {"revolute", "prismatic", "compound"}:
        raise _fail("TEMPLATE_SCHEMA_INVALID", "joint family is unsupported")
    variants = payload.get("capability_variants")
    if not isinstance(variants, list) or not variants or len(variants) > 16:
        raise _fail("TEMPLATE_SCHEMA_INVALID", "capability_variants must be a non-empty bounded list")
    seen_variants: set[str] = set()
    active_plugins = _active_plugins_by_id()
    for index, item in enumerate(variants):
        label = f"capability_variants[{index}]"
        if not isinstance(item, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        base_fields = frozenset(
            {"variant_id", "plugin_id", "mechanism_kind", "runtime_status"}
        )
        contract_fields = frozenset({"capability_ids", "model_boundary_ids"})
        if frozenset(item) not in {base_fields, base_fields | contract_fields}:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label} fields must be the base variant fields with an optional complete capability contract",
            )
        variant_id = _identifier(item.get("variant_id"), label=f"{label}.variant_id")
        if variant_id in seen_variants:
            raise _fail("TEMPLATE_SCHEMA_INVALID", "capability variant ids must be unique")
        seen_variants.add(variant_id)
        status = item.get("runtime_status")
        if status not in _RUNTIME_STATUSES:
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label}.runtime_status is unsupported")
        plugin_id = item.get("plugin_id")
        if status == "ACTIVE":
            if not isinstance(plugin_id, str) or plugin_id not in active_plugins:
                raise _fail("TEMPLATE_PLUGIN_INVALID", f"{label}.plugin_id is not an active Harness plugin")
            if family not in active_plugins[plugin_id].families:
                raise _fail(
                    "TEMPLATE_PLUGIN_INVALID",
                    f"{label}.plugin_id does not support joint family {family!r}",
                )
            expected_capabilities = sorted(
                active_plugins[plugin_id].implementation_capability_ids
            )
            expected_boundaries = sorted(
                active_plugins[plugin_id].model_boundary_ids
            )
            if expected_capabilities or expected_boundaries:
                actual_capabilities = _string_list(
                    item.get("capability_ids"),
                    label=f"{label}.capability_ids",
                    allow_empty=True,
                )
                actual_boundaries = _string_list(
                    item.get("model_boundary_ids"),
                    label=f"{label}.model_boundary_ids",
                    allow_empty=True,
                )
                if (
                    actual_capabilities != expected_capabilities
                    or actual_boundaries != expected_boundaries
                ):
                    raise _fail(
                        "TEMPLATE_PLUGIN_CAPABILITY_MISMATCH",
                        f"{label} capability contract differs from active plugin {plugin_id!r}",
                    )
            elif contract_fields & frozenset(item):
                if item.get("capability_ids") or item.get("model_boundary_ids"):
                    raise _fail(
                        "TEMPLATE_PLUGIN_CAPABILITY_MISMATCH",
                        f"{label} claims capabilities absent from active plugin {plugin_id!r}",
                    )
        elif plugin_id is not None:
            _text(plugin_id, label=f"{label}.plugin_id", limit=128)
        mechanism_kind = item.get("mechanism_kind")
        if mechanism_kind is not None:
            _text(mechanism_kind, label=f"{label}.mechanism_kind", limit=128)
    wrench_kinds = _string_list(
        payload.get("input_wrench_kinds"),
        label="joint input_wrench_kinds",
    )
    if not set(wrench_kinds).issubset(_WRENCH_KINDS):
        raise _fail("TEMPLATE_SCHEMA_INVALID", "joint input_wrench_kinds contains an unsupported value")
    _string_list(payload.get("observable_quantities"), label="joint observable_quantities")
    _string_list(payload.get("components"), label="joint components", allow_empty=True)
    _string_list(payload.get("required_instance_fields"), label="joint required_instance_fields")
    sources = payload.get("source_bindings")
    if not isinstance(sources, list) or not sources or len(sources) > 32:
        raise _fail("TEMPLATE_SCHEMA_INVALID", "joint source_bindings must be a non-empty bounded list")
    for index, item in enumerate(sources):
        _validate_source_binding(
            item,
            label=f"source_bindings[{index}]",
            repository_root=repository_root,
            workspace_root=workspace_root,
            forbid_project_adapter=forbid_project_adapter,
        )
    _string_list(payload.get("constraints"), label="joint constraints")
    _validate_approval(
        payload.get("approval"),
        label="joint approval",
        required_status=approval_status,
    )


def _validate_task_suite_template(
    payload: dict[str, Any],
    *,
    approval_status: str = "APPROVED",
    installed_programs: frozenset[str] | None = None,
    installed_predicates: frozenset[str] | None = None,
) -> None:
    _closed(
        payload,
        required=frozenset(
            {
                "schema_version",
                "kind",
                "template_id",
                "version",
                "title",
                "machine_family",
                "goal",
                "joint_template_refs",
                "profiles",
                "scenarios",
                "fixtures",
                "tasks",
                "source_projects",
                "approval",
            }
        ),
        label="machine task suite",
    )
    if (
        payload.get("schema_version") != TASK_SUITE_TEMPLATE_SCHEMA
        or payload.get("kind") != "task_suite"
    ):
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "machine task suite identity fields are invalid",
        )
    _identifier(payload.get("template_id"), label="task suite template_id")
    _version(payload.get("version"), label="task suite version")
    _text(payload.get("title"), label="task suite title", limit=256)
    _identifier(payload.get("machine_family"), label="task suite machine_family")
    _text(payload.get("goal"), label="task suite goal")

    refs = payload.get("joint_template_refs")
    if not isinstance(refs, list) or not refs or len(refs) > 32:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "task suite joint_template_refs must be non-empty and bounded",
        )
    seen_refs: set[tuple[str, str, str]] = set()
    for index, ref in enumerate(refs):
        label = f"task suite joint_template_refs[{index}]"
        if not isinstance(ref, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        _closed(
            ref,
            required=frozenset(
                {"template_id", "version", "variant_id", "sha256"}
            ),
            label=label,
        )
        identity = (
            _identifier(ref.get("template_id"), label=f"{label}.template_id"),
            _version(ref.get("version"), label=f"{label}.version"),
            _identifier(ref.get("variant_id"), label=f"{label}.variant_id"),
        )
        _hex(ref.get("sha256"), label=f"{label}.sha256")
        if identity in seen_refs:
            raise _fail(
                "TEMPLATE_REFERENCE_INVALID",
                "task suite joint_template_refs must be unique",
            )
        seen_refs.add(identity)

    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles or len(profiles) > 32:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "task suite profiles must be non-empty and bounded",
        )
    profile_ids: set[str] = set()
    for index, profile in enumerate(profiles):
        label = f"task suite profiles[{index}]"
        if not isinstance(profile, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        _closed(
            profile,
            required=frozenset({"profile_id", "title", "description"}),
            label=label,
        )
        profile_id = _identifier(
            profile.get("profile_id"), label=f"{label}.profile_id"
        )
        if profile_id in profile_ids:
            raise _fail("TEMPLATE_SCHEMA_INVALID", "task suite profile ids must be unique")
        profile_ids.add(profile_id)
        _text(profile.get("title"), label=f"{label}.title", limit=256)
        _text(profile.get("description"), label=f"{label}.description")

    fixtures = payload.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) > 32:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "task suite fixtures must be a bounded list",
        )
    fixture_ids: set[str] = set()
    for index, fixture in enumerate(fixtures):
        label = f"task suite fixtures[{index}]"
        if not isinstance(fixture, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        _closed(
            fixture,
            required=frozenset({"fixture_id", "title", "parameters"}),
            label=label,
        )
        fixture_id = _identifier(
            fixture.get("fixture_id"), label=f"{label}.fixture_id"
        )
        if fixture_id in fixture_ids:
            raise _fail("TEMPLATE_SCHEMA_INVALID", "task suite fixture ids must be unique")
        fixture_ids.add(fixture_id)
        _text(fixture.get("title"), label=f"{label}.title", limit=256)
        parameters = fixture.get("parameters")
        if not isinstance(parameters, list) or not parameters or len(parameters) > 64:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.parameters must be non-empty and bounded",
            )
        parameter_ids: set[str] = set()
        for parameter_index, parameter in enumerate(parameters):
            parameter_label = f"{label}.parameters[{parameter_index}]"
            if not isinstance(parameter, dict):
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{parameter_label} must be an object",
                )
            _closed(
                parameter,
                required=frozenset(
                    {"parameter_id", "value", "unit", "description"}
                ),
                label=parameter_label,
            )
            parameter_id = _identifier(
                parameter.get("parameter_id"),
                label=f"{parameter_label}.parameter_id",
            )
            if parameter_id in parameter_ids:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{label}.parameter ids must be unique",
                )
            parameter_ids.add(parameter_id)
            _finite_json(parameter.get("value"), label=f"{parameter_label}.value")
            _text(parameter.get("unit"), label=f"{parameter_label}.unit", limit=64)
            _text(
                parameter.get("description"),
                label=f"{parameter_label}.description",
            )

    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or not tasks or len(tasks) > 128:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "task suite tasks must be non-empty and bounded",
        )
    task_ids: set[str] = set()
    task_scenarios: dict[str, str] = {}
    task_profiles: dict[str, set[str]] = {}
    for index, task in enumerate(tasks):
        label = f"task suite tasks[{index}]"
        if not isinstance(task, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        _closed(
            task,
            required=frozenset(
                {
                    "task_id",
                    "title",
                    "goal",
                    "scenario_id",
                    "profile_ids",
                    "runtime_status",
                    "criteria",
                    "failure_diagnostics",
                    "evidence_obligations",
                }
            ),
            label=label,
        )
        task_id = _identifier(task.get("task_id"), label=f"{label}.task_id")
        if task_id in task_ids:
            raise _fail("TEMPLATE_SCHEMA_INVALID", "task suite task ids must be unique")
        task_ids.add(task_id)
        _text(task.get("title"), label=f"{label}.title", limit=256)
        _text(task.get("goal"), label=f"{label}.goal")
        scenario_id = _identifier(
            task.get("scenario_id"), label=f"{label}.scenario_id"
        )
        task_scenarios[task_id] = scenario_id
        selected_profiles = set(
            _string_list(task.get("profile_ids"), label=f"{label}.profile_ids")
        )
        if not selected_profiles.issubset(profile_ids):
            raise _fail(
                "TEMPLATE_REFERENCE_INVALID",
                f"{label}.profile_ids contains an unknown profile",
            )
        task_profiles[task_id] = selected_profiles
        if task.get("runtime_status") not in _RUNTIME_STATUSES:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.runtime_status is unsupported",
            )
        criteria = task.get("criteria")
        if not isinstance(criteria, list) or not criteria or len(criteria) > 64:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.criteria must be non-empty and bounded",
            )
        criterion_ids: set[str] = set()
        for criterion_index, criterion in enumerate(criteria):
            criterion_label = f"{label}.criteria[{criterion_index}]"
            if not isinstance(criterion, dict):
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{criterion_label} must be an object",
                )
            _closed(
                criterion,
                required=frozenset(
                    {
                        "criterion_id",
                        "description",
                        "disposition",
                        "evaluator_kind",
                        "evaluator_id",
                        "observable_quantity",
                        "comparator",
                        "expected",
                        "unit",
                        "tolerance",
                    }
                ),
                label=criterion_label,
            )
            criterion_id = _identifier(
                criterion.get("criterion_id"),
                label=f"{criterion_label}.criterion_id",
            )
            if criterion_id in criterion_ids:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{label}.criterion ids must be unique",
                )
            criterion_ids.add(criterion_id)
            _text(
                criterion.get("description"),
                label=f"{criterion_label}.description",
            )
            if criterion.get("disposition") not in _CRITERION_DISPOSITIONS:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{criterion_label}.disposition is unsupported",
                )
            evaluator_kind = criterion.get("evaluator_kind")
            if evaluator_kind not in _EVALUATOR_KINDS:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{criterion_label}.evaluator_kind is unsupported",
                )
            evaluator_id = _identifier(
                criterion.get("evaluator_id"),
                label=f"{criterion_label}.evaluator_id",
            )
            observable = _text(
                criterion.get("observable_quantity"),
                label=f"{criterion_label}.observable_quantity",
                limit=128,
            )
            comparator = criterion.get("comparator")
            if comparator not in _COMPARATORS:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{criterion_label}.comparator is unsupported",
                )
            _finite_json(
                criterion.get("expected"),
                label=f"{criterion_label}.expected",
            )
            unit = _text(
                criterion.get("unit"), label=f"{criterion_label}.unit", limit=64
            )
            if evaluator_kind == "registered_metric":
                definition = METRIC_REGISTRY.get(evaluator_id)
                if definition is None:
                    if task.get("runtime_status") != "IMPLEMENTATION_REQUIRED":
                        raise _fail(
                            "TEMPLATE_METRIC_INVALID",
                            f"{criterion_label}.evaluator_id is not registered",
                        )
                else:
                    if observable not in definition.compatible_quantities:
                        raise _fail(
                            "TEMPLATE_METRIC_INVALID",
                            f"{criterion_label} uses an incompatible observable quantity",
                        )
                    if comparator not in definition.comparators:
                        raise _fail(
                            "TEMPLATE_METRIC_INVALID",
                            f"{criterion_label} uses an incompatible comparator",
                        )
                    if unit != definition.unit:
                        raise _fail(
                            "TEMPLATE_METRIC_INVALID",
                            f"{criterion_label}.unit differs from the metric registry",
                        )
            elif (
                evaluator_kind == "suite_predicate"
                and installed_predicates is not None
                and task.get("runtime_status") == "ACTIVE"
                and evaluator_id not in installed_predicates
            ):
                raise _fail(
                    "TEMPLATE_METRIC_INVALID",
                    f"{criterion_label}.evaluator_id is not an installed "
                    "suite predicate; mark the task IMPLEMENTATION_REQUIRED",
                )
            tolerance = criterion.get("tolerance")
            if not isinstance(tolerance, dict):
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{criterion_label}.tolerance must be an object",
                )
            _closed(
                tolerance,
                required=frozenset({"mode", "value", "unit"}),
                label=f"{criterion_label}.tolerance",
            )
            mode = tolerance.get("mode")
            if mode not in _TOLERANCE_MODES:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{criterion_label}.tolerance.mode is unsupported",
                )
            tolerance_value = tolerance.get("value")
            tolerance_unit = tolerance.get("unit")
            if mode == "none":
                if tolerance_value is not None or tolerance_unit is not None:
                    raise _fail(
                        "TEMPLATE_SCHEMA_INVALID",
                        f"{criterion_label}.tolerance none must use null value/unit",
                    )
            else:
                if (
                    isinstance(tolerance_value, bool)
                    or not isinstance(tolerance_value, (int, float))
                    or not math.isfinite(float(tolerance_value))
                    or float(tolerance_value) < 0.0
                ):
                    raise _fail(
                        "TEMPLATE_SCHEMA_INVALID",
                        f"{criterion_label}.tolerance.value must be finite and non-negative",
                    )
                tolerance_unit_text = _text(
                    tolerance_unit,
                    label=f"{criterion_label}.tolerance.unit",
                    limit=64,
                )
                if mode == "relative_percent" and tolerance_unit_text != "percent":
                    raise _fail(
                        "TEMPLATE_SCHEMA_INVALID",
                        f"{criterion_label}.relative_percent tolerance must use percent",
                    )
                if mode != "relative_percent" and tolerance_unit_text != unit:
                    raise _fail(
                        "TEMPLATE_SCHEMA_INVALID",
                        f"{criterion_label}.absolute tolerance unit must match criterion unit",
                    )
        diagnostics = task.get("failure_diagnostics")
        if not isinstance(diagnostics, list) or len(diagnostics) > 32:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.failure_diagnostics must be a bounded list",
            )
        diagnostic_codes: set[str] = set()
        for diagnostic_index, diagnostic in enumerate(diagnostics):
            diagnostic_label = f"{label}.failure_diagnostics[{diagnostic_index}]"
            if not isinstance(diagnostic, dict):
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{diagnostic_label} must be an object",
                )
            _closed(
                diagnostic,
                required=frozenset({"code", "description"}),
                label=diagnostic_label,
            )
            code = _diagnostic_code(
                diagnostic.get("code"), label=f"{diagnostic_label}.code"
            )
            if code in diagnostic_codes:
                raise _fail(
                    "TEMPLATE_SCHEMA_INVALID",
                    f"{label}.diagnostic codes must be unique",
                )
            diagnostic_codes.add(code)
            _text(
                diagnostic.get("description"),
                label=f"{diagnostic_label}.description",
            )
        _string_list(
            task.get("evidence_obligations"),
            label=f"{label}.evidence_obligations",
        )

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or len(scenarios) > 64:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "task suite scenarios must be non-empty and bounded",
        )
    scenario_ids: set[str] = set()
    scenario_task_sets: dict[str, set[str]] = {}
    for index, scenario in enumerate(scenarios):
        label = f"task suite scenarios[{index}]"
        if not isinstance(scenario, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        _closed(
            scenario,
            required=frozenset(
                {
                    "scenario_id",
                    "title",
                    "test_program",
                    "test_load_case_id",
                    "profile_ids",
                    "required_branches",
                    "runtime_status",
                    "task_ids",
                    "execution_requirements",
                }
            ),
            label=label,
        )
        scenario_id = _identifier(
            scenario.get("scenario_id"), label=f"{label}.scenario_id"
        )
        if scenario_id in scenario_ids:
            raise _fail("TEMPLATE_SCHEMA_INVALID", "task suite scenario ids must be unique")
        scenario_ids.add(scenario_id)
        _text(scenario.get("title"), label=f"{label}.title", limit=256)
        test_program = _text(
            scenario.get("test_program"), label=f"{label}.test_program", limit=128
        )
        if installed_programs is not None:
            installed = test_program in installed_programs
            status = scenario.get("runtime_status")
            if status == "ACTIVE" and not installed:
                raise _fail(
                    "TEMPLATE_PROGRAM_STATUS_INVALID",
                    f"{label}.test_program is not an installed program; mark "
                    "the scenario IMPLEMENTATION_REQUIRED",
                )
            if status == "IMPLEMENTATION_REQUIRED" and installed:
                raise _fail(
                    "TEMPLATE_PROGRAM_STATUS_INVALID",
                    f"{label}.test_program is already installed; mark the "
                    "scenario ACTIVE",
                )
        _text(
            scenario.get("test_load_case_id"),
            label=f"{label}.test_load_case_id",
            limit=128,
        )
        selected_profiles = set(
            _string_list(
                scenario.get("profile_ids"), label=f"{label}.profile_ids"
            )
        )
        if not selected_profiles.issubset(profile_ids):
            raise _fail(
                "TEMPLATE_REFERENCE_INVALID",
                f"{label}.profile_ids contains an unknown profile",
            )
        branches = _string_list(
            scenario.get("required_branches"),
            label=f"{label}.required_branches",
        )
        if not set(branches).issubset(BRANCHES):
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.required_branches contains an unsupported branch",
            )
        if scenario.get("runtime_status") not in _RUNTIME_STATUSES:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.runtime_status is unsupported",
            )
        selected_tasks = set(
            _string_list(scenario.get("task_ids"), label=f"{label}.task_ids")
        )
        if not selected_tasks.issubset(task_ids):
            raise _fail(
                "TEMPLATE_REFERENCE_INVALID",
                f"{label}.task_ids contains an unknown task",
            )
        for task_id in selected_tasks:
            if not task_profiles[task_id].issubset(selected_profiles):
                raise _fail(
                    "TEMPLATE_REFERENCE_INVALID",
                    f"{label}.profile_ids do not cover task {task_id!r}",
                )
        scenario_task_sets[scenario_id] = selected_tasks
        _string_list(
            scenario.get("execution_requirements"),
            label=f"{label}.execution_requirements",
        )
    if set(task_scenarios.values()) != scenario_ids:
        raise _fail(
            "TEMPLATE_REFERENCE_INVALID",
            "every scenario must own at least one task and every task must select a scenario",
        )
    for scenario_id, selected_tasks in scenario_task_sets.items():
        expected_tasks = {
            task_id
            for task_id, selected_scenario in task_scenarios.items()
            if selected_scenario == scenario_id
        }
        if selected_tasks != expected_tasks:
            raise _fail(
                "TEMPLATE_REFERENCE_INVALID",
                f"scenario {scenario_id!r} task_ids differ from task scenario bindings",
            )

    sources = payload.get("source_projects")
    if not isinstance(sources, list) or not sources or len(sources) > 32:
        raise _fail(
            "TEMPLATE_SCHEMA_INVALID",
            "task suite source_projects must be non-empty and bounded",
        )
    source_ids: set[str] = set()
    for index, source in enumerate(sources):
        label = f"task suite source_projects[{index}]"
        if not isinstance(source, dict):
            raise _fail("TEMPLATE_SCHEMA_INVALID", f"{label} must be an object")
        _closed(
            source,
            required=frozenset({"project_id", "evidence_status", "note"}),
            label=label,
        )
        project_id = _identifier(
            source.get("project_id"), label=f"{label}.project_id"
        )
        if project_id in source_ids:
            raise _fail("TEMPLATE_SCHEMA_INVALID", "source project ids must be unique")
        source_ids.add(project_id)
        if source.get("evidence_status") not in _EVIDENCE_STATUSES:
            raise _fail(
                "TEMPLATE_SCHEMA_INVALID",
                f"{label}.evidence_status is unsupported",
            )
        _text(source.get("note"), label=f"{label}.note")
    _validate_approval(
        payload.get("approval"),
        label="task suite approval",
        required_status=approval_status,
    )


def _actual_template_files(root: Path) -> set[str]:
    actual: set[str] = set()
    for directory_name in ("joints", "task_suites"):
        directory = root / directory_name
        if directory.is_symlink() or not directory.is_dir():
            raise _fail("TEMPLATE_PATH_UNSAFE", f"template directory is missing or unsafe: {directory}")
        for node in directory.rglob("*"):
            if node.is_symlink():
                raise _fail("TEMPLATE_PATH_UNSAFE", f"template library contains a symlink: {node}")
            if node.is_dir():
                continue
            if not node.is_file() or node.suffix != ".json":
                raise _fail("TEMPLATE_PATH_UNSAFE", f"template library contains an unexpected node: {node}")
            actual.add(node.relative_to(root).as_posix())
    return actual


def _validate_catalog_at(
    root: Path,
    *,
    repository_root: Path | None = None,
) -> TemplateCatalog:
    """Internal test/maintenance hook; public loading always uses fixed roots."""

    root = Path(root).resolve()
    repository_root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root).resolve()
    index_path = root / _INDEX_NAME
    index = _read_json(index_path, max_bytes=_MAX_INDEX_BYTES, label="template catalog")
    _closed(
        index,
        required=frozenset(
            {"schema_version", "catalog_id", "catalog_version", "entries", "catalog_sha256"}
        ),
        label="template catalog",
    )
    if index.get("schema_version") != CATALOG_SCHEMA:
        raise _fail("TEMPLATE_CATALOG_INVALID", "template catalog schema is unsupported")
    catalog_id = _identifier(index.get("catalog_id"), label="catalog_id")
    catalog_version = _version(index.get("catalog_version"), label="catalog_version")
    expected_catalog_hash = _hex(index.get("catalog_sha256"), label="catalog_sha256")
    core = dict(index)
    core.pop("catalog_sha256")
    if _canonical_sha256(core) != expected_catalog_hash:
        raise _fail("TEMPLATE_CATALOG_CHANGED", "template catalog does not match catalog_sha256")
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries or len(entries) > 512:
        raise _fail("TEMPLATE_CATALOG_INVALID", "template catalog entries must be non-empty and bounded")
    seen_identity: set[tuple[str, str, str]] = set()
    seen_paths: set[str] = set()
    records: list[TemplateRecord] = []
    for index_number, entry in enumerate(entries):
        label = f"entries[{index_number}]"
        if not isinstance(entry, dict):
            raise _fail("TEMPLATE_CATALOG_INVALID", f"{label} must be an object")
        _closed(entry, required=frozenset({"kind", "template_id", "version", "path", "sha256"}), label=label)
        kind = entry.get("kind")
        if kind not in _KINDS:
            raise _fail("TEMPLATE_CATALOG_INVALID", f"{label}.kind is unsupported")
        template_id = _identifier(entry.get("template_id"), label=f"{label}.template_id")
        version = _version(entry.get("version"), label=f"{label}.version")
        relative = _text(entry.get("path"), label=f"{label}.path", limit=1_024)
        expected_relative = f"{kind}s/{template_id}/{version}.json"
        if relative != expected_relative:
            raise _fail("TEMPLATE_PATH_INVALID", f"{label}.path must be exactly {expected_relative}")
        expected_hash = _hex(entry.get("sha256"), label=f"{label}.sha256")
        identity = (kind, template_id, version)
        if identity in seen_identity or relative in seen_paths:
            raise _fail("TEMPLATE_CATALOG_DUPLICATE", f"{label} duplicates an identity or path")
        seen_identity.add(identity)
        seen_paths.add(relative)
        path = _safe_catalog_path(root, relative, label=f"{label}.path")
        payload = _read_json(path, max_bytes=_MAX_TEMPLATE_BYTES, label=f"{kind} template")
        if _sha256_file(path) != expected_hash:
            raise _fail("TEMPLATE_FILE_CHANGED", f"{relative} does not match its catalog hash")
        if payload.get("template_id") != template_id or payload.get("version") != version or payload.get("kind") != kind:
            raise _fail("TEMPLATE_IDENTITY_MISMATCH", f"{relative} identity differs from the catalog")
        if kind == "joint":
            _validate_joint_template(payload, repository_root=repository_root)
        else:
            _validate_task_suite_template(payload)
        records.append(
            TemplateRecord(
                kind=kind,
                template_id=template_id,
                version=version,
                path=relative,
                sha256=expected_hash,
                _payload=payload,
            )
        )
    if seen_paths != _actual_template_files(root):
        missing = sorted(_actual_template_files(root) - seen_paths)
        stale = sorted(seen_paths - _actual_template_files(root))
        raise _fail(
            "TEMPLATE_CATALOG_INCOMPLETE",
            f"catalog and template files differ; unindexed={missing}, missing={stale}",
        )
    by_identity = {(item.kind, item.template_id, item.version): item for item in records}
    for record in records:
        if record.kind != "task_suite":
            continue
        for ref in record._payload["joint_template_refs"]:
            target = by_identity.get(("joint", ref["template_id"], ref["version"]))
            if target is None:
                raise _fail("TEMPLATE_REFERENCE_INVALID", f"{record.template_id} has a dangling joint ref")
            if target.sha256 != ref["sha256"]:
                raise _fail("TEMPLATE_REFERENCE_CHANGED", f"{record.template_id} joint ref hash differs")
            variants = {
                item["variant_id"]: item
                for item in target._payload["capability_variants"]
            }
            variant = variants.get(ref["variant_id"])
            if variant is None:
                raise _fail(
                    "TEMPLATE_REFERENCE_INVALID",
                    f"{record.template_id} selects an unknown joint variant",
                )
    records.sort(key=lambda item: (item.kind, item.template_id, item.version))
    return TemplateCatalog(
        catalog_id=catalog_id,
        catalog_version=catalog_version,
        catalog_sha256=expected_catalog_hash,
        records=tuple(records),
    )


def load_builtin_template_catalog() -> TemplateCatalog:
    """Load and verify the Harness-owned catalog from its fixed code root."""

    return _validate_catalog_at(_BUILTIN_ROOT, repository_root=_REPOSITORY_ROOT)


def list_builtin_templates(kind: str) -> tuple[TemplateRecord, ...]:
    if kind not in _KINDS:
        raise PlanningTemplateError("TEMPLATE_KIND_INVALID", f"unsupported template kind {kind!r}")
    return load_builtin_template_catalog().records_for(kind)


def resolve_builtin_template(
    reference: Mapping[str, Any],
    *,
    expected_kind: str,
) -> TemplateRecord:
    if expected_kind not in _KINDS:
        raise PlanningTemplateError("TEMPLATE_KIND_INVALID", f"unsupported template kind {expected_kind!r}")
    if not isinstance(reference, Mapping):
        raise PlanningTemplateError("TEMPLATE_REFERENCE_INVALID", "template reference must be an object")
    _closed(reference, required=frozenset({"template_id", "version", "sha256"}), label="template reference")
    template_id = _identifier(reference.get("template_id"), label="template reference id")
    version = _version(reference.get("version"), label="template reference version")
    expected_hash = _hex(reference.get("sha256"), label="template reference sha256")
    try:
        record = load_builtin_template_catalog().resolve(expected_kind, template_id, version)
    except KeyError as exc:
        raise PlanningTemplateError("TEMPLATE_REFERENCE_INVALID", str(exc)) from exc
    if record.sha256 != expected_hash:
        raise PlanningTemplateError("TEMPLATE_REFERENCE_CHANGED", "template reference hash differs")
    return record


def public_template_catalog_projection() -> dict[str, Any]:
    catalog = load_builtin_template_catalog()
    entries: list[dict[str, Any]] = []
    for record in catalog.records:
        payload = record._payload
        entry = {
            "kind": record.kind,
            "template_id": record.template_id,
            "version": record.version,
            "sha256": record.sha256,
            "title": payload["title"],
            "goal": payload["goal"],
        }
        if record.kind == "joint":
            entry.update(
                {
                    "category": payload["category"],
                    "family": payload["family"],
                    "capability_variants": payload["capability_variants"],
                    "input_wrench_kinds": payload["input_wrench_kinds"],
                    "observable_quantities": payload["observable_quantities"],
                    "components": payload["components"],
                    "required_instance_fields": payload["required_instance_fields"],
                    "constraints": payload["constraints"],
                }
            )
        else:
            entry.update(
                {
                    "machine_family": payload["machine_family"],
                    "joint_template_refs": payload["joint_template_refs"],
                    "profiles": payload["profiles"],
                    "scenarios": payload["scenarios"],
                    "fixtures": payload["fixtures"],
                    "tasks": payload["tasks"],
                }
            )
        entries.append(entry)
    return {
        "schema_version": "simulation_ready_cad_public_template_catalog_v1",
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "catalog_sha256": catalog.catalog_sha256,
        "entries": entries,
    }
