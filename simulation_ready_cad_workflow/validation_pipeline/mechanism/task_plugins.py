"""AI-reviewer TaskAdapter plugins for new programs, windows, and evaluators.

Task suites often reuse an already-bound joint. New acceptance methods do not
belong on MechanismAdapter. A reviewer TaskAdapter claims the approved
IMPLEMENTATION_REQUIRED programs and evaluator ids, then supplies any missing
load cases, the exact task-window policy, and unregistered metric evaluation.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from abc import ABC
from pathlib import Path
from typing import Any, Mapping

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from .io import sha256_file
from .models import LoadCase, MechanismPlan, MechanismValidationError
from .plugin_boundary import plugin_boundary_error
from .task_motion_evidence import TaskProgramEvidenceSpec


class TaskAdapter(ABC):
    """Trusted task-program plugin. Export as ``TASK_ADAPTER``."""

    plugin_id: str
    programs: frozenset[str] = frozenset()
    evaluator_ids: frozenset[str] = frozenset()

    def supports(self, plan: MechanismPlan) -> bool:
        """Return whether this task plugin owns work for the exact plan."""

        return False

    def load_cases(self, plan: MechanismPlan) -> tuple[LoadCase, ...]:
        """Emit load cases the joint adapter does not already provide."""

        return ()

    def program_policy(self, load_case: LoadCase) -> TaskProgramEvidenceSpec:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_POLICY_MISSING",
            f"task adapter {self.plugin_id!r} has no window policy for "
            f"{load_case.program!r}",
        )

    def evaluate(
        self,
        requirement: Mapping[str, Any],
        binding: Mapping[str, Any],
        trace: Mapping[str, Any],
        *,
        canonical_generalized_observable_id: str | None = None,
        validated_program: Any = None,
        resolved_channels: Mapping[str, Any] | None = None,
        verified_wrench_audit: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metric_id = requirement.get("metric_id")
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_EVALUATOR_MISSING",
            f"task adapter {self.plugin_id!r} does not evaluate {metric_id!r}",
        )


_REVIEWER_TASK_ADAPTERS: dict[str, TaskAdapter] = {}
_REVIEWER_TASK_ADAPTER_CACHE: dict[tuple[str, str], TaskAdapter] = {}
_REVIEWER_TASK_ADAPTER_SOURCES: dict[str, dict[str, str]] = {}


def _reviewer_task_artifact_failure(
    exc: MechanismValidationError,
    *,
    plugin_id: str | None = None,
) -> MechanismValidationError:
    """Attach the reviewer-owned delivery boundary before the plugin is bound."""

    if exc.failure_envelope is not None:
        return exc
    return MechanismValidationError(
        exc.code,
        exc.message,
        details=exc.details,
        failure=make_failure_envelope(
            exc.code,
            FailureOwner.REVIEWER_TASK_PLUGIN,
            source_component="mechanism.task_plugins.reviewer_artifact",
            message=exc.message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            plugin_id=plugin_id,
            plugin_kind="task",
            registration_source="reviewer_task_adapter",
            details=exc.details,
        ),
    )


def _reviewer_task_registry_failure(
    code: str,
    message: str,
) -> MechanismValidationError:
    """The Loop writes task-plugin registry rows, so malformed rows are Harness-owned."""

    return MechanismValidationError(
        code,
        message,
        failure=make_failure_envelope(
            code,
            FailureOwner.HARNESS_CODE,
            source_component="mechanism.task_plugins.reviewer_registry",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        ),
    )


def reviewer_task_adapter_source(plugin_id: str) -> dict[str, str] | None:
    source = _REVIEWER_TASK_ADAPTER_SOURCES.get(plugin_id)
    return None if source is None else dict(source)


def clear_reviewer_task_adapters() -> None:
    """Drop in-process reviewer task adapters. Tests reload a clean registry."""

    _REVIEWER_TASK_ADAPTERS.clear()
    _REVIEWER_TASK_ADAPTER_CACHE.clear()
    _REVIEWER_TASK_ADAPTER_SOURCES.clear()


def drop_reviewer_task_adapter(plugin_id: str) -> None:
    """Unbind one in-process reviewer task adapter so a revision can replace it."""

    _REVIEWER_TASK_ADAPTERS.pop(plugin_id, None)
    _REVIEWER_TASK_ADAPTER_SOURCES.pop(plugin_id, None)
    stale = [
        key
        for key, plugin in _REVIEWER_TASK_ADAPTER_CACHE.items()
        if plugin.plugin_id == plugin_id
    ]
    for key in stale:
        _REVIEWER_TASK_ADAPTER_CACHE.pop(key, None)


def registered_reviewer_task_adapters() -> tuple[TaskAdapter, ...]:
    return tuple(_REVIEWER_TASK_ADAPTERS.values())


def _reviewer_task_callback_failure(
    plugin: TaskAdapter,
    callback: str,
    exc: BaseException,
    *,
    code: str | None = None,
    message: str | None = None,
    details: Mapping[str, Any] | None = None,
    contract_violation: bool = False,
) -> MechanismValidationError:
    return plugin_boundary_error(
        exc,
        registration={
            "plugin_id": plugin.plugin_id,
            "registration_source": "reviewer_task_adapter",
        },
        plugin_id=plugin.plugin_id,
        plugin_kind="task",
        callback=callback,
        source_component=f"mechanism.task_plugins.{callback}",
        code=code,
        message=message,
        details=details,
        contract_violation=contract_violation,
    )


def _supports_plan(plugin: TaskAdapter, plan: MechanismPlan) -> bool:
    try:
        supported = plugin.supports(plan)
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise _reviewer_task_callback_failure(
            plugin,
            "supports",
            exc,
        ) from exc
    except Exception as exc:
        raise _reviewer_task_callback_failure(
            plugin,
            "supports",
            exc,
            code="REVIEWER_TASK_ADAPTER_SUPPORTS_FAILED",
            message=f"task adapter {plugin.plugin_id!r} supports() failed: {exc}",
        ) from exc
    if not isinstance(supported, bool):
        error = MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_SUPPORTS_INVALID",
            "task adapter supports() must return bool",
        )
        raise _reviewer_task_callback_failure(
            plugin,
            "supports",
            error,
            contract_violation=True,
        )
    return supported


def claimed_reviewer_task_programs(
    plan: MechanismPlan | None = None,
) -> frozenset[str]:
    claimed: set[str] = set()
    for plugin in _REVIEWER_TASK_ADAPTERS.values():
        if plan is not None and not _supports_plan(plugin, plan):
            continue
        claimed.update(plugin.programs)
    return frozenset(claimed)


def _identifier_set(value: Any, *, label: str, code: str) -> frozenset[str]:
    if not isinstance(value, frozenset) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise MechanismValidationError(
            code,
            f"reviewer task adapter {label} must be a frozenset of strings",
        )
    return value


def _owner_of_program(program: str) -> str | None:
    for plugin in _REVIEWER_TASK_ADAPTERS.values():
        if program in plugin.programs:
            return plugin.plugin_id
    return None


def _owner_of_evaluator(evaluator_id: str) -> str | None:
    for plugin in _REVIEWER_TASK_ADAPTERS.values():
        if evaluator_id in plugin.evaluator_ids:
            return plugin.plugin_id
    return None


def task_adapter_for_program(
    program: str,
    *,
    plan: MechanismPlan | None = None,
) -> TaskAdapter | None:
    owner = _owner_of_program(program)
    if owner is None:
        return None
    plugin = _REVIEWER_TASK_ADAPTERS.get(owner)
    if plugin is None or (plan is not None and not _supports_plan(plugin, plan)):
        return None
    return plugin


def task_adapter_for_evaluator(
    evaluator_id: str,
    *,
    plan: MechanismPlan | None = None,
) -> TaskAdapter | None:
    owner = _owner_of_evaluator(evaluator_id)
    if owner is None:
        return None
    plugin = _REVIEWER_TASK_ADAPTERS.get(owner)
    if plugin is None or (plan is not None and not _supports_plan(plugin, plan)):
        return None
    return plugin


def _load_reviewer_task_adapter(
    source: str | Path,
    expected_sha256: str,
    *,
    bind: bool = True,
    replace_plugin_id: str | None = None,
) -> TaskAdapter:
    """Import one AI-reviewer TaskAdapter from its hash-bound file."""

    path = Path(source).resolve()
    if not str(path).endswith(".py") or path.is_symlink() or not path.is_file():
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_PATH_INVALID",
            "reviewer task adapter must be a regular Python file",
        )
    if sha256_file(path) != expected_sha256:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_HASH_MISMATCH",
            "reviewer task adapter artifact does not match its approved hash",
        )
    cache_key = (str(path), expected_sha256)
    if bind:
        cached = _REVIEWER_TASK_ADAPTER_CACHE.get(cache_key)
        if cached is not None:
            return cached
    module_name = (
        "_simulation_ready_reviewer_task_adapter_"
        f"{expected_sha256}_{hashlib.sha256(str(path).encode('utf-8')).hexdigest()}"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_LOAD_FAILED",
            f"cannot load reviewer task adapter artifact: {path.name}",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        if (
            isinstance(exc, MechanismValidationError)
            and exc.failure_envelope is not None
        ):
            raise
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_LOAD_FAILED", str(exc)
        ) from exc
    plugin = getattr(module, "TASK_ADAPTER", None)
    if not isinstance(plugin, TaskAdapter):
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_EXPORT_INVALID",
            "reviewer task adapter file must export TASK_ADAPTER",
        )
    if not plugin.plugin_id.startswith("reviewer."):
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_ID_INVALID",
            "reviewer task adapter plugin_id must start with 'reviewer.'",
        )
    programs = _identifier_set(
        plugin.programs,
        label="programs",
        code="REVIEWER_TASK_ADAPTER_PROGRAMS_INVALID",
    )
    evaluators = _identifier_set(
        plugin.evaluator_ids,
        label="evaluator_ids",
        code="REVIEWER_TASK_ADAPTER_EVALUATORS_INVALID",
    )
    if not programs and not evaluators:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_CLAIMS_EMPTY",
            "reviewer task adapter must claim at least one program or evaluator_id",
        )
    if type(plugin).supports is TaskAdapter.supports:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_SUPPORTS_REQUIRED",
            "reviewer task adapter must implement supports(plan)",
        )
    if programs and type(plugin).program_policy is TaskAdapter.program_policy:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_POLICY_MISSING",
            "reviewer task adapter must implement program_policy() for claimed programs",
        )
    if evaluators and type(plugin).evaluate is TaskAdapter.evaluate:
        raise MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_EVALUATOR_MISSING",
            "reviewer task adapter must implement evaluate() for claimed evaluator_ids",
        )
    for program in programs:
        owner = _owner_of_program(program)
        if (
            owner is not None
            and owner != plugin.plugin_id
            and owner != replace_plugin_id
        ):
            raise MechanismValidationError(
                "REVIEWER_TASK_ADAPTER_PROGRAM_CONFLICT",
                f"test_program {program!r} is already owned by {owner!r}",
            )
    for evaluator_id in evaluators:
        owner = _owner_of_evaluator(evaluator_id)
        if (
            owner is not None
            and owner != plugin.plugin_id
            and owner != replace_plugin_id
        ):
            raise MechanismValidationError(
                "REVIEWER_TASK_ADAPTER_EVALUATOR_CONFLICT",
                f"evaluator_id {evaluator_id!r} is already owned by {owner!r}",
            )
    existing = _REVIEWER_TASK_ADAPTERS.get(plugin.plugin_id)
    if existing is not None and existing is not plugin:
        prior = _REVIEWER_TASK_ADAPTER_SOURCES.get(plugin.plugin_id)
        replacing = (
            replace_plugin_id is not None
            and plugin.plugin_id == replace_plugin_id
        )
        if (
            prior is not None
            and prior.get("sha256") != expected_sha256
            and not replacing
        ):
            raise MechanismValidationError(
                "REVIEWER_TASK_ADAPTER_ID_CONFLICT",
                f"reviewer task adapter {plugin.plugin_id!r} is already bound",
            )
    if bind:
        _REVIEWER_TASK_ADAPTERS[plugin.plugin_id] = plugin
        _REVIEWER_TASK_ADAPTER_CACHE[cache_key] = plugin
        _REVIEWER_TASK_ADAPTER_SOURCES[plugin.plugin_id] = {
            "path": str(path),
            "sha256": expected_sha256,
            "registration_source": "reviewer_task_adapter",
        }
    return plugin


def load_reviewer_task_adapter(
    source: str | Path,
    expected_sha256: str,
    *,
    bind: bool = True,
    replace_plugin_id: str | None = None,
) -> TaskAdapter:
    """Import a reviewer TaskAdapter and preserve delivery ownership on error."""

    try:
        return _load_reviewer_task_adapter(
            source,
            expected_sha256,
            bind=bind,
            replace_plugin_id=replace_plugin_id,
        )
    except MechanismValidationError as exc:
        owned = _reviewer_task_artifact_failure(
            exc,
            plugin_id=replace_plugin_id,
        )
        if owned is exc:
            raise
        raise owned from exc


def load_reviewer_task_plugin_entries(
    entries: list[Any],
    *,
    registry_path: Path,
    workspace_root: str | Path | None,
) -> tuple[TaskAdapter, ...]:
    """Bind hash-bound TaskAdapter rows from a reviewer plugin registry."""

    root = None if workspace_root is None else Path(workspace_root).resolve()
    loaded: list[TaskAdapter] = []
    for item in entries:
        if not isinstance(item, dict):
            raise _reviewer_task_registry_failure(
                "REVIEWER_TASK_ADAPTER_REGISTRY_INVALID",
                "reviewer task plugin registry entries must be objects",
            )
        relative = item.get("path")
        digest = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise _reviewer_task_registry_failure(
                "REVIEWER_TASK_ADAPTER_REGISTRY_INVALID",
                "reviewer task plugin registry entries need path and sha256",
            )
        source = Path(relative)
        if not source.is_absolute():
            if root is None:
                source = registry_path.parent / relative
            else:
                source = root.joinpath(*Path(relative).parts)
        loaded.append(load_reviewer_task_adapter(source, digest))
    return tuple(loaded)


def merge_mechanism_and_task_load_cases(
    mechanism_cases: tuple[LoadCase, ...],
    plan: MechanismPlan,
) -> tuple[LoadCase, ...]:
    """Keep joint-adapter cases and add TaskAdapter programs that are missing."""

    extra: list[LoadCase] = []
    seen_ids = {item.load_case_id for item in mechanism_cases}
    seen_programs = {item.program for item in mechanism_cases}
    for plugin in _REVIEWER_TASK_ADAPTERS.values():
        if not _supports_plan(plugin, plan):
            continue
        plan_to_dict = getattr(plan, "to_dict", None)
        plan_before = plan_to_dict() if callable(plan_to_dict) else None
        try:
            plugin_cases = plugin.load_cases(plan)
        except MechanismValidationError as exc:
            if exc.failure_envelope is not None:
                raise
            raise _reviewer_task_callback_failure(
                plugin,
                "load_cases",
                exc,
            ) from exc
        except Exception as exc:
            raise _reviewer_task_callback_failure(
                plugin,
                "load_cases",
                exc,
                code="REVIEWER_TASK_ADAPTER_LOAD_CASE_FAILED",
                message=(
                    f"task adapter {plugin.plugin_id!r} load_cases() failed: {exc}"
                ),
            ) from exc
        if plan_before is not None and plan_to_dict() != plan_before:
            error = MechanismValidationError(
                "REVIEWER_TASK_ADAPTER_LOAD_CASE_INVALID",
                "task adapter load_cases() must not mutate the mechanism plan",
            )
            raise _reviewer_task_callback_failure(
                plugin,
                "load_cases",
                error,
                contract_violation=True,
            )
        if not isinstance(plugin_cases, tuple) or any(
            not isinstance(item, LoadCase) for item in plugin_cases
        ):
            error = MechanismValidationError(
                "REVIEWER_TASK_ADAPTER_LOAD_CASE_INVALID",
                "task adapter load_cases() must return a tuple of LoadCase values",
            )
            raise _reviewer_task_callback_failure(
                plugin,
                "load_cases",
                error,
                contract_violation=True,
            )
        for load_case in plugin_cases:
            if load_case.program not in plugin.programs:
                error = MechanismValidationError(
                    "REVIEWER_TASK_ADAPTER_LOAD_CASE_INVALID",
                    "task adapter emitted a load case for unclaimed program "
                    f"{load_case.program!r}",
                    details={
                        "plugin_id": plugin.plugin_id,
                        "program": load_case.program,
                        "load_case_id": load_case.load_case_id,
                    },
                )
                raise _reviewer_task_callback_failure(
                    plugin,
                    "load_cases",
                    error,
                    contract_violation=True,
                )
            if load_case.program in seen_programs:
                continue
            if load_case.load_case_id in seen_ids:
                error = MechanismValidationError(
                    "REVIEWER_TASK_ADAPTER_LOAD_CASE_CONFLICT",
                    "task adapter load_case_id "
                    f"{load_case.load_case_id!r} collides with a joint load case",
                    details={
                        "plugin_id": plugin.plugin_id,
                        "load_case_id": load_case.load_case_id,
                    },
                )
                raise _reviewer_task_callback_failure(
                    plugin,
                    "load_cases",
                    error,
                    contract_violation=True,
                )
            extra.append(load_case)
            seen_ids.add(load_case.load_case_id)
            seen_programs.add(load_case.program)
    if not extra:
        return mechanism_cases
    return mechanism_cases + tuple(extra)


def reviewer_task_program_policy(
    load_case: LoadCase,
    *,
    plan: MechanismPlan | None = None,
) -> TaskProgramEvidenceSpec | None:
    plugin = task_adapter_for_program(load_case.program, plan=plan)
    if plugin is None:
        return None
    try:
        spec = plugin.program_policy(load_case)
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise _reviewer_task_callback_failure(
            plugin,
            "program_policy",
            exc,
            details={
                "program": load_case.program,
                "load_case_id": load_case.load_case_id,
            },
            contract_violation=(
                exc.code == "REVIEWER_TASK_ADAPTER_POLICY_MISSING"
            ),
        ) from exc
    except Exception as exc:
        raise _reviewer_task_callback_failure(
            plugin,
            "program_policy",
            exc,
            code="REVIEWER_TASK_ADAPTER_POLICY_FAILED",
            message=(
                f"task adapter {plugin.plugin_id!r} program_policy() failed: {exc}"
            ),
            details={
                "program": load_case.program,
                "load_case_id": load_case.load_case_id,
            },
        ) from exc
    if not isinstance(spec, TaskProgramEvidenceSpec):
        error = MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_POLICY_INVALID",
            "task adapter program_policy must return TaskProgramEvidenceSpec",
        )
        raise _reviewer_task_callback_failure(
            plugin,
            "program_policy",
            error,
            details={
                "program": load_case.program,
                "load_case_id": load_case.load_case_id,
            },
            contract_violation=True,
        )
    if spec.program != load_case.program:
        error = MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_POLICY_INVALID",
            "task adapter program_policy program does not match the load case",
        )
        raise _reviewer_task_callback_failure(
            plugin,
            "program_policy",
            error,
            details={
                "program": load_case.program,
                "load_case_id": load_case.load_case_id,
            },
            contract_violation=True,
        )
    return spec


def evaluate_with_reviewer_task_adapter(
    requirement: Mapping[str, Any],
    binding: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    plan: MechanismPlan | None = None,
    canonical_generalized_observable_id: str | None = None,
    validated_program: Any = None,
    resolved_channels: Mapping[str, Any] | None = None,
    verified_wrench_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    metric_id = str(requirement.get("metric_id") or "")
    plugin = task_adapter_for_evaluator(metric_id, plan=plan)
    if plugin is None:
        return None
    try:
        result = plugin.evaluate(
            requirement,
            binding,
            trace,
            canonical_generalized_observable_id=canonical_generalized_observable_id,
            validated_program=validated_program,
            resolved_channels=resolved_channels,
            verified_wrench_audit=verified_wrench_audit,
        )
    except MechanismValidationError as exc:
        if exc.failure_envelope is not None:
            raise
        raise _reviewer_task_callback_failure(
            plugin,
            "evaluate",
            exc,
            details={"evaluator_id": metric_id},
            contract_violation=(
                exc.code == "REVIEWER_TASK_ADAPTER_EVALUATOR_MISSING"
            ),
        ) from exc
    except Exception as exc:
        raise _reviewer_task_callback_failure(
            plugin,
            "evaluate",
            exc,
            code="REVIEWER_TASK_ADAPTER_EVALUATE_FAILED",
            message=f"task adapter {plugin.plugin_id!r} evaluate() failed: {exc}",
            details={"evaluator_id": metric_id},
        ) from exc
    if (
        not isinstance(result, dict)
        or not isinstance(result.get("passed"), bool)
    ):
        error = MechanismValidationError(
            "REVIEWER_TASK_ADAPTER_EVALUATE_INVALID",
            "task adapter evaluate() must return a dict with boolean passed",
        )
        raise _reviewer_task_callback_failure(
            plugin,
            "evaluate",
            error,
            details={"evaluator_id": metric_id},
            contract_violation=True,
        )
    return result
