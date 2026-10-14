from __future__ import annotations

"""Trusted, typed acceptance contracts compiled from immutable requirements.

The package may describe how it implements a task, but it may not choose the
metric, target, tolerance, or branch that decides whether that task passed.
Those authorities live in the external requirements document and are compiled
here into a bounded, hashable contract.
"""

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Mapping

from .clock_task_suite import (
    CLOCK_TASK_METRIC_VERSION,
    ClockTaskCriterion,
    ClockTaskSuiteError,
    is_clock_task_id,
    resolve_clock_task_criterion,
    validate_clock_requirement_entry,
    verify_clock_task_suite_binding,
)
from .egg_beater_task_suite import (
    EGG_BEATER_TASK_METRIC_VERSION,
    EggBeaterTaskCriterion,
    EggBeaterTaskSuiteError,
    is_egg_beater_task_id,
    resolve_egg_beater_task_criterion,
    validate_egg_beater_requirement_entry,
    verify_egg_beater_task_suite_binding,
)
from .prompt_clause_binding import prompt_clause_is_bound
from .rack_pinion_gate_task_suite import (
    RACK_PINION_GATE_TASK_METRIC_VERSION,
    RackPinionGateTaskCriterion,
    RackPinionGateTaskSuiteError,
    is_rack_pinion_gate_task_id,
    resolve_rack_pinion_gate_task_criterion,
    validate_rack_pinion_gate_requirement_entry,
    verify_rack_pinion_gate_task_suite_binding,
)
from .tongs_task_suite import (
    TONGS_TASK_METRIC_VERSION,
    TongsTaskCriterion,
    TongsTaskSuiteError,
    is_tongs_task_id,
    resolve_tongs_task_criterion,
    validate_tongs_requirement_entry,
    verify_tongs_task_suite_binding,
)
from .utility_knife_task_suite import (
    UTILITY_KNIFE_TASK_METRIC_VERSION,
    UtilityKnifeTaskCriterion,
    UtilityKnifeTaskSuiteError,
    is_utility_knife_task_id,
    resolve_utility_knife_task_criterion,
    validate_utility_knife_requirement_entry,
    verify_utility_knife_task_suite_binding,
)


ACCEPTANCE_CONTRACT_SCHEMA = "acceptance_contract_v1"
METRIC_REGISTRY_VERSION = "metric_registry_v8"
BRANCHES = frozenset({"physical", "ideal_reference", "ideal_dynamics"})
_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    value_kind: str
    unit: str
    comparators: frozenset[str]
    compatible_quantities: frozenset[str]
    dynamics_required: bool = False


METRIC_REGISTRY: dict[str, MetricDefinition] = {
    item.metric_id: item
    for item in (
        MetricDefinition(
            "continuous_rotation_rad",
            "number",
            "rad",
            frozenset({"at_least", "at_most", "equals", "within"}),
            frozenset({"pose_6d"}),
        ),
        MetricDefinition(
            "return_to_start_rad",
            "number",
            "rad",
            frozenset({"at_least", "at_most", "equals", "within"}),
            frozenset({"pose_6d"}),
        ),
        MetricDefinition(
            "output_torque_nm",
            "number",
            "N·m",
            frozenset({"at_most"}),
            frozenset({"pose_6d"}),
        ),
        MetricDefinition(
            "return_to_start_m",
            "number",
            "m",
            frozenset({"at_most", "equals", "within"}),
            frozenset({"pose_6d", "position_3d"}),
        ),
        MetricDefinition(
            "signed_displacement_m",
            "number",
            "m",
            frozenset({"at_least", "at_most", "equals", "within"}),
            frozenset({"pose_6d", "position_3d"}),
        ),
        MetricDefinition(
            "angular_ratio",
            "number",
            "ratio",
            frozenset({"equals", "within"}),
            frozenset({"pose_6d"}),
        ),
        MetricDefinition(
            "displacement_ratio",
            "number",
            "ratio",
            frozenset({"equals", "within"}),
            frozenset({"pose_6d", "position_3d"}),
        ),
        MetricDefinition(
            "rolling_slip_ratio",
            "number",
            "ratio",
            frozenset({"at_most"}),
            frozenset({"pose_6d"}),
        ),
        MetricDefinition(
            "powered_contact_coverage_ratio",
            "number",
            "ratio",
            frozenset({"at_least"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "ground_separation_m",
            "number",
            "m",
            frozenset({"at_most"}),
            frozenset({"contact_event", "pose_6d"}),
        ),
        MetricDefinition(
            "cumulative_travel_m",
            "number",
            "m",
            frozenset({"at_least", "at_most", "equals", "within"}),
            frozenset({"pose_6d", "position_3d"}),
        ),
        MetricDefinition(
            "expected_contact",
            "boolean",
            "bool",
            frozenset({"equals"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_two_sided_grip",
            "boolean",
            "bool",
            frozenset({"equals"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_object_starts_on_table",
            "boolean",
            "bool",
            frozenset({"equals"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_pickup_lift_m",
            "number",
            "m",
            frozenset({"at_least"}),
            frozenset({"position_3d", "pose_6d", "contact_event"}),
        ),
        MetricDefinition(
            "tong_pickup_table_clear",
            "boolean",
            "bool",
            frozenset({"equals"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_hold_slip_m",
            "number",
            "m",
            frozenset({"at_most"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_hold_relative_slip_m",
            "number",
            "m",
            frozenset({"at_most"}),
            frozenset({"position_3d", "pose_6d", "contact_event"}),
        ),
        MetricDefinition(
            "tong_release_drop_m",
            "number",
            "m",
            frozenset({"at_least"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_release_contact_clear",
            "boolean",
            "bool",
            frozenset({"equals"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "tong_release_returns_to_table",
            "boolean",
            "bool",
            frozenset({"equals"}),
            frozenset({"contact_event"}),
        ),
        MetricDefinition(
            "forbidden_penetration_m",
            "number",
            "m",
            frozenset({"at_most"}),
            frozenset({"contact_event", "pose_6d", "position_3d"}),
        ),
        MetricDefinition(
            "actuation_provenance",
            "string",
            "enum",
            frozenset({"equals"}),
            frozenset({"actuation_log"}),
        ),
    )
}


def _suite_metric_definition(
    criterion: (
        ClockTaskCriterion
        | EggBeaterTaskCriterion
        | UtilityKnifeTaskCriterion
        | TongsTaskCriterion
        | RackPinionGateTaskCriterion
    ),
) -> MetricDefinition:
    return MetricDefinition(
        criterion.evaluator_id,
        criterion.value_kind,
        criterion.unit,
        frozenset({criterion.comparator}),
        frozenset({criterion.observable_quantity}),
    )


def _task_suite_criterion_for(
    requirement: Mapping[str, Any],
) -> (
    ClockTaskCriterion
    | EggBeaterTaskCriterion
    | UtilityKnifeTaskCriterion
    | TongsTaskCriterion
    | RackPinionGateTaskCriterion
    | None
):
    """Resolve the approved suite criterion bound to a requirement, if any."""

    task_id = str(requirement.get("task_id") or "")
    try:
        if is_utility_knife_task_id(task_id):
            criterion = verify_utility_knife_task_suite_binding(requirement)
        elif is_egg_beater_task_id(task_id):
            criterion = verify_egg_beater_task_suite_binding(requirement)
        elif is_tongs_task_id(task_id):
            criterion = verify_tongs_task_suite_binding(requirement)
        elif is_rack_pinion_gate_task_id(task_id):
            criterion = verify_rack_pinion_gate_task_suite_binding(requirement)
        else:
            criterion = verify_clock_task_suite_binding(requirement)
    except (
        ClockTaskSuiteError,
        EggBeaterTaskSuiteError,
        UtilityKnifeTaskSuiteError,
        TongsTaskSuiteError,
        RackPinionGateTaskSuiteError,
    ):
        return None
    if criterion is None and (
        is_clock_task_id(task_id)
        or is_egg_beater_task_id(task_id)
        or is_utility_knife_task_id(task_id)
        or is_tongs_task_id(task_id)
        or is_rack_pinion_gate_task_id(task_id)
    ):
        return None
    return criterion


def metric_definition_for_requirement(
    requirement: Mapping[str, Any],
) -> MetricDefinition | None:
    """Resolve a core metric or an exact task-suite-owned evaluator."""

    metric_id = str(requirement.get("metric_id") or "")
    definition = METRIC_REGISTRY.get(metric_id)
    task_id = str(requirement.get("task_id") or "")
    criterion = _task_suite_criterion_for(requirement)
    if criterion is None and (
        is_clock_task_id(task_id)
        or is_egg_beater_task_id(task_id)
        or is_utility_knife_task_id(task_id)
        or is_tongs_task_id(task_id)
        or is_rack_pinion_gate_task_id(task_id)
    ):
        return None
    if criterion is not None:
        suite_definition = _suite_metric_definition(criterion)
        if definition is None:
            definition = suite_definition
        elif (
            definition.value_kind != suite_definition.value_kind
            or definition.unit != suite_definition.unit
            or criterion.comparator not in definition.comparators
            or criterion.observable_quantity
            not in definition.compatible_quantities
        ):
            return None
    return definition


def metric_version_for_requirement(
    requirement: Mapping[str, Any],
) -> str | None:
    metric_id = str(requirement.get("metric_id") or "")
    task_id = str(requirement.get("task_id") or "")
    criterion = _task_suite_criterion_for(requirement)
    if criterion is None and (
        is_clock_task_id(task_id)
        or is_egg_beater_task_id(task_id)
        or is_utility_knife_task_id(task_id)
        or is_tongs_task_id(task_id)
        or is_rack_pinion_gate_task_id(task_id)
    ):
        return None
    if metric_id in METRIC_REGISTRY:
        return METRIC_REGISTRY_VERSION
    if criterion is not None:
        if is_utility_knife_task_id(task_id):
            return UTILITY_KNIFE_TASK_METRIC_VERSION
        if is_egg_beater_task_id(task_id):
            return EGG_BEATER_TASK_METRIC_VERSION
        if is_tongs_task_id(task_id):
            return TONGS_TASK_METRIC_VERSION
        if is_rack_pinion_gate_task_id(task_id):
            return RACK_PINION_GATE_TASK_METRIC_VERSION
        return CLOCK_TASK_METRIC_VERSION
    return None


class AcceptanceContractError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        json_path: str,
        expected: Any = None,
        actual: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.json_path = json_path
        self.expected = expected
        self.actual = actual


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fail(
    code: str,
    message: str,
    path: str,
    *,
    expected: Any = None,
    actual: Any = None,
) -> AcceptanceContractError:
    return AcceptanceContractError(
        code,
        message,
        json_path=path,
        expected=expected,
        actual=actual,
    )


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise _fail(
            "ACCEPTANCE_IDENTIFIER_INVALID",
            "Acceptance identifiers must be stable bounded identifiers.",
            path,
            expected="letter followed by letters, digits, '.', '_' or '-'",
            actual=value,
        )
    return value


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise _fail(
            "ACCEPTANCE_TEXT_INVALID",
            "Acceptance text must be non-empty and bounded.",
            path,
            actual=value,
        )
    return value.strip()


def _target(
    value: Any,
    *,
    definition: MetricDefinition,
    path: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(
            "ACCEPTANCE_TARGET_INVALID",
            "A typed metric requires an object target.",
            path,
            expected={
                "comparator": "...",
                "value": "...",
                "unit": definition.unit,
            },
            actual=value,
        )
    comparator = value.get("comparator")
    if comparator not in definition.comparators:
        raise _fail(
            "ACCEPTANCE_COMPARATOR_INVALID",
            "The comparator is not registered for this metric.",
            f"{path}.comparator",
            expected=sorted(definition.comparators),
            actual=comparator,
        )
    unit = value.get("unit")
    if unit != definition.unit:
        raise _fail(
            "ACCEPTANCE_UNIT_INVALID",
            "The target unit does not match the trusted metric registry.",
            f"{path}.unit",
            expected=definition.unit,
            actual=unit,
        )
    target_value = value.get("value")
    if definition.value_kind == "number":
        if (
            isinstance(target_value, bool)
            or not isinstance(target_value, (int, float))
            or not math.isfinite(float(target_value))
        ):
            raise _fail(
                "ACCEPTANCE_VALUE_INVALID",
                "The metric target must be a finite number.",
                f"{path}.value",
                actual=target_value,
            )
        target_value = float(target_value)
    elif definition.value_kind == "boolean":
        if not isinstance(target_value, bool):
            raise _fail(
                "ACCEPTANCE_VALUE_INVALID",
                "The metric target must be boolean.",
                f"{path}.value",
                actual=target_value,
            )
    elif definition.value_kind == "string":
        target_value = _text(target_value, f"{path}.value")
    tolerance = value.get("tolerance")
    if comparator in {"equals", "within"} and definition.value_kind == "number":
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise _fail(
                "ACCEPTANCE_TOLERANCE_INVALID",
                "Equality and window comparisons require a finite non-negative tolerance.",
                f"{path}.tolerance",
                actual=tolerance,
            )
        tolerance = float(tolerance)
    elif tolerance is not None:
        raise _fail(
            "ACCEPTANCE_TOLERANCE_UNEXPECTED",
            "This comparator does not accept a tolerance.",
            f"{path}.tolerance",
            actual=tolerance,
        )
    result = {
        "comparator": comparator,
        "value": target_value,
        "unit": definition.unit,
    }
    if tolerance is not None:
        result["tolerance"] = tolerance
    return result


def _string_list(
    value: Any,
    path: str,
    *,
    allowed: set[str] | None = None,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise _fail(
            "ACCEPTANCE_LIST_REQUIRED",
            "This acceptance field requires a non-empty list.",
            path,
            actual=value,
        )
    result: list[str] = []
    for index, item in enumerate(value):
        item = _identifier(item, f"{path}[{index}]")
        if allowed is not None and item not in allowed:
            raise _fail(
                "ACCEPTANCE_LIST_VALUE_INVALID",
                "The acceptance list contains an unsupported value.",
                f"{path}[{index}]",
                expected=sorted(allowed),
                actual=item,
            )
        if item not in result:
            result.append(item)
    return result


def _legacy_contract(requirements: Mapping[str, Any]) -> dict[str, Any]:
    task_ids = [
        task.get("task_id")
        for task in requirements.get("tasks", [])
        if isinstance(task, Mapping) and isinstance(task.get("task_id"), str)
    ]
    payload = {
        "schema_version": ACCEPTANCE_CONTRACT_SCHEMA,
        "metric_registry_version": METRIC_REGISTRY_VERSION,
        "status": "LEGACY_UNTYPED",
        "certificate_authority": False,
        "requirements_id": requirements.get("requirements_id"),
        "task_ids": task_ids,
        "requirements": [],
        "failure_reasons": ["typed_acceptance_requirements_missing"],
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def compile_acceptance_contract(
    requirements: Mapping[str, Any],
) -> dict[str, Any]:
    """Compile strict schema 0.2 requirements; preserve schema 0.1 as legacy."""

    tasks = requirements.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise _fail(
            "ACCEPTANCE_TASKS_REQUIRED",
            "Acceptance compilation requires external tasks.",
            "$.tasks",
            actual=tasks,
        )
    typed = any(
        isinstance(task, Mapping) and "acceptance_requirements" in task
        for task in tasks
    )
    if not typed:
        if requirements.get("schema_version") == "0.1":
            return _legacy_contract(requirements)
        raise _fail(
            "TYPED_ACCEPTANCE_REQUIRED",
            "New requirements must include typed acceptance requirements for every task.",
            "$.tasks",
        )
    prompt = _text(requirements.get("prompt"), "$.prompt")
    requirements_id = _identifier(
        requirements.get("requirements_id"),
        "$.requirements_id",
    )
    compiled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task_index, task in enumerate(tasks):
        task_path = f"$.tasks[{task_index}]"
        if not isinstance(task, Mapping):
            raise _fail(
                "ACCEPTANCE_TASK_INVALID",
                "Each acceptance task must be an object.",
                task_path,
                actual=task,
            )
        task_id = _identifier(task.get("task_id"), f"{task_path}.task_id")
        entries = task.get("acceptance_requirements")
        if not isinstance(entries, list) or not entries:
            raise _fail(
                "TYPED_ACCEPTANCE_REQUIRED",
                "Every task requires at least one typed acceptance requirement.",
                f"{task_path}.acceptance_requirements",
                actual=entries,
            )
        for entry_index, entry in enumerate(entries):
            path = f"{task_path}.acceptance_requirements[{entry_index}]"
            if not isinstance(entry, Mapping):
                raise _fail(
                    "ACCEPTANCE_REQUIREMENT_INVALID",
                    "Each typed acceptance requirement must be an object.",
                    path,
                    actual=entry,
                )
            requirement_id = _identifier(
                entry.get("requirement_id"),
                f"{path}.requirement_id",
            )
            if requirement_id in seen:
                raise _fail(
                    "ACCEPTANCE_REQUIREMENT_DUPLICATE",
                    "Typed acceptance requirement IDs must be unique.",
                    f"{path}.requirement_id",
                    actual=requirement_id,
                )
            seen.add(requirement_id)
            clause = _text(
                entry.get("prompt_clause"),
                f"{path}.prompt_clause",
            )
            if not prompt_clause_is_bound(
                prompt,
                clause,
                target_role=(
                    str(entry.get("target_role"))
                    if isinstance(entry.get("target_role"), str)
                    else None
                ),
                input_role=(
                    str(entry.get("input_role"))
                    if isinstance(entry.get("input_role"), str)
                    else None
                ),
            ):
                raise _fail(
                    "PROMPT_CLAUSE_UNBOUND",
                    "The cited requirement clause is not present in the immutable prompt.",
                    f"{path}.prompt_clause",
                    actual=clause,
                )
            metric_id = _identifier(
                entry.get("metric_id"),
                f"{path}.metric_id",
            )
            try:
                suite_criterion = resolve_clock_task_criterion(
                    task_id,
                    requirement_id,
                    metric_id,
                )
                if suite_criterion is not None:
                    validate_clock_requirement_entry(entry, suite_criterion)
            except ClockTaskSuiteError as exc:
                raise _fail(
                    exc.code,
                    str(exc),
                    path,
                    actual=entry,
                ) from exc
            if suite_criterion is None:
                try:
                    suite_criterion = resolve_egg_beater_task_criterion(
                        task_id,
                        requirement_id,
                        metric_id,
                    )
                    if suite_criterion is not None:
                        validate_egg_beater_requirement_entry(
                            entry,
                            suite_criterion,
                        )
                except EggBeaterTaskSuiteError as exc:
                    raise _fail(
                        exc.code,
                        str(exc),
                        path,
                        actual=entry,
                    ) from exc
            if suite_criterion is None:
                try:
                    suite_criterion = resolve_utility_knife_task_criterion(
                        task_id,
                        requirement_id,
                        metric_id,
                    )
                    if suite_criterion is not None:
                        validate_utility_knife_requirement_entry(
                            entry,
                            suite_criterion,
                        )
                except UtilityKnifeTaskSuiteError as exc:
                    raise _fail(
                        exc.code,
                        str(exc),
                        path,
                        actual=entry,
                    ) from exc
            if suite_criterion is None:
                try:
                    suite_criterion = resolve_tongs_task_criterion(
                        task_id,
                        requirement_id,
                        metric_id,
                    )
                    if suite_criterion is not None:
                        validate_tongs_requirement_entry(
                            entry,
                            suite_criterion,
                        )
                except TongsTaskSuiteError as exc:
                    raise _fail(
                        exc.code,
                        str(exc),
                        path,
                        actual=entry,
                    ) from exc
            if suite_criterion is None:
                try:
                    suite_criterion = resolve_rack_pinion_gate_task_criterion(
                        task_id,
                        requirement_id,
                        metric_id,
                    )
                    if suite_criterion is not None:
                        validate_rack_pinion_gate_requirement_entry(
                            entry,
                            suite_criterion,
                        )
                except RackPinionGateTaskSuiteError as exc:
                    raise _fail(
                        exc.code,
                        str(exc),
                        path,
                        actual=entry,
                    ) from exc
            definition = METRIC_REGISTRY.get(metric_id)
            if definition is None and suite_criterion is not None:
                definition = _suite_metric_definition(suite_criterion)
            if definition is None:
                raise _fail(
                    "ACCEPTANCE_METRIC_UNREGISTERED",
                    "Every automatic criterion must use a trusted registered metric.",
                    f"{path}.metric_id",
                    expected=sorted(METRIC_REGISTRY),
                    actual=metric_id,
                )
            observable_quantity = entry.get("observable_quantity")
            if observable_quantity is not None:
                observable_quantity = _identifier(
                    observable_quantity,
                    f"{path}.observable_quantity",
                )
                if observable_quantity not in definition.compatible_quantities:
                    raise _fail(
                        "ACCEPTANCE_OBSERVABLE_QUANTITY_INVALID",
                        (
                            "The frozen observable quantity cannot feed the "
                            "registered metric."
                        ),
                        f"{path}.observable_quantity",
                        expected=sorted(definition.compatible_quantities),
                        actual=observable_quantity,
                    )
            branches = _string_list(
                entry.get("required_branches"),
                f"{path}.required_branches",
                allowed=set(BRANCHES),
            )
            if (
                definition.dynamics_required
                and "ideal_dynamics" not in branches
            ):
                raise _fail(
                    "IDEAL_DYNAMICS_BRANCH_REQUIRED",
                    "This metric requires the optional ideal dynamics branch.",
                    f"{path}.required_branches",
                )
            allow_direct = entry.get("allow_direct_output_actuation")
            if not isinstance(allow_direct, bool):
                raise _fail(
                    "DIRECT_ACTUATION_POLICY_REQUIRED",
                    "The contract must explicitly allow or forbid direct output actuation.",
                    f"{path}.allow_direct_output_actuation",
                    actual=allow_direct,
                )
            direction = entry.get("direction", "either")
            if direction not in {"positive", "negative", "either"}:
                raise _fail(
                    "ACCEPTANCE_DIRECTION_INVALID",
                    "Direction must be positive, negative, or either.",
                    f"{path}.direction",
                    expected=["either", "negative", "positive"],
                    actual=direction,
                )
            compiled.append(
                {
                    "requirement_id": requirement_id,
                    "task_id": task_id,
                    "prompt_clause": clause,
                    "target_role": _text(
                        entry.get("target_role"),
                        f"{path}.target_role",
                    ),
                    "reference_role": (
                        _text(
                            entry.get("reference_role"),
                            f"{path}.reference_role",
                        )
                        if entry.get("reference_role") is not None
                        else None
                    ),
                    "input_role": (
                        _text(
                            entry.get("input_role"),
                            f"{path}.input_role",
                        )
                        if entry.get("input_role") is not None
                        else None
                    ),
                    "metric_id": metric_id,
                    "metric_version": (
                        METRIC_REGISTRY_VERSION
                        if metric_id in METRIC_REGISTRY
                        else (
                            UTILITY_KNIFE_TASK_METRIC_VERSION
                            if is_utility_knife_task_id(task_id)
                            else (
                                EGG_BEATER_TASK_METRIC_VERSION
                                if is_egg_beater_task_id(task_id)
                                else (
                                    TONGS_TASK_METRIC_VERSION
                                    if is_tongs_task_id(task_id)
                                    else (
                                        RACK_PINION_GATE_TASK_METRIC_VERSION
                                        if is_rack_pinion_gate_task_id(task_id)
                                        else CLOCK_TASK_METRIC_VERSION
                                    )
                                )
                            )
                        )
                    ),
                    **(
                        {"task_suite_binding": suite_criterion.binding()}
                        if suite_criterion is not None
                        else {}
                    ),
                    **(
                        {"observable_quantity": observable_quantity}
                        if observable_quantity is not None
                        else {}
                    ),
                    "direction": direction,
                    "target": _target(
                        (
                            suite_criterion.effective_target()
                            if suite_criterion is not None
                            else entry.get("target")
                        ),
                        definition=definition,
                        path=f"{path}.target",
                    ),
                    "coordinate_frame_role": (
                        _text(
                            entry.get("coordinate_frame_role"),
                            f"{path}.coordinate_frame_role",
                        )
                        if entry.get("coordinate_frame_role") is not None
                        else None
                    ),
                    "working_mode": (
                        _identifier(
                            entry.get("working_mode"),
                            f"{path}.working_mode",
                        )
                        if entry.get("working_mode") is not None
                        else "default"
                    ),
                    "test_program": _identifier(
                        entry.get("test_program"),
                        f"{path}.test_program",
                    ),
                    "test_load_case_id": _identifier(
                        entry.get("test_load_case_id"),
                        f"{path}.test_load_case_id",
                    ),
                    "allow_direct_output_actuation": allow_direct,
                    "required_branches": branches,
                    "expected_contact_ids": _string_list(
                        entry.get(
                            "expected_contact_ids",
                            ["not_applicable"],
                        ),
                        f"{path}.expected_contact_ids",
                    ),
                    "forbidden_contact_ids": _string_list(
                        entry.get(
                            "forbidden_contact_ids",
                            ["unintended_penetration"],
                        ),
                        f"{path}.forbidden_contact_ids",
                    ),
                }
            )
    payload = {
        "schema_version": ACCEPTANCE_CONTRACT_SCHEMA,
        "metric_registry_version": METRIC_REGISTRY_VERSION,
        "status": "PASS",
        "certificate_authority": True,
        "requirements_id": requirements_id,
        "prompt_identity": {
            "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "character_count": len(prompt),
        },
        "task_ids": [entry["task_id"] for entry in compiled],
        "requirements": compiled,
        "failure_reasons": [],
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    return payload


def contract_for_tasks(
    contract: Mapping[str, Any],
    task_ids: set[str],
) -> dict[str, Any]:
    requirements = [
        dict(item)
        for item in contract.get("requirements", [])
        if isinstance(item, Mapping) and item.get("task_id") in task_ids
    ]
    payload = {
        key: contract[key]
        for key in (
            "schema_version",
            "metric_registry_version",
            "status",
            "certificate_authority",
            "requirements_id",
            "prompt_identity",
            "contract_sha256",
        )
        if key in contract
    }
    payload["task_ids"] = sorted(task_ids)
    payload["requirements"] = requirements
    payload["task_contract_sha256"] = _canonical_sha256(
        {
            "parent": contract.get("contract_sha256"),
            "requirements": requirements,
        }
    )
    return payload


def verify_acceptance_contract(
    contract: Mapping[str, Any],
) -> None:
    """Fail closed when a saved full contract or metric registry was changed."""

    if not isinstance(contract, Mapping):
        raise _fail(
            "ACCEPTANCE_CONTRACT_INVALID",
            "Acceptance contract must be an object.",
            "$.acceptance_contract",
        )
    recorded = contract.get("contract_sha256")
    payload = {
        key: value
        for key, value in contract.items()
        if key != "contract_sha256"
    }
    expected = _canonical_sha256(payload)
    if recorded != expected:
        raise _fail(
            "ACCEPTANCE_CONTRACT_HASH_MISMATCH",
            "Saved acceptance contract content does not match its digest.",
            "$.acceptance_contract.contract_sha256",
            expected=expected,
            actual=recorded,
        )
    if contract.get("schema_version") != ACCEPTANCE_CONTRACT_SCHEMA:
        raise _fail(
            "ACCEPTANCE_CONTRACT_SCHEMA_INVALID",
            "Acceptance contract schema is unsupported.",
            "$.acceptance_contract.schema_version",
            expected=ACCEPTANCE_CONTRACT_SCHEMA,
            actual=contract.get("schema_version"),
        )
    if contract.get("metric_registry_version") != METRIC_REGISTRY_VERSION:
        raise _fail(
            "ACCEPTANCE_METRIC_REGISTRY_CHANGED",
            "Acceptance contract does not bind the active metric registry.",
            "$.acceptance_contract.metric_registry_version",
            expected=METRIC_REGISTRY_VERSION,
            actual=contract.get("metric_registry_version"),
        )
    for index, requirement in enumerate(contract.get("requirements") or []):
        if not isinstance(requirement, Mapping):
            raise _fail(
                "ACCEPTANCE_REQUIREMENT_INVALID",
                "Acceptance requirement must be an object.",
                f"$.acceptance_contract.requirements[{index}]",
            )
        metric_id = requirement.get("metric_id")
        definition = metric_definition_for_requirement(requirement)
        if definition is None:
            raise _fail(
                "ACCEPTANCE_METRIC_UNREGISTERED",
                "Acceptance contract references an unregistered metric.",
                f"$.acceptance_contract.requirements[{index}].metric_id",
                expected=sorted(METRIC_REGISTRY),
                actual=metric_id,
            )
        expected_version = metric_version_for_requirement(requirement)
        if requirement.get("metric_version") != expected_version:
            raise _fail(
                "ACCEPTANCE_METRIC_REGISTRY_CHANGED",
                "Requirement metric version is stale.",
                f"$.acceptance_contract.requirements[{index}].metric_version",
                expected=expected_version,
                actual=requirement.get("metric_version"),
            )


def verify_task_contract(
    contract: Mapping[str, Any],
) -> None:
    """Verify a capsule-scoped contract without trusting its stored hash."""

    if not isinstance(contract, Mapping):
        raise _fail(
            "TASK_CONTRACT_INVALID",
            "Task contract must be an object.",
            "$.task_contract",
        )
    requirements = [
        dict(item)
        for item in contract.get("requirements") or []
        if isinstance(item, Mapping)
    ]
    expected = _canonical_sha256(
        {
            "parent": contract.get("contract_sha256"),
            "requirements": requirements,
        }
    )
    if contract.get("task_contract_sha256") != expected:
        raise _fail(
            "TASK_CONTRACT_HASH_MISMATCH",
            "Saved task contract content does not match its digest.",
            "$.task_contract.task_contract_sha256",
            expected=expected,
            actual=contract.get("task_contract_sha256"),
        )
    for index, requirement in enumerate(requirements):
        metric_id = requirement.get("metric_id")
        definition = metric_definition_for_requirement(requirement)
        if definition is None:
            raise _fail(
                "ACCEPTANCE_METRIC_UNREGISTERED",
                "Task contract references an unregistered metric.",
                f"$.task_contract.requirements[{index}].metric_id",
                expected=sorted(METRIC_REGISTRY),
                actual=metric_id,
            )
        expected_version = metric_version_for_requirement(requirement)
        if requirement.get("metric_version") != expected_version:
            raise _fail(
                "ACCEPTANCE_METRIC_REGISTRY_CHANGED",
                "Task contract metric version is stale.",
                f"$.task_contract.requirements[{index}].metric_version",
                expected=expected_version,
                actual=requirement.get("metric_version"),
            )
