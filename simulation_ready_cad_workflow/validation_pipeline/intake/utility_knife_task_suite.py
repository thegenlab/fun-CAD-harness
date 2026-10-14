from __future__ import annotations

"""Strict acceptance bindings for the approved auto-lock utility-knife task suite.

The generic metric registry remains unchanged.  This module is the narrow
entry adapter that lets schema-0.2 requirements cite evaluator IDs owned by
``task_suite.auto_lock_utility_knife`` while freezing the exact approved file,
task, scenario, criterion, runtime inputs, and evaluation window.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


UTILITY_KNIFE_TASK_SUITE_ID = "task_suite.auto_lock_utility_knife"
UTILITY_KNIFE_TASK_SUITE_VERSION = "1.0.0"
UTILITY_KNIFE_TASK_SUITE_SHA256 = (
    "d7b29e1ffba038820421774863e3f0bd595cad21ffae3c89d6da9b39f8bec3b0"
)
UTILITY_KNIFE_TASK_METRIC_VERSION = (
    f"{UTILITY_KNIFE_TASK_SUITE_ID}@{UTILITY_KNIFE_TASK_SUITE_VERSION}:"
    f"{UTILITY_KNIFE_TASK_SUITE_SHA256}"
)
UTILITY_KNIFE_TASK_SUITE_EVALUATOR_IDS = frozenset(
    {
        "utility_knife_reverse_load_magnitude",
        "utility_knife_loaded_hold_duration",
        "utility_knife_maximum_loaded_slip",
    }
)
UTILITY_KNIFE_TASK_IDS = frozenset(
    {
        "task.extend_to_working_position",
        "task.hold_working_position_under_load",
        "task.select_second_position",
        "task.retract_blade",
    }
)

_TASK_SUITE_PATH = (
    Path(__file__).resolve().parents[1]
    / "planning_templates"
    / "task_suites"
    / UTILITY_KNIFE_TASK_SUITE_ID
    / f"{UTILITY_KNIFE_TASK_SUITE_VERSION}.json"
)


class UtilityKnifeTaskSuiteError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class UtilityKnifeTaskCriterion:
    task_id: str
    scenario_id: str
    criterion_id: str
    evaluator_kind: str
    evaluator_id: str
    observable_quantity: str
    comparator: str
    expected: Any
    unit: str
    tolerance_mode: str
    tolerance_value: float | None
    test_program: str
    test_load_case_id: str
    required_branches: tuple[str, ...]
    evaluation_phase_ids: tuple[str, ...]
    allowed_input_port_ids: tuple[str, ...]
    reference_policy: str

    @property
    def value_kind(self) -> str:
        if isinstance(self.expected, bool):
            return "boolean"
        if isinstance(self.expected, (int, float)):
            return "number"
        return "string"

    def declared_target(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "comparator": self.comparator,
            "value": self.expected,
            "unit": self.unit,
            "tolerance_mode": self.tolerance_mode,
        }
        if self.tolerance_value is not None:
            payload["tolerance"] = self.tolerance_value
        return payload

    def effective_target(self) -> dict[str, Any]:
        value = self.expected
        tolerance: float | None = None
        if self.tolerance_mode == "absolute":
            tolerance = self.tolerance_value
        elif self.tolerance_mode == "relative_percent":
            tolerance = abs(float(value)) * float(self.tolerance_value) / 100.0
        elif self.tolerance_mode == "shortfall":
            if self.comparator != "at_least":
                raise UtilityKnifeTaskSuiteError(
                    "UTILITY_KNIFE_TASK_TOLERANCE_INVALID",
                    "utility-knife shortfall tolerance requires an at_least criterion",
                )
            value = float(value) - float(self.tolerance_value)
        elif (
            self.tolerance_mode == "none"
            and self.value_kind == "number"
            and self.comparator in {"equals", "within"}
        ):
            # The contract compiler requires a finite tolerance for numeric
            # equality windows; a suite-declared "none" compiles to an exact
            # zero tolerance.
            tolerance = 0.0
        return {
            "comparator": self.comparator,
            "value": value,
            "unit": self.unit,
            **({"tolerance": tolerance} if tolerance is not None else {}),
        }

    def binding(self) -> dict[str, Any]:
        return {
            "template_id": UTILITY_KNIFE_TASK_SUITE_ID,
            "version": UTILITY_KNIFE_TASK_SUITE_VERSION,
            "sha256": UTILITY_KNIFE_TASK_SUITE_SHA256,
            "task_id": self.task_id,
            "scenario_id": self.scenario_id,
            "criterion_id": self.criterion_id,
            "evaluator_kind": self.evaluator_kind,
            "evaluator_id": self.evaluator_id,
            "declared_target": self.declared_target(),
            "evaluation_phase_ids": list(self.evaluation_phase_ids),
            "allowed_input_port_ids": list(self.allowed_input_port_ids),
            "reference_policy": self.reference_policy,
        }


_RUNTIME_POLICY: dict[str, dict[str, Any]] = {
    "task.extend_to_working_position": {
        "allowed_input_port_ids": ("thumb_press_and_slide",),
        "default_phase_ids": (
            "press_and_slide_to_detent",
            "release_into_detent",
        ),
        "criterion_phase_ids": {},
    },
    "task.hold_working_position_under_load": {
        "allowed_input_port_ids": ("thumb_press_and_slide",),
        "default_phase_ids": ("axial_load_hold",),
        "criterion_phase_ids": {
            # The final-position check measures displacement from home, so its
            # window spans the reach-and-hold stroke instead of the hold alone.
            "criterion.loaded_final_position": (
                "press_and_slide_to_detent",
                "release_into_detent",
                "axial_load_hold",
            ),
        },
    },
    "task.select_second_position": {
        "allowed_input_port_ids": ("thumb_press_and_slide",),
        "default_phase_ids": (
            "press_and_slide_to_detent",
            "release_into_detent",
        ),
        "criterion_phase_ids": {},
    },
    "task.retract_blade": {
        "allowed_input_port_ids": ("thumb_press_and_slide",),
        "default_phase_ids": ("release_home_detent",),
        "criterion_phase_ids": {},
    },
}


def is_utility_knife_task_id(task_id: str) -> bool:
    return task_id in UTILITY_KNIFE_TASK_IDS


def _load_suite() -> dict[str, Any]:
    try:
        raw = _TASK_SUITE_PATH.read_bytes()
    except OSError as exc:
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_SUITE_MISSING",
            "approved auto-lock utility-knife task suite is unavailable",
        ) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != UTILITY_KNIFE_TASK_SUITE_SHA256:
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_SUITE_CHANGED",
            "approved auto-lock utility-knife task suite hash changed",
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_SUITE_INVALID",
            "approved auto-lock utility-knife task suite is invalid JSON",
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("template_id") != UTILITY_KNIFE_TASK_SUITE_ID
        or value.get("version") != UTILITY_KNIFE_TASK_SUITE_VERSION
    ):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_SUITE_IDENTITY_INVALID",
            "approved auto-lock utility-knife task suite identity changed",
        )
    return value


def _finite_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def resolve_utility_knife_task_criterion(
    task_id: str,
    criterion_id: str,
    evaluator_id: str,
) -> UtilityKnifeTaskCriterion | None:
    """Resolve one exact ACTIVE utility-knife criterion, or return None off-family."""

    if not is_utility_knife_task_id(task_id):
        return None
    suite = _load_suite()
    tasks = {
        item.get("task_id"): item
        for item in suite.get("tasks", [])
        if isinstance(item, Mapping)
    }
    task = tasks.get(task_id)
    if not isinstance(task, Mapping):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_UNKNOWN",
            f"utility-knife task {task_id!r} is not in the approved task suite",
        )
    if task.get("runtime_status") != "ACTIVE":
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_NOT_ACTIVE",
            f"utility-knife task {task_id!r} is not ACTIVE",
        )
    criteria = {
        item.get("criterion_id"): item
        for item in task.get("criteria", [])
        if isinstance(item, Mapping) and item.get("disposition") == "PASS"
    }
    criterion = criteria.get(criterion_id)
    if not isinstance(criterion, Mapping):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_CRITERION_UNKNOWN",
            f"criterion {criterion_id!r} is not an approved PASS criterion for {task_id!r}",
        )
    if criterion.get("evaluator_id") != evaluator_id:
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_EVALUATOR_MISMATCH",
            f"criterion {criterion_id!r} does not use evaluator {evaluator_id!r}",
        )
    scenarios = {
        item.get("scenario_id"): item
        for item in suite.get("scenarios", [])
        if isinstance(item, Mapping)
    }
    scenario_id = str(task.get("scenario_id") or "")
    scenario = scenarios.get(scenario_id)
    if (
        not isinstance(scenario, Mapping)
        or scenario.get("runtime_status") != "ACTIVE"
        or task_id not in (scenario.get("task_ids") or [])
    ):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_SCENARIO_NOT_ACTIVE",
            f"utility-knife scenario {scenario_id!r} is not ACTIVE for {task_id!r}",
        )
    tolerance = criterion.get("tolerance")
    if not isinstance(tolerance, Mapping):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_TOLERANCE_INVALID",
            f"criterion {criterion_id!r} has no trusted tolerance",
        )
    runtime = _RUNTIME_POLICY.get(task_id)
    if runtime is None:
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_RUNTIME_POLICY_MISSING",
            f"utility-knife task {task_id!r} has no strict runtime policy",
        )
    phase_ids = runtime["criterion_phase_ids"].get(
        criterion_id,
        runtime["default_phase_ids"],
    )
    return UtilityKnifeTaskCriterion(
        task_id=task_id,
        scenario_id=scenario_id,
        criterion_id=criterion_id,
        evaluator_kind=str(criterion.get("evaluator_kind") or ""),
        evaluator_id=evaluator_id,
        observable_quantity=str(criterion.get("observable_quantity") or ""),
        comparator=str(criterion.get("comparator") or ""),
        expected=criterion.get("expected"),
        unit=str(criterion.get("unit") or ""),
        tolerance_mode=str(tolerance.get("mode") or ""),
        tolerance_value=(
            float(tolerance["value"])
            if isinstance(tolerance.get("value"), (int, float))
            and not isinstance(tolerance.get("value"), bool)
            else None
        ),
        test_program=str(scenario.get("test_program") or ""),
        test_load_case_id=str(scenario.get("test_load_case_id") or ""),
        required_branches=tuple(str(item) for item in scenario.get("required_branches") or ()),
        evaluation_phase_ids=tuple(str(item) for item in phase_ids),
        allowed_input_port_ids=tuple(runtime["allowed_input_port_ids"]),
        reference_policy="default",
    )


def validate_utility_knife_requirement_entry(
    entry: Mapping[str, Any],
    criterion: UtilityKnifeTaskCriterion,
) -> None:
    """Reject any external utility-knife requirement that changes the approved suite."""

    exact = {
        "observable_quantity": criterion.observable_quantity,
        "test_program": criterion.test_program,
        "test_load_case_id": criterion.test_load_case_id,
        "required_branches": list(criterion.required_branches),
    }
    for field, expected in exact.items():
        if entry.get(field) != expected:
            raise UtilityKnifeTaskSuiteError(
                "UTILITY_KNIFE_TASK_REQUIREMENT_MISMATCH",
                f"utility-knife requirement changes approved {field}",
            )
    target = entry.get("target")
    if not isinstance(target, Mapping):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_REQUIREMENT_MISMATCH",
            "utility-knife requirement target is missing",
        )
    for field, expected in {
        "comparator": criterion.comparator,
        "value": criterion.expected,
        "unit": criterion.unit,
    }.items():
        if not _finite_equal(target.get(field), expected):
            raise UtilityKnifeTaskSuiteError(
                "UTILITY_KNIFE_TASK_REQUIREMENT_MISMATCH",
                f"utility-knife requirement changes approved target {field}",
            )
    expected_tolerance = None
    if criterion.tolerance_mode == "absolute":
        expected_tolerance = criterion.tolerance_value
    elif criterion.tolerance_mode == "relative_percent":
        expected_tolerance = (
            abs(float(criterion.expected))
            * float(criterion.tolerance_value)
            / 100.0
        )
    elif criterion.tolerance_mode == "shortfall":
        expected_tolerance = criterion.tolerance_value
    actual_tolerance = target.get("tolerance")
    if expected_tolerance is None:
        if actual_tolerance is not None:
            raise UtilityKnifeTaskSuiteError(
                "UTILITY_KNIFE_TASK_REQUIREMENT_MISMATCH",
                "utility-knife requirement adds an unapproved tolerance",
            )
    elif not _finite_equal(actual_tolerance, expected_tolerance):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_REQUIREMENT_MISMATCH",
            "utility-knife requirement changes the approved tolerance",
        )


def verify_utility_knife_task_suite_binding(
    requirement: Mapping[str, Any],
) -> UtilityKnifeTaskCriterion | None:
    binding = requirement.get("task_suite_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_BINDING_INVALID",
            "utility-knife task-suite binding must be an object",
        )
    criterion = resolve_utility_knife_task_criterion(
        str(requirement.get("task_id") or ""),
        str(requirement.get("requirement_id") or ""),
        str(requirement.get("metric_id") or ""),
    )
    if criterion is None or dict(binding) != criterion.binding():
        raise UtilityKnifeTaskSuiteError(
            "UTILITY_KNIFE_TASK_BINDING_CHANGED",
            "saved utility-knife task-suite binding differs from the approved template",
        )
    return criterion
