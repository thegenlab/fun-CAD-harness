from __future__ import annotations

"""Strict acceptance bindings for the approved hand-crank egg-beater task suite.

The generic metric registry remains unchanged.  This module is the narrow
entry adapter that lets schema-0.2 requirements cite evaluator IDs owned by
``task_suite.hand_crank_egg_beater`` while freezing the exact approved file,
task, scenario, criterion, runtime inputs, and evaluation window.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


EGG_BEATER_TASK_SUITE_ID = "task_suite.hand_crank_egg_beater"
EGG_BEATER_TASK_SUITE_VERSION = "1.0.0"
EGG_BEATER_TASK_SUITE_SHA256 = (
    "c4fcca4f75c70bcea9b4c01e7e324ada00a03aaa165b0b884c8bfae6d3fe8eca"
)
EGG_BEATER_TASK_METRIC_VERSION = (
    f"{EGG_BEATER_TASK_SUITE_ID}@{EGG_BEATER_TASK_SUITE_VERSION}:"
    f"{EGG_BEATER_TASK_SUITE_SHA256}"
)
EGG_BEATER_TASK_SUITE_EVALUATOR_IDS = frozenset(
    {
        "egg_beater_outputs_opposite_directions",
        "egg_beater_output_speed_magnitude_ratio",
        "egg_beater_crank_speed_magnitude_ratio",
        "egg_beater_axis_bound_direction_consistency",
    }
)
EGG_BEATER_TASK_IDS = frozenset(
    {
        "task.rotate_left_beater",
        "task.rotate_right_beater",
        "task.counter_rotate_beaters",
        "task.speed_up_from_crank",
    }
)

_TASK_SUITE_PATH = (
    Path(__file__).resolve().parents[1]
    / "planning_templates"
    / "task_suites"
    / EGG_BEATER_TASK_SUITE_ID
    / f"{EGG_BEATER_TASK_SUITE_VERSION}.json"
)


class EggBeaterTaskSuiteError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class EggBeaterTaskCriterion:
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
                raise EggBeaterTaskSuiteError(
                    "EGG_BEATER_TASK_TOLERANCE_INVALID",
                    "egg-beater shortfall tolerance requires an at_least criterion",
                )
            value = float(value) - float(self.tolerance_value)
        return {
            "comparator": self.comparator,
            "value": value,
            "unit": self.unit,
            **({"tolerance": tolerance} if tolerance is not None else {}),
        }

    def binding(self) -> dict[str, Any]:
        return {
            "template_id": EGG_BEATER_TASK_SUITE_ID,
            "version": EGG_BEATER_TASK_SUITE_VERSION,
            "sha256": EGG_BEATER_TASK_SUITE_SHA256,
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
    "task.rotate_left_beater": {
        "allowed_input_port_ids": ("hand_crank_torque",),
        "default_phase_ids": ("crank_forward",),
        "criterion_phase_ids": {},
    },
    "task.rotate_right_beater": {
        "allowed_input_port_ids": ("hand_crank_torque",),
        "default_phase_ids": ("crank_forward",),
        "criterion_phase_ids": {},
    },
    "task.counter_rotate_beaters": {
        "allowed_input_port_ids": ("hand_crank_torque",),
        "default_phase_ids": ("crank_forward",),
        "criterion_phase_ids": {},
    },
    "task.speed_up_from_crank": {
        "allowed_input_port_ids": ("hand_crank_torque",),
        "default_phase_ids": ("crank_forward",),
        "criterion_phase_ids": {},
    },
}


def is_egg_beater_task_id(task_id: str) -> bool:
    return task_id in EGG_BEATER_TASK_IDS


def _load_suite() -> dict[str, Any]:
    try:
        raw = _TASK_SUITE_PATH.read_bytes()
    except OSError as exc:
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_SUITE_MISSING",
            "approved hand-crank egg-beater task suite is unavailable",
        ) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != EGG_BEATER_TASK_SUITE_SHA256:
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_SUITE_CHANGED",
            "approved hand-crank egg-beater task suite hash changed",
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_SUITE_INVALID",
            "approved hand-crank egg-beater task suite is invalid JSON",
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("template_id") != EGG_BEATER_TASK_SUITE_ID
        or value.get("version") != EGG_BEATER_TASK_SUITE_VERSION
    ):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_SUITE_IDENTITY_INVALID",
            "approved hand-crank egg-beater task suite identity changed",
        )
    return value


def _finite_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def resolve_egg_beater_task_criterion(
    task_id: str,
    criterion_id: str,
    evaluator_id: str,
) -> EggBeaterTaskCriterion | None:
    """Resolve one exact ACTIVE egg-beater criterion, or return None off-family."""

    if not is_egg_beater_task_id(task_id):
        return None
    suite = _load_suite()
    tasks = {
        item.get("task_id"): item
        for item in suite.get("tasks", [])
        if isinstance(item, Mapping)
    }
    task = tasks.get(task_id)
    if not isinstance(task, Mapping):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_UNKNOWN",
            f"egg-beater task {task_id!r} is not in the approved task suite",
        )
    if task.get("runtime_status") != "ACTIVE":
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_NOT_ACTIVE",
            f"egg-beater task {task_id!r} is not ACTIVE",
        )
    criteria = {
        item.get("criterion_id"): item
        for item in task.get("criteria", [])
        if isinstance(item, Mapping) and item.get("disposition") == "PASS"
    }
    criterion = criteria.get(criterion_id)
    if not isinstance(criterion, Mapping):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_CRITERION_UNKNOWN",
            f"criterion {criterion_id!r} is not an approved PASS criterion for {task_id!r}",
        )
    if criterion.get("evaluator_id") != evaluator_id:
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_EVALUATOR_MISMATCH",
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
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_SCENARIO_NOT_ACTIVE",
            f"egg-beater scenario {scenario_id!r} is not ACTIVE for {task_id!r}",
        )
    tolerance = criterion.get("tolerance")
    if not isinstance(tolerance, Mapping):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_TOLERANCE_INVALID",
            f"criterion {criterion_id!r} has no trusted tolerance",
        )
    runtime = _RUNTIME_POLICY.get(task_id)
    if runtime is None:
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_RUNTIME_POLICY_MISSING",
            f"egg-beater task {task_id!r} has no strict runtime policy",
        )
    phase_ids = runtime["criterion_phase_ids"].get(
        criterion_id,
        runtime["default_phase_ids"],
    )
    return EggBeaterTaskCriterion(
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


def validate_egg_beater_requirement_entry(
    entry: Mapping[str, Any],
    criterion: EggBeaterTaskCriterion,
) -> None:
    """Reject any external egg-beater requirement that changes the approved suite."""

    exact = {
        "observable_quantity": criterion.observable_quantity,
        "test_program": criterion.test_program,
        "test_load_case_id": criterion.test_load_case_id,
        "required_branches": list(criterion.required_branches),
    }
    for field, expected in exact.items():
        if entry.get(field) != expected:
            raise EggBeaterTaskSuiteError(
                "EGG_BEATER_TASK_REQUIREMENT_MISMATCH",
                f"egg-beater requirement changes approved {field}",
            )
    target = entry.get("target")
    if not isinstance(target, Mapping):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_REQUIREMENT_MISMATCH",
            "egg-beater requirement target is missing",
        )
    for field, expected in {
        "comparator": criterion.comparator,
        "value": criterion.expected,
        "unit": criterion.unit,
    }.items():
        if not _finite_equal(target.get(field), expected):
            raise EggBeaterTaskSuiteError(
                "EGG_BEATER_TASK_REQUIREMENT_MISMATCH",
                f"egg-beater requirement changes approved target {field}",
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
            raise EggBeaterTaskSuiteError(
                "EGG_BEATER_TASK_REQUIREMENT_MISMATCH",
                "egg-beater requirement adds an unapproved tolerance",
            )
    elif not _finite_equal(actual_tolerance, expected_tolerance):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_REQUIREMENT_MISMATCH",
            "egg-beater requirement changes the approved tolerance",
        )


def verify_egg_beater_task_suite_binding(
    requirement: Mapping[str, Any],
) -> EggBeaterTaskCriterion | None:
    binding = requirement.get("task_suite_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_BINDING_INVALID",
            "egg-beater task-suite binding must be an object",
        )
    criterion = resolve_egg_beater_task_criterion(
        str(requirement.get("task_id") or ""),
        str(requirement.get("requirement_id") or ""),
        str(requirement.get("metric_id") or ""),
    )
    if criterion is None or dict(binding) != criterion.binding():
        raise EggBeaterTaskSuiteError(
            "EGG_BEATER_TASK_BINDING_CHANGED",
            "saved egg-beater task-suite binding differs from the approved template",
        )
    return criterion
