from __future__ import annotations

"""Strict acceptance bindings for the approved mechanical-clock task suite.

The generic metric registry remains unchanged.  This module is the narrow
entry adapter that lets schema-0.2 requirements cite evaluator IDs owned by
``task_suite.mechanical_alarm_clock`` while freezing the exact approved file,
task, scenario, criterion, runtime inputs, and evaluation window.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


CLOCK_TASK_SUITE_ID = "task_suite.mechanical_alarm_clock"
CLOCK_TASK_SUITE_VERSION = "1.0.0"
CLOCK_TASK_SUITE_SHA256 = (
    "defd211367257ac5595eea988b94bf34efd84a55d08eb9b04c01a407a3276228"
)
CLOCK_TASK_METRIC_VERSION = (
    f"{CLOCK_TASK_SUITE_ID}@{CLOCK_TASK_SUITE_VERSION}:"
    f"{CLOCK_TASK_SUITE_SHA256}"
)
CLOCK_TASK_SUITE_EVALUATOR_IDS = frozenset(
    {
        "clock.display_direction_consistent",
        "clock.maximum_hand_drift_rad",
        "clock.alarm_target_latch_error_rad",
        "clock.alarm_target_latch_span_rad",
        "clock.selected_alarm_position_error_rad",
        "clock.alarm_target_drift_rad",
    }
)
CLOCK_TASK_IDS = frozenset(
    {
        "run_three_hand_clock_logic",
        "set_time_manually",
        "ring_alarm_independently",
        "set_alarm_time_by_turning_hour_hand",
        "trigger_alarm_at_selected_hand_position",
    }
)

_TASK_SUITE_PATH = (
    Path(__file__).resolve().parents[1]
    / "planning_templates"
    / "task_suites"
    / CLOCK_TASK_SUITE_ID
    / f"{CLOCK_TASK_SUITE_VERSION}.json"
)


class ClockTaskSuiteError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class ClockTaskCriterion:
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
                raise ClockTaskSuiteError(
                    "CLOCK_TASK_TOLERANCE_INVALID",
                    "clock shortfall tolerance requires an at_least criterion",
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
            "template_id": CLOCK_TASK_SUITE_ID,
            "version": CLOCK_TASK_SUITE_VERSION,
            "sha256": CLOCK_TASK_SUITE_SHA256,
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
    "run_three_hand_clock_logic": {
        "allowed_input_port_ids": ("winding_key_torque",),
        "default_phase_ids": ("wind_and_advance",),
        "criterion_phase_ids": {},
    },
    "set_time_manually": {
        "allowed_input_port_ids": ("manual_setting_torque",),
        "default_phase_ids": ("engage_and_set_forward",),
        "criterion_phase_ids": {},
    },
    "ring_alarm_independently": {
        "allowed_input_port_ids": ("alarm_winding_key_torque",),
        "default_phase_ids": ("wind_and_advance",),
        "criterion_phase_ids": {},
    },
    "set_alarm_time_by_turning_hour_hand": {
        "allowed_input_port_ids": ("alarm_time_by_hour_hand_torque",),
        "default_phase_ids": ("alarm_setting_latched",),
        "criterion_phase_ids": {
            "clock.indirect_alarm_target_actuation": (
                "turn_hour_hand_to_alarm_target",
            ),
        },
    },
    "trigger_alarm_at_selected_hand_position": {
        "allowed_input_port_ids": (
            "alarm_time_by_hour_hand_torque",
            "winding_key_torque",
        ),
        "default_phase_ids": ("advance_to_alarm_time",),
        "criterion_phase_ids": {
            "clock.selected_alarm_position_error": (
                "advance_to_alarm_time",
            ),
            "clock.alarm_target_drift_before_trigger": (
                "advance_to_alarm_time",
            ),
            "clock.pretrigger_hammer_quiet": (
                "turn_hour_hand_to_alarm_target",
                "alarm_setting_latched",
                "restore_hour_hand_to_current_time",
                "alarm_target_ready",
                "advance_to_alarm_time",
            ),
            "clock.posttrigger_hammer_swing": ("ring_with_timekeeping",),
            "clock.posttrigger_minute_hand_travel": (
                "ring_with_timekeeping",
            ),
        },
    },
}


def is_clock_task_id(task_id: str) -> bool:
    return task_id in CLOCK_TASK_IDS


def _load_suite() -> dict[str, Any]:
    try:
        raw = _TASK_SUITE_PATH.read_bytes()
    except OSError as exc:
        raise ClockTaskSuiteError(
            "CLOCK_TASK_SUITE_MISSING",
            "approved mechanical-clock task suite is unavailable",
        ) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != CLOCK_TASK_SUITE_SHA256:
        raise ClockTaskSuiteError(
            "CLOCK_TASK_SUITE_CHANGED",
            "approved mechanical-clock task suite hash changed",
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise ClockTaskSuiteError(
            "CLOCK_TASK_SUITE_INVALID",
            "approved mechanical-clock task suite is invalid JSON",
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("template_id") != CLOCK_TASK_SUITE_ID
        or value.get("version") != CLOCK_TASK_SUITE_VERSION
    ):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_SUITE_IDENTITY_INVALID",
            "approved mechanical-clock task suite identity changed",
        )
    return value


def _finite_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def resolve_clock_task_criterion(
    task_id: str,
    criterion_id: str,
    evaluator_id: str,
) -> ClockTaskCriterion | None:
    """Resolve one exact ACTIVE clock criterion, or return None off-family."""

    if not is_clock_task_id(task_id):
        return None
    suite = _load_suite()
    tasks = {
        item.get("task_id"): item
        for item in suite.get("tasks", [])
        if isinstance(item, Mapping)
    }
    task = tasks.get(task_id)
    if not isinstance(task, Mapping):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_UNKNOWN",
            f"clock task {task_id!r} is not in the approved task suite",
        )
    if task.get("runtime_status") != "ACTIVE":
        raise ClockTaskSuiteError(
            "CLOCK_TASK_NOT_ACTIVE",
            f"clock task {task_id!r} is not ACTIVE",
        )
    criteria = {
        item.get("criterion_id"): item
        for item in task.get("criteria", [])
        if isinstance(item, Mapping) and item.get("disposition") == "PASS"
    }
    criterion = criteria.get(criterion_id)
    if not isinstance(criterion, Mapping):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_CRITERION_UNKNOWN",
            f"criterion {criterion_id!r} is not an approved PASS criterion for {task_id!r}",
        )
    if criterion.get("evaluator_id") != evaluator_id:
        raise ClockTaskSuiteError(
            "CLOCK_TASK_EVALUATOR_MISMATCH",
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
        raise ClockTaskSuiteError(
            "CLOCK_TASK_SCENARIO_NOT_ACTIVE",
            f"clock scenario {scenario_id!r} is not ACTIVE for {task_id!r}",
        )
    tolerance = criterion.get("tolerance")
    if not isinstance(tolerance, Mapping):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_TOLERANCE_INVALID",
            f"criterion {criterion_id!r} has no trusted tolerance",
        )
    runtime = _RUNTIME_POLICY.get(task_id)
    if runtime is None:
        raise ClockTaskSuiteError(
            "CLOCK_TASK_RUNTIME_POLICY_MISSING",
            f"clock task {task_id!r} has no strict runtime policy",
        )
    phase_ids = runtime["criterion_phase_ids"].get(
        criterion_id,
        runtime["default_phase_ids"],
    )
    reference_policy = (
        "distinct_output"
        if criterion.get("evaluator_id") == "angular_ratio"
        else "default"
    )
    return ClockTaskCriterion(
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
        reference_policy=reference_policy,
    )


def validate_clock_requirement_entry(
    entry: Mapping[str, Any],
    criterion: ClockTaskCriterion,
) -> None:
    """Reject any external clock requirement that changes the approved suite."""

    exact = {
        "observable_quantity": criterion.observable_quantity,
        "test_program": criterion.test_program,
        "test_load_case_id": criterion.test_load_case_id,
        "required_branches": list(criterion.required_branches),
    }
    for field, expected in exact.items():
        if entry.get(field) != expected:
            raise ClockTaskSuiteError(
                "CLOCK_TASK_REQUIREMENT_MISMATCH",
                f"clock requirement changes approved {field}",
            )
    target = entry.get("target")
    if not isinstance(target, Mapping):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_REQUIREMENT_MISMATCH",
            "clock requirement target is missing",
        )
    for field, expected in {
        "comparator": criterion.comparator,
        "value": criterion.expected,
        "unit": criterion.unit,
    }.items():
        if not _finite_equal(target.get(field), expected):
            raise ClockTaskSuiteError(
                "CLOCK_TASK_REQUIREMENT_MISMATCH",
                f"clock requirement changes approved target {field}",
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
            raise ClockTaskSuiteError(
                "CLOCK_TASK_REQUIREMENT_MISMATCH",
                "clock requirement adds an unapproved tolerance",
            )
    elif not _finite_equal(actual_tolerance, expected_tolerance):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_REQUIREMENT_MISMATCH",
            "clock requirement changes the approved tolerance",
        )


def verify_clock_task_suite_binding(
    requirement: Mapping[str, Any],
) -> ClockTaskCriterion | None:
    binding = requirement.get("task_suite_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise ClockTaskSuiteError(
            "CLOCK_TASK_BINDING_INVALID",
            "clock task-suite binding must be an object",
        )
    criterion = resolve_clock_task_criterion(
        str(requirement.get("task_id") or ""),
        str(requirement.get("requirement_id") or ""),
        str(requirement.get("metric_id") or ""),
    )
    if criterion is None or dict(binding) != criterion.binding():
        raise ClockTaskSuiteError(
            "CLOCK_TASK_BINDING_CHANGED",
            "saved clock task-suite binding differs from the approved template",
        )
    return criterion
