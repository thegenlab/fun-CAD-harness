from __future__ import annotations

"""Strict acceptance bindings for the approved spring kitchen-tongs task suite.

The generic metric registry remains unchanged.  This module is the narrow
entry adapter that lets schema-0.2 requirements cite evaluator IDs owned by
``task_suite.spring_kitchen_tongs`` while freezing the exact approved file,
task, scenario, criterion, runtime inputs, and evaluation window.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


TONGS_TASK_SUITE_ID = "task_suite.spring_kitchen_tongs"
TONGS_TASK_SUITE_VERSION = "1.0.0"
TONGS_TASK_SUITE_SHA256 = (
    "07d98533258e0dff05e29720df53bfd01333714ceb3f8817815b18b4ec3d845b"
)
TONGS_TASK_METRIC_VERSION = (
    f"{TONGS_TASK_SUITE_ID}@{TONGS_TASK_SUITE_VERSION}:"
    f"{TONGS_TASK_SUITE_SHA256}"
)
TONGS_TASK_SUITE_EVALUATOR_IDS = frozenset(
    {
        "tongs.hold_duration_s",
        "tongs.release_input_zero",
    }
)
TONGS_TASK_IDS = frozenset(
    {
        "tongs.pickup_standard_block",
        "tongs.hold_standard_block",
        "tongs.release_standard_block",
    }
)

_TASK_SUITE_PATH = (
    Path(__file__).resolve().parents[1]
    / "planning_templates"
    / "task_suites"
    / TONGS_TASK_SUITE_ID
    / f"{TONGS_TASK_SUITE_VERSION}.json"
)


class TongsTaskSuiteError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class TongsTaskCriterion:
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
                raise TongsTaskSuiteError(
                    "TONGS_TASK_TOLERANCE_INVALID",
                    "tongs shortfall tolerance requires an at_least criterion",
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
            "template_id": TONGS_TASK_SUITE_ID,
            "version": TONGS_TASK_SUITE_VERSION,
            "sha256": TONGS_TASK_SUITE_SHA256,
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
    # Phase IDs mirror mechanism.loads.default_revolute_load_cases for the
    # axis-spring tabletop pickup fixture.  The default window is the powered
    # grip-and-lift stroke; point-in-time state checks and the settle-to-lift
    # displacement freeze the exact phases they measure instead.
    "tongs.pickup_standard_block": {
        "allowed_input_port_ids": ("hand_squeeze_torque",),
        "default_phase_ids": (
            "tong_pickup_close",
            "tong_pickup_grip_verify",
            "tong_pickup_lift",
            "tong_pickup_verify",
        ),
        "criterion_phase_ids": {
            "tongs.block_starts_on_table": ("tong_pickup_initial_table_settle",),
            "tongs.bilateral_grip": ("tong_pickup_verify",),
            "tongs.minimum_pickup_lift": (
                "tong_pickup_initial_table_settle",
                "tong_pickup_approach",
                "tong_pickup_close",
                "tong_pickup_grip_verify",
                "tong_pickup_lift",
                "tong_pickup_verify",
            ),
            "tongs.block_clears_table": ("tong_pickup_verify",),
        },
    },
    "tongs.hold_standard_block": {
        "allowed_input_port_ids": ("hand_squeeze_torque",),
        "default_phase_ids": ("tong_hold_verify",),
        "criterion_phase_ids": {},
    },
    "tongs.release_standard_block": {
        "allowed_input_port_ids": ("hand_squeeze_torque",),
        "default_phase_ids": ("tong_release_observe",),
        "criterion_phase_ids": {
            # The passive-release verification window starts only after the
            # ramped hand input reaches zero; the zero-input proof and the
            # spring return both span the passive tail of the program.
            "tongs.hand_input_zero_during_release": (
                "tong_release_passive_return",
                "tong_release_final_table_settle",
                "tong_release_observe",
            ),
            "tongs.joint_returns_to_zero": (
                "tong_release_passive_return",
                "tong_release_final_table_settle",
                "tong_release_observe",
            ),
        },
    },
}


def is_tongs_task_id(task_id: str) -> bool:
    return task_id in TONGS_TASK_IDS


def _load_suite() -> dict[str, Any]:
    try:
        raw = _TASK_SUITE_PATH.read_bytes()
    except OSError as exc:
        raise TongsTaskSuiteError(
            "TONGS_TASK_SUITE_MISSING",
            "approved spring kitchen-tongs task suite is unavailable",
        ) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != TONGS_TASK_SUITE_SHA256:
        raise TongsTaskSuiteError(
            "TONGS_TASK_SUITE_CHANGED",
            "approved spring kitchen-tongs task suite hash changed",
        )
    try:
        value = json.loads(raw)
    except (UnicodeError, ValueError) as exc:
        raise TongsTaskSuiteError(
            "TONGS_TASK_SUITE_INVALID",
            "approved spring kitchen-tongs task suite is invalid JSON",
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("template_id") != TONGS_TASK_SUITE_ID
        or value.get("version") != TONGS_TASK_SUITE_VERSION
    ):
        raise TongsTaskSuiteError(
            "TONGS_TASK_SUITE_IDENTITY_INVALID",
            "approved spring kitchen-tongs task suite identity changed",
        )
    return value


def _finite_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-15)
    return left == right


def resolve_tongs_task_criterion(
    task_id: str,
    criterion_id: str,
    evaluator_id: str,
) -> TongsTaskCriterion | None:
    """Resolve one exact ACTIVE tongs criterion, or return None off-family."""

    if not is_tongs_task_id(task_id):
        return None
    suite = _load_suite()
    tasks = {
        item.get("task_id"): item
        for item in suite.get("tasks", [])
        if isinstance(item, Mapping)
    }
    task = tasks.get(task_id)
    if not isinstance(task, Mapping):
        raise TongsTaskSuiteError(
            "TONGS_TASK_UNKNOWN",
            f"tongs task {task_id!r} is not in the approved task suite",
        )
    if task.get("runtime_status") != "ACTIVE":
        raise TongsTaskSuiteError(
            "TONGS_TASK_NOT_ACTIVE",
            f"tongs task {task_id!r} is not ACTIVE",
        )
    criteria = {
        item.get("criterion_id"): item
        for item in task.get("criteria", [])
        if isinstance(item, Mapping) and item.get("disposition") == "PASS"
    }
    criterion = criteria.get(criterion_id)
    if not isinstance(criterion, Mapping):
        raise TongsTaskSuiteError(
            "TONGS_TASK_CRITERION_UNKNOWN",
            f"criterion {criterion_id!r} is not an approved PASS criterion for {task_id!r}",
        )
    if criterion.get("evaluator_id") != evaluator_id:
        raise TongsTaskSuiteError(
            "TONGS_TASK_EVALUATOR_MISMATCH",
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
        raise TongsTaskSuiteError(
            "TONGS_TASK_SCENARIO_NOT_ACTIVE",
            f"tongs scenario {scenario_id!r} is not ACTIVE for {task_id!r}",
        )
    tolerance = criterion.get("tolerance")
    if not isinstance(tolerance, Mapping):
        raise TongsTaskSuiteError(
            "TONGS_TASK_TOLERANCE_INVALID",
            f"criterion {criterion_id!r} has no trusted tolerance",
        )
    runtime = _RUNTIME_POLICY.get(task_id)
    if runtime is None:
        raise TongsTaskSuiteError(
            "TONGS_TASK_RUNTIME_POLICY_MISSING",
            f"tongs task {task_id!r} has no strict runtime policy",
        )
    phase_ids = runtime["criterion_phase_ids"].get(
        criterion_id,
        runtime["default_phase_ids"],
    )
    return TongsTaskCriterion(
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


def validate_tongs_requirement_entry(
    entry: Mapping[str, Any],
    criterion: TongsTaskCriterion,
) -> None:
    """Reject any external tongs requirement that changes the approved suite."""

    exact = {
        "observable_quantity": criterion.observable_quantity,
        "test_program": criterion.test_program,
        "test_load_case_id": criterion.test_load_case_id,
        "required_branches": list(criterion.required_branches),
    }
    for field, expected in exact.items():
        if entry.get(field) != expected:
            raise TongsTaskSuiteError(
                "TONGS_TASK_REQUIREMENT_MISMATCH",
                f"tongs requirement changes approved {field}",
            )
    target = entry.get("target")
    if not isinstance(target, Mapping):
        raise TongsTaskSuiteError(
            "TONGS_TASK_REQUIREMENT_MISMATCH",
            "tongs requirement target is missing",
        )
    for field, expected in {
        "comparator": criterion.comparator,
        "value": criterion.expected,
        "unit": criterion.unit,
    }.items():
        if not _finite_equal(target.get(field), expected):
            raise TongsTaskSuiteError(
                "TONGS_TASK_REQUIREMENT_MISMATCH",
                f"tongs requirement changes approved target {field}",
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
            raise TongsTaskSuiteError(
                "TONGS_TASK_REQUIREMENT_MISMATCH",
                "tongs requirement adds an unapproved tolerance",
            )
    elif not _finite_equal(actual_tolerance, expected_tolerance):
        raise TongsTaskSuiteError(
            "TONGS_TASK_REQUIREMENT_MISMATCH",
            "tongs requirement changes the approved tolerance",
        )


def verify_tongs_task_suite_binding(
    requirement: Mapping[str, Any],
) -> TongsTaskCriterion | None:
    binding = requirement.get("task_suite_binding")
    if binding is None:
        return None
    if not isinstance(binding, Mapping):
        raise TongsTaskSuiteError(
            "TONGS_TASK_BINDING_INVALID",
            "tongs task-suite binding must be an object",
        )
    criterion = resolve_tongs_task_criterion(
        str(requirement.get("task_id") or ""),
        str(requirement.get("requirement_id") or ""),
        str(requirement.get("metric_id") or ""),
    )
    if criterion is None or dict(binding) != criterion.binding():
        raise TongsTaskSuiteError(
            "TONGS_TASK_BINDING_CHANGED",
            "saved tongs task-suite binding differs from the approved template",
        )
    return criterion
