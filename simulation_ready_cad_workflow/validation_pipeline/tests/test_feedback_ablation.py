"""Tests for feedback degradation used in ablation experiments."""
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.feedback_ablation import transform_feedback

SAMPLE_RESULT = {
    "status": "NEEDS_REVISION",
    "issues": ["Template tasks or original-source motion checks did not pass"],
    "tests": [
        {
            "test_id": "loaded_roll_brake_fold",
            "execution_started": True,
            "physical_pass": False,
            "execution_status": "completed",
            "assertions": [
                {"id": "payload", "passed": True, "expression": "max(...) <= 0.05", "observed_operands": [0.0, 0.05]},
                {"id": "brake_motion", "passed": False, "expression": "span(...) <= 0.02", "observed_operands": [0.087, 0.02]},
            ],
            "failure": {"id": "loaded_roll_brake_fold", "status": "completed", "error": "assertion_failures"},
            "geometry_audit": {"passed": False, "status": "not_completed_for_rejected_test"},
            "trace_file": "/some/path/native-trace.json",
        },
        {
            "test_id": "stability_check",
            "execution_started": True,
            "physical_pass": True,
            "execution_status": "completed",
            "assertions": [{"id": "stable", "passed": True}],
            "failure": None,
            "geometry_audit": {"passed": True},
            "trace_file": "/some/path/trace2.json",
        },
    ],
    "failure_cause": "not_uniquely_determined",
    "available_repair_strategies": ["modify_mechanics", "simplify_simulation"],
    "budget_timing": {
        "total_seconds": 10800.0,
        "active_seconds": 2049.0,
        "remaining_seconds": 8751.0,
    },
    "test_coverage": {"all_tests_visited": False},
    "attempt": "/some/path/a001",
}


def test_full_mode_passthrough():
    assert transform_feedback(SAMPLE_RESULT, "full") is SAMPLE_RESULT


def test_none_mode_passthrough():
    assert transform_feedback(SAMPLE_RESULT, None) is SAMPLE_RESULT


def test_binary_strips_assertions():
    result = transform_feedback(SAMPLE_RESULT, "binary")
    assert result["status"] == "NEEDS_REVISION"
    assert result["tests"] == [
        {"test_id": "loaded_roll_brake_fold", "passed": False},
        {"test_id": "stability_check", "passed": True},
    ]
    assert result["budget_timing"] == SAMPLE_RESULT["budget_timing"]
    assert "assertions" not in str(result)
    assert "failure_cause" not in result
    assert "available_repair_strategies" not in result
    assert "issues" not in result
    assert "geometry_audit" not in str(result["tests"])


def test_binary_no_budget_timing():
    no_budget = {k: v for k, v in SAMPLE_RESULT.items() if k != "budget_timing"}
    result = transform_feedback(no_budget, "binary")
    assert "budget_timing" not in result
    assert result["status"] == "NEEDS_REVISION"
    assert len(result["tests"]) == 2


def test_binary_empty_tests():
    empty = {"status": "NEEDS_REVISION", "issues": ["geometry errors"]}
    result = transform_feedback(empty, "binary")
    assert result == {"status": "NEEDS_REVISION", "tests": []}


def test_binary_with_completed_tests_fallback():
    """When 'tests' is absent but 'completed_tests' exists, binary uses it."""
    alt = {
        "status": "NEEDS_REVISION",
        "completed_tests": [{"test_id": "t1", "physical_pass": True}],
    }
    result = transform_feedback(alt, "binary")
    assert result["tests"] == [{"test_id": "t1", "passed": True}]


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError, match="Unknown feedback mode"):
        transform_feedback(SAMPLE_RESULT, "raw")
