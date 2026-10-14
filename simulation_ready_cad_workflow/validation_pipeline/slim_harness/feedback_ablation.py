"""Feedback degradation for ablation experiments."""
from __future__ import annotations


def transform_feedback(result: dict, mode: str | None) -> dict:
    """Return a degraded copy of the revision result for the given feedback mode.

    ``full`` (or ``None``) passes the result through unchanged.
    ``binary`` keeps only the top-level status, per-test pass/fail, and budget timing.
    """
    if mode is None or mode == "full":
        return result
    if mode == "binary":
        transformed: dict = {"status": result.get("status")}
        tests = result.get("tests") or result.get("completed_tests") or []
        transformed["tests"] = [
            {"test_id": t.get("test_id"), "passed": bool(t.get("physical_pass"))}
            for t in tests
            if isinstance(t, dict) and "test_id" in t
        ]
        if "budget_timing" in result:
            transformed["budget_timing"] = result["budget_timing"]
        return transformed
    raise ValueError(f"Unknown feedback mode: {mode!r}")
