"""Explicit manual audit for the first recorded drawer closed-loop run.

This historical, workspace-dependent check is intentionally outside
``test_*.py`` fixed-suite discovery.  Missing evidence is NOT_AVAILABLE and
returns non-zero; it is never reported as a skipped or passing regression.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import (
    normalize_run,
)


MIGRATED_FIXED_TEST_ID = (
    "simulation_ready_cad_workflow.validation_pipeline.tests.test_dashboard."
    "DashboardFinalPassIntegrationTests."
    "test_recorded_first_closed_loop_is_auditable_when_present"
)


def audit_recorded_first_closed_loop(
    workspace_root: str | Path,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    run = (
        workspace
        / ".runtime_runs"
        / "simple-drawer-closed-loop-v1"
        / "closed-loop-kimi-r2-final-pass"
    )
    if not run.is_dir():
        return {
            "schema_version": (
                "simulation_ready_cad_manual_dashboard_history_audit_v1"
            ),
            "status": "NOT_AVAILABLE",
            "migrated_fixed_test_id": MIGRATED_FIXED_TEST_ID,
            "run": str(run),
            "reason": "recorded first closed-loop run is not present",
        }
    data = normalize_run(run, workspace_root=workspace)
    checks = {
        "status_passed": data["run"]["status"] == "passed",
        "default_stage_whole_machine": (
            data["run"]["default_stage_id"] == "whole-machine"
        ),
        "task_count": data["whole_machine"]["task_passed"] == 2,
        "all_stage_errors_pass": all(
            item["error_pass"] for item in data["whole_machine"]["stages"]
        ),
        "all_stage_durations_pass": all(
            item["duration_pass"]
            for item in data["whole_machine"]["stages"]
        ),
        "all_stage_forces_pass": all(
            item["force_pass"] for item in data["whole_machine"]["stages"]
        ),
    }
    return {
        "schema_version": (
            "simulation_ready_cad_manual_dashboard_history_audit_v1"
        ),
        "status": "PASSED" if all(checks.values()) else "FAILED",
        "migrated_fixed_test_id": MIGRATED_FIXED_TEST_ID,
        "run": str(run),
        "checks": checks,
    }


def main() -> int:
    workspace = Path(__file__).resolve().parents[3]
    result = audit_recorded_first_closed_loop(workspace)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASSED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
