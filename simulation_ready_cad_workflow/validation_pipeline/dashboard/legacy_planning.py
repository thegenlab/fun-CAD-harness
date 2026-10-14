"""Read-only loader for the pre-ReAct task-planning contract.

Historical runs persisted ``planning/task_planning.json`` through the retired
operator-charter workflow.  The dashboard keeps displaying those contracts,
but it reads them as plain evidence: no operator receipt, signature, or
authority chain is consulted, and nothing is ever rewritten.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .legacy_io import DashboardReadError, read_stable_json_object


TASK_PLANNING_SCHEMA = "simulation_ready_cad_task_planning_v1"
TASK_PLANNING_RELATIVE = Path("planning") / "task_planning.json"
_MAX_CONTRACT_BYTES = 2 * 1024 * 1024


class LegacyPlanningError(ValueError):
    """The historical task-planning artifact cannot be displayed."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def load_legacy_task_planning(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Load one historical planning contract and its authored-test evidence."""

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    destination = run / TASK_PLANNING_RELATIVE
    try:
        contract, _record = read_stable_json_object(
            destination,
            label="legacy task-planning contract",
            allowed_root=run,
            max_bytes=_MAX_CONTRACT_BYTES,
        )
    except (OSError, ValueError, DashboardReadError) as exc:
        raise LegacyPlanningError(
            "TASK_PLANNING_CONTRACT_INVALID",
            f"cannot read task-planning contract: {exc}",
        ) from exc
    if contract.get("schema_version") != TASK_PLANNING_SCHEMA:
        raise LegacyPlanningError(
            "TASK_PLANNING_CONTRACT_INVALID",
            "task-planning contract has an unknown schema_version",
        )
    if (
        not isinstance(contract.get("task_understanding"), dict)
        or not isinstance(contract.get("joint_plan"), list)
        or not isinstance(contract.get("test_plan"), list)
        or not isinstance(contract.get("authored_test_tasks"), list)
        or not isinstance(contract.get("authored_test_task_count"), int)
        or not isinstance(contract.get("task_charter_binding"), dict)
    ):
        raise LegacyPlanningError(
            "TASK_PLANNING_CONTRACT_INVALID",
            "task-planning contract is missing its recorded planning fields",
        )
    evidence = [destination]
    authored_root = (run / "planning" / "authored_tests").resolve()
    for item in contract["authored_test_tasks"]:
        reference = item.get("path") if isinstance(item, dict) else None
        if not isinstance(reference, str) or not reference:
            continue
        candidate = (workspace / reference).resolve()
        try:
            candidate.relative_to(authored_root)
        except ValueError:
            continue
        if candidate.is_file():
            evidence.append(candidate)
    return contract, evidence
