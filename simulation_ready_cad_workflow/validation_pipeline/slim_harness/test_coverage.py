"""First-pass execution opportunities and their reserved author budget.

This ledger schedules work only. It never supplies physical evidence or PASS.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from ..mechanism.io import write_json_atomic


class FirstPassReserveReached(RuntimeError):
    """Stop author work while there is still budget for initial test coverage."""
    def __init__(self,message,session_id=None):
        super().__init__(message)
        self.session_id=session_id


def _read(project):
    path=Path(project)/"first-pass-coverage.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"visits":[]}


def planned_tests(project):
    state_path=Path(project)/"workflow.json"
    if not state_path.is_file():return None
    state=json.loads(state_path.read_text(encoding="utf-8"))
    path=Path(project)/"runs"/f"r{int(state.get('revision',1)):02d}"/"design"/"scene.json"
    try:
        value=json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError,ValueError):value={}
    if not isinstance(value,dict):value={}
    tests=value.get("tests")
    if tests is None and (Path(project)/"frozen_experiment.json").is_file():
        tests=json.loads((Path(project)/"frozen_experiment.json").read_text(encoding="utf-8"))["tests"]
    return tests if isinstance(tests,list) and tests and all(isinstance(test,dict) and isinstance(test.get("id"),str) and test["id"] for test in tests) else None


def unvisited(project, tests):
    visits=_read(project)["visits"]
    return [test for test in tests if not any(row["test"]==test and row.get("execution_started") for row in visits)]


def ordered_tests(project, tests):
    pending=unvisited(project,tests)
    return [*pending,*[test for test in tests if test not in pending]]


def record_visit(project,test,outcome,attempt,execution_started):
    ledger=_read(project)
    if any(row["test"]==test and row.get("execution_started") for row in ledger["visits"]):return
    ledger["schema_version"]="slim_first_pass_coverage_v1"
    ledger["visits"].append({"test":test,"outcome":outcome,"attempt":str(attempt),
        "execution_started":bool(execution_started),"at":datetime.now(timezone.utc).isoformat()})
    write_json_atomic(Path(project)/"first-pass-coverage.json",ledger)


def reserve_seconds(project,tier):
    if tier is None:return 0.0
    tests=planned_tests(project)
    count=len(unvisited(project,tests)) if tests is not None else tier.max_tasks_per_design
    return count*tier.simulation_timeout_seconds


def author_allowance(project,tier,remaining):
    return max(0.0,remaining-reserve_seconds(project,tier))


def coverage_summary(tests, rows, pending):
    return {"policy":"first_pass_before_repair", "declared_test_ids":[test["id"] for test in tests],
            "visited_test_ids":[row["test_id"] for row in rows], "pending_test_ids":list(pending),
            "all_tests_visited":bool(tests) and not pending,
            "not_run_test_ids":[*[row["test_id"] for row in rows if not row.get("execution_started")],*pending],
            "started_test_ids":[row["test_id"] for row in rows if row.get("execution_started")],
            "pending_completion_test_ids":[*[row["test_id"] for row in rows if not row.get("trace_file")],*pending],
            "outcomes":{row["test_id"]:row.get("execution_status","completed") for row in rows}}


def development_pending(project,tests,selected):
    if project is None:return []
    pending=unvisited(project,tests)
    return [test["id"] for test in pending] if selected not in pending else []
