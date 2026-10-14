"""Authored post-failure choice: revise, retune execution, relax an adjustable task, or stop.

Root calls these helpers. This module does not classify failures, assign an
action from remaining time, mutate workflow.json, or rewrite PASS/result
labels. Frozen benchmarks stay non-adjustable.
"""
from __future__ import annotations

import copy
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cold_start.worker import RESPONSE_SCHEMA
from ..mechanism.io import write_json_atomic
from .budget import AdjustmentLog, CriteriaAdjustment
from .trace_storage import file_digest


BUDGET_ACTIONS = (
    "revise_mechanism",
    "adjust_execution",
    "adjust_task",
    "finish_partial",
)
CRITERIA_LOCK_NAME = "criteria.lock.json"
# Codex rejects a turn input above 1,048,576 characters; a revision prompt may carry the feedback twice.
FEEDBACK_INLINE_CHARS = 200_000

def initial_task_criteria(task: str) -> list[dict[str, str]]:
    """Bind the complete user request without inventing or reviewing initial targets."""
    return [{"id": "task", "kind": "function", "statement": task,
             "source_quote": task, "verification": "Author-defined simulation tasks and program hard checks."}]


def task_definition(task_text: str, scene: dict) -> dict:
    return {"task_text": task_text, "tests": copy.deepcopy(scene.get("tests", [])),
            "gravity_m_s2": copy.deepcopy(scene.get("gravity_m_s2", [0, -9.81, 0]))}


def review_task_change(reviewer, author, workspace, project, design, turn_dir,
                       proposed, *, reason="Submitted simulation task changed", context=None,
                       tier_name="light", budget=None):
    """Review only changed task definitions, then hand all available advice to the author."""
    from .task_change_review import run_change_review
    project, turn_dir = Path(project), Path(turn_dir)
    baseline_path = project / "simulation-task-baseline.json"
    if not baseline_path.is_file():
        write_json_atomic(baseline_path, proposed)
        return {"status": "ready", "accepted": True, "reviewed": False, "reason": "Initial task baseline"}
    before = json.loads(baseline_path.read_text(encoding="utf-8"))
    if before == proposed:
        return {"status": "ready", "accepted": True, "reviewed": False, "reason": "Simulation task unchanged"}
    turn_dir.mkdir(parents=True, exist_ok=True)
    change = {"before": before, "proposed": proposed}
    cache_path = project / "last-task-change-review.json"
    cached = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.is_file() else {}
    if cached.get("change") == change:
        advice = cached["advice"]
    else:
        prompt = (
            "Review ONLY whether this simulation-task change is reasonable, given the original machine function, "
            "failure feedback and remaining time. Explain whether the new loads, duration, inputs and assertions "
            "still test a meaningful function or merely hide failure. Do not audit geometry, delivery, source files, "
            "Harness enforcement, or final results. Work from the concrete before/after definitions below; "
            "do not explore the repository or launch tools unless essential to interpreting this change. "
            "Return your conclusion immediately within the time limit. Your advice does not issue PASS or approval; "
            "the author decides and all code hard gates remain binding.\n"
            f"AUTHOR REASON: {reason}\nCONTEXT: {json.dumps(context or {}, ensure_ascii=False)}\n"
            f"TASK CHANGE: {json.dumps(change, ensure_ascii=False)}"
        )
        advice = run_change_review(reviewer, Path(workspace), prompt, turn_dir / "reviewer",
                                   tier_name=tier_name, budget=budget)
        write_json_atomic(cache_path, {"change": change, "advice": advice})
    write_json_atomic(turn_dir / "reviewer-output.json", advice)
    schema = copy.deepcopy(RESPONSE_SCHEMA)
    schema["properties"]["task_change_decision"] = {
        "type": "object", "additionalProperties": False, "required": ["action", "reason"],
        "properties": {"action": {"type": "string", "enum": ["accept", "revise", "retain"]},
                       "reason": {"type": "string"}}}
    schema["required"].append("task_change_decision")
    prompt = (
        "Here is the task-change reviewer's entire available output, including any timeout or partial conclusion. "
        "Judge for yourself whether the changed simulation task is reasonable and can count as satisfied if "
        "its physical tests pass. Choose accept, revise, or retain the previous task, and explain why. "
        "Reviewer disagreement or timeout is advisory, not a veto. You cannot override failed program physics, "
        "geometry, task/part or time-limit checks. Do not edit files in this decision turn; if changes are needed "
        "choose revise or retain and the normal author revision follows.\n"
        f"CURRENT REMAINING ACTIVE SECONDS: {budget.tracker.remaining_active if budget is not None and budget.tracker is not None else (context or {}).get('remaining_active_seconds')}\n"
        f"TASK CHANGE: {json.dumps(change, ensure_ascii=False)}\n"
        f"REVIEWER OUTPUT (UNMODIFIED): {json.dumps(advice, ensure_ascii=False)}"
    )
    author_dir = turn_dir / "author-decision"
    author_dir.mkdir(parents=True, exist_ok=True)
    (author_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    response = author.run_turn(Path(design), prompt, author_dir, 1800, response_schema=schema)
    decision = response.get("task_change_decision", {})
    accepted = response.get("status") == "ready" and decision.get("action") == "accept" and bool(decision.get("reason", "").strip())
    record = {"status": "ready" if accepted else "needs_revision", "accepted": accepted,
              "reviewed": True, "change": change, "advice": advice, "author_response": response}
    write_json_atomic(turn_dir / "task-change-decision.json", record)
    return record


def adjustment_allowed(state: Any) -> bool:
    """True only for an explicit adjustable, non-frozen request."""
    request = _request(state)
    if not isinstance(request, dict):
        return False
    if request.get("task_adjustable") is not True:
        return False
    if request.get("frozen_experiment_sha256"):
        return False
    return True


def revision_response_schema() -> dict[str, Any]:
    """Copy of RESPONSE_SCHEMA plus a required authored budget_decision."""
    schema = copy.deepcopy(RESPONSE_SCHEMA)
    schema["properties"]["budget_decision"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["action", "reason", "tradeoff"],
        "properties": {
            "action": {"type": "string", "enum": list(BUDGET_ACTIONS)},
            "reason": {"type": "string"},
            "tradeoff": {"type": "string"},
        },
    }
    schema["required"] = [*schema["required"], "budget_decision"]
    return schema


def decision_context(state: Any, budget_snapshot: dict[str, Any], feedback: Any) -> dict[str, Any]:
    """Pass through actual remaining budget and allowed actions. Do not choose."""
    if not isinstance(budget_snapshot, dict) or "remaining_active_seconds" not in budget_snapshot:
        raise ValueError("budget_snapshot must include the actual remaining_active_seconds")
    allowed = adjustment_allowed(state)
    actions = [action for action in BUDGET_ACTIONS if action != "adjust_task" or allowed]
    context = {
        "adjustment_allowed": allowed,
        "allowed_actions": actions,
        "remaining_active_seconds": budget_snapshot["remaining_active_seconds"],
        "feedback": feedback,
    }
    for key in ("wall_seconds", "remaining_wall_seconds", "accounting_basis"):
        if key in budget_snapshot:
            context[key] = budget_snapshot[key]
    return context


def feedback_text(feedback: Any, context_path: Path | None = None) -> str:
    """Inline the feedback; past the client's input limit, inline its start and name the persisted copy."""
    text = json.dumps(feedback, ensure_ascii=False)
    if len(text) <= FEEDBACK_INLINE_CHARS or context_path is None:
        return text
    return (f"[{len(text)} characters; the complete feedback is the \"feedback\" field of {context_path}. "
            f"First {FEEDBACK_INLINE_CHARS} characters:] {text[:FEEDBACK_INLINE_CHARS]}")


def decision_prompt(context: dict[str, Any], context_path: Path | None = None) -> str:
    """Ask the author to choose under the remaining budget and record the tradeoff."""
    remaining = context["remaining_active_seconds"]
    wall = context.get("remaining_wall_seconds", context.get("wall_seconds"))
    basis = context.get("accounting_basis")
    actions = ", ".join(context["allowed_actions"])
    feedback = context.get("feedback")
    return (
        "A previous attempt failed. Choose the next action yourself from the allowed "
        "list under the remaining active budget. Record the tradeoff you are accepting. "
        "The program does not classify this failure or assign an action from remaining time. "
        "Source-bound equivalent joint or contact models are available only "
        "inside the already selected modeling scope. Hard physics, timeout, and geometry "
        "gates stay in force.\n"
        f"REMAINING ACTIVE SECONDS: {remaining}\n"
        f"WALL ACCOUNTING: {wall}\n"
        f"ACCOUNTING BASIS: {basis}\n"
        f"ALLOWED ACTIONS: {actions}\n"
        f"FAILURE FEEDBACK: {feedback_text(feedback, context_path)}\n"
        "Return the existing revision response fields plus budget_decision "
        "{action, reason, tradeoff}."
    )


def record_decision(path: Path, response: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Persist the authored choice and the pre-turn budget context. Do not choose."""
    decision = response.get("budget_decision") if isinstance(response, dict) else None
    if not isinstance(decision, dict):
        raise ValueError("Authored budget_decision is required; the program does not choose")
    action = decision.get("action")
    allowed = list(context.get("allowed_actions") or [])
    if action not in allowed:
        raise ValueError(f"Action {action!r} is not allowed in this budget context")
    if (not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
            or not isinstance(decision.get("tradeoff"), str) or not decision["tradeoff"].strip()):
        raise ValueError("budget_decision requires authored reason and tradeoff strings")
    record = {
        "schema_version": "slim_budget_decision_log_v1",
        "kind": "decision_log",
        "mechanical_certification": False,
        "budget_decision": {
            "action": action,
            "reason": decision["reason"],
            "tradeoff": decision["tradeoff"],
        },
        "pre_turn_budget": {
            key: context[key]
            for key in (
                "remaining_active_seconds",
                "wall_seconds",
                "remaining_wall_seconds",
                "accounting_basis",
            )
            if key in context
        },
        "allowed_actions": allowed,
        "adjustment_allowed": bool(context.get("adjustment_allowed")),
        "feedback": context.get("feedback"),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(Path(path), record)
    return record


def review_and_apply_adjustment(
    worker: Any,
    workspace: Path,
    project: Path,
    design: Path,
    turn_dir: Path,
    context: dict[str, Any],
    *, author: Any, tier_name: str = "light", budget: Any = None,
) -> dict[str, Any]:
    """Apply an author-decided task change after bounded independent advice."""
    workspace = Path(workspace)
    project = Path(project)
    design = Path(design)
    turn_dir = Path(turn_dir)
    state = json.loads((project / "workflow.json").read_text(encoding="utf-8-sig"))
    if not adjustment_allowed(state) or not context.get("adjustment_allowed"):
        raise ValueError(
            "Task adjustment requires explicit request.task_adjustable true "
            "and no frozen_experiment_sha256"
        )
    if (project / "frozen_experiment.json").is_file():
        raise ValueError("Frozen benchmark tasks cannot be adjusted")
    proposal_path = design / "task_adjustment.json"
    proposal = json.loads(proposal_path.read_text(encoding="utf-8-sig"))
    proposed = proposal.get("task_text")
    reason = proposal.get("reason")
    if not isinstance(proposed, str) or not proposed.strip():
        raise ValueError("Proposed task_text must be a nonempty change")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Proposed task adjustment requires a recorded reason")
    current_task = (project / "task.md").read_text(encoding="utf-8")
    if proposed == current_task:
        raise ValueError("Proposed task_text must be a nonempty change")
    before = _snapshot_current(project)
    original_task = state.get("request", {}).get("brief") or json.loads(before["criteria.json"])["task_text"]
    write_json_atomic(turn_dir/"task-adjustment-proposal.json", proposal)
    criteria = proposal.get("criteria") or initial_task_criteria(proposed)
    _validate_proposed_criteria(criteria, proposed)
    scene_path = design / "scene.json"
    scene_bytes = scene_path.read_bytes()
    scene = json.loads(scene_bytes)
    definition = task_definition(proposed, scene)
    if not (project / "simulation-task-baseline.json").is_file():
        write_json_atomic(project / "simulation-task-baseline.json", task_definition(current_task, scene))
    result = review_task_change(worker, author, workspace, project, design, turn_dir, definition,
                                reason=reason, context=context, tier_name=tier_name, budget=budget)
    if not result.get("accepted"):
        return {**result, "applied": False}
    if scene_path.read_bytes() != scene_bytes or json.loads(proposal_path.read_text(encoding="utf-8-sig")) != proposal:
        raise ValueError("Task proposal changed during the author decision; submit the new change")
    if (project / "task.md").read_bytes() != before["task.md"]:
        raise ValueError("Current task changed during review")
    if (project / "criteria.json").read_bytes() != before["criteria.json"]:
        raise ValueError("Current criteria changed during review")
    history = _next_history_dir(project)
    (history / "task.md").write_bytes(before["task.md"])
    (history / "criteria.json").write_bytes(before["criteria.json"])
    if "criteria.lock.json" in before:
        (history / CRITERIA_LOCK_NAME).write_bytes(before["criteria.lock.json"])
    write_json_atomic(history / "task_adjustment.json", proposal)
    write_json_atomic(
        history / "task_change_decision.json", result,
    )
    write_json_atomic(
        history / "original_vs_adjusted.json",
        {
            "original_task_text": original_task,
            "current_task_text_before": current_task,
            "adjusted_task_text": proposed,
            "reason": reason,
            "prior_outcomes_remain_historical": True,
            "new_tests": definition["tests"],
            "program_physics_and_geometry_checks_required": True,
            "preflight_or_final_model_review_required": False,
        },
    )
    log = AdjustmentLog()
    previous = json.loads(before["criteria.json"])
    log.record(
        CriteriaAdjustment(
            "task_text",
            {"task_text": current_task},
            {"task_text": proposed},
            reason,
        )
    )
    old_rows = {row["id"]: row for row in previous.get("criteria", []) if isinstance(row, dict) and row.get("id")}
    new_ids = set()
    for row in criteria:
        new_ids.add(row["id"])
        prior = old_rows.get(row["id"], {})
        if prior != row:
            log.record(CriteriaAdjustment(row["id"], prior or {"id": row["id"]}, row, reason))
    for cid, prior in old_rows.items():
        if cid not in new_ids:
            log.record(CriteriaAdjustment(cid, prior, {}, reason))
    log.save(history / "adjustment_log.json")
    _write_bytes_atomic(project / "task.md", proposed.encode("utf-8"))
    write_json_atomic(
        project / "criteria.json",
        {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(project / "task.md"),
            "task_text": proposed,
            "criteria": criteria,
            "reviewer_summary": "Author accepted task-change advice; no reviewer approval gate.",
            "source": "author_task_change_decision",
        },
    )
    write_json_atomic(
        project / CRITERIA_LOCK_NAME,
        {
            "schema_version": "slim_criteria_lock_v1",
            "criteria_sha256": file_digest(project / "criteria.json"),
            "task_sha256": file_digest(project / "task.md"),
        },
    )
    write_json_atomic(project / "simulation-task-baseline.json", definition)
    return {
        "status": "ready",
        "applied": True,
        "criteria": criteria,
        "criteria_sha256": file_digest(project / "criteria.json"),
        "adjustment_history_dir": str(history),
    }


def _request(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("request")
    return None


def _snapshot_current(project: Path) -> dict[str, bytes]:
    snapshot = {
        "task.md": (project / "task.md").read_bytes(),
        "criteria.json": (project / "criteria.json").read_bytes(),
    }
    lock = project / CRITERIA_LOCK_NAME
    if lock.is_file():
        snapshot["criteria.lock.json"] = lock.read_bytes()
    return snapshot


def _next_history_dir(project: Path) -> Path:
    root = project / "task_adjustments"
    root.mkdir(parents=True, exist_ok=True)
    prior = [int(path.name[1:]) for path in root.iterdir() if path.is_dir() and path.name[1:].isdigit() and path.name.startswith("a")]
    path = root / f"a{max(prior, default=0) + 1:03d}"
    path.mkdir()
    return path


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o666,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("Write made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _validate_proposed_criteria(criteria: Any, task_text: str) -> None:
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("Proposed criteria contain duplicate IDs or unsupported source quotes")
    ids = []
    for row in criteria:
        if not isinstance(row, dict):
            raise ValueError("Proposed criteria contain duplicate IDs or unsupported source quotes")
        cid = row.get("id")
        if not isinstance(cid, str) or not cid.strip():
            raise ValueError("Proposed criteria contain duplicate IDs or unsupported source quotes")
        ids.append(cid)
        if row.get("kind") not in {"appearance", "function"}:
            raise ValueError("Proposed criteria contain duplicate IDs or unsupported source quotes")
        quote = row.get("source_quote")
        if not isinstance(quote, str) or not quote.strip() or quote not in task_text:
            raise ValueError("Proposed criteria contain duplicate IDs or unsupported source quotes")
    if len(set(ids)) != len(ids):
        raise ValueError("Proposed criteria contain duplicate IDs or unsupported source quotes")

