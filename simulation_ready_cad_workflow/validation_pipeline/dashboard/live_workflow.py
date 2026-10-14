"""Small, read-only cold-start progress and per-call usage projections.

This module never loads geometry or invokes an author, renderer, or validator.
Workflow progress is presentation data and has no certificate authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import threading
from typing import Any

from ..run_layout import parse_canonical_run
from ..cold_start.token_watch import normalize_token_usage


_LOCK = threading.RLock()
_JSON_CACHE: dict[Path, tuple[tuple[int, int, int], dict[str, Any]]] = {}
_TURN_CACHE: dict[Path, "_TurnEvents"] = {}
_TURN_NAME = re.compile(r"t[0-9]+_.+")


def _stamp(path: Path) -> tuple[int, int, int] | None:
    try:
        stat = path.stat()
        # Atomic replacement can retain size and coarse exFAT mtime while its
        # file identity changes. That is still a new usage snapshot.
        return stat.st_mtime_ns, stat.st_size, stat.st_ino
    except OSError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    stamp = _stamp(path)
    if stamp is None:
        return {}
    with _LOCK:
        cached = _JSON_CACHE.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        value = value if isinstance(value, dict) else {}
        _JSON_CACHE[path] = stamp, value
        return value


def _context(workspace_root: str | Path, run_dir: str | Path):
    identity = parse_canonical_run(workspace_root, run_dir)
    if identity is None or not identity.run_dir.is_dir():
        return None, {}
    state = _read_json(identity.project_dir / "workflow.json")
    request = state.get("request")
    revision = state.get("revision")
    if (
        state.get("schema_version") not in {"cold_start_workflow_v1", "slim_harness_workflow_v1"}
        or not isinstance(request, dict)
        or request.get("project_id") != identity.project_id
        or not isinstance(revision, int) or isinstance(revision, bool) or revision < 1
        or not isinstance(state.get("phase"), str) or not state["phase"]
    ):
        return None, {}
    return identity, state


def is_cold_start_run(*, workspace_root: str | Path, run_dir: str | Path) -> bool:
    """Recognize a real project revision even before its first submission."""
    identity, _state = _context(workspace_root, run_dir)
    return identity is not None


def _turn_directories(project: Path) -> tuple[Path, ...]:
    try:
        return tuple(sorted(
            path for path in (project / "worker_turns").iterdir()
            if path.is_dir() and _TURN_NAME.fullmatch(path.name)
        ))
    except OSError:
        return ()


def live_project_metadata_paths(
    *, workspace_root: str | Path, run_dir: str | Path,
) -> tuple[Path, ...]:
    """Files to stat for progress invalidation, with no recursive model walk."""
    identity, _state = _context(workspace_root, run_dir)
    if identity is None:
        return ()
    project = identity.project_dir
    paths = [project / "workflow.json", project / "workflow_runtime.json", project / "construction_budget.json", project / "external_author_runtime.json", project / "external-author-usage.json", project / "template-progress.json"]
    paths.extend((identity.run_dir / "result.json", identity.run_dir / "slim_evidence" / "preview.json",
                  identity.run_dir / "slim_evidence" / "partial-motion-preview.json", project / "final-motion.json"))
    for turn in _turn_directories(project):
        paths.extend((turn / "events.jsonl", turn / "response.json", turn / "usage.live.json"))
    return tuple(paths)


def _tokens(value: Any) -> dict[str, int | None] | None:
    usage = normalize_token_usage(value)
    if usage is None:
        return None
    # Cached input is included in input_tokens, not an additional charge.
    return {"input_tokens": usage["input_tokens"],
            "cached_input_tokens": usage.get("cached_input_tokens"),
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"]}


@dataclass
class _TurnEvents:
    call_id: str = ""
    stamp: tuple[int, int, int] | None = None
    offset: int = 0
    tail: bytes = b""
    usage: dict[str, int | None] | None = None
    completed: bool = False


def _read_turn_events(path: Path, *, call_id: str = "") -> _TurnEvents:
    stamp = _stamp(path)
    with _LOCK:
        cached = _TURN_CACHE.setdefault(path, _TurnEvents())
        if call_id != cached.call_id:
            cached = _TurnEvents(call_id=call_id)
            _TURN_CACHE[path] = cached
        if stamp is None or stamp == cached.stamp:
            return cached
        # Rewritten/truncated files are a new snapshot; append-only logs reuse
        # the byte offset, including a buffered partially written final line.
        if stamp[1] < cached.offset or (cached.stamp and stamp[1] == cached.stamp[1]):
            cached = _TurnEvents(call_id=call_id)
            _TURN_CACHE[path] = cached
        try:
            with path.open("rb") as stream:
                stream.seek(cached.offset)
                chunk = stream.read()
                cached.offset = stream.tell()
        except OSError:
            return cached
        lines = (cached.tail + chunk).split(b"\n")
        cached.tail = lines.pop()
        # A complete final JSON value is also readable before a producer adds
        # its trailing newline. Keep it buffered so later appends remain valid.
        for line in (*lines, cached.tail):
            try:
                event = json.loads(line)
            except (UnicodeError, json.JSONDecodeError):
                continue
            if isinstance(event, dict) and event.get("type") in {"turn.completed", "result"}:
                if event.get("type") == "turn.completed":
                    cached.completed = True
                usage = _tokens(event.get("usage"))
                if usage is not None:
                    cached.completed = True
                    # One output directory owns one provider call. A reread or
                    # duplicate completion must not add that call twice.
                    cached.usage = usage
        cached.stamp = stamp
        return cached


def _datetime(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result
    except (ValueError, TypeError):
        return None


def _seconds(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= 0 else None


def _construction_elapsed(project: Path, *, exhausted: bool = False) -> float | None:
    """Effective construction work, not calendar time across an overnight stop."""
    ledger = _read_json(project / "construction_budget.json")
    if ledger.get("schema_version") != "slim_construction_budget_v1":
        return None
    elapsed = _seconds(ledger.get("active_seconds"))
    if elapsed is None:
        return None
    config = ledger.get("config") if isinstance(ledger.get("config"), dict) else {}
    hours = config.get("hours")
    from ..slim_harness.construction_budget import BUDGET_HOURS
    if hours not in BUDGET_HOURS:
        return elapsed
    limit = float(hours) * 3600.0
    reserve = _seconds(config.get("summary_reserve_seconds")) or 0.0
    if exhausted or elapsed + reserve >= limit:
        return limit
    return min(elapsed, limit)


def _process_matches(runtime: dict[str, Any]) -> bool | None:
    """A reused PID cannot keep an old workflow clock running."""
    pid = runtime.get("pid")
    started = runtime.get("process_started_at_epoch")
    if not isinstance(pid, int) or not isinstance(started, (int, float)):
        return None
    try:
        import psutil
        process = psutil.Process(pid)
        return process.is_running() and abs(process.create_time() - float(started)) < 0.1
    except ImportError:
        return None
    except Exception as exc:
        # AccessDenied does not establish death; NoSuchProcess does.
        return False if type(exc).__name__ in {"NoSuchProcess", "ZombieProcess"} else None


def _slim_test_progress(project: Path, state: dict[str, Any], running: bool) -> dict[str, Any]:
    """Project only small persisted metadata; never load traces or run tools."""
    if state.get("schema_version") != "slim_harness_workflow_v1":
        return {}
    run = project / "runs" / f"r{int(state['revision']):02d}"
    if state.get("request",{}).get("mode")=="free":
        saved=_read_json(project/"template-progress.json")
        attempt=Path(saved.get("attempt", "."))
        if not attempt.is_relative_to(run):return {}
        plan=saved.get("plan") or {};outcomes={row.get("test_id"):row for row in saved.get("tests",[]) if isinstance(row,dict)}
        active=state.get("active_test") if running and state.get("phase")=="physics" else None
        rows=[]
        for trial in plan.get("trials",[]):
            row={key:outcomes.get(trial["id"],{})[key] for key in ("execution_status","execution_started","physical_pass","failure_status","geometry_audit") if key in outcomes.get(trial["id"],{})}
            row.update(test_id=trial["id"],requested_duration_s=trial.get("duration_s"))
            row.setdefault("execution_status","running" if active==trial["id"] else "not_run")
            rows.append(row)
        return {"attempt":attempt.name,"active_test":active,"returned":len(outcomes),"total":len(rows),"tests":rows}
    tests = _read_json(run / "design" / "scene.json").get("tests") or []
    if not isinstance(tests, list):
        tests = []
    attempts = [p for p in (run / "attempts").glob("a[0-9]*") if p.is_dir() and p.name[1:].isdigit()]
    attempt = max(attempts, key=lambda p: int(p.name[1:]), default=None)
    active = state.get("active_test") if running and state.get("phase") == "physics" else None
    rows, returned = [], 0
    for test in tests:
        if not isinstance(test, dict):
            continue
        identity = test.get("id")
        if not identity:
            continue
        folder = attempt / "tests" / identity if attempt else None
        outcome = _read_json(folder / "test-outcome.json") if folder else {}
        returned += bool(outcome)
        row = {key: outcome[key] for key in ("execution_status", "execution_started", "physical_pass", "failure_status", "geometry_audit") if key in outcome}
        row.update(test_id=identity, requested_duration_s=test.get("duration_s"))
        row.setdefault("execution_status", "running" if identity == active else "pending_result" if folder and folder.is_dir() else "not_run")
        rows.append(row)
    return {"attempt": attempt.name if attempt else None, "active_test": active,
            "returned": returned, "total": len(rows), "tests": rows}


def _workflow(identity: Any, state: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    runtime = _read_json(identity.project_dir / "workflow_runtime.json")
    from ..snapshot_paths import imported, workspace_for
    archive_root = workspace_for(identity.project_dir)
    archive = imported(archive_root) if archive_root else None
    if archive:
        runtime = {**runtime, 'status': 'complete' if state.get('complete_instance_pass') else 'stopped',
                   'active_turn': {}, 'stop_reason': 'imported_snapshot'}
    if runtime.get("schema_version") not in {"cold_start_runtime_v1", "workflow_runtime_v1"}:
        runtime = {}
    heartbeat = _datetime(runtime.get("heartbeat_at"))
    updated = _datetime(state.get("updated_at"))
    runtime_current = bool(runtime and heartbeat and (updated is None or heartbeat >= updated))
    progress = runtime if runtime_current else state
    stop_reason = progress.get("stop_reason") or ""
    complete = bool(state.get("complete_instance_pass")) or (
        runtime_current and runtime.get("status") == "complete"
    )
    stopped = bool(archive or stop_reason or complete or (runtime_current and runtime.get("status") == "stopped"))
    alive = _process_matches(runtime) if runtime_current and not stopped else None
    recent = bool(heartbeat and -5 <= (now - heartbeat).total_seconds() <= 10)
    running = bool(not stopped and runtime_current and runtime.get("status") == "running"
                   and (alive is True or (alive is None and recent)))
    # An external author's public-event bridge is presentation evidence only.
    # It never impersonates workflow_runtime.json or owns the Harness lock.
    external_path = identity.project_dir / "external_author_runtime.json"
    external = _read_json(external_path)
    # The author may be on another host whose wall clock differs. Freshness
    # belongs to this host's observed receipt, not the sender's clock.
    external_stamp = _stamp(external_path)
    external_at = datetime.fromtimestamp(external_stamp[0] / 1e9, timezone.utc) if external_stamp else None
    external_recent = (external.get("schema_version") == "external_author_runtime_v1"
                       and external_at is not None and -5 <= (now-external_at).total_seconds() <= 15)
    external_live = external_recent and external.get("status") == "running"
    external_paused = external_recent and external.get("status") in {"paused", "waiting"}
    external_override = False
    if not archive and external_live and not complete and not running:
        progress = {**state, "phase": "author", "active_turn": external.get("active_turn", {}),
                    "elapsed_seconds": external.get("elapsed_seconds")}
        running=True;stopped=False;stop_reason="";alive=None
        heartbeat=external_at;runtime_current=False
        external_override=True
    elif not archive and external_paused and not complete and not running:
        progress = {**state, "phase": "paused", "active_turn": external.get("active_turn", {}),
                    "elapsed_seconds": external.get("elapsed_seconds")}
        stopped=True;stop_reason=external.get("reason") or "external_author_waiting";alive=None
        heartbeat=external_at;runtime_current=False;external_override=True
    status = "complete" if complete else "paused" if external_override and external_paused else "stopped" if stopped or alive is False else "running" if running else "activity_unknown"
    construction = _construction_elapsed(
        identity.project_dir,
        exhausted=str(stop_reason) in {"budget_exhausted", "BUDGET_EXHAUSTED", "ACTIVE_BUDGET_EXHAUSTED"},
    )
    elapsed = construction
    if elapsed is None:
        elapsed = _seconds(progress.get("elapsed_seconds"))
    if elapsed is None:
        elapsed = _seconds(state.get("elapsed_seconds"))
    started = _datetime(state.get("started_at"))
    last_at = external_at if external_override else heartbeat if runtime_current else updated
    if elapsed is None and started and last_at:
        elapsed = max(0.0, (last_at - started).total_seconds())
    if construction is None and running and started:
        elapsed = max(elapsed or 0.0, (now - started).total_seconds())
    revision = progress.get("revision", state.get("revision"))
    request = state["request"]
    return {
        "project_id": identity.project_id,
        "workflow_kind": "slim_harness" if state.get("schema_version") == "slim_harness_workflow_v1" else "cold_start",
        "title": request.get("title") or request.get("brief") or identity.project_id,
        "brief": request.get("brief") or "",
        "phase": progress.get("phase", state.get("phase", "")),
        "revision": revision,
        "status": status,
        "started_at": state.get("started_at") or "",
        "updated_at": last_at.isoformat() if last_at else state.get("updated_at") or "",
        "ended_at": (last_at.isoformat() if last_at else "") if status in {"complete", "stopped"} else "",
        "elapsed_seconds": elapsed,
        "is_running": running,
        "active_turn": {} if archive else progress.get("active_turn") or {},
        "test_progress": _slim_test_progress(identity.project_dir, state, running),
        "stop_reason": stop_reason,
        # An imported snapshot's displayed stop reason hides the recorded one.
        "outcome": run_outcome(state, state.get("stop_reason") if archive else stop_reason,
                               complete=complete, running=running,
                               result=None if running else _revision_result(identity.project_dir, revision)),
        "recent_activity": [{"kind":event.get("kind"),"at":event.get("at"),
                             "status":(event.get("result") or {}).get("status"),
                             "summary":(event.get("result") or {}).get("summary")}
                            for event in (state.get("events") or [])[-12:] if isinstance(event,dict)],
    }


def _usage(project: Path, state: dict[str, Any], workflow: dict[str, Any]) -> dict[str, Any]:
    # Workflow events are a bounded fallback for a recorded provider response.
    # The directory key merges events, response.json and CLI events for a call.
    by_turn: dict[str, dict[str, int | None] | None] = {}
    workflow_usage_at: dict[str, datetime | None] = {}
    pending_turns: set[str] = set()
    final_usage_turns: set[str] = set()
    live_sources: set[str] = set()
    completed: set[str] = set()
    updated = workflow.get("updated_at") or ""
    for event in state.get("events") or []:
        if not isinstance(event, dict):
            continue
        result = event.get("result")
        if not isinstance(result, dict) or not result.get("turn_directory"):
            continue
        name = str(result["turn_directory"]).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        if not _TURN_NAME.fullmatch(name):
            continue
        completed.add(name)
        by_turn[name] = _tokens(result.get("usage"))
        workflow_usage_at[name] = _datetime(event.get("at"))
        if by_turn[name] is not None:
            final_usage_turns.add(name)
    for directory in _turn_directories(project):
        if _read_json(directory/"program.json").get("schema_version")=="harness_program_capture_v1":
            continue  # Program trajectory events are not model calls or unknown token usage.
        response_path = directory / "response.json"
        response = _read_json(response_path)
        events_path = directory / "events.jsonl"
        live_path = directory / "usage.live.json"
        live = _read_json(live_path)
        if live.get("schema_version") != "cold_start_live_token_usage_v1":
            live = {}
        events = _read_turn_events(events_path, call_id=str(live.get("call_id") or ""))
        name = directory.name
        workflow_usage = by_turn.get(name)
        call_started = _datetime(live.get("call_started_at"))
        if call_started and (workflow_usage_at.get(name) is None or workflow_usage_at[name] < call_started):
            workflow_usage = None
            final_usage_turns.discard(name)
            completed.discard(name)
        final_usage = events.usage or _tokens(response.get("usage")) or workflow_usage
        live_usage = _tokens(live.get("usage"))
        completed_live = (
            live_usage is not None and live.get("pending") is False
            and live.get("source") in {"codex_cli_turn_completed", "cursor_agent_result_usage"}
        )
        if completed_live:
            final_usage = final_usage or live_usage
        by_turn[name] = final_usage or live_usage
        if final_usage is not None:
            final_usage_turns.add(name)
        elif live_usage is not None:
            pending_turns.add(name)
            live_sources.add(str(live.get("source") or "codex_raw_turn_token_usage"))
        if live.get("updated_at"):
            updated = max(updated, str(live["updated_at"]))
        if events.completed or response or completed_live:
            completed.add(name)
        for path in (events_path, response_path):
            stamp = _stamp(path)
            if stamp:
                at = datetime.fromtimestamp(stamp[0] / 1e9, timezone.utc).isoformat()
                updated = max(updated, at)
    active = workflow.get("active_turn") or {}
    active_name = str(active.get("directory") or "").replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if not active_name and isinstance(active.get("id"),str) and _TURN_NAME.fullmatch(active["id"]):
        active_name=active["id"]
    active_pending = bool(active_name and active_name not in final_usage_turns)
    values = [value for value in by_turn.values() if value is not None]
    external = _read_json(project / "external-author-usage.json")
    external_author = {}
    if external.get("schema_version") == "external_author_usage_v1" and external.get("project_id") == workflow.get("project_id"):
        observed = _tokens(external.get("usage"))
        if observed is not None:
            values.append(observed)
            live_sources.add("external_codex_subagent")
            external_author = {key: external.get(key) for key in ("session_id", "model", "reasoning_effort", "updated_at", "status")}
    totals: dict[str, int | None] = {}
    for field_name in ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
        parts = [value[field_name] for value in values]
        totals[field_name] = sum(parts) if parts and all(part is not None for part in parts) else None
    return {**totals, "completed_turn_count": len(completed), "external_author": external_author,
            "active_tokens_pending": active_pending, "updated_at": updated,
            "source": "+".join(["program_worker_turns" if workflow.get("workflow_kind")=="slim_harness" else "cold_start_worker_turns", *sorted(live_sources)]),
            "partial": active_pending or bool(pending_turns) or any(value is None for value in by_turn.values()) or not values}


# Ends that close a run without a pass; the project list counts them as failed.
ENDED_WITHOUT_PASS = {"final_evaluation_failed", "partial_delivery"}
# Stop reasons that end a run after the reserved last revision (first-pass handoff).
_FINAL_EVALUATION_STOPS = {"EVIDENCE_REQUIRED", "NEEDS_REVISION", "SIMULATION_TIMEOUT"}


def _revision_result(project_dir: Path, revision: Any) -> dict[str, Any]:
    try:
        return _read_json(project_dir / "runs" / f"r{int(revision):02d}" / "result.json")
    except (TypeError, ValueError):
        return {}


def run_outcome(state: dict[str, Any], stop_reason: str, *, complete: bool, running: bool,
                result: dict[str, Any] | None = None) -> str:
    """How an ended run finished, for the people watching it; "" while it runs.

    ``result`` is the last revision's result.json: construction runs record a
    Harness error there as ``reported_status`` and still stop on NEEDS_REVISION."""
    if running:
        return ""
    if complete:
        return "passed"
    reason = str(stop_reason or "")
    if reason in {"budget_exhausted", "BUDGET_EXHAUSTED", "ACTIVE_BUDGET_EXHAUSTED"}:
        return "budget_exhausted"
    if reason in {"provider_error", "review_provider_unavailable"}:
        return "provider_error"
    if reason == "agent_chose_partial_delivery":
        return "partial_delivery"
    if reason == "no_executable_scene":
        return "scene_untestable"
    if reason == "HARNESS_ERROR":
        return "harness_error"
    if state.get("first_pass_handoff") and reason in _FINAL_EVALUATION_STOPS:
        result = result or {}
        tests = [row for row in result.get("tests") or [] if isinstance(row, dict)]
        if result.get("reported_status") == "HARNESS_ERROR":
            return "harness_error"
        if tests and all(row.get("physical_pass") for row in tests):
            return "author_unfinished"
        return "final_evaluation_failed"
    return "stopped"


def read_live_project(*, workspace_root: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Read current project progress without normalizing the evidence tree."""
    identity, state = _context(workspace_root, run_dir)
    if identity is None:
        return {"live_workflow": {}, "live_usage": {}}
    workflow = _workflow(identity, state)
    return {"live_workflow": workflow, "live_usage": _usage(identity.project_dir, state, workflow)}
