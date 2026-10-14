from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Sequence

from .mechanism.io import JsonDocumentError, read_json, relative_to_root, write_json_atomic


_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_ROOT_RELATIVE = Path(".runtime_runs") / "project_usage"


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise JsonDocumentError(
            f"{label} must match {_IDENTIFIER.pattern!r}: {value!r}"
        )
    return value


def _token_counter(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise JsonDocumentError(f"{label} must be a non-negative integer")
    return value


def _timestamp(epoch_seconds: float) -> str:
    if not math.isfinite(epoch_seconds) or epoch_seconds < 0:
        raise JsonDocumentError("usage clock returned an invalid epoch")
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _usage_root(workspace_root: str | Path) -> tuple[Path, Path]:
    workspace = Path(workspace_root).resolve()
    root = (workspace / _ROOT_RELATIVE).resolve()
    root.relative_to(workspace)
    return workspace, root


def _session_path(root: Path, session_id: str) -> Path:
    return root / "sessions" / f"{session_id}.json"


def _project_path(root: Path, project_id: str) -> Path:
    return root / "projects" / f"{project_id}.json"


def automatic_usage_session_id(project_id: str, revision: str) -> str:
    """Return the stable accounting identity for one canonical rNN run."""

    project = _identifier(project_id, "project_id")
    run_revision = _identifier(revision, "revision")
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:12]
    return f"auto-{digest}-{run_revision}"


def assert_project_usage_settled(
    *,
    workspace_root: str | Path,
    project_id: str,
) -> None:
    """Refuse a new revision while an older automatic record is unfinished."""

    project = _identifier(project_id, "project_id")
    _workspace, root = _usage_root(workspace_root)
    pending = [
        item
        for item in _project_sessions(root, project)
        if item.get("status") in {"RUNNING", "AWAITING_TOKEN_COUNTER"}
    ]
    if pending:
        labels = ", ".join(str(item.get("run_id") or "?") for item in pending)
        raise JsonDocumentError(
            "previous project usage must be settled before creating another "
            f"revision: {labels}"
        )


def begin_project_usage(
    *,
    workspace_root: str | Path,
    project_id: str,
    session_id: str,
    run_id: str,
    token_counter: int | None,
    token_counter_source: str,
    token_counter_allocation: str | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    project = _identifier(project_id, "project_id")
    session = _identifier(session_id, "session_id")
    run = _identifier(run_id, "run_id")
    baseline = (
        _token_counter(token_counter, "token_counter")
        if token_counter is not None
        else None
    )
    source = _identifier(token_counter_source, "token_counter_source")
    if token_counter_allocation not in {None, "allocated", "raw"}:
        raise JsonDocumentError(
            "token_counter_allocation must be allocated, raw, or null"
        )
    workspace, root = _usage_root(workspace_root)
    destination = _session_path(root, session)
    if destination.exists():
        existing = read_json(destination)
        identity = (
            existing.get("project_id"),
            existing.get("session_id"),
            existing.get("run_id"),
            existing.get("token_counter_source"),
            existing.get("token_counter_start"),
        )
        expected = (project, session, run, source, baseline)
        if identity != expected:
            raise JsonDocumentError(
                f"usage session already exists with different identity: {session}"
            )
        if (
            token_counter_allocation is not None
            and existing.get("token_counter_start_allocation") is None
        ):
            existing["token_counter_start_allocation"] = (
                token_counter_allocation
            )
            write_json_atomic(destination, existing)
        return existing
    started_epoch = float(clock())
    payload: dict[str, object] = {
        "schema_version": "0.1",
        "status": "RUNNING",
        "project_id": project,
        "session_id": session,
        "run_id": run,
        "started_at": _timestamp(started_epoch),
        "started_at_epoch_seconds": started_epoch,
        "token_counter_source": source,
        "token_counter_start": baseline,
        "session_artifact": relative_to_root(workspace, destination),
    }
    if token_counter_allocation is not None:
        payload["token_counter_start_allocation"] = token_counter_allocation
    write_json_atomic(destination, payload)
    rebuild_project_usage_summary(
        workspace_root=workspace,
        project_id=project,
    )
    return payload


def _project_sessions(root: Path, project_id: str) -> list[dict[str, object]]:
    sessions: list[dict[str, object]] = []
    session_root = root / "sessions"
    if not session_root.exists():
        return sessions
    for path in sorted(session_root.glob("*.json"), key=lambda item: item.name):
        if not path.is_file() or path.is_symlink():
            raise JsonDocumentError(f"usage session must be a regular file: {path}")
        payload = read_json(path)
        if payload.get("project_id") == project_id:
            sessions.append(payload)
    return sessions


def rebuild_project_usage_summary(
    *,
    workspace_root: str | Path,
    project_id: str,
) -> dict[str, object]:
    project = _identifier(project_id, "project_id")
    workspace, root = _usage_root(workspace_root)
    sessions = _project_sessions(root, project)
    completed = [item for item in sessions if item.get("status") == "COMPLETED"]
    running = [item for item in sessions if item.get("status") == "RUNNING"]
    awaiting = [
        item
        for item in sessions
        if item.get("status") == "AWAITING_TOKEN_COUNTER"
    ]
    unmeasured = [
        item for item in sessions if item.get("status") == "CLOSED_UNMEASURED"
    ]
    measured_wall_time = completed + [
        item
        for item in unmeasured
        if item.get("wall_time_measured")
        and item.get("wall_time_seconds") is not None
    ]
    total_seconds = sum(
        float(item["wall_time_seconds"]) for item in measured_wall_time
    )
    token_totals_by_source: dict[str, int] = {}
    token_sessions_by_source: dict[str, int] = {}
    for item in completed:
        source = str(item["token_counter_source"])
        token_totals_by_source[source] = (
            token_totals_by_source.get(source, 0) + int(item["total_tokens"])
        )
        token_sessions_by_source[source] = (
            token_sessions_by_source.get(source, 0) + 1
        )
    running_token_totals: dict[str, int] = {}
    unallocated_backfill_tokens = codex_project_unallocated_token_counter(
        workspace_root=workspace,
        project_id=project,
    )
    if running:
        live_codex_counter = codex_project_token_counter(
            workspace_root=workspace,
            project_id=project,
        )
        for item in running:
            source = str(item["token_counter_source"])
            start = item.get("token_counter_start")
            live_tokens = (
                max(0, live_codex_counter - int(start))
                if source == "codex_turn_usage" and isinstance(start, int)
                else 0
            )
            running_token_totals[str(item["session_id"])] = live_tokens
            token_totals_by_source[source] = (
                token_totals_by_source.get(source, 0) + live_tokens
            )
            token_sessions_by_source[source] = (
                token_sessions_by_source.get(source, 0) + 1
            )
    if unallocated_backfill_tokens:
        token_totals_by_source[CODEX_TOKEN_SOURCE] = (
            token_totals_by_source.get(CODEX_TOKEN_SOURCE, 0)
            + unallocated_backfill_tokens
        )
        token_sessions_by_source.setdefault(CODEX_TOKEN_SOURCE, 0)
    if "codex_goal_counter" in token_totals_by_source:
        benchmark_source = "codex_goal_counter"
    elif len(token_totals_by_source) == 1:
        benchmark_source = next(iter(token_totals_by_source))
    else:
        benchmark_source = ""
    benchmark_total_tokens = (
        token_totals_by_source.get(benchmark_source)
        if benchmark_source
        else None
    )
    raw_total_tokens = sum(token_totals_by_source.values())
    destination = _project_path(root, project)
    payload: dict[str, object] = {
        "schema_version": "0.1",
        "project_id": project,
        "status": (
            "RUNNING"
            if running
            else ("TOKEN_PENDING" if awaiting else "CURRENT")
        ),
        "completed_session_count": len(completed),
        "running_session_count": len(running),
        "pending_token_session_count": len(awaiting),
        "total_wall_time_seconds": round(total_seconds, 6),
        "total_tokens": benchmark_total_tokens,
        "benchmark_token_counter_source": benchmark_source,
        "raw_total_tokens_all_sources": raw_total_tokens,
        "unallocated_backfill_tokens": unallocated_backfill_tokens,
        "mixed_token_counter_sources": len(token_totals_by_source) > 1,
        "token_totals_by_source": [
            {
                "token_counter_source": source,
                "total_tokens": total,
                "session_count": token_sessions_by_source[source],
                "completed_session_count": sum(
                    1
                    for item in completed
                    if item["token_counter_source"] == source
                ),
                "running_session_count": sum(
                    1 for item in running if item["token_counter_source"] == source
                ),
                "included_in_benchmark": source == benchmark_source,
            }
            for source, total in sorted(token_totals_by_source.items())
        ],
        "run_ids": sorted({str(item["run_id"]) for item in completed}),
        "completed_sessions": [
            {
                "session_id": item["session_id"],
                "run_id": item["run_id"],
                "wall_time_seconds": item["wall_time_seconds"],
                "total_tokens": item["total_tokens"],
                "completed_at": item["completed_at"],
                "token_counter_source": item["token_counter_source"],
            }
            for item in completed
        ],
        "running_sessions": [
            {
                "session_id": item["session_id"],
                "run_id": item["run_id"],
                "started_at": item["started_at"],
                "total_tokens": running_token_totals.get(
                    str(item["session_id"]), 0
                ),
                "token_counter_source": item["token_counter_source"],
            }
            for item in running
        ],
        "pending_token_sessions": [
            {
                "session_id": item["session_id"],
                "run_id": item["run_id"],
                "wall_time_seconds": item["wall_time_seconds"],
                "completed_at": item["completed_at"],
                "token_counter_source": item["token_counter_source"],
                "token_counter_start": item["token_counter_start"],
            }
            for item in awaiting
        ],
        "unmeasured_session_count": len(unmeasured),
        "unmeasured_wall_time_seconds": round(
            sum(
                float(item.get("wall_time_seconds") or 0.0)
                for item in unmeasured
            ),
            6,
        ),
        "unmeasured_sessions": [
            {
                "session_id": item["session_id"],
                "run_id": item["run_id"],
                "wall_time_seconds": item.get("wall_time_seconds"),
                "token_counter_source": item["token_counter_source"],
                "migration_reason": item.get("migration_reason"),
            }
            for item in unmeasured
        ],
        "summary_artifact": relative_to_root(workspace, destination),
    }
    write_json_atomic(destination, payload)
    return payload


def finish_project_usage(
    *,
    workspace_root: str | Path,
    session_id: str,
    token_counter: int,
    token_counter_allocation: str | None = None,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    session = _identifier(session_id, "session_id")
    final_counter = _token_counter(token_counter, "token_counter")
    if token_counter_allocation not in {None, "allocated", "raw"}:
        raise JsonDocumentError(
            "token_counter_allocation must be allocated, raw, or null"
        )
    workspace, root = _usage_root(workspace_root)
    destination = _session_path(root, session)
    payload = read_json(destination)
    project = _identifier(str(payload.get("project_id")), "project_id")
    if payload.get("status") == "COMPLETED":
        if payload.get("token_counter_end") != final_counter:
            raise JsonDocumentError(
                "completed usage session cannot be closed with a different counter"
            )
        return rebuild_project_usage_summary(
            workspace_root=workspace,
            project_id=project,
        )
    if payload.get("status") not in {"RUNNING", "AWAITING_TOKEN_COUNTER"}:
        raise JsonDocumentError(
            f"usage session has invalid status: {payload.get('status')!r}"
        )
    baseline = _token_counter(
        payload.get("token_counter_start"),
        "token_counter_start",
    )
    if final_counter < baseline:
        raise JsonDocumentError(
            "token counter moved backwards; refusing to fabricate usage"
        )
    if payload.get("status") == "AWAITING_TOKEN_COUNTER":
        finished_epoch = float(payload["completed_at_epoch_seconds"])
        wall_time_seconds = float(payload["wall_time_seconds"])
    else:
        finished_epoch = float(clock())
        started_epoch = float(payload.get("started_at_epoch_seconds", -1.0))
        if finished_epoch < started_epoch:
            raise JsonDocumentError(
                "usage clock moved backwards; refusing a negative duration"
            )
        wall_time_seconds = round(finished_epoch - started_epoch, 6)
    completed = dict(payload)
    completed.update(
        {
            "status": "COMPLETED",
            "completed_at": _timestamp(finished_epoch),
            "completed_at_epoch_seconds": finished_epoch,
            "wall_time_seconds": wall_time_seconds,
            "token_counter_end": final_counter,
            "total_tokens": final_counter - baseline,
        }
    )
    if token_counter_allocation is not None:
        completed["token_counter_end_allocation"] = (
            token_counter_allocation
        )
    write_json_atomic(destination, completed)
    return rebuild_project_usage_summary(
        workspace_root=workspace,
        project_id=project,
    )


def close_project_usage_wall_time(
    *,
    workspace_root: str | Path,
    project_id: str,
    revision: str,
    run_status: str,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """Close wall time automatically and wait for the external token counter."""

    project = _identifier(project_id, "project_id")
    session_id = automatic_usage_session_id(project, revision)
    workspace, root = _usage_root(workspace_root)
    destination = _session_path(root, session_id)
    payload = read_json(destination)
    if payload.get("status") in {"COMPLETED", "AWAITING_TOKEN_COUNTER"}:
        return rebuild_project_usage_summary(
            workspace_root=workspace,
            project_id=project,
        )
    if payload.get("status") != "RUNNING":
        raise JsonDocumentError(
            f"automatic usage session has invalid status: {payload.get('status')!r}"
        )
    finished_epoch = float(clock())
    started_epoch = float(payload.get("started_at_epoch_seconds", -1.0))
    if finished_epoch < started_epoch:
        raise JsonDocumentError(
            "usage clock moved backwards; refusing a negative duration"
        )
    awaiting = dict(payload)
    awaiting.update(
        {
            "status": "AWAITING_TOKEN_COUNTER",
            "completed_at": _timestamp(finished_epoch),
            "completed_at_epoch_seconds": finished_epoch,
            "wall_time_seconds": round(finished_epoch - started_epoch, 6),
            "run_status": str(run_status),
            "token_settlement_required": True,
        }
    )
    write_json_atomic(destination, awaiting)
    return rebuild_project_usage_summary(
        workspace_root=workspace,
        project_id=project,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="project-usage",
        description=(
            "Record auditable wall-clock and externally measured token totals "
            "for a validation project."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    begin = subparsers.add_parser("begin")
    begin.add_argument("--workspace-root", required=True, type=Path)
    begin.add_argument("--project-id", required=True)
    begin.add_argument("--session-id", required=True)
    begin.add_argument("--run-id", required=True)
    begin.add_argument("--token-counter", required=True, type=int)
    begin.add_argument("--token-counter-source", required=True)
    finish = subparsers.add_parser("finish")
    finish.add_argument("--workspace-root", required=True, type=Path)
    finish.add_argument("--session-id", required=True)
    finish.add_argument("--token-counter", required=True, type=int)
    settle_run = subparsers.add_parser("settle-run")
    settle_run.add_argument("--workspace-root", required=True, type=Path)
    settle_run.add_argument("--project-id", required=True)
    settle_run.add_argument("--revision", required=True)
    settle_run.add_argument("--token-counter", required=True, type=int)
    close = subparsers.add_parser("close-run")
    close.add_argument("--workspace-root", required=True, type=Path)
    close.add_argument("--project-id", required=True)
    close.add_argument("--revision", required=True)
    close.add_argument("--run-status", required=True)
    summary = subparsers.add_parser("summary")
    summary.add_argument("--workspace-root", required=True, type=Path)
    summary.add_argument("--project-id", required=True)
    migrate = subparsers.add_parser(
        "migrate-legacy-tokens",
        help=(
            "Close sessions stuck on the legacy Kimi token counter as "
            "CLOSED_UNMEASURED; dry-run unless --apply is given."
        ),
    )
    migrate.add_argument("--workspace-root", required=True, type=Path)
    migrate.add_argument("--project-id", required=True)
    migrate.add_argument("--apply", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "begin":
            payload = begin_project_usage(
                workspace_root=args.workspace_root,
                project_id=args.project_id,
                session_id=args.session_id,
                run_id=args.run_id,
                token_counter=args.token_counter,
                token_counter_source=args.token_counter_source,
            )
        elif args.command == "finish":
            payload = finish_project_usage(
                workspace_root=args.workspace_root,
                session_id=args.session_id,
                token_counter=args.token_counter,
            )
        elif args.command == "settle-run":
            payload = finish_project_usage(
                workspace_root=args.workspace_root,
                session_id=automatic_usage_session_id(
                    args.project_id,
                    args.revision,
                ),
                token_counter=args.token_counter,
            )
        elif args.command == "close-run":
            payload = close_project_usage_wall_time(
                workspace_root=args.workspace_root,
                project_id=args.project_id,
                revision=args.revision,
                run_status=args.run_status,
            )
        elif args.command == "migrate-legacy-tokens":
            payload = migrate_legacy_token_sessions(
                workspace_root=args.workspace_root,
                project_id=args.project_id,
                apply=args.apply,
            )
        else:
            payload = rebuild_project_usage_summary(
                workspace_root=args.workspace_root,
                project_id=args.project_id,
            )
    except (JsonDocumentError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "error_code": type(exc).__name__,
                    "message": str(exc),
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


# 2026-08-13 ATTENTION AI EDITORS: token accounting now comes from the
# durable per-project Codex ledger below, fed by ``turn.completed.usage``
# events (input_tokens + output_tokens; cached input is already included
# in input_tokens).  The kimi-code reader is retained only so that legacy
# sessions can be recognized and migrated; it is never consulted for new
# accounting.  The begin/finish wrappers are fail-soft on purpose: usage
# recording must never break a run.

CODEX_TOKEN_SOURCE = "codex_turn_usage"
_CODEX_LEDGER_SCHEMA = "simulation_ready_cad_codex_token_ledger_v1"

KIMI_WORKER_TITLE_FINGERPRINT = "mechanical-design agent"
KIMI_WIRE_TOKEN_SOURCE = "kimi_wire_worker_turn_usage"
_USAGE_RECORD_KEYS = (
    "inputOther",
    "output",
    "inputCacheRead",
    "inputCacheCreation",
)
_UNSET: object = object()


def kimi_worker_token_counter(
    *,
    workspace_root: str | Path | None = None,
    sessions_root: str | Path | None = None,
    title_fingerprint: str = KIMI_WORKER_TITLE_FINGERPRINT,
) -> int | None:
    """Cumulative worker-session tokens from kimi-code wire records.

    Only per-turn usage records are summed (session-scope records restate
    cumulative totals). When ``workspace_root`` is given, only the session
    bucket of that checkout (``wd_<dirname>_*``) is considered, and only
    sessions whose title carries the worker fingerprint count. Returns
    ``None`` when no matching record exists (counter unavailable).
    """

    root = (
        Path(sessions_root)
        if sessions_root is not None
        else Path.home() / ".kimi-code" / "sessions"
    )
    if not root.is_dir():
        return None
    prefix = ""
    if workspace_root is not None:
        prefix = f"wd_{Path(workspace_root).resolve().name}_"
    total = 0
    matched = False
    for bucket in sorted(root.iterdir()):
        if not bucket.is_dir() or bucket.is_symlink():
            continue
        if prefix and not bucket.name.startswith(prefix):
            continue
        for state_path in sorted(bucket.glob("session_*/state.json")):
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(state, dict):
                continue
            if title_fingerprint not in str(state.get("title") or ""):
                continue
            for wire in sorted(state_path.parent.glob("agents/*/wire.jsonl")):
                if wire.is_symlink():
                    continue
                try:
                    lines = wire.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                except OSError:
                    continue
                for line in lines:
                    if '"usage.record"' not in line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    if (
                        record.get("type") != "usage.record"
                        or record.get("usageScope") != "turn"
                    ):
                        continue
                    usage = record.get("usage")
                    if not isinstance(usage, dict):
                        continue
                    matched = True
                    for key in _USAGE_RECORD_KEYS:
                        value = usage.get(key, 0)
                        if isinstance(value, bool):
                            continue
                        if isinstance(value, (int, float)) and math.isfinite(value):
                            total += int(value)
    return total if matched else None


def _codex_ledger_path(root: Path, project_id: str) -> Path:
    return root / "codex_ledger" / f"{project_id}.json"


def _ledger_allocated_total(ledger: dict[str, object]) -> int:
    entries = ledger.get("entries")
    if "unallocated_backfill_tokens" in ledger and isinstance(entries, list):
        return sum(
            _token_counter(item.get("total_tokens"), "entry.total_tokens")
            for item in entries
            if isinstance(item, dict)
            and item.get("allocation") != "unallocated_backfill"
        )
    return _token_counter(
        ledger.get("allocated_total_tokens", ledger.get("total_tokens")),
        "allocated_total_tokens",
    )


def _counter_allocated_total(
    ledger: dict[str, object], counter_id: str, fallback: int
) -> int:
    entries = ledger.get("entries")
    if "unallocated_backfill_tokens" in ledger and isinstance(entries, list):
        return sum(
            _token_counter(item.get("total_tokens"), "entry.total_tokens")
            for item in entries
            if isinstance(item, dict)
            and str(item.get("counter_id") or "default") == counter_id
            and item.get("allocation") != "unallocated_backfill"
        )
    return fallback


def record_codex_turn_usage(
    *,
    workspace_root: str | Path,
    project_id: str,
    input_tokens: int,
    output_tokens: int,
    counter_id: str = "default",
    allocate_delta: bool = True,
    context: str = "",
) -> dict[str, object]:
    """Record one cumulative Codex ``turn.completed.usage`` reading.

    Codex reports conversation-to-date counters.  The project ledger therefore
    adds only the monotonic difference from the preceding reading for the same
    ``counter_id``.  Visible Worker and Reviewer conversations use independent
    counter ids; a backwards reading within one counter is rejected.
    """

    project = _identifier(project_id, "project_id")
    counter = _identifier(counter_id, "counter_id")
    if not isinstance(allocate_delta, bool):
        raise JsonDocumentError("allocate_delta must be boolean")
    tokens_in = _token_counter(input_tokens, "input_tokens")
    tokens_out = _token_counter(output_tokens, "output_tokens")
    workspace, root = _usage_root(workspace_root)
    destination = _codex_ledger_path(root, project)
    if destination.exists():
        ledger = read_json(destination)
        if ledger.get("schema_version") != _CODEX_LEDGER_SCHEMA:
            raise JsonDocumentError(
                f"codex token ledger has an unknown schema: {destination}"
            )
    else:
        ledger = {
            "schema_version": _CODEX_LEDGER_SCHEMA,
            "project_id": project,
            "total_tokens": 0,
            "entries": [],
        }
    raw_counters = ledger.get("counters") or {}
    if not isinstance(raw_counters, dict):
        raise JsonDocumentError("codex token ledger counters are malformed")
    counters = dict(raw_counters)
    prior_counter = counters.get(counter) or {}
    if not isinstance(prior_counter, dict):
        raise JsonDocumentError("codex token ledger counter is malformed")
    previous_in = _token_counter(
        prior_counter.get(
            "last_cumulative_input_tokens",
            ledger.get("last_cumulative_input_tokens", 0)
            if counter == "default"
            else 0,
        ),
        "last_cumulative_input_tokens",
    )
    previous_out = _token_counter(
        prior_counter.get(
            "last_cumulative_output_tokens",
            ledger.get("last_cumulative_output_tokens", 0)
            if counter == "default"
            else 0,
        ),
        "last_cumulative_output_tokens",
    )
    if tokens_in < previous_in or tokens_out < previous_out:
        raise JsonDocumentError(
            "Codex session cumulative token usage moved backwards"
        )
    delta_in = tokens_in - previous_in
    delta_out = tokens_out - previous_out
    entry = {
        "recorded_at": _timestamp(float(time.time())),
        "cumulative_input_tokens": tokens_in,
        "cumulative_output_tokens": tokens_out,
        "input_tokens": delta_in,
        "output_tokens": delta_out,
        "total_tokens": delta_in + delta_out,
        "counter_id": counter,
        "allocation": (
            "revision" if allocate_delta else "unallocated_backfill"
        ),
        "context": str(context)[:256],
    }
    ledger["total_tokens"] = int(ledger["total_tokens"]) + entry["total_tokens"]
    allocated_total = _ledger_allocated_total(ledger) - (
        entry["total_tokens"]
        if allocate_delta and "unallocated_backfill_tokens" not in ledger
        else 0
    )
    unallocated_total = _token_counter(
        ledger.get("unallocated_backfill_tokens", 0),
        "unallocated_backfill_tokens",
    )
    if allocate_delta:
        allocated_total += entry["total_tokens"]
    else:
        unallocated_total += entry["total_tokens"]
    ledger["allocated_total_tokens"] = allocated_total
    ledger["unallocated_backfill_tokens"] = unallocated_total
    prior_total = _token_counter(
        prior_counter.get(
            "total_tokens",
            previous_in + previous_out,
        ),
        "counter.total_tokens",
    )
    prior_allocated = _counter_allocated_total(
        ledger,
        counter,
        _token_counter(
            prior_counter.get("allocated_tokens", prior_total),
            "counter.allocated_tokens",
        ),
    )
    counters[counter] = {
        "last_cumulative_input_tokens": tokens_in,
        "last_cumulative_output_tokens": tokens_out,
        "total_tokens": prior_total + entry["total_tokens"],
        "allocated_tokens": (
            prior_allocated + entry["total_tokens"]
            if allocate_delta
            else prior_allocated
        ),
    }
    ledger["counters"] = counters
    if counter == "default":
        ledger["last_cumulative_input_tokens"] = tokens_in
        ledger["last_cumulative_output_tokens"] = tokens_out
    entries = list(ledger.get("entries") or [])
    entries.append(entry)
    ledger["entries"] = entries[-4096:]
    ledger["ledger_artifact"] = relative_to_root(workspace, destination)
    write_json_atomic(destination, ledger)
    return ledger


def codex_project_token_counter(
    *,
    workspace_root: str | Path,
    project_id: str,
) -> int:
    """Cumulative Codex tokens recorded for one project (0 before any turn)."""

    project = _identifier(project_id, "project_id")
    _workspace, root = _usage_root(workspace_root)
    destination = _codex_ledger_path(root, project)
    if not destination.is_file() or destination.is_symlink():
        return 0
    ledger = read_json(destination)
    if ledger.get("schema_version") != _CODEX_LEDGER_SCHEMA:
        raise JsonDocumentError(
            f"codex token ledger has an unknown schema: {destination}"
        )
    return _ledger_allocated_total(ledger)


def codex_project_unallocated_token_counter(
    *,
    workspace_root: str | Path,
    project_id: str,
) -> int:
    """Historical visible-role tokens known only at project scope."""

    project = _identifier(project_id, "project_id")
    _workspace, root = _usage_root(workspace_root)
    destination = _codex_ledger_path(root, project)
    if not destination.is_file() or destination.is_symlink():
        return 0
    ledger = read_json(destination)
    if ledger.get("schema_version") != _CODEX_LEDGER_SCHEMA:
        raise JsonDocumentError(
            f"codex token ledger has an unknown schema: {destination}"
        )
    entries = ledger.get("entries")
    if isinstance(entries, list) and "unallocated_backfill_tokens" in ledger:
        return sum(
            _token_counter(item.get("total_tokens"), "entry.total_tokens")
            for item in entries
            if isinstance(item, dict)
            and item.get("allocation") == "unallocated_backfill"
        )
    return _token_counter(
        ledger.get("unallocated_backfill_tokens", 0),
        "unallocated_backfill_tokens",
    )


def migrate_codex_visible_backfill_allocation(
    *,
    workspace_root: str | Path,
    project_id: str,
    apply: bool = False,
) -> dict[str, object]:
    """Separate pre-recorder visible-role history from revision deltas.

    Early visible-role ledgers treated each conversation's first cumulative
    reading as a current-revision delta.  For projects with older completed
    zero-token sessions, the first synchronization batch and explicitly
    labelled historical backfills are project history, not evidence for one
    rNN.  This migration preserves the project total while shifting cumulative
    session baselines so later revision deltas remain exact.
    """

    project = _identifier(project_id, "project_id")
    workspace, root = _usage_root(workspace_root)
    destination = _codex_ledger_path(root, project)
    if not destination.is_file() or destination.is_symlink():
        return {"project_id": project, "status": "NO_LEDGER", "applied": False}
    ledger = read_json(destination)
    if ledger.get("schema_version") != _CODEX_LEDGER_SCHEMA:
        raise JsonDocumentError(
            f"codex token ledger has an unknown schema: {destination}"
        )
    if "allocated_total_tokens" in ledger:
        unallocated = _token_counter(
            ledger.get("unallocated_backfill_tokens", 0),
            "unallocated_backfill_tokens",
        )
        reconciled = 0
        if apply and unallocated:
            for session in _project_sessions(root, project):
                if session.get("token_counter_source") != CODEX_TOKEN_SOURCE:
                    continue
                migrated = dict(session)
                changed = False
                for field in ("token_counter_start", "token_counter_end"):
                    value = migrated.get(field)
                    marker = migrated.get(f"{field}_allocation")
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and marker != "allocated"
                    ):
                        migrated[field] = (
                            value - unallocated if value >= unallocated else value
                        )
                        migrated[f"{field}_allocation"] = "allocated"
                        changed = True
                start = migrated.get("token_counter_start")
                end = migrated.get("token_counter_end")
                if isinstance(start, int) and isinstance(end, int):
                    migrated["total_tokens"] = end - start
                if changed:
                    write_json_atomic(
                        _session_path(root, str(migrated["session_id"])),
                        migrated,
                    )
                    reconciled += 1
            rebuild_project_usage_summary(
                workspace_root=workspace,
                project_id=project,
            )
        return {
            "project_id": project,
            "status": "RECONCILED" if reconciled else "ALREADY_MIGRATED",
            "applied": bool(reconciled),
            "unallocated_backfill_tokens": unallocated,
            "session_count": reconciled,
        }
    sessions = _project_sessions(root, project)
    completed_zero = [
        item
        for item in sessions
        if item.get("status") == "COMPLETED"
        and item.get("token_counter_source") == CODEX_TOKEN_SOURCE
        and int(item.get("total_tokens") or 0) == 0
    ]
    entries = list(ledger.get("entries") or [])
    if not completed_zero or not entries:
        return {
            "project_id": project,
            "status": "NOT_HISTORICAL_BACKFILL",
            "applied": False,
        }
    try:
        first_recorded = min(
            datetime.fromisoformat(str(item["recorded_at"])) for item in entries
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise JsonDocumentError("codex token ledger timestamps are malformed") from exc
    initial_cutoff = first_recorded + timedelta(seconds=10)
    migrated_entries: list[dict[str, object]] = []
    unallocated = 0
    unallocated_by_counter: dict[str, int] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise JsonDocumentError("codex token ledger entry is malformed")
        item = dict(raw)
        recorded = datetime.fromisoformat(str(item.get("recorded_at") or ""))
        historical = bool(
            str(item.get("context") or "") == "historical-visible-role-backfill"
            or recorded <= initial_cutoff
        )
        amount = _token_counter(item.get("total_tokens"), "entry.total_tokens")
        item["allocation"] = (
            "unallocated_backfill" if historical else "revision"
        )
        if historical:
            unallocated += amount
            counter_id = str(item.get("counter_id") or "default")
            unallocated_by_counter[counter_id] = (
                unallocated_by_counter.get(counter_id, 0) + amount
            )
        migrated_entries.append(item)
    total = _token_counter(ledger.get("total_tokens"), "total_tokens")
    if unallocated > total:
        raise JsonDocumentError("unallocated backfill exceeds ledger total")
    migrated_ledger = dict(ledger)
    migrated_ledger["entries"] = migrated_entries
    migrated_ledger["allocated_total_tokens"] = total - unallocated
    migrated_ledger["unallocated_backfill_tokens"] = unallocated
    counters = dict(migrated_ledger.get("counters") or {})
    for counter_id, raw in list(counters.items()):
        if not isinstance(raw, dict):
            raise JsonDocumentError("codex token ledger counter is malformed")
        counter = dict(raw)
        counter_total = _token_counter(
            counter.get("total_tokens"), "counter.total_tokens"
        )
        counter["allocated_tokens"] = max(
            0, counter_total - unallocated_by_counter.get(counter_id, 0)
        )
        counters[counter_id] = counter
    migrated_ledger["counters"] = counters
    migrated_sessions: list[tuple[Path, dict[str, object]]] = []
    for session in sessions:
        if session.get("token_counter_source") != CODEX_TOKEN_SOURCE:
            continue
        migrated = dict(session)
        for field in ("token_counter_start", "token_counter_end"):
            value = migrated.get(field)
            if isinstance(value, int) and not isinstance(value, bool):
                migrated[field] = max(0, value - unallocated)
                migrated[f"{field}_allocation"] = "allocated"
        start = migrated.get("token_counter_start")
        end = migrated.get("token_counter_end")
        if isinstance(start, int) and isinstance(end, int):
            migrated["total_tokens"] = end - start
        migrated_sessions.append(
            (_session_path(root, str(migrated["session_id"])), migrated)
        )
    report: dict[str, object] = {
        "project_id": project,
        "status": "MIGRATION_READY",
        "applied": bool(apply),
        "unallocated_backfill_tokens": unallocated,
        "allocated_total_tokens": total - unallocated,
        "session_count": len(migrated_sessions),
    }
    if apply:
        write_json_atomic(destination, migrated_ledger)
        for path, session in migrated_sessions:
            write_json_atomic(path, session)
        rebuild_project_usage_summary(
            workspace_root=workspace,
            project_id=project,
        )
        report["status"] = "MIGRATED"
    return report


def begin_run_usage(
    *,
    workspace_root: str | Path,
    project_id: str,
    revision: str,
    run_id: str,
    token_counter: int | None | object = _UNSET,
) -> None:
    """Best-effort usage begin for one canonical run; never raises."""

    try:
        counter = (
            codex_project_token_counter(
                workspace_root=workspace_root,
                project_id=project_id,
            )
            if token_counter is _UNSET
            else token_counter
        )
        begin_project_usage(
            workspace_root=workspace_root,
            project_id=project_id,
            session_id=automatic_usage_session_id(project_id, revision),
            run_id=run_id,
            token_counter=counter,  # type: ignore[arg-type]
            token_counter_source=CODEX_TOKEN_SOURCE,
            token_counter_allocation="allocated",
        )
    except Exception:  # noqa: BLE001 - recording must never break a run
        return


def finish_run_usage(
    run_dir: str | Path,
    *,
    run_status: str = "",
    token_counter: int | None | object = _UNSET,
) -> None:
    """Best-effort usage close for a terminal run; never raises."""

    try:
        from .run_layout import parse_canonical_run

        run = Path(run_dir).resolve()
        workspace = run.parents[4]
        identity = parse_canonical_run(workspace, run)
        if identity is None:
            return
        project = identity.project_id
        revision = identity.revision
        session_id = automatic_usage_session_id(project, revision)
        _workspace, root = _usage_root(workspace)
        session_path = _session_path(root, session_id)
        if not session_path.is_file() or session_path.is_symlink():
            return  # run predates the recorder; nothing to close
        payload = read_json(session_path)
        if payload.get("status") not in {"RUNNING", "AWAITING_TOKEN_COUNTER"}:
            return
        if token_counter is _UNSET:
            # Only sessions opened against the Codex ledger may settle from
            # it; legacy sessions (e.g. Kimi-based baselines) keep their
            # wall time honest and wait for the one-time migration entry.
            counter = (
                codex_project_token_counter(
                    workspace_root=workspace,
                    project_id=project,
                )
                if payload.get("token_counter_source") == CODEX_TOKEN_SOURCE
                else None
            )
        else:
            counter = token_counter
        start = payload.get("token_counter_start")
        if counter is None or start is None:
            close_project_usage_wall_time(
                workspace_root=workspace,
                project_id=project,
                revision=revision,
                run_status=run_status or "TERMINAL",
            )
            return
        finish_project_usage(
            workspace_root=workspace,
            session_id=session_id,
            token_counter=counter,  # type: ignore[arg-type]
            token_counter_allocation=(
                "allocated"
                if payload.get("token_counter_source") == CODEX_TOKEN_SOURCE
                else None
            ),
        )
    except Exception:  # noqa: BLE001 - recording must never break a run
        return


def migrate_legacy_token_sessions(
    *,
    workspace_root: str | Path,
    project_id: str,
    apply: bool = False,
    clock: Callable[[], float] = time.time,
) -> dict[str, object]:
    """One-time fix for sessions stuck on a legacy (Kimi) token counter.

    Kimi sessions left RUNNING or AWAITING_TOKEN_COUNTER can never settle
    against the Codex ledger.  Tokens remain explicitly unmeasured, while
    wall time is retained from an existing completion or derived from the
    next revision's start (falling back to the latest run artifact).  With
    ``apply=False`` (default) this only reports what would change.
    """

    project = _identifier(project_id, "project_id")
    workspace, root = _usage_root(workspace_root)
    project_sessions = _project_sessions(root, project)
    candidates = [
        item
        for item in project_sessions
        if item.get("token_counter_source") == KIMI_WIRE_TOKEN_SOURCE
        and item.get("status") in {"RUNNING", "AWAITING_TOKEN_COUNTER"}
    ]
    migrated: list[dict[str, object]] = []
    for payload in candidates:
        session_id = _identifier(str(payload.get("session_id")), "session_id")
        was_running = payload.get("status") == "RUNNING"
        closed = dict(payload)
        wall_time_basis = "recorded_completion"
        if was_running:
            started = float(payload["started_at_epoch_seconds"])
            revision = session_id.rsplit("-", 1)[-1]
            later_starts = [
                float(item["started_at_epoch_seconds"])
                for item in project_sessions
                if float(item.get("started_at_epoch_seconds", -1.0)) > started
            ]
            finished = min(later_starts, default=None)
            wall_time_basis = "next_session_started_at"
            run_dir = (
                workspace
                / ".runtime_runs"
                / "projects"
                / project
                / "runs"
                / revision
            )
            if finished is None:
                artifact_times = (
                    [
                        path.stat().st_mtime
                        for path in run_dir.rglob("*")
                        if path.is_file() and not path.is_symlink()
                    ]
                    if run_dir.is_dir()
                    else []
                )
                finished = max(artifact_times, default=None)
                wall_time_basis = "latest_run_artifact_mtime"
            if finished is not None and finished >= started:
                closed.update(
                    {
                        "completed_at": _timestamp(finished),
                        "completed_at_epoch_seconds": finished,
                        "wall_time_seconds": round(finished - started, 6),
                    }
                )
        closed.update(
            {
                "status": "CLOSED_UNMEASURED",
                "closed_unmeasured_at": _timestamp(float(clock())),
                "migration_reason": (
                    "legacy token counter source "
                    f"{payload.get('token_counter_source')!r} cannot settle "
                    "against the Codex ledger"
                ),
                "wall_time_measured": closed.get("wall_time_seconds") is not None,
                "wall_time_basis": wall_time_basis,
            }
        )
        if apply:
            write_json_atomic(_session_path(root, session_id), closed)
        migrated.append(
            {
                "session_id": session_id,
                "run_id": closed.get("run_id"),
                "previous_status": payload.get("status"),
                "wall_time_seconds": closed.get("wall_time_seconds"),
                "wall_time_measured": closed["wall_time_measured"],
                "wall_time_basis": closed["wall_time_basis"],
            }
        )
    if apply and migrated:
        rebuild_project_usage_summary(
            workspace_root=workspace,
            project_id=project,
        )
    return {
        "project_id": project,
        "applied": bool(apply),
        "migrated_session_count": len(migrated),
        "migrated_sessions": migrated,
    }


if __name__ == "__main__":
    raise SystemExit(main())
