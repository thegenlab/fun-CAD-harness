from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


_REVISION = re.compile(r"r(0*[1-9][0-9]*)\Z", flags=re.IGNORECASE)


def _revision_from_run_id(run_id: str) -> str:
    match = re.search(
        r"(?:^|-)(r0*[1-9][0-9]*)\Z",
        str(run_id),
        flags=re.IGNORECASE,
    )
    return "" if match is None else match.group(1).lower()


def _revision_inventory(
    ctx: Any,
    project_id: str,
    *,
    completed_sessions: list[dict[str, Any]] | None = None,
    running_sessions: list[dict[str, Any]] | None = None,
    pending_token_sessions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    runs_root = (
        ctx.workspace_root
        / ".runtime_runs"
        / "projects"
        / project_id
        / "runs"
    )
    if not runs_root.is_dir():
        return []
    completed_by_revision: dict[str, list[dict[str, Any]]] = {}
    running_by_revision: dict[str, list[dict[str, Any]]] = {}
    pending_by_revision: dict[str, list[dict[str, Any]]] = {}
    for item in completed_sessions or []:
        revision = _revision_from_run_id(str(item.get("run_id") or ""))
        if revision:
            completed_by_revision.setdefault(revision, []).append(item)
    for item in running_sessions or []:
        revision = _revision_from_run_id(str(item.get("run_id") or ""))
        if revision:
            running_by_revision.setdefault(revision, []).append(item)
    for item in pending_token_sessions or []:
        revision = _revision_from_run_id(str(item.get("run_id") or ""))
        if revision:
            pending_by_revision.setdefault(revision, []).append(item)

    retained: list[dict[str, Any]] = []
    candidates = [
        path
        for path in runs_root.iterdir()
        if path.is_dir() and not path.is_symlink() and _REVISION.fullmatch(path.name)
    ]
    candidates.sort(
        key=lambda path: int(_REVISION.fullmatch(path.name).group(1)),  # type: ignore[union-attr]
        reverse=True,
    )
    for path in candidates:
        revision = path.name.lower()
        completed = completed_by_revision.get(revision, [])
        running = running_by_revision.get(revision, [])
        pending = pending_by_revision.get(revision, [])
        status_path = path / "run_status.json"
        run_status = "UNFINISHED"
        if status_path.is_file() and not status_path.is_symlink():
            try:
                value = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                value = {}
            if isinstance(value, dict):
                run_status = str(value.get("status") or run_status)
        usage_status = (
            "running"
            if running
            else (
                "awaiting_token"
                if pending
                else ("recorded" if completed else "missing")
            )
        )
        retained.append(
            {
                "revision": revision,
                "run_id": f"{project_id}-{revision}",
                "run_status": run_status,
                "usage_status": usage_status,
                "wall_time_seconds": (
                    sum(float(item["wall_time_seconds"]) for item in completed)
                    if completed
                    else (
                        sum(float(item["wall_time_seconds"]) for item in pending)
                        if pending
                        else None
                    )
                ),
                "total_tokens": (
                    sum(int(item["total_tokens"]) for item in completed)
                    if completed
                    else (
                        sum(int(item.get("total_tokens") or 0) for item in running)
                        if running
                        else None
                    )
                ),
                "completed_at": (
                    max(str(item.get("completed_at") or "") for item in completed)
                    if completed
                    else (
                        max(str(item.get("completed_at") or "") for item in pending)
                        if pending
                        else ""
                    )
                ),
                "started_at": (
                    min(str(item.get("started_at") or "") for item in running)
                    if running
                    else ""
                ),
                "session_count": len(completed) + len(running) + len(pending),
                "token_counter_sources": sorted(
                    {
                        str(item.get("token_counter_source") or "")
                        for item in (*completed, *running, *pending)
                        if item.get("token_counter_source")
                    }
                ),
            }
        )
    return retained


def load_project_usage(
    ctx: Any, package_id: str, run_id: str = ""
) -> dict[str, Any]:
    """Load one project-level runtime/token ledger without guessing missing values."""

    projects_root = (
        ctx.workspace_root / ".runtime_runs" / "project_usage" / "projects"
    )
    current_revision = (
        str(run_id).lower() if _REVISION.fullmatch(str(run_id)) else ""
    )
    revision_records = _revision_inventory(ctx, package_id)
    empty = {
        "status": "unrecorded",
        "recorded": False,
        "package_id": package_id,
        "project_id": package_id,
        "binding": "none",
        "current_revision": current_revision,
        "revision_count": len(revision_records),
        "recorded_revision_count": 0,
        "missing_revision_count": len(revision_records),
        "revision_records": revision_records,
        "total_wall_time_seconds": None,
        "total_tokens": None,
        "benchmark_total_tokens": None,
        "benchmark_token_counter_source": "",
        "unallocated_backfill_tokens": 0,
        "token_totals_by_source": [],
        "mixed_token_counter_sources": False,
        "completed_session_count": None,
        "running_session_count": None,
        "pending_token_session_count": None,
        "completed_sessions": [],
        "running_sessions": [],
        "pending_token_sessions": [],
        "source_path": "",
        "issues": [],
    }
    if (
        not package_id
        or package_id in {".", ".."}
        or Path(package_id).name != package_id
        or any(separator in package_id for separator in ("/", "\\"))
    ):
        issue = "项目 ID 不能安全映射到投入记录文件"
        ctx.diagnostics.append(
            {
                "level": "malformed",
                "role": "project usage",
                "path": "",
                "detail": issue,
            }
        )
        return {**empty, "status": "unavailable", "issues": [issue]}

    usage_path = projects_root / f"{package_id}.json"
    binding = "package_id"
    if not usage_path.is_file() and run_id:
        candidates: list[Path] = []
        for candidate in sorted(projects_root.glob("*.json")):
            try:
                candidate_raw = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(candidate_raw, dict):
                continue
            recorded_run_ids = [str(item) for item in candidate_raw.get("run_ids", [])]
            if run_id in recorded_run_ids:
                candidates.append(candidate)
        if len(candidates) == 1:
            usage_path = candidates[0]
            binding = "run_id"
        elif len(candidates) > 1:
            issue = "同一个 run_id 出现在多个项目投入记录中"
            ctx.diagnostics.append(
                {
                    "level": "malformed",
                    "role": "project usage",
                    "path": ctx.relative(projects_root),
                    "detail": issue,
                }
            )
            return {**empty, "status": "unavailable", "issues": [issue]}

    source_path = ctx.relative(usage_path)
    if not usage_path.is_file():
        return {**empty, "source_path": source_path}

    raw = ctx.read_json(usage_path, role="project usage")
    recorded_project_id = str(raw.get("project_id") or "")
    if raw.get("benchmark_token_counter_source") == "codex_turn_usage":
        try:
            from ..project_usage import (
                codex_project_token_counter,
                codex_project_unallocated_token_counter,
            )

            allocated_counter = codex_project_token_counter(
                workspace_root=ctx.workspace_root,
                project_id=recorded_project_id or package_id,
            )
            unallocated_counter = codex_project_unallocated_token_counter(
                workspace_root=ctx.workspace_root,
                project_id=recorded_project_id or package_id,
            )
            def allocated_session_counters(
                session_id: str,
            ) -> tuple[int | None, int | None]:
                session_path = (
                    ctx.workspace_root
                    / ".runtime_runs"
                    / "project_usage"
                    / "sessions"
                    / f"{session_id}.json"
                )
                if not session_path.is_file() or session_path.is_symlink():
                    return None, None
                session = ctx.read_json(
                    session_path, role="project usage session"
                )
                values: list[int | None] = []
                for field in ("token_counter_start", "token_counter_end"):
                    value = session.get(field)
                    if not isinstance(value, int) or isinstance(value, bool):
                        values.append(None)
                    elif session.get(f"{field}_allocation") == "allocated":
                        values.append(value)
                    else:
                        values.append(max(0, value - unallocated_counter))
                return values[0], values[1]

            completed_tokens = 0
            refreshed_completed: list[dict[str, Any]] = []
            for item in raw.get("completed_sessions") or []:
                if not isinstance(item, dict):
                    continue
                refreshed = dict(item)
                start, end = allocated_session_counters(
                    str(item.get("session_id") or "")
                )
                if start is not None and end is not None:
                    refreshed["total_tokens"] = max(0, end - start)
                completed_tokens += int(refreshed.get("total_tokens") or 0)
                refreshed_completed.append(refreshed)
            running_tokens = 0
            refreshed_running: list[dict[str, Any]] = []
            for item in raw.get("running_sessions") or []:
                if not isinstance(item, dict):
                    continue
                refreshed = dict(item)
                session_id = str(item.get("session_id") or "")
                baseline, _end = allocated_session_counters(session_id)
                baseline = baseline or 0
                refreshed["total_tokens"] = max(
                    0, allocated_counter - baseline
                )
                running_tokens += int(refreshed["total_tokens"])
                refreshed_running.append(refreshed)
            raw = {
                **raw,
                "completed_sessions": refreshed_completed,
                "running_sessions": refreshed_running,
                "unallocated_backfill_tokens": unallocated_counter,
                "total_tokens": (
                    completed_tokens + running_tokens + unallocated_counter
                ),
            }
        except (OSError, TypeError, ValueError):
            pass
    issues: list[str] = []
    if not raw:
        issues.append("投入记录不是有效的 JSON 对象")
    if raw.get("schema_version") != "0.1":
        issues.append("投入记录 schema_version 不是 0.1")
    if binding == "package_id" and recorded_project_id != package_id:
        issues.append("投入记录的 project_id 与当前机械不一致")
    if binding == "run_id" and run_id not in [str(item) for item in raw.get("run_ids", [])]:
        issues.append("投入记录不再包含用于绑定的 run_id")

    def nonnegative_int(value: Any, field: str) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(f"{field} 必须是非负整数")
            return None
        return value

    def nonnegative_number(value: Any, field: str) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            issues.append(f"{field} 必须是非负数")
            return None
        number = float(value)
        if not math.isfinite(number) or number < 0:
            issues.append(f"{field} 必须是有限的非负数")
            return None
        return number

    def normalize_sessions(value: Any, *, running: bool) -> list[dict[str, Any]]:
        label = "running_sessions" if running else "completed_sessions"
        if not isinstance(value, list):
            issues.append(f"{label} 必须是数组")
            return []
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                issues.append(f"{label}[{index}] 必须是对象")
                continue
            session_id = str(item.get("session_id") or "").strip()
            run_id = str(item.get("run_id") or "").strip()
            if not session_id:
                issues.append(f"{label}[{index}].session_id 不能为空")
            if not run_id:
                issues.append(f"{label}[{index}].run_id 不能为空")
            session: dict[str, Any] = {
                "session_id": session_id,
                "run_id": run_id,
                "token_counter_source": str(item.get("token_counter_source") or ""),
            }
            if running:
                session["started_at"] = str(item.get("started_at") or "")
                session["total_tokens"] = nonnegative_int(
                    item.get("total_tokens", 0),
                    f"{label}[{index}].total_tokens",
                )
            else:
                session["wall_time_seconds"] = nonnegative_number(
                    item.get("wall_time_seconds"),
                    f"{label}[{index}].wall_time_seconds",
                )
                session["total_tokens"] = nonnegative_int(
                    item.get("total_tokens"),
                    f"{label}[{index}].total_tokens",
                )
                session["completed_at"] = str(item.get("completed_at") or "")
            normalized.append(session)
        return normalized

    def normalize_pending_sessions(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if not isinstance(value, list):
            issues.append("pending_token_sessions 必须是数组")
            return []
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                issues.append(f"pending_token_sessions[{index}] 必须是对象")
                continue
            normalized.append(
                {
                    "session_id": str(item.get("session_id") or ""),
                    "run_id": str(item.get("run_id") or ""),
                    "wall_time_seconds": nonnegative_number(
                        item.get("wall_time_seconds"),
                        f"pending_token_sessions[{index}].wall_time_seconds",
                    ),
                    "completed_at": str(item.get("completed_at") or ""),
                    "token_counter_source": str(
                        item.get("token_counter_source") or ""
                    ),
                    "token_counter_start": nonnegative_int(
                        item.get("token_counter_start"),
                        f"pending_token_sessions[{index}].token_counter_start",
                    ),
                }
            )
        return normalized

    completed_count = nonnegative_int(
        raw.get("completed_session_count"), "completed_session_count"
    )
    running_count = nonnegative_int(
        raw.get("running_session_count"), "running_session_count"
    )
    pending_count = nonnegative_int(
        raw.get("pending_token_session_count", 0),
        "pending_token_session_count",
    )
    total_wall_time = nonnegative_number(
        raw.get("total_wall_time_seconds"), "total_wall_time_seconds"
    )
    raw_total_tokens_value = raw.get("total_tokens")
    total_tokens = (
        None
        if raw_total_tokens_value is None
        else nonnegative_int(raw_total_tokens_value, "total_tokens")
    )
    unallocated_backfill_tokens = nonnegative_int(
        raw.get("unallocated_backfill_tokens", 0),
        "unallocated_backfill_tokens",
    )
    completed_sessions = normalize_sessions(raw.get("completed_sessions"), running=False)
    running_sessions = normalize_sessions(raw.get("running_sessions"), running=True)
    pending_token_sessions = normalize_pending_sessions(
        raw.get("pending_token_sessions", [])
    )

    if completed_count is not None and completed_count != len(completed_sessions):
        issues.append("completed_session_count 与会话明细数量不一致")
    if running_count is not None and running_count != len(running_sessions):
        issues.append("running_session_count 与会话明细数量不一致")
    if pending_count is not None and pending_count != len(pending_token_sessions):
        issues.append("pending_token_session_count 与会话明细数量不一致")
    if total_wall_time is not None and all(
        item.get("wall_time_seconds") is not None for item in completed_sessions
    ):
        detail_wall_time = sum(
            float(item["wall_time_seconds"]) for item in completed_sessions
        )
        if not math.isclose(
            total_wall_time, detail_wall_time, rel_tol=1e-9, abs_tol=1e-6
        ):
            issues.append("total_wall_time_seconds 与已完成会话之和不一致")

    totals_by_source: dict[str, int] = {}
    sessions_by_source: dict[str, int] = {}
    for item in (*completed_sessions, *running_sessions):
        source = str(item.get("token_counter_source") or "")
        if not source or item.get("total_tokens") is None:
            continue
        totals_by_source[source] = (
            totals_by_source.get(source, 0) + int(item["total_tokens"])
        )
        sessions_by_source[source] = sessions_by_source.get(source, 0) + 1
    if unallocated_backfill_tokens:
        backfill_source = str(
            raw.get("benchmark_token_counter_source") or "codex_turn_usage"
        )
        totals_by_source[backfill_source] = (
            totals_by_source.get(backfill_source, 0)
            + unallocated_backfill_tokens
        )
        sessions_by_source.setdefault(backfill_source, 0)
    if "codex_goal_counter" in totals_by_source:
        benchmark_source = "codex_goal_counter"
    elif len(totals_by_source) == 1:
        benchmark_source = next(iter(totals_by_source))
    else:
        benchmark_source = ""
    benchmark_total_tokens = (
        totals_by_source.get(benchmark_source)
        if benchmark_source
        else None
    )
    token_totals_by_source = [
        {
            "token_counter_source": source,
            "total_tokens": value,
            "completed_session_count": sessions_by_source[source],
            "included_in_benchmark": source == benchmark_source,
        }
        for source, value in sorted(totals_by_source.items())
    ]
    if issues:
        ctx.diagnostics.append(
            {
                "level": "malformed",
                "role": "project usage",
                "path": source_path,
                "detail": "；".join(issues),
            }
        )
        return {
            **empty,
            "status": "unavailable",
            "source_path": source_path,
            "issues": issues,
        }

    completed_sessions.sort(
        key=lambda item: str(item.get("completed_at") or ""), reverse=True
    )
    revision_records = _revision_inventory(
        ctx,
        package_id,
        completed_sessions=completed_sessions,
        running_sessions=running_sessions,
        pending_token_sessions=pending_token_sessions,
    )
    recorded_revision_count = sum(
        item["usage_status"] == "recorded" for item in revision_records
    )
    missing_revision_count = sum(
        item["usage_status"] == "missing" for item in revision_records
    )
    return {
        "status": "recorded",
        "recorded": True,
        "package_id": package_id,
        "project_id": recorded_project_id,
        "binding": binding,
        "current_revision": current_revision,
        "revision_count": len(revision_records),
        "recorded_revision_count": recorded_revision_count,
        "missing_revision_count": missing_revision_count,
        "revision_records": revision_records,
        "record_status": str(raw.get("status") or "CURRENT"),
        "total_wall_time_seconds": total_wall_time,
        "total_tokens": total_tokens,
        "benchmark_total_tokens": benchmark_total_tokens,
        "benchmark_token_counter_source": benchmark_source,
        "unallocated_backfill_tokens": unallocated_backfill_tokens or 0,
        "token_totals_by_source": token_totals_by_source,
        "mixed_token_counter_sources": len(totals_by_source) > 1,
        "completed_session_count": completed_count,
        "running_session_count": running_count,
        "pending_token_session_count": pending_count,
        "completed_sessions": completed_sessions,
        "running_sessions": running_sessions,
        "pending_token_sessions": pending_token_sessions,
        "source_path": source_path,
        "issues": [],
    }
