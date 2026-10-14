"""Bounded, read-only projection of public author events; never load prompts."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re

from .live_workflow import _context, _read_json, _turn_directories

from ..trajectory.events import (PAGE_BYTES, PAGE_ITEMS, TEXT_LIMIT, display_text,
    event_rows as _event_rows, event_ms as _event_ms, merge_rows as _merge_rows)


def _page(path, before=None):
    try:
        size = path.stat().st_size
        end = size if before is None else min(size, max(0, int(before)))
        with path.open("rb") as stream:
            scan_start = max(0, end - PAGE_BYTES)
            stream.seek(scan_start)
            if stream.tell():
                stream.readline(PAGE_BYTES)  # discard only the leading partial record
            start = min(end, stream.tell())
            raw = stream.read(max(0, end - start))
    except FileNotFoundError:
        return [], None, False
    records, offset = [], start
    fragmented = start > scan_start and start - scan_start >= PAGE_BYTES // 2
    for line in raw.splitlines(keepends=True):
        try:
            event = json.loads(line)
        except (UnicodeError, ValueError):
            offset += len(line)
            continue  # incomplete live tail is retried on the next refresh
        for row in _event_rows(event, offset):
            records.append((offset, row))
        offset += len(line)
    if fragmented:
        records.insert(0, (scan_start, {"id":"fragment-"+str(scan_start), "kind":"status", "title":"长记录未展开",
            "text":"一条较长的原始事件超出本页读取范围，内容未解析。原始日志保留；可向前翻阅其他记录。", "at":""}))
    records = records[-PAGE_ITEMS:]
    next_before = records[0][0] if records else scan_start
    items = _merge_rows(records)
    return items, next_before if next_before > 0 else None, True


def trajectory_view(*, workspace_root, run_dir, turn=None, before=None, from_ms=None, to_ms=None, snapshot=None, after=None):
    identity, state = _context(workspace_root, run_dir)
    if identity is None:
        raise ValueError("This run has no recorded author workflow")
    directories = _turn_directories(identity.project_dir)
    choices = [{"id":p.name,"label":p.name.replace("_", " · ", 1)} for p in directories]
    selected = next((p for p in directories if p.name == turn), None) if turn else (directories[-1] if directories else None)
    if turn and selected is None:
        raise ValueError("Unknown author turn in this project")
    windowed = from_ms is not None or to_ms is not None
    from ..trajectory.store import HistoryStore
    store = HistoryStore(identity.project_dir)
    indexed = None
    sequence_cursor = isinstance(before, str) and before.startswith("seq:")
    if store.path.is_file() and (before is None or sequence_cursor or snapshot or after is not None or windowed):
        indexed = store.event_page(turn=turn or (selected.name if selected and not snapshot else None),
            before=int(before[4:]) if sequence_cursor else None, after=after, from_ms=from_ms, to_ms=to_ms, snapshot=snapshot)
        items, cursor, recorded = indexed["items"], indexed["next_before"], indexed["events_recorded"]
        selected = next((p for p in directories if p.name == indexed["turn"]), None)
    elif windowed or snapshot:
        items, cursor, recorded = [], None, False
        indexed = {"indexing":True,"message":"正在建立历史索引，暂不能按时段定位。"}
    else:
        items, cursor, recorded = _page(selected / "events.jsonl", before) if selected else ([], None, False)
    response = _read_json(selected / "response.json") if selected else {}
    result = _read_json(identity.run_dir / "result.json")
    feedback = {key:display_text(result[key]) for key in ("status","issues","pending_tests") if key in result and result[key] not in (None, "", [], {})}
    payload = {"scope":"project_author_trajectory", "project_id":identity.project_id,
            "selected_revision":identity.run_dir.name, "turns":choices, "turn":selected.name if selected else "",
            "items":items, "next_before":cursor, "events_recorded":recorded,
            "task":display_text(state["request"].get("brief", "")),
            "response":{key:display_text(response[key]) for key in ("status","summary","issues") if key in response and response[key] not in (None, "", [], {})},
            "feedback":feedback, "read_only":True}
    if indexed:
        payload.update(indexed)
        if indexed.get("cursor_kind") == "sequence" and payload.get("next_before") is not None:
            payload["next_before"] = "seq:" + str(payload["next_before"])
    if windowed:
        payload["time_filter"] = {"from_ms": from_ms, "to_ms": to_ms}
    return payload
