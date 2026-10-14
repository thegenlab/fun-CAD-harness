"""Observe one CLI call's reported token usage without reading other sessions.

The CLI emits its thread id early but reports usage only when the call ends.
Its exact registered rollout also records actual ``turn_token_usage`` after
each model response. Those records are snapshots for one task_started turn,
not differences of a persistent session counter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid

from ..mechanism.io import write_json_atomic


SCHEMA = "cold_start_live_token_usage_v1"
_CHUNK_BYTES = 256 * 1024
_LOOKBACK_BYTES = 512 * 1024
_MAX_LINE_BYTES = 64 * 1024
_RAW_KINDS = (b'"token_usage_record"', b'"task_started"', b'"task_complete"')
_CLI_KINDS = (b'"thread.started"', b'"turn.completed"', b'"type":"result"', b'"type": "result"')
_USAGE_KEYS = (
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "reasoning_output_tokens",
)
_USAGE_ALIASES = {
    "inputTokens": "input_tokens",
    "outputTokens": "output_tokens",
    "cacheWriteInputTokens": "cache_write_input_tokens",
    "cacheWriteTokens": "cache_write_input_tokens",
    "cachedInputTokens": "cached_input_tokens",
    "cacheReadTokens": "cached_input_tokens",
    "cache_read_input_tokens": "cached_input_tokens",
    "cache_creation_input_tokens": "cache_write_input_tokens",
    "reasoningOutputTokens": "reasoning_output_tokens",
    "totalTokens": "total_tokens",
}
_SEPARATE_CACHE_KEYS = ("cacheReadTokens", "cacheWriteTokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _epoch(value: Any) -> float | None:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if result.tzinfo is None:
            result = result.replace(tzinfo=timezone.utc)
        return result.timestamp()
    except (ValueError, TypeError, OverflowError):
        return None


def _usage(value: Any) -> dict[str, int] | None:
    """Accept Codex snake_case, cursor-agent print-mode camelCase and Claude Code.

    Cursor ``result.usage`` reports uncached ``inputTokens`` plus
    ``cacheReadTokens`` / ``cacheWriteTokens``; Claude Code reports uncached
    ``input_tokens`` plus ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``. Cache is billed input and is included in
    ``input_tokens``, matching Codex and the visible-session ledger.
    """
    if not isinstance(value, dict):
        return None
    separate_cache = any(key in value for key in _SEPARATE_CACHE_KEYS)
    result: dict[str, int] = {}
    for key, item in value.items():
        dest = _USAGE_ALIASES.get(key, key)
        if dest in _USAGE_KEYS and isinstance(item, int) and not isinstance(item, bool) and item >= 0:
            result[dest] = item
    if "input_tokens" not in result or "output_tokens" not in result:
        return None
    if separate_cache:
        result["input_tokens"] = (
            result["input_tokens"]
            + result.get("cached_input_tokens", 0)
            + result.get("cache_write_input_tokens", 0)
        )
    result["total_tokens"] = result["input_tokens"] + result["output_tokens"]
    return result


normalize_token_usage = _usage


class _JsonlCursor:
    """Read a bounded append chunk, dropping large non-statistics lines."""

    def __init__(self, path: Path, *, offset: int = 0, skip_partial: bool = False):
        self.path = path
        self.offset = offset
        self.tail = b""
        self.dropping = skip_partial

    def read(self, kinds: tuple[bytes, ...]) -> list[dict[str, Any]]:
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self.offset, self.tail, self.dropping = 0, b"", False
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                chunk = stream.read(_CHUNK_BYTES)
                self.offset = stream.tell()
        except OSError:
            return []
        lines = chunk.split(b"\n")
        records: list[dict[str, Any]] = []
        for index, part in enumerate(lines):
            ended = index < len(lines) - 1
            if self.dropping:
                if ended:
                    self.dropping = False
                continue
            line = self.tail + part
            self.tail = b""
            if len(line) > _MAX_LINE_BYTES:
                self.dropping = not ended
                continue
            if not ended:
                self.tail = line
                continue
            if not any(kind in line for kind in kinds):
                continue
            try:
                value = json.loads(line)
            except (UnicodeError, ValueError):
                continue
            if isinstance(value, dict):
                records.append(value)
        return records


class TokenWatch:
    """One bounded observer, owned and stopped by one CodexCliWorker call."""

    def __init__(
        self,
        output_dir: Path,
        *,
        session_id: str | None = None,
        codex_home: Path | None = None,
        started_at_epoch: float | None = None,
        interval: float = 1.25,
    ):
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / "usage.live.json"
        self.codex_home = Path(codex_home) if codex_home is not None else Path(
            os.environ.get("CODEX_HOME") or Path.home() / ".codex"
        )
        self.started_at = time.time() if started_at_epoch is None else started_at_epoch
        self.expected_session_id = session_id
        self.session_id: str | None = None
        self.turn_id: str | None = None
        self.cli = _JsonlCursor(self.output_dir / "events.jsonl")
        self.raw: _JsonlCursor | None = None
        self.prepared: tuple[Path, int] | None = None
        self.last_lookup = float("-inf")
        self.cli_usage: dict[str, int] | None = None
        self.interval = max(1.0, float(interval))
        self.closed = threading.Event()
        self.thread: threading.Thread | None = None
        self._mono_started = time.monotonic()
        self.arrivals: list[dict[str, Any]] = []
        self.value: dict[str, Any] = {
            "schema_version": SCHEMA, "call_id": uuid.uuid4().hex,
            "call_started_at": _iso(self.started_at), "thread_id": None, "turn_id": None,
            "status": "pending", "pending": True, "source": "pending",
            "usage": None, "updated_at": _iso(self.started_at),
        }
        # Capture the old end before launching a resumed call; never scan the
        # persistent session's earlier conversations to derive a counter delta.
        if session_id:
            path = self._rollout_path(session_id)
            if path is not None:
                try:
                    self.prepared = path, path.stat().st_size
                except OSError:
                    pass
        self._publish()

    def _rollout_path(self, session_id: str) -> Path | None:
        database = self.codex_home / "state_5.sqlite"
        try:
            with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1)) as connection:
                row = connection.execute(
                    "SELECT rollout_path FROM threads WHERE id = ?", (session_id,)
                ).fetchone()
            return Path(row[0]) if row and isinstance(row[0], str) and row[0] else None
        except (OSError, ValueError, sqlite3.Error):
            return None

    def _publish(self) -> None:
        try:
            write_json_atomic(self.path, self.value)
        except OSError:
            # Dashboard observability must not fail an author or Harness turn.
            pass

    def _consume_raw(self, records: list[dict[str, Any]]) -> bool:
        changed = False
        for record in records:
            timestamp = _epoch(record.get("timestamp"))
            if timestamp is None or timestamp < self.started_at:
                continue
            payload = record.get("payload")
            if not isinstance(payload, dict):
                continue
            if record.get("type") == "event_msg" and payload.get("type") == "task_started":
                turn_id = payload.get("turn_id")
                if self.turn_id is None and isinstance(turn_id, str) and turn_id:
                    self.turn_id = turn_id
                    self.value.update(turn_id=turn_id)
                    self.arrivals.append({
                        "kind": "task_started",
                        "event_timestamp": record.get("timestamp"),
                        "local_monotonic_arrival": time.monotonic() - self._mono_started,
                    })
                    changed = True
            elif record.get("type") == "token_usage_record" and self.turn_id and payload.get("turn_id") == self.turn_id:
                usage = _usage(payload.get("turn_token_usage"))
                if usage is not None and usage != self.value.get("usage"):
                    self.value.update(usage=usage, status="running", pending=True,
                                      source="codex_raw_turn_token_usage", updated_at=record["timestamp"])
                    self.arrivals.append({
                        "kind": "token_usage_record",
                        "event_timestamp": record.get("timestamp"),
                        "local_monotonic_arrival": time.monotonic() - self._mono_started,
                    })
                    changed = True
        return changed

    def _attach_raw(self) -> bool:
        if self.raw is not None or not self.session_id:
            return False
        now = time.monotonic()
        if now - self.last_lookup < self.interval:
            return False
        self.last_lookup = now
        path = self._rollout_path(self.session_id)
        if path is None:
            return False
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if self.prepared is not None and self.prepared[0] == path and self.session_id == self.expected_session_id:
            self.raw = _JsonlCursor(path, offset=min(self.prepared[1], size))
            return False
        # A new thread has its task_started near the beginning. Inspect only
        # one bounded prefix, then jump over large prompt/tool records to the
        # recent tail. A resumed thread with no readable pre-launch DB snapshot
        # uses only the tail; if its start cannot be established, stay pending.
        changed = False
        if not self.expected_session_id and size > _LOOKBACK_BYTES:
            prefix = _JsonlCursor(path)
            changed = self._consume_raw(prefix.read(_RAW_KINDS))
        offset = max(0, size - _LOOKBACK_BYTES)
        self.raw = _JsonlCursor(path, offset=offset, skip_partial=offset > 0)
        return changed

    def poll(self) -> None:
        """One small read cycle; also directly usable by deterministic tests."""
        changed = False
        for event in self.cli.read(_CLI_KINDS):
            if event.get("type") == "thread.started" and not self.session_id:
                session = event.get("thread_id")
                if isinstance(session, str) and session:
                    self.session_id = session
                    self.value["thread_id"] = session
                    changed = True
            elif event.get("type") in {"turn.completed", "result"}:
                usage = _usage(event.get("usage"))
                if usage is not None:
                    source = (
                        "codex_cli_turn_completed" if event.get("type") == "turn.completed"
                        else "cursor_agent_result_usage"
                    )
                    self.cli_usage = usage
                    self.value.update(usage=usage, status="completed", pending=False,
                                      source=source, updated_at=_iso(time.time()))
                    self.arrivals.append({
                        "kind": event.get("type"),
                        "event_timestamp": event.get("timestamp"),
                        "local_monotonic_arrival": time.monotonic() - self._mono_started,
                    })
                    changed = True
        if self.cli_usage is None:
            changed = self._attach_raw() or changed
            if self.raw is not None:
                if self.turn_id:
                    try:
                        size = self.raw.path.stat().st_size
                        if size - self.raw.offset > _LOOKBACK_BYTES:
                            self.raw = _JsonlCursor(self.raw.path, offset=size - _LOOKBACK_BYTES, skip_partial=True)
                    except OSError:
                        pass
                changed = self._consume_raw(self.raw.read(_RAW_KINDS)) or changed
        if changed:
            self._publish()

    def _loop(self) -> None:
        while not self.closed.is_set():
            try:
                self.poll()
            except (OSError, ValueError, sqlite3.Error):
                pass
            if self.closed.wait(self.interval):
                return

    def start(self) -> None:
        if self.thread is not None or self.closed.is_set():
            return
        self.thread = threading.Thread(target=self._loop, name="codex-turn-token-watch", daemon=True)
        try:
            self.thread.start()
        except RuntimeError:
            self.thread = None

    def stop(self) -> None:
        self.closed.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def finish(self, *, usage: dict[str, Any] | None = None, completed: bool = False,
               source: str | None = None) -> None:
        self.stop()
        try:
            self.poll()
        except (OSError, ValueError, sqlite3.Error):
            pass
        final = _usage(usage) or self.cli_usage
        if final is not None and (completed or self.cli_usage is not None):
            self.value.update(usage=final, status="completed", pending=False,
                              source=source or "codex_cli_turn_completed", updated_at=_iso(time.time()))
        else:
            self.value.update(status="stopped", pending=True)
        self._publish()
