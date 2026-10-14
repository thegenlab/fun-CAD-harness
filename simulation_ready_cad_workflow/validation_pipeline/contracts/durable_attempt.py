"""Durable attempt documents for Harness and Queue ownership."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..mechanism.io import read_json, write_json_atomic


HARNESS_ATTEMPT_SCHEMA = "harness_attempt_state.v1"
QUEUE_ATTEMPT_SCHEMA = "queue_attempt_state.v1"
ATTEMPT_STATUSES = frozenset(
    {
        "PENDING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
        "RESOURCE_BLOCKED",
        "ORPHANED",
        "OWNED_ELSEWHERE",
        "INTERRUPTED",
    }
)


class AttemptStateError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def new_attempt_id() -> str:
    return uuid.uuid4().hex


def new_owner_token() -> str:
    return uuid.uuid4().hex


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def harness_attempt_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "harness_attempt_state.json"


def queue_attempt_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "queue_attempt.json"


def new_attempt_document(
    *,
    schema_version: str,
    input_digest: str,
    stage: str,
    process_identity: Mapping[str, Any],
    deadline_at: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    now = utc_now()
    document = {
        "schema_version": schema_version,
        "attempt_id": new_attempt_id(),
        "owner_token": new_owner_token(),
        "input_digest": input_digest,
        "stage": stage,
        "status": "RUNNING",
        "pid": process_identity.get("pid"),
        "process_create_time": process_identity.get("create_time"),
        "lease_expires_at": None,
        "heartbeat_at": now,
        "deadline_at": deadline_at,
        "started_at": now,
        "finished_at": None,
        "terminal_receipt": None,
    }
    if extra:
        document.update(dict(extra))
    return document


def write_attempt_document(path: str | Path, document: Mapping[str, Any]) -> None:
    write_json_atomic(path, dict(document))


def read_attempt_document(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        return None
    try:
        payload = read_json(candidate)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def heartbeat_attempt(document: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    updated = dict(document)
    updated["stage"] = stage
    updated["heartbeat_at"] = utc_now()
    updated["status"] = "RUNNING"
    return updated


def finish_attempt(
    document: Mapping[str, Any],
    *,
    status: str,
    terminal_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in ATTEMPT_STATUSES:
        raise AttemptStateError(
            "ATTEMPT_STATUS_INVALID",
            f"unsupported attempt status {status!r}",
        )
    updated = dict(document)
    updated["status"] = status
    updated["finished_at"] = utc_now()
    updated["heartbeat_at"] = updated["finished_at"]
    if terminal_receipt is not None:
        updated["terminal_receipt"] = dict(terminal_receipt)
    return updated
