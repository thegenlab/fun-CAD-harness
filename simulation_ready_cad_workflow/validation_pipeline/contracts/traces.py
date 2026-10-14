"""The three sanctioned trace families of the validation pipeline.

* **Worker action trace** (``worker_action_trace_v1``) — what the Worker
  Loop did.  The Loop already journals every state transition as a
  hash-chained event stream (``react_loop/events/e*.json``); this module
  provides the versioned, chain-verifying read view.  Nothing else may
  reinterpret those files.
* **Harness execution trace** (``harness_execution_trace_v1``) — what the
  Harness executed for one run.  Every ``publish_run_status`` call appends
  a hash-chained entry to ``<run>/execution_trace.json``; the trace is the
  auditable stage history of the run.
* **Maintenance trace** (``maintenance_trace_v1``) — what the Loop Loop
  (operator/reviewer) changed and why.  Every intervention appends a
  hash-chained entry to ``<project>/maintenance/maintenance_trace.json``
  and must state explicitly what, if anything, became visible to the
  Worker (information-isolation audit field).

All three fail closed on hash-chain damage.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..mechanism.io import canonical_sha256, read_json, write_json_atomic

WORKER_ACTION_TRACE_SCHEMA = "worker_action_trace_v1"
HARNESS_EXECUTION_TRACE_SCHEMA = "harness_execution_trace_v1"
MAINTENANCE_TRACE_SCHEMA = "maintenance_trace_v1"

_LOOP_EVENT_SCHEMAS = frozenset(
    {"simulation_ready_cad_react_loop_event_v2"}
)

_MAINTENANCE_KINDS = frozenset(
    {
        "harness_defect_fix",
        "resource_repair",
        "template_publication",
        "regression_run",
        "controlled_communication",
        "loop_restart",
        "proposal_review",
        "proposal_decision",
        "proposal_landing",
        "other",
    }
)


class TraceContractError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Harness execution trace
# ---------------------------------------------------------------------------


def _load_chain(path: Path, schema: str) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": schema, "entries": []}
    try:
        document = read_json(path)
    except (OSError, ValueError) as error:
        raise TraceContractError(
            "TRACE_UNREADABLE", f"{path}: {error}"
        ) from error
    if document.get("schema_version") != schema:
        raise TraceContractError(
            "TRACE_SCHEMA_MISMATCH",
            f"{path}: expected {schema}, found "
            f"{document.get('schema_version')!r}",
        )
    _verify_chain(document.get("entries"), path=path)
    return document


def _verify_chain(entries: Any, *, path: Path) -> None:
    if not isinstance(entries, list):
        raise TraceContractError(
            "TRACE_INVALID", f"{path}: entries must be a list"
        )
    previous: str | None = None
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise TraceContractError(
                "TRACE_INVALID", f"{path}: entry {position} is not an object"
            )
        core = {
            key: value
            for key, value in entry.items()
            if key != "entry_sha256"
        }
        if entry.get("entry_sha256") != canonical_sha256(core):
            raise TraceContractError(
                "TRACE_CHAIN_BROKEN",
                f"{path}: entry {position} digest mismatch",
            )
        if entry.get("sequence") != position:
            raise TraceContractError(
                "TRACE_CHAIN_BROKEN",
                f"{path}: entry {position} has sequence "
                f"{entry.get('sequence')!r}",
            )
        if entry.get("previous_entry_sha256") != previous:
            raise TraceContractError(
                "TRACE_CHAIN_BROKEN",
                f"{path}: entry {position} does not link to its predecessor",
            )
        previous = str(entry["entry_sha256"])


def _append_chained_entry(
    path: Path, schema: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    document = _load_chain(path, schema)
    entries: list[dict[str, Any]] = document["entries"]
    previous = entries[-1]["entry_sha256"] if entries else None
    core = {
        "sequence": len(entries) + 1,
        "previous_entry_sha256": previous,
        "created_at": _utc_now(),
        **payload,
    }
    entry = {**core, "entry_sha256": canonical_sha256(core)}
    entries.append(entry)
    write_json_atomic(path, document)
    return entry


def execution_trace_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "execution_trace.json"


def append_execution_trace(
    run_dir: str | Path,
    *,
    status: str,
    last_completed_stage: str | None,
    detail: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _append_chained_entry(
        execution_trace_path(run_dir),
        HARNESS_EXECUTION_TRACE_SCHEMA,
        {
            "status": str(status),
            "last_completed_stage": last_completed_stage,
            "detail": dict(detail or {}),
        },
    )


def load_execution_trace(run_dir: str | Path) -> dict[str, Any]:
    path = execution_trace_path(run_dir)
    if not path.is_file():
        raise TraceContractError(
            "TRACE_MISSING", f"no execution trace at {path}"
        )
    return _load_chain(path, HARNESS_EXECUTION_TRACE_SCHEMA)


# ---------------------------------------------------------------------------
# Maintenance trace (Loop Loop interventions)
# ---------------------------------------------------------------------------


def maintenance_trace_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "maintenance" / "maintenance_trace.json"


def append_maintenance_record(
    project_dir: str | Path,
    *,
    actor: str,
    intervention_kind: str,
    reason: str,
    worker_visible_change: str = "none",
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record one Loop Loop intervention.

    ``worker_visible_change`` is the information-isolation audit field: it
    must describe exactly what the Worker can now observe that it could
    not before (``"none"`` for purely internal repairs).  Refusing to
    default this silently is deliberate.
    """

    if intervention_kind not in _MAINTENANCE_KINDS:
        raise TraceContractError(
            "MAINTENANCE_KIND_UNKNOWN",
            f"{intervention_kind!r} is not one of "
            f"{sorted(_MAINTENANCE_KINDS)}",
        )
    if not str(reason).strip():
        raise TraceContractError(
            "MAINTENANCE_REASON_REQUIRED",
            "a maintenance record must explain why it happened",
        )
    if not str(worker_visible_change).strip():
        raise TraceContractError(
            "MAINTENANCE_VISIBILITY_REQUIRED",
            "state what the worker can now see, or 'none'",
        )
    path = maintenance_trace_path(project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _append_chained_entry(
        path,
        MAINTENANCE_TRACE_SCHEMA,
        {
            "actor": str(actor),
            "intervention_kind": intervention_kind,
            "reason": str(reason),
            "worker_visible_change": str(worker_visible_change),
            "evidence": dict(evidence or {}),
        },
    )


def load_maintenance_trace(project_dir: str | Path) -> dict[str, Any]:
    path = maintenance_trace_path(project_dir)
    if not path.is_file():
        return {"schema_version": MAINTENANCE_TRACE_SCHEMA, "entries": []}
    return _load_chain(path, MAINTENANCE_TRACE_SCHEMA)


# ---------------------------------------------------------------------------
# Worker action trace (chain-verifying view over the Loop event journal)
# ---------------------------------------------------------------------------


def load_worker_action_trace(project_dir: str | Path) -> dict[str, Any]:
    """Assemble the versioned worker-action view from the Loop journal.

    Verifies the existing ``event_sha256``/``previous_event_sha256`` chain
    written by the Loop and fails closed on any damage.  The returned
    document is read-only derived data; the journal stays authoritative.
    """

    events_dir = Path(project_dir) / "react_loop" / "events"
    if not events_dir.is_dir():
        raise TraceContractError(
            "TRACE_MISSING", f"no loop event journal under {events_dir}"
        )
    events: list[dict[str, Any]] = []
    previous: str | None = None
    for position, path in enumerate(
        sorted(events_dir.glob("e*.json")), start=1
    ):
        try:
            event = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise TraceContractError(
                "TRACE_UNREADABLE", f"{path}: {error}"
            ) from error
        if event.get("schema_version") not in _LOOP_EVENT_SCHEMAS:
            raise TraceContractError(
                "TRACE_SCHEMA_MISMATCH",
                f"{path}: unknown loop event schema "
                f"{event.get('schema_version')!r}",
            )
        core = {
            key: value
            for key, value in event.items()
            if key != "event_sha256"
        }
        if event.get("event_sha256") != canonical_sha256(core):
            raise TraceContractError(
                "TRACE_CHAIN_BROKEN", f"{path}: event digest mismatch"
            )
        if int(event.get("sequence") or 0) != position:
            raise TraceContractError(
                "TRACE_CHAIN_BROKEN",
                f"{path}: event sequence {event.get('sequence')!r} at "
                f"position {position}",
            )
        if position > 1 and event.get("previous_event_sha256") != previous:
            raise TraceContractError(
                "TRACE_CHAIN_BROKEN",
                f"{path}: event does not link to its predecessor",
            )
        previous = str(event["event_sha256"])
        events.append(
            {
                "sequence": event["sequence"],
                "event_type": event.get("event_type"),
                "revision": event.get("revision"),
                "loop_status": event.get("loop_status"),
                "created_at": event.get("created_at"),
                "event_sha256": event["event_sha256"],
            }
        )
    return {
        "schema_version": WORKER_ACTION_TRACE_SCHEMA,
        "project_dir": str(project_dir),
        "event_count": len(events),
        "head_sha256": previous,
        "events": events,
    }
