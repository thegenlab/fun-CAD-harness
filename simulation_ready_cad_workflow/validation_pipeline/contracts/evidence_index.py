"""Versioned Evidence Index published by the Harness with every result.

The index is the only sanctioned way for the Loop, the Loop Loop, the
dashboard, and benchmark tooling to locate Harness-produced evidence.
Consumers must never derive meaning from the Harness run-directory
layout, private error-code sets, or natural-language error text; they
read entries here instead.

Every entry records at least: role, producer, the referenced document's
schema version (when declared), a workspace-relative path, content hash,
execution status (``EXECUTED`` / ``REUSED`` / ``INVALIDATED``), the
related test/requirement identifiers, and provenance.  ``worker_visible``
marks the bounded subset the Loop may materialize into the worker's
failure-review evidence catalog.

Missing, corrupt, oversized, or version-incompatible indexes fail
closed with :class:`~.harness_result.ContractViolation`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..mechanism.io import (
    read_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .harness_result import ContractViolation


EVIDENCE_INDEX_SCHEMA_ID = "simulation_ready_cad_evidence_index_v1"
EVIDENCE_INDEX_FILENAME = "evidence_index.json"
MAX_EVIDENCE_INDEX_BYTES = 4 * 1024 * 1024
MAX_ENTRIES = 512

EVIDENCE_STATUSES = frozenset({"EXECUTED", "REUSED", "INVALIDATED"})

_ENTRY_REQUIRED = {
    "evidence_id",
    "role",
    "title",
    "producer",
    "path",
    "sha256",
    "size_bytes",
    "status",
    "worker_visible",
}
_ENTRY_OPTIONAL = {
    "document_schema_version",
    "source_revision",
    "related",
    "lineage",
    "provenance",
}
_HEX = set("0123456789abcdef")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractViolation("EVIDENCE_INDEX_INVALID", message)


def validate_evidence_entry(entry: Mapping[str, Any], *, index: int) -> None:
    label = f"entries[{index}]"
    _require(isinstance(entry, Mapping), f"{label} must be an object")
    keys = set(entry)
    _require(
        _ENTRY_REQUIRED.issubset(keys) and keys.issubset(_ENTRY_REQUIRED | _ENTRY_OPTIONAL),
        f"{label} has missing or unknown fields: "
        f"missing={sorted(_ENTRY_REQUIRED - keys)} unknown={sorted(keys - _ENTRY_REQUIRED - _ENTRY_OPTIONAL)}",
    )
    for field in ("evidence_id", "role", "title", "producer", "path"):
        value = entry[field]
        _require(
            isinstance(value, str) and 0 < len(value) <= 1024,
            f"{label}.{field} must be a bounded non-empty string",
        )
    digest = entry["sha256"]
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in _HEX for character in digest),
        f"{label}.sha256 must be a SHA-256 hex digest",
    )
    size = entry["size_bytes"]
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label}.size_bytes must be a non-negative integer",
    )
    _require(
        entry["status"] in EVIDENCE_STATUSES,
        f"{label}.status must be one of {sorted(EVIDENCE_STATUSES)}",
    )
    _require(
        isinstance(entry["worker_visible"], bool),
        f"{label}.worker_visible must be a boolean",
    )
    related = entry.get("related")
    _require(
        related is None
        or (
            isinstance(related, Mapping)
            and all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in related.items()
            )
        ),
        f"{label}.related must map string keys to string values",
    )
    lineage = entry.get("lineage")
    _require(
        lineage is None
        or (
            isinstance(lineage, list)
            and len(lineage) <= 64
            and all(isinstance(item, str) for item in lineage)
        ),
        f"{label}.lineage must be a bounded list of strings",
    )


def validate_evidence_index(document: Mapping[str, Any]) -> None:
    _require(isinstance(document, Mapping), "index must be a JSON object")
    _require(
        document.get("schema_version") == EVIDENCE_INDEX_SCHEMA_ID,
        "index schema_version is not "
        f"{EVIDENCE_INDEX_SCHEMA_ID!r}: {document.get('schema_version')!r}",
    )
    _require(
        isinstance(document.get("producer"), str) and document["producer"],
        "index producer must be a non-empty string",
    )
    entries = document.get("entries")
    _require(
        isinstance(entries, list) and len(entries) <= MAX_ENTRIES,
        f"index entries must be a list of at most {MAX_ENTRIES} items",
    )
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        validate_evidence_entry(entry, index=index)
        evidence_id = entry["evidence_id"]
        _require(
            evidence_id not in seen,
            f"duplicate evidence_id {evidence_id!r}",
        )
        seen.add(evidence_id)


def write_evidence_index(
    *,
    workspace_root: Path,
    run_dir: Path,
    producer: str,
    entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Atomically publish the index and return its hash-bound reference."""

    document = {
        "schema_version": EVIDENCE_INDEX_SCHEMA_ID,
        "producer": producer,
        "run": relative_to_root(workspace_root, run_dir),
        "entries": [dict(entry) for entry in entries],
    }
    validate_evidence_index(document)
    destination = run_dir / EVIDENCE_INDEX_FILENAME
    write_json_atomic(destination, document)
    return {
        "schema_version": EVIDENCE_INDEX_SCHEMA_ID,
        "path": relative_to_root(workspace_root, destination),
        "sha256": sha256_file(destination),
        "entry_count": len(document["entries"]),
    }


def load_evidence_index(
    *,
    workspace_root: Path,
    reference: Mapping[str, Any],
) -> dict[str, Any]:
    """Load an index through its hash-bound reference; fail closed."""

    _require(isinstance(reference, Mapping), "evidence index reference must be an object")
    _require(
        reference.get("schema_version") == EVIDENCE_INDEX_SCHEMA_ID,
        "evidence index reference schema_version is incompatible: "
        f"{reference.get('schema_version')!r}",
    )
    relative = reference.get("path")
    digest = reference.get("sha256")
    _require(
        isinstance(relative, str) and relative and isinstance(digest, str),
        "evidence index reference has no path or sha256",
    )
    workspace = workspace_root.resolve()
    path = (workspace / relative).resolve()
    try:
        path.relative_to(workspace)
    except ValueError:
        raise ContractViolation(
            "EVIDENCE_INDEX_INVALID",
            "evidence index reference escapes the workspace",
        ) from None
    _require(
        not path.is_symlink() and path.is_file(),
        f"evidence index file is missing: {relative}",
    )
    _require(
        sha256_file(path) == digest,
        "evidence index content hash does not match its reference",
    )
    try:
        document = read_json(path, max_bytes=MAX_EVIDENCE_INDEX_BYTES)
    except (OSError, ValueError) as exc:
        raise ContractViolation(
            "EVIDENCE_INDEX_INVALID",
            f"evidence index could not be read: {exc}",
        ) from exc
    validate_evidence_index(document)
    entry_count = reference.get("entry_count")
    _require(
        entry_count is None or entry_count == len(document["entries"]),
        "evidence index entry count does not match its reference",
    )
    return document
