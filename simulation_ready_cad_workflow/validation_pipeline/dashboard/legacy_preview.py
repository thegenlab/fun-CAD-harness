"""Display-only compatibility for the exact pre-publication run cohort."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .stage_manifest import (
    read_stable_json_object,
    stable_regular_file_record,
)


LEGACY_PREVIEW_ALLOWLIST_NAME = (
    "legacy_dashboard_revision_preview_allowlist.json"
)
LEGACY_PREVIEW_ALLOWLIST_SCHEMA = (
    "simulation_ready_cad_legacy_dashboard_display_cohort_v2"
)
LEGACY_PREVIEW_ALLOWLIST_POLICY = (
    "display_only_exact_pre_v2_run_cohort_no_verdict_authority_v1"
)
LEGACY_PREVIEW_EXPECTED_ENTRY_COUNT = 356
LEGACY_PREVIEW_EXPECTED_MANIFEST_PRESENT_COUNT = 330
LEGACY_PREVIEW_EXPECTED_INVENTORY_PRESENT_COUNT = 326
_DOCUMENT_FIELDS = frozenset(
    {
        "schema_version",
        "policy",
        "entry_count",
        "entries",
        "entries_canonical_sha256",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "run_id",
        "revision_preview_path",
        "run_manifest",
        "run_status",
        "artifact_inventory",
        "revision_preview",
    }
)
_PRESENT_BINDING_FIELDS = frozenset(
    {"state", "path", "size_bytes", "sha256"}
)
_MISSING_BINDING_FIELDS = frozenset({"state", "path"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^r(?:0*[1-9][0-9]*)$")
_TOP_LEVEL_LEGACY_RUN = re.compile(
    r"^\.runtime_runs/ubuntu-[a-z0-9-]+-20260723-r(?:0*[1-9][0-9]*)$"
)
_PREVIEW_SUFFIXES = (
    "design_quality/assets/01_iso.png",
    "visual/stage1_design_package.png",
    "visual/cad_iso.png",
)
_DISPLAY_MEDIA_SUFFIXES = frozenset({".png", ".mp4", ".glb"})


class LegacyPreviewError(ValueError):
    """The display-only legacy compatibility boundary is invalid."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _normalized_workspace_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or candidate.drive
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        return None
    return value


def _canonical_run_id(value: Any) -> str | None:
    normalized = _normalized_workspace_path(value)
    if normalized is None:
        return None
    parts = Path(normalized).parts
    if (
        len(parts) == 2
        and _TOP_LEVEL_LEGACY_RUN.fullmatch(normalized)
    ):
        return normalized
    if (
        len(parts) != 5
        or parts[0] != ".runtime_runs"
        or parts[1] != "projects"
        or parts[3] != "runs"
        or not parts[2]
        or not _REVISION.fullmatch(parts[4])
    ):
        return None
    return normalized


def _validate_binding(
    value: Any,
    *,
    expected_path: str,
    entry_index: int,
    role: str,
    allow_missing: bool = False,
) -> dict[str, Any]:
    if (
        allow_missing
        and isinstance(value, dict)
        and set(value) == _MISSING_BINDING_FIELDS
        and value.get("state") == "MISSING"
        and value.get("path") == expected_path
    ):
        return dict(value)
    if (
        not isinstance(value, dict)
        or set(value) != _PRESENT_BINDING_FIELDS
        or value.get("state") != "PRESENT"
        or value.get("path") != expected_path
        or not isinstance(value.get("size_bytes"), int)
        or isinstance(value.get("size_bytes"), bool)
        or int(value["size_bytes"]) <= 0
        or not isinstance(value.get("sha256"), str)
        or not _SHA256.fullmatch(str(value["sha256"]))
    ):
        raise LegacyPreviewError(
            f"legacy preview allowlist entry {entry_index} has an invalid "
            f"{role} binding"
        )
    return dict(value)


def validate_legacy_revision_preview_allowlist(
    document: Any,
    *,
    expected_entry_count: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate the closed, sorted legacy-run cohort document."""

    if (
        not isinstance(document, dict)
        or set(document) != _DOCUMENT_FIELDS
        or document.get("schema_version")
        != LEGACY_PREVIEW_ALLOWLIST_SCHEMA
        or document.get("policy") != LEGACY_PREVIEW_ALLOWLIST_POLICY
        or not isinstance(document.get("entry_count"), int)
        or isinstance(document.get("entry_count"), bool)
        or not isinstance(document.get("entries"), list)
        or document.get("entry_count") != len(document["entries"])
        or (
            expected_entry_count is not None
            and document.get("entry_count") != expected_entry_count
        )
        or not isinstance(document.get("entries_canonical_sha256"), str)
        or document.get("entries_canonical_sha256")
        != _canonical_sha256(document["entries"])
    ):
        raise LegacyPreviewError(
            "legacy revision preview allowlist has an invalid closed schema"
        )

    validated: list[dict[str, Any]] = []
    identities: list[tuple[str, str]] = []
    for index, raw in enumerate(document["entries"]):
        if not isinstance(raw, dict) or set(raw) != _ENTRY_FIELDS:
            raise LegacyPreviewError(
                f"legacy preview allowlist entry {index} has an invalid shape"
            )
        run_id = _canonical_run_id(raw.get("run_id"))
        preview_path = _normalized_workspace_path(
            raw.get("revision_preview_path")
        )
        expected_preview_paths = (
            {
                f"{run_id}/{suffix}"
                for suffix in _PREVIEW_SUFFIXES
            }
            if run_id is not None
            else set()
        )
        if (
            run_id is None
            or preview_path not in expected_preview_paths
        ):
            raise LegacyPreviewError(
                f"legacy preview allowlist entry {index} has an invalid identity"
            )
        entry = {
            "run_id": run_id,
            "revision_preview_path": preview_path,
            "run_manifest": _validate_binding(
                raw.get("run_manifest"),
                expected_path=f"{run_id}/run_manifest.json",
                entry_index=index,
                role="run manifest",
                allow_missing=True,
            ),
            "run_status": _validate_binding(
                raw.get("run_status"),
                expected_path=f"{run_id}/run_status.json",
                entry_index=index,
                role="run status",
            ),
            "artifact_inventory": _validate_binding(
                raw.get("artifact_inventory"),
                expected_path=f"{run_id}/artifacts.sha256.json",
                entry_index=index,
                role="artifact inventory",
                allow_missing=True,
            ),
            "revision_preview": _validate_binding(
                raw.get("revision_preview"),
                expected_path=str(preview_path),
                entry_index=index,
                role="revision preview",
            ),
        }
        identity = (run_id, str(preview_path))
        identities.append(identity)
        validated.append(entry)
    if (
        identities != sorted(identities)
        or len(set(identities)) != len(identities)
        or len({run_id for run_id, _path in identities}) != len(identities)
        or len({_path for _run_id, _path in identities}) != len(identities)
    ):
        raise LegacyPreviewError(
            "legacy revision preview allowlist entries are not uniquely sorted"
        )
    if expected_entry_count == LEGACY_PREVIEW_EXPECTED_ENTRY_COUNT and (
        sum(
            entry["run_manifest"]["state"] == "PRESENT"
            for entry in validated
        )
        != LEGACY_PREVIEW_EXPECTED_MANIFEST_PRESENT_COUNT
        or sum(
            entry["artifact_inventory"]["state"] == "PRESENT"
            for entry in validated
        )
        != LEGACY_PREVIEW_EXPECTED_INVENTORY_PRESENT_COUNT
    ):
        raise LegacyPreviewError(
            "production legacy preview cohort has invalid presence counts"
        )
    return tuple(validated)


@lru_cache(maxsize=1)
def legacy_dashboard_display_allowlist(
) -> tuple[dict[str, Any], ...]:
    """Load the exact legacy run cohort without verdict authority.

    Historical media is a local, read-only viewer concern. It does not wait
    for validator-release promotion and cannot authorize a terminal status,
    publication, certificate, or new run identity.
    """

    pipeline_root = Path(__file__).resolve().parent.parent
    allowlist_path = pipeline_root / LEGACY_PREVIEW_ALLOWLIST_NAME
    if not allowlist_path.exists():
        # Distributions without the private legacy cohort show no legacy media.
        return ()
    document, _record = read_stable_json_object(
        allowlist_path,
        code="DASHBOARD_LEGACY_DISPLAY_ALLOWLIST_INVALID",
    )
    return validate_legacy_revision_preview_allowlist(
        document,
        expected_entry_count=LEGACY_PREVIEW_EXPECTED_ENTRY_COUNT,
    )


def read_legacy_dashboard_media(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    run_id: str,
    relative_path: str,
) -> tuple[Path, bytes]:
    """Read stable media from one exact legacy run, for display only."""

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    matching_runs = [
        entry
        for entry in legacy_dashboard_display_allowlist()
        if entry["run_id"] == run_id
    ]
    if len(matching_runs) != 1:
        raise LegacyPreviewError(
            "requested run is not in the exact legacy display cohort"
        )
    normalized = _normalized_workspace_path(relative_path)
    if (
        normalized is None
        or Path(normalized).suffix.lower() not in _DISPLAY_MEDIA_SUFFIXES
    ):
        raise LegacyPreviewError(
            "requested legacy display path is not supported media"
        )

    lexical_run = workspace / run_id
    if (
        lexical_run.resolve() != run
        or run.relative_to(workspace).as_posix() != run_id
    ):
        raise LegacyPreviewError(
            "legacy display run path contains indirection"
        )

    candidate = workspace / normalized
    resolved = candidate.resolve()
    try:
        resolved.relative_to(run)
    except ValueError as exc:
        raise LegacyPreviewError(
            "requested legacy display media resolves outside its run"
        ) from exc
    if resolved.relative_to(workspace).as_posix() != normalized:
        raise LegacyPreviewError(
            "requested legacy display path contains indirection"
        )
    identity = _file_identity(os.lstat(candidate))
    _record, payload = stable_regular_file_record(
        candidate,
        capture_bytes=True,
        code="DASHBOARD_LEGACY_DISPLAY_MEDIA_INVALID",
    )
    if (
        lexical_run.resolve() != run
        or candidate.resolve() != resolved
        or _file_identity(os.lstat(candidate)) != identity
    ):
        raise LegacyPreviewError(
            "legacy display media changed during verification"
        )
    return candidate, payload


__all__ = [
    "LEGACY_PREVIEW_ALLOWLIST_NAME",
    "LEGACY_PREVIEW_ALLOWLIST_POLICY",
    "LEGACY_PREVIEW_ALLOWLIST_SCHEMA",
    "LEGACY_PREVIEW_EXPECTED_ENTRY_COUNT",
    "LEGACY_PREVIEW_EXPECTED_INVENTORY_PRESENT_COUNT",
    "LEGACY_PREVIEW_EXPECTED_MANIFEST_PRESENT_COUNT",
    "LegacyPreviewError",
    "legacy_dashboard_display_allowlist",
    "read_legacy_dashboard_media",
    "validate_legacy_revision_preview_allowlist",
]
