"""Small read-only JSON helper shared by new and legacy dashboard views."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
from typing import Any

from ..mechanism.io import canonical_sha256
from ..run_layout import parse_canonical_run


_REACT_V2_PROFILE = "single_conversation_react_v2"
_REACT_V2_MEDIA_SUFFIXES = frozenset({".png", ".mp4", ".glb"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _strict_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("JSON number must be finite")
    return value


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


class DashboardReadError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def read_stable_json_object(
    path: str | Path,
    *,
    label: str,
    allowed_root: str | Path,
    max_bytes: int = 16 * 1024 * 1024,
) -> tuple[dict[str, Any], dict[str, Any]]:
    requested = Path(path)
    root = Path(allowed_root).resolve()
    if requested.is_symlink():
        raise DashboardReadError("DASHBOARD_JSON_UNSAFE", f"{label} is a symlink")
    resolved = requested.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise DashboardReadError(
            "DASHBOARD_JSON_OUTSIDE_ROOT", f"{label} is outside its read root"
        ) from exc
    before = resolved.stat()
    if not resolved.is_file() or before.st_size > max_bytes:
        raise DashboardReadError(
            "DASHBOARD_JSON_INVALID", f"{label} is missing or too large"
        )
    raw = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (before.st_size, before.st_mtime_ns, before.st_ino)
    identity_after = (after.st_size, after.st_mtime_ns, after.st_ino)
    if identity_before != identity_after or len(raw) != after.st_size:
        raise DashboardReadError(
            "DASHBOARD_JSON_CHANGED", f"{label} changed while being read"
        )
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
            parse_float=_strict_json_float,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DashboardReadError(
            "DASHBOARD_JSON_INVALID", f"{label} is not valid JSON"
        ) from exc
    if not isinstance(value, dict):
        raise DashboardReadError(
            "DASHBOARD_JSON_INVALID", f"{label} must be a JSON object"
        )
    return value, {
        "path": str(resolved),
        "size_bytes": len(raw),
        "mtime_ns": after.st_mtime_ns,
    }


def _react_v2_media_references(
    value: dict[str, Any],
) -> list[tuple[str, str | None]]:
    references: list[tuple[str, str | None]] = []
    pending: list[Any] = [value]
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > 4096:
            raise DashboardReadError(
                "DASHBOARD_MEDIA_INDEX_INVALID",
                "ReAct media index is too complex",
            )
        if isinstance(current, dict):
            for key, child in current.items():
                if (
                    isinstance(child, str)
                    and PurePosixPath(child).suffix.lower()
                    in _REACT_V2_MEDIA_SUFFIXES
                ):
                    declared = current.get(f"{key}_sha256")
                    if declared is None and key in {
                        "path",
                        "file",
                        "scene",
                        "video",
                        "output",
                    }:
                        declared = current.get("sha256")
                    references.append(
                        (child, declared if isinstance(declared, str) else None)
                    )
                    if len(references) > 2048:
                        raise DashboardReadError(
                            "DASHBOARD_MEDIA_INDEX_INVALID",
                            "ReAct media index has too many entries",
                        )
                elif isinstance(child, (dict, list)):
                    pending.append(child)
        elif isinstance(current, list):
            pending.extend(
                child for child in current if isinstance(child, (dict, list))
            )
    return references


def _stable_sha256(path: Path) -> str:
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise DashboardReadError(
            "DASHBOARD_MEDIA_INVALID", "ReAct media must be a regular file"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_size,
        before.st_mtime_ns,
        before.st_ino,
    ) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise DashboardReadError(
            "DASHBOARD_MEDIA_CHANGED", "ReAct media changed while being read"
        )
    return digest.hexdigest()


def react_v2_media_allowlist(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """Return exact run-local media referenced by a native v2 visual index."""

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    try:
        parts = run.relative_to(workspace).parts
    except ValueError as exc:
        raise DashboardReadError(
            "DASHBOARD_MEDIA_RUN_INVALID", "ReAct run is outside the workspace"
        ) from exc
    if (
        len(parts) != 5
        or parts[:2] != (".runtime_runs", "projects")
        or parts[3] != "runs"
    ):
        raise DashboardReadError(
            "DASHBOARD_MEDIA_RUN_INVALID", "ReAct media needs an exact revision"
        )
    manifest, _record = read_stable_json_object(
        run / "run_manifest.json",
        label="dashboard ReAct run manifest",
        allowed_root=run,
    )
    summary_path = run / "visual" / "dashboard_visuals.summary.json"
    expected_summary = summary_path.relative_to(workspace).as_posix()
    if (
        manifest.get("workflow_profile") != _REACT_V2_PROFILE
        or manifest.get("project_id") != parts[2]
        or manifest.get("revision") != parts[4]
        or manifest.get("visualization_index") != expected_summary
    ):
        raise DashboardReadError(
            "DASHBOARD_MEDIA_RUN_INVALID",
            "ReAct run manifest does not bind the visual index",
        )
    if not summary_path.exists():
        return {}
    summary, _summary_record = read_stable_json_object(
        summary_path,
        label="dashboard ReAct visual index",
        allowed_root=run,
    )
    allowed: dict[str, str] = {}
    for reference, declared_sha256 in _react_v2_media_references(summary):
        if "\\" in reference or "\x00" in reference:
            raise DashboardReadError(
                "DASHBOARD_MEDIA_PATH_INVALID", "ReAct media path is not portable"
            )
        relative = PurePosixPath(reference)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise DashboardReadError(
                "DASHBOARD_MEDIA_PATH_INVALID", "ReAct media path is not relative"
            )
        lexical = workspace.joinpath(*relative.parts)
        try:
            resolved = lexical.resolve(strict=True)
            resolved.relative_to(run)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise DashboardReadError(
                "DASHBOARD_MEDIA_PATH_INVALID",
                "ReAct media is missing or outside its exact run",
            ) from exc
        if lexical != resolved or lexical.is_symlink():
            raise DashboardReadError(
                "DASHBOARD_MEDIA_PATH_INVALID", "ReAct media cannot use symlinks"
            )
        digest = _stable_sha256(resolved)
        if declared_sha256 is not None and (
            _SHA256.fullmatch(declared_sha256) is None
            or declared_sha256 != digest
        ):
            raise DashboardReadError(
                "DASHBOARD_MEDIA_HASH_MISMATCH",
                "ReAct media does not match its declared digest",
            )
        allowed[relative.as_posix()] = digest
    return allowed


def react_reference_pose_media_allowlist(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """Allow the post-run physical/ideal pose video stored outside rNN."""

    workspace = Path(workspace_root).resolve()
    identity = parse_canonical_run(workspace, run_dir)
    if identity is None:
        return {}
    diagnostic_root = (
        workspace
        / ".runtime_runs"
        / "diagnostics"
        / f"{identity.run_id}-ideal-pose"
    )
    diagnostics_parent = (workspace / ".runtime_runs" / "diagnostics").resolve()
    try:
        resolved_root = diagnostic_root.resolve()
        resolved_root.relative_to(diagnostics_parent)
    except (OSError, ValueError):
        return {}
    if not resolved_root.is_dir():
        return {}
    videos = [
        *sorted(resolved_root.glob("*/physical_vs_ideal_roundtrip.mp4")),
        resolved_root / "physical_vs_ideal_roundtrip.mp4",
    ]
    allowed: dict[str, str] = {}
    for video in videos:
        if not video.is_file():
            continue
        sidecar = video.with_suffix(".sidecar.json")
        if not sidecar.is_file():
            continue
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        hashes = payload.get("trace_sha256") or {}
        if (
            payload.get("status") != "ENCODED"
            or not hashes.get("physical")
            or not hashes.get("reduced")
        ):
            continue
        try:
            resolved = video.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        allowed[resolved.relative_to(workspace).as_posix()] = _stable_sha256(
            resolved
        )
    return allowed


def react_physics_evidence_media_allowlist(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """Allow the post-run interactive physics GLB stored outside rNN."""

    workspace = Path(workspace_root).resolve()
    identity = parse_canonical_run(workspace, run_dir)
    if identity is None:
        return {}
    diagnostic_root = (
        workspace
        / ".runtime_runs"
        / "diagnostics"
        / f"{identity.run_id}-physics-evidence"
    )
    diagnostics_parent = (workspace / ".runtime_runs" / "diagnostics").resolve()
    try:
        resolved_root = diagnostic_root.resolve()
        resolved_root.relative_to(diagnostics_parent)
    except (OSError, ValueError):
        return {}
    if not resolved_root.is_dir():
        return {}
    allowed: dict[str, str] = {}
    for glb in sorted(resolved_root.glob("*/stage5_physics_evidence.glb")):
        manifest_path = glb.with_name("stage5_physics_evidence.scene.json")
        if not glb.is_file() or not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            resolved = glb.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (OSError, ValueError):
            continue
        digest = _stable_sha256(resolved)
        declared = str(payload.get("output_sha256") or "")
        if not declared or declared != digest:
            continue
        allowed[resolved.relative_to(workspace).as_posix()] = digest
    return allowed


def react_static_visual_media_allowlist(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> dict[str, str]:
    """Allow only images bound by immutable pre-physics review requests."""

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    try:
        parts = run.relative_to(workspace).parts
    except ValueError as exc:
        raise DashboardReadError(
            "DASHBOARD_MEDIA_RUN_INVALID", "ReAct run is outside the workspace"
        ) from exc
    if (
        len(parts) != 5
        or parts[:2] != (".runtime_runs", "projects")
        or parts[3] != "runs"
    ):
        raise DashboardReadError(
            "DASHBOARD_MEDIA_RUN_INVALID",
            "static visual media needs an exact ReAct revision",
        )
    project = workspace / ".runtime_runs" / "projects" / parts[2]
    candidates: dict[Path, str | None] = {
        path: None
        for path in sorted((run / "static_visual_review").glob("request-*.json"))
    }
    reuse_path = run / "static_visual_review" / "reuse.json"
    if reuse_path.is_file() and not reuse_path.is_symlink():
        try:
            reuse, _record = read_stable_json_object(
                reuse_path,
                label="dashboard static visual review reuse",
                allowed_root=run,
            )
        except (OSError, ValueError):
            reuse = {}
        unsigned = {
            key: value for key, value in reuse.items() if key != "reuse_sha256"
        }
        source_request = reuse.get("source_request")
        if (
            reuse.get("project_id") == parts[2]
            and reuse.get("revision") == parts[4]
            and reuse.get("reuse_sha256") == canonical_sha256(unsigned)
            and isinstance(source_request, dict)
        ):
            candidates[
                workspace / str(source_request.get("request_path") or "")
            ] = str(source_request.get("request_sha256") or "")
    state_path = project / "react_loop" / "state.json"
    if state_path.is_file() and not state_path.is_symlink():
        try:
            state, _record = read_stable_json_object(
                state_path,
                label="dashboard ReAct Loop state",
                allowed_root=project,
            )
        except (OSError, ValueError):
            state = {}
        if state.get("current_run") == run.relative_to(workspace).as_posix():
            review = state.get("static_visual_review")
            binding = review.get("request") if isinstance(review, dict) else None
            if isinstance(binding, dict):
                requested = workspace / str(binding.get("request_path") or "")
                candidates[requested] = str(binding.get("request_sha256") or "")

    allowed: dict[str, str] = {}
    for request_path, expected_request_sha256 in candidates.items():
        try:
            request, _record = read_stable_json_object(
                request_path,
                label="dashboard static visual review request",
                allowed_root=project,
            )
        except (OSError, ValueError):
            continue
        if expected_request_sha256 and (
            _SHA256.fullmatch(expected_request_sha256) is None
            or _stable_sha256(request_path.resolve()) != expected_request_sha256
        ):
            continue
        request_unsigned = {
            key: value for key, value in request.items() if key != "request_id"
        }
        if request.get("request_id") != canonical_sha256(request_unsigned):
            continue
        request_run = workspace / str(request.get("run") or "")
        if (
            request.get("project_id") != parts[2]
            or request_run.is_symlink()
            or not request_run.is_dir()
        ):
            continue
        try:
            request_run.resolve().relative_to(project / "runs")
        except ValueError:
            continue
        for view in request.get("views") or []:
            if not isinstance(view, dict):
                continue
            reference = str(view.get("path") or "")
            declared = str(view.get("sha256") or "")
            candidate = workspace / reference
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(request_run.resolve())
            except (FileNotFoundError, OSError, ValueError):
                continue
            if (
                candidate.is_symlink()
                or _SHA256.fullmatch(declared) is None
                or _stable_sha256(resolved) != declared
            ):
                continue
            allowed[resolved.relative_to(workspace).as_posix()] = declared
    return allowed
