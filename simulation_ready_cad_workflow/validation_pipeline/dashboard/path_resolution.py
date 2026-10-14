"""Resolve saved run paths after a trusted run is copied between workspaces."""

from __future__ import annotations

from pathlib import Path


WORKSPACE_ANCHOR = "simulation_ready_cad_workflow"
RUNS_ANCHOR = ".runtime_runs"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _anchored_candidates(workspace: Path, value: str | Path) -> list[Path]:
    parts = [
        part
        for part in str(value).replace("\\", "/").split("/")
        if part not in {"", "."}
    ]
    candidates: list[Path] = []

    if WORKSPACE_ANCHOR in parts:
        index = parts.index(WORKSPACE_ANCHOR)
        suffix = parts[index + 1 :]
        if workspace.name.casefold() == WORKSPACE_ANCHOR.casefold():
            candidates.append(workspace.joinpath(*suffix))
        candidates.append(workspace.joinpath(WORKSPACE_ANCHOR, *suffix))

    if RUNS_ANCHOR in parts:
        index = parts.index(RUNS_ANCHOR)
        suffix = parts[index + 1 :]
        candidates.append(workspace.joinpath(RUNS_ANCHOR, *suffix))
        candidates.append(
            workspace.joinpath(WORKSPACE_ANCHOR, RUNS_ANCHOR, *suffix)
        )

    return candidates


def resolve_workspace_reference(
    workspace_root: str | Path,
    value: str | Path,
    *,
    require_within_workspace: bool = False,
    require_exists: bool = False,
) -> Path:
    """Resolve a run reference, relocating a stale cross-host path when possible.

    Validator artifacts remain immutable. Relocation is presentation-only and
    is accepted only when the referenced suffix already exists under a known
    workspace anchor on the receiving host.
    """

    workspace = Path(workspace_root).resolve()
    from ..snapshot_paths import mapped_path
    mapped = mapped_path(workspace, value)
    if mapped is not None:
        if require_exists and not mapped.exists():
            raise FileNotFoundError(mapped)
        if require_within_workspace and not mapped.is_relative_to(workspace):
            raise ValueError(f'Imported path escapes its workspace: {value}')
        return mapped
    raw = Path(value)
    original = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()

    if original.exists() and (
        not require_within_workspace or _inside(original, workspace)
    ):
        return original

    for candidate in _anchored_candidates(workspace, value):
        resolved = candidate.resolve()
        if _inside(resolved, workspace) and resolved.exists():
            return resolved

    if require_within_workspace and not _inside(original, workspace):
        raise ValueError(f"path is outside workspace and could not be relocated: {value}")
    if require_exists and not original.exists():
        raise FileNotFoundError(original)
    return original
