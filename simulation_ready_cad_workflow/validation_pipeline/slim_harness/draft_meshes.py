"""Cached meshes that sandbox draft commands derive for each attempt.

Every draft command prepares its own OBJ/GLB under ``<attempt>/geometry`` from
that attempt's retained source snapshot. Build, preview, check, simulate and
probe never read another attempt's meshes; ``audit`` reads those of the saved
simulation request it is given. Source snapshots, STEP files, results, logs
and traces are evidence and are never treated as cached meshes here.

A cached mesh is stored once per project: a new revision receives it as a hard
link to the previous revision's file, so every recorded path keeps working.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

ATTEMPTS = ('.cadharness-draft', '.runtime_runs', 'author_tools')
MESH_SUFFIXES = frozenset({'.obj', '.glb'})
RECENT_ATTEMPTS = 3
REMOVED_NOTE = 'OLDER_MESHES_REMOVED.txt'


def is_derived_mesh(relative):
    """True for a cached mesh below one draft attempt's geometry folder."""
    parts = Path(relative).parts
    return (parts[:3] == ATTEMPTS and len(parts) > 5 and 'geometry' in parts[4:-1]
            and Path(parts[-1]).suffix.lower() in MESH_SUFFIXES)


def declared_sources(design):
    """Files the design's own scene names as CAD sources; these are never cache."""
    design = Path(design)
    try:
        scene = json.loads((design / 'scene.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return set()
    if not isinstance(scene, dict):
        return set()
    bodies = scene.get('bodies') if isinstance(scene.get('bodies'), list) else []
    paths = [body['source'] for body in bodies if isinstance(body, dict) and isinstance(body.get('source'), str)]
    authoring = scene.get('authoring_sources')
    paths += [path for path in authoring if isinstance(path, str)] if isinstance(authoring, list) else []

    def source_refs(value):
        if isinstance(value, dict):
            refs = value.get('source_refs')
            for ref in refs if isinstance(refs, list) else []:
                if isinstance(ref, dict) and isinstance(ref.get('path'), str):
                    paths.append(ref['path'])
            for child in value.values():
                source_refs(child)
        elif isinstance(value, list):
            for child in value:
                source_refs(child)
    source_refs(scene)
    return {(design / path).resolve() for path in paths}


def _derived_meshes(design):
    """Ordinary cached mesh files of a design, as (path, relative) pairs."""
    design = Path(design)
    root = design.joinpath(*ATTEMPTS)
    if not root.is_dir() or any(path.is_symlink() for path in (root, *root.parents) if path.is_relative_to(design)):
        return
    declared = declared_sources(design)
    for current, directories, names in os.walk(root, followlinks=False):
        directories[:] = sorted(name for name in directories if not os.path.islink(os.path.join(current, name)))
        for name in sorted(names):
            path = Path(current) / name
            relative = path.relative_to(design)
            if (is_derived_mesh(relative) and not path.is_symlink() and path.is_file()
                    and path.resolve() not in declared):
                yield path, relative


def revision_copy_ignore(previous):
    """copytree filter for a new revision: skip caches that are hard-linked instead of copied."""
    shared = {relative.as_posix() for _path, relative in _derived_meshes(previous)}

    def ignore(directory, names):
        folder = Path(os.path.relpath(directory, previous))
        return {name for name in names
                if name in {'__pycache__', '.presim_gate'} or (folder / name).as_posix() in shared}
    return ignore


def share_derived_meshes(previous, design):
    """Give the new revision the previous revision's cached meshes as hard links.

    Draft records name a mesh by its path in the revision where the attempt
    was made and are copied unchanged, so that path must stay valid. A hard
    link keeps both paths with one copy of the bytes. Publication replaces a
    changed file by rename and leaves an unchanged one alone, so the previous
    revision can never be written through its link. A file that cannot be
    linked is copied; nothing is moved or removed, so the caller may link into
    a staged folder and discard it if the revision is not completed.
    """
    previous, design = Path(previous), Path(design)
    linked = total = 0
    try:
        shared = list(_derived_meshes(previous))
    except OSError:
        shared = []
    for path, relative in shared:
        target = design / relative
        try:
            if target.exists() or target.is_symlink() or not target.parent.is_dir():
                continue
            size = path.stat().st_size
            os.link(path, target)
        except OSError:
            try:
                shutil.copy2(path, target)
            except OSError:
                try:
                    target.unlink()  # Never leave a truncated mesh in the new revision.
                except OSError:
                    pass
            continue
        linked += 1
        total += size
    return {'files': linked, 'bytes': total}


def _started(attempt):
    try:
        value = json.loads((attempt / 'capture.json').read_text(encoding='utf-8')).get('started_ms')
    except (OSError, ValueError, AttributeError):
        value = None
    try:
        modified = attempt.stat().st_mtime_ns
    except OSError:
        modified = 0
    return (value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0, modified)


def prune_older_attempt_meshes(attempts, current, design, keep=RECENT_ATTEMPTS):
    """Bound a sandbox draft: only the most recent attempts keep their cached OBJ meshes.

    An older attempt is reduced only once its command has finished, and never
    when it saved a simulation request: ``audit`` reads that attempt's meshes
    again through the request. Reduced attempts keep their source snapshot,
    results, logs, traces and GLB. Runs inside the agent's own workspace and
    never fails the calling command.
    """
    try:
        attempts, current = Path(attempts), Path(current)
        declared = declared_sources(design)
        older = sorted((path for path in attempts.iterdir()
                        if path.is_dir() and not path.is_symlink() and path != current), key=_started)
        for attempt in older[:max(0, len(older) - (keep - 1))]:
            if (attempt / REMOVED_NOTE).is_file() or not (attempt / 'result.json').is_file():
                continue  # Already reduced, or its command is still running.
            meshes, has_request = [], False
            for current_dir, directories, names in os.walk(attempt, followlinks=False):
                directories[:] = [name for name in directories if not os.path.islink(os.path.join(current_dir, name))]
                for name in names:
                    path = Path(current_dir) / name
                    has_request = has_request or name == 'request.json'
                    relative = Path(*ATTEMPTS) / path.relative_to(attempts)
                    if (path.suffix.lower() == '.obj' and is_derived_mesh(relative) and not path.is_symlink()
                            and path.is_file() and path.resolve() not in declared):
                        meshes.append(path)
            if has_request or not meshes:
                continue
            total = 0
            for path in meshes:
                total += path.stat().st_size
                path.unlink()
            (attempt / REMOVED_NOTE).write_text(
                f'{len(meshes)} cached OBJ meshes ({total} bytes) of this older draft attempt were removed; '
                f'only the {keep} most recent attempts keep them.\n'
                'Its source snapshot, results, logs, traces and GLB remain. '
                'Run build or check on the source again to regenerate meshes.\n', encoding='utf-8')
    except Exception:
        pass
