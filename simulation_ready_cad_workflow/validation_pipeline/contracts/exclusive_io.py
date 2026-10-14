"""Crash-safe exclusive JSON files used by the ReAct parent."""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any


def encode_json(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _matches_regular_file(path: Path, payload: bytes) -> bool:
    """Compare one existing non-symlink regular file with bounded bytes."""

    try:
        metadata = path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != len(payload):
        return False
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        current = bytearray()
        while len(current) < len(payload):
            chunk = os.read(descriptor, len(payload) - len(current))
            if not chunk:
                break
            current.extend(chunk)
        return bytes(current) == payload and os.read(descriptor, 1) == b""
    finally:
        os.close(descriptor)


def write_new_bytes(
    path: Path,
    payload: bytes,
    *,
    allow_identical: bool = False,
) -> bool:
    """Publish complete bytes without overwriting an existing destination.

    The payload is fsynced under a private temporary name and then hard-linked
    into place, so a process crash cannot leave a partial destination.  When a
    caller is retrying a state transition, it may explicitly accept an existing
    byte-identical file.  A different or unsafe destination still fails closed.
    """

    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("exclusive JSON write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            if allow_identical and _matches_regular_file(path, payload):
                return False
            raise
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) != 1:
                raise
            # exFAT and some Windows volumes reject hard links with
            # ERROR_INVALID_FUNCTION.  A same-directory Windows rename is
            # still atomic and refuses to replace an existing destination.
            try:
                os.rename(temporary, path)
            except FileExistsError:
                if allow_identical and _matches_regular_file(path, payload):
                    return False
                raise
        if os.name != "nt":
            directory = os.open(
                path.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_new_json(
    path: Path,
    value: Any,
    *,
    allow_identical: bool = False,
) -> bool:
    """Encode and exclusively publish canonical human-readable JSON."""

    return write_new_bytes(
        path,
        encode_json(value),
        allow_identical=allow_identical,
    )
