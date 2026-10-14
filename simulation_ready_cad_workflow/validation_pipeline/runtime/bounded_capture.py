from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import BinaryIO


def empty_log_capture() -> dict[str, object]:
    return {
        "bytes_seen": 0,
        "bytes_stored": 0,
        "truncated": False,
        "error": None,
    }


class BoundedPipeCapture:
    def __init__(
        self,
        name: str,
        stream: BinaryIO,
        path: Path,
        max_bytes: int,
    ) -> None:
        self.name = name
        self.stream = stream
        self.path = path
        self.max_bytes = max_bytes
        self.bytes_seen = 0
        self.error: str | None = None
        self._head = bytearray()
        self._tail = bytearray()
        self._lock = threading.Lock()
        self._head_limit = max_bytes // 2
        self._tail_limit = max_bytes - self._head_limit
        self._thread = threading.Thread(
            target=self._drain,
            name=f"queue-{name}-capture",
            daemon=True,
        )
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def _drain(self) -> None:
        try:
            while True:
                chunk = self.stream.read(64 * 1024)
                if not chunk:
                    break
                with self._lock:
                    self.bytes_seen += len(chunk)
                    head_remaining = self._head_limit - len(self._head)
                    if head_remaining > 0:
                        self._head.extend(chunk[:head_remaining])
                        chunk = chunk[head_remaining:]
                    if chunk:
                        self._tail.extend(chunk)
                        overflow = len(self._tail) - self._tail_limit
                        if overflow > 0:
                            del self._tail[:overflow]
        except BaseException as exc:
            with self._lock:
                self.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                self.stream.close()
            except OSError:
                pass

    def exceeds_limit(self) -> bool:
        with self._lock:
            return self.bytes_seen > self.max_bytes

    def finish(self, *, timeout_seconds: float) -> dict[str, object]:
        if self._started:
            self._thread.join(timeout=timeout_seconds)
        else:
            with self._lock:
                self.error = self.error or "capture thread was not started"
            try:
                self.stream.close()
            except OSError as exc:
                with self._lock:
                    self.error = self.error or f"stream close failed: {exc}"
        with self._lock:
            if self._thread.is_alive():
                self.error = self.error or "capture thread did not stop"
            bytes_seen = self.bytes_seen
            error = self.error
            head = bytes(self._head)
            tail = bytes(self._tail)

        truncated = bytes_seen > self.max_bytes
        if truncated:
            marker = (
                "\n[... process log truncated; "
                f"{bytes_seen - self.max_bytes} or more bytes omitted ...]\n"
            ).encode("utf-8")
            content_budget = max(0, self.max_bytes - len(marker))
            head_bytes = min(len(head), content_budget // 2)
            tail_bytes = min(
                len(tail),
                content_budget - head_bytes,
            )
            payload = (
                head[:head_bytes]
                + marker
                + (
                    tail[-tail_bytes:]
                    if tail_bytes
                    else b""
                )
            )
        else:
            payload = head + tail

        try:
            atomic_write_bytes(self.path, payload)
        except OSError as exc:
            error = error or f"could not write bounded log: {exc}"
        bytes_stored = 0
        try:
            bytes_stored = self.path.stat().st_size
        except OSError:
            pass
        return {
            "bytes_seen": bytes_seen,
            "bytes_stored": bytes_stored,
            "truncated": truncated,
            "error": error,
        }


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor: int | None = None
    write_error: OSError | None = None
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_BINARY", 0)
        descriptor = os.open(temporary, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if not isinstance(written, int) or written <= 0:
                raise OSError("bounded log write made no forward progress")
            offset += written
        os.fsync(descriptor)
    except OSError as exc:
        write_error = exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError as exc:
                write_error = write_error or exc

    try:
        if write_error is not None:
            raise write_error
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
