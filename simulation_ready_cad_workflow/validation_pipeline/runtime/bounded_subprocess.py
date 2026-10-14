"""Run one Python child with wall, RSS, output, and process-group bounds."""

from __future__ import annotations

import os
import math
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .process_containment import (
    WINDOWS_CREATE_SUSPENDED,
    ProcessContainment,
)
from .system_probe import SystemProbe


DEFAULT_OUTPUT_BYTES = 16 * 1024 * 1024


class BoundedSubprocessError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class BoundedSubprocessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    memory_budget: dict | None = None


def run_bounded_subprocess(
    command: Sequence[str],
    *,
    timeout_seconds: float | None,
    max_rss_bytes: int | str | None = None,
    max_output_bytes: int = DEFAULT_OUTPUT_BYTES,
    env: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    input_bytes: bytes = b"",
    poll_seconds: float = 0.05,
    experimental_no_timeout: bool = False,
    cooperative_stop: Callable[[subprocess.Popen], None] | None = None,
    cleanup_grace_seconds: float = 5.0,
    popen_options: Mapping[str, object] | None = None,
) -> BoundedSubprocessResult:
    started = time.monotonic()
    cooperative = None
    if cooperative_stop is not None:
        if (not callable(cooperative_stop) or type(cleanup_grace_seconds) not in (int, float)
                or not math.isfinite(cleanup_grace_seconds) or not 0 < cleanup_grace_seconds <= 90):
            raise ValueError('A trusted stop callback requires a finite cleanup grace of at most90seconds')
        cooperative = {'requested': False, 'grace_seconds': cleanup_grace_seconds,
                       'root_and_pipes_closed_during_grace': False, 'fallback_used': False}
    # Explicit experiment-only opt-in. Ordinary callers retain their deadline.
    if timeout_seconds is None and not experimental_no_timeout:
        raise BoundedSubprocessError("BOUNDED_SUBPROCESS_TIMEOUT", "An unlimited run requires the experimental opt-in")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise BoundedSubprocessError(
            "BOUNDED_SUBPROCESS_TIMEOUT",
            "timeout_seconds must be positive",
        )
    memory_guard=None
    def memory_report():
        return memory_guard.report if memory_guard else {"mode":"auto" if max_rss_bytes=="auto" else "explicit","max_rss_bytes":max_rss_bytes}
    def failure(code,message):
        error=BoundedSubprocessError(code,message)
        error.memory_budget=memory_report()
        error.wall_seconds_including_cleanup=time.monotonic()-started
        if cooperative is not None:error.cooperative_cleanup=dict(cooperative)
        return error
    if max_rss_bytes=="auto":
        from .memory_budget import AutoMemoryLimit
        try:memory_guard=AutoMemoryLimit()
        except (OSError,ValueError) as exc:
            raise failure("BOUNDED_SUBPROCESS_MEMORY_UNAVAILABLE",str(exc)) from exc
        max_rss_bytes=memory_guard.initial
    containment = ProcessContainment()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            list(command),
            cwd=None if cwd is None else str(cwd),
            env=dict(env) if env is not None else None,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            start_new_session=os.name != "nt",
            creationflags=(
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | WINDOWS_CREATE_SUSPENDED
                if os.name == "nt"
                else 0
            ),
            **dict(popen_options or {}),
        )
        containment.attach_and_resume(
            process,
            # The watcher owns the typed RSS error.  A native job memory
            # limit could kill the worker before the watcher records why.
            max_job_memory_bytes=None,
        )
    except OSError as exc:
        if process is not None:
            _kill_process_group(process)
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
        try:
            containment.close()
        except OSError:
            pass
        raise failure(
            "BOUNDED_SUBPROCESS_START_FAILED",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    stop = threading.Event()
    rss_exceeded = threading.Event()

    def _watch_rss() -> None:
        if max_rss_bytes is None:
            return
        probe = SystemProbe()
        while not stop.wait(max(0.2 if memory_guard else 0.05, poll_seconds)):
            try:
                rss = probe.process_tree_rss_bytes(process.pid)
                limit=memory_guard.limit_for_rss(rss) if memory_guard else max_rss_bytes
            except Exception:
                continue
            if rss > limit:
                rss_exceeded.set()
                return

    watcher = threading.Thread(target=_watch_rss, daemon=True)
    watcher.start()
    communication_done = threading.Event()
    communication_result: list[tuple[bytes, bytes]] = []
    communication_error: list[BaseException] = []

    def _communicate() -> None:
        try:
            communication_result.append(
                process.communicate(input=input_bytes or None)
            )
        except BaseException as exc:
            communication_error.append(exc)
        finally:
            communication_done.set()

    communicator = threading.Thread(target=_communicate, daemon=True)
    communicator.start()
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    timed_out = False
    latched_limit = None
    tree_closed = False
    try:
        while not communication_done.is_set():
            now = time.monotonic()
            root_finished = process.poll() is not None
            deadline_reached = deadline is not None and now >= deadline
            if root_finished or rss_exceeded.is_set() or deadline_reached:
                timed_out = deadline_reached and not root_finished
                latched_limit = ('BOUNDED_SUBPROCESS_TIMEOUT' if timed_out else
                                 'BOUNDED_SUBPROCESS_RSS' if rss_exceeded.is_set() else None)
                if latched_limit is not None and cooperative_stop is not None and not root_finished:
                    cooperative['latched_limit'] = latched_limit
                    cooperative['decision_wall_seconds'] = time.monotonic()-started
                    cleanup_started = time.monotonic()
                    try:
                        cooperative_stop(process)
                        cooperative['requested'] = True
                        cooperative['root_and_pipes_closed_during_grace'] = communication_done.wait(cleanup_grace_seconds)
                    except Exception as exc:
                        cooperative['callback_error_type'] = type(exc).__name__
                    cooperative['cleanup_wall_seconds'] = time.monotonic()-cleanup_started
                    cooperative['fallback_used'] = not cooperative['root_and_pipes_closed_during_grace']
                _close_process_tree(process, containment)
                tree_closed = True
                break
            communication_done.wait(
                max(0.01, poll_seconds) if deadline is None else min(max(0.01, poll_seconds), max(0.0, deadline - now))
            )
        if not communication_done.wait(1.0):
            _kill_process_group(process)
            communication_done.wait(1.0)
    finally:
        stop.set()
        watcher.join(timeout=1.0)
        if not tree_closed:
            _close_process_tree(process, containment)
        communicator.join(timeout=1.0)
    if latched_limit == 'BOUNDED_SUBPROCESS_TIMEOUT' or timed_out:
        raise failure(
            "BOUNDED_SUBPROCESS_TIMEOUT",
            f"child exceeded {timeout_seconds:.3f}s wall budget",
        )
    if latched_limit == 'BOUNDED_SUBPROCESS_RSS' or rss_exceeded.is_set():
        raise failure("BOUNDED_SUBPROCESS_RSS","child process tree exceeded memory budget")
    if communication_error:
        exc = communication_error[0]
        raise failure(
            "BOUNDED_SUBPROCESS_IO_FAILED",
            f"{type(exc).__name__}: {exc}",
        ) from exc
    if not communication_result:
        raise failure(
            "BOUNDED_SUBPROCESS_CLEANUP_FAILED",
            "child process pipes did not close after process-tree cleanup",
        )
    stdout, stderr = communication_result[0]
    if len(stdout) > max_output_bytes or len(stderr) > max_output_bytes:
        raise failure(
            "BOUNDED_SUBPROCESS_OUTPUT_LIMIT",
            f"child output exceeded {max_output_bytes} bytes",
        )
    return BoundedSubprocessResult(
        returncode=int(process.returncode or 0),
        stdout=stdout or b"",
        stderr=stderr or b"",
        memory_budget=memory_report(),
    )


def _close_process_tree(
    process: subprocess.Popen[bytes],
    containment: ProcessContainment,
) -> None:
    """Close native containment before a vanished root loses its children."""

    if os.name == "nt":
        try:
            containment.close()
            return
        except OSError:
            pass
    _kill_process_group(process)


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except (ProcessLookupError, PermissionError, OSError):
                return
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except (ProcessLookupError, PermissionError, OSError):
            return
