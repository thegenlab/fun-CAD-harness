from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime, timezone


try:
    import psutil
except ImportError:  # pragma: no cover - exercised by deployment checks
    psutil = None


def process_exists(pid: int) -> bool:
    """Return True when a PID exists or liveness cannot be measured safely."""
    if psutil is None:
        return True
    try:
        return bool(psutil.pid_exists(pid))
    except (OSError, ValueError):
        return True


def terminate_process_tree(
    pid: int,
    grace_seconds: float,
) -> tuple[int, ...]:
    """Terminate a process tree within one shared wall-clock grace budget.

    The returned tuple contains PIDs that could not be confirmed stopped.
    Callers may use an operating-system-specific fallback for those PIDs.
    """
    if psutil is None:
        return (pid,)

    try:
        root = psutil.Process(pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        return ()
    except psutil.AccessDenied:
        return (pid,)

    deadline = time.monotonic() + max(float(grace_seconds), 0.0)
    processes = [*reversed(descendants), root]
    uncertain_pids: set[int] = set()
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            uncertain_pids.discard(int(process.pid))
        except psutil.AccessDenied:
            uncertain_pids.add(int(process.pid))

    first_wait = max(0.0, (deadline - time.monotonic()) / 2.0)
    if first_wait > 0:
        gone, alive = psutil.wait_procs(processes, timeout=first_wait)
        uncertain_pids.difference_update(int(process.pid) for process in gone)
    else:
        alive = processes
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            uncertain_pids.discard(int(process.pid))
        except psutil.AccessDenied:
            uncertain_pids.add(int(process.pid))

    final_wait = max(0.0, deadline - time.monotonic())
    if alive and final_wait > 0:
        gone, alive = psutil.wait_procs(alive, timeout=final_wait)
        uncertain_pids.difference_update(int(process.pid) for process in gone)
    survivors = set(uncertain_pids)
    for process in alive:
        try:
            if process.is_running():
                survivors.add(int(process.pid))
        except psutil.NoSuchProcess:
            survivors.discard(int(process.pid))
        except psutil.AccessDenied:
            survivors.add(int(process.pid))
    return tuple(sorted(survivors))


@dataclass(frozen=True)
class SystemSnapshot:
    captured_at: str
    cpu_percent: float
    memory_percent: float
    available_memory_bytes: int
    total_memory_bytes: int
    free_disk_bytes: int
    total_disk_bytes: int

    @property
    def available_memory_gib(self) -> float:
        return self.available_memory_bytes / (1024**3)

    @property
    def free_disk_gib(self) -> float:
        return self.free_disk_bytes / (1024**3)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["available_memory_gib"] = round(self.available_memory_gib, 3)
        payload["free_disk_gib"] = round(self.free_disk_gib, 3)
        return payload


class SystemProbe:
    """psutil-backed measurements used by the fail-closed queue runner."""

    def __init__(
        self,
        *,
        disk_path: str | os.PathLike[str] | None = None,
    ) -> None:
        if psutil is None:
            raise RuntimeError(
                "psutil is required for resource-safe execution; "
                "install it in the Python environment running the queue"
            )
        self.disk_path = Path(disk_path or os.getcwd()).resolve()

    def sample(self, interval_seconds: float = 0.0) -> SystemSnapshot:
        cpu_percent = float(psutil.cpu_percent(interval=interval_seconds))
        memory = psutil.virtual_memory()
        disk = shutil.disk_usage(self.disk_path)
        return SystemSnapshot(
            captured_at=datetime.now(timezone.utc).isoformat(),
            cpu_percent=cpu_percent,
            memory_percent=float(memory.percent),
            available_memory_bytes=int(memory.available),
            total_memory_bytes=int(memory.total),
            free_disk_bytes=int(disk.free),
            total_disk_bytes=int(disk.total),
        )

    def process_tree_rss_bytes(self, pid: int) -> int:
        total = 0
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0
        for process in processes:
            try:
                total += int(process.memory_info().rss)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                return 0
        return total

    def lower_process_tree_priority(self, pid: int) -> None:
        if os.name != "nt":
            return
        try:
            root = psutil.Process(pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return
        for process in processes:
            try:
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def pid_exists(self, pid: int) -> bool:
        return process_exists(pid)

    def process_create_time(self, pid: int) -> float | None:
        try:
            return float(psutil.Process(pid).create_time())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
