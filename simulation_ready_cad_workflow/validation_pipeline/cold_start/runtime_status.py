"""Program-owned activity reporting, independent of author and Harness turns."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import threading
from typing import Any

from ..mechanism.io import write_json_atomic


class RuntimeHeartbeat:
    """Keep the dashboard current while a long operation owns the main thread."""

    def __init__(self, project: Path, state: dict[str, Any], interval: float = 2.0):
        self.path = Path(project) / "workflow_runtime.json"
        self.state = state
        self.interval = interval
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.pid = os.getpid()
        try:
            import psutil

            self.process_started_at = psutil.Process(self.pid).create_time()
        except Exception:
            self.process_started_at = None

    def _publish(self, status: str) -> None:
        now = datetime.now(timezone.utc)
        elapsed = self.state.get("elapsed_seconds", 0.0)
        try:
            started = datetime.fromisoformat(self.state["started_at"].replace("Z", "+00:00"))
            elapsed = max(float(elapsed), 0.0, (now - started).total_seconds())
        except (KeyError, TypeError, ValueError):
            pass
        value = {
            "schema_version": "workflow_runtime_v1" if self.state.get("schema_version") == "slim_harness_workflow_v1" else "cold_start_runtime_v1",
            "pid": self.pid,
            "process_started_at_epoch": self.process_started_at,
            "heartbeat_at": now.isoformat(),
            "status": status,
            "elapsed_seconds": elapsed,
            "phase": self.state.get("phase"),
            "revision": self.state.get("revision"),
            "active_turn": self.state.get("active_turn"),
            "stop_reason": self.state.get("stop_reason"),
        }
        try:
            write_json_atomic(self.path, value)
        except OSError:
            # A presentation write must never turn a mechanical run into a failure.
            logging.getLogger(__name__).warning("Could not publish workflow heartbeat", exc_info=True)

    def _loop(self) -> None:
        while not self.stop.wait(self.interval):
            self._publish("running")

    def __enter__(self) -> "RuntimeHeartbeat":
        self._publish("running")
        self.thread = threading.Thread(target=self._loop, name="workflow-heartbeat", daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        self._publish("complete" if self.state.get("complete_instance_pass") else "stopped")
