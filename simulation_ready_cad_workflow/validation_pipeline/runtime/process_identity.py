"""Process owner identity that is not fooled by PID reuse."""

from __future__ import annotations

import os
from typing import Any, Mapping

from .system_probe import SystemProbe


def current_process_identity(probe: SystemProbe | None = None) -> dict[str, Any]:
    pid = os.getpid()
    helper = probe or SystemProbe()
    return {
        "pid": pid,
        "create_time": helper.process_create_time(pid),
    }


def process_identity_matches(
    recorded: Mapping[str, Any] | None,
    *,
    probe: SystemProbe | None = None,
) -> bool:
    """True only when the recorded PID is still the same process start."""

    if not isinstance(recorded, Mapping):
        return False
    pid = recorded.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    helper = probe or SystemProbe()
    if not helper.pid_exists(pid):
        return False
    recorded_start = recorded.get("create_time")
    live_start = helper.process_create_time(pid)
    if recorded_start is None or live_start is None:
        return False
    return abs(float(live_start) - float(recorded_start)) < 1e-3
