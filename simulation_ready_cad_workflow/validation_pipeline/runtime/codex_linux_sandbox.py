"""Codex tool-sandbox policy that can actually start on this host.

Codex 0.153+ defaults tool commands to bubblewrap, which remounts ``/`` as
slave. A uid-0 process without ``CAP_SYS_ADMIN`` (typical Docker/Kubernetes
pod) fails before the command runs: ``bwrap: Failed to make / slave:
Permission denied``. Unprivileged processes create a user namespace first and
do not hit that path.
"""
from __future__ import annotations

import os
from pathlib import Path


_SYS_ADMIN = 1 << 21


def filesystem_sandbox_starts() -> bool:
    if os.name == "nt":
        return True
    # A container image that is itself the isolation boundary usually blocks
    # the user namespaces bubblewrap needs, even for unprivileged users.
    if os.environ.get("SIMREADY_CONTAINER_ISOLATION") == "1":
        return False
    try:
        if os.geteuid() != 0:
            return True
        cap = int(Path("/proc/self/status").read_text().split("CapEff:")[1].split()[0], 16)
    except (AttributeError, OSError, IndexError, ValueError):
        return True
    return bool(cap & _SYS_ADMIN)


def app_thread_sandbox(*, readonly: bool) -> str:
    if filesystem_sandbox_starts():
        return "read-only" if readonly else "workspace-write"
    return "danger-full-access"


def app_turn_sandbox_policy(*, readonly: bool, writable_roots=None) -> dict:
    if filesystem_sandbox_starts():
        if readonly:
            return {"type": "readOnly"}
        return {
            "type": "workspaceWrite",
            "writableRoots": list(writable_roots or []),
            "networkAccess": False,
        }
    return {"type": "dangerFullAccess"}


def cli_sandbox_command(*, readonly: bool) -> list[str]:
    if filesystem_sandbox_starts():
        if readonly:
            return ["--sandbox", "read-only", "-c", 'approval_policy="never"']
        return []
    return ["--sandbox", "danger-full-access", "-c", 'approval_policy="never"']
