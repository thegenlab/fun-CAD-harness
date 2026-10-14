"""Bounded host-resource admission using an explicit declared requirement."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable


RESOURCE_POLICY_REQUIRED = "RESOURCE_POLICY_REQUIRED"


def _as_requirement(value: Any, *, source: str) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{source}: resource settings must be an object")
    block = value.get("resource_admission", value)
    if not isinstance(block, dict):
        raise ValueError(f"{source}: resource_admission must be an object")
    raw = block.get("required_available_memory_gib")
    if raw is None and "required_available_memory_gib" not in block and "resource_admission" not in value:
        return None
    try:
        required = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: required_available_memory_gib must be explicit, positive and finite") from exc
    if isinstance(raw, bool) or not math.isfinite(required) or required <= 0:
        raise ValueError(f"{source}: required_available_memory_gib must be explicit, positive and finite")
    try:
        max_wait = float(block.get("max_resource_wait_seconds", 30.0))
        poll = float(block.get("resource_poll_seconds", 0.05))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: resource wait and poll durations must be finite numbers") from exc
    if not math.isfinite(max_wait) or not math.isfinite(poll) or max_wait < 0 or poll <= 0:
        raise ValueError(f"{source}: resource wait must be finite and nonnegative; poll must be positive and finite")
    return {
        "required_available_memory_gib": required,
        "max_resource_wait_seconds": max_wait,
        "resource_poll_seconds": poll,
        "source": source,
    }


def load_declared_resource_requirement(project, modeling_policy=None) -> dict[str, Any] | None:
    """Meaningful predeclared host-memory requirement, or None.

    A process RSS cap is not a host-available-memory admission requirement.
    """
    root = Path(project) if project is not None else None
    if root is not None:
        for name in ("resource_policy.json", "scope.json"):
            path = root / name
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            found = _as_requirement(payload, source=str(path))
            if found is not None:
                return found
    return _as_requirement(modeling_policy, source="modeling_policy")


def wait_for_observed_resources(
    probe,
    required_available_memory_gib: float,
    *,
    max_wait_seconds: float,
    poll_seconds: float = 0.05,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Pause until observed available memory meets the declared requirement."""
    try:
        required = float(required_available_memory_gib)
        max_wait = float(max_wait_seconds)
        poll = float(poll_seconds)
        valid = (not isinstance(required_available_memory_gib, bool)
                 and math.isfinite(required) and required > 0
                 and math.isfinite(max_wait) and max_wait >= 0
                 and math.isfinite(poll) and poll > 0)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        return {
            "status": RESOURCE_POLICY_REQUIRED,
            "waited_seconds": 0.0,
            "reason": "Memory and poll thresholds must be explicit, positive and finite; the wait must be finite and nonnegative",
            "required_available_memory_gib": required_available_memory_gib,
            "samples": [],
        }
    started = time.monotonic()
    samples: list[dict[str, Any]] = []
    while True:
        snapshot = probe.sample(0.0)
        payload = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
        samples.append(payload)
        samples = samples[-3:]
        available = float(getattr(snapshot, "available_memory_gib", payload.get("available_memory_gib") or 0.0))
        if math.isfinite(available) and available >= required:
            return {
                "status": "ready",
                "waited_seconds": time.monotonic() - started,
                "required_available_memory_gib": required_available_memory_gib,
                "observed_available_memory_gib": available,
                "samples": samples[-3:],
            }
        elapsed = time.monotonic() - started
        if elapsed >= max_wait:
            return {
                "status": "wait_exhausted",
                "waited_seconds": elapsed,
                "required_available_memory_gib": required_available_memory_gib,
                "observed_available_memory_gib": available,
                "samples": samples[-3:],
            }
        if should_stop and should_stop():
            return {
                "status": "interrupted",
                "waited_seconds": elapsed,
                "required_available_memory_gib": required_available_memory_gib,
                "observed_available_memory_gib": available,
                "samples": samples[-3:],
            }
        time.sleep(min(poll, max(0.0, max_wait - elapsed)))


def resource_policy_required_result(*, reason: str, evidence=None) -> dict[str, Any]:
    return {
        "status": RESOURCE_POLICY_REQUIRED,
        "issues": [reason],
        "recoverable": True,
        "needs_external_action": True,
        "evidence": evidence or {},
        "recovery_scope": "Do not rerun the same OOM without a declared host-memory requirement",
    }
