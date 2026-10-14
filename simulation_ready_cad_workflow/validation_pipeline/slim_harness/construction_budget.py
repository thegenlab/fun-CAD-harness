"""Active-work accounting for benchmark construction, never frozen evaluation.

The budget counts the wall-clock union of live author/reviewer/tool intervals,
not a sum of nested subprocesses. An agent interval measures observed request
processing (including tool use), not otherwise-unobservable exact thinking time.
Setup, provider stalls and human intervention can explicitly suspend charging.
Idle time between invocations is free. Heartbeat leases bound crash recovery:
dead owners charge through their last heartbeat, a stalled owner at most through
its lease. No project-creation-to-now wall-clock subtraction is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any
import uuid


LEDGER_NAME = "construction_budget.json"
# Simulation wall time is not charged either, but a simulation, unlike a wait,
# starts only while work time remains. A "simulation" session overrides only
# Harness-side waiting (see _settle); "simulation_wait" is the worker's own
# exclusion for a simulation shell command the author is blocked on, so it
# overrides agent time like the other waits but is never gated itself: the
# simulation it accompanies already passed the gate when it started.
EXCLUDED_KINDS = frozenset({"environment_setup", "human_intervention", "provider_wait", "stalled_wait",
                            "simulation", "simulation_wait"})
_UNGATED_KINDS = EXCLUDED_KINDS - {"simulation"}


# Construction budgets a run may choose, in hours.
BUDGET_HOURS = (1, 2, 3, 5, 8, 12)


@dataclass(frozen=True)
class BudgetConfig:
    hours: int = 2
    simulation_timeout_seconds: float = 180
    max_tasks: int = 3
    max_tests: int = 3
    max_geometry_units: int = 2048
    max_geometry_units_per_body: int = 128
    summary_reserve_seconds: float = 120
    # Source-motion audit cost differs from native solving; keep its cap
    # independent while charging both stages to the same project budget.
    audit_timeout_seconds: float = 600

    def __post_init__(self):
        if isinstance(self.hours, bool) or self.hours not in BUDGET_HOURS:
            raise ValueError("Construction budget hours must be one of " + ", ".join(map(str, BUDGET_HOURS)))
        for name in ("max_tasks", "max_tests", "max_geometry_units", "max_geometry_units_per_body"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("simulation_timeout_seconds", "audit_timeout_seconds", "summary_reserve_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.summary_reserve_seconds >= self.total_seconds:
            raise ValueError("Summary reserve must be smaller than the project budget")

    @property
    def total_seconds(self) -> float:
        return self.hours * 3600.0


class BudgetExhausted(RuntimeError):
    """Callers catch this and produce a partial completion summary, not PASS/FAIL."""

    def __init__(self, snapshot: dict[str, Any]):
        self.snapshot = snapshot
        self.details = {
            "code": "CONSTRUCTION_BUDGET_EXHAUSTED" if snapshot["exhausted"] else "CONSTRUCTION_SUMMARY_RESERVE",
            "remaining_seconds": snapshot["remaining_seconds"],
            "active_seconds": snapshot["active_seconds"],
            "required_action": "write_completion_summary",
        }
        self.code = self.details["code"]
        super().__init__(json.dumps(self.details, ensure_ascii=False))


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        # Windows GetTickCount64/time.monotonic share the machine boot origin.
        return str(round((time.time() - time.monotonic()) / 60))


def _owner_identity(pid: int) -> str | None:
    try:
        # PID reuse must not revive a crashed worker's activity interval.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _pid_namespace() -> str | None:
    try:
        return os.readlink('/proc/self/ns/pid')
    except OSError:
        return None


def _owner_alive(session: dict[str, Any]) -> bool:
    if os.name == "nt":
        # Windows os.kill(pid, 0) is not a POSIX existence probe. Lease expiry
        # still bounds interruption recovery there without signalling an owner.
        return True
    namespace = session.get('pid_namespace')
    if namespace is not None and namespace != _pid_namespace():
        # A model tool can see the ledger but not its host supervisor PID.
        # Absence in this sandbox is not owner death; use the shared lease.
        return True
    pid = session["pid"]
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    identity = session.get("process_identity")
    return identity is None or identity == _owner_identity(pid)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            if os.name == "nt" and os.fstat(stream.fileno()).st_size == 0:
                stream.write(b"0")
                stream.flush()
            yield
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write(path: Path, state: dict[str, Any]):
    name = str(path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp"))
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        deadline = time.monotonic() + 2.0
        while True:
            try:
                os.replace(name, path)
                break
            except OSError as error:
                if (os.name != 'nt' or getattr(error, 'winerror', None) not in {5, 32}
                        or time.monotonic() >= deadline):
                    raise
                # Windows readers can briefly deny replacement without owning
                # the writer lock. Keep the old complete ledger until they close.
                time.sleep(.02)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def initialize_budget(project: Path, config: BudgetConfig | None = None) -> "ProjectBudget":
    """Idempotent initialization; resuming cannot reset already-consumed time."""
    budget = ProjectBudget(project)
    with _locked(budget.lock_path):
        if budget.path.exists():
            state = json.loads(budget.path.read_text(encoding="utf-8"))
            if config is not None and asdict(BudgetConfig(**state["config"])) != asdict(config):
                raise ValueError("Construction budget already initialized with different configuration")
        else:
            config = config or BudgetConfig()
            now = time.monotonic()
            _write(budget.path, {
                "schema_version": "slim_construction_budget_v1", "config": asdict(config),
                "accounting": "union_of_observed_active_intervals", "boot_id": _boot_id(),
                "created_at_epoch": time.time(), "updated_at_epoch": time.time(),
                "settled_monotonic": now, "active_seconds": 0.0,
                "excluded_seconds": 0.0, "excluded_by_reason": {}, "sessions": {},
                "pause": None, "recovered_sessions": [],
            })
    return budget


class ProjectBudget:
    def __init__(self, project: Path, *, heartbeat_seconds: float = 1.0, lease_seconds: float = 5.0):
        if heartbeat_seconds <= 0 or lease_seconds <= heartbeat_seconds:
            raise ValueError("Heartbeat must be positive and shorter than its lease")
        self.project = Path(project).resolve()
        self.path = self.project / LEDGER_NAME
        self.lock_path = self.project / (LEDGER_NAME + ".lock")
        self.heartbeat_seconds = heartbeat_seconds
        self.lease_seconds = lease_seconds

    def _settle(self, state: dict[str, Any], now: float):
        if state["boot_id"] != _boot_id():
            # Monotonic times cannot span a reboot. Persisted work remains charged;
            # the unobserved powered-off interval is not attributed to the agent.
            state["recovered_sessions"].extend(state["sessions"])
            state["sessions"] = {}
            state["boot_id"] = _boot_id()
            state["settled_monotonic"] = now
        start = state["settled_monotonic"]
        now = max(now, start)
        intervals = []
        expired = []
        for token, session in state["sessions"].items():
            alive = _owner_alive(session)
            end = session["heartbeat_monotonic"] + (session["lease_seconds"] if alive else 0)
            if end < now or not alive:
                expired.append(token)
            if end > start:
                intervals.append((min(end, now), session["kind"]))
        # Excluded activities override parent agent intervals: installing a
        # dependency inside a model tool call is still environment setup. A
        # simulation session overrides Harness-side waiting only; an author who
        # keeps working while a background simulation runs stays charged, while
        # the worker's "simulation_wait" scopes a blocking simulation command.
        boundaries = sorted({start, now, *(end for end, _ in intervals)})
        for left, right in zip(boundaries, boundaries[1:]):
            kinds = {kind for end, kind in intervals if end > left}
            excluded = kinds & EXCLUDED_KINDS
            if "agent" in kinds:
                excluded.discard("simulation")
            reasons = sorted(excluded)
            pause = state.get("pause")
            if pause or reasons:
                reason = pause["reason"] if pause else reasons[0]
                elapsed = right - left
                state["excluded_seconds"] += elapsed
                state["excluded_by_reason"][reason] = state["excluded_by_reason"].get(reason, 0.0) + elapsed
            elif kinds:
                state["active_seconds"] += right - left
        for token in expired:
            state["sessions"].pop(token)
            state["recovered_sessions"].append(token)
        state["recovered_sessions"] = state["recovered_sessions"][-32:]
        state["settled_monotonic"] = now
        state["updated_at_epoch"] = time.time()

    def _transaction(self, change=None):
        with _locked(self.lock_path):
            state = json.loads(self.path.read_text(encoding="utf-8"))
            if state.get("schema_version") != "slim_construction_budget_v1":
                raise ValueError("Unsupported construction budget ledger")
            now = time.monotonic()
            self._settle(state, now)
            if change is not None:
                change(state, now)
            _write(self.path, state)
            return state

    @staticmethod
    def _snapshot(state):
        config = BudgetConfig(**state["config"])
        remaining = max(0.0, config.total_seconds - state["active_seconds"])
        if remaining <= 0:
            tier = "exhausted"
        elif remaining <= config.summary_reserve_seconds:
            tier = "finalizing"
        elif remaining <= max(config.total_seconds * .2, config.summary_reserve_seconds * 2):
            tier = "time_tight"
        else:
            tier = "sufficient"
        return {
            "schema_version": "slim_construction_budget_snapshot_v1",
            # Older construction ledgers omit newly introduced independent
            # caps. Expose effective defaults without changing recorded work.
            "config": asdict(config), "total_seconds": config.total_seconds,
            "active_seconds": state["active_seconds"], "excluded_seconds": state["excluded_seconds"],
            "excluded_by_reason": state["excluded_by_reason"], "remaining_seconds": remaining,
            "work_remaining_seconds": max(0.0, remaining - config.summary_reserve_seconds),
            "strategy_tier": tier, "exhausted": remaining <= 0, "pause": state["pause"],
            "paused": state["pause"] is not None,
            "active_sessions": len(state["sessions"]), "accounting": state["accounting"],
            "recovered_sessions": state["recovered_sessions"],
            "wall_seconds": max(0.0,state["updated_at_epoch"]-state["created_at_epoch"]),
            "idle_seconds": max(0.0,state["updated_at_epoch"]-state["created_at_epoch"]-state["active_seconds"]-state["excluded_seconds"]),
        }

    def snapshot(self) -> dict[str, Any]:
        return self._snapshot(self._transaction())

    @property
    def remaining_seconds(self) -> float:
        return self.snapshot()["remaining_seconds"]

    def check_available(self, *, for_summary: bool = False) -> dict[str, Any]:
        snapshot = self.snapshot()
        if snapshot["remaining_seconds" if for_summary else "work_remaining_seconds"] <= 0:
            raise BudgetExhausted(snapshot)
        return snapshot

    def pause(self, reason: str):
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A pause needs a concrete exclusion reason")
        return self._snapshot(self._transaction(lambda state, now: state.update(
            pause={"reason": reason, "started_at_epoch": time.time()})))

    def resume(self):
        return self._snapshot(self._transaction(lambda state, now: state.update(pause=None)))

    def reclassify_active(self, reason: str, seconds: float):
        """Move already-charged work into an exclusion. Does not reset the clock.

        Use when settled agent time was environment or provider failure, not
        construction work. ``seconds`` must be finite and not exceed the
        current active total.
        """
        if reason not in EXCLUDED_KINDS:
            raise ValueError("reclassify_active requires an existing exclusion reason")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("reclassify_active seconds must be finite and positive")

        def change(state, now):
            if seconds > state["active_seconds"] + 1e-9:
                raise ValueError("reclassify_active cannot exceed charged active seconds")
            state["active_seconds"] -= seconds
            state["excluded_seconds"] += seconds
            state["excluded_by_reason"][reason] = state["excluded_by_reason"].get(reason, 0.0) + seconds

        return self._snapshot(self._transaction(change))

    @contextmanager
    def activity(self, kind: str = "agent", *, heartbeat: bool = True):
        """Track live work; nested processes sharing a project never double count.

        heartbeat=False is useful only for an externally heartbeating monitor.
        Excluded waits are allowed even when the work budget has run out.
        """
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Activity needs a kind")
        token = uuid.uuid4().hex
        session = {"kind": kind, "pid": os.getpid(), "process_identity": _owner_identity(os.getpid()),
                   "pid_namespace": _pid_namespace(), "started_at_epoch": time.time(),
                   "lease_seconds": self.lease_seconds}

        def begin(state, now):
            snapshot = self._snapshot(state)
            key = "remaining_seconds" if kind == "completion_summary" else "work_remaining_seconds"
            if kind not in _UNGATED_KINDS and snapshot[key] <= 0:
                raise BudgetExhausted(snapshot)
            state["sessions"][token] = {**session, "heartbeat_monotonic": now}

        self._transaction(begin)
        handle = _Activity(self, token, session)
        if heartbeat:
            handle.start()
        try:
            yield handle
        finally:
            handle.stop()
            try:
                self._transaction(lambda state, now: state["sessions"].pop(token, None))
            except OSError:
                pass


class _Activity:
    def __init__(self, budget: ProjectBudget, token: str, session: dict[str, Any]):
        self.budget, self.token, self.session = budget, token, session
        self._stopped = threading.Event()
        self._thread = None
        self._error = None

    def heartbeat(self):
        def touch(state, now):
            session = state["sessions"].get(self.token)
            if session is None:
                if self._stopped.is_set():
                    return
                # A shared-filesystem stall can expire a live owner; only its uncharged gap is lost.
                session = state["sessions"][self.token] = dict(self.session)
            session["heartbeat_monotonic"] = now
        self.budget._transaction(touch)

    def check_available(self, *, for_summary: bool = False):
        if self._error is not None:
            raise self._error
        return self.budget.check_available(for_summary=for_summary)

    def start(self):
        def monitor():
            while not self._stopped.wait(self.budget.heartbeat_seconds):
                try:
                    self.heartbeat()
                except Exception as exc:
                    self._error = exc
                    return
        self._thread = threading.Thread(target=monitor, name="construction-budget-heartbeat", daemon=True)
        self._thread.start()

    def stop(self):
        self._stopped.set()
        if self._thread is not None:
            self._thread.join(timeout=self.budget.heartbeat_seconds + 1)
