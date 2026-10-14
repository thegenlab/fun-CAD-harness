"""Project budget, simulation timeout, and scale-limit contract.

This is the single authoritative source for tier configurations, active-budget
accounting, simulation timeout enforcement, task/part count limits, checkpoint
save/restore, and incomplete-summary generation.  Timing logic must not be
scattered across individual simulation call sites.

Active budget tracks only deliberate work: agent thinking, mechanical design,
physical simulation execution.  It explicitly excludes infrastructure waits
(memory stalls, process hangs, queue/scheduling delays, token/session recovery,
external service recovery).  Both active-budget time and wall-clock time are
always recorded side by side.
"""
from __future__ import annotations

import json
import math
import time
import threading
import os
import re
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Tier configuration contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TierConfig:
    """Immutable, testable per-tier limits.  Values here are the contract."""
    name: str
    active_budget_seconds: float
    simulation_timeout_seconds: float
    max_tasks_per_design: int
    min_parts: int
    max_parts: int

    def __post_init__(self):
        if not math.isfinite(self.active_budget_seconds) or self.active_budget_seconds <= 0:
            raise ValueError("active_budget_seconds must be positive and finite")
        if not math.isfinite(self.simulation_timeout_seconds) or self.simulation_timeout_seconds <= 0:
            raise ValueError("simulation_timeout_seconds must be positive and finite")
        if self.max_tasks_per_design < 1:
            raise ValueError("max_tasks_per_design must be at least 1")
        if self.max_parts < 1 or self.min_parts < 0 or self.min_parts > self.max_parts:
            raise ValueError("part limits must satisfy 0 <= min_parts <= max_parts >= 1")


TIERS: dict[str, TierConfig] = {
    "light": TierConfig(
        name="light",
        active_budget_seconds=3600,       # 1 hour
        simulation_timeout_seconds=60,    # 1 minute
        max_tasks_per_design=2,
        min_parts=0,
        max_parts=10,
    ),
    "medium": TierConfig(
        name="medium",
        active_budget_seconds=7200,       # 2 hours
        simulation_timeout_seconds=180,   # 3 minutes
        max_tasks_per_design=2,
        min_parts=0,
        max_parts=15,
    ),
    "high": TierConfig(
        name="high",
        active_budget_seconds=10800,      # 3 hours
        simulation_timeout_seconds=300,   # 5 minutes
        max_tasks_per_design=5,
        min_parts=0,
        max_parts=20,
    ),
}


def recommended_limits(tier_name: str) -> tuple[int, float]:
    """Construction hours and per-simulation cap a new run of this tier gets by default.

    The distribution may recommend longer high-tier limits (execution_profile);
    the tier table itself stays as the benchmark uses it."""
    from . import execution_profile
    tier = get_tier(tier_name)
    hours, seconds = int(tier.active_budget_seconds / 3600), tier.simulation_timeout_seconds
    if tier.name == "high":
        hours = execution_profile.DEFAULT_BUDGET_HOURS or hours
        seconds = execution_profile.DEFAULT_SIMULATION_TIMEOUT_SECONDS or seconds
    return hours, seconds


def get_tier(name: str) -> TierConfig:
    """Look up a tier by name; raises ValueError for unknown tiers."""
    if name not in TIERS:
        raise ValueError(f"Unknown tier {name!r}; available: {sorted(TIERS)}")
    return TIERS[name]


GRANT_LEDGER_NAME = "budget_grants.json"
MAX_RECOVERY_GRANT_SECONDS = 3600.0


def _grant_ledger(project):
    path=Path(project)/GRANT_LEDGER_NAME
    if not path.is_file():return {"schema_version":"slim_user_budget_grants_v1","grants":[]}
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version")!="slim_user_budget_grants_v1" or not isinstance(data.get("grants"),list):
        raise ValueError("Invalid user budget grant ledger")
    ids=set();total=0.
    for grant in data["grants"]:
        identifier=grant.get("grant_id")
        seconds=grant.get("seconds")
        if not isinstance(identifier,str) or not identifier.strip() or identifier in ids:
            raise ValueError("Budget grants require unique nonempty IDs")
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=MAX_RECOVERY_GRANT_SECONDS:
            raise ValueError("Each recovery grant must be positive and at most one hour")
        if not all(isinstance(grant.get(key),str) and grant[key].strip() for key in ("reason","authorization")):
            raise ValueError("Budget grants require the explicit user authorization and reason")
        ids.add(identifier);total+=seconds
    if total>MAX_RECOVERY_GRANT_SECONDS:
        raise ValueError("This recovery authorization permits at most one additional hour per project")
    return data


def read_user_budget_grants(project):
    """Read the validated grant ledger without changing project state or time."""
    return _grant_ledger(project)


def apply_user_budget_grant(project, *, grant_id, seconds, reason, authorization, evidence_paths=()):
    """Record explicitly authorized light-project credit; never resume or reset it.

    The grant ledger is separate from measured active/wall history. A later
    ProjectBudget/restore loads it. Repeating the same grant is a no-op.
    """
    project=Path(project).resolve()
    checkpoint=json.loads((project/"budget_checkpoint.json").read_text(encoding="utf-8"))
    tier=TierConfig(**checkpoint["tier_config"]) if checkpoint.get("tier_config") else get_tier(checkpoint["tier"])
    if tier!=TIERS["light"]:raise ValueError("This recovery grant applies to the original light tier only")
    proposed={"grant_id":grant_id,"seconds":seconds,"reason":reason,"authorization":authorization,
              "evidence_paths":[str(path) for path in evidence_paths]}
    ledger=_grant_ledger(project)
    previous=next((row for row in ledger["grants"] if row["grant_id"]==grant_id),None)
    if previous is not None:
        if any(previous.get(key)!=value for key,value in proposed.items()):
            raise ValueError("Grant ID already exists with different terms")
        return previous
    if not isinstance(grant_id,str) or not grant_id.strip():raise ValueError("Grant ID is required")
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<seconds<=MAX_RECOVERY_GRANT_SECONDS:
        raise ValueError("Each recovery grant must be positive and at most one hour")
    if not all(isinstance(value,str) and value.strip() for value in (reason,authorization)):
        raise ValueError("Record the explicit user authorization and reason")
    if sum(row["seconds"] for row in ledger["grants"])+seconds>MAX_RECOVERY_GRANT_SECONDS:
        raise ValueError("This recovery authorization permits at most one additional hour per project")
    from .workflow import assert_inactive
    assert_inactive(project)
    if not checkpoint.get("paused",False):raise ValueError("Pause the project before granting recovery time")
    record={**proposed,"created_at":datetime.now(timezone.utc).isoformat()}
    from ..mechanism.io import write_json_atomic
    write_json_atomic(project/GRANT_LEDGER_NAME,{**ledger,"grants":[*ledger["grants"],record]})
    return record


# ---------------------------------------------------------------------------
# Result status vocabulary
# ---------------------------------------------------------------------------

class BudgetStatus(str, Enum):
    PASS = "PASS"
    NEEDS_REVISION = "NEEDS_REVISION"
    HARNESS_ERROR = "HARNESS_ERROR"
    SIMULATION_TIMEOUT = "SIMULATION_TIMEOUT"
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    ACTIVE_BUDGET_EXHAUSTED = "ACTIVE_BUDGET_EXHAUSTED"
    INCOMPLETE_EVIDENCE = "INCOMPLETE_EVIDENCE"
    EVIDENCE_REQUIRED = "EVIDENCE_REQUIRED"
    RESOURCE_POLICY_REQUIRED = "RESOURCE_POLICY_REQUIRED"

    def is_terminal(self) -> bool:
        return self in {
            BudgetStatus.PASS,
            BudgetStatus.ACTIVE_BUDGET_EXHAUSTED,
            BudgetStatus.INFRASTRUCTURE_FAILURE,
        }


# ---------------------------------------------------------------------------
# Active-budget tracker with pause/resume
# ---------------------------------------------------------------------------

@dataclass
class ActiveBudgetTracker:
    """Tracks active work time separately from wall-clock time.

    Call ``pause()`` when entering a non-work wait (infrastructure, recovery,
    memory stall) and ``resume()`` when real work resumes.  ``elapsed_active``
    only advances while not paused.  ``elapsed_wall`` always advances.
    """
    tier: TierConfig
    _start_wall: float = field(default_factory=time.monotonic, init=False)
    _start_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), init=False)
    _active_seconds: float = field(default=0.0, init=False)
    _active_segment_start: float = field(default=0.0, init=False)
    _paused: bool = field(default=False, init=False)
    _pause_count: int = field(default=0, init=False)
    _pause_seconds: float = field(default=0.0, init=False)
    _pause_start: float = field(default=0.0, init=False)
    _checkpoint_path: Path | None = field(default=None, init=False)
    _wall_offset: float = field(default=0.0, init=False)
    _granted_active_seconds: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._active_segment_start = self._start_wall

    @property
    def elapsed_active(self) -> float:
        if self._paused:
            return self._active_seconds
        return self._active_seconds + (time.monotonic() - self._active_segment_start)

    @property
    def elapsed_wall(self) -> float:
        return self._wall_offset + time.monotonic() - self._start_wall

    @property
    def remaining_active(self) -> float:
        return max(0.0, self.budget_limit_seconds - self.elapsed_active)

    @property
    def budget_limit_seconds(self) -> float:
        return self.tier.active_budget_seconds+self._granted_active_seconds

    @property
    def is_exhausted(self) -> bool:
        return self.elapsed_active >= self.budget_limit_seconds

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self, reason: str = ""):
        """Stop counting active time; infrastructure wait begins."""
        if self._paused:
            return
        now = time.monotonic()
        self._active_seconds += now - self._active_segment_start
        self._paused = True
        self._pause_start = now
        self._pause_count += 1

    def resume(self):
        """Resume counting active time after infrastructure wait."""
        if not self._paused:
            return
        now = time.monotonic()
        self._pause_seconds += now - self._pause_start
        self._paused = False
        self._active_segment_start = now

    def check_budget(self) -> bool:
        """Return True if active budget is still available."""
        return not self.is_exhausted

    def snapshot(self) -> dict[str, Any]:
        """Serializable state for checkpoint persistence."""
        result = {
            "schema_version": "budget_checkpoint_v1",
            "tier": self.tier.name,
            "tier_config": asdict(self.tier),
            "active_seconds": self.elapsed_active,
            "wall_seconds": self.elapsed_wall,
            "paused": self._paused,
            "pause_count": self._pause_count,
            "total_pause_seconds": self._pause_seconds + (
                (time.monotonic() - self._pause_start) if self._paused else 0.0
            ),
            "budget_limit_seconds": self.budget_limit_seconds,
            "remaining_active_seconds": self.remaining_active,
            "started_at": self._start_utc,
            "checkpoint_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._granted_active_seconds:
            result.update(base_budget_limit_seconds=self.tier.active_budget_seconds,
                          granted_active_seconds=self._granted_active_seconds,
                          grant_ledger=GRANT_LEDGER_NAME)
        return result

    def save_checkpoint(self, path: Path):
        """Persist checkpoint to disk for recovery after stalls."""
        self._checkpoint_path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        from ..mechanism.io import write_json_atomic
        write_json_atomic(path, data)
        return data

    @classmethod
    def restore(cls, path: Path, tier: TierConfig | None = None) -> "ActiveBudgetTracker":
        """Restore a tracker from a saved checkpoint.

        The restored tracker starts in paused state so recovery time is not
        counted against the active budget.  Pass ``tier`` directly when the
        checkpoint's tier name is not in the production TIERS registry.
        """
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "budget_checkpoint_v1":
            raise ValueError("Unknown checkpoint schema")
        if tier is None:
            if data.get("tier_config"):
                tier = TierConfig(**data["tier_config"])
        if tier is None:
            try:
                tier = get_tier(data["tier"])
            except ValueError:
                tier = TierConfig(
                    name=data["tier"],
                    active_budget_seconds=float(data["budget_limit_seconds"]),
                    simulation_timeout_seconds=60,
                    max_tasks_per_design=2,
                    min_parts=0,
                    max_parts=10,
                )
        tracker = cls(tier=tier)
        tracker._active_seconds = float(data["active_seconds"])
        tracker._paused = True
        tracker._pause_start = time.monotonic()
        tracker._pause_count = int(data.get("pause_count", 0)) + 1
        tracker._pause_seconds = float(data.get("total_pause_seconds", 0.0))
        tracker._checkpoint_path = path
        ledger=_grant_ledger(path.parent)
        tracker._granted_active_seconds=sum(row["seconds"] for row in ledger["grants"])
        if data.get("granted_active_seconds",0)>tracker._granted_active_seconds:
            raise ValueError("Saved budget credit is missing from its user grant ledger")
        checkpoint_at = datetime.fromisoformat(data["checkpoint_at"])
        offline = max(0.0, (datetime.now(timezone.utc)-checkpoint_at).total_seconds())
        tracker._wall_offset = float(data.get("wall_seconds", 0.0)) + offline
        tracker._start_utc = data.get("started_at", tracker._start_utc)
        tracker._pause_seconds += offline
        return tracker


# ---------------------------------------------------------------------------
# Simulation timeout wrapper
# ---------------------------------------------------------------------------

@dataclass
class SimulationTimeoutResult:
    """Outcome of a single simulation run with timeout enforcement."""
    timed_out: bool
    elapsed_seconds: float
    partial_evidence: dict[str, Any] | None = None
    log_path: Path | None = None
    error: str | None = None

    @property
    def status(self) -> BudgetStatus:
        if self.timed_out:
            return BudgetStatus.SIMULATION_TIMEOUT
        return BudgetStatus.PASS


def enforce_simulation_timeout(tier: TierConfig, actual_timeout: float | None = None) -> float:
    """Return the effective simulation timeout for a single test.

    The tier's simulation_timeout_seconds is the hard upper bound.
    ``actual_timeout`` may be smaller but never larger.
    """
    limit = tier.simulation_timeout_seconds
    if actual_timeout is not None:
        if not math.isfinite(actual_timeout) or actual_timeout <= 0:
            raise ValueError("actual_timeout must be positive and finite")
        limit = min(limit, actual_timeout)
    return limit


# ---------------------------------------------------------------------------
# Limit enforcement
# ---------------------------------------------------------------------------

def check_task_count(tests: list, tier: TierConfig) -> list[str]:
    """Validate that the number of tests does not exceed the tier limit.

    Returns a list of violation messages (empty means OK).
    Never silently drops tasks.
    """
    count = len(tests)
    if count > tier.max_tasks_per_design:
        return [
            f"Design has {count} tests but tier {tier.name!r} allows "
            f"at most {tier.max_tasks_per_design}"
        ]
    return []


def check_part_count(bodies: list, tier: TierConfig) -> list[str]:
    """Validate that the number of parts is within the tier's range.

    Returns a list of violation messages (empty means OK).
    """
    count = len(bodies)
    violations = []
    if count > tier.max_parts:
        violations.append(
            f"Design has {count} parts but tier {tier.name!r} allows "
            f"at most {tier.max_parts}"
        )
    return violations


# ---------------------------------------------------------------------------
# Incomplete summary
# ---------------------------------------------------------------------------

def build_incomplete_summary(
    *,
    tier: TierConfig,
    tracker: ActiveBudgetTracker,
    completed_tests: list[dict[str, Any]],
    pending_tests: list[str],
    issues: list[Any],
    evidence_paths: list[str],
    blockers: list[str],
) -> dict[str, Any]:
    """Generate the structured summary required when budget is exhausted."""
    snap = tracker.snapshot()
    passed = [t for t in completed_tests if t.get("physical_pass") and t.get("geometry_audit", {}).get("passed")]
    failed = [t for t in completed_tests if not t.get("physical_pass") or not t.get("geometry_audit", {}).get("passed")]
    return {
        "schema_version": "budget_incomplete_summary_v1",
        "status": BudgetStatus.ACTIVE_BUDGET_EXHAUSTED.value,
        "tier": asdict(tier),
        "timing": {
            "active_seconds": snap["active_seconds"],
            "wall_seconds": snap["wall_seconds"],
            "budget_limit_seconds": snap["budget_limit_seconds"],
            "pause_count": snap["pause_count"],
            "total_pause_seconds": snap["total_pause_seconds"],
        },
        "completed_tests": [
            {"id": t.get("test_id", "unknown"), "physical_pass": t.get("physical_pass"),
             "geometry_pass": t.get("geometry_audit", {}).get("passed"),
             "status": ("NOT_RUN" if t.get("execution_started") is False else "TIMEOUT" if t.get("execution_status")=="SIMULATION_TIMEOUT"
                        else "ERROR" if t.get("failure_status")=="HARNESS_ERROR" else "PASS" if t in passed else "FAIL"),
             **({"execution_started":t.get("execution_started"),"execution_status":t["execution_status"],"trace_file":t.get("trace_file")} if "execution_status" in t else {})}
            for t in completed_tests
        ],
        "pending_tests": pending_tests,
        "severity": "functional" if failed or pending_tests else "minor",
        "issues": issues,
        "evidence_paths": evidence_paths,
        "blockers": blockers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Checkpoint recovery helpers
# ---------------------------------------------------------------------------

MAX_RECOVERY_RETRIES = 3
BACKOFF_BASE_SECONDS = 5.0
# A provider outage (model at capacity, gateway balance, 5xx) can last many
# minutes. Retry agent turns longer; recovery waits are not active time.
PROVIDER_RECOVERY_RETRIES = 10
PROVIDER_BACKOFF_CAP_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Adjustable criteria tracking
# ---------------------------------------------------------------------------

@dataclass
class CriteriaAdjustment:
    """Record of one criteria adjustment for a non-frozen, adjustable task.

    Frozen benchmark criteria and tests must NEVER be adjusted.
    Only tasks explicitly declared ``adjustable: true`` permit adjustment.
    """
    criterion_id: str
    before: dict[str, Any]
    after: dict[str, Any]
    reason: str
    adjusted_at: str = ""
    frozen: bool = False

    def __post_init__(self):
        if self.frozen:
            raise ValueError(
                f"Cannot adjust frozen criterion {self.criterion_id!r}; "
                "frozen benchmark criteria are immutable"
            )
        if not self.adjusted_at:
            self.adjusted_at = datetime.now(timezone.utc).isoformat()
        if not self.reason:
            raise ValueError("Adjustment requires a recorded reason")
        if self.before == self.after:
            raise ValueError("No actual change in criteria")

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "before": self.before,
            "after": self.after,
            "reason": self.reason,
            "adjusted_at": self.adjusted_at,
        }


class AdjustmentLog:
    """Append-only log of criteria adjustments for a project run."""

    def __init__(self):
        self._entries: list[CriteriaAdjustment] = []

    def record(self, adjustment: CriteriaAdjustment):
        self._entries.append(adjustment)

    @property
    def entries(self) -> list[CriteriaAdjustment]:
        return list(self._entries)

    def to_list(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "schema_version": "criteria_adjustment_log_v1",
            "adjustments": self.to_list(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AdjustmentLog":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != "criteria_adjustment_log_v1":
            raise ValueError("Unknown adjustment log schema")
        log = cls()
        for entry in data.get("adjustments", []):
            adj = CriteriaAdjustment(
                criterion_id=entry["criterion_id"],
                before=entry["before"],
                after=entry["after"],
                reason=entry["reason"],
                adjusted_at=entry.get("adjusted_at", ""),
            )
            log._entries.append(adj)
        return log


def recovery_delay(attempt: int, cap: float = 60.0) -> float:
    """Exponential backoff for recovery retries, capped at 60 seconds by default."""
    return min(BACKOFF_BASE_SECONDS * (2 ** attempt), cap)


def provider_needs_external_action(error) -> bool:
    """Quota/auth without an explicit reset time must not be polled."""
    if not isinstance(error, dict):
        return False
    if error.get("needs_external_action"):
        return True
    kind = str(error.get("kind", "")).lower()
    if kind not in {"quota", "auth"}:
        return False
    return _provider_retry_delay(error) is None


def provider_outage(error, summary: str = "") -> bool:
    """A provider-side failure that usually clears by itself, e.g. a model at capacity."""
    # A used-up turn limit or an unreadable local response is not an outage.
    if not isinstance(error, dict) or provider_needs_external_action(error) or error.get("kind") in {"timeout", "response"}:
        return False
    text = str(error.get("message", summary)).lower()
    # Status codes only next to their label: messages also carry durations and
    # token counts. Gateways answer an exhausted or rotating pool balance with 403.
    return any(word in text for word in ("connection", "timed out", "temporarily", "overloaded", "at capacity",
                                         "internal server error", "bad gateway", "service unavailable",
                                         "gateway timeout", "stream disconnected")
               ) or re.search(r"\b(?:api error|status|http)\W{0,4}(?:403|5\d\d)\b", text) is not None


def _provider_retry_delay(error):
    raw=error.get("retry_after_seconds")
    if raw is None:raw=error.get("retry_after")
    if isinstance(raw,bool):return None
    try:delay=float(raw)
    except (TypeError,ValueError):return None
    return delay if math.isfinite(delay) and delay>=0 else None


def should_retry(attempt: int, error: str, limit: int = MAX_RECOVERY_RETRIES) -> bool:
    """Decide whether to retry after an infrastructure failure.

    Never retries after active-budget exhaustion or explicit termination.
    """
    if attempt >= limit:
        return False
    non_retryable = {"ACTIVE_BUDGET_EXHAUSTED", "TERMINATED", "PASS"}
    if any(keyword in error for keyword in non_retryable):
        return False
    return True


class ActiveBudgetExhausted(RuntimeError):
    """Control-flow signal; never a mechanical failure or permission request."""


class ObservedWaitRecoveryRequired(RuntimeError):
    """A nested activity lost its observable owner; do not invent a pause."""


def _observed_activity_totals(directory, started, now, *, owner_pid=None, last_seen=None):
    """Union explicit local activity intervals, retaining unfinished records."""
    import psutil
    intervals={"resource_admission":[],"physical_simulation":[]}
    issues=[]
    last_seen={} if last_seen is None else last_seen
    for path in Path(directory).glob("*.json"):
        try:
            # An atomic replacement can briefly deny a Windows reader. Retry
            # only that access error: three reads, at most 30ms of waiting.
            for read_attempt in range(3):
                try:
                    encoded=path.read_text(encoding="utf-8")
                    break
                except PermissionError:
                    if read_attempt==2:raise
                    time.sleep(.015)
            row=json.loads(encoded)
            kind=row.get("activity")
            if kind not in intervals:continue
            start=max(started,float(row["started_monotonic"]))
            if row.get("ended_monotonic") is not None:
                end=min(now,float(row["ended_monotonic"]))
            else:
                issuer=psutil.Process(int(row.get("issuer_pid") or owner_pid or 0))
                alive=issuer.is_running()
                expected=row.get("issuer_started_at_epoch")
                if expected is not None:alive=alive and abs(issuer.create_time()-float(expected))<.1
                if not alive:raise ProcessLookupError("Observation owner is no longer running")
                end=now
                last_seen[str(path)]=(start,end,kind)
            if not math.isfinite(start) or not math.isfinite(end):
                raise ValueError("Observation timestamps must be finite")
        except (OSError,ValueError,TypeError,KeyError,psutil.Error) as exc:
            issues.append(f"{path}: {exc}; external recovery is required for the unfinished observation")
            if str(path) not in last_seen:continue
            start,end,kind=last_seen[str(path)]
        if end>start:intervals[kind].append((start,end))
    def union(rows):
        total=0.;end=-math.inf
        for left,right in sorted(rows):
            total+=max(0.,right-max(left,end));end=max(end,right)
        return total
    work=union(intervals["physical_simulation"])
    # Known concurrent solver work remains active even if another child waits.
    waits=union(intervals["resource_admission"]+intervals["physical_simulation"])-work
    return max(0.,waits),work,issues


class ProjectBudget:
    """One persisted budget across initialize, validate and subsequent CLI calls.

    Worker elapsed time includes thinking and author tool use. Providers do not
    expose a trustworthy split of thinking versus internal service waits, so it
    is explicitly an upper estimate. Only observed recovery waits are excluded.
    Automatic preparation, rendering and audit are reported in wall time only.
    """

    def __init__(self, project, tier=None, tracker=None):
        self.project = Path(project)
        self.path = self.project/"budget_checkpoint.json"
        selected = get_tier(tier) if isinstance(tier, str) else tier
        self.prior_unmeasured_wall_seconds = 0.0
        self.historical_active_incomplete = False
        self.observed_work_seconds = 0.0
        self.observed_wait_seconds = 0.0
        self.unclassified_provider_seconds = 0.0
        self.construction=None
        if (self.project/"construction_budget.json").is_file():
            from .construction import budget_for,construction_requested
            self.construction=budget_for(self.project) if construction_requested(self.project) else None
        if self.construction:
            state=json.loads((self.project/"workflow.json").read_text(encoding="utf-8"))
            stored=state["request"].get("tier")
            if stored and selected is not None and selected.name!=stored:
                raise ValueError("Project budget tier is frozen; resume with its saved tier")
            selected=get_tier(stored) if stored else selected
        if self.path.is_file():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            self.prior_unmeasured_wall_seconds = saved.get("prior_unmeasured_wall_seconds", 0.0)
            self.historical_active_incomplete = saved.get("historical_active_incomplete", "accounting_basis" not in saved)
            self.observed_work_seconds = float(saved.get("observed_work_seconds", 0.0) or 0.0)
            self.observed_wait_seconds = float(saved.get("observed_wait_seconds", 0.0) or 0.0)
            self.unclassified_provider_seconds = float(saved.get("unclassified_provider_seconds", 0.0) or 0.0)
            restored = ActiveBudgetTracker.restore(self.path)
            if selected is not None and selected != restored.tier:
                raise ValueError("Project budget tier is frozen; resume with its saved tier")
            self.tracker = tracker or restored
        else:
            self.tracker = tracker or (ActiveBudgetTracker(selected) if selected and not self.construction else None)
        if self.tracker:
            self.tracker._granted_active_seconds=sum(row["seconds"] for row in _grant_ledger(self.project)["grants"])
        self.tier = self.tracker.tier if self.tracker else selected if self.construction else None
        self.stop = threading.Event()
        self.thread = None
        self.lock = threading.RLock()
        self.phase = "not_working"
        self.worker_wait_directory = None
        self.worker_observer_pid = None
        self._worker_last_seen = {}
        self.worker_observation_issues = []
        self._worker_wait_total = self._worker_work_total = 0.0
        if self.tracker:
            self.tracker.pause("outside_active_work")
            if not self.path.is_file():
                state=json.loads((self.project/"workflow.json").read_text(encoding="utf-8"))
                started=datetime.fromisoformat(state["started_at"].replace("Z", "+00:00"))
                self.prior_unmeasured_wall_seconds=max(0.0,(datetime.now(timezone.utc)-started).total_seconds()-self.tracker.elapsed_wall)
                self.tracker._wall_offset+=self.prior_unmeasured_wall_seconds
                self.tracker._start_utc=state["started_at"]
                self.historical_active_incomplete = self.prior_unmeasured_wall_seconds>1
            self.save()

    def record_observed_work(self, seconds):
        if seconds > 0:
            with self.lock:
                self.observed_work_seconds += float(seconds)

    def record_observed_wait(self, seconds):
        if seconds > 0:
            with self.lock:
                self.observed_wait_seconds += float(seconds)

    def record_unclassified_provider(self, seconds):
        if seconds > 0:
            with self.lock:
                self.unclassified_provider_seconds += float(seconds)

    def save(self):
        if self.tracker:
            with self.lock:
                self.refresh_worker_observations()
                data = self.tracker.snapshot()
                data["checkpoint_monotonic"]=time.monotonic()
                from .physical_parts import is_time_only_continuation
                part_basis = (
                    "time-only continuation retains declared tests/parts; ordinary medium 2/15 is not applied"
                    if is_time_only_continuation(self.project, self.tier)
                    else "named physical-part inventory (parts.json / declared manifest); scene.bodies is rigid-body count only"
                )
                data.update(accounting_basis="worker_elapsed_upper_estimate_plus_physics_elapsed",
                            prior_unmeasured_wall_seconds=self.prior_unmeasured_wall_seconds,
                            historical_active_incomplete=self.historical_active_incomplete,
                            part_limit_basis=part_basis,
                            active_phase=self.phase,
                            worker_wait_directory=str(self.worker_wait_directory) if self.worker_wait_directory else None,
                            worker_observer_started_monotonic=getattr(self,"_worker_observer_started",None),
                            worker_observer_pid=self.worker_observer_pid,
                            worker_observed_wait_seconds=self._worker_wait_total,
                            worker_observation_issues=list(self.worker_observation_issues),
                            observed_work_seconds=self.observed_work_seconds,
                            observed_wait_seconds=self.observed_wait_seconds,
                            unclassified_provider_seconds=self.unclassified_provider_seconds,
                            timing_overlap={
                                "observed_work_overlaps_active": True,
                                "observed_wait_excluded_from_active": True,
                                "unclassified_provider_overlaps_active_upper_estimate": True,
                                "unclassified_is_not_net_thinking": True,
                                "token_counts_are_not_seconds": True,
                            },
                            unresolved_timing="Provider-internal thinking, queue and stall time cannot be separated; hard-crash work since the last checkpoint is unknown (target interval 1s; scheduling or blocking may extend it).",
                            excluded_observed_waits="CLI downtime, explicit recovery backoff, declared resource admission wait; automatic Harness preparation/render/audit counted in wall only")
                from ..mechanism.io import write_json_atomic
                write_json_atomic(self.path, data)

    def __enter__(self):
        def checkpoint():
            while not self.stop.wait(1.0):
                self.save()
        if self.tracker:
            self.thread = threading.Thread(target=checkpoint, name="budget-checkpoint", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.tracker:
            self.tracker.pause("runner_stopped")
            self.phase = "stopped"
            self.save()

    def require_available(self):
        if self.construction:self.construction.check_available()
        if self.tracker and self.tracker.is_exhausted:
            raise ActiveBudgetExhausted("Project active budget exhausted")

    def limit(self, seconds):
        self.require_available()
        return min(seconds,self.construction.check_available()["work_remaining_seconds"]) if self.construction else min(seconds, self.tracker.remaining_active) if self.tracker else seconds

    can_observe_wait = True

    @contextmanager
    def observed_wait(self, reason):
        started=time.monotonic()
        with self.lock:
            was_active=self.tracker is not None and not self.tracker.is_paused
            if was_active:self.tracker.pause(reason)
        try:
            yield
        finally:
            self.record_observed_wait(time.monotonic()-started)
            with self.lock:
                if was_active:self.tracker.resume()
            self.save()

    def start_worker_observer(self, directory):
        with self.lock:
            self.worker_wait_directory=Path(directory)/"observed-activities"/uuid.uuid4().hex
            self.worker_wait_directory.mkdir(parents=True)
            self._worker_observer_started=time.monotonic()
            self._worker_wait_total=self._worker_work_total=0.0
            self.worker_observer_pid=None
            self._worker_last_seen={}
            self.worker_observation_issues=[]
            self.save()

    def refresh_worker_observations(self):
        if self.worker_wait_directory is None:return 0.0
        with self.lock:
            waits,work,issues=_observed_activity_totals(self.worker_wait_directory,
                self._worker_observer_started,time.monotonic(),owner_pid=self.worker_observer_pid,
                last_seen=self._worker_last_seen)
            self.worker_observation_issues=issues
            delta=max(0.,waits-self._worker_wait_total)
            if self.tracker:
                self.tracker._active_seconds-=delta
                self.tracker._pause_seconds+=delta
            self.observed_wait_seconds+=delta
            self.observed_work_seconds+=max(0.,work-self._worker_work_total)
            self._worker_wait_total=waits;self._worker_work_total=work
        return waits

    def author_limit(self, seconds):
        self.require_available()
        if not self.tracker:return seconds
        from .test_coverage import author_allowance, FirstPassReserveReached
        available=author_allowance(self.project,self.tier,self.tracker.remaining_active)
        if available<=0:
            raise FirstPassReserveReached("Remaining active time is reserved for the first attempt of untested tasks")
        return min(seconds,available)

    def worker_remaining(self, call_limit, call_started):
        waits=self.refresh_worker_observations()
        if self.worker_observation_issues:
            raise ObservedWaitRecoveryRequired("; ".join(self.worker_observation_issues))
        from .test_coverage import author_allowance
        available=author_allowance(self.project,self.tier,self.tracker.remaining_active)
        return min(available,call_limit-max(0.,time.monotonic()-call_started-waits))

    @contextmanager
    def active(self, phase):
        self.require_available()
        started = time.monotonic()
        if self.tracker:
            self.phase = phase
            self.tracker.resume()
            self.save()
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            if phase == "physical_simulation":
                self.record_observed_work(elapsed)
            if self.tracker:
                self.tracker.pause("outside_active_work")
                self.phase = "not_working"
                self.save()

    def annotate(self, result):
        if self.construction:
            snap=self.construction.snapshot()
            result["construction_budget"]=snap
            result["budget_timing"]={**snap,"budget_limit_seconds":snap["total_seconds"],"remaining_active_seconds":snap["work_remaining_seconds"],"accounting_basis":snap["accounting"]}
            if self.tier:result["tier"]=asdict(self.tier)
        if self.tracker:
            self.save()
            result["budget_timing"] = json.loads(self.path.read_text(encoding="utf-8"))
            result["tier"] = asdict(self.tier)
            from .test_coverage import reserve_seconds
            result["first_pass_reserved_seconds"]=reserve_seconds(self.project,self.tier)
        return result

    def incomplete(self, *, issues=()):
        """Keep existing per-test evidence; no solver-state resume is implied."""
        state = json.loads((self.project/"workflow.json").read_text(encoding="utf-8"))
        run = self.project/"runs"/f"r{int(state['revision']):02d}"
        scene_path = run/"design"/"scene.json"
        prior_path = run/"result.json"
        prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.is_file() else {}
        if self.construction:
            from .test_coverage import planned_tests,coverage_summary
            from ..mechanism.io import write_json_atomic
            tests=planned_tests(self.project) or []
            rows=[{**row,"test_id":row.get("test_id",row.get("id"))} for row in prior.get("completed_tests",prior.get("tests",[]))]
            seen={row.get("test_id",row.get("id")) for row in rows}
            coverage=coverage_summary(tests,rows,[test["id"] for test in tests if test["id"] not in seen])
            result=self.annotate({**prior,"status":"BUDGET_EXHAUSTED","complete_instance_pass":False,"eligible_pass":False,
                "issues":[*prior.get("issues",[]),*issues],"completed_tests":rows,"test_coverage":coverage,
                "pending_tests":coverage["pending_completion_test_ids"],"unrun_tests":coverage["not_run_test_ids"]})
            state.update(phase="budget_exhausted",stop_reason="BUDGET_EXHAUSTED",complete_instance_pass=False,active_turn={},active_test=None)
            write_json_atomic(self.project/"workflow.json",state);write_json_atomic(prior_path,result)
            return result
        scene = json.loads(scene_path.read_text(encoding="utf-8")) if scene_path.is_file() else {}
        completed = prior.get("completed_tests", prior.get("tests", []))
        if not completed and scene_path.is_file():
            from .trace_storage import file_digest
            for bundle_path in sorted((run/"attempts").glob("a*/validated-physics.json"),reverse=True):
                bundle=json.loads(bundle_path.read_text(encoding="utf-8"))
                bindings=bundle.get("source_bindings",{})
                if bindings.get(str(scene_path))!=file_digest(scene_path):continue
                if bundle.get("criteria_sha256")!=file_digest(self.project/"criteria.json"):continue
                if any(not Path(p).is_file() or file_digest(Path(p))!=expected for p,expected in bindings.items()):continue
                completed=bundle.get("tests",[])
                break
        # Persisted completed-tests summaries may already use the public id key.
        completed = [{**row, "test_id":row.get("test_id",row.get("id")),
                      "geometry_audit":row.get("geometry_audit",{"passed":row.get("geometry_pass")})}
                     for row in completed]
        done = {row.get("test_id", row.get("id")) for row in completed if "execution_started" not in row or row.get("trace_file")}
        pending = [t["id"] for t in scene.get("tests", []) if t["id"] not in done]
        evidence = [str(p) for p in (run/"attempts").glob("a*/tests/*/physical.trace*")]
        evidence += [str(p) for p in (run/"attempts").glob("a*/*.png")]
        result = build_incomplete_summary(tier=self.tier, tracker=self.tracker,
            completed_tests=completed, pending_tests=pending,
            issues=[*prior.get("issues", []), *issues], evidence_paths=evidence,
            blockers=[] if scene else ["Design has not supplied a complete scene"])
        if prior.get("test_coverage"):result["test_coverage"]=prior["test_coverage"]
        result["completion"] = {"completed_test_count":len(done), "declared_test_count":len(scene.get("tests", [])),
                                "whole_machine_pass":False, "functional_issues":prior.get("issues", []),
                                "minor_issues":[], "severity_basis":"Unverified function remains a blocker; issue severity requires evidence, not an automatic success label"}
        result.update(schema_version="slim_harness_result_v1", revision=int(state["revision"]))
        self.annotate(result)
        state.update(phase="budget_exhausted", stop_reason="ACTIVE_BUDGET_EXHAUSTED",
                     complete_instance_pass=False, active_turn={}, active_test=None)
        from ..mechanism.io import write_json_atomic
        write_json_atomic(self.project/"workflow.json", state)
        write_json_atomic(run/"result.json", result)
        timing=result["budget_timing"]
        lines=["# Project budget closeout", "", "Status: ACTIVE_BUDGET_EXHAUSTED; whole-machine PASS: false.", "",
               f"Active estimate: {timing['active_seconds']:.3f} s / {timing['budget_limit_seconds']:g} s; total wall: {timing['wall_seconds']:.3f} s.",
               "Worker time is an upper estimate; internal queue/stall versus thinking is unresolved. Explicit recovery and offline waits are excluded.",
               f"Historical active incomplete: {timing['historical_active_incomplete']}.", "",
               f"Completed tests: {len(completed)}; pending: {', '.join(pending) or 'none declared / see blockers'}.", ""]
        lines += [f"- {row['id']}: physical={row['physical_pass']}, geometry={row['geometry_pass']}" for row in result["completed_tests"]]
        lines += ["", "Issues and blockers:", "", *["- "+json.dumps(value,ensure_ascii=False) for value in [*result["issues"],*result["blockers"]]],
                  "", "Evidence (trace prefixes are not solver continuation):", "", *[f"- [{Path(p).name}]({Path(p).as_posix()})" for p in evidence]]
        (run/"budget-summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        return result

    def worker(self, worker):
        return BudgetedWorker(worker, self) if self.tracker else worker


class BudgetedWorker:
    """Bound the real worker process and retry only observed transient failures."""
    def __init__(self, worker, budget):
        self.worker = worker
        self.budget = budget

    def __getattr__(self, name):
        return getattr(self.worker, name)

    def run_turn(self, workspace, prompt, output_dir, timeout_seconds, **kwargs):
        output_dir = Path(output_dir)
        tier=self.budget.tier
        total_limit=self.budget.tracker.budget_limit_seconds
        grant_note=(f" Base tier remains {tier.name}; {self.budget.tracker._granted_active_seconds:g}s is separately user-authorized recovery credit."
                    if self.budget.tracker._granted_active_seconds else "")
        from .physical_parts import is_time_only_continuation
        if is_time_only_continuation(self.budget.project, tier):
            prompt += (f"\nPROJECT BUDGET: {total_limit:g} active seconds total; "
                       f"each physical simulation at most {tier.simulation_timeout_seconds:g}s wall. "
                       +grant_note+
                       "This is a time-only continuation: retain every original test and physical part. "
                       "Do not shrink the design to an ordinary medium 2-test / 15-part cap.")
        else:
            usual={"light":"up to 10", "medium":"usually 10-15", "high":"usually 15-20"}.get(tier.name,str(tier.max_parts))
            prompt += (f"\nPROJECT BUDGET: {total_limit:g} active seconds total; "
                       f"each physical simulation at most {tier.simulation_timeout_seconds:g}s wall; "
                       f"at most {tier.max_tasks_per_design} declared tests and {tier.max_parts} parts ({usual}). "
                       "The part range describes intended scale, not a minimum requiring filler parts. "
                       "Never remove required tests or change frozen criteria to fit a limit; report a scope conflict."+grant_note)
        from .test_coverage import reserve_seconds
        prompt += (f"\nFIRST TEST PASS: {reserve_seconds(self.budget.project,tier):g} active seconds are reserved for untested tasks. "
                   "Deliver a runnable scene before author time runs out. Run every declared task once before repeating a failed development test. "
                   "Do not use this reserved time for cosmetic work or repeated repairs; the Harness will evaluate the latest available scene.")
        attempt = 0
        while True:
            timeout = self.budget.author_limit(timeout_seconds)
            directory = output_dir if attempt == 0 else output_dir/f"recovery-{attempt}"
            from ..cold_start.worker import CodexCliWorker
            observable=isinstance(self.worker,CodexCliWorker)
            if observable:
                self.budget.start_worker_observer(directory)
                self.worker.budget_controller=self.budget
            try:
                with self.budget.active("worker_elapsed_upper_estimate"):
                    worker_started=time.monotonic()
                    try:
                        result = self.worker.run_turn(workspace, prompt, directory, timeout, **kwargs)
                    finally:
                        worker_elapsed=time.monotonic()-worker_started
                        if observable:self.budget.refresh_worker_observations()
            finally:
                if observable:
                    self.worker.budget_controller=None
                    self.budget.worker_wait_directory=None
                    self.budget.save()
            elapsed = result.get("elapsed_seconds")
            if elapsed is None:
                segments_path = directory/"timing.segments.json"
                if segments_path.is_file():
                    try:
                        elapsed = json.loads(segments_path.read_text(encoding="utf-8")).get("unclassified_provider_seconds")
                    except (OSError, ValueError):
                        elapsed = None
            if elapsed is None:elapsed=worker_elapsed
            known=self.budget._worker_wait_total+self.budget._worker_work_total if observable else 0.0
            self.budget.record_unclassified_provider(max(0.,float(elapsed or 0.0)-known))
            if self.budget.tracker.is_exhausted:
                raise ActiveBudgetExhausted("Worker reached remaining project budget; scoped worker stopped")
            from .test_coverage import author_allowance, FirstPassReserveReached
            if result.get("status") != "ready" and author_allowance(self.budget.project,tier,self.budget.tracker.remaining_active)<=0:
                raise FirstPassReserveReached("Author time ended; preserve the remaining budget for first test coverage",getattr(self.worker,"session_id",None))
            if result.get("status") != "provider_error":
                return result
            error = result.get("provider_error", {})
            if provider_needs_external_action(error):
                return {**result, "recovery_attempts":attempt, "recoverable":True, "needs_external_action":True}
            text = str(error.get("message", result.get("summary", ""))).lower()
            explicit_delay=_provider_retry_delay(error)
            outage = provider_outage(error, result.get("summary", ""))
            transient = outage or explicit_delay is not None or error.get("kind") in {"timeout", "launch"}
            # An outage only waits; a timed-out turn retries active work.
            if not transient or not should_retry(attempt, text, PROVIDER_RECOVERY_RETRIES if outage else MAX_RECOVERY_RETRIES):
                return {**result, "recovery_attempts":attempt, "recoverable":True}
            delay = explicit_delay if explicit_delay is not None else recovery_delay(attempt, PROVIDER_BACKOFF_CAP_SECONDS)
            from ..mechanism.io import write_json_atomic
            state_path = self.budget.project/"workflow.json"
            state = getattr(self.budget,"state",None)
            if state is None:state=json.loads(state_path.read_text(encoding="utf-8"))
            state.update(phase="recovery_wait", stop_reason="", recovery_attempt=attempt+1,
                         recovery_delay_seconds=delay,
                         recovery_from_phase=state.get("recovery_from_phase") if state.get("phase")=="recovery_wait" else state.get("phase"))
            write_json_atomic(state_path, state)
            write_json_atomic(output_dir/f"recovery-{attempt+1}.json", {
                "status":"recovery_wait", "delay_seconds":delay, "error":error,
                "prior_turn_directory":str(directory)})
            self.budget.save()
            with self.budget.observed_wait("provider_recovery"):
                wait_until=time.monotonic()+delay
                while True:
                    if self.budget.stop.is_set():
                        raise ObservedWaitRecoveryRequired("Provider recovery wait was explicitly stopped")
                    remaining_wait=wait_until-time.monotonic()
                    if remaining_wait<=0:break
                    time.sleep(min(1.,remaining_wait))
            state.update(phase=state.pop("recovery_from_phase"))
            write_json_atomic(state_path,state)
            attempt += 1


@contextmanager
def physics_budget_session(project, *, workspace=None):
    """Own standalone accounting or publish nested observations to the owner."""
    if project is not None and (Path(project)/"budget_checkpoint.json").is_file():
        import psutil
        runtime_path=Path(project)/"workflow_runtime.json"
        runtime=json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
        if runtime.get("status")=="running":
            try:
                owner=psutil.Process(int(runtime["pid"]))
                alive=owner.is_running() and abs(owner.create_time()-float(runtime["process_started_at_epoch"]))<.1
            except (psutil.NoSuchProcess,KeyError,TypeError,ValueError):alive=False
            if alive:
                if owner.pid not in {p.pid for p in psutil.Process().parents()} | {os.getpid()}:
                    raise RuntimeError("Project has another active runner; preserve its scoped budget")
                yield NestedPhysicsBudget(Path(project))
                return
    with ProjectBudget(project or workspace or Path.cwd()) as budget:
        yield budget


class NestedPhysicsBudget:
    """Observations only; the parent is the sole writer of project counters."""
    def __init__(self,project):
        self.project=project
        self.path=project/"budget_checkpoint.json"
        checkpoint=json.loads(self.path.read_text(encoding="utf-8"))
        self.tier=ActiveBudgetTracker.restore(self.path).tier
        value=checkpoint.get("worker_wait_directory")
        self.directory=Path(value) if value else None
        self.can_observe_wait=self.directory is not None and all(
            checkpoint.get(key) is not None for key in (
                "checkpoint_monotonic","worker_observer_started_monotonic","worker_observed_wait_seconds"))
        self.stop=threading.Event()
        self.tracker=None

    def limit(self,seconds):
        checkpoint=json.loads(self.path.read_text(encoding="utf-8"))
        remaining=float(checkpoint["remaining_active_seconds"])
        anchor=checkpoint.get("checkpoint_monotonic")
        if anchor is not None and not checkpoint["paused"]:
            now=time.monotonic()
            elapsed=max(0.,now-float(anchor))
            if self.directory is not None:
                waits,_,issues=_observed_activity_totals(self.directory,
                    float(checkpoint["worker_observer_started_monotonic"]),now,
                    owner_pid=checkpoint.get("worker_observer_pid"))
                if issues:raise ObservedWaitRecoveryRequired("; ".join(issues))
                elapsed=max(0.,elapsed-max(0.,waits-float(checkpoint["worker_observed_wait_seconds"])))
            remaining-=elapsed
        if remaining<=0:raise ActiveBudgetExhausted("Project active budget exhausted")
        return min(seconds,self.tier.simulation_timeout_seconds,remaining)

    @contextmanager
    def _observation(self,kind):
        if self.directory is None:
            yield
            return
        from ..mechanism.io import write_json_atomic
        import psutil
        path=self.directory/f"{os.getpid()}-{uuid.uuid4().hex}.json"
        record={"activity":kind,"started_monotonic":time.monotonic(),"issuer_pid":os.getpid(),
                "issuer_started_at_epoch":psutil.Process().create_time()}
        write_json_atomic(path,record)
        try:yield
        finally:
            record["ended_monotonic"]=time.monotonic()
            write_json_atomic(path,record)

    def observed_wait(self,reason):return self._observation(reason)
    def active(self,phase):return self._observation(phase)


@contextmanager
def draft_simulation_budget(project, default_timeout=600):
    """Nested author simulation shares the parent's accounting, without double charge."""
    if project is None or not (Path(project)/"budget_checkpoint.json").is_file():
        yield default_timeout
        return
    project=Path(project)
    runtime_path=project/"workflow_runtime.json"
    runtime=json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.is_file() else {}
    if runtime.get("status")=="running":
        import os
        import psutil
        try:
            owner=psutil.Process(int(runtime["pid"]))
            alive=owner.is_running() and abs(owner.create_time()-float(runtime["process_started_at_epoch"]))<.1
        except (psutil.NoSuchProcess, KeyError, ValueError, TypeError):
            alive=False
        if alive:
            ancestors={p.pid for p in psutil.Process().parents()} | {os.getpid()}
            if owner.pid not in ancestors:
                raise RuntimeError("Project has another active runner; preserve its scoped budget")
            tracker=ActiveBudgetTracker.restore(project/"budget_checkpoint.json")
            # Parent heartbeat may lag by <=1s; its own remaining-time deadline
            # still contains this nested process and owns the final hard stop.
            remaining=tracker.remaining_active
            if remaining<=0: raise ActiveBudgetExhausted("Project active budget exhausted")
            yield min(default_timeout,tracker.tier.simulation_timeout_seconds,remaining)
            return
    with ProjectBudget(project) as budget:
        try:
            limit=budget.limit(min(default_timeout,budget.tier.simulation_timeout_seconds))
            with budget.active("physical_simulation"):
                yield limit
        except Exception as exc:
            from ..runtime.bounded_subprocess import BoundedSubprocessError
            exhausted=isinstance(exc,ActiveBudgetExhausted) or (
                isinstance(exc,BoundedSubprocessError) and exc.code=="BOUNDED_SUBPROCESS_TIMEOUT"
                and budget.tracker.is_exhausted)
            if exhausted:
                budget.incomplete(issues=["Standalone development simulation exhausted the project active budget; files retained, solver continuation not provided"])
                raise ActiveBudgetExhausted("Project active budget exhausted during development simulation") from exc
            raise
