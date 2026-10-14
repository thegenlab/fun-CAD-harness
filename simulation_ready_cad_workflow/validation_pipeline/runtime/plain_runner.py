from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from .bounded_capture import BoundedPipeCapture, atomic_write_bytes
from .child_environment import (
    MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS,
    update_environment_case_insensitively,
    windows_environment_utf16_units,
)
from .models import GIB, JobKind, JobSpec, JobStatus, ManifestError, QueueManifest
from .process_identity import current_process_identity, process_identity_matches
from .system_probe import SystemProbe, SystemSnapshot


MAX_STATE_BYTES = 16 * 1024 * 1024
class PlainQueueRunError(RuntimeError):
    """A plain physical queue cannot be started or completed safely."""


def kill_recorded_process_group(pid: int) -> None:
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=2.0,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            return


def kill_orphan_jobs_from_state(
    state: Mapping[str, Any], *, probe: SystemProbe | None = None
) -> None:
    jobs = state.get("jobs")
    if not isinstance(jobs, Mapping):
        return
    for record in jobs.values():
        if not isinstance(record, Mapping):
            continue
        identity = {
            "pid": record.get("pid"),
            "create_time": record.get("process_create_time"),
        }
        if not process_identity_matches(identity, probe=probe):
            continue
        kill_recorded_process_group(int(identity["pid"]))


def reconcile_queue_directory(
    queue_dir: Path, *, probe: SystemProbe | None = None
) -> None:
    """Kill leftover live jobs in one queue directory and mark it orphaned."""

    from ..contracts.durable_attempt import (
        finish_attempt,
        queue_attempt_path,
        read_attempt_document,
        write_attempt_document,
    )

    attempt_path = queue_attempt_path(queue_dir)
    previous = read_attempt_document(attempt_path)
    previous_state: dict[str, Any] | None = None
    state_path = Path(queue_dir) / "queue_state.json"
    if state_path.is_file() and not state_path.is_symlink():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            loaded = None
        if isinstance(loaded, dict):
            previous_state = loaded
    if previous_state is not None:
        kill_orphan_jobs_from_state(previous_state, probe=probe)
    if isinstance(previous, dict) and previous.get("status") == "RUNNING":
        write_attempt_document(
            attempt_path,
            finish_attempt(previous, status="ORPHANED"),
        )


def reconcile_run_queue_orphans(
    run: Path, *, probe: SystemProbe | None = None
) -> None:
    """Scan every ``runtime_queue/attempt-*`` left by a prior resume."""

    root = Path(run) / "runtime_queue"
    if not root.is_dir() or root.is_symlink():
        return
    for attempt_dir in sorted(root.glob("attempt-*")):
        if not attempt_dir.is_dir() or attempt_dir.is_symlink():
            continue
        for queue_dir in attempt_dir.iterdir():
            if queue_dir.is_dir() and not queue_dir.is_symlink():
                reconcile_queue_directory(queue_dir, probe=probe)


@dataclass
class _ActiveJob:
    job: JobSpec
    process: subprocess.Popen[bytes]
    captures: tuple[BoundedPipeCapture, BoundedPipeCapture]
    stdout_path: Path
    stderr_path: Path
    temporary_directory: Path
    started_at: str
    started_monotonic: float
    preflight_snapshot: SystemSnapshot
    reservation_gib: float
    peak_tree_rss_bytes: int = 0
    critical_samples: int = 0
    last_snapshot: SystemSnapshot | None = None
    capture_metrics: dict[str, dict[str, object]] = field(default_factory=dict)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_regular_file_sha256(path: Path) -> str:
    before = path.stat()
    if path.is_symlink() or not path.is_file():
        raise OSError(f"not a non-symlink regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    after = path.stat()
    before_identity = (
        int(before.st_size),
        int(before.st_mtime_ns),
        getattr(before, "st_ino", None),
    )
    after_identity = (
        int(after.st_size),
        int(after.st_mtime_ns),
        getattr(after, "st_ino", None),
    )
    if before_identity != after_identity:
        raise OSError(f"file changed while being hashed: {path}")
    return digest.hexdigest()


class PlainQueueRunner:
    """Resource-aware runner for ordinary physical jobs.

    This execution path deliberately has no authority, signing, receipt,
    launch-credential, generation, candidate-regression, or migration branch.
    """

    def __init__(
        self,
        manifest: QueueManifest,
        *,
        workspace_root: str | Path,
        run_dir: str | Path,
        probe: SystemProbe | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(manifest, QueueManifest):
            raise ManifestError("manifest must be a QueueManifest")
        manifest.validate()
        self.manifest = manifest
        self.workspace_root = Path(workspace_root).resolve()
        self.run_dir = Path(run_dir).resolve()
        if not self.workspace_root.is_dir():
            raise PlainQueueRunError(
                f"workspace root is not a directory: {self.workspace_root}"
            )
        self._require_inside_workspace(self.run_dir, "run_dir")
        self.state_path = self.run_dir / "queue_state.json"
        self.probe = probe or SystemProbe(disk_path=self.workspace_root)
        self.sleep = sleep
        self.monotonic = monotonic
        self._state_lock = threading.RLock()
        self._active: dict[str, _ActiveJob] = {}
        self._resolved_executables: dict[str, Path] = {}
        self._attempt: dict[str, Any] | None = None
        self._previous_state: dict[str, Any] | None = None
        self._validate_jobs()

    def run(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._reconcile_previous_attempt()
        state = self._new_state()
        self._write_state(state)
        last_budget_sample = self.monotonic()
        active_wall_seconds = 0.0
        pending = list(self.manifest.jobs)
        resource_wait_started: dict[str, float] = {}
        idle_admission_started: float | None = None
        first_hard_failure = False

        try:
            state["status"] = "RUNNING"
            state["started_at"] = _utc_now()
            self._event(state, "QUEUE_STARTED", "Plain physical queue started")
            if self.manifest.queue_policy.reuse_successful:
                pending = self._reuse_completed_outputs(state, pending)
            self._write_state(state)

            while pending or self._active:
                now = self.monotonic()
                if self._active:
                    active_wall_seconds += max(0.0, now - last_budget_sample)
                last_budget_sample = now
                remaining_wall_seconds = max(
                    0.0,
                    self.manifest.queue_policy.max_wall_time_seconds
                    - active_wall_seconds,
                )
                if remaining_wall_seconds <= 0.0:
                    first_hard_failure = True
                    for active in list(self._active.values()):
                        self._terminate_process_group(active.process)
                        self._finish_active(
                            state,
                            active,
                            JobStatus.TIMED_OUT,
                            "Queue wall-time limit expired",
                        )
                    self._cancel_pending(
                        state,
                        pending,
                        "Cancelled after queue wall-time limit expired",
                    )
                    pending.clear()
                    break

                failure_status = self._poll_active(state)
                if failure_status is not None:
                    if (
                        failure_status is JobStatus.TIMED_OUT
                        and self._requeue_timed_out_jobs(state, pending)
                    ):
                        pending = self._ordered_pending(pending)
                        resource_wait_started.clear()
                        idle_admission_started = None
                        continue
                    resource_pressure = (
                        failure_status is JobStatus.RESOURCE_BLOCKED
                    )
                    if not resource_pressure:
                        first_hard_failure = True
                    self._cancel_all_active(
                        state,
                        (
                            "Cancelled because another lane became "
                            "RESOURCE_BLOCKED"
                            if resource_pressure
                            else "Cancelled after the first lane failure"
                        ),
                    )
                    self._cancel_pending(
                        state,
                        pending,
                        (
                            "Cancelled before start because another lane "
                            "became RESOURCE_BLOCKED"
                            if resource_pressure
                            else "Cancelled before start after the first "
                            "lane failure"
                        ),
                    )
                    pending.clear()
                    break

                slots = (
                    self.manifest.queue_policy.max_parallel_jobs
                    - len(self._active)
                )
                admitted = False
                snapshot: SystemSnapshot | None = None
                blocked_reasons: list[str] = []
                blocked_required = 0.0
                if pending and slots > 0:
                    snapshot = self._sample_for_admission()
                    active_reservation = sum(
                        item.reservation_gib for item in self._active.values()
                    )
                    for job in list(pending):
                        if slots <= 0:
                            break
                        reasons, required = self._resource_block_reasons(
                            job,
                            snapshot=snapshot,
                            active_reservation_gib=active_reservation,
                        )
                        if reasons:
                            if not blocked_reasons:
                                blocked_reasons = reasons
                                blocked_required = required
                            started = resource_wait_started.setdefault(
                                job.job_id,
                                self.monotonic(),
                            )
                            elapsed = max(0.0, self.monotonic() - started)
                            record = state["jobs"][job.job_id]
                            record.update(
                                {
                                    "status": "WAITING_FOR_RESOURCE",
                                    "lane_phase": "WAITING_FOR_RESOURCE",
                                    "reason": "; ".join(reasons),
                                    "resource_wait_seconds": round(elapsed, 6),
                                    "required_available_memory_gib": round(
                                        required, 3
                                    ),
                                    "last_system_snapshot": snapshot.to_dict(),
                                }
                            )
                            if elapsed >= (
                                self.manifest.resource_policy
                                .max_resource_wait_seconds
                            ):
                                reason = (
                                    "Resource wait window reached: "
                                    + "; ".join(reasons)
                                )
                                resource_wait_started[job.job_id] = (
                                    self.monotonic()
                                )
                                self._event(
                                    state,
                                    "JOB_RESOURCE_WAIT_CONTINUES",
                                    reason,
                                    job_id=job.job_id,
                                )
                                self._write_state(state)
                                break
                            continue

                        resource_wait_started.pop(job.job_id, None)
                        try:
                            active = self._start_job(state, job, snapshot)
                        except Exception as exc:
                            record = state["jobs"][job.job_id]
                            record.update(
                                {
                                    "status": JobStatus.FAILED.value,
                                    "lane_phase": JobStatus.FAILED.value,
                                    "reason": "Could not start job: "
                                    f"{type(exc).__name__}: {exc}",
                                    "finished_at": _utc_now(),
                                }
                            )
                            self._event(
                                state,
                                "JOB_FAILED",
                                str(record["reason"]),
                                job_id=job.job_id,
                            )
                            pending.remove(job)
                            first_hard_failure = True
                            self._cancel_all_active(
                                state,
                                "Cancelled after the first lane failure",
                            )
                            self._cancel_pending(
                                state,
                                pending,
                                "Cancelled before start after the first lane "
                                "failure",
                            )
                            pending.clear()
                            self._write_state(state)
                            break
                        self._active[job.job_id] = active
                        pending.remove(job)
                        active_reservation += active.reservation_gib
                        slots -= 1
                        admitted = True
                        idle_admission_started = None

                    self._write_state(state)

                if pending and not self._active:
                    if snapshot is None:
                        snapshot = self._sample_for_admission()
                        blocked_reasons, blocked_required = (
                            self._resource_block_reasons(
                                pending[0],
                                snapshot=snapshot,
                                active_reservation_gib=0.0,
                            )
                        )
                    if idle_admission_started is None:
                        idle_admission_started = now
                    idle_elapsed = max(
                        0.0, self.monotonic() - idle_admission_started
                    )
                    wait_budget = (
                        self.manifest.resource_policy.max_resource_wait_seconds
                    )
                    if blocked_reasons and idle_elapsed >= wait_budget:
                        reason = (
                            "Idle resource admission wait exceeded "
                            f"{wait_budget:.3f} seconds: "
                            + "; ".join(blocked_reasons)
                        )
                        self._resource_block_pending(
                            state,
                            pending,
                            snapshot=snapshot,
                            required_available_memory_gib=blocked_required,
                            reason=reason,
                            resource_wait_seconds=idle_elapsed,
                        )
                        pending.clear()
                        break
                else:
                    idle_admission_started = None

                if not pending and not self._active:
                    break
                if self._active or (pending and not admitted):
                    self.sleep(
                        min(
                            self.manifest.resource_policy.resource_poll_seconds,
                            remaining_wall_seconds,
                        )
                    )
        except KeyboardInterrupt:
            self._cancel_all_active(state, "Interrupted by controlling process")
            self._cancel_pending(state, pending, "Queue was interrupted")
            state["status"] = "INTERRUPTED"
            state["finished_at"] = _utc_now()
            self._write_state(state)
            raise
        except BaseException:
            try:
                self._cancel_all_active(state, "Runner failed internally")
                self._cancel_pending(
                    state,
                    pending,
                    "Cancelled before start because the runner failed",
                )
                pending.clear()
                state["status"] = "FAILED"
                state["finished_at"] = _utc_now()
                self._write_state(state)
            except Exception:
                pass
            raise
        finally:
            if self._active:
                self._cancel_all_active(state, "Runner exited before completion")

        state["status"] = self._queue_status(state, first_hard_failure)
        state["finished_at"] = _utc_now()
        state["parallelism"]["active_job_ids"] = []
        state["parallelism"]["waiting_job_ids"] = []
        self._event(
            state,
            "QUEUE_FINISHED",
            f"Plain physical queue finished as {state['status']}",
        )
        self._write_state(state)
        self._finish_queue_attempt(state)
        return state

    def _reconcile_previous_attempt(self) -> None:
        from ..contracts.durable_attempt import (
            QUEUE_ATTEMPT_SCHEMA,
            finish_attempt,
            new_attempt_document,
            queue_attempt_path,
            read_attempt_document,
            write_attempt_document,
        )

        attempt_path = queue_attempt_path(self.run_dir)
        previous = read_attempt_document(attempt_path)
        previous_state: dict[str, Any] | None = None
        if self.state_path.is_file() and not self.state_path.is_symlink():
            try:
                loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                loaded = None
            if isinstance(loaded, dict):
                previous_state = loaded
        self._previous_state = previous_state
        if (
            isinstance(previous, dict)
            and previous.get("status") == "RUNNING"
        ):
            identity = {
                "pid": previous.get("pid"),
                "create_time": previous.get("process_create_time"),
            }
            same_digest = (
                previous.get("input_digest") == self.manifest.fingerprint
            )
            if process_identity_matches(identity, probe=self.probe) and same_digest:
                raise PlainQueueRunError(
                    "queue attempt is still owned by a live runner"
                )
            if previous_state is not None:
                kill_orphan_jobs_from_state(previous_state, probe=self.probe)
            write_attempt_document(
                attempt_path,
                finish_attempt(previous, status="ORPHANED"),
            )
        elif previous_state is not None:
            kill_orphan_jobs_from_state(previous_state, probe=self.probe)
        identity = current_process_identity(self.probe)
        self._attempt = new_attempt_document(
            schema_version=QUEUE_ATTEMPT_SCHEMA,
            input_digest=self.manifest.fingerprint,
            stage="queue",
            process_identity=identity,
            extra={"manifest_fingerprint": self.manifest.fingerprint},
        )
        write_attempt_document(attempt_path, self._attempt)

    def _kill_orphan_jobs(self, previous_state: dict[str, Any]) -> None:
        kill_orphan_jobs_from_state(previous_state, probe=self.probe)

    def _kill_recorded_process_group(self, pid: int) -> None:
        kill_recorded_process_group(pid)

    def _finish_queue_attempt(self, state: dict[str, Any]) -> None:
        from ..contracts.durable_attempt import (
            finish_attempt,
            queue_attempt_path,
            write_attempt_document,
        )

        if not isinstance(self._attempt, dict):
            return
        status = str(state.get("status") or "FAILED")
        mapped = {
            "SUCCEEDED": "SUCCEEDED",
            "RESOURCE_BLOCKED": "RESOURCE_BLOCKED",
            "FAILED": "FAILED",
            "INTERRUPTED": "INTERRUPTED",
            "INCOMPLETE": "FAILED",
        }.get(status, "FAILED")
        write_attempt_document(
            queue_attempt_path(self.run_dir),
            finish_attempt(
                self._attempt,
                status=mapped,
                terminal_receipt={"queue_status": status},
            ),
        )

    def _validate_jobs(self) -> None:
        runner_jobs_root = (self.run_dir / "jobs").resolve()
        for job in self.manifest.jobs:
            self._verify_lane_source_binding(job)
            environment = self._build_environment(job, temp_directory=None)
            executable = shutil.which(
                job.command[0],
                path=environment.get("PATH"),
            )
            if executable is None:
                raise ManifestError(
                    f"job {job.job_id!r} executable is unavailable: "
                    f"{job.command[0]!r}"
                )
            resolved_executable = Path(executable).resolve(strict=True)
            if not resolved_executable.is_file():
                raise ManifestError(
                    f"job {job.job_id!r} executable is not a regular file: "
                    f"{resolved_executable}"
                )
            self._resolved_executables[job.job_id] = resolved_executable
            working_directory = self._workspace_path(job.working_directory)
            if not working_directory.is_dir():
                raise ManifestError(
                    f"job {job.job_id!r} working_directory does not exist: "
                    f"{working_directory}"
                )
            self._require_declared_inputs(job)
            for output in job.expected_outputs:
                output_path = self._workspace_path(output)
                if output_path == self.state_path:
                    raise ManifestError(
                        f"job {job.job_id!r} expected output cannot be "
                        "queue_state.json"
                    )
                try:
                    output_path.relative_to(runner_jobs_root)
                except ValueError:
                    pass
                else:
                    raise ManifestError(
                        f"job {job.job_id!r} expected output uses the "
                        "runner-owned jobs directory"
                    )
                if output_path.is_symlink():
                    raise ManifestError(
                        f"job {job.job_id!r} expected output cannot be a "
                        f"symlink: {output}"
                    )

    @staticmethod
    def _verify_lane_source_binding(job: JobSpec) -> None:
        expected = job.environment.get(
            "SIMREADY_MECHANISM_SOURCE_SHA256"
        )
        if expected is None:
            return
        from ..mechanism.io import lane_execution_source_sha256

        actual = lane_execution_source_sha256()
        if expected != actual:
            raise PlainQueueRunError(
                "VALIDATOR_SOURCE_CHANGED: job "
                f"{job.job_id!r} was bound to {expected}, but the live lane "
                f"execution source is {actual}; rebuild the queue before run"
            )

    def _new_state(self) -> dict[str, Any]:
        return {
            "schema_version": "0.1",
            "runner": "plain_queue_runner_v1",
            "run_name": self.manifest.run_name,
            "manifest_fingerprint": self.manifest.fingerprint,
            "workspace_root": ".",
            "run_dir": self.run_dir.relative_to(self.workspace_root).as_posix(),
            "created_at": _utc_now(),
            "started_at": None,
            "finished_at": None,
            "status": "PENDING",
            "parallelism": {
                "max_parallel_jobs": (
                    self.manifest.queue_policy.max_parallel_jobs
                ),
                "active_job_ids": [],
                "waiting_job_ids": [],
            },
            "jobs": {
                job.job_id: {
                    "fingerprint": job.fingerprint,
                    "kind": job.kind.value,
                    "command": list(job.command),
                    "status": JobStatus.PENDING.value,
                    "lane_phase": JobStatus.PENDING.value,
                    "attempts": 0,
                    "reason": "Not started",
                    "started_at": None,
                    "finished_at": None,
                    "duration_seconds": None,
                    "exit_code": None,
                }
                for job in self.manifest.jobs
            },
            "events": [],
        }

    def _sample_for_admission(self) -> SystemSnapshot:
        try:
            return self.probe.sample(
                self.manifest.resource_policy.preflight_sample_seconds
            )
        except Exception as exc:
            raise PlainQueueRunError(
                "resource admission probe failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _resource_block_reasons(
        self,
        job: JobSpec,
        *,
        snapshot: SystemSnapshot,
        active_reservation_gib: float,
    ) -> tuple[list[str], float]:
        policy = self.manifest.resource_policy
        required = self._required_available_memory_gib(job)
        required_with_active = required + max(0.0, active_reservation_gib)
        reasons: list[str] = []
        if snapshot.cpu_percent > policy.preflight_cpu_percent:
            reasons.append(
                f"CPU {snapshot.cpu_percent:.1f}% exceeds "
                f"{policy.preflight_cpu_percent:.1f}%"
            )
        if snapshot.memory_percent > policy.preflight_memory_percent:
            reasons.append(
                f"memory {snapshot.memory_percent:.1f}% exceeds "
                f"{policy.preflight_memory_percent:.1f}%"
            )
        if snapshot.available_memory_gib < required_with_active:
            reasons.append(
                f"available memory {snapshot.available_memory_gib:.2f} GiB "
                f"is below required {required_with_active:.2f} GiB"
            )
        if snapshot.free_disk_gib < policy.min_free_disk_gib:
            reasons.append(
                f"free disk {snapshot.free_disk_gib:.2f} GiB is below "
                f"required {policy.min_free_disk_gib:.2f} GiB"
            )
        return reasons, required_with_active

    def _required_available_memory_gib(self, job: JobSpec) -> float:
        policy = self.manifest.resource_policy
        required = float(policy.min_available_memory_gib)
        if job.estimated_peak_memory_gib is not None:
            required = max(required, float(job.estimated_peak_memory_gib) * 1.25)
        if job.max_process_tree_memory_gib is not None:
            log_reserve = 2.0 * policy.max_log_bytes_per_stream / GIB
            required = max(
                required,
                float(job.max_process_tree_memory_gib)
                + float(policy.critical_available_memory_gib)
                + log_reserve,
            )
        return required

    def _reservation_gib(self, job: JobSpec) -> float:
        values = [0.0]
        if job.estimated_peak_memory_gib is not None:
            values.append(float(job.estimated_peak_memory_gib) * 1.25)
        if job.max_process_tree_memory_gib is not None:
            values.append(
                float(job.max_process_tree_memory_gib)
                + 2.0
                * self.manifest.resource_policy.max_log_bytes_per_stream
                / GIB
            )
        return max(values)

    def _start_job(
        self,
        state: dict[str, Any],
        job: JobSpec,
        snapshot: SystemSnapshot,
    ) -> _ActiveJob:
        # Recompute at the last safe boundary.  A separate validate command
        # cannot authorize a later launch if the imported lane source changed.
        self._verify_lane_source_binding(job)
        self._require_declared_inputs(job)
        # Completed and stale outputs are fresh per attempt. A validator-owned
        # resumable PHYSICS_PARTIAL trace is retained so the worker can restore
        # the interrupted lane instead of replaying it from time zero.
        self._prepare_expected_outputs(job)
        job_dir = (self.run_dir / "jobs" / job.job_id).resolve()
        self._require_inside_run(job_dir, f"job directory {job.job_id!r}")
        job_dir.mkdir(parents=True, exist_ok=True)
        attempt = int(state["jobs"][job.job_id].get("attempts", 0)) + 1
        stdout_path = job_dir / f"attempt-{attempt:03d}.stdout.log"
        stderr_path = job_dir / f"attempt-{attempt:03d}.stderr.log"
        temporary = job_dir / f"attempt-{attempt:03d}-{uuid.uuid4().hex}.tmp"
        temporary.mkdir(mode=0o700)
        try:
            environment = self._build_environment(job, temporary)
            process = subprocess.Popen(
                [str(self._resolved_executables[job.job_id]), *job.command[1:]],
                cwd=self._workspace_path(job.working_directory),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                text=False,
                start_new_session=os.name != "nt",
                creationflags=(
                    getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                    if os.name == "nt"
                    else 0
                ),
            )
        except BaseException:
            self._remove_temporary_directory(temporary)
            raise
        captures: tuple[BoundedPipeCapture, ...] = ()
        try:
            if process.stdout is None or process.stderr is None:
                raise PlainQueueRunError(
                    "child stdout/stderr pipes are unavailable"
                )
            captures = (
                BoundedPipeCapture(
                    "stdout",
                    process.stdout,
                    stdout_path,
                    self.manifest.resource_policy.max_log_bytes_per_stream,
                ),
                BoundedPipeCapture(
                    "stderr",
                    process.stderr,
                    stderr_path,
                    self.manifest.resource_policy.max_log_bytes_per_stream,
                ),
            )
            for capture in captures:
                capture.start()
            if self.manifest.resource_policy.lower_process_priority:
                self.probe.lower_process_tree_priority(process.pid)
            started_at = _utc_now()
            record = state["jobs"][job.job_id]
            record.update(
                {
                    "status": JobStatus.RUNNING.value,
                    "lane_phase": JobStatus.RUNNING.value,
                    "attempts": attempt,
                    "reason": None,
                    "started_at": started_at,
                    "finished_at": None,
                    "stdout_log": stdout_path.relative_to(self.run_dir).as_posix(),
                    "stderr_log": stderr_path.relative_to(self.run_dir).as_posix(),
                    "preflight_system_snapshot": snapshot.to_dict(),
                    "pid": process.pid,
                    "process_create_time": self.probe.process_create_time(
                        process.pid
                    ),
                    "owner_token": (
                        None
                        if self._attempt is None
                        else self._attempt.get("owner_token")
                    ),
                }
            )
            self._event(
                state,
                "JOB_STARTED",
                f"Started job {job.job_id}",
                job_id=job.job_id,
            )
            return _ActiveJob(
                job=job,
                process=process,
                captures=captures,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                temporary_directory=temporary,
                started_at=started_at,
                started_monotonic=self.monotonic(),
                preflight_snapshot=snapshot,
                reservation_gib=self._reservation_gib(job),
                last_snapshot=snapshot,
            )
        except BaseException:
            self._terminate_process_group(process)
            for capture in captures:
                try:
                    capture.finish(timeout_seconds=0.5)
                except Exception:
                    pass
            self._remove_temporary_directory(temporary)
            raise

    def _poll_active(self, state: dict[str, Any]) -> JobStatus | None:
        policy = self.manifest.resource_policy
        snapshot: SystemSnapshot | None = None
        if self._active:
            try:
                snapshot = self.probe.sample(0.0)
            except Exception as exc:
                raise PlainQueueRunError(
                    "resource monitor failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        for active in list(self._active.values()):
            job = active.job
            process = active.process
            active.peak_tree_rss_bytes = max(
                active.peak_tree_rss_bytes,
                self.probe.process_tree_rss_bytes(process.pid),
            )
            if snapshot is not None:
                active.last_snapshot = snapshot
                critical = (
                    snapshot.memory_percent >= policy.critical_memory_percent
                    or snapshot.available_memory_gib
                    <= policy.critical_available_memory_gib
                    or snapshot.free_disk_gib <= policy.critical_free_disk_gib
                )
                active.critical_samples = (
                    active.critical_samples + 1 if critical else 0
                )
            if (
                job.max_process_tree_memory_gib is not None
                and active.peak_tree_rss_bytes
                > job.max_process_tree_memory_gib * GIB
            ):
                self._terminate_process_group(process)
                self._finish_active(
                    state,
                    active,
                    JobStatus.RESOURCE_BLOCKED,
                    "Process tree exceeded declared memory limit",
                )
                return JobStatus.RESOURCE_BLOCKED
            if active.critical_samples >= policy.critical_pressure_samples:
                self._terminate_process_group(process)
                self._finish_active(
                    state,
                    active,
                    JobStatus.RESOURCE_BLOCKED,
                    "Sustained critical memory or disk pressure",
                )
                return JobStatus.RESOURCE_BLOCKED
            elapsed = self.monotonic() - active.started_monotonic
            if elapsed >= job.timeout_seconds and process.poll() is None:
                self._terminate_process_group(process)
                self._finish_active(
                    state,
                    active,
                    JobStatus.TIMED_OUT,
                    f"Wall-clock timeout after {job.timeout_seconds:.3f} seconds",
                )
                return JobStatus.TIMED_OUT
            exit_code = process.poll()
            if exit_code is None:
                continue
            if exit_code != 0:
                self._finish_active(
                    state,
                    active,
                    JobStatus.FAILED,
                    f"Process exited with code {exit_code}",
                )
                return JobStatus.FAILED
            try:
                self._require_declared_inputs(job)
                missing = self._missing_outputs(job)
            except (ManifestError, PlainQueueRunError, OSError) as exc:
                self._finish_active(
                    state,
                    active,
                    JobStatus.FAILED,
                    f"Post-run validation failed: {exc}",
                )
                return JobStatus.FAILED
            if missing:
                self._finish_active(
                    state,
                    active,
                    JobStatus.FAILED,
                    "Expected outputs are missing or not regular files: "
                    + ", ".join(missing),
                )
                return JobStatus.FAILED
            self._finish_active(
                state,
                active,
                JobStatus.SUCCEEDED,
                "Process exited successfully and expected outputs exist",
            )
        return None

    @staticmethod
    def _is_resumable_partial_output(
        path: Path,
        *,
        expected_validator_source: str | None,
    ) -> bool:
        try:
            if path.stat().st_size > MAX_STATE_BYTES:
                return False
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        checkpoint = (
            payload.get("checkpoint")
            if isinstance(payload, dict)
            else None
        )
        return bool(
            payload.get("status") == "PHYSICS_PARTIAL"
            and isinstance(checkpoint, dict)
            and checkpoint.get("resumable") is True
            and isinstance(checkpoint.get("resume_state"), dict)
            and (
                expected_validator_source is None
                or payload.get("validator_source_sha256")
                == expected_validator_source
            )
        )

    def _prepare_expected_outputs(self, job: JobSpec) -> None:
        for relative in job.expected_outputs:
            raw = self.workspace_root / relative
            if raw.is_symlink():
                raise PlainQueueRunError(
                    f"job {job.job_id!r} expected output cannot be a symlink: "
                    f"{relative}"
                )
            path = self._workspace_path(relative)
            if not path.exists():
                continue
            if not path.is_file():
                raise PlainQueueRunError(
                    f"job {job.job_id!r} expected output is not a regular "
                    f"file: {relative}"
                )
            if self._is_resumable_partial_output(
                path,
                expected_validator_source=job.environment.get(
                    "SIMREADY_MECHANISM_SOURCE_SHA256"
                ),
            ):
                continue
            path.unlink()

    def _has_resumable_partial_output(self, job: JobSpec) -> bool:
        """Return whether this job left a checkpoint its worker can restore."""

        expected_validator_source = job.environment.get(
            "SIMREADY_MECHANISM_SOURCE_SHA256"
        )
        return any(
            self._is_resumable_partial_output(
                self._workspace_path(relative),
                expected_validator_source=expected_validator_source,
            )
            for relative in job.expected_outputs
        )

    def _finish_active(
        self,
        state: dict[str, Any],
        active: _ActiveJob,
        status: JobStatus,
        reason: str,
    ) -> None:
        process = active.process
        if process.poll() is None:
            self._terminate_process_group(process)
        capture_errors: list[str] = []
        capture_deadline = time.monotonic() + max(
            0.1,
            self.manifest.resource_policy.terminate_grace_seconds,
        )
        for capture in active.captures:
            metrics = capture.finish(
                timeout_seconds=max(0.0, capture_deadline - time.monotonic())
            )
            active.capture_metrics[capture.name] = metrics
            if metrics.get("error") is not None:
                capture_errors.append(f"{capture.name}: {metrics['error']}")
        if capture_errors and status is JobStatus.SUCCEEDED:
            status = JobStatus.FAILED
            reason = "Log capture failed: " + "; ".join(capture_errors)
        finished_at = _utc_now()
        record = state["jobs"][active.job.job_id]
        record.update(
            {
                "status": status.value,
                "lane_phase": status.value,
                "reason": reason,
                "exit_code": process.poll(),
                "finished_at": finished_at,
                "duration_seconds": round(
                    max(0.0, self.monotonic() - active.started_monotonic), 6
                ),
                "log_capture": active.capture_metrics,
                "resource_metrics": {
                    "peak_process_tree_rss_bytes": active.peak_tree_rss_bytes,
                    "peak_process_tree_rss_gib": round(
                        active.peak_tree_rss_bytes / GIB, 3
                    ),
                    "last_system_snapshot": (
                        active.last_snapshot.to_dict()
                        if active.last_snapshot is not None
                        else None
                    ),
                },
            }
        )
        record.pop("pid", None)
        self._active.pop(active.job.job_id, None)
        temporary_cleanup_error = self._remove_temporary_directory(
            active.temporary_directory
        )
        if temporary_cleanup_error is not None:
            record["temporary_cleanup_error"] = temporary_cleanup_error
            if status is JobStatus.SUCCEEDED:
                status = JobStatus.FAILED
                reason = temporary_cleanup_error
                record["status"] = JobStatus.FAILED.value
                record["lane_phase"] = JobStatus.FAILED.value
                record["reason"] = reason
        self._event(
            state,
            f"JOB_{status.value}",
            reason,
            job_id=active.job.job_id,
        )
        self._write_state(state)

    def _cancel_all_active(self, state: dict[str, Any], reason: str) -> None:
        active_jobs = list(self._active.values())
        for active in active_jobs:
            self._terminate_process_group(active.process)
        for active in active_jobs:
            if active.job.job_id in self._active:
                self._finish_active(state, active, JobStatus.CANCELLED, reason)

    def _ordered_pending(self, pending: list[JobSpec]) -> list[JobSpec]:
        pending_ids = {job.job_id for job in pending}
        return [
            job
            for job in self.manifest.jobs
            if job.job_id in pending_ids
        ]

    def _reuse_completed_outputs(
        self,
        state: dict[str, Any],
        pending: list[JobSpec],
    ) -> list[JobSpec]:
        remaining: list[JobSpec] = []
        for job in pending:
            if not job.expected_outputs or self._missing_outputs(job):
                remaining.append(job)
                continue
            previous_jobs = (
                self._previous_state.get("jobs")
                if isinstance(self._previous_state, Mapping)
                else None
            )
            previous_job = (
                previous_jobs.get(job.job_id)
                if isinstance(previous_jobs, Mapping)
                else None
            )
            if self._previous_state is not None and (
                not isinstance(previous_job, Mapping)
                or previous_job.get("status") != JobStatus.SUCCEEDED.value
                or previous_job.get("fingerprint") != job.fingerprint
            ):
                remaining.append(job)
                continue
            record = state["jobs"][job.job_id]
            record.update(
                {
                    "status": JobStatus.SUCCEEDED.value,
                    "lane_phase": JobStatus.SUCCEEDED.value,
                    "reason": "Reused existing completed outputs",
                    "finished_at": _utc_now(),
                }
            )
            self._event(
                state,
                "JOB_REUSED",
                "Existing completed outputs were kept",
                job_id=job.job_id,
            )
        return remaining

    def _queue_job_for_resume(
        self,
        state: dict[str, Any],
        job: JobSpec,
        *,
        counter_name: str,
        reason: str,
    ) -> None:
        record = state["jobs"][job.job_id]
        interruption = {
            "status": record.get("status"),
            "reason": record.get("reason"),
            "exit_code": record.get("exit_code"),
            "finished_at": record.get("finished_at"),
            "duration_seconds": record.get("duration_seconds"),
        }
        record[counter_name] = int(record.get(counter_name, 0)) + 1
        record["last_interruption"] = interruption
        record.update(
            {
                "status": JobStatus.PENDING.value,
                "lane_phase": JobStatus.PENDING.value,
                "reason": reason,
                "started_at": None,
                "finished_at": None,
                "duration_seconds": None,
                "exit_code": None,
            }
        )
        self._event(
            state,
            "JOB_RESUME_QUEUED",
            reason,
            job_id=job.job_id,
        )

    def _requeue_timed_out_jobs(
        self,
        state: dict[str, Any],
        pending: list[JobSpec],
    ) -> bool:
        limit = self.manifest.queue_policy.max_timeout_resume_attempts
        timed_out_ids = {
            job_id
            for job_id, record in state["jobs"].items()
            if record.get("status") == JobStatus.TIMED_OUT.value
        }
        if not timed_out_ids:
            return False
        jobs_by_id = {job.job_id: job for job in self.manifest.jobs}
        for job_id in sorted(timed_out_ids):
            record = state["jobs"][job_id]
            resumable = self._has_resumable_partial_output(jobs_by_id[job_id])
            record["timeout_checkpoint_resumable"] = resumable
            if not resumable:
                record["timeout_resume_decision"] = (
                    "not_requeued_no_resumable_checkpoint"
                )
                self._event(
                    state,
                    "JOB_TIMEOUT_NOT_RESUMABLE",
                    "Timed-out job did not publish a resumable checkpoint",
                    job_id=job_id,
                )
                self._write_state(state)
                return False
            if int(record.get("timeout_resume_count", 0)) >= limit:
                return False
        for job_id in sorted(timed_out_ids):
            job = jobs_by_id[job_id]
            self._queue_job_for_resume(
                state,
                job,
                counter_name="timeout_resume_count",
                reason="Queued to resume after the lane timeout",
            )
            pending.append(job)
        self._write_state(state)
        return True

    def _resource_block_pending(
        self,
        state: dict[str, Any],
        pending: list[JobSpec],
        *,
        snapshot: SystemSnapshot,
        required_available_memory_gib: float,
        reason: str,
        resource_wait_seconds: float,
    ) -> None:
        if not pending:
            return
        job = pending[0]
        record = state["jobs"][job.job_id]
        record.update(
            {
                "status": JobStatus.RESOURCE_BLOCKED.value,
                "lane_phase": JobStatus.RESOURCE_BLOCKED.value,
                "reason": reason,
                "finished_at": _utc_now(),
                "resource_wait_seconds": round(resource_wait_seconds, 6),
                "required_available_memory_gib": round(
                    required_available_memory_gib, 3
                ),
                "last_system_snapshot": snapshot.to_dict(),
            }
        )
        self._event(
            state,
            "JOB_RESOURCE_BLOCKED",
            reason,
            job_id=job.job_id,
        )
        self._cancel_pending(
            state,
            pending[1:],
            "Cancelled because another lane became RESOURCE_BLOCKED",
        )

    def _cancel_pending(
        self,
        state: dict[str, Any],
        pending: list[JobSpec],
        reason: str,
    ) -> None:
        for job in pending:
            record = state["jobs"][job.job_id]
            record.update(
                {
                    "status": JobStatus.CANCELLED.value,
                    "lane_phase": JobStatus.CANCELLED.value,
                    "reason": reason,
                    "finished_at": _utc_now(),
                }
            )
            self._event(
                state,
                "JOB_CANCELLED",
                reason,
                job_id=job.job_id,
            )
        self._write_state(state)

    def _terminate_process_group(self, process: subprocess.Popen[bytes]) -> None:
        leader_running = process.poll() is None
        grace = max(
            0.1,
            float(self.manifest.resource_policy.terminate_grace_seconds),
        )
        deadline = time.monotonic() + grace
        if os.name == "nt":
            if not leader_running:
                return
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=grace,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            except PermissionError:
                pass
            if leader_running:
                try:
                    process.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
            else:
                # poll may already have reaped the group leader while a
                # descendant remains in the inherited process group. A
                # group-directed hard kill is the only bounded cleanup path.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except (ProcessLookupError, PermissionError):
                    return
                process.wait(timeout=1.0)

    def _require_declared_inputs(self, job: JobSpec) -> None:
        for relative, expected in job.input_files.items():
            raw = self.workspace_root / relative
            if raw.is_symlink():
                raise ManifestError(
                    f"job {job.job_id!r} input cannot be a symlink: {relative}"
                )
            path = self._workspace_path(relative)
            try:
                actual = _stable_regular_file_sha256(path)
            except OSError as exc:
                raise ManifestError(
                    f"job {job.job_id!r} input cannot be verified: "
                    f"{relative}: {exc}"
                ) from exc
            if actual != expected:
                raise ManifestError(
                    f"job {job.job_id!r} input SHA-256 mismatch: {relative}"
                )

    def _missing_outputs(self, job: JobSpec) -> list[str]:
        missing: list[str] = []
        for relative in job.expected_outputs:
            raw = self.workspace_root / relative
            if raw.is_symlink():
                missing.append(relative)
                continue
            path = self._workspace_path(relative)
            if not path.is_file():
                missing.append(relative)
        return missing

    def _build_environment(
        self,
        job: JobSpec,
        temp_directory: Path | None,
    ) -> dict[str, str]:
        environment = (
            os.environ.copy() if job.inherit_parent_environment else {}
        )
        update_environment_case_insensitively(environment, job.environment)
        policy = self.manifest.resource_policy
        default_threads = (
            policy.pychrono_cpu_threads
            if job.kind is JobKind.PYCHRONO
            else policy.generic_cpu_threads
        )
        threads = min(job.cpu_threads or default_threads, default_threads)
        controlled = {
            "OMP_NUM_THREADS": str(threads),
            "OMP_THREAD_LIMIT": str(threads),
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "SIMREADY_PROJECT_ADAPTER_ISOLATE": "0",
        }
        if temp_directory is not None:
            controlled.update(
                {
                    "TEMP": str(temp_directory),
                    "TMP": str(temp_directory),
                    "TMPDIR": str(temp_directory),
                }
            )
        update_environment_case_insensitively(environment, controlled)
        if os.name == "nt":
            units = windows_environment_utf16_units(environment)
            if units > MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS:
                raise PlainQueueRunError(
                    "Windows child environment block exceeds "
                    f"{MAX_WINDOWS_ENVIRONMENT_UTF16_UNITS} UTF-16 units: "
                    f"{units}"
                )
        return environment

    def _remove_temporary_directory(self, path: Path) -> str | None:
        if (
            path.parent.parent.parent != self.run_dir
            or path.parent.parent.name != "jobs"
            or path.parent.name not in {job.job_id for job in self.manifest.jobs}
            or not path.name.startswith("attempt-")
            or not path.name.endswith(".tmp")
        ):
            return f"refused unsafe temporary directory {path}"
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            return (
                f"could not remove job temporary directory {path}: "
                f"{type(exc).__name__}: {exc}"
            )
        return None

    def _queue_status(
        self,
        state: dict[str, Any],
        first_hard_failure: bool,
    ) -> str:
        statuses = {
            record.get("status") for record in state["jobs"].values()
        }
        if statuses == {JobStatus.SUCCEEDED.value}:
            return "SUCCEEDED"
        if JobStatus.RESOURCE_BLOCKED.value in statuses and not first_hard_failure:
            return JobStatus.RESOURCE_BLOCKED.value
        if statuses & {JobStatus.FAILED.value, JobStatus.TIMED_OUT.value}:
            return "FAILED"
        if JobStatus.CANCELLED.value in statuses:
            return "FAILED"
        return "INCOMPLETE"

    def _event(
        self,
        state: dict[str, Any],
        code: str,
        message: str,
        *,
        job_id: str | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "at": _utc_now(),
            "code": code,
            "message": message,
        }
        if job_id is not None:
            event["job_id"] = job_id
        events = state.setdefault("events", [])
        events.append(event)
        if len(events) > 500:
            del events[:-500]

    def _write_state(self, state: dict[str, Any]) -> None:
        with self._state_lock:
            state["updated_at"] = _utc_now()
            counts: dict[str, int] = {}
            for record in state["jobs"].values():
                status = str(record.get("status") or JobStatus.PENDING.value)
                counts[status] = counts.get(status, 0) + 1
            counts["total"] = len(state["jobs"])
            state["summary"] = dict(sorted(counts.items()))
            state["parallelism"]["active_job_ids"] = sorted(self._active)
            state["parallelism"]["waiting_job_ids"] = sorted(
                job_id
                for job_id, record in state["jobs"].items()
                if record.get("lane_phase") == "WAITING_FOR_RESOURCE"
            )
            payload = json.dumps(
                state,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            if len(payload) > MAX_STATE_BYTES:
                raise PlainQueueRunError(
                    f"queue state exceeds {MAX_STATE_BYTES} byte limit"
                )
            atomic_write_bytes(self.state_path, payload + b"\n")

    def _workspace_path(self, relative: str) -> Path:
        path = (self.workspace_root / relative).resolve()
        self._require_inside_workspace(path, relative)
        return path

    def _require_inside_workspace(self, path: Path, label: str) -> None:
        try:
            path.relative_to(self.workspace_root)
        except ValueError as exc:
            raise PlainQueueRunError(
                f"{label} escapes workspace root: {path}"
            ) from exc

    def _require_inside_run(self, path: Path, label: str) -> None:
        try:
            path.relative_to(self.run_dir)
        except ValueError as exc:
            raise PlainQueueRunError(
                f"{label} escapes run directory: {path}"
            ) from exc
