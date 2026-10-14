from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.runtime import (
    JobSpec,
    JobStatus,
    PlainQueueRunner,
    QueueManifest,
    QueuePolicy,
    ResourcePolicy,
    SystemSnapshot,
)


class _ConstantProbe:
    def __init__(
        self,
        *,
        available_gib: float = 8.0,
        process_tree_rss_bytes: int = 8 * 1024**2,
    ) -> None:
        self.available_gib = available_gib
        self.process_tree_rss = process_tree_rss_bytes

    def sample(self, interval_seconds: float = 0.0) -> SystemSnapshot:
        return SystemSnapshot(
            captured_at="2026-08-01T00:00:00+00:00",
            cpu_percent=1.0,
            memory_percent=20.0,
            available_memory_bytes=int(self.available_gib * 1024**3),
            total_memory_bytes=16 * 1024**3,
            free_disk_bytes=100 * 1024**3,
            total_disk_bytes=1000 * 1024**3,
        )

    def process_tree_rss_bytes(self, pid: int) -> int:
        return self.process_tree_rss

    def lower_process_tree_priority(self, pid: int) -> None:
        return None

    def pid_exists(self, pid: int) -> bool:
        return pid == os.getpid()

    def process_create_time(self, pid: int) -> float | None:
        return 1.0 if pid == os.getpid() else None


def _policy(**changes: object) -> ResourcePolicy:
    base = ResourcePolicy(
        preflight_cpu_percent=95.0,
        preflight_memory_percent=95.0,
        min_available_memory_gib=0.5,
        min_free_disk_gib=1.0,
        critical_memory_percent=98.0,
        critical_available_memory_gib=0.25,
        critical_free_disk_gib=0.5,
        preflight_sample_seconds=0.0,
        resource_poll_seconds=0.02,
        max_resource_wait_seconds=2.0,
        critical_pressure_samples=2,
        cooldown_seconds=0.0,
        pychrono_cpu_threads=1,
        generic_cpu_threads=1,
        terminate_grace_seconds=1.0,
        lower_process_priority=False,
    )
    return replace(base, **changes)


def _job(
    job_id: str,
    code: str,
    *,
    output: str | None = None,
    estimated_peak_memory_gib: float | None = None,
    environment: dict[str, str] | None = None,
) -> JobSpec:
    return JobSpec(
        job_id=job_id,
        command=(sys.executable, "-c", code),
        timeout_seconds=5.0,
        expected_outputs=((output,) if output else ()),
        estimated_peak_memory_gib=estimated_peak_memory_gib,
        environment=environment or {},
    )


def _manifest(
    *jobs: JobSpec,
    max_parallel_jobs: int,
    resource_policy: ResourcePolicy | None = None,
    reuse_successful: bool = False,
    max_timeout_resume_attempts: int = 0,
) -> QueueManifest:
    return QueueManifest(
        run_name="parallel-targeted",
        jobs=tuple(jobs),
        resource_policy=resource_policy or _policy(),
        queue_policy=QueuePolicy(
            max_wall_time_seconds=20.0,
            stop_on_failure=True,
            reuse_successful=reuse_successful,
            max_timeout_resume_attempts=max_timeout_resume_attempts,
            max_parallel_jobs=max_parallel_jobs,
        ),
    )


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ParallelRuntimeTargetedTests(unittest.TestCase):
    def test_parallel_timeout_resumes_only_interrupted_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resumable_code = (
                "import json,time; from pathlib import Path; "
                "p=Path('resumable.json'); "
                "v=json.loads(p.read_text()) if p.exists() else {}; "
                "step=v.get('checkpoint',{}).get('resume_state',{}).get('step',0); "
                "p.write_text(json.dumps({'status':'PASS'})) if step else "
                "(p.write_text(json.dumps({'status':'PHYSICS_PARTIAL',"
                "'checkpoint':{'resumable':True,'resume_state':{'step':1}}})),"
                "time.sleep(1.0))"
            )
            companion_code = (
                "import time; from pathlib import Path; time.sleep(0.3); "
                "Path('companion.txt').write_text('done')"
            )
            resumable = replace(
                _job("resumable", resumable_code, output="resumable.json"),
                timeout_seconds=0.15,
            )
            companion = replace(
                _job("companion", companion_code, output="companion.txt"),
                timeout_seconds=2.0,
            )
            state = PlainQueueRunner(
                _manifest(
                    resumable,
                    companion,
                    max_parallel_jobs=2,
                    max_timeout_resume_attempts=1,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "parallel-resume",
                probe=_ConstantProbe(),
            ).run()

            self.assertEqual(state["status"], JobStatus.SUCCEEDED.value)
            self.assertEqual(state["parallelism"]["max_parallel_jobs"], 2)
            self.assertEqual(state["jobs"]["resumable"]["attempts"], 2)
            self.assertEqual(
                state["jobs"]["resumable"]["timeout_resume_count"], 1
            )
            self.assertEqual(state["jobs"]["companion"]["attempts"], 1)
            self.assertEqual(
                json.loads((root / "resumable.json").read_text())["status"],
                "PASS",
            )

    def test_timeout_without_resumable_checkpoint_is_not_retried(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = (
                "import json,time; from pathlib import Path; "
                "p=Path('partial.json'); "
                "p.write_text(json.dumps({'status':'PHYSICS_PARTIAL',"
                "'checkpoint':{'resumable':False,'resume_state':None}})); "
                "time.sleep(1.0)"
            )
            job = replace(
                _job("nonresumable", code, output="partial.json"),
                timeout_seconds=0.15,
            )

            state = PlainQueueRunner(
                _manifest(
                    job,
                    max_parallel_jobs=1,
                    max_timeout_resume_attempts=3,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "nonresumable-timeout",
                probe=_ConstantProbe(),
            ).run()

            record = state["jobs"]["nonresumable"]
            self.assertEqual(state["status"], JobStatus.FAILED.value)
            self.assertEqual(record["status"], JobStatus.TIMED_OUT.value)
            self.assertEqual(record["attempts"], 1)
            self.assertNotIn("timeout_resume_count", record)
            self.assertFalse(record["timeout_checkpoint_resumable"])
            self.assertEqual(
                record["timeout_resume_decision"],
                "not_requeued_no_resumable_checkpoint",
            )
            self.assertIn(
                "JOB_TIMEOUT_NOT_RESUMABLE",
                [event["code"] for event in state["events"]],
            )

    def test_idle_resource_wait_becomes_resource_blocked(self) -> None:
        clock = _FakeClock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = _policy(
                max_resource_wait_seconds=0.02,
                resource_poll_seconds=0.01,
            )
            job = _job(
                "starved",
                "from pathlib import Path; Path('starved.txt').write_text('done')",
                output="starved.txt",
                estimated_peak_memory_gib=10.0,
            )
            companion = _job(
                "also-starved",
                "from pathlib import Path; Path('also.txt').write_text('done')",
                output="also.txt",
                estimated_peak_memory_gib=10.0,
            )
            runner = PlainQueueRunner(
                _manifest(
                    job,
                    companion,
                    max_parallel_jobs=2,
                    resource_policy=policy,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "idle-resource-wait",
                probe=_ConstantProbe(available_gib=1.2),
                sleep=clock.sleep,
                monotonic=clock.monotonic,
            )
            state = runner.run()

        self.assertEqual(state["status"], JobStatus.RESOURCE_BLOCKED.value)
        self.assertEqual(
            state["jobs"]["starved"]["status"],
            JobStatus.RESOURCE_BLOCKED.value,
        )
        self.assertEqual(
            state["jobs"]["also-starved"]["status"],
            JobStatus.CANCELLED.value,
        )
        self.assertIn(
            "Idle resource admission wait exceeded",
            state["jobs"]["starved"]["reason"],
        )
        self.assertGreaterEqual(
            state["jobs"]["starved"]["resource_wait_seconds"],
            0.02,
        )
        self.assertLess(clock.now, 1.0)
        self.assertIn("last_system_snapshot", state["jobs"]["starved"])
        self.assertGreater(
            state["jobs"]["starved"]["required_available_memory_gib"],
            1.2,
        )

    def test_resource_wait_does_not_cancel_active_parallel_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = _policy(
                max_resource_wait_seconds=0.04,
                resource_poll_seconds=0.01,
            )
            first = _job(
                "first",
                "import time; from pathlib import Path; time.sleep(0.15); "
                "Path('first.txt').write_text('done')",
                output="first.txt",
                estimated_peak_memory_gib=0.6,
            )
            second = _job(
                "second",
                "from pathlib import Path; Path('second.txt').write_text('done')",
                output="second.txt",
                estimated_peak_memory_gib=0.6,
            )
            state = PlainQueueRunner(
                _manifest(
                    first,
                    second,
                    max_parallel_jobs=2,
                    resource_policy=policy,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "resource-wait",
                probe=_ConstantProbe(available_gib=1.2),
            ).run()

            self.assertEqual(state["status"], JobStatus.SUCCEEDED.value)
            self.assertEqual(state["parallelism"]["max_parallel_jobs"], 2)
            self.assertEqual(state["jobs"]["first"]["attempts"], 1)
            self.assertEqual(state["jobs"]["second"]["attempts"], 1)
            self.assertIn(
                "JOB_RESOURCE_WAIT_CONTINUES",
                [event["code"] for event in state["events"]],
            )

    def test_serial_parallel_equivalence_and_real_overlap_three_times(
        self,
    ) -> None:
        """Hard acceptance: parallel is real and semantically equivalent.

        Two independent jobs must genuinely overlap in wall time when
        parallelism allows it, produce byte-identical semantic outputs to
        the serial reference, and stay stable across three repetitions
        (no race, no ordering dependence, no intermittent loss).
        """

        job_code = (
            "import json,sys,time; from pathlib import Path; "
            "start=time.monotonic(); time.sleep(0.25); "
            "Path(sys.argv[1]).write_text(json.dumps("
            "{'payload': sys.argv[1], 'span': [start, time.monotonic()]}))"
        )

        def _timed_job(job_id: str, output: str) -> JobSpec:
            return JobSpec(
                job_id=job_id,
                command=(sys.executable, "-c", job_code, output),
                timeout_seconds=5.0,
                expected_outputs=(output,),
                environment={},
            )

        def _run(root: Path, *, parallel: int, run_name: str) -> dict:
            state = PlainQueueRunner(
                _manifest(
                    _timed_job("alpha", "alpha.json"),
                    _timed_job("beta", "beta.json"),
                    max_parallel_jobs=parallel,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / run_name,
                probe=_ConstantProbe(),
            ).run()
            self.assertEqual(state["status"], JobStatus.SUCCEEDED.value)
            outputs = {
                name: json.loads((root / f"{name}.json").read_text())
                for name in ("alpha", "beta")
            }
            return outputs

        with tempfile.TemporaryDirectory() as temporary:
            serial_root = Path(temporary) / "serial"
            serial_root.mkdir()
            serial = _run(serial_root, parallel=1, run_name="serial-ref")
            serial_payloads = {
                name: value["payload"] for name, value in serial.items()
            }

            for repetition in range(3):
                parallel_root = Path(temporary) / f"parallel-{repetition}"
                parallel_root.mkdir()
                outputs = _run(
                    parallel_root,
                    parallel=2,
                    run_name=f"parallel-rep-{repetition}",
                )
                self.assertEqual(
                    {n: v["payload"] for n, v in outputs.items()},
                    serial_payloads,
                    f"repetition {repetition} diverged from serial reference",
                )
                alpha_start, alpha_end = outputs["alpha"]["span"]
                beta_start, beta_end = outputs["beta"]["span"]
                overlap = min(alpha_end, beta_end) - max(
                    alpha_start, beta_start
                )
                self.assertGreater(
                    overlap,
                    0.0,
                    f"repetition {repetition}: no real wall-time overlap "
                    f"(alpha {alpha_start:.3f}-{alpha_end:.3f}, "
                    f"beta {beta_start:.3f}-{beta_end:.3f})",
                )

    def test_reuse_successful_keeps_completed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "complete.txt"
            output.write_text("old", encoding="utf-8")
            job = _job(
                "complete",
                "from pathlib import Path; Path('complete.txt').write_text('new')",
                output="complete.txt",
            )
            state = PlainQueueRunner(
                _manifest(
                    job,
                    max_parallel_jobs=1,
                    reuse_successful=True,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "reuse-complete",
                probe=_ConstantProbe(),
            ).run()

            self.assertEqual(state["status"], JobStatus.SUCCEEDED.value)
            self.assertEqual(state["jobs"]["complete"]["attempts"], 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "old")
            self.assertIn(
                "JOB_REUSED",
                [event["code"] for event in state["events"]],
            )

    def test_reuse_successful_reexecutes_when_job_fingerprint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / ".runtime_runs" / "reuse-changed"
            first = _job(
                "complete",
                "from pathlib import Path; Path('complete.txt').write_text('first')",
                output="complete.txt",
            )
            PlainQueueRunner(
                _manifest(first, max_parallel_jobs=1),
                workspace_root=root,
                run_dir=run_dir,
                probe=_ConstantProbe(),
            ).run()
            changed = _job(
                "complete",
                "from pathlib import Path; Path('complete.txt').write_text('changed')",
                output="complete.txt",
            )
            state = PlainQueueRunner(
                _manifest(
                    changed,
                    max_parallel_jobs=1,
                    reuse_successful=True,
                ),
                workspace_root=root,
                run_dir=run_dir,
                probe=_ConstantProbe(),
            ).run()

            self.assertEqual(state["jobs"]["complete"]["attempts"], 1)
            self.assertEqual(
                (root / "complete.txt").read_text(encoding="utf-8"),
                "changed",
            )
            self.assertNotIn(
                "JOB_REUSED",
                [event["code"] for event in state["events"]],
            )

    def test_partial_from_previous_validator_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "partial.json"
            output.write_text(
                json.dumps(
                    {
                        "status": "PHYSICS_PARTIAL",
                        "validator_source_sha256": "a" * 64,
                        "checkpoint": {
                            "resumable": True,
                            "resume_state": {"step": 4},
                        },
                    }
                ),
                encoding="utf-8",
            )
            job = _job(
                "new-validator",
                "from pathlib import Path; "
                "Path('partial.json').write_text('fresh')",
                output="partial.json",
                environment={
                    "SIMREADY_MECHANISM_SOURCE_SHA256": "b" * 64,
                },
            )
            state = PlainQueueRunner(
                _manifest(job, max_parallel_jobs=1),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "new-validator",
                probe=_ConstantProbe(),
            ).run()

            self.assertEqual(state["status"], JobStatus.SUCCEEDED.value)
            self.assertEqual(output.read_text(encoding="utf-8"), "fresh")

    def test_plain_cleanup_targets_group_after_parent_exit(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            plain_runner,
        )

        class CompletedProcess:
            pid = 4321

            def poll(self) -> int:
                return 0

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = PlainQueueRunner(
                _manifest(
                    _job("unused", "print('unused')"),
                    max_parallel_jobs=1,
                ),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "plain-cleanup",
                probe=_ConstantProbe(),
            )
            with (
                patch.object(plain_runner.os, "name", "posix"),
                patch.object(plain_runner.os, "killpg") as killpg,
            ):
                runner._terminate_process_group(CompletedProcess())

        self.assertEqual(
            [call.args for call in killpg.call_args_list],
            [
                (4321, plain_runner.signal.SIGTERM),
                (4321, plain_runner.signal.SIGKILL),
            ],
        )

    def test_plain_process_memory_limit_is_resource_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = replace(
                _job("memory-limited", "import time; time.sleep(5)"),
                max_process_tree_memory_gib=0.5,
            )
            state = PlainQueueRunner(
                _manifest(job, max_parallel_jobs=1),
                workspace_root=root,
                run_dir=root / ".runtime_runs" / "plain-memory-limit",
                probe=_ConstantProbe(process_tree_rss_bytes=1024**3),
            ).run()

        self.assertEqual(state["status"], JobStatus.RESOURCE_BLOCKED.value)
        record = state["jobs"][job.job_id]
        self.assertEqual(record["status"], JobStatus.RESOURCE_BLOCKED.value)
        self.assertIn("declared memory limit", record["reason"])

    def test_parallelism_is_explicit_and_serial_default_round_trips(
        self,
    ) -> None:
        serial = _manifest(
            _job("serial", "pass"),
            max_parallel_jobs=1,
        )
        payload = serial.to_dict()
        self.assertNotIn("max_parallel_jobs", payload["queue_policy"])
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            round_tripped = QueueManifest.from_file(manifest_path)
        self.assertEqual(
            round_tripped.fingerprint,
            serial.fingerprint,
        )


if __name__ == "__main__":
    unittest.main()
