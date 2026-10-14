from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simulation_ready_cad_workflow.validation_pipeline.runtime import (
    JobKind,
    ManifestError,
    QueueManifest,
    QueuePolicy,
    ResourcePolicy,
)


def _resource_policy(**changes: object) -> ResourcePolicy:
    base = ResourcePolicy(
        preflight_cpu_percent=95.0,
        preflight_memory_percent=95.0,
        min_available_memory_gib=0.5,
        min_free_disk_gib=1.0,
        critical_memory_percent=96.0,
        critical_available_memory_gib=0.25,
        critical_free_disk_gib=0.5,
        preflight_sample_seconds=0.0,
        resource_poll_seconds=0.05,
        max_resource_wait_seconds=0.0,
        critical_pressure_samples=2,
        cooldown_seconds=0.0,
        pychrono_cpu_threads=2,
        generic_cpu_threads=2,
        terminate_grace_seconds=1.0,
        lower_process_priority=False,
    )
    return replace(base, **changes)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RuntimeManifestTests(unittest.TestCase):
    def _write_manifest(self, root: Path, payload: object) -> Path:
        path = root / "queue.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def _valid_payload(self) -> dict[str, object]:
        return {
            "schema_version": "0.1",
            "run_name": "test-run",
            "jobs": [
                {
                    "job_id": "probe",
                    "kind": "pychrono",
                    "command": ["conda", "--version"],
                    "timeout_seconds": 30,
                    "working_directory": ".",
                    "estimated_peak_memory_gib": 0.5,
                    "max_process_tree_memory_gib": 1.0,
                    "estimated_wall_time_seconds": 10,
                    "estimated_simulation_steps": 1,
                }
            ],
        }

    def test_default_queue_budget_leaves_overnight_headroom(self) -> None:
        self.assertEqual(
            QueuePolicy().max_wall_time_seconds,
            7 * 60 * 60,
        )
        self.assertFalse(QueuePolicy().reuse_successful)
        self.assertEqual(QueuePolicy().max_timeout_resume_attempts, 0)

    def test_process_containment_rejects_invalid_native_memory_limit(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            process_containment,
        )

        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    process_containment.ProcessContainment().attach_and_resume(
                        object(),  # type: ignore[arg-type]
                        max_job_memory_bytes=value,  # type: ignore[arg-type]
                    )

    def test_invalid_utf8_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            path.write_bytes(b"\xff\xfe")
            with self.assertRaisesRegex(ManifestError, "UTF-8"):
                QueueManifest.from_file(path)

    def test_oversized_manifest_is_rejected_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), self._valid_payload())
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "models.MAX_MANIFEST_BYTES",
                16,
            ):
                with self.assertRaisesRegex(ManifestError, "byte limit"):
                    QueueManifest.from_file(path)

    def test_manifest_change_during_bounded_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), self._valid_payload())
            before = path.stat()

            class ChangedStat:
                st_size = before.st_size
                st_mtime_ns = before.st_mtime_ns + 1
                st_ino = getattr(before, "st_ino", None)

            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "models.Path.stat",
                side_effect=(before, ChangedStat()),
            ), self.assertRaisesRegex(ManifestError, "changed while being read"):
                QueueManifest.from_file(path)

    def test_deep_manifest_is_rejected_before_json_decode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            depth = 65
            path.write_text(
                '{"nested":' * depth + "0" + "}" * depth,
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "nesting"):
                QueueManifest.from_file(path)

    def test_case_colliding_job_ids_are_rejected(self) -> None:
        payload = self._valid_payload()
        duplicate = dict(payload["jobs"][0])
        duplicate["job_id"] = "PROBE"
        payload["jobs"].append(duplicate)
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "case-colliding"):
                QueueManifest.from_file(path)

    def test_case_colliding_outputs_are_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["expected_outputs"] = [
            "Result.json",
            "result.json",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "case-colliding"):
                QueueManifest.from_file(path)

    def test_case_colliding_environment_names_are_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["environment"] = {
            "QUEUE_FLAG": "1",
            "queue_flag": "2",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "case-colliding"):
                QueueManifest.from_file(path)

    def test_resource_policy_cannot_disable_hard_safety_envelope(self) -> None:
        unsafe_policies = (
            {"preflight_cpu_percent": 96.0},
            {"preflight_memory_percent": 96.0},
            {"critical_memory_percent": 99.0},
            {"min_available_memory_gib": 0.25},
            {"critical_available_memory_gib": 0.1},
            {"min_free_disk_gib": 0.5},
            {"critical_free_disk_gib": 0.25},
            {"min_free_disk_gib": 1.0, "critical_free_disk_gib": 2.0},
            {"resource_poll_seconds": 6.0},
            {"terminate_grace_seconds": 31.0},
            {
                "resource_poll_seconds": 5.0,
                "critical_pressure_samples": 4,
            },
        )
        for policy in unsafe_policies:
            with self.subTest(policy=policy):
                payload = self._valid_payload()
                payload["resource_policy"] = policy
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write_manifest(Path(directory), payload)
                    with self.assertRaises(ManifestError):
                        QueueManifest.from_file(path)

    def test_pychrono_manifest_requires_complete_budget(self) -> None:
        fields = (
            "estimated_peak_memory_gib",
            "max_process_tree_memory_gib",
            "estimated_wall_time_seconds",
            "estimated_simulation_steps",
        )
        for field in fields:
            with self.subTest(field=field):
                payload = self._valid_payload()
                payload["jobs"][0].pop(field)
                with tempfile.TemporaryDirectory() as directory:
                    path = self._write_manifest(Path(directory), payload)
                    with self.assertRaisesRegex(
                        ManifestError,
                        "resource budget|estimated_simulation_steps",
                    ):
                        QueueManifest.from_file(path)

    def test_render_manifest_requires_memory_and_time_budget(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["kind"] = "render"
        payload["jobs"][0].pop("estimated_simulation_steps")
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            manifest = QueueManifest.from_file(path)
        self.assertEqual(manifest.jobs[0].kind, JobKind.RENDER)

    def test_log_buffers_must_fit_available_memory_reserve(self) -> None:
        policy = _resource_policy(
            min_available_memory_gib=0.5,
            critical_available_memory_gib=0.25,
            max_log_bytes_per_stream=256 * 1024 * 1024,
        )
        with self.assertRaisesRegex(
            ManifestError,
            r"combined stdout/stderr buffers \(0[.]500 GiB\).*"
            r"\(0[.]250 GiB\)",
        ):
            policy.validate()

    def test_log_limit_must_stay_in_safe_range(self) -> None:
        payload = self._valid_payload()
        payload["resource_policy"] = {"max_log_bytes_per_stream": 512}
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "max_log_bytes"):
                QueueManifest.from_file(path)

    def test_valid_manifest_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), self._valid_payload())
            manifest = QueueManifest.from_file(path)
        self.assertEqual(manifest.jobs[0].kind, JobKind.PYCHRONO)
        self.assertEqual(manifest.resource_policy.pychrono_cpu_threads, 2)
        self.assertEqual(manifest.jobs[0].estimated_wall_time_seconds, 10)
        self.assertEqual(manifest.jobs[0].estimated_simulation_steps, 1)
        self.assertEqual(manifest.jobs[0].max_process_tree_memory_gib, 1.0)

    def test_estimated_wall_time_cannot_exceed_timeout(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["estimated_wall_time_seconds"] = 31
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "no greater than"):
                QueueManifest.from_file(path)

    def test_memory_limit_cannot_be_lower_than_estimate(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["max_process_tree_memory_gib"] = 0.25
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "cannot be lower"):
                QueueManifest.from_file(path)

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queue.json"
            path.write_text(
                '{"schema_version":"0.1","run_name":"a","run_name":"b",'
                '"jobs":[{"job_id":"j","command":["x"],'
                '"timeout_seconds":1}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ManifestError, "duplicate JSON key"):
                QueueManifest.from_file(path)

    def test_shell_wrapper_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["command"] = ["powershell", "-Command", "echo hi"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "shell wrapper"):
                QueueManifest.from_file(path)

    def test_path_traversal_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["working_directory"] = "../outside"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "inside the workspace"):
                QueueManifest.from_file(path)

    def test_windows_alternate_data_stream_path_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["expected_outputs"] = ["artifact.txt:secret"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "Windows-unsafe"):
                QueueManifest.from_file(path)

    def test_windows_device_name_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["expected_outputs"] = ["logs/CON.txt"]
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "Windows-unsafe"):
                QueueManifest.from_file(path)

    def test_windows_device_run_name_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["run_name"] = "NUL"
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "device name"):
                QueueManifest.from_file(path)

    def test_unknown_field_is_rejected(self) -> None:
        payload = self._valid_payload()
        payload["jobs"][0]["timeuot_seconds"] = 12
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest(Path(directory), payload)
            with self.assertRaisesRegex(ManifestError, "unknown field"):
                QueueManifest.from_file(path)


class RuntimeCliTests(unittest.TestCase):
    def test_filesystem_error_returns_infrastructure_exit_code(self) -> None:
        from contextlib import redirect_stderr
        from io import StringIO

        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            cli as runtime_cli,
        )

        stderr = StringIO()
        with (
            patch.object(
                runtime_cli.QueueManifest,
                "from_file",
                side_effect=OSError("synthetic filesystem failure"),
            ),
            redirect_stderr(stderr),
        ):
            exit_code = runtime_cli.main(
                ["validate", "--manifest", "queue.json"]
            )

        self.assertEqual(exit_code, 2)
        self.assertIn("synthetic filesystem failure", stderr.getvalue())

    def test_cli_returns_nonzero_for_failed_plain_queue(self) -> None:
        from contextlib import redirect_stdout
        from io import StringIO

        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            cli as runtime_cli,
        )

        manifest = Mock(run_name="failed-run")
        state = {
            "run_name": "failed-run",
            "status": "FAILED",
        }
        stdout = StringIO()
        with (
            patch.object(
                runtime_cli.QueueManifest,
                "from_file",
                return_value=manifest,
            ),
            patch.object(runtime_cli, "PlainQueueRunner") as runner_type,
            redirect_stdout(stdout),
        ):
            runner_type.return_value.run.return_value = state
            exit_code = runtime_cli.main(
                ["run", "--manifest", "queue.json"]
            )

        self.assertEqual(exit_code, 3)
        self.assertIn("status: FAILED", stdout.getvalue())


class SafeQueueRunnerTests(unittest.TestCase):
    def test_atomic_log_write_handles_partial_writes_and_fsyncs(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            bounded_capture,
        )

        payload = b"partial-write-check" * 512
        real_write = bounded_capture.os.write

        def partial_write(descriptor: int, data: bytes) -> int:
            chunk_size = max(1, len(data) // 3)
            return real_write(descriptor, data[:chunk_size])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stdout.log"
            with (
                patch.object(
                    bounded_capture.os,
                    "write",
                    side_effect=partial_write,
                ),
                patch.object(
                    bounded_capture.os,
                    "fsync",
                    wraps=bounded_capture.os.fsync,
                ) as fsync,
            ):
                bounded_capture.atomic_write_bytes(path, payload)
            content = path.read_bytes()
            temporary_files = list(root.glob(".stdout.log.*.tmp"))

        self.assertEqual(content, payload)
        fsync.assert_called_once()
        self.assertEqual(temporary_files, [])

    def test_atomic_log_replace_failure_preserves_existing_log(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            bounded_capture,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "stdout.log"
            path.write_bytes(b"existing")
            with (
                patch.object(
                    bounded_capture.os,
                    "replace",
                    side_effect=OSError("synthetic replace failure"),
                ),
                self.assertRaisesRegex(
                    OSError,
                    "synthetic replace failure",
                ),
            ):
                bounded_capture.atomic_write_bytes(path, b"replacement")
            content = path.read_bytes()
            temporary_files = list(root.glob(".stdout.log.*.tmp"))

        self.assertEqual(content, b"existing")
        self.assertEqual(temporary_files, [])

    def test_process_tree_termination_shares_one_grace_deadline(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            system_probe as runtime_probe,
        )

        class FakeProcess:
            def __init__(self, pid: int, children: list[object] | None = None) -> None:
                self.pid = pid
                self._children = children or []
                self.terminated = False
                self.killed = False

            def children(self, recursive: bool = False) -> list[object]:
                return list(self._children)

            def terminate(self) -> None:
                self.terminated = True

            def kill(self) -> None:
                self.killed = True

            def is_running(self) -> bool:
                return True

        class FakePsutil:
            class NoSuchProcess(Exception):
                pass

            class AccessDenied(Exception):
                pass

            def __init__(self, root: FakeProcess, clock: FakeClock) -> None:
                self.root = root
                self.clock = clock
                self.wait_calls = 0

            def Process(self, pid: int) -> FakeProcess:
                self.assert_pid = pid
                return self.root

            def wait_procs(
                self,
                processes: list[FakeProcess],
                *,
                timeout: float,
            ) -> tuple[list[FakeProcess], list[FakeProcess]]:
                self.wait_calls += 1
                self.clock.sleep(timeout)
                if self.wait_calls == 1:
                    return [], list(processes)
                return list(processes), []

        clock = FakeClock()
        child = FakeProcess(2)
        root = FakeProcess(1, [child])
        fake_psutil = FakePsutil(root, clock)
        with patch.object(
            runtime_probe,
            "psutil",
            fake_psutil,
        ), patch.object(
            runtime_probe.time,
            "monotonic",
            side_effect=clock.monotonic,
        ):
            survivors = runtime_probe.terminate_process_tree(1, 1.0)

        self.assertEqual(survivors, ())
        self.assertLessEqual(clock.now, 1.0)
        self.assertTrue(root.terminated)
        self.assertTrue(child.terminated)
        self.assertTrue(root.killed)
        self.assertTrue(child.killed)

    def test_process_tree_termination_reports_access_denied_survivor(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            system_probe as runtime_probe,
        )

        class FakeProcess:
            def __init__(
                self,
                pid: int,
                *,
                denied: bool = False,
                children: list[object] | None = None,
            ) -> None:
                self.pid = pid
                self.denied = denied
                self._children = children or []

            def children(self, recursive: bool = False) -> list[object]:
                return list(self._children)

            def terminate(self) -> None:
                if self.denied:
                    raise fake_psutil.AccessDenied()

            def kill(self) -> None:
                if self.denied:
                    raise fake_psutil.AccessDenied()

            def is_running(self) -> bool:
                if self.denied:
                    raise fake_psutil.AccessDenied()
                return False

        class FakePsutil:
            class NoSuchProcess(Exception):
                pass

            class AccessDenied(Exception):
                pass

            def __init__(self, root: FakeProcess) -> None:
                self.root = root

            def Process(self, pid: int) -> FakeProcess:
                return self.root

            def wait_procs(
                self,
                processes: list[FakeProcess],
                *,
                timeout: float,
            ) -> tuple[list[FakeProcess], list[FakeProcess]]:
                return [], list(processes)

        child = FakeProcess(2, denied=True)
        root = FakeProcess(1, children=[child])
        fake_psutil = FakePsutil(root)
        with patch.object(runtime_probe, "psutil", fake_psutil):
            survivors = runtime_probe.terminate_process_tree(1, 0.01)

        self.assertEqual(survivors, (2,))

    def test_process_tree_rss_access_denied_fails_closed(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime import (
            system_probe as runtime_probe,
        )

        class FakeProcess:
            def __init__(
                self,
                pid: int,
                *,
                denied: bool = False,
                children: list[object] | None = None,
            ) -> None:
                self.pid = pid
                self.denied = denied
                self._children = children or []

            def children(self, recursive: bool = False) -> list[object]:
                return list(self._children)

            def memory_info(self) -> object:
                if self.denied:
                    raise fake_psutil.AccessDenied()
                return SimpleNamespace(rss=1024)

        class FakePsutil:
            class NoSuchProcess(Exception):
                pass

            class AccessDenied(Exception):
                pass

            def __init__(self, root: FakeProcess) -> None:
                self.root = root

            def Process(self, pid: int) -> FakeProcess:
                return self.root

        child = FakeProcess(2, denied=True)
        root = FakeProcess(1, children=[child])
        fake_psutil = FakePsutil(root)
        probe = runtime_probe.SystemProbe.__new__(runtime_probe.SystemProbe)
        with patch.object(runtime_probe, "psutil", fake_psutil):
            measured = probe.process_tree_rss_bytes(1)

        self.assertEqual(measured, 0)


if __name__ == "__main__":
    unittest.main()
