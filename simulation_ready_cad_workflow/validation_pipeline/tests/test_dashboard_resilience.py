"""Dashboard resilience: crash recovery, corrupt inputs, frozen runs.

Locks the acceptance behaviors that the change-watch contract tests do
not cover:

* the server survives partial/corrupt state files and its own restart
  (client tokens from a previous process keep working);
* serving a frozen historical run modifies nothing on disk;
* local refresh latency is measured, not just claimed.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Iterator
from urllib.parse import urlencode

from simulation_ready_cad_workflow.validation_pipeline.contracts.traces import (
    append_maintenance_record,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.server import (
    DashboardHandler,
    DashboardServer,
)

OPERATOR = "operator-secret-identity-93b1"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _canonical_run(root: Path) -> tuple[Path, str]:
    project = root / ".runtime_runs" / "projects" / "machine"
    _write_json(
        project / "project.json",
        {"project_id": "machine", "package_id": "machine"},
    )
    run = project / "runs" / "r01"
    _write_json(run / "run_status.json", {"status": "RUNNING", "step": 1})
    return run, run.relative_to(root).as_posix()


@contextmanager
def _dashboard(root: Path) -> Iterator[DashboardServer]:
    server = object.__new__(DashboardServer)
    server.base_path = ""
    server.workspace_root = root.resolve()
    server.runs_root = (root / ".runtime_runs").resolve()
    server._runs_cache_lock = threading.Lock()
    server._runs_cache = None
    server._runs_cache_completed_at = 0.0
    server._runs_cache_ttl_s = 10.0
    server._runs_cache_refreshing = False
    server._runs_fresh_lock = threading.Lock()
    server._runs_cache_catalog_token = None
    server._change_probe_interval_s = 0.02
    server._change_watch_condition = threading.Condition()
    server._change_watch_stop = threading.Event()
    server._change_watch_thread = None
    server._change_watch_refcounts = {}
    server._change_watch_snapshots = {}
    server._catalog_change_cache_lock = threading.Lock()
    server._catalog_change_compute_lock = threading.Lock()
    server._catalog_change_cache_ttl_s = 0.0
    server._catalog_change_cache = {}
    server._catalog_change_issued_scopes = {}
    server._catalog_status_projection_cache = {}
    try:
        yield server
    finally:
        server._stop_dashboard_change_watcher()


def _get(
    server: DashboardServer,
    path: str,
) -> tuple[int, bytes, dict[str, str]]:
    handler = object.__new__(DashboardHandler)
    handler.server = server
    handler.path = path
    handler.command = "GET"
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.close_connection = True
    handler.client_address = ("127.0.0.1", 0)
    handler.wfile = BytesIO()
    handler.do_GET()
    raw = handler.wfile.getvalue()
    head, _, body = raw.partition(b"\r\n\r\n")
    lines = head.decode("iso-8859-1").split("\r\n")
    status = int(lines[0].split(" ", 2)[1])
    headers = {
        name: value.strip()
        for line in lines[1:]
        if ":" in line
        for name, value in [line.split(":", 1)]
    }
    return status, body, headers


class WorkerPauseTest(unittest.TestCase):
    def test_repeated_error_pause_stops_worker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            state_path = run.parent.parent / "react_loop" / "state.json"
            _write_json(state_path, {"status": "REPEATED_ERROR_PAUSED"})
            with _dashboard(root) as server:
                status, body, _headers = _get(
                    server, "/api/run?" + urlencode({"id": run_id})
                )
                self.assertEqual(status, HTTPStatus.OK)
                json.loads(body)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "REPEATED_ERROR_PAUSED")


class ReuseDisplayTest(unittest.TestCase):
    def test_run_payload_exposes_reuse_and_mesh_cache_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            _write_json(
                run / "results" / "reuse_manifest.json",
                {
                    "schema_version": "mechanism_reuse_manifest_v1",
                    "lane_jobs": [
                        {"job_id": "a", "fingerprint": "f1", "action": "EXECUTED"},
                        {"job_id": "b", "fingerprint": "f2", "action": "EXECUTED"},
                    ],
                    "mesh_cache": {
                        "capsule": {"hits": 155, "misses": 0, "stores": 0}
                    },
                },
            )
            with _dashboard(root) as server:
                status, body, _headers = _get(
                    server, "/api/run?" + urlencode({"id": run_id})
                )
                self.assertEqual(status, HTTPStatus.OK)
                payload = json.loads(body)
                reuse = payload["reuse"]
                self.assertTrue(reuse["recorded"])
                self.assertEqual(
                    reuse["mesh_cache"]["capsule"]["hits"], 155
                )
                self.assertEqual(reuse["lane_actions"]["EXECUTED"], 2)
                self.assertEqual(reuse["lane_fingerprint_count"], 2)


class CrashRecoveryTest(unittest.TestCase):
    def test_tokens_survive_a_server_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            with _dashboard(root) as first:
                status, body, _headers = _get(
                    first, "/api/changes?" + urlencode({"run": run_id})
                )
                self.assertEqual(status, HTTPStatus.OK)
                token = json.loads(body)["token"]

            # Process "crash": the first server object is gone; a new one
            # rebuilds every token from on-disk metadata.
            with _dashboard(root) as second:
                status, body, headers = _get(
                    second,
                    "/api/changes?"
                    + urlencode(
                        {"run": run_id, "since": token, "timeout": "0.05"}
                    ),
                )
                self.assertEqual(status, HTTPStatus.NO_CONTENT)
                self.assertEqual(
                    headers["X-Dashboard-Change-Token"], token
                )

                _write_json(
                    run / "run_status.json",
                    {"status": "RUNNING", "step": 2},
                )
                status, body, _headers = _get(
                    second,
                    "/api/changes?"
                    + urlencode(
                        {"run": run_id, "since": token, "timeout": "2"}
                    ),
                )
                self.assertEqual(status, HTTPStatus.OK)
                self.assertTrue(json.loads(body)["changed"])

    def test_partial_and_corrupt_files_never_crash_the_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            with _dashboard(root) as server:
                before = server.dashboard_change_snapshot(run_id)

                # Simulate a crash mid-write: truncated JSON on disk.
                (run / "run_status.json").write_text(
                    '{"status": "RUNN', encoding="utf-8"
                )
                after = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(
                    after["run_token"], before["run_token"]
                )

                status, body, _headers = _get(
                    server, "/api/run?" + urlencode({"id": run_id})
                )
                self.assertIn(
                    status, {HTTPStatus.OK, HTTPStatus.NOT_FOUND}
                )
                json.loads(body)  # structured response either way

                # Repair completes the write; the page recovers.
                _write_json(
                    run / "run_status.json",
                    {"status": "RUNNING", "step": 3},
                )
                status, body, _headers = _get(
                    server, "/api/run?" + urlencode({"id": run_id})
                )
                self.assertEqual(status, HTTPStatus.OK)


class ReadOnlyHistoryTest(unittest.TestCase):
    def _tree_signature(self, root: Path) -> dict[str, tuple[int, int]]:
        signature: dict[str, tuple[int, int]] = {}
        for path in sorted(root.rglob("*")):
            if path.is_file():
                stat = path.stat()
                signature[str(path)] = (stat.st_mtime_ns, stat.st_size)
        return signature

    def test_serving_a_frozen_run_modifies_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            _write_json(
                run / "run_status.json",
                {"status": "TASK_CERTIFIED", "last_completed_stage": "done"},
            )
            project_dir = run.parent.parent
            append_maintenance_record(
                project_dir,
                actor=OPERATOR,
                intervention_kind="regression_run",
                reason="full suite before freeze",
            )
            before = self._tree_signature(project_dir)
            with _dashboard(root) as server:
                for path in (
                    "/",
                    "/api/run?" + urlencode({"id": run_id}),
                    "/api/changes?" + urlencode({"run": run_id}),
                ):
                    status, _body, _headers = _get(server, path)
                    self.assertIn(
                        status, {HTTPStatus.OK, HTTPStatus.NO_CONTENT}
                    )
            self.assertEqual(self._tree_signature(project_dir), before)


class RefreshLatencyTest(unittest.TestCase):
    def test_local_refresh_latency_is_measured_and_bounded(self) -> None:
        """Measured write-to-notification latency, not a claim.

        The production probe interval is 0.20 s; the contract here is
        that a change on disk is visible to a waiting client within one
        second at test probe granularity (0.02 s).
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            latencies: list[float] = []
            with _dashboard(root) as server:
                for round_index in range(3):
                    snapshot = server.dashboard_change_snapshot(run_id)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        waiter = executor.submit(
                            server.wait_for_dashboard_change,
                            run_id=run_id,
                            since=snapshot["token"],
                            timeout_s=5.0,
                        )
                        time.sleep(0.05)
                        started = time.monotonic()
                        _write_json(
                            run / "run_status.json",
                            {"status": "RUNNING", "step": 10 + round_index},
                        )
                        result = waiter.result(timeout=6.0)
                        latencies.append(time.monotonic() - started)
                    self.assertIsNotNone(result)
            latencies.sort()
            median = latencies[len(latencies) // 2]
            self.assertLess(
                median,
                1.0,
                f"median refresh latency {median:.3f}s exceeds 1s "
                f"(all: {[round(v, 3) for v in latencies]})",
            )


if __name__ == "__main__":
    unittest.main()
