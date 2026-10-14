from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from http import HTTPStatus
from io import BytesIO
from pathlib import Path
from typing import Iterator
from unittest.mock import patch
from urllib.parse import urlencode

from simulation_ready_cad_workflow.validation_pipeline.dashboard.server import (
    DashboardHandler,
    DashboardServer,
)


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
    # Candidate release tests run without socket permission.  Exercise the
    # production server/handler methods directly, as the older dashboard
    # server tests do, rather than weakening that sandbox.
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
    server._change_probe_interval_s = 0.01
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


class DashboardChangeWatchTests(unittest.TestCase):
    def test_change_watch_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            with _dashboard(root) as server:
                # One risk-focused target covers the complete contract so the
                # release suite does not multiply near-identical server tests.
                initial = server.dashboard_change_snapshot(run_id)

                _write_json(
                    run / "run_status.json",
                    {"status": "RUNNING", "step": 22},
                )
                status_changed = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    status_changed["catalog_token"],
                    initial["catalog_token"],
                )
                self.assertNotEqual(
                    status_changed["run_token"],
                    initial["run_token"],
                )

                _write_json(
                    run / "run_status.json",
                    {"status": "NOT_CERTIFIED", "step": 22},
                )
                terminal_changed = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(
                    terminal_changed["catalog_token"],
                    status_changed["catalog_token"],
                )
                _write_json(
                    run / "run_status.json",
                    {"status": "RUNNING", "step": 23},
                )
                status_changed = server.dashboard_change_snapshot(run_id)

                _write_json(
                    run / "runtime_queue" / "physical" / "queue_state.json",
                    {"state": "RUNNING", "completed": 1},
                )
                queue_changed = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    queue_changed["catalog_token"],
                    status_changed["catalog_token"],
                )
                self.assertNotEqual(
                    queue_changed["run_token"],
                    status_changed["run_token"],
                )

                _write_json(
                    run
                    / "runtime_queue"
                    / "physical"
                    / "jobs"
                    / "a"
                    / "attempt-001.result.json",
                    {"status": "passed"},
                )
                result_changed = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    result_changed["catalog_token"],
                    queue_changed["catalog_token"],
                )
                self.assertNotEqual(
                    result_changed["run_token"],
                    queue_changed["run_token"],
                )

                authored_test = (
                    run / "planning" / "authored_tests" / "travel.json"
                )
                _write_json(authored_test, {"measure": "travel_m"})
                authored_created = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    authored_created["catalog_token"],
                    result_changed["catalog_token"],
                )
                self.assertNotEqual(
                    authored_created["run_token"],
                    result_changed["run_token"],
                )

                _write_json(
                    authored_test,
                    {"measure": "tampered_travel_m"},
                )
                authored_tampered = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    authored_tampered["catalog_token"],
                    authored_created["catalog_token"],
                )
                self.assertNotEqual(
                    authored_tampered["run_token"],
                    authored_created["run_token"],
                )

                _write_json(
                    run / "inputs" / "task" / "design.json",
                    {"revision": 1},
                )
                input_changed = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    input_changed["catalog_token"],
                    authored_tampered["catalog_token"],
                )
                self.assertNotEqual(
                    input_changed["run_token"],
                    authored_tampered["run_token"],
                )

                react_state_path = (
                    run.parent.parent / "react_loop" / "state.json"
                )
                _write_json(
                    react_state_path,
                    {"phase": "DESIGN", "turn_index": 4},
                )
                react_state_changed = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    react_state_changed["catalog_token"],
                    input_changed["catalog_token"],
                )
                self.assertNotEqual(
                    react_state_changed["run_token"],
                    input_changed["run_token"],
                )

                _write_json(
                    react_state_path,
                    {"phase": "HARNESS", "turn_index": 5},
                )
                react_state_advanced = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    react_state_advanced["catalog_token"],
                    react_state_changed["catalog_token"],
                )
                self.assertNotEqual(
                    react_state_advanced["run_token"],
                    react_state_changed["run_token"],
                )

                step_path = run / "inputs" / "cad" / "assembly.step"
                step_path.parent.mkdir(parents=True, exist_ok=True)
                step_path.write_text(
                    "ISO-10303-21;\nEND-ISO-10303-21;\n",
                    encoding="utf-8",
                )
                step_changed = server.dashboard_change_snapshot(run_id)
                self.assertEqual(
                    step_changed["catalog_token"],
                    input_changed["catalog_token"],
                )
                self.assertNotEqual(
                    step_changed["run_token"],
                    input_changed["run_token"],
                )

                _write_json(
                    run / "design_quality" / "review" / "evaluation.json",
                    {"status": "EVIDENCE_COMPLETE"},
                )
                design_quality_changed = server.dashboard_change_snapshot(
                    run_id
                )
                self.assertEqual(
                    design_quality_changed["catalog_token"],
                    step_changed["catalog_token"],
                )
                self.assertNotEqual(
                    design_quality_changed["run_token"],
                    step_changed["run_token"],
                )

                next_run = run.parent / "r02"
                next_run.mkdir()
                revision_created = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(
                    revision_created["catalog_token"],
                    design_quality_changed["catalog_token"],
                )
                self.assertEqual(
                    revision_created["run_token"],
                    design_quality_changed["run_token"],
                )

                _write_json(
                    next_run / "run_status.json",
                    {"status": "PREPARING"},
                )
                revision_published = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(
                    revision_published["catalog_token"],
                    revision_created["catalog_token"],
                )

                _write_json(
                    root
                    / ".runtime_runs"
                    / "project_usage"
                    / "projects"
                    / "machine.json",
                    {"schema_version": "0.1", "total_tokens": 1234},
                )
                usage_changed = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(
                    usage_changed["token"],
                    revision_published["token"],
                )

                outside = root / "outside.json"
                _write_json(outside, {"value": 1})
                (run / "outside-link.json").symlink_to(outside)
                before_outside_change = server.dashboard_change_snapshot(run_id)
                _write_json(outside, {"value": 222222})
                self.assertEqual(
                    server.dashboard_change_snapshot(run_id),
                    before_outside_change,
                )

                initial_status, initial_body, _headers = _get(
                    server,
                    "/api/changes?" + urlencode({"run": run_id}),
                )
                self.assertEqual(initial_status, HTTPStatus.OK)
                initial = json.loads(initial_body)
                self.assertFalse(initial["changed"])

                watch_path = "/api/changes?" + urlencode(
                    {
                        "run": run_id,
                        "since": initial["token"],
                        "timeout": "2",
                    }
                )
                with ThreadPoolExecutor(max_workers=1) as executor:
                    response = executor.submit(_get, server, watch_path)
                    _write_json(
                        run / "runtime_queue" / "physical" / "queue_state.json",
                        {"state": "RUNNING", "completed": 999},
                    )
                    status, body, _headers = response.result(timeout=3)

                self.assertEqual(status, HTTPStatus.OK)
                changed = json.loads(body)
                self.assertTrue(changed["changed"])
                self.assertTrue(changed["run_changed"])
                self.assertFalse(changed["catalog_changed"])
                self.assertNotEqual(changed["token"], initial["token"])

                token = changed["token"]
                with patch(
                    "simulation_ready_cad_workflow.validation_pipeline.dashboard.server.scan_runs"
                ) as scan_runs, patch(
                    "simulation_ready_cad_workflow.validation_pipeline.dashboard.server.normalize_run"
                ) as normalize_run:
                    status, body, headers = _get(
                        server,
                        "/api/changes?"
                        + urlencode(
                            {
                                "run": run_id,
                                "since": token,
                                "timeout": "0.05",
                            }
                        ),
                    )
                    self.assertEqual(status, HTTPStatus.NO_CONTENT)
                    self.assertEqual(body, b"")
                    self.assertEqual(
                        headers["X-Dashboard-Change-Token"],
                        token,
                    )
                    scan_runs.assert_not_called()
                    normalize_run.assert_not_called()

                status, _body, _headers = _get(
                    server,
                    "/api/changes?" + urlencode({"run": "../outside"}),
                )
                self.assertEqual(status, HTTPStatus.NOT_FOUND)

                status, _body, _headers = _get(
                    server,
                    "/api/changes?"
                    + urlencode(
                        {"run": run_id, "since": "attacker-token"}
                    ),
                )
                self.assertEqual(status, HTTPStatus.BAD_REQUEST)

                status, _body, _headers = _get(
                    server,
                    "/api/changes?"
                    + urlencode({"run": run_id, "timeout": "31"}),
                )
                self.assertEqual(status, HTTPStatus.BAD_REQUEST)

                current_catalog_token = server.dashboard_change_snapshot(run_id)[
                    "catalog_token"
                ]
                forged_token = (
                    "0" * 64
                    if current_catalog_token != "0" * 64
                    else "1" * 64
                )
                with patch.object(server, "_scan_runs_catalog") as scan_catalog:
                    status, _body, _headers = _get(
                        server,
                        "/api/runs?"
                        + urlencode(
                            {"fresh": "1", "change": forged_token}
                        ),
                    )
                    self.assertEqual(status, HTTPStatus.CONFLICT)
                    scan_catalog.assert_not_called()

                stale_catalog_token = current_catalog_token
                (run.parent / "r03").mkdir()
                current_catalog_token = server.dashboard_change_snapshot(run_id)[
                    "catalog_token"
                ]
                self.assertNotEqual(
                    current_catalog_token,
                    stale_catalog_token,
                )
                with patch.object(server, "_scan_runs_catalog") as scan_catalog:
                    status, _body, _headers = _get(
                        server,
                        "/api/runs?"
                        + urlencode(
                            {"fresh": "1", "change": stale_catalog_token}
                        ),
                    )
                    self.assertEqual(status, HTTPStatus.CONFLICT)
                    scan_catalog.assert_not_called()

                catalog_token = current_catalog_token
                with patch.object(
                    server,
                    "_scan_runs_catalog",
                    return_value=[{"id": run_id}],
                ) as scan_catalog:
                    first = server.fresh_runs_for_catalog_change(catalog_token)
                    second = server.fresh_runs_for_catalog_change(catalog_token)

                self.assertEqual(first, [{"id": run_id}])
                self.assertEqual(second, first)
                scan_catalog.assert_called_once_with()

                app_source = (
                    Path(__file__).resolve().parent.parent
                    / "dashboard"
                    / "static"
                    / "app.js"
                ).read_text(encoding="utf-8")
                self.assertIn("/changes?", app_source)
                self.assertIn("/runs?fresh=1&change=", app_source)
                self.assertNotIn("setInterval(refreshLive", app_source)
                watch_source = app_source[
                    app_source.index("async function watchLiveChanges") :
                    app_source.index("function startLiveWatch")
                ]
                refresh_call = watch_source.index(
                    "const refreshed = await refreshLive(change);"
                )
                acknowledgement = watch_source.index(
                    "since = change.token || since;",
                    refresh_call,
                )
                self.assertLess(refresh_call, acknowledgement)

    def test_running_lease_expiry_changes_tokens_without_file_write(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            heartbeat = datetime(
                2026,
                8,
                1,
                tzinfo=timezone.utc,
            )
            heartbeat_epoch = heartbeat.timestamp()
            status_path = run / "run_status.json"
            _write_json(
                status_path,
                {
                    "status": "RUNNING",
                    "running_lease": {
                        "heartbeat_at": heartbeat.isoformat().replace(
                            "+00:00", "Z"
                        ),
                        "ttl_seconds": 30,
                    },
                },
            )
            status_metadata = status_path.stat()

            with _dashboard(root) as server:
                # A long generic cache TTL proves that the lease deadline,
                # rather than another filesystem write, invalidates tokens.
                server._catalog_change_cache_ttl_s = 3600.0
                with patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "dashboard.server.time.time",
                    return_value=heartbeat_epoch + 29,
                ):
                    before_expiry = server.dashboard_change_snapshot(run_id)

                with patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "dashboard.server.time.time",
                    return_value=heartbeat_epoch + 31,
                ):
                    after_expiry = server.dashboard_change_snapshot(run_id)

            self.assertEqual(status_path.stat(), status_metadata)
            self.assertNotEqual(
                after_expiry["catalog_token"],
                before_expiry["catalog_token"],
            )
            self.assertNotEqual(
                after_expiry["run_token"],
                before_expiry["run_token"],
            )

    def test_concurrent_clients_share_one_run_probe_watcher(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _run, run_id = _canonical_run(root)
            with _dashboard(root) as server:
                initial = {
                    "catalog_token": "a" * 64,
                    "run_token": "b" * 64,
                    "token": f"v1.{'a' * 64}.{'b' * 64}",
                }
                changed = {
                    "catalog_token": "a" * 64,
                    "run_token": "c" * 64,
                    "token": f"v1.{'a' * 64}.{'c' * 64}",
                }
                clients_ready = threading.Event()
                release_change = threading.Event()
                calls_lock = threading.Lock()
                client_threads: set[int] = set()
                watcher_threads: set[int] = set()

                def snapshot(_run_id: str) -> dict[str, str]:
                    current = threading.current_thread()
                    with calls_lock:
                        if current.name == "dashboard-change-watch":
                            watcher_threads.add(current.ident or 0)
                        else:
                            client_threads.add(current.ident or 0)
                            if len(client_threads) == 2:
                                clients_ready.set()
                    return changed if release_change.is_set() else initial

                with patch.object(
                    server,
                    "dashboard_change_snapshot",
                    side_effect=snapshot,
                ):
                    with ThreadPoolExecutor(max_workers=2) as executor:
                        futures = [
                            executor.submit(
                                server.wait_for_dashboard_change,
                                run_id=run_id,
                                since=initial["token"],
                                timeout_s=2.0,
                            )
                            for _index in range(2)
                        ]
                        self.assertTrue(clients_ready.wait(timeout=1.0))
                        release_change.set()
                        results = [future.result(timeout=2.0) for future in futures]

                self.assertEqual(len(watcher_threads), 1)
                self.assertTrue(all(result is not None for result in results))
                self.assertTrue(all(result[2] for result in results if result))

    def test_catalog_probe_skips_symlinked_catalog_directories(self) -> None:
        for symlink_kind in ("projects", "revisions"):
            with self.subTest(symlink_kind=symlink_kind):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    runtime = root / ".runtime_runs"
                    selected = runtime / "selected-legacy"
                    _write_json(
                        selected / "run_status.json",
                        {"status": "RUNNING"},
                    )
                    outside = root / f"outside-{symlink_kind}"
                    outside.mkdir()
                    if symlink_kind == "projects":
                        (runtime / "projects").symlink_to(
                            outside,
                            target_is_directory=True,
                        )
                    else:
                        project = runtime / "projects" / "other-project"
                        _write_json(
                            project / "project.json",
                            {"project_id": "other-project"},
                        )
                        (project / "runs").symlink_to(
                            outside,
                            target_is_directory=True,
                        )

                    with _dashboard(root) as server:
                        before = server.catalog_change_token(force=True)
                        _write_json(
                            outside / "must-not-be-scanned" / "run_status.json",
                            {"status": "TASK_CERTIFIED"},
                        )
                        after = server.catalog_change_token(force=True)

                    self.assertEqual(after, before)

    def test_catalog_probe_is_bounded_and_selected_scope_is_cached(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, run_id = _canonical_run(root)
            for index in range(24):
                _write_json(
                    run.parent / f"r{index + 2:02d}" / "run_status.json",
                    {"status": "NOT_CERTIFIED"},
                )
                _write_json(
                    root
                    / ".runtime_runs"
                    / "projects"
                    / f"other-{index:02d}"
                    / "runs"
                    / "r01"
                    / "run_status.json",
                    {"status": "NOT_CERTIFIED"},
                )

            with _dashboard(root) as server, patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "dashboard.server._CATALOG_CHANGE_MAX_SCANNED_ENTRIES",
                12,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "dashboard.server._CATALOG_CHANGE_MAX_PROJECTS",
                4,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "dashboard.server._CATALOG_CHANGE_MAX_REVISIONS",
                4,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "dashboard.server._CATALOG_CHANGE_MAX_SCOPED_REVISIONS",
                4,
            ):
                server._catalog_change_cache_ttl_s = 60.0
                with patch.object(
                    server,
                    "_compute_catalog_change_token",
                    wraps=server._compute_catalog_change_token,
                ) as compute:
                    before = server.dashboard_change_snapshot(run_id)
                    cached = server.dashboard_change_snapshot(run_id)
                    self.assertEqual(compute.call_count, 1)
                self.assertEqual(cached, before)

                # Even when sibling enumeration is capped, the selected
                # project's runs-directory stat detects revision creation.
                (run.parent / "r99").mkdir()
                server._catalog_change_cache_ttl_s = 0.0
                after = server.dashboard_change_snapshot(run_id)
                self.assertNotEqual(
                    after["catalog_token"],
                    before["catalog_token"],
                )


if __name__ == "__main__":
    unittest.main()
