"""HTTP contract for starting, stopping and resuming runs from the dashboard."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from io import BytesIO
from unittest import mock

import psutil

from simulation_ready_cad_workflow.validation_pipeline.dashboard import cli, launcher
from simulation_ready_cad_workflow.validation_pipeline.dashboard import server as server_module
from simulation_ready_cad_workflow.validation_pipeline.dashboard.server import DashboardHandler, DashboardServer

from .test_dashboard_changes import _dashboard


TOKEN = "launch-test-capability"
CAD_PYTHON = "/opt/cad/bin/python"
AUTH = {"X-Human-Decision-Capability": TOKEN, "Content-Type": "application/json"}


def _workspace(root: Path) -> None:
    (root / ".runtime_runs" / "projects").mkdir(parents=True)


@contextmanager
def _launch_server():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary).resolve()
        _workspace(root)
        with mock.patch.object(launcher, "CODE_ROOT", root), mock.patch.dict(
            os.environ, {"SIMREADY_CAD_PYTHON": CAD_PYTHON}
        ), _dashboard(root) as server:
            server.launch_enabled = True
            server.launch_limit = 1
            server.human_decision_capability = TOKEN
            server._runs_overview_lock = threading.Lock()
            server._runs_overview_cache = None
            yield root, server


def _call(server, method: str, path: str, body=None, headers=None) -> tuple[int, dict]:
    payload = b"" if body is None else body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    handler = object.__new__(DashboardHandler)
    handler.server = server
    handler.path = path
    handler.command = method
    handler.request_version = "HTTP/1.1"
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.close_connection = True
    handler.client_address = ("127.0.0.1", 0)
    handler.headers = {"Content-Length": str(len(payload)), "Host": "127.0.0.1:8765", **(headers or {})}
    handler.rfile = BytesIO(payload)
    handler.wfile = BytesIO()
    getattr(handler, "do_" + method)()
    head, _, response = handler.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    try:
        return status, json.loads(response)
    except ValueError:
        return status, {"raw": response.decode("utf-8", "replace")}


class _FakeProcess:
    """Popen double: optionally creates the project the CLI would create."""

    calls: list["_FakeProcess"] = []

    def __init__(self, argv, *, creates=True, exit_code=None, output=b"", heartbeat=False, **kwargs):
        self.argv, self.kwargs = argv, kwargs
        self.pid = os.getpid()
        self.returncode = exit_code
        kwargs["stdout"].write(output)
        kwargs["stdout"].flush()
        workspace = Path(kwargs["cwd"])
        if creates and "--project" in argv:
            project = workspace / ".runtime_runs/projects" / argv[argv.index("--project") + 1]
            (project / "runs/r01").mkdir(parents=True)
            (project / "workflow.json").write_text('{"schema_version": "slim_harness_workflow_v1"}', encoding="utf-8")
        if heartbeat:
            project = workspace / ".runtime_runs/projects" / argv[-1]
            (project / "workflow_runtime.json").write_text(json.dumps(
                {"status": "stopped", "heartbeat_at": "2999-01-01T00:00:00+00:00"}), encoding="utf-8")
        _FakeProcess.calls.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


@contextmanager
def _popen(**behaviour):
    _FakeProcess.calls = []
    with mock.patch.object(launcher.subprocess, "Popen", lambda argv, **kwargs: _FakeProcess(argv, **behaviour, **kwargs)):
        yield _FakeProcess.calls


class LaunchCreateTest(unittest.TestCase):
    def test_free_launch_spawns_the_exact_cli_detached_and_records_it(self) -> None:
        task = "Design a crank-driven gear reducer.\n"
        with _launch_server() as (root, server), _popen() as calls, mock.patch.dict(
            os.environ, {"SIMREADY_CONSTRUCTION_PROJECT": "outer", "SIMREADY_TRAJECTORY_TURN": "t1"}
        ):
            status, payload = _call(server, "POST", "/api/launch", {
                "mode": "free", "project": "demo", "task": task, "provider": "codex", "model": "gpt-test",
                "reasoning": "medium", "tier": "light", "budget_hours": 2,
            }, AUTH)
            self.assertEqual(status, 200, payload)
            self.assertEqual(payload["run_id"], ".runtime_runs/projects/demo/runs/r01")
            [launch] = list((root / ".runtime_runs/dashboard_launches").iterdir())
            self.assertTrue(launch.name.startswith("demo-"))
            self.assertEqual((launch / "task.md").read_text(encoding="utf-8"), task)
            [process] = calls
            self.assertEqual(process.argv, [
                CAD_PYTHON, "-X", "utf8", "-m", "simulation_ready_cad_workflow.validation_pipeline.slim_harness",
                "--workspace", str(root), "--worker-provider", "codex", "--model", "gpt-test", "--reasoning", "medium",
                "--tier", "light", "create", "--mode", "free", "--project", "demo", "--construction", "--task", str(launch / "task.md"),
                "--budget-hours", "2",
            ])
            self.assertEqual(process.kwargs["cwd"], root)
            self.assertIs(process.kwargs["stdin"], subprocess.DEVNULL)
            self.assertIs(process.kwargs["stderr"], subprocess.STDOUT)
            self.assertEqual(process.kwargs["stdout"].name, str(launch / "launch.log"))
            self.assertTrue(process.kwargs["start_new_session"])
            env = process.kwargs["env"]
            self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], str(root))
            self.assertFalse([key for key in env if key.startswith(("SIMREADY_CONSTRUCTION_", "SIMREADY_TRAJECTORY_"))])
            record = json.loads((launch / "launch.json").read_text(encoding="utf-8"))
            self.assertEqual(record["argv"], process.argv)
            self.assertEqual(record["pid"], os.getpid())
            self.assertAlmostEqual(record["create_time"], psutil.Process().create_time(), places=2)
            self.assertEqual(record["kind"], "create")

    def test_free_mode_rules_are_validated_before_spawning(self) -> None:
        cases = [
            ({"mode": "free", "project": "p"}, "LAUNCH_TASK_REQUIRED"),
            ({"mode": "free", "project": "p", "task": "   "}, "LAUNCH_TASK_REQUIRED"),
            ({"mode": "solo", "project": "p", "task": "x"}, "LAUNCH_MODE_UNAVAILABLE"),
            ({"mode": "benchmark", "project": "p", "task": "x"}, "LAUNCH_MODE_UNAVAILABLE"),
            ({"mode": "free", "project": "p", "task": "x", "family": "BM01"}, "LAUNCH_FIELD_UNKNOWN"),
            ({"mode": "free", "project": "../p", "task": "x"}, "LAUNCH_PROJECT_INVALID"),
            ({"mode": "free", "project": " p", "task": "x"}, "LAUNCH_PROJECT_INVALID"),
            ({"mode": "free", "project": "p", "task": "x", "budget_hours": 4}, "LAUNCH_FIELD_INVALID"),
            ({"mode": "free", "project": "p", "task": "x", "budget_hours": True}, "LAUNCH_FIELD_INVALID"),
            ({"mode": "free", "project": "p", "task": "x", "model": "--danger"}, "LAUNCH_FIELD_INVALID"),
            ({"mode": "free", "project": "p", "task": "x", "provider": "nobody"}, "LAUNCH_FIELD_INVALID"),
            ({"mode": "free", "project": "p", "task": "x", "executable": "sh"}, "LAUNCH_FIELD_UNKNOWN"),
        ]
        with _launch_server() as (root, server), _popen() as calls:
            for body, code in cases:
                with self.subTest(body=body):
                    status, payload = _call(server, "POST", "/api/launch", body, AUTH)
                    self.assertEqual((status, payload.get("error")), (400, code), payload)
            self.assertEqual(calls, [])
            self.assertFalse((root / ".runtime_runs/dashboard_launches").exists())

    def test_existing_project_is_never_relaunched(self) -> None:
        with _launch_server() as (root, server), _popen() as calls:
            (root / ".runtime_runs/projects/demo").mkdir()
            status, payload = _call(server, "POST", "/api/launch", {"mode": "free", "project": "demo", "task": "x"}, AUTH)
            self.assertEqual((status, payload["error"]), (409, "LAUNCH_PROJECT_EXISTS"))
            self.assertEqual(calls, [])

    def test_one_dashboard_run_at_a_time_by_default(self) -> None:
        with _launch_server() as (root, server), _popen() as calls:
            # The double reports this live test process as the child.
            self.assertEqual(_call(server, "POST", "/api/launch", {"mode": "free", "project": "one", "task": "x"}, AUTH)[0], 200)
            status, payload = _call(server, "POST", "/api/launch", {"mode": "free", "project": "two", "task": "x"}, AUTH)
            self.assertEqual((status, payload["error"]), (409, "LAUNCH_LIMIT"))
            self.assertEqual(len(calls), 1)

    def test_child_exit_before_project_creation_returns_the_log_tail(self) -> None:
        with _launch_server() as (root, server), _popen(creates=False, exit_code=2, output=b"Traceback: sandbox missing\n"):
            status, payload = _call(server, "POST", "/api/launch", {"mode": "free", "project": "demo", "task": "x"}, AUTH)
            self.assertEqual((status, payload["error"]), (502, "LAUNCH_EXITED"))
            self.assertIn("sandbox missing", payload["log_tail"])

    def test_workspace_must_be_the_code_root(self) -> None:
        with _launch_server() as (root, server), _popen() as calls, mock.patch.object(launcher, "CODE_ROOT", root / "elsewhere"):
            status, payload = _call(server, "POST", "/api/launch", {"mode": "free", "project": "demo", "task": "x"}, AUTH)
            self.assertEqual((status, payload["error"]), (409, "LAUNCH_WORKSPACE"))
            self.assertEqual(calls, [])


class LaunchRequestGuardTest(unittest.TestCase):
    BODY = {"mode": "free", "project": "demo", "task": "x"}

    def test_post_requires_capability_json_and_same_origin(self) -> None:
        with _launch_server() as (root, server), _popen() as calls:
            for path in ("/api/launch", "/api/launch/stop", "/api/launch/resume"):
                status, payload = _call(server, "POST", path, self.BODY, {"Content-Type": "application/json"})
                self.assertEqual((status, payload["error"]), (403, "HUMAN_DECISION_CAPABILITY_REQUIRED"))
                status, payload = _call(server, "POST", path, self.BODY, {**AUTH, "X-Human-Decision-Capability": "wrong"})
                self.assertEqual(status, 403)
            status, payload = _call(server, "POST", "/api/launch", self.BODY, {**AUTH, "Content-Type": "text/plain"})
            self.assertEqual((status, payload["error"]), (415, "LAUNCH_CONTENT_TYPE"))
            status, payload = _call(server, "POST", "/api/launch", self.BODY, {**AUTH, "Origin": "http://evil.example"})
            self.assertEqual((status, payload["error"]), (403, "LAUNCH_ORIGIN"))
            status, payload = _call(server, "POST", "/api/launch", self.BODY, {**AUTH, "Origin": "null"})
            self.assertEqual((status, payload["error"]), (403, "LAUNCH_ORIGIN"))
            self.assertEqual(calls, [])
            status, payload = _call(server, "POST", "/api/launch", self.BODY, {**AUTH, "Origin": "http://127.0.0.1:8765"})
            self.assertEqual(status, 200, payload)
            self.assertEqual(len(calls), 1)

    def test_launch_routes_are_absent_unless_enabled(self) -> None:
        with _launch_server() as (root, server), _popen() as calls:
            server.launch_enabled = False
            for path in ("/launch", "/api/launch/options", "/api/launch/prompt", "/api/launch/status?project=demo"):
                self.assertEqual(_call(server, "GET", path)[0], 404, path)
            for path in ("/api/launch", "/api/launch/stop", "/api/launch/resume"):
                self.assertEqual(_call(server, "POST", path, self.BODY, AUTH)[0], 404, path)
            status, page = _call(server, "GET", "/")
            self.assertIn('"launch":false', page["raw"])
            self.assertEqual(calls, [])

    def test_imported_snapshot_workspace_never_launches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".runtime_runs").mkdir()
            (root / ".hf-import.json").write_text('{"path_mapping": {}, "source_workspace": "/x"}', encoding="utf-8")
            server = DashboardServer(("127.0.0.1", 0), workspace_root=root, runs_root=root / ".runtime_runs",
                                     bind_and_activate=False, launch_enabled=True)
            try:
                self.assertFalse(server.launch_enabled)
            finally:
                server.server_close()

    def test_cli_requires_explicit_remote_launch_on_non_loopback_hosts(self) -> None:
        with mock.patch.object(cli, "serve") as serve, mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["serve", "--enable-launch", "--host", "0.0.0.0"])
            self.assertEqual(raised.exception.code, 2)
            serve.assert_not_called()
            cli.main(["serve", "--host", "0.0.0.0"])
            self.assertFalse(serve.call_args.kwargs["launch_enabled"])
            cli.main(["serve", "--enable-launch", "--host", "0.0.0.0", "--allow-remote-launch"])
            self.assertTrue(serve.call_args.kwargs["launch_enabled"])
            for host in ("127.0.0.1", "localhost", "::1"):
                cli.main(["serve", "--enable-launch", "--host", host])
                self.assertTrue(serve.call_args.kwargs["launch_enabled"])
                self.assertEqual(serve.call_args.kwargs["launch_limit"], 4)


class LaunchControlTest(unittest.TestCase):
    def _record(self, root: Path, project: str, pid: int, create_time: float) -> Path:
        directory = root / ".runtime_runs/dashboard_launches" / f"{project}-20260101T000000000000Z"
        directory.mkdir(parents=True)
        (directory / "launch.json").write_text(json.dumps({
            "schema_version": "dashboard_launch_v1", "kind": "create", "project": project, "argv": [],
            "pid": pid, "create_time": create_time, "started_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
        return directory / "launch.json"

    def test_stop_signals_only_the_recorded_process_identity(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            created = psutil.Process(child.pid).create_time()
            with _launch_server() as (root, server):
                record = self._record(root, "demo", child.pid, created - 50)
                status, payload = _call(server, "POST", "/api/launch/stop", {"project": "demo"}, AUTH)
                self.assertEqual((status, payload["error"]), (409, "LAUNCH_NOT_RUNNING"))
                self.assertIsNone(child.poll())
                record.write_text(record.read_text(encoding="utf-8").replace(str(created - 50), str(created)), encoding="utf-8")
                status, payload = _call(server, "GET", "/api/launch/status?project=demo")
                self.assertTrue(payload["can_stop"])
                status, payload = _call(server, "POST", "/api/launch/stop", {"project": "demo", "extra": 1}, AUTH)
                self.assertEqual((status, payload["error"]), (400, "LAUNCH_FIELD_UNKNOWN"))
                status, payload = _call(server, "POST", "/api/launch/stop", {"project": "demo"}, AUTH)
                self.assertEqual(status, 200, payload)
                self.assertTrue(payload["stopping"])
                self.assertIsNotNone(child.wait(timeout=10))
                self.assertTrue(json.loads(record.read_text(encoding="utf-8"))["stop_requested_at"])
                status, payload = _call(server, "GET", "/api/launch/status?project=demo")
                self.assertEqual((payload["running"], payload["can_stop"]), (False, False))
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_status_and_resume_of_a_stopped_slim_project(self) -> None:
        with _launch_server() as (root, server), _popen(heartbeat=True) as calls:
            project = root / ".runtime_runs/projects/demo"
            (project / "runs/r01").mkdir(parents=True)
            state = {"schema_version": "slim_harness_workflow_v1", "phase": "stopped", "stop_reason": "operator_stop"}
            (project / "workflow.json").write_text(json.dumps(state), encoding="utf-8")
            status, payload = _call(server, "GET", "/api/launch/status?project=demo")
            self.assertEqual(status, 200)
            self.assertEqual((payload["can_resume"], payload["can_stop"], payload["launched"]), (True, False, False))
            status, payload = _call(server, "POST", "/api/launch/resume", {"project": "demo"}, AUTH)
            self.assertEqual(status, 200, payload)
            self.assertEqual(calls[0].argv, [CAD_PYTHON, "-X", "utf8", "-m",
                "simulation_ready_cad_workflow.validation_pipeline.slim_harness", "--workspace", str(root), "resume", "demo"])
            self.assertEqual(payload["kind"], "resume")
            (project / "workflow.json").write_text(json.dumps({**state, "complete_instance_pass": True}), encoding="utf-8")
            status, payload = _call(server, "POST", "/api/launch/resume", {"project": "demo"}, AUTH)
            self.assertEqual((status, payload["error"]), (409, "LAUNCH_COMPLETE"))
            status, payload = _call(server, "POST", "/api/launch/resume", {"project": "missing"}, AUTH)
            self.assertEqual((status, payload["error"]), (404, "LAUNCH_NOT_SLIM"))


class LaunchOptionsTest(unittest.TestCase):
    def test_options_and_examples(self) -> None:
        with _launch_server() as (root, server), mock.patch.object(launcher, "_checks", return_value={"ready": False, "items": []}):
            (root / "examples/tasks").mkdir(parents=True)
            (root / "examples/tasks/gripper.md").write_text("Design a gripper.", encoding="utf-8")
            status, options = _call(server, "GET", "/api/launch/options")
            self.assertEqual(status, 200)
            self.assertIsNone(options["unavailable"])
            self.assertEqual(options["modes"], ["free"])
            self.assertNotIn("families", options)
            self.assertEqual(options["examples"], [{"name": "gripper", "title": "Gripper", "text": "Design a gripper."}])
            providers = {item["id"]: item for item in options["providers"]}
            self.assertEqual(providers["codex"]["default_reasoning"], "high")
            self.assertNotIn("cursor_cli", providers)
            self.assertTrue(providers["cursor"]["experimental"])
            self.assertEqual(options["budget_hours"], [1, 2, 3, 5, 8, 12])
            self.assertEqual([tier["id"] for tier in options["tiers"]], ["light", "medium", "high"])
            self.assertIn(options["execution"]["mode"], {"formal", "local", "container", None})
            self.assertEqual(options["checks"], {"ready": False, "items": []})
            self.assertEqual(_call(server, "GET", "/api/launch/prompt")[0], 404)

    def test_environment_checks_report_cli_and_login_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary)
            codex = bin_dir / "codex"
            codex.write_text("#!/bin/sh\n[ \"$1\" = --version ] && echo 'codex-cli 9.9' && exit 0\nexit 1\n", encoding="utf-8")
            codex.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": str(bin_dir)}):
                os.environ.pop("OPENAI_API_KEY", None)
                item = launcher._agent_check("codex")
                self.assertEqual((item["cli_found"], item["version"], item["auth"], item["ok"]), (True, "codex-cli 9.9", "missing", False))
                with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "set"}):
                    item = launcher._agent_check("codex")
                self.assertEqual((item["auth"], item["auth_detail"], item["ok"]), ("env", "OPENAI_API_KEY", True))
                item = launcher._agent_check("kimi")
                self.assertEqual((item["cli_found"], item["ok"]), (False, False))
            claude = bin_dir / "claude"
            claude.write_text("#!/bin/sh\necho '2.1.0 (Claude Code)'\n", encoding="utf-8")
            claude.chmod(0o755)
            with mock.patch.dict(os.environ, {"PATH": str(bin_dir), "ANTHROPIC_API_KEY": "set"}), \
                    mock.patch.object(launcher.sys, "platform", "linux"):
                os.environ.pop("SIMREADY_CONTAINER_ISOLATION", None)
                item = launcher._agent_check("claude_code")
                self.assertEqual((item["ok"], item["sandbox_missing"]), (False, ["bwrap", "socat"]))
                with mock.patch.dict(os.environ, {"SIMREADY_CONTAINER_ISOLATION": "1"}):
                    item = launcher._agent_check("claude_code")
                self.assertEqual((item["ok"], "sandbox_missing" in item), (True, False))
            check = launcher._python_check(sys.executable, ("json",))
            self.assertTrue(check["ok"])
            check = launcher._python_check(sys.executable, ("module_that_does_not_exist",))
            self.assertFalse(check["ok"])
            self.assertIn("module_that_does_not_exist", check["detail"])


class OptionalSurfacesTest(unittest.TestCase):
    """The open-source export omits the ReAct maintenance and team archive packages."""

    @contextmanager
    def _server(self, **kwargs):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".runtime_runs").mkdir()
            server = DashboardServer(("127.0.0.1", 0), workspace_root=root, runs_root=root / ".runtime_runs", **kwargs)
            server.runs = lambda: []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(*server.server_address, timeout=5)
            try:
                yield server, connection
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join()

    @staticmethod
    def _fetch(connection, method, path, body=None, headers=None):
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read().decode("utf-8", "replace")

    def test_missing_maintenance_and_team_archive_answer_404_and_hide_ui(self) -> None:
        with mock.patch.object(server_module, "maintenance_routes", None), mock.patch.object(server_module, "TEAM_LIBRARY", False):
            with self._server() as (server, connection):
                for path in ("/maintenance", "/maintenance/proposals", "/api/maintenance?run=x", "/api/role-sessions?run=x&surface=loop",
                             "/api/resource-approval?run=x", "/api/proposals?run=x", "/api/team-library", "/imports/abc/"):
                    self.assertEqual(self._fetch(connection, "GET", path)[0], 404, path)
                for path in ("/api/worker-resume", "/api/resource-approval/decide", "/api/proposals/submit"):
                    status, _body = self._fetch(connection, "POST", path, b"{}", {
                        "Content-Type": "application/json", "X-Human-Decision-Capability": server.human_decision_capability})
                    self.assertEqual(status, 404, path)
                status, runs = self._fetch(connection, "GET", "/api/runs")
                self.assertEqual(json.loads(runs)["team_library"]["items"], [])
                status, html = self._fetch(connection, "GET", "/")
                self.assertIn('"features":{"maintenance":false,"team_library":false,"launch":false}', html)

    def test_launch_page_is_served_below_the_base_path(self) -> None:
        with self._server(base_path="/harness/dashboard", launch_enabled=True) as (server, connection):
            status, html = self._fetch(connection, "GET", "/harness/dashboard/launch")
            self.assertEqual(status, 200)
            self.assertIn('src="/harness/dashboard/static/launch.js"', html)
            self.assertIn('"apiBase":"/harness/dashboard/api"', html)
            self.assertIn('"launch":true', html)
            status, script = self._fetch(connection, "GET", "/harness/dashboard/static/launch.js")
            self.assertEqual(status, 200)
            self.assertIn("launchRedirectUrl", script)
            status, body = self._fetch(connection, "POST", "/harness/dashboard/api/launch", b"{}", {
                "Content-Type": "application/json", "X-Human-Decision-Capability": server.human_decision_capability,
                "Origin": "http://evil.example"})
            self.assertEqual((status, json.loads(body)["error"]), (403, "LAUNCH_ORIGIN"))
        with self._server() as (server, connection):
            self.assertEqual(self._fetch(connection, "GET", "/launch")[0], 404)


if __name__ == "__main__":
    unittest.main()
