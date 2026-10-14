from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ..cold_start.cursor_worker import CursorCliWorker, DEFAULT_CURSOR_MODEL
from ..cold_start import cli_invocation
from ..cold_start.worker_factory import (
    create_worker,
    default_worker_executable,
    default_worker_model,
    normalize_worker_provider,
)
from ..slim_harness.cli import create_project, start
from ..slim_harness.construction import recover_author_session, worker_for
from ..slim_harness.construction_budget import BudgetConfig, initialize_budget


class CursorWorkerFactoryTests(unittest.TestCase):
    def test_aliases_and_defaults_select_cursor_cli(self) -> None:
        self.assertEqual("cursor", normalize_worker_provider("cursor_agent"))
        self.assertEqual("cursor", normalize_worker_provider("cursor_cli"))
        self.assertEqual(DEFAULT_CURSOR_MODEL, default_worker_model("cursor"))
        self.assertEqual("cursor-agent", default_worker_executable("cursor", "codex"))
        self.assertEqual("/fake/agent", default_worker_executable("cursor", "/fake/agent"))
        worker = create_worker(provider="cursor_agent", reasoning_effort="max")
        self.assertIsInstance(worker, CursorCliWorker)
        self.assertEqual(DEFAULT_CURSOR_MODEL, worker.model)
        self.assertEqual("cursor-agent", worker.executable)
        self.assertEqual("max", worker.reasoning_effort)

    def test_unknown_provider_names_cursor(self) -> None:
        with self.assertRaisesRegex(ValueError, "cursor"):
            create_worker(provider="unknown")


class CursorCliWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        invocation = patch("simulation_ready_cad_workflow.validation_pipeline.cold_start.cursor_worker.cli_command",
                           side_effect=lambda command: command)
        invocation.start()
        self.addCleanup(invocation.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def successful_run(self, command, **kwargs):
        result = {
            "status": "ready",
            "summary": "Authored",
            "evidence_paths": ["design/scene.json"],
            "issues": [],
        }
        kwargs["stdout"].write(json.dumps({
            "type": "system",
            "subtype": "init",
            "session_id": "exact-session",
        }) + "\n")
        kwargs["stdout"].write(json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(result),
            "session_id": "exact-session",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }) + "\n")
        return Mock(returncode=0, stdin=None, poll=Mock(return_value=0))

    def test_author_resumes_exact_session_and_parses_usage(self) -> None:
        worker = CursorCliWorker()
        with patch("subprocess.Popen", side_effect=self.successful_run) as run:
            first = worker.run_turn(self.root, "create", self.root / "one", 10)
            second = worker.run_turn(self.root, "repair", self.root / "two", 10)
        self.assertEqual(first["status"], "ready")
        self.assertEqual(first["session_id"], "exact-session")
        self.assertEqual(second["usage"], {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20})
        self.assertNotIn("--resume", run.call_args_list[0].args[0])
        command = run.call_args.args[0]
        self.assertEqual(command[0], "cursor-agent")
        self.assertIn("-p", command)
        self.assertEqual(command[command.index("--output-format") + 1], "stream-json")
        self.assertIn("--trust", command)
        self.assertIn("--force", command)
        self.assertEqual(command[command.index("--resume") + 1], "exact-session")
        self.assertEqual(command[command.index("--model") + 1], DEFAULT_CURSOR_MODEL)
        self.assertEqual(command[command.index("--workspace") + 1], str(self.root.resolve()))
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertNotIn("--ephemeral", command)
        self.assertNotIn("exec", command)

    def cursor_print_usage_run(self, command, **kwargs):
        result = {
            "status": "ready",
            "summary": "Authored",
            "evidence_paths": ["design/scene.json"],
            "issues": [],
        }
        kwargs["stdout"].write(json.dumps({
            "type": "system",
            "subtype": "init",
            "session_id": "exact-session",
        }) + "\n")
        kwargs["stdout"].write(json.dumps({
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(result),
            "session_id": "exact-session",
            "usage": {
                "inputTokens": 120,
                "outputTokens": 71244,
                "cacheReadTokens": 8094641,
                "cacheWriteTokens": 182253,
            },
        }) + "\n")
        return Mock(returncode=0, stdin=None, poll=Mock(return_value=0))

    def test_cursor_result_usage_is_normalized_into_live_file(self) -> None:
        output = self.root / "author"
        with patch("subprocess.Popen", side_effect=self.cursor_print_usage_run):
            result = CursorCliWorker().run_turn(self.root, "create", output, 10)
        usage = {
            "input_tokens": 120 + 8094641 + 182253,
            "cached_input_tokens": 8094641,
            "cache_write_input_tokens": 182253,
            "output_tokens": 71244,
            "total_tokens": 120 + 8094641 + 182253 + 71244,
        }
        self.assertEqual(result["usage"], usage)
        live = json.loads((output / "usage.live.json").read_text(encoding="utf-8"))
        self.assertEqual(live["usage"], usage)
        self.assertEqual(live["status"], "completed")
        self.assertFalse(live["pending"])
        self.assertEqual(live["source"], "cursor_agent_result_usage")

    def test_review_is_ask_mode_and_preserves_author_session(self) -> None:
        worker = CursorCliWorker(model="explicit-model", session_id="author-session")
        image = self.root / "view.png"
        image.write_bytes(b"png")
        with patch("subprocess.Popen", side_effect=self.successful_run) as run:
            worker.run_turn(
                self.root, "review", self.root / "review", 10,
                images=[image], readonly=True, fresh=True,
            )
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--mode") + 1], "ask")
        self.assertNotIn("--force", command)
        self.assertNotIn("--resume", command)
        self.assertEqual(command[command.index("--model") + 1], "explicit-model")
        self.assertIn(str(image.resolve()), (self.root / "review" / "prompt.txt").read_text(encoding="utf-8"))
        self.assertEqual(worker.session_id, "author-session")

    def test_construction_adds_project_and_author_tool_roots(self) -> None:
        workspace = self.root
        project = create_project(workspace, "machine", "A slider must move.")
        design = project / "runs" / "r01" / "design"
        state = json.loads((project / "workflow.json").read_text(encoding="utf-8"))
        state["request"]["construction"] = True
        (project / "workflow.json").write_text(json.dumps(state), encoding="utf-8")
        initialize_budget(project, BudgetConfig(hours=1))
        worker = worker_for(project, CursorCliWorker())
        with patch("subprocess.Popen", side_effect=OSError("launch sentinel")) as run:
            result = worker.run_turn(design, "fixture", project / "worker_turns" / "t001_author", 1)
        command = run.call_args.args[0]
        roots = [command[index + 1] for index, argument in enumerate(command[:-1]) if argument == "--add-dir"]
        self.assertEqual(roots, [str(project.resolve()), str(workspace / ".runtime_runs" / "author_tools")])
        self.assertEqual(result["provider_error"]["kind"], "launch")

    def test_long_prompt_uses_file_pointer_instead_of_argv(self) -> None:
        prompt = "build " + ("x" * 7000)
        with patch("subprocess.Popen", side_effect=self.successful_run) as run:
            CursorCliWorker().run_turn(self.root, prompt, self.root / "long", 10)
        command = run.call_args.args[0]
        self.assertNotIn(prompt, command)
        self.assertIn("prompt.txt", command[-1])

    def test_nonzero_exit_and_provider_events_are_not_success(self) -> None:
        for message, kind in (("You've hit your usage limit", "quota"), ("401 token_revoked", "auth")):
            with self.subTest(kind=kind):
                def fail(command, **kwargs):
                    kwargs["stdout"].write(json.dumps({
                        "type": "result",
                        "subtype": "error",
                        "is_error": True,
                        "result": message,
                        "session_id": "failed-session",
                    }) + "\n")
                    return Mock(returncode=1, stdin=None, poll=Mock(return_value=1))
                with patch("subprocess.Popen", side_effect=fail):
                    result = CursorCliWorker().run_turn(self.root, "work", self.root / kind, 10)
                self.assertEqual(result["status"], "provider_error")
                self.assertEqual(result["provider_error"]["kind"], kind)

    def test_missing_response_cannot_reuse_stale_success(self) -> None:
        output = self.root / "missing"
        output.mkdir()
        (output / "response.json").write_text('{"status":"ready","summary":"stale","evidence_paths":[],"issues":[]}', encoding="utf-8")
        process = Mock(returncode=0, stdin=None, poll=Mock(return_value=0))
        with patch("subprocess.Popen", return_value=process):
            result = CursorCliWorker().run_turn(self.root, "work", output, 10)
        self.assertEqual(result["status"], "provider_error")

    def test_timeout_retains_observed_author_session_for_the_next_call(self) -> None:
        process = Mock(pid=12345, returncode=-15, stdin=None)
        process.communicate.side_effect = subprocess.TimeoutExpired("cursor-agent", 2)
        process.poll.side_effect = [None, -15]

        def timeout(command, **kwargs):
            kwargs["stdout"].write(json.dumps({
                "type": "system", "subtype": "init", "session_id": "exact-session",
            }) + "\n")
            return process

        worker = CursorCliWorker()
        with patch("subprocess.Popen", side_effect=timeout), patch(
            "simulation_ready_cad_workflow.validation_pipeline.cold_start.cursor_worker.terminate_process_tree",
            return_value=(),
        ):
            first = worker.run_turn(self.root, "work", self.root / "timed-out", 2)
        self.assertEqual("provider_error", first["status"])
        self.assertEqual("exact-session", worker.session_id)
        with patch("subprocess.Popen", side_effect=self.successful_run) as continuation:
            second = worker.run_turn(self.root, "continue saved work", self.root / "continued", 10)
        self.assertEqual("ready", second["status"])
        command = continuation.call_args.args[0]
        self.assertEqual("exact-session", command[command.index("--resume") + 1])

    def test_written_response_json_is_accepted(self) -> None:
        def write_file(command, **kwargs):
            turn = self.root / "file"
            (turn / "response.json").write_text(json.dumps({
                "status": "ready", "summary": "From file", "evidence_paths": [], "issues": [],
            }), encoding="utf-8")
            kwargs["stdout"].write(json.dumps({
                "type": "system", "subtype": "init", "session_id": "file-session",
            }) + "\n")
            kwargs["stdout"].write(json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "result": "wrote the file", "session_id": "file-session",
            }) + "\n")
            return Mock(returncode=0, stdin=None, poll=Mock(return_value=0))

        with patch("subprocess.Popen", side_effect=write_file):
            result = CursorCliWorker().run_turn(self.root, "create", self.root / "file", 10)
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["summary"], "From file")


class WindowsCliInvocationTests(unittest.TestCase):
    def test_windows_batch_and_powershell_shims_use_the_powershell_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "cursor-agent.ps1"
            script.write_text("param()", encoding="utf-8")
            def lookup(name):
                return str(script.with_suffix(".CMD")) if name == "cursor-agent" else "installed-shell"
            with patch.object(cli_invocation.sys, "platform", "win32"), patch.object(cli_invocation.shutil, "which", side_effect=lookup):
                self.assertEqual(cli_invocation.cli_command(["cursor-agent", "--version"]),
                                 ["installed-shell", "-NoProfile", "-File", str(script), "--version"])

    def test_windows_path_script_preserves_arguments_and_uses_installed_shell(self):
        with tempfile.TemporaryDirectory(prefix="cli shim ") as directory:
            script = Path(directory) / "cursor-agent.ps1"
            script.write_text("param()", encoding="utf-8")
            command = ["cursor-agent", "-p", 'spaces "quotes" $value; literal']
            for shell in ("pwsh", "powershell"):
                with self.subTest(shell=shell), patch.object(cli_invocation.sys, "platform", "win32"), patch.object(
                    cli_invocation.os, "get_exec_path", return_value=[directory]
                ), patch.object(cli_invocation.shutil, "which", side_effect=lambda name: "installed-shell" if name == shell else None):
                    self.assertEqual(cli_invocation.cli_command(command),
                                     ["installed-shell", "-NoProfile", "-File", str(script.resolve()), *command[1:]])

    def test_explicit_script_requires_shell_and_does_not_bypass_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "custom.ps1"
            script.write_text("param()", encoding="utf-8")
            with patch.object(cli_invocation.sys, "platform", "win32"), patch.object(cli_invocation.shutil, "which", return_value=None):
                with self.assertRaisesRegex(FileNotFoundError, "PowerShell is required"):
                    cli_invocation.cli_command([str(script), "argument"])

    def test_native_and_linux_commands_are_unchanged(self):
        command = ["cursor-agent", "argument"]
        with patch.object(cli_invocation.sys, "platform", "linux"), patch.object(cli_invocation.shutil, "which") as lookup:
            self.assertEqual(cli_invocation.cli_command(command), command)
            lookup.assert_not_called()
        with patch.object(cli_invocation.sys, "platform", "win32"), patch.object(cli_invocation.shutil, "which", return_value="cursor-agent.exe"):
            self.assertEqual(cli_invocation.cli_command(command), command)


class CursorAuthorSessionRecoveryTests(unittest.TestCase):
    def test_recover_reads_cursor_session_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            project = create_project(workspace, "recover", "A slider must move.")
            initialize_budget(project, BudgetConfig(hours=1))
            turn = project / "worker_turns" / "t001_author"
            turn.mkdir(parents=True)
            (turn / "events.jsonl").write_text(
                json.dumps({"type": "system", "subtype": "init", "session_id": "cursor-session"}) + "\n",
                encoding="utf-8",
            )
            state = json.loads((project / "workflow.json").read_text(encoding="utf-8"))
            recovered = recover_author_session(project, state)
            self.assertEqual("cursor-session", recovered["author_session_id"])
            self.assertEqual(str(turn / "events.jsonl"), recovered["author_session_recovered_from"])


class CursorCliStartTests(unittest.TestCase):
    def test_create_persists_cursor_provider_and_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.initialize_project",
                       return_value={"status": "ready"}):
                start(workspace, "cursor-machine", "A slider must move.", worker_provider="cursor")
            request = json.loads((workspace / ".runtime_runs" / "projects" / "cursor-machine" / "workflow.json").read_text())["request"]
            self.assertEqual("cursor", request["worker_provider"])
            self.assertEqual(DEFAULT_CURSOR_MODEL, request["model"])


if __name__ == "__main__":
    unittest.main()
