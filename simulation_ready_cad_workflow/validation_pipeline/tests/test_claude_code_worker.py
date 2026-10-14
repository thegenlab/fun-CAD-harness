from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ..cold_start.claude_code_worker import ClaudeCodeWorker, DEFAULT_CLAUDE_CODE_MODEL
from ..cold_start.token_watch import normalize_token_usage
from ..cold_start.worker import RESPONSE_SCHEMA
from ..cold_start.worker_factory import (
    WORKER_PROVIDERS,
    create_worker,
    default_worker_executable,
    default_worker_model,
    normalize_worker_provider,
)
from ..slim_harness import cli
from ..slim_harness.cli import create_project, start
from ..slim_harness.construction import worker_for
from ..slim_harness.construction_budget import BudgetConfig, initialize_budget
from ..trajectory.configuration import worker_configuration
from ..trajectory.events import completed_tool_output, event_rows, merge_rows
from ..trajectory.store import HistoryStore


# Recorded from a real `claude -p` 2.1.296 read-only turn (claude-haiku-5-5, low
# effort) through ClaudeCodeWorker; session/message ids and local paths scrubbed.
FIXTURE = Path(__file__).parent / "fixtures" / "claude_code_stream.jsonl"
FIXTURE_SCHEMA = {"type": "object", "additionalProperties": False,
                  "required": ["answer"], "properties": {"answer": {"type": "string"}}}
FIXTURE_SESSION = "00000000-0000-4000-8000-000000000001"
ANSWER = {"status": "ready", "summary": "Authored", "evidence_paths": ["design/scene.json"], "issues": []}
REVIEW_TOOLS = ["--permission-mode", "dontAsk", "--tools", "Read,Grep,Glob", "--allowedTools", "Read,Grep,Glob"]


def head(schema=RESPONSE_SCHEMA, model=DEFAULT_CLAUDE_CODE_MODEL, effort="high"):
    return ["claude", "-p", "--output-format", "stream-json", "--verbose", "--json-schema", json.dumps(schema),
            "--model", model, "--effort", effort]


def sandbox(*roots):
    value = {"enabled": True, "failIfUnavailable": True, "autoAllowBashIfSandboxed": True,
             "allowUnsandboxedCommands": False}
    if sys.platform.startswith("linux"):
        value["network"] = {"allowAllUnixSockets": True}
    if roots:
        value["filesystem"] = {"allowWrite": [str(root) for root in roots]}
    return ["--permission-mode", "acceptEdits", "--settings", json.dumps({"sandbox": value})]


def emitting(*events, code=0):
    def run(command, **kwargs):
        for event in events:
            kwargs["stdout"].write(json.dumps(event) + "\n")
        return Mock(returncode=code, stdin=None, poll=Mock(return_value=code))
    return run


def success(session="exact-session", structured=ANSWER):
    return emitting(
        {"type": "system", "subtype": "init", "session_id": session, "model": "claude-opus-5-5[1m]"},
        {"type": "result", "subtype": "success", "is_error": False, "result": "Done; see structured output.",
         "structured_output": structured, "session_id": session, "num_turns": 7, "total_cost_usd": 0.42,
         "usage": {"input_tokens": 10, "cache_creation_input_tokens": 200, "cache_read_input_tokens": 3000,
                   "output_tokens": 40, "service_tier": "standard", "cache_creation": {"ephemeral_1h_input_tokens": 200}}},
    )


class ClaudeCodeFactoryTests(unittest.TestCase):
    def test_aliases_defaults_and_cli_choices_select_claude_code(self) -> None:
        self.assertEqual("claude_code", normalize_worker_provider("claude"))
        self.assertIn("claude_code", WORKER_PROVIDERS)
        self.assertIn("claude", WORKER_PROVIDERS)
        self.assertEqual(DEFAULT_CLAUDE_CODE_MODEL, default_worker_model("claude"))
        # The reviewer executable falls back to "codex" when the providers differ.
        self.assertEqual("claude", default_worker_executable("claude_code", "codex"))
        self.assertEqual("/fake/claude", default_worker_executable("claude", "/fake/claude"))
        worker = create_worker(provider="claude", reasoning_effort="ultra")
        self.assertIsInstance(worker, ClaudeCodeWorker)
        self.assertEqual(("claude", DEFAULT_CLAUDE_CODE_MODEL, "ultra", "max"),
                         (worker.executable, worker.model, worker.reasoning_effort, worker.effort))
        with tempfile.TemporaryDirectory() as directory, patch.object(cli, "start", return_value={"status": "ready"}) as started, \
                redirect_stdout(io.StringIO()):
            cli.main(["--workspace", directory, "--worker-provider", "claude", "create", "--mode", "free",
                      "--project", "machine", "--brief", "A slider must move.", "--reviewer-provider", "claude_code"])
        self.assertEqual(("claude", "claude_code"), (started.call_args.kwargs["worker_provider"],
                                                     started.call_args.kwargs["reviewer_provider"]))

    def test_create_persists_claude_code_provider_and_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.initialize_project",
                       return_value={"status": "ready"}):
                start(workspace, "claude-machine", "A slider must move.", worker_provider="claude")
            request = json.loads((workspace / ".runtime_runs/projects/claude-machine/workflow.json").read_text())["request"]
        self.assertEqual("claude_code", request["worker_provider"])
        self.assertEqual(DEFAULT_CLAUDE_CODE_MODEL, request["model"])


class ClaudeCodeWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        invocation = patch("simulation_ready_cad_workflow.validation_pipeline.cold_start.cursor_worker.cli_command",
                           side_effect=lambda command: command)
        invocation.start()
        self.addCleanup(invocation.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("SIMREADY_CONTAINER_ISOLATION", None)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_author_resumes_exact_session_from_stdin_with_structured_output(self) -> None:
        worker = ClaudeCodeWorker()
        with patch("subprocess.Popen", side_effect=success()) as run:
            first = worker.run_turn(self.root, "create", self.root / "one", 10)
            second = worker.run_turn(self.root, "repair", self.root / "two", 10)
        self.assertEqual(ANSWER, {key: first[key] for key in ANSWER})
        self.assertEqual("exact-session", first["session_id"])
        self.assertEqual(run.call_args_list[0].args[0], head() + sandbox())
        self.assertEqual(run.call_args.args[0], head() + sandbox() + ["--resume", "exact-session"])
        # No positional prompt: the turn prompt goes to stdin unchanged.
        self.assertEqual("repair", (self.root / "two" / "prompt.txt").read_text(encoding="utf-8"))
        self.assertFalse(run.call_args.kwargs["shell"])
        self.assertEqual(second["usage"], {"input_tokens": 3210, "cached_input_tokens": 3000,
                                           "cache_write_input_tokens": 200, "output_tokens": 40, "total_tokens": 3250})
        self.assertEqual({key: second[key] for key in ("provider", "model", "requested_model", "reasoning_effort",
                                                       "total_cost_usd", "num_turns")},
                         {"provider": "claude_code", "model": "claude-opus-5-5[1m]", "requested_model": "claude-opus-5-5",
                          "reasoning_effort": "high", "total_cost_usd": 0.42, "num_turns": 7})
        self.assertEqual(json.loads((self.root / "two" / "response.json").read_text()), ANSWER)

    def test_construction_author_adds_project_and_tool_roots_to_dirs_and_sandbox(self) -> None:
        project = create_project(self.root, "machine", "A slider must move.")
        state = json.loads((project / "workflow.json").read_text(encoding="utf-8"))
        state["request"]["construction"] = True
        (project / "workflow.json").write_text(json.dumps(state), encoding="utf-8")
        initialize_budget(project, BudgetConfig(hours=1))
        worker = worker_for(project, ClaudeCodeWorker(reasoning_effort="ultra"))
        tools = project.resolve().parents[2] / ".runtime_runs" / "author_tools"
        with patch("subprocess.Popen", side_effect=OSError("launch sentinel")) as run:
            result = worker.run_turn(project / "runs/r01/design", "fixture", project / "worker_turns/t001_author", 1)
        self.assertEqual(run.call_args.args[0], head(effort="max") + sandbox(project.resolve(), tools)
                         + ["--add-dir", str(project.resolve()), "--add-dir", str(tools)])
        self.assertEqual("launch", result["provider_error"]["kind"])
        with patch.dict(os.environ, {"SIMREADY_CONTAINER_ISOLATION": "1"}), \
                patch("subprocess.Popen", side_effect=OSError("launch sentinel")) as run:
            worker.run_turn(project / "runs/r01/design", "fixture", project / "worker_turns/t002_author", 1)
        self.assertEqual(run.call_args.args[0], head(effort="max") + ["--permission-mode", "bypassPermissions",
                         "--add-dir", str(project.resolve()), "--add-dir", str(tools)])

    def test_only_linux_sandbox_allows_unix_sockets(self) -> None:
        # Recorded on Ubuntu 24.04: without this every sandboxed command failed with
        # "apply-seccomp: write /proc/self/setgroups (nested userns is capability-restricted ...)".
        for platform, expected in (("linux", {"allowAllUnixSockets": True}), ("darwin", None)):
            with self.subTest(platform=platform), patch.object(sys, "platform", platform), \
                    patch("subprocess.Popen", side_effect=success()) as run:
                ClaudeCodeWorker().run_turn(self.root, "work", self.root / platform, 10)
                command = run.call_args.args[0]
                settings = json.loads(command[command.index("--settings") + 1])
                self.assertEqual(expected, settings["sandbox"].get("network"))

    def test_review_is_read_only_ephemeral_and_preserves_author_session(self) -> None:
        worker = ClaudeCodeWorker(model="explicit-model", session_id="author-session", reasoning_effort="medium")
        image = self.root / "view.png"
        image.write_bytes(b"png")
        with patch("subprocess.Popen", side_effect=success("review-session")) as run:
            result = worker.run_turn(self.root, "review", self.root / "review", 10, images=[image], readonly=True, fresh=True)
        self.assertEqual(run.call_args.args[0], head(model="explicit-model", effort="medium") + REVIEW_TOOLS
                         + ["--no-session-persistence"])
        prompt = (self.root / "review" / "prompt.txt").read_text(encoding="utf-8")
        self.assertIn(str(image.resolve()), prompt)
        self.assertNotIn("response.json", prompt)
        self.assertEqual("ready", result["status"])
        self.assertEqual("author-session", worker.session_id)

    def test_fresh_writable_turn_is_sandboxed_and_ephemeral(self) -> None:
        # Template filling is an independent turn that writes its binding file.
        worker = ClaudeCodeWorker(session_id="author-session")
        with patch("subprocess.Popen", side_effect=success("fill-session")) as run:
            worker.run_turn(self.root, "fill", self.root / "fill", 10, fresh=True)
        self.assertEqual(run.call_args.args[0], head() + sandbox() + ["--no-session-persistence"])
        self.assertEqual("author-session", worker.session_id)

    def test_retained_review_persists_then_resumes_only_its_own_session(self) -> None:
        worker = ClaudeCodeWorker(session_id="author-session")
        with patch("subprocess.Popen", side_effect=success("review-session")) as run:
            first = worker.run_turn(self.root, "phase one", self.root / "p1", 10, readonly=True, fresh=True,
                                    retain_independent_session=True)
            worker.run_turn(self.root, "phase two", self.root / "p2", 10, readonly=True,
                            retain_independent_session=True, independent_session_id=first["observed_session_id"])
        self.assertEqual(run.call_args_list[0].args[0], head() + REVIEW_TOOLS)
        self.assertEqual(run.call_args.args[0], head() + REVIEW_TOOLS + ["--resume", "review-session"])
        self.assertEqual("author-session", worker.session_id)

    def test_provider_failures_are_classified_never_design_results(self) -> None:
        # Recorded: an author turn on a host without socat, where the sandbox is required.
        sandbox_missing = {"type": "result", "subtype": "error_during_execution", "duration_ms": 0, "is_error": True,
                           "num_turns": 0, "session_id": FIXTURE_SESSION, "total_cost_usd": 0,
                           "usage": {"input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
                                     "output_tokens": 0}, "permission_denials": [],
                           "errors": ["Sandbox required but unavailable: sandbox is enabled but dependencies are missing: "
                                      "socat not installed"]}
        cases = (
            (sandbox_missing, 1, "exit"),
            ({"type": "result", "subtype": "success", "is_error": True, "result": "Not logged in · Please run /login",
              "session_id": FIXTURE_SESSION}, 1, "auth"),
            ({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 429,
              "result": "You've hit your session limit · resets 3pm", "session_id": FIXTURE_SESSION}, 1, "quota"),
            ({"type": "result", "subtype": "success", "is_error": True, "api_error_status": 401,
              "result": "API Error", "session_id": FIXTURE_SESSION}, 1, "auth"),
            # Token counts and timings, even exactly 401, are not an authentication failure.
            ({"type": "result", "subtype": "error_max_turns", "is_error": False, "duration_ms": 14012,
              "duration_api_ms": 401, "usage": {"input_tokens": 40100, "output_tokens": 401},
              "modelUsage": {"claude-opus-5-5": {"outputTokens": 401}}, "session_id": FIXTURE_SESSION}, 0, "exit"),
        )
        for index, (event, code, kind) in enumerate(cases):
            with self.subTest(kind=kind, subtype=event["subtype"]):
                with patch("subprocess.Popen", side_effect=emitting(event, code=code)):
                    result = ClaudeCodeWorker().run_turn(self.root, "work", self.root / f"failure-{index}", 10)
                self.assertEqual("provider_error", result["status"])
                self.assertEqual(kind, result["provider_error"]["kind"])
                self.assertNotIn("usage", json.loads(result["provider_error"]["message"]))

    def test_missing_structured_output_cannot_reuse_stale_success(self) -> None:
        output = self.root / "missing"
        output.mkdir()
        (output / "response.json").write_text(json.dumps(ANSWER), encoding="utf-8")
        run = emitting({"type": "result", "subtype": "success", "is_error": False, "result": "no object",
                        "session_id": FIXTURE_SESSION})
        with patch("subprocess.Popen", side_effect=run):
            result = ClaudeCodeWorker().run_turn(self.root, "work", output, 10)
        self.assertEqual("provider_error", result["status"])
        self.assertEqual("response", result["provider_error"]["kind"])


class ClaudeCodeRecordedStreamTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]

    @unittest.skipUnless(os.name == "posix", "The fake claude executable is a POSIX script")
    def test_fake_executable_replays_recorded_turn_through_the_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "bin" / "claude"
            fake.parent.mkdir()
            fake.write_text(
                f"#!{sys.executable}\n"
                "import json, sys\n"
                f"open({str(root / 'argv.json')!r}, 'w').write(json.dumps(sys.argv[1:]))\n"
                f"open({str(root / 'stdin.txt')!r}, 'w').write(sys.stdin.read())\n"
                f"sys.stdout.write(open({str(FIXTURE)!r}, encoding='utf-8').read())\n",
                encoding="utf-8")
            fake.chmod(0o755)
            image = root / "view.png"
            image.write_bytes(b"png")
            worker = ClaudeCodeWorker(model="claude-haiku-5-5", reasoning_effort="low", executable=str(fake),
                                      session_id="author-session")
            result = worker.run_turn(root, "Return the single word.", root / "turn", 30, images=[image],
                                     readonly=True, fresh=True, response_schema=FIXTURE_SCHEMA)
            argv = json.loads((root / "argv.json").read_text())
            stdin = (root / "stdin.txt").read_text()
            live = json.loads((root / "turn" / "usage.live.json").read_text())
        self.assertEqual(argv, head(FIXTURE_SCHEMA, "claude-haiku-5-5", "low")[1:] + REVIEW_TOOLS
                         + ["--no-session-persistence"])
        self.assertTrue(stdin.startswith("Return the single word.\n\nRead these local evidence images"))
        self.assertIn(str(image.resolve()), stdin)
        self.assertEqual("pelican", result["answer"])
        self.assertEqual(FIXTURE_SESSION, result["session_id"])
        self.assertEqual("author-session", worker.session_id)
        usage = {"input_tokens": 4 + 6496 + 9284, "cached_input_tokens": 9284, "cache_write_input_tokens": 6496,
                 "output_tokens": 124, "total_tokens": 4 + 6496 + 9284 + 124}
        self.assertEqual(usage, result["usage"])
        self.assertEqual((usage, "completed"), (live["usage"], live["status"]))
        self.assertEqual({key: result[key] for key in ("provider", "model", "total_cost_usd", "num_turns")},
                         {"provider": "claude_code", "model": "claude-haiku-5-5", "total_cost_usd": 0.00145444,
                          "num_turns": 3})

    def test_recorded_usage_keeps_cache_tokens(self) -> None:
        result = next(event for event in self.events if event["type"] == "result")
        self.assertEqual(normalize_token_usage(result["usage"])["cached_input_tokens"], 9284)

    def test_trajectory_projects_tool_use_and_tool_result_rows(self) -> None:
        records = [(offset, row) for offset, event in enumerate(self.events) for row in event_rows(event, offset)]
        items = merge_rows(records)
        self.assertEqual([(item["kind"], item["title"], item["status"]) for item in items],
                         [("tool", "Read", "completed"), ("tool", "StructuredOutput", "completed")])
        self.assertIn("/workspace/outside/note.txt", items[0]["text"])
        self.assertEqual("1\tpelican\n2\t", items[0]["output"])
        read_result = next(event for event in self.events if event["type"] == "user")
        self.assertEqual("1\tpelican\n2\t", completed_tool_output(read_result))
        text = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Checking."}, {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}}
        failed = {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                                                          "content": [{"type": "text", "text": "denied"}]}]}}
        items = merge_rows([(0, row) for row in event_rows(text, 0)] + [(1, row) for row in event_rows(failed, 1)])
        self.assertEqual([(item["kind"], item["title"], item.get("status")) for item in items],
                         [("message", "Agent", None), ("tool", "Bash", "failed")])
        self.assertEqual("denied", items[1]["output"])

    def test_live_poll_after_the_call_keeps_its_title_and_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = create_project(Path(directory), "machine", "A slider must move.")
            log = project / "worker_turns" / "t001_author" / "events.jsonl"
            log.parent.mkdir(parents=True)
            lines = FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)
            store = HistoryStore(project)
            log.write_text("".join(lines[:2]), encoding="utf-8")
            store.index_log(log, "t001_author", drain=True)
            first = store.event_page(turn="t001_author")
            log.write_text("".join(lines), encoding="utf-8")
            store.index_log(log, "t001_author", drain=True)
            later = store.event_page(turn="t001_author", after=first["next_after"])
        self.assertEqual([("Read", "running")], [(item["title"], item["status"]) for item in first["items"]])
        result = next(item for item in later["items"] if item["event_key"] == first["items"][0]["event_key"])
        self.assertEqual(("Read", first["items"][0]["text"], "completed", "1\tpelican\n2\t"),
                         (result["title"], result["text"], result["status"], result["output"]))

    def test_trajectory_configuration_names_the_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data = worker_configuration(ClaudeCodeWorker(), None, Path(directory), True)
        self.assertEqual("claude_code", data["worker"]["provider"])


if __name__ == "__main__":
    unittest.main()
