from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from ..cold_start import provider_transport as transport
from ..cold_start.provider_workers import CodexAppWorker, KimiCliWorker
from ..cold_start.worker import RESPONSE_SCHEMA
from ..cold_start.worker_factory import create_worker, default_worker_executable


ANSWER = {"status": "ready", "summary": "fixture", "evidence_paths": [], "issues": []}


class ProviderTransportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "prompt.txt").write_text("fixture", encoding="utf-8")
        (self.root / "schema.json").write_text(json.dumps(RESPONSE_SCHEMA), encoding="utf-8")
        self.request = {"provider": "codex_app", "executable": "codex", "model": "fixture-model",
                        "reasoning_effort": "high", "cwd": str(self.root),
                        "prompt_path": str(self.root / "prompt.txt"), "schema_path": str(self.root / "schema.json"),
                        "readonly": False, "writable_roots": [str(self.root)], "session_id": None,
                        "images": [str(self.root / "view.png")]}

    def child(self, events, code=0):
        return Mock(stdin=io.StringIO(), stdout=io.StringIO("\n".join(json.dumps(e) for e in events) + "\n"),
                    poll=Mock(return_value=code), wait=Mock(return_value=code))

    def app_events(self):
        return [
            {"id": 1, "result": {}},
            {"id": 2, "result": {"thread": {"id": "owned-thread"}}},
            # Notifications may arrive before the request acknowledgement.
            {"method": "item/completed", "params": {"threadId": "owned-thread", "turnId": "turn-1",
             "item": {"type": "agentMessage", "id": "answer", "text": json.dumps(ANSWER)}}},
            {"id": 3, "result": {"turn": {"id": "turn-1"}}},
            {"method": "thread/tokenUsage/updated", "params": {"threadId": "owned-thread", "tokenUsage":
             {"total": {"inputTokens": 100000, "outputTokens": 9000}, "last": {"inputTokens": 12, "outputTokens": 3}}}},
            {"method": "turn/completed", "params": {"threadId": "owned-thread", "turn": {"id": "turn-1", "status": "completed"}}},
        ]

    def test_factory_preserves_routes_and_effective_kimi_effort(self):
        self.assertIsInstance(create_worker("codex_app"), CodexAppWorker)
        kimi = create_worker("kimi_cli", reasoning_effort="max")
        self.assertIsInstance(kimi, KimiCliWorker)
        self.assertIsNone(kimi.reasoning_effort)
        self.assertEqual(kimi.requested_reasoning_effort, "max")
        self.assertEqual(default_worker_executable("kimi_cli", "codex"), "kimi")

    def test_app_uses_json_rpc_schema_images_and_actual_completion(self):
        child = self.child(self.app_events())
        with patch.object(transport, "_launch", return_value=child) as launch, patch.object(transport, "emit") as emit:
            transport.app_turn(self.request)
        self.assertEqual(launch.call_args.args[0], ["codex", "app-server", "--stdio"])
        sent = [json.loads(line) for line in child.stdin.getvalue().splitlines()]
        self.assertEqual([e["method"] for e in sent], ["initialize", "initialized", "thread/start", "turn/start"])
        params = sent[-1]["params"]
        self.assertEqual(params["outputSchema"], RESPONSE_SCHEMA)
        self.assertEqual(params["effort"], "high")
        self.assertEqual(params["input"][1]["type"], "localImage")
        result = emit.call_args.args[0]
        self.assertEqual(json.loads(result["result"]), ANSWER)
        self.assertEqual(result["usage"], {})  # Session total/last-response are not whole-turn usage.

    def test_app_exact_resume_and_readonly_policy(self):
        self.request.update(session_id="owned-thread", readonly=True)
        child = self.child(self.app_events())
        with patch.object(transport, "_launch", return_value=child), patch.object(transport, "emit"):
            transport.app_turn(self.request)
        sent = [json.loads(line) for line in child.stdin.getvalue().splitlines()]
        self.assertEqual(sent[2]["method"], "thread/resume")
        self.assertEqual(sent[2]["params"]["threadId"], "owned-thread")
        self.assertEqual(sent[-1]["params"]["sandboxPolicy"], {"type": "readOnly"})

    def test_app_new_thread_uses_latest_exact_turn_total_including_all_responses(self):
        self.request["ephemeral"] = True
        events = self.app_events()
        events[-2]["params"]["turnId"] = "turn-1"
        events[-2]["params"]["tokenUsage"]["total"] = {"inputTokens": 18000, "outputTokens": 20}
        final_usage = {"inputTokens": 20662, "outputTokens": 29, "cacheWriteInputTokens": 7}
        events.insert(-1, {"method": "thread/tokenUsage/updated", "params": {
            "threadId": "owned-thread", "turnId": "turn-1",
            "tokenUsage": {"total": final_usage, "last": {"inputTokens": 8, "outputTokens": 2}}}})
        with patch.object(transport, "_launch", return_value=self.child(events)), patch.object(transport, "emit") as emit:
            transport.app_turn(self.request)
        result = emit.call_args.args[0]
        self.assertEqual(result["usage"], {"input_tokens": 20662, "output_tokens": 29, "cache_write_input_tokens": 7})
        self.assertEqual(result["usage_source"], "codex_app_new_thread_total")
        self.assertEqual(result["usage_scope"], "turn")
        self.assertEqual(result["turn_id"], "turn-1")
        self.assertIn(events[-2], [call.args[0] for call in emit.call_args_list])

    def test_app_resumed_total_and_wrong_turn_are_never_used(self):
        for session, thread, turn in (("owned-thread", "owned-thread", "turn-1"),
                                      (None, "owned-thread", "previous-turn"),
                                      (None, "another-thread", "turn-1")):
            with self.subTest(session=session, thread=thread, turn=turn):
                self.request["session_id"] = session
                events = self.app_events()
                events[-2]["params"].update(threadId=thread, turnId=turn)
                with patch.object(transport, "_launch", return_value=self.child(events)), patch.object(transport, "emit") as emit:
                    transport.app_turn(self.request)
                self.assertEqual(emit.call_args.args[0]["usage"], {})

    def test_app_explicit_completion_usage_takes_precedence(self):
        events = self.app_events()
        events[-2]["params"]["turnId"] = "turn-1"
        events[-1]["params"]["turn"]["usage"] = {"inputTokens": 40, "outputTokens": 6}
        with patch.object(transport, "_launch", return_value=self.child(events)), patch.object(transport, "emit") as emit:
            transport.app_turn(self.request)
        self.assertEqual(emit.call_args.args[0]["usage"], {"input_tokens": 40, "output_tokens": 6})
        self.assertEqual(emit.call_args.args[0]["usage_source"], "codex_app_turn_completed")

    def test_app_worker_preserves_usage_source_and_resumed_watch_fallback(self):
        def success(command, **kwargs):
            kwargs["stdout"].write(json.dumps({"type": "result", "is_error": False, "session_id": "new",
                "result": json.dumps(ANSWER), "usage": {"input_tokens": 20662, "output_tokens": 29},
                "usage_source": "codex_app_new_thread_total"}) + "\n")
            return Mock(returncode=0, stdin=None, poll=Mock(return_value=0))
        worker = CodexAppWorker()
        with patch("subprocess.Popen", side_effect=success):
            result = worker.run_turn(self.root, "fresh", self.root / "fresh", 2, fresh=True)
        self.assertEqual(result["usage"], {"input_tokens": 20662, "output_tokens": 29})
        self.assertEqual(result["usage_source"], "codex_app_new_thread_total")
        self.assertEqual(result["usage_scope"], "turn")
        live = json.loads((self.root / "fresh" / "usage.live.json").read_text(encoding="utf-8"))
        self.assertEqual(live["source"], result["usage_source"])
        watch = Mock(value={"usage": {"input_tokens": 7, "output_tokens": 4}, "source": "codex_raw_turn_token_usage"})
        self.assertEqual(worker._final_usage({}, watch), watch.value["usage"])
        self.assertEqual(worker._provider_metadata()["usage_source"], "codex_raw_turn_token_usage")

    def test_app_failed_turn_and_wrong_thread_fail(self):
        events = self.app_events()
        events[-1]["params"]["turn"].update(status="failed", error={"message": "quota limit"})
        with patch.object(transport, "_launch", return_value=self.child(events)), patch.object(transport, "emit"):
            with self.assertRaisesRegex(RuntimeError, "quota"):
                transport.app_turn(self.request)
        self.request["session_id"] = "another-thread"
        with patch.object(transport, "_launch", return_value=self.child(self.app_events())), patch.object(transport, "emit"):
            with self.assertRaisesRegex(RuntimeError, "matching thread"):
                transport.app_turn(self.request)

    def test_kimi_commands_readonly_profile_and_exact_resume(self):
        self.request.update(provider="kimi", executable="kimi", readonly=True)
        command = transport.kimi_command(self.request)
        self.assertNotIn("-S", command)
        self.assertNotIn("--auto", command)
        self.assertNotIn("--plan", command)
        profile = Path(command[command.index("--agent-file") + 1]).read_text(encoding="utf-8")
        self.assertIn("  - Read", profile)
        self.assertNotIn("  - Bash", profile)
        self.request["session_id"] = "review-session"
        command = transport.kimi_command(self.request)
        self.assertEqual(command[command.index("-S") + 1], "review-session")
        self.assertNotIn("--agent-file", command)

    def test_kimi_stream_preserves_session_and_unknown_usage(self):
        events = [{"role": "meta", "type": "session.resume_hint", "session_id": "kimi-thread"},
                  {"role": "assistant", "content": [{"type": "text", "text": json.dumps(ANSWER)}]}]
        with patch.object(transport, "_launch", return_value=self.child(events)), patch.object(transport, "_kimi_wire_path", return_value=None), patch.object(transport, "emit") as emit:
            transport.kimi_turn(self.request)
        result = emit.call_args.args[0]
        self.assertEqual(result["session_id"], "kimi-thread")
        self.assertEqual(result["usage"], {})
        self.assertEqual(json.loads(result["result"]), ANSWER)

    def test_kimi_wire_usage_ignores_earlier_turn_and_session_snapshots(self):
        wire = self.root / "wire.jsonl"
        old = json.dumps({"type": "usage.record", "usageScope": "turn", "usage": {"inputOther": 900, "output": 90}}) + "\n"
        fresh = [{"type": "turn.prompt", "input": [{"type": "text", "text": "fixture"}]},
                 {"type": "usage.record", "usageScope": "session", "usage": {"inputOther": 9999, "output": 9999}},
                 {"type": "usage.record", "usageScope": "turn", "usage": {"inputOther": 5, "inputCacheRead": 2, "output": 3}}]
        wire.write_text(old + "\n".join(json.dumps(e) for e in fresh), encoding="utf-8")
        usage = transport._kimi_wire_usage(wire, len(old.encode()), "fixture")
        self.assertEqual(usage, {"input_tokens": 7, "output_tokens": 3, "cached_input_tokens": 2})
        self.assertEqual(transport._kimi_wire_usage(wire, len(old.encode()), "another"), {})

    def test_kimi_cannot_accept_a_tool_call_as_final_answer(self):
        events = [{"role": "meta", "type": "session.resume_hint", "session_id": "kimi-thread"},
                  {"role": "assistant", "content": json.dumps(ANSWER),
                   "tool_calls": [{"id": "tool-1", "function": {"name": "Read", "arguments": "{}"}}]}]
        with patch.object(transport, "_launch", return_value=self.child(events)), patch.object(transport, "_kimi_wire_path", return_value=None), patch.object(transport, "emit"):
            with self.assertRaisesRegex(RuntimeError, "no final assistant"):
                transport.kimi_turn(self.request)

    def test_worker_review_is_fresh_and_preserves_author(self):
        worker = CodexAppWorker(session_id="author")
        def success(command, **kwargs):
            kwargs["stdout"].write(json.dumps({"type": "result", "is_error": False, "session_id": "review",
                                               "result": json.dumps(ANSWER)}) + "\n")
            return Mock(returncode=0, stdin=None, poll=Mock(return_value=0))
        with patch("subprocess.Popen", side_effect=success):
            result = worker.run_turn(self.root, "review", self.root / "review", 2, readonly=True)
        request = json.loads((self.root / "review" / "provider-request.json").read_text(encoding="utf-8"))
        self.assertIsNone(request["session_id"])
        self.assertTrue(request["ephemeral"])
        self.assertEqual(worker.session_id, "author")
        self.assertEqual(result["status"], "ready")

    def test_secret_redaction(self):
        self.assertEqual(transport.redact({"access_token": "secret", "text": "Bearer abcdefghijklmnop"}),
                         {"access_token": "[REDACTED]", "text": "[REDACTED]"})


if __name__ == "__main__":
    unittest.main()
