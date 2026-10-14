"""Owned subprocess transports; no legacy role or approval lifecycle.

Run by absolute file path so the author workspace need not be the repository.
Raw provider notifications are retained; normalized events feed the existing
budget, token and trajectory observers. The parent owns timeout/tree cleanup.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from simulation_ready_cad_workflow.validation_pipeline.runtime.codex_linux_sandbox import (
    app_thread_sandbox,
    app_turn_sandbox_policy,
)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: "[REDACTED]" if re.search(r"api.?key|access.?token|refresh.?token|authorization|password", key, re.I)
                else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)Bearer\s+[^\s\"']+|\bsk-[A-Za-z0-9_-]{12,}", "[REDACTED]", value)
        for key, secret in os.environ.items():
            if len(secret) >= 12 and re.search(r"API_?KEY|TOKEN|SECRET", key, re.I):
                value = value.replace(secret, "[REDACTED]")
    return value


def emit(event: dict) -> None:
    print(json.dumps(redact(event), ensure_ascii=False), flush=True)


def _stderr(source) -> None:
    for line in source:
        sys.stderr.write(redact(line))
        sys.stderr.flush()


def _launch(command: list[str], cwd: str) -> subprocess.Popen:
    child = subprocess.Popen(command, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                             bufsize=1, shell=False,
                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    threading.Thread(target=_stderr, args=(child.stderr,), daemon=True).start()
    return child


def _stop(child) -> None:
    # Kill descendants while the parent is still discoverable. Never touch an
    # existing desktop app-server: this process only owns the child it started.
    if child.poll() is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from simulation_ready_cad_workflow.validation_pipeline.runtime.system_probe import terminate_process_tree
        survivors = terminate_process_tree(child.pid, 2.0)
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
        if survivors:
            raise RuntimeError("Provider process cleanup was not confirmed: " + str(survivors))


def _send(child, message):
    child.stdin.write(json.dumps(message) + "\n")
    child.stdin.flush()


def _tokens(value):
    if not isinstance(value, dict):
        return {}
    aliases = {"inputTokens": "input_tokens", "outputTokens": "output_tokens",
               "cacheWriteInputTokens": "cache_write_input_tokens",
               "cacheWriteTokens": "cache_write_input_tokens",
               "cachedInputTokens": "cached_input_tokens",
               "cacheReadTokens": "cached_input_tokens",
               "reasoningOutputTokens": "reasoning_output_tokens",
               "totalTokens": "total_tokens"}
    return {aliases.get(key, key): number for key, number in value.items()
            if key in {*aliases, *aliases.values()} and isinstance(number, int)
            and not isinstance(number, bool) and number >= 0}


def app_turn(request: dict) -> None:
    command = [request["executable"], "app-server", "--stdio"]
    for feature in request.get("disabled_features", []):
        command.extend(["--disable", feature])
    child = _launch(command, request["cwd"])
    session = request.get("session_id")
    new_thread = not session
    active_turn = None
    usage = {}
    new_thread_usage = {}
    messages = {}
    deltas = {}
    deferred = []

    def receive():
        line = child.stdout.readline()
        if not line:
            raise RuntimeError("Codex App closed before turn completion")
        try:
            event = json.loads(line)
        except ValueError:
            return {}
        if not isinstance(event, dict):
            return {}
        if "id" in event and "method" in event:
            _send(child, {"id": event["id"], "error": {"code": -32601,
                          "message": "Cold-start client does not provide interactive requests"}})
            raise RuntimeError("Codex App requires user action: " + str(event["method"]))
        return event

    def rpc(identifier, method, params):
        _send(child, {"id": identifier, "method": method, "params": params})
        while True:
            event = receive()
            if event.get("id") == identifier:
                if "error" in event:
                    raise RuntimeError("Codex App " + method + ": " + json.dumps(redact(event["error"])))
                return event.get("result") or {}
            if event.get("method"):
                deferred.append(event)

    try:
        rpc(1, "initialize", {"clientInfo": {"name": "cold-start-harness", "version": "1"},
                              "capabilities": {"experimentalApi": True}})
        _send(child, {"method": "initialized"})
        common = {"cwd": request["cwd"], "model": request["model"], "approvalPolicy": "never",
                  "sandbox": app_thread_sandbox(readonly=request["readonly"])}
        if session:
            opened = rpc(2, "thread/resume", {**common, "threadId": session})
        else:
            opened = rpc(2, "thread/start", {**common, "ephemeral": request.get("ephemeral", False)})
        thread = opened.get("thread", {})
        observed = thread.get("id")
        if not isinstance(observed, str) or not observed or (session and observed != session):
            raise RuntimeError("Codex App returned no matching thread")
        session = observed
        emit({"type": "thread.started", "thread_id": session})
        inputs = [{"type": "text", "text": Path(request["prompt_path"]).read_text(encoding="utf-8")}]
        inputs.extend({"type": "localImage", "path": path, "detail": "high"} for path in request.get("images", []))
        params = {"threadId": session, "cwd": request["cwd"], "input": inputs,
                  "model": request["model"], "approvalPolicy": "never",
                  "outputSchema": json.loads(Path(request["schema_path"]).read_text(encoding="utf-8")),
                  "sandboxPolicy": app_turn_sandbox_policy(
                      readonly=request["readonly"], writable_roots=request.get("writable_roots"))}
        if request.get("reasoning_effort"):
            params["effort"] = request["reasoning_effort"]
        started = rpc(3, "turn/start", params)
        active_turn = started.get("turn", {}).get("id")
        if not active_turn:
            raise RuntimeError("Codex App returned no turn ID")
        while True:
            event = deferred.pop(0) if deferred else receive()
            method = event.get("method")
            details = event.get("params") or {}
            if details.get("threadId") not in (None, session):
                continue
            if details.get("turnId") not in (None, active_turn):
                continue
            if method:
                emit(event)
            if method == "thread/tokenUsage/updated":
                # Only a thread created by this invocation has a proven zero
                # baseline. Its total includes every response/tool cycle in
                # this single turn. Never use `last` or resumed-thread totals.
                if new_thread and details.get("threadId") == session and details.get("turnId") == active_turn:
                    candidate = _tokens((details.get("tokenUsage") or {}).get("total"))
                    if "input_tokens" in candidate and "output_tokens" in candidate:
                        new_thread_usage = candidate
                continue
            if method == "item/agentMessage/delta":
                key = details.get("itemId", "message")
                deltas[key] = deltas.get(key, "") + str(details.get("delta", ""))
            if method in {"item/started", "item/completed"}:
                item = details.get("item") or {}
                if item.get("type") == "agentMessage" and method == "item/completed":
                    messages[item.get("id", "message")] = str(item.get("text", ""))
                if item.get("type") == "commandExecution":
                    emit({"type": method.replace("/", "."),
                          "item": {**item, "type": "command_execution"}})
            if method == "error" and not details.get("willRetry", False):
                raise RuntimeError(json.dumps(redact(details.get("error") or details)))
            if method == "turn/completed":
                turn = details.get("turn") or {}
                if turn.get("id") != active_turn:
                    continue
                if turn.get("status") != "completed":
                    raise RuntimeError("Codex App turn " + str(turn.get("status")) + ": " + json.dumps(redact(turn.get("error"))))
                usage = _tokens(turn.get("usage"))
                usage_source = "codex_app_turn_completed"
                if not usage and new_thread_usage:
                    usage = new_thread_usage
                    usage_source = "codex_app_new_thread_total"
                result = next(reversed(messages.values()), "") or next(reversed(deltas.values()), "")
                emit({"type": "result", "is_error": False, "subtype": "success", "result": result,
                      "session_id": session, "turn_id": active_turn, "usage": usage,
                      "usage_source": usage_source if usage else None, "usage_scope": "turn" if usage else None})
                return
    finally:
        _stop(child)


def kimi_command(request: dict) -> list[str]:
    prompt = Path(request["prompt_path"]).read_text(encoding="utf-8")
    # Avoid Windows' argv length limit; the read tool opens the exact turn file.
    if len(prompt.encode("utf-8")) > 6000:
        prompt = "Follow all instructions in " + request["prompt_path"] + ". Return its required JSON object as your final message."
    # Print mode already uses auto permission and rejects --auto/--plan/--yolo.
    command = [request["executable"], "-m", request["model"], "-p", prompt, "--output-format", "stream-json"]
    if request.get("session_id"):
        command.extend(["-S", request["session_id"]])
    elif request["readonly"]:
        profile = Path(request["prompt_path"]).with_name("kimi-reviewer.md")
        profile.write_text("---\nname: harness-reviewer\ndescription: Independent read-only mechanical evidence reviewer\ntools:\n  - Read\n  - Grep\n  - Glob\nsubagents: []\n---\n${base_prompt}\n\nReview evidence without changing files. Return the requested JSON in your final message.\n", encoding="utf-8")
        command.extend(["--agent-file", str(profile)])
    if not request["readonly"]:
        for root in request.get("writable_roots", [])[1:]:
            command.extend(["--add-dir", root])
    return command


def _kimi_wire_path(session_id):
    """Resolve only the requested CLI session, using its provider-owned index."""
    home = Path(os.environ.get("KIMI_CODE_HOME") or Path.home() / ".kimi-code")
    found = None
    try:
        with (home / "session_index.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get("sessionId") == session_id and entry.get("sessionDir"):
                    found = Path(entry["sessionDir"]) / "agents" / "main" / "wire.jsonl"
    except OSError:
        pass
    return found


def _kimi_usage(raw):
    if not isinstance(raw, dict):
        return {}
    billed = [raw[key] for key in ("inputOther", "inputCacheRead", "inputCacheCreation") if key in raw]
    output = raw.get("output")
    if not billed or not all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in [*billed, output]):
        return _tokens(raw)
    value = {"input_tokens": sum(billed), "output_tokens": output}
    if "inputCacheRead" in raw:
        value["cached_input_tokens"] = raw["inputCacheRead"]
    return value


def _kimi_wire_usage(path, offset, prompt):
    usage = {}
    matched = False
    try:
        with path.open("rb") as stream:
            stream.seek(offset)
            for line in stream:
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "turn.prompt":
                    # Only this invocation's prompt is allowed to contribute.
                    content = event.get("input") or []
                    if not isinstance(content, list):
                        content = []
                    candidate = event.get("prompt")
                    if candidate is None:
                        candidate = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
                    if matched:
                        break
                    matched = candidate == prompt
                if matched and event.get("type") == "usage.record" and event.get("usageScope") == "turn":
                    for key, number in _kimi_usage(event.get("usage")).items():
                        usage[key] = usage.get(key, 0) + number
    except OSError:
        return {}
    return usage


def kimi_turn(request: dict) -> None:
    command = kimi_command(request)
    previous_wire = _kimi_wire_path(request["session_id"]) if request.get("session_id") else None
    try:
        previous_offset = previous_wire.stat().st_size if previous_wire else None
    except OSError:
        previous_offset = None
    child = _launch(command, request["cwd"])
    child.stdin.close()
    session = request.get("session_id")
    text = ""
    final_seen = False
    usage = {}
    failed = False
    try:
        for line in child.stdout:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            emit(event)
            if event.get("type") == "session.resume_hint" and event.get("session_id"):
                session = event["session_id"]
                emit({"type": "thread.started", "thread_id": session})
            if event.get("type") in {"error", "turn.failed"} or event.get("is_error") is True:
                failed = True
            if event.get("type") == "turn.ended" and event.get("reason") != "completed":
                failed = True
            if event.get("role") == "assistant":
                content = event.get("content", "")
                if isinstance(content, list):
                    content = "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
                if isinstance(content, str) and content:
                    text = content
                    final_seen = not bool(event.get("tool_calls"))
                if event.get("tool_calls"):
                    final_seen = False
                for call in event.get("tool_calls") or []:
                    function = call.get("function") or {}
                    if function.get("name", "").lower() in {"bash", "shell"}:
                        emit({"type": "tool_call", "subtype": "started", "call_id": call.get("id"),
                              "tool_call": {"function": function}})
            if event.get("role") == "tool":
                final_seen = False
                emit({"type": "tool_call", "subtype": "completed", "call_id": event.get("tool_call_id"),
                      "tool_call": {"function": {"name": "bash", "arguments": {}}}})
            if event.get("type") == "usage.record" and event.get("usageScope") == "turn":
                for key, number in _kimi_usage(event.get("usage")).items():
                    usage[key] = usage.get(key, 0) + number
        code = child.wait()
        if code != 0 or failed:
            raise RuntimeError(f"Kimi CLI exited {code}; see streamed provider errors and stderr.log")
        if not text or not final_seen:
            raise RuntimeError("Kimi CLI returned no final assistant response")
        if not session:
            raise RuntimeError("Kimi CLI returned no session.resume_hint; continuation cannot be preserved")
        wire = _kimi_wire_path(session)
        if not usage and wire is not None:
            if not request.get("session_id"):
                usage = _kimi_wire_usage(wire, 0, command[command.index("-p") + 1])
            elif wire == previous_wire and previous_offset is not None:
                usage = _kimi_wire_usage(wire, previous_offset, command[command.index("-p") + 1])
        emit({"type": "result", "is_error": False, "subtype": "success", "result": text,
              "session_id": session, "usage": usage})
    finally:
        _stop(child)


def main():
    try:
        request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        if request["provider"] == "codex_app":
            app_turn(request)
        elif request["provider"] == "kimi":
            kimi_turn(request)
        else:
            raise ValueError("Unknown provider transport")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        kind = "launch" if isinstance(exc, OSError) else "transport"
        emit({"type": "result", "is_error": True, "subtype": "error", "result": str(exc),
              "provider_error": {"kind": kind}})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
