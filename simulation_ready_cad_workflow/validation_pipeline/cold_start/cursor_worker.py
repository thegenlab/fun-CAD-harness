"""Persistent Cursor CLI author and independent, ephemeral review turns."""

from __future__ import annotations

from ..trajectory.capture import record_turn, turn_environment, emit_public_event

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

from ..runtime.system_probe import terminate_process_tree
from .cli_invocation import cli_command
from .token_watch import TokenWatch
from .worker import RESPONSE_SCHEMA


DEFAULT_CURSOR_MODEL = "cursor-grok-4.6-xhigh-fast"
DEFAULT_CURSOR_EXECUTABLE = "cursor-agent"
_PROMPT_ARGV_LIMIT = 6000


def session_id_from_event(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    thread = event.get("thread_id")
    if event.get("type") == "thread.started" and isinstance(thread, str) and thread:
        return thread
    session = event.get("session_id")
    if isinstance(session, str) and session:
        return session
    return None


def _json_object(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    if payload.startswith("```"):
        payload = payload.removeprefix("```json").removeprefix("```").strip()
        if payload.endswith("```"):
            payload = payload[: payload.rfind("```")].strip()
    try:
        value = json.loads(payload)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    for index, char in enumerate(payload):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(payload, index)
        except ValueError:
            continue
        if isinstance(value, dict):
            found = value
    return found


def _schema_valid(response: dict[str, Any], schema: dict[str, Any], *, completed: bool) -> bool:
    if schema is not RESPONSE_SCHEMA and schema != RESPONSE_SCHEMA:
        try:
            from jsonschema import Draft202012Validator, ValidationError
        except ImportError:
            return completed and isinstance(response, dict)
        try:
            Draft202012Validator(schema).validate(response)
        except ValidationError as exc:
            raise ValueError(exc.message) from exc
        return completed and isinstance(response, dict)
    return (
        completed
        and isinstance(response, dict)
        and set(response) == set(RESPONSE_SCHEMA["required"])
        and response["status"] in RESPONSE_SCHEMA["properties"]["status"]["enum"]
        and isinstance(response["summary"], str)
        and all(
            isinstance(response[key], list)
            and all(isinstance(value, str) for value in response[key])
            for key in ("evidence_paths", "issues")
        )
    )


class CursorCliWorker:
    provider_label = "Cursor CLI"
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str = "high",
        executable: str = DEFAULT_CURSOR_EXECUTABLE,
        session_id: str | None = None,
        disabled_features: Sequence[str] | None = None,
    ) -> None:
        self.model = model or DEFAULT_CURSOR_MODEL
        self.reasoning_effort = reasoning_effort
        self.executable = executable or DEFAULT_CURSOR_EXECUTABLE
        self.session_id = session_id
        self.disabled_features = tuple(disabled_features) if disabled_features is not None else ()

    def _result_text(self, event: dict[str, Any]) -> str:
        return event["result"] if isinstance(event.get("result"), str) else ""

    def _error_text(self, event: dict[str, Any]) -> str:
        """Failed result event as reported and classified below."""
        return json.dumps(event, ensure_ascii=False)

    def _observe_event(self, event: dict[str, Any]) -> None:
        """Provider-specific turn metadata from one parsed stream event."""

    def _turn_prompt(
        self,
        prompt: str,
        *,
        images: Sequence[Path],
        schema_path: Path,
        response_path: Path,
    ) -> str:
        text = prompt
        if images:
            listed = "\n".join(str(Path(image).resolve()) for image in images)
            text += "\n\nRead these local evidence images:\n" + listed + "\n"
        return (
            text
            + "\n\nFinal response contract: write only a JSON object matching "
            + str(schema_path)
            + " as your last message, and write that same object to "
            + str(response_path)
            + ".\n"
        )

    def _command(
        self,
        workspace: Path,
        turn_prompt: str,
        prompt_path: Path,
        schema_path: Path,
        response_path: Path,
        *,
        readonly: bool,
        independent: bool,
        retain_independent_session: bool,
        independent_session_id: str | None,
    ) -> list[str]:
        command = [
            self.executable,
            "-p",
            "--output-format",
            "stream-json",
            "--trust",
            "--workspace",
            str(workspace),
        ]
        if readonly:
            command.extend(["--mode", "ask"])
        else:
            command.append("--force")
            project = getattr(self, "construction_project", None)
            if project is not None:
                project = Path(project).resolve()
                tool_outputs = project.parents[2] / ".runtime_runs" / "author_tools"
                tool_outputs.mkdir(parents=True, exist_ok=True)
                command.extend(["--add-dir", str(project), "--add-dir", str(tool_outputs)])
        if self.model is not None:
            command.extend(["--model", self.model])
        resume = None if independent and not retain_independent_session else (
            independent_session_id if independent_session_id is not None else (None if independent else self.session_id)
        )
        if resume:
            command.extend(["--resume", resume])
        if len(turn_prompt.encode("utf-8")) > _PROMPT_ARGV_LIMIT:
            command.append(
                f"Follow every instruction in {prompt_path}. "
                f"Your final message must be only the JSON object matching {schema_path}, "
                f"and you must write that same object to {response_path}."
            )
        else:
            command.append(turn_prompt)
        return command

    @record_turn
    def run_turn(
        self,
        workspace: Path,
        prompt: str,
        output_dir: Path,
        timeout_seconds: float,
        images: Sequence[Path] = (),
        readonly: bool = False,
        fresh: bool = False,
        response_schema: dict[str, Any] | None = None,
        retain_independent_session: bool = False,
        independent_session_id: str | None = None,
    ) -> dict[str, Any]:
        if retain_independent_session and not readonly:
            raise ValueError("Retained independent sessions are read-only")
        if independent_session_id is not None and not (readonly and retain_independent_session):
            raise ValueError("Independent resume requires a retained read-only session")
        workspace = Path(workspace).resolve()
        output_dir = Path(output_dir).resolve()
        launch = None
        context = getattr(self, 'benchmark_sandbox_context', None)
        if context is not None:
            from ..slim_harness.project_isolation import turns_for
            turns = turns_for(context)
            if turns is None:
                from ..benchmark_sandbox.formal_worker import run_turn as run_sandbox_turn
                return run_sandbox_turn(self, workspace, prompt, output_dir, timeout_seconds,
                    images=images, readonly=readonly, fresh=fresh, response_schema=response_schema,
                    retain_independent_session=retain_independent_session,
                    independent_session_id=independent_session_id)
            if turns.executable is not None:
                self.executable = str(turns.executable)
            launch = turns.launch()
        output_dir.mkdir(parents=True, exist_ok=True)
        independent = fresh or readonly
        response_path = output_dir / "response.json"
        response_path.unlink(missing_ok=True)
        schema_path = output_dir / "response-schema.json"
        schema = response_schema or RESPONSE_SCHEMA
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        prompt_path = output_dir / "prompt.txt"
        turn_prompt = self._turn_prompt(
            prompt, images=images, schema_path=schema_path, response_path=response_path
        )
        prompt_path.write_text(turn_prompt, encoding="utf-8")
        command = self._command(
            workspace,
            turn_prompt,
            prompt_path,
            schema_path,
            response_path,
            readonly=readonly,
            independent=independent,
            retain_independent_session=retain_independent_session,
            independent_session_id=independent_session_id,
        )

        started = time.monotonic()
        exit_code: int | None = None
        error: dict[str, Any] | None = None
        events_path = output_dir / "events.jsonl"
        stderr_path = output_dir / "stderr.log"
        controller = getattr(self, "budget_controller", None)
        from ..slim_harness.budget import ObservedWaitRecoveryRequired
        with events_path.open("w", encoding="utf-8") as events, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            token_watch = TokenWatch(
                output_dir,
                session_id=independent_session_id if independent_session_id is not None else None if independent else self.session_id,
            )
            process: subprocess.Popen[str] | None = None
            try:
                command = cli_command(command)
                process = subprocess.Popen(
                    command,
                    cwd=str(workspace),
                    stdin=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    stdout=events,
                    env=launch.environment(turn_environment()) if launch is not None else turn_environment(),
                    stderr=stderr,
                    shell=False,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                    **(launch.popen_options if launch is not None else {}),
                )
                token_watch.start()
                if getattr(self, "construction_project", None) is not None:
                    from ..slim_harness.construction_worker import communicate_budgeted
                    communicate_budgeted(process, turn_prompt, timeout_seconds, self.construction_project, events_path)
                elif controller is None:
                    process.communicate(input=turn_prompt, timeout=timeout_seconds)
                else:
                    controller.worker_observer_pid = process.pid
                    controller.save()
                    pending_input = turn_prompt
                    while True:
                        remaining = controller.worker_remaining(timeout_seconds, started)
                        if remaining <= 0:
                            raise subprocess.TimeoutExpired(command, timeout_seconds)
                        try:
                            process.communicate(input=pending_input, timeout=min(0.05, remaining))
                            break
                        except subprocess.TimeoutExpired:
                            pending_input = None
                    controller.refresh_worker_observations()
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                error = {
                    "kind": "timeout",
                    "message": f"{self.provider_label} turn exceeded its time budget.",
                    "cleanup_confirmed": False,
                    "survivor_pids": [],
                }
            except ObservedWaitRecoveryRequired as exc:
                error = {"kind": "resource_observation", "message": str(exc), "needs_external_action": True}
            except OSError as exc:
                error = {"kind": "launch", "message": str(exc)}
            finally:
                try:
                    if process is not None:
                        if process.poll() is None:
                            survivors = terminate_process_tree(process.pid, 2.0)
                            if process.poll() is None:
                                process.kill()
                            try:
                                process.wait(timeout=2.0)
                            except subprocess.TimeoutExpired:
                                survivors = (*survivors, process.pid)
                            if survivors:
                                error = error or {"kind": "cleanup", "message": ""}
                                error["message"] += (
                                    " Could not confirm process-tree cleanup for PIDs: "
                                    + ", ".join(str(pid) for pid in survivors)
                                )
                            if error is not None:
                                error["cleanup_confirmed"] = not survivors
                                error["survivor_pids"] = list(survivors)
                        exit_code = process.returncode
                        if process.stdin is not None:
                            process.stdin.close()
                finally:
                    token_watch.stop()
                if controller is not None:
                    controller.refresh_worker_observations()
                    if controller.worker_observation_issues:
                        error = error or {
                            "kind": "resource_observation",
                            "message": "; ".join(controller.worker_observation_issues),
                        }
                        error["needs_external_action"] = True

        usage: dict[str, Any] = {}
        usage_source: str | None = None
        observed_session_id: str | None = None
        event_errors: list[str] = []
        provider_error_kind: str | None = None
        completed = False
        result_text = ""
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            session = session_id_from_event(event)
            if session:
                observed_session_id = session
            if not isinstance(event, dict):
                continue
            self._observe_event(event)
            if event.get("type") == "result":
                if event.get("is_error") is False and not str(event.get("subtype") or "").startswith("error"):
                    completed = True
                    result_text = self._result_text(event)
                else:
                    event_errors.append(self._error_text(event))
                    if isinstance(event.get("provider_error"), dict):
                        provider_error_kind = event["provider_error"].get("kind")
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                    usage_source = event.get("usage_source") or "cursor_agent_result_usage"
            elif event.get("type") in {"error", "turn.failed"}:
                event_errors.append(json.dumps(event, ensure_ascii=False))
        if observed_session_id and not independent:
            self.session_id = observed_session_id
        token_watch.finish(usage=usage, completed=completed, source=usage_source)
        if hasattr(self, "_final_usage"):
            usage = self._final_usage(usage, token_watch)
        elif token_watch.value.get("usage"):
            usage = token_watch.value["usage"]
            usage_source = token_watch.value.get("source") or usage_source
        elapsed_seconds = time.monotonic() - started
        from ..mechanism.io import write_json_atomic
        write_json_atomic(output_dir / "timing.segments.json", {
            "schema_version": "slim_timing_segments_v1",
            "wall_seconds": elapsed_seconds,
            "observed_work_seconds": getattr(getattr(self, "budget_controller", None), "_worker_work_total", 0.0),
            "observed_wait_seconds": getattr(getattr(self, "budget_controller", None), "_worker_wait_total", 0.0),
            "observation_issues": list(getattr(controller, "worker_observation_issues", [])),
            "unclassified_provider_seconds": max(
                0.0,
                elapsed_seconds
                - getattr(getattr(self, "budget_controller", None), "_worker_work_total", 0.0)
                - getattr(getattr(self, "budget_controller", None), "_worker_wait_total", 0.0),
            ),
            "events": list(token_watch.arrivals),
            "overlap": {
                "unclassified_overlaps_active_upper_estimate": True,
                "unclassified_is_not_net_thinking": True,
                "token_counts_are_not_seconds": True,
            },
        })
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if error is None and (exit_code != 0 or event_errors):
            message = "\n".join(event_errors) or stderr_text.strip() or f"{self.provider_label} exited {exit_code}."
            lowered = (message + "\n" + stderr_text).lower()
            kind = provider_error_kind or "exit"
            if any(term in lowered for term in ("usage limit", "hit your", "quota", "rate_limit", "rate limit")):
                kind = "quota"
            # Result events carry token counts and UUIDs; match the status code only as a word.
            elif re.search(r"unauthorized|authentication|token_revoked|\b401\b|login", lowered):
                kind = "auth"
            elif "active writer" in lowered or "already running" in lowered:
                kind = "busy"
            error = {"kind": kind, "message": message}
        response: dict[str, Any] = {}
        if error is None:
            try:
                if response_path.is_file():
                    response = json.loads(response_path.read_text(encoding="utf-8"))
                if not isinstance(response, dict) or not response:
                    parsed = _json_object(result_text)
                    if parsed is None:
                        raise ValueError(f"{self.provider_label} did not complete with the required response schema.")
                    response = parsed
                if not _schema_valid(response, schema, completed=completed):
                    raise ValueError(f"{self.provider_label} did not complete with the required response schema.")
            except (OSError, ValueError, TypeError) as exc:
                error = {"kind": "response", "message": str(exc)}
        partial_response_path = None
        if error is not None:
            if retain_independent_session and response_path.is_file():
                partial_response_path = output_dir / "response.partial.json"
                partial_response_path.write_bytes(response_path.read_bytes())
            response = {
                "status": "provider_error",
                "summary": error["message"],
                "evidence_paths": [],
                "issues": [error["message"]],
                "provider_error": error,
            }
            response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        elif not response_path.is_file():
            response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
        result = {
            **response,
            "session_id": observed_session_id or independent_session_id or (None if independent else self.session_id),
            "usage": usage,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed_seconds,
        }
        if hasattr(self, "_provider_metadata"):
            result.update(self._provider_metadata())
        if retain_independent_session:
            result.update(
                observed_session_id=observed_session_id,
                requested_independent_session_id=independent_session_id,
                process_pid=process.pid if process is not None and isinstance(process.pid, int) else None,
                partial_response_path=str(partial_response_path) if partial_response_path else None,
            )
        if getattr(self, "construction_project", None) is not None:
            from ..slim_harness.construction_budget import ProjectBudget
            result["construction_budget"] = ProjectBudget(self.construction_project).snapshot()
        return result
