"""One persistent CLI author and independent, ephemeral review turns."""

from __future__ import annotations

from ..trajectory.capture import record_turn, turn_environment, emit_public_event

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from ..runtime.system_probe import terminate_process_tree
from .token_watch import TokenWatch
from ..runtime.codex_linux_sandbox import cli_sandbox_command


RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "summary", "evidence_paths", "issues"],
    "properties": {
        "status": {
            "type": "string",
            "enum": ["ready", "needs_revision", "unsupported", "harness_defect"],
        },
        "summary": {"type": "string"},
        "evidence_paths": {"type": "array", "items": {"type": "string"}},
        "issues": {"type": "array", "items": {"type": "string"}},
    },
}


class CodexCliWorker:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str = "high",
        executable: str = "codex",
        session_id: str | None = None,
        disabled_features: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.executable = executable
        self.session_id = session_id
        self.disabled_features = tuple(disabled_features) if disabled_features is not None else ("apps", "plugins", "multi_agent", "multi_agent_v2")

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
        # A caller may retry a turn directory. Never accept its previous answer.
        response_path.unlink(missing_ok=True)
        (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        schema_path = output_dir / "response-schema.json"
        schema = response_schema or RESPONSE_SCHEMA
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        command = [
            self.executable, "exec", "--skip-git-repo-check", "--json", "--color", "never",
            "-c", 'service_tier="default"',
        ]
        for feature in self.disabled_features:
            command.extend(["--disable", feature])
        command.extend([
            "-C", str(workspace),
            "-c", "model_reasoning_effort=" + json.dumps(self.reasoning_effort),
            "--output-schema", str(schema_path), "-o", str(response_path),
        ])
        if self.model is not None:
            command.extend(["--model", self.model])
        if readonly:
            command.extend(cli_sandbox_command(readonly=True))
        else:
            sandbox = cli_sandbox_command(readonly=False)
            command.extend(sandbox or ["--approve-for-me"])
            project=getattr(self,'construction_project',None)
            if project is not None:
                project=Path(project).resolve();tool_outputs=project.parents[2]/'.runtime_runs'/'author_tools'
                tool_outputs.mkdir(parents=True,exist_ok=True)
                command.extend(['--add-dir',str(project),'--add-dir',str(tool_outputs)])
        if independent and not retain_independent_session:
            command.append("--ephemeral")
        if independent_session_id is not None:
            command.extend(["resume",independent_session_id])
        elif self.session_id and not independent:
            command.extend(["resume", self.session_id])
        for image in images:
            command.extend(["--image", str(Path(image).resolve())])
        command.append("-")
        if launch is not None:
            command = launch.codex_command(command, output_dir, sys.executable)

        started = time.monotonic()
        exit_code: int | None = None
        error: dict[str, Any] | None = None
        events_path = output_dir / "events.jsonl"
        stderr_path = output_dir / "stderr.log"
        controller=getattr(self,"budget_controller",None)
        from ..slim_harness.budget import ObservedWaitRecoveryRequired
        with events_path.open("w", encoding="utf-8") as events, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            token_watch = TokenWatch(
                output_dir, session_id=independent_session_id if independent_session_id is not None else None if independent else self.session_id,
                codex_home=launch.codex_home if launch is not None else None,
            )
            process: subprocess.Popen[str] | None = None
            try:
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
                if getattr(self, 'construction_project', None) is not None:
                    from ..slim_harness.construction_worker import communicate_budgeted
                    communicate_budgeted(process,prompt,timeout_seconds,self.construction_project,events_path)
                elif controller is None:
                    process.communicate(input=prompt, timeout=timeout_seconds)
                else:
                    controller.worker_observer_pid=process.pid
                    controller.save()
                    pending_input=prompt
                    while True:
                        remaining=controller.worker_remaining(timeout_seconds,started)
                        if remaining<=0:raise subprocess.TimeoutExpired(command,timeout_seconds)
                        try:
                            process.communicate(input=pending_input,timeout=min(.05,remaining))
                            break
                        except subprocess.TimeoutExpired:
                            pending_input=None
                    controller.refresh_worker_observations()
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                error = {"kind": "timeout", "message": "Codex turn exceeded its time budget.",
                         "cleanup_confirmed": False, "survivor_pids": []}
            except ObservedWaitRecoveryRequired as exc:
                error = {"kind": "resource_observation", "message": str(exc),
                         "needs_external_action": True}
            except OSError as exc:
                error = {"kind": "launch", "message": str(exc)}
            finally:
                try:
                    if process is not None:
                        if process.poll() is None:
                            # Enumerate descendants while the CLI parent still exists.
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
                        error = error or {"kind":"resource_observation",
                            "message":"; ".join(controller.worker_observation_issues)}
                        error["needs_external_action"]=True
        if launch is not None:
            try:
                launch.finish(output_dir)
            except (OSError, ValueError, RuntimeError) as exc:
                error = error or {"kind": "visual_guard", "message": f"No-visual guard audit is unavailable: {exc}"}

        usage: dict[str, Any] = {}
        observed_session_id: str | None = None
        event_errors: list[str] = []
        completed = False
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "thread.started":
                value = event.get("thread_id")
                if isinstance(value, str) and value:
                    observed_session_id = value
            elif event.get("type") == "turn.completed":
                completed = True
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
            elif event.get("type") in {"error", "turn.failed"}:
                event_errors.append(json.dumps(event, ensure_ascii=False))
                if event.get("type") == "turn.failed":
                    completed = False
        if observed_session_id and not independent:
            self.session_id = observed_session_id
        token_watch.finish(usage=usage, completed=completed)
        elapsed_seconds = time.monotonic() - started
        from ..mechanism.io import write_json_atomic
        write_json_atomic(output_dir / "timing.segments.json", {
            "schema_version": "slim_timing_segments_v1",
            "wall_seconds": elapsed_seconds,
            "observed_work_seconds": getattr(getattr(self,"budget_controller",None),"_worker_work_total",0.0),
            "observed_wait_seconds": getattr(getattr(self,"budget_controller",None),"_worker_wait_total",0.0),
            "observation_issues": list(getattr(controller,"worker_observation_issues",[])),
            "unclassified_provider_seconds": max(0.,elapsed_seconds-
                getattr(getattr(self,"budget_controller",None),"_worker_work_total",0.0)-
                getattr(getattr(self,"budget_controller",None),"_worker_wait_total",0.0)),
            "events": list(token_watch.arrivals),
            "overlap": {
                "unclassified_overlaps_active_upper_estimate": True,
                "unclassified_is_not_net_thinking": True,
                "token_counts_are_not_seconds": True,
            },
        })
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")
        # Reconnect notices can precede a successful completed turn. Keep
        # terminal failures and still require the final response schema below.
        if error is None and (exit_code != 0 or (event_errors and not completed)):
            message = "\n".join(event_errors) or stderr_text.strip() or f"Codex exited {exit_code}."
            lowered = message.lower()
            kind = "exit"
            if any(term in lowered for term in ("usage limit", "quota", "rate_limit", "rate limit")):
                kind = "quota"
            elif any(term in lowered for term in ("unauthorized", "authentication", "token_revoked", "401", "login")):
                kind = "auth"
            error = {"kind": kind, "message": message}
        response: dict[str, Any] = {}
        if error is None:
            try:
                response = json.loads(response_path.read_text(encoding="utf-8"))
                valid = (
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
                if response_schema is not None:
                    from jsonschema import Draft202012Validator, ValidationError

                    try:
                        Draft202012Validator(response_schema).validate(response)
                    except ValidationError as exc:
                        raise ValueError(exc.message) from exc
                    valid = completed and isinstance(response, dict)
                if not valid:
                    raise ValueError("Codex did not complete with the required response schema.")
            except (OSError, ValueError, TypeError) as exc:
                error = {"kind": "response", "message": str(exc)}
        partial_response_path=None
        if error is not None:
            if retain_independent_session and response_path.is_file():
                partial_response_path=output_dir/"response.partial.json"
                partial_response_path.write_bytes(response_path.read_bytes())
            response = {
                "status": "provider_error", "summary": error["message"],
                "evidence_paths": [], "issues": [error["message"]],
                "provider_error": error,
            }
            response_path.write_text(
                json.dumps(response, ensure_ascii=False), encoding="utf-8"
            )
        result = {
            **response,
            "session_id": observed_session_id or independent_session_id or (None if independent else self.session_id),
            "usage": usage,
            "exit_code": exit_code,
            "elapsed_seconds": elapsed_seconds,
        }
        if retain_independent_session:
            result.update(observed_session_id=observed_session_id,
                          requested_independent_session_id=independent_session_id,
                          process_pid=process.pid if process is not None and isinstance(process.pid,int) else None,
                          partial_response_path=str(partial_response_path) if partial_response_path else None)
        if getattr(self,'construction_project',None) is not None:
            from ..slim_harness.construction_budget import ProjectBudget
            result['construction_budget']=ProjectBudget(self.construction_project).snapshot()
        return result
