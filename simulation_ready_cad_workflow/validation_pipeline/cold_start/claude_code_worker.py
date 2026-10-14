"""Claude Code CLI (``claude -p``) author and independent review turns."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

from ..slim_harness.execution import CONTAINER_ENV
from .cursor_worker import CursorCliWorker


DEFAULT_CLAUDE_CODE_MODEL = "claude-opus-5-5"
DEFAULT_CLAUDE_CODE_EXECUTABLE = "claude"
_EFFORTS = {"ultra": "max"}
_REVIEW_TOOLS = "Read,Grep,Glob"


class ClaudeCodeWorker(CursorCliWorker):
    """Prompt on stdin, final object through ``--json-schema`` structured output."""

    provider = "claude_code"
    provider_label = "Claude Code"

    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str = "high",
        executable: str = DEFAULT_CLAUDE_CODE_EXECUTABLE,
        session_id: str | None = None,
        disabled_features: Sequence[str] | None = None,
    ) -> None:
        super().__init__(model=model or DEFAULT_CLAUDE_CODE_MODEL, reasoning_effort=reasoning_effort,
                         executable=executable or DEFAULT_CLAUDE_CODE_EXECUTABLE, session_id=session_id,
                         disabled_features=disabled_features)
        self._turn: dict[str, Any] = {}

    @property
    def effort(self) -> str:
        return _EFFORTS.get(self.reasoning_effort, self.reasoning_effort)

    def _turn_prompt(self, prompt, *, images, schema_path, response_path):
        self._turn = {}
        if not images:
            return prompt
        listed = "\n".join(str(Path(image).resolve()) for image in images)
        return prompt + "\n\nRead these local evidence images with the Read tool:\n" + listed + "\n"

    def _command(
        self, workspace, turn_prompt, prompt_path, schema_path, response_path,
        *, readonly, independent, retain_independent_session, independent_session_id,
    ):
        command = [
            self.executable, "-p", "--output-format", "stream-json", "--verbose",
            "--json-schema", schema_path.read_text(encoding="utf-8"),
            "--model", self.model, "--effort", self.effort,
        ]
        roots: list[str] = []
        project = getattr(self, "construction_project", None)
        if project is not None and not readonly:
            project = Path(project).resolve()
            tool_outputs = project.parents[2] / ".runtime_runs" / "author_tools"
            tool_outputs.mkdir(parents=True, exist_ok=True)
            roots = [str(project), str(tool_outputs)]
        if readonly:
            command.extend(["--permission-mode", "dontAsk", "--tools", _REVIEW_TOOLS, "--allowedTools", _REVIEW_TOOLS])
        elif os.environ.get(CONTAINER_ENV) == "1":
            command.extend(["--permission-mode", "bypassPermissions"])
        else:
            # Bash runs only in Claude Code's own sandbox; a host that cannot
            # start it fails at launch instead of running the shell unconfined.
            sandbox = {"enabled": True, "failIfUnavailable": True,
                       "autoAllowBashIfSandboxed": True, "allowUnsandboxedCommands": False}
            if sys.platform.startswith("linux"):
                # Its Unix-socket filter needs a nested user namespace, which Ubuntu's
                # AppArmor bwrap profile denies, failing every command. Unfiltered,
                # commands can reach local services (e.g. the systemd user bus) and
                # through them run outside the sandbox.
                sandbox["network"] = {"allowAllUnixSockets": True}
            if roots:
                sandbox["filesystem"] = {"allowWrite": roots}
            command.extend(["--permission-mode", "acceptEdits", "--settings", json.dumps({"sandbox": sandbox})])
        for root in roots:
            command.extend(["--add-dir", root])
        if independent and not retain_independent_session:
            command.append("--no-session-persistence")
        else:
            resume = independent_session_id if independent_session_id is not None else (None if independent else self.session_id)
            if resume:
                command.extend(["--resume", resume])
        return command

    def _result_text(self, event):
        if event.get("structured_output") is not None:
            return json.dumps(event["structured_output"], ensure_ascii=False)
        return super()._result_text(event)

    def _error_text(self, event):
        # The classifier scans this text; token counts, timings and model usage are not evidence.
        return json.dumps({key: event[key] for key in ("subtype", "is_error", "api_error_status", "result", "errors")
                           if key in event}, ensure_ascii=False)

    def _observe_event(self, event):
        if event.get("type") == "system" and event.get("subtype") == "init" and isinstance(event.get("model"), str):
            self._turn["model"] = event["model"]
        elif event.get("type") == "result":
            self._turn.update({key: event[key] for key in ("total_cost_usd", "num_turns") if key in event})

    def _provider_metadata(self):
        return {"provider": self.provider, "model": self._turn.get("model", self.model),
                "requested_model": self.model, "reasoning_effort": self.effort,
                "requested_reasoning_effort": self.reasoning_effort,
                **{key: self._turn[key] for key in ("total_cost_usd", "num_turns") if key in self._turn}}
