"""App-server and Kimi transports using the existing budgeted turn runner."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Sequence

from .cursor_worker import CursorCliWorker


DEFAULT_KIMI_MODEL = "kimi-code/k3"
DEFAULT_KIMI_EXECUTABLE = "kimi"


class _TransportWorker(CursorCliWorker):
    """Keep process ownership, response validation and trajectory capture shared."""

    def _final_usage(self, usage, watch):
        self._usage_source = watch.value.get("source") if usage or watch.value.get("usage") else None
        if self.provider == "codex_app" and not usage:
            return watch.value.get("usage") or {}
        return usage

    def _provider_metadata(self):
        metadata = {"provider": self.provider, "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "requested_reasoning_effort": getattr(self, "requested_reasoning_effort", self.reasoning_effort),
                "reasoning_effort_supported": self.provider != "kimi"}
        if self.provider == "codex_app" and getattr(self, "_usage_source", None):
            metadata.update(usage_source=self._usage_source, usage_scope="turn")
        return metadata

    def _command(
        self, workspace, turn_prompt, prompt_path, schema_path, response_path,
        *, readonly, independent, retain_independent_session, independent_session_id,
    ):
        resume = independent_session_id if retain_independent_session else None
        if not independent:
            resume = self.session_id
        roots = [str(workspace)]
        project = getattr(self, "construction_project", None)
        if project is not None and not readonly:
            project = Path(project).resolve()
            outputs = project.parents[2] / ".runtime_runs" / "author_tools"
            outputs.mkdir(parents=True, exist_ok=True)
            roots.extend([str(project), str(outputs)])
        request = {
            "provider": self.provider,
            "executable": self.executable,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "cwd": str(workspace),
            "prompt_path": str(prompt_path),
            "schema_path": str(schema_path),
            "readonly": readonly,
            "session_id": resume,
            "ephemeral": independent and not retain_independent_session,
            "writable_roots": roots,
            "disabled_features": list(self.disabled_features),
            "images": getattr(self, "_turn_images", []),
        }
        request_path = response_path.parent / "provider-request.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        return [sys.executable, "-X", "utf8", "-u", str(Path(__file__).with_name("provider_transport.py")), str(request_path)]

    def _turn_prompt(self, prompt, *, images, schema_path, response_path):
        self._turn_images = [str(Path(image).resolve()) for image in images]
        # The driver persists the final object. Read-only reviewers need no file writes.
        text = prompt
        if images and self.provider == "kimi":
            text += "\n\nRead these local evidence images:\n" + "\n".join(self._turn_images)
        return text + "\n\nReturn only a final JSON object matching this schema:\n" + schema_path.read_text(encoding="utf-8")


class CodexAppWorker(_TransportWorker):
    """Own a dedicated app-server process and resume only this worker's thread."""

    provider = "codex_app"
    provider_label = "Codex App"

    def __init__(self, model=None, reasoning_effort="high", executable="codex", session_id=None,
                 disabled_features: Sequence[str] | None = None):
        super().__init__(model=model or "gpt-6-astra", reasoning_effort=reasoning_effort,
                         executable=executable, session_id=session_id,
                         disabled_features=disabled_features if disabled_features is not None
                         else ("apps", "plugins", "multi_agent", "multi_agent_v2"))


class KimiCliWorker(_TransportWorker):
    provider = "kimi"
    provider_label = "Kimi CLI"

    def __init__(self, model=None, reasoning_effort=None, executable=DEFAULT_KIMI_EXECUTABLE,
                 session_id=None, disabled_features: Sequence[str] | None = None):
        # Kimi's CLI has no effort flag. Preserve the request separately, never
        # record an unsupported effort as an effective provider setting.
        self.requested_reasoning_effort = reasoning_effort
        super().__init__(model=model or DEFAULT_KIMI_MODEL, reasoning_effort=None,
                         executable=executable, session_id=session_id, disabled_features=disabled_features)
