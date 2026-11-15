"""LLM worker factory for Codex CLI/App, Cursor, Kimi, Claude Code and Anthropic."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any, Sequence

from .claude_code_worker import DEFAULT_CLAUDE_CODE_EXECUTABLE, DEFAULT_CLAUDE_CODE_MODEL
from .cursor_worker import DEFAULT_CURSOR_EXECUTABLE, DEFAULT_CURSOR_MODEL
from .provider_workers import DEFAULT_CLAUDE_MODEL, DEFAULT_KIMI_EXECUTABLE, DEFAULT_KIMI_MODEL


# The Anthropic SDK worker and the formal sandbox backends (Claude Code, DSH)
# are not part of every distribution.
_ANTHROPIC = ("anthropic",) if find_spec(__package__ + ".anthropic_worker") is not None else ()
_SANDBOX = __package__.rsplit(".", 1)[0] + ".benchmark_sandbox"
_FORMAL_CLAUDE = find_spec(_SANDBOX + ".claude_cpu") is not None
_DSH = ("dsh",) if find_spec(_SANDBOX + ".dsh_cpu") is not None else ()
WORKER_PROVIDERS = (*_DSH, "codex", "claude_code", "claude", "codex_app", *_ANTHROPIC, "cursor", "cursor_cli",
                    "cursor_agent", "kimi", "kimi_cli")
# "claude_code" is the Claude Code CLI run directly. "claude" is the same CLI
# inside the formal sandbox where that backend is installed, and another name
# for "claude_code" everywhere else.
_CURSOR_ALIASES = {"cursor": "cursor", "cursor_cli": "cursor", "cursor_agent": "cursor", "kimi_cli": "kimi",
                   **({} if _FORMAL_CLAUDE else {"claude": "claude_code"})}
_DEFAULT_MODELS = {
    "dsh": "deepseek-flash",
    "codex": "gpt-6-astra",
    "anthropic": "claude-opus-4-6",
    "cursor": DEFAULT_CURSOR_MODEL,
    "codex_app": "gpt-6-astra",
    "kimi": DEFAULT_KIMI_MODEL,
    "claude": DEFAULT_CLAUDE_MODEL,
    "claude_code": DEFAULT_CLAUDE_CODE_MODEL,
}
_DEFAULT_EXECUTABLES = {
    "cursor": DEFAULT_CURSOR_EXECUTABLE,
    "kimi": DEFAULT_KIMI_EXECUTABLE,
    "claude_code": DEFAULT_CLAUDE_CODE_EXECUTABLE,
}


def normalize_worker_provider(provider: str) -> str:
    return _CURSOR_ALIASES.get(provider, provider)


def default_worker_model(provider: str, model: str | None = None) -> str:
    if model:
        return model
    return _DEFAULT_MODELS.get(normalize_worker_provider(provider), "gpt-6-astra")


def default_worker_executable(provider: str, executable: str | None = None) -> str:
    default = _DEFAULT_EXECUTABLES.get(normalize_worker_provider(provider))
    if default is not None and executable in {None, "", "codex"}:
        return default
    return executable or "codex"


def create_worker(
    provider: str = "codex",
    model: str | None = None,
    reasoning_effort: str | None = None,
    executable: str = "codex",
    session_id: str | None = None,
    disabled_features: Sequence[str] | None = None,
    **kwargs: Any,
):
    provider = normalize_worker_provider(provider)
    if reasoning_effort is None:
        reasoning_effort = "high" if provider == "dsh" else "medium"
    if provider == "claude":
        from .provider_workers import ClaudeCodeWorker
        return ClaudeCodeWorker(model=default_worker_model(provider, model), reasoning_effort=reasoning_effort,
                                executable="claude" if executable == "codex" else executable,
                                session_id=session_id, disabled_features=disabled_features)
    if provider in {"codex_app", "kimi"}:
        from .provider_workers import CodexAppWorker, KimiCliWorker
        cls = CodexAppWorker if provider == "codex_app" else KimiCliWorker
        return cls(model=default_worker_model(provider, model), reasoning_effort=reasoning_effort,
                   executable=default_worker_executable(provider, executable), session_id=session_id,
                   disabled_features=disabled_features)
    if provider == "dsh":
        from .dsh_worker import DshWorker
        return DshWorker(model=default_worker_model(provider, model), reasoning_effort=reasoning_effort,
                         executable="dsh" if executable == "codex" else executable, session_id=session_id)
    if provider == "codex":
        from .worker import CodexCliWorker
        return CodexCliWorker(
            model=model,
            reasoning_effort=reasoning_effort,
            executable=executable,
            session_id=session_id,
            disabled_features=disabled_features,
        )
    if provider == "anthropic":
        from .anthropic_worker import AnthropicWorker
        return AnthropicWorker(
            model=model or default_worker_model(provider),
            reasoning_effort=reasoning_effort,
            session_id=session_id,
            disabled_features=disabled_features,
            **kwargs,
        )
    if provider == "cursor":
        from .cursor_worker import CursorCliWorker
        return CursorCliWorker(
            model=default_worker_model(provider, model),
            reasoning_effort=reasoning_effort,
            executable=default_worker_executable(provider, executable),
            session_id=session_id,
            disabled_features=disabled_features,
        )
    if provider == "claude_code":
        from .claude_code_worker import ClaudeCodeWorker
        return ClaudeCodeWorker(
            model=default_worker_model(provider, model),
            reasoning_effort=reasoning_effort,
            executable=default_worker_executable(provider, executable),
            session_id=session_id,
            disabled_features=disabled_features,
        )
    raise ValueError(
        f"Unknown worker provider: {provider!r}. Use one of {', '.join(WORKER_PROVIDERS)}."
    )
