"""Attach trusted sandbox dispatch to the existing formal worker instances."""
from __future__ import annotations

import os
from pathlib import Path


def bind_worker(worker, workspace, project, role, author_visual_policy=None, *, context_window=None):
    from ..cold_start.claude_code_worker import ClaudeCodeWorker as DirectClaudeCodeWorker
    from ..cold_start.cursor_worker import CursorCliWorker
    from ..cold_start.provider_workers import ClaudeCodeWorker, CodexAppWorker, KimiCliWorker
    from ..cold_start.worker import CodexCliWorker
    from ..cold_start.dsh_worker import DshWorker
    from .budget import BudgetedWorker
    from .execution import formal_sandbox_required

    if not formal_sandbox_required():
        # Local distributions run the original CLI turns of every provider.
        return worker
    target = worker
    while isinstance(target, BudgetedWorker):
        target = target.worker
    anthropic = any(base.__name__ == 'AnthropicWorker'
        and base.__module__ == 'simulation_ready_cad_workflow.validation_pipeline.cold_start.anthropic_worker'
        for base in type(target).__mro__)
    # The direct Claude Code worker subclasses CursorCliWorker; reject it before it is bound as client cursor.
    if anthropic or isinstance(target, (CodexAppWorker, DirectClaudeCodeWorker)):
        raise ValueError('Formal sandbox execution supports the original Codex, Cursor and Kimi CLIs, the sandboxed Claude Code client and CPU DSH SDK only')
    if isinstance(target, ClaudeCodeWorker):
        client = 'claude'
    elif isinstance(target, KimiCliWorker):
        client = 'kimi'
    elif isinstance(target, DshWorker):
        client = 'dsh'
    elif isinstance(target, CodexCliWorker):
        client = 'codex'
    elif isinstance(target, CursorCliWorker):
        client = 'cursor'
    else:
        # Injected in-process workers do not launch a model or a host CLI.
        return worker
    if role not in ('author', 'reviewer'):
        raise ValueError('Formal worker role must be author or reviewer')
    workspace = Path(workspace).resolve()
    project = Path(project).resolve()
    if context_window is not None and (type(context_window) is not int or context_window <= 0):
        raise ValueError('Context window must be a positive integer')
    target.context_window = context_window
    configured = os.environ.get('CADHARNESS_SANDBOX_CONFIG')
    config = Path(configured).expanduser() if configured else workspace / '.runtime_runs/benchmark_sandbox/runtime.json'
    if not config.is_absolute():
        raise ValueError('CADHARNESS_SANDBOX_CONFIG must name an absolute trusted configuration file')
    from ..benchmark_sandbox.visual_policy import validate
    author_visual_policy = validate(author_visual_policy if role == 'author' else 'full')
    if author_visual_policy != 'full' and client != 'codex':
        raise ValueError('worker_no_visuals_v1 requires the original Codex author')
    context = {'workspace': str(workspace), 'project': str(project), 'client': client,
               'code_root': str(Path(__file__).resolve().parents[3]),
               'role': role, 'config_path': str(config.resolve()),
               'author_visual_policy': author_visual_policy}
    stop_file = os.environ.get('CADHARNESS_SANDBOX_STOP_FILE')
    if stop_file:
        path = Path(stop_file).expanduser()
        if not path.is_absolute():
            raise ValueError('CADHARNESS_SANDBOX_STOP_FILE must name an absolute owned stop file')
        context['stop_file'] = str(path.resolve())
    target.benchmark_sandbox_context = context
    return worker
