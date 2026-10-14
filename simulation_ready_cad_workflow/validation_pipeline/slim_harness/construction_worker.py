"""Bound a CLI model turn by effective project time, including nested tools."""
from __future__ import annotations

import json
import re
import shlex
import subprocess
from contextlib import ExitStack
from pathlib import Path

from .construction_budget import ProjectBudget


_SHELL_TOOL_KEYS = ("shellToolCall", "bashToolCall", "execToolCall")
_SHELL_FUNCTION_NAMES = frozenset({
    "shell", "bash", "zsh", "cmd", "powershell",
    "command_execution", "run_terminal_cmd", "runcommand",
})
_SLIM_MODULE = "simulation_ready_cad_workflow.validation_pipeline.slim_harness"
_SLIM_INVOCATION = re.compile(
    r"(?:^|[;&|\s])-m\s+" + _SLIM_MODULE.replace(".", r"\.") + r"(?P<runtime>\.runtime)?(?=\s|$)(?P<rest>[^;&|]*)")


def _simulation_command(command):
    """True for a shell command that blocks on one bounded simulation.

    Matches the runtime module directly and the Slim CLI's simulate/probe
    subcommands; global CLI options may precede the subcommand.
    """
    match = _SLIM_INVOCATION.search(command or "")
    if match is None:
        return False
    if match.group("runtime"):
        return True
    try:
        tokens = shlex.split(match.group("rest"))
    except ValueError:
        return False
    index = 0
    while index < len(tokens) and tokens[index].startswith("--"):
        index += 1 if "=" in tokens[index] else 2
    return index < len(tokens) and tokens[index] in {"simulate", "probe"}


def _function_command(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return arguments
    if isinstance(arguments, dict):
        return str(arguments.get("command") or arguments.get("fullCommand") or arguments.get("cmd") or "")
    return ""


def shell_execution_events(event):
    """Return [(phase, identifier, command)] for Codex, Cursor or Claude CLI shell events.

    One Claude message can batch several parallel tool calls, so every matching
    block is returned; a dropped tool_result would leak its exclusion.
    """
    if not isinstance(event, dict):
        return []
    item = event.get("item")
    if isinstance(item, dict) and item.get("type") == "command_execution":
        phase = {"item.started": "started", "item.completed": "completed", "item.failed": "failed"}.get(event.get("type"))
        return [(phase, item.get("id"), item.get("command", ""))] if phase else []
    if event.get("type") != "tool_call":
        # Claude Code stream-json: an assistant tool_use starts the shell call
        # and the matching user tool_result ends it.
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            return []
        if event.get("type") == "assistant":
            return [("started", block.get("id"), _function_command(block.get("input")))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                    and str(block.get("name") or "").lower() in _SHELL_FUNCTION_NAMES]
        if event.get("type") == "user":
            return [("completed", block.get("tool_use_id"), "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "tool_result"]
        return []
    phase = {"started": "started", "completed": "completed", "failed": "failed", "error": "failed"}.get(event.get("subtype"))
    tool = event.get("tool_call")
    if phase is None or not isinstance(tool, dict):
        return []
    command = None
    for key in _SHELL_TOOL_KEYS:
        payload = tool.get(key)
        if isinstance(payload, dict):
            command = _function_command(payload.get("args"))
            break
    function = tool.get("function")
    if command is None and isinstance(function, dict) and str(function.get("name") or "").lower() in _SHELL_FUNCTION_NAMES:
        command = _function_command(function.get("arguments"))
    if command is None:
        return []
    return [(phase, event.get("call_id"), command)]


def communicate_budgeted(process,prompt,timeout_seconds,project,events_path):
    budget=ProjectBudget(project);offset=0;pending=b'';install_items=set();simulation_items=set()
    # Only executable package-manager commands are auto-excluded. All unknown
    # shell work remains charged; explicit pauses cover human intervention.
    setup=re.compile(r'(?:^|[;&|]\s*|\s)(?:uv\s+(?:pip\s+install|sync|venv)|pip3?\s+install|conda\s+(?:create|install|env\s+create)|npm\s+(?:ci|install)|npx\s+playwright\s+install)\b')
    start_active=budget.snapshot()['active_seconds']
    with budget.activity('agent') as activity,ExitStack() as installations,ExitStack() as simulations:
        first=True
        while True:
            if Path(events_path).exists():
                with Path(events_path).open('rb') as stream:
                    stream.seek(offset);pending+=stream.read();offset=stream.tell()
                lines=pending.split(b'\n');pending=lines.pop()
                for line in lines:
                    try:event=json.loads(line)
                    except (ValueError,UnicodeDecodeError):continue
                    # Automatic exclusions belong to this monitored
                    # invocation. A hard-killed owner loses its lease;
                    # unlike a manual pause it cannot suspend the next
                    # worker forever. Never write/resume a user's pause.
                    # A simulation command excludes only its own blocking span;
                    # a detached background simulation keeps later work charged.
                    for phase,identifier,command in shell_execution_events(event):
                        if phase=='started' and setup.search(command or ''):
                            if not install_items:
                                installations.enter_context(budget.activity('environment_setup'))
                            install_items.add(identifier)
                        if phase=='started' and _simulation_command(command):
                            if not simulation_items:
                                simulations.enter_context(budget.activity('simulation_wait'))
                            simulation_items.add(identifier)
                        if phase in {'completed','failed'} and identifier in install_items:
                            install_items.discard(identifier)
                            if not install_items:installations.close()
                        if phase in {'completed','failed'} and identifier in simulation_items:
                            simulation_items.discard(identifier)
                            if not simulation_items:simulations.close()
            if Path(events_path).parent.name.endswith('_author'):
                from .construction import author_turn_timeout
                from .test_coverage import FirstPassReserveReached
                try:author_turn_timeout(project)
                except FirstPassReserveReached:
                    # Let the worker close its process tree and parse the session ID.
                    # The workflow then persists the handoff with the updated test set.
                    raise subprocess.TimeoutExpired(process.args,timeout_seconds)
            activity.check_available()
            snap=budget.snapshot()
            if snap['work_remaining_seconds']<=0 or snap['active_seconds']-start_active>=timeout_seconds:
                raise subprocess.TimeoutExpired(process.args,timeout_seconds)
            try:
                process.communicate(input=prompt if first else None,timeout=.5)
                return
            except subprocess.TimeoutExpired:
                first=False
