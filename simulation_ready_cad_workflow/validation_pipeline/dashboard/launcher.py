"""Start, stop and resume Slim Harness runs from the dashboard.

The CLI owns each run. The dashboard spawns ``slim_harness create`` or
``resume`` as a detached process, records its identity under
``.runtime_runs/dashboard_launches/<project>-<UTC time>/`` and only signals
that exact process (pid plus creation time).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from ..mechanism.io import write_json_atomic

CODE_ROOT = Path(__file__).resolve().parents[3]
MODULE = "simulation_ready_cad_workflow.validation_pipeline.slim_harness"
SLIM_SCHEMA = "slim_harness_workflow_v1"
# Per-simulation wall-time caps offered in the GUI, in seconds.
SIMULATION_TIMEOUTS = (60, 180, 300, 600, 900, 1200)
EXPERIMENTAL_PROVIDERS = frozenset({"codex_app", "cursor", "kimi"})
START_WAIT_SECONDS = 30.0
STOP_ESCALATE_SECONDS = 100.0
# provider -> (CLI on PATH, login probe, credential variables that replace the login)
AGENTS = {
    "codex": ("codex", ("login", "status"), ("OPENAI_API_KEY",)),
    "codex_app": ("codex", ("login", "status"), ("OPENAI_API_KEY",)),
    "claude_code": ("claude", ("auth", "status"), ("ANTHROPIC_API_KEY",)),
    "anthropic": (None, None, ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY")),
    "cursor": ("cursor-agent", None, ()),
    "kimi": ("kimi", None, ()),
}
_FIELDS = frozenset({"mode", "project", "task", "provider", "model", "reasoning", "tier",
                     "budget_hours", "simulation_timeout_seconds", "chrono_python",
                     "reviewer_provider", "reviewer_model", "reviewer_reasoning"})
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@+\[\]-]{0,127}")
_LOCK = threading.Lock()
_CHILDREN: dict[int, subprocess.Popen] = {}
_STOPPING: set[tuple[int, float]] = set()
_CHECKS: dict[str, dict] = {}
_CHECKS_LOCK = threading.Lock()


class LaunchError(ValueError):
    def __init__(self, code: str, message: str, status: int = 400, log: str | None = None):
        super().__init__(message)
        self.code, self.status, self.log = code, status, log


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _python() -> str:
    return os.environ.get("SIMREADY_CAD_PYTHON") or sys.executable


def _environment() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("SIMREADY_CONSTRUCTION_", "SIMREADY_TRAJECTORY_"))}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(CODE_ROOT), env.get("PYTHONPATH"))))
    return env


def _detached() -> dict:
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    return {"start_new_session": True}


def _projects(workspace: Path) -> Path:
    return workspace / ".runtime_runs" / "projects"


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, stream.seek(0, os.SEEK_END) - limit))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def check_workspace(workspace, runs_root=None) -> None:
    workspace = Path(workspace).resolve()
    if workspace != CODE_ROOT:
        raise LaunchError("LAUNCH_WORKSPACE", f"Launching needs --workspace-root to be this code checkout ({CODE_ROOT})", 409)
    if runs_root is not None and Path(runs_root).resolve() != workspace / ".runtime_runs":
        raise LaunchError("LAUNCH_WORKSPACE", f"Launching needs --runs-root {workspace / '.runtime_runs'}", 409)


def _project_id(value) -> str:
    from ..run_layout import RunLayoutError, validate_project_id
    if not isinstance(value, str):
        raise LaunchError("LAUNCH_PROJECT_INVALID", "Enter a project id")
    try:
        project = validate_project_id(value)
    except RunLayoutError as exc:
        raise LaunchError("LAUNCH_PROJECT_INVALID", "Use letters, digits, '.', '_' or '-' (max 128) for the project id") from exc
    if project != value:
        raise LaunchError("LAUNCH_PROJECT_INVALID", "The project id must not contain surrounding spaces")
    return project


def _providers() -> list[dict]:
    from ..cold_start.worker_factory import WORKER_PROVIDERS, default_worker_model, normalize_worker_provider
    return [{"id": name, "default_model": default_worker_model(name),
             "default_reasoning": "high" if name == "codex" else "medium",
             "experimental": name in EXPERIMENTAL_PROVIDERS}
            for name in WORKER_PROVIDERS if normalize_worker_provider(name) == name]


def _examples() -> list[dict]:
    root = CODE_ROOT / "examples" / "tasks"
    return [{"name": path.stem, "title": path.stem.replace("_", " ").capitalize(), "text": path.read_text(encoding="utf-8")}
            for path in sorted(root.glob("*.md"))] if root.is_dir() else []


# Claude Code accepts these model ids; Codex lists its own in ~/.codex/models_cache.json.
CLAUDE_CODE_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1")


def _models() -> dict[str, list[dict]]:
    """Model suggestions per provider; the field still accepts any model id."""
    cache = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "models_cache.json"
    codex = [{"id": item["slug"], "label": item.get("display_name") or item["slug"],
              "reasoning": [level.get("effort") for level in item.get("supported_reasoning_levels") or []
                            if isinstance(level, dict) and level.get("effort")]}
             for item in _read(cache).get("models") or []
             if isinstance(item, dict) and item.get("slug") and item.get("visibility", "list") == "list"]
    return {"codex": codex, "codex_app": codex,
            "claude_code": [{"id": model, "label": model, "reasoning": []} for model in CLAUDE_CODE_MODELS]}


def _probe(argv, timeout: float) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                   errors="replace", timeout=timeout, env=_environment())
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    return completed.returncode, (completed.stdout + completed.stderr).strip()


def _python_check(python: str | None, modules: tuple[str, ...]) -> dict:
    if not python:
        return {"ok": False, "path": "", "detail": "not configured"}
    code, output = _probe([python, "-c", "import " + ", ".join(modules)], 180)
    return {"ok": code == 0, "path": python, "detail": "" if code == 0 else output[-600:]}


def _agent_check(provider: str) -> dict:
    executable, login, keys = AGENTS.get(provider, (None, None, ()))
    item = {"provider": provider, "experimental": provider in EXPERIMENTAL_PROVIDERS, "cli": executable}
    found = shutil.which(executable) if executable else None
    if executable:
        item["cli_found"] = bool(found)
        code, output = _probe([found, "--version"], 20) if found else (None, "")
        item["version"] = output.splitlines()[0][:160] if code == 0 and output else ""
    key = next((name for name in keys if os.environ.get(name)), None)
    if key:
        item["auth"], item["auth_detail"] = "env", key
    elif login and found:
        code, output = _probe([found, *login], 20)
        detail = output.splitlines()[0][:160] if output else ""
        try:
            # The JSON status can be followed by CLI notices on stderr.
            start = output.find("{")
            status = json.JSONDecoder().raw_decode(output, start)[0] if start >= 0 else json.loads(output)
            logged_in = code == 0 and status.get("loggedIn") is not False
            # Claude Code reports JSON; show how it is logged in rather than its first line.
            detail = str(status.get("authMethod") or "")[:160]
        except (ValueError, AttributeError):
            logged_in = code == 0
        item["auth"], item["auth_detail"] = ("login" if logged_in else "missing"), detail
    else:
        item["auth"] = "missing" if keys or login else "unchecked"
    item["ok"] = (not executable or bool(found)) and item["auth"] != "missing"
    if provider == "claude_code" and sys.platform.startswith("linux") and os.environ.get("SIMREADY_CONTAINER_ISOLATION") != "1":
        # Claude Code author turns require its own Bash sandbox outside a container.
        missing = [tool for tool in ("bwrap", "socat") if not shutil.which(tool)]
        if missing:
            item["ok"], item["sandbox_missing"] = False, missing
    return item


def _run_checks(target: dict) -> None:
    jobs = {"cad_python": lambda: _python_check(_python(), ("build123d", "OCP", "trimesh")),
            "chrono_python": lambda: _python_check(os.environ.get("SIMREADY_PYCHRONO_EXECUTABLE"), ("pychrono",))}
    for item in _providers():
        jobs["provider:" + item["id"]] = lambda name=item["id"]: _agent_check(name)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {key: pool.submit(job) for key, job in jobs.items()}
        items = [{"id": key, **future.result()} for key, future in futures.items()]
    with _CHECKS_LOCK:
        target.update(items=items, ready=True, checked_at=_now())


def _checks(refresh: bool) -> dict:
    """Environment probes run once in the background; the page polls until ready."""
    with _CHECKS_LOCK:
        current = _CHECKS.get("value")
        if current is None or (refresh and current["ready"]):
            current = _CHECKS["value"] = {"ready": False, "items": [], "checked_at": None}
            threading.Thread(target=_run_checks, args=(current,), name="launch-environment-checks", daemon=True).start()
        return dict(current)


def options(workspace, *, runs_root=None, refresh: bool = False, limit: int = 4) -> dict:
    from ..runtime.bounded_subprocess import BoundedSubprocessError
    from ..slim_harness import execution, modes
    from ..slim_harness.budget import TIERS, recommended_limits
    from ..slim_harness.cli import REASONING_EFFORTS
    from ..slim_harness.construction_budget import BUDGET_HOURS
    workspace = Path(workspace).resolve()
    try:
        check_workspace(workspace, runs_root)
        unavailable = None
    except LaunchError as exc:
        unavailable = {"error": exc.code, "detail": str(exc)}
    try:
        execution_mode, execution_error = execution.mode(), None
    except BoundedSubprocessError as exc:
        execution_mode, execution_error = None, str(exc)
    return {
        "unavailable": unavailable,
        "modes": list(modes.available_modes(workspace)),
        "examples": _examples(),
        "providers": _providers(),
        "models": _models(),
        "reasoning": list(REASONING_EFFORTS),
        # hours / simulation_timeout_seconds: the recommended limits a new run of the tier gets.
        "tiers": [{"id": tier.name, "hours": recommended_limits(tier.name)[0],
                   "simulation_timeout_seconds": recommended_limits(tier.name)[1],
                   "max_tasks": tier.max_tasks_per_design, "max_parts": tier.max_parts} for tier in TIERS.values()],
        "default_tier": "high",
        "budget_hours": list(BUDGET_HOURS),
        "simulation_timeouts": list(SIMULATION_TIMEOUTS),
        "reviewer": {"default_provider": "codex", "default_reasoning": "medium", "applies_to": ["codex", "cursor"]},
        "execution": {"mode": execution_mode, "error": execution_error},
        "chrono_python": os.environ.get("SIMREADY_PYCHRONO_EXECUTABLE", ""),
        "concurrency": {"limit": limit, "active": [record["project"] for _path, record in _records(workspace) if _alive(record)]},
        "queue": queue(workspace),
        "checks": _checks(refresh),
    }


def _optional(body: dict, key: str, kind=str):
    value = body.get(key)
    if value in (None, ""):
        return None
    if type(value) is not kind:
        raise LaunchError("LAUNCH_FIELD_INVALID", f"{key} has the wrong type")
    return value


def _choice(body: dict, key: str, allowed, default=None):
    value = _optional(body, key)
    if value is None:
        return default
    if value not in allowed:
        raise LaunchError("LAUNCH_FIELD_INVALID", f"Unsupported {key}: {value}")
    return value


def _name(body: dict, key: str):
    value = _optional(body, key)
    if value is not None and not _NAME.fullmatch(value):
        raise LaunchError("LAUNCH_FIELD_INVALID", f"{key} must be a single model name")
    return value


def validate(workspace, body: dict, *, runs_root=None) -> dict:
    """Normalize one free-creation request with the same rules as slim_harness modes/cli."""
    from ..slim_harness import modes
    from ..slim_harness.budget import TIERS
    from ..slim_harness.cli import REASONING_EFFORTS
    from ..slim_harness.construction_budget import BUDGET_HOURS
    from ..cold_start.worker_factory import default_worker_model
    workspace = Path(workspace).resolve()
    check_workspace(workspace, runs_root)
    if set(body) - _FIELDS:
        raise LaunchError("LAUNCH_FIELD_UNKNOWN", "Unknown launch fields: " + ", ".join(sorted(set(body) - _FIELDS)))
    project = _project_id(body.get("project"))
    if (_projects(workspace) / project).exists():
        raise LaunchError("LAUNCH_PROJECT_EXISTS", f"Project {project} already exists; resume it or choose another id", 409)
    mode = body.get("mode", "free")
    if mode not in modes.available_modes(workspace):
        raise LaunchError("LAUNCH_MODE_UNAVAILABLE", f"Mode {mode!r} is not available in this workspace")
    task = _optional(body, "task")
    if task is None or not task.strip():
        raise LaunchError("LAUNCH_TASK_REQUIRED", "Free creation needs a task description")
    providers = {item["id"] for item in _providers()}
    provider = _choice(body, "provider", providers, "codex")
    hours = _optional(body, "budget_hours", int)
    if hours is not None and hours not in BUDGET_HOURS:
        raise LaunchError("LAUNCH_FIELD_INVALID", "budget_hours must be one of " + ", ".join(map(str, BUDGET_HOURS)))
    simulation_seconds = _optional(body, "simulation_timeout_seconds", int)
    if simulation_seconds is not None and simulation_seconds not in SIMULATION_TIMEOUTS:
        raise LaunchError("LAUNCH_FIELD_INVALID",
                          "simulation_timeout_seconds must be one of " + ", ".join(map(str, SIMULATION_TIMEOUTS)))
    chrono = _optional(body, "chrono_python")
    if chrono is not None and not Path(chrono).is_file():
        raise LaunchError("LAUNCH_CHRONO_MISSING", f"Chrono Python not found: {chrono}")
    return {
        "kind": "create", "project": project, "mode": mode, "task": task,
        "provider": provider, "model": _name(body, "model") or default_worker_model(provider),
        "reasoning": _choice(body, "reasoning", REASONING_EFFORTS, "high" if provider == "codex" else "medium"),
        "tier": _choice(body, "tier", TIERS), "budget_hours": hours,
        "simulation_timeout_seconds": simulation_seconds, "chrono_python": chrono,
        "reviewer_provider": _choice(body, "reviewer_provider", providers),
        "reviewer_model": _name(body, "reviewer_model"),
        "reviewer_reasoning": _choice(body, "reviewer_reasoning", REASONING_EFFORTS),
    }


def _records(workspace: Path, project: str | None = None) -> list[tuple[Path, dict]]:
    root = workspace / ".runtime_runs" / "dashboard_launches"
    found = [(path, _read(path)) for path in root.glob("*/launch.json")] if root.is_dir() else []
    return sorted(((path, record) for path, record in found
                   if record.get("project") and (project is None or record["project"] == project)),
                  key=lambda item: str(item[1].get("started_at") or ""))


def _alive(record: dict) -> bool:
    import psutil
    try:
        process = psutil.Process(int(record["pid"]))
        return process.is_running() and abs(process.create_time() - float(record["create_time"])) < 0.1
    except (psutil.Error, KeyError, TypeError, ValueError):
        return False


def _workflow_active(project_dir: Path) -> bool:
    from ..slim_harness.workflow import ActiveWorkflowError, assert_inactive
    try:
        assert_inactive(project_dir)
    except ActiveWorkflowError:
        return True
    return False


def _require_capacity(workspace: Path, limit: int) -> None:
    active = [record["project"] for _path, record in _records(workspace) if _alive(record)]
    if len(active) >= limit:
        raise LaunchError("LAUNCH_LIMIT", f"{len(active)} dashboard run(s) already active ({', '.join(active)}); stop one first", 409)


def _launch_dir(workspace: Path, project: str) -> Path:
    directory = workspace / ".runtime_runs" / "dashboard_launches" / (
        project + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True)
    return directory


def _start(workspace: Path, directory: Path, project: str, kind: str, argv: list[str]) -> tuple[subprocess.Popen, dict]:
    import psutil
    with (directory / "launch.log").open("wb") as log:
        process = subprocess.Popen(argv, cwd=workspace, env=_environment(), stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, **_detached())
    _CHILDREN[process.pid] = process
    threading.Thread(target=process.wait, name=f"launch-reaper-{process.pid}", daemon=True).start()
    try:
        created = psutil.Process(process.pid).create_time()
    except psutil.Error:
        created = None
    record = {"schema_version": "dashboard_launch_v1", "kind": kind, "project": project, "argv": argv,
              "pid": process.pid, "create_time": created, "started_at": _now()}
    write_json_atomic(directory / "launch.json", record)
    return process, record


def _await(process: subprocess.Popen, ready) -> int | None:
    """Return the exit code if the child ended before ``ready()``; None when ready or still starting."""
    deadline = time.monotonic() + START_WAIT_SECONDS
    while not ready():
        if process.poll() is not None:
            return None if ready() else process.returncode
        if time.monotonic() > deadline:
            return None
        time.sleep(0.2)
    return None


def spawn(workspace, request: dict, *, limit: int = 1) -> dict:
    workspace = Path(workspace).resolve()
    project = request["project"]
    with _LOCK:
        _require_capacity(workspace, limit)
        if (_projects(workspace) / project).exists():
            raise LaunchError("LAUNCH_PROJECT_EXISTS", f"Project {project} already exists", 409)
        directory = _launch_dir(workspace, project)
        argv = [_python(), "-X", "utf8", "-m", MODULE, "--workspace", str(workspace),
                "--worker-provider", request["provider"], "--model", request["model"], "--reasoning", request["reasoning"]]
        if request.get("tier"):
            argv += ["--tier", request["tier"]]
        # Supervised construction keeps the loop going after agent mistakes at every tier.
        argv += ["create", "--mode", request["mode"], "--project", project, "--construction"]
        if request.get("task") is not None:
            (directory / "task.md").write_text(request["task"], encoding="utf-8")
            argv += ["--task", str(directory / "task.md")]
        if request.get("budget_hours"):
            argv += ["--budget-hours", str(request["budget_hours"])]
        if request.get("simulation_timeout_seconds"):
            argv += ["--simulation-timeout", str(request["simulation_timeout_seconds"])]
        if request.get("chrono_python"):
            argv += ["--chrono-python", request["chrono_python"]]
        for key in ("reviewer_provider", "reviewer_model", "reviewer_reasoning"):
            if request.get(key):
                argv += ["--" + key.replace("_", "-"), request[key]]
        process, record = _start(workspace, directory, project, "create", argv)
        workflow = _projects(workspace) / project / "workflow.json"
        code = _await(process, workflow.is_file)
    log = directory / "launch.log"
    if code is not None:
        raise LaunchError("LAUNCH_EXITED", f"The harness exited with code {code} before creating the project", 502, _tail(log))
    if not workflow.is_file():
        raise LaunchError("LAUNCH_START_TIMEOUT", f"The harness is still starting after {START_WAIT_SECONDS:.0f} s "
                          "without creating the project; it keeps running, see the log", 504, _tail(log))
    return {"project": project, "run_id": f".runtime_runs/projects/{project}/runs/r01", "pid": record["pid"],
            "log": (directory / "launch.log").relative_to(workspace).as_posix()}


def resume(workspace, project, *, runs_root=None, limit: int = 1) -> dict:
    workspace = Path(workspace).resolve()
    check_workspace(workspace, runs_root)
    project = _project_id(project)
    project_dir = _projects(workspace) / project
    state = _read(project_dir / "workflow.json")
    if state.get("schema_version") != SLIM_SCHEMA:
        raise LaunchError("LAUNCH_NOT_SLIM", f"{project} is not a Slim Harness project", 404)
    if state.get("complete_instance_pass"):
        raise LaunchError("LAUNCH_COMPLETE", f"{project} already passed", 409)
    with _LOCK:
        _require_capacity(workspace, limit)
        if _workflow_active(project_dir) or any(_alive(record) for _path, record in _records(workspace, project)):
            raise LaunchError("LAUNCH_ACTIVE", f"{project} already has an active runner", 409)
        directory = _launch_dir(workspace, project)
        argv = [_python(), "-X", "utf8", "-m", MODULE, "--workspace", str(workspace), "resume", project]
        process, record = _start(workspace, directory, project, "resume", argv)
        started = record["started_at"]
        code = _await(process, lambda: str(_read(project_dir / "workflow_runtime.json").get("heartbeat_at") or "") > started)
    if code:
        raise LaunchError("LAUNCH_EXITED", f"The harness exited with code {code} while resuming", 502, _tail(directory / "launch.log"))
    return status(workspace, project)


def _escalate(record: dict, grace: float) -> None:
    deadline = time.monotonic() + grace
    while _alive(record) and time.monotonic() < deadline:
        time.sleep(1.0)
    if _alive(record):
        from ..runtime.system_probe import terminate_process_tree
        terminate_process_tree(int(record["pid"]), 5.0)


def stop(workspace, project) -> dict:
    """Ask the recorded CLI for an operator stop; its supervisor saves progress and summarizes."""
    workspace = Path(workspace).resolve()
    project = _project_id(project)
    records = _records(workspace, project)
    if not records or not _alive(records[-1][1]):
        raise LaunchError("LAUNCH_NOT_RUNNING", f"No run started from this dashboard is active for {project}", 409)
    path, record = records[-1]
    key = (int(record["pid"]), float(record["create_time"]))
    grace = STOP_ESCALATE_SECONDS
    try:
        os.kill(key[0], signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    except ProcessLookupError:
        return status(workspace, project)
    except OSError:
        grace = 0.0  # the stop cannot be delivered (no console on Windows): end the tree now
    if not record.get("stop_requested_at"):
        record["stop_requested_at"] = _now()
        write_json_atomic(path, record)
    if key not in _STOPPING:
        _STOPPING.add(key)
        threading.Thread(target=_escalate, args=(record, grace), name=f"launch-stop-{key[0]}", daemon=True).start()
    return status(workspace, project)


def status(workspace, project) -> dict:
    workspace = Path(workspace).resolve()
    project = _project_id(project)
    project_dir = _projects(workspace) / project
    records = _records(workspace, project)
    path, record = records[-1] if records else (None, {})
    running = bool(record) and _alive(record)
    state = _read(project_dir / "workflow.json")
    active = running or (bool(state) and _workflow_active(project_dir))
    child = _CHILDREN.get(record.get("pid")) if record else None
    complete = bool(state.get("complete_instance_pass"))
    return {
        "project": project, "exists": bool(state), "launched": bool(record), "kind": record.get("kind"),
        "pid": record.get("pid"), "started_at": record.get("started_at"),
        "running": running, "stopping": running and bool(record.get("stop_requested_at")),
        "exit_code": child.returncode if child is not None and not running else None,
        "active": active, "phase": state.get("phase"), "stop_reason": state.get("stop_reason"), "complete": complete,
        "can_stop": running,
        "can_resume": state.get("schema_version") == SLIM_SCHEMA and not active and not complete,
        "log_tail": _tail(path.parent / "launch.log", 2000) if path else "",
    }


# Serial queue: each queued run starts once no run started from this dashboard is
# active, in queue order. The dashboard's scheduler thread advances it.
_QUEUE_LOCK = threading.Lock()
QUEUE_KEEP_FINISHED = 50
# A start outlives this only when the dashboard stopped while starting the run.
QUEUE_START_STALE_SECONDS = START_WAIT_SECONDS + 60


def _queue_path(workspace: Path) -> Path:
    return workspace / ".runtime_runs" / "dashboard_launches" / "queue.json"


def _queue_items(workspace: Path) -> list[dict]:
    items = _read(_queue_path(workspace)).get("items")
    return items if isinstance(items, list) else []


def _save_queue(workspace: Path, items: list[dict]) -> None:
    waiting = [item for item in items if item.get("status") in ("queued", "starting")]
    done = [item for item in items if item.get("status") not in ("queued", "starting")][-QUEUE_KEEP_FINISHED:]
    path = _queue_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, {"schema_version": "dashboard_launch_queue_v1",
                             "items": sorted(done + waiting, key=lambda item: item.get("queued_at") or "")})


def _queue_view(item: dict, position: int | None = None) -> dict:
    request = item.get("request") or {}
    return {"id": item.get("id"), "status": item.get("status"), "position": position,
            "project": request.get("project"), "mode": request.get("mode"), "provider": request.get("provider"),
            "model": request.get("model"), "reasoning": request.get("reasoning"), "tier": request.get("tier"),
            "task": (request.get("task") or "")[:300], "queued_at": item.get("queued_at"),
            "started_at": item.get("started_at"), "run_id": item.get("run_id"), "error": item.get("error")}


def queue(workspace) -> list[dict]:
    """Waiting and recently failed queue entries, in run order."""
    with _QUEUE_LOCK:
        items = _queue_items(Path(workspace).resolve())
    waiting = [item for item in items if item.get("status") in ("queued", "starting")]
    failed = [item for item in items if item.get("status") == "failed"]
    return [_queue_view(item, index + 1) for index, item in enumerate(waiting)] + [_queue_view(item) for item in failed]


def enqueue(workspace, body: dict, *, runs_root=None) -> dict:
    workspace = Path(workspace).resolve()
    request = validate(workspace, body, runs_root=runs_root)
    with _QUEUE_LOCK:
        items = _queue_items(workspace)
        if any(item.get("status") in ("queued", "starting") and (item.get("request") or {}).get("project") == request["project"]
               for item in items):
            raise LaunchError("LAUNCH_PROJECT_QUEUED", f"Project {request['project']} is already waiting in the queue", 409)
        item = {"id": uuid.uuid4().hex[:12], "status": "queued", "queued_at": _now(), "request": request}
        items.append(item)
        _save_queue(workspace, items)
        position = sum(1 for other in items if other.get("status") in ("queued", "starting"))
    return _queue_view(item, position)


def cancel_queued(workspace, item_id) -> dict:
    workspace = Path(workspace).resolve()
    with _QUEUE_LOCK:
        items = _queue_items(workspace)
        item = next((item for item in items if item.get("id") == item_id), None)
        if item is None or item.get("status") not in ("queued", "failed"):
            raise LaunchError("LAUNCH_QUEUE_ITEM", "This queue entry is no longer waiting", 409)
        items.remove(item)
        _save_queue(workspace, items)
    return {"id": item_id, "removed": True}


def _update_queue_item(workspace: Path, item_id: str, **changes) -> None:
    with _QUEUE_LOCK:
        items = _queue_items(workspace)
        for item in items:
            if item.get("id") == item_id:
                item.update(changes)
        _save_queue(workspace, items)


def _settle_interrupted_starts(workspace: Path, items: list[dict]) -> bool:
    """Resolve entries left "starting" by a dashboard that stopped mid-start; True when any changed."""
    changed = False
    for item in items:
        since = _epoch(item.get("starting_at"))
        if item.get("status") != "starting" or (since and time.time() - since < QUEUE_START_STALE_SECONDS):
            continue
        project = (item.get("request") or {}).get("project")
        launched = [(path, record) for path, record in _records(workspace, project)
                    if (_epoch(record.get("started_at")) or 0) >= (since or 0)]
        if launched and (_projects(workspace) / project / "workflow.json").is_file():
            item.update(status="started", started_at=launched[0][1].get("started_at"),
                        run_id=f".runtime_runs/projects/{project}/runs/r01")
        else:
            log = f"; see {launched[0][0].parent.relative_to(workspace).as_posix()}/launch.log" if launched else ""
            item.update(status="failed", failed_at=_now(),
                        error=f"LAUNCH_INTERRUPTED: the dashboard stopped while starting this run{log}")
        changed = True
    return changed


def advance_queue(workspace, *, limit: int = 1) -> dict | None:
    """Start the next queued run when no dashboard run is active; return the started or failed entry."""
    workspace = Path(workspace).resolve()
    with _QUEUE_LOCK:
        items = _queue_items(workspace)
        if _settle_interrupted_starts(workspace, items):
            _save_queue(workspace, items)
        if any(item.get("status") == "starting" for item in items):
            return None
        waiting = [item for item in items if item.get("status") == "queued"]
        if not waiting or any(_alive(record) for _path, record in _records(workspace)):
            return None
        item = waiting[0]
        item.update(status="starting", starting_at=_now())
        _save_queue(workspace, items)
    try:
        started = spawn(workspace, item["request"], limit=limit)
    except (LaunchError, OSError) as exc:
        _update_queue_item(workspace, item["id"], status="failed", failed_at=_now(),
                           error=f"{getattr(exc, 'code', type(exc).__name__)}: {exc}")
    else:
        _update_queue_item(workspace, item["id"], status="started", started_at=_now(), run_id=started["run_id"])
    return next((entry for entry in _queue_items(workspace) if entry.get("id") == item["id"]), None)


def _budget(project_dir: Path) -> dict | None:
    """Active (working) time against the project budget, as last recorded by the harness."""
    construction = _read(project_dir / "construction_budget.json")
    if construction.get("config"):
        return {"limit_seconds": float(construction["config"].get("hours") or 0) * 3600,
                "active_seconds": float(construction.get("active_seconds") or 0),
                "as_of_ms": int(float(construction.get("updated_at_epoch") or 0) * 1000)}
    checkpoint = _read(project_dir / "budget_checkpoint.json")
    if checkpoint.get("budget_limit_seconds"):
        as_of = _epoch(checkpoint.get("checkpoint_at"))
        return {"limit_seconds": float(checkpoint["budget_limit_seconds"]),
                "active_seconds": float(checkpoint.get("active_seconds") or 0),
                "as_of_ms": int(as_of * 1000) if as_of else None}
    return None


def _latest_result(project_dir: Path) -> dict | None:
    results = sorted(project_dir.glob("runs/r[0-9]*/result.json"))
    if not results:
        return None
    result = _read(results[-1])
    tests = [row for row in result.get("tests") or [] if isinstance(row, dict)]
    return {"revision": results[-1].parent.name, "status": result.get("status"),
            "reported_status": result.get("reported_status"), "tests_total": len(tests),
            "tests_passed": sum(1 for row in tests if row.get("physical_pass")),
            "issue": str((result.get("issues") or [""])[0])[:240]}


def _run_state(workflow: dict, stopping: bool, running: bool, complete: bool) -> str:
    if running:
        return "stopping" if stopping else "running"
    return workflow.get("outcome") or ("passed" if complete else "stopped")


def _epoch(value) -> float | None:
    try:
        return datetime.fromisoformat(str(value)).timestamp()
    except ValueError:
        return None


def latest_motion(project_dir: Path) -> tuple[Path, dict] | None:
    """The newest revision's best playable motion, with its revision folder."""
    from ..slim_harness.motion_preview import best_motion, revision_motions
    for run in sorted((path for path in project_dir.glob("runs/r[0-9]*") if path.is_dir()), reverse=True):
        motion = best_motion(revision_motions(run))
        if motion:
            return run, motion
    return None


def floating_parts(project_dir: Path) -> dict | None:
    """Advisory floating-part findings of the newest checked revision."""
    for report in sorted(project_dir.glob("runs/r[0-9]*/template_runs/a*/floating-parts.json"), reverse=True):
        findings = _read(report).get("findings")
        if isinstance(findings, list):
            return {"revision": report.parents[2].name, "count": len(findings),
                    "messages": [str(item.get("message", ""))[:400] for item in findings if isinstance(item, dict)][:6]}
    return None


def runs_overview(workspace) -> dict:
    """Every Slim project with what a person waiting on it wants to know."""
    from .live_workflow import read_live_project
    from .project_summary import activity_view
    workspace = Path(workspace).resolve()
    by_project: dict[str, list] = {}
    for _path, record in _records(workspace):
        by_project.setdefault(record["project"], []).append(record)
    rows = []
    for state_file in _projects(workspace).glob("*/workflow.json"):
        project_dir = state_file.parent
        state = _read(state_file)
        if state.get("schema_version") != SLIM_SCHEMA:
            continue
        revision = int(state.get("revision") or 1)
        run_id = f".runtime_runs/projects/{project_dir.name}/runs/r{revision:02d}"
        live = read_live_project(workspace_root=workspace, run_dir=workspace / run_id)
        workflow = live.get("live_workflow") or {}
        record = (by_project.get(project_dir.name) or [{}])[-1]
        alive = bool(record) and _alive(record)
        running = alive or bool(workflow.get("is_running"))
        complete = bool(state.get("complete_instance_pass"))
        try:
            activity = activity_view(project_dir, workflow)
        except Exception:  # Activity is a convenience; a damaged trajectory must not hide the run.
            activity = {"available": False}
        task_file = project_dir / "task.md"
        request = state.get("request") or {}
        found = latest_motion(project_dir)
        motion = found and {
            "revision": found[0].name, "run_id": f".runtime_runs/projects/{project_dir.name}/runs/{found[0].name}",
            "id": found[1].get("id"), "label": found[1].get("label") or found[1].get("id"),
            "frames": found[1].get("motion_frame_count"), "status": found[1].get("status"),
            "preview_kind": found[1].get("preview_kind"), "recorded_duration_s": found[1].get("recorded_duration_s")}
        rows.append({
            "project": project_dir.name, "run_id": run_id, "revision": revision,
            "status": _run_state(workflow, alive and bool(record.get("stop_requested_at")), running, complete),
            "phase": workflow.get("phase") or state.get("phase"), "stop_reason": workflow.get("stop_reason") or "",
            "active_test": (workflow.get("test_progress") or {}).get("active_test"),
            "task": task_file.read_text(encoding="utf-8", errors="replace")[:400] if task_file.is_file() else "",
            "provider": request.get("worker_provider"), "model": request.get("model"),
            "reasoning": request.get("reasoning_effort"), "tier": request.get("tier"),
            "started_at": workflow.get("started_at") or state.get("started_at"),
            "ended_at": workflow.get("ended_at") or "", "updated_at": workflow.get("updated_at") or state.get("updated_at"),
            "elapsed_seconds": workflow.get("elapsed_seconds"), "budget": _budget(project_dir),
            "tokens": (live.get("live_usage") or {}).get("total_tokens"), "activity": activity,
            "latest_result": _latest_result(project_dir), "launched": bool(record), "motion": motion,
            "floating_parts": floating_parts(project_dir),
            # The clock a waiting person watches restarts when a stopped run is resumed.
            "attempt_started_at": record.get("started_at") if alive else None,
            "resumed": alive and record.get("kind") == "resume",
            "can_stop": alive, "can_resume": not running and not complete,
        })
    rows.sort(key=lambda row: (row["status"] not in ("running", "stopping"), -(_epoch(row["started_at"]) or 0)))
    return {"runs": rows, "queue": queue(workspace), "server_time_ms": int(time.time() * 1000)}
