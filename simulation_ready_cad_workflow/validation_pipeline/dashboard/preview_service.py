"""Automatically prepare source-geometry previews without involving an author."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import uuid
from typing import Any

from ..mechanism.io import write_json_atomic

CAD_PYTHON = os.environ.get(
    "SIMREADY_CAD_PYTHON",
    "python" if os.name == "nt" else str(Path.home() / ".micromamba-simulation-ready/envs/sr-cad-py312/bin/python"),
)
def resolve_cad_python(recorded=None):
    """Use the recorded environment when present, otherwise this host's CAD configuration."""
    if recorded and Path(recorded).expanduser().is_file():
        return str(Path(recorded).expanduser().absolute())
    configured=os.environ.get("SIMREADY_CAD_PYTHON") or CAD_PYTHON
    # Absolute spelling must preserve a virtualenv's executable symlink.
    return str(Path(configured).expanduser().absolute()) if Path(configured).expanduser().is_file() else configured


PREVIEW_VERSION = "source-studio-4"
_SERVICES: dict[str, "PreviewService"] = {}
_SERVICES_LOCK = threading.Lock()


def _read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


_PREVIEW_PATH_FIELDS = (
    "source_path", "model_path", "scene_path", "replay_path",
    "motion_model_path", "preview_path", "motion_trace",
)


def _preview_workspace(run: Path) -> Path | None:
    from ..snapshot_paths import workspace_for
    root = workspace_for(run)
    if root is not None:
        return root
    run = Path(run).resolve()
    for parent in (run, *run.parents):
        if parent.name == ".runtime_runs":
            return parent.parent
    return None


def _locate_preview_file(workspace: Path | None, recorded, archive=None) -> Path:
    from ..snapshot_paths import resolve
    if workspace is not None and archive:
        return resolve(workspace, recorded)
    name = str(recorded).replace("\\", "/")
    if name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        return Path(recorded)
    if workspace is None:
        return Path(recorded)
    return (workspace / name).resolve()


def _imported_source_stat(archive: dict[str, Any], recorded, relative: str):
    stats = archive.get("source_stats") or {}
    name = str(recorded).replace("\\", "/")
    if name in stats:
        return stats[name]
    mapping = archive.get("path_mapping") or {}
    for key in (name, relative):
        for old, new in mapping.items():
            if new == key and old in stats:
                return stats[old]
    return None


def _slim_preview(run: Path) -> dict[str, Any]:
    value = _read(run / "slim_evidence" / "preview.json")
    if value.get("schema_version") != "slim_studio_preview_v1":
        return {}
    from ..snapshot_paths import imported
    workspace = _preview_workspace(run)
    archive = imported(workspace) if workspace else None
    for original in value.get("input_stamps") or []:
        try:
            path = _locate_preview_file(workspace, original[0], archive)
            stat = path.stat()
            current = [stat.st_size, stat.st_mtime_ns, stat.st_ino]
            if archive:
                relative = path.relative_to(workspace).as_posix()
                if (_imported_source_stat(archive, original[0], relative) != list(original[1:])
                        or (archive.get("receiving_stats") or {}).get(relative) != current):
                    return {}
            # Local identity is path + size + mtime. inode is filesystem-local and
            # changes after rsync/copy even when the published CAD bytes did not.
            elif list(original[1:3]) != current[:2]:
                return {}
        except (OSError, IndexError, TypeError, ValueError):
            return {}
    if workspace:
        for item in [value, *(value.get("motions") or [])]:
            for field in _PREVIEW_PATH_FIELDS:
                raw = item.get(field)
                if raw:
                    try:
                        item[field] = str(_locate_preview_file(workspace, raw, archive))
                    except (OSError, ValueError):
                        pass
    scene=_read(run / "design" / "scene.json")
    gravity=scene.get("gravity_m_s2") or [0,-9.81,0]
    if isinstance(gravity,list) and len(gravity)==3:
        length=sum(float(x)**2 for x in gravity)**.5
        if length>1e-9: value["up_axis"]=[-float(x)/length for x in gravity]
    partial = _read(run / "slim_evidence" / "partial-motion-preview.json")
    if (partial.get("schema_version") == "slim_partial_motion_preview_v1"
            and partial.get("input_version") == value.get("input_version")):
        complete = {item.get("load_case_id") for item in value.get("motions", [])}
        value["motions"] = [*value.get("motions", []), *[
            item for item in partial.get("motions", [])
            if item.get("status") == "UNVERIFIED_PREFIX" and item.get("load_case_id") not in complete]]
    return value


def _stamp(path: Path) -> tuple[Any, ...]:
    try:
        value = path.stat()
        return (str(path), value.st_size, value.st_mtime_ns)
    except OSError:
        return (str(path), 0, 0)


def _source(workspace: Path, run: Path) -> Path | None:
    package_dirs = [run / "design"]
    for plan in sorted((run / "plans").glob("*.json"))[:8]:
        value = _read(plan).get("package_dir")
        if isinstance(value, str) and value:
            candidate = Path(value)
            package_dirs.append(candidate if candidate.is_absolute() else workspace / candidate)
    for directory in package_dirs:
        manifest = _read(directory / "package.json")
        names = [manifest.get("assembly_file"), "assembly.step", "assembly.stp", "assembly.glb", "assembly.obj", "assembly.stl"]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            path = (directory / name).resolve()
            if path.is_relative_to(workspace) and path.is_file() and path.suffix.lower() in {".step", ".stp", ".glb", ".obj", ".stl"}:
                return path
    return None


def _inputs(workspace: Path, run: Path) -> tuple[Path | None, list[tuple[Any, ...]], list[Path]]:
    source = _source(workspace, run)
    if source is None:
        return None, [], []
    traces = sorted((run / "traces").glob("*/*/physical.trace.json"))[:24]
    paths = [source, source.parent / "assembly_contract.json", *sorted((run / "plans").glob("*.json")), *traces]
    return source, [_stamp(path) for path in paths], traces


def _cache_root(workspace: Path, run: Path) -> Path:
    relative = run.relative_to(workspace)
    if run.parent.name == "runs":
        return workspace / ".runtime_runs" / "dashboard_previews" / run.parent.parent.name / run.name
    return workspace / ".runtime_runs" / "dashboard_previews" / "legacy" / "__".join(relative.parts)


def _read_timeline(cache: Path) -> dict[str, Any]:
    value = _read(cache / "timeline.json")
    return value if isinstance(value.get("frames"), list) else {"frames": []}


class PreviewService:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self.lock = threading.Lock()
        self.jobs: queue.Queue[Any] = queue.PriorityQueue()
        self.records: dict[str, dict[str, Any]] = {}
        self.image_inputs = {}
        self.process: subprocess.Popen | None = None
        self.closed = threading.Event()
        self.thread = threading.Thread(target=self._worker, name="source-preview", daemon=True)
        self.thread.start()

    def ensure(self, run: Path) -> dict[str, Any]:
        run = run.resolve()
        run.relative_to(self.workspace)
        slim = _slim_preview(run)
        if slim:
            return slim
        source, inputs, traces = _inputs(self.workspace, run)
        if source is None:
            return {"status": "missing", "source_kind": "original_geometry", "source_revision": run.name,
                    "error": "当前版本尚未保存可预览的模型"}
        identity = str(run)
        fingerprint = json.dumps([PREVIEW_VERSION, inputs], separators=(",", ":"))
        now = time.monotonic()
        with self.lock:
            record = self.records.get(identity)
            if record is None:
                cached = _read(_cache_root(self.workspace, run) / "preview.json")
                if cached.get("inputs") == [list(item) for item in inputs] and cached.get("generator") == PREVIEW_VERSION:
                    if cached.get("model_path") and Path(cached["model_path"]).is_file():
                        record = {"fingerprint": fingerprint, "first_seen": now, "manifest": cached, "queued": False}
                        self.records[identity] = record
            if record is None or record["fingerprint"] != fingerprint:
                prior = dict(record["manifest"]) if record else {}
                prior.update({"status": "queued", "source_kind": "original_geometry", "source_revision": run.name,
                              "pending_update": bool(prior.get("model_path")), "error": None})
                record = {"fingerprint": fingerprint, "first_seen": now, "manifest": prior, "queued": False}
                self.records[identity] = record
            if not record["queued"] and record["manifest"]["status"] == "queued" and now - record["first_seen"] >= 0.75:
                record["queued"] = True
                self.jobs.put((10, time.monotonic(), (run, source, inputs, traces, fingerprint)))
            return dict(record["manifest"])

    def _record_timeline_frame(self, cache: Path, result: dict[str, Any], inputs: list[tuple[Any, ...]]) -> None:
        version = result.get("input_version")
        model = result.get("model_path")
        if not isinstance(version, str) or not version or not isinstance(model, str) or not model:
            return
        try:
            relative = Path(model).resolve().relative_to(cache.resolve())
        except (OSError, ValueError):
            return
        with self.lock:
            timeline = _read_timeline(cache)
            frames = timeline["frames"]
            if any(frame.get("id") == version for frame in frames):
                return
            try:
                mtime_ns = int(inputs[0][2])
            except (IndexError, TypeError, ValueError):
                mtime_ns = 0
            created = (datetime.fromtimestamp(mtime_ns / 1e9, timezone.utc).isoformat()
                       if mtime_ns else result.get("updated_at"))
            try:
                size = Path(model).stat().st_size
            except OSError:
                size = 0
            frames.append({"id": version, "created_at": created, "model": relative.as_posix(), "bytes": size})
            timeline["frames"] = frames
            write_json_atomic(cache / "timeline.json", timeline)

    def _worker(self) -> None:
        while not self.closed.is_set():
            _priority, _queued_at, job = self.jobs.get()
            if isinstance(job, dict) and job.get("history"):
                from .history_preview import build
                build(self, job)
                continue
            if job is None:
                return
            run, source, inputs, traces, fingerprint = job
            identity = str(run)
            with self.lock:
                record = self.records.get(identity)
                if record is None or record["fingerprint"] != fingerprint:
                    continue
                record["manifest"]["status"] = "building"
            cache = _cache_root(self.workspace, run)
            version = f"{inputs[0][2]:x}-{inputs[0][1]:x}"
            output = cache / version / ("build_" + uuid.uuid4().hex[:12])
            output.mkdir(parents=True, exist_ok=True)
            request = {"workspace": str(self.workspace), "run": str(run), "source": str(source),
                       "output": str(output), "model_cache": str(cache / version / "model.glb"),
                       "traces": [str(path) for path in traces], "input_version": version}
            request_path = output / "request.json"
            write_json_atomic(request_path, request)
            try:
                environment = dict(os.environ)
                environment["SIMREADY_STEP_SHAPE_CACHE_DIR"] = str(self.workspace / ".runtime_runs" / "dashboard_previews" / "step_cache")
                log_path = output / "build.log"
                with log_path.open("w", encoding="utf-8") as log:
                    with self.lock:
                        if self.closed.is_set():
                            return
                        self.process = subprocess.Popen(
                            [CAD_PYTHON, "-X", "utf8", "-m", "simulation_ready_cad_workflow.validation_pipeline.dashboard.preview_worker", str(request_path)],
                            cwd=str(Path(__file__).resolve().parents[3]), env=environment, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                    try:
                        code = self.process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
                        raise TimeoutError("模型预览生成超时，原始产物已保留")
                result = _read(output / "result.json")
                if code != 0 or not result.get("model_path"):
                    raise ValueError(result.get("error") or f"预览生成失败，详情见 {log_path.name}")
                result.update({"status": "ready", "source_kind": "original_geometry", "source_revision": run.name,
                               "input_version": version, "inputs": [list(item) for item in inputs],
                               "generator": PREVIEW_VERSION, "pending_update": False,
                               "updated_at": datetime.now(timezone.utc).isoformat()})
                # Discard a render if the author changed its source while it was loading.
                _current_source, current_inputs, _current_traces = _inputs(self.workspace, run)
                if current_inputs != inputs:
                    continue
                write_json_atomic(cache / "preview.json", result)
                self._record_timeline_frame(cache, result, inputs)
                with self.lock:
                    if self.records.get(identity, {}).get("fingerprint") == fingerprint:
                        self.records[identity]["manifest"] = result
            except Exception as exc:
                with self.lock:
                    if self.records.get(identity, {}).get("fingerprint") == fingerprint:
                        self.records[identity]["manifest"].update({"status": "failed", "error": str(exc),
                            "updated_at": datetime.now(timezone.utc).isoformat()})
            finally:
                with self.lock:
                    self.process = None

    def close(self) -> None:
        with self.lock:
            self.closed.set()
            process = self.process
        self.jobs.put((-1, time.monotonic(), None))
        if process is not None and process.poll() is None:
            process.terminate()
        self.thread.join(timeout=2)


def _service(workspace: Path) -> PreviewService:
    key = str(workspace.resolve())
    with _SERVICES_LOCK:
        if key not in _SERVICES:
            _SERVICES[key] = PreviewService(workspace)
        return _SERVICES[key]


def ensure_preview(*, workspace_root: Path, run_dir: Path) -> dict[str, Any]:
    return _service(Path(workspace_root)).ensure(Path(run_dir))


def preview_media_paths(*, workspace_root: Path, run_dir: Path) -> dict[str, Path]:
    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    cache = _cache_root(workspace, run).resolve()
    slim = _slim_preview(run)
    if slim:
        result = {}
        for item in [slim, *(slim.get("motions") or [])]:
            for field in ("source_path", "model_path", "scene_path", "replay_path", "motion_model_path", "preview_path"):
                raw = item.get(field)
                if isinstance(raw, str) and raw:
                    path = Path(raw).resolve()
                    if path.is_relative_to(run) and path.is_file():
                        result["slim/" + path.relative_to(run).as_posix()] = path
        for frame in _read_timeline(cache)["frames"]:
            path = (cache / frame["model"]).resolve()
            if path.is_relative_to(cache) and path.is_file():
                result["timeline/" + frame["model"]] = path
        return result
    value = _read(cache / "preview.json")
    result = {}
    source = _source(workspace, run)
    if source is not None and value.get("source_path") == str(source):
        result[f"source/{source.name}"] = source
    for item in [value, *(value.get("motions") or [])]:
        for field in ("model_path", "scene_path", "replay_path", "motion_model_path", "preview_path"):
            raw = item.get(field)
            if isinstance(raw, str) and raw:
                path = Path(raw).resolve()
                if path.is_relative_to(cache) and path.is_file():
                    result[path.relative_to(cache).as_posix()] = path
    for frame in _read_timeline(cache)["frames"]:
        model = frame.get("model")
        if not isinstance(model, str) or not model:
            continue
        path = (cache / model).resolve()
        if path.is_relative_to(cache) and path.is_file():
            result["timeline/" + path.relative_to(cache).as_posix()] = path
    return result


def preview_timeline(*, workspace_root: Path, run_dir: Path) -> list[dict[str, Any]]:
    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    cache = _cache_root(workspace, run).resolve()
    frames = []
    for frame in _read_timeline(cache)["frames"]:
        model = frame.get("model")
        if not isinstance(model, str) or not model:
            continue
        path = (cache / model).resolve()
        if path.is_relative_to(cache) and path.is_file():
            frames.append(dict(frame))
    return frames


def shutdown_previews(workspace_root: Path) -> None:
    with _SERVICES_LOCK:
        service = _SERVICES.pop(str(Path(workspace_root).resolve()), None)
    if service is not None:
        service.close()
