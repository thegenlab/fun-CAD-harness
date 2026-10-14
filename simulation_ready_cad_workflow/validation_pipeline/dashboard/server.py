from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import mimetypes
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlparse

from ..run_layout import parse_canonical_run
from .legacy_io import (
    DashboardReadError,
    react_physics_evidence_media_allowlist,
    react_reference_pose_media_allowlist,
    react_static_visual_media_allowlist,
    react_v2_media_allowlist,
    read_stable_json_object,
)
from .model import (
    _react_loop_state_path,
    compare_revision_geometry,
    group_runs_by_project,
    is_react_loop_run,
    normalize_run,
    scan_runs,
)
from .path_resolution import resolve_workspace_reference
from .render import STATIC_ROOT, render_live_shell, render_snapshot
from .url_paths import prefix_url_fields


_CHANGE_TOKEN_SCHEMA = "simulation_ready_cad_dashboard_change_token_v1"
_CHANGE_RESPONSE_SCHEMA = "simulation_ready_cad_dashboard_changes_v1"
_CHANGE_TOKEN_PREFIX = "v1"
_CHANGE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_CHANGE_DEFAULT_TIMEOUT_S = 25.0
_CHANGE_MAX_TIMEOUT_S = 30.0
_CHANGE_PROBE_INTERVAL_S = 0.20
_CATALOG_CHANGE_CACHE_TTL_S = 1.0
_RUN_CHANGE_MAX_RECORDS = 512
_RUN_CHANGE_MAX_SCANNED_ENTRIES = 2048
_RUN_AUTHORED_TEST_MAX_RECORDS = 128
_CATALOG_CHANGE_MAX_SCANNED_ENTRIES = 2048
_CATALOG_CHANGE_MAX_RUNTIME_ROOTS = 64
_CATALOG_CHANGE_MAX_PROJECTS = 256
_CATALOG_CHANGE_MAX_REVISIONS = 1024
_CATALOG_CHANGE_MAX_LEGACY_RUNS = 256
_CATALOG_CHANGE_MAX_SCOPED_REVISIONS = 512
_CATALOG_CHANGE_MAX_SCOPES = 128
_CATALOG_CHANGE_MAX_ISSUED_TOKENS = 256
MIN_RUNNING_LEASE_TTL_S = 30
MAX_RUNNING_LEASE_TTL_S = 24 * 60 * 60
OperatorAuthorityError = DashboardReadError
_PIPELINE = __package__.rsplit(".", 1)[0]


def _present(*modules: str) -> bool:
    try:
        return all(find_spec(name) is not None for name in modules)
    except ModuleNotFoundError:
        return False


# The legacy ReAct maintenance surfaces and the team archive are private
# packages; without them their routes answer 404 and the page hides them.
maintenance_routes = import_module(f"{__package__}.maintenance_routes") if _present(
    *(f"{_PIPELINE}.{name}" for name in ("react_loop", "harness_change_proposals", "project_resource_review")),
    f"{__package__}.maintenance_routes",
) else None
TEAM_LIBRARY = _present(f"{_PIPELINE}.hf_sync", f"{_PIPELINE}.hf_sync.client")
_LAUNCH_GET_ROUTES = frozenset({"/launch", "/api/launch/options", "/api/launch/status"})
_LAUNCH_POST_ROUTES = frozenset({"/api/launch", "/api/launch/stop", "/api/launch/resume",
                                 "/api/launch/queue", "/api/launch/queue/cancel"})
QUEUE_POLL_SECONDS = 5.0
_RUNS_OVERVIEW_TTL_SECONDS = 1.5


class CatalogChangeConflict(ValueError):
    """A requested catalog refresh token is no longer current."""


def _digest_record(digest: object, *values: object) -> None:
    """Add an unambiguous metadata record to a SHA-256 digest."""

    encoded = json.dumps(
        values,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _stat_record(path: Path) -> tuple[object, ...]:
    """Return metadata only; change watches never read artifact contents."""

    try:
        value = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return ("missing",)
    except OSError as exc:
        return ("unavailable", type(exc).__name__)
    return (
        "present",
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
        value.st_dev,
        value.st_ino,
    )


def _regular_file_present(path: Path) -> bool:
    value = _stat_record(path)
    return bool(
        len(value) > 2
        and value[0] == "present"
        and value[1] == stat.S_IFREG
    )


def _run_surface_metadata_digest(
    root: Path,
    *,
    status_time_projection: tuple[str, ...] = (),
) -> str:
    """Fingerprint the bounded surface consumed by the live dashboard.

    Traces, images, videos, and arbitrary job workspaces can contain many
    thousands of immutable files.  Their visible progress is committed via a
    queue state/checkpoint, result summary, or dashboard publication, so the
    change watch never walks those bulk trees.
    """

    digest = hashlib.sha256()
    _digest_record(digest, _CHANGE_TOKEN_SCHEMA, "run-surface")
    _digest_record(digest, "status-time-projection", *status_time_projection)
    recorded: set[str] = set()
    overflowed: set[str] = set()
    record_count = 0
    scanned_entries = 0

    def relative(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return "outside-run"

    def overflow(directory: Path, reason: str) -> None:
        label = relative(directory)
        if label in overflowed:
            return
        overflowed.add(label)
        _digest_record(
            digest,
            "overflow",
            label,
            reason,
            *_stat_record(directory),
        )

    def add(path: Path) -> bool:
        nonlocal record_count
        label = relative(path)
        if label in recorded:
            return True
        if record_count >= _RUN_CHANGE_MAX_RECORDS:
            overflow(path.parent, "record-limit")
            return False
        recorded.add(label)
        record_count += 1
        _digest_record(digest, label, *_stat_record(path))
        return True

    def entries(directory: Path) -> list[os.DirEntry[str]] | None:
        nonlocal scanned_entries
        remaining = _RUN_CHANGE_MAX_SCANNED_ENTRIES - scanned_entries
        if remaining <= 0:
            overflow(directory, "scan-limit")
            return []
        try:
            with os.scandir(directory) as iterator:
                values: list[os.DirEntry[str]] = []
                exceeded = False
                for item in iterator:
                    if len(values) >= remaining:
                        exceeded = True
                        break
                    values.append(item)
                values.sort(key=lambda item: item.name)
        except OSError as exc:
            _digest_record(
                digest,
                "scan-unavailable",
                relative(directory),
                type(exc).__name__,
            )
            return None
        scanned_entries += len(values)
        if exceeded:
            overflow(directory, "scan-limit")
        return values

    # Exact state/publication files are recorded even while absent so their
    # first atomic publication changes the token.
    for path in (
        root / "run_status.json",
        root / "planning" / "task_planning.json",
        root / "artifacts.sha256.json",
        root / "dashboard" / "publication.json",
        root / "dashboard" / "workflow_stage_manifest.json",
        root / "visual" / "dashboard_visuals.summary.json",
    ):
        add(path)
    # The ReAct loop is project-scoped so that it can move from rNN to
    # rNN without exposing its history to a design stage.  It is still one
    # exact bounded file for the selected run, never a project-tree scan.
    if root.parent.name == "runs":
        from .live_workflow import live_project_metadata_paths

        for live_path in live_project_metadata_paths(workspace_root=root.parents[4], run_dir=root):
            _digest_record(digest, "live-project", str(live_path), *_stat_record(live_path))
        react_state = _react_loop_state_path(
            root.parents[4], root.parent.parent
        ) or root.parent.parent / "react_loop" / "state.json"
        _digest_record(
            digest,
            "selected-react-loop-state",
            *_stat_record(react_state),
        )
        _digest_record(
            digest,
            "selected-react-loop-continuation",
            *_stat_record(react_state.parent / "driver" / "continuation.json"),
        )

    root_entries = entries(root) or []
    for entry in root_entries:
        if not entry.is_file(follow_symlinks=False):
            continue
        name = entry.name
        if (
            (name.startswith("intake") and name.endswith(".json"))
            or name == "queue_manifest.json"
            or (
                name.endswith("queue_manifest.json")
                and ("physical" in name or "ideal" in name)
            )
        ):
            add(Path(entry.path))

    for entry in entries(root / "plans") or []:
        if entry.is_file(follow_symlinks=False) and entry.name.endswith(".json"):
            add(Path(entry.path))

    # Authored test tasks are executable planning inputs and therefore part
    # of the live surface, independent of a later planning-contract rewrite.
    # Keep this arbitrary-file tree tightly bounded and never follow links.
    authored_root = root / "planning" / "authored_tests"
    _digest_record(
        digest,
        "authored-tests-root",
        *_stat_record(authored_root),
    )
    authored_pending = [(authored_root, 0)]
    authored_records = 0
    authored_limit_reached = False
    while (
        authored_pending
        and scanned_entries < _RUN_CHANGE_MAX_SCANNED_ENTRIES
        and not authored_limit_reached
    ):
        directory, depth = authored_pending.pop()
        child_directories: list[tuple[Path, int]] = []
        for entry in entries(directory) or []:
            entry_path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if depth < 8:
                    child_directories.append((entry_path, depth + 1))
                else:
                    overflow(entry_path, "depth-limit")
            elif entry.is_file(follow_symlinks=False):
                if authored_records >= _RUN_AUTHORED_TEST_MAX_RECORDS:
                    overflow(authored_root, "authored-test-limit")
                    authored_limit_reached = True
                    break
                if add(entry_path):
                    authored_records += 1
        authored_pending.extend(reversed(child_directories))
    if authored_pending and not authored_limit_reached:
        overflow(authored_root, "scan-limit")
    _digest_record(
        digest,
        "authored-tests-bound",
        authored_records,
        authored_limit_reached,
    )

    # Queue trees are traversed only to locate queue_state/checkpoint files
    # and completed job result markers.  Both traversal and recorded matches
    # have hard ceilings.
    queue_root = root / "runtime_queue"
    pending = [(queue_root, 0)]
    while pending and scanned_entries < _RUN_CHANGE_MAX_SCANNED_ENTRIES:
        directory, depth = pending.pop()
        child_directories: list[tuple[Path, int]] = []
        for entry in entries(directory) or []:
            entry_path = Path(entry.path)
            if entry.is_dir(follow_symlinks=False):
                if depth < 8:
                    child_directories.append((entry_path, depth + 1))
                else:
                    overflow(entry_path, "depth-limit")
            elif entry.is_file(follow_symlinks=False) and (
                entry.name == "queue_state.json"
                or entry.name.endswith(".checkpoint.json")
                or entry.name.endswith(".result")
                or entry.name.endswith(".result.json")
            ):
                add(entry_path)
        pending.extend(reversed(child_directories))
    if pending:
        overflow(queue_root, "scan-limit")

    results_root = root / "results"
    for mechanism in entries(results_root) or []:
        if not mechanism.is_dir(follow_symlinks=False):
            continue
        mechanism_root = Path(mechanism.path)
        add(mechanism_root / "certificate.json")
        add(mechanism_root / "replacement.json")
        add(mechanism_root / "task_acceptance.json")
        comparison_root = mechanism_root / "comparison"
        for comparison in entries(comparison_root) or []:
            if (
                comparison.is_file(follow_symlinks=False)
                and comparison.name.endswith(".json")
            ):
                add(Path(comparison.path))

    # Stage-2 source/evaluation documents are small control trees. Include
    # JSON and authoritative STEP metadata, while deliberately excluding
    # images, videos, traces, and other bulk presentation artifacts.
    for tree_root, suffixes in (
        (root / "inputs", {".json", ".step"}),
        (root / "design_quality", {".json"}),
    ):
        pending = [(tree_root, 0)]
        while pending and scanned_entries < _RUN_CHANGE_MAX_SCANNED_ENTRIES:
            directory, depth = pending.pop()
            child_directories: list[tuple[Path, int]] = []
            for entry in entries(directory) or []:
                entry_path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    if depth < 8:
                        child_directories.append((entry_path, depth + 1))
                    else:
                        overflow(entry_path, "depth-limit")
                elif (
                    entry.is_file(follow_symlinks=False)
                    and entry_path.suffix.lower() in suffixes
                ):
                    add(entry_path)
            pending.extend(reversed(child_directories))
        if pending:
            overflow(tree_root, "scan-limit")

    _digest_record(
        digest,
        "bounds",
        record_count,
        scanned_entries,
        len(overflowed),
    )
    return digest.hexdigest()


def _token_parts(token: str) -> tuple[str, str] | None:
    parts = token.split(".")
    if (
        len(parts) != 3
        or parts[0] != _CHANGE_TOKEN_PREFIX
        or not _CHANGE_DIGEST_RE.fullmatch(parts[1])
        or not _CHANGE_DIGEST_RE.fullmatch(parts[2])
    ):
        return None
    return parts[1], parts[2]


class DashboardServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        *,
        workspace_root: Path,
        runs_root: Path,
        base_path: str = "",
        bind_and_activate: bool = True,
        launch_enabled: bool = False,
        launch_limit: int = 4,
    ) -> None:
        super().__init__(server_address, DashboardHandler, bind_and_activate=bind_and_activate)
        self.base_path = "/" + base_path.strip("/") if base_path.strip("/") else ""
        self.workspace_root = workspace_root.resolve()
        self.runs_root = runs_root.resolve()
        self._runs_cache_lock = threading.Lock()
        self._runs_cache: list[dict[str, object]] | None = None
        self._runs_cache_completed_at = 0.0
        self._runs_cache_ttl_s = 10.0
        self._runs_cache_refreshing = False
        self._runs_fresh_lock = threading.Lock()
        self._runs_cache_catalog_token: str | None = None
        self._change_probe_interval_s = _CHANGE_PROBE_INTERVAL_S
        self._change_watch_condition = threading.Condition()
        self._change_watch_stop = threading.Event()
        self._change_watch_thread: threading.Thread | None = None
        self._change_watch_refcounts: dict[str, int] = {}
        self._change_watch_snapshots: dict[
            str, tuple[float, dict[str, str]]
        ] = {}
        self._catalog_change_cache_lock = threading.Lock()
        self._catalog_change_compute_lock = threading.Lock()
        self._catalog_change_cache_ttl_s = _CATALOG_CHANGE_CACHE_TTL_S
        self._catalog_change_cache: dict[
            str, tuple[str, float, float | None]
        ] = {}
        self._catalog_change_issued_scopes: dict[
            str, tuple[str, float]
        ] = {}
        self._catalog_status_projection_cache: dict[
            str,
            tuple[tuple[object, ...], tuple[str, ...], float | None],
        ] = {}
        self._revision_mesh_cache_lock = threading.Lock()
        self._revision_mesh_cache: dict[tuple[str, str], bytes] = {}
        self.resource_loop_supervisor = maintenance_routes and maintenance_routes.ResourceLoopDriverSupervisor()
        self.human_decision_capability = secrets.token_urlsafe(32)
        from ..snapshot_paths import imported
        from ..trajectory.capture import HistoryIndexer
        self.history_indexer = None if imported(self.workspace_root) else HistoryIndexer(self.workspace_root)
        # Imported snapshots are historical: they never start or stop runs.
        self.launch_enabled = launch_enabled and not imported(self.workspace_root)
        self.launch_limit = launch_limit
        self._runs_overview_lock = threading.Lock()
        self._runs_overview_cache = None
        # Queued runs advance only while a launch-enabled dashboard is serving.
        self._queue_stop = threading.Event()
        self._queue_thread = None
        if self.launch_enabled:
            self._queue_thread = threading.Thread(target=self._advance_queue_loop, name="launch-queue", daemon=True)
            self._queue_thread.start()
        self._image_service = None
        self._image_service_lock = threading.Lock()
        self.import_parent = None
        self.import_contexts = {}
        self.import_context_lock = threading.Lock()

    def runs_overview(self):
        """Monitor data shared by all open monitor pages for a moment."""
        from . import launcher

        with self._runs_overview_lock:
            cached = self._runs_overview_cache
            if cached is None or time.monotonic() - cached[0] > _RUNS_OVERVIEW_TTL_SECONDS:
                value = launcher.runs_overview(self.workspace_root)
                for row in value["runs"]:
                    motion = row.get("motion")
                    if not motion:
                        continue
                    try:
                        _run, recorded, target, up = self.motion_for(motion["run_id"], motion["id"])
                        status = self.motion_thumbnails().status(target, recorded, up)
                    except (OSError, ValueError, KeyError):
                        status = "failed"
                    motion["thumbnail"] = {"status": status, "url": (
                        f"/api/motion-thumbnail?run={quote(motion['run_id'], safe='')}&motion={quote(str(motion['id']), safe='')}"
                        f"&v={quote(target.stem if status == 'ready' else '', safe='')}") if status == "ready" else None}
                value.update(launch=self.launch_enabled, limit=self.launch_limit,
                             queue_poll_seconds=QUEUE_POLL_SECONDS)
                cached = self._runs_overview_cache = (time.monotonic(), value)
        return cached[1]

    def forget_runs_overview(self):
        """Let the monitor's refresh right after an action see its effect."""
        with self._runs_overview_lock:
            self._runs_overview_cache = None

    def team_library(self):
        parent = self.import_parent or self
        if not TEAM_LIBRARY:
            return {'local_url': parent.base_path + '/', 'items': []}
        from ..hf_sync.client import imports
        return {'local_url': parent.base_path + '/', 'items': [
            {**row, 'url': parent.base_path + '/imports/' + row['id'] + '/'}
            for row in imports(parent.workspace_root)]}

    def import_context(self, identity):
        if not TEAM_LIBRARY:
            raise KeyError(identity)
        from ..hf_sync.client import imports
        with self.import_context_lock:
            if identity in self.import_contexts:
                return self.import_contexts[identity]
            record = next((r for r in imports(self.workspace_root) if r['id'] == identity), None)
            if record is None:
                raise KeyError(identity)
            workspace = Path(record['workspace'])
            context = DashboardServer(('127.0.0.1', 0), workspace_root=workspace,
                runs_root=workspace / '.runtime_runs', base_path=self.base_path + '/imports/' + identity,
                bind_and_activate=False)
            context.import_parent = self
            self.import_contexts[identity] = context
            return context

    def _ensure_change_watcher_locked(self) -> None:
        thread = self._change_watch_thread
        if thread is not None and thread.is_alive():
            return
        if self._change_watch_stop.is_set():
            return
        thread = threading.Thread(
            target=self._dashboard_change_watch_loop,
            name="dashboard-change-watch",
            daemon=True,
        )
        self._change_watch_thread = thread
        thread.start()

    def _dashboard_change_watch_loop(self) -> None:
        """Probe each watched run once per interval, regardless of clients."""

        while not self._change_watch_stop.wait(
            self._change_probe_interval_s
        ):
            with self._change_watch_condition:
                run_ids = sorted(
                    run_id
                    for run_id, count in self._change_watch_refcounts.items()
                    if count > 0
                )
            updates: list[tuple[str, float, dict[str, str]]] = []
            for run_id in run_ids:
                if self._change_watch_stop.is_set():
                    break
                try:
                    snapshot = self.dashboard_change_snapshot(run_id)
                except (KeyError, OSError):
                    continue
                updates.append((run_id, time.monotonic(), snapshot))
            if not updates:
                continue
            with self._change_watch_condition:
                changed = False
                for run_id, observed_at, snapshot in updates:
                    previous = self._change_watch_snapshots.get(run_id)
                    if previous is not None and previous[0] > observed_at:
                        continue
                    self._change_watch_snapshots[run_id] = (
                        observed_at,
                        snapshot,
                    )
                    if (
                        previous is None
                        or previous[1]["token"] != snapshot["token"]
                    ):
                        changed = True
                if changed:
                    self._change_watch_condition.notify_all()

    def _stop_dashboard_change_watcher(self) -> None:
        event = getattr(self, "_change_watch_stop", None)
        if event is None:
            return
        event.set()
        condition = getattr(self, "_change_watch_condition", None)
        if condition is not None:
            with condition:
                condition.notify_all()
        thread = getattr(self, "_change_watch_thread", None)
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)

    def _advance_queue_loop(self) -> None:
        from . import launcher

        while not self._queue_stop.wait(QUEUE_POLL_SECONDS):
            try:
                launcher.advance_queue(self.workspace_root, limit=self.launch_limit)
            except Exception:  # A broken queue entry must not stop the dashboard; it stays visible.
                logging.getLogger(__name__).exception("Advancing the launch queue failed")

    def server_close(self) -> None:
        from .preview_service import shutdown_previews

        if getattr(self, "_queue_stop", None):
            self._queue_stop.set()
        for context in getattr(self, 'import_contexts', {}).values():
            context.server_close()
        if getattr(self,"_image_service",None): self._image_service.close()
        self._stop_dashboard_change_watcher()
        if getattr(self, "history_indexer", None): self.history_indexer.close()
        shutdown_previews(self.workspace_root)
        super().server_close()

    def _scan_runs_catalog(self) -> list[dict[str, object]]:
        # New ReAct discovery is an ordinary read-only catalog operation.
        # Legacy terminal rows retain their own compatibility verification in
        # model.scan_runs(), but the whole live scan is no longer wrapped in
        # an authority transaction.
        rows = scan_runs(
            self.runs_root,
            workspace_root=self.workspace_root,
            include_history=True,
            include_active=True,
        )
        # A preflight's private probe runs are evidence of their parent revision,
        # not hundreds of separate mechanical projects in the project picker.
        return [row for row in rows if "/.presim_gate/" not in str(row.get("id") or "")]

    def _refresh_runs_cache(self) -> None:
        try:
            rows = self._scan_runs_catalog()
        except Exception:
            # A failed refresh must not blank a catalog that was already
            # usable. The next request may retry after the ordinary TTL.
            with self._runs_cache_lock:
                self._runs_cache_refreshing = False
            return
        with self._runs_cache_lock:
            self._runs_cache = rows
            self._runs_cache_completed_at = time.monotonic()
            self._runs_cache_refreshing = False

    def runs(self) -> list[dict[str, object]]:
        # The live selector exposes immutable history plus explicit PREPARING
        # and RUNNING canonical revisions. Publishing/grouping APIs keep the
        # stricter terminal-only default used by scan_runs().
        now = time.monotonic()
        if (
            self._runs_cache is not None
            and now - self._runs_cache_completed_at < self._runs_cache_ttl_s
        ):
            return copy.deepcopy(self._runs_cache)
        # A catalog scan can cover years of immutable history. Never perform
        # that scan while a request owns the cache lock, including on first
        # load. An empty list means the initial background scan is still
        # warming; /api/runs exposes that state explicitly.
        start_refresh = False
        with self._runs_cache_lock:
            now = time.monotonic()
            if (
                self._runs_cache is not None
                and now - self._runs_cache_completed_at < self._runs_cache_ttl_s
            ):
                return copy.deepcopy(self._runs_cache)
            rows = copy.deepcopy(self._runs_cache or [])
            if not getattr(self, "_runs_cache_refreshing", False):
                self._runs_cache_refreshing = True
                start_refresh = True
        if start_refresh:
            try:
                threading.Thread(
                    target=self._refresh_runs_cache,
                    name="dashboard-catalog-refresh",
                    daemon=True,
                ).start()
            except Exception:
                with self._runs_cache_lock:
                    self._runs_cache_refreshing = False
                raise
        return rows

    def catalog_state(self) -> dict[str, bool]:
        """Return the non-authoritative cache lifecycle for the live UI."""

        with self._runs_cache_lock:
            return {
                "ready": self._runs_cache is not None,
                "refreshing": bool(
                    getattr(self, "_runs_cache_refreshing", False)
                ),
            }

    def refresh_runs(self) -> list[dict[str, object]]:
        """Refresh the catalog only for an explicit user request."""

        with self._runs_fresh_lock:
            rows = self._scan_runs_catalog()
            with self._runs_cache_lock:
                self._runs_cache = rows
                self._runs_cache_completed_at = time.monotonic()
                self._runs_cache_refreshing = False
                self._runs_cache_catalog_token = None
            return copy.deepcopy(rows)

    def _change_runtime_roots(self) -> tuple[list[Path], bool]:
        """Return a bounded set of real runtime roots without following links."""

        if self.runs_root != self.workspace_root:
            return [self.runs_root], False
        roots = [self.workspace_root / ".runtime_runs"]
        exceeded = False
        try:
            with os.scandir(self.workspace_root) as iterator:
                observed = 0
                for entry in iterator:
                    if observed >= _CATALOG_CHANGE_MAX_RUNTIME_ROOTS:
                        exceeded = True
                        break
                    observed += 1
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    candidate = Path(entry.path) / ".runtime_runs"
                    metadata = _stat_record(candidate)
                    if (
                        len(metadata) > 2
                        and metadata[0] == "present"
                        and metadata[1] == stat.S_IFDIR
                    ):
                        roots.append(candidate)
        except OSError:
            pass
        return sorted(set(roots)), exceeded

    @staticmethod
    def _catalog_cache_entry_current(
        entry: tuple[str, float, float | None] | None,
        *,
        now: float,
        wall_time: float,
        ttl_s: float,
    ) -> bool:
        return bool(
            entry is not None
            and now - entry[1] < ttl_s
            and (entry[2] is None or wall_time <= entry[2])
        )

    def _record_catalog_token_locked(
        self,
        *,
        scope: str,
        token: str,
        completed_at: float,
        deadline_epoch: float | None,
    ) -> None:
        self._catalog_change_cache[scope] = (
            token,
            completed_at,
            deadline_epoch,
        )
        self._catalog_change_issued_scopes[token] = (scope, completed_at)
        while len(self._catalog_change_cache) > _CATALOG_CHANGE_MAX_SCOPES:
            oldest_scope = min(
                self._catalog_change_cache,
                key=lambda item: self._catalog_change_cache[item][1],
            )
            self._catalog_change_cache.pop(oldest_scope, None)
        while (
            len(self._catalog_change_issued_scopes)
            > _CATALOG_CHANGE_MAX_ISSUED_TOKENS
        ):
            oldest_token = min(
                self._catalog_change_issued_scopes,
                key=lambda item: self._catalog_change_issued_scopes[item][1],
            )
            self._catalog_change_issued_scopes.pop(oldest_token, None)

    def catalog_change_token(
        self,
        *,
        run_id: str | None = None,
        force: bool = False,
    ) -> str:
        """Fingerprint catalog-visible project/revision publication metadata.

        The selected project's revision directory is always covered.  A
        bounded supplementary catalog sample keeps other projects reasonably
        fresh without replaying all historical runs on every probe.
        """

        scope = str(run_id or "")
        now = time.monotonic()
        wall_time = time.time()
        with self._catalog_change_cache_lock:
            cached = self._catalog_change_cache.get(scope)
            if not force and self._catalog_cache_entry_current(
                cached,
                now=now,
                wall_time=wall_time,
                ttl_s=self._catalog_change_cache_ttl_s,
            ):
                assert cached is not None
                self._catalog_change_issued_scopes[cached[0]] = (
                    scope,
                    cached[1],
                )
                return cached[0]

        # Serialize cache misses. A slower pre-change traversal must never
        # overwrite a newer token computed by another request.
        with self._catalog_change_compute_lock:
            now = time.monotonic()
            wall_time = time.time()
            with self._catalog_change_cache_lock:
                cached = self._catalog_change_cache.get(scope)
                if not force and self._catalog_cache_entry_current(
                    cached,
                    now=now,
                    wall_time=wall_time,
                    ttl_s=self._catalog_change_cache_ttl_s,
                ):
                    assert cached is not None
                    self._catalog_change_issued_scopes[cached[0]] = (
                        scope,
                        cached[1],
                    )
                    return cached[0]
            token, deadline_epoch = self._compute_catalog_change_token(
                run_id=run_id
            )
            completed_at = time.monotonic()
            with self._catalog_change_cache_lock:
                self._record_catalog_token_locked(
                    scope=scope,
                    token=token,
                    completed_at=completed_at,
                    deadline_epoch=deadline_epoch,
                )
            return token

    def _catalog_status_projection(
        self,
        path: Path,
    ) -> tuple[tuple[str, ...], float | None]:
        """Return stable status/lease phase plus its next time boundary."""

        metadata = _stat_record(path)
        cache_key = path.as_posix()
        wall_time = time.time()
        cached = self._catalog_status_projection_cache.get(cache_key)
        if (
            cached is not None
            and cached[0] == metadata
            and (cached[2] is None or wall_time <= cached[2])
        ):
            return cached[1], cached[2]
        deadline_epoch: float | None = None
        if not _regular_file_present(path):
            projection = ("missing",)
        else:
            try:
                value, _record = read_stable_json_object(
                    path,
                    label="dashboard catalog run status projection",
                    allowed_root=self.workspace_root,
                )
                status = str(value.get("status") or "").strip().upper()
                projection = (
                    "status",
                    status,
                )
                lease = value.get("running_lease")
                if status == "RUNNING" and isinstance(lease, dict):
                    heartbeat_text = str(
                        lease.get("heartbeat_at") or ""
                    ).strip()
                    try:
                        heartbeat = datetime.fromisoformat(
                            heartbeat_text.replace("Z", "+00:00")
                        )
                        if heartbeat.tzinfo is None:
                            heartbeat = heartbeat.replace(
                                tzinfo=timezone.utc
                            )
                        ttl_seconds = lease.get("ttl_seconds")
                        if (
                            not isinstance(ttl_seconds, int)
                            or isinstance(ttl_seconds, bool)
                            or not (
                                MIN_RUNNING_LEASE_TTL_S
                                <= ttl_seconds
                                <= MAX_RUNNING_LEASE_TTL_S
                            )
                        ):
                            raise ValueError("invalid running lease ttl")
                        lease_deadline = (
                            heartbeat.astimezone(timezone.utc).timestamp()
                            + ttl_seconds
                        )
                    except (OverflowError, TypeError, ValueError):
                        projection = (*projection, "lease", "invalid")
                    else:
                        healthy = wall_time <= lease_deadline
                        projection = (
                            *projection,
                            "lease",
                            "healthy" if healthy else "expired",
                        )
                        if healthy:
                            deadline_epoch = lease_deadline
            except (OSError, ValueError, OperatorAuthorityError) as exc:
                projection = ("invalid", type(exc).__name__)
        self._catalog_status_projection_cache[cache_key] = (
            metadata,
            projection,
            deadline_epoch,
        )
        return projection, deadline_epoch

    def _compute_catalog_change_token(
        self,
        *,
        run_id: str | None,
    ) -> tuple[str, float | None]:
        next_deadline_epoch: float | None = None
        scanned_entries = 0
        project_count = 0
        revision_count = 0
        legacy_count = 0
        seen_revisions: set[str] = set()

        def status_projection(path: Path) -> tuple[str, ...]:
            nonlocal next_deadline_epoch
            projection, deadline_epoch = self._catalog_status_projection(path)
            if deadline_epoch is not None and (
                next_deadline_epoch is None
                or deadline_epoch < next_deadline_epoch
            ):
                next_deadline_epoch = deadline_epoch
            return projection

        digest = hashlib.sha256()
        _digest_record(
            digest,
            _CHANGE_TOKEN_SCHEMA,
            "catalog",
            str(run_id or ""),
        )

        def directory_entries(
            directory: Path,
            *,
            label: str,
            limit: int,
        ) -> list[os.DirEntry[str]]:
            """Read at most ``limit`` entries from one real directory."""

            nonlocal scanned_entries
            metadata = _stat_record(directory)
            _digest_record(digest, "directory", label, *metadata)
            if not (
                len(metadata) > 2
                and metadata[0] == "present"
                and metadata[1] == stat.S_IFDIR
            ):
                _digest_record(digest, "directory-skipped", label)
                return []
            remaining = (
                _CATALOG_CHANGE_MAX_SCANNED_ENTRIES - scanned_entries
            )
            permitted = max(0, min(limit, remaining))
            if permitted <= 0:
                _digest_record(digest, "directory-overflow", label, "budget")
                return []
            try:
                with os.scandir(directory) as iterator:
                    values: list[os.DirEntry[str]] = []
                    for _index in range(permitted):
                        try:
                            entry = next(iterator)
                        except StopIteration:
                            break
                        values.append(entry)
            except OSError as exc:
                _digest_record(
                    digest,
                    "scan-unavailable",
                    label,
                    type(exc).__name__,
                )
                return []
            scanned_entries += len(values)
            values.sort(key=lambda item: item.name)
            if len(values) >= permitted:
                # This may conservatively mark an exactly-full directory.  The
                # directory stat still detects additions/removals without an
                # unbounded look-ahead read.
                _digest_record(
                    digest,
                    "directory-overflow",
                    label,
                    "entry-limit",
                    permitted,
                )
            return values

        def add_revision(revision_dir: Path, *, label: str) -> None:
            key = revision_dir.as_posix()
            if key in seen_revisions:
                return
            seen_revisions.add(key)
            _digest_record(
                digest,
                label,
                "revision",
                *status_projection(revision_dir / "run_status.json"),
                *_stat_record(revision_dir / "artifacts.sha256.json"),
                *_stat_record(
                    revision_dir / "dashboard" / "publication.json"
                ),
            )

        scoped_project: Path | None = None
        if run_id:
            scoped_run = self.resolve_run(run_id)
            _digest_record(
                digest,
                "selected-run",
                run_id,
                *_stat_record(scoped_run.parent),
            )
            identity = parse_canonical_run(self.workspace_root, scoped_run)
            if identity is not None:
                scoped_project = scoped_run.parent.parent
                scoped_runs_root = scoped_run.parent
                # exFAT directory mtimes can remain unchanged across rapid
                # revision creation. Names are cheap; no revision contents are read.
                try:
                    with os.scandir(scoped_runs_root) as iterator:
                        names = []
                        for entry in iterator:
                            names.append(entry.name)
                            if len(names) >= _RUN_CHANGE_MAX_SCANNED_ENTRIES:
                                break
                    _digest_record(digest, "selected-revision-names", *sorted(names))
                except OSError:
                    pass
                _digest_record(
                    digest,
                    "selected-project",
                    identity.project_id,
                    *_stat_record(scoped_project / "project.json"),
                    *_stat_record(scoped_runs_root),
                    *_stat_record(
                        self.workspace_root
                        / ".runtime_runs"
                        / "project_usage"
                        / "projects"
                        / f"{identity.project_id}.json"
                    ),
                )
                add_revision(scoped_run, label=f"selected/{run_id}")
                scoped_entries = directory_entries(
                    scoped_runs_root,
                    label=f"selected/{identity.project_id}/runs",
                    limit=_CATALOG_CHANGE_MAX_SCOPED_REVISIONS,
                )
                for entry in scoped_entries:
                    if not entry.is_dir(follow_symlinks=False):
                        continue
                    revision_count += 1
                    add_revision(
                        Path(entry.path),
                        label=(
                            f"selected/{identity.project_id}/runs/"
                            f"{entry.name}"
                        ),
                    )
            else:
                # A legacy selected run still gets exact status/publication
                # coverage, while its parent stat detects sibling creation.
                add_revision(scoped_run, label=f"selected/{run_id}")

        if run_id:
            # Notice new projects and their current revisions without walking
            # historical runs or loading geometry/trace files.
            runtime_roots, _overflow = self._change_runtime_roots()
            for runtime_root in runtime_roots:
                projects_root = runtime_root / "projects"
                _digest_record(digest, "project-list", str(projects_root), *_stat_record(projects_root))
                for entry in directory_entries(
                    projects_root, label=f"{projects_root}:live-projects", limit=_CATALOG_CHANGE_MAX_PROJECTS,
                ):
                    if entry.is_dir(follow_symlinks=False):
                        project = Path(entry.path)
                        _digest_record(digest, "project-current", str(project),
                                       *_stat_record(project / "workflow.json"),
                                       *_stat_record(project / "project.json"))
            _digest_record(
                digest,
                "catalog-bounds",
                scanned_entries,
                project_count,
                revision_count,
                legacy_count,
                _CATALOG_CHANGE_MAX_SCANNED_ENTRIES,
                "selected-scope-only",
            )
            return digest.hexdigest(), next_deadline_epoch

        runtime_roots, runtime_roots_overflow = self._change_runtime_roots()
        _digest_record(
            digest,
            "runtime-root-discovery",
            *_stat_record(self.workspace_root),
            len(runtime_roots),
            runtime_roots_overflow,
        )
        for runtime_root in runtime_roots:
            try:
                root_label = runtime_root.relative_to(
                    self.workspace_root
                ).as_posix()
            except ValueError:
                continue
            root_entries = directory_entries(
                runtime_root,
                label=f"{root_label}:runtime-root",
                limit=max(
                    0,
                    _CATALOG_CHANGE_MAX_LEGACY_RUNS - legacy_count,
                ),
            )

            # Legacy runs are direct children with a run status.
            for entry in root_entries:
                if (
                    entry.name == "projects"
                    or not entry.is_dir(follow_symlinks=False)
                ):
                    continue
                run_dir = Path(entry.path)
                status_path = run_dir / "run_status.json"
                if not _regular_file_present(status_path):
                    continue
                legacy_count += 1
                add_revision(
                    run_dir,
                    label=f"{root_label}/legacy/{entry.name}",
                )

            projects_root = runtime_root / "projects"
            projects = directory_entries(
                projects_root,
                label=f"{root_label}/projects",
                limit=max(0, _CATALOG_CHANGE_MAX_PROJECTS - project_count),
            )
            for project_entry in projects:
                if not project_entry.is_dir(follow_symlinks=False):
                    continue
                project_count += 1
                project_dir = Path(project_entry.path)
                project_label = f"{root_label}/projects/{project_entry.name}"
                _digest_record(
                    digest,
                    project_label,
                    "project",
                    *_stat_record(project_dir / "project.json"),
                )
                revisions_root = project_dir / "runs"
                if scoped_project == project_dir:
                    # The selected project was already scanned first, before
                    # the global history budget could be consumed.
                    _digest_record(
                        digest,
                        f"{project_label}/runs",
                        "selected-project-covered",
                        *_stat_record(revisions_root),
                    )
                    continue
                revisions = directory_entries(
                    revisions_root,
                    label=f"{project_label}/runs",
                    limit=max(
                        0,
                        _CATALOG_CHANGE_MAX_REVISIONS - revision_count,
                    ),
                )
                for revision_entry in revisions:
                    if not revision_entry.is_dir(follow_symlinks=False):
                        continue
                    revision_count += 1
                    revision_dir = Path(revision_entry.path)
                    add_revision(
                        revision_dir,
                        label=(
                            f"{project_label}/runs/{revision_entry.name}"
                        ),
                    )
        _digest_record(
            digest,
            "catalog-bounds",
            scanned_entries,
            project_count,
            revision_count,
            legacy_count,
            _CATALOG_CHANGE_MAX_SCANNED_ENTRIES,
        )
        return digest.hexdigest(), next_deadline_epoch

    def dashboard_change_snapshot(self, run_id: str) -> dict[str, str]:
        run_dir = self.resolve_run(run_id)
        catalog_token = self.catalog_change_token(run_id=run_id)
        with self._catalog_change_compute_lock:
            status_projection, _deadline_epoch = (
                self._catalog_status_projection(run_dir / "run_status.json")
            )
        run_token = _run_surface_metadata_digest(
            run_dir,
            status_time_projection=status_projection,
        )
        return {
            "catalog_token": catalog_token,
            "run_token": run_token,
            "token": f"{_CHANGE_TOKEN_PREFIX}.{catalog_token}.{run_token}",
        }

    def wait_for_dashboard_change(
        self,
        *,
        run_id: str,
        since: str,
        timeout_s: float,
    ) -> tuple[dict[str, str], bool, bool] | None:
        """Wait for a stat-derived change; ``None`` is a keepalive timeout."""

        previous = _token_parts(since)
        if previous is None:
            raise ValueError("since must be a dashboard change token")
        deadline = time.monotonic() + timeout_s
        snapshot = self.dashboard_change_snapshot(run_id)
        observed_at = time.monotonic()
        current = (snapshot["catalog_token"], snapshot["run_token"])
        if current != previous:
            return (
                snapshot,
                current[0] != previous[0],
                current[1] != previous[1],
            )
        if timeout_s <= 0:
            return None

        with self._change_watch_condition:
            self._change_watch_refcounts[run_id] = (
                self._change_watch_refcounts.get(run_id, 0) + 1
            )
            recorded = self._change_watch_snapshots.get(run_id)
            if recorded is None or recorded[0] <= observed_at:
                self._change_watch_snapshots[run_id] = (
                    observed_at,
                    snapshot,
                )
            self._ensure_change_watcher_locked()
            try:
                while not self._change_watch_stop.is_set():
                    recorded = self._change_watch_snapshots.get(run_id)
                    watched = recorded[1] if recorded is not None else snapshot
                    current = (
                        watched["catalog_token"],
                        watched["run_token"],
                    )
                    if current != previous:
                        return (
                            watched,
                            current[0] != previous[0],
                            current[1] != previous[1],
                        )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                    self._change_watch_condition.wait(timeout=remaining)
                return None
            finally:
                remaining_refs = self._change_watch_refcounts.get(run_id, 1) - 1
                if remaining_refs > 0:
                    self._change_watch_refcounts[run_id] = remaining_refs
                else:
                    self._change_watch_refcounts.pop(run_id, None)
                    self._change_watch_snapshots.pop(run_id, None)

    def fresh_runs_for_catalog_change(
        self,
        catalog_token: str,
    ) -> list[dict[str, object]]:
        """Perform at most one catalog scan for one observed change token."""

        if not _CHANGE_DIGEST_RE.fullmatch(catalog_token):
            raise ValueError("change token is invalid")
        with self._runs_fresh_lock:
            with self._catalog_change_cache_lock:
                issued = self._catalog_change_issued_scopes.get(
                    catalog_token
                )
            if issued is None:
                raise CatalogChangeConflict(
                    "catalog change token was not issued by this server"
                )
            scope, _issued_at = issued
            with self._runs_cache_lock:
                if (
                    self._runs_cache is not None
                    and self._runs_cache_catalog_token == catalog_token
                ):
                    return copy.deepcopy(self._runs_cache)
            try:
                current_token = self.catalog_change_token(
                    run_id=scope or None,
                    force=True,
                )
            except KeyError as exc:
                raise CatalogChangeConflict(
                    "catalog change scope is no longer available"
                ) from exc
            if current_token != catalog_token:
                raise CatalogChangeConflict(
                    "catalog change token is stale or was not issued by this server"
                )
            rows = self._scan_runs_catalog()
            token_after_scan = self.catalog_change_token(
                run_id=scope or None,
                force=True,
            )
            with self._runs_cache_lock:
                self._runs_cache = rows
                self._runs_cache_completed_at = time.monotonic()
                self._runs_cache_refreshing = False
                self._runs_cache_catalog_token = (
                    catalog_token
                    if token_after_scan == catalog_token
                    else None
                )
            return copy.deepcopy(rows)

    def resolve_run(self, run_id: str) -> Path:
        from .live_workflow import is_cold_start_run

        # Exact run reads must not depend on a full historical catalog. Keep
        # the same boundary as scan_runs: a real directory under runs_root
        # with a run_status.json, with no lexical alias or symlink indirection.
        if not isinstance(run_id, str) or not run_id:
            raise KeyError(run_id)
        lexical = Path(run_id)
        if (
            lexical.is_absolute()
            or lexical.drive
            or lexical.as_posix() != run_id
            or any(part in {"", ".", ".."} for part in lexical.parts)
        ):
            raise KeyError(run_id)
        candidate = self.workspace_root / lexical
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.runs_root)
            exact = resolved.relative_to(self.workspace_root).as_posix()
        except ValueError as exc:
            raise KeyError(run_id) from exc
        if (
            exact != run_id
            or not resolved.is_dir()
            or (
                not (resolved / "run_status.json").is_file()
                and not is_react_loop_run(self.workspace_root, resolved)
                and not is_cold_start_run(workspace_root=self.workspace_root, run_dir=resolved)
            )
        ):
            raise KeyError(run_id)
        return resolved

    def _revision_row(self, run_id: str) -> dict[str, object]:
        row = next(
            (item for item in self.runs() if item.get("id") == run_id),
            None,
        )
        if row is None:
            raise KeyError(run_id)
        return row

    def read_revision_frame(self, run_id: str) -> tuple[Path, bytes]:
        """Read the catalog-bound fixed camera frame used only for playback."""

        run_dir = self.resolve_run(run_id)
        row = self._revision_row(run_id)
        relative = str(row.get("revision_preview_path") or "")
        expected = str(row.get("revision_preview_sha256") or "")
        if not relative or len(expected) != 64:
            raise ValueError("revision has no fixed camera frame")
        lexical = self.workspace_root / relative
        if lexical.is_symlink():
            raise ValueError("revision frame cannot use a symbolic link")
        candidate = lexical.resolve()
        candidate.relative_to(run_dir)
        if not candidate.is_file():
            raise ValueError("revision frame is unavailable")
        payload = candidate.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError("revision frame changed after catalog binding")
        return candidate, payload

    def read_revision_mesh(self, run_id: str) -> bytes:
        """Derive and cache one draggable GLB without mutating the run."""

        run_dir = self.resolve_run(run_id)
        run_token = self.dashboard_change_snapshot(run_id)["run_token"]
        cache_key = (run_id, run_token)
        lock = getattr(self, "_revision_mesh_cache_lock", None)
        cache = getattr(self, "_revision_mesh_cache", None)
        if lock is None or cache is None:
            lock = threading.Lock()
            cache = {}
            self._revision_mesh_cache_lock = lock
            self._revision_mesh_cache = cache
        with lock:
            cached = cache.get(cache_key)
        if cached is not None:
            return cached
        from .scene_assets import interactive_glb_bytes, load_revision_scene_meshes

        scene = load_revision_scene_meshes(
            workspace_root=self.workspace_root,
            run_dir=run_dir,
        )
        payload = interactive_glb_bytes(scene, capsule_id="revision")
        with lock:
            cache[cache_key] = payload
            while len(cache) > 32:
                cache.pop(next(iter(cache)))
        return payload
    def studio_preview(self, run_id: str, run_dir: Path) -> dict[str, Any]:
        from .preview_service import ensure_preview, preview_media_paths, preview_timeline

        from .design_history import history_view
        from .preview_service import _slim_preview
        history = history_view(self.workspace_root, run_id, run_dir)
        if history["total"]:
            preview = dict(_slim_preview(run_dir))
            if not preview:
                selected = history.get("selected") or {}
                preview = {"source_kind":"original_geometry", "source_revision":run_dir.name,
                    "model_url":selected.get("model_url"), "input_version":selected.get("id"),
                    "status":selected.get("preview_status","missing"), "error":selected.get("preview_error")}
        else:
            preview = dict(ensure_preview(workspace_root=self.workspace_root, run_dir=run_dir))
        def attach_urls(items, owner_id, media, version_default):
            for item in items:
                for kind in ("model", "motion_model", "scene", "replay", "preview", "source"):
                    raw = item.get(f"{kind}_path")
                    if raw:
                        target = Path(raw)
                        if not target.is_absolute():
                            target = self.workspace_root / target
                        name = next((name for name, path in media.items() if Path(path).resolve() == target.resolve()), None)
                        if name is not None:
                            version = version_default if kind == "model" else item.get("replay_version", version_default)
                            item[f"{kind}_url"] = (
                                f"/api/preview-file?run={quote(owner_id, safe='')}&name={quote(name, safe='')}"
                                f"&v={quote(str(version or ''), safe='')}"
                            )

        from ..slim_harness.motion_preview import best_motion
        media = preview_media_paths(workspace_root=self.workspace_root, run_dir=run_dir)
        attach_urls([preview, *(preview.get("motions") or [])], run_id, media, preview.get("input_version"))
        best = best_motion(preview.get("motions") or [])
        if best:
            preview["best_motion_id"] = best.get("id")
        elif run_dir.parent.name == "runs":
            # This revision has no playable motion yet: show the newest earlier one.
            for earlier in sorted((path for path in run_dir.parent.glob("r[0-9]*") if path.is_dir() and path.name < run_dir.name), reverse=True):
                earlier_preview = _slim_preview(earlier)
                fallback = best_motion(earlier_preview.get("motions") or [])
                if fallback:
                    # Test ids repeat across revisions: the earlier one is chosen by its own id.
                    fallback = {**fallback, "revision": earlier.name, "id": f"{earlier.name}:{fallback.get('id')}"}
                    attach_urls([fallback], str(PurePosixPath(run_id).parent / earlier.name),
                                preview_media_paths(workspace_root=self.workspace_root, run_dir=earlier),
                                earlier_preview.get("input_version"))
                    if fallback.get("replay_url"):
                        preview["fallback_motion"] = fallback
                    break
        timeline = []
        for frame in preview_timeline(workspace_root=self.workspace_root, run_dir=run_dir):
            name = "timeline/" + frame["model"]
            if name not in media:
                continue
            timeline.append({
                "id": frame["id"], "created_at": frame.get("created_at"), "bytes": frame.get("bytes", 0),
                "active": frame["id"] == preview.get("input_version"),
                "model_url": (
                    f"/api/preview-file?run={quote(run_id, safe='')}&name={quote(name, safe='')}"
                    f"&v={quote(frame['id'], safe='')}"
                ),
            })
        preview["timeline"] = timeline
        if history["total"]:
            preview["timeline"] = history["items"]
        if "markers" in history: preview["history"] = history
        return preview

    def live_progress(self, run_id: str) -> dict[str, Any]:
        from ..snapshot_paths import imported
        from .live_workflow import read_live_project

        run_dir = self.resolve_run(run_id)
        projection = read_live_project(workspace_root=self.workspace_root, run_dir=run_dir)
        workflow = projection.get("live_workflow") or {}
        payload = {
            "run_id": run_id,
            **{key: value or None for key, value in projection.items()},
            "activity": self.studio_activity(run_dir, workflow),
            "server_time": datetime.now(timezone.utc).isoformat(),
        }
        if imported(self.workspace_root) is None:
            payload["studio_preview"] = self.studio_preview(run_id, run_dir)
        return payload

    def motion_thumbnails(self):
        with self._image_service_lock:
            if getattr(self, "_motion_thumbnails", None) is None:
                from .motion_thumbnail import MotionThumbnails
                from .preview_service import resolve_cad_python
                self._motion_thumbnails = MotionThumbnails(resolve_cad_python())
            return self._motion_thumbnails

    def motion_for(self, run_id: str, motion_id: str):
        """(revision folder, motion with absolute paths, thumbnail file, up axis) of one recorded motion."""
        from ..slim_harness.motion_preview import _read, revision_motions
        from .motion_thumbnail import thumbnail_file
        from .preview_service import _cache_root
        run = self.resolve_run(run_id)
        motion = next((dict(item) for item in revision_motions(run) if item.get("id") == motion_id), None)
        if motion is None:
            raise ValueError("No recorded motion " + motion_id)
        for field in ("motion_model_path", "scene_path", "replay_path"):
            raw = Path(str(motion.get(field) or ""))
            motion[field] = str(raw if raw.is_absolute() else self.workspace_root / raw)
        gravity = _read(run / "design" / "scene.json").get("gravity_m_s2") or [0, -9.81, 0]
        length = sum(float(x) ** 2 for x in gravity) ** .5 if len(gravity) == 3 else 0
        up = [-float(x) / length for x in gravity] if length > 1e-9 else [0, 1, 0]
        return run, motion, thumbnail_file(_cache_root(self.workspace_root, run), motion), up

    def images(self):
        with self._image_service_lock:
            if self._image_service is None:
                from .image_service import ImageService
                self._image_service=ImageService(self.workspace_root)
            return self._image_service

    def studio_activity(self, run_dir, workflow):
        from .project_summary import activity_view
        from ..trajectory.store import project_from_path
        project = project_from_path(run_dir)
        return activity_view(project, workflow) if project else {"available":False,"summary":"暂无公开活动记录"}

    def _has_bound_dashboard_publication(self, run_dir: Path) -> bool:
        from .publication import load_bound_dashboard_publication

        try:
            status, _record = read_stable_json_object(
                run_dir / "run_status.json",
                label="dashboard exact run status",
                allowed_root=self.workspace_root,
            )
        except (OSError, ValueError, OperatorAuthorityError):
            return False
        publication, _reason = load_bound_dashboard_publication(
            workspace_root=self.workspace_root,
            run_dir=run_dir,
            terminal_status=status,
            require_inventory=True,
        )
        return publication is not None

    def legacy_display_media(
        self,
        *,
        run_id: str,
        run_dir: Path,
        relative_path: str,
    ) -> tuple[Path, bytes]:
        """Read media from the exact pre-v2 cohort for display only.

        This grants no terminal, certificate, or publication authority. It
        is deliberately unavailable to a run that binds or claims the
        governed dashboard publisher.
        """

        if self._has_bound_dashboard_publication(run_dir):
            raise ValueError(
                "legacy preview fallback is unavailable to published runs"
            )

        from .legacy_preview import read_legacy_dashboard_media

        return read_legacy_dashboard_media(
            workspace_root=self.workspace_root,
            run_dir=run_dir,
            run_id=run_id,
            relative_path=relative_path,
        )

    def read_media(
        self,
        *,
        run_id: str,
        relative_path: str,
    ) -> tuple[Path, bytes | None]:
        """Resolve one media file from the selected run's read-only allowlist."""

        run_dir = self.resolve_run(run_id)
        react_run = is_react_loop_run(self.workspace_root, run_dir)
        candidate = (self.workspace_root / relative_path).resolve()
        normalized = candidate.relative_to(self.workspace_root).as_posix()
        payload: bytes | None = None
        allowlist = self.media_allowlist(run_dir)
        if normalized not in allowlist:
            if react_run:
                raise ValueError("media is not listed by this ReAct revision")
            candidate, payload = self.legacy_display_media(
                run_id=run_id,
                run_dir=run_dir,
                relative_path=relative_path,
            )
        return candidate, payload

    def evidence_roots(self, run_dir: Path) -> list[Path]:
        roots = [run_dir.resolve()]
        identity = parse_canonical_run(self.workspace_root, run_dir)
        if identity is not None:
            ideal_pose_diagnostics = (
                self.workspace_root
                / ".runtime_runs"
                / "diagnostics"
                / f"{identity.run_id}-ideal-pose"
            ).resolve()
            diagnostics_root = (
                self.workspace_root / ".runtime_runs" / "diagnostics"
            ).resolve()
            for suffix in ("ideal-pose", "physics-evidence"):
                diagnostic_dir = (
                    self.workspace_root
                    / ".runtime_runs"
                    / "diagnostics"
                    / f"{identity.run_id}-{suffix}"
                ).resolve()
                try:
                    diagnostic_dir.relative_to(diagnostics_root)
                    if diagnostic_dir.is_dir():
                        roots.append(diagnostic_dir)
                except ValueError:
                    pass
        try:
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
            continuation = status.get("continuation") if isinstance(status, dict) else None
            if (
                isinstance(continuation, dict)
                and continuation.get("mode")
                == "cross_run_whole_machine_continuation_v1"
                and continuation.get("upstream_run")
            ):
                upstream = resolve_workspace_reference(
                    self.workspace_root,
                    continuation["upstream_run"],
                    require_within_workspace=True,
                    require_exists=True,
                )
                upstream.relative_to(self.runs_root)
                if (upstream / "run_status.json").is_file():
                    roots.append(upstream)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return roots

    def media_allowlist(self, run_dir: Path) -> dict[str, str]:
        """Return media approved by the new or legacy read-only adapter."""

        if is_react_loop_run(self.workspace_root, run_dir):
            allowed = react_v2_media_allowlist(
                workspace_root=self.workspace_root,
                run_dir=run_dir,
            )
            allowed.update(
                react_static_visual_media_allowlist(
                    workspace_root=self.workspace_root,
                    run_dir=run_dir,
                )
            )
            allowed.update(
                react_reference_pose_media_allowlist(
                    workspace_root=self.workspace_root,
                    run_dir=run_dir,
                )
            )
            allowed.update(
                react_physics_evidence_media_allowlist(
                    workspace_root=self.workspace_root,
                    run_dir=run_dir,
                )
            )
            return allowed

        from .publication import published_media_allowlist

        allowed: dict[str, str] = {}
        for evidence_root in self.evidence_roots(run_dir):
            status_path = evidence_root / "run_status.json"
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(status, dict):
                continue
            allowed.update(
                published_media_allowlist(
                    workspace_root=self.workspace_root,
                    run_dir=evidence_root,
                    terminal_status=status,
                )
            )
        return allowed


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_bytes(
        self,
        payload: bytes,
        *,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        disposition: str | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers may close in-flight frame-prefetch requests when the
            # replay panel or tab is closed.
            return

    def _send_json(self, value: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(prefix_url_fields(value, self.server.base_path), ensure_ascii=False).encode("utf-8"),
            content_type="application/json; charset=utf-8",
            status=status,
        )

    def _human_decision_authorized(self) -> bool:
        supplied = str(
            self.headers.get("X-Human-Decision-Capability", "")
        )
        expected = str(
            getattr(self.server, "human_decision_capability", "")
        )
        return bool(
            supplied
            and expected
            and secrets.compare_digest(supplied, expected)
        )

    def _require_human_decision_capability(self) -> bool:
        if self._human_decision_authorized():
            return True
        self._send_json(
            {
                "error": "HUMAN_DECISION_CAPABILITY_REQUIRED",
                "detail": (
                    "open the human Loop dashboard URL printed by the "
                    "dashboard server"
                ),
            },
            status=HTTPStatus.FORBIDDEN,
        )
        return False

    def _send_change_keepalive(self, snapshot: dict[str, str]) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Dashboard-Change-Token", snapshot["token"])
        self.send_header(
            "X-Dashboard-Catalog-Token",
            snapshot["catalog_token"],
        )
        self.send_header("X-Dashboard-Run-Token", snapshot["run_token"])
        self.end_headers()

    def _launch_enabled(self) -> bool:
        # Import contexts are built without launch; older test fixtures lack the attribute.
        return bool(getattr(self.server, "launch_enabled", False))

    def _send_live_shell(self, page: str = "index.html") -> None:
        features = {
            "maintenance": maintenance_routes is not None,
            "team_library": TEAM_LIBRARY,
            "launch": self._launch_enabled(),
        }
        self._send_bytes(
            render_live_shell(self.server.base_path, features, page).encode("utf-8"),
            content_type="text/html; charset=utf-8",
        )

    def _send_launch_error(self, exc) -> None:
        self._send_json(
            {"error": exc.code, "detail": str(exc), **({"log_tail": exc.log} if exc.log else {})},
            status=HTTPStatus(exc.status),
        )

    def _launch_get(self, parsed) -> None:
        from . import launcher

        if parsed.path == "/launch":
            self._send_live_shell("launch.html")
            return
        query = parse_qs(parsed.query)
        workspace = self.server.workspace_root
        try:
            if parsed.path == "/api/launch/options":
                value = launcher.options(
                    workspace,
                    runs_root=self.server.runs_root,
                    refresh=query.get("refresh") == ["1"],
                    limit=getattr(self.server, "launch_limit", 1),
                )
            else:
                value = launcher.status(workspace, (query.get("project") or [""])[0])
        except launcher.LaunchError as exc:
            self._send_launch_error(exc)
            return
        self._send_json(value)

    def _launch_post(self, parsed) -> None:
        from . import launcher

        if not self._require_human_decision_capability():
            return
        if str(self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            self._send_json(
                {"error": "LAUNCH_CONTENT_TYPE", "detail": "launch requests must be application/json"},
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return
        origin = self.headers.get("Origin")
        if origin is not None and urlparse(origin).netloc != self.headers.get("Host"):
            self._send_json(
                {"error": "LAUNCH_ORIGIN", "detail": "cross-origin launch requests are refused"},
                status=HTTPStatus.FORBIDDEN,
            )
            return
        body = self._read_json_body()
        if body is None:
            return
        workspace = self.server.workspace_root
        limit = getattr(self.server, "launch_limit", 1)
        try:
            if parsed.path == "/api/launch":
                request = launcher.validate(workspace, body, runs_root=self.server.runs_root)
                value = launcher.spawn(workspace, request, limit=limit)
            elif parsed.path == "/api/launch/queue":
                # The scheduler thread starts it within QUEUE_POLL_SECONDS once nothing else runs.
                value = launcher.enqueue(workspace, body, runs_root=self.server.runs_root)
            elif parsed.path == "/api/launch/queue/cancel":
                if set(body) != {"id"}:
                    raise launcher.LaunchError("LAUNCH_FIELD_UNKNOWN", "send only the queue entry id")
                value = launcher.cancel_queued(workspace, body["id"])
            elif set(body) != {"project"}:
                raise launcher.LaunchError("LAUNCH_FIELD_UNKNOWN", "send only the project id")
            elif parsed.path == "/api/launch/stop":
                value = launcher.stop(workspace, body["project"])
            else:
                value = launcher.resume(workspace, body["project"], runs_root=self.server.runs_root, limit=limit)
        except launcher.LaunchError as exc:
            self._send_launch_error(exc)
            return
        self.server.forget_runs_overview()
        self._send_json(value)

    def _request_url(self):
        parsed = urlparse(self.path)
        base_path = self.server.base_path
        if base_path:
            if parsed.path == base_path:
                return parsed._replace(path="/")
            if not parsed.path.startswith(base_path + "/"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return None
            parsed = parsed._replace(path=parsed.path[len(base_path):])
        return parsed

    def _dispatch_import(self, method):
        parsed = self._request_url()
        if parsed is None:
            return True
        if not parsed.path.startswith('/imports/'):
            return False
        identity = parsed.path.split('/')[2]
        parent = self.server
        try:
            self.server = parent.import_context(identity)
        except KeyError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return True
        try:
            getattr(self, method)()
        finally:
            self.server = parent
        return True

    def do_GET(self) -> None:
        if self._dispatch_import('do_GET'):
            return
        parsed = self._request_url()
        if parsed is None:
            return
        if parsed.path == '/api/team-library' and TEAM_LIBRARY:
            self._send_json(self.server.team_library())
            return
        if parsed.path in {"/", "/index.html"}:
            self._send_live_shell()
            return
        if maintenance_routes is not None and maintenance_routes.handle_get(self, parsed):
            return
        if parsed.path == "/runs":
            self._send_live_shell("runs.html")
            return
        if parsed.path == "/api/runs-overview":
            self._send_json(self.server.runs_overview())
            return
        if parsed.path in _LAUNCH_GET_ROUTES:
            if self._launch_enabled():
                self._launch_get(parsed)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parsed.path.startswith("/static/"):
            name = parsed.path.removeprefix("/static/")
            if name not in {"app.css", "app.js", "launch.js", "runs.js", "viewer3d.js", "workbench.js", "i18n.js", "i18n/en.js"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            path = STATIC_ROOT / name
            content_type = (
                "text/css; charset=utf-8"
                if path.suffix == ".css"
                else "text/javascript; charset=utf-8"
            )
            self._send_bytes(path.read_bytes(), content_type=content_type)
            return
        if parsed.path == "/api/runs":
            query = parse_qs(parsed.query)
            fresh_values = query.get("fresh") or []
            if fresh_values and fresh_values != ["1"]:
                self._send_json(
                    {"error": "invalid_catalog_refresh"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            change_values = query.get("change") or []
            if len(change_values) > 1:
                self._send_json(
                    {"error": "invalid_catalog_change_token"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if fresh_values and change_values:
                try:
                    runs = self.server.fresh_runs_for_catalog_change(
                        change_values[0]
                    )
                except CatalogChangeConflict as exc:
                    self._send_json(
                        {
                            "error": "catalog_change_conflict",
                            "detail": str(exc),
                        },
                        status=HTTPStatus.CONFLICT,
                    )
                    return
                except ValueError as exc:
                    self._send_json(
                        {
                            "error": "invalid_catalog_change_token",
                            "detail": str(exc),
                        },
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                except Exception as exc:
                    self._send_json(
                        {
                            "error": "catalog_refresh_failed",
                            "detail": str(exc),
                        },
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
            elif fresh_values:
                try:
                    runs = self.server.refresh_runs()
                except Exception as exc:
                    self._send_json(
                        {
                            "error": "catalog_refresh_failed",
                            "detail": str(exc),
                        },
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
            else:
                runs = self.server.runs()
            catalog = self.server.catalog_state()
            # The initial refresh can finish between those two reads. Take
            # the now-hot snapshot instead of reporting a ready-but-empty
            # transient that the UI could mistake for an empty workspace.
            if not runs and catalog["ready"]:
                runs = self.server.runs()
                catalog = self.server.catalog_state()
            self._send_json(
                {
                    "projects": group_runs_by_project(runs),
                    "runs": runs,
                    "catalog_ready": catalog["ready"],
                    "catalog_refreshing": catalog["refreshing"],
                    "team_library": self.server.team_library(),
                }
            )
            return
        if parsed.path == "/api/motion-thumbnail":
            query = parse_qs(parsed.query)
            try:
                _run, _motion, target, _up = self.server.motion_for((query.get("run") or [""])[0], (query.get("motion") or [""])[0])
                self._send_bytes(target.read_bytes(), content_type="image/gif")
            except (OSError, ValueError, KeyError) as exc:
                self._send_json({"error": "motion_thumbnail_not_available", "detail": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        if parsed.path in {"/api/project-thumbnail","/api/history-compare","/api/workbench-image"}:
            from .image_service import model_reference
            from ..trajectory.store import project_from_path
            query=parse_qs(parsed.query)
            def argument(name,default=None):return (query.get(name) or [default])[0]
            try:
                if parsed.path=="/api/workbench-image":
                    image=self.server.images().read_image(argument("id",""),int(argument("index",0)))
                    self._send_bytes(image.read_bytes(),content_type="image/png")
                    return
                run=self.server.resolve_run(argument("run", ""))
                project=project_from_path(run)
                if parsed.path=="/api/project-thumbnail":
                    from .preview_service import _slim_preview,preview_media_paths
                    existing=_slim_preview(run)
                    image_path=existing.get("preview_path")
                    if image_path and Path(image_path).is_file():
                        media=preview_media_paths(workspace_root=self.server.workspace_root,run_dir=run)
                        name=next((name for name,path in media.items() if Path(path)==Path(image_path)),None)
                        if name:
                            self._send_json({"status":"ready","urls":[f"/api/preview-file?run={quote(argument('run'),safe='')}&name={quote(name,safe='')}"]})
                            return
                    reference=model_reference(self.server.workspace_root,run,prepare=True)
                    value=self.server.images().request(project,[reference])
                else:
                    if not argument("a") or not argument("b"):raise ValueError("Choose two saved model records")
                    other=self.server.resolve_run(argument("other_run",""))
                    if project_from_path(other)!=project:raise ValueError("A/B models must belong to the same project")
                    references=[model_reference(self.server.workspace_root,run,argument("a"),prepare=True),
                                model_reference(self.server.workspace_root,other,argument("b"),prepare=True)]
                    value=self.server.images().request(project,references,kind="compare",view=argument("view","iso"))
                payload={key:value.get(key) for key in ("id","status","error","camera","width","height")}
                payload["urls"]=[f"/api/workbench-image?id={quote(value['id'],safe='')}&index={i}" for i,_ in enumerate(value.get("images",[]))]
                self._send_json(payload)
            except (ValueError,KeyError,OSError) as exc:
                self._send_json({"error":"image_not_available","detail":str(exc)},status=HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/project-summary":
            from .project_summary import project_summary
            self._send_json(project_summary(self.server))
            return
        if parsed.path == "/api/run-config":
            from .run_config import configuration_view
            query = parse_qs(parsed.query)
            try:
                run = self.server.resolve_run((query.get("run") or [""])[0])
                value = configuration_view(run,snapshot_id=(query.get("snapshot") or [None])[0],test_id=(query.get("test") or [None])[0])
                self._send_json(value)
            except (ValueError,KeyError,OSError) as exc:
                self._send_json({"error":"config_not_available","detail":str(exc)},status=HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/agent-trajectory":
            from .agent_trajectory import trajectory_view
            query = parse_qs(parsed.query)
            def _millis(name):
                raw = (query.get(name) or [None])[0]
                try:
                    return int(raw) if raw not in (None, "") else None
                except (TypeError, ValueError):
                    return None
            try:
                run_dir = self.server.resolve_run((query.get("run") or [""])[0])
                view = trajectory_view(workspace_root=self.server.workspace_root, run_dir=run_dir,
                                       turn=(query.get("turn") or [None])[0], before=(query.get("before") or [None])[0],
                                       from_ms=_millis("from_ms"), to_ms=_millis("to_ms"),
                                       snapshot=(query.get("snapshot") or [None])[0], after=_millis("after"))
            except (KeyError, ValueError, OSError) as exc:
                self._send_json({"error":"trajectory_not_available", "detail":str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(view)
            return
        if parsed.path in {"/api/design-history", "/api/design-source"}:
            from .design_history import history_view, store_for
            query = parse_qs(parsed.query)
            def argument(key, default=None): return (query.get(key) or [default])[0]
            try:
                run_id = argument("run", "")
                run = self.server.resolve_run(run_id)
                if parsed.path.endswith("design-history"):
                    payload = history_view(self.server.workspace_root, run_id, run,
                        around=int(argument("around")) if argument("around") else None,
                        selected=argument("selected"), before=argument("before"),scope=argument("scope","revision"))
                else:
                    store = store_for(run)
                    snapshot = store.snapshot(argument("snapshot", ""))
                    if not snapshot or snapshot["revision"] != run.name: raise ValueError("Unknown revision snapshot")
                    payload = store.source_view(snapshot["id"], name=argument("file"), offset=int(argument("offset", 0)),across_revisions=argument("scope")=="project")
                self._send_json(payload)
            except (ValueError, KeyError, OSError) as exc:
                self._send_json({"error":"history_not_available","detail":str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/progress":
            run_id = (parse_qs(parsed.query).get("run") or [""])[0]
            try:
                progress = self.server.live_progress(run_id)
            except (KeyError, ValueError, OSError) as exc:
                self._send_json({"error": "progress_not_available", "detail": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(progress)
            return
        if parsed.path == "/api/preview-file":
            from .preview_service import preview_media_paths

            query = parse_qs(parsed.query)
            run_id = (query.get("run") or [""])[0]
            name = (query.get("name") or [""])[0]
            try:
                run_dir = self.server.resolve_run(run_id)
                if name.startswith("history/"):
                    from .design_history import read_model
                    path = read_model(self.server.workspace_root, run_dir, name[len("history/"):])
                else:
                    media = preview_media_paths(workspace_root=self.server.workspace_root, run_dir=run_dir)
                    path = Path(media[name])
                content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                self._send_bytes(path.read_bytes(), content_type=content_type)
            except (KeyError, ValueError, OSError):
                self._send_json({"error": "preview_file_not_available"}, status=HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/changes":
            query = parse_qs(parsed.query)
            run_values = query.get("run") or []
            since_values = query.get("since") or []
            timeout_values = query.get("timeout") or []
            if (
                len(run_values) != 1
                or len(since_values) > 1
                or len(timeout_values) > 1
            ):
                self._send_json(
                    {"error": "invalid_change_watch"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            run_id = run_values[0]
            try:
                timeout_s = (
                    float(timeout_values[0])
                    if timeout_values
                    else _CHANGE_DEFAULT_TIMEOUT_S
                )
            except ValueError:
                timeout_s = math.nan
            if (
                not math.isfinite(timeout_s)
                or timeout_s < 0
                or timeout_s > _CHANGE_MAX_TIMEOUT_S
            ):
                self._send_json(
                    {"error": "invalid_change_timeout"},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            try:
                if not since_values:
                    snapshot = self.server.dashboard_change_snapshot(run_id)
                    self._send_json(
                        {
                            "schema_version": _CHANGE_RESPONSE_SCHEMA,
                            "run": run_id,
                            **snapshot,
                            "changed": False,
                            "catalog_changed": False,
                            "run_changed": False,
                        }
                    )
                    return
                changed = self.server.wait_for_dashboard_change(
                    run_id=run_id,
                    since=since_values[0],
                    timeout_s=timeout_s,
                )
            except (KeyError, OSError) as exc:
                self._send_json(
                    {"error": "run_not_available", "detail": str(exc)},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            except ValueError as exc:
                self._send_json(
                    {"error": "invalid_change_watch", "detail": str(exc)},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if changed is None:
                previous = _token_parts(since_values[0])
                assert previous is not None
                snapshot = {
                    "catalog_token": previous[0],
                    "run_token": previous[1],
                    "token": since_values[0],
                }
                self._send_change_keepalive(snapshot)
                return
            snapshot, catalog_changed, run_changed = changed
            self._send_json(
                {
                    "schema_version": _CHANGE_RESPONSE_SCHEMA,
                    "run": run_id,
                    **snapshot,
                    "changed": True,
                    "catalog_changed": catalog_changed,
                    "run_changed": run_changed,
                }
            )
            return
        if parsed.path == "/api/file":
            query = parse_qs(parsed.query)
            run_id = (query.get("run") or [""])[0]
            relative_path = (query.get("path") or [""])[0]
            try:
                candidate, payload = self.server.read_media(
                    run_id=run_id,
                    relative_path=relative_path,
                )
            except (KeyError, ValueError, OSError) as exc:
                self._send_json(
                    {"error": "preview_not_available", "detail": str(exc)},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
            self._send_bytes(
                payload if payload is not None else candidate.read_bytes(),
                content_type=content_type,
            )
            return
        if parsed.path == "/api/revision-frame":
            run_id = (parse_qs(parsed.query).get("run") or [""])[0]
            try:
                candidate, payload = self.server.read_revision_frame(run_id)
            except (KeyError, ValueError, OSError) as exc:
                self._send_json(
                    {"error": "revision_frame_not_available", "detail": str(exc)},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_bytes(
                payload,
                content_type=(
                    mimetypes.guess_type(candidate.name)[0]
                    or "application/octet-stream"
                ),
            )
            return
        if parsed.path == "/api/revision-mesh":
            run_id = (parse_qs(parsed.query).get("run") or [""])[0]
            try:
                payload = self.server.read_revision_mesh(run_id)
            except (KeyError, ValueError, OSError) as exc:
                self._send_json(
                    {"error": "revision_mesh_not_available", "detail": str(exc)},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_bytes(payload, content_type="model/gltf-binary")
            return
        if parsed.path == "/api/revision-geometry":
            query = parse_qs(parsed.query)
            base_id = (query.get("base") or [""])[0]
            target_id = (query.get("target") or [""])[0]
            try:
                rows_by_id = {
                    str(item["id"]): item
                    for item in self.server.runs()
                }
                base_row = rows_by_id[base_id]
                target_row = rows_by_id[target_id]
                base_project = str(
                    base_row.get("project_id") or base_row.get("package_id") or ""
                )
                target_project = str(
                    target_row.get("project_id") or target_row.get("package_id") or ""
                )
                if not base_project or base_project != target_project:
                    raise ValueError("revision pair must belong to the same project")
                base_run = self.server.resolve_run(base_id)
                target_run = self.server.resolve_run(target_id)
                comparison = compare_revision_geometry(base_run, target_run)
                comparison["base"]["run_id"] = base_id
                comparison["target"]["run_id"] = target_id
            except (KeyError, ValueError, OSError) as exc:
                self._send_json(
                    {"error": "revision_geometry_not_available", "detail": str(exc)},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            self._send_json(comparison)
            return
        if parsed.path in {"/api/run", "/api/export"}:
            run_id = (parse_qs(parsed.query).get("id") or [""])[0]
            try:
                run_dir = self.server.resolve_run(run_id)
                data = normalize_run(
                    run_dir,
                    workspace_root=self.server.workspace_root,
                    mode="live" if parsed.path == "/api/run" else "snapshot",
                )
            except (KeyError, ValueError, OSError) as exc:
                self._send_json(
                    {"error": "run_not_available", "detail": str(exc)},
                    status=HTTPStatus.NOT_FOUND,
                )
                return
            if parsed.path == "/api/run":
                for key in ("live_workflow", "live_usage"):
                    if not data.get(key):
                        data[key] = None
                data["studio_preview"] = self.server.studio_preview(run_id, run_dir)
                self._send_json(data)
                return
            html = render_snapshot(data).encode("utf-8")
            filename = quote(f"{run_dir.name}-quest-dashboard.html")
            self._send_bytes(
                html,
                content_type="text/html; charset=utf-8",
                disposition=f"attachment; filename*=UTF-8''{filename}",
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _read_json_body(self, *, max_bytes: int = 256 * 1024) -> dict | None:
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > max_bytes:
            self._send_json(
                {"error": "invalid_request_body"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return None
        try:
            body = json.loads(self.rfile.read(content_length))
        except ValueError:
            self._send_json(
                {"error": "invalid_request_body"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return None
        if not isinstance(body, dict):
            self._send_json(
                {"error": "invalid_request_body"},
                status=HTTPStatus.BAD_REQUEST,
            )
            return None
        return body

    def do_POST(self) -> None:
        if self._dispatch_import('do_POST'):
            return
        parsed = self._request_url()
        if parsed is None:
            return
        if getattr(self.server, 'import_parent', None) is not None:
            self._send_json({'error': 'Imported snapshots are historical and cannot resume execution'},
                            status=HTTPStatus.CONFLICT)
            return
        if maintenance_routes is not None and maintenance_routes.handle_post(self, parsed):
            return
        if parsed.path in _LAUNCH_POST_ROUTES:
            if self._launch_enabled():
                self._launch_post(parsed)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        if parsed.path != "/api/revision-video":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 80 * 1024 * 1024:
            self._send_json(
                {"error": "invalid_video_size"},
                status=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            )
            return
        if not str(self.headers.get("Content-Type") or "").startswith("video/webm"):
            self._send_json(
                {"error": "unsupported_video_type"},
                status=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return
        ffmpeg = _ffmpeg_binary()
        if ffmpeg is None:
            self._send_json(
                {"error": "video_transcoder_not_available"},
                status=HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        payload = self.rfile.read(content_length)
        try:
            with tempfile.TemporaryDirectory(prefix="quest-revision-video-") as temporary:
                temporary_root = Path(temporary)
                source = temporary_root / "replay.webm"
                output = temporary_root / "replay.mp4"
                source.write_bytes(payload)
                completed = subprocess.run(
                    [
                        str(ffmpeg),
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(source),
                        "-vf",
                        "fps=30,format=yuv420p",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "20",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(output),
                    ],
                    check=False,
                    capture_output=True,
                    timeout=120,
                )
                if completed.returncode or not output.is_file():
                    raise ValueError("ffmpeg could not transcode the recording")
                transcoded = output.read_bytes()
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            self._send_json(
                {"error": "video_transcode_failed", "detail": str(exc)},
                status=HTTPStatus.UNPROCESSABLE_ENTITY,
            )
            return
        self._send_bytes(transcoded, content_type="video/mp4")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _ffmpeg_binary() -> Path | None:
    configured = str(os.environ.get("QUEST_FFMPEG_BIN") or "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(sys.executable).resolve().with_name("ffmpeg"),
        Path(found) if (found := shutil.which("ffmpeg")) else None,
    ]
    return next(
        (
            candidate
            for candidate in candidates
            if candidate is not None and candidate.is_file() and os.access(candidate, os.X_OK)
        ),
        None,
    )


def serve(
    *,
    workspace_root: Path,
    runs_root: Path,
    host: str,
    port: int,
    base_path: str = "",
    launch_enabled: bool = False,
    launch_limit: int = 4,
) -> None:
    server = DashboardServer(
        (host, port), workspace_root=workspace_root, runs_root=runs_root, base_path=base_path,
        launch_enabled=launch_enabled, launch_limit=launch_limit,
    )
    # Begin warming before the browser's first /api/runs request. This never
    # delays startup or exact /api/run and /api/file reads.
    server.runs()
    # A wildcard bind is opened in the browser through the loopback address.
    shown = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "[::1]"}.get(host, host)
    origin = f"http://{shown}:{port}{server.base_path}"
    if server.launch_enabled:
        print(f"New task: {origin}/launch#human={server.human_decision_capability}", flush=True)
    print(f"Dashboard: {origin}/", flush=True)
    monitor = f"/runs#human={server.human_decision_capability}" if server.launch_enabled else "/runs"
    print(f"Run monitor: {origin}{monitor}", flush=True)
    print(f"Dashboard with actions: {origin}/#human={server.human_decision_capability}", flush=True)
    if maintenance_routes is not None:
        print(f"Human maintenance: {origin}/maintenance#human={server.human_decision_capability}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
