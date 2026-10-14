"""Canonical project/revision storage for validation runs.

New runs live at::

    .runtime_runs/projects/<project-id>/runs/rNN/

The project directory owns stable project metadata.  Each revision directory
owns one immutable validation attempt and every artifact derived from it.
Legacy flat run directories remain readable, but this module never creates a
new one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_GOVERNED_REVISIONS = 150
RUN_SCHEMA = "simulation_ready_cad_run_v1"
TASK_PLANNING_POLICY = "required_before_attempt_v1"
PROJECTS_RELATIVE = Path(".runtime_runs") / "projects"
_PROJECT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_REVISION = re.compile(r"^r(0*[1-9][0-9]*)$")
NON_TERMINAL_STATUSES = frozenset(
    {
        "PENDING",
        # A signed queue invocation exists, but no exact worker process is
        # currently active.  PREPARING carries no heartbeat, PASS, publication,
        # or terminal authority.
        "PREPARING",
        "RUNNING",
        "CAPABILITY_PROPOSAL_DRAFT_REQUIRED",
        # A scoped iteration attempt has closed, but it deliberately has no
        # certificate or project-terminal authority.
        "ITERATION_SCOPE_COMPLETE",
        # Dashboard stages may truthfully report that their evidence exists,
        # but that is never a project/revision terminal verdict.
        "EVIDENCE_COMPLETE",
    }
)
CERTIFYING_TERMINAL_STATUSES = frozenset(
    {
        "KINEMATIC_CERTIFIED",
        "TASK_CERTIFIED",
        "WHOLE_MACHINE_TASK_PASSED",
    }
)
NON_CERTIFYING_TERMINAL_STATUSES = frozenset(
    {
        # Read-only compatibility for governance-v1 gap runs and the signed
        # terminal state reached after a v2 proposal validates.
        "CAPABILITY_REVIEW_REQUIRED",
        "NOT_CERTIFIED",
        "PHYSICAL_PREFLIGHT_FAILED",
        "WHOLE_MACHINE_TASK_FAILED",
    }
)
CANONICAL_TERMINAL_STATUSES = frozenset(
    CERTIFYING_TERMINAL_STATUSES | NON_CERTIFYING_TERMINAL_STATUSES
)
CANONICAL_RUN_STATUSES = frozenset(
    NON_TERMINAL_STATUSES | CANONICAL_TERMINAL_STATUSES
)


class RunLayoutError(ValueError):
    """Raised when a new run does not obey the canonical storage contract."""

    code = "RUN_LAYOUT_INVALID"


class IterationLimitReachedError(RunLayoutError):
    """A new governed revision would exceed the closed-loop budget."""

    code = "ITERATION_LIMIT_REACHED"


@dataclass(frozen=True)
class CanonicalRun:
    workspace_root: Path
    project_id: str
    revision: str
    project_dir: Path
    run_dir: Path

    @property
    def run_id(self) -> str:
        return f"{self.project_id}-{self.revision}"

    def to_dict(self) -> dict[str, str]:
        return {
            "schema_version": RUN_SCHEMA,
            "project_id": self.project_id,
            "revision": self.revision,
            "run_id": self.run_id,
            "project_dir": self.project_dir.relative_to(
                self.workspace_root
            ).as_posix(),
            "run_dir": self.run_dir.relative_to(
                self.workspace_root
            ).as_posix(),
        }


def reviewer_plugin_registry_path(project_dir: str | Path) -> Path:
    """Return the project-scoped registry shared by Loop and Harness."""

    return Path(project_dir) / "react_loop" / "reviewer_plugins" / "registry.json"


def validate_project_id(value: str) -> str:
    project_id = str(value).strip()
    if not _PROJECT_ID.fullmatch(project_id) or project_id in {
        ".",
        "..",
        "_diagnostics",
        "project_usage",
        "projects",
    }:
        raise RunLayoutError(f"invalid project id: {value!r}")
    return project_id


def validate_revision(value: str) -> str:
    revision = str(value).strip().lower()
    match = _REVISION.fullmatch(revision)
    if match is None or revision != f"r{int(match.group(1)):02d}":
        raise RunLayoutError(
            "revision must use the canonical zero-padded form r01, r02, ...: "
            f"{value!r}"
        )
    return revision


def canonical_run_path(
    workspace_root: str | Path,
    project_id: str,
    revision: str,
) -> Path:
    workspace = Path(workspace_root).resolve()
    project = validate_project_id(project_id)
    run_revision = validate_revision(revision)
    return workspace / PROJECTS_RELATIVE / project / "runs" / run_revision


def parse_canonical_run(
    workspace_root: str | Path,
    run_dir: str | Path,
) -> CanonicalRun | None:
    """Parse the exact five-part canonical relative path."""

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    try:
        relative = run.relative_to(workspace)
    except ValueError:
        return None
    parts = relative.parts
    if (
        len(parts) != 5
        or parts[0] != ".runtime_runs"
        or parts[1] != "projects"
        or parts[3] != "runs"
    ):
        return None
    try:
        project_id = validate_project_id(parts[2])
        revision = validate_revision(parts[4])
    except RunLayoutError:
        return None
    return CanonicalRun(
        workspace_root=workspace,
        project_id=project_id,
        revision=revision,
        project_dir=workspace / PROJECTS_RELATIVE / project_id,
        run_dir=run,
    )


def next_revision(project_dir: str | Path) -> str:
    runs_dir = Path(project_dir) / "runs"
    maximum = 0
    if runs_dir.is_dir():
        for candidate in runs_dir.iterdir():
            match = _REVISION.fullmatch(candidate.name)
            if candidate.is_dir() and match:
                maximum = max(maximum, int(match.group(1)))
    if maximum >= MAX_GOVERNED_REVISIONS:
        raise IterationLimitReachedError(
            "governed revision limit reached: no new revision may exceed "
            f"r{MAX_GOVERNED_REVISIONS:02d}"
        )
    return f"r{maximum + 1:02d}"


def is_terminal_status(value: Any) -> bool:
    """Return true only for a closed, harness-defined terminal status.

    Unknown strings used to be treated as terminal merely because they were
    not RUNNING.  That let a producer invent a PASS-like label and make a run
    immutable/discoverable.  Canonical status authority is now an allowlist.
    """

    return (
        isinstance(value, str)
        and value.strip().upper() in CANONICAL_TERMINAL_STATUSES
    )


def normalize_canonical_status(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunLayoutError(f"run status must be a non-empty string: {value!r}")
    normalized = value.strip().upper()
    if normalized not in CANONICAL_RUN_STATUSES:
        raise RunLayoutError(
            "unknown canonical run status; expected one of: "
            + ", ".join(sorted(CANONICAL_RUN_STATUSES))
            + f"; got {value!r}"
        )
    return normalized
