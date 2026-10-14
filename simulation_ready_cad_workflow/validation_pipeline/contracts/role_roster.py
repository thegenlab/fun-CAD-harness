"""Human-visible Loop topology contract.

A roster is either the two-role mechanical Loop (worker plus project
resource reviewer) or the four-role Loop Loop that also binds a Harness
operator and Harness-change reviewer.  The immutable roster binds every
listed role to the same project input, starting Harness commit, prompt
bytes, authority surface, and independently visible agent session.
Mutable liveness observations live in a separate runtime document bound
back to the immutable roster hash.

This module deliberately contains no agent launcher, message transport, or
Loop state mutation.  It is the small fail-closed contract used by those
outer layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from ..mechanism.io import (
    JsonDocumentError,
    canonical_sha256,
    read_json,
    write_json_atomic,
)
from .exclusive_io import write_new_json


ROLE_ROSTER_SCHEMA = "simulation_ready_cad_role_roster_v1"
ROLE_RUNTIME_SCHEMA = "simulation_ready_cad_role_runtime_v1"

WORKER_ROLE = "worker"
PROJECT_RESOURCE_REVIEWER_ROLE = "project_resource_reviewer"
HARNESS_OPERATOR_ROLE = "harness_operator"
HARNESS_CHANGE_REVIEWER_ROLE = "harness_change_reviewer"

MANDATORY_ROLES = (
    WORKER_ROLE,
    PROJECT_RESOURCE_REVIEWER_ROLE,
    HARNESS_OPERATOR_ROLE,
    HARNESS_CHANGE_REVIEWER_ROLE,
)
MECHANICAL_LOOP_ROLES = (WORKER_ROLE, PROJECT_RESOURCE_REVIEWER_ROLE)
LOOP_LOOP_TOPOLOGY = "loop_loop"
MECHANICAL_LOOP_TOPOLOGY = "mechanical_loop"

AUTHORITY_SURFACE_BY_ROLE = MappingProxyType(
    {
        WORKER_ROLE: "worker_loop",
        PROJECT_RESOURCE_REVIEWER_ROLE: "loop_review",
        HARNESS_OPERATOR_ROLE: "loop_loop",
        HARNESS_CHANGE_REVIEWER_ROLE: "harness_change_review",
    }
)

RESOURCE_REVIEW_MODES = frozenset({"HUMAN", "AI"})
DEFAULT_RESOURCE_REVIEW_MODE = "HUMAN"

ROLE_RUNTIME_STATES = frozenset(
    {
        "ACTIVE",
        "READY",
        "DORMANT",
        "UNKNOWN",
        "STALE",
        "UNREACHABLE",
    }
)
UNAVAILABLE_ROLE_STATES = frozenset(
    {"UNKNOWN", "STALE", "UNREACHABLE"}
)

_DEFAULT_STATE_BY_ROLE = MappingProxyType(
    {
        # Registration never authorizes work.  Every role, including the
        # Worker, starts asleep and becomes ACTIVE only for one exact Harness
        # activation.
        WORKER_ROLE: "DORMANT",
        PROJECT_RESOURCE_REVIEWER_ROLE: "DORMANT",
        HARNESS_OPERATOR_ROLE: "DORMANT",
        HARNESS_CHANGE_REVIEWER_ROLE: "DORMANT",
    }
)
_DEFAULT_ACTIVITY_BY_ROLE = MappingProxyType(
    {
        WORKER_ROLE: "awaiting_harness_interrupt",
        PROJECT_RESOURCE_REVIEWER_ROLE: "human_review_mode",
        HARNESS_OPERATOR_ROLE: "awaiting_harness_interrupt",
        HARNESS_CHANGE_REVIEWER_ROLE: "awaiting_harness_change_proposal",
    }
)

_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_MAX_TEXT_CHARS = 4096
_MAX_PROMPT_BYTES = 4 * 1024 * 1024

_REGISTRATION_KEYS = frozenset(
    {
        "provider",
        "model",
        "visibility_surface",
        "visible_session_ref",
        "capability_receipt_sha256",
        "prompt_binding",
    }
)
_REGISTRATION_KEYS_WITH_REASONING = _REGISTRATION_KEYS | frozenset(
    {"reasoning_effort"}
)
_ROLE_RECORD_KEYS = _REGISTRATION_KEYS | frozenset(
    {
        "role_id",
        "project_id",
        "requirements_sha256",
        "starting_commit",
        "authority_surface",
    }
)
_ROLE_RECORD_KEYS_WITH_REASONING = _REGISTRATION_KEYS_WITH_REASONING | frozenset(
    {
        "role_id",
        "project_id",
        "requirements_sha256",
        "starting_commit",
        "authority_surface",
    }
)
_ROSTER_KEYS = frozenset(
    {
        "schema_version",
        "project_id",
        "requirements_sha256",
        "starting_commit",
        "created_at",
        "roles",
        "roster_sha256",
    }
)
_OBSERVATION_KEYS = frozenset({"state", "observed_at", "activity"})
_RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "roster_sha256",
        "project_id",
        "requirements_sha256",
        "starting_commit",
        "resource_review_mode",
        "roles",
        "updated_at",
        "runtime_sha256",
    }
)


class RoleRosterError(RuntimeError):
    """A role topology document is malformed, stale, or unavailable."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def required_role_ids(role_map: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the frozen role order for a two-role Loop or four-role Loop Loop."""

    keys = frozenset(role_map)
    if keys == frozenset(MECHANICAL_LOOP_ROLES):
        return MECHANICAL_LOOP_ROLES
    if keys == frozenset(MANDATORY_ROLES):
        return MANDATORY_ROLES
    raise RoleRosterError(
        "ROLE_SET_INVALID",
        "roles must be worker plus project_resource_reviewer, "
        "or the four-role Loop Loop set",
    )


def topology_id_for_roles(role_ids: Sequence[str]) -> str:
    """Return the prompt/control topology id for a frozen role order."""

    if tuple(role_ids) == MECHANICAL_LOOP_ROLES:
        return MECHANICAL_LOOP_TOPOLOGY
    if tuple(role_ids) == MANDATORY_ROLES:
        return LOOP_LOOP_TOPOLOGY
    raise RoleRosterError(
        "ROLE_SET_INVALID",
        "roles must be worker plus project_resource_reviewer, "
        "or the four-role Loop Loop set",
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def role_roster_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "maintenance" / "role_roster.json"


def role_runtime_path(project_dir: str | Path) -> Path:
    return Path(project_dir) / "maintenance" / "role_runtime.json"


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RoleRosterError("ROLE_TOPOLOGY_INVALID", f"{label} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise RoleRosterError(
            "ROLE_TOPOLOGY_INVALID",
            f"{label} fields must be exactly {sorted(expected)}; "
            f"found {sorted(str(item) for item in actual)}",
        )


def _require_text(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_TEXT_CHARS
        or "\x00" in value
    ):
        raise RoleRosterError(
            "ROLE_TOPOLOGY_INVALID",
            f"{label} must be a bounded, non-empty, trimmed string",
        )
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise RoleRosterError(
            "ROLE_TOPOLOGY_INVALID",
            f"{label} must be a lowercase SHA-256 digest",
        )
    return value


def _require_starting_commit(value: Any) -> str:
    if not isinstance(value, str) or _GIT_COMMIT.fullmatch(value) is None:
        raise RoleRosterError(
            "ROLE_TOPOLOGY_INVALID",
            "starting_commit must be a full lowercase Git object ID",
        )
    return value


def _require_timestamp(value: Any, label: str) -> str:
    text = _require_text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise RoleRosterError(
            "ROLE_TOPOLOGY_INVALID", f"{label} is not an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise RoleRosterError(
            "ROLE_TOPOLOGY_INVALID", f"{label} must include a timezone"
        )
    return text


def _prompt_source_path(workspace_root: str | Path, source: Any) -> Path:
    source_text = _require_text(source, "prompt_binding.source")
    relative = PurePosixPath(source_text)
    if (
        relative.is_absolute()
        or "\\" in source_text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise RoleRosterError(
            "ROLE_PROMPT_PATH_INVALID",
            "prompt source must be a safe workspace-relative POSIX path",
        )
    root = Path(workspace_root).resolve()
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise RoleRosterError(
                "ROLE_PROMPT_UNREADABLE", f"cannot inspect prompt source {source_text}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RoleRosterError(
                "ROLE_PROMPT_PATH_INVALID",
                f"prompt source contains a symlink: {source_text}",
            )
    return current


def _read_prompt_bytes(workspace_root: str | Path, source: Any) -> bytes:
    path = _prompt_source_path(workspace_root, source)
    flags = os.O_RDONLY | (getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RoleRosterError(
            "ROLE_PROMPT_UNREADABLE", f"cannot open prompt source {source!r}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RoleRosterError(
                "ROLE_PROMPT_PATH_INVALID", "prompt source must be a regular file"
            )
        if before.st_size <= 0 or before.st_size > _MAX_PROMPT_BYTES:
            raise RoleRosterError(
                "ROLE_PROMPT_SIZE_INVALID",
                f"prompt source must contain 1..{_MAX_PROMPT_BYTES} bytes",
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_size,
        before.st_mtime_ns,
        getattr(before, "st_ino", None),
    )
    identity_after = (
        after.st_size,
        after.st_mtime_ns,
        getattr(after, "st_ino", None),
    )
    if identity_before != identity_after or len(payload) != before.st_size:
        raise RoleRosterError(
            "ROLE_PROMPT_CHANGED_DURING_READ", "prompt source changed while read"
        )
    try:
        payload.decode("utf-8")
    except UnicodeError as error:
        raise RoleRosterError(
            "ROLE_PROMPT_ENCODING_INVALID", "prompt source must be UTF-8 text"
        ) from error
    return payload


def prompt_binding_from_source(
    workspace_root: str | Path, source: str
) -> dict[str, Any]:
    """Return an exact size/hash binding for one checked prompt source."""

    payload = _read_prompt_bytes(workspace_root, source)
    return {
        "source": source,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_prompt_binding(
    binding: Any, *, workspace_root: str | Path, role_id: str
) -> None:
    value = _require_mapping(binding, f"roles.{role_id}.prompt_binding")
    _require_exact_keys(
        value,
        frozenset({"source", "size_bytes", "sha256"}),
        f"roles.{role_id}.prompt_binding",
    )
    source = _require_text(
        value.get("source"), f"roles.{role_id}.prompt_binding.source"
    )
    size = value.get("size_bytes")
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or size > _MAX_PROMPT_BYTES
    ):
        raise RoleRosterError(
            "ROLE_PROMPT_SIZE_INVALID",
            f"roles.{role_id}.prompt_binding.size_bytes is invalid",
        )
    digest = _require_sha256(
        value.get("sha256"), f"roles.{role_id}.prompt_binding.sha256"
    )
    actual = _read_prompt_bytes(workspace_root, source)
    if len(actual) != size or hashlib.sha256(actual).hexdigest() != digest:
        raise RoleRosterError(
            "ROLE_PROMPT_BINDING_CHANGED",
            f"prompt bytes for role {role_id!r} do not match the roster",
        )


def build_role_registration(
    *,
    provider: str,
    model: str,
    visibility_surface: str,
    visible_session_ref: str,
    capability_receipt_sha256: str,
    prompt_binding: Mapping[str, Any],
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    """Build the role-specific portion consumed by :func:`build_role_roster`."""

    registration = {
        "provider": provider,
        "model": model,
        "visibility_surface": visibility_surface,
        "visible_session_ref": visible_session_ref,
        "capability_receipt_sha256": capability_receipt_sha256,
        "prompt_binding": dict(prompt_binding),
    }
    if reasoning_effort is not None:
        registration["reasoning_effort"] = reasoning_effort
    return registration


def build_role_roster(
    *,
    workspace_root: str | Path,
    project_id: str,
    requirements_sha256: str,
    starting_commit: str,
    registrations: Mapping[str, Mapping[str, Any]],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build and validate one immutable four-role roster."""

    project = _require_text(project_id, "project_id")
    requirements = _require_sha256(requirements_sha256, "requirements_sha256")
    commit = _require_starting_commit(starting_commit)
    registration_map = _require_mapping(registrations, "registrations")
    role_ids = required_role_ids(registration_map)
    roles: dict[str, dict[str, Any]] = {}
    for role_id in role_ids:
        registration = _require_mapping(
            registration_map[role_id], f"registrations.{role_id}"
        )
        if frozenset(registration) not in {
            _REGISTRATION_KEYS,
            _REGISTRATION_KEYS_WITH_REASONING,
        }:
            raise RoleRosterError(
                "ROLE_TOPOLOGY_INVALID",
                f"registrations.{role_id} fields are invalid",
            )
        roles[role_id] = {
            "role_id": role_id,
            "project_id": project,
            "requirements_sha256": requirements,
            "starting_commit": commit,
            "authority_surface": AUTHORITY_SURFACE_BY_ROLE[role_id],
            **dict(registration),
        }
    core: dict[str, Any] = {
        "schema_version": ROLE_ROSTER_SCHEMA,
        "project_id": project,
        "requirements_sha256": requirements,
        "starting_commit": commit,
        "created_at": created_at or _utc_now(),
        "roles": roles,
    }
    document = {**core, "roster_sha256": canonical_sha256(core)}
    validate_role_roster(document, workspace_root=workspace_root)
    return document


def validate_role_roster(
    document: Mapping[str, Any], *, workspace_root: str | Path
) -> dict[str, Any]:
    """Validate roster shape, scope, prompt bytes, sessions, and hash."""

    roster = _require_mapping(document, "role roster")
    _require_exact_keys(roster, _ROSTER_KEYS, "role roster")
    if roster.get("schema_version") != ROLE_ROSTER_SCHEMA:
        raise RoleRosterError(
            "ROLE_ROSTER_SCHEMA_MISMATCH",
            f"expected {ROLE_ROSTER_SCHEMA!r}",
        )
    project = _require_text(roster.get("project_id"), "project_id")
    requirements = _require_sha256(
        roster.get("requirements_sha256"), "requirements_sha256"
    )
    commit = _require_starting_commit(roster.get("starting_commit"))
    _require_timestamp(roster.get("created_at"), "created_at")
    roles = _require_mapping(roster.get("roles"), "roles")
    role_ids = required_role_ids(roles)
    session_refs: list[str] = []
    for role_id in role_ids:
        role = _require_mapping(roles[role_id], f"roles.{role_id}")
        if frozenset(role) not in {
            _ROLE_RECORD_KEYS,
            _ROLE_RECORD_KEYS_WITH_REASONING,
        }:
            raise RoleRosterError(
                "ROLE_TOPOLOGY_INVALID", f"roles.{role_id} fields are invalid"
            )
        expected_scope = {
            "role_id": role_id,
            "project_id": project,
            "requirements_sha256": requirements,
            "starting_commit": commit,
            "authority_surface": AUTHORITY_SURFACE_BY_ROLE[role_id],
        }
        for field, expected in expected_scope.items():
            if role.get(field) != expected:
                raise RoleRosterError(
                    "ROLE_SCOPE_MISMATCH",
                    f"roles.{role_id}.{field} must be {expected!r}",
                )
        for field in ("provider", "model", "visibility_surface"):
            _require_text(role.get(field), f"roles.{role_id}.{field}")
        if "reasoning_effort" in role:
            _require_text(
                role.get("reasoning_effort"),
                f"roles.{role_id}.reasoning_effort",
            )
        session_refs.append(
            _require_text(
                role.get("visible_session_ref"),
                f"roles.{role_id}.visible_session_ref",
            )
        )
        _require_sha256(
            role.get("capability_receipt_sha256"),
            f"roles.{role_id}.capability_receipt_sha256",
        )
        _validate_prompt_binding(
            role.get("prompt_binding"),
            workspace_root=workspace_root,
            role_id=role_id,
        )
    if len(set(session_refs)) != len(role_ids):
        raise RoleRosterError(
            "ROLE_VISIBLE_SESSION_REUSED",
            "every listed role requires a non-empty, distinct visible session",
        )
    expected_digest = canonical_sha256(
        {key: value for key, value in roster.items() if key != "roster_sha256"}
    )
    supplied_digest = _require_sha256(roster.get("roster_sha256"), "roster_sha256")
    if supplied_digest != expected_digest:
        raise RoleRosterError(
            "ROLE_ROSTER_HASH_MISMATCH", "immutable roster hash does not match"
        )
    return dict(roster)


def _default_observations(
    *, observed_at: str, mode: str, role_ids: Sequence[str]
) -> dict[str, Any]:
    states = dict(_DEFAULT_STATE_BY_ROLE)
    activities = dict(_DEFAULT_ACTIVITY_BY_ROLE)
    if mode == "AI":
        # AI mode makes resource-review interrupts routable; it does not wake
        # the reviewer before a concrete request exists.
        activities[PROJECT_RESOURCE_REVIEWER_ROLE] = "awaiting_resource_request"
    return {
        role_id: {
            "state": states[role_id],
            "observed_at": observed_at,
            "activity": activities[role_id],
        }
        for role_id in role_ids
    }


def build_role_runtime(
    *,
    roster: Mapping[str, Any],
    workspace_root: str | Path,
    resource_review_mode: str = DEFAULT_RESOURCE_REVIEW_MODE,
    observations: Mapping[str, Mapping[str, Any]] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build one roster-bound runtime observation snapshot."""

    checked_roster = validate_role_roster(roster, workspace_root=workspace_root)
    timestamp = updated_at or _utc_now()
    mode = str(resource_review_mode)
    if mode not in RESOURCE_REVIEW_MODES:
        raise RoleRosterError(
            "RESOURCE_REVIEW_MODE_INVALID",
            f"resource review mode must be one of {sorted(RESOURCE_REVIEW_MODES)}",
        )
    role_observations: Mapping[str, Mapping[str, Any]]
    if observations is None:
        role_observations = _default_observations(
            observed_at=timestamp,
            mode=mode,
            role_ids=required_role_ids(checked_roster["roles"]),
        )
    else:
        role_observations = observations
    core: dict[str, Any] = {
        "schema_version": ROLE_RUNTIME_SCHEMA,
        "roster_sha256": checked_roster["roster_sha256"],
        "project_id": checked_roster["project_id"],
        "requirements_sha256": checked_roster["requirements_sha256"],
        "starting_commit": checked_roster["starting_commit"],
        "resource_review_mode": mode,
        "roles": {
            role_id: dict(role_observations[role_id])
            for role_id in role_observations
        },
        "updated_at": timestamp,
    }
    document = {**core, "runtime_sha256": canonical_sha256(core)}
    validate_role_runtime(
        document, roster=checked_roster, workspace_root=workspace_root
    )
    return document


def validate_role_runtime(
    document: Mapping[str, Any],
    *,
    roster: Mapping[str, Any],
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Validate a runtime snapshot without hiding unavailable states."""

    checked_roster = validate_role_roster(roster, workspace_root=workspace_root)
    runtime = _require_mapping(document, "role runtime")
    _require_exact_keys(runtime, _RUNTIME_KEYS, "role runtime")
    if runtime.get("schema_version") != ROLE_RUNTIME_SCHEMA:
        raise RoleRosterError(
            "ROLE_RUNTIME_SCHEMA_MISMATCH",
            f"expected {ROLE_RUNTIME_SCHEMA!r}",
        )
    expected_bindings = {
        "roster_sha256": checked_roster["roster_sha256"],
        "project_id": checked_roster["project_id"],
        "requirements_sha256": checked_roster["requirements_sha256"],
        "starting_commit": checked_roster["starting_commit"],
    }
    for field, expected in expected_bindings.items():
        if runtime.get(field) != expected:
            raise RoleRosterError(
                "ROLE_RUNTIME_SCOPE_MISMATCH",
                f"runtime {field} is not bound to the immutable roster",
            )
    mode = runtime.get("resource_review_mode")
    if mode not in RESOURCE_REVIEW_MODES:
        raise RoleRosterError(
            "RESOURCE_REVIEW_MODE_INVALID",
            f"resource review mode must be one of {sorted(RESOURCE_REVIEW_MODES)}",
        )
    _require_timestamp(runtime.get("updated_at"), "updated_at")
    roles = _require_mapping(runtime.get("roles"), "runtime.roles")
    role_ids = required_role_ids(checked_roster["roles"])
    if frozenset(roles) != frozenset(role_ids):
        raise RoleRosterError(
            "ROLE_SET_INVALID",
            "runtime roles must match the immutable roster",
        )
    for role_id in role_ids:
        observation = _require_mapping(
            roles[role_id], f"runtime.roles.{role_id}"
        )
        _require_exact_keys(
            observation, _OBSERVATION_KEYS, f"runtime.roles.{role_id}"
        )
        state = observation.get("state")
        if state not in ROLE_RUNTIME_STATES:
            raise RoleRosterError(
                "ROLE_RUNTIME_STATE_INVALID",
                f"runtime.roles.{role_id}.state must be one of "
                f"{sorted(ROLE_RUNTIME_STATES)}",
            )
        _require_timestamp(
            observation.get("observed_at"),
            f"runtime.roles.{role_id}.observed_at",
        )
        _require_text(
            observation.get("activity"), f"runtime.roles.{role_id}.activity"
        )
    reviewer_state = roles[PROJECT_RESOURCE_REVIEWER_ROLE].get("state")
    reviewer_activity = str(
        roles[PROJECT_RESOURCE_REVIEWER_ROLE].get("activity") or ""
    )
    if (
        mode == "HUMAN"
        and reviewer_state != "DORMANT"
        and not reviewer_activity.startswith(
            ("plugin_implementation:", "static_visual_review:")
        )
    ):
        raise RoleRosterError(
            "RESOURCE_REVIEWER_NOT_DORMANT",
            "project resource reviewer must be DORMANT in HUMAN mode",
        )
    expected_digest = canonical_sha256(
        {key: value for key, value in runtime.items() if key != "runtime_sha256"}
    )
    supplied_digest = _require_sha256(
        runtime.get("runtime_sha256"), "runtime_sha256"
    )
    if supplied_digest != expected_digest:
        raise RoleRosterError(
            "ROLE_RUNTIME_HASH_MISMATCH", "runtime hash does not match"
        )
    return dict(runtime)


def assert_role_topology_ready(
    *,
    roster: Mapping[str, Any],
    runtime: Mapping[str, Any],
    workspace_root: str | Path,
) -> None:
    """Fail closed unless every mandatory visible session is reachable."""

    checked = validate_role_runtime(
        runtime, roster=roster, workspace_root=workspace_root
    )
    unavailable = {
        role_id: checked["roles"][role_id]["state"]
        for role_id in required_role_ids(roster["roles"])
        if checked["roles"][role_id]["state"] in UNAVAILABLE_ROLE_STATES
    }
    if unavailable:
        detail = ", ".join(
            f"{role_id}={state}" for role_id, state in unavailable.items()
        )
        raise RoleRosterError(
            "ROLE_TOPOLOGY_UNAVAILABLE",
            f"mandatory visible role probes are not usable: {detail}",
        )


def load_role_roster(
    project_dir: str | Path, *, workspace_root: str | Path
) -> dict[str, Any]:
    path = role_roster_path(project_dir)
    if not path.is_file():
        raise RoleRosterError("ROLE_ROSTER_MISSING", f"no role roster at {path}")
    try:
        document = read_json(path)
    except (OSError, JsonDocumentError) as error:
        raise RoleRosterError(
            "ROLE_ROSTER_UNREADABLE", f"cannot read role roster at {path}"
        ) from error
    return validate_role_roster(document, workspace_root=workspace_root)


def load_role_runtime(
    project_dir: str | Path,
    *,
    roster: Mapping[str, Any],
    workspace_root: str | Path,
) -> dict[str, Any]:
    path = role_runtime_path(project_dir)
    if not path.is_file():
        raise RoleRosterError("ROLE_RUNTIME_MISSING", f"no role runtime at {path}")
    try:
        document = read_json(path)
    except (OSError, JsonDocumentError) as error:
        raise RoleRosterError(
            "ROLE_RUNTIME_UNREADABLE", f"cannot read role runtime at {path}"
        ) from error
    return validate_role_runtime(
        document, roster=roster, workspace_root=workspace_root
    )


def write_role_roster(
    project_dir: str | Path,
    document: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> Path:
    """Publish an immutable roster; only a byte-identical retry is allowed."""

    checked = validate_role_roster(document, workspace_root=workspace_root)
    path = role_roster_path(project_dir)
    try:
        write_new_json(path, checked, allow_identical=True)
    except FileExistsError as error:
        raise RoleRosterError(
            "ROLE_ROSTER_IMMUTABLE",
            f"a different immutable role roster already exists at {path}",
        ) from error
    return path


def write_role_runtime(
    project_dir: str | Path,
    document: Mapping[str, Any],
    *,
    roster: Mapping[str, Any],
    workspace_root: str | Path,
    expected_runtime_sha256: str | None = None,
) -> Path:
    """Atomically publish runtime state with an optimistic update guard."""

    checked = validate_role_runtime(
        document, roster=roster, workspace_root=workspace_root
    )
    path = role_runtime_path(project_dir)
    if path.exists():
        if expected_runtime_sha256 is None:
            raise RoleRosterError(
                "ROLE_RUNTIME_EXPECTED_HASH_REQUIRED",
                "updating role runtime requires its previously observed hash",
            )
        _require_sha256(expected_runtime_sha256, "expected_runtime_sha256")
        previous = load_role_runtime(
            project_dir, roster=roster, workspace_root=workspace_root
        )
        if previous["runtime_sha256"] != expected_runtime_sha256:
            raise RoleRosterError(
                "ROLE_RUNTIME_CONFLICT",
                "role runtime changed after it was observed",
            )
    elif expected_runtime_sha256 is not None:
        raise RoleRosterError(
            "ROLE_RUNTIME_CONFLICT",
            "expected an existing role runtime, but none is present",
        )
    write_json_atomic(path, checked)
    return path
