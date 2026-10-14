"""Harness-owned, project-adaptive workflow stage manifests.

Callers may register evidence-bearing custom stages, but they never provide a
stage verdict.  The dashboard publisher materializes the final manifest from
artifact-discovered stages plus every registered declaration and assigns the
only custom-stage state: ``EVIDENCE_COMPLETE``.

The filesystem is not an authority boundary.  These checks make the standard
Harness route deterministic, bounded, fail-closed, and auditable; protecting
the files from a process running as the same OS identity requires an external
identity or signer.
"""

from __future__ import annotations

from ..runtime.file_lock import lock_file, unlock_file
import hashlib
import json
import math
import os
import re
import stat
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..mechanism.io import canonical_sha256, relative_to_root
from ..run_layout import (
    CANONICAL_TERMINAL_STATUSES,
    NON_TERMINAL_STATUSES,
    CanonicalRun,
    parse_canonical_run,
)


STAGE_DECLARATION_SCHEMA = (
    "simulation_ready_cad_workflow_stage_declaration_v1"
)
STAGE_DECLARATION_REGISTRAR_ID = (
    "simulation_ready_cad_workflow_stage_registrar_v1"
)
STAGE_MANIFEST_SCHEMA = "simulation_ready_cad_workflow_stage_manifest_v1"
STAGE_MANIFEST_PUBLISHER_ID = (
    "simulation_ready_cad_workflow_stage_manifest_publisher_v1"
)
STAGE_MANIFEST_RELATIVE = (
    Path("dashboard") / "workflow_stage_manifest.json"
)
STAGE_DECLARATIONS_RELATIVE = (
    Path("dashboard") / "workflow_stage_declarations"
)

_MAX_JSON_BYTES = 2 * 1024 * 1024
_MAX_JSON_NESTING = 32
_MAX_DECLARATIONS = 256
_MAX_STAGES = 1024
_MAX_EVIDENCE_PER_STAGE = 128
_MAX_SCOPE_NODES = 512
_MAX_SCOPE_DEPTH = 12
_MAX_SCOPE_COLLECTION = 128
_MAX_SCOPE_STRING = 2048
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STAGE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_STAGE_KIND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SCOPE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_DECLARATION_FIELDS = frozenset(
    {
        "schema_version",
        "registrar_id",
        "certification_authority",
        "run_layout",
        "id",
        "kind",
        "label",
        "required",
        "depends_on",
        "scope",
        "evidence",
        "declaration_payload_sha256",
    }
)
_DECLARATION_EVIDENCE_FIELDS = frozenset(
    {"path", "sha256", "size_bytes"}
)
_STAGE_FIELDS = frozenset(
    {
        "id",
        "kind",
        "label",
        "order",
        "required",
        "depends_on",
        "scope",
        "status",
        "evidence",
    }
)
_STAGE_EVIDENCE_FIELDS = frozenset({"path", "sha256"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "publisher_id",
        "certification_authority",
        "run_layout",
        "automatic_stage_count",
        "custom_stage_count",
        "declaration_count",
        "declarations",
        "stage_count",
        "stages",
        "stages_canonical_sha256",
        "canonical_payload_sha256",
    }
)
_DECLARATION_RECORD_FIELDS = frozenset(
    {"stage_id", "path", "sha256", "size_bytes"}
)


class WorkflowStageManifestError(ValueError):
    """A custom declaration or materialized stage manifest is invalid."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> WorkflowStageManifestError:
    return WorkflowStageManifestError(code, message)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def stable_regular_file_record(
    path: str | Path,
    *,
    max_bytes: int | None = None,
    capture_bytes: bool = True,
    code: str = "DASHBOARD_STAGE_FILE_INVALID",
) -> tuple[dict[str, Any], bytes]:
    """Read and hash one unchanged regular non-symlink file."""

    source = Path(path)
    try:
        lexical = source.lstat()
    except OSError as exc:
        raise _fail(code, f"cannot inspect {source}: {exc}") from exc
    if stat.S_ISLNK(lexical.st_mode) or not stat.S_ISREG(lexical.st_mode):
        raise _fail(code, f"expected a regular non-symlink file: {source}")
    if max_bytes is not None and lexical.st_size > max_bytes:
        raise _fail(code, f"file exceeds {max_bytes} bytes: {source}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(source, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise _fail(code, f"expected a regular file: {source}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise _fail(code, f"file exceeds {max_bytes} bytes: {source}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if max_bytes is not None and total > max_bytes:
                raise _fail(code, f"file exceeds {max_bytes} bytes: {source}")
            digest.update(block)
            if capture_bytes:
                chunks.append(block)
        after = os.fstat(descriptor)
    except WorkflowStageManifestError:
        raise
    except OSError as exc:
        raise _fail(code, f"cannot read {source}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    try:
        final = source.lstat()
    except OSError as exc:
        raise _fail(code, f"cannot re-inspect {source}: {exc}") from exc
    if (
        stat.S_ISLNK(final.st_mode)
        or _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(final)
        or total != int(after.st_size)
    ):
        raise _fail(code, f"file changed while being read: {source}")
    return (
        {
            "sha256": digest.hexdigest(),
            "size_bytes": total,
        },
        b"".join(chunks) if capture_bytes else b"",
    )


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise _fail(
                "DASHBOARD_STAGE_JSON_DUPLICATE_KEY",
                f"duplicate JSON key: {key!r}",
            )
        value[key] = child
    return value


def _reject_json_constant(value: str) -> None:
    raise _fail(
        "DASHBOARD_STAGE_JSON_NONFINITE",
        f"non-finite JSON number: {value}",
    )


def _json_nesting(text: str) -> int:
    depth = 0
    maximum = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            maximum = max(maximum, depth)
        elif character in "]}":
            depth = max(0, depth - 1)
    return maximum


def _finite_tree(value: Any, *, label: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise _fail(
            "DASHBOARD_STAGE_JSON_NONFINITE",
            f"{label} contains a non-finite number",
        )
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, label=f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, label=f"{label}[{index}]")


def read_stable_json_object(
    path: str | Path,
    *,
    code: str = "DASHBOARD_STAGE_JSON_INVALID",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strictly parse one bounded JSON object and return its stable record."""

    record, raw = stable_regular_file_record(
        path,
        max_bytes=_MAX_JSON_BYTES,
        code=code,
    )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise _fail(code, f"invalid UTF-8 JSON in {path}: {exc}") from exc
    if _json_nesting(text) > _MAX_JSON_NESTING:
        raise _fail(
            code,
            f"JSON nesting exceeds {_MAX_JSON_NESTING}: {path}",
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except WorkflowStageManifestError:
        raise
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise _fail(code, f"invalid bounded JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _fail(code, f"JSON root must be an object: {path}")
    _finite_tree(value)
    return value, record


def parse_scope_json(text: str) -> dict[str, Any]:
    """Parse the CLI scope argument with the same closed JSON rules."""

    if not isinstance(text, str) or len(text.encode("utf-8")) > _MAX_JSON_BYTES:
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            "scope JSON is missing or too large",
        )
    if _json_nesting(text) > _MAX_JSON_NESTING:
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            "scope JSON nesting is too deep",
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_json_constant,
        )
    except WorkflowStageManifestError:
        raise
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            f"invalid scope JSON: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            "scope JSON must be an object",
        )
    _validate_scope(value)
    return value


def _canonical_context(
    workspace_root: str | Path,
    run_dir: str | Path,
) -> tuple[Path, Path, CanonicalRun]:
    workspace = Path(workspace_root).resolve()
    lexical_run = Path(run_dir)
    if not lexical_run.is_absolute():
        lexical_run = workspace / lexical_run
    if lexical_run.is_symlink():
        raise _fail(
            "DASHBOARD_STAGE_RUN_INVALID",
            "canonical run directory cannot be a symlink",
        )
    run = lexical_run.resolve()
    identity = parse_canonical_run(workspace, run)
    if identity is None:
        raise _fail(
            "DASHBOARD_STAGE_REQUIRES_CANONICAL_RUN",
            "workflow stages require .runtime_runs/projects/<id>/runs/rNN",
        )
    try:
        run_status = run.lstat()
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_RUN_INVALID",
            f"cannot inspect canonical run: {exc}",
        ) from exc
    if stat.S_ISLNK(run_status.st_mode) or not stat.S_ISDIR(
        run_status.st_mode
    ):
        raise _fail(
            "DASHBOARD_STAGE_RUN_INVALID",
            "canonical run must be a regular directory",
        )
    return workspace, run, identity


def _lock_path(identity: CanonicalRun) -> Path:
    return (
        identity.project_dir
        / ".dashboard_stage_locks"
        / f"{identity.revision}.lock"
    )


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    try:
        value = path.lstat()
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_DIRECTORY_INVALID",
            f"cannot inspect directory {path}: {exc}",
        ) from exc
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
        raise _fail(
            "DASHBOARD_STAGE_DIRECTORY_INVALID",
            f"expected a regular directory: {path}",
        )


@contextmanager
def _stage_lock(identity: CanonicalRun) -> Iterator[None]:
    lock = _lock_path(identity)
    _ensure_directory(lock.parent)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lock, flags, 0o600)
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_LOCK_INVALID",
            f"cannot open stage lock: {exc}",
        ) from exc
    try:
        if os.name == "nt":
            import msvcrt

            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            lock_file(descriptor)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                unlock_file(descriptor)
        finally:
            os.close(descriptor)


def _encode_json(value: Mapping[str, Any]) -> bytes:
    _finite_tree(value)
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json_exclusive(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    code: str = "DASHBOARD_STAGE_WRITE_FAILED",
) -> dict[str, Any]:
    """Create one JSON file exactly once without replacement."""

    destination = Path(path)
    _ensure_directory(destination.parent)
    payload = _encode_json(value)
    descriptor: int | None = None
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("JSON write made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
    except FileExistsError:
        raise
    except OSError as exc:
        raise _fail(code, f"cannot create {destination}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    record, raw = stable_regular_file_record(
        destination,
        max_bytes=_MAX_JSON_BYTES,
        code=code,
    )
    if raw != payload:
        raise _fail(code, f"created JSON bytes changed: {destination}")
    return record


def _safe_text(
    value: Any,
    *,
    label: str,
    maximum: int,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise _fail(
            "DASHBOARD_STAGE_FIELD_INVALID",
            f"{label} must be a non-empty, trimmed string",
        )
    if len(value) > maximum or any(
        ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise _fail(
            "DASHBOARD_STAGE_FIELD_INVALID",
            f"{label} contains unsafe or excessive text",
        )
    if pattern is not None and not pattern.fullmatch(value):
        raise _fail(
            "DASHBOARD_STAGE_FIELD_INVALID",
            f"{label} has an unsafe identifier: {value!r}",
        )
    return value


def _validate_stage_id(value: Any, *, label: str = "stage id") -> str:
    return _safe_text(
        value,
        label=label,
        maximum=128,
        pattern=_STAGE_ID,
    )


def _validate_stage_kind(value: Any) -> str:
    return _safe_text(
        value,
        label="stage kind",
        maximum=96,
        pattern=_STAGE_KIND,
    )


def _validate_label(value: Any) -> str:
    return _safe_text(value, label="stage label", maximum=256)


def _validate_scope(
    value: Any,
    *,
    depth: int = 0,
    state: list[int] | None = None,
) -> None:
    if state is None:
        state = [0]
    state[0] += 1
    if state[0] > _MAX_SCOPE_NODES or depth > _MAX_SCOPE_DEPTH:
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            "scope exceeds the bounded object contract",
        )
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _fail(
                "DASHBOARD_STAGE_SCOPE_INVALID",
                "scope contains a non-finite number",
            )
        return
    if isinstance(value, str):
        if len(value) > _MAX_SCOPE_STRING or any(
            ord(character) < 32 or ord(character) == 127
            for character in value
        ):
            raise _fail(
                "DASHBOARD_STAGE_SCOPE_INVALID",
                "scope contains unsafe or excessive text",
            )
        return
    if isinstance(value, list):
        if len(value) > _MAX_SCOPE_COLLECTION:
            raise _fail(
                "DASHBOARD_STAGE_SCOPE_INVALID",
                "scope list is too large",
            )
        for child in value:
            _validate_scope(child, depth=depth + 1, state=state)
        return
    if isinstance(value, dict):
        if len(value) > _MAX_SCOPE_COLLECTION:
            raise _fail(
                "DASHBOARD_STAGE_SCOPE_INVALID",
                "scope object is too large",
            )
        for key, child in value.items():
            if not isinstance(key, str) or not _SCOPE_KEY.fullmatch(key):
                raise _fail(
                    "DASHBOARD_STAGE_SCOPE_INVALID",
                    f"scope key is unsafe: {key!r}",
                )
            _validate_scope(child, depth=depth + 1, state=state)
        return
    raise _fail(
        "DASHBOARD_STAGE_SCOPE_INVALID",
        f"scope contains unsupported JSON type: {type(value).__name__}",
    )


def _normalized_run_reference(
    workspace: Path,
    run: Path,
    reference: Any,
) -> tuple[str, Path]:
    if (
        not isinstance(reference, str)
        or not reference
        or "\x00" in reference
        or "\\" in reference
    ):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_PATH_INVALID",
            f"invalid evidence path: {reference!r}",
        )
    pure = PurePosixPath(reference)
    if (
        pure.is_absolute()
        or "." in pure.parts
        or ".." in pure.parts
        or pure.as_posix() != reference
    ):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_PATH_INVALID",
            f"evidence path is not normalized: {reference!r}",
        )
    lexical = workspace.joinpath(*pure.parts)
    try:
        candidate = lexical.resolve(strict=True)
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_MISSING",
            f"cannot resolve evidence {reference}: {exc}",
        ) from exc
    try:
        candidate.relative_to(run)
    except ValueError as exc:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_OUTSIDE_RUN",
            f"evidence is outside the canonical run: {reference}",
        ) from exc
    normalized = relative_to_root(workspace, candidate)
    if normalized != reference or lexical.is_symlink():
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_PATH_INVALID",
            f"evidence path is aliased or not normalized: {reference}",
        )
    return normalized, candidate


def _evidence_record(
    workspace: Path,
    run: Path,
    path: str | Path,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_absolute():
        source = workspace / source
    try:
        candidate = source.resolve(strict=True)
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_MISSING",
            f"cannot resolve evidence {source}: {exc}",
        ) from exc
    try:
        candidate.relative_to(run)
    except ValueError as exc:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_OUTSIDE_RUN",
            f"evidence is outside the canonical run: {source}",
        ) from exc
    if source.is_symlink():
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_SYMLINK",
            f"evidence cannot be a symlink: {source}",
        )
    normalized = relative_to_root(workspace, candidate)
    normalized_again, verified = _normalized_run_reference(
        workspace,
        run,
        normalized,
    )
    record, _ = stable_regular_file_record(
        verified,
        capture_bytes=False,
        code="DASHBOARD_STAGE_EVIDENCE_INVALID",
    )
    return {
        "path": normalized_again,
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def _validate_evidence_binding(
    workspace: Path,
    run: Path,
    value: Any,
    *,
    include_size: bool,
) -> dict[str, Any]:
    fields = (
        _DECLARATION_EVIDENCE_FIELDS
        if include_size
        else _STAGE_EVIDENCE_FIELDS
    )
    if not isinstance(value, dict) or set(value) != fields:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_FIELDS_INVALID",
            "evidence binding has missing or extra fields",
        )
    reference, path = _normalized_run_reference(
        workspace,
        run,
        value.get("path"),
    )
    digest = value.get("sha256")
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_HASH_INVALID",
            f"invalid evidence hash for {reference}",
        )
    record, _ = stable_regular_file_record(
        path,
        capture_bytes=False,
        code="DASHBOARD_STAGE_EVIDENCE_INVALID",
    )
    if record["sha256"] != digest:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_HASH_MISMATCH",
            f"evidence changed after registration: {reference}",
        )
    if include_size:
        size = value.get("size_bytes")
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or size != record["size_bytes"]
        ):
            raise _fail(
                "DASHBOARD_STAGE_EVIDENCE_SIZE_MISMATCH",
                f"evidence size changed after registration: {reference}",
            )
    return {
        "path": reference,
        "sha256": digest,
        **(
            {"size_bytes": record["size_bytes"]}
            if include_size
            else {}
        ),
    }


def _declaration_path(run: Path, stage_id: str) -> Path:
    name = hashlib.sha256(stage_id.encode("utf-8")).hexdigest() + ".json"
    return run / STAGE_DECLARATIONS_RELATIVE / name


def _assert_run_not_terminal(run: Path) -> None:
    status_path = run / "run_status.json"
    if not status_path.exists() and not status_path.is_symlink():
        return
    status, _ = read_stable_json_object(
        status_path,
        code="DASHBOARD_STAGE_RUN_STATUS_INVALID",
    )
    recorded = status.get("status")
    normalized = (
        recorded.strip().upper()
        if isinstance(recorded, str) and recorded.strip()
        else ""
    )
    if normalized in CANONICAL_TERMINAL_STATUSES:
        raise _fail(
            "DASHBOARD_STAGE_RUN_TERMINAL",
            f"terminal run cannot be changed: {normalized}",
        )
    if normalized not in NON_TERMINAL_STATUSES:
        raise _fail(
            "DASHBOARD_STAGE_RUN_STATUS_INVALID",
            f"run status is outside the closed enum: {recorded!r}",
        )


def assert_run_open_for_dashboard_publication(
    workspace_root: str | Path,
    run_dir: str | Path,
) -> tuple[Path, Path, CanonicalRun]:
    """Return canonical context only while the recorded run is non-terminal."""

    workspace, run, identity = _canonical_context(
        workspace_root,
        run_dir,
    )
    _assert_run_not_terminal(run)
    return workspace, run, identity


def _assert_registration_open(run: Path) -> None:
    _assert_run_not_terminal(run)
    for path in (
        run / STAGE_MANIFEST_RELATIVE,
        run / "dashboard" / "publication.json",
    ):
        if path.exists() or path.is_symlink():
            raise _fail(
                "DASHBOARD_STAGE_REGISTRY_SEALED",
                "stage registration is sealed once materialization starts",
            )


def _declaration_unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: child
        for key, child in value.items()
        if key != "declaration_payload_sha256"
    }


def _validate_declaration(
    value: Any,
    *,
    workspace: Path,
    run: Path,
    identity: CanonicalRun,
    declaration_path: Path,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _DECLARATION_FIELDS:
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_FIELDS_INVALID",
            "stage declaration has missing or extra fields; callers cannot "
            "supply a status or terminal verdict",
        )
    if (
        value.get("schema_version") != STAGE_DECLARATION_SCHEMA
        or value.get("registrar_id") != STAGE_DECLARATION_REGISTRAR_ID
        or value.get("certification_authority") is not False
        or value.get("run_layout") != identity.to_dict()
    ):
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_IDENTITY_INVALID",
            "stage declaration is not bound to this canonical run/registrar",
        )
    stage_id = _validate_stage_id(value.get("id"))
    kind = _validate_stage_kind(value.get("kind"))
    label = _validate_label(value.get("label"))
    if declaration_path.name != _declaration_path(run, stage_id).name:
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_PATH_INVALID",
            "declaration filename does not match its unique stage id",
        )
    required = value.get("required")
    if not isinstance(required, bool):
        raise _fail(
            "DASHBOARD_STAGE_FIELD_INVALID",
            "required must be a JSON boolean",
        )
    depends_on = value.get("depends_on")
    if (
        not isinstance(depends_on, list)
        or len(depends_on) > _MAX_STAGES
        or any(not isinstance(item, str) for item in depends_on)
    ):
        raise _fail(
            "DASHBOARD_STAGE_DEPENDENCY_INVALID",
            "depends_on must be a bounded string array",
        )
    normalized_dependencies = [
        _validate_stage_id(item, label="dependency id")
        for item in depends_on
    ]
    if (
        len(set(normalized_dependencies)) != len(normalized_dependencies)
        or stage_id in normalized_dependencies
    ):
        raise _fail(
            "DASHBOARD_STAGE_DEPENDENCY_INVALID",
            "dependencies cannot be duplicated or self-referential",
        )
    scope = value.get("scope")
    if not isinstance(scope, dict):
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            "scope must be a JSON object",
        )
    _validate_scope(scope)
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, list)
        or not evidence
        or len(evidence) > _MAX_EVIDENCE_PER_STAGE
    ):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_MISSING",
            "custom stages require at least one bounded evidence binding",
        )
    normalized_evidence = [
        _validate_evidence_binding(
            workspace,
            run,
            item,
            include_size=True,
        )
        for item in evidence
    ]
    paths = [item["path"] for item in normalized_evidence]
    if len(set(paths)) != len(paths):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_DUPLICATE",
            "custom stage evidence paths must be unique",
        )
    digest = value.get("declaration_payload_sha256")
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or digest != canonical_sha256(_declaration_unsigned(value))
    ):
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_HASH_INVALID",
            "declaration canonical payload hash is invalid",
        )
    return {
        **value,
        "id": stage_id,
        "kind": kind,
        "label": label,
        "depends_on": normalized_dependencies,
        "scope": scope,
        "evidence": normalized_evidence,
    }


def _read_declarations(
    *,
    workspace: Path,
    run: Path,
    identity: CanonicalRun,
) -> list[tuple[dict[str, Any], dict[str, Any], Path]]:
    root = run / STAGE_DECLARATIONS_RELATIVE
    if not root.exists() and not root.is_symlink():
        return []
    try:
        root_status = root.lstat()
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_DIRECTORY_INVALID",
            f"cannot inspect declaration directory: {exc}",
        ) from exc
    if stat.S_ISLNK(root_status.st_mode) or not stat.S_ISDIR(
        root_status.st_mode
    ):
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_DIRECTORY_INVALID",
            "declaration root must be a regular non-symlink directory",
        )
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_DIRECTORY_INVALID",
            f"cannot scan declaration directory: {exc}",
        ) from exc
    if len(entries) > _MAX_DECLARATIONS:
        raise _fail(
            "DASHBOARD_STAGE_DECLARATION_LIMIT",
            f"more than {_MAX_DECLARATIONS} declarations are not allowed",
        )
    declarations: list[
        tuple[dict[str, Any], dict[str, Any], Path]
    ] = []
    seen: set[str] = set()
    for path in entries:
        try:
            item_status = path.lstat()
        except OSError as exc:
            raise _fail(
                "DASHBOARD_STAGE_DECLARATION_FILE_INVALID",
                f"cannot inspect declaration {path}: {exc}",
            ) from exc
        if (
            path.suffix != ".json"
            or stat.S_ISLNK(item_status.st_mode)
            or not stat.S_ISREG(item_status.st_mode)
        ):
            raise _fail(
                "DASHBOARD_STAGE_DECLARATION_EXTRA_ENTRY",
                f"unexpected declaration entry: {path.name}",
            )
        value, record = read_stable_json_object(
            path,
            code="DASHBOARD_STAGE_DECLARATION_JSON_INVALID",
        )
        declaration = _validate_declaration(
            value,
            workspace=workspace,
            run=run,
            identity=identity,
            declaration_path=path,
        )
        stage_id = declaration["id"]
        if stage_id in seen:
            raise _fail(
                "DASHBOARD_STAGE_ID_DUPLICATE",
                f"duplicate declared stage id: {stage_id}",
            )
        seen.add(stage_id)
        declarations.append((declaration, record, path))
    return sorted(declarations, key=lambda item: item[0]["id"])


def register_workflow_stage(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    stage_id: str,
    kind: str,
    label: str,
    evidence_paths: Iterable[str | Path],
    required: bool = True,
    depends_on: Sequence[str] = (),
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Register one evidence-bearing custom stage through the standard route.

    There is deliberately no ``status`` argument.  During materialization the
    Harness assigns ``EVIDENCE_COMPLETE`` only after rehashing every declared
    evidence file.
    """

    workspace, run, identity = _canonical_context(
        workspace_root,
        run_dir,
    )
    normalized_id = _validate_stage_id(stage_id)
    normalized_kind = _validate_stage_kind(kind)
    normalized_label = _validate_label(label)
    if not isinstance(required, bool):
        raise _fail(
            "DASHBOARD_STAGE_FIELD_INVALID",
            "required must be a boolean",
        )
    dependencies = [
        _validate_stage_id(value, label="dependency id")
        for value in depends_on
    ]
    if (
        len(dependencies) > _MAX_STAGES
        or len(set(dependencies)) != len(dependencies)
        or normalized_id in dependencies
    ):
        raise _fail(
            "DASHBOARD_STAGE_DEPENDENCY_INVALID",
            "dependencies are duplicated, self-referential, or excessive",
        )
    normalized_scope = dict(scope or {})
    _validate_scope(normalized_scope)
    if isinstance(evidence_paths, (str, bytes)):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_MISSING",
            "evidence_paths must be an iterable of paths, not a string",
        )
    paths = list(evidence_paths)
    if not paths or len(paths) > _MAX_EVIDENCE_PER_STAGE:
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_MISSING",
            "custom stages require at least one evidence file",
        )
    with _stage_lock(identity):
        _assert_registration_open(run)
        # A malformed or duplicate pre-existing declaration closes the route.
        _read_declarations(
            workspace=workspace,
            run=run,
            identity=identity,
        )
        evidence = [
            _evidence_record(workspace, run, path)
            for path in paths
        ]
        evidence_references = [item["path"] for item in evidence]
        if len(set(evidence_references)) != len(evidence_references):
            raise _fail(
                "DASHBOARD_STAGE_EVIDENCE_DUPLICATE",
                "custom stage evidence paths must be unique",
            )
        unsigned: dict[str, Any] = {
            "schema_version": STAGE_DECLARATION_SCHEMA,
            "registrar_id": STAGE_DECLARATION_REGISTRAR_ID,
            "certification_authority": False,
            "run_layout": identity.to_dict(),
            "id": normalized_id,
            "kind": normalized_kind,
            "label": normalized_label,
            "required": required,
            "depends_on": dependencies,
            "scope": normalized_scope,
            "evidence": evidence,
        }
        declaration = {
            **unsigned,
            "declaration_payload_sha256": canonical_sha256(unsigned),
        }
        destination = _declaration_path(run, normalized_id)
        try:
            record = write_json_exclusive(
                destination,
                declaration,
                code="DASHBOARD_STAGE_DECLARATION_WRITE_FAILED",
            )
        except FileExistsError as exc:
            raise _fail(
                "DASHBOARD_STAGE_REGISTRATION_REPLAY",
                f"stage id is already registered: {normalized_id}",
            ) from exc
        loaded, loaded_record = read_stable_json_object(
            destination,
            code="DASHBOARD_STAGE_DECLARATION_JSON_INVALID",
        )
        _validate_declaration(
            loaded,
            workspace=workspace,
            run=run,
            identity=identity,
            declaration_path=destination,
        )
        if loaded_record != record:
            raise _fail(
                "DASHBOARD_STAGE_DECLARATION_WRITE_FAILED",
                "declaration changed immediately after registration",
            )
        return {
            "schema_version": STAGE_DECLARATION_SCHEMA,
            "path": relative_to_root(workspace, destination),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
            "stage_id": normalized_id,
            "kind": normalized_kind,
            "required": required,
            "certification_authority": False,
        }


def _validate_stage(
    value: Any,
    *,
    workspace: Path,
    run: Path,
    expected_order: int,
    earlier_ids: set[str],
    all_ids: set[str],
    custom_declaration: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _STAGE_FIELDS:
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_STAGE_FIELDS_INVALID",
            "materialized stage has missing or extra fields",
        )
    stage_id = _validate_stage_id(value.get("id"))
    kind = _validate_stage_kind(value.get("kind"))
    label = _validate_label(value.get("label"))
    order = value.get("order")
    required = value.get("required")
    if (
        not isinstance(order, int)
        or isinstance(order, bool)
        or order != expected_order
        or not isinstance(required, bool)
    ):
        raise _fail(
            "DASHBOARD_STAGE_ORDER_INVALID",
            f"stage {stage_id} has invalid order/required fields",
        )
    dependencies = value.get("depends_on")
    if (
        not isinstance(dependencies, list)
        or any(not isinstance(item, str) for item in dependencies)
        or len(dependencies) > _MAX_STAGES
    ):
        raise _fail(
            "DASHBOARD_STAGE_DEPENDENCY_INVALID",
            f"stage {stage_id} has invalid dependencies",
        )
    normalized_dependencies = [
        _validate_stage_id(item, label="dependency id")
        for item in dependencies
    ]
    if (
        len(set(normalized_dependencies)) != len(normalized_dependencies)
        or any(item not in all_ids for item in normalized_dependencies)
        or any(item not in earlier_ids for item in normalized_dependencies)
    ):
        raise _fail(
            "DASHBOARD_STAGE_DEPENDENCY_INVALID",
            f"stage {stage_id} dependencies are missing, duplicated, cyclic, "
            "or out of order",
        )
    scope = value.get("scope")
    if not isinstance(scope, dict):
        raise _fail(
            "DASHBOARD_STAGE_SCOPE_INVALID",
            f"stage {stage_id} scope must be an object",
        )
    _validate_scope(scope)
    status_value = value.get("status")
    status_text = _safe_text(
        status_value,
        label="stage status",
        maximum=128,
    )
    evidence = value.get("evidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) > _MAX_EVIDENCE_PER_STAGE
    ):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_INVALID",
            f"stage {stage_id} evidence must be a bounded array",
        )
    normalized_evidence = [
        _validate_evidence_binding(
            workspace,
            run,
            item,
            include_size=False,
        )
        for item in evidence
    ]
    paths = [item["path"] for item in normalized_evidence]
    if len(set(paths)) != len(paths):
        raise _fail(
            "DASHBOARD_STAGE_EVIDENCE_DUPLICATE",
            f"stage {stage_id} contains duplicate evidence",
        )
    if custom_declaration is not None:
        expected_evidence = [
            {
                "path": item["path"],
                "sha256": item["sha256"],
            }
            for item in custom_declaration["evidence"]
        ]
        if (
            status_text != "EVIDENCE_COMPLETE"
            or not normalized_evidence
            or kind != custom_declaration["kind"]
            or label != custom_declaration["label"]
            or required is not custom_declaration["required"]
            or normalized_dependencies
            != custom_declaration["depends_on"]
            or scope != custom_declaration["scope"]
            or normalized_evidence != expected_evidence
        ):
            raise _fail(
                "DASHBOARD_CUSTOM_STAGE_REBUILD_MISMATCH",
                f"custom stage does not exactly rebuild declaration: {stage_id}",
            )
    return {
        "id": stage_id,
        "kind": kind,
        "label": label,
        "order": order,
        "required": required,
        "depends_on": normalized_dependencies,
        "scope": scope,
        "status": status_text,
        "evidence": normalized_evidence,
    }


def _normalize_automatic_stages(
    stages: Sequence[Mapping[str, Any]],
    *,
    workspace: Path,
    run: Path,
) -> list[dict[str, Any]]:
    if not isinstance(stages, Sequence) or isinstance(
        stages, (str, bytes)
    ):
        raise _fail(
            "DASHBOARD_AUTOMATIC_STAGES_INVALID",
            "artifact-derived stages must be an array",
        )
    if not stages or len(stages) > _MAX_STAGES:
        raise _fail(
            "DASHBOARD_AUTOMATIC_STAGES_INVALID",
            "artifact-derived stages must contain a bounded truthful route",
        )
    stage_ids = [
        _validate_stage_id(
            stage.get("id") if isinstance(stage, Mapping) else None
        )
        for stage in stages
    ]
    if len(set(stage_ids)) != len(stage_ids):
        raise _fail(
            "DASHBOARD_STAGE_ID_DUPLICATE",
            "artifact-derived stage IDs are not unique",
        )
    all_ids = set(stage_ids)
    earlier: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        item = _validate_stage(
            dict(stage),
            workspace=workspace,
            run=run,
            expected_order=index,
            earlier_ids=earlier,
            all_ids=all_ids,
            custom_declaration=None,
        )
        normalized.append(item)
        earlier.add(item["id"])
    return normalized


def _merge_stages(
    *,
    workspace: Path,
    run: Path,
    automatic_stages: Sequence[Mapping[str, Any]],
    declarations: list[
        tuple[dict[str, Any], dict[str, Any], Path]
    ],
) -> list[dict[str, Any]]:
    merged = _normalize_automatic_stages(
        automatic_stages,
        workspace=workspace,
        run=run,
    )
    automatic_ids = {stage["id"] for stage in merged}
    declaration_by_id = {
        declaration["id"]: declaration
        for declaration, _record, _path in declarations
    }
    custom_ids = set(declaration_by_id)
    collision = automatic_ids & custom_ids
    if collision:
        raise _fail(
            "DASHBOARD_STAGE_ID_COLLISION",
            "custom declarations cannot delete or override automatically "
            f"discovered stages: {', '.join(sorted(collision))}",
        )
    all_ids = automatic_ids | custom_ids
    for declaration in declaration_by_id.values():
        missing = set(declaration["depends_on"]) - all_ids
        if missing:
            raise _fail(
                "DASHBOARD_STAGE_DEPENDENCY_MISSING",
                f"stage {declaration['id']} depends on unknown stages: "
                + ", ".join(sorted(missing)),
            )
    remaining = dict(declaration_by_id)
    earlier = set(automatic_ids)
    while remaining:
        ready = sorted(
            stage_id
            for stage_id, declaration in remaining.items()
            if set(declaration["depends_on"]) <= earlier
        )
        if not ready:
            raise _fail(
                "DASHBOARD_STAGE_DEPENDENCY_CYCLE",
                "custom stage dependencies contain a cycle",
            )
        for stage_id in ready:
            declaration = remaining.pop(stage_id)
            merged.append(
                {
                    "id": stage_id,
                    "kind": declaration["kind"],
                    "label": declaration["label"],
                    "order": len(merged) + 1,
                    "required": declaration["required"],
                    "depends_on": declaration["depends_on"],
                    "scope": declaration["scope"],
                    "status": "EVIDENCE_COMPLETE",
                    "evidence": [
                        {
                            "path": item["path"],
                            "sha256": item["sha256"],
                        }
                        for item in declaration["evidence"]
                    ],
                }
            )
            earlier.add(stage_id)
    return merged


def _declaration_records(
    declarations: list[
        tuple[dict[str, Any], dict[str, Any], Path]
    ],
    *,
    workspace: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": declaration["id"],
            "path": relative_to_root(workspace, path),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for declaration, record, path in declarations
    ]


def _manifest_unsigned(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: child
        for key, child in value.items()
        if key != "canonical_payload_sha256"
    }


def _build_manifest(
    *,
    identity: CanonicalRun,
    automatic_stage_count: int,
    stages: list[dict[str, Any]],
    declaration_records: list[dict[str, Any]],
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": STAGE_MANIFEST_SCHEMA,
        "publisher_id": STAGE_MANIFEST_PUBLISHER_ID,
        "certification_authority": False,
        "run_layout": identity.to_dict(),
        "automatic_stage_count": automatic_stage_count,
        "custom_stage_count": len(declaration_records),
        "declaration_count": len(declaration_records),
        "declarations": declaration_records,
        "stage_count": len(stages),
        "stages": stages,
        "stages_canonical_sha256": canonical_sha256(stages),
    }
    return {
        **unsigned,
        "canonical_payload_sha256": canonical_sha256(unsigned),
    }


def _manifest_binding(
    *,
    workspace: Path,
    path: Path,
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": relative_to_root(workspace, path),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
        "canonical_payload_sha256": manifest[
            "canonical_payload_sha256"
        ],
    }


def _validate_manifest(
    value: Any,
    *,
    workspace: Path,
    run: Path,
    identity: CanonicalRun,
    declarations: list[
        tuple[dict[str, Any], dict[str, Any], Path]
    ],
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _MANIFEST_FIELDS:
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_FIELDS_INVALID",
            "workflow stage manifest has missing or extra fields",
        )
    if (
        value.get("schema_version") != STAGE_MANIFEST_SCHEMA
        or value.get("publisher_id") != STAGE_MANIFEST_PUBLISHER_ID
        or value.get("certification_authority") is not False
        or value.get("run_layout") != identity.to_dict()
    ):
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_IDENTITY_INVALID",
            "workflow stage manifest identity is invalid",
        )
    digest = value.get("canonical_payload_sha256")
    if (
        not isinstance(digest, str)
        or not _SHA256.fullmatch(digest)
        or digest != canonical_sha256(_manifest_unsigned(value))
    ):
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_HASH_INVALID",
            "workflow stage manifest canonical hash is invalid",
        )
    expected_declarations = _declaration_records(
        declarations,
        workspace=workspace,
    )
    declaration_records = value.get("declarations")
    if (
        not isinstance(declaration_records, list)
        or any(
            not isinstance(item, dict)
            or set(item) != _DECLARATION_RECORD_FIELDS
            for item in declaration_records
        )
        or declaration_records != expected_declarations
        or value.get("declaration_count") != len(expected_declarations)
        or value.get("custom_stage_count") != len(expected_declarations)
    ):
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_DECLARATIONS_INVALID",
            "manifest does not bind the exact declaration set",
        )
    stages = value.get("stages")
    stage_count = value.get("stage_count")
    automatic_count = value.get("automatic_stage_count")
    if (
        not isinstance(stages, list)
        or not stages
        or len(stages) > _MAX_STAGES
        or not isinstance(stage_count, int)
        or isinstance(stage_count, bool)
        or stage_count != len(stages)
        or not isinstance(automatic_count, int)
        or isinstance(automatic_count, bool)
        or automatic_count < 1
        or automatic_count + len(expected_declarations) != len(stages)
        or value.get("stages_canonical_sha256")
        != canonical_sha256(stages)
    ):
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_COUNT_INVALID",
            "manifest stage counts or canonical stage hash are invalid",
        )
    stage_ids = [
        _validate_stage_id(
            stage.get("id") if isinstance(stage, dict) else None
        )
        for stage in stages
    ]
    if len(set(stage_ids)) != len(stage_ids):
        raise _fail(
            "DASHBOARD_STAGE_ID_DUPLICATE",
            "materialized stage IDs are not unique",
        )
    all_ids = set(stage_ids)
    custom_by_id = {
        declaration["id"]: declaration
        for declaration, _record, _path in declarations
    }
    custom_positions = {
        stage_id: index
        for index, stage_id in enumerate(stage_ids)
        if stage_id in custom_by_id
    }
    if (
        set(custom_positions) != set(custom_by_id)
        or any(index < automatic_count for index in custom_positions.values())
    ):
        raise _fail(
            "DASHBOARD_STAGE_MANIFEST_CUSTOM_SET_INVALID",
            "manifest does not contain the exact custom stage set after "
            "artifact-derived stages",
        )
    earlier: set[str] = set()
    normalized_stages: list[dict[str, Any]] = []
    for index, stage in enumerate(stages, start=1):
        stage_id = stage_ids[index - 1]
        normalized = _validate_stage(
            stage,
            workspace=workspace,
            run=run,
            expected_order=index,
            earlier_ids=earlier,
            all_ids=all_ids,
            custom_declaration=custom_by_id.get(stage_id),
        )
        normalized_stages.append(normalized)
        earlier.add(stage_id)
    return {
        **value,
        "stages": normalized_stages,
    }


def load_workflow_stage_manifest(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and fully revalidate the run's formal workflow stage manifest."""

    workspace, run, identity = _canonical_context(
        workspace_root,
        run_dir,
    )
    declarations = _read_declarations(
        workspace=workspace,
        run=run,
        identity=identity,
    )
    path = run / STAGE_MANIFEST_RELATIVE
    value, record = read_stable_json_object(
        path,
        code="DASHBOARD_STAGE_MANIFEST_JSON_INVALID",
    )
    manifest = _validate_manifest(
        value,
        workspace=workspace,
        run=run,
        identity=identity,
        declarations=declarations,
    )
    return manifest, _manifest_binding(
        workspace=workspace,
        path=path,
        record=record,
        manifest=manifest,
    )


def materialize_workflow_stage_manifest(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    automatic_stages: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Seal declarations and materialize the exact project-specific route."""

    workspace, run, identity = assert_run_open_for_dashboard_publication(
        workspace_root,
        run_dir,
    )
    path = run / STAGE_MANIFEST_RELATIVE
    with _stage_lock(identity):
        _assert_run_not_terminal(run)
        declarations = _read_declarations(
            workspace=workspace,
            run=run,
            identity=identity,
        )
        stages = _merge_stages(
            workspace=workspace,
            run=run,
            automatic_stages=automatic_stages,
            declarations=declarations,
        )
        declaration_records = _declaration_records(
            declarations,
            workspace=workspace,
        )
        expected = _build_manifest(
            identity=identity,
            automatic_stage_count=len(automatic_stages),
            stages=stages,
            declaration_records=declaration_records,
        )
        if path.exists() or path.is_symlink():
            existing, _record = read_stable_json_object(
                path,
                code="DASHBOARD_STAGE_MANIFEST_JSON_INVALID",
            )
            validated = _validate_manifest(
                existing,
                workspace=workspace,
                run=run,
                identity=identity,
                declarations=declarations,
            )
            if validated != expected:
                raise _fail(
                    "DASHBOARD_STAGE_MANIFEST_REBUILD_MISMATCH",
                    "existing manifest does not equal freshly rebuilt truth",
                )
        else:
            try:
                write_json_exclusive(
                    path,
                    expected,
                    code="DASHBOARD_STAGE_MANIFEST_WRITE_FAILED",
                )
            except FileExistsError as exc:
                raise _fail(
                    "DASHBOARD_STAGE_MANIFEST_CONCURRENT_WRITE",
                    "manifest appeared outside the standard finalizer lock",
                ) from exc
        manifest, binding = load_workflow_stage_manifest(
            workspace_root=workspace,
            run_dir=run,
        )
        if manifest != expected:
            raise _fail(
                "DASHBOARD_STAGE_MANIFEST_REBUILD_MISMATCH",
                "materialized manifest changed before publication",
            )
        return manifest, binding


__all__ = [
    "STAGE_DECLARATION_REGISTRAR_ID",
    "STAGE_DECLARATION_SCHEMA",
    "STAGE_DECLARATIONS_RELATIVE",
    "STAGE_MANIFEST_PUBLISHER_ID",
    "STAGE_MANIFEST_RELATIVE",
    "STAGE_MANIFEST_SCHEMA",
    "WorkflowStageManifestError",
    "assert_run_open_for_dashboard_publication",
    "load_workflow_stage_manifest",
    "materialize_workflow_stage_manifest",
    "parse_scope_json",
    "read_stable_json_object",
    "register_workflow_stage",
    "stable_regular_file_record",
    "write_json_exclusive",
]
