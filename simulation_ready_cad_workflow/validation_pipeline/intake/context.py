from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, NoReturn

from .models import IntakeReport, Severity


ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
PART_ROLES = {"fixed", "dynamic", "context", "hardware", "unknown"}
COVERAGE_CLASSES = {"static", "capsule_member", "environment", "unclassified"}
WRENCH_TYPES = {"force", "torque"}
REDUCED_MODEL_FAMILIES = {
    "prismatic",
    "revolute",
    "cylindrical",
    "helical",
    "spherical",
    "planar",
    "compound",
    "state_machine",
    "custom",
}
WINDOWS_RESERVED_NAMES = {
    "AUX",
    "CLOCK$",
    "CON",
    "CONIN$",
    "CONOUT$",
    "NUL",
    "PRN",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}
WINDOWS_FORBIDDEN_PATH_CHARACTERS = set('<>:"|?*')


FORBIDDEN_AUTHORITY_KEYS = {
    "verdict",
    "pass_fail",
    "certified",
    "validator_thresholds",
    "hidden_tests",
    "final_status",
}
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_JSON_NESTING = 128
MAX_LIST_ITEMS = 10_000


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        super().__init__(f"Duplicate JSON object key: {key}")
        self.key = key


class _NonFiniteJsonNumberError(ValueError):
    def __init__(self, token: str) -> None:
        super().__init__(f"Non-finite JSON number: {token}")
        self.token = token if len(token) <= 128 else f"{token[:125]}..."


def _strict_json_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise _NonFiniteJsonNumberError(token)
    return value


def _strict_json_int(token: str) -> int:
    if len(token.lstrip("-")) > 309:
        raise _NonFiniteJsonNumberError(token)
    value = int(token)
    try:
        finite = math.isfinite(float(value))
    except OverflowError:
        finite = False
    if not finite:
        raise _NonFiniteJsonNumberError(token)
    return value


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(key)
        result[key] = value
    return result


def _reject_nonfinite_json_number(token: str) -> NoReturn:
    raise _NonFiniteJsonNumberError(token)


def _max_json_nesting(text: str) -> int:
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
        elif character in "]}" and depth:
            depth -= 1
    return maximum


def _is_windows_reserved_name(value: str) -> bool:
    stem = value.rstrip(" .").split(".", 1)[0].upper()
    return stem in WINDOWS_RESERVED_NAMES


def _normalized_json_key(value: str) -> str:
    with_word_boundaries = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        value,
    )
    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        with_word_boundaries,
    ).strip("_").casefold()


def _is_path_junction(path: Path) -> bool:
    checker = getattr(path, "is_junction", None)
    if checker is None:
        return False
    try:
        return bool(checker())
    except OSError:
        return True


def _file_identity(stat: Any) -> tuple[int, int, int | None]:
    return (
        int(stat.st_size),
        int(stat.st_mtime_ns),
        getattr(stat, "st_ino", None),
    )


class ValidationContext:
    def __init__(
        self,
        package_dir: str | Path,
        requirements_path: str | Path | None,
    ) -> None:
        self.package_dir = Path(package_dir).expanduser().resolve()
        self.requirements_path = (
            Path(requirements_path).expanduser().resolve()
            if requirements_path is not None
            else None
        )
        self.report = IntakeReport(
            package_dir=str(self.package_dir),
            requirements_file=(
                str(self.requirements_path)
                if self.requirements_path is not None
                else None
            ),
        )
        self.manifest: dict[str, Any] | None = None
        self.requirements: dict[str, Any] | None = None
        self.assembly: dict[str, Any] | None = None
        self.coverage: dict[str, Any] | None = None
        self.task_mapping: dict[str, Any] | None = None
        self.acceptance_contract: dict[str, Any] | None = None
        self.semantic_binding: dict[str, Any] | None = None
        self.capsules: dict[str, dict[str, Any]] = {}
        self.capsule_files: dict[str, Path] = {}
        self.part_ids: set[str] = set()
        self.part_roles: dict[str, str] = {}
        self.part_groups: dict[str, str] = {}
        self.group_ids: set[str] = set()
        self.frame_ids: set[str] = set()
        self.frame_groups: dict[str, str | None] = {}
        self.requirement_task_ids: set[str] = set()
        self.capsule_port_ids: dict[str, set[str]] = {}
        self.capsule_observable_ids: dict[str, set[str]] = {}
        self.reduced_ready_capsules: set[str] = set()
        self.step_geometry_verified = False
        self.step_solid_ids: set[str] = set()
        self.step_solid_inventory: dict[str, dict[str, Any]] = {}
        self.required_artifact_paths: set[str] = set()
        self.declared_artifact_paths: set[str] = set()
        self.package_file_paths: set[str] = set()
        self.package_directory_paths: set[str] = set()
        self.hashed_file_identities: dict[
            Path,
            tuple[tuple[int, int, int | None], str],
        ] = {}
        self.loaded_document_identities: dict[
            Path,
            tuple[tuple[int, int, int | None], str],
        ] = {}
        self.paths: dict[str, Path | None] = {}

    def error(self, code: str, message: str, **kwargs: Any) -> None:
        self.report.add(code, Severity.ERROR, message, **kwargs)

    def warning(self, code: str, message: str, **kwargs: Any) -> None:
        self.report.add(code, Severity.WARNING, message, **kwargs)

    def info(self, code: str, message: str, **kwargs: Any) -> None:
        self.report.add(code, Severity.INFO, message, **kwargs)

    def run_phase(self, name: str, callback: Callable[[], None]) -> None:
        before_errors = self.report.total_error_count
        before_warnings = self.report.total_warning_count
        note = None
        try:
            callback()
        except Exception as exc:  # Validator bugs must become visible evidence.
            self.error(
                "INTERNAL_VALIDATOR_ERROR",
                f"Validator phase {name!r} raised an unexpected exception.",
                actual=f"{type(exc).__name__}: {exc}",
                hint="Report this as a validator bug; do not repair the design package around it.",
            )
            note = f"{type(exc).__name__}: {exc}"
        errors_added = self.report.total_error_count - before_errors
        warnings_added = self.report.total_warning_count - before_warnings
        status = "FAIL" if errors_added else ("WARN" if warnings_added else "PASS")
        self.report.add_check(
            name,
            status,
            errors_added=errors_added,
            warnings_added=warnings_added,
            note=note,
        )

    def relative_label(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            # Labels are diagnostic metadata, so deriving one must not perform
            # another filesystem lookup. In particular, callers use this
            # while reporting files that disappeared or changed between two
            # deliberately bounded stat calls.
            absolute_path = (
                path if path.is_absolute() else Path.cwd() / path
            )
            return absolute_path.relative_to(self.package_dir).as_posix()
        except (OSError, RuntimeError, ValueError):
            return str(path)

    def safe_package_path(
        self,
        value: Any,
        *,
        manifest_field: str,
        fallback: str | None = None,
        diagnostic_file: str = "package.json",
        diagnostic_json_path: str | None = None,
    ) -> Path | None:
        json_path = diagnostic_json_path or f"$.{manifest_field}"
        relative = value if isinstance(value, str) and value.strip() else fallback
        if relative is None:
            self.error(
                "MANIFEST_FIELD_MISSING",
                f"package.json is missing {manifest_field}.",
                file=diagnostic_file,
                json_path=json_path,
                expected="non-empty relative path",
                actual=value,
            )
            return None
        if "\x00" in relative:
            self.error(
                "PACKAGE_PATH_INVALID",
                "Package artifact path contains a NUL character.",
                file=diagnostic_file,
                json_path=json_path,
                actual=relative,
            )
            return None
        candidate = Path(relative)
        if candidate.is_absolute() or candidate.drive:
            self.error(
                "PACKAGE_PATH_ABSOLUTE",
                "Package artifact paths must be relative to the package root.",
                file=diagnostic_file,
                json_path=json_path,
                expected="relative path",
                actual=relative,
            )
            return None
        if ".." in candidate.parts:
            self.error(
                "PACKAGE_PATH_TRAVERSAL",
                "Package artifact path contains a parent-directory segment.",
                file=diagnostic_file,
                json_path=json_path,
                expected="normalized path contained by package directory",
                actual=relative,
                hint="Remove '..' segments and use the canonical package path.",
            )
            return None
        unsafe_part = next(
            (
                part
                for part in candidate.parts
                if part not in {"", "."}
                and (
                    any(
                        character in WINDOWS_FORBIDDEN_PATH_CHARACTERS
                        for character in part
                    )
                    or part.endswith((" ", "."))
                    or _is_windows_reserved_name(part)
                )
            ),
            None,
        )
        if unsafe_part is not None:
            self.error(
                "PACKAGE_PATH_WINDOWS_UNSAFE",
                "Package artifact path contains a Windows-unsafe component.",
                file=diagnostic_file,
                json_path=json_path,
                expected="portable regular-file path",
                actual=unsafe_part,
                hint=(
                    "Remove device names, trailing spaces or dots, NTFS "
                    "alternate-data-stream colons, and reserved characters."
                ),
            )
            return None
        try:
            resolved = (self.package_dir / candidate).resolve()
        except (OSError, RuntimeError) as exc:
            self.error(
                "PACKAGE_PATH_INVALID",
                "Package artifact path could not be resolved safely.",
                file=diagnostic_file,
                json_path=json_path,
                actual=f"{type(exc).__name__}: {exc}",
            )
            return None
        try:
            resolved.relative_to(self.package_dir)
        except ValueError:
            self.error(
                "PACKAGE_PATH_TRAVERSAL",
                "Package artifact path escapes the package root.",
                file=diagnostic_file,
                json_path=json_path,
                expected="path contained by package directory",
                actual=relative,
                hint="Move the artifact into the package and use a relative path.",
            )
            return None
        current = self.package_dir
        for part in candidate.parts:
            if part in ("", "."):
                continue
            current = current / part
            if current.is_symlink():
                self.error(
                    "PACKAGE_SYMLINK_FORBIDDEN",
                    "Package artifact paths may not traverse symbolic links.",
                    file=diagnostic_file,
                    json_path=json_path,
                    actual=relative,
                    hint="Replace the link with a real file inside the package.",
                )
                return None
            if _is_path_junction(current):
                self.error(
                    "PACKAGE_JUNCTION_FORBIDDEN",
                    "Package artifact paths may not traverse directory junctions.",
                    file=diagnostic_file,
                    json_path=json_path,
                    actual=relative,
                    hint="Replace the junction with a real directory inside the package.",
                )
                return None
        return resolved

    def require_file(
        self,
        path: Path | None,
        *,
        code: str,
        label: str,
        hint: str | None = None,
    ) -> bool:
        if path is None:
            return False
        if not path.is_file():
            self.error(
                code,
                f"Missing required {label}.",
                file=self.relative_label(path),
                expected="existing regular file",
                actual="missing",
                hint=hint,
            )
            return False
        return True

    def load_json(
        self,
        path: Path | None,
        *,
        required: bool,
        label: str,
        missing_code: str = "REQUIRED_FILE_MISSING",
    ) -> dict[str, Any] | None:
        if path is None:
            return None
        if not path.is_file():
            if required:
                self.error(
                    missing_code,
                    f"Missing required {label}.",
                    file=self.relative_label(path),
                    expected="JSON object file",
                    actual="missing",
                )
            else:
                self.warning(
                    missing_code,
                    f"Optional {label} is missing.",
                    file=self.relative_label(path),
                )
            return None
        try:
            before_stat = path.stat()
            size = before_stat.st_size
        except OSError as exc:
            self.error(
                "FILE_READ_ERROR",
                f"Could not inspect {label}.",
                file=self.relative_label(path),
                actual=str(exc),
            )
            return None
        if size > MAX_JSON_BYTES:
            self.error(
                "JSON_FILE_TOO_LARGE",
                f"{label} exceeds the intake JSON size limit.",
                file=self.relative_label(path),
                expected=f"at most {MAX_JSON_BYTES} bytes",
                actual=size,
                hint="Split large sampled data from the contract JSON.",
            )
            return None
        try:
            with path.open("rb") as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            after_stat = path.stat()
        except OSError as exc:
            self.error(
                "FILE_READ_ERROR",
                f"Could not read {label}.",
                file=self.relative_label(path),
                actual=str(exc),
            )
            return None
        if after_stat.st_size > MAX_JSON_BYTES or len(raw) > MAX_JSON_BYTES:
            self.error(
                "JSON_FILE_TOO_LARGE",
                f"{label} exceeds the intake JSON size limit.",
                file=self.relative_label(path),
                expected=f"at most {MAX_JSON_BYTES} bytes",
                actual=max(after_stat.st_size, len(raw)),
                hint="Split large sampled data from the contract JSON.",
            )
            return None
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            self.error(
                "JSON_ENCODING_ERROR",
                f"{label} is not valid UTF-8.",
                file=self.relative_label(path),
                actual=str(exc),
                hint="Write JSON as UTF-8.",
            )
            return None
        before_identity = _file_identity(before_stat)
        after_identity = _file_identity(after_stat)
        if (
            before_identity != after_identity
            or len(raw) != after_stat.st_size
        ):
            self.error(
                "FILE_CHANGED_DURING_VALIDATION",
                f"{label} changed while it was being read.",
                file=self.relative_label(path),
                expected=before_identity,
                actual=after_identity,
                hint="Stop writers and rerun intake on an immutable package.",
            )
            return None
        self.loaded_document_identities[path] = (
            after_identity,
            hashlib.sha256(raw).hexdigest(),
        )
        nesting = _max_json_nesting(text)
        if nesting > MAX_JSON_NESTING:
            self.error(
                "JSON_NESTING_TOO_DEEP",
                f"{label} exceeds the JSON nesting limit.",
                file=self.relative_label(path),
                expected=f"at most {MAX_JSON_NESTING} object/array levels",
                actual=nesting,
                hint="Flatten the document and reference repeated structures by ID.",
            )
            return None
        try:
            value = json.loads(
                text,
                object_pairs_hook=_strict_json_object,
                parse_constant=_reject_nonfinite_json_number,
                parse_float=_strict_json_float,
                parse_int=_strict_json_int,
            )
        except json.JSONDecodeError as exc:
            self.error(
                "JSON_PARSE_ERROR",
                f"{label} contains malformed JSON.",
                file=self.relative_label(path),
                json_path=f"line {exc.lineno}, column {exc.colno}",
                actual=exc.msg,
                hint="Repair JSON syntax, then rerun intake.",
            )
            return None
        except _DuplicateJsonKeyError as exc:
            self.error(
                "JSON_DUPLICATE_KEY",
                f"{label} contains a duplicate object key.",
                file=self.relative_label(path),
                actual=exc.key,
                hint=(
                    "Keep each JSON object key exactly once; duplicate keys "
                    "are ambiguous across parsers."
                ),
            )
            return None
        except _NonFiniteJsonNumberError as exc:
            self.error(
                "JSON_NUMBER_NONFINITE",
                f"{label} contains a non-finite numeric token.",
                file=self.relative_label(path),
                expected="finite JSON number or null",
                actual=exc.token,
                hint="Replace NaN or Infinity with a finite value or null.",
            )
            return None
        except RecursionError:
            self.error(
                "JSON_NESTING_TOO_DEEP",
                f"{label} exceeds the parser recursion limit.",
                file=self.relative_label(path),
                expected=f"at most {MAX_JSON_NESTING} object/array levels",
                actual=nesting,
            )
            return None
        if not isinstance(value, dict):
            self.error(
                "JSON_ROOT_TYPE_INVALID",
                f"{label} must contain a JSON object at its root.",
                file=self.relative_label(path),
                json_path="$",
                expected="object",
                actual=type(value).__name__,
            )
            return None
        return value

    def verify_loaded_documents_unchanged(self) -> None:
        for path, expected in sorted(
            self.loaded_document_identities.items(),
            key=lambda item: str(item[0]).casefold(),
        ):
            expected_identity, expected_digest = expected
            try:
                actual_identity = _file_identity(path.stat())
            except OSError as exc:
                self.error(
                    "DOCUMENT_STABILITY_CHECK_FAILED",
                    "Could not recheck a loaded JSON document.",
                    file=self.relative_label(path),
                    actual=str(exc),
                )
                continue
            actual_digest = self.sha256_file(path)
            if actual_digest is None:
                continue
            if (
                actual_identity != expected_identity
                or actual_digest != expected_digest
            ):
                self.error(
                    "DOCUMENT_CHANGED_DURING_VALIDATION",
                    "A loaded JSON document changed before intake completed.",
                    file=self.relative_label(path),
                    expected={
                        "identity": expected_identity,
                        "sha256": expected_digest,
                    },
                    actual={
                        "identity": actual_identity,
                        "sha256": actual_digest,
                    },
                    hint=(
                        "Stop the generator and rerun intake on an immutable "
                        "design package and requirements file."
                    ),
                )

    def check_schema_version(
        self,
        document: dict[str, Any] | None,
        file: Path | str | None,
        *,
        supported: tuple[str, ...] = ("0.1",),
    ) -> None:
        if document is None:
            return
        value = document.get("schema_version")
        if value not in supported:
            self.error(
                "SCHEMA_VERSION_UNSUPPORTED",
                "Unsupported or missing schema_version.",
                file=self.relative_label(file) if isinstance(file, Path) else file,
                json_path="$.schema_version",
                expected=(
                    supported[0]
                    if len(supported) == 1
                    else list(supported)
                ),
                actual=value,
            )

    def require_string(
        self,
        document: dict[str, Any],
        key: str,
        *,
        file: Path | str,
        base_path: str = "$",
        code: str = "FIELD_REQUIRED",
    ) -> str | None:
        value = document.get(key)
        if not isinstance(value, str) or not value.strip():
            self.error(
                code,
                f"{key} must be a non-empty string.",
                file=self.relative_label(file) if isinstance(file, Path) else file,
                json_path=f"{base_path}.{key}",
                expected="non-empty string",
                actual=value,
            )
            return None
        return value

    def require_id(
        self,
        document: dict[str, Any],
        key: str,
        *,
        file: Path | str,
        base_path: str = "$",
    ) -> str | None:
        value = self.require_string(
            document, key, file=file, base_path=base_path, code="ID_REQUIRED"
        )
        if value is not None and not ID_PATTERN.fullmatch(value):
            self.error(
                "ID_FORMAT_INVALID",
                f"{key} is not a stable ASCII identifier.",
                file=self.relative_label(file) if isinstance(file, Path) else file,
                json_path=f"{base_path}.{key}",
                expected="^[a-z][a-z0-9_]{0,63}$",
                actual=value,
                hint="Use lowercase ASCII letters, digits, and underscores.",
            )
            return None
        return value

    def require_list(
        self,
        document: dict[str, Any],
        key: str,
        *,
        file: Path | str,
        base_path: str = "$",
        nonempty: bool = False,
        code: str = "LIST_REQUIRED",
    ) -> list[Any] | None:
        value = document.get(key)
        if not isinstance(value, list) or (nonempty and not value):
            self.error(
                code,
                f"{key} must be {'a non-empty' if nonempty else 'an'} array.",
                file=self.relative_label(file) if isinstance(file, Path) else file,
                json_path=f"{base_path}.{key}",
                expected="non-empty array" if nonempty else "array",
                actual=value,
            )
            return None
        if len(value) > MAX_LIST_ITEMS:
            self.error(
                "LIST_TOO_LARGE",
                f"{key} exceeds the intake collection limit.",
                file=self.relative_label(file) if isinstance(file, Path) else file,
                json_path=f"{base_path}.{key}",
                expected=f"at most {MAX_LIST_ITEMS} items",
                actual=len(value),
                hint="Split the design into smaller referenced documents.",
            )
            return None
        return value

    def vector(
        self,
        value: Any,
        *,
        length: int,
        file: Path | str,
        json_path: str,
        code: str = "VECTOR_INVALID",
        nonzero: bool = False,
        normalized: bool = False,
        normalization_code: str = "VECTOR_NOT_UNIT",
        normalization_tolerance: float = 1e-6,
    ) -> list[float] | None:
        label = self.relative_label(file) if isinstance(file, Path) else file
        if (
            not isinstance(value, list)
            or len(value) != length
            or any(
                isinstance(item, bool) or not isinstance(item, (int, float))
                for item in value
            )
            or any(not math.isfinite(float(item)) for item in value)
        ):
            self.error(
                code,
                f"Expected a finite numeric vector of length {length}.",
                file=label,
                json_path=json_path,
                expected=f"{length} finite numbers",
                actual=value,
            )
            return None
        result = [float(item) for item in value]
        norm = math.sqrt(sum(item * item for item in result))
        if nonzero and norm <= 1e-12:
            self.error(
                "VECTOR_ZERO",
                "Direction vector must not be zero.",
                file=label,
                json_path=json_path,
                expected="non-zero vector",
                actual=value,
            )
            return None
        if normalized and abs(norm - 1.0) > normalization_tolerance:
            self.error(
                normalization_code,
                "Vector must be normalized to unit length.",
                file=label,
                json_path=json_path,
                expected="Euclidean norm = 1",
                actual={"value": value, "norm": norm},
                hint=(
                    "Normalize the vector; put force or torque magnitude "
                    "in declared_range."
                ),
            )
            return None
        return result

    def sha256_file(self, path: Path) -> str | None:
        try:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
        except OSError as exc:
            self.error(
                "FILE_HASH_READ_ERROR",
                "Could not read a file while computing SHA-256.",
                file=self.relative_label(path),
                actual=str(exc),
                hint="Make the file readable and stop writers before intake.",
            )
            return None
        before_identity = (
            before.st_size,
            before.st_mtime_ns,
            getattr(before, "st_ino", None),
        )
        after_identity = (
            after.st_size,
            after.st_mtime_ns,
            getattr(after, "st_ino", None),
        )
        if before_identity != after_identity:
            self.error(
                "FILE_CHANGED_DURING_VALIDATION",
                "File changed while its SHA-256 was being computed.",
                file=self.relative_label(path),
                expected=before_identity,
                actual=after_identity,
                hint="Stop the generator and rerun intake on an immutable package.",
            )
            return None
        digest_hex = digest.hexdigest()
        try:
            identity_path = path.resolve()
        except (OSError, RuntimeError):
            identity_path = path.absolute()
        self.hashed_file_identities.setdefault(
            identity_path,
            (after_identity, digest_hex),
        )
        return digest_hex

    def warn_unknown_fields(
        self,
        document: dict[str, Any],
        allowed: set[str],
        *,
        file: Path | str,
        json_path: str = "$",
    ) -> None:
        label = self.relative_label(file) if isinstance(file, Path) else file
        unknown = sorted(set(document) - allowed)
        for key in unknown[:20]:
            self.warning(
                "FIELD_UNRECOGNIZED",
                "Field is not defined by intake schema 0.1 and is ignored.",
                file=label,
                json_path=f"{json_path}.{key}",
                actual=key,
                hint=(
                    "Correct a spelling mistake or remove the field. "
                    "Do not assume ignored metadata affects validation."
                ),
            )
        if len(unknown) > 20:
            self.warning(
                "FIELD_UNRECOGNIZED_LIMIT",
                "Additional unrecognized fields were omitted from the report.",
                file=label,
                json_path=json_path,
                actual={"total": len(unknown), "reported": 20},
            )

    def scan_for_forbidden_authority(
        self, value: Any, *, file: Path | str, path: str = "$"
    ) -> None:
        label = self.relative_label(file) if isinstance(file, Path) else file
        stack: list[tuple[Any, str]] = [(value, path)]
        while stack:
            current, current_path = stack.pop()
            if isinstance(current, dict):
                for key, child in reversed(list(current.items())):
                    child_path = f"{current_path}.{key}"
                    normalized_key = _normalized_json_key(key)
                    if normalized_key in FORBIDDEN_AUTHORITY_KEYS:
                        self.error(
                            "AI_AUTHORITY_FIELD_FORBIDDEN",
                            f"AI output may not define validator-owned field {key!r}.",
                            file=label,
                            json_path=child_path,
                            actual=child,
                            hint=(
                                "Remove the field; the validator owns hidden "
                                "tests, thresholds, verdicts, and certificates."
                            ),
                        )
                    if (
                        normalized_key == "verification_status"
                        and child != "UNVERIFIED"
                    ):
                        self.error(
                            "AI_VERIFICATION_CLAIM_FORBIDDEN",
                            "AI-provided verification_status must remain UNVERIFIED.",
                            file=label,
                            json_path=child_path,
                            expected="UNVERIFIED",
                            actual=child,
                        )
                    stack.append((child, child_path))
            elif isinstance(current, list):
                for index in range(len(current) - 1, -1, -1):
                    stack.append(
                        (current[index], f"{current_path}[{index}]")
                    )
