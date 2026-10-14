from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any


MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_TRACE_JSON_BYTES = 512 * 1024 * 1024
_TRACE_JSON_WRITE_BUFFER_BYTES = 1024 * 1024
MAX_JSON_NESTING = 64
MECHANISM_SOURCE_SHA256_ENV = "SIMREADY_MECHANISM_SOURCE_SHA256"

# ``family_plugins.py`` owns callbacks for every validation stage.  Its Python
# import graph therefore reaches certificate and whole-machine modules even
# when a queue job is only executing one physical/ideal/comparison lane.  Stop
# the lane source closure at those stage boundaries so changes that cannot
# alter a lane result do not invalidate otherwise reusable physics evidence.
# Seeds are relative to ``validation_pipeline`` so the closure also follows
# real lane imports in ``contracts`` and ``runtime``. Runtime and comparison
# roots are explicit because several family callbacks import them lazily.
_LANE_EXECUTION_SOURCE_SEEDS = (
    "mechanism.bicycle_chain_comparison",
    "mechanism.bicycle_chain_runtime",
    "mechanism.bicycle_ground_roll_comparison",
    "mechanism.bicycle_ground_roll_runtime",
    "mechanism.click_pen_comparison",
    "mechanism.click_pen_runtime",
    "mechanism.clock_gear_comparison",
    "mechanism.clock_gear_runtime",
    "mechanism.comparison",
    "mechanism.compound_runtime",
    "mechanism.detent_slider_comparison",
    "mechanism.detent_slider_runtime",
    "mechanism.ideal_comparison",
    "mechanism.ideal_reference",
    "mechanism.lane_worker",
    "mechanism.multi_output_shaft_comparison",
    "mechanism.physical_ideal_comparison",
    # Project adapters execute callbacks through this ``-m`` target, whose
    # module name is data rather than a Python import edge.
    "mechanism.project_adapter_worker",
    "mechanism.rack_pinion_comparison_v3",
    "mechanism.rack_pinion_runtime_v10",
    "mechanism.scissor_lift_comparison",
    "mechanism.scissor_lift_runtime",
    "mechanism.worker",
)
_LANE_EXECUTION_SOURCE_BOUNDARIES = frozenset(
    {
        "mechanism.bicycle_chain_certificate",
        "mechanism.bicycle_chain_whole_machine",
        "mechanism.bicycle_ground_roll_certificate",
        "mechanism.bicycle_ground_roll_whole_machine",
        "mechanism.certificate",
        "mechanism.click_pen_certificate",
        "mechanism.clock_gear_certificate",
        "mechanism.clock_gear_whole_machine",
        "mechanism.detent_slider_certificate",
        "mechanism.finalize",
        "mechanism.multi_output_shaft_certificate",
        "mechanism.rack_pinion_certificate",
        "mechanism.rack_pinion_whole_machine_v4",
        "mechanism.reuse",
        "mechanism.scissor_lift_certificate",
        "mechanism.scissor_lift_whole_machine",
        "mechanism.whole_machine",
        "mechanism.whole_machine_worker",
    }
)
_LANE_EXECUTION_SOURCE_RESOURCES = (
    "mechanism/templates/bicycle_chain_sprocket_drive_v2.json",
)


class JsonDocumentError(ValueError):
    """A trusted pipeline document is malformed or unsafe to consume."""


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonDocumentError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise JsonDocumentError(f"non-finite JSON number: {value}")


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


def read_json(path: str | Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    source = Path(path)
    try:
        before = source.stat()
    except OSError as exc:
        raise JsonDocumentError(f"cannot inspect {source}: {exc}") from exc
    if not source.is_file() or source.is_symlink():
        raise JsonDocumentError(
            f"JSON input must be a regular non-symlink file: {source}"
        )
    if before.st_size > max_bytes:
        raise JsonDocumentError(f"JSON input exceeds {max_bytes} bytes: {source}")
    try:
        raw = source.read_bytes()
        after = source.stat()
    except OSError as exc:
        raise JsonDocumentError(f"cannot read {source}: {exc}") from exc
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
    if identity_before != identity_after or len(raw) != after.st_size:
        raise JsonDocumentError(f"JSON input changed while being read: {source}")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as exc:
        raise JsonDocumentError(f"invalid UTF-8 JSON in {source}: {exc}") from exc
    nesting = _max_json_nesting(text)
    if nesting > MAX_JSON_NESTING:
        raise JsonDocumentError(
            f"JSON nesting {nesting} exceeds {MAX_JSON_NESTING}: {source}"
        )
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except JsonDocumentError:
        raise
    except (ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise JsonDocumentError(f"invalid bounded JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise JsonDocumentError(f"JSON root must be an object: {source}")
    _finite_tree(value)
    return value


def read_trace_json(path: str | Path) -> dict[str, Any]:
    """Read a full-cadence trace under a separate bounded artifact budget."""

    return read_json(path, max_bytes=MAX_TRACE_JSON_BYTES)


def ensure_inside(root: str | Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise JsonDocumentError(
            "artifact path must be a non-empty string without NUL"
        )
    path = Path(relative)
    if path.is_absolute() or path.drive or ".." in path.parts:
        raise JsonDocumentError(f"artifact path must be relative: {relative!r}")
    root_path = Path(root).resolve()
    candidate = (root_path / path).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise JsonDocumentError(
            f"artifact path escapes package: {relative!r}"
        ) from exc
    return candidate


def relative_to_root(root: str | Path, path: str | Path) -> str:
    root_path = Path(root).resolve()
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise JsonDocumentError(f"path is outside workspace: {candidate}") from exc


def validator_import_root() -> Path:
    """Directory whose children include the installed validator package.

    Subprocess lanes must import the exact code that scheduled them, no
    matter which workspace they validate: benchmark bundles run in clean
    scratch directories where nothing is importable from the workspace, so
    the import root is derived from this installed module, never from the
    workspace path.
    """

    return Path(__file__).resolve().parents[3]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    source = Path(path)
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(root: str | Path) -> str:
    source_root = Path(root).resolve()
    sources = tuple(
        path
        for path in sorted(source_root.glob("*.py"), key=lambda item: item.name)
        if path.is_file() and not path.is_symlink()
    )
    if not sources:
        raise JsonDocumentError(f"source tree contains no Python files: {source_root}")
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def mechanism_source_sha256() -> str:
    return source_tree_sha256(Path(__file__).resolve().parent)


def _source_module_path(source_root: Path, module: str) -> Path | None:
    relative = Path(*module.split("."))
    module_file = source_root / relative.with_suffix(".py")
    if module_file.is_file() and not module_file.is_symlink():
        return module_file
    package_file = source_root / relative / "__init__.py"
    if package_file.is_file() and not package_file.is_symlink():
        return package_file
    return None


def _relative_source_imports(source: Path, module: str) -> frozenset[str]:
    try:
        tree = ast.parse(source.read_bytes(), filename=str(source))
    except (OSError, SyntaxError, ValueError) as exc:
        raise JsonDocumentError(
            f"cannot parse Python source {source}: {exc}"
        ) from exc
    package = module.split(".")[:-1]
    imports: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level < 1:
            continue
        ascend = node.level - 1
        if ascend > len(package):
            # The import leaves ``validation_pipeline`` and is outside this
            # source identity.
            continue
        base = package[: len(package) - ascend]
        if node.module:
            imports.add(".".join([*base, *node.module.split(".")]))
        else:
            imports.update(
                ".".join([*base, alias.name]) for alias in node.names
            )
    return frozenset(imports)


def _source_import_closure(
    source_root: Path,
    *,
    seeds: Iterable[str],
    boundaries: frozenset[str] = frozenset(),
) -> tuple[Path, ...]:
    seed_modules = tuple(seeds)
    pending = list(seed_modules)
    discovered: dict[str, Path] = {}
    while pending:
        module = pending.pop()
        if module in discovered or module in boundaries:
            continue
        source = _source_module_path(source_root, module)
        if source is None:
            if module in seed_modules:
                raise JsonDocumentError(
                    f"source closure seed does not exist: {module!r}"
                )
            continue
        discovered[module] = source
        pending.extend(_relative_source_imports(source, module))
    return tuple(discovered[module] for module in sorted(discovered))


def _source_import_closure_sha256(
    source_root: Path,
    *,
    seeds: Iterable[str],
    boundaries: frozenset[str] = frozenset(),
    resources: Iterable[str] = (),
) -> str:
    sources = list(
        _source_import_closure(
            source_root,
            seeds=seeds,
            boundaries=boundaries,
        )
    )
    for relative in resources:
        source = source_root / relative
        if not source.is_file() or source.is_symlink():
            raise JsonDocumentError(
                f"source closure resource does not exist: {relative!r}"
            )
        sources.append(source)
    sources.sort(key=lambda source: source.relative_to(source_root).as_posix())
    digest = hashlib.sha256()
    for source in sources:
        digest.update(source.relative_to(source_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    return digest.hexdigest()


def lane_execution_source_sha256() -> str:
    """Identity of code that can change a per-load lane's saved evidence."""

    return _source_import_closure_sha256(
        Path(__file__).resolve().parent.parent,
        seeds=_LANE_EXECUTION_SOURCE_SEEDS,
        boundaries=_LANE_EXECUTION_SOURCE_BOUNDARIES,
        resources=_LANE_EXECUTION_SOURCE_RESOURCES,
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _finite_tree(value: Any, *, label: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise JsonDocumentError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, label=f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _finite_tree(child, label=f"{label}[{index}]")


def write_json_atomic(path: str | Path, value: Any) -> None:
    _finite_tree(value)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("JSON write made no forward progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_trace_json_atomic(path: str | Path, value: Any) -> None:
    """Atomically stream a bounded full-cadence trace to disk.

    Trace payloads may contain tens of thousands of samples.  Encoding the
    entire document to one ``str`` and then one ``bytes`` object temporarily
    tripled the live trace storage.  ``iterencode`` preserves the same compact,
    fail-closed JSON contract while bounding the writer's additional memory to
    one encoder chunk.  The governance JSON writer above remains unchanged.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o666,
        )
        encoded_bytes = 0
        encoder = json.JSONEncoder(
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        buffer = bytearray()

        def write_bytes(payload: bytes | bytearray) -> None:
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError(
                        "trace JSON write made no forward progress"
                    )
                offset += written
            view.release()

        def flush_buffer() -> None:
            if buffer:
                write_bytes(buffer)
                buffer.clear()

        def write_chunk(chunk: str) -> None:
            nonlocal encoded_bytes
            encoded = chunk.encode("utf-8")
            encoded_bytes += len(encoded)
            if encoded_bytes > MAX_TRACE_JSON_BYTES:
                raise JsonDocumentError(
                    "trace JSON exceeds "
                    f"{MAX_TRACE_JSON_BYTES} bytes: {destination}"
                )
            if len(buffer) + len(encoded) > _TRACE_JSON_WRITE_BUFFER_BYTES:
                flush_buffer()
            if len(encoded) >= _TRACE_JSON_WRITE_BUFFER_BYTES:
                write_bytes(encoded)
            else:
                buffer.extend(encoded)

        for chunk in encoder.iterencode(value):
            write_chunk(chunk)
        write_chunk("\n")
        flush_buffer()
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, destination)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
