"""Content-addressed cache of imported STEP shapes.

Importing a STEP through OCCT (``ReadFile`` + ``TransferRoots``) costs seconds
per megabyte, and one validation run imports the same assembly at cold start,
intake, plan derivation, collider materialization, and every rendered preview.
The transferred shape is a pure function of the file bytes, the OCCT build,
and the import unit, so it is imported once and stored as an OCCT binary BRep
next to a JSON solid inventory (bounds and B-rep validity per solid, which are
the other slow per-solid computations).  Reloading the binary BRep takes tens
of milliseconds instead of seconds.

Cache layout::

    <root>/occt-<OCP version>-<unit>/<sha256 of file bytes>.brep
    <root>/occt-<OCP version>-<unit>/<sha256 of file bytes>.json
    <root>/occt-<OCP version>-<unit>/<sha256 of file bytes>.<name>.json  (sidecars)

The JSON inventory is written last, so a directory listing with the ``.json``
present means the entry is complete.  Writes are atomic renames, which makes
concurrent lanes populating the same entry safe: the loser simply overwrites
identical bytes.  Any read problem (missing file, truncated BRep, solid count
mismatch) is treated as a miss and the STEP is imported again.

Configuration (environment):

- ``SIMREADY_STEP_SHAPE_CACHE_DIR`` overrides the cache root
  (default ``~/.cache/simready/step_shapes``).
- ``SIMREADY_STEP_SHAPE_CACHE=0`` disables reads and writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CACHE_SCHEMA = "step_shape_cache_v1"
DEFAULT_UNIT = "MM"
DEFAULT_MAX_SOLIDS = 20_000
_ENV_DIR = "SIMREADY_STEP_SHAPE_CACHE_DIR"
_ENV_ENABLE = "SIMREADY_STEP_SHAPE_CACHE"
_SIDECAR_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


class StepShapeError(RuntimeError):
    """A STEP file could not be turned into a shape."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CachedStepShape:
    """One imported STEP: the compound shape, its solids, and their inventory."""

    shape: Any
    solids: tuple[Any, ...]
    inventory: tuple[dict[str, Any], ...]
    source_sha256: str
    cache_hit: bool


def cache_enabled() -> bool:
    return os.environ.get(_ENV_ENABLE, "1").strip().lower() not in {"0", "false", "no", "off"}


def cache_root() -> Path:
    override = os.environ.get(_ENV_DIR, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "simready" / "step_shapes"


def _occt_version() -> str:
    try:
        import OCP

        version = str(getattr(OCP, "__version__", "") or "unknown")
    except ImportError:
        version = "unknown"
    return re.sub(r"[^0-9A-Za-z.]+", "_", version)


def entry_dir(unit: str = DEFAULT_UNIT) -> Path:
    return cache_root() / f"occt-{_occt_version()}-{unit.upper()}"


def step_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _explore_solids(shape: Any) -> list[Any]:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    solids: list[Any] = []
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    return solids


def inventory_from_solids(solids: list[Any]) -> list[dict[str, Any]]:
    """Per-solid exact bounds (mm) and B-rep validity, in explorer order.

    This is the slow part of a solid inventory after the import itself
    (``AddOptimal`` exact bounds and ``BRepCheck_Analyzer``), so it is stored
    in the cache next to the shape.
    """

    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopoDS import TopoDS

    inventory: list[dict[str, Any]] = []
    for index, raw in enumerate(solids, start=1):
        solid = TopoDS.Solid_s(raw)
        box = Bnd_Box()
        BRepBndLib.AddOptimal_s(solid, box, False, False)
        if box.IsVoid():
            minimum = [0.0, 0.0, 0.0]
            maximum = [0.0, 0.0, 0.0]
        else:
            min_x, min_y, min_z, max_x, max_y, max_z = box.Get()
            minimum = [float(min_x), float(min_y), float(min_z)]
            maximum = [float(max_x), float(max_y), float(max_z)]
        inventory.append(
            {
                "solid_id": f"solid_{index}",
                "valid_brep": bool(BRepCheck_Analyzer(solid).IsValid()),
                "bounds_mm": {"min": minimum, "max": maximum},
                "size_mm": [maximum[axis] - minimum[axis] for axis in range(3)],
            }
        )
    return inventory


def import_step_shape(path: Path, *, unit: str = DEFAULT_UNIT) -> Any:
    """Import a STEP with OCCT and return the compound of all roots."""

    from OCP.IFSelect import IFSelect_RetDone
    from OCP.Interface import Interface_Static
    from OCP.STEPControl import STEPControl_Reader

    Interface_Static.SetCVal_s("xstep.cascade.unit", unit.upper())
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(path))
    if int(status) != int(IFSelect_RetDone):
        raise StepShapeError(
            "STEP_READ_FAILED", f"STEP reader returned status {int(status)}"
        )
    transferred = int(reader.TransferRoots())
    shape = reader.OneShape()
    if transferred <= 0 or shape.IsNull():
        raise StepShapeError("STEP_TRANSFER_EMPTY", "STEP transfer produced no shape")
    return shape


def _write_atomic(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        # Prove ordinary Python can create a file here before handing the path
        # to an OCCT native writer.  On Windows, BinTools.Write_s can terminate
        # the process with 0xC0000005 when its destination is access-denied,
        # so the surrounding OSError handler never gets a chance to run.
        with temporary.open("xb"):
            pass
        result = writer(temporary)
        if result is False:
            raise OSError(f"atomic writer failed for {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json(path: Path, payload: Any) -> None:
    def writer(temporary: Path) -> None:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )

    _write_atomic(path, writer)


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _entry_paths(source_sha256: str, unit: str) -> tuple[Path, Path]:
    base = entry_dir(unit) / source_sha256
    return base.with_suffix(".brep"), base.with_suffix(".json")


def _read_entry(
    source_sha256: str, unit: str
) -> tuple[Any, list[Any], list[dict[str, Any]]] | None:
    brep_path, json_path = _entry_paths(source_sha256, unit)
    payload = _read_json(json_path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CACHE_SCHEMA
        or not isinstance(payload.get("inventory"), list)
        or not brep_path.is_file()
    ):
        return None
    try:
        from OCP.BinTools import BinTools
        from OCP.TopoDS import TopoDS_Shape

        shape = TopoDS_Shape()
        BinTools.Read_s(shape, str(brep_path))
    except Exception:
        return None
    if shape.IsNull():
        return None
    solids = _explore_solids(shape)
    inventory = payload["inventory"]
    if len(solids) != len(inventory):
        return None
    return shape, solids, inventory


def store_step_shape(
    *,
    shape: Any,
    source_sha256: str,
    source_path: Path,
    unit: str = DEFAULT_UNIT,
    inventory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Persist an already-imported shape; returns the inventory that was stored."""

    solids = _explore_solids(shape)
    if inventory is None:
        inventory = inventory_from_solids(solids)
    if not cache_enabled():
        return inventory
    brep_path, json_path = _entry_paths(source_sha256, unit)
    try:
        from OCP.BinTools import BinTools

        _write_atomic(brep_path, lambda temporary: BinTools.Write_s(shape, str(temporary)))
        _write_json(
            json_path,
            {
                "schema_version": CACHE_SCHEMA,
                "source_sha256": source_sha256,
                "source_path": str(source_path),
                "occt_version": _occt_version(),
                "unit": unit.upper(),
                "solid_count": len(solids),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
                "inventory": inventory,
            },
        )
    except OSError:
        # A read-only or full cache directory must never fail the import.
        pass
    return inventory


def load_step_shape(
    path: Path,
    *,
    unit: str = DEFAULT_UNIT,
    max_solids: int = DEFAULT_MAX_SOLIDS,
) -> CachedStepShape:
    """Return the imported STEP shape, from cache when the bytes were seen before.

    Raises ``StepShapeError`` when the STEP cannot be read, transfers nothing,
    or contains more than ``max_solids`` solids.  OCCT import errors are never
    cached; only successful imports are.
    """

    source = Path(path)
    source_sha256 = step_file_sha256(source)
    if cache_enabled():
        cached = _read_entry(source_sha256, unit)
        if cached is not None:
            shape, solids, inventory = cached
            if len(solids) > max_solids:
                raise StepShapeError(
                    "STEP_TOO_MANY_SOLIDS",
                    f"STEP contains more than {max_solids} solids; "
                    "split the simulation-focused export",
                )
            return CachedStepShape(
                shape=shape,
                solids=tuple(solids),
                inventory=tuple(inventory),
                source_sha256=source_sha256,
                cache_hit=True,
            )
    shape = import_step_shape(source, unit=unit)
    solids = _explore_solids(shape)
    if len(solids) > max_solids:
        raise StepShapeError(
            "STEP_TOO_MANY_SOLIDS",
            f"STEP contains more than {max_solids} solids; "
            "split the simulation-focused export",
        )
    inventory = store_step_shape(
        shape=shape, source_sha256=source_sha256, source_path=source, unit=unit
    )
    return CachedStepShape(
        shape=shape,
        solids=tuple(solids),
        inventory=tuple(inventory),
        source_sha256=source_sha256,
        cache_hit=False,
    )


def read_sidecar(source_sha256: str, name: str, *, unit: str = DEFAULT_UNIT) -> Any | None:
    """Read a JSON sidecar (e.g. the XDE product tree) attached to an entry."""

    if not cache_enabled() or _SIDECAR_NAME.fullmatch(name) is None:
        return None
    payload = _read_json(entry_dir(unit) / f"{source_sha256}.{name}.json")
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA:
        return None
    return payload.get("value")


def write_sidecar(source_sha256: str, name: str, value: Any, *, unit: str = DEFAULT_UNIT) -> None:
    if not cache_enabled():
        return
    if _SIDECAR_NAME.fullmatch(name) is None:
        raise ValueError(f"invalid sidecar name {name!r}")
    try:
        _write_json(
            entry_dir(unit) / f"{source_sha256}.{name}.json",
            {"schema_version": CACHE_SCHEMA, "source_sha256": source_sha256, "value": value},
        )
    except OSError:
        pass
