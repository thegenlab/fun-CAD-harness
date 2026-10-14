"""Deterministic authoring helpers for dynamic convex collision components.

Source-aware CAD generators should export the same convex construction members
used to build a detailed fused visual solid.  Imported STEP packages may use
the optional CoACD backend to propose components, but the existing authored-
component fidelity validator remains the only acceptance authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
from typing import Any, Mapping, Sequence

from .authored_collision_components import (
    COLLISION_COMPONENT_MANIFEST_PATH,
    COLLISION_COMPONENT_MANIFEST_SCHEMA,
    load_authored_collision_components,
)
from .io import canonical_sha256, read_json, sha256_file, write_json_atomic
from .occt_collision_fidelity import shape_solids


_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,95}\Z")
_SOLID_ID = re.compile(r"solid_[1-9][0-9]*\Z")


class CollisionComponentAuthoringError(ValueError):
    pass


@dataclass(frozen=True)
class CoacdPreset:
    threshold_m: float
    max_convex_hulls: int
    resolution: int
    mcts_iterations: int
    mcts_max_depth: int


COACD_PRESETS: Mapping[str, CoacdPreset] = {
    "draft": CoacdPreset(0.002, 16, 2_000, 80, 4),
    "balanced": CoacdPreset(0.001, 32, 2_000, 100, 4),
    "contact": CoacdPreset(0.0005, 64, 4_000, 120, 5),
}


def _identifier(value: str, label: str) -> str:
    text = str(value or "").strip()
    if _IDENTIFIER.fullmatch(text) is None:
        raise CollisionComponentAuthoringError(
            f"{label} must be a bounded identifier"
        )
    return text


def _source_solid_id(value: str) -> str:
    text = str(value or "").strip()
    if _SOLID_ID.fullmatch(text) is None:
        raise CollisionComponentAuthoringError(
            "source_solid_id must use canonical solid_N syntax"
        )
    return text


def _write_step(path: Path, shape: Any) -> None:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer

    solids = shape_solids(shape)
    if len(solids) != 1:
        raise CollisionComponentAuthoringError(
            "each collision component must contain exactly one solid"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = STEPControl_Writer()
    if writer.Transfer(solids[0], STEPControl_AsIs) != IFSelect_RetDone:
        raise CollisionComponentAuthoringError(
            f"could not transfer collision component {path.name!r}"
        )
    if writer.Write(str(path)) != IFSelect_RetDone:
        raise CollisionComponentAuthoringError(
            f"could not write collision component {path.name!r}"
        )


def _manifest(package: Path) -> dict[str, Any]:
    path = package / COLLISION_COMPONENT_MANIFEST_PATH
    if not path.is_file():
        return {
            "schema_version": COLLISION_COMPONENT_MANIFEST_SCHEMA,
            "sources": [],
        }
    value = read_json(path)
    if (
        not isinstance(value, dict)
        or value.get("schema_version")
        != COLLISION_COMPONENT_MANIFEST_SCHEMA
        or not isinstance(value.get("sources"), list)
    ):
        raise CollisionComponentAuthoringError(
            "existing collision component manifest is invalid"
        )
    return value


def certify_draft_collision_components(
    package_dir: str | Path,
    *,
    workspace_root: str | Path,
    expected_body_by_solid: Mapping[str, str],
    mesh_linear_deflection_m: float = 0.0005,
    collision_envelope_m: float = 1.0e-5,
) -> tuple[dict[str, Any], ...]:
    """Certify draft proxy unions against their authoritative STEP solids."""

    import numpy as np
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    from .occt_collision_fidelity import mesh_shape_m

    package = Path(package_dir).resolve()
    workspace = Path(workspace_root).resolve()
    assembly = package / "assembly.step"
    if assembly.is_symlink() or not assembly.is_file():
        raise CollisionComponentAuthoringError(
            "draft collision certification requires assembly.step"
        )
    reader = STEPControl_Reader()
    if (
        reader.ReadFile(str(assembly)) != IFSelect_RetDone
        or reader.TransferRoots() <= 0
    ):
        raise CollisionComponentAuthoringError(
            "OCCT could not import the draft assembly STEP"
        )
    solids = list(shape_solids(reader.OneShape()))
    if not solids:
        raise CollisionComponentAuthoringError(
            "draft assembly STEP contains no solids"
        )
    artifacts = {
        path.relative_to(package).as_posix(): sha256_file(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and path.name != "package.json"
    }
    plan = SimpleNamespace(
        artifact_hashes=artifacts,
        package_binding_sha256=canonical_sha256(artifacts),
    )
    with tempfile.TemporaryDirectory(
        prefix="simready-draft-collision-cert-"
    ) as temporary:
        scratch = Path(temporary)
        expected: dict[tuple[str, str], tuple[Any, Any]] = {}
        characteristic_lengths = []
        for source_solid_id, body_id in sorted(
            expected_body_by_solid.items()
        ):
            match = _SOLID_ID.fullmatch(str(source_solid_id))
            if match is None:
                raise CollisionComponentAuthoringError(
                    f"invalid draft proxy source: {source_solid_id}"
                )
            index = int(source_solid_id[6:]) - 1
            if index < 0 or index >= len(solids):
                raise CollisionComponentAuthoringError(
                    f"draft proxy source is outside STEP: {source_solid_id}"
                )
            shape = solids[index]
            mesh = mesh_shape_m(
                shape,
                scratch / source_solid_id / "source.stl",
            )
            characteristic_lengths.append(float(np.linalg.norm(mesh.extents)))
            expected[(str(body_id), str(source_solid_id))] = (shape, mesh)
        if not expected:
            return ()
        results = load_authored_collision_components(
            plan,
            package=package,
            workspace=workspace,
            expected_sources=expected,
            scratch_root=scratch / "components",
            characteristic_length_m=max(characteristic_lengths),
            mesh_linear_deflection_m=mesh_linear_deflection_m,
            collision_envelope_m=collision_envelope_m,
        )
        return tuple(dict(item.report) for item in results)


def write_source_aware_components(
    package_dir: str | Path,
    *,
    body_id: str,
    source_solid_id: str,
    components: Sequence[tuple[str, Any]],
    replace: bool = False,
) -> dict[str, Any]:
    """Export named convex-source candidates and update the closed manifest.

    This function deliberately does not claim convexity or fidelity.  The
    Harness imports and certifies the written STEP files independently.
    """

    package = Path(package_dir).resolve()
    if not package.is_dir():
        raise CollisionComponentAuthoringError("package directory is missing")
    owner = _identifier(body_id, "body_id")
    solid_id = _source_solid_id(source_solid_id)
    if len(components) < 2:
        raise CollisionComponentAuthoringError(
            "a nonconvex source needs at least two convex components"
        )
    names = [_identifier(item[0], "component_id") for item in components]
    if len(set(names)) != len(names):
        raise CollisionComponentAuthoringError("component IDs must be unique")

    document = _manifest(package)
    existing = [
        item
        for item in document["sources"]
        if isinstance(item, dict)
        and item.get("source_solid_id") == solid_id
    ]
    if existing and not replace:
        raise CollisionComponentAuthoringError(
            f"{solid_id} already has authored collision components"
        )

    rows: list[dict[str, str]] = []
    for component_id, (_raw_name, shape) in zip(names, components, strict=True):
        relative = (
            Path("collision_components") / solid_id / f"{component_id}.step"
        )
        destination = package / relative
        if destination.exists() and not replace:
            raise CollisionComponentAuthoringError(
                f"collision component already exists: {relative.as_posix()}"
            )
        _write_step(destination, shape)
        rows.append(
            {
                "component_id": component_id,
                "step_file": relative.as_posix(),
                "sha256": sha256_file(destination),
            }
        )

    sources = [
        item
        for item in document["sources"]
        if not (
            isinstance(item, dict)
            and item.get("source_solid_id") == solid_id
        )
    ]
    source = {
        "body_id": owner,
        "source_solid_id": solid_id,
        "components": rows,
    }
    sources.append(source)
    sources.sort(key=lambda item: str(item.get("source_solid_id") or ""))
    updated = {
        "schema_version": COLLISION_COMPONENT_MANIFEST_SCHEMA,
        "sources": sources,
    }
    write_json_atomic(package / COLLISION_COMPONENT_MANIFEST_PATH, updated)
    return source


def convex_hull_brep(vertices_m: Any) -> Any:
    """Convert convex mesh vertices in metres to one valid polyhedral BRep."""

    import numpy as np
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing,
    )
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.TopoDS import TopoDS, TopoDS_Solid
    from OCP.gp import gp_Pnt
    from scipy.spatial import ConvexHull, QhullError

    points_m = np.asarray(vertices_m, dtype=float)
    if (
        points_m.ndim != 2
        or points_m.shape[1:] != (3,)
        or len(points_m) < 4
        or not np.all(np.isfinite(points_m))
    ):
        raise CollisionComponentAuthoringError(
            "convex hull vertices must be a finite Nx3 array"
        )
    points_mm = points_m * 1_000.0
    try:
        hull = ConvexHull(points_mm)
    except QhullError as exc:
        raise CollisionComponentAuthoringError(
            "component vertices do not form a three-dimensional hull"
        ) from exc
    sewing = BRepBuilderAPI_Sewing(1.0e-7, True, True, True, False)
    for simplex, equation in zip(
        hull.simplices, hull.equations, strict=True
    ):
        triangle = points_mm[simplex]
        normal = np.cross(
            triangle[1] - triangle[0], triangle[2] - triangle[0]
        )
        if float(np.dot(normal, equation[:3])) < 0.0:
            triangle = triangle[[0, 2, 1]]
        polygon = BRepBuilderAPI_MakePolygon()
        for point in triangle:
            polygon.Add(gp_Pnt(*(float(value) for value in point)))
        polygon.Close()
        sewing.Add(BRepBuilderAPI_MakeFace(polygon.Wire()).Face())
    sewing.Perform()
    shell = TopoDS.Shell_s(sewing.SewedShape())
    builder = BRep_Builder()
    solid = TopoDS_Solid()
    builder.MakeSolid(solid)
    builder.Add(solid, shell)
    fixed = ShapeFix_Solid(solid)
    fixed.Perform()
    result = fixed.Solid()
    if len(shape_solids(result)) != 1:
        raise CollisionComponentAuthoringError(
            "convex hull could not be converted to one BRep solid"
        )
    return result


def orthogonal_brep_boxes(shape: Any) -> tuple[list[Any], dict[str, Any]]:
    """Partition one axis-aligned planar BRep into disjoint exact boxes.

    Every planar face coordinate defines a grid boundary.  Occupied cells are
    classified against the source solid, then greedily merged into larger
    disjoint boxes.  This is exact for the common box/cut/fuse CAD subset and
    fails closed for curved or rotated faces.
    """

    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.GeomAbs import GeomAbs_Plane
    from OCP.TopAbs import TopAbs_FACE, TopAbs_IN, TopAbs_ON
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from OCP.gp import gp_Pnt

    solids = shape_solids(shape)
    if len(solids) != 1:
        raise CollisionComponentAuthoringError(
            "orthogonal decomposition requires exactly one source solid"
        )
    solid = solids[0]
    coordinates: list[set[float]] = [set(), set(), set()]
    explorer = TopExp_Explorer(solid, TopAbs_FACE)
    face_count = 0
    while explorer.More():
        face_count += 1
        surface = BRepAdaptor_Surface(TopoDS.Face_s(explorer.Current()))
        if surface.GetType() != GeomAbs_Plane:
            raise CollisionComponentAuthoringError(
                "orthogonal decomposition requires planar faces"
            )
        plane = surface.Plane()
        direction = plane.Axis().Direction()
        magnitudes = [
            abs(float(direction.X())),
            abs(float(direction.Y())),
            abs(float(direction.Z())),
        ]
        axis = max(range(3), key=magnitudes.__getitem__)
        if magnitudes[axis] < 1.0 - 1.0e-9 or any(
            magnitudes[index] > 1.0e-9
            for index in range(3)
            if index != axis
        ):
            raise CollisionComponentAuthoringError(
                "orthogonal decomposition requires axis-aligned faces"
            )
        location = plane.Location()
        coordinate = (
            float(location.X()),
            float(location.Y()),
            float(location.Z()),
        )[axis]
        coordinates[axis].add(round(coordinate, 9))
        explorer.Next()
    values = [sorted(item) for item in coordinates]
    if face_count < 4 or any(len(item) < 2 for item in values):
        raise CollisionComponentAuthoringError(
            "orthogonal source does not define a closed coordinate grid"
        )
    cell_budget = math.prod(len(item) - 1 for item in values)
    if cell_budget > 100_000:
        raise CollisionComponentAuthoringError(
            "orthogonal decomposition grid exceeds 100000 cells"
        )

    occupied: set[tuple[int, int, int]] = set()
    for x_index in range(len(values[0]) - 1):
        for y_index in range(len(values[1]) - 1):
            for z_index in range(len(values[2]) - 1):
                midpoint = (
                    0.5
                    * (values[0][x_index] + values[0][x_index + 1]),
                    0.5
                    * (values[1][y_index] + values[1][y_index + 1]),
                    0.5
                    * (values[2][z_index] + values[2][z_index + 1]),
                )
                classifier = BRepClass3d_SolidClassifier(
                    solid,
                    gp_Pnt(*midpoint),
                    1.0e-7,
                )
                if classifier.State() in (TopAbs_IN, TopAbs_ON):
                    occupied.add((x_index, y_index, z_index))
    if not occupied:
        raise CollisionComponentAuthoringError(
            "orthogonal decomposition found no occupied cells"
        )

    remaining = set(occupied)
    blocks: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    while remaining:
        lower = list(min(remaining))
        upper = [item + 1 for item in lower]
        for axis in (0, 1, 2):
            while upper[axis] < len(values[axis]) - 1:
                candidate = upper[axis]
                ranges = [
                    range(lower[index], upper[index]) for index in range(3)
                ]
                ranges[axis] = range(candidate, candidate + 1)
                if not all(
                    (x_index, y_index, z_index) in remaining
                    for x_index in ranges[0]
                    for y_index in ranges[1]
                    for z_index in ranges[2]
                ):
                    break
                upper[axis] += 1
        cells = {
            (x_index, y_index, z_index)
            for x_index in range(lower[0], upper[0])
            for y_index in range(lower[1], upper[1])
            for z_index in range(lower[2], upper[2])
        }
        remaining.difference_update(cells)
        blocks.append((tuple(lower), tuple(upper)))

    boxes: list[Any] = []
    for lower, upper in blocks:
        minimum = tuple(values[axis][lower[axis]] for axis in range(3))
        maximum = tuple(values[axis][upper[axis]] for axis in range(3))
        lengths = tuple(
            maximum[axis] - minimum[axis] for axis in range(3)
        )
        if any(length <= 0.0 for length in lengths):
            raise CollisionComponentAuthoringError(
                "orthogonal decomposition produced a degenerate box"
            )
        boxes.append(
            BRepPrimAPI_MakeBox(gp_Pnt(*minimum), *lengths).Shape()
        )
    return boxes, {
        "backend": "orthogonal_brep_grid_v1",
        "face_count": face_count,
        "grid_cell_count": cell_budget,
        "occupied_cell_count": len(occupied),
        "component_count": len(boxes),
    }


def _coacd_cache_key(mesh: Any, preset_name: str, preset: CoacdPreset) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(
        canonical_sha256(
            {
                "backend": "coacd",
                "preset": preset_name,
                "parameters": preset.__dict__,
                "seed": 0,
            }
        ).encode("ascii")
    )
    digest.update(np.asarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.asarray(mesh.faces, dtype=np.int32).tobytes())
    return digest.hexdigest()


def coacd_convex_hulls(
    mesh: Any,
    *,
    preset_name: str,
    cache_dir: str | Path | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Return deterministic CoACD hull BReps and cache the expensive proposal."""

    import numpy as np

    try:
        import coacd
    except ImportError as exc:
        raise CollisionComponentAuthoringError(
            "CoACD is not installed; source-aware components are still available"
        ) from exc
    preset = COACD_PRESETS.get(preset_name)
    if preset is None:
        raise CollisionComponentAuthoringError(
            f"unknown CoACD preset {preset_name!r}"
        )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    key = _coacd_cache_key(mesh, preset_name, preset)
    cache_path = (
        Path(cache_dir).resolve() / f"{key}.npz"
        if cache_dir is not None
        else None
    )
    cached = False
    parts: list[tuple[Any, Any]] = []
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            count = int(archive["count"][0])
            parts = [
                (
                    archive[f"vertices_{index}"].copy(),
                    archive[f"faces_{index}"].copy(),
                )
                for index in range(count)
            ]
        cached = True
    else:
        coacd.set_log_level("off")
        raw = coacd.run_coacd(
            coacd.Mesh(vertices, faces),
            threshold=preset.threshold_m,
            max_convex_hull=preset.max_convex_hulls,
            preprocess_mode="auto",
            resolution=preset.resolution,
            mcts_nodes=20,
            mcts_iterations=preset.mcts_iterations,
            mcts_max_depth=preset.mcts_max_depth,
            merge=True,
            decimate=False,
            max_ch_vertex=128,
            seed=0,
            real_metric=True,
        )
        parts = [
            (
                np.asarray(item[0], dtype=np.float64),
                np.asarray(item[1], dtype=np.int32),
            )
            for item in raw
        ]
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "count": np.asarray((len(parts),), dtype=np.int32)
            }
            for index, (part_vertices, part_faces) in enumerate(parts):
                payload[f"vertices_{index}"] = part_vertices
                payload[f"faces_{index}"] = part_faces
            np.savez_compressed(cache_path, **payload)
    if len(parts) < 2:
        raise CollisionComponentAuthoringError(
            "CoACD did not produce a nontrivial convex decomposition"
        )
    shapes = [convex_hull_brep(item[0]) for item in parts]
    return shapes, {
        "backend": "coacd",
        "preset": preset_name,
        "component_count": len(shapes),
        "cache_key": key,
        "cache_hit": cached,
    }


__all__ = [
    "COACD_PRESETS",
    "CoacdPreset",
    "CollisionComponentAuthoringError",
    "coacd_convex_hulls",
    "convex_hull_brep",
    "orthogonal_brep_boxes",
    "write_source_aware_components",
]
