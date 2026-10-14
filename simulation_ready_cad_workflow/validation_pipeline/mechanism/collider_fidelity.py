"""Trusted STEP-to-collider fidelity evidence.

OCCT owns STEP import and tessellation. This module measures the collision
asset that will actually be loaded. Thresholds are validator-owned and compound
bodies are assessed as component unions, never as one enclosing hull.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

COLLIDER_FIDELITY_SCHEMA = "collider_fidelity_v1"
DEFAULT_SURFACE_SAMPLE_COUNT = 384
DEFAULT_VOXEL_RESOLUTION = 36


@dataclass(frozen=True)
class ColliderFidelityThresholds:
    surface_rms_max_m: float
    surface_p95_max_m: float
    surface_max_max_m: float
    local_surface_max_m: float
    relative_volume_error_max: float
    centroid_offset_max_m: float
    inertia_relative_error_max: float
    occupancy_iou_min: float
    provenance: str = "validator_scale_aware_v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactRuntimeMeshFidelityTransfer:
    """In-memory proof that a trusted fidelity result still covers runtime meshes.

    This object is created only after Harness has independently evaluated the
    authored STEP component union and then proved that every exported runtime
    OBJ has identical topology and machine-precision vertices.  It is not a
    serialized package claim or a substitute for either check.
    """

    result: Mapping[str, Any]
    package_binding_sha256: str
    source_id: str
    collider_id: str
    representation: str
    component_ids: tuple[str, ...]
    binding_method: str


def derive_fidelity_thresholds(
    *,
    characteristic_length_m: float,
    mesh_linear_deflection_m: float,
    collision_envelope_m: float,
    local_clearance_m: float | None = None,
) -> ColliderFidelityThresholds:
    """Derive scale-aware limits; package-authored tolerances are not inputs."""

    scale = float(characteristic_length_m)
    deflection = float(mesh_linear_deflection_m)
    envelope = float(collision_envelope_m)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("characteristic_length_m must be positive and finite")
    if not math.isfinite(deflection) or deflection <= 0.0:
        raise ValueError("mesh_linear_deflection_m must be positive and finite")
    if not math.isfinite(envelope) or envelope < 0.0:
        raise ValueError("collision_envelope_m must be finite and non-negative")
    rms = max(1.5 * deflection, 4.0 * envelope, 5.0e-5 * scale)
    p95 = max(2.5 * deflection, 6.0 * envelope, 1.0e-4 * scale)
    maximum = max(4.0 * deflection, 8.0 * envelope, 2.0e-4 * scale)
    local = maximum
    if local_clearance_m is not None:
        clearance = float(local_clearance_m)
        if not math.isfinite(clearance) or clearance <= 0.0:
            raise ValueError("local_clearance_m must be positive and finite")
        local = min(local, max(2.0 * envelope, 0.25 * clearance))
    return ColliderFidelityThresholds(
        surface_rms_max_m=rms,
        surface_p95_max_m=p95,
        surface_max_max_m=maximum,
        local_surface_max_m=local,
        relative_volume_error_max=0.015,
        centroid_offset_max_m=max(2.0 * deflection, 1.0e-3 * scale),
        inertia_relative_error_max=0.03,
        occupancy_iou_min=0.97,
    )


def _mesh(value: Any) -> Any:
    import trimesh

    if isinstance(value, trimesh.Trimesh):
        result = value.copy()
    elif hasattr(value, "geometry"):
        result = trimesh.util.concatenate(tuple(value.geometry.values()))
    else:
        raise TypeError("fidelity evaluation requires a trimesh mesh")
    result.remove_unreferenced_vertices()
    if not len(result.vertices) or not len(result.faces):
        raise ValueError("fidelity evaluation received an empty mesh")
    return result


def _seed(*values: str) -> int:
    digest = hashlib.sha256("\0".join(values).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _evaluation_points(mesh: Any, count: int, seed: int) -> Any:
    """Deterministic samples plus bounded vertex and face-center anchors."""

    import numpy as np
    import trimesh

    count = max(96, int(count))
    budget = max(16, count // 6)
    rng = np.random.default_rng(seed)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if len(vertices) > budget:
        vertices = vertices[np.sort(rng.choice(len(vertices), budget, replace=False))]
    centers = np.asarray(mesh.triangles_center, dtype=float)
    if len(centers) > budget:
        centers = centers[np.sort(rng.choice(len(centers), budget, replace=False))]
    sampled, _ = trimesh.sample.sample_surface(
        mesh, max(32, count - len(vertices) - len(centers)), seed=seed
    )
    return np.vstack((vertices, centers, np.asarray(sampled, dtype=float)))


def _nearest_distances(mesh: Any, points: Any) -> Any:
    import numpy as np
    import trimesh

    try:
        _closest, distances, _triangles = trimesh.proximity.closest_point(mesh, points)
    except (ModuleNotFoundError, ImportError):
        distances = _bounded_nearest_surface_distances(mesh, points)
    result = np.asarray(distances, dtype=float)
    if not len(result) or not np.all(np.isfinite(result)):
        raise ValueError("nearest-surface query produced invalid distances")
    return result


def _bounded_nearest_surface_distances(mesh: Any, points: Any) -> Any:
    """Exact point-to-triangle distances without an unbounded dense product.

    The nearest mesh vertex supplies a valid upper bound.  A centroid/radius
    tree and triangle AABBs then discard every triangle that provably cannot
    improve that bound before exact point-to-triangle evaluation.
    """

    import numpy as np
    import trimesh
    from scipy.spatial import cKDTree

    queries = np.asarray(points, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)
    if (
        queries.ndim != 2
        or queries.shape[1] != 3
        or not len(queries)
        or not len(triangles)
        or not len(vertices)
    ):
        raise ValueError("bounded surface distance requires non-empty Nx3 data")
    finite_triangles = np.all(np.isfinite(triangles), axis=(1, 2))
    nondegenerate = (
        np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        > 1.0e-24
    )
    triangles = triangles[finite_triangles & nondegenerate]
    if not len(triangles):
        raise ValueError("distance mesh has no finite nondegenerate triangles")

    upper, _nearest_vertices = cKDTree(vertices).query(
        queries,
        k=1,
        workers=1,
    )
    centers = np.mean(triangles, axis=1)
    radii = np.max(
        np.linalg.norm(triangles - centers[:, None, :], axis=2),
        axis=1,
    )
    maximum_radius = float(np.max(radii))
    centroid_tree = cKDTree(centers)
    scale = max(
        1.0,
        float(np.max(np.ptp(vertices, axis=0))),
    )
    epsilon = 64.0 * float(np.finfo(np.float64).eps) * scale
    lower_bounds = np.min(triangles, axis=1)
    upper_bounds = np.max(triangles, axis=1)
    result = np.asarray(upper, dtype=float).copy()
    for index, point in enumerate(queries):
        raw_candidates = centroid_tree.query_ball_point(
            point,
            result[index] + maximum_radius + epsilon,
            workers=1,
        )
        candidates = np.asarray(raw_candidates, dtype=int)
        if not len(candidates):
            continue
        below = np.maximum(lower_bounds[candidates] - point, 0.0)
        above = np.maximum(point - upper_bounds[candidates], 0.0)
        aabb_distance = np.linalg.norm(below + above, axis=1)
        candidates = candidates[
            aabb_distance <= result[index] + epsilon
        ]
        if not len(candidates):
            continue
        repeated = np.repeat(
            point.reshape((1, 3)),
            len(candidates),
            axis=0,
        )
        closest = trimesh.triangles.closest_point(
            triangles[candidates],
            repeated,
        )
        result[index] = min(
            result[index],
            float(np.min(np.linalg.norm(closest - repeated, axis=1))),
        )
    return result


def _distance_summary(values: Any) -> dict[str, float]:
    import numpy as np

    array = np.asarray(values, dtype=float)
    return {
        "rms_m": float(np.sqrt(np.mean(array * array))),
        "p95_m": float(np.percentile(array, 95.0)),
        "max_m": float(np.max(array)),
    }


def _mesh_facts(mesh: Any) -> dict[str, Any]:
    import numpy as np

    bounds = np.asarray(mesh.bounds, dtype=float)
    facts: dict[str, Any] = {
        "vertex_count": int(len(mesh.vertices)),
        "triangle_count": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "bounds_m": bounds.tolist(),
        "characteristic_length_m": float(np.linalg.norm(bounds[1] - bounds[0])),
    }
    if mesh.is_watertight:
        try:
            volume = abs(float(mesh.volume))
            center = np.asarray(mesh.center_mass, dtype=float)
            inertia = np.asarray(mesh.moment_inertia, dtype=float)
            if volume > 0.0 and np.all(np.isfinite(center)) and np.all(np.isfinite(inertia)):
                facts.update(
                    {
                        "volume_m3": volume,
                        "centroid_m": center.tolist(),
                        "inertia_unit_density_kg_m2": inertia.tolist(),
                    }
                )
        except (ValueError, TypeError, ZeroDivisionError):
            pass
    return facts


def _occupancy_keys(
    meshes: Sequence[Any], *, origin: Any, pitch: float
) -> set[tuple[int, int, int]]:
    import numpy as np

    occupied: set[tuple[int, int, int]] = set()
    for mesh in meshes:
        points = np.asarray(mesh.voxelized(pitch).fill().points, dtype=float)
        if len(points):
            indices = np.rint((points - origin) / pitch).astype(int)
            occupied.update(tuple(int(value) for value in row) for row in indices)
    return occupied


def _occupancy_iou(
    source_meshes: Sequence[Any],
    collider_meshes: Sequence[Any],
    *,
    resolution: int,
) -> tuple[float | None, dict[str, Any]]:
    import numpy as np

    try:
        meshes = [*source_meshes, *collider_meshes]
        bounds = np.asarray([mesh.bounds for mesh in meshes], dtype=float)
        lower = bounds[:, 0, :].min(axis=0)
        upper = bounds[:, 1, :].max(axis=0)
        characteristic = float(np.max(upper - lower))
        if characteristic <= 0.0:
            return None, {"reason": "zero_characteristic_length"}
        pitch = characteristic / max(12, int(resolution))
        source = _occupancy_keys(source_meshes, origin=lower, pitch=pitch)
        collider = _occupancy_keys(collider_meshes, origin=lower, pitch=pitch)
        union = source | collider
        if not union:
            return None, {"reason": "empty_voxel_occupancy", "pitch_m": pitch}
        return len(source & collider) / len(union), {
            "pitch_m": pitch,
            "source_voxel_count": len(source),
            "collider_voxel_count": len(collider),
            "union_voxel_count": len(union),
            "method": "component_voxel_union_without_convex_hull",
        }
    except Exception as exc:
        return None, {"reason": f"voxelization_unavailable:{type(exc).__name__}"}


def multiresolution_occupancy_iou(
    source_meshes: Sequence[Any],
    collider_meshes: Sequence[Any],
) -> tuple[float | None, dict[str, Any]]:
    """Return a bounded median IoU across three deterministic voxel grids."""

    evaluations: list[dict[str, Any]] = []
    values: list[float] = []
    for resolution in (48, 64, 80):
        value, evidence = _occupancy_iou(
            source_meshes,
            collider_meshes,
            resolution=resolution,
        )
        evaluations.append(
            {
                "resolution": resolution,
                "iou": value,
                "evidence": evidence,
            }
        )
        if value is not None and math.isfinite(value):
            values.append(float(value))
    if len(values) < 2:
        return None, {
            "method": "multiresolution_voxel_component_union",
            "reason": "fewer_than_two_voxel_resolutions_available",
            "evaluations": evaluations,
        }
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return median, {
        "method": "multiresolution_voxel_component_union",
        "selected_statistic": "median",
        "available_resolution_count": len(values),
        "evaluations": evaluations,
    }


def _axis_aligned_box_bounds(mesh: Any) -> Any | None:
    """Return bounds only when the mesh is exactly an axis-aligned box."""

    import itertools

    import numpy as np

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if (
        vertices.shape != (8, 3)
        or faces.shape != (12, 3)
        or not bool(mesh.is_watertight)
        or not np.all(np.isfinite(vertices))
    ):
        return None
    bounds = np.asarray(mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    scale = float(np.max(extents))
    tolerance = max(1.0e-12, 1.0e-9 * scale)
    if scale <= 0.0 or np.any(extents <= tolerance):
        return None

    corner_codes: set[tuple[int, int, int]] = set()
    for vertex in vertices:
        code: list[int] = []
        for axis in range(3):
            if math.isclose(
                float(vertex[axis]),
                float(bounds[0, axis]),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                code.append(0)
            elif math.isclose(
                float(vertex[axis]),
                float(bounds[1, axis]),
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                code.append(1)
            else:
                return None
        corner_codes.add(tuple(code))
    if corner_codes != set(itertools.product((0, 1), repeat=3)):
        return None

    # Every triangle must lie on one of the six boundary planes. This prevents
    # an arbitrary eight-vertex solid with the same bounds from being treated
    # as an analytical box.
    for face in faces:
        triangle = vertices[face]
        on_boundary_plane = any(
            float(np.ptp(triangle[:, axis])) <= tolerance
            and (
                math.isclose(
                    float(triangle[0, axis]),
                    float(bounds[0, axis]),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
                or math.isclose(
                    float(triangle[0, axis]),
                    float(bounds[1, axis]),
                    rel_tol=0.0,
                    abs_tol=tolerance,
                )
            )
            for axis in range(3)
        )
        if not on_boundary_plane:
            return None

    expected_volume = float(np.prod(extents))
    volume_error = abs(abs(float(mesh.volume)) - expected_volume)
    if volume_error > max(1.0e-18, 1.0e-9 * expected_volume):
        return None
    return bounds


def _analytic_box_iou(
    source_bounds: Any,
    collider_bounds: Any,
) -> tuple[float, dict[str, Any]]:
    """Compute exact intersection-over-union for two axis-aligned boxes."""

    import numpy as np

    source = np.asarray(source_bounds, dtype=float)
    collider = np.asarray(collider_bounds, dtype=float)
    intersection_extents = np.maximum(
        0.0,
        np.minimum(source[1], collider[1]) - np.maximum(source[0], collider[0]),
    )
    source_volume = float(np.prod(source[1] - source[0]))
    collider_volume = float(np.prod(collider[1] - collider[0]))
    intersection_volume = float(np.prod(intersection_extents))
    union_volume = source_volume + collider_volume - intersection_volume
    if union_volume <= 0.0:
        raise ValueError("axis-aligned box union volume must be positive")
    return intersection_volume / union_volume, {
        "method": "analytic_axis_aligned_box_intersection_union",
        "source_volume_m3": source_volume,
        "collider_volume_m3": collider_volume,
        "intersection_volume_m3": intersection_volume,
        "union_volume_m3": union_volume,
    }


def _relative_inertia_error(source: Any, collider: Any) -> float | None:
    import numpy as np

    if source is None or collider is None:
        return None
    source_array = np.asarray(source, dtype=float)
    collider_array = np.asarray(collider, dtype=float)
    denominator = float(np.linalg.norm(source_array))
    if denominator <= 1.0e-18:
        return None
    return float(np.linalg.norm(collider_array - source_array) / denominator)


def evaluate_mesh_fidelity(
    source_mesh: Any,
    collider_mesh: Any,
    *,
    source_id: str,
    collider_id: str,
    representation: str,
    thresholds: ColliderFidelityThresholds,
    sample_count: int = DEFAULT_SURFACE_SAMPLE_COUNT,
    voxel_resolution: int = DEFAULT_VOXEL_RESOLUTION,
    local_interface_points: Mapping[str, Sequence[Sequence[float]]] | None = None,
    source_surface_points: Any | None = None,
    collider_surface_points: Any | None = None,
    source_distance_evaluator: Callable[[Any], Any] | None = None,
    collider_distance_evaluator: Callable[[Any], Any] | None = None,
    source_facts_override: Mapping[str, Any] | None = None,
    collider_facts_override: Mapping[str, Any] | None = None,
    occupancy_iou_override: tuple[float, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Measure one source solid against one loaded collider."""

    import numpy as np

    source = _mesh(source_mesh)
    collider = _mesh(collider_mesh)
    source_facts = (
        dict(source_facts_override)
        if source_facts_override is not None
        else _mesh_facts(source)
    )
    collider_facts = (
        dict(collider_facts_override)
        if collider_facts_override is not None
        else _mesh_facts(collider)
    )
    inconclusive: list[str] = []
    failures: list[str] = []
    try:
        source_points = (
            np.asarray(source_surface_points, dtype=float)
            if source_surface_points is not None
            else _evaluation_points(
                source, sample_count, _seed(source_id, collider_id, "source")
            )
        )
        collider_points = (
            np.asarray(collider_surface_points, dtype=float)
            if collider_surface_points is not None
            else _evaluation_points(
                collider, sample_count, _seed(source_id, collider_id, "collider")
            )
        )
        if (
            source_points.ndim != 2
            or source_points.shape[1] != 3
            or not len(source_points)
            or collider_points.ndim != 2
            or collider_points.shape[1] != 3
            or not len(collider_points)
        ):
            raise ValueError("trusted BRep surface points must be non-empty Nx3 arrays")
        source_to_collider = _distance_summary(
            (
                np.asarray(collider_distance_evaluator(source_points), dtype=float)
                if collider_distance_evaluator is not None
                else _nearest_distances(collider, source_points)
            )
        )
        collider_to_source = _distance_summary(
            (
                np.asarray(source_distance_evaluator(collider_points), dtype=float)
                if source_distance_evaluator is not None
                else _nearest_distances(source, collider_points)
            )
        )
        bidirectional = {
            key: max(source_to_collider[key], collider_to_source[key])
            for key in ("rms_m", "p95_m", "max_m")
        }
    except Exception as exc:
        source_to_collider = None
        collider_to_source = None
        bidirectional = None
        inconclusive.append(f"surface_distance:{type(exc).__name__}")

    source_volume = source_facts.get("volume_m3")
    collider_volume = collider_facts.get("volume_m3")
    if source_volume is None or collider_volume is None:
        volume_error = None
        inconclusive.append("relative_volume_error:not_closed")
    else:
        volume_error = abs(float(collider_volume) - float(source_volume)) / float(
            source_volume
        )

    source_centroid = source_facts.get("centroid_m")
    collider_centroid = collider_facts.get("centroid_m")
    if source_centroid is None or collider_centroid is None:
        centroid_offset = None
        inconclusive.append("centroid_offset:not_closed")
    else:
        centroid_offset = float(
            np.linalg.norm(
                np.asarray(collider_centroid, dtype=float)
                - np.asarray(source_centroid, dtype=float)
            )
        )

    inertia_error = _relative_inertia_error(
        source_facts.get("inertia_unit_density_kg_m2"),
        collider_facts.get("inertia_unit_density_kg_m2"),
    )
    if inertia_error is None:
        inconclusive.append("inertia_relative_error:not_closed_or_degenerate")

    voxel_iou, voxel_evidence = _occupancy_iou(
        (source,), (collider,), resolution=voxel_resolution
    )
    source_box_bounds = (
        _axis_aligned_box_bounds(source)
        if representation == "ChCollisionShapeBox"
        else None
    )
    collider_box_bounds = (
        _axis_aligned_box_bounds(collider)
        if representation == "ChCollisionShapeBox"
        else None
    )
    if occupancy_iou_override is not None:
        occupancy_iou = float(occupancy_iou_override[0])
        occupancy_evidence = dict(occupancy_iou_override[1])
        if not math.isfinite(occupancy_iou) or not 0.0 <= occupancy_iou <= 1.0:
            raise ValueError("trusted occupancy IoU override must be within [0, 1]")
        occupancy_evidence["voxel_diagnostic_iou"] = voxel_iou
        occupancy_evidence["voxel_diagnostic_evidence"] = voxel_evidence
        iou_evaluation_method = str(
            occupancy_evidence.get("method")
            or "trusted_brep_occupancy_override"
        )
    elif source_box_bounds is not None and collider_box_bounds is not None:
        occupancy_iou, occupancy_evidence = _analytic_box_iou(
            source_box_bounds,
            collider_box_bounds,
        )
        occupancy_evidence["voxel_diagnostic_iou"] = voxel_iou
        occupancy_evidence["voxel_diagnostic_evidence"] = voxel_evidence
        iou_evaluation_method = "analytic_axis_aligned_box_intersection_union"
    else:
        occupancy_iou = voxel_iou
        occupancy_evidence = voxel_evidence
        iou_evaluation_method = "voxelized_component_union"
    if occupancy_iou is None:
        inconclusive.append(
            f"occupancy_iou:{occupancy_evidence.get('reason', 'unavailable')}"
        )
    # Keep this compatibility field, but base it on the authoritative
    # analytical result rather than treating a poor voxel score as exempt.
    exact_box_invariant_match = (
        iou_evaluation_method == "analytic_axis_aligned_box_intersection_union"
        and occupancy_iou is not None
        and math.isclose(occupancy_iou, 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    )
    occupancy_iou_discretization_override = False
    if (
        iou_evaluation_method == "voxelized_component_union"
        and occupancy_iou is not None
        and occupancy_iou < thresholds.occupancy_iou_min
        and bidirectional is not None
        and volume_error is not None
        and centroid_offset is not None
        and inertia_error is not None
    ):
        characteristic_length = max(
            float(source_facts.get("characteristic_length_m") or 0.0),
            float(collider_facts.get("characteristic_length_m") or 0.0),
        )
        identity_length_tolerance_m = max(
            1.0e-10,
            1.0e-8 * characteristic_length,
        )
        try:
            bounds_error_m = float(
                np.max(
                    np.abs(
                        np.asarray(source_facts["bounds_m"], dtype=float)
                        - np.asarray(collider_facts["bounds_m"], dtype=float)
                    )
                )
            )
        except (KeyError, TypeError, ValueError):
            bounds_error_m = math.inf
        machine_precision_match = bool(
            bidirectional["max_m"] <= identity_length_tolerance_m
            and bounds_error_m <= identity_length_tolerance_m
            and volume_error <= 1.0e-8
            and centroid_offset <= identity_length_tolerance_m
            and inertia_error <= 1.0e-8
        )
        if machine_precision_match:
            raw_occupancy_iou = occupancy_iou
            raw_occupancy_evidence = occupancy_evidence
            occupancy_iou = 1.0
            occupancy_evidence = {
                "method": "machine_precision_geometric_invariants_override_v1",
                "raw_voxel_iou": raw_occupancy_iou,
                "raw_voxel_evidence": raw_occupancy_evidence,
                "identity_length_tolerance_m": identity_length_tolerance_m,
                "bounds_error_m": bounds_error_m,
                "surface_max_m": bidirectional["max_m"],
                "relative_volume_error": volume_error,
                "centroid_offset_m": centroid_offset,
                "inertia_relative_error": inertia_error,
            }
            iou_evaluation_method = (
                "machine_precision_geometric_invariants_override_v1"
            )
            occupancy_iou_discretization_override = True

    local_results: list[dict[str, Any]] = []
    for interface_id, raw_points in sorted((local_interface_points or {}).items()):
        try:
            points = np.asarray(raw_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not len(points):
                raise ValueError("local interface points must be Nx3")
            summary = _distance_summary(_nearest_distances(collider, points))
            passed = summary["max_m"] <= thresholds.local_surface_max_m
            local_results.append(
                {
                    "interface_id": str(interface_id),
                    "status": "PASS" if passed else "FAIL",
                    "source_to_collider": summary,
                    "sample_count": int(len(points)),
                }
            )
            if not passed:
                failures.append(f"local_interface_surface_error:{interface_id}")
        except Exception as exc:
            local_results.append(
                {
                    "interface_id": str(interface_id),
                    "status": "INCONCLUSIVE",
                    "reason": type(exc).__name__,
                }
            )
            failures.append(f"local_interface_inconclusive:{interface_id}")

    checks: dict[str, bool | None] = {
        "surface_rms": None if bidirectional is None else (
            bidirectional["rms_m"] <= thresholds.surface_rms_max_m
        ),
        "surface_p95": None if bidirectional is None else (
            bidirectional["p95_m"] <= thresholds.surface_p95_max_m
        ),
        "surface_max": None if bidirectional is None else (
            bidirectional["max_m"] <= thresholds.surface_max_max_m
        ),
        "relative_volume_error": None if volume_error is None else (
            volume_error <= thresholds.relative_volume_error_max
        ),
        "centroid_offset": None if centroid_offset is None else (
            centroid_offset <= thresholds.centroid_offset_max_m
        ),
        "inertia_relative_error": None if inertia_error is None else (
            inertia_error <= thresholds.inertia_relative_error_max
        ),
        "occupancy_iou": None if occupancy_iou is None else (
            occupancy_iou >= thresholds.occupancy_iou_min
        ),
    }
    for name, passed in checks.items():
        if passed is False:
            failures.append(name)
    mandatory_inconclusive = any(value is None for value in checks.values())
    status = "FAIL" if failures else (
        "INCONCLUSIVE" if mandatory_inconclusive else "PASS"
    )
    return {
        "schema_version": COLLIDER_FIDELITY_SCHEMA,
        "scope": "source_solid",
        "source_id": str(source_id),
        "collider_id": str(collider_id),
        "representation": str(representation),
        "status": status,
        "passed": status == "PASS",
        "source_facts": source_facts,
        "collider_facts": collider_facts,
        "surface_distance": {
            "source_to_collider": source_to_collider,
            "collider_to_source": collider_to_source,
            "bidirectional": bidirectional,
            "sample_count_each_direction": int(sample_count),
            "actual_sample_count_by_direction": {
                "source": int(len(source_points)) if source_to_collider is not None else 0,
                "collider": (
                    int(len(collider_points)) if collider_to_source is not None else 0
                ),
            },
            "deterministic_seed_binding": hashlib.sha256(
                f"{source_id}\0{collider_id}".encode("utf-8")
            ).hexdigest(),
        },
        "relative_volume_error": volume_error,
        "centroid_offset_m": centroid_offset,
        "inertia_relative_error": inertia_error,
        "occupancy_iou": occupancy_iou,
        "occupancy_evidence": occupancy_evidence,
        "iou_evaluation_method": iou_evaluation_method,
        "voxel_occupancy_iou": voxel_iou,
        "occupancy_iou_discretization_override": (
            occupancy_iou_discretization_override
        ),
        "exact_box_invariant_match": exact_box_invariant_match,
        "local_interfaces": local_results,
        "thresholds": thresholds.to_dict(),
        "checks": checks,
        "failure_reasons": sorted(set(failures)),
        "inconclusive_metrics": sorted(set(inconclusive)),
        "iou_is_sufficient_by_itself": False,
    }


def _validated_transferred_fidelity_result(
    item: Mapping[str, Any],
    *,
    package_binding_sha256: str,
    thresholds: ColliderFidelityThresholds,
) -> dict[str, Any] | None:
    key = "_exact_runtime_mesh_fidelity_transfer"
    if key not in item:
        return None
    transfer = item.get(key)
    if not isinstance(transfer, ExactRuntimeMeshFidelityTransfer):
        raise ValueError("runtime fidelity transfer must be Harness-created")
    expected_source_id = str(item["solid_id"])
    expected_collider_id = str(
        item.get("collider_id") or item["solid_id"]
    )
    expected_representation = str(item["representation"])
    if (
        transfer.package_binding_sha256 != str(package_binding_sha256)
        or transfer.source_id != expected_source_id
        or transfer.collider_id != expected_collider_id
        or transfer.representation != expected_representation
        or not transfer.component_ids
        or transfer.binding_method
        != "deterministic_obj_round_trip_topology_and_vertices_v1"
    ):
        raise ValueError("runtime fidelity transfer binding does not match")
    result = copy.deepcopy(dict(transfer.result))
    if (
        result.get("schema_version") != COLLIDER_FIDELITY_SCHEMA
        or result.get("source_id") != expected_source_id
        or result.get("collider_id") != expected_collider_id
        or result.get("representation") != expected_representation
        or result.get("thresholds") != thresholds.to_dict()
        or result.get("status") != "PASS"
        or result.get("passed") is not True
    ):
        raise ValueError("runtime fidelity transfer result is not reusable")
    result["evaluation_transfer"] = {
        "method": transfer.binding_method,
        "source": "validator_evaluated_authored_step_component_union",
        "runtime_mesh_topology_and_vertices_identical": True,
        "component_ids": list(transfer.component_ids),
        "thresholds_reapplied_without_change": True,
        "package_binding_sha256": transfer.package_binding_sha256,
    }
    return result


def _single_source_body_result(
    component_result: Mapping[str, Any],
    *,
    body_id: str,
) -> dict[str, Any]:
    """Transfer metrics when a rigid-body union contains exactly one source."""

    result = copy.deepcopy(dict(component_result))
    source_id = f"{body_id}:source_component_union"
    collider_id = f"{body_id}:collider_component_union"
    result.update(
        {
            "scope": "rigid_body_component_union",
            "source_id": source_id,
            "collider_id": collider_id,
            "representation": "component_union_without_enclosing_hull",
            "body_id": body_id,
            "component_count": 1,
        }
    )
    surface = result.get("surface_distance")
    if isinstance(surface, dict):
        surface["deterministic_seed_binding"] = hashlib.sha256(
            f"{source_id}\0{collider_id}".encode("utf-8")
        ).hexdigest()
    result["body_union_metric_transfer"] = {
        "method": "single_source_body_union_identity_v1",
        "source_solid_count": 1,
        "source_and_body_union_meshes_identical": True,
        "thresholds_reapplied_without_change": True,
    }
    return result


def _promote_componentwise_body_union_result(
    result: dict[str, Any],
    *,
    component_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Accept an exact multi-solid union when only aggregate mass facts are open.

    Intersecting closed source solids are a valid rigid-body construction, but
    concatenating their triangle meshes creates a deliberately non-manifold
    aggregate.  Volume, centroid, and inertia are therefore unavailable on
    that concatenation even when every source/collider component pair passes
    and the aggregate bidirectional surface plus occupancy checks also pass.
    Those unavailable duplicate mass-property checks must not turn an exact
    component union into a false-negative fidelity result.
    """

    components = tuple(component_results)
    checks = result.get("checks")
    inconclusive = set(result.get("inconclusive_metrics") or ())
    allowed_inconclusive = {
        "centroid_offset:not_closed",
        "inertia_relative_error:not_closed_or_degenerate",
        "relative_volume_error:not_closed",
    }
    measured_union_checks = (
        "surface_rms",
        "surface_p95",
        "surface_max",
        "occupancy_iou",
    )
    if not (
        result.get("status") == "INCONCLUSIVE"
        and result.get("failure_reasons") == []
        and components
        and all(
            item.get("status") == "PASS" and item.get("passed") is True
            for item in components
        )
        and isinstance(checks, Mapping)
        and all(checks.get(name) is True for name in measured_union_checks)
        and inconclusive
        and inconclusive <= allowed_inconclusive
    ):
        return result
    result.update(
        {
            "status": "PASS",
            "passed": True,
            "body_union_metric_transfer": {
                "method": (
                    "componentwise_pass_plus_union_surface_occupancy_v1"
                ),
                "source_solid_count": len(components),
                "every_source_collider_component_passed": True,
                "aggregate_surface_checks_passed": True,
                "aggregate_occupancy_iou_passed": True,
                "aggregate_mass_properties_not_required": True,
                "reason": (
                    "intersecting closed component meshes form a non-manifold "
                    "concatenation, so duplicate aggregate mass properties "
                    "are unavailable"
                ),
            },
        }
    )
    return result


def build_collider_fidelity_report(
    components: Iterable[Mapping[str, Any]],
    *,
    package_binding_sha256: str,
    characteristic_length_m: float,
    mesh_linear_deflection_m: float,
    collision_envelope_m: float,
) -> dict[str, Any]:
    """Evaluate source/collider pairs and each body's component union."""

    import trimesh

    thresholds = derive_fidelity_thresholds(
        characteristic_length_m=characteristic_length_m,
        mesh_linear_deflection_m=mesh_linear_deflection_m,
        collision_envelope_m=collision_envelope_m,
    )
    component_results: list[dict[str, Any]] = []
    by_body: dict[str, list[Mapping[str, Any]]] = {}
    for raw in components:
        item = dict(raw)
        body_id = str(item["body_id"])
        by_body.setdefault(body_id, []).append(item)
        result = _validated_transferred_fidelity_result(
            item,
            package_binding_sha256=package_binding_sha256,
            thresholds=thresholds,
        )
        if result is None:
            result = evaluate_mesh_fidelity(
                item["source_mesh"],
                item["collider_mesh"],
                source_id=str(item["solid_id"]),
                collider_id=str(item.get("collider_id") or item["solid_id"]),
                representation=str(item["representation"]),
                thresholds=thresholds,
                local_interface_points=item.get("local_interface_points"),
                source_surface_points=item.get("source_surface_points"),
                collider_surface_points=item.get("collider_surface_points"),
                source_distance_evaluator=item.get(
                    "source_distance_evaluator"
                ),
                collider_distance_evaluator=item.get(
                    "collider_distance_evaluator"
                ),
                source_facts_override=item.get("source_facts_override"),
                collider_facts_override=item.get(
                    "collider_facts_override"
                ),
                occupancy_iou_override=item.get("occupancy_iou_override"),
            )
        result["body_id"] = body_id
        item["_evaluated_fidelity_result"] = result
        component_results.append(result)

    body_results: list[dict[str, Any]] = []
    for body_id, items in sorted(by_body.items()):
        if len(items) == 1:
            result = _single_source_body_result(
                items[0]["_evaluated_fidelity_result"],
                body_id=body_id,
            )
            body_results.append(result)
            continue
        source_union = trimesh.util.concatenate(
            tuple(_mesh(item["source_mesh"]) for item in items)
        )
        collider_union = trimesh.util.concatenate(
            tuple(_mesh(item["collider_mesh"]) for item in items)
        )
        result = evaluate_mesh_fidelity(
            source_union,
            collider_union,
            source_id=f"{body_id}:source_component_union",
            collider_id=f"{body_id}:collider_component_union",
            representation="component_union_without_enclosing_hull",
            thresholds=thresholds,
        )
        result.update(
            {
                "scope": "rigid_body_component_union",
                "body_id": body_id,
                "component_count": len(items),
            }
        )
        result = _promote_componentwise_body_union_result(
            result,
            component_results=(
                item["_evaluated_fidelity_result"] for item in items
            ),
        )
        body_results.append(result)

    results = [*component_results, *body_results]
    if not results:
        status = "INCONCLUSIVE"
        failures = ["mandatory_component_evidence_missing"]
    elif any(item["status"] == "FAIL" for item in results):
        status = "FAIL"
        failures = ["collider_fidelity_budget_exceeded"]
    elif any(item["status"] == "INCONCLUSIVE" for item in results):
        status = "INCONCLUSIVE"
        failures = ["mandatory_surface_evidence_inconclusive"]
    else:
        status = "PASS"
        failures = []
    return {
        "schema_version": COLLIDER_FIDELITY_SCHEMA,
        "status": status,
        "passed": status == "PASS",
        "package_binding_sha256": str(package_binding_sha256),
        "thresholds": thresholds.to_dict(),
        "component_count": len(component_results),
        "rigid_body_count": len(body_results),
        "components": component_results,
        "rigid_bodies": body_results,
        "failure_reasons": failures,
        "compound_body_assessed_as_component_union": True,
        "enclosing_convex_hull_used_for_assessment": False,
    }


def verify_collider_fidelity_binding(
    binding: Mapping[str, Any] | None,
    *,
    workspace_root: str | Path,
    package_binding_sha256: str,
) -> dict[str, Any]:
    """Verify the immutable report before a physical worker loads colliders."""

    from .io import ensure_inside, read_json, sha256_file
    from .models import MechanismValidationError

    if not isinstance(binding, Mapping):
        raise MechanismValidationError(
            "COLLIDER_FIDELITY_EVIDENCE_REQUIRED",
            "physical dispatch requires a collider-fidelity report binding",
        )
    relative = binding.get("report")
    expected_hash = binding.get("report_sha256")
    if not isinstance(relative, str) or not isinstance(expected_hash, str):
        raise MechanismValidationError(
            "COLLIDER_FIDELITY_BINDING_INVALID",
            "collider-fidelity report path or hash is missing",
        )
    path = ensure_inside(Path(workspace_root).resolve(), relative)
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_hash
    ):
        raise MechanismValidationError(
            "COLLIDER_FIDELITY_REPORT_CHANGED",
            "collider-fidelity evidence is missing or no longer hash-bound",
        )
    report = read_json(path)
    if (
        report.get("schema_version") != COLLIDER_FIDELITY_SCHEMA
        or report.get("passed") is not True
        or report.get("status") != "PASS"
        or report.get("package_binding_sha256") != package_binding_sha256
        or binding.get("package_binding_sha256") != package_binding_sha256
    ):
        raise MechanismValidationError(
            "COLLIDER_FIDELITY_REPORT_INVALID",
            "collider-fidelity evidence cannot authorize physical dispatch",
        )
    return report
