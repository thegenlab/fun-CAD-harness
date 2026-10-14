"""Strict package component intake for approved dynamic nonconvex STEP solids.

The package may declare *where* its convex STEP components are.  It cannot
declare that they are valid.  This module independently imports every
hash-bound component, proves convexity and unique source ownership, and
compares the component union to the authoritative assembly solid.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    RepairScope,
    make_failure_envelope,
    make_repair_contract,
)
from .collider_fidelity import multiresolution_occupancy_iou
from .io import ensure_inside, read_json, relative_to_root, sha256_file
from .occt_collision_fidelity import (
    hybrid_shape_distance_evaluator,
    mesh_shape_m,
    occupancy_iou_brep_vs_brep_union,
    shape_facts_m,
    shape_solids,
    shape_surface_points_m,
)
from .models import MechanismPlan, MechanismValidationError
from .reuse import solid_geometry_sha256


COLLISION_COMPONENT_MANIFEST_SCHEMA = (
    "simulation_ready_cad_collision_components_v1"
)
COLLISION_COMPONENT_MANIFEST_PATH = "collision_components/manifest.json"
COLLISION_COMPONENT_TEMPLATE_PATH = (
    "simulation_ready_cad_workflow/validation_pipeline/mechanism/templates/"
    "collision_components_manifest_v1.json"
)
COLLISION_COMPONENT_GUIDE_PATH = (
    "simulation_ready_cad_workflow/validation_pipeline/mechanism/templates/"
    "collision_components_manifest_v1.md"
)


def authored_collision_execution_identity(
    package_dir: str | Path,
) -> dict[str, Any] | None:
    """Bind authored component geometry without counting STEP header bytes."""

    package = Path(package_dir).resolve()
    manifest_path = package / COLLISION_COMPONENT_MANIFEST_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    document = _closed(
        read_json(manifest_path),
        {"schema_version", "sources"},
        label=COLLISION_COMPONENT_MANIFEST_PATH,
    )
    if document.get("schema_version") != COLLISION_COMPONENT_MANIFEST_SCHEMA:
        raise _fail(
            "COLLISION_COMPONENT_MANIFEST_SCHEMA_INVALID",
            "collision component manifest has an unsupported schema version",
        )
    sources: list[dict[str, Any]] = []
    for raw_source in document.get("sources") or ():
        source = _closed(
            raw_source,
            {"body_id", "source_solid_id", "components"},
            label="collision_components.sources[]",
        )
        solid_id = source.get("source_solid_id")
        if not isinstance(solid_id, str):
            raise _fail(
                "COLLISION_COMPONENT_SOURCE_BINDING_INVALID",
                "collision component source_solid_id must be a string",
            )
        component_geometry: list[str] = []
        for raw_component in source.get("components") or ():
            component = _closed(
                raw_component,
                {"component_id", "step_file", "sha256"},
                label="collision_components.components[]",
            )
            relative = component.get("step_file")
            if not isinstance(relative, str):
                raise _fail(
                    "COLLISION_COMPONENT_PATH_INVALID",
                    "component step_file must be a string",
                )
            step_path = ensure_inside(package, relative)
            component_id = str(component.get("component_id") or "")
            component_geometry.append(
                solid_geometry_sha256(
                    _component_shape(step_path, component_id)
                )
            )
        sources.append(
            {
                "source_solid_id": solid_id,
                "component_geometry_sha256s": sorted(component_geometry),
            }
        )
    return {
        "schema_version": COLLISION_COMPONENT_MANIFEST_SCHEMA,
        "sources": sorted(sources, key=lambda item: item["source_solid_id"]),
    }


_COMPONENT_ID = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,95}")
_CANONICAL_SOLID_ID = re.compile(r"solid_[1-9][0-9]*")
_MAX_COMPONENTS_PER_SOURCE_SOLID = 512
_MAX_TOTAL_COMPONENTS = 1024
_MAX_COMPONENT_OVERLAP_FRACTION = 0.05
_MAX_COMPONENT_MESH_VERTICES = 8192
_MAX_COMPONENT_MESH_FACES = 16384
_MAX_COMPONENT_HULL_FACETS = 16384
_MAX_COMPONENT_CONVEXITY_LP_ROWS = 8_000_000
_MAX_MANIFEST_CONVEXITY_LP_ROWS = 64_000_000
_MAX_RELATIVE_CONVEX_HULL_VOLUME_GAP = 1.0e-4
_MAX_EXACT_BREP_COMPONENTS = 8


@dataclass(frozen=True)
class AuthoredCollisionComponent:
    component_id: str
    source_solid_id: str
    step_path: Path
    step_relative: str
    step_sha256: str
    shape: Any
    mesh_world_m: Any
    exact_volume_m3: float
    convexity_certificate: Mapping[str, Any]


@dataclass(frozen=True)
class AuthoredCollisionSource:
    body_id: str
    source_solid_id: str
    source_shape: Any
    source_mesh_world_m: Any
    components: tuple[AuthoredCollisionComponent, ...]
    collider_union_shape: Any
    collider_union_mesh_world_m: Any
    fidelity_thresholds: Any
    fidelity_input: Mapping[str, Any]
    report: Mapping[str, Any]


def _fail(
    code: str,
    message: str,
    *,
    details: Mapping[str, Any] | None = None,
    evidence_validity: EvidenceValidity = EvidenceValidity.NOT_APPLICABLE,
) -> MechanismValidationError:
    """Return a package-owned rejection before plugin boundaries can relabel it."""

    failure_details = dict(details or {})
    return MechanismValidationError(
        code,
        message,
        details=failure_details,
        failure=make_failure_envelope(
            code,
            FailureOwner.DESIGN_PACKAGE,
            source_component="mechanism.authored_collision_components",
            message=message,
            evidence_validity=evidence_validity,
            details=failure_details,
            repair=make_repair_contract(
                RepairScope.DERIVED_ARTIFACT,
                preserves=(
                    "assembly.step",
                    "static_visual_review",
                    "approved_plan",
                ),
                invalidates=(
                    "collision_materialization",
                    "collider_fidelity",
                    "physical_preflight",
                ),
                resume_stage="preflight",
            ),
        ),
    )


def _closed(
    value: Any,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise _fail(
            "COLLISION_COMPONENT_MANIFEST_SCHEMA_INVALID",
            f"{label} must use the exact closed schema; "
            f"unknown={sorted(actual - keys)!r}, "
            f"missing={sorted(keys - actual)!r}",
        )
    return value


def _component_shape(path: Path, component_id: str) -> Any:
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    if reader.ReadFile(str(path)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise _fail(
            "COLLISION_COMPONENT_STEP_IMPORT_FAILED",
            f"OCCT could not import component {component_id!r}",
        )
    solids = shape_solids(reader.OneShape())
    if len(solids) != 1:
        raise _fail(
            "COLLISION_COMPONENT_SOLID_COUNT_INVALID",
            f"component {component_id!r} must contain exactly one STEP solid",
        )
    shape = solids[0]
    if not BRepCheck_Analyzer(shape).IsValid():
        raise _fail(
            "COLLISION_COMPONENT_BREP_INVALID",
            f"component {component_id!r} is not a valid closed BRep solid",
        )
    return shape


def _fuse_components(components: Sequence[AuthoredCollisionComponent]) -> Any:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse

    # A deterministic balanced reduction avoids repeatedly rebuilding an
    # ever-growing BRep against one tiny component.  The Boolean operation and
    # final fidelity gate are unchanged.
    level = [component.shape for component in components]
    while len(level) > 1:
        next_level: list[Any] = []
        for index in range(0, len(level), 2):
            if index + 1 == len(level):
                next_level.append(level[index])
                continue
            operation = BRepAlgoAPI_Fuse(
                level[index],
                level[index + 1],
            )
            operation.Build()
            if not operation.IsDone():
                raise _fail(
                    "COLLISION_COMPONENT_UNION_FAILED",
                    "OCCT could not form the authored component union",
                )
            next_level.append(operation.Shape())
        level = next_level
    result = level[0]
    if not shape_solids(result):
        raise _fail(
            "COLLISION_COMPONENT_UNION_EMPTY",
            "the authored component union contains no solid",
        )
    return result


def _authored_union_occupancy(
    *,
    source_shape: Any,
    component_union_shape: Any,
    components: Sequence[AuthoredCollisionComponent],
    source_mesh: Any,
    component_union_mesh: Any,
) -> tuple[float, dict[str, Any]]:
    """Measure occupancy without sending large component sets to General Fuse."""

    if len(components) <= _MAX_EXACT_BREP_COMPONENTS:
        return occupancy_iou_brep_vs_brep_union(
            source_shape,
            component_union_shape,
            component_shapes=tuple(item.shape for item in components),
        )
    value, raw_evidence = multiresolution_occupancy_iou(
        (source_mesh,),
        (component_union_mesh,),
    )
    if value is None:
        raise _fail(
            "COLLISION_COMPONENT_OCCUPANCY_INCONCLUSIVE",
            "bounded multi-resolution occupancy could not evaluate the "
            "authored component union",
            details={
                "component_count": len(components),
                "exact_component_limit": _MAX_EXACT_BREP_COMPONENTS,
                "occupancy_evidence": raw_evidence,
            },
        )
    evidence = dict(raw_evidence)
    evidence.update(
        {
            "component_count": len(components),
            "exact_component_limit": _MAX_EXACT_BREP_COMPONENTS,
            "exact_general_fuse_skipped": True,
            "bounded_work_reason": "component_count_exceeds_exact_limit",
        }
    )
    return float(value), evidence


def _authored_surface_evaluators(
    *,
    source_shape: Any,
    source_mesh: Any,
    component_union_shape: Any,
    component_union_mesh: Any,
    components: Sequence[AuthoredCollisionComponent],
    exact_query_trigger_m: float,
) -> tuple[Any | None, Any | None, str]:
    """Use exact point queries only while their BRep complexity is bounded."""

    if len(components) > _MAX_EXACT_BREP_COMPONENTS:
        return None, None, "bounded_mesh_nearest_surface_v1"
    return (
        hybrid_shape_distance_evaluator(
            source_shape,
            source_mesh,
            exact_query_trigger_m=exact_query_trigger_m,
        ),
        hybrid_shape_distance_evaluator(
            component_union_shape,
            component_union_mesh,
            exact_query_trigger_m=exact_query_trigger_m,
        ),
        "vertex_upper_bound_then_occt_exact_v1",
    )


def _validate_component_convexity(
    mesh: Any,
    *,
    component_id: str,
    collision_envelope_m: float,
    manifest_lp_rows_remaining: int = _MAX_MANIFEST_CONVEXITY_LP_ROWS,
) -> dict[str, Any]:
    import numpy as np
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, QhullError

    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    envelope = float(collision_envelope_m)
    if not math.isfinite(envelope) or envelope < 0.0:
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_TOLERANCE_INVALID",
            "the validator collision envelope must be finite and non-negative",
        )
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or faces.ndim != 2
        or faces.shape[1:] != (3,)
        or len(vertices) < 4
        or len(faces) < 4
        or not np.all(np.isfinite(vertices))
        or not np.issubdtype(faces.dtype, np.integer)
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise _fail(
            "COLLISION_COMPONENT_MESH_INVALID",
            f"component {component_id!r} produced invalid mesh topology",
        )
    signed_volume = float(mesh.volume)
    if (
        not bool(mesh.is_watertight)
        or not bool(mesh.is_winding_consistent)
        or int(mesh.body_count) != 1
        or not bool(mesh.is_volume)
        or not math.isfinite(signed_volume)
        or signed_volume <= 0.0
    ):
        raise _fail(
            "COLLISION_COMPONENT_MESH_NOT_CLOSED",
            f"component {component_id!r} did not produce one closed, "
            "consistently wound, positive-volume mesh body",
        )

    bounds = np.asarray((np.min(vertices, axis=0), np.max(vertices, axis=0)))
    center = np.mean(bounds, axis=0)
    characteristic_length_m = float(np.linalg.norm(bounds[1] - bounds[0]))
    if (
        not math.isfinite(characteristic_length_m)
        or characteristic_length_m <= 0.0
    ):
        raise _fail(
            "COLLISION_COMPONENT_MESH_INVALID",
            f"component {component_id!r} has degenerate bounds",
        )
    normalized_vertices = (vertices - center) / characteristic_length_m
    triangles = normalized_vertices[faces]
    doubled_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    area_floor = 256.0 * float(np.finfo(np.float64).eps)
    if (
        not np.all(np.isfinite(doubled_areas))
        or bool(np.any(doubled_areas <= area_floor))
    ):
        raise _fail(
            "COLLISION_COMPONENT_MESH_DEGENERATE_FACE",
            f"component {component_id!r} contains a degenerate triangle",
        )

    vertex_count = int(len(vertices))
    face_count = int(len(faces))
    if (
        vertex_count > _MAX_COMPONENT_MESH_VERTICES
        or face_count > _MAX_COMPONENT_MESH_FACES
    ):
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_BUDGET_EXCEEDED",
            f"component {component_id!r} exceeds the convexity mesh budget",
        )
    try:
        hull = ConvexHull(normalized_vertices)
    except QhullError as exc:
        raise _fail(
            "COLLISION_COMPONENT_NOT_CONVEX",
            f"component {component_id!r} has no three-dimensional convex hull",
        ) from exc
    equations = np.asarray(hull.equations, dtype=float)
    if (
        equations.ndim != 2
        or equations.shape[1:] != (4,)
        or not np.all(np.isfinite(equations))
    ):
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_INCONCLUSIVE",
            f"component {component_id!r} produced invalid hull halfspaces",
        )
    normal_lengths = np.linalg.norm(equations[:, :3], axis=1)
    if (
        not np.all(np.isfinite(normal_lengths))
        or bool(np.any(normal_lengths <= 0.0))
    ):
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_INCONCLUSIVE",
            f"component {component_id!r} produced degenerate hull halfspaces",
        )
    equations = equations / normal_lengths[:, np.newaxis]
    hull_facet_count = int(len(equations))
    lp_constraint_rows = face_count * hull_facet_count
    if (
        hull_facet_count > _MAX_COMPONENT_HULL_FACETS
        or lp_constraint_rows > _MAX_COMPONENT_CONVEXITY_LP_ROWS
    ):
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_BUDGET_EXCEEDED",
            f"component {component_id!r} exceeds the convexity LP budget",
        )
    if (
        not isinstance(manifest_lp_rows_remaining, int)
        or manifest_lp_rows_remaining < 0
        or lp_constraint_rows > manifest_lp_rows_remaining
    ):
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_MANIFEST_BUDGET_EXCEEDED",
            "the collision-component manifest exceeds the cumulative "
            "convexity LP budget",
        )

    vertex_slacks_normalized = -(
        normalized_vertices @ equations[:, :3].T
        + equations[:, 3][np.newaxis, :]
    )
    if not np.all(np.isfinite(vertex_slacks_normalized)):
        raise _fail(
            "COLLISION_COMPONENT_CONVEXITY_INCONCLUSIVE",
            f"component {component_id!r} produced non-finite hull slacks",
        )
    nearest_vertex_slack_normalized = np.min(
        vertex_slacks_normalized,
        axis=1,
    )
    maximum_vertex_inward_depth_m = characteristic_length_m * max(
        0.0,
        float(np.max(nearest_vertex_slack_normalized)),
    )
    maximum_vertex_outward_error_m = characteristic_length_m * max(
        0.0,
        -float(np.min(vertex_slacks_normalized)),
    )

    maximum_face_inward_depth_normalized = 0.0
    equality = np.asarray(((1.0, 1.0, 1.0, 0.0),), dtype=float)
    for face_index, face in enumerate(faces):
        triangle_slacks = vertex_slacks_normalized[face].T
        inequalities = np.column_stack(
            (
                -triangle_slacks,
                np.ones(hull_facet_count, dtype=float),
            )
        )
        result = linprog(
            np.asarray((0.0, 0.0, 0.0, -1.0), dtype=float),
            A_ub=inequalities,
            b_ub=np.zeros(hull_facet_count, dtype=float),
            A_eq=equality,
            b_eq=np.asarray((1.0,), dtype=float),
            bounds=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, None)),
            method="highs",
        )
        if (
            result.status != 0
            or not bool(result.success)
            or result.x is None
            or len(result.x) != 4
            or not np.all(np.isfinite(result.x))
        ):
            raise _fail(
                "COLLISION_COMPONENT_CONVEXITY_INCONCLUSIVE",
                f"component {component_id!r} face {face_index} convexity LP "
                "did not reach a finite optimum",
            )
        maximum_face_inward_depth_normalized = max(
            maximum_face_inward_depth_normalized,
            max(0.0, float(result.x[3])),
        )
    maximum_face_inward_depth_m = (
        characteristic_length_m * maximum_face_inward_depth_normalized
    )

    mesh_volume = signed_volume
    hull_volume = float(hull.volume) * characteristic_length_m**3
    relative_hull_gap = abs(hull_volume - mesh_volume) / mesh_volume
    passed = (
        maximum_vertex_inward_depth_m <= envelope
        and maximum_vertex_outward_error_m <= envelope
        and maximum_face_inward_depth_m <= envelope
        and relative_hull_gap <= _MAX_RELATIVE_CONVEX_HULL_VOLUME_GAP
    )
    if not passed:
        raise _fail(
            "COLLISION_COMPONENT_NOT_CONVEX",
            f"component {component_id!r} is not a verified convex solid; "
            f"vertex_inward={maximum_vertex_inward_depth_m:.9g} m, "
            f"face_inward={maximum_face_inward_depth_m:.9g} m, "
            f"volume_gap={relative_hull_gap:.9g}",
        )
    return {
        "schema_version": "collision_component_convexity_certificate_v1",
        "status": "PASS",
        "passed": True,
        "method": (
            "normalized_convex_hull_halfspaces_and_triangle_interior_lp_v1"
        ),
        "directed_metric": (
            "mesh_surface_to_convex_hull_boundary_inward_depth"
        ),
        "not_claimed_as_bidirectional_hausdorff": True,
        "finite_vertices": True,
        "single_mesh_body": True,
        "watertight": True,
        "winding_consistent": True,
        "positive_oriented_volume": True,
        "nondegenerate_triangles": True,
        "vertex_count": vertex_count,
        "triangle_count": face_count,
        "hull_facet_count": hull_facet_count,
        "characteristic_length_m": characteristic_length_m,
        "collision_envelope_tolerance_m": envelope,
        "maximum_vertex_inward_depth_m": maximum_vertex_inward_depth_m,
        "maximum_vertex_outward_error_m": maximum_vertex_outward_error_m,
        "maximum_triangle_interior_inward_depth_m": (
            maximum_face_inward_depth_m
        ),
        "relative_convex_hull_volume_gap": relative_hull_gap,
        "relative_convex_hull_volume_gap_max": (
            _MAX_RELATIVE_CONVEX_HULL_VOLUME_GAP
        ),
        "lp_solver": "scipy_optimize_highs",
        "lp_count": face_count,
        "lp_constraint_rows": lp_constraint_rows,
        "budgets": {
            "maximum_component_vertices": _MAX_COMPONENT_MESH_VERTICES,
            "maximum_component_triangles": _MAX_COMPONENT_MESH_FACES,
            "maximum_component_hull_facets": _MAX_COMPONENT_HULL_FACETS,
            "maximum_component_lp_constraint_rows": (
                _MAX_COMPONENT_CONVEXITY_LP_ROWS
            ),
            "maximum_manifest_lp_constraint_rows": (
                _MAX_MANIFEST_CONVEXITY_LP_ROWS
            ),
        },
        "trimesh_is_convex_used_as_evidence": False,
    }


def _manifest_path(plan: MechanismPlan, package: Path) -> Path:
    path = ensure_inside(package, COLLISION_COMPONENT_MANIFEST_PATH)
    expected = plan.artifact_hashes.get(COLLISION_COMPONENT_MANIFEST_PATH)
    if (
        not path.is_file()
        or path.is_symlink()
        or expected is None
        or sha256_file(path) != expected
    ):
        raise _fail(
            "DYNAMIC_COLLISION_COMPONENT_MANIFEST_REQUIRED",
            "approved dynamic nonconvex execution requires the hash-bound "
            f"{COLLISION_COMPONENT_MANIFEST_PATH}. Copy "
            f"{COLLISION_COMPONENT_TEMPLATE_PATH} and read "
            f"{COLLISION_COMPONENT_GUIDE_PATH}; export local convex STEP "
            "components for each required solid and list every file in "
            "package.json artifacts. The files are untrusted design inputs; "
            "Harness decides validity.",
        )
    return path


def _component_asset(
    raw: Any,
    *,
    package: Path,
    plan: MechanismPlan,
    source_solid_id: str,
    scratch: Path,
    collision_envelope_m: float,
    manifest_lp_rows_remaining: int,
) -> AuthoredCollisionComponent:
    component_id, relative, expected_hash, path = (
        _component_declaration_identity(
            raw,
            package=package,
            plan=plan,
            source_solid_id=source_solid_id,
        )
    )
    shape = _component_shape(path, component_id)
    mesh = mesh_shape_m(
        shape,
        scratch / source_solid_id / f"{component_id}.stl",
    )
    convexity_certificate = _validate_component_convexity(
        mesh,
        component_id=component_id,
        collision_envelope_m=collision_envelope_m,
        manifest_lp_rows_remaining=manifest_lp_rows_remaining,
    )
    facts = shape_facts_m(shape)
    exact_volume_m3 = float(facts["volume_m3"])
    if not math.isfinite(exact_volume_m3) or exact_volume_m3 <= 0.0:
        raise _fail(
            "COLLISION_COMPONENT_VOLUME_INVALID",
            f"component {component_id!r} has no positive exact volume",
        )
    return AuthoredCollisionComponent(
        component_id=component_id,
        source_solid_id=source_solid_id,
        step_path=path,
        step_relative=relative,
        step_sha256=expected_hash,
        shape=shape,
        mesh_world_m=mesh,
        exact_volume_m3=exact_volume_m3,
        convexity_certificate=convexity_certificate,
    )


def _component_declaration_identity(
    raw: Any,
    *,
    package: Path,
    plan: MechanismPlan,
    source_solid_id: str,
) -> tuple[str, str, str, Path]:
    """Validate one hash-bound component declaration without importing STEP."""

    item = _closed(
        raw,
        {"component_id", "step_file", "sha256"},
        label=f"{source_solid_id}.components[]",
    )
    component_id = item.get("component_id")
    if not isinstance(component_id, str) or _COMPONENT_ID.fullmatch(component_id) is None:
        raise _fail(
            "COLLISION_COMPONENT_ID_INVALID",
            f"{source_solid_id} has an invalid component_id",
        )
    relative = item.get("step_file")
    expected_hash = item.get("sha256")
    if not isinstance(relative, str) or not relative.endswith(".step"):
        raise _fail(
            "COLLISION_COMPONENT_PATH_INVALID",
            f"component {component_id!r} must reference a .step file",
        )
    path = ensure_inside(package, relative)
    component_root = (package / "collision_components").resolve()
    try:
        path.relative_to(component_root)
    except ValueError as exc:
        raise _fail(
            "COLLISION_COMPONENT_PATH_INVALID",
            f"component {component_id!r} must stay under collision_components/",
        ) from exc
    intake_hash = plan.artifact_hashes.get(relative)
    if (
        not isinstance(expected_hash, str)
        or intake_hash != expected_hash
        or not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_hash
    ):
        raise _fail(
            "COLLISION_COMPONENT_ASSET_CHANGED",
            f"component {component_id!r} is missing or no longer intake-bound",
        )
    return component_id, relative, expected_hash, path


def load_authored_collision_components(
    plan: MechanismPlan,
    *,
    package: Path,
    workspace: Path,
    expected_sources: Mapping[tuple[str, str], tuple[Any, Any]],
    scratch_root: Path,
    characteristic_length_m: float,
    mesh_linear_deflection_m: float,
    collision_envelope_m: float,
) -> tuple[AuthoredCollisionSource, ...]:
    """Load and independently certify all required nonconvex dynamic sources."""

    from .collider_fidelity import (
        derive_fidelity_thresholds,
        evaluate_mesh_fidelity,
    )

    if not expected_sources:
        return ()
    manifest_path = _manifest_path(plan, package)
    document = _closed(
        read_json(manifest_path),
        {"schema_version", "sources"},
        label=COLLISION_COMPONENT_MANIFEST_PATH,
    )
    if document.get("schema_version") != COLLISION_COMPONENT_MANIFEST_SCHEMA:
        raise _fail(
            "COLLISION_COMPONENT_MANIFEST_SCHEMA_INVALID",
            "collision component manifest has an unsupported schema version",
        )
    raw_sources = document.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise _fail(
            "COLLISION_COMPONENT_MANIFEST_SCHEMA_INVALID",
            "collision component manifest sources must be a non-empty array",
        )
    # ``solid_N`` is the package-global identity of authoritative CAD
    # geometry. A family-specific body name is only its runtime owner in that
    # capsule. One declaration per source solid lets another approved task
    # reuse the same certified assets under a different body_id.
    parsed_by_solid: dict[str, tuple[str, Mapping[str, Any]]] = {}
    manifest_component_ids: set[str] = set()
    manifest_component_paths: set[str] = set()
    manifest_component_count = 0
    for raw in raw_sources:
        item = _closed(
            raw,
            {"body_id", "source_solid_id", "components"},
            label="collision_components.sources[]",
        )
        body_id = item.get("body_id")
        solid_id = item.get("source_solid_id")
        if not isinstance(body_id, str) or not isinstance(solid_id, str):
            raise _fail(
                "COLLISION_COMPONENT_SOURCE_BINDING_INVALID",
                "every component source needs string body_id and source_solid_id",
            )
        if _CANONICAL_SOLID_ID.fullmatch(solid_id) is None:
            raise _fail(
                "COLLISION_COMPONENT_SOURCE_BINDING_INVALID",
                f"{body_id}/{solid_id} does not use canonical solid_N syntax",
            )
        if solid_id in parsed_by_solid:
            raise _fail(
                "COLLISION_COMPONENT_SOURCE_DUPLICATE",
                f"component source {solid_id} is declared more than once",
            )
        raw_components = item.get("components")
        if (
            not isinstance(raw_components, list)
            or len(raw_components) < 2
            or len(raw_components) > _MAX_COMPONENTS_PER_SOURCE_SOLID
        ):
            raise _fail(
                "COLLISION_COMPONENT_COUNT_INVALID",
                f"{body_id}/{solid_id} needs 2.."
                f"{_MAX_COMPONENTS_PER_SOURCE_SOLID} local convex components; "
                "one enclosing hull or box is forbidden",
            )
        for component_raw in raw_components:
            component_id, relative, _expected_hash, _path = (
                _component_declaration_identity(
                    component_raw,
                    package=package,
                    plan=plan,
                    source_solid_id=solid_id,
                )
            )
            folded_path = relative.casefold()
            if (
                component_id in manifest_component_ids
                or folded_path in manifest_component_paths
            ):
                raise _fail(
                    "COLLISION_COMPONENT_OWNERSHIP_NOT_UNIQUE",
                    f"{body_id}/{solid_id} reuses component ownership; "
                    f"component_id={component_id!r}, path={relative!r}",
                )
            manifest_component_ids.add(component_id)
            manifest_component_paths.add(folded_path)
            manifest_component_count += 1
        if manifest_component_count > _MAX_TOTAL_COMPONENTS:
            raise _fail(
                "COLLISION_COMPONENT_TOTAL_LIMIT",
                f"collision component manifest exceeds {_MAX_TOTAL_COMPONENTS} assets",
            )
        parsed_by_solid[solid_id] = (body_id, item)

    expected_by_solid: dict[str, str] = {}
    for body_id, solid_id in expected_sources:
        prior_body = expected_by_solid.get(solid_id)
        if prior_body is not None:
            raise _fail(
                "COLLISION_COMPONENT_RUNTIME_SOURCE_AMBIGUOUS",
                f"{solid_id} is assigned to both {prior_body!r} and "
                f"{body_id!r} in one mechanism family",
            )
        expected_by_solid[solid_id] = body_id
    missing = sorted(set(expected_by_solid) - set(parsed_by_solid))
    if missing:
        raise _fail(
            "COLLISION_COMPONENT_SOURCE_SET_MISMATCH",
            "component manifest must bind every required dynamic nonconvex "
            f"source solid; missing={missing!r}",
        )

    thresholds = derive_fidelity_thresholds(
        characteristic_length_m=characteristic_length_m,
        mesh_linear_deflection_m=mesh_linear_deflection_m,
        collision_envelope_m=collision_envelope_m,
    )
    results: list[AuthoredCollisionSource] = []
    manifest_convexity_lp_rows = 0
    for key in sorted(expected_sources):
        body_id, source_solid_id = key
        source_shape, source_mesh = expected_sources[key]
        declared_body_id, declaration = parsed_by_solid[source_solid_id]
        raw_components = declaration["components"]
        loaded_components: list[AuthoredCollisionComponent] = []
        for raw in raw_components:
            component = _component_asset(
                raw,
                package=package,
                plan=plan,
                source_solid_id=source_solid_id,
                scratch=scratch_root,
                collision_envelope_m=collision_envelope_m,
                manifest_lp_rows_remaining=(
                    _MAX_MANIFEST_CONVEXITY_LP_ROWS
                    - manifest_convexity_lp_rows
                ),
            )
            manifest_convexity_lp_rows += int(
                component.convexity_certificate["lp_constraint_rows"]
            )
            loaded_components.append(component)
        components = tuple(loaded_components)
        ids = [item.component_id for item in components]
        unique_source_ownership_verified = True

        union_shape = _fuse_components(components)
        union_mesh = mesh_shape_m(
            union_shape,
            scratch_root / source_solid_id / "component_union.stl",
        )
        source_points = shape_surface_points_m(
            source_shape,
            maximum_spacing_m=max(
                0.25 * thresholds.surface_max_max_m,
                5.0e-5,
            ),
        )
        collider_points = shape_surface_points_m(
            union_shape,
            maximum_spacing_m=max(
                0.25 * thresholds.surface_max_max_m,
                5.0e-5,
            ),
        )
        source_facts = shape_facts_m(source_shape)
        collider_facts = shape_facts_m(union_shape)
        occupancy_override = _authored_union_occupancy(
            source_shape=source_shape,
            component_union_shape=union_shape,
            components=components,
            source_mesh=source_mesh,
            component_union_mesh=union_mesh,
        )
        exact_query_trigger_m = 0.25 * thresholds.surface_max_max_m
        (
            source_distance_evaluator,
            collider_distance_evaluator,
            surface_distance_method,
        ) = _authored_surface_evaluators(
            source_shape=source_shape,
            source_mesh=source_mesh,
            component_union_shape=union_shape,
            component_union_mesh=union_mesh,
            components=components,
            exact_query_trigger_m=exact_query_trigger_m,
        )
        fidelity = evaluate_mesh_fidelity(
            source_mesh,
            union_mesh,
            source_id=source_solid_id,
            collider_id=f"{source_solid_id}:authored_convex_component_union",
            representation="ChCollisionShapeTriangleMeshDynamicConvexComponents",
            thresholds=thresholds,
            source_surface_points=source_points,
            collider_surface_points=collider_points,
            source_distance_evaluator=source_distance_evaluator,
            collider_distance_evaluator=collider_distance_evaluator,
            source_facts_override=source_facts,
            collider_facts_override=collider_facts,
            occupancy_iou_override=occupancy_override,
        )
        component_volume_sum = sum(
            item.exact_volume_m3 for item in components
        )
        union_volume = float(collider_facts["volume_m3"])
        overlap_fraction = max(
            0.0,
            (component_volume_sum - union_volume)
            / max(float(source_facts["volume_m3"]), 1.0e-18),
        )
        if overlap_fraction > _MAX_COMPONENT_OVERLAP_FRACTION:
            raise _fail(
                "COLLISION_COMPONENT_OVERLAP_EXCESSIVE",
                f"{body_id}/{source_solid_id} component overlap "
                f"{overlap_fraction:.6f} exceeds the validator budget",
                details={
                    "body_id": body_id,
                    "source_solid_id": source_solid_id,
                    "component_overlap_fraction": overlap_fraction,
                    "maximum_component_overlap_fraction": (
                        _MAX_COMPONENT_OVERLAP_FRACTION
                    ),
                },
                evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            )
        if fidelity.get("passed") is not True:
            raise _fail(
                "COLLISION_COMPONENT_UNION_FIDELITY_FAILED",
                f"{body_id}/{source_solid_id} component union does not match "
                "the authoritative STEP solid; "
                f"failed={fidelity.get('failure_reasons')!r}, "
                f"surface={fidelity.get('surface_distance')!r}, "
                f"volume={fidelity.get('relative_volume_error')!r}, "
                f"inertia={fidelity.get('inertia_relative_error')!r}, "
                f"iou={fidelity.get('occupancy_iou')!r}",
                details={
                    "body_id": body_id,
                    "source_solid_id": source_solid_id,
                    "failure_reasons": fidelity.get("failure_reasons") or [],
                    "surface_distance": fidelity.get("surface_distance"),
                    "relative_volume_error": fidelity.get(
                        "relative_volume_error"
                    ),
                    "inertia_relative_error": fidelity.get(
                        "inertia_relative_error"
                    ),
                    "occupancy_iou": fidelity.get("occupancy_iou"),
                },
                evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            )
        fidelity_input = {
            "body_id": body_id,
            "solid_id": source_solid_id,
            "collider_id": (
                f"{source_solid_id}:authored_convex_component_union"
            ),
            "representation": (
                "ChCollisionShapeTriangleMeshDynamicConvexComponents"
            ),
            "source_mesh": source_mesh,
            "collider_mesh": union_mesh,
            "source_surface_points": source_points,
            "collider_surface_points": collider_points,
            "source_distance_evaluator": source_distance_evaluator,
            "collider_distance_evaluator": collider_distance_evaluator,
            "source_facts_override": source_facts,
            "collider_facts_override": collider_facts,
            "occupancy_iou_override": occupancy_override,
        }
        report = {
            "schema_version": COLLISION_COMPONENT_MANIFEST_SCHEMA,
            "status": "PASS",
            "passed": True,
            "manifest": relative_to_root(workspace, manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "package_binding_sha256": plan.package_binding_sha256,
            "body_id": body_id,
            "declared_body_id": declared_body_id,
            "runtime_body_alias_applied": declared_body_id != body_id,
            "source_solid_id": source_solid_id,
            "source_authority": "intake_bound_authoritative_assembly_step",
            "component_count": len(components),
            "component_ids": ids,
            "component_step_files": [
                item.step_relative for item in components
            ],
            "component_step_sha256": [
                item.step_sha256 for item in components
            ],
            "component_convexity_certificates": [
                item.convexity_certificate for item in components
            ],
            "each_component_convex": True,
            "each_component_positive_volume": True,
            "unique_source_ownership": unique_source_ownership_verified,
            "ownership_verification": {
                "scope": "entire_manifest",
                "component_ids_globally_unique": True,
                "component_paths_globally_unique": True,
                "manifest_component_count": manifest_component_count,
            },
            "whole_body_convex_hull_used": False,
            "whole_body_box_proxy_used": False,
            "design_claim_used_as_fidelity_evidence": False,
            "surface_distance_evaluator": surface_distance_method,
            "component_mesh_export_authority": "harness_occt",
            "component_volume_sum_m3": component_volume_sum,
            "component_union_volume_m3": union_volume,
            "component_overlap_fraction": overlap_fraction,
            "maximum_component_overlap_fraction": (
                _MAX_COMPONENT_OVERLAP_FRACTION
            ),
            "fidelity": fidelity,
        }
        results.append(
            AuthoredCollisionSource(
                body_id=body_id,
                source_solid_id=source_solid_id,
                source_shape=source_shape,
                source_mesh_world_m=source_mesh,
                components=components,
                collider_union_shape=union_shape,
                collider_union_mesh_world_m=union_mesh,
                fidelity_thresholds=thresholds,
                fidelity_input=fidelity_input,
                report=report,
            )
        )
    for item in results:
        item.report["manifest_convexity_lp_constraint_rows"] = (
            manifest_convexity_lp_rows
        )
        item.report["manifest_convexity_lp_constraint_rows_max"] = (
            _MAX_MANIFEST_CONVEXITY_LP_ROWS
        )
    return tuple(results)


def bind_exported_component_meshes(
    source: AuthoredCollisionSource,
    exported_world_meshes: Mapping[str, Any],
    *,
    exported_asset_paths: Mapping[str, Path],
    exported_asset_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind fidelity to the exact OBJ bytes that the runtime will load."""

    import numpy as np

    expected_ids = {item.component_id for item in source.components}
    if (
        set(exported_world_meshes) != expected_ids
        or set(exported_asset_paths) != expected_ids
        or set(exported_asset_sha256) != expected_ids
    ):
        raise _fail(
            "COLLISION_COMPONENT_RUNTIME_EXPORT_SET_MISMATCH",
            f"{source.body_id}/{source.source_solid_id} runtime exports do not "
            "match the certified component set",
        )
    export_results: list[dict[str, Any]] = []
    for declared in source.components:
        exported = exported_world_meshes[declared.component_id]
        asset_path = exported_asset_paths[declared.component_id]
        asset_sha256 = exported_asset_sha256[declared.component_id]
        if re.fullmatch(r"[0-9a-f]{64}", asset_sha256) is None:
            raise _fail(
                "COLLISION_COMPONENT_RUNTIME_EXPORT_HASH_INVALID",
                f"runtime OBJ for {declared.component_id!r} has no valid hash",
            )
        if (
            not asset_path.is_file()
            or asset_path.is_symlink()
            or sha256_file(asset_path) != asset_sha256
        ):
            raise _fail(
                "COLLISION_COMPONENT_RUNTIME_EXPORT_HASH_CHANGED",
                f"runtime OBJ bytes changed for {declared.component_id!r}",
            )
        expected_vertices = np.asarray(
            declared.mesh_world_m.vertices,
            dtype=float,
        )
        actual_vertices = np.asarray(exported.vertices, dtype=float)
        expected_faces = np.asarray(declared.mesh_world_m.faces, dtype=int)
        actual_faces = np.asarray(exported.faces, dtype=int)
        characteristic_length_m = max(
            float(
                np.linalg.norm(
                    np.ptp(expected_vertices, axis=0),
                )
            ),
            1.0e-12,
        )
        numeric_tolerance_m = max(
            5.0e-13,
            128.0
            * float(np.finfo(np.float64).eps)
            * max(1.0, characteristic_length_m),
        )
        topology_equal = bool(
            expected_vertices.shape == actual_vertices.shape
            and expected_faces.shape == actual_faces.shape
            and np.array_equal(expected_faces, actual_faces)
        )
        maximum_vertex_error_m = (
            float(
                np.max(
                    np.linalg.norm(
                        expected_vertices - actual_vertices,
                        axis=1,
                    )
                )
            )
            if topology_equal and len(expected_vertices)
            else math.inf
        )
        if (
            not topology_equal
            or not math.isfinite(maximum_vertex_error_m)
            or maximum_vertex_error_m > numeric_tolerance_m
        ):
            raise _fail(
                "COLLISION_COMPONENT_RUNTIME_EXPORT_BINDING_FAILED",
                f"runtime OBJ for {declared.component_id!r} no longer matches "
                "the Harness-exported component topology and vertices",
            )
        export_results.append(
            {
                "component_id": declared.component_id,
                "status": "PASS",
                "passed": True,
                "runtime_obj_sha256": asset_sha256,
                "binding_method": (
                    "deterministic_obj_round_trip_topology_and_vertices_v1"
                ),
                "topology_equal": topology_equal,
                "vertex_count": int(len(actual_vertices)),
                "triangle_count": int(len(actual_faces)),
                "maximum_vertex_error_m": maximum_vertex_error_m,
                "numeric_tolerance_m": numeric_tolerance_m,
                "convexity_certificate": declared.convexity_certificate,
                "convexity_certificate_transfer": (
                    "exact_topology_and_machine_precision_vertex_binding"
                ),
            }
        )
    occupancy_override = source.fidelity_input.get("occupancy_iou_override")
    if (
        not isinstance(occupancy_override, tuple)
        or len(occupancy_override) != 2
        or not isinstance(occupancy_override[1], Mapping)
    ):
        raise _fail(
            "COLLISION_COMPONENT_RUNTIME_FIDELITY_TRANSFER_INVALID",
            f"{source.body_id}/{source.source_solid_id} has no trusted "
            "authored-union occupancy evidence to transfer",
        )
    source_fidelity_result = source.report.get("fidelity")
    if not isinstance(source_fidelity_result, Mapping):
        raise _fail(
            "COLLISION_COMPONENT_RUNTIME_FIDELITY_TRANSFER_INVALID",
            f"{source.body_id}/{source.source_solid_id} has no trusted "
            "authored-union fidelity result to transfer",
        )
    from .collider_fidelity import ExactRuntimeMeshFidelityTransfer

    fidelity_input = dict(source.fidelity_input)
    # Exact round-trip binding above proves that each runtime OBJ is the same
    # component mesh imported from its STEP.  Keep the OCCT Boolean-union
    # exterior for surface/facts checks: concatenating component meshes would
    # incorrectly expose triangles inside overlap regions as exterior faces.
    fidelity_input["occupancy_iou_override"] = occupancy_override
    fidelity_input["_exact_runtime_mesh_fidelity_transfer"] = (
        ExactRuntimeMeshFidelityTransfer(
            result=source_fidelity_result,
            package_binding_sha256=str(
                source.report["package_binding_sha256"]
            ),
            source_id=source.source_solid_id,
            collider_id=(
                f"{source.source_solid_id}:authored_convex_component_union"
            ),
            representation=(
                "ChCollisionShapeTriangleMeshDynamicConvexComponents"
            ),
            component_ids=tuple(
                item.component_id for item in source.components
            ),
            binding_method=(
                "deterministic_obj_round_trip_topology_and_vertices_v1"
            ),
        )
    )
    report = dict(source.report)
    report["runtime_component_mesh_exports"] = export_results
    report["runtime_component_mesh_export_count"] = len(export_results)
    report["runtime_component_mesh_bytes_fidelity_verified"] = all(
        item["passed"] is True for item in export_results
    )
    report["runtime_component_mesh_binding_method"] = (
        "deterministic_obj_round_trip_topology_and_vertices_v1"
    )
    report["runtime_union_surface_authority"] = (
        "occt_boolean_union_of_round_trip_bound_component_steps"
    )
    report["runtime_union_occupancy_iou"] = occupancy_override[0]
    report["runtime_union_occupancy_evidence"] = occupancy_override[1]
    report["runtime_union_occupancy_recomputed"] = False
    report["runtime_union_occupancy_transfer_method"] = (
        "deterministic_obj_round_trip_topology_and_vertices_v1"
    )
    report["runtime_fidelity_result_transfer_ready"] = True
    return fidelity_input, report
