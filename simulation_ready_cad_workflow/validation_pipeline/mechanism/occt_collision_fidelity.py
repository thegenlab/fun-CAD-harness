"""OCCT-backed helpers for authoritative STEP/collider fidelity checks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Sequence


def _fail(code: str, message: str) -> Exception:
    from .models import MechanismValidationError

    return MechanismValidationError(code, message)


def shape_solids(shape: Any) -> list[Any]:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids: list[Any] = []
    while explorer.More():
        solids.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()
    return solids


def shape_bounds_mm(shape: Any) -> tuple[tuple[float, float, float], ...]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    # Keep validation grids independent of call order. Add_s() uses an
    # existing face triangulation by default, so merely meshing the same BRep
    # can change its reported bounds by the mesh deflection. Fidelity checks
    # call this helper both before and after meshing; use exact geometry and
    # deliberately ignore cached triangulation in both cases.
    BRepBndLib.AddOptimal_s(
        shape,
        box,
        False,  # useTriangulation
        False,  # useShapeTolerance
    )
    if box.IsVoid():
        raise _fail(
            "COLLISION_FIDELITY_BREP_BOUNDS_FAILED",
            "OCCT could not bound a STEP collision shape",
        )
    values = tuple(float(value) for value in box.Get())
    return (
        (values[0], values[1], values[2]),
        (values[3], values[4], values[5]),
    )


def shape_facts_m(shape: Any) -> dict[str, Any]:
    import numpy as np
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    bounds_mm = shape_bounds_mm(shape)
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, properties)
    volume_m3 = abs(float(properties.Mass())) * 1.0e-9
    center = properties.CentreOfMass()
    inertia = properties.MatrixOfInertia()
    inertia_m5 = [
        [
            float(inertia.Value(row + 1, column + 1)) * 1.0e-15
            for column in range(3)
        ]
        for row in range(3)
    ]
    bounds_m = np.asarray(bounds_mm, dtype=float) * 1.0e-3
    return {
        "watertight": True,
        "bounds_m": bounds_m.tolist(),
        "characteristic_length_m": float(
            np.linalg.norm(bounds_m[1] - bounds_m[0])
        ),
        "volume_m3": volume_m3,
        "centroid_m": [
            float(center.X()) * 1.0e-3,
            float(center.Y()) * 1.0e-3,
            float(center.Z()) * 1.0e-3,
        ],
        "inertia_unit_density_kg_m2": inertia_m5,
        "source": "occt_brep_volume_properties",
    }


def mesh_shape_m(shape: Any, destination: Path) -> Any:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    import trimesh

    destination.parent.mkdir(parents=True, exist_ok=True)
    mesher = BRepMesh_IncrementalMesh(shape, 0.20, False, 0.50, True)
    mesher.Perform()
    if not mesher.IsDone():
        raise _fail(
            "COLLISION_COMPONENT_MESH_FAILED",
            "OCCT could not mesh a STEP collision shape",
        )
    StlAPI_Writer().Write(shape, str(destination))
    if not destination.is_file() or destination.stat().st_size == 0:
        raise _fail(
            "COLLISION_COMPONENT_MESH_FAILED",
            "OCCT did not export a STEP collision shape",
        )
    mesh = trimesh.load_mesh(destination, process=True, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.process(validate=True)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise _fail(
            "COLLISION_COMPONENT_MESH_EMPTY",
            "a STEP collision shape produced an empty mesh",
        )
    mesh.apply_scale(1.0e-3)
    mesh.remove_unreferenced_vertices()
    return mesh


def _null_face_surface_samples_m(
    face: Any,
    *,
    maximum_spacing_m: float,
) -> Any:
    import numpy as np
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepClass import BRepClass_FaceClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt2d

    surface = BRepAdaptor_Surface(face)
    first_u = float(surface.FirstUParameter())
    last_u = float(surface.LastUParameter())
    first_v = float(surface.FirstVParameter())
    last_v = float(surface.LastVParameter())
    if not all(
        math.isfinite(value)
        for value in (first_u, last_u, first_v, last_v)
    ):
        raise _fail(
            "COLLISION_FIDELITY_NULL_FACE_UNBOUNDED",
            "an untriangulated STEP face has unbounded parameters",
        )
    probe_points: list[tuple[float, float, float]] = []
    for u_value in np.linspace(first_u, last_u, 9):
        for v_value in np.linspace(first_v, last_v, 9):
            point = surface.Value(float(u_value), float(v_value))
            probe_points.append(
                (float(point.X()), float(point.Y()), float(point.Z()))
            )
    physical_span_m = max(
        float(np.linalg.norm(np.ptp(np.asarray(probe_points), axis=0))) * 1.0e-3,
        1.0e-6,
    )
    samples_per_axis = max(
        17,
        min(257, int(math.ceil(physical_span_m / maximum_spacing_m)) + 1),
    )
    accepted: list[tuple[float, float, float]] = []
    for u_value in np.linspace(first_u, last_u, samples_per_axis):
        for v_value in np.linspace(first_v, last_v, samples_per_axis):
            classifier = BRepClass_FaceClassifier(
                face,
                gp_Pnt2d(float(u_value), float(v_value)),
                1.0e-7,
            )
            if classifier.State() not in (TopAbs_IN, TopAbs_ON):
                continue
            point = surface.Value(float(u_value), float(v_value))
            accepted.append(
                (
                    float(point.X()) * 1.0e-3,
                    float(point.Y()) * 1.0e-3,
                    float(point.Z()) * 1.0e-3,
                )
            )
    if not accepted:
        raise _fail(
            "COLLISION_FIDELITY_NULL_FACE_UNSAMPLED",
            "an untriangulated STEP face produced no classified samples",
        )
    return np.asarray(accepted, dtype=float)


def shape_surface_points_m(
    shape: Any,
    *,
    maximum_spacing_m: float,
    maximum_point_count: int = 4096,
) -> Any:
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    mesher = BRepMesh_IncrementalMesh(shape, 0.20, False, 0.50, True)
    mesher.Perform()
    if not mesher.IsDone():
        raise _fail(
            "COLLISION_FIDELITY_BREP_MESH_FAILED",
            "OCCT could not triangulate STEP collision surfaces",
        )
    point_sets: list[Any] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = TopoDS.Face_s(face_explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            point_sets.append(
                _null_face_surface_samples_m(
                    face,
                    maximum_spacing_m=maximum_spacing_m,
                )
            )
        else:
            transformation = location.Transformation()
            nodes = []
            for index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(index).Transformed(transformation)
                nodes.append(
                    (
                        float(point.X()) * 1.0e-3,
                        float(point.Y()) * 1.0e-3,
                        float(point.Z()) * 1.0e-3,
                    )
                )
            if nodes:
                point_sets.append(np.asarray(nodes, dtype=float))
        face_explorer.Next()

    edge_points: list[tuple[float, float, float]] = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge_s(edge_explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        first = float(curve.FirstParameter())
        last = float(curve.LastParameter())
        if math.isfinite(first) and math.isfinite(last):
            for parameter in np.linspace(first, last, 17):
                point = curve.Value(float(parameter))
                edge_points.append(
                    (
                        float(point.X()) * 1.0e-3,
                        float(point.Y()) * 1.0e-3,
                        float(point.Z()) * 1.0e-3,
                    )
                )
        edge_explorer.Next()
    if edge_points:
        point_sets.append(np.asarray(edge_points, dtype=float))
    if not point_sets:
        raise _fail(
            "COLLISION_FIDELITY_BREP_SURFACE_EMPTY",
            "a STEP collision shape has no sampled BRep surface",
        )
    points = np.vstack(point_sets)
    points = np.unique(np.round(points, decimals=12), axis=0)
    if len(points) < 4:
        raise _fail(
            "COLLISION_FIDELITY_BREP_SURFACE_DEGENERATE",
            "a STEP collision shape has fewer than four surface points",
        )
    if maximum_point_count < 96:
        raise ValueError("maximum_point_count must be at least 96")
    if len(points) > maximum_point_count:
        indices = np.linspace(
            0,
            len(points) - 1,
            maximum_point_count,
            dtype=int,
        )
        points = points[indices]
    return points


def distance_points_to_shape_m(shape: Any, points_m: Any) -> Any:
    import numpy as np
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.gp import gp_Pnt

    distances: list[float] = []
    for point in np.asarray(points_m, dtype=float):
        vertex = BRepBuilderAPI_MakeVertex(
            gp_Pnt(
                float(point[0]) * 1.0e3,
                float(point[1]) * 1.0e3,
                float(point[2]) * 1.0e3,
            )
        ).Vertex()
        query = BRepExtrema_DistShapeShape(shape, vertex)
        query.Perform()
        if not query.IsDone():
            raise _fail(
                "COLLISION_FIDELITY_BREP_DISTANCE_FAILED",
                "OCCT could not evaluate a mesh-to-BRep distance",
            )
        distances.append(float(query.Value()) * 1.0e-3)
    return np.asarray(distances, dtype=float)


def _nearest_mesh_distances(mesh: Any, points: Any) -> Any:
    from .collider_fidelity import _nearest_distances

    return _nearest_distances(mesh, points)


def hybrid_shape_distance_evaluator(
    shape: Any,
    source_mesh_m: Any,
    *,
    exact_query_trigger_m: float,
) -> Callable[[Any], Any]:
    """Use bounded triangle distance, then exact OCCT outside a tiny band."""

    def evaluate(points: Any) -> Any:
        import numpy as np

        values = _nearest_mesh_distances(source_mesh_m, points)
        candidates = np.flatnonzero(values > exact_query_trigger_m)
        if len(candidates):
            values[candidates] = distance_points_to_shape_m(
                shape,
                np.asarray(points, dtype=float)[candidates],
            )
        return values

    return evaluate


def occupancy_iou_brep_vs_brep_union(
    source_shape: Any,
    component_union_shape: Any,
    *,
    component_shapes: Sequence[Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Measure solid occupancy with the exact BRep intersection volume.

    A global voxel grid is not a stable authority for thin mechanisms: a wheel
    can occupy only seven cells at the standard pitch, so one boundary cell
    changes IoU from 1.0 to 6/7.  Both authoritative shapes already exist as
    OCCT BReps here; use their Boolean common and exact volume properties.
    """

    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepGProp import BRepGProp
    from OCP.BOPAlgo import BOPAlgo_Builder
    from OCP.GProp import GProp_GProps
    from OCP.TopTools import TopTools_ListOfShape

    def volume_m3(shape: Any) -> float:
        properties = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, properties)
        return abs(float(properties.Mass())) * 1.0e-9

    source_volume = volume_m3(source_shape)
    collider_volume = volume_m3(component_union_shape)
    if (
        not math.isfinite(source_volume)
        or not math.isfinite(collider_volume)
        or source_volume <= 0.0
        or collider_volume <= 0.0
    ):
        raise _fail(
            "COLLISION_FIDELITY_OCCUPANCY_EMPTY",
            "BRep occupancy comparison requires two finite positive volumes",
        )

    smaller_volume = min(source_volume, collider_volume)
    numerical_tolerance = max(smaller_volume * 1.0e-6, 1.0e-18)
    intersection_operand = component_union_shape
    method = "occt_brep_boolean_intersection_over_union"
    partition_evidence: dict[str, Any] = {}
    if component_shapes is not None:
        shapes = tuple(component_shapes)
        if not shapes:
            raise _fail(
                "COLLISION_FIDELITY_OCCUPANCY_EMPTY",
                "component-aware occupancy requires at least one BRep",
            )
        arguments = TopTools_ListOfShape()
        for shape in shapes:
            arguments.Append(shape)
        builder = BOPAlgo_Builder()
        builder.SetArguments(arguments)
        builder.SetNonDestructive(True)
        builder.SetRunParallel(False)
        builder.Perform()
        partitioned_union = builder.Shape()
        partitioned_solids = shape_solids(partitioned_union)
        if (
            builder.HasErrors()
            or not partitioned_solids
            or not BRepCheck_Analyzer(partitioned_union).IsValid()
        ):
            raise _fail(
                "COLLISION_FIDELITY_OCCUPANCY_GENERAL_FUSE_FAILED",
                "OCCT could not partition the authored component union",
            )
        partitioned_volume = volume_m3(partitioned_union)
        if (
            not math.isfinite(partitioned_volume)
            or abs(partitioned_volume - collider_volume) > numerical_tolerance
        ):
            raise _fail(
                "COLLISION_FIDELITY_OCCUPANCY_GENERAL_FUSE_INVALID",
                "OCCT component partition changed the collider-union volume; "
                f"union={collider_volume!r}, partition={partitioned_volume!r}, "
                f"tolerance={numerical_tolerance!r}",
            )
        # Common() can silently discard nearly all coincident material when
        # its second argument is one heavily split, pairwise-fused solid.  A
        # General Fuse keeps the same union volume but exposes non-overlapping
        # Boolean cells, which Common() handles deterministically.
        intersection_operand = partitioned_union
        method = "occt_brep_general_fuse_intersection_over_union"
        partition_evidence = {
            "component_shape_count": len(shapes),
            "component_partition_solid_count": len(partitioned_solids),
            "component_partition_volume_m3": partitioned_volume,
            "component_partition_volume_delta_m3": (
                partitioned_volume - collider_volume
            ),
        }

    operation = BRepAlgoAPI_Common(source_shape, intersection_operand)
    operation.Build()
    if not operation.IsDone():
        raise _fail(
            "COLLISION_FIDELITY_OCCUPANCY_BOOLEAN_FAILED",
            "OCCT could not intersect the source and component-union BReps",
        )
    raw_intersection_volume = volume_m3(operation.Shape())
    if (
        not math.isfinite(raw_intersection_volume)
        or raw_intersection_volume < 0.0
        or raw_intersection_volume > smaller_volume + numerical_tolerance
    ):
        raise _fail(
            "COLLISION_FIDELITY_OCCUPANCY_BOOLEAN_INVALID",
            "OCCT returned an invalid source/component intersection volume; "
            f"source={source_volume!r}, collider={collider_volume!r}, "
            f"intersection={raw_intersection_volume!r}, "
            f"tolerance={numerical_tolerance!r}",
        )
    intersection_volume = min(raw_intersection_volume, smaller_volume)
    union_volume = source_volume + collider_volume - intersection_volume
    if not math.isfinite(union_volume) or union_volume <= 0.0:
        raise _fail(
            "COLLISION_FIDELITY_OCCUPANCY_EMPTY",
            "BRep occupancy comparison produced no positive union volume",
        )
    iou = intersection_volume / union_volume
    return iou, {
        "method": method,
        "source_volume_m3": source_volume,
        "component_union_volume_m3": collider_volume,
        "raw_intersection_volume_m3": raw_intersection_volume,
        "intersection_volume_m3": intersection_volume,
        "intersection_clamp_m3": max(
            0.0, raw_intersection_volume - intersection_volume
        ),
        "union_volume_m3": union_volume,
        "numerical_tolerance_m3": numerical_tolerance,
        **partition_evidence,
    }
