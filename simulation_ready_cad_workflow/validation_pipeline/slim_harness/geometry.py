"""Prepare faithful, metre-based rigid-body geometry for the slim Harness.

Sources are already assembled in world coordinates. STEP units are read by
OCCT; ``scene.units`` supplies the numerical unit of mesh files. Original
surfaces remain, with conformal midpoint refinement and source-contained
analytic axial material supplements. No hull or hole filling is made.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from ..mechanism.io import write_json_atomic
from .implementation import source_sha256
from .source_mesh_refinement import MAX_SOURCE_TRIANGLE_EDGE_M, refine_source_triangles
from .source_analytic_material import contained_axial_capsules


DEFAULT_DENSITY_KG_M3 = 7800.0
GEOMETRY_IMPLEMENTATION_SHA256 = hashlib.sha256(''.join(source_sha256(name) for name in (
    "geometry.py", "source_mesh_refinement.py", "source_analytic_material.py",
)).encode('ascii')).hexdigest()
# Control face interiors as well as topological edges. Without that control,
# a long diagonal on a trimmed cylinder missed its surface by 0.119 mm even
# with 0.01 mm absolute edge deflection. A globally smaller angle overmeshes
# the other surfaces without addressing that missing interior check.
# The shared mesh target is 0.05 mm, as selected for ordinary Harness work.
# This discretization target never replaces the original-solid motion audit.
STEP_DEFLECTION_MM = 0.05
STEP_ANGULAR_TOLERANCE_RAD = 0.12
# Solids declared in one rigid body must share material or touch within this
# gap. It is a modelling-tolerance allowance (twice the mesh target), well
# below any pin/bearing fit clearance, so a deliberately cut clearance or a
# stop pad hanging in space is not silently welded to its body.
RIGID_BODY_CONTACT_GAP_MM = 0.1
_FORMATS = {".step", ".stp", ".stl", ".obj", ".glb"}


class GeometryError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class _Part:
    solid_id: str
    mesh: trimesh.Trimesh
    volume_m3: float
    center_m: np.ndarray
    inertia_unit_density: np.ndarray
    node_name: str
    analytic_material: tuple[dict, ...] = ()
    shape: Any = None  # original millimetre BRep for STEP sources; None for mesh sources


@dataclass
class _Source:
    path: Path
    sha256: str
    stamp: tuple[int, int, int]
    parts: list[_Part]
    kind: str
    unit_description: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stamp(path: Path) -> tuple[int, int, int]:
    value = path.stat()
    return value.st_size, value.st_mtime_ns, value.st_ino


def _closed_mesh(mesh: Any, label: str) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise GeometryError("EMPTY_SOURCE_GEOMETRY", label)
    if not np.isfinite(mesh.vertices).all():
        raise GeometryError("NONFINITE_SOURCE_GEOMETRY", label)
    mesh = mesh.copy()
    # STEP/STL/GLB may repeat the same geometric vertex at face/normal seams.
    # Welding coincident positions changes indexing only, not the surface.
    mesh.merge_vertices(digits_vertex=12)
    # Spherical poles can leave exactly zero-area triangles after seam welding.
    # Drop only degenerate indexing/surfaces; retain every nonzero-area face.
    faces = np.asarray(mesh.faces)
    repeated = np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0, axis=1)
    degenerate = repeated | (mesh.area_faces == 0.0)
    if np.any(degenerate):
        mesh.update_faces(~degenerate)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise GeometryError("SOURCE_MESH_NOT_CLOSED", f"{label}: source must be closed and consistently oriented")
    if not math.isfinite(float(mesh.volume)) or float(mesh.volume) <= 0:
        raise GeometryError("SOURCE_VOLUME_NOT_POSITIVE", label)
    return mesh


def _tessellate_step_mm(shape: Any) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate an unchanged millimetre BRep with absolute OCCT deflection."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepTools import BRepTools
    from OCP.IMeshTools import IMeshTools_Parameters
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    # Discard only cached discretization, which may have been made with a
    # relative tolerance by another reader. The analytic BRep is not modified.
    BRepTools.Clean_s(shape)
    parameters = IMeshTools_Parameters()
    parameters.Deflection = STEP_DEFLECTION_MM
    parameters.Angle = STEP_ANGULAR_TOLERANCE_RAD
    parameters.Relative = False
    parameters.InParallel = True
    parameters.ControlSurfaceDeflection = True
    parameters.EnableControlSurfaceDeflectionAllSurfaces = True
    mesher = BRepMesh_IncrementalMesh(shape, parameters)
    if not mesher.IsDone():
        raise GeometryError("SOURCE_TRIANGULATION_FAILED", "OCCT absolute-deflection meshing failed")
    vertices, triangles = [], []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None:
            raise GeometryError("SOURCE_TRIANGULATION_FAILED", "An original BRep face has no triangulation")
        transform = location.Transformation()
        offset = len(vertices)
        for index in range(1, triangulation.NbNodes() + 1):
            point = triangulation.Node(index).Transformed(transform)
            vertices.append((point.X(), point.Y(), point.Z()))
        reverse = face.Orientation() == TopAbs_REVERSED
        for index in range(1, triangulation.NbTriangles() + 1):
            a, b, c = triangulation.Triangle(index).Get()
            if reverse:
                b, c = c, b
            triangles.append((offset + a - 1, offset + b - 1, offset + c - 1))
        explorer.Next()
    return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=int)


def _refine_mesh(mesh: trimesh.Trimesh, label: str) -> trimesh.Trimesh:
    vertices, faces = refine_source_triangles(mesh.vertices, mesh.faces)
    if len(faces) == len(mesh.faces):
        return mesh
    return _closed_mesh(trimesh.Trimesh(vertices=vertices, faces=faces, process=False), label)


def _step_source(path: Path, *, inventory_check=None) -> _Source:
    import xml.parsers.expat  # load the stdlib DLL before the OCCT wheel
    from ..intake.step_shape_cache import load_step_shape
    from ..mechanism.occt_collision_fidelity import shape_facts_m

    stamp = _stamp(path)
    loaded = load_step_shape(path, unit="MM")
    if inventory_check is not None:
        # Reject excessive construction inventories before tessellating them.
        inventory_check([f"solid_{index}" for index in range(1, len(loaded.solids) + 1)])
    parts = []
    for index, raw in enumerate(loaded.solids, start=1):
        solid_id = f"solid_{index}"
        if not loaded.inventory[index - 1].get("valid_brep"):
            raise GeometryError("INVALID_SOURCE_BREP", f"{path.name}/{solid_id}")
        vertices, faces = _tessellate_step_mm(raw)
        mesh = _closed_mesh(trimesh.Trimesh(
            vertices=vertices * 0.001, faces=faces, process=False,
        ), f"{path.name}/{solid_id}")
        mesh = _refine_mesh(mesh, f"{path.name}/{solid_id}")
        facts = shape_facts_m(raw)
        volume = float(facts["volume_m3"])
        if not math.isfinite(volume) or volume <= 0:
            raise GeometryError("SOURCE_VOLUME_NOT_POSITIVE", f"{path.name}/{solid_id}")
        parts.append(_Part(solid_id, mesh, volume, np.asarray(facts["centroid_m"]),
                           np.asarray(facts["inertia_unit_density_kg_m2"]), solid_id,
                           tuple(contained_axial_capsules(raw)), raw))
    if not parts:
        raise GeometryError("EMPTY_SOURCE_GEOMETRY", str(path))
    if _stamp(path) != stamp:
        raise GeometryError("SOURCE_CHANGED_DURING_PREPARATION", str(path))
    return _Source(path, loaded.source_sha256, stamp, parts, "step", "STEP-declared units imported as millimetres, converted to metres")


def _mesh_source(path: Path, scale: float, *, inventory_check=None) -> _Source:
    stamp = _stamp(path)
    digest = _sha256(path)
    loaded = trimesh.load(path, force="scene", process=False)
    if not isinstance(loaded, trimesh.Scene):
        loaded = trimesh.Scene(loaded)
    if inventory_check is not None:
        inventory_check([f"solid_{index}" for index in range(1, len(loaded.graph.nodes_geometry) + 1)])
    parts = []
    # Inventory graph instances, not geometry definitions: one mesh may appear
    # more than once under different transforms in an authored GLB scene.
    for index, node in enumerate(sorted(loaded.graph.nodes_geometry, key=str), start=1):
        transform, geometry_name = loaded.graph[node]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        mesh.apply_scale(scale)
        solid_id = f"solid_{index}"
        mesh = _closed_mesh(mesh, f"{path.name}/{solid_id}")
        properties = trimesh.triangles.mass_properties(mesh.triangles, density=1.0)
        mesh = _refine_mesh(mesh, f"{path.name}/{solid_id}")
        parts.append(_Part(solid_id, mesh, float(properties.volume),
                           np.asarray(properties.center_mass), np.asarray(properties.inertia), str(node)))
    if not parts:
        raise GeometryError("EMPTY_SOURCE_GEOMETRY", str(path))
    if _stamp(path) != stamp:
        raise GeometryError("SOURCE_CHANGED_DURING_PREPARATION", str(path))
    return _Source(path, digest, stamp, parts, "mesh", f"scene.units; scale to metres = {scale:g}")


def _color(value: Any) -> tuple[list[float] | None, np.ndarray | None]:
    if value is None:
        return None, None
    if isinstance(value,list) and len(value)==3:
        value=[*value,255 if any(isinstance(x,(int,float)) and x>1 for x in value) else 1]
    if not isinstance(value, list) or len(value) != 4 or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
        for item in value
    ):
        raise GeometryError("INVALID_BODY_COLOR", "color must contain four finite RGBA values")
    normalized = np.asarray(value, dtype=float)
    if np.any(normalized > 1):
        normalized = normalized / 255.0
    if np.any(normalized < 0) or np.any(normalized > 1):
        raise GeometryError("INVALID_BODY_COLOR", "color must use RGBA in 0..1 or 0..255")
    return normalized.tolist(), np.rint(normalized * 255).astype(np.uint8)


def _write_obj(path: Path, mesh: trimesh.Trimesh) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# original source surface; body-local metres\n")
        for vertex in mesh.vertices:
            stream.write("v " + " ".join(format(float(value), ".17g") for value in vertex) + "\n")
        for face in mesh.faces:
            stream.write("f " + " ".join(str(int(value) + 1) for value in face) + "\n")
    temporary.replace(path)
    restored = trimesh.load(path, force="mesh", process=False)
    if (not isinstance(restored, trimesh.Trimesh) or restored.vertices.shape != mesh.vertices.shape
            or not np.array_equal(restored.faces, mesh.faces)):
        raise GeometryError("MESH_EXPORT_TOPOLOGY_CHANGED", str(path))
    error = float(np.max(np.linalg.norm(restored.vertices - mesh.vertices, axis=1)))
    if error > 1.0e-12:
        raise GeometryError("MESH_EXPORT_COORDINATES_CHANGED", str(path))
    return {"sha256": _sha256(path), "vertex_count": len(mesh.vertices), "triangle_count": len(mesh.faces),
            "export_vertex_error_max_m": error}


def _body_properties(parts: list[_Part], density: float) -> tuple[float, np.ndarray, np.ndarray]:
    masses = np.asarray([part.volume_m3 * density for part in parts])
    mass = float(masses.sum())
    center = sum((part.center_m * weight for part, weight in zip(parts, masses)), np.zeros(3)) / mass
    inertia = np.zeros((3, 3))
    for part, weight in zip(parts, masses):
        offset = part.center_m - center
        inertia += part.inertia_unit_density * density + weight * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )
    if (not math.isfinite(mass) or mass <= 0 or not np.isfinite(center).all()
            or not np.isfinite(inertia).all() or np.linalg.eigvalsh(inertia).min() <= 0):
        raise GeometryError("INVALID_MASS_PROPERTIES", "source mass properties must be finite and physically positive")
    return mass, center, inertia


def _material_connected(first: Any, second: Any) -> bool:
    """True when two millimetre BRep solids touch within RIGID_BODY_CONTACT_GAP_MM or share material."""
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    distance = BRepExtrema_DistShapeShape(first, second)
    distance.Perform()
    if distance.IsDone() and distance.Value() <= RIGID_BODY_CONTACT_GAP_MM:
        return True
    # Boundaries can stay apart while one solid lies inside the other's material.
    common = BRepAlgoAPI_Common(first, second)
    common.Build()
    if not common.IsDone():
        return False
    properties = GProp_GProps()
    BRepGProp.VolumeProperties_s(common.Shape(), properties)
    return abs(float(properties.Mass())) > 1e-6


def _rigid_body_solid_groups(parts: list[_Part]) -> list[list[str]]:
    """Group a declared rigid body's STEP solids by actual material connection."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    boxes = []
    for part in parts:
        box = Bnd_Box()
        BRepBndLib.Add_s(part.shape, box)
        box.Enlarge(RIGID_BODY_CONTACT_GAP_MM)
        boxes.append(box)
    adjacency = {index: set() for index in range(len(parts))}
    for left in range(len(parts)):
        for right in range(left + 1, len(parts)):
            if boxes[left].IsOut(boxes[right]):
                continue
            if _material_connected(parts[left].shape, parts[right].shape):
                adjacency[left].add(right)
                adjacency[right].add(left)
    groups, remaining = [], set(range(len(parts)))
    while remaining:
        pending, group = [min(remaining)], []
        while pending:
            index = pending.pop()
            if index in remaining:
                remaining.remove(index)
                group.append(index)
                pending.extend(adjacency[index] & remaining)
        groups.append([parts[index].solid_id for index in sorted(group)])
    return groups


def _check_rigid_body_connectivity(body_id: str, parts: list[_Part]) -> None:
    """A rigid body may only weld solids that are physically joined; a floating solid never enters physics."""
    if len(parts) < 2 or any(part.shape is None for part in parts):
        return  # single solids are trivially connected; mesh sources carry no BRep to measure
    groups = _rigid_body_solid_groups(parts)
    if len(groups) > 1:
        raise GeometryError(
            "RIGID_BODY_MATERIAL_DISCONNECTED",
            f"{body_id}: its solids form {len(groups)} material-disconnected groups "
            f"(contact gap > {RIGID_BODY_CONTACT_GAP_MM:g} mm and no shared volume): "
            + " | ".join("[" + ", ".join(group) + "]" for group in groups)
            + ". Join them with real material or declare the separated solids as their own bodies.")


def _prepare(scene_path: Path, output_dir: Path, *, construction_limits=None) -> dict[str, Any]:
    from ..benchmark_sandbox.visual_policy import WORKER_NO_VISUALS, runtime_policy

    visual_enabled = runtime_policy() != WORKER_NO_VISUALS
    scene = json.loads(scene_path.read_text(encoding="utf-8-sig"))
    if not isinstance(scene, dict) or scene.get("schema_version") != "slim_scene_v1":
        raise GeometryError("INVALID_SCENE_SCHEMA", "expected slim_scene_v1")
    units = scene.get("units")
    if units not in {"mm", "m"}:
        raise GeometryError("INVALID_SCENE_UNITS", "units must be mm or m")
    specs = scene.get("bodies")
    if not isinstance(specs, list) or not specs:
        raise GeometryError("SCENE_BODIES_MISSING", "scene must contain at least one body")
    if construction_limits is not None:
        from .construction_limits import scene_preflight, check_geometry_counts
        scene_preflight(scene, construction_limits)
    sources: dict[Path, _Source] = {}
    source_users: dict[Path, list[dict[str, Any]]] = {}
    ids: set[str] = set()
    resolved_specs: list[tuple[dict[str, Any], Path]] = []
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("id"), str) or not spec["id"].strip():
            raise GeometryError("INVALID_BODY_ID", "every body needs a non-empty id")
        if spec["id"] in ids:
            raise GeometryError("DUPLICATE_BODY_ID", spec["id"])
        ids.add(spec["id"])
        spec.setdefault("fixed", False)
        if not isinstance(spec.get("fixed"), bool):
            raise GeometryError("INVALID_BODY_FIXED", f"{spec['id']}: fixed must be boolean")
        reference = spec.get("source")
        if not isinstance(reference, str) or not reference or Path(reference).is_absolute():
            raise GeometryError("INVALID_SOURCE_PATH", f"{spec['id']}: source must be relative to the scene file")
        path = (scene_path.parent / reference).resolve()
        if not path.is_relative_to(scene_path.parent):
            raise GeometryError("SOURCE_OUTSIDE_DESIGN", f"{spec['id']}: source must stay below the scene directory")
        if not path.is_file() or path.suffix.lower() not in _FORMATS:
            raise GeometryError("SOURCE_UNAVAILABLE", str(path))
        source_users.setdefault(path, []).append(spec)
        resolved_specs.append((spec, path))
    known_inventory = {}
    for path in source_users:
        def inventory_check(solid_ids, current_path=path):
            for spec in source_users[current_path]:
                selected = spec.get("solid_ids", solid_ids)
                if isinstance(selected, list):
                    known_inventory[spec["id"]] = len(selected)
            if construction_limits is not None:
                from .construction_limits import check_geometry_counts
                check_geometry_counts(known_inventory, construction_limits)
        kwargs = {"inventory_check": inventory_check} if construction_limits is not None else {}
        sources[path] = (_step_source(path, **kwargs) if path.suffix.lower() in {".step", ".stp"}
                         else _mesh_source(path, 0.001 if units == "mm" else 1.0, **kwargs))

    assignments: dict[str, list[_Part]] = {}
    coverage: dict[Path, dict[str, str]] = {path: {} for path in sources}
    for spec, path in resolved_specs:
        source = sources[path]
        requested = spec.get("solid_ids")
        if requested is None:
            if len(source_users[path]) != 1:
                raise GeometryError("SOURCE_SOLID_ASSIGNMENT_REQUIRED", f"{path.name}: shared sources require explicit solid_ids on every body")
            requested = [part.solid_id for part in source.parts]
        if not isinstance(requested, list) or not requested or any(not isinstance(value, str) for value in requested):
            raise GeometryError("INVALID_SOURCE_SOLID_IDS", spec["id"])
        inventory = {part.solid_id: part for part in source.parts}
        assigned = []
        for solid_id in requested:
            if solid_id not in inventory:
                raise GeometryError("UNKNOWN_SOURCE_SOLID", f"{spec['id']}: {solid_id}")
            if solid_id in coverage[path]:
                raise GeometryError("SOURCE_SOLID_ASSIGNED_TWICE", f"{path.name}/{solid_id}: {coverage[path][solid_id]} and {spec['id']}")
            coverage[path][solid_id] = spec["id"]
            assigned.append(inventory[solid_id])
        assignments[spec["id"]] = assigned
    for path, source in sources.items():
        missing = {part.solid_id for part in source.parts} - coverage[path].keys()
        if missing:
            raise GeometryError("SOURCE_SOLIDS_UNASSIGNED", f"{path.name}: {', '.join(sorted(missing))}")
    for body_id, parts in assignments.items():
        _check_rigid_body_connectivity(body_id, parts)

    construction_geometry = None
    if construction_limits is not None:
        from .construction_limits import check_geometry_counts
        construction_geometry = check_geometry_counts(
            {key: len(parts) for key, parts in assignments.items()}, construction_limits,
            triangle_counts={key: sum(len(part.mesh.faces) for part in parts) for key, parts in assignments.items()})

    output_dir.mkdir(parents=True, exist_ok=True)
    display = trimesh.Scene() if visual_enabled else None
    bodies = []
    for index, (spec, path) in enumerate(resolved_specs, start=1):
        source = sources[path]
        parts = assignments[spec["id"]]
        density = spec.get("density_kg_m3", DEFAULT_DENSITY_KG_M3)
        if isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(density) or density <= 0:
            raise GeometryError("INVALID_BODY_DENSITY", spec["id"])
        mass, center, inertia = _body_properties(parts, float(density))
        normalized_color, rgba = _color(spec.get("color"))
        meshes = []
        components = []
        for part_index, part in enumerate(parts, start=1):
            local = part.mesh.copy()
            local.vertices = np.asarray(local.vertices) - center
            meshes.append(local)
            component_file = output_dir / "source_components" / f"body-{index:03d}-solid-{part_index:03d}.obj"
            exported = _write_obj(component_file, local)
            components.append({"solid_id": part.solid_id, "source_node": part.node_name,
                               "mesh_file": str(component_file), **exported})
            if display is not None:
                shown = part.mesh.copy()
                if rgba is not None:
                    shown.visual.face_colors = rgba
                node = f"body_{index:03d}__{part.solid_id}"
                display.add_geometry(shown, node_name=node, geom_name=node,
                                     metadata={"body_id": spec["id"], "source_solid_id": part.solid_id})
        combined = trimesh.util.concatenate(meshes)
        analytic_material = [
            {**item, "source_solid_id": part.solid_id,
             "center_local_m": (np.asarray(item["center_source_m"])-center).tolist()}
            for part in parts for item in part.analytic_material
        ]
        mesh_file = output_dir / f"body-{index:03d}.obj"
        exported = _write_obj(mesh_file, combined)
        bodies.append({
            "id": spec["id"], "fixed": spec["fixed"], "color": normalized_color,
            "position_m": center.tolist(), "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "source_position_m": center.tolist(), "density_kg_m3": float(density),
            "mass_kg": mass, "volume_m3": sum(part.volume_m3 for part in parts),
            "inertia_kg_m2": inertia.tolist(), "mesh_file": str(mesh_file),
            "audit_mesh_file": str(mesh_file), "mesh_sha256": exported["sha256"],
            "source_file": str(path), "source_sha256": source.sha256, "source_stat": list(source.stamp),
            "source_kind": source.kind, "source_solid_ids": [part.solid_id for part in parts],
            "source_components": components,
            "source_analytic_material": analytic_material,
            "mass_properties_source": "occt_brep" if source.kind == "step" else "closed_original_mesh",
            "mass_combination": "sum_source_properties_with_parallel_axis_theorem",
            "fidelity": {
                "representation": "original_source_surface_mesh", "convex_hull_used": False,
                "decomposition_used": False, "decimation_used": False, "topology_repaired": False,
                "coincident_vertex_seams_welded": True,
                "step_linear_deflection_m": STEP_DEFLECTION_MM * 0.001 if source.kind == "step" else None,
                "step_angular_tolerance_rad": STEP_ANGULAR_TOLERANCE_RAD if source.kind == "step" else None,
                "step_all_surface_deflection_control": True if source.kind == "step" else None,
                "brep_surface_distance_independently_measured": False if source.kind == "step" else None,
                "collider_and_audit_use_same_original_mesh": not bool(analytic_material),
                "original_mesh_retained_in_collider": True,
                "source_mesh_refinement": "conformal midpoint subdivision of original facets",
                "source_mesh_max_edge_m": MAX_SOURCE_TRIANGLE_EDGE_M,
                "contained_analytic_material_supplements": len(analytic_material),
                "export_vertex_error_max_m": exported["export_vertex_error_max_m"],
                "triangle_count": exported["triangle_count"],
            },
            "distance_query": {"space": "world_metres", "inside_material_distance_m": 0.0,
                               "backend": "original_step_brep" if source.kind == "step" else "original_closed_mesh"},
        })
    model = None
    if display is not None:
        model = output_dir / "model.glb"
        temporary = output_dir / "model.glb.pending"
        temporary.write_bytes(trimesh.exchange.gltf.export_glb(display, include_normals=True))
        temporary.replace(model)
    for path, source in sources.items():
        if _stamp(path) != source.stamp:
            raise GeometryError("SOURCE_CHANGED_DURING_PREPARATION", str(path))
    result = {
        "schema_version": "slim_geometry_v1", "scene_file": str(scene_path), "scene_sha256": _sha256(scene_path),
        "units": "m", "input_units": units, "bodies": bodies, "model_file": str(model) if model is not None else None,
        "source_bindings": [{"source_file": str(source.path), "source_sha256": source.sha256,
                             "source_kind": source.kind, "unit_conversion": source.unit_description,
                             "solid_count": len(source.parts), "solid_owners": coverage[path]}
                            for path, source in sources.items()],
        "errors": [],
    }
    if construction_geometry is not None:
        result["construction_geometry"] = construction_geometry
    return result


def prepare_geometry(scene_path: Path, output_dir: Path, *, construction_limits=None) -> dict[str, Any]:
    """Return a serializable geometry result; errors never produce a ready body."""
    scene_path, output_dir = Path(scene_path).resolve(), Path(output_dir).resolve()
    try:
        result = _prepare(scene_path, output_dir, construction_limits=construction_limits)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        result = {"schema_version": "slim_geometry_v1", "scene_file": str(scene_path), "units": "m",
                  "bodies": [], "model_file": None, "source_bindings": [],
                  "errors": [{"code": getattr(exc, "code", "GEOMETRY_PREPARATION_FAILED"), "message": str(exc),
                              **getattr(exc, "details", {})}]}
    result["geometry_implementation_sha256"] = GEOMETRY_IMPLEMENTATION_SHA256
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "geometry.json", result)
    return result


@lru_cache(maxsize=64)
def _bound_step(path: str, digest: str, stamp: tuple[int, int, int]):
    import xml.parsers.expat
    from ..intake.step_shape_cache import load_step_shape
    loaded = load_step_shape(Path(path), unit="MM")
    if loaded.source_sha256 != digest:
        raise GeometryError("SOURCE_BINDING_CHANGED", path)
    return loaded


@lru_cache(maxsize=128)
def _bound_mesh(path: str, digest: str, stamp: tuple[int, int, int]):
    if _sha256(Path(path)) != digest:
        raise GeometryError("SOURCE_BINDING_CHANGED", path)
    mesh = trimesh.load(path, force="mesh", process=False)
    return _closed_mesh(mesh, path)


def geometry_distance_to_point(bodyrecord: dict[str, Any], worldpoint_m: Any) -> float:
    """Distance to source material (zero inside), with optional live body pose.

Callers may replace position_m/rotation_wxyz with a saved/current body pose;
source_position_m remains the original source-frame COM. STEP queries use
the exact BRep, while mesh queries use the bound, unmodified source surface.
"""
    point = np.asarray(worldpoint_m, dtype=float)
    quaternion = np.asarray(bodyrecord["rotation_wxyz"], dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all() or quaternion.shape != (4,) or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) == 0:
        raise GeometryError("INVALID_DISTANCE_QUERY", "point and body pose must be finite")
    rotation = trimesh.transformations.quaternion_matrix(quaternion)[:3, :3]
    local = rotation.T @ (point - np.asarray(bodyrecord["position_m"], dtype=float))
    if bodyrecord["source_kind"] == "step":
        from ..mechanism.occt_collision_fidelity import distance_points_to_shape_m
        path = Path(bodyrecord["source_file"])
        stamp = _stamp(path)
        if stamp != tuple(bodyrecord["source_stat"]):
            raise GeometryError("SOURCE_BINDING_CHANGED", str(path))
        source = _bound_step(str(path), bodyrecord["source_sha256"], stamp)
        original_point = local + np.asarray(bodyrecord["source_position_m"], dtype=float)
        return min(float(distance_points_to_shape_m(source.solids[int(solid_id.split("_")[1]) - 1], [original_point])[0])
                   for solid_id in bodyrecord["source_solid_ids"])
    distances = []
    for component in bodyrecord["source_components"]:
        path = Path(component["mesh_file"])
        mesh = _bound_mesh(str(path), component["sha256"], _stamp(path))
        if bool(mesh.contains([local])[0]):
            return 0.0
        distances.append(float(trimesh.proximity.closest_point(mesh, [local])[1][0]))
    return min(distances)
