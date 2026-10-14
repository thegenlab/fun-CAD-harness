"""Bounded CAD measurements for print_delivery; never consumes author PASS flags."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import sys

import numpy as np
import trimesh

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import _step_source
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.source_queries import unsigned_distance
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.static_delivery import binding, read
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.static_delivery_worker import facts, consume_matches

EXPORT_TOLERANCE_MM = .05
VOLUME_RELATIVE_TOLERANCE = .005


def transform(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            or not math.isclose(float(np.linalg.det(matrix[:3, :3])), 1, abs_tol=1e-8)):
        raise ValueError("Print placement must be a finite proper rigid transform, without scale or reflection")
    return matrix


def samples(mesh):
    return np.vstack((mesh.vertices, mesh.triangles.mean(axis=1), mesh.vertices[mesh.edges_unique].mean(axis=1)))


def closed_mesh(path):
    mesh = trimesh.load(path, force="mesh", process=True)
    if (not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight
            or not mesh.is_winding_consistent or mesh.volume <= 0):
        raise ValueError("Print export is not a closed consistently oriented positive-volume mesh: " + str(path))
    return mesh


def compare_mesh(reference, exported, exact_volume_mm3, *, label="export"):
    # Vertex/edge/face samples are an export tessellation check, never a claim
    # of exact continuous BRep equivalence or collider fidelity acceptance.
    forward = float(unsigned_distance(exported, samples(reference)).max())
    reverse = float(unsigned_distance(reference, samples(exported)).max())
    volume_error = abs(exported.volume - exact_volume_mm3) / exact_volume_mm3
    if max(forward, reverse) > EXPORT_TOLERANCE_MM or volume_error > VOLUME_RELATIVE_TOLERANCE:
        raise ValueError(f"Export/source mismatch [{label}]: sampled distances {forward:g}/{reverse:g} mm; volume error {volume_error:g}")
    return {"source_to_export_sample_max_mm": forward, "export_to_source_sample_max_mm": reverse,
            "relative_volume_error": float(volume_error), "continuous_surface_bound": False}


def source_mesh(root, relative, solid_ids=None):
    source = _step_source(root / relative)
    parts = [part for part in source.parts if solid_ids is None or part.solid_id in solid_ids]
    if not parts or (solid_ids is not None and {p.solid_id for p in parts} != set(solid_ids)):
        raise ValueError("Source solid selection is empty or incomplete")
    mesh = trimesh.util.concatenate([part.mesh for part in parts])
    mesh.apply_scale(1000)
    return mesh, sum(part.volume_m3 for part in parts) * 1e9


def unique(rows, key, label):
    values = {row[key]: row for row in rows}
    if len(values) != len(rows):
        raise ValueError(label + " has duplicate IDs")
    return values


def quantity_rows(value):
    # These are the three inventory structures delivered by the current pilots.
    rows = value if isinstance(value, list) else value.get("parts", value.get("items"))
    if not isinstance(rows, list):
        raise ValueError("Quantity list must contain part records")
    result = {}
    for row in rows:
        key = row.get("id", row.get("part"))
        if not isinstance(key, str) or key in result:
            raise ValueError("Quantity list has missing or duplicate part IDs")
        result[key] = row
    return result


def inspect(payload):
    import build123d as b
    import OCP
    request = payload["request"]
    original, rebuilt = Path(payload["design"]), Path(payload["rebuilt_design"])
    scene = read(original / "scene.json")
    fresh_scene = read(rebuilt / "scene.json")
    for field in ("schema_version", "units", "bodies", "joints", "tests", "gravity_m_s2"):
        if scene.get(field) != fresh_scene.get(field):
            raise ValueError("Rebuilding source changes physical scene field: " + field)
    bodies = unique(scene["bodies"], "id", "Scene")
    parts = unique(request["parts"], "id", "Print inventory")
    excluded = set(request["excluded_context_body_ids"])
    if not excluded <= set(bodies) or any(not bodies[key].get("fixed") for key in excluded):
        raise ValueError("Only explicitly fixed external context bodies can be excluded")
    represented = [row["body_id"] for row in parts.values() if row.get("body_id")]
    if len(represented) != len(set(represented)) or set(represented) != set(bodies) - excluded:
        raise ValueError("Each non-context scene body must correspond to exactly one printed part")
    if not parts or not request["plates"] or not request["documents"]:
        raise ValueError("Print inventory, layouts and documentation must be present")
    quantities = quantity_rows(read(original / request["quantity_file"]))
    external_rows = set(quantities)-set(parts)
    if (set(parts)-set(quantities) or not external_rows <= excluded
            or any(quantities[key].get("category") != "external_fixture" for key in external_rows)):
        raise ValueError("Quantity list does not match all and only the delivered print instances")
    for key, row in parts.items():
        quantity = quantities[key]
        if quantity.get("quantity") != 1 or quantity.get("stl") != row["stl"]:
            raise ValueError("Quantity/STL correspondence differs for " + key)
    parsed = []
    for relative in request["source_files"]:
        if Path(relative).suffix == ".py":
            ast.parse((original / relative).read_text(encoding="utf-8-sig"), filename=relative)
            parsed.append(relative)
    if not parsed:
        raise ValueError("No readable modeling source delivered")
    limit = np.asarray(request["max_part_mm"], dtype=float)
    bed = np.asarray(request["bed_mm"], dtype=float)
    if limit.shape != (3,) or bed.shape != (2,) or np.any(limit <= 0) or np.any(bed <= 0):
        raise ValueError("Invalid task print envelope")
    density = float(request["density_kg_m3"])
    if not math.isfinite(density) or density <= 0:
        raise ValueError("Invalid task material density")
    for key in represented:
        if not math.isclose(float(bodies[key].get("density_kg_m3", 7800)), density, abs_tol=1e-9):
            raise ValueError("Scene density differs from declared print material: " + key)
    expected_assembly = []
    print_meshes = {}
    measured = []
    rebuilt_files = {"scene.json", request["assembly_step"]}
    for key, row in parts.items():
        if row["category"] not in {"machine", "test", "coupon"} or int(row["quantity"]) != 1:
            raise ValueError("List every physical print instance once, with its category")
        body_id = row.get("body_id")
        if body_id:
            body = bodies[body_id]
            if row["source"] != body["source"] or row.get("solid_ids") != body.get("solid_ids"):
                raise ValueError("Print/source ownership disagrees with the physical scene: " + key)
            if row["category"] == "coupon":
                raise ValueError("A fit coupon cannot be a simulated machine body")
        elif row["category"] != "coupon":
            raise ValueError("Only separately accounted coupons may lack a scene body")
        shape = b.import_step(original / row["source"])
        fresh_shape = b.import_step(rebuilt / row["source"])
        original_facts = [facts(solid) for solid in shape.solids()]
        consume_matches(original_facts, [facts(solid) for solid in fresh_shape.solids()])
        selected = row.get("solid_ids")
        selected_facts = [value for index, value in enumerate(original_facts, 1)
                          if selected is None or f"solid_{index}" in selected]
        if len(selected_facts) != 1:
            raise ValueError("Each physical print part must be one connected source solid: " + key)
        if body_id:
            expected_assembly.extend(selected_facts)
        source, volume = source_mesh(original, row["source"], row.get("solid_ids"))
        source.apply_transform(transform(row["source_to_print_mm"]))
        exported = closed_mesh(original / row["stl"])
        fresh_export = closed_mesh(rebuilt / row["stl"])
        for mesh in (exported, fresh_export):
            groups = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
            positive = sum(mesh.submesh([faces], append=True, repair=False).volume > 0 for faces in groups)
            if positive != 1:
                raise ValueError("Single-part STL contains disconnected material shells: " + key)
        correspondence = compare_mesh(source, exported, volume, label=f"{key}: original STL")
        fresh_correspondence = compare_mesh(source, fresh_export, volume, label=f"{key}: rebuilt STL")
        if np.any(exported.extents > limit + EXPORT_TOLERANCE_MM) or abs(exported.bounds[0, 2]) > EXPORT_TOLERANCE_MM:
            raise ValueError("Part does not fit the print envelope or does not rest at Z=0: " + key)
        print_meshes[key] = exported
        rebuilt_files.update((row["source"], row["stl"]))
        measured.append({"id": key, "body_id": body_id, "category": row["category"], "quantity": 1,
                         "source": row["source"], "stl": row["stl"], "size_mm": exported.extents.tolist(),
                         "volume_mm3": volume, "source_to_print_mm": row["source_to_print_mm"],
                         "source_correspondence": correspondence, "rebuilt_correspondence": fresh_correspondence})
        print(json.dumps({"checked_part": key, "size_mm": exported.extents.tolist()}), flush=True)
    assemblies = {}
    for name, root in (("original", original), ("rebuilt", rebuilt)):
        assembly = b.import_step(root / request["assembly_step"])
        inventory = [facts(solid) for solid in assembly.solids()]
        assemblies[name] = {"solid_count": len(inventory), "source_matches": consume_matches(expected_assembly, inventory)}
    glb = trimesh.load(original / request["assembly_glb"], force="scene", process=False)
    glb_mesh = glb.to_mesh()
    if not len(glb_mesh.faces) or not np.isfinite(glb_mesh.vertices).all():
        raise ValueError("GLB cannot be opened as finite assembly geometry")
    glb_mesh.apply_scale(float(request["glb_units_to_mm"]))
    glb_mesh.apply_transform(transform(request["glb_to_assembly_mm"]))
    assembled_meshes, assembled_volume = [], 0.
    for key in represented:
        mesh, volume = source_mesh(original, bodies[key]["source"], bodies[key].get("solid_ids"))
        assembled_meshes.append(mesh)
        assembled_volume += volume
    glb_check = compare_mesh(trimesh.util.concatenate(assembled_meshes), glb_mesh, assembled_volume, label="assembled GLB")
    layout_rows, placed = [], []
    for plate in request["plates"]:
        bounds, meshes = [], []
        for placement in plate["placements"]:
            key = placement["part_id"]
            mesh = print_meshes[key].copy()
            mesh.apply_transform(transform(placement["print_to_plate_mm"]))
            if (np.any(mesh.bounds[0, :2] < -EXPORT_TOLERANCE_MM)
                    or np.any(mesh.bounds[1, :2] > bed + EXPORT_TOLERANCE_MM)
                    or abs(mesh.bounds[0, 2]) > EXPORT_TOLERANCE_MM):
                raise ValueError("Layout leaves the declared bed or lifts a part off Z=0")
            for other_id, other_bounds in bounds:
                separation = np.maximum(np.maximum(mesh.bounds[0, :2] - other_bounds[1, :2],
                                                   other_bounds[0, :2] - mesh.bounds[1, :2]), 0.)
                if np.linalg.norm(separation) < float(request["part_spacing_mm"]) - 1e-6:
                    raise ValueError(f"Layout spacing below required value: {key}/{other_id}")
            bounds.append((key, mesh.bounds))
            meshes.append(mesh)
            placed.append(key)
        if not meshes:
            raise ValueError("Empty print plate")
        expected = trimesh.util.concatenate(meshes)
        exported = closed_mesh(original / plate["stl"])
        layout_rows.append({"id": plate["id"], "parts": [p["part_id"] for p in plate["placements"]],
                            "stl": plate["stl"], "bounds_mm": exported.bounds.tolist(),
                            "correspondence": compare_mesh(expected, exported, sum(mesh.volume for mesh in meshes), label=f"{plate['id']}: layout STL")})
    if sorted(placed) != sorted(parts):
        raise ValueError("Layouts must include every print instance exactly once")
    sphere_rows = []
    if request.get("sphere_group"):
        group = request["sphere_group"]
        radius = float(group["diameter_mm"]) / 2
        for key in group["body_ids"]:
            shape = b.import_step(original / bodies[key]["source"])
            box = shape.bounding_box()
            if (len(shape.solids()) != 1 or len(shape.faces()) != 1
                    or shape.faces()[0].geom_type != b.GeomType.SPHERE
                    or not np.allclose([box.size.X, box.size.Y, box.size.Z], radius * 2, atol=1e-6)
                    or not math.isclose(shape.volume, 4 * math.pi * radius**3 / 3, rel_tol=1e-7)
                    or bodies[key].get("fixed")
                    or any(key in (joint["body_a"], joint["body_b"]) for joint in scene["joints"])):
                raise ValueError("Test bead is not the requested independent free sphere: " + key)
            sphere_rows.append({"body_id": key, "diameter_mm": radius * 2, "volume_mm3": shape.volume, "unconstrained": True})
    documents = []
    for relative in request["documents"]:
        path = original / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Missing or empty delivery document: " + relative)
        if path.suffix.lower() in {".md", ".html", ".txt", ".json"}:
            path.read_text(encoding="utf-8-sig")
        documents.append({"path": str(path), "bytes": path.stat().st_size})
    runtime_paths = {str(Path(sys.executable).resolve())}
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if path and name.startswith(("build123d", "OCP", "trimesh", "numpy", "rtree")) and Path(path).is_file():
            runtime_paths.add(str(Path(path).resolve()))
    return {"completed": True, "errors": [], "parts": measured, "assemblies": assemblies,
            "glb_correspondence": glb_check, "plates": layout_rows, "sphere_measurements": sphere_rows,
            "source_parse": parsed, "physical_scene_consistent_after_rebuild": True,
            "counts": {category: sum(row["category"] == category for row in measured) for category in ("machine", "test", "coupon")},
            "density_kg_m3": density, "quantity_file": str(original / request["quantity_file"]),
            "documents": documents, "rebuilt_files": sorted(rebuilt_files), "runtime_bindings": binding(runtime_paths),
            "runtime_versions": {"python": sys.version, "build123d": b.__version__, "OCP": getattr(OCP, "__version__", "unknown")},
            "export_tolerance_mm": EXPORT_TOLERANCE_MM, "volume_relative_tolerance": VOLUME_RELATIVE_TOLERANCE,
            "scope": "Executed CAD rebuild, inventory correspondence, sampled export geometry, task print envelopes and mapped plate spacing. Quantity-list/document consistency, actual insertability, material suitability and criterion-scope adequacy require independent review. No mechanical or real-print acceptance."}


def main():
    payload = read(sys.argv[1])
    try:
        result = inspect(payload)
    except Exception as exc:
        result = {"completed": False, "errors": [f"{type(exc).__name__}: {exc}"]}
    Path(payload["output"]).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    sys.exit(main())
