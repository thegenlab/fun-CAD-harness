"""Build run-bound lightweight GLB evidence scenes without rerunning simulation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

from .path_resolution import resolve_workspace_reference


@dataclass
class SceneMeshes:
    fixed: list[trimesh.Trimesh]
    moving: list[trimesh.Trimesh]
    source_kind: str
    source_paths: list[Path]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load_mesh(path, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"mesh scene is empty: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"mesh is empty: {path}")
    return loaded


def _solid_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 10_000


def _quaternion_matrix(wxyz: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(wxyz), dtype=float)
    values /= max(float(np.linalg.norm(values)), 1e-15)
    w, x, y, z = values
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    return transform


def _absolute_workspace_path(workspace: Path, value: str | Path) -> Path:
    candidate = resolve_workspace_reference(
        workspace,
        value,
        require_within_workspace=True,
        require_exists=True,
    )
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_per_solid(run: Path, capsule_id: str) -> SceneMeshes | None:
    root = run / "derived_adapters" / capsule_id
    fixed_paths = sorted(
        (root / "fixed.nonconvex_solid_stl").glob("*.stl"), key=_solid_number
    )
    moving_paths = sorted(
        (root / "moving.nonconvex_solid_stl").glob("*.stl"), key=_solid_number
    )
    if not fixed_paths or not moving_paths:
        return None
    fixed = [_load_mesh(path) for path in fixed_paths]
    moving = [_load_mesh(path) for path in moving_paths]
    for path, mesh in zip(fixed_paths, fixed):
        mesh.metadata["quest_name"] = f"fixed_{path.stem}"
    for path, mesh in zip(moving_paths, moving):
        mesh.metadata["quest_name"] = f"moving_{path.stem}"
    return SceneMeshes(
        fixed=fixed,
        moving=moving,
        source_kind="step-derived-per-solid-stl",
        source_paths=[*fixed_paths, *moving_paths],
    )


def _load_package_step_directory(
    workspace: Path,
    run: Path,
    package_dir: Path,
    capsule_id: str,
    *,
    fixed_groups: set[str] | None = None,
    extra_source_paths: Iterable[Path] = (),
) -> SceneMeshes | None:
    """Tessellate one frozen package STEP into viewer-ready fixed/moving meshes."""

    package_dir = package_dir.resolve()
    package_dir.relative_to(workspace)
    step_path = package_dir / "assembly.step"
    contract_path = package_dir / "assembly_contract.json"
    if not step_path.is_file() or not contract_path.is_file():
        return None

    # Some pip OCP wheels bundle an older libexpat.  If OCP loads first it can
    # occupy the process-wide SONAME and later break Python's pyexpat import.
    # Load the environment-matched standard-library extension before OCP.
    import xml.parsers.expat  # noqa: F401

    from build123d import import_step

    contract = _read_json(contract_path)
    role_by_solid: dict[str, str] = {}
    fixed_group_ids = set(fixed_groups or ()) | set(
        contract.get("grounded_rigid_group_ids") or ()
    )
    for part in contract.get("parts") or []:
        group_id = str(part.get("rigid_group_id") or "")
        role = (
            "fixed"
            if group_id in fixed_group_ids or part.get("role") == "fixed"
            else "moving"
        )
        for solid_id in part.get("source_solid_ids") or []:
            role_by_solid[str(solid_id)] = role

    shape = import_step(str(step_path))
    solids = list(shape.solids())
    fixed: list[trimesh.Trimesh] = []
    moving: list[trimesh.Trimesh] = []
    for index, solid in enumerate(solids, start=1):
        solid_id = f"solid_{index}"
        vertices, faces = solid.tessellate(0.45, 0.12)
        mesh = trimesh.Trimesh(
            vertices=np.asarray([tuple(vertex) for vertex in vertices], dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        if not len(mesh.vertices) or not len(mesh.faces):
            raise ValueError(f"STEP tessellation is empty for {solid_id}: {step_path}")
        role = role_by_solid.get(solid_id)
        if role not in {"fixed", "moving"}:
            raise ValueError(f"assembly contract does not own {solid_id}: {contract_path}")
        mesh.metadata["quest_name"] = f"{role}_{solid_id}"
        (fixed if role == "fixed" else moving).append(mesh)
    if not fixed or not moving:
        raise ValueError(f"STEP scene lacks fixed or moving solids: {step_path}")
    return SceneMeshes(
        fixed=fixed,
        moving=moving,
        source_kind="package-step-per-solid-tessellation",
        source_paths=[*extra_source_paths, step_path, contract_path],
    )


def _load_package_step(
    workspace: Path, run: Path, capsule_id: str
) -> SceneMeshes | None:
    """Tessellate package STEP solids when family runtimes do not emit STL adapters."""

    plan_path = run / "plans" / f"{capsule_id}.json"
    if not plan_path.is_file():
        return None
    plan = _read_json(plan_path)
    package_ref = plan.get("package_dir")
    if not package_ref:
        return None
    try:
        package_dir = resolve_workspace_reference(
            workspace,
            package_ref,
            require_within_workspace=True,
        )
    except ValueError:
        return None
    return _load_package_step_directory(
        workspace,
        run,
        package_dir,
        capsule_id,
        fixed_groups=set(plan.get("fixed_group_ids") or []),
        extra_source_paths=(plan_path,),
    )


def _load_trace_collision(
    workspace: Path, run: Path, capsule_id: str
) -> SceneMeshes:
    trace_root = run / "traces" / capsule_id
    candidates = sorted(trace_root.glob("*/physical.trace.json"))
    if not candidates:
        candidates = sorted(trace_root.glob("*/reduced.trace.json"))
    if not candidates:
        raise ValueError("run has neither per-solid STL nor a saved trace collision scene")
    trace_path = candidates[0]
    trace = _read_json(trace_path)
    samples = trace.get("samples") or []
    if not samples:
        raise ValueError(f"saved trace has no samples: {trace_path}")
    collision = trace["collision_geometry"]
    fixed_path = _absolute_workspace_path(workspace, collision["fixed"]["derived"])
    moving_path = _absolute_workspace_path(workspace, collision["moving"]["derived"])
    fixed = _load_mesh(fixed_path).copy()
    moving = _load_mesh(moving_path).copy()

    fixed_pose = samples[0].get("fixed_pose")
    fixed_transform = (
        _quaternion_matrix(fixed_pose["rotation_abs_quat_wxyz"])
        if isinstance(fixed_pose, dict)
        else np.eye(4)
    )
    fixed_transform[:3, 3] = np.asarray(
        (
            fixed_pose["position_abs_m"]
            if isinstance(fixed_pose, dict)
            else collision["fixed"]["body_origin_abs_m"]
        ),
        dtype=float,
    )
    fixed.apply_transform(fixed_transform)
    fixed.apply_scale(1000.0)

    pose = samples[0]["moving_pose"]
    moving_transform = _quaternion_matrix(pose["rotation_abs_quat_wxyz"])
    moving_transform[:3, 3] = np.asarray(pose["position_abs_m"], dtype=float)
    moving.apply_transform(moving_transform)
    moving.apply_scale(1000.0)
    fixed.metadata["quest_name"] = "fixed_collision_group"
    moving.metadata["quest_name"] = "moving_collision_group"
    return SceneMeshes(
        fixed=[fixed],
        moving=[moving],
        source_kind="saved-trace-step-derived-collision-mesh",
        source_paths=[trace_path, fixed_path, moving_path],
    )


def load_scene_meshes(
    *, workspace_root: Path, run_dir: Path, capsule_id: str
) -> SceneMeshes:
    """Load the same-source CAD scene before any collider-derived fallback.

    The engineering overview must preserve the assembly placement authored in
    ``assembly.step``.  Per-solid collider meshes remain a fallback for legacy
    runs and are still used by the dedicated physics-evidence scene.
    """

    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    run.relative_to(workspace)
    package_step = _load_package_step(workspace, run, capsule_id)
    if package_step is not None:
        return package_step
    per_solid = _load_per_solid(run, capsule_id)
    return per_solid or _load_trace_collision(workspace, run, capsule_id)


def _revision_package_directory(workspace: Path, run: Path) -> Path:
    inventories = sorted(
        (run / "design_submissions").glob("*/submission_inventory.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for inventory_path in inventories:
        if inventory_path.is_symlink() or not inventory_path.is_file():
            continue
        inventory = _read_json(inventory_path)
        package_ref = inventory.get("package_dir")
        if not isinstance(package_ref, str) or not package_ref:
            continue
        package = resolve_workspace_reference(
            workspace,
            package_ref,
            require_within_workspace=True,
            require_exists=True,
        )
        package.relative_to(run)
        if package.is_dir() and not package.is_symlink():
            return package
    raise ValueError("revision has no frozen design package")


def load_revision_scene_meshes(
    *, workspace_root: Path, run_dir: Path
) -> SceneMeshes:
    """Load one revision's draggable design mesh without writing into the run."""

    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    run.relative_to(workspace)
    for plan_path in sorted((run / "plans").glob("*.json")):
        plan = _read_json(plan_path)
        capsule_id = str(plan.get("capsule_id") or plan_path.stem)
        try:
            return load_scene_meshes(
                workspace_root=workspace,
                run_dir=run,
                capsule_id=capsule_id,
            )
        except (KeyError, OSError, ValueError):
            continue
    package = _revision_package_directory(workspace, run)
    result = _load_package_step_directory(
        workspace,
        run,
        package,
        "revision",
    )
    if result is None:
        raise ValueError("revision package has no STEP scene")
    return result


def _interactive_scene(
    scene_meshes: SceneMeshes, *, capsule_id: str
) -> tuple[trimesh.Scene, list[dict[str, str]]]:
    scene = trimesh.Scene()
    manifest_nodes: list[dict[str, str]] = []
    colors = {
        "fixed": np.asarray([113, 132, 155, 255], dtype=np.uint8),
        "moving": np.asarray([217, 164, 65, 255], dtype=np.uint8),
    }
    for role, meshes in (
        ("fixed", scene_meshes.fixed),
        ("moving", scene_meshes.moving),
    ):
        for index, original in enumerate(meshes, start=1):
            mesh = original.copy()
            raw_name = str(original.metadata.get("quest_name") or f"{role}_{index}")
            name = raw_name if raw_name.startswith(role) else f"{role}_{raw_name}"
            mesh.metadata.update(
                {
                    "name": name,
                    "questRole": role,
                    "capsuleRole": role,
                    "capsuleId": capsule_id,
                }
            )
            mesh.visual.face_colors = np.tile(colors[role], (len(mesh.faces), 1))
            scene.add_geometry(mesh, geom_name=name, node_name=name)
            manifest_nodes.append({"node": name, "role": role})
    return scene, manifest_nodes


def interactive_glb_bytes(
    scene_meshes: SceneMeshes, *, capsule_id: str
) -> bytes:
    """Serialize one revision mesh for the browser without a filesystem write."""

    scene, _nodes = _interactive_scene(scene_meshes, capsule_id=capsule_id)
    return trimesh.exchange.gltf.export_glb(scene, include_normals=True)


def export_interactive_glb(
    scene_meshes: SceneMeshes,
    output: Path,
    *,
    workspace_root: Path,
    run_dir: Path,
    capsule_id: str,
) -> dict[str, Any]:
    scene, manifest_nodes = _interactive_scene(
        scene_meshes, capsule_id=capsule_id
    )
    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    manifest = {
        "schema_version": "0.1",
        "artifact_kind": "interactive_evidence_scene",
        "source_run": str(run.relative_to(workspace).as_posix()),
        "capsule_id": capsule_id,
        "source_kind": scene_meshes.source_kind,
        "source_files": [
            {
                "path": str(path.resolve().relative_to(workspace).as_posix()),
                "sha256": _sha256(path),
            }
            for path in scene_meshes.source_paths
        ],
        "nodes": manifest_nodes,
        "output": str(output.resolve().relative_to(workspace).as_posix()),
        "output_sha256": _sha256(output),
        "viewer_semantics": {
            "fixed": "gray context or fixed structure",
            "moving": "highlighted moving mechanism",
        },
    }
    manifest_path = output.with_suffix(".scene.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest
