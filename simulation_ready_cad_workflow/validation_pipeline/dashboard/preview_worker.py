"""Isolated source-geometry conversion used by the dashboard's single worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def _write(path: Path, value: Any) -> None:
    from ..mechanism.io import write_json_atomic

    write_json_atomic(path, value)


def _export(scene: Any, path: Path) -> None:
    import trimesh

    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    temporary = path.with_name(path.name + ".pending")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _step_scene(source: Path) -> Any:
    import xml.parsers.expat  # load the matching stdlib DLL before OCP
    import numpy as np
    import trimesh
    from build123d import Solid
    from ..intake.step_shape_cache import load_step_shape

    scene = trimesh.Scene()
    shapes = load_step_shape(source).solids
    colors = [(164, 174, 165, 255), (102, 123, 135, 255), (185, 165, 128, 255), (136, 147, 161, 255)]
    for index, raw in enumerate(shapes, 1):
        vertices, faces = Solid(raw).tessellate(0.05, 0.12)
        mesh = trimesh.Trimesh(vertices=np.asarray([tuple(v) for v in vertices]) * 0.001,
                               faces=np.asarray(faces), process=False)
        if not len(mesh.faces):
            raise ValueError(f"原始零件 solid_{index} 无法生成表面网格")
        mesh.visual.face_colors = colors[(index - 1) % len(colors)]
        scene.add_geometry(mesh, node_name=f"solid_{index}", geom_name=f"solid_{index}")
    if not scene.geometry:
        raise ValueError("原始模型没有可显示的实体")
    return scene


def _scene_meshes(scene: Any) -> dict[str, Any]:
    meshes = {}
    for node in scene.graph.nodes_geometry:
        transform, geometry = scene.graph[node]
        mesh = scene.geometry[geometry].copy()
        mesh.apply_transform(transform)
        meshes[str(node)] = mesh
    return meshes


def _motion(source: Path, scene: Any, traces: list[Path], output: Path) -> dict[str, Any]:
    import numpy as np
    import trimesh
    from .scene_assets import _quaternion_matrix

    meshes = _scene_meshes(scene)
    source_digest = None
    choices = []
    for trace_path in traces:
        trace = _read(trace_path)
        collision = trace.get("collision_geometry") or {}
        bodies = collision.get("bodies") or (collision.get("replay_attachment_geometry") or {}).get("bodies") or []
        samples = trace.get("samples") or []
        if not bodies or not samples:
            continue
        if not samples[0].get("body_poses"):
            if not (samples[0].get("fixed_pose") and samples[0].get("moving_pose")):
                continue
            samples = [{**sample, "body_poses": {"fixed_boundary": sample.get("fixed_pose"),
                                                 "moving_group": sample.get("moving_pose")}} for sample in samples]
        expected = collision.get("assembly_sha256")
        if not isinstance(expected, str) or not expected:
            # Plain two-body traces bind their original STEP through the plan.
            from ..mechanism.io import canonical_sha256
            from ..mechanism.models import MechanismPlan

            plan_path = trace_path.parents[3] / "plans" / f"{trace_path.parents[1].name}.json"
            plan = _read(plan_path)
            if trace.get("plan_sha256") != canonical_sha256(MechanismPlan.from_dict(plan).to_dict()):
                continue
            expected = (plan.get("artifact_hashes") or {}).get(source.name)
            if not expected:
                continue
        if source_digest is None:
            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected != source_digest:
            continue
        # Prefer the task's functional motion over static/gravity diagnostics.
        preference = 0 if "task" in trace_path.parent.name or "roundtrip" in trace_path.parent.name else 1
        choices.append((preference, trace_path, trace, bodies, samples))
    if not choices:
        return {}
    _rank, trace_path, trace, bodies, samples = sorted(choices, key=lambda item: (item[0], str(item[1])))[0]
    motion_scene = trimesh.Scene()
    manifest_bodies = []
    body_corners = {}
    assigned: set[str] = set()
    for body in bodies:
        body_id = str(body.get("body_id") or "")
        source_ids = [str(value) for value in body.get("source_solid_ids") or []]
        if not body_id or not source_ids or any(identity not in meshes for identity in source_ids):
            raise ValueError("原模型与运动记录的零件归属不完整，保留静态预览")
        if assigned.intersection(source_ids):
            raise ValueError("运动记录重复分配原始零件，保留静态预览")
        assigned.update(source_ids)
        center = np.asarray(body.get("initial_center_abs_m") or [0, 0, 0], dtype=float)
        rotation = _quaternion_matrix(body.get("initial_rotation_abs_quat_wxyz") or [1, 0, 0, 0])
        transform = rotation.copy()
        transform[:3, 3] = center
        components = []
        corners = []
        for source_id in source_ids:
            mesh = meshes[source_id].copy()
            mesh.vertices = (np.asarray(mesh.vertices) - center) @ rotation[:3, :3]
            low, high = mesh.bounds
            corners.extend([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
            node = f"physics__{body_id}__{source_id}"
            motion_scene.add_geometry(mesh, node_name=node, geom_name=node, transform=transform)
            components.append({"node": node, "solid_id": source_id, "representation": "original_step_surface",
                               "triangle_count": len(mesh.faces)})
        body_corners[body_id] = np.asarray(corners)
        manifest_bodies.append({"body_id": body_id, "fixed": bool(body.get("fixed")),
                                "source_solid_ids": source_ids, "components": components,
                                "initial_center_abs_m": center.tolist(),
                                "initial_rotation_abs_quat_wxyz": body.get("initial_rotation_abs_quat_wxyz") or [1, 0, 0, 0]})
    # An unmapped part must not silently disappear or be invented as stationary.
    if assigned != set(meshes):
        raise ValueError("部分原始零件没有对应运动记录，保留完整静态模型")
    indices = sorted({0, len(samples) - 1, *np.linspace(0, len(samples) - 1, min(720, len(samples)), dtype=int).tolist()})
    frames = []
    for index in indices:
        sample = samples[index]
        poses = sample.get("body_poses") or {}
        if any(body["body_id"] not in poses for body in manifest_bodies):
            raise ValueError("运动记录缺少零件姿态，保留静态预览")
        frames.append({"time_s": sample.get("time_s", sample.get("time", sample.get("t", 0))),
                       "body_poses": poses})
    minimum = np.full(3, np.inf)
    maximum = np.full(3, -np.inf)
    for frame in frames:
        for body_id, corners in body_corners.items():
            pose = frame["body_poses"][body_id]
            rotation = _quaternion_matrix(pose["rotation_abs_quat_wxyz"])[:3, :3]
            points = corners @ rotation.T + np.asarray(pose["position_abs_m"])
            minimum = np.minimum(minimum, points.min(axis=0))
            maximum = np.maximum(maximum, points.max(axis=0))
    motion_path = output / "motion.glb"
    scene_path = output / "motion.scene.json"
    replay_path = output / "motion.replay.json"
    output.mkdir(parents=True, exist_ok=True)
    _export(motion_scene, motion_path)
    _write(scene_path, {"schema_version": "source_geometry_scene_v1", "source_kind": "original_geometry",
                        "coordinate_frame": "absolute_world_metres", "component_vertices": "body_local_metres",
                        "motion_bounds_abs_m": {"min": minimum.tolist(), "max": maximum.tolist()},
                        "bodies": manifest_bodies, "claims": {"scene_uses_original_geometry": True,
                        "scene_uses_collision_components_from_plan": False}})
    _write(replay_path, {"schema_version": "source_geometry_replay_v1", "source_kind": "original_geometry",
                         "source_trace": str(trace_path), "source_sample_count": len(samples),
                         "display_frame_count": len(frames), "landmarks": [],
                         "branches": {"physical": {"branch": "physical", "frames": frames,
                         "status": trace.get("status", "RECORDED"), "field_availability": {}, "maxima": {}}},
                         "claims": {"simulation_was_rerun_for_visuals": False,
                                    "rendered_geometry_is_original": True, "certification_authority": False}})
    stamp = trace_path.stat()
    return {"id": f"{trace_path.parents[1].name}/{trace_path.parent.name}", "load_case_id": trace_path.parent.name,
            "label": trace_path.parent.name.replace("_", " "),
            "motion_model_path": str(motion_path), "scene_path": str(scene_path), "replay_path": str(replay_path),
            "motion_trace": str(trace_path), "motion_frame_count": len(frames),
            "replay_version": f"{stamp.st_mtime_ns:x}-{stamp.st_size:x}"}


def _slim_scene(source: Path, bindings: dict[str, str]) -> Any:
    """Display original scene surfaces without compiling physics or colliders."""
    import trimesh
    import numpy as np
    value = _read(source)
    scene = trimesh.Scene()
    scale = .001 if value.get("units", "mm") == "mm" else 1.
    for body in value.get("bodies", []):
        path = Path(bindings.get(body["source"]) or source.parent / body["source"])
        step = path.suffix.lower() in {".step", ".stp"}
        parts = _step_scene(path) if step else trimesh.load(path, force="scene", process=False)
        selected = body.get("solid_ids")
        instances = [(name, geometry, None) for name, geometry in parts.geometry.items()] if step else [
            (node, parts.geometry[parts.graph[node][1]], parts.graph[node][0])
            for node in sorted(parts.graph.nodes_geometry)]
        for index, (node, geometry, transform) in enumerate(instances, 1):
            if selected and f"solid_{index}" not in selected:
                continue
            mesh = geometry.copy()
            if not step:
                mesh.apply_transform(transform)
                mesh.apply_scale(scale)
            color = body.get("color")
            if color:
                rgba = list(color[:3]) + [color[3] if len(color) > 3 else (1 if max(color) <= 1 else 255)]
                mesh.visual.face_colors = np.asarray(rgba) * (255 if max(rgba) <= 1 else 1)
            name = f"{body['id']}__{index}"
            scene.add_geometry(mesh, node_name=name, geom_name=name)
    if not scene.geometry:
        raise ValueError("场景尚无可预览的原始几何")
    return scene


def build(request: dict[str, Any]) -> dict[str, Any]:
    import trimesh

    source = Path(request["source"])
    output = Path(request["output"])
    output.mkdir(parents=True, exist_ok=True)
    model = Path(request.get("model_cache") or output / "model.glb")
    model.parent.mkdir(parents=True, exist_ok=True)
    if model.is_file() and model.stat().st_size > 20 and model.read_bytes()[:4] == b"glTF":
        scene = trimesh.load(model, force="scene", process=False)
    else:
        scene = (_slim_scene(source, request.get("source_bindings", {})) if source.suffix.lower() == ".json"
                 else _step_scene(source) if source.suffix.lower() in {".step", ".stp"}
                 else trimesh.load(source, force="scene", process=False))
        if not scene.geometry:
            raise ValueError("模型文件为空")
        _export(scene, model)
    result = {"model_path": str(model), "source_path": str(source), "part_count": len(scene.geometry),
              "triangle_count": sum(len(mesh.faces) for mesh in scene.geometry.values()),
              "certification_authority": False, "error": None}
    result["motions"] = []
    result["motion_errors"] = []
    traces = sorted([Path(path) for path in request.get("traces", [])],
                    key=lambda path: (not ("task" in path.parent.name or "roundtrip" in path.parent.name), str(path)))
    for motion_index, trace_path in enumerate(traces):
        try:
            directory = output / "motions" / f"m{motion_index:02d}"
            motion = _motion(source, scene, [trace_path], directory)
            if motion:
                result["motions"].append(motion)
        except Exception as exc:
            result["motion_errors"].append({"trace": str(trace_path), "error": str(exc)})
    if result["motions"]:
        result.update(result["motions"][0])
    return result


def main() -> int:
    request = _read(Path(sys.argv[1]))
    output = Path(request["output"])
    try:
        result = build(request)
        _write(output / "result.json", result)
        return 0
    except Exception as exc:
        _write(output / "result.json", {"error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
