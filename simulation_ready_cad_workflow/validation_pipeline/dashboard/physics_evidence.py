"""Build interactive, hash-bound physics replay assets from saved traces.

This module is presentation-only.  It never imports or invokes a simulator.
The original trace remains authoritative; replay payloads are bounded,
down-sampled indexes that keep source hashes and selected sample indices.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import trimesh

from .path_resolution import resolve_workspace_reference


MAX_REPLAY_FRAMES = 180
REPLAY_SCHEMA = "physics_evidence_replay_v4"
SCENE_SCHEMA = "physics_evidence_scene_v1"
COMPONENT_NODE_BINDING = "unique_per_collision_component_v3"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(workspace: Path, value: str | Path) -> Path:
    candidate = resolve_workspace_reference(
        workspace,
        value,
        require_within_workspace=True,
        require_exists=True,
    )
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _component_mesh_reference(component: dict[str, Any]) -> str:
    return str(
        component.get("mesh")
        or component.get("derived")
        or component.get("source")
        or ""
    )


def _relative(workspace: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace).as_posix()


def _scene_manifest_is_current(
    workspace: Path,
    scene_path: Path,
    manifest: dict[str, Any],
    collision: dict[str, Any],
) -> bool:
    if (
        manifest.get("schema_version") != SCENE_SCHEMA
        or manifest.get("component_node_binding") != COMPONENT_NODE_BINDING
        or not scene_path.is_file()
        or manifest.get("output_sha256") != _sha256(scene_path)
    ):
        return False
    expected: dict[str, str] = {}
    for body in collision.get("bodies") or []:
        for component in body.get("components") or []:
            try:
                path = _workspace_path(workspace, _component_mesh_reference(component))
            except (FileNotFoundError, ValueError):
                return False
            expected[_relative(workspace, path)] = _sha256(path)
    recorded = {
        str(item.get("path") or ""): str(item.get("sha256") or "")
        for item in manifest.get("source_files") or []
        if isinstance(item, dict)
    }
    return expected == recorded


def _safe_name(value: str) -> str:
    # Three.js PropertyBinding removes ``[].:/`` from loaded glTF node names.
    # Keeping those characters in the scene manifest made split collision
    # components impossible to bind to their saved body poses in the browser.
    clean = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return clean.strip("_") or "unnamed"


def _round_number(value: Any, digits: int = 9) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _vector(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    result = [_round_number(item) for item in value[:3]]
    return result if all(item is not None for item in result) else None


def _magnitude(value: Any) -> float:
    vector = _vector(value)
    if vector is None:
        return 0.0
    return float(math.sqrt(sum(item * item for item in vector)))


def _dict_numbers(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, float] = {}
    for key, item in value.items():
        number = _round_number(item)
        if number is not None:
            result[str(key)] = number
    return result


def _dict_vectors(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, list[float]] = {}
    for key, item in value.items():
        vector = _vector(item)
        if vector is not None:
            result[str(key)] = vector
    return result


def _pose(value: Any) -> dict[str, list[float]] | None:
    if not isinstance(value, dict):
        return None
    position = _vector(value.get("position_abs_m"))
    raw_quaternion = value.get("rotation_abs_quat_wxyz")
    if not isinstance(raw_quaternion, (list, tuple)) or len(raw_quaternion) < 4:
        return None
    quaternion = [_round_number(item) for item in raw_quaternion[:4]]
    if position is None or not all(item is not None for item in quaternion):
        return None
    return {"position_abs_m": position, "rotation_abs_quat_wxyz": quaternion}


def _moving_chain_state(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    text_fields = ("schema_version", "contract_sha256", "state_sha256")
    numeric_fields = (
        "input_angle_rad",
        "output_angle_rad",
        "input_surface_displacement_m",
        "output_surface_displacement_m",
        "force_coordinate_error_m",
        "phase_unwrapped_m",
        "phase_modulo_m",
        "input_phase_residual_m",
        "output_phase_residual_m",
    )
    result = {
        field: str(value.get(field) or "")
        for field in text_fields
    }
    for field in numeric_fields:
        number = _round_number(value.get(field), digits=12)
        if number is None:
            return None
        result[field] = number
    if not all(result[field] for field in text_fields):
        return None
    return result


def _peak_score(sample: dict[str, Any]) -> tuple[float, float, float, float]:
    drift = max(
        [
            *_dict_numbers(sample.get("joint_drifts_m")).values(),
            *_dict_numbers(sample.get("support_drifts_m")).values(),
            0.0,
        ]
    )
    external = max(
        _magnitude(sample.get("applied_force_abs_n")),
        _magnitude(sample.get("applied_torque_abs_nm")),
    )
    return (
        float(sample.get("contact_count") or 0.0),
        float(sample.get("max_bushing_reaction_force_n") or 0.0),
        drift,
        external,
    )


def select_frame_indices(
    samples: list[dict[str, Any]],
    *,
    max_frames: int = MAX_REPLAY_FRAMES,
    required_source_indices: Sequence[int] = (),
) -> list[int]:
    """Keep boundaries and physical extrema, then fill the time range uniformly."""

    count = len(samples)
    if count <= max_frames:
        return list(range(count))
    required = {0, count - 1}
    for raw_index in required_source_indices:
        index = int(raw_index)
        if 0 <= index < count:
            required.add(index)
    previous_phase = str(samples[0].get("phase_id") or "")
    for index, sample in enumerate(samples[1:], start=1):
        phase = str(sample.get("phase_id") or "")
        if phase != previous_phase:
            required.update({index - 1, index})
            previous_phase = phase
    scores = [_peak_score(sample) for sample in samples]
    for channel in range(4):
        required.add(max(range(count), key=lambda index: scores[index][channel]))

    remaining = max(0, max_frames - len(required))
    if remaining:
        for value in np.linspace(0, count - 1, remaining + 2)[1:-1]:
            required.add(int(round(float(value))))
    if len(required) < max_frames:
        for index in range(count):
            required.add(index)
            if len(required) >= max_frames:
                break
    return sorted(required)[:max_frames]


def _compact_frame(
    sample: dict[str, Any],
    source_index: int,
    *,
    family: dict[str, Any] | None = None,
) -> dict[str, Any]:
    family = family or {}
    poses = {
        str(body_id): compact
        for body_id, value in (sample.get("body_poses") or {}).items()
        if (compact := _pose(value)) is not None
    }
    force = _vector(sample.get("applied_force_abs_n"))
    torque = _vector(sample.get("applied_torque_abs_nm"))
    force_magnitude = _magnitude(force)
    torque_magnitude = _magnitude(torque)
    if force is not None and (force_magnitude > 1e-12 or torque is None):
        external_kind = "force"
        external_vector = force
        external_magnitude = force_magnitude
        external_unit = "N"
    elif torque is not None:
        external_kind = "torque"
        external_vector = torque
        external_magnitude = torque_magnitude
        external_unit = "N·m"
    else:
        external_kind = "none"
        external_vector = [0.0, 0.0, 0.0]
        external_magnitude = 0.0
        external_unit = ""
    active_input_body_id = str(
        sample.get("active_input_body_id")
        or family.get("input_body_id")
        or ""
    )
    active_input_point = (
        _vector(sample.get("active_input_point_abs_m"))
        or _vector(family.get("input_point_abs_m"))
    )
    input_loads: list[dict[str, Any]] = []
    if external_magnitude > 1e-12:
        input_loads.append(
            {
                "id": "primary",
                "body_id": active_input_body_id,
                "point_abs_m": active_input_point,
                "kind": external_kind,
                "vector_abs": external_vector,
                "magnitude": _round_number(external_magnitude),
                "unit": external_unit,
            }
        )
    additional_alarm_torque = (
        _round_number(sample.get("additional_alarm_torque_nm")) or 0.0
    )
    alarm_axis = _vector(family.get("alarm_input_axis_abs"))
    if abs(additional_alarm_torque) > 1e-12 and alarm_axis is not None:
        alarm_vector = [
            round(additional_alarm_torque * component, 9)
            for component in alarm_axis
        ]
        input_loads.append(
            {
                "id": "additional_alarm",
                "body_id": str(family.get("alarm_input_body_id") or ""),
                "point_abs_m": _vector(family.get("alarm_input_point_abs_m")),
                "kind": "torque",
                "vector_abs": alarm_vector,
                "magnitude": round(abs(additional_alarm_torque), 9),
                "unit": "N·m",
            }
        )
    contact_pairs = {
        str(key): int(value)
        for key, value in (sample.get("contact_pair_counts") or {}).items()
        if isinstance(value, (int, float))
    }
    result: dict[str, Any] = {
        "source_sample_index": source_index,
        "time_s": _round_number(sample.get("time_s")),
        "phase_id": str(sample.get("phase_id") or ""),
        "body_poses": poses,
        "external_load": {
            "kind": external_kind,
            "vector_abs": external_vector,
            "magnitude": _round_number(external_magnitude),
            "unit": external_unit,
        },
        "active_input_body_id": active_input_body_id,
        "active_input_point_abs_m": active_input_point,
        "input_loads": input_loads,
        "body_angular_velocities_abs_rad_s": _dict_vectors(
            sample.get("body_angular_velocities_abs_rad_s")
        ),
        "contact_count": int(sample.get("contact_count") or 0),
        "contact_pair_counts": contact_pairs,
        "max_bushing_reaction_force_n": _round_number(
            sample.get("max_bushing_reaction_force_n")
        ),
        "bushing_reaction_forces_n": _dict_numbers(
            sample.get("bushing_reaction_forces_n")
        ),
        "joint_drifts_m": _dict_numbers(sample.get("joint_drifts_m")),
        "support_drifts_m": _dict_numbers(sample.get("support_drifts_m")),
    }
    moving_chain = _moving_chain_state(sample.get("moving_chain_path_state"))
    if moving_chain is not None:
        result["moving_chain_path_state"] = moving_chain
    for key in ("setting_clutch_engaged", "alarm_setting_clutch_engaged"):
        if sample.get(key) is not None:
            result[key] = bool(sample.get(key))
    for key in (
        "generalized_progress_m",
        "generalized_angle_rad",
        "rack_progress_m",
        "pinion_angle_rad",
        "compound_input_angle_rad",
        "compound_output_angle_rad",
        "lateral_shift_m",
        "off_axis_rotation_rad",
        "off_plane_rotation_rad",
    ):
        value = _round_number(sample.get(key))
        if value is not None:
            result[key] = value
    return result


def _branch_payload(
    workspace: Path,
    trace_path: Path,
    trace: dict[str, Any],
    *,
    required_source_indices: Sequence[int] = (),
    family: dict[str, Any] | None = None,
) -> dict[str, Any]:
    samples = [item for item in (trace.get("samples") or []) if isinstance(item, dict)]
    indices = select_frame_indices(
        samples,
        required_source_indices=required_source_indices,
    )
    frames = [
        _compact_frame(samples[index], index, family=family)
        for index in indices
    ]
    field_availability = {
        "body_poses": bool(frames and frames[0].get("body_poses")),
        "external_load_vector": any(
            frame["external_load"]["kind"] != "none" for frame in frames
        ),
        "contact_pair_counts": any(frame["contact_pair_counts"] for frame in frames),
        "global_max_bushing_reaction": any(
            (frame.get("max_bushing_reaction_force_n") or 0.0) > 0.0
            for frame in frames
        ),
        "per_joint_bushing_reaction": any(
            frame["bushing_reaction_forces_n"] for frame in frames
        ),
        "per_joint_drift": any(frame["joint_drifts_m"] for frame in frames),
        "per_support_drift": any(frame["support_drifts_m"] for frame in frames),
        "per_body_angular_velocity": any(
            frame["body_angular_velocities_abs_rad_s"] for frame in frames
        ),
        "explicit_input_loads": any(frame["input_loads"] for frame in frames),
        "moving_chain_path_state": bool(
            frames
            and all(frame.get("moving_chain_path_state") for frame in frames)
        ),
    }
    maxima = {
        "external_load": max(
            (frame["external_load"]["magnitude"] or 0.0 for frame in frames),
            default=0.0,
        ),
        "max_bushing_reaction_force_n": max(
            (frame.get("max_bushing_reaction_force_n") or 0.0 for frame in frames),
            default=0.0,
        ),
        "joint_drift_m": max(
            (
                max(
                    [
                        *frame["joint_drifts_m"].values(),
                        *frame["support_drifts_m"].values(),
                        0.0,
                    ]
                )
                for frame in frames
            ),
            default=0.0,
        ),
        "contact_count": max((frame["contact_count"] for frame in frames), default=0),
    }
    collision = trace.get("collision_geometry") or {}
    return {
        "branch": str(trace.get("branch") or ""),
        "status": str(trace.get("status") or ""),
        "engine": str(trace.get("engine") or ""),
        "source_trace": _relative(workspace, trace_path),
        "source_trace_sha256": _sha256(trace_path),
        "source_sample_count": len(samples),
        "retained_sample_indices": indices,
        "retained_frame_count": len(frames),
        "validator_source_sha256": str(trace.get("validator_source_sha256") or ""),
        "plan_sha256": str(trace.get("plan_sha256") or ""),
        "load_case_sha256": str(trace.get("load_case_sha256") or ""),
        "package_binding_sha256": str(trace.get("package_binding_sha256") or ""),
        "collision_enabled": bool(collision.get("contact_enabled")),
        "branch_semantics": trace.get("branch_semantics") or {},
        "strict_non_claims": trace.get("strict_non_claims") or {},
        "visual_replay_semantics": trace.get("visual_replay_semantics") or {},
        "moving_chain_path_contract": copy.deepcopy(
            trace.get("moving_chain_path_contract") or {}
        ),
        "checkpoint": trace.get("checkpoint") or {},
        "input_point_abs_m": _vector(trace.get("input_point_abs_m")),
        "field_availability": field_availability,
        "maxima": maxima,
        "frames": frames,
    }


def _quaternion_matrix(wxyz: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(wxyz), dtype=float)
    values /= max(float(np.linalg.norm(values)), 1e-15)
    w, x, y, z = values
    result = np.eye(4)
    result[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return result


def _initial_transform(body: dict[str, Any]) -> np.ndarray:
    transform = _quaternion_matrix(
        body.get("initial_rotation_abs_quat_wxyz") or [1.0, 0.0, 0.0, 0.0]
    )
    transform[:3, 3] = np.asarray(
        body.get("initial_center_abs_m") or [0.0, 0.0, 0.0], dtype=float
    )
    return transform


def export_physics_evidence_scene(
    *,
    workspace_root: Path,
    run_dir: Path,
    collision: dict[str, Any],
    output: Path,
    capsule_id: str,
) -> dict[str, Any]:
    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    run.relative_to(workspace)
    scene = trimesh.Scene()
    bodies_manifest: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    colors = {
        True: np.asarray([126, 143, 161, 255], dtype=np.uint8),
        False: np.asarray([208, 157, 61, 255], dtype=np.uint8),
    }
    for body in collision.get("bodies") or []:
        body_id = str(body.get("body_id") or "body")
        fixed = bool(body.get("fixed"))
        transform = _initial_transform(body)
        components_manifest: list[dict[str, Any]] = []
        body_components = body.get("components") or []
        for component_index, component in enumerate(body_components, start=1):
            path = _workspace_path(workspace, _component_mesh_reference(component))
            loaded = trimesh.load_mesh(path, process=False)
            if not isinstance(loaded, trimesh.Trimesh):
                geometries = tuple(loaded.geometry.values())
                if not geometries:
                    raise ValueError(f"empty collision component: {path}")
                loaded = trimesh.util.concatenate(geometries)
            mesh = loaded.copy()
            solid_id = str(component.get("solid_id") or path.stem)
            base_node = f"physics__{_safe_name(body_id)}__{_safe_name(solid_id)}"
            node = (
                base_node
                if len(body_components) == 1
                else (
                    f"{base_node}__component_{component_index:04d}_"
                    f"{_safe_name(path.stem)}"
                )
            )
            mesh.metadata.update(
                {
                    "name": node,
                    "physicsBodyId": body_id,
                    "physicsSolidId": solid_id,
                    "questRole": "fixed" if fixed else "moving",
                }
            )
            mesh.visual.face_colors = np.tile(colors[fixed], (len(mesh.faces), 1))
            scene.add_geometry(
                mesh,
                geom_name=node,
                node_name=node,
                transform=transform,
            )
            recorded_sha = str(
                component.get("sha256")
                or component.get("derived_sha256")
                or component.get("source_sha256")
                or ""
            )
            actual_sha = _sha256(path)
            if recorded_sha and recorded_sha != actual_sha:
                raise ValueError(f"collision component hash mismatch: {path}")
            source_files.append({"path": _relative(workspace, path), "sha256": actual_sha})
            components_manifest.append(
                {
                    "solid_id": solid_id,
                    "node": node,
                    "component_index": component_index,
                    "component_name": path.stem,
                    "representation": str(
                        component.get("representation")
                        or component.get("collision_representation")
                        or ""
                    ),
                    "convex": bool(component.get("convex")),
                    "watertight": bool(component.get("watertight")),
                    "triangle_count": int(component.get("triangle_count") or len(mesh.faces)),
                    "source_mesh": _relative(workspace, path),
                    "source_mesh_sha256": actual_sha,
                }
            )
        bodies_manifest.append(
            {
                "body_id": body_id,
                "fixed": fixed,
                "group_ids": [str(value) for value in body.get("group_ids") or []],
                "source_solid_ids": [
                    str(value) for value in body.get("source_solid_ids") or []
                ],
                "initial_center_abs_m": body.get("initial_center_abs_m") or [],
                "initial_rotation_abs_quat_wxyz": body.get(
                    "initial_rotation_abs_quat_wxyz"
                )
                or [],
                "components": components_manifest,
            }
        )

    payload = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    manifest = {
        "schema_version": SCENE_SCHEMA,
        "artifact_kind": "interactive_physics_evidence_scene",
        "source_run": _relative(workspace, run),
        "capsule_id": capsule_id,
        "coordinate_frame": "absolute_world_metres",
        "component_vertices": "body_local_metres",
        "component_node_binding": COMPONENT_NODE_BINDING,
        "collision_source": str(collision.get("source") or ""),
        "component_count": int(collision.get("component_count") or 0),
        "whole_body_box_proxy_used": bool(collision.get("whole_body_box_proxy_used")),
        "whole_body_convex_hull_used": bool(
            collision.get("whole_body_convex_hull_used")
        ),
        "convex_hull_used": bool(collision.get("convex_hull_used")),
        "bodies": bodies_manifest,
        "source_files": source_files,
        "output": _relative(workspace, output),
        "output_sha256": _sha256(output),
        "claims": {
            "scene_uses_collision_components_from_plan": True,
            "simulation_was_rerun_for_scene": False,
            "dynamic_component_convexity_is_not_a_whole_body_convex_hull": True,
        },
    }
    _write_json(output.with_suffix(".scene.json"), manifest)
    return manifest


def _inverse_pose_point(
    point_abs: list[float], pose: dict[str, Any]
) -> list[float] | None:
    compact = _pose(pose)
    if compact is None:
        return None
    transform = _quaternion_matrix(compact["rotation_abs_quat_wxyz"])
    origin = np.asarray(compact["position_abs_m"], dtype=float)
    local = transform[:3, :3].T @ (np.asarray(point_abs, dtype=float) - origin)
    return [round(float(value), 9) for value in local]


def _landmarks(family: dict[str, Any]) -> list[dict[str, Any]]:
    body_initial = {
        str(body.get("body_id")): {
            "position_abs_m": body.get("center_abs_m") or [0.0, 0.0, 0.0],
            "rotation_abs_quat_wxyz": body.get("rotation_abs_quat_wxyz")
            or [1.0, 0.0, 0.0, 0.0],
        }
        for body in family.get("bodies") or []
    }
    result: list[dict[str, Any]] = []
    for joint in family.get("revolute_joints") or []:
        parent = str(joint.get("parent_body_id") or "")
        origin = _vector(joint.get("origin_abs_m"))
        local = _inverse_pose_point(origin, body_initial.get(parent) or {}) if origin else None
        result.append(
            {
                "id": str(joint.get("joint_id") or "joint"),
                "kind": "revolute_joint",
                "parent_body_id": parent,
                "child_body_id": str(joint.get("child_body_id") or ""),
                "origin_abs_m": origin,
                "origin_parent_local_m": local,
                "axis_abs": _vector(joint.get("axis_abs")),
            }
        )
    fixed_body_id = str(
        family.get("fixed_body_id")
        or next(
            (
                body.get("body_id")
                for body in family.get("bodies") or []
                if body.get("fixed")
            ),
            "ground",
        )
    )
    for shaft in family.get("shafts") or []:
        shaft_id = str(shaft.get("shaft_id") or "")
        child_body_id = str(shaft.get("body_id") or "")
        origin = _vector(shaft.get("origin_abs_m"))
        if not shaft_id or not child_body_id or origin is None:
            continue
        result.append(
            {
                "id": shaft_id,
                "kind": "shaft_support",
                "parent_body_id": fixed_body_id,
                "child_body_id": child_body_id,
                "origin_abs_m": origin,
                "origin_parent_local_m": _inverse_pose_point(
                    origin, body_initial.get(fixed_body_id) or {}
                ),
                "axis_abs": _vector(shaft.get("axis_abs")),
            }
        )
    if str(family.get("mechanism_kind") or "") == "rack_pinion":
        pinion = str(family.get("pinion_body_id") or "pinion")
        rack = str(family.get("rack_body_id") or "rack")
        pinion_origin = _vector(family.get("pinion_origin_abs_m"))
        if pinion_origin:
            result.append(
                {
                    "id": "pinion_bearing",
                    "kind": "support",
                    "parent_body_id": str(family.get("fixed_body_id") or "ground"),
                    "child_body_id": pinion,
                    "origin_abs_m": pinion_origin,
                    "origin_parent_local_m": _inverse_pose_point(
                        pinion_origin,
                        body_initial.get(str(family.get("fixed_body_id") or "ground"))
                        or {},
                    ),
                    "axis_abs": _vector(family.get("pinion_axis_abs")),
                }
            )
        rack_pose = body_initial.get(rack) or {}
        rack_origin = _vector(rack_pose.get("position_abs_m"))
        if rack_origin:
            result.append(
                {
                    "id": "rack_guide",
                    "kind": "support",
                    "parent_body_id": str(family.get("fixed_body_id") or "ground"),
                    "child_body_id": rack,
                    "origin_abs_m": rack_origin,
                    "origin_parent_local_m": _inverse_pose_point(
                        rack_origin,
                        body_initial.get(str(family.get("fixed_body_id") or "ground"))
                        or {},
                    ),
                    "axis_abs": _vector(family.get("rack_axis_abs")),
                }
            )
    return result


def _clock_transmission_visuals(
    family: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Describe only declared clock transmission wheels and their graph.

    This metadata is presentation-only. The viewer still derives the current
    direction from the saved per-body angular velocity and limits arrows to
    the connected component of the currently loaded input body.
    """

    if str(family.get("mechanism_kind") or "") != "spur_gear_clock":
        return [], []
    body_initial = {
        str(body.get("body_id")): {
            "position_abs_m": body.get("center_abs_m") or [0.0, 0.0, 0.0],
            "rotation_abs_quat_wxyz": body.get("rotation_abs_quat_wxyz")
            or [1.0, 0.0, 0.0, 0.0],
        }
        for body in family.get("bodies") or []
    }
    gear_body_ids: set[str] = set()
    edges: list[dict[str, Any]] = []

    def add_edges(
        items: Any,
        *,
        kind: str,
        id_key: str,
        engaged_field: str | None = None,
    ) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            first = str(
                item.get("driver_body_id")
                or item.get("input_body_id")
                or item.get("cam_body_id")
                or ""
            )
            second = str(
                item.get("driven_body_id")
                or item.get("follower_body_id")
                or ""
            )
            if not first or not second:
                continue
            edge: dict[str, Any] = {
                "id": str(item.get(id_key) or f"{kind}:{first}:{second}"),
                "kind": kind,
                "first_body_id": first,
                "second_body_id": second,
            }
            if engaged_field:
                edge["engaged_field"] = engaged_field
            edges.append(edge)

    for item in family.get("gear_meshes") or []:
        if not isinstance(item, dict):
            continue
        for key in ("driver_body_id", "driven_body_id"):
            body_id = str(item.get(key) or "")
            if body_id:
                gear_body_ids.add(body_id)
    for item in family.get("shaft_transfers") or []:
        if not isinstance(item, dict):
            continue
        for key in ("driver_body_id", "driven_body_id"):
            body_id = str(item.get(key) or "")
            if body_id:
                gear_body_ids.add(body_id)

    add_edges(family.get("gear_meshes"), kind="gear_mesh", id_key="mesh_id")
    add_edges(
        family.get("shaft_transfers"),
        kind="shaft_transfer",
        id_key="transfer_id",
    )
    for key in ("setting_clutch", "alarm_setting_clutch"):
        item = family.get(key)
        if isinstance(item, dict):
            add_edges(
                [item],
                kind="clutch",
                id_key="clutch_id",
                engaged_field=f"{key}_engaged",
            )

    annotations: list[dict[str, Any]] = []
    for shaft in family.get("shafts") or []:
        if not isinstance(shaft, dict):
            continue
        body_id = str(shaft.get("body_id") or "")
        origin = _vector(shaft.get("origin_abs_m"))
        axis = _vector(shaft.get("axis_abs"))
        if body_id not in gear_body_ids or origin is None or axis is None:
            continue
        annotations.append(
            {
                "id": str(shaft.get("shaft_id") or f"shaft_{body_id}"),
                "kind": "gear_rotation",
                "body_id": body_id,
                "origin_abs_m": origin,
                "origin_body_local_m": _inverse_pose_point(
                    origin, body_initial.get(body_id) or {}
                ),
                "axis_abs": axis,
            }
        )
    return annotations, edges


def upgrade_replay_for_presentation(
    replay: dict[str, Any],
    *,
    workspace_root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Add clock arrow fields from the already-saved authoritative trace.

    Older terminal runs contain replay-v2 frame indexes but intentionally omit
    per-body angular velocity.  The dashboard may enrich that in memory from
    the hash-bound source trace.  It never writes into the run and never
    invokes a simulator.
    """

    family = ((plan.get("adapter_spec") or {}).get("family_spec") or {})
    rotation_annotations, transmission_edges = _clock_transmission_visuals(family)
    if not rotation_annotations or not isinstance(replay, dict):
        return replay
    branches = replay.get("branches")
    physical = branches.get("physical") if isinstance(branches, dict) else None
    if not isinstance(physical, dict):
        return replay
    source_reference = str(physical.get("source_trace") or "")
    if not source_reference:
        return replay
    try:
        source_path = _workspace_path(workspace_root.resolve(), source_reference)
    except (FileNotFoundError, ValueError):
        return replay
    expected_sha256 = str(physical.get("source_trace_sha256") or "")
    actual_sha256 = _sha256(source_path)
    if expected_sha256 and expected_sha256 != actual_sha256:
        return replay
    source = _read_json(source_path)
    samples = source.get("samples") or []
    if not isinstance(samples, list):
        return replay

    upgraded = copy.deepcopy(replay)
    upgraded["source_schema_version"] = str(
        replay.get("source_schema_version")
        or replay.get("schema_version")
        or ""
    )
    upgraded["schema_version"] = REPLAY_SCHEMA
    upgraded["rotation_annotations"] = rotation_annotations
    upgraded["transmission_edges"] = transmission_edges
    upgraded["presentation_overlay"] = {
        "schema_version": "clock_transmission_arrow_overlay_v1",
        "derived_from_saved_trace": True,
        "source_trace": source_reference,
        "source_trace_sha256": actual_sha256,
        "simulation_was_rerun": False,
        "run_artifacts_were_modified": False,
    }
    upgraded_branches = upgraded.get("branches") or {}
    upgraded_physical = upgraded_branches.get("physical") or {}
    for frame in upgraded_physical.get("frames") or []:
        try:
            source_index = int(frame.get("source_sample_index"))
        except (TypeError, ValueError):
            continue
        if not 0 <= source_index < len(samples):
            continue
        sample = samples[source_index]
        if not isinstance(sample, dict):
            continue
        compact = _compact_frame(sample, source_index, family=family)
        for key in (
            "external_load",
            "active_input_body_id",
            "active_input_point_abs_m",
            "input_loads",
            "body_angular_velocities_abs_rad_s",
            "setting_clutch_engaged",
            "alarm_setting_clutch_engaged",
        ):
            if key in compact:
                frame[key] = compact[key]
    field_availability = upgraded_physical.setdefault("field_availability", {})
    field_availability["per_body_angular_velocity"] = any(
        frame.get("body_angular_velocities_abs_rad_s")
        for frame in upgraded_physical.get("frames") or []
    )
    field_availability["explicit_input_loads"] = any(
        frame.get("input_loads") for frame in upgraded_physical.get("frames") or []
    )
    claims = upgraded.setdefault("claims", {})
    claims["transmission_arrows_derived_from_saved_trace"] = True
    claims["simulation_was_rerun_for_replay"] = False
    claims["original_trace_remains_authoritative"] = True
    return upgraded


def generate_physics_evidence(
    *,
    workspace_root: Path,
    run_dir: Path,
    plan: dict[str, Any],
    force: bool = False,
    namespace_assets: bool = False,
    visual_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate one shared scene and one replay payload per saved load pair."""

    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    run.relative_to(workspace)
    capsule_id = str(plan.get("capsule_id") or "")
    family = ((plan.get("adapter_spec") or {}).get("family_spec") or {})
    rotation_annotations, transmission_edges = _clock_transmission_visuals(family)
    collision = family.get("collision_geometry") or {}
    if not capsule_id:
        return {"status": "NOT_AVAILABLE", "reason": "capsule id missing"}

    trace_root = run / "traces" / capsule_id
    if not collision.get("bodies") and trace_root.is_dir():
        from ..mechanism.physical_legality import normalized_physical_audit_inputs

        for physical_path in sorted(trace_root.glob("*/physical.trace.json")):
            trace_collision, _trace_samples = normalized_physical_audit_inputs(
                _read_json(physical_path)
            )
            if trace_collision.get("bodies"):
                collision = trace_collision
                break
    if not collision.get("bodies"):
        return {"status": "NOT_AVAILABLE", "reason": "family collision geometry missing"}

    if visual_dir is None:
        visual = run / "visual"
    else:
        visual = Path(visual_dir).resolve()
        visual.relative_to(workspace)
    asset_stem = (
        f"stage5_physics_evidence__{capsule_id}"
        if namespace_assets
        else "stage5_physics_evidence"
    )
    scene_path = visual / f"{asset_stem}.glb"
    scene_manifest_path = scene_path.with_suffix(".scene.json")
    scene_manifest = (
        _read_json(scene_manifest_path) if scene_manifest_path.is_file() else {}
    )
    if force or not _scene_manifest_is_current(
        workspace, scene_path, scene_manifest, collision
    ):
        scene_manifest = export_physics_evidence_scene(
            workspace_root=workspace,
            run_dir=run,
            collision=collision,
            output=scene_path,
            capsule_id=capsule_id,
        )

    representation_counts: dict[str, int] = {}
    for body in collision.get("bodies") or []:
        for component in body.get("components") or []:
            key = str(
                component.get("representation")
                or component.get("collision_representation")
                or "unknown"
            )
            representation_counts[key] = representation_counts.get(key, 0) + 1

    penetration_indices: dict[str, set[int]] = {}
    penetration_evidence_sources: list[dict[str, str]] = []
    for summary_path in (
        run / "results" / "penetration_evidence_v1.json",
        run / "migration" / "penetration_evidence_v1.json",
    ):
        if not summary_path.is_file():
            continue
        penetration_evidence_sources.append(
            {
                "path": _relative(workspace, summary_path),
                "sha256": _sha256(summary_path),
            }
        )
        for event in _read_json(summary_path).get("events") or []:
            if not isinstance(event, dict):
                continue
            load_id = str(event.get("load_case_id") or "")
            source_index = event.get("source_sample_index")
            if load_id and source_index is not None:
                penetration_indices.setdefault(load_id, set()).add(int(source_index))

    replay_outputs: dict[str, dict[str, Any]] = {}
    # A capsule with collision geometry but no saved traces yet (e.g. its
    # physics lane has not run in this revision) must degrade to an empty
    # replay set, not abort visualization for the whole run.
    load_dirs = (
        sorted(path for path in trace_root.iterdir() if path.is_dir())
        if trace_root.is_dir()
        else []
    )
    for load_dir in load_dirs:
        physical_path = load_dir / "physical.trace.json"
        ideal_dynamics_path = load_dir / "ideal_dynamics.trace.json"
        legacy_reduced_path = load_dir / "reduced.trace.json"
        ideal_reference_path = load_dir / "ideal_reference.trace.json"
        comparison_path = (
            ideal_dynamics_path
            if ideal_dynamics_path.is_file()
            else ideal_reference_path
            if ideal_reference_path.is_file()
            else legacy_reduced_path
            if legacy_reduced_path.is_file()
            else None
        )
        if not physical_path.is_file():
            continue
        physical = _read_json(physical_path)
        reduced = _read_json(comparison_path) if comparison_path is not None else None
        ideal_replay_error = None
        if not physical.get("samples"):
            continue
        if not ((physical.get("samples") or [{}])[0].get("body_poses")):
            from ..mechanism.physical_legality import normalized_physical_audit_inputs

            _physical_geometry, physical_samples = normalized_physical_audit_inputs(
                physical
            )
            physical = {**physical, "samples": physical_samples}
        if reduced is not None and reduced.get("samples"):
            if reduced.get("branch") == "ideal_reference":
                first_ideal_sample = (reduced.get("samples") or [{}])[0]
                if not isinstance(
                    first_ideal_sample.get("body_poses"),
                    dict,
                ):
                    try:
                        from .ideal_replay import synthesize_ideal_kinematic_trace

                        reduced = synthesize_ideal_kinematic_trace(
                            plan=plan,
                            physical_trace=physical,
                            ideal_trace=reduced,
                        )
                    except Exception as exc:
                        ideal_replay_error = f"{type(exc).__name__}: {exc}"
                        reduced = None
            elif not ((reduced.get("samples") or [{}])[0].get("body_poses")):
                from ..mechanism.physical_legality import normalized_physical_audit_inputs

                _reduced_geometry, reduced_samples = normalized_physical_audit_inputs(
                    reduced
                )
                reduced = {**reduced, "samples": reduced_samples}
        else:
            reduced = None
        load_id = load_dir.name
        input_body_id = str(
            family.get("input_body_id")
            or family.get("pinion_body_id")
            or next(
                (
                    body.get("body_id")
                    for body in (family.get("bodies") or collision.get("bodies") or [])
                    if not body.get("fixed")
                ),
                "",
            )
        )
        first_pose = ((physical.get("samples") or [{}])[0].get("body_poses") or {}).get(
            input_body_id
        ) or {}
        input_point = _vector(physical.get("input_point_abs_m"))
        physical_branch = _branch_payload(
            workspace,
            physical_path,
            physical,
            required_source_indices=sorted(penetration_indices.get(load_id, set())),
            family=family,
        )
        reduced_branch = (
            _branch_payload(
                workspace,
                comparison_path,
                reduced,
                family=family,
            )
            if comparison_path is not None and reduced is not None
            else None
        )
        replay = {
            "schema_version": REPLAY_SCHEMA,
            "artifact_kind": "interactive_physics_evidence_replay",
            "source_run": _relative(workspace, run),
            "capsule_id": capsule_id,
            "load_case_id": load_id,
            "held_out": load_id.startswith("held_out"),
            "scene": {
                "path": _relative(workspace, scene_path),
                "sha256": scene_manifest.get("output_sha256"),
                "coordinate_frame": "absolute_world_metres",
            },
            "input": {
                "body_id": input_body_id,
                "point_abs_m_at_start": input_point,
                "point_body_local_m": (
                    _inverse_pose_point(input_point, first_pose) if input_point else None
                ),
            },
            "rotation_annotations": rotation_annotations,
            "transmission_edges": transmission_edges,
            "landmarks": _landmarks(family),
            "collision": {
                "source": str(collision.get("source") or ""),
                "mode": str((physical.get("collision_geometry") or {}).get("mode") or ""),
                "component_count": int(collision.get("component_count") or 0),
                "representation_counts": representation_counts,
                "whole_body_box_proxy_used": bool(
                    collision.get("whole_body_box_proxy_used")
                ),
                "whole_body_convex_hull_used": bool(
                    collision.get("whole_body_convex_hull_used")
                ),
                "convex_hull_used": bool(collision.get("convex_hull_used")),
                "dynamic_rule": (
                    "逐 STEP 实体原形碰撞；运动子实体须为准确基本体或本身经验证为凸，"
                    "不自动生成整件凸包"
                ),
            },
            "provenance": {
                "package_binding_sha256": str(
                    physical.get("package_binding_sha256") or ""
                ),
                "assembly_sha256": str(collision.get("assembly_sha256") or ""),
                "assembly_contract_sha256": str(
                    collision.get("assembly_contract_sha256") or ""
                ),
                "load_case_sha256": str(physical.get("load_case_sha256") or ""),
                "plan_sha256": str(physical.get("plan_sha256") or ""),
                "validator_source_sha256": str(
                    physical.get("validator_source_sha256") or ""
                ),
                "penetration_evidence_sources": penetration_evidence_sources,
            },
            "branches": {
                "physical": physical_branch,
                **({"reduced": reduced_branch} if reduced_branch is not None else {}),
            },
            "claims": {
                "simulation_was_rerun_for_replay": False,
                "frames_are_derived_from_saved_traces": True,
                "original_trace_remains_authoritative": True,
                "physical_trace_complete": (
                    physical_branch.get("status") == "COMPLETED"
                ),
                "partial_physical_trace_has_no_certificate_authority": (
                    physical_branch.get("status") != "COMPLETED"
                ),
                "visualization_is_not_an_external_cryptographic_attestation": True,
                "legacy_trace_has_spatial_joint_reaction_distribution": bool(
                    physical_branch["field_availability"][
                        "per_joint_bushing_reaction"
                    ]
                ),
                "ideal_kinematic_animation_is_equation_derived": bool(
                    reduced_branch is not None
                    and reduced_branch.get("branch") == "ideal_reference"
                ),
                "ideal_animation_has_contact_authority": False,
                "ideal_animation_has_dynamic_response_authority": False,
                "ideal_replay_error": ideal_replay_error,
            },
        }
        output = visual / (
            f"stage5_physics_evidence__{capsule_id}__{load_id}.json"
            if namespace_assets
            else f"stage5_physics_evidence__{load_id}.json"
        )
        existing = _read_json(output) if output.is_file() else {}
        existing_branches = existing.get("branches") or {}
        replay_is_current = (
            existing.get("schema_version") == REPLAY_SCHEMA
            and (existing.get("scene") or {}).get("sha256")
            == replay["scene"]["sha256"]
            and (existing_branches.get("physical") or {}).get(
                "source_trace_sha256"
            )
            == physical_branch["source_trace_sha256"]
            and (existing.get("provenance") or {}).get(
                "penetration_evidence_sources"
            )
            == penetration_evidence_sources
            and (
                (
                    reduced_branch is None
                    and "reduced" not in existing_branches
                )
                or (
                    reduced_branch is not None
                    and (existing_branches.get("reduced") or {}).get(
                        "source_trace_sha256"
                    )
                    == reduced_branch["source_trace_sha256"]
                )
            )
        )
        if force or not replay_is_current:
            _write_json(output, replay)
        else:
            replay = existing
        replay_outputs[load_id] = {
            "path": _relative(workspace, output),
            "sha256": _sha256(output),
            "physical_trace_sha256": replay["branches"]["physical"][
                "source_trace_sha256"
            ],
            "ideal_trace_sha256": (
                (replay["branches"].get("reduced") or {}).get(
                    "source_trace_sha256"
                )
            ),
            "ideal_trace_branch": (
                (replay["branches"].get("reduced") or {}).get("branch")
            ),
            "ideal_dynamics_trace_sha256": (
                (replay["branches"].get("reduced") or {}).get(
                    "source_trace_sha256"
                )
                if (replay["branches"].get("reduced") or {}).get("branch")
                in {"ideal_dynamics", "reduced"}
                else None
            ),
            "physical_only_replay": "reduced" not in replay["branches"],
            "physical_trace_status": replay["branches"]["physical"].get(
                "status"
            ),
            "physical_trace_complete": (
                replay["branches"]["physical"].get("status") == "COMPLETED"
            ),
        }

    return {
        "status": "GENERATED" if replay_outputs else "NOT_AVAILABLE",
        "scene": {
            "path": _relative(workspace, scene_path),
            "sha256": scene_manifest.get("output_sha256"),
            "manifest": _relative(workspace, scene_manifest_path),
        },
        "replays": replay_outputs,
        "claims": {
            "simulation_was_rerun_for_visuals": False,
            "source_traces_are_hash_bound": True,
        },
    }
