"""Render two saved mechanism traces as one synchronized replay.

Drawn labels are Chinese by default and English when ``SIMREADY_RENDER_LANG=en``
(see ``render_lang``).

This module is deliberately a renderer, not a simulator or a verdict engine.  It
only reads already-produced physical/comparison trace JSON files and the
physical trace's bound collision meshes.  The comparison side may be either a
reduced dynamics trace or a deterministic ideal-reference trace.  The
accompanying sidecar records exactly which traces were used and whether video
encoding completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .render_lang import label, sans_serif_fonts


def _configure_ffmpeg_writer() -> bool:
    """Find ffmpeg beside either Python or Matplotlib's owning environment."""

    if writers.is_available("ffmpeg"):
        return True
    candidates = [Path(sys.prefix) / "bin" / "ffmpeg"]
    candidates.extend(
        parent / "bin" / "ffmpeg"
        for parent in Path(matplotlib.__file__).resolve().parents
    )
    for candidate in candidates:
        if candidate.is_file():
            matplotlib.rcParams["animation.ffmpeg_path"] = str(candidate)
            if writers.is_available("ffmpeg"):
                return True
    return False


PHYSICAL_COLOR = "#2f6fed"
PHYSICAL_EDGE = "#194ca8"
REDUCED_COLOR = "#21a56b"
REDUCED_EDGE = "#126843"
FIXED_COLOR = "#8a98a8"
FIXED_EDGE = "#536273"
INK = "#17212b"
MUTED = "#5d6b7a"
GRID = "#dfe6ee"
PAPER = "#f5f8fb"

# Clean user-facing labels for both linear and angular mechanisms.  The *_ZH
# tables are the Chinese text (NOTICE_ZH is also the sidecar's ``notice_zh``);
# drawn text picks *_ZH or *_EN at call time through ``render_lang.label``.
NOTICE_ZH = "保存轨迹回放 · 未重跑仿真"
NOTICE_EN = "Saved-trajectory replay · simulation not rerun"
PHASE_LABELS_ZH = {
    "initial": "起始",
    "forward_force": "正向施力",
    "forward_torque": "正向扭矩",
    "forward_settle": "正向稳定",
    "reverse_force": "反向施力",
    "reverse_torque": "反向扭矩",
    "reverse_settle": "反向稳定",
    "transverse_force": "横向施力",
    "transverse_torque": "横向扭矩",
    "transverse_settle": "横向稳定",
    "transverse_reverse_force": "反向横向施力",
    "transverse_reverse_torque": "反向横向扭矩",
    "transverse_reverse_settle": "反向横向稳定",
    "task_outbound_force": "打开",
    "task_outbound_torque": "打开",
    "task_outbound_settle": "打开后观察",
    "task_return_force": "关闭",
    "task_return_torque": "关闭",
    "task_return_settle": "关闭后观察",
    "cylinder_extend": "液压缸伸出",
    "raised_settle": "展开后稳定",
    "cylinder_retract": "液压缸缩回",
    "home_settle": "折叠后稳定",
    "gravity_settle": "重力静置",
    "zero_input": "零输入",
    "hold": "保持",
    "settle": "稳定",
    "outbound": "打开行程",
    "return": "返回行程",
}
LOAD_LABELS_ZH = {
    "nominal_light": "轻载往返",
    "nominal_loaded": "重载往返",
    "zero_input_gravity": "零输入稳定性",
    "transverse_probe": "横向扰动",
    "transverse_torque_probe": "横向扭矩扰动",
    "held_out_mid": "保留工况",
    "task_envelope_roundtrip": "全行程打开与关闭",
    "task_angle_roundtrip": "全角度打开与关闭",
}


LOAD_LABELS_ZH.update(
    {
        "task_clock_cycle": "三针正常走时",
        "task_manual_time_setting": "手动调时",
        "task_alarm_cycle": "机械打铃",
        "tong_tabletop_pickup": "桌面夹取并抬起",
        "tong_tabletop_hold": "离桌保持",
        "tong_tabletop_release": "松手回桌",
    }
)

PHASE_LABELS_ZH.update(
    {
        "tong_pickup_initial_table_settle": "物体在桌面静置",
        "tong_pickup_approach": "整把夹子靠近",
        "tong_pickup_close": "夹紧物体",
        "tong_pickup_grip_verify": "确认双面夹紧",
        "tong_pickup_lift": "整把夹子抬起",
        "tong_pickup_verify": "确认离桌夹取",
        "tong_hold_initial_table_settle": "物体在桌面静置",
        "tong_hold_approach": "整把夹子靠近",
        "tong_hold_close": "夹紧物体",
        "tong_hold_grip_verify": "确认双面夹紧",
        "tong_hold_lift": "整把夹子抬起",
        "tong_hold_verify": "离桌保持",
        "tong_release_initial_table_settle": "物体在桌面静置",
        "tong_release_approach": "整把夹子靠近",
        "tong_release_close": "夹紧物体",
        "tong_release_grip_verify": "确认双面夹紧",
        "tong_release_lift": "整把夹子抬起",
        "tong_release_preload": "抬起后短时保持",
        "tong_release_passive_return": "松手弹簧张开",
        "tong_release_final_table_settle": "物体落回桌面",
        "tong_release_observe": "确认释放完成",
    }
)

PHASE_LABELS_EN = {
    "initial": "Start",
    "forward_force": "Forward force",
    "forward_torque": "Forward torque",
    "forward_settle": "Forward settle",
    "reverse_force": "Reverse force",
    "reverse_torque": "Reverse torque",
    "reverse_settle": "Reverse settle",
    "transverse_force": "Transverse force",
    "transverse_torque": "Transverse torque",
    "transverse_settle": "Transverse settle",
    "transverse_reverse_force": "Reverse transverse force",
    "transverse_reverse_torque": "Reverse transverse torque",
    "transverse_reverse_settle": "Reverse transverse settle",
    "task_outbound_force": "Open",
    "task_outbound_torque": "Open",
    "task_outbound_settle": "Observe after opening",
    "task_return_force": "Close",
    "task_return_torque": "Close",
    "task_return_settle": "Observe after closing",
    "cylinder_extend": "Cylinder extends",
    "raised_settle": "Settle when raised",
    "cylinder_retract": "Cylinder retracts",
    "home_settle": "Settle when folded",
    "gravity_settle": "Gravity settle",
    "zero_input": "Zero input",
    "hold": "Hold",
    "settle": "Settle",
    "outbound": "Opening travel",
    "return": "Return travel",
    "tong_pickup_initial_table_settle": "Object rests on table",
    "tong_pickup_approach": "Tongs approach",
    "tong_pickup_close": "Grip object",
    "tong_pickup_grip_verify": "Confirm two-sided grip",
    "tong_pickup_lift": "Tongs lift",
    "tong_pickup_verify": "Confirm lift-off grip",
    "tong_hold_initial_table_settle": "Object rests on table",
    "tong_hold_approach": "Tongs approach",
    "tong_hold_close": "Grip object",
    "tong_hold_grip_verify": "Confirm two-sided grip",
    "tong_hold_lift": "Tongs lift",
    "tong_hold_verify": "Hold off the table",
    "tong_release_initial_table_settle": "Object rests on table",
    "tong_release_approach": "Tongs approach",
    "tong_release_close": "Grip object",
    "tong_release_grip_verify": "Confirm two-sided grip",
    "tong_release_lift": "Tongs lift",
    "tong_release_preload": "Brief hold after lift",
    "tong_release_passive_return": "Release, spring opens",
    "tong_release_final_table_settle": "Object back on table",
    "tong_release_observe": "Confirm release",
}
LOAD_LABELS_EN = {
    "nominal_light": "Light-load round trip",
    "nominal_loaded": "Heavy-load round trip",
    "zero_input_gravity": "Zero-input stability",
    "transverse_probe": "Transverse disturbance",
    "transverse_torque_probe": "Transverse torque disturbance",
    "held_out_mid": "Held-out load case",
    "task_envelope_roundtrip": "Full-travel open and close",
    "task_angle_roundtrip": "Full-angle open and close",
    "task_clock_cycle": "Normal three-hand timekeeping",
    "task_manual_time_setting": "Manual time setting",
    "task_alarm_cycle": "Mechanical alarm ring",
    "tong_tabletop_pickup": "Grip and lift from table",
    "tong_tabletop_hold": "Hold off the table",
    "tong_tabletop_release": "Release back to table",
}


def _configure_chinese_font() -> None:
    matplotlib.rcParams["font.sans-serif"] = sans_serif_fonts()
    matplotlib.rcParams["axes.unicode_minus"] = False


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"trace root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_input(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _resolve_workspace_asset(workspace: Path, value: str | Path) -> Path:
    root = workspace.resolve()
    raw = Path(value)
    candidate = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"trace asset escapes workspace root: {value}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _load_mesh(workspace: Path, item: dict[str, Any]) -> trimesh.Trimesh:
    mesh_path = _resolve_workspace_asset(workspace, item["derived"])
    loaded = trimesh.load_mesh(mesh_path, process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        geometries = tuple(loaded.geometry.values())
        if not geometries:
            raise ValueError(f"trace mesh scene is empty: {mesh_path}")
        loaded = trimesh.util.concatenate(geometries)
    if not len(loaded.vertices) or not len(loaded.faces):
        raise ValueError(f"trace mesh is empty: {mesh_path}")
    return loaded


def _normalized_quaternion(values: Iterable[float]) -> np.ndarray:
    quaternion = np.asarray(list(values), dtype=float)
    if quaternion.shape != (4,):
        raise ValueError("rotation quaternion must contain four values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-15:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quaternion / norm


def _quaternion_matrix(values: Iterable[float]) -> np.ndarray:
    w, x, y, z = _normalized_quaternion(values)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _slerp(left: Iterable[float], right: Iterable[float], fraction: float) -> np.ndarray:
    q0 = _normalized_quaternion(left)
    q1 = _normalized_quaternion(right)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        result = q0 + fraction * (q1 - q0)
        return result / max(float(np.linalg.norm(result)), 1e-15)
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return (
        math.sin((1.0 - fraction) * angle) / denominator * q0
        + math.sin(fraction * angle) / denominator * q1
    )


def _transform_vertices_mm(
    vertices_local_m: np.ndarray,
    position_abs_m: Iterable[float],
    rotation_abs_quat_wxyz: Iterable[float],
) -> np.ndarray:
    rotation = _quaternion_matrix(rotation_abs_quat_wxyz)
    position = np.asarray(list(position_abs_m), dtype=float)
    return ((rotation @ vertices_local_m.T).T + position) * 1000.0


def _validated_samples(trace: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    raw = trace.get("samples")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"trace contains no samples: {path}")
    samples: list[dict[str, Any]] = []
    previous_time = -math.inf
    for index, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise ValueError(f"trace sample {index} is not an object: {path}")
        time_s = float(sample["time_s"])
        if not math.isfinite(time_s) or time_s < previous_time:
            raise ValueError(f"trace sample times are not monotonic: {path}")
        pose = sample.get("moving_pose")
        body_poses = sample.get("body_poses")
        if not isinstance(pose, dict) and not isinstance(body_poses, dict):
            raise ValueError(
                f"trace sample {index} has neither moving_pose nor body_poses: {path}"
            )
        samples.append(sample)
        previous_time = time_s
    return samples


def _validate_trace_pair(
    physical: dict[str, Any],
    reduced: dict[str, Any],
) -> None:
    if physical.get("branch") != "physical":
        raise ValueError("left trace must declare branch=physical")
    if reduced.get("branch") not in {"reduced", "ideal_reference"}:
        raise ValueError(
            "right trace must declare branch=reduced or branch=ideal_reference"
        )
    for key in ("capsule_id", "load_case_id", "package_binding_sha256"):
        if physical.get(key) != reduced.get(key):
            raise ValueError(f"trace pair differs in {key}")


def _sample_bracket(samples: list[dict[str, Any]], time_s: float) -> tuple[int, int, float]:
    times = np.asarray([float(sample["time_s"]) for sample in samples], dtype=float)
    if time_s <= times[0] or len(times) == 1:
        return 0, 0, 0.0
    if time_s >= times[-1]:
        last = len(times) - 1
        return last, last, 0.0
    right = int(np.searchsorted(times, time_s, side="right"))
    left = right - 1
    span = float(times[right] - times[left])
    fraction = 0.0 if span <= 1e-15 else (time_s - float(times[left])) / span
    return left, right, float(fraction)


def _interpolated_pose(
    samples: list[dict[str, Any]], time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    left_index, right_index, fraction = _sample_bracket(samples, time_s)
    left = samples[left_index]["moving_pose"]
    right = samples[right_index]["moving_pose"]
    left_position = np.asarray(left["position_abs_m"], dtype=float)
    right_position = np.asarray(right["position_abs_m"], dtype=float)
    position = left_position + fraction * (right_position - left_position)
    quaternion = _slerp(
        left["rotation_abs_quat_wxyz"],
        right["rotation_abs_quat_wxyz"],
        fraction,
    )
    return position, quaternion


def _interpolated_body_pose(
    samples: list[dict[str, Any]], time_s: float, body_id: str | None
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one compound body pose, or the legacy moving pose."""

    if body_id is None:
        return _interpolated_pose(samples, time_s)
    left_index, right_index, fraction = _sample_bracket(samples, time_s)
    left_poses = samples[left_index].get("body_poses") or {}
    right_poses = samples[right_index].get("body_poses") or {}
    left = left_poses.get(body_id)
    right = right_poses.get(body_id)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError(f"trace samples do not expose body pose {body_id!r}")
    left_position = np.asarray(left["position_abs_m"], dtype=float)
    right_position = np.asarray(right["position_abs_m"], dtype=float)
    position = left_position + fraction * (right_position - left_position)
    quaternion = _slerp(
        left["rotation_abs_quat_wxyz"],
        right["rotation_abs_quat_wxyz"],
        fraction,
    )
    return position, quaternion


def _video_mesh_proxy(mesh: trimesh.Trimesh, *, fixed: bool) -> trimesh.Trimesh:
    """Reduce only the plotting mesh; saved collision evidence stays untouched."""

    target_faces = 2200 if fixed else 900
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        return mesh
    if not isinstance(simplified, trimesh.Trimesh) or not len(simplified.faces):
        return mesh
    return simplified


ORIGINAL_STEP_MESH_SOURCE = "original_step_solids"


def _original_step_body_meshes(
    workspace: Path,
    collision: dict[str, Any],
) -> dict[str, trimesh.Trimesh]:
    """Body-local meshes tessellated from the authoritative STEP, by body id.

    The saved collision components are what the solver collided with; they are
    not what the machine looks like, and for template-owned bodies they may be
    a proven convex hull or nothing at all.  Every rendering of a trace must
    therefore show the original STEP solids.  Each compound body is rebuilt
    from the ``source_solid_ids`` the trace records for it, expressed in the
    body frame the trace poses use (``initial_center_abs_m`` and
    ``initial_rotation_abs_quat_wxyz``).  Returns an empty mapping when the
    package STEP is unavailable or no longer matches the trace's hash, so the
    caller can fall back and label the fallback.
    """

    contract_relative = collision.get("assembly_contract")
    assembly_file = collision.get("assembly_file")
    if not isinstance(contract_relative, str) or not isinstance(assembly_file, str):
        return {}
    try:
        package_dir = _resolve_workspace_asset(workspace, contract_relative).parent
    except (FileNotFoundError, ValueError):
        return {}
    step_path = package_dir / assembly_file
    if not step_path.is_file():
        return {}
    expected = collision.get("assembly_sha256")
    if isinstance(expected, str) and expected and _sha256(step_path) != expected:
        return {}
    try:
        from ..mechanism.semantic_preview import cached_step_meshes

        solids = cached_step_meshes(step_path)
    except Exception:
        return {}
    result: dict[str, trimesh.Trimesh] = {}
    for item in collision.get("bodies") or ():
        if not isinstance(item, dict):
            continue
        indices: list[int] = []
        for solid_id in item.get("source_solid_ids") or ():
            digits = str(solid_id).removeprefix("solid_")
            if digits.isdigit() and 1 <= int(digits) <= len(solids):
                indices.append(int(digits))
        if not indices:
            continue
        pieces = [solids[index - 1].copy() for index in indices]
        merged = trimesh.util.concatenate(pieces) if len(pieces) > 1 else pieces[0]
        center = np.asarray(
            item.get("initial_center_abs_m") or (0.0, 0.0, 0.0), dtype=float
        )
        rotation = _quaternion_matrix(
            item.get("initial_rotation_abs_quat_wxyz") or (1.0, 0.0, 0.0, 0.0)
        )
        # Body-local frame: R^T (v - c), as family_collision._world_to_body_local.
        merged.vertices = (np.asarray(merged.vertices, dtype=float) - center) @ rotation
        result[str(item["body_id"])] = merged
    return result


def _mesh_source_summary(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    """What geometry each rendered body came from; STEP is the only clean answer."""

    sources = {
        str(body.get("body_id")): str(body.get("mesh_source") or "trace_mesh")
        for body in bodies
    }
    return {
        "mesh_sources": sources,
        "original_step_solids_rendered": all(
            source == ORIGINAL_STEP_MESH_SOURCE for source in sources.values()
        ),
        "collision_geometry_rendered": any(
            source == "collision_components_fallback" for source in sources.values()
        ),
        "size_box_rendered": any(
            source == "size_box_fallback" for source in sources.values()
        ),
    }


def _trace_scene_bodies(
    workspace: Path,
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return renderable bodies for legacy two-body and compound traces.

    Compound bodies render from the original STEP solids whenever the trace's
    package is available; the saved collision components (or the historical
    ``size_m`` box) are only a labelled fallback, never the default.
    """

    collision = trace.get("collision_geometry") or {}
    compound = collision.get("bodies")
    if isinstance(compound, list) and compound:
        step_meshes = _original_step_body_meshes(workspace, collision)
        bodies: list[dict[str, Any]] = []
        for item in compound:
            if not isinstance(item, dict):
                raise ValueError("compound collision body must be an object")
            body_id = str(item["body_id"])
            components = item.get("components")
            mesh_source = ORIGINAL_STEP_MESH_SOURCE
            if body_id in step_meshes:
                mesh = step_meshes[body_id]
            elif isinstance(components, list) and components:
                mesh_source = "collision_components_fallback"
                meshes: list[trimesh.Trimesh] = []
                for component in components:
                    if not isinstance(component, dict):
                        raise ValueError(
                            f"compound body {body_id!r} has an invalid component"
                        )
                    mesh_reference = (
                        component.get("derived")
                        or component.get("mesh")
                        or component.get("source")
                    )
                    if not mesh_reference:
                        raise ValueError(
                            f"compound body {body_id!r} component has no mesh reference"
                        )
                    path = _resolve_workspace_asset(workspace, mesh_reference)
                    expected_hash = (
                        component.get("derived_sha256")
                        or component.get("sha256")
                        or component.get("source_sha256")
                    )
                    if expected_hash and _sha256(path) != expected_hash:
                        raise ValueError(
                            f"compound body {body_id!r} component hash changed"
                        )
                    loaded = trimesh.load_mesh(path, process=False)
                    if not isinstance(loaded, trimesh.Trimesh):
                        loaded = trimesh.util.concatenate(
                            tuple(loaded.geometry.values())
                        )
                    if len(loaded.vertices) and len(loaded.faces):
                        meshes.append(loaded)
                if not meshes:
                    raise ValueError(
                        f"compound body {body_id!r} has no renderable components"
                    )
                mesh = trimesh.util.concatenate(tuple(meshes))
            else:
                mesh_source = "size_box_fallback"
                size = np.asarray(item["size_m"], dtype=float)
                if (
                    size.shape != (3,)
                    or not np.all(np.isfinite(size))
                    or np.any(size <= 0.0)
                ):
                    raise ValueError(f"compound body {body_id!r} has invalid size_m")
                mesh = trimesh.creation.box(extents=size)
            fixed = bool(item.get("fixed"))
            mesh = _video_mesh_proxy(mesh, fixed=fixed)
            bodies.append(
                {
                    "body_id": body_id,
                    "fixed": fixed,
                    "mesh": mesh,
                    "mesh_source": mesh_source,
                }
            )
        fixture = collision.get("external_contact_fixture")
        if (
            isinstance(fixture, dict)
            and fixture.get("runtime_enabled") is True
            and fixture.get("shape") == "passive_phone_box"
        ):
            fixture_size = np.asarray(fixture.get("size_m"), dtype=float)
            fixture_id = str(fixture.get("fixture_id") or "")
            if (
                fixture_size.shape != (3,)
                or not np.all(np.isfinite(fixture_size))
                or np.any(fixture_size <= 0.0)
                or not fixture_id
            ):
                raise ValueError("passive phone fixture has invalid render data")
            bodies.append(
                {
                    "body_id": fixture_id,
                    "fixed": False,
                    "mesh": trimesh.creation.box(extents=fixture_size),
                    "render_color": "#202633",
                    "render_alpha": 0.94,
                    "optional_pose": True,
                }
            )
        return bodies

    fixed_item = collision.get("fixed")
    moving_item = collision.get("moving")
    if not isinstance(fixed_item, dict) or not isinstance(moving_item, dict):
        raise ValueError("trace collision geometry has neither bodies nor fixed/moving meshes")
    samples = trace.get("samples") or []
    first_body_poses = (
        samples[0].get("body_poses")
        if samples and isinstance(samples[0], dict)
        else None
    )
    exposes_moving_tool = isinstance(first_body_poses, dict)
    bodies = [
        {
            "body_id": "fixed_boundary" if exposes_moving_tool else "__fixed__",
            "fixed": True,
            "mesh": _load_mesh(workspace, fixed_item),
            **(
                {}
                if exposes_moving_tool
                else {
                    "constant_pose": (
                        np.asarray(fixed_item["body_origin_abs_m"], dtype=float),
                        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
                    )
                }
            ),
        },
        {
            "body_id": "moving_group" if exposes_moving_tool else None,
            "fixed": False,
            "mesh": _load_mesh(workspace, moving_item),
        },
    ]
    fixture = collision.get("external_contact_fixture")
    if (
        exposes_moving_tool
        and isinstance(fixture, dict)
        and fixture.get("runtime_enabled") is True
        and fixture.get("shape") == "tabletop_pickup_box"
    ):
        object_size = np.asarray(fixture.get("size_m"), dtype=float)
        table_size = np.asarray(fixture.get("table_size_m"), dtype=float)
        if (
            object_size.shape != (3,)
            or table_size.shape != (3,)
            or np.any(object_size <= 0.0)
            or np.any(table_size <= 0.0)
        ):
            raise ValueError("tabletop pickup fixture has invalid render dimensions")
        bodies.extend(
            [
                {
                    "body_id": "spring_grip_test_object",
                    "fixed": False,
                    "mesh": trimesh.creation.box(extents=object_size),
                    "render_color": "#ef9b25",
                    "render_alpha": 0.96,
                },
                {
                    "body_id": "spring_pickup_table",
                    "fixed": True,
                    "mesh": trimesh.creation.box(extents=table_size),
                    "render_color": "#8b6f55",
                    "render_alpha": 0.42,
                },
            ]
        )
    return bodies


def _body_pose_at(
    body: dict[str, Any], samples: list[dict[str, Any]], time_s: float
) -> tuple[np.ndarray, np.ndarray]:
    constant = body.get("constant_pose")
    if constant is not None:
        return constant
    if body.get("optional_pose") is True:
        body_id = body.get("body_id")
        available = [
            sample
            for sample in samples
            if isinstance((sample.get("body_poses") or {}).get(body_id), dict)
        ]
        if not available:
            raise ValueError(f"trace never exposes optional body pose {body_id!r}")
        left_index, right_index, _fraction = _sample_bracket(samples, time_s)
        left_pose = (samples[left_index].get("body_poses") or {}).get(body_id)
        right_pose = (samples[right_index].get("body_poses") or {}).get(body_id)
        if not isinstance(left_pose, dict) or not isinstance(right_pose, dict):
            nearest = min(
                available,
                key=lambda sample: abs(float(sample["time_s"]) - time_s),
            )
            pose = nearest["body_poses"][body_id]
            return (
                np.asarray(pose["position_abs_m"], dtype=float),
                np.asarray(pose["rotation_abs_quat_wxyz"], dtype=float),
            )
    return _interpolated_body_pose(samples, time_s, body.get("body_id"))


def _body_vertices_mm(
    body: dict[str, Any], samples: list[dict[str, Any]], time_s: float
) -> np.ndarray:
    position, quaternion = _body_pose_at(body, samples, time_s)
    return _transform_vertices_mm(
        np.asarray(body["mesh"].vertices, dtype=float), position, quaternion
    )


def _shared_scene_bounds_mm(
    scene_groups: Iterable[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
) -> tuple[np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    for bodies, samples in scene_groups:
        stride = max(1, len(samples) // 24)
        sampled = samples[::stride]
        if sampled[-1] is not samples[-1]:
            sampled = [*sampled, samples[-1]]
        for body in bodies:
            corners = _mesh_bound_corners(
                np.asarray(body["mesh"].vertices, dtype=float)
            )
            for sample in sampled:
                position, quaternion = _body_pose_at(
                    body, samples, float(sample["time_s"])
                )
                points.append(_transform_vertices_mm(corners, position, quaternion))
    if not points:
        raise ValueError("trace scene contains no renderable bodies")
    combined = np.vstack(points)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    span = maximum - minimum
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.18)
    center = (minimum + maximum) * 0.5
    half = span * 0.62
    return center - half, center + half


def _interpolated_scalar(
    samples: list[dict[str, Any]], time_s: float, key: str
) -> float:
    left, right, fraction = _sample_bracket(samples, time_s)
    left_value = float(samples[left].get(key, 0.0))
    right_value = float(samples[right].get(key, left_value))
    return left_value + fraction * (right_value - left_value)


def _moving_chain_marker_positions_mm(
    trace: dict[str, Any],
    samples: list[dict[str, Any]],
    time_s: float,
) -> np.ndarray | None:
    """Interpolate only the validator-authored chain material phase."""

    contract = trace.get("moving_chain_path_contract")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("moving-chain path contract is not an object")
    points = np.asarray(contract.get("centerline_points_abs_m"), dtype=float)
    cumulative = np.asarray(contract.get("cumulative_lengths_m"), dtype=float)
    total = float(contract.get("total_length_m") or 0.0)
    material_direction = contract.get("material_motion_per_phase_sign")
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or len(points) < 3
        or cumulative.shape != (len(points),)
        or not np.all(np.isfinite(points))
        or not np.all(np.isfinite(cumulative))
        or total <= 0.0
        or material_direction != -1
        or not np.all(np.diff(cumulative) > 0.0)
        or not math.isclose(float(cumulative[-1]), total, abs_tol=1.0e-9)
    ):
        raise ValueError("moving-chain path geometry is malformed")
    left, right, fraction = _sample_bracket(samples, time_s)

    def phase(index: int) -> float:
        state = samples[index].get("moving_chain_path_state")
        if not isinstance(state, dict):
            raise ValueError(
                f"saved sample {index} has no moving-chain path state"
            )
        if state.get("contract_sha256") != contract.get("contract_sha256"):
            raise ValueError("moving-chain state is bound to another contract")
        return float(state["phase_unwrapped_m"])

    phase_m = phase(left) + fraction * (phase(right) - phase(left))
    marker_count = max(
        1,
        min(64, int(contract.get("visual_marker_count") or 64)),
    )
    marker_distances = (
        material_direction * phase_m
        + np.arange(marker_count, dtype=float) * total / marker_count
    ) % total
    result = np.empty((marker_count, 3), dtype=float)
    for marker_index, distance in enumerate(marker_distances):
        edge = int(np.searchsorted(cumulative, distance, side="right") - 1)
        edge = max(0, min(edge, len(points) - 2))
        span = float(cumulative[edge + 1] - cumulative[edge])
        local = 0.0 if span <= 0.0 else (
            float(distance) - float(cumulative[edge])
        ) / span
        result[marker_index] = points[edge] + local * (
            points[edge + 1] - points[edge]
        )
    return result * 1000.0


def _motion_coordinate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Return readable coordinates for linear, angular and multi-shaft traces."""

    sample = samples[0]
    angular_key = next(
        (
            key
            for key in (
                "generalized_angle_rad",
                "rear_drive_wheel_angle_rad",
                "input_crank_angle_rad",
            )
            if key in sample
        ),
        None,
    )
    if angular_key is not None:
        secondary_key = (
            "max_joint_drift_m"
            if "max_joint_drift_m" in sample
            else "max_support_drift_m"
            if "max_support_drift_m" in sample
            else "pivot_drift_m"
        )
        secondary_labels = {
            "max_joint_drift_m": "关节漂移",
            "max_support_drift_m": "轴承支撑漂移",
            "pivot_drift_m": "转轴漂移",
        }
        secondary_labels_en = {
            "max_joint_drift_m": "Joint drift",
            "max_support_drift_m": "Bearing drift",
            "pivot_drift_m": "Pivot drift",
        }
        return {
            "kind": "angular",
            "primary_key": angular_key,
            "primary_scale": 180.0 / math.pi,
            "primary_label_zh": (
                "后轮转角 / °"
                if angular_key == "rear_drive_wheel_angle_rad"
                else "曲柄转角 / °"
                if angular_key == "input_crank_angle_rad"
                else "转角 / °"
            ),
            "primary_label_en": (
                "Rear wheel / °"
                if angular_key == "rear_drive_wheel_angle_rad"
                else "Crank angle / °"
                if angular_key == "input_crank_angle_rad"
                else "Angle / °"
            ),
            "secondary_key": secondary_key,
            "secondary_quantity_zh": secondary_labels[secondary_key],
            "secondary_quantity_en": secondary_labels_en[secondary_key],
        }
    if "generalized_progress_m" in sample:
        return {
            "kind": "linear",
            "primary_key": "generalized_progress_m",
            "primary_scale": 1000.0,
            "primary_label_zh": "行程 / mm",
            "primary_label_en": "Travel / mm",
            "secondary_key": "lateral_shift_m",
            "secondary_quantity_zh": "横向偏移",
            "secondary_quantity_en": "Lateral shift",
        }
    raise ValueError("trace does not expose a supported generalized coordinate")


def _stepped_value(
    samples: list[dict[str, Any]], time_s: float, key: str, default: Any
) -> Any:
    left, _, _ = _sample_bracket(samples, time_s)
    return samples[left].get(key, default)


def _coordinate_text(coordinate: dict[str, Any], field: str) -> str:
    """Drawn text of a ``_motion_coordinate`` label field, e.g. ``primary_label``."""

    return label(str(coordinate[f"{field}_zh"]), str(coordinate[f"{field}_en"]))


def _phase_label(phase_id: Any) -> str:
    phase = str(phase_id or "initial")
    return label(
        PHASE_LABELS_ZH.get(phase, "其他阶段"),
        PHASE_LABELS_EN.get(phase, "Other phase"),
    )


def _load_label(load_case_id: str) -> str:
    return label(
        LOAD_LABELS_ZH.get(load_case_id, "机构工况"),
        LOAD_LABELS_EN.get(load_case_id, "Mechanism load case"),
    )


def _fixed_vertices_mm(
    mesh: trimesh.Trimesh, collision_item: dict[str, Any]
) -> np.ndarray:
    return _transform_vertices_mm(
        np.asarray(mesh.vertices, dtype=float),
        collision_item["body_origin_abs_m"],
        [1.0, 0.0, 0.0, 0.0],
    )


def _mesh_bound_corners(vertices: np.ndarray) -> np.ndarray:
    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ],
        dtype=float,
    )


def _shared_bounds_mm(
    fixed_vertices: Iterable[np.ndarray],
    moving_meshes: Iterable[trimesh.Trimesh],
    sample_groups: Iterable[list[dict[str, Any]]],
) -> tuple[np.ndarray, np.ndarray]:
    points = [np.asarray(vertices, dtype=float) for vertices in fixed_vertices]
    for mesh, samples in zip(moving_meshes, sample_groups):
        corners = _mesh_bound_corners(np.asarray(mesh.vertices, dtype=float))
        for sample in samples:
            pose = sample["moving_pose"]
            points.append(
                _transform_vertices_mm(
                    corners,
                    pose["position_abs_m"],
                    pose["rotation_abs_quat_wxyz"],
                )
            )
    combined = np.vstack(points)
    minimum = combined.min(axis=0)
    maximum = combined.max(axis=0)
    span = maximum - minimum
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.18)
    center = (minimum + maximum) * 0.5
    half = span * 0.62
    return center - half, center + half


def _configure_3d_axis(axis, minimum: np.ndarray, maximum: np.ndarray) -> None:
    span = maximum - minimum
    axis.set_xlim(float(minimum[0]), float(maximum[0]))
    axis.set_ylim(float(minimum[1]), float(maximum[1]))
    axis.set_zlim(float(minimum[2]), float(maximum[2]))
    axis.set_box_aspect(np.maximum(span, 1e-6))
    axis.view_init(elev=22.0, azim=-58.0)
    axis.set_facecolor("#ffffff")
    axis.grid(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.set_zlabel("")
    for pane in (axis.xaxis.pane, axis.yaxis.pane, axis.zaxis.pane):
        pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))


def _padded_limits(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    span = maximum - minimum
    if span <= 1e-12:
        pad = max(abs(maximum) * 0.18, 0.01)
    else:
        pad = span * 0.16
    return minimum - pad, maximum + pad


def _build_sidecar_base(
    physical_path: Path,
    reduced_path: Path,
    physical: dict[str, Any],
    reduced: dict[str, Any],
    physical_samples: list[dict[str, Any]],
    reduced_samples: list[dict[str, Any]],
    time_grid: np.ndarray,
    output: Path,
    fps: int,
    dpi: int,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "artifact_kind": "saved_trace_pair_replay",
        "notice_zh": NOTICE_ZH,
        "verdict": None,
        "status": "ENCODING",
        "encoded_frames": 0,
        "trace_sha256": {
            "physical": _sha256(physical_path),
            "reduced": _sha256(reduced_path),
        },
        "trace_paths": {
            "physical": str(physical_path),
            "reduced": str(reduced_path),
        },
        "sample_counts": {
            "physical_declared": physical.get("sample_count"),
            "physical_loaded": len(physical_samples),
            "reduced_declared": reduced.get("sample_count"),
            "reduced_loaded": len(reduced_samples),
        },
        "capsule_id": physical.get("capsule_id"),
        "load_case_id": physical.get("load_case_id"),
        "time_grid_s": [round(float(value), 9) for value in time_grid],
        "source_time_span_s": {
            "physical": [
                float(physical_samples[0]["time_s"]),
                float(physical_samples[-1]["time_s"]),
            ],
            "reduced": [
                float(reduced_samples[0]["time_s"]),
                float(reduced_samples[-1]["time_s"]),
            ],
        },
        "synchronization": {
            "method": "shared source-time grid",
            "position_and_scalars": "linear interpolation with endpoint clamp",
            "rotation": "quaternion slerp with endpoint clamp",
            "phase_and_contact_count": "previous saved sample",
        },
        "rendering": {
            "saved_collision_geometry_is_unchanged": True,
            "video_plot_mesh_may_be_decimated": True,
            "video_plot_mesh_is_used_for_physics": False,
        },
        "encoding": {
            "status": "ENCODING",
            "encoded_frames": 0,
            "fps": fps,
            "dpi": dpi,
            "frame_size_px": [1280, 720],
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "output": str(output),
        },
    }


def _sidecar_path(output: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    return output.with_suffix(".sidecar.json")


def render_physical_trace_video(
    *,
    workspace_root: str | Path,
    physical_trace: str | Path,
    output: str | Path,
    sidecar: str | Path | None = None,
    fps: int = 20,
    dpi: int = 100,
    minimum_playback_s: float = 4.0,
    force: bool = False,
) -> dict[str, Any]:
    """Render one complete or partial saved physical trace without inventing a pair."""

    if fps <= 0 or dpi <= 0 or minimum_playback_s <= 0:
        raise ValueError("fps, dpi, and minimum_playback_s must be positive")
    workspace = Path(workspace_root).resolve()
    physical_path = _resolve_input(physical_trace)
    output_path = Path(output).resolve()
    sidecar_path = _sidecar_path(output_path, sidecar)
    physical = _read_json(physical_path)
    if physical.get("branch") != "physical":
        raise ValueError("trace must declare branch=physical")
    samples = _validated_samples(physical, physical_path)
    trace_complete = physical.get("status") == "COMPLETED"
    trace_sha256 = _sha256(physical_path)
    if (
        not force
        and output_path.is_file()
        and sidecar_path.is_file()
    ):
        existing = _read_json(sidecar_path)
        if (
            existing.get("status") == "ENCODED"
            and existing.get("trace_sha256") == trace_sha256
            and existing.get("trace_complete") is trace_complete
        ):
            return existing

    _configure_chinese_font()
    if not _configure_ffmpeg_writer():
        raise RuntimeError("ffmpeg writer is unavailable")
    bodies = _trace_scene_bodies(workspace, physical)
    coordinate = _motion_coordinate(samples)
    start_time = float(samples[0]["time_s"])
    end_time = float(samples[-1]["time_s"])
    frame_count = max(
        len(samples),
        int(math.ceil(minimum_playback_s * fps)),
    )
    frame_count = min(frame_count, 180)
    time_grid = np.linspace(start_time, end_time, frame_count)
    minimum, maximum = _shared_scene_bounds_mm(((bodies, samples),))

    source_times = np.asarray(
        [float(sample["time_s"]) for sample in samples],
        dtype=float,
    )
    primary_values = np.asarray(
        [
            float(sample.get(coordinate["primary_key"], 0.0))
            * float(coordinate["primary_scale"])
            for sample in samples
        ],
        dtype=float,
    )
    secondary_values = np.asarray(
        [
            float(sample.get(coordinate["secondary_key"], 0.0)) * 1000.0
            for sample in samples
        ],
        dtype=float,
    )
    contact_values = np.asarray(
        [float(sample.get("contact_count", 0.0)) for sample in samples],
        dtype=float,
    )

    fig = plt.figure(figsize=(12.8, 7.2), facecolor=PAPER)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(2.15, 1.0),
        width_ratios=(1.34, 1.0),
        hspace=0.22,
        wspace=0.15,
    )
    assembly = fig.add_subplot(grid[:, 0], projection="3d")
    motion_axis = fig.add_subplot(grid[0, 1])
    diagnostic_axis = fig.add_subplot(grid[1, 1])
    _configure_3d_axis(assembly, minimum, maximum)

    moving_colors = (
        "#2f6fed",
        "#19a577",
        "#d39b24",
        "#8d55c7",
        "#e24a4a",
        "#2f9dac",
    )
    artists: list[tuple[dict[str, Any], Poly3DCollection]] = []
    for body_index, body in enumerate(bodies):
        vertices = _body_vertices_mm(body, samples, start_time)
        fixed = bool(body.get("fixed"))
        color = body.get("render_color") or (
            FIXED_COLOR if fixed else moving_colors[body_index % len(moving_colors)]
        )
        artist = Poly3DCollection(
            vertices[body["mesh"].faces],
            facecolor=color,
            edgecolor=FIXED_EDGE if fixed else color,
            linewidth=0.10 if fixed else 0.16,
            alpha=float(
                body.get("render_alpha", 0.26 if fixed else 0.82)
            ),
        )
        assembly.add_collection3d(artist)
        artists.append((body, artist))
    chain_positions = _moving_chain_marker_positions_mm(
        physical,
        samples,
        start_time,
    )
    chain_artist = None
    if chain_positions is not None:
        chain_artist = assembly.scatter(
            chain_positions[:, 0],
            chain_positions[:, 1],
            chain_positions[:, 2],
            s=13.0,
            c="#f0aa24",
            edgecolors="#7a4a00",
            linewidths=0.25,
            depthshade=False,
        )
    assembly.set_title(
        label("真实物理保存轨迹回放", "Physical saved-trajectory replay"),
        color=INK,
        fontsize=13,
        fontweight="bold",
    )

    for axis in (motion_axis, diagnostic_axis):
        axis.set_facecolor("#ffffff")
        axis.grid(color=GRID, linewidth=0.8, alpha=0.9)
        axis.set_xlim(start_time, max(end_time, start_time + 1.0e-9))
    motion_axis.plot(
        source_times,
        primary_values,
        color=PHYSICAL_COLOR,
        linewidth=2.2,
    )
    motion_axis.set_ylabel(_coordinate_text(coordinate, "primary_label"))
    motion_axis.set_ylim(*_padded_limits(primary_values))
    secondary_quantity = _coordinate_text(coordinate, "secondary_quantity")
    diagnostic_axis.plot(
        source_times,
        secondary_values,
        color="#c17b1f",
        linewidth=2.0,
        label=f"{secondary_quantity} / mm",
    )
    contact_axis = diagnostic_axis.twinx()
    contact_axis.plot(
        source_times,
        contact_values,
        color="#8d55c7",
        linewidth=1.6,
        alpha=0.82,
        label=label("接触数", "Contacts"),
    )
    diagnostic_axis.set_ylabel(f"{secondary_quantity} / mm")
    contact_axis.set_ylabel(label("接触数", "Contacts"))
    diagnostic_axis.set_xlabel(label("已保存的仿真时间 / s", "Saved simulation time / s"))
    motion_cursor = motion_axis.axvline(
        start_time,
        color=INK,
        linewidth=1.0,
        alpha=0.55,
    )
    diagnostic_cursor = diagnostic_axis.axvline(
        start_time,
        color=INK,
        linewidth=1.0,
        alpha=0.55,
    )
    status = fig.text(
        0.585,
        0.955,
        "",
        color=INK,
        fontsize=11.5,
        fontweight="bold",
    )
    completion_note = (
        label("完整物理轨迹", "Complete physical trajectory")
        if trace_complete
        else label(
            "部分物理轨迹 · 超时前检查点 · 不可用于证书",
            "Partial physical trajectory · pre-timeout checkpoint · not certifiable",
        )
    )
    note = label(
        f"{completion_note} · 只回放已原子保存的状态，没有重跑或补帧",
        f"{completion_note}\nReplays atomically saved states only; no rerun or added frames",
    )
    for line_index, note_line in enumerate(note.split("\n")):
        fig.text(
            0.585,
            0.925 - 0.022 * line_index,
            note_line,
            color=MUTED,
            fontsize=9.2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_payload = {
        "schema_version": "0.1",
        "artifact_kind": "saved_physical_trace_replay",
        "status": "ENCODING",
        "verdict": None,
        "trace_path": str(physical_path),
        "trace_sha256": trace_sha256,
        "trace_status": str(physical.get("status") or ""),
        "trace_complete": trace_complete,
        "certification_authority": False,
        "sample_count": len(samples),
        "rendering": _mesh_source_summary(bodies),
        "moving_chain_phase_rendered": chain_artist is not None,
        "moving_chain_contract_sha256": str(
            (physical.get("moving_chain_path_contract") or {}).get(
                "contract_sha256"
            )
            or ""
        ),
        "source_time_span_s": [start_time, end_time],
        "encoded_frames": 0,
        "encoding": {
            "fps": fps,
            "dpi": dpi,
            "codec": "libx264",
            "pixel_format": "yuv420p",
            "output": str(output_path),
        },
    }
    _write_json_atomic(sidecar_path, sidecar_payload)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=5200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    try:
        with writer.saving(fig, str(output_path), dpi):
            for frame_index, time_s in enumerate(time_grid):
                for body, artist in artists:
                    vertices = _body_vertices_mm(body, samples, float(time_s))
                    artist.set_verts(vertices[body["mesh"].faces])
                if chain_artist is not None:
                    chain_positions = _moving_chain_marker_positions_mm(
                        physical,
                        samples,
                        float(time_s),
                    )
                    chain_artist._offsets3d = (
                        chain_positions[:, 0],
                        chain_positions[:, 1],
                        chain_positions[:, 2],
                    )
                motion_cursor.set_xdata([time_s, time_s])
                diagnostic_cursor.set_xdata([time_s, time_s])
                phase = _phase_label(
                    _stepped_value(samples, float(time_s), "phase_id", "initial")
                )
                # English drops the completion note here (it is on the line
                # below) so the status line fits the frame.
                status.set_text(
                    label(
                        f"{completion_note}  ·  {phase}  ·  "
                        f"{time_s:.3f} s  ·  帧 {frame_index + 1}/{len(time_grid)}",
                        f"{phase}  ·  {time_s:.3f} s  ·  "
                        f"frame {frame_index + 1}/{len(time_grid)}",
                    )
                )
                writer.grab_frame()
        sidecar_payload["status"] = "ENCODED"
        sidecar_payload["encoded_frames"] = len(time_grid)
        sidecar_payload["output_sha256"] = _sha256(output_path)
        sidecar_payload["encoding"]["status"] = "ENCODED"
        sidecar_payload["encoding"]["encoded_frames"] = len(time_grid)
        _write_json_atomic(sidecar_path, sidecar_payload)
        return sidecar_payload
    except Exception as exc:
        sidecar_payload["status"] = "FAILED"
        sidecar_payload["error"] = f"{type(exc).__name__}: {exc}"
        sidecar_payload["encoding"]["status"] = "FAILED"
        _write_json_atomic(sidecar_path, sidecar_payload)
        raise
    finally:
        plt.close(fig)


def render_trace_pair_video(
    *,
    workspace_root: str | Path,
    physical_trace: str | Path,
    reduced_trace: str | Path,
    output: str | Path,
    sidecar: str | Path | None = None,
    fps: int = 20,
    dpi: int = 100,
    minimum_playback_s: float = 4.0,
    force: bool = False,
) -> dict[str, Any]:
    """Encode one synchronized MP4 from two existing traces.

    The function never imports or invokes a physics worker.  ``status`` in the
    returned sidecar is an encoding state (``ENCODED``/``FAILED``), never a
    mechanism PASS verdict.
    """

    if fps <= 0 or dpi <= 0 or minimum_playback_s <= 0:
        raise ValueError("fps, dpi, and minimum_playback_s must be positive")
    workspace = Path(workspace_root).resolve()
    physical_path = _resolve_input(physical_trace)
    reduced_path = _resolve_input(reduced_trace)
    output_path = Path(output).resolve()
    sidecar_path = _sidecar_path(output_path, sidecar)

    physical = _read_json(physical_path)
    reduced = _read_json(reduced_path)
    _validate_trace_pair(physical, reduced)
    physical_samples = _validated_samples(physical, physical_path)
    reduced_samples = _validated_samples(reduced, reduced_path)

    physical_hash = _sha256(physical_path)
    reduced_hash = _sha256(reduced_path)
    if output_path.is_file() and sidecar_path.is_file() and not force:
        existing = _read_json(sidecar_path)
        hashes = existing.get("trace_sha256") or {}
        if (
            existing.get("status") == "ENCODED"
            and hashes.get("physical") == physical_hash
            and hashes.get("reduced") == reduced_hash
        ):
            return existing
        raise FileExistsError(
            f"output exists but its sidecar is absent or stale; use force=True: {output_path}"
        )
    if output_path.exists() and not force:
        raise FileExistsError(output_path)

    start_s = min(
        float(physical_samples[0]["time_s"]),
        float(reduced_samples[0]["time_s"]),
    )
    end_s = max(
        float(physical_samples[-1]["time_s"]),
        float(reduced_samples[-1]["time_s"]),
    )
    source_duration_s = max(0.0, end_s - start_s)
    encoded_frames = max(
        2,
        int(math.ceil(max(source_duration_s, minimum_playback_s) * fps)) + 1,
    )
    time_grid = np.linspace(start_s, end_s, encoded_frames, dtype=float)
    metadata = _build_sidecar_base(
        physical_path,
        reduced_path,
        physical,
        reduced,
        physical_samples,
        reduced_samples,
        time_grid,
        output_path,
        fps,
        dpi,
    )
    metadata["trace_sha256"] = {
        "physical": physical_hash,
        "reduced": reduced_hash,
    }
    right_branch = str(reduced.get("branch") or "")
    right_is_ideal = right_branch == "ideal_reference"
    right_label = (
        label("理想参考", "Ideal reference")
        if right_is_ideal
        else label("简化机构", "Reduced mechanism")
    )
    metadata["comparison_branch"] = right_branch
    coordinate = _motion_coordinate(physical_samples)
    reduced_coordinate = _motion_coordinate(reduced_samples)
    is_clock = physical.get("mechanism_plugin_id") == "spur_gear_clock_v1"
    if coordinate["kind"] != reduced_coordinate["kind"]:
        raise ValueError("trace pair uses different generalized coordinate families")
    metadata["motion_coordinate"] = {
        "kind": coordinate["kind"],
        "primary_label_zh": coordinate["primary_label_zh"],
        "secondary_quantity_zh": coordinate["secondary_quantity_zh"],
    }
    _write_json_atomic(sidecar_path, metadata)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_output = output_path.with_name(output_path.stem + ".partial" + output_path.suffix)
    if partial_output.exists():
        partial_output.unlink()

    fig = None
    frames_written = 0
    try:
        if not _configure_ffmpeg_writer():
            raise RuntimeError("Matplotlib cannot find ffmpeg for MP4 encoding")
        _configure_chinese_font()

        p_bodies = _trace_scene_bodies(workspace, physical)
        r_bodies = (
            _trace_scene_bodies(workspace, physical)
            if right_is_ideal
            else _trace_scene_bodies(workspace, reduced)
        )
        metadata["rendering"].update(_mesh_source_summary(p_bodies))
        lower, upper = _shared_scene_bounds_mm(
            (
                (p_bodies, physical_samples),
                (r_bodies, reduced_samples),
            )
        )

        p_progress = np.asarray(
            [
                _interpolated_scalar(physical_samples, time_s, "generalized_progress_m")
                * 1000.0
                for time_s in time_grid
            ],
            dtype=float,
        )
        r_progress = np.asarray(
            [
                _interpolated_scalar(reduced_samples, time_s, "generalized_progress_m")
                * 1000.0
                for time_s in time_grid
            ],
            dtype=float,
        )
        p_lateral_m = np.asarray(
            [
                _interpolated_scalar(physical_samples, time_s, "lateral_shift_m")
                for time_s in time_grid
            ],
            dtype=float,
        )
        r_lateral_m = np.asarray(
            [
                _interpolated_scalar(reduced_samples, time_s, "lateral_shift_m")
                for time_s in time_grid
            ],
            dtype=float,
        )
        maximum_lateral_mm = max(
            float(np.max(np.abs(p_lateral_m))) * 1000.0,
            float(np.max(np.abs(r_lateral_m))) * 1000.0,
        )
        if maximum_lateral_mm < 0.1:
            lateral_scale = 1_000_000.0
            lateral_label = label("横向偏移 / μm", "Lateral shift / μm")
        else:
            lateral_scale = 1000.0
            lateral_label = label("横向偏移 / mm", "Lateral shift / mm")
        p_lateral = p_lateral_m * lateral_scale
        r_lateral = r_lateral_m * lateral_scale

        # The saved trace declares whether its generalized coordinate is
        # linear or angular; never infer units from a demo name.
        primary_key = str(coordinate["primary_key"])
        primary_scale = float(coordinate["primary_scale"])
        secondary_key = str(coordinate["secondary_key"])
        p_progress = np.asarray(
            [_interpolated_scalar(physical_samples, time_s, primary_key) * primary_scale for time_s in time_grid],
            dtype=float,
        )
        r_progress = np.asarray(
            [_interpolated_scalar(reduced_samples, time_s, primary_key) * primary_scale for time_s in time_grid],
            dtype=float,
        )
        p_secondary_m = np.asarray(
            [_interpolated_scalar(physical_samples, time_s, secondary_key) for time_s in time_grid],
            dtype=float,
        )
        r_secondary_m = np.asarray(
            [_interpolated_scalar(reduced_samples, time_s, secondary_key) for time_s in time_grid],
            dtype=float,
        )
        maximum_secondary_mm = max(
            float(np.max(np.abs(p_secondary_m))) * 1000.0,
            float(np.max(np.abs(r_secondary_m))) * 1000.0,
        )
        secondary_quantity = _coordinate_text(coordinate, "secondary_quantity")
        if maximum_secondary_mm < 0.1:
            secondary_scale = 1_000_000.0
            secondary_label = f"{secondary_quantity} / μm"
        else:
            secondary_scale = 1000.0
            secondary_label = f"{secondary_quantity} / mm"
        p_lateral = p_secondary_m * secondary_scale
        r_lateral = r_secondary_m * secondary_scale
        if right_is_ideal:
            p_lateral = p_progress - r_progress
            r_lateral = np.zeros_like(p_lateral)
            secondary_label = (
                label("真实物理 − 理想参考误差 / °", "Physical − ideal error / °")
                if coordinate["kind"] == "angular"
                else label("真实物理 − 理想参考误差 / mm", "Physical − ideal error / mm")
            )

        # Keep the encoded canvas at 1280 x 720 for every accepted DPI value.
        fig = plt.figure(figsize=(1280.0 / dpi, 720.0 / dpi), facecolor=PAPER)
        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=(4.5, 0.88, 0.88),
            left=0.055,
            right=0.98,
            bottom=0.085,
            top=0.855,
            hspace=0.22,
            wspace=0.025,
        )
        physical_axis = fig.add_subplot(grid[0, 0], projection="3d")
        reduced_axis = fig.add_subplot(grid[0, 1], projection="3d")
        progress_axis = fig.add_subplot(grid[1, :])
        lateral_axis = fig.add_subplot(grid[2, :], sharex=progress_axis)
        for axis in (physical_axis, reduced_axis):
            _configure_3d_axis(axis, lower, upper)

        def add_scene_artists(
            axis,
            bodies: list[dict[str, Any]],
            samples: list[dict[str, Any]],
            moving_color: str,
            moving_edge: str,
        ) -> list[tuple[dict[str, Any], Poly3DCollection]]:
            artists: list[tuple[dict[str, Any], Poly3DCollection]] = []
            for body_index, body in enumerate(bodies):
                vertices = _body_vertices_mm(body, samples, float(time_grid[0]))
                mesh = body["mesh"]
                fixed = bool(body.get("fixed"))
                artist = Poly3DCollection(
                    vertices[mesh.faces],
                    facecolor=FIXED_COLOR if fixed else moving_color,
                    edgecolor=FIXED_EDGE if fixed else moving_edge,
                    linewidth=0.18 if fixed else 0.24,
                    alpha=0.18 if fixed else max(0.62, 0.90 - body_index * 0.06),
                )
                axis.add_collection3d(artist)
                artists.append((body, artist))
            return artists

        p_artists = add_scene_artists(
            physical_axis,
            p_bodies,
            physical_samples,
            PHYSICAL_COLOR,
            PHYSICAL_EDGE,
        )
        r_artists = add_scene_artists(
            reduced_axis,
            r_bodies,
            reduced_samples,
            REDUCED_COLOR,
            REDUCED_EDGE,
        )
        chain_positions = _moving_chain_marker_positions_mm(
            physical,
            physical_samples,
            float(time_grid[0]),
        )
        chain_artist = None
        if chain_positions is not None:
            chain_artist = physical_axis.scatter(
                chain_positions[:, 0],
                chain_positions[:, 1],
                chain_positions[:, 2],
                s=10.0,
                c="#f0aa24",
                edgecolors="#7a4a00",
                linewidths=0.22,
                depthshade=False,
            )
        metadata["rendering"]["moving_chain_phase_rendered"] = (
            chain_artist is not None
        )
        metadata["rendering"]["moving_chain_contract_sha256"] = str(
            (physical.get("moving_chain_path_contract") or {}).get(
                "contract_sha256"
            )
            or ""
        )

        physical_axis.set_title(
            label(
                "真实实体物理\n有限柔性接口传力",
                "Physical solid model\nCompliant interfaces\ntransmit force",
            ),
            color=PHYSICAL_EDGE,
            fontsize=12,
            fontweight="bold",
            pad=3,
        )
        reduced_axis.set_title(
            (
                label(
                    "理想机构参考\n几何方程产生运动",
                    "Ideal mechanism reference\nMotion from geometric\nequations",
                )
                if right_is_ideal
                else label(
                    "简化机构模型\n约束关系产生运动",
                    "Reduced mechanism model\nMotion from constraints",
                )
            ),
            color=REDUCED_EDGE,
            fontsize=12,
            fontweight="bold",
            pad=3,
        )
        physical_badge = physical_axis.text2D(
            0.03,
            0.91,
            "",
            transform=physical_axis.transAxes,
            color=PHYSICAL_EDGE,
            fontsize=9.5,
            bbox={"facecolor": "#eef4ff", "edgecolor": "#cbdcff", "boxstyle": "round,pad=0.32"},
        )
        reduced_badge = reduced_axis.text2D(
            0.03,
            0.91,
            (
                label(
                    "无碰撞、无动力学，仅作运动目标",
                    "No collision or dynamics; motion target only",
                )
                if right_is_ideal
                else label("内部接触已替换", "Internal contacts replaced")
            ),
            transform=reduced_axis.transAxes,
            color=REDUCED_EDGE,
            fontsize=9.5,
            bbox={"facecolor": "#ecf8f1", "edgecolor": "#c6ead6", "boxstyle": "round,pad=0.32"},
        )
        _ = reduced_badge

        physical_axis.set_title(
            label(
                "真实实体物理\n有限柔性接口传力",
                "Physical solid model\nCompliant interfaces\ntransmit force",
            ),
            color=PHYSICAL_EDGE, fontsize=12, fontweight="bold", pad=3,
        )
        reduced_axis.set_title(
            (
                label(
                    "理想机构参考\n几何方程产生运动",
                    "Ideal mechanism reference\nMotion from geometric\nequations",
                )
                if right_is_ideal
                else label(
                    "简化机构模型\n约束关系产生运动",
                    "Reduced mechanism model\nMotion from constraints",
                )
            ),
            color=REDUCED_EDGE, fontsize=12, fontweight="bold", pad=3,
        )
        reduced_badge.set_text(
            (
                label(
                    "无碰撞、无动力学，仅作运动目标",
                    "No collision or dynamics; motion target only",
                )
                if right_is_ideal
                else label("内部接触已替换", "Internal contacts replaced")
            )
        )

        if is_clock:
            physical_axis.set_title(
                label(
                    "真实实体分支\n有限柔性啮合传递运动",
                    "Physical solid branch\nCompliant gear mesh\ntransmits motion",
                ),
                color=PHYSICAL_EDGE,
                fontsize=12,
                fontweight="bold",
                pad=3,
            )
            reduced_axis.set_title(
                (
                    label(
                        "理想机构参考\n解析齿轮方程产生运动",
                        "Ideal mechanism reference\nMotion from analytic\ngear equations",
                    )
                    if right_is_ideal
                    else label(
                        "简化机构分支\n解析齿轮与轴约束传递运动",
                        "Reduced mechanism branch\nAnalytic gear and shaft\nconstraints carry motion",
                    )
                ),
                color=REDUCED_EDGE,
                fontsize=12,
                fontweight="bold",
                pad=3,
            )
            reduced_badge.set_text(
                (
                    label(
                        "无碰撞、无动力学，仅作运动目标",
                        "No collision or dynamics; motion target only",
                    )
                    if right_is_ideal
                    else label(
                        "内部传动已替换为解析约束",
                        "Internal drive replaced by analytic constraints",
                    )
                )
            )

        for axis in (progress_axis, lateral_axis):
            axis.set_facecolor("#ffffff")
            axis.grid(axis="y", color=GRID, linewidth=0.8)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color("#bdc8d5")
            axis.tick_params(colors=MUTED, labelsize=8.5)
        progress_axis.plot(
            time_grid,
            p_progress,
            color=PHYSICAL_COLOR,
            linewidth=2.8 if right_is_ideal else 2.0,
            label=label("真实物理", "Physical"),
        )
        progress_axis.plot(
            time_grid,
            r_progress,
            color=REDUCED_COLOR,
            linewidth=1.6 if right_is_ideal else 2.0,
            linestyle="--" if right_is_ideal else "-",
            label=right_label,
        )
        lateral_axis.plot(
            time_grid,
            p_lateral,
            color=PHYSICAL_COLOR,
            linewidth=2.0,
        )
        lateral_axis.plot(
            time_grid,
            r_lateral,
            color=REDUCED_COLOR,
            linewidth=1.2,
            linestyle="--" if right_is_ideal else "-",
        )
        progress_axis.set_ylabel(label("行程 / mm", "Travel / mm"), color=INK, fontsize=9.5)
        lateral_axis.set_ylabel(lateral_label, color=INK, fontsize=9.5)
        lateral_axis.set_xlabel(
            label("保存轨迹时间 / 秒", "Saved trajectory time / s"), color=INK, fontsize=9.5
        )
        progress_axis.lines[0].set_label(label("真实物理", "Physical"))
        progress_axis.lines[1].set_label(right_label)
        progress_axis.set_ylabel(
            _coordinate_text(coordinate, "primary_label"), color=INK, fontsize=9.5
        )
        lateral_axis.set_ylabel(secondary_label, color=INK, fontsize=9.5)
        lateral_axis.set_xlabel(
            label("保存轨迹时间 / 秒", "Saved trajectory time / s"), color=INK, fontsize=9.5
        )
        progress_axis.set_ylim(*_padded_limits(np.concatenate((p_progress, r_progress))))
        lateral_axis.set_ylim(*_padded_limits(np.concatenate((p_lateral, r_lateral))))
        if end_s <= start_s:
            progress_axis.set_xlim(start_s - 0.5, end_s + 0.5)
        else:
            progress_axis.set_xlim(start_s, end_s)
        progress_axis.legend(
            loc="upper right", frameon=False, fontsize=8.5, ncol=2, borderaxespad=0.2
        )
        progress_cursor = progress_axis.axvline(
            float(time_grid[0]), color=INK, linewidth=1.0, alpha=0.62
        )
        lateral_cursor = lateral_axis.axvline(
            float(time_grid[0]), color=INK, linewidth=1.0, alpha=0.62
        )
        p_progress_marker, = progress_axis.plot(
            [time_grid[0]], [p_progress[0]], "o", color=PHYSICAL_COLOR, markersize=4.5
        )
        r_progress_marker, = progress_axis.plot(
            [time_grid[0]], [r_progress[0]], "o", color=REDUCED_COLOR, markersize=4.5
        )
        p_lateral_marker, = lateral_axis.plot(
            [time_grid[0]], [p_lateral[0]], "o", color=PHYSICAL_COLOR, markersize=4.2
        )
        r_lateral_marker, = lateral_axis.plot(
            [time_grid[0]], [r_lateral[0]], "o", color=REDUCED_COLOR, markersize=4.2
        )

        load_case_id = str(physical.get("load_case_id") or "")
        load_label = _load_label(load_case_id)
        notice = label(NOTICE_ZH, NOTICE_EN)
        fig.suptitle(
            label(
                f"{load_label}｜真实物理与{right_label}同步回放",
                f"{load_label} | Synchronized replay: physical vs {right_label.lower()}",
            ),
            x=0.5,
            y=0.965,
            color=INK,
            fontsize=15,
            fontweight="bold",
        )
        fig.suptitle(
            label(
                f"{load_label}｜真实物理与{right_label}同步回放",
                f"{load_label} | Synchronized replay: physical vs {right_label.lower()}",
            ),
            x=0.5, y=0.965, color=INK, fontsize=15, fontweight="bold",
        )
        fig.text(
            0.5,
            0.922,
            notice,
            ha="center",
            color=MUTED,
            fontsize=10,
        )
        phase_text = fig.text(
            0.5,
            0.875,
            "",
            ha="center",
            va="center",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )
        fig.patches.append(
            FancyBboxPatch(
                (0.375, 0.857),
                0.25,
                0.034,
                transform=fig.transFigure,
                boxstyle="round,pad=0.004,rounding_size=0.01",
                facecolor="#ffffff",
                edgecolor="#d5dee8",
                linewidth=0.8,
                zorder=-1,
            )
        )

        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=5200,
            metadata={
                "title": f"{load_label} - {notice}",
                "comment": "Saved-trace replay only; no simulation was rerun and no verdict is asserted.",
            },
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        with writer.saving(fig, str(partial_output), dpi):
            for frame_index, time_s in enumerate(time_grid):
                for body, artist in p_artists:
                    vertices = _body_vertices_mm(
                        body, physical_samples, float(time_s)
                    )
                    artist.set_verts(vertices[body["mesh"].faces])
                for body, artist in r_artists:
                    vertices = _body_vertices_mm(
                        body, reduced_samples, float(time_s)
                    )
                    artist.set_verts(vertices[body["mesh"].faces])
                if chain_artist is not None:
                    chain_positions = _moving_chain_marker_positions_mm(
                        physical,
                        physical_samples,
                        float(time_s),
                    )
                    chain_artist._offsets3d = (
                        chain_positions[:, 0],
                        chain_positions[:, 1],
                        chain_positions[:, 2],
                    )

                contact_count = int(
                    _stepped_value(physical_samples, float(time_s), "contact_count", 0)
                    or 0
                )
                physical_badge.set_text(
                    label(f"已保存接触数：{contact_count}", f"Saved contacts: {contact_count}")
                )
                p_phase = _phase_label(
                    _stepped_value(physical_samples, float(time_s), "phase_id", "initial")
                )
                r_phase = _phase_label(
                    _stepped_value(reduced_samples, float(time_s), "phase_id", "initial")
                )
                physical_badge.set_text(
                    label(f"已保存接触数：{contact_count}", f"Saved contacts: {contact_count}")
                )
                _discard = (
                )

                phase_display = (
                    p_phase
                    if p_phase == r_phase
                    else label(
                        f"真实：{p_phase}　{right_label}：{r_phase}",
                        f"{p_phase} vs {r_phase}",
                    )
                )
                phase_text.set_text(
                    label(
                        f"当前阶段：{phase_display}　｜　轨迹 {float(time_s):.3f} 秒",
                        f"Phase: {phase_display} | t = {float(time_s):.3f} s",
                    )
                )

                for cursor in (progress_cursor, lateral_cursor):
                    cursor.set_xdata([float(time_s), float(time_s)])
                p_progress_marker.set_data([time_s], [p_progress[frame_index]])
                r_progress_marker.set_data([time_s], [r_progress[frame_index]])
                phase_display = (
                    p_phase
                    if p_phase == r_phase
                    else label(
                        f"真实：{p_phase}　{right_label}：{r_phase}",
                        f"{p_phase} vs {r_phase}",
                    )
                )
                phase_text.set_text(
                    label(
                        f"当前阶段：{phase_display}　｜　轨迹 {float(time_s):.3f} 秒",
                        f"Phase: {phase_display} | t = {float(time_s):.3f} s",
                    )
                )
                p_lateral_marker.set_data([time_s], [p_lateral[frame_index]])
                r_lateral_marker.set_data([time_s], [r_lateral[frame_index]])
                if is_clock:
                    physical_badge.set_text(
                        label(
                            f"实体碰撞已启用 · 当前接触点 {contact_count}",
                            f"Solid collision enabled · contacts now {contact_count}",
                        )
                    )
                writer.grab_frame()
                frames_written += 1

        if not partial_output.is_file() or partial_output.stat().st_size <= 0:
            raise RuntimeError("ffmpeg produced no MP4 output")
        partial_output.replace(output_path)
        metadata["status"] = "ENCODED"
        metadata["encoded_frames"] = frames_written
        metadata["encoding"].update(
            {
                "status": "ENCODED",
                "encoded_frames": frames_written,
                "playback_seconds": frames_written / fps,
                "output_sha256": _sha256(output_path),
                "output_bytes": output_path.stat().st_size,
            }
        )
        _write_json_atomic(sidecar_path, metadata)
        return metadata
    except Exception as exc:
        if partial_output.exists():
            partial_output.unlink()
        metadata["status"] = "FAILED"
        metadata["encoded_frames"] = frames_written
        metadata["encoding"].update(
            {
                "status": "FAILED",
                "encoded_frames": frames_written,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _write_json_atomic(sidecar_path, metadata)
        raise
    finally:
        if fig is not None:
            plt.close(fig)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从已有真实/简化 trace 生成中文左右同步 MP4；不会重跑仿真。"
    )
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--physical-trace", required=True, type=Path)
    parser.add_argument("--reduced-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--minimum-playback-s", type=float, default=4.0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = render_trace_pair_video(
            workspace_root=args.workspace_root,
            physical_trace=args.physical_trace,
            reduced_trace=args.reduced_trace,
            output=args.output,
            sidecar=args.sidecar,
            fps=args.fps,
            dpi=args.dpi,
            minimum_playback_s=args.minimum_playback_s,
            force=args.force,
        )
    except Exception as exc:
        print(f"轨迹回放编码失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "encoded_frames": result["encoded_frames"],
                "output": result["encoding"]["output"],
                "sidecar": str(_sidecar_path(Path(args.output).resolve(), args.sidecar)),
                "notice_zh": result["notice_zh"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
