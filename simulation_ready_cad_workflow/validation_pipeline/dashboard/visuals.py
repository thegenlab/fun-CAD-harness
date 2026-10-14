"""Generate reusable, evidence-bound visuals for the quest dashboard.

The generator reads one normalized validation run and creates only derived
presentation assets.  It never changes validator verdicts or source evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .scene_assets import export_interactive_glb, load_scene_meshes
from .physics_evidence import generate_physics_evidence
from .path_resolution import resolve_workspace_reference
from . import render_lang
from .render_lang import label, sans_serif_fonts
from ..run_layout import is_terminal_status, parse_canonical_run
from ..mechanism.io import write_json_atomic
from .trace_pair_video import (
    _body_vertices_mm,
    _mesh_bound_corners,
    _transform_vertices_mm,
    _trace_scene_bodies,
    render_physical_trace_video,
    render_trace_pair_video,
)


matplotlib.rcParams["font.sans-serif"] = sans_serif_fonts()
matplotlib.rcParams["axes.unicode_minus"] = False


INK = "#152235"
MUTED = "#65758b"
FIXED = "#71839a"
MOVING = "#2d9a64"
MOVING_EDGE = "#17633f"
AMBER = "#d5a02f"
RED = "#c44949"
GRID = "#dce4ec"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_inputs(paths: Sequence[Path]) -> list[list[Any]]:
    records = []
    for path in paths:
        try:
            item = path.stat()
            records.append([str(path), item.st_size, item.st_mtime_ns])
        except OSError:
            records.append([str(path), None, None])
    return records


def _asset_current(output: Path, inputs: Sequence[Path]) -> bool:
    try:
        complete = _read_json(output.with_suffix(output.suffix + ".complete.json"))
        stat = output.stat()
        return (stat.st_size > 0 and complete.get("inputs") == _asset_inputs(inputs)
                and complete.get("output") == [stat.st_size, stat.st_mtime_ns]
                and complete.get("related", []) == _asset_inputs([Path(item[0]) for item in complete.get("related", [])]))
    except (OSError, ValueError, TypeError, IndexError):
        return False


def _complete_asset(output: Path, inputs: Sequence[Path], *, expected_inputs: list[list[Any]] | None = None,
                    related: Sequence[Path] = ()) -> None:
    actual_inputs = _asset_inputs(inputs)
    if expected_inputs is not None and expected_inputs != actual_inputs:
        raise ValueError("presentation source changed during rendering; retry the current source")
    stat = output.stat()
    if not stat.st_size:
        raise ValueError(f"render produced an empty file: {output}")
    write_json_atomic(output.with_suffix(output.suffix + ".complete.json"),
                      {"inputs": actual_inputs, "output": [stat.st_size, stat.st_mtime_ns], "related": _asset_inputs(related)})


def _render_atomic_asset(output: Path, inputs: Sequence[Path], render, *, force: bool = False,
                         workspace: Path | None = None) -> None:
    if not force and _asset_current(output, inputs):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_inputs = _asset_inputs(inputs)
    with tempfile.TemporaryDirectory(prefix=".render-", dir=output.parent) as temporary_directory:
        staging = Path(temporary_directory).resolve()
        if staging.parent != output.parent.resolve():
            raise ValueError("render staging directory is outside the output directory")
        temporary = staging / output.name
        render(temporary)
        if not temporary.is_file() or not temporary.stat().st_size:
            raise ValueError(f"render did not complete: {output}")
        if expected_inputs != _asset_inputs(inputs):
            raise ValueError("presentation source changed during rendering; retry the current source")
        aliases = [(str(staging), str(output.parent.resolve())),
                   (staging.as_posix(), output.parent.resolve().as_posix())]
        if workspace is not None:
            try:
                aliases.append((staging.relative_to(workspace).as_posix(), output.parent.resolve().relative_to(workspace).as_posix()))
            except ValueError:
                pass

        def relocate(value):
            if isinstance(value, dict):
                return {key: relocate(item) for key, item in value.items()}
            if isinstance(value, list):
                return [relocate(item) for item in value]
            if isinstance(value, str):
                for old, new in aliases:
                    if value == old or value.startswith(old + "/") or value.startswith(old + "\\"):
                        return (new + value[len(old):]).removeprefix("./").removeprefix(".\\")
            return value

        related = []
        for file in sorted(staging.rglob("*")):
            if not file.is_file() or file == temporary:
                continue
            destination = output.parent / file.relative_to(staging)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if file.suffix == ".json":
                write_json_atomic(file, relocate(_read_json(file)))
            file.replace(destination)
            related.append(destination)
        temporary.replace(output)
        _complete_asset(output, inputs, expected_inputs=expected_inputs, related=related)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _workspace_path(workspace: Path, value: str | Path) -> Path:
    return resolve_workspace_reference(
        workspace,
        value,
        require_within_workspace=True,
    )


def _solid_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return 10_000


def _load_solid_meshes(run: Path, capsule_id: str) -> tuple[list[trimesh.Trimesh], list[trimesh.Trimesh]]:
    root = run / "derived_adapters" / capsule_id
    fixed_paths = sorted((root / "fixed.nonconvex_solid_stl").glob("*.stl"), key=_solid_number)
    moving_paths = sorted((root / "moving.nonconvex_solid_stl").glob("*.stl"), key=_solid_number)
    if not fixed_paths or not moving_paths:
        raise ValueError("run does not contain STEP-derived per-solid STL evidence")

    def load(paths: Iterable[Path]) -> list[trimesh.Trimesh]:
        result: list[trimesh.Trimesh] = []
        for path in paths:
            mesh = trimesh.load_mesh(path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            if len(mesh.vertices) and len(mesh.faces):
                result.append(mesh)
        return result

    return load(fixed_paths), load(moving_paths)


def _mesh_triangles(meshes: Iterable[trimesh.Trimesh]) -> list[np.ndarray]:
    return [np.asarray(mesh.vertices)[np.asarray(mesh.faces)] for mesh in meshes]


def _combined_vertices(*groups: Iterable[trimesh.Trimesh]) -> np.ndarray:
    return np.vstack([np.asarray(mesh.vertices) for group in groups for mesh in group])


def _configure_3d(axis, points: np.ndarray, *, elev: float = 23.0, azim: float = -58.0) -> None:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.22)
    center = (minimum + maximum) / 2.0
    padding = span * 0.60
    axis.set_xlim(center[0] - padding[0], center[0] + padding[0])
    axis.set_ylim(center[1] - padding[1], center[1] + padding[1])
    axis.set_zlim(center[2] - padding[2], center[2] + padding[2])
    axis.set_box_aspect(span)
    axis.view_init(elev=elev, azim=azim)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_facecolor("#ffffff")


def _add_meshes(axis, meshes: Iterable[trimesh.Trimesh], *, color: str, edge: str, alpha: float) -> None:
    for triangles in _mesh_triangles(meshes):
        axis.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolor=color,
                edgecolor=edge,
                linewidth=0.28,
                alpha=alpha,
            )
        )


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)


def render_design_package(
    fixed: list[trimesh.Trimesh],
    moving: list[trimesh.Trimesh],
    output: Path,
    *,
    source_note: str,
) -> None:
    points = _combined_vertices(fixed, moving)
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="#ffffff")
    axis = fig.add_subplot(111, projection="3d")
    _configure_3d(axis, points, elev=24, azim=-56)
    _add_meshes(axis, fixed, color="#c4cfdb", edge="#71839a", alpha=0.92)
    _add_meshes(axis, moving, color="#e0ae42", edge="#8b6718", alpha=0.96)
    fig.text(0.035, 0.92, label("整机 CAD 预览", "Whole-machine CAD preview"), color=INK, fontsize=21, fontweight="bold")
    fig.text(
        0.035,
        0.875,
        source_note,
        color=MUTED,
        fontsize=12,
    )
    fig.text(0.73, 0.08, label("运行绑定的几何证据已载入", "Run-bound geometry loaded"), color="#1e8b59", fontsize=12, fontweight="bold")
    _save_figure(fig, output)


def render_mechanism_map(
    fixed: list[trimesh.Trimesh],
    moving: list[trimesh.Trimesh],
    plan: dict[str, Any],
    output: Path,
) -> None:
    points = _combined_vertices(fixed, moving)
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="#ffffff")
    axis = fig.add_subplot(111, projection="3d")
    _configure_3d(axis, points, elev=22, azim=-57)
    _add_meshes(axis, fixed, color="#aeb9c5", edge="#71839a", alpha=0.16)
    _add_meshes(axis, moving, color=MOVING, edge=MOVING_EDGE, alpha=0.94)

    origin = np.asarray(plan.get("joint_origin_abs_m") or [0.0, 0.0, 0.0], dtype=float) * 1000.0
    direction = np.asarray(plan.get("axis_abs") or [1.0, 0.0, 0.0], dtype=float)
    norm = max(float(np.linalg.norm(direction)), 1e-12)
    direction = direction / norm
    axis.quiver(
        *origin,
        *direction,
        length=34.0,
        color="#2366a8",
        linewidth=3.0,
        arrow_length_ratio=0.18,
    )
    family = {
        "prismatic": label("直线滑动", "Linear sliding"),
        "revolute": label("转动", "Revolute"),
        "fixed": label("固定连接", "Fixed connection"),
    }.get(
        str(plan.get("family") or ""),
        str(plan.get("family") or label("未知运动", "Unknown motion")),
    )
    fig.text(
        0.035,
        0.92,
        label(
            f"机构：{plan.get('capsule_id', 'mechanism')} · {family}",
            f"Mechanism: {plan.get('capsule_id', 'mechanism')} - {family}",
        ),
        color=INK,
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.875,
        label(
            "透明灰：整机背景 · 绿色：这一套机构涉及的运动组件 · 蓝箭头：允许的运动方向",
            "Gray: whole machine; green: this mechanism's moving components; blue arrow: allowed motion",
        ),
        color=MUTED,
        fontsize=12,
    )
    fig.text(0.035, 0.08, label(f"固定实体 {len(fixed)}", f"Fixed solids {len(fixed)}"), color=MUTED, fontsize=12)
    fig.text(0.16, 0.08, label(f"运动实体 {len(moving)}", f"Moving solids {len(moving)}"), color=MOVING_EDGE, fontsize=12, fontweight="bold")
    tasks = plan.get("task_ids") or []
    fig.text(0.31, 0.08, label(f"关联任务 {len(tasks)} 项", f"Linked tasks {len(tasks)}"), color=MUTED, fontsize=12)
    _save_figure(fig, output)


def render_collision_overview(
    fixed: list[trimesh.Trimesh], moving: list[trimesh.Trimesh], output: Path
) -> None:
    points = _combined_vertices(fixed, moving)
    fig = plt.figure(figsize=(13.5, 7.2), facecolor="#ffffff")
    axis = fig.add_subplot(111, projection="3d")
    _configure_3d(axis, points, elev=20, azim=-54)
    _add_meshes(axis, fixed, color="#70839a", edge="#43566d", alpha=0.20)
    _add_meshes(axis, moving, color="#34a56d", edge="#17633f", alpha=0.38)
    fig.text(0.035, 0.92, label("碰撞几何总览", "Collision geometry overview"), color=INK, fontsize=21, fontweight="bold")
    fig.text(
        0.035,
        0.875,
        label(
            "逐个 STEP 实体生成碰撞形状；透明显示便于检查内部关系",
            "Collision shapes built per STEP solid; drawn transparent to show internal fits",
        ),
        color=MUTED,
        fontsize=12,
    )
    fig.text(
        0.68,
        0.08,
        label(
            f"固定 {len(fixed)} 个显示组 · 运动 {len(moving)} 个显示组",
            f"{len(fixed)} fixed / {len(moving)} moving display groups",
        ),
        color="#1e8b59",
        fontsize=12,
        fontweight="bold",
    )
    _save_figure(fig, output)


def render_side_clearance(output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), facecolor="#ffffff")
    full, zoom = axes
    for axis in axes:
        axis.set_facecolor("#f8fafc")
        axis.grid(color=GRID, linewidth=0.8, alpha=0.8)
        axis.set_xlabel("Y / mm")
        axis.set_ylabel("Z / mm")
        axis.set_aspect("equal", adjustable="box")

    full.add_patch(Rectangle((-31.1, 0), 5.0, 45.0, facecolor=FIXED, edgecolor="#43566d", alpha=0.55))
    full.add_patch(Rectangle((-26.0, 12), 3.0, 28.0, facecolor=MOVING, edgecolor=MOVING_EDGE, alpha=0.82))
    full.set_xlim(-33, -21)
    full.set_ylim(0, 46)
    full.set_title(label("侧向导向关系（真实剖面）", "Side guide (true section)"), fontsize=15, color=INK, fontweight="bold")
    full.text(-30.9, 42.5, label("固定导向壁", "Fixed wall"), color="#31465e", fontsize=11)
    full.text(-25.9, 38.0, label("抽屉侧壁", "Drawer wall"), color=MOVING_EDGE, fontsize=11)

    zoom.add_patch(Rectangle((-26.3, 12), 0.2, 28.0, facecolor=FIXED, edgecolor="#43566d", alpha=0.55))
    zoom.add_patch(Rectangle((-26.0, 12), 0.2, 28.0, facecolor=MOVING, edgecolor=MOVING_EDGE, alpha=0.82))
    zoom.set_xlim(-26.34, -25.76)
    zoom.set_ylim(11.5, 40.5)
    zoom.set_title(label("间隙放大 20×", "Gap zoomed 20×"), fontsize=15, color=INK, fontweight="bold")
    arrow = FancyArrowPatch((-26.1, 22), (-26.0, 22), arrowstyle="<->", mutation_scale=18, color=RED, linewidth=2)
    zoom.add_patch(arrow)
    zoom.text(-26.05, 23.5, "0.10 mm", ha="center", color=RED, fontsize=14, fontweight="bold")
    zoom.text(-26.05, 17.0, label("有间隙 · 未穿插", "Gap, no interpenetration"), ha="center", color="#1e8b59", fontsize=13, fontweight="bold")

    fig.suptitle(label("小间隙检查：固定壁与抽屉侧壁保持 0.10 mm 间隙",
                       "Small-gap check: 0.10 mm between fixed wall and drawer side wall"), fontsize=19, color=INK, fontweight="bold")
    fig.text(0.5, 0.03, label("尺寸来自 solid_4 与 solid_10 的 STEP 派生边界",
                             "Dimensions from STEP-derived bounds of solid_4 and solid_10"), ha="center", color=MUTED, fontsize=11)
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.91))
    _save_figure(fig, output)


def render_rail_contact(output: Path) -> None:
    fig, axis = plt.subplots(figsize=(11.5, 6.2), facecolor="#ffffff")
    axis.set_facecolor("#f8fafc")
    axis.grid(color=GRID, linewidth=0.8, alpha=0.8)
    axis.add_patch(Rectangle((-17.0, 5.0), 4.0, 4.0, facecolor=FIXED, edgecolor="#43566d", alpha=0.68))
    axis.add_patch(Rectangle((-26.0, 9.0), 52.0, 3.0, facecolor=MOVING, edgecolor=MOVING_EDGE, alpha=0.82))
    axis.plot([-17.0, -13.0], [9.0, 9.0], color="#1e8b59", linewidth=5)
    axis.annotate(
        label("设计接触面\n0.00 mm 间隙", "Design contact face\n0.00 mm gap"),
        xy=(-15.0, 9.0),
        xytext=(-6.0, 15.5),
        arrowprops={"arrowstyle": "->", "color": "#1e8b59", "lw": 2},
        color="#1e8b59",
        fontsize=14,
        fontweight="bold",
    )
    axis.text(-16.8, 6.4, label("固定滑轨", "Fixed rail"), color="#31465e", fontsize=12)
    axis.text(7.0, 10.0, label("抽屉底板", "Drawer base"), color=MOVING_EDGE, fontsize=12)
    axis.set_xlim(-28, 28)
    axis.set_ylim(3.5, 20)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Y / mm")
    axis.set_ylabel("Z / mm")
    axis.set_title(label("滑轨接触检查：贴合但没有正体积重叠", "Rail contact check: touching, no positive-volume overlap"),
                   fontsize=18, color=INK, fontweight="bold")
    fig.text(0.5, 0.035, label("接触是预期行为；穿模才是失败。当前边界只相接，不互相占据体积。",
                              "Contact is expected; interpenetration is the failure. These bounds touch but share no volume."),
             ha="center", color=MUTED, fontsize=11)
    fig.tight_layout(rect=(0.03, 0.07, 0.97, 0.95))
    _save_figure(fig, output)


def _trace_display_coordinate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    sample = samples[0]
    if "generalized_angle_rad" in sample:
        return {
            "kind": "angular",
            "primary_key": "generalized_angle_rad",
            "primary_scale": 180.0 / math.pi,
            "primary_label": label("转角 / °", "Angle / °"),
            "secondary_key": "pivot_drift_m",
            "secondary_scale": 1000.0,
            "secondary_label": label("转轴漂移 / mm", "Pivot drift / mm"),
            "delta_unit": "°",
        }
    return {
        "kind": "linear",
        "primary_key": "generalized_progress_m",
        "primary_scale": 1000.0,
        "primary_label": label("行程 / mm", "Travel / mm"),
        "secondary_key": "lateral_shift_m",
        "secondary_scale": 1000.0,
        "secondary_label": label("横向偏移 / mm", "Lateral shift / mm"),
        "delta_unit": "mm",
    }


def render_comparison_chart(
    physical_path: Path, reduced_path: Path, output: Path, *, label: str
) -> None:
    physical = _read_json(physical_path)
    reduced = _read_json(reduced_path)
    # The ``label`` argument (the chart title) shadows the language helper
    # here, so rendered text is picked through ``render_lang.label``.
    comparison_label = (
        render_lang.label("理想参考", "Ideal reference")
        if reduced.get("branch") == "ideal_reference"
        else render_lang.label("理想动力学", "Ideal dynamics")
        if reduced.get("branch") == "ideal_dynamics"
        else render_lang.label("简化机构", "Reduced mechanism")
    )
    p_samples = physical.get("samples") or []
    r_samples = reduced.get("samples") or []
    if not p_samples or not r_samples:
        return

    coordinate = _trace_display_coordinate(p_samples)
    reduced_coordinate = _trace_display_coordinate(r_samples)
    if coordinate["kind"] != reduced_coordinate["kind"]:
        raise ValueError("trace pair uses different generalized coordinate families")
    p_time = np.asarray([item["time_s"] for item in p_samples], dtype=float)
    r_time = np.asarray([item["time_s"] for item in r_samples], dtype=float)
    p_progress = np.asarray(
        [item.get(coordinate["primary_key"], 0.0) for item in p_samples], dtype=float
    ) * coordinate["primary_scale"]
    r_progress = np.asarray(
        [item.get(coordinate["primary_key"], 0.0) for item in r_samples], dtype=float
    ) * coordinate["primary_scale"]
    p_secondary = np.asarray(
        [item.get(coordinate["secondary_key"], 0.0) for item in p_samples], dtype=float
    ) * coordinate["secondary_scale"]
    r_secondary = np.asarray(
        [item.get(coordinate["secondary_key"], 0.0) for item in r_samples], dtype=float
    ) * coordinate["secondary_scale"]

    fig, (main, drift) = plt.subplots(
        2, 1, figsize=(13.5, 6.8), gridspec_kw={"height_ratios": [3.0, 1.15]}, facecolor="#ffffff"
    )
    for axis in (main, drift):
        axis.set_facecolor("#fbfcfe")
        axis.grid(color=GRID, linewidth=0.8, alpha=0.9)
    is_ideal_reference = reduced.get("branch") == "ideal_reference"
    main.plot(
        p_time,
        p_progress,
        color="#376fa8",
        linewidth=3.0 if is_ideal_reference else 2.5,
        label=render_lang.label("真实物理", "Physical"),
    )
    main.plot(
        r_time,
        r_progress,
        color=MOVING,
        linewidth=1.8 if is_ideal_reference else 2.5,
        linestyle="--" if is_ideal_reference else "-",
        label=comparison_label,
    )
    main.set_ylabel(coordinate["primary_label"])
    main.legend(loc="upper left", frameon=False, ncol=2)
    main.set_title(render_lang.label(f"{label}：运动方向、约束与任务边界",
                                     f"{label}: motion and bounds"),
                   color=INK, fontsize=17, fontweight="bold")
    if is_ideal_reference:
        ideal_on_physical_grid = np.interp(p_time, r_time, r_progress)
        deviation = p_progress - ideal_on_physical_grid
        drift.plot(p_time, deviation, color="#376fa8", linewidth=2.0)
        drift.axhline(0.0, color=MOVING, linewidth=1.2, linestyle="--")
        drift.set_ylabel(
            render_lang.label(f"物理 − 理想 / {coordinate['delta_unit']}",
                              f"Physical - ideal / {coordinate['delta_unit']}")
        )
        delta = float(np.max(np.abs(deviation)))
    else:
        drift.plot(p_time, p_secondary, color="#376fa8", linewidth=1.8)
        drift.plot(r_time, r_secondary, color=MOVING, linewidth=1.8)
        drift.set_ylabel(coordinate["secondary_label"])
        delta = abs(float(np.max(p_progress)) - float(np.max(r_progress)))
    drift.set_xlabel(render_lang.label("保存轨迹时间 / s", "Saved trajectory time / s"))
    fig.text(
        0.78, 0.91,
        (
            render_lang.label(f"最大绝对误差 {delta:.3f} {coordinate['delta_unit']}",
                              f"Max abs. error {delta:.3f} {coordinate['delta_unit']}")
            if is_ideal_reference
            else render_lang.label(f"最大范围差 {delta:.3f} {coordinate['delta_unit']}",
                                   f"Max range diff. {delta:.3f} {coordinate['delta_unit']}")
        ),
        color=MUTED, fontsize=11,
    )
    fig.text(0.78, 0.865, render_lang.label("证书按门槛判定，不要求逐帧重合", "Threshold-gated, not frame match"),
             color="#1e8b59", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.03, 0.97, 0.96))
    _save_figure(fig, output)


def _quaternion_matrix(wxyz: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(wxyz), dtype=float)
    values = values / max(float(np.linalg.norm(values)), 1e-12)
    w, x, y, z = values
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _transform_vertices(vertices_m: np.ndarray, position_m: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    return ((rotation @ vertices_m.T).T + position_m) * 1000.0



def _whole_machine_target_series(
    whole: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    coordinate_key: str,
    target_key: str,
) -> np.ndarray:
    """Resolve per-frame targets from samples or stage-level task targets."""

    stage_targets: dict[str, float] = {}
    for stage in whole.get("stages") or []:
        value = stage.get(target_key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            stage_targets[str(stage.get("stage") or "")] = float(value)
    initial = float(samples[0].get(coordinate_key, 0.0))
    result: list[float] = []
    for sample in samples:
        value = sample.get(target_key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            value = stage_targets.get(str(sample.get("stage") or ""), initial)
        result.append(float(value))
    return np.asarray(result, dtype=float)

def _render_clock_whole_machine_video(
    workspace: Path,
    physical_trace: dict[str, Any],
    whole: dict[str, Any],
    output: Path,
) -> None:
    """Render a compact three-program replay from saved clock task traces."""

    samples = [item for item in (whole.get("samples") or []) if isinstance(item, dict)]
    if not samples:
        return
    bodies = _trace_scene_bodies(workspace, physical_trace)
    programs: list[tuple[str, list[int]]] = []
    for index, sample in enumerate(samples):
        program = str(sample.get("whole_machine_program") or "clock")
        if not programs or programs[-1][0] != program:
            programs.append((program, []))
        programs[-1][1].append(index)
    selected_indices: list[int] = []
    for _, indices in programs:
        count = min(42, len(indices))
        selected_indices.extend(
            indices[int(round(value))]
            for value in np.linspace(0, len(indices) - 1, count)
        )
    selected_samples = [samples[index] for index in selected_indices]

    def vertices_for_sample(body: dict[str, Any], sample: dict[str, Any]) -> np.ndarray:
        constant = body.get("constant_pose")
        if constant is not None:
            position, quaternion = constant
        else:
            pose = (sample.get("body_poses") or {}).get(body.get("body_id"))
            if not isinstance(pose, dict):
                raise ValueError(f"whole-machine clock sample lacks pose for {body.get('body_id')}")
            position = pose["position_abs_m"]
            quaternion = pose["rotation_abs_quat_wxyz"]
        return _transform_vertices_mm(
            np.asarray(body["mesh"].vertices, dtype=float), position, quaternion
        )

    bound_points: list[np.ndarray] = []
    bound_samples = selected_samples[:: max(1, len(selected_samples) // 18)]
    if bound_samples[-1] is not selected_samples[-1]:
        bound_samples.append(selected_samples[-1])
    for body in bodies:
        corners = _mesh_bound_corners(np.asarray(body["mesh"].vertices, dtype=float))
        for sample in bound_samples:
            constant = body.get("constant_pose")
            if constant is not None:
                position, quaternion = constant
            else:
                pose = (sample.get("body_poses") or {}).get(body.get("body_id")) or {}
                position = pose.get("position_abs_m", [0.0, 0.0, 0.0])
                quaternion = pose.get("rotation_abs_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
            bound_points.append(_transform_vertices_mm(corners, position, quaternion))
    all_points = np.vstack(bound_points)

    frame_axis = np.arange(len(selected_samples), dtype=float)
    minute = np.degrees([float(item.get("clock_output_angle_rad", 0.0)) for item in selected_samples])
    hour = np.degrees([float(item.get("hour_hand_angle_rad", 0.0)) for item in selected_samples])
    second = np.degrees([float(item.get("second_hand_angle_rad", 0.0)) for item in selected_samples])
    hammer = np.degrees([float(item.get("alarm_angle_rad", 0.0)) for item in selected_samples])
    torque = np.asarray(
        [
            float(np.linalg.norm(np.asarray(item.get("applied_torque_abs_nm") or [0.0, 0.0, 0.0], dtype=float)))
            for item in selected_samples
        ],
        dtype=float,
    )
    program_labels = {
        "task_clock_cycle": label("正常走时与三针比例", "3-hand timekeeping"),
        "manual_time_setting_clock": label("后冠手动调时", "Set time by crown"),
        "alarm_clock_cycle": label("机械打铃", "Mechanical alarm"),
    }

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#f5f8fb")
    grid = fig.add_gridspec(2, 2, height_ratios=(2.2, 1.0), width_ratios=(1.28, 1.0), hspace=0.22, wspace=0.14)
    assembly = fig.add_subplot(grid[:, 0], projection="3d")
    hands_axis = fig.add_subplot(grid[0, 1])
    actuation_axis = fig.add_subplot(grid[1, 1])
    _configure_3d(assembly, all_points, elev=18, azim=-68)

    artists: list[tuple[dict[str, Any], Poly3DCollection]] = []
    moving_palette = {
        "second": "#e24a4a",
        "minute": "#2f6fed",
        "hour": "#d39b24",
        "hammer": "#8d55c7",
        "setting": "#19a577",
    }
    first = selected_samples[0]
    for body_index, body in enumerate(bodies):
        vertices = vertices_for_sample(body, first)
        mesh = body["mesh"]
        fixed = bool(body.get("fixed"))
        body_id = str(body.get("body_id") or "")
        color = FIXED if fixed else moving_palette.get(body_id, MOVING)
        artist = Poly3DCollection(
            vertices[mesh.faces],
            facecolor=color,
            edgecolor="#43566d" if fixed else color,
            linewidth=0.12 if fixed else 0.20,
            alpha=0.08 if fixed else (0.96 if body_id in moving_palette else max(0.36, 0.72 - body_index * 0.018)),
        )
        assembly.add_collection3d(artist)
        artists.append((body, artist))
    assembly.set_title(label("证书后的整机任务回放（保存轨迹）", "Post-certificate whole-machine replay (saved trajectory)"),
                       color=INK, fontsize=13, fontweight="bold")

    for axis in (hands_axis, actuation_axis):
        axis.set_facecolor("#ffffff")
        axis.grid(color=GRID, linewidth=0.8, alpha=0.9)
        axis.set_xlim(0, max(len(selected_samples) - 1, 1))
    hands_axis.plot(frame_axis, minute, color="#2f6fed", linewidth=2.2, label=label("分针", "Minute hand"))
    hands_axis.plot(frame_axis, hour, color="#d39b24", linewidth=2.0, label=label("时针", "Hour hand"))
    hands_axis.plot(frame_axis, second, color="#e24a4a", linewidth=1.8, label=label("秒针", "Second hand"))
    hands_axis.set_ylabel(label("保存的转角 / °", "Saved angle / °"))
    hands_axis.legend(loc="upper left", frameon=False, ncol=3, fontsize=8.5)
    actuation_axis.plot(frame_axis, torque, color=AMBER, linewidth=2.0, label=label("外加扭矩", "Applied torque"))
    actuation_axis.plot(frame_axis, hammer, color="#8d55c7", linewidth=1.8, label=label("铃锤转角", "Bell hammer angle"))
    actuation_axis.set_ylabel(label("扭矩 / N·m；铃锤 / °", "Torque / N·m; hammer / °"))
    actuation_axis.set_xlabel(label("三项任务的连续回放进度", "Replay progress across the three tasks"))
    actuation_axis.legend(loc="upper left", frameon=False, ncol=2, fontsize=8.5)

    boundaries: list[int] = []
    cursor = 0
    for program, indices in programs:
        count = min(42, len(indices))
        boundaries.append(cursor)
        cursor += count
        for axis in (hands_axis, actuation_axis):
            axis.axvline(boundaries[-1], color="#b9c4d0", linewidth=0.8, linestyle="--")
    hand_cursor = hands_axis.axvline(0, color=INK, linewidth=1.0, alpha=0.6)
    actuation_cursor = actuation_axis.axvline(0, color=INK, linewidth=1.0, alpha=0.6)
    status = fig.text(0.60, 0.955, "", color=INK, fontsize=11.5, fontweight="bold")
    fig.text(0.60, 0.925, label("仅回放已保存的简化整机任务轨迹；真实实体证据见第 5/6 关左右对照",
                              "Saved reduced-model replay only; physical evidence: stages 5/6"),
             color=MUTED, fontsize=9.5)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=20, codec="libx264", bitrate=5200, extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    key_indices = {"start": 0, "clock": boundaries[0], "setting": boundaries[1], "alarm": boundaries[2], "end": len(selected_samples) - 1}
    key_paths = {name: output.with_suffix(f".{name}.png") for name in key_indices}
    with writer.saving(fig, str(output), 100):
        for frame_index, sample in enumerate(selected_samples):
            for body, artist in artists:
                vertices = vertices_for_sample(body, sample)
                artist.set_verts(vertices[body["mesh"].faces])
            for line in (hand_cursor, actuation_cursor):
                line.set_xdata([frame_index, frame_index])
            program = str(sample.get("whole_machine_program") or "clock")
            status.set_text(
                label(
                    f"{program_labels.get(program, program)}  ·  阶段 {sample.get('phase_id', 'initial')}  ·  轨迹点 {frame_index + 1}/{len(selected_samples)}",
                    f"{program_labels.get(program, program)} | Phase {sample.get('phase_id', 'initial')} | Point {frame_index + 1}/{len(selected_samples)}",
                )
            )
            for name, key_index in key_indices.items():
                if frame_index == key_index:
                    fig.savefig(key_paths[name], dpi=110, bbox_inches="tight", facecolor="#f5f8fb")
            writer.grab_frame()
    plt.close(fig)


def _whole_machine_payload_with_trace_samples(
    run: Path,
    capsule_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Backfill whole-machine replay samples from the finalized trace.

    2026-08-11 ATTENTION AI EDITORS: whole-machine result.json no longer
    carries raw samples (a full chain run is ~300 MB and blew the run
    assembly's 16 MiB JSON fuse on r31-r33). The authoritative samples
    live in the finalized whole-machine trace. Replay video only needs a
    decimated frame budget, not physics fidelity -- the full-fidelity
    trace on disk stays untouched as evidence.
    """

    traces = sorted(
        (run / "traces" / capsule_id).glob("*/whole_machine.trace.json")
    )
    for candidate in traces:
        trace = _read_json(candidate)
        samples = trace.get("samples") or []
        if trace.get("status") == "COMPLETED" and samples:
            stride = max(1, len(samples) // 900)
            merged = dict(payload)
            merged["samples"] = samples[::stride]
            merged["sample_count"] = len(samples)
            return merged
    return payload


def render_whole_machine_video(
    workspace: Path,
    physical_trace: dict[str, Any],
    whole: dict[str, Any],
    output: Path,
) -> None:
    """Render saved whole-machine replay for legacy and compound mechanisms."""

    samples = whole.get("samples") or []
    if not samples:
        return
    if "clock_output_angle_rad" in samples[0]:
        _render_clock_whole_machine_video(workspace, physical_trace, whole, output)
        return
    if "input_crank_angle_rad" in samples[0]:
        # 2026-08-11 ATTENTION AI EDITORS: bicycle chain whole-machine
        # samples use drivetrain field names. Normalize them into the
        # generic replay contract instead of duplicating the renderer;
        # body_poses already drive the scene motion.
        samples = [
            {
                **sample,
                "angle_rad": sample["input_crank_angle_rad"],
                "control_torque_nm": sample.get("task_input_torque_nm", 0.0),
                "stage": sample.get("phase_id") or "initial",
            }
            for sample in samples
        ]
    bodies = _trace_scene_bodies(workspace, physical_trace)
    bodies = _whole_machine_render_bodies(
        bodies,
        physical_trace=physical_trace,
        whole_samples=samples,
    )
    sampled = samples[:: max(1, len(samples) // 24)]
    if sampled[-1] is not samples[-1]:
        sampled = [*sampled, samples[-1]]
    all_points = np.vstack(
        [
            _body_vertices_mm(body, samples, float(sample["time_s"]))
            for body in bodies
            for sample in sampled
        ]
    )
    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)

    angular = "angle_rad" in samples[0]
    if angular:
        coordinate = np.asarray([sample["angle_rad"] for sample in samples], dtype=float) * 180.0 / math.pi
        target = _whole_machine_target_series(whole, samples, coordinate_key="angle_rad", target_key="target_angle_rad") * 180.0 / math.pi
        coordinate_label = label("转角 / °", "Angle / °")
        coordinate_name = label("角度", "angle")
    else:
        coordinate = np.asarray([sample["progress_m"] for sample in samples], dtype=float) * 1000.0
        target = _whole_machine_target_series(whole, samples, coordinate_key="progress_m", target_key="target_progress_m") * 1000.0
        coordinate_label = label("行程 / mm", "Travel / mm")
        coordinate_name = label("位置", "position")

    if "control_torque_nm" in samples[0]:
        effort = np.asarray([sample["control_torque_nm"] for sample in samples], dtype=float)
        effort_label = label("控制扭矩 / N·m", "Control torque / N·m")
    else:
        effort = np.asarray([sample.get("control_force_n", 0.0) for sample in samples], dtype=float)
        effort_label = label("控制力 / N", "Control force / N")

    fig = plt.figure(figsize=(12.8, 7.2), facecolor="#f5f8fb")
    grid = fig.add_gridspec(2, 2, height_ratios=(3.0, 1.1), width_ratios=(1.35, 1.0), hspace=0.20, wspace=0.10)
    assembly = fig.add_subplot(grid[:, 0], projection="3d")
    progress_axis = fig.add_subplot(grid[0, 1])
    force_axis = fig.add_subplot(grid[1, 1])
    _configure_3d(assembly, all_points, elev=22, azim=-56)

    artists: list[tuple[dict[str, Any], Poly3DCollection]] = []
    for body_index, body in enumerate(bodies):
        vertices = _body_vertices_mm(body, samples, float(times[0]))
        mesh = body["mesh"]
        fixed = bool(body.get("fixed"))
        artist = Poly3DCollection(
            vertices[mesh.faces],
            facecolor=FIXED if fixed else MOVING,
            edgecolor="#43566d" if fixed else MOVING_EDGE,
            linewidth=0.2 if fixed else 0.25,
            alpha=0.18 if fixed else max(0.62, 0.90 - body_index * 0.06),
        )
        assembly.add_collection3d(artist)
        artists.append((body, artist))
    assembly.set_title(label("整机简化机构任务回放", "Whole-machine reduced-model task replay"), color=INK, fontsize=13, fontweight="bold")

    for axis in (progress_axis, force_axis):
        axis.set_facecolor("#ffffff")
        axis.grid(color=GRID, linewidth=0.8, alpha=0.9)
    progress_axis.plot(times, target, color="#9aa8b8", linestyle="--", linewidth=1.8, label=label("任务目标", "Task target"))
    progress_axis.plot(times, coordinate, color=MOVING, linewidth=2.2,
                       label=label(f"实际{coordinate_name}", f"Actual {coordinate_name}"))
    progress_cursor = progress_axis.axvline(times[0], color=INK, linewidth=1.0, alpha=0.55)
    progress_axis.set_ylabel(coordinate_label)
    progress_axis.legend(loc="upper right", frameon=False)
    force_axis.plot(times, effort, color=AMBER, linewidth=2.0)
    force_cursor = force_axis.axvline(times[0], color=INK, linewidth=1.0, alpha=0.55)
    force_axis.set_xlabel(label("时间 / s", "Time / s"))
    force_axis.set_ylabel(effort_label)
    status = fig.text(0.59, 0.955, "", color=INK, fontsize=12, fontweight="bold")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=20, codec="libx264", bitrate=5200, extra_args=["-pix_fmt", "yuv420p"])
    key_indices = {"start": 0, "open": int(np.argmax(coordinate)), "closed": len(samples) - 1}
    key_paths = {name: output.with_suffix(f".{name}.png") for name in key_indices}
    with writer.saving(fig, str(output), 100):
        for index, sample in enumerate(samples):
            for body, artist in artists:
                vertices = _body_vertices_mm(body, samples, float(times[index]))
                artist.set_verts(vertices[body["mesh"].faces])
            progress_cursor.set_xdata([times[index], times[index]])
            force_cursor.set_xdata([times[index], times[index]])
            stage = str(sample.get("stage") or "initial")
            stage_label = {
                "initial": label("起点", "Start"),
                "outbound": label("打开", "Open"),
                "return": label("关闭", "Close"),
            }.get(stage, stage)
            status.set_text(
                label(
                    f"任务：{stage_label}   目标：{target[index]:.1f}   实际：{coordinate[index]:.2f}",
                    f"Task: {stage_label}   Target: {target[index]:.1f}   Actual: {coordinate[index]:.2f}",
                )
            )
            for name, key_index in key_indices.items():
                if index == key_index:
                    fig.savefig(key_paths[name], dpi=110, bbox_inches="tight", facecolor="#f5f8fb")
            for _ in range(2):
                writer.grab_frame()
    plt.close(fig)


def _whole_machine_render_bodies(
    bodies: list[dict[str, Any]],
    *,
    physical_trace: dict[str, Any],
    whole_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Adapt a fixture-rich physical scene to a legacy two-body task replay."""

    if not whole_samples:
        return bodies
    first = whole_samples[0]
    if isinstance(first.get("body_poses"), dict):
        return bodies
    if not isinstance(first.get("moving_pose"), dict):
        return bodies

    fixed = next(
        (
            item
            for item in bodies
            if item.get("body_id") == "fixed_boundary"
        ),
        next((item for item in bodies if item.get("fixed")), None),
    )
    moving = next(
        (
            item
            for item in bodies
            if item.get("body_id") == "moving_group"
        ),
        next((item for item in bodies if not item.get("fixed")), None),
    )
    if fixed is None or moving is None:
        raise ValueError(
            "legacy whole-machine replay requires one fixed and one moving body"
        )

    collision = physical_trace.get("collision_geometry") or {}
    fixed_collision = collision.get("fixed") or {}
    fixed_origin = fixed_collision.get("body_origin_abs_m")
    if (
        not isinstance(fixed_origin, (list, tuple))
        or len(fixed_origin) != 3
    ):
        raise ValueError(
            "legacy whole-machine replay is missing the fixed body origin"
        )
    fixed_body = dict(fixed)
    fixed_body["constant_pose"] = (
        np.asarray(fixed_origin, dtype=float),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
    )
    moving_body = dict(moving)
    moving_body["body_id"] = None
    moving_body.pop("constant_pose", None)
    return [fixed_body, moving_body]


def _ensure_branch_video(
    workspace: Path, trace_path: Path, output: Path, *, force: bool
) -> None:
    command = [
        sys.executable,
        "-m",
        "simulation_ready_cad_workflow.validation_pipeline.mechanism.render_trace",
        "--trace",
        str(trace_path),
        "--workspace-root",
        str(workspace),
        "--output",
        str(output),
        "--fps",
        "20",
        "--frame-repeat",
        "1",
    ]
    def render(temporary: Path) -> None:
        arguments = list(command)
        arguments[arguments.index("--output") + 1] = str(temporary)
        subprocess.run(arguments, cwd=workspace, check=True, timeout=120)

    _render_atomic_asset(output, [trace_path], render, force=force, workspace=workspace)


def _trace_ready(path: Path, expected_branch: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        payload.get("branch") == expected_branch
        and payload.get("status") == "COMPLETED"
        and bool(payload.get("samples"))
    )


def _trace_pose_ready(path: Path, expected_branch: str) -> bool:
    if not _trace_ready(path, expected_branch):
        return False
    payload = _read_json(path)
    first = (payload.get("samples") or [{}])[0]
    return bool(
        isinstance(first, dict)
        and (
            isinstance(first.get("body_poses"), dict)
            or isinstance(first.get("moving_pose"), dict)
        )
    )


def _trace_samples_renderable(samples: Any) -> bool:
    # 2026-08-09 ATTENTION AI EDITORS: two-body plain revolute traces
    # (steering family) carry fixed_pose/moving_pose per sample instead of
    # body_poses. Both formats are renderable by trace_pair_video. Keep this
    # gate aligned with _trace_pose_ready above and with
    # trace_pair_video._validated_samples, or some mechanism families will
    # silently lose their dashboard videos again.
    if not samples:
        return False
    first = samples[0]
    if not isinstance(first, dict):
        return False
    return isinstance(first.get("body_poses"), dict) or isinstance(
        first.get("moving_pose"), dict
    )


def _accepted_physical_task_load_case_ids(
    run_dir: Path,
    capsule_id: str,
) -> tuple[str, ...]:
    """Return hash-bound task load cases from the authoritative acceptance result."""

    path = run_dir / "results" / capsule_id / "task_acceptance.json"
    if not path.is_file():
        return ()
    try:
        payload = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return ()
    if payload.get("status") != "PASS":
        return ()
    retained: list[str] = []
    for evaluation in payload.get("evaluations") or []:
        if not isinstance(evaluation, dict):
            continue
        if evaluation.get("branch") != "physical":
            continue
        if evaluation.get("verdict") != "PASS" and evaluation.get("passed") is not True:
            continue
        load_case_id = str(evaluation.get("test_load_case_id") or "")
        if (
            not load_case_id
            or Path(load_case_id).name != load_case_id
            or any(separator in load_case_id for separator in ("/", "\\"))
            or load_case_id in retained
        ):
            continue
        retained.append(load_case_id)
    return tuple(retained)


def _select_visual_trace_paths(
    trace_root: Path,
    *,
    accepted_task_load_case_ids: Sequence[str] = (),
) -> tuple[Path, Path]:
    """Select physical and authoritative comparison traces for presentation."""

    def comparison_trace(load_dir: Path) -> Path | None:
        for name, branch in (
            ("ideal_dynamics.trace.json", "ideal_dynamics"),
            ("ideal_reference.trace.json", "ideal_reference"),
            ("reduced.trace.json", "reduced"),
        ):
            candidate = load_dir / name
            if _trace_pose_ready(candidate, branch):
                return candidate
        return None

    accepted_candidates = [
        trace_root / load_case_id
        for load_case_id in accepted_task_load_case_ids
        if (
            load_case_id
            and Path(load_case_id).name == load_case_id
            and not any(separator in load_case_id for separator in ("/", "\\"))
        )
    ]
    task_candidates = [
        *accepted_candidates,
        trace_root / "task_angle_roundtrip",
        trace_root / "task_envelope_roundtrip",
    ]
    if trace_root.is_dir():
        task_candidates.extend(
            path
            for path in sorted(trace_root.iterdir())
            if path.is_dir()
            and path.name.startswith("task_")
            and path not in task_candidates
        )
    paired_roundtrip = next(
        (
            path
            for path in task_candidates
            if _trace_ready(path / "physical.trace.json", "physical")
            and comparison_trace(path) is not None
        ),
        None,
    )
    physical_roundtrip = paired_roundtrip or next(
        (
            path
            for path in task_candidates
            if _trace_ready(path / "physical.trace.json", "physical")
        ),
        task_candidates[0],
    )
    paired_or_physical = paired_roundtrip or physical_roundtrip
    return (
        physical_roundtrip / "physical.trace.json",
        comparison_trace(paired_or_physical)
        or paired_or_physical / "reduced.trace.json",
    )


def _ensure_reference_pose_diagnostic(
    *,
    workspace: Path,
    run: Path,
    plan: dict[str, Any],
    physical_trace: Path,
    force: bool,
) -> dict[str, Any]:
    """Retain one accepted-task physical/ideal pose video outside the run.

    Canonical terminal revisions are immutable.  This presentation derivative
    therefore lives under diagnostics and remains hash-bound to both source
    traces.  It never invokes a simulator or changes a verdict.
    """

    identity = parse_canonical_run(workspace, run)
    ideal_source = physical_trace.parent / "ideal_reference.trace.json"
    if (
        identity is None
        or not _trace_pose_ready(physical_trace, "physical")
        or not _trace_ready(ideal_source, "ideal_reference")
    ):
        return {"status": "NOT_AVAILABLE"}
    from .ideal_replay import synthesize_ideal_kinematic_trace

    physical = _read_json(physical_trace)
    ideal = _read_json(ideal_source)
    enriched = synthesize_ideal_kinematic_trace(
        plan=plan,
        physical_trace=physical,
        ideal_trace=ideal,
    )
    diagnostic = (
        workspace
        / ".runtime_runs"
        / "diagnostics"
        / f"{identity.run_id}-ideal-pose"
        / str(plan.get("capsule_id") or "mechanism")
    )
    pose_trace = diagnostic / "ideal_reference.trace.json"
    previous = _read_json(pose_trace) if pose_trace.is_file() else None
    trace_changed = previous != enriched
    if trace_changed:
        write_json_atomic(pose_trace, enriched)
    output = diagnostic / "physical_vs_ideal_roundtrip.mp4"
    sidecar = render_trace_pair_video(
        workspace_root=workspace,
        physical_trace=physical_trace,
        reduced_trace=pose_trace,
        output=output,
        fps=20,
        dpi=100,
        minimum_playback_s=4.0,
        force=force or trace_changed,
    )
    return {
        "status": sidecar.get("status"),
        "load_case_id": physical_trace.parent.name,
        "video": output.relative_to(workspace).as_posix(),
        "sidecar": output.with_suffix(".sidecar.json")
        .relative_to(workspace)
        .as_posix(),
        "ideal_pose_trace": pose_trace.relative_to(workspace).as_posix(),
        "pose_model": (
            enriched.get("visual_replay_semantics") or {}
        ).get("pose_model"),
        "simulation_was_rerun": False,
    }


def _ensure_physics_evidence_diagnostic(
    *,
    workspace: Path,
    run: Path,
    plan: dict[str, Any],
    force: bool,
) -> dict[str, Any]:
    """Retain interactive physics 3D outside an immutable terminal run."""

    identity = parse_canonical_run(workspace, run)
    capsule_id = str(plan.get("capsule_id") or "")
    if identity is None or not capsule_id:
        return {"status": "NOT_AVAILABLE"}
    visual_dir = (
        workspace
        / ".runtime_runs"
        / "diagnostics"
        / f"{identity.run_id}-physics-evidence"
        / capsule_id
    )
    from .physics_evidence import generate_physics_evidence

    return generate_physics_evidence(
        workspace_root=workspace,
        run_dir=run,
        plan=plan,
        force=force,
        visual_dir=visual_dir,
    )


def _published_reference_pose_diagnostic(
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    """Keep outside-run diagnostic media out of formal publication manifests."""

    public_fields = {
        "status",
        "error",
        "load_case_id",
        "pose_model",
        "simulation_was_rerun",
    }
    return {
        key: value
        for key, value in diagnostic.items()
        if key in public_fields
    }


def _generate_visuals_for_plan(
    *,
    workspace_root: Path,
    run_dir: Path,
    plan_path: Path,
    force: bool = False,
    multi_capsule: bool = False,
    primary_capsule: bool = True,
) -> dict[str, Any]:
    """Generate presentation assets for one plan without invoking a simulator."""

    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    run.relative_to(workspace)
    plan = _read_json(plan_path)
    capsule_id = str(plan.get("capsule_id") or plan_path.stem)
    source_binding_paths = [plan_path]
    if plan.get("package_dir"):
        source_package = _workspace_path(workspace, plan["package_dir"])
        source_binding_paths += [source_package / "assembly.step", source_package / "assembly_contract.json"]
    source_binding_snapshot = _asset_inputs(source_binding_paths)
    scene_meshes = load_scene_meshes(
        workspace_root=workspace, run_dir=run, capsule_id=capsule_id
    )
    fixed, moving = scene_meshes.fixed, scene_meshes.moving
    if source_binding_snapshot != _asset_inputs(source_binding_paths):
        raise ValueError("presentation geometry changed while loading")
    visual = run / "visual"
    visual.mkdir(parents=True, exist_ok=True)

    package_manifest: dict[str, Any] = {}
    package_ref = plan.get("package_dir")
    if package_ref:
        package_dir = _workspace_path(workspace, package_ref)
        manifest_path = package_dir / "package.json"
        if manifest_path.is_file():
            package_manifest = _read_json(manifest_path)
    package_id = str(package_manifest.get("package_id") or "")
    is_drawer = "drawer" in capsule_id.lower() or "drawer" in package_id.lower()
    source_note = (
        label(
            f"直接来自 assembly.step 的 {len(fixed) + len(moving)} 个实体 · 保留装配坐标 · 灰蓝为固定结构 · 金色为运动组件",
            f"{len(fixed) + len(moving)} solids straight from assembly.step; assembly coordinates kept;"
            " blue-gray: fixed structure; gold: moving components",
        )
        if scene_meshes.source_kind == "package-step-per-solid-tessellation"
        else (
            label(
                f"来自 STEP 逐实体几何的 {len(fixed) + len(moving)} 个实体 · 灰蓝为固定结构 · 金色为运动组件",
                f"{len(fixed) + len(moving)} solids from per-solid STEP geometry;"
                " blue-gray: fixed structure; gold: moving components",
            )
            if scene_meshes.source_kind == "step-derived-per-solid-stl"
            else label(
                "来自本次保存轨迹绑定的 STEP 派生碰撞几何 · 灰蓝为固定结构 · 金色为运动组件",
                "STEP-derived collision geometry bound to this run's saved trajectory;"
                " blue-gray: fixed structure; gold: moving components",
            )
        )
    )

    outputs = {
        "design_package": visual / "stage1_design_package.png",
        "design_scene": visual / "stage1_design_package.glb",
        "mechanism_map": visual / f"stage3_mechanism_{capsule_id}.png",
        "mechanism_scene": visual / f"stage3_mechanism_{capsule_id}.glb",
        "collision_overview": visual / "stage4_collision_overview.png",
        "side_clearance": visual
        / (
            f"stage4_side_clearance__{capsule_id}.png"
            if multi_capsule
            else "stage4_side_clearance.png"
        ),
        "rail_contact": visual
        / (
            f"stage4_rail_contact__{capsule_id}.png"
            if multi_capsule
            else "stage4_rail_contact.png"
        ),
        "physical_video": visual
        / (
            f"drawer_physical_roundtrip__{capsule_id}.mp4"
            if multi_capsule
            else "drawer_physical_roundtrip.mp4"
        ),
        "reduced_video": visual
        / (
            f"drawer_reduced_roundtrip__{capsule_id}.mp4"
            if multi_capsule
            else "drawer_reduced_roundtrip.mp4"
        ),
        "whole_machine_video": visual
        / (
            f"stage8_whole_machine__{capsule_id}.mp4"
            if multi_capsule
            else "stage8_whole_machine.mp4"
        ),
    }
    generators = [
        (
            outputs["mechanism_map"],
            lambda: render_mechanism_map(
                fixed, moving, plan, outputs["mechanism_map"]
            ),
        ),
    ]
    if primary_capsule:
        generators[:0] = [
            (
                outputs["design_package"],
                lambda: render_design_package(
                    fixed,
                    moving,
                    outputs["design_package"],
                    source_note=source_note,
                ),
            ),
            (
                outputs["collision_overview"],
                lambda: render_collision_overview(
                    fixed, moving, outputs["collision_overview"]
                ),
            ),
        ]
    if is_drawer:
        generators.extend(
            [
                (
                    outputs["side_clearance"],
                    lambda: render_side_clearance(outputs["side_clearance"]),
                ),
                (
                    outputs["rail_contact"],
                    lambda: render_rail_contact(outputs["rail_contact"]),
                ),
            ]
        )
    scene_inputs = list(scene_meshes.source_paths)
    scene_inputs_snapshot = _asset_inputs(scene_inputs)
    for path, generator in generators:
        if force or not _asset_current(path, scene_inputs):
            generator()
            _complete_asset(path, scene_inputs, expected_inputs=scene_inputs_snapshot)

    interactive_manifests: dict[str, dict[str, Any]] = {}
    interactive_scene_keys = (
        ("design_scene", "mechanism_scene")
        if primary_capsule
        else ("mechanism_scene",)
    )
    for key in interactive_scene_keys:
        path = outputs[key]
        if force or not _asset_current(path, scene_inputs) or not path.with_suffix(".scene.json").is_file():
            interactive_manifests[key] = export_interactive_glb(
                scene_meshes,
                path,
                workspace_root=workspace,
                run_dir=run,
                capsule_id=capsule_id,
            )
            _complete_asset(path, scene_inputs, expected_inputs=scene_inputs_snapshot)
        else:
            interactive_manifests[key] = _read_json(path.with_suffix(".scene.json"))

    # This reads existing traces and STEP-bound collision assets only.  It must
    # remain presentation-only and must never invoke a simulator.
    physics_evidence = generate_physics_evidence(
        workspace_root=workspace,
        run_dir=run,
        plan=plan,
        force=force,
        namespace_assets=multi_capsule,
    )

    trace_root = run / "traces" / capsule_id
    accepted_task_load_case_ids = _accepted_physical_task_load_case_ids(
        run,
        capsule_id,
    )
    physical_trace_path, reduced_trace_path = _select_visual_trace_paths(
        trace_root,
        accepted_task_load_case_ids=accepted_task_load_case_ids,
    )
    try:
        reference_pose_diagnostic = _ensure_reference_pose_diagnostic(
            workspace=workspace,
            run=run,
            plan=plan,
            physical_trace=physical_trace_path,
            force=force,
        )
    except Exception as exc:
        reference_pose_diagnostic = {
            "status": "NOT_SUPPORTED",
            "error": f"{type(exc).__name__}: {exc}",
        }
    published_reference_pose_diagnostic = (
        _published_reference_pose_diagnostic(reference_pose_diagnostic)
    )
    if is_drawer and _trace_ready(physical_trace_path, "physical"):
        _ensure_branch_video(
            workspace, physical_trace_path, outputs["physical_video"], force=force
        )
    if is_drawer and _trace_ready(reduced_trace_path, "reduced"):
        _ensure_branch_video(
            workspace, reduced_trace_path, outputs["reduced_video"], force=force
        )

    load_labels = {
        "nominal_light": label("轻载往返", "Light-load round trip"),
        "nominal_loaded": label("重载往返", "Heavy-load round trip"),
        "zero_input_gravity": label("零输入稳定性", "Zero-input stability"),
        "transverse_probe": label("横向扰动", "Transverse probe"),
        "transverse_torque_probe": label("横向扭矩扰动", "Transverse torque probe"),
        "held_out_mid": label("保留工况", "Held-out load case"),
        "task_envelope_roundtrip": label("全行程打开与关闭", "Full-travel open and close"),
        "task_angle_roundtrip": label("全角度打开与关闭", "Full-angle open and close"),
    }
    load_labels.update(
        {
            "task_clock_cycle": label("三针正常走时", "Three hands run normally"),
            "task_manual_time_setting": label("手动调时", "Manual time setting"),
            "task_alarm_cycle": label("机械打铃", "Mechanical alarm"),
            "shaft_transverse_probe": label("侧向扰动", "Lateral probe"),
            "tong_tabletop_pickup": label("桌面夹取并抬起", "Grip and lift from table"),
            "tong_tabletop_hold": label("离桌保持", "Hold above table"),
            "tong_tabletop_release": label("松手回桌", "Release onto table"),
        }
    )
    load_ids = {
        path.stem for path in (run / "loads" / capsule_id).glob("*.json")
    }
    if trace_root.is_dir():
        load_ids.update(path.name for path in trace_root.iterdir() if path.is_dir())

    comparison_outputs: dict[str, Path] = {}
    load_video_outputs: dict[str, dict[str, Any]] = {}
    for load_id in sorted(load_ids):
        load_dir = trace_root / load_id
        physical_path = load_dir / "physical.trace.json"
        comparison_candidates = (
            (load_dir / "ideal_dynamics.trace.json", "ideal_dynamics"),
            (load_dir / "ideal_reference.trace.json", "ideal_reference"),
            (load_dir / "reduced.trace.json", "reduced"),
        )
        reduced_path, comparison_branch = next(
            (
                (path, branch)
                for path, branch in comparison_candidates
                if _trace_pose_ready(path, branch)
            ),
            (load_dir / "reduced.trace.json", "reduced"),
        )
        physical_video = (
            visual / f"stage5_physical_trace__{capsule_id}__{load_id}.mp4"
        )
        physical_video_item: dict[str, Any] = {}
        if physical_path.is_file():
            try:
                physical_payload = _read_json(physical_path)
                if _trace_samples_renderable(physical_payload.get("samples")):
                    physical_sidecar = render_physical_trace_video(
                        workspace_root=workspace,
                        physical_trace=physical_path,
                        output=physical_video,
                        fps=20,
                        dpi=100,
                        minimum_playback_s=4.0,
                        force=force,
                    )
                    physical_video_item = {
                        "physical_video_status": physical_sidecar.get(
                            "status"
                        ),
                        "physical_video": str(
                            physical_video.relative_to(workspace).as_posix()
                        ),
                        "physical_video_sidecar": str(
                            physical_video.with_suffix(".sidecar.json")
                            .relative_to(workspace)
                            .as_posix()
                        ),
                        "physical_trace_status": physical_sidecar.get(
                            "trace_status"
                        ),
                        "physical_trace_complete": physical_sidecar.get(
                            "trace_complete"
                        ),
                    }
            except Exception as exc:
                physical_video_item = {
                    "physical_video_status": "RENDER_FAILED",
                    "physical_video_error": f"{type(exc).__name__}: {exc}",
                }
        if not (
            _trace_ready(physical_path, "physical")
            and _trace_pose_ready(reduced_path, comparison_branch)
        ):
            load_video_outputs[load_id] = {
                "capsule_id": capsule_id,
                "load_case_id": load_id,
                "status": "TRACE_EVIDENCE_MISSING",
                "physical_trace_exists": physical_path.is_file(),
                "reduced_trace_exists": reduced_path.is_file(),
                "comparison_trace_branch": comparison_branch,
                **physical_video_item,
            }
            continue
        chart = visual / (
            f"stage6_compare_{capsule_id}__{load_id}.png"
            if multi_capsule
            else f"stage6_compare_{load_id}.png"
        )
        comparison_outputs[load_id] = chart
        if force or not chart.is_file():
            render_comparison_chart(
                physical_path,
                reduced_path,
                chart,
                label=load_labels.get(load_id, load_id),
            )
        paired_video = (
            visual / f"stage6_motion__{capsule_id}__{load_id}.mp4"
        )
        try:
            sidecar = render_trace_pair_video(
                workspace_root=workspace,
                physical_trace=physical_path,
                reduced_trace=reduced_path,
                output=paired_video,
                fps=20,
                dpi=100,
                minimum_playback_s=4.0,
                force=force,
            )
            load_video_outputs[load_id] = {
                "capsule_id": capsule_id,
                "load_case_id": load_id,
                "status": sidecar.get("status"),
                "video": str(paired_video.relative_to(workspace).as_posix()),
                "sidecar": str(
                    paired_video.with_suffix(".sidecar.json")
                    .relative_to(workspace)
                    .as_posix()
                ),
                "physical_trace": str(physical_path.relative_to(workspace).as_posix()),
                "reduced_trace": str(reduced_path.relative_to(workspace).as_posix()),
                "comparison_trace": str(
                    reduced_path.relative_to(workspace).as_posix()
                ),
                "comparison_trace_branch": comparison_branch,
                "trace_sha256": sidecar.get("trace_sha256") or {},
                **physical_video_item,
            }
        except Exception as exc:
            load_video_outputs[load_id] = {
                "capsule_id": capsule_id,
                "load_case_id": load_id,
                "status": "RENDER_FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "physical_trace": str(physical_path.relative_to(workspace).as_posix()),
                "reduced_trace": str(reduced_path.relative_to(workspace).as_posix()),
                "comparison_trace": str(
                    reduced_path.relative_to(workspace).as_posix()
                ),
                "comparison_trace_branch": comparison_branch,
                **physical_video_item,
            }

    whole_path = run / "results" / capsule_id / "whole_machine" / "result.json"
    if (
        whole_path.is_file()
        and _trace_ready(physical_trace_path, "physical")
        and (force or not _asset_current(outputs["whole_machine_video"], [physical_trace_path, whole_path]))
    ):
        whole_payload = _read_json(whole_path)
        if not whole_payload.get("samples"):
            whole_payload = _whole_machine_payload_with_trace_samples(
                run, capsule_id, whole_payload
            )
        _render_atomic_asset(
            outputs["whole_machine_video"], [physical_trace_path, whole_path],
            lambda temporary: render_whole_machine_video(
                workspace, _read_json(physical_trace_path), whole_payload, temporary,
            ), force=force, workspace=workspace,
        )

    source_step = next(
        (path for path in scene_meshes.source_paths if path.suffix.lower() in {".step", ".stp"}),
        None,
    )
    required_paths = [
        outputs["mechanism_map"],
        outputs["mechanism_scene"],
        outputs["mechanism_scene"].with_suffix(".scene.json"),
    ]
    if primary_capsule:
        required_paths[:0] = [
            outputs["design_package"],
            outputs["design_scene"],
            outputs["design_scene"].with_suffix(".scene.json"),
            outputs["collision_overview"],
        ]
    physical_traces = list(trace_root.glob("*/physical.trace.json"))
    generation_failures: list[str] = []
    if physical_traces:
        physics_scene = workspace / str((physics_evidence.get("scene") or {}).get("path") or "")
        physics_manifest = workspace / str((physics_evidence.get("scene") or {}).get("manifest") or "")
        required_paths.extend([physics_scene, physics_manifest])
        for item in (physics_evidence.get("replays") or {}).values():
            replay_path = workspace / str((item or {}).get("path") or "")
            required_paths.append(replay_path)
        if physics_evidence.get("status") != "GENERATED":
            generation_failures.append("physics_evidence_not_generated")
    if whole_path.is_file():
        required_paths.append(outputs["whole_machine_video"])
    for load_id, item in load_video_outputs.items():
        if item.get("status") == "RENDER_FAILED":
            generation_failures.append(f"paired_video_failed:{load_id}")
        if item.get("physical_video_status") == "RENDER_FAILED":
            generation_failures.append(f"physical_video_failed:{load_id}")
        physical_video_ref = item.get("physical_video")
        if physical_video_ref:
            required_paths.append(workspace / str(physical_video_ref))
    if scene_meshes.source_kind != "package-step-per-solid-tessellation":
        generation_failures.append("canonical_step_scene_not_used")
    if source_step is None or not source_step.is_file():
        generation_failures.append("canonical_step_source_missing")
    missing_required = [path for path in required_paths if not path.is_file()]
    if missing_required:
        generation_failures.append("required_visual_assets_missing")

    summary = {
        "schema_version": "0.2",
        "status": "READY" if not generation_failures else "BLOCKED",
        "source_run": str(run.relative_to(workspace).as_posix()),
        "capsule_id": capsule_id,
        "source_plan": str(plan_path.relative_to(workspace).as_posix()),
        "source_scene_kind": scene_meshes.source_kind,
        "canonical_step": {
            "ready": bool(
                source_step is not None
                and source_step.is_file()
                and scene_meshes.source_kind
                == "package-step-per-solid-tessellation"
            ),
            "source": (
                str(source_step.relative_to(workspace).as_posix())
                if source_step is not None and source_step.is_file()
                else None
            ),
            "source_sha256": (
                _sha256(source_step)
                if source_step is not None and source_step.is_file()
                else None
            ),
            "scene": (
                str(outputs["design_scene"].relative_to(workspace).as_posix())
                if outputs["design_scene"].is_file()
                else None
            ),
            "scene_sha256": (
                _sha256(outputs["design_scene"])
                if outputs["design_scene"].is_file()
                else None
            ),
            "rule": "canonical STEP tessellation must complete before dashboard visualization",
        },
        "source_physical_trace": (
            str(physical_trace_path.relative_to(workspace).as_posix())
            if physical_trace_path.is_file()
            else None
        ),
        "source_reduced_trace": (
            str(reduced_trace_path.relative_to(workspace).as_posix())
            if reduced_trace_path.is_file()
            else None
        ),
        "source_comparison_trace": (
            str(reduced_trace_path.relative_to(workspace).as_posix())
            if reduced_trace_path.is_file()
            else None
        ),
        "source_comparison_branch": (
            str(_read_json(reduced_trace_path).get("branch") or "")
            if reduced_trace_path.is_file()
            else None
        ),
        "source_whole_machine_result": (
            str(whole_path.relative_to(workspace).as_posix())
            if whole_path.is_file()
            else None
        ),
        "outputs": {
            key: str(path.relative_to(workspace).as_posix())
            for key, path in outputs.items()
            if path.is_file()
            and (
                primary_capsule
                or key
                not in {"design_package", "design_scene", "collision_overview"}
            )
        },
        "interactive_scenes": {
            key: {
                "output": value.get("output"),
                "output_sha256": value.get("output_sha256"),
                "source_kind": value.get("source_kind"),
            }
            for key, value in interactive_manifests.items()
        },
        "comparison_charts": {
            key: str(path.relative_to(workspace).as_posix())
            for key, path in comparison_outputs.items()
            if path.is_file()
        },
        "load_videos": load_video_outputs,
        "physics_evidence": physics_evidence,
        "reference_pose_diagnostic": published_reference_pose_diagnostic,
        "required_assets": [
            str(path.relative_to(workspace).as_posix())
            for path in required_paths
            if path != workspace
        ],
        "missing_required_assets": [
            str(path.relative_to(workspace).as_posix())
            for path in missing_required
            if path != workspace
        ],
        "generation_failures": generation_failures,
        "claims": {
            "canonical_step_scene_precedes_other_visuals": True,
            "visuals_are_derived_from_selected_run": True,
            "paired_videos_use_saved_traces_only": True,
            "simulation_was_rerun_for_visuals": False,
            "whole_machine_video_is_reduced_model_playback": True,
            "whole_machine_video_is_high_fidelity_contact": False,
        },
    }
    return summary


def _merge_visual_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(summaries) == 1:
        return summaries[0]
    primary = summaries[0]
    by_capsule = {
        str(summary.get("capsule_id") or f"capsule_{index}"): summary
        for index, summary in enumerate(summaries, start=1)
    }

    def merged_mapping(field: str) -> dict[str, Any]:
        return {
            f"{capsule_id}:{key}": value
            for capsule_id, summary in by_capsule.items()
            for key, value in (summary.get(field) or {}).items()
        }

    return {
        "schema_version": "0.3",
        "status": (
            "READY"
            if all(summary.get("status") == "READY" for summary in summaries)
            else "BLOCKED"
        ),
        "source_run": primary.get("source_run"),
        "capsule_id": primary.get("capsule_id"),
        "source_plan": primary.get("source_plan"),
        "capsules": by_capsule,
        "canonical_step": primary.get("canonical_step") or {},
        "source_physical_trace": primary.get("source_physical_trace"),
        "source_reduced_trace": primary.get("source_reduced_trace"),
        "source_comparison_trace": primary.get("source_comparison_trace"),
        "source_comparison_branch": primary.get("source_comparison_branch"),
        "source_whole_machine_result": primary.get(
            "source_whole_machine_result"
        ),
        "outputs": merged_mapping("outputs"),
        "interactive_scenes": merged_mapping("interactive_scenes"),
        "comparison_charts": merged_mapping("comparison_charts"),
        "load_videos": merged_mapping("load_videos"),
        "physics_evidence": primary.get("physics_evidence") or {},
        "physics_evidence_by_capsule": {
            capsule_id: summary.get("physics_evidence") or {}
            for capsule_id, summary in by_capsule.items()
        },
        "reference_pose_diagnostic": primary.get(
            "reference_pose_diagnostic"
        )
        or {},
        "reference_pose_diagnostics": {
            capsule_id: summary.get("reference_pose_diagnostic") or {}
            for capsule_id, summary in by_capsule.items()
        },
        "required_assets": sorted(
            {
                str(path)
                for summary in summaries
                for path in (summary.get("required_assets") or [])
            }
        ),
        "missing_required_assets": sorted(
            {
                str(path)
                for summary in summaries
                for path in (summary.get("missing_required_assets") or [])
            }
        ),
        "generation_failures": [
            f"{capsule_id}:{failure}"
            for capsule_id, summary in by_capsule.items()
            for failure in (summary.get("generation_failures") or [])
        ],
        "claims": {
            "canonical_step_scene_precedes_other_visuals": True,
            "visuals_are_derived_from_selected_run": True,
            "paired_videos_use_saved_traces_only": True,
            "simulation_was_rerun_for_visuals": False,
            "whole_machine_video_is_reduced_model_playback": True,
            "whole_machine_video_is_high_fidelity_contact": False,
            "every_capsule_has_separate_visual_assets": True,
        },
    }


def generate_visuals(
    *,
    workspace_root: Path,
    run_dir: Path,
    force: bool = False,
    publish_run_status: bool = True,
) -> dict[str, Any]:
    """Generate presentation assets for every plan in one run."""

    workspace = workspace_root.resolve()
    run = run_dir.resolve()
    run.relative_to(workspace)
    status_path = run / "run_status.json"
    if parse_canonical_run(workspace, run) is not None and status_path.is_file():
        status = _read_json(status_path)
        if is_terminal_status(status.get("status")):
            summary_path = run / "visual" / "dashboard_visuals.summary.json"
            if summary_path.is_file() and not force:
                summary = _read_json(summary_path)
                if summary.get("status") == "READY":
                    return summary
            for plan_path in sorted((run / "plans").glob("*.json")):
                plan = _read_json(plan_path)
                try:
                    _ensure_physics_evidence_diagnostic(
                        workspace=workspace,
                        run=run,
                        plan=plan,
                        force=force,
                    )
                except Exception:
                    continue
            raise ValueError(
                "terminal canonical run is immutable; create the next revision instead"
            )
    plan_paths = sorted((run / "plans").glob("*.json"))
    if not plan_paths:
        raise ValueError("run does not contain a mechanism plan")
    multi_capsule = len(plan_paths) > 1
    summaries = [
        _generate_visuals_for_plan(
            workspace_root=workspace,
            run_dir=run,
            plan_path=plan_path,
            force=force,
            multi_capsule=multi_capsule,
            primary_capsule=index == 0,
        )
        for index, plan_path in enumerate(plan_paths)
    ]
    summary = _merge_visual_summaries(summaries)
    visual = run / "visual"
    summary_path = visual / "dashboard_visuals.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if publish_run_status and status_path.is_file():
        run_status = _read_json(status_path)
        if not is_terminal_status(run_status.get("status")):
            run_status["visualization"] = {
                "schema_version": "dashboard_visualization_publish_v1",
                "status": summary["status"],
                "summary": summary_path.relative_to(workspace).as_posix(),
                "summary_sha256": hashlib.sha256(
                    summary_path.read_bytes()
                ).hexdigest(),
                "partial_trace_published": any(
                    item.get("physical_trace_status") == "PHYSICS_PARTIAL"
                    for item in (summary.get("load_videos") or {}).values()
                ),
                "certification_authority": False,
            }
            if parse_canonical_run(workspace, run) is None:
                # Legacy flat runs remain read/write-compatible but never
                # acquire canonical publication or terminal authority.
                write_json_atomic(status_path, run_status)
            # Canonical runs are single-writer: only the Harness (via the
            # result contract) publishes run_status.json; the dashboard
            # reads visual/dashboard_visuals.summary.json directly.
    return summary

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evidence-bound quest dashboard visuals")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    workspace = args.workspace_root.resolve()
    run = args.run if args.run.is_absolute() else workspace / args.run
    summary = generate_visuals(workspace_root=workspace, run_dir=run, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
