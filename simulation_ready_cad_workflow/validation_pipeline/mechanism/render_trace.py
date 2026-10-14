"""Render a mechanism validation trace as a physics diagnostic video."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


BALL_COLORS = (
    "#f4b942",
    "#e76f51",
    "#2a9d8f",
    "#4f86c6",
    "#9b5de5",
    "#f15bb5",
)
MARKER_COLORS = ("#ff1744", "#00b8d4", "#ffab00", "#651fff")
PHASE_COLORS = ("#2a9d8f", "#e76f51", "#4f86c6", "#9b5de5")


def quaternion_matrix(wxyz: list[float] | tuple[float, ...]) -> np.ndarray:
    values = np.asarray(wxyz, dtype=float)
    norm = float(np.linalg.norm(values))
    if norm <= 1e-15:
        return np.eye(3)
    w, x, y, z = values / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quaternion_angle_deg(q0: list[float], q1: list[float]) -> float:
    left = np.asarray(q0, dtype=float)
    right = np.asarray(q1, dtype=float)
    left /= max(float(np.linalg.norm(left)), 1e-15)
    right /= max(float(np.linalg.norm(right)), 1e-15)
    dot = abs(float(np.dot(left, right)))
    return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot))))


def transformed_vertices_mm(
    vertices_local_m: np.ndarray, pose: dict[str, list[float]]
) -> np.ndarray:
    rotation = quaternion_matrix(pose["rotation_abs_quat_wxyz"])
    position = np.asarray(pose["position_abs_m"], dtype=float)
    return ((rotation @ vertices_local_m.T).T + position) * 1000.0


def resolved_workspace_path(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve()
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"trace asset escapes workspace root: {relative_path}") from exc
    return candidate


def load_mesh(workspace_root: Path, item: dict) -> trimesh.Trimesh:
    path = resolved_workspace_path(workspace_root, item["derived"])
    mesh = trimesh.load_mesh(path, process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"trace mesh is empty: {path}")
    return mesh


def configure_axis(
    axis,
    points_mm: np.ndarray,
    *,
    elev: float,
    azim: float,
    title: str,
) -> None:
    minimum = points_mm.min(axis=0)
    maximum = points_mm.max(axis=0)
    span = maximum - minimum
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.16)
    center = (minimum + maximum) * 0.5
    padded = span * 0.62
    axis.set_xlim(center[0] - padded[0], center[0] + padded[0])
    axis.set_ylim(center[1] - padded[1], center[1] + padded[1])
    axis.set_zlim(center[2] - padded[2], center[2] + padded[2])
    axis.set_box_aspect(span)
    axis.view_init(elev=elev, azim=azim)
    axis.set_facecolor("#ffffff")
    axis.set_xlabel("X (mm)")
    axis.set_ylabel("Y (mm)")
    axis.set_zlabel("Z (mm)")
    axis.tick_params(labelsize=7)
    axis.set_title(title, fontsize=10)


def phase_spans(samples: list[dict]) -> list[tuple[str, float, float]]:
    spans: list[tuple[str, float, float]] = []
    start = 0
    for index in range(1, len(samples) + 1):
        if index == len(samples) or samples[index]["phase_id"] != samples[start]["phase_id"]:
            spans.append(
                (
                    str(samples[start]["phase_id"]),
                    float(samples[start]["time_s"]),
                    float(samples[index - 1]["time_s"]),
                )
            )
            start = index
    return spans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a trusted mechanism trace with collision geometry and motion markers."
    )
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--workspace-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--frame-repeat", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.trace.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    if not samples:
        raise ValueError("trace contains no samples")
    if args.fps <= 0 or args.dpi <= 0 or args.frame_repeat <= 0:
        raise ValueError("fps, dpi, and frame-repeat must be positive")

    collision = payload["collision_geometry"]
    fixed_mesh = load_mesh(args.workspace_root, collision["fixed"])
    moving_mesh = load_mesh(args.workspace_root, collision["moving"])
    fixed_pose = {
        "position_abs_m": collision["fixed"]["body_origin_abs_m"],
        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    fixed_vertices = transformed_vertices_mm(fixed_mesh.vertices, fixed_pose)
    fixed_triangles = fixed_vertices[fixed_mesh.faces]

    moving_vertices_by_sample = [
        transformed_vertices_mm(moving_mesh.vertices, sample["moving_pose"])
        for sample in samples
    ]
    initial_moving_triangles = moving_vertices_by_sample[0][moving_mesh.faces]

    rolling = collision.get("rolling_elements") or {}
    rolling_states_present = bool(samples[0].get("rolling_element_states"))
    radius_mm = float(rolling.get("radius_m", 0.001)) * 1000.0
    sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius_mm)
    initial_states = {
        item["element_id"]: item for item in samples[0].get("rolling_element_states", [])
    }
    element_ids = sorted(initial_states)
    tracked_ids = element_ids[:4]

    all_points = [fixed_vertices, moving_vertices_by_sample[0], moving_vertices_by_sample[-1]]
    ball_points = []
    if rolling_states_present:
        ball_points = [
            np.asarray(item["position_abs_m"], dtype=float) * 1000.0
            for sample in samples
            for item in sample.get("rolling_element_states", [])
        ]
        all_points.append(np.asarray(ball_points))
    assembly_points = np.vstack(all_points)
    detail_points = (
        np.vstack((np.asarray(ball_points), moving_vertices_by_sample[0], moving_vertices_by_sample[-1]))
        if ball_points
        else np.vstack((moving_vertices_by_sample[0], moving_vertices_by_sample[-1]))
    )

    times = np.asarray([sample["time_s"] for sample in samples], dtype=float)
    progress_mm = np.asarray(
        [sample["generalized_progress_m"] * 1000.0 for sample in samples], dtype=float
    )
    lateral_mm = np.asarray(
        [sample["lateral_shift_m"] * 1000.0 for sample in samples], dtype=float
    )
    tracked_angle_deg = np.zeros(len(samples))
    if tracked_ids:
        tracked_id = tracked_ids[0]
        q0 = initial_states[tracked_id]["rotation_abs_quat_wxyz"]
        for index, sample in enumerate(samples):
            state = next(
                item
                for item in sample["rolling_element_states"]
                if item["element_id"] == tracked_id
            )
            tracked_angle_deg[index] = quaternion_angle_deg(
                q0, state["rotation_abs_quat_wxyz"]
            )
    else:
        tracked_id = None

    fig = plt.figure(figsize=(16, 9), facecolor="#f3f5f7")
    grid = fig.add_gridspec(
        2, 2, height_ratios=(3.0, 1.12), width_ratios=(1.18, 1.0), hspace=0.18, wspace=0.08
    )
    ax_full = fig.add_subplot(grid[0, 0], projection="3d")
    ax_detail = fig.add_subplot(grid[0, 1], projection="3d")
    ax_plot = fig.add_subplot(grid[1, :])
    configure_axis(
        ax_full,
        assembly_points,
        elev=22,
        azim=-58,
        title="Collision assembly: fixed and moving meshes are transparent",
    )
    configure_axis(
        ax_detail,
        detail_points,
        elev=15,
        azim=-72,
        title="Contact detail: colored markers follow rolling-body quaternions",
    )

    fixed_artists = []
    moving_artists = []
    for axis in (ax_full, ax_detail):
        fixed_artist = Poly3DCollection(
            fixed_triangles,
            facecolor="#8b98a5",
            edgecolor="#52606d",
            linewidth=0.15,
            alpha=0.16,
        )
        moving_artist = Poly3DCollection(
            initial_moving_triangles,
            facecolor="#2f7fd3",
            edgecolor="#174f83",
            linewidth=0.15,
            alpha=0.20,
        )
        axis.add_collection3d(fixed_artist)
        axis.add_collection3d(moving_artist)
        fixed_artists.append(fixed_artist)
        moving_artists.append(moving_artist)

    ball_artists: dict[str, list[Poly3DCollection]] = {}
    for index, element_id in enumerate(element_ids):
        state = initial_states[element_id]
        vertices = transformed_vertices_mm(sphere.vertices, state)
        triangles = vertices[sphere.faces]
        color = BALL_COLORS[index % len(BALL_COLORS)]
        artists = []
        for axis in (ax_full, ax_detail):
            artist = Poly3DCollection(
                triangles,
                facecolor=color,
                edgecolor="#513b1c",
                linewidth=0.12,
                alpha=0.98,
            )
            axis.add_collection3d(artist)
            artists.append(artist)
        ball_artists[element_id] = artists

    ax_plot.set_facecolor("#ffffff")
    for index, (phase_id, start_s, end_s) in enumerate(phase_spans(samples)):
        ax_plot.axvspan(
            start_s,
            end_s,
            color=PHASE_COLORS[index % len(PHASE_COLORS)],
            alpha=0.09,
            label=phase_id,
        )
    ax_plot.axhline(0.0, color="#aab2ba", linewidth=0.8)
    ax_plot.plot(times, progress_mm, color="#2f7fd3", linewidth=2.0, label="axis progress")
    ax_plot.plot(times, lateral_mm, color="#d62828", linewidth=1.4, label="lateral shift")
    angle_axis = ax_plot.twinx()
    if tracked_id:
        angle_axis.plot(
            times,
            tracked_angle_deg,
            color="#9b5de5",
            linewidth=1.5,
            label=f"{tracked_id} orientation",
        )
    cursor = ax_plot.axvline(times[0], color="#17202a", linewidth=1.0, alpha=0.55)
    progress_marker, = ax_plot.plot([], [], "o", color="#2f7fd3", markersize=5)
    ax_plot.set_xlim(float(times[0]), max(float(times[-1]), float(times[0]) + 1e-6))
    motion_bound = max(0.25, float(max(np.abs(progress_mm).max(), np.abs(lateral_mm).max())) * 1.2)
    ax_plot.set_ylim(-motion_bound, motion_bound)
    angle_axis.set_ylim(0.0, max(10.0, float(tracked_angle_deg.max()) * 1.15))
    ax_plot.set_xlabel("simulation time (s)")
    ax_plot.set_ylabel("displacement (mm)")
    angle_axis.set_ylabel("rolling-body orientation change (deg)", color="#7b2cbf")
    ax_plot.grid(True, alpha=0.2)
    handles, labels = ax_plot.get_legend_handles_labels()
    handles2, labels2 = angle_axis.get_legend_handles_labels()
    ax_plot.legend(
        handles + handles2,
        labels + labels2,
        loc="upper center",
        ncol=min(6, len(handles + handles2)),
        frameon=False,
        fontsize=8,
    )
    metric_text = ax_plot.text(
        0.01,
        0.04,
        "",
        transform=ax_plot.transAxes,
        va="bottom",
        fontsize=8.5,
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 2.5},
    )

    status = payload.get("status", "UNKNOWN")
    fig.suptitle(
        f"{payload.get('capsule_id', 'mechanism')} / {payload.get('load_case_id', 'load')}"
        f" | {payload.get('branch', 'unknown')} branch | {status}",
        fontsize=14,
        fontweight="bold",
    )

    keyframe_indices = {
        "start": 0,
        "max_progress": int(np.argmax(progress_mm)),
        "min_progress": int(np.argmin(progress_mm)),
        "end": len(samples) - 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    keyframe_paths = {
        name: args.output.with_suffix(f".{name}.png") for name in keyframe_indices
    }
    writer = FFMpegWriter(
        fps=args.fps,
        codec="libx264",
        bitrate=6500,
        extra_args=["-pix_fmt", "yuv420p"],
    )

    dynamic_artists = []
    with writer.saving(fig, str(args.output), args.dpi):
        for sample_index, sample in enumerate(samples):
            moving_triangles = moving_vertices_by_sample[sample_index][moving_mesh.faces]
            for artist in moving_artists:
                artist.set_verts(moving_triangles)

            state_map = {
                item["element_id"]: item
                for item in sample.get("rolling_element_states", [])
            }
            for element_id, state in state_map.items():
                vertices = transformed_vertices_mm(sphere.vertices, state)
                triangles = vertices[sphere.faces]
                for artist in ball_artists[element_id]:
                    artist.set_verts(triangles)

            for artist in dynamic_artists:
                artist.remove()
            dynamic_artists = []

            for tracked_index, element_id in enumerate(tracked_ids):
                state = state_map[element_id]
                center = np.asarray(state["position_abs_m"], dtype=float) * 1000.0
                marker_direction = quaternion_matrix(state["rotation_abs_quat_wxyz"]) @ np.asarray(
                    [1.0, 0.0, 0.0]
                )
                tail = center - marker_direction * radius_mm * 0.65
                color = MARKER_COLORS[tracked_index]
                for axis in (ax_full, ax_detail):
                    dynamic_artists.append(
                        axis.quiver(
                            *tail,
                            *marker_direction,
                            length=radius_mm * 1.8,
                            color=color,
                            linewidth=3.0,
                            arrow_length_ratio=0.25,
                        )
                    )

            force = np.asarray(sample["applied_force_abs_n"], dtype=float)
            force_norm = float(np.linalg.norm(force))
            if force_norm > 1e-12:
                point = np.asarray(sample["force_application_point_abs_m"], dtype=float) * 1000.0
                direction = force / force_norm
                dynamic_artists.append(
                    ax_full.quiver(
                        *point,
                        *direction,
                        length=max(float(np.ptp(assembly_points, axis=0).max()) * 0.18, 2.0),
                        color="#d62828",
                        linewidth=2.5,
                        arrow_length_ratio=0.24,
                    )
                )

            gravity = np.asarray(payload.get("engine", {}).get("gravity_abs_m_s2", [0, 0, 0]), dtype=float)
            gravity_norm = float(np.linalg.norm(gravity))
            if gravity_norm > 1e-12:
                minimum = assembly_points.min(axis=0)
                maximum = assembly_points.max(axis=0)
                anchor = np.asarray([minimum[0], maximum[1], maximum[2]], dtype=float)
                dynamic_artists.append(
                    ax_full.quiver(
                        *anchor,
                        *(gravity / gravity_norm),
                        length=max(float(np.ptp(assembly_points, axis=0).max()) * 0.14, 1.5),
                        color="#202020",
                        linewidth=2.0,
                        arrow_length_ratio=0.25,
                    )
                )

            cursor.set_xdata([sample["time_s"], sample["time_s"]])
            progress_marker.set_data([sample["time_s"]], [progress_mm[sample_index]])
            metric_text.set_text(
                f"t={sample['time_s']:.3f}s  phase={sample['phase_id']}  "
                f"axis={progress_mm[sample_index]:+.3f}mm  "
                f"lateral={lateral_mm[sample_index]:.3f}mm  "
                f"rotation={math.degrees(sample['rotation_from_start_rad']):.3f}deg  "
                f"contacts={sample['contact_count']}"
            )

            for name, keyframe_index in keyframe_indices.items():
                if sample_index == keyframe_index:
                    fig.savefig(keyframe_paths[name], dpi=args.dpi, bbox_inches="tight")
            for _ in range(args.frame_repeat):
                writer.grab_frame()

    sheet_path = args.output.with_suffix(".contact_sheet.png")
    sheet_fig, sheet_axes = plt.subplots(2, 2, figsize=(15, 8.5), facecolor="#f3f5f7")
    for axis, (name, path) in zip(sheet_axes.flat, keyframe_paths.items()):
        axis.imshow(mpimg.imread(path))
        axis.set_title(name.replace("_", " "), fontsize=10)
        axis.axis("off")
    sheet_fig.tight_layout()
    sheet_fig.savefig(sheet_path, dpi=100, bbox_inches="tight")
    plt.close(sheet_fig)

    summary = {
        "ok": args.output.is_file() and args.output.stat().st_size > 0,
        "trace": str(args.trace.resolve()),
        "video": str(args.output.resolve()),
        "contact_sheet": str(sheet_path.resolve()),
        "keyframes": {name: str(path.resolve()) for name, path in keyframe_paths.items()},
        "simulation_samples": len(samples),
        "encoded_frames": len(samples) * args.frame_repeat,
        "fps": args.fps,
        "playback_seconds": len(samples) * args.frame_repeat / args.fps,
        "fixed_collision_triangles": int(len(fixed_mesh.faces)),
        "moving_collision_triangles": int(len(moving_mesh.faces)),
        "rolling_body_count": len(element_ids),
        "tracked_rolling_bodies": tracked_ids,
        "transparent_collision_meshes": True,
        "max_axis_progress_mm": float(progress_mm.max()),
        "min_axis_progress_mm": float(progress_mm.min()),
        "max_lateral_shift_mm": float(lateral_mm.max()),
        "max_tracked_orientation_change_deg": float(tracked_angle_deg.max()),
    }
    summary_path = args.output.with_suffix(".render_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
