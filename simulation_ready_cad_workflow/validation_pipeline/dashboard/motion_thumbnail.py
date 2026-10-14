"""Small looping animation of a recorded motion, for the run monitor.

The renderer draws the original body meshes of a replay at a few evenly spaced
recorded frames with one fixed orthographic camera and writes an animated GIF.
Rendering runs in a background thread through a separate interpreter and is
cached per replay version, so monitor refreshes never wait for it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading

FRAMES = 18
WIDTH, HEIGHT = 320, 200
MAX_TRIANGLES = 30000
RENDER_TIMEOUT_SECONDS = 240


def thumbnail_file(cache: Path, motion: dict) -> Path:
    """Cache path of one motion's thumbnail; a new replay gets a new file."""
    version = str(motion.get("replay_version") or "")
    if not version:
        replay = Path(str(motion.get("replay_path") or ""))
        stat = replay.stat() if replay.is_file() else None
        version = hashlib.sha256(f"{replay}:{stat and stat.st_size}:{stat and stat.st_mtime_ns}".encode()).hexdigest()
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(motion.get("id") or "motion"))[:60]
    return cache / "motion-thumbnails" / f"{name}-{version[:16]}.gif"


class MotionThumbnails:
    """One background renderer shared by all monitor requests of a dashboard."""

    def __init__(self, python: str):
        self.python = python
        self.jobs: queue.Queue = queue.Queue()
        self.pending: set[str] = set()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, name="motion-thumbnails", daemon=True)
        self.thread.start()

    def status(self, target: Path, motion: dict, up_axis=None) -> str:
        """ready, failed or pending; a missing thumbnail is queued for rendering."""
        if target.is_file():
            return "ready"
        if target.with_suffix(".failed.json").is_file():
            return "failed"
        with self.lock:
            if str(target) not in self.pending:
                self.pending.add(str(target))
                self.jobs.put({"model": motion.get("motion_model_path"), "replay": motion.get("replay_path"),
                               "scene": motion.get("scene_path"), "up_axis": up_axis, "output": str(target)})
        return "pending"

    def _loop(self):
        while True:
            request = self.jobs.get()
            target = Path(request["output"])
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                path = target.with_suffix(".request.json")
                path.write_text(json.dumps(request), encoding="utf-8")
                completed = subprocess.run([self.python, "-X", "utf8", "-m", __name__, str(path)],
                                           capture_output=True, timeout=RENDER_TIMEOUT_SECONDS)
                if completed.returncode or not target.is_file():
                    raise RuntimeError(completed.stderr.decode("utf-8", "replace")[-2000:])
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                target.with_suffix(".failed.json").write_text(json.dumps({"error": str(exc)}), encoding="utf-8")
            finally:
                with self.lock:
                    self.pending.discard(str(target))


def render(request: dict) -> None:
    import io

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.collections import PolyCollection
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import trimesh

    scene = trimesh.load(request["model"], force="scene", process=False)
    meshes = {}
    for node in scene.graph.nodes_geometry:
        match = re.fullmatch(r"physics__(.+)__original", str(node))
        mesh = scene.geometry[scene.graph[node][1]]
        if match and isinstance(mesh, trimesh.Trimesh) and len(mesh.faces):
            meshes[match.group(1)] = mesh
    if not meshes:
        raise ValueError("The replay model has no body surfaces")
    stride = max(1, -(-sum(len(mesh.faces) for mesh in meshes.values()) // MAX_TRIANGLES))
    local = {}
    for body, mesh in meshes.items():
        faces = np.arange(0, len(mesh.faces), stride)
        visual = mesh.visual.to_color() if hasattr(mesh.visual, "to_color") else mesh.visual
        rgba = np.asarray(visual.face_colors, dtype=float)[faces] / 255
        if len(rgba) != len(faces):
            rgba = np.tile([.5, .6, .65, 1], (len(faces), 1))
        local[body] = (mesh.triangles[faces], mesh.face_normals[faces], rgba)
    frames = json.loads(Path(request["replay"]).read_text(encoding="utf-8"))["branches"]["physical"]["frames"]
    picks = sorted({round(i) for i in np.linspace(0, len(frames) - 1, min(FRAMES, len(frames)))})
    # Orthographic camera above the gravity-up axis, at the dashboard's usual iso view.
    up = np.asarray(request.get("up_axis") or [0, 1, 0], dtype=float)
    world = trimesh.geometry.align_vectors(up, [0, 0, 1])[:3, :3]
    elevation, azimuth = np.radians(23), np.radians(-58)
    eye = np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.cross([0, 0, 1], eye)
    right /= np.linalg.norm(right)
    view = np.vstack([right, np.cross(eye, right), eye]) @ world  # screen x, screen y, depth towards the eye
    light = view.T @ np.array([-.35, .45, .82])

    def project(index):
        poses = frames[index]["body_poses"]
        polygons, depths, colors = [], [], []
        for body, (triangles, normals, rgba) in local.items():
            pose = poses.get(body)
            if not pose:
                continue
            rotation = trimesh.transformations.quaternion_matrix(pose["rotation_abs_quat_wxyz"])[:3, :3]
            points = (triangles @ rotation.T + np.asarray(pose["position_abs_m"], dtype=float)) @ view.T
            shade = np.clip(.72 + .28 * ((normals @ rotation.T) @ light), .45, 1)
            polygons.append(points[:, :, :2])
            depths.append(points[:, :, 2].mean(axis=1))
            colors.append(np.column_stack([rgba[:, :3] * shade[:, None], rgba[:, 3]]))
        if not polygons:
            return np.zeros((0, 3, 2)), np.zeros((0, 4))
        polygons, depths, colors = np.concatenate(polygons), np.concatenate(depths), np.concatenate(colors)
        order = np.argsort(depths)
        finite = np.isfinite(polygons[order]).all(axis=(1, 2))
        return polygons[order][finite], colors[order][finite]

    projected = [project(index) for index in picks]
    # Frame the first pose; later poses may widen the shot up to 3x, so parts thrown far
    # away by a diverged simulation do not shrink the machine to a dot.
    first = projected[0][0].reshape(-1, 2)
    low, high = first.min(axis=0), first.max(axis=0)
    center, size = (low + high) / 2, np.maximum(high - low, 1e-4)
    for polygons, _colors in projected[1:]:
        if len(polygons):
            points = polygons.reshape(-1, 2)
            low = np.maximum(np.minimum(low, points.min(axis=0)), center - 1.5 * size)
            high = np.minimum(np.maximum(high, points.max(axis=0)), center + 1.5 * size)
    center, size = (low + high) / 2, (high - low) * 1.12
    scale = max(size[0] / WIDTH, size[1] / HEIGHT)
    half = np.array([WIDTH, HEIGHT]) * scale / 2

    figure = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor="#f6f7f3")
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_axis_off()
    axes.set_xlim(center[0] - half[0], center[0] + half[0])
    axes.set_ylim(center[1] - half[1], center[1] + half[1])
    images, durations = [], []
    times = [float(frames[i].get("time_s") or 0.0) for i in picks]
    for number, (polygons, colors) in enumerate(projected):
        for collection in list(axes.collections):
            collection.remove()
        axes.add_collection(PolyCollection(polygons, facecolors=colors, edgecolors=colors, linewidths=.2))
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=100, facecolor=figure.get_facecolor())
        images.append(Image.open(buffer).convert("P", palette=Image.ADAPTIVE, colors=128))
        # Real recorded time between frames, kept between 60 ms and 400 ms per frame.
        gap = (times[number + 1] - times[number]) if number + 1 < len(times) else 0.6
        durations.append(int(min(400, max(60, gap * 1000))))
    plt.close(figure)
    target = Path(request["output"])
    temporary = target.with_suffix(".pending.gif")
    images[0].save(temporary, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)
    temporary.replace(target)


if __name__ == "__main__":
    render(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
