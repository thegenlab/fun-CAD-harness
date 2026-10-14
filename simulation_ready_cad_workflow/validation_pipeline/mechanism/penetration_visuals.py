from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import (
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .models import MechanismValidationError
from .physical_legality import (
    _component_path,
    _pose_matrix,
    normalized_physical_audit_inputs,
)


SUMMARY_NAME = "penetration_evidence_v1.json"
MAX_EVENTS_PER_TRACE = 3


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def _node_name(solid_ref: str) -> str:
    body_id, solid_id = solid_ref.split("/", 1)
    return f"physics__{_safe_name(body_id)}__{_safe_name(solid_id)}"


def _audit_path(run: Path, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit).resolve()
    candidates = (
        run / "results" / "physical_preflight.json",
        run / "migration" / "legacy_sampled_physical_audit_v1.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise _fail(
        "PENETRATION_EVIDENCE_AUDIT_MISSING",
        "run has no physical preflight or legacy sampled physical audit",
    )


def _event_source_index(trace_result: Mapping[str, Any], event: Mapping[str, Any]) -> int:
    if event.get("source_sample_index") is not None:
        return int(event["source_sample_index"])
    audit_index = int(event.get("audit_sample_index", event.get("sample_index", 0)))
    source_indices = (trace_result.get("sampling") or {}).get("selected_source_indices")
    if isinstance(source_indices, list) and 0 <= audit_index < len(source_indices):
        return int(source_indices[audit_index])
    return audit_index


def _select_events(
    trace_result: Mapping[str, Any], *, maximum: int
) -> list[dict[str, Any]]:
    geometry = trace_result.get("geometry_legality") or {}
    raw = geometry.get("worst_events") if isinstance(geometry, Mapping) else None
    if not isinstance(raw, list):
        return []
    ranked = sorted(
        (dict(item) for item in raw if isinstance(item, Mapping)),
        key=lambda item: float(item.get("penetration_depth_m") or 0.0),
        reverse=True,
    )
    retained: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, ...]] = set()
    for event in ranked:
        pair = tuple(str(item) for item in event.get("solid_pair") or [])
        if len(pair) != 2 or pair in seen_pairs:
            continue
        event["source_sample_index"] = _event_source_index(trace_result, event)
        event["audit_sample_index"] = int(
            event.get("audit_sample_index", event.get("sample_index", 0))
        )
        retained.append(event)
        seen_pairs.add(pair)
        if len(retained) >= maximum:
            break
    return retained


def _load_world_components(
    geometry: Mapping[str, Any],
    sample: Mapping[str, Any],
    *,
    workspace: Path,
) -> list[dict[str, Any]]:
    import numpy as np
    import trimesh

    poses = sample.get("body_poses")
    if not isinstance(poses, Mapping):
        raise _fail(
            "PENETRATION_EVIDENCE_POSES_MISSING",
            "source trace frame has no body poses",
        )
    rendered: list[dict[str, Any]] = []
    for body in geometry.get("bodies") or []:
        if not isinstance(body, Mapping):
            continue
        body_id = str(body.get("body_id") or "")
        if body_id not in poses:
            raise _fail(
                "PENETRATION_EVIDENCE_BODY_POSE_MISSING",
                f"source trace frame has no pose for {body_id}",
            )
        transform = _pose_matrix(poses[body_id])
        for component in body.get("components") or []:
            if not isinstance(component, Mapping):
                continue
            path = _component_path(component, workspace)
            recorded_sha256 = str(component.get("sha256") or "")
            actual_sha256 = sha256_file(path)
            if recorded_sha256 and recorded_sha256 != actual_sha256:
                raise _fail(
                    "PENETRATION_EVIDENCE_COMPONENT_HASH_MISMATCH",
                    f"collision mesh changed after audit: {path}",
                )
            mesh = trimesh.load_mesh(path, process=False, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                geometries = tuple(mesh.geometry.values())
                if not geometries:
                    raise _fail(
                        "PENETRATION_EVIDENCE_COMPONENT_EMPTY",
                        f"collision mesh is empty: {path}",
                    )
                mesh = trimesh.util.concatenate(geometries)
            vertices = np.asarray(mesh.vertices, dtype=float)
            world = vertices @ transform[:3, :3].T + transform[:3, 3]
            solid_id = str(component.get("solid_id") or path.stem)
            solid_ref = f"{body_id}/{solid_id}"
            rendered.append(
                {
                    "solid_ref": solid_ref,
                    "node": _node_name(solid_ref),
                    "body_id": body_id,
                    "solid_id": solid_id,
                    "fixed": bool(body.get("fixed")),
                    "vertices_mm": world * 1000.0,
                    "faces": mesh.faces,
                    "source_mesh": relative_to_root(workspace, path),
                    "source_mesh_sha256": actual_sha256,
                }
            )
    if not rendered:
        raise _fail(
            "PENETRATION_EVIDENCE_COMPONENTS_MISSING",
            "collision geometry has no renderable components",
        )
    return rendered


def _box_faces(minimum: Any, maximum: Any) -> list[list[list[float]]]:
    x0, y0, z0 = (float(item) for item in minimum)
    x1, y1, z1 = (float(item) for item in maximum)
    corners = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    return [
        [corners[i] for i in face]
        for face in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                     (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    ]


def _set_bounds(axis: Any, points: Any, *, padding_ratio: float) -> None:
    import numpy as np

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 0.1)
    largest = max(float(span.max()), 0.1)
    span = np.maximum(span, largest * 0.28)
    center = (minimum + maximum) * 0.5
    half = span * (0.5 + padding_ratio)
    axis.set_xlim(center[0] - half[0], center[0] + half[0])
    axis.set_ylim(center[1] - half[1], center[1] + half[1])
    axis.set_zlim(center[2] - half[2], center[2] + half[2])
    axis.set_box_aspect(span)


def _view_angles(normal: Sequence[float] | None) -> tuple[float, float]:
    if not isinstance(normal, Sequence) or len(normal) < 3:
        return 24.0, -56.0
    x, y, z = (float(item) for item in normal[:3])
    length = math.sqrt(x * x + y * y + z * z)
    if length <= 1.0e-12:
        return 24.0, -56.0
    x, y, z = x / length, y / length, z / length
    return math.degrees(math.asin(z)), math.degrees(math.atan2(y, x))


def _render_event(
    components: Sequence[Mapping[str, Any]],
    event: Mapping[str, Any],
    *,
    output: Path,
    load_case_id: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False
    pair = [str(item) for item in event.get("solid_pair") or []]
    if len(pair) != 2:
        raise _fail(
            "PENETRATION_EVIDENCE_SOLID_PAIR_INVALID",
            "penetration event does not identify two solids",
        )
    by_ref = {str(item["solid_ref"]): item for item in components}
    if any(item not in by_ref for item in pair):
        raise _fail(
            "PENETRATION_EVIDENCE_SOLID_PAIR_MISSING",
            f"collision component is missing for {pair}",
        )
    all_points = np.vstack([item["vertices_mm"] for item in components])
    pair_points = np.vstack([by_ref[item]["vertices_mm"] for item in pair])
    contact = np.asarray(event.get("contact_position_abs_m") or [0, 0, 0], dtype=float) * 1000.0
    overlap_min = np.asarray(event.get("overlap_bounds_min_abs_m") or contact / 1000.0, dtype=float) * 1000.0
    overlap_max = np.asarray(event.get("overlap_bounds_max_abs_m") or contact / 1000.0, dtype=float) * 1000.0
    overlap_span = np.maximum(overlap_max - overlap_min, 0.1)
    depth_mm = float(event.get("penetration_depth_m") or 0.0) * 1000.0
    local_radius = max(float(overlap_span.max()) * 3.5, depth_mm * 6.0, 6.0)
    local_points = np.vstack(
        (contact - local_radius, contact + local_radius, overlap_min, overlap_max)
    )
    colors = {pair[0]: "#267fca", pair[1]: "#ef9b32"}

    fig = plt.figure(figsize=(15.6, 7.8), facecolor="#f4f7fa")
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.36, 0.78), wspace=0.035)
    context = fig.add_subplot(grid[0, 0], projection="3d")
    closeup = fig.add_subplot(grid[0, 1], projection="3d")
    info = fig.add_subplot(grid[0, 2])

    def add_component(axis: Any, item: Mapping[str, Any], *, close: bool) -> None:
        faces = item["faces"]
        if len(faces) > 26000:
            faces = faces[:: math.ceil(len(faces) / 26000)]
        selected = item["solid_ref"] in colors
        if close and not selected:
            return
        facecolor = colors.get(str(item["solid_ref"]), "#aeb8c3")
        edgecolor = "#174d7c" if item["solid_ref"] == pair[0] else "#8d5010" if selected else "#7f8b97"
        axis.add_collection3d(
            Poly3DCollection(
                item["vertices_mm"][faces],
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=0.12 if not selected else 0.24,
                alpha=(
                    0.12
                    if not selected
                    else (
                        0.30
                        if close and item["solid_ref"] == pair[0]
                        else 0.82
                        if close
                        else 0.76
                    )
                ),
            )
        )

    for component in components:
        add_component(context, component, close=False)
        add_component(closeup, component, close=True)

    overlap_faces = _box_faces(overlap_min, overlap_max)
    for axis in (context, closeup):
        axis.add_collection3d(
            Poly3DCollection(
                overlap_faces,
                facecolor="#e63232",
                edgecolor="#b41010",
                linewidth=1.1,
                alpha=0.30,
            )
        )
        axis.scatter(
            [contact[0]], [contact[1]], [contact[2]],
            color="#ff1515", edgecolors="#7f0000", s=72 if axis is context else 150,
            depthshade=False,
        )
        normal = np.asarray(event.get("contact_normal_abs") or [0, 0, 0], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > 1.0e-12:
            normal /= norm
            length = max(float((overlap_max - overlap_min).max()) * 2.5, 2.0)
            start = contact - normal * length * 0.5
            axis.quiver(
                start[0], start[1], start[2], normal[0], normal[1], normal[2],
                length=length, color="#d41414", linewidth=2.0, arrow_length_ratio=0.22,
            )
        axis.set_proj_type("ortho")
        axis.set_axis_off()
        axis.set_facecolor("#ffffff")

    _set_bounds(context, all_points, padding_ratio=0.10)
    _set_bounds(closeup, local_points, padding_ratio=0.30)
    elevation, azimuth = _view_angles(event.get("contact_normal_abs"))
    context.view_init(elev=24, azim=-56)
    closeup.view_init(elev=elevation, azim=azimuth)
    context.set_title("整机中的位置", fontsize=14, fontweight="bold", pad=10)
    closeup.set_title("穿模位置特写", fontsize=16, fontweight="bold", pad=10, color="#a72121")

    info.axis("off")
    time_s = float(event.get("time_s") or 0.0)
    source_index = int(event.get("source_sample_index") or 0)
    info.text(0.0, 0.96, "实体穿模证据", fontsize=20, fontweight="bold", color="#17212b", va="top")
    info.text(0.0, 0.86, f"工况  {load_case_id}", fontsize=12.5, color="#405164", va="top")
    info.text(0.0, 0.79, f"时间  {time_s:.4f} s", fontsize=15, fontweight="bold", color="#17212b", va="top")
    info.text(0.0, 0.72, f"原始轨迹帧  {source_index}", fontsize=13, color="#17212b", va="top")
    info.text(0.0, 0.62, "碰撞零件", fontsize=12, fontweight="bold", color="#526172", va="top")
    info.text(0.0, 0.56, pair[0], fontsize=11, color="#267fca", va="top", wrap=True)
    info.text(0.0, 0.49, pair[1], fontsize=11, color="#d47d15", va="top", wrap=True)
    info.text(0.0, 0.37, "最大检出穿入", fontsize=12, fontweight="bold", color="#526172", va="top")
    info.text(0.0, 0.28, f"{depth_mm:.4f} mm", fontsize=24, fontweight="bold", color="#c62828", va="top")
    info.text(
        0.0, 0.13,
        "红点：检测到的接触位置\n红色半透明框：局部重叠邻域\n蓝/橙：发生穿模的两个 STEP 实体",
        fontsize=10.5, color="#5d6a78", linespacing=1.55, va="top",
    )
    fig.suptitle(
        "保存的真实物理轨迹 · 未重跑仿真",
        x=0.02, y=0.985, ha="left", fontsize=10.5, color="#647487",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="#f4f7fa")
    plt.close(fig)


def build_penetration_visual_evidence(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    audit_report: str | Path | None = None,
    maximum_events_per_trace: int = MAX_EVENTS_PER_TRACE,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    try:
        run.relative_to(workspace)
    except ValueError as exc:
        raise _fail(
            "PENETRATION_EVIDENCE_RUN_OUTSIDE_WORKSPACE",
            "run must remain inside the workspace",
        ) from exc
    if "scissor-lift" in run.name.lower().replace("_", "-"):
        raise _fail(
            "PENETRATION_EVIDENCE_RUN_EXCLUDED",
            "scissor-lift runs are explicitly excluded from this migration",
        )
    maximum = int(maximum_events_per_trace)
    if not 1 <= maximum <= 8:
        raise _fail(
            "PENETRATION_EVIDENCE_EVENT_LIMIT_INVALID",
            "maximum events per trace must be between 1 and 8",
        )
    report_path = _audit_path(run, audit_report)
    try:
        report_path.relative_to(run)
    except ValueError as exc:
        raise _fail(
            "PENETRATION_EVIDENCE_AUDIT_OUTSIDE_RUN",
            "audit report must remain inside the selected run",
        ) from exc
    report = read_json(report_path)
    report_sha256 = sha256_file(report_path)
    output_root = report_path.parent / "penetration_evidence_v1"
    events: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for trace_result in report.get("traces") or []:
        if not isinstance(trace_result, Mapping) or trace_result.get("passed") is True:
            continue
        trace_ref = str(trace_result.get("trace") or "")
        if not trace_ref:
            failures.append({"trace": "", "reason": "physical trace path missing"})
            continue
        trace_path = (workspace / trace_ref).resolve()
        try:
            trace_path.relative_to(workspace)
        except ValueError:
            failures.append({"trace": trace_ref, "reason": "physical trace outside workspace"})
            continue
        if not trace_path.is_file():
            failures.append({"trace": trace_ref, "reason": "physical trace missing"})
            continue
        trace_sha256 = sha256_file(trace_path)
        recorded_trace_sha256 = str(trace_result.get("trace_sha256") or "")
        if recorded_trace_sha256 and recorded_trace_sha256 != trace_sha256:
            failures.append({"trace": trace_ref, "reason": "physical trace hash mismatch"})
            continue
        trace = read_trace_json(trace_path)
        geometry, samples = normalized_physical_audit_inputs(trace)
        selected_events = _select_events(trace_result, maximum=maximum)
        load_case_id = str(trace_result.get("load_case_id") or trace.get("load_case_id") or trace_path.parent.name)
        capsule_id = str(trace_result.get("capsule_id") or trace.get("capsule_id") or trace_path.parent.parent.name)
        for rank, event in enumerate(selected_events, start=1):
            source_index = int(event["source_sample_index"])
            if not 0 <= source_index < len(samples):
                failures.append(
                    {
                        "trace": trace_ref,
                        "reason": f"source sample index outside trace: {source_index}",
                    }
                )
                continue
            if not event.get("contact_position_abs_m"):
                failures.append(
                    {
                        "trace": trace_ref,
                        "reason": "audit event has no contact position; regenerate the audit",
                    }
                )
                continue
            try:
                components = _load_world_components(
                    geometry,
                    samples[source_index],
                    workspace=workspace,
                )
                event_id = f"{_safe_name(capsule_id)}__{_safe_name(load_case_id)}__f{source_index:06d}__r{rank}"
                snapshot_path = output_root / _safe_name(capsule_id) / _safe_name(load_case_id) / f"{event_id}.png"
                _render_event(
                    components,
                    event,
                    output=snapshot_path,
                    load_case_id=load_case_id,
                )
                pair = [str(item) for item in event.get("solid_pair") or []]
                focus = {
                    "event_id": event_id,
                    "stage_id": "branch-simulation",
                    "load_case_id": load_case_id,
                    "branch": "physical",
                    "frame": source_index,
                    "layer": "selected",
                    "primary_node": _node_name(pair[0]),
                    "secondary_node": _node_name(pair[1]),
                    "solid_pair": pair,
                    "contact_position_abs_m": event.get("contact_position_abs_m"),
                    "contact_normal_abs": event.get("contact_normal_abs"),
                    "overlap_bounds_min_abs_m": event.get("overlap_bounds_min_abs_m"),
                    "overlap_bounds_max_abs_m": event.get("overlap_bounds_max_abs_m"),
                }
                sidecar_path = snapshot_path.with_suffix(".json")
                payload = {
                    "schema_version": "0.1",
                    "event_id": event_id,
                    "run": relative_to_root(workspace, run),
                    "capsule_id": capsule_id,
                    "load_case_id": load_case_id,
                    "phase_id": event.get("phase_id"),
                    "time_s": float(event.get("time_s") or samples[source_index].get("time_s") or 0.0),
                    "audit_sample_index": int(event.get("audit_sample_index") or 0),
                    "source_sample_index": source_index,
                    "body_pair": list(event.get("body_pair") or []),
                    "solid_pair": pair,
                    "penetration_depth_m": float(event.get("penetration_depth_m") or 0.0),
                    "fcl_reported_depth_m": float(event.get("fcl_reported_depth_m") or 0.0),
                    "contact_position_abs_m": event.get("contact_position_abs_m"),
                    "contact_position_source": event.get("contact_position_source"),
                    "contact_normal_abs": event.get("contact_normal_abs"),
                    "contact_normal_source": event.get("contact_normal_source"),
                    "overlap_bounds_min_abs_m": event.get("overlap_bounds_min_abs_m"),
                    "overlap_bounds_max_abs_m": event.get("overlap_bounds_max_abs_m"),
                    "snapshot": relative_to_root(workspace, snapshot_path),
                    "snapshot_sha256": sha256_file(snapshot_path),
                    "physical_trace": relative_to_root(workspace, trace_path),
                    "physical_trace_sha256": trace_sha256,
                    "audit_report": relative_to_root(workspace, report_path),
                    "audit_report_sha256": report_sha256,
                    "dashboard_focus": focus,
                    "simulation_rerun_for_evidence": False,
                    "red_region_claim": "detected_contact_neighborhood_not_exact_boolean_intersection",
                }
                write_json_atomic(sidecar_path, payload)
                payload["sidecar"] = relative_to_root(workspace, sidecar_path)
                events.append(payload)
            except (MechanismValidationError, OSError, ValueError, TypeError) as exc:
                failures.append({"trace": trace_ref, "reason": str(exc)})

    summary = {
        "schema_version": "0.1",
        "status": "PENETRATION_VISUAL_EVIDENCE_GENERATED" if events else "PENETRATION_VISUAL_EVIDENCE_NOT_AVAILABLE",
        "run": relative_to_root(workspace, run),
        "audit_report": relative_to_root(workspace, report_path),
        "audit_report_sha256": report_sha256,
        "event_count": len(events),
        "generation_failure_count": len(failures),
        "maximum_events_per_failed_trace": maximum,
        "selection_policy": "deepest_event_per_distinct_solid_pair",
        "simulation_rerun_for_evidence": False,
        "red_region_claim": "detected_contact_neighborhood_not_exact_boolean_intersection",
        "events": events,
        "generation_failures": failures,
    }
    summary_path = report_path.parent / SUMMARY_NAME
    write_json_atomic(summary_path, summary)
    summary["summary"] = relative_to_root(workspace, summary_path)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render close-up evidence for retained physical interpenetration events."
    )
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--audit-report")
    parser.add_argument("--maximum-events-per-trace", type=int, default=MAX_EVENTS_PER_TRACE)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = build_penetration_visual_evidence(
        workspace_root=args.workspace_root,
        run_dir=args.run,
        audit_report=args.audit_report,
        maximum_events_per_trace=args.maximum_events_per_trace,
    )
    print(summary["status"])
    print(f"events={summary['event_count']} failures={summary['generation_failure_count']}")
    return 0 if summary["event_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
