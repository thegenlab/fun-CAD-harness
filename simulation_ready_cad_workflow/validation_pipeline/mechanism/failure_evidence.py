from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from .io import (
    canonical_sha256,
    ensure_inside,
    read_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .models import MechanismPlan


MAX_RETAINED_FAILURES = 3
_FAILURE_SEVERITY = {
    "PHYSICAL_FAILURE": 4,
    "CLAIM_MISMATCH": 3,
    "PHYSICS_INCONCLUSIVE": 2,
    "NOT_AVAILABLE": 1,
}
_SAMPLE_SIGNAL_KEYS = (
    "max_joint_drift_m",
    "max_support_drift_m",
    "off_plane_rotation_rad",
    "off_axis_rotation_rad",
    "lateral_shift_m",
    "coupling_error_m",
    "rotation_from_start_rad",
)


def _render_lang_is_en() -> bool:
    # Same rule as dashboard.render_lang; mechanism code may not import dashboard.
    return os.environ.get("SIMREADY_RENDER_LANG", "").strip().lower().startswith("en")


def _label(zh: str, en: str) -> str:
    # Same rule as dashboard.render_lang.label; mechanism code may not import dashboard.
    return en if _render_lang_is_en() else zh


def _sans_serif_fonts() -> list[str]:
    # Same rule as dashboard.render_lang.sans_serif_fonts: CJK fonts lead only
    # when the rendered labels are Chinese.
    if _render_lang_is_en():
        return ["DejaVu Sans"]
    return ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]


def _failure_score(comparison: dict[str, Any]) -> tuple[int, int]:
    return (
        _FAILURE_SEVERITY.get(str(comparison.get("verdict")), 0),
        len(comparison.get("failure_reasons") or ()),
    )


def select_representative_failures(
    comparisons: Sequence[dict[str, Any]],
    *,
    limit: int = MAX_RETAINED_FAILURES,
) -> list[tuple[str, dict[str, Any]]]:
    failures = [item for item in comparisons if item.get("verdict") != "PASS"]
    if not failures or limit <= 0:
        return []
    selected: list[tuple[str, dict[str, Any]]] = []

    def add(role: str, item: dict[str, Any]) -> None:
        if len(selected) >= limit:
            return
        load_id = str(item.get("load_case_id") or "")
        if any(str(existing.get("load_case_id") or "") == load_id for _, existing in selected):
            return
        selected.append((role, item))

    add("first_failure", failures[0])
    remaining = [
        item
        for item in failures
        if all(
            item.get("load_case_id") != existing.get("load_case_id")
            for _, existing in selected
        )
    ]
    if remaining:
        add("worst_failure", max(remaining, key=_failure_score))
    remaining = [
        item
        for item in failures
        if all(
            item.get("load_case_id") != existing.get("load_case_id")
            for _, existing in selected
        )
    ]
    if remaining:
        add("closest_to_pass", min(remaining, key=_failure_score))
    for item in failures:
        add("additional_representative", item)
    return selected


def _missing_requirement_for_case(case: dict[str, Any]) -> str:
    existing = str(case.get("missing_requirement") or "").strip()
    if existing and re.search(r"[\u4e00-\u9fff]", existing):
        return existing
    tokens = " ".join(
        [
            existing,
            str(case.get("failure_code") or ""),
            str(case.get("failure_stage") or ""),
            *[str(item) for item in case.get("failure_reasons") or ()],
        ]
    ).lower()
    if "timeout" in tokens or "runtime_budget" in tokens or "runtime budget" in tokens:
        return _label(
            "实体仿真没有在时间预算内完成，缺少可判定的完整轨迹。",
            "The solid simulation did not finish within its time budget, "
            "so no complete trajectory can be judged.",
        )
    if "trace_not_completed" in tokens or "trace_evidence_missing" in tokens:
        return _label(
            "实体仿真的运行记录不完整，暂时不能判断它是否满足要求。",
            "The solid simulation's run record is incomplete, "
            "so it cannot yet be judged against the requirements.",
        )
    if "convex component" in tokens or "geometry_legality" in tokens:
        return _label(
            "运动实体的碰撞几何不合法，必须换成受支持且可验证的真实部件。",
            "The moving solid's collision geometry is invalid; "
            "replace it with a supported, verifiable real part.",
        )
    if "joint_drift" in tokens or "support_drift" in tokens:
        return _label(
            "运动部件没有被轴承或导轨稳定约束，运行后偏离了应有位置。",
            "Moving parts are not held steadily by bearings or guides "
            "and drifted from their intended position.",
        )
    if "off_plane" in tokens or "off_axis" in tokens:
        return _label(
            "机构出现明显侧翻或轴外转动，说明支撑与约束仍不完整。",
            "The mechanism visibly tipped sideways or rotated off its axis, "
            "so supports and constraints are still incomplete.",
        )
    if "did_not_move" in tokens or "target_not_reached" in tokens:
        return _label(
            "机构没有完成这个工况要求的有效行程。",
            "The mechanism did not complete the effective travel this load case requires.",
        )
    if "dwell" in tokens or "home_stop" in tokens:
        return _label(
            "机构没有可靠到达并保持目标位置。",
            "The mechanism did not reliably reach and hold its target position.",
        )
    return _label(
        "实体物理运行没有满足这个工况的安全运动要求。",
        "The physical run did not meet this load case's safe-motion requirements.",
    )


def _historical_failure_signature(case: dict[str, Any]) -> tuple[str, ...]:
    failure_code = str(case.get("failure_code") or "").strip()
    reasons = tuple(sorted(str(item) for item in case.get("failure_reasons") or () if item))
    if failure_code:
        return ("code", failure_code)
    if reasons:
        return ("reasons", *reasons)
    return ("role", str(case.get("role") or "unknown"))


def carry_forward_failure_snapshots(
    plan: MechanismPlan,
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    limit: int = MAX_RETAINED_FAILURES,
) -> tuple[dict[str, Any], list[Path]]:
    """Copy a few diverse failure snapshots from older runs into the new run."""
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    output.relative_to(workspace)
    if limit <= 0:
        return {"case_count": 0, "cases": []}, []

    candidates: list[dict[str, Any]] = []
    candidate_snapshot_digests: set[str] = set()

    def add_cases(summary: dict[str, Any], source_run: Path, priority: int) -> None:
        try:
            source_mtime = source_run.stat().st_mtime_ns
        except OSError:
            source_mtime = 0
        for raw_case in summary.get("cases") or ():
            if not isinstance(raw_case, dict) or not raw_case.get("snapshot"):
                continue
            if raw_case.get("capsule_id") not in (None, "", plan.capsule_id):
                continue
            try:
                snapshot = ensure_inside(workspace, str(raw_case["snapshot"]))
            except (OSError, ValueError):
                continue
            if not snapshot.is_file() or snapshot.is_symlink():
                continue
            snapshot_digest = sha256_file(snapshot)
            if snapshot_digest in candidate_snapshot_digests:
                continue
            candidate_snapshot_digests.add(snapshot_digest)
            case = dict(raw_case)
            case["_snapshot_path"] = snapshot
            case["_source_run"] = source_run
            case["_source_mtime"] = source_mtime
            case["_priority"] = priority
            candidates.append(case)

    existing_path = output / "historical_failure_evidence.json"
    if existing_path.is_file() and not existing_path.is_symlink():
        try:
            add_cases(read_json(existing_path), output, 2)
        except (OSError, ValueError):
            pass

    for sibling in output.parent.iterdir():
        if sibling == output or not sibling.is_dir() or sibling.is_symlink():
            continue
        plan_matches = False
        plan_root = sibling / "plans"
        if plan_root.is_dir() and not plan_root.is_symlink():
            for source_plan_path in sorted(plan_root.glob("*.json")):
                try:
                    source_plan = read_json(source_plan_path)
                except (OSError, ValueError):
                    continue
                if (
                    source_plan.get("package_id") == plan.package_id
                    and source_plan.get("capsule_id") == plan.capsule_id
                ):
                    plan_matches = True
                    break
        if not plan_matches:
            continue
        for summary_path in (
            sibling / "runtime_failure_evidence.json",
            sibling / "results" / plan.capsule_id / "failure_evidence" / "summary.json",
        ):
            if not summary_path.is_file() or summary_path.is_symlink():
                continue
            try:
                add_cases(read_json(summary_path), sibling, 1)
            except (OSError, ValueError):
                continue

    ordered = sorted(
        candidates,
        key=lambda item: (
            int(item.get("_priority") or 0),
            int(item.get("_source_mtime") or 0),
            _FAILURE_SEVERITY.get(str(item.get("verdict") or ""), 0),
            len(item.get("failure_reasons") or ()),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    seen_signatures: set[tuple[str, ...]] = set()
    seen_loads: set[str] = set()

    def select(case: dict[str, Any]) -> None:
        selected.append(case)
        selected_ids.add(id(case))
        seen_signatures.add(_historical_failure_signature(case))
        seen_loads.add(str(case.get("load_case_id") or ""))

    for case in ordered:
        if len(selected) >= limit:
            break
        signature = _historical_failure_signature(case)
        load_id = str(case.get("load_case_id") or "")
        if signature in seen_signatures or load_id in seen_loads:
            continue
        select(case)
    for case in ordered:
        if len(selected) >= limit:
            break
        if id(case) in selected_ids:
            continue
        load_id = str(case.get("load_case_id") or "")
        if load_id in seen_loads:
            continue
        select(case)
    for case in ordered:
        if len(selected) >= limit:
            break
        if id(case) not in selected_ids:
            select(case)

    retained: list[dict[str, Any]] = []
    generated: list[Path] = []
    history_root = output / "results" / plan.capsule_id / "failure_history"
    for index, case in enumerate(selected, start=1):
        source_snapshot = Path(case.pop("_snapshot_path"))
        source_run = Path(case.pop("_source_run"))
        case.pop("_source_mtime", None)
        case.pop("_priority", None)
        load_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case.get("load_case_id") or "case"))
        digest = sha256_file(source_snapshot)
        target = history_root / f"{index:02d}_{load_id}_{digest[:8]}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or sha256_file(target) != digest:
            shutil.copy2(source_snapshot, target)
        case["source_snapshot"] = relative_to_root(workspace, source_snapshot)
        case["source_run"] = relative_to_root(workspace, source_run)
        case["snapshot"] = relative_to_root(workspace, target)
        case["snapshot_sha256"] = digest
        case["missing_requirement"] = _missing_requirement_for_case(case)
        case["simulation_rerun_for_snapshot"] = False
        case["future_comparison_slot"] = "failed_before_valid_entity"
        case["historical"] = True
        retained.append(case)
        generated.append(target)

    summary = {
        "schema_version": "0.1",
        "package_id": plan.package_id,
        "capsule_id": plan.capsule_id,
        "selection_policy": [
            "different_failure_type",
            "different_load_case",
            "most_recent_then_most_severe",
        ],
        "maximum_retained_cases": limit,
        "case_count": len(retained),
        "cases": retained,
        "comparison_ready_when_valid_entity_exists": True,
        "snapshots_copied_into_current_run": True,
    }
    if retained:
        write_json_atomic(existing_path, summary)
        generated.append(existing_path)
    return summary, generated


def informative_sample_index(samples: Sequence[dict[str, Any]]) -> int:
    if not samples:
        raise ValueError("physical trace has no samples")
    maxima = {
        key: max(
            abs(float(item.get(key, 0.0)))
            for item in samples
            if isinstance(item.get(key, 0.0), (int, float))
            and math.isfinite(float(item.get(key, 0.0)))
        )
        for key in _SAMPLE_SIGNAL_KEYS
    }

    def score(sample: dict[str, Any]) -> float:
        total = 0.0
        for key, maximum in maxima.items():
            value = sample.get(key, 0.0)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                total += abs(float(value)) / max(maximum, 1e-15)
        return total

    return max(range(len(samples)), key=lambda index: score(samples[index]))


def _reason_text(reasons: Iterable[Any]) -> str:
    mapping = {
        "physical_wrong_motion_subspace": _label(
            "实体运动偏离声明的机构方向",
            "Solid motion left the declared mechanism direction",
        ),
        "physical_lateral_drift_exceeded": _label(
            "实体出现过大的侧向漂移",
            "Solid shows excessive lateral drift",
        ),
        "physical_joint_drift_exceeded": _label(
            "实体关节支撑漂移超限",
            "Solid joint support drift exceeds the threshold",
        ),
        "physical_off_plane_rotation_exceeded": _label(
            "实体出现过大的离平面转动",
            "Solid shows excessive off-plane rotation",
        ),
        "physical_coupling_error_exceeded": _label(
            "实体传动耦合误差超限",
            "Solid drive coupling error exceeds the threshold",
        ),
        "TRACE_EVIDENCE_MISSING": _label(
            "实体轨迹不完整，无法形成有效判定",
            "Solid trajectory is incomplete, so no valid verdict is possible",
        ),
        "physical_illegal_interpenetration": _label(
            "运动过程中两个真实碰撞实体发生了超限穿透",
            "Two physical collision solids interpenetrated beyond the threshold during motion",
        ),
        "physical_broad_body_pair_collision_exclusion_forbidden": _label(
            "实体分支关闭了整对刚体碰撞，接触证据不完整",
            "The solid branch disabled collision for a whole rigid-body pair, "
            "so contact evidence is incomplete",
        ),
    }
    values = [str(item) for item in reasons]
    return _label("；", ". ").join(mapping.get(item, item) for item in values) or _label(
        "实体工况未通过", "Solid load case failed"
    )


def _render_trace_failure_snapshot(
    workspace: Path,
    trace: dict[str, Any],
    comparison: dict[str, Any],
    *,
    sample_index: int,
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ..dashboard.trace_pair_video import _body_vertices_mm, _trace_scene_bodies

    matplotlib.rcParams["font.sans-serif"] = _sans_serif_fonts()
    matplotlib.rcParams["axes.unicode_minus"] = False
    samples = trace["samples"]
    sample = samples[sample_index]
    time_s = float(sample["time_s"])
    bodies = _trace_scene_bodies(workspace, trace)
    vertices_by_body = [
        _body_vertices_mm(body, samples, time_s) for body in bodies
    ]
    points = np.vstack(vertices_by_body)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.20)
    center = (minimum + maximum) * 0.5
    padding = span * 0.62

    fig = plt.figure(figsize=(13.2, 7.0), facecolor="#f5f7fa")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.45, 0.85), wspace=0.04)
    axis = fig.add_subplot(grid[0, 0], projection="3d")
    info = fig.add_subplot(grid[0, 1])
    for body_index, (body, vertices) in enumerate(zip(bodies, vertices_by_body)):
        faces = body["mesh"].faces
        if len(faces) > 24_000:
            faces = faces[:: math.ceil(len(faces) / 24_000)]
        fixed = bool(body.get("fixed"))
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor="#aab5c2" if fixed else "#df5a4f",
                edgecolor="#667484" if fixed else "#8f261f",
                linewidth=0.16,
                alpha=0.30 if fixed else max(0.64, 0.92 - body_index * 0.05),
            )
        )
    axis.set_xlim(center[0] - padding[0], center[0] + padding[0])
    axis.set_ylim(center[1] - padding[1], center[1] + padding[1])
    axis.set_zlim(center[2] - padding[2], center[2] + padding[2])
    axis.set_box_aspect(span)
    axis.view_init(elev=24, azim=-56)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_facecolor("#ffffff")
    axis.set_title(
        _label("失败时的实体状态（红色为运动实体）", "Solid state at failure (red = moving solids)"),
        fontsize=15,
        fontweight="bold",
    )

    info.axis("off")
    reasons = comparison.get("failure_reasons") or []
    info.text(
        0.0, 0.93, _label("为什么失败", "Why it failed"),
        fontsize=18, fontweight="bold", color="#17212b",
    )
    info.text(
        0.0,
        0.83,
        _reason_text(reasons),
        fontsize=12,
        color="#9f2f2a",
        wrap=True,
        va="top",
    )
    info.text(
        0.0, 0.62, _label("保留的证据", "Retained evidence"),
        fontsize=14, fontweight="bold", color="#17212b",
    )
    info.text(
        0.0,
        0.54,
        _label(
            f"工况：{comparison.get('load_case_id', '—')}\n"
            f"判定：{comparison.get('verdict', '—')}\n"
            f"时间：{time_s:.4f} s\n"
            f"接触数：{sample.get('contact_count', '—')}\n"
            f"轨迹样本：{len(samples)} 帧",
            f"Load case: {comparison.get('load_case_id', '—')}\n"
            f"Verdict: {comparison.get('verdict', '—')}\n"
            f"Time: {time_s:.4f} s\n"
            f"Contacts: {sample.get('contact_count', '—')}\n"
            f"Trajectory samples: {len(samples)} frames",
        ),
        fontsize=11.5,
        color="#344353",
        linespacing=1.6,
        va="top",
    )
    info.text(
        0.0,
        0.18,
        _label(
            "这张图来自失败运行的已保存轨迹，\n没有为了截图重新运行仿真。",
            "Drawn from the failed run's saved trajectory;\n"
            "the simulation was not rerun for this image.",
        ),
        fontsize=11,
        color="#647487",
        linespacing=1.5,
        va="top",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="#f5f7fa")
    plt.close(fig)


def build_trace_failure_evidence(
    plan: MechanismPlan,
    comparisons: Sequence[dict[str, Any]],
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], list[Path]]:
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    evidence_root = output / "results" / plan.capsule_id / "failure_evidence"
    generated: list[Path] = []
    cases: list[dict[str, Any]] = []
    for role, comparison in select_representative_failures(comparisons):
        load_id = str(comparison.get("load_case_id") or "unknown")
        trace_root = output / "traces" / plan.capsule_id / load_id
        physical_path = trace_root / "physical.trace.json"
        if not physical_path.is_file():
            continue
        try:
            physical = json.loads(physical_path.read_text(encoding="utf-8"))
            samples = physical.get("samples") or []
            if physical.get("status") != "COMPLETED" or not samples:
                continue
            geometry_legality = comparison.get("physical_geometry_legality")
            worst_events = (
                geometry_legality.get("worst_events")
                if isinstance(geometry_legality, dict)
                else None
            )
            if isinstance(worst_events, list) and worst_events:
                sample_index = int(worst_events[0]["sample_index"])
                if not 0 <= sample_index < len(samples):
                    raise ValueError("penetration evidence sample index is outside trace")
            else:
                sample_index = informative_sample_index(samples)
            snapshot_path = evidence_root / load_id / "failure_snapshot.png"
            _render_trace_failure_snapshot(
                workspace,
                physical,
                comparison,
                sample_index=sample_index,
                output=snapshot_path,
            )
        except (OSError, ValueError, KeyError, TypeError):
            continue
        comparison_path = output / "results" / plan.capsule_id / "comparison" / f"{load_id}.json"
        reduced_path = trace_root / "reduced.trace.json"
        sidecar_path = evidence_root / load_id / "failure_evidence.json"
        case = {
            "schema_version": "0.1",
            "role": role,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_id,
            "verdict": comparison.get("verdict"),
            "failure_reasons": list(comparison.get("failure_reasons") or []),
            "informative_sample_index": sample_index,
            "informative_time_s": float(samples[sample_index]["time_s"]),
            "snapshot": relative_to_root(workspace, snapshot_path),
            "snapshot_sha256": sha256_file(snapshot_path),
            "physical_trace": relative_to_root(workspace, physical_path),
            "physical_trace_sha256": sha256_file(physical_path),
            "reduced_trace": (
                relative_to_root(workspace, reduced_path)
                if reduced_path.is_file()
                else None
            ),
            "reduced_trace_sha256": (
                sha256_file(reduced_path) if reduced_path.is_file() else None
            ),
            "comparison": (
                relative_to_root(workspace, comparison_path)
                if comparison_path.is_file()
                else None
            ),
            "comparison_sha256": (
                sha256_file(comparison_path) if comparison_path.is_file() else None
            ),
            "package_binding_sha256": plan.package_binding_sha256,
            "simulation_rerun_for_snapshot": False,
            "future_comparison_slot": "failed_before_valid_entity",
        }
        write_json_atomic(sidecar_path, case)
        case["sidecar"] = relative_to_root(workspace, sidecar_path)
        cases.append(case)
        generated.extend((snapshot_path, sidecar_path))

    summary = {
        "schema_version": "0.1",
        "capsule_id": plan.capsule_id,
        "selection_policy": ["first_failure", "worst_failure", "closest_to_pass"],
        "maximum_retained_cases": MAX_RETAINED_FAILURES,
        "case_count": len(cases),
        "cases": cases,
        "comparison_ready_when_valid_entity_exists": True,
    }
    summary_path = evidence_root / "summary.json"
    write_json_atomic(summary_path, summary)
    generated.append(summary_path)
    return summary, generated


def _last_collision_progress(stderr_path: Path | None) -> dict[str, Any] | None:
    if stderr_path is None or not stderr_path.is_file():
        return None
    last: dict[str, Any] | None = None
    prefix = "SIMREADY_COLLISION_PROGRESS "
    for line in stderr_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = json.loads(line[len(prefix):])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            last = payload
    return last


def write_runtime_timeout_failure(
    collision_geometry: dict[str, Any],
    *,
    workspace_root: str | Path,
    evidence_root: str | Path,
    capsule_id: str,
    load_case_id: str,
    request_file: str,
    job_id: str,
    timeout_seconds: float,
    last_progress: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[Path]]:
    """Render a truthful snapshot for an exact-geometry job that timed out."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ..dashboard.trace_pair_video import _quaternion_matrix

    workspace = Path(workspace_root).resolve()
    root = Path(evidence_root).resolve()
    root.relative_to(workspace)
    rendered: list[tuple[np.ndarray, Any, bool]] = []
    all_points: list[np.ndarray] = []
    for body in collision_geometry.get("bodies") or []:
        is_moving = not bool(body.get("fixed"))
        rotation = _quaternion_matrix(body["initial_rotation_abs_quat_wxyz"])
        center = np.asarray(body["initial_center_abs_m"], dtype=float)
        for component in body.get("components") or []:
            path = ensure_inside(workspace, str(component["mesh"]))
            if sha256_file(path) != component.get("sha256"):
                raise ValueError(f"runtime-timeout component hash changed: {path}")
            mesh = trimesh.load_mesh(path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            vertices = (rotation @ np.asarray(mesh.vertices).T).T + center
            vertices_mm = vertices * 1000.0
            rendered.append((vertices_mm, mesh.faces, is_moving))
            all_points.append(vertices_mm)
    if not rendered:
        raise ValueError("runtime-timeout evidence has no renderable components")

    points = np.vstack(all_points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.20)
    center = (minimum + maximum) * 0.5
    padding = span * 0.62
    matplotlib.rcParams["font.sans-serif"] = _sans_serif_fonts()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.2, 7.0), facecolor="#f5f7fa")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.5, 0.8), wspace=0.04)
    axis = fig.add_subplot(grid[0, 0], projection="3d")
    info = fig.add_subplot(grid[0, 1])
    for vertices, faces, is_moving in rendered:
        if len(faces) > 16_000:
            faces = faces[:: math.ceil(len(faces) / 16_000)]
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor="#ef8b2c" if is_moving else "#b6c1cd",
                edgecolor="#9b4d0f" if is_moving else "#687786",
                linewidth=0.16,
                alpha=0.90 if is_moving else 0.28,
            )
        )
    axis.set_xlim(center[0] - padding[0], center[0] + padding[0])
    axis.set_ylim(center[1] - padding[1], center[1] + padding[1])
    axis.set_zlim(center[2] - padding[2], center[2] + padding[2])
    axis.set_box_aspect(span)
    axis.view_init(elev=24, azim=-56)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_facecolor("#ffffff")
    axis.set_title(
        _label("实体合法，但本次物理任务超时", "Solids are valid, but this physical job timed out"),
        fontsize=16,
        fontweight="bold",
    )
    info.axis("off")
    component_count = int(collision_geometry.get("component_count") or 0)
    progress_known = last_progress is not None
    progress_event = str(
        (last_progress or {}).get("event") or _label("未记录", "not recorded")
    )
    step_count = (last_progress or {}).get("step_count")
    progress_line = progress_event
    if step_count is not None:
        progress_line += _label(f" · 第 {step_count} 步", f" · step {step_count}")
    info.text(
        0.0, 0.93, _label("这次失败缺的是什么？", "What is this failure missing?"),
        fontsize=18, fontweight="bold", color="#17212b",
    )
    info.text(
        0.0,
        0.82,
        _label(
            f"完整载荷程序没有在 {timeout_seconds:.0f} 秒预算内跑完。\n"
            "这不是 CAD 非法，也不是退回代理几何；它缺的是足够快的精确接触执行，或有依据的更长预算。",
            "The full load program did not finish within\n"
            f"its {timeout_seconds:.0f} s budget. This is not invalid CAD\n"
            "or a fallback to proxy geometry; it needs faster\n"
            "exact-contact execution or a justified longer budget.",
        ),
        fontsize=12,
        color="#9b4d0f",
        va="top",
        linespacing=1.5,
        wrap=True,
    )
    info.text(
        0.0, 0.56, _label("已经确认", "Confirmed so far"),
        fontsize=14, fontweight="bold", color="#17212b",
    )
    info.text(
        0.0,
        0.49,
        _label(
            f"已绑定：权威 STEP 哈希\n已装配：{component_count} 个实体碰撞件\n"
            f"已进入：真实动力学求解\n最后记录：{progress_line}",
            "Bound: authoritative STEP hash\n"
            f"Assembled: {component_count} solid collision parts\n"
            "Entered: physical dynamics solve\n"
            f"Last record: {progress_line}",
        ),
        fontsize=11,
        color="#344353",
        va="top",
        linespacing=1.5,
    )
    info.text(
        0.0,
        0.12,
        _label(
            "橙色是运动实体。未来通过后，可把完成轨迹/视频与这张失败证据并排比较。",
            "Orange = moving solids. Once a run passes, compare its\n"
            "complete trajectory/video side by side with this evidence.",
        ),
        fontsize=11,
        color="#647487",
        va="top",
        wrap=True,
    )
    snapshot_path = root / "failure_snapshot.png"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(snapshot_path, dpi=150, bbox_inches="tight", facecolor="#f5f7fa")
    plt.close(fig)
    payload = {
        "schema_version": "0.1",
        "role": "runtime_timeout",
        "capsule_id": capsule_id,
        "load_case_id": load_case_id,
        "job_id": job_id,
        "failure_stage": "physical_runtime_budget",
        "failure_code": "PHYSICAL_RUNTIME_TIMEOUT",
        "timeout_seconds": timeout_seconds,
        "last_progress": last_progress,
        # A timeout alone proves neither home-pose legality nor illegality.  In
        # particular, conda may buffer the worker's progress stream until exit;
        # preserve unknown as null instead of claiming that simulation did not
        # start merely because the buffered progress log is empty.
        "geometry_legal": None,
        "geometry_importable": True,
        "home_pose_interpenetration_checked": False,
        "all_components_loaded": (
            None
            if not progress_known
            else progress_event not in {
                "body_build_begin", "component_classify_begin", "component_mesh_load_begin",
                "component_attach_triangle_mesh_begin", "component_attach_box_begin",
                "component_attach_cylinder_begin", "component_attach_sphere_begin",
            }
        ),
        "simulation_started": (
            None
            if not progress_known
            else progress_event in {
                "first_dynamics_step_begin", "first_dynamics_step_complete", "dynamics_progress"
            }
        ),
        "trace_complete": False,
        "missing_requirement": "complete the full exact-contact load program within its declared runtime budget",
        "request_file": request_file,
        "snapshot": relative_to_root(workspace, snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "package_binding_sha256": collision_geometry.get("package_binding_sha256"),
        "component_inventory_sha256": canonical_sha256(collision_geometry.get("bodies") or []),
        "simulation_rerun_for_snapshot": False,
        "future_comparison_slot": "failed_before_valid_entity",
    }
    sidecar_path = root / "failure_evidence.json"
    write_json_atomic(sidecar_path, payload)
    payload["sidecar"] = relative_to_root(workspace, sidecar_path)
    return payload, [snapshot_path, sidecar_path]


def build_runtime_failure_evidence(
    queue_state: dict[str, Any],
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
) -> tuple[dict[str, Any], list[Path]]:
    """Retain at most three representative physical runtime timeouts."""

    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    run_dir = Path(str(queue_state.get("run_dir") or "")).resolve()
    run_dir.relative_to(workspace)
    cases: list[dict[str, Any]] = []
    generated: list[Path] = []
    jobs = queue_state.get("jobs") or {}
    for job_id, job in jobs.items():
        if len(cases) >= MAX_RETAINED_FAILURES or job.get("status") != "TIMED_OUT":
            continue
        command = list(job.get("command") or [])
        if "--request" not in command:
            continue
        index = command.index("--request")
        if index + 1 >= len(command):
            continue
        request_relative = str(command[index + 1])
        request = read_json(ensure_inside(workspace, request_relative))
        if request.get("branch") != "physical":
            continue
        plan = read_json(ensure_inside(workspace, str(request["plan_file"])))
        load_case = read_json(ensure_inside(workspace, str(request["load_case_file"])))
        family_spec = ((plan.get("adapter_spec") or {}).get("family_spec") or {})
        collision_geometry = family_spec.get("collision_geometry")
        if not isinstance(collision_geometry, dict):
            continue
        stderr_path = None
        if job.get("stderr_log"):
            candidate = (run_dir / str(job["stderr_log"])).resolve()
            candidate.relative_to(run_dir)
            stderr_path = candidate
        match = re.search(r"after\s+([0-9.]+)\s+seconds", str(job.get("reason") or ""))
        timeout_seconds = float(match.group(1)) if match else float(job.get("duration_seconds") or 0.0)
        capsule_id = str(plan["capsule_id"])
        load_case_id = str(load_case["load_case_id"])
        evidence, files = write_runtime_timeout_failure(
            collision_geometry,
            workspace_root=workspace,
            evidence_root=output / "results" / capsule_id / "failure_evidence" / load_case_id / "runtime_timeout",
            capsule_id=capsule_id,
            load_case_id=load_case_id,
            request_file=request_relative,
            job_id=str(job_id),
            timeout_seconds=timeout_seconds,
            last_progress=_last_collision_progress(stderr_path),
        )
        cases.append(evidence)
        generated.extend(files)
    summary = {
        "schema_version": "0.1",
        "selection_policy": "first_three_physical_timeouts",
        "maximum_retained_cases": MAX_RETAINED_FAILURES,
        "case_count": len(cases),
        "cases": cases,
        "comparison_ready_when_valid_entity_exists": True,
    }
    summary_path = output / "runtime_failure_evidence.json"
    write_json_atomic(summary_path, summary)
    generated.append(summary_path)
    return summary, generated


def write_geometry_legality_failure(
    collision_geometry: dict[str, Any],
    *,
    workspace_root: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Save a visual explanation when STEP bodies cannot enter physics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from ..dashboard.trace_pair_video import _quaternion_matrix

    workspace = Path(workspace_root).resolve()
    root = Path(evidence_root).resolve()
    root.relative_to(workspace)
    unsupported = {
        (str(item["body_id"]), str(item["solid_id"]))
        for item in collision_geometry.get("unsupported_dynamic_components") or []
    }
    rendered: list[tuple[np.ndarray, Any, bool]] = []
    all_points: list[np.ndarray] = []
    for body in collision_geometry.get("bodies") or []:
        body_id = str(body["body_id"])
        rotation = _quaternion_matrix(body["initial_rotation_abs_quat_wxyz"])
        center = np.asarray(body["initial_center_abs_m"], dtype=float)
        for component in body.get("components") or []:
            path = ensure_inside(workspace, str(component["mesh"]))
            if sha256_file(path) != component.get("sha256"):
                raise ValueError(f"geometry failure component hash changed: {path}")
            mesh = trimesh.load_mesh(path, process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            vertices = (rotation @ np.asarray(mesh.vertices).T).T + center
            vertices_mm = vertices * 1000.0
            is_unsupported = (body_id, str(component["solid_id"])) in unsupported
            rendered.append((vertices_mm, mesh.faces, is_unsupported))
            all_points.append(vertices_mm)
    if not rendered:
        raise ValueError("geometry failure evidence has no renderable components")

    points = np.vstack(all_points)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    largest = max(float(span.max()), 1.0)
    span = np.maximum(span, largest * 0.20)
    center = (minimum + maximum) * 0.5
    padding = span * 0.62
    matplotlib.rcParams["font.sans-serif"] = _sans_serif_fonts()
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig = plt.figure(figsize=(13.2, 7.0), facecolor="#f5f7fa")
    grid = fig.add_gridspec(1, 2, width_ratios=(1.5, 0.8), wspace=0.04)
    axis = fig.add_subplot(grid[0, 0], projection="3d")
    info = fig.add_subplot(grid[0, 1])
    for vertices, faces, is_unsupported in rendered:
        if len(faces) > 16_000:
            faces = faces[:: math.ceil(len(faces) / 16_000)]
        axis.add_collection3d(
            Poly3DCollection(
                vertices[faces],
                facecolor="#e34f45" if is_unsupported else "#b6c1cd",
                edgecolor="#8c211b" if is_unsupported else "#687786",
                linewidth=0.16,
                alpha=0.95 if is_unsupported else 0.26,
            )
        )
    axis.set_xlim(center[0] - padding[0], center[0] + padding[0])
    axis.set_ylim(center[1] - padding[1], center[1] + padding[1])
    axis.set_zlim(center[2] - padding[2], center[2] + padding[2])
    axis.set_box_aspect(span)
    axis.view_init(elev=24, azim=-56)
    axis.set_proj_type("ortho")
    axis.set_axis_off()
    axis.set_facecolor("#ffffff")
    axis.set_title(
        _label("尚不能进入实体物理模拟", "Not yet ready for physical solid simulation"),
        fontsize=16,
        fontweight="bold",
    )
    info.axis("off")
    labels = [f"{body}/{solid}" for body, solid in sorted(unsupported)]
    info.text(
        0.0, 0.93, _label("红色部分缺什么", "What the red parts lack"),
        fontsize=18, fontweight="bold", color="#17212b",
    )
    info.text(
        0.0,
        0.82,
        _label(
            "这些运动实体仍是不可直接用于动态接触的非凸整块。\n"
            "需要拆成可验证的凸实体，或改成可证明等价的精确基础形状。",
            "These moving solids are still non-convex blocks\n"
            "that cannot be used directly for dynamic contact.\n"
            "Split them into verifiable convex solids, or use\n"
            "provably equivalent exact primitive shapes.",
        ),
        fontsize=12,
        color="#9f2f2a",
        va="top",
        linespacing=1.5,
        wrap=True,
    )
    info.text(
        0.0, 0.56, _label("被拒绝的实体", "Rejected solids"),
        fontsize=14, fontweight="bold", color="#17212b",
    )
    info.text(
        0.0,
        0.49,
        "\n".join(labels[:12])
        + (
            _label(f"\n另有 {len(labels) - 12} 个", f"\n... and {len(labels) - 12} more")
            if len(labels) > 12
            else ""
        ),
        fontsize=10.5,
        family="monospace",
        color="#344353",
        va="top",
        linespacing=1.35,
    )
    info.text(
        0.0,
        0.12,
        _label(
            "旧流程会用盒子替代并继续；新流程在这里失败关闭。",
            "The old flow swapped in boxes and carried on;\n"
            "the new flow fails closed here.",
        ),
        fontsize=11,
        color="#647487",
        va="top",
    )
    snapshot_path = root / "failure_snapshot.png"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(snapshot_path, dpi=150, bbox_inches="tight", facecolor="#f5f7fa")
    plt.close(fig)
    payload = {
        "schema_version": "0.1",
        "failure_stage": "physical_geometry_legality",
        "failure_code": "FAMILY_DYNAMIC_COLLIDER_REPRESENTATION_UNSUPPORTED",
        "unsupported_dynamic_components": [
            {"body_id": body, "solid_id": solid}
            for body, solid in sorted(unsupported)
        ],
        "missing_requirement": "dynamic solid must be an exact supported primitive or a verified convex component",
        "snapshot": relative_to_root(workspace, snapshot_path),
        "snapshot_sha256": sha256_file(snapshot_path),
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "package_binding_sha256": collision_geometry.get("package_binding_sha256"),
        "component_inventory_sha256": canonical_sha256(
            collision_geometry.get("bodies") or []
        ),
        "simulation_started": False,
        "future_comparison_slot": "failed_before_valid_entity",
    }
    sidecar_path = root / "failure_evidence.json"
    write_json_atomic(sidecar_path, payload)
    payload["sidecar"] = relative_to_root(workspace, sidecar_path)
    return payload
