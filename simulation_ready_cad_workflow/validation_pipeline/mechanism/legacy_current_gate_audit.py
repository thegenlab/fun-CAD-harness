from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "legacy_current_gate_audit_v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _certificate_summary(run_dir: Path) -> dict[str, Any]:
    files = sorted((run_dir / "results").glob("*/certificate.json"))
    verdicts: list[str] = []
    for path in files:
        value = _read_json(path)
        verdicts.append(str(value.get("verdict") or "NOT_AVAILABLE"))
    return {
        "count": len(files),
        "verdicts": verdicts,
        "historical_pass_count": sum(item == "KINEMATIC_CERTIFIED" for item in verdicts),
    }


def _migration_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "recorded": False,
            "passed": False,
            "status": "NOT_RECORDED",
            "trace_count": 0,
            "passed_trace_count": 0,
            "failed_trace_count": 0,
            "selected_pose_count": 0,
            "max_penetration_mm": None,
            "broad_exclusion_count": 0,
            "component_contact_authoritative_count": 0,
        }
    value = _read_json(path)
    traces = [item for item in value.get("traces") or [] if isinstance(item, dict)]
    selected = 0
    maximum = 0.0
    exclusions = 0
    authoritative = 0
    for item in traces:
        sampling = item.get("sampling") or {}
        geometry = item.get("geometry_legality") or {}
        selected += int(
            sampling.get("selected_sample_count")
            or geometry.get("sample_count")
            or 0
        )
        maximum = max(maximum, float(geometry.get("max_penetration_m") or 0.0))
        exclusions += len(item.get("historical_broad_body_pair_collision_exclusions") or [])
        contact = geometry.get("component_contact_intent") or {}
        authoritative += int(contact.get("authoritative") is True)
    return {
        "recorded": True,
        "path": str(path),
        "sha256": _sha256(path),
        "passed": value.get("passed") is True,
        "status": str(value.get("status") or "NOT_AVAILABLE"),
        "trace_count": int(value.get("trace_count") or len(traces)),
        "passed_trace_count": int(value.get("passed_trace_count") or 0),
        "failed_trace_count": int(value.get("failed_trace_count") or 0),
        "selected_pose_count": selected,
        "max_penetration_mm": round(maximum * 1000.0, 6),
        "broad_exclusion_count": exclusions,
        "component_contact_authoritative_count": authoritative,
    }


def _plan_summary(plan_files: list[Path]) -> dict[str, Any]:
    plans = [_read_json(path) for path in plan_files]
    acceptance = [
        item.get("acceptance_contract")
        for item in plans
        if isinstance(item.get("acceptance_contract"), dict)
    ]
    accepted = bool(plans) and len(acceptance) == len(plans) and all(
        item.get("status") == "PASS"
        and item.get("certificate_authority") is True
        for item in acceptance
    )
    return {
        "count": len(plan_files),
        "files": [str(path) for path in plan_files],
        "typed_acceptance_recorded": len(acceptance) == len(plans) and bool(plans),
        "typed_acceptance_passed": accepted,
        "families": sorted(
            {
                str(item.get("family") or "unknown")
                for item in plans
            }
        ),
    }


def _contact_policy_summary(physical_traces: list[Path]) -> dict[str, Any]:
    authoritative = 0
    legacy_support_only = 0
    for path in physical_traces:
        trace = _read_json(path)
        semantics = trace.get("branch_semantics") or {}
        geometry = trace.get("collision_geometry") or {}
        policy = (
            semantics.get("component_contact_intent_policy")
            or geometry.get("component_contact_intent_policy")
        )
        interface = semantics.get("physical_interface_policy")
        if isinstance(policy, dict) or isinstance(interface, dict):
            authoritative += 1
        elif (
            semantics.get("physical_compliant_supports")
            or semantics.get("analytical_transmission_interfaces")
        ):
            legacy_support_only += 1
    return {
        "trace_count": len(physical_traces),
        "authoritative_trace_count": authoritative,
        "legacy_support_only_trace_count": legacy_support_only,
        "complete": bool(physical_traces) and authoritative == len(physical_traces),
        "note": (
            "legacy support metadata does not identify exact body/solid pairs"
            if legacy_support_only
            else ""
        ),
    }


def _collider_fidelity_summary(run_dir: Path, physical_count: int) -> dict[str, Any]:
    reports = sorted(run_dir.glob("derived_adapters/**/collider_fidelity.json"))
    values = [_read_json(path) for path in reports]
    passed = bool(values) and all(item.get("passed") is True for item in values)
    derived_assets = any(run_dir.glob("derived_adapters/**/*.obj"))
    return {
        "recorded": bool(reports),
        "passed": passed,
        "report_count": len(reports),
        "reports": [str(path) for path in reports],
        "derivable_without_pychrono": bool(physical_count and derived_assets),
    }


def audit_run(
    workspace_root: str | Path,
    run: str | Path,
    *,
    skip: bool = False,
    skip_reason: str = "",
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    run_dir = _inside(workspace, run)
    relative = run_dir.relative_to(workspace).as_posix()
    if skip:
        return {
            "run": relative,
            "run_name": run_dir.name,
            "status": "SKIPPED_BY_SCOPE",
            "skip_reason": skip_reason or "explicitly excluded",
            "run_directory_read": False,
            "historical_artifacts_modified": False,
            "pychrono_rerun": False,
        }

    status_path = run_dir / "run_status.json"
    status = _read_json(status_path)
    plan_files = sorted((run_dir / "plans").glob("*.json"))
    physical_traces = sorted((run_dir / "traces").glob("**/physical.trace.json"))
    ideal_references = sorted((run_dir / "traces").glob("**/ideal_reference.trace.json"))
    ideal_dynamics = sorted((run_dir / "traces").glob("**/ideal_dynamics.trace.json"))
    legacy_reduced = sorted((run_dir / "traces").glob("**/reduced.trace.json"))
    migration = _migration_summary(
        run_dir / "migration" / "legacy_sampled_physical_audit_v1.json"
    )
    plans = _plan_summary(plan_files)
    contact = _contact_policy_summary(physical_traces)
    fidelity = _collider_fidelity_summary(run_dir, len(physical_traces))
    certificate = _certificate_summary(run_dir)

    gaps: list[str] = []
    if not fidelity["passed"]:
        gaps.append("collider_fidelity_not_recorded_or_not_passing")
    if not plans["typed_acceptance_passed"]:
        gaps.append("typed_acceptance_contract_not_authoritative")
    if not migration["recorded"]:
        gaps.append("saved_pose_sampled_audit_not_recorded")
    elif not migration["passed"]:
        gaps.append("saved_pose_geometry_legality_failed")
    if not contact["complete"]:
        gaps.append("component_contact_intent_not_authoritative")
    if len(ideal_references) < len(physical_traces):
        gaps.append("deterministic_ideal_reference_missing")
    if migration["broad_exclusion_count"]:
        gaps.append("historical_broad_body_pair_collision_exclusion_disclosed")

    geometry_repair = migration["recorded"] and not migration["passed"]
    if geometry_repair:
        action = "REPAIR_AUTHORITATIVE_CAD_THEN_NEW_FULL_RUN"
        rationale = (
            "saved physical poses contain illegal STEP-solid penetration; "
            "a geometry change invalidates old hashes, so current recertification "
            "requires a new physical run"
        )
    elif not migration["recorded"]:
        action = "RUN_DERIVED_SAMPLED_AUDIT_FIRST"
        rationale = (
            "generate the saved-pose audit from existing traces before deciding "
            "whether any PyChrono rerun is necessary"
        )
    elif gaps:
        action = "DERIVE_MISSING_EVIDENCE_BEFORE_RERUN"
        rationale = (
            "reuse immutable traces and collision assets for fidelity, contact "
            "policy, and ideal-reference derivation; start a new physical run "
            "only if new bindings or repaired geometry make the old trace stale"
        )
    else:
        action = "CURRENT_GATES_RECORDED"
        rationale = "all current evidence categories are recorded and passing"

    return {
        "run": relative,
        "run_name": run_dir.name,
        "status": (
            "CURRENT_GATES_RECORDED"
            if not gaps
            else "CURRENT_CERTIFICATION_NOT_ESTABLISHED"
        ),
        "historical_run_status": str(status.get("status") or "NOT_AVAILABLE"),
        "historical_certificate": certificate,
        "plans": plans,
        "physical_trace_count": len(physical_traces),
        "ideal_reference_trace_count": len(ideal_references),
        "ideal_dynamics_trace_count": len(ideal_dynamics),
        "legacy_reduced_trace_count": len(legacy_reduced),
        "collider_fidelity": fidelity,
        "sampled_physical_legality": migration,
        "component_contact_intent": contact,
        "current_gate_gaps": gaps,
        "geometry_repair_and_physics_rerun_required_for_current_certificate": geometry_repair,
        "recommended_action": action,
        "recommendation_rationale": rationale,
        "historical_artifacts_modified": False,
        "pychrono_rerun": False,
    }


def build_report(
    workspace_root: str | Path,
    runs: Iterable[str | Path],
    *,
    skip_prefixes: Iterable[str] = (),
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    prefixes = tuple(str(item) for item in skip_prefixes)
    cases = []
    for run in runs:
        name = Path(run).name
        skipped = any(name.startswith(prefix) for prefix in prefixes)
        cases.append(
            audit_run(
                workspace,
                run,
                skip=skipped,
                skip_reason=(
                    "Kimi scissor-lift case remains untouched by explicit user scope"
                    if skipped
                    else ""
                ),
            )
        )
    audited = [item for item in cases if item["status"] != "SKIPPED_BY_SCOPE"]
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "read_only_saved_artifact_inventory_and_existing_sampled_audit",
        "workspace_root": str(workspace),
        "pychrono_rerun": False,
        "historical_artifact_mutation": False,
        "case_count": len(cases),
        "audited_case_count": len(audited),
        "skipped_case_count": len(cases) - len(audited),
        "current_gates_recorded_count": sum(
            item["status"] == "CURRENT_GATES_RECORDED" for item in audited
        ),
        "physics_rerun_required_for_current_certificate_count": sum(
            item.get(
                "geometry_repair_and_physics_rerun_required_for_current_certificate"
            )
            is True
            for item in audited
        ),
        "cases": cases,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 旧机械在当前 Harness 门槛下的只读审计",
        "",
        "本报告只读取已有 plan、trace、证书和已生成的抽样实体审查；没有重跑 PyChrono，也没有改写任何历史运行产物。",
        "",
        "| 运行 | 历史状态 | 物理轨迹 | 抽样实体审查 | 最大穿入 | Collider 保真 | 类型化目标 | 零件级接触 | 理想参考 | 当前证书需重跑物理 |",
        "|---|---|---:|---|---:|---|---|---|---:|---|",
    ]
    for item in report.get("cases") or []:
        if item.get("status") == "SKIPPED_BY_SCOPE":
            lines.append(
                f"| {item['run_name']} | 按范围跳过 | — | 未读取 | — | — | — | — | — | 否（未判定） |"
            )
            continue
        audit = item["sampled_physical_legality"]
        fidelity = item["collider_fidelity"]
        acceptance = item["plans"]
        contact = item["component_contact_intent"]
        maximum = audit.get("max_penetration_mm")
        lines.append(
            "| "
            + " | ".join(
                [
                    item["run_name"],
                    item["historical_run_status"],
                    str(item["physical_trace_count"]),
                    (
                        f"{audit['passed_trace_count']}/{audit['trace_count']} 通过"
                        if audit["recorded"]
                        else "未记录"
                    ),
                    "—" if maximum is None else f"{maximum:.3f} mm",
                    "通过" if fidelity["passed"] else "缺少当前证据",
                    "通过" if acceptance["typed_acceptance_passed"] else "缺少当前绑定",
                    "通过" if contact["complete"] else "缺少实体对声明",
                    str(item["ideal_reference_trace_count"]),
                    (
                        "是"
                        if item[
                            "geometry_repair_and_physics_rerun_required_for_current_certificate"
                        ]
                        else "先不重跑"
                    ),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"- 审计 {report.get('audited_case_count', 0)} 个案例，跳过 {report.get('skipped_case_count', 0)} 个案例。",
            f"- 需要先修权威 CAD、再以新哈希重跑物理才能申请当前证书：{report.get('physics_rerun_required_for_current_certificate_count', 0)} 个。",
            "- “先不重跑”不等于当前 PASS，只表示应先从已有不可变产物派生 Collider 保真、零件级接触或理想参考，避免浪费旧工作。",
            "- 历史证书仍作为历史事实保留；没有任何旧 PASS 被升级成当前规则 PASS。",
            "",
            "## 每个案例的最小动作",
            "",
        ]
    )
    action_text = {
        "REPAIR_AUTHORITATIVE_CAD_THEN_NEW_FULL_RUN": (
            "先修权威 CAD；已有抽样姿态存在非法实体穿入。几何和哈希改变后，"
            "若要申请当前证书，必须在新运行目录重跑真实物理"
        ),
        "RUN_DERIVED_SAMPLED_AUDIT_FIRST": (
            "先用已有轨迹生成抽样实体审查，不重跑 PyChrono；看过派生结果后"
            "再决定是否需要新物理运行"
        ),
        "DERIVE_MISSING_EVIDENCE_BEFORE_RERUN": (
            "先从已有不可变轨迹和碰撞资产派生 Collider 保真、零件级接触和"
            "理想参考；只有新绑定或修复几何使旧轨迹失效时才重跑物理"
        ),
        "CURRENT_GATES_RECORDED": "当前证据类别已记录并通过",
    }
    for item in report.get("cases") or []:
        if item.get("status") == "SKIPPED_BY_SCOPE":
            detail = "按用户范围跳过，未读取 Kimi 剪叉运行目录。"
        else:
            detail = action_text.get(
                item.get("recommended_action"),
                "按 JSON 报告中的缺口逐项补齐。",
            )
        lines.append(f"- **{item['run_name']}**：{detail}")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit historical runs against current harness evidence gates without rerunning physics."
    )
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--skip-prefix", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args(argv)

    workspace = Path(args.workspace_root).resolve()
    report = build_report(
        workspace,
        args.run,
        skip_prefixes=args.skip_prefix,
    )
    json_path = _inside(workspace, args.output_json)
    md_path = _inside(workspace, args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8", newline="")
    print(json.dumps({
        "output_json": str(json_path),
        "output_md": str(md_path),
        "audited_case_count": report["audited_case_count"],
        "skipped_case_count": report["skipped_case_count"],
        "pychrono_rerun": False,
        "historical_artifact_mutation": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
