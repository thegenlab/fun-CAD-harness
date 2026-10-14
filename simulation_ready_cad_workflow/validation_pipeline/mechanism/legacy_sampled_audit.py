from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from .collision_policy import legacy_interface_declarations
from .io import (
    JsonDocumentError,
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .models import MechanismValidationError
from .physical_legality import (
    audit_family_collision_geometry,
    disclosed_broad_collision_exclusions,
)
from .sampling import select_saved_trace_samples


REPORT_NAME = "legacy_sampled_physical_audit_v1.json"


def _is_excluded_run(run_dir: Path) -> bool:
    lowered = run_dir.name.lower().replace("_", "-")
    return "scissor-lift" in lowered


def _normalized_audit_inputs(
    trace: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    geometry = trace.get("collision_geometry")
    samples = trace.get("samples")
    if not isinstance(geometry, Mapping) or not isinstance(samples, list):
        raise MechanismValidationError(
            "LEGACY_AUDIT_EVIDENCE_MISSING",
            "physical trace lacks collision geometry or saved poses",
        )
    if isinstance(geometry.get("bodies"), list):
        return geometry, [dict(item) for item in samples]

    fixed_components = geometry.get("fixed_components")
    moving_components = geometry.get("moving_components")
    fixed_asset = geometry.get("fixed")
    if not (
        isinstance(fixed_components, list)
        and fixed_components
        and isinstance(moving_components, list)
        and moving_components
        and isinstance(fixed_asset, Mapping)
    ):
        return geometry, [dict(item) for item in samples]

    def components(items: list[Any], role: str) -> list[dict[str, Any]]:
        retained: list[dict[str, Any]] = []
        for index, raw in enumerate(items):
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            source = str(item.get("source") or item.get("derived") or "")
            item["solid_id"] = Path(source).stem or f"{role}_{index + 1}"
            retained.append(item)
        return retained

    family_geometry = dict(geometry)
    family_geometry["schema"] = "legacy_plain_two_body_local_v1"
    family_geometry["bodies"] = [
        {
            "body_id": "fixed_boundary",
            "fixed": True,
            "components": components(fixed_components, "fixed"),
        },
        {
            "body_id": "moving_group",
            "fixed": False,
            "components": components(moving_components, "moving"),
        },
    ]
    fixed_position = list(fixed_asset.get("body_origin_abs_m") or (0.0, 0.0, 0.0))
    normalized_samples: list[dict[str, Any]] = []
    for raw in samples:
        item = dict(raw)
        moving_pose = item.get("moving_pose")
        if not isinstance(moving_pose, Mapping):
            raise MechanismValidationError(
                "LEGACY_AUDIT_POSES_MISSING",
                "plain physical sample has no moving pose",
            )
        item["body_poses"] = {
            "fixed_boundary": {
                "position_abs_m": fixed_position,
                "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "moving_group": dict(moving_pose),
        }
        normalized_samples.append(item)
    return family_geometry, normalized_samples

def _trace_result(
    trace_path: Path,
    *,
    workspace_root: Path,
    nominal_interval_s: float,
) -> dict[str, Any]:
    trace = read_trace_json(trace_path)
    if trace.get("branch") != "physical":
        raise MechanismValidationError(
            "LEGACY_AUDIT_BRANCH_INVALID",
            f"expected a physical trace: {trace_path}",
        )
    geometry, samples = _normalized_audit_inputs(trace)
    selected, sampling = select_saved_trace_samples(
        samples,
        nominal_interval_s=nominal_interval_s,
    )
    audit = audit_family_collision_geometry(
        geometry,
        selected,
        workspace_root=workspace_root,
    )
    declarations = legacy_interface_declarations(trace)
    exclusions = disclosed_broad_collision_exclusions(trace)
    return {
        "status": (
            "SAMPLED_TRAJECTORY_AUDITED"
            if audit["passed"]
            else "SAMPLED_TRAJECTORY_FAILED"
        ),
        "passed": bool(audit["passed"]),
        "trace": relative_to_root(workspace_root, trace_path),
        "trace_sha256": sha256_file(trace_path),
        "package_id": trace.get("package_id"),
        "design_version": trace.get("design_version"),
        "capsule_id": trace.get("capsule_id"),
        "load_case_id": trace.get("load_case_id"),
        "mechanism_plugin_id": trace.get("mechanism_plugin_id"),
        "sampling": sampling,
        "physical_interface_declarations": list(declarations),
        "historical_broad_body_pair_collision_exclusions": exclusions,
        "geometry_legality": audit,
    }


def audit_legacy_run(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
    nominal_interval_s: float = 0.05,
    write_report: bool = True,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    try:
        run.relative_to(workspace)
    except ValueError as exc:
        raise MechanismValidationError(
            "LEGACY_AUDIT_RUN_OUTSIDE_WORKSPACE",
            "legacy run must remain inside the workspace",
        ) from exc
    if _is_excluded_run(run):
        raise MechanismValidationError(
            "LEGACY_AUDIT_RUN_EXCLUDED",
            "scissor-lift runs are explicitly excluded from this migration",
        )
    traces_root = run / "traces"
    trace_paths = sorted(traces_root.rglob("physical.trace.json"))
    if not trace_paths:
        raise MechanismValidationError(
            "LEGACY_AUDIT_TRACES_MISSING",
            f"no physical traces found under {traces_root}",
        )
    if len(trace_paths) > 64:
        raise MechanismValidationError(
            "LEGACY_AUDIT_TRACE_CAP_EXCEEDED",
            "legacy audit accepts at most 64 physical traces per run",
        )

    results: list[dict[str, Any]] = []
    for trace_path in trace_paths:
        try:
            result = _trace_result(
                trace_path,
                workspace_root=workspace,
                nominal_interval_s=nominal_interval_s,
            )
        except (MechanismValidationError, JsonDocumentError) as exc:
            metadata: Mapping[str, Any] = {}
            try:
                metadata = read_trace_json(trace_path)
            except JsonDocumentError:
                pass
            result = {
                "status": "SAMPLED_TRAJECTORY_INCONCLUSIVE",
                "passed": False,
                "trace": relative_to_root(workspace, trace_path),
                "trace_sha256": sha256_file(trace_path),
                "package_id": metadata.get("package_id"),
                "design_version": metadata.get("design_version"),
                "capsule_id": metadata.get("capsule_id"),
                "load_case_id": metadata.get("load_case_id"),
                "mechanism_plugin_id": metadata.get("mechanism_plugin_id"),
                "failure_code": getattr(exc, "code", "LEGACY_AUDIT_INPUT_INVALID"),
                "detail": str(exc),
            }
        results.append(result)

    failed = [item for item in results if item["status"] == "SAMPLED_TRAJECTORY_FAILED"]
    inconclusive = [
        item for item in results
        if item["status"] == "SAMPLED_TRAJECTORY_INCONCLUSIVE"
    ]
    if failed:
        status = "LEGACY_SAMPLED_PHYSICAL_AUDIT_FAILED"
    elif inconclusive:
        status = "LEGACY_SAMPLED_PHYSICAL_AUDIT_INCONCLUSIVE"
    else:
        status = "LEGACY_SAMPLED_PHYSICAL_AUDIT_PASSED"
    report = {
        "schema_version": "0.1",
        "status": status,
        "passed": status == "LEGACY_SAMPLED_PHYSICAL_AUDIT_PASSED",
        "run": relative_to_root(workspace, run),
        "policy": {
            "physical_simulation_rerun": False,
            "historical_certificate_modified": False,
            "historical_replacement_modified": False,
            "audit_source": "saved_physical_trace_and_hash_bound_step_colliders",
            "nominal_sample_interval_s": float(nominal_interval_s),
            "phase_boundaries_retained": True,
            "contact_transitions_retained": True,
            "simulation_collision_filters_ignored": True,
            "continuous_between_sample_collision_claim": False,
            "migration_certificate_issued": False,
        },
        "trace_count": len(results),
        "passed_trace_count": sum(item["status"] == "SAMPLED_TRAJECTORY_AUDITED" for item in results),
        "failed_trace_count": len(failed),
        "inconclusive_trace_count": len(inconclusive),
        "traces": results,
    }
    if write_report:
        destination = run / "migration" / REPORT_NAME
        write_json_atomic(destination, report)
        report["report"] = relative_to_root(workspace, destination)
        if not report["passed"]:
            try:
                from .penetration_visuals import build_penetration_visual_evidence

                visual = build_penetration_visual_evidence(
                    workspace_root=workspace,
                    run_dir=run,
                    audit_report=destination,
                )
                report["penetration_visual_evidence"] = {
                    "summary": visual.get("summary"),
                    "event_count": visual.get("event_count", 0),
                    "generation_failure_count": visual.get(
                        "generation_failure_count", 0
                    ),
                }
            except (MechanismValidationError, OSError, ValueError) as exc:
                report["penetration_visual_evidence"] = {
                    "summary": None,
                    "event_count": 0,
                    "generation_failure_count": 1,
                    "detail": str(exc),
                }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit saved poses from a historical non-scissor physical run."
    )
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--sample-interval-s", type=float, default=0.05)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = audit_legacy_run(
        workspace_root=args.workspace_root,
        run_dir=args.run,
        nominal_interval_s=args.sample_interval_s,
        write_report=not args.no_write,
    )
    print(report["status"])
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
