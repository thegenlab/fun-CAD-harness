from __future__ import annotations

"""Run one complete mechanism load-case pipeline.

Each queue job owns one load case and performs its physical simulation,
optional ideal-dynamics simulation, and ideal-reference comparisons in order
before returning a single result.  Saved-pose CAD/FCL legality is deliberately
performed by the outer CAD process after the queue barrier.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    failure_envelope_from_exception,
    make_failure_envelope,
    make_plugin_failure_envelope,
    unclassified_failure_envelope,
    validate_failure_envelope,
)
from .comparison import compare_composite_lane_ideal_dynamics_trace_files
from .ideal_comparison import compare_physical_to_ideal
from .ideal_reference import generate_ideal_reference_trace
from .io import (
    canonical_sha256,
    lane_execution_source_sha256,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
    write_trace_json_atomic,
)
from .physical_ideal_comparison import (
    compare_composite_lane_physical_behavior_to_ideal_reference,
)
from .worker import (
    _verify_validator_source_binding,
    load_execution_request,
    run_request,
)


LANE_SCHEMA = "simulation_ready_cad_mechanism_load_lane_v1"

_ADAPTER_BRANCH_CONTRACT_REASONS = frozenset(
    {
        "physical_branch_contains_internal_ideal_joint",
        "physical_contact_pair_is_disabled",
    }
)


def _lane_paths(execution: Any, workspace: Path) -> dict[str, Path]:
    trace_root = execution.output_file.parent.resolve()
    if (
        trace_root.name != execution.load_case.load_case_id
        or trace_root.parent.name != execution.plan.capsule_id
        or trace_root.parent.parent.name != "traces"
    ):
        raise ValueError(
            "physical output does not use the canonical "
            "traces/<capsule>/<load>/physical.trace.json layout"
        )
    output = trace_root.parent.parent.parent.resolve()
    output.relative_to(workspace)
    result_root = output / "results" / execution.plan.capsule_id
    load_id = execution.load_case.load_case_id
    return {
        "physical_trace": execution.output_file.resolve(),
        "physical_trace_audit": (
            result_root / "physical_trace_audit" / f"{load_id}.json"
        ),
        "ideal_reference_trace": trace_root / "ideal_reference.trace.json",
        "ideal_dynamics_trace": trace_root / "ideal_dynamics.trace.json",
        "ideal_dynamics_comparison": (
            result_root / "ideal_dynamics_comparison" / f"{load_id}.json"
        ),
        "ideal_reference_comparison": (
            result_root / "ideal_reference_comparison" / f"{load_id}.json"
        ),
        "comparison": result_root / "comparison" / f"{load_id}.json",
        "lane_result": result_root / "lanes" / f"{load_id}.json",
    }


def _artifact_bindings(
    paths: dict[str, Path],
    *,
    workspace: Path,
    include_ideal_dynamics: bool,
) -> dict[str, dict[str, str]]:
    excluded = {"lane_result", "physical_trace_audit"}
    if not include_ideal_dynamics:
        excluded.update(
            {"ideal_dynamics_trace", "ideal_dynamics_comparison"}
        )
    return {
        label: {
            "file": relative_to_root(workspace, path),
            "sha256": sha256_file(path),
        }
        for label, path in sorted(paths.items())
        if label not in excluded and path.is_file()
    }


def _lane_result(
    execution: Any,
    paths: dict[str, Path],
    *,
    workspace: Path,
    status: str,
    completed_phase: str,
    include_ideal_dynamics: bool,
    failure_code: str | None = None,
    detail: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": LANE_SCHEMA,
        "status": status,
        "completed_phase": completed_phase,
        "terminal_authority": False,
        "certificate_authority": False,
        "validator_source_sha256": lane_execution_source_sha256(),
        "package_binding_sha256": execution.plan.package_binding_sha256,
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "plan_sha256": canonical_sha256(execution.plan.to_dict()),
        "load_case_sha256": canonical_sha256(
            execution.load_case.to_dict()
        ),
        "artifacts": _artifact_bindings(
            paths,
            workspace=workspace,
            include_ideal_dynamics=include_ideal_dynamics,
        ),
    }
    if failure_code is not None:
        payload["failure_code"] = failure_code
    if detail is not None:
        payload["detail"] = detail
    if failure is not None:
        payload["failure"] = validate_failure_envelope(failure)
    return payload


def _upstream_failure(
    payload: Mapping[str, Any],
    *,
    code: str,
    message: str,
) -> dict[str, Any]:
    carried = payload.get("failure")
    if isinstance(carried, Mapping):
        return validate_failure_envelope(carried)
    return unclassified_failure_envelope(
        code,
        source_component="mechanism.lane_worker",
        message=message,
    )


def _comparison_failure(
    code: str,
    reasons: Sequence[Any],
    *,
    physical_trace: Mapping[str, Any],
    progress_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_reasons = [str(item) for item in reasons]
    message = ", ".join(normalized_reasons) or "comparison failed"
    adapter_contract_failed = any(
        item in _ADAPTER_BRANCH_CONTRACT_REASONS for item in normalized_reasons
    )
    plugin_id = str(
        physical_trace.get("mechanism_plugin_id")
        or physical_trace.get("plugin_id")
        or ""
    ).strip()
    if adapter_contract_failed:
        wrench_audit = physical_trace.get("wrench_audit")
        registration = (
            wrench_audit.get("capability_registration")
            if isinstance(wrench_audit, Mapping)
            else None
        )
        return make_plugin_failure_envelope(
            code,
            registration=(registration if isinstance(registration, Mapping) else None),
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            source_component="mechanism.lane_worker.comparison",
            message=message,
            evidence_validity=EvidenceValidity.INVALID_MEASUREMENT,
            causes=[
                {"code": str(item), "message": str(item)} for item in reasons
            ],
        )
    return make_failure_envelope(
        code,
        FailureOwner.DESIGN_PACKAGE,
        source_component="mechanism.lane_worker.comparison",
        message=message,
        evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
        causes=[
            {
                "code": str(item),
                "message": str(item),
            }
            for item in reasons
        ],
        details=(
            {"progress_observation": dict(progress_observation)}
            if progress_observation is not None
            else None
        ),
    )


def _comparison_progress_observation(
    execution: Any,
    reasons: Sequence[Any],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "capsule_id": execution.plan.capsule_id,
        "load_case_id": execution.load_case.load_case_id,
        "program": execution.load_case.program,
        "failure_reasons": [str(item) for item in reasons],
        "physical_metrics": comparison.get("physical_metrics"),
        "ideal_reference_metrics": comparison.get(
            "ideal_reference_metrics"
        ),
        "ideal_reference_threshold": comparison.get(
            "ideal_reference_threshold"
        ),
    }
    thresholds = comparison.get("thresholds")
    if isinstance(thresholds, Mapping) and thresholds:
        observation["thresholds"] = dict(thresholds)
    return observation


def run_lane(
    request_file: str | Path,
    *,
    workspace_root: str | Path,
    ideal_dynamics_request_file: str | Path | None = None,
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    _verify_validator_source_binding()
    execution = load_execution_request(
        request_file,
        workspace_root=workspace,
    )
    if execution.artifact_branch != "physical":
        raise ValueError("a load pipeline requires a physical request")
    paths = _lane_paths(execution, workspace)
    ideal_dynamics_execution = None
    if ideal_dynamics_request_file is not None:
        ideal_dynamics_execution = load_execution_request(
            ideal_dynamics_request_file,
            workspace_root=workspace,
        )
        if ideal_dynamics_execution.artifact_branch != "ideal_dynamics":
            raise ValueError(
                "the optional load-pipeline request must be ideal_dynamics"
            )
        if (
            ideal_dynamics_execution.plan.to_dict()
            != execution.plan.to_dict()
            or ideal_dynamics_execution.load_case.to_dict()
            != execution.load_case.to_dict()
            or ideal_dynamics_execution.output_file.resolve()
            != paths["ideal_dynamics_trace"]
        ):
            raise ValueError(
                "physical and ideal-dynamics requests describe different "
                "load pipelines"
            )
    completed_phase = "request_validation"
    try:
        physical = run_request(request_file, workspace_root=workspace)
        completed_phase = "physical"
        if physical.get("status") != "COMPLETED":
            error = physical.get("error")
            code = (
                str(error.get("code"))
                if isinstance(error, dict) and error.get("code")
                else "PHYSICAL_TRACE_NOT_COMPLETED"
            )
            detail = (
                str(error.get("message"))
                if isinstance(error, dict) and error.get("message")
                else "physical worker did not produce a completed trace"
            )
            result = _lane_result(
                execution,
                paths,
                workspace=workspace,
                status="FAIL",
                completed_phase=completed_phase,
                include_ideal_dynamics=(ideal_dynamics_execution is not None),
                failure_code=code,
                detail=detail,
                failure=_upstream_failure(
                    physical, code=code, message=detail
                ),
            )
            write_json_atomic(paths["lane_result"], result)
            return result

        # The atomic writer has completed before run_request returns. Release
        # its large in-memory object before loading the persisted trace that is
        # authoritative for every downstream comparison. Keeping both copies
        # alive made full-cadence lanes consume roughly twice their trace RAM.
        del physical
        physical = read_trace_json(paths["physical_trace"])
        physical_trace_sha256 = sha256_file(paths["physical_trace"])

        if ideal_dynamics_execution is not None:
            ideal_dynamics = run_request(
                ideal_dynamics_request_file,
                workspace_root=workspace,
            )
            completed_phase = "ideal_dynamics"
            if ideal_dynamics.get("status") != "COMPLETED":
                error = ideal_dynamics.get("error")
                code = (
                    str(error.get("code"))
                    if isinstance(error, dict) and error.get("code")
                    else "IDEAL_DYNAMICS_TRACE_NOT_COMPLETED"
                )
                detail = (
                    str(error.get("message"))
                    if isinstance(error, dict) and error.get("message")
                    else "ideal-dynamics worker did not complete the trace"
                )
                result = _lane_result(
                    execution,
                    paths,
                    workspace=workspace,
                    status="FAIL",
                    completed_phase=completed_phase,
                    include_ideal_dynamics=True,
                    failure_code=code,
                    detail=detail,
                    failure=_upstream_failure(
                        ideal_dynamics, code=code, message=detail
                    ),
                )
                write_json_atomic(paths["lane_result"], result)
                return result

            # The file-based comparison below parses the authoritative output.
            # Do not retain a second full trace until then.
            del ideal_dynamics
            ideal_dynamics_trace_sha256 = sha256_file(
                paths["ideal_dynamics_trace"]
            )
            ideal_dynamics_comparison = (
                compare_composite_lane_ideal_dynamics_trace_files(
                    execution.plan,
                    execution.load_case,
                    paths["physical_trace"],
                    paths["ideal_dynamics_trace"],
                    workspace_root=workspace,
                )
            )
            ideal_dynamics_comparison["evidence_bindings"] = {
                "physical_trace_sha256": physical_trace_sha256,
                "ideal_dynamics_trace_sha256": (
                    ideal_dynamics_trace_sha256
                ),
            }
            write_json_atomic(
                paths["ideal_dynamics_comparison"],
                ideal_dynamics_comparison,
            )
            completed_phase = "ideal_dynamics_comparison"
            if ideal_dynamics_comparison.get("verdict") != "PASS":
                result = _lane_result(
                    execution,
                    paths,
                    workspace=workspace,
                    status="FAIL",
                    completed_phase=completed_phase,
                    include_ideal_dynamics=True,
                    failure_code="IDEAL_DYNAMICS_COMPARISON_FAILED",
                    detail=", ".join(
                        str(item)
                        for item in ideal_dynamics_comparison.get(
                            "failure_reasons"
                        ) or ()
                    ),
                    failure=_comparison_failure(
                        "IDEAL_DYNAMICS_COMPARISON_FAILED",
                        list(
                            ideal_dynamics_comparison.get("failure_reasons")
                            or ()
                        ),
                        physical_trace=physical,
                    ),
                )
                write_json_atomic(paths["lane_result"], result)
                return result

        ideal = generate_ideal_reference_trace(
            execution.plan,
            execution.load_case,
            physical,
        )
        write_trace_json_atomic(paths["ideal_reference_trace"], ideal)
        ideal_reference_trace_sha256 = sha256_file(
            paths["ideal_reference_trace"]
        )
        completed_phase = "ideal_reference"

        ideal_comparison = compare_physical_to_ideal(
            execution.plan,
            execution.load_case,
            physical,
            ideal,
        )
        ideal_comparison["evidence_bindings"] = {
            "physical_trace_sha256": physical_trace_sha256,
            "ideal_reference_trace_sha256": (
                ideal_reference_trace_sha256
            ),
        }
        write_json_atomic(
            paths["ideal_reference_comparison"],
            ideal_comparison,
        )
        comparison = (
            compare_composite_lane_physical_behavior_to_ideal_reference(
                execution.plan,
                execution.load_case,
                physical,
                ideal_comparison,
                workspace_root=str(workspace),
            )
        )
        comparison["evidence_bindings"] = {
            "physical_trace_sha256": physical_trace_sha256,
            "ideal_reference_trace_sha256": (
                ideal_reference_trace_sha256
            ),
        }
        write_json_atomic(paths["comparison"], comparison)
        completed_phase = "comparison"
        passed = (
            ideal_comparison.get("verdict") == "PASS"
            and comparison.get("verdict") == "PASS"
        )
        comparison_reasons = (
            list(ideal_comparison.get("failure_reasons") or ())
            + list(comparison.get("failure_reasons") or ())
        )
        result = _lane_result(
            execution,
            paths,
            workspace=workspace,
            status="PASS" if passed else "FAIL",
            completed_phase=completed_phase,
            include_ideal_dynamics=(ideal_dynamics_execution is not None),
            failure_code=(None if passed else "LOAD_COMPARISON_FAILED"),
            detail=(
                None
                if passed
                else ", ".join(
                    str(item)
                    for item in comparison_reasons
                )
            ),
            failure=(
                None
                if passed
                else _comparison_failure(
                    "LOAD_COMPARISON_FAILED",
                    comparison_reasons,
                    physical_trace=physical,
                    progress_observation=_comparison_progress_observation(
                        execution,
                        comparison_reasons,
                        comparison,
                    ),
                )
            ),
        )
        write_json_atomic(paths["lane_result"], result)
        return result
    except Exception as exc:
        failure = (
            failure_envelope_from_exception(
                exc,
                default_owner=FailureOwner.UNCLASSIFIED,
                source_component="mechanism.lane_worker",
                evidence_validity=EvidenceValidity.NOT_APPLICABLE,
            )
        )
        result = _lane_result(
            execution,
            paths,
            workspace=workspace,
            status="FAIL",
            completed_phase=completed_phase,
            include_ideal_dynamics=(ideal_dynamics_execution is not None),
            failure_code=str(
                getattr(exc, "code", type(exc).__name__)
            ),
            detail=str(exc),
            failure=failure,
        )
        write_json_atomic(paths["lane_result"], result)
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one complete mechanism load-case pipeline."
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--ideal-dynamics-request", type=Path)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_lane(
            args.request,
            workspace_root=args.workspace_root,
            ideal_dynamics_request_file=args.ideal_dynamics_request,
        )
    except Exception as exc:
        failure = failure_envelope_from_exception(
            exc,
            default_owner=FailureOwner.UNCLASSIFIED,
            source_component="mechanism.lane_worker.request",
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        )
        print(
            json.dumps(
                {
                    "status": "INVALID_REQUEST",
                    "error_code": getattr(exc, "code", type(exc).__name__),
                    "message": str(exc),
                    "failure": failure,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
