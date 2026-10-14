from __future__ import annotations

import argparse
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..mechanism.io import (
    MECHANISM_SOURCE_SHA256_ENV,
    lane_execution_source_sha256,
    read_json,
    sha256_file,
    validator_import_root,
    write_json_atomic,
)


class SingleLaneBuildError(ValueError):
    """A diagnostic queue cannot be cropped without preserving its bindings."""


def _inside(workspace: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise SingleLaneBuildError(f"{label} is outside workspace: {resolved}") from exc
    return resolved


def _request_for_job(
    workspace: Path,
    job: dict[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    command = job.get("command")
    if not isinstance(command, list) or "--request" not in command:
        return None
    request_index = command.index("--request") + 1
    if request_index >= len(command) or not isinstance(command[request_index], str):
        raise SingleLaneBuildError("source job has an invalid --request argument")
    relative = command[request_index]
    request_path = _inside(workspace, workspace / relative, "source request")
    return relative, read_json(request_path)


def _replace_prefix(value: str, *, old: str, new: str, label: str) -> str:
    if value == old:
        return new
    prefix = old + "/"
    if value.startswith(prefix):
        return new + value[len(old) :]
    raise SingleLaneBuildError(f"{label} is not rooted in the source plan: {value!r}")


def build_single_lane_smoke(
    *,
    workspace_root: str | Path,
    source_plan_dir: str | Path,
    output_dir: str | Path,
    load_case_id: str,
    max_wall_time_seconds: float = 180.0,
) -> dict[str, Any]:
    """Crop one planned lane and bind it to the live execution source.

    The source digest is intentionally not an argument: callers cannot inject
    or reuse a remembered value. It is derived from the imported Harness tree
    that will also be placed first on the child worker's PYTHONPATH.
    """

    if not isinstance(load_case_id, str) or not load_case_id:
        raise SingleLaneBuildError("load_case_id must be a non-empty string")
    if not 30.0 <= float(max_wall_time_seconds) <= 180.0:
        raise SingleLaneBuildError(
            "max_wall_time_seconds must stay in [30, 180]"
        )

    workspace = Path(workspace_root).resolve()
    source = _inside(
        workspace, Path(source_plan_dir), "source plan directory"
    )
    output = _inside(workspace, Path(output_dir), "output directory")
    harness = validator_import_root().resolve()
    if not workspace.is_dir():
        raise SingleLaneBuildError(f"workspace root is missing: {workspace}")
    if not source.is_dir() or not harness.is_dir():
        raise SingleLaneBuildError("source plan or imported Harness is missing")
    if output.exists():
        raise SingleLaneBuildError(f"output directory already exists: {output}")
    if not output.parent.is_dir():
        raise SingleLaneBuildError(f"output parent is missing: {output.parent}")

    source_manifest = read_json(source / "physical_queue_manifest.json")
    jobs = source_manifest.get("jobs")
    if not isinstance(jobs, list):
        raise SingleLaneBuildError("source queue jobs must be an array")

    matches: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for candidate in jobs:
        if not isinstance(candidate, dict):
            continue
        request_entry = _request_for_job(workspace, candidate)
        if request_entry is None:
            continue
        request_relative, request = request_entry
        load_case_file = request.get("load_case_file")
        if (
            isinstance(load_case_file, str)
            and Path(load_case_file).stem == load_case_id
        ):
            matches.append((candidate, request_relative, request))
    if len(matches) != 1:
        raise SingleLaneBuildError(
            "source queue does not contain exactly one requested lane"
        )

    source_job, old_request_relative, request = matches[0]
    job = deepcopy(source_job)
    command = list(job["command"])
    request_index = command.index("--request") + 1
    plan_file = request.get("plan_file")
    if not isinstance(plan_file, str):
        raise SingleLaneBuildError("request has no plan_file")
    capsule_id = Path(plan_file).stem
    if not capsule_id:
        raise SingleLaneBuildError("request has no capsule identity")

    output_relative = output.relative_to(workspace).as_posix()
    source_relative = source.relative_to(workspace).as_posix()
    new_request_relative = (
        f"{output_relative}/requests/{capsule_id}/{load_case_id}/physical.json"
    )
    request = deepcopy(request)
    request["output_file"] = (
        f"{output_relative}/traces/{capsule_id}/{load_case_id}/"
        "physical.trace.json"
    )
    new_request = workspace / new_request_relative
    write_json_atomic(new_request, request)

    command[request_index] = new_request_relative
    job["job_id"] = f"diagnostic-{capsule_id}-{load_case_id}"
    job["command"] = command
    job["timeout_seconds"] = float(max_wall_time_seconds)
    job["estimated_wall_time_seconds"] = min(
        float(
            job.get("estimated_wall_time_seconds")
            or max_wall_time_seconds
        ),
        float(max_wall_time_seconds),
    )
    outputs = job.get("expected_outputs")
    if not isinstance(outputs, list):
        raise SingleLaneBuildError("source job expected_outputs must be an array")
    job["expected_outputs"] = [
        _replace_prefix(
            item,
            old=source_relative,
            new=output_relative,
            label="expected output",
        )
        for item in outputs
        if isinstance(item, str)
    ]
    if len(job["expected_outputs"]) != len(outputs):
        raise SingleLaneBuildError("source job expected_outputs contains non-strings")

    live_source = lane_execution_source_sha256()
    environment = dict(job.get("environment") or {})
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(harness),
            "PYTHONUNBUFFERED": "1",
            MECHANISM_SOURCE_SHA256_ENV: live_source,
        }
    )
    job["environment"] = environment

    inputs = job.get("input_files")
    if not isinstance(inputs, dict) or old_request_relative not in inputs:
        raise SingleLaneBuildError(
            "source request is not content-bound in the queue"
        )
    inputs = dict(inputs)
    inputs.pop(old_request_relative)
    inputs[new_request_relative] = sha256_file(new_request)
    job["input_files"] = dict(sorted(inputs.items()))

    manifest = deepcopy(source_manifest)
    manifest["run_name"] = f"diagnostic-{capsule_id}-{load_case_id}"
    manifest["jobs"] = [job]
    manifest["queue_policy"] = {
        "max_wall_time_seconds": float(max_wall_time_seconds),
        "stop_on_failure": True,
        "reuse_successful": False,
        "max_timeout_resume_attempts": 0,
        "max_parallel_jobs": 1,
    }

    if lane_execution_source_sha256() != live_source:
        raise SingleLaneBuildError(
            "validator source changed while the single-lane queue was built"
        )

    report: dict[str, Any] = {
        "schema_version": "single_lane_diagnostic_build_v2",
        "status": "PASS",
        "capsule_id": capsule_id,
        "load_case_id": load_case_id,
        "job_count": 1,
        "request_file": new_request_relative,
        "request_sha256": sha256_file(new_request),
        "plan_file": request["plan_file"],
        "plan_sha256": request["plan_sha256"],
        "load_case_file": request["load_case_file"],
        "load_case_sha256": request["load_case_sha256"],
        "package_binding_sha256": request["package_binding_sha256"],
        "source_identity_kind": "lane_execution_source_sha256",
        "validator_source_sha256": live_source,
        "max_wall_time_seconds": float(max_wall_time_seconds),
        "reuse_successful": False,
        "max_timeout_resume_attempts": 0,
        "pychrono_started": False,
    }
    marker = {
        "schema_version": "diagnostic_only_marker_v2",
        "authority": "diagnostic_only",
        "canonical_certificate_allowed": False,
        "package_binding_sha256": request["package_binding_sha256"],
        "source_plan_file": request["plan_file"],
        "source_identity_kind": "lane_execution_source_sha256",
        "validator_source_sha256": live_source,
        "max_wall_time_seconds": float(max_wall_time_seconds),
        "reuse_successful": False,
        "max_timeout_resume_attempts": 0,
    }
    write_json_atomic(output / "physical_queue_manifest.json", manifest)
    write_json_atomic(output / "queue_manifest.json", manifest)
    write_json_atomic(output / "diagnostic_only.json", marker)
    write_json_atomic(output / "single_lane_build_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build-single-lane-smoke",
        description=(
            "Crop one planned physical lane and bind it to the live Harness "
            "source; the source digest cannot be supplied by the caller."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--source-plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--load-case-id", required=True)
    parser.add_argument(
        "--max-wall-time-seconds", type=float, default=180.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        build_single_lane_smoke(
            workspace_root=args.workspace_root,
            source_plan_dir=args.source_plan_dir,
            output_dir=args.output_dir,
            load_case_id=args.load_case_id,
            max_wall_time_seconds=args.max_wall_time_seconds,
        )
    except (OSError, KeyError, TypeError, SingleLaneBuildError) as exc:
        raise SystemExit(f"build-single-lane-smoke: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
