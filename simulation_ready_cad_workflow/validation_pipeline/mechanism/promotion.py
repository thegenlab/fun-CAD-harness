"""Contract for adopting a successful diagnostic lane replay as input work.

The candidate remains diagnostic evidence.  A later canonical run may use the
bound files as an optimization input, but this document never grants terminal
or certificate authority.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from ..runtime.models import ManifestError, QueueManifest
from .io import (
    MECHANISM_SOURCE_SHA256_ENV,
    canonical_sha256,
    ensure_inside,
    lane_execution_source_sha256,
    read_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)


LANE_REPLAY_PROMOTION_SCHEMA = (
    "simulation_ready_cad_lane_replay_promotion_candidate_v1"
)
LANE_REPLAY_PROMOTION_BINDING_SCHEMA = (
    "simulation_ready_cad_lane_replay_promotion_binding_v1"
)
LANE_REPLAY_RESULT_SCHEMA = "simulation_ready_cad_stage_replay_v1"
LANE_REPLAY_SOURCE_IDENTITY_KIND = "lane_execution_source_sha256"

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_CANDIDATE_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "authority",
        "canonical_effect",
        "stage",
        "source_run",
        "replay_output",
        "execution_root",
        "job_id",
        "capsule_id",
        "load_case_id",
        "source_identity",
        "outer_physical_audit",
        "bindings",
        "job_outputs",
    }
)
_REQUIRED_BINDINGS = frozenset(
    {
        "queue_manifest",
        "queue_state",
        "physical_request",
        "physical_preflight",
        "initial_functional_geometry",
        "plan",
        "load_case",
        "lane_result",
        "physical_trace",
    }
)


class LaneReplayPromotionError(ValueError):
    """A diagnostic replay is not a valid lane promotion candidate."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> LaneReplayPromotionError:
    return LaneReplayPromotionError(code, message)


def _initial_geometry_identity_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        has_assembly_bytes = isinstance(value.get("assembly_sha256"), str)
        return {
            key: _initial_geometry_identity_value(item)
            for key, item in sorted(value.items())
            if not (key == "assembly_file" and has_assembly_bytes)
        }
    if isinstance(value, (list, tuple)):
        return [_initial_geometry_identity_value(item) for item in value]
    return value


def initial_functional_geometry_identity_sha256(
    report: Mapping[str, Any],
) -> str:
    """Path-independent identity of one initial-geometry decision report.

    Bicycle-chain reports name the run-local assembly path alongside its exact
    byte digest.  A topology successor copies those same assembly bytes to a
    new package path, so the locator cannot decide whether physics inputs are
    equal; every digest, measurement, gate, and verdict remains load-bearing.
    """

    return canonical_sha256(
        {
            "schema_version": "initial_functional_geometry_identity_v1",
            "report": _initial_geometry_identity_value(report),
        }
    )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("LANE_PROMOTION_CONTRACT_INVALID", f"{label} must be an object")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            f"{label} must be a non-empty string",
        )
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            f"{label} must be a lowercase SHA-256 digest",
        )
    return value


def _workspace_path(workspace: Path, value: Any, label: str) -> Path:
    relative = _text(value, label)
    try:
        return ensure_inside(workspace, relative)
    except (OSError, ValueError) as exc:
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            f"{label} is not a workspace-relative path",
        ) from exc


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            f"{label} is outside its required root",
        ) from exc


def _file_binding(workspace: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise _fail(
            "LANE_PROMOTION_ARTIFACT_INVALID",
            f"promotion artifact is not a regular file: {resolved}",
        )
    return {
        "file": relative_to_root(workspace, resolved),
        "sha256": sha256_file(resolved),
    }


def _read_binding(
    workspace: Path,
    value: Any,
    label: str,
    *,
    root: Path | None = None,
) -> tuple[dict[str, str], Path]:
    binding = _mapping(value, label)
    if set(binding) != {"file", "sha256"}:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            f"{label} must contain exactly file and sha256",
        )
    path = _workspace_path(workspace, binding.get("file"), f"{label}.file")
    if root is not None:
        _inside(path, root, label)
    expected = _digest(binding.get("sha256"), f"{label}.sha256")
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected:
        raise _fail(
            "LANE_PROMOTION_ARTIFACT_CHANGED",
            f"{label} no longer matches its binding",
        )
    return {"file": str(binding["file"]), "sha256": expected}, path


def _command_argument(command: tuple[str, ...], flag: str) -> str | None:
    positions = [index for index, item in enumerate(command) if item == flag]
    if not positions:
        return None
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            f"promotion job has an invalid {flag} argument",
        )
    return command[positions[0] + 1]


def _expected_source_run_relative(
    workspace: Path,
    expected_source_run: str | Path | None,
) -> str | None:
    if expected_source_run is None:
        return None
    candidate = Path(expected_source_run)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        return candidate.resolve().relative_to(workspace).as_posix()
    except ValueError as exc:
        raise _fail(
            "LANE_PROMOTION_SOURCE_RUN_MISMATCH",
            "expected source run is outside the workspace",
        ) from exc


def validate_lane_replay_promotion_candidate(
    payload: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    expected_source_run: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize one candidate against all referenced bytes."""

    workspace = Path(workspace_root).resolve()
    candidate = dict(_mapping(payload, "promotion candidate"))
    if set(candidate) != _CANDIDATE_FIELDS:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            "promotion candidate fields do not match the v1 contract",
        )
    if (
        candidate.get("schema_version") != LANE_REPLAY_PROMOTION_SCHEMA
        or candidate.get("status") != "CANDIDATE_READY"
        or candidate.get("authority") != "diagnostic_only"
        or candidate.get("canonical_effect") is not False
        or candidate.get("stage") != "lane"
    ):
        raise _fail(
            "LANE_PROMOTION_AUTHORITY_INVALID",
            "promotion candidate must be a non-canonical diagnostic lane candidate",
        )

    source_run_relative = _text(candidate.get("source_run"), "source_run")
    source_run = _workspace_path(workspace, source_run_relative, "source_run")
    expected_relative = _expected_source_run_relative(
        workspace, expected_source_run
    )
    if expected_relative is not None and source_run_relative != expected_relative:
        raise _fail(
            "LANE_PROMOTION_SOURCE_RUN_MISMATCH",
            "promotion candidate belongs to another source run",
        )
    run_parts = Path(source_run_relative).parts
    if (
        len(run_parts) != 5
        or run_parts[:2] != (".runtime_runs", "projects")
        or run_parts[-2] != "runs"
        or not run_parts[-1].startswith("r")
    ):
        # Canonical paths end in projects/<project>/runs/rNN.  The explicit
        # shape check keeps a diagnostic directory from posing as source run.
        raise _fail(
            "LANE_PROMOTION_SOURCE_RUN_MISMATCH",
            "promotion source_run is not a canonical revision path",
        )
    if not source_run.is_dir():
        raise _fail(
            "LANE_PROMOTION_SOURCE_RUN_MISMATCH",
            "promotion source run no longer exists",
        )

    replay_output = _workspace_path(
        workspace, candidate.get("replay_output"), "replay_output"
    )
    diagnostics = (workspace / ".runtime_runs" / "_diagnostics").resolve()
    _inside(replay_output, diagnostics, "replay_output")
    execution_root = _workspace_path(
        workspace, candidate.get("execution_root"), "execution_root"
    )
    if execution_root != (replay_output / "artifacts").resolve():
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            "execution_root must be the replay artifacts directory",
        )

    job_id = _text(candidate.get("job_id"), "job_id")
    capsule_id = _text(candidate.get("capsule_id"), "capsule_id")
    load_case_id = _text(candidate.get("load_case_id"), "load_case_id")
    source_identity = _mapping(candidate.get("source_identity"), "source_identity")
    if set(source_identity) != {"kind", "sha256"}:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            "source_identity must contain exactly kind and sha256",
        )
    if source_identity.get("kind") != LANE_REPLAY_SOURCE_IDENTITY_KIND:
        raise _fail(
            "LANE_PROMOTION_SOURCE_IDENTITY_INVALID",
            "promotion source identity kind is invalid",
        )
    source_sha256 = _digest(
        source_identity.get("sha256"), "source_identity.sha256"
    )
    if source_sha256 != lane_execution_source_sha256():
        raise _fail(
            "LANE_PROMOTION_SOURCE_IDENTITY_CHANGED",
            "current lane execution source no longer matches the replay",
        )

    bindings = _mapping(candidate.get("bindings"), "bindings")
    if set(bindings) != _REQUIRED_BINDINGS:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            "promotion bindings do not match the v1 contract",
        )
    resolved: dict[str, tuple[dict[str, str], Path]] = {}
    for label in sorted(_REQUIRED_BINDINGS):
        if label in {"plan", "load_case"}:
            root = execution_root
        elif label == "initial_functional_geometry":
            root = source_run
        else:
            root = replay_output
        resolved[label] = _read_binding(
            workspace, bindings[label], f"bindings.{label}", root=root
        )

    manifest_path = resolved["queue_manifest"][1]
    if manifest_path != (replay_output / "queue_manifest.json").resolve():
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            "queue manifest is not the replay-owned manifest",
        )
    try:
        manifest = QueueManifest.from_file(manifest_path)
    except (ManifestError, OSError, ValueError) as exc:
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            f"promotion queue manifest is invalid: {exc}",
        ) from exc
    matching_jobs = [job for job in manifest.jobs if job.job_id == job_id]
    if len(manifest.jobs) != 1 or len(matching_jobs) != 1:
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            "promotion manifest must contain exactly the selected lane job",
        )
    job = matching_jobs[0]
    if (
        job.environment.get(MECHANISM_SOURCE_SHA256_ENV) != source_sha256
        or _command_argument(job.command, "--request")
        != resolved["physical_request"][0]["file"]
    ):
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            "promotion job does not bind the current source and physical request",
        )
    for input_file, expected_sha256 in job.input_files.items():
        input_path = _workspace_path(
            workspace, input_file, f"job input {input_file!r}"
        )
        if (
            not input_path.is_file()
            or input_path.is_symlink()
            or sha256_file(input_path) != expected_sha256
        ):
            raise _fail(
                "LANE_PROMOTION_ARTIFACT_CHANGED",
                f"promotion job input changed: {input_file}",
            )

    queue_state = read_json(resolved["queue_state"][1])
    queue_jobs = queue_state.get("jobs")
    queue_job = queue_jobs.get(job_id) if isinstance(queue_jobs, Mapping) else None
    if (
        queue_state.get("status") != "SUCCEEDED"
        or not isinstance(queue_job, Mapping)
        or queue_job.get("status") != "SUCCEEDED"
    ):
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            "promotion queue and selected job must both have succeeded",
        )

    expected_paths = {
        "queue_state": (replay_output / "queue" / "queue_state.json").resolve(),
        "physical_request": (
            replay_output / "requests" / job_id / "physical.json"
        ).resolve(),
        "physical_preflight": (
            execution_root / "results" / "physical_preflight.json"
        ).resolve(),
        "plan": (execution_root / "plans" / f"{capsule_id}.json").resolve(),
        "load_case": (
            execution_root
            / "loads"
            / capsule_id
            / f"{load_case_id}.json"
        ).resolve(),
        "physical_trace": (
            execution_root
            / "traces"
            / capsule_id
            / load_case_id
            / "physical.trace.json"
        ).resolve(),
        "lane_result": (
            execution_root
            / "results"
            / capsule_id
            / "lanes"
            / f"{load_case_id}.json"
        ).resolve(),
    }
    if any(resolved[label][1] != path for label, path in expected_paths.items()):
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            "promotion queue, request, trace, or lane path is not canonical",
        )

    request = read_json(resolved["physical_request"][1])
    if (
        request.get("plan_file") != resolved["plan"][0]["file"]
        or request.get("load_case_file") != resolved["load_case"][0]["file"]
        or request.get("output_file") != resolved["physical_trace"][0]["file"]
        or request.get("initial_functional_geometry_file")
        != resolved["initial_functional_geometry"][0]["file"]
        or request.get("initial_functional_geometry_sha256")
        != resolved["initial_functional_geometry"][0]["sha256"]
    ):
        raise _fail(
            "LANE_PROMOTION_INPUT_IDENTITY_INVALID",
            "physical request does not bind the candidate plan, load, and trace",
        )
    plan = read_json(resolved["plan"][1])
    load_case = read_json(resolved["load_case"][1])
    if (
        plan.get("capsule_id") != capsule_id
        or load_case.get("load_case_id") != load_case_id
    ):
        raise _fail(
            "LANE_PROMOTION_INPUT_IDENTITY_INVALID",
            "plan or load identity does not match the candidate",
        )

    lane = read_json(resolved["lane_result"][1])
    lane_artifacts = lane.get("artifacts")
    lane_trace = (
        lane_artifacts.get("physical_trace")
        if isinstance(lane_artifacts, Mapping)
        else None
    )
    if (
        lane.get("schema_version")
        != "simulation_ready_cad_mechanism_load_lane_v1"
        or lane.get("status") != "PASS"
        or lane.get("completed_phase") != "comparison"
        or lane.get("capsule_id") != capsule_id
        or lane.get("load_case_id") != load_case_id
        or lane.get("validator_source_sha256") != source_sha256
        or lane_trace != resolved["physical_trace"][0]
    ):
        raise _fail(
            "LANE_PROMOTION_LANE_RESULT_INVALID",
            "candidate lane result is not a source-current bound PASS",
        )

    outputs_value = candidate.get("job_outputs")
    if not isinstance(outputs_value, list):
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID", "job_outputs must be an array"
        )
    outputs = [
        _read_binding(
            workspace,
            value,
            f"job_outputs[{index}]",
            root=execution_root,
        )[0]
        for index, value in enumerate(outputs_value)
    ]
    if [item["file"] for item in outputs] != list(job.expected_outputs):
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            "candidate job outputs do not match the replay job",
        )
    if resolved["lane_result"][0] not in outputs:
        raise _fail(
            "LANE_PROMOTION_LANE_RESULT_INVALID",
            "candidate job outputs do not bind the lane result",
        )

    audit_value = _mapping(
        candidate.get("outer_physical_audit"), "outer_physical_audit"
    )
    if set(audit_value) != {"file", "sha256", "passed", "trace_sha256"}:
        raise _fail(
            "LANE_PROMOTION_CONTRACT_INVALID",
            "outer_physical_audit fields do not match the v1 contract",
        )
    audit_binding, audit_path = _read_binding(
        workspace,
        {"file": audit_value.get("file"), "sha256": audit_value.get("sha256")},
        "outer_physical_audit",
        root=execution_root,
    )
    expected_audit_path = (
        execution_root
        / "results"
        / capsule_id
        / "physical_trace_audit"
        / f"{load_case_id}.json"
    ).resolve()
    if audit_path != expected_audit_path:
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            "outer physical audit path is not canonical",
        )
    audit = read_json(audit_path)
    trace_sha256 = resolved["physical_trace"][0]["sha256"]
    if (
        audit_value.get("passed") is not True
        or audit_value.get("trace_sha256") != trace_sha256
        or audit.get("passed") is not True
        or audit.get("physical_trace_sha256") != trace_sha256
        or audit.get("audit_execution_environment")
        != "outer_cad_after_queue_barrier"
        or audit.get("worker_produced_audit_accepted") is not False
    ):
        raise _fail(
            "LANE_PROMOTION_OUTER_AUDIT_INVALID",
            "diagnostic outer physical audit is not a bound PASS",
        )

    preflight = read_json(resolved["physical_preflight"][1])
    rows = preflight.get("traces")
    matching_rows = [
        row
        for row in rows
        if isinstance(row, Mapping)
        and row.get("capsule_id") == capsule_id
        and row.get("load_case_id") == load_case_id
        and row.get("trace") == resolved["physical_trace"][0]["file"]
    ] if isinstance(rows, list) else []
    if (
        preflight.get("schema_version")
        != "simulation_ready_cad_physical_preflight_v2"
        or preflight.get("status") != "PASS"
        or preflight.get("passed") is not True
        or preflight.get("queue_status") != "SUCCEEDED"
        or preflight.get("trace_count") != 1
        or preflight.get("passed_trace_count") != 1
        or preflight.get("failed_trace_count") != 0
        or len(matching_rows) != 1
        or matching_rows[0].get("passed") is not True
        or matching_rows[0].get("trace_sha256") != trace_sha256
        or matching_rows[0].get("physical_trace_audit")
        != audit_binding["file"]
        or matching_rows[0].get("physical_trace_audit_sha256")
        != audit_binding["sha256"]
    ):
        raise _fail(
            "LANE_PROMOTION_OUTER_AUDIT_INVALID",
            "diagnostic aggregate physical preflight is not a bound PASS",
        )

    return {
        **candidate,
        "source_identity": {
            "kind": LANE_REPLAY_SOURCE_IDENTITY_KIND,
            "sha256": source_sha256,
        },
        "outer_physical_audit": {
            **audit_binding,
            "passed": True,
            "trace_sha256": trace_sha256,
        },
        "bindings": {
            label: resolved[label][0] for label in sorted(resolved)
        },
        "job_outputs": outputs,
    }


def write_lane_replay_promotion_candidate(
    destination: str | Path,
    *,
    workspace_root: str | Path,
    source_run: str | Path,
    replay_output: str | Path,
    execution_root: str | Path,
    job_id: str,
    queue_manifest: str | Path,
    queue_state: str | Path,
    physical_request: str | Path,
    lane_result: str | Path,
    outer_physical_audit: str | Path,
    physical_preflight: str | Path,
) -> dict[str, Any]:
    """Build, validate, and atomically write one promotion candidate."""

    workspace = Path(workspace_root).resolve()
    source = Path(source_run).resolve()
    replay = Path(replay_output).resolve()
    execution = Path(execution_root).resolve()
    manifest_path = Path(queue_manifest).resolve()
    queue_state_path = Path(queue_state).resolve()
    request_path = Path(physical_request).resolve()
    lane_path = Path(lane_result).resolve()
    audit_path = Path(outer_physical_audit).resolve()
    physical_preflight_path = Path(physical_preflight).resolve()

    manifest = QueueManifest.from_file(manifest_path)
    jobs = [job for job in manifest.jobs if job.job_id == job_id]
    if len(manifest.jobs) != 1 or len(jobs) != 1:
        raise _fail(
            "LANE_PROMOTION_JOB_INVALID",
            "promotion manifest must contain exactly the selected lane job",
        )
    job = jobs[0]
    request = read_json(request_path)
    plan_path = _workspace_path(workspace, request.get("plan_file"), "plan_file")
    load_path = _workspace_path(
        workspace, request.get("load_case_file"), "load_case_file"
    )
    trace_path = _workspace_path(
        workspace, request.get("output_file"), "output_file"
    )
    initial_geometry_path = _workspace_path(
        workspace,
        request.get("initial_functional_geometry_file"),
        "initial_functional_geometry_file",
    )
    initial_geometry_binding = _file_binding(workspace, initial_geometry_path)
    if (
        request.get("initial_functional_geometry_sha256")
        != initial_geometry_binding["sha256"]
    ):
        raise _fail(
            "LANE_PROMOTION_INPUT_IDENTITY_INVALID",
            "initial functional geometry no longer matches the replay request",
        )
    lane = read_json(lane_path)
    audit = read_json(audit_path)
    source_sha256 = _digest(
        job.environment.get(MECHANISM_SOURCE_SHA256_ENV),
        "job validator source identity",
    )
    trace_binding = _file_binding(workspace, trace_path)
    candidate = {
        "schema_version": LANE_REPLAY_PROMOTION_SCHEMA,
        "status": "CANDIDATE_READY",
        "authority": "diagnostic_only",
        "canonical_effect": False,
        "stage": "lane",
        "source_run": relative_to_root(workspace, source),
        "replay_output": relative_to_root(workspace, replay),
        "execution_root": relative_to_root(workspace, execution),
        "job_id": job_id,
        "capsule_id": lane.get("capsule_id"),
        "load_case_id": lane.get("load_case_id"),
        "source_identity": {
            "kind": LANE_REPLAY_SOURCE_IDENTITY_KIND,
            "sha256": source_sha256,
        },
        "outer_physical_audit": {
            **_file_binding(workspace, audit_path),
            "passed": audit.get("passed") is True,
            "trace_sha256": trace_binding["sha256"],
        },
        "bindings": {
            "queue_manifest": _file_binding(workspace, manifest_path),
            "queue_state": _file_binding(workspace, queue_state_path),
            "physical_request": _file_binding(workspace, request_path),
            "physical_preflight": _file_binding(
                workspace, physical_preflight_path
            ),
            "initial_functional_geometry": initial_geometry_binding,
            "plan": _file_binding(workspace, plan_path),
            "load_case": _file_binding(workspace, load_path),
            "lane_result": _file_binding(workspace, lane_path),
            "physical_trace": trace_binding,
        },
        "job_outputs": [
            _file_binding(workspace, ensure_inside(workspace, relative))
            for relative in job.expected_outputs
        ],
    }
    normalized = validate_lane_replay_promotion_candidate(
        candidate,
        workspace_root=workspace,
        expected_source_run=source,
    )
    destination_path = Path(destination).resolve()
    if destination_path != (replay / "promotion_candidate.json").resolve():
        raise _fail(
            "LANE_PROMOTION_PATH_INVALID",
            "candidate must be written as replay_output/promotion_candidate.json",
        )
    write_json_atomic(destination_path, normalized)
    return normalized


def _promotion_metadata(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: candidate[key]
        for key in (
            "source_run",
            "execution_root",
            "job_id",
            "capsule_id",
            "load_case_id",
            "source_identity",
            "outer_physical_audit",
        )
    }


def load_lane_replay_promotion_report(
    path: str | Path,
    *,
    workspace_root: str | Path,
    expected_source_run: str | Path | None = None,
) -> dict[str, Any]:
    """Strictly load a replay result and its referenced promotion candidate."""

    workspace = Path(workspace_root).resolve()
    report_path = Path(path)
    if not report_path.is_absolute():
        report_path = workspace / report_path
    report_path = report_path.resolve()
    diagnostics = (workspace / ".runtime_runs" / "_diagnostics").resolve()
    _inside(report_path, diagnostics, "replay_result")
    report = read_json(report_path)
    if (
        report.get("schema_version") != LANE_REPLAY_RESULT_SCHEMA
        or report.get("replay_status") != "COMPLETED"
        or report.get("authority") != "diagnostic_only"
        or report.get("canonical_effect") is not False
        or report.get("stage") != "lane"
        or _mapping(report.get("observed"), "observed").get("status") != "PASS"
    ):
        raise _fail(
            "LANE_PROMOTION_REPORT_INVALID",
            "replay result is not a completed non-canonical lane PASS",
        )
    expected_relative = _expected_source_run_relative(
        workspace, expected_source_run
    )
    source_run = _text(report.get("source_run"), "report.source_run")
    if expected_relative is not None and source_run != expected_relative:
        raise _fail(
            "LANE_PROMOTION_SOURCE_RUN_MISMATCH",
            "replay result belongs to another source run",
        )
    output_relative = relative_to_root(workspace, report_path.parent)
    if report.get("output") != output_relative:
        raise _fail(
            "LANE_PROMOTION_REPORT_INVALID",
            "replay result output does not identify its containing directory",
        )
    promotion = _mapping(report.get("promotion"), "promotion")
    candidate_relative = _text(
        report.get("promotion_candidate"), "promotion_candidate"
    )
    if promotion.get("candidate") != candidate_relative:
        raise _fail(
            "LANE_PROMOTION_REPORT_INVALID",
            "replay promotion references disagree",
        )
    candidate_path = _workspace_path(
        workspace, candidate_relative, "promotion.candidate"
    )
    if candidate_path != (report_path.parent / "promotion_candidate.json").resolve():
        raise _fail(
            "LANE_PROMOTION_REPORT_INVALID",
            "replay result must reference its sibling promotion candidate",
        )
    candidate = validate_lane_replay_promotion_candidate(
        read_json(candidate_path),
        workspace_root=workspace,
        expected_source_run=(expected_source_run or source_run),
    )
    metadata = _promotion_metadata(candidate)
    expected_promotion = {
        "status": "CANDIDATE_READY",
        "candidate": candidate_relative,
        **metadata,
    }
    if dict(promotion) != expected_promotion:
        raise _fail(
            "LANE_PROMOTION_REPORT_INVALID",
            "replay promotion metadata does not match the candidate",
        )
    observed = _mapping(report.get("observed"), "observed")
    if observed.get("promotion") != expected_promotion:
        raise _fail(
            "LANE_PROMOTION_REPORT_INVALID",
            "observed promotion metadata does not match the candidate",
        )

    binding = {
        "schema_version": LANE_REPLAY_PROMOTION_BINDING_SCHEMA,
        "replay_result": _file_binding(workspace, report_path),
        "candidate": _file_binding(workspace, candidate_path),
        "metadata": metadata,
    }
    return {"candidate": candidate, "binding": binding}


def load_lane_promotion_candidate(
    workspace_root: str | Path,
    binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve a saved promotion binding, returning ``None`` on cache miss."""

    try:
        workspace = Path(workspace_root).resolve()
        saved = _mapping(binding, "lane replay promotion binding")
        if (
            saved.get("schema_version")
            != LANE_REPLAY_PROMOTION_BINDING_SCHEMA
            or set(saved) != {
                "schema_version",
                "replay_result",
                "candidate",
                "metadata",
            }
        ):
            raise _fail(
                "LANE_PROMOTION_BINDING_INVALID",
                "saved lane replay promotion binding is invalid",
            )
        report_binding, report_path = _read_binding(
            workspace, saved.get("replay_result"), "replay_result"
        )
        candidate_binding, _candidate_path = _read_binding(
            workspace, saved.get("candidate"), "candidate"
        )
        metadata = dict(_mapping(saved.get("metadata"), "metadata"))
        loaded = load_lane_replay_promotion_report(
            report_path,
            workspace_root=workspace,
            expected_source_run=metadata.get("source_run"),
        )
        if (
            loaded["binding"]["replay_result"] != report_binding
            or loaded["binding"]["candidate"] != candidate_binding
            or loaded["binding"]["metadata"] != metadata
        ):
            raise _fail(
                "LANE_PROMOTION_BINDING_CHANGED",
                "saved lane replay promotion metadata changed",
            )
        return loaded["candidate"]
    except (LaneReplayPromotionError, ManifestError, OSError, TypeError, ValueError):
        return None


__all__ = [
    "LANE_REPLAY_PROMOTION_BINDING_SCHEMA",
    "LANE_REPLAY_PROMOTION_SCHEMA",
    "LaneReplayPromotionError",
    "initial_functional_geometry_identity_sha256",
    "load_lane_promotion_candidate",
    "load_lane_replay_promotion_report",
    "validate_lane_replay_promotion_candidate",
    "write_lane_replay_promotion_candidate",
]
