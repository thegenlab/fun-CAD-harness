from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import mimetypes
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from ..mechanism.io import canonical_sha256
from ..contracts.observability import LOOP_STATE_SCHEMA
from ..run_layout import (
    CANONICAL_TERMINAL_STATUSES,
    parse_canonical_run,
)
from .legacy_io import DashboardReadError, read_stable_json_object
from .path_resolution import resolve_workspace_reference
from .physics_evidence import upgrade_replay_for_presentation
from .project_usage import load_project_usage
from .live_workflow import ENDED_WITHOUT_PASS, read_live_project, is_cold_start_run


STATUS_LABELS = {
    "passed": "通过",
    "evidence_complete": "证据齐全",
    "evidence_incomplete": "证据不全",
    "warning": "警告",
    "preparing": "准备中（无活动 worker）",
    "running": "进行中",
    "awaiting_review": "等待方案/审批",
    "failed": "失败",
    "blocked": "阻塞",
    "stale": "已失效",
    "not_started": "未开始",
}

FINAL_PASS_STATUSES = {
    "TASK_CERTIFIED",
    "WHOLE_MACHINE_TASK_PASSED",
}

LEGACY_GENERIC_PASS_STATUSES = {
    "PASS",
    "PASSED",
    "SUCCEEDED",
    "COMPLETED",
    "RESOLVED",
    "ACCEPTED",
}

PREFINAL_STATUSES = {
    "KINEMATIC_CERTIFIED",
    "PHYSICAL_PREFLIGHT_PASSED",
    "PHYSICAL_DISPATCH_PREFLIGHT_PASSED",
    "UBUNTU_PORTABILITY_READY",
}

EVIDENCE_HEALTH_LABELS = {
    "complete": "证据完整",
    "incomplete": "证据未完成",
    "stale": "证据已失效",
    "conflict": "证据有冲突",
    "legacy_unverified": "旧运行未完整复核",
}

REACT_WORKFLOW_PROFILE = "single_conversation_react_v2"
REACT_PROJECT_SCHEMA = "simulation_ready_cad_react_project_v2"
TASK_PLANNING_RELATIVE = Path("planning") / "task_planning.json"
OperatorAuthorityError = DashboardReadError


def _react_manifest_path(project_dir: Path) -> Path:
    sidecar = project_dir / "react_project.json"
    return sidecar if sidecar.is_file() and not sidecar.is_symlink() else (
        project_dir / "project.json"
    )


def _react_loop_state_path(
    workspace: Path, project_dir: Path
) -> Path | None:
    try:
        project, _record = read_stable_json_object(
            _react_manifest_path(project_dir),
            label="dashboard ReAct project manifest",
            allowed_root=workspace,
        )
        if (
            project.get("schema_version") != REACT_PROJECT_SCHEMA
            or project.get("workflow_profile") != REACT_WORKFLOW_PROFILE
        ):
            return None
        relative = project.get("loop_state")
        if not isinstance(relative, str) or not relative:
            relative = project.get("controller_state")
        candidate = (
            workspace / relative
            if isinstance(relative, str) and relative
            else project_dir / "react_loop" / "state.json"
        ).resolve()
        candidate.relative_to(project_dir.resolve())
        return candidate
    except (OSError, ValueError, DashboardReadError):
        return None


def _react_project_state(
    workspace: Path,
    project_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    """Read ReAct display state without consulting the old governance chain."""

    try:
        project, _project_record = read_stable_json_object(
            _react_manifest_path(project_dir),
            label="dashboard ReAct project manifest",
            allowed_root=workspace,
        )
        if (
            project.get("schema_version") != REACT_PROJECT_SCHEMA
            or project.get("workflow_profile") != REACT_WORKFLOW_PROFILE
            or project.get("project_id") != project_dir.name
        ):
            return {}, None
        state_path = _react_loop_state_path(workspace, project_dir)
        if state_path is None:
            return {}, None
        state, _state_record = read_stable_json_object(
            state_path,
            label="dashboard ReAct loop state",
            allowed_root=project_dir,
        )
        state_core = {
            key: value for key, value in state.items() if key != "state_sha256"
        }
        conversation = state.get("conversation")
        if (
            state.get("schema_version") != LOOP_STATE_SCHEMA
            or state.get("state_sha256") != canonical_sha256(state_core)
            or state.get("project_id") != project_dir.name
            or not isinstance(conversation, dict)
            or conversation.get("memory_enabled") is not False
            or not str(conversation.get("conversation_id") or "").strip()
        ):
            return {}, None
        current_run_text = str(state.get("current_run") or "")
        current_revision = str(state.get("current_revision") or "")
        if not current_run_text or not current_revision:
            return {}, None
        current_run_path = workspace / current_run_text
        current_run = current_run_path.resolve()
        if (
            current_run.parent != (project_dir / "runs").resolve()
            or current_run.name != current_revision
            or not current_run.is_dir()
            or current_run_path.is_symlink()
        ):
            return {}, None
        manifest, _manifest_record = read_stable_json_object(
            current_run / "run_manifest.json",
            label="dashboard ReAct run manifest",
            allowed_root=current_run,
        )
        if (
            manifest.get("workflow_profile") != REACT_WORKFLOW_PROFILE
            or manifest.get("project_id") != project_dir.name
            or manifest.get("revision") != current_revision
        ):
            return {}, None
        return state, current_run
    except (OSError, ValueError, OperatorAuthorityError):
        return {}, None


def is_react_loop_run(
    workspace_root: str | Path,
    run_dir: str | Path,
) -> bool:
    """Return whether an exact run opts into the new ReAct dashboard path."""

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    try:
        parts = run.relative_to(workspace).parts
    except ValueError:
        return False
    if (
        len(parts) != 5
        or parts[:2] != (".runtime_runs", "projects")
        or parts[3] != "runs"
    ):
        return False
    try:
        manifest, _record = read_stable_json_object(
            run / "run_manifest.json",
            label="dashboard ReAct run manifest",
            allowed_root=run,
        )
    except (OSError, ValueError, OperatorAuthorityError):
        return False
    return bool(
        manifest.get("workflow_profile") == REACT_WORKFLOW_PROFILE
        and manifest.get("project_id") == parts[2]
        and manifest.get("revision") == parts[4]
    )


def _react_project_revision_states(
    workspace: Path,
    project_dir: Path,
) -> dict[Path, dict[str, Any]]:
    """Project every canonical ReAct revision into the dashboard catalog.

    The loop state intentionally points only at the active revision.
    Historical revisions are instead bound to their latest immutable Harness
    outcome.  A revision must not disappear merely because intake failed before
    the production pipeline wrote ``run_status.json``.
    """

    state, current_run = _react_project_state(workspace, project_dir)
    if not state or current_run is None:
        return {}

    outcomes: dict[str, tuple[dict[str, Any], Path]] = {}
    outcome_root = project_dir / "react_loop" / "harness_outcomes"
    for outcome_path in sorted(outcome_root.glob("h*.json")):
        try:
            outcome, _record = read_stable_json_object(
                outcome_path,
                label="dashboard ReAct Harness outcome",
                allowed_root=project_dir,
            )
        except (OSError, ValueError, OperatorAuthorityError):
            continue
        revision = str(outcome.get("revision") or "").strip()
        if (
            outcome.get("project_id") != project_dir.name
            or re.fullmatch(r"r[0-9]{2,}", revision) is None
        ):
            continue
        # Outcome filenames begin with the zero-padded loop event
        # sequence, so sorted replacement selects the latest attempt for a
        # revision without consulting mutable directory timestamps.
        outcomes[revision] = (outcome, outcome_path)

    static_reviews: dict[str, dict[str, Any]] = {}
    for request_path in sorted(
        (project_dir / "runs").glob(
            "r*/static_visual_review/request-*.json"
        )
    ):
        try:
            request, _record = read_stable_json_object(
                request_path,
                label="dashboard historical static visual request",
                allowed_root=project_dir,
            )
        except (OSError, ValueError, OperatorAuthorityError):
            continue
        revision = str(request.get("revision") or "")
        request_id = str(request.get("request_id") or "")
        request_unsigned = {
            key: value for key, value in request.items() if key != "request_id"
        }
        if (
            request.get("project_id") != project_dir.name
            or request_path.parents[1].name != revision
        ):
            continue
        if request_id != canonical_sha256(request_unsigned):
            static_reviews[revision] = {
                "schema_version": (
                    "simulation_ready_cad_static_visual_review_state_v1"
                ),
                "status": "INVALID",
                "calls_used": 0,
                "max_calls": 3,
                "cycle_started_revision": revision,
                "integrity_error": "static visual request identity mismatch",
            }
            continue
        decision_path = request_path.with_name(f"decision-{request_id}.json")
        decision: dict[str, Any] | None = None
        decision_invalid = False
        if decision_path.is_file() and not decision_path.is_symlink():
            try:
                decision, _record = read_stable_json_object(
                    decision_path,
                    label="dashboard historical static visual decision",
                    allowed_root=project_dir,
                )
            except (OSError, ValueError, OperatorAuthorityError):
                decision = None
                decision_invalid = True
            if decision is not None:
                decision_unsigned = {
                    key: value
                    for key, value in decision.items()
                    if key != "record_sha256"
                }
                if (
                    decision.get("record_sha256")
                    != canonical_sha256(decision_unsigned)
                    or decision.get("project_id") != project_dir.name
                    or decision.get("revision") != revision
                    or decision.get("run") != request.get("run")
                    or decision.get("request_id") != request_id
                    or decision.get("request_path")
                    != request_path.relative_to(workspace).as_posix()
                    or decision.get("request_sha256")
                    != _cached_sha256_file(request_path)
                ):
                    decision = None
                    decision_invalid = True
        effective = str((decision or {}).get("decision") or "")
        review = {
            "schema_version": "simulation_ready_cad_static_visual_review_state_v1",
            "status": (
                "APPROVED"
                if effective == "APPROVE"
                else "CHANGES_REQUESTED"
                if effective == "REQUEST_CHANGES"
                else "BUDGET_EXHAUSTED_CONTINUE"
                if effective == "BUDGET_EXHAUSTED_CONTINUE"
                else "REJECTED"
                if effective == "REJECT"
                else "INVALID"
                if decision_invalid
                else "PENDING"
            ),
            "calls_used": int(
                (decision or {}).get("call_index")
                or max(0, int(request.get("call_index") or 1) - 1)
            ),
            "max_calls": int(request.get("max_calls") or 3),
            "cycle_started_revision": revision,
            "assembly_step_sha256": str(
                request.get("assembly_step_sha256") or ""
            ),
            "appearance_qualification": str(
                (decision or {}).get("appearance_qualification") or ""
            ),
            "benchmark_appearance_eligible": (
                effective == "APPROVE"
            ),
            "request": {
                "request_id": request_id,
                "request_path": request_path.relative_to(workspace).as_posix(),
                "request_sha256": _cached_sha256_file(request_path),
            },
        }
        if decision_invalid:
            review["integrity_error"] = (
                "static visual decision identity mismatch"
            )
        if decision is not None:
            review["decision"] = {
                "path": decision_path.relative_to(workspace).as_posix(),
                "sha256": _cached_sha256_file(decision_path),
                "record_sha256": str(decision.get("record_sha256") or ""),
            }
        if effective == "APPROVE":
            review["approved_assembly_step_sha256"] = review[
                "assembly_step_sha256"
            ]
            review["approved_request_id"] = request_id
        static_reviews[revision] = review
    for reuse_path in sorted(
        (project_dir / "runs").glob("r*/static_visual_review/reuse.json")
    ):
        try:
            reuse, _record = read_stable_json_object(
                reuse_path,
                label="dashboard historical static visual reuse",
                allowed_root=project_dir,
            )
        except (OSError, ValueError, OperatorAuthorityError):
            continue
        revision = str(reuse.get("revision") or "")
        unsigned = {
            key: value for key, value in reuse.items() if key != "reuse_sha256"
        }
        source_request = reuse.get("source_request")
        source_decision = reuse.get("source_decision")
        if (
            reuse.get("project_id") != project_dir.name
            or reuse_path.parents[1].name != revision
            or reuse.get("reuse_sha256") != canonical_sha256(unsigned)
            or not isinstance(source_request, dict)
            or not isinstance(source_decision, dict)
        ):
            continue
        reused_status = str(
            reuse.get("source_review_status") or "APPROVED"
        )
        reused_review = {
            "schema_version": "simulation_ready_cad_static_visual_review_state_v1",
            "status": reused_status,
            "calls_used": int(reuse.get("calls_used") or 0),
            "max_calls": int(reuse.get("max_calls") or 3),
            "cycle_started_revision": str(
                reuse.get("cycle_started_revision") or revision
            ),
            "assembly_step_sha256": str(
                reuse.get("assembly_step_sha256") or ""
            ),
            "appearance_qualification": str(
                reuse.get("appearance_qualification") or ""
            ),
            "benchmark_appearance_eligible": bool(
                reuse.get("benchmark_appearance_eligible")
            ),
            "request": dict(source_request),
            "decision": dict(source_decision),
            "reused_for_revision": revision,
            "reuse": {
                "path": reuse_path.relative_to(workspace).as_posix(),
                "sha256": _cached_sha256_file(reuse_path),
                "reuse_sha256": reuse["reuse_sha256"],
            },
        }
        if reused_status == "APPROVED":
            reused_review["approved_assembly_step_sha256"] = str(
                reuse.get("assembly_step_sha256") or ""
            )
            reused_review["approved_request_id"] = str(
                reuse.get("approved_request_id") or ""
            )
        else:
            reused_review["continued_assembly_step_sha256"] = str(
                reuse.get("assembly_step_sha256") or ""
            )
            reused_review["continued_request_id"] = str(
                reuse.get("approved_request_id") or ""
            )
        static_reviews[revision] = reused_review
    for outcome_path in sorted(
        (
            project_dir
            / "react_loop"
            / "static_visual_reviews"
            / "outcomes"
        ).glob("v*.json")
    ):
        try:
            outcome, _record = read_stable_json_object(
                outcome_path,
                label="dashboard historical static visual outcome",
                allowed_root=project_dir,
            )
        except (OSError, ValueError, OperatorAuthorityError):
            continue
        revision = str(outcome.get("revision") or "")
        recorded_status = str(outcome.get("recorded_status") or "")
        if (
            outcome.get("project_id") == project_dir.name
            and revision not in static_reviews
            and recorded_status == "STATIC_VISUAL_EVIDENCE_REQUIRED"
        ):
            static_reviews[revision] = {
                "schema_version": (
                    "simulation_ready_cad_static_visual_review_state_v1"
                ),
                "status": "EVIDENCE_REQUIRED",
                "calls_used": 0,
                "max_calls": 3,
                "cycle_started_revision": revision,
            }

    projected: dict[Path, dict[str, Any]] = {}
    runs_root = (project_dir / "runs").resolve()
    for manifest_path in sorted((project_dir / "runs").glob("*/run_manifest.json")):
        run_dir = manifest_path.parent.resolve()
        try:
            manifest, _record = read_stable_json_object(
                manifest_path,
                label="dashboard ReAct revision manifest",
                allowed_root=project_dir,
            )
        except (OSError, ValueError, OperatorAuthorityError):
            continue
        revision = str(manifest.get("revision") or "").strip()
        if (
            run_dir.parent != runs_root
            or run_dir.name != revision
            or manifest.get("workflow_profile") != REACT_WORKFLOW_PROFILE
            or manifest.get("project_id") != project_dir.name
        ):
            continue
        if run_dir == current_run:
            current = dict(state)
            state_path = _react_loop_state_path(workspace, project_dir)
            if state_path is not None:
                current["_dashboard_evidence_path"] = state_path.relative_to(
                    workspace
                ).as_posix()
            projected[run_dir] = current
            continue

        outcome_entry = outcomes.get(revision)
        if outcome_entry is None:
            historical = {
                "status": "REVISION_SUPERSEDED",
                "project_id": project_dir.name,
                "current_revision": revision,
                "_dashboard_historical": True,
                "_dashboard_evidence_path": manifest_path.relative_to(
                    workspace
                ).as_posix(),
            }
            if revision in static_reviews:
                historical["static_visual_review"] = static_reviews[revision]
            projected[run_dir] = historical
            continue
        outcome, outcome_path = outcome_entry
        kind = str(outcome.get("kind") or "").strip().upper()
        projected_status = {
            "PASS": "COMPLETE",
            "FAIL_CONTINUE_LOOP": "REVISION_FAILED",
            "RESOURCE_BLOCKED": "RESOURCE_RETRY_REQUIRED",
            "TERMINATE": "TERMINATED",
        }.get(kind, "REVISION_SUPERSEDED")
        historical = {
            "status": projected_status,
            "project_id": project_dir.name,
            "current_revision": revision,
            "harness_outcome_kind": kind,
            "harness_recorded_status": str(
                outcome.get("recorded_status") or ""
            ),
            "_dashboard_historical": True,
            "_dashboard_evidence_path": outcome_path.relative_to(
                workspace
            ).as_posix(),
        }
        if revision in static_reviews:
            historical["static_visual_review"] = static_reviews[revision]
        projected[run_dir] = historical
    return projected


def _react_verdict(
    state: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    loop_status = str(state.get("status") or "")
    if not loop_status:
        return fallback
    if loop_status == "COMPLETE":
        status, final = "passed", True
    elif loop_status in {"TERMINATED", "REVISION_FAILED"}:
        status, final = "failed", True
    elif loop_status == "REVISION_SUPERSEDED":
        status, final = "stale", True
    elif loop_status in {
        "PLAN_REVIEW_REQUIRED",
        "RESOURCE_APPROVAL_REQUIRED",
        "PLUGIN_IMPLEMENTATION_REQUIRED",
        "STATIC_VISUAL_REVIEW_REQUIRED",
        "EXTERNAL_VALIDATION_REVIEW_REQUIRED",
        "VALIDATION_FAILURE_REVIEW_REQUIRED",
        "WAITING_FOR_EXTERNAL_APPROVAL",
    }:
        status, final = "awaiting_review", False
    elif loop_status in {
        "RESOURCE_RETRY_REQUIRED",
        "REPEATED_ERROR_PAUSED",
    }:
        status, final = "blocked", False
    elif loop_status == "PREFLIGHT_REQUIRED":
        status, final = "preparing", False
    else:
        status, final = "running", False
    return {
        "status": status,
        "final": final,
        "authority": (
            "react_harness_outcome"
            if state.get("_dashboard_historical")
            else "react_loop_observation"
        ),
        "reason_code": loop_status,
    }


def _harness_defect_banner(
    state: dict[str, Any],
    run_status: dict[str, Any],
) -> dict[str, Any] | None:
    failure = state.get("previous_failure")
    routing = (
        failure.get("failure_routing") if isinstance(failure, dict) else None
    )
    routing = routing if isinstance(routing, dict) else {}
    status = str(state.get("status") or "")
    if (
        routing.get("category") != "HARNESS_DEFECT"
        and status != "EXTERNAL_VALIDATION_REVIEW_REQUIRED"
    ):
        return None
    fact = routing.get("failure_fact")
    envelope = fact.get("envelope") if isinstance(fact, dict) else None
    return {
        "category": "HARNESS_DEFECT",
        "code": (
            routing.get("reason_code")
            or run_status.get("failure_reason")
            or run_status.get("error_code")
            or status
            or None
        ),
        "owner": routing.get("responsible_authority") or "harness_code",
        "message": routing.get("guidance") or "",
        "envelope": envelope,
        "recovery_action": routing.get("recovery_action"),
        "route_target": routing.get("route_target") or "human",
    }


def _react_effective_state(verdict: dict[str, Any]) -> dict[str, Any]:
    status = str(verdict["status"])
    final = bool(verdict["final"])
    reason = str(verdict["reason_code"])
    return {
        "recorded_status": status,
        "recorded_status_raw": reason,
        "effective_status": status,
        "effective_status_raw": reason,
        "effective_final": final,
        "reason_code": reason,
        "terminal_ready": final,
        "terminal_reason": "react-loop-state" if final else "react-loop-live",
        "eligible_pass": status == "passed",
        "healthy_running": status == "running",
        "lease": None,
        "iteration": None,
        "governance": {
            "action": "NONE",
            "ledger_valid": True,
            "ledger_reason_code": "NOT_APPLICABLE_REACT_LOOP",
            "event": None,
        },
    }


_LEGACY_GOVERNANCE_LEDGER = {
    "valid": True,
    "reason_code": "LEGACY_READ_ONLY_NO_GOVERNANCE",
    "event_count": 0,
    "head_sha256": "",
    "events": [],
}


def _legacy_terminal_ready(status_value: Any) -> bool:
    return (
        isinstance(status_value, str)
        and status_value.strip().upper() in CANONICAL_TERMINAL_STATUSES
    )


def _legacy_effective_state(
    verdict: dict[str, Any],
    *,
    recorded_status_raw: str,
    terminal_ready: bool,
) -> dict[str, Any]:
    """Read-only view of a pre-ReAct run.

    The retired governance/authority stack is never consulted; recorded
    evidence is displayed as-is and no run state is ever rewritten.
    """

    status = str(verdict["status"])
    return {
        "recorded_status": status,
        "recorded_status_raw": recorded_status_raw,
        "effective_status": status,
        "effective_status_raw": recorded_status_raw,
        "effective_final": terminal_ready,
        "reason_code": "LEGACY_READ_ONLY",
        "terminal_ready": terminal_ready,
        "terminal_reason": (
            "legacy-read-only" if terminal_ready else "legacy-run-not-terminal"
        ),
        "eligible_pass": status == "passed",
        "healthy_running": status == "running",
        "lease": None,
        "iteration": None,
        "governance": {
            "action": "NONE",
            "ledger_valid": True,
            "ledger_reason_code": "LEGACY_READ_ONLY_NO_GOVERNANCE",
            "event": None,
        },
    }


def revision_geometry_inventory(run_dir: Path) -> dict[str, Any]:
    """Return the run-scoped collision components derived from authoritative STEP."""

    components: list[dict[str, Any]] = []
    assembly_hashes: set[str] = set()
    comparison_profiles: list[dict[str, Any]] = []
    source_paths: list[str] = []
    for provenance_path in sorted(
        run_dir.glob("derived_adapters/*/family_collision/provenance.json")
    ):
        try:
            value = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        adapter_id = provenance_path.parent.parent.name
        comparison_profiles.append(
            {
                "adapter_id": adapter_id,
                "schema": str(value.get("schema") or ""),
                "mesh_linear_deflection_m": value.get("mesh_linear_deflection_m"),
                "mesh_angular_deflection_rad": value.get(
                    "mesh_angular_deflection_rad"
                ),
            }
        )
        assembly_sha256 = str(value.get("assembly_sha256") or "")
        if assembly_sha256:
            assembly_hashes.add(assembly_sha256)
        source_paths.append(provenance_path.relative_to(run_dir).as_posix())
        bodies = value.get("bodies")
        if not isinstance(bodies, list):
            continue
        for body in bodies:
            if not isinstance(body, dict):
                continue
            body_id = str(body.get("body_id") or "unnamed_body")
            fixed = bool(body.get("fixed"))
            initial_center = body.get("initial_center_abs_m")
            initial_rotation = body.get("initial_rotation_abs_quat_wxyz")
            body_components = body.get("components")
            if not isinstance(body_components, list):
                continue
            for component in body_components:
                if not isinstance(component, dict):
                    continue
                solid_id = str(component.get("solid_id") or "")
                component_id = str(
                    component.get("component_id") or solid_id
                )
                sha256 = str(component.get("sha256") or "")
                if not component_id or not solid_id or not sha256:
                    continue
                comparison_record = {
                    "mesh_sha256": sha256,
                    "source_solid_id": solid_id,
                    "initial_center_abs_m": initial_center,
                    "initial_rotation_abs_quat_wxyz": initial_rotation,
                    "fixed": fixed,
                    "representation": str(
                        component.get("representation") or "unknown"
                    ),
                    "body_local": component.get("body_local"),
                }
                comparison_sha256 = hashlib.sha256(
                    json.dumps(
                        comparison_record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                components.append(
                    {
                        "key": f"{adapter_id}/{body_id}/{component_id}",
                        "adapter_id": adapter_id,
                        "body_id": body_id,
                        "component_id": component_id,
                        "solid_id": solid_id,
                        "sha256": sha256,
                        "comparison_sha256": comparison_sha256,
                        "initial_center_abs_m": initial_center,
                        "initial_rotation_abs_quat_wxyz": initial_rotation,
                        "fixed": fixed,
                        "representation": str(
                            component.get("representation") or "unknown"
                        ),
                    }
                )
    components.sort(key=lambda item: item["key"])
    component_keys = [item["key"] for item in components]
    duplicate_keys = sorted(
        {
            key
            for key in component_keys
            if component_keys.count(key) > 1
        }
    )
    return {
        "available": bool(components) and not duplicate_keys,
        "source": "harness_collision_provenance",
        "source_paths": source_paths,
        "assembly_sha256s": sorted(assembly_hashes),
        "comparison_profiles": sorted(
            comparison_profiles,
            key=lambda item: item["adapter_id"],
        ),
        "component_count": len(components),
        "duplicate_keys": duplicate_keys,
        "components": components,
    }


def compare_revision_geometry(
    base_run_dir: Path,
    target_run_dir: Path,
) -> dict[str, Any]:
    """Compare stable collision-component references and their recorded hashes."""

    base = revision_geometry_inventory(base_run_dir)
    target = revision_geometry_inventory(target_run_dir)
    base_by_key = {item["key"]: item for item in base["components"]}
    target_by_key = {item["key"]: item for item in target["components"]}
    comparable = (
        bool(base["available"] and target["available"])
        and base["comparison_profiles"] == target["comparison_profiles"]
    )
    shared_keys = sorted(base_by_key.keys() & target_by_key.keys())
    added = [target_by_key[key] for key in sorted(target_by_key.keys() - base_by_key.keys())]
    removed = [base_by_key[key] for key in sorted(base_by_key.keys() - target_by_key.keys())]
    modified = [
        {
            "key": key,
            "adapter_id": target_by_key[key]["adapter_id"],
            "body_id": target_by_key[key]["body_id"],
            "solid_id": target_by_key[key]["solid_id"],
            "before_sha256": base_by_key[key]["sha256"],
            "after_sha256": target_by_key[key]["sha256"],
            "before_comparison_sha256": base_by_key[key]["comparison_sha256"],
            "after_comparison_sha256": target_by_key[key]["comparison_sha256"],
        }
        for key in shared_keys
        if base_by_key[key]["comparison_sha256"]
        != target_by_key[key]["comparison_sha256"]
    ]
    unchanged_count = sum(
        base_by_key[key]["comparison_sha256"]
        == target_by_key[key]["comparison_sha256"]
        for key in shared_keys
    )
    return {
        "schema_version": "revision_geometry_comparison_v1",
        "available": comparable,
        "source": "harness_collision_provenance",
        "identity_scope": "harness_import_reference",
        "hash_scope": "derived_collision_mesh_bytes_plus_body_placement",
        "basis": "component reference, derived collision-mesh SHA-256, body placement, and collision representation",
        "base": {
            "component_count": base["component_count"],
            "assembly_sha256s": base["assembly_sha256s"],
            "source_paths": base["source_paths"],
            "comparison_profiles": base["comparison_profiles"],
        },
        "target": {
            "component_count": target["component_count"],
            "assembly_sha256s": target["assembly_sha256s"],
            "source_paths": target["source_paths"],
            "comparison_profiles": target["comparison_profiles"],
        },
        "summary": {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
            "unchanged": unchanged_count,
        },
        "added": added,
        "removed": removed,
        "modified": modified,
        "limitations": [
            "This compares recorded harness collision meshes and placements, not a STEP Boolean diff.",
            "Component references follow STEP import order and are not persistent CAD GUIDs.",
            "A renamed or renumbered component is reported as one removal plus one addition.",
        ],
    }


LOAD_LABELS = {
    "nominal_light": "轻载",
    "nominal_loaded": "重载",
    "zero_input_gravity": "零输入重力",
    "transverse_probe": "横向扰动",
    "transverse_torque_probe": "侧向扭矩扰动",
    "held_out_mid": "保留测试",
    "task_envelope_roundtrip": "全行程往返",
    "task_angle_roundtrip": "全角度往返",
    "transverse_rack_probe": "侧向扰动",
    "task_rack_roundtrip": "完整升降",
}

LOAD_ORDER = {load_id: index for index, load_id in enumerate(LOAD_LABELS)}
LOAD_LABELS.update(
    {
        "task_clock_cycle": "三针走时任务",
        "task_manual_time_setting": "手动调时任务",
        "task_alarm_cycle": "机械打铃任务",
        "shaft_transverse_probe": "侧向扰动",
    }
)


@dataclass
class ReadContext:
    workspace_root: Path
    diagnostics: list[dict[str, str]]

    def read_json(self, path: Path | None, *, role: str) -> dict[str, Any]:
        if path is None:
            return {}
        if not path.is_file():
            self.diagnostics.append(
                {"level": "missing", "role": role, "path": self.relative(path)}
            )
            return {}
        try:
            value, _record = read_stable_json_object(
                path,
                label=role,
                allowed_root=self.workspace_root,
            )
        except (OSError, OperatorAuthorityError) as exc:
            self.diagnostics.append(
                {
                    "level": "malformed",
                    "role": role,
                    "path": self.relative(path),
                    "detail": str(exc),
                }
            )
            return (
                {"status": "RUN_STATUS_DOCUMENT_FAILED_VALIDATION"}
                if role == "run status"
                else {}
            )
        if not isinstance(value, dict):
            self.diagnostics.append(
                {
                    "level": "malformed",
                    "role": role,
                    "path": self.relative(path),
                    "detail": "expected a JSON object",
                }
            )
            return {}
        return value

    def resolve(self, value: str | Path | None) -> Path | None:
        if value is None or str(value).strip() == "":
            return None
        return resolve_workspace_reference(self.workspace_root, value)

    def relative(self, path: Path | None) -> str:
        if path is None:
            return ""
        try:
            return path.resolve(strict=False).relative_to(self.workspace_root).as_posix()
        except ValueError:
            return str(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_display_name(run_dir: Path, base: str) -> str:
    """Append a dashboard-only annotation without mutating run evidence."""

    parts = run_dir.resolve().parts
    if (
        len(parts) >= 5
        and parts[-5] == ".runtime_runs"
        and parts[-4] == "projects"
        and parts[-2] == "runs"
    ):
        # Canonical display identity is derived from signed run/project data.
        # An ad-hoc file inside an immutable rNN cannot rename it.
        return base
    annotation_path = run_dir / "dashboard_annotation.json"
    if not annotation_path.is_file():
        return base
    try:
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return base
    if not isinstance(annotation, dict) or annotation.get("schema_version") != "0.1":
        return base
    suffix = str(annotation.get("display_suffix") or "").strip()
    return f"{base} · {suffix}" if suffix else base


def _human_size(size: int | None) -> str:
    if size is None:
        return ""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _sha256_file(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=4096)
def _cached_sha256_file_value(path_value: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    return _sha256_file(Path(path_value))


def _cached_sha256_file(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        stat = path.stat()
    except OSError:
        return ""
    if not path.is_file():
        return ""
    return _cached_sha256_file_value(
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


def _data_uri(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    mime = {".glb": "model/gltf-binary"}.get(path.suffix.lower()) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


STATIC_VISUAL_DASHBOARD_ROLES = (
    "front",
    "rear",
    "left",
    "right",
    "top",
    "bottom",
)
STATIC_VISUAL_HARNESS_RENDER_AUTHORITY = "harness"
STATIC_VISUAL_LEGACY_RENDER_AUTHORITY = "legacy_unknown"
STATIC_VISUAL_REVIEW_REQUEST_SCHEMA_V2 = (
    "simulation_ready_cad_static_visual_review_request_v2"
)
STATIC_VISUAL_RENDER_BUNDLE_SCHEMA = (
    "simulation_ready_cad_static_render_bundle_v1"
)
WORKER_SUPPLEMENTAL_RENDER_AUTHORITY = "worker_supplemental"


def _resolve_bundle_relative(
    workspace_root: Path,
    bundle_dir: Path,
    relative: str,
) -> Path | None:
    text = str(relative or "").strip()
    posix = PurePosixPath(text)
    if (
        not text
        or posix.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        return None
    path = bundle_dir.joinpath(*posix.parts).resolve()
    try:
        path.relative_to(bundle_dir)
        path.relative_to(workspace_root)
    except ValueError:
        return None
    return path


def _harness_render_bundle_verified(
    ctx: ReadContext,
    request: dict[str, Any],
    *,
    renderer_version: str,
) -> tuple[dict[str, Any], Path] | None:
    binding = request.get("render_bundle")
    if not isinstance(binding, dict):
        return None
    relative = str(binding.get("path") or "").strip()
    file_digest = str(binding.get("sha256") or "").strip().lower()
    content_digest = str(binding.get("bundle_sha256") or "").strip().lower()
    posix = PurePosixPath(relative)
    if (
        not relative
        or posix.is_absolute()
        or any(part in {"", ".", ".."} for part in posix.parts)
        or len(file_digest) != 64
        or len(content_digest) != 64
    ):
        return None
    workspace_root = ctx.workspace_root.resolve()
    bundle_path = workspace_root.joinpath(*posix.parts).resolve()
    try:
        bundle_path.relative_to(workspace_root)
    except ValueError:
        return None
    if bundle_path.is_symlink() or not bundle_path.is_file():
        return None
    if _cached_sha256_file(bundle_path) != file_digest:
        return None
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    if not isinstance(bundle, dict):
        return None
    unsigned = {
        key: value for key, value in bundle.items() if key != "bundle_sha256"
    }
    assembly_step = str(request.get("assembly_step_sha256") or "").strip()
    if not (
        assembly_step
        and str(bundle.get("schema_version") or "")
        == STATIC_VISUAL_RENDER_BUNDLE_SCHEMA
        and str(bundle.get("render_authority") or "").strip()
        == STATIC_VISUAL_HARNESS_RENDER_AUTHORITY
        and str(bundle.get("renderer_version") or "").strip() == renderer_version
        and str(bundle.get("source_step_sha256") or "").strip() == assembly_step
        and str(bundle.get("bundle_sha256") or "").strip().lower() == content_digest
        and canonical_sha256(unsigned) == content_digest
    ):
        return None
    return bundle, bundle_path


def _harness_request_views_verified(
    ctx: ReadContext,
    request: dict[str, Any],
    bundle: dict[str, Any],
    bundle_path: Path,
) -> bool:
    request_views = request.get("views")
    bundle_views = bundle.get("views")
    expected_roles = list(STATIC_VISUAL_DASHBOARD_ROLES)
    if (
        not isinstance(request_views, list)
        or not isinstance(bundle_views, list)
        or len(request_views) != len(expected_roles)
        or len(bundle_views) != len(expected_roles)
        or any(not isinstance(item, dict) for item in request_views)
        or any(not isinstance(item, dict) for item in bundle_views)
        or [item.get("role") for item in request_views] != expected_roles
        or [item.get("role") for item in bundle_views] != expected_roles
    ):
        return False
    workspace_root = ctx.workspace_root.resolve()
    bundle_dir = bundle_path.parent.resolve()
    for request_view, bundle_view in zip(
        request_views, bundle_views, strict=True
    ):
        if (
            str(request_view.get("role") or "")
            != str(bundle_view.get("role") or "")
            or str(request_view.get("projection") or "") != "orthographic"
            or str(bundle_view.get("projection") or "") != "orthographic"
        ):
            return False
        try:
            request_png = ctx.resolve(request_view.get("path"))
            request_meta = ctx.resolve(request_view.get("metadata_path"))
        except ValueError:
            return False
        bundle_png = _resolve_bundle_relative(
            workspace_root,
            bundle_dir,
            str(bundle_view.get("path") or ""),
        )
        bundle_meta = _resolve_bundle_relative(
            workspace_root,
            bundle_dir,
            str(bundle_view.get("metadata_path") or ""),
        )
        if (
            request_png is None
            or request_meta is None
            or bundle_png is None
            or bundle_meta is None
            or request_png != bundle_png
            or request_meta != bundle_meta
        ):
            return False
        png_digest = str(bundle_view.get("sha256") or "").strip().lower()
        metadata_digest = str(
            bundle_view.get("metadata_sha256") or ""
        ).strip().lower()
        if (
            str(request_view.get("sha256") or "").strip().lower() != png_digest
            or str(request_view.get("metadata_sha256") or "").strip().lower()
            != metadata_digest
            or len(png_digest) != 64
            or len(metadata_digest) != 64
            or _cached_sha256_file(request_png) != png_digest
            or _cached_sha256_file(request_meta) != metadata_digest
        ):
            return False
    return True


def _projected_static_render_authority(
    ctx: ReadContext,
    request: dict[str, Any],
) -> tuple[str, str]:
    authority = str(request.get("render_authority") or "").strip()
    version = str(request.get("renderer_version") or "").strip()
    if (
        str(request.get("schema_version") or "")
        != STATIC_VISUAL_REVIEW_REQUEST_SCHEMA_V2
        or authority != STATIC_VISUAL_HARNESS_RENDER_AUTHORITY
        or not version
    ):
        return STATIC_VISUAL_LEGACY_RENDER_AUTHORITY, ""
    verified = _harness_render_bundle_verified(
        ctx, request, renderer_version=version
    )
    if verified is None:
        return STATIC_VISUAL_LEGACY_RENDER_AUTHORITY, ""
    bundle, bundle_path = verified
    if not _harness_request_views_verified(ctx, request, bundle, bundle_path):
        return STATIC_VISUAL_LEGACY_RENDER_AUTHORITY, ""
    return STATIC_VISUAL_HARNESS_RENDER_AUTHORITY, version


def _is_supplemental_design_issue(code: str) -> bool:
    return str(code).startswith("VISUAL_REVIEW_")


def _split_design_quality_issues(
    evaluation: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issues = [
        dict(item)
        for item in evaluation.get("issues") or []
        if isinstance(item, dict)
    ]
    supplemental = [
        dict(item)
        for item in evaluation.get("supplemental_issues") or []
        if isinstance(item, dict)
    ]
    blocking = [
        item for item in issues if not _is_supplemental_design_issue(str(item.get("code") or ""))
    ]
    migrated = [
        item for item in issues if _is_supplemental_design_issue(str(item.get("code") or ""))
    ]
    return blocking, supplemental + migrated


def _static_visual_review_projection(
    ctx: ReadContext,
    state: dict[str, Any],
    *,
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Project the Loop-owned pre-physics review without granting PASS."""

    empty = {
        "recorded": False,
        "status": "NOT_REQUESTED",
        "call_index": 0,
        "calls_used": 0,
        "max_calls": 0,
        "allowed_decisions": [],
        "assembly_step_sha256": "",
        "cycle_started_revision": "",
        "request": None,
        "views": [],
        "views_complete": False,
        "decision": None,
        "reused": False,
        "reused_for_revision": "",
        "authority": {
            "scope": "static_appearance_before_physics",
            "physical_pass": False,
        },
        "render_authority": "",
        "renderer_version": "",
    }
    current = state.get("static_visual_review")
    if not isinstance(current, dict):
        return empty, []

    request_binding = current.get("request")
    request_path: Path | None = None
    request: dict[str, Any] = {}
    if isinstance(request_binding, dict):
        try:
            request_path = ctx.resolve(request_binding.get("request_path"))
        except ValueError as exc:
            ctx.diagnostics.append(
                {
                    "level": "invalid",
                    "role": "static visual review request",
                    "path": str(request_binding.get("request_path") or ""),
                    "detail": str(exc),
                }
            )
        expected_request_sha256 = str(
            request_binding.get("request_sha256") or ""
        )
        if request_path is not None and (
            not expected_request_sha256
            or _cached_sha256_file(request_path) == expected_request_sha256
        ):
            request = ctx.read_json(
                request_path, role="static visual review request"
            )
        elif request_path is not None:
            ctx.diagnostics.append(
                {
                    "level": "conflict",
                    "role": "static visual review request",
                    "path": ctx.relative(request_path),
                    "detail": "request bytes do not match the Loop binding",
                }
            )
    if request:
        request_unsigned = {
            key: value for key, value in request.items() if key != "request_id"
        }
        if request.get("request_id") != canonical_sha256(request_unsigned):
            ctx.diagnostics.append(
                {
                    "level": "conflict",
                    "role": "static visual review request",
                    "path": ctx.relative(request_path),
                    "detail": "request identity does not match its contents",
                }
            )
            request = {}

    call_index = int(request.get("call_index") or current.get("calls_used") or 0)
    max_calls = int(request.get("max_calls") or current.get("max_calls") or 0)
    render_authority, renderer_version = _projected_static_render_authority(
        ctx, request
    )
    view_rows: list[dict[str, Any]] = []
    view_artifacts: list[dict[str, Any]] = []
    expected_view_roles = tuple(
        str(item).strip().lower()
        for item in request.get("required_view_roles")
        or STATIC_VISUAL_DASHBOARD_ROLES
        if str(item).strip()
    )
    for raw in request.get("views") or []:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role") or "").strip().lower()
        if role not in expected_view_roles:
            continue
        label = str(raw.get("label") or role)
        try:
            image_path = ctx.resolve(raw.get("path"))
        except ValueError:
            image_path = None
        expected_sha256 = str(raw.get("sha256") or "")
        actual_sha256 = _cached_sha256_file(image_path)
        bound = bool(expected_sha256 and actual_sha256 == expected_sha256)
        artifact = _artifact(
            ctx,
            image_path,
            label=f"静态外形审核：{label}",
            kind="static-visual-image",
            stage_id="design-package",
            role="evidence",
            summary=(
                f"Harness 从冻结 STEP 统一生成的权威审核视图（renderer {renderer_version}）"
                if render_authority == STATIC_VISUAL_HARNESS_RENDER_AUTHORITY
                else "历史/来源未验证"
            ),
            sha256=expected_sha256,
            status=(
                "evidence_complete"
                if bound
                else "stale"
                if image_path is not None and image_path.is_file()
                else "evidence_incomplete"
            ),
        )
        artifact["details"].update(
            {
                "visual_role": f"static-review-{role}",
                "static_visual_review": True,
                "call_index": call_index,
                "render_authority": render_authority,
                "renderer_version": renderer_version,
            }
        )
        artifact["preview_type"] = "image"
        if mode == "snapshot" and bound:
            artifact["preview_data_uri"] = _data_uri(image_path)
        view_artifacts.append(artifact)
        view_rows.append(
            {
                "role": role,
                "label": label,
                "path": artifact["path"],
                "sha256": expected_sha256,
                "exists": artifact["exists"],
                "bound": bound,
                "preview_type": artifact["preview_type"],
                "preview_data_uri": artifact["preview_data_uri"],
                "render_authority": render_authority,
                "renderer_version": renderer_version,
                "kind": str(raw.get("kind") or "view"),
                "source_step": str(raw.get("source_step") or "assembly.step"),
            }
        )

    decision_projection: dict[str, Any] | None = None
    decision_binding = current.get("decision")
    if isinstance(decision_binding, dict):
        try:
            decision_path = ctx.resolve(decision_binding.get("path"))
        except ValueError:
            decision_path = None
        expected_decision_sha256 = str(decision_binding.get("sha256") or "")
        decision: dict[str, Any] = {}
        if decision_path is not None and (
            not expected_decision_sha256
            or _cached_sha256_file(decision_path) == expected_decision_sha256
        ):
            decision = ctx.read_json(
                decision_path, role="static visual review decision"
            )
        if decision:
            decision_unsigned = {
                key: value
                for key, value in decision.items()
                if key != "record_sha256"
            }
            if (
                decision.get("record_sha256")
                != canonical_sha256(decision_unsigned)
                or decision.get("request_id") != request.get("request_id")
                or decision.get("request_path")
                != (ctx.relative(request_path) if request_path else "")
                or decision.get("request_sha256")
                != (
                    _cached_sha256_file(request_path)
                    if request_path is not None
                    else ""
                )
            ):
                ctx.diagnostics.append(
                    {
                        "level": "conflict",
                        "role": "static visual review decision",
                        "path": ctx.relative(decision_path),
                        "detail": "decision identity does not match its request",
                    }
                )
                decision = {}
        if decision:
            decision_projection = {
                "path": ctx.relative(decision_path),
                "sha256": expected_decision_sha256,
                "requested_decision": str(
                    decision.get("requested_decision") or ""
                ),
                "decision": str(decision.get("decision") or ""),
                "reviewer": str(decision.get("reviewer") or ""),
                "feedback": str(decision.get("feedback") or ""),
                "findings": [
                    dict(item)
                    for item in decision.get("findings") or []
                    if isinstance(item, dict)
                ],
                "call_index": int(decision.get("call_index") or call_index),
                "max_calls": int(decision.get("max_calls") or max_calls),
                "budget_exhausted": bool(decision.get("budget_exhausted")),
                "appearance_qualification": str(
                    decision.get("appearance_qualification") or ""
                ),
            }

    roles = [item["role"] for item in view_rows if item["bound"]]
    reused_for_revision = str(current.get("reused_for_revision") or "")
    return {
        "recorded": True,
        "status": str(current.get("status") or "UNKNOWN"),
        "call_index": call_index,
        "calls_used": int(current.get("calls_used") or 0),
        "max_calls": max_calls,
        "allowed_decisions": [
            str(item) for item in request.get("allowed_decisions") or []
        ],
        "assembly_step_sha256": str(
            current.get("assembly_step_sha256")
            or request.get("assembly_step_sha256")
            or ""
        ),
        "cycle_started_revision": str(
            current.get("cycle_started_revision") or ""
        ),
        "request": (
            {
                "request_id": str(request_binding.get("request_id") or ""),
                "path": ctx.relative(request_path),
                "sha256": str(request_binding.get("request_sha256") or ""),
            }
            if isinstance(request_binding, dict)
            else None
        ),
        "views": view_rows,
        "views_complete": (
            len(roles) == len(expected_view_roles)
            and set(roles) == set(expected_view_roles)
        ),
        "decision": decision_projection,
        "reused": bool(reused_for_revision),
        "reused_for_revision": reused_for_revision,
        "appearance_qualification": str(
            current.get("appearance_qualification") or ""
        ),
        "benchmark_appearance_eligible": bool(
            current.get("benchmark_appearance_eligible")
        ),
        "authority": {
            "scope": "static_appearance_before_physics",
            "physical_pass": False,
        },
        "render_authority": render_authority,
        "renderer_version": renderer_version,
    }, view_artifacts


def _issue_detail(item: dict[str, Any]) -> dict[str, str]:
    code = str(item.get("code") or "ISSUE")
    known = {
        "NATIVE_ASSEMBLY_MISSING": {
            "title": "只有通用 STEP，没有附原设计文件",
            "summary": "这次提供的是可交换的 STEP 整机，没有同时附上 SolidWorks、Fusion 360、Onshape 等软件中的可编辑装配源文件。",
            "impact": "不影响本次读取几何和继续仿真；只是无法查看原设计软件里的装配约束和建模历史。",
            "non_blocking_reason": "设计资料能够打开，且后续几何检查、机构验证和整机任务均已获准继续。",
        },
        "STEP_KERNEL_VALIDATION_DEFERRED": {
            "title": "OCCT 几何内核检查被延期",
            "summary": "当前环境没有完成几何内核验证。",
            "impact": "这是物理预检阻塞项，不能据此推断几何可用。",
            "non_blocking_reason": "不可接受；必须恢复 OCCT 检查后重跑。",
        },
        "FIELD_UNRECOGNIZED": {
            "title": "存在未识别字段",
            "summary": "该字段没有参与当前验证语义。",
            "impact": "已识别字段仍照常检查；未知字段不产生放行能力。",
            "non_blocking_reason": "仅当该字段不属于必需验证语义时才是非阻塞警告。",
        },
    }
    detail = dict(known.get(code) or {})
    detail.setdefault("title", code)
    detail.setdefault("summary", str(item.get("message") or item.get("hint") or "未提供说明。"))
    detail.setdefault("impact", "请根据 Intake 能力门判断是否阻塞后续阶段。")
    detail.setdefault("non_blocking_reason", "没有可证明的非阻塞理由。")
    detail["code"] = code
    detail["severity"] = str(item.get("severity") or "")
    detail["source"] = str(item.get("file") or "")
    return detail



WHOLE_TASK_LABELS = {
    "raise_gate": "升起闸门",
    "lower_gate": "降下闸门",
    "lower_gate_to_bottom": "降下闸门到底",
    "prevent_gate_drop_after_release": "松手防坠",
    "open": "打开",
    "close": "关闭",
}


WHOLE_TASK_LABELS.update(
    {
        "run_three_hand_clock": "三针正常走时",
        "verify_three_hand_ratio": "三针比例正确",
        "set_time_manually": "后冠手动调时",
        "ring_twin_bell_alarm": "双铃机械打铃",
        "zero_time_and_resume": "校零后松手续走",
        "alarm_after_one_minute": "一分钟后边走边响",
        "run_three_hand_clock_logic": "三针正常走时",
        "ring_alarm_independently": "独立机械打铃",
        "set_alarm_time_by_turning_hour_hand": "拨针设定闹铃时间",
        "trigger_alarm_at_selected_hand_position": "指针到位触发闹铃",
    }
)


def _task_id_for_stage(task_ids: list[str], stage_name: str, index: int) -> str:
    preferred = (
        ("open", "raise", "lift", "extend")
        if stage_name == "outbound"
        else ("close", "lower", "drop", "retract")
        if stage_name == "return"
        else ("drop", "hold", "prevent")
        if stage_name == "anti_drop"
        else ()
    )
    for task_id in task_ids:
        lowered = task_id.lower()
        if any(token in lowered for token in preferred):
            return task_id
    return task_ids[index] if index < len(task_ids) else stage_name

def _whole_task_rows(
    whole: dict[str, Any],
    plan: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    plan = plan or {}
    task_ids = list(whole.get("task_ids") or [])
    task_results = [
        item for item in (whole.get("task_results") or []) if isinstance(item, dict)
    ]
    if task_results:
        plan_coordinate_kind = str(plan.get("coordinate_kind") or "").lower()
        clock_task_ids = {
            "run_three_hand_clock",
            "verify_three_hand_ratio",
            "set_time_manually",
            "ring_twin_bell_alarm",
            "zero_time_and_resume",
            "alarm_after_one_minute",
        }
        clock_tasks = (
            plan_coordinate_kind == "clock_multifunction"
            or any(
                str(item.get("task_id") or "") in clock_task_ids
                for item in task_results
            )
        )
        expected_hour = _finite(plan.get("expected_hour_to_minute_ratio")) or (1.0 / 12.0)
        expected_second = _finite(plan.get("expected_second_to_minute_ratio")) or 60.0
        ratio_tolerance = _finite(plan.get("hand_ratio_relative_tolerance")) or 0.02
        minimum_setting = _finite(plan.get("minimum_manual_setting_travel_rad")) or 0.02
        minimum_alarm = _finite(plan.get("minimum_alarm_swing_rad")) or 0.08
        minimum_clock = max(
            0.0,
            (_finite(plan.get("outbound_angle_rad")) or 0.35)
            - (_finite(plan.get("angle_tolerance_rad")) or 0.08),
        )
        functional_rows: list[dict[str, Any]] = []
        for item in task_results:
            task_id = str(item.get("task_id") or f"task_{len(functional_rows) + 1}")
            evidence = item.get("evidence") or {}
            result_status = (
                "passed"
                if item.get("passed") is True
                else "failed"
                if item.get("passed") is False
                else _status_from_value(item.get("status"))
            )
            passed = result_status == "passed"
            if task_id == "run_three_hand_clock":
                observed = _finite(evidence.get("minute_span_rad")) or 0.0
                summary = f"分针实际走过 {math.degrees(observed):.2f}°，最低要求 {math.degrees(minimum_clock):.2f}°。"
            elif task_id == "verify_three_hand_ratio":
                hour_ratio = _finite(evidence.get("hour_to_minute_ratio")) or 0.0
                second_ratio = _finite(evidence.get("second_to_minute_ratio")) or 0.0
                same_direction = bool(evidence.get("same_display_direction"))
                summary = (
                    f"时针/分针 {hour_ratio:.6f}（目标 {expected_hour:.6f}）；"
                    f"秒针/分针 {second_ratio:.4f}（目标 {expected_second:.1f}）；"
                    f"三针方向{'一致' if same_direction else '不一致'}。"
                )
            elif task_id == "set_time_manually":
                minute_span = _finite(evidence.get("minute_span_rad")) or 0.0
                crown_span = _finite(evidence.get("setting_crown_span_rad")) or 0.0
                summary = (
                    f"只驱动后冠：后冠转动 {math.degrees(crown_span):.2f}°，"
                    f"分针被传动 {math.degrees(minute_span):.2f}°；最低要求 {math.degrees(minimum_setting):.2f}°。"
                )
            elif task_id == "ring_twin_bell_alarm":
                swing = _finite(evidence.get("alarm_hammer_swing_rad")) or 0.0
                summary = (
                    f"只驱动发条轴，铃锤摆幅 {math.degrees(swing):.2f}°；"
                    f"最低要求 {math.degrees(minimum_alarm):.2f}°。"
                )
            elif task_id == "zero_time_and_resume":
                residual = _finite(evidence.get("zero_residual_rad")) or 0.0
                released = _finite(evidence.get("release_drift_rad")) or 0.0
                resumed = _finite(evidence.get("post_release_time_travel_rad")) or 0.0
                summary = (
                    f"回零残差 {math.degrees(residual):.2f}°；松手漂移 "
                    f"{math.degrees(released):.2f}°；随后续走 {math.degrees(resumed):.2f}°。"
                )
            elif task_id == "alarm_after_one_minute":
                second = _finite(evidence.get("trigger_second_hand_travel_rad")) or 0.0
                pretrigger = _finite(evidence.get("pretrigger_alarm_hammer_span_rad")) or 0.0
                ring = _finite(evidence.get("ring_alarm_hammer_span_rad")) or 0.0
                running = _finite(evidence.get("ring_timekeeping_travel_rad")) or 0.0
                summary = (
                    f"秒针走 {math.degrees(second):.2f}° 后触发；此前铃锤仅漂移 "
                    f"{math.degrees(pretrigger):.4f}°；响铃摆幅 {math.degrees(ring):.2f}°，"
                    f"同时分针继续走 {math.degrees(running):.2f}°。"
                )
            else:
                summary = str(
                    item.get("summary")
                    or evidence.get("summary")
                    or "已保存该整机任务的独立结果。"
                )
            functional_rows.append(
                {
                    "id": task_id,
                    "label": str(
                        item.get("label")
                        or WHOLE_TASK_LABELS.get(task_id, task_id)
                    ),
                    "stage": task_id,
                    "status": result_status,
                    "reached": passed,
                    "coordinate_kind": (
                        "clock_multifunction"
                        if clock_tasks
                        else plan_coordinate_kind or "task"
                    ),
                    "summary": summary,
                    "evidence": evidence,
                    "input_policy": str(evidence.get("input_policy") or ""),
                    "expected_hour_to_minute_ratio": expected_hour,
                    "expected_second_to_minute_ratio": expected_second,
                    "ratio_relative_tolerance": ratio_tolerance,
                }
            )
        return functional_rows
    labels = {"outbound": "打开", "anti_drop": "松手防坠", "return": "关闭"}
    stages = [item for item in (whole.get("stages") or []) if isinstance(item, dict)]
    rack_pinion = str(plan.get("coordinate_kind") or "").lower() == "rack_pinion_progress"
    angular = str(plan.get("coordinate_kind") or "").lower() == "rotation" or any(
        "target_angle_rad" in item for item in stages
    )
    angular_force_control = (
        angular
        and _finite(plan.get("max_control_force_n")) is not None
        and _finite(plan.get("max_control_torque_nm")) is None
    )
    if angular:
        tolerance = math.degrees(_finite(plan.get("angle_tolerance_rad")) or 0.0)
        effort_field = (
            "max_control_force_n" if angular_force_control else "max_control_torque_nm"
        )
        effort_sample_field = (
            "control_force_n" if angular_force_control else "control_torque_nm"
        )
        effort_limit = _finite(plan.get(effort_field))
        coordinate_unit = "°"
        effort_unit = "N" if angular_force_control else "N·m"
    else:
        tolerance_key = "progress_tolerance_m" if rack_pinion else "position_tolerance_m"
        effort_key = "max_control_torque_nm" if rack_pinion else "max_control_force_n"
        effort_field = effort_key
        effort_sample_field = "control_torque_nm" if rack_pinion else "control_force_n"
        tolerance = (_finite(plan.get(tolerance_key)) or 0.0) * 1000
        effort_limit = _finite(plan.get(effort_key))
        coordinate_unit = "mm"
        effort_unit = "N·m" if rack_pinion else "N"
    duration_limit_s = _finite(plan.get("max_stage_duration_s"))

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(stages):
        stage_name = str(item.get("stage") or f"stage_{index + 1}")
        if angular:
            target = math.degrees(_finite(item.get("target_angle_rad")) or 0.0)
            observed_rad = (
                _finite(item.get("min_angle_rad"))
                if stage_name in {"return", "anti_drop"}
                else _finite(item.get("max_angle_rad"))
            )
            observed = math.degrees(observed_rad or 0.0)
            max_effort = _finite(item.get(effort_field))
        else:
            target = (_finite(item.get("target_progress_m")) or 0.0) * 1000
            observed_m = (
                _finite(item.get("min_progress_m"))
                if stage_name in {"return", "anti_drop"}
                else _finite(item.get("max_progress_m"))
            )
            observed = (observed_m or 0.0) * 1000
            max_effort = _finite(item.get(effort_field))
        if max_effort is None:
            max_effort = max(
                abs(_finite(sample.get(effort_sample_field)) or 0.0)
                for sample in (whole.get("samples") or [{}])
            )
        error = abs(observed - target)
        reached = bool(item.get("reached"))
        duration_s = _finite(item.get("duration_s"))
        task_id = _task_id_for_stage(task_ids, stage_name, index)
        row = {
            "id": task_id,
            "label": WHOLE_TASK_LABELS.get(task_id, labels.get(stage_name, stage_name)),
            "stage": stage_name,
            "status": "passed" if reached else "failed",
            "reached": reached,
            "coordinate_kind": "angular" if angular else "linear",
            "coordinate_unit": coordinate_unit,
            "target": _round(target, 4),
            "observed": _round(observed, 4),
            "error": _round(error, 4),
            "tolerance": _round(tolerance, 4),
            "error_pass": error <= tolerance,
            "duration_s": _round(duration_s, 4),
            "duration_limit_s": _round(duration_limit_s, 4),
            "duration_pass": (
                duration_s <= duration_limit_s
                if duration_s is not None and duration_limit_s is not None
                else None
            ),
            "effort_unit": effort_unit,
            "max_effort": _round(max_effort, 4),
            "effort_limit": _round(effort_limit, 4),
            "effort_pass": (
                max_effort <= effort_limit
                if max_effort is not None and effort_limit is not None
                else None
            ),
        }
        # Retain the original linear keys for existing snapshots and consumers.
        if angular:
            row.update({
                "target_deg": row["target"],
                "observed_deg": row["observed"],
                "error_deg": row["error"],
                "angle_tolerance_deg": row["tolerance"],
                "max_torque_nm": row["max_effort"],
                "torque_limit_nm": row["effort_limit"],
                "force_pass": row["effort_pass"],
            })
        else:
            row.update({
                "target_mm": row["target"],
                "observed_mm": row["observed"],
                "error_mm": row["error"],
                "position_tolerance_mm": row["tolerance"],
                "max_force_n": row["max_effort"],
                "force_limit_n": row["effort_limit"],
                "force_pass": row["effort_pass"],
            })
        rows.append(row)
    return rows


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _round(value: Any, digits: int = 3) -> float | None:
    number = _finite(value)
    return None if number is None else round(number, digits)


def _whole_coordinate_summary(
    whole: dict[str, Any],
    plan: dict[str, Any],
    task_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the common coordinate/effort summary for one whole-machine result."""

    coordinate_kind = str(plan.get("coordinate_kind") or "").lower()
    is_clock = coordinate_kind == "clock_multifunction" or any(
        item.get("coordinate_kind") == "clock_multifunction"
        for item in task_rows
    )
    is_angular = coordinate_kind == "rotation" or any(
        item.get("coordinate_kind") == "angular" for item in task_rows
    )
    is_rack_pinion = coordinate_kind == "rack_pinion_progress"
    angular_force_control = (
        is_angular
        and _finite(plan.get("max_control_force_n")) is not None
        and _finite(plan.get("max_control_torque_nm")) is None
    )
    if is_clock:
        return {
            "kind": "clock_multifunction",
            "name": "多功能机械闹钟",
            "unit": "",
            "initial": 0.0,
            "tolerance_label": "按各功能独立门槛",
            "tolerance": None,
            "effort_label": "仅允许指定输入轴",
            "effort_unit": "",
            "effort_limit": None,
            "constraint_label": "功能闭环",
            "constraint_unit": "项",
            "constraint_value": sum(
                item.get("status") == "passed" for item in task_rows
            ),
            "secondary_constraint_label": "独立任务",
            "secondary_constraint_unit": "项",
            "secondary_constraint_value": len(task_rows),
        }
    if is_angular:
        return {
            "kind": "angular",
            "name": "角度",
            "unit": "°",
            "initial": math.degrees(
                _finite(plan.get("home_angle_rad")) or 0.0
            ),
            "tolerance_label": "角度误差",
            "tolerance": math.degrees(
                _finite(plan.get("angle_tolerance_rad")) or 0.0
            ),
            "effort_label": "控制力" if angular_force_control else "控制扭矩",
            "effort_unit": "N" if angular_force_control else "N·m",
            "effort_limit": _finite(
                plan.get(
                    "max_control_force_n"
                    if angular_force_control
                    else "max_control_torque_nm"
                )
            ),
            "constraint_label": "最大转轴漂移",
            "constraint_unit": "mm",
            "constraint_value": (
                _finite(whole.get("max_pivot_drift_m")) or 0.0
            )
            * 1000,
            "secondary_constraint_label": "最大轴外转动",
            "secondary_constraint_unit": "°",
            "secondary_constraint_value": math.degrees(
                _finite(whole.get("max_off_axis_rotation_rad")) or 0.0
            ),
        }
    if is_rack_pinion:
        return {
            "kind": "linear",
            "name": "位置",
            "unit": "mm",
            "initial": 0.0,
            "tolerance_label": "位置误差",
            "tolerance": (
                _finite(plan.get("progress_tolerance_m")) or 0.0
            )
            * 1000,
            "effort_label": "控制扭矩",
            "effort_unit": "N·m",
            "effort_limit": _finite(plan.get("max_control_torque_nm")),
            "constraint_label": "最大横向偏移（观测）",
            "constraint_unit": "mm",
            "constraint_value": (
                _finite(whole.get("max_lateral_shift_m")) or 0.0
            )
            * 1000,
            "secondary_constraint_label": "最大转动（观测）",
            "secondary_constraint_unit": "°",
            "secondary_constraint_value": math.degrees(
                _finite(whole.get("max_rotation_rad")) or 0.0
            ),
        }
    return {
        "kind": coordinate_kind or "linear",
        "name": "位置",
        "unit": "mm",
        "initial": 0.0,
        "tolerance_label": "位置误差",
        "tolerance": (
            _finite(plan.get("position_tolerance_m")) or 0.0
        )
        * 1000,
        "effort_label": "控制力",
        "effort_unit": "N",
        "effort_limit": _finite(plan.get("max_control_force_n")),
        "constraint_label": "最大横向偏移（观测）",
        "constraint_unit": "mm",
        "constraint_value": (
            _finite(whole.get("max_lateral_shift_m")) or 0.0
        )
        * 1000,
        "secondary_constraint_label": "最大转动（观测）",
        "secondary_constraint_unit": "°",
        "secondary_constraint_value": math.degrees(
            _finite(whole.get("max_rotation_rad")) or 0.0
        ),
    }


def _additional_whole_machine_view(
    *,
    ctx: ReadContext,
    run: Path,
    registration: dict[str, Any],
    mechanisms_by_capsule: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Normalize one non-primary registered whole-machine result."""

    result_path = ctx.resolve(registration.get("result_file"))
    result = ctx.read_json(result_path, role="whole-machine result")
    plan_path = result_path.parent / "execution_plan.json" if result_path else None
    execution_plan = (
        ctx.read_json(plan_path, role="whole-machine execution plan")
        if plan_path
        else {}
    )
    capsule_id = str(
        registration.get("capsule_id")
        or execution_plan.get("capsule_id")
        or result.get("capsule_id")
        or ""
    )
    mechanism = mechanisms_by_capsule.get(capsule_id, {})
    certificate_path = ctx.resolve(
        mechanism.get("certificate")
        or (
            f"{ctx.relative(run)}/results/{capsule_id}/certificate.json"
            if capsule_id
            else None
        )
    )
    replacement_path = ctx.resolve(
        mechanism.get("replacement")
        or (
            f"{ctx.relative(run)}/results/{capsule_id}/replacement.json"
            if capsule_id
            else None
        )
    )
    certificate = ctx.read_json(certificate_path, role="mechanism certificate")
    replacement = ctx.read_json(replacement_path, role="replacement")
    certificate_sha256 = _sha256_file(certificate_path)
    replacement_sha256 = _sha256_file(replacement_path)
    package_binding_sha256 = str(
        (certificate.get("package") or {}).get("package_binding_sha256") or ""
    )
    certificate_status = _status_from_value(certificate.get("verdict"))
    replacement_status = _status_from_value(replacement.get("status"))
    replacement_stale_reasons: list[str] = []
    if replacement_status == "passed":
        if (
            replacement.get("certificate_sha256")
            and replacement.get("certificate_sha256") != certificate_sha256
        ):
            replacement_stale_reasons.append(
                "Replacement 绑定的证书哈希与当前证书文件不一致。"
            )
        if (
            replacement.get("package_binding_sha256")
            and replacement.get("package_binding_sha256")
            != package_binding_sha256
        ):
            replacement_stale_reasons.append(
                "Replacement 与证书的设计包绑定不一致。"
            )
    if replacement_stale_reasons:
        replacement_status = "stale"
    status = _status_from_value(result.get("status"))
    replacement_policy = str(
        mechanism.get("replacement_policy")
        or result.get("replacement_policy")
        or "certified_mechanism"
    )
    stale_reasons: list[str] = []
    if status == "passed":
        if certificate_status != "passed":
            stale_reasons.append(
                "整机结果对应的机构证书缺失或未通过。"
            )
        if (
            replacement_policy != "reference_only_single_mechanism"
            and replacement_status != "passed"
        ):
            stale_reasons.append(
                "整机结果对应的 Replacement 缺失或未启用。"
            )
        if (
            result.get("certificate_sha256")
            and result.get("certificate_sha256") != certificate_sha256
        ):
            stale_reasons.append(
                "整机结果绑定的证书哈希与当前证书文件不一致。"
            )
        if (
            result.get("replacement_sha256")
            and result.get("replacement_sha256") != replacement_sha256
        ):
            stale_reasons.append(
                "整机结果绑定的 Replacement 哈希与当前文件不一致。"
            )
        if (
            result.get("package_binding_sha256")
            and result.get("package_binding_sha256")
            != package_binding_sha256
        ):
            stale_reasons.append("整机结果与证书的设计包绑定不一致。")
    if stale_reasons:
        status = "stale"
    stage_status = (
        "passed"
        if replacement_policy == "reference_only_single_mechanism"
        else status
    )
    task_rows = _whole_task_rows(result, execution_plan)
    task_rows = [
        {**item, "capsule_id": capsule_id}
        for item in task_rows
    ]
    coordinate = _whole_coordinate_summary(result, execution_plan, task_rows)
    boundary = (
        "单机构包不启用简化整机替换；外部任务结论直接来自已认证 physical 轨迹。"
        if replacement_policy == "reference_only_single_mechanism"
        else "这是已认证简化替换的整机外部任务；不是高保真内部接触运行。"
        if status == "passed"
        else "尚未生成通过的整机任务证据；空表或未完成结果不能解释为已完成。"
    )
    view = {
        "capsule_id": capsule_id,
        "status": stage_status,
        "status_raw": str(result.get("status") or "NOT_AVAILABLE"),
        "required": replacement_policy != "reference_only_single_mechanism",
        "replacement_policy": replacement_policy,
        "task_ids": result.get("task_ids") or [],
        "stages": task_rows,
        "task_passed": sum(
            item.get("status") == "passed" for item in task_rows
        ),
        "task_total": len(task_rows),
        "coordinate": coordinate,
        "max_lateral_shift_mm": _round(
            (_finite(result.get("max_lateral_shift_m")) or 0.0) * 1000,
            6,
        ),
        "max_rotation_deg": _round(
            math.degrees(_finite(result.get("max_rotation_rad")) or 0.0),
            6,
        ),
        "max_pivot_drift_mm": _round(
            (_finite(result.get("max_pivot_drift_m")) or 0.0) * 1000,
            6,
        ),
        "max_off_axis_rotation_deg": _round(
            math.degrees(
                _finite(result.get("max_off_axis_rotation_rad")) or 0.0
            ),
            9,
        ),
        "position_tolerance_mm": (
            _round(
                (_finite(execution_plan.get("position_tolerance_m")) or 0.0)
                * 1000,
                4,
            )
            if execution_plan.get("position_tolerance_m") is not None
            else None
        ),
        "angle_tolerance_deg": (
            _round(
                math.degrees(
                    _finite(execution_plan.get("angle_tolerance_rad")) or 0.0
                ),
                4,
            )
            if execution_plan.get("angle_tolerance_rad") is not None
            else None
        ),
        "max_stage_duration_s": _round(
            execution_plan.get("max_stage_duration_s"), 4
        ),
        "max_control_force_n": _round(
            execution_plan.get("max_control_force_n"), 4
        ),
        "max_control_torque_nm": _round(
            execution_plan.get("max_control_torque_nm"), 4
        ),
        "sample_count": result.get("sample_count"),
        "strict_non_claims": result.get("strict_non_claims") or {},
        "stale_reasons": stale_reasons,
        "boundary": boundary,
    }
    artifact = _artifact(
        ctx,
        result_path,
        label=f"整机任务结果 · {capsule_id or '未标识机构'}",
        kind="whole-machine",
        stage_id="whole-machine",
        role="output",
        summary=(
            f"记录 {capsule_id or '该机构'} 的 {len(task_rows)} 项整机任务结果。"
        ),
        status=stage_status,
        sha256=_sha256_file(result_path),
    )
    artifact["details"] = {
        "type": "whole-machine",
        **view,
    }
    certificate_artifact = _artifact(
        ctx,
        certificate_path,
        label=f"运动学证书 · {capsule_id or '未标识机构'}",
        kind="certificate",
        stage_id="certificate",
        role="output",
        summary=f"{capsule_id or '该机构'} 的独立运动学证书。",
        status=certificate_status,
        sha256=certificate_sha256,
    )
    certificate_artifact["details"].update(
        {
            "type": "certificate",
            "capsule_id": capsule_id,
            "status": certificate_status,
            "verdict": str(certificate.get("verdict") or "NOT_AVAILABLE"),
            "certificate_sha256": certificate_sha256,
            "package_binding_sha256": package_binding_sha256,
        }
    )
    replacement_artifact = _artifact(
        ctx,
        replacement_path,
        label=f"简化机构启用记录 · {capsule_id or '未标识机构'}",
        kind="replacement",
        stage_id="replacement",
        role="output",
        summary=f"{capsule_id or '该机构'} 的独立 Replacement 绑定。",
        status=replacement_status,
        sha256=replacement_sha256,
    )
    replacement_artifact["details"].update(
        {
            "type": "replacement",
            "capsule_id": capsule_id,
            "status": str(replacement.get("status") or "NOT_AVAILABLE"),
            "normalized_status": replacement_status,
            "certificate_sha256": str(
                replacement.get("certificate_sha256") or ""
            ),
            "package_binding_sha256": str(
                replacement.get("package_binding_sha256") or ""
            ),
            "stale_reasons": replacement_stale_reasons,
        }
    )
    plan_artifact = (
        _artifact(
            ctx,
            plan_path,
            label=f"整机任务目标与容差 · {capsule_id or '未标识机构'}",
            kind="whole-machine-plan",
            stage_id="whole-machine",
            role="input",
            summary=f"记录 {capsule_id or '该机构'} 的任务输入与公开门槛。",
            status="passed" if execution_plan else "not_started",
            sha256=_sha256_file(plan_path),
        )
        if plan_path
        else None
    )
    if plan_artifact is not None:
        plan_artifact["details"].update({"capsule_id": capsule_id})
    artifacts = [certificate_artifact, replacement_artifact, artifact]
    if plan_artifact is not None:
        artifacts.append(plan_artifact)
    return view, artifacts



def _rack_pinion_comparison_checks(
    load: dict[str, Any],
    physical: dict[str, Any],
    reduced: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def numeric(
        label: str,
        physical_value: float | None,
        reduced_value: float | None,
        limit: float,
        *,
        rule: str,
        unit: str,
        note: str = "",
    ) -> None:
        def result(value: float | None) -> tuple[bool | None, float | None]:
            if value is None:
                return None, None
            passed = value <= limit if rule == "max" else value >= limit
            margin = limit - value if rule == "max" else value - limit
            return passed, _round(margin, 4)

        physical_pass, physical_margin = result(physical_value)
        reduced_pass, reduced_margin = result(reduced_value)
        rows.append(
            {
                "label": label,
                "rule": "≤" if rule == "max" else "≥",
                "limit": _round(limit, 4),
                "unit": unit,
                "physical": _round(physical_value, 4),
                "reduced": _round(reduced_value, 4),
                "physical_pass": physical_pass,
                "reduced_pass": reduced_pass,
                "physical_margin": physical_margin,
                "reduced_margin": reduced_margin,
                "note": note,
            }
        )

    millimetres = 1000.0
    numeric(
        "导向漂移",
        (_finite(physical.get("max_support_drift_m")) or 0.0) * millimetres,
        (_finite(reduced.get("max_support_drift_m")) or 0.0) * millimetres,
        (_finite(thresholds.get("max_support_drift_m")) or 0.0) * millimetres,
        rule="max",
        unit="mm",
        note="齿轮轴承和齿条导轨不能明显偏离设计位置。",
    )
    numeric(
        "轴外转动",
        math.degrees(_finite(physical.get("max_off_axis_rotation_rad")) or 0.0),
        math.degrees(_finite(reduced.get("max_off_axis_rotation_rad")) or 0.0),
        math.degrees(_finite(thresholds.get("max_off_axis_rotation_rad")) or 0.0),
        rule="max",
        unit="°",
        note="齿条只应直线升降，不能发生多余翻转。",
    )
    numeric(
        "真实啮合误差",
        (_finite(physical.get("max_coupling_error_m")) or 0.0) * millimetres,
        None,
        (_finite(thresholds.get("max_physical_coupling_error_m")) or 0.0) * millimetres,
        rule="max",
        unit="mm",
        note="有限刚度齿面允许轻微压缩，但不能失去正确传动关系。",
    )
    numeric(
        "简化齿轮比误差",
        None,
        (_finite(reduced.get("max_coupling_error_m")) or 0.0) * millimetres,
        (_finite(thresholds.get("max_reduced_coupling_error_m")) or 0.0) * millimetres,
        rule="max",
        unit="mm",
        note="简化模型必须保持齿轮转角与齿条位移的比例。",
    )

    physical_reaction = _finite(physical.get("max_bushing_reaction_force_n")) or 0.0
    reduced_reaction = _finite(reduced.get("max_bushing_reaction_force_n")) or 0.0
    rows.append(
        {
            "label": "支撑模型确实不同",
            "rule": "真实>0，简化=0",
            "limit": "分支语义",
            "unit": "N",
            "physical": _round(physical_reaction, 4),
            "reduced": _round(reduced_reaction, 4),
            "physical_pass": physical_reaction > 1e-9,
            "reduced_pass": reduced_reaction <= 1e-9,
            "physical_margin": None,
            "reduced_margin": None,
            "note": "真实分支使用有限轴承/导轨反力，简化分支使用理想约束。",
        }
    )

    home = _finite(thresholds.get("home_progress_m")) or 0.0
    outbound = _finite(thresholds.get("outbound_progress_m")) or 0.0
    tolerance = _finite(thresholds.get("output_progress_tolerance_m")) or 0.0
    numeric(
        "下限不越界",
        (_finite(physical.get("min_progress_m")) or 0.0) * millimetres,
        (_finite(reduced.get("min_progress_m")) or 0.0) * millimetres,
        (home - tolerance) * millimetres,
        rule="min",
        unit="mm",
    )
    numeric(
        "上限不越界",
        (_finite(physical.get("max_progress_m")) or 0.0) * millimetres,
        (_finite(reduced.get("max_progress_m")) or 0.0) * millimetres,
        (outbound + tolerance) * millimetres,
        rule="max",
        unit="mm",
    )

    program = str(load.get("program") or "")
    if program == "bidirectional_rack_pinion":
        directional = (
            ("crank_lift", "手摇升起"),
            ("crank_lower", "反向降下"),
        )
        limit = (_finite(thresholds.get("min_directional_progress_m")) or 0.0) * millimetres
        for phase_id, label in directional:
            numeric(
                label,
                (_finite((physical.get("phase_directional_travel_m") or {}).get(phase_id)) or 0.0) * millimetres,
                (_finite((reduced.get("phase_directional_travel_m") or {}).get(phase_id)) or 0.0) * millimetres,
                limit,
                rule="min",
                unit="mm",
            )
    elif program == "zero_input_gravity":
        numeric(
            "不摇动时的下滑",
            (_finite(physical.get("max_abs_progress_delta_m")) or 0.0) * millimetres,
            (_finite(reduced.get("max_abs_progress_delta_m")) or 0.0) * millimetres,
            (_finite(thresholds.get("max_zero_input_drift_m")) or 0.0) * millimetres,
            rule="max",
            unit="mm",
        )
    elif program == "transverse_rack_probe":
        numeric(
            "侧向扰动下的误升降",
            (_finite(physical.get("max_abs_progress_delta_m")) or 0.0) * millimetres,
            (_finite(reduced.get("max_abs_progress_delta_m")) or 0.0) * millimetres,
            (_finite(thresholds.get("max_transverse_progress_m")) or 0.0) * millimetres,
            rule="max",
            unit="mm",
        )
    elif program == "task_rack_roundtrip":
        lift_physical = _finite((physical.get("phase_progress_bounds_m") or {}).get("crank_lift", {}).get("max_m"))
        lift_reduced = _finite((reduced.get("phase_progress_bounds_m") or {}).get("crank_lift", {}).get("max_m"))
        numeric(
            "升闸目标到达",
            None if lift_physical is None else lift_physical * millimetres,
            None if lift_reduced is None else lift_reduced * millimetres,
            (outbound - tolerance) * millimetres,
            rule="min",
            unit="mm",
        )
        numeric(
            "降闸回到原位",
            abs((_finite(physical.get("final_progress_m")) or 0.0) - home) * millimetres,
            abs((_finite(reduced.get("final_progress_m")) or 0.0) - home) * millimetres,
            tolerance * millimetres,
            rule="max",
            unit="mm",
        )
        slip_physical = _finite(
            (physical.get("phase_downward_slip_m") or {}).get("zero_input_hold")
        )
        slip_reduced = _finite(
            (reduced.get("phase_downward_slip_m") or {}).get("zero_input_hold")
        )
        if slip_physical is not None or slip_reduced is not None:
            numeric(
                "松手防坠下滑",
                None if slip_physical is None else slip_physical * millimetres,
                None if slip_reduced is None else slip_reduced * millimetres,
                (_finite(thresholds.get("max_anti_drop_slip_m")) or tolerance)
                * millimetres,
                rule="max",
                unit="mm",
            )
    return rows

def _clock_comparison_checks(
    load: dict[str, Any],
    physical: dict[str, Any],
    reduced: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expose the same clock gates used by the validator in readable form."""

    rows: list[dict[str, Any]] = []

    def numeric(
        label: str,
        physical_value: float | None,
        reduced_value: float | None,
        limit: float,
        *,
        rule: str,
        unit: str,
        note: str = "",
        reduced_limit: float | None = None,
    ) -> None:
        def result(value: float | None, boundary: float) -> tuple[bool | None, float | None]:
            if value is None:
                return None, None
            passed = value <= boundary if rule == "max" else value >= boundary
            margin = boundary - value if rule == "max" else value - boundary
            return passed, _round(margin, 4)

        physical_pass, physical_margin = result(physical_value, limit)
        reduced_boundary = limit if reduced_limit is None else reduced_limit
        reduced_pass, reduced_margin = result(reduced_value, reduced_boundary)
        rows.append(
            {
                "label": label,
                "rule": "≤" if rule == "max" else "≥",
                "limit": _round(limit, 4),
                "reduced_limit": _round(reduced_boundary, 4),
                "unit": unit,
                "physical": _round(physical_value, 4),
                "reduced": _round(reduced_value, 4),
                "physical_pass": physical_pass,
                "reduced_pass": reduced_pass,
                "physical_margin": physical_margin,
                "reduced_margin": reduced_margin,
                "note": note,
            }
        )

    numeric(
        "轴承支撑漂移",
        (_finite(physical.get("max_support_drift_m")) or 0.0) * 1000.0,
        (_finite(reduced.get("max_support_drift_m")) or 0.0) * 1000.0,
        (_finite(thresholds.get("max_support_drift_m")) or 0.0) * 1000.0,
        rule="max",
        unit="mm",
    )
    numeric(
        "轴外摆动",
        math.degrees(_finite(physical.get("max_off_axis_rotation_rad")) or 0.0),
        math.degrees(_finite(reduced.get("max_off_axis_rotation_rad")) or 0.0),
        math.degrees(_finite(thresholds.get("max_off_axis_rotation_rad")) or 0.0),
        rule="max",
        unit="°",
    )
    numeric(
        "齿轮/传动约束误差",
        math.degrees(_finite(physical.get("max_mesh_error_rad")) or 0.0),
        math.degrees(_finite(reduced.get("max_mesh_error_rad")) or 0.0),
        math.degrees(_finite(thresholds.get("max_mesh_error_rad")) or 0.0),
        reduced_limit=math.degrees(0.005),
        rule="max",
        unit="°",
        note="真实分支允许有限柔性啮合；简化分支使用更严格的解析约束误差。",
    )
    program = str(load.get("program") or "")
    if program in {"zero_input_clock_train", "transverse_clock_probe"}:
        numeric(
            "表盘无意外走动",
            math.degrees(_finite(physical.get("output_span_rad")) or 0.0),
            math.degrees(_finite(reduced.get("output_span_rad")) or 0.0),
            math.degrees(_finite(thresholds.get("output_angle_tolerance_rad")) or 0.0),
            rule="max",
            unit="°",
        )
        return rows

    expected_hour = _finite(thresholds.get("expected_hour_to_minute_ratio")) or (1.0 / 12.0)
    expected_second = _finite(thresholds.get("expected_second_to_minute_ratio")) or 60.0
    ratio_tolerance = _finite(thresholds.get("hand_ratio_relative_tolerance")) or 0.02
    numeric(
        "时针/分针比例相对误差",
        abs((_finite(physical.get("hour_to_minute_ratio")) or 0.0) / expected_hour - 1.0) * 100.0,
        abs((_finite(reduced.get("hour_to_minute_ratio")) or 0.0) / expected_hour - 1.0) * 100.0,
        ratio_tolerance * 100.0,
        rule="max",
        unit="%",
    )
    numeric(
        "秒针/分针比例相对误差",
        abs((_finite(physical.get("second_to_minute_ratio")) or 0.0) / expected_second - 1.0) * 100.0,
        abs((_finite(reduced.get("second_to_minute_ratio")) or 0.0) / expected_second - 1.0) * 100.0,
        ratio_tolerance * 100.0,
        rule="max",
        unit="%",
    )
    rows.append(
        {
            "label": "三针运行方向",
            "rule": "=",
            "limit": "一致",
            "unit": "",
            "physical": "一致" if physical.get("hand_direction_consistent") else "不一致",
            "reduced": "一致" if reduced.get("hand_direction_consistent") else "不一致",
            "physical_pass": bool(physical.get("hand_direction_consistent")),
            "reduced_pass": bool(reduced.get("hand_direction_consistent")),
            "physical_margin": None,
            "reduced_margin": None,
            "note": "时针、分针、秒针在表盘上必须同向推进。",
        }
    )
    if program == "manual_time_setting_clock":
        numeric(
            "后冠确实转动",
            math.degrees(_finite(physical.get("setting_input_span_rad")) or 0.0),
            math.degrees(_finite(reduced.get("setting_input_span_rad")) or 0.0),
            math.degrees(0.02),
            rule="min",
            unit="°",
        )
        numeric(
            "后冠带动表针",
            math.degrees(_finite(physical.get("output_span_rad")) or 0.0),
            math.degrees(_finite(reduced.get("output_span_rad")) or 0.0),
            math.degrees(0.02),
            rule="min",
            unit="°",
        )
        return rows

    numeric(
        "走时阶段向前推进",
        math.degrees(_finite((physical.get("phase_output_travel_rad") or {}).get("wind_and_advance")) or 0.0),
        math.degrees(_finite((reduced.get("phase_output_travel_rad") or {}).get("wind_and_advance")) or 0.0),
        math.degrees(0.02),
        rule="min",
        unit="°",
    )
    numeric(
        "反向压力探针产生回退",
        -math.degrees(_finite((physical.get("phase_output_travel_rad") or {}).get("controlled_return")) or 0.0),
        -math.degrees(_finite((reduced.get("phase_output_travel_rad") or {}).get("controlled_return")) or 0.0),
        math.degrees(0.01),
        rule="min",
        unit="°",
        note="这是耐受性探针，不要求日常走时自动倒转。",
    )
    if program in {"task_clock_cycle", "alarm_clock_cycle"}:
        numeric(
            "铃锤摆幅",
            math.degrees(_finite(physical.get("alarm_swing_rad")) or 0.0),
            math.degrees(_finite(reduced.get("alarm_swing_rad")) or 0.0),
            math.degrees(_finite(thresholds.get("minimum_alarm_swing_rad")) or 0.0),
            rule="min",
            unit="°",
        )
    if program == "task_clock_cycle":
        target = (
            (_finite(thresholds.get("outbound_output_angle_rad")) or 0.0)
            - (_finite(thresholds.get("output_angle_tolerance_rad")) or 0.0)
        )
        numeric(
            "分针任务目标到达",
            math.degrees(_finite(physical.get("max_output_angle_rad")) or 0.0),
            math.degrees(_finite(reduced.get("max_output_angle_rad")) or 0.0),
            math.degrees(target),
            rule="min",
            unit="°",
        )
    return rows


def _comparison_checks(
    load: dict[str, Any],
    physical: dict[str, Any],
    reduced: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    if thresholds.get("source") == "validated_clock_gear_family_spec":
        return _clock_comparison_checks(load, physical, reduced, thresholds)
    if "max_physical_coupling_error_m" in thresholds:
        return _rack_pinion_comparison_checks(load, physical, reduced, thresholds)

    rows: list[dict[str, Any]] = []

    def numeric(label: str, physical_value: float | None, reduced_value: float | None, limit: float, *, rule: str, unit: str, note: str = "") -> None:
        def result(value: float | None) -> tuple[bool | None, float | None]:
            if value is None:
                return None, None
            passed = value <= limit if rule == "max" else value >= limit
            margin = (limit - value) if rule == "max" else (value - limit)
            return passed, _round(margin, 4)
        physical_pass, physical_margin = result(physical_value)
        reduced_pass, reduced_margin = result(reduced_value)
        rows.append({
            "label": label,
            "rule": "≤" if rule == "max" else "≥",
            "limit": _round(limit, 4),
            "unit": unit,
            "physical": _round(physical_value, 4),
            "reduced": _round(reduced_value, 4),
            "physical_pass": physical_pass,
            "reduced_pass": reduced_pass,
            "physical_margin": physical_margin,
            "reduced_margin": reduced_margin,
            "note": note,
        })

    angular = any(
        key in physical or key in reduced
        for key in ("max_pivot_drift_m", "max_off_axis_rotation_rad", "max_abs_angle_rad")
    ) or "min_directional_angle_rad" in thresholds
    if angular:
        numeric(
            "转轴漂移",
            (_finite(physical.get("max_pivot_drift_m")) or 0.0) * 1000,
            (_finite(reduced.get("max_pivot_drift_m")) or 0.0) * 1000,
            (_finite(thresholds.get("max_pivot_drift_m")) or 0.0) * 1000,
            rule="max", unit="mm",
            note="转轴中心相对设计轴线的最大偏移。",
        )
        numeric(
            "轴外转动",
            math.degrees(_finite(physical.get("max_off_axis_rotation_rad")) or 0.0),
            math.degrees(_finite(reduced.get("max_off_axis_rotation_rad")) or 0.0),
            math.degrees(_finite(thresholds.get("max_off_axis_rotation_rad")) or 0.0),
            rule="max", unit="°",
            note="除目标转轴以外的多余转动。",
        )
        program = str(load.get("program") or "")
        phase_labels = {
            "forward_torque": "正向顺载转角",
            "reverse_torque": "反向顺载转角",
            "task_outbound_torque": "打开顺载转角",
            "task_return_torque": "关闭顺载转角",
        }
        if program == "bidirectional_rotation":
            for phase in load.get("phases") or []:
                if not isinstance(phase, dict) or phase.get("wrench_kind") == "none":
                    continue
                phase_id = str(phase.get("phase_id") or "phase")
                numeric(
                    phase_labels.get(phase_id, f"{phase_id} 顺载转角"),
                    math.degrees(_finite((physical.get("phase_directional_travel_rad") or {}).get(phase_id)) or 0.0),
                    math.degrees(_finite((reduced.get("phase_directional_travel_rad") or {}).get(phase_id)) or 0.0),
                    math.degrees(_finite(thresholds.get("min_directional_angle_rad")) or 0.0),
                    rule="min", unit="°",
                )
        elif program == "transverse_torque_probe":
            numeric(
                "横向扰动转角",
                math.degrees(_finite(physical.get("max_abs_angle_rad")) or 0.0),
                math.degrees(_finite(reduced.get("max_abs_angle_rad")) or 0.0),
                math.degrees(_finite(thresholds.get("max_transverse_angle_rad")) or 0.0),
                rule="max", unit="°",
                note="侧向扭矩下，机构不应沿主转轴大幅转动。",
            )
        elif program == "task_angle_roundtrip":
            tolerance_rad = _finite(thresholds.get("min_directional_angle_rad")) or 0.0
            physical_bounds = physical.get("phase_angle_bounds_rad") or {}
            reduced_travel = reduced.get("phase_directional_travel_rad") or {}
            for phase in load.get("phases") or []:
                if not isinstance(phase, dict) or phase.get("wrench_kind") == "none":
                    continue
                phase_id = str(phase.get("phase_id") or "phase")
                numeric(
                    f"简化模型{phase_labels.get(phase_id, phase_id)}",
                    None,
                    math.degrees(_finite(reduced_travel.get(phase_id)) or 0.0),
                    math.degrees(tolerance_rad),
                    rule="min", unit="°",
                )
                stop = str(phase.get("stop_condition") or "")
                if not stop:
                    continue
                target_rad = _finite(phase.get("target_position_si"))
                if target_rad is None:
                    target_rad = _finite(phase.get("target_angle_rad"))
                if target_rad is None:
                    continue
                bounds = physical_bounds.get(phase_id) or {}
                is_lower = stop in {"position_at_least", "progress_at_least"}
                observed_rad = _finite(bounds.get("max_rad" if is_lower else "min_rad")) or 0.0
                boundary_rad = target_rad - tolerance_rad if is_lower else target_rad + tolerance_rad
                passed = observed_rad >= boundary_rad if is_lower else observed_rad <= boundary_rad
                margin_rad = observed_rad - boundary_rad if is_lower else boundary_rad - observed_rad
                rows.append({
                    "label": f"真实模型{phase_labels.get(phase_id, phase_id)}目标",
                    "rule": "≥" if is_lower else "≤",
                    "limit": _round(math.degrees(boundary_rad), 4),
                    "unit": "°",
                    "physical": _round(math.degrees(observed_rad), 4),
                    "reduced": None,
                    "physical_pass": passed,
                    "reduced_pass": None,
                    "physical_margin": _round(math.degrees(margin_rad), 4),
                    "reduced_margin": None,
                    "note": f"目标 {math.degrees(target_rad):.3f}°，允许误差 {math.degrees(tolerance_rad):.3f}°。",
                })
        if "contact_observed" in physical:
            rows.append({
                "label": "真实接触证据",
                "rule": "=",
                "limit": "必须观察到",
                "unit": "",
                "physical": "是" if physical.get("contact_observed") else "否",
                "reduced": "不适用",
                "physical_pass": bool(physical.get("contact_observed")),
                "reduced_pass": None,
                "physical_margin": None,
                "reduced_margin": None,
                "note": "真实分支必须记录到至少一次几何接触；简化分支不适用。",
            })
        return rows

    numeric(
        "横向漂移",
        (_finite(physical.get("max_lateral_shift_m")) or 0.0) * 1000,
        (_finite(reduced.get("max_lateral_shift_m")) or 0.0) * 1000,
        (_finite(thresholds.get("max_lateral_shift_m")) or 0.0) * 1000,
        rule="max", unit="mm",
    )
    numeric(
        "旋转偏差",
        math.degrees(_finite(physical.get("max_rotation_rad")) or 0.0),
        math.degrees(_finite(reduced.get("max_rotation_rad")) or 0.0),
        math.degrees(_finite(thresholds.get("max_rotation_rad")) or 0.0),
        rule="max", unit="°",
    )
    program = str(load.get("program") or "")
    if program in {"bidirectional_axis", "task_range_roundtrip"}:
        numeric(
            "轴向纯度",
            _finite(physical.get("axis_purity")),
            _finite(reduced.get("axis_purity")),
            _finite(thresholds.get("min_axis_purity")) or 0.0,
            rule="min", unit="",
            note="沿目标轴的位移占全部位移的比例；1 表示完全沿目标轴运动。",
        )
        phase_ids = [
            str(item.get("phase_id"))
            for item in load.get("phases") or []
            if isinstance(item, dict) and item.get("wrench_kind") != "none"
        ]
        physical_travel = [_finite((physical.get("phase_directional_travel_m") or {}).get(phase_id)) for phase_id in phase_ids]
        reduced_travel = [_finite((reduced.get("phase_directional_travel_m") or {}).get(phase_id)) for phase_id in phase_ids]
        physical_min = min((item for item in physical_travel if item is not None), default=None)
        reduced_min = min((item for item in reduced_travel if item is not None), default=None)
        numeric(
            "最小顺载运动",
            None if program == "task_range_roundtrip" else (physical_min or 0.0) * 1000,
            (reduced_min or 0.0) * 1000,
            (_finite(thresholds.get("min_directional_travel_m")) or 0.0) * 1000,
            rule="min", unit="mm",
            note="完整行程工况的真实分支改用目标到达门控。" if program == "task_range_roundtrip" else "",
        )
    elif program == "zero_input_gravity":
        numeric(
            "零输入总漂移",
            (_finite(physical.get("max_total_displacement_m")) or 0.0) * 1000,
            (_finite(reduced.get("max_total_displacement_m")) or 0.0) * 1000,
            (_finite(thresholds.get("max_zero_input_drift_m")) or 0.0) * 1000,
            rule="max", unit="mm",
        )
    elif program == "transverse_probe":
        numeric(
            "横向扰动总位移",
            (_finite(physical.get("max_total_displacement_m")) or 0.0) * 1000,
            (_finite(reduced.get("max_total_displacement_m")) or 0.0) * 1000,
            (_finite(thresholds.get("max_transverse_motion_m")) or 0.0) * 1000,
            rule="max", unit="mm",
        )

    if program == "task_range_roundtrip":
        tolerance = (_finite(thresholds.get("min_directional_travel_m")) or 0.0) * 1000
        bounds_by_phase = physical.get("phase_progress_bounds_m") or {}
        for phase in load.get("phases") or []:
            if not isinstance(phase, dict) or phase.get("stop_condition") is None:
                continue
            phase_id = str(phase.get("phase_id") or "phase")
            target_mm = (_finite(phase.get("target_progress_m")) or 0.0) * 1000
            bounds = bounds_by_phase.get(phase_id) or {}
            is_lower_bound = phase.get("stop_condition") == "progress_at_least"
            observed_mm = ((_finite(bounds.get("max_m")) or 0.0) if is_lower_bound else (_finite(bounds.get("min_m")) or 0.0)) * 1000
            boundary_mm = target_mm - tolerance if is_lower_bound else target_mm + tolerance
            passed = observed_mm >= boundary_mm if is_lower_bound else observed_mm <= boundary_mm
            margin_mm = observed_mm - boundary_mm if is_lower_bound else boundary_mm - observed_mm
            rows.append({
                "label": f"{phase_id} 单向目标边界",
                "rule": "≥" if is_lower_bound else "≤",
                "limit": _round(boundary_mm, 4),
                "unit": "mm",
                "physical": _round(observed_mm, 4),
                "reduced": None,
                "physical_pass": passed,
                "reduced_pass": None,
                "physical_margin": _round(margin_mm, 4),
                "reduced_margin": None,
                "note": (
                    f"目标 {target_mm:.3f} mm，容差 {tolerance:.3f} mm；"
                    f"这是单向到达门控，沿任务方向越过目标仍算通过。"
                ),
            })

    if "contact_observed" in physical:
        rows.append({
            "label": "真实接触证据",
            "rule": "=",
            "limit": "必须观察到",
            "unit": "",
            "physical": "是" if physical.get("contact_observed") else "否",
            "reduced": "不适用",
            "physical_pass": bool(physical.get("contact_observed")),
            "reduced_pass": None,
            "physical_margin": None,
            "reduced_margin": None,
            "note": "真实运动记录中的“已观察接触”必须为“是”，表示轨迹里至少记录到一次几何接触；简化分支不适用。",
        })
    return rows


def _status_from_value(value: Any, *, missing: str = "not_started") -> str:
    text = str(value or "").upper()
    if not text:
        return missing
    if text == "COMPLETED_WITH_TIMEBOX":
        return "evidence_incomplete"
    if text in {
        "EVIDENCE_COMPLETE",
        "PASS",
        "PASSED",
        "SUCCEEDED",
        "COMPLETED",
        "ACCEPTED",
        "ACTIVE",
        "KINEMATIC_CERTIFIED",
        "TASK_CERTIFIED",
        "WHOLE_MACHINE_TASK_PASSED",
        "PHYSICAL_PREFLIGHT_PASSED",
        "PHYSICAL_DISPATCH_PREFLIGHT_PASSED",
        "LEGACY_SAMPLED_PHYSICAL_AUDIT_PASSED",
    }:
        return "evidence_complete" if text == "EVIDENCE_COMPLETE" else "passed"
    if text == "EVIDENCE_INCOMPLETE":
        return "evidence_incomplete"
    if text == "CAPABILITY_PROPOSAL_DRAFT_REQUIRED":
        return "not_started"
    if text in {
        "CAPABILITY_REVIEW_REQUIRED",
        "DESIGN_REVIEW_REQUIRED",
    }:
        return "blocked"
    if text == "ACCEPTED_WITH_WARNINGS" or "WARNING" in text:
        return "warning"
    if any(token in text for token in ("RUNNING", "STARTED", "EXECUTING")):
        return "running"
    if text in {"INACTIVE", "STALE", "INVALID", "SUPERSEDED"}:
        return "stale"
    if any(
        token in text
        for token in (
            "FAIL",
            "REJECT",
            "ERROR",
            "INCONCLUSIVE",
            "MISMATCH",
            "CANCEL",
            "TIMEOUT",
            "NOT_CERTIFIED",
        )
    ):
        return "failed"
    if any(token in text for token in ("BLOCK", "DEFER")):
        return "blocked"
    if any(token in text for token in ("PLAN", "PENDING", "READY", "QUEUED")):
        return "not_started"
    return missing


def _resolve_run_verdict(value: Any) -> dict[str, Any]:
    """Resolve only validator-owned run terminal states.

    Stage-local PASS-like values intentionally do not close a whole run.  This
    resolver is shared by the catalog and detail view so they cannot disagree
    merely because the detail page happens to reconstruct different stages.
    """

    raw = str(value or "").strip().upper()
    if raw in FINAL_PASS_STATUSES | LEGACY_GENERIC_PASS_STATUSES:
        status = "passed"
        final = True
        authority = "terminal_run_status"
    elif raw == "COMPLETED_WITH_TIMEBOX":
        status = "evidence_incomplete"
        final = False
        authority = "noncertifying_timebox_completion"
    elif raw in {"PENDING", "RUNNING", "ITERATION_SCOPE_COMPLETE"}:
        status = "running"
        final = False
        authority = (
            "active_iteration_status"
            if raw == "ITERATION_SCOPE_COMPLETE"
            else "active_run_status"
        )
    elif raw in {
        "PREPARING",
        "WAITING_FOR_RESOURCES",
        "RESOURCE_BLOCKED",
        "REUSING",
    }:
        status = "preparing"
        final = False
        authority = "nontrusted_queue_phase"
    elif raw == "CAPABILITY_PROPOSAL_DRAFT_REQUIRED":
        status = "not_started"
        final = False
        authority = "local_design_action_required"
    elif raw in {
        "SUPERSEDED",
        "DESIGN_SUPERSEDED_BEFORE_EXECUTION",
        "DESIGN_REVISED_BEFORE_VALIDATION",
    }:
        status = "stale"
        final = True
        authority = "terminal_run_status"
    elif raw in PREFINAL_STATUSES:
        status = "not_started"
        final = False
        authority = "preterminal_run_status"
    elif raw in {
        "CAPABILITY_REVIEW_REQUIRED",
        "DESIGN_REVIEW_REQUIRED",
    }:
        status = "blocked"
        final = True
        authority = "terminal_run_status"
    elif raw in {
        "FAILED",
        "NOT_CERTIFIED",
        "WHOLE_MACHINE_TASK_FAILED",
        "PHYSICAL_PREFLIGHT_FAILED",
        "PHYSICAL_DISPATCH_PREFLIGHT_FAILED",
    } or any(
        token in raw
        for token in (
            "FAIL",
            "REJECT",
            "ERROR",
            "INCONCLUSIVE",
            "MISMATCH",
            "CANCEL",
            "TIMEOUT",
            "NOT_CERTIFIED",
        )
    ):
        status = "failed"
        final = True
        authority = "terminal_run_status"
    elif any(token in raw for token in ("BLOCK", "DEFER")):
        status = "blocked"
        final = True
        authority = "terminal_run_status"
    else:
        status = "not_started"
        final = False
        authority = "unrecognized_or_preterminal_status"
    return {
        "status": status,
        "final": final,
        "authority": authority,
        "reason_code": raw or "NOT_AVAILABLE",
    }


def _evidence_health(
    *,
    verdict: dict[str, Any],
    canonical: bool,
    terminal_ready: bool,
    physical_audit: dict[str, Any] | None = None,
    conflict_reasons: Iterable[str] = (),
    assessment_scope: str = "full_registered_evidence",
) -> dict[str, Any]:
    reasons = [str(item) for item in conflict_reasons if str(item)]
    audit = physical_audit or {}
    audit_status = _status_from_value(audit.get("status"))
    if (
        verdict["status"] == "passed"
        and audit
        and audit_status != "passed"
    ):
        reasons.append("PHYSICAL_AUDIT_CONFLICT")

    legacy_unverified = bool(
        verdict["final"] and not canonical and not terminal_ready
    )
    if legacy_unverified:
        status = "legacy_unverified"
        reasons.insert(0, "LEGACY_TERMINAL_UNVERIFIED")
    elif reasons:
        status = "conflict"
    elif verdict["status"] == "stale":
        status = "stale"
    elif terminal_ready:
        status = "complete"
    else:
        status = "incomplete"
    return {
        "status": status,
        "status_label": EVIDENCE_HEALTH_LABELS[status],
        "reason_codes": list(dict.fromkeys(reasons)),
        "terminal_inventory_ready": bool(terminal_ready),
        "assessment_scope": assessment_scope,
    }


def _combine_status(statuses: Iterable[str], *, empty: str = "not_started") -> str:
    values = list(statuses)
    if not values:
        return empty
    for status in (
        "failed",
        "stale",
        "blocked",
        "evidence_incomplete",
        "awaiting_review",
        "preparing",
        "running",
        "warning",
    ):
        if status in values:
            return status
    if all(status in {"passed", "evidence_complete"} for status in values):
        return "passed" if "passed" in values else "evidence_complete"
    return "not_started"


def _normalize_design_evidence_status(evaluation: dict[str, Any]) -> str:
    """Read design-input/package completeness; visual-review codes stay supplemental."""
    raw_status = str(evaluation.get("status") or "").strip().lower()
    blocking, _supplemental = _split_design_quality_issues(evaluation)
    if blocking:
        return "evidence_incomplete"
    raw_issues = [
        item
        for item in evaluation.get("issues") or []
        if isinstance(item, dict)
    ]
    visual_only_issues = bool(raw_issues) and all(
        _is_supplemental_design_issue(str(item.get("code") or ""))
        for item in raw_issues
    )
    if raw_status == "evidence_incomplete":
        return "evidence_complete" if visual_only_issues else "evidence_incomplete"
    if raw_status == "evidence_complete":
        return raw_status

    # Legacy `passed`/`failed`/`stale` could differ only because the submitting
    # design agent asserted a verdict, or because visual-review hashes aged.
    # With no blocking design/package issue, all three migrate to completeness.
    if raw_status in {"passed", "failed", "stale"}:
        return "evidence_complete"
    return "evidence_incomplete"


def _evidence_id(path: str, label: str) -> str:
    return hashlib.sha1(f"{path}|{label}".encode("utf-8")).hexdigest()[:12]


def _artifact(
    ctx: ReadContext,
    path: Path | None,
    *,
    label: str,
    kind: str,
    stage_id: str,
    role: str,
    summary: str = "",
    status: str | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    exists = bool(path and path.is_file())
    size = path.stat().st_size if exists and path else None
    relative = ctx.relative(path)
    return {
        "id": _evidence_id(relative, label),
        "label": label,
        "kind": kind,
        "role": role,
        "stage_id": stage_id,
        "path": relative,
        "exists": exists,
        "status": status or ("passed" if exists else "not_started"),
        "size_bytes": size,
        "size_label": _human_size(size),
        "summary": summary,
        "sha256": sha256 or "",
        "details": {},
        "preview_type": "",
        "preview_data_uri": "",
    }


def _bound_available_visuals(ctx: ReadContext, run: Path, summary: dict[str, Any]) -> set[Path]:
    """Keep each available indexed artifact independent of sibling failures."""
    source_run = summary.get("source_run")
    if source_run:
        try:
            if ctx.resolve(source_run) != run:
                return set()
        except (OSError, ValueError):
            return set()
    allowed: set[Path] = set()
    invalid: set[Path] = set()
    pending: list[Any] = [summary]
    while pending:
        value = pending.pop()
        if isinstance(value, list):
            pending.extend(item for item in value if isinstance(item, (dict, list)))
            continue
        if not isinstance(value, dict):
            continue
        for key, reference in value.items():
            if isinstance(reference, (dict, list)):
                pending.append(reference)
                continue
            if not isinstance(reference, str) or Path(reference).suffix.lower() not in {".png", ".mp4", ".glb", ".json"}:
                continue
            try:
                path = ctx.resolve(reference)
                if path is None or not path.is_relative_to(run / "visual") or not path.is_file():
                    continue
                declared = value.get(f"{key}_sha256")
                if declared is None and key in {"path", "file", "scene", "video", "output"}:
                    declared = value.get("sha256")
                if declared is not None and declared != _cached_sha256_file(path):
                    invalid.add(path)
                    continue
                allowed.add(path)
            except (OSError, ValueError):
                continue
    return allowed - invalid


def _visual_metadata(path: Path) -> dict[str, str]:
    """Map generated media to the stage and human meaning it actually supports."""
    name = path.name
    stem = path.stem
    if name == "stage1_design_package.png":
        return {
            "stage_id": "design-package",
            "visual_role": "cad-overview",
            "label": "整机 CAD 预览",
            "summary": "由本次设计包的 STEP solid 生成，用于检查整机外形和零件分组。",
        }
    if name == "stage1_design_package.glb":
        return {
            "stage_id": "design-package",
            "visual_role": "cad-scene",
            "label": "可交互整机 3D",
            "summary": "由本次运行绑定的 STEP 派生实体生成；可拖动旋转、缩放和平移。",
        }
    if name.startswith("stage3_mechanism_") and path.suffix.lower() == ".png":
        return {
            "stage_id": "mechanism-plan",
            "visual_role": "mechanism-map",
            "capsule_id": stem.removeprefix("stage3_mechanism_"),
            "label": "机构范围高亮图",
            "summary": "灰色为整机背景，彩色部分是当前选中的可运动机构。",
        }
    if name.startswith("stage3_mechanism_") and path.suffix.lower() == ".glb":
        return {
            "stage_id": "mechanism-plan",
            "visual_role": "mechanism-scene",
            "capsule_id": stem.removeprefix("stage3_mechanism_"),
            "label": "可交互机构范围",
            "summary": "拖动查看整机；半透明灰色是背景，彩色实体是当前机构。",
        }
    geometry_visuals = {
        "stage4_collision_overview.png": (
            "collision-overview",
            "碰撞预检总览",
            "由 STEP 派生碰撞体生成，检查固定件与运动件的空间关系。",
        ),
        "stage4_side_clearance.png": (
            "side-clearance",
            "侧向间隙特写",
            "放大展示导向壁与抽屉侧面的 0.10 mm 间隙。",
        ),
        "stage4_rail_contact.png": (
            "rail-contact",
            "导轨接触特写",
            "展示设计接触面贴合，但不存在正体积穿插。",
        ),
    }
    if name in geometry_visuals:
        role, label, summary = geometry_visuals[name]
        return {
            "stage_id": "geometry-preflight",
            "visual_role": role,
            "label": label,
            "summary": summary,
        }
    namespaced_geometry = {
        "stage4_side_clearance__": geometry_visuals[
            "stage4_side_clearance.png"
        ],
        "stage4_rail_contact__": geometry_visuals[
            "stage4_rail_contact.png"
        ],
    }
    for prefix, (role, label, summary) in namespaced_geometry.items():
        if name.startswith(prefix) and path.suffix.lower() == ".png":
            return {
                "stage_id": "geometry-preflight",
                "visual_role": role,
                "capsule_id": stem.removeprefix(prefix),
                "label": label,
                "summary": summary,
            }
    if name == "stage5_physics_evidence.glb" or (
        name.startswith("stage5_physics_evidence__")
        and path.suffix.lower() == ".glb"
    ):
        capsule_id = (
            stem.removeprefix("stage5_physics_evidence__")
            if name != "stage5_physics_evidence.glb"
            else ""
        )
        return {
            "stage_id": "branch-simulation",
            "visual_role": "physics-evidence-scene",
            "capsule_id": capsule_id,
            "label": "可交互物理证据场景",
            "summary": "逐 STEP 实体碰撞场景；由本次运行保存的碰撞几何生成，没有重跑仿真。",
        }
    if name == "stage5_physics_evidence.scene.json" or (
        name.startswith("stage5_physics_evidence__")
        and name.endswith(".scene.json")
    ):
        capsule_id = (
            name.removeprefix("stage5_physics_evidence__").removesuffix(
                ".scene.json"
            )
            if name != "stage5_physics_evidence.scene.json"
            else ""
        )
        return {
            "stage_id": "branch-simulation",
            "visual_role": "physics-evidence-scene-manifest",
            "capsule_id": capsule_id,
            "label": "物理证据场景来源清单",
            "summary": "记录每个 3D 节点对应的刚体、STEP 实体、碰撞表示和源文件哈希。",
        }
    if name.startswith("stage5_physics_evidence__") and path.suffix.lower() == ".json":
        identity = stem.removeprefix("stage5_physics_evidence__")
        capsule_id, separator, load_case_id = identity.rpartition("__")
        return {
            "stage_id": "branch-simulation",
            "visual_role": "physics-evidence-replay",
            "capsule_id": capsule_id if separator else "",
            "load_case_id": load_case_id if separator else identity,
            "label": "本工况可交互物理轨迹",
            "summary": "真实/简化轨迹的逐帧索引、碰撞表示、力学量和来源哈希；只读取已保存轨迹。",
        }
    if (
        name.startswith("stage5_physical_trace__")
        and path.suffix.lower() == ".mp4"
    ):
        identity = stem.removeprefix("stage5_physical_trace__")
        capsule_id, separator, load_case_id = identity.rpartition("__")
        return {
            "stage_id": "branch-simulation",
            "visual_role": "physical-diagnostic-video",
            "capsule_id": capsule_id if separator else "",
            "load_case_id": load_case_id if separator else identity,
            "label": "真实物理保存轨迹视频",
            "summary": (
                "只回放已经原子保存的真实物理状态；轨迹若未完整结束，"
                "视频会明确标注为诊断用途且不具备证书权威。"
            ),
        }
    if name == "drawer_physical_roundtrip.mp4" or (
        name.startswith("drawer_physical_roundtrip__")
        and path.suffix.lower() == ".mp4"
    ):
        capsule_id = (
            stem.removeprefix("drawer_physical_roundtrip__")
            if name != "drawer_physical_roundtrip.mp4"
            else ""
        )
        return {
            "stage_id": "branch-simulation",
            "visual_role": "physical-video",
            "capsule_id": capsule_id,
            "label": "真实物理分支回放",
            "summary": "由本次真实物理轨迹逐帧渲染；内部没有理想关节。",
        }
    if name == "drawer_reduced_roundtrip.mp4" or (
        name.startswith("drawer_reduced_roundtrip__")
        and path.suffix.lower() == ".mp4"
    ):
        capsule_id = (
            stem.removeprefix("drawer_reduced_roundtrip__")
            if name != "drawer_reduced_roundtrip.mp4"
            else ""
        )
        return {
            "stage_id": "branch-simulation",
            "visual_role": "reduced-video",
            "capsule_id": capsule_id,
            "label": "简化分支回放",
            "summary": "由本次简化分支轨迹逐帧渲染，用来与真实物理分支并排比较。",
        }
    if name.startswith("drawer_physical_roundtrip") and path.suffix.lower() == ".png":
        return {
            "stage_id": "branch-simulation",
            "visual_role": "physical-keyframe",
            "label": "真实物理分支关键帧",
            "summary": "真实物理分支的起点、最大行程或终点画面。",
        }
    if name.startswith("drawer_reduced_roundtrip") and path.suffix.lower() == ".png":
        return {
            "stage_id": "branch-simulation",
            "visual_role": "reduced-keyframe",
            "label": "简化分支关键帧",
            "summary": "简化分支的起点、最大行程或终点画面。",
        }
    if name.startswith("stage6_motion__") and path.suffix.lower() == ".mp4":
        parts = stem.split("__", 2)
        return {
            "stage_id": "certificate",
            "visual_role": "load-comparison-video",
            "capsule_id": parts[1] if len(parts) > 1 else "",
            "load_case_id": parts[2] if len(parts) > 2 else "",
            "label": "本工况双分支同步回放",
            "summary": "只用已保存轨迹渲染：左侧是真实物理，右侧是简化机构，没有重新运行仿真。",
        }
    if name.startswith("stage6_compare_") and path.suffix.lower() == ".png":
        identity = stem.removeprefix("stage6_compare_")
        capsule_id, separator, load_case_id = identity.rpartition("__")
        return {
            "stage_id": "certificate",
            "visual_role": "comparison-chart",
            "capsule_id": capsule_id if separator else "",
            "load_case_id": load_case_id if separator else identity,
            "label": "工况运动对比图",
            "summary": "把同一工况下真实物理与简化分支的轴向运动画在同一张图上。",
        }
    if name == "stage8_whole_machine.mp4" or (
        name.startswith("stage8_whole_machine__")
        and path.suffix.lower() == ".mp4"
    ):
        capsule_id = (
            stem.removeprefix("stage8_whole_machine__")
            if name != "stage8_whole_machine.mp4"
            else ""
        )
        return {
            "stage_id": "whole-machine",
            "visual_role": "whole-machine-video",
            "capsule_id": capsule_id,
            "label": "整机任务回放",
            "summary": "由本次整机任务结果生成的简化模型回放，不冒充高保真接触仿真。",
        }
    if name.startswith("stage8_whole_machine") and path.suffix.lower() == ".png":
        return {
            "stage_id": "whole-machine",
            "visual_role": "whole-machine-keyframe",
            "label": "整机任务关键帧",
            "summary": "整机任务的关键位置。",
        }
    return {
        "stage_id": "branch-simulation",
        "visual_role": "supporting-media",
        "label": name,
        "summary": "本次运行生成的辅助可视化证据。",
    }


def _append_physics_visual_artifact(
    *,
    ctx: "ReadContext",
    workspace: Path,
    path: Path,
    evidence: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    plans_by_capsule: dict[str, Any],
    plan: dict[str, Any],
    mode: str,
    capsule_id: str = "",
) -> None:
    """Attach one physics-evidence file, including replay/manifest payloads."""

    suffix = path.suffix.lower()
    meta = _visual_metadata(path)
    role = str(meta.get("visual_role") or "")
    if not role.startswith("physics-evidence-"):
        return
    if capsule_id and not meta.get("capsule_id"):
        meta["capsule_id"] = capsule_id
    key = (
        role,
        str(meta.get("capsule_id") or ""),
        str(meta.get("load_case_id") or ""),
    )
    already = {
        (
            str((item.get("details") or {}).get("visual_role") or ""),
            str((item.get("details") or {}).get("capsule_id") or ""),
            str((item.get("details") or {}).get("load_case_id") or ""),
        )
        for item in evidence
    }
    if key in already:
        return
    item = _artifact(
        ctx,
        path,
        label=meta["label"],
        kind=(
            "interactive-model"
            if suffix == ".glb"
            else "render-summary"
        ),
        stage_id=meta["stage_id"],
        role="evidence",
        summary=meta["summary"],
    )
    item["details"].update(
        {
            field: value
            for field, value in meta.items()
            if field not in {"stage_id", "label", "summary"}
        }
    )
    if item["details"].get("visual_role") == "physics-evidence-replay":
        replay = ctx.read_json(path, role="physics-evidence-replay")
        try:
            replay_capsule_id = str(
                replay.get("capsule_id") or item["details"].get("capsule_id") or ""
            )
            replay = upgrade_replay_for_presentation(
                replay,
                workspace_root=workspace,
                plan=plans_by_capsule.get(replay_capsule_id, plan),
            )
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            diagnostics.append(
                {
                    "level": "warning",
                    "role": "clock transmission arrow presentation overlay",
                    "path": ctx.relative(path),
                    "detail": str(exc),
                }
            )
        item["details"]["replay"] = replay
        item["details"]["capsule_id"] = str(
            replay.get("capsule_id") or item["details"].get("capsule_id") or ""
        )
        item["details"]["load_case_id"] = str(
            replay.get("load_case_id") or item["details"].get("load_case_id") or ""
        )
    if item["details"].get("visual_role") == "physics-evidence-scene-manifest":
        item["details"]["scene_manifest"] = ctx.read_json(
            path, role="physics-evidence-scene-manifest"
        )
    item["preview_type"] = "model" if suffix == ".glb" else ""
    if mode == "snapshot" and item["preview_type"]:
        item["preview_data_uri"] = _data_uri(path)
    evidence.append(item)


def _attach_diagnostic_physics_evidence(
    *,
    ctx: "ReadContext",
    workspace: Path,
    run_identity: Any,
    evidence: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    plans_by_capsule: dict[str, Any],
    plan: dict[str, Any],
    mode: str,
) -> None:
    """Surface interactive physics 3D written outside an immutable rNN."""

    if run_identity is None:
        return
    diagnostic_root = (
        workspace
        / ".runtime_runs"
        / "diagnostics"
        / f"{run_identity.run_id}-physics-evidence"
    )
    if not diagnostic_root.is_dir():
        return
    for visual_dir in sorted(
        path for path in diagnostic_root.iterdir() if path.is_dir()
    ):
        for path in sorted(visual_dir.iterdir()):
            if path.suffix.lower() not in {".glb", ".json"}:
                continue
            _append_physics_visual_artifact(
                ctx=ctx,
                workspace=workspace,
                path=path,
                evidence=evidence,
                diagnostics=diagnostics,
                plans_by_capsule=plans_by_capsule,
                plan=plan,
                mode=mode,
                capsule_id=visual_dir.name,
            )


def _stage(
    stage_id: str,
    order: int,
    title: str,
    short_title: str,
    goal: str,
    status: str,
    *,
    why: str,
    prerequisite: str,
    process: list[str],
    inputs: list[str],
    outputs: list[str],
    evidence_ids: list[str],
    metrics: list[dict[str, str]] | None = None,
    issues: list[str] | None = None,
    caveats: list[str] | None = None,
    kind: str | None = None,
    scope: dict[str, Any] | None = None,
    required: bool = True,
    applicability: str = "applicable",
) -> dict[str, Any]:
    return {
        "id": stage_id,
        "kind": kind or stage_id,
        "order": order,
        "title": title,
        "short_title": short_title,
        "goal": goal,
        "status": status,
        "status_label": STATUS_LABELS[status],
        "why": why,
        "prerequisite": prerequisite,
        "process": process,
        "inputs": inputs,
        "outputs": outputs,
        "evidence_ids": evidence_ids,
        "metrics": metrics or [],
        "issues": issues or [],
        "caveats": caveats or [],
        "scope": scope or {},
        "required": required,
        "applicability": applicability,
    }


def _materialize_published_stages(
    *,
    publication: dict[str, Any],
    base_stages: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    ctx: ReadContext,
) -> list[dict[str, Any]]:
    """Expand publication stage instances without collapsing equal kinds."""

    template_by_kind = {
        str(stage.get("kind") or ""): stage for stage in base_stages
    }
    evidence_by_path = {
        str(item.get("path") or ""): item
        for item in evidence
        if str(item.get("path") or "")
    }
    materialized: list[dict[str, Any]] = []
    for index, declared in enumerate(publication.get("stages") or [], start=1):
        if not isinstance(declared, dict):
            continue
        stage_id = str(declared.get("id") or "")
        kind = str(declared.get("kind") or stage_id)
        label = str(declared.get("label") or stage_id)
        template = template_by_kind.get(kind)
        if template is not None:
            stage = copy.deepcopy(template)
        else:
            stage = _stage(
                stage_id,
                index,
                label,
                label,
                "展示本次运行正式发布的阶段及其哈希绑定证据。",
                "not_started",
                why="该阶段由 harness 发布清单声明。",
                prerequisite="以上游发布阶段为准。",
                process=["读取正式发布清单", "核对证据哈希"],
                inputs=["正式发布清单"],
                outputs=["绑定证据"],
                evidence_ids=[],
                kind=kind,
            )

        published_evidence_ids: list[str] = []
        published_refs: list[dict[str, str]] = []
        for reference in declared.get("evidence") or []:
            if not isinstance(reference, dict):
                continue
            path_value = str(reference.get("path") or "")
            digest = str(reference.get("sha256") or "")
            if not path_value:
                continue
            artifact = evidence_by_path.get(path_value)
            if artifact is None:
                path = (ctx.workspace_root / path_value).resolve()
                artifact = _artifact(
                    ctx,
                    path,
                    label=path.name,
                    kind="published-evidence",
                    stage_id=stage_id,
                    role="published-output",
                    summary="由 harness 发布清单绑定的阶段证据。",
                    status="passed" if path.is_file() else "evidence_incomplete",
                    sha256=digest,
                )
                evidence.append(artifact)
                evidence_by_path[path_value] = artifact
            published_evidence_ids.append(str(artifact["id"]))
            published_refs.append({"path": path_value, "sha256": digest})

        published_status_raw = str(
            declared.get("status") or "EVIDENCE_COMPLETE"
        )
        instance_metrics = copy.deepcopy(stage.get("metrics") or [])
        if published_refs:
            primary_reference = published_refs[0]
            if kind == "task-planning":
                planning_suffix = TASK_PLANNING_RELATIVE.as_posix()
                primary_reference = next(
                    (
                        reference
                        for reference in published_refs
                        if (
                            reference["path"].replace("\\", "/")
                            == planning_suffix
                            or reference["path"].replace("\\", "/").endswith(
                                f"/{planning_suffix}"
                            )
                        )
                    ),
                    {},
                )
            primary_value: dict[str, Any] = {}
            if primary_reference:
                primary_path = (
                    ctx.workspace_root / primary_reference["path"]
                ).resolve()
                primary_value = ctx.read_json(
                    primary_path,
                    role=f"published {kind} stage evidence",
                )
            if kind == "task-planning" and primary_value:
                instance_metrics = [
                    {
                        "label": "任务理解",
                        "value": "已记录",
                    },
                    {
                        "label": "关节计划",
                        "value": str(len(primary_value.get("joint_plan") or [])),
                    },
                    {
                        "label": "测试计划",
                        "value": str(len(primary_value.get("test_plan") or [])),
                    },
                    {
                        "label": "新增测试任务",
                        "value": str(
                            len(primary_value.get("authored_test_tasks") or [])
                        ),
                    },
                ]
            elif kind == "mechanism-plan" and primary_value:
                instance_metrics = [
                    {
                        "label": "运动方式",
                        "value": str(primary_value.get("family") or "—"),
                    },
                    {
                        "label": "固定组",
                        "value": str(
                            len(primary_value.get("fixed_group_ids") or [])
                        ),
                    },
                    {
                        "label": "运动组",
                        "value": str(
                            len(primary_value.get("moving_group_ids") or [])
                        ),
                    },
                ]
            elif kind == "certificate" and primary_value:
                instance_metrics = [
                    {
                        "label": "证书",
                        "value": str(
                            primary_value.get("overall_verdict")
                            or primary_value.get("verdict")
                            or "NOT_CERTIFIED"
                        ),
                    },
                    {
                        "label": "允许简化替换",
                        "value": (
                            "是"
                            if primary_value.get("replacement_eligible")
                            else "否"
                        ),
                    },
                ]
            elif kind == "replacement" and primary_value:
                instance_metrics = [
                    {
                        "label": "简化机构",
                        "value": str(
                            primary_value.get("status") or "INACTIVE"
                        ),
                    },
                    {
                        "label": "运动方式",
                        "value": str(
                            (
                                primary_value.get("replacement_model")
                                or {}
                            ).get("family")
                            or "—"
                        ),
                    },
                ]
            elif kind == "whole-machine" and primary_value:
                task_rows = [
                    item
                    for item in (primary_value.get("stages") or [])
                    if isinstance(item, dict)
                ]
                instance_metrics = [
                    {
                        "label": "任务状态",
                        "value": str(
                            primary_value.get("status") or "NOT_CERTIFIED"
                        ),
                    },
                    {
                        "label": "任务步骤",
                        "value": str(len(task_rows)),
                    },
                ]
        stage.update(
            {
                "id": stage_id,
                "kind": kind,
                "order": index,
                "title": label,
                "short_title": label,
                "status": _status_from_value(published_status_raw),
                "status_label": STATUS_LABELS[
                    _status_from_value(published_status_raw)
                ],
                "scope": (
                    declared.get("scope")
                    if isinstance(declared.get("scope"), dict)
                    else {}
                ),
                "required": bool(declared.get("required", True)),
                "depends_on": [
                    str(item)
                    for item in (declared.get("depends_on") or [])
                    if str(item)
                ],
                "applicability": "declared_by_dashboard_publication",
                "evidence_ids": list(dict.fromkeys(published_evidence_ids)),
                "publication_status_raw": published_status_raw,
                "publication_evidence": published_refs,
                "metrics": instance_metrics,
            }
        )
        materialized.append(stage)
    return materialized


def _physical_execution_lanes(
    manifest: dict[str, Any],
    queue_state: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Join validator-owned queue jobs to their physical load-case lanes."""

    jobs_state = queue_state.get("jobs")
    jobs_state = jobs_state if isinstance(jobs_state, dict) else {}
    queue_status = str(queue_state.get("status") or "").strip().upper()
    queue_finished_with_failure = queue_status in {
        "BUDGET_EXHAUSTED",
        "FAILED",
        "FAILED_INTERNAL",
        "INTERRUPTED",
        "RESOURCE_ABORTED",
        "TIMED_OUT",
    }
    lanes: dict[tuple[str, str], dict[str, Any]] = {}
    for job in manifest.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        job_id = str(job.get("job_id") or "")
        if not job_id:
            continue
        binding: tuple[str, str] | None = None
        for output in job.get("expected_outputs") or []:
            if not isinstance(output, str):
                continue
            normalized = output.replace("\\", "/")
            match = re.search(
                r"(?:^|/)traces/([^/]+)/([^/]+)/physical[.]trace[.]json$",
                normalized,
            )
            if match is not None:
                binding = (match.group(1), match.group(2))
                break
        if binding is None:
            continue
        record = jobs_state.get(job_id)
        record = record if isinstance(record, dict) else {}
        raw_status = str(record.get("status") or "PENDING").strip().upper()
        lane_phase = str(record.get("lane_phase") or "").strip().upper()
        if raw_status == "SUCCEEDED":
            state = "PASSED"
            semantic = "passed"
        elif raw_status == "RUNNING":
            state = "RUNNING"
            semantic = "running"
        elif raw_status in {
            "FAILED",
            "TIMED_OUT",
            "RESOURCE_ABORTED",
            "BUDGET_EXHAUSTED",
        }:
            state = "FAILED"
            semantic = "failed"
        elif raw_status in {"CANCELLED", "INTERRUPTED"}:
            state = "CANCELLED"
            semantic = "blocked"
        elif raw_status == "RESOURCE_BLOCKED" or lane_phase in {
            "WAITING_FOR_MEMORY",
            "WAITING_FOR_RESOURCE",
            "WAITING_FOR_RESOURCES",
        }:
            state = "WAITING_FOR_RESOURCE"
            semantic = "running"
        elif queue_finished_with_failure:
            state = "CANCELLED"
            semantic = "blocked"
        else:
            state = "QUEUED"
            semantic = "not_started"
        lanes[binding] = {
            "job_id": job_id,
            "state": state,
            "status": semantic,
            "raw_status": raw_status,
            "lane_phase": lane_phase,
            "reason": str(record.get("reason") or ""),
            "attempts": int(record.get("attempts") or 0),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "duration_seconds": record.get("duration_seconds"),
        }
    return lanes


def _branch_queue_state_path(run_dir: Path, branch: str) -> Path:
    """Return the current queue state, including attempt-scoped reruns."""

    direct = run_dir / "runtime_queue" / branch / "queue_state.json"
    if direct.is_file():
        return direct

    attempt_states: list[tuple[int, Path]] = []
    for candidate in (run_dir / "runtime_queue").glob(
        f"attempt-*/{branch}/queue_state.json"
    ):
        match = re.fullmatch(r"attempt-(\d+)", candidate.parent.parent.name)
        if match is not None and candidate.is_file():
            attempt_states.append((int(match.group(1)), candidate))
    if attempt_states:
        return max(attempt_states, key=lambda item: item[0])[1]
    return direct


def _task_planning_warnings(
    planning: dict[str, Any],
    plans: list[dict[str, Any]],
    load_cases: list[dict[str, Any]],
) -> list[str]:
    """Compare optional planning targets with later evidence as warnings.

    Planning remains non-authoritative.  A comparison is made only when the
    AI supplied the corresponding stable identity (`capsule_id` or
    `load_case_id`), so free-form notes cannot accidentally become gates.
    """

    warnings: list[str] = []
    actual_plans = {
        str(item.get("capsule_id") or ""): item
        for item in plans
        if isinstance(item, dict) and item.get("capsule_id")
    }
    for item in planning.get("joint_plan") or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        target = target if isinstance(target, dict) else {}
        capsule_id = str(target.get("capsule_id") or "")
        if not capsule_id:
            continue
        actual = actual_plans.get(capsule_id)
        if actual is None:
            warnings.append(
                f"关节计划中的 {capsule_id} 未出现在 Harness 机构识别结果中。"
            )
            continue
        planned_family = str(target.get("family") or "")
        actual_family = str(actual.get("family") or "")
        if planned_family and actual_family and planned_family != actual_family:
            warnings.append(
                f"{capsule_id} 的计划关节类型 {planned_family} 与识别结果 "
                f"{actual_family} 不一致。"
            )

    actual_loads = {
        (
            str(item.get("capsule_id") or ""),
            str(item.get("id") or ""),
        ): item
        for item in load_cases
        if isinstance(item, dict) and item.get("id")
    }
    for item in planning.get("test_plan") or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        target = target if isinstance(target, dict) else {}
        capsule_id = str(target.get("capsule_id") or "")
        load_case_id = str(target.get("load_case_id") or "")
        if not load_case_id:
            continue
        actual = actual_loads.get((capsule_id, load_case_id))
        if actual is None and not capsule_id:
            candidates = [
                value
                for (_candidate_capsule, candidate_load), value in actual_loads.items()
                if candidate_load == load_case_id
            ]
            actual = candidates[0] if len(candidates) == 1 else None
        if actual is None:
            warnings.append(
                f"测试计划中的 {capsule_id + '/' if capsule_id else ''}"
                f"{load_case_id} 未出现在 Harness 工况中。"
            )
            continue
        planned_thresholds = target.get("thresholds")
        actual_thresholds = actual.get("thresholds")
        if (
            isinstance(planned_thresholds, dict)
            and planned_thresholds
            and isinstance(actual_thresholds, dict)
            and actual_thresholds
            and json.dumps(
                planned_thresholds,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            != json.dumps(
                actual_thresholds,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
        ):
            warnings.append(
                f"{load_case_id} 的规划阈值与 Harness 实际比较阈值不一致。"
            )
    return warnings


def _task_execution_coverage(
    planning: dict[str, Any],
    load_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose planning-to-execution bindings without creating new authority."""

    actual_loads = {
        (
            str(item.get("capsule_id") or ""),
            str(item.get("id") or ""),
        ): item
        for item in load_cases
        if isinstance(item, dict) and item.get("id")
    }
    coverage: list[dict[str, Any]] = []
    for item in planning.get("test_plan") or []:
        if not isinstance(item, dict):
            continue
        target = item.get("target")
        target = target if isinstance(target, dict) else {}
        capsule_id = str(target.get("capsule_id") or "")
        load_case_id = str(target.get("load_case_id") or "")
        actual = actual_loads.get((capsule_id, load_case_id))
        required_branches = [
            str(branch)
            for branch in (item.get("required_branches") or [])
            if str(branch)
        ]
        branch_status: dict[str, str] = {}
        if actual is not None:
            for branch in required_branches:
                if branch == "physical":
                    branch_record = actual.get("physical")
                elif branch == actual.get("ideal_kind"):
                    branch_record = actual.get("reduced")
                else:
                    branch_record = None
                branch_status[branch] = str(
                    branch_record.get("status")
                    if isinstance(branch_record, dict)
                    else "not_started"
                )
        if actual is None:
            status = "not_started"
        elif any(
            value in {"not_started", "evidence_incomplete", "blocked"}
            for value in branch_status.values()
        ):
            status = "evidence_incomplete"
        else:
            status = str(actual.get("status") or "evidence_incomplete")
        coverage.append(
            {
                "test_id": str(item.get("test_id") or "test"),
                "goal": str(item.get("goal") or ""),
                "capsule_id": capsule_id,
                "load_case_id": load_case_id,
                "metric_id": str(target.get("metric_id") or ""),
                "required_branches": required_branches,
                "branch_status": branch_status,
                "status": status,
                "comparison_status": (
                    str(actual.get("comparison_status") or "not_started")
                    if actual is not None
                    else "not_started"
                ),
            }
        )
    return coverage


def _capsule_evidence_gaps(
    run_dir: Path,
    plans: list[dict[str, Any]],
) -> list[str]:
    """Report missing per-capsule evidence; multiple capsules are valid."""

    capsule_plans = [
        plan
        for plan in plans
        if isinstance(plan, dict) and str(plan.get("capsule_id") or "")
    ]
    # This helper protects dynamic *multi-capsule aggregation*.  A single
    # capsule's missing artifacts are already represented by its normal stage
    # statuses; reporting them again here mislabels cross-run continuations as
    # a multi-scope aggregation problem.
    if len(capsule_plans) <= 1:
        return []

    reasons: list[str] = []
    for plan in capsule_plans:
        capsule_id = str(plan.get("capsule_id") or "")
        trace_root = run_dir / "traces" / capsule_id
        if not trace_root.is_dir() or not any(
            trace_root.glob("*/*.trace.json")
        ):
            reasons.append(f"CAPSULE_TRACE_EVIDENCE_MISSING:{capsule_id}")
        if not (run_dir / "results" / capsule_id / "certificate.json").is_file():
            reasons.append(
                f"CAPSULE_CERTIFICATE_EVIDENCE_MISSING:{capsule_id}"
            )
    return reasons


def _whole_machine_registration_gaps(
    registrations: list[dict[str, Any]],
) -> list[str]:
    """Allow multiple results while rejecting ambiguous capsule ownership."""

    if len(registrations) <= 1:
        return []
    reasons: list[str] = []
    seen_capsules: set[str] = set()
    for index, item in enumerate(registrations):
        capsule_id = str(item.get("capsule_id") or "")
        if not capsule_id:
            reasons.append(f"WHOLE_MACHINE_RESULT_CAPSULE_MISSING:{index}")
        elif capsule_id in seen_capsules:
            reasons.append(
                f"WHOLE_MACHINE_RESULT_CAPSULE_DUPLICATE:{capsule_id}"
            )
        else:
            seen_capsules.add(capsule_id)
    return reasons


def _load_case(
    ctx: ReadContext,
    run_dir: Path,
    capsule_id: str,
    load_path: Path,
    evidence: list[dict[str, Any]],
    *,
    execution_lane: dict[str, Any] | None = None,
    physical_trace_record: dict[str, Any] | None = None,
    ideal_reference_trace_record: dict[str, Any] | None = None,
    ideal_reference_status: Any = None,
) -> dict[str, Any]:
    load = ctx.read_json(load_path, role="load case")
    load_id = str(load.get("load_case_id") or load_path.stem)
    load_label = LOAD_LABELS.get(load_id, load_id)
    is_clock = (
        "clock" in str(load.get("program") or "").lower()
        or capsule_id == "clock_gear_train"
    )

    def clock_primary_angle_rad(metrics: dict[str, Any]) -> float:
        if load_id == "task_manual_time_setting":
            return _finite(metrics.get("setting_input_span_rad")) or 0.0
        if load_id == "task_alarm_cycle":
            return _finite(metrics.get("alarm_swing_rad")) or 0.0
        return (
            _finite(metrics.get("output_span_rad"))
            or _finite(metrics.get("max_abs_angle_rad"))
            or _finite(metrics.get("max_output_angle_rad"))
            or 0.0
        )
    if load_id == "held_out_mid" and load.get("program") == "bidirectional_rack_pinion":
        load_label = "中等负载"
    base = run_dir / "traces" / capsule_id / load_id
    physical_path = base / "physical.trace.json"
    ideal_dynamics_path = base / "ideal_dynamics.trace.json"
    ideal_reference_path = base / "ideal_reference.trace.json"
    legacy_reduced_path = base / "reduced.trace.json"
    if ideal_dynamics_path.is_file():
        ideal_path = ideal_dynamics_path
        ideal_kind = "ideal_dynamics"
        ideal_label = "理想动力学（按需）"
        ideal_summary_text = "明确要求动态对照时才运行的理想约束分支"
    elif ideal_reference_path.is_file():
        ideal_path = ideal_reference_path
        ideal_kind = "ideal_reference"
        ideal_label = "简化动力学分支（方程计算）"
        ideal_summary_text = "由机构方程直接计算，不运行碰撞或动力学求解"
    else:
        ideal_path = legacy_reduced_path
        ideal_kind = "legacy_reduced"
        ideal_label = "历史简化动力学"
        ideal_summary_text = "旧版 harness 保存的简化动力学分支"
    comparison_path = (
        run_dir / "results" / capsule_id / "comparison" / f"{load_id}.json"
    )
    physical_record = (
        physical_trace_record if isinstance(physical_trace_record, dict) else {}
    )
    physical_record_status = physical_record.get("status")
    if not physical_record_status and physical_record.get("passed") is not None:
        physical_record_status = (
            "PASSED" if physical_record.get("passed") is True else "FAILED"
        )
    physical_record_geometry = physical_record.get("geometry_legality") or {}
    physical_record_sampling = physical_record.get("sampling") or {}
    if physical_record and physical_record_status:
        physical = {
            "status": physical_record_status,
            "sample_count": (
                physical_record_geometry.get("sample_count")
                or physical_record_sampling.get("selected_sample_count")
            ),
        }
        physical_status_source = "physical_preflight"
    else:
        physical = ctx.read_json(
            physical_path, role=f"{load_id} physical trace"
        )
        physical_status_source = "trace"

    ideal_record = (
        ideal_reference_trace_record
        if isinstance(ideal_reference_trace_record, dict)
        else {}
    )
    if ideal_kind == "ideal_reference" and ideal_record and ideal_reference_status:
        reduced = {
            "status": ideal_reference_status,
            "sample_count": ideal_record.get("sample_count"),
        }
        ideal_status_source = "ideal_reference_summary"
    else:
        reduced = ctx.read_json(ideal_path, role=f"{load_id} ideal trace")
        ideal_status_source = "trace"
    comparison = ctx.read_json(comparison_path, role=f"{load_id} comparison")

    artifacts = [
        _artifact(
            ctx,
            load_path,
            label=f"{load_label} · 工况定义",
            kind="load",
            stage_id="branch-simulation",
            role="input",
            summary="验证器生成的载荷和阶段计划",
        ),
        _artifact(
            ctx,
            physical_path,
            label=f"{load_label} · 真实轨迹",
            kind="trace-physical",
            stage_id="branch-simulation",
            role="output",
            summary="无内部理想关节的接触分支轨迹",
            status=_status_from_value(physical.get("status")),
            sha256=str(physical_record.get("physical_trace_sha256") or ""),
        ),
        _artifact(
            ctx,
            ideal_path,
            label=f"{load_label} · {ideal_label}",
            kind=f"trace-{ideal_kind.replace('_', '-')}",
            stage_id="branch-simulation",
            role="output",
            summary=ideal_summary_text,
            status=_status_from_value(reduced.get("status")),
            sha256=str(ideal_record.get("trace_sha256") or ""),
        ),
        _artifact(
            ctx,
            comparison_path,
            label=f"{load_label} · 对照结论",
            kind="comparison",
            stage_id="certificate",
            role="output",
            summary="真实与简化分支的运动学行为门控",
            status=_status_from_value(comparison.get("verdict")),
        ),
    ]
    evidence.extend(artifacts)

    physical_metrics = comparison.get("physical_metrics") or {}
    reduced_metrics = comparison.get("reduced_metrics") or {}
    comparison_mode = str(comparison.get("comparison_mode") or "")
    ideal_reference_metrics = comparison.get("ideal_reference_metrics") or {}
    contact_gate_recorded = "contact_observed" in physical_metrics
    comparison_status = _status_from_value(comparison.get("verdict"))
    comparison_checks = _comparison_checks(
        load, physical_metrics, reduced_metrics, comparison.get("thresholds") or {}
    )
    if comparison_mode == "physical_vs_ideal_reference":
        for check in comparison_checks:
            check["reduced"] = None
            check["reduced_pass"] = None
            check["reduced_margin"] = None
        threshold = comparison.get("ideal_reference_threshold") or {}
        error = _finite(ideal_reference_metrics.get("max_absolute_error"))
        limit = _finite(threshold.get("max_absolute_error"))
        unit = str(threshold.get("unit") or "")
        scale = 1000.0 if unit == "m" else (180.0 / math.pi if unit == "rad" else 1.0)
        display_unit = "mm" if unit == "m" else ("°" if unit == "rad" else unit)
        comparison_checks.append(
            {
                "label": "相对理想机构的最大输出偏差",
                "rule": "≤",
                "limit": _round(None if limit is None else limit * scale, 4),
                "unit": display_unit,
                "physical": _round(None if error is None else error * scale, 4),
                "reduced": 0.0,
                "physical_pass": (
                    None if error is None or limit is None else error <= limit
                ),
                "reduced_pass": True,
                "physical_margin": (
                    None if error is None or limit is None else _round((limit - error) * scale, 4)
                ),
                "reduced_margin": (
                    None if limit is None else _round(limit * scale, 4)
                ),
                "note": "理想参考由机构方程直接计算，不提供接触或动态响应证据。",
            }
        )

    ideal_samples = [
        item for item in reduced.get("samples") or [] if isinstance(item, dict)
    ]
    ideal_progress_values = [
        value
        for item in ideal_samples
        for value in (
            _finite(item.get("generalized_progress_m")),
            _finite(item.get("rack_progress_m")),
            _finite(item.get("platform_progress_m")),
            (
                _finite(item.get("output_coordinate"))
                if item.get("output_coordinate_unit") == "m"
                else None
            ),
        )
        if value is not None
    ]
    ideal_angle_values = [
        value
        for item in ideal_samples
        for value in (
            _finite(item.get("generalized_angle_rad")),
            _finite(item.get("clock_output_angle_rad")),
            (
                _finite(item.get("output_coordinate"))
                if item.get("output_coordinate_unit") == "rad"
                else None
            ),
        )
        if value is not None
    ]
    ideal_max_progress_m = (
        max(ideal_progress_values, default=0.0)
        if ideal_kind == "ideal_reference"
        else (_finite(reduced_metrics.get("max_progress_m")) or 0.0)
    )
    ideal_max_angle_rad = (
        max((abs(value) for value in ideal_angle_values), default=0.0)
        if ideal_kind == "ideal_reference"
        else (
            clock_primary_angle_rad(reduced_metrics)
            if is_clock
            else (
                _finite(reduced_metrics.get("max_abs_angle_rad"))
                or _finite(reduced_metrics.get("max_output_angle_rad"))
                or 0.0
            )
        )
    )
    branch_statuses = [
        _status_from_value(physical.get("status")),
        _status_from_value(reduced.get("status")),
    ]
    status = comparison_status if comparison else _combine_status(branch_statuses)
    physical_summary = {
        "type": "trace", "branch": "真实物理", "status": branch_statuses[0],
        "status_source": physical_status_source,
        "coordinate_kind": "angular" if is_clock or str(load.get("family") or "").lower() == "revolute" else "linear",
        "sample_count": physical.get("sample_count"),
        "contact_observed": physical_metrics.get("contact_observed"),
        "max_progress_mm": _round((_finite(physical_metrics.get("max_progress_m")) or 0.0) * 1000, 4),
        "max_angle_deg": _round(math.degrees(clock_primary_angle_rad(physical_metrics) if is_clock else (_finite(physical_metrics.get("max_abs_angle_rad")) or _finite(physical_metrics.get("max_output_angle_rad")) or 0.0)), 4),
        "max_pivot_drift_mm": _round((_finite(physical_metrics.get("max_pivot_drift_m")) or 0.0) * 1000, 4),
        "max_off_axis_rotation_deg": _round(math.degrees(_finite(physical_metrics.get("max_off_axis_rotation_rad")) or 0.0), 4),
        "max_lateral_shift_mm": _round((_finite(physical_metrics.get("max_lateral_shift_m")) or 0.0) * 1000, 4),
        "max_rotation_deg": _round(math.degrees(_finite(physical_metrics.get("max_rotation_rad")) or 0.0), 4),
    }
    reduced_summary = {
        "type": "trace", "branch": ideal_label, "status": branch_statuses[1],
        "status_source": ideal_status_source,
        "coordinate_kind": "angular" if is_clock or str(load.get("family") or "").lower() == "revolute" else "linear",
        "sample_count": reduced.get("sample_count"), "contact_observed": False,
        "max_progress_mm": _round(ideal_max_progress_m * 1000, 4),
        "max_angle_deg": _round(math.degrees(ideal_max_angle_rad), 4),
        "max_pivot_drift_mm": _round((_finite(reduced_metrics.get("max_pivot_drift_m")) or 0.0) * 1000, 4),
        "max_off_axis_rotation_deg": _round(math.degrees(_finite(reduced_metrics.get("max_off_axis_rotation_rad")) or 0.0), 4),
        "max_lateral_shift_mm": _round((_finite(reduced_metrics.get("max_lateral_shift_m")) or 0.0) * 1000, 4),
        "max_rotation_deg": _round(math.degrees(_finite(reduced_metrics.get("max_rotation_rad")) or 0.0), 4),
    }
    artifacts[1]["details"] = physical_summary
    artifacts[2]["details"] = reduced_summary
    artifacts[3]["details"] = {
        "type": "comparison", "capsule_id": capsule_id,
        "load_case_id": load_id, "label": load_label,
        "status": comparison_status,
        "verdict": str(comparison.get("verdict") or "NOT_AVAILABLE"),
        "checks": comparison_checks,
        "failure_reasons": [str(item) for item in comparison.get("failure_reasons", [])],
        "progress_delta_mm": _round(
            (
                (_finite(ideal_reference_metrics.get("max_absolute_error")) or 0.0)
                if comparison_mode == "physical_vs_ideal_reference"
                and str((comparison.get("ideal_reference_threshold") or {}).get("unit") or "") == "m"
                else abs(
                    (_finite(physical_metrics.get("max_progress_m")) or 0.0)
                    - ideal_max_progress_m
                )
            )
            * 1000,
            4,
        ),
        "policy": (
            (
                "真实分支负责接触、运动与约束证据；简化动力学分支（方程计算）只给出机构方程目标，不提供接触或动态响应证据。"
                if contact_gate_recorded
                else "真实分支负责运动与约束证据；本工况没有发布接触门控。简化动力学分支（方程计算）只给出机构方程目标。"
            )
            if comparison_mode == "physical_vs_ideal_reference"
            else (
                f"门控比较转动方向、转轴约束{'、真实接触' if contact_gate_recorded else ''}和任务角度边界；不要求两分支逐时刻角度相等。"
                if str(load.get("family") or "").lower() == "revolute"
                else f"门控比较运动方向、约束{'、接触' if contact_gate_recorded else ''}和任务边界；不要求两分支位移幅值相等，也不使用速度或时间匹配作为通过条件。"
            )
        ),
    }
    for artifact in artifacts[:3]:
        artifact.setdefault("details", {})["capsule_id"] = capsule_id
        artifact["details"]["load_case_id"] = load_id
    return {
        "id": load_id,
        "capsule_id": capsule_id,
        "label": load_label,
        "purpose": str(load.get("purpose") or "unknown"),
        "program": str(load.get("program") or "unknown"),
        "comparison_mode": comparison_mode or "physical_vs_legacy_reduced",
        "ideal_kind": ideal_kind,
        "ideal_label": ideal_label,
        "ideal_dynamic_response": ideal_kind in {"ideal_dynamics", "legacy_reduced"},
        "ideal_contact_authority": False,
        "mechanism_kind": "clock" if is_clock else "",
        "status": status,
        "branch_status": _combine_status(branch_statuses),
        "comparison_status": comparison_status,
        "status_label": STATUS_LABELS[status],
        "verdict": str(comparison.get("verdict") or "NOT_AVAILABLE"),
        "required": bool(load.get("required_for_certificate")),
        "moving_mass_kg": _round(load.get("moving_mass_kg"), 4),
        "coordinate_kind": "angular" if is_clock or str(load.get("family") or "").lower() == "revolute" else "linear",
        "max_torque_nm": _round(max([_finite(item.get("magnitude_si")) or 0.0 for item in load.get("phases", []) if isinstance(item, dict)] or [0.0]), 4),
        "max_force_n": _round(
            max(
                [
                    _finite(item.get("magnitude_si")) or 0.0
                    for item in load.get("phases", [])
                    if isinstance(item, dict)
                ]
                or [0.0]
            ),
            4,
        ),
        "physical": {
            "status": branch_statuses[0],
            "sample_count": physical.get("sample_count"),
            "contact_observed": physical_metrics.get("contact_observed"),
            "max_progress_mm": _round(
                (_finite(physical_metrics.get("max_progress_m")) or 0.0) * 1000,
                3,
            ),
            "max_angle_deg": _round(math.degrees(clock_primary_angle_rad(physical_metrics) if is_clock else (_finite(physical_metrics.get("max_abs_angle_rad")) or _finite(physical_metrics.get("max_output_angle_rad")) or 0.0)), 3),
            "max_pivot_drift_mm": _round((_finite(physical_metrics.get("max_pivot_drift_m")) or 0.0) * 1000, 4),
            "max_off_axis_rotation_deg": _round(math.degrees(_finite(physical_metrics.get("max_off_axis_rotation_rad")) or 0.0), 4),
            "lateral_mm": _round(
                (_finite(physical_metrics.get("max_lateral_shift_m")) or 0.0)
                * 1000,
                4,
            ),
            "rotation_deg": _round(
                math.degrees(_finite(physical_metrics.get("max_rotation_rad")) or 0.0),
                3,
            ),
            "max_total_displacement_mm": _round((_finite(physical_metrics.get("max_total_displacement_m")) or 0.0) * 1000, 4),
            "axis_purity": _round(physical_metrics.get("axis_purity"), 6),
            "phase_directional_travel_mm": {
                str(key): _round((_finite(value) or 0.0) * 1000, 4)
                for key, value in (physical_metrics.get("phase_directional_travel_m") or {}).items()
            },
        },
        "reduced": {
            "kind": ideal_kind,
            "label": ideal_label,
            "dynamic_response": ideal_kind in {"ideal_dynamics", "legacy_reduced"},
            "contact_authority": False,
            "status": branch_statuses[1],
            "sample_count": reduced.get("sample_count"),
            "max_progress_mm": _round(ideal_max_progress_m * 1000, 3),
            "max_angle_deg": _round(math.degrees(ideal_max_angle_rad), 3),
            "max_pivot_drift_mm": _round((_finite(reduced_metrics.get("max_pivot_drift_m")) or 0.0) * 1000, 4),
            "max_off_axis_rotation_deg": _round(math.degrees(_finite(reduced_metrics.get("max_off_axis_rotation_rad")) or 0.0), 4),
            "lateral_mm": _round(
                (_finite(reduced_metrics.get("max_lateral_shift_m")) or 0.0)
                * 1000,
                4,
            ),
            "rotation_deg": _round(
                math.degrees(_finite(reduced_metrics.get("max_rotation_rad")) or 0.0),
                3,
            ),
            "max_total_displacement_mm": _round((_finite(reduced_metrics.get("max_total_displacement_m")) or 0.0) * 1000, 4),
            "axis_purity": _round(reduced_metrics.get("axis_purity"), 6),
            "phase_directional_travel_mm": {
                str(key): _round((_finite(value) or 0.0) * 1000, 4)
                for key, value in (reduced_metrics.get("phase_directional_travel_m") or {}).items()
            },
        },
        "checks": comparison_checks,
        "progress_delta_mm": _round(
            (
                (_finite(ideal_reference_metrics.get("max_absolute_error")) or 0.0)
                if comparison_mode == "physical_vs_ideal_reference"
                and str((comparison.get("ideal_reference_threshold") or {}).get("unit") or "") == "m"
                else abs(
                    (_finite(physical_metrics.get("max_progress_m")) or 0.0)
                    - ideal_max_progress_m
                )
            )
            * 1000,
            4,
        ),
        "gate_policy": (
            (
                "真实分支先通过接触、约束和任务门槛，再检查它相对确定性理想机构的输出偏差。"
                if contact_gate_recorded
                else "真实分支先通过运动、约束和任务门槛，再检查它相对确定性理想机构的输出偏差；本工况不要求接触证据。"
            )
            if comparison_mode == "physical_vs_ideal_reference"
            else (
                f"行为包络门控：转动方向、转轴漂移、轴外转动{'、真实接触' if contact_gate_recorded else ''}和任务角度边界。"
                if str(load.get("family") or "").lower() == "revolute"
                else f"行为包络门控：方向、约束{'、接触' if contact_gate_recorded else ''}和任务边界；位移幅值差与逐时刻速度差不是本证书门控。"
            )
        ),
        "thresholds": comparison.get("thresholds") or {},
        "failure_reasons": [str(item) for item in comparison.get("failure_reasons", [])],
        "evidence_ids": [item["id"] for item in artifacts],
        "execution_lane": dict(execution_lane or {}),
    }


def normalize_run(
    run_dir: str | Path,
    *,
    workspace_root: str | Path,
    mode: str = "live",
) -> dict[str, Any]:
    workspace = Path(workspace_root).resolve()
    requested_run = Path(run_dir).resolve()
    try:
        requested_run.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("run directory must stay inside workspace_root") from exc
    requested_is_react = is_react_loop_run(workspace, requested_run)
    live_project = read_live_project(workspace_root=workspace, run_dir=requested_run)
    slim_result = {}
    if live_project["live_workflow"].get("workflow_kind") == "slim_harness":
        from ..slim_harness.publication import read_result
        slim_result = read_result(requested_run)

    diagnostics: list[dict[str, str]] = []
    ctx = ReadContext(workspace, diagnostics)
    evidence: list[dict[str, Any]] = []
    run_status_path = requested_run / "run_status.json"
    run_status = ctx.read_json(run_status_path, role="run status")
    evidence.append(
        _artifact(
            ctx,
            run_status_path,
            label="最终运行状态",
            kind="run-status",
            stage_id="whole-machine",
            role="output",
            summary="流水线当前权威状态",
            status=_status_from_value(run_status.get("status")),
        )
    )

    # A cross-run whole-machine continuation intentionally contains only its
    # terminal task evidence. Its immutable status names the upstream run that
    # owns the design, intake, physical, certificate, and replacement evidence.
    # Read those earlier stages in place while keeping the requested run's
    # status and registered whole-machine result authoritative.
    run = requested_run
    continuation = run_status.get("continuation")
    if isinstance(continuation, dict) and str(continuation.get("mode") or "") == (
        "cross_run_whole_machine_continuation_v1"
    ):
        upstream_ref = continuation.get("upstream_run")
        if upstream_ref:
            try:
                upstream_run = resolve_workspace_reference(
                    workspace,
                    upstream_ref,
                    require_within_workspace=True,
                    require_exists=True,
                )
                if not (upstream_run / "run_status.json").is_file():
                    raise FileNotFoundError(upstream_run / "run_status.json")
                run = upstream_run
            except (FileNotFoundError, ValueError) as exc:
                diagnostics.append(
                    {
                        "level": "missing",
                        "role": "continuation upstream run",
                        "path": str(upstream_ref),
                        "detail": str(exc),
                    }
                )

    task_planning_path = run / TASK_PLANNING_RELATIVE
    task_planning: dict[str, Any] = {}
    task_planning_error = ""
    task_planning_exists = bool(
        task_planning_path.is_file() or task_planning_path.is_symlink()
    )
    if task_planning_exists:
        if requested_is_react:
            from ..contracts.planning import (
                NativeStorageError,
                load_planning_contract,
            )

            try:
                task_planning, _planning_path = load_planning_contract(
                    workspace_root=workspace,
                    run_dir=run,
                )
            except NativeStorageError as exc:
                task_planning_error = f"REACT_PLANNING_INVALID: {exc}"
        else:
            from .legacy_planning import (
                LegacyPlanningError,
                load_legacy_task_planning,
            )

            try:
                task_planning, _planning_evidence = load_legacy_task_planning(
                    workspace_root=workspace,
                    run_dir=run,
                )
            except LegacyPlanningError as exc:
                task_planning_error = f"{exc.code}: {exc.message}"
        if task_planning_error:
            diagnostics.append(
                {
                    "level": "invalid",
                    "role": "task understanding and validation planning",
                    "path": ctx.relative(task_planning_path),
                    "detail": task_planning_error,
                }
            )
    task_planning_artifact = _artifact(
        ctx,
        task_planning_path,
        label="接收任务与测试规划",
        kind="task-planning",
        stage_id="task-planning",
        role="output",
        summary="冻结任务理解、关节计划、测试计划和按需制作的测试任务。",
        status=(
            "evidence_complete"
            if task_planning
            else "evidence_incomplete"
            if task_planning_exists
            else "not_started"
        ),
    )
    react_loop_state: dict[str, Any] = {}
    react_loop_state_error = ""
    react_loop_state_path: Path | None = None
    requested_identity = parse_canonical_run(workspace, requested_run)
    if requested_identity is not None:
        candidate = _react_loop_state_path(
            workspace, requested_identity.project_dir
        )
        if candidate is None:
            candidate = requested_identity.project_dir / "react_loop" / "state.json"
        if candidate.is_file() or candidate.is_symlink():
            react_loop_state_path = candidate
            try:
                candidate_value, _candidate_record = read_stable_json_object(
                    candidate,
                    label="dashboard ReAct loop state",
                    allowed_root=requested_identity.project_dir,
                )
                state_core = {
                    key: value
                    for key, value in candidate_value.items()
                    if key != "state_sha256"
                }
                conversation = candidate_value.get("conversation")
                expected_run = requested_identity.run_dir.relative_to(
                    workspace
                ).as_posix()
                if (
                    candidate_value.get("schema_version") != LOOP_STATE_SCHEMA
                    or candidate_value.get("state_sha256")
                    != canonical_sha256(state_core)
                    or candidate_value.get("project_id")
                    != requested_identity.project_id
                    or not isinstance(conversation, dict)
                    or conversation.get("memory_enabled") is not False
                    or not str(
                        conversation.get("conversation_id") or ""
                    ).strip()
                ):
                    raise ValueError(
                        "ReAct state or task-scoped conversation binding is invalid"
                    )
                # One project-scoped loop advances across revisions.
                # Selecting an older revision must keep its evidence readable,
                # not diagnose the valid loop for a newer rNN as corrupt.
                if candidate_value.get("current_run") == expected_run:
                    react_loop_state = candidate_value
                else:
                    historical = _react_project_revision_states(
                        workspace,
                        requested_identity.project_dir,
                    ).get(requested_run)
                    if historical is not None:
                        react_loop_state = historical
                        evidence_path = historical.get(
                            "_dashboard_evidence_path"
                        )
                        if isinstance(evidence_path, str) and evidence_path:
                            resolved_evidence = (
                                workspace / evidence_path
                            ).resolve()
                            resolved_evidence.relative_to(
                                requested_identity.project_dir.resolve()
                            )
                            react_loop_state_path = resolved_evidence
                        else:
                            react_loop_state_path = None
                    else:
                        react_loop_state_path = None
            except (OSError, ValueError, OperatorAuthorityError) as exc:
                react_loop_state_error = str(exc)
                diagnostics.append(
                    {
                        "level": "invalid",
                        "role": "stage-isolated ReAct loop state",
                        "path": ctx.relative(candidate),
                        "detail": react_loop_state_error,
                    }
                )
    react_loop_display_status = {
        "PLAN_REVIEW_REQUIRED": "awaiting_review",
        "RESOURCE_APPROVAL_REQUIRED": "awaiting_review",
        "PLUGIN_IMPLEMENTATION_REQUIRED": "awaiting_review",
        "STATIC_VISUAL_REVIEW_REQUIRED": "awaiting_review",
        "EXTERNAL_VALIDATION_REVIEW_REQUIRED": "awaiting_review",
        "VALIDATION_FAILURE_REVIEW_REQUIRED": "awaiting_review",
        "RESOURCE_RETRY_REQUIRED": "blocked",
        "REPEATED_ERROR_PAUSED": "blocked",
        "WAITING_FOR_EXTERNAL_APPROVAL": "awaiting_review",
        "COMPLETE": "evidence_complete",
        "TERMINATED": "failed",
        "REVISION_FAILED": "failed",
        "REVISION_SUPERSEDED": "stale",
    }.get(str(react_loop_state.get("status") or ""), "running")
    react_loop_artifact = (
        _artifact(
            ctx,
            react_loop_state_path,
            label=(
                "历史 Harness 结果"
                if react_loop_state.get("_dashboard_historical")
                else "逐环 AI 控制状态"
            ),
            kind="react-loop",
            stage_id="task-planning",
            role="state",
            summary=(
                "该 revision 的父端 Harness 结果。"
                if react_loop_state.get("_dashboard_historical")
                else "父端编排状态；没有机械 PASS、审批签名或证书权限。"
            ),
            status=(
                react_loop_display_status
                if react_loop_state
                else "evidence_incomplete"
                if react_loop_state_path is not None
                else "not_started"
            ),
        )
        if react_loop_state_path is not None
        else None
    )
    requested_is_canonical_terminal = bool(
        parse_canonical_run(workspace, requested_run) is not None
        and str(run_status.get("status") or "").strip().upper()
        in CANONICAL_TERMINAL_STATUSES
    )
    if task_planning_exists and not requested_is_canonical_terminal:
        evidence.append(task_planning_artifact)
    if react_loop_artifact is not None and not requested_is_canonical_terminal:
        evidence.append(react_loop_artifact)
    static_visual_review, static_visual_artifacts = (
        _static_visual_review_projection(
            ctx,
            react_loop_state,
            mode=mode,
        )
    )
    evidence.extend(static_visual_artifacts)

    plan_paths = sorted((run / "plans").glob("*.json")) if (run / "plans").is_dir() else []
    plans = [ctx.read_json(path, role="mechanism plan") for path in plan_paths]
    plan = plans[0] if plans else {}
    plans_by_capsule = {
        str(item.get("capsule_id") or ""): item
        for item in plans
        if isinstance(item, dict) and item.get("capsule_id")
    }
    plan_path = plan_paths[0] if plan_paths else None
    registered_mechanisms = [
        item
        for item in (run_status.get("mechanisms") or [])
        if isinstance(item, dict)
    ]
    mechanisms_by_capsule = {
        str(item.get("capsule_id") or ""): item
        for item in registered_mechanisms
        if item.get("capsule_id")
    }
    registered_whole_results = [
        item
        for item in (run_status.get("whole_machine_results") or [])
        if isinstance(item, dict)
    ]
    stage_aggregation_reasons = _capsule_evidence_gaps(run, plans)
    stage_aggregation_reasons.extend(
        _whole_machine_registration_gaps(registered_whole_results)
    )
    if stage_aggregation_reasons:
        diagnostics.append(
            {
                "level": "warning",
                "role": "dynamic stage aggregation",
                "path": ctx.relative(requested_run),
                "detail": (
                    "检测到多机构证据缺口或整机结果身份冲突；正式判定保持不变，"
                    "但证据健康不得显示为完整。"
                ),
            }
        )
    capsule_id = str(
        plan.get("capsule_id")
        or (registered_mechanisms[0] if registered_mechanisms else {}).get("capsule_id")
        or "mechanism"
    )
    package_dir = ctx.resolve(plan.get("package_dir"))
    intake_path = ctx.resolve(plan.get("intake_report"))
    package_manifest_path = package_dir / "package.json" if package_dir else None
    assembly_path = package_dir / "assembly.step" if package_dir else None
    contract_path = package_dir / "assembly_contract.json" if package_dir else None
    package_manifest = ctx.read_json(package_manifest_path, role="package manifest")
    intake = ctx.read_json(intake_path, role="intake report")

    design_quality_root = run / "design_quality"
    design_input_path = design_quality_root / "design_input.json"
    visual_review_path = design_quality_root / "visual_review.json"
    design_evaluation_path = design_quality_root / "evaluation.json"
    design_input = (
        ctx.read_json(design_input_path, role="original design prompt and engineering brief")
        if design_input_path.is_file()
        else {}
    )
    visual_review = (
        ctx.read_json(visual_review_path, role="hash-bound visual CAD review")
        if visual_review_path.is_file()
        else {}
    )
    design_evaluation = (
        ctx.read_json(design_evaluation_path, role="design-input/package completeness evaluation")
        if design_evaluation_path.is_file()
        else {}
    )
    design_quality_recorded = any(
        path.is_file()
        for path in (design_input_path, visual_review_path, design_evaluation_path)
    )
    design_quality_status = (
        _normalize_design_evidence_status(design_evaluation)
        if design_evaluation
        else "warning"
    )
    design_blocking_issues, design_supplemental_issues = (
        _split_design_quality_issues(design_evaluation)
    )
    design_reported_findings = (
        design_evaluation.get("reported_findings")
        or design_evaluation.get("open_findings")
        or {"P0": 0, "P1": 0, "P2": 0}
    )
    if not isinstance(design_reported_findings, dict):
        design_reported_findings = {"P0": 0, "P1": 0, "P2": 0}

    provenance_candidates = (
        run / "derived_adapters" / capsule_id / "provenance.json",
        run
        / "derived_adapters"
        / capsule_id
        / "family_collision"
        / "provenance.json",
    )
    provenance_path = next(
        (path for path in provenance_candidates if path.is_file()),
        provenance_candidates[0],
    )
    provenance = ctx.read_json(provenance_path, role="derived adapter provenance")
    compound_geometry_path: Path | None = None
    compound_geometry: dict[str, Any] = {}
    if not provenance and (plan.get("adapter_spec") or {}).get("family_spec"):
        for candidate in sorted((run / "traces" / capsule_id).glob("*/physical.trace.json")):
            payload = ctx.read_json(candidate, role="compound collision geometry trace")
            bodies = (payload.get("collision_geometry") or {}).get("bodies")
            if payload.get("status") == "COMPLETED" and isinstance(bodies, list) and bodies:
                compound_geometry_path = candidate
                compound_geometry = payload.get("collision_geometry") or {}
                break
    geometry_evidence = provenance or compound_geometry

    legacy_queue_state_candidates = (
        run / "runtime_queue" / "queue_state.json",
        run / "queue_run" / "queue_state.json",
    )
    legacy_queue_state_path = next(
        (path for path in legacy_queue_state_candidates if path.is_file()),
        None,
    )
    physical_queue_state_path = _branch_queue_state_path(run, "physical")
    reduced_queue_state_path = _branch_queue_state_path(run, "reduced")
    branch_queues = run_status.get("branch_queues") or {}
    physical_queue_state = (
        ctx.read_json(physical_queue_state_path, role="physical branch queue state")
        if physical_queue_state_path.is_file()
        else dict(branch_queues.get("physical") or {})
    )
    if not physical_queue_state and run_status.get("physical_queue_summary"):
        physical_queue_state = {
            "status": run_status.get("physical_queue_status"),
            "summary": run_status.get("physical_queue_summary") or {},
        }
    physical_queue_manifest_path = run / "physical_queue_manifest.json"
    physical_queue_manifest = (
        ctx.read_json(
            physical_queue_manifest_path,
            role="physical branch queue manifest",
        )
        if physical_queue_manifest_path.is_file()
        else {}
    )
    physical_execution_lanes = _physical_execution_lanes(
        physical_queue_manifest,
        physical_queue_state,
    )
    reduced_queue_state = (
        ctx.read_json(reduced_queue_state_path, role="reduced branch queue state")
        if reduced_queue_state_path.is_file()
        else dict(branch_queues.get("reduced") or {})
    )
    queue_state_path = (
        legacy_queue_state_path
        or (physical_queue_state_path if physical_queue_state_path.is_file() else None)
        or (reduced_queue_state_path if reduced_queue_state_path.is_file() else None)
        or run_status_path
    )
    queue_state = ctx.read_json(queue_state_path, role="branch queue state")

    physical_preflight_path = run / "results" / "physical_preflight.json"
    physical_preflight_file = (
        ctx.read_json(physical_preflight_path, role="physical geometry preflight")
        if physical_preflight_path.is_file()
        else {}
    )
    physical_preflight = physical_preflight_file or dict(
        run_status.get("physical_preflight") or {}
    )
    physical_trace_records = {
        (
            str(item.get("capsule_id") or ""),
            str(item.get("load_case_id") or ""),
        ): item
        for item in physical_preflight.get("traces") or []
        if isinstance(item, dict)
        and item.get("capsule_id")
        and item.get("load_case_id")
    }
    reuse_manifest_path = run / "results" / "reuse_manifest.json"
    reuse_manifest = (
        ctx.read_json(reuse_manifest_path, role="reuse manifest")
        if reuse_manifest_path.is_file()
        else {}
    )
    reuse_lane_jobs = [
        item
        for item in (reuse_manifest.get("lane_jobs") or [])
        if isinstance(item, dict)
    ]
    reuse_summary = {
        "recorded": bool(reuse_manifest),
        "mesh_cache": reuse_manifest.get("mesh_cache") or {},
        "lane_actions": {
            action: sum(
                1 for item in reuse_lane_jobs if item.get("action") == action
            )
            for action in ("EXECUTED", "REUSED", "INVALIDATED")
        },
        "lane_fingerprint_count": len(
            {
                str(item.get("fingerprint") or "")
                for item in reuse_lane_jobs
                if item.get("fingerprint")
            }
        ),
    }
    ideal_reference_summary = run_status.get("ideal_reference") or {}
    ideal_reference_summary = (
        ideal_reference_summary
        if isinstance(ideal_reference_summary, dict)
        else {}
    )
    ideal_reference_summary_path = (
        run / "results" / "ideal_reference_summary.json"
    )
    if (
        not ideal_reference_summary.get("traces")
        and ideal_reference_summary_path.is_file()
    ):
        ideal_reference_summary = ctx.read_json(
            ideal_reference_summary_path,
            role="ideal reference summary",
        )
    ideal_reference_trace_records = {
        (
            str(item.get("capsule_id") or ""),
            str(item.get("load_case_id") or ""),
        ): item
        for item in ideal_reference_summary.get("traces") or []
        if isinstance(item, dict)
        and item.get("capsule_id")
        and item.get("load_case_id")
    }
    legacy_physical_audit_path = (
        run / "migration" / "legacy_sampled_physical_audit_v1.json"
    )
    legacy_physical_audit = (
        ctx.read_json(legacy_physical_audit_path, role="legacy sampled physical audit")
        if legacy_physical_audit_path.is_file()
        else {}
    )
    current_penetration_summary_path = run / "results" / "penetration_evidence_v1.json"
    legacy_penetration_summary_path = run / "migration" / "penetration_evidence_v1.json"
    penetration_summary_path = (
        current_penetration_summary_path
        if current_penetration_summary_path.is_file()
        else legacy_penetration_summary_path
    )
    penetration_summary = (
        ctx.read_json(
            penetration_summary_path,
            role="physical penetration visual evidence",
        )
        if penetration_summary_path.is_file()
        else {}
    )

    package_artifacts = [
        _artifact(
            ctx,
            package_manifest_path,
            label="设计包清单",
            kind="package",
            stage_id="design-package",
            role="output",
            summary="身份、版本、文件清单和哈希绑定",
        ),
        _artifact(
            ctx,
            assembly_path,
            label="STEP 整机几何",
            kind="step",
            stage_id="design-package",
            role="output",
            summary="真实 B-rep 装配几何",
            sha256=(plan.get("artifact_hashes") or {}).get("assembly.step"),
        ),
        _artifact(
            ctx,
            contract_path,
            label="装配合同",
            kind="contract",
            stage_id="design-package",
            role="output",
            summary="solid、零件、刚体组和单位映射",
        ),
    ]
    evidence.extend(package_artifacts)

    design_input_artifact = _artifact(
        ctx,
        design_input_path,
        label="原始生成 Prompt 与工程说明",
        kind="design-input",
        stage_id="design-package",
        role="input",
        summary="记录这次要造什么、原始 Prompt、主要组件和工程假设。",
        status=design_quality_status if design_quality_recorded else "warning",
    )
    visual_review_artifact = _artifact(
        ctx,
        visual_review_path,
        label="Worker 补充图（worker_supplemental）",
        kind="visual-review",
        stage_id="design-package",
        role="evidence",
        summary="可选 Worker 补充图（worker_supplemental）；review_authority=false，不是 Harness 冻结审核视图。",
        status=design_quality_status if design_quality_recorded else "warning",
    )
    design_evaluation_artifact = _artifact(
        ctx,
        design_evaluation_path,
        label="设计输入与包装完整性",
        kind="visual-evidence-completeness",
        stage_id="design-package",
        role="output",
        summary="确认 design_input 与包装文件齐全；Worker 补充图不阻塞，也不评价整机通过。",
        status=design_quality_status if design_quality_recorded else "warning",
    )
    evidence.extend(
        [design_input_artifact, visual_review_artifact, design_evaluation_artifact]
    )
    design_render_artifacts: list[dict[str, Any]] = []
    for index, asset in enumerate(visual_review.get("snapshot_render_assets") or []):
        if not isinstance(asset, dict):
            continue
        relative_path = str(asset.get("path") or "").strip()
        if not relative_path:
            continue
        asset_path = (run / relative_path).resolve(strict=False)
        role = str(asset.get("role") or f"view-{index + 1}")
        label = str(asset.get("label") or role)
        render_authority = str(
            asset.get("render_authority") or WORKER_SUPPLEMENTAL_RENDER_AUTHORITY
        )
        review_authority = bool(asset.get("review_authority"))
        artifact = _artifact(
            ctx,
            asset_path,
            label=f"Worker 补充图（worker_supplemental）：{label}",
            kind="visual-image",
            stage_id="design-package",
            role="evidence",
            summary="Worker 补充图（worker_supplemental）；review_authority=false；不是 Harness 冻结审核视图。",
            sha256=str(asset.get("sha256") or ""),
            status=design_quality_status,
        )
        artifact["details"].update(
            {
                "visual_role": f"design-review-{role}",
                "review_round": asset.get("round"),
                "render_authority": render_authority,
                "review_authority": review_authority,
            }
        )
        artifact["preview_type"] = "image"
        if mode == "snapshot" and artifact["exists"]:
            artifact["preview_data_uri"] = _data_uri(asset_path)
        evidence.append(artifact)
        design_render_artifacts.append(artifact)

    intake_artifact = _artifact(
        ctx,
        intake_path,
        label="仿真前检查结果",
        kind="intake",
        stage_id="intake",
        role="output",
        summary="记录设计资料是否齐全、整机几何能否打开，以及是否可以继续验证。",
        status=_status_from_value(intake.get("status")),
    )
    evidence.append(intake_artifact)
    plan_artifact = _artifact(
        ctx,
        plan_path,
        label=f"{capsule_id} 机构计划",
        kind="plan",
        stage_id="mechanism-plan",
        role="output",
        summary="可信机构边界、轴、输入、观测量和适配器",
    )
    evidence.append(plan_artifact)
    geometry_artifact_path = provenance_path if provenance else compound_geometry_path
    provenance_artifact = _artifact(
        ctx,
        geometry_artifact_path,
        label="碰撞与关节几何证据" if compound_geometry else "STEP 派生碰撞证据",
        kind="geometry",
        stage_id="geometry-preflight",
        role="output",
        summary=(
            "保存每个刚体的碰撞尺寸、固定状态和运行时姿态"
            if compound_geometry
            else "每个 solid 的碰撞几何和来源哈希"
        ),
    )
    evidence.append(provenance_artifact)
    queue_artifact = _artifact(
        ctx,
        queue_state_path,
        label="双分支串行队列状态",
        kind="queue",
        stage_id="branch-simulation",
        role="output",
        summary="受资源限制的物理/简化任务执行结果",
        status=_status_from_value(queue_state.get("status")),
    )
    evidence.append(queue_artifact)
    physical_preflight_artifact = _artifact(
        ctx,
        physical_preflight_path if physical_preflight_path.is_file() else None,
        label="\u771f\u5b9e\u5b9e\u4f53\u65f6\u95f4\u70b9\u590d\u5ba1",
        kind="physical-preflight",
        stage_id="branch-simulation",
        role="output",
        summary="\u7528\u539f\u59cb STEP \u78b0\u649e\u51e0\u4f55\u68c0\u67e5\u771f\u5b9e\u8f68\u8ff9\u4e2d\u5df2\u4fdd\u5b58\u7684 3D \u65f6\u95f4\u70b9\u3002",
        status=_status_from_value(physical_preflight.get("status")),
    )
    if physical_preflight:
        physical_preflight_artifact["details"].update(
            {
                "trace_count": physical_preflight.get("trace_count"),
                "passed_trace_count": physical_preflight.get("passed_trace_count"),
                "failed_trace_count": physical_preflight.get("failed_trace_count"),
                "reduced_branch_authorized": physical_preflight.get("reduced_branch_authorized"),
            }
        )
        evidence.append(physical_preflight_artifact)
    legacy_physical_audit_artifact = _artifact(
        ctx,
        legacy_physical_audit_path if legacy_physical_audit_path.is_file() else None,
        label="\u65e7\u8fd0\u884c\u8ffd\u52a0\u5b9e\u4f53\u590d\u5ba1",
        kind="legacy-physical-audit",
        stage_id="branch-simulation",
        role="evidence",
        summary="\u4e0d\u91cd\u8dd1\u4eff\u771f\uff0c\u53ea\u5bf9\u65e7\u8f68\u8ff9\u5df2\u4fdd\u5b58\u65f6\u95f4\u70b9\u505a\u72ec\u7acb STEP \u7a7f\u6a21\u5ba1\u67e5\uff1b\u4e0d\u6539\u5199\u65e7\u8bc1\u4e66\u3002",
        status=_status_from_value(legacy_physical_audit.get("status")),
    )
    if legacy_physical_audit:
        evidence.append(legacy_physical_audit_artifact)

    for event in penetration_summary.get("events") or []:
        if not isinstance(event, dict) or not event.get("snapshot"):
            continue
        snapshot_path = ctx.resolve(event.get("snapshot"))
        load_case_id = str(event.get("load_case_id") or "")
        time_s = float(event.get("time_s") or 0.0)
        depth_mm = float(event.get("penetration_depth_m") or 0.0) * 1000.0
        penetration_artifact = _artifact(
            ctx,
            snapshot_path,
            label=f"{load_case_id or 'physical'} · {time_s:.4f} s 穿模特写",
            kind="penetration-visual",
            stage_id="branch-simulation",
            role="evidence",
            summary=(
                f"原始轨迹帧 {int(event.get('source_sample_index') or 0)}，"
                f"最大检出穿入 {depth_mm:.4f} mm；"
                "红色表示 FCL 检出的接触位置和局部重叠邻域。"
            ),
            status="failed",
            sha256=str(event.get("snapshot_sha256") or ""),
        )
        penetration_artifact["details"].update(
            {
                "type": "penetration-event",
                "visual_role": "penetration-failure-closeup",
                "event_id": str(event.get("event_id") or ""),
                "load_case_id": load_case_id,
                "time_s": time_s,
                "source_sample_index": int(event.get("source_sample_index") or 0),
                "audit_sample_index": int(event.get("audit_sample_index") or 0),
                "penetration_depth_mm": depth_mm,
                "solid_pair": list(event.get("solid_pair") or []),
                "dashboard_focus": dict(event.get("dashboard_focus") or {}),
                "simulation_rerun_for_evidence": bool(
                    event.get("simulation_rerun_for_evidence")
                ),
                "red_region_claim": str(event.get("red_region_claim") or ""),
            }
        )
        penetration_artifact["preview_type"] = "image"
        if mode == "snapshot" and penetration_artifact["exists"]:
            penetration_artifact["preview_data_uri"] = _data_uri(snapshot_path)
        evidence.append(penetration_artifact)

    for failure_summary_path in (
        run / "runtime_failure_evidence.json",
        run / "historical_failure_evidence.json",
    ):
        if not failure_summary_path.is_file():
            continue
        failure_summary = ctx.read_json(
            failure_summary_path,
            role="representative physical failure evidence",
        )
        for case in failure_summary.get("cases") or []:
            if not isinstance(case, dict) or not case.get("snapshot"):
                continue
            snapshot_path = ctx.resolve(case.get("snapshot"))
            load_case_id = str(case.get("load_case_id") or "")
            failure_artifact = _artifact(
                ctx,
                snapshot_path,
                label=f"{load_case_id or '实体物理任务'}：代表性失败快照",
                kind="visual-image",
                stage_id="branch-simulation",
                role="evidence",
                summary="保留失败时的真实实体画面和缺失条件；通过后与完成轨迹并排比较。",
                status="failed",
                sha256=str(case.get("snapshot_sha256") or ""),
            )
            failure_artifact["details"].update(
                {
                    "visual_role": "runtime-failure-snapshot",
                    "load_case_id": load_case_id,
                    "failure_code": str(case.get("failure_code") or ""),
                    "failure_stage": str(case.get("failure_stage") or ""),
                    "timeout_seconds": case.get("timeout_seconds"),
                    "missing_requirement": str(case.get("missing_requirement") or ""),
                    "future_comparison_slot": str(case.get("future_comparison_slot") or ""),
                    "historical": failure_summary_path.name.startswith("historical"),
                }
            )
            failure_artifact["preview_type"] = "image"
            if mode == "snapshot" and failure_artifact["exists"]:
                failure_artifact["preview_data_uri"] = _data_uri(snapshot_path)
            evidence.append(failure_artifact)

    load_cases = []
    planned_capsules = list(
        dict.fromkeys(
            str(item.get("capsule_id") or "")
            for item in plans
            if str(item.get("capsule_id") or "")
        )
    )
    if capsule_id and capsule_id not in planned_capsules:
        planned_capsules.insert(0, capsule_id)
    for load_capsule_id in planned_capsules:
        load_root = run / "loads" / load_capsule_id
        if not load_root.is_dir():
            continue
        load_paths = sorted(
            load_root.glob("*.json"),
            key=lambda path: (LOAD_ORDER.get(path.stem, 999), path.name),
        )
        for path in load_paths:
            load_cases.append(
                _load_case(
                    ctx,
                    run,
                    load_capsule_id,
                    path,
                    evidence,
                    execution_lane=physical_execution_lanes.get(
                        (load_capsule_id, path.stem)
                    ),
                    physical_trace_record=physical_trace_records.get(
                        (load_capsule_id, path.stem)
                    ),
                    ideal_reference_trace_record=ideal_reference_trace_records.get(
                        (load_capsule_id, path.stem)
                    ),
                    ideal_reference_status=ideal_reference_summary.get("status"),
                )
            )

    ideal_labels = {str(item.get("ideal_label") or "") for item in load_cases if item.get("ideal_label")}
    if len(ideal_labels) == 1:
        comparison_ideal_label = next(iter(ideal_labels))
    elif ideal_labels:
        comparison_ideal_label = "混合对照（" + "、".join(sorted(ideal_labels)) + "）"
    else:
        comparison_ideal_label = "未记录理想参考"

    whole_results = registered_whole_results
    primary_whole_registration = whole_results[0] if whole_results else {}
    whole_capsule_id = str(
        primary_whole_registration.get("capsule_id") or capsule_id
    )
    mechanism_status = (
        mechanisms_by_capsule.get(whole_capsule_id)
        or (registered_mechanisms[0] if registered_mechanisms else {})
    )
    certificate_path = ctx.resolve(
        mechanism_status.get("certificate")
        or f"{ctx.relative(run)}/results/{whole_capsule_id}/certificate.json"
    )
    replacement_path = ctx.resolve(
        mechanism_status.get("replacement")
        or f"{ctx.relative(run)}/results/{whole_capsule_id}/replacement.json"
    )
    whole_result_ref = primary_whole_registration.get("result_file")
    # Whole-machine evidence is authoritative only when this run explicitly
    # registers it. A leftover result file must never make another run look closed.
    whole_path = ctx.resolve(whole_result_ref) if whole_result_ref else None
    whole_plan_path = whole_path.parent / "execution_plan.json" if whole_path else None
    certificate = ctx.read_json(certificate_path, role="mechanism certificate")
    replacement = ctx.read_json(replacement_path, role="replacement")
    whole = ctx.read_json(whole_path, role="whole-machine result")
    whole_plan = (
        ctx.read_json(whole_plan_path, role="whole-machine execution plan")
        if whole_plan_path
        else {}
    )

    certificate_sha256_actual = _sha256_file(certificate_path)
    replacement_sha256_actual = _sha256_file(replacement_path)
    certificate_status = _status_from_value(certificate.get("verdict"))
    replacement_status = _status_from_value(replacement.get("status"))
    whole_status = _status_from_value(whole.get("status"))
    replacement_policy = str(
        mechanism_status.get("replacement_policy")
        or run_status.get("replacement_policy")
        or certificate.get("replacement_policy")
        or replacement.get("replacement_policy")
        or "certified_mechanism"
    )
    reference_only_single_mechanism = (
        replacement_policy == "reference_only_single_mechanism"
    )
    certificate_package_binding = str((certificate.get("package") or {}).get("package_binding_sha256") or "")
    legacy_replacement_checkpoint_overwrite = bool(
        str(run_status.get("status") or "") == "WHOLE_MACHINE_TASK_PASSED"
        and certificate_status == "passed"
        and whole_status == "passed"
        and str(replacement.get("status") or "") == "PHYSICS_PARTIAL"
        and str(replacement.get("branch") or "") == "reduced"
        and isinstance(replacement.get("samples"), list)
        and not replacement.get("replacement_model")
        and whole.get("replacement_sha256")
        and replacement_sha256_actual
        and whole.get("replacement_sha256") != replacement_sha256_actual
    )
    replacement_stale_reasons: list[str] = []
    replacement_audit_warnings: list[str] = []
    if legacy_replacement_checkpoint_overwrite:
        replacement_status = "warning"
        replacement_audit_warnings.append(
            "历史 worker 将简化分支检查点覆盖到了 replacement.json；"
            "最终整机 PASS 与已保存物理证据保持原样，此处仅提示旧文件审核不一致。"
        )
        diagnostics.append(
            {
                "level": "warning",
                "role": "legacy replacement record reconciliation",
                "path": ctx.relative(replacement_path),
                "detail": replacement_audit_warnings[0],
            }
        )
    elif replacement_status == "passed":
        if replacement.get("certificate_sha256") and certificate_sha256_actual and replacement.get("certificate_sha256") != certificate_sha256_actual:
            replacement_stale_reasons.append("绑定的证书哈希与当前证书文件不一致。")
        if replacement.get("package_binding_sha256") and certificate_package_binding and replacement.get("package_binding_sha256") != certificate_package_binding:
            replacement_stale_reasons.append("Replacement 与证书的设计包绑定不一致。")
    if replacement_stale_reasons:
        replacement_status = "stale"

    whole_stale_reasons: list[str] = []
    whole_audit_warnings: list[str] = []
    if whole_status == "passed":
        if certificate_status != "passed":
            whole_stale_reasons.append(
                "整机结果对应的机构证书缺失或未通过。"
            )
        if (
            not reference_only_single_mechanism
            and replacement_status != "passed"
            and not legacy_replacement_checkpoint_overwrite
        ):
            whole_stale_reasons.append(
                "整机结果对应的 Replacement 缺失或未启用。"
            )
        if whole.get("certificate_sha256") and whole.get("certificate_sha256") != certificate_sha256_actual:
            whole_stale_reasons.append("整机结果绑定的证书哈希与当前证书文件不一致。")
        if whole.get("replacement_sha256") and whole.get("replacement_sha256") != replacement_sha256_actual:
            if legacy_replacement_checkpoint_overwrite:
                whole_audit_warnings.append(
                    "整机结果仍绑定运行时已启用的 Replacement 哈希；"
                    "当前同名文件后来被旧检查点覆盖，属于历史文件审核提示。"
                )
            else:
                whole_stale_reasons.append("整机结果绑定的 Replacement 哈希与当前文件不一致。")
        if whole.get("package_binding_sha256") and whole.get("package_binding_sha256") != certificate_package_binding:
            whole_stale_reasons.append("整机结果与证书的设计包绑定不一致。")
    if whole_stale_reasons:
        whole_status = "stale"
    replacement_effectively_active = (
        replacement_status == "passed" or legacy_replacement_checkpoint_overwrite
    )
    replacement_stage_status = (
        "passed" if reference_only_single_mechanism else replacement_status
    )
    whole_stage_status = (
        "passed" if reference_only_single_mechanism else whole_status
    )
    whole_task_rows = [
        {**item, "capsule_id": whole_capsule_id}
        for item in _whole_task_rows(whole, whole_plan)
    ]
    whole_task_labels = [
        str(item.get("label") or item.get("id") or "未命名任务")
        for item in whole_task_rows
    ]
    whole_coordinate = _whole_coordinate_summary(
        whole,
        whole_plan,
        whole_task_rows,
    )

    certificate_artifact = _artifact(
        ctx,
        certificate_path,
        label="运动学证书",
        kind="certificate",
        stage_id="certificate",
        role="output",
        summary="比较覆盖、适用包、质量、载荷和运动区间",
        status=certificate_status,
        sha256=certificate_sha256_actual,
    )
    replacement_artifact = _artifact(
        ctx,
        replacement_path,
        label="简化机构启用记录",
        kind="replacement",
        stage_id="replacement",
        role="output",
        summary="记录简化机构是否已启用，以及它绑定的设计版本和运动证书。",
        status=replacement_stage_status,
        sha256=replacement_sha256_actual,
    )
    whole_artifact = _artifact(
        ctx,
        whole_path,
        label="整机任务结果",
        kind="whole-machine",
        stage_id="whole-machine",
        role="output",
        summary=(
            f"记录简化机构放回整机后，{len(whole_task_rows)} 项已登记任务是否分别完成。"
        ),
        status=whole_stage_status,
        sha256=_sha256_file(whole_path),
    )
    whole_plan_artifact = (
        _artifact(
            ctx,
            whole_plan_path,
            label="整机任务目标与容差",
            kind="whole-machine-plan",
            stage_id="whole-machine",
            role="input",
            summary=(
                f"记录 {len(whole_task_rows)} 项整机任务的目标、输入限制和各自门槛。"
            ),
            status="passed" if whole_plan else "not_started",
            sha256=_sha256_file(whole_plan_path),
        )
        if whole_plan_path
        else None
    )
    tested_envelope = certificate.get("tested_envelope") or {}
    certificate_artifact["details"] = {
        "type": "certificate",
        "capsule_id": whole_capsule_id,
        "status": certificate_status,
        "verdict": str(certificate.get("verdict") or "NOT_AVAILABLE"),
        "level": str(certificate.get("certificate_level") or "—"),
        "issued_at": str(certificate.get("issued_at") or ""),
        "package_id": str((certificate.get("package") or {}).get("package_id") or "—"),
        "design_version": str((certificate.get("package") or {}).get("design_version") or "—"),
        "certificate_sha256": certificate_sha256_actual,
        "package_binding_sha256": certificate_package_binding,
        "comparison_sha256": str(certificate.get("comparison_sha256") or ""),
        "mass_kg": tested_envelope.get("moving_mass_kg") or {},
        "force_n": tested_envelope.get("applied_force_n") or {},
        "gravity_m_s2": tested_envelope.get("gravity_abs_m_s2") or [],
        "progress_mm": {
            "min": _round((_finite((tested_envelope.get("generalized_progress_m") or {}).get("min")) or 0.0) * 1000, 4),
            "max": _round((_finite((tested_envelope.get("generalized_progress_m") or {}).get("max")) or 0.0) * 1000, 4),
        },
        "angle_deg": {
            "min": _round(math.degrees(_finite((tested_envelope.get("generalized_angle_rad") or {}).get("min")) or 0.0), 4),
            "max": _round(math.degrees(_finite((tested_envelope.get("generalized_angle_rad") or {}).get("max")) or 0.0), 4),
        },
        "certifiable_angle_limit_deg": _round(math.degrees(_finite((certificate.get("physical_contact_model") or {}).get("certifiable_angle_limit_rad")) or 0.0), 4),
        "certifiable_progress_limit_mm": _round((_finite((certificate.get("physical_contact_model") or {}).get("certifiable_progress_limit_m")) or 0.0) * 1000, 4),
        "load_case_ids": tested_envelope.get("load_case_ids") or [],
        "boundary": (
            "仅认证记录包络内的运动学行为；不认证逐时刻速度、动态响应、真实摩擦效率、强度、疲劳、制造或现实安全。"
            if certificate_status == "passed"
            else "聚合证书未签发；以下数值只描述已执行测试，不构成可用认证包络。"
        ),
    }
    replacement_certificate_hash = str(replacement.get("certificate_sha256") or "")
    replacement_package_binding = str(replacement.get("package_binding_sha256") or "")
    replacement_artifact["details"] = {
        "type": "replacement",
        "capsule_id": whole_capsule_id,
        "status": str(replacement.get("status") or "NOT_AVAILABLE"),
        "normalized_status": replacement_status,
        "family": str(
            (replacement.get("replacement_model") or {}).get("family")
            or plan.get("family")
            or (certificate.get("mechanism") or {}).get("family")
            or "—"
        ),
        "certificate_sha256": replacement_certificate_hash,
        "certificate_hash_matches": bool(
            replacement_certificate_hash
            and certificate_sha256_actual
            and replacement_certificate_hash == certificate_sha256_actual
        ),
        "package_binding_sha256": replacement_package_binding,
        "package_hash_matches": bool(
            replacement_package_binding
            and certificate_package_binding
            and replacement_package_binding == certificate_package_binding
        ),
        "envelope": replacement.get("validity_envelope") or {},
        "invalidation_rules": replacement.get("invalidation_rules") or [],
        "fallback": str(replacement.get("fallback") or "—"),
        "stale_reasons": replacement_stale_reasons,
        "audit_warnings": replacement_audit_warnings,
        "historical_terminal_authority": legacy_replacement_checkpoint_overwrite,
    }
    whole_artifact["details"] = {
        "type": "whole-machine",
        "capsule_id": whole_capsule_id,
        "status": str(whole.get("status") or "NOT_AVAILABLE"),
        "tasks": whole_task_rows,
        "task_passed": sum(item["status"] == "passed" for item in whole_task_rows),
        "task_total": len(whole_task_rows),
        "coordinate": whole_coordinate,
        "max_lateral_shift_mm": _round((_finite(whole.get("max_lateral_shift_m")) or 0.0) * 1000, 6),
        "max_rotation_deg": _round(math.degrees(_finite(whole.get("max_rotation_rad")) or 0.0), 6),
        "max_pivot_drift_mm": _round((_finite(whole.get("max_pivot_drift_m")) or 0.0) * 1000, 6),
        "max_off_axis_rotation_deg": _round(math.degrees(_finite(whole.get("max_off_axis_rotation_rad")) or 0.0), 9),
        "position_tolerance_mm": _round((_finite(whole_plan.get("position_tolerance_m")) or 0.0) * 1000, 4) if whole_plan.get("position_tolerance_m") is not None else None,
        "angle_tolerance_deg": _round(math.degrees(_finite(whole_plan.get("angle_tolerance_rad")) or 0.0), 4) if whole_plan.get("angle_tolerance_rad") is not None else None,
        "max_stage_duration_s": _round(whole_plan.get("max_stage_duration_s"), 4),
        "max_control_force_n": _round(whole_plan.get("max_control_force_n"), 4),
        "max_control_torque_nm": _round(whole_plan.get("max_control_torque_nm"), 4),
        "sample_count": whole.get("sample_count"),
        "certificate_sha256": str(whole.get("certificate_sha256") or ""),
        "replacement_sha256": str(whole.get("replacement_sha256") or ""),
        "package_binding_sha256": str(whole.get("package_binding_sha256") or ""),
        "stale_reasons": whole_stale_reasons,
        "audit_warnings": whole_audit_warnings,
        "boundary": (
            "单机构包不启用简化整机替换；外部任务结论直接来自已认证 physical 轨迹。"
            if reference_only_single_mechanism
            else "这是已认证简化替换的整机外部任务；不是高保真内部接触运行，控制器也不代表真实硬件。"
            if whole_status == "passed"
            else "尚未生成通过的整机任务证据；空表或未完成结果不能解释为已消费认证替换。"
        ),
    }
    evidence.extend([certificate_artifact, replacement_artifact, whole_artifact])
    if whole_plan_artifact:
        whole_plan_artifact["details"].update(
            {"capsule_id": whole_capsule_id}
        )
        evidence.append(whole_plan_artifact)

    primary_whole_view = {
        "capsule_id": whole_capsule_id,
        "status": whole_stage_status,
        "status_raw": str(whole.get("status") or "NOT_AVAILABLE"),
        "required": not reference_only_single_mechanism,
        "replacement_policy": replacement_policy,
        "task_ids": whole.get("task_ids") or [],
        "stages": whole_task_rows,
        "task_passed": sum(
            item.get("status") == "passed" for item in whole_task_rows
        ),
        "task_total": len(whole_task_rows),
        "coordinate": whole_coordinate,
        "max_lateral_shift_mm": whole_artifact["details"][
            "max_lateral_shift_mm"
        ],
        "max_rotation_deg": whole_artifact["details"]["max_rotation_deg"],
        "max_pivot_drift_mm": whole_artifact["details"][
            "max_pivot_drift_mm"
        ],
        "max_off_axis_rotation_deg": whole_artifact["details"][
            "max_off_axis_rotation_deg"
        ],
        "position_tolerance_mm": whole_artifact["details"][
            "position_tolerance_mm"
        ],
        "angle_tolerance_deg": whole_artifact["details"][
            "angle_tolerance_deg"
        ],
        "max_stage_duration_s": whole_artifact["details"][
            "max_stage_duration_s"
        ],
        "max_control_force_n": whole_artifact["details"][
            "max_control_force_n"
        ],
        "max_control_torque_nm": whole_artifact["details"][
            "max_control_torque_nm"
        ],
        "sample_count": whole.get("sample_count"),
        "strict_non_claims": whole.get("strict_non_claims") or {},
        "stale_reasons": whole_stale_reasons,
        "boundary": whole_artifact["details"]["boundary"],
    }
    whole_machine_views: list[dict[str, Any]] = []
    if whole_path is not None:
        whole_machine_views.append(primary_whole_view)
    for registration in registered_whole_results[1:]:
        view, scoped_artifacts = _additional_whole_machine_view(
            ctx=ctx,
            run=run,
            registration=registration,
            mechanisms_by_capsule=mechanisms_by_capsule,
        )
        whole_machine_views.append(view)
        evidence.extend(scoped_artifacts)
    if len(whole_machine_views) > 1:
        whole_task_rows = [
            task
            for view in whole_machine_views
            for task in view.get("stages", [])
        ]
        whole_task_labels = [
            str(item.get("label") or item.get("id") or "未命名任务")
            for item in whole_task_rows
        ]
        whole_stage_status = _combine_status(
            [str(view.get("status") or "not_started") for view in whole_machine_views]
        )
        whole_status = whole_stage_status
        whole_stale_reasons = [
            str(reason)
            for view in whole_machine_views
            for reason in (view.get("stale_reasons") or [])
        ]

    visual_root = run / "visual"
    canonical_visuals_ready = True
    available_bound_visuals: set[Path] = set()
    run_identity = parse_canonical_run(workspace, run)
    run_has_physical_trace = any(
        (run / "traces").glob("**/physical.trace.json")
    )
    if run_identity is not None and run_has_physical_trace:
        visual_run_status = (
            run_status
            if run == requested_run
            else ctx.read_json(run / "run_status.json", role="source run status")
        )
        visual_summary = ctx.read_json(
            visual_root / "dashboard_visuals.summary.json",
            role="dashboard visualization index",
        )
        canonical_visuals_ready = bool(
            (visual_run_status.get("visualization") or {}).get("status") == "READY"
            and visual_summary.get("status") == "READY"
            and (visual_summary.get("canonical_step") or {}).get("ready") is True
        )
        available_bound_visuals = _bound_available_visuals(ctx, run, visual_summary)
        if not canonical_visuals_ready:
            diagnostics.append(
                {
                    "level": "missing",
                    "role": "canonical STEP visualization precondition",
                    "path": ctx.relative(visual_root),
                    "detail": "the visualization set is incomplete; existing individually bound media remain visible without changing validation or certificate status",
                }
            )
    preview_safe_names = {
        "stage1_design_package.png",
        "stage1_design_package.render.json",
        "design_preview.summary.json",
        "dashboard_visuals.summary.json",
    }
    if visual_root.is_dir():
        for path in sorted(visual_root.iterdir()):
            if path.suffix.lower() not in {".mp4", ".png", ".glb", ".json"}:
                continue
            if not path.is_file():
                continue
            if not canonical_visuals_ready and path.name not in preview_safe_names and path.resolve() not in available_bound_visuals:
                continue
            suffix = path.suffix.lower()
            meta = _visual_metadata(path)
            item = _artifact(
                ctx,
                path,
                label=meta["label"],
                kind=("visual-video" if suffix == ".mp4" else "visual-image" if suffix == ".png" else "interactive-model" if suffix == ".glb" else "render-summary"),
                stage_id=meta["stage_id"],
                role="evidence",
                summary=meta["summary"],
            )
            item["details"].update(
                {
                    key: value
                    for key, value in meta.items()
                    if key not in {"stage_id", "label", "summary"}
                }
            )
            if item["details"].get("visual_role") == "physics-evidence-replay":
                replay = ctx.read_json(
                    path, role="physics-evidence-replay"
                )
                try:
                    replay_capsule_id = str(
                        replay.get("capsule_id")
                        or item["details"].get("capsule_id")
                        or ""
                    )
                    replay = upgrade_replay_for_presentation(
                        replay,
                        workspace_root=workspace,
                        plan=plans_by_capsule.get(replay_capsule_id, plan),
                    )
                except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                    diagnostics.append(
                        {
                            "level": "warning",
                            "role": "clock transmission arrow presentation overlay",
                            "path": ctx.relative(path),
                            "detail": str(exc),
                        }
                    )
                item["details"]["replay"] = replay
                item["details"]["capsule_id"] = str(
                    replay.get("capsule_id")
                    or item["details"].get("capsule_id")
                    or ""
                )
                item["details"]["load_case_id"] = str(
                    replay.get("load_case_id")
                    or item["details"].get("load_case_id")
                    or ""
                )
            if item["details"].get("visual_role") == "physics-evidence-scene-manifest":
                item["details"]["scene_manifest"] = ctx.read_json(
                    path, role="physics-evidence-scene-manifest"
                )
            item["preview_type"] = "video" if suffix == ".mp4" else "image" if suffix == ".png" else "model" if suffix == ".glb" else ""
            if mode == "snapshot" and item["preview_type"]:
                item["preview_data_uri"] = _data_uri(path)
            evidence.append(item)
    # Terminal rNN directories are immutable.  A pose-rich ideal replay may
    # therefore be derived later from the same hash-bound physical trace and
    # stored under diagnostics rather than injected into the terminal run.
    # Surface that explicitly labelled post-run presentation asset in the
    # reference-only stage without changing the run's verdict or hash index.
    if reference_only_single_mechanism and run_identity is not None:
        diagnostic_root = (
            workspace
            / ".runtime_runs"
            / "diagnostics"
            / f"{run_identity.run_id}-ideal-pose"
        )
        scoped_reference_video = (
            diagnostic_root
            / capsule_id
            / "physical_vs_ideal_roundtrip.mp4"
        )
        legacy_reference_video = (
            diagnostic_root / "physical_vs_ideal_roundtrip.mp4"
        )
        reference_video = (
            scoped_reference_video
            if scoped_reference_video.is_file()
            else legacy_reference_video
        )
        reference_sidecar = reference_video.with_suffix(".sidecar.json")
        if reference_video.is_file() and reference_sidecar.is_file():
            sidecar = ctx.read_json(
                reference_sidecar,
                role="post-run ideal pose comparison sidecar",
            )
            trace_hashes = sidecar.get("trace_sha256") or {}
            if (
                sidecar.get("status") == "ENCODED"
                and trace_hashes.get("physical")
                and trace_hashes.get("reduced")
            ):
                item = _artifact(
                    ctx,
                    reference_video,
                    label="真实物理与理想参考同步回放",
                    kind="visual-video",
                    stage_id="replacement",
                    role="evidence",
                    summary=(
                        "由已保存的真实物理轨迹和方程计算的理想刚体姿态生成；"
                        "没有重跑仿真，也不赋予理想参考物理判定权限。"
                    ),
                    sha256=_sha256_file(reference_video),
                )
                item["details"].update(
                    {
                        "visual_role": "reference-comparison-video",
                        "capsule_id": capsule_id,
                        "sidecar": ctx.relative(reference_sidecar),
                        "comparison_branch": "ideal_reference",
                        "load_case_id": str(
                            sidecar.get("load_case_id") or ""
                        ),
                        "trace_sha256": trace_hashes,
                        "post_run_presentation_derivative": True,
                    }
                )
                item["preview_type"] = "video"
                if mode == "snapshot":
                    item["preview_data_uri"] = _data_uri(reference_video)
                evidence.append(item)
    _attach_diagnostic_physics_evidence(
        ctx=ctx,
        workspace=workspace,
        run_identity=run_identity,
        evidence=evidence,
        diagnostics=diagnostics,
        plans_by_capsule=plans_by_capsule,
        plan=plan,
        mode=mode,
    )

    intake_summary = intake.get("summary") or {}
    intake_capabilities = intake.get("capabilities") or {}
    package_summary = intake.get("package_summary") or {}
    intake_status = _status_from_value(intake.get("status"), missing="blocked")
    package_complete = all(item["exists"] for item in package_artifacts)
    react_status = str(react_loop_state.get("status") or "")
    static_review_state = react_loop_state.get("static_visual_review")
    static_review_status = str(
        static_review_state.get("status")
        if isinstance(static_review_state, dict)
        else ""
    )
    if static_review_status == "PENDING":
        design_status = "awaiting_review"
    elif static_review_status in {
        "CHANGES_REQUESTED",
        "REJECTED",
    }:
        design_status = "failed"
    elif static_review_status in {"EVIDENCE_REQUIRED", "INVALID"}:
        design_status = "blocked"
    elif react_status in {"ASSET_DRAFT_REQUIRED", "DESIGN_REQUIRED"}:
        design_status = "running"
    elif react_status == "CERTIFICATION_SUCCESSOR_REQUIRED":
        design_status = "preparing"
    elif react_status in {"PREFLIGHT_REQUIRED", "TEST_REQUIRED"}:
        design_status = "preparing"
    elif not package_complete:
        design_status = "blocked"
    elif design_quality_recorded:
        design_status = design_quality_status
    else:
        design_status = "warning"
    plan_status = "passed" if plan else ("blocked" if intake_status in {"failed", "blocked"} else "not_started")
    geometry_status = "passed" if geometry_evidence else ("blocked" if plan_status == "passed" else "not_started")
    if legacy_queue_state_path:
        queue_summaries = [
            queue_state.get("summary") or run_status.get("queue_summary") or {}
        ]
        queue_statuses = [_status_from_value(queue_state.get("status"))]
    else:
        queue_summaries = [
            item.get("summary") or {}
            for item in (physical_queue_state, reduced_queue_state)
            if item
        ]
        queue_statuses = [
            _status_from_value(item.get("status"))
            for item in (physical_queue_state, reduced_queue_state)
            if item
        ]
    job_total = sum(int(item.get("total") or 0) for item in queue_summaries)
    job_pass = sum(int(item.get("SUCCEEDED") or 0) for item in queue_summaries)
    queue_status = _combine_status(queue_statuses)
    if job_total > 0 and job_pass >= job_total:
        queue_status = "passed"
    trace_status = _combine_status(
        [
            branch["status"]
            for case in load_cases
            for branch in (case["physical"], case["reduced"])
        ]
    )
    simulation_status = _combine_status([queue_status, trace_status])
    comparison_status = _combine_status([case["status"] for case in load_cases])
    cert_stage_status = _combine_status([comparison_status, certificate_status])

    active_physical_audit = physical_preflight or legacy_physical_audit
    physical_audit_status = _status_from_value(
        active_physical_audit.get("status")
    )
    if active_physical_audit:
        simulation_status = _combine_status(
            [simulation_status, physical_audit_status]
        )
    audit_traces = active_physical_audit.get("traces") or []
    physical_audit_trace_count = int(
        active_physical_audit.get("trace_count") or len(audit_traces)
    )
    physical_audit_passed_count = int(
        active_physical_audit.get("passed_trace_count")
        or sum(item.get("passed") is True for item in audit_traces)
    )
    physical_audit_failed_count = int(
        active_physical_audit.get("failed_trace_count")
        or sum(item.get("passed") is not True for item in audit_traces)
    )
    physical_audit_max_penetration_m = max(
        (
            _finite((item.get("geometry_legality") or {}).get("max_penetration_m"))
            or 0.0
            for item in audit_traces
            if isinstance(item, dict)
        ),
        default=0.0,
    )
    physical_audit_selected_pose_count = sum(
        int(
            (item.get("sampling") or {}).get("selected_sample_count")
            or (item.get("geometry_legality") or {}).get("sample_count")
            or 0
        )
        for item in audit_traces
        if isinstance(item, dict)
    )
    component_contact_audits = [
        item.get("component_contact_intent")
        for item in audit_traces
        if isinstance(item, dict)
        and isinstance(item.get("component_contact_intent"), dict)
    ]
    component_contact_outcomes = [
        outcome
        for audit in component_contact_audits
        for outcome in audit.get("outcomes") or ()
        if isinstance(outcome, dict)
    ]
    component_contact_summary = {
        "recorded": bool(component_contact_audits),
        "authoritative_audit_count": sum(
            item.get("authoritative") is True for item in component_contact_audits
        ),
        "expected_count": sum(
            item.get("kind") == "EXPECTED_REAL_CONTACT"
            for item in component_contact_outcomes
        ),
        "expected_observed_count": sum(
            item.get("status") == "EXPECTED_CONTACT_OBSERVED"
            for item in component_contact_outcomes
        ),
        "support_count": sum(
            item.get("kind") == "COMPLIANT_SUPPORT"
            for item in component_contact_outcomes
        ),
        "analytical_transmission_count": sum(
            item.get("kind") == "ANALYTICAL_TRANSMISSION"
            for item in component_contact_outcomes
        ),
        "forbidden_contact_failure_count": sum(
            item.get("status") == "FORBIDDEN_CONTACT_OBSERVED"
            for item in component_contact_outcomes
        ),
        "uncertain_attribution_event_count": sum(
            int(item.get("ambiguous_attribution_event_count") or 0)
            for item in component_contact_audits
        ),
        "runtime_body_pair_contacts_used_as_component_evidence": False,
    }
    if "reduced_branch_started" in run_status:
        reduced_branch_started = bool(run_status.get("reduced_branch_started"))
    else:
        reduced_branch_started = bool(
            reduced_queue_state
            or any(case["reduced"]["status"] != "not_started" for case in load_cases)
        )

    intake_issue_details = [
        _issue_detail(item)
        for item in intake.get("issues", [])[:8]
        if isinstance(item, dict)
    ]
    intake_issues = [f"{item['title']}：{item['summary']}" for item in intake_issue_details]
    comparison_pass = sum(case["status"] == "passed" for case in load_cases)
    comparison_total = len(load_cases)
    if provenance:
        fixed_count = len((provenance.get("fixed") or {}).get("solid_ids") or [])
        moving_count = len((provenance.get("moving") or {}).get("solid_ids") or [])
    else:
        contract = ctx.read_json(contract_path, role="assembly contract for geometry counts")
        fixed_groups = set(plan.get("fixed_group_ids") or [])
        fixed_count = sum(
            len(part.get("source_solid_ids") or [])
            for part in contract.get("parts") or []
            if part.get("rigid_group_id") in fixed_groups or part.get("role") == "fixed"
        )
        moving_count = sum(
            len(part.get("source_solid_ids") or [])
            for part in contract.get("parts") or []
            if not (part.get("rigid_group_id") in fixed_groups or part.get("role") == "fixed")
        )

    joint_plan = task_planning.get("joint_plan") or []
    test_plan = task_planning.get("test_plan") or []
    authored_test_tasks = task_planning.get("authored_test_tasks") or []
    planning_warnings = _task_planning_warnings(
        task_planning,
        plans,
        load_cases,
    )
    task_execution_coverage = _task_execution_coverage(
        task_planning,
        load_cases,
    )
    planning_stage_status = (
        "evidence_complete"
        if task_planning
        else "awaiting_review"
        if react_status in {
            "PLAN_REVIEW_REQUIRED",
            "RESOURCE_APPROVAL_REQUIRED",
            "PLUGIN_IMPLEMENTATION_REQUIRED",
        }
        else "running"
        if react_status in {
            "TASK_INPUT_REPAIR_REQUIRED",
            "TASK_UNDERSTANDING_REQUIRED",
            "TASK_PLAN_REQUIRED",
        }
        else "evidence_incomplete"
        if task_planning_exists
        else "not_started"
    )
    stages = [
        _stage(
            "task-planning",
            1,
            "接收任务与测试规划",
            "任务规划",
            "先理解冻结的机械任务，再确定关节方案和验证任务，并按需制作测试任务。",
            planning_stage_status,
            why=(
                "任务理解、关节计划、测试计划和按需测试任务已绑定到本 revision。"
                if task_planning
                else "Worker 已提交项目资源，等待独立审核后继续规划。"
                if react_status == "RESOURCE_APPROVAL_REQUIRED"
                else "关节表已批，等待 AI 审核员对照该表挂上 MechanismAdapter。"
                if react_status == "PLUGIN_IMPLEMENTATION_REQUIRED"
                else "任务规划文件存在，但没有通过 revision 与任务来源复核。"
                if task_planning_exists
                else "当前 revision 尚未保存标准任务规划产物。"
            ),
            prerequisite="人类要求和任务合同已经冻结到本 revision。",
            process=[
                "理解机械任务和交付要求",
                "规划所需关节或关节模板",
                "规划每项验证任务",
                "按需制作缺少的测试任务",
            ],
            inputs=["冻结的人类任务要求", "任务合同"],
            outputs=["任务理解", "关节计划", "测试计划", "按需测试任务"],
            evidence_ids=(
                ([task_planning_artifact["id"]] if task_planning_exists else [])
                + (
                    [react_loop_artifact["id"]]
                    if react_loop_artifact is not None
                    else []
                )
            ),
            metrics=[
                {
                    "label": "任务理解",
                    "value": "已记录" if task_planning else "未记录",
                },
                {"label": "关节计划", "value": str(len(joint_plan))},
                {"label": "测试计划", "value": str(len(test_plan))},
                {
                    "label": "新增测试任务",
                    "value": str(len(authored_test_tasks)),
                },
                {
                    "label": "逐环控制",
                    "value": react_status or "未启动",
                },
                {
                    "label": "已有执行证据",
                    "value": (
                        f"{sum(item['status'] == 'passed' for item in task_execution_coverage)}"
                        f"/{len(task_execution_coverage)}"
                    ),
                },
            ],
            issues=(
                [task_planning_error]
                if task_planning_error
                else [react_loop_state_error]
                if react_loop_state_error
                else planning_warnings
            ),
            caveats=[
                "关节或测试目标与后续证据不一致只在这里警告；规划本身不签发机械 PASS。"
            ],
        ),
        _stage(
            "design-package",
            2,
            "AI 完整设计包",
            "设计包",
            "确认 AI 交付了完整、可查看且与权威 STEP 绑定的证据包。",
            design_status,
            why=(
                "所需正交、iso、detail、section 与 pose 视图已绑定，"
                "正在等待独立 Reviewer 决定；物理尚未开始。"
                if static_review_status == "PENDING"
                else "独立 Reviewer 已要求修改外形，等待 Worker 返修。"
                if static_review_status == "CHANGES_REQUESTED"
                else "独立 Reviewer 已拒绝当前外形，自动视觉返修已经停止。"
                if static_review_status == "REJECTED"
                else "当前版本还没有物理前所需的六个正交 PNG 视图。"
                if static_review_status == "EVIDENCE_REQUIRED"
                else "独立外形审核记录未通过完整性复核，不能据此继续。"
                if static_review_status == "INVALID"
                else
                "原始 Prompt、工程说明、整机 CAD 和设计输入/包装均已绑定；这不是整机通过。"
                if design_status == "evidence_complete"
                else "历史运行未保存完整 Prompt 或包装文件，无法判断设计输入完整性。"
                if design_status == "warning"
                else "Prompt 或包装文件不完整，暂时不能继续。"
            ),
            prerequisite="原始生成 Prompt、工程目标与 AI 交付版本已锁定。",
            process=[
                "读取原始 Prompt 与工程目标",
                "生成并打开权威整机 CAD",
                "Harness 冻结 STEP 并生成前、后、左、右、上、下六个正交视图",
                "由独立 Reviewer 在物理前批准、退回或拒绝",
            ],
            inputs=["原始生成 Prompt", "工程样机设计说明", "视觉验收清单"],
            outputs=["权威整机 CAD", "装配说明", "Harness 冻结六视图", "独立外形审核决定"],
            evidence_ids=[item["id"] for item in evidence if item["stage_id"] == "design-package"],
            metrics=[
                {"label": "CAD 零件", "value": str(package_summary.get("step_solid_count") or "—")},
                {"label": "包内文件", "value": str(package_summary.get("package_file_count") or len(package_manifest.get("artifacts") or []))},
                {"label": "证据轮次", "value": str(len(visual_review.get("rounds") or [])) if design_quality_recorded else "未记录"},
                {
                    "label": "提交方视觉备注",
                    "value": str(
                        int(design_reported_findings.get("P0") or 0)
                        + int(design_reported_findings.get("P1") or 0)
                        + int(design_reported_findings.get("P2") or 0)
                    ) if design_quality_recorded else "未知",
                },
            ],
            caveats=[
                "EVIDENCE_COMPLETE 只说明设计输入与包装文件齐全；Worker 补充图不阻塞；不判断画面质量、机构功能或整机通过。"
            ],
        ),
        _stage(
            "intake",
            3,
            "进入仿真前检查",
            "仿真前检查",
            "确认设计包资料齐全、几何能打开，可以交给后续验证。",
            intake_status,
            why=(
                "没有阻塞问题；当前提醒只表示没有附上原设计软件的可编辑装配文件。"
                if intake_status == "warning" and not intake_summary.get("error_count")
                else f"检查结果：{STATUS_LABELS[intake_status]}。"
            ),
            prerequisite="检查期间设计包保持不变。",
            process=["锁定本次设计包", "打开整机几何", "核对文件是否能互相对应", "给出能否继续"],
            inputs=["设计包", "外部设计要求"],
            outputs=["检查结果", "是否可以继续"],
            evidence_ids=[intake_artifact["id"]],
            metrics=[
                {"label": "阻塞问题", "value": str(intake_summary.get("error_count") or 0)},
                {"label": "提醒", "value": str(intake_summary.get("warning_count") or 0)},
                {"label": "可继续", "value": "是" if intake_capabilities.get("can_enter_physical_preflight") else "否"},
            ],
            issues=intake_issues,
            caveats=["资料检查通过后，仍需验证机构真的能动。"],
        ),
        _stage(
            "mechanism-plan",
            4,
            "识别可运动机构",
            "机构识别",
            "从整机里找出哪些零件一起运动、沿什么方向运动，并连接到对应任务。",
            plan_status,
            why="已识别并绑定可运动机构。" if plan else "还没有找到可用的机构说明。",
            prerequisite="仿真前检查允许继续。",
            process=["读取机构说明", "找出固定和运动零件", "确认运动方向", "连接对应任务"],
            inputs=["整机 CAD 与装配说明", "AI 标出的机构范围", "外部任务要求"],
            outputs=["可运动机构清单", "固定/运动零件分组", "运动方向和任务对应关系"],
            evidence_ids=[item["id"] for item in evidence if item["stage_id"] == "mechanism-plan"],
            metrics=[
                {"label": "运动方式", "value": str(plan.get("family") or "—")},
                {"label": "固定组", "value": str(len(plan.get("fixed_group_ids") or []))},
                {"label": "运动组", "value": str(len(plan.get("moving_group_ids") or []))},
            ],
            caveats=["这里只确定机构范围和运动方式，下一阶段才检查碰撞。"],
        ),
        _stage(
            "geometry-preflight",
            5,
            "几何与碰撞预检",
            "碰撞预检",
            "在开始仿真前确认零件没有穿模，导轨与间隙关系正确。",
            geometry_status,
            why="固定件和运动件均已绑定到本次 CAD，运行时碰撞/关节几何证据完整。" if geometry_evidence else "缺少碰撞预检证据。",
            prerequisite="每个零件都必须唯一属于固定组或运动组。",
            process=["检查零件是否穿插", "放大关键间隙", "生成碰撞形状", "记录它们来自哪份 CAD"],
            inputs=["整机 CAD", "机构分组"],
            outputs=["碰撞预检总览", "关键间隙特写", "可用于仿真的碰撞形状"],
            evidence_ids=[item["id"] for item in evidence if item["stage_id"] == "geometry-preflight"],
            metrics=[
                {"label": "固定零件", "value": str(fixed_count or "—")},
                {"label": "运动零件", "value": str(moving_count or "—")},
                {"label": "整体粗略包围盒", "value": "未使用" if geometry_evidence and not provenance.get("convex_hull_used") else "—"},
            ],
            caveats=["绿色通过只覆盖当前 CAD 版本和当前碰撞形状。"],
        ),
        _stage(
            "branch-simulation",
            6,
            "真实物理与理想参考",
            "物理与理想参考",
            f"先完成 {len(load_cases)} 种真实物理测试和实体复审，再由机构方程生成轻量理想目标；只有任务明确要求时才额外运行理想动力学。",
            simulation_status,
            why=f"已完成 {job_pass}/{job_total} 个真实物理任务，并读取 {len(load_cases)} 组理想证据。",
            prerequisite="碰撞预检通过。",
            process=[f"准备 {len(load_cases)} 种测试", "运行真实物理模型", "逐保存姿态复查 STEP 实体", "生成简化动力学分支（方程计算）", "按需运行理想动力学"],
            inputs=["机构分组", f"{len(load_cases)} 种测试条件", "CAD 碰撞形状", "机构族方程"],
            outputs=["真实物理 3D 证据", "零件级接触审计", "简化动力学分支（方程计算）", "按需理想动力学"],
            evidence_ids=[item["id"] for item in evidence if item["stage_id"] == "branch-simulation"],
            metrics=[
                {"label": "物理仿真任务", "value": f"{job_pass}/{job_total}"},
                {"label": "测试工况", "value": str(len(load_cases))},
                {"label": "真实分支内部理想关节", "value": "0"},
            ],
            caveats=["理想参考不提供接触或动态响应证据；真实物理分支负责这些结论。"],
        ),
        _stage(
            "certificate",
            7,
            "物理结果与理想目标认证",
            "运动认证",
            f"逐个检查 {len(load_cases)} 种真实物理测试，并核对确定性理想目标；全部满足适用的传动或接触证据、约束和任务边界后才签发证书。",
            cert_stage_status,
            why=(
                f"{comparison_pass}/{comparison_total} 个必需工况全部通过，已签发运动学证书。"
                if cert_stage_status == "passed"
                else f"仅 {comparison_pass}/{comparison_total} 个工况通过，未签发证书。"
            ),
            prerequisite="每个工况都要有真实物理结果、理想参考和比较结果。",
            process=["锁定同一工况与测试程序", "检查适用的传动或接触证据和物理约束", "比较理想输出偏差", "逐项检查任务门槛", "汇总并签发证书"],
            inputs=["真实物理记录", "简化动力学分支（方程计算）", f"{len(load_cases)} 种测试的通过门槛", "本次设计版本"],
            outputs=(
                [f"{len(load_cases)} 份工况对比", "总判定通过", "运动学证书"]
                if cert_stage_status == "passed"
                else ["工况对比", "未通过原因", "未签发证书"]
            ),
            evidence_ids=[
                item["id"]
                for item in evidence
                if item["stage_id"] == "certificate"
                or (item.get("details") or {}).get("visual_role")
                in {
                    "physics-evidence-scene",
                    "physics-evidence-scene-manifest",
                    "physics-evidence-replay",
                    *(
                        {"reference-comparison-video"}
                        if reference_only_single_mechanism
                        else set()
                    ),
                }
            ],
            metrics=[
                {"label": "工况通过", "value": f"{comparison_pass}/{comparison_total}"},
                {"label": "证书", "value": "已签发" if certificate_status == "passed" else "未签发"},
                {"label": "允许简化替换", "value": "是" if certificate.get("replacement_eligible") else "否"},
            ],
            caveats=["证书只覆盖运动方式，不覆盖强度、寿命或现实安全。"],
        ),
        _stage(
            "replacement",
            8,
            (
                "保留理想参考"
                if reference_only_single_mechanism
                else "启用简化机构"
            ),
            (
                "仅作参考"
                if reference_only_single_mechanism
                else "简化替换"
            ),
            (
                "该包只有一个机构，reduced 只保留为理想参考，不获得整机替代或物理判定权限。"
                if reference_only_single_mechanism
                else "证书有效时，把已验证的内部机构换成更快的简化模型；失效就立即停用。"
            ),
            replacement_stage_status,
            why=(
                "单机构包无需简化整机替代；physical 是唯一物理判定来源。"
                if reference_only_single_mechanism
                else "证书与本次设计版本匹配，简化机构已启用。"
                if replacement_status == "passed"
                else "最终整机任务已通过；旧 worker 后来覆盖了同名中间文件，本关保留 PASS 权威并显示文件审核提示。"
                if legacy_replacement_checkpoint_overwrite
                else "证书缺失、失效或与本次设计不匹配，简化机构未启用。"
            ),
            prerequisite=(
                "真实物理结果和任务证书通过。"
                if reference_only_single_mechanism
                else "运动学证书通过且仍与本次设计匹配。"
            ),
            process=(
                ["保留 reduced 对照", "禁止它签发物理结论", "不生成整机替代任务"]
                if reference_only_single_mechanism
                else ["确认本次证书仍有效", "加载简化机构", "保留失效与回退规则"]
            ),
            inputs=(
                ["已通过的运动证书", "本次设计版本"]
                if replacement_effectively_active
                else ["当前未具备：已通过的运动证书", "本次设计版本"]
            ),
            outputs=(
                (
                    ["历史已启用的简化机构", "旧文件覆盖审核提示"]
                    if legacy_replacement_checkpoint_overwrite
                    else ["已启用的简化机构", "失效与回退规则"]
                )
                if replacement_effectively_active
                else ["未启用结论", "需要重新验证的原因"]
            ),
            evidence_ids=[
                replacement_artifact["id"],
                certificate_artifact["id"],
                *[
                    item["id"]
                    for item in evidence
                    if (item.get("details") or {}).get("visual_role") in {
                        "physical-video",
                        "reduced-video",
                        "reference-comparison-video",
                    }
                ],
            ],
            metrics=[
                {
                    "label": "简化机构",
                    "value": (
                        "历史已启用 · 文件提示"
                        if legacy_replacement_checkpoint_overwrite
                        else "已启用"
                        if replacement_status == "passed"
                        else "未启用"
                    ),
                },
                {
                    "label": "运动方式",
                    "value": str(
                        (replacement.get("replacement_model") or {}).get("family")
                        or plan.get("family")
                        or (certificate.get("mechanism") or {}).get("family")
                        or "—"
                    ),
                },
                {"label": "覆盖工况", "value": str(len((replacement.get("validity_envelope") or {}).get("load_case_ids") or []))},
            ],
            issues=[*replacement_stale_reasons, *replacement_audit_warnings],
            caveats=[
                (
                    "reduced 只能帮助理解理想运动，不能覆盖 physical 的失败。"
                    if reference_only_single_mechanism
                    else "设计版本或证书变化后会自动失效并回退。"
                )
            ],
        ),
        _stage(
            "whole-machine",
            9,
            f"整机任务（{len(whole_task_rows)} 项）",
            "整机任务",
            (
                f"把已验证的机构放回整机，逐项完成已登记的 {len(whole_task_rows)} 个外部任务。"
            ),
            whole_stage_status,
            why=(
                f"{sum(item['status'] == 'passed' for item in whole_task_rows)}/{len(whole_task_rows)} 个整机任务到达目标。"
                if whole_task_rows and whole_status == "passed"
                else "当前运行没有登记完整通过的整机任务结果。"
            ),
            prerequisite="简化机构已启用，所有任务目标均位于已验证包络内。",
            process=[
                "加载整机与已验证机构",
                *[
                    f"执行：{label}"
                    for label in whole_task_labels
                ],
                "汇总每项任务的目标、实测值与门槛",
            ],
            inputs=[
                (
                    "已启用的简化机构"
                    if replacement_effectively_active
                    else "当前未具备：已启用的简化机构"
                ),
                f"{len(whole_task_rows)} 项已登记整机任务",
            ],
            outputs=(
                [
                    "整机任务回放",
                    f"{len(whole_task_rows)} 项任务逐项结果",
                    "任务约束与输入检查",
                ]
                if whole_status == "passed"
                else ["尚无通过的整机任务结果"]
            ),
            evidence_ids=[
                item["id"]
                for item in evidence
                if item["stage_id"] == "whole-machine"
            ] + [replacement_artifact["id"], certificate_artifact["id"]],
            metrics=[
                {"label": "任务通过", "value": f"{sum(item['status'] == 'passed' for item in whole_task_rows)}/{len(whole_task_rows)}"},
                {"label": "横向偏移", "value": f"{_round((_finite(whole.get('max_lateral_shift_m')) or 0.0) * 1000, 4)} mm" if whole else "—"},
                {"label": "最大转动", "value": f"{_round(math.degrees(_finite(whole.get('max_rotation_rad')) or 0.0), 4)}°" if whole else "—"},
            ],
            issues=[*whole_stale_reasons, *whole_audit_warnings],
            caveats=[
                "没有生成 reduced 整机回放；请查看第 5、6 关的 physical 轨迹与任务判定。"
                if reference_only_single_mechanism
                else "本阶段展示的是已验证简化机构的整机任务回放，不是整机高保真接触认证。"
                if whole_status == "passed"
                else "没有登记的整机结果时，页面必须显示未完成。"
            ],
        ),
    ]

    branch_stage = next(stage for stage in stages if stage["id"] == "branch-simulation")
    branch_stage.update(
        {
            "title": "\u771f\u5b9e\u7269\u7406\u4e0e\u7406\u60f3\u53c2\u8003",
            "short_title": "\u7269\u7406\u4e0e\u7406\u60f3\u53c2\u8003",
            "goal": "\u5148\u8fd0\u884c\u771f\u5b9e\u6a21\u578b\uff0c\u518d\u7528\u539f\u59cb STEP \u68c0\u67e5\u5df2\u4fdd\u5b58\u7684 3D \u65f6\u95f4\u70b9\uff1b\u901a\u8fc7\u540e\u751f\u6210\u8f7b\u91cf\u7684\u786e\u5b9a\u6027\u7406\u60f3\u53c2\u8003\uff0c\u53ea\u5728\u4efb\u52a1\u660e\u786e\u9700\u8981\u65f6\u8fd0\u884c\u7406\u60f3\u52a8\u529b\u5b66\u3002",
            "why": (
                f"\u771f\u5b9e\u5b9e\u4f53\u590d\u5ba1\u6709 {physical_audit_failed_count}/{physical_audit_trace_count} \u4e2a\u5de5\u51b5\u672a\u901a\u8fc7\uff1b\u7406\u60f3\u53c2\u8003\u4e0d\u5f97\u7528\u6765\u8986\u76d6\u8fd9\u4e2a\u5931\u8d25\u3002"
                if physical_preflight and physical_audit_status != "passed"
                else f"\u65e7\u8fd0\u884c\u8ffd\u52a0\u590d\u5ba1\u6709 {physical_audit_failed_count}/{physical_audit_trace_count} \u4e2a\u5de5\u51b5\u672a\u901a\u8fc7\uff1b\u65e7\u8bc1\u4e66\u4fdd\u6301\u539f\u6837\uff0c\u65b0\u770b\u677f\u4e0d\u518d\u5c06\u5b83\u5f53\u6210\u5f53\u524d\u5b9e\u4f53\u5408\u6cd5\u6027\u901a\u8fc7\u3002"
                if legacy_physical_audit and physical_audit_status != "passed"
                else f"\u5df2\u590d\u5ba1 {physical_audit_passed_count}/{physical_audit_trace_count} \u4e2a\u771f\u5b9e\u5de5\u51b5\uff0c\u5bf9\u7167\u4fa7\u4f7f\u7528{comparison_ideal_label}\u3002"
                if active_physical_audit
                else f"\u5df2\u5b8c\u6210 {job_pass}/{job_total} \u4e2a\u771f\u5b9e\u7269\u7406\u4efb\u52a1\uff0c\u5e76\u8bfb\u53d6 {len(load_cases)} \u7ec4{comparison_ideal_label}\u3002"
            ),
            "prerequisite": "\u78b0\u649e\u9884\u68c0\u901a\u8fc7\uff0c\u771f\u5b9e\u5206\u652f\u4e0d\u5f97\u4f7f\u7528\u6574\u5bf9\u521a\u4f53\u7981\u78b0\u3002",
            "process": [
                "\u5148\u8fd0\u884c\u5168\u90e8\u771f\u5b9e\u7269\u7406\u5de5\u51b5",
                "\u7528\u539f\u59cb STEP \u68c0\u67e5\u6bcf\u4e2a\u5df2\u4fdd\u5b58\u65f6\u95f4\u70b9",
                "\u53ea\u6709\u5168\u90e8\u5408\u6cd5\u624d\u751f\u6210\u786e\u5b9a\u6027\u7406\u60f3\u53c2\u8003",
                "\u4ec5\u5728\u7c7b\u578b\u5316\u4efb\u52a1\u660e\u786e\u8981\u6c42\u65f6\u8fd0\u884c\u7406\u60f3\u52a8\u529b\u5b66",
            ],
            "outputs": (
                ["\u771f\u5b9e\u8f68\u8ff9", "\u7a7f\u6a21\u4f4d\u7f6e\u4e0e\u6df1\u5ea6", "\u4e0d\u5f97\u7b7e\u53d1\u65b0\u8bc1\u4e66\u7684\u8bb0\u5f55"]
                if physical_preflight and physical_audit_status != "passed"
                else ["\u771f\u5b9e\u7269\u7406 3D \u8bc1\u636e", "\u96f6\u4ef6\u7ea7\u63a5\u89e6\u5ba1\u8ba1", "\u786e\u5b9a\u6027\u7406\u60f3\u53c2\u8003"]
            ),
            "metrics": [
                {"label": "\u771f\u5b9e\u5de5\u51b5\u590d\u5ba1", "value": f"{physical_audit_passed_count}/{physical_audit_trace_count}" if active_physical_audit else "\u672a\u8bb0\u5f55"},
                {"label": "\u5df2\u68c0\u67e5 3D \u65f6\u95f4\u70b9", "value": str(physical_audit_selected_pose_count) if active_physical_audit else "\u672a\u8bb0\u5f55"},
                {"label": "\u6700\u5927\u68c0\u51fa\u7a7f\u5165", "value": f"{physical_audit_max_penetration_m * 1000.0:.3f} mm" if active_physical_audit else "\u672a\u8bb0\u5f55"},
                {"label": "\u5bf9\u7167\u4fa7", "value": comparison_ideal_label},
            ],
            "caveats": ["\u5b9e\u4f53\u590d\u5ba1\u68c0\u67e5\u7ea6 0.05 s \u95f4\u9694\u7684\u5df2\u4fdd\u5b58\u65f6\u95f4\u70b9\uff0c\u5e76\u989d\u5916\u4fdd\u7559\u9636\u6bb5\u8fb9\u754c\u548c\u63a5\u89e6\u53d8\u5316\uff1b\u786e\u5b9a\u6027\u7406\u60f3\u53c2\u8003\u4e0d\u627f\u62c5\u63a5\u89e6\u6216\u52a8\u529b\u5b66\u8bc1\u660e\u3002"],
        }
    )
    branch_stage["status_label"] = STATUS_LABELS[branch_stage["status"]]

    raw_run_status = str(run_status.get("status") or "")
    raw_run_status_upper = raw_run_status.strip().upper()
    if requested_is_react:
        dashboard_publication = None
        publication_reason = "NOT_APPLICABLE_REACT_LOOP"
    else:
        from .publication import load_bound_dashboard_publication

        dashboard_publication, publication_reason = load_bound_dashboard_publication(
            workspace_root=workspace,
            run_dir=requested_run,
            terminal_status=run_status,
            require_inventory=True,
        )
    if dashboard_publication is not None:
        stages = _materialize_published_stages(
            publication=dashboard_publication,
            base_stages=stages,
            evidence=evidence,
            ctx=ctx,
        )
        stage_aggregation_reasons = []
        diagnostics[:] = [
            item
            for item in diagnostics
            if item.get("role") != "dynamic stage aggregation"
        ]
    else:
        # Legacy runs predate formal publication.  Keep their evidence
        # readable, but do not pretend their kind-level inference can express
        # multiple equal-kind capsule instances.
        workflow_manifest = run_status.get("workflow_stage_manifest") or []
        manifest_by_kind = {
            str(item.get("kind") or item.get("id") or ""): item
            for item in workflow_manifest
            if isinstance(item, dict)
        }
        has_task_planning_evidence = task_planning_exists and not (
            requested_is_canonical_terminal
        )
        has_react_loop_evidence = bool(react_loop_state) and not (
            requested_is_canonical_terminal
        )
        has_intake_evidence = bool(intake) or bool(
            intake_path and intake_path.is_file()
        )
        has_plan_evidence = bool(plan)
        has_geometry_evidence = bool(geometry_evidence)
        has_simulation_evidence = bool(
            load_cases
            or queue_statuses
            or physical_preflight
            or legacy_physical_audit
        )
        has_certificate_evidence = bool(certificate) or any(
            case.get("status") != "not_started" for case in load_cases
        )
        has_replacement_evidence = bool(replacement) or bool(
            replacement_path and replacement_path.is_file()
        )
        has_whole_machine_evidence = bool(whole) or bool(whole_plan) or bool(
            whole_results
        )

        applicable_kinds = {"design-package"}
        if has_task_planning_evidence or has_react_loop_evidence:
            applicable_kinds.add("task-planning")
        if (
            has_intake_evidence
            or has_plan_evidence
            or has_geometry_evidence
            or has_simulation_evidence
            or has_certificate_evidence
            or has_replacement_evidence
            or has_whole_machine_evidence
        ):
            applicable_kinds.add("intake")
        if (
            has_plan_evidence
            or has_geometry_evidence
            or has_simulation_evidence
            or has_certificate_evidence
            or has_replacement_evidence
            or has_whole_machine_evidence
        ):
            applicable_kinds.add("mechanism-plan")
        if (
            has_geometry_evidence
            or has_simulation_evidence
            or has_certificate_evidence
            or has_replacement_evidence
            or has_whole_machine_evidence
        ):
            applicable_kinds.add("geometry-preflight")
        if (
            has_simulation_evidence
            or has_certificate_evidence
            or has_replacement_evidence
            or has_whole_machine_evidence
        ):
            applicable_kinds.add("branch-simulation")
        if (
            has_certificate_evidence
            or has_replacement_evidence
            or has_whole_machine_evidence
            or raw_run_status_upper in FINAL_PASS_STATUSES
        ):
            applicable_kinds.add("certificate")
        if (
            not reference_only_single_mechanism
            and (has_replacement_evidence or has_whole_machine_evidence)
        ):
            applicable_kinds.add("replacement")
        if (
            has_whole_machine_evidence
            or raw_run_status_upper == "WHOLE_MACHINE_TASK_PASSED"
        ):
            applicable_kinds.add("whole-machine")
        applicable_kinds.update(
            kind
            for kind in manifest_by_kind
            if kind in {stage["kind"] for stage in stages}
        )

        filtered_stages: list[dict[str, Any]] = []
        for item in stages:
            if item["kind"] not in applicable_kinds:
                continue
            manifest_item = manifest_by_kind.get(item["kind"]) or {}
            item["order"] = len(filtered_stages) + 1
            # 2026-08-08 DEFECT #4 fix: branch-simulation/certificate stages span
            # every capsule's load cases; pinning scope to the primary capsule
            # made the frontend intersect filter hide all non-primary loads.
            stage_load_capsules = {
                str(case.get("capsule_id") or "") for case in load_cases
            }
            item["scope"] = {
                "capsule_id": (
                    ""
                    if item["kind"]
                    in {
                        "task-planning",
                        "design-package",
                        "intake",
                        "whole-machine",
                    }
                    or (
                        item["kind"] in {"branch-simulation", "certificate"}
                        and len(stage_load_capsules) > 1
                    )
                    else capsule_id
                ),
                "load_case_ids": (
                    [str(case.get("id") or "") for case in load_cases]
                    if item["kind"] in {"branch-simulation", "certificate"}
                    else []
                ),
                "task_ids": (
                    [str(task.get("id") or "") for task in whole_task_rows]
                    if item["kind"] == "whole-machine"
                    else []
                ),
                "result_id": (
                    ctx.relative(whole_path)
                    if item["kind"] == "whole-machine"
                    else ""
                ),
            }
            item["required"] = bool(manifest_item.get("required", True))
            item["applicability"] = (
                "declared"
                if manifest_item
                else "inferred_from_registered_artifacts"
            )
            filtered_stages.append(item)
        stages = filtered_stages

    complete_count = sum(
        stage["status"] in {"passed", "warning", "evidence_complete"}
        for stage in stages
        if stage.get("required", True)
    )
    required_stage_count = sum(stage.get("required", True) for stage in stages)
    first_attention = next(
        (
            stage["id"]
            for stage in stages
            if stage.get("required", True)
            if stage["status"] not in {"passed", "warning", "evidence_complete"}
        ),
        stages[-1]["id"] if stages else "",
    )
    verdict = _resolve_run_verdict(raw_run_status)
    overall_status = str(verdict["status"])
    registered_by_capsule = {
        str(item.get("capsule_id") or ""): item
        for item in registered_mechanisms
        if str(item.get("capsule_id") or "")
    }
    mechanisms: list[dict[str, Any]] = []
    mechanism_plans = plans or ([plan] if plan else [])
    for mechanism_plan in mechanism_plans:
        mechanism_capsule = str(
            mechanism_plan.get("capsule_id") or capsule_id
        )
        registration = registered_by_capsule.get(mechanism_capsule) or {}

        def registered_document(
            key: str,
            fallback: Path,
        ) -> dict[str, Any]:
            reference = registration.get(key)
            path: Path | None = None
            if reference:
                try:
                    path = ctx.resolve(reference)
                except ValueError:
                    path = None
            if path is None:
                path = fallback
            return (
                ctx.read_json(
                    path,
                    role=f"{mechanism_capsule} {key}",
                )
                if path.is_file()
                else {}
            )

        mechanism_certificate = registered_document(
            "certificate",
            run / "results" / mechanism_capsule / "certificate.json",
        )
        mechanism_replacement = registered_document(
            "replacement",
            run / "results" / mechanism_capsule / "replacement.json",
        )
        mechanism_envelope = (
            mechanism_certificate.get("tested_envelope")
            or mechanism_replacement.get("validity_envelope")
            or {}
        )
        mechanisms.append(
            {
                "id": mechanism_capsule,
                "family": str(
                    mechanism_plan.get("family")
                    or (
                        mechanism_certificate.get("mechanism") or {}
                    ).get("family")
                    or "unknown"
                ),
                "axis": (
                    mechanism_plan.get("axis_abs")
                    or (
                        mechanism_certificate.get("mechanism") or {}
                    ).get("axis_abs")
                    or []
                ),
                "origin_m": (
                    mechanism_plan.get("joint_origin_abs_m")
                    or (
                        mechanism_certificate.get("mechanism") or {}
                    ).get("joint_origin_abs_m")
                    or []
                ),
                "fixed_groups": (
                    mechanism_plan.get("fixed_group_ids") or []
                ),
                "moving_groups": (
                    mechanism_plan.get("moving_group_ids") or []
                ),
                "task_ids": (
                    mechanism_plan.get("task_ids")
                    or registration.get("task_ids")
                    or whole.get("task_ids")
                    or []
                ),
                "certificate_verdict": str(
                    mechanism_certificate.get("overall_verdict")
                    or mechanism_certificate.get("verdict")
                    or "NOT_AVAILABLE"
                ),
                "replacement_status": str(
                    mechanism_replacement.get("status")
                    or "NOT_AVAILABLE"
                ),
                "envelope": mechanism_envelope,
            }
        )
    primary_warning = intake_issue_details[0] if intake_issue_details else {}
    whole_task_passed = sum(item["status"] == "passed" for item in whole_task_rows)
    multiple_whole_machine_results = len(whole_machine_views) > 1
    whole_machine_compatibility_view = {
        **primary_whole_view,
        "compatibility_view": True,
        "canonical_field": "whole_machines",
        "aggregation_kind": (
            "multi_capsule_summary"
            if multiple_whole_machine_results
            else "single_capsule"
        ),
        "status": whole_stage_status,
        "stages": whole_task_rows,
        "task_passed": whole_task_passed,
        "task_total": len(whole_task_rows),
    }
    if multiple_whole_machine_results:
        whole_machine_compatibility_view.update(
            {
                "capsule_id": "",
                "status_raw": "MULTIPLE_REGISTERED_RESULTS",
                "required": any(
                    bool(item.get("required"))
                    for item in whole_machine_views
                ),
                "replacement_policy": "per_capsule",
                "task_ids": [
                    str(item.get("id") or "")
                    for item in whole_task_rows
                ],
                "coordinate": {
                    "kind": "multi_capsule",
                    "name": "多机构任务",
                    "unit": "",
                },
                "max_lateral_shift_mm": None,
                "max_rotation_deg": None,
                "max_pivot_drift_mm": None,
                "max_off_axis_rotation_deg": None,
                "position_tolerance_mm": None,
                "angle_tolerance_deg": None,
                "max_stage_duration_s": None,
                "max_control_force_n": None,
                "max_control_torque_nm": None,
                "sample_count": None,
                "strict_non_claims": {},
                "boundary": (
                    "兼容汇总视图；逐机构权威结果见 whole_machines。"
                ),
            }
        )
    package_id = str(
        (certificate.get("package") or {}).get("package_id")
        or package_manifest.get("package_id")
        or (run_identity.project_id if run_identity is not None else None)
        or live_project["live_workflow"].get("project_id")
        or run.parent.name
    )
    project_usage = load_project_usage(ctx, package_id, run.name)
    overview = {
        "package": {
            "id": package_id,
            "version": str((certificate.get("package") or {}).get("design_version") or package_manifest.get("design_version") or "—"),
        },
        "warning": {
            "count": int(intake_summary.get("warning_count") or 0),
            "title": primary_warning.get("title") or "无 Intake 警告",
            "summary": primary_warning.get("summary") or "当前 Intake 未报告警告。",
            "impact": primary_warning.get("impact") or "",
            "non_blocking_reason": primary_warning.get("non_blocking_reason") or "",
        },
        "certificate": {
            "status": certificate_status,
            "level": "运动学认证" if certificate.get("certificate_level") == "KINEMATIC_CERTIFIED" else str(certificate.get("certificate_level") or "未认证"),
            "boundary": (
                "方向、约束、接触与任务行程；不含动力学、摩擦效率、强度或现实安全。"
                if certificate_status == "passed"
                else "证书未签发；当前只展示已执行测试和失败门控。"
            ),
        },
        "replacement": {
            "status": replacement_stage_status,
            "status_raw": str(replacement.get("status") or "NOT_AVAILABLE"),
            "hash_bound": replacement_status == "passed" and not replacement_stale_reasons,
            "enabled": replacement_effectively_active,
            "policy": replacement_policy,
            "reference_only": reference_only_single_mechanism,
            "historical_terminal_authority": legacy_replacement_checkpoint_overwrite,
            "audit_warnings": replacement_audit_warnings,
        },
        "whole_machine": {
            "status": whole_stage_status,
            "required": not reference_only_single_mechanism,
            "passed": whole_task_passed,
            "total": len(whole_task_rows),
        },
    }
    caveats = [
        "Intake 接受只代表包可进入物理预检。",
        "证书仅覆盖记录的几何哈希、质量、载荷、重力、观测量和运动区间。",
        "KINEMATIC_CERTIFIED 不代表动力学、摩擦效率、强度、疲劳、制造或现实安全认证。",
        (
            "单机构包不运行 reduced 整机替代；整机任务结论直接来自 physical。"
            if reference_only_single_mechanism
            else "整机任务使用简化 replacement，不是高保真整机接触运行。"
        ),
    ]
    if legacy_replacement_checkpoint_overwrite:
        caveats.append(
            "本运行的最终整机 PASS 保持权威；旧 worker 覆盖同名 replacement 中间文件的情况仅作为历史文件审核提示。"
        )

    requested_identity = parse_canonical_run(workspace, requested_run)
    recorded_verdict = verdict
    if requested_is_react:
        recorded_verdict = _react_verdict(
            react_loop_state,
            recorded_verdict,
        )
        effective_state = _react_effective_state(recorded_verdict)
        requested_terminal_ready = bool(effective_state["terminal_ready"])
        requested_terminal_reason = str(effective_state["terminal_reason"])
        governance_ledger = {
            "valid": True,
            "reason_code": "NOT_APPLICABLE_REACT_LOOP",
            "event_count": 0,
            "head_sha256": "",
            "events": [],
        }
    else:
        requested_terminal_ready = _legacy_terminal_ready(
            run_status.get("status")
        )
        governance_ledger = dict(_LEGACY_GOVERNANCE_LEDGER)
        effective_state = _legacy_effective_state(
            recorded_verdict,
            recorded_status_raw=raw_run_status,
            terminal_ready=requested_terminal_ready,
        )
        requested_terminal_reason = str(effective_state["terminal_reason"])
    if live_project["live_workflow"] and not run_status:
        live_workflow = live_project["live_workflow"]
        try:
            live_current = bool(requested_identity and requested_identity.revision == f"r{int(live_workflow.get('revision')):02d}")
        except (TypeError, ValueError):
            live_current = False
        live_running = bool(live_current and live_workflow.get("is_running"))
        live_status = "running" if live_running else "blocked" if live_current and live_workflow.get("stop_reason") else "preparing"
        effective_state.update({
            "effective_status": live_status,
            "effective_status_raw": "COLD_START_UNSUBMITTED",
            "effective_final": False,
            "eligible_pass": False,
            "healthy_running": live_running,
            "reason_code": "COLD_START_UNSUBMITTED",
        })
        requested_terminal_reason = "not_submitted"
    if live_project["live_workflow"].get("workflow_kind") == "slim_harness":
        raw = slim_result.get("status") or "SLIM_HARNESS_PENDING"
        passed = bool(slim_result.get("eligible_pass"))
        ended_unpassed = live_project["live_workflow"].get("outcome") in ENDED_WITHOUT_PASS
        semantic = "passed" if passed else "failed" if raw == "NEEDS_REVISION" or ended_unpassed else "blocked" if raw in {"HARNESS_ERROR","EVIDENCE_REQUIRED"} else effective_state["effective_status"]
        effective_state.update(effective_status=semantic,effective_status_raw=raw,effective_final=passed,
                               eligible_pass=passed,reason_code=raw,healthy_running=bool(live_project["live_workflow"].get("is_running")))
        requested_terminal_ready=passed
        requested_terminal_reason="slim_original_geometry_and_review" if passed else raw
    overall_status = str(effective_state["effective_status"])
    verdict = {
        "status": overall_status,
        "final": bool(effective_state["effective_final"]),
        "authority": (
            recorded_verdict["authority"]
            if overall_status == recorded_verdict["status"]
            else "dashboard_governance"
        ),
        "reason_code": str(effective_state["reason_code"]),
    }
    evidence_conflicts = [
        *("REPLACEMENT_BINDING_STALE" for _ in replacement_stale_reasons),
        *("WHOLE_MACHINE_BINDING_STALE" for _ in whole_stale_reasons),
    ]
    if live_project["live_workflow"].get("workflow_kind") == "slim_harness":
        verdict["authority"]="slim_harness"
    if recorded_verdict["status"] == "passed" and any(
        item.get("required", True)
        and item["status"]
        not in {"passed", "warning", "evidence_complete"}
        for item in stages
    ):
        evidence_conflicts.append("TERMINAL_STAGE_CONFLICT")
    if (
        recorded_verdict["status"] == "passed"
        and requested_identity is not None
        and not requested_terminal_ready
    ):
        evidence_conflicts.append(
            f"TERMINAL_INVENTORY_NOT_READY:{requested_terminal_reason}"
        )
    if recorded_verdict["status"] == "passed" and stage_aggregation_reasons:
        evidence_conflicts.append("STAGE_AGGREGATION_PARTIAL")
    run_evidence_health = _evidence_health(
        verdict=verdict,
        canonical=requested_identity is not None,
        terminal_ready=requested_terminal_ready,
        physical_audit=active_physical_audit,
        conflict_reasons=evidence_conflicts,
    )
    stage_summary_status = _combine_status(
        [
            item["status"]
            for item in stages
            if item.get("required", True)
        ]
    )
    return {
        "schema_version": "0.1",
        "generated_at": _now(),
        "mode": mode,
        **live_project,
        "slim_result": slim_result,
        "run": {
            "id": ctx.relative(requested_run),
            "name": requested_run.name,
            "project_id": (
                requested_identity.project_id if requested_identity else ""
            ),
            "revision": (
                requested_identity.revision if requested_identity else ""
            ),
            "logical_run_id": (
                requested_identity.run_id if requested_identity else requested_run.name
            ),
            "display_name": _run_display_name(
                requested_run,
                f"{overview['package']['id']} v{overview['package']['version']} · {requested_run.name}",
            ),
            "path": ctx.relative(requested_run),
            "status_raw": str(
                react_loop_state.get("status")
                or (slim_result.get("status") or "SLIM_HARNESS_PENDING" if live_project["live_workflow"].get("workflow_kind") == "slim_harness" else "")
                or run_status.get("status")
                or ("COLD_START_UNSUBMITTED" if live_project["live_workflow"] else "")
                or "NOT_AVAILABLE"
            ),
            # Exact repeated-error pause record (reason, offending stage,
            # last error text, resume stage) so the board shows the real
            # cause instead of only a generic paused label.
            "pause": (
                dict(react_loop_state["pause"])
                if isinstance(react_loop_state.get("pause"), dict)
                else None
            ),
            "harness_defect": _harness_defect_banner(
                react_loop_state, run_status
            ),
            "status": overall_status,
            "status_label": STATUS_LABELS[overall_status],
            "verdict": verdict,
            "recorded_status": str(recorded_verdict["status"]),
            "recorded_status_raw": (
                str(react_loop_state.get("status") or "")
                or raw_run_status
                or "NOT_AVAILABLE"
            ),
            "recorded_verdict": recorded_verdict,
            "effective_status": overall_status,
            "effective_status_raw": str(
                effective_state["effective_status_raw"]
            ),
            "terminal_ready": requested_terminal_ready,
            "terminal_reason": requested_terminal_reason,
            "eligible_pass": bool(effective_state["eligible_pass"]),
            "healthy_running": bool(effective_state["healthy_running"]),
            "running_lease": effective_state["lease"],
            "iteration_outcome": effective_state["iteration"],
            "governance": effective_state["governance"],
            "evidence_health": run_evidence_health,
            "stage_summary_status": stage_summary_status,
            "last_completed_stage": str(run_status.get("last_completed_stage") or ""),
            "default_stage_id": first_attention,
            "progress": {"completed": complete_count, "total": required_stage_count},
            "jobs": {"completed": job_pass, "total": job_total},
            "comparisons": {"passed": comparison_pass, "total": comparison_total},
            "updated_at": datetime.fromtimestamp(
                (
                    run_status_path
                    if run_status_path.exists()
                    else react_loop_state_path
                ).stat().st_mtime,
                timezone.utc,
            ).isoformat()
            if (
                run_status_path.exists()
                or react_loop_state_path is not None
                and react_loop_state_path.exists()
            )
            else "",
        },
        "overview": overview,
        "task_planning": {
            "joint_plan": joint_plan,
            "test_plan": test_plan,
            "execution_coverage": task_execution_coverage,
        },
        "project_usage": project_usage,
        "design_quality": {
            "status": design_quality_status if design_quality_recorded else "warning",
            "recorded": design_quality_recorded,
            "original_prompt": str(design_input.get("original_prompt") or ""),
            "plain_language_goal": str(design_input.get("plain_language_goal") or ""),
            "engineering_brief": design_input.get("engineering_brief") or {},
            "visual_acceptance_checklist": design_input.get("visual_acceptance_checklist") or [],
            "generator": design_input.get("generator") or {},
            "review_round_count": len(visual_review.get("rounds") or []),
            "reported_findings": design_reported_findings,
            "open_findings": design_evaluation.get("open_findings") or {"P0": 0, "P1": 0, "P2": 0},
            "issues": design_blocking_issues,
            "supplemental_issues": design_supplemental_issues,
            "authority": design_evaluation.get("authority") or {
                "scope": "design_input_package_completeness",
                "certificate_authority": False,
                "submitter_review_status_used": False,
                "submitter_finding_status_used": False,
            },
            "assembly_step_sha256": str(design_evaluation.get("assembly_step_sha256") or ""),
            "render_asset_count": len(design_render_artifacts),
        },
        "static_visual_review": static_visual_review,
        "reuse": reuse_summary,
        "physical_audit": {
            "recorded": bool(active_physical_audit),
            "source": "legacy_saved_trace_migration" if legacy_physical_audit else "current_physical_preflight" if physical_preflight else "not_recorded",
            "status": physical_audit_status,
            "trace_count": physical_audit_trace_count,
            "passed_trace_count": physical_audit_passed_count,
            "failed_trace_count": physical_audit_failed_count,
            "selected_pose_count": physical_audit_selected_pose_count,
            "max_penetration_mm": _round(physical_audit_max_penetration_m * 1000.0, 4),
            "reduced_branch_started": reduced_branch_started,
            "historical_certificate_modified": legacy_physical_audit.get("policy", {}).get("historical_certificate_modified") if legacy_physical_audit else None,
            "component_contact": component_contact_summary,
        },
        "workflow": {
            "stage_model": (
                "publication_instances_v1"
                if dashboard_publication is not None
                else "legacy_kind_inference_v1"
            ),
            "publication": {
                "bound": dashboard_publication is not None,
                "reason_code": publication_reason,
                "schema_version": str(
                    (dashboard_publication or {}).get("schema_version") or ""
                ),
                "stage_manifest_sha256": str(
                    (dashboard_publication or {}).get(
                        "stage_manifest_sha256"
                    )
                    or ""
                ),
            },
            "governance": {
                "ledger_valid": bool(governance_ledger["valid"]),
                "reason_code": str(governance_ledger["reason_code"]),
                "event_count": int(governance_ledger["event_count"]),
                "head_sha256": str(governance_ledger["head_sha256"]),
            },
            "aggregation_status": (
                "partial" if stage_aggregation_reasons else "complete"
            ),
            "reason_codes": stage_aggregation_reasons,
            "scope_counts": {
                "mechanism_plans": len(plans),
                "registered_mechanisms": len(registered_mechanisms),
                "whole_machine_results": len(registered_whole_results),
            },
        },
        "stages": stages,
        "mechanisms": mechanisms,
        "load_cases": load_cases,
        "whole_machine": whole_machine_compatibility_view,
        "whole_machines": whole_machine_views,
        "evidence": evidence,
        "caveats": caveats,
        "diagnostics": diagnostics,
    }


def scan_runs(
    runs_root: str | Path,
    *,
    workspace_root: str | Path,
    limit: int | None = None,
    include_history: bool = False,
    include_active: bool = False,
) -> list[dict[str, Any]]:
    root = Path(runs_root).resolve()
    workspace = Path(workspace_root).resolve()
    try:
        root.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("runs_root must stay inside workspace_root") from exc

    # When the live server is pointed at the workspace, scan only explicit
    # runtime artifact roots.  Never recurse through unrelated caches (for
    # example a damaged WSL virtual disk under runtime_cache).
    search_roots = [root]
    if root == workspace:
        search_roots = [workspace / ".runtime_runs"]
        search_roots.extend(
            path for path in workspace.glob("*/.runtime_runs") if path.is_dir()
        )

    project_aliases: dict[str, str] = {}
    canonical_project_ids: list[str] = []
    react_states_by_run: dict[Path, dict[str, Any]] = {}
    for project_manifest in sorted(
        (workspace / ".runtime_runs" / "projects").glob("*/project.json")
    ):
        try:
            project_value, _record = read_stable_json_object(
                project_manifest,
                label="dashboard canonical project manifest",
                allowed_root=workspace,
            )
        except (OSError, OperatorAuthorityError):
            continue
        project_id = str(project_value.get("project_id") or "").strip()
        if not project_id:
            continue
        canonical_project_ids.append(project_id)
        project_aliases[project_id] = project_id
        package_id = str(project_value.get("package_id") or "").strip()
        if package_id:
            project_aliases[package_id] = project_id
        react_states_by_run.update(
            _react_project_revision_states(
                workspace,
                project_manifest.parent,
            )
        )

    def legacy_project_id(run_dir: Path, package_label: str) -> str:
        if package_label in project_aliases:
            return project_aliases[package_label]
        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "-",
            run_dir.name.lower(),
        ).strip("-")
        candidates: list[tuple[int, str]] = []
        for project_id in canonical_project_ids:
            project_stem = re.sub(r"_v\d+$", "", project_id.lower())
            normalized_stem = re.sub(
                r"[^a-z0-9]+",
                "-",
                project_stem,
            ).strip("-")
            if normalized_stem and normalized_stem in normalized_name:
                candidates.append((len(normalized_stem), project_id))
        if candidates:
            return max(candidates)[1]
        return package_label

    status_paths: set[Path] = set()
    ignored_runtime_trees = {
        "abandoned_candidate_release_regression",
        "candidate_release_regression",
        "dashboard_previews",
        "hf_sync",
        "quarantine",
    }
    for search_root in search_roots:
        if search_root.is_dir():
            for directory, child_names, file_names in os.walk(
                search_root,
                followlinks=False,
            ):
                directory_path = Path(directory)
                child_names[:] = [
                    name
                    for name in child_names
                    if (
                        name not in ignored_runtime_trees
                        and not (directory_path / name).is_symlink()
                    )
                ]
                if "run_status.json" in file_names:
                    status_paths.add(
                        directory_path / "run_status.json"
                    )

    # Every ReAct revision remains visible, including an intake-failed
    # historical revision that never produced run_status.json.  The current
    # revision comes from the hash-bound loop state; historical rows are
    # projected from their own Harness outcomes.
    for react_run in react_states_by_run:
        if any(
            react_run == search_root or react_run.is_relative_to(search_root)
            for search_root in search_roots
        ):
            status_paths.add(react_run / "run_status.json")

    rows: list[dict[str, Any]] = []
    for status_path in sorted(status_paths):
        run_dir = status_path.parent
        react_state = react_states_by_run.get(run_dir, {})
        react_run = is_react_loop_run(workspace, run_dir)
        try:
            value, _record = read_stable_json_object(
                status_path,
                label="dashboard run status",
                allowed_root=workspace,
            )
        except FileNotFoundError:
            value = {} if react_run else {
                "status": "RUN_STATUS_DOCUMENT_FAILED_VALIDATION",
            }
        except (OSError, OperatorAuthorityError):
            value = {
                "status": "RUN_STATUS_DOCUMENT_FAILED_VALIDATION",
            }
        canonical = parse_canonical_run(workspace, run_dir)
        if react_run:
            recorded_verdict = _resolve_run_verdict(value.get("status"))
            direct_verdict = _react_verdict(react_state, recorded_verdict)
            direct_state = _react_effective_state(direct_verdict)
            ready = bool(direct_state["terminal_ready"])
            _reason = str(direct_state["terminal_reason"])
        else:
            ready = _legacy_terminal_ready(value.get("status"))
            _reason = (
                "legacy-read-only" if ready else "legacy-run-not-terminal"
            )
        if canonical is not None and not ready and not react_run:
            active_status = str(value.get("status") or "").strip().upper()
            if (
                active_status
                in {
                    "PENDING",
                    "PREPARING",
                    "RUNNING",
                    "ITERATION_SCOPE_COMPLETE",
                    "WAITING_FOR_RESOURCES",
                    "RESOURCE_BLOCKED",
                    "REUSING",
                }
                and not include_active
            ):
                continue
        try:
            run_id = run_dir.relative_to(workspace).as_posix()
        except ValueError:
            continue
        recorded_verdict = _resolve_run_verdict(value.get("status"))
        if react_run:
            governance_ledger = {
                "valid": True,
                "reason_code": "NOT_APPLICABLE_REACT_LOOP",
                "event_count": 0,
                "head_sha256": "",
                "events": [],
            }
            recorded_verdict = _react_verdict(react_state, recorded_verdict)
            effective_state = _react_effective_state(recorded_verdict)
        else:
            governance_ledger = dict(_LEGACY_GOVERNANCE_LEDGER)
            effective_state = _legacy_effective_state(
                recorded_verdict,
                recorded_status_raw=str(value.get("status") or "NOT_AVAILABLE"),
                terminal_ready=ready,
            )
        semantic = str(effective_state["effective_status"])
        verdict = {
            "status": semantic,
            "final": bool(effective_state["effective_final"]),
            "authority": (
                recorded_verdict["authority"]
                if semantic == recorded_verdict["status"]
                else "dashboard_governance"
            ),
            "reason_code": str(effective_state["reason_code"]),
        }
        physical_audit_path = run_dir / "results" / "physical_preflight.json"
        if not physical_audit_path.is_file():
            physical_audit_path = (
                run_dir / "migration" / "legacy_sampled_physical_audit_v1.json"
            )
        physical_audit = {}
        if physical_audit_path.is_file():
            try:
                physical_audit, _record = read_stable_json_object(
                    physical_audit_path,
                    label="dashboard physical preflight",
                    allowed_root=workspace,
                )
            except (OSError, OperatorAuthorityError):
                physical_audit = {}
        scan_plans: list[dict[str, Any]] = []
        for plan_path in sorted((run_dir / "plans").glob("*.json")):
            try:
                plan_value, _record = read_stable_json_object(
                    plan_path,
                    label="dashboard mechanism plan",
                    allowed_root=workspace,
                )
            except (OSError, OperatorAuthorityError):
                continue
            scan_plans.append(plan_value)
        catalog_aggregation_reasons = _capsule_evidence_gaps(
            run_dir,
            scan_plans,
        )
        whole_result_count = sum(
            isinstance(item, dict)
            for item in (value.get("whole_machine_results") or [])
        )
        if whole_result_count > 1:
            catalog_aggregation_reasons.extend(
                _whole_machine_registration_gaps(
                    [
                        item
                        for item in (value.get("whole_machine_results") or [])
                        if isinstance(item, dict)
                    ]
                )
            )
        if react_run:
            dashboard_publication = None
            publication_reason = "NOT_APPLICABLE_REACT_LOOP"
        else:
            from .publication import load_bound_dashboard_publication

            dashboard_publication, publication_reason = load_bound_dashboard_publication(
                workspace_root=workspace,
                run_dir=run_dir,
                terminal_status=value,
                require_inventory=True,
            )
        if dashboard_publication is not None:
            catalog_aggregation_reasons = []
        evidence_health = _evidence_health(
            verdict=verdict,
            canonical=canonical is not None,
            terminal_ready=ready,
            physical_audit=physical_audit,
            conflict_reasons=(
                ["STAGE_AGGREGATION_PARTIAL"]
                if recorded_verdict["status"] == "passed"
                and catalog_aggregation_reasons
                else []
            ),
            assessment_scope="catalog_terminal_inventory_and_physical_audit",
        )
        package_label = canonical.project_id if canonical else run_dir.parent.name
        package_version = ""
        for certificate_path in sorted((run_dir / "results").glob("*/certificate.json")):
            try:
                certificate_value, _record = read_stable_json_object(
                    certificate_path,
                    label="dashboard certificate",
                    allowed_root=workspace,
                )
                package = certificate_value.get("package") or {}
                package_label = str(package.get("package_id") or package_label)
                package_version = str(package.get("design_version") or "")
                break
            except (OSError, OperatorAuthorityError):
                continue
        if not package_version:
            for plan_path in sorted((run_dir / "plans").glob("*.json")):
                try:
                    plan_value, _record = read_stable_json_object(
                        plan_path,
                        label="dashboard mechanism plan",
                        allowed_root=workspace,
                    )
                    package_label = str(
                        plan_value.get("package_id") or package_label
                    )
                    package_version = str(
                        plan_value.get("design_version") or ""
                    )
                    break
                except (OSError, OperatorAuthorityError):
                    continue
        revision_preview_path = ""
        revision_preview_sha256 = ""
        for preview_candidate in (
            run_dir / "design_quality" / "assets" / "01_iso.png",
            run_dir / "visual" / "stage1_design_package.png",
        ):
            if preview_candidate.is_file():
                revision_preview_path = preview_candidate.relative_to(workspace).as_posix()
                revision_preview_sha256 = _cached_sha256_file(preview_candidate)
                break
        try:
            react_evidence_path: Path | None = None
            react_evidence = react_state.get("_dashboard_evidence_path")
            if isinstance(react_evidence, str) and react_evidence:
                candidate = (workspace / react_evidence).resolve()
                candidate.relative_to(workspace)
                if candidate.is_file() and not candidate.is_symlink():
                    react_evidence_path = candidate
            status_mtime = (
                status_path.stat().st_mtime
                if status_path.is_file()
                else (
                    react_evidence_path
                    or _react_loop_state_path(
                        workspace, run_dir.parent.parent
                    )
                    or run_dir.parent.parent / "react_loop" / "state.json"
                ).stat().st_mtime
            )
            updated_mtime = (
                max(status_mtime, physical_audit_path.stat().st_mtime)
                if physical_audit_path.is_file()
                else status_mtime
            )
        except OSError:
            # A temporary fixture can disappear while a live catalog scan is
            # in progress. It is not a dashboard run and must not abort the
            # entire historical catalog.
            continue
        rows.append(
            {
                "id": run_id,
                "name": run_dir.name,
                "project_id": (
                    canonical.project_id
                    if canonical
                    else legacy_project_id(run_dir, package_label)
                ),
                "revision": canonical.revision if canonical else "",
                "logical_run_id": canonical.run_id if canonical else run_dir.name,
                "package_id": package_label,
                "package_version": package_version,
                "display_name": _run_display_name(
                    run_dir,
                    f"{package_label} v{package_version} · {canonical.revision if canonical else run_dir.name}"
                    if package_version
                    else f"{package_label} · {canonical.revision if canonical else run_dir.name}",
                ),
                "status": semantic,
                "status_label": STATUS_LABELS[semantic],
                "status_raw": str(
                    react_state.get("status")
                    or value.get("status")
                    or "NOT_AVAILABLE"
                ),
                "recorded_status": str(recorded_verdict["status"]),
                "recorded_status_raw": str(
                    react_state.get("status")
                    or value.get("status")
                    or "NOT_AVAILABLE"
                ),
                "recorded_verdict": recorded_verdict,
                "effective_status": semantic,
                "effective_status_raw": str(
                    effective_state["effective_status_raw"]
                ),
                "verdict": verdict,
                "terminal_ready": ready,
                "terminal_reason": _reason,
                "terminal_valid": bool(
                    recorded_verdict["final"]
                    and ready
                ),
                "eligible_pass": bool(
                    effective_state["eligible_pass"]
                    and evidence_health["status"]
                    not in {"conflict", "stale", "incomplete"}
                ),
                "healthy_running": bool(
                    effective_state["healthy_running"]
                ),
                "running_lease": effective_state["lease"],
                "iteration_outcome": effective_state["iteration"],
                "governance": effective_state["governance"],
                "dashboard_publication": {
                    "bound": dashboard_publication is not None,
                    "reason_code": publication_reason,
                    "stage_count": int(
                        (dashboard_publication or {}).get("stage_count") or 0
                    ),
                },
                "evidence_health": evidence_health,
                "physical_audit_status_raw": str(
                    physical_audit.get("status") or "NOT_AVAILABLE"
                ),
                "physical_audit_recorded": bool(physical_audit),
                "revision_preview_path": revision_preview_path,
                "revision_preview_sha256": revision_preview_sha256,
                "updated_at": datetime.fromtimestamp(
                    updated_mtime,
                    timezone.utc,
                ).isoformat(),
            }
        )
    # A cold-start revision exists before project.json/run_status.json. Read
    # only the project workflow and its explicit runs directory to discover it.
    rows_by_id = {row["id"]: row for row in rows}
    for workflow_path in sorted((workspace / ".runtime_runs" / "projects").glob("*/workflow.json")):
        for cold_run in sorted((workflow_path.parent / "runs").glob("r[0-9]*")):
            if not any(cold_run == search_root or cold_run.is_relative_to(search_root) for search_root in search_roots):
                continue
            if not is_cold_start_run(workspace_root=workspace, run_dir=cold_run):
                continue
            live = read_live_project(workspace_root=workspace, run_dir=cold_run)
            workflow = live["live_workflow"]
            identity = parse_canonical_run(workspace, cold_run)
            if identity is None:
                continue
            run_id = cold_run.relative_to(workspace).as_posix()
            try:
                current_revision = f"r{int(workflow.get('revision')):02d}"
            except (TypeError, ValueError):
                current_revision = ""
            is_current = current_revision == identity.revision
            row = rows_by_id.get(run_id)
            if row is None:
                semantic = "running" if is_current and workflow.get("is_running") else "blocked" if is_current and workflow.get("stop_reason") else "preparing"
                verdict = {"status": semantic, "final": False, "authority": "cold_start_workflow", "reason_code": "COLD_START_UNSUBMITTED"}
                row = {
                    "id": run_id, "name": identity.revision,
                    "project_id": identity.project_id, "revision": identity.revision,
                    "logical_run_id": identity.run_id, "package_id": identity.project_id,
                    "package_version": "", "display_name": f"{identity.project_id} · {identity.revision}",
                    "status": semantic, "status_label": STATUS_LABELS[semantic],
                    "status_raw": "COLD_START_UNSUBMITTED", "recorded_status": "not_started",
                    "recorded_status_raw": "NOT_AVAILABLE", "recorded_verdict": _resolve_run_verdict(None),
                    "effective_status": semantic, "effective_status_raw": "COLD_START_UNSUBMITTED",
                    "verdict": verdict, "terminal_ready": False, "terminal_reason": "not_submitted",
                    "terminal_valid": False, "eligible_pass": False,
                    "healthy_running": bool(is_current and workflow.get("is_running")),
                    "evidence_health": {"status": "incomplete", "reason_codes": ["NOT_SUBMITTED"]},
                    "physical_audit_status_raw": "NOT_AVAILABLE", "physical_audit_recorded": False,
                    "revision_preview_path": "", "revision_preview_sha256": "",
                    "updated_at": workflow.get("updated_at") or "",
                }
                rows.append(row)
                rows_by_id[run_id] = row
            row.update(live)
            if workflow.get("workflow_kind") == "slim_harness":
                from ..slim_harness.publication import read_result
                result=read_result(cold_run)
                raw=result.get("status") or "SLIM_HARNESS_PENDING"
                if result.get("updated_at"):
                    row["updated_at"] = result["updated_at"]
                passed=bool(result.get("eligible_pass"))
                ended_unpassed=workflow.get("outcome") in ENDED_WITHOUT_PASS
                semantic="passed" if passed else "failed" if raw=="NEEDS_REVISION" or ended_unpassed else "blocked" if raw in {"HARNESS_ERROR","EVIDENCE_REQUIRED"} else row["status"]
                row.update(status=semantic,status_label=STATUS_LABELS[semantic],status_raw=raw,effective_status=semantic,
                           effective_status_raw=raw,eligible_pass=passed,terminal_ready=passed,terminal_valid=passed,
                           verdict={"status":semantic,"final":passed,"authority":"slim_harness","reason_code":raw})
                if is_current and not passed and workflow.get("status") == "paused":
                    row.update(status="blocked",status_label="已暂停",effective_status="blocked")
            row["workflow_current"] = is_current
            row["workflow_revision"] = current_revision
            if is_current:
                row["updated_at"] = max(str(row.get("updated_at") or ""), str(workflow.get("updated_at") or ""))
                if row.get("status") == "running":
                    row["healthy_running"] = bool(workflow.get("is_running"))
                    if not row["healthy_running"]:
                        row["status"] = "blocked" if workflow.get("stop_reason") else "preparing"
                        row["status_label"] = STATUS_LABELS[row["status"]]
                        row["effective_status"] = row["status"]
    rows.sort(key=lambda item: item["updated_at"], reverse=True)
    if limit is not None and limit > 0 and len(rows) > limit:
        # A caller-supplied history limit is soft: first protect every
        # project's latest revision, latest authoritative PASS, and active
        # revision, then fill the remaining space by recency.
        projects = group_runs_by_project(rows)
        protected_ids = {
            str(project.get(field) or "")
            for project in projects
            for field in (
                "latest_run_id",
                "published_run_id",
                "active_run_id",
                "latest_valid_terminal_run_id",
            )
            if project.get(field)
        }
        protected = [row for row in rows if row["id"] in protected_ids]
        remaining = [row for row in rows if row["id"] not in protected_ids]
        target = max(limit, len(protected))
        rows = [*protected, *remaining[: max(0, target - len(protected))]]
        rows.sort(key=lambda item: item["updated_at"], reverse=True)
    if include_history:
        return rows

    row_by_id = {str(row["id"]): row for row in rows}
    current_runs = [
        row_by_id[str(project["default_run_id"])]
        for project in group_runs_by_project(rows)
        if str(project.get("default_run_id") or "") in row_by_id
    ]
    current_runs.sort(key=lambda item: item["updated_at"], reverse=True)
    return current_runs


def group_runs_by_project(
    runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the dashboard's machine-first, revision-second selector model."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        project_id = str(
            row.get("project_id")
            or row.get("package_id")
            or Path(str(row.get("id") or "")).parent.as_posix()
        )
        if project_id in {"", ".runtime_runs"}:
            continue
        grouped.setdefault(project_id, []).append(row)

    def revision_key(row: dict[str, Any]) -> tuple[int, int, str]:
        revision = str(row.get("revision") or row.get("name") or "")
        match = re.fullmatch(r"r(\d+)", revision)
        return (
            1 if match else 0,
            int(match.group(1)) if match else 0,
            str(row.get("updated_at") or ""),
        )

    def compact_run(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "id": str(row.get("id") or ""),
            "revision": str(row.get("revision") or row.get("name") or ""),
            "status": str(row.get("status") or "not_started"),
            "recorded_status": str(
                row.get("recorded_status")
                or row.get("status")
                or "not_started"
            ),
            "effective_status": str(
                row.get("effective_status")
                or row.get("status")
                or "not_started"
            ),
            "status_label": str(
                row.get("status_label")
                or STATUS_LABELS.get(str(row.get("status") or ""), "未闭环")
            ),
            "evidence_health": str(
                (row.get("evidence_health") or {}).get("status") or "incomplete"
            ),
            "terminal_valid": bool(row.get("terminal_valid", True)),
            "eligible_pass": bool(
                row.get(
                    "eligible_pass",
                    row.get("status") == "passed",
                )
            ),
            "healthy_running": bool(
                row.get(
                    "healthy_running",
                    row.get("status") == "running",
                )
            ),
            "updated_at": str(row.get("updated_at") or ""),
        }

    projects: list[dict[str, Any]] = []
    for project_id, versions in grouped.items():
        ordered_versions = sorted(versions, key=revision_key, reverse=True)
        selectable_versions = [
            row
            for row in ordered_versions
            if row.get("workflow_current") is True or (str(
                (row.get("evidence_health") or {}).get("status") or ""
            )
            != "legacy_unverified"
            and not (
                row.get("status") == "running"
                and row.get("healthy_running") is not True
            ))
        ]
        if not selectable_versions:
            # Keep unverified legacy rows and unsigned legacy RUNNING rows
            # available for explicit forensic inspection, but never let them
            # become a project's latest/default/published/active state.
            continue
        latest = selectable_versions[0]
        workflow_current = next((row for row in ordered_versions if row.get("workflow_current") is True), None)
        published = next(
            (
                row
                for row in selectable_versions
                if row.get("status") == "passed"
                and bool(row.get("eligible_pass", True))
                and (row.get("verdict") or {}).get("authority")
                in {None, "terminal_run_status", "slim_harness"}
            ),
            None,
        )
        active = next(
            (
                row
                for row in selectable_versions
                if row.get("status") == "running"
                and bool(row.get("healthy_running", True))
            ),
            None,
        )
        latest_valid_terminal = next(
            (
                row
                for row in selectable_versions
                if row.get("status") != "running"
                and bool(
                    row.get(
                        "terminal_valid",
                        (row.get("verdict") or {}).get("final", True),
                    )
                )
            ),
            None,
        )
        active_is_newer = bool(
            active
            and active is latest
            and (
                published is None
                or revision_key(active) > revision_key(published)
            )
        )
        default = workflow_current or (
            active
            if active_is_newer
            else published
            or latest_valid_terminal
            or latest
        )
        latest_summary = compact_run(latest)
        published_summary = compact_run(published)
        active_summary = compact_run(active)
        latest_valid_terminal_summary = compact_run(latest_valid_terminal)
        projects.append(
            {
                "id": project_id,
                "title": str((default.get("live_workflow") or {}).get("title") or ""),
                "label": str(
                    default.get("package_id")
                    or default.get("project_id")
                    or project_id
                ),
                "current_run_id": str(default.get("id") or ""),
                "default_run_id": str(default.get("id") or ""),
                "latest_run_id": str(latest.get("id") or ""),
                "published_run_id": str(published.get("id") or "") if published else "",
                "last_pass_run_id": str(published.get("id") or "") if published else "",
                "active_run_id": str(active.get("id") or "") if active else "",
                "latest_valid_terminal_run_id": (
                    str(latest_valid_terminal.get("id") or "")
                    if latest_valid_terminal
                    else ""
                ),
                "default_reason": (
                    "workflow_current"
                    if workflow_current is not None
                    else "active_newer_than_published"
                    if active_is_newer
                    else "published"
                    if published
                    else "latest_valid_terminal"
                    if latest_valid_terminal
                    else "latest"
                ),
                "status": str(default.get("status") or "not_started"),
                "status_label": str(
                    default.get("status_label")
                    or STATUS_LABELS.get(
                        str(default.get("status") or "not_started"),
                        "未闭环",
                    )
                ),
                "latest_status": str(latest.get("status") or "not_started"),
                "published_status": (
                    str(published.get("status") or "not_started")
                    if published
                    else "not_started"
                ),
                "last_pass_status": (
                    str(published.get("status") or "not_started")
                    if published
                    else "not_started"
                ),
                "active_status": (
                    str(active.get("status") or "not_started")
                    if active
                    else "not_started"
                ),
                "latest": latest_summary,
                "published": published_summary,
                "last_pass": published_summary,
                "active": active_summary,
                "latest_valid_terminal": latest_valid_terminal_summary,
                "version_count": len(ordered_versions),
                "updated_at": str(latest.get("updated_at") or ""),
                "runs": ordered_versions,
            }
        )
    projects.sort(key=lambda item: item["updated_at"], reverse=True)
    return projects
