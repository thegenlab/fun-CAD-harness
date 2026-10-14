from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts.appearance import (
    AppearancePlanError,
    STATIC_VISUAL_BASE_ROLES_V2,
    STATIC_VISUAL_GATE_VERSION,
    validate_appearance_plan,
)

SCHEMA_VERSION = "0.1"
DESIGN_INPUT_NAME = "design_input.json"
VISUAL_REVIEW_RELATIVE = Path("visual_review") / "summary.json"
REQUIRED_BASE_RENDER_ROLES = frozenset({"iso", "front", "side", "top"})
STATIC_VISUAL_RENDER_ROLES = frozenset(
    {"front", "rear", "left", "right", "top", "bottom"}
)
EVIDENCE_COMPLETE = "evidence_complete"
EVIDENCE_INCOMPLETE = "evidence_incomplete"
_SNAPSHOT_ROLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DesignQualityError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


WORKER_SUPPLEMENTAL_RENDER_AUTHORITY = "worker_supplemental"


@dataclass(frozen=True)
class DesignQualityEvaluation:
    status: str
    design_input: dict[str, Any]
    visual_review: dict[str, Any]
    issues: tuple[dict[str, str], ...]
    supplemental_issues: tuple[dict[str, str], ...]
    reported_findings: dict[str, int]
    open_findings: dict[str, int]
    render_assets: tuple[dict[str, Any], ...]
    assembly_step_sha256: str

    @property
    def evidence_complete(self) -> bool:
        return self.status == EVIDENCE_COMPLETE

    @property
    def ready(self) -> bool:
        """Compatibility alias for the pre-execution evidence gate."""
        return self.evidence_complete

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status,
            "ready": self.ready,
            "evidence_complete": self.evidence_complete,
            "authority": {
                "scope": "design_input_package_completeness",
                "certificate_authority": False,
                "submitter_review_status_used": False,
                "submitter_finding_status_used": False,
            },
            "assembly_step_sha256": self.assembly_step_sha256,
            "reported_findings": dict(self.reported_findings),
            "open_findings": dict(self.open_findings),
            "issues": [dict(item) for item in self.issues],
            "supplemental_issues": [dict(item) for item in self.supplemental_issues],
            "render_assets": [dict(item) for item in self.render_assets],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, code: str) -> dict[str, Any]:
    if not path.is_file():
        raise DesignQualityError(code, f"required record is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DesignQualityError(code, f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DesignQualityError(code, f"record must contain a JSON object: {path}")
    return value


def _required_text(value: Any, *, field: str, issues: list[dict[str, str]]) -> str:
    text = str(value or "").strip()
    if not text:
        issues.append(
            {
                "code": "DESIGN_INPUT_FIELD_MISSING",
                "field": field,
                "message": f"required design-input field is empty: {field}",
            }
        )
    return text


def _inside(root: Path, value: str, *, field: str) -> Path:
    candidate = (root / value).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise DesignQualityError(
            "VISUAL_REVIEW_ASSET_UNSAFE",
            f"{field} must stay inside {root}",
        ) from exc
    return candidate


def _iter_round_assets(review: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for round_item in review.get("rounds") or []:
        if not isinstance(round_item, dict):
            continue
        round_number = round_item.get("round")
        for asset in round_item.get("render_assets") or []:
            if not isinstance(asset, dict):
                continue
            yield {**asset, "round": round_number}


def evaluate_design_quality(
    package_dir: str | Path,
    *,
    design_input_path: str | Path | None = None,
    visual_review_path: str | Path | None = None,
) -> DesignQualityEvaluation:
    package = Path(package_dir).resolve()
    record_root = package.parent
    manifest_path = package / "package.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        manifest = {}
    single_source = manifest.get("single_source_cad") if isinstance(manifest, dict) else None
    standard_required = (
        isinstance(single_source, dict) and single_source.get("policy") == "required_v1"
    )
    design_path = (
        Path(design_input_path).resolve()
        if design_input_path is not None
        else record_root / DESIGN_INPUT_NAME
    )
    review_path = (
        Path(visual_review_path).resolve()
        if visual_review_path is not None
        else record_root / VISUAL_REVIEW_RELATIVE
    )
    assembly_path = package / "assembly.step"
    issues: list[dict[str, str]] = []

    try:
        design = _read_object(design_path, code="DESIGN_INPUT_MISSING")
    except DesignQualityError as exc:
        return DesignQualityEvaluation(
            status=EVIDENCE_INCOMPLETE,
            design_input={},
            visual_review={},
            issues=({"code": exc.code, "field": "", "message": str(exc)},),
            supplemental_issues=(),
            reported_findings={"P0": 0, "P1": 0, "P2": 0},
            open_findings={"P0": 0, "P1": 0, "P2": 0},
            render_assets=(),
            assembly_step_sha256=sha256_file(assembly_path) if assembly_path.is_file() else "",
        )

    _required_text(design.get("original_prompt"), field="original_prompt", issues=issues)
    _required_text(
        design.get("plain_language_goal"),
        field="plain_language_goal",
        issues=issues,
    )
    brief = design.get("engineering_brief")
    if not isinstance(brief, dict) or not brief:
        issues.append(
            {
                "code": "DESIGN_INPUT_FIELD_MISSING",
                "field": "engineering_brief",
                "message": "engineering_brief must be a non-empty object",
            }
        )
    checklist = design.get("visual_acceptance_checklist")
    if not isinstance(checklist, list) or not any(str(item).strip() for item in checklist):
        issues.append(
            {
                "code": "DESIGN_INPUT_FIELD_MISSING",
                "field": "visual_acceptance_checklist",
                "message": "visual_acceptance_checklist must contain at least one item",
            }
        )
    motion_roles_value = design.get("motion_review_roles")
    motion_review_roles = tuple(
        str(item).strip()
        for item in (motion_roles_value if isinstance(motion_roles_value, list) else [])
        if str(item).strip()
    )
    if str(design.get("schema_version") or "") != SCHEMA_VERSION:
        issues.append(
            {
                "code": "DESIGN_INPUT_SCHEMA_UNSUPPORTED",
                "field": "schema_version",
                "message": f"design input must use schema {SCHEMA_VERSION}",
            }
        )

    if not assembly_path.is_file():
        issues.append(
            {
                "code": "AUTHORITATIVE_STEP_MISSING",
                "field": "assembly.step",
                "message": f"authoritative STEP is missing: {assembly_path}",
            }
        )
        assembly_sha256 = ""
    else:
        assembly_sha256 = sha256_file(assembly_path)

    supplemental_issues: list[dict[str, str]] = []
    review_present = False
    try:
        review = _read_object(review_path, code="VISUAL_REVIEW_MISSING")
        review_present = True
    except DesignQualityError as exc:
        review = {}
        supplemental_issues.append(
            {"code": exc.code, "field": "", "message": str(exc)}
        )

    if review_present and str(review.get("schema_version") or "") != SCHEMA_VERSION:
        supplemental_issues.append(
            {
                "code": "VISUAL_REVIEW_SCHEMA_UNSUPPORTED",
                "field": "schema_version",
                "message": f"visual review must use schema {SCHEMA_VERSION}",
            }
        )
    static_visual_v1 = review.get("static_visual_gate_version") == 1
    static_visual_v2 = (
        review.get("static_visual_gate_version") == STATIC_VISUAL_GATE_VERSION
    )
    appearance_plan: dict[str, Any] | None = None
    if static_visual_v2 or design.get("appearance_plan") is not None:
        try:
            appearance_plan = validate_appearance_plan(
                design.get("appearance_plan")
            )
        except AppearancePlanError as exc:
            issues.append(
                {
                    "code": "APPEARANCE_PLAN_INVALID",
                    "field": "appearance_plan",
                    "message": str(exc),
                }
            )
    base_render_roles = (
        frozenset(STATIC_VISUAL_BASE_ROLES_V2)
        if static_visual_v2
        else
        STATIC_VISUAL_RENDER_ROLES
        if static_visual_v1
        else REQUIRED_BASE_RENDER_ROLES
    )
    has_motion_pose = any(
        role not in base_render_roles for role in motion_review_roles
    )
    if standard_required and not has_motion_pose:
        issues.append(
            {
                "code": "DESIGN_INPUT_FIELD_MISSING",
                "field": "motion_review_roles",
                "message": "motion_review_roles must declare a non-base mechanism pose",
            }
        )
    if appearance_plan is not None:
        pose_roles = {
            item["role"]
            for item in appearance_plan["view_requirements"]
            if item["kind"] == "pose"
        }
        missing_pose_specs = sorted(set(motion_review_roles) - pose_roles)
        if missing_pose_specs:
            issues.append(
                {
                    "code": "APPEARANCE_PLAN_POSE_MISSING",
                    "field": "appearance_plan.view_requirements",
                    "message": (
                        "motion roles need pose view specifications: "
                        + ", ".join(missing_pose_specs)
                    ),
                }
            )
    if review_present:
        review_sha256 = str(review.get("assembly_step_sha256") or "").lower()
        if not review_sha256 or review_sha256 != assembly_sha256.lower():
            supplemental_issues.append(
                {
                    "code": "VISUAL_REVIEW_STEP_STALE",
                    "field": "assembly_step_sha256",
                    "message": "visual review does not match the authoritative STEP hash",
                }
            )

    reported_counts = {"P0": 0, "P1": 0, "P2": 0}
    open_counts = {"P0": 0, "P1": 0, "P2": 0}
    findings: list[dict[str, Any]] = []
    for round_item in review.get("rounds") or []:
        if not isinstance(round_item, dict):
            continue
        for finding in round_item.get("findings") or []:
            if not isinstance(finding, dict):
                continue
            findings.append(finding)
    latest_by_id: dict[str, dict[str, Any]] = {}
    for index, finding in enumerate(findings):
        finding_id = str(finding.get("finding_id") or "").strip()
        latest_by_id[finding_id or f"__anonymous_{index}"] = finding
    for finding in latest_by_id.values():
        severity = str(finding.get("severity") or "").upper()
        state = str(finding.get("status") or "OPEN").upper()
        if severity in reported_counts:
            reported_counts[severity] += 1
            if state != "RESOLVED":
                open_counts[severity] += 1

    assets: list[dict[str, Any]] = []
    asset_root = review_path.parent.parent
    for asset in _iter_round_assets(review):
        role = str(asset.get("role") or "render")
        relative = str(asset.get("path") or "").strip()
        expected_hash = str(asset.get("sha256") or "").lower()
        if not relative:
            supplemental_issues.append(
                {
                    "code": "VISUAL_REVIEW_ASSET_MISSING",
                    "field": role,
                    "message": f"render asset path is missing for role {role}",
                }
            )
            continue
        try:
            asset_path = _inside(asset_root, relative, field=role)
        except DesignQualityError as exc:
            supplemental_issues.append(
                {"code": exc.code, "field": role, "message": str(exc)}
            )
            continue
        actual_hash = sha256_file(asset_path) if asset_path.is_file() else ""
        if not actual_hash:
            supplemental_issues.append(
                {
                    "code": "VISUAL_REVIEW_ASSET_MISSING",
                    "field": role,
                    "message": f"render asset is missing: {relative}",
                }
            )
            continue
        if expected_hash != actual_hash.lower():
            supplemental_issues.append(
                {
                    "code": "VISUAL_REVIEW_ASSET_STALE",
                    "field": role,
                    "message": f"render asset hash mismatch: {relative}",
                }
            )
            continue
        assets.append(
            {
                "role": role,
                "label": str(asset.get("label") or role),
                "path": str(asset_path),
                "sha256": actual_hash,
                "round": asset.get("round"),
                "render_authority": WORKER_SUPPLEMENTAL_RENDER_AUTHORITY,
                "review_authority": False,
            }
        )

    if review_present:
        review_rounds = [
            item for item in review.get("rounds") or [] if isinstance(item, dict)
        ]
        latest_round = review_rounds[-1].get("round") if review_rounds else None
        available_roles = {
            str(item.get("role") or "").strip()
            for item in assets if item.get("round") == latest_round
        }
        required_roles = set(base_render_roles).union(motion_review_roles)
        if appearance_plan is not None:
            required_roles.update(
                item["role"] for item in appearance_plan["view_requirements"]
            )
        missing_roles = sorted(
            role for role in required_roles if role not in available_roles
        )
        if standard_required and missing_roles:
            supplemental_issues.append(
                {
                    "code": "VISUAL_REVIEW_REQUIRED_ROLE_MISSING",
                    "field": "rounds[].render_assets[]",
                    "message": (
                        "visual review must list each required view under "
                        "rounds[].render_assets[]; top-level render_assets, views, "
                        "or renders fields are ignored; missing required views: "
                        + ", ".join(missing_roles)
                    ),
                }
            )

    status = EVIDENCE_INCOMPLETE if issues else EVIDENCE_COMPLETE
    return DesignQualityEvaluation(
        status=status,
        design_input=design,
        visual_review=review,
        issues=tuple(issues),
        supplemental_issues=tuple(supplemental_issues),
        reported_findings=reported_counts,
        open_findings=open_counts,
        render_assets=tuple(assets),
        assembly_step_sha256=assembly_sha256,
    )


def snapshot_design_quality(
    package_dir: str | Path,
    output_dir: str | Path,
) -> DesignQualityEvaluation:
    package = Path(package_dir).resolve()
    output = Path(output_dir).resolve()
    evaluation = evaluate_design_quality(package)
    destination = output / "design_quality"
    destination.mkdir(parents=True, exist_ok=True)
    copied_assets: list[dict[str, Any]] = []

    if evaluation.design_input:
        (destination / "design_input.json").write_text(
            json.dumps(evaluation.design_input, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if evaluation.visual_review:
        review_copy = json.loads(json.dumps(evaluation.visual_review))
        assets_dir = destination / "assets"
        for index, asset in enumerate(evaluation.render_assets, start=1):
            source = Path(asset["path"])
            suffix = source.suffix.lower() or ".bin"
            role = asset.get("role")
            if not isinstance(role, str) or _SNAPSHOT_ROLE.fullmatch(role) is None:
                raise DesignQualityError(
                    "VISUAL_REVIEW_ASSET_ROLE_INVALID",
                    "render asset roles used in snapshot filenames must use "
                    "only letters, digits, dot, underscore, or hyphen",
                )
            target = assets_dir / f"{index:02d}_{role}{suffix}"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_assets.append(
                {
                    **asset,
                    "path": target.relative_to(output).as_posix(),
                }
            )
        review_copy["snapshot_render_assets"] = copied_assets
        (destination / "visual_review.json").write_text(
            json.dumps(review_copy, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    evaluation_payload = evaluation.to_dict()
    evaluation_payload["render_assets"] = copied_assets
    (destination / "evaluation.json").write_text(
        json.dumps(evaluation_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return evaluation
