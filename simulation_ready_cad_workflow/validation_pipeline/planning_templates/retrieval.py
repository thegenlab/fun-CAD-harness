"""Small deterministic template recommendations for one frozen task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..intake.acceptance_contract import compile_acceptance_contract
from ..mechanism.io import canonical_sha256
from .loader import load_builtin_template_catalog


RECOMMENDATION_SCHEMA = "simulation_ready_cad_planning_recommendations_v1"


def retrieve_builtin_template_recommendations(
    requirements: Mapping[str, Any], *, max_task_suites: int = 3
) -> dict[str, Any]:
    contract = compile_acceptance_contract(requirements)
    requirement_entries = [
        item
        for item in contract.get("requirements") or []
        if isinstance(item, Mapping)
    ]
    catalog = load_builtin_template_catalog()
    scored: list[tuple[int, str, str, str, Any, set[str]]] = []
    for record in catalog.records_for("task_suite"):
        payload = record.payload()
        covered: set[str] = set()
        for requirement in requirement_entries:
            required_branches = set(requirement.get("required_branches") or [])
            for scenario in payload.get("scenarios") or []:
                if not isinstance(scenario, Mapping):
                    continue
                if (
                    scenario.get("runtime_status") == "ACTIVE"
                    and scenario.get("test_program")
                    == requirement.get("test_program")
                    and scenario.get("test_load_case_id")
                    == requirement.get("test_load_case_id")
                    and required_branches
                    <= set(scenario.get("required_branches") or [])
                ):
                    covered.add(str(requirement.get("requirement_id") or ""))
                    break
        if covered:
            scored.append(
                (
                    -len(covered),
                    record.template_id,
                    record.version,
                    record.sha256,
                    record,
                    covered,
                )
            )
    selected = sorted(scored)[:max_task_suites]
    covered_ids = set().union(*(item[5] for item in selected)) if selected else set()
    selected_joint_refs: dict[tuple[str, str], dict[str, Any]] = {}
    entries: list[dict[str, Any]] = []
    for _score, _template_id, _version, _sha256, record, covered in selected:
        payload = record.payload()
        entries.append(
            {
                "kind": "task_suite",
                "template_id": record.template_id,
                "version": record.version,
                "sha256": record.sha256,
                "covered_requirement_ids": sorted(covered),
                "payload": payload,
            }
        )
        for reference in payload.get("joint_template_refs") or []:
            if not isinstance(reference, Mapping):
                continue
            key = (
                str(reference.get("template_id") or ""),
                str(reference.get("version") or ""),
            )
            selected_joint_refs.setdefault(key, dict(reference))
    for (template_id, version), reference in sorted(selected_joint_refs.items()):
        try:
            record = catalog.resolve("joint", template_id, version)
        except KeyError:
            continue
        entries.append(
            {
                "kind": "joint",
                "template_id": record.template_id,
                "version": record.version,
                "sha256": record.sha256,
                "selected_variant_id": reference.get("variant_id"),
                "payload": record.payload(),
            }
        )
    all_ids = {
        str(item.get("requirement_id") or "") for item in requirement_entries
    }
    uncovered = sorted(all_ids - covered_ids)
    status = (
        "EXACT_MATCH"
        if all_ids and not uncovered
        else "PARTIAL_MATCH"
        if covered_ids
        else "NO_EXACT_MATCH"
    )
    report = {
        "schema_version": RECOMMENDATION_SCHEMA,
        "requirements_id": requirements.get("requirements_id"),
        "catalog_id": catalog.catalog_id,
        "catalog_version": catalog.catalog_version,
        "catalog_sha256": catalog.catalog_sha256,
        "status": status,
        "covered_requirement_ids": sorted(covered_ids),
        "uncovered_requirement_ids": uncovered,
        "entries": entries,
        "instruction": (
            "Use exact recommended templates when they cover the plan. The "
            "full catalog remains the final selection authority. If any "
            "requirement is uncovered, author or request the missing "
            "catalog-shaped capability instead of treating an approximate "
            "template as certified."
        ),
    }
    report["recommendation_sha256"] = canonical_sha256(report)
    return report
