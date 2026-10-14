from __future__ import annotations

from pathlib import Path
from typing import Any

from .context import COVERAGE_CLASSES, ValidationContext
from .rules_assembly import _string_items


def _document_label(
    ctx: ValidationContext, key: str, fallback: str
) -> str:
    path = ctx.paths.get(key)
    return ctx.relative_label(path) if isinstance(path, Path) else fallback


def validate_coverage(ctx: ValidationContext) -> None:
    document = ctx.coverage
    if document is None:
        return
    file = _document_label(
        ctx, "coverage", "mechanism_coverage.json"
    )
    ctx.check_schema_version(document, file)
    ctx.scan_for_forbidden_authority(document, file=file)
    ctx.warn_unknown_fields(
        document,
        {"schema_version", "coverage"},
        file=file,
    )
    entries = ctx.require_list(
        document, "coverage", file=file, nonempty=True
    )
    if entries is None:
        return
    seen: set[str] = set()
    declarations: dict[str, tuple[str, set[str]]] = {}
    for index, entry in enumerate(entries):
        base = f"$.coverage[{index}]"
        if not isinstance(entry, dict):
            ctx.error(
                "COVERAGE_ENTRY_INVALID",
                "Each coverage entry must be an object.",
                file=file,
                json_path=base,
                actual=entry,
            )
            continue
        ctx.warn_unknown_fields(
            entry,
            {"part_id", "classification", "capsule_ids"},
            file=file,
            json_path=base,
        )
        part_id = entry.get("part_id")
        if not isinstance(part_id, str) or part_id not in ctx.part_ids:
            ctx.error(
                "COVERAGE_PART_UNKNOWN",
                "Coverage references an unknown part.",
                file=file,
                json_path=f"{base}.part_id",
                expected=sorted(ctx.part_ids),
                actual=part_id,
            )
            continue
        if part_id in seen:
            ctx.error(
                "COVERAGE_PART_DUPLICATE",
                "Every part must have exactly one coverage entry.",
                file=file,
                json_path=f"{base}.part_id",
                actual=part_id,
            )
        seen.add(part_id)
        classification = entry.get("classification")
        if classification not in COVERAGE_CLASSES:
            ctx.error(
                "COVERAGE_CLASS_INVALID",
                "Coverage classification is not recognized.",
                file=file,
                json_path=f"{base}.classification",
                expected=sorted(COVERAGE_CLASSES),
                actual=classification,
            )
        capsule_values = _string_items(
            ctx,
            entry.get("capsule_ids"),
            file=file,
            json_path=f"{base}.capsule_ids",
            nonempty=False,
        )
        capsule_ids = set(capsule_values or [])
        for capsule_index, capsule_id in enumerate(capsule_values or []):
            if capsule_id not in ctx.capsules:
                ctx.error(
                    "COVERAGE_CAPSULE_UNKNOWN",
                    "Coverage references an unknown capsule.",
                    file=file,
                    json_path=f"{base}.capsule_ids[{capsule_index}]",
                    actual=capsule_id,
                )
        if classification == "capsule_member" and not capsule_ids:
            ctx.error(
                "CAPSULE_MEMBER_WITHOUT_CAPSULE",
                "capsule_member coverage needs at least one capsule_id.",
                file=file,
                json_path=f"{base}.capsule_ids",
                actual=capsule_values,
            )
        if classification != "capsule_member" and capsule_ids:
            ctx.error(
                "NON_MEMBER_HAS_CAPSULE",
                "Only capsule_member entries may list capsule_ids.",
                file=file,
                json_path=base,
                expected=[],
                actual=sorted(capsule_ids),
            )
        role = ctx.part_roles.get(part_id)
        if role == "dynamic" and classification != "capsule_member":
            ctx.error(
                "DYNAMIC_PART_NOT_COVERED",
                "Every dynamic part must be owned by a mechanism capsule.",
                file=file,
                json_path=f"{base}.classification",
                expected="capsule_member",
                actual=classification,
                hint=(
                    "Add the part's rigid group to a capsule or correct its "
                    "assembly role."
                ),
            )
        if classification == "unclassified":
            ctx.warning(
                "PART_UNCLASSIFIED",
                "Part has not been assigned a simulation responsibility.",
                file=file,
                json_path=f"{base}.classification",
                actual=part_id,
            )
        if isinstance(classification, str):
            declarations[part_id] = (classification, capsule_ids)

    for part_id in sorted(ctx.part_ids - seen):
        ctx.error(
            "PART_COVERAGE_MISSING",
            "Assembly part has no mechanism coverage entry.",
            file=file,
            json_path="$.coverage",
            actual=part_id,
        )

    for capsule_id, capsule in ctx.capsules.items():
        member_groups = capsule.get("member_rigid_group_ids")
        if not isinstance(member_groups, list):
            continue
        for part_id, group_id in ctx.part_groups.items():
            if group_id not in member_groups:
                continue
            classification, capsule_ids = declarations.get(
                part_id, ("", set())
            )
            if (
                classification != "capsule_member"
                or capsule_id not in capsule_ids
            ):
                ctx.error(
                    "CAPSULE_MEMBERSHIP_NOT_COVERED",
                    "Capsule member group is not reflected in part coverage.",
                    file=file,
                    json_path="$.coverage",
                    expected={
                        "part_id": part_id,
                        "classification": "capsule_member",
                        "capsule_id": capsule_id,
                    },
                    actual={
                        "classification": classification or None,
                        "capsule_ids": sorted(capsule_ids),
                    },
                )
    ctx.report.package_summary["covered_part_count"] = len(seen)


def _validate_task_refs(
    ctx: ValidationContext,
    values: Any,
    *,
    file: str,
    base: str,
    kind: str,
    declared_capsules: set[str],
) -> None:
    if not isinstance(values, list) or not values:
        ctx.error(
            f"TASK_{kind.upper()}S_MISSING",
            f"Task mapping needs at least one {kind} reference.",
            file=file,
            json_path=base,
            expected="non-empty array",
            actual=values,
        )
        return
    for index, value in enumerate(values):
        path = f"{base}[{index}]"
        if not isinstance(value, dict):
            ctx.error(
                f"TASK_{kind.upper()}_INVALID",
                f"Each task {kind} reference must be an object.",
                file=file,
                json_path=path,
                actual=value,
            )
            continue
        item_key = "port_id" if kind == "input" else "observable_id"
        ctx.warn_unknown_fields(
            value,
            {"capsule_id", item_key},
            file=file,
            json_path=path,
        )
        capsule_id = value.get("capsule_id")
        item_id = value.get(item_key)
        if capsule_id not in ctx.capsules:
            ctx.error(
                f"TASK_{kind.upper()}_CAPSULE_UNKNOWN",
                f"Task {kind} references an unknown capsule.",
                file=file,
                json_path=f"{path}.capsule_id",
                actual=capsule_id,
            )
            continue
        if capsule_id not in declared_capsules:
            ctx.error(
                f"TASK_{kind.upper()}_CAPSULE_UNDECLARED",
                f"Task {kind} capsule is absent from task capsule_ids.",
                file=file,
                json_path=f"{path}.capsule_id",
                expected=sorted(declared_capsules),
                actual=capsule_id,
            )
        known = (
            ctx.capsule_port_ids.get(capsule_id, set())
            if kind == "input"
            else ctx.capsule_observable_ids.get(capsule_id, set())
        )
        if item_id not in known:
            ctx.error(
                f"TASK_{kind.upper()}_UNKNOWN",
                f"Task {kind} references an unknown {item_key}.",
                file=file,
                json_path=f"{path}.{item_key}",
                expected=sorted(known),
                actual=item_id,
            )


def validate_task_mapping(ctx: ValidationContext) -> None:
    document = ctx.task_mapping
    if document is None:
        return
    file = _document_label(
        ctx, "task_mapping", "machine_task_mapping.json"
    )
    ctx.check_schema_version(document, file)
    ctx.scan_for_forbidden_authority(document, file=file)
    ctx.warn_unknown_fields(
        document,
        {"schema_version", "requirements_id", "task_mappings"},
        file=file,
    )
    external_id = (
        ctx.requirements.get("requirements_id")
        if ctx.requirements is not None
        else None
    )
    declared_id = document.get("requirements_id")
    if declared_id != external_id:
        ctx.error(
            "TASK_MAPPING_REQUIREMENTS_MISMATCH",
            "Task mapping targets a different requirements document.",
            file=file,
            json_path="$.requirements_id",
            expected=external_id,
            actual=declared_id,
        )
    mappings = ctx.require_list(
        document, "task_mappings", file=file, nonempty=True
    )
    if mappings is None:
        return
    seen: set[str] = set()
    for index, mapping in enumerate(mappings):
        base = f"$.task_mappings[{index}]"
        if not isinstance(mapping, dict):
            ctx.error(
                "TASK_MAPPING_ENTRY_INVALID",
                "Each task mapping must be an object.",
                file=file,
                json_path=base,
                actual=mapping,
            )
            continue
        ctx.warn_unknown_fields(
            mapping,
            {
                "task_id",
                "capsule_ids",
                "inputs",
                "observables",
                "implementation_claim",
            },
            file=file,
            json_path=base,
        )
        task_id = mapping.get("task_id")
        if task_id not in ctx.requirement_task_ids:
            ctx.error(
                "TASK_MAPPING_TASK_UNKNOWN",
                "Task mapping references an unknown external task.",
                file=file,
                json_path=f"{base}.task_id",
                expected=sorted(ctx.requirement_task_ids),
                actual=task_id,
            )
        elif task_id in seen:
            ctx.error(
                "TASK_MAPPING_TASK_DUPLICATE",
                "Each external task may be mapped only once.",
                file=file,
                json_path=f"{base}.task_id",
                actual=task_id,
            )
        else:
            seen.add(task_id)
        capsule_values = _string_items(
            ctx,
            mapping.get("capsule_ids"),
            file=file,
            json_path=f"{base}.capsule_ids",
        )
        declared_capsules = set(capsule_values or [])
        for capsule_index, capsule_id in enumerate(capsule_values or []):
            if capsule_id not in ctx.capsules:
                ctx.error(
                    "TASK_MAPPING_CAPSULE_UNKNOWN",
                    "Task mapping references an unknown capsule.",
                    file=file,
                    json_path=f"{base}.capsule_ids[{capsule_index}]",
                    actual=capsule_id,
                )
        _validate_task_refs(
            ctx,
            mapping.get("inputs"),
            file=file,
            base=f"{base}.inputs",
            kind="input",
            declared_capsules=declared_capsules,
        )
        _validate_task_refs(
            ctx,
            mapping.get("observables"),
            file=file,
            base=f"{base}.observables",
            kind="observable",
            declared_capsules=declared_capsules,
        )
        ctx.require_string(
            mapping,
            "implementation_claim",
            file=file,
            base_path=base,
        )

    for task_id in sorted(ctx.requirement_task_ids - seen):
        ctx.error(
            "REQUIRED_TASK_UNMAPPED",
            "External requirement task has no machine-task mapping.",
            file=file,
            json_path="$.task_mappings",
            actual=task_id,
            hint=(
                "Map the task to concrete capsule inputs and observables; "
                "do not remove or weaken the external requirement."
            ),
        )
    ctx.report.package_summary["mapped_task_count"] = len(seen)


def derive_capabilities(ctx: ValidationContext) -> None:
    checks = {item["name"]: item for item in ctx.report.checks}
    preflight_phases = {
        "load_documents",
        "manifest",
        "external_requirements",
        "step_exchange",
        "assembly_contract",
        "mechanism_capsules",
        "mechanism_coverage",
        "artifact_integrity",
        "document_stability",
        "package_stability",
    }
    failed_preflight_phases = sorted(
        name
        for name in preflight_phases
        if checks.get(name, {}).get("status") == "FAIL"
    )
    invalid_breps = sorted(
        solid_id
        for solid_id, facts in ctx.step_solid_inventory.items()
        if facts.get("valid_brep") is not True
    )
    geometry_ready = (
        ctx.step_geometry_verified
        and bool(ctx.step_solid_inventory)
        and not invalid_breps
    )
    preflight_blockers = [
        f"phase:{name}" for name in failed_preflight_phases
    ]
    if not ctx.step_geometry_verified:
        preflight_blockers.append("STEP_GEOMETRY_NOT_KERNEL_VERIFIED")
    elif invalid_breps:
        preflight_blockers.append("STEP_BREP_INVALID")
    can_preflight = not failed_preflight_phases and geometry_ready
    all_capsules = set(ctx.capsules)
    reduced_ready = set(ctx.reduced_ready_capsules)
    mapping_ok = (
        checks.get("machine_task_mapping", {}).get("status") != "FAIL"
        and len(ctx.requirement_task_ids)
        == ctx.report.package_summary.get("mapped_task_count", -1)
    )
    acceptance_summary = ctx.report.package_summary.get(
        "acceptance_contract"
    )
    typed_acceptance_ready = (
        isinstance(acceptance_summary, dict)
        and acceptance_summary.get("certificate_authority") is True
    )
    ctx.report.capabilities = {
        "can_enter_physical_preflight": can_preflight,
        "can_attempt_reduced_certification": (
            can_preflight
            and bool(all_capsules)
            and reduced_ready == all_capsules
        ),
        "can_run_whole_machine_task_test": can_preflight and mapping_ok,
        "can_issue_typed_task_certificate": (
            can_preflight and typed_acceptance_ready
        ),
        "physical_preflight_blockers": preflight_blockers,
        "capsules_ready_for_reduced_certification": sorted(reduced_ready),
        "capsules_requiring_contact_model_only": sorted(
            all_capsules - reduced_ready
        ),
    }
