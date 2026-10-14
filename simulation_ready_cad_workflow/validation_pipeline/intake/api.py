from __future__ import annotations

from pathlib import Path

from .context import ValidationContext
from .rules_acceptance import validate_acceptance_contract
from .models import IntakeReport
from .rules_assembly import validate_assembly
from .rules_capsules import validate_capsules
from .rules_integrity import (
    derive_capabilities,
    validate_coverage,
    validate_task_mapping,
)
from .rules_manifest import (
    load_documents,
    validate_artifacts,
    validate_manifest,
    validate_requirements,
    validate_step_file,
    verify_package_stability,
)


def validate_design_package(
    package_dir: str | Path,
    requirements_path: str | Path | None,
) -> IntakeReport:
    """Validate one AI-generated design package without running physics."""
    ctx = ValidationContext(package_dir, requirements_path)
    phases = (
        ("load_documents", lambda: load_documents(ctx)),
        ("manifest", lambda: validate_manifest(ctx)),
        ("external_requirements", lambda: validate_requirements(ctx)),
        ("step_exchange", lambda: validate_step_file(ctx)),
        ("assembly_contract", lambda: validate_assembly(ctx)),
        ("mechanism_capsules", lambda: validate_capsules(ctx)),
        ("mechanism_coverage", lambda: validate_coverage(ctx)),
        ("machine_task_mapping", lambda: validate_task_mapping(ctx)),
        ("typed_acceptance_contract", lambda: validate_acceptance_contract(ctx)),
        ("artifact_integrity", lambda: validate_artifacts(ctx)),
        (
            "document_stability",
            lambda: ctx.verify_loaded_documents_unchanged(),
        ),
        (
            "package_stability",
            lambda: verify_package_stability(ctx),
        ),
    )
    for name, callback in phases:
        ctx.run_phase(name, callback)
    ctx.run_phase(
        "capability_derivation",
        lambda: derive_capabilities(ctx),
    )
    if ctx.manifest is not None:
        ctx.report.package_summary["package_id"] = ctx.manifest.get(
            "package_id"
        )
        ctx.report.package_summary["design_version"] = ctx.manifest.get(
            "design_version"
        )
    return ctx.report
