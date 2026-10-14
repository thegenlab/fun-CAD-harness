from __future__ import annotations

"""Intake phase that compiles and binds task acceptance evidence."""

from typing import Any

from .acceptance_contract import (
    AcceptanceContractError,
    compile_acceptance_contract,
)
from .context import ValidationContext
from .semantic_binding import (
    SemanticBindingError,
    bind_acceptance_contract,
)


def validate_acceptance_contract(ctx: ValidationContext) -> None:
    if (
        ctx.requirements is None
        or ctx.task_mapping is None
        or ctx.assembly is None
        or not ctx.capsules
    ):
        return
    try:
        contract = compile_acceptance_contract(ctx.requirements)
    except AcceptanceContractError as exc:
        ctx.error(
            exc.code,
            str(exc),
            file=ctx.requirements_path,
            json_path=exc.json_path,
            expected=exc.expected,
            actual=exc.actual,
            hint=(
                "Regenerate the typed contract from the immutable prompt; "
                "do not weaken the target to fit the CAD."
            ),
        )
        ctx.acceptance_contract = {
            "status": "FAIL",
            "certificate_authority": False,
            "failure_reasons": [exc.code],
        }
        return

    ctx.acceptance_contract = contract
    try:
        binding = bind_acceptance_contract(
            contract,
            task_mapping=ctx.task_mapping,
            capsules=ctx.capsules,
            assembly=ctx.assembly,
        )
    except SemanticBindingError as exc:
        ctx.error(
            exc.code,
            str(exc),
            file=ctx.paths.get("task_mapping"),
            json_path="$.task_mappings",
            actual={
                "requirement_id": exc.requirement_id,
                "evidence": exc.evidence,
            },
            hint=(
                "Correct package semantics or geometry so one mapped object "
                "wins deterministically; do not add a manual approval step."
            ),
        )
        binding = {
            "schema_version": "semantic_binding_v1",
            "status": "FAIL",
            "certificate_authority": False,
            "contract_sha256": contract.get("contract_sha256"),
            "bindings": [],
            "failure_reasons": [exc.code],
        }
    ctx.semantic_binding = binding
    ctx.report.package_summary["acceptance_contract"] = {
        "status": contract.get("status"),
        "certificate_authority": (
            contract.get("certificate_authority") is True
            and binding.get("certificate_authority") is True
            and binding.get("status") == "PASS"
        ),
        "contract": contract,
        "semantic_binding": binding,
    }
    if contract.get("status") == "LEGACY_UNTYPED":
        ctx.info(
            "LEGACY_ACCEPTANCE_CONTRACT",
            (
                "This schema 0.1 requirements file has no typed task contract. "
                "Historical evidence remains readable, but it does not gain "
                "new typed-task certificate authority."
            ),
            file=ctx.requirements_path,
            hint=(
                "Create a new requirements revision only when the case needs "
                "a new certificate; do not rewrite historical results."
            ),
        )


def acceptance_capability(summary: Any) -> bool:
    return (
        isinstance(summary, dict)
        and summary.get("certificate_authority") is True
        and isinstance(summary.get("contract"), dict)
        and summary["contract"].get("status") == "PASS"
        and isinstance(summary.get("semantic_binding"), dict)
        and summary["semantic_binding"].get("status") == "PASS"
    )
