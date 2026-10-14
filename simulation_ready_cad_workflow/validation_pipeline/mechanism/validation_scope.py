from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .io import canonical_sha256
from .models import LoadCase, MechanismPlan, MechanismValidationError


VALIDATION_SCOPE_SCHEMA = "mechanism_validation_scope_v1"
VALIDATION_MODE_FINAL_FULL = "final-full"
VALIDATION_MODE_ITERATION_TASK = "iteration-task"
VALIDATION_MODES = frozenset(
    {
        VALIDATION_MODE_FINAL_FULL,
        VALIDATION_MODE_ITERATION_TASK,
    }
)
FINAL_FULL_SELECTOR_AUTHORITY = "plugin.load_cases(plan)"
ITERATION_TASK_SELECTOR_AUTHORITY = (
    "plan.acceptance_contract.requirements[*]."
    "(test_load_case_id,test_program)->plugin.load_cases(plan)."
    "(canonical_test_load_case_id,program)"
)
_CONTRACT_BRANCHES = frozenset(
    {"physical", "ideal_reference", "ideal_dynamics"}
)


@dataclass(frozen=True)
class ResolvedValidationScope:
    """Trusted dispatch selection derived without caller-provided case IDs."""

    document: dict[str, Any]
    selected_case_ids_by_capsule: dict[str, tuple[str, ...]]
    required_branches_by_selector: dict[tuple[str, str], tuple[str, ...]]

    def selected_case_ids(self, capsule_id: str) -> tuple[str, ...]:
        return self.selected_case_ids_by_capsule[capsule_id]

    def required_branches(
        self,
        capsule_id: str,
        load_case_id: str,
    ) -> tuple[str, ...]:
        return self.required_branches_by_selector[
            (capsule_id, load_case_id)
        ]


def validate_validation_mode(mode: Any) -> str:
    if not isinstance(mode, str) or mode not in VALIDATION_MODES:
        raise MechanismValidationError(
            "VALIDATION_SCOPE_MODE_UNKNOWN",
            "validation mode must be final-full or iteration-task",
        )
    return mode


def _selector(capsule_id: str, load_case_id: str) -> dict[str, str]:
    return {
        "capsule_id": capsule_id,
        "load_case_id": load_case_id,
    }


def _authoritative_contract(plan: MechanismPlan) -> Mapping[str, Any]:
    contract = plan.acceptance_contract
    if (
        not isinstance(contract, Mapping)
        or contract.get("status") != "PASS"
        or contract.get("certificate_authority") is not True
    ):
        raise MechanismValidationError(
            "VALIDATION_SCOPE_CONTRACT_NOT_AUTHORITATIVE",
            "iteration-task requires a passing authority-issued "
            f"acceptance contract for capsule {plan.capsule_id!r}",
        )
    requirements = contract.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise MechanismValidationError(
            "VALIDATION_SCOPE_CONTRACT_CASE_MISSING",
            "iteration-task acceptance contract has no requirements for "
            f"capsule {plan.capsule_id!r}",
        )
    return contract


def _iteration_selection(
    plan: MechanismPlan,
    *,
    inventory: tuple[LoadCase, ...],
) -> tuple[tuple[str, ...], dict[str, tuple[str, ...]]]:
    contract = _authoritative_contract(plan)
    selected: list[str] = []
    seen: set[str] = set()
    contract_branches: dict[str, set[str]] = {}
    for index, requirement in enumerate(contract["requirements"]):
        if not isinstance(requirement, Mapping):
            raise MechanismValidationError(
                "VALIDATION_SCOPE_CONTRACT_CASE_MISSING",
                "iteration-task acceptance contract requirement "
                f"{index} is not an object",
            )
        load_case_id = requirement.get("test_load_case_id")
        if not isinstance(load_case_id, str) or not load_case_id:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_CONTRACT_CASE_MISSING",
                "iteration-task acceptance contract requirement "
                f"{index} has no test_load_case_id",
            )
        test_program = requirement.get("test_program")
        if not isinstance(test_program, str) or not test_program:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_CONTRACT_PROGRAM_MISSING",
                "iteration-task acceptance contract requirement "
                f"{index} has no test_program",
            )
        matches = tuple(
            case
            for case in inventory
            if case.canonical_test_load_case_id == load_case_id
            and case.program == test_program
        )
        if not matches:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_CONTRACT_CASE_UNKNOWN",
                f"contract load case {load_case_id!r} with program "
                f"{test_program!r} is not present in plugin.load_cases(plan) "
                f"for capsule {plan.capsule_id!r}",
            )
        if len(matches) != 1:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_CONTRACT_CASE_AMBIGUOUS",
                f"contract load case {load_case_id!r} with program "
                f"{test_program!r} matches multiple plugin cases for capsule "
                f"{plan.capsule_id!r}",
            )
        resolved_case_id = matches[0].load_case_id
        required = requirement.get("required_branches")
        if (
            not isinstance(required, list)
            or not required
            or not all(isinstance(item, str) for item in required)
            or not set(required).issubset(_CONTRACT_BRANCHES)
        ):
            raise MechanismValidationError(
                "VALIDATION_SCOPE_REQUIRED_BRANCHES_INVALID",
                f"contract load case {load_case_id!r} with program "
                f"{test_program!r} has invalid "
                "required_branches",
            )
        if resolved_case_id not in seen:
            seen.add(resolved_case_id)
            selected.append(resolved_case_id)
        contract_branches.setdefault(resolved_case_id, set()).update(required)

    scheduled_branches = {
        load_case_id: (
            ("physical", "ideal_dynamics")
            if "ideal_dynamics" in contract_branches[load_case_id]
            else ("physical",)
        )
        for load_case_id in selected
    }
    return tuple(selected), scheduled_branches


def _final_full_branches(plan: MechanismPlan) -> tuple[str, ...]:
    contract = plan.acceptance_contract
    ideal_dynamics = (
        isinstance(contract, Mapping)
        and contract.get("status") == "PASS"
        and contract.get("certificate_authority") is True
        and any(
            isinstance(requirement, Mapping)
            and "ideal_dynamics"
            in requirement.get("required_branches", [])
            for requirement in contract.get("requirements", [])
        )
    )
    return (
        ("physical", "ideal_dynamics")
        if ideal_dynamics
        else ("physical",)
    )


def resolve_validation_scope(
    plans: Sequence[MechanismPlan],
    load_cases_by_capsule: Mapping[str, Sequence[LoadCase]],
    *,
    mode: str,
) -> ResolvedValidationScope:
    """Resolve a deterministic, hash-bound dispatch scope.

    ``final-full`` selects the complete trusted plugin inventory.  The narrower
    ``iteration-task`` mode accepts no selector argument: its selectors come
    only from an authoritative acceptance contract and must exist in that same
    plugin inventory.
    """

    resolved_mode = validate_validation_mode(mode)
    all_selectors: list[dict[str, str]] = []
    selected_selectors: list[dict[str, str]] = []
    omitted_selectors: list[dict[str, str]] = []
    branch_records: list[dict[str, Any]] = []
    selected_by_capsule: dict[str, tuple[str, ...]] = {}
    branches_by_selector: dict[tuple[str, str], tuple[str, ...]] = {}

    seen_capsules: set[str] = set()
    for plan in plans:
        capsule_id = plan.capsule_id
        if capsule_id in seen_capsules:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_CAPSULE_DUPLICATE",
                f"duplicate mechanism capsule in dispatch: {capsule_id!r}",
            )
        seen_capsules.add(capsule_id)
        cases = load_cases_by_capsule.get(capsule_id)
        if cases is None:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_PLUGIN_INVENTORY_MISSING",
                f"plugin load-case inventory is missing for {capsule_id!r}",
            )
        case_inventory = tuple(cases)
        inventory = tuple(case.load_case_id for case in case_inventory)
        if not inventory:
            raise MechanismValidationError(
                "VALIDATION_SCOPE_PLUGIN_INVENTORY_EMPTY",
                f"plugin.load_cases(plan) returned no cases for {capsule_id!r}",
            )
        if len(set(inventory)) != len(inventory):
            raise MechanismValidationError(
                "VALIDATION_SCOPE_PLUGIN_CASE_DUPLICATE",
                f"plugin.load_cases(plan) returned duplicate IDs for "
                f"{capsule_id!r}",
            )

        if resolved_mode == VALIDATION_MODE_FINAL_FULL:
            selected_ids = inventory
            per_case_branches = {
                load_case_id: _final_full_branches(plan)
                for load_case_id in selected_ids
            }
        else:
            selected_ids, per_case_branches = _iteration_selection(
                plan,
                inventory=case_inventory,
            )
        selected_set = set(selected_ids)
        selected_by_capsule[capsule_id] = selected_ids

        all_selectors.extend(
            _selector(capsule_id, load_case_id)
            for load_case_id in inventory
        )
        selected_selectors.extend(
            _selector(capsule_id, load_case_id)
            for load_case_id in selected_ids
        )
        omitted_selectors.extend(
            _selector(capsule_id, load_case_id)
            for load_case_id in inventory
            if load_case_id not in selected_set
        )
        for load_case_id in selected_ids:
            branches = per_case_branches[load_case_id]
            branches_by_selector[(capsule_id, load_case_id)] = branches
            branch_records.append(
                {
                    **_selector(capsule_id, load_case_id),
                    "branches": list(branches),
                }
            )

    scope: dict[str, Any] = {
        "schema": VALIDATION_SCOPE_SCHEMA,
        "mode": resolved_mode,
        "selector_authority": (
            FINAL_FULL_SELECTOR_AUTHORITY
            if resolved_mode == VALIDATION_MODE_FINAL_FULL
            else ITERATION_TASK_SELECTOR_AUTHORITY
        ),
        "all": all_selectors,
        "selected": selected_selectors,
        "omitted": omitted_selectors,
        "required_branches": branch_records,
        # A dispatch scope never issues a certificate.  A complete final
        # scope may be consumed by the separate terminal authority.
        "certificate_authority": False,
        "terminal_evidence_eligible": (
            resolved_mode == VALIDATION_MODE_FINAL_FULL
        ),
    }
    scope["scope_sha256"] = canonical_sha256(scope)
    return ResolvedValidationScope(
        document=scope,
        selected_case_ids_by_capsule=selected_by_capsule,
        required_branches_by_selector=branches_by_selector,
    )
