from __future__ import annotations

"""Deterministic semantic binding from task roles to package objects."""

import hashlib
import json
import re
from typing import Any, Mapping

from .acceptance_contract import metric_definition_for_requirement


SEMANTIC_BINDING_SCHEMA = "semantic_binding_v1"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "assembly",
        "complete",
        "machine",
        "mechanism",
        "of",
        "part",
        "the",
    }
)


class SemanticBindingError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        requirement_id: str,
        evidence: Any = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.requirement_id = requirement_id
        self.evidence = evidence


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(value.casefold())
        if token not in _STOP_WORDS and len(token) > 1
    }


def _capsule_item(
    capsules: Mapping[str, Mapping[str, Any]],
    capsule_id: str,
    collection: str,
    item_key: str,
    item_id: str,
) -> Mapping[str, Any] | None:
    capsule = capsules.get(capsule_id)
    if not isinstance(capsule, Mapping):
        return None
    for item in capsule.get(collection, []):
        if isinstance(item, Mapping) and item.get(item_key) == item_id:
            return item
    return None


def _group_evidence(
    assembly: Mapping[str, Any],
    group_id: str,
) -> dict[str, Any]:
    part_ids: list[str] = []
    source_solid_ids: list[str] = []
    searchable: list[str] = [group_id]
    for part in assembly.get("parts", []):
        if not isinstance(part, Mapping) or part.get("rigid_group_id") != group_id:
            continue
        part_id = part.get("part_id")
        if isinstance(part_id, str):
            part_ids.append(part_id)
            searchable.append(part_id)
        for solid_id in part.get("source_solid_ids", []):
            if isinstance(solid_id, str):
                source_solid_ids.append(solid_id)
    return {
        "rigid_group_id": group_id,
        "part_ids": sorted(set(part_ids)),
        "source_solid_ids": sorted(set(source_solid_ids)),
        "searchable_text": " ".join(searchable),
    }


def _observable_candidates(
    refs: Any,
    capsules: Mapping[str, Mapping[str, Any]],
    assembly: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, Mapping):
            continue
        capsule_id = ref.get("capsule_id")
        observable_id = ref.get("observable_id")
        if not isinstance(capsule_id, str) or not isinstance(observable_id, str):
            continue
        observable = _capsule_item(
            capsules,
            capsule_id,
            "observables",
            "observable_id",
            observable_id,
        )
        if observable is None:
            continue
        group_id = observable.get("rigid_group_id")
        if not isinstance(group_id, str):
            continue
        capsule = capsules[capsule_id]
        group = _group_evidence(assembly, group_id)
        searchable = " ".join(
            (
                capsule_id,
                observable_id,
                group_id,
                str(observable.get("frame_id") or ""),
                group["searchable_text"],
            )
        )
        candidates.append(
            {
                "capsule_id": capsule_id,
                "observable_id": observable_id,
                "quantity": observable.get("quantity"),
                "frame_id": observable.get("frame_id"),
                "rigid_group_id": group_id,
                "part_ids": group["part_ids"],
                "source_solid_ids": group["source_solid_ids"],
                "searchable_text": searchable,
            }
        )
    return candidates


def _input_candidates(
    refs: Any,
    capsules: Mapping[str, Mapping[str, Any]],
    assembly: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for ref in refs if isinstance(refs, list) else []:
        if not isinstance(ref, Mapping):
            continue
        capsule_id = ref.get("capsule_id")
        port_id = ref.get("port_id")
        if not isinstance(capsule_id, str) or not isinstance(port_id, str):
            continue
        port = _capsule_item(
            capsules,
            capsule_id,
            "input_ports",
            "port_id",
            port_id,
        )
        if port is None:
            continue
        group_id = port.get("rigid_group_id")
        if not isinstance(group_id, str):
            continue
        capsule = capsules[capsule_id]
        group = _group_evidence(assembly, group_id)
        searchable = " ".join(
            (
                capsule_id,
                port_id,
                group_id,
                str(port.get("frame_id") or ""),
                group["searchable_text"],
            )
        )
        candidates.append(
            {
                "capsule_id": capsule_id,
                "port_id": port_id,
                "wrench_type": port.get("wrench_type"),
                "direction_abs": port.get("direction_abs"),
                "frame_id": port.get("frame_id"),
                "rigid_group_id": group_id,
                "part_ids": group["part_ids"],
                "source_solid_ids": group["source_solid_ids"],
                "searchable_text": searchable,
            }
        )
    return candidates


def _score(
    role: str,
    candidate: Mapping[str, Any],
    *,
    compatible_quantities: frozenset[str] | None = None,
) -> int:
    role_tokens = _tokens(role)
    candidate_tokens = _tokens(str(candidate.get("searchable_text") or ""))
    overlap = role_tokens & candidate_tokens
    score = 4 * len(overlap)
    normalized_role = " ".join(sorted(role_tokens))
    normalized_candidate = " ".join(sorted(candidate_tokens))
    if normalized_role and normalized_role in normalized_candidate:
        score += 5
    if (
        compatible_quantities is not None
        and candidate.get("quantity") in compatible_quantities
    ):
        score += 2
    return score


def _choose(
    role: str,
    candidates: list[dict[str, Any]],
    *,
    requirement_id: str,
    kind: str,
    compatible_quantities: frozenset[str] | None = None,
    required_quantity: str | None = None,
    exclude_observable_id: str | None = None,
    exclude_rigid_group_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    quantity_rejections = []
    for candidate in candidates:
        if (
            exclude_observable_id is not None
            and candidate.get("observable_id") == exclude_observable_id
        ):
            continue
        if (
            exclude_rigid_group_id is not None
            and candidate.get("rigid_group_id") == exclude_rigid_group_id
        ):
            continue
        if (
            required_quantity is not None
            and candidate.get("quantity") != required_quantity
        ):
            quantity_rejections.append(
                {
                    "score": None,
                    "capsule_id": candidate.get("capsule_id"),
                    "object_id": candidate.get(
                        "observable_id",
                        candidate.get("port_id"),
                    ),
                    "rigid_group_id": candidate.get("rigid_group_id"),
                    "quantity": candidate.get("quantity"),
                    "rejection": "frozen_quantity_mismatch",
                }
            )
            continue
        score = _score(
            role,
            candidate,
            compatible_quantities=compatible_quantities,
        )
        scored.append(
            {
                "score": score,
                "capsule_id": candidate.get("capsule_id"),
                "object_id": candidate.get(
                    "observable_id",
                    candidate.get("port_id"),
                ),
                "rigid_group_id": candidate.get("rigid_group_id"),
                "quantity": candidate.get("quantity"),
            }
        )
    scored.sort(
        key=lambda item: (
            -int(item["score"]),
            str(item["capsule_id"]),
            str(item["object_id"]),
        )
    )
    if not scored or int(scored[0]["score"]) <= 0:
        raise SemanticBindingError(
            "SEMANTIC_BINDING_MISSING",
            f"No mapped {kind} matches the required semantic role.",
            requirement_id=requirement_id,
            evidence={
                "role": role,
                "required_quantity": required_quantity,
                "candidates": scored or quantity_rejections[:5],
            },
        )
    if len(scored) > 1 and int(scored[0]["score"]) - int(scored[1]["score"]) < 2:
        raise SemanticBindingError(
            "SEMANTIC_BINDING_AMBIGUOUS",
            f"The best {kind} candidate is not clearly better than the runner-up.",
            requirement_id=requirement_id,
            evidence={"role": role, "candidates": scored[:5]},
        )
    winner_id = scored[0]["object_id"]
    winner = next(
        item
        for item in candidates
        if item.get("observable_id", item.get("port_id")) == winner_id
        and item.get("capsule_id") == scored[0]["capsule_id"]
    )
    result = dict(winner)
    result.pop("searchable_text", None)
    result["semantic_role"] = role
    result["selection_score"] = scored[0]["score"]
    return result, scored[:5]


def bind_acceptance_contract(
    contract: Mapping[str, Any],
    *,
    task_mapping: Mapping[str, Any],
    capsules: Mapping[str, Mapping[str, Any]],
    assembly: Mapping[str, Any],
) -> dict[str, Any]:
    if contract.get("status") == "LEGACY_UNTYPED":
        payload = {
            "schema_version": SEMANTIC_BINDING_SCHEMA,
            "status": "LEGACY_UNTYPED",
            "certificate_authority": False,
            "contract_sha256": contract.get("contract_sha256"),
            "bindings": [],
            "failure_reasons": ["typed_acceptance_requirements_missing"],
        }
        payload["semantic_binding_sha256"] = _canonical_sha256(payload)
        return payload
    if (
        contract.get("status") != "PASS"
        or contract.get("certificate_authority") is not True
    ):
        raise SemanticBindingError(
            "ACCEPTANCE_CONTRACT_NOT_AUTHORITATIVE",
            "Only a passing typed acceptance contract can be semantically bound.",
            requirement_id="contract",
            evidence={"status": contract.get("status")},
        )
    mappings = {
        item.get("task_id"): item
        for item in task_mapping.get("task_mappings", [])
        if isinstance(item, Mapping) and isinstance(item.get("task_id"), str)
    }
    bindings: list[dict[str, Any]] = []
    for requirement in contract.get("requirements", []):
        if not isinstance(requirement, Mapping):
            continue
        requirement_id = str(requirement.get("requirement_id"))
        task_id = requirement.get("task_id")
        mapping = mappings.get(task_id)
        if not isinstance(mapping, Mapping):
            raise SemanticBindingError(
                "SEMANTIC_TASK_MAPPING_MISSING",
                "The typed requirement has no package task mapping.",
                requirement_id=requirement_id,
                evidence={"task_id": task_id},
            )
        metric_id = str(requirement.get("metric_id"))
        definition = metric_definition_for_requirement(requirement)
        if definition is None:
            raise SemanticBindingError(
                "SEMANTIC_METRIC_UNREGISTERED",
                "The typed requirement references an unknown metric.",
                requirement_id=requirement_id,
                evidence={"metric_id": metric_id},
            )
        frozen_quantity = requirement.get("observable_quantity")
        if not isinstance(frozen_quantity, str):
            frozen_quantity = None
        observable_candidates = _observable_candidates(
            mapping.get("observables"),
            capsules,
            assembly,
        )
        target, target_scores = _choose(
            str(requirement.get("target_role") or ""),
            observable_candidates,
            requirement_id=requirement_id,
            kind="observable",
            compatible_quantities=definition.compatible_quantities,
            required_quantity=frozen_quantity,
        )
        reference = None
        reference_scores = None
        reference_role = requirement.get("reference_role")
        if isinstance(reference_role, str):
            reference, reference_scores = _choose(
                reference_role,
                observable_candidates,
                requirement_id=requirement_id,
                kind="reference observable",
                compatible_quantities=definition.compatible_quantities,
                required_quantity=frozen_quantity,
                exclude_observable_id=str(target["observable_id"]),
                exclude_rigid_group_id=str(target["rigid_group_id"]),
            )
        input_candidates = _input_candidates(
            mapping.get("inputs"),
            capsules,
            assembly,
        )
        input_role = requirement.get("input_role")
        if isinstance(input_role, str):
            selected_input, input_scores = _choose(
                input_role,
                input_candidates,
                requirement_id=requirement_id,
                kind="input",
            )
        elif len(input_candidates) == 1:
            selected_input = dict(input_candidates[0])
            selected_input.pop("searchable_text", None)
            selected_input["semantic_role"] = "sole_mapped_input"
            selected_input["selection_score"] = None
            input_scores = []
        else:
            raise SemanticBindingError(
                "SEMANTIC_INPUT_AMBIGUOUS",
                "Multiple mapped inputs require an explicit input semantic role.",
                requirement_id=requirement_id,
                evidence={
                    "input_count": len(input_candidates),
                    "input_ids": [
                        item.get("port_id") for item in input_candidates
                    ],
                },
            )
        if (
            requirement.get("allow_direct_output_actuation") is False
            and selected_input.get("rigid_group_id")
            == target.get("rigid_group_id")
        ):
            raise SemanticBindingError(
                "DIRECT_OUTPUT_ACTUATION_FORBIDDEN",
                "The selected input acts directly on the output rigid group.",
                requirement_id=requirement_id,
                evidence={
                    "input_port_id": selected_input.get("port_id"),
                    "observable_id": target.get("observable_id"),
                    "rigid_group_id": target.get("rigid_group_id"),
                },
            )
        entry = {
            "requirement_id": requirement_id,
            "task_id": task_id,
            "metric_id": metric_id,
            "target": target,
            "reference": reference,
            "input": selected_input,
            "selection_evidence": {
                "target_candidates": target_scores,
                "reference_candidates": reference_scores,
                "input_candidates": input_scores,
                "frozen_observable_quantity": frozen_quantity,
                "required_margin": 2,
            },
            "hard_checks": {
                "task_mapping_member": True,
                "quantity_compatible": (
                    (
                        frozen_quantity is None
                        and target.get("quantity") in definition.compatible_quantities
                    )
                    or (
                        frozen_quantity is not None
                        and target.get("quantity") == frozen_quantity
                    )
                    or metric_id
                    in {
                        "actuation_provenance",
                        "expected_contact",
                        "forbidden_penetration_m",
                    }
                ),
                "direct_output_actuation_allowed": requirement.get(
                    "allow_direct_output_actuation"
                ),
            },
            "status": "PASS",
        }
        if entry["hard_checks"]["quantity_compatible"] is not True:
            raise SemanticBindingError(
                "SEMANTIC_QUANTITY_INCOMPATIBLE",
                "The selected observable cannot feed the registered metric.",
                requirement_id=requirement_id,
                evidence={
                    "metric_id": metric_id,
                    "quantity": target.get("quantity"),
                },
            )
        bindings.append(entry)
    payload = {
        "schema_version": SEMANTIC_BINDING_SCHEMA,
        "status": "PASS",
        "certificate_authority": True,
        "contract_sha256": contract.get("contract_sha256"),
        "binding_count": len(bindings),
        "bindings": bindings,
        "failure_reasons": [],
    }
    payload["semantic_binding_sha256"] = _canonical_sha256(payload)
    return payload


def binding_for_tasks(
    binding: Mapping[str, Any],
    task_ids: set[str],
) -> dict[str, Any]:
    bindings = [
        dict(item)
        for item in binding.get("bindings", [])
        if isinstance(item, Mapping) and item.get("task_id") in task_ids
    ]
    payload = {
        key: binding[key]
        for key in (
            "schema_version",
            "status",
            "certificate_authority",
            "contract_sha256",
            "semantic_binding_sha256",
        )
        if key in binding
    }
    payload["task_ids"] = sorted(task_ids)
    payload["bindings"] = bindings
    payload["task_binding_sha256"] = _canonical_sha256(
        {
            "parent": binding.get("semantic_binding_sha256"),
            "bindings": bindings,
        }
    )
    return payload


def verify_semantic_binding(binding: Mapping[str, Any]) -> None:
    """Verify the full validator-owned role binding digest."""

    if not isinstance(binding, Mapping):
        raise SemanticBindingError(
            "SEMANTIC_BINDING_INVALID",
            "Semantic binding must be an object.",
            requirement_id="binding",
        )
    recorded = binding.get("semantic_binding_sha256")
    payload = {
        key: value
        for key, value in binding.items()
        if key != "semantic_binding_sha256"
    }
    expected = _canonical_sha256(payload)
    if recorded != expected:
        raise SemanticBindingError(
            "SEMANTIC_BINDING_HASH_MISMATCH",
            "Saved semantic binding content does not match its digest.",
            requirement_id="binding",
            evidence={"expected": expected, "actual": recorded},
        )
    if binding.get("schema_version") != SEMANTIC_BINDING_SCHEMA:
        raise SemanticBindingError(
            "SEMANTIC_BINDING_SCHEMA_INVALID",
            "Semantic binding schema is unsupported.",
            requirement_id="binding",
            evidence={"schema": binding.get("schema_version")},
        )


def verify_task_binding(binding: Mapping[str, Any]) -> None:
    """Verify a capsule-scoped semantic binding digest."""

    if not isinstance(binding, Mapping):
        raise SemanticBindingError(
            "TASK_BINDING_INVALID",
            "Task binding must be an object.",
            requirement_id="binding",
        )
    bindings = [
        dict(item)
        for item in binding.get("bindings") or []
        if isinstance(item, Mapping)
    ]
    expected = _canonical_sha256(
        {
            "parent": binding.get("semantic_binding_sha256"),
            "bindings": bindings,
        }
    )
    if binding.get("task_binding_sha256") != expected:
        raise SemanticBindingError(
            "TASK_BINDING_HASH_MISMATCH",
            "Saved task binding content does not match its digest.",
            requirement_id="binding",
            evidence={
                "expected": expected,
                "actual": binding.get("task_binding_sha256"),
            },
        )
