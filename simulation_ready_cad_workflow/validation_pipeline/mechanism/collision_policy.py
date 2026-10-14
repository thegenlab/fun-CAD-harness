from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from .models import MechanismValidationError


CONTACT_PHYSICAL = "CONTACT_PHYSICAL"
COMPLIANT_INTERFACE_PHYSICAL = "COMPLIANT_INTERFACE_PHYSICAL"
ANALYTICAL_TRANSMISSION = "ANALYTICAL_TRANSMISSION"
IDEAL_REDUCED = "IDEAL_REDUCED"
PHYSICAL_INTERFACE_KINDS = {
    CONTACT_PHYSICAL,
    COMPLIANT_INTERFACE_PHYSICAL,
    ANALYTICAL_TRANSMISSION,
}
VALIDATOR_LOCAL_COLLISION_EXCEPTION_SCHEMA = (
    "validator_step_bound_local_collision_exception_v1"
)


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _identifier(value: Any, field: str) -> str:
    text = str(value or "")
    if not text or any(character.isspace() for character in text):
        raise _fail("PHYSICAL_INTERFACE_INVALID", f"{field} is invalid")
    return text


def _validator_local_collision_exception(
    classification: str,
    transmission_gate: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Derive the only allowed analytical-interface collider suppression.

    The exception is not an AI-authored mask.  It is a deterministic projection
    of the already validated, hash-bound STEP endpoints used by the per-step
    transmission gate.  The complete STEP inventory remains in the trace and
    therefore remains subject to saved-pose interpenetration audit.
    """

    if classification != ANALYTICAL_TRANSMISSION or transmission_gate is None:
        return None
    endpoint_a = transmission_gate.get("endpoint_a")
    endpoint_b = transmission_gate.get("endpoint_b")
    if not isinstance(endpoint_a, Mapping) or not isinstance(
        endpoint_b,
        Mapping,
    ):
        raise _fail(
            "PHYSICAL_LOCAL_COLLISION_EXCEPTION_INVALID",
            "analytical transmission endpoints are incomplete",
        )
    endpoint_a_refs = sorted(
        str(item)
        for item in endpoint_a.get("local_collision_component_refs") or ()
    )
    endpoint_b_refs = sorted(
        str(item)
        for item in endpoint_b.get("local_collision_component_refs") or ()
    )
    if not endpoint_a_refs and not endpoint_b_refs:
        return None
    return {
        "schema_version": VALIDATOR_LOCAL_COLLISION_EXCEPTION_SCHEMA,
        "authority": "validator_existing_step_component_inventory_v1",
        "mode": "omit_bound_interface_components_from_runtime_hard_collision",
        "endpoint_a_component_refs": endpoint_a_refs,
        "endpoint_b_component_refs": endpoint_b_refs,
        "full_step_saved_pose_audit_required": True,
    }


def interface_declaration(
    *,
    interface_id: str,
    classification: str,
    body_a_id: str,
    body_b_id: str,
    model: str,
    source: str,
    free_coordinates: Sequence[str] = (),
    parameters: Mapping[str, Any] | None = None,
    transmission_gate: Mapping[str, Any] | None = None,
    historical_runtime_collision_excluded: bool = False,
    historical_evidence: bool = False,
) -> dict[str, Any]:
    validated_transmission_gate = None
    if transmission_gate is not None:
        from .conditional_interface import (
            validate_conditional_transmission_gate,
        )

        validated_transmission_gate = validate_conditional_transmission_gate(
            transmission_gate,
            body_a_id=body_a_id,
            body_b_id=body_b_id,
        )
    payload = {
        "interface_id": _identifier(interface_id, "interface_id"),
        "classification": str(classification),
        "body_a_id": _identifier(body_a_id, "body_a_id"),
        "body_b_id": _identifier(body_b_id, "body_b_id"),
        "model": _identifier(model, "model"),
        "source": _identifier(source, "source"),
        "free_coordinates": [str(item) for item in free_coordinates],
        "parameters": deepcopy(dict(parameters or {})),
        "whole_body_pair_collision_enabled": not historical_runtime_collision_excluded,
        "local_collision_exception": _validator_local_collision_exception(
            str(classification),
            validated_transmission_gate,
        ),
        "full_step_geometry_legality_required": True,
        "historical_runtime_collision_excluded": bool(
            historical_runtime_collision_excluded
        ),
    }
    if historical_evidence:
        payload["historical_evidence"] = True
    if validated_transmission_gate is not None:
        payload["transmission_gate"] = validated_transmission_gate
    validate_interface_declarations(
        (payload,),
        allow_historical_exclusions=(
            historical_runtime_collision_excluded or historical_evidence
        ),
    )
    return payload


def validate_interface_declarations(
    declarations: Iterable[Mapping[str, Any]],
    *,
    valid_body_ids: Iterable[str] | None = None,
    allow_historical_exclusions: bool = False,
) -> tuple[dict[str, Any], ...]:
    valid = None if valid_body_ids is None else {str(item) for item in valid_body_ids}
    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in declarations:
        if not isinstance(raw, Mapping):
            raise _fail("PHYSICAL_INTERFACE_INVALID", "interface entry is not an object")
        item = deepcopy(dict(raw))
        interface_id = _identifier(item.get("interface_id"), "interface_id")
        if interface_id in seen:
            raise _fail(
                "PHYSICAL_INTERFACE_DUPLICATE",
                f"duplicate physical interface: {interface_id}",
            )
        seen.add(interface_id)
        classification = str(item.get("classification") or "")
        if classification not in PHYSICAL_INTERFACE_KINDS:
            raise _fail(
                "PHYSICAL_INTERFACE_CLASSIFICATION_INVALID",
                f"unsupported physical interface classification: {classification}",
            )
        body_a = _identifier(item.get("body_a_id"), "body_a_id")
        body_b = _identifier(item.get("body_b_id"), "body_b_id")
        if body_a == body_b:
            raise _fail(
                "PHYSICAL_INTERFACE_BODY_PAIR_INVALID",
                "a physical interface must join two distinct rigid bodies",
            )
        if valid is not None and ({body_a, body_b} - valid):
            raise _fail(
                "PHYSICAL_INTERFACE_BODY_UNKNOWN",
                f"interface {interface_id!r} refers to an unknown body",
            )
        historical = bool(item.get("historical_runtime_collision_excluded"))
        historical_evidence = bool(item.get("historical_evidence"))
        if historical and not allow_historical_exclusions:
            raise _fail(
                "PHYSICAL_BODY_PAIR_COLLISION_EXCLUSION_FORBIDDEN",
                "a compliant or analytical interface may not disable an entire body pair",
            )
        if item.get("whole_body_pair_collision_enabled") is not (not historical):
            raise _fail(
                "PHYSICAL_INTERFACE_COLLISION_DISCLOSURE_INVALID",
                "interface collision disclosure is internally inconsistent",
            )
        transmission_gate = item.get("transmission_gate")
        if transmission_gate is not None:
            from .conditional_interface import (
                validate_conditional_transmission_gate,
            )

            transmission_gate = validate_conditional_transmission_gate(
                transmission_gate,
                body_a_id=body_a,
                body_b_id=body_b,
            )
            item["transmission_gate"] = transmission_gate
        expected_local_exception = _validator_local_collision_exception(
            classification,
            transmission_gate,
        )
        if item.get("local_collision_exception") != expected_local_exception:
            raise _fail(
                "PHYSICAL_LOCAL_COLLISION_EXCEPTION_INVALID",
                "local collision suppression must be derived exactly from the "
                "validated STEP-bound analytical interface endpoints",
            )
        if item.get("full_step_geometry_legality_required") is not True:
            raise _fail(
                "PHYSICAL_INTERFACE_GEOMETRY_GATE_REQUIRED",
                "every physical interface must retain independent geometry legality audit",
            )
        parameters = item.get("parameters")
        if not isinstance(parameters, Mapping):
            raise _fail(
                "PHYSICAL_INTERFACE_PARAMETERS_INVALID",
                f"interface {interface_id!r} parameters must be an object",
            )
        suppressed_refs = parameters.get(
            "validator_owned_suppressed_component_refs", ()
        )
        if not isinstance(suppressed_refs, (list, tuple)):
            raise _fail(
                "PHYSICAL_LOCAL_COLLISION_EXCEPTION_INVALID",
                f"interface {interface_id!r} component refs must be a list",
            )
        if suppressed_refs and classification != CONTACT_PHYSICAL:
            raise _fail(
                "PHYSICAL_LOCAL_COLLISION_EXCEPTION_INVALID",
                "only a physical contact interface may suppress a bound component",
            )
        for ref in suppressed_refs:
            text = str(ref)
            if "/" not in text or any(character.isspace() for character in text):
                raise _fail(
                    "PHYSICAL_LOCAL_COLLISION_EXCEPTION_INVALID",
                    f"interface {interface_id!r} has an invalid component ref",
                )
        if (
            classification == ANALYTICAL_TRANSMISSION
            and transmission_gate is None
            and not (
                (historical or historical_evidence)
                and allow_historical_exclusions
            )
        ):
            raise _fail(
                "ANALYTICAL_TRANSMISSION_GEOMETRY_GATE_REQUIRED",
                f"analytical interface {interface_id!r} must carry a "
                "STEP-bound, per-physics-step transmission gate",
            )
        if (
            classification == COMPLIANT_INTERFACE_PHYSICAL
            and transmission_gate is None
            and not (
                (historical or historical_evidence)
                and allow_historical_exclusions
            )
        ):
            raise _fail(
                "COMPLIANT_SUPPORT_GEOMETRY_GATE_REQUIRED",
                f"compliant support {interface_id!r} must carry a "
                "STEP-bound, per-physics-step finite action envelope",
            )
        if (
            classification == COMPLIANT_INTERFACE_PHYSICAL
            and transmission_gate is not None
        ):
            from .bounded_support import validate_bounded_support_gate

            validate_bounded_support_gate(transmission_gate)
        _identifier(item.get("model"), "model")
        _identifier(item.get("source"), "source")
        retained.append(item)
    return tuple(retained)


def collision_policy_disclosure(
    declarations: Sequence[Mapping[str, Any]],
    *,
    branch: str,
) -> dict[str, Any]:
    if branch == "physical":
        validated = validate_interface_declarations(declarations)
        policy = "all_inter_body_collision_pairs_enabled"
    else:
        validated = tuple(deepcopy(dict(item)) for item in declarations)
        policy = "replaced_internal_collision_disabled_for_reduced_branch"
    canonical = json.dumps(validated, sort_keys=True, separators=(",", ":"))
    local_exceptions = []
    for item in validated:
        legacy_component_refs = list(
            item.get("parameters", {}).get(
                "validator_owned_suppressed_component_refs",
                (),
            )
        )
        analytical_exception = item.get("local_collision_exception")
        if not legacy_component_refs and not isinstance(
            analytical_exception,
            Mapping,
        ):
            continue
        local_exceptions.append(
            {
                "interface_id": item["interface_id"],
                "component_refs": legacy_component_refs,
                "endpoint_a_component_refs": list(
                    analytical_exception.get(
                        "endpoint_a_component_refs",
                        (),
                    )
                    if isinstance(analytical_exception, Mapping)
                    else ()
                ),
                "endpoint_b_component_refs": list(
                    analytical_exception.get(
                        "endpoint_b_component_refs",
                        (),
                    )
                    if isinstance(analytical_exception, Mapping)
                    else ()
                ),
                "mode": (
                    analytical_exception.get("mode")
                    if isinstance(analytical_exception, Mapping)
                    else "validator_owned_component_suppression"
                ),
                "model": item["model"],
            }
        )
    conditional_interfaces = [
        item for item in validated if item.get("transmission_gate") is not None
    ]
    ungated_analytical_transmissions = [
        item
        for item in validated
        if item.get("classification") == ANALYTICAL_TRANSMISSION
        and item.get("transmission_gate") is None
    ]
    ungated_compliant_supports = [
        item
        for item in validated
        if item.get("classification") == COMPLIANT_INTERFACE_PHYSICAL
        and item.get("transmission_gate") is None
    ]
    return {
        "schema_version": "0.1",
        "branch": branch,
        "whole_body_pair_policy": policy,
        "ai_authored_collision_masks_allowed": False,
        "validator_owned_local_exception_count": len(local_exceptions),
        "validator_owned_local_exceptions": local_exceptions,
        "interface_declaration_count": len(validated),
        "interface_declarations": list(validated),
        "interface_declarations_sha256": hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest(),
        "conditional_interface_count": len(conditional_interfaces),
        "functional_interface_geometry_binding_required": bool(
            conditional_interfaces
        ),
        "analytical_transmission_geometry_gate_required": branch == "physical",
        "compliant_support_geometry_gate_required": branch == "physical",
        "ungated_analytical_transmission_count": len(
            ungated_analytical_transmissions
        ),
        "ungated_compliant_support_count": len(
            ungated_compliant_supports
        ),
        "independent_step_geometry_legality_required": branch == "physical",
    }


def _legacy_body_pair(exclusion: Mapping[str, Any]) -> tuple[str, str] | None:
    candidates = (
        (exclusion.get("parent_body_id"), exclusion.get("child_body_id")),
        (exclusion.get("fixed_body_id"), exclusion.get("moving_body_id")),
        (exclusion.get("body_a_id"), exclusion.get("body_b_id")),
    )
    for left, right in candidates:
        if left and right and str(left) != str(right):
            return str(left), str(right)
    return None


def legacy_interface_declarations(
    physical_trace: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    """Translate old force-model disclosures without rewriting old evidence."""

    declarations: list[dict[str, Any]] = []
    semantics = physical_trace.get("branch_semantics")
    collision = physical_trace.get("collision_geometry")
    sections = [item for item in (semantics, collision) if isinstance(item, Mapping)]
    seen_pairs: set[tuple[str, str, str]] = set()
    for section in sections:
        for field, value in section.items():
            if not (
                isinstance(field, str)
                and field.startswith("excluded_")
                and "collision" in field
                and isinstance(value, list)
            ):
                continue
            for index, raw in enumerate(value):
                if not isinstance(raw, Mapping):
                    continue
                pair = _legacy_body_pair(raw)
                if pair is None:
                    continue
                reason = str(raw.get("reason") or "historical interface")
                lowered = reason.lower()
                analytical = any(
                    token in lowered
                    for token in ("gear", "mesh", "cam", "follower", "clutch", "torsional")
                )
                classification = (
                    ANALYTICAL_TRANSMISSION
                    if analytical
                    else COMPLIANT_INTERFACE_PHYSICAL
                )
                identity = str(
                    raw.get("joint_id")
                    or raw.get("support_id")
                    or f"{field}-{index}-{pair[0]}-{pair[1]}"
                )
                key = tuple(sorted(pair)) + (identity,)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                declarations.append(
                    interface_declaration(
                        interface_id=f"legacy-{identity}",
                        classification=classification,
                        body_a_id=pair[0],
                        body_b_id=pair[1],
                        model=reason.replace(" ", "_"),
                        source="trusted_legacy_trace_migration_v1",
                        historical_runtime_collision_excluded=True,
                        historical_evidence=True,
                    )
                )

    plugin_id = str(physical_trace.get("mechanism_plugin_id") or "")
    if not declarations and plugin_id in {"plain_revolute_v1", "plain_prismatic_v1"}:
        bodies: list[str] = []
        if isinstance(collision, Mapping):
            bodies = [
                str(item.get("body_id"))
                for item in collision.get("bodies") or []
                if isinstance(item, Mapping) and item.get("body_id")
            ]
            if len(bodies) < 2 and isinstance(collision.get("fixed"), Mapping):
                if isinstance(collision.get("moving"), Mapping):
                    bodies = ["fixed_boundary", "moving_group"]
        if len(bodies) >= 2:
            revolute = plugin_id == "plain_revolute_v1"
            declarations.append(
                interface_declaration(
                    interface_id=(
                        "legacy-plain-revolute-bearing"
                        if revolute
                        else "legacy-plain-prismatic-contact"
                    ),
                    classification=(
                        COMPLIANT_INTERFACE_PHYSICAL if revolute else CONTACT_PHYSICAL
                    ),
                    body_a_id=bodies[0],
                    body_b_id=bodies[1],
                    model=(
                        "finite_plain_revolute_bearing_support"
                        if revolute
                        else "step_derived_plain_prismatic_contact"
                    ),
                    source="trusted_legacy_trace_migration_v1",
                    historical_evidence=True,
                )
            )
    return validate_interface_declarations(
        declarations,
        allow_historical_exclusions=True,
    )
