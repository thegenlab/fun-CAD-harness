from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from .io import canonical_sha256
from .models import MechanismValidationError

COMPLIANT_SUPPORT = "COMPLIANT_SUPPORT"
ANALYTICAL_TRANSMISSION = "ANALYTICAL_TRANSMISSION"
EXPECTED_REAL_CONTACT = "EXPECTED_REAL_CONTACT"
FORBIDDEN_CONTACT = "FORBIDDEN_CONTACT"
KINDS = {COMPLIANT_SUPPORT, ANALYTICAL_TRANSMISSION, EXPECTED_REAL_CONTACT, FORBIDDEN_CONTACT}


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _id(value: Any, field: str) -> str:
    text = str(value or "")
    if not text or any(char.isspace() for char in text):
        raise _fail("COMPONENT_CONTACT_INTENT_INVALID", f"{field} is invalid")
    return text


def collision_component_inventory(geometry: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    bodies = geometry.get("bodies")
    if not isinstance(bodies, list) or not bodies:
        raise _fail("COMPONENT_CONTACT_GEOMETRY_INVALID", "no component inventory")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for body in bodies:
        if not isinstance(body, Mapping):
            raise _fail("COMPONENT_CONTACT_GEOMETRY_INVALID", "invalid body")
        body_id = _id(body.get("body_id"), "body_id")
        components = body.get("components")
        if not isinstance(components, list) or not components:
            raise _fail("COMPONENT_CONTACT_GEOMETRY_INVALID", body_id)
        for component in components:
            if not isinstance(component, Mapping):
                raise _fail("COMPONENT_CONTACT_GEOMETRY_INVALID", body_id)
            solid_id = _id(component.get("solid_id"), "solid_id")
            component_id = _id(
                component.get("component_id") or solid_id,
                "component_id",
            )
            ref = f"{body_id}/{component_id}"
            digest = component.get("sha256") or component.get("derived_sha256")
            if ref in seen:
                raise _fail("COMPONENT_CONTACT_COMPONENT_DUPLICATE", ref)
            if not isinstance(digest, str) or len(digest) != 64:
                raise _fail("COMPONENT_CONTACT_COMPONENT_HASH_MISSING", ref)
            seen.add(ref)
            result.append({
                "component_ref": ref,
                "body_id": body_id,
                "component_id": component_id,
                "solid_id": solid_id,
                "source_solid_id": solid_id,
                "component_sha256": digest,
                "fixed": bool(body.get("fixed")),
            })
    return tuple(result)


def contact_intent_declaration(
    *, intent_id: str, kind: str, component_a_refs: Sequence[str],
    component_b_refs: Sequence[str], source: str, interface_id: str | None = None,
    active_phase_ids: Sequence[str] = (), max_penetration_m: float | None = None,
    max_separation_m: float | None = None,
    binding_scope: str = "explicit_component_subset",
) -> dict[str, Any]:
    if kind not in KINDS:
        raise _fail("COMPONENT_CONTACT_INTENT_KIND_INVALID", kind)
    left = sorted({_id(v, "component_a_refs") for v in component_a_refs})
    right = sorted({_id(v, "component_b_refs") for v in component_b_refs})
    phases = sorted({_id(v, "active_phase_ids") for v in active_phase_ids})
    if not left or not right:
        raise _fail("COMPONENT_CONTACT_COMPONENTS_REQUIRED", intent_id)
    if kind == EXPECTED_REAL_CONTACT and not phases:
        raise _fail("EXPECTED_CONTACT_PHASES_REQUIRED", intent_id)
    threshold = None if max_penetration_m is None else float(max_penetration_m)
    if threshold is not None and not 0.0 <= threshold <= 0.01:
        raise _fail("COMPONENT_CONTACT_PENETRATION_LIMIT_INVALID", intent_id)
    separation = None if max_separation_m is None else float(max_separation_m)
    if separation is not None and (
        kind != EXPECTED_REAL_CONTACT or not 0.0 <= separation <= 0.001
    ):
        raise _fail("COMPONENT_CONTACT_SEPARATION_LIMIT_INVALID", intent_id)
    result = {
        "intent_id": _id(intent_id, "intent_id"), "kind": kind,
        "component_a_refs": left, "component_b_refs": right,
        "interface_id": None if interface_id is None else _id(interface_id, "interface_id"),
        "active_phase_ids": phases, "max_penetration_m": threshold,
        "source": _id(source, "source"), "binding_scope": _id(binding_scope, "binding_scope"),
        "counts_as_expected_contact": kind == EXPECTED_REAL_CONTACT,
        "allows_whole_body_pair_collision_exclusion": False,
    }
    if separation is not None:
        result["max_separation_m"] = separation
        result["requires_same_sample_runtime_body_pair_contact"] = True
    return result


def validate_contact_intent_declarations(
    declarations: Iterable[Mapping[str, Any]], *, collision_geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    known = {str(i["component_ref"]): i for i in collision_component_inventory(collision_geometry)}
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in declarations:
        if not isinstance(raw, Mapping):
            raise _fail("COMPONENT_CONTACT_INTENT_INVALID", "not an object")
        item = deepcopy(dict(raw))
        intent_id = _id(item.get("intent_id"), "intent_id")
        kind = str(item.get("kind") or "")
        left = tuple(str(v) for v in item.get("component_a_refs") or ())
        right = tuple(str(v) for v in item.get("component_b_refs") or ())
        if intent_id in seen:
            raise _fail("COMPONENT_CONTACT_INTENT_DUPLICATE", intent_id)
        if kind not in KINDS:
            raise _fail("COMPONENT_CONTACT_INTENT_KIND_INVALID", kind)
        if not left or not right or set(left) - set(known) or set(right) - set(known):
            raise _fail("COMPONENT_CONTACT_BINDING_INVALID", intent_id)
        if set(left) & set(right):
            raise _fail("COMPONENT_CONTACT_SELF_PAIR", intent_id)
        lb = {str(known[v]["body_id"]) for v in left}
        rb = {str(known[v]["body_id"]) for v in right}
        if len(lb) != 1 or len(rb) != 1 or lb == rb:
            raise _fail("COMPONENT_CONTACT_BODY_PAIR_INVALID", intent_id)
        phases = tuple(str(v) for v in item.get("active_phase_ids") or ())
        if kind == EXPECTED_REAL_CONTACT and not phases:
            raise _fail("EXPECTED_CONTACT_PHASES_REQUIRED", intent_id)
        if item.get("counts_as_expected_contact") is not (kind == EXPECTED_REAL_CONTACT):
            raise _fail("COMPONENT_CONTACT_CLAIM_INVALID", intent_id)
        if item.get("allows_whole_body_pair_collision_exclusion") is not False:
            raise _fail("PHYSICAL_BODY_PAIR_COLLISION_EXCLUSION_FORBIDDEN", intent_id)
        threshold = item.get("max_penetration_m")
        if threshold is not None and not 0.0 <= float(threshold) <= 0.01:
            raise _fail("COMPONENT_CONTACT_PENETRATION_LIMIT_INVALID", intent_id)
        separation = item.get("max_separation_m")
        if separation is not None and (
            kind != EXPECTED_REAL_CONTACT
            or not 0.0 <= float(separation) <= 0.001
            or item.get("requires_same_sample_runtime_body_pair_contact") is not True
        ):
            raise _fail("COMPONENT_CONTACT_SEPARATION_LIMIT_INVALID", intent_id)
        _id(item.get("source"), "source")
        _id(item.get("binding_scope"), "binding_scope")
        seen.add(intent_id)
        result.append(item)
    return tuple(result)


def build_component_contact_intent_policy(
    interfaces: Sequence[Mapping[str, Any]], geometry: Mapping[str, Any], *,
    phase_ids: Sequence[str], explicit_declarations: Sequence[Mapping[str, Any]] = (),
    source: str = "validator_component_binder_v1",
) -> dict[str, Any]:
    inventory = collision_component_inventory(geometry)
    by_body: dict[str, list[str]] = {}
    for item in inventory:
        by_body.setdefault(str(item["body_id"]), []).append(str(item["component_ref"]))
    declarations = [deepcopy(dict(i)) for i in explicit_declarations]
    covered = {str(i.get("interface_id")) for i in declarations if i.get("interface_id")}
    from .collision_policy import (
        ANALYTICAL_TRANSMISSION as INTERFACE_ANALYTICAL,
        COMPLIANT_INTERFACE_PHYSICAL, CONTACT_PHYSICAL, validate_interface_declarations,
    )
    phases = tuple(dict.fromkeys(_id(v, "phase_ids") for v in phase_ids))
    for interface in validate_interface_declarations(interfaces, valid_body_ids=by_body):
        iid = str(interface["interface_id"])
        if iid in covered:
            continue
        classification = str(interface["classification"])
        if classification == CONTACT_PHYSICAL:
            kind, active = EXPECTED_REAL_CONTACT, phases
        elif classification == COMPLIANT_INTERFACE_PHYSICAL:
            kind, active = COMPLIANT_SUPPORT, ()
        elif classification == INTERFACE_ANALYTICAL:
            kind, active = ANALYTICAL_TRANSMISSION, ()
        else:
            raise _fail("COMPONENT_CONTACT_INTERFACE_CLASSIFICATION_UNSUPPORTED", iid)
        gate = interface.get("transmission_gate")
        parameters = interface.get("parameters")
        explicit_left = (
            parameters.get("contact_component_a_refs")
            if isinstance(parameters, Mapping)
            else None
        )
        explicit_right = (
            parameters.get("contact_component_b_refs")
            if isinstance(parameters, Mapping)
            else None
        )
        explicit_separation = (
            parameters.get("contact_evidence_max_separation_m")
            if isinstance(parameters, Mapping)
            else None
        )
        explicit_contact_binding = any(
            item is not None
            for item in (explicit_left, explicit_right, explicit_separation)
        )
        if (
            classification in {
                INTERFACE_ANALYTICAL,
                COMPLIANT_INTERFACE_PHYSICAL,
            }
            and isinstance(gate, Mapping)
        ):
            local_left = tuple(
                sorted(
                    str(item)
                    for item in gate["endpoint_a"][
                        "local_collision_component_refs"
                    ]
                )
            )
            local_right = tuple(
                sorted(
                    str(item)
                    for item in gate["endpoint_b"][
                        "local_collision_component_refs"
                    ]
                )
            )
            left = local_left or tuple(
                sorted(by_body[str(interface["body_a_id"])])
            )
            right = local_right or tuple(
                sorted(by_body[str(interface["body_b_id"])])
            )
            if local_left and local_right:
                binding_scope = "validator_step_bound_interface_components"
            elif not local_left and not local_right:
                binding_scope = (
                    "body_owned_exact_component_subset_no_suppression"
                )
            else:
                binding_scope = (
                    "validator_step_bound_partial_collision_suppression"
                )
            max_separation_m = None
        elif classification == CONTACT_PHYSICAL and explicit_contact_binding:
            if not (
                isinstance(explicit_left, Sequence)
                and not isinstance(explicit_left, (str, bytes))
                and explicit_left
                and isinstance(explicit_right, Sequence)
                and not isinstance(explicit_right, (str, bytes))
                and explicit_right
                and explicit_separation is not None
            ):
                raise _fail(
                    "COMPONENT_CONTACT_EXPLICIT_BINDING_INVALID",
                    iid,
                )
            left = tuple(sorted(_id(item, "contact_component_a_refs") for item in explicit_left))
            right = tuple(sorted(_id(item, "contact_component_b_refs") for item in explicit_right))
            if set(left) - set(by_body[str(interface["body_a_id"])]) or set(
                right
            ) - set(by_body[str(interface["body_b_id"])]):
                raise _fail(
                    "COMPONENT_CONTACT_EXPLICIT_BINDING_INVALID",
                    iid,
                )
            binding_scope = "validator_step_bound_explicit_contact_components"
            max_separation_m = float(explicit_separation)
        else:
            left = tuple(sorted(by_body[str(interface["body_a_id"])]))
            right = tuple(sorted(by_body[str(interface["body_b_id"])]))
            binding_scope = (
                "single_component_exact"
                if len(left) == len(right) == 1
                else "body_owned_exact_component_subset"
            )
            max_separation_m = None
        declarations.append(contact_intent_declaration(
            intent_id=f"interface-{iid}", kind=kind, component_a_refs=left,
            component_b_refs=right, source=source, interface_id=iid, active_phase_ids=active,
            max_penetration_m=(
                float(geometry.get("interpenetration_tolerance_m", 0.00035))
                if kind == EXPECTED_REAL_CONTACT else None
            ),
            max_separation_m=max_separation_m,
            binding_scope=binding_scope,
        ))
    validated = validate_contact_intent_declarations(declarations, collision_geometry=geometry)
    return {
        "schema_version": "component_contact_intent_v1", "source": _id(source, "source"),
        "package_binding_sha256": geometry.get("package_binding_sha256"),
        "assembly_sha256": geometry.get("assembly_sha256"),
        "component_inventory_sha256": canonical_sha256(list(inventory)),
        "declarations_sha256": canonical_sha256(list(validated)),
        "declarations": list(validated),
        "implicit_rule": (
            "undeclared inter-body solid pairs remain collision-enabled; "
            "penetration beyond saved-pose tolerance is forbidden"
        ),
        "whole_body_pair_collision_exclusion_allowed": False,
        "runtime_body_pair_contacts_can_prove_component_contact": False,
    }


def validate_component_contact_intent_policy(
    policy: Mapping[str, Any], *, collision_geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    if policy.get("schema_version") != "component_contact_intent_v1":
        raise _fail("COMPONENT_CONTACT_POLICY_SCHEMA_INVALID", "unsupported schema")
    if policy.get("whole_body_pair_collision_exclusion_allowed") is not False:
        raise _fail("PHYSICAL_BODY_PAIR_COLLISION_EXCLUSION_FORBIDDEN", "policy")
    inventory = collision_component_inventory(collision_geometry)
    if policy.get("component_inventory_sha256") != canonical_sha256(list(inventory)):
        raise _fail("COMPONENT_CONTACT_INVENTORY_BINDING_INVALID", "collider binding")
    if policy.get("package_binding_sha256") != collision_geometry.get("package_binding_sha256"):
        raise _fail("COMPONENT_CONTACT_PACKAGE_BINDING_INVALID", "package binding")
    declarations = validate_contact_intent_declarations(
        policy.get("declarations") or (), collision_geometry=collision_geometry,
    )
    if policy.get("declarations_sha256") != canonical_sha256(list(declarations)):
        raise _fail("COMPONENT_CONTACT_DECLARATION_BINDING_INVALID", "declaration binding")
    return declarations


def _event_pair(event: Mapping[str, Any]) -> frozenset[str] | None:
    pair = event.get("solid_pair")
    if not isinstance(pair, list) or len(pair) != 2:
        return None
    values = frozenset(str(v) for v in pair)
    return values if len(values) == 2 else None


def audit_component_contact_intents(
    policy: Mapping[str, Any], geometry_audit: Mapping[str, Any], *,
    collision_geometry: Mapping[str, Any],
    runtime_contact_pair_counts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    declarations = validate_component_contact_intent_policy(
        policy, collision_geometry=collision_geometry,
    )
    raw_events = geometry_audit.get("contact_events")
    if not isinstance(raw_events, list):
        raise _fail("COMPONENT_CONTACT_EVENT_EVIDENCE_MISSING", "no saved-pose contacts")
    known = {str(i["component_ref"]) for i in collision_component_inventory(collision_geometry)}
    events: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for raw in raw_events:
        pair = _event_pair(raw) if isinstance(raw, Mapping) else None
        if pair is None or not pair <= known:
            ambiguous.append({"reason": "solid_pair_not_hash_bound", "event": deepcopy(raw)})
        else:
            events.append(deepcopy(dict(raw)))
    outcomes: list[dict[str, Any]] = []
    failures: list[str] = []
    covered: set[frozenset[str]] = set()
    for item in declarations:
        pairs = {frozenset((a, b)) for a in item["component_a_refs"] for b in item["component_b_refs"]}
        covered.update(pairs)
        matching = [e for e in events if _event_pair(e) in pairs]
        phases = set(str(v) for v in item.get("active_phase_ids") or ())
        in_phase = [e for e in matching if not phases or str(e.get("phase_id")) in phases]
        out_phase = [e for e in matching if phases and str(e.get("phase_id")) not in phases]
        kind = str(item["kind"])
        passed, reasons = True, []
        threshold = item.get("max_penetration_m")
        excessive = [
            event
            for event in matching
            if threshold is not None
            and float(event.get("penetration_depth_m", 0.0)) > float(threshold)
        ]
        if kind == EXPECTED_REAL_CONTACT:
            status = "EXPECTED_CONTACT_OBSERVED"
            if not in_phase:
                passed, status = False, "MISSING_EXPECTED_CONTACT"
                reasons.append("expected_component_contact_missing")
            if out_phase:
                passed, status = False, "CONTACT_OUTSIDE_DECLARED_PHASE"
                reasons.append("component_contact_outside_declared_phase")
            if excessive:
                passed, status = False, "CONTACT_PENETRATION_LIMIT_EXCEEDED"
                reasons.append("component_contact_penetration_exceeds_limit")
        elif kind == FORBIDDEN_CONTACT:
            status = "FORBIDDEN_CONTACT_CLEAR"
            if in_phase:
                passed, status = False, "FORBIDDEN_CONTACT_OBSERVED"
                reasons.append("forbidden_component_contact_observed")
        elif kind == COMPLIANT_SUPPORT:
            status = "SUPPORT_CONTACT_OBSERVED" if matching else "SUPPORT_DECLARED"
        else:
            status = "ANALYTICAL_PAIR_CONTACT_OBSERVED" if matching else "ANALYTICAL_TRANSMISSION_DECLARED"
        for reason in reasons:
            if reason not in failures:
                failures.append(reason)
        outcomes.append({
            "intent_id": item["intent_id"], "kind": kind, "status": status, "passed": passed,
            "matched_saved_pose_event_count": len(matching),
            "in_phase_event_count": len(in_phase), "out_of_phase_event_count": len(out_phase),
            "excessive_penetration_event_count": len(excessive),
            "max_observed_penetration_m": max(
                (float(event.get("penetration_depth_m", 0.0)) for event in matching),
                default=0.0,
            ),
            "counts_as_expected_contact": kind == EXPECTED_REAL_CONTACT,
            "failure_reasons": reasons,
        })
    if ambiguous:
        failures.append("component_contact_attribution_uncertain")
    runtime_count = 0
    if isinstance(runtime_contact_pair_counts, Mapping):
        runtime_count = sum(int(v) for v in runtime_contact_pair_counts.values() if isinstance(v, (int, float)))
    return {
        "schema_version": "component_contact_intent_audit_v1", "passed": not failures,
        "failure_reasons": failures, "saved_pose_contact_event_count": len(events),
        "ambiguous_attribution_event_count": len(ambiguous), "ambiguous_events": ambiguous[:12],
        "undeclared_contact_event_count": sum(1 for e in events if _event_pair(e) not in covered),
        "undeclared_contacts_are_not_expected_contact_evidence": True,
        "runtime_body_pair_contact_observation_count": runtime_count,
        "runtime_body_pair_contacts_used_as_component_evidence": False,
        "outcomes": outcomes,
    }

def build_runtime_component_contact_intent_policy(
    interfaces: Sequence[Mapping[str, Any]],
    collision_manifest: Mapping[str, Any],
    collision_bodies: Sequence[Mapping[str, Any]],
    executed_phases: Sequence[Mapping[str, Any]],
    *,
    explicit_declarations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Create the policy from the exact component inventory written to a trace."""

    geometry = dict(collision_manifest)
    geometry["bodies"] = [deepcopy(dict(item)) for item in collision_bodies]
    phase_ids = [
        str(item.get("phase_id"))
        for item in executed_phases
        if isinstance(item, Mapping) and item.get("phase_id")
    ]
    return build_component_contact_intent_policy(
        interfaces,
        geometry,
        phase_ids=phase_ids,
        explicit_declarations=explicit_declarations,
        source="validator_runtime_component_binder_v1",
    )
