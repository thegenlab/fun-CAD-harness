from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from .models import MechanismValidationError, finite, identifier, vector3


CONDITIONAL_INTERFACE_SCHEMA = "conditional_compliant_interface_v1"
INACTIVE = "INACTIVE"
APPROACHING = "APPROACHING"
ENGAGED = "ENGAGED"
RELEASING = "RELEASING"
INTERFACE_STATES = {INACTIVE, APPROACHING, ENGAGED, RELEASING}
AT_OR_ABOVE = "AT_OR_ABOVE"
AT_OR_BELOW = "AT_OR_BELOW"
ACTIVATION_DIRECTIONS = {AT_OR_ABOVE, AT_OR_BELOW}
EXACT_ZERO = "EXACT_ZERO"


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _component_refs(
    values: Sequence[str],
    *,
    body_id: str,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    if (
        isinstance(values, (str, bytes))
        or not isinstance(values, Sequence)
        or (not values and not allow_empty)
    ):
        raise _fail(
            "CONDITIONAL_INTERFACE_COMPONENT_BINDING_REQUIRED",
            f"{label} must bind at least one original STEP component",
        )
    retained: list[str] = []
    for value in values:
        ref = str(value)
        parts = ref.split("/")
        if (
            len(parts) != 2
            or parts[0] != body_id
            or not all(parts)
            or "\\" in ref
            or any(part in {".", ".."} for part in parts)
            or any(character.isspace() for character in ref)
        ):
            raise _fail(
                "CONDITIONAL_INTERFACE_COMPONENT_BINDING_INVALID",
                f"{label} has an invalid body-owned component ref: {ref!r}",
            )
        retained.append(ref)
    if len(set(retained)) != len(retained):
        raise _fail(
            "CONDITIONAL_INTERFACE_COMPONENT_BINDING_INVALID",
            f"{label} contains duplicate component refs",
        )
    return retained


def conditional_transmission_gate(
    *,
    geometry_plugin_id: str,
    body_a_id: str,
    body_b_id: str,
    endpoint_a_component_refs: Sequence[str],
    endpoint_b_component_refs: Sequence[str],
    endpoint_a_local_collision_component_refs: Sequence[str] | None = None,
    endpoint_b_local_collision_component_refs: Sequence[str] | None = None,
    activation_coordinate: str,
    enter_threshold_si: float,
    release_threshold_si: float,
    proof_source: str,
    activation_direction: str = AT_OR_ABOVE,
    geometry_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Declare a validator-owned, pose-dependent transmission gate.

    The declaration deliberately separates geometric eligibility from the
    constitutive law that computes an effort. A runtime may compute any finite
    compliant effort it has declared, but this gate is the final owner of
    whether that effort reaches either body.
    """

    payload = {
        "schema_version": CONDITIONAL_INTERFACE_SCHEMA,
        "geometry_plugin_id": identifier(
            geometry_plugin_id, "geometry_plugin_id"
        ),
        "endpoint_a": {
            "body_id": identifier(body_a_id, "body_a_id"),
            "component_refs": _component_refs(
                endpoint_a_component_refs,
                body_id=body_a_id,
                label="endpoint_a_component_refs",
            ),
            "local_collision_component_refs": _component_refs(
                (
                    endpoint_a_component_refs
                    if endpoint_a_local_collision_component_refs is None
                    else endpoint_a_local_collision_component_refs
                ),
                body_id=body_a_id,
                label="endpoint_a_local_collision_component_refs",
                allow_empty=True,
            ),
        },
        "endpoint_b": {
            "body_id": identifier(body_b_id, "body_b_id"),
            "component_refs": _component_refs(
                endpoint_b_component_refs,
                body_id=body_b_id,
                label="endpoint_b_component_refs",
            ),
            "local_collision_component_refs": _component_refs(
                (
                    endpoint_b_component_refs
                    if endpoint_b_local_collision_component_refs is None
                    else endpoint_b_local_collision_component_refs
                ),
                body_id=body_b_id,
                label="endpoint_b_local_collision_component_refs",
                allow_empty=True,
            ),
        },
        "activation_coordinate": identifier(
            activation_coordinate, "activation_coordinate"
        ),
        "activation_direction": str(activation_direction),
        "enter_threshold_si": finite(
            enter_threshold_si, "enter_threshold_si"
        ),
        "release_threshold_si": finite(
            release_threshold_si, "release_threshold_si"
        ),
        "geometry_parameters": deepcopy(dict(geometry_parameters or {})),
        "proof_source": identifier(proof_source, "proof_source"),
        "geometry_legality_source": "ORIGINAL_STEP_COMPONENTS",
        "evaluate_every_physics_step": True,
        "inactive_wrench_policy": EXACT_ZERO,
        "capture_reference_on_engage": True,
        "initially_inactive_is_warning": True,
    }
    return validate_conditional_transmission_gate(
        payload,
        body_a_id=body_a_id,
        body_b_id=body_b_id,
    )


def validate_conditional_transmission_gate(
    raw: Mapping[str, Any],
    *,
    body_a_id: str | None = None,
    body_b_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise _fail(
            "CONDITIONAL_INTERFACE_GATE_INVALID",
            "conditional transmission gate must be an object",
        )
    item = deepcopy(dict(raw))
    if item.get("schema_version") != CONDITIONAL_INTERFACE_SCHEMA:
        raise _fail(
            "CONDITIONAL_INTERFACE_SCHEMA_INVALID",
            f"expected {CONDITIONAL_INTERFACE_SCHEMA}",
        )
    identifier(item.get("geometry_plugin_id"), "geometry_plugin_id")
    identifier(item.get("activation_coordinate"), "activation_coordinate")
    identifier(item.get("proof_source"), "proof_source")

    direction = str(item.get("activation_direction") or "")
    if direction not in ACTIVATION_DIRECTIONS:
        raise _fail(
            "CONDITIONAL_INTERFACE_DIRECTION_INVALID",
            f"unsupported activation direction: {direction!r}",
        )
    enter = finite(item.get("enter_threshold_si"), "enter_threshold_si")
    release = finite(item.get("release_threshold_si"), "release_threshold_si")
    if (
        direction == AT_OR_ABOVE
        and enter <= release
        or direction == AT_OR_BELOW
        and enter >= release
    ):
        raise _fail(
            "CONDITIONAL_INTERFACE_HYSTERESIS_INVALID",
            "enter and release thresholds must create a non-zero hysteresis band",
        )

    expected_bodies = (body_a_id, body_b_id)
    for name, expected in zip(("endpoint_a", "endpoint_b"), expected_bodies):
        endpoint = item.get(name)
        if not isinstance(endpoint, Mapping):
            raise _fail(
                "CONDITIONAL_INTERFACE_ENDPOINT_INVALID",
                f"{name} must be an object",
            )
        endpoint_body = identifier(endpoint.get("body_id"), f"{name}.body_id")
        if expected is not None and endpoint_body != expected:
            raise _fail(
                "CONDITIONAL_INTERFACE_ENDPOINT_BODY_MISMATCH",
                f"{name} does not match the enclosing physical interface",
            )
        normalized = dict(endpoint)
        normalized["component_refs"] = _component_refs(
            endpoint.get("component_refs"),
            body_id=endpoint_body,
            label=f"{name}.component_refs",
        )
        normalized["local_collision_component_refs"] = _component_refs(
            endpoint.get(
                "local_collision_component_refs",
                normalized["component_refs"],
            ),
            body_id=endpoint_body,
            label=f"{name}.local_collision_component_refs",
            allow_empty=True,
        )
        local_refs = normalized["local_collision_component_refs"]
        if local_refs and not set(normalized["component_refs"]).issubset(
            local_refs
        ):
            raise _fail(
                "CONDITIONAL_INTERFACE_COMPONENT_BINDING_INVALID",
                f"{name}.local_collision_component_refs must include every "
                "functional endpoint component",
            )
        item[name] = normalized
    if item["endpoint_a"]["body_id"] == item["endpoint_b"]["body_id"]:
        raise _fail(
            "CONDITIONAL_INTERFACE_BODY_PAIR_INVALID",
            "a conditional interface must connect distinct bodies",
        )

    parameters = item.get("geometry_parameters")
    if not isinstance(parameters, Mapping):
        raise _fail(
            "CONDITIONAL_INTERFACE_GEOMETRY_PARAMETERS_INVALID",
            "geometry_parameters must be an object",
        )
    item["geometry_parameters"] = deepcopy(dict(parameters))
    required_policy = {
        "geometry_legality_source": "ORIGINAL_STEP_COMPONENTS",
        "evaluate_every_physics_step": True,
        "inactive_wrench_policy": EXACT_ZERO,
        "capture_reference_on_engage": True,
        "initially_inactive_is_warning": True,
    }
    for field_name, required in required_policy.items():
        if item.get(field_name) != required:
            raise _fail(
                "CONDITIONAL_INTERFACE_POLICY_INVALID",
                f"{field_name} must be {required!r}",
            )
    return item


@dataclass(frozen=True)
class InterfaceGeometryObservation:
    """One geometry-plugin observation at a saved or live physics pose."""

    time_s: float
    activation_coordinate_si: float
    activation_rate_si_s: float
    constraints_satisfied: bool
    can_engage_later: bool
    metrics: Mapping[str, float] = field(default_factory=dict)
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        finite(self.time_s, "time_s")
        finite(self.activation_coordinate_si, "activation_coordinate_si")
        finite(self.activation_rate_si_s, "activation_rate_si_s")
        if not isinstance(self.constraints_satisfied, bool):
            raise _fail(
                "CONDITIONAL_INTERFACE_OBSERVATION_INVALID",
                "constraints_satisfied must be boolean",
            )
        if not isinstance(self.can_engage_later, bool):
            raise _fail(
                "CONDITIONAL_INTERFACE_OBSERVATION_INVALID",
                "can_engage_later must be boolean",
            )
        normalized_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            normalized_metrics[identifier(key, "metric name")] = finite(
                value, f"metrics.{key}"
            )
        object.__setattr__(self, "metrics", normalized_metrics)
        normalized_blockers = tuple(
            identifier(item, "blocker") for item in self.blockers
        )
        if self.constraints_satisfied and normalized_blockers:
            raise _fail(
                "CONDITIONAL_INTERFACE_OBSERVATION_INVALID",
                "satisfied geometry cannot retain blockers",
            )
        object.__setattr__(self, "blockers", normalized_blockers)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metrics"] = dict(self.metrics)
        payload["blockers"] = list(self.blockers)
        return payload


@dataclass(frozen=True)
class ConditionalInterfaceDecision:
    time_s: float
    previous_state: str
    state: str
    transmission_active: bool
    effort_scale: float
    transitioned: bool
    capture_reference: bool
    reason: str
    observation: InterfaceGeometryObservation

    def transmit_scalar(self, requested_effort_si: float) -> float:
        requested = finite(requested_effort_si, "requested_effort_si")
        if not self.transmission_active:
            return 0.0
        return requested

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "previous_state": self.previous_state,
            "state": self.state,
            "transmission_active": self.transmission_active,
            "effort_scale": self.effort_scale,
            "transitioned": self.transitioned,
            "capture_reference": self.capture_reference,
            "reason": self.reason,
            "observation": self.observation.to_dict(),
        }


class ConditionalCompliantInterface:
    """Generic hysteretic runtime owner for a pose-dependent interface."""

    def __init__(self, transmission_gate: Mapping[str, Any]):
        self.gate = validate_conditional_transmission_gate(transmission_gate)
        self.state = INACTIVE
        self.last_decision: ConditionalInterfaceDecision | None = None

    def _at_enter(self, coordinate: float) -> bool:
        if self.gate["activation_direction"] == AT_OR_ABOVE:
            return coordinate >= self.gate["enter_threshold_si"]
        return coordinate <= self.gate["enter_threshold_si"]

    def _within_release(self, coordinate: float) -> bool:
        if self.gate["activation_direction"] == AT_OR_ABOVE:
            return coordinate > self.gate["release_threshold_si"]
        return coordinate < self.gate["release_threshold_si"]

    def _approaching(self, rate: float) -> bool:
        if self.gate["activation_direction"] == AT_OR_ABOVE:
            return rate > 0.0
        return rate < 0.0

    def step(
        self,
        observation: InterfaceGeometryObservation,
    ) -> ConditionalInterfaceDecision:
        previous = self.state
        coordinate = observation.activation_coordinate_si
        at_enter = (
            observation.constraints_satisfied and self._at_enter(coordinate)
        )
        can_hold = (
            observation.constraints_satisfied
            and self._within_release(coordinate)
        )

        if previous == ENGAGED:
            state = ENGAGED if can_hold else RELEASING
        elif previous == RELEASING:
            state = ENGAGED if at_enter else INACTIVE
        elif at_enter:
            state = ENGAGED
        elif (
            observation.constraints_satisfied
            and observation.can_engage_later
            and self._approaching(observation.activation_rate_si_s)
        ):
            state = APPROACHING
        else:
            state = INACTIVE

        active = state == ENGAGED
        capture = active and previous != ENGAGED
        if active:
            reason = (
                "GEOMETRY_ENTER_THRESHOLD_REACHED"
                if capture
                else "GEOMETRY_WITHIN_RELEASE_THRESHOLD"
            )
        elif state == RELEASING:
            reason = (
                "GEOMETRY_CONSTRAINT_LOST"
                if not observation.constraints_satisfied
                else "GEOMETRY_RELEASE_THRESHOLD_CROSSED"
            )
        elif state == APPROACHING:
            reason = "GEOMETRY_APPROACHING_ENTER_THRESHOLD"
        elif not observation.constraints_satisfied:
            reason = "GEOMETRY_CONSTRAINT_UNSATISFIED"
        elif not observation.can_engage_later:
            reason = "GEOMETRY_NOT_ENGAGEABLE"
        else:
            reason = "GEOMETRY_BELOW_ENTER_THRESHOLD"

        decision = ConditionalInterfaceDecision(
            time_s=observation.time_s,
            previous_state=previous,
            state=state,
            transmission_active=active,
            effort_scale=1.0 if active else 0.0,
            transitioned=state != previous,
            capture_reference=capture,
            reason=reason,
            observation=observation,
        )
        self.state = state
        self.last_decision = decision
        return decision


def assess_initial_functional_geometry(
    transmission_gate: Mapping[str, Any],
    observation: InterfaceGeometryObservation,
) -> dict[str, Any]:
    """Return a CAD-stage PASS/WARNING without inventing a force path."""

    controller = ConditionalCompliantInterface(transmission_gate)
    decision = controller.step(observation)
    if decision.transmission_active:
        status = "PASS"
        code = "FUNCTIONAL_INTERFACE_INITIALLY_ENGAGED"
        message = "initial STEP-bound geometry satisfies the transmission gate"
    elif observation.can_engage_later:
        status = "WARNING"
        code = "FUNCTIONAL_INTERFACE_INITIALLY_INACTIVE"
        message = (
            "initial STEP-bound geometry does not transmit; runtime geometry "
            "may activate the interface later"
        )
    else:
        status = "WARNING"
        code = "FUNCTIONAL_INTERFACE_NON_TRANSMITTING_AT_INITIAL_POSE"
        message = (
            "current geometry has no proven transmission path; runtime effort "
            "must remain exactly zero unless a later pose satisfies the gate"
        )
    return {
        "schema_version": CONDITIONAL_INTERFACE_SCHEMA,
        "status": status,
        "code": code,
        "message": message,
        "hard_error": False,
        "initial_state": decision.state,
        "transmission_active": decision.transmission_active,
        "inactive_wrench_policy": EXACT_ZERO,
        "geometry_plugin_id": controller.gate["geometry_plugin_id"],
        "endpoint_a": deepcopy(controller.gate["endpoint_a"]),
        "endpoint_b": deepcopy(controller.gate["endpoint_b"]),
        "observation": observation.to_dict(),
    }


def build_functional_interface_geometry_binding(
    interfaces: Sequence[Mapping[str, Any]],
    collision_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind conditional endpoints to the existing hashed STEP inventory."""

    from .collision_policy import validate_interface_declarations
    from .contact_intent import collision_component_inventory
    from .io import canonical_sha256

    inventory = collision_component_inventory(collision_geometry)
    known = {str(item["component_ref"]): item for item in inventory}
    records: list[dict[str, Any]] = []
    for interface in validate_interface_declarations(
        interfaces,
        valid_body_ids={str(item["body_id"]) for item in inventory},
    ):
        gate = interface.get("transmission_gate")
        if gate is None:
            continue
        validated_gate = validate_conditional_transmission_gate(
            gate,
            body_a_id=str(interface["body_a_id"]),
            body_b_id=str(interface["body_b_id"]),
        )
        endpoint_records: dict[str, dict[str, list[dict[str, str]]]] = {}
        for endpoint_name in ("endpoint_a", "endpoint_b"):
            endpoint = validated_gate[endpoint_name]
            bound_sets: dict[str, list[dict[str, str]]] = {}
            for ref_field in (
                "component_refs",
                "local_collision_component_refs",
            ):
                bound: list[dict[str, str]] = []
                for ref in endpoint[ref_field]:
                    component = known.get(str(ref))
                    if (
                        component is None
                        or component["body_id"] != endpoint["body_id"]
                    ):
                        raise _fail(
                            "CONDITIONAL_INTERFACE_STEP_BINDING_INVALID",
                            f"{interface['interface_id']} references an "
                            f"unknown STEP component: {ref}",
                        )
                    bound.append(
                        {
                            "component_ref": str(ref),
                            "component_sha256": str(
                                component["component_sha256"]
                            ),
                        }
                    )
                bound_sets[ref_field] = bound
            endpoint_records[endpoint_name] = bound_sets
        records.append(
            {
                "interface_id": str(interface["interface_id"]),
                "geometry_plugin_id": validated_gate["geometry_plugin_id"],
                "gate_sha256": canonical_sha256(validated_gate),
                "endpoint_a": endpoint_records["endpoint_a"][
                    "component_refs"
                ],
                "endpoint_b": endpoint_records["endpoint_b"][
                    "component_refs"
                ],
                "endpoint_a_local_collision_components": endpoint_records[
                    "endpoint_a"
                ]["local_collision_component_refs"],
                "endpoint_b_local_collision_components": endpoint_records[
                    "endpoint_b"
                ]["local_collision_component_refs"],
            }
        )
    return {
        "schema_version": "functional_interface_geometry_binding_v1",
        "package_binding_sha256": collision_geometry.get(
            "package_binding_sha256"
        ),
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "component_inventory_sha256": canonical_sha256(list(inventory)),
        "conditional_interface_count": len(records),
        "conditional_interfaces_sha256": canonical_sha256(records),
        "conditional_interfaces": records,
        "source": "validator_existing_step_component_inventory_v1",
    }


def validate_functional_interface_geometry_binding(
    binding: Mapping[str, Any],
    *,
    interfaces: Sequence[Mapping[str, Any]],
    collision_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Rebuild and compare the binding before runtime force activation."""

    if not isinstance(binding, Mapping):
        raise _fail(
            "CONDITIONAL_INTERFACE_STEP_BINDING_REQUIRED",
            "conditional interfaces require a hash-bound STEP component binding",
        )
    expected = build_functional_interface_geometry_binding(
        interfaces, collision_geometry
    )
    if deepcopy(dict(binding)) != expected:
        raise _fail(
            "CONDITIONAL_INTERFACE_STEP_BINDING_CHANGED",
            "functional-interface geometry or its STEP inventory changed",
        )
    return expected


def audit_conditional_interface_runtime_evidence(
    interfaces: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    runtime_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Fail closed on missing evaluations or any inactive effort leakage."""

    from .collision_policy import (
        COMPLIANT_INTERFACE_PHYSICAL,
        validate_interface_declarations,
    )

    gated = {
        str(item["interface_id"]): item
        for item in validate_interface_declarations(interfaces)
        if item.get("transmission_gate") is not None
    }
    if not gated:
        return {
            "schema_version": "conditional_interface_runtime_audit_v1",
            "status": "NOT_REQUIRED",
            "passed": True,
            "failure_reasons": [],
            "conditional_interface_count": 0,
        }
    failures: list[str] = []
    summary = runtime_summary if isinstance(runtime_summary, Mapping) else {}
    if summary.get("schema_version") != "conditional_interface_runtime_v1":
        failures.append("conditional_interface_runtime_summary_missing")
    if summary.get("evaluated_every_physics_step") is not True:
        failures.append("conditional_interface_not_evaluated_every_step")
    raw_step_count = summary.get("physics_step_count")
    if (
        isinstance(raw_step_count, bool)
        or not isinstance(raw_step_count, int)
        or raw_step_count <= 0
    ):
        failures.append("conditional_interface_physics_step_count_invalid")
        physics_step_count = 0
    else:
        physics_step_count = raw_step_count

    raw_summaries = summary.get("interfaces")
    by_id = {
        str(item.get("interface_id")): item
        for item in raw_summaries or ()
        if isinstance(item, Mapping) and item.get("interface_id")
    }
    if set(by_id) != set(gated):
        failures.append("conditional_interface_runtime_inventory_mismatch")
    for interface_id in gated:
        item = by_id.get(interface_id, {})
        evaluation_count = item.get("evaluation_count")
        if (
            isinstance(evaluation_count, bool)
            or not isinstance(evaluation_count, int)
            or evaluation_count < physics_step_count
        ):
            failures.append("conditional_interface_evaluation_count_incomplete")
        maximum = item.get("max_inactive_abs_transmitted_effort_si")
        try:
            inactive_maximum = finite(
                maximum,
                "max_inactive_abs_transmitted_effort_si",
            )
        except MechanismValidationError:
            inactive_maximum = math.inf
        if inactive_maximum != 0.0:
            failures.append("conditional_interface_inactive_effort_nonzero")
        state_path = item.get("state_path")
        if (
            not isinstance(state_path, list)
            or not state_path
            or set(str(value) for value in state_path) - INTERFACE_STATES
        ):
            failures.append("conditional_interface_state_path_invalid")

    saved_evidence_count = 0
    for sample in samples:
        states = sample.get("conditional_interface_states")
        if not isinstance(states, list):
            failures.append("conditional_interface_saved_state_missing")
            continue
        saved = {
            str(item.get("interface_id")): item
            for item in states
            if isinstance(item, Mapping) and item.get("interface_id")
        }
        if set(saved) != set(gated):
            failures.append("conditional_interface_saved_inventory_mismatch")
            continue
        saved_evidence_count += 1
        for interface_id, item in saved.items():
            state = str(item.get("state") or "")
            active = item.get("transmission_active")
            if state not in INTERFACE_STATES or active is not (state == ENGAGED):
                failures.append("conditional_interface_saved_state_invalid")
                continue
            try:
                effort = finite(
                    item.get("transmitted_effort_si"),
                    "transmitted_effort_si",
                )
            except MechanismValidationError:
                failures.append("conditional_interface_saved_effort_invalid")
                continue
            if not active and effort != 0.0:
                failures.append("conditional_interface_inactive_effort_nonzero")
            support_interface = (
                gated[interface_id].get("classification")
                == COMPLIANT_INTERFACE_PHYSICAL
            )
            torque = item.get("transmitted_support_torque_nm")
            requested_torque = item.get("requested_support_torque_nm")
            if support_interface or torque is not None:
                try:
                    torque_nm = finite(
                        torque,
                        "transmitted_support_torque_nm",
                    )
                    finite(
                        requested_torque,
                        "requested_support_torque_nm",
                    )
                except MechanismValidationError:
                    failures.append(
                        "conditional_interface_support_torque_invalid"
                    )
                else:
                    if not active and torque_nm != 0.0:
                        failures.append(
                            "conditional_interface_inactive_torque_nonzero"
                        )
            if item.get("constraints_satisfied") is False and active:
                failures.append("conditional_interface_active_without_geometry")

    unique_failures = list(dict.fromkeys(failures))
    return {
        "schema_version": "conditional_interface_runtime_audit_v1",
        "status": "PASS" if not unique_failures else "FAIL",
        "passed": not unique_failures,
        "failure_reasons": unique_failures,
        "conditional_interface_count": len(gated),
        "physics_step_count": physics_step_count,
        "saved_state_sample_count": saved_evidence_count,
        "inactive_effort_requires_exact_zero": True,
    }


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def _scale(
    factor: float,
    value: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(factor * item for item in value)


def _cross(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Sequence[float], label: str) -> tuple[float, float, float]:
    result = vector3(value, label)
    length = _norm(result)
    return tuple(item / length for item in result)


def _circle_basis(
    axis: Sequence[float],
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
]:
    """Return a deterministic orthonormal basis in a circle's plane."""

    axis_unit = _unit(axis, "circle_axis")
    least_aligned_index = min(
        range(3),
        key=lambda index: abs(axis_unit[index]),
    )
    reference = tuple(
        1.0 if index == least_aligned_index else 0.0
        for index in range(3)
    )
    first = _unit(_cross(axis_unit, reference), "circle_basis")
    return first, _cross(axis_unit, first)


def _nearest_point_on_circle(
    *,
    center: Sequence[float],
    axis: Sequence[float],
    radius: float,
    point: Sequence[float],
    fallback_radial: Sequence[float],
) -> tuple[float, float, float]:
    delta = _sub(point, center)
    radial = _sub(delta, _scale(_dot(delta, axis), axis))
    if _norm(radial) <= 1.0e-12:
        radial = tuple(fallback_radial)
    return _add(center, _scale(radius, _unit(radial, "circle_radial")))


def _closest_pitch_circle_points(
    *,
    driver_center: Sequence[float],
    driver_axis: Sequence[float],
    driver_radius: float,
    driven_center: Sequence[float],
    driven_axis: Sequence[float],
    driven_radius: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    float,
]:
    """Find the closest points on two finite 3-D pitch circles.

    For one point on the driver circle, its exact nearest point on the driven
    circle is closed-form.  A deterministic coarse angular scan followed by a
    bounded golden-section refinement therefore solves only one periodic
    coordinate.  This is cheap enough to repeat at every physical step.
    """

    driver_axis_unit = _unit(driver_axis, "driver_axis")
    driven_axis_unit = _unit(driven_axis, "driven_axis")
    driver_u, driver_v = _circle_basis(driver_axis_unit)
    driven_u, _ = _circle_basis(driven_axis_unit)

    def pair_at(
        angle: float,
    ) -> tuple[
        float,
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        driver_radial = _add(
            _scale(math.cos(angle), driver_u),
            _scale(math.sin(angle), driver_v),
        )
        driver_point = _add(
            driver_center,
            _scale(driver_radius, driver_radial),
        )
        driven_point = _nearest_point_on_circle(
            center=driven_center,
            axis=driven_axis_unit,
            radius=driven_radius,
            point=driver_point,
            fallback_radial=driven_u,
        )
        delta = _sub(driver_point, driven_point)
        return _dot(delta, delta), driver_point, driven_point

    coarse_count = 32
    coarse_step = 2.0 * math.pi / coarse_count
    coarse = [
        (pair_at(index * coarse_step)[0], index)
        for index in range(coarse_count)
    ]
    _, best_index = min(coarse)
    lower = (best_index - 1) * coarse_step
    upper = (best_index + 1) * coarse_step
    inverse_phi = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - inverse_phi * (upper - lower)
    right = lower + inverse_phi * (upper - lower)
    left_pair = pair_at(left)
    right_pair = pair_at(right)
    for _ in range(24):
        if left_pair[0] <= right_pair[0]:
            upper = right
            right = left
            right_pair = left_pair
            left = upper - inverse_phi * (upper - lower)
            left_pair = pair_at(left)
        else:
            lower = left
            left = right
            left_pair = right_pair
            right = lower + inverse_phi * (upper - lower)
            right_pair = pair_at(right)
    best = min(
        left_pair,
        right_pair,
        pair_at(0.5 * (lower + upper)),
        key=lambda item: item[0],
    )
    return (
        best[1],
        best[2],
        math.sqrt(max(0.0, best[0])),
    )


@dataclass(frozen=True)
class AxialGearPose:
    center_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    face_width_m: float
    linear_velocity_abs_m_s: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_abs_m",
            vector3(self.center_abs_m, "center_abs_m", allow_zero=True),
        )
        object.__setattr__(
            self,
            "axis_abs",
            _unit(self.axis_abs, "axis_abs"),
        )
        object.__setattr__(
            self,
            "linear_velocity_abs_m_s",
            vector3(
                self.linear_velocity_abs_m_s,
                "linear_velocity_abs_m_s",
                allow_zero=True,
            ),
        )
        if finite(self.face_width_m, "face_width_m") <= 0.0:
            raise _fail(
                "GEAR_FACE_WIDTH_INVALID",
                "face_width_m must be positive",
            )


@dataclass(frozen=True)
class AxialSlidingSpurGearGeometry:
    """Geometry plugin for two external spur gears with axial engagement."""

    driver_pitch_radius_m: float
    driven_pitch_radius_m: float
    radial_tolerance_m: float
    axis_parallel_tolerance_rad: float
    driver_module_m: float | None = None
    driven_module_m: float | None = None
    module_relative_tolerance: float = 0.005
    plugin_id: str = "axial_sliding_spur_gear_mesh_v1"

    def __post_init__(self) -> None:
        for name in (
            "driver_pitch_radius_m",
            "driven_pitch_radius_m",
            "radial_tolerance_m",
            "axis_parallel_tolerance_rad",
        ):
            value = finite(getattr(self, name), name)
            if value <= 0.0:
                raise _fail(
                    "GEAR_GEOMETRY_PARAMETER_INVALID",
                    f"{name} must be positive",
                )
        if (self.driver_module_m is None) != (
            self.driven_module_m is None
        ):
            raise _fail(
                "GEAR_GEOMETRY_PARAMETER_INVALID",
                "driver and driven modules must be supplied together",
            )
        if self.driver_module_m is not None:
            for name in ("driver_module_m", "driven_module_m"):
                if finite(getattr(self, name), name) <= 0.0:
                    raise _fail(
                        "GEAR_GEOMETRY_PARAMETER_INVALID",
                        f"{name} must be positive",
                    )
        if finite(
            self.module_relative_tolerance,
            "module_relative_tolerance",
        ) <= 0.0:
            raise _fail(
                "GEAR_GEOMETRY_PARAMETER_INVALID",
                "module_relative_tolerance must be positive",
            )
        identifier(self.plugin_id, "plugin_id")

    def evaluate(
        self,
        *,
        time_s: float,
        driver: AxialGearPose,
        driven: AxialGearPose,
    ) -> InterfaceGeometryObservation:
        axis_dot = _dot(driver.axis_abs, driven.axis_abs)
        aligned_driver_axis = (
            driver.axis_abs
            if axis_dot >= 0.0
            else tuple(-item for item in driver.axis_abs)
        )
        common_axis_raw = tuple(
            left + right
            for left, right in zip(aligned_driver_axis, driven.axis_abs)
        )
        common_axis = _unit(common_axis_raw, "common_axis")
        axis_parallel_error = math.acos(
            max(-1.0, min(1.0, abs(axis_dot)))
        )

        center_delta = _sub(driver.center_abs_m, driven.center_abs_m)
        axial_offset_signed = _dot(center_delta, common_axis)
        axial_offset = abs(axial_offset_signed)
        radial_delta = tuple(
            component - axial_offset_signed * axis_component
            for component, axis_component in zip(center_delta, common_axis)
        )
        measured_center_distance = _norm(radial_delta)
        required_center_distance = (
            self.driver_pitch_radius_m + self.driven_pitch_radius_m
        )
        radial_mismatch = abs(
            measured_center_distance - required_center_distance
        )
        face_overlap = (
            0.5 * (driver.face_width_m + driven.face_width_m)
            - axial_offset
        )

        relative_velocity = _sub(
            driver.linear_velocity_abs_m_s,
            driven.linear_velocity_abs_m_s,
        )
        axial_velocity_signed = _dot(relative_velocity, common_axis)
        if abs(axial_offset_signed) <= 1e-15:
            overlap_rate = 0.0
        else:
            overlap_rate = (
                -math.copysign(1.0, axial_offset_signed)
                * axial_velocity_signed
            )

        blockers: list[str] = []
        if axis_parallel_error > self.axis_parallel_tolerance_rad:
            blockers.append("AXES_NOT_PARALLEL")
        if radial_mismatch > self.radial_tolerance_m:
            blockers.append("PITCH_CIRCLES_NOT_TANGENT")
        module_relative_error = 0.0
        if (
            self.driver_module_m is not None
            and self.driven_module_m is not None
        ):
            mean_module = 0.5 * (
                self.driver_module_m + self.driven_module_m
            )
            module_relative_error = (
                abs(self.driver_module_m - self.driven_module_m)
                / mean_module
            )
            if module_relative_error > self.module_relative_tolerance:
                blockers.append("GEAR_MODULE_MISMATCH")
        constraints_satisfied = not blockers
        return InterfaceGeometryObservation(
            time_s=time_s,
            activation_coordinate_si=face_overlap,
            activation_rate_si_s=overlap_rate,
            constraints_satisfied=constraints_satisfied,
            can_engage_later=constraints_satisfied,
            metrics={
                "axis_parallel_error_rad": axis_parallel_error,
                "axial_offset_m": axial_offset,
                "face_overlap_m": face_overlap,
                "measured_center_distance_m": measured_center_distance,
                "required_center_distance_m": required_center_distance,
                "radial_pitch_mismatch_m": radial_mismatch,
                "module_relative_error": module_relative_error,
            },
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class RightAngleBevelGearGeometry:
    """Live gate for a finite right-angle bevel/crown gear engagement.

    Unlike a parallel spur pair, crossed-axis gears cannot be validated from
    one scalar shaft-center distance.  This plugin resolves the nearest points
    on both finite pitch circles.  Their separation controls engagement, while
    the two positive-rotation tangents independently determine the only
    physically compatible signed transmission ratio.
    """

    driver_pitch_radius_m: float
    driven_pitch_radius_m: float
    pitch_surface_tolerance_m: float
    required_shaft_angle_rad: float
    shaft_angle_tolerance_rad: float
    driver_face_width_m: float
    driven_face_width_m: float
    minimum_tangent_alignment: float = 0.5
    plugin_id: str = "right_angle_bevel_gear_mesh_v1"

    def __post_init__(self) -> None:
        for name in (
            "driver_pitch_radius_m",
            "driven_pitch_radius_m",
            "pitch_surface_tolerance_m",
            "required_shaft_angle_rad",
            "shaft_angle_tolerance_rad",
            "driver_face_width_m",
            "driven_face_width_m",
        ):
            if finite(getattr(self, name), name) <= 0.0:
                raise _fail(
                    "BEVEL_GEAR_GEOMETRY_PARAMETER_INVALID",
                    f"{name} must be positive",
                )
        alignment = finite(
            self.minimum_tangent_alignment,
            "minimum_tangent_alignment",
        )
        if not 0.0 < alignment <= 1.0:
            raise _fail(
                "BEVEL_GEAR_GEOMETRY_PARAMETER_INVALID",
                "minimum_tangent_alignment must be in (0, 1]",
            )
        identifier(self.plugin_id, "plugin_id")

    def evaluate(
        self,
        *,
        time_s: float,
        driver: AxialGearPose,
        driven: AxialGearPose,
    ) -> InterfaceGeometryObservation:
        axis_dot = abs(_dot(driver.axis_abs, driven.axis_abs))
        measured_shaft_angle = math.acos(
            max(-1.0, min(1.0, axis_dot))
        )
        shaft_angle_error = abs(
            measured_shaft_angle - self.required_shaft_angle_rad
        )
        driver_contact, driven_contact, pitch_surface_gap = (
            _closest_pitch_circle_points(
                driver_center=driver.center_abs_m,
                driver_axis=driver.axis_abs,
                driver_radius=self.driver_pitch_radius_m,
                driven_center=driven.center_abs_m,
                driven_axis=driven.axis_abs,
                driven_radius=self.driven_pitch_radius_m,
            )
        )
        driver_tangent = _cross(
            driver.axis_abs,
            _sub(driver_contact, driver.center_abs_m),
        )
        driven_tangent = _cross(
            driven.axis_abs,
            _sub(driven_contact, driven.center_abs_m),
        )
        tangent_alignment = _dot(
            _unit(driver_tangent, "driver_contact_tangent"),
            _unit(driven_tangent, "driven_contact_tangent"),
        )
        mesh_direction_sign = 1.0 if tangent_alignment > 0.0 else -1.0
        relative_velocity = _sub(
            driver.linear_velocity_abs_m_s,
            driven.linear_velocity_abs_m_s,
        )
        contact_delta = _sub(driver_contact, driven_contact)
        if pitch_surface_gap <= 1.0e-15:
            pitch_surface_gap_rate = 0.0
        else:
            pitch_surface_gap_rate = (
                _dot(contact_delta, relative_velocity)
                / pitch_surface_gap
            )
        blockers: list[str] = []
        if shaft_angle_error > self.shaft_angle_tolerance_rad:
            blockers.append("SHAFTS_NOT_AT_DECLARED_ANGLE")
        if abs(tangent_alignment) < self.minimum_tangent_alignment:
            blockers.append("PITCH_CONTACT_TANGENTS_INCOMPATIBLE")
        return InterfaceGeometryObservation(
            time_s=time_s,
            activation_coordinate_si=pitch_surface_gap,
            activation_rate_si_s=pitch_surface_gap_rate,
            constraints_satisfied=not blockers,
            can_engage_later=True,
            metrics={
                "pitch_surface_gap_m": pitch_surface_gap,
                "pitch_surface_tolerance_m": (
                    self.pitch_surface_tolerance_m
                ),
                "shaft_angle_rad": measured_shaft_angle,
                "shaft_angle_error_rad": shaft_angle_error,
                "tangent_alignment": tangent_alignment,
                "mesh_direction_sign": mesh_direction_sign,
            },
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class CoplanarPulleyTransferGeometry:
    """Geometry gate for an analytical belt/shaft transfer between pulleys."""

    required_center_distance_m: float
    center_distance_tolerance_m: float
    axis_parallel_tolerance_rad: float
    plugin_id: str = "coplanar_pulley_transfer_v1"

    def __post_init__(self) -> None:
        for name in (
            "required_center_distance_m",
            "center_distance_tolerance_m",
            "axis_parallel_tolerance_rad",
        ):
            value = finite(getattr(self, name), name)
            if value <= 0.0:
                raise _fail(
                    "PULLEY_TRANSFER_GEOMETRY_PARAMETER_INVALID",
                    f"{name} must be positive",
                )
        identifier(self.plugin_id, "plugin_id")

    def evaluate(
        self,
        *,
        time_s: float,
        driver: AxialGearPose,
        driven: AxialGearPose,
    ) -> InterfaceGeometryObservation:
        axis_dot = _dot(driver.axis_abs, driven.axis_abs)
        aligned_driver_axis = (
            driver.axis_abs
            if axis_dot >= 0.0
            else tuple(-item for item in driver.axis_abs)
        )
        common_axis = _unit(
            tuple(
                left + right
                for left, right in zip(
                    aligned_driver_axis,
                    driven.axis_abs,
                )
            ),
            "common_axis",
        )
        axis_parallel_error = math.acos(
            max(-1.0, min(1.0, abs(axis_dot)))
        )
        center_delta = _sub(
            driver.center_abs_m,
            driven.center_abs_m,
        )
        axial_offset_signed = _dot(center_delta, common_axis)
        radial_delta = tuple(
            component - axial_offset_signed * axis_component
            for component, axis_component in zip(
                center_delta,
                common_axis,
            )
        )
        center_distance = _norm(radial_delta)
        center_distance_error = abs(
            center_distance - self.required_center_distance_m
        )
        face_overlap = (
            0.5 * (driver.face_width_m + driven.face_width_m)
            - abs(axial_offset_signed)
        )
        relative_velocity = _sub(
            driver.linear_velocity_abs_m_s,
            driven.linear_velocity_abs_m_s,
        )
        axial_velocity_signed = _dot(relative_velocity, common_axis)
        overlap_rate = (
            0.0
            if abs(axial_offset_signed) <= 1.0e-15
            else -math.copysign(1.0, axial_offset_signed)
            * axial_velocity_signed
        )
        blockers: list[str] = []
        if axis_parallel_error > self.axis_parallel_tolerance_rad:
            blockers.append("PULLEY_AXES_NOT_PARALLEL")
        if center_distance_error > self.center_distance_tolerance_m:
            blockers.append("PULLEY_CENTER_DISTANCE_CHANGED")
        return InterfaceGeometryObservation(
            time_s=time_s,
            activation_coordinate_si=face_overlap,
            activation_rate_si_s=overlap_rate,
            constraints_satisfied=not blockers,
            can_engage_later=not blockers,
            metrics={
                "axis_parallel_error_rad": axis_parallel_error,
                "face_overlap_m": face_overlap,
                "measured_center_distance_m": center_distance,
                "required_center_distance_m": (
                    self.required_center_distance_m
                ),
                "center_distance_error_m": center_distance_error,
            },
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class RadialReachPose:
    center_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    radial_reach_m: float
    face_width_m: float
    linear_velocity_abs_m_s: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "center_abs_m",
            vector3(self.center_abs_m, "center_abs_m", allow_zero=True),
        )
        object.__setattr__(
            self,
            "axis_abs",
            _unit(self.axis_abs, "axis_abs"),
        )
        object.__setattr__(
            self,
            "linear_velocity_abs_m_s",
            vector3(
                self.linear_velocity_abs_m_s,
                "linear_velocity_abs_m_s",
                allow_zero=True,
            ),
        )
        for name in ("radial_reach_m", "face_width_m"):
            if finite(getattr(self, name), name) <= 0.0:
                raise _fail(
                    "RADIAL_REACH_POSE_INVALID",
                    f"{name} must be positive",
                )


@dataclass(frozen=True)
class RadialReachGeometry:
    """Swept-envelope gate for analytical cam/follower transmission."""

    radial_gap_tolerance_m: float
    axis_parallel_tolerance_rad: float
    plugin_id: str = "radial_cam_follower_reach_v1"

    def __post_init__(self) -> None:
        for name in (
            "radial_gap_tolerance_m",
            "axis_parallel_tolerance_rad",
        ):
            if finite(getattr(self, name), name) <= 0.0:
                raise _fail(
                    "RADIAL_REACH_GEOMETRY_PARAMETER_INVALID",
                    f"{name} must be positive",
                )
        identifier(self.plugin_id, "plugin_id")

    def evaluate(
        self,
        *,
        time_s: float,
        driver: RadialReachPose,
        follower: RadialReachPose,
    ) -> InterfaceGeometryObservation:
        axis_dot = _dot(driver.axis_abs, follower.axis_abs)
        aligned_driver_axis = (
            driver.axis_abs
            if axis_dot >= 0.0
            else tuple(-item for item in driver.axis_abs)
        )
        common_axis = _unit(
            tuple(
                left + right
                for left, right in zip(
                    aligned_driver_axis,
                    follower.axis_abs,
                )
            ),
            "common_axis",
        )
        axis_parallel_error = math.acos(
            max(-1.0, min(1.0, abs(axis_dot)))
        )
        center_delta = _sub(
            driver.center_abs_m,
            follower.center_abs_m,
        )
        axial_offset_signed = _dot(center_delta, common_axis)
        radial_delta = tuple(
            component - axial_offset_signed * axis_component
            for component, axis_component in zip(
                center_delta,
                common_axis,
            )
        )
        center_distance = _norm(radial_delta)
        surface_gap = (
            center_distance
            - driver.radial_reach_m
            - follower.radial_reach_m
        )
        face_overlap = (
            0.5 * (driver.face_width_m + follower.face_width_m)
            - abs(axial_offset_signed)
        )
        relative_velocity = _sub(
            driver.linear_velocity_abs_m_s,
            follower.linear_velocity_abs_m_s,
        )
        radial_direction = (
            tuple(item / center_distance for item in radial_delta)
            if center_distance > 1.0e-15
            else (0.0, 0.0, 0.0)
        )
        gap_rate = _dot(relative_velocity, radial_direction)
        blockers: list[str] = []
        if axis_parallel_error > self.axis_parallel_tolerance_rad:
            blockers.append("CAM_FOLLOWER_AXES_NOT_PARALLEL")
        if face_overlap <= 0.0:
            blockers.append("CAM_FOLLOWER_FACE_BANDS_SEPARATED")
        return InterfaceGeometryObservation(
            time_s=time_s,
            activation_coordinate_si=surface_gap,
            activation_rate_si_s=gap_rate,
            constraints_satisfied=not blockers,
            can_engage_later=axis_parallel_error
            <= self.axis_parallel_tolerance_rad,
            metrics={
                "axis_parallel_error_rad": axis_parallel_error,
                "face_overlap_m": face_overlap,
                "center_distance_m": center_distance,
                "driver_radial_reach_m": driver.radial_reach_m,
                "follower_radial_reach_m": follower.radial_reach_m,
                "surface_gap_m": surface_gap,
            },
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class AxialClutchPose:
    face_center_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    radial_extent_m: float
    linear_velocity_abs_m_s: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "face_center_abs_m",
            vector3(
                self.face_center_abs_m,
                "face_center_abs_m",
                allow_zero=True,
            ),
        )
        object.__setattr__(
            self,
            "axis_abs",
            _unit(self.axis_abs, "axis_abs"),
        )
        object.__setattr__(
            self,
            "linear_velocity_abs_m_s",
            vector3(
                self.linear_velocity_abs_m_s,
                "linear_velocity_abs_m_s",
                allow_zero=True,
            ),
        )
        if finite(self.radial_extent_m, "radial_extent_m") <= 0.0:
            raise _fail(
                "AXIAL_CLUTCH_POSE_INVALID",
                "radial_extent_m must be positive",
            )


@dataclass(frozen=True)
class AxialClutchGeometry:
    """Face-gap gate for a coaxial analytical clutch."""

    radial_alignment_tolerance_m: float
    axis_parallel_tolerance_rad: float
    plugin_id: str = "coaxial_clutch_face_gap_v1"

    def __post_init__(self) -> None:
        for name in (
            "radial_alignment_tolerance_m",
            "axis_parallel_tolerance_rad",
        ):
            if finite(getattr(self, name), name) <= 0.0:
                raise _fail(
                    "AXIAL_CLUTCH_GEOMETRY_PARAMETER_INVALID",
                    f"{name} must be positive",
                )
        identifier(self.plugin_id, "plugin_id")

    def evaluate(
        self,
        *,
        time_s: float,
        input_face: AxialClutchPose,
        driven_face: AxialClutchPose,
    ) -> InterfaceGeometryObservation:
        axis_dot = _dot(input_face.axis_abs, driven_face.axis_abs)
        aligned_input_axis = (
            input_face.axis_abs
            if axis_dot >= 0.0
            else tuple(-item for item in input_face.axis_abs)
        )
        common_axis = _unit(
            tuple(
                left + right
                for left, right in zip(
                    aligned_input_axis,
                    driven_face.axis_abs,
                )
            ),
            "common_axis",
        )
        axis_parallel_error = math.acos(
            max(-1.0, min(1.0, abs(axis_dot)))
        )
        face_delta = _sub(
            input_face.face_center_abs_m,
            driven_face.face_center_abs_m,
        )
        signed_gap = _dot(face_delta, common_axis)
        radial_delta = tuple(
            component - signed_gap * axis_component
            for component, axis_component in zip(
                face_delta,
                common_axis,
            )
        )
        radial_offset = _norm(radial_delta)
        relative_velocity = _sub(
            input_face.linear_velocity_abs_m_s,
            driven_face.linear_velocity_abs_m_s,
        )
        signed_rate = _dot(relative_velocity, common_axis)
        surface_gap = abs(signed_gap)
        gap_rate = (
            0.0
            if surface_gap <= 1.0e-15
            else math.copysign(1.0, signed_gap) * signed_rate
        )
        radial_overlap = (
            input_face.radial_extent_m
            + driven_face.radial_extent_m
            - radial_offset
        )
        blockers: list[str] = []
        if axis_parallel_error > self.axis_parallel_tolerance_rad:
            blockers.append("CLUTCH_AXES_NOT_PARALLEL")
        if radial_offset > self.radial_alignment_tolerance_m:
            blockers.append("CLUTCH_FACES_NOT_COAXIAL")
        if radial_overlap <= 0.0:
            blockers.append("CLUTCH_FACE_RADII_DO_NOT_OVERLAP")
        return InterfaceGeometryObservation(
            time_s=time_s,
            activation_coordinate_si=surface_gap,
            activation_rate_si_s=gap_rate,
            constraints_satisfied=not blockers,
            can_engage_later=axis_parallel_error
            <= self.axis_parallel_tolerance_rad,
            metrics={
                "axis_parallel_error_rad": axis_parallel_error,
                "surface_gap_m": surface_gap,
                "radial_offset_m": radial_offset,
                "radial_overlap_m": radial_overlap,
            },
            blockers=tuple(blockers),
        )


@dataclass(frozen=True)
class RackPinionPose:
    """Live STEP-bound reference geometry for one rack-and-pinion interface."""

    pinion_center_abs_m: tuple[float, float, float]
    pinion_axis_abs: tuple[float, float, float]
    pinion_face_width_m: float
    rack_pitch_line_point_abs_m: tuple[float, float, float]
    rack_axis_abs: tuple[float, float, float]
    rack_face_axis_abs: tuple[float, float, float]
    rack_face_width_m: float
    pinion_linear_velocity_abs_m_s: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    rack_linear_velocity_abs_m_s: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        for name in (
            "pinion_center_abs_m",
            "rack_pitch_line_point_abs_m",
            "pinion_linear_velocity_abs_m_s",
            "rack_linear_velocity_abs_m_s",
        ):
            object.__setattr__(
                self,
                name,
                vector3(getattr(self, name), name, allow_zero=True),
            )
        for name in (
            "pinion_axis_abs",
            "rack_axis_abs",
            "rack_face_axis_abs",
        ):
            object.__setattr__(
                self,
                name,
                _unit(getattr(self, name), name),
            )
        for name in ("pinion_face_width_m", "rack_face_width_m"):
            if finite(getattr(self, name), name) <= 0.0:
                raise _fail(
                    "RACK_PINION_FACE_WIDTH_INVALID",
                    f"{name} must be positive",
                )


@dataclass(frozen=True)
class RackPinionGeometry:
    """Geometry gate for a finite analytical rack-and-pinion force law."""

    required_pitch_line_offset_m: float
    pitch_line_offset_tolerance_m: float
    axis_orthogonality_tolerance_rad: float
    face_axis_parallel_tolerance_rad: float
    plugin_id: str = "rack_pinion_mesh_geometry_v1"

    def __post_init__(self) -> None:
        for name in (
            "required_pitch_line_offset_m",
            "pitch_line_offset_tolerance_m",
            "axis_orthogonality_tolerance_rad",
            "face_axis_parallel_tolerance_rad",
        ):
            value = finite(getattr(self, name), name)
            if value <= 0.0:
                raise _fail(
                    "RACK_PINION_GEOMETRY_PARAMETER_INVALID",
                    f"{name} must be positive",
                )
        identifier(self.plugin_id, "plugin_id")

    def evaluate(
        self,
        *,
        time_s: float,
        pose: RackPinionPose,
    ) -> InterfaceGeometryObservation:
        pinion_axis = pose.pinion_axis_abs
        rack_axis = pose.rack_axis_abs
        face_axis = (
            pose.rack_face_axis_abs
            if _dot(pinion_axis, pose.rack_face_axis_abs) >= 0.0
            else tuple(-item for item in pose.rack_face_axis_abs)
        )
        common_face_axis = _unit(
            tuple(
                left + right
                for left, right in zip(pinion_axis, face_axis)
            ),
            "common_face_axis",
        )

        center_delta = _sub(
            pose.pinion_center_abs_m,
            pose.rack_pitch_line_point_abs_m,
        )
        along_rack = _dot(center_delta, rack_axis)
        normal_delta = tuple(
            component - along_rack * axis_component
            for component, axis_component in zip(center_delta, rack_axis)
        )
        measured_pitch_line_offset = _norm(normal_delta)
        pitch_line_offset_error = abs(
            measured_pitch_line_offset
            - self.required_pitch_line_offset_m
        )
        axis_orthogonality_error = math.asin(
            min(1.0, abs(_dot(pinion_axis, rack_axis)))
        )
        face_axis_parallel_error = math.acos(
            max(
                -1.0,
                min(1.0, abs(_dot(pinion_axis, face_axis))),
            )
        )
        face_offset_signed = _dot(center_delta, common_face_axis)
        face_overlap = (
            0.5
            * (pose.pinion_face_width_m + pose.rack_face_width_m)
            - abs(face_offset_signed)
        )

        relative_velocity = _sub(
            pose.pinion_linear_velocity_abs_m_s,
            pose.rack_linear_velocity_abs_m_s,
        )
        face_velocity_signed = _dot(relative_velocity, common_face_axis)
        if abs(face_offset_signed) <= 1e-15:
            overlap_rate = 0.0
        else:
            overlap_rate = (
                -math.copysign(1.0, face_offset_signed)
                * face_velocity_signed
            )

        blockers: list[str] = []
        if (
            axis_orthogonality_error
            > self.axis_orthogonality_tolerance_rad
        ):
            blockers.append("PINION_AXIS_NOT_ORTHOGONAL_TO_RACK")
        if (
            face_axis_parallel_error
            > self.face_axis_parallel_tolerance_rad
        ):
            blockers.append("PINION_AND_RACK_FACE_AXES_NOT_PARALLEL")
        if (
            pitch_line_offset_error
            > self.pitch_line_offset_tolerance_m
        ):
            blockers.append("PINION_NOT_AT_RACK_PITCH_LINE")
            blockers.append("PITCH_CIRCLES_NOT_TANGENT")
        constraints_satisfied = not blockers
        return InterfaceGeometryObservation(
            time_s=time_s,
            activation_coordinate_si=face_overlap,
            activation_rate_si_s=overlap_rate,
            constraints_satisfied=constraints_satisfied,
            can_engage_later=(
                "PITCH_CIRCLES_NOT_TANGENT" not in blockers
            ),
            metrics={
                "axis_orthogonality_error_rad": (
                    axis_orthogonality_error
                ),
                "face_axis_parallel_error_rad": (
                    face_axis_parallel_error
                ),
                "face_overlap_m": face_overlap,
                "measured_pitch_line_offset_m": (
                    measured_pitch_line_offset
                ),
                "required_pitch_line_offset_m": (
                    self.required_pitch_line_offset_m
                ),
                "pitch_line_offset_error_m": pitch_line_offset_error,
            },
            blockers=tuple(blockers),
        )
