from __future__ import annotations

"""Finite action envelopes for non-contact compliant supports.

Penalty contacts already stop producing force when their surfaces separate.
An analytical bushing or spring does not: without an explicit live-geometry
gate its force grows forever with displacement.  This module supplies the
mechanism-agnostic bottom line for those force paths.  A family may add a more
specific pin, track, sleeve, gear, or spring geometry gate, but it may not omit
the bounded endpoint envelope.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .conditional_interface import (
    AT_OR_BELOW,
    ConditionalCompliantInterface,
    InterfaceGeometryObservation,
    conditional_transmission_gate,
)
from .contact_intent import collision_component_inventory
from .models import MechanismValidationError, finite, identifier, vector3


BOUNDED_SUPPORT_PLUGIN_ID = "bounded_compliant_support_envelope_v1"
CONTINUE_WITH_ZERO_FORCE = "CONTINUE_WITH_ZERO_FORCE"
RELATIVE_ANCHOR_DRIFT = "RELATIVE_ANCHOR_DRIFT"
TRANSVERSE_ANCHOR_DRIFT = "TRANSVERSE_ANCHOR_DRIFT"
SUPPORTED_COORDINATES = {
    RELATIVE_ANCHOR_DRIFT,
    TRANSVERSE_ANCHOR_DRIFT,
}


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _scale(
    value: Sequence[float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(float(item) * factor for item in value)


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(
    value: Sequence[float],
    label: str,
) -> tuple[float, float, float]:
    return vector3(tuple(value), label, unit=True, allow_zero=False)


def body_component_refs(
    collision_geometry: Mapping[str, Any],
    body_id: str,
) -> tuple[str, ...]:
    """Return the authoritative STEP inventory owned by one rigid body."""

    requested = identifier(body_id, "body_id")
    refs = tuple(
        str(item["component_ref"])
        for item in collision_component_inventory(collision_geometry)
        if item["body_id"] == requested
    )
    if not refs:
        raise _fail(
            "BOUNDED_SUPPORT_STEP_BINDING_REQUIRED",
            f"no authoritative STEP components belong to {requested!r}",
        )
    return refs


def bounded_support_gate(
    *,
    body_a_id: str,
    body_b_id: str,
    endpoint_a_component_refs: Sequence[str],
    endpoint_b_component_refs: Sequence[str],
    maximum_drift_m: float,
    coordinate_mode: str,
    proof_source: str,
    maximum_axis_misalignment_rad: float | None = None,
    template_id: str,
) -> dict[str, Any]:
    """Build the mandatory finite-action gate for a compliant support.

    ``maximum_drift_m`` is the outer certified force envelope.  It normally
    comes from a stricter STEP-derived capture range or, when such a feature
    template is unavailable, from the package's already bounded acceptance
    envelope.  The latter is conservative: the support stops no later than the
    point at which the run could no longer certify.
    """

    maximum = finite(maximum_drift_m, "maximum_drift_m")
    if maximum <= 0.0:
        raise _fail(
            "BOUNDED_SUPPORT_RANGE_INVALID",
            "maximum_drift_m must be positive",
        )
    mode = str(coordinate_mode)
    if mode not in SUPPORTED_COORDINATES:
        raise _fail(
            "BOUNDED_SUPPORT_COORDINATE_INVALID",
            f"unsupported coordinate mode: {mode!r}",
        )
    angular = (
        None
        if maximum_axis_misalignment_rad is None
        else finite(
            maximum_axis_misalignment_rad,
            "maximum_axis_misalignment_rad",
        )
    )
    if angular is not None and angular <= 0.0:
        raise _fail(
            "BOUNDED_SUPPORT_RANGE_INVALID",
            "maximum_axis_misalignment_rad must be positive",
        )
    return conditional_transmission_gate(
        geometry_plugin_id=BOUNDED_SUPPORT_PLUGIN_ID,
        body_a_id=body_a_id,
        body_b_id=body_b_id,
        endpoint_a_component_refs=endpoint_a_component_refs,
        endpoint_b_component_refs=endpoint_b_component_refs,
        activation_coordinate=(
            "relative_anchor_drift_m"
            if mode == RELATIVE_ANCHOR_DRIFT
            else "transverse_anchor_drift_m"
        ),
        # The inner threshold is only a re-entry hysteresis boundary.  A
        # support transmits exact zero beyond the outer boundary.  Re-entry is
        # allowed only after the template's live geometry reaches this stricter
        # inner threshold; the physical simulation itself keeps running.
        enter_threshold_si=0.9 * maximum,
        release_threshold_si=maximum,
        activation_direction=AT_OR_BELOW,
        proof_source=proof_source,
        geometry_parameters={
            "template_id": identifier(template_id, "template_id"),
            "coordinate_mode": mode,
            "maximum_drift_m": maximum,
            "maximum_axis_misalignment_rad": angular,
            "loss_policy": CONTINUE_WITH_ZERO_FORCE,
            "range_authority": (
                "STEP_TEMPLATE_OR_CERTIFIABLE_ACCEPTANCE_ENVELOPE"
            ),
            "force_outside_range": "EXACT_ZERO",
        },
    )


def validate_bounded_support_gate(
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate that a conditional gate is a structural support envelope."""

    from .conditional_interface import validate_conditional_transmission_gate

    normalized = validate_conditional_transmission_gate(gate)
    if normalized["geometry_plugin_id"] != BOUNDED_SUPPORT_PLUGIN_ID:
        raise _fail(
            "BOUNDED_SUPPORT_GATE_INVALID",
            "compliant support must use the bounded support geometry plugin",
        )
    parameters = normalized["geometry_parameters"]
    mode = str(parameters.get("coordinate_mode") or "")
    if mode not in SUPPORTED_COORDINATES:
        raise _fail(
            "BOUNDED_SUPPORT_GATE_INVALID",
            "bounded support coordinate mode is invalid",
        )
    maximum = finite(parameters.get("maximum_drift_m"), "maximum_drift_m")
    if maximum <= 0.0 or normalized["release_threshold_si"] != maximum:
        raise _fail(
            "BOUNDED_SUPPORT_GATE_INVALID",
            "bounded support release threshold must equal its finite range",
        )
    if parameters.get("loss_policy") != CONTINUE_WITH_ZERO_FORCE:
        raise _fail(
            "BOUNDED_SUPPORT_GATE_INVALID",
            "a separated compliant support must continue with zero force",
        )
    if parameters.get("force_outside_range") != "EXACT_ZERO":
        raise _fail(
            "BOUNDED_SUPPORT_GATE_INVALID",
            "compliant support must transmit exact zero outside its range",
        )
    identifier(parameters.get("template_id"), "template_id")
    angular = parameters.get("maximum_axis_misalignment_rad")
    if angular is not None and finite(
        angular,
        "maximum_axis_misalignment_rad",
    ) <= 0.0:
        raise _fail(
            "BOUNDED_SUPPORT_GATE_INVALID",
            "bounded support angular range must be positive",
        )
    return normalized


@dataclass
class BoundedSupportRuntime:
    """Runtime state and evidence for one finite compliant force path."""

    interface_id: str
    gate: Mapping[str, Any]

    def __post_init__(self) -> None:
        self.interface_id = identifier(self.interface_id, "interface_id")
        self.gate = validate_bounded_support_gate(self.gate)
        self.controller = ConditionalCompliantInterface(self.gate)
        self.evaluation_count = 0
        self.state_path: list[str] = []
        self.max_inactive_abs_transmitted_effort_si = 0.0
        self.current_state: dict[str, Any] | None = None
        self.ever_engaged = False

    @property
    def parameters(self) -> Mapping[str, Any]:
        return self.gate["geometry_parameters"]

    def evaluate(
        self,
        *,
        time_s: float,
        anchor_a_abs_m: Sequence[float],
        anchor_b_abs_m: Sequence[float],
        velocity_a_abs_m_s: Sequence[float],
        velocity_b_abs_m_s: Sequence[float],
        requested_effort_si: float,
        free_axis_abs: Sequence[float] | None = None,
        axis_misalignment_rad: float | None = None,
        additional_constraints_satisfied: bool = True,
        additional_metrics: Mapping[str, float] | None = None,
        additional_blockers: Sequence[str] = (),
    ) -> dict[str, Any]:
        displacement = _sub(anchor_b_abs_m, anchor_a_abs_m)
        relative_velocity = _sub(
            velocity_b_abs_m_s,
            velocity_a_abs_m_s,
        )
        mode = str(self.parameters["coordinate_mode"])
        if mode == TRANSVERSE_ANCHOR_DRIFT:
            if free_axis_abs is None:
                raise _fail(
                    "BOUNDED_SUPPORT_AXIS_REQUIRED",
                    "transverse support envelope requires a live free axis",
                )
            axis = _unit(free_axis_abs, "free_axis_abs")
            displacement = _sub(
                displacement,
                _scale(axis, _dot(displacement, axis)),
            )
            relative_velocity = _sub(
                relative_velocity,
                _scale(axis, _dot(relative_velocity, axis)),
            )
        coordinate = _norm(displacement)
        rate = (
            _dot(displacement, relative_velocity) / coordinate
            if coordinate > 1.0e-15
            else 0.0
        )
        maximum_angle = self.parameters.get(
            "maximum_axis_misalignment_rad"
        )
        angle_ok = (
            True
            if maximum_angle is None
            else (
                axis_misalignment_rad is not None
                and finite(
                    axis_misalignment_rad,
                    "axis_misalignment_rad",
                )
                <= float(maximum_angle)
            )
        )
        if not isinstance(additional_constraints_satisfied, bool):
            raise _fail(
                "BOUNDED_SUPPORT_OBSERVATION_INVALID",
                "additional_constraints_satisfied must be boolean",
            )
        blockers = [
            *(() if angle_ok else ("support_axis_misaligned",)),
            *(
                ()
                if additional_constraints_satisfied
                else tuple(
                    identifier(item, "additional_blocker")
                    for item in additional_blockers
                )
            ),
        ]
        constraints_satisfied = (
            angle_ok and additional_constraints_satisfied
        )
        decision = self.controller.step(
            InterfaceGeometryObservation(
                time_s=float(time_s),
                activation_coordinate_si=coordinate,
                activation_rate_si_s=rate,
                constraints_satisfied=constraints_satisfied,
                can_engage_later=True,
                metrics={
                    "support_drift_m": coordinate,
                    **(
                        {}
                        if axis_misalignment_rad is None
                        else {
                            "axis_misalignment_rad": float(
                                axis_misalignment_rad
                            )
                        }
                    ),
                    **{
                        identifier(key, "additional_metric"): finite(
                            value,
                            f"additional_metrics.{key}",
                        )
                        for key, value in (
                            additional_metrics or {}
                        ).items()
                    },
                },
                blockers=tuple(blockers),
            )
        )
        requested = abs(finite(requested_effort_si, "requested_effort_si"))
        transmitted = decision.transmit_scalar(requested)
        if not decision.transmission_active:
            self.max_inactive_abs_transmitted_effort_si = max(
                self.max_inactive_abs_transmitted_effort_si,
                abs(transmitted),
            )
        self.evaluation_count += 1
        if not self.state_path or self.state_path[-1] != decision.state:
            self.state_path.append(decision.state)
        self.ever_engaged = self.ever_engaged or decision.transmission_active
        self.current_state = {
            "interface_id": self.interface_id,
            "state": decision.state,
            "transmission_active": decision.transmission_active,
            "requested_effort_si": requested,
            "transmitted_effort_si": transmitted,
            "support_drift_m": coordinate,
            "axis_misalignment_rad": axis_misalignment_rad,
            "loss_policy": CONTINUE_WITH_ZERO_FORCE,
            "reason": decision.reason,
            "observation": decision.observation.to_dict(),
        }
        return deepcopy(self.current_state)

    def disengaged(self) -> bool:
        return bool(
            self.ever_engaged
            and self.current_state is not None
            and not self.current_state["transmission_active"]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "interface_id": self.interface_id,
            "evaluation_count": self.evaluation_count,
            "state_path": list(self.state_path),
            "max_inactive_abs_transmitted_effort_si": (
                self.max_inactive_abs_transmitted_effort_si
            ),
            "loss_policy": CONTINUE_WITH_ZERO_FORCE,
            "range_m": float(self.parameters["maximum_drift_m"]),
        }


def conditional_runtime_summary(
    supports: Sequence[BoundedSupportRuntime],
    *,
    physics_step_count: int,
    additional_interface_summaries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "schema_version": "conditional_interface_runtime_v1",
        "evaluated_every_physics_step": True,
        "physics_step_count": int(physics_step_count),
        "interfaces": [
            *(item.summary() for item in supports),
            *(deepcopy(dict(item)) for item in additional_interface_summaries),
        ],
    }
