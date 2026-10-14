from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .compound_spec import CompoundBodySpec, _body_from_dict
from .models import MechanismValidationError, finite, identifier, vector3


@dataclass(frozen=True)
class RackPinionSpec:
    """Immutable execution contract for one hand-driven rack-pinion mechanism."""

    mechanism_kind: str
    mechanism_id: str
    bodies: tuple[CompoundBodySpec, ...]
    fixed_body_id: str
    pinion_body_id: str
    rack_body_id: str
    pinion_origin_abs_m: tuple[float, float, float]
    pinion_axis_abs: tuple[float, float, float]
    rack_axis_abs: tuple[float, float, float]
    pinion_pitch_radius_m: float
    pressure_angle_rad: float
    home_progress_m: float
    outbound_progress_m: float
    nominal_input_torque_nm: float
    return_input_torque_nm: float
    max_torque_phase_s: float
    max_return_phase_s: float
    settle_phase_s: float
    output_progress_tolerance_m: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    max_coupling_error_m: float
    friction: float = 0.08
    geometry_reference: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bodies"] = [asdict(item) for item in self.bodies]
        return payload


def _unit3(value: Any, label: str) -> tuple[float, float, float]:
    return vector3(tuple(value), label, unit=True, allow_zero=False)


def _point3(value: Any, label: str) -> tuple[float, float, float]:
    return vector3(tuple(value), label, allow_zero=True)


def rack_pinion_from_dict(value: Mapping[str, Any] | None) -> RackPinionSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "RACK_PINION_SPEC_REQUIRED",
            "rack-pinion mechanism adapter requires a family_spec object",
        )
    if value.get("mechanism_kind") != "rack_pinion":
        raise MechanismValidationError(
            "RACK_PINION_KIND_INVALID",
            "rack-pinion family_spec must declare mechanism_kind='rack_pinion'",
        )
    mechanism_id = str(value.get("mechanism_id") or "")
    identifier(mechanism_id, "family_spec.mechanism_id")
    raw_bodies = value.get("bodies", ())
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(raw_bodies)
        if isinstance(item, Mapping)
    )
    if len(bodies) != len(tuple(raw_bodies)) or len(bodies) < 3:
        raise MechanismValidationError(
            "RACK_PINION_BODY_COUNT_INVALID",
            "rack-pinion mechanism requires at least ground, pinion, and rack bodies",
        )
    body_ids = {item.body_id for item in bodies}
    if len(body_ids) != len(bodies):
        raise MechanismValidationError(
            "RACK_PINION_BODY_ID_DUPLICATE", "rack-pinion body IDs must be unique"
        )
    groups = [group for body in bodies for group in body.group_ids]
    if len(groups) != len(set(groups)):
        raise MechanismValidationError(
            "RACK_PINION_GROUP_OVERLAP",
            "a rigid group cannot belong to multiple rack-pinion bodies",
        )
    fixed_body_id = str(value.get("fixed_body_id") or "")
    pinion_body_id = str(value.get("pinion_body_id") or "")
    rack_body_id = str(value.get("rack_body_id") or "")
    for body_id, label in (
        (fixed_body_id, "fixed_body_id"),
        (pinion_body_id, "pinion_body_id"),
        (rack_body_id, "rack_body_id"),
    ):
        if body_id not in body_ids:
            raise MechanismValidationError(
                "RACK_PINION_BODY_UNKNOWN", f"{label} references an unknown body"
            )
    if len({fixed_body_id, pinion_body_id, rack_body_id}) != 3:
        raise MechanismValidationError(
            "RACK_PINION_BODY_ROLE_OVERLAP",
            "ground, pinion, and rack must be distinct rigid bodies",
        )
    fixed = [item.body_id for item in bodies if item.fixed]
    if fixed != [fixed_body_id]:
        raise MechanismValidationError(
            "RACK_PINION_FIXED_BODY_INVALID",
            "rack-pinion mechanism requires exactly its declared fixed body to be fixed",
        )

    pinion_axis = _unit3(value.get("pinion_axis_abs"), "pinion_axis_abs")
    rack_axis = _unit3(value.get("rack_axis_abs"), "rack_axis_abs")
    if abs(sum(a * b for a, b in zip(pinion_axis, rack_axis))) > 1e-6:
        raise MechanismValidationError(
            "RACK_PINION_AXIS_NOT_ORTHOGONAL",
            "rack travel axis must be perpendicular to the pinion axis",
        )
    radius = finite(value.get("pinion_pitch_radius_m"), "pinion_pitch_radius_m")
    pressure = finite(value.get("pressure_angle_rad", math.radians(20.0)), "pressure_angle_rad")
    if radius <= 0 or not 0 < pressure < math.pi / 2:
        raise MechanismValidationError(
            "RACK_PINION_GEAR_GEOMETRY_INVALID",
            "pitch radius and pressure angle must be physically valid",
        )
    home = finite(value.get("home_progress_m"), "home_progress_m")
    outbound = finite(value.get("outbound_progress_m"), "outbound_progress_m")
    if outbound <= home:
        raise MechanismValidationError(
            "RACK_PINION_TRAVEL_INVALID",
            "outbound rack progress must exceed home progress",
        )
    nominal_torque = finite(value.get("nominal_input_torque_nm"), "nominal_input_torque_nm")
    return_torque = finite(value.get("return_input_torque_nm"), "return_input_torque_nm")
    if nominal_torque <= 0 or return_torque <= 0:
        raise MechanismValidationError(
            "RACK_PINION_INPUT_TORQUE_INVALID",
            "outbound and return crank torques must be positive",
        )
    max_torque_phase = finite(value.get("max_torque_phase_s"), "max_torque_phase_s")
    max_return_phase = finite(value.get("max_return_phase_s"), "max_return_phase_s")
    settle_phase = finite(value.get("settle_phase_s"), "settle_phase_s")
    if max_torque_phase <= 0 or max_return_phase <= 0 or settle_phase < 0:
        raise MechanismValidationError(
            "RACK_PINION_PHASE_DURATION_INVALID",
            "rack-pinion phase durations are invalid",
        )
    thresholds = {
        "output_progress_tolerance_m": finite(
            value.get("output_progress_tolerance_m"), "output_progress_tolerance_m"
        ),
        "max_support_drift_m": finite(
            value.get("max_support_drift_m"), "max_support_drift_m"
        ),
        "max_off_axis_rotation_rad": finite(
            value.get("max_off_axis_rotation_rad"), "max_off_axis_rotation_rad"
        ),
        "max_coupling_error_m": finite(
            value.get("max_coupling_error_m"), "max_coupling_error_m"
        ),
    }
    if any(item <= 0 for item in thresholds.values()):
        raise MechanismValidationError(
            "RACK_PINION_THRESHOLD_INVALID",
            "rack-pinion comparison thresholds must be positive",
        )
    friction = finite(value.get("friction", 0.08), "friction")
    if friction < 0:
        raise MechanismValidationError(
            "RACK_PINION_FRICTION_INVALID", "friction cannot be negative"
        )
    geometry_reference = value.get("geometry_reference")
    if geometry_reference is not None and not isinstance(geometry_reference, Mapping):
        raise MechanismValidationError(
            "RACK_PINION_GEOMETRY_REFERENCE_INVALID",
            "geometry_reference must be an object when present",
        )
    return RackPinionSpec(
        mechanism_kind="rack_pinion",
        mechanism_id=mechanism_id,
        bodies=bodies,
        fixed_body_id=fixed_body_id,
        pinion_body_id=pinion_body_id,
        rack_body_id=rack_body_id,
        pinion_origin_abs_m=_point3(
            value.get("pinion_origin_abs_m"), "pinion_origin_abs_m"
        ),
        pinion_axis_abs=pinion_axis,
        rack_axis_abs=rack_axis,
        pinion_pitch_radius_m=radius,
        pressure_angle_rad=pressure,
        home_progress_m=home,
        outbound_progress_m=outbound,
        nominal_input_torque_nm=nominal_torque,
        return_input_torque_nm=return_torque,
        max_torque_phase_s=max_torque_phase,
        max_return_phase_s=max_return_phase,
        settle_phase_s=settle_phase,
        output_progress_tolerance_m=thresholds["output_progress_tolerance_m"],
        max_support_drift_m=thresholds["max_support_drift_m"],
        max_off_axis_rotation_rad=thresholds["max_off_axis_rotation_rad"],
        max_coupling_error_m=thresholds["max_coupling_error_m"],
        friction=friction,
        geometry_reference=(
            dict(geometry_reference) if geometry_reference is not None else {}
        ),
    )
