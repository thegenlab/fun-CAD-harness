from __future__ import annotations

"""Typed, bounded constitutive data for a spring on one joint coordinate."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .models import MechanismValidationError


@dataclass(frozen=True)
class AxisSpringContactFixtureSpec:
    shape: str
    center_abs_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    friction: float
    hold_duration_s: float
    mass_kg: float | None = None
    grip_torque_nm: float | None = None
    loading_support_center_abs_m: tuple[float, float, float] | None = None
    loading_support_size_m: tuple[float, float, float] | None = None
    table_center_abs_m: tuple[float, float, float] | None = None
    table_size_m: tuple[float, float, float] | None = None
    approach_offset_abs_m: tuple[float, float, float] | None = None
    grip_offset_abs_m: tuple[float, float, float] | None = None
    lift_offset_abs_m: tuple[float, float, float] | None = None
    lift_duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["center_abs_m"] = list(self.center_abs_m)
        value["size_m"] = list(self.size_m)
        if self.loading_support_center_abs_m is not None:
            value["loading_support_center_abs_m"] = list(
                self.loading_support_center_abs_m
            )
        if self.loading_support_size_m is not None:
            value["loading_support_size_m"] = list(self.loading_support_size_m)
        for field in (
            "table_center_abs_m",
            "table_size_m",
            "approach_offset_abs_m",
            "grip_offset_abs_m",
            "lift_offset_abs_m",
        ):
            vector = getattr(self, field)
            if vector is not None:
                value[field] = list(vector)
        return value


@dataclass(frozen=True)
class AxisSpringSpec:
    schema_version: str
    mechanism_kind: str
    joint_family: str
    rest_position_si: float
    stiffness_si: float
    damping_si: float
    maximum_deflection_si: float
    maximum_stop_penetration_si: float
    bearing_off_axis_stiffness_si: float
    bearing_off_axis_damping_si: float
    moving_mass_kg: float
    contact_fixture: AxisSpringContactFixtureSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.contact_fixture is not None:
            value["contact_fixture"] = self.contact_fixture.to_dict()
        return value


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail("AXIS_SPRING_VALUE_INVALID", f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise _fail("AXIS_SPRING_VALUE_INVALID", f"{field} must be finite")
    return result


def _tuple3(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            f"{field} must contain three numbers",
        )
    return tuple(_finite(item, field) for item in value)


def _contact_fixture(value: Any) -> AxisSpringContactFixtureSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            "contact_fixture must be an object",
        )
    allowed = {
        "shape",
        "center_abs_m",
        "size_m",
        "friction",
        "hold_duration_s",
        "mass_kg",
        "grip_torque_nm",
        "loading_support_center_abs_m",
        "loading_support_size_m",
        "table_center_abs_m",
        "table_size_m",
        "approach_offset_abs_m",
        "grip_offset_abs_m",
        "lift_offset_abs_m",
        "lift_duration_s",
    }
    unknown = set(value) - allowed
    if unknown:
        raise _fail(
            "AXIS_SPRING_FIXTURE_FIELD_UNKNOWN",
            "contact fixture contains unknown fields: "
            + ", ".join(sorted(unknown)),
        )
    shape = value.get("shape")
    if shape not in {"fixed_box", "dynamic_grip_box", "tabletop_pickup_box"}:
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            "contact fixture shape must be fixed_box, dynamic_grip_box, "
            "or tabletop_pickup_box",
        )
    center = _tuple3(
        value.get("center_abs_m"),
        "contact_fixture.center_abs_m",
    )
    size = _tuple3(value.get("size_m"), "contact_fixture.size_m")
    friction = _finite(value.get("friction"), "contact_fixture.friction")
    hold_duration = _finite(
        value.get("hold_duration_s"),
        "contact_fixture.hold_duration_s",
    )
    if any(abs(item) > 2.0 for item in center):
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            "contact fixture center exceeds the small-machine workspace",
        )
    if any(item < 0.001 or item > 0.25 for item in size):
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            "contact fixture size must stay within 1 mm to 250 mm",
        )
    if not 0.0 <= friction <= 2.0 or not 0.02 <= hold_duration <= 2.0:
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            "contact fixture friction or hold duration is out of range",
        )
    mass = value.get("mass_kg")
    grip_torque = value.get("grip_torque_nm")
    support_center = value.get("loading_support_center_abs_m")
    support_size = value.get("loading_support_size_m")
    table_center = value.get("table_center_abs_m")
    table_size = value.get("table_size_m")
    approach_offset = value.get("approach_offset_abs_m")
    grip_offset = value.get("grip_offset_abs_m")
    lift_offset = value.get("lift_offset_abs_m")
    lift_duration = value.get("lift_duration_s")
    grip_torque_value = (
        None
        if grip_torque is None
        else _finite(grip_torque, "contact_fixture.grip_torque_nm")
    )
    if (
        grip_torque_value is not None
        and (
            shape not in {"dynamic_grip_box", "tabletop_pickup_box"}
            or not 0.05 <= grip_torque_value <= 1.5
        )
    ):
        raise _fail(
            "AXIS_SPRING_FIXTURE_INVALID",
            "grip torque is available only for dynamic grip fixtures and must "
            "stay within 0.05 N*m to 1.5 N*m",
        )
    if shape == "dynamic_grip_box":
        mass_value = _finite(mass, "contact_fixture.mass_kg")
        if not 0.001 <= mass_value <= 0.5:
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "dynamic grip box mass must stay within 1 g to 500 g",
            )
        if support_center is None or support_size is None:
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "dynamic grip box requires a loading support center and size",
            )
        support_center_value = _tuple3(
            support_center, "contact_fixture.loading_support_center_abs_m"
        )
        support_size_value = _tuple3(
            support_size, "contact_fixture.loading_support_size_m"
        )
        if any(abs(item) > 2.0 for item in support_center_value) or any(
            item < 0.001 or item > 0.25 for item in support_size_value
        ):
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "dynamic grip loading support exceeds bounded fixture dimensions",
            )
        if any(
            item is not None
            for item in (
                table_center,
                table_size,
                approach_offset,
                grip_offset,
                lift_offset,
                lift_duration,
            )
        ):
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "dynamic grip box does not accept tabletop carrier fields",
            )
        table_center_value = None
        table_size_value = None
        approach_offset_value = None
        grip_offset_value = None
        lift_offset_value = None
        lift_duration_value = None
    elif shape == "tabletop_pickup_box":
        mass_value = _finite(mass, "contact_fixture.mass_kg")
        if not 0.001 <= mass_value <= 0.5:
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "tabletop pickup box mass must stay within 1 g to 500 g",
            )
        if support_center is not None or support_size is not None:
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "tabletop pickup box does not accept a temporary loading support",
            )
        required_tabletop = (
            table_center,
            table_size,
            approach_offset,
            grip_offset,
            lift_offset,
        )
        if any(item is None for item in required_tabletop):
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "tabletop pickup box requires table geometry and all carrier offsets",
            )
        table_center_value = _tuple3(
            table_center, "contact_fixture.table_center_abs_m"
        )
        table_size_value = _tuple3(table_size, "contact_fixture.table_size_m")
        approach_offset_value = _tuple3(
            approach_offset, "contact_fixture.approach_offset_abs_m"
        )
        grip_offset_value = _tuple3(
            grip_offset, "contact_fixture.grip_offset_abs_m"
        )
        lift_offset_value = _tuple3(
            lift_offset, "contact_fixture.lift_offset_abs_m"
        )
        lift_duration_value = (
            0.40
            if lift_duration is None
            else _finite(lift_duration, "contact_fixture.lift_duration_s")
        )
        if not 0.20 <= lift_duration_value <= 2.0:
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "tabletop pickup lift duration must stay within 0.2 s to 2.0 s",
            )
        if any(abs(item) > 2.0 for item in table_center_value) or any(
            item < 0.001 or item > 0.5 for item in table_size_value
        ):
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "tabletop pickup table exceeds bounded fixture dimensions",
            )
        if any(
            abs(item) > 0.25
            for vector in (
                approach_offset_value,
                grip_offset_value,
                lift_offset_value,
            )
            for item in vector
        ):
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "tabletop carrier offset exceeds the small-machine workspace",
            )
        support_center_value = None
        support_size_value = None
    else:
        if any(
            item is not None
            for item in (
                mass,
                grip_torque,
                support_center,
                support_size,
                table_center,
                table_size,
                approach_offset,
                grip_offset,
                lift_offset,
                lift_duration,
            )
        ):
            raise _fail(
                "AXIS_SPRING_FIXTURE_INVALID",
                "fixed_box does not accept dynamic or tabletop fields",
            )
        mass_value = None
        support_center_value = None
        support_size_value = None
        table_center_value = None
        table_size_value = None
        approach_offset_value = None
        grip_offset_value = None
        lift_offset_value = None
        lift_duration_value = None
    return AxisSpringContactFixtureSpec(
        shape=str(shape),
        center_abs_m=center,
        size_m=size,
        friction=friction,
        hold_duration_s=hold_duration,
        mass_kg=mass_value,
        grip_torque_nm=grip_torque_value,
        loading_support_center_abs_m=support_center_value,
        loading_support_size_m=support_size_value,
        table_center_abs_m=table_center_value,
        table_size_m=table_size_value,
        approach_offset_abs_m=approach_offset_value,
        grip_offset_abs_m=grip_offset_value,
        lift_offset_abs_m=lift_offset_value,
        lift_duration_s=lift_duration_value,
    )


def axis_spring_from_dict(
    value: Mapping[str, Any] | None,
    *,
    expected_family: str | None = None,
) -> AxisSpringSpec:
    if not isinstance(value, Mapping):
        raise _fail(
            "AXIS_SPRING_SPEC_INVALID",
            "axis spring family spec must be an object",
        )
    allowed = {
        "schema_version",
        "mechanism_kind",
        "joint_family",
        "rest_position_si",
        "stiffness_si",
        "damping_si",
        "maximum_deflection_si",
        "maximum_stop_penetration_si",
        "bearing_off_axis_stiffness_si",
        "bearing_off_axis_damping_si",
        "moving_mass_kg",
        "contact_fixture",
    }
    unknown = set(value) - allowed
    if unknown:
        raise _fail(
            "AXIS_SPRING_FIELD_UNKNOWN",
            "axis spring family spec contains unknown fields: "
            + ", ".join(sorted(unknown)),
        )
    if value.get("schema_version") != "0.1":
        raise _fail(
            "AXIS_SPRING_SCHEMA_INVALID",
            "axis spring schema_version must be 0.1",
        )
    if value.get("mechanism_kind") != "axis_spring":
        raise _fail(
            "AXIS_SPRING_KIND_INVALID",
            "mechanism_kind must be axis_spring",
        )
    family = value.get("joint_family")
    if family not in {"revolute", "prismatic"}:
        raise _fail(
            "AXIS_SPRING_FAMILY_INVALID",
            "joint_family must be revolute or prismatic",
        )
    if expected_family is not None and family != expected_family:
        raise _fail(
            "AXIS_SPRING_FAMILY_MISMATCH",
            f"axis spring joint_family must match {expected_family}",
        )
    rest = _finite(value.get("rest_position_si"), "rest_position_si")
    stiffness = _finite(value.get("stiffness_si"), "stiffness_si")
    damping = _finite(value.get("damping_si"), "damping_si")
    maximum = _finite(
        value.get("maximum_deflection_si"),
        "maximum_deflection_si",
    )
    maximum_stop_penetration = _finite(
        value.get(
            "maximum_stop_penetration_si",
            math.radians(0.5) if family == "revolute" else 0.0005,
        ),
        "maximum_stop_penetration_si",
    )
    bearing_stiffness = _finite(
        value.get("bearing_off_axis_stiffness_si", 10.0),
        "bearing_off_axis_stiffness_si",
    )
    bearing_damping = _finite(
        value.get("bearing_off_axis_damping_si", 0.1),
        "bearing_off_axis_damping_si",
    )
    moving_mass = _finite(value.get("moving_mass_kg", 0.45), "moving_mass_kg")
    if abs(rest) > 1e-12:
        raise _fail(
            "AXIS_SPRING_REST_UNSUPPORTED",
            "the first axis-spring runtime supports only the authored home pose "
            "as zero spring deflection",
        )
    if (
        stiffness <= 0.0
        or damping < 0.0
        or maximum <= 0.0
        or maximum_stop_penetration <= 0.0
        or bearing_stiffness <= 0.0
        or bearing_damping < 0.0
        or moving_mass <= 0.0
    ):
        raise _fail(
            "AXIS_SPRING_VALUE_INVALID",
            "spring stiffness, maximum deflection, stop penetration and bearing "
            "stiffness must be positive; damping values must be non-negative",
        )
    if family == "revolute":
        if (
            stiffness > 1_000.0
            or damping > 100.0
            or maximum > 2.0 * math.pi
            or maximum_stop_penetration > math.radians(5.0)
            or bearing_stiffness > 10_000.0
            or bearing_damping > 1_000.0
            or moving_mass > 5.0
        ):
            raise _fail(
                "AXIS_SPRING_VALUE_OUT_OF_RANGE",
                "revolute spring data exceeds the bounded small-machine range",
            )
    else:
        if (
            stiffness > 1_000_000.0
            or damping > 100_000.0
            or maximum > 1.0
            or maximum_stop_penetration > 0.05
            or bearing_stiffness > 10_000.0
            or bearing_damping > 1_000.0
            or moving_mass > 5.0
        ):
            raise _fail(
                "AXIS_SPRING_VALUE_OUT_OF_RANGE",
                "prismatic spring data exceeds the bounded small-machine range",
            )
    fixture = _contact_fixture(value.get("contact_fixture"))
    return AxisSpringSpec(
        schema_version="0.1",
        mechanism_kind="axis_spring",
        joint_family=str(family),
        rest_position_si=rest,
        stiffness_si=stiffness,
        damping_si=damping,
        maximum_deflection_si=maximum,
        maximum_stop_penetration_si=maximum_stop_penetration,
        bearing_off_axis_stiffness_si=bearing_stiffness,
        bearing_off_axis_damping_si=bearing_damping,
        moving_mass_kg=moving_mass,
        contact_fixture=fixture,
    )
