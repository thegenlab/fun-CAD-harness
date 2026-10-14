from __future__ import annotations

"""Closed execution contract for straight-line bicycle ground rolling."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from .compound_spec import CompoundBodySpec, _body_from_dict
from .models import MechanismValidationError, finite, identifier, vector3


BICYCLE_GROUND_ROLL_PLUGIN_ID = "bicycle_ground_roll_v1"
BICYCLE_GROUND_ROLL_KIND = "bicycle_ground_roll"
BICYCLE_GROUND_ROLL_PROGRAM = "task_bicycle_ground_roll"
NOMINAL_REAR_DRIVE_TORQUE_NM = 15.0
MINIMUM_GROUND_RUNOUT_M = 2.75
MINIMUM_FRONT_ROTATION_RAD = 5.0
MINIMUM_CHASSIS_TRAVEL_M = 1.50
MAXIMUM_ROLLING_SLIP_RATIO = 0.12
MAXIMUM_AXLE_DRIFT_M = 0.004
MAXIMUM_GUIDE_DRIFT_M = 0.006
MAXIMUM_OFF_AXIS_ROTATION_RAD = math.radians(2.0)
MAXIMUM_GROUND_PENETRATION_M = 0.004
MAXIMUM_GROUND_SEPARATION_M = 0.0002
MINIMUM_POWERED_CONTACT_COVERAGE = 0.90


@dataclass(frozen=True)
class BicycleGroundRollSpec:
    mechanism_kind: str
    mechanism_id: str
    bodies: tuple[CompoundBodySpec, ...]
    ground_body_id: str
    chassis_body_id: str
    steering_body_id: str
    rear_wheel_body_id: str
    front_wheel_body_id: str
    rear_axle_origin_abs_m: tuple[float, float, float]
    front_axle_origin_abs_m: tuple[float, float, float]
    steering_origin_abs_m: tuple[float, float, float]
    wheel_axis_abs: tuple[float, float, float]
    steering_axis_abs: tuple[float, float, float]
    travel_axis_abs: tuple[float, float, float]
    ground_normal_abs: tuple[float, float, float]
    ground_plane_height_m: float
    rear_wheel_radius_m: float
    front_wheel_radius_m: float
    nominal_rear_drive_torque_nm: float
    ground_friction: float
    maximum_axle_drift_m: float
    maximum_guide_drift_m: float
    maximum_off_axis_rotation_rad: float
    maximum_rolling_slip_ratio: float
    maximum_ground_penetration_m: float
    maximum_ground_separation_m: float
    minimum_powered_contact_coverage: float
    minimum_front_rotation_rad: float
    minimum_chassis_travel_m: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bodies"] = [asdict(item) for item in self.bodies]
        return payload


def _point3(value: Any, label: str) -> tuple[float, float, float]:
    return vector3(value, label, allow_zero=True)


def _unit3(value: Any, label: str) -> tuple[float, float, float]:
    return vector3(value, label, unit=True)


def _positive(value: Any, label: str) -> float:
    number = finite(value, label)
    if number <= 0.0:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_PARAMETER_INVALID",
            f"{label} must be positive",
        )
    return number


def _frozen(
    value: Any,
    expected: float,
    label: str,
    *,
    tolerance: float = 1.0e-12,
) -> float:
    number = finite(value, label)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=tolerance):
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_THRESHOLD_CHANGED",
            f"{label} must remain the public v1 value {expected!r}",
        )
    return number


def bicycle_ground_roll_from_dict(
    value: Mapping[str, Any] | None,
) -> BicycleGroundRollSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_SPEC_REQUIRED",
            "bicycle ground rolling requires a family_spec object",
        )
    if value.get("mechanism_kind") != BICYCLE_GROUND_ROLL_KIND:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_KIND_INVALID",
            "mechanism_kind must be bicycle_ground_roll",
        )
    mechanism_id = str(value.get("mechanism_id") or "")
    identifier(mechanism_id, "family_spec.mechanism_id")
    raw_bodies = value.get("bodies")
    if not isinstance(raw_bodies, list):
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_BODY_COUNT_INVALID",
            "bodies must be a five-body array",
        )
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(raw_bodies)
        if isinstance(item, Mapping)
    )
    if len(bodies) != len(raw_bodies) or len(bodies) != 5:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_BODY_COUNT_INVALID",
            "ground roll requires ground, chassis, steering, rear wheel, and front wheel bodies",
        )
    body_ids = {item.body_id for item in bodies}
    groups = [group for body in bodies for group in body.group_ids]
    if len(body_ids) != len(bodies) or len(groups) != len(set(groups)):
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_BODY_BINDING_INVALID",
            "body IDs and rigid-group ownership must be unique",
        )
    role_fields = (
        "ground_body_id",
        "chassis_body_id",
        "steering_body_id",
        "rear_wheel_body_id",
        "front_wheel_body_id",
    )
    roles = {field: str(value.get(field) or "") for field in role_fields}
    if set(roles.values()) != body_ids or len(set(roles.values())) != 5:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_BODY_ROLE_INVALID",
            "the five body roles must bind every body exactly once",
        )
    fixed_ids = {item.body_id for item in bodies if item.fixed}
    if fixed_ids != {roles["ground_body_id"]}:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_FIXED_BODY_INVALID",
            "only the declared ground body may be fixed",
        )
    ground_body = next(
        item for item in bodies if item.body_id == roles["ground_body_id"]
    )
    if len(ground_body.group_ids) < 2:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_UPRIGHT_FIXTURE_MISSING",
            "the fixed body must own separate ground-surface and visible upright-fixture groups",
        )

    ground_min_x = ground_body.center_abs_m[0] - 0.5 * ground_body.size_m[0]
    ground_max_x = ground_body.center_abs_m[0] + 0.5 * ground_body.size_m[0]
    initial_axle_x = (
        _point3(value.get("rear_axle_origin_abs_m"), "rear_axle_origin_abs_m")[0],
        _point3(value.get("front_axle_origin_abs_m"), "front_axle_origin_abs_m")[0],
    )
    required_min_x = min(initial_axle_x) - 0.10
    required_max_x = max(initial_axle_x) + MINIMUM_GROUND_RUNOUT_M + 0.10
    if ground_min_x > required_min_x or ground_max_x < required_max_x:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_SURFACE_TOO_SHORT",
            "the finite ground surface must cover both initial wheel contacts, "
            "the public ground-runout envelope, and 0.10 m end margins",
        )

    travel = _unit3(value.get("travel_axis_abs"), "travel_axis_abs")
    wheel = _unit3(value.get("wheel_axis_abs"), "wheel_axis_abs")
    normal = _unit3(value.get("ground_normal_abs"), "ground_normal_abs")
    steering = _unit3(value.get("steering_axis_abs"), "steering_axis_abs")
    for left, right, label in (
        (travel, wheel, "travel/wheel"),
        (travel, normal, "travel/ground-normal"),
        (wheel, normal, "wheel/ground-normal"),
    ):
        if abs(sum(a * b for a, b in zip(left, right))) > 1.0e-6:
            raise MechanismValidationError(
                "BICYCLE_GROUND_ROLL_AXIS_INVALID",
                f"{label} axes must be perpendicular",
            )
    if abs(sum(a * b for a, b in zip(steering, normal))) < 1.0 - 1.0e-6:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_AXIS_INVALID",
            "the steering axis must be parallel to the ground normal",
        )
    # The v1 finite-guide matrix is intentionally defined in world XYZ.  A
    # future arbitrary-axis variant needs a separately reviewed frame mapping.
    expected_axes = {
        "travel_axis_abs": (1.0, 0.0, 0.0),
        "wheel_axis_abs": (0.0, 1.0, 0.0),
        "ground_normal_abs": (0.0, 0.0, 1.0),
        "steering_axis_abs": (0.0, 0.0, 1.0),
    }
    for label, actual in (
        ("travel_axis_abs", travel),
        ("wheel_axis_abs", wheel),
        ("ground_normal_abs", normal),
        ("steering_axis_abs", steering),
    ):
        if any(abs(a - b) > 1.0e-9 for a, b in zip(actual, expected_axes[label])):
            raise MechanismValidationError(
                "BICYCLE_GROUND_ROLL_AXIS_VARIANT_UNSUPPORTED",
                f"v1 requires {label}={expected_axes[label]!r}",
            )

    rear_radius = _positive(value.get("rear_wheel_radius_m"), "rear_wheel_radius_m")
    front_radius = _positive(value.get("front_wheel_radius_m"), "front_wheel_radius_m")
    if not 0.15 <= rear_radius <= 0.5 or not 0.15 <= front_radius <= 0.5:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_WHEEL_RADIUS_INVALID",
            "wheel radii must be within the reviewed bicycle range [0.15, 0.5] m",
        )
    friction = finite(value.get("ground_friction"), "ground_friction")
    if not 0.2 <= friction <= 1.5:
        raise MechanismValidationError(
            "BICYCLE_GROUND_ROLL_FRICTION_INVALID",
            "ground friction must be in [0.2, 1.5]",
        )

    return BicycleGroundRollSpec(
        mechanism_kind=BICYCLE_GROUND_ROLL_KIND,
        mechanism_id=mechanism_id,
        bodies=bodies,
        **roles,
        rear_axle_origin_abs_m=_point3(value.get("rear_axle_origin_abs_m"), "rear_axle_origin_abs_m"),
        front_axle_origin_abs_m=_point3(value.get("front_axle_origin_abs_m"), "front_axle_origin_abs_m"),
        steering_origin_abs_m=_point3(value.get("steering_origin_abs_m"), "steering_origin_abs_m"),
        wheel_axis_abs=wheel,
        steering_axis_abs=steering,
        travel_axis_abs=travel,
        ground_normal_abs=normal,
        ground_plane_height_m=finite(
            value.get("ground_plane_height_m"),
            "ground_plane_height_m",
        ),
        rear_wheel_radius_m=rear_radius,
        front_wheel_radius_m=front_radius,
        nominal_rear_drive_torque_nm=_frozen(
            value.get("nominal_rear_drive_torque_nm"),
            NOMINAL_REAR_DRIVE_TORQUE_NM,
            "nominal_rear_drive_torque_nm",
        ),
        ground_friction=friction,
        maximum_axle_drift_m=_frozen(value.get("maximum_axle_drift_m"), MAXIMUM_AXLE_DRIFT_M, "maximum_axle_drift_m"),
        maximum_guide_drift_m=_frozen(value.get("maximum_guide_drift_m"), MAXIMUM_GUIDE_DRIFT_M, "maximum_guide_drift_m"),
        maximum_off_axis_rotation_rad=_frozen(value.get("maximum_off_axis_rotation_rad"), MAXIMUM_OFF_AXIS_ROTATION_RAD, "maximum_off_axis_rotation_rad"),
        maximum_rolling_slip_ratio=_frozen(value.get("maximum_rolling_slip_ratio"), MAXIMUM_ROLLING_SLIP_RATIO, "maximum_rolling_slip_ratio"),
        maximum_ground_penetration_m=_frozen(value.get("maximum_ground_penetration_m"), MAXIMUM_GROUND_PENETRATION_M, "maximum_ground_penetration_m"),
        maximum_ground_separation_m=_frozen(value.get("maximum_ground_separation_m"), MAXIMUM_GROUND_SEPARATION_M, "maximum_ground_separation_m"),
        minimum_powered_contact_coverage=_frozen(value.get("minimum_powered_contact_coverage"), MINIMUM_POWERED_CONTACT_COVERAGE, "minimum_powered_contact_coverage"),
        minimum_front_rotation_rad=_frozen(value.get("minimum_front_rotation_rad"), MINIMUM_FRONT_ROTATION_RAD, "minimum_front_rotation_rad"),
        minimum_chassis_travel_m=_frozen(value.get("minimum_chassis_travel_m"), MINIMUM_CHASSIS_TRAVEL_M, "minimum_chassis_travel_m"),
    )
