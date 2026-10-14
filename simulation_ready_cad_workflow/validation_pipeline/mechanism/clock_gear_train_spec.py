from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .compound_spec import CompoundBodySpec, _body_from_dict
from .models import MechanismValidationError, finite, identifier, vector3


def _tuple3(value: Any, label: str, *, unit: bool = False) -> tuple[float, float, float]:
    return vector3(tuple(value), label, unit=unit, allow_zero=not unit)


@dataclass(frozen=True)
class ClockShaftSpec:
    shaft_id: str
    body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]


@dataclass(frozen=True)
class ClockGearMeshSpec:
    mesh_id: str
    driver_body_id: str
    driven_body_id: str
    driver_teeth: int
    driven_teeth: int
    mesh_stiffness_nm_rad: float
    mesh_damping_nm_s_rad: float
    phase_offset_rad: float = 0.0

    @property
    def ratio(self) -> float:
        return self.driver_teeth / self.driven_teeth


@dataclass(frozen=True)
class ClockShaftTransferSpec:
    transfer_id: str
    driver_body_id: str
    driven_body_id: str
    ratio: float
    stiffness_nm_rad: float
    damping_nm_s_rad: float
    max_torque_nm: float


@dataclass(frozen=True)
class ClockCamFollowerSpec:
    follower_id: str
    cam_body_id: str
    follower_body_id: str
    lobe_count: int
    center_angle_rad: float
    amplitude_rad: float
    stiffness_nm_rad: float
    damping_nm_s_rad: float


@dataclass(frozen=True)
class ClockCollisionExclusionSpec:
    body_a_id: str
    body_b_id: str
    reason: str


@dataclass(frozen=True)
class ClockSettingClutchSpec:
    clutch_id: str
    input_body_id: str
    driven_body_id: str
    ratio: float
    stiffness_nm_rad: float
    damping_nm_s_rad: float
    max_torque_nm: float


@dataclass(frozen=True)
class ClockScheduledAlarmSpec:
    setting_body_id: str
    target_time_seconds: float
    setting_target_angle_rad: float
    setting_angle_tolerance_rad: float
    target_second_hand_travel_rad: float
    target_second_hand_tolerance_rad: float
    target_minute_hand_travel_rad: float
    target_hour_hand_travel_rad: float
    minimum_post_trigger_time_travel_rad: float


@dataclass(frozen=True)
class ClockGearTrainSpec:
    train_id: str
    mechanism_kind: str
    bodies: tuple[CompoundBodySpec, ...]
    shafts: tuple[ClockShaftSpec, ...]
    gear_meshes: tuple[ClockGearMeshSpec, ...]
    shaft_transfers: tuple[ClockShaftTransferSpec, ...]
    cam_followers: tuple[ClockCamFollowerSpec, ...]
    collision_exclusions: tuple[ClockCollisionExclusionSpec, ...]
    setting_clutch: ClockSettingClutchSpec | None
    alarm_setting_clutch: ClockSettingClutchSpec | None
    fixed_body_id: str
    input_body_id: str
    setting_input_body_id: str
    output_body_id: str
    hour_body_id: str
    second_body_id: str
    alarm_body_id: str
    input_point_abs_m: tuple[float, float, float]
    input_axis_abs: tuple[float, float, float]
    setting_input_point_abs_m: tuple[float, float, float]
    setting_input_axis_abs: tuple[float, float, float]
    expected_hour_to_minute_ratio: float
    expected_second_to_minute_ratio: float
    hand_ratio_relative_tolerance: float
    home_output_angle_rad: float
    outbound_output_angle_rad: float
    nominal_input_torque_nm: float
    return_input_torque_nm: float
    max_torque_phase_s: float
    max_return_phase_s: float
    settle_phase_s: float
    output_angle_tolerance_rad: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    max_mesh_error_rad: float
    minimum_alarm_swing_rad: float
    friction: float = 0.06
    alarm_input_body_id: str | None = None
    alarm_input_point_abs_m: tuple[float, float, float] | None = None
    alarm_input_axis_abs: tuple[float, float, float] | None = None
    nominal_alarm_input_torque_nm: float | None = None
    scheduled_alarm: ClockScheduledAlarmSpec | None = None

    @property
    def alarm_isolated(self) -> bool:
        return (
            self.alarm_input_body_id is not None
            and self.alarm_input_body_id != self.input_body_id
        )

    @property
    def effective_alarm_input_body_id(self) -> str:
        return self.alarm_input_body_id or self.input_body_id

    @property
    def effective_alarm_input_point_abs_m(self) -> tuple[float, float, float]:
        return self.alarm_input_point_abs_m or self.input_point_abs_m

    @property
    def effective_alarm_input_axis_abs(self) -> tuple[float, float, float]:
        return self.alarm_input_axis_abs or self.input_axis_abs

    @property
    def effective_nominal_alarm_input_torque_nm(self) -> float:
        return self.nominal_alarm_input_torque_nm or self.nominal_input_torque_nm

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bodies"] = [asdict(item) for item in self.bodies]
        payload["shafts"] = [asdict(item) for item in self.shafts]
        payload["gear_meshes"] = [asdict(item) for item in self.gear_meshes]
        payload["shaft_transfers"] = [asdict(item) for item in self.shaft_transfers]
        payload["cam_followers"] = [asdict(item) for item in self.cam_followers]
        payload["collision_exclusions"] = [
            asdict(item) for item in self.collision_exclusions
        ]
        payload["setting_clutch"] = (
            None if self.setting_clutch is None else asdict(self.setting_clutch)
        )
        if self.alarm_setting_clutch is None:
            payload.pop("alarm_setting_clutch", None)
        else:
            payload["alarm_setting_clutch"] = asdict(
                self.alarm_setting_clutch
            )
        if self.alarm_input_body_id is None:
            # Preserve the exact serialized legacy spec and therefore its hash.
            # Old clock packages continue to use the time barrel for the alarm.
            for key in (
                "alarm_input_body_id",
                "alarm_input_point_abs_m",
                "alarm_input_axis_abs",
                "nominal_alarm_input_torque_nm",
            ):
                payload.pop(key, None)
        if self.scheduled_alarm is None:
            payload.pop("scheduled_alarm", None)
        else:
            payload["scheduled_alarm"] = asdict(self.scheduled_alarm)
        return payload


def _shaft_from_dict(value: Mapping[str, Any], index: int) -> ClockShaftSpec:
    shaft_id = str(value.get("shaft_id") or "")
    body_id = str(value.get("body_id") or "")
    identifier(shaft_id, f"shafts[{index}].shaft_id")
    identifier(body_id, f"shafts[{index}].body_id")
    return ClockShaftSpec(
        shaft_id=shaft_id,
        body_id=body_id,
        origin_abs_m=_tuple3(value.get("origin_abs_m"), f"shafts[{index}].origin_abs_m"),
        axis_abs=_tuple3(value.get("axis_abs"), f"shafts[{index}].axis_abs", unit=True),
    )


def _mesh_from_dict(value: Mapping[str, Any], index: int) -> ClockGearMeshSpec:
    mesh_id = str(value.get("mesh_id") or "")
    driver = str(value.get("driver_body_id") or "")
    driven = str(value.get("driven_body_id") or "")
    identifier(mesh_id, f"gear_meshes[{index}].mesh_id")
    identifier(driver, f"gear_meshes[{index}].driver_body_id")
    identifier(driven, f"gear_meshes[{index}].driven_body_id")
    if driver == driven:
        raise MechanismValidationError(
            "CLOCK_GEAR_SELF_MESH", f"gear mesh {mesh_id!r} connects one body to itself"
        )
    driver_teeth = int(value.get("driver_teeth") or 0)
    driven_teeth = int(value.get("driven_teeth") or 0)
    if driver_teeth < 6 or driven_teeth < 6:
        raise MechanismValidationError(
            "CLOCK_GEAR_TEETH_INVALID", f"gear mesh {mesh_id!r} needs at least six teeth per gear"
        )
    stiffness = finite(value.get("mesh_stiffness_nm_rad"), f"{mesh_id}.mesh_stiffness_nm_rad")
    damping = finite(value.get("mesh_damping_nm_s_rad"), f"{mesh_id}.mesh_damping_nm_s_rad")
    if stiffness <= 0 or damping < 0:
        raise MechanismValidationError(
            "CLOCK_GEAR_COMPLIANCE_INVALID", f"gear mesh {mesh_id!r} has invalid compliance"
        )
    return ClockGearMeshSpec(
        mesh_id=mesh_id,
        driver_body_id=driver,
        driven_body_id=driven,
        driver_teeth=driver_teeth,
        driven_teeth=driven_teeth,
        mesh_stiffness_nm_rad=stiffness,
        mesh_damping_nm_s_rad=damping,
        phase_offset_rad=finite(value.get("phase_offset_rad", 0.0), f"{mesh_id}.phase_offset_rad"),
    )


def _shaft_transfer_from_dict(
    value: Mapping[str, Any], index: int
) -> ClockShaftTransferSpec:
    transfer_id = str(value.get("transfer_id") or "")
    driver = str(value.get("driver_body_id") or "")
    driven = str(value.get("driven_body_id") or "")
    identifier(transfer_id, f"shaft_transfers[{index}].transfer_id")
    identifier(driver, f"shaft_transfers[{index}].driver_body_id")
    identifier(driven, f"shaft_transfers[{index}].driven_body_id")
    ratio = finite(value.get("ratio"), f"{transfer_id}.ratio")
    stiffness = finite(value.get("stiffness_nm_rad"), f"{transfer_id}.stiffness_nm_rad")
    damping = finite(value.get("damping_nm_s_rad"), f"{transfer_id}.damping_nm_s_rad")
    max_torque = finite(value.get("max_torque_nm"), f"{transfer_id}.max_torque_nm")
    if driver == driven or abs(ratio) <= 1.0e-9 or stiffness <= 0 or damping < 0 or max_torque <= 0:
        raise MechanismValidationError(
            "CLOCK_SHAFT_TRANSFER_INVALID",
            f"shaft transfer {transfer_id!r} is invalid",
        )
    return ClockShaftTransferSpec(
        transfer_id=transfer_id,
        driver_body_id=driver,
        driven_body_id=driven,
        ratio=ratio,
        stiffness_nm_rad=stiffness,
        damping_nm_s_rad=damping,
        max_torque_nm=max_torque,
    )


def _follower_from_dict(value: Mapping[str, Any], index: int) -> ClockCamFollowerSpec:
    follower_id = str(value.get("follower_id") or "")
    cam = str(value.get("cam_body_id") or "")
    follower = str(value.get("follower_body_id") or "")
    identifier(follower_id, f"cam_followers[{index}].follower_id")
    identifier(cam, f"cam_followers[{index}].cam_body_id")
    identifier(follower, f"cam_followers[{index}].follower_body_id")
    lobes = int(value.get("lobe_count") or 0)
    amplitude = finite(value.get("amplitude_rad"), f"{follower_id}.amplitude_rad")
    stiffness = finite(value.get("stiffness_nm_rad"), f"{follower_id}.stiffness_nm_rad")
    damping = finite(value.get("damping_nm_s_rad"), f"{follower_id}.damping_nm_s_rad")
    if cam == follower or lobes < 1 or amplitude <= 0 or stiffness <= 0 or damping < 0:
        raise MechanismValidationError(
            "CLOCK_CAM_FOLLOWER_INVALID", f"cam follower {follower_id!r} is invalid"
        )
    return ClockCamFollowerSpec(
        follower_id=follower_id,
        cam_body_id=cam,
        follower_body_id=follower,
        lobe_count=lobes,
        center_angle_rad=finite(value.get("center_angle_rad", 0.0), f"{follower_id}.center_angle_rad"),
        amplitude_rad=amplitude,
        stiffness_nm_rad=stiffness,
        damping_nm_s_rad=damping,
    )


def _collision_exclusion_from_dict(
    value: Mapping[str, Any], index: int
) -> ClockCollisionExclusionSpec:
    body_a = str(value.get("body_a_id") or "")
    body_b = str(value.get("body_b_id") or "")
    reason = str(value.get("reason") or "").strip()
    identifier(body_a, f"collision_exclusions[{index}].body_a_id")
    identifier(body_b, f"collision_exclusions[{index}].body_b_id")
    if body_a == body_b or len(reason) < 8:
        raise MechanismValidationError(
            "CLOCK_COLLISION_EXCLUSION_INVALID",
            "clock collision exclusions need two bodies and a specific reason",
        )
    return ClockCollisionExclusionSpec(body_a, body_b, reason)


def _setting_clutch_from_dict(
    value: Mapping[str, Any] | None,
    *,
    field_name: str = "setting_clutch",
) -> ClockSettingClutchSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "CLOCK_SETTING_CLUTCH_INVALID",
            f"{field_name} must be an object",
        )
    clutch_id = str(value.get("clutch_id") or "")
    input_body = str(value.get("input_body_id") or "")
    driven_body = str(value.get("driven_body_id") or "")
    identifier(clutch_id, f"{field_name}.clutch_id")
    identifier(input_body, f"{field_name}.input_body_id")
    identifier(driven_body, f"{field_name}.driven_body_id")
    ratio = finite(value.get("ratio"), f"{field_name}.ratio")
    stiffness = finite(
        value.get("stiffness_nm_rad"),
        f"{field_name}.stiffness_nm_rad",
    )
    damping = finite(
        value.get("damping_nm_s_rad"),
        f"{field_name}.damping_nm_s_rad",
    )
    max_torque = finite(
        value.get("max_torque_nm"),
        f"{field_name}.max_torque_nm",
    )
    if input_body == driven_body or abs(ratio) <= 1.0e-9 or stiffness <= 0 or damping < 0 or max_torque <= 0:
        raise MechanismValidationError(
            "CLOCK_SETTING_CLUTCH_INVALID",
            "setting clutch needs two bodies, a nonzero ratio, finite compliance, and a torque limit",
        )
    return ClockSettingClutchSpec(
        clutch_id=clutch_id,
        input_body_id=input_body,
        driven_body_id=driven_body,
        ratio=ratio,
        stiffness_nm_rad=stiffness,
        damping_nm_s_rad=damping,
        max_torque_nm=max_torque,
    )


def _scheduled_alarm_from_dict(
    value: Mapping[str, Any] | None,
) -> ClockScheduledAlarmSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "CLOCK_SCHEDULED_ALARM_INVALID",
            "scheduled_alarm must be an object",
        )
    setting_body = str(value.get("setting_body_id") or "")
    identifier(setting_body, "scheduled_alarm.setting_body_id")
    fields = {
        "target_time_seconds": finite(
            value.get("target_time_seconds"),
            "scheduled_alarm.target_time_seconds",
        ),
        "setting_target_angle_rad": finite(
            value.get("setting_target_angle_rad"),
            "scheduled_alarm.setting_target_angle_rad",
        ),
        "setting_angle_tolerance_rad": finite(
            value.get("setting_angle_tolerance_rad"),
            "scheduled_alarm.setting_angle_tolerance_rad",
        ),
        "target_second_hand_travel_rad": finite(
            value.get("target_second_hand_travel_rad"),
            "scheduled_alarm.target_second_hand_travel_rad",
        ),
        "target_second_hand_tolerance_rad": finite(
            value.get("target_second_hand_tolerance_rad"),
            "scheduled_alarm.target_second_hand_tolerance_rad",
        ),
        "target_minute_hand_travel_rad": finite(
            value.get("target_minute_hand_travel_rad"),
            "scheduled_alarm.target_minute_hand_travel_rad",
        ),
        "target_hour_hand_travel_rad": finite(
            value.get("target_hour_hand_travel_rad"),
            "scheduled_alarm.target_hour_hand_travel_rad",
        ),
        "minimum_post_trigger_time_travel_rad": finite(
            value.get("minimum_post_trigger_time_travel_rad"),
            "scheduled_alarm.minimum_post_trigger_time_travel_rad",
        ),
    }
    if any(item <= 0 for item in fields.values()):
        raise MechanismValidationError(
            "CLOCK_SCHEDULED_ALARM_INVALID",
            "scheduled alarm targets and tolerances must be positive",
        )
    return ClockScheduledAlarmSpec(setting_body_id=setting_body, **fields)


def clock_gear_train_from_dict(value: Mapping[str, Any] | None) -> ClockGearTrainSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "CLOCK_GEAR_SPEC_REQUIRED", "clock gear train requires a family_spec object"
        )
    if value.get("mechanism_kind") != "spur_gear_clock":
        raise MechanismValidationError(
            "CLOCK_GEAR_KIND_INVALID", "clock gear train must declare mechanism_kind='spur_gear_clock'"
        )
    train_id = str(value.get("train_id") or "")
    identifier(train_id, "family_spec.train_id")
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(value.get("bodies", ()))
        if isinstance(item, Mapping)
    )
    shafts = tuple(
        _shaft_from_dict(item, index)
        for index, item in enumerate(value.get("shafts", ()))
        if isinstance(item, Mapping)
    )
    meshes = tuple(
        _mesh_from_dict(item, index)
        for index, item in enumerate(value.get("gear_meshes", ()))
        if isinstance(item, Mapping)
    )
    shaft_transfers = tuple(
        _shaft_transfer_from_dict(item, index)
        for index, item in enumerate(value.get("shaft_transfers", ()))
        if isinstance(item, Mapping)
    )
    followers = tuple(
        _follower_from_dict(item, index)
        for index, item in enumerate(value.get("cam_followers", ()))
        if isinstance(item, Mapping)
    )
    collision_exclusions = tuple(
        _collision_exclusion_from_dict(item, index)
        for index, item in enumerate(value.get("collision_exclusions", ()))
        if isinstance(item, Mapping)
    )
    setting_clutch = _setting_clutch_from_dict(value.get("setting_clutch"))
    alarm_setting_clutch = _setting_clutch_from_dict(
        value.get("alarm_setting_clutch"),
        field_name="alarm_setting_clutch",
    )
    scheduled_alarm = _scheduled_alarm_from_dict(value.get("scheduled_alarm"))
    if len(bodies) < 7 or len(shafts) < 6 or len(meshes) < 4:
        raise MechanismValidationError(
            "CLOCK_GEAR_TOPOLOGY_TOO_SMALL",
            "clock challenge requires at least one fixed body, six shafts, and four gear meshes",
        )
    body_ids = [item.body_id for item in bodies]
    if len(body_ids) != len(set(body_ids)):
        raise MechanismValidationError("CLOCK_BODY_ID_DUPLICATE", "clock body IDs must be unique")
    fixed = [item.body_id for item in bodies if item.fixed]
    if len(fixed) != 1:
        raise MechanismValidationError("CLOCK_FIXED_BODY_COUNT", "clock gear train requires exactly one fixed body")
    moving_ids = {item.body_id for item in bodies if not item.fixed}
    groups = [group for body in bodies for group in body.group_ids]
    if len(groups) != len(set(groups)):
        raise MechanismValidationError("CLOCK_GROUP_OVERLAP", "a rigid group cannot belong to multiple clock bodies")
    shaft_ids = [item.shaft_id for item in shafts]
    shaft_body_ids = [item.body_id for item in shafts]
    if len(shaft_ids) != len(set(shaft_ids)) or len(shaft_body_ids) != len(set(shaft_body_ids)):
        raise MechanismValidationError("CLOCK_SHAFT_DUPLICATE", "each clock body must have one unique shaft")
    if set(shaft_body_ids) != moving_ids:
        raise MechanismValidationError("CLOCK_SHAFT_COVERAGE", "every moving clock body must have exactly one shaft")
    mesh_ids = [item.mesh_id for item in meshes]
    if len(mesh_ids) != len(set(mesh_ids)):
        raise MechanismValidationError("CLOCK_MESH_ID_DUPLICATE", "clock gear mesh IDs must be unique")
    for mesh in meshes:
        if mesh.driver_body_id not in moving_ids or mesh.driven_body_id not in moving_ids:
            raise MechanismValidationError("CLOCK_MESH_BODY_UNKNOWN", f"gear mesh {mesh.mesh_id!r} references an unknown moving body")
    transfer_ids = [item.transfer_id for item in shaft_transfers]
    if len(transfer_ids) != len(set(transfer_ids)):
        raise MechanismValidationError(
            "CLOCK_SHAFT_TRANSFER_ID_DUPLICATE",
            "clock shaft-transfer IDs must be unique",
        )
    for transfer in shaft_transfers:
        if transfer.driver_body_id not in moving_ids or transfer.driven_body_id not in moving_ids:
            raise MechanismValidationError(
                "CLOCK_SHAFT_TRANSFER_BODY_UNKNOWN",
                f"shaft transfer {transfer.transfer_id!r} references an unknown moving body",
            )
    for follower in followers:
        if follower.cam_body_id not in moving_ids or follower.follower_body_id not in moving_ids:
            raise MechanismValidationError("CLOCK_FOLLOWER_BODY_UNKNOWN", f"cam follower {follower.follower_id!r} references an unknown moving body")
    if setting_clutch is not None and {
        setting_clutch.input_body_id,
        setting_clutch.driven_body_id,
    } - moving_ids:
        raise MechanismValidationError(
            "CLOCK_SETTING_CLUTCH_BODY_UNKNOWN",
            "setting clutch references an unknown moving body",
        )
    if alarm_setting_clutch is not None and {
        alarm_setting_clutch.input_body_id,
        alarm_setting_clutch.driven_body_id,
    } - moving_ids:
        raise MechanismValidationError(
            "CLOCK_ALARM_SETTING_CLUTCH_BODY_UNKNOWN",
            "alarm setting clutch references an unknown moving body",
        )
    if scheduled_alarm is not None and scheduled_alarm.setting_body_id not in moving_ids:
        raise MechanismValidationError(
            "CLOCK_SCHEDULED_ALARM_BODY_UNKNOWN",
            "scheduled alarm setting body is not a declared moving shaft",
        )
    exclusion_pairs: set[tuple[str, str]] = set()
    for exclusion in collision_exclusions:
        pair = tuple(sorted((exclusion.body_a_id, exclusion.body_b_id)))
        if pair in exclusion_pairs or set(pair) - set(body_ids):
            raise MechanismValidationError(
                "CLOCK_COLLISION_EXCLUSION_INVALID",
                "clock collision exclusions must be unique and reference known bodies",
            )
        exclusion_pairs.add(pair)
    fixed_body = str(value.get("fixed_body_id") or fixed[0])
    input_body = str(value.get("input_body_id") or "")
    output_body = str(value.get("output_body_id") or "")
    setting_input_body = str(
        value.get("setting_input_body_id")
        or (setting_clutch.input_body_id if setting_clutch is not None else input_body)
    )
    hour_body = str(value.get("hour_body_id") or output_body)
    second_body = str(value.get("second_body_id") or output_body)
    alarm_body = str(value.get("alarm_body_id") or "")
    alarm_input_declared = value.get("alarm_input_body_id") is not None
    alarm_input_body = str(value.get("alarm_input_body_id") or input_body)
    if fixed_body != fixed[0] or {
        input_body,
        setting_input_body,
        output_body,
        hour_body,
        second_body,
        alarm_body,
        alarm_input_body,
    } - moving_ids:
        raise MechanismValidationError(
            "CLOCK_IO_BODY_UNKNOWN",
            "clock fixed, input, setting input, hand, or alarm body is unknown",
        )
    adjacency = {body_id: set() for body_id in moving_ids}
    for mesh in meshes:
        adjacency[mesh.driver_body_id].add(mesh.driven_body_id)
        adjacency[mesh.driven_body_id].add(mesh.driver_body_id)
    for follower in followers:
        adjacency[follower.cam_body_id].add(follower.follower_body_id)
        adjacency[follower.follower_body_id].add(follower.cam_body_id)
    for transfer in shaft_transfers:
        adjacency[transfer.driver_body_id].add(transfer.driven_body_id)
        adjacency[transfer.driven_body_id].add(transfer.driver_body_id)
    if setting_clutch is not None:
        adjacency[setting_clutch.input_body_id].add(setting_clutch.driven_body_id)
        adjacency[setting_clutch.driven_body_id].add(setting_clutch.input_body_id)
    if alarm_setting_clutch is not None:
        adjacency[alarm_setting_clutch.input_body_id].add(
            alarm_setting_clutch.driven_body_id
        )
        adjacency[alarm_setting_clutch.driven_body_id].add(
            alarm_setting_clutch.input_body_id
        )
    def connected_from(start: str) -> set[str]:
        visited: set[str] = set()
        pending = [start]
        while pending:
            current = pending.pop()
            if current in visited:
                continue
            visited.add(current)
            pending.extend(adjacency[current] - visited)
        return visited

    visited = connected_from(input_body)
    if {output_body, hour_body, second_body, setting_input_body} - visited:
        raise MechanismValidationError(
            "CLOCK_TRAIN_DISCONNECTED",
            "clock timekeeping train must connect the barrel, three-hand display, and setting shaft",
        )
    alarm_isolated = alarm_input_declared and alarm_input_body != input_body
    if scheduled_alarm is not None and not alarm_isolated:
        raise MechanismValidationError(
            "CLOCK_SCHEDULED_ALARM_REQUIRES_ISOLATION",
            "scheduled ringing requires an independent alarm barrel",
        )
    if alarm_isolated:
        alarm_visited = connected_from(alarm_input_body)
        if alarm_body not in alarm_visited:
            raise MechanismValidationError(
                "CLOCK_ALARM_TRAIN_DISCONNECTED",
                "independent alarm input must connect to the alarm follower",
            )
        if alarm_body in visited or {
            output_body,
            hour_body,
            second_body,
        } & alarm_visited:
            raise MechanismValidationError(
                "CLOCK_ALARM_TRAIN_NOT_ISOLATED",
                "independent alarm train must not connect to any clock-hand shaft",
            )
    elif alarm_body not in visited:
        raise MechanismValidationError(
            "CLOCK_TRAIN_DISCONNECTED",
            "legacy clock train must connect the barrel to the alarm follower",
        )
    if setting_clutch is not None and setting_input_body != setting_clutch.input_body_id:
        raise MechanismValidationError(
            "CLOCK_SETTING_INPUT_MISMATCH",
            "setting_input_body_id must match the declared setting clutch input body",
        )
    if alarm_setting_clutch is not None:
        if scheduled_alarm is None:
            raise MechanismValidationError(
                "CLOCK_ALARM_SETTING_CLUTCH_UNUSED",
                "alarm setting clutch requires a scheduled_alarm declaration",
            )
        if (
            alarm_setting_clutch.input_body_id != hour_body
            or alarm_setting_clutch.driven_body_id
            != scheduled_alarm.setting_body_id
        ):
            raise MechanismValidationError(
                "CLOCK_ALARM_SETTING_CLUTCH_MISMATCH",
                "alarm setting clutch must transmit the hour hand to the "
                "scheduled-alarm target body",
            )
    home = finite(value.get("home_output_angle_rad"), "home_output_angle_rad")
    outbound = finite(value.get("outbound_output_angle_rad"), "outbound_output_angle_rad")
    nominal = finite(value.get("nominal_input_torque_nm"), "nominal_input_torque_nm")
    reverse = finite(value.get("return_input_torque_nm", nominal), "return_input_torque_nm")
    nominal_alarm = finite(
        value.get("nominal_alarm_input_torque_nm", nominal),
        "nominal_alarm_input_torque_nm",
    )
    phase = finite(value.get("max_torque_phase_s"), "max_torque_phase_s")
    return_phase = finite(value.get("max_return_phase_s", phase), "max_return_phase_s")
    settle = finite(value.get("settle_phase_s", 0.05), "settle_phase_s")
    if outbound <= home or nominal <= 0 or nominal_alarm <= 0 or reverse <= 0 or phase <= 0 or return_phase <= 0 or settle < 0:
        raise MechanismValidationError("CLOCK_TASK_RANGE_INVALID", "clock task range, torque, or duration is invalid")
    thresholds = {
        "output_angle_tolerance_rad": finite(value.get("output_angle_tolerance_rad"), "output_angle_tolerance_rad"),
        "max_support_drift_m": finite(value.get("max_support_drift_m"), "max_support_drift_m"),
        "max_off_axis_rotation_rad": finite(value.get("max_off_axis_rotation_rad"), "max_off_axis_rotation_rad"),
        "max_mesh_error_rad": finite(value.get("max_mesh_error_rad"), "max_mesh_error_rad"),
        "minimum_alarm_swing_rad": finite(value.get("minimum_alarm_swing_rad"), "minimum_alarm_swing_rad"),
    }
    if any(item <= 0 for item in thresholds.values()):
        raise MechanismValidationError("CLOCK_THRESHOLD_INVALID", "clock comparison thresholds must be positive")
    hour_ratio = finite(
        value.get("expected_hour_to_minute_ratio", 1.0),
        "expected_hour_to_minute_ratio",
    )
    second_ratio = finite(
        value.get("expected_second_to_minute_ratio", 1.0),
        "expected_second_to_minute_ratio",
    )
    ratio_tolerance = finite(
        value.get("hand_ratio_relative_tolerance", 0.05),
        "hand_ratio_relative_tolerance",
    )
    if hour_ratio <= 0 or second_ratio <= 0 or not 0 < ratio_tolerance < 0.5:
        raise MechanismValidationError(
            "CLOCK_HAND_RATIO_INVALID",
            "three-hand ratios must be positive and use a relative tolerance below 0.5",
        )
    if scheduled_alarm is not None:
        expected_minute = (
            scheduled_alarm.target_second_hand_travel_rad / second_ratio
        )
        expected_hour = expected_minute * hour_ratio
        if (
            abs(
                scheduled_alarm.target_minute_hand_travel_rad / expected_minute
                - 1.0
            )
            > ratio_tolerance
            or abs(
                scheduled_alarm.target_hour_hand_travel_rad / expected_hour
                - 1.0
            )
            > ratio_tolerance
        ):
            raise MechanismValidationError(
                "CLOCK_SCHEDULED_ALARM_RATIO_INVALID",
                "scheduled alarm target must agree with the declared hand ratios",
            )
    return ClockGearTrainSpec(
        train_id=train_id,
        mechanism_kind="spur_gear_clock",
        bodies=bodies,
        shafts=shafts,
        gear_meshes=meshes,
        shaft_transfers=shaft_transfers,
        cam_followers=followers,
        collision_exclusions=collision_exclusions,
        setting_clutch=setting_clutch,
        alarm_setting_clutch=alarm_setting_clutch,
        fixed_body_id=fixed_body,
        input_body_id=input_body,
        setting_input_body_id=setting_input_body,
        output_body_id=output_body,
        hour_body_id=hour_body,
        second_body_id=second_body,
        alarm_body_id=alarm_body,
        input_point_abs_m=_tuple3(value.get("input_point_abs_m"), "input_point_abs_m"),
        input_axis_abs=_tuple3(value.get("input_axis_abs"), "input_axis_abs", unit=True),
        setting_input_point_abs_m=_tuple3(
            value.get("setting_input_point_abs_m", value.get("input_point_abs_m")),
            "setting_input_point_abs_m",
        ),
        setting_input_axis_abs=_tuple3(
            value.get("setting_input_axis_abs", value.get("input_axis_abs")),
            "setting_input_axis_abs",
            unit=True,
        ),
        alarm_input_body_id=alarm_input_body if alarm_input_declared else None,
        alarm_input_point_abs_m=(
            _tuple3(
                value.get("alarm_input_point_abs_m", value.get("input_point_abs_m")),
                "alarm_input_point_abs_m",
            )
            if alarm_input_declared
            else None
        ),
        alarm_input_axis_abs=(
            _tuple3(
                value.get("alarm_input_axis_abs", value.get("input_axis_abs")),
                "alarm_input_axis_abs",
                unit=True,
            )
            if alarm_input_declared
            else None
        ),
        nominal_alarm_input_torque_nm=nominal_alarm if alarm_input_declared else None,
        scheduled_alarm=scheduled_alarm,
        expected_hour_to_minute_ratio=hour_ratio,
        expected_second_to_minute_ratio=second_ratio,
        hand_ratio_relative_tolerance=ratio_tolerance,
        home_output_angle_rad=home,
        outbound_output_angle_rad=outbound,
        nominal_input_torque_nm=nominal,
        return_input_torque_nm=reverse,
        max_torque_phase_s=phase,
        max_return_phase_s=return_phase,
        settle_phase_s=settle,
        friction=finite(value.get("friction", 0.06), "friction"),
        **thresholds,
    )
