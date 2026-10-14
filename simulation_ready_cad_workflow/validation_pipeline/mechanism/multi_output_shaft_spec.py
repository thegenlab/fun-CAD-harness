from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .compound_spec import CompoundBodySpec, _body_from_dict
from .models import MechanismValidationError, finite, identifier, vector3


SUPPORTED_KINDS = {
    "crown_bevel_dual_output",
    "multi_output_shaft_coupling",
}


def _tuple3(
    value: Any,
    label: str,
    *,
    unit: bool = False,
) -> tuple[float, float, float]:
    return vector3(tuple(value), label, unit=unit, allow_zero=not unit)


@dataclass(frozen=True)
class MultiOutputShaft:
    shaft_id: str
    body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]


@dataclass(frozen=True)
class MultiOutputShaftCoupling:
    mesh_id: str
    driver_body_id: str
    driven_body_id: str
    driver_teeth: int
    driven_teeth: int
    declared_ratio: float
    shaft_angle_rad: float
    driver_pitch_radius_m: float
    driven_pitch_radius_m: float
    driver_face_width_m: float
    driven_face_width_m: float
    geometry_contact_tolerance_m: float
    axis_perpendicular_tolerance_rad: float
    mesh_stiffness_nm_rad: float
    mesh_damping_nm_s_rad: float
    phase_offset_rad: float = 0.0

    @property
    def ratio(self) -> float:
        """Chrono/clock-runtime convention: driven + ratio * driver = 0."""

        return -self.declared_ratio


@dataclass(frozen=True)
class MultiOutputShaftSpec:
    mechanism_id: str
    mechanism_kind: str
    bodies: tuple[CompoundBodySpec, ...]
    shafts: tuple[MultiOutputShaft, ...]
    gear_meshes: tuple[MultiOutputShaftCoupling, ...]
    fixed_body_id: str
    input_body_id: str
    output_body_ids: tuple[str, ...]
    nominal_input_torque_nm: float
    return_input_torque_nm: float
    max_torque_phase_s: float
    max_return_phase_s: float
    settle_phase_s: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    max_mesh_error_rad: float
    minimum_output_travel_rad: float = 1.0
    output_pair_ratio_tolerance: float = 0.08
    input_output_ratio_tolerance: float = 0.20
    friction: float = 0.06

    @property
    def train_id(self) -> str:
        return self.mechanism_id

    @property
    def output_body_id(self) -> str:
        return self.output_body_ids[0]

    @property
    def hour_body_id(self) -> str:
        return self.output_body_ids[0]

    @property
    def second_body_id(self) -> str:
        return self.output_body_ids[1]

    @property
    def alarm_body_id(self) -> str:
        return self.output_body_ids[-1]

    @property
    def setting_input_body_id(self) -> str:
        return self.input_body_id

    @property
    def input_point_abs_m(self) -> tuple[float, float, float]:
        return self.shaft_by_body[self.input_body_id].origin_abs_m

    @property
    def input_axis_abs(self) -> tuple[float, float, float]:
        return self.shaft_by_body[self.input_body_id].axis_abs

    @property
    def setting_input_point_abs_m(self) -> tuple[float, float, float]:
        return self.input_point_abs_m

    @property
    def setting_input_axis_abs(self) -> tuple[float, float, float]:
        return self.input_axis_abs

    @property
    def output_angle_tolerance_rad(self) -> float:
        return self.max_mesh_error_rad

    @property
    def home_output_angle_rad(self) -> float:
        return 0.0

    @property
    def outbound_output_angle_rad(self) -> float:
        return self.minimum_output_travel_rad

    @property
    def expected_hour_to_minute_ratio(self) -> float:
        return 1.0

    @property
    def expected_second_to_minute_ratio(self) -> float:
        left = self.declared_ratio_for_body(self.output_body_ids[0])
        right = self.declared_ratio_for_body(self.output_body_ids[1])
        return right / left

    @property
    def hand_ratio_relative_tolerance(self) -> float:
        return self.output_pair_ratio_tolerance

    @property
    def minimum_alarm_swing_rad(self) -> float:
        return 1.0

    @property
    def shaft_transfers(self) -> tuple[Any, ...]:
        return ()

    @property
    def cam_followers(self) -> tuple[Any, ...]:
        return ()

    @property
    def collision_exclusions(self) -> tuple[Any, ...]:
        return ()

    @property
    def setting_clutch(self) -> None:
        return None

    @property
    def alarm_setting_clutch(self) -> None:
        return None

    @property
    def scheduled_alarm(self) -> None:
        return None

    @property
    def alarm_input_body_id(self) -> None:
        return None

    @property
    def alarm_isolated(self) -> bool:
        return False

    @property
    def effective_alarm_input_body_id(self) -> str:
        return self.input_body_id

    @property
    def effective_alarm_input_point_abs_m(self) -> tuple[float, float, float]:
        return self.input_point_abs_m

    @property
    def effective_alarm_input_axis_abs(self) -> tuple[float, float, float]:
        return self.input_axis_abs

    @property
    def effective_nominal_alarm_input_torque_nm(self) -> float:
        return self.nominal_input_torque_nm

    @property
    def shaft_by_body(self) -> dict[str, MultiOutputShaft]:
        return {item.body_id: item for item in self.shafts}

    def declared_ratio_for_body(self, body_id: str) -> float:
        ratios = {self.input_body_id: 1.0}
        pending = [self.input_body_id]
        while pending:
            current = pending.pop(0)
            for mesh in self.gear_meshes:
                if (
                    mesh.driver_body_id == current
                    and mesh.driven_body_id not in ratios
                ):
                    ratios[mesh.driven_body_id] = (
                        mesh.declared_ratio * ratios[current]
                    )
                    pending.append(mesh.driven_body_id)
                elif (
                    mesh.driven_body_id == current
                    and mesh.driver_body_id not in ratios
                ):
                    ratios[mesh.driver_body_id] = (
                        ratios[current] / mesh.declared_ratio
                    )
                    pending.append(mesh.driver_body_id)
        if body_id not in ratios:
            raise MechanismValidationError(
                "MULTI_OUTPUT_SHAFT_PATH_MISSING",
                f"input shaft has no declared coupling path to {body_id!r}",
            )
        return ratios[body_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mechanism_kind": self.mechanism_kind,
            "mechanism_id": self.mechanism_id,
            "bodies": [asdict(item) for item in self.bodies],
            "shafts": [asdict(item) for item in self.shafts],
            "crown_bevel_meshes": [
                {
                    **asdict(item),
                    "ratio": item.declared_ratio,
                }
                for item in self.gear_meshes
            ],
            "fixed_body_id": self.fixed_body_id,
            "input_body_id": self.input_body_id,
            "output_body_ids": list(self.output_body_ids),
            "nominal_input_torque_nm": self.nominal_input_torque_nm,
            "return_input_torque_nm": self.return_input_torque_nm,
            "max_torque_phase_s": self.max_torque_phase_s,
            "max_return_phase_s": self.max_return_phase_s,
            "settle_phase_s": self.settle_phase_s,
            "max_support_drift_m": self.max_support_drift_m,
            "max_off_axis_rotation_rad": self.max_off_axis_rotation_rad,
            "max_mesh_error_rad": self.max_mesh_error_rad,
            "minimum_output_travel_rad": (
                self.minimum_output_travel_rad
            ),
            "output_pair_ratio_tolerance": (
                self.output_pair_ratio_tolerance
            ),
            "input_output_ratio_tolerance": (
                self.input_output_ratio_tolerance
            ),
            "friction": self.friction,
        }


def _shaft_from_dict(
    value: Mapping[str, Any],
    index: int,
) -> MultiOutputShaft:
    shaft_id = str(value.get("shaft_id") or "")
    body_id = str(value.get("body_id") or "")
    identifier(shaft_id, f"shafts[{index}].shaft_id")
    identifier(body_id, f"shafts[{index}].body_id")
    return MultiOutputShaft(
        shaft_id=shaft_id,
        body_id=body_id,
        origin_abs_m=_tuple3(
            value.get("origin_abs_m"),
            f"shafts[{index}].origin_abs_m",
        ),
        axis_abs=_tuple3(
            value.get("axis_abs"),
            f"shafts[{index}].axis_abs",
            unit=True,
        ),
    )


def _coupling_from_dict(
    value: Mapping[str, Any],
    index: int,
) -> MultiOutputShaftCoupling:
    mesh_id = str(value.get("mesh_id") or "")
    driver = str(value.get("driver_body_id") or "")
    driven = str(value.get("driven_body_id") or "")
    identifier(mesh_id, f"crown_bevel_meshes[{index}].mesh_id")
    identifier(driver, f"{mesh_id}.driver_body_id")
    identifier(driven, f"{mesh_id}.driven_body_id")
    if driver == driven:
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_SELF_COUPLING",
            f"shaft coupling {mesh_id!r} connects one body to itself",
        )
    driver_teeth = int(value.get("driver_teeth") or 0)
    driven_teeth = int(value.get("driven_teeth") or 0)
    ratio = finite(value.get("ratio"), f"{mesh_id}.ratio")
    if (
        driver_teeth < 6
        or driven_teeth < 6
        or abs(ratio) <= 1.0e-9
        or not math.isclose(
            abs(ratio),
            driver_teeth / driven_teeth,
            rel_tol=0.02,
        )
    ):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_RATIO_INVALID",
            f"shaft coupling {mesh_id!r} has inconsistent teeth or ratio",
        )
    shaft_angle = finite(
        value.get("shaft_angle_rad"),
        f"{mesh_id}.shaft_angle_rad",
    )
    perpendicular_tolerance = finite(
        value.get("axis_perpendicular_tolerance_rad"),
        f"{mesh_id}.axis_perpendicular_tolerance_rad",
    )
    if (
        abs(shaft_angle - 0.5 * math.pi) > perpendicular_tolerance
        or perpendicular_tolerance <= 0.0
    ):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_ANGLE_INVALID",
            f"shaft coupling {mesh_id!r} is not a declared right-angle drive",
        )
    positive = {
        name: finite(value.get(name), f"{mesh_id}.{name}")
        for name in (
            "driver_pitch_radius_m",
            "driven_pitch_radius_m",
            "driver_face_width_m",
            "driven_face_width_m",
            "geometry_contact_tolerance_m",
        )
    }
    if any(item <= 0.0 for item in positive.values()):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_GEOMETRY_INVALID",
            f"shaft coupling {mesh_id!r} has non-positive gear geometry",
        )
    stiffness = finite(
        value.get("mesh_stiffness_nm_rad", 30.0),
        f"{mesh_id}.mesh_stiffness_nm_rad",
    )
    damping = finite(
        value.get("mesh_damping_nm_s_rad", 0.04),
        f"{mesh_id}.mesh_damping_nm_s_rad",
    )
    if stiffness <= 0.0 or damping < 0.0:
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_COMPLIANCE_INVALID",
            f"shaft coupling {mesh_id!r} has invalid compliance",
        )
    return MultiOutputShaftCoupling(
        mesh_id=mesh_id,
        driver_body_id=driver,
        driven_body_id=driven,
        driver_teeth=driver_teeth,
        driven_teeth=driven_teeth,
        declared_ratio=ratio,
        shaft_angle_rad=shaft_angle,
        driver_pitch_radius_m=positive["driver_pitch_radius_m"],
        driven_pitch_radius_m=positive["driven_pitch_radius_m"],
        driver_face_width_m=positive["driver_face_width_m"],
        driven_face_width_m=positive["driven_face_width_m"],
        geometry_contact_tolerance_m=positive[
            "geometry_contact_tolerance_m"
        ],
        axis_perpendicular_tolerance_rad=perpendicular_tolerance,
        mesh_stiffness_nm_rad=stiffness,
        mesh_damping_nm_s_rad=damping,
        phase_offset_rad=finite(
            value.get("phase_offset_rad", 0.0),
            f"{mesh_id}.phase_offset_rad",
        ),
    )


def multi_output_shaft_from_dict(
    value: Mapping[str, Any] | None,
) -> MultiOutputShaftSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_SPEC_REQUIRED",
            "multi-output shaft adapter requires a family_spec object",
        )
    kind = str(value.get("mechanism_kind") or "")
    if kind not in SUPPORTED_KINDS:
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_KIND_INVALID",
            f"unsupported multi-output shaft mechanism kind: {kind!r}",
        )
    mechanism_id = str(value.get("mechanism_id") or "")
    identifier(mechanism_id, "family_spec.mechanism_id")
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(value.get("bodies") or ())
        if isinstance(item, Mapping)
    )
    shafts = tuple(
        _shaft_from_dict(item, index)
        for index, item in enumerate(value.get("shafts") or ())
        if isinstance(item, Mapping)
    )
    couplings = tuple(
        _coupling_from_dict(item, index)
        for index, item in enumerate(
            value.get("crown_bevel_meshes")
            or value.get("shaft_couplings")
            or ()
        )
        if isinstance(item, Mapping)
    )
    body_ids = {item.body_id for item in bodies}
    if len(bodies) < 4 or len(body_ids) != len(bodies):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_BODY_SET_INVALID",
            "multi-output shaft template requires unique housing, input, and outputs",
        )
    fixed_body_id = str(value.get("fixed_body_id") or "")
    input_body_id = str(value.get("input_body_id") or "")
    output_body_ids = tuple(
        str(item) for item in value.get("output_body_ids") or ()
    )
    for body_id in (fixed_body_id, input_body_id, *output_body_ids):
        identifier(body_id, "multi_output_shaft.body_id")
        if body_id not in body_ids:
            raise MechanismValidationError(
                "MULTI_OUTPUT_SHAFT_BODY_UNKNOWN",
                f"multi-output shaft references unknown body {body_id!r}",
            )
    if (
        len(output_body_ids) < 2
        or len(set(output_body_ids)) != len(output_body_ids)
        or fixed_body_id in output_body_ids
        or input_body_id in output_body_ids
        or fixed_body_id == input_body_id
    ):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_IO_INVALID",
            "multi-output shaft needs one input and at least two distinct outputs",
        )
    fixed = {item.body_id for item in bodies if item.fixed}
    if fixed != {fixed_body_id}:
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_FIXED_BODY_INVALID",
            "exactly the declared housing body must be fixed",
        )
    shaft_by_body = {item.body_id: item for item in shafts}
    moving = body_ids - {fixed_body_id}
    if set(shaft_by_body) != moving or len(shaft_by_body) != len(shafts):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_SUPPORT_INVENTORY_INVALID",
            "every moving body needs exactly one declared shaft support",
        )
    mesh_ids = {item.mesh_id for item in couplings}
    if len(couplings) < len(output_body_ids) or len(mesh_ids) != len(couplings):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_COUPLING_INVENTORY_INVALID",
            "every output needs a uniquely identified shaft coupling",
        )
    for coupling in couplings:
        if (
            coupling.driver_body_id not in moving
            or coupling.driven_body_id not in moving
        ):
            raise MechanismValidationError(
                "MULTI_OUTPUT_SHAFT_COUPLING_BODY_UNKNOWN",
                f"shaft coupling {coupling.mesh_id!r} references an unsupported body",
            )
        driver_axis = shaft_by_body[coupling.driver_body_id].axis_abs
        driven_axis = shaft_by_body[coupling.driven_body_id].axis_abs
        measured_angle = math.acos(
            max(-1.0, min(1.0, abs(sum(
                left * right
                for left, right in zip(driver_axis, driven_axis)
            ))))
        )
        if abs(measured_angle - coupling.shaft_angle_rad) > (
            coupling.axis_perpendicular_tolerance_rad
        ):
            raise MechanismValidationError(
                "MULTI_OUTPUT_SHAFT_AXIS_BINDING_INVALID",
                f"shaft coupling {coupling.mesh_id!r} axes differ from its declaration",
            )
    positive = {
        name: finite(value.get(name), name)
        for name in (
            "nominal_input_torque_nm",
            "max_torque_phase_s",
            "max_return_phase_s",
            "max_support_drift_m",
            "max_off_axis_rotation_rad",
            "max_mesh_error_rad",
            "minimum_output_travel_rad",
            "output_pair_ratio_tolerance",
            "input_output_ratio_tolerance",
        )
    }
    if any(item <= 0.0 for item in positive.values()):
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_THRESHOLD_INVALID",
            "multi-output shaft torques, durations, and thresholds must be positive",
        )
    return_torque = finite(
        value.get("return_input_torque_nm"),
        "return_input_torque_nm",
    )
    settle = finite(value.get("settle_phase_s"), "settle_phase_s")
    if return_torque < 0.0 or settle < 0.0:
        raise MechanismValidationError(
            "MULTI_OUTPUT_SHAFT_PHASE_INVALID",
            "return torque and settle duration cannot be negative",
        )
    spec = MultiOutputShaftSpec(
        mechanism_id=mechanism_id,
        mechanism_kind=kind,
        bodies=bodies,
        shafts=shafts,
        gear_meshes=couplings,
        fixed_body_id=fixed_body_id,
        input_body_id=input_body_id,
        output_body_ids=output_body_ids,
        nominal_input_torque_nm=positive["nominal_input_torque_nm"],
        return_input_torque_nm=return_torque,
        max_torque_phase_s=positive["max_torque_phase_s"],
        max_return_phase_s=positive["max_return_phase_s"],
        settle_phase_s=settle,
        max_support_drift_m=positive["max_support_drift_m"],
        max_off_axis_rotation_rad=positive[
            "max_off_axis_rotation_rad"
        ],
        max_mesh_error_rad=positive["max_mesh_error_rad"],
        minimum_output_travel_rad=positive[
            "minimum_output_travel_rad"
        ],
        output_pair_ratio_tolerance=positive[
            "output_pair_ratio_tolerance"
        ],
        input_output_ratio_tolerance=positive[
            "input_output_ratio_tolerance"
        ],
        friction=finite(value.get("friction", 0.06), "friction"),
    )
    for output_body_id in output_body_ids:
        spec.declared_ratio_for_body(output_body_id)
    return spec
