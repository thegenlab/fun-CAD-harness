from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .models import MechanismValidationError, finite, identifier, vector3


@dataclass(frozen=True)
class CompoundBodySpec:
    body_id: str
    group_ids: tuple[str, ...]
    fixed: bool
    mass_kg: float
    center_abs_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    rotation_abs_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class CompoundRevoluteJointSpec:
    joint_id: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    min_angle_rad: float | None = None
    max_angle_rad: float | None = None
    free_axis_damping_n_m_s_rad: float = 0.05
    limit_stiffness_n_m_rad: float = 650.0
    limit_damping_n_m_s_rad: float = 5.0
    limit_approach_zone_rad: float = 0.0
    limit_approach_damping_n_m_s_rad: float = 0.0


@dataclass(frozen=True)
class CompoundLinkageSpec:
    linkage_id: str
    bodies: tuple[CompoundBodySpec, ...]
    revolute_joints: tuple[CompoundRevoluteJointSpec, ...]
    input_body_id: str
    output_body_id: str
    input_joint_id: str
    output_joint_id: str
    input_point_abs_m: tuple[float, float, float]
    input_force_direction_abs: tuple[float, float, float]
    output_observable_point_abs_m: tuple[float, float, float]
    home_output_angle_rad: float
    outbound_output_angle_rad: float
    nominal_input_force_n: float
    return_input_force_n: float
    max_force_phase_s: float
    max_return_phase_s: float
    settle_phase_s: float
    output_angle_tolerance_rad: float
    max_joint_drift_m: float
    max_off_plane_rotation_rad: float
    friction: float = 0.08
    input_actuation_kind: str = "point_force"
    nominal_input_torque_nm: float = 0.0
    return_input_torque_nm: float = 0.0

    @property
    def loop_count(self) -> int:
        return max(0, len(self.revolute_joints) - len(self.bodies) + 1)

    @property
    def input_wrench_kind(self) -> str:
        return "torque" if self.input_actuation_kind == "joint_torque" else "force"

    @property
    def input_actuation_direction_abs(self) -> tuple[float, float, float]:
        if self.input_actuation_kind == "joint_torque":
            return next(
                item.axis_abs for item in self.revolute_joints
                if item.joint_id == self.input_joint_id
            )
        return self.input_force_direction_abs

    @property
    def nominal_input_magnitude_si(self) -> float:
        return (
            self.nominal_input_torque_nm
            if self.input_actuation_kind == "joint_torque"
            else self.nominal_input_force_n
        )

    @property
    def return_input_magnitude_si(self) -> float:
        return (
            self.return_input_torque_nm
            if self.input_actuation_kind == "joint_torque"
            else self.return_input_force_n
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bodies"] = [asdict(item) for item in self.bodies]
        payload["revolute_joints"] = [
            asdict(item) for item in self.revolute_joints
        ]
        return payload


def _tuple3(value: Any, label: str, *, unit: bool = False) -> tuple[float, float, float]:
    return vector3(tuple(value), label, unit=unit, allow_zero=not unit)


def _body_from_dict(value: Mapping[str, Any], index: int) -> CompoundBodySpec:
    body_id = str(value.get("body_id") or "")
    identifier(body_id, f"bodies[{index}].body_id")
    groups = tuple(str(item) for item in value.get("group_ids", ()))
    if not groups:
        raise MechanismValidationError(
            "COMPOUND_BODY_GROUPS_REQUIRED",
            f"compound body {body_id!r} has no rigid groups",
        )
    for group_index, group_id in enumerate(groups):
        identifier(group_id, f"bodies[{index}].group_ids[{group_index}]")
    mass = finite(value.get("mass_kg"), f"bodies[{index}].mass_kg")
    if mass <= 0:
        raise MechanismValidationError(
            "COMPOUND_BODY_MASS_INVALID",
            f"compound body {body_id!r} mass must be positive",
        )
    size = _tuple3(value.get("size_m"), f"bodies[{index}].size_m")
    if any(component <= 0 for component in size):
        raise MechanismValidationError(
            "COMPOUND_BODY_SIZE_INVALID",
            f"compound body {body_id!r} size must be positive",
        )
    quaternion = tuple(
        finite(item, f"bodies[{index}].rotation_abs_quat_wxyz")
        for item in value.get("rotation_abs_quat_wxyz", (1.0, 0.0, 0.0, 0.0))
    )
    if len(quaternion) != 4 or sum(item * item for item in quaternion) <= 1e-12:
        raise MechanismValidationError(
            "COMPOUND_BODY_ROTATION_INVALID",
            f"compound body {body_id!r} rotation quaternion is invalid",
        )
    return CompoundBodySpec(
        body_id=body_id,
        group_ids=groups,
        fixed=bool(value.get("fixed", False)),
        mass_kg=mass,
        center_abs_m=_tuple3(
            value.get("center_abs_m"), f"bodies[{index}].center_abs_m"
        ),
        size_m=size,
        rotation_abs_quat_wxyz=quaternion,
    )


def _joint_from_dict(
    value: Mapping[str, Any], index: int
) -> CompoundRevoluteJointSpec:
    joint_id = str(value.get("joint_id") or "")
    parent = str(value.get("parent_body_id") or "")
    child = str(value.get("child_body_id") or "")
    identifier(joint_id, f"revolute_joints[{index}].joint_id")
    identifier(parent, f"revolute_joints[{index}].parent_body_id")
    identifier(child, f"revolute_joints[{index}].child_body_id")
    if parent == child:
        raise MechanismValidationError(
            "COMPOUND_JOINT_SELF_REFERENCE",
            f"compound joint {joint_id!r} connects a body to itself",
        )
    minimum = value.get("min_angle_rad")
    maximum = value.get("max_angle_rad")
    minimum_value = None if minimum is None else finite(minimum, f"{joint_id}.min_angle_rad")
    maximum_value = None if maximum is None else finite(maximum, f"{joint_id}.max_angle_rad")
    if (minimum_value is None) != (maximum_value is None):
        raise MechanismValidationError(
            "COMPOUND_JOINT_LIMIT_INCOMPLETE",
            f"compound joint {joint_id!r} must declare both angle limits or neither",
        )
    if (
        minimum_value is not None
        and maximum_value is not None
        and minimum_value >= maximum_value
    ):
        raise MechanismValidationError(
            "COMPOUND_JOINT_LIMIT_INVALID",
            f"compound joint {joint_id!r} limits are reversed",
        )
    free_axis_damping = finite(
        value.get("free_axis_damping_n_m_s_rad", 0.05),
        f"{joint_id}.free_axis_damping_n_m_s_rad",
    )
    limit_stiffness = finite(
        value.get("limit_stiffness_n_m_rad", 650.0),
        f"{joint_id}.limit_stiffness_n_m_rad",
    )
    limit_damping = finite(
        value.get("limit_damping_n_m_s_rad", 5.0),
        f"{joint_id}.limit_damping_n_m_s_rad",
    )
    limit_approach_zone = finite(
        value.get("limit_approach_zone_rad", 0.0),
        f"{joint_id}.limit_approach_zone_rad",
    )
    limit_approach_damping = finite(
        value.get("limit_approach_damping_n_m_s_rad", 0.0),
        f"{joint_id}.limit_approach_damping_n_m_s_rad",
    )
    if (
        not 0.0 <= free_axis_damping <= 100.0
        or not 0.0 < limit_stiffness <= 1_000_000.0
        or not 0.0 <= limit_damping <= 10_000.0
        or not 0.0 <= limit_approach_zone <= 0.5
        or not 0.0 <= limit_approach_damping <= 10_000.0
        or (limit_approach_zone == 0.0) != (limit_approach_damping == 0.0)
    ):
        raise MechanismValidationError(
            "COMPOUND_JOINT_PHYSICAL_PARAMETER_INVALID",
            f"compound joint {joint_id!r} damping or stop parameters are outside validator bounds",
        )
    return CompoundRevoluteJointSpec(
        joint_id=joint_id,
        parent_body_id=parent,
        child_body_id=child,
        origin_abs_m=_tuple3(
            value.get("origin_abs_m"), f"revolute_joints[{index}].origin_abs_m"
        ),
        axis_abs=_tuple3(
            value.get("axis_abs"), f"revolute_joints[{index}].axis_abs", unit=True
        ),
        min_angle_rad=minimum_value,
        max_angle_rad=maximum_value,
        free_axis_damping_n_m_s_rad=free_axis_damping,
        limit_stiffness_n_m_rad=limit_stiffness,
        limit_damping_n_m_s_rad=limit_damping,
        limit_approach_zone_rad=limit_approach_zone,
        limit_approach_damping_n_m_s_rad=limit_approach_damping,
    )


def compound_linkage_from_dict(value: Mapping[str, Any] | None) -> CompoundLinkageSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "COMPOUND_SPEC_REQUIRED",
            "compound mechanism adapter requires a family_spec object",
        )
    linkage_id = str(value.get("linkage_id") or "")
    identifier(linkage_id, "family_spec.linkage_id")
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(value.get("bodies", ()))
        if isinstance(item, Mapping)
    )
    joints = tuple(
        _joint_from_dict(item, index)
        for index, item in enumerate(value.get("revolute_joints", ()))
        if isinstance(item, Mapping)
    )
    if len(bodies) < 3 or len(joints) < 2:
        raise MechanismValidationError(
            "COMPOUND_TOPOLOGY_TOO_SMALL",
            "compound linkage requires at least three bodies and two joints",
        )
    body_ids = {item.body_id for item in bodies}
    if len(body_ids) != len(bodies):
        raise MechanismValidationError(
            "COMPOUND_BODY_ID_DUPLICATE", "compound body IDs must be unique"
        )
    joint_ids = {item.joint_id for item in joints}
    if len(joint_ids) != len(joints):
        raise MechanismValidationError(
            "COMPOUND_JOINT_ID_DUPLICATE", "compound joint IDs must be unique"
        )
    fixed = [item.body_id for item in bodies if item.fixed]
    if len(fixed) != 1:
        raise MechanismValidationError(
            "COMPOUND_FIXED_BODY_COUNT",
            "compound linkage requires exactly one fixed body",
        )
    groups = [group for body in bodies for group in body.group_ids]
    if len(groups) != len(set(groups)):
        raise MechanismValidationError(
            "COMPOUND_GROUP_OVERLAP",
            "a rigid group cannot belong to multiple compound bodies",
        )
    for joint in joints:
        if joint.parent_body_id not in body_ids or joint.child_body_id not in body_ids:
            raise MechanismValidationError(
                "COMPOUND_JOINT_BODY_UNKNOWN",
                f"compound joint {joint.joint_id!r} references an unknown body",
            )
    input_body = str(value.get("input_body_id") or "")
    output_body = str(value.get("output_body_id") or "")
    input_joint = str(value.get("input_joint_id") or "")
    output_joint = str(value.get("output_joint_id") or "")
    if input_body not in body_ids or output_body not in body_ids:
        raise MechanismValidationError(
            "COMPOUND_IO_BODY_UNKNOWN", "compound input/output body is unknown"
        )
    if input_joint not in joint_ids or output_joint not in joint_ids:
        raise MechanismValidationError(
            "COMPOUND_IO_JOINT_UNKNOWN", "compound input/output joint is unknown"
        )
    joints_by_id = {item.joint_id: item for item in joints}
    if joints_by_id[input_joint].child_body_id != input_body:
        raise MechanismValidationError(
            "COMPOUND_INPUT_CHILD_MISMATCH",
            "compound input body must be the child of the declared input joint",
        )
    if joints_by_id[output_joint].child_body_id != output_body:
        raise MechanismValidationError(
            "COMPOUND_OUTPUT_CHILD_MISMATCH",
            "compound output body must be the child of the declared output joint",
        )
    adjacency = {body_id: set() for body_id in body_ids}
    for joint in joints:
        adjacency[joint.parent_body_id].add(joint.child_body_id)
        adjacency[joint.child_body_id].add(joint.parent_body_id)
    visited: set[str] = set()
    pending = [fixed[0]]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    if visited != body_ids:
        raise MechanismValidationError(
            "COMPOUND_TOPOLOGY_DISCONNECTED",
            "compound body/joint graph is disconnected",
        )
    home = finite(value.get("home_output_angle_rad"), "home_output_angle_rad")
    outbound = finite(
        value.get("outbound_output_angle_rad"), "outbound_output_angle_rad"
    )
    if outbound <= home:
        raise MechanismValidationError(
            "COMPOUND_OUTPUT_RANGE_INVALID",
            "compound outbound output angle must exceed home angle",
        )
    nominal_force = finite(value.get("nominal_input_force_n"), "nominal_input_force_n")
    return_force = finite(value.get("return_input_force_n", 0.0), "return_input_force_n")
    if nominal_force <= 0 or return_force < 0:
        raise MechanismValidationError(
            "COMPOUND_INPUT_FORCE_INVALID",
            "compound nominal force must be positive and return force non-negative",
        )
    actuation_kind = str(value.get("input_actuation_kind") or "point_force")
    if actuation_kind not in {"point_force", "joint_torque"}:
        raise MechanismValidationError(
            "COMPOUND_INPUT_ACTUATION_INVALID",
            "compound input actuation must be point_force or joint_torque",
        )
    nominal_torque = finite(
        value.get("nominal_input_torque_nm", 0.0), "nominal_input_torque_nm"
    )
    return_torque = finite(
        value.get("return_input_torque_nm", 0.0), "return_input_torque_nm"
    )
    if nominal_torque < 0 or return_torque < 0:
        raise MechanismValidationError(
            "COMPOUND_INPUT_TORQUE_INVALID",
            "compound input torque values cannot be negative",
        )
    if actuation_kind == "joint_torque" and nominal_torque <= 0:
        raise MechanismValidationError(
            "COMPOUND_INPUT_TORQUE_INVALID",
            "joint_torque actuation requires a positive nominal input torque",
        )
    maximum_phase = finite(value.get("max_force_phase_s"), "max_force_phase_s")
    maximum_return_phase = finite(
        value.get("max_return_phase_s", maximum_phase), "max_return_phase_s"
    )
    settle = finite(value.get("settle_phase_s"), "settle_phase_s")
    if maximum_phase <= 0 or maximum_return_phase <= 0 or settle < 0:
        raise MechanismValidationError(
            "COMPOUND_PHASE_DURATION_INVALID",
            "compound phase durations are invalid",
        )
    tolerance = finite(
        value.get("output_angle_tolerance_rad"), "output_angle_tolerance_rad"
    )
    joint_drift = finite(value.get("max_joint_drift_m"), "max_joint_drift_m")
    off_plane = finite(
        value.get("max_off_plane_rotation_rad"), "max_off_plane_rotation_rad"
    )
    if tolerance <= 0 or joint_drift <= 0 or off_plane <= 0:
        raise MechanismValidationError(
            "COMPOUND_THRESHOLD_INVALID",
            "compound comparison thresholds must be positive",
        )
    return CompoundLinkageSpec(
        linkage_id=linkage_id,
        bodies=bodies,
        revolute_joints=joints,
        input_body_id=input_body,
        output_body_id=output_body,
        input_joint_id=input_joint,
        output_joint_id=output_joint,
        input_point_abs_m=_tuple3(value.get("input_point_abs_m"), "input_point_abs_m"),
        input_force_direction_abs=_tuple3(
            value.get("input_force_direction_abs"),
            "input_force_direction_abs",
            unit=True,
        ),
        output_observable_point_abs_m=_tuple3(
            value.get("output_observable_point_abs_m"),
            "output_observable_point_abs_m",
        ),
        home_output_angle_rad=home,
        outbound_output_angle_rad=outbound,
        nominal_input_force_n=nominal_force,
        return_input_force_n=return_force,
        max_force_phase_s=maximum_phase,
        max_return_phase_s=maximum_return_phase,
        settle_phase_s=settle,
        output_angle_tolerance_rad=tolerance,
        max_joint_drift_m=joint_drift,
        max_off_plane_rotation_rad=off_plane,
        friction=finite(value.get("friction", 0.08), "friction"),
        input_actuation_kind=actuation_kind,
        nominal_input_torque_nm=nominal_torque,
        return_input_torque_nm=return_torque,
    )
