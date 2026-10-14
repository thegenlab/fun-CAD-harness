"""Family spec for the ``hydraulic_scissor_lift`` mechanism family.

A single-tier hydraulic scissor lift: a fixed base frame, two scissor frames
cross-pinned at the center, a level-keeping lift platform (the output body),
and a hydraulic cylinder (barrel + rod) that drives the inner frame.  The
reduced branch models the declared topology with ideal links (revolute pins,
prismatic roller tracks, one cylindrical barrel-rod link); the physical branch
replaces every joint with a finite bushing compliance mount and enforces the
declared travel limits as forces.  Bodies reuse :class:`CompoundBodySpec` so
collision-geometry binding and body construction stay shared with the other
compound families.  Mirrors ``rack_pinion_spec.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .compound_spec import (
    CompoundBodySpec,
    CompoundRevoluteJointSpec,
    _body_from_dict,
    _joint_from_dict,
)
from .finite_linear_capture import (
    FiniteLinearCaptureSpec,
    finite_linear_capture_from_dict,
)
from .models import MechanismValidationError, finite, identifier, vector3


@dataclass(frozen=True)
class ScissorLiftPrismaticJointSpec:
    """Roller-track prismatic joint between two scissor-lift bodies."""

    joint_id: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]


@dataclass(frozen=True)
class ScissorLiftCylindricalJointSpec:
    """Cylindrical barrel-rod joint with declared axial travel limits."""

    joint_id: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    min_length_m: float
    max_length_m: float


@dataclass(frozen=True)
class ScissorLiftActuatorSpec:
    """Hydraulic cylinder modelled as a force pair between two anchors."""

    kind: str
    barrel_body_id: str
    rod_body_id: str
    barrel_anchor_abs_m: tuple[float, float, float]
    rod_anchor_abs_m: tuple[float, float, float]
    free_axial_rotation: bool
    min_length_m: float
    max_length_m: float


@dataclass(frozen=True)
class ScissorLiftSpec:
    """Immutable execution contract for one hydraulic scissor-lift mechanism."""

    mechanism_kind: str
    mechanism_id: str
    bodies: tuple[CompoundBodySpec, ...]
    fixed_body_id: str
    revolute_joints: tuple[CompoundRevoluteJointSpec, ...]
    prismatic_joints: tuple[ScissorLiftPrismaticJointSpec, ...]
    cylindrical_joint: ScissorLiftCylindricalJointSpec
    finite_linear_captures: tuple[FiniteLinearCaptureSpec, ...]
    actuator: ScissorLiftActuatorSpec
    output_body_id: str
    output_axis_abs: tuple[float, float, float]
    output_reference_point_abs_m: tuple[float, float, float]
    home_progress_m: float
    outbound_progress_m: float
    nominal_input_force_n: float
    return_input_force_n: float
    max_force_phase_s: float
    max_return_phase_s: float
    settle_phase_s: float
    output_progress_tolerance_m: float
    max_joint_drift_m: float
    max_off_plane_rotation_rad: float
    platform_level_tolerance_rad: float
    friction: float = 0.08
    geometry_reference: dict[str, Any] = field(default_factory=dict)

    @property
    def joint_ids(self) -> tuple[str, ...]:
        return (
            *(item.joint_id for item in self.revolute_joints),
            *(item.joint_id for item in self.prismatic_joints),
            self.cylindrical_joint.joint_id,
        )

    @property
    def joint_count(self) -> int:
        return len(self.revolute_joints) + len(self.prismatic_joints) + 1

    @property
    def loop_count(self) -> int:
        return max(0, self.joint_count - len(self.bodies) + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expected_ideal_links(spec: ScissorLiftSpec) -> tuple[str, ...]:
    """Ideal-link class list the reduced branch builds, in declaration order.

    For the canonical single-tier family spec (5 revolute pins, 2 roller
    tracks, 1 cylindrical barrel-rod joint) this is exactly
    ``("ChLinkLockRevolute",) * 5 + ("ChLinkLockRevolutePrismatic",) * 2 +
    ("ChLinkLockCylindrical",)``.  The roller tracks are revolute-prismatic
    links: the roller end of the scissor frame slides along the track AND
    swings about the scissor hinge axis, so a plain prismatic link would
    lock the arm-swing rotation and over-constrain the loop.
    """
    return (
        *("ChLinkLockRevolute" for _ in spec.revolute_joints),
        *("ChLinkLockRevolutePrismatic" for _ in spec.prismatic_joints),
        "ChLinkLockCylindrical",
    )


def _unit3(value: Any, label: str) -> tuple[float, float, float]:
    return vector3(tuple(value), label, unit=True, allow_zero=False)


def _point3(value: Any, label: str) -> tuple[float, float, float]:
    return vector3(tuple(value), label, allow_zero=True)


def _prismatic_from_dict(
    value: Mapping[str, Any], index: int
) -> ScissorLiftPrismaticJointSpec:
    joint_id = str(value.get("joint_id") or "")
    parent = str(value.get("parent_body_id") or "")
    child = str(value.get("child_body_id") or "")
    identifier(joint_id, f"prismatic_joints[{index}].joint_id")
    identifier(parent, f"prismatic_joints[{index}].parent_body_id")
    identifier(child, f"prismatic_joints[{index}].child_body_id")
    if parent == child:
        raise MechanismValidationError(
            "SCISSOR_LIFT_JOINT_SELF_REFERENCE",
            f"scissor-lift joint {joint_id!r} connects a body to itself",
        )
    return ScissorLiftPrismaticJointSpec(
        joint_id=joint_id,
        parent_body_id=parent,
        child_body_id=child,
        origin_abs_m=_point3(
            value.get("origin_abs_m"), f"prismatic_joints[{index}].origin_abs_m"
        ),
        axis_abs=_unit3(
            value.get("axis_abs"), f"prismatic_joints[{index}].axis_abs"
        ),
    )


def _cylindrical_from_dict(value: Any) -> ScissorLiftCylindricalJointSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "SCISSOR_LIFT_CYLINDRICAL_REQUIRED",
            "hydraulic scissor lift requires a cylindrical_joint object",
        )
    joint_id = str(value.get("joint_id") or "")
    parent = str(value.get("parent_body_id") or "")
    child = str(value.get("child_body_id") or "")
    identifier(joint_id, "cylindrical_joint.joint_id")
    identifier(parent, "cylindrical_joint.parent_body_id")
    identifier(child, "cylindrical_joint.child_body_id")
    if parent == child:
        raise MechanismValidationError(
            "SCISSOR_LIFT_JOINT_SELF_REFERENCE",
            f"scissor-lift joint {joint_id!r} connects a body to itself",
        )
    minimum = finite(value.get("min_length_m"), "cylindrical_joint.min_length_m")
    maximum = finite(value.get("max_length_m"), "cylindrical_joint.max_length_m")
    if minimum <= 0 or maximum <= minimum:
        raise MechanismValidationError(
            "SCISSOR_LIFT_CYLINDER_TRAVEL_INVALID",
            "cylindrical joint travel limits must be positive and ordered",
        )
    return ScissorLiftCylindricalJointSpec(
        joint_id=joint_id,
        parent_body_id=parent,
        child_body_id=child,
        origin_abs_m=_point3(value.get("origin_abs_m"), "cylindrical_joint.origin_abs_m"),
        axis_abs=_unit3(value.get("axis_abs"), "cylindrical_joint.axis_abs"),
        min_length_m=minimum,
        max_length_m=maximum,
    )


def _actuator_from_dict(value: Any) -> ScissorLiftActuatorSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "SCISSOR_LIFT_ACTUATOR_REQUIRED",
            "hydraulic scissor lift requires an actuator object",
        )
    if value.get("kind") != "hydraulic_cylinder_force_pair":
        raise MechanismValidationError(
            "SCISSOR_LIFT_ACTUATOR_KIND_INVALID",
            "scissor-lift actuator must declare kind='hydraulic_cylinder_force_pair'",
        )
    barrel = str(value.get("barrel_body_id") or "")
    rod = str(value.get("rod_body_id") or "")
    identifier(barrel, "actuator.barrel_body_id")
    identifier(rod, "actuator.rod_body_id")
    if barrel == rod:
        raise MechanismValidationError(
            "SCISSOR_LIFT_ACTUATOR_SELF_REFERENCE",
            "scissor-lift actuator barrel and rod must be distinct bodies",
        )
    minimum = finite(value.get("min_length_m"), "actuator.min_length_m")
    maximum = finite(value.get("max_length_m"), "actuator.max_length_m")
    if minimum <= 0 or maximum <= minimum:
        raise MechanismValidationError(
            "SCISSOR_LIFT_CYLINDER_TRAVEL_INVALID",
            "actuator length limits must be positive and ordered",
        )
    return ScissorLiftActuatorSpec(
        kind="hydraulic_cylinder_force_pair",
        barrel_body_id=barrel,
        rod_body_id=rod,
        barrel_anchor_abs_m=_point3(
            value.get("barrel_anchor_abs_m"), "actuator.barrel_anchor_abs_m"
        ),
        rod_anchor_abs_m=_point3(
            value.get("rod_anchor_abs_m"), "actuator.rod_anchor_abs_m"
        ),
        free_axial_rotation=bool(value.get("free_axial_rotation", True)),
        min_length_m=minimum,
        max_length_m=maximum,
    )


def scissor_lift_from_dict(value: Mapping[str, Any] | None) -> ScissorLiftSpec:
    if not isinstance(value, Mapping):
        raise MechanismValidationError(
            "SCISSOR_LIFT_SPEC_REQUIRED",
            "scissor-lift mechanism adapter requires a family_spec object",
        )
    if value.get("mechanism_kind") != "hydraulic_scissor_lift":
        raise MechanismValidationError(
            "SCISSOR_LIFT_KIND_INVALID",
            "scissor-lift family_spec must declare "
            "mechanism_kind='hydraulic_scissor_lift'",
        )
    mechanism_id = str(value.get("mechanism_id") or "")
    identifier(mechanism_id, "family_spec.mechanism_id")
    raw_bodies = value.get("bodies", ())
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(raw_bodies)
        if isinstance(item, Mapping)
    )
    if len(bodies) != len(tuple(raw_bodies)) or len(bodies) < 4:
        raise MechanismValidationError(
            "SCISSOR_LIFT_BODY_COUNT_INVALID",
            "scissor lift requires at least base, two scissor frames, and platform bodies",
        )
    body_ids = {item.body_id for item in bodies}
    if len(body_ids) != len(bodies):
        raise MechanismValidationError(
            "SCISSOR_LIFT_BODY_ID_DUPLICATE", "scissor-lift body IDs must be unique"
        )
    groups = [group for body in bodies for group in body.group_ids]
    if len(groups) != len(set(groups)):
        raise MechanismValidationError(
            "SCISSOR_LIFT_GROUP_OVERLAP",
            "a rigid group cannot belong to multiple scissor-lift bodies",
        )
    fixed_body_id = str(value.get("fixed_body_id") or "")
    if fixed_body_id not in body_ids:
        raise MechanismValidationError(
            "SCISSOR_LIFT_BODY_UNKNOWN", "fixed_body_id references an unknown body"
        )
    fixed = [item.body_id for item in bodies if item.fixed]
    if fixed != [fixed_body_id]:
        raise MechanismValidationError(
            "SCISSOR_LIFT_FIXED_BODY_INVALID",
            "scissor lift requires exactly its declared fixed body to be fixed",
        )
    revolute_joints = tuple(
        _joint_from_dict(item, index)
        for index, item in enumerate(value.get("revolute_joints", ()))
        if isinstance(item, Mapping)
    )
    prismatic_joints = tuple(
        _prismatic_from_dict(item, index)
        for index, item in enumerate(value.get("prismatic_joints", ()))
        if isinstance(item, Mapping)
    )
    if len(revolute_joints) < 3 or len(prismatic_joints) < 1:
        raise MechanismValidationError(
            "SCISSOR_LIFT_TOPOLOGY_TOO_SMALL",
            "scissor lift requires at least three revolute pins and one roller track",
        )
    cylindrical_joint = _cylindrical_from_dict(value.get("cylindrical_joint"))
    finite_linear_captures = tuple(
        finite_linear_capture_from_dict(
            item,
            label=f"finite_linear_captures[{index}]",
        )
        for index, item in enumerate(value.get("finite_linear_captures", ()))
        if isinstance(item, Mapping)
    )
    raw_captures = tuple(value.get("finite_linear_captures", ()))
    if (
        len(finite_linear_captures) != len(raw_captures)
        or len(finite_linear_captures) != 1
    ):
        raise MechanismValidationError(
            "SCISSOR_LIFT_FINITE_CAPTURE_REQUIRED",
            "scissor-lift cylindrical joint requires exactly one finite "
            "STEP-bound capture declaration",
        )
    actuator = _actuator_from_dict(value.get("actuator"))
    all_joints: tuple[Any, ...] = (
        *revolute_joints,
        *prismatic_joints,
        cylindrical_joint,
    )
    joint_ids = {item.joint_id for item in all_joints}
    if len(joint_ids) != len(all_joints):
        raise MechanismValidationError(
            "SCISSOR_LIFT_JOINT_ID_DUPLICATE", "scissor-lift joint IDs must be unique"
        )
    for joint in all_joints:
        if joint.parent_body_id not in body_ids or joint.child_body_id not in body_ids:
            raise MechanismValidationError(
                "SCISSOR_LIFT_JOINT_BODY_UNKNOWN",
                f"scissor-lift joint {joint.joint_id!r} references an unknown body",
            )
    capture = finite_linear_captures[0]
    if (
        capture.capture_id != cylindrical_joint.joint_id
        or capture.parent_body_id != cylindrical_joint.parent_body_id
        or capture.child_body_id != cylindrical_joint.child_body_id
        or capture.parent_anchor_abs_m != actuator.barrel_anchor_abs_m
        or capture.child_anchor_abs_m != actuator.rod_anchor_abs_m
        or capture.axis_abs != cylindrical_joint.axis_abs
        or abs(
            capture.maximum_separation_m
            - cylindrical_joint.max_length_m
        )
        > 1e-9
    ):
        raise MechanismValidationError(
            "SCISSOR_LIFT_FINITE_CAPTURE_MISMATCH",
            "finite capture must bind the cylindrical joint bodies, anchors, "
            "axis, and maximum separation",
        )
    for body_id, label in (
        (actuator.barrel_body_id, "actuator.barrel_body_id"),
        (actuator.rod_body_id, "actuator.rod_body_id"),
    ):
        if body_id not in body_ids:
            raise MechanismValidationError(
                "SCISSOR_LIFT_BODY_UNKNOWN", f"{label} references an unknown body"
            )
    output_body_id = str(value.get("output_body_id") or "")
    if output_body_id not in body_ids:
        raise MechanismValidationError(
            "SCISSOR_LIFT_BODY_UNKNOWN", "output_body_id references an unknown body"
        )
    adjacency = {body_id: set() for body_id in body_ids}
    for joint in all_joints:
        adjacency[joint.parent_body_id].add(joint.child_body_id)
        adjacency[joint.child_body_id].add(joint.parent_body_id)
    visited: set[str] = set()
    pending = [fixed_body_id]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    if visited != body_ids:
        raise MechanismValidationError(
            "SCISSOR_LIFT_TOPOLOGY_DISCONNECTED",
            "scissor-lift body/joint graph is disconnected",
        )
    home = finite(value.get("home_progress_m"), "home_progress_m")
    outbound = finite(value.get("outbound_progress_m"), "outbound_progress_m")
    if outbound <= home:
        raise MechanismValidationError(
            "SCISSOR_LIFT_TRAVEL_INVALID",
            "outbound platform progress must exceed home progress",
        )
    nominal_force = finite(value.get("nominal_input_force_n"), "nominal_input_force_n")
    return_force = finite(value.get("return_input_force_n"), "return_input_force_n")
    if nominal_force <= 0 or return_force <= 0:
        raise MechanismValidationError(
            "SCISSOR_LIFT_INPUT_FORCE_INVALID",
            "outbound and return cylinder forces must be positive",
        )
    max_force_phase = finite(value.get("max_force_phase_s"), "max_force_phase_s")
    max_return_phase = finite(value.get("max_return_phase_s"), "max_return_phase_s")
    settle_phase = finite(value.get("settle_phase_s"), "settle_phase_s")
    if max_force_phase <= 0 or max_return_phase <= 0 or settle_phase < 0:
        raise MechanismValidationError(
            "SCISSOR_LIFT_PHASE_DURATION_INVALID",
            "scissor-lift phase durations are invalid",
        )
    thresholds = {
        "output_progress_tolerance_m": finite(
            value.get("output_progress_tolerance_m"), "output_progress_tolerance_m"
        ),
        "max_joint_drift_m": finite(
            value.get("max_joint_drift_m"), "max_joint_drift_m"
        ),
        "max_off_plane_rotation_rad": finite(
            value.get("max_off_plane_rotation_rad"), "max_off_plane_rotation_rad"
        ),
        "platform_level_tolerance_rad": finite(
            value.get("platform_level_tolerance_rad"), "platform_level_tolerance_rad"
        ),
    }
    if any(item <= 0 for item in thresholds.values()):
        raise MechanismValidationError(
            "SCISSOR_LIFT_THRESHOLD_INVALID",
            "scissor-lift comparison thresholds must be positive",
        )
    friction = finite(value.get("friction", 0.08), "friction")
    if friction < 0:
        raise MechanismValidationError(
            "SCISSOR_LIFT_FRICTION_INVALID", "friction cannot be negative"
        )
    geometry_reference = value.get("geometry_reference")
    if geometry_reference is not None and not isinstance(geometry_reference, Mapping):
        raise MechanismValidationError(
            "SCISSOR_LIFT_GEOMETRY_REFERENCE_INVALID",
            "geometry_reference must be an object when present",
        )
    return ScissorLiftSpec(
        mechanism_kind="hydraulic_scissor_lift",
        mechanism_id=mechanism_id,
        bodies=bodies,
        fixed_body_id=fixed_body_id,
        revolute_joints=revolute_joints,
        prismatic_joints=prismatic_joints,
        cylindrical_joint=cylindrical_joint,
        finite_linear_captures=finite_linear_captures,
        actuator=actuator,
        output_body_id=output_body_id,
        output_axis_abs=_unit3(value.get("output_axis_abs"), "output_axis_abs"),
        output_reference_point_abs_m=_point3(
            value.get("output_reference_point_abs_m"), "output_reference_point_abs_m"
        ),
        home_progress_m=home,
        outbound_progress_m=outbound,
        nominal_input_force_n=nominal_force,
        return_input_force_n=return_force,
        max_force_phase_s=max_force_phase,
        max_return_phase_s=max_return_phase,
        settle_phase_s=settle_phase,
        output_progress_tolerance_m=thresholds["output_progress_tolerance_m"],
        max_joint_drift_m=thresholds["max_joint_drift_m"],
        max_off_plane_rotation_rad=thresholds["max_off_plane_rotation_rad"],
        platform_level_tolerance_rad=thresholds["platform_level_tolerance_rad"],
        friction=friction,
        geometry_reference=(
            dict(geometry_reference) if geometry_reference is not None else {}
        ),
    )
