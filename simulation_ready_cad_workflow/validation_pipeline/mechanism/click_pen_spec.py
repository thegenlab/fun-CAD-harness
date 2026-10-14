from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .models import MechanismValidationError, finite, identifier, vector3


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _positive(value: Any, label: str) -> float:
    result = finite(value, label)
    if result <= 0.0:
        raise _fail("CLICK_PEN_RANGE_INVALID", f"{label} must be positive")
    return result


def _nonnegative(value: Any, label: str) -> float:
    result = finite(value, label)
    if result < 0.0:
        raise _fail("CLICK_PEN_RANGE_INVALID", f"{label} cannot be negative")
    return result


def _tuple3(
    value: Any,
    label: str,
    *,
    unit: bool = False,
) -> tuple[float, float, float]:
    return vector3(value, label, unit=unit, allow_zero=not unit)


def _component_ref(value: Any, label: str) -> str:
    result = str(value or "")
    body_id, separator, solid_id = result.partition("/")
    if separator != "/" or not body_id or not solid_id:
        raise _fail(
            "CLICK_PEN_COMPONENT_REF_INVALID",
            f"{label} must be body_id/solid_id",
        )
    identifier(body_id, f"{label}.body_id")
    identifier(solid_id, f"{label}.solid_id")
    return result


def _component_refs(value: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
    ):
        raise _fail(
            "CLICK_PEN_COMPONENT_REF_INVALID",
            f"{label} must be a non-empty array",
        )
    result = tuple(
        _component_ref(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise _fail(
            "CLICK_PEN_COMPONENT_REF_INVALID",
            f"{label} contains duplicates",
        )
    return result


@dataclass(frozen=True)
class ClickPenBodySpec:
    body_id: str
    group_ids: tuple[str, ...]
    fixed: bool
    mass_kg: float
    center_abs_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    rotation_abs_quat_wxyz: tuple[float, float, float, float] = (
        1.0,
        0.0,
        0.0,
        0.0,
    )


@dataclass(frozen=True)
class ClickPenLinearGuideSpec:
    joint_id: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    minimum_position_m: float
    maximum_position_m: float
    spring_stiffness_n_m: float
    damping_n_s_m: float
    spring_preload_n: float


@dataclass(frozen=True)
class ClickPenCylindricalGuideSpec:
    joint_id: str
    parent_body_id: str
    child_body_id: str
    origin_abs_m: tuple[float, float, float]
    axis_abs: tuple[float, float, float]
    minimum_position_m: float
    maximum_position_m: float
    index_angle_rad: float


@dataclass(frozen=True)
class ClickPenGeometrySpec:
    button_contact_component_refs: tuple[str, ...]
    cam_button_face_component_refs: tuple[str, ...]
    cam_follower_component_ref: str
    refill_carrier_component_ref: str
    barrel_guide_component_refs: tuple[str, ...]
    cam_lobe_component_refs: tuple[str, ...]
    home_contact_tolerance_m: float
    full_press_offset_abs_m: tuple[float, float, float]
    extended_latched_cam_offset_abs_m: tuple[float, float, float]
    extended_latched_refill_offset_abs_m: tuple[float, float, float]


@dataclass(frozen=True)
class ClickPenStateSpec:
    states: tuple[str, ...]
    initial_state: str
    first_click_target_m: float
    second_click_target_m: float
    latch_contact_tolerance_m: float
    release_button_return_m: float
    geometry_contract: ClickPenGeometrySpec


@dataclass(frozen=True)
class ClickPenCamLatchSpec:
    mechanism_kind: str
    mechanism_id: str
    bodies: tuple[ClickPenBodySpec, ...]
    button_prismatic_joint: ClickPenLinearGuideSpec
    refill_prismatic_joint: ClickPenLinearGuideSpec
    cam_cylindrical_joint: ClickPenCylindricalGuideSpec
    click_state_machine: ClickPenStateSpec
    input_body_id: str
    output_body_id: str
    nominal_press_force_n: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    friction: float

    @property
    def fixed_body_id(self) -> str:
        fixed = [body.body_id for body in self.bodies if body.fixed]
        if len(fixed) != 1:
            raise _fail(
                "CLICK_PEN_FIXED_BODY_INVALID",
                "click pen requires exactly one fixed body",
            )
        return fixed[0]

    @property
    def cam_body_id(self) -> str:
        return self.cam_cylindrical_joint.child_body_id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _body(value: Mapping[str, Any], index: int) -> ClickPenBodySpec:
    label = f"bodies[{index}]"
    groups_value = value.get("group_ids")
    if (
        not isinstance(groups_value, Sequence)
        or isinstance(groups_value, (str, bytes))
        or not groups_value
    ):
        raise _fail(
            "CLICK_PEN_GROUPS_INVALID",
            f"{label}.group_ids must be a non-empty array",
        )
    groups = tuple(
        identifier(item, f"{label}.group_ids[{group_index}]")
        for group_index, item in enumerate(groups_value)
    )
    fixed = value.get("fixed")
    if not isinstance(fixed, bool):
        raise _fail(
            "CLICK_PEN_BODY_INVALID",
            f"{label}.fixed must be boolean",
        )
    size = _tuple3(value.get("size_m"), f"{label}.size_m")
    if any(item <= 0.0 for item in size):
        raise _fail(
            "CLICK_PEN_BODY_INVALID",
            f"{label}.size_m must be positive",
        )
    rotation_value = value.get(
        "rotation_abs_quat_wxyz",
        (1.0, 0.0, 0.0, 0.0),
    )
    if not isinstance(rotation_value, Sequence) or len(rotation_value) != 4:
        raise _fail(
            "CLICK_PEN_BODY_INVALID",
            f"{label}.rotation_abs_quat_wxyz must contain four numbers",
        )
    return ClickPenBodySpec(
        body_id=identifier(value.get("body_id"), f"{label}.body_id"),
        group_ids=groups,
        fixed=fixed,
        mass_kg=_positive(value.get("mass_kg"), f"{label}.mass_kg"),
        center_abs_m=_tuple3(value.get("center_abs_m"), f"{label}.center_abs_m"),
        size_m=size,
        rotation_abs_quat_wxyz=tuple(
            finite(item, f"{label}.rotation_abs_quat_wxyz[{component}]")
            for component, item in enumerate(rotation_value)
        ),
    )


def _linear_guide(
    value: Mapping[str, Any],
    label: str,
    *,
    minimum_default: float,
    maximum_fields: tuple[str, ...],
    spring_defaults: tuple[float, float, float],
) -> ClickPenLinearGuideSpec:
    minimum = finite(
        value.get(
            "minimum_position_m",
            value.get("min_position_m", minimum_default),
        ),
        f"{label}.minimum_position_m",
    )
    maximum_value = None
    for field in maximum_fields:
        if value.get(field) is not None:
            maximum_value = value[field]
            break
    maximum = finite(maximum_value, f"{label}.maximum_position_m")
    if maximum <= minimum:
        raise _fail(
            "CLICK_PEN_GUIDE_RANGE_INVALID",
            f"{label} maximum must exceed minimum",
        )
    stiffness_default, damping_default, preload_default = spring_defaults
    return ClickPenLinearGuideSpec(
        joint_id=identifier(value.get("joint_id"), f"{label}.joint_id"),
        parent_body_id=identifier(
            value.get("parent_body_id"),
            f"{label}.parent_body_id",
        ),
        child_body_id=identifier(
            value.get("child_body_id"),
            f"{label}.child_body_id",
        ),
        origin_abs_m=_tuple3(value.get("origin_abs_m"), f"{label}.origin_abs_m"),
        axis_abs=_tuple3(value.get("axis_abs"), f"{label}.axis_abs", unit=True),
        minimum_position_m=minimum,
        maximum_position_m=maximum,
        spring_stiffness_n_m=_nonnegative(
            value.get("spring_stiffness_n_m", stiffness_default),
            f"{label}.spring_stiffness_n_m",
        ),
        damping_n_s_m=_nonnegative(
            value.get("damping_n_s_m", damping_default),
            f"{label}.damping_n_s_m",
        ),
        spring_preload_n=_nonnegative(
            value.get("spring_preload_n", preload_default),
            f"{label}.spring_preload_n",
        ),
    )


def click_pen_cam_latch_from_dict(
    value: Mapping[str, Any] | None,
) -> ClickPenCamLatchSpec:
    if not isinstance(value, Mapping):
        raise _fail(
            "CLICK_PEN_SPEC_REQUIRED",
            "click pen family spec must be an object",
        )
    if value.get("mechanism_kind") != "click_pen_cam_latch":
        raise _fail(
            "CLICK_PEN_KIND_INVALID",
            "mechanism_kind must be click_pen_cam_latch",
        )
    bodies_value = value.get("bodies")
    if (
        not isinstance(bodies_value, Sequence)
        or isinstance(bodies_value, (str, bytes))
        or len(bodies_value) != 4
        or not all(isinstance(item, Mapping) for item in bodies_value)
    ):
        raise _fail(
            "CLICK_PEN_BODIES_INVALID",
            "click pen requires exactly four bodies",
        )
    bodies = tuple(_body(item, index) for index, item in enumerate(bodies_value))
    body_ids = {body.body_id for body in bodies}
    if len(body_ids) != len(bodies):
        raise _fail("CLICK_PEN_BODY_DUPLICATE", "body IDs must be unique")

    button_value = value.get("button_prismatic_joint")
    refill_value = value.get("refill_prismatic_joint")
    cam_value = value.get("cam_cylindrical_joint")
    state_value = value.get("click_state_machine")
    if not all(
        isinstance(item, Mapping)
        for item in (button_value, refill_value, cam_value, state_value)
    ):
        raise _fail(
            "CLICK_PEN_INTERFACE_REQUIRED",
            "button, refill, cam, and click state declarations are required",
        )
    button = _linear_guide(
        button_value,
        "button_prismatic_joint",
        minimum_default=0.0,
        maximum_fields=("maximum_position_m", "max_position_m", "travel_m"),
        spring_defaults=(180.0, 2.0, 0.6),
    )
    refill = _linear_guide(
        refill_value,
        "refill_prismatic_joint",
        minimum_default=float(refill_value.get("retracted_position_m", 0.0)),
        maximum_fields=(
            "maximum_position_m",
            "max_position_m",
            "extended_position_m",
        ),
        spring_defaults=(520.0, 1.2, 1.8),
    )
    cam_minimum = finite(
        cam_value.get("minimum_position_m", 0.0),
        "cam_cylindrical_joint.minimum_position_m",
    )
    cam_maximum = finite(
        cam_value.get(
            "maximum_position_m",
            cam_value.get("axial_travel_m"),
        ),
        "cam_cylindrical_joint.maximum_position_m",
    )
    if cam_maximum <= cam_minimum:
        raise _fail(
            "CLICK_PEN_GUIDE_RANGE_INVALID",
            "cam axial maximum must exceed minimum",
        )
    cam = ClickPenCylindricalGuideSpec(
        joint_id=identifier(
            cam_value.get("joint_id"),
            "cam_cylindrical_joint.joint_id",
        ),
        parent_body_id=identifier(
            cam_value.get("parent_body_id"),
            "cam_cylindrical_joint.parent_body_id",
        ),
        child_body_id=identifier(
            cam_value.get("child_body_id"),
            "cam_cylindrical_joint.child_body_id",
        ),
        origin_abs_m=_tuple3(
            cam_value.get("origin_abs_m"),
            "cam_cylindrical_joint.origin_abs_m",
        ),
        axis_abs=_tuple3(
            cam_value.get("axis_abs"),
            "cam_cylindrical_joint.axis_abs",
            unit=True,
        ),
        minimum_position_m=cam_minimum,
        maximum_position_m=cam_maximum,
        index_angle_rad=_positive(
            cam_value.get("index_angle_rad"),
            "cam_cylindrical_joint.index_angle_rad",
        ),
    )
    for joint in (button, refill, cam):
        if {joint.parent_body_id, joint.child_body_id} - body_ids:
            raise _fail(
                "CLICK_PEN_GUIDE_BODY_UNKNOWN",
                f"joint {joint.joint_id!r} refers to an unknown body",
            )

    geometry_value = state_value.get("geometry_contract")
    if not isinstance(geometry_value, Mapping):
        raise _fail(
            "CLICK_PEN_GEOMETRY_REQUIRED",
            "click state requires a STEP geometry contract",
        )
    button_refs = geometry_value.get("button_contact_component_refs")
    if button_refs is None:
        button_refs = [geometry_value.get("button_contact_component_ref")]
    cam_button_refs = geometry_value.get("cam_button_face_component_refs")
    if cam_button_refs is None:
        cam_button_refs = [geometry_value.get("cam_button_face_component_ref")]
    geometry = ClickPenGeometrySpec(
        button_contact_component_refs=_component_refs(
            button_refs,
            "button_contact_component_refs",
        ),
        cam_button_face_component_refs=_component_refs(
            cam_button_refs,
            "cam_button_face_component_refs",
        ),
        cam_follower_component_ref=_component_ref(
            geometry_value.get("cam_follower_component_ref"),
            "cam_follower_component_ref",
        ),
        refill_carrier_component_ref=_component_ref(
            geometry_value.get("refill_carrier_component_ref"),
            "refill_carrier_component_ref",
        ),
        barrel_guide_component_refs=_component_refs(
            geometry_value.get("barrel_guide_component_refs"),
            "barrel_guide_component_refs",
        ),
        cam_lobe_component_refs=_component_refs(
            geometry_value.get("cam_lobe_component_refs"),
            "cam_lobe_component_refs",
        ),
        home_contact_tolerance_m=_positive(
            geometry_value.get("home_contact_tolerance_m"),
            "home_contact_tolerance_m",
        ),
        full_press_offset_abs_m=_tuple3(
            geometry_value.get("full_press_offset_abs_m"),
            "full_press_offset_abs_m",
        ),
        extended_latched_cam_offset_abs_m=_tuple3(
            geometry_value.get("extended_latched_cam_offset_abs_m"),
            "extended_latched_cam_offset_abs_m",
        ),
        extended_latched_refill_offset_abs_m=_tuple3(
            geometry_value.get("extended_latched_refill_offset_abs_m"),
            "extended_latched_refill_offset_abs_m",
        ),
    )
    states_value = state_value.get("states")
    if (
        not isinstance(states_value, Sequence)
        or isinstance(states_value, (str, bytes))
        or len(states_value) < 2
    ):
        raise _fail(
            "CLICK_PEN_STATES_INVALID",
            "click state machine needs at least two states",
        )
    states = tuple(
        identifier(item, f"click_state_machine.states[{index}]")
        for index, item in enumerate(states_value)
    )
    initial_state = identifier(
        state_value.get("initial_state"),
        "click_state_machine.initial_state",
    )
    if initial_state not in states:
        raise _fail(
            "CLICK_PEN_STATES_INVALID",
            "initial click state is not declared",
        )
    state = ClickPenStateSpec(
        states=states,
        initial_state=initial_state,
        first_click_target_m=_positive(
            state_value.get("first_click_target_m"),
            "click_state_machine.first_click_target_m",
        ),
        second_click_target_m=_nonnegative(
            state_value.get("second_click_target_m"),
            "click_state_machine.second_click_target_m",
        ),
        latch_contact_tolerance_m=_positive(
            state_value.get("latch_contact_tolerance_m"),
            "click_state_machine.latch_contact_tolerance_m",
        ),
        release_button_return_m=_positive(
            state_value.get("release_button_return_m"),
            "click_state_machine.release_button_return_m",
        ),
        geometry_contract=geometry,
    )
    input_body = identifier(value.get("input_body_id"), "input_body_id")
    output_body = identifier(value.get("output_body_id"), "output_body_id")
    if {input_body, output_body} - body_ids or input_body == output_body:
        raise _fail(
            "CLICK_PEN_INPUT_OUTPUT_INVALID",
            "input and output must be distinct known bodies",
        )
    spec = ClickPenCamLatchSpec(
        mechanism_kind="click_pen_cam_latch",
        mechanism_id=identifier(value.get("mechanism_id"), "mechanism_id"),
        bodies=bodies,
        button_prismatic_joint=button,
        refill_prismatic_joint=refill,
        cam_cylindrical_joint=cam,
        click_state_machine=state,
        input_body_id=input_body,
        output_body_id=output_body,
        nominal_press_force_n=_positive(
            value.get("nominal_press_force_n"),
            "nominal_press_force_n",
        ),
        max_support_drift_m=_positive(
            value.get("max_support_drift_m"),
            "max_support_drift_m",
        ),
        max_off_axis_rotation_rad=_positive(
            value.get("max_off_axis_rotation_rad"),
            "max_off_axis_rotation_rad",
        ),
        friction=_positive(value.get("friction", 0.20), "friction"),
    )
    fixed_body = spec.fixed_body_id
    if any(
        joint.parent_body_id != fixed_body
        for joint in (button, refill, cam)
    ):
        raise _fail(
            "CLICK_PEN_TOPOLOGY_INVALID",
            "all three guides must be supported by the fixed barrel",
        )
    if (
        button.child_body_id != input_body
        or refill.child_body_id != output_body
    ):
        raise _fail(
            "CLICK_PEN_TOPOLOGY_INVALID",
            "button and refill guides must bind the declared input and output",
        )
    if state.first_click_target_m > refill.maximum_position_m + 1.0e-12:
        raise _fail(
            "CLICK_PEN_STATE_RANGE_INVALID",
            "first click target exceeds refill guide travel",
        )
    owners = {
        **{ref: input_body for ref in geometry.button_contact_component_refs},
        **{
            ref: cam.child_body_id
            for ref in (
                *geometry.cam_button_face_component_refs,
                geometry.cam_follower_component_ref,
                *geometry.cam_lobe_component_refs,
            )
        },
        geometry.refill_carrier_component_ref: output_body,
        **{
            ref: fixed_body
            for ref in geometry.barrel_guide_component_refs
        },
    }
    wrong = [
        ref
        for ref, owner in owners.items()
        if not ref.startswith(f"{owner}/")
    ]
    if wrong:
        raise _fail(
            "CLICK_PEN_GEOMETRY_BODY_MISMATCH",
            "functional STEP component belongs to the wrong rigid body",
        )
    return spec
