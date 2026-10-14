from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .models import MechanismValidationError, finite, identifier, vector3


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _positive(value: Any, label: str) -> float:
    result = finite(value, label)
    if result <= 0.0:
        raise _fail("DETENT_SLIDER_RANGE_INVALID", f"{label} must be positive")
    return result


def _tuple3(value: Any, label: str, *, unit: bool = False) -> tuple[float, float, float]:
    return vector3(value, label, unit=unit, allow_zero=not unit)


def _component_ref(value: Any, label: str) -> str:
    result = str(value or "")
    body_id, separator, solid_id = result.partition("/")
    if separator != "/" or not body_id or not solid_id:
        raise _fail(
            "DETENT_SLIDER_COMPONENT_REF_INVALID",
            f"{label} must be body_id/solid_id",
        )
    identifier(body_id, f"{label}.body_id")
    identifier(solid_id, f"{label}.solid_id")
    return result


@dataclass(frozen=True)
class DetentBodySpec:
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
class LinearJointSpec:
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
class DetentGeometrySpec:
    rack_component_refs: tuple[str, ...]
    seat_component_refs: tuple[str, ...]
    tab_component_ref: str
    rack_tooth_centers_abs_m: tuple[tuple[float, float, float], ...]
    rack_tooth_size_m: tuple[float, float, float]
    seat_centers_abs_m: tuple[tuple[float, float, float], ...]
    seat_size_m: tuple[float, float, float]
    tab_home_center_abs_m: tuple[float, float, float]
    tab_size_m: tuple[float, float, float]
    pressed_offset_abs_m: tuple[float, float, float]
    contact_tolerance_m: float
    minimum_pressed_clearance_m: float


@dataclass(frozen=True)
class DetentInterfaceSpec:
    rack_body_id: str
    tab_body_id: str
    positions_m: tuple[float, ...]
    capture_tolerance_m: float
    release_press_m: float
    holding_force_n: float
    geometry_contract: DetentGeometrySpec


@dataclass(frozen=True)
class PrismaticDetentLatchSpec:
    mechanism_kind: str
    mechanism_id: str
    bodies: tuple[DetentBodySpec, ...]
    blade_prismatic_joint: LinearJointSpec
    latch_press_joint: LinearJointSpec
    detent_interface: DetentInterfaceSpec
    input_body_id: str
    output_body_id: str
    nominal_press_force_n: float
    nominal_slide_force_n: float
    adversarial_axial_force_n: float
    max_support_drift_m: float
    max_off_axis_rotation_rad: float
    friction: float

    @property
    def fixed_body_id(self) -> str:
        fixed = [body.body_id for body in self.bodies if body.fixed]
        if len(fixed) != 1:
            raise _fail(
                "DETENT_SLIDER_FIXED_BODY_INVALID",
                "prismatic detent latch requires exactly one fixed body",
            )
        return fixed[0]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def declared_input_wrench_direction_abs(
    family_spec: Mapping[str, Any],
) -> tuple[float, float, float] | None:
    """Unit wrench of the declared press-and-slide force split.

    Capsule ``input_ports[].direction_abs`` is this combined wrench, not the
    blade prismatic DOF.  Missing or non-finite split fields yield ``None``
    so channel binding fails closed.
    """

    blade_joint = family_spec.get("blade_prismatic_joint")
    latch_joint = family_spec.get("latch_press_joint")
    slide = (
        blade_joint.get("axis_abs")
        if isinstance(blade_joint, Mapping)
        else None
    )
    press = (
        latch_joint.get("axis_abs")
        if isinstance(latch_joint, Mapping)
        else None
    )
    slide_n = family_spec.get("nominal_slide_force_n")
    press_n = family_spec.get("nominal_press_force_n")
    if (
        not isinstance(slide, (list, tuple))
        or not isinstance(press, (list, tuple))
        or len(slide) != 3
        or len(press) != 3
        or isinstance(slide_n, bool)
        or isinstance(press_n, bool)
        or not isinstance(slide_n, (int, float))
        or not isinstance(press_n, (int, float))
        or not math.isfinite(float(slide_n))
        or not math.isfinite(float(press_n))
    ):
        return None
    value = tuple(
        float(slide_n) * float(a) + float(press_n) * float(b)
        for a, b in zip(slide, press)
    )
    length = math.sqrt(sum(item * item for item in value))
    if not math.isfinite(length) or length <= 0.0:
        return None
    return tuple(item / length for item in value)


def _body_from_dict(value: Mapping[str, Any], index: int) -> DetentBodySpec:
    body_id = identifier(value.get("body_id"), f"bodies[{index}].body_id")
    groups_value = value.get("group_ids")
    if (
        not isinstance(groups_value, Sequence)
        or isinstance(groups_value, (str, bytes))
        or not groups_value
    ):
        raise _fail(
            "DETENT_SLIDER_GROUPS_INVALID",
            f"bodies[{index}].group_ids must be a non-empty array",
        )
    groups = tuple(
        identifier(item, f"bodies[{index}].group_ids[{group_index}]")
        for group_index, item in enumerate(groups_value)
    )
    if len(set(groups)) != len(groups):
        raise _fail(
            "DETENT_SLIDER_GROUPS_INVALID",
            f"bodies[{index}].group_ids contains duplicates",
        )
    fixed = value.get("fixed")
    if not isinstance(fixed, bool):
        raise _fail(
            "DETENT_SLIDER_BODY_INVALID",
            f"bodies[{index}].fixed must be boolean",
        )
    mass = _positive(value.get("mass_kg"), f"bodies[{index}].mass_kg")
    size = _tuple3(value.get("size_m"), f"bodies[{index}].size_m")
    if any(item <= 0.0 for item in size):
        raise _fail(
            "DETENT_SLIDER_BODY_INVALID",
            f"bodies[{index}].size_m must be positive",
        )
    rotation_value = value.get(
        "rotation_abs_quat_wxyz",
        (1.0, 0.0, 0.0, 0.0),
    )
    if not isinstance(rotation_value, Sequence) or len(rotation_value) != 4:
        raise _fail(
            "DETENT_SLIDER_BODY_INVALID",
            f"bodies[{index}].rotation_abs_quat_wxyz must contain four numbers",
        )
    rotation = tuple(
        finite(item, f"bodies[{index}].rotation_abs_quat_wxyz[{component}]")
        for component, item in enumerate(rotation_value)
    )
    return DetentBodySpec(
        body_id=body_id,
        group_ids=groups,
        fixed=fixed,
        mass_kg=mass,
        center_abs_m=_tuple3(
            value.get("center_abs_m"),
            f"bodies[{index}].center_abs_m",
        ),
        size_m=size,
        rotation_abs_quat_wxyz=rotation,
    )


def _joint_from_dict(
    value: Mapping[str, Any],
    label: str,
    *,
    default_minimum_m: float,
    default_maximum_field: str,
) -> LinearJointSpec:
    minimum = finite(
        value.get(
            "min_position_m",
            value.get("minimum_position_m", default_minimum_m),
        ),
        f"{label}.min_position_m",
    )
    maximum = finite(
        value.get(
            "max_position_m",
            value.get(
                "maximum_position_m",
                value.get(default_maximum_field),
            ),
        ),
        f"{label}.max_position_m",
    )
    if maximum <= minimum:
        raise _fail(
            "DETENT_SLIDER_JOINT_RANGE_INVALID",
            f"{label} maximum must exceed minimum",
        )
    return LinearJointSpec(
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
        spring_stiffness_n_m=finite(
            value.get("spring_stiffness_n_m", 0.0),
            f"{label}.spring_stiffness_n_m",
        ),
        damping_n_s_m=finite(
            value.get("damping_n_s_m", 0.0),
            f"{label}.damping_n_s_m",
        ),
        spring_preload_n=finite(
            value.get("spring_preload_n", 0.0),
            f"{label}.spring_preload_n",
        ),
    )


def prismatic_detent_latch_from_dict(
    value: Mapping[str, Any] | None,
) -> PrismaticDetentLatchSpec:
    if not isinstance(value, Mapping):
        raise _fail(
            "DETENT_SLIDER_SPEC_REQUIRED",
            "prismatic detent latch family spec must be an object",
        )
    if value.get("mechanism_kind") != "prismatic_detent_latch":
        raise _fail(
            "DETENT_SLIDER_KIND_INVALID",
            "mechanism_kind must be prismatic_detent_latch",
        )
    bodies_value = value.get("bodies")
    if (
        not isinstance(bodies_value, Sequence)
        or isinstance(bodies_value, (str, bytes))
        or len(bodies_value) < 3
        or not all(isinstance(item, Mapping) for item in bodies_value)
    ):
        raise _fail(
            "DETENT_SLIDER_BODIES_INVALID",
            "prismatic detent latch requires at least three bodies",
        )
    bodies = tuple(
        _body_from_dict(item, index)
        for index, item in enumerate(bodies_value)
    )
    body_ids = {body.body_id for body in bodies}
    if len(body_ids) != len(bodies):
        raise _fail("DETENT_SLIDER_BODY_DUPLICATE", "body IDs must be unique")

    blade_value = value.get("blade_prismatic_joint")
    latch_value = value.get("latch_press_joint")
    detent_value = value.get("detent_interface")
    if not all(
        isinstance(item, Mapping)
        for item in (blade_value, latch_value, detent_value)
    ):
        raise _fail(
            "DETENT_SLIDER_INTERFACE_REQUIRED",
            "blade, latch and detent interface declarations are required",
        )
    blade = _joint_from_dict(
        blade_value,
        "blade_prismatic_joint",
        default_minimum_m=0.0,
        default_maximum_field="travel_m",
    )
    latch = _joint_from_dict(
        latch_value,
        "latch_press_joint",
        default_minimum_m=0.0,
        default_maximum_field="travel_m",
    )
    for joint in (blade, latch):
        if {joint.parent_body_id, joint.child_body_id} - body_ids:
            raise _fail(
                "DETENT_SLIDER_JOINT_BODY_UNKNOWN",
                f"joint {joint.joint_id!r} refers to an unknown body",
            )
        if joint.spring_preload_n < 0.0:
            raise _fail(
                "DETENT_SLIDER_JOINT_RANGE_INVALID",
                f"joint {joint.joint_id!r} spring preload cannot be negative",
            )

    positions_value = detent_value.get("positions_m")
    if (
        not isinstance(positions_value, Sequence)
        or isinstance(positions_value, (str, bytes))
        or len(positions_value) < 2
    ):
        raise _fail(
            "DETENT_SLIDER_POSITIONS_INVALID",
            "detent positions must contain at least two entries",
        )
    positions = tuple(
        finite(item, f"detent_interface.positions_m[{index}]")
        for index, item in enumerate(positions_value)
    )
    if tuple(sorted(set(positions))) != positions:
        raise _fail(
            "DETENT_SLIDER_POSITIONS_INVALID",
            "detent positions must be unique and strictly increasing",
        )
    if (
        positions[0] < blade.minimum_position_m - 1.0e-12
        or positions[-1] > blade.maximum_position_m + 1.0e-12
    ):
        raise _fail(
            "DETENT_SLIDER_POSITIONS_INVALID",
            "detent positions must lie inside blade travel",
        )
    geometry_value = detent_value.get("geometry_contract")
    if not isinstance(geometry_value, Mapping):
        raise _fail(
            "DETENT_SLIDER_GEOMETRY_REQUIRED",
            "detent interface requires a STEP geometry contract",
        )
    rack_refs_value = geometry_value.get("rack_component_refs")
    centers_value = geometry_value.get("rack_tooth_centers_abs_m")
    seat_refs_value = geometry_value.get("seat_component_refs") or ()
    seat_centers_value = geometry_value.get("seat_centers_abs_m") or ()
    if (
        not isinstance(rack_refs_value, Sequence)
        or isinstance(rack_refs_value, (str, bytes))
        or not rack_refs_value
        or not isinstance(centers_value, Sequence)
        or len(centers_value) != len(rack_refs_value)
    ):
        raise _fail(
            "DETENT_SLIDER_GEOMETRY_INVALID",
            "rack refs and tooth centers must be non-empty equal-length arrays",
        )
    rack_refs = tuple(
        _component_ref(item, f"rack_component_refs[{index}]")
        for index, item in enumerate(rack_refs_value)
    )
    if (
        not isinstance(seat_refs_value, Sequence)
        or isinstance(seat_refs_value, (str, bytes))
        or not isinstance(seat_centers_value, Sequence)
        or len(seat_centers_value) != len(seat_refs_value)
        or (seat_refs_value and len(seat_refs_value) != len(positions))
    ):
        raise _fail(
            "DETENT_SLIDER_GEOMETRY_INVALID",
            "seat refs and centers must be equal-length arrays with one seat per detent",
        )
    seat_refs = tuple(
        _component_ref(item, f"seat_component_refs[{index}]")
        for index, item in enumerate(seat_refs_value)
    )
    seat_centers = tuple(
        _tuple3(item, f"seat_centers_abs_m[{index}]")
        for index, item in enumerate(seat_centers_value)
    )
    centers = tuple(
        _tuple3(item, f"rack_tooth_centers_abs_m[{index}]")
        for index, item in enumerate(centers_value)
    )
    geometry = DetentGeometrySpec(
        rack_component_refs=rack_refs,
        seat_component_refs=seat_refs,
        tab_component_ref=_component_ref(
            geometry_value.get("tab_component_ref"),
            "tab_component_ref",
        ),
        rack_tooth_centers_abs_m=centers,
        rack_tooth_size_m=_tuple3(
            geometry_value.get("rack_tooth_size_m"),
            "rack_tooth_size_m",
        ),
        seat_centers_abs_m=seat_centers,
        seat_size_m=_tuple3(
            geometry_value.get("seat_size_m", (0.0, 0.0, 0.0)),
            "seat_size_m",
        ),
        tab_home_center_abs_m=_tuple3(
            geometry_value.get("tab_home_center_abs_m"),
            "tab_home_center_abs_m",
        ),
        tab_size_m=_tuple3(
            geometry_value.get("tab_size_m"),
            "tab_size_m",
        ),
        pressed_offset_abs_m=_tuple3(
            geometry_value.get("pressed_offset_abs_m"),
            "pressed_offset_abs_m",
        ),
        contact_tolerance_m=_positive(
            geometry_value.get("contact_tolerance_m"),
            "contact_tolerance_m",
        ),
        minimum_pressed_clearance_m=_positive(
            geometry_value.get("minimum_pressed_clearance_m"),
            "minimum_pressed_clearance_m",
        ),
    )
    if (
        any(item <= 0.0 for item in geometry.rack_tooth_size_m)
        or any(item <= 0.0 for item in geometry.tab_size_m)
        or (
            geometry.seat_component_refs
            and any(item <= 0.0 for item in geometry.seat_size_m)
        )
    ):
        raise _fail(
            "DETENT_SLIDER_GEOMETRY_INVALID",
            "rack tooth and tab sizes must be positive",
        )
    rack_body = identifier(
        detent_value.get("rack_body_id"),
        "detent_interface.rack_body_id",
    )
    tab_body = identifier(
        detent_value.get("tab_body_id"),
        "detent_interface.tab_body_id",
    )
    if {rack_body, tab_body} - body_ids:
        raise _fail(
            "DETENT_SLIDER_INTERFACE_BODY_UNKNOWN",
            "detent interface refers to an unknown body",
        )
    if any(
        not ref.startswith(f"{rack_body}/")
        for ref in (*rack_refs, *seat_refs)
    ):
        raise _fail(
            "DETENT_SLIDER_GEOMETRY_BODY_MISMATCH",
            "every rack component must belong to rack_body_id",
        )
    if not geometry.tab_component_ref.startswith(f"{tab_body}/"):
        raise _fail(
            "DETENT_SLIDER_GEOMETRY_BODY_MISMATCH",
            "tab component must belong to tab_body_id",
        )
    detent = DetentInterfaceSpec(
        rack_body_id=rack_body,
        tab_body_id=tab_body,
        positions_m=positions,
        capture_tolerance_m=_positive(
            detent_value.get("capture_tolerance_m"),
            "capture_tolerance_m",
        ),
        release_press_m=_positive(
            detent_value.get("release_press_m"),
            "release_press_m",
        ),
        holding_force_n=_positive(
            detent_value.get("holding_force_n"),
            "holding_force_n",
        ),
        geometry_contract=geometry,
    )
    if detent.release_press_m > latch.maximum_position_m:
        raise _fail(
            "DETENT_SLIDER_RELEASE_INVALID",
            "release press exceeds latch travel",
        )
    pressed_distance = sum(
        component * axis
        for component, axis in zip(
            geometry.pressed_offset_abs_m,
            latch.axis_abs,
        )
    )
    pressed_transverse = tuple(
        component - pressed_distance * axis
        for component, axis in zip(
            geometry.pressed_offset_abs_m,
            latch.axis_abs,
        )
    )
    if (
        pressed_distance < detent.release_press_m
        or pressed_distance > latch.maximum_position_m + 1.0e-12
        or sum(item * item for item in pressed_transverse)
        > geometry.contact_tolerance_m**2
    ):
        raise _fail(
            "DETENT_SLIDER_RELEASE_GEOMETRY_INVALID",
            "pressed offset must follow the latch axis through the release range",
        )
    input_body = identifier(value.get("input_body_id"), "input_body_id")
    output_body = identifier(value.get("output_body_id"), "output_body_id")
    if {input_body, output_body} - body_ids or input_body == output_body:
        raise _fail(
            "DETENT_SLIDER_INPUT_OUTPUT_INVALID",
            "input and output must be distinct known bodies",
        )
    spec = PrismaticDetentLatchSpec(
        mechanism_kind="prismatic_detent_latch",
        mechanism_id=identifier(value.get("mechanism_id"), "mechanism_id"),
        bodies=bodies,
        blade_prismatic_joint=blade,
        latch_press_joint=latch,
        detent_interface=detent,
        input_body_id=input_body,
        output_body_id=output_body,
        nominal_press_force_n=_positive(
            value.get("nominal_press_force_n"),
            "nominal_press_force_n",
        ),
        nominal_slide_force_n=_positive(
            value.get("nominal_slide_force_n"),
            "nominal_slide_force_n",
        ),
        adversarial_axial_force_n=_positive(
            value.get("adversarial_axial_force_n"),
            "adversarial_axial_force_n",
        ),
        max_support_drift_m=_positive(
            value.get("max_support_drift_m"),
            "max_support_drift_m",
        ),
        max_off_axis_rotation_rad=_positive(
            value.get("max_off_axis_rotation_rad"),
            "max_off_axis_rotation_rad",
        ),
        friction=_positive(value.get("friction", 0.32), "friction"),
    )
    spec.fixed_body_id
    if spec.adversarial_axial_force_n > detent.holding_force_n:
        raise _fail(
            "DETENT_SLIDER_HOLD_LOAD_INVALID",
            "adversarial axial force exceeds the declared detent holding force",
        )
    if blade.parent_body_id != spec.fixed_body_id:
        raise _fail(
            "DETENT_SLIDER_TOPOLOGY_INVALID",
            "blade guide parent must be the fixed body",
        )
    if (
        blade.child_body_id != output_body
        or latch.parent_body_id != output_body
        or latch.child_body_id != input_body
    ):
        raise _fail(
            "DETENT_SLIDER_TOPOLOGY_INVALID",
            "input latch must ride on the output blade carriage",
        )
    slide_axis = blade.axis_abs
    rack_centers = sorted(
        sum(
            coordinate * axis
            for coordinate, axis in zip(center, slide_axis)
        )
        for center in geometry.rack_tooth_centers_abs_m
    )
    tab_home = sum(
        coordinate * axis
        for coordinate, axis in zip(
            geometry.tab_home_center_abs_m,
            slide_axis,
        )
    )
    rack_half = 0.5 * sum(
        abs(axis) * size
        for axis, size in zip(slide_axis, geometry.rack_tooth_size_m)
    )
    tab_half = 0.5 * sum(
        abs(axis) * size
        for axis, size in zip(slide_axis, geometry.tab_size_m)
    )
    face_tolerance = max(
        geometry.contact_tolerance_m,
        1.0e-9,
    )
    for index, position in enumerate(detent.positions_m):
        tab_center = tab_home + position
        left_faces = [
            center + rack_half
            for center in rack_centers
            if center < tab_center
        ]
        right_faces = [
            center - rack_half
            for center in rack_centers
            if center > tab_center
        ]
        if not left_faces or not right_faces:
            raise _fail(
                "DETENT_SLIDER_BIDIRECTIONAL_CAPTURE_REQUIRED",
                f"detent position {index} is not bounded on both slide directions",
            )
        left_gap = (tab_center - tab_half) - max(left_faces)
        right_gap = min(right_faces) - (tab_center + tab_half)
        if (
            abs(left_gap) > face_tolerance
            or abs(right_gap) > face_tolerance
        ):
            raise _fail(
                "DETENT_SLIDER_BIDIRECTIONAL_CAPTURE_REQUIRED",
                f"detent position {index} lacks two face-contact stops",
            )
    if geometry.seat_component_refs:
        seat_half_slide = 0.5 * sum(
            abs(axis) * size
            for axis, size in zip(slide_axis, geometry.seat_size_m)
        )
        tab_half_press = 0.5 * sum(
            abs(axis) * size
            for axis, size in zip(
                latch.axis_abs,
                geometry.tab_size_m,
            )
        )
        seat_half_press = 0.5 * sum(
            abs(axis) * size
            for axis, size in zip(
                latch.axis_abs,
                geometry.seat_size_m,
            )
        )
        if seat_half_slide + face_tolerance < tab_half:
            raise _fail(
                "DETENT_SLIDER_SEAT_GEOMETRY_INVALID",
                "each unloaded seat must span the tab along the slide axis",
            )
        for index, (position, center) in enumerate(
            zip(detent.positions_m, geometry.seat_centers_abs_m)
        ):
            tab_center = tuple(
                coordinate + position * axis
                for coordinate, axis in zip(
                    geometry.tab_home_center_abs_m,
                    slide_axis,
                )
            )
            delta = tuple(
                seat - tab
                for seat, tab in zip(center, tab_center)
            )
            slide_offset = sum(
                component * axis
                for component, axis in zip(delta, slide_axis)
            )
            press_offset = sum(
                component * axis
                for component, axis in zip(delta, latch.axis_abs)
            )
            initial_gap = (
                -press_offset
                - tab_half_press
                - seat_half_press
            )
            pressed_clearance = initial_gap + pressed_distance
            if (
                abs(slide_offset) > face_tolerance
                or press_offset >= 0.0
                or abs(initial_gap) > face_tolerance
                or pressed_clearance
                < geometry.minimum_pressed_clearance_m
            ):
                raise _fail(
                    "DETENT_SLIDER_SEAT_GEOMETRY_INVALID",
                    f"detent position {index} lacks a separating unloaded seat",
                )
    return spec
