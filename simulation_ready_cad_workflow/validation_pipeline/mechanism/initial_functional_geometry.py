from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from .adapters import AdapterSpec
from .conditional_interface import (
    AT_OR_BELOW,
    AxialClutchGeometry,
    AxialClutchPose,
    AxialGearPose,
    AxialSlidingSpurGearGeometry,
    CoplanarPulleyTransferGeometry,
    InterfaceGeometryObservation,
    RadialReachGeometry,
    RadialReachPose,
    RackPinionGeometry,
    RackPinionPose,
    RightAngleBevelGearGeometry,
    assess_initial_functional_geometry,
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
)
from .io import canonical_sha256, ensure_inside, read_json, sha256_file
from .models import MechanismPlan, MechanismValidationError


SCHEMA_VERSION = "initial_functional_geometry_preflight_v1"

_STEP_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[+-]?\d+)?"


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _add(
    left: Sequence[float],
    right: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


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
    length = _norm(value)
    if length <= 1.0e-12:
        raise _fail(
            "INITIAL_FUNCTIONAL_GEOMETRY_AXIS_INVALID",
            f"{label} cannot be zero",
        )
    return tuple(float(item) / length for item in value)


def _rotation_matrix(
    quaternion_wxyz: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    if len(quaternion_wxyz) != 4:
        raise _fail(
            "INITIAL_FUNCTIONAL_GEOMETRY_ROTATION_INVALID",
            "body quaternion must have four values",
        )
    w_value, x_value, y_value, z_value = (
        float(item) for item in quaternion_wxyz
    )
    magnitude = math.sqrt(
        w_value * w_value
        + x_value * x_value
        + y_value * y_value
        + z_value * z_value
    )
    if magnitude <= 1.0e-12:
        raise _fail(
            "INITIAL_FUNCTIONAL_GEOMETRY_ROTATION_INVALID",
            "body quaternion cannot be zero",
        )
    w_value, x_value, y_value, z_value = (
        item / magnitude
        for item in (w_value, x_value, y_value, z_value)
    )
    return (
        (
            1.0 - 2.0 * (y_value * y_value + z_value * z_value),
            2.0 * (x_value * y_value - z_value * w_value),
            2.0 * (x_value * z_value + y_value * w_value),
        ),
        (
            2.0 * (x_value * y_value + z_value * w_value),
            1.0 - 2.0 * (x_value * x_value + z_value * z_value),
            2.0 * (y_value * z_value - x_value * w_value),
        ),
        (
            2.0 * (x_value * z_value - y_value * w_value),
            2.0 * (y_value * z_value + x_value * w_value),
            1.0 - 2.0 * (x_value * x_value + y_value * y_value),
        ),
    )


def _world_vector_to_local(
    value: Sequence[float],
    rotation_abs_from_body: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    return tuple(
        sum(
            float(rotation_abs_from_body[row][column]) * float(value[row])
            for row in range(3)
        )
        for column in range(3)
    )


def _world_point_to_local(
    point_abs_m: Sequence[float],
    *,
    center_abs_m: Sequence[float],
    rotation_abs_from_body: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    return _world_vector_to_local(
        _sub(point_abs_m, center_abs_m),
        rotation_abs_from_body,
    )


def _component_vertices(
    component: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[tuple[float, float, float], ...]:
    relative = component.get("mesh")
    expected_sha256 = component.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha256, str):
        raise _fail(
            "INITIAL_FUNCTIONAL_COMPONENT_BINDING_MISSING",
            "functional-interface component has no hash-bound STEP mesh",
        )
    path = ensure_inside(workspace, relative)
    if (
        not path.is_file()
        or path.is_symlink()
        or sha256_file(path) != expected_sha256
    ):
        raise _fail(
            "INITIAL_FUNCTIONAL_COMPONENT_BINDING_CHANGED",
            f"functional-interface STEP component changed: {relative}",
        )
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) == 4:
                vertices.append(
                    (
                        float(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                    )
                )
    if not vertices:
        raise _fail(
            "INITIAL_FUNCTIONAL_COMPONENT_MESH_EMPTY",
            f"functional-interface STEP component has no vertices: {relative}",
        )
    return tuple(vertices)


def _body_manifest(
    collision_geometry: Mapping[str, Any],
    body_id: str,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in collision_geometry.get("bodies") or ()
        if isinstance(item, Mapping) and item.get("body_id") == body_id
    ]
    if len(matches) != 1:
        raise _fail(
            "INITIAL_FUNCTIONAL_BODY_GEOMETRY_MISSING",
            f"STEP collision geometry does not contain exactly one body {body_id!r}",
        )
    return matches[0]


def _matrix_vector(
    matrix: Sequence[Sequence[float]],
    value: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        sum(float(matrix[row][column]) * float(value[column]) for column in range(3))
        for row in range(3)
    )


def _component_axis_records(
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
    body_id: str,
    shaft_origin_abs_m: Sequence[float],
    shaft_axis_abs: Sequence[float],
) -> tuple[dict[str, Any], ...]:
    body = _body_manifest(collision_geometry, body_id)
    center = body.get("initial_center_abs_m")
    quaternion = body.get("initial_rotation_abs_quat_wxyz")
    if not isinstance(center, list) or not isinstance(quaternion, list):
        raise _fail(
            "INITIAL_FUNCTIONAL_BODY_POSE_MISSING",
            f"body {body_id!r} has no initial STEP pose",
        )
    rotation = _rotation_matrix(quaternion)
    shaft_origin_local = _world_point_to_local(
        shaft_origin_abs_m,
        center_abs_m=center,
        rotation_abs_from_body=rotation,
    )
    shaft_axis_local = _unit(
        _world_vector_to_local(shaft_axis_abs, rotation),
        f"{body_id}.shaft_axis_local",
    )
    records: list[dict[str, Any]] = []
    for component in body.get("components") or ():
        if not isinstance(component, Mapping):
            continue
        vertices = _component_vertices(component, workspace=workspace)
        axial_values: list[float] = []
        radial_values: list[float] = []
        local_deltas: list[tuple[float, float, float]] = []
        world_vertices: list[tuple[float, float, float]] = []
        for vertex in vertices:
            delta = _sub(vertex, shaft_origin_local)
            axial = _dot(delta, shaft_axis_local)
            radial = _sub(delta, _scale(shaft_axis_local, axial))
            axial_values.append(axial)
            radial_values.append(_norm(radial))
            local_deltas.append(delta)
            world_vertices.append(
                _add(center, _matrix_vector(rotation, vertex))
            )
        component_center_delta = tuple(
            sum(item[index] for item in local_deltas)
            / len(local_deltas)
            for index in range(3)
        )
        component_axial_center = _dot(
            component_center_delta,
            shaft_axis_local,
        )
        component_radial_center = _sub(
            component_center_delta,
            _scale(shaft_axis_local, component_axial_center),
        )
        records.append(
            {
                "body_id": body_id,
                "solid_id": str(component.get("solid_id") or ""),
                "component_ref": (
                    f"{body_id}/{str(component.get('solid_id') or '')}"
                ),
                "sha256": str(component.get("sha256") or ""),
                "axial_min_m": min(axial_values),
                "axial_max_m": max(axial_values),
                "axial_center_m": component_axial_center,
                "radial_min_m": min(radial_values),
                "radial_max_m": max(radial_values),
                "radial_center_m": _norm(component_radial_center),
                "world_aabb_min_m": [
                    min(item[index] for item in world_vertices)
                    for index in range(3)
                ],
                "world_aabb_max_m": [
                    max(item[index] for item in world_vertices)
                    for index in range(3)
                ],
            }
        )
    return tuple(records)


def _right_angle_tooth_band(
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
    body_id: str,
    shaft_origin_abs_m: Sequence[float],
    shaft_axis_abs: Sequence[float],
    expected_teeth: int,
    expected_pitch_radius_m: float,
    expected_face_width_m: float,
) -> dict[str, Any]:
    """Bind a finite crown/bevel tooth ring to exact STEP components.

    A right-angle gear's rectangular tooth AABBs change with angular
    position, so the parallel-axis clock detector cannot group them by equal
    radial extrema.  The component centre remains on the declared pitch ring
    regardless of tooth angle.  Bind only components whose STEP-derived
    centre, face width, and axial layer all agree with that ring, then require
    the exact declared tooth count.
    """

    if expected_teeth <= 0:
        raise _fail(
            "MULTI_OUTPUT_STEP_TOOTH_COUNT_INVALID",
            f"body {body_id!r} must declare a positive tooth count",
        )
    if expected_pitch_radius_m <= 0.0 or expected_face_width_m <= 0.0:
        raise _fail(
            "MULTI_OUTPUT_STEP_TOOTH_GEOMETRY_INVALID",
            f"body {body_id!r} must declare positive pitch radius and face width",
        )
    records = _component_axis_records(
        collision_geometry,
        workspace=workspace,
        body_id=body_id,
        shaft_origin_abs_m=shaft_origin_abs_m,
        shaft_axis_abs=shaft_axis_abs,
    )
    radial_tolerance = max(
        0.2 * expected_pitch_radius_m,
        0.5 * expected_face_width_m,
        0.001,
    )
    axial_center_tolerance = max(
        0.75 * expected_face_width_m,
        0.002,
    )
    face_width_tolerance = max(
        0.35 * expected_face_width_m,
        0.001,
    )
    teeth = [
        item
        for item in records
        if (
            abs(
                float(item["radial_center_m"])
                - expected_pitch_radius_m
            )
            <= radial_tolerance
            and abs(float(item["axial_center_m"]))
            <= axial_center_tolerance
            and abs(
                (
                    float(item["axial_max_m"])
                    - float(item["axial_min_m"])
                )
                - expected_face_width_m
            )
            <= face_width_tolerance
        )
    ]
    if len(teeth) != expected_teeth:
        raise _fail(
            "MULTI_OUTPUT_STEP_TOOTH_BAND_AMBIGUOUS",
            f"body {body_id!r} has {len(teeth)} STEP components on the "
            f"declared pitch ring; expected {expected_teeth}",
        )
    teeth.sort(
        key=lambda item: int(str(item["solid_id"]).split("_")[-1])
    )
    measured_pitch_radius = sum(
        float(item["radial_center_m"]) for item in teeth
    ) / len(teeth)
    if (
        abs(measured_pitch_radius - expected_pitch_radius_m)
        > radial_tolerance
    ):
        raise _fail(
            "MULTI_OUTPUT_STEP_PITCH_RING_MISMATCH",
            f"body {body_id!r} STEP pitch ring does not match its declaration",
        )
    radial_minimum = min(
        float(item["radial_min_m"]) for item in teeth
    )
    radial_maximum = max(
        float(item["radial_max_m"]) for item in teeth
    )
    axial_minimum = min(
        float(item["axial_min_m"]) for item in teeth
    )
    axial_maximum = max(
        float(item["axial_max_m"]) for item in teeth
    )
    tooth_refs = {
        str(item["component_ref"]) for item in teeth
    }
    maximum_local_axial_width = max(
        0.006,
        6.0 * (axial_maximum - axial_minimum),
    )
    local_collision_records = list(teeth)
    for item in records:
        if str(item["component_ref"]) in tooth_refs:
            continue
        candidate_axial_width = (
            float(item["axial_max_m"]) - float(item["axial_min_m"])
        )
        axial_overlap = min(
            axial_maximum,
            float(item["axial_max_m"]),
        ) - max(
            axial_minimum,
            float(item["axial_min_m"]),
        )
        if not (
            candidate_axial_width <= maximum_local_axial_width
            and axial_overlap > 0.0
            and float(item["radial_max_m"]) >= radial_minimum
            and float(item["radial_min_m"]) <= radial_maximum
            and float(item["radial_max_m"])
            <= radial_maximum + radial_tolerance
        ):
            continue
        local_collision_records.append(item)
    local_collision_records.sort(
        key=lambda item: int(str(item["solid_id"]).split("_")[-1])
    )
    return {
        "body_id": body_id,
        "expected_teeth": expected_teeth,
        "bound_component_count": len(teeth),
        "component_refs": [
            str(item["component_ref"]) for item in teeth
        ],
        "local_collision_component_refs": [
            str(item["component_ref"])
            for item in local_collision_records
        ],
        "component_inventory_sha256": canonical_sha256(
            [
                {
                    "component_ref": str(item["component_ref"]),
                    "component_sha256": str(item["sha256"]),
                }
                for item in teeth
            ]
        ),
        "local_collision_component_inventory_sha256": canonical_sha256(
            [
                {
                    "component_ref": str(item["component_ref"]),
                    "component_sha256": str(item["sha256"]),
                }
                for item in local_collision_records
            ]
        ),
        "shaft_origin_abs_m": [
            float(item) for item in shaft_origin_abs_m
        ],
        "shaft_axis_abs": [float(item) for item in shaft_axis_abs],
        "radial_min_m": radial_minimum,
        "radial_max_m": radial_maximum,
        "measured_pitch_radius_m": measured_pitch_radius,
        "declared_pitch_radius_m": expected_pitch_radius_m,
        "axial_min_relative_to_shaft_m": axial_minimum,
        "axial_max_relative_to_shaft_m": axial_maximum,
        "face_width_m": axial_maximum - axial_minimum,
    }


def _aabb_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    squared = 0.0
    for left_min, left_max, right_min, right_max in zip(
        left["world_aabb_min_m"],
        left["world_aabb_max_m"],
        right["world_aabb_min_m"],
        right["world_aabb_max_m"],
    ):
        separation = max(
            float(right_min) - float(left_max),
            float(left_min) - float(right_max),
            0.0,
        )
        squared += separation * separation
    return math.sqrt(squared)


def _nearest_component_pair(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any], float]:
    if not left or not right:
        raise _fail(
            "INITIAL_FUNCTIONAL_COMPONENT_BINDING_MISSING",
            "functional interface has no STEP components to bind",
        )
    ranked = sorted(
        (
            (
                _aabb_distance(left_item, right_item),
                str(left_item["component_ref"]),
                str(right_item["component_ref"]),
                left_item,
                right_item,
            )
            for left_item in left
            for right_item in right
        ),
        key=lambda item: item[:3],
    )
    distance, _, _, left_item, right_item = ranked[0]
    return left_item, right_item, float(distance)


def _radial_band_candidates(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    tolerance = 5.0e-5
    groups: dict[tuple[int, int, int, int], list[Mapping[str, Any]]] = {}
    for item in records:
        radial_thickness = (
            float(item["radial_max_m"]) - float(item["radial_min_m"])
        )
        axial_width = (
            float(item["axial_max_m"]) - float(item["axial_min_m"])
        )
        if not (
            float(item["radial_max_m"]) > 0.002
            and 0.0002 <= radial_thickness <= 0.006
            and 0.0003 <= axial_width <= 0.0035
        ):
            continue
        key = tuple(
            int(round(float(item[field]) / tolerance))
            for field in (
                "radial_min_m",
                "radial_max_m",
                "axial_min_m",
                "axial_max_m",
            )
        )
        groups.setdefault(key, []).append(item)
    by_count: dict[int, list[tuple[Mapping[str, Any], ...]]] = {}
    for values in groups.values():
        if len(values) >= 4:
            by_count.setdefault(len(values), []).append(tuple(values))
    return {
        count: candidates[0]
        for count, candidates in by_count.items()
        if len(candidates) == 1
    }


def _band_payload(
    *,
    body_id: str,
    shaft_origin_abs_m: Sequence[float],
    shaft_axis_abs: Sequence[float],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    radial_minimum = min(float(item["radial_min_m"]) for item in records)
    radial_maximum = max(float(item["radial_max_m"]) for item in records)
    axial_minimum = min(float(item["axial_min_m"]) for item in records)
    axial_maximum = max(float(item["axial_max_m"]) for item in records)
    # The rectangular engineering proxy retains 0.08 mm tooth-tip clearance.
    # Recover module from the measured tip radius and exact bound tooth count;
    # this also remains valid for small pinions whose root radius is clamped.
    inferred_module = (
        radial_maximum + 0.00008
    ) / (0.5 * len(records) + 1.0)
    if inferred_module <= 0.0:
        raise _fail(
            "CLOCK_STEP_TOOTH_MODULE_INVALID",
            f"body {body_id!r} has no positive STEP-derived tooth module",
        )
    return {
        "body_id": body_id,
        "expected_teeth": len(records),
        "bound_component_count": len(records),
        "component_refs": [
            str(item["component_ref"]) for item in records
        ],
        "component_inventory_sha256": canonical_sha256(
            [
                {
                    "component_ref": str(item["component_ref"]),
                    "component_sha256": str(item["sha256"]),
                }
                for item in records
            ]
        ),
        "shaft_origin_abs_m": [
            float(item) for item in shaft_origin_abs_m
        ],
        "shaft_axis_abs": [float(item) for item in shaft_axis_abs],
        "radial_min_m": radial_minimum,
        "radial_max_m": radial_maximum,
        "inferred_module_m": inferred_module,
        "inferred_pitch_radius_m": (
            0.5 * len(records) * inferred_module
        ),
        "axial_min_relative_to_shaft_m": axial_minimum,
        "axial_max_relative_to_shaft_m": axial_maximum,
        "face_width_m": axial_maximum - axial_minimum,
    }


def _clock_tooth_band(
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
    body_id: str,
    shaft_origin_abs_m: Sequence[float],
    shaft_axis_abs: Sequence[float],
    expected_teeth: int,
) -> dict[str, Any]:
    body = _body_manifest(collision_geometry, body_id)
    center = body.get("initial_center_abs_m")
    quaternion = body.get("initial_rotation_abs_quat_wxyz")
    if not isinstance(center, list) or not isinstance(quaternion, list):
        raise _fail(
            "INITIAL_FUNCTIONAL_BODY_POSE_MISSING",
            f"body {body_id!r} has no initial STEP pose",
        )
    rotation = _rotation_matrix(quaternion)
    shaft_origin_local = _world_point_to_local(
        shaft_origin_abs_m,
        center_abs_m=center,
        rotation_abs_from_body=rotation,
    )
    shaft_axis_local = _unit(
        _world_vector_to_local(shaft_axis_abs, rotation),
        f"{body_id}.shaft_axis_local",
    )

    all_records: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for component in body.get("components") or ():
        if not isinstance(component, Mapping):
            continue
        vertices = _component_vertices(component, workspace=workspace)
        axial_values: list[float] = []
        radial_values: list[float] = []
        for vertex in vertices:
            delta = _sub(vertex, shaft_origin_local)
            axial = _dot(delta, shaft_axis_local)
            radial = _sub(delta, _scale(shaft_axis_local, axial))
            axial_values.append(axial)
            radial_values.append(_norm(radial))
        radial_minimum = min(radial_values)
        radial_maximum = max(radial_values)
        axial_minimum = min(axial_values)
        axial_maximum = max(axial_values)
        radial_thickness = radial_maximum - radial_minimum
        axial_width = axial_maximum - axial_minimum
        record = {
            "solid_id": str(component.get("solid_id") or ""),
            "sha256": str(component.get("sha256") or ""),
            "radial_min_m": radial_minimum,
            "radial_max_m": radial_maximum,
            "axial_min_m": axial_minimum,
            "axial_max_m": axial_maximum,
        }
        all_records.append(record)
        if not (
            radial_maximum > 0.002
            and 0.0002 <= radial_thickness <= 0.006
            and 0.0003 <= axial_width <= 0.0035
        ):
            continue
        records.append(record)

    tolerance = 5.0e-5
    groups: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for item in records:
        key = tuple(
            int(round(float(item[field]) / tolerance))
            for field in (
                "radial_min_m",
                "radial_max_m",
                "axial_min_m",
                "axial_max_m",
            )
        )
        groups.setdefault(key, []).append(item)
    candidates = [
        values
        for values in groups.values()
        if len(values) == expected_teeth
    ]
    if len(candidates) != 1:
        raise _fail(
            "CLOCK_STEP_TOOTH_BAND_AMBIGUOUS",
            f"body {body_id!r} has {len(candidates)} STEP tooth bands with "
            f"{expected_teeth} teeth",
        )
    teeth = candidates[0]
    radial_minimum = min(float(item["radial_min_m"]) for item in teeth)
    radial_maximum = max(float(item["radial_max_m"]) for item in teeth)
    axial_minimum = min(float(item["axial_min_m"]) for item in teeth)
    axial_maximum = max(float(item["axial_max_m"]) for item in teeth)
    # The trusted clock CAD uses a rectangular engineering tooth with 0.08 mm
    # tooth-tip clearance. Recover module from that measured tip surface and
    # the exact STEP-bound tooth count. Unlike a root-span formula, this stays
    # valid for small pinions whose root radius is deliberately clamped.
    inferred_module = (
        radial_maximum + 0.00008
    ) / (0.5 * expected_teeth + 1.0)
    if inferred_module <= 0.0:
        raise _fail(
            "CLOCK_STEP_TOOTH_MODULE_INVALID",
            f"body {body_id!r} has no positive STEP-derived tooth module",
        )
    inferred_pitch_radius = (
        0.5 * expected_teeth * inferred_module
    )
    tooth_solid_ids = {str(item["solid_id"]) for item in teeth}
    maximum_local_axial_width = max(
        0.006,
        6.0 * (axial_maximum - axial_minimum),
    )
    local_collision_records = list(teeth)
    for item in all_records:
        if str(item["solid_id"]) in tooth_solid_ids:
            continue
        candidate_axial_width = (
            float(item["axial_max_m"]) - float(item["axial_min_m"])
        )
        axial_overlap = min(
            axial_maximum,
            float(item["axial_max_m"]),
        ) - max(
            axial_minimum,
            float(item["axial_min_m"]),
        )
        # Gear roots are separate STEP solids in the trusted decomposition.
        # Include only the thin, same-layer solids that directly underlap the
        # bound teeth.  Long shafts and other coaxial layers stay outside this
        # validator-owned local collision exception.
        if not (
            candidate_axial_width <= maximum_local_axial_width
            and axial_overlap > 0.0
            and float(item["radial_max_m"]) >= radial_minimum + 0.0001
            and float(item["radial_min_m"]) <= radial_maximum
            and float(item["radial_max_m"]) <= radial_maximum + 0.0001
        ):
            continue
        local_collision_records.append(item)
    local_collision_records.sort(
        key=lambda item: int(str(item["solid_id"]).split("_")[-1])
    )
    return {
        "body_id": body_id,
        "expected_teeth": expected_teeth,
        "bound_component_count": len(teeth),
        "component_refs": [
            f"{body_id}/{item['solid_id']}" for item in teeth
        ],
        "local_collision_component_refs": [
            f"{body_id}/{item['solid_id']}"
            for item in local_collision_records
        ],
        "local_collision_component_inventory_sha256": canonical_sha256(
            [
                {
                    "component_ref": f"{body_id}/{item['solid_id']}",
                    "component_sha256": item["sha256"],
                }
                for item in local_collision_records
            ]
        ),
        "component_inventory_sha256": canonical_sha256(
            [
                {
                    "component_ref": f"{body_id}/{item['solid_id']}",
                    "component_sha256": item["sha256"],
                }
                for item in teeth
            ]
        ),
        "shaft_origin_abs_m": [float(item) for item in shaft_origin_abs_m],
        "shaft_axis_abs": [float(item) for item in shaft_axis_abs],
        "radial_min_m": radial_minimum,
        "radial_max_m": radial_maximum,
        "inferred_module_m": inferred_module,
        "inferred_pitch_radius_m": inferred_pitch_radius,
        "axial_min_relative_to_shaft_m": axial_minimum,
        "axial_max_relative_to_shaft_m": axial_maximum,
        "face_width_m": axial_maximum - axial_minimum,
    }


def build_clock_gear_geometry_contracts(
    spec: Any,
    collision_geometry: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build the one shared STEP-bound gate contract used before and in physics."""

    workspace = Path(workspace_root).resolve()
    shaft_by_body = {item.body_id: item for item in spec.shafts}
    contracts: list[dict[str, Any]] = []
    for mesh in spec.gear_meshes:
        driver_shaft = shaft_by_body[mesh.driver_body_id]
        driven_shaft = shaft_by_body[mesh.driven_body_id]
        driver = _clock_tooth_band(
            collision_geometry,
            workspace=workspace,
            body_id=mesh.driver_body_id,
            shaft_origin_abs_m=driver_shaft.origin_abs_m,
            shaft_axis_abs=driver_shaft.axis_abs,
            expected_teeth=mesh.driver_teeth,
        )
        driven = _clock_tooth_band(
            collision_geometry,
            workspace=workspace,
            body_id=mesh.driven_body_id,
            shaft_origin_abs_m=driven_shaft.origin_abs_m,
            shaft_axis_abs=driven_shaft.axis_abs,
            expected_teeth=mesh.driven_teeth,
        )
        mean_module = 0.5 * (
            float(driver["inferred_module_m"])
            + float(driven["inferred_module_m"])
        )
        radial_tolerance = max(0.0001, 0.1 * mean_module)
        axis_tolerance = max(
            math.radians(0.25),
            0.5 * float(spec.max_off_axis_rotation_rad),
        )
        gate = conditional_transmission_gate(
            geometry_plugin_id="axial_sliding_spur_gear_mesh_v1",
            body_a_id=mesh.driver_body_id,
            body_b_id=mesh.driven_body_id,
            endpoint_a_component_refs=driver["component_refs"],
            endpoint_b_component_refs=driven["component_refs"],
            endpoint_a_local_collision_component_refs=driver[
                "local_collision_component_refs"
            ],
            endpoint_b_local_collision_component_refs=driven[
                "local_collision_component_refs"
            ],
            activation_coordinate="face_overlap_m",
            enter_threshold_si=0.0002,
            release_threshold_si=0.00005,
            proof_source="validator_clock_spur_gear_step_geometry_v1",
            geometry_parameters={
                "driver_pitch_radius_m": driver[
                    "inferred_pitch_radius_m"
                ],
                "driven_pitch_radius_m": driven[
                    "inferred_pitch_radius_m"
                ],
                "driver_module_m": driver["inferred_module_m"],
                "driven_module_m": driven["inferred_module_m"],
                # Six-tooth rectangular STEP proxies have a conservative
                # corner-derived tip radius; 1.5% still rejects a wrong
                # module while covering that measured discretization bound.
                "module_relative_tolerance": 0.015,
                "radial_tolerance_m": radial_tolerance,
                "axis_parallel_tolerance_rad": axis_tolerance,
                "driver_face_width_m": driver["face_width_m"],
                "driven_face_width_m": driven["face_width_m"],
                "driver_axial_center_offset_m": 0.5
                * (
                    driver["axial_min_relative_to_shaft_m"]
                    + driver["axial_max_relative_to_shaft_m"]
                ),
                "driven_axial_center_offset_m": 0.5
                * (
                    driven["axial_min_relative_to_shaft_m"]
                    + driven["axial_max_relative_to_shaft_m"]
                ),
            },
        )
        contracts.append(
            {
                "interface_id": f"gear-{mesh.mesh_id}",
                "mesh_id": mesh.mesh_id,
                "body_a_id": mesh.driver_body_id,
                "body_b_id": mesh.driven_body_id,
                "gate": gate,
                "driver": driver,
                "driven": driven,
                "driver_shaft_origin_abs_m": list(
                    driver_shaft.origin_abs_m
                ),
                "driven_shaft_origin_abs_m": list(
                    driven_shaft.origin_abs_m
                ),
                "driver_shaft_axis_abs": list(driver_shaft.axis_abs),
                "driven_shaft_axis_abs": list(driven_shaft.axis_abs),
            }
        )
    return tuple(contracts)


def _axial_band_pose(
    band: Mapping[str, Any],
) -> AxialGearPose:
    axis = tuple(float(item) for item in band["shaft_axis_abs"])
    origin = tuple(
        float(item) for item in band["shaft_origin_abs_m"]
    )
    offset = 0.5 * (
        float(band["axial_min_relative_to_shaft_m"])
        + float(band["axial_max_relative_to_shaft_m"])
    )
    return AxialGearPose(
        center_abs_m=_add(origin, _scale(axis, offset)),
        axis_abs=axis,
        face_width_m=float(band["face_width_m"]),
    )


def _component_endpoint(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "body_id": str(record["body_id"]),
        "component_refs": [str(record["component_ref"])],
        "component_inventory_sha256": canonical_sha256(
            [
                {
                    "component_ref": str(record["component_ref"]),
                    "component_sha256": str(record["sha256"]),
                }
            ]
        ),
        "axial_min_relative_to_shaft_m": float(
            record["axial_min_m"]
        ),
        "axial_max_relative_to_shaft_m": float(
            record["axial_max_m"]
        ),
        "axial_center_offset_m": 0.5
        * (
            float(record["axial_min_m"])
            + float(record["axial_max_m"])
        ),
        "face_width_m": (
            float(record["axial_max_m"])
            - float(record["axial_min_m"])
        ),
        "radial_min_m": float(record["radial_min_m"]),
        "radial_max_m": float(record["radial_max_m"]),
    }


def _clock_transfer_contracts(
    spec: Any,
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[dict[str, Any], ...]:
    shaft_by_body = {item.body_id: item for item in spec.shafts}
    contracts: list[dict[str, Any]] = []
    for transfer in spec.shaft_transfers:
        driver_shaft = shaft_by_body[transfer.driver_body_id]
        driven_shaft = shaft_by_body[transfer.driven_body_id]
        driver_records = _component_axis_records(
            collision_geometry,
            workspace=workspace,
            body_id=transfer.driver_body_id,
            shaft_origin_abs_m=driver_shaft.origin_abs_m,
            shaft_axis_abs=driver_shaft.axis_abs,
        )
        driven_records = _component_axis_records(
            collision_geometry,
            workspace=workspace,
            body_id=transfer.driven_body_id,
            shaft_origin_abs_m=driven_shaft.origin_abs_m,
            shaft_axis_abs=driven_shaft.axis_abs,
        )
        driver_candidates = _radial_band_candidates(driver_records)
        driven_candidates = _radial_band_candidates(driven_records)
        ranked: list[
            tuple[
                float,
                int,
                int,
                tuple[Mapping[str, Any], ...],
                tuple[Mapping[str, Any], ...],
            ]
        ] = []
        for driver_count, driver_group in driver_candidates.items():
            for driven_count, driven_group in driven_candidates.items():
                measured_ratio = driver_count / driven_count
                ranked.append(
                    (
                        abs(
                            measured_ratio
                            - abs(float(transfer.ratio))
                        ),
                        driver_count,
                        driven_count,
                        driver_group,
                        driven_group,
                    )
                )
        ranked.sort(key=lambda item: item[:3])
        if not ranked or ranked[0][0] > 0.02:
            raise _fail(
                "CLOCK_STEP_TRANSFER_PULLEY_BINDING_MISSING",
                f"shaft transfer {transfer.transfer_id!r} has no STEP-bound "
                "pulley pair matching its ratio",
            )
        if len(ranked) > 1 and abs(ranked[1][0] - ranked[0][0]) <= 1.0e-12:
            raise _fail(
                "CLOCK_STEP_TRANSFER_PULLEY_BINDING_AMBIGUOUS",
                f"shaft transfer {transfer.transfer_id!r} has ambiguous "
                "STEP-bound pulley pairs",
            )
        _, _, _, driver_group, driven_group = ranked[0]
        driver = _band_payload(
            body_id=transfer.driver_body_id,
            shaft_origin_abs_m=driver_shaft.origin_abs_m,
            shaft_axis_abs=driver_shaft.axis_abs,
            records=driver_group,
        )
        driven = _band_payload(
            body_id=transfer.driven_body_id,
            shaft_origin_abs_m=driven_shaft.origin_abs_m,
            shaft_axis_abs=driven_shaft.axis_abs,
            records=driven_group,
        )
        driver_pose = _axial_band_pose(driver)
        driven_pose = _axial_band_pose(driven)
        center_delta = _sub(
            driver_pose.center_abs_m,
            driven_pose.center_abs_m,
        )
        axis = _unit(
            _add(driver_pose.axis_abs, driven_pose.axis_abs),
            f"{transfer.transfer_id}.common_axis",
        )
        axial = _dot(center_delta, axis)
        center_distance = _norm(
            _sub(center_delta, _scale(axis, axial))
        )
        center_tolerance = max(
            0.0005,
            1.5 * float(spec.max_support_drift_m),
        )
        axis_tolerance = max(
            math.radians(0.25),
            0.5 * float(spec.max_off_axis_rotation_rad),
        )
        gate = conditional_transmission_gate(
            geometry_plugin_id="coplanar_pulley_transfer_v1",
            body_a_id=transfer.driver_body_id,
            body_b_id=transfer.driven_body_id,
            endpoint_a_component_refs=driver["component_refs"],
            endpoint_b_component_refs=driven["component_refs"],
            activation_coordinate="face_overlap_m",
            enter_threshold_si=0.0002,
            release_threshold_si=0.00005,
            proof_source="validator_clock_pulley_step_geometry_v1",
            geometry_parameters={
                "required_center_distance_m": center_distance,
                "center_distance_tolerance_m": center_tolerance,
                "axis_parallel_tolerance_rad": axis_tolerance,
                "driver_face_width_m": driver["face_width_m"],
                "driven_face_width_m": driven["face_width_m"],
                "driver_axial_center_offset_m": 0.5
                * (
                    driver["axial_min_relative_to_shaft_m"]
                    + driver["axial_max_relative_to_shaft_m"]
                ),
                "driven_axial_center_offset_m": 0.5
                * (
                    driven["axial_min_relative_to_shaft_m"]
                    + driven["axial_max_relative_to_shaft_m"]
                ),
            },
        )
        observation = CoplanarPulleyTransferGeometry(
            required_center_distance_m=center_distance,
            center_distance_tolerance_m=center_tolerance,
            axis_parallel_tolerance_rad=axis_tolerance,
        ).evaluate(
            time_s=0.0,
            driver=driver_pose,
            driven=driven_pose,
        )
        contracts.append(
            {
                "interface_id": f"transfer-{transfer.transfer_id}",
                "interface_kind": "shaft_transfer",
                "model": "finite_conditional_torsional_pulley_transfer",
                "body_a_id": transfer.driver_body_id,
                "body_b_id": transfer.driven_body_id,
                "gate": gate,
                "endpoint_a": driver,
                "endpoint_b": driven,
                "initial_observation": observation.to_dict(),
            }
        )
    return tuple(contracts)


def _clock_cam_contracts(
    spec: Any,
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[dict[str, Any], ...]:
    shaft_by_body = {item.body_id: item for item in spec.shafts}
    contracts: list[dict[str, Any]] = []
    for follower in spec.cam_followers:
        driver_shaft = shaft_by_body[follower.cam_body_id]
        follower_shaft = shaft_by_body[follower.follower_body_id]
        driver_records = _component_axis_records(
            collision_geometry,
            workspace=workspace,
            body_id=follower.cam_body_id,
            shaft_origin_abs_m=driver_shaft.origin_abs_m,
            shaft_axis_abs=driver_shaft.axis_abs,
        )
        follower_records = _component_axis_records(
            collision_geometry,
            workspace=workspace,
            body_id=follower.follower_body_id,
            shaft_origin_abs_m=follower_shaft.origin_abs_m,
            shaft_axis_abs=follower_shaft.axis_abs,
        )
        driver_record, follower_record, _ = (
            _nearest_component_pair(driver_records, follower_records)
        )
        endpoint_a = _component_endpoint(driver_record)
        endpoint_b = _component_endpoint(follower_record)
        axis_tolerance = max(
            math.radians(0.25),
            0.5 * float(spec.max_off_axis_rotation_rad),
        )
        radial_gap_tolerance = 0.0015
        gate = conditional_transmission_gate(
            geometry_plugin_id="radial_cam_follower_reach_v1",
            body_a_id=follower.cam_body_id,
            body_b_id=follower.follower_body_id,
            endpoint_a_component_refs=endpoint_a["component_refs"],
            endpoint_b_component_refs=endpoint_b["component_refs"],
            activation_coordinate="surface_gap_m",
            enter_threshold_si=radial_gap_tolerance,
            release_threshold_si=radial_gap_tolerance + 0.0015,
            activation_direction=AT_OR_BELOW,
            proof_source="validator_clock_cam_step_geometry_v1",
            geometry_parameters={
                "radial_gap_tolerance_m": radial_gap_tolerance,
                "axis_parallel_tolerance_rad": axis_tolerance,
                "driver_radial_reach_m": endpoint_a["radial_max_m"],
                "follower_radial_reach_m": endpoint_b["radial_max_m"],
                "driver_face_width_m": endpoint_a["face_width_m"],
                "follower_face_width_m": endpoint_b["face_width_m"],
                "driver_axial_center_offset_m": endpoint_a[
                    "axial_center_offset_m"
                ],
                "follower_axial_center_offset_m": endpoint_b[
                    "axial_center_offset_m"
                ],
            },
        )
        driver_pose = RadialReachPose(
            center_abs_m=_add(
                driver_shaft.origin_abs_m,
                _scale(
                    driver_shaft.axis_abs,
                    endpoint_a["axial_center_offset_m"],
                ),
            ),
            axis_abs=driver_shaft.axis_abs,
            radial_reach_m=endpoint_a["radial_max_m"],
            face_width_m=endpoint_a["face_width_m"],
        )
        follower_pose = RadialReachPose(
            center_abs_m=_add(
                follower_shaft.origin_abs_m,
                _scale(
                    follower_shaft.axis_abs,
                    endpoint_b["axial_center_offset_m"],
                ),
            ),
            axis_abs=follower_shaft.axis_abs,
            radial_reach_m=endpoint_b["radial_max_m"],
            face_width_m=endpoint_b["face_width_m"],
        )
        observation = RadialReachGeometry(
            radial_gap_tolerance_m=radial_gap_tolerance,
            axis_parallel_tolerance_rad=axis_tolerance,
        ).evaluate(
            time_s=0.0,
            driver=driver_pose,
            follower=follower_pose,
        )
        contracts.append(
            {
                "interface_id": f"cam-{follower.follower_id}",
                "interface_kind": "cam_follower",
                "model": "finite_conditional_analytical_cam_follower",
                "body_a_id": follower.cam_body_id,
                "body_b_id": follower.follower_body_id,
                "gate": gate,
                "endpoint_a": endpoint_a,
                "endpoint_b": endpoint_b,
                "initial_observation": observation.to_dict(),
            }
        )
    return tuple(contracts)


def _nearest_axial_faces(
    endpoint_a: Mapping[str, Any],
    endpoint_b: Mapping[str, Any],
    *,
    shaft_a: Any,
    shaft_b: Any,
) -> tuple[float, float]:
    axis = _unit(
        _add(shaft_a.axis_abs, shaft_b.axis_abs),
        "clutch.common_axis",
    )
    origin_a = _dot(shaft_a.origin_abs_m, axis)
    origin_b = _dot(shaft_b.origin_abs_m, axis)
    a_min = origin_a + float(
        endpoint_a["axial_min_relative_to_shaft_m"]
    )
    a_max = origin_a + float(
        endpoint_a["axial_max_relative_to_shaft_m"]
    )
    b_min = origin_b + float(
        endpoint_b["axial_min_relative_to_shaft_m"]
    )
    b_max = origin_b + float(
        endpoint_b["axial_max_relative_to_shaft_m"]
    )
    if a_max <= b_min:
        return a_max - origin_a, b_min - origin_b
    if b_max <= a_min:
        return a_min - origin_a, b_max - origin_b
    shared = 0.5 * (max(a_min, b_min) + min(a_max, b_max))
    return shared - origin_a, shared - origin_b


def _clock_clutch_contract(
    spec: Any,
    clutch: Any,
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
    interface_kind: str,
) -> dict[str, Any] | None:
    if clutch is None:
        return None
    shaft_by_body = {item.body_id: item for item in spec.shafts}
    input_shaft = shaft_by_body[clutch.input_body_id]
    driven_shaft = shaft_by_body[clutch.driven_body_id]
    input_records = _component_axis_records(
        collision_geometry,
        workspace=workspace,
        body_id=clutch.input_body_id,
        shaft_origin_abs_m=input_shaft.origin_abs_m,
        shaft_axis_abs=input_shaft.axis_abs,
    )
    driven_records = _component_axis_records(
        collision_geometry,
        workspace=workspace,
        body_id=clutch.driven_body_id,
        shaft_origin_abs_m=driven_shaft.origin_abs_m,
        shaft_axis_abs=driven_shaft.axis_abs,
    )
    input_record, driven_record, _ = _nearest_component_pair(
        input_records,
        driven_records,
    )
    endpoint_a = _component_endpoint(input_record)
    endpoint_b = _component_endpoint(driven_record)
    input_face_offset, driven_face_offset = _nearest_axial_faces(
        endpoint_a,
        endpoint_b,
        shaft_a=input_shaft,
        shaft_b=driven_shaft,
    )
    endpoint_a["face_center_offset_m"] = input_face_offset
    endpoint_b["face_center_offset_m"] = driven_face_offset
    axis_tolerance = max(
        math.radians(0.25),
        0.5 * float(spec.max_off_axis_rotation_rad),
    )
    radial_tolerance = max(
        0.0005,
        1.5 * float(spec.max_support_drift_m),
    )
    gate = conditional_transmission_gate(
        geometry_plugin_id="coaxial_clutch_face_gap_v1",
        body_a_id=clutch.input_body_id,
        body_b_id=clutch.driven_body_id,
        endpoint_a_component_refs=endpoint_a["component_refs"],
        endpoint_b_component_refs=endpoint_b["component_refs"],
        activation_coordinate="surface_gap_m",
        enter_threshold_si=0.0015,
        release_threshold_si=0.003,
        activation_direction=AT_OR_BELOW,
        proof_source="validator_clock_clutch_step_geometry_v1",
        geometry_parameters={
            "radial_alignment_tolerance_m": radial_tolerance,
            "axis_parallel_tolerance_rad": axis_tolerance,
            "input_face_center_offset_m": input_face_offset,
            "driven_face_center_offset_m": driven_face_offset,
            "input_radial_extent_m": endpoint_a["radial_max_m"],
            "driven_radial_extent_m": endpoint_b["radial_max_m"],
        },
    )
    input_pose = AxialClutchPose(
        face_center_abs_m=_add(
            input_shaft.origin_abs_m,
            _scale(input_shaft.axis_abs, input_face_offset),
        ),
        axis_abs=input_shaft.axis_abs,
        radial_extent_m=endpoint_a["radial_max_m"],
    )
    driven_pose = AxialClutchPose(
        face_center_abs_m=_add(
            driven_shaft.origin_abs_m,
            _scale(driven_shaft.axis_abs, driven_face_offset),
        ),
        axis_abs=driven_shaft.axis_abs,
        radial_extent_m=endpoint_b["radial_max_m"],
    )
    observation = AxialClutchGeometry(
        radial_alignment_tolerance_m=radial_tolerance,
        axis_parallel_tolerance_rad=axis_tolerance,
    ).evaluate(
        time_s=0.0,
        input_face=input_pose,
        driven_face=driven_pose,
    )
    return {
        "interface_id": f"clutch-{clutch.clutch_id}",
        "interface_kind": interface_kind,
        "model": "finite_conditional_torque_limited_setting_clutch",
        "body_a_id": clutch.input_body_id,
        "body_b_id": clutch.driven_body_id,
        "gate": gate,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "initial_observation": observation.to_dict(),
    }


def build_clock_functional_geometry_contracts(
    spec: Any,
    collision_geometry: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build every STEP-bound analytical clock interface contract."""

    workspace = Path(workspace_root).resolve()
    contracts: list[dict[str, Any]] = []
    for gear in build_clock_gear_geometry_contracts(
        spec,
        collision_geometry,
        workspace_root=workspace,
    ):
        geometry = gear["gate"]["geometry_parameters"]
        observation = AxialSlidingSpurGearGeometry(
            driver_pitch_radius_m=float(
                geometry["driver_pitch_radius_m"]
            ),
            driven_pitch_radius_m=float(
                geometry["driven_pitch_radius_m"]
            ),
            radial_tolerance_m=float(
                geometry["radial_tolerance_m"]
            ),
            axis_parallel_tolerance_rad=float(
                geometry["axis_parallel_tolerance_rad"]
            ),
            driver_module_m=float(geometry["driver_module_m"]),
            driven_module_m=float(geometry["driven_module_m"]),
            module_relative_tolerance=float(
                geometry["module_relative_tolerance"]
            ),
        ).evaluate(
            time_s=0.0,
            driver=_axial_band_pose(gear["driver"]),
            driven=_axial_band_pose(gear["driven"]),
        )
        contracts.append(
            {
                **gear,
                "interface_kind": "spur_gear_mesh",
                "model": "finite_conditional_compliant_gear_mesh_torque",
                "endpoint_a": gear["driver"],
                "endpoint_b": gear["driven"],
                "initial_observation": observation.to_dict(),
            }
        )
    contracts.extend(
        _clock_transfer_contracts(
            spec,
            collision_geometry,
            workspace=workspace,
        )
    )
    contracts.extend(
        _clock_cam_contracts(
            spec,
            collision_geometry,
            workspace=workspace,
        )
    )
    clutch = _clock_clutch_contract(
        spec,
        spec.setting_clutch,
        collision_geometry,
        workspace=workspace,
        interface_kind="setting_clutch",
    )
    if clutch is not None:
        contracts.append(clutch)
    alarm_setting_clutch = _clock_clutch_contract(
        spec,
        spec.alarm_setting_clutch,
        collision_geometry,
        workspace=workspace,
        interface_kind="alarm_setting_clutch",
    )
    if alarm_setting_clutch is not None:
        contracts.append(alarm_setting_clutch)
    return tuple(contracts)


def build_multi_output_shaft_functional_geometry_contracts(
    spec: Any,
    collision_geometry: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> tuple[dict[str, Any], ...]:
    """Build STEP-bound gates for reusable right-angle shaft couplings."""

    workspace = Path(workspace_root).resolve()
    shaft_by_body = {item.body_id: item for item in spec.shafts}
    contracts: list[dict[str, Any]] = []
    for mesh in spec.gear_meshes:
        driver_shaft = shaft_by_body[mesh.driver_body_id]
        driven_shaft = shaft_by_body[mesh.driven_body_id]
        driver = _right_angle_tooth_band(
            collision_geometry,
            workspace=workspace,
            body_id=mesh.driver_body_id,
            shaft_origin_abs_m=driver_shaft.origin_abs_m,
            shaft_axis_abs=driver_shaft.axis_abs,
            expected_teeth=mesh.driver_teeth,
            expected_pitch_radius_m=mesh.driver_pitch_radius_m,
            expected_face_width_m=mesh.driver_face_width_m,
        )
        driven = _right_angle_tooth_band(
            collision_geometry,
            workspace=workspace,
            body_id=mesh.driven_body_id,
            shaft_origin_abs_m=driven_shaft.origin_abs_m,
            shaft_axis_abs=driven_shaft.axis_abs,
            expected_teeth=mesh.driven_teeth,
            expected_pitch_radius_m=mesh.driven_pitch_radius_m,
            expected_face_width_m=mesh.driven_face_width_m,
        )
        geometry = RightAngleBevelGearGeometry(
            driver_pitch_radius_m=mesh.driver_pitch_radius_m,
            driven_pitch_radius_m=mesh.driven_pitch_radius_m,
            pitch_surface_tolerance_m=(
                mesh.geometry_contact_tolerance_m
            ),
            required_shaft_angle_rad=mesh.shaft_angle_rad,
            shaft_angle_tolerance_rad=(
                mesh.axis_perpendicular_tolerance_rad
            ),
            driver_face_width_m=driver["face_width_m"],
            driven_face_width_m=driven["face_width_m"],
        )
        observation = geometry.evaluate(
            time_s=0.0,
            driver=AxialGearPose(
                center_abs_m=driver_shaft.origin_abs_m,
                axis_abs=driver_shaft.axis_abs,
                face_width_m=driver["face_width_m"],
            ),
            driven=AxialGearPose(
                center_abs_m=driven_shaft.origin_abs_m,
                axis_abs=driven_shaft.axis_abs,
                face_width_m=driven["face_width_m"],
            ),
        )
        geometry_direction_sign = float(
            observation.metrics["mesh_direction_sign"]
        )
        declared_direction_sign = math.copysign(
            1.0,
            mesh.declared_ratio,
        )
        if geometry_direction_sign != declared_direction_sign:
            raise _fail(
                "MULTI_OUTPUT_STEP_MESH_DIRECTION_MISMATCH",
                f"shaft coupling {mesh.mesh_id!r} declares ratio "
                f"{mesh.declared_ratio:+g}, but its STEP-bound pitch contact "
                f"requires sign {geometry_direction_sign:+g}",
            )
        gate = conditional_transmission_gate(
            geometry_plugin_id="right_angle_bevel_gear_mesh_v1",
            body_a_id=mesh.driver_body_id,
            body_b_id=mesh.driven_body_id,
            endpoint_a_component_refs=driver["component_refs"],
            endpoint_b_component_refs=driven["component_refs"],
            endpoint_a_local_collision_component_refs=driver[
                "local_collision_component_refs"
            ],
            endpoint_b_local_collision_component_refs=driven[
                "local_collision_component_refs"
            ],
            activation_coordinate="pitch_surface_gap_m",
            activation_direction="AT_OR_BELOW",
            enter_threshold_si=mesh.geometry_contact_tolerance_m,
            release_threshold_si=(
                1.5 * mesh.geometry_contact_tolerance_m
            ),
            proof_source="validator_multi_output_bevel_step_geometry_v2",
            geometry_parameters={
                "driver_pitch_radius_m": mesh.driver_pitch_radius_m,
                "driven_pitch_radius_m": mesh.driven_pitch_radius_m,
                "pitch_surface_tolerance_m": (
                    mesh.geometry_contact_tolerance_m
                ),
                "required_shaft_angle_rad": mesh.shaft_angle_rad,
                "shaft_angle_tolerance_rad": (
                    mesh.axis_perpendicular_tolerance_rad
                ),
                "driver_face_width_m": driver["face_width_m"],
                "driven_face_width_m": driven["face_width_m"],
                "declared_ratio": mesh.declared_ratio,
                "geometry_derived_ratio_sign": (
                    geometry_direction_sign
                ),
            },
        )
        contracts.append(
            {
                "interface_id": f"gear-{mesh.mesh_id}",
                "interface_kind": "right_angle_bevel_gear_mesh",
                "model": (
                    "finite_conditional_compliant_multi_output_gear_torque"
                ),
                "mesh_id": mesh.mesh_id,
                "body_a_id": mesh.driver_body_id,
                "body_b_id": mesh.driven_body_id,
                "gate": gate,
                "endpoint_a": driver,
                "endpoint_b": driven,
                "driver": driver,
                "driven": driven,
                "driver_shaft_origin_abs_m": list(
                    driver_shaft.origin_abs_m
                ),
                "driven_shaft_origin_abs_m": list(
                    driven_shaft.origin_abs_m
                ),
                "driver_shaft_axis_abs": list(driver_shaft.axis_abs),
                "driven_shaft_axis_abs": list(driven_shaft.axis_abs),
                "initial_observation": observation.to_dict(),
            }
        )
    return tuple(contracts)


def _multi_output_shaft_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    from .collision_policy import (
        ANALYTICAL_TRANSMISSION,
        interface_declaration,
    )
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )
    from .multi_output_shaft_spec import multi_output_shaft_from_dict

    spec = multi_output_shaft_from_dict(adapter.family_spec)
    collision_geometry = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    contracts = build_multi_output_shaft_functional_geometry_contracts(
        spec,
        collision_geometry,
        workspace_root=workspace,
    )
    expected_interface_ids = {
        f"gear-{item.mesh_id}" for item in spec.gear_meshes
    }
    actual_interface_ids = {
        str(item["interface_id"]) for item in contracts
    }
    failures: list[str] = []
    if actual_interface_ids != expected_interface_ids:
        failures.append(
            "analytical_transmission_geometry_gate_inventory_mismatch"
        )
    declarations: list[dict[str, Any]] = []
    interfaces: list[dict[str, Any]] = []
    for contract in contracts:
        raw = contract["initial_observation"]
        observation = InterfaceGeometryObservation(
            time_s=float(raw["time_s"]),
            activation_coordinate_si=float(
                raw["activation_coordinate_si"]
            ),
            activation_rate_si_s=float(
                raw["activation_rate_si_s"]
            ),
            constraints_satisfied=bool(
                raw["constraints_satisfied"]
            ),
            can_engage_later=bool(raw["can_engage_later"]),
            metrics=dict(raw.get("metrics") or {}),
            blockers=tuple(raw.get("blockers") or ()),
        )
        assessment = assess_initial_functional_geometry(
            contract["gate"],
            observation,
        )
        declarations.append(
            interface_declaration(
                interface_id=contract["interface_id"],
                classification=ANALYTICAL_TRANSMISSION,
                body_a_id=contract["body_a_id"],
                body_b_id=contract["body_b_id"],
                model=contract["model"],
                source="trusted_multi_output_shaft_geometry_contract_v1",
                transmission_gate=contract["gate"],
            )
        )
        interfaces.append(
            {
                "interface_id": contract["interface_id"],
                "interface_kind": contract["interface_kind"],
                "classification": "ANALYTICAL_TRANSMISSION",
                "body_a_id": contract["body_a_id"],
                "body_b_id": contract["body_b_id"],
                "registered_geometry_gate": True,
                "gate_sha256": canonical_sha256(contract["gate"]),
                **assessment,
                "initially_transmitting": assessment[
                    "transmission_active"
                ],
                "can_engage_later": observation.can_engage_later,
                "endpoint_a": contract["endpoint_a"],
                "endpoint_b": contract["endpoint_b"],
            }
        )
    initially_inactive = [
        str(item["interface_id"])
        for item in interfaces
        if item["initially_transmitting"] is False
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "FAIL"
            if failures
            else (
                "PASS_WITH_WARNINGS"
                if initially_inactive
                else "PASS"
            )
        ),
        "passed": not failures,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "analytical_interface_count": len(expected_interface_ids),
        "registered_geometry_gate_count": len(actual_interface_ids),
        "ungated_analytical_interface_count": len(
            expected_interface_ids - actual_interface_ids
        ),
        "initially_nontransmitting_interface_count": len(
            initially_inactive
        ),
        "initially_nontransmitting_interfaces": initially_inactive,
        "warning_count": sum(
            item.get("status") == "WARNING" for item in interfaces
        ),
        "failure_reasons": failures,
        "interfaces": interfaces,
        "functional_interface_geometry_binding": (
            build_functional_interface_geometry_binding(
                declarations,
                collision_geometry,
            )
        ),
        "physical_simulation_started": False,
    }


def _detent_slider_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Bind the declared rack and tab to the authoritative STEP inventory."""

    from .contact_intent import collision_component_inventory
    from .detent_slider_spec import prismatic_detent_latch_from_dict
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )

    spec = prismatic_detent_latch_from_dict(adapter.family_spec)
    collision_geometry = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    inventory = {
        str(item["component_ref"]): item
        for item in collision_component_inventory(collision_geometry)
    }
    geometry = spec.detent_interface.geometry_contract
    declared_refs = (
        *geometry.rack_component_refs,
        *geometry.seat_component_refs,
        geometry.tab_component_ref,
    )
    failures: list[str] = []
    missing = [ref for ref in declared_refs if ref not in inventory]
    if missing:
        failures.append("detent_step_component_binding_missing")
    wrong_rack_owner = [
        ref
        for ref in (
            *geometry.rack_component_refs,
            *geometry.seat_component_refs,
        )
        if ref in inventory
        and inventory[ref]["body_id"] != spec.detent_interface.rack_body_id
    ]
    wrong_tab_owner = (
        geometry.tab_component_ref in inventory
        and inventory[geometry.tab_component_ref]["body_id"]
        != spec.detent_interface.tab_body_id
    )
    if wrong_rack_owner or wrong_tab_owner:
        failures.append("detent_step_component_owner_mismatch")
    bound_components = [
        {
            "component_ref": ref,
            "component_sha256": inventory[ref]["component_sha256"],
        }
        for ref in declared_refs
        if ref in inventory
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL" if failures else "PASS",
        "passed": not failures,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "analytical_interface_count": 0,
        "registered_geometry_gate_count": 0,
        "ungated_analytical_interface_count": 0,
        "physical_contact_interface_count": 1,
        "initially_nontransmitting_interface_count": 0,
        "initially_nontransmitting_interfaces": [],
        "warning_count": 0,
        "failure_reasons": failures,
        "interfaces": [
            {
                "interface_id": "contact-discrete-detent",
                "interface_kind": "finite_step_surface_contact",
                "classification": "CONTACT_PHYSICAL",
                "body_a_id": spec.detent_interface.rack_body_id,
                "body_b_id": spec.detent_interface.tab_body_id,
                "registered_geometry_gate": False,
                "analytical_force_path": False,
                "rack_component_refs": list(
                    geometry.rack_component_refs
                ),
                "seat_component_refs": list(
                    geometry.seat_component_refs
                ),
                "tab_component_ref": geometry.tab_component_ref,
                "bound_component_count": len(bound_components),
            }
        ],
        "functional_interface_geometry_binding": {
            "schema_version": "detent_contact_step_binding_v1",
            "package_binding_sha256": plan.package_binding_sha256,
            "components": bound_components,
            "binding_sha256": canonical_sha256(bound_components),
        },
        "physical_simulation_started": False,
    }


def _click_pen_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Bind all three click-pen contact paths to authoritative STEP solids."""

    from .click_pen_spec import click_pen_cam_latch_from_dict
    from .contact_intent import collision_component_inventory
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )

    spec = click_pen_cam_latch_from_dict(adapter.family_spec)
    collision_geometry = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    inventory = {
        str(item["component_ref"]): item
        for item in collision_component_inventory(collision_geometry)
    }
    geometry = spec.click_state_machine.geometry_contract
    interfaces = (
        (
            "contact-button-cam-drive",
            spec.input_body_id,
            spec.cam_body_id,
            geometry.button_contact_component_refs,
            geometry.cam_button_face_component_refs,
        ),
        (
            "contact-cam-refill-follower",
            spec.cam_body_id,
            spec.output_body_id,
            (geometry.cam_follower_component_ref,),
            (geometry.refill_carrier_component_ref,),
        ),
        (
            "contact-cam-barrel-index",
            spec.fixed_body_id,
            spec.cam_body_id,
            geometry.barrel_guide_component_refs,
            geometry.cam_lobe_component_refs,
        ),
    )
    declared_refs = tuple(
        ref
        for _interface_id, _body_a, _body_b, refs_a, refs_b in interfaces
        for ref in (*refs_a, *refs_b)
    )
    failures: list[str] = []
    if any(ref not in inventory for ref in declared_refs):
        failures.append("click_pen_step_component_binding_missing")
    for _interface_id, body_a, body_b, refs_a, refs_b in interfaces:
        if any(
            ref in inventory and inventory[ref]["body_id"] != body_a
            for ref in refs_a
        ) or any(
            ref in inventory and inventory[ref]["body_id"] != body_b
            for ref in refs_b
        ):
            failures.append("click_pen_step_component_owner_mismatch")
            break
    bound_components = [
        {
            "component_ref": ref,
            "component_sha256": inventory[ref]["component_sha256"],
        }
        for ref in dict.fromkeys(declared_refs)
        if ref in inventory
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL" if failures else "PASS",
        "passed": not failures,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "analytical_interface_count": 0,
        "registered_geometry_gate_count": 0,
        "ungated_analytical_interface_count": 0,
        "physical_contact_interface_count": len(interfaces),
        "initially_nontransmitting_interface_count": 0,
        "initially_nontransmitting_interfaces": [],
        "warning_count": 0,
        "failure_reasons": failures,
        "interfaces": [
            {
                "interface_id": interface_id,
                "interface_kind": "finite_step_surface_contact",
                "classification": "CONTACT_PHYSICAL",
                "body_a_id": body_a,
                "body_b_id": body_b,
                "registered_geometry_gate": False,
                "analytical_force_path": False,
                "component_a_refs": list(refs_a),
                "component_b_refs": list(refs_b),
                "bound_component_count": sum(
                    ref in inventory for ref in (*refs_a, *refs_b)
                ),
            }
            for interface_id, body_a, body_b, refs_a, refs_b in interfaces
        ],
        "functional_interface_geometry_binding": {
            "schema_version": "click_pen_contact_step_binding_v1",
            "package_binding_sha256": plan.package_binding_sha256,
            "components": bound_components,
            "binding_sha256": canonical_sha256(bound_components),
        },
        "physical_simulation_started": False,
    }


def _clock_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    from .clock_gear_train_spec import clock_gear_train_from_dict
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )

    spec = clock_gear_train_from_dict(adapter.family_spec)
    collision_geometry = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    interfaces: list[dict[str, Any]] = []
    failure_reasons: list[str] = []
    interface_declarations: list[dict[str, Any]] = []

    if spec.collision_exclusions:
        failure_reasons.append(
            "clock_body_pair_collision_exclusion_forbidden"
        )

    from .collision_policy import (
        ANALYTICAL_TRANSMISSION,
        interface_declaration,
    )

    contracts = build_clock_functional_geometry_contracts(
        spec,
        collision_geometry,
        workspace_root=workspace,
    )
    expected_interface_ids = {
        *(f"gear-{item.mesh_id}" for item in spec.gear_meshes),
        *(
            f"transfer-{item.transfer_id}"
            for item in spec.shaft_transfers
        ),
        *(f"cam-{item.follower_id}" for item in spec.cam_followers),
        *(
            [f"clutch-{spec.setting_clutch.clutch_id}"]
            if spec.setting_clutch is not None
            else []
        ),
        *(
            [f"clutch-{spec.alarm_setting_clutch.clutch_id}"]
            if spec.alarm_setting_clutch is not None
            else []
        ),
    }
    actual_interface_ids = {
        str(item["interface_id"]) for item in contracts
    }
    missing_interface_ids = (
        expected_interface_ids - actual_interface_ids
    )
    if actual_interface_ids != expected_interface_ids:
        failure_reasons.append(
            "analytical_transmission_geometry_gate_inventory_mismatch"
        )
    for contract in contracts:
        observation_raw = contract["initial_observation"]
        observation = InterfaceGeometryObservation(
            time_s=float(observation_raw["time_s"]),
            activation_coordinate_si=float(
                observation_raw["activation_coordinate_si"]
            ),
            activation_rate_si_s=float(
                observation_raw["activation_rate_si_s"]
            ),
            constraints_satisfied=bool(
                observation_raw["constraints_satisfied"]
            ),
            can_engage_later=bool(
                observation_raw["can_engage_later"]
            ),
            metrics=dict(observation_raw.get("metrics") or {}),
            blockers=tuple(observation_raw.get("blockers") or ()),
        )
        assessment = assess_initial_functional_geometry(
            contract["gate"],
            observation,
        )
        interface_declarations.append(
            interface_declaration(
                interface_id=contract["interface_id"],
                classification=ANALYTICAL_TRANSMISSION,
                body_a_id=contract["body_a_id"],
                body_b_id=contract["body_b_id"],
                model=contract["model"],
                source="trusted_clock_functional_geometry_contract_v1",
                transmission_gate=contract["gate"],
            )
        )
        interfaces.append(
            {
                "interface_id": contract["interface_id"],
                "interface_kind": contract["interface_kind"],
                "classification": "ANALYTICAL_TRANSMISSION",
                "body_a_id": contract["body_a_id"],
                "body_b_id": contract["body_b_id"],
                "registered_geometry_gate": True,
                "gate_sha256": canonical_sha256(contract["gate"]),
                **assessment,
                "initially_transmitting": assessment[
                    "transmission_active"
                ],
                "can_engage_later": observation.can_engage_later,
                "endpoint_a": contract["endpoint_a"],
                "endpoint_b": contract["endpoint_b"],
            }
        )

    ungated_interfaces = [
        item
        for item in interfaces
        if item.get("registered_geometry_gate") is not True
    ]
    if ungated_interfaces:
        failure_reasons.append(
            "analytical_transmission_geometry_gate_missing"
        )
    initially_inactive = [
        str(item["interface_id"])
        for item in interfaces
        if item.get("initially_transmitting") is False
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "FAIL"
            if failure_reasons
            else ("PASS_WITH_WARNINGS" if initially_inactive else "PASS")
        ),
        "passed": not failure_reasons,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "analytical_interface_count": len(expected_interface_ids),
        "registered_geometry_gate_count": len(actual_interface_ids),
        "ungated_analytical_interface_count": (
            len(ungated_interfaces) + len(missing_interface_ids)
        ),
        "initially_nontransmitting_interface_count": len(initially_inactive),
        "initially_nontransmitting_interfaces": initially_inactive,
        "warning_count": sum(
            item.get("status") == "WARNING" for item in interfaces
        ),
        "failure_reasons": list(dict.fromkeys(failure_reasons)),
        "interfaces": interfaces,
        "functional_interface_geometry_binding": (
            build_functional_interface_geometry_binding(
                interface_declarations,
                collision_geometry,
            )
        ),
        "physical_simulation_started": False,
    }


def _rack_pinion_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    from .collision_policy import (
        ANALYTICAL_TRANSMISSION,
        interface_declaration,
    )
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )
    from .rack_pinion_runtime_v10 import (
        _rack_pinion_interface_component_refs,
    )
    from .rack_pinion_spec import rack_pinion_from_dict

    spec = rack_pinion_from_dict(adapter.family_spec)
    collision_geometry = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    body_specs = {item.body_id: item for item in spec.bodies}
    pinion_body = body_specs[spec.pinion_body_id]
    rack_body = body_specs[spec.rack_body_id]
    pinion_rotation = _rotation_matrix(
        pinion_body.rotation_abs_quat_wxyz
    )
    rack_rotation = _rotation_matrix(rack_body.rotation_abs_quat_wxyz)
    pinion_center_local = _world_point_to_local(
        spec.pinion_origin_abs_m,
        center_abs_m=pinion_body.center_abs_m,
        rotation_abs_from_body=pinion_rotation,
    )
    pinion_axis_local = _world_vector_to_local(
        spec.pinion_axis_abs,
        pinion_rotation,
    )
    rack_pitch_line_local = _world_point_to_local(
        plan.observable_point_abs_m,
        center_abs_m=rack_body.center_abs_m,
        rotation_abs_from_body=rack_rotation,
    )
    rack_axis_local = _world_vector_to_local(
        spec.rack_axis_abs,
        rack_rotation,
    )
    rack_face_axis_local = _world_vector_to_local(
        spec.pinion_axis_abs,
        rack_rotation,
    )
    pinion_refs, rack_refs, face_width = (
        _rack_pinion_interface_component_refs(
            collision_geometry,
            workspace=workspace,
            spec=spec,
            pinion_center_local_m=pinion_center_local,
            pinion_axis_local=pinion_axis_local,
            rack_pitch_line_local_m=rack_pitch_line_local,
            rack_axis_local=rack_axis_local,
            rack_face_axis_local=rack_face_axis_local,
        )
    )
    pitch_delta = _sub(
        spec.pinion_origin_abs_m,
        plan.observable_point_abs_m,
    )
    pitch_delta = _sub(
        pitch_delta,
        _scale(spec.rack_axis_abs, _dot(pitch_delta, spec.rack_axis_abs)),
    )
    required_offset = _norm(pitch_delta)
    pitch_tolerance = max(0.0005, 1.5 * spec.max_support_drift_m)
    axis_tolerance = max(
        math.radians(0.25),
        0.5 * spec.max_off_axis_rotation_rad,
    )
    gate = conditional_transmission_gate(
        geometry_plugin_id="rack_pinion_mesh_geometry_v1",
        body_a_id=spec.pinion_body_id,
        body_b_id=spec.rack_body_id,
        endpoint_a_component_refs=pinion_refs,
        endpoint_b_component_refs=rack_refs,
        activation_coordinate="face_overlap_m",
        enter_threshold_si=0.0005,
        release_threshold_si=0.0001,
        proof_source="validator_rack_pinion_step_geometry_v1",
        geometry_parameters={
            "required_pitch_line_offset_m": required_offset,
            "pitch_line_offset_tolerance_m": pitch_tolerance,
            "axis_orthogonality_tolerance_rad": axis_tolerance,
            "face_axis_parallel_tolerance_rad": axis_tolerance,
            "pinion_face_width_m": face_width,
            "rack_face_width_m": face_width,
        },
    )
    interface = interface_declaration(
        interface_id="transmission-pinion-rack",
        classification=ANALYTICAL_TRANSMISSION,
        body_a_id=spec.pinion_body_id,
        body_b_id=spec.rack_body_id,
        model="finite_conditional_penalty_gear_mesh",
        source="trusted_rack_pinion_preflight_v1",
        transmission_gate=gate,
    )
    observation = RackPinionGeometry(
        required_pitch_line_offset_m=required_offset,
        pitch_line_offset_tolerance_m=pitch_tolerance,
        axis_orthogonality_tolerance_rad=axis_tolerance,
        face_axis_parallel_tolerance_rad=axis_tolerance,
    ).evaluate(
        time_s=0.0,
        pose=RackPinionPose(
            pinion_center_abs_m=spec.pinion_origin_abs_m,
            pinion_axis_abs=spec.pinion_axis_abs,
            pinion_face_width_m=face_width,
            rack_pitch_line_point_abs_m=plan.observable_point_abs_m,
            rack_axis_abs=spec.rack_axis_abs,
            rack_face_axis_abs=spec.pinion_axis_abs,
            rack_face_width_m=face_width,
        ),
    )
    assessment = assess_initial_functional_geometry(gate, observation)
    binding = build_functional_interface_geometry_binding(
        (interface,),
        collision_geometry,
    )
    passed = assessment["status"] in {"PASS", "WARNING"}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": collision_geometry.get("assembly_sha256"),
        "analytical_interface_count": 1,
        "registered_geometry_gate_count": 1,
        "ungated_analytical_interface_count": 0,
        "initially_nontransmitting_interface_count": (
            0 if assessment["transmission_active"] else 1
        ),
        "initially_nontransmitting_interfaces": (
            [] if assessment["transmission_active"]
            else ["transmission-pinion-rack"]
        ),
        "warning_count": int(assessment["status"] == "WARNING"),
        "failure_reasons": [],
        "interfaces": [
            {
                "interface_id": "transmission-pinion-rack",
                "interface_kind": "rack_pinion_mesh",
                "classification": "ANALYTICAL_TRANSMISSION",
                "body_a_id": spec.pinion_body_id,
                "body_b_id": spec.rack_body_id,
                "registered_geometry_gate": True,
                "gate_sha256": canonical_sha256(gate),
                **assessment,
            }
        ],
        "functional_interface_geometry_binding": binding,
        "physical_simulation_started": False,
    }


def _step_bicycle_torus_inventory(
    assembly: Path,
) -> tuple[dict[str, Any], ...]:
    """Read topology-bound toroidal STEP surfaces.

    A radius/placement record alone is not authoritative: it must be used by
    an ``ADVANCED_FACE`` belonging to one and only one STEP solid.  The solid
    ordering is the same canonical ``solid_N`` ordering consumed by the
    assembly contract and family collider.
    """

    text = assembly.read_text(encoding="latin-1")
    if "SI_UNIT(.MILLI.,.METRE.)" not in text:
        raise _fail(
            "BICYCLE_CHAIN_STEP_LENGTH_UNIT_UNSUPPORTED",
            "bicycle-chain pitch binding requires an explicit millimetre STEP unit",
        )
    records = {
        int(match.group(1)): match.group(2).strip()
        for match in re.finditer(
            r"#(\d+)\s*=\s*(.*?);",
            text,
            flags=re.DOTALL,
        )
    }

    face_surface: dict[int, int] = {}
    for entity_id, record in records.items():
        if not record.lstrip().startswith("ADVANCED_FACE"):
            continue
        references = [
            int(value)
            for value in re.findall(r"#(\d+)", record)
        ]
        if references:
            # ADVANCED_FACE(bounds, face_geometry, same_sense): face_geometry
            # is the final entity reference in the record.
            face_surface[entity_id] = references[-1]

    shell_faces: dict[int, frozenset[int]] = {}
    for entity_id, record in records.items():
        if record.lstrip().startswith("CLOSED_SHELL"):
            shell_faces[entity_id] = frozenset(
                int(value) for value in re.findall(r"#(\d+)", record)
            )

    solid_shells: list[tuple[int, int]] = []
    for entity_id, record in records.items():
        if not record.lstrip().startswith("MANIFOLD_SOLID_BREP"):
            continue
        references = [
            int(value)
            for value in re.findall(r"#(\d+)", record)
        ]
        if len(references) == 1:
            solid_shells.append((entity_id, references[0]))
    solid_shells.sort()

    surface_solids: dict[int, set[str]] = {}
    surface_face_counts: dict[tuple[int, str], int] = {}
    for solid_index, (_, shell_id) in enumerate(solid_shells, start=1):
        solid_id = f"solid_{solid_index}"
        for face_id in shell_faces.get(shell_id, ()):
            surface_id = face_surface.get(face_id)
            if surface_id is not None:
                surface_solids.setdefault(surface_id, set()).add(solid_id)
                key = (surface_id, solid_id)
                surface_face_counts[key] = surface_face_counts.get(key, 0) + 1

    def numbers(record: str) -> tuple[float, ...]:
        match = re.search(r"\(\s*(" + _STEP_NUMBER + r"(?:\s*,\s*" + _STEP_NUMBER + r")*)\s*\)", record)
        if match is None:
            return ()
        return tuple(
            float(item)
            for item in re.findall(_STEP_NUMBER, match.group(1))
        )

    inventory: list[dict[str, Any]] = []
    torus_pattern = re.compile(
        r"TOROIDAL_SURFACE\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*,\s*("
        + _STEP_NUMBER
        + r")\s*,\s*("
        + _STEP_NUMBER
        + r")\s*\)",
        flags=re.DOTALL,
    )
    placement_pattern = re.compile(
        r"AXIS2_PLACEMENT_3D\s*\(\s*'[^']*'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)",
        flags=re.DOTALL,
    )
    for entity_id, record in records.items():
        torus = torus_pattern.search(record)
        if torus is None:
            continue
        placement_id = int(torus.group(1))
        placement = placement_pattern.search(records.get(placement_id, ""))
        if placement is None:
            continue
        point_id = int(placement.group(1))
        axis_id = int(placement.group(2))
        point = numbers(records.get(point_id, ""))
        axis = numbers(records.get(axis_id, ""))
        if len(point) != 3 or len(axis) != 3:
            continue
        try:
            axis_abs = _unit(axis, f"STEP torus #{entity_id} axis")
        except MechanismValidationError:
            continue
        owning_solids = sorted(surface_solids.get(entity_id, ()))
        inventory.append(
            {
                "surface_entity_id": f"#{entity_id}",
                "placement_entity_id": f"#{placement_id}",
                "solid_id": (
                    owning_solids[0] if len(owning_solids) == 1 else None
                ),
                "topology_bound": len(owning_solids) == 1,
                "center_abs_m": [item * 0.001 for item in point],
                "axis_abs": list(axis_abs),
                "major_radius_m": float(torus.group(2)) * 0.001,
                "minor_radius_m": float(torus.group(3)) * 0.001,
                "exposed_face_count": (
                    surface_face_counts.get(
                        (entity_id, owning_solids[0]),
                        0,
                    )
                    if len(owning_solids) == 1
                    else 0
                ),
                "proof_source": "authoritative_step_toroidal_surface",
            }
        )
    return tuple(inventory)


def _circular_coverage_rad(angles: Sequence[float]) -> float:
    """Return the shortest circular arc containing every supplied angle."""

    normalized = sorted(float(value) % (2.0 * math.pi) for value in angles)
    if len(normalized) < 2:
        return 0.0
    gaps = [
        normalized[index + 1] - normalized[index]
        for index in range(len(normalized) - 1)
    ]
    gaps.append(normalized[0] + 2.0 * math.pi - normalized[-1])
    return 2.0 * math.pi - max(gaps)


def _visible_chain_wrap_metrics(
    points_m: Any,
    *,
    center_abs_m: Sequence[float],
    axis_abs: Sequence[float],
    pitch_radius_m: float,
    wrap_band_m: float,
) -> dict[str, Any]:
    """Measure visible-chain coverage around one fixed sprocket pitch circle."""

    import numpy as np

    points = np.asarray(points_m, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or not len(points):
        raise _fail(
            "BICYCLE_VISIBLE_CHAIN_SURFACE_EMPTY",
            "visible-chain STEP geometry has no finite surface samples",
        )
    if not np.isfinite(points).all():
        raise _fail(
            "BICYCLE_VISIBLE_CHAIN_SURFACE_INVALID",
            "visible-chain STEP geometry contains non-finite coordinates",
        )
    center = np.asarray(tuple(float(value) for value in center_abs_m))
    axis = np.asarray(_unit(axis_abs, "visible chain pitch axis"))
    deltas = points - center
    axial = deltas @ axis
    radial = deltas - axial[:, None] * axis[None, :]
    radial_norm = np.linalg.norm(radial, axis=1)
    near = np.flatnonzero(
        np.abs(radial_norm - float(pitch_radius_m))
        <= float(wrap_band_m)
    )
    if not len(near):
        return {
            "near_pitch_band_sample_count": 0,
            "wrap_angle_rad": 0.0,
            "chain_plane_center_abs_m": None,
            "pitch_plane_offset_m": None,
            "maximum_pitch_plane_deviation_m": None,
        }
    least_aligned = int(np.argmin(np.abs(axis)))
    reference = np.zeros(3, dtype=float)
    reference[least_aligned] = 1.0
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    selected = radial[near]
    angles = np.arctan2(selected @ basis_v, selected @ basis_u)
    near_axial = axial[near]
    plane_center = 0.5 * (
        float(np.min(near_axial)) + float(np.max(near_axial))
    )
    return {
        "near_pitch_band_sample_count": int(len(near)),
        "wrap_angle_rad": _circular_coverage_rad(angles.tolist()),
        "chain_plane_center_abs_m": (
            center + plane_center * axis
        ).tolist(),
        "pitch_plane_offset_m": abs(plane_center),
        # The midpoint is useful for diagnosis, but it can hide a chain whose
        # samples straddle the pitch plane symmetrically. Gate on the worst
        # sampled deviation instead.
        "maximum_pitch_plane_deviation_m": float(
            np.max(np.abs(near_axial))
        ),
    }


def _visible_chain_measurement_failures(
    measured: Mapping[str, Any],
    *,
    maximum_chain_sprocket_separation_m: float,
    minimum_wrap_angle_rad: float,
    minimum_wrap_sample_count: int,
    maximum_pitch_plane_offset_m: float,
    wrap_band_m: float,
) -> list[str]:
    """Evaluate public visible-chain thresholds against measured STEP data."""

    failures: list[str] = []
    for role in ("input", "output"):
        role_metrics = measured.get(role)
        if not isinstance(role_metrics, Mapping):
            failures.append(f"visible_chain_{role}_measurement_missing")
            continue
        try:
            separation_m = float(
                role_metrics["minimum_sprocket_separation_m"]
            )
            wrap_angle_rad = float(role_metrics["wrap_angle_rad"])
            sample_count = int(
                role_metrics["near_pitch_band_sample_count"]
            )
            plane_offset_m = float(role_metrics["pitch_plane_offset_m"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"visible_chain_{role}_measurement_invalid")
            continue
        values = (
            separation_m,
            wrap_angle_rad,
            plane_offset_m,
        )
        if not all(math.isfinite(value) for value in values):
            failures.append(f"visible_chain_{role}_measurement_invalid")
            continue
        if separation_m > maximum_chain_sprocket_separation_m:
            failures.append(
                f"visible_chain_{role}_separation_exceeded:"
                f"{separation_m!r}>{maximum_chain_sprocket_separation_m!r}"
            )
        if wrap_angle_rad < minimum_wrap_angle_rad:
            failures.append(
                f"visible_chain_{role}_wrap_insufficient:"
                f"{wrap_angle_rad!r}<{minimum_wrap_angle_rad!r}"
            )
        if sample_count < minimum_wrap_sample_count:
            failures.append(
                f"visible_chain_{role}_samples_insufficient:"
                f"{sample_count}<{minimum_wrap_sample_count}"
            )
        if plane_offset_m > maximum_pitch_plane_offset_m:
            failures.append(
                f"visible_chain_{role}_pitch_plane_offset_exceeded:"
                f"{plane_offset_m!r}>"
                f"{maximum_pitch_plane_offset_m!r}"
            )
    try:
        collision_radius_m = float(measured["audited_collision_radius_m"])
    except (KeyError, TypeError, ValueError):
        failures.append("visible_chain_collision_radius_measurement_missing")
    else:
        if (
            not math.isfinite(collision_radius_m)
            or collision_radius_m < 0.00025
            or collision_radius_m > wrap_band_m
        ):
            failures.append(
                "visible_chain_collision_radius_outside_wrap_band:"
                f"{collision_radius_m!r}>{wrap_band_m!r}"
            )
    return failures


def _measure_visible_chain_geometry(
    assembly: Path,
    *,
    chain_solid_ids: Sequence[str],
    input_solid_ids: Sequence[str],
    output_solid_ids: Sequence[str],
    input_center_abs_m: Sequence[float],
    output_center_abs_m: Sequence[float],
    input_axis_abs: Sequence[float],
    output_axis_abs: Sequence[float],
    input_pitch_radius_m: float,
    output_pitch_radius_m: float,
    wrap_band_m: float,
) -> dict[str, Any]:
    """Measure a context chain directly from the authoritative STEP solids."""

    import numpy as np
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    from .family_collision import _load_step_solids
    from .occt_collision_fidelity import shape_surface_points_m

    solids = _load_step_solids(assembly)

    def shapes_for(ids: Sequence[str], label: str) -> list[Any]:
        retained: list[Any] = []
        for solid_id in ids:
            match = re.fullmatch(r"solid_([1-9]\d*)", str(solid_id))
            index = int(match.group(1)) if match is not None else 0
            if index < 1 or index > len(solids):
                raise _fail(
                    "BICYCLE_VISIBLE_CHAIN_SOLID_BINDING_INVALID",
                    f"{label} references unavailable {solid_id!r}",
                )
            retained.append(solids[index - 1])
        if not retained:
            raise _fail(
                "BICYCLE_VISIBLE_CHAIN_SOLID_BINDING_INVALID",
                f"{label} has no STEP solids",
            )
        return retained

    chain_shapes = shapes_for(chain_solid_ids, "visible chain")
    sprocket_shapes = {
        "input": shapes_for(input_solid_ids, "input sprocket"),
        "output": shapes_for(output_solid_ids, "output sprocket"),
    }
    point_sets = [
        shape_surface_points_m(
            shape,
            maximum_spacing_m=0.002,
            maximum_point_count=8192,
        )
        for shape in chain_shapes
    ]
    points = np.unique(
        np.round(np.vstack(point_sets), decimals=12),
        axis=0,
    )
    from .bicycle_chain_path import (
        build_external_tangent_centerline,
        centerline_distances_m,
    )

    centerline = build_external_tangent_centerline(
        input_center_abs_m=input_center_abs_m,
        output_center_abs_m=output_center_abs_m,
        axis_abs=input_axis_abs,
        input_pitch_radius_m=input_pitch_radius_m,
        output_pitch_radius_m=output_pitch_radius_m,
    )
    centerline_distances = centerline_distances_m(
        points,
        centerline["centerline_points_abs_m"],
    )
    audited_collision_radius_m = max(
        0.00025,
        float(np.max(centerline_distances)),
    )

    def minimum_distance_m(targets: Sequence[Any]) -> float:
        distances: list[float] = []
        for chain_shape in chain_shapes:
            for target in targets:
                query = BRepExtrema_DistShapeShape(chain_shape, target)
                query.Perform()
                if not query.IsDone():
                    raise _fail(
                        "BICYCLE_VISIBLE_CHAIN_DISTANCE_FAILED",
                        "OCCT could not measure chain-to-sprocket separation",
                    )
                distances.append(float(query.Value()) * 0.001)
        return min(distances)

    result: dict[str, Any] = {
        "schema_version": "bicycle_visible_chain_geometry_v1",
        "source": "authoritative_step_solid_measurement",
        "chain_solid_ids": sorted(str(value) for value in chain_solid_ids),
        "surface_sample_count": int(len(points)),
        "audited_collision_radius_m": audited_collision_radius_m,
        "maximum_surface_to_external_tangent_centerline_m": (
            audited_collision_radius_m
        ),
        "external_tangent_centerline_sha256": canonical_sha256(centerline),
    }
    for role, center, axis, radius in (
        (
            "input",
            input_center_abs_m,
            input_axis_abs,
            input_pitch_radius_m,
        ),
        (
            "output",
            output_center_abs_m,
            output_axis_abs,
            output_pitch_radius_m,
        ),
    ):
        result[role] = {
            "minimum_sprocket_separation_m": minimum_distance_m(
                sprocket_shapes[role]
            ),
            **_visible_chain_wrap_metrics(
                points,
                center_abs_m=center,
                axis_abs=axis,
                pitch_radius_m=radius,
                wrap_band_m=wrap_band_m,
            ),
        }
    return result


def _declared_visible_chain_rows(
    context_part_rows: Sequence[Mapping[str, Any]],
    *,
    visible_chain_part_id: str,
) -> list[dict[str, Any]]:
    """Select only the context part that explicitly declares the visible chain."""

    return [
        dict(item)
        for item in context_part_rows
        if item.get("part_id") == visible_chain_part_id
    ]


def _bicycle_role_solid_ids(
    contract: Mapping[str, Any],
    *,
    role: str,
    part_id: str,
    occurrence_id: str,
    allowed_group_ids: frozenset[str],
) -> frozenset[str]:
    """Resolve one declared moving role to its exact contract STEP solids."""

    parts = [
        item
        for item in contract.get("parts", ())
        if isinstance(item, Mapping) and item.get("part_id") == part_id
    ]
    if len(parts) != 1:
        raise _fail(
            "BICYCLE_CHAIN_ROLE_PART_BINDING_INVALID",
            f"{role} must bind exactly one contract part {part_id!r}",
        )
    part = parts[0]
    group_id = part.get("rigid_group_id")
    if (
        part.get("occurrence_id") != occurrence_id
        or part.get("role") != "dynamic"
        or not isinstance(group_id, str)
        or group_id not in allowed_group_ids
    ):
        raise _fail(
            "BICYCLE_CHAIN_ROLE_PART_BINDING_INVALID",
            f"{role} part/occurrence must belong to its declared moving rigid group",
        )
    groups = [
        item
        for item in contract.get("rigid_groups", ())
        if isinstance(item, Mapping) and item.get("rigid_group_id") == group_id
    ]
    if (
        len(groups) != 1
        or part_id not in (groups[0].get("part_ids") or ())
    ):
        raise _fail(
            "BICYCLE_CHAIN_ROLE_GROUP_BINDING_INVALID",
            f"{role} contract rigid group does not own part {part_id!r}",
        )
    solid_ids = part.get("source_solid_ids")
    if (
        not isinstance(solid_ids, Sequence)
        or isinstance(solid_ids, (str, bytes))
        or not solid_ids
        or any(
            re.fullmatch(r"solid_[1-9]\d*", str(value)) is None
            for value in solid_ids
        )
    ):
        raise _fail(
            "BICYCLE_CHAIN_ROLE_SOLID_BINDING_INVALID",
            f"{role} moving part has no canonical STEP solid binding",
        )
    return frozenset(str(value) for value in solid_ids)


def _bind_bicycle_pitch_surface(
    inventory: Sequence[Mapping[str, Any]],
    *,
    role: str,
    allowed_solid_ids: frozenset[str],
    origin_abs_m: Sequence[float],
    required_radius_m: float,
    required_minor_radius_m: float,
    required_axis_abs: Sequence[float],
    radius_tolerance_m: float,
    center_tolerance_m: float,
    axis_tolerance_rad: float,
    minimum_exposed_face_count: int,
) -> tuple[dict[str, Any] | None, list[float]]:
    """Bind one pitch torus to its role, exact STEP solid and unoriented axis."""

    role_surfaces = [
        item
        for item in inventory
        if item.get("topology_bound") is True
        and item.get("solid_id") in allowed_solid_ids
    ]
    exact: list[Mapping[str, Any]] = []
    for item in role_surfaces:
        center = item.get("center_abs_m")
        axis = item.get("axis_abs")
        if (
            not isinstance(center, Sequence)
            or isinstance(center, (str, bytes))
            or len(center) != 3
            or not isinstance(axis, Sequence)
            or isinstance(axis, (str, bytes))
            or len(axis) != 3
        ):
            continue
        center_error_m = _norm(_sub(center, origin_abs_m))
        # A STEP toroidal surface carries an unoriented geometric axis line:
        # reversing its parameterisation does not reverse the mechanism.  Keep
        # transmission direction authority in the signed shaft/frame/DOF law.
        axis_dot = max(
            0.0,
            min(1.0, abs(_dot(axis, required_axis_abs))),
        )
        axis_error_rad = math.acos(axis_dot)
        if (
            abs(float(item["major_radius_m"]) - required_radius_m)
            <= radius_tolerance_m
            and abs(
                float(item["minor_radius_m"])
                - required_minor_radius_m
            )
            <= radius_tolerance_m
            and center_error_m <= center_tolerance_m
            and axis_error_rad <= axis_tolerance_rad
            and int(item.get("exposed_face_count", 0))
            >= minimum_exposed_face_count
        ):
            candidate = dict(item)
            candidate["role"] = role
            candidate["center_error_m"] = center_error_m
            candidate[
                "unoriented_surface_axis_parallel_error_rad"
            ] = axis_error_rad
            candidate["surface_axis_semantics"] = (
                "unoriented_surface_axis_parallel"
            )
            candidate["provides_transmission_direction_evidence"] = False
            candidate["required_minor_radius_m"] = required_minor_radius_m
            candidate["minimum_exposed_face_count"] = (
                minimum_exposed_face_count
            )
            exact.append(candidate)
    selected = (
        min(
            exact,
            key=lambda item: (
                float(item["center_error_m"]),
                str(item["surface_entity_id"]),
            ),
        )
        if exact
        else None
    )
    return selected, sorted(
        {
            round(float(item["major_radius_m"]), 12)
            for item in role_surfaces
        }
    )


def _bicycle_chain_geometry_evidence(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Bind the finite chain law to authored pitch circles and context path."""

    from .bicycle_chain_spec import (
        bicycle_chain_from_dict,
        resolve_bicycle_contract_bindings,
    )
    from .family_collision import _package_geometry_binding

    raw_family_spec = adapter.family_spec
    if isinstance(raw_family_spec, Mapping):
        raw_family_spec = {
            key: value
            for key, value in raw_family_spec.items()
            if key != "collision_geometry"
        }
    spec = bicycle_chain_from_dict(raw_family_spec)
    assembly, _, contract_path, contract = _package_geometry_binding(plan)
    try:
        assembly.relative_to(workspace)
        contract_path.relative_to(workspace)
    except ValueError as exc:
        raise _fail(
            "BICYCLE_CHAIN_PACKAGE_OUTSIDE_WORKSPACE",
            "bicycle-chain geometry evidence escaped the workspace",
        ) from exc
    inventory = _step_bicycle_torus_inventory(assembly)
    contract_bindings = resolve_bicycle_contract_bindings(
        spec,
        contract,
        assembly_contract_sha256=sha256_file(contract_path),
    )
    sprockets = {item.body_id: item for item in spec.sprockets}
    geometry_gate = spec.geometry_gate
    if (
        geometry_gate is None
        or set(sprockets) != {spec.input_body_id, spec.output_body_id}
    ):
        raise _fail(
            "BICYCLE_CHAIN_TYPED_GEOMETRY_BINDING_INVALID",
            "bicycle-chain v2 needs exactly one typed sprocket per moving body",
        )
    required_entities = {
        item.body_id: item
        for item in geometry_gate.required_step_entities
    }
    if set(required_entities) != {spec.input_body_id, spec.output_body_id}:
        raise _fail(
            "BICYCLE_CHAIN_TYPED_GEOMETRY_BINDING_INVALID",
            "geometry gate must bind one STEP entity to each moving body",
        )
    input_solid_ids = frozenset(
        contract_bindings["roles"]["input"]["source_solid_ids"]
    )
    output_solid_ids = frozenset(
        contract_bindings["roles"]["output"]["source_solid_ids"]
    )
    if input_solid_ids & output_solid_ids:
        raise _fail(
            "BICYCLE_CHAIN_ROLE_SOLID_BINDING_OVERLAP",
            "input and output moving roles cannot share a STEP solid",
        )
    input_sprocket = sprockets[spec.input_body_id]
    output_sprocket = sprockets[spec.output_body_id]
    input_entity = required_entities[spec.input_body_id]
    output_entity = required_entities[spec.output_body_id]
    for role, sprocket, entity, role_solids in (
        ("input", input_sprocket, input_entity, input_solid_ids),
        ("output", output_sprocket, output_entity, output_solid_ids),
    ):
        if (
            sprocket.step_solid_id != entity.step_solid_id
            or sprocket.step_solid_id not in role_solids
            or sprocket.pitch_loop_center_abs_m != entity.center_abs_m
            or sprocket.pitch_axis_abs != entity.axis_abs
            or sprocket.pitch_radius_m != entity.major_radius_m
            or sprocket.toroidal_minor_radius_m != entity.minor_radius_m
            or entity.surface_exposure_required is not True
            or entity.minimum_exposed_face_count < 1
        ):
            raise _fail(
                "BICYCLE_CHAIN_TYPED_GEOMETRY_BINDING_INVALID",
                f"{role} typed sprocket, geometry gate, body, and STEP solid disagree",
            )

    input_surface, input_found = _bind_bicycle_pitch_surface(
        inventory,
        role="input",
        allowed_solid_ids=frozenset({input_entity.step_solid_id}),
        origin_abs_m=input_entity.center_abs_m,
        required_radius_m=input_entity.major_radius_m,
        required_minor_radius_m=input_entity.minor_radius_m,
        required_axis_abs=input_entity.axis_abs,
        radius_tolerance_m=geometry_gate.pitch_radius_tolerance_m,
        center_tolerance_m=geometry_gate.pitch_center_tolerance_m,
        axis_tolerance_rad=geometry_gate.maximum_axis_parallel_error_rad,
        minimum_exposed_face_count=input_entity.minimum_exposed_face_count,
    )
    output_surface, output_found = _bind_bicycle_pitch_surface(
        inventory,
        role="output",
        allowed_solid_ids=frozenset({output_entity.step_solid_id}),
        origin_abs_m=output_entity.center_abs_m,
        required_radius_m=output_entity.major_radius_m,
        required_minor_radius_m=output_entity.minor_radius_m,
        required_axis_abs=output_entity.axis_abs,
        radius_tolerance_m=geometry_gate.pitch_radius_tolerance_m,
        center_tolerance_m=geometry_gate.pitch_center_tolerance_m,
        axis_tolerance_rad=geometry_gate.maximum_axis_parallel_error_rad,
        minimum_exposed_face_count=output_entity.minimum_exposed_face_count,
    )

    groups = {
        item.get("rigid_group_id"): item
        for item in contract.get("rigid_groups", ())
        if isinstance(item, Mapping)
        and isinstance(item.get("rigid_group_id"), str)
    }
    context_parts = [
        item
        for item in contract.get("parts", ())
        if isinstance(item, Mapping) and item.get("role") == "context"
    ]
    context_solid_ids: set[str] = set()
    context_part_rows: list[dict[str, Any]] = []
    centers_spanned = False
    for part in context_parts:
        group_id = part.get("rigid_group_id")
        part_id = part.get("part_id")
        group = groups.get(group_id)
        raw_solids = part.get("source_solid_ids")
        solids = (
            tuple(str(item) for item in raw_solids)
            if isinstance(raw_solids, Sequence)
            and not isinstance(raw_solids, (str, bytes))
            and raw_solids
            and all(
                re.fullmatch(r"solid_[1-9]\d*", str(item)) is not None
                for item in raw_solids
            )
            else ()
        )
        owned = bool(
            isinstance(group, Mapping)
            and isinstance(group.get("part_ids"), Sequence)
            and part_id in group["part_ids"]
        )
        if not owned or not solids:
            continue
        context_solid_ids.update(solids)
        bounds = part.get("declared_bounds_m")
        bounds_min = (
            bounds.get("min") if isinstance(bounds, Mapping) else None
        )
        bounds_max = (
            bounds.get("max") if isinstance(bounds, Mapping) else None
        )
        spans_centers = bool(
            isinstance(bounds_min, Sequence)
            and isinstance(bounds_max, Sequence)
            and len(bounds_min) == len(bounds_max) == 3
            and all(
                float(bounds_min[index])
                <= float(origin[index])
                <= float(bounds_max[index])
                for origin in (
                    spec.input_origin_abs_m,
                    spec.output_origin_abs_m,
                )
                for index in (0, 2)
            )
        )
        centers_spanned = centers_spanned or spans_centers
        context_part_rows.append(
            {
                "part_id": part_id,
                "occurrence_id": part.get("occurrence_id"),
                "rigid_group_id": group_id,
                "source_solid_ids": list(solids),
                "group_owns_part": owned,
                "centers_spanned_xz": spans_centers,
            }
        )
    context_solid_ids = frozenset(context_solid_ids)
    context_physical_overlap = context_solid_ids & (
        input_solid_ids | output_solid_ids
    )
    # A context part cannot substitute for either governed moving sprocket.
    # The uniquely named visible-chain part is nevertheless measured below as
    # an independent functional-geometry gate and is required for activation.
    # Exclude overlap from the optional torus diagnostics so those diagnostics
    # cannot impersonate the physical input/output pitch surfaces.
    context_solid_ids = context_solid_ids - context_physical_overlap

    def diagnostic_context_surface(
        *,
        role: str,
        entity: Any,
    ) -> tuple[dict[str, Any] | None, list[float]]:
        candidates = []
        found = []
        for item in inventory:
            if item.get("solid_id") not in context_solid_ids:
                continue
            major = float(item.get("major_radius_m", math.nan))
            found.append(major)
            center = item.get("center_abs_m")
            axis = item.get("axis_abs")
            if (
                not isinstance(center, Sequence)
                or len(center) != 3
                or not isinstance(axis, Sequence)
                or len(axis) != 3
            ):
                continue
            axis_dot = max(
                0.0,
                min(1.0, abs(_dot(axis, entity.axis_abs))),
            )
            if (
                item.get("topology_bound") is True
                and abs(major - entity.major_radius_m)
                <= geometry_gate.pitch_radius_tolerance_m
                and abs(float(center[0]) - entity.center_abs_m[0])
                <= geometry_gate.pitch_center_tolerance_m
                and abs(float(center[2]) - entity.center_abs_m[2])
                <= geometry_gate.pitch_center_tolerance_m
                and math.acos(axis_dot)
                <= geometry_gate.maximum_axis_parallel_error_rad
            ):
                candidate = dict(item)
                candidate["role"] = role
                candidate["diagnostic_only"] = True
                candidate["affects_physical_gate"] = False
                candidates.append(candidate)
        selected = (
            min(
                candidates,
                key=lambda item: (
                    abs(float(item["center_abs_m"][1])),
                    str(item["surface_entity_id"]),
                ),
            )
            if candidates
            else None
        )
        return selected, sorted(set(round(value, 12) for value in found))

    input_context_surface, input_context_found = diagnostic_context_surface(
        role="context_input_wrap",
        entity=input_entity,
    )
    output_context_surface, output_context_found = diagnostic_context_surface(
        role="context_output_wrap",
        entity=output_entity,
    )
    context_group_ids = {
        str(item["rigid_group_id"])
        for item in context_part_rows
    }
    context_frames = [
        {
            "frame_id": item.get("frame_id"),
            "rigid_group_id": item.get("rigid_group_id"),
            "origin_m": item.get("origin_m"),
        }
        for item in contract.get("frames", ())
        if isinstance(item, Mapping)
        and item.get("rigid_group_id") in context_group_ids
    ]
    path_bound = bool(
        context_part_rows
        and context_frames
        and centers_spanned
        and not context_physical_overlap
    )
    visible_chain_rows = _declared_visible_chain_rows(
        context_part_rows,
        visible_chain_part_id=geometry_gate.visible_chain_part_id,
    )
    visible_chain_declared = bool(visible_chain_rows)
    visible_chain_geometry: dict[str, Any] = {
        "schema_version": "bicycle_visible_chain_geometry_v1",
        "declared": visible_chain_declared,
        "required_part_id": geometry_gate.visible_chain_part_id,
        "passed": False,
        "status": "FAIL",
        "binding_meaning": (
            "the required visible chain is measured from its authoritative "
            "STEP solids and must be seated around both "
            "governed sprocket pitch circles"
        ),
    }
    visible_chain_failures: list[str] = (
        []
        if visible_chain_declared
        else ["required_visible_chain_not_declared"]
    )
    if visible_chain_declared:
        if len(visible_chain_rows) != 1:
            visible_chain_failures.append(
                "visible_chain_part_binding_invalid:"
                f"required_exactly_one={geometry_gate.visible_chain_part_id!r},"
                f"found={len(visible_chain_rows)}"
            )
        else:
            chain_solid_ids = frozenset(
                str(value)
                for value in visible_chain_rows[0]["source_solid_ids"]
            )
            chain_overlap = chain_solid_ids & (
                input_solid_ids | output_solid_ids
            )
            if chain_overlap:
                visible_chain_failures.append(
                    "visible_chain_physical_solid_overlap:"
                    f"{sorted(chain_overlap)!r}"
                )
            else:
                measured = _measure_visible_chain_geometry(
                    assembly,
                    chain_solid_ids=tuple(chain_solid_ids),
                    input_solid_ids=tuple(input_solid_ids),
                    output_solid_ids=tuple(output_solid_ids),
                    input_center_abs_m=input_entity.center_abs_m,
                    output_center_abs_m=output_entity.center_abs_m,
                    input_axis_abs=input_entity.axis_abs,
                    output_axis_abs=output_entity.axis_abs,
                    input_pitch_radius_m=input_entity.major_radius_m,
                    output_pitch_radius_m=output_entity.major_radius_m,
                    wrap_band_m=geometry_gate.visible_chain_wrap_band_m,
                )
                visible_chain_geometry.update(measured)
                visible_chain_failures.extend(
                    _visible_chain_measurement_failures(
                        measured,
                        maximum_chain_sprocket_separation_m=(
                            geometry_gate.maximum_chain_sprocket_separation_m
                        ),
                        minimum_wrap_angle_rad=(
                            geometry_gate.minimum_visible_chain_wrap_angle_rad
                        ),
                        minimum_wrap_sample_count=(
                            geometry_gate.minimum_visible_chain_wrap_sample_count
                        ),
                        maximum_pitch_plane_offset_m=(
                            geometry_gate.maximum_pitch_plane_offset_m
                        ),
                        wrap_band_m=geometry_gate.visible_chain_wrap_band_m,
                    )
                )
                visible_chain_geometry["passed"] = not visible_chain_failures
                visible_chain_geometry["status"] = (
                    "PASS" if not visible_chain_failures else "FAIL"
                )
    visible_chain_geometry["failure_reasons"] = list(
        visible_chain_failures
    )
    visible_chain_geometry["thresholds"] = {
        "maximum_chain_sprocket_separation_m": (
            geometry_gate.maximum_chain_sprocket_separation_m
        ),
        "wrap_band_m": geometry_gate.visible_chain_wrap_band_m,
        "minimum_wrap_angle_rad": (
            geometry_gate.minimum_visible_chain_wrap_angle_rad
        ),
        "minimum_wrap_sample_count": (
            geometry_gate.minimum_visible_chain_wrap_sample_count
        ),
        "maximum_pitch_plane_offset_m": (
            geometry_gate.maximum_pitch_plane_offset_m
        ),
    }
    alignment_error_m = (
        abs(
            float(input_surface["center_abs_m"][1])
            - float(output_surface["center_abs_m"][1])
        )
        if input_surface is not None and output_surface is not None
        else math.inf
    )
    failures: list[str] = []
    if input_surface is None:
        failures.append(
            "input_pitch_surface_missing:"
            f"required={spec.input_pitch_radius_m:.12g},found={input_found!r}"
        )
    if output_surface is None:
        failures.append(
            "output_pitch_surface_missing:"
            f"required={spec.output_pitch_radius_m:.12g},found={output_found!r}"
        )
    if alignment_error_m > geometry_gate.maximum_pitch_plane_offset_m:
        failures.append(
            "sprocket_axial_alignment_exceeded:"
            f"{alignment_error_m!r}>"
            f"{geometry_gate.maximum_pitch_plane_offset_m!r}"
        )
    failures.extend(visible_chain_failures)
    passed = not failures
    gate_parameters = {
        "input_pitch_radius_m": spec.input_pitch_radius_m,
        "output_pitch_radius_m": spec.output_pitch_radius_m,
        "pitch_radius_tolerance_m": geometry_gate.pitch_radius_tolerance_m,
        "pitch_center_tolerance_m": geometry_gate.pitch_center_tolerance_m,
        "maximum_pitch_plane_offset_m": (
            geometry_gate.maximum_pitch_plane_offset_m
        ),
        "maximum_unoriented_surface_axis_parallel_error_rad": (
            geometry_gate.maximum_axis_parallel_error_rad
        ),
        "surface_axis_semantics": "unoriented_surface_axis_parallel",
        "surface_axis_provides_transmission_direction_evidence": False,
        "input_step_solid_id": input_entity.step_solid_id,
        "output_step_solid_id": output_entity.step_solid_id,
        "input_pitch_minor_radius_m": input_entity.minor_radius_m,
        "output_pitch_minor_radius_m": output_entity.minor_radius_m,
        "minimum_input_exposed_face_count": (
            input_entity.minimum_exposed_face_count
        ),
        "minimum_output_exposed_face_count": (
            output_entity.minimum_exposed_face_count
        ),
        "contract_role_binding_sha256": contract_bindings[
            "binding_sha256"
        ],
        "visible_chain_declared": visible_chain_declared,
        "visible_chain_geometry_passed": visible_chain_geometry["passed"],
        "visible_chain_part_id": geometry_gate.visible_chain_part_id,
        "maximum_chain_sprocket_separation_m": (
            geometry_gate.maximum_chain_sprocket_separation_m
        ),
        "visible_chain_wrap_band_m": (
            geometry_gate.visible_chain_wrap_band_m
        ),
        "minimum_visible_chain_wrap_angle_rad": (
            geometry_gate.minimum_visible_chain_wrap_angle_rad
        ),
        "minimum_visible_chain_wrap_sample_count": (
            geometry_gate.minimum_visible_chain_wrap_sample_count
        ),
    }
    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "failure_reasons": failures,
        "assembly_file": str(assembly),
        "assembly_sha256": sha256_file(assembly),
        "assembly_contract_sha256": sha256_file(contract_path),
        "contract_role_bindings": contract_bindings,
        "contract_role_binding_sha256": contract_bindings[
            "binding_sha256"
        ],
        "input_pitch_surface": input_surface,
        "output_pitch_surface": output_surface,
        "input_context_wrap_surface": input_context_surface,
        "output_context_wrap_surface": output_context_surface,
        "input_local_torus_radii_m": input_found,
        "output_local_torus_radii_m": output_found,
        "input_context_torus_radii_m": input_context_found,
        "output_context_torus_radii_m": output_context_found,
        "input_role_solid_ids": sorted(input_solid_ids),
        "output_role_solid_ids": sorted(output_solid_ids),
        "axial_alignment_error_m": (
            alignment_error_m if math.isfinite(alignment_error_m) else None
        ),
        "authored_chain_path_binding": {
            "bound": path_bound,
            "diagnostic_only": False,
            "affects_physical_gate": True,
            "required_for_physical_pass": True,
            "visible_chain_geometry_required_for_physical_pass": True,
            "authoritative_visible_chain_gate": "visible_chain_geometry",
            "parts": context_part_rows,
            "frames": context_frames,
            "group_ids": sorted(context_group_ids),
            "centers_spanned": centers_spanned,
            "direct_rigid_body_chain_simulated": False,
            "finite_compliant_transmission_binding_required": True,
            "runtime_role": (
                "authoritative STEP width and seating source for the "
                "validator-owned moving chain-path state"
            ),
            "static_context_alone_can_satisfy_moving_chain_obligation": False,
            "source_solid_ids": sorted(context_solid_ids),
            "overlapping_physical_source_solid_ids": sorted(
                context_physical_overlap
            ),
            "disjoint_from_physical": not context_physical_overlap,
            "cannot_substitute_for_drive_geometry": True,
            "surface_diagnostics_required_for_pass": False,
            "input_wrap_surface_diagnostic_found": (
                input_context_surface is not None
            ),
            "output_wrap_surface_diagnostic_found": (
                output_context_surface is not None
            ),
        },
        "visible_chain_geometry": visible_chain_geometry,
        "geometry_gate": {
            "geometry_plugin_id": "bicycle_chain_pitch_geometry_v1",
            "parameters": gate_parameters,
            "gate_sha256": canonical_sha256(gate_parameters),
            "initially_active": passed,
        },
        "toroidal_surface_inventory": list(inventory),
    }


def _bicycle_ground_roll_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    """Bind the rolling radii and contact plane to the submitted STEP solids."""

    from .bicycle_ground_roll_spec import bicycle_ground_roll_from_dict
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )

    raw = adapter.family_spec
    if isinstance(raw, Mapping):
        raw = {
            key: item
            for key, item in raw.items()
            if key != "collision_geometry"
        }
    spec = bicycle_ground_roll_from_dict(raw)
    collision = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    wheel_tolerance_m = 0.008
    contact_height_tolerance_m = 0.006

    def wheel_record(
        body_id: str,
        origin: Sequence[float],
        radius_m: float,
    ) -> dict[str, Any]:
        records = _component_axis_records(
            collision,
            workspace=workspace,
            body_id=body_id,
            shaft_origin_abs_m=origin,
            shaft_axis_abs=spec.wheel_axis_abs,
        )
        observed = max(
            (float(item["radial_max_m"]) for item in records),
            default=-math.inf,
        )
        radius_error = abs(observed - radius_m)
        contact_height_error = abs(
            float(origin[2]) - radius_m - spec.ground_plane_height_m
        )
        return {
            "body_id": body_id,
            "declared_radius_m": radius_m,
            "step_outer_radius_m": observed if math.isfinite(observed) else None,
            "radius_error_m": radius_error if math.isfinite(radius_error) else None,
            "contact_height_error_m": contact_height_error,
            "radius_tolerance_m": wheel_tolerance_m,
            "contact_height_tolerance_m": contact_height_tolerance_m,
            "component_count": len(records),
            "passed": (
                bool(records)
                and radius_error <= wheel_tolerance_m
                and contact_height_error <= contact_height_tolerance_m
            ),
        }

    rear = wheel_record(
        spec.rear_wheel_body_id,
        spec.rear_axle_origin_abs_m,
        spec.rear_wheel_radius_m,
    )
    front = wheel_record(
        spec.front_wheel_body_id,
        spec.front_axle_origin_abs_m,
        spec.front_wheel_radius_m,
    )
    ground_records = _component_axis_records(
        collision,
        workspace=workspace,
        body_id=spec.ground_body_id,
        shaft_origin_abs_m=(0.0, 0.0, 0.0),
        shaft_axis_abs=spec.ground_normal_abs,
    )
    axle_x = (
        float(spec.rear_axle_origin_abs_m[0]),
        float(spec.front_axle_origin_abs_m[0]),
    )
    axle_y = (
        float(spec.rear_axle_origin_abs_m[1]),
        float(spec.front_axle_origin_abs_m[1]),
    )
    surface_candidates = [
        item
        for item in ground_records
        if abs(
            float(item["world_aabb_max_m"][2])
            - spec.ground_plane_height_m
        ) <= contact_height_tolerance_m
        and float(item["world_aabb_min_m"][0]) <= min(axle_x) - 0.02
        and float(item["world_aabb_max_m"][0]) >= max(axle_x) + 0.02
        and float(item["world_aabb_min_m"][1]) <= min(axle_y) - 0.03
        and float(item["world_aabb_max_m"][1]) >= max(axle_y) + 0.03
    ]
    ground_body = next(
        item for item in spec.bodies if item.body_id == spec.ground_body_id
    )
    fixture_bound = len(ground_body.group_ids) >= 2
    ground_bound = bool(surface_candidates)
    wheels_bound = rear["passed"] is True and front["passed"] is True
    failures: list[str] = []
    if rear["passed"] is not True:
        failures.append("rear_wheel_step_radius_or_contact_height_invalid")
    if front["passed"] is not True:
        failures.append("front_wheel_step_radius_or_contact_height_invalid")
    if not ground_bound:
        failures.append("ground_contact_plane_not_found_in_step")
    if not fixture_bound:
        failures.append("visible_upright_fixture_group_missing")
    passed = not failures
    interfaces = [
        {
            "interface_id": f"{body_id}-ground-native-contact",
            "interface_kind": "finite_coulomb_wheel_ground_contact",
            "classification": "NATIVE_PHYSICAL_CONTACT",
            "body_a_id": spec.ground_body_id,
            "body_b_id": body_id,
            "registered_geometry_gate": True,
            "status": "PASS" if item["passed"] and ground_bound else "FAIL",
            "transmission_active": item["passed"] and ground_bound,
            "failure_reasons": [],
        }
        for body_id, item in (
            (spec.rear_wheel_body_id, rear),
            (spec.front_wheel_body_id, front),
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": collision["assembly_sha256"],
        "analytical_interface_count": 0,
        "registered_geometry_gate_count": 2,
        "ungated_analytical_interface_count": 0,
        "initially_nontransmitting_interface_count": 0 if passed else len(failures),
        "initially_nontransmitting_interfaces": (
            [] if passed else [item["interface_id"] for item in interfaces if item["status"] != "PASS"]
        ),
        "warning_count": 0,
        "failure_reasons": failures,
        "interfaces": interfaces,
        "wheel_geometry_bound_to_step": wheels_bound,
        "ground_plane_bound_to_step": ground_bound,
        "visible_upright_fixture_bound": fixture_bound,
        "ground_plane_height_m": spec.ground_plane_height_m,
        "rear_wheel_geometry": rear,
        "front_wheel_geometry": front,
        "ground_surface_component_refs": [
            item["component_ref"] for item in surface_candidates
        ],
        "physical_simulation_started": False,
    }


def _bicycle_chain_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    workspace: Path,
) -> dict[str, Any]:
    from .bicycle_chain_runtime import (
        _bicycle_prescribed_swept_envelope_samples,
    )
    from .bicycle_chain_spec import bicycle_chain_from_dict
    from .bicycle_chain_path import build_moving_chain_path_contract
    from .family_collision import (
        family_collision_geometry_for_static_preflight,
    )
    from .physical_legality import (
        audit_bicycle_sprocket_contact_component_geometry,
        audit_prescribed_swept_envelope,
    )

    raw_family_spec = adapter.family_spec
    if isinstance(raw_family_spec, Mapping):
        raw_family_spec = {
            key: value
            for key, value in raw_family_spec.items()
            if key != "collision_geometry"
        }
    spec = bicycle_chain_from_dict(raw_family_spec)
    geometry = _bicycle_chain_geometry_evidence(
        plan,
        adapter,
        workspace=workspace,
    )
    collision = family_collision_geometry_for_static_preflight(
        adapter,
        expected_bodies=spec.bodies,
    )
    prescribed_samples = _bicycle_prescribed_swept_envelope_samples(
        spec
    )
    contact_geometry = (
        audit_bicycle_sprocket_contact_component_geometry(
            collision,
            build_moving_chain_path_contract(
                spec,
                geometry["visible_chain_geometry"],
            ),
            prescribed_samples[0]["body_poses"],
            workspace_root=workspace,
        )
    )
    swept_envelope = audit_prescribed_swept_envelope(
        collision,
        prescribed_samples,
        workspace_root=workspace,
    )
    geometry_failures = list(geometry["failure_reasons"])
    if contact_geometry.get("passed") is not True:
        geometry_failures.append(
            "sprocket_contact_component_geometry_invalid"
        )
    if swept_envelope.get("passed") is not True:
        geometry_failures.append("prescribed_swept_envelope_intersection")
    passed = (
        geometry["passed"] is True
        and contact_geometry.get("passed") is True
        and swept_envelope.get("passed") is True
    )
    geometry = {
        **geometry,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "failure_reasons": geometry_failures,
        "sprocket_contact_component_geometry_audit": contact_geometry,
        "prescribed_swept_envelope_preflight": swept_envelope,
    }
    interface = {
        "interface_id": "transmission-input-sprocket-chain-output-sprocket",
        "interface_kind": "bicycle_chain_sprocket_drive",
        "classification": "ANALYTICAL_TRANSMISSION",
        "body_a_id": spec.input_body_id,
        "body_b_id": spec.output_body_id,
        "registered_geometry_gate": True,
        "gate_sha256": geometry["geometry_gate"]["gate_sha256"],
        "status": "PASS" if passed else "FAIL",
        "transmission_active": passed,
        "failure_reasons": list(geometry_failures),
        "endpoint_a": {"component_refs": []},
        "endpoint_b": {"component_refs": []},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": spec.mechanism_kind,
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_sha256": geometry["assembly_sha256"],
        "analytical_interface_count": 1,
        "registered_geometry_gate_count": 1,
        "ungated_analytical_interface_count": 0,
        "initially_nontransmitting_interface_count": 0 if passed else 1,
        "initially_nontransmitting_interfaces": (
            [] if passed else [interface["interface_id"]]
        ),
        "warning_count": 0,
        "failure_reasons": list(geometry_failures),
        "interfaces": [interface],
        # ATTENTION AI EDITORS (added 2026-08-07, CURRENT binding rule,
        # NOT a legacy note — do not ignore): this report's shape is a
        # live two-ended protocol with worker.py load_execution_request
        # (functional_geometry_evidence extraction). Changing the layout
        # here REQUIRES updating the extraction there in the same change,
        # and vice versa. Flat reports (e.g. ground-roll) must keep
        # top-level "passed". Origin: commit f1cd3a8.
        "bicycle_chain_geometry": geometry,
        "physical_simulation_started": False,
    }


def build_initial_functional_geometry_preflight(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    worker_backend: str,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Audit static functional-interface geometry before any physics job.

    The report consumes only the intake-bound mechanism specification and the
    hash-bound, per-solid STEP collider inventory.  It deliberately has no
    PyChrono dependency, so a failed interface cannot consume a physics slot.
    """

    workspace = Path(workspace_root).resolve()
    family_spec = (
        adapter.family_spec
        if isinstance(adapter.family_spec, Mapping)
        else {}
    )
    mechanism_kind = family_spec.get("mechanism_kind")
    if mechanism_kind is not None:
        from .family_plugins import family_spec_plugin_for_kind
        from .joint_realization import validate_adapter_joint_realizations

        plugin = family_spec_plugin_for_kind(mechanism_kind)
        if plugin is not None and plugin.worker_backend == worker_backend:
            package = Path(plan.package_dir).resolve()
            assembly_contract = read_json(
                package / "assembly_contract.json"
            )
            joint_report = validate_adapter_joint_realizations(
                step_path=package / "assembly.step",
                assembly_contract=assembly_contract,
                family_spec=family_spec,
                adapter=adapter,
                plugin=plugin,
                capsule_id=plan.capsule_id,
                has_revolute_claim=(
                    getattr(plan, "family", None) == "revolute"
                    or bool(family_spec.get("revolute_joints"))
                ),
            )
            if joint_report is not None:
                failures = [
                    str(failure.get("code") or "JOINT_REALIZATION_FAILED")
                    for proof in joint_report.get("proofs") or ()
                    for failure in proof.get("failures") or ()
                    if isinstance(proof, Mapping)
                    and isinstance(failure, Mapping)
                ]
                return {
                    "schema_version": SCHEMA_VERSION,
                    "status": joint_report["status"],
                    "passed": joint_report["passed"],
                    "capsule_id": plan.capsule_id,
                    "mechanism_kind": mechanism_kind,
                    "package_binding_sha256": plan.package_binding_sha256,
                    "analytical_interface_count": 0,
                    "registered_geometry_gate_count": len(
                        joint_report["proofs"]
                    ),
                    "ungated_analytical_interface_count": 0,
                    "initially_nontransmitting_interface_count": 0,
                    "initially_nontransmitting_interfaces": [],
                    "warning_count": 0,
                    "failure_reasons": failures,
                    "interfaces": [],
                    "joint_realization": joint_report,
                    "physical_simulation_started": False,
                }
    if worker_backend == "bicycle_ground_roll_v1":
        return _bicycle_ground_roll_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    if worker_backend == "bicycle_chain_sprocket_drive_v2":
        return _bicycle_chain_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    if worker_backend == "rack_pinion_v1":
        return _rack_pinion_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    if worker_backend == "spur_gear_clock_v1":
        return _clock_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    if worker_backend == "multi_output_shaft_coupling_v1":
        return _multi_output_shaft_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    if worker_backend == "prismatic_detent_latch_v1":
        return _detent_slider_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    if worker_backend == "click_pen_cam_latch_v1":
        return _click_pen_preflight(
            plan,
            adapter,
            workspace=workspace,
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "NOT_REQUIRED",
        "passed": True,
        "capsule_id": plan.capsule_id,
        "mechanism_kind": (
            adapter.family_spec.get("mechanism_kind")
            if isinstance(adapter.family_spec, Mapping)
            else plan.family
        ),
        "package_binding_sha256": plan.package_binding_sha256,
        "analytical_interface_count": 0,
        "registered_geometry_gate_count": 0,
        "ungated_analytical_interface_count": 0,
        "initially_nontransmitting_interface_count": 0,
        "initially_nontransmitting_interfaces": [],
        "warning_count": 0,
        "failure_reasons": [],
        "interfaces": [],
        "physical_simulation_started": False,
    }
