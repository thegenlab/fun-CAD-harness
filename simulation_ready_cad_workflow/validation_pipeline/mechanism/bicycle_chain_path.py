"""Validator-owned moving evidence for the reduced bicycle-chain model.

The physical solver keeps the approved finite spring/damper transmission.  It
does not simulate individual links or tooth contact.  This module supplies the
missing auditable geometry state: one closed external-tangent chain path,
world-stationary flexible regions around the two sprockets, and material phase
derived from the exact generalized coordinates used by the force law.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .io import canonical_sha256
from .models import MechanismValidationError


MOVING_CHAIN_PATH_SCHEMA = "bicycle_moving_chain_path_contract_v1"
MOVING_CHAIN_STATE_SCHEMA = "bicycle_moving_chain_path_state_v1"


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _add(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(a) + float(b) for a, b in zip(left, right))


def _sub(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return tuple(float(a) - float(b) for a, b in zip(left, right))


def _scale(
    value: Sequence[float], factor: float
) -> tuple[float, float, float]:
    return tuple(float(item) * float(factor) for item in value)


def _cross(
    left: Sequence[float], right: Sequence[float]
) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", f"{label} must be a 3-vector")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", f"{label} is not finite")
    magnitude = _norm(result)
    if magnitude <= 1.0e-12:
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", f"{label} is zero")
    return _scale(result, 1.0 / magnitude)


def _lerp(
    left: Sequence[float], right: Sequence[float], fraction: float
) -> tuple[float, float, float]:
    return _add(left, _scale(_sub(right, left), fraction))


def _arc_points(
    *,
    center: Sequence[float],
    radius_m: float,
    basis_u: Sequence[float],
    basis_v: Sequence[float],
    start_angle_rad: float,
    sweep_rad: float,
    interval_count: int,
) -> list[tuple[float, float, float]]:
    return [
        _add(
            center,
            _scale(
                _add(
                    _scale(basis_u, math.cos(start_angle_rad + sweep_rad * index / interval_count)),
                    _scale(basis_v, math.sin(start_angle_rad + sweep_rad * index / interval_count)),
                ),
                radius_m,
            ),
        )
        for index in range(interval_count + 1)
    ]


def build_external_tangent_centerline(
    *,
    input_center_abs_m: Sequence[float],
    output_center_abs_m: Sequence[float],
    axis_abs: Sequence[float],
    input_pitch_radius_m: float,
    output_pitch_radius_m: float,
    straight_interval_count: int = 32,
    arc_interval_count: int = 64,
) -> dict[str, Any]:
    """Return a deterministic closed open-belt centerline in world space."""

    input_center = tuple(float(item) for item in input_center_abs_m)
    output_center = tuple(float(item) for item in output_center_abs_m)
    if len(input_center) != 3 or len(output_center) != 3:
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", "sprocket centers must be 3-vectors")
    axis = _unit(axis_abs, "chain pitch axis")
    center_delta = _sub(output_center, input_center)
    axial_delta = _dot(center_delta, axis)
    planar_delta = _sub(center_delta, _scale(axis, axial_delta))
    center_distance = _norm(planar_delta)
    input_radius = float(input_pitch_radius_m)
    output_radius = float(output_pitch_radius_m)
    if (
        not all(math.isfinite(value) for value in (input_radius, output_radius))
        or min(input_radius, output_radius) <= 0.0
        or center_distance <= abs(input_radius - output_radius) + 1.0e-9
        or abs(axial_delta) > 5.0e-4
        or not 4 <= straight_interval_count <= 256
        or not 8 <= arc_interval_count <= 512
    ):
        raise _fail(
            "BICYCLE_CHAIN_PATH_INVALID",
            "sprocket centers, radii, planes, or sampling counts cannot form a finite external-tangent path",
        )
    basis_u = _scale(planar_delta, 1.0 / center_distance)
    basis_v = _unit(_cross(axis, basis_u), "chain path plane basis")
    radial_u = (input_radius - output_radius) / center_distance
    radial_v = math.sqrt(max(0.0, 1.0 - radial_u * radial_u))
    angle = math.atan2(radial_v, radial_u)
    plus = _add(_scale(basis_u, radial_u), _scale(basis_v, radial_v))
    minus = _sub(_scale(basis_u, radial_u), _scale(basis_v, radial_v))
    input_plus = _add(input_center, _scale(plus, input_radius))
    output_plus = _add(output_center, _scale(plus, output_radius))
    output_minus = _add(output_center, _scale(minus, output_radius))
    input_minus = _add(input_center, _scale(minus, input_radius))

    points: list[tuple[float, float, float]] = [
        _lerp(input_plus, output_plus, index / straight_interval_count)
        for index in range(straight_interval_count + 1)
    ]
    output_arc = _arc_points(
        center=output_center,
        radius_m=output_radius,
        basis_u=basis_u,
        basis_v=basis_v,
        start_angle_rad=angle,
        sweep_rad=-2.0 * angle,
        interval_count=arc_interval_count,
    )
    points.extend(output_arc[1:])
    lower = [
        _lerp(output_minus, input_minus, index / straight_interval_count)
        for index in range(straight_interval_count + 1)
    ]
    points.extend(lower[1:])
    input_sweep = 2.0 * math.pi - 2.0 * angle
    input_arc = _arc_points(
        center=input_center,
        radius_m=input_radius,
        basis_u=basis_u,
        basis_v=basis_v,
        start_angle_rad=-angle,
        sweep_rad=-input_sweep,
        interval_count=arc_interval_count,
    )
    closure_gap_m = _norm(_sub(input_arc[-1], input_plus))
    if closure_gap_m > 1.0e-10:
        raise _fail(
            "BICYCLE_CHAIN_PATH_INVALID",
            "input sprocket wrap does not close at the upper tangent",
        )
    points.extend(input_arc[1:-1])
    points.append(points[0])
    lengths = [_norm(_sub(right, left)) for left, right in zip(points, points[1:])]
    if not lengths or min(lengths) <= 1.0e-12:
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", "chain centerline has a zero edge")
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return {
        "centerline_points_abs_m": [list(point) for point in points],
        "edge_lengths_m": lengths,
        "cumulative_lengths_m": cumulative,
        "total_length_m": cumulative[-1],
        "input_wrap_angle_rad": input_sweep,
        "output_wrap_angle_rad": 2.0 * angle,
        "tangent_family": "external_open_belt",
        "path_direction": "input_plus_to_output_plus_then_output_wrap",
        "material_motion_per_phase_sign": -1,
        "basis_u_abs": list(basis_u),
        "basis_v_abs": list(basis_v),
    }


def centerline_distances_m(
    points_abs_m: Any,
    centerline_points_abs_m: Sequence[Sequence[float]],
) -> Any:
    """Vectorized minimum point-to-polyline distance used for STEP binding."""

    import numpy as np

    points = np.asarray(points_abs_m, dtype=float)
    centerline = np.asarray(centerline_points_abs_m, dtype=float)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", "surface points are invalid")
    if centerline.ndim != 2 or centerline.shape[1:] != (3,) or len(centerline) < 3:
        raise _fail("BICYCLE_CHAIN_PATH_INVALID", "centerline points are invalid")
    starts = centerline[:-1]
    deltas = centerline[1:] - starts
    squared = np.sum(deltas * deltas, axis=1)
    minimum = np.full(len(points), np.inf, dtype=float)
    for start in range(0, len(points), 1_024):
        block = points[start : start + 1_024]
        relative = block[:, None, :] - starts[None, :, :]
        fractions = np.sum(relative * deltas[None, :, :], axis=2) / squared[None, :]
        fractions = np.clip(fractions, 0.0, 1.0)
        closest = starts[None, :, :] + fractions[:, :, None] * deltas[None, :, :]
        distances = np.linalg.norm(block[:, None, :] - closest, axis=2)
        minimum[start : start + len(block)] = np.min(distances, axis=1)
    return minimum


def _split_centerline_at_zone_boundaries(
    centerline: Mapping[str, Any],
    boundaries: Sequence[tuple[Sequence[float], float]],
) -> dict[str, Any]:
    """Insert exact sphere crossings so no edge straddles a flex boundary."""

    raw_points = centerline["centerline_points_abs_m"]
    points: list[tuple[float, float, float]] = []
    for raw_left, raw_right in zip(raw_points, raw_points[1:]):
        left = tuple(float(value) for value in raw_left)
        right = tuple(float(value) for value in raw_right)
        delta = _sub(right, left)
        squared_length = _dot(delta, delta)
        fractions = [0.0, 1.0]
        for raw_center, raw_radius in boundaries:
            center = tuple(float(value) for value in raw_center)
            radius = float(raw_radius)
            offset = _sub(left, center)
            linear = 2.0 * _dot(offset, delta)
            constant = _dot(offset, offset) - radius * radius
            discriminant = linear * linear - 4.0 * squared_length * constant
            if discriminant < 0.0:
                continue
            root = math.sqrt(max(0.0, discriminant))
            for fraction in (
                (-linear - root) / (2.0 * squared_length),
                (-linear + root) / (2.0 * squared_length),
            ):
                if 1.0e-12 < fraction < 1.0 - 1.0e-12:
                    fractions.append(fraction)
        for fraction in sorted(set(round(value, 15) for value in fractions))[:-1]:
            point = _lerp(left, right, fraction)
            if not points or _norm(_sub(point, points[-1])) > 1.0e-12:
                points.append(point)
    final_point = tuple(float(value) for value in raw_points[-1])
    if not points or _norm(_sub(final_point, points[-1])) > 1.0e-12:
        points.append(final_point)
    lengths = [
        _norm(_sub(right, left))
        for left, right in zip(points, points[1:])
    ]
    cumulative = [0.0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    return {
        **centerline,
        "centerline_points_abs_m": [list(point) for point in points],
        "edge_lengths_m": lengths,
        "cumulative_lengths_m": cumulative,
        "total_length_m": cumulative[-1],
    }


def build_moving_chain_path_contract(
    spec: Any,
    visible_chain_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the moving evidence path to approved spec and measured STEP width."""

    try:
        collision_radius = float(
            visible_chain_geometry["audited_collision_radius_m"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(
            "BICYCLE_CHAIN_PATH_WIDTH_MISSING",
            "visible-chain STEP evidence has no measured collision radius",
        ) from exc
    if not 0.00025 <= collision_radius <= spec.geometry_gate.visible_chain_wrap_band_m:
        raise _fail(
            "BICYCLE_CHAIN_PATH_WIDTH_INVALID",
            "measured visible-chain collision radius is outside its public wrap band",
        )
    input_contact_radius = float(
        spec.input_sprocket.toroidal_minor_radius_m
    )
    output_contact_radius = float(
        spec.output_sprocket.toroidal_minor_radius_m
    )
    input_contact_component_ids = list(
        spec.input_sprocket.contact_collision_component_ids
    )
    output_contact_component_ids = list(
        spec.output_sprocket.contact_collision_component_ids
    )
    input_approved_contact_component_ids = list(
        spec.input_sprocket.approved_contact_collision_component_ids
    )
    output_approved_contact_component_ids = list(
        spec.output_sprocket.approved_contact_collision_component_ids
    )
    if (
        input_contact_component_ids[
            : len(input_approved_contact_component_ids)
        ]
        != input_approved_contact_component_ids
        or output_contact_component_ids[
            : len(output_approved_contact_component_ids)
        ]
        != output_approved_contact_component_ids
    ):
        raise _fail(
            "BICYCLE_CHAIN_PATH_REGION_INVALID",
            "sprocket contact bindings do not preserve their approved seed IDs",
        )
    maximum_separation = float(
        spec.geometry_gate.maximum_chain_sprocket_separation_m
    )
    input_zone_radius = (
        spec.input_pitch_radius_m
        + input_contact_radius
        + collision_radius
        + maximum_separation
    )
    output_zone_radius = (
        spec.output_pitch_radius_m
        + output_contact_radius
        + collision_radius
        + maximum_separation
    )
    input_penetration_cap = input_contact_radius + collision_radius
    output_penetration_cap = output_contact_radius + collision_radius
    centerline = _split_centerline_at_zone_boundaries(
        build_external_tangent_centerline(
            input_center_abs_m=spec.input_sprocket.pitch_loop_center_abs_m,
            output_center_abs_m=spec.output_sprocket.pitch_loop_center_abs_m,
            axis_abs=spec.axis_abs,
            input_pitch_radius_m=spec.input_pitch_radius_m,
            output_pitch_radius_m=spec.output_pitch_radius_m,
        ),
        (
            (
                spec.input_sprocket.pitch_loop_center_abs_m,
                input_zone_radius,
            ),
            (
                spec.output_sprocket.pitch_loop_center_abs_m,
                output_zone_radius,
            ),
        ),
    )
    edge_regions: list[str] = []
    points = centerline["centerline_points_abs_m"]
    for left, right in zip(points, points[1:]):
        midpoint = _lerp(left, right, 0.5)
        edge_samples = (left, midpoint, right)
        if all(
            _norm(
                _sub(
                    point,
                    spec.input_sprocket.pitch_loop_center_abs_m,
                )
            )
            <= input_zone_radius + 1.0e-12
            for point in edge_samples
        ):
            edge_regions.append("flex.input_sprocket")
        elif all(
            _norm(
                _sub(
                    point,
                    spec.output_sprocket.pitch_loop_center_abs_m,
                )
            )
            <= output_zone_radius + 1.0e-12
            for point in edge_samples
        ):
            edge_regions.append("flex.output_sprocket")
        else:
            edge_regions.append("rigid.tangent_span")
    if not {"flex.input_sprocket", "flex.output_sprocket", "rigid.tangent_span"}.issubset(edge_regions):
        raise _fail(
            "BICYCLE_CHAIN_PATH_REGION_INVALID",
            "fixed spatial regions do not cover both sprockets and the tangent spans",
        )
    marker_count = 64
    core = {
        "schema_version": MOVING_CHAIN_PATH_SCHEMA,
        "model": "reduced_phase_locked_chain_path_evidence",
        "force_model": "finite_compliant_force_pair",
        "input_body_id": spec.input_body_id,
        "output_body_id": spec.output_body_id,
        "input_sprocket_solid_id": spec.input_sprocket.step_solid_id,
        "output_sprocket_solid_id": spec.output_sprocket.step_solid_id,
        "input_sprocket_contact_component_ids": (
            input_contact_component_ids
        ),
        "output_sprocket_contact_component_ids": (
            output_contact_component_ids
        ),
        "input_sprocket_approved_contact_component_ids": (
            input_approved_contact_component_ids
        ),
        "output_sprocket_approved_contact_component_ids": (
            output_approved_contact_component_ids
        ),
        "input_center_abs_m": list(spec.input_sprocket.pitch_loop_center_abs_m),
        "output_center_abs_m": list(spec.output_sprocket.pitch_loop_center_abs_m),
        "axis_abs": list(spec.axis_abs),
        "input_pitch_radius_m": spec.input_pitch_radius_m,
        "output_pitch_radius_m": spec.output_pitch_radius_m,
        "input_sprocket_contact_radius_m": input_contact_radius,
        "output_sprocket_contact_radius_m": output_contact_radius,
        "maximum_chain_sprocket_separation_m": maximum_separation,
        "chain_collision_radius_m": collision_radius,
        "maximum_matching_sprocket_penetration_m": max(
            input_penetration_cap,
            output_penetration_cap,
        ),
        "centerline_points_abs_m": points,
        "edge_region_ids": edge_regions,
        "edge_lengths_m": centerline["edge_lengths_m"],
        "cumulative_lengths_m": centerline["cumulative_lengths_m"],
        "total_length_m": centerline["total_length_m"],
        "input_wrap_angle_rad": centerline["input_wrap_angle_rad"],
        "output_wrap_angle_rad": centerline["output_wrap_angle_rad"],
        "tangent_family": centerline["tangent_family"],
        "path_direction": centerline["path_direction"],
        "material_motion_per_phase_sign": centerline[
            "material_motion_per_phase_sign"
        ],
        "flex_zones": [
            {
                "zone_id": "flex.input_sprocket",
                "center_abs_m": list(spec.input_sprocket.pitch_loop_center_abs_m),
                "radius_m": input_zone_radius,
                "stationary_frame": "authored_world",
                "allowed_body_id": spec.input_body_id,
                "allowed_solid_id": spec.input_sprocket.step_solid_id,
                "allowed_component_ids": input_contact_component_ids,
                "approved_seed_component_ids": (
                    input_approved_contact_component_ids
                ),
                "maximum_matching_sprocket_penetration_m": (
                    input_penetration_cap
                ),
            },
            {
                "zone_id": "flex.output_sprocket",
                "center_abs_m": list(spec.output_sprocket.pitch_loop_center_abs_m),
                "radius_m": output_zone_radius,
                "stationary_frame": "authored_world",
                "allowed_body_id": spec.output_body_id,
                "allowed_solid_id": spec.output_sprocket.step_solid_id,
                "allowed_component_ids": output_contact_component_ids,
                "approved_seed_component_ids": (
                    output_approved_contact_component_ids
                ),
                "maximum_matching_sprocket_penetration_m": (
                    output_penetration_cap
                ),
            },
        ],
        "phase_coordinate": (
            "q_chain_m = 0.5 * (r_input * theta_input + "
            "r_output * theta_output)"
        ),
        "force_coordinate": (
            "e_chain_m = r_input * theta_input - r_output * theta_output"
        ),
        "visual_marker_count": marker_count,
        "visual_marker_spacing_m": centerline["total_length_m"] / marker_count,
        "model_boundaries": [
            "no individual-link dynamics",
            "no individual tooth-contact dynamics",
            "no full flexible-body chain physics",
        ],
        "source_visible_chain_geometry_sha256": canonical_sha256(
            visible_chain_geometry
        ),
    }
    return {**core, "contract_sha256": canonical_sha256(core)}


def build_moving_chain_path_state(
    contract: Mapping[str, Any],
    *,
    input_angle_rad: float,
    output_angle_rad: float,
    coordinate_error_m: float,
) -> dict[str, Any]:
    """Create one compact chain state from the force-law coordinates."""

    contract_sha256 = contract.get("contract_sha256")
    core_contract = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    if (
        contract.get("schema_version") != MOVING_CHAIN_PATH_SCHEMA
        or contract_sha256 != canonical_sha256(core_contract)
    ):
        raise _fail("BICYCLE_CHAIN_PATH_CONTRACT_CHANGED", "chain path contract digest is invalid")
    input_angle = float(input_angle_rad)
    output_angle = float(output_angle_rad)
    error = float(coordinate_error_m)
    input_displacement = float(contract["input_pitch_radius_m"]) * input_angle
    output_displacement = float(contract["output_pitch_radius_m"]) * output_angle
    expected_error = input_displacement - output_displacement
    if not math.isclose(error, expected_error, rel_tol=0.0, abs_tol=1.0e-10):
        raise _fail(
            "BICYCLE_CHAIN_PATH_FORCE_COORDINATE_MISMATCH",
            "chain path state does not use the force-law coordinate",
        )
    phase_unwrapped = 0.5 * (input_displacement + output_displacement)
    total_length = float(contract["total_length_m"])
    core = {
        "schema_version": MOVING_CHAIN_STATE_SCHEMA,
        "contract_sha256": contract_sha256,
        "input_angle_rad": input_angle,
        "output_angle_rad": output_angle,
        "input_surface_displacement_m": input_displacement,
        "output_surface_displacement_m": output_displacement,
        "force_coordinate_error_m": error,
        "phase_unwrapped_m": phase_unwrapped,
        "phase_modulo_m": phase_unwrapped % total_length,
        "input_phase_residual_m": phase_unwrapped - input_displacement,
        "output_phase_residual_m": phase_unwrapped - output_displacement,
    }
    return {**core, "state_sha256": canonical_sha256(core)}


def audit_moving_chain_path_states(
    contract: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify closure, region assignment, force binding, hashes, and cadence."""

    try:
        core_contract = {
            key: value for key, value in contract.items() if key != "contract_sha256"
        }
        if (
            contract.get("schema_version") != MOVING_CHAIN_PATH_SCHEMA
            or contract.get("contract_sha256") != canonical_sha256(core_contract)
        ):
            raise _fail("BICYCLE_CHAIN_PATH_CONTRACT_CHANGED", "chain path contract digest is invalid")
        points = contract["centerline_points_abs_m"]
        regions = contract["edge_region_ids"]
        if (
            not isinstance(points, list)
            or len(points) < 16
            or not isinstance(regions, list)
            or len(regions) != len(points) - 1
            or _norm(_sub(points[0], points[-1])) > 1.0e-10
            or set(regions)
            != {"flex.input_sprocket", "flex.output_sprocket", "rigid.tangent_span"}
        ):
            raise _fail("BICYCLE_CHAIN_PATH_TOPOLOGY_INVALID", "chain path is open or its fixed regions are incomplete")
        collision_radius = float(contract["chain_collision_radius_m"])
        input_contact_radius = float(
            contract["input_sprocket_contact_radius_m"]
        )
        output_contact_radius = float(
            contract["output_sprocket_contact_radius_m"]
        )
        maximum_separation = float(
            contract["maximum_chain_sprocket_separation_m"]
        )
        input_contact_component_ids = contract[
            "input_sprocket_contact_component_ids"
        ]
        output_contact_component_ids = contract[
            "output_sprocket_contact_component_ids"
        ]
        input_approved_contact_component_ids = contract[
            "input_sprocket_approved_contact_component_ids"
        ]
        output_approved_contact_component_ids = contract[
            "output_sprocket_approved_contact_component_ids"
        ]
        if (
            not isinstance(input_contact_component_ids, list)
            or not input_contact_component_ids
            or len(set(input_contact_component_ids))
            != len(input_contact_component_ids)
            or not all(
                isinstance(item, str) and item
                for item in input_contact_component_ids
            )
            or not isinstance(output_contact_component_ids, list)
            or not output_contact_component_ids
            or len(set(output_contact_component_ids))
            != len(output_contact_component_ids)
            or not all(
                isinstance(item, str) and item
                for item in output_contact_component_ids
            )
            or not isinstance(input_approved_contact_component_ids, list)
            or not input_approved_contact_component_ids
            or input_contact_component_ids[
                : len(input_approved_contact_component_ids)
            ]
            != input_approved_contact_component_ids
            or not isinstance(output_approved_contact_component_ids, list)
            or not output_approved_contact_component_ids
            or output_contact_component_ids[
                : len(output_approved_contact_component_ids)
            ]
            != output_approved_contact_component_ids
        ):
            raise _fail(
                "BICYCLE_CHAIN_PATH_REGION_INVALID",
                "sprocket contact component bindings are missing or invalid",
            )
        expected_input_zone_radius = (
            float(contract["input_pitch_radius_m"])
            + input_contact_radius
            + collision_radius
            + maximum_separation
        )
        expected_output_zone_radius = (
            float(contract["output_pitch_radius_m"])
            + output_contact_radius
            + collision_radius
            + maximum_separation
        )
        expected_input_penetration = input_contact_radius + collision_radius
        expected_output_penetration = output_contact_radius + collision_radius
        expected_path = _split_centerline_at_zone_boundaries(
            build_external_tangent_centerline(
                input_center_abs_m=contract["input_center_abs_m"],
                output_center_abs_m=contract["output_center_abs_m"],
                axis_abs=contract["axis_abs"],
                input_pitch_radius_m=float(contract["input_pitch_radius_m"]),
                output_pitch_radius_m=float(contract["output_pitch_radius_m"]),
            ),
            (
                (
                    contract["input_center_abs_m"],
                    expected_input_zone_radius,
                ),
                (
                    contract["output_center_abs_m"],
                    expected_output_zone_radius,
                ),
            ),
        )
        for field in (
            "centerline_points_abs_m",
            "edge_lengths_m",
            "cumulative_lengths_m",
        ):
            actual_values = contract[field]
            expected_values = expected_path[field]
            if len(actual_values) != len(expected_values):
                raise _fail(
                    "BICYCLE_CHAIN_PATH_GEOMETRY_MISMATCH",
                    f"chain path {field} length differs from the rederived path",
                )
            actual_flat = (
                [item for row in actual_values for item in row]
                if field == "centerline_points_abs_m"
                else actual_values
            )
            expected_flat = (
                [item for row in expected_values for item in row]
                if field == "centerline_points_abs_m"
                else expected_values
            )
            if any(
                not math.isclose(
                    float(actual),
                    float(expected),
                    rel_tol=0.0,
                    abs_tol=1.0e-10,
                )
                for actual, expected in zip(actual_flat, expected_flat)
            ):
                raise _fail(
                    "BICYCLE_CHAIN_PATH_GEOMETRY_MISMATCH",
                    f"chain path {field} differs from the rederived external tangent",
                )
        for field in (
            "total_length_m",
            "input_wrap_angle_rad",
            "output_wrap_angle_rad",
        ):
            if not math.isclose(
                float(contract[field]),
                float(expected_path[field]),
                rel_tol=0.0,
                abs_tol=1.0e-10,
            ):
                raise _fail(
                    "BICYCLE_CHAIN_PATH_GEOMETRY_MISMATCH",
                    f"chain path {field} differs from the rederived external tangent",
                )
        for field in (
            "tangent_family",
            "path_direction",
            "material_motion_per_phase_sign",
        ):
            if contract[field] != expected_path[field]:
                raise _fail(
                    "BICYCLE_CHAIN_PATH_GEOMETRY_MISMATCH",
                    f"chain path {field} differs from the validator model",
                )
        maximum_matching_penetration = float(
            contract["maximum_matching_sprocket_penetration_m"]
        )
        expected_maximum_matching_penetration = max(
            expected_input_penetration,
            expected_output_penetration,
        )
        if (
            not math.isfinite(maximum_matching_penetration)
            or maximum_matching_penetration <= 0.0
            or not math.isclose(
                maximum_matching_penetration,
                expected_maximum_matching_penetration,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise _fail(
                "BICYCLE_CHAIN_PATH_GEOMETRY_MISMATCH",
                "matching sprocket penetration cap is not bound to the public sprocket and chain geometry",
            )
        zones = contract.get("flex_zones")
        if not isinstance(zones, list) or len(zones) != 2:
            raise _fail(
                "BICYCLE_CHAIN_PATH_REGION_INVALID",
                "chain path must declare exactly two fixed sprocket zones",
            )
        zones_by_id = {
            str(zone.get("zone_id")): zone
            for zone in zones
            if isinstance(zone, Mapping)
        }
        input_zone = zones_by_id.get("flex.input_sprocket")
        output_zone = zones_by_id.get("flex.output_sprocket")
        if (
            not isinstance(input_zone, Mapping)
            or not isinstance(output_zone, Mapping)
            or input_zone.get("stationary_frame") != "authored_world"
            or output_zone.get("stationary_frame") != "authored_world"
            or input_zone.get("center_abs_m")
            != contract.get("input_center_abs_m")
            or output_zone.get("center_abs_m")
            != contract.get("output_center_abs_m")
            or input_zone.get("allowed_body_id") != contract.get("input_body_id")
            or output_zone.get("allowed_body_id") != contract.get("output_body_id")
            or input_zone.get("allowed_solid_id")
            != contract.get("input_sprocket_solid_id")
            or output_zone.get("allowed_solid_id")
            != contract.get("output_sprocket_solid_id")
            or input_zone.get("allowed_component_ids")
            != input_contact_component_ids
            or output_zone.get("allowed_component_ids")
            != output_contact_component_ids
            or input_zone.get("approved_seed_component_ids")
            != input_approved_contact_component_ids
            or output_zone.get("approved_seed_component_ids")
            != output_approved_contact_component_ids
            or not math.isclose(
                float(input_zone.get("radius_m")),
                expected_input_zone_radius,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(output_zone.get("radius_m")),
                expected_output_zone_radius,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(
                    input_zone.get(
                        "maximum_matching_sprocket_penetration_m"
                    )
                ),
                expected_input_penetration,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                float(
                    output_zone.get(
                        "maximum_matching_sprocket_penetration_m"
                    )
                ),
                expected_output_penetration,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise _fail(
                "BICYCLE_CHAIN_PATH_REGION_INVALID",
                "fixed sprocket zones are not bound to their exact bodies and solids",
            )
        expected_regions: list[str] = []
        for left, right in zip(points, points[1:]):
            midpoint = _lerp(left, right, 0.5)
            edge_samples = (left, midpoint, right)
            if all(
                _norm(_sub(point, input_zone["center_abs_m"]))
                <= float(input_zone["radius_m"]) + 1.0e-12
                for point in edge_samples
            ):
                expected_regions.append("flex.input_sprocket")
            elif all(
                _norm(_sub(point, output_zone["center_abs_m"]))
                <= float(output_zone["radius_m"]) + 1.0e-12
                for point in edge_samples
            ):
                expected_regions.append("flex.output_sprocket")
            else:
                expected_regions.append("rigid.tangent_span")
        if regions != expected_regions:
            raise _fail(
                "BICYCLE_CHAIN_PATH_REGION_INVALID",
                "chain edge regions differ from the fixed authored-world zones",
            )
        if not isinstance(samples, Sequence) or not samples:
            raise _fail("BICYCLE_CHAIN_PATH_STATE_MISSING", "physical trace has no chain path states")
        maximum_force_mismatch = 0.0
        maximum_phase_residual = 0.0
        previous_time: float | None = None
        for index, sample in enumerate(samples):
            state = sample.get("moving_chain_path_state") if isinstance(sample, Mapping) else None
            if not isinstance(state, Mapping):
                raise _fail("BICYCLE_CHAIN_PATH_STATE_MISSING", f"sample {index} has no moving chain path state")
            state_core = {key: value for key, value in state.items() if key != "state_sha256"}
            if (
                state.get("schema_version") != MOVING_CHAIN_STATE_SCHEMA
                or state.get("contract_sha256") != contract.get("contract_sha256")
                or state.get("state_sha256") != canonical_sha256(state_core)
            ):
                raise _fail("BICYCLE_CHAIN_PATH_STATE_CHANGED", f"sample {index} chain path state digest is invalid")
            expected = build_moving_chain_path_state(
                contract,
                input_angle_rad=float(sample["input_crank_angle_rad"]),
                output_angle_rad=float(sample["rear_drive_wheel_angle_rad"]),
                coordinate_error_m=float(sample["chain_coordinate_error_m"]),
            )
            if dict(state) != expected:
                raise _fail("BICYCLE_CHAIN_PATH_STATE_MISMATCH", f"sample {index} chain path state differs from solver coordinates")
            mismatch = abs(
                float(state["input_surface_displacement_m"])
                - float(state["output_surface_displacement_m"])
                - float(state["force_coordinate_error_m"])
            )
            maximum_force_mismatch = max(maximum_force_mismatch, mismatch)
            maximum_phase_residual = max(
                maximum_phase_residual,
                abs(float(state["input_phase_residual_m"])),
                abs(float(state["output_phase_residual_m"])),
            )
            time_s = float(sample.get("time_s", 0.0))
            phase = float(state["phase_unwrapped_m"])
            if (
                not math.isfinite(time_s)
                or not math.isfinite(phase)
                or (
                    previous_time is not None
                    and time_s <= previous_time
                )
            ):
                raise _fail("BICYCLE_CHAIN_PATH_PHASE_DISCONTINUITY", f"sample {index} chain phase cadence is not monotonic in time")
            previous_time = time_s
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, MechanismValidationError):
            raise
        raise _fail("BICYCLE_CHAIN_PATH_EVIDENCE_INVALID", f"chain path evidence is malformed: {exc}") from exc
    return {
        "schema_version": "bicycle_moving_chain_path_audit_v1",
        "status": "PASS",
        "passed": True,
        "sample_count": len(samples),
        "closed_path_verified": True,
        "external_tangent_family_verified": True,
        "external_tangent_geometry_rederived": True,
        "wrap_angles_rederived": True,
        "fixed_spatial_flex_zones_verified": True,
        "force_coordinate_consistency_verified": True,
        "phase_samples_finite_and_time_monotonic_verified": True,
        "maximum_force_coordinate_mismatch_m": maximum_force_mismatch,
        "maximum_phase_surface_residual_m": maximum_phase_residual,
        "contract_sha256": contract["contract_sha256"],
    }


__all__ = [
    "MOVING_CHAIN_PATH_SCHEMA",
    "MOVING_CHAIN_STATE_SCHEMA",
    "audit_moving_chain_path_states",
    "build_external_tangent_centerline",
    "build_moving_chain_path_contract",
    "build_moving_chain_path_state",
    "centerline_distances_m",
]
