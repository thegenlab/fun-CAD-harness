from __future__ import annotations

from copy import deepcopy
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .io import (
    canonical_sha256,
    ensure_inside,
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .models import MechanismValidationError


DEFAULT_INTERPENETRATION_TOLERANCE_M = 0.00035
DEFAULT_RUNTIME_INTERPENETRATION_TOLERANCE_M = 0.00005
MAX_RETAINED_PENETRATION_EVENTS = 12



def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def runtime_interpenetration_tolerance_m(
    collision_geometry: Mapping[str, Any],
) -> float:
    """Return the validator-owned saved-pose runtime penetration budget."""

    raw = collision_geometry.get(
        "runtime_interpenetration_tolerance_m",
        collision_geometry.get(
            "interpenetration_tolerance_m",
            DEFAULT_INTERPENETRATION_TOLERANCE_M,
        ),
    )
    try:
        tolerance = float(raw)
    except (TypeError, ValueError) as exc:
        raise _fail(
            "RUNTIME_INTERPENETRATION_TOLERANCE_INVALID",
            "runtime interpenetration tolerance is not numeric",
        ) from exc
    envelope = float(collision_geometry.get("collision_envelope_m") or 0.0)
    minimum = max(1.0e-6, 4.0 * envelope)
    if not minimum <= tolerance <= DEFAULT_INTERPENETRATION_TOLERANCE_M:
        raise _fail(
            "RUNTIME_INTERPENETRATION_TOLERANCE_INVALID",
            "runtime interpenetration tolerance must cover the collision "
            "envelope without falling back to the coarse legacy home budget",
        )
    return tolerance


def audit_dispatch_home_interpenetration(
    collision_geometry: Mapping[str, Any],
    initial_functional_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the runtime-safe home budget outside exact flexible endpoints."""

    home_audit = collision_geometry.get("home_interpenetration_audit")
    if not isinstance(home_audit, Mapping):
        raise _fail(
            "HOME_INTERPENETRATION_AUDIT_REQUIRED",
            "dispatch has no hash-bound home interpenetration evidence",
        )
    events = home_audit.get("contact_events")
    if not isinstance(events, list):
        raise _fail(
            "HOME_INTERPENETRATION_EVENTS_REQUIRED",
            "home interpenetration evidence has no component-level events",
        )
    allowed_flexible_pairs: set[frozenset[str]] = set()
    for interface in initial_functional_geometry.get("interfaces") or ():
        if (
            not isinstance(interface, Mapping)
            or interface.get("classification") != "ANALYTICAL_TRANSMISSION"
        ):
            continue
        endpoint_a = interface.get("endpoint_a")
        endpoint_b = interface.get("endpoint_b")
        if not isinstance(endpoint_a, Mapping) or not isinstance(
            endpoint_b,
            Mapping,
        ):
            continue
        for left in endpoint_a.get(
            "local_collision_component_refs",
            endpoint_a.get("component_refs"),
        ) or ():
            for right in endpoint_b.get(
                "local_collision_component_refs",
                endpoint_b.get("component_refs"),
            ) or ():
                allowed_flexible_pairs.add(
                    frozenset((str(left), str(right)))
                )
    tolerance = runtime_interpenetration_tolerance_m(collision_geometry)
    illegal: list[dict[str, Any]] = []
    exempted_count = 0
    for raw_event in events:
        if not isinstance(raw_event, Mapping):
            continue
        solid_pair = raw_event.get("solid_pair")
        if not isinstance(solid_pair, list) or len(solid_pair) != 2:
            continue
        depth = float(raw_event.get("penetration_depth_m") or 0.0)
        if depth <= tolerance:
            continue
        if frozenset(str(item) for item in solid_pair) in allowed_flexible_pairs:
            exempted_count += 1
            continue
        illegal.append(dict(raw_event))
    illegal.sort(
        key=lambda item: float(item.get("penetration_depth_m") or 0.0),
        reverse=True,
    )
    return {
        "schema_version": "dispatch_home_interpenetration_v1",
        "status": "PASS" if not illegal else "FAIL",
        "passed": not illegal,
        "runtime_tolerance_m": tolerance,
        "contact_event_count": len(events),
        "validator_step_bound_flexible_pair_count": len(
            allowed_flexible_pairs
        ),
        "flexible_pair_event_exemption_count": exempted_count,
        "illegal_event_count": len(illegal),
        "worst_events": illegal[:MAX_RETAINED_PENETRATION_EVENTS],
        "flexible_regions_remain_subject_to_post_simulation_step_audit": True,
        "source_home_audit_method": home_audit.get("method"),
    }


def _component_path(component: Mapping[str, Any], workspace: Path) -> Path:
    relative = component.get("mesh") or component.get("derived")
    if not isinstance(relative, str):
        raise _fail(
            "INTERPENETRATION_AUDIT_COMPONENT_PATH_MISSING",
            "collision component has no workspace-relative mesh path",
        )
    path = ensure_inside(workspace, relative)
    expected_sha256 = component.get("sha256") or component.get("derived_sha256")
    if not path.is_file() or path.is_symlink():
        raise _fail(
            "INTERPENETRATION_AUDIT_COMPONENT_MISSING",
            f"collision component is missing: {relative}",
        )
    if not isinstance(expected_sha256, str) or sha256_file(path) != expected_sha256:
        raise _fail(
            "INTERPENETRATION_AUDIT_COMPONENT_CHANGED",
            f"collision component no longer matches its hash: {relative}",
        )
    return path


def _pose_matrix(pose: Mapping[str, Any]) -> Any:
    try:
        import numpy as np

        position = [float(value) for value in pose["position_abs_m"]]
        quaternion = [float(value) for value in pose["rotation_abs_quat_wxyz"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(
            "INTERPENETRATION_AUDIT_POSE_INVALID",
            "saved physical body pose is incomplete or non-numeric",
        ) from exc
    if len(position) != 3 or len(quaternion) != 4:
        raise _fail(
            "INTERPENETRATION_AUDIT_POSE_INVALID",
            "saved physical body pose has the wrong dimension",
        )
    w_value, x_value, y_value, z_value = quaternion
    magnitude = (
        w_value * w_value
        + x_value * x_value
        + y_value * y_value
        + z_value * z_value
    ) ** 0.5
    if magnitude <= 1.0e-12:
        raise _fail(
            "INTERPENETRATION_AUDIT_POSE_INVALID",
            "saved physical body quaternion is zero",
        )
    w_value, x_value, y_value, z_value = (
        value / magnitude for value in quaternion
    )
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(
        (
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
        ),
        dtype=float,
    )
    matrix[:3, 3] = position
    return matrix


def _load_collision_manager(
    collision_geometry: Mapping[str, Any],
    *,
    workspace: Path,
) -> tuple[
    dict[str, Any],
    dict[str, tuple[str, str, str]],
    dict[str, bool],
    dict[str, Any],
    dict[str, Any],
]:
    try:
        import trimesh
        from trimesh.collision import CollisionManager
    except Exception as exc:
        raise _fail(
            "INTERPENETRATION_AUDIT_BACKEND_UNAVAILABLE",
            "trimesh with the python-fcl collision backend is required",
        ) from exc

    bodies = collision_geometry.get("bodies")
    if not isinstance(bodies, list) or len(bodies) < 2:
        raise _fail(
            "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
            "collision geometry must contain at least two bodies",
        )
    body_managers: dict[str, Any] = {}
    object_map: dict[str, tuple[str, str, str]] = {}
    object_vertices: dict[str, Any] = {}
    object_meshes: dict[str, Any] = {}
    body_fixed: dict[str, bool] = {}
    for body_index, body in enumerate(bodies):
        if not isinstance(body, Mapping):
            raise _fail(
                "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
                "collision body entry is invalid",
            )
        body_id = str(body.get("body_id") or "")
        components = body.get("components")
        if not body_id or not isinstance(components, list) or not components:
            raise _fail(
                "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
                "collision body has no identifier or components",
            )
        if body_id in body_managers:
            raise _fail(
                "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
                f"collision body identifier is duplicated: {body_id}",
            )
        try:
            manager = CollisionManager()
        except Exception as exc:
            raise _fail(
                "INTERPENETRATION_AUDIT_BACKEND_UNAVAILABLE",
                "python-fcl could not initialize the collision manager",
            ) from exc
        body_managers[body_id] = manager
        body_fixed[body_id] = bool(body.get("fixed"))
        for component_index, component in enumerate(components):
            if not isinstance(component, Mapping):
                raise _fail(
                    "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
                    "collision component entry is invalid",
                )
            solid_id = str(component.get("solid_id") or "")
            component_id = str(component.get("component_id") or solid_id)
            if not solid_id or not component_id:
                raise _fail(
                    "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
                    "collision component has no component or source-solid identifier",
                )
            path = _component_path(component, workspace)
            mesh = trimesh.load_mesh(path, process=False, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            if not len(mesh.vertices) or not len(mesh.faces):
                raise _fail(
                    "INTERPENETRATION_AUDIT_COMPONENT_EMPTY",
                    f"collision component is empty: {path}",
                )
            name = f"b{body_index}:c{component_index}"
            manager.add_object(name, mesh)
            object_map[name] = (body_id, component_id, solid_id)
            object_vertices[name] = mesh.vertices.copy()
            object_meshes[name] = mesh
    return (
        body_managers,
        object_map,
        body_fixed,
        object_vertices,
        object_meshes,
    )


def audit_bicycle_sprocket_contact_component_geometry(
    collision_geometry: Mapping[str, Any],
    contract: Mapping[str, Any],
    body_poses: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Certify design-bound contact extensions against the public tori.

    Every allowed component must resolve to the declared body and STEP solid.
    Template-approved seed components may not bury the chain radially beyond
    the public sprocket contact radius.  Design-bound extensions must stay in
    the tighter public torus.  This prevents either label from granting
    arbitrary local collision immunity.
    """

    try:
        import numpy as np
        import trimesh
    except Exception as exc:
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BACKEND_UNAVAILABLE",
            "numpy and trimesh are required for sprocket contact certification",
        ) from exc

    zones = contract.get("flex_zones")
    bodies = collision_geometry.get("bodies")
    if (
        not isinstance(zones, list)
        or len(zones) != 2
        or not isinstance(bodies, list)
        or not isinstance(body_poses, Mapping)
    ):
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
            "sprocket contact certification has incomplete geometry bindings",
        )
    try:
        mesh_linear_deflection = float(
            collision_geometry["mesh_linear_deflection_m"]
        )
        axis = np.asarray(contract["axis_abs"], dtype=float)
        axis_magnitude = float(np.linalg.norm(axis))
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
            "sprocket contact certification parameters are malformed",
        ) from exc
    if (
        not math.isfinite(mesh_linear_deflection)
        or not 0.0 <= mesh_linear_deflection <= 0.001
        or not math.isfinite(axis_magnitude)
        or axis_magnitude <= 1.0e-12
    ):
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
            "sprocket contact certification tolerances or axis are invalid",
        )
    axis /= axis_magnitude

    components: list[tuple[str, str, Mapping[str, Any]]] = []
    for body in bodies:
        if not isinstance(body, Mapping):
            continue
        body_id = str(body.get("body_id") or "")
        for component in body.get("components") or ():
            if not isinstance(component, Mapping):
                continue
            solid_id = str(component.get("solid_id") or "")
            component_id = str(
                component.get("component_id") or solid_id
            )
            components.append((body_id, component_id, component))

    zone_geometry = {
        "flex.input_sprocket": (
            float(contract["input_pitch_radius_m"]),
            float(contract["input_sprocket_contact_radius_m"]),
        ),
        "flex.output_sprocket": (
            float(contract["output_pitch_radius_m"]),
            float(contract["output_sprocket_contact_radius_m"]),
        ),
    }
    workspace = Path(workspace_root).resolve()
    component_reports: list[dict[str, Any]] = []
    seed_reports: list[dict[str, Any]] = []
    extension_reports: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    approved_by_body: dict[str, list[str]] = {}
    extensions_by_body: dict[str, list[str]] = {}
    maximum_observed_distance = 0.0
    maximum_seed_radial_overshoot = 0.0
    maximum_seed_axial_offset = 0.0
    for zone in zones:
        if not isinstance(zone, Mapping):
            raise _fail(
                "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
                "sprocket contact certification contains an invalid flex zone",
            )
        zone_id = str(zone.get("zone_id") or "")
        body_id = str(zone.get("allowed_body_id") or "")
        solid_id = str(zone.get("allowed_solid_id") or "")
        allowed_ids = zone.get("allowed_component_ids")
        approved_ids = zone.get("approved_seed_component_ids")
        if (
            zone_id not in zone_geometry
            or not body_id
            or not solid_id
            or not isinstance(allowed_ids, list)
            or not allowed_ids
            or not isinstance(approved_ids, list)
            or not approved_ids
            or allowed_ids[: len(approved_ids)] != approved_ids
            or body_id not in body_poses
        ):
            raise _fail(
                "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
                "sprocket contact extensions do not preserve their approved seed bindings",
            )
        approved_by_body[body_id] = list(approved_ids)
        extension_ids = list(allowed_ids[len(approved_ids) :])
        extensions_by_body[body_id] = extension_ids
        center = np.asarray(zone["center_abs_m"], dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise _fail(
                "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
                "sprocket contact certification center is invalid",
            )
        pitch_radius, contact_radius = zone_geometry[zone_id]
        allowed_distance = contact_radius + mesh_linear_deflection
        transform = _pose_matrix(body_poses[body_id])
        for component_id in allowed_ids:
            matching = [
                component
                for candidate_body, candidate_id, component in components
                if candidate_body == body_id
                and candidate_id == component_id
                and str(component.get("solid_id") or "") == solid_id
            ]
            if len(matching) != 1:
                raise _fail(
                    "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
                    "each sprocket contact component must resolve exactly once "
                    "inside its bound body and source solid",
                )
            path = _component_path(matching[0], workspace)
            mesh = trimesh.load_mesh(path, process=False, force="mesh")
            if not isinstance(mesh, trimesh.Trimesh):
                mesh = trimesh.util.concatenate(
                    tuple(mesh.geometry.values())
                )
            mesh.process(validate=True)
            if not len(mesh.vertices):
                raise _fail(
                    "BICYCLE_CHAIN_CLEARANCE_GEOMETRY_INVALID",
                    f"sprocket contact component is empty: {component_id}",
                )
            if not mesh.is_watertight:
                raise _fail(
                    "BICYCLE_CHAIN_CLEARANCE_GEOMETRY_INVALID",
                    f"sprocket contact component is not a closed solid: {component_id}",
                )
            vertices = np.asarray(mesh.vertices, dtype=float)
            homogeneous = np.ones((len(vertices), 4), dtype=float)
            homogeneous[:, :3] = vertices
            world_vertices = (transform @ homogeneous.T).T[:, :3]
            offsets = world_vertices - center
            axial = offsets @ axis
            radial_vectors = offsets - np.outer(axial, axis)
            radial = np.linalg.norm(radial_vectors, axis=1)
            if component_id in approved_ids:
                radial_overshoot = max(
                    0.0,
                    float(np.max(radial)) - pitch_radius,
                )
                axial_offset = float(np.max(np.abs(axial)))
                maximum_seed_radial_overshoot = max(
                    maximum_seed_radial_overshoot,
                    radial_overshoot,
                )
                maximum_seed_axial_offset = max(
                    maximum_seed_axial_offset,
                    axial_offset,
                )
                passed = radial_overshoot <= allowed_distance + 1.0e-12
                report = {
                    "certification_kind": "approved_seed_envelope",
                    "zone_id": zone_id,
                    "body_id": body_id,
                    "solid_id": solid_id,
                    "component_id": component_id,
                    "maximum_radial_overshoot_m": radial_overshoot,
                    "diagnostic_maximum_axial_offset_m": axial_offset,
                    "maximum_allowed_seed_radial_overshoot_m": (
                        allowed_distance
                    ),
                    "passed": passed,
                }
                seed_reports.append(report)
                component_reports.append(report)
                if not passed:
                    failures.append(report)
                continue
            torus_distance = np.sqrt(
                (radial - pitch_radius) ** 2 + axial**2
            )
            maximum_distance = float(np.max(torus_distance))
            maximum_observed_distance = max(
                maximum_observed_distance,
                maximum_distance,
            )
            passed = maximum_distance <= allowed_distance + 1.0e-12
            report = {
                "certification_kind": "design_bound_torus_extension",
                "zone_id": zone_id,
                "body_id": body_id,
                "solid_id": solid_id,
                "component_id": component_id,
                "maximum_torus_centerline_distance_m": maximum_distance,
                "maximum_allowed_torus_centerline_distance_m": (
                    allowed_distance
                ),
                "passed": passed,
            }
            extension_reports.append(report)
            component_reports.append(report)
            if not passed:
                failures.append(report)
    passed = not failures
    return {
        "schema_version": "bicycle_sprocket_contact_geometry_audit_v1",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "approved_seed_component_ids_by_body": approved_by_body,
        "extension_component_ids_by_body": extensions_by_body,
        "approved_seed_component_count": len(seed_reports),
        "extension_component_count": len(extension_reports),
        "mesh_linear_deflection_m": mesh_linear_deflection,
        "maximum_observed_seed_radial_overshoot_m": (
            maximum_seed_radial_overshoot
        ),
        "diagnostic_maximum_observed_seed_axial_offset_m": (
            maximum_seed_axial_offset
        ),
        "seed_axial_offset_is_gate_criterion": False,
        "maximum_observed_torus_centerline_distance_m": (
            maximum_observed_distance
        ),
        "component_reports": component_reports,
        "failure_count": len(failures),
        "failures": failures,
        "criterion": (
            "approved seed radial overshoot stays inside the public sprocket "
            "contact radius; design-bound extension vertices stay inside "
            "the public sprocket torus; both include mesh deflection"
        ),
    }


def _chain_edge_collision_manager(
    contract: Mapping[str, Any],
    *,
    included_regions: frozenset[str] | None,
    audit_radius_m: float,
) -> tuple[Any, dict[str, str]]:
    try:
        import trimesh
        from trimesh.collision import CollisionManager
    except Exception as exc:
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BACKEND_UNAVAILABLE",
            "trimesh with python-fcl is required for saved-state chain clearance",
        ) from exc
    points = contract.get("centerline_points_abs_m")
    regions = contract.get("edge_region_ids")
    if (
        not isinstance(points, list)
        or not isinstance(regions, list)
        or len(regions) != len(points) - 1
    ):
        raise _fail(
            "BICYCLE_CHAIN_PATH_TOPOLOGY_INVALID",
            "moving chain contract has no closed edge inventory",
        )
    manager = CollisionManager()
    region_by_name: dict[str, str] = {}
    retained = 0
    for index, (left, right, region) in enumerate(
        zip(points, points[1:], regions)
    ):
        region_id = str(region)
        if included_regions is not None and region_id not in included_regions:
            continue
        try:
            mesh = trimesh.creation.cylinder(
                radius=audit_radius_m,
                segment=[left, right],
                sections=8,
            )
        except Exception as exc:
            raise _fail(
                "BICYCLE_CHAIN_CLEARANCE_GEOMETRY_INVALID",
                f"cannot build chain audit edge {index}: {exc}",
            ) from exc
        name = f"chain:e{index}:{region_id}"
        manager.add_object(name, mesh)
        region_by_name[name] = region_id
        retained += 1
    if retained == 0:
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_GEOMETRY_INVALID",
            "chain audit region contains no edges",
        )
    return manager, region_by_name


def audit_bicycle_moving_chain_path_clearance(
    collision_geometry: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Audit every moving-chain state against the original STEP colliders.

    Flexible regions are fixed in authored world space.  An allowed component
    receives local contact allowance only after its hash-bound mesh passes the
    public seed-envelope or torus-extension certification above.  Outside its
    local flex zone it remains subject to the full-body collision audit.
    """

    from .bicycle_chain_path import audit_moving_chain_path_states

    state_audit = audit_moving_chain_path_states(contract, samples)
    first_sample = samples[0] if samples else None
    first_body_poses = (
        first_sample.get("body_poses")
        if isinstance(first_sample, Mapping)
        else None
    )
    if not isinstance(first_body_poses, Mapping):
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_POSES_MISSING",
            "physical samples do not contain an initial collision-body pose",
        )
    contact_geometry_audit = (
        audit_bicycle_sprocket_contact_component_geometry(
            collision_geometry,
            contract,
            first_body_poses,
            workspace_root=workspace_root,
        )
    )
    if contact_geometry_audit.get("passed") is not True:
        failures = contact_geometry_audit.get("failures") or ()
        detail = ", ".join(
            f"{item.get('component_id')}: "
            f"{item.get('certification_kind') or 'unknown'}"
            for item in failures[:5]
        )
        raise _fail(
            "BICYCLE_CHAIN_CONTACT_COMPONENT_OUTSIDE_SPROCKET",
            "sprocket contact components extend outside their public "
            "geometry envelope"
            + (f": {detail}" if detail else ""),
        )
    tolerance = runtime_interpenetration_tolerance_m(collision_geometry)
    try:
        authored_radius = float(contract["chain_collision_radius_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(
            "BICYCLE_CHAIN_PATH_WIDTH_MISSING",
            "moving chain contract has no collision radius",
        ) from exc
    audit_radius = authored_radius - tolerance
    if audit_radius < 0.0001:
        raise _fail(
            "BICYCLE_CHAIN_PATH_WIDTH_INVALID",
            "chain audit radius does not exceed the runtime tolerance",
        )
    input_body_id = str(contract.get("input_body_id") or "")
    output_body_id = str(contract.get("output_body_id") or "")
    input_solid_id = str(contract.get("input_sprocket_solid_id") or "")
    output_solid_id = str(contract.get("output_sprocket_solid_id") or "")
    zones = contract.get("flex_zones")
    if not isinstance(zones, list):
        raise _fail(
            "BICYCLE_CHAIN_PATH_PENETRATION_CAP_MISSING",
            "moving chain contract has no fixed flex-zone inventory",
        )
    mechanical_limits_by_body: dict[str, float] = {}
    allowed_component_ids_by_body: dict[str, frozenset[str]] = {}
    approved_seed_component_ids_by_body: dict[str, frozenset[str]] = {}
    certified_extension_component_ids_by_body: dict[
        str, frozenset[str]
    ] = {}
    try:
        for zone in zones:
            body_id = str(zone["allowed_body_id"])
            mechanical_limits_by_body[body_id] = float(
                zone["maximum_matching_sprocket_penetration_m"]
            )
            component_ids = zone["allowed_component_ids"]
            approved_seed_ids = zone["approved_seed_component_ids"]
            if (
                not isinstance(component_ids, list)
                or not component_ids
                or not all(
                    isinstance(component_id, str) and component_id
                    for component_id in component_ids
                )
                or len(set(component_ids)) != len(component_ids)
                or not isinstance(approved_seed_ids, list)
                or not approved_seed_ids
                or component_ids[: len(approved_seed_ids)]
                != approved_seed_ids
            ):
                raise ValueError(
                    "allowed component ids must preserve approved seed ids"
                )
            allowed_component_ids_by_body[body_id] = frozenset(
                component_ids
            )
            approved_seed_component_ids_by_body[body_id] = frozenset(
                approved_seed_ids
            )
            certified_extension_component_ids_by_body[body_id] = (
                frozenset(component_ids[len(approved_seed_ids) :])
            )
        maximum_mechanical_penetration = float(
            contract["maximum_matching_sprocket_penetration_m"]
        )
        mesh_linear_deflection = float(
            collision_geometry["mesh_linear_deflection_m"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(
            "BICYCLE_CHAIN_PATH_PENETRATION_CAP_MISSING",
            "moving chain flex zones have no matching-sprocket penetration cap",
        ) from exc
    if (
        set(mechanical_limits_by_body) != {input_body_id, output_body_id}
        or set(allowed_component_ids_by_body)
        != {input_body_id, output_body_id}
        or set(approved_seed_component_ids_by_body)
        != {input_body_id, output_body_id}
        or set(certified_extension_component_ids_by_body)
        != {input_body_id, output_body_id}
        or any(
            not math.isfinite(limit) or limit <= 0.0
            for limit in mechanical_limits_by_body.values()
        )
        or not math.isfinite(maximum_mechanical_penetration)
        or maximum_mechanical_penetration <= 0.0
        or not math.isclose(
            maximum_mechanical_penetration,
            max(mechanical_limits_by_body.values()),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        or not math.isfinite(mesh_linear_deflection)
        or not 0.0 <= mesh_linear_deflection <= 0.001
    ):
        raise _fail(
            "BICYCLE_CHAIN_PATH_PENETRATION_CAP_INVALID",
            "matching-sprocket penetration caps are invalid or inconsistent",
        )
    collision_query_limits_by_body = {
        body_id: mechanical_limit + mesh_linear_deflection
        for body_id, mechanical_limit in mechanical_limits_by_body.items()
    }
    maximum_collision_query_penetration = max(
        collision_query_limits_by_body.values()
    )
    try:
        import numpy as np

        axis = np.asarray(contract["axis_abs"], dtype=float)
        axis_magnitude = float(np.linalg.norm(axis))
        if not math.isfinite(axis_magnitude) or axis_magnitude <= 1.0e-12:
            raise ValueError("sprocket axis is degenerate")
        axis /= axis_magnitude
        maximum_zone_pose_error = (
            float(contract["maximum_chain_sprocket_separation_m"])
            + mesh_linear_deflection
        )
        fixed_zone_frames: dict[
            str, tuple[Any, Any, Any, float]
        ] = {}
        for zone in zones:
            body_id = str(zone["allowed_body_id"])
            initial_transform = _pose_matrix(first_body_poses[body_id])
            inverse_initial = np.linalg.inv(initial_transform)
            authored_center = np.asarray(
                zone["center_abs_m"],
                dtype=float,
            )
            local_center = (
                inverse_initial
                @ np.asarray([*authored_center, 1.0], dtype=float)
            )[:3]
            local_axis = initial_transform[:3, :3].T @ axis
            fixed_zone_frames[body_id] = (
                authored_center,
                local_center,
                local_axis,
                float(zone["radius_m"]),
            )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
            "fixed sprocket contact zones have malformed pose bindings",
        ) from exc
    if (
        set(fixed_zone_frames) != {input_body_id, output_body_id}
        or not math.isfinite(maximum_zone_pose_error)
        or maximum_zone_pose_error <= 0.0
    ):
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
            "fixed sprocket contact zones have invalid pose tolerances",
        )
    workspace = Path(workspace_root).resolve()
    (
        body_managers,
        object_map,
        _body_fixed,
        _object_vertices,
        object_meshes,
    ) = _load_collision_manager(collision_geometry, workspace=workspace)
    if (
        input_body_id not in body_managers
        or output_body_id not in body_managers
        or input_body_id == output_body_id
        or not input_solid_id
        or not output_solid_id
    ):
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
            "moving chain contract does not resolve distinct sprocket bodies and solids",
        )
    all_regions = frozenset(
        {"flex.input_sprocket", "flex.output_sprocket", "rigid.tangent_span"}
    )
    all_chain, all_chain_names = _chain_edge_collision_manager(
        contract,
        included_regions=all_regions,
        audit_radius_m=audit_radius,
    )
    not_input_flex, not_input_names = _chain_edge_collision_manager(
        contract,
        included_regions=all_regions - {"flex.input_sprocket"},
        audit_radius_m=audit_radius,
    )
    input_flex, input_flex_names = _chain_edge_collision_manager(
        contract,
        included_regions=frozenset({"flex.input_sprocket"}),
        audit_radius_m=audit_radius,
    )
    not_output_flex, not_output_names = _chain_edge_collision_manager(
        contract,
        included_regions=all_regions - {"flex.output_sprocket"},
        audit_radius_m=audit_radius,
    )
    output_flex, output_flex_names = _chain_edge_collision_manager(
        contract,
        included_regions=frozenset({"flex.output_sprocket"}),
        audit_radius_m=audit_radius,
    )

    try:
        from trimesh.collision import CollisionManager
    except Exception as exc:
        raise _fail(
            "BICYCLE_CHAIN_CLEARANCE_BACKEND_UNAVAILABLE",
            "python-fcl collision manager is unavailable",
        ) from exc

    forbidden_local_managers: dict[str, Any | None] = {}
    forbidden_names: dict[str, dict[str, tuple[str, str, str]]] = {}
    for body_id, allowed_solid_id in (
        (input_body_id, input_solid_id),
        (output_body_id, output_solid_id),
    ):
        allowed_component_ids = allowed_component_ids_by_body[body_id]
        manager = CollisionManager()
        names: dict[str, tuple[str, str, str]] = {}
        for name, (
            candidate_body,
            component_id,
            solid_id,
        ) in object_map.items():
            if candidate_body != body_id or (
                solid_id == allowed_solid_id
                and component_id in allowed_component_ids
            ):
                continue
            manager.add_object(name, object_meshes[name])
            names[name] = (candidate_body, component_id, solid_id)
        forbidden_local_managers[body_id] = manager if names else None
        forbidden_names[body_id] = names
        matching = {
            name: binding
            for name, binding in object_map.items()
            if (
                binding[0] == body_id
                and binding[1] in allowed_component_ids
                and binding[2] == allowed_solid_id
            )
        }
        resolved_component_ids = {
            binding[1] for binding in matching.values()
        }
        if resolved_component_ids != set(allowed_component_ids):
            raise _fail(
                "BICYCLE_CHAIN_CLEARANCE_BINDING_INVALID",
                "each approved sprocket contact component must resolve "
                "exactly inside its bound body and source solid",
            )

    queries: dict[
        str,
        list[
            tuple[
                Any,
                dict[str, str],
                Any,
                dict[str, tuple[str, str, str]],
            ]
        ],
    ] = {}
    for body_id, manager in body_managers.items():
        body_object_names = {
            name: binding
            for name, binding in object_map.items()
            if binding[0] == body_id
        }
        if body_id == input_body_id:
            rows = [
                (
                    not_input_flex,
                    not_input_names,
                    manager,
                    body_object_names,
                ),
            ]
            forbidden = forbidden_local_managers[body_id]
            if forbidden is not None:
                rows.append(
                    (
                        input_flex,
                        input_flex_names,
                        forbidden,
                        forbidden_names[body_id],
                    )
                )
            queries[body_id] = rows
        elif body_id == output_body_id:
            rows = [
                (
                    not_output_flex,
                    not_output_names,
                    manager,
                    body_object_names,
                ),
            ]
            forbidden = forbidden_local_managers[body_id]
            if forbidden is not None:
                rows.append(
                    (
                        output_flex,
                        output_flex_names,
                        forbidden,
                        forbidden_names[body_id],
                    )
                )
            queries[body_id] = rows
        else:
            queries[body_id] = [
                (
                    all_chain,
                    all_chain_names,
                    manager,
                    body_object_names,
                )
            ]

    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    retained_events: list[dict[str, Any]] = []
    illegal_event_count = 0
    maximum_observed_zone_center_drift = 0.0
    maximum_observed_zone_axis_surface_drift = 0.0
    query_count = 0
    cache_hit_count = 0
    for sample_index, sample in enumerate(samples):
        poses = sample.get("body_poses") if isinstance(sample, Mapping) else None
        if not isinstance(poses, Mapping) or set(body_managers) - set(poses):
            raise _fail(
                "BICYCLE_CHAIN_CLEARANCE_POSES_MISSING",
                "physical sample does not contain every collision body pose",
            )
        for body_id, query_rows in queries.items():
            pose = poses[body_id]
            pose_sha256 = canonical_sha256(pose)
            cache_key = (body_id, pose_sha256)
            cached = cache.get(cache_key)
            if cached is not None:
                cache_hit_count += 1
                body_events = cached
            else:
                transform = _pose_matrix(pose)
                for name, (candidate_body, _component_id, _solid_id) in object_map.items():
                    if candidate_body == body_id:
                        body_managers[body_id].set_transform(name, transform)
                        forbidden = forbidden_local_managers.get(body_id)
                        if forbidden is not None and name in forbidden_names[body_id]:
                            forbidden.set_transform(name, transform)
                body_events = []
                fixed_frame = fixed_zone_frames.get(body_id)
                if fixed_frame is not None:
                    (
                        authored_center,
                        local_center,
                        local_axis,
                        zone_radius,
                    ) = fixed_frame
                    current_center = (
                        transform
                        @ np.asarray([*local_center, 1.0], dtype=float)
                    )[:3]
                    current_axis = transform[:3, :3] @ local_axis
                    center_drift = float(
                        np.linalg.norm(current_center - authored_center)
                    )
                    axis_surface_drift = float(
                        np.linalg.norm(np.cross(current_axis, axis))
                        * zone_radius
                    )
                    maximum_observed_zone_center_drift = max(
                        maximum_observed_zone_center_drift,
                        center_drift,
                    )
                    maximum_observed_zone_axis_surface_drift = max(
                        maximum_observed_zone_axis_surface_drift,
                        axis_surface_drift,
                    )
                    if (
                        center_drift > maximum_zone_pose_error + 1.0e-12
                        or axis_surface_drift
                        > maximum_zone_pose_error + 1.0e-12
                    ):
                        body_events.append(
                            {
                                "body_id": body_id,
                                "component_id": None,
                                "solid_id": None,
                                "chain_edge": None,
                                "chain_region_id": (
                                    "flex.input_sprocket"
                                    if body_id == input_body_id
                                    else "flex.output_sprocket"
                                ),
                                "zone_center_drift_m": center_drift,
                                "zone_axis_surface_drift_m": (
                                    axis_surface_drift
                                ),
                                "maximum_allowed_zone_pose_error_m": (
                                    maximum_zone_pose_error
                                ),
                                "reason": (
                                    "sprocket_contact_envelope_pose_drift"
                                ),
                            }
                        )
                for (
                    chain_manager,
                    chain_names,
                    component_manager,
                    component_names,
                ) in query_rows:
                    query_count += 1
                    try:
                        collision_result = chain_manager.in_collision_other(
                            component_manager,
                            return_names=True,
                            return_data=False,
                        )
                    except Exception as exc:
                        raise _fail(
                            "BICYCLE_CHAIN_CLEARANCE_BACKEND_FAILED",
                            f"chain clearance query failed for body {body_id!r}: {exc}",
                        ) from exc
                    hit, pairs = collision_result[:2]
                    if not hit:
                        continue
                    for pair in sorted(tuple(sorted(str(name) for name in pair)) for pair in pairs):
                        chain_name = next(
                            (name for name in pair if name in chain_names),
                            None,
                        )
                        component_name = next(
                            (name for name in pair if name in component_names),
                            None,
                        )
                        if chain_name is None or component_name is None:
                            continue
                        component_body, component_id, solid_id = component_names[
                            component_name
                        ]
                        body_events.append(
                            {
                                "body_id": component_body,
                                "component_id": component_id,
                                "solid_id": solid_id,
                                "chain_edge": chain_name,
                                "chain_region_id": chain_names[chain_name],
                                "reason": "forbidden_component_contact",
                            }
                        )
                cache[cache_key] = body_events
            for event in body_events:
                illegal_event_count += 1
                if len(retained_events) < MAX_RETAINED_PENETRATION_EVENTS:
                    retained_events.append(
                        {
                            **event,
                            "sample_index": sample_index,
                            "time_s": float(sample.get("time_s", 0.0)),
                            "phase_id": str(sample.get("phase_id") or ""),
                        }
                    )
    passed = illegal_event_count == 0
    return {
        **state_audit,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "saved_state_clearance_verified": passed,
        "original_step_components_used": True,
        "audit_sample_count": len(samples),
        "collision_query_count": query_count,
        "pose_cache_hit_count": cache_hit_count,
        "authored_chain_radius_m": authored_radius,
        "runtime_tolerance_m": tolerance,
        "eroded_audit_radius_m": audit_radius,
        "maximum_matching_sprocket_penetration_m": (
            maximum_collision_query_penetration
        ),
        "maximum_mechanical_matching_sprocket_penetration_m": (
            maximum_mechanical_penetration
        ),
        "collision_mesh_linear_deflection_m": mesh_linear_deflection,
        "maximum_allowed_sprocket_zone_pose_error_m": (
            maximum_zone_pose_error
        ),
        "maximum_observed_sprocket_zone_center_drift_m": (
            maximum_observed_zone_center_drift
        ),
        "maximum_observed_sprocket_zone_axis_surface_drift_m": (
            maximum_observed_zone_axis_surface_drift
        ),
        "matching_sprocket_penetration_limits_by_body_m": dict(
            sorted(collision_query_limits_by_body.items())
        ),
        "mechanical_matching_sprocket_penetration_limits_by_body_m": dict(
            sorted(mechanical_limits_by_body.items())
        ),
        "matching_sprocket_contact_component_ids_by_body": {
            body_id: sorted(component_ids)
            for body_id, component_ids in sorted(
                allowed_component_ids_by_body.items()
            )
        },
        "geometry_certified_approved_seed_component_ids_by_body": {
            body_id: sorted(component_ids)
            for body_id, component_ids in sorted(
                approved_seed_component_ids_by_body.items()
            )
        },
        "geometry_certified_local_extension_component_ids_by_body": {
            body_id: sorted(component_ids)
            for body_id, component_ids in sorted(
                certified_extension_component_ids_by_body.items()
            )
        },
        "sprocket_contact_component_geometry_audit": (
            contact_geometry_audit
        ),
        "maximum_observed_matching_sprocket_penetration_m": None,
        "matching_sprocket_runtime_epa_depth_observed": False,
        "matching_sprocket_local_contact_depth_method": (
            "static_hash_bound_geometry_envelope_not_epa"
        ),
        "illegal_event_count": illegal_event_count,
        "retained_illegal_event_count": len(retained_events),
        "worst_events": retained_events,
        "flex_zone_policy": (
            "hash-bound seed-envelope and torus-extension certifications grant "
            "local contact without unstable EPA depth; fixed-zone pose drift is "
            "bounded; every allowed component remains forbidden outside its zone"
        ),
    }


def _runtime_body_pair_contact_count(
    sample: Mapping[str, Any],
    body_a_id: str,
    body_b_id: str,
) -> int:
    counts = sample.get("contact_pair_counts")
    if not isinstance(counts, Mapping):
        return 0
    expected = frozenset((body_a_id, body_b_id))
    total = 0
    for raw_pair, raw_count in counts.items():
        if not isinstance(raw_count, (int, float)):
            continue
        pair = frozenset(str(raw_pair).split("|"))
        if pair == expected:
            total += int(raw_count)
    return total


def audit_family_collision_geometry(
    collision_geometry: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    workspace_root: str | Path,
    tolerance_m: float | None = None,
    contact_intent_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit every saved pose against every distinct rigid-body collider.

    The simulation contact filter is deliberately ignored. Same-body and
    fixed-versus-fixed intersections are irrelevant to rigid-body motion; every
    pair containing a moving body is checked, including filtered pairs.
    """

    workspace = Path(workspace_root).resolve()
    tolerance = float(
        tolerance_m
        if tolerance_m is not None
        else collision_geometry.get(
            "interpenetration_tolerance_m",
            DEFAULT_INTERPENETRATION_TOLERANCE_M,
        )
    )
    if not 0.0 <= tolerance <= 0.01:
        raise _fail(
            "INTERPENETRATION_AUDIT_TOLERANCE_INVALID",
            "interpenetration tolerance must be between 0 and 10 mm",
        )
    if not isinstance(samples, Sequence) or not samples:
        raise _fail(
            "INTERPENETRATION_AUDIT_SAMPLES_MISSING",
            "physical trace has no saved poses to audit",
        )

    (
        body_managers,
        object_map,
        body_fixed,
        object_vertices,
        object_meshes,
    ) = _load_collision_manager(collision_geometry, workspace=workspace)
    body_objects: dict[str, list[str]] = {body_id: [] for body_id in body_fixed}
    for name, (body_id, _component_id, _solid_id) in object_map.items():
        body_objects[body_id].append(name)
    object_names_by_component_ref = {
        f"{body_id}/{component_id}": name
        for name, (body_id, component_id, _solid_id) in object_map.items()
    }
    source_solid_ref_by_component_ref = {
        f"{body_id}/{component_id}": f"{body_id}/{solid_id}"
        for body_id, component_id, solid_id in object_map.values()
    }

    proximity_declarations: list[dict[str, Any]] = []
    proximity_managers: dict[str, Any] = {}
    penetration_exemptions: dict[frozenset[str], tuple[float, frozenset[str]]] = {}
    if isinstance(contact_intent_policy, Mapping):
        from .contact_intent import (
            EXPECTED_REAL_CONTACT,
            validate_component_contact_intent_policy,
        )

        declarations = validate_component_contact_intent_policy(
            contact_intent_policy,
            collision_geometry=collision_geometry,
        )
        for declaration in declarations:
            separation = declaration.get("max_separation_m")
            if (
                declaration.get("kind") != EXPECTED_REAL_CONTACT
                or separation is None
            ):
                continue
            left = tuple(str(item) for item in declaration["component_a_refs"])
            right = tuple(str(item) for item in declaration["component_b_refs"])
            left_bodies = {item.partition("/")[0] for item in left}
            right_bodies = {item.partition("/")[0] for item in right}
            if len(left_bodies) != 1 or len(right_bodies) != 1:
                raise _fail(
                    "COMPONENT_CONTACT_PROXIMITY_BINDING_INVALID",
                    str(declaration["intent_id"]),
                )
            proximity_declarations.append(
                {
                    "intent_id": str(declaration["intent_id"]),
                    "left": left,
                    "right": right,
                    "body_a_id": next(iter(left_bodies)),
                    "body_b_id": next(iter(right_bodies)),
                    "active_phase_ids": frozenset(
                        str(item)
                        for item in declaration.get("active_phase_ids") or ()
                    ),
                    "max_separation_m": float(separation),
                }
            )
            for component_ref in (*left, *right):
                name = object_names_by_component_ref[component_ref]
                if name in proximity_managers:
                    continue
                component_manager = type(next(iter(body_managers.values())))()
                component_manager.add_object(name, object_meshes[name])
                proximity_managers[name] = component_manager
        # ATTENTION AI EDITORS (2026-08-08): penetration exemption contract.
        # A saved-pose penetration event deeper than the global runtime
        # tolerance is LEGAL only when the solid pair is covered by a
        # disclosed EXPECTED_REAL_CONTACT declaration, the event phase is
        # inside the declaration's active phases, and the depth stays within
        # the declaration's own max_penetration_m.  Undeclared pairs always
        # use the global runtime tolerance.  Both ends must move together:
        # the declared budget is written by
        # contact_intent.build_component_contact_intent_policy (and by
        # explicit declarations) and consumed here and by
        # contact_intent.audit_component_contact_intents.  Update all three
        # places in the same commit. (DEFECT #4 fix)
        for declaration in declarations:
            if declaration.get("kind") != EXPECTED_REAL_CONTACT:
                continue
            exemption_limit = declaration.get("max_penetration_m")
            if exemption_limit is None:
                continue
            exemption_phases = frozenset(
                str(item) for item in declaration.get("active_phase_ids") or ()
            )
            for ref_a in declaration["component_a_refs"]:
                for ref_b in declaration["component_b_refs"]:
                    key = frozenset((str(ref_a), str(ref_b)))
                    previous = penetration_exemptions.get(key)
                    if previous is None or float(exemption_limit) > previous[0]:
                        penetration_exemptions[key] = (
                            float(exemption_limit),
                            exemption_phases,
                        )

    maximum_depth = 0.0
    collision_event_count = 0
    illegal_event_count = 0
    declared_exempted_event_count = 0
    first_illegal_event: dict[str, Any] | None = None
    audited_body_pairs = tuple(
        sorted(
            tuple(sorted((body_a, body_b)))
            for body_a, body_b in combinations(body_fixed, 2)
            if not (body_fixed[body_a] and body_fixed[body_b])
        )
    )
    illegal_body_pairs: set[tuple[str, str]] = set()
    retained: list[dict[str, Any]] = []
    contact_events_by_pair_phase: dict[tuple[str, str, str], dict[str, Any]] = {}
    proximity_events_by_pair_phase: dict[
        tuple[str, str, str],
        dict[str, Any],
    ] = {}

    import numpy as np

    for sample_index, sample in enumerate(samples):
        poses = sample.get("body_poses") if isinstance(sample, Mapping) else None
        if not isinstance(poses, Mapping) or set(body_objects) - set(poses):
            raise _fail(
                "INTERPENETRATION_AUDIT_POSES_MISSING",
                "physical sample does not contain every collision body pose",
            )
        world_bounds: dict[str, tuple[Any, Any]] = {}
        for body_id, names in body_objects.items():
            transform = _pose_matrix(poses[body_id])
            for name in names:
                body_managers[body_id].set_transform(name, transform)
                if name in proximity_managers:
                    proximity_managers[name].set_transform(name, transform)
                vertices = object_vertices[name]
                world = vertices @ transform[:3, :3].T + transform[:3, 3]
                world_bounds[name] = (world.min(axis=0), world.max(axis=0))

        contact_data: list[Any] = []
        for body_a, body_b in audited_body_pairs:
            try:
                _hit, _pairs, pair_contact_data = body_managers[
                    body_a
                ].in_collision_other(
                    body_managers[body_b],
                    return_names=True,
                    return_data=True,
                )
            except Exception as exc:
                raise _fail(
                    "INTERPENETRATION_AUDIT_BACKEND_FAILED",
                    "collision backend failed at saved sample "
                    f"{sample_index} for rigid-body pair "
                    f"{body_a!r}, {body_b!r}: {exc}",
                ) from exc
            contact_data.extend(pair_contact_data)

        per_pair_depth: dict[tuple[str, str], dict[str, Any]] = {}
        for contact in contact_data:
            names = tuple(sorted(str(name) for name in contact.names))
            if len(names) != 2 or names[0] not in object_map or names[1] not in object_map:
                continue
            body_a, component_a, solid_a = object_map[names[0]]
            body_b, component_b, solid_b = object_map[names[1]]
            if body_a == body_b or (body_fixed[body_a] and body_fixed[body_b]):
                continue
            body_pair = tuple(sorted((body_a, body_b)))
            fcl_depth = max(0.0, float(contact.depth))
            minimum = np.maximum(world_bounds[names[0]][0], world_bounds[names[1]][0])
            maximum = np.minimum(world_bounds[names[0]][1], world_bounds[names[1]][1])
            aabb_depth_bound = max(0.0, float(np.min(maximum - minimum)))
            depth = min(fcl_depth, aabb_depth_bound)
            solid_pair = tuple(
                sorted(
                    (
                        f"{body_a}/{component_a}",
                        f"{body_b}/{component_b}",
                    )
                )
            )
            previous = per_pair_depth.get(solid_pair)
            if previous is None or depth > previous["depth"]:
                contact_position = getattr(contact, "pos", None)
                contact_normal = getattr(contact, "normal", None)

                def vector3(value: Any) -> list[float] | None:
                    if value is None:
                        return None
                    try:
                        values = np.asarray(value, dtype=float).reshape(-1)
                    except (TypeError, ValueError):
                        return None
                    if len(values) < 3 or not np.all(np.isfinite(values[:3])):
                        return None
                    return [float(item) for item in values[:3]]

                contact_position_vector = vector3(contact_position)
                contact_position_source = "fcl_contact"
                if contact_position_vector is None:
                    contact_position_vector = [
                        float(item) for item in (minimum + maximum) * 0.5
                    ]
                    contact_position_source = "world_aabb_overlap_center"
                contact_normal_vector = vector3(contact_normal)
                contact_normal_source = "fcl_contact"
                if contact_normal_vector is None:
                    overlap = maximum - minimum
                    axis = int(np.argmin(overlap))
                    first_center = (
                        world_bounds[names[0]][0] + world_bounds[names[0]][1]
                    ) * 0.5
                    second_center = (
                        world_bounds[names[1]][0] + world_bounds[names[1]][1]
                    ) * 0.5
                    contact_normal_vector = [0.0, 0.0, 0.0]
                    contact_normal_vector[axis] = (
                        1.0 if second_center[axis] >= first_center[axis] else -1.0
                    )
                    contact_normal_source = "minimum_world_aabb_overlap_axis"
                per_pair_depth[solid_pair] = {
                    "depth": depth,
                    "fcl_depth": fcl_depth,
                    "aabb_depth_bound": aabb_depth_bound,
                    "body_pair": body_pair,
                    "source_solid_pair": tuple(
                        sorted(
                            (
                                f"{body_a}/{solid_a}",
                                f"{body_b}/{solid_b}",
                            )
                        )
                    ),
                    "contact_position_abs_m": contact_position_vector,
                    "contact_position_source": contact_position_source,
                    "contact_normal_abs": contact_normal_vector,
                    "contact_normal_source": contact_normal_source,
                    "overlap_bounds_min_abs_m": [float(item) for item in minimum],
                    "overlap_bounds_max_abs_m": [float(item) for item in maximum],
                }

        phase_id = str(sample.get("phase_id") or "")
        for declaration in proximity_declarations:
            active_phases = declaration["active_phase_ids"]
            if active_phases and phase_id not in active_phases:
                continue
            runtime_count = _runtime_body_pair_contact_count(
                sample,
                declaration["body_a_id"],
                declaration["body_b_id"],
            )
            if runtime_count <= 0:
                continue
            for left_ref in declaration["left"]:
                left_name = object_names_by_component_ref[left_ref]
                left_manager = proximity_managers[left_name]
                for right_ref in declaration["right"]:
                    right_name = object_names_by_component_ref[right_ref]
                    right_body_id = object_map[right_name][0]
                    right_transform = _pose_matrix(poses[right_body_id])
                    try:
                        separation = float(
                            left_manager.min_distance_single(
                                object_meshes[right_name],
                                transform=right_transform,
                            )
                        )
                    except Exception as exc:
                        raise _fail(
                            "COMPONENT_CONTACT_PROXIMITY_AUDIT_FAILED",
                            f"distance query failed at saved sample {sample_index}: {exc}",
                        ) from exc
                    if not 0.0 <= separation <= declaration["max_separation_m"]:
                        continue
                    solid_pair = tuple(sorted((left_ref, right_ref)))
                    contact_key = (solid_pair[0], solid_pair[1], phase_id)
                    event = {
                        "sample_index": sample_index,
                        "audit_sample_index": sample_index,
                        "source_sample_index": int(
                            sample.get("_audit_source_sample_index", sample_index)
                        ),
                        "time_s": float(sample.get("time_s", 0.0)),
                        "phase_id": sample.get("phase_id"),
                        "body_pair": sorted(
                            (
                                declaration["body_a_id"],
                                declaration["body_b_id"],
                            )
                        ),
                        "solid_pair": list(solid_pair),
                        "source_solid_pair": sorted(
                            (
                                source_solid_ref_by_component_ref[left_ref],
                                source_solid_ref_by_component_ref[right_ref],
                            )
                        ),
                        "penetration_depth_m": 0.0,
                        "separation_distance_m": separation,
                        "max_separation_m": declaration["max_separation_m"],
                        "runtime_body_pair_contact_count": runtime_count,
                        "evidence_kind": (
                            "hash_bound_component_proximity_with_same_sample_"
                            "runtime_body_pair_contact"
                        ),
                        "intent_id": declaration["intent_id"],
                    }
                    prior = proximity_events_by_pair_phase.get(contact_key)
                    if (
                        prior is None
                        or separation < float(prior["separation_distance_m"])
                    ):
                        proximity_events_by_pair_phase[contact_key] = event

        for solid_pair, contact_record in per_pair_depth.items():
            depth = float(contact_record["depth"])
            fcl_depth = float(contact_record["fcl_depth"])
            aabb_depth_bound = float(contact_record["aabb_depth_bound"])
            body_pair = contact_record["body_pair"]
            collision_event_count += 1
            maximum_depth = max(maximum_depth, depth)
            event = {
                "sample_index": sample_index,
                "audit_sample_index": sample_index,
                "source_sample_index": int(
                    sample.get("_audit_source_sample_index", sample_index)
                ),
                "time_s": float(sample.get("time_s", 0.0)),
                "phase_id": sample.get("phase_id"),
                "body_pair": list(body_pair),
                "solid_pair": list(solid_pair),
                "source_solid_pair": list(
                    contact_record["source_solid_pair"]
                ),
                "penetration_depth_m": depth,
                "fcl_reported_depth_m": fcl_depth,
                "world_aabb_overlap_bound_m": aabb_depth_bound,
                "contact_position_abs_m": contact_record[
                    "contact_position_abs_m"
                ],
                "contact_position_source": contact_record[
                    "contact_position_source"
                ],
                "contact_normal_abs": contact_record["contact_normal_abs"],
                "contact_normal_source": contact_record[
                    "contact_normal_source"
                ],
                "overlap_bounds_min_abs_m": contact_record[
                    "overlap_bounds_min_abs_m"
                ],
                "overlap_bounds_max_abs_m": contact_record[
                    "overlap_bounds_max_abs_m"
                ],
            }
            contact_key = (
                solid_pair[0],
                solid_pair[1],
                str(sample.get("phase_id") or ""),
            )
            prior_contact = contact_events_by_pair_phase.get(contact_key)
            if (
                prior_contact is None
                or depth > float(prior_contact["penetration_depth_m"])
            ):
                contact_events_by_pair_phase[contact_key] = event
            if depth <= tolerance:
                continue
            exemption = penetration_exemptions.get(frozenset(solid_pair))
            if exemption is not None:
                exemption_limit, exemption_phases = exemption
                if (
                    not exemption_phases or phase_id in exemption_phases
                ) and depth <= exemption_limit:
                    declared_exempted_event_count += 1
                    continue
            illegal_event_count += 1
            if first_illegal_event is None:
                first_illegal_event = dict(event)
            illegal_body_pairs.add(body_pair)
            retained.append(event)
            retained.sort(key=lambda item: item["penetration_depth_m"], reverse=True)
            del retained[MAX_RETAINED_PENETRATION_EVENTS:]

    for contact_key, event in proximity_events_by_pair_phase.items():
        contact_events_by_pair_phase.setdefault(contact_key, event)
    contact_events = sorted(
        contact_events_by_pair_phase.values(),
        key=lambda item: item["penetration_depth_m"],
        reverse=True,
    )
    return {
        "schema_version": "0.1",
        "method": "fcl_contact_depth_bounded_by_world_aabb_overlap_per_solid_saved_pose",
        "passed": illegal_event_count == 0,
        "tolerance_m": tolerance,
        "sample_count": len(samples),
        "component_count": len(object_map),
        "tested_body_pair_count": len(audited_body_pairs),
        "collision_event_count": collision_event_count,
        "illegal_event_count": illegal_event_count,
        "declared_expected_contact_exempted_event_count": (
            declared_exempted_event_count
        ),
        "first_illegal_sample_index": (
            None
            if first_illegal_event is None
            else int(first_illegal_event["audit_sample_index"])
        ),
        "first_illegal_event": first_illegal_event,
        "illegal_body_pair_count": len(illegal_body_pairs),
        "max_penetration_m": maximum_depth,
        "worst_events": retained,
        "contact_events": contact_events,
        "runtime_confirmed_proximity_contact_event_count": len(
            proximity_events_by_pair_phase
        ),
        "runtime_confirmed_proximity_contact_policy_enabled": bool(
            proximity_declarations
        ),
        "contact_event_compression": "max_depth_per_solid_pair_per_phase",
        "fixed_fixed_pairs_ignored": True,
        "same_rigid_body_components_ignored": True,
        "same_rigid_body_components_submitted_to_backend": False,
        "collision_query_partition": (
            "all_cross_rigid_body_pairs_with_at_least_one_moving_body"
        ),
        "simulation_collision_filters_ignored": True,
    }


def audit_prescribed_swept_envelope(
    collision_geometry: Mapping[str, Any],
    pose_samples: Sequence[Mapping[str, Any]],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Fail closed on a deterministic pre-simulation motion envelope.

    Mechanism families own the kinematic sample construction; this generic
    layer owns the same hash-bound component collision audit used for saved
    physical states.
    """

    if len(pose_samples) < 2:
        raise _fail(
            "PRESCRIBED_SWEPT_ENVELOPE_SAMPLES_INVALID",
            "a swept-envelope audit requires at least two prescribed poses",
        )
    audit = audit_family_collision_geometry(
        collision_geometry,
        pose_samples,
        workspace_root=workspace_root,
        tolerance_m=runtime_interpenetration_tolerance_m(
            collision_geometry
        ),
    )
    return {
        **audit,
        "schema_version": "prescribed_swept_envelope_v1",
        "status": "PASS" if audit["passed"] else "FAIL",
        "pre_simulation": True,
        "sample_source": "mechanism_family_prescribed_kinematic_envelope",
    }


def disclosed_broad_collision_exclusions(
    physical_trace: Mapping[str, Any],
) -> list[dict[str, Any]]:
    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    for section_name in ("branch_semantics", "collision_geometry"):
        section = physical_trace.get(section_name)
        if not isinstance(section, Mapping):
            continue
        for field, value in section.items():
            if not (
                isinstance(field, str)
                and field.startswith("excluded_")
                and "collision" in field
                and isinstance(value, list)
            ):
                continue
            for item in value:
                payload = {
                    "source": f"{section_name}.{field}",
                    "exclusion": deepcopy(item),
                }
                fingerprint = repr(item)
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    retained.append(payload)
    return retained


def _component_contact_policy(
    physical_trace: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    semantics = physical_trace.get("branch_semantics")
    policy = (
        semantics.get("component_contact_intent_policy")
        if isinstance(semantics, Mapping)
        else None
    )
    if not isinstance(policy, Mapping):
        policy = geometry.get("component_contact_intent_policy")
    interface_policy = (
        semantics.get("physical_interface_policy")
        if isinstance(semantics, Mapping)
        else None
    )
    if not isinstance(policy, Mapping) and isinstance(interface_policy, Mapping):
        from .contact_intent import build_component_contact_intent_policy

        samples = physical_trace.get("samples")
        phase_ids = [
            str(item.get("phase_id"))
            for item in samples or ()
            if isinstance(item, Mapping) and item.get("phase_id")
        ]
        policy = build_component_contact_intent_policy(
            interface_policy.get("interface_declarations") or (),
            geometry,
            phase_ids=phase_ids,
            source="validator_saved_trace_component_binder_v1",
        )
    return policy if isinstance(policy, Mapping) else None


def _audit_component_contact_policy(
    physical_trace: Mapping[str, Any],
    geometry: Mapping[str, Any],
    geometry_audit: Mapping[str, Any],
    *,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_policy = policy or _component_contact_policy(
        physical_trace,
        geometry,
    )
    if not isinstance(resolved_policy, Mapping):
        return {
            "schema_version": "component_contact_intent_audit_v1",
            "status": "LEGACY_POLICY_NOT_DECLARED",
            "passed": True,
            "authoritative": False,
            "failure_reasons": [],
            "detail": (
                "historical trace has no physical interface declarations; "
                "global saved-pose penetration audit still applies"
            ),
        }

    runtime_counts: dict[str, int] = {}
    samples = physical_trace.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            counts = sample.get("contact_pair_counts") if isinstance(sample, Mapping) else None
            if not isinstance(counts, Mapping):
                continue
            for pair, value in counts.items():
                if isinstance(value, (int, float)):
                    runtime_counts[str(pair)] = runtime_counts.get(str(pair), 0) + int(value)

    from .contact_intent import audit_component_contact_intents

    result = audit_component_contact_intents(
        resolved_policy,
        geometry_audit,
        collision_geometry=geometry,
        runtime_contact_pair_counts=runtime_counts,
    )
    result["status"] = "PASS" if result["passed"] else "FAIL"
    result["authoritative"] = True
    return result


def _audit_moving_chain_path_if_required(
    physical_trace: Mapping[str, Any],
    geometry: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    from .bicycle_chain_spec import BICYCLE_CHAIN_PLUGIN_ID

    contract = physical_trace.get("moving_chain_path_contract")
    required = (
        physical_trace.get("mechanism_plugin_id")
        == BICYCLE_CHAIN_PLUGIN_ID
        or contract is not None
    )
    if not required:
        return {
            "schema_version": "bicycle_moving_chain_path_audit_v1",
            "status": "NOT_REQUIRED",
            "passed": True,
            "authoritative": False,
        }
    if not isinstance(contract, Mapping):
        raise _fail(
            "BICYCLE_CHAIN_PATH_CONTRACT_MISSING",
            "bicycle physical trace has no moving-chain path contract",
        )
    result = audit_bicycle_moving_chain_path_clearance(
        geometry,
        samples,
        contract,
        workspace_root=workspace_root,
    )
    result["authoritative"] = True
    return result


def apply_saved_physical_legality_gate(
    comparison: Mapping[str, Any],
    physical_trace: Mapping[str, Any],
    *,
    workspace_root: str | Path,
) -> dict[str, Any]:
    result = deepcopy(dict(comparison))
    failures = list(result.get("failure_reasons") or [])
    exclusions = disclosed_broad_collision_exclusions(physical_trace)
    geometry = physical_trace.get("collision_geometry")
    samples = physical_trace.get("samples")
    try:
        if not isinstance(geometry, Mapping):
            raise _fail(
                "INTERPENETRATION_AUDIT_GEOMETRY_MISSING",
                "physical trace has no collision geometry manifest",
            )
        if not isinstance(samples, list):
            raise _fail(
                "INTERPENETRATION_AUDIT_SAMPLES_MISSING",
                "physical trace has no saved samples",
            )
        contact_policy = _component_contact_policy(
            physical_trace,
            geometry,
        )
        audit = audit_family_collision_geometry(
            geometry,
            samples,
            workspace_root=workspace_root,
            tolerance_m=runtime_interpenetration_tolerance_m(geometry),
            contact_intent_policy=contact_policy,
        )
    except MechanismValidationError as exc:
        result["physical_geometry_legality"] = {
            "passed": False,
            "audit_completed": False,
            "failure_code": exc.code,
            "detail": str(exc),
        }
        if exc.code not in failures:
            failures.append(exc.code)
        result["verdict"] = "PHYSICS_INCONCLUSIVE"
    else:
        result["physical_geometry_legality"] = audit | {"audit_completed": True}
        metrics = result.get("physical_metrics")
        if isinstance(metrics, dict):
            metrics["max_illegal_interpenetration_m"] = audit["max_penetration_m"]
            metrics["illegal_interpenetration_event_count"] = audit[
                "illegal_event_count"
            ]
        thresholds = result.get("thresholds")
        if isinstance(thresholds, dict):
            thresholds["max_interpenetration_m"] = audit["tolerance_m"]
        if not audit["passed"]:
            if "physical_illegal_interpenetration" not in failures:
                failures.append("physical_illegal_interpenetration")
            result["verdict"] = "PHYSICAL_FAILURE"
        try:
            contact_intent = _audit_component_contact_policy(
                physical_trace,
                geometry,
                audit,
                policy=contact_policy,
            )
        except MechanismValidationError as exc:
            contact_intent = {
                "status": "INCONCLUSIVE",
                "passed": False,
                "authoritative": True,
                "failure_reasons": [exc.code],
                "detail": str(exc),
            }
        result["component_contact_intent"] = contact_intent
        for reason in contact_intent.get("failure_reasons") or ():
            if reason not in failures:
                failures.append(reason)
        if not contact_intent.get("passed"):
            if result.get("verdict") != "PHYSICAL_FAILURE":
                result["verdict"] = "PHYSICS_INCONCLUSIVE"

    moving_chain_path_audit: dict[str, Any]
    try:
        if not isinstance(geometry, Mapping) or not isinstance(samples, list):
            raise _fail(
                "BICYCLE_CHAIN_PATH_AUDIT_INPUT_MISSING",
                "physical trace has no geometry or saved samples",
            )
        moving_chain_path_audit = _audit_moving_chain_path_if_required(
            physical_trace,
            geometry,
            samples,
            workspace_root=workspace_root,
        )
    except MechanismValidationError as exc:
        moving_chain_path_audit = {
            "schema_version": "bicycle_moving_chain_path_audit_v1",
            "status": "INCONCLUSIVE",
            "passed": False,
            "authoritative": True,
            "failure_reasons": [exc.code],
            "detail": str(exc),
        }
        if exc.code not in failures:
            failures.append(exc.code)
        if result.get("verdict") != "PHYSICAL_FAILURE":
            result["verdict"] = "PHYSICS_INCONCLUSIVE"
    else:
        if not moving_chain_path_audit.get("passed"):
            code = "bicycle_moving_chain_path_clearance_failed"
            if code not in failures:
                failures.append(code)
            result["verdict"] = "PHYSICAL_FAILURE"
    result["moving_chain_path_audit"] = moving_chain_path_audit

    if exclusions:
        result["broad_body_pair_collision_exclusions"] = exclusions
        code = "physical_broad_body_pair_collision_exclusion_forbidden"
        if code not in failures:
            failures.append(code)
        if result.get("verdict") != "PHYSICAL_FAILURE":
            result["verdict"] = "PHYSICS_INCONCLUSIVE"
    else:
        result["broad_body_pair_collision_exclusions"] = []

    result["failure_reasons"] = failures
    non_claims = result.setdefault("strict_non_claims", {})
    if isinstance(non_claims, dict):
        non_claims["broad_body_pair_collision_filter_used"] = bool(exclusions)
        non_claims["all_saved_physical_poses_interpenetration_audited"] = bool(
            result.get("physical_geometry_legality", {}).get("audit_completed")
        )
        non_claims["individual_chain_link_or_tooth_physics_claimed"] = False
    return result

def normalized_physical_audit_inputs(
    physical_trace: Mapping[str, Any],
) -> tuple[Mapping[str, Any], list[dict[str, Any]]]:
    """Normalize family and two-body traces to one independent mesh audit schema."""

    geometry = physical_trace.get("collision_geometry")
    samples = physical_trace.get("samples")
    if not isinstance(geometry, Mapping):
        raise _fail(
            "INTERPENETRATION_AUDIT_GEOMETRY_MISSING",
            "physical trace has no collision geometry manifest",
        )
    if not isinstance(samples, list):
        raise _fail(
            "INTERPENETRATION_AUDIT_SAMPLES_MISSING",
            "physical trace has no saved samples",
        )
    if isinstance(geometry.get("bodies"), list):
        return geometry, [dict(item) for item in samples]

    fixed_components = geometry.get("fixed_components")
    moving_components = geometry.get("moving_components")
    fixed_asset = geometry.get("fixed")
    if not (
        isinstance(fixed_components, list)
        and fixed_components
        and isinstance(moving_components, list)
        and moving_components
        and isinstance(fixed_asset, Mapping)
    ):
        raise _fail(
            "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
            "physical collision geometry is not a supported per-body schema",
        )

    def components(items: Sequence[Any], role: str) -> list[dict[str, Any]]:
        retained: list[dict[str, Any]] = []
        for index, raw in enumerate(items):
            if not isinstance(raw, Mapping):
                raise _fail(
                    "INTERPENETRATION_AUDIT_GEOMETRY_INVALID",
                    f"{role} collision component is invalid",
                )
            item = dict(raw)
            source = str(item.get("source") or item.get("derived") or "")
            item["solid_id"] = Path(source).stem or f"{role}_{index + 1}"
            retained.append(item)
        return retained

    normalized_geometry = dict(geometry)
    normalized_geometry["schema"] = "plain_two_body_local_v1"
    normalized_geometry["bodies"] = [
        {
            "body_id": "fixed_boundary",
            "fixed": True,
            "components": components(fixed_components, "fixed"),
        },
        {
            "body_id": "moving_group",
            "fixed": False,
            "components": components(moving_components, "moving"),
        },
    ]
    fixed_position = list(fixed_asset.get("body_origin_abs_m") or (0.0, 0.0, 0.0))
    normalized_samples: list[dict[str, Any]] = []
    for raw in samples:
        item = dict(raw)
        moving_pose = item.get("moving_pose")
        fixed_pose = item.get("fixed_pose")
        if not isinstance(moving_pose, Mapping):
            raise _fail(
                "INTERPENETRATION_AUDIT_POSES_MISSING",
                "plain physical sample has no moving body pose",
            )
        if fixed_pose is not None and not isinstance(fixed_pose, Mapping):
            raise _fail(
                "INTERPENETRATION_AUDIT_POSES_MISSING",
                "plain physical sample has an invalid fixed body pose",
            )
        item["body_poses"] = {
            "fixed_boundary": (
                dict(fixed_pose)
                if isinstance(fixed_pose, Mapping)
                else {
                    "position_abs_m": fixed_position,
                    "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                }
            ),
            "moving_group": dict(moving_pose),
        }
        normalized_samples.append(item)
    return normalized_geometry, normalized_samples


def audit_physical_trace_evidence(
    physical_trace: Mapping[str, Any],
    *,
    workspace_root: str | Path,
    require_interface_policy: bool = True,
    require_wrench_audit: bool = True,
) -> dict[str, Any]:
    """Fail-closed physical-only gate that runs before any reduced branch."""

    if physical_trace.get("branch") != "physical":
        raise _fail(
            "PHYSICAL_PREFLIGHT_BRANCH_INVALID",
            "physical preflight received a non-physical trace",
        )
    geometry, samples = normalized_physical_audit_inputs(physical_trace)
    exclusions = disclosed_broad_collision_exclusions(physical_trace)
    failures: list[str] = []
    wrench_audit: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "passed": True,
    }
    if require_wrench_audit:
        from .wrench_ledger import verify_saved_wrench_audit

        try:
            wrench_audit = verify_saved_wrench_audit(
                physical_trace.get("wrench_audit")
            )
            wrench_audit = {
                **wrench_audit,
                "status": "PASS",
                "passed": True,
            }
        except MechanismValidationError as exc:
            wrench_audit = {
                "status": "FAIL",
                "passed": False,
                "failure_reasons": [exc.code],
                "detail": str(exc),
            }
            failures.append(exc.code)
    if exclusions:
        failures.append("physical_broad_body_pair_collision_exclusion_forbidden")

    semantics = physical_trace.get("branch_semantics")
    policy = semantics.get("physical_interface_policy") if isinstance(semantics, Mapping) else None
    validated_interfaces: tuple[dict[str, Any], ...] = ()
    if require_interface_policy:
        if not isinstance(policy, Mapping):
            failures.append("physical_interface_policy_missing")
        else:
            from .collision_policy import (
                collision_policy_disclosure,
                validate_interface_declarations,
            )

            body_ids = [
                str(item.get("body_id"))
                for item in geometry.get("bodies") or []
                if isinstance(item, Mapping) and item.get("body_id")
            ]
            validated_interfaces = validate_interface_declarations(
                policy.get("interface_declarations") or (),
                valid_body_ids=body_ids,
            )
            expected = collision_policy_disclosure(
                validated_interfaces,
                branch="physical",
            )
            for field in (
                "whole_body_pair_policy",
                "ai_authored_collision_masks_allowed",
                "validator_owned_local_exception_count",
                "interface_declarations_sha256",
                "conditional_interface_count",
                "functional_interface_geometry_binding_required",
                "analytical_transmission_geometry_gate_required",
                "compliant_support_geometry_gate_required",
                "ungated_analytical_transmission_count",
                "ungated_compliant_support_count",
            ):
                if policy.get(field) != expected[field]:
                    failures.append("physical_interface_policy_binding_invalid")
                    break

    functional_geometry_binding: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "passed": True,
        "conditional_interface_count": 0,
    }
    conditional_runtime_audit: dict[str, Any] = {
        "status": "NOT_REQUIRED",
        "passed": True,
        "conditional_interface_count": 0,
    }
    conditional_interfaces = [
        item
        for item in validated_interfaces
        if item.get("transmission_gate") is not None
    ]
    ungated_analytical_transmissions = [
        str(item["interface_id"])
        for item in validated_interfaces
        if item.get("classification") == "ANALYTICAL_TRANSMISSION"
        and item.get("transmission_gate") is None
    ]
    if ungated_analytical_transmissions:
        failures.append("analytical_transmission_geometry_gate_missing")
    ungated_compliant_supports = [
        str(item["interface_id"])
        for item in validated_interfaces
        if item.get("classification") == "COMPLIANT_INTERFACE_PHYSICAL"
        and item.get("transmission_gate") is None
    ]
    if ungated_compliant_supports:
        failures.append("compliant_support_geometry_gate_missing")
    if conditional_interfaces:
        from .conditional_interface import (
            audit_conditional_interface_runtime_evidence,
            validate_functional_interface_geometry_binding,
        )

        binding = (
            semantics.get("functional_interface_geometry_binding")
            if isinstance(semantics, Mapping)
            else None
        )
        try:
            functional_geometry_binding = (
                validate_functional_interface_geometry_binding(
                    binding,
                    interfaces=validated_interfaces,
                    collision_geometry=geometry,
                )
            )
            functional_geometry_binding = dict(functional_geometry_binding)
            functional_geometry_binding["status"] = "PASS"
            functional_geometry_binding["passed"] = True
        except MechanismValidationError as exc:
            functional_geometry_binding = {
                "status": "FAIL",
                "passed": False,
                "failure_reasons": [exc.code],
                "detail": str(exc),
            }
            failures.append("conditional_interface_geometry_binding_invalid")

        runtime_summary = (
            semantics.get("conditional_interface_runtime")
            if isinstance(semantics, Mapping)
            else None
        )
        conditional_runtime_audit = (
            audit_conditional_interface_runtime_evidence(
                validated_interfaces,
                samples,
                runtime_summary,
            )
        )
        for reason in conditional_runtime_audit.get("failure_reasons") or ():
            if reason not in failures:
                failures.append(str(reason))

    contact_policy = _component_contact_policy(
        physical_trace,
        geometry,
    )
    audit = audit_family_collision_geometry(
        geometry,
        samples,
        workspace_root=workspace_root,
        tolerance_m=runtime_interpenetration_tolerance_m(geometry),
        contact_intent_policy=contact_policy,
    )
    if not audit["passed"]:
        failures.append("physical_illegal_interpenetration")
    try:
        contact_intent = _audit_component_contact_policy(
            physical_trace,
            geometry,
            audit,
            policy=contact_policy,
        )
    except MechanismValidationError as exc:
        contact_intent = {
            "status": "INCONCLUSIVE",
            "passed": False,
            "authoritative": True,
            "failure_reasons": [exc.code],
            "detail": str(exc),
        }
    for reason in contact_intent.get("failure_reasons") or ():
        if reason not in failures:
            failures.append(reason)
    try:
        moving_chain_path_audit = _audit_moving_chain_path_if_required(
            physical_trace,
            geometry,
            samples,
            workspace_root=workspace_root,
        )
    except MechanismValidationError as exc:
        moving_chain_path_audit = {
            "schema_version": "bicycle_moving_chain_path_audit_v1",
            "status": "FAIL",
            "passed": False,
            "authoritative": True,
            "failure_reasons": [exc.code],
            "detail": str(exc),
        }
        if exc.code not in failures:
            failures.append(exc.code)
    if not moving_chain_path_audit.get("passed"):
        code = "bicycle_moving_chain_path_clearance_failed"
        if code not in failures:
            failures.append(code)
    semantics = physical_trace.get("branch_semantics")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    runtime_bindings = semantics.get("runtime_bindings")
    runtime_bindings = (
        runtime_bindings if isinstance(runtime_bindings, Mapping) else {}
    )
    bodies = {
        str(item.get("body_id")): item
        for item in geometry.get("bodies", [])
        if isinstance(item, Mapping) and item.get("body_id")
    }
    fixed_id = str(runtime_bindings.get("fixed_body_id") or "")
    input_id = str(runtime_bindings.get("input_body_id") or "")
    output_id = str(runtime_bindings.get("output_body_id") or "")
    role_ids_are_distinct = (
        bool(fixed_id and input_id and output_id)
        and len({fixed_id, input_id, output_id}) == 3
    )
    saved_role_poses_complete = bool(samples) and all(
        isinstance(sample.get("body_poses"), Mapping)
        and {fixed_id, input_id, output_id}.issubset(sample["body_poses"])
        for sample in samples
    )
    fixed_and_moving_passed = bool(
        role_ids_are_distinct
        and bodies.get(fixed_id, {}).get("fixed") is True
        and bodies.get(input_id, {}).get("fixed") is False
        and bodies.get(output_id, {}).get("fixed") is False
        and saved_role_poses_complete
    )
    indirect_passed = bool(
        wrench_audit.get("passed") is True
        and semantics.get("direct_output_actuation_used") is False
        and semantics.get("task_input_body_id") == input_id
        and semantics.get("unique_task_output_body_id") == output_id
        and wrench_audit.get("direct_output_drive_detected") is False
    )
    finite_force_path_passed = bool(
        wrench_audit.get("passed") is True
        and functional_geometry_binding.get("passed") is True
        and conditional_runtime_audit.get("passed") is True
        and not ungated_analytical_transmissions
        and not ungated_compliant_supports
    )
    functional_geometry = physical_trace.get("functional_geometry")
    geometry_exposed_passed = bool(
        isinstance(functional_geometry, Mapping)
        and functional_geometry.get("status") == "PASS"
        and bodies
        and all(
            isinstance(body.get("components"), list)
            and bool(body["components"])
            for body in bodies.values()
        )
        and moving_chain_path_audit.get("original_step_components_used") is True
    )
    saved_pose_legality_passed = bool(
        audit.get("passed") is True
        and contact_intent.get("passed") is True
        and functional_geometry_binding.get("passed") is True
        and conditional_runtime_audit.get("passed") is True
        and not exclusions
        and not ungated_analytical_transmissions
        and not ungated_compliant_supports
    )
    requirement_checks = {
        "validator.mechanical.saved_pose_legality": {
            "passed": saved_pose_legality_passed,
            "evidence_roles": [
                "validation.saved_pose_geometry_audit",
                "validation.original_step_collision_geometry",
            ],
        },
        "validator.actuation.indirect_only": {
            "passed": indirect_passed,
            "evidence_roles": [
                "validation.task_metric_evidence",
                "validation.wrench_audit",
            ],
        },
        "validator.mechanical.fixed_chassis_and_moving_groups": {
            "passed": fixed_and_moving_passed,
            "evidence_roles": ["validation.body_pose_trace"],
        },
        "validator.mechanical.unique_input_and_output": {
            "passed": indirect_passed and role_ids_are_distinct,
            "evidence_roles": ["validation.wrench_audit"],
        },
        "validator.mechanical.finite_support_and_transmission": {
            "passed": finite_force_path_passed,
            "evidence_roles": [
                "validation.conditional_interface",
                "validation.wrench_audit",
            ],
        },
        "validator.mechanical.audit_geometry_exposed": {
            "passed": geometry_exposed_passed,
            "evidence_roles": [
                "validation.moving_chain_path_audit",
                "validation.original_step_collision_geometry",
            ],
        },
        "validator.chain.moving_path_state": {
            "passed": moving_chain_path_audit.get("passed") is True,
            "evidence_roles": ["validation.moving_chain_path_audit"],
        },
        "validator.chain.saved_state_clearance": {
            "passed": bool(
                moving_chain_path_audit.get("passed") is True
                and moving_chain_path_audit.get(
                    "saved_state_clearance_verified"
                )
                is True
            ),
            "evidence_roles": [
                "validation.moving_chain_path_audit",
                "validation.original_step_collision_geometry",
            ],
        },
    }
    return {
        "schema_version": "0.1",
        "status": "PHYSICAL_PREFLIGHT_PASSED" if not failures else "PHYSICAL_PREFLIGHT_FAILED",
        "passed": not failures,
        "failure_reasons": failures,
        "physical_interface_declarations": list(validated_interfaces),
        "functional_interface_geometry_binding": functional_geometry_binding,
        "conditional_interface_runtime": conditional_runtime_audit,
        "wrench_audit": wrench_audit,
        "ungated_analytical_transmissions": ungated_analytical_transmissions,
        "broad_body_pair_collision_exclusions": exclusions,
        "geometry_legality": audit,
        "component_contact_intent": contact_intent,
        "moving_chain_path_audit": moving_chain_path_audit,
        "requirement_checks": requirement_checks,
        "reduced_branch_authorized": not failures,
    }


def audit_composite_lane_physical_trace(
    *,
    trace_path: str | Path,
    lane_path: str | Path,
    audit_path: str | Path,
    workspace_root: str | Path,
) -> dict[str, Any]:
    """Recompute one lane audit after its queue barrier in the CAD process.

    Final certification separately binds the plan, package, deferred worker
    marker, and aggregate outer preflight receipt.
    """

    workspace = Path(workspace_root).resolve()
    trace = Path(trace_path).resolve()
    lane_result_path = Path(lane_path).resolve()
    destination = Path(audit_path).resolve()
    for label, path in (
        ("physical trace", trace),
        ("lane result", lane_result_path),
        ("physical audit", destination),
    ):
        try:
            path.relative_to(workspace)
        except ValueError as exc:
            raise _fail(
                "COMPOSITE_LANE_PATH_INVALID",
                f"{label} must stay inside the validation workspace",
            ) from exc

    lane = read_json(lane_result_path)
    artifacts = lane.get("artifacts")
    trace_binding = (
        artifacts.get("physical_trace")
        if isinstance(artifacts, Mapping)
        else None
    )
    trace_relative = relative_to_root(workspace, trace)
    trace_digest = sha256_file(trace)
    capsule_id = lane.get("capsule_id")
    load_case_id = lane.get("load_case_id")
    trace_root = trace.parent
    output = trace_root.parent.parent.parent
    expected_lane_path = (
        output / "results" / str(capsule_id) / "lanes" / f"{load_case_id}.json"
    ).resolve()
    expected_audit_path = (
        output
        / "results"
        / str(capsule_id)
        / "physical_trace_audit"
        / f"{load_case_id}.json"
    ).resolve()
    binding_file = (
        trace_binding.get("file")
        if isinstance(trace_binding, Mapping)
        else None
    )
    binding_file_ok = binding_file == trace_relative
    if (
        not binding_file_ok
        and isinstance(binding_file, str)
        and isinstance(capsule_id, str)
        and isinstance(load_case_id, str)
    ):
        # A lane adopted from a sibling run keeps its origin path inside
        # the lane result. Foreign paths are accepted only when this run
        # carries a chain-verified sign-off receipt that binds exactly
        # this origin file and these trace bytes; a damaged receipt chain
        # raises and fails the audit closed.
        from .reuse import receipt_for_lane

        receipt = receipt_for_lane(
            output,
            capsule_id=capsule_id,
            load_case_id=load_case_id,
        )
        adopted = (
            receipt.get("artifacts", {}).get("physical_trace")
            if isinstance(receipt, Mapping)
            else None
        )
        binding_file_ok = (
            isinstance(adopted, Mapping)
            and adopted.get("origin_file") == binding_file
            and adopted.get("sha256") == trace_digest
        )
    if (
        lane.get("schema_version")
        != "simulation_ready_cad_mechanism_load_lane_v1"
        or lane.get("status") != "PASS"
        or lane.get("completed_phase") != "comparison"
        or not isinstance(capsule_id, str)
        or not capsule_id
        or not isinstance(load_case_id, str)
        or not load_case_id
        or trace.name != "physical.trace.json"
        or trace_root.name != load_case_id
        or trace_root.parent.name != capsule_id
        or trace_root.parent.parent.name != "traces"
        or lane_result_path != expected_lane_path
        or destination != expected_audit_path
        or not isinstance(trace_binding, Mapping)
        or not binding_file_ok
        or trace_binding.get("sha256") != trace_digest
        or (
            isinstance(artifacts, Mapping)
            and "physical_trace_audit" in artifacts
        )
    ):
        raise _fail(
            "COMPOSITE_LANE_EVIDENCE_BINDING_INVALID",
            "load lane did not PASS at the canonical path with exactly one "
            "bound physical trace",
        )

    physical_trace = read_trace_json(trace)
    if physical_trace.get("status") != "COMPLETED":
        raise _fail(
            "PHYSICAL_TRACE_NOT_COMPLETED",
            "composite lane physical trace is not completed",
        )
    audit = audit_physical_trace_evidence(
        physical_trace,
        workspace_root=workspace,
    )
    audit = {
        **audit,
        "physical_trace_sha256": trace_digest,
        "audit_execution_environment": "outer_cad_after_queue_barrier",
        "worker_produced_audit_accepted": False,
    }
    write_json_atomic(destination, audit)
    return {
        **audit,
        "trace": trace_relative,
        "trace_sha256": trace_digest,
        "physical_trace_audit": relative_to_root(workspace, destination),
        "physical_trace_audit_sha256": sha256_file(destination),
        "lane_status": lane.get("status"),
        "lane_completed_phase": lane.get("completed_phase"),
        "lane_result": relative_to_root(workspace, lane_result_path),
        "capsule_id": capsule_id,
        "load_case_id": load_case_id,
    }


def preflight_lane_audit_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Audit one lane's physical trace; safe to run in a worker process.

    2026-08-11 ATTENTION AI EDITORS: this is the CURRENT design, not a
    legacy leftover — do not treat it as old code to be "cleaned up".
    The physical preflight used to audit lanes serially inside
    ``harness.engine._physical_preflight``; it now fans out to a
    process pool, and this function is the pool entry point. It MUST
    stay a top-level function with a plain-dict in / plain-dict out
    contract so it remains picklable. The audit semantics are a red
    line and did NOT change: same canonical evidence paths, same audit
    fields, same hash contract, still executed by outer harness code
    after the queue barrier, and worker-produced audits are still never
    accepted. Only the concurrency changed. Do not alter semantics
    here under any "optimization" pretext.
    """

    workspace = Path(str(payload["workspace"])).resolve()
    trace_path = Path(str(payload["trace"]))
    relative = str(payload["relative"])
    try:
        if payload.get("kind") == "composite":
            audit = audit_composite_lane_physical_trace(
                trace_path=trace_path,
                lane_path=Path(str(payload["lane"])),
                audit_path=Path(str(payload["audit"])),
                workspace_root=workspace,
            )
            return {
                **dict(audit),
                "capsule_id": str(payload["capsule_id"]),
                "load_case_id": str(payload["load_case_id"]),
            }
        trace = read_trace_json(trace_path)
        if trace.get("status") != "COMPLETED":
            raise _fail(
                "PHYSICAL_TRACE_NOT_COMPLETED",
                "physical worker did not complete the trace",
            )
        audit = audit_physical_trace_evidence(trace, workspace_root=workspace)
        return {
            **audit,
            "capsule_id": str(trace.get("capsule_id") or ""),
            "load_case_id": str(trace.get("load_case_id") or ""),
            "trace": relative,
            "trace_sha256": sha256_file(trace_path),
        }
    except (OSError, ValueError, MechanismValidationError) as exc:
        return {
            "status": "PHYSICAL_PREFLIGHT_INCONCLUSIVE",
            "passed": False,
            "trace": relative,
            "failure_reasons": [
                str(getattr(exc, "code", type(exc).__name__))
            ],
            "detail": str(exc)[:4096],
        }
