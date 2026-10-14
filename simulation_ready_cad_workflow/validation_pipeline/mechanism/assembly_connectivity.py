from __future__ import annotations

"""Reject spatially isolated mechanism bodies from authoritative STEP meshes."""

import math
from typing import Any, Mapping, Sequence

from .models import MechanismValidationError


ASSEMBLY_CONNECTIVITY_SCHEMA = "assembly_spatial_connectivity_v1"
_MAX_BODY_PAIR_COUNT = 8_192


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(code, message)


def _positive_finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise _fail(
            "ASSEMBLY_CONNECTIVITY_TOLERANCE_INVALID",
            f"{label} must be positive and finite",
        )
    return result


def _connected_components(
    body_ids: Sequence[str],
    adjacency: Mapping[str, set[str]],
) -> list[list[str]]:
    remaining = set(body_ids)
    result: list[list[str]] = []
    while remaining:
        component: list[str] = []
        pending = [min(remaining)]
        while pending:
            body_id = pending.pop()
            if body_id not in remaining:
                continue
            remaining.remove(body_id)
            component.append(body_id)
            pending.extend(adjacency[body_id] & remaining)
        result.append(sorted(component))
    return sorted(result, key=lambda item: (item[0], len(item)))


def _collision_manager(
    body_id: str,
    components: Sequence[tuple[str, Any]],
) -> Any:
    import trimesh

    if not components:
        raise _fail(
            "ASSEMBLY_CONNECTIVITY_BODY_EMPTY",
            f"body {body_id!r} has no authoritative STEP mesh components",
        )
    manager = trimesh.collision.CollisionManager()
    for solid_id, mesh in components:
        if not isinstance(solid_id, str) or not solid_id:
            raise _fail(
                "ASSEMBLY_CONNECTIVITY_SOLID_ID_INVALID",
                f"body {body_id!r} contains an invalid solid ID",
            )
        if not isinstance(mesh, trimesh.Trimesh):
            raise _fail(
                "ASSEMBLY_CONNECTIVITY_MESH_INVALID",
                f"{body_id}/{solid_id} is not a triangle mesh",
            )
        if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise _fail(
                "ASSEMBLY_CONNECTIVITY_MESH_EMPTY",
                f"{body_id}/{solid_id} is an empty triangle mesh",
            )
        manager.add_object(solid_id, mesh)
    return manager


def build_assembly_connectivity_report(
    body_meshes: Mapping[str, Sequence[tuple[str, Any]]],
    *,
    package_binding_sha256: str,
    mesh_linear_deflection_m: float,
    collision_envelope_m: float,
    interpenetration_tolerance_m: float,
) -> dict[str, Any]:
    """Build a joint-agnostic, geometry-only assembly-island report.

    Nodes are mechanism rigid bodies.  An edge exists when the closest
    authoritative STEP-derived mesh surfaces overlap, touch, or are separated
    by no more than the validator-owned clearance budget.  Task objects and the
    environment are outside this inventory and therefore cannot make a
    disconnected mechanism pass.

    This is deliberately a gross assembly-connectivity gate.  It rejects
    floating bodies but does not claim that a nearby joint, bearing, or contact
    interface is mechanically correct.
    """

    if not isinstance(package_binding_sha256, str) or not package_binding_sha256:
        raise _fail(
            "ASSEMBLY_CONNECTIVITY_PACKAGE_BINDING_MISSING",
            "assembly connectivity requires a package binding",
        )
    deflection = _positive_finite(
        mesh_linear_deflection_m,
        "mesh_linear_deflection_m",
    )
    envelope = _positive_finite(
        collision_envelope_m,
        "collision_envelope_m",
    )
    penetration = _positive_finite(
        interpenetration_tolerance_m,
        "interpenetration_tolerance_m",
    )
    # Each of the two tessellated surfaces may deviate by one mesh deflection.
    # Collision envelope and the existing static overlap tolerance complete the
    # validator-owned allowance for a normal assembly clearance.
    clearance_budget_m = 2.0 * deflection + 2.0 * envelope + penetration

    normalized: dict[str, tuple[tuple[str, Any], ...]] = {}
    solid_owners: dict[str, str] = {}
    for body_id, components in body_meshes.items():
        if not isinstance(body_id, str) or not body_id:
            raise _fail(
                "ASSEMBLY_CONNECTIVITY_BODY_ID_INVALID",
                "assembly body IDs must be non-empty strings",
            )
        normalized_components = tuple(components)
        for solid_id, _mesh in normalized_components:
            previous = solid_owners.get(solid_id)
            if previous is not None and previous != body_id:
                raise _fail(
                    "ASSEMBLY_CONNECTIVITY_SOLID_OWNER_CONFLICT",
                    f"{solid_id!r} belongs to both {previous!r} and {body_id!r}",
                )
            solid_owners[solid_id] = body_id
        normalized[body_id] = normalized_components
    if not normalized:
        raise _fail(
            "ASSEMBLY_CONNECTIVITY_BODY_INVENTORY_EMPTY",
            "assembly connectivity received no mechanism bodies",
        )

    body_ids = sorted(normalized)
    pair_count = len(body_ids) * (len(body_ids) - 1) // 2
    if pair_count > _MAX_BODY_PAIR_COUNT:
        raise _fail(
            "ASSEMBLY_CONNECTIVITY_PAIR_BUDGET_EXCEEDED",
            "assembly body-pair distance budget exceeded",
        )
    managers = {
        body_id: _collision_manager(body_id, normalized[body_id])
        for body_id in body_ids
    }
    adjacency = {body_id: set() for body_id in body_ids}
    pair_measurements: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(body_ids):
        for right_id in body_ids[left_index + 1 :]:
            signed_distance_m, solid_pair = managers[left_id].min_distance_other(
                managers[right_id],
                return_names=True,
            )
            signed_distance_m = float(signed_distance_m)
            if not math.isfinite(signed_distance_m):
                raise _fail(
                    "ASSEMBLY_CONNECTIVITY_DISTANCE_FAILED",
                    f"mesh distance failed for {left_id!r} and {right_id!r}",
                )
            connected = signed_distance_m <= clearance_budget_m
            if connected:
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)
            pair_measurements.append(
                {
                    "body_pair": [left_id, right_id],
                    "closest_solid_pair": list(solid_pair),
                    "signed_distance_m": signed_distance_m,
                    "surface_gap_m": max(0.0, signed_distance_m),
                    "connected": connected,
                    "edge_reason": (
                        "INTERSECTING_OR_TOUCHING"
                        if signed_distance_m <= 0.0
                        else (
                            "WITHIN_VALIDATOR_CLEARANCE_BUDGET"
                            if connected
                            else "GAP_EXCEEDS_VALIDATOR_CLEARANCE_BUDGET"
                        )
                    ),
                }
            )

    components = _connected_components(body_ids, adjacency)
    passed = len(components) == 1
    bridges = [
        item
        for item in sorted(
            pair_measurements,
            key=lambda item: (
                float(item["surface_gap_m"]),
                tuple(item["body_pair"]),
            ),
        )
        if not item["connected"]
        and any(
            item["body_pair"][0] in component
            and item["body_pair"][1] not in component
            for component in components
        )
    ]
    failure_reasons: list[str] = []
    if not passed:
        failure_reasons.append(
            f"mechanism geometry forms {len(components)} spatial assembly islands"
        )
        if bridges:
            nearest = bridges[0]
            failure_reasons.append(
                "nearest disconnected bodies "
                f"{nearest['body_pair'][0]}/{nearest['closest_solid_pair'][0]} "
                "and "
                f"{nearest['body_pair'][1]}/{nearest['closest_solid_pair'][1]} "
                f"have a {float(nearest['surface_gap_m']) * 1000.0:.3f} mm gap"
            )

    return {
        "schema_version": ASSEMBLY_CONNECTIVITY_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "package_binding_sha256": package_binding_sha256,
        "source_geometry": "INTAKE_BOUND_AUTHORITATIVE_STEP_DERIVED_MESHES",
        "scope": "MECHANISM_BODY_INVENTORY_ONLY",
        "joint_declarations_used": False,
        "joint_correctness_certified": False,
        "task_objects_can_create_edges": False,
        "environment_can_create_edges": False,
        "physical_simulation_started": False,
        "clearance_policy": {
            "mesh_linear_deflection_m": deflection,
            "collision_envelope_m": envelope,
            "interpenetration_tolerance_m": penetration,
            "maximum_surface_gap_m": clearance_budget_m,
            "formula": (
                "2*mesh_linear_deflection_m + 2*collision_envelope_m + "
                "interpenetration_tolerance_m"
            ),
            "machine_authored_override_allowed": False,
        },
        "body_count": len(body_ids),
        "solid_count": len(solid_owners),
        "body_pair_count": pair_count,
        "connected_edge_count": sum(
            1 for item in pair_measurements if item["connected"]
        ),
        "assembly_island_count": len(components),
        "assembly_islands": components,
        "pair_measurements": pair_measurements,
        "failure_reasons": failure_reasons,
    }
