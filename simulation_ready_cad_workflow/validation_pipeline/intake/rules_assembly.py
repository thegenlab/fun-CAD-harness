from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .context import PART_ROLES, ValidationContext


def _label(ctx: ValidationContext) -> str:
    path = ctx.paths.get("assembly_contract")
    return (
        ctx.relative_label(path)
        if isinstance(path, Path)
        else "assembly_contract.json"
    )


def _string_items(
    ctx: ValidationContext,
    value: Any,
    *,
    file: str,
    json_path: str,
    nonempty: bool = True,
) -> list[str] | None:
    if (
        not isinstance(value, list)
        or (nonempty and not value)
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        ctx.error(
            "STRING_LIST_INVALID",
            "Expected an array of non-empty strings.",
            file=file,
            json_path=json_path,
            expected="non-empty string array" if nonempty else "string array",
            actual=value,
        )
        return None
    seen: set[str] = set()
    for index, item in enumerate(value):
        if item in seen:
            ctx.error(
                "STRING_LIST_DUPLICATE",
                "String array entries must be unique.",
                file=file,
                json_path=f"{json_path}[{index}]",
                actual=item,
                hint="Remove the duplicate reference.",
            )
        seen.add(item)
    return value


def _validate_units(ctx: ValidationContext, file: str) -> None:
    assert ctx.assembly is not None
    units = ctx.assembly.get("units")
    expected = {"length": "m", "mass": "kg", "angle": "rad"}
    if not isinstance(units, dict):
        ctx.error(
            "ASSEMBLY_UNITS_MISSING",
            "Assembly contract must declare canonical SI units.",
            file=file,
            json_path="$.units",
            expected=expected,
            actual=units,
        )
        return
    ctx.warn_unknown_fields(
        units,
        {"length", "mass", "angle"},
        file=file,
        json_path="$.units",
    )
    for key, expected_value in expected.items():
        if units.get(key) != expected_value:
            ctx.error(
                "ASSEMBLY_UNIT_INVALID",
                f"Assembly {key} unit must use the pipeline canonical unit.",
                file=file,
                json_path=f"$.units.{key}",
                expected=expected_value,
                actual=units.get(key),
                hint=(
                    "Convert values before packaging; do not rely on implicit "
                    "units."
                ),
            )


def _validate_world_frame(ctx: ValidationContext, file: str) -> None:
    assert ctx.assembly is not None
    frame = ctx.assembly.get("world_frame")
    if not isinstance(frame, dict):
        ctx.error(
            "WORLD_FRAME_MISSING",
            "Assembly contract must define a world frame and gravity.",
            file=file,
            json_path="$.world_frame",
            expected={
                "right_handed": True,
                "up_axis": "+Z",
                "gravity_m_s2": [0, 0, -9.81],
            },
            actual=frame,
        )
        return
    ctx.warn_unknown_fields(
        frame,
        {"right_handed", "up_axis", "gravity_m_s2"},
        file=file,
        json_path="$.world_frame",
    )
    if frame.get("right_handed") is not True:
        ctx.error(
            "WORLD_FRAME_HANDEDNESS_INVALID",
            "World frame must be explicitly right handed.",
            file=file,
            json_path="$.world_frame.right_handed",
            expected=True,
            actual=frame.get("right_handed"),
        )
    if frame.get("up_axis") != "+Z":
        ctx.error(
            "WORLD_UP_AXIS_INVALID",
            "World up axis must use the canonical +Z convention.",
            file=file,
            json_path="$.world_frame.up_axis",
            expected="+Z",
            actual=frame.get("up_axis"),
        )
    gravity = ctx.vector(
        frame.get("gravity_m_s2"),
        length=3,
        file=file,
        json_path="$.world_frame.gravity_m_s2",
    )
    if gravity is not None and (
        abs(gravity[0]) > 1e-6
        or abs(gravity[1]) > 1e-6
        or abs(gravity[2] + 9.81) > 0.05
    ):
        ctx.error(
            "WORLD_GRAVITY_INVALID",
            "Schema 0.1 uses standard Earth gravity along world -Z.",
            file=file,
            json_path="$.world_frame.gravity_m_s2",
            expected=[0.0, 0.0, -9.81],
            actual=gravity,
            hint=(
                "Use canonical Earth gravity. A different environment must "
                "be introduced by a future external-requirements contract, "
                "not chosen by the AI design package."
            ),
        )


def _validate_declared_part_bounds(
    ctx: ValidationContext,
    part: dict[str, Any],
    solid_ids: list[str] | None,
    *,
    file: str,
    base: str,
) -> None:
    value = part.get("declared_bounds_m")
    if not isinstance(value, dict):
        ctx.error(
            "PART_BOUNDS_MISSING",
            "Part must declare world-aligned bounds in metres.",
            file=file,
            json_path=f"{base}.declared_bounds_m",
            expected={"min": [0, 0, 0], "max": [0, 0, 0]},
            actual=value,
            hint=(
                "Copy the bounds from step_solid_inventory, convert mm to m, "
                "and union all source solids assigned to this part."
            ),
        )
        return
    ctx.warn_unknown_fields(
        value,
        {"min", "max"},
        file=file,
        json_path=f"{base}.declared_bounds_m",
    )
    minimum = ctx.vector(
        value.get("min"),
        length=3,
        file=file,
        json_path=f"{base}.declared_bounds_m.min",
        code="PART_BOUNDS_VECTOR_INVALID",
    )
    maximum = ctx.vector(
        value.get("max"),
        length=3,
        file=file,
        json_path=f"{base}.declared_bounds_m.max",
        code="PART_BOUNDS_VECTOR_INVALID",
    )
    if minimum is None or maximum is None:
        return
    if any(minimum[index] > maximum[index] for index in range(3)):
        ctx.error(
            "PART_BOUNDS_ORDER_INVALID",
            "Part bounds min must not exceed max on any axis.",
            file=file,
            json_path=f"{base}.declared_bounds_m",
            expected="min <= max on x, y, and z",
            actual=value,
        )
        return
    if not ctx.step_geometry_verified or not solid_ids:
        return
    solid_facts = [ctx.step_solid_inventory.get(item) for item in solid_ids]
    if any(facts is None for facts in solid_facts):
        return
    bounds = [facts["bounds_mm"] for facts in solid_facts if facts is not None]
    actual_minimum = [
        min(float(item["min"][axis]) for item in bounds) / 1000.0
        for axis in range(3)
    ]
    actual_maximum = [
        max(float(item["max"][axis]) for item in bounds) / 1000.0
        for axis in range(3)
    ]
    declared_size = [maximum[index] - minimum[index] for index in range(3)]
    actual_size = [
        actual_maximum[index] - actual_minimum[index] for index in range(3)
    ]
    scale = max([*declared_size, *actual_size, 1e-3])
    tolerance = max(1e-6, scale * 1e-4)
    differences = [
        abs(minimum[index] - actual_minimum[index])
        for index in range(3)
    ] + [
        abs(maximum[index] - actual_maximum[index])
        for index in range(3)
    ]
    if max(differences) > tolerance:
        ctx.error(
            "PART_BOUNDS_MISMATCH",
            "Declared metric bounds do not match imported STEP geometry.",
            file=file,
            json_path=f"{base}.declared_bounds_m",
            expected={"min": actual_minimum, "max": actual_maximum},
            actual={"min": minimum, "max": maximum},
            hint=(
                "Fix CAD export units or update stale bounds; a 1000x mismatch "
                "usually means metres were modeled as millimetres."
            ),
        )


def _validate_parts(ctx: ValidationContext, file: str) -> None:
    assert ctx.assembly is not None
    parts = ctx.require_list(ctx.assembly, "parts", file=file, nonempty=True)
    if parts is None:
        return
    occurrences: set[str] = set()
    solid_owners: dict[str, str] = {}
    for index, part in enumerate(parts):
        base = f"$.parts[{index}]"
        if not isinstance(part, dict):
            ctx.error(
                "PART_ENTRY_INVALID",
                "Each part entry must be an object.",
                file=file,
                json_path=base,
                expected="object",
                actual=part,
            )
            continue
        ctx.warn_unknown_fields(
            part,
            {
                "part_id",
                "occurrence_id",
                "rigid_group_id",
                "role",
                "source_solid_ids",
                "declared_bounds_m",
                "material",
            },
            file=file,
            json_path=base,
        )
        part_id = ctx.require_id(part, "part_id", file=file, base_path=base)
        occurrence_id = ctx.require_string(
            part, "occurrence_id", file=file, base_path=base
        )
        group_id = ctx.require_id(
            part, "rigid_group_id", file=file, base_path=base
        )
        role = part.get("role")
        if role not in PART_ROLES:
            ctx.error(
                "PART_ROLE_INVALID",
                "Part role is not recognized.",
                file=file,
                json_path=f"{base}.role",
                expected=sorted(PART_ROLES),
                actual=role,
            )
        elif role == "unknown":
            ctx.warning(
                "PART_ROLE_UNKNOWN",
                "Part role remains unknown.",
                file=file,
                json_path=f"{base}.role",
                hint=(
                    "Classify the part as fixed, dynamic, context, or "
                    "hardware."
                ),
            )
        solid_ids = _string_items(
            ctx,
            part.get("source_solid_ids"),
            file=file,
            json_path=f"{base}.source_solid_ids",
        )
        owner = part_id or f"parts[{index}]"
        for solid_index, solid_id in enumerate(solid_ids or []):
            solid_path = f"{base}.source_solid_ids[{solid_index}]"
            previous = solid_owners.get(solid_id)
            if previous is not None:
                ctx.error(
                    "SOURCE_SOLID_ASSIGNED_MORE_THAN_ONCE",
                    "A STEP solid must belong to exactly one declared part.",
                    file=file,
                    json_path=solid_path,
                    actual={"solid_id": solid_id, "owners": [previous, owner]},
                    hint="Split or merge part declarations without reusing a solid ID.",
                )
            else:
                solid_owners[solid_id] = owner
            if ctx.step_geometry_verified and solid_id not in ctx.step_solid_ids:
                ctx.error(
                    "SOURCE_SOLID_UNKNOWN",
                    "Part references a solid ID not produced by STEP import.",
                    file=file,
                    json_path=solid_path,
                    expected=sorted(ctx.step_solid_ids),
                    actual=solid_id,
                    hint=(
                        "Use the canonical solid_N IDs listed in "
                        "package_summary.step_solid_inventory."
                    ),
                )
        _validate_declared_part_bounds(
            ctx,
            part,
            solid_ids,
            file=file,
            base=base,
        )
        if part_id:
            if part_id in ctx.part_ids:
                ctx.error(
                    "PART_ID_DUPLICATE",
                    "part_id must be unique.",
                    file=file,
                    json_path=f"{base}.part_id",
                    actual=part_id,
                )
            else:
                ctx.part_ids.add(part_id)
                if isinstance(role, str):
                    ctx.part_roles[part_id] = role
                if group_id:
                    ctx.part_groups[part_id] = group_id
        if occurrence_id:
            if occurrence_id in occurrences:
                ctx.error(
                    "OCCURRENCE_ID_DUPLICATE",
                    "occurrence_id must identify one CAD occurrence.",
                    file=file,
                    json_path=f"{base}.occurrence_id",
                    actual=occurrence_id,
                )
            occurrences.add(occurrence_id)
        material = part.get("material")
        if not isinstance(material, dict):
            ctx.warning(
                "PART_MATERIAL_MISSING",
                "Part has no material and density claim.",
                file=file,
                json_path=f"{base}.material",
                hint=(
                    "Add material.name and a positive density_kg_m3 for "
                    "mass-property preflight."
                ),
            )
        else:
            ctx.warn_unknown_fields(
                material,
                {"name", "density_kg_m3"},
                file=file,
                json_path=f"{base}.material",
            )
            ctx.require_string(
                material,
                "name",
                file=file,
                base_path=f"{base}.material",
            )
            density = material.get("density_kg_m3")
            invalid_density = (
                isinstance(density, bool)
                or not isinstance(density, (int, float))
                or not math.isfinite(float(density))
                or density <= 0
            )
            if invalid_density:
                ctx.error(
                    "MATERIAL_DENSITY_INVALID",
                    "Material density must be a positive number.",
                    file=file,
                    json_path=f"{base}.material.density_kg_m3",
                    expected="number > 0",
                    actual=density,
                )
    if ctx.step_geometry_verified:
        missing = sorted(ctx.step_solid_ids - set(solid_owners))
        if missing:
            ctx.error(
                "STEP_SOLID_UNASSIGNED",
                "Every imported STEP solid must belong to one declared part.",
                file=file,
                json_path="$.parts",
                expected=sorted(ctx.step_solid_ids),
                actual={"unassigned": missing},
                hint="Add each missing canonical solid_N ID to exactly one part.",
            )


def _validate_rigid_groups(ctx: ValidationContext, file: str) -> None:
    assert ctx.assembly is not None
    groups = ctx.require_list(
        ctx.assembly, "rigid_groups", file=file, nonempty=True
    )
    if groups is None:
        return
    membership: dict[str, str] = {}
    for index, group in enumerate(groups):
        base = f"$.rigid_groups[{index}]"
        if not isinstance(group, dict):
            ctx.error(
                "RIGID_GROUP_ENTRY_INVALID",
                "Each rigid group must be an object.",
                file=file,
                json_path=base,
                actual=group,
            )
            continue
        ctx.warn_unknown_fields(
            group,
            {"rigid_group_id", "part_ids", "connection_evidence"},
            file=file,
            json_path=base,
        )
        group_id = ctx.require_id(
            group, "rigid_group_id", file=file, base_path=base
        )
        part_ids = _string_items(
            ctx,
            group.get("part_ids"),
            file=file,
            json_path=f"{base}.part_ids",
        )
        if group_id:
            if group_id in ctx.group_ids:
                ctx.error(
                    "RIGID_GROUP_ID_DUPLICATE",
                    "rigid_group_id must be unique.",
                    file=file,
                    json_path=f"{base}.rigid_group_id",
                    actual=group_id,
                )
            ctx.group_ids.add(group_id)
        if part_ids is None:
            continue
        if len(part_ids) > 1:
            evidence = group.get("connection_evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                ctx.error(
                    "RIGID_GROUP_EVIDENCE_INVALID",
                    "A multi-part rigid group needs textual connection evidence.",
                    file=file,
                    json_path=f"{base}.connection_evidence",
                    expected="non-empty evidence string",
                    actual=evidence,
                    hint=(
                        "Name the fused solid, fastener, weld, adhesive, "
                        "press-fit, or native fixed mate; otherwise split "
                        "the group."
                    ),
                )
        roles = {ctx.part_roles.get(part_id) for part_id in part_ids}
        if {"fixed", "dynamic"}.issubset(roles):
            ctx.error(
                "RIGID_GROUP_ROLE_CONFLICT",
                "Fixed and dynamic parts cannot share one rigid group.",
                file=file,
                json_path=f"{base}.part_ids",
                actual={
                    part_id: ctx.part_roles.get(part_id) for part_id in part_ids
                },
                hint="Split physically separate moving and fixed parts into distinct groups.",
            )
        for part_index, part_id in enumerate(part_ids):
            path = f"{base}.part_ids[{part_index}]"
            if part_id not in ctx.part_ids:
                ctx.error(
                    "RIGID_GROUP_PART_UNKNOWN",
                    "Rigid group references an unknown part.",
                    file=file,
                    json_path=path,
                    actual=part_id,
                )
                continue
            previous = membership.get(part_id)
            if previous is not None and previous != group_id:
                ctx.error(
                    "PART_IN_MULTIPLE_RIGID_GROUPS",
                    "A part cannot belong to multiple rigid groups.",
                    file=file,
                    json_path=path,
                    actual={
                        "part_id": part_id,
                        "groups": [previous, group_id],
                    },
                )
            elif group_id:
                membership[part_id] = group_id
            declared = ctx.part_groups.get(part_id)
            if group_id and declared != group_id:
                ctx.error(
                    "PART_GROUP_MISMATCH",
                    "Part and rigid-group declarations disagree.",
                    file=file,
                    json_path=path,
                    expected=declared,
                    actual=group_id,
                )
    for part_id in sorted(ctx.part_ids):
        if part_id not in membership:
            ctx.error(
                "PART_NOT_IN_RIGID_GROUP",
                "Every part must appear in exactly one rigid group.",
                file=file,
                json_path="$.rigid_groups",
                actual=part_id,
            )


def _validate_grounding_and_connections(
    ctx: ValidationContext, file: str
) -> None:
    assert ctx.assembly is not None
    grounded = _string_items(
        ctx,
        ctx.assembly.get("grounded_rigid_group_ids"),
        file=file,
        json_path="$.grounded_rigid_group_ids",
        nonempty=False,
    )
    grounded_set = set(grounded or [])
    if not grounded_set:
        ctx.warning(
            "GROUNDED_GROUP_MISSING",
            "No rigid group is declared grounded.",
            file=file,
            json_path="$.grounded_rigid_group_ids",
            hint="Declare the support or world-fixed group.",
        )
    for index, group_id in enumerate(grounded or []):
        if group_id not in ctx.group_ids:
            ctx.error(
                "GROUNDED_GROUP_UNKNOWN",
                "Grounding references an unknown rigid group.",
                file=file,
                json_path=f"$.grounded_rigid_group_ids[{index}]",
                actual=group_id,
            )
    for part_id, role in ctx.part_roles.items():
        group_id = ctx.part_groups.get(part_id)
        if role == "dynamic" and group_id in grounded_set:
            ctx.error(
                "DYNAMIC_GROUP_GROUNDED",
                "A dynamic part cannot be in a grounded rigid group.",
                file=file,
                json_path="$.grounded_rigid_group_ids",
                actual={"part_id": part_id, "rigid_group_id": group_id},
            )
        if role == "fixed" and group_id not in grounded_set:
            ctx.warning(
                "FIXED_GROUP_NOT_GROUNDED",
                "A part declared fixed is not in a grounded group.",
                file=file,
                actual={"part_id": part_id, "rigid_group_id": group_id},
            )

    connections = ctx.assembly.get("connections")
    if not isinstance(connections, list) or not connections:
        ctx.warning(
            "ASSEMBLY_CONNECTIONS_MISSING",
            "No inter-group connection evidence was supplied.",
            file=file,
            json_path="$.connections",
            hint=(
                "List joints, contacts, fasteners, or other physical "
                "relations between rigid groups."
            ),
        )
        return
    seen: set[str] = set()
    for index, connection in enumerate(connections):
        base = f"$.connections[{index}]"
        if not isinstance(connection, dict):
            ctx.error(
                "CONNECTION_ENTRY_INVALID",
                "Each connection must be an object.",
                file=file,
                json_path=base,
                actual=connection,
            )
            continue
        ctx.warn_unknown_fields(
            connection,
            {
                "connection_id",
                "between_rigid_group_ids",
                "connection_type",
                "evidence",
            },
            file=file,
            json_path=base,
        )
        connection_id = ctx.require_id(
            connection, "connection_id", file=file, base_path=base
        )
        if connection_id:
            if connection_id in seen:
                ctx.error(
                    "CONNECTION_ID_DUPLICATE",
                    "connection_id must be unique.",
                    file=file,
                    json_path=f"{base}.connection_id",
                    actual=connection_id,
                )
            seen.add(connection_id)
        groups = _string_items(
            ctx,
            connection.get("between_rigid_group_ids"),
            file=file,
            json_path=f"{base}.between_rigid_group_ids",
        )
        if groups is not None and len(groups) != 2:
            ctx.error(
                "CONNECTION_ENDPOINT_COUNT_INVALID",
                "A connection must name exactly two rigid groups.",
                file=file,
                json_path=f"{base}.between_rigid_group_ids",
                expected=2,
                actual=len(groups),
            )
        for endpoint_index, group_id in enumerate(groups or []):
            if group_id not in ctx.group_ids:
                ctx.error(
                    "CONNECTION_GROUP_UNKNOWN",
                    "Connection references an unknown rigid group.",
                    file=file,
                    json_path=(
                        f"{base}.between_rigid_group_ids[{endpoint_index}]"
                    ),
                    actual=group_id,
                )
        ctx.require_string(
            connection,
            "connection_type",
            file=file,
            base_path=base,
            code="CONNECTION_TYPE_REQUIRED",
        )
        evidence = connection.get("evidence")
        if evidence is None:
            ctx.warning(
                "CONNECTION_EVIDENCE_MISSING",
                "Connection has no supporting CAD or design evidence.",
                file=file,
                json_path=f"{base}.evidence",
                hint=(
                    "Reference a mate, contact surface, fastener, weld, or "
                    "design note."
                ),
            )
        elif not isinstance(evidence, str) or not evidence.strip():
            ctx.error(
                "CONNECTION_EVIDENCE_INVALID",
                "Connection evidence must be a non-empty string.",
                file=file,
                json_path=f"{base}.evidence",
                expected="non-empty evidence string",
                actual=evidence,
            )


def _validate_frames(ctx: ValidationContext, file: str) -> None:
    assert ctx.assembly is not None
    frames = ctx.require_list(
        ctx.assembly, "frames", file=file, nonempty=True
    )
    if frames is None:
        return
    for index, frame in enumerate(frames):
        base = f"$.frames[{index}]"
        if not isinstance(frame, dict):
            ctx.error(
                "FRAME_ENTRY_INVALID",
                "Each frame must be an object.",
                file=file,
                json_path=base,
                actual=frame,
            )
            continue
        ctx.warn_unknown_fields(
            frame,
            {
                "frame_id",
                "part_id",
                "rigid_group_id",
                "origin_m",
                "orientation_xyzw",
            },
            file=file,
            json_path=base,
        )
        frame_id = ctx.require_id(
            frame, "frame_id", file=file, base_path=base
        )
        part_id = frame.get("part_id")
        group_id = frame.get("rigid_group_id")
        has_part = isinstance(part_id, str)
        has_group = isinstance(group_id, str)
        if has_part == has_group:
            ctx.error(
                "FRAME_OWNER_INVALID",
                "A frame must name exactly one part_id or rigid_group_id.",
                file=file,
                json_path=base,
                expected="exactly one owner",
                actual={"part_id": part_id, "rigid_group_id": group_id},
            )
            owner_group = None
        elif has_part:
            owner_group = ctx.part_groups.get(part_id)
            if part_id not in ctx.part_ids:
                ctx.error(
                    "FRAME_PART_UNKNOWN",
                    "Frame references an unknown part.",
                    file=file,
                    json_path=f"{base}.part_id",
                    actual=part_id,
                )
        else:
            owner_group = group_id
            if group_id not in ctx.group_ids:
                ctx.error(
                    "FRAME_GROUP_UNKNOWN",
                    "Frame references an unknown rigid group.",
                    file=file,
                    json_path=f"{base}.rigid_group_id",
                    actual=group_id,
                )
        if frame_id:
            if frame_id in ctx.frame_ids:
                ctx.error(
                    "FRAME_ID_DUPLICATE",
                    "frame_id must be unique.",
                    file=file,
                    json_path=f"{base}.frame_id",
                    actual=frame_id,
                )
            ctx.frame_ids.add(frame_id)
            ctx.frame_groups[frame_id] = owner_group
        ctx.vector(
            frame.get("origin_m"),
            length=3,
            file=file,
            json_path=f"{base}.origin_m",
        )
        ctx.vector(
            frame.get("orientation_xyzw"),
            length=4,
            file=file,
            json_path=f"{base}.orientation_xyzw",
            nonzero=True,
            normalized=True,
            normalization_code="QUATERNION_NOT_UNIT",
        )


def validate_assembly(ctx: ValidationContext) -> None:
    if ctx.assembly is None:
        return
    file = _label(ctx)
    ctx.check_schema_version(ctx.assembly, file)
    ctx.scan_for_forbidden_authority(ctx.assembly, file=file)
    ctx.warn_unknown_fields(
        ctx.assembly,
        {
            "schema_version",
            "units",
            "world_frame",
            "parts",
            "rigid_groups",
            "grounded_rigid_group_ids",
            "connections",
            "frames",
        },
        file=file,
    )
    _validate_units(ctx, file)
    _validate_world_frame(ctx, file)
    _validate_parts(ctx, file)
    _validate_rigid_groups(ctx, file)
    _validate_grounding_and_connections(ctx, file)
    _validate_frames(ctx, file)
    ctx.report.package_summary.update(
        {
            "part_count": len(ctx.part_ids),
            "rigid_group_count": len(ctx.group_ids),
            "frame_count": len(ctx.frame_ids),
        }
    )
