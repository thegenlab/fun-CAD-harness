from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .context import (
    REDUCED_MODEL_FAMILIES,
    WRENCH_TYPES,
    ValidationContext,
)
from .rules_assembly import _string_items


AXIS_MODEL_FAMILIES = {
    "prismatic",
    "revolute",
    "cylindrical",
    "helical",
    "planar",
}
CANONICAL_RANGE_UNITS = {"force": "N", "torque": "N*m"}


def _same_frame_group(
    ctx: ValidationContext,
    *,
    file: str,
    base: str,
    frame_id: Any,
    group_id: Any,
) -> None:
    if not isinstance(frame_id, str) or frame_id not in ctx.frame_ids:
        ctx.error(
            "CAPSULE_FRAME_UNKNOWN",
            "Capsule port or observable references an unknown frame.",
            file=file,
            json_path=f"{base}.frame_id",
            actual=frame_id,
        )
        return
    frame_group = ctx.frame_groups.get(frame_id)
    if isinstance(group_id, str) and frame_group != group_id:
        ctx.error(
            "CAPSULE_FRAME_GROUP_MISMATCH",
            "Frame owner and declared rigid group disagree.",
            file=file,
            json_path=f"{base}.frame_id",
            expected=group_id,
            actual=frame_group,
        )


def _validate_range(
    ctx: ValidationContext,
    value: Any,
    *,
    file: str,
    base: str,
    wrench_type: Any,
) -> bool:
    if value is None:
        ctx.warning(
            "INPUT_RANGE_MISSING",
            "Input port has no declared safe or intended load range.",
            file=file,
            json_path=f"{base}.declared_range",
            hint="Add min, max, and unit without claiming validator thresholds.",
        )
        return False
    if not isinstance(value, dict):
        ctx.error(
            "INPUT_RANGE_INVALID",
            "declared_range must be an object.",
            file=file,
            json_path=f"{base}.declared_range",
            expected={"min": 0, "max": 1, "unit": "N"},
            actual=value,
        )
        return False
    ctx.warn_unknown_fields(
        value,
        {"min", "max", "unit"},
        file=file,
        json_path=f"{base}.declared_range",
    )
    minimum = value.get("min")
    maximum = value.get("max")
    valid = True
    for key, number in (("min", minimum), ("max", maximum)):
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
        ):
            ctx.error(
                "INPUT_RANGE_BOUND_INVALID",
                f"declared_range.{key} must be finite.",
                file=file,
                json_path=f"{base}.declared_range.{key}",
                actual=number,
            )
            valid = False
    if (
        isinstance(minimum, (int, float))
        and not isinstance(minimum, bool)
        and isinstance(maximum, (int, float))
        and not isinstance(maximum, bool)
        and minimum > maximum
    ):
        ctx.error(
            "INPUT_RANGE_ORDER_INVALID",
            "declared_range min must not exceed max.",
            file=file,
            json_path=f"{base}.declared_range",
            expected="min <= max",
            actual={"min": minimum, "max": maximum},
        )
        valid = False
    unit = value.get("unit")
    if not isinstance(unit, str) or not unit.strip():
        ctx.error(
            "INPUT_RANGE_UNIT_MISSING",
            "declared_range needs an explicit unit.",
            file=file,
            json_path=f"{base}.declared_range.unit",
            actual=unit,
        )
        valid = False
    else:
        expected_unit = CANONICAL_RANGE_UNITS.get(wrench_type)
        if expected_unit is not None and unit != expected_unit:
            ctx.error(
                "INPUT_RANGE_UNIT_MISMATCH",
                "declared_range unit does not match wrench_type.",
                file=file,
                json_path=f"{base}.declared_range.unit",
                expected=expected_unit,
                actual=unit,
                hint=(
                    "Use N for force and N*m for torque; convert the numeric "
                    "range before packaging."
                ),
            )
            valid = False
    return valid


def _validate_ports(
    ctx: ValidationContext,
    capsule: dict[str, Any],
    *,
    file: str,
    capsule_id: str,
    members: set[str],
) -> None:
    ports = ctx.require_list(
        capsule, "input_ports", file=file, nonempty=True
    )
    seen: set[str] = set()
    ctx.capsule_port_ids[capsule_id] = seen
    if ports is None:
        return
    for index, port in enumerate(ports):
        base = f"$.input_ports[{index}]"
        if not isinstance(port, dict):
            ctx.error(
                "INPUT_PORT_ENTRY_INVALID",
                "Each input port must be an object.",
                file=file,
                json_path=base,
                actual=port,
            )
            continue
        ctx.warn_unknown_fields(
            port,
            {
                "port_id",
                "rigid_group_id",
                "frame_id",
                "wrench_type",
                "direction_abs",
                "declared_range",
            },
            file=file,
            json_path=base,
        )
        port_id = ctx.require_id(
            port, "port_id", file=file, base_path=base
        )
        if port_id:
            if port_id in seen:
                ctx.error(
                    "INPUT_PORT_ID_DUPLICATE",
                    "port_id must be unique within a capsule.",
                    file=file,
                    json_path=f"{base}.port_id",
                    actual=port_id,
                )
            seen.add(port_id)
        group_id = port.get("rigid_group_id")
        if group_id not in ctx.group_ids:
            ctx.error(
                "INPUT_PORT_GROUP_UNKNOWN",
                "Input port references an unknown rigid group.",
                file=file,
                json_path=f"{base}.rigid_group_id",
                actual=group_id,
            )
        elif group_id not in members:
            ctx.error(
                "INPUT_PORT_NOT_ON_MEMBER",
                "Input port must act on a capsule member group.",
                file=file,
                json_path=f"{base}.rigid_group_id",
                expected=sorted(members),
                actual=group_id,
            )
        _same_frame_group(
            ctx,
            file=file,
            base=base,
            frame_id=port.get("frame_id"),
            group_id=group_id,
        )
        wrench_type = port.get("wrench_type")
        if wrench_type not in WRENCH_TYPES:
            ctx.error(
                "WRENCH_TYPE_INVALID",
                "Input port wrench_type is not supported.",
                file=file,
                json_path=f"{base}.wrench_type",
                expected=sorted(WRENCH_TYPES),
                actual=wrench_type,
            )
        ctx.vector(
            port.get("direction_abs"),
            length=3,
            file=file,
            json_path=f"{base}.direction_abs",
            nonzero=True,
            normalized=True,
            normalization_code="DIRECTION_NOT_UNIT",
        )
        _validate_range(
            ctx,
            port.get("declared_range"),
            file=file,
            base=base,
            wrench_type=wrench_type,
        )


def _validate_observables(
    ctx: ValidationContext,
    capsule: dict[str, Any],
    *,
    file: str,
    capsule_id: str,
    members: set[str],
) -> None:
    observables = ctx.require_list(
        capsule, "observables", file=file, nonempty=True
    )
    seen: set[str] = set()
    ctx.capsule_observable_ids[capsule_id] = seen
    if observables is None:
        return
    for index, observable in enumerate(observables):
        base = f"$.observables[{index}]"
        if not isinstance(observable, dict):
            ctx.error(
                "OBSERVABLE_ENTRY_INVALID",
                "Each observable must be an object.",
                file=file,
                json_path=base,
                actual=observable,
            )
            continue
        ctx.warn_unknown_fields(
            observable,
            {
                "observable_id",
                "rigid_group_id",
                "frame_id",
                "quantity",
            },
            file=file,
            json_path=base,
        )
        observable_id = ctx.require_id(
            observable, "observable_id", file=file, base_path=base
        )
        if observable_id:
            if observable_id in seen:
                ctx.error(
                    "OBSERVABLE_ID_DUPLICATE",
                    "observable_id must be unique within a capsule.",
                    file=file,
                    json_path=f"{base}.observable_id",
                    actual=observable_id,
                )
            seen.add(observable_id)
        group_id = observable.get("rigid_group_id")
        if group_id not in ctx.group_ids:
            ctx.error(
                "OBSERVABLE_GROUP_UNKNOWN",
                "Observable references an unknown rigid group.",
                file=file,
                json_path=f"{base}.rigid_group_id",
                actual=group_id,
            )
        elif group_id not in members:
            ctx.error(
                "OBSERVABLE_NOT_ON_MEMBER",
                "Observable must measure a capsule member group.",
                file=file,
                json_path=f"{base}.rigid_group_id",
                expected=sorted(members),
                actual=group_id,
            )
        _same_frame_group(
            ctx,
            file=file,
            base=base,
            frame_id=observable.get("frame_id"),
            group_id=group_id,
        )
        ctx.require_string(
            observable, "quantity", file=file, base_path=base
        )


def _validate_expected_behavior(
    ctx: ValidationContext,
    capsule: dict[str, Any],
    *,
    file: str,
) -> None:
    behavior = capsule.get("expected_behavior")
    if not isinstance(behavior, dict):
        ctx.error(
            "EXPECTED_BEHAVIOR_MISSING",
            "Capsule must describe expected behavior.",
            file=file,
            json_path="$.expected_behavior",
            expected="object",
            actual=behavior,
        )
        return
    ctx.warn_unknown_fields(
        behavior,
        {"description", "dof", "failure_conditions"},
        file=file,
        json_path="$.expected_behavior",
    )
    ctx.require_string(
        behavior,
        "description",
        file=file,
        base_path="$.expected_behavior",
    )
    dof = behavior.get("dof")
    if not isinstance(dof, list) or not dof:
        ctx.warning(
            "EXPECTED_DOF_MISSING",
            "Expected behavior does not name intended degrees of freedom.",
            file=file,
            json_path="$.expected_behavior.dof",
            hint=(
                "Describe translational and rotational freedoms without "
                "claiming that they were verified."
            ),
        )
    else:
        for index, entry in enumerate(dof):
            if not isinstance(entry, dict):
                ctx.error(
                    "EXPECTED_DOF_ENTRY_INVALID",
                    "Each expected degree of freedom must be an object.",
                    file=file,
                    json_path=f"$.expected_behavior.dof[{index}]",
                    actual=entry,
                )
                continue
            ctx.warn_unknown_fields(
                entry,
                {"motion"},
                file=file,
                json_path=f"$.expected_behavior.dof[{index}]",
            )
            ctx.require_string(
                entry,
                "motion",
                file=file,
                base_path=f"$.expected_behavior.dof[{index}]",
            )
    failures = behavior.get("failure_conditions")
    if not isinstance(failures, list) or not failures:
        ctx.warning(
            "FAILURE_CONDITIONS_MISSING",
            "Capsule does not declare observable failure conditions.",
            file=file,
            json_path="$.expected_behavior.failure_conditions",
            hint=(
                "List jams, detachments, excessive leakage, wrong-axis "
                "motion, or other externally observable failures."
            ),
        )
    else:
        for index, failure in enumerate(failures):
            if not isinstance(failure, str) or not failure.strip():
                ctx.error(
                    "FAILURE_CONDITION_INVALID",
                    "Each failure condition must be a non-empty string.",
                    file=file,
                    json_path=(
                        f"$.expected_behavior.failure_conditions[{index}]"
                    ),
                    actual=failure,
                )


def _validate_reduced_model(
    ctx: ValidationContext,
    capsule: dict[str, Any],
    *,
    file: str,
    capsule_id: str,
    members: set[str],
    boundaries: set[str],
) -> None:
    claim = capsule.get("reduced_model_claim")
    if claim is None:
        ctx.warning(
            "REDUCED_MODEL_CLAIM_MISSING",
            "No reduced joint or state-machine claim was provided.",
            file=file,
            json_path="$.reduced_model_claim",
            hint=(
                "Add an UNVERIFIED reduced model candidate for later "
                "certification against real contact."
            ),
        )
        return
    if not isinstance(claim, dict):
        ctx.error(
            "REDUCED_MODEL_CLAIM_INVALID",
            "reduced_model_claim must be an object.",
            file=file,
            json_path="$.reduced_model_claim",
            actual=claim,
        )
        return
    ctx.warn_unknown_fields(
        claim,
        {
            "verification_status",
            "family",
            "fixed_group_ids",
            "moving_group_ids",
            "joint_frame_id",
            "axis_abs",
            "pitch_m_per_rad",
            "family_spec_file",
        },
        file=file,
        json_path="$.reduced_model_claim",
    )
    errors_before = ctx.report.total_error_count
    if "verification_status" not in claim:
        ctx.error(
            "REDUCED_MODEL_VERIFICATION_STATUS_REQUIRED",
            "A reduced model claim must explicitly remain UNVERIFIED.",
            file=file,
            json_path="$.reduced_model_claim.verification_status",
            expected="UNVERIFIED",
            actual="missing",
            hint=(
                "Set verification_status to UNVERIFIED; only the validator "
                "may certify the model later."
            ),
        )
    family = claim.get("family")
    if family not in REDUCED_MODEL_FAMILIES:
        ctx.error(
            "REDUCED_MODEL_FAMILY_INVALID",
            "Reduced model family is not supported.",
            file=file,
            json_path="$.reduced_model_claim.family",
            expected=sorted(REDUCED_MODEL_FAMILIES),
            actual=family,
        )
    family_spec_file = claim.get("family_spec_file")
    if family == "compound":
        if not isinstance(family_spec_file, str) or not family_spec_file:
            ctx.error(
                "COMPOUND_FAMILY_SPEC_REQUIRED",
                "A compound reduced model must reference a family_spec_file.",
                file=file,
                json_path="$.reduced_model_claim.family_spec_file",
                actual=family_spec_file,
            )
    elif family_spec_file is not None and family not in {"revolute", "prismatic"}:
        ctx.warning(
            "REDUCED_MODEL_UNUSED_FAMILY_SPEC",
            "family_spec_file is not consumed by this mechanism plugin.",
            file=file,
            json_path="$.reduced_model_claim.family_spec_file",
            actual=family_spec_file,
        )
    if family_spec_file is not None:
        family_spec_path = ctx.safe_package_path(
            family_spec_file,
            manifest_field="reduced_model_claim.family_spec_file",
            diagnostic_file=file,
            diagnostic_json_path="$.reduced_model_claim.family_spec_file",
        )
        ctx.require_file(
            family_spec_path,
            code="REDUCED_MODEL_FAMILY_SPEC_MISSING",
            label="reduced-model family specification",
        )
    fixed = _string_items(
        ctx,
        claim.get("fixed_group_ids"),
        file=file,
        json_path="$.reduced_model_claim.fixed_group_ids",
    )
    moving = _string_items(
        ctx,
        claim.get("moving_group_ids"),
        file=file,
        json_path="$.reduced_model_claim.moving_group_ids",
    )
    for path, values, allowed in (
        ("fixed_group_ids", fixed, boundaries | members),
        ("moving_group_ids", moving, members),
    ):
        for index, group_id in enumerate(values or []):
            if group_id not in allowed:
                ctx.error(
                    "REDUCED_MODEL_GROUP_INVALID",
                    "Reduced model references a group outside its capsule role.",
                    file=file,
                    json_path=f"$.reduced_model_claim.{path}[{index}]",
                    expected=sorted(allowed),
                    actual=group_id,
                )
    if fixed is not None and moving is not None and set(fixed) & set(moving):
        ctx.error(
            "REDUCED_MODEL_GROUP_OVERLAP",
            "Reduced model fixed and moving groups must be disjoint.",
            file=file,
            json_path="$.reduced_model_claim",
            actual=sorted(set(fixed) & set(moving)),
        )
    frame_id = claim.get("joint_frame_id")
    if frame_id not in ctx.frame_ids:
        ctx.error(
            "REDUCED_MODEL_FRAME_UNKNOWN",
            "Reduced model references an unknown joint frame.",
            file=file,
            json_path="$.reduced_model_claim.joint_frame_id",
            actual=frame_id,
        )
    axis = claim.get("axis_abs")
    if family in AXIS_MODEL_FAMILIES or axis is not None:
        ctx.vector(
            axis,
            length=3,
            file=file,
            json_path="$.reduced_model_claim.axis_abs",
            nonzero=True,
            normalized=True,
            normalization_code="AXIS_NOT_UNIT",
        )
    pitch = claim.get("pitch_m_per_rad")
    if family == "helical":
        if (
            isinstance(pitch, bool)
            or not isinstance(pitch, (int, float))
            or not math.isfinite(float(pitch))
            or pitch == 0
        ):
            ctx.error(
                "HELICAL_PITCH_INVALID",
                "A helical reduced model needs a finite non-zero pitch.",
                file=file,
                json_path="$.reduced_model_claim.pitch_m_per_rad",
                actual=pitch,
            )
    elif pitch is not None:
        ctx.warning(
            "REDUCED_MODEL_UNUSED_PITCH",
            "pitch_m_per_rad is ignored for a non-helical family.",
            file=file,
            json_path="$.reduced_model_claim.pitch_m_per_rad",
            actual=pitch,
        )
    if ctx.report.total_error_count == errors_before:
        ctx.reduced_ready_capsules.add(capsule_id)


def _validate_one_capsule(
    ctx: ValidationContext,
    capsule: dict[str, Any],
    *,
    path: Path,
) -> None:
    file = ctx.relative_label(path) or str(path)
    ctx.check_schema_version(capsule, file)
    ctx.scan_for_forbidden_authority(capsule, file=file)
    ctx.warn_unknown_fields(
        capsule,
        {
            "schema_version",
            "capsule_id",
            "name",
            "member_rigid_group_ids",
            "boundary_rigid_group_ids",
            "input_ports",
            "observables",
            "expected_behavior",
            "reduced_model_claim",
            "dependencies",
            "declared_nominal_tests",
        },
        file=file,
    )
    capsule_id = ctx.require_id(capsule, "capsule_id", file=file)
    ctx.require_string(capsule, "name", file=file)
    if not capsule_id:
        return
    if capsule_id in ctx.capsules:
        ctx.error(
            "CAPSULE_ID_DUPLICATE",
            "capsule_id must be unique across the package.",
            file=file,
            json_path="$.capsule_id",
            actual=capsule_id,
        )
        return
    ctx.capsules[capsule_id] = capsule
    ctx.capsule_files[capsule_id] = path

    member_values = _string_items(
        ctx,
        capsule.get("member_rigid_group_ids"),
        file=file,
        json_path="$.member_rigid_group_ids",
    )
    boundary_values = _string_items(
        ctx,
        capsule.get("boundary_rigid_group_ids"),
        file=file,
        json_path="$.boundary_rigid_group_ids",
        nonempty=False,
    )
    members = set(member_values or [])
    boundaries = set(boundary_values or [])
    overlap = members & boundaries
    if overlap:
        ctx.error(
            "CAPSULE_MEMBER_BOUNDARY_OVERLAP",
            "A rigid group cannot be both capsule member and boundary.",
            file=file,
            json_path="$",
            actual=sorted(overlap),
        )
    for key, values in (
        ("member_rigid_group_ids", member_values),
        ("boundary_rigid_group_ids", boundary_values),
    ):
        for index, group_id in enumerate(values or []):
            if group_id not in ctx.group_ids:
                ctx.error(
                    "CAPSULE_GROUP_UNKNOWN",
                    "Capsule references an unknown rigid group.",
                    file=file,
                    json_path=f"$.{key}[{index}]",
                    actual=group_id,
                )

    _validate_ports(
        ctx,
        capsule,
        file=file,
        capsule_id=capsule_id,
        members=members,
    )
    _validate_observables(
        ctx,
        capsule,
        file=file,
        capsule_id=capsule_id,
        members=members,
    )
    _validate_expected_behavior(ctx, capsule, file=file)
    _validate_reduced_model(
        ctx,
        capsule,
        file=file,
        capsule_id=capsule_id,
        members=members,
        boundaries=boundaries,
    )
    dependencies = capsule.get("dependencies")
    if dependencies is not None:
        _string_items(
            ctx,
            dependencies,
            file=file,
            json_path="$.dependencies",
            nonempty=False,
        )
    tests = capsule.get("declared_nominal_tests")
    if not isinstance(tests, list) or not tests:
        ctx.warning(
            "DECLARED_NOMINAL_TESTS_MISSING",
            "Capsule has no proposed nominal excitation cases.",
            file=file,
            json_path="$.declared_nominal_tests",
            hint=(
                "Add candidate input schedules and expected observables; "
                "the validator will still own acceptance thresholds."
            ),
        )
    else:
        known_ports = ctx.capsule_port_ids.get(capsule_id, set())
        for index, test in enumerate(tests):
            base = f"$.declared_nominal_tests[{index}]"
            if not isinstance(test, dict):
                ctx.error(
                    "NOMINAL_TEST_ENTRY_INVALID",
                    "Each declared nominal test must be an object.",
                    file=file,
                    json_path=base,
                    expected="object",
                    actual=test,
                )
                continue
            ctx.warn_unknown_fields(
                test,
                {"input_port_id", "schedule"},
                file=file,
                json_path=base,
            )
            input_port_id = test.get("input_port_id")
            if input_port_id not in known_ports:
                ctx.error(
                    "NOMINAL_TEST_INPUT_UNKNOWN",
                    "Nominal test references an unknown input port.",
                    file=file,
                    json_path=f"{base}.input_port_id",
                    expected=sorted(known_ports),
                    actual=input_port_id,
                )
            ctx.require_string(
                test,
                "schedule",
                file=file,
                base_path=base,
                code="NOMINAL_TEST_SCHEDULE_REQUIRED",
            )


def _validate_dependencies(ctx: ValidationContext) -> None:
    graph: dict[str, set[str]] = {}
    for capsule_id, capsule in ctx.capsules.items():
        file = ctx.relative_label(ctx.capsule_files[capsule_id])
        dependencies = capsule.get("dependencies")
        valid: set[str] = set()
        if isinstance(dependencies, list):
            for index, dependency in enumerate(dependencies):
                if not isinstance(dependency, str):
                    continue
                if dependency not in ctx.capsules:
                    ctx.error(
                        "CAPSULE_DEPENDENCY_UNKNOWN",
                        "Capsule depends on an unknown capsule.",
                        file=file,
                        json_path=f"$.dependencies[{index}]",
                        actual=dependency,
                    )
                else:
                    valid.add(dependency)
        graph[capsule_id] = valid

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        if node in visited:
            return
        if node in visiting:
            start = trail.index(node) if node in trail else 0
            cycle = trail[start:] + [node]
            ctx.error(
                "CAPSULE_DEPENDENCY_CYCLE",
                "Capsule dependencies contain a cycle.",
                file=ctx.relative_label(ctx.capsule_files[node]),
                json_path="$.dependencies",
                actual=cycle,
                hint="Dependencies must form a directed acyclic graph.",
            )
            return
        visiting.add(node)
        trail.append(node)
        for dependency in sorted(graph.get(node, set())):
            visit(dependency, trail)
        trail.pop()
        visiting.remove(node)
        visited.add(node)

    for capsule_id in sorted(graph):
        visit(capsule_id, [])


def validate_capsules(ctx: ValidationContext) -> None:
    paths = ctx.paths.get("capsules")
    if not isinstance(paths, list):
        return
    for path in paths:
        if not isinstance(path, Path):
            continue
        document = ctx.load_json(
            path,
            required=True,
            label="mechanism capsule",
            missing_code="MECHANISM_CAPSULE_MISSING",
        )
        if document is not None:
            _validate_one_capsule(ctx, document, path=path)
    if not ctx.capsules:
        ctx.error(
            "MECHANISM_CAPSULES_EMPTY",
            "No valid mechanism capsule could be loaded.",
            expected="at least one capsule",
            actual=0,
        )
    _validate_dependencies(ctx)
    ctx.report.package_summary["capsule_count"] = len(ctx.capsules)
