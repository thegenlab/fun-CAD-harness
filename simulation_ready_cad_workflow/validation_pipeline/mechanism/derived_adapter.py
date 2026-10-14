from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from .adapters import (
    DERIVED_PLAIN_PRISMATIC_ADAPTER_ID,
    DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
    AdapterSpec,
    MeshAssetSpec,
    TwoBodyGeometrySpec,
    WholeMachineAngularTaskSpec,
    WholeMachineTaskSpec,
)
from .io import ensure_inside, read_json, relative_to_root, sha256_file, write_json_atomic
from .models import MechanismPlan, MechanismValidationError
from .rigid_attachment import materialize_rigid_attachment_proof


_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_DISTANCE_M = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*m\b", re.IGNORECASE)
_ANGLE_DEG = re.compile(
    r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(?:deg|degree|degrees)\b",
    re.IGNORECASE,
)
_ANGLE_RAD = re.compile(
    r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)\s*(?:rad|radian|radians)\b",
    re.IGNORECASE,
)
_MAX_DERIVED_COLLIDER_FACES = 5000
_MAX_INTERFERENCE_CANDIDATE_PAIRS = 4096
_MIN_INTERFERENCE_VOLUME_MM3 = 1e-6
_MAX_REPORTED_INTERFERENCES = 16
_MAX_CONNECTIVITY_CANDIDATE_PAIRS = 4096
_RIGID_GROUP_CONNECTIVITY_TOLERANCE_MM = 0.01


def _fail(
    code: str,
    message: str,
    *,
    details: dict[str, Any] | None = None,
    owner: FailureOwner | None = None,
) -> MechanismValidationError:
    if owner is None:
        return MechanismValidationError(code, message, details=details)
    return MechanismValidationError(
        code,
        message,
        details=details,
        failure=make_failure_envelope(
            code,
            owner,
            source_component="mechanism.derived_adapter",
            message=message,
            evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            details=details,
        ),
    )


def _solid_indices(contract: dict[str, Any], group_ids: set[str]) -> tuple[int, ...]:
    indices: set[int] = set()
    for part in contract.get("parts", []):
        if not isinstance(part, dict) or part.get("rigid_group_id") not in group_ids:
            continue
        for solid_id in part.get("source_solid_ids", []):
            match = re.fullmatch(r"solid_(\d+)", str(solid_id))
            if match is None:
                raise _fail(
                    "DERIVED_SOLID_ID_INVALID",
                    f"cannot derive collider from solid ID {solid_id!r}",
                )
            indices.add(int(match.group(1)))
    if not indices:
        raise _fail(
            "DERIVED_SOLID_GROUP_EMPTY",
            "rigid group has no canonical STEP solid bindings",
        )
    return tuple(sorted(indices))


def _initial_interpenetrations(
    solids: list[Any],
    fixed_indices: tuple[int, ...],
    moving_indices: tuple[int, ...],
) -> tuple[dict[str, Any], ...]:
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    selected_indices = set(fixed_indices) | set(moving_indices)
    for index in selected_indices:
        if index < 1 or index > len(solids):
            raise _fail(
                "DERIVED_SOLID_ID_OUT_OF_RANGE",
                f"solid_{index} is outside the imported STEP inventory",
            )

    bounds: dict[int, tuple[float, ...]] = {}
    for index in selected_indices:
        box = Bnd_Box()
        BRepBndLib.Add_s(solids[index - 1], box)
        if box.IsVoid():
            raise _fail(
                "DERIVED_INTERFERENCE_BOUNDS_FAILED",
                f"cannot bound solid_{index} for the home-pose interference check",
            )
        bounds[index] = tuple(float(value) for value in box.Get())

    def boxes_overlap(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
        return all(
            left[axis + 3] >= right[axis]
            and right[axis + 3] >= left[axis]
            for axis in range(3)
        )

    candidate_count = 0
    intersections: list[dict[str, Any]] = []
    for fixed_index in fixed_indices:
        for moving_index in moving_indices:
            if not boxes_overlap(bounds[fixed_index], bounds[moving_index]):
                continue
            candidate_count += 1
            if candidate_count > _MAX_INTERFERENCE_CANDIDATE_PAIRS:
                raise _fail(
                    "DERIVED_INTERFERENCE_CHECK_TOO_LARGE",
                    "home-pose interference check exceeds the bounded pair budget",
                )
            common = BRepAlgoAPI_Common(
                solids[fixed_index - 1], solids[moving_index - 1]
            )
            common.Build()
            if not common.IsDone():
                raise _fail(
                    "DERIVED_INTERFERENCE_CHECK_FAILED",
                    f"OCCT boolean common failed for solid_{fixed_index} and "
                    f"solid_{moving_index}",
                )
            properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(common.Shape(), properties)
            volume_mm3 = abs(float(properties.Mass()))
            if volume_mm3 <= _MIN_INTERFERENCE_VOLUME_MM3:
                continue
            if len(intersections) < _MAX_REPORTED_INTERFERENCES:
                intersections.append(
                    {
                        "fixed_solid_id": f"solid_{fixed_index}",
                        "moving_solid_id": f"solid_{moving_index}",
                        "overlap_volume_mm3": volume_mm3,
                    }
                )
    return tuple(intersections)


def audit_step_group_interpenetration(
    assembly_step: str | Path,
    *,
    fixed_solid_ids: Sequence[str],
    moving_solid_ids: Sequence[str],
    moving_collision_components: dict[str, Sequence[str | Path]] | None = None,
) -> dict[str, Any]:
    """Run the exact bounded home-pose BRep overlap audit before physics."""

    assembly = Path(assembly_step).resolve()
    if assembly.is_symlink() or not assembly.is_file():
        raise _fail(
            "DERIVED_STEP_IMPORT_FAILED",
            "home-pose audit assembly STEP is unavailable",
        )

    def indices(values: Sequence[str], label: str) -> tuple[int, ...]:
        result = []
        for value in values:
            match = re.fullmatch(r"solid_(\d+)", str(value))
            if match is None:
                raise _fail(
                    "DERIVED_SOLID_ID_INVALID",
                    f"{label} contains invalid solid ID {value!r}",
                )
            result.append(int(match.group(1)))
        unique = tuple(sorted(set(result)))
        if not unique:
            raise _fail(
                "DERIVED_SOLID_GROUP_EMPTY",
                f"{label} must contain at least one STEP solid",
            )
        return unique

    fixed = indices(fixed_solid_ids, "fixed_solid_ids")
    moving = indices(moving_solid_ids, "moving_solid_ids")
    if set(fixed) & set(moving):
        raise _fail(
            "DERIVED_SOLID_GROUP_OVERLAP",
            "fixed and moving groups share canonical STEP solids",
        )

    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    if (
        reader.ReadFile(str(assembly)) != IFSelect_RetDone
        or reader.TransferRoots() <= 0
    ):
        raise _fail(
            "DERIVED_STEP_IMPORT_FAILED",
            "OCCT could not import the home-pose audit assembly STEP",
        )
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    solids: list[Any] = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    proxy_shapes: dict[int, list[tuple[str, Any]]] = {}
    for source_solid_id, component_paths in (
        moving_collision_components or {}
    ).items():
        match = re.fullmatch(r"solid_(\d+)", str(source_solid_id))
        if match is None or int(match.group(1)) not in moving:
            raise _fail(
                "DERIVED_SOLID_ID_INVALID",
                "moving collision proxy source must belong to the moving "
                "solid inventory",
            )
        imported: list[tuple[str, Any]] = []
        for component_path in component_paths:
            path = Path(component_path).resolve()
            component_reader = STEPControl_Reader()
            if (
                path.is_symlink()
                or not path.is_file()
                or component_reader.ReadFile(str(path)) != IFSelect_RetDone
                or component_reader.TransferRoots() <= 0
            ):
                raise _fail(
                    "DERIVED_STEP_IMPORT_FAILED",
                    f"could not import collision component {path.name}",
                )
            component_explorer = TopExp_Explorer(
                component_reader.OneShape(), TopAbs_SOLID
            )
            component_solids: list[Any] = []
            while component_explorer.More():
                component_solids.append(component_explorer.Current())
                component_explorer.Next()
            if len(component_solids) != 1:
                raise _fail(
                    "DERIVED_STEP_IMPORT_FAILED",
                    "each collision component must contain exactly one solid",
                )
            imported.append((path.name, component_solids[0]))
        if not imported:
            raise _fail(
                "DERIVED_SOLID_GROUP_EMPTY",
                f"{source_solid_id} has no collision proxy components",
            )
        proxy_shapes[int(match.group(1))] = imported

    if not proxy_shapes:
        intersections = _initial_interpenetrations(solids, fixed, moving)
    else:
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps

        measured: list[dict[str, Any]] = []
        candidate_count = 0
        for fixed_index in fixed:
            fixed_shape = solids[fixed_index - 1]
            for moving_index in moving:
                candidates = proxy_shapes.get(
                    moving_index,
                    [(f"solid_{moving_index}", solids[moving_index - 1])],
                )
                for component_id, moving_shape in candidates:
                    candidate_count += 1
                    if candidate_count > _MAX_INTERFERENCE_CANDIDATE_PAIRS:
                        raise _fail(
                            "DERIVED_INTERFERENCE_CHECK_TOO_LARGE",
                            "home-pose proxy interference check exceeds the "
                            "bounded pair budget",
                        )
                    common = BRepAlgoAPI_Common(fixed_shape, moving_shape)
                    common.Build()
                    if not common.IsDone():
                        raise _fail(
                            "DERIVED_INTERFERENCE_CHECK_FAILED",
                            "OCCT boolean common failed for a draft collision "
                            "proxy",
                        )
                    properties = GProp_GProps()
                    BRepGProp.VolumeProperties_s(
                        common.Shape(), properties
                    )
                    volume_mm3 = abs(float(properties.Mass()))
                    if volume_mm3 <= _MIN_INTERFERENCE_VOLUME_MM3:
                        continue
                    if len(measured) < _MAX_REPORTED_INTERFERENCES:
                        measured.append(
                            {
                                "fixed_solid_id": f"solid_{fixed_index}",
                                "moving_solid_id": f"solid_{moving_index}",
                                "moving_component_id": component_id,
                                "overlap_volume_mm3": volume_mm3,
                            }
                        )
        intersections = tuple(measured)
    return {
        "schema_version": "simulation_ready_cad_draft_collision_preflight_v1",
        "status": "PASS" if not intersections else "FAIL",
        "passed": not intersections,
        "fixed_solid_ids": [f"solid_{index}" for index in fixed],
        "moving_solid_ids": [f"solid_{index}" for index in moving],
        "intersection_count": len(intersections),
        "intersections": [dict(item) for item in intersections],
        "proxy_source_count": len(proxy_shapes),
        "method": "occt_brep_boolean_common_volume",
    }


def _disconnected_rigid_groups(
    solids: list[Any],
    group_indices: dict[str, tuple[int, ...]],
) -> tuple[dict[str, Any], ...]:
    """Return rigid groups whose STEP solids lack a physical contact path."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape

    selected_indices = {
        index for indices in group_indices.values() for index in indices
    }
    bounds: dict[int, tuple[float, ...]] = {}
    for index in selected_indices:
        if index < 1 or index > len(solids):
            raise _fail(
                "DERIVED_SOLID_ID_OUT_OF_RANGE",
                f"solid_{index} is outside the imported STEP inventory",
            )
        box = Bnd_Box()
        BRepBndLib.Add_s(solids[index - 1], box)
        if box.IsVoid():
            raise _fail(
                "DERIVED_CONNECTIVITY_BOUNDS_FAILED",
                f"cannot bound solid_{index} for the rigid-group connectivity check",
            )
        bounds[index] = tuple(float(value) for value in box.Get())

    def boxes_within_tolerance(
        left: tuple[float, ...], right: tuple[float, ...]
    ) -> bool:
        tolerance = _RIGID_GROUP_CONNECTIVITY_TOLERANCE_MM
        return all(
            left[axis + 3] + tolerance >= right[axis]
            and right[axis + 3] + tolerance >= left[axis]
            for axis in range(3)
        )

    disconnected: list[dict[str, Any]] = []
    candidate_count = 0
    for group_id, indices in sorted(group_indices.items()):
        if len(indices) <= 1:
            continue
        adjacency = {index: set() for index in indices}
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                if not boxes_within_tolerance(
                    bounds[left_index], bounds[right_index]
                ):
                    continue
                candidate_count += 1
                if candidate_count > _MAX_CONNECTIVITY_CANDIDATE_PAIRS:
                    raise _fail(
                        "DERIVED_CONNECTIVITY_CHECK_TOO_LARGE",
                        "rigid-group connectivity check exceeds the bounded pair budget",
                    )
                distance = BRepExtrema_DistShapeShape(
                    solids[left_index - 1], solids[right_index - 1]
                )
                distance.Perform()
                if not distance.IsDone():
                    raise _fail(
                        "DERIVED_CONNECTIVITY_CHECK_FAILED",
                        f"OCCT distance check failed for solid_{left_index} and "
                        f"solid_{right_index}",
                    )
                if float(distance.Value()) <= _RIGID_GROUP_CONNECTIVITY_TOLERANCE_MM:
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)

        components: list[list[int]] = []
        remaining = set(indices)
        while remaining:
            component: list[int] = []
            pending = [min(remaining)]
            while pending:
                index = pending.pop()
                if index not in remaining:
                    continue
                remaining.remove(index)
                component.append(index)
                pending.extend(adjacency[index] & remaining)
            components.append(sorted(component))
        if len(components) > 1:
            disconnected.append(
                {
                    "rigid_group_id": group_id,
                    "components": [
                        [f"solid_{index}" for index in component]
                        for component in components
                    ],
                }
            )
    return tuple(disconnected)


def _task_text(task: dict[str, Any]) -> str:
    pieces = [str(task.get("task_id", "")), str(task.get("description", ""))]
    for criterion in task.get("success_criteria", []):
        if isinstance(criterion, str):
            pieces.append(criterion)
        elif isinstance(criterion, dict):
            pieces.append(" ".join(str(value) for value in criterion.values()))
    return " ".join(pieces).casefold()


def _task_spec(plan: MechanismPlan, requirements: dict[str, Any]) -> WholeMachineTaskSpec:
    tasks = [item for item in requirements.get("tasks", []) if isinstance(item, dict)]
    mapped = [item for item in tasks if item.get("task_id") in set(plan.task_ids)]
    schema_02_targets: list[tuple[dict[str, Any], float]] = []
    if requirements.get("schema_version") == "0.2":
        for task in mapped:
            for requirement in task.get("acceptance_requirements", []):
                if not isinstance(requirement, dict):
                    continue
                target = requirement.get("target")
                if (
                    requirement.get("metric_id") != "signed_displacement_m"
                    or not isinstance(target, dict)
                    or target.get("unit") != "m"
                    or target.get("comparator")
                    not in {"at_least", "equals", "within"}
                ):
                    continue
                value = target.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    schema_02_targets.append((task, float(value)))
    if schema_02_targets:
        outbound, travel_m = max(schema_02_targets, key=lambda item: item[1])
        travel_limit = min(plan.characteristic_length_m * 0.5, 0.25)
        if not 0.001 <= travel_m <= travel_limit:
            raise _fail(
                "DERIVED_TASK_TRAVEL_UNAVAILABLE",
                "schema-0.2 prismatic target is outside the supported travel range",
            )
        task_id = str(outbound["task_id"])
        return WholeMachineTaskSpec(
            outbound_task_id=task_id,
            # The whole-machine envelope returns to home for validation, but
            # a positive-only schema-0.2 task does not acquire a synthetic
            # public return criterion.
            return_task_id=task_id,
            outbound_progress_m=travel_m,
            home_progress_m=0.0,
            moving_mass_kg=0.20,
            outbound_force_n=0.45,
            return_force_n=0.90,
            max_force_phase_s=1.0,
            settle_phase_s=0.15,
        )
    if len(mapped) < 2:
        raise _fail(
            "DERIVED_TASK_PAIR_REQUIRED",
            "plain prismatic validation requires outbound and return machine tasks",
        )

    def primary_text(task: dict[str, Any]) -> str:
        return (
            f"{task.get('task_id', '')} {task.get('description', '')}"
        ).casefold()

    outbound = next(
        (
            item
            for item in mapped
            if "+x" in _task_text(item)
            or any(
                token in primary_text(item)
                for token in ("open", "outbound", "pull")
            )
        ),
        None,
    )
    returned = next(
        (
            item
            for item in mapped
            if "-x" in _task_text(item)
            or any(
                token in primary_text(item)
                for token in ("close", "return", "reverse")
            )
        ),
        None,
    )
    if outbound is None or returned is None or outbound is returned:
        raise _fail(
            "DERIVED_TASK_DIRECTION_AMBIGUOUS",
            "cannot identify positive-axis and return tasks from external requirements",
        )

    candidates: list[float] = []
    for task in mapped:
        for criterion in task.get("success_criteria", []):
            if isinstance(criterion, dict):
                for key in ("target_m", "minimum_m", "travel_m", "value_m"):
                    value = criterion.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        candidates.append(float(value))
            elif isinstance(criterion, str):
                candidates.extend(float(match) for match in _DISTANCE_M.findall(criterion))
    travel_limit = min(plan.characteristic_length_m * 0.5, 0.25)
    candidates = [value for value in candidates if 0.001 <= value <= travel_limit]
    if not candidates:
        raise _fail(
            "DERIVED_TASK_TRAVEL_UNAVAILABLE",
            "external requirements do not provide a usable SI travel target",
        )
    travel_m = max(candidates)
    return WholeMachineTaskSpec(
        outbound_task_id=str(outbound["task_id"]),
        return_task_id=str(returned["task_id"]),
        outbound_progress_m=travel_m,
        home_progress_m=0.0,
        moving_mass_kg=0.20,
        outbound_force_n=0.45,
        return_force_n=0.90,
        max_force_phase_s=1.0,
        settle_phase_s=0.15,
    )


def _angular_task_spec(
    plan: MechanismPlan,
    requirements: dict[str, Any],
) -> WholeMachineAngularTaskSpec:
    tasks = [item for item in requirements.get("tasks", []) if isinstance(item, dict)]
    mapped = [item for item in tasks if item.get("task_id") in set(plan.task_ids)]
    if not mapped:
        raise _fail(
            "DERIVED_ANGULAR_TASK_REQUIRED",
            "plain revolute validation requires a mapped angular machine task",
        )

    schema_02_targets: list[tuple[dict[str, Any], float]] = []
    if requirements.get("schema_version") == "0.2":
        for task in mapped:
            for requirement in task.get("acceptance_requirements", []):
                if not isinstance(requirement, dict):
                    continue
                target = requirement.get("target")
                if (
                    requirement.get("metric_id") != "continuous_rotation_rad"
                    or not isinstance(target, dict)
                    or target.get("unit") != "rad"
                    or target.get("comparator") not in {"at_least", "equals", "within"}
                ):
                    continue
                value = target.get("value")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    schema_02_targets.append((task, abs(float(value))))
    if schema_02_targets:
        outbound, angle_rad = max(schema_02_targets, key=lambda item: item[1])
        if not math.radians(2.0) <= angle_rad <= math.radians(170.0):
            raise _fail(
                "DERIVED_TASK_ANGLE_UNAVAILABLE",
                "schema-0.2 angular target is outside the supported revolute range",
            )
        task_id = str(outbound["task_id"])
        moving_mass_kg = 0.45
        lever_arm_m = max(0.01, 0.5 * max(plan.moving_extents_m))
        gravity_moment_nm = moving_mass_kg * 9.81 * lever_arm_m
        outbound_torque_nm = max(0.05, gravity_moment_nm * 1.20)
        return_torque_nm = max(0.05, gravity_moment_nm * 0.40)
        return WholeMachineAngularTaskSpec(
            outbound_task_id=task_id,
            # The roundtrip load returns the joint for review, but a single
            # schema-0.2 task does not acquire a synthetic return criterion.
            return_task_id=task_id,
            outbound_angle_rad=angle_rad,
            home_angle_rad=0.0,
            moving_mass_kg=moving_mass_kg,
            outbound_torque_nm=outbound_torque_nm,
            return_torque_nm=return_torque_nm,
            max_torque_phase_s=3.0,
            settle_phase_s=0.002,
        )

    if len(mapped) < 2:
        raise _fail(
            "DERIVED_TASK_PAIR_REQUIRED",
            "legacy plain revolute validation requires outbound and return machine tasks",
        )

    def primary_text(task: dict[str, Any]) -> str:
        return f"{task.get('task_id', '')} {task.get('description', '')}".casefold()

    outbound = next(
        (
            item for item in mapped
            if any(token in primary_text(item) for token in (
                "open", "outbound", "raise", "lift", "rotate_positive",
                "grip", "pickup",
            ))
        ),
        None,
    )
    returned = next(
        (
            item for item in mapped
            if any(token in primary_text(item) for token in (
                "close", "return", "lower", "rotate_negative",
                "release",
            ))
        ),
        None,
    )
    if outbound is None or returned is None or outbound is returned:
        raise _fail(
            "DERIVED_TASK_DIRECTION_AMBIGUOUS",
            "cannot identify positive-angle and return tasks from external requirements",
        )

    candidates_rad: list[float] = []
    for task in mapped:
        for criterion in task.get("success_criteria", []):
            if isinstance(criterion, dict):
                for key in (
                    "target_rad", "minimum_rad", "angle_rad", "value_rad"
                ):
                    value = criterion.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        candidates_rad.append(float(value))
                for key in (
                    "target_deg", "minimum_deg", "angle_deg", "value_deg"
                ):
                    value = criterion.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        candidates_rad.append(math.radians(float(value)))
            elif isinstance(criterion, str):
                candidates_rad.extend(
                    math.radians(float(match)) for match in _ANGLE_DEG.findall(criterion)
                )
                candidates_rad.extend(float(match) for match in _ANGLE_RAD.findall(criterion))
    candidates_rad = [
        value for value in candidates_rad
        if math.radians(2.0) <= value <= math.radians(170.0)
    ]
    if not candidates_rad:
        raise _fail(
            "DERIVED_TASK_ANGLE_UNAVAILABLE",
            "external requirements do not provide a usable angular target",
        )
    angle_rad = max(candidates_rad)
    moving_mass_kg = 0.45
    lever_arm_m = max(0.01, 0.5 * max(plan.moving_extents_m))
    gravity_moment_nm = moving_mass_kg * 9.81 * lever_arm_m
    outbound_torque_nm = max(0.05, gravity_moment_nm * 1.20)
    return_torque_nm = max(0.05, gravity_moment_nm * 0.40)
    return WholeMachineAngularTaskSpec(
        outbound_task_id=str(outbound["task_id"]),
        return_task_id=str(returned["task_id"]),
        outbound_angle_rad=angle_rad,
        home_angle_rad=0.0,
        moving_mass_kg=moving_mass_kg,
        outbound_torque_nm=outbound_torque_nm,
        return_torque_nm=return_torque_nm,
        max_torque_phase_s=3.0,
        settle_phase_s=0.002,
    )


def _load_requirements(plan: MechanismPlan) -> dict[str, Any]:
    report = read_json(Path(plan.intake_report))
    requirements_file = report.get("requirements_file")
    if not isinstance(requirements_file, str):
        raise _fail(
            "DERIVED_REQUIREMENTS_PATH_MISSING",
            "intake report does not identify the immutable requirements file",
        )
    path = Path(requirements_file).resolve()
    if not path.is_file() or path.is_symlink():
        raise _fail(
            "DERIVED_REQUIREMENTS_UNAVAILABLE",
            "immutable requirements file is unavailable",
        )
    if sha256_file(path) != plan.requirements_sha256:
        raise _fail(
            "DERIVED_REQUIREMENTS_CHANGED",
            "requirements changed after intake",
        )
    return read_json(path)


def _export_group_mesh(
    solids: list[Any],
    indices: tuple[int, ...],
    *,
    destination: Path,
    workspace: Path,
) -> tuple[MeshAssetSpec, dict[str, Any], tuple[MeshAssetSpec, ...]]:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    import trimesh

    meshes = []
    component_assets: list[MeshAssetSpec] = []
    component_metadata: list[dict[str, Any]] = []
    source_dir = destination.parent / f"{destination.stem}_solid_stl"
    component_dir = destination.parent / f"{destination.stem}.components"
    source_dir.mkdir(parents=True, exist_ok=True)
    component_dir.mkdir(parents=True, exist_ok=True)
    for solid_index in indices:
        if solid_index < 1 or solid_index > len(solids):
            raise _fail(
                "DERIVED_SOLID_ID_OUT_OF_RANGE",
                f"solid_{solid_index} is outside the imported STEP inventory",
            )
        solid = solids[solid_index - 1]
        mesher = BRepMesh_IncrementalMesh(solid, 0.20, False, 0.50, True)
        mesher.Perform()
        if not mesher.IsDone():
            raise _fail(
                "DERIVED_STEP_MESH_FAILED",
                f"OCCT could not mesh solid_{solid_index}",
            )
        stl_path = source_dir / f"solid_{solid_index}.stl"
        StlAPI_Writer().Write(solid, str(stl_path))
        if not stl_path.is_file() or stl_path.stat().st_size == 0:
            raise _fail(
                "DERIVED_STL_EXPORT_FAILED",
                f"OCCT did not export solid_{solid_index}",
            )
        mesh = trimesh.load_mesh(stl_path, process=True)
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            mesh.process(validate=True)
        if not len(mesh.vertices) or not len(mesh.faces):
            raise _fail(
                "DERIVED_STL_EMPTY",
                f"solid_{solid_index} produced an empty triangle mesh",
            )
        mesh.apply_scale(0.001)
        mesh.remove_unreferenced_vertices()
        convex = bool(mesh.is_watertight and mesh.is_convex)
        component_path = component_dir / f"solid_{solid_index}.obj"
        component_text = trimesh.exchange.obj.export_obj(
            mesh,
            include_normals=False,
            include_color=False,
            include_texture=False,
        )
        component_path.write_text(component_text, encoding="ascii", newline="\n")
        component_digest = sha256_file(component_path)
        component_asset = MeshAssetSpec(
            workspace_relative_path=relative_to_root(workspace, component_path),
            sha256=component_digest,
            units_to_m=1.0,
            rotation_abs_from_source=_IDENTITY,
            convex=convex,
        )
        component_assets.append(component_asset)
        component_metadata.append(
            {
                "solid_id": f"solid_{solid_index}",
                "mesh": component_asset.workspace_relative_path,
                "sha256": component_digest,
                "vertex_count": int(len(mesh.vertices)),
                "triangle_count": int(len(mesh.faces)),
                "convex": convex,
            }
        )
        meshes.append(mesh)

    combined = trimesh.util.concatenate(tuple(meshes))
    combined.remove_unreferenced_vertices()
    bounds_abs_m = combined.bounds
    bbox_center_abs_m = 0.5 * (
        bounds_abs_m[0] + bounds_abs_m[1]
    )
    if len(combined.faces) > _MAX_DERIVED_COLLIDER_FACES:
        raise _fail(
            "DERIVED_COLLIDER_TOO_COMPLEX",
            f"derived collider has {len(combined.faces)} faces; limit is "
            f"{_MAX_DERIVED_COLLIDER_FACES}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    obj_text = trimesh.exchange.obj.export_obj(
        combined,
        include_normals=False,
        include_color=False,
        include_texture=False,
    )
    destination.write_text(obj_text, encoding="ascii", newline="\n")
    digest = sha256_file(destination)
    combined_asset = MeshAssetSpec(
        workspace_relative_path=relative_to_root(workspace, destination),
        sha256=digest,
        units_to_m=1.0,
        rotation_abs_from_source=_IDENTITY,
        convex=(len(component_assets) == 1 and component_assets[0].convex),
    )
    return (
        combined_asset,
        {
            "solid_ids": [f"solid_{index}" for index in indices],
            "mesh": relative_to_root(workspace, destination),
            "sha256": digest,
            "vertex_count": int(len(combined.vertices)),
            "triangle_count": int(len(combined.faces)),
            "units": "m",
            "convex_hull_used": False,
            "bounds_abs_m": {
                "min": [float(value) for value in bounds_abs_m[0]],
                "max": [float(value) for value in bounds_abs_m[1]],
            },
            "bbox_center_abs_m": [
                float(value) for value in bbox_center_abs_m
            ],
            "collision_components": component_metadata,
        },
        tuple(component_assets),
    )

def _materialize_plain_two_body_adapter(
    plan: MechanismPlan,
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    expected_adapter_id: str,
    expected_family: str,
    family_spec: dict[str, Any] | None = None,
) -> AdapterSpec:
    if plan.adapter_id != expected_adapter_id or plan.family != expected_family:
        raise _fail(
            "DERIVED_ADAPTER_FAMILY_INVALID",
            f"plain derived adapter requires a {expected_family} mechanism plan",
        )
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(workspace)
    except ValueError as exc:
        raise _fail(
            "DERIVED_ADAPTER_OUTPUT_UNSAFE",
            "derived adapter output must stay inside the workspace",
        ) from exc

    package = Path(plan.package_dir).resolve()
    manifest = read_json(package / "package.json")
    assembly_relative = manifest.get("assembly_file")
    contract_relative = manifest.get("assembly_contract_file")
    if not isinstance(assembly_relative, str) or not isinstance(contract_relative, str):
        raise _fail(
            "DERIVED_PACKAGE_POINTER_MISSING",
            "package does not identify assembly and contract files",
        )
    assembly = ensure_inside(package, assembly_relative)
    contract = read_json(ensure_inside(package, contract_relative))
    expected_assembly_hash = plan.artifact_hashes.get(assembly_relative)
    if expected_assembly_hash is None or sha256_file(assembly) != expected_assembly_hash:
        raise _fail(
            "DERIVED_ASSEMBLY_CHANGED",
            "assembly STEP no longer matches the intake-bound artifact",
        )

    fixed_indices = _solid_indices(contract, set(plan.fixed_group_ids))
    moving_indices = _solid_indices(contract, set(plan.moving_group_ids))
    if len(plan.moving_group_ids) != 1:
        raise _fail(
            "DERIVED_TWO_BODY_MOVING_GROUP_REQUIRED",
            f"plain {expected_family} validation requires exactly one moving rigid group",
        )
    if set(fixed_indices) & set(moving_indices):
        raise _fail(
            "DERIVED_SOLID_GROUP_OVERLAP",
            "fixed and moving groups share canonical STEP solids",
        )

    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(assembly)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise _fail(
            "DERIVED_STEP_IMPORT_FAILED",
            "OCCT could not import the intake-bound assembly STEP",
        )
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    solids: list[Any] = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    if not solids:
        raise _fail("DERIVED_STEP_EMPTY", "assembly STEP contains no solids")
    intersections = _initial_interpenetrations(solids, fixed_indices, moving_indices)
    if intersections:
        details = ", ".join(
            f"{item['fixed_solid_id']}/{item['moving_solid_id']}="
            f"{item['overlap_volume_mm3']:.6g} mm^3"
            for item in intersections
        )
        raise _fail(
            "DERIVED_INITIAL_INTERPENETRATION",
            "fixed and moving rigid groups overlap at the declared home pose: " + details,
        )
    derived_root = output / "derived_adapters" / plan.capsule_id
    rigid_attachment = materialize_rigid_attachment_proof(
        solids,
        {"fixed": fixed_indices, "moving": moving_indices},
        package_binding_sha256=plan.package_binding_sha256,
        workspace_root=workspace,
        destination=derived_root / "rigid_attachment_proof.json",
        body_group_ids={
            "fixed": tuple(plan.fixed_group_ids),
            "moving": tuple(plan.moving_group_ids),
        },
    )
    fixed_mesh, fixed_meta, fixed_components = _export_group_mesh(
        solids,
        fixed_indices,
        destination=derived_root / "fixed.nonconvex.obj",
        workspace=workspace,
    )
    moving_mesh, moving_meta, moving_components = _export_group_mesh(
        solids,
        moving_indices,
        destination=derived_root / "moving.nonconvex.obj",
        workspace=workspace,
    )
    unresolved_dynamic_components = []
    if expected_family == "revolute":
        from .worker import _sphere_spec

        unresolved_dynamic_components = [
            (solid_index, component)
            for solid_index, component in zip(
                moving_indices, moving_components, strict=True
            )
            if not component.convex
            and _sphere_spec(
                workspace / component.workspace_relative_path
            ) is None
        ]
    if unresolved_dynamic_components:
        raise _fail(
            "DERIVED_REVOLUTE_DYNAMIC_DECOMPOSITION_REQUIRED",
            "plain revolute moving solids must form a verified convex decomposition "
            "or an exact supported primitive decomposition",
            details={
                "unsupported_dynamic_components": [
                    {
                        "body_id": "moving",
                        "source_solid_id": f"solid_{solid_index}",
                    }
                    for solid_index, _component in unresolved_dynamic_components
                ],
                "repair_action": "generate_certified_collision_components",
                "repair_tool_module": (
                    "simulation_ready_cad_workflow.validation_pipeline.tools."
                    "compile_design_package"
                ),
                "authoritative_geometry_policy": (
                    "preserve_assembly_step_bytes"
                ),
            },
            owner=FailureOwner.DESIGN_PACKAGE,
        )

    requirements = _load_requirements(plan)
    linear_task = None
    angular_task = None
    certifiable_progress_limit_m = None
    certifiable_angle_limit_rad = None
    if expected_family == "prismatic":
        linear_task = _task_spec(plan, requirements)
        certifiable_progress_limit_m = linear_task.outbound_progress_m
        task_summary = {
            "coordinate": "translation",
            "outbound": linear_task.outbound_progress_m,
            "home": linear_task.home_progress_m,
            "unit": "m",
            "task_ids": [linear_task.outbound_task_id, linear_task.return_task_id],
        }
    else:
        angular_task = _angular_task_spec(plan, requirements)
        if family_spec is not None:
            from .axis_spring_spec import axis_spring_from_dict

            spring = axis_spring_from_dict(
                family_spec,
                expected_family="revolute",
            )
            if angular_task.outbound_angle_rad > spring.maximum_deflection_si:
                raise _fail(
                    "AXIS_SPRING_TASK_OUT_OF_RANGE",
                    "task angle exceeds the hash-bound spring deflection range",
                )
            lever_arm_m = max(0.01, 0.5 * max(plan.moving_extents_m))
            gravity_moment_nm = (
                spring.moving_mass_kg * 9.81 * lever_arm_m
            )
            angular_task = replace(
                angular_task,
                moving_mass_kg=spring.moving_mass_kg,
                outbound_torque_nm=(
                    max(0.05, gravity_moment_nm * 1.20)
                    + 1.15 * spring.stiffness_si * angular_task.outbound_angle_rad
                ),
                return_torque_nm=0.0,
            )
        certifiable_angle_limit_rad = abs(
            angular_task.outbound_angle_rad - angular_task.home_angle_rad
        )
        task_summary = {
            "coordinate": "rotation",
            "outbound": angular_task.outbound_angle_rad,
            "home": angular_task.home_angle_rad,
            "unit": "rad",
            "task_ids": [angular_task.outbound_task_id, angular_task.return_task_id],
        }

    supported_programs = (
        {
            "bidirectional_axis",
            "zero_input_gravity",
            "transverse_probe",
            "task_range_roundtrip",
        }
        if expected_family == "prismatic"
        else {
            "bidirectional_rotation",
            "zero_input_gravity",
            "transverse_torque_probe",
            "task_angle_roundtrip",
        }
    )
    if (
        expected_family == "revolute"
        and isinstance(family_spec, dict)
        and isinstance(family_spec.get("contact_fixture"), dict)
    ):
        fixture_shape = family_spec["contact_fixture"].get("shape")
        if fixture_shape == "dynamic_grip_box":
            supported_programs.update(
                {"tong_grip_object", "tong_hold_object", "tong_release_object"}
            )
        elif fixture_shape == "tabletop_pickup_box":
            supported_programs.update(
                {
                    "tong_tabletop_pickup",
                    "tong_tabletop_hold",
                    "tong_tabletop_release",
                }
            )
    adapter = AdapterSpec(
        adapter_id=expected_adapter_id,
        family=expected_family,
        required_source_hashes=frozenset(),
        supported_programs=frozenset(supported_programs),
        required_requirements_sha256=plan.requirements_sha256,
        whole_machine_task=linear_task,
        whole_machine_angular_task=angular_task,
        family_spec=family_spec,
        geometry=TwoBodyGeometrySpec(
            fixed_mesh=fixed_mesh,
            moving_mesh=moving_mesh,
            fixed_origin_abs_m=tuple(
                float(value) for value in fixed_meta["bbox_center_abs_m"]
            ),
            moving_origin_abs_m=tuple(
                float(value) for value in moving_meta["bbox_center_abs_m"]
            ),
            fixed_collision_meshes=fixed_components,
            moving_collision_meshes=moving_components,
            fixed_mass_kg=2.0,
            friction=0.04 if expected_family == "revolute" else 0.02,
            collision_envelope_m=10e-6,
            collision_margin_m=2e-6,
            certifiable_progress_limit_m=certifiable_progress_limit_m,
            certifiable_angle_limit_rad=certifiable_angle_limit_rad,
            collision_scope=(
                "all STEP solids mapped to declared fixed rigid groups",
                "all STEP solids mapped to declared moving rigid groups",
            ),
            omitted_contact_geometry=(),
        ),
    )
    if (
        expected_family == "prismatic"
        and isinstance(family_spec, Mapping)
        and family_spec.get("mechanism_kind")
        == "spring_scale_calibrated_prismatic"
        and "collision_components/manifest.json" in plan.artifact_hashes
    ):
        from .compound_spec import CompoundBodySpec
        from .family_collision import materialize_family_collision_adapter

        raw_bodies = family_spec.get("bodies")
        if not isinstance(raw_bodies, Sequence) or isinstance(
            raw_bodies, (str, bytes)
        ):
            raise _fail(
                "COLLISION_COMPONENT_BODY_BINDING_REQUIRED",
                "authored prismatic collision components require typed bodies",
            )
        collision_bodies: list[CompoundBodySpec] = []
        for raw_body in raw_bodies:
            if not isinstance(raw_body, Mapping):
                raise _fail(
                    "COLLISION_COMPONENT_BODY_BINDING_REQUIRED",
                    "every authored prismatic collision body must be an object",
                )
            body_id = str(raw_body.get("body_id") or "")
            group_ids = tuple(str(item) for item in raw_body.get("group_ids") or ())
            fixed = raw_body.get("fixed")
            if not body_id or not group_ids or not isinstance(fixed, bool):
                raise _fail(
                    "COLLISION_COMPONENT_BODY_BINDING_REQUIRED",
                    "authored prismatic collision bodies need id, groups, and fixed",
                )
            center = (
                adapter.geometry.fixed_origin_abs_m
                if fixed
                else adapter.geometry.moving_origin_abs_m
            )
            size = (
                (plan.characteristic_length_m,) * 3
                if fixed
                else tuple(plan.moving_extents_m)
            )
            collision_bodies.append(
                CompoundBodySpec(
                    body_id=body_id,
                    group_ids=group_ids,
                    fixed=fixed,
                    mass_kg=2.0 if fixed else 1.0,
                    center_abs_m=tuple(center),
                    size_m=tuple(size),
                )
            )
        return materialize_family_collision_adapter(
            plan,
            adapter,
            tuple(collision_bodies),
            workspace_root=workspace,
            output_dir=output,
        )
    from .collider_fidelity import build_collider_fidelity_report
    from .family_collision import _representation

    import trimesh

    fidelity_components: list[dict[str, Any]] = []
    for role, metadata, fixed in (
        ("fixed", fixed_meta, True),
        ("moving", moving_meta, False),
    ):
        source_root = derived_root / f"{role}.nonconvex_solid_stl"
        for component in metadata["collision_components"]:
            solid_id = str(component["solid_id"])
            source_path = source_root / f"{solid_id}.stl"
            collider_path = ensure_inside(workspace, str(component["mesh"]))
            source_mesh = trimesh.load_mesh(source_path, process=True, force="mesh")
            collider_mesh = trimesh.load_mesh(
                collider_path, process=False, force="mesh"
            )
            if not isinstance(source_mesh, trimesh.Trimesh):
                source_mesh = trimesh.util.concatenate(
                    tuple(source_mesh.geometry.values())
                )
            if not isinstance(collider_mesh, trimesh.Trimesh):
                collider_mesh = trimesh.util.concatenate(
                    tuple(collider_mesh.geometry.values())
                )
            source_mesh.apply_scale(0.001)
            source_mesh.remove_unreferenced_vertices()
            fidelity_components.append(
                {
                    "body_id": role,
                    "solid_id": solid_id,
                    "collider_id": str(component["mesh"]),
                    "representation": _representation(
                        collider_path,
                        fixed=fixed,
                        convex=bool(component.get("convex")),
                    ),
                    "source_mesh": source_mesh,
                    "collider_mesh": collider_mesh,
                }
            )
    fidelity_report = build_collider_fidelity_report(
        fidelity_components,
        package_binding_sha256=plan.package_binding_sha256,
        characteristic_length_m=plan.characteristic_length_m,
        mesh_linear_deflection_m=0.0002,
        collision_envelope_m=adapter.geometry.collision_envelope_m,
    )
    fidelity_path = derived_root / "collider_fidelity.json"
    write_json_atomic(fidelity_path, fidelity_report)
    fidelity_binding = {
        "schema_version": fidelity_report["schema_version"],
        "status": fidelity_report["status"],
        "passed": fidelity_report["passed"],
        "report": relative_to_root(workspace, fidelity_path),
        "report_sha256": sha256_file(fidelity_path),
        "package_binding_sha256": plan.package_binding_sha256,
        "rigid_attachment_proof": rigid_attachment,
    }
    if not fidelity_report["passed"]:
        raise _fail(
            "COLLIDER_FIDELITY_PREFLIGHT_FAILED",
            "STEP-to-collider fidelity evidence did not pass trusted budgets",
        )
    adapter = replace(adapter, collider_fidelity=fidelity_binding)

    write_json_atomic(
        derived_root / "provenance.json",
        {
            "schema_version": "0.1",
            "adapter_id": adapter.adapter_id,
            "mechanism_family": expected_family,
            "package_binding_sha256": plan.package_binding_sha256,
            "assembly_file": assembly_relative,
            "assembly_sha256": expected_assembly_hash,
            "fixed": fixed_meta,
            "moving": moving_meta,
            "task": task_summary,
            "moving_collision_components_all_convex": all(
                component.convex for component in moving_components
            ),
            "convex_hull_used": False,
            "ideal_joint_used": False,
            "rigid_attachment_proof": rigid_attachment,
            "collider_fidelity": adapter.collider_fidelity,
        },
    )
    return adapter


def materialize_plain_prismatic_adapter(
    plan: MechanismPlan,
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    family_spec: dict[str, Any] | None = None,
) -> AdapterSpec:
    return _materialize_plain_two_body_adapter(
        plan,
        workspace_root=workspace_root,
        output_dir=output_dir,
        expected_adapter_id=DERIVED_PLAIN_PRISMATIC_ADAPTER_ID,
        expected_family="prismatic",
        family_spec=family_spec,
    )


def materialize_plain_revolute_adapter(
    plan: MechanismPlan,
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
    family_spec: dict[str, Any] | None = None,
) -> AdapterSpec:
    return _materialize_plain_two_body_adapter(
        plan,
        workspace_root=workspace_root,
        output_dir=output_dir,
        expected_adapter_id=DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
        expected_family="revolute",
        family_spec=family_spec,
    )
