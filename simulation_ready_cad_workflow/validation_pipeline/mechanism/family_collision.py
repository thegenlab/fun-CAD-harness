from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    RepairScope,
    make_failure_envelope,
    make_repair_contract,
)
from .adapters import AdapterSpec, adapter_from_dict, adapter_to_dict
from .assembly_connectivity import build_assembly_connectivity_report
from .io import (
    canonical_sha256,
    ensure_inside,
    read_json,
    relative_to_root,
    sha256_file,
    write_json_atomic,
)
from .finite_linear_capture import (
    build_finite_linear_capture_report,
    finite_linear_capture_from_dict,
)
from .models import MechanismPlan, MechanismValidationError
from .rigid_attachment import build_rigid_attachment_proof

if TYPE_CHECKING:
    from .compound_spec import CompoundBodySpec


FAMILY_COLLISION_SCHEMA = "step_per_solid_body_local_v1"
FAMILY_COLLISION_PREFLIGHT_SNAPSHOT_SCHEMA = (
    "family_collision_release_bound_preflight_snapshot_v1"
)
FAMILY_COLLISION_PREFLIGHT_INPUT_SCHEMA = (
    "family_collision_preflight_input_binding_v1"
)
FAMILY_COLLISION_PREFLIGHT_STATUS = "EVIDENCE_COMPLETE"
_PREFLIGHT_SNAPSHOT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "plugin_id",
        "input_binding",
        "materialized_adapter",
        "materialized_adapter_sha256",
        "materialized_root",
        "materialized_files",
        "materialized_file_count",
        "materialized_total_size_bytes",
        "physical_simulation_started",
        "saved_trace_files",
        "authority",
    }
)
_PREFLIGHT_ADAPTER_BINDING_KEYS = frozenset(
    {"path", "sha256", "size_bytes"}
)
_PREFLIGHT_FILE_BINDING_KEYS = frozenset(
    {"path", "sha256", "size_bytes"}
)
_PREFLIGHT_AUTHORITY = {
    "activation_authority": False,
    "contract_pass": False,
    "physical_pass": False,
    "terminal_pass": False,
}
_PREFLIGHT_ALLOWED_FILE_SUFFIXES = frozenset({".json", ".obj", ".stl"})
_PREFLIGHT_MAX_FILE_COUNT = 512
_PREFLIGHT_MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024
_PREFLIGHT_MAX_TOTAL_SIZE_BYTES = 64 * 1024 * 1024
TRACE_COLLISION_THRESHOLD_FIELDS = (
    "mesh_linear_deflection_m",
    "collision_envelope_m",
    "collision_margin_m",
    "interpenetration_tolerance_m",
    "runtime_interpenetration_tolerance_m",
)
_AUTHORED_COLLISION_COMPONENT_PLUGIN_BY_MECHANISM_KIND = {
    "bicycle_chain_sprocket_drive": "bicycle_chain_sprocket_drive_v2",
    "bicycle_ground_roll": "bicycle_ground_roll_v1",
    "clothespin_dual_jaw_torsion": (
        "reviewer.clothespin_dual_jaw_torsion_v1"
    ),
    "deadbolt_door_lock": "reviewer.deadbolt_door_lock_v1",
    "rotary_detent_phone_stand": "reviewer.rotary_detent_phone_stand_v1",
    "spring_scale_prismatic_spring": (
        "reviewer.spring_scale_prismatic_spring_v1"
    ),
    "spring_scale_calibrated_prismatic": (
        "reviewer.spring_scale_calibrated_prismatic_v1"
    ),
    "foldable_phone_stand_rotary_detent": "revolute_linkage_v1",
}


def _fail(
    code: str,
    message: str,
    *,
    owner: FailureOwner | None = None,
    details: Mapping[str, Any] | None = None,
    repair_scope: RepairScope | None = None,
) -> MechanismValidationError:
    if owner is None:
        return MechanismValidationError(code, message)
    return MechanismValidationError(
        code,
        message,
        details=details,
        failure=make_failure_envelope(
            code,
            owner,
            source_component="mechanism.family_collision",
            message=message,
            evidence_validity=EvidenceValidity.VALID_MEASUREMENT,
            details=details,
            repair=(
                make_repair_contract(
                    repair_scope,
                    preserves=(
                        "assembly.step",
                        "static_visual_review",
                        "approved_plan",
                    ),
                    invalidates=(
                        "collision_materialization",
                        "collider_fidelity",
                        "physical_preflight",
                    ),
                    resume_stage="preflight",
                )
                if repair_scope is not None
                else None
            ),
        ),
    )


def _authored_collision_component_plugin_id(
    adapter: AdapterSpec,
) -> str | None:
    """Return the installed plugin allowed to consume component sidecars."""

    if not isinstance(adapter.family_spec, Mapping):
        return None
    protocol_id = adapter.family_spec.get(
        "collision_component_protocol_id"
    ) or adapter.family_spec.get("mechanism_kind") or adapter.family_spec.get(
        "linkage_id"
    )
    plugin_id = _AUTHORED_COLLISION_COMPONENT_PLUGIN_BY_MECHANISM_KIND.get(
        protocol_id
    )
    if plugin_id is None:
        return None
    from .family_plugins import (
        plugin_registration_for,
        registration_allows_runtime_execution,
    )

    registration = plugin_registration_for(plugin_id)
    return (
        plugin_id
        if registration_allows_runtime_execution(registration)
        else None
    )


def family_collision_preflight_input_binding(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    *,
    plugin_id: str,
) -> dict[str, Any]:
    """Bind reusable static collision evidence to current immutable inputs."""

    if (
        not isinstance(plugin_id, str)
        or not plugin_id
        or adapter.family_spec is None
        or adapter.collider_fidelity is not None
        or "collision_geometry" in adapter.family_spec
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_INPUT_INVALID",
            "preflight input must be one unmaterialized family adapter",
        )
    return {
        "schema_version": FAMILY_COLLISION_PREFLIGHT_INPUT_SCHEMA,
        "plugin_id": plugin_id,
        "package_id": plan.package_id,
        "design_version": plan.design_version,
        "capsule_id": plan.capsule_id,
        "package_binding_sha256": plan.package_binding_sha256,
        "requirements_sha256": plan.requirements_sha256,
        "artifact_hashes_sha256": canonical_sha256(
            plan.artifact_hashes
        ),
        "source_component_hashes_sha256": canonical_sha256(
            plan.source_component_hashes
        ),
        "unmaterialized_adapter_sha256": canonical_sha256(
            adapter_to_dict(adapter)
        ),
    }


def _canonical_preflight_path(
    workspace: Path,
    value: Any,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or Path(value).as_posix() != value:
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
            f"{label} is not a canonical workspace-relative path",
        )
    try:
        path = ensure_inside(workspace, value)
    except (OSError, ValueError) as exc:
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
            f"{label} escapes the workspace",
        ) from exc
    current = workspace / Path(value)
    while current != workspace:
        if current.is_symlink():
            raise _fail(
                "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
                f"{label} traverses a symbolic link",
            )
        current = current.parent
    return path


def _preflight_file_inventory(
    *,
    workspace: Path,
    materialized_root: Path,
) -> tuple[list[dict[str, Any]], int]:
    if (
        not materialized_root.is_dir()
        or materialized_root.is_symlink()
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_FILES_INVALID",
            "materialized preflight root is missing or unsafe",
        )
    rows: list[dict[str, Any]] = []
    total_size = 0
    for path in sorted(
        materialized_root.rglob("*"),
        key=lambda item: item.as_posix(),
    ):
        if path.is_dir() and not path.is_symlink():
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or path.suffix.casefold()
            not in _PREFLIGHT_ALLOWED_FILE_SUFFIXES
            or path.name.endswith(".trace.json")
            or path.name in {"queue_state.json", "run_status.json"}
        ):
            raise _fail(
                "FAMILY_COLLISION_PREFLIGHT_FILES_INVALID",
                f"preflight contains an unsafe file: {path}",
            )
        size = path.stat().st_size
        if size > _PREFLIGHT_MAX_FILE_SIZE_BYTES:
            raise _fail(
                "FAMILY_COLLISION_PREFLIGHT_FILES_INVALID",
                f"preflight file exceeds its bound: {path}",
            )
        total_size += size
        if total_size > _PREFLIGHT_MAX_TOTAL_SIZE_BYTES:
            raise _fail(
                "FAMILY_COLLISION_PREFLIGHT_FILES_INVALID",
                "preflight file inventory exceeds its total byte bound",
            )
        rows.append(
            {
                "path": relative_to_root(workspace, path),
                "sha256": sha256_file(path),
                "size_bytes": size,
            }
        )
        if len(rows) > _PREFLIGHT_MAX_FILE_COUNT:
            raise _fail(
                "FAMILY_COLLISION_PREFLIGHT_FILES_INVALID",
                "preflight file inventory exceeds its count bound",
            )
    if not rows:
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_FILES_INVALID",
            "preflight materialization produced no files",
        )
    return rows, total_size


def _validate_materialized_adapter_for_preflight(
    plan: MechanismPlan,
    unmaterialized: AdapterSpec,
    materialized: AdapterSpec,
    *,
    expected_bodies: Sequence[CompoundBodySpec],
) -> None:
    expected = adapter_to_dict(unmaterialized)
    observed = adapter_to_dict(materialized)
    observed_family = observed.get("family_spec")
    if not isinstance(observed_family, dict):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_ADAPTER_INVALID",
            "materialized adapter has no family specification",
        )
    collision = observed_family.pop("collision_geometry", None)
    observed["collider_fidelity"] = None
    if (
        observed != expected
        or not isinstance(collision, dict)
        or collision.get("package_binding_sha256")
        != plan.package_binding_sha256
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_ADAPTER_INVALID",
            "materialized adapter differs beyond collision evidence",
        )
    geometry = family_collision_geometry(
        materialized,
        expected_bodies=expected_bodies,
    )
    if geometry.get("package_binding_sha256") != plan.package_binding_sha256:
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_ADAPTER_INVALID",
            "materialized collision evidence belongs to another package",
        )


def write_family_collision_preflight_snapshot(
    plan: MechanismPlan,
    unmaterialized_adapter: AdapterSpec,
    materialized_adapter: AdapterSpec,
    *,
    plugin_id: str,
    expected_bodies: Sequence[CompoundBodySpec],
    workspace_root: str | Path,
    materialized_root: str | Path,
    preflight_root: str | Path,
) -> dict[str, Any]:
    """Write a release artifact that carries evidence but no verdict authority."""

    workspace = Path(workspace_root).resolve()
    materialized = Path(materialized_root).resolve()
    preflight = Path(preflight_root).resolve()
    try:
        materialized.relative_to(workspace)
        preflight.relative_to(workspace)
    except ValueError as exc:
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
            "preflight output must stay inside the workspace",
        ) from exc
    if preflight.exists() or preflight.is_symlink():
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
            "preflight destination was preoccupied",
        )
    _validate_materialized_adapter_for_preflight(
        plan,
        unmaterialized_adapter,
        materialized_adapter,
        expected_bodies=expected_bodies,
    )
    files, total_size = _preflight_file_inventory(
        workspace=workspace,
        materialized_root=materialized,
    )
    adapter_path = preflight / "adapter_spec.json"
    manifest_path = preflight / "preflight_manifest.json"
    adapter_payload = adapter_to_dict(materialized_adapter)
    write_json_atomic(adapter_path, adapter_payload)
    adapter_binding = {
        "path": relative_to_root(workspace, adapter_path),
        "sha256": sha256_file(adapter_path),
        "size_bytes": adapter_path.stat().st_size,
    }
    manifest = {
        "schema_version": FAMILY_COLLISION_PREFLIGHT_SNAPSHOT_SCHEMA,
        "status": FAMILY_COLLISION_PREFLIGHT_STATUS,
        "plugin_id": plugin_id,
        "input_binding": family_collision_preflight_input_binding(
            plan,
            unmaterialized_adapter,
            plugin_id=plugin_id,
        ),
        "materialized_adapter": adapter_binding,
        "materialized_adapter_sha256": canonical_sha256(
            adapter_payload
        ),
        "materialized_root": relative_to_root(
            workspace,
            materialized,
        ),
        "materialized_files": files,
        "materialized_file_count": len(files),
        "materialized_total_size_bytes": total_size,
        "physical_simulation_started": False,
        "saved_trace_files": [],
        "authority": dict(_PREFLIGHT_AUTHORITY),
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def load_family_collision_preflight_snapshot(
    plan: MechanismPlan,
    unmaterialized_adapter: AdapterSpec,
    *,
    plugin_id: str,
    expected_bodies: Sequence[CompoundBodySpec],
    workspace_root: str | Path,
    manifest_file: str | Path,
) -> AdapterSpec:
    """Verify and load an immutable release-bound materialized adapter."""

    workspace = Path(workspace_root).resolve()
    requested = Path(manifest_file)
    if requested.is_absolute():
        try:
            manifest_relative = requested.relative_to(workspace).as_posix()
        except ValueError as exc:
            raise _fail(
                "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
                "preflight manifest is outside the workspace",
            ) from exc
    else:
        manifest_relative = requested.as_posix()
    manifest_path = _canonical_preflight_path(
        workspace,
        manifest_relative,
        label="preflight manifest",
    )
    if not manifest_path.is_file():
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_PATH_INVALID",
            "preflight manifest is missing",
        )
    manifest = read_json(manifest_path)
    adapter_binding = manifest.get("materialized_adapter")
    files = manifest.get("materialized_files")
    if (
        set(manifest) != _PREFLIGHT_SNAPSHOT_KEYS
        or manifest.get("schema_version")
        != FAMILY_COLLISION_PREFLIGHT_SNAPSHOT_SCHEMA
        or manifest.get("status")
        != FAMILY_COLLISION_PREFLIGHT_STATUS
        or manifest.get("plugin_id") != plugin_id
        or manifest.get("input_binding")
        != family_collision_preflight_input_binding(
            plan,
            unmaterialized_adapter,
            plugin_id=plugin_id,
        )
        or manifest.get("physical_simulation_started") is not False
        or manifest.get("saved_trace_files") != []
        or manifest.get("authority") != _PREFLIGHT_AUTHORITY
        or not isinstance(
            manifest.get("materialized_adapter_sha256"),
            str,
        )
        or len(manifest["materialized_adapter_sha256"]) != 64
        or not isinstance(adapter_binding, dict)
        or set(adapter_binding) != _PREFLIGHT_ADAPTER_BINDING_KEYS
        or not isinstance(files, list)
        or not files
        or any(
            not isinstance(item, dict)
            or set(item) != _PREFLIGHT_FILE_BINDING_KEYS
            for item in files
        )
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_MANIFEST_INVALID",
            "preflight manifest is open, stale, or claims verdict authority",
        )
    materialized_root = _canonical_preflight_path(
        workspace,
        manifest.get("materialized_root"),
        label="materialized root",
    )
    current_files, total_size = _preflight_file_inventory(
        workspace=workspace,
        materialized_root=materialized_root,
    )
    if (
        current_files != files
        or manifest.get("materialized_file_count") != len(current_files)
        or manifest.get("materialized_total_size_bytes") != total_size
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_FILES_CHANGED",
            "materialized preflight files changed after release binding",
        )
    adapter_path = _canonical_preflight_path(
        workspace,
        adapter_binding.get("path"),
        label="materialized adapter",
    )
    if (
        not adapter_path.is_file()
        or adapter_path.is_symlink()
        or adapter_path.stat().st_size
        != adapter_binding.get("size_bytes")
        or sha256_file(adapter_path) != adapter_binding.get("sha256")
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_ADAPTER_CHANGED",
            "materialized adapter bytes changed after release binding",
        )
    try:
        adapter_payload = read_json(adapter_path)
        materialized_adapter = adapter_from_dict(adapter_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_ADAPTER_INVALID",
            "materialized adapter is malformed",
        ) from exc
    if (
        canonical_sha256(adapter_payload)
        != manifest["materialized_adapter_sha256"]
    ):
        raise _fail(
            "FAMILY_COLLISION_PREFLIGHT_ADAPTER_CHANGED",
            "materialized adapter canonical binding changed",
        )
    _validate_materialized_adapter_for_preflight(
        plan,
        unmaterialized_adapter,
        materialized_adapter,
        expected_bodies=expected_bodies,
    )
    return materialized_adapter


def trace_collision_thresholds(
    collision_manifest: Mapping[str, Any],
) -> dict[str, float]:
    """Carry validator-owned collision budgets into the saved trace."""

    payload: dict[str, float] = {}
    for field in TRACE_COLLISION_THRESHOLD_FIELDS:
        try:
            payload[field] = float(collision_manifest[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise _fail(
                "TRACE_COLLISION_THRESHOLD_MISSING",
                f"collision manifest has no valid {field}",
            ) from exc
    return payload


def _rotation_matrix_from_quaternion(
    quaternion_wxyz: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    w_value, x_value, y_value, z_value = (
        float(value) for value in quaternion_wxyz
    )
    magnitude = math.sqrt(
        w_value * w_value
        + x_value * x_value
        + y_value * y_value
        + z_value * z_value
    )
    if magnitude <= 1e-12:
        raise _fail(
            "FAMILY_COLLIDER_BODY_ROTATION_INVALID",
            "body rotation quaternion cannot be zero",
        )
    w_value, x_value, y_value, z_value = (
        value / magnitude for value in (w_value, x_value, y_value, z_value)
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


def _world_to_body_local(
    point_abs_m: Sequence[float],
    *,
    center_abs_m: Sequence[float],
    rotation_abs_from_body: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    offset = tuple(
        float(point_abs_m[index]) - float(center_abs_m[index])
        for index in range(3)
    )
    # R maps body-local vectors to the absolute frame, so R^T maps back.
    return tuple(
        sum(float(rotation_abs_from_body[row][column]) * offset[row] for row in range(3))
        for column in range(3)
    )


def _body_local_to_world(
    point_body_m: Sequence[float],
    *,
    center_abs_m: Sequence[float],
    rotation_abs_from_body: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    return tuple(
        float(center_abs_m[row])
        + sum(
            float(rotation_abs_from_body[row][column])
            * float(point_body_m[column])
            for column in range(3)
        )
        for row in range(3)
    )


def _load_step_solids(assembly: Path) -> list[Any]:
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer

    reader = STEPControl_Reader()
    if reader.ReadFile(str(assembly)) != IFSelect_RetDone or reader.TransferRoots() <= 0:
        raise _fail(
            "FAMILY_COLLIDER_STEP_IMPORT_FAILED",
            "OCCT could not import the intake-bound assembly STEP",
        )
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)
    solids: list[Any] = []
    while explorer.More():
        solids.append(explorer.Current())
        explorer.Next()
    if not solids:
        raise _fail(
            "FAMILY_COLLIDER_STEP_EMPTY",
            "the intake-bound assembly STEP contains no solids",
        )
    return solids


_MESH_PARAMETERS = {
    "linear_deflection": 0.20,
    "relative": False,
    "angular_deflection": 0.50,
    "in_parallel": True,
    "exporter": "StlAPI_Writer",
}


def _export_solid_mesh(
    solid: Any,
    solid_id: str,
    destination: Path,
    *,
    mesh_cache: Any = None,
) -> Any:
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    import trimesh

    destination.parent.mkdir(parents=True, exist_ok=True)
    cache_key: str | None = None
    if mesh_cache is not None:
        # Tessellation is a pure function of BRep content and the fixed
        # mesher parameters; cached bytes are hash-verified and still flow
        # through every downstream fidelity check, so a hit cannot weaken
        # validation.
        try:
            cache_key = mesh_cache.solid_key(
                solid, mesh_parameters=_MESH_PARAMETERS
            )
        except Exception:
            cache_key = None
    if not (
        cache_key is not None and mesh_cache.fetch(cache_key, destination)
    ):
        mesher = BRepMesh_IncrementalMesh(solid, 0.20, False, 0.50, True)
        mesher.Perform()
        if not mesher.IsDone():
            raise _fail(
                "FAMILY_COLLIDER_STEP_MESH_FAILED",
                f"OCCT could not mesh {solid_id}",
            )
        StlAPI_Writer().Write(solid, str(destination))
        if not destination.is_file() or destination.stat().st_size == 0:
            raise _fail(
                "FAMILY_COLLIDER_STL_EXPORT_FAILED",
                f"OCCT did not export {solid_id}",
            )
        if cache_key is not None:
            mesh_cache.store(cache_key, destination)
    mesh = trimesh.load_mesh(destination, process=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        mesh.process(validate=True)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise _fail(
            "FAMILY_COLLIDER_MESH_EMPTY",
            f"{solid_id} produced an empty triangle mesh",
        )
    mesh.apply_scale(0.001)
    mesh.remove_unreferenced_vertices()
    return mesh


def _representation(path: Path, *, fixed: bool, convex: bool) -> str:
    from .worker import (
        _axis_aligned_y_cylinder_spec,
        _oriented_box_spec,
        _sphere_spec,
    )

    if _sphere_spec(path) is not None:
        return "ChCollisionShapeSphere"
    if _axis_aligned_y_cylinder_spec(path) is not None:
        return "ChCollisionShapeCylinder"
    if _oriented_box_spec(path) is not None:
        return "ChCollisionShapeBox"
    if fixed:
        return "ChCollisionShapeTriangleMeshStaticNonconvex"
    if convex:
        return "ChCollisionShapeTriangleMeshDynamicConvex"
    return "UNSUPPORTED_DYNAMIC_NONCONVEX_SOLID"


def _validated_component_representation(
    component: Mapping[str, Any],
    path: Path,
    *,
    fixed: bool,
) -> str:
    """Bind a collider declaration to the runtime's exact classifier."""

    actual = _representation(
        path,
        fixed=fixed,
        convex=bool(component.get("convex")),
    )
    declared = component.get("representation")
    if not isinstance(declared, str) or actual != declared:
        raise _fail(
            "FAMILY_COLLIDER_REPRESENTATION_CHANGED",
            "collision representation changed for "
            f"{component.get('component_id') or component.get('solid_id')}",
        )
    return actual


def _package_geometry_binding(
    plan: MechanismPlan,
) -> tuple[Path, str, Path, dict[str, Any]]:
    package = Path(plan.package_dir).resolve()
    manifest = read_json(package / "package.json")
    assembly_relative = manifest.get("assembly_file")
    contract_relative = manifest.get("assembly_contract_file")
    if not isinstance(assembly_relative, str) or not isinstance(contract_relative, str):
        raise _fail(
            "FAMILY_COLLIDER_PACKAGE_POINTER_MISSING",
            "package does not identify its assembly STEP and assembly contract",
        )
    assembly = ensure_inside(package, assembly_relative)
    expected_assembly_sha256 = plan.artifact_hashes.get(assembly_relative)
    if (
        expected_assembly_sha256 is None
        or not assembly.is_file()
        or assembly.is_symlink()
        or sha256_file(assembly) != expected_assembly_sha256
    ):
        raise _fail(
            "FAMILY_COLLIDER_ASSEMBLY_CHANGED",
            "assembly STEP no longer matches the intake-bound artifact",
        )
    contract_path = ensure_inside(package, contract_relative)
    expected_contract_sha256 = plan.artifact_hashes.get(contract_relative)
    if (
        expected_contract_sha256 is None
        or not contract_path.is_file()
        or contract_path.is_symlink()
        or sha256_file(contract_path) != expected_contract_sha256
    ):
        raise _fail(
            "FAMILY_COLLIDER_CONTRACT_CHANGED",
            "assembly contract no longer matches the intake-bound artifact",
        )
    return assembly, assembly_relative, contract_path, read_json(contract_path)


def materialize_family_collision_adapter(
    plan: MechanismPlan,
    adapter: AdapterSpec,
    bodies: Sequence[CompoundBodySpec],
    *,
    workspace_root: str | Path,
    output_dir: str | Path,
) -> AdapterSpec:
    """Bind compound-family physics to every authored STEP solid.

    The generated meshes are body-local and hash-bound in ``family_spec``.
    There is no whole-body convex hull or box fallback. A dynamic STEP solid
    must itself be an exact supported primitive or a verified convex component.
    """

    from .derived_adapter import _solid_indices

    if not isinstance(adapter.family_spec, dict):
        raise _fail(
            "FAMILY_COLLIDER_SPEC_REQUIRED",
            "family collision materialization requires a family_spec object",
        )
    workspace = Path(workspace_root).resolve()
    output = Path(output_dir).resolve()
    try:
        output.relative_to(workspace)
    except ValueError as exc:
        raise _fail(
            "FAMILY_COLLIDER_OUTPUT_UNSAFE",
            "family collider output must stay inside the workspace",
        ) from exc

    assembly, assembly_relative, contract_path, contract = _package_geometry_binding(plan)
    solids = _load_step_solids(assembly)
    derived_root = output / "derived_adapters" / plan.capsule_id / "family_collision"
    source_stl_root = derived_root / "source_stl"
    component_root = derived_root / "bodies"

    body_indices: dict[str, tuple[int, ...]] = {}
    claimed_indices: dict[int, str] = {}
    for body in bodies:
        indices = _solid_indices(contract, set(body.group_ids))
        for index in indices:
            previous = claimed_indices.get(index)
            if previous is not None and previous != body.body_id:
                raise _fail(
                    "FAMILY_COLLIDER_SOLID_BODY_OVERLAP",
                    f"solid_{index} is assigned to both {previous!r} and {body.body_id!r}",
                )
            claimed_indices[index] = body.body_id
        body_indices[body.body_id] = indices

    rigid_attachment_path = derived_root / "rigid_attachment_proof.json"
    rigid_attachment_report = build_rigid_attachment_proof(
        solids,
        body_indices,
        package_binding_sha256=plan.package_binding_sha256,
        body_group_ids={
            body.body_id: tuple(body.group_ids)
            for body in bodies
        },
    )
    write_json_atomic(rigid_attachment_path, rigid_attachment_report)
    rigid_attachment = {
        "schema_version": rigid_attachment_report["schema_version"],
        "status": rigid_attachment_report["status"],
        "passed": rigid_attachment_report["passed"],
        "report": relative_to_root(workspace, rigid_attachment_path),
        "report_sha256": sha256_file(rigid_attachment_path),
        "package_binding_sha256": plan.package_binding_sha256,
    }

    from .reuse import mesh_cache_for_run

    mesh_cache = mesh_cache_for_run(output)
    raw_meshes: dict[int, Any] = {}
    for index in sorted(claimed_indices):
        if index < 1 or index > len(solids):
            raise _fail(
                "FAMILY_COLLIDER_SOLID_OUT_OF_RANGE",
                f"solid_{index} is outside the imported STEP inventory",
            )
        raw_meshes[index] = _export_solid_mesh(
            solids[index - 1],
            f"solid_{index}",
            source_stl_root / f"solid_{index}.stl",
            mesh_cache=mesh_cache,
        )

    # Only explicitly approved mechanism/plugin pairs may consume package
    # component declarations.  The package merely locates hash-bound convex
    # STEP assets; Harness independently proves every component and its union.
    dynamic_nonconvex_sources: dict[
        tuple[str, str],
        tuple[Any, Any],
    ] = {}
    for body in bodies:
        if body.fixed:
            continue
        for index in body_indices[body.body_id]:
            mesh = raw_meshes[index]
            if not bool(mesh.is_watertight and mesh.is_convex):
                dynamic_nonconvex_sources[
                    (body.body_id, f"solid_{index}")
                ] = (solids[index - 1], mesh)
    authored_collision_sources: dict[tuple[str, str], Any] = {}
    if dynamic_nonconvex_sources:
        bicycle_sidecar_authorized = (
            _authored_collision_component_plugin_id(adapter) is not None
        )
        if bicycle_sidecar_authorized:
            from .authored_collision_components import (
                load_authored_collision_components,
            )

            authored_collision_sources = {
                (item.body_id, item.source_solid_id): item
                for item in load_authored_collision_components(
                    plan,
                    package=Path(plan.package_dir).resolve(),
                    workspace=workspace,
                    expected_sources=dynamic_nonconvex_sources,
                    scratch_root=derived_root / "authored_component_validation",
                    characteristic_length_m=plan.characteristic_length_m,
                    mesh_linear_deflection_m=0.0002,
                    collision_envelope_m=12e-6,
                )
            }
    raw_capture_declarations = adapter.family_spec.get(
        "finite_linear_captures",
        (),
    )
    if not isinstance(raw_capture_declarations, Sequence) or isinstance(
        raw_capture_declarations,
        (str, bytes),
    ):
        raise _fail(
            "FINITE_CAPTURE_DECLARATION_INVALID",
            "finite_linear_captures must be an array",
        )
    capture_declarations = tuple(
        finite_linear_capture_from_dict(
            item,
            label=f"finite_linear_captures[{index}]",
        )
        for index, item in enumerate(raw_capture_declarations)
        if isinstance(item, Mapping)
    )
    if len(capture_declarations) != len(raw_capture_declarations):
        raise _fail(
            "FINITE_CAPTURE_DECLARATION_INVALID",
            "every finite_linear_captures entry must be an object",
        )
    finite_capture_report = build_finite_linear_capture_report(
        capture_declarations,
        raw_meshes=raw_meshes,
        body_solid_indices=body_indices,
        package_binding_sha256=plan.package_binding_sha256,
        mesh_tolerance_m=0.0002,
    )
    finite_capture_path = derived_root / "finite_linear_capture.json"
    write_json_atomic(finite_capture_path, finite_capture_report)

    body_payloads: list[dict[str, Any]] = []
    unsupported: list[dict[str, str]] = []
    fidelity_components: list[dict[str, Any]] = []
    authored_component_reports: list[dict[str, Any]] = []
    for body in bodies:
        rotation = _rotation_matrix_from_quaternion(body.rotation_abs_quat_wxyz)
        components: list[dict[str, Any]] = []
        for index in body_indices[body.body_id]:
            import trimesh

            solid_id = f"solid_{index}"
            authored = authored_collision_sources.get(
                (body.body_id, solid_id)
            )
            if authored is not None:
                exported_world_meshes: dict[str, Any] = {}
                exported_asset_paths: dict[str, Path] = {}
                exported_asset_sha256: dict[str, str] = {}
                for declared in authored.components:
                    mesh = declared.mesh_world_m.copy()
                    mesh.vertices = [
                        _world_to_body_local(
                            vertex,
                            center_abs_m=body.center_abs_m,
                            rotation_abs_from_body=rotation,
                        )
                        for vertex in mesh.vertices
                    ]
                    mesh.remove_unreferenced_vertices()
                    component_path = (
                        component_root
                        / body.body_id
                        / solid_id
                        / f"{declared.component_id}.obj"
                    )
                    component_path.parent.mkdir(parents=True, exist_ok=True)
                    component_path.write_text(
                        trimesh.exchange.obj.export_obj(
                            mesh,
                            include_normals=False,
                            include_color=False,
                            include_texture=False,
                            digits=17,
                        ),
                        encoding="ascii",
                        newline="\n",
                    )
                    loaded = trimesh.load_mesh(
                        component_path,
                        process=False,
                        force="mesh",
                    )
                    if not isinstance(loaded, trimesh.Trimesh):
                        loaded = trimesh.util.concatenate(
                            tuple(loaded.geometry.values())
                        )
                    loaded.vertices = [
                        _body_local_to_world(
                            vertex,
                            center_abs_m=body.center_abs_m,
                            rotation_abs_from_body=rotation,
                        )
                        for vertex in loaded.vertices
                    ]
                    loaded.remove_unreferenced_vertices()
                    exported_world_meshes[declared.component_id] = loaded
                    exported_asset_paths[declared.component_id] = (
                        component_path
                    )
                    exported_asset_sha256[declared.component_id] = (
                        sha256_file(component_path)
                    )
                    components.append(
                        {
                            "component_id": declared.component_id,
                            "solid_id": solid_id,
                            "mesh": relative_to_root(
                                workspace,
                                component_path,
                            ),
                            "sha256": sha256_file(component_path),
                            "vertex_count": int(len(mesh.vertices)),
                            "triangle_count": int(len(mesh.faces)),
                            "watertight": bool(mesh.is_watertight),
                            "convex": True,
                            "representation": _representation(
                                component_path,
                                fixed=body.fixed,
                                convex=True,
                            ),
                            "body_local": True,
                            "units": "m",
                            "source_component_step": declared.step_relative,
                            "source_component_step_sha256": (
                                declared.step_sha256
                            ),
                            "materialization_authority": "harness_occt",
                            "decomposition_kind": (
                                "package_declared_harness_verified_convex_steps"
                            ),
                        }
                    )
                from .authored_collision_components import (
                    bind_exported_component_meshes,
                )

                fidelity_input, bound_report = bind_exported_component_meshes(
                    authored,
                    exported_world_meshes,
                    exported_asset_paths=exported_asset_paths,
                    exported_asset_sha256=exported_asset_sha256,
                )
                fidelity_components.append(fidelity_input)
                report_path = (
                    derived_root
                    / "authored_component_reports"
                    / f"{body.body_id}.{solid_id}.json"
                )
                write_json_atomic(report_path, bound_report)
                authored_component_reports.append(
                    {
                        "body_id": body.body_id,
                        "source_solid_id": solid_id,
                        "status": "PASS",
                        "passed": True,
                        "report": relative_to_root(workspace, report_path),
                        "report_sha256": sha256_file(report_path),
                        "component_count": len(authored.components),
                        "runtime_component_mesh_bytes_fidelity_verified": True,
                    }
                )
                continue

            source_world_mesh = raw_meshes[index].copy()
            mesh = source_world_mesh.copy()
            mesh.vertices = [
                _world_to_body_local(
                    vertex,
                    center_abs_m=body.center_abs_m,
                    rotation_abs_from_body=rotation,
                )
                for vertex in mesh.vertices
            ]
            mesh.remove_unreferenced_vertices()
            component_path = component_root / body.body_id / f"{solid_id}.obj"
            component_path.parent.mkdir(parents=True, exist_ok=True)
            component_path.write_text(
                trimesh.exchange.obj.export_obj(
                    mesh,
                    include_normals=False,
                    include_color=False,
                    include_texture=False,
                ),
                encoding="ascii",
                newline="\n",
            )
            convex = bool(mesh.is_watertight and mesh.is_convex)
            representation = _representation(
                component_path,
                fixed=body.fixed,
                convex=convex,
            )
            collider_mesh = trimesh.load_mesh(
                component_path,
                process=False,
                force="mesh",
            )
            if not isinstance(collider_mesh, trimesh.Trimesh):
                collider_mesh = trimesh.util.concatenate(
                    tuple(collider_mesh.geometry.values())
                )
            fidelity_components.append(
                {
                    "body_id": body.body_id,
                    "solid_id": solid_id,
                    "collider_id": relative_to_root(workspace, component_path),
                    "representation": representation,
                    "source_mesh": mesh.copy(),
                    "collider_mesh": collider_mesh,
                }
            )
            components.append(
                {
                    "component_id": solid_id,
                    "solid_id": solid_id,
                    "mesh": relative_to_root(workspace, component_path),
                    "sha256": sha256_file(component_path),
                    "vertex_count": int(len(mesh.vertices)),
                    "triangle_count": int(len(mesh.faces)),
                    "watertight": bool(mesh.is_watertight),
                    "convex": convex,
                    "representation": representation,
                    "body_local": True,
                    "units": "m",
                    "materialization_authority": "harness_occt",
                }
            )
            if representation == "UNSUPPORTED_DYNAMIC_NONCONVEX_SOLID":
                unsupported.append(
                    {"body_id": body.body_id, "solid_id": solid_id}
                )
        body_payloads.append(
            {
                "body_id": body.body_id,
                "group_ids": list(body.group_ids),
                "source_solid_ids": [
                    f"solid_{index}" for index in body_indices[body.body_id]
                ],
                "fixed": body.fixed,
                "initial_center_abs_m": list(body.center_abs_m),
                "initial_rotation_abs_quat_wxyz": list(
                    body.rotation_abs_quat_wxyz
                ),
                "components": components,
            }
        )

    collision_geometry = {
        "schema": FAMILY_COLLISION_SCHEMA,
        "source": "intake_bound_authoritative_assembly_step",
        "package_binding_sha256": plan.package_binding_sha256,
        "assembly_file": assembly_relative,
        "assembly_sha256": sha256_file(assembly),
        "assembly_contract": relative_to_root(workspace, contract_path),
        "assembly_contract_sha256": sha256_file(contract_path),
        "body_count": len(body_payloads),
        "component_count": sum(
            len(body["components"]) for body in body_payloads
        ),
        "bodies": body_payloads,
        "contact_enabled_in_physical_branch": True,
        "contact_disabled_in_reduced_branch": True,
        "whole_body_box_proxy_used": False,
        "whole_body_convex_hull_used": False,
        "convex_hull_used": False,
        "authored_collision_component_manifest_used": bool(
            authored_component_reports
        ),
        "authored_collision_component_reports": authored_component_reports,
        "component_mesh_export_authority": "harness_occt",
        "mesh_cache": (
            mesh_cache.stats() if mesh_cache is not None else None
        ),
        "mesh_linear_deflection_m": 0.0002,
        "mesh_angular_deflection_rad": 0.5,
        "collision_envelope_m": 12e-6,
        "collision_margin_m": 2e-6,
        "interpenetration_tolerance_m": 0.00035,
        "runtime_interpenetration_tolerance_m": 0.00005,
        "physical_inter_body_collision_policy": "all_pairs_enabled",
        "rigid_attachment_proof": rigid_attachment,
        "unsupported_dynamic_components": unsupported,
        "finite_linear_capture": {
            **finite_capture_report,
            "report": relative_to_root(workspace, finite_capture_path),
            "report_sha256": sha256_file(finite_capture_path),
        },
    }
    assembly_connectivity_path = (
        derived_root / "assembly_spatial_connectivity.json"
    )
    assembly_connectivity_report = build_assembly_connectivity_report(
        {
            body.body_id: tuple(
                (f"solid_{index}", raw_meshes[index])
                for index in body_indices[body.body_id]
            )
            for body in bodies
        },
        package_binding_sha256=plan.package_binding_sha256,
        mesh_linear_deflection_m=float(
            collision_geometry["mesh_linear_deflection_m"]
        ),
        collision_envelope_m=float(
            collision_geometry["collision_envelope_m"]
        ),
        interpenetration_tolerance_m=float(
            collision_geometry["interpenetration_tolerance_m"]
        ),
    )
    write_json_atomic(
        assembly_connectivity_path,
        assembly_connectivity_report,
    )
    collision_geometry["assembly_connectivity"] = {
        "schema_version": assembly_connectivity_report["schema_version"],
        "status": assembly_connectivity_report["status"],
        "passed": assembly_connectivity_report["passed"],
        "report": relative_to_root(
            workspace,
            assembly_connectivity_path,
        ),
        "report_sha256": sha256_file(assembly_connectivity_path),
        "package_binding_sha256": plan.package_binding_sha256,
    }
    if unsupported:
        try:
            from .failure_evidence import write_geometry_legality_failure

            collision_geometry["failure_evidence"] = write_geometry_legality_failure(
                collision_geometry,
                workspace_root=workspace,
                evidence_root=derived_root / "failure_evidence" / "geometry_legality",
            )
        except Exception as exc:
            collision_geometry["failure_evidence_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
        write_json_atomic(derived_root / "provenance.json", collision_geometry)
        detail = ", ".join(
            f"{item['body_id']}/{item['solid_id']}" for item in unsupported[:12]
        )
        if len(unsupported) > 12:
            detail += f", and {len(unsupported) - 12} more"
        repair_targets = [
            {
                "body_id": str(item["body_id"]),
                "solid_id": str(item["solid_id"]),
                "source_solid_id": str(item["solid_id"]),
            }
            for item in unsupported
        ]
        raise _fail(
            "FAMILY_DYNAMIC_COLLIDER_REPRESENTATION_UNSUPPORTED",
            "dynamic STEP solids must be exact supported primitives or verified "
            f"convex components; unsupported: {detail}",
            owner=FailureOwner.DESIGN_PACKAGE,
            repair_scope=RepairScope.DERIVED_ARTIFACT,
            details={
                "unsupported_dynamic_components": repair_targets,
                "repair_action": "generate_certified_collision_components",
                "repair_tool_module": (
                    "simulation_ready_cad_workflow.validation_pipeline.tools."
                    "compile_design_package"
                ),
                "authoritative_geometry_policy": (
                    "preserve_assembly_step_bytes"
                ),
                "repair_command_arguments": [
                    f"--collision-source={item['body_id']}:"
                    f"{item['source_solid_id']}"
                    for item in repair_targets
                ],
            },
        )

    from .collider_fidelity import build_collider_fidelity_report

    fidelity_report = build_collider_fidelity_report(
        fidelity_components,
        package_binding_sha256=plan.package_binding_sha256,
        characteristic_length_m=plan.characteristic_length_m,
        mesh_linear_deflection_m=float(
            collision_geometry["mesh_linear_deflection_m"]
        ),
        collision_envelope_m=float(collision_geometry["collision_envelope_m"]),
    )
    fidelity_path = derived_root / "collider_fidelity.json"
    write_json_atomic(fidelity_path, fidelity_report)
    collision_geometry["collider_fidelity"] = {
        "schema_version": fidelity_report["schema_version"],
        "status": fidelity_report["status"],
        "passed": fidelity_report["passed"],
        "report": relative_to_root(workspace, fidelity_path),
        "report_sha256": sha256_file(fidelity_path),
        "package_binding_sha256": plan.package_binding_sha256,
        "rigid_attachment_proof": rigid_attachment,
    }
    from .physical_legality import audit_family_collision_geometry

    home_sample = {
        "time_s": 0.0,
        "phase_id": "home_geometry_preflight",
        "body_poses": {
            item["body_id"]: {
                "position_abs_m": item["initial_center_abs_m"],
                "rotation_abs_quat_wxyz": item[
                    "initial_rotation_abs_quat_wxyz"
                ],
            }
            for item in body_payloads
        },
    }
    home_audit = audit_family_collision_geometry(
        collision_geometry,
        [home_sample],
        workspace_root=workspace,
    )
    collision_geometry["home_interpenetration_audit"] = home_audit
    write_json_atomic(
        derived_root / "home_interpenetration_audit.json",
        home_audit,
    )
    static_checks = [
        {
            "check_id": "rigid_attachment",
            "status": rigid_attachment_report["status"],
            "passed": rigid_attachment_report["passed"],
            "report": rigid_attachment["report"],
            "report_sha256": rigid_attachment["report_sha256"],
        },
        {
            "check_id": "assembly_connectivity",
            "status": assembly_connectivity_report["status"],
            "passed": assembly_connectivity_report["passed"],
            "report": collision_geometry["assembly_connectivity"]["report"],
            "report_sha256": collision_geometry[
                "assembly_connectivity"
            ]["report_sha256"],
            "assembly_island_count": assembly_connectivity_report[
                "assembly_island_count"
            ],
            "failure_reasons": assembly_connectivity_report[
                "failure_reasons"
            ],
        },
        {
            "check_id": "collider_fidelity",
            "status": fidelity_report["status"],
            "passed": fidelity_report["passed"],
            "report": collision_geometry["collider_fidelity"]["report"],
            "report_sha256": collision_geometry[
                "collider_fidelity"
            ]["report_sha256"],
        },
        {
            "check_id": "home_interpenetration",
            "status": "PASS" if home_audit["passed"] else "FAIL",
            "passed": home_audit["passed"],
            "report": relative_to_root(
                workspace,
                derived_root / "home_interpenetration_audit.json",
            ),
            "report_sha256": sha256_file(
                derived_root / "home_interpenetration_audit.json"
            ),
            "authorization_requires_passing_collider_fidelity": True,
        },
        {
            "check_id": "finite_linear_capture",
            "status": finite_capture_report["status"],
            "passed": finite_capture_report["passed"],
            "required": finite_capture_report["required"],
            "report": relative_to_root(workspace, finite_capture_path),
            "report_sha256": sha256_file(finite_capture_path),
            "failure_reasons": finite_capture_report["failure_reasons"],
        },
    ]
    static_passed = all(item["passed"] is True for item in static_checks)
    static_report = {
        "schema_version": "family_static_geometry_preflight_v1",
        "status": "PASS" if static_passed else "FAIL",
        "passed": static_passed,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "all_independent_checks_executed": True,
        "physical_simulation_started": False,
        "checks": static_checks,
        "failure_reasons": [
            f"{item['check_id']}_failed"
            for item in static_checks
            if item["passed"] is not True
        ],
    }
    static_report_path = derived_root / "static_geometry_preflight.json"
    write_json_atomic(static_report_path, static_report)
    collision_geometry["static_geometry_preflight"] = {
        "schema_version": static_report["schema_version"],
        "status": static_report["status"],
        "passed": static_report["passed"],
        "report": relative_to_root(workspace, static_report_path),
        "report_sha256": sha256_file(static_report_path),
        "package_binding_sha256": plan.package_binding_sha256,
    }

    write_json_atomic(derived_root / "provenance.json", collision_geometry)

    family_spec = dict(adapter.family_spec)
    family_spec["collision_geometry"] = collision_geometry
    return replace(
        adapter,
        family_spec=family_spec,
        collider_fidelity=dict(collision_geometry["collider_fidelity"]),
    )


def _validated_family_collision_manifest(
    adapter: AdapterSpec,
    *,
    expected_bodies: Sequence[CompoundBodySpec],
) -> dict[str, Any]:
    family_spec = adapter.family_spec
    geometry = (
        family_spec.get("collision_geometry")
        if isinstance(family_spec, Mapping)
        else None
    )
    if not isinstance(geometry, dict) or geometry.get("schema") != FAMILY_COLLISION_SCHEMA:
        raise _fail(
            "FAMILY_COLLIDER_NOT_MATERIALIZED",
            "physical family execution requires a materialized STEP per-solid collision manifest",
        )
    if geometry.get("package_binding_sha256") is None:
        raise _fail(
            "FAMILY_COLLIDER_PACKAGE_BINDING_MISSING",
            "family collision manifest is not package-bound",
        )
    payloads = geometry.get("bodies")
    if not isinstance(payloads, list):
        raise _fail(
            "FAMILY_COLLIDER_BODY_MANIFEST_INVALID",
            "family collision manifest has no body inventory",
        )
    expected = {body.body_id: body for body in expected_bodies}
    actual = {
        str(item.get("body_id")): item
        for item in payloads
        if isinstance(item, dict)
    }
    if set(actual) != set(expected):
        raise _fail(
            "FAMILY_COLLIDER_BODY_SET_MISMATCH",
            "family collision manifest does not match the executed body set",
        )
    for body_id, body in expected.items():
        item = actual[body_id]
        if tuple(item.get("group_ids", ())) != body.group_ids:
            raise _fail(
                "FAMILY_COLLIDER_GROUP_SET_MISMATCH",
                f"collision groups changed for body {body_id!r}",
            )
        if bool(item.get("fixed")) != body.fixed:
            raise _fail(
                "FAMILY_COLLIDER_FIXED_STATE_MISMATCH",
                f"collision fixed state changed for body {body_id!r}",
            )
        components = item.get("components")
        if not isinstance(components, list) or not components:
            raise _fail(
                "FAMILY_COLLIDER_COMPONENTS_MISSING",
                f"body {body_id!r} has no STEP collision components",
            )
    authored_reports = geometry.get("authored_collision_component_reports", [])
    authored_used = geometry.get(
        "authored_collision_component_manifest_used",
        False,
    )
    if not isinstance(authored_reports, list) or bool(authored_reports) != bool(
        authored_used
    ):
        raise _fail(
            "COLLISION_COMPONENT_REPORT_BINDING_INVALID",
            "authored collision-component report inventory is inconsistent",
        )
    expected_authored: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for body_id, body_payload in actual.items():
        for component in body_payload.get("components", []):
            if (
                isinstance(component, Mapping)
                and component.get("decomposition_kind")
                == "package_declared_harness_verified_convex_steps"
            ):
                expected_authored.setdefault(
                    (body_id, str(component.get("solid_id"))),
                    [],
                ).append(component)
    actual_authored: set[tuple[str, str]] = set()
    for binding in authored_reports:
        if not isinstance(binding, Mapping):
            raise _fail(
                "COLLISION_COMPONENT_REPORT_BINDING_INVALID",
                "authored collision-component report entry is invalid",
            )
        key = (
            str(binding.get("body_id")),
            str(binding.get("source_solid_id")),
        )
        if key in actual_authored:
            raise _fail(
                "COLLISION_COMPONENT_REPORT_BINDING_INVALID",
                f"duplicate authored collision-component report for {key!r}",
            )
        actual_authored.add(key)
        expected_components = expected_authored.get(key, [])
        if (
            binding.get("passed") is not True
            or binding.get("status") != "PASS"
            or binding.get("runtime_component_mesh_bytes_fidelity_verified")
            is not True
            or int(binding.get("component_count", -1))
            != len(expected_components)
        ):
            raise _fail(
                "COLLISION_COMPONENT_REPORT_BINDING_INVALID",
                f"authored collision-component report summary is invalid for {key!r}",
            )
    if actual_authored != set(expected_authored):
        raise _fail(
            "COLLISION_COMPONENT_REPORT_BINDING_INVALID",
            "authored collision-component reports do not match runtime components",
        )
    return geometry


def family_collision_geometry_for_static_preflight(
    adapter: AdapterSpec,
    *,
    expected_bodies: Sequence[CompoundBodySpec],
) -> dict[str, Any]:
    """Read hash-bound STEP components without granting execution authority."""

    return _validated_family_collision_manifest(
        adapter,
        expected_bodies=expected_bodies,
    )


def family_collision_geometry(
    adapter: AdapterSpec,
    *,
    expected_bodies: Sequence[CompoundBodySpec],
) -> dict[str, Any]:
    geometry = _validated_family_collision_manifest(
        adapter,
        expected_bodies=expected_bodies,
    )
    static_preflight = geometry.get("static_geometry_preflight")
    if (
        not isinstance(static_preflight, Mapping)
        or static_preflight.get("passed") is not True
        or static_preflight.get("status") != "PASS"
        or static_preflight.get("package_binding_sha256")
        != geometry.get("package_binding_sha256")
    ):
        raise _fail(
            "FAMILY_STATIC_GEOMETRY_PREFLIGHT_REQUIRED",
            "physical family execution requires passing aggregate static "
            "geometry preflight evidence",
        )
    fidelity = geometry.get("collider_fidelity")
    if not isinstance(fidelity, Mapping) or fidelity.get("passed") is not True:
        raise _fail(
            "COLLIDER_FIDELITY_EVIDENCE_REQUIRED",
            "physical family execution requires passing collider-fidelity evidence",
        )
    if fidelity.get("package_binding_sha256") != geometry.get(
        "package_binding_sha256"
    ):
        raise _fail(
            "COLLIDER_FIDELITY_PACKAGE_BINDING_MISMATCH",
            "collider-fidelity evidence is not bound to the executed package",
        )
    return geometry


def make_family_body(
    chrono: Any,
    *,
    body_spec: CompoundBodySpec,
    geometry: Mapping[str, Any],
    workspace: Path,
    material: Any,
    inertia_xx: tuple[float, float, float],
    branch: str,
    runtime_suppressed_solid_ids: Sequence[str] = (),
    runtime_suppressed_component_ids: Sequence[str] = (),
) -> tuple[Any, dict[str, Any]]:
    body_manifest = next(
        item
        for item in geometry["bodies"]
        if item.get("body_id") == body_spec.body_id
    )
    components = body_manifest["components"]
    suppressed = frozenset(str(item) for item in runtime_suppressed_solid_ids)
    suppressed_components = frozenset(
        str(item) for item in runtime_suppressed_component_ids
    )
    known_solid_ids = {str(item["solid_id"]) for item in components}
    known_component_ids = {
        str(item.get("component_id") or item["solid_id"])
        for item in components
    }
    unknown_suppressed = suppressed - known_solid_ids
    if unknown_suppressed:
        raise _fail(
            "RUNTIME_COMPONENT_SUPPRESSION_BINDING_INVALID",
            f"unknown suppressed collider components for {body_spec.body_id}: "
            + ", ".join(sorted(unknown_suppressed)),
        )
    unknown_suppressed_components = (
        suppressed_components - known_component_ids
    )
    if unknown_suppressed_components:
        raise _fail(
            "RUNTIME_COMPONENT_SUPPRESSION_BINDING_INVALID",
            f"unknown suppressed collider component IDs for "
            f"{body_spec.body_id}: "
            + ", ".join(sorted(unknown_suppressed_components)),
        )
    attached_components = [
        item
        for item in components
        if str(item["solid_id"]) not in suppressed
        and str(item.get("component_id") or item["solid_id"])
        not in suppressed_components
    ]
    if not attached_components and (suppressed or suppressed_components):
        raise _fail(
            "RUNTIME_COMPONENT_SUPPRESSION_BODY_EMPTY",
            f"runtime component suppression removed every collider from {body_spec.body_id}",
        )
    if branch == "physical":
        fidelity = geometry.get("collider_fidelity")
        if not isinstance(fidelity, Mapping):
            raise _fail(
                "COLLIDER_FIDELITY_EVIDENCE_REQUIRED",
                "physical execution has no collider-fidelity report binding",
            )
        fidelity_path = ensure_inside(workspace, str(fidelity.get("report") or ""))
        if (
            not fidelity_path.is_file()
            or fidelity_path.is_symlink()
            or sha256_file(fidelity_path) != fidelity.get("report_sha256")
        ):
            raise _fail(
                "COLLIDER_FIDELITY_REPORT_CHANGED",
                "collider-fidelity evidence is missing or no longer hash-bound",
            )
        fidelity_report = read_json(fidelity_path)
        if (
            fidelity_report.get("passed") is not True
            or fidelity_report.get("package_binding_sha256")
            != geometry.get("package_binding_sha256")
        ):
            raise _fail(
                "COLLIDER_FIDELITY_REPORT_INVALID",
                "collider-fidelity evidence cannot authorize physical execution",
            )
        authored_by_solid: dict[str, list[Mapping[str, Any]]] = {}
        for component in components:
            if (
                isinstance(component, Mapping)
                and component.get("decomposition_kind")
                == "package_declared_harness_verified_convex_steps"
            ):
                authored_by_solid.setdefault(
                    str(component.get("solid_id")),
                    [],
                ).append(component)
        report_bindings = geometry.get(
            "authored_collision_component_reports",
            [],
        )
        for solid_id, authored_components in sorted(authored_by_solid.items()):
            binding = next(
                (
                    item
                    for item in report_bindings
                    if isinstance(item, Mapping)
                    and item.get("body_id") == body_spec.body_id
                    and item.get("source_solid_id") == solid_id
                ),
                None,
            )
            if not isinstance(binding, Mapping):
                raise _fail(
                    "COLLISION_COMPONENT_REPORT_BINDING_INVALID",
                    f"runtime has no authored component report for "
                    f"{body_spec.body_id}/{solid_id}",
                )
            report_path = ensure_inside(
                workspace,
                str(binding.get("report") or ""),
            )
            if (
                not report_path.is_file()
                or report_path.is_symlink()
                or sha256_file(report_path) != binding.get("report_sha256")
            ):
                raise _fail(
                    "COLLISION_COMPONENT_REPORT_CHANGED",
                    f"authored component report changed for "
                    f"{body_spec.body_id}/{solid_id}",
                )
            report = read_json(report_path)
            expected_ids = sorted(
                str(item.get("component_id"))
                for item in authored_components
            )
            expected_source_hashes = sorted(
                str(item.get("source_component_step_sha256"))
                for item in authored_components
            )
            expected_runtime_hashes = sorted(
                (
                    str(item.get("component_id")),
                    str(item.get("sha256")),
                )
                for item in authored_components
            )
            raw_runtime_exports = report.get(
                "runtime_component_mesh_exports",
            )
            reported_runtime_hashes = (
                sorted(
                    (
                        str(item.get("component_id")),
                        str(item.get("runtime_obj_sha256")),
                    )
                    for item in raw_runtime_exports
                    if isinstance(item, Mapping)
                )
                if isinstance(raw_runtime_exports, list)
                else []
            )
            if (
                report.get("passed") is not True
                or report.get("status") != "PASS"
                or report.get("package_binding_sha256")
                != geometry.get("package_binding_sha256")
                or report.get("body_id") != body_spec.body_id
                or report.get("source_solid_id") != solid_id
                or sorted(report.get("component_ids", [])) != expected_ids
                or sorted(report.get("component_step_sha256", []))
                != expected_source_hashes
                or reported_runtime_hashes != expected_runtime_hashes
                or report.get("runtime_component_mesh_binding_method")
                != "deterministic_obj_round_trip_topology_and_vertices_v1"
                or report.get("runtime_union_surface_authority")
                != (
                    "occt_boolean_union_of_round_trip_bound_component_steps"
                )
                or report.get(
                    "runtime_component_mesh_bytes_fidelity_verified"
                )
                is not True
            ):
                raise _fail(
                    "COLLISION_COMPONENT_REPORT_INVALID",
                    f"authored component report cannot authorize "
                    f"{body_spec.body_id}/{solid_id}",
                )
        local_meshes: list[tuple[Path, bool]] = []
        for component in attached_components:
            path = ensure_inside(workspace, str(component["mesh"]))
            if not path.is_file() or path.is_symlink():
                raise _fail(
                    "FAMILY_COLLIDER_ASSET_MISSING",
                    f"collision asset is missing for {body_spec.body_id}/{component['solid_id']}",
                )
            if sha256_file(path) != component.get("sha256"):
                raise _fail(
                    "FAMILY_COLLIDER_ASSET_CHANGED",
                    f"collision asset changed for {body_spec.body_id}/{component['solid_id']}",
                )
            _validated_component_representation(
                component,
                path,
                fixed=body_spec.fixed,
            )
            local_meshes.append((path, bool(component.get("convex"))))

        from .worker import _make_body

        body = _make_body(
            chrono,
            name=body_spec.body_id,
            local_meshes=local_meshes,
            origin_abs_m=body_spec.center_abs_m,
            fixed=body_spec.fixed,
            mass_kg=body_spec.mass_kg,
            inertia_xx=inertia_xx,
            material=material,
            collision_radius_m=0.0,
            physical_contact_required=True,
        )
        body.SetRot(chrono.ChQuaterniond(*body_spec.rotation_abs_quat_wxyz))
        contact_enabled = True
    else:
        body = chrono.ChBodyAuxRef()
        body.SetName(body_spec.body_id)
        body.SetFixed(body_spec.fixed)
        body.SetPos(chrono.ChVector3d(*body_spec.center_abs_m))
        body.SetRot(chrono.ChQuaterniond(*body_spec.rotation_abs_quat_wxyz))
        body.SetMass(body_spec.mass_kg)
        body.SetInertiaXX(chrono.ChVector3d(*inertia_xx))
        body.EnableCollision(False)
        contact_enabled = False

    audit = {
        "body_id": body_spec.body_id,
        "group_ids": list(body_spec.group_ids),
        "fixed": body_spec.fixed,
        "contact_enabled": contact_enabled,
        "component_count": len(components),
        "runtime_attached_component_count": (
            len(attached_components) if branch == "physical" else 0
        ),
        "validator_owned_suppressed_solid_ids": sorted(suppressed),
        "validator_owned_suppressed_component_ids": sorted(
            suppressed_components
        ),
        "source_solid_ids": [item["solid_id"] for item in components],
        "representations": [item["representation"] for item in components],
        "component_hashes": [item["sha256"] for item in components],
        "components": [
            {
                "component_id": (
                    item.get("component_id") or item["solid_id"]
                ),
                "solid_id": item["solid_id"],
                "derived": item["mesh"],
                "derived_sha256": item["sha256"],
                "representation": item["representation"],
                "convex": bool(item["convex"]),
            }
            for item in components
        ],
    }
    return body, audit
