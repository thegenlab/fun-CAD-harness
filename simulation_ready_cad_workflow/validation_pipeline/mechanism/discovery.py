from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from ..contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from .adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    DERIVED_PLAIN_PRISMATIC_ADAPTER_ID,
    DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
    adapter_to_dict,
    get_adapter,
    select_adapter,
)
from .io import canonical_sha256, ensure_inside, read_json, sha256_file
from ..intake.acceptance_contract import (
    compile_acceptance_contract,
    contract_for_tasks,
    verify_acceptance_contract,
    verify_task_contract,
)
from ..intake.semantic_binding import (
    bind_acceptance_contract,
    binding_for_tasks,
    verify_semantic_binding,
    verify_task_binding,
)
from .models import MechanismPlan, MechanismValidationError, vector3
from .plugin_boundary import plugin_boundary_error
from .task_motion_evidence import build_task_channel_registry


_ACCEPTED_STATUSES = {"ACCEPTED", "ACCEPTED_WITH_WARNINGS"}
_PROTOCOLS = {
    "prismatic": "prismatic_slider",
    "revolute": "revolute_pivot_hinge",
    "compound": "revolute_linkage",
    "continuous_revolute": "continuous_revolute",
    "cylindrical": "cylindrical",
    "helical": "screw_helical",
    "screw": "screw_helical",
}


def _fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(
        code,
        message,
        failure=make_failure_envelope(
            code,
            FailureOwner.DESIGN_PACKAGE,
            source_component="mechanism.discovery",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        ),
    )


def _plan_fail(code: str, message: str) -> MechanismValidationError:
    return MechanismValidationError(
        code,
        message,
        failure=make_failure_envelope(
            code,
            FailureOwner.WORKER_PLAN,
            source_component="mechanism.discovery",
            message=message,
            evidence_validity=EvidenceValidity.NOT_APPLICABLE,
        ),
    )


def _serialized_family_spec(
    raw_family_spec: dict[str, Any],
    parsed_family_spec: Any,
    spec_plugin: Any,
    *,
    preserve_project_adapter_file: bool = False,
) -> dict[str, Any]:
    """Keep typed fields as detached, finite, JSON-native data.

    Typed mechanism dataclasses deliberately use tuples for immutable vectors
    and identifier collections. dataclasses.asdict preserves those tuples,
    but the adapter payload is a JSON document and downstream runtime binding
    validates its arrays as lists. Normalize once at this boundary so an
    in-process discovery has the same shape as a persisted/reloaded plan.
    """

    try:
        payload = json.loads(
            json.dumps(
                parsed_family_spec.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        from .family_plugins import plugin_registration_or_none

        message = f"typed family spec is not finite JSON: {exc}"
        boundary_error = plugin_boundary_error(
            exc,
            registration=plugin_registration_or_none(
                str(spec_plugin.plugin_id)
            ),
            plugin_id=str(spec_plugin.plugin_id),
            plugin_kind="mechanism",
            callback="parse_family_spec",
            source_component="mechanism.discovery.parse_family_spec",
            code="MECHANISM_FAMILY_SPEC_SERIALIZATION_INVALID",
            message=message,
            contract_violation=True,
        )
        if boundary_error is exc:
            raise
        raise boundary_error from exc
    if not isinstance(payload, dict):
        from .family_plugins import plugin_registration_or_none

        message = "typed family spec must serialize to a JSON object"
        raise plugin_boundary_error(
            TypeError(message),
            registration=plugin_registration_or_none(
                str(spec_plugin.plugin_id)
            ),
            plugin_id=str(spec_plugin.plugin_id),
            plugin_kind="mechanism",
            callback="parse_family_spec",
            source_component="mechanism.discovery.parse_family_spec",
            code="MECHANISM_FAMILY_SPEC_SERIALIZATION_INVALID",
            message=message,
            contract_violation=True,
        )
    if (
        "compliant_surface_contacts" in spec_plugin.extra_family_spec_fields
        and "compliant_surface_contacts" in raw_family_spec
    ):
        contacts = raw_family_spec["compliant_surface_contacts"]
        if not isinstance(contacts, list):
            raise _fail(
                "COMPLIANT_SURFACE_CONTACTS_INVALID",
                "compliant_surface_contacts must be a list",
        )
        payload["compliant_surface_contacts"] = deepcopy(contacts)
    if (
        preserve_project_adapter_file
        and "project_adapter_file" in raw_family_spec
    ):
        payload["project_adapter_file"] = raw_family_spec[
            "project_adapter_file"
        ]
    from .family_plugins import reviewer_adapter_source

    binding = reviewer_adapter_source(spec_plugin.plugin_id)
    if binding is not None:
        payload["reviewer_adapter"] = binding
    return payload


def _assert_registered_family_fields(
    raw_family_spec: dict[str, Any],
    parsed_family_spec: Any,
    spec_plugin: Any,
) -> None:
    schema_version = raw_family_spec.get("schema_version")
    if schema_version is not None and schema_version != "0.1":
        raise _fail(
            "MECHANISM_FAMILY_SPEC_SCHEMA_UNSUPPORTED",
            "family spec schema_version must be 0.1 when provided",
        )
    recognized = (
        set(parsed_family_spec.to_dict())
        | {"mechanism_kind", "schema_version"}
        | set(spec_plugin.extra_family_spec_fields)
    )
    if "project_adapter_file" in raw_family_spec:
        recognized.add("project_adapter_file")
    unknown = sorted(set(raw_family_spec) - recognized)
    if unknown:
        raise _plan_fail(
            "MECHANISM_TEMPLATE_EXTENSION_REQUIRED",
            "family spec requests fields outside the registered template and "
            "requires human capability review: "
            + ", ".join(unknown),
        )


def _parse_registered_axis_family_spec(
    raw_family_spec: dict[str, Any],
    spec_plugin: Any,
) -> tuple[dict[str, Any], set[str], set[str]]:
    """Parse one reviewer-owned revolute/prismatic family specification.

    Axis families historically hard-coded ``axis_spring`` parsing even after
    a hash-bound Reviewer plugin had registered a more specific mechanism
    kind.  Keep the same plugin boundary and rigid-group checks used for
    compound families so the selected axis plugin owns its schema without
    weakening the immutable package binding.
    """

    from .family_plugins import plugin_registration_or_none

    plugin_id = str(getattr(spec_plugin, "plugin_id", ""))
    mechanism_kind = raw_family_spec.get("mechanism_kind")
    registration = plugin_registration_or_none(plugin_id)
    try:
        parsed_family_spec = spec_plugin.parse_family_spec(raw_family_spec)
    except MechanismValidationError as exc:
        boundary_error = plugin_boundary_error(
            exc,
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            callback="parse_family_spec",
            source_component="mechanism.discovery.parse_family_spec",
            details={
                "plugin_id": plugin_id,
                "mechanism_kind": mechanism_kind,
            },
        )
        if boundary_error is exc:
            raise
        raise boundary_error from exc
    except Exception as exc:
        message = f"mechanism adapter parse_family_spec() failed: {exc}"
        raise plugin_boundary_error(
            exc,
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            callback="parse_family_spec",
            source_component="mechanism.discovery.parse_family_spec",
            code="MECHANISM_ADAPTER_FAMILY_SPEC_PARSE_FAILED",
            message=message,
            details={
                "plugin_id": plugin_id,
                "mechanism_kind": mechanism_kind,
            },
        ) from exc
    try:
        _assert_registered_family_fields(
            raw_family_spec,
            parsed_family_spec,
            spec_plugin,
        )
        spec_fixed = {
            group_id
            for body in parsed_family_spec.bodies
            if body.fixed
            for group_id in body.group_ids
        }
        spec_moving = {
            group_id
            for body in parsed_family_spec.bodies
            if not body.fixed
            for group_id in body.group_ids
        }
    except (AttributeError, TypeError) as exc:
        message = f"mechanism adapter returned an invalid family spec: {exc}"
        raise plugin_boundary_error(
            exc,
            registration=registration,
            plugin_id=plugin_id,
            plugin_kind="mechanism",
            callback="parse_family_spec",
            source_component="mechanism.discovery.parse_family_spec",
            code="MECHANISM_ADAPTER_FAMILY_SPEC_RESULT_INVALID",
            message=message,
            details={
                "plugin_id": plugin_id,
                "mechanism_kind": mechanism_kind,
            },
            contract_violation=True,
        ) from exc
    return (
        _serialized_family_spec(
            raw_family_spec,
            parsed_family_spec,
            spec_plugin,
        ),
        spec_fixed,
        spec_moving,
    )


def _artifact_inventory(
    package_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, str]:
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        raise _fail("ARTIFACT_INVENTORY_MISSING", "package has no artifacts")
    actual: dict[str, str] = {}
    folded: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise _fail(
                "ARTIFACT_ENTRY_INVALID",
                f"artifacts[{index}] must contain only path and sha256",
            )
        relative = entry["path"]
        expected = entry["sha256"]
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise _fail(
                "ARTIFACT_ENTRY_INVALID",
                f"artifacts[{index}] has invalid values",
            )
        path = ensure_inside(package_dir, relative)
        if not path.is_file() or path.is_symlink():
            raise _fail(
                "ARTIFACT_MISSING",
                f"artifact is missing or unsafe: {relative}",
            )
        key = relative.casefold()
        if key in folded:
            raise _fail(
                "ARTIFACT_PATH_DUPLICATE",
                f"artifact path is duplicated or case-colliding: {relative}",
            )
        folded.add(key)
        digest = sha256_file(path)
        if digest.casefold() != expected.casefold():
            raise _fail(
                "ARTIFACT_CHANGED_AFTER_INTAKE",
                f"artifact digest does not match package.json: {relative}",
            )
        actual[relative] = digest
    return dict(sorted(actual.items()))


def _validate_intake_report(
    package_dir: Path,
    report_path: Path,
    *,
    approved_project_adapters: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    report = read_json(report_path)
    if report.get("status") not in _ACCEPTED_STATUSES:
        raise _fail(
            "INTAKE_NOT_ACCEPTED",
            "package intake status is not accepted",
        )
    reported_package = report.get("package_dir")
    if not isinstance(reported_package, str):
        raise _fail("INTAKE_PACKAGE_MISSING", "intake report has no package_dir")
    if Path(reported_package).resolve() != package_dir:
        raise _fail(
            "INTAKE_PACKAGE_MISMATCH",
            "intake report belongs to a different package directory",
        )
    capabilities = report.get("capabilities")
    if not isinstance(capabilities, dict):
        raise _fail(
            "INTAKE_CAPABILITIES_MISSING",
            "intake report has no capability decision",
        )
    if capabilities.get("can_enter_physical_preflight") is not True:
        blockers = capabilities.get("physical_preflight_blockers", [])
        detail = ", ".join(str(item) for item in blockers) or (
            "the intake report lists no physical_preflight_blockers; read "
            f"{report_path} at capabilities.can_enter_physical_preflight for the recorded "
            "decision"
        )
        raise _fail(
            "INTAKE_PREFLIGHT_BLOCKED",
            "mechanism validation cannot enter physical preflight because "
            f"capabilities.can_enter_physical_preflight=false in {report_path}; "
            f"blockers: {detail}. Fix the named package/CAD issue and resubmit.",
        )
    if capabilities.get("can_attempt_reduced_certification") is not True:
        capsules = capabilities.get("capsules_requiring_contact_model_only", [])
        if capsules and not approved_project_adapters:
            raise _plan_fail(
                "INTAKE_REDUCED_CAPABILITY_MISSING",
                "physical preflight is available, but these capsules lack an "
                "executable reduced/contact model: "
                + ", ".join(str(item) for item in capsules)
                + f". See {report_path} at "
                "capabilities.capsules_requiring_contact_model_only. Return "
                "to plan revision and submit the missing project mechanism "
                "adapter for resource approval.",
            )
        if not capsules:
            raise _fail(
                "INTAKE_CAPABILITY_BLOCKED",
                "capabilities.can_attempt_reduced_certification=false in "
                f"{report_path}, but no missing contact-model capsule was named; "
                "fix the intake package information and resubmit.",
            )
    return report


def _group_extents_from_contract(
    contract: dict[str, Any],
    group_ids: set[str],
) -> tuple[float, float, float]:
    minima = [math.inf, math.inf, math.inf]
    maxima = [-math.inf, -math.inf, -math.inf]
    matched = 0
    for part in contract.get("parts", []):
        if not isinstance(part, dict) or part.get("rigid_group_id") not in group_ids:
            continue
        bounds = part.get("declared_bounds_m")
        if not isinstance(bounds, dict):
            continue
        lower = bounds.get("min")
        upper = bounds.get("max")
        if (
            not isinstance(lower, list)
            or not isinstance(upper, list)
            or len(lower) != 3
            or len(upper) != 3
        ):
            continue
        for index in range(3):
            minima[index] = min(minima[index], float(lower[index]))
            maxima[index] = max(maxima[index], float(upper[index]))
        matched += 1
    if matched == 0:
        raise _fail(
            "CHARACTERISTIC_LENGTH_UNAVAILABLE",
            "cannot derive characteristic length from assembly contract",
        )
    extents = tuple(maxima[index] - minima[index] for index in range(3))
    if any(not math.isfinite(item) or item <= 0 for item in extents):
        raise _fail(
            "GROUP_EXTENTS_INVALID",
            "derived rigid-group extents are invalid",
        )
    return extents


def _axis_length_from_contract(
    contract: dict[str, Any],
    group_ids: set[str],
    axis: tuple[float, float, float],
) -> float:
    extents = _group_extents_from_contract(contract, group_ids)
    length = sum(abs(axis[index]) * extents[index] for index in range(3))
    if not math.isfinite(length) or length <= 0:
        raise _fail(
            "CHARACTERISTIC_LENGTH_INVALID",
            "derived characteristic length is invalid",
        )
    return length


def _frames(contract: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for frame in contract.get("frames", []):
        if not isinstance(frame, dict) or not isinstance(frame.get("frame_id"), str):
            continue
        frame_id = frame["frame_id"]
        if frame_id in result:
            raise _fail("FRAME_ID_DUPLICATE", f"duplicate frame: {frame_id}")
        result[frame_id] = frame
    return result


def _frame_origin(
    frame_inventory: dict[str, dict[str, Any]],
    frame_id: str,
) -> tuple[float, float, float]:
    frame = frame_inventory.get(frame_id)
    if frame is None:
        raise _fail("FRAME_NOT_FOUND", f"frame is missing: {frame_id}")
    try:
        return vector3(
            frame.get("origin_m"),
            f"frames.{frame_id}.origin_m",
            allow_zero=True,
        )
    except MechanismValidationError as exc:
        raise _fail("FRAME_ORIGIN_INVALID", str(exc)) from exc


def _find_port(capsule: dict[str, Any]) -> tuple[str, str]:
    for port in capsule.get("input_ports", []):
        if not isinstance(port, dict):
            continue
        if port.get("wrench_type") not in {"force", "torque"}:
            continue
        port_id = port.get("port_id")
        frame_id = port.get("frame_id")
        if isinstance(port_id, str) and isinstance(frame_id, str):
            return port_id, frame_id
    raise _fail("INPUT_PORT_UNUSABLE", "capsule has no usable wrench input")


def _find_observable(capsule: dict[str, Any]) -> tuple[str, str]:
    for observable in capsule.get("observables", []):
        if not isinstance(observable, dict):
            continue
        if observable.get("quantity") not in {"pose_6d", "position_3d"}:
            continue
        observable_id = observable.get("observable_id")
        frame_id = observable.get("frame_id")
        if isinstance(observable_id, str) and isinstance(frame_id, str):
            return observable_id, frame_id
    raise _fail("OBSERVABLE_UNUSABLE", "capsule has no usable pose observable")


def _mapped_tasks(
    mapping: dict[str, Any],
    capsule_id: str,
) -> tuple[str, ...]:
    task_ids: list[str] = []
    for entry in mapping.get("task_mappings", []):
        if not isinstance(entry, dict):
            continue
        capsules = entry.get("capsule_ids")
        task_id = entry.get("task_id")
        if (
            isinstance(capsules, list)
            and capsule_id in capsules
            and isinstance(task_id, str)
        ):
            task_ids.append(task_id)
    if not task_ids:
        raise _fail(
            "CAPSULE_TASK_MAPPING_MISSING",
            f"capsule {capsule_id!r} is not mapped to a machine task",
        )
    return tuple(sorted(set(task_ids)))


def _assert_report_matches_documents(
    report: dict[str, Any],
    manifest: dict[str, Any],
    contract: dict[str, Any],
    capsules: list[dict[str, Any]],
    mapping: dict[str, Any],
) -> None:
    summary = report.get("package_summary")
    if not isinstance(summary, dict):
        raise _fail("INTAKE_SUMMARY_MISSING", "intake report has no summary")
    expected = {
        "package_id": manifest.get("package_id"),
        "design_version": manifest.get("design_version"),
        "part_count": len(contract.get("parts", [])),
        "rigid_group_count": len(contract.get("rigid_groups", [])),
        "frame_count": len(contract.get("frames", [])),
        "capsule_count": len(capsules),
        "mapped_task_count": len(mapping.get("task_mappings", [])),
    }
    mismatches = [
        key for key, value in expected.items() if summary.get(key) != value
    ]
    if mismatches:
        raise _fail(
            "INTAKE_REPORT_STALE",
            "intake summary no longer matches package documents: "
            + ", ".join(mismatches),
        )


def discover_mechanisms(
    package_dir: str | Path,
    intake_report: str | Path,
    *,
    approved_project_adapters: Mapping[str, str] | None = None,
) -> tuple[MechanismPlan, ...]:
    package_root = Path(package_dir).resolve()
    report_path = Path(intake_report).resolve()
    if not package_root.is_dir() or package_root.is_symlink():
        raise _fail("PACKAGE_DIRECTORY_UNSAFE", "package directory is unsafe")
    report = _validate_intake_report(
        package_root,
        report_path,
        approved_project_adapters=approved_project_adapters,
    )
    manifest_path = package_root / "package.json"
    manifest = read_json(manifest_path)
    artifacts = _artifact_inventory(package_root, manifest)

    contract_relative = manifest.get("assembly_contract_file")
    mapping_relative = manifest.get("machine_task_mapping_file")
    capsule_files = manifest.get("mechanism_capsule_files")
    if not isinstance(contract_relative, str) or not isinstance(mapping_relative, str):
        raise _fail(
            "PACKAGE_POINTER_MISSING",
            "package manifest is missing contract or task mapping",
        )
    if not isinstance(capsule_files, list) or not capsule_files:
        raise _fail("CAPSULES_MISSING", "package has no mechanism capsules")
    contract = read_json(ensure_inside(package_root, contract_relative))
    mapping = read_json(ensure_inside(package_root, mapping_relative))
    capsule_documents = [
        read_json(ensure_inside(package_root, relative))
        for relative in capsule_files
        if isinstance(relative, str)
    ]
    if len(capsule_documents) != len(capsule_files):
        raise _fail("CAPSULE_POINTER_INVALID", "capsule paths must be strings")
    _assert_report_matches_documents(
        report,
        manifest,
        contract,
        capsule_documents,
        mapping,
    )

    package_id = manifest.get("package_id")
    design_version = manifest.get("design_version")
    requirements_ref = manifest.get("requirements_ref")
    if not isinstance(package_id, str) or not isinstance(design_version, str):
        raise _fail("PACKAGE_ID_MISSING", "package identity is incomplete")
    requirements_sha256 = (
        requirements_ref.get("sha256")
        if isinstance(requirements_ref, dict)
        else None
    )
    if not isinstance(requirements_sha256, str):
        raise _fail(
            "REQUIREMENTS_BINDING_MISSING",
            "package requirements hash is missing",
        )

    frame_inventory = _frames(contract)
    world = contract.get("world_frame", {})
    gravity = vector3(
        world.get("gravity_m_s2", [0.0, 0.0, -9.81]),
        "world_frame.gravity_m_s2",
        allow_zero=True,
    )
    report_sha256 = sha256_file(report_path)
    manifest_sha256 = sha256_file(manifest_path)
    binding = canonical_sha256(
        {
            "manifest_sha256": manifest_sha256,
            "artifact_hashes": artifacts,
            "requirements_ref": manifest.get("requirements_ref"),
        }
    )
    source_hashes = {
        path: digest
        for path, digest in artifacts.items()
        if path.casefold().startswith("source_components/")
    }

    plans: list[MechanismPlan] = []
    seen_capsules: set[str] = set()
    for relative, capsule in zip(capsule_files, capsule_documents, strict=True):
        capsule_id = capsule.get("capsule_id")
        if not isinstance(capsule_id, str) or capsule_id in seen_capsules:
            raise _fail("CAPSULE_ID_INVALID", "capsule IDs are missing or duplicated")
        seen_capsules.add(capsule_id)
        claim = capsule.get("reduced_model_claim")
        if not isinstance(claim, dict):
            raise _fail(
                "REDUCED_CLAIM_MISSING",
                f"capsule {capsule_id!r} has no reduced model claim",
            )
        if claim.get("verification_status") != "UNVERIFIED":
            raise _fail(
                "AI_CLAIM_PRECERTIFIED",
                "AI reduced-model claims must remain UNVERIFIED",
            )
        family = claim.get("family")
        if family not in _PROTOCOLS:
            raise _fail(
                "REDUCED_FAMILY_UNSUPPORTED",
                f"unsupported reduced-model family: {family!r}",
            )
        try:
            axis = vector3(claim.get("axis_abs"), "axis_abs", unit=True)
        except MechanismValidationError as exc:
            raise _fail(
                "REDUCED_CLAIM_AXIS_INVALID",
                f"capsule {capsule_id!r} has an invalid reduced-model axis: "
                f"{exc.message}",
            ) from exc
        fixed = claim.get("fixed_group_ids")
        moving = claim.get("moving_group_ids")
        joint_frame_id = claim.get("joint_frame_id")
        if (
            not isinstance(fixed, list)
            or not isinstance(moving, list)
            or not isinstance(joint_frame_id, str)
        ):
            raise _fail(
                "REDUCED_CLAIM_INVALID",
                f"capsule {capsule_id!r} has invalid rigid groups or frame",
            )
        input_port_id, input_frame_id = _find_port(capsule)
        observable_id, observable_frame_id = _find_observable(capsule)
        group_ids = set(fixed) | set(moving)
        characteristic_length = _axis_length_from_contract(
            contract,
            group_ids,
            axis,
        )
        moving_extents = _group_extents_from_contract(contract, set(moving))
        adapter = select_adapter(
            str(family),
            artifacts,
            requirements_sha256,
        )
        adapter_id = None if adapter is None else adapter.adapter_id
        summary = report.get("package_summary", {})
        generator_relative = manifest.get("source_cad_generator_file")
        assembly_relative = manifest.get("assembly_file")
        derived_adapter_ids = {
            "prismatic": DERIVED_PLAIN_PRISMATIC_ADAPTER_ID,
            "revolute": DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
            "compound": DERIVED_COMPOUND_ADAPTER_ID,
        }
        if (
            adapter_id is None
            and family in derived_adapter_ids
            and isinstance(generator_relative, str)
            and generator_relative in artifacts
            and isinstance(assembly_relative, str)
            and assembly_relative in artifacts
            and isinstance(summary, dict)
            and int(summary.get("step_solid_count", 0)) >= 2
        ):
            adapter = get_adapter(derived_adapter_ids[str(family)])
            adapter_id = adapter.adapter_id
        if family == "compound":
            family_spec_file = claim.get("family_spec_file")
            if not isinstance(family_spec_file, str) or family_spec_file not in artifacts:
                raise _fail(
                    "COMPOUND_SPEC_UNBOUND",
                    "compound family_spec_file must be an immutable package artifact",
                )
            raw_family_spec = read_json(ensure_inside(package_root, family_spec_file))
            mechanism_kind = raw_family_spec.get("mechanism_kind")
            from .family_plugins import (
                family_spec_plugin_for_kind,
                load_project_mechanism_adapter,
                plugin_registration_for,
            )

            project_adapter_file = raw_family_spec.get(
                "project_adapter_file"
            )
            # Registered Harness capability owns the mechanism kind.  A stale
            # project_adapter_file in a frozen package is migration metadata,
            # not authority to replace that installed/reviewer plugin.
            spec_plugin = family_spec_plugin_for_kind(mechanism_kind)
            preserve_project_adapter_file = False
            if spec_plugin is None and project_adapter_file is not None:
                approved = approved_project_adapters or {}
                if (
                    not isinstance(project_adapter_file, str)
                    or approved.get(project_adapter_file)
                    != artifacts.get(project_adapter_file)
                ):
                    raise _plan_fail(
                        "PROJECT_MECHANISM_ADAPTER_NOT_APPROVED",
                        "project_adapter_file must be the exact Python "
                        "resource approved in the current task plan: "
                        f"{project_adapter_file!r}",
                    )
                spec_plugin = load_project_mechanism_adapter(
                    package_root,
                    project_adapter_file,
                    approved[project_adapter_file],
                )
                preserve_project_adapter_file = True
            if spec_plugin is None:
                raise _plan_fail(
                    "MECHANISM_KIND_UNSUPPORTED",
                    "compound mechanism_kind is not registered in the trusted "
                    f"template library: {mechanism_kind!r}",
                )
            plugin_id = str(getattr(spec_plugin, "plugin_id", ""))
            plugin_family_spec = (
                raw_family_spec
                if preserve_project_adapter_file
                else {
                    key: value
                    for key, value in raw_family_spec.items()
                    if key != "project_adapter_file"
                }
            )
            try:
                plugin_registration = plugin_registration_for(plugin_id)
            except MechanismValidationError:
                plugin_registration = None
            try:
                parsed_family_spec = spec_plugin.parse_family_spec(
                    plugin_family_spec
                )
            except MechanismValidationError as exc:
                details = {
                    "plugin_id": plugin_id,
                    "mechanism_kind": mechanism_kind,
                }
                boundary_error = plugin_boundary_error(
                    exc,
                    registration=plugin_registration,
                    plugin_id=plugin_id,
                    plugin_kind="mechanism",
                    callback="parse_family_spec",
                    source_component="mechanism.discovery.parse_family_spec",
                    details=details,
                )
                if boundary_error is exc:
                    raise
                raise boundary_error from exc
            except Exception as exc:
                code = "MECHANISM_ADAPTER_FAMILY_SPEC_PARSE_FAILED"
                message = f"mechanism adapter parse_family_spec() failed: {exc}"
                details = {
                    "plugin_id": plugin_id,
                    "mechanism_kind": mechanism_kind,
                }
                raise plugin_boundary_error(
                    exc,
                    registration=plugin_registration,
                    plugin_id=plugin_id,
                    plugin_kind="mechanism",
                    callback="parse_family_spec",
                    source_component="mechanism.discovery.parse_family_spec",
                    code=code,
                    message=message,
                    details=details,
                ) from exc
            try:
                _assert_registered_family_fields(
                    raw_family_spec,
                    parsed_family_spec,
                    spec_plugin,
                )
                spec_fixed = {
                    group_id for body in parsed_family_spec.bodies if body.fixed
                    for group_id in body.group_ids
                }
                spec_moving = {
                    group_id for body in parsed_family_spec.bodies if not body.fixed
                    for group_id in body.group_ids
                }
            except (AttributeError, TypeError) as exc:
                code = "MECHANISM_ADAPTER_FAMILY_SPEC_RESULT_INVALID"
                message = f"mechanism adapter returned an invalid family spec: {exc}"
                details = {
                    "plugin_id": plugin_id,
                    "mechanism_kind": mechanism_kind,
                }
                raise plugin_boundary_error(
                    exc,
                    registration=plugin_registration,
                    plugin_id=plugin_id,
                    plugin_kind="mechanism",
                    callback="parse_family_spec",
                    source_component="mechanism.discovery.parse_family_spec",
                    code=code,
                    message=message,
                    details=details,
                    contract_violation=True,
                ) from exc
            if spec_fixed != set(fixed) or spec_moving != set(moving):
                raise _fail(
                    "COMPOUND_SPEC_GROUP_MISMATCH",
                    "compound family spec groups differ from the reduced-model claim",
                )
            if adapter is None:
                adapter = get_adapter(DERIVED_COMPOUND_ADAPTER_ID)
                adapter_id = adapter.adapter_id
            adapter = replace(
                adapter,
                family_spec=_serialized_family_spec(
                    raw_family_spec,
                    parsed_family_spec,
                    spec_plugin,
                    preserve_project_adapter_file=(
                        preserve_project_adapter_file
                    ),
                ),
            )
        elif family in {"revolute", "prismatic"}:
            family_spec_file = claim.get("family_spec_file")
            inline_family_spec = claim.get("family_spec")
            if family_spec_file is not None and inline_family_spec is not None:
                raise _fail(
                    "AXIS_SPEC_BINDING_AMBIGUOUS",
                    "axis reduced-model claim cannot bind both "
                    "family_spec_file and inline family_spec",
                )
            if family_spec_file is not None:
                if (
                    not isinstance(family_spec_file, str)
                    or family_spec_file not in artifacts
                ):
                    raise _fail(
                        "AXIS_SPRING_SPEC_UNBOUND",
                        "axis family_spec_file must be an immutable package artifact",
                    )
                raw_family_spec = read_json(
                    ensure_inside(package_root, family_spec_file)
                )
                if adapter is None:
                    adapter = get_adapter(derived_adapter_ids[str(family)])
                    adapter_id = adapter.adapter_id
                from .family_plugins import family_spec_plugin_for_kind

                mechanism_kind = raw_family_spec.get("mechanism_kind")
                spec_plugin = family_spec_plugin_for_kind(mechanism_kind)
                if spec_plugin is None:
                    from .axis_spring_spec import axis_spring_from_dict

                    parsed_family_spec = axis_spring_from_dict(
                        raw_family_spec,
                        expected_family=str(family),
                    )
                    serialized_family_spec = parsed_family_spec.to_dict()
                else:
                    if str(family) not in spec_plugin.families:
                        raise _plan_fail(
                            "MECHANISM_KIND_FAMILY_MISMATCH",
                            f"registered mechanism_kind {mechanism_kind!r} "
                            f"does not support axis family {family!r}",
                        )
                    (
                        serialized_family_spec,
                        spec_fixed,
                        spec_moving,
                    ) = _parse_registered_axis_family_spec(
                        raw_family_spec,
                        spec_plugin,
                    )
                    if spec_fixed != set(fixed) or spec_moving != set(moving):
                        raise _fail(
                            "AXIS_SPEC_GROUP_MISMATCH",
                            "axis family spec groups differ from the "
                            "reduced-model claim",
                        )
                adapter = replace(
                    adapter,
                    family_spec=serialized_family_spec,
                )
            elif inline_family_spec is not None:
                if not isinstance(inline_family_spec, dict):
                    raise _fail(
                        "AXIS_SPEC_INVALID",
                        "inline axis family_spec must be an object",
                    )
                raw_family_spec = deepcopy(inline_family_spec)
                claim_kind = claim.get("mechanism_kind")
                inline_kind = raw_family_spec.get("mechanism_kind")
                if (
                    claim_kind is not None
                    and inline_kind is not None
                    and claim_kind != inline_kind
                ):
                    raise _fail(
                        "AXIS_SPEC_KIND_MISMATCH",
                        "inline family_spec mechanism_kind differs from the "
                        "reduced-model claim",
                    )
                mechanism_kind = inline_kind or claim_kind
                if not isinstance(mechanism_kind, str) or not mechanism_kind:
                    raise _fail(
                        "AXIS_SPEC_KIND_MISSING",
                        "inline axis family_spec must identify mechanism_kind",
                    )
                raw_family_spec["mechanism_kind"] = mechanism_kind
                from .family_plugins import family_spec_plugin_for_kind

                spec_plugin = family_spec_plugin_for_kind(mechanism_kind)
                if spec_plugin is None:
                    raise _plan_fail(
                        "MECHANISM_KIND_UNSUPPORTED",
                        "inline axis mechanism_kind is not registered in the "
                        f"trusted template library: {mechanism_kind!r}",
                    )
                if str(family) not in spec_plugin.families:
                    raise _plan_fail(
                        "MECHANISM_KIND_FAMILY_MISMATCH",
                        f"registered mechanism_kind {mechanism_kind!r} "
                        f"does not support axis family {family!r}",
                    )
                (
                    serialized_family_spec,
                    spec_fixed,
                    spec_moving,
                ) = _parse_registered_axis_family_spec(
                    raw_family_spec,
                    spec_plugin,
                )
                if spec_fixed != set(fixed) or spec_moving != set(moving):
                    raise _fail(
                        "AXIS_SPEC_GROUP_MISMATCH",
                        "inline axis family spec groups differ from the "
                        "reduced-model claim",
                    )
                if adapter is None:
                    adapter = get_adapter(derived_adapter_ids[str(family)])
                    adapter_id = adapter.adapter_id
                adapter = replace(
                    adapter,
                    family_spec=serialized_family_spec,
                )
        adapter_spec = None if adapter is None else adapter_to_dict(adapter)
        task_ids = _mapped_tasks(mapping, capsule_id)
        task_contract = None
        task_semantic_binding = None
        acceptance_summary = report.get("package_summary", {}).get(
            "acceptance_contract"
        )
        if isinstance(acceptance_summary, dict):
            full_contract = acceptance_summary.get("contract")
            full_binding = acceptance_summary.get("semantic_binding")
            if isinstance(full_contract, dict) and isinstance(full_binding, dict):
                try:
                    verify_acceptance_contract(full_contract)
                    verify_semantic_binding(full_binding)
                except ValueError as exc:
                    raise _fail(
                        str(
                            getattr(
                                exc,
                                "code",
                                "ACCEPTANCE_AUTHORITY_INVALID",
                            )
                        ),
                        str(exc),
                    ) from exc
                requirements_path_value = report.get("requirements_file")
                if not isinstance(requirements_path_value, str):
                    raise _fail(
                        "ACCEPTANCE_REQUIREMENTS_SOURCE_MISSING",
                        "typed acceptance authority has no external requirements source",
                    )
                requirements_path = Path(
                    requirements_path_value
                ).resolve()
                if (
                    not requirements_path.is_file()
                    or requirements_path.is_symlink()
                    or sha256_file(requirements_path) != requirements_sha256
                ):
                    raise _fail(
                        "ACCEPTANCE_REQUIREMENTS_SOURCE_CHANGED",
                        "external requirements changed after intake",
                    )
                try:
                    fresh_contract = compile_acceptance_contract(
                        read_json(requirements_path)
                    )
                    fresh_binding = bind_acceptance_contract(
                        fresh_contract,
                        task_mapping=mapping,
                        capsules={
                            str(item.get("capsule_id") or ""): item
                            for item in capsule_documents
                        },
                        assembly=contract,
                    )
                except ValueError as exc:
                    raise _fail(
                        str(
                            getattr(
                                exc,
                                "code",
                                "ACCEPTANCE_RECOMPILE_FAILED",
                            )
                        ),
                        str(exc),
                    ) from exc
                if fresh_contract != full_contract:
                    raise _fail(
                        "ACCEPTANCE_CONTRACT_SOURCE_MISMATCH",
                        "saved acceptance contract differs from external requirements",
                    )
                if fresh_binding != full_binding:
                    raise _fail(
                        "SEMANTIC_BINDING_SOURCE_MISMATCH",
                        "saved task binding differs from package semantics",
                    )
                task_contract = contract_for_tasks(full_contract, set(task_ids))
                task_semantic_binding = binding_for_tasks(
                    full_binding, set(task_ids)
                )
                try:
                    verify_task_contract(task_contract)
                    verify_task_binding(task_semantic_binding)
                except ValueError as exc:
                    raise _fail(
                        str(
                            getattr(
                                exc,
                                "code",
                                "TASK_ACCEPTANCE_AUTHORITY_INVALID",
                            )
                        ),
                        str(exc),
                    ) from exc
        plan = MechanismPlan(
                package_dir=str(package_root),
                intake_report=str(report_path),
                intake_report_sha256=report_sha256,
                package_id=package_id,
                design_version=design_version,
                requirements_sha256=requirements_sha256,
                package_binding_sha256=binding,
                capsule_id=capsule_id,
                capsule_file=relative,
                family=str(family),
                protocol_id=_PROTOCOLS[str(family)],
                fixed_group_ids=tuple(fixed),
                moving_group_ids=tuple(moving),
                axis_abs=axis,
                joint_origin_abs_m=_frame_origin(
                    frame_inventory,
                    joint_frame_id,
                ),
                gravity_abs_m_s2=gravity,
                joint_frame_id=joint_frame_id,
                input_port_id=input_port_id,
                input_frame_id=input_frame_id,
                input_point_abs_m=_frame_origin(
                    frame_inventory,
                    input_frame_id,
                ),
                observable_id=observable_id,
                observable_frame_id=observable_frame_id,
                observable_point_abs_m=_frame_origin(
                    frame_inventory,
                    observable_frame_id,
                ),
                task_ids=task_ids,
                characteristic_length_m=characteristic_length,
                moving_extents_m=moving_extents,
                adapter_id=adapter_id,
                adapter_spec=adapter_spec,
                acceptance_contract=task_contract,
                semantic_binding=task_semantic_binding,
                artifact_hashes=artifacts,
                source_component_hashes=source_hashes,
            )
        if task_contract is not None and task_semantic_binding is not None:
            plan = replace(
                plan,
                task_channel_registry=build_task_channel_registry(
                    plan,
                    capsule,
                ),
            )
        plans.append(
            plan
        )
    return tuple(plans)


def source_component_geometry_identities(
    plan: MechanismPlan,
) -> dict[str, Any]:
    """Return geometry-only identities for source-component CAD inputs."""

    from .joint_realization import step_solid_geometry_sha256s

    package = Path(plan.package_dir).resolve()
    identities: dict[str, Any] = {}
    for relative in sorted(plan.source_component_hashes):
        path = ensure_inside(package, relative)
        suffix = path.suffix.casefold()
        if suffix in {".step", ".stp"}:
            identities[relative] = {
                "solid_geometry_sha256s": step_solid_geometry_sha256s(path)
            }
        elif suffix in {".obj", ".stl"}:
            import trimesh

            mesh = trimesh.load_mesh(path, process=False)
            triangles: list[tuple[tuple[float, float, float], ...]] = []
            for face in mesh.faces:
                vertices = tuple(
                    tuple(
                        0.0 if float(value) == 0.0 else float(value)
                        for value in mesh.vertices[index]
                    )
                    for index in face
                )
                triangles.append(
                    min(
                        vertices,
                        vertices[1:] + vertices[:1],
                        vertices[2:] + vertices[:2],
                    )
                )
            identities[relative] = {
                "mesh_geometry_sha256": canonical_sha256(
                    sorted(triangles)
                )
            }
    return identities
