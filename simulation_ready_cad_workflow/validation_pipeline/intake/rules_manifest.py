from __future__ import annotations

import os
import re
import struct
import zipfile
from pathlib import Path
from typing import Any

from ..contracts.input_schemas import (
    DESIGN_PACKAGE_INPUT_SCHEMA_ID,
    REQUIREMENTS_INPUT_SCHEMA_ID,
    SchemaCheckError,
    check_design_package_manifest,
    check_requirements_input,
)
from .context import (
    ID_PATTERN,
    SHA256_PATTERN,
    ValidationContext,
    _is_path_junction,
)
from .step_geometry import inspect_step_geometry_isolated as inspect_step_geometry


STEP_ENTITY_PATTERN = re.compile(r"#[0-9]+\s*=")
STEP_GEOMETRY_MARKERS = {
    "ADVANCED_BREP_SHAPE_REPRESENTATION",
    "BREP_WITH_VOIDS",
    "FACETED_BREP",
    "MANIFOLD_SOLID_BREP",
    "SHAPE_REPRESENTATION",
    "TESSELLATED_SOLID",
}
NATIVE_CAD_EXTENSIONS = {
    ".asm",
    ".catpart",
    ".catproduct",
    ".f3d",
    ".fcstd",
    ".iam",
    ".ipt",
    ".par",
    ".prt",
    ".psm",
    ".sldasm",
    ".sldprt",
}
CAD_GENERATOR_EXTENSIONS = {".py", ".scad"}
MAX_PACKAGE_FILES = 20_000
MAX_PACKAGE_ENTRIES = 25_000
MAX_PACKAGE_BYTES = 10 * 1024**3
MAX_STEP_BYTES = 512 * 1024**2
STEP_SCAN_CHUNK_BYTES = 1024 * 1024
STEP_SCAN_OVERLAP_CHARS = 4096
MAX_CAPSULE_FILES = 500
MAX_ARTIFACT_ENTRIES = 20_000
MAX_FCSTD_ENTRIES = 10_000
MAX_FCSTD_CENTRAL_DIRECTORY_BYTES = 64 * 1024**2
MAX_FCSTD_UNCOMPRESSED_BYTES = 2 * 1024**3
ZIP_EOCD_SIGNATURE = b"PK\x05\x06"
ZIP_EOCD_FIXED_BYTES = 22
ZIP_MAX_COMMENT_BYTES = 65_535
MAX_FCSTD_COMPRESSION_RATIO = 1_000.0


def _inventory_package(ctx: ValidationContext) -> None:
    stack = [ctx.package_dir]
    file_count = 0
    entry_count = 0
    total_bytes = 0
    symlink_count = 0
    junction_count = 0
    hardlink_count = 0
    folded_paths: dict[str, str] = {}
    while stack:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            ctx.error(
                "PACKAGE_INVENTORY_READ_ERROR",
                "Could not enumerate a package directory.",
                file=ctx.relative_label(directory),
                actual=str(exc),
            )
            continue
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_PACKAGE_ENTRIES:
                    ctx.error(
                        "PACKAGE_ENTRY_COUNT_EXCEEDED",
                        "Package contains too many filesystem entries.",
                        expected=f"at most {MAX_PACKAGE_ENTRIES}",
                        actual=entry_count,
                        hint="Remove generated caches and split oversized designs.",
                    )
                    ctx.report.package_summary["package_inventory_truncated"] = True
                    ctx.report.package_summary["package_file_count"] = file_count
                    ctx.report.package_summary["package_total_bytes"] = total_bytes
                    return
                path = Path(entry.path)
                try:
                    if entry.is_symlink():
                        symlink_count += 1
                        if symlink_count <= 20:
                            ctx.error(
                                "PACKAGE_SYMLINK_FORBIDDEN",
                                "Design packages may not contain symbolic links.",
                                file=ctx.relative_label(path),
                                hint="Replace the link with a real package file.",
                            )
                        continue
                    if _is_path_junction(path):
                        junction_count += 1
                        if junction_count <= 20:
                            ctx.error(
                                "PACKAGE_JUNCTION_FORBIDDEN",
                                "Design packages may not contain directory junctions.",
                                file=ctx.relative_label(path),
                                hint=(
                                    "Replace the junction with a real package "
                                    "directory."
                                ),
                            )
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        ctx.package_directory_paths.add(
                            path.relative_to(ctx.package_dir).as_posix()
                        )
                        stack.append(path)
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        ctx.error(
                            "PACKAGE_SPECIAL_FILE_FORBIDDEN",
                            "Package contains a non-regular filesystem entry.",
                            file=ctx.relative_label(path),
                        )
                        continue
                    # DirEntry.stat() reports st_nlink=0 on some Windows Python
                    # builds, while Path.stat() returns the real hard-link count.
                    stat = path.stat(follow_symlinks=False)
                    size = stat.st_size
                except OSError as exc:
                    ctx.error(
                        "PACKAGE_INVENTORY_READ_ERROR",
                        "Could not inspect a package entry.",
                        file=ctx.relative_label(path),
                        actual=str(exc),
                    )
                    continue
                file_count += 1
                total_bytes += size
                normalized = path.relative_to(ctx.package_dir).as_posix()
                folded = normalized.casefold()
                previous_path = folded_paths.get(folded)
                if previous_path is not None and previous_path != normalized:
                    ctx.error(
                        "PACKAGE_PATH_CASE_COLLISION",
                        "Package contains case-colliding file paths.",
                        file=normalized,
                        expected=previous_path,
                        actual=normalized,
                    )
                else:
                    folded_paths[folded] = normalized
                ctx.package_file_paths.add(normalized)
                if getattr(stat, "st_nlink", 1) > 1:
                    hardlink_count += 1
                    ctx.error(
                        "PACKAGE_HARDLINK_FORBIDDEN",
                        "Design packages may not contain hard-linked files.",
                        file=normalized,
                        expected="self-contained regular file with one link",
                        actual={"link_count": stat.st_nlink},
                        hint=(
                            "Copy the file content into the package and "
                            "recompute its artifact hash."
                        ),
                    )
                if file_count > MAX_PACKAGE_FILES:
                    ctx.error(
                        "PACKAGE_FILE_COUNT_EXCEEDED",
                        "Package contains too many files.",
                        expected=f"at most {MAX_PACKAGE_FILES}",
                        actual=file_count,
                        hint="Remove generated caches and split oversized designs.",
                    )
                    ctx.report.package_summary["package_inventory_truncated"] = True
                    ctx.report.package_summary["package_file_count"] = file_count
                    ctx.report.package_summary["package_total_bytes"] = total_bytes
                    return
                if total_bytes > MAX_PACKAGE_BYTES:
                    ctx.error(
                        "PACKAGE_SIZE_EXCEEDED",
                        "Package exceeds the intake size budget.",
                        expected=f"at most {MAX_PACKAGE_BYTES} bytes",
                        actual=total_bytes,
                        hint="Remove derived outputs or split the design package.",
                    )
                    ctx.report.package_summary["package_inventory_truncated"] = True
                    ctx.report.package_summary["package_file_count"] = file_count
                    ctx.report.package_summary["package_total_bytes"] = total_bytes
                    return
    if symlink_count > 20:
        ctx.error(
            "PACKAGE_SYMLINK_LIMIT_REACHED",
            "Additional package symbolic links were omitted from the report.",
            expected=0,
            actual=symlink_count,
        )
    if junction_count > 20:
        ctx.error(
            "PACKAGE_JUNCTION_LIMIT_REACHED",
            "Additional package directory junctions were omitted from the report.",
            expected=0,
            actual=junction_count,
        )
    ctx.report.package_summary["package_file_count"] = file_count
    ctx.report.package_summary["package_total_bytes"] = total_bytes
    ctx.report.package_summary["package_symlink_count"] = symlink_count
    ctx.report.package_summary["package_junction_count"] = junction_count
    ctx.report.package_summary["package_hardlink_count"] = hardlink_count


def _manifest_value(
    ctx: ValidationContext,
    key: str,
    fallback: str,
) -> str:
    if ctx.manifest is None:
        return fallback
    value = ctx.manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        ctx.error(
            "MANIFEST_FIELD_MISSING",
            f"package.json must declare {key}.",
            file="package.json",
            json_path=f"$.{key}",
            expected="non-empty relative path",
            actual=value,
            hint=f"Set {key} to {fallback!r} when using the standard layout.",
        )
        return fallback
    return value


def load_documents(ctx: ValidationContext) -> None:
    if not ctx.package_dir.is_dir():
        ctx.error(
            "PACKAGE_DIRECTORY_MISSING",
            "Design package directory does not exist or is not a directory.",
            file=str(ctx.package_dir),
            expected="existing directory",
            actual="missing",
        )
        return

    _inventory_package(ctx)

    manifest_path = ctx.package_dir / "package.json"
    ctx.paths["manifest"] = manifest_path
    if manifest_path.is_symlink():
        return
    ctx.manifest = ctx.load_json(
        manifest_path,
        required=True,
        label="package manifest",
        missing_code="PACKAGE_MANIFEST_MISSING",
    )

    assembly_rel = _manifest_value(ctx, "assembly_file", "assembly.step")
    contract_rel = _manifest_value(
        ctx, "assembly_contract_file", "assembly_contract.json"
    )
    coverage_rel = _manifest_value(
        ctx, "mechanism_coverage_file", "mechanism_coverage.json"
    )
    task_rel = _manifest_value(
        ctx, "machine_task_mapping_file", "machine_task_mapping.json"
    )
    ctx.paths["assembly"] = ctx.safe_package_path(
        assembly_rel, manifest_field="assembly_file"
    )
    ctx.paths["assembly_contract"] = ctx.safe_package_path(
        contract_rel, manifest_field="assembly_contract_file"
    )
    ctx.paths["coverage"] = ctx.safe_package_path(
        coverage_rel, manifest_field="mechanism_coverage_file"
    )
    ctx.paths["task_mapping"] = ctx.safe_package_path(
        task_rel, manifest_field="machine_task_mapping_file"
    )

    native_rel = (
        ctx.manifest.get("source_native_assembly_file")
        if ctx.manifest is not None
        else None
    )
    if isinstance(native_rel, str) and native_rel.strip():
        ctx.paths["native_assembly"] = ctx.safe_package_path(
            native_rel, manifest_field="source_native_assembly_file"
        )
    else:
        ctx.paths["native_assembly"] = None
        ctx.warning(
            "NATIVE_ASSEMBLY_MISSING",
            "No native CAD assembly was provided; STEP-only recovery may lose mate metadata.",
            file="package.json",
            json_path="$.source_native_assembly_file",
            hint="Add the native assembly when the CAD system supports it.",
        )

    generator_rel = (
        ctx.manifest.get("source_cad_generator_file")
        if ctx.manifest is not None
        else None
    )
    if isinstance(generator_rel, str) and generator_rel.strip():
        ctx.paths["cad_generator"] = ctx.safe_package_path(
            generator_rel, manifest_field="source_cad_generator_file"
        )
    else:
        ctx.paths["cad_generator"] = None

    capsule_values: list[Any] = []
    allow_capsule_discovery = True
    if ctx.manifest is not None:
        value = ctx.manifest.get("mechanism_capsule_files")
        if isinstance(value, list) and value:
            if len(value) > MAX_CAPSULE_FILES:
                ctx.error(
                    "CAPSULE_FILE_COUNT_EXCEEDED",
                    "package.json lists too many mechanism capsule files.",
                    file="package.json",
                    json_path="$.mechanism_capsule_files",
                    expected=f"at most {MAX_CAPSULE_FILES} entries",
                    actual=len(value),
                    hint="Split the machine into smaller design packages.",
                )
                allow_capsule_discovery = False
            else:
                capsule_values = value
        else:
            ctx.error(
                "CAPSULE_FILE_LIST_MISSING",
                "package.json must list at least one mechanism capsule file.",
                file="package.json",
                json_path="$.mechanism_capsule_files",
                expected="non-empty array of relative JSON paths",
                actual=value,
            )
    if not capsule_values and allow_capsule_discovery:
        discovered = sorted(
            (ctx.package_dir / "mechanism_capsules").glob("*.json")
        )
        if len(discovered) > MAX_CAPSULE_FILES:
            ctx.error(
                "CAPSULE_FILE_COUNT_EXCEEDED",
                "Discovered too many mechanism capsule files.",
                file="mechanism_capsules",
                expected=f"at most {MAX_CAPSULE_FILES} files",
                actual=len(discovered),
            )
        else:
            capsule_values = [
                path.relative_to(ctx.package_dir).as_posix()
                for path in discovered
            ]

    capsule_paths: list[Path] = []
    seen_paths: set[str] = set()
    for index, value in enumerate(capsule_values):
        if not isinstance(value, str) or not value.strip():
            ctx.error(
                "CAPSULE_FILE_PATH_INVALID",
                "Mechanism capsule file path must be a non-empty string.",
                file="package.json",
                json_path=f"$.mechanism_capsule_files[{index}]",
                actual=value,
            )
            continue
        path = ctx.safe_package_path(
            value, manifest_field=f"mechanism_capsule_files[{index}]"
        )
        if path is None:
            continue
        key = str(path).casefold()
        if key in seen_paths:
            ctx.error(
                "CAPSULE_FILE_DUPLICATE",
                "Mechanism capsule file is listed more than once.",
                file="package.json",
                json_path=f"$.mechanism_capsule_files[{index}]",
                actual=value,
            )
            continue
        seen_paths.add(key)
        capsule_paths.append(path)
    ctx.paths["capsules"] = capsule_paths  # type: ignore[assignment]

    ctx.assembly = ctx.load_json(
        ctx.paths["assembly_contract"],
        required=True,
        label="assembly contract",
        missing_code="ASSEMBLY_CONTRACT_MISSING",
    )
    ctx.coverage = ctx.load_json(
        ctx.paths["coverage"],
        required=True,
        label="mechanism coverage",
        missing_code="MECHANISM_COVERAGE_MISSING",
    )
    ctx.task_mapping = ctx.load_json(
        ctx.paths["task_mapping"],
        required=True,
        label="machine task mapping",
        missing_code="MACHINE_TASK_MAPPING_MISSING",
    )

    if ctx.requirements_path is None:
        ctx.error(
            "REQUIREMENTS_FILE_REQUIRED",
            "An immutable external requirements file is required for intake.",
            expected="requirements_path argument",
            actual=None,
            hint="Pass --requirements or the requirements_path API argument.",
        )
    else:
        try:
            ctx.requirements_path.relative_to(ctx.package_dir)
        except ValueError:
            pass
        else:
            ctx.error(
                "REQUIREMENTS_FILE_NOT_EXTERNAL",
                "Immutable requirements must be stored outside the AI package.",
                file=str(ctx.requirements_path),
                expected="path outside package_dir",
                actual="inside package_dir",
                hint="Move the original requirements file outside the package.",
            )
            # The design package is controlled by the authoring AI.  Once the
            # alleged immutable requirements resolve inside that package, do
            # not parse or derive any downstream task state from those bytes.
            return
        ctx.requirements = ctx.load_json(
            ctx.requirements_path,
            required=True,
            label="external requirements",
            missing_code="REQUIREMENTS_FILE_MISSING",
        )


def validate_manifest(ctx: ValidationContext) -> None:
    document = ctx.manifest
    if document is None:
        return
    try:
        check_design_package_manifest(document)
    except SchemaCheckError as exc:
        ctx.error(
            "DESIGN_PACKAGE_INPUT_SCHEMA_VIOLATION",
            "package.json violates the formal design package input schema.",
            file="package.json",
            expected=DESIGN_PACKAGE_INPUT_SCHEMA_ID,
            actual=exc.errors,
            hint=(
                "Fix the manifest structure to match "
                "design_package_input.v1.schema.json."
            ),
        )
    ctx.check_schema_version(document, "package.json")
    ctx.scan_for_forbidden_authority(document, file="package.json")
    ctx.warn_unknown_fields(
        document,
        {
            "schema_version",
            "package_id",
            "design_version",
            "created_by_ai",
            "assembly_file",
            "assembly_contract_file",
            "mechanism_coverage_file",
            "machine_task_mapping_file",
            "mechanism_capsule_files",
            "requirements_ref",
            "artifacts",
            "source_native_assembly_file",
            "source_cad_generator_file",
            "single_source_cad",
        },
        file="package.json",
    )
    ctx.require_id(document, "package_id", file="package.json")
    ctx.require_string(document, "design_version", file="package.json")
    if document.get("created_by_ai") is not True:
        ctx.error(
            "CREATED_BY_AI_FLAG_INVALID",
            "package.json must explicitly identify the package as AI generated.",
            file="package.json",
            json_path="$.created_by_ai",
            expected=True,
            actual=document.get("created_by_ai"),
        )
    single_source = document.get("single_source_cad")
    if single_source is not None:
        if not isinstance(single_source, dict):
            ctx.error(
                "SINGLE_SOURCE_CAD_INVALID",
                "single_source_cad must be an object.",
                file="package.json",
                json_path="$.single_source_cad",
                expected={
                    "policy": "required_v1",
                    "authoritative_artifact": document.get("assembly_file"),
                    "derived_assets": "hash_bound_and_regenerated",
                },
                actual=single_source,
            )
        else:
            ctx.warn_unknown_fields(
                single_source,
                {"policy", "authoritative_artifact", "derived_assets"},
                file="package.json",
                json_path="$.single_source_cad",
            )
            if single_source.get("policy") != "required_v1":
                ctx.error(
                    "SINGLE_SOURCE_CAD_POLICY_INVALID",
                    "single_source_cad must use the supported required_v1 policy.",
                    file="package.json",
                    json_path="$.single_source_cad.policy",
                    expected="required_v1",
                    actual=single_source.get("policy"),
                )
            if single_source.get("authoritative_artifact") != document.get("assembly_file"):
                ctx.error(
                    "SINGLE_SOURCE_CAD_AUTHORITY_INVALID",
                    "the authoritative CAD must be the package assembly artifact.",
                    file="package.json",
                    json_path="$.single_source_cad.authoritative_artifact",
                    expected=document.get("assembly_file"),
                    actual=single_source.get("authoritative_artifact"),
                )
            if single_source.get("derived_assets") != "hash_bound_and_regenerated":
                ctx.error(
                    "SINGLE_SOURCE_CAD_DERIVATION_INVALID",
                    "derived render and physics assets must be hash-bound and regenerated.",
                    file="package.json",
                    json_path="$.single_source_cad.derived_assets",
                    expected="hash_bound_and_regenerated",
                    actual=single_source.get("derived_assets"),
                )
    requirements_ref = document.get("requirements_ref")
    if not isinstance(requirements_ref, dict):
        ctx.error(
            "REQUIREMENTS_REF_MISSING",
            "package.json must bind the design to immutable external requirements.",
            file="package.json",
            json_path="$.requirements_ref",
            expected={"requirements_id": "...", "sha256": "64 hex characters"},
            actual=requirements_ref,
        )
    else:
        ctx.warn_unknown_fields(
            requirements_ref,
            {"requirements_id", "sha256"},
            file="package.json",
            json_path="$.requirements_ref",
        )
        ctx.require_id(
            requirements_ref,
            "requirements_id",
            file="package.json",
            base_path="$.requirements_ref",
        )
        digest = requirements_ref.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            ctx.error(
                "REQUIREMENTS_HASH_INVALID",
                "requirements_ref.sha256 must be a SHA-256 digest.",
                file="package.json",
                json_path="$.requirements_ref.sha256",
                expected="64 hexadecimal characters",
                actual=digest,
            )
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        ctx.error(
            "ARTIFACT_MANIFEST_MISSING",
            "package.json must list package artifacts and SHA-256 hashes.",
            file="package.json",
            json_path="$.artifacts",
            expected="non-empty array",
            actual=artifacts,
        )
    elif len(artifacts) > MAX_ARTIFACT_ENTRIES:
        ctx.error(
            "ARTIFACT_COUNT_EXCEEDED",
            "package.json lists too many artifacts.",
            file="package.json",
            json_path="$.artifacts",
            expected=f"at most {MAX_ARTIFACT_ENTRIES} entries",
            actual=len(artifacts),
            hint="Remove derived outputs or split the design package.",
        )


def validate_requirements(ctx: ValidationContext) -> None:
    document = ctx.requirements
    path = ctx.requirements_path
    if document is None or path is None:
        return
    try:
        check_requirements_input(document)
    except SchemaCheckError as exc:
        ctx.error(
            "REQUIREMENTS_INPUT_SCHEMA_VIOLATION",
            "The external requirements file violates the formal input schema.",
            file=str(path),
            expected=REQUIREMENTS_INPUT_SCHEMA_ID,
            actual=exc.errors,
            hint=(
                "Fix the requirements structure to match "
                "requirements_input.v1.schema.json; do not weaken the "
                "requirements themselves."
            ),
        )
    schema_version = document.get("schema_version")
    ctx.check_schema_version(
        document,
        path,
        supported=("0.1", "0.2"),
    )
    requirements_id = ctx.require_id(document, "requirements_id", file=path)
    tasks = ctx.require_list(document, "tasks", file=path, nonempty=True)
    if tasks is None:
        return
    seen: set[str] = set()
    for index, task in enumerate(tasks):
        base = f"$.tasks[{index}]"
        if not isinstance(task, dict):
            ctx.error(
                "TASK_ENTRY_INVALID",
                "Each external requirement task must be an object.",
                file=str(path),
                json_path=base,
                expected="object",
                actual=task,
            )
            continue
        task_id = ctx.require_id(task, "task_id", file=path, base_path=base)
        ctx.require_string(task, "description", file=path, base_path=base)
        if schema_version == "0.1":
            criteria = ctx.require_list(
                task,
                "success_criteria",
                file=path,
                base_path=base,
                nonempty=True,
                code="SUCCESS_CRITERIA_REQUIRED",
            )
            if criteria is not None:
                for criterion_index, criterion in enumerate(criteria):
                    if not isinstance(criterion, (str, dict)):
                        ctx.error(
                            "SUCCESS_CRITERION_INVALID",
                            "Each success criterion must be a string or object.",
                            file=str(path),
                            json_path=(
                                f"{base}.success_criteria[{criterion_index}]"
                            ),
                            expected="string or object",
                            actual=criterion,
                        )
        if task_id:
            if task_id in seen:
                ctx.error(
                    "TASK_ID_DUPLICATE",
                    "External requirement task_id must be unique.",
                    file=str(path),
                    json_path=f"{base}.task_id",
                    actual=task_id,
                )
            else:
                seen.add(task_id)
                ctx.requirement_task_ids.add(task_id)
    if requirements_id:
        ctx.report.package_summary["requirements_id"] = requirements_id


def _scan_step_structure(path: Path) -> tuple[int, set[str]]:
    entity_count = 0
    markers: set[str] = set()
    carry = ""
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(STEP_SCAN_CHUNK_BYTES)
            if not chunk:
                break
            old_carry_length = len(carry)
            upper = (carry + chunk.decode("ascii", errors="ignore")).upper()
            entity_count += sum(
                1
                for match in STEP_ENTITY_PATTERN.finditer(upper)
                if match.end() > old_carry_length
            )
            for marker in STEP_GEOMETRY_MARKERS:
                if marker in upper:
                    markers.add(marker)
            carry = upper[-STEP_SCAN_OVERLAP_CHARS:]
    return entity_count, markers


def validate_step_file(ctx: ValidationContext) -> None:
    path = ctx.paths.get("assembly")
    if not isinstance(path, Path):
        return
    if not ctx.require_file(
        path,
        code="ASSEMBLY_FILE_MISSING",
        label="STEP assembly",
        hint="Export the complete assembly to STEP and update package.json.",
    ):
        return
    ctx.required_artifact_paths.add(path.relative_to(ctx.package_dir).as_posix())
    if path.suffix.casefold() not in {".step", ".stp"}:
        ctx.error(
            "ASSEMBLY_EXTENSION_INVALID",
            "The primary assembly must be a STEP/STP file.",
            file=ctx.relative_label(path),
            expected=".step or .stp",
            actual=path.suffix,
        )
    try:
        size = path.stat().st_size
    except OSError as exc:
        ctx.error(
            "ASSEMBLY_READ_ERROR",
            "Could not inspect the STEP assembly.",
            file=ctx.relative_label(path),
            actual=str(exc),
        )
        return
    if size > MAX_STEP_BYTES:
        ctx.error(
            "ASSEMBLY_FILE_TOO_LARGE",
            "STEP assembly exceeds the safe kernel-import size limit.",
            file=ctx.relative_label(path),
            expected=f"at most {MAX_STEP_BYTES} bytes",
            actual=size,
            hint="Split the assembly or provide a simulation-focused export.",
        )
        return
    try:
        with path.open("rb") as stream:
            head = stream.read(16384).decode("ascii", errors="ignore").upper()
            stream.seek(max(0, size - 16384))
            tail = stream.read(16384).decode("ascii", errors="ignore").upper()
    except OSError as exc:
        ctx.error(
            "ASSEMBLY_READ_ERROR",
            "Could not read the STEP assembly.",
            file=ctx.relative_label(path),
            actual=str(exc),
        )
        return
    if size < 128:
        ctx.error(
            "ASSEMBLY_FILE_TOO_SMALL",
            "STEP assembly is too small to contain a valid exchange file.",
            file=ctx.relative_label(path),
            expected="at least 128 bytes",
            actual=size,
        )
    for token in ("ISO-10303-21;", "HEADER;", "DATA;"):
        if token not in head:
            ctx.error(
                "STEP_HEADER_INVALID",
                f"STEP header token {token!r} is missing.",
                file=ctx.relative_label(path),
                expected=token,
            )
    if "END-ISO-10303-21;" not in tail:
        ctx.error(
            "STEP_TERMINATOR_MISSING",
            "STEP file terminator is missing.",
            file=ctx.relative_label(path),
            expected="END-ISO-10303-21;",
        )
    try:
        entity_count, geometry_markers = _scan_step_structure(path)
    except OSError as exc:
        ctx.error(
            "ASSEMBLY_READ_ERROR",
            "Could not scan STEP entities.",
            file=ctx.relative_label(path),
            actual=str(exc),
        )
        return
    ctx.report.package_summary["step_entity_count"] = entity_count
    if entity_count < 10:
        ctx.error(
            "STEP_ENTITY_COUNT_TOO_LOW",
            "STEP contains too few entities to describe usable geometry.",
            file=ctx.relative_label(path),
            expected="at least 10 entity assignments",
            actual=entity_count,
            hint="Export real B-rep assembly geometry instead of a header-only placeholder.",
        )
    if not geometry_markers:
        ctx.error(
            "STEP_GEOMETRY_REPRESENTATION_MISSING",
            "STEP contains no recognized shape representation.",
            file=ctx.relative_label(path),
            expected=sorted(STEP_GEOMETRY_MARKERS),
            actual=[],
            hint="Export the modeled solids from a CAD kernel.",
        )

    inspection = inspect_step_geometry(path)
    if inspection.source_sha256 is not None:
        ctx.report.package_summary["step_kernel_source_sha256"] = (
            inspection.source_sha256
        )
        relative_step = path.relative_to(ctx.package_dir).as_posix().casefold()
        declared_step_digests: set[str] = set()
        artifacts = (
            ctx.manifest.get("artifacts")
            if isinstance(ctx.manifest, dict)
            else None
        )
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                artifact_path = artifact.get("path")
                digest = artifact.get("sha256")
                if not isinstance(artifact_path, str):
                    continue
                normalized = Path(artifact_path).as_posix().casefold()
                if (
                    normalized == relative_step
                    and isinstance(digest, str)
                    and SHA256_PATTERN.fullmatch(digest)
                ):
                    declared_step_digests.add(digest.casefold())
        if (
            declared_step_digests
            and inspection.source_sha256.casefold()
            not in declared_step_digests
        ):
            ctx.error(
                "STEP_KERNEL_SNAPSHOT_HASH_MISMATCH",
                "OCCT validated a STEP snapshot that does not match the "
                "artifact digest declared by package.json.",
                file=ctx.relative_label(path),
                json_path="$.artifacts",
                expected=sorted(declared_step_digests),
                actual=inspection.source_sha256,
                hint=(
                    "Stop all writers, restore the declared STEP bytes, "
                    "refresh hashes only after export completes, and rerun intake."
                ),
            )
            return
    ctx.report.package_summary["step_geometry_kernel"] = (
        "deferred"
        if inspection.failure_category == "infrastructure"
        else ("OCCT" if inspection.kernel_available else "unavailable")
    )
    if inspection.failure_category == "infrastructure":
        ctx.warning(
            "STEP_KERNEL_VALIDATION_DEFERRED",
            "OCCT geometry validation was deferred by an infrastructure failure.",
            file=ctx.relative_label(path),
            actual=inspection.error,
            hint=(
                "Retry on a healthy machine snapshot; do not alter the CAD "
                "solely to repair a timeout, memory guard, worker crash, or "
                "monitoring failure."
            ),
        )
        return
    if not inspection.kernel_available:
        ctx.warning(
            "STEP_KERNEL_VALIDATION_UNAVAILABLE",
            "OCCT is unavailable; STEP geometry received only structural checks.",
            file=ctx.relative_label(path),
            actual=inspection.error,
            hint="Install the OCP/OCCT runtime before physical preflight.",
        )
        return
    if not inspection.transferred:
        ctx.error(
            "STEP_KERNEL_IMPORT_FAILED",
            "OCCT could not transfer the STEP file into CAD geometry.",
            file=ctx.relative_label(path),
            actual=inspection.error,
            hint="Repair or re-export the STEP assembly from the source CAD system.",
        )
        return
    if not inspection.solids:
        ctx.error(
            "STEP_SOLID_MISSING",
            "STEP imports successfully but contains no solid body.",
            file=ctx.relative_label(path),
            expected="at least one solid",
            actual=0,
        )
        return

    ctx.step_geometry_verified = True
    ctx.step_solid_ids = {solid.solid_id for solid in inspection.solids}
    inventory = [solid.to_dict() for solid in inspection.solids]
    ctx.step_solid_inventory = {
        solid.solid_id: solid.to_dict() for solid in inspection.solids
    }
    ctx.report.package_summary["step_solid_count"] = len(inventory)
    ctx.report.package_summary["step_solid_inventory"] = inventory
    invalid = [solid.solid_id for solid in inspection.solids if not solid.valid_brep]
    if invalid:
        ctx.warning(
            "STEP_BREP_INVALID",
            "OCCT reports invalid topology for one or more imported solids.",
            file=ctx.relative_label(path),
            actual=invalid,
            hint="Heal or re-export these solids before collision-mesh generation.",
        )


def _inspect_zip_central_directory(
    path: Path,
    size: int,
) -> tuple[int, int]:
    tail_size = min(size, ZIP_EOCD_FIXED_BYTES + ZIP_MAX_COMMENT_BYTES)
    with path.open("rb") as stream:
        stream.seek(size - tail_size)
        tail = stream.read(tail_size)
    marker = tail.rfind(ZIP_EOCD_SIGNATURE)
    fields: tuple[bytes, int, int, int, int, int, int, int] | None = None
    while marker >= 0:
        if marker + ZIP_EOCD_FIXED_BYTES <= len(tail):
            candidate = struct.unpack_from("<4s4H2LH", tail, marker)
            if (
                marker + ZIP_EOCD_FIXED_BYTES + candidate[-1]
                == len(tail)
            ):
                fields = candidate
                break
        marker = tail.rfind(ZIP_EOCD_SIGNATURE, 0, marker)
    if fields is None:
        raise ValueError("ZIP end-of-central-directory record is missing")
    (
        _signature,
        disk_index,
        central_directory_disk,
        entries_on_disk,
        total_entries,
        central_directory_bytes,
        central_directory_offset,
        _comment_bytes,
    ) = fields
    if (
        disk_index != 0
        or central_directory_disk != 0
        or entries_on_disk != total_entries
    ):
        raise ValueError("multi-disk ZIP containers are not supported")
    if (
        total_entries == 0xFFFF
        or central_directory_bytes == 0xFFFFFFFF
        or central_directory_offset == 0xFFFFFFFF
    ):
        raise ValueError("ZIP64 FCStd containers are not supported")
    absolute_eocd_offset = size - tail_size + marker
    if (
        central_directory_offset + central_directory_bytes
        > absolute_eocd_offset
    ):
        raise ValueError("ZIP central directory extends beyond its end record")
    return total_entries, central_directory_bytes


def _validate_native_assembly_content(
    ctx: ValidationContext, path: Path
) -> None:
    label = ctx.relative_label(path)
    try:
        size = path.stat().st_size
    except OSError as exc:
        ctx.error(
            "NATIVE_ASSEMBLY_READ_ERROR",
            "Could not inspect the declared native CAD assembly.",
            file=label,
            actual=str(exc),
        )
        return
    if size < 64:
        ctx.error(
            "NATIVE_ASSEMBLY_FILE_TOO_SMALL",
            "Declared native CAD assembly is too small to be usable.",
            file=label,
            expected="at least 64 bytes",
            actual=size,
            hint="Export the real native CAD document instead of a placeholder.",
        )
        return
    if path.suffix.casefold() == ".fcstd":
        try:
            announced_entries, announced_directory_bytes = (
                _inspect_zip_central_directory(path, size)
            )
        except (OSError, struct.error, ValueError) as exc:
            ctx.error(
                "NATIVE_ASSEMBLY_CONTAINER_INVALID",
                "Declared FreeCAD assembly is not a readable FCStd container.",
                file=label,
                expected="bounded single-disk ZIP-based FCStd",
                actual=f"{type(exc).__name__}: {exc}",
                hint="Re-save the native document from FreeCAD.",
            )
            return
        if announced_entries > MAX_FCSTD_ENTRIES:
            ctx.error(
                "NATIVE_ASSEMBLY_ARCHIVE_ENTRY_LIMIT",
                "FCStd container contains too many archive entries.",
                file=label,
                expected=f"at most {MAX_FCSTD_ENTRIES} entries",
                actual=announced_entries,
            )
            return
        if (
            announced_directory_bytes
            > MAX_FCSTD_CENTRAL_DIRECTORY_BYTES
        ):
            ctx.error(
                "NATIVE_ASSEMBLY_ARCHIVE_DIRECTORY_LIMIT",
                "FCStd central directory exceeds the safe intake limit.",
                file=label,
                expected=(
                    f"at most {MAX_FCSTD_CENTRAL_DIRECTORY_BYTES} bytes"
                ),
                actual=announced_directory_bytes,
                hint="Re-save or simplify the FreeCAD document.",
            )
            return
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > MAX_FCSTD_ENTRIES:
                    ctx.error(
                        "NATIVE_ASSEMBLY_ARCHIVE_ENTRY_LIMIT",
                        "FCStd container contains too many archive entries.",
                        file=label,
                        expected=f"at most {MAX_FCSTD_ENTRIES} entries",
                        actual=len(entries),
                    )
                    return
                uncompressed_bytes = sum(entry.file_size for entry in entries)
                compressed_bytes = sum(entry.compress_size for entry in entries)
                if uncompressed_bytes > MAX_FCSTD_UNCOMPRESSED_BYTES:
                    ctx.error(
                        "NATIVE_ASSEMBLY_ARCHIVE_SIZE_LIMIT",
                        "FCStd expands beyond the safe intake limit.",
                        file=label,
                        expected=(
                            f"at most {MAX_FCSTD_UNCOMPRESSED_BYTES} "
                            "uncompressed bytes"
                        ),
                        actual=uncompressed_bytes,
                    )
                    return
                ratio = uncompressed_bytes / max(compressed_bytes, 1)
                if (
                    uncompressed_bytes > 100 * 1024**2
                    and ratio > MAX_FCSTD_COMPRESSION_RATIO
                ):
                    ctx.error(
                        "NATIVE_ASSEMBLY_ARCHIVE_RATIO_UNSAFE",
                        "FCStd has a suspicious compression ratio.",
                        file=label,
                        expected=(
                            f"ratio at most {MAX_FCSTD_COMPRESSION_RATIO:g} "
                            "for large archives"
                        ),
                        actual=ratio,
                        hint="Re-save the native document from FreeCAD.",
                    )
                    return
                unsafe_names = []
                encrypted_names = []
                names: set[str] = set()
                for entry in entries:
                    normalized = entry.filename.replace("\\", "/")
                    parts = normalized.split("/")
                    if (
                        normalized.startswith("/")
                        or ".." in parts
                        or re.match(r"^[A-Za-z]:", normalized)
                    ):
                        unsafe_names.append(entry.filename)
                    if entry.flag_bits & 0x1:
                        encrypted_names.append(entry.filename)
                    names.add(normalized.casefold())
                if unsafe_names:
                    ctx.error(
                        "NATIVE_ASSEMBLY_ARCHIVE_PATH_UNSAFE",
                        "FCStd contains unsafe archive paths.",
                        file=label,
                        actual=unsafe_names[:20],
                    )
                    return
                if encrypted_names:
                    ctx.error(
                        "NATIVE_ASSEMBLY_ARCHIVE_ENCRYPTED",
                        "FCStd contains encrypted entries that cannot be verified.",
                        file=label,
                        actual=encrypted_names[:20],
                    )
                    return
        except (OSError, zipfile.BadZipFile) as exc:
            ctx.error(
                "NATIVE_ASSEMBLY_CONTAINER_INVALID",
                "Declared FreeCAD assembly is not a readable FCStd container.",
                file=label,
                expected="ZIP-based FCStd containing Document.xml",
                actual=f"{type(exc).__name__}: {exc}",
                hint=(
                "Save the real FreeCAD document; renaming another file "
                "is invalid."
            ),
            )
            return
        if "document.xml" not in names:
            ctx.error(
                "NATIVE_ASSEMBLY_DOCUMENT_MISSING",
                "FCStd container does not contain Document.xml.",
                file=label,
                expected="Document.xml at the archive root",
                actual=sorted(names)[:20],
                hint="Re-save the document from FreeCAD.",
            )
    else:
        ctx.warning(
            "NATIVE_ASSEMBLY_CONTENT_UNVERIFIED",
            "Native CAD extension is recognized, but this intake "
            "runtime has no parser for its contents.",
            file=label,
            actual=path.suffix,
            hint=(
                "Keep the source file; a format-specific importer must "
                "verify it before mate recovery."
            ),
        )


def validate_artifacts(ctx: ValidationContext) -> None:
    document = ctx.manifest
    if document is None:
        return
    if ctx.report.package_summary.get("package_inventory_truncated") is True:
        ctx.warning(
            "ARTIFACT_INTEGRITY_SKIPPED",
            "Artifact hashing was skipped because package inventory exceeded "
            "a safety limit.",
            hint=(
                "Reduce the package below the reported file, entry, or size "
                "limit, then rerun intake for complete integrity findings."
            ),
        )
        return

    for key in (
        "assembly",
        "assembly_contract",
        "coverage",
        "task_mapping",
    ):
        path = ctx.paths.get(key)
        if isinstance(path, Path):
            ctx.required_artifact_paths.add(
                path.relative_to(ctx.package_dir).as_posix()
            )
    capsule_paths = ctx.paths.get("capsules")
    if isinstance(capsule_paths, list):
        for path in capsule_paths:
            if isinstance(path, Path):
                ctx.required_artifact_paths.add(
                    path.relative_to(ctx.package_dir).as_posix()
                )
    native_path = ctx.paths.get("native_assembly")
    if isinstance(native_path, Path):
        native_exists = ctx.require_file(
            native_path,
            code="NATIVE_ASSEMBLY_FILE_MISSING",
            label="declared native assembly",
        )
        if native_exists:
            ctx.required_artifact_paths.add(
                native_path.relative_to(ctx.package_dir).as_posix()
            )
        if native_path.suffix.casefold() not in NATIVE_CAD_EXTENSIONS:
            ctx.error(
                "NATIVE_ASSEMBLY_EXTENSION_INVALID",
                "source_native_assembly_file is not a recognized native CAD format.",
                file=ctx.relative_label(native_path),
                expected=sorted(NATIVE_CAD_EXTENSIONS),
                actual=native_path.suffix,
                hint=(
                    "Declare reproducible Python/OpenSCAD source as "
                    "source_cad_generator_file, not as a native assembly."
                ),
            )
        elif native_exists:
            _validate_native_assembly_content(ctx, native_path)

    generator_path = ctx.paths.get("cad_generator")
    if isinstance(generator_path, Path):
        if ctx.require_file(
            generator_path,
            code="CAD_GENERATOR_FILE_MISSING",
            label="declared CAD generator source",
        ):
            ctx.required_artifact_paths.add(
                generator_path.relative_to(ctx.package_dir).as_posix()
            )
        if generator_path.suffix.casefold() not in CAD_GENERATOR_EXTENSIONS:
            ctx.error(
                "CAD_GENERATOR_EXTENSION_INVALID",
                "source_cad_generator_file is not a supported source format.",
                file=ctx.relative_label(generator_path),
                expected=sorted(CAD_GENERATOR_EXTENSIONS),
                actual=generator_path.suffix,
            )
    if (
        isinstance(native_path, Path)
        and isinstance(generator_path, Path)
        and native_path == generator_path
    ):
        ctx.error(
            "CAD_SOURCE_ROLE_CONFLICT",
            "One file cannot be both native CAD and generator source.",
            file="package.json",
            json_path="$.source_native_assembly_file",
            actual=ctx.relative_label(native_path),
        )

    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        return
    if len(artifacts) > MAX_ARTIFACT_ENTRIES:
        return
    seen: set[str] = set()
    for index, artifact in enumerate(artifacts):
        base = f"$.artifacts[{index}]"
        if not isinstance(artifact, dict):
            ctx.error(
                "ARTIFACT_ENTRY_INVALID",
                "Each artifact manifest entry must be an object.",
                file="package.json",
                json_path=base,
                actual=artifact,
            )
            continue
        ctx.warn_unknown_fields(
            artifact,
            {"path", "sha256"},
            file="package.json",
            json_path=base,
        )
        relative = artifact.get("path")
        if not isinstance(relative, str) or not relative.strip():
            ctx.error(
                "ARTIFACT_PATH_INVALID",
                "Artifact path must be a non-empty relative string.",
                file="package.json",
                json_path=f"{base}.path",
                actual=relative,
            )
            continue
        path = ctx.safe_package_path(
            relative, manifest_field=f"artifacts[{index}].path"
        )
        if path is None:
            continue
        normalized = path.relative_to(ctx.package_dir).as_posix()
        key = normalized.casefold()
        if key == "package.json":
            ctx.error(
                "PACKAGE_MANIFEST_SELF_HASH_FORBIDDEN",
                "package.json cannot hash itself.",
                file="package.json",
                json_path=f"{base}.path",
                hint=(
                    "Remove package.json from artifacts; its content includes "
                    "the artifact manifest itself."
                ),
            )
            continue
        if key in seen:
            ctx.error(
                "ARTIFACT_PATH_DUPLICATE",
                "Artifact path is listed more than once.",
                file="package.json",
                json_path=f"{base}.path",
                actual=relative,
            )
            continue
        seen.add(key)
        ctx.declared_artifact_paths.add(normalized)
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            ctx.error(
                "ARTIFACT_HASH_INVALID",
                "Artifact sha256 must contain 64 hexadecimal characters.",
                file="package.json",
                json_path=f"{base}.sha256",
                actual=digest,
            )
        if not path.is_file():
            ctx.error(
                "DECLARED_ARTIFACT_MISSING",
                "Declared artifact file is missing.",
                file=normalized,
                json_path=f"{base}.path",
                actual="missing",
            )
            continue
        if isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest):
            actual_digest = ctx.sha256_file(path)
            if (
                actual_digest is not None
                and actual_digest.casefold() != digest.casefold()
            ):
                ctx.error(
                    "ARTIFACT_HASH_MISMATCH",
                    "Artifact content does not match its declared SHA-256.",
                    file=normalized,
                    json_path=f"{base}.sha256",
                    expected=digest.casefold(),
                    actual=actual_digest,
                    hint="Regenerate hashes only after all package files are final.",
                )

    declared_keys = {path.casefold() for path in ctx.declared_artifact_paths}
    undeclared = sorted(
        path
        for path in ctx.package_file_paths
        if path.casefold() != "package.json"
        and path.casefold() not in declared_keys
    )
    if undeclared:
        ctx.error(
            "PACKAGE_FILE_UNDECLARED",
            "Every package file except package.json must have an artifact hash.",
            file="package.json",
            json_path="$.artifacts",
            expected="all regular package files listed exactly once",
            actual={
                "undeclared_count": len(undeclared),
                "paths": undeclared[:100],
            },
            hint=(
                "Remove logs, caches, reports, and scratch files; add every "
                "intentional artifact with its SHA-256."
            ),
        )
    for required in sorted(ctx.required_artifact_paths):
        if required.casefold() not in declared_keys:
            ctx.error(
                "REQUIRED_ARTIFACT_UNHASHED",
                "Required artifact is not listed in package.json artifacts.",
                file=required,
                expected="artifact entry with SHA-256",
                actual="not declared",
            )

    requirements_ref = document.get("requirements_ref")
    if (
        isinstance(requirements_ref, dict)
        and ctx.requirements_path is not None
        and ctx.requirements_path.is_file()
    ):
        expected_digest = requirements_ref.get("sha256")
        if isinstance(expected_digest, str) and SHA256_PATTERN.fullmatch(
            expected_digest
        ):
            actual_digest = ctx.sha256_file(ctx.requirements_path)
            if (
                actual_digest is not None
                and actual_digest.casefold() != expected_digest.casefold()
            ):
                ctx.error(
                    "REQUIREMENTS_HASH_MISMATCH",
                    "Package is bound to a different requirements file.",
                    file="package.json",
                    json_path="$.requirements_ref.sha256",
                    expected=expected_digest.casefold(),
                    actual=actual_digest,
                    hint=(
                        "Do not modify requirements to fit the design; "
                        "use the original immutable file."
                    ),
                )
        external_id = (
            ctx.requirements.get("requirements_id")
            if ctx.requirements is not None
            else None
        )
        declared_id = requirements_ref.get("requirements_id")
        if external_id is not None and declared_id != external_id:
            ctx.error(
                "REQUIREMENTS_ID_MISMATCH",
                "package.json references a different requirements_id.",
                file="package.json",
                json_path="$.requirements_ref.requirements_id",
                expected=external_id,
                actual=declared_id,
            )

def verify_package_stability(ctx: ValidationContext) -> None:
    current_files: set[str] = set()
    current_directories: set[str] = set()
    stack = [ctx.package_dir]
    entry_count = 0
    while stack:
        directory = stack.pop()
        try:
            entries = os.scandir(directory)
        except OSError as exc:
            ctx.error(
                "PACKAGE_STABILITY_SCAN_FAILED",
                "Could not rescan a package directory at intake completion.",
                file=ctx.relative_label(directory),
                actual=str(exc),
            )
            continue
        with entries:
            for entry in entries:
                entry_count += 1
                if entry_count > MAX_PACKAGE_ENTRIES:
                    ctx.error(
                        "PACKAGE_STABILITY_ENTRY_LIMIT",
                        "Package grew beyond the stability-rescan limit.",
                        expected=f"at most {MAX_PACKAGE_ENTRIES} entries",
                        actual=entry_count,
                        hint=(
                            "Stop all writers, remove generated scratch data, "
                            "and rerun intake."
                        ),
                    )
                    return
                path = Path(entry.path)
                normalized = path.relative_to(ctx.package_dir).as_posix()
                try:
                    if entry.is_symlink():
                        ctx.error(
                            "PACKAGE_CHANGED_DURING_VALIDATION",
                            "A symbolic link appeared during intake.",
                            file=normalized,
                            actual="symlink",
                            hint=(
                                "Stop package writers and rerun on an "
                                "immutable directory."
                            ),
                        )
                        continue
                    if _is_path_junction(path):
                        ctx.error(
                            "PACKAGE_CHANGED_DURING_VALIDATION",
                            "A directory junction appeared during intake.",
                            file=normalized,
                            actual="junction",
                            hint=(
                                "Stop package writers and rerun on an "
                                "immutable directory."
                            ),
                        )
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        current_directories.add(normalized)
                        stack.append(path)
                        continue
                    if entry.is_file(follow_symlinks=False):
                        current_files.add(normalized)
                        continue
                except OSError as exc:
                    ctx.error(
                        "PACKAGE_STABILITY_SCAN_FAILED",
                        "Could not inspect an entry during the final rescan.",
                        file=normalized,
                        actual=str(exc),
                    )
                    continue
                ctx.error(
                    "PACKAGE_CHANGED_DURING_VALIDATION",
                    "A non-regular filesystem entry appeared during intake.",
                    file=normalized,
                    actual="special filesystem entry",
                )

    added_files = sorted(current_files - ctx.package_file_paths)
    removed_files = sorted(ctx.package_file_paths - current_files)
    added_directories = sorted(
        current_directories - ctx.package_directory_paths
    )
    removed_directories = sorted(
        ctx.package_directory_paths - current_directories
    )
    if (
        added_files
        or removed_files
        or added_directories
        or removed_directories
    ):
        ctx.error(
            "PACKAGE_LAYOUT_CHANGED_DURING_VALIDATION",
            "Package files or directories changed before intake completed.",
            expected={
                "file_count": len(ctx.package_file_paths),
                "directory_count": len(ctx.package_directory_paths),
            },
            actual={
                "file_count": len(current_files),
                "directory_count": len(current_directories),
                "added_files": added_files[:100],
                "removed_files": removed_files[:100],
                "added_directories": added_directories[:100],
                "removed_directories": removed_directories[:100],
            },
            hint=(
                "Stop generators, sync tools, and report writers; then rerun "
                "intake on a stable package snapshot."
            ),
        )

    for path, (expected_identity, expected_digest) in sorted(
        ctx.hashed_file_identities.items(),
        key=lambda item: str(item[0]).casefold(),
    ):
        try:
            stat = path.stat()
            actual_identity = (
                int(stat.st_size),
                int(stat.st_mtime_ns),
                getattr(stat, "st_ino", None),
            )
        except OSError as exc:
            ctx.error(
                "HASHED_FILE_STABILITY_CHECK_FAILED",
                "Could not recheck a file that was hashed during intake.",
                file=ctx.relative_label(path),
                actual=str(exc),
            )
            continue
        actual_digest = (
            ctx.sha256_file(path)
            if actual_identity == expected_identity
            else None
        )
        if (
            actual_identity != expected_identity
            or actual_digest != expected_digest
        ):
            ctx.error(
                "HASHED_FILE_CHANGED_DURING_VALIDATION",
                "A hashed artifact or requirements file changed before intake completed.",
                file=ctx.relative_label(path),
                expected={
                    "identity": expected_identity,
                    "sha256": expected_digest,
                },
                actual={
                    "identity": actual_identity,
                    "sha256": actual_digest,
                },
                hint=(
                    "Stop all writers, restore the declared artifact content, "
                    "and rerun intake."
                ),
            )
