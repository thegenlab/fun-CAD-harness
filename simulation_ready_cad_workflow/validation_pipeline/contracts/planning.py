"""Read-side planning contract shared by the Loop, the Harness, and readers.

The Loop writes ``planning/task_planning.json`` (see
``react_loop.storage.write_planning_contract``).  Everything else —
the Harness before dispatch, the dashboard, benchmark tooling — must
revalidate that frozen document through :func:`load_planning_contract`
here instead of trusting Loop internals.

Schema policy: new documents are only written with ``PLANNING_SCHEMA``.
The older v2–v4 schemas are read-compatible for frozen historical runs
and are accepted exclusively through this boundary reader.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from ..mechanism.io import canonical_sha256, read_json, sha256_file
from ..run_layout import parse_canonical_run
from .errors import ReactLoopError
from .payloads import _plan_payload, _understanding_payload, strict_json_copy


LEGACY_PLANNING_SCHEMA = "simulation_ready_cad_react_task_planning_v2"
CATALOG_PLANNING_SCHEMA = "simulation_ready_cad_react_task_planning_v3"
PREVIOUS_PLANNING_SCHEMA = "simulation_ready_cad_react_task_planning_v4"
PLANNING_SCHEMA = "simulation_ready_cad_react_task_planning_v5"
WORKFLOW_PROFILE = "single_conversation_react_v2"

#: Every schema this boundary reader can still interpret.  New writers
#: must use ``PLANNING_SCHEMA`` only.
READABLE_PLANNING_SCHEMAS = frozenset(
    {
        LEGACY_PLANNING_SCHEMA,
        CATALOG_PLANNING_SCHEMA,
        PREVIOUS_PLANNING_SCHEMA,
        PLANNING_SCHEMA,
    }
)


class NativeStorageError(ValueError):
    pass


def frozen_project_adapter_bindings(
    *,
    workspace_root: str | Path,
    contract_path: str | Path,
) -> dict[str, str]:
    """Read legacy adapter metadata from an already-validated frozen plan.

    The mapping exists only for the exact project-adapter fallback still used
    by historical, unregistered mechanism kinds.  It is not a planning
    template or a new resource approval.
    """

    workspace = Path(workspace_root).resolve()
    contract = read_json(contract_path, max_bytes=4 * 1024 * 1024)
    output: dict[str, str] = {}
    for item in contract.get("mechanism_templates") or ():
        if not isinstance(item, dict):
            continue
        path_text = item.get("path")
        digest = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if not isinstance(path_text, str) or not path_text.endswith(".py"):
            continue
        source = (workspace / path_text).resolve()
        if (
            not isinstance(digest, str)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or source.is_symlink()
            or not source.is_file()
            or source.stat().st_size != size_bytes
            or sha256_file(source) != digest
        ):
            raise NativeStorageError("frozen project adapter binding changed")
        relative = f"mechanism_templates/{source.name}"
        if relative in output and output[relative] != digest:
            raise NativeStorageError("frozen project adapter bindings conflict")
        output[relative] = digest
    return dict(sorted(output.items()))


def load_planning_contract(
    *,
    workspace_root: str | Path,
    run_dir: str | Path,
) -> tuple[dict[str, Any], Path]:
    """Read one planning contract and return its current execution view.

    Legacy Python rows are rechecked as frozen metadata, then removed from the
    JSON-only planning/resource view.  Their exact fallback binding remains
    available through :func:`frozen_project_adapter_bindings`.
    """

    workspace = Path(workspace_root).resolve()
    run = Path(run_dir).resolve()
    identity = parse_canonical_run(workspace, run)
    if identity is None or identity.run_dir != run:
        raise NativeStorageError("planning run is not an exact canonical revision")
    try:
        manifest = read_json(run / "run_manifest.json")
        contract_path = run / "planning" / "task_planning.json"
        contract = read_json(contract_path, max_bytes=4 * 1024 * 1024)
    except Exception as exc:
        raise NativeStorageError(f"planning document could not be read: {exc}") from exc

    expected_layout = identity.to_dict()
    task_source_binding = manifest.get("task_source_binding")
    if (
        manifest.get("workflow_profile") != WORKFLOW_PROFILE
        or manifest.get("task_planning_policy") != "required_before_attempt_v2"
        or not isinstance(task_source_binding, dict)
        or any(manifest.get(key) != value for key, value in expected_layout.items())
    ):
        raise NativeStorageError("run manifest is not bound to native v2 planning")

    common_required = {
        "schema_version",
        "producer",
        "run_layout",
        "task_source_binding_sha256",
        "task_understanding",
        "joint_plan",
        "test_plan",
        "authored_test_tasks",
        "canonical_payload_sha256",
    }
    schema_version = contract.get("schema_version")
    catalog_required = {
        "planning_template_catalog",
        "template_selections",
    }
    if schema_version == PLANNING_SCHEMA:
        required = common_required | catalog_required | {
            "mechanism_templates",
            "resource_dependencies",
            "resource_request",
            "resource_approval",
            "requirement_obligations",
            "obligation_coverage",
        }
    elif schema_version == PREVIOUS_PLANNING_SCHEMA:
        required = common_required | catalog_required | {
            "mechanism_templates",
            "resource_dependencies",
            "resource_request",
            "resource_approval",
        }
    elif schema_version == CATALOG_PLANNING_SCHEMA:
        required = common_required | catalog_required
    elif schema_version == LEGACY_PLANNING_SCHEMA:
        required = common_required
    else:
        required = set()
    core = {
        key: value
        for key, value in contract.items()
        if key != "canonical_payload_sha256"
    }
    if (
        set(contract) != required
        or schema_version not in READABLE_PLANNING_SCHEMAS
        or contract.get("producer") != "react_parent"
        or contract.get("run_layout") != expected_layout
        or contract.get("task_source_binding_sha256")
        != canonical_sha256(task_source_binding)
        or contract.get("canonical_payload_sha256") != canonical_sha256(core)
    ):
        raise NativeStorageError("planning contract identity or digest is invalid")

    raw_authored = contract.get("authored_test_tasks")
    authored_for_shape: list[dict[str, str]] = []
    seen_authored_paths: set[str] = set()
    if not isinstance(raw_authored, list) or len(raw_authored) > 128:
        raise NativeStorageError("authored test bindings must be a bounded list")
    for item in raw_authored:
        if not isinstance(item, dict) or set(item) != {
            "task_id",
            "path",
            "sha256",
            "size_bytes",
        }:
            raise NativeStorageError("authored test binding fields are invalid")
        task_id = item.get("task_id")
        relative_text = item.get("path")
        digest = item.get("sha256")
        size_bytes = item.get("size_bytes")
        if (
            not isinstance(task_id, str)
            or not isinstance(relative_text, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
            or size_bytes > 16 * 1024 * 1024
        ):
            raise NativeStorageError("authored test binding values are invalid")
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or "\\" in relative_text
            or "\x00" in relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative_text in seen_authored_paths
        ):
            raise NativeStorageError("authored test binding path is invalid")
        seen_authored_paths.add(relative_text)
        lexical = workspace.joinpath(*relative.parts)
        resolved = lexical.resolve()
        try:
            loop_relative = resolved.relative_to(identity.project_dir)
        except ValueError as exc:
            raise NativeStorageError(
                "authored test binding is outside its project"
            ) from exc
        loop_parts = loop_relative.parts
        if (
            lexical != resolved
            or lexical.is_symlink()
            or len(loop_parts) != 5
            or loop_parts[0] not in {"react_loop", "react_loop_v2"}
            or loop_parts[1] != "turns"
            or not loop_parts[2].startswith("t")
            or loop_parts[3] != "authored_tests"
            or not resolved.is_file()
            or resolved.stat().st_size != size_bytes
            or sha256_file(resolved) != digest
        ):
            raise NativeStorageError("authored test file binding is invalid")
        authored_for_shape.append(
            {"task_id": task_id, "path": f"authored_tests/{relative.name}"}
        )

    frozen_adapter_records: list[dict[str, Any]] = []
    if schema_version in {PLANNING_SCHEMA, PREVIOUS_PLANNING_SCHEMA}:
        from .resource_approval import (
            normalize_dependencies,
            validate_resource_approval,
            validate_resource_request,
        )

        raw_mechanisms = contract.get("mechanism_templates")
        if not isinstance(raw_mechanisms, list) or len(raw_mechanisms) > 128:
            raise NativeStorageError(
                "mechanism template bindings must be a bounded list"
            )
        for item in raw_mechanisms:
            if not isinstance(item, dict) or set(item) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise NativeStorageError(
                    "mechanism template binding fields are invalid"
                )
            relative_text = item.get("path")
            digest = item.get("sha256")
            size_bytes = item.get("size_bytes")
            if (
                not isinstance(relative_text, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or not isinstance(size_bytes, int)
                or isinstance(size_bytes, bool)
                or size_bytes < 0
                or size_bytes > 16 * 1024 * 1024
            ):
                raise NativeStorageError(
                    "mechanism template binding values are invalid"
                )
            relative = PurePosixPath(relative_text)
            if (
                relative.is_absolute()
                or "\\" in relative_text
                or "\x00" in relative_text
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                raise NativeStorageError(
                    "mechanism template binding path is invalid"
                )
            lexical = workspace.joinpath(*relative.parts)
            path = lexical.resolve()
            try:
                loop_relative = path.relative_to(identity.project_dir)
            except ValueError as exc:
                raise NativeStorageError(
                    "mechanism template binding is outside its project"
                ) from exc
            parts = loop_relative.parts
            if (
                lexical != path
                or lexical.is_symlink()
                or not path.is_file()
                or len(parts) != 5
                or parts[0] not in {"react_loop", "react_loop_v2"}
                or parts[1] != "turns"
                or not parts[2].startswith("t")
                or parts[3] != "mechanism_templates"
                or path.suffix not in {".json", ".py"}
                or path.stat().st_size != size_bytes
                or sha256_file(path) != digest
            ):
                raise NativeStorageError(
                    "mechanism template file binding is invalid"
                )
            if path.suffix == ".py":
                frozen_adapter_records.append(item)
        if frozen_adapter_records and (
            len(frozen_adapter_records) != len(raw_mechanisms)
            or raw_authored
            or contract.get("resource_dependencies")
        ):
            raise NativeStorageError(
                "frozen Python adapter metadata cannot be mixed with planning resources"
            )
        try:
            dependencies = normalize_dependencies(
                contract.get("resource_dependencies")
            )
        except ReactLoopError as exc:
            raise NativeStorageError(str(exc)) from exc
        if dependencies != contract.get("resource_dependencies"):
            raise NativeStorageError(
                "resource dependencies are not in canonical order"
            )
        request_binding = contract.get("resource_request")
        approval_binding = contract.get("resource_approval")
        has_resources = bool(raw_authored or raw_mechanisms)
        if has_resources:
            if (
                not isinstance(request_binding, dict)
                or set(request_binding)
                != {"request_id", "path", "sha256", "bundle_sha256"}
                or not isinstance(approval_binding, dict)
                or set(approval_binding)
                != {"request_id", "path", "sha256", "effect"}
            ):
                raise NativeStorageError(
                    "authored resources have no complete approval binding"
                )
            for binding, directory in (
                (request_binding, "requests"),
                (approval_binding, "decisions"),
            ):
                path_text = binding.get("path")
                digest = binding.get("sha256")
                request_id = binding.get("request_id")
                if (
                    not isinstance(path_text, str)
                    or not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                    or not isinstance(request_id, str)
                    or not request_id
                ):
                    raise NativeStorageError(
                        "resource approval binding values are invalid"
                    )
                relative = PurePosixPath(path_text)
                if (
                    relative.is_absolute()
                    or "\\" in path_text
                    or "\x00" in path_text
                    or any(part in {"", ".", ".."} for part in relative.parts)
                ):
                    raise NativeStorageError(
                        "resource approval binding path is invalid"
                    )
                candidate = workspace.joinpath(*relative.parts).resolve()
                try:
                    project_relative = candidate.relative_to(identity.project_dir)
                except ValueError as exc:
                    raise NativeStorageError(
                        "resource approval binding is outside its project"
                    ) from exc
                approval_parts = project_relative.parts
                if (
                    len(approval_parts) != 4
                    or approval_parts[0] not in {"react_loop", "react_loop_v2"}
                    or approval_parts[1:] != (
                        "resource_approvals",
                        directory,
                        f"{request_id}.json",
                    )
                ):
                    raise NativeStorageError(
                        "resource approval binding path is not canonical"
                    )
            request_lexical = workspace / request_binding["path"]
            marker_lexical = workspace / approval_binding["path"]
            request_path = request_lexical.resolve()
            marker_path = marker_lexical.resolve()
            if (
                request_lexical != request_path
                or request_lexical.is_symlink()
                or not request_path.is_file()
                or sha256_file(request_path) != request_binding["sha256"]
                or marker_lexical != marker_path
                or marker_lexical.is_symlink()
                or not marker_path.is_file()
                or sha256_file(marker_path) != approval_binding["sha256"]
            ):
                raise NativeStorageError(
                    "resource request or approval marker changed"
                )
            try:
                request_document = read_json(
                    request_path, max_bytes=4 * 1024 * 1024
                )
                marker_document = read_json(
                    marker_path, max_bytes=4 * 1024 * 1024
                )
                request = validate_resource_request(workspace, request_document)
                marker = validate_resource_approval(
                    workspace_root=workspace,
                    project_id=identity.project_id,
                    requirements_sha256=task_source_binding["requirements_sha256"],
                    request=request,
                    marker=marker_document,
                )
            except ReactLoopError as exc:
                raise NativeStorageError(str(exc)) from exc
            if (
                request_binding["request_id"] != request["request_id"]
                or request_binding["bundle_sha256"] != request["bundle_sha256"]
                or approval_binding["request_id"] != request["request_id"]
                or approval_binding["effect"] != marker["effect"]
                or request["dependencies"] != dependencies
                or {item["path"] for item in request["resources"]}
                != {
                    item["path"]
                    for item in [*raw_authored, *raw_mechanisms]
                }
            ):
                raise NativeStorageError(
                    "resource approval does not cover the exact planning resources"
                )
        elif (
            request_binding is not None
            or approval_binding is not None
        ):
            raise NativeStorageError(
                "resource approval data exists without authored resources"
            )
    try:
        from .planning_catalog import (
            normalize_template_selections,
            review_template_selections,
            validate_catalog_snapshot,
        )

        understanding = _understanding_payload(
            {"task_understanding": contract.get("task_understanding")}
        )["task_understanding"]
        raw_tests = contract.get("test_plan")
        tests_for_shape = strict_json_copy(
            raw_tests, label="planning test plan"
        )
        if schema_version == LEGACY_PLANNING_SCHEMA:
            for test in tests_for_shape if isinstance(tests_for_shape, list) else []:
                target = test.get("target") if isinstance(test, dict) else None
                if isinstance(target, dict):
                    thresholds = target.pop("thresholds", None)
                    if thresholds is not None and (
                        not isinstance(thresholds, dict)
                        or not thresholds
                        or len(thresholds) > 128
                    ):
                        raise ReactLoopError(
                            "REACT_RESPONSE_INVALID",
                            "legacy planning thresholds are malformed",
                        )
        catalog_snapshot = (
            validate_catalog_snapshot(contract.get("planning_template_catalog"))
            if schema_version
            in {
                PLANNING_SCHEMA,
                PREVIOUS_PLANNING_SCHEMA,
                CATALOG_PLANNING_SCHEMA,
            }
            else None
        )
        plan = _plan_payload(
            {
                "joint_plan": contract.get("joint_plan"),
                "test_plan": tests_for_shape,
                "authored_test_tasks": authored_for_shape,
                "template_selections": (
                    contract.get("template_selections")
                    if schema_version
                    in {
                        PLANNING_SCHEMA,
                        PREVIOUS_PLANNING_SCHEMA,
                        CATALOG_PLANNING_SCHEMA,
                    }
                    else []
                ),
            }
        )
        selections, _warnings = review_template_selections(
            joint_plan=plan["joint_plan"],
            test_plan=plan["test_plan"],
            template_selections=plan["template_selections"],
            catalog_snapshot=catalog_snapshot,
        )
    except ReactLoopError as exc:
        raise NativeStorageError(f"planning payload is invalid: {exc}") from exc
    if (
        understanding != contract["task_understanding"]
        or plan["joint_plan"] != contract["joint_plan"]
        or plan["test_plan"] != tests_for_shape
        or (
            schema_version
            in {
                PLANNING_SCHEMA,
                PREVIOUS_PLANNING_SCHEMA,
                CATALOG_PLANNING_SCHEMA,
            }
            and (
                normalize_template_selections(
                    contract["template_selections"],
                    catalog_snapshot=catalog_snapshot,
                )
                != selections
                or contract["template_selections"] != selections
                or contract["planning_template_catalog"] != catalog_snapshot
            )
        )
    ):
        raise NativeStorageError(
            f"planning payload is not in canonical {schema_version!r} form"
        )
    if schema_version == PLANNING_SCHEMA:
        from .requirement_obligations import (
            build_requirement_obligation_inventory,
            validate_obligation_coverage,
        )

        requirements_text = task_source_binding.get("requirements_path")
        if not isinstance(requirements_text, str):
            raise NativeStorageError(
                "task source has no frozen requirements path for obligation validation"
            )
        requirements_path = (workspace / requirements_text).resolve()
        try:
            requirements_path.relative_to(identity.project_dir)
        except ValueError as exc:
            raise NativeStorageError(
                "frozen requirements escaped the planning project"
            ) from exc
        if (
            requirements_path.is_symlink()
            or not requirements_path.is_file()
            or sha256_file(requirements_path)
            != task_source_binding.get("requirements_sha256")
        ):
            raise NativeStorageError("frozen requirements are missing or changed")
        expected_obligations = build_requirement_obligation_inventory(
            read_json(requirements_path, max_bytes=2 * 1024 * 1024)
        )
        if contract.get("requirement_obligations") != expected_obligations:
            raise NativeStorageError(
                "planning requirement obligation inventory differs from the frozen task"
            )
        try:
            normalized_coverage = validate_obligation_coverage(
                inventory=expected_obligations,
                coverage=contract.get("obligation_coverage"),
                joint_plan=plan["joint_plan"],
                test_plan=plan["test_plan"],
                template_selections=selections,
                catalog_snapshot=catalog_snapshot,
            )
        except ReactLoopError as exc:
            raise NativeStorageError(
                f"planning obligation coverage is invalid: {exc}"
            ) from exc
        if contract.get("obligation_coverage") != normalized_coverage:
            raise NativeStorageError(
                "planning obligation coverage is not in canonical order"
            )
    if frozen_adapter_records:
        # The source file was validated against its recorded digest above; the
        # returned execution view gets its own canonical digest after removing
        # historical adapter metadata from planning/resource fields.
        projected = dict(contract)
        projected["mechanism_templates"] = []
        projected["resource_dependencies"] = []
        projected["resource_request"] = None
        projected["resource_approval"] = None
        projected_core = {
            key: value
            for key, value in projected.items()
            if key != "canonical_payload_sha256"
        }
        projected["canonical_payload_sha256"] = canonical_sha256(
            projected_core
        )
        return projected, contract_path
    return contract, contract_path
