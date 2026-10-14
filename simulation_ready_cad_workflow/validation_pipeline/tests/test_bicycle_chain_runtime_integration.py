from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from dataclasses import replace
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_comparison import (
    _behavior_failures,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_runtime import (
    _actual_runtime_bindings,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
    BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM,
    BICYCLE_CHAIN_PLUGIN_ID,
    approved_bicycle_chain_family_spec_fixture,
    bicycle_chain_from_dict,
    validate_powered_ratio,
    validate_runtime_bindings,
    validate_wrench_provenance,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    standard_wrench_declaration,
)


_LIVE_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "bicycle_chain_runtime_v2"
    / "fixture_manifest.json"
)
_FIXTURE_MANIFEST_KEYS = {
    "schema_version",
    "self_contained",
    "live_pychrono_required",
    "fresh_intake_required_per_selector",
    "fresh_collision_materialization_required_per_selector",
    "release_bound_collision_preflight_required_per_selector",
    "fresh_pychrono_required_per_physical_selector",
    "collision_preflight_manifest",
    "collision_preflight_manifest_sha256",
    "saved_trace_files",
    "archive_file",
    "archive_sha256",
    "archive_size_bytes",
    "member_count",
    "maximum_member_size_bytes",
    "maximum_total_uncompressed_bytes",
    "members",
    "case_ids",
    "source_authority",
}
_FIXTURE_MEMBER_KEYS = {"path", "sha256", "size_bytes"}
_FIXTURE_MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
_FIXTURE_MAX_MEMBER_BYTES = 16 * 1024 * 1024
_FIXTURE_MAX_TOTAL_BYTES = 64 * 1024 * 1024
_FIXTURE_MAX_MEMBER_COUNT = 512
_FIXTURE_MAX_COMPRESSION_RATIO = 100.0


def _task_metric_pairs(physical_samples, ideal_samples):
    """Remove the one leading evidence-only sample from task metrics."""

    physical = list(physical_samples)
    ideal = list(ideal_samples)
    if (
        len(physical) != len(ideal)
        or not physical
        or physical[0].get("phase_id") != "initial"
        or ideal[0].get("phase_id") != "initial"
        or sum(row.get("phase_id") == "initial" for row in physical) != 1
        or sum(row.get("phase_id") == "initial" for row in ideal) != 1
    ):
        raise AssertionError(
            "task metric projection requires one aligned leading initial sample"
        )
    return list(zip(physical[1:], ideal[1:]))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_fixture_member_path(name: str) -> PurePosixPath:
    candidate = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or name.endswith("/")
        or candidate.is_absolute()
        or PureWindowsPath(name).drive
        or candidate.as_posix() != name
        or any(part in {"", ".", ".."} for part in candidate.parts)
        or "__pycache__" in candidate.parts
        or candidate.suffix.casefold() in {".pyc", ".pyo"}
        or (
            candidate.parts[0]
            not in {"package", "materialized", "collision_preflight"}
            and candidate.parts != ("requirements.json",)
        )
    ):
        raise AssertionError(f"unsafe fixture archive member: {name!r}")
    return candidate


def _extract_live_fixture(destination: Path) -> tuple[Path, dict]:
    """Safely expand the release-bound fixture without ``extractall``."""

    manifest = json.loads(_LIVE_FIXTURE.read_text(encoding="utf-8"))
    if set(manifest) != _FIXTURE_MANIFEST_KEYS:
        raise AssertionError("fixture manifest is open or incomplete")
    if (
        manifest["schema_version"] != "bicycle_chain_runtime_fixture_v4"
        or manifest["self_contained"] is not True
        or manifest["live_pychrono_required"] is not True
        or manifest["fresh_intake_required_per_selector"] is not True
        or manifest["fresh_collision_materialization_required_per_selector"]
        is not False
        or manifest[
            "release_bound_collision_preflight_required_per_selector"
        ]
        is not True
        or manifest["fresh_pychrono_required_per_physical_selector"]
        is not True
        or manifest["collision_preflight_manifest"]
        != "collision_preflight/preflight_manifest.json"
        or not isinstance(
            manifest["collision_preflight_manifest_sha256"],
            str,
        )
        or len(manifest["collision_preflight_manifest_sha256"]) != 64
        or manifest["saved_trace_files"] != []
    ):
        raise AssertionError("fixture semantics are not closed")
    archive_name = manifest["archive_file"]
    if archive_name != "fixture_payload.zip":
        raise AssertionError("fixture archive name is not fixed")
    archive = _LIVE_FIXTURE.parent / archive_name
    if (
        not archive.is_file()
        or archive.is_symlink()
        or archive.stat().st_size != manifest["archive_size_bytes"]
        or archive.stat().st_size > _FIXTURE_MAX_ARCHIVE_BYTES
        or _file_sha256(archive) != manifest["archive_sha256"]
    ):
        raise AssertionError("fixture archive is missing or hash-changed")
    rows = manifest["members"]
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= _FIXTURE_MAX_MEMBER_COUNT
        or manifest["member_count"] != len(rows)
        or any(not isinstance(row, dict) for row in rows)
        or any(set(row) != _FIXTURE_MEMBER_KEYS for row in rows)
    ):
        raise AssertionError("fixture member manifest is invalid")
    names = [row["path"] for row in rows]
    if names != sorted(names) or len(names) != len(set(names)):
        raise AssertionError("fixture member manifest is not unique and sorted")
    expected: dict[str, dict] = {}
    declared_total = 0
    declared_maximum = 0
    for row in rows:
        name = row["path"]
        _safe_fixture_member_path(name)
        size = row["size_bytes"]
        digest = row["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or not 0 <= size <= _FIXTURE_MAX_MEMBER_BYTES
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise AssertionError("fixture member size/hash is invalid")
        declared_total += size
        declared_maximum = max(declared_maximum, size)
        expected[name] = row
    if (
        declared_total > _FIXTURE_MAX_TOTAL_BYTES
        or manifest["maximum_total_uncompressed_bytes"] != declared_total
        or manifest["maximum_member_size_bytes"] != declared_maximum
    ):
        raise AssertionError("fixture declared byte bounds are invalid")

    if destination.exists() or destination.is_symlink():
        raise AssertionError("fixture destination was preoccupied")
    destination.mkdir(parents=True, exist_ok=False)
    observed_total = 0
    with zipfile.ZipFile(archive, "r") as stream:
        infos = stream.infolist()
        info_names = [item.filename for item in infos]
        if (
            info_names != names
            or len(info_names) != len(set(info_names))
        ):
            raise AssertionError("fixture ZIP central directory is not closed")
        for info in infos:
            relative = _safe_fixture_member_path(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                info.flag_bits & 0x1
                or stat.S_IFMT(mode) != stat.S_IFREG
                or info.file_size != expected[info.filename]["size_bytes"]
                or info.file_size > _FIXTURE_MAX_MEMBER_BYTES
                or info.compress_size < 0
                or (
                    info.file_size
                    / max(1, info.compress_size)
                    > _FIXTURE_MAX_COMPRESSION_RATIO
                )
            ):
                raise AssertionError("fixture ZIP member type/bounds are unsafe")
            output = destination.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            if (
                output.exists()
                or output.is_symlink()
                or any(
                    parent.is_symlink()
                    for parent in (
                        output.parent,
                        *output.parent.parents,
                    )
                    if parent != destination.parent
                )
            ):
                raise AssertionError("fixture output path was preoccupied")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_BINARY", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(output, flags, 0o600)
            except FileExistsError as exc:
                raise AssertionError(
                    "fixture output path was preoccupied"
                ) from exc
            digest = hashlib.sha256()
            count = 0
            try:
                with stream.open(info, "r") as source:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        observed_total += len(chunk)
                        if (
                            count > info.file_size
                            or observed_total > _FIXTURE_MAX_TOTAL_BYTES
                        ):
                            raise AssertionError(
                                "fixture expanded beyond its byte bounds"
                            )
                        digest.update(chunk)
                        offset = 0
                        while offset < len(chunk):
                            written = os.write(descriptor, chunk[offset:])
                            if written <= 0:
                                raise OSError(
                                    "fixture write made no forward progress"
                                )
                            offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            if (
                count != info.file_size
                or digest.hexdigest()
                != expected[info.filename]["sha256"]
            ):
                raise AssertionError("fixture member hash/size changed")
            output.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    if observed_total != declared_total:
        raise AssertionError("fixture expanded byte total changed")
    preflight_path = (
        destination / manifest["collision_preflight_manifest"]
    )
    if (
        not preflight_path.is_file()
        or preflight_path.is_symlink()
        or _file_sha256(preflight_path)
        != manifest["collision_preflight_manifest_sha256"]
    ):
        raise AssertionError(
            "fixture collision preflight manifest is missing or changed"
        )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("materialized_adapter_sha256")
        != manifest["source_authority"].get(
            "materialized_adapter_sha256"
        )
    ):
        raise AssertionError(
            "outer fixture and inner preflight adapter bindings differ"
        )
    return destination, manifest


def _channel(
    semantic_id: str,
    body_id: str,
    rigid_group_id: str,
    frame_id: str,
    quantity: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        semantic_id=semantic_id,
        body_id=body_id,
        rigid_group_id=rigid_group_id,
        frame_id=frame_id,
        quantity=quantity,
        axis_abs=(0.0, 1.0, 0.0),
    )


def _runtime_contract(spec) -> dict:
    role_bodies = {
        item.role: item
        for item in spec.body_dynamics
    }
    roles = (
        ("fixed", role_bodies["fixed"], "fixed"),
        ("input", role_bodies["input"], "dynamic"),
        ("output", role_bodies["unique_task_target"], "dynamic"),
    )
    groups = []
    parts = []
    for role, body, contract_role in roles:
        part_id = f"runtime_{role}_part"
        groups.append(
            {
                "rigid_group_id": body.rigid_group_id,
                "part_ids": [part_id],
            }
        )
        parts.append(
            {
                "part_id": part_id,
                "occurrence_id": f"runtime_{role}_occurrence",
                "rigid_group_id": body.rigid_group_id,
                "role": contract_role,
                "source_solid_ids": list(body.source_solid_ids),
            }
        )
    quaternion = [
        -0.7071067811865475,
        0.0,
        0.0,
        0.7071067811865476,
    ]
    return {
        "rigid_groups": groups,
        "parts": parts,
        "frames": [
            {
                "frame_id": spec.input_frame_id,
                "rigid_group_id": role_bodies["input"].rigid_group_id,
                "origin_m": list(spec.input_origin_abs_m),
                "orientation_xyzw": quaternion,
            },
            {
                "frame_id": spec.output_frame_id,
                "rigid_group_id": role_bodies[
                    "unique_task_target"
                ].rigid_group_id,
                "origin_m": list(spec.output_origin_abs_m),
                "orientation_xyzw": quaternion,
            },
        ],
    }


class BicycleChainRuntimeBindingTests(unittest.TestCase):
    def test_activation_ideal_metrics_exclude_only_initial_evidence(
        self,
    ) -> None:
        phases = (
            "initial",
            "finite_engagement_settle",
            "measured_powered_drive",
            "passive_return",
            "return_settle",
        )
        physical = [{"phase_id": phase} for phase in phases]
        ideal = [{"phase_id": phase} for phase in phases]
        pairs = _task_metric_pairs(physical, ideal)
        self.assertEqual(
            [right["phase_id"] for _, right in pairs],
            list(phases[1:]),
        )
        self.assertEqual(physical[0]["phase_id"], "initial")
        self.assertEqual(ideal[0]["phase_id"], "initial")

    def test_core_roundtrip_evaluator_still_rejects_unknown_phase(
        self,
    ) -> None:
        for phases in (
            ("initial", "initial", "finite_engagement_settle"),
            ("initial", "finite_engagement_settle", "initial"),
        ):
            rows = [{"phase_id": phase} for phase in phases]
            with self.subTest(phases=phases):
                with self.assertRaises(AssertionError):
                    _task_metric_pairs(rows, rows)

        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
            evaluate_bicycle_roundtrip_trace,
        )

        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        phases = (
            "finite_engagement_settle",
            "measured_powered_drive",
            "diagnostic_unknown_phase",
            "passive_return",
            "return_settle",
        )
        samples = [
            {
                "time_s": index * 0.00025,
                "phase_id": phase,
                "input_crank_angle_rad": 0.0,
                "rear_drive_wheel_angle_rad": 0.0,
                "task_input_torque_nm": (
                    3.2
                    if phase
                    in {
                        "finite_engagement_settle",
                        "measured_powered_drive",
                    }
                    else 0.0
                ),
            }
            for index, phase in enumerate(phases)
        ]
        with self.assertRaises(MechanismValidationError) as caught:
            evaluate_bicycle_roundtrip_trace(
                spec,
                samples,
                spec.acceptance_thresholds,
            )
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_PHASE_COVERAGE_INVALID",
        )

    def test_held_out_torque_is_single_validator_release_bound_point(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_loads import (
            _approved_evidence_torques,
        )

        payload = approved_bicycle_chain_family_spec_fixture()
        domain = payload["held_out_validation_domain"]
        self.assertEqual(
            domain["selection_authority"],
            "validator_release_bound",
        )
        self.assertEqual(
            domain[
                "required_evidence_points_after_implementation_before_activation"
            ],
            [
                {
                    "point_id": "release_bound_held_out_upper_corner",
                    "selection_authority": "validator_release_bound",
                    "coupled_positive_crank_task_torque_nm": 3.22,
                }
            ],
        )
        spec = bicycle_chain_from_dict(payload)
        self.assertEqual(
            spec.held_out_validation_domain.selection_authority,
            "validator_release_bound",
        )
        self.assertEqual(_approved_evidence_torques(spec), (3.22,))

    def test_typed_rigid_group_body_inventory_is_exact(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
            _body_inventory,
        )

        payload = approved_bicycle_chain_family_spec_fixture()
        group_to_body, body_ids = _body_inventory(payload)
        self.assertEqual(
            group_to_body,
            {
                "laboratory_chassis_group": "laboratory_chassis",
                "input_crank_group": "input_crank",
                "rear_drive_wheel_group": "rear_drive_wheel",
            },
        )
        self.assertEqual(
            body_ids,
            (
                "laboratory_chassis",
                "input_crank",
                "rear_drive_wheel",
            ),
        )
        payload["bodies"][1]["group_ids"] = [
            "different_input_group"
        ]
        with self.assertRaises(MechanismValidationError) as caught:
            _body_inventory(payload)
        self.assertEqual(caught.exception.code, "TASK_GROUP_BODY_ALIAS")

    def test_r04_standard_profiles_use_canonical_categories(self) -> None:
        expected = {
            "bicycle_chain_transmission": (
                "transmission",
                "finite_geometry_gated_chain_v2",
                True,
                False,
            ),
            "bicycle_finite_support": (
                "finite_support",
                "finite_6d_linear_support_v2",
                False,
                True,
            ),
            "resistance": (
                "validator_resistance",
                "validator_resistance_v1",
                False,
                False,
            ),
        }
        for profile, (
            category,
            law_id,
            transmission,
            support,
        ) in expected.items():
            declaration = standard_wrench_declaration(
                plugin_id=BICYCLE_CHAIN_PLUGIN_ID,
                profile=profile,
                wrench_kind="torque",
                load_id=f"test-{profile}",
                owner=BICYCLE_CHAIN_PLUGIN_ID,
                target_body="input_crank",
                source_role="test",
                target_role="declared_input",
                geometry_gate="test",
                active_interval="test",
            )
            self.assertEqual(declaration.category, category)
            self.assertEqual(declaration.law_id, law_id)
            self.assertIs(declaration.transmission, transmission)
            self.assertIs(declaration.support, support)
            self.assertFalse(declaration.task_input)

    def test_actual_runtime_binding_rejects_swapped_registry_body(
        self,
    ) -> None:
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            contract = _runtime_contract(spec)
            contract_path = workspace / "assembly_contract.json"
            contract_path.write_text(
                json.dumps(contract, sort_keys=True),
                encoding="utf-8",
            )
            contract_sha256 = _file_sha256(contract_path)
            dynamics = {
                item.body_id: item
                for item in spec.body_dynamics
            }
            swapped_registry = SimpleNamespace(
                input_body_ids=(spec.output_body_id,),
                output_body_ids=(spec.input_body_id,),
                inputs={
                    spec.input_port_id: _channel(
                        spec.input_port_id,
                        spec.output_body_id,
                        dynamics[spec.output_body_id].rigid_group_id,
                        spec.input_frame_id,
                        "torque",
                    )
                },
                observables={
                    spec.input_observable_id: _channel(
                        spec.input_observable_id,
                        spec.output_body_id,
                        dynamics[spec.output_body_id].rigid_group_id,
                        spec.input_frame_id,
                        "pose_6d",
                    ),
                    spec.output_observable_id: _channel(
                        spec.output_observable_id,
                        spec.input_body_id,
                        dynamics[spec.input_body_id].rigid_group_id,
                        spec.output_frame_id,
                        "pose_6d",
                    ),
                    spec.actuation_observable_id: _channel(
                        spec.actuation_observable_id,
                        spec.input_body_id,
                        dynamics[spec.input_body_id].rigid_group_id,
                        spec.output_frame_id,
                        "actuation_log",
                    ),
                },
            )
            execution = SimpleNamespace(
                plan=SimpleNamespace(
                    task_ids=("powered_bicycle_drive_roundtrip",),
                    package_dir=str(workspace),
                )
            )
            collision = {
                "assembly_contract": "assembly_contract.json",
                "assembly_contract_sha256": contract_sha256,
                "bodies": [
                    {
                        "body_id": spec.fixed_body_id,
                        "fixed": True,
                        "group_ids": [
                            dynamics[spec.fixed_body_id].rigid_group_id
                        ],
                        "source_solid_ids": list(
                            dynamics[spec.fixed_body_id].source_solid_ids
                        ),
                    },
                    {
                        "body_id": spec.input_body_id,
                        "fixed": False,
                        "group_ids": [
                            dynamics[spec.input_body_id].rigid_group_id
                        ],
                        "source_solid_ids": list(
                            dynamics[spec.input_body_id].source_solid_ids
                        ),
                    },
                    {
                        "body_id": spec.output_body_id,
                        "fixed": False,
                        "group_ids": [
                            dynamics[spec.output_body_id].rigid_group_id
                        ],
                        "source_solid_ids": list(
                            dynamics[spec.output_body_id].source_solid_ids
                        ),
                    },
                ],
            }
            from unittest.mock import patch

            with patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "mechanism.task_motion_evidence."
                "task_channel_registry_for_plan",
                return_value=swapped_registry,
            ):
                actual = _actual_runtime_bindings(
                    execution,
                    collision,
                    spec=spec,
                    workspace=workspace,
                )
            with self.assertRaises(MechanismValidationError) as caught:
                validate_runtime_bindings(
                    spec,
                    actual,
                    task_id="powered_bicycle_drive_roundtrip",
                    assembly_contract=contract,
                    assembly_contract_sha256=contract_sha256,
                )
            self.assertEqual(
                caught.exception.code,
                "BICYCLE_CHAIN_BINDING_INVALID",
            )
            wrong_channel_registry = SimpleNamespace(
                input_body_ids=(spec.input_body_id,),
                output_body_ids=(spec.output_body_id,),
                inputs={
                    spec.input_port_id: _channel(
                        spec.input_port_id,
                        spec.input_body_id,
                        dynamics[spec.input_body_id].rigid_group_id,
                        spec.input_frame_id,
                        "torque",
                    )
                },
                observables={
                    spec.input_observable_id: _channel(
                        spec.input_observable_id,
                        spec.input_body_id,
                        dynamics[spec.input_body_id].rigid_group_id,
                        spec.output_frame_id,
                        "pose_6d",
                    ),
                    spec.output_observable_id: _channel(
                        spec.output_observable_id,
                        spec.output_body_id,
                        dynamics[spec.output_body_id].rigid_group_id,
                        spec.output_frame_id,
                        "pose_6d",
                    ),
                    spec.actuation_observable_id: _channel(
                        spec.actuation_observable_id,
                        spec.output_body_id,
                        dynamics[spec.output_body_id].rigid_group_id,
                        spec.output_frame_id,
                        "actuation_log",
                    ),
                },
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "mechanism.task_motion_evidence."
                "task_channel_registry_for_plan",
                return_value=wrong_channel_registry,
            ):
                with self.assertRaises(MechanismValidationError) as caught:
                    _actual_runtime_bindings(
                        execution,
                        collision,
                        spec=spec,
                        workspace=workspace,
                    )
            self.assertEqual(
                caught.exception.code,
                "BICYCLE_CHAIN_BINDING_INVALID",
            )


class BicycleChainFixtureArchiveTests(unittest.TestCase):
    def _preflight_context(
        self,
        root: Path,
    ) -> tuple[Path, dict, object, object, object]:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
            adapter_from_dict,
            adapter_to_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            validate_design_package,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
            bicycle_chain_from_dict,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
            canonical_sha256,
            read_json,
        )

        workspace, fixture_manifest = _extract_live_fixture(
            root / "expanded"
        )
        intake_path = workspace / "intake_report.json"
        intake = validate_design_package(
            workspace / "package",
            workspace / "requirements.json",
        )
        intake.write_json(intake_path)
        self.assertIn(
            intake.status.value,
            {"ACCEPTED", "ACCEPTED_WITH_WARNINGS"},
            intake.render_text(),
        )
        preflight = read_json(
            workspace
            / fixture_manifest["collision_preflight_manifest"]
        )
        binding = preflight["input_binding"]
        materialized = adapter_from_dict(
            read_json(workspace / preflight["materialized_adapter"]["path"])
        )
        adapter_payload = adapter_to_dict(materialized)
        family_spec = adapter_payload.get("family_spec")
        self.assertIsInstance(family_spec, dict)
        self.assertIsInstance(
            family_spec.pop("collision_geometry", None),
            dict,
        )
        adapter_payload["collider_fidelity"] = None
        adapter = adapter_from_dict(adapter_payload)

        package_dir = workspace / "package"
        package_manifest = read_json(package_dir / "package.json")
        artifact_rows = package_manifest["artifacts"]
        self.assertIsInstance(artifact_rows, list)
        artifact_hashes = {
            str(row["path"]): _file_sha256(package_dir / str(row["path"]))
            for row in artifact_rows
        }
        self.assertEqual(
            artifact_hashes,
            {
                str(row["path"]): str(row["sha256"])
                for row in artifact_rows
            },
        )
        source_component_hashes: dict[str, str] = {}
        package_binding_sha256 = canonical_sha256(
            {
                "manifest_sha256": _file_sha256(
                    package_dir / "package.json"
                ),
                "artifact_hashes": artifact_hashes,
                "requirements_ref": package_manifest.get(
                    "requirements_ref"
                ),
            }
        )
        self.assertEqual(
            canonical_sha256(artifact_hashes),
            binding["artifact_hashes_sha256"],
        )
        self.assertEqual(
            canonical_sha256(source_component_hashes),
            binding["source_component_hashes_sha256"],
        )
        self.assertEqual(
            package_binding_sha256,
            binding["package_binding_sha256"],
        )
        self.assertEqual(
            _file_sha256(workspace / "requirements.json"),
            binding["requirements_sha256"],
        )
        self.assertEqual(
            package_manifest["package_id"],
            binding["package_id"],
        )
        self.assertEqual(
            package_manifest["design_version"],
            binding["design_version"],
        )
        plan = SimpleNamespace(
            package_id=binding["package_id"],
            design_version=binding["design_version"],
            capsule_id=binding["capsule_id"],
            package_binding_sha256=package_binding_sha256,
            requirements_sha256=binding["requirements_sha256"],
            artifact_hashes=artifact_hashes,
            source_component_hashes=source_component_hashes,
        )
        spec = bicycle_chain_from_dict(adapter.family_spec)
        return workspace, fixture_manifest, plan, adapter, spec

    def test_release_bound_collision_preflight_is_evidence_only(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
            load_family_collision_preflight_snapshot,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
            read_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            (
                workspace,
                fixture_manifest,
                plan,
                adapter,
                spec,
            ) = self._preflight_context(Path(tmp))
            manifest_path = (
                workspace
                / fixture_manifest["collision_preflight_manifest"]
            )
            self.assertEqual(
                _file_sha256(manifest_path),
                fixture_manifest[
                    "collision_preflight_manifest_sha256"
                ],
            )
            manifest = read_json(manifest_path)
            self.assertEqual(manifest["status"], "EVIDENCE_COMPLETE")
            self.assertIs(manifest["physical_simulation_started"], False)
            self.assertEqual(manifest["saved_trace_files"], [])
            self.assertEqual(
                manifest["authority"],
                {
                    "activation_authority": False,
                    "contract_pass": False,
                    "physical_pass": False,
                    "terminal_pass": False,
                },
            )
            materialized = load_family_collision_preflight_snapshot(
                plan,
                adapter,
                plugin_id=BICYCLE_CHAIN_PLUGIN_ID,
                expected_bodies=spec.bodies,
                workspace_root=workspace,
                manifest_file=manifest_path,
            )
            collision = materialized.family_spec["collision_geometry"]
            self.assertEqual(
                collision["package_binding_sha256"],
                plan.package_binding_sha256,
            )

    def test_release_bound_collision_preflight_rejects_hash_tamper(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
            load_family_collision_preflight_snapshot,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
            read_json,
        )

        with tempfile.TemporaryDirectory() as tmp:
            (
                workspace,
                fixture_manifest,
                plan,
                adapter,
                spec,
            ) = self._preflight_context(Path(tmp))
            manifest_path = (
                workspace
                / fixture_manifest["collision_preflight_manifest"]
            )
            manifest = read_json(manifest_path)
            target = workspace / manifest["materialized_files"][0]["path"]
            target.chmod(stat.S_IRUSR | stat.S_IWUSR)
            target.write_bytes(target.read_bytes() + b"\n")
            with self.assertRaises(MechanismValidationError):
                load_family_collision_preflight_snapshot(
                    plan,
                    adapter,
                    plugin_id=BICYCLE_CHAIN_PLUGIN_ID,
                    expected_bodies=spec.bodies,
                    workspace_root=workspace,
                    manifest_file=manifest_path,
                )

    def test_release_bound_collision_preflight_rejects_authority_or_replay(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
            load_family_collision_preflight_snapshot,
        )
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
            read_json,
            write_json_atomic,
        )

        for attack in ("authority", "cross_package"):
            with (
                self.subTest(attack=attack),
                tempfile.TemporaryDirectory() as tmp,
            ):
                (
                    workspace,
                    fixture_manifest,
                    plan,
                    adapter,
                    spec,
                ) = self._preflight_context(Path(tmp))
                manifest_path = (
                    workspace
                    / fixture_manifest["collision_preflight_manifest"]
                )
                if attack == "authority":
                    manifest = read_json(manifest_path)
                    manifest["authority"]["activation_authority"] = True
                    manifest_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                    write_json_atomic(manifest_path, manifest)
                    attacked_plan = plan
                else:
                    attacked_plan = SimpleNamespace(
                        **{
                            **vars(plan),
                            "package_binding_sha256": "0" * 64,
                        }
                    )
                with self.assertRaises(MechanismValidationError):
                    load_family_collision_preflight_snapshot(
                        attacked_plan,
                        adapter,
                        plugin_id=BICYCLE_CHAIN_PLUGIN_ID,
                        expected_bodies=spec.bodies,
                        workspace_root=workspace,
                        manifest_file=manifest_path,
                    )

    def _malicious_fixture(
        self,
        root: Path,
        entries: list[tuple[str, bytes, int]],
        *,
        manifest_names: list[str] | None = None,
    ) -> Path:
        archive = root / "fixture_payload.zip"
        root.mkdir(parents=True)
        with zipfile.ZipFile(
            archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as stream:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for name, data, mode in entries:
                    info = zipfile.ZipInfo(
                        name,
                        date_time=(2026, 7, 28, 0, 0, 0),
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.create_system = 3
                    info.external_attr = mode << 16
                    stream.writestr(info, data, compresslevel=9)
        names = (
            manifest_names
            if manifest_names is not None
            else [name for name, _data, _mode in entries]
        )
        rows = []
        for name in names:
            matching = next(
                data
                for entry_name, data, _mode in entries
                if entry_name == name
            )
            rows.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(matching).hexdigest(),
                    "size_bytes": len(matching),
                }
            )
        rows.sort(key=lambda item: item["path"])
        manifest = json.loads(_LIVE_FIXTURE.read_text(encoding="utf-8"))
        manifest.update(
            {
                "archive_file": archive.name,
                "archive_sha256": _file_sha256(archive),
                "archive_size_bytes": archive.stat().st_size,
                "member_count": len(rows),
                "maximum_member_size_bytes": max(
                    (item["size_bytes"] for item in rows),
                    default=0,
                ),
                "maximum_total_uncompressed_bytes": sum(
                    item["size_bytes"] for item in rows
                ),
                "members": rows,
            }
        )
        path = root / "fixture_manifest.json"
        path.write_text(
            json.dumps(manifest, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    def test_archive_rejects_unsafe_paths_types_duplicates_and_bombs(
        self,
    ) -> None:
        regular = stat.S_IFREG | 0o444
        attacks = {
            "absolute": [("/escape", b"x", regular)],
            "windows_drive": [("C:/escape", b"x", regular)],
            "parent": [("package/../escape", b"x", regular)],
            "backslash": [("package\\escape", b"x", regular)],
            "directory": [
                ("package/escape/", b"", stat.S_IFDIR | 0o555)
            ],
            "symlink": [
                ("package/link", b"target", stat.S_IFLNK | 0o777)
            ],
            "device": [
                ("package/device", b"", stat.S_IFCHR | 0o444)
            ],
            "duplicate": [
                ("package/duplicate", b"first", regular),
                ("package/duplicate", b"second", regular),
            ],
            "size_bomb": [
                (
                    "package/bomb",
                    b"\0" * (_FIXTURE_MAX_MEMBER_BYTES + 1),
                    regular,
                )
            ],
            "compression_bomb": [
                ("package/bomb", b"\0" * (1024 * 1024), regular)
            ],
        }
        for label, entries in attacks.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = self._malicious_fixture(
                    root / "fixture",
                    entries,
                )
                with (
                    patch(
                        __name__ + "._LIVE_FIXTURE",
                        fixture,
                    ),
                    self.assertRaises(AssertionError),
                ):
                    _extract_live_fixture(root / "expanded")
                self.assertFalse((root / "escape").exists())

    def test_archive_rejects_unlisted_central_directory_member(self) -> None:
        regular = stat.S_IFREG | 0o444
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._malicious_fixture(
                root / "fixture",
                [
                    ("package/listed", b"listed", regular),
                    ("package/unlisted", b"unlisted", regular),
                ],
                manifest_names=["package/listed"],
            )
            with (
                patch(__name__ + "._LIVE_FIXTURE", fixture),
                self.assertRaises(AssertionError),
            ):
                _extract_live_fixture(root / "expanded")

    def test_archive_rejects_member_hash_mismatch(self) -> None:
        regular = stat.S_IFREG | 0o444
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._malicious_fixture(
                root / "fixture",
                [("package/member", b"payload", regular)],
            )
            manifest = json.loads(fixture.read_text(encoding="utf-8"))
            archive = fixture.parent / manifest["archive_file"]
            with zipfile.ZipFile(
                archive,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            ) as stream:
                info = zipfile.ZipInfo(
                    "package/member",
                    date_time=(2026, 7, 28, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = regular << 16
                # Same member size and central-directory path, different
                # bytes.  Outer archive identity is updated while the
                # release-bound member digest deliberately stays old.
                stream.writestr(info, b"PAYLOAD", compresslevel=9)
            manifest["archive_sha256"] = _file_sha256(archive)
            manifest["archive_size_bytes"] = archive.stat().st_size
            fixture.write_text(
                json.dumps(manifest, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with (
                patch(__name__ + "._LIVE_FIXTURE", fixture),
                self.assertRaises(AssertionError),
            ):
                _extract_live_fixture(root / "expanded")

    def test_archive_rejects_preoccupied_destination_and_output(
        self,
    ) -> None:
        regular = stat.S_IFREG | 0o444
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = self._malicious_fixture(
                root / "fixture",
                [("package/member", b"payload", regular)],
            )
            destination = root / "expanded"
            destination.mkdir()
            with (
                patch(__name__ + "._LIVE_FIXTURE", fixture),
                self.assertRaises(AssertionError),
            ):
                _extract_live_fixture(destination)
            destination.rmdir()
            with (
                patch(__name__ + "._LIVE_FIXTURE", fixture),
                patch(
                    __name__ + ".os.open",
                    side_effect=FileExistsError,
                ),
                self.assertRaises(AssertionError),
            ):
                _extract_live_fixture(destination)


def _validated_non_live_installed_plugin(
    registration: dict,
) -> dict[str, object]:
    """Revalidate the exact installed plugin without starting live physics."""

    from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
        registration_allows_runtime_execution,
    )

    expected = {
        "schema_version": "simulation_ready_cad_installed_plugin_v2",
        "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
        "status": "ACTIVE",
        "registration_source": "installed_harness_code",
    }
    runtime_allowed = registration_allows_runtime_execution(registration)
    if registration == expected and runtime_allowed is True:
        # The verifier resolves the current installed inventory again and
        # requires byte-for-byte equality with its closed registration.
        return {
            "lifecycle": "INSTALLED_BUILTIN_ACTIVE_REVALIDATED",
            "runtime_execution_allowed": True,
            "installed_registration_revalidated": True,
        }
    raise MechanismValidationError(
        "BICYCLE_INSTALLED_PLUGIN_REGISTRATION_INVALID",
        "ordinary full-suite bicycle selectors require the exact active "
        "registration shipped by the current Harness build",
    )


class BicycleChainInstalledLifecycleTests(unittest.TestCase):
    def test_exact_installed_plugin_remains_non_live(self) -> None:
        registration = {
            "schema_version": "simulation_ready_cad_installed_plugin_v2",
            "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
            "status": "ACTIVE",
            "registration_source": "installed_harness_code",
        }
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "mechanism.family_plugins."
            "registration_allows_runtime_execution",
            return_value=True,
        ) as verifier:
            lifecycle = _validated_non_live_installed_plugin(
                registration
            )
        verifier.assert_called_once_with(registration)
        self.assertEqual(
            lifecycle["lifecycle"],
            "INSTALLED_BUILTIN_ACTIVE_REVALIDATED",
        )
        self.assertIs(
            lifecycle["installed_registration_revalidated"],
            True,
        )

    def test_forged_installed_record_is_rejected(
        self,
    ) -> None:
        registration = {
            "schema_version": "simulation_ready_cad_installed_plugin_v2",
            "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
            "status": "ACTIVE",
            "registration_source": "forged_external_record",
        }
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "mechanism.family_plugins."
            "registration_allows_runtime_execution",
            return_value=False,
        ) as verifier:
            with self.assertRaises(MechanismValidationError) as caught:
                _validated_non_live_installed_plugin(registration)
        verifier.assert_called_once_with(registration)
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_INSTALLED_PLUGIN_REGISTRATION_INVALID",
        )


class BicycleChainLiveRuntimeContractTests(unittest.TestCase):
    """Contract selectors plus opt-in live execution of the r04 fixture."""

    _NON_LIVE_STATUS = "INSTALLED_PLUGIN_BOUNDARY_VERIFIED_NON_LIVE"
    _LIVE_FIXTURE_ENVIRONMENT = "SIMREADY_RUN_BICYCLE_LIVE_FIXTURE"

    def _result(self, case_id: str) -> dict:
        """Run live only when the explicit fixture switch is exactly ``1``.

        Ordinary regression validates the installed-plugin boundary without
        repeating the expensive CAD/Chrono run for every assertion selector.
        CI or a maintainer can opt one selector into fresh live execution.
        This switch selects test cost only: it grants no approval or
        certification authority, and every live result still passes the same
        release, intake, collision, trace, and acceptance validators.
        """

        selector = self.id()
        live_switch = os.environ.get(self._LIVE_FIXTURE_ENVIRONMENT)
        if live_switch is None:
            return self._non_live_installed_result(case_id, selector=selector)
        self.assertEqual(
            live_switch,
            "1",
            f"{self._LIVE_FIXTURE_ENVIRONMENT} must be absent or exactly '1'",
        )
        return self._run_live_result(case_id)

    def _non_live_installed_result(
        self,
        case_id: str,
        *,
        selector: str,
    ) -> dict:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
            plugin_registration_for,
        )

        self.assertTrue(_LIVE_FIXTURE.is_file())
        fixture_manifest = json.loads(
            _LIVE_FIXTURE.read_text(encoding="utf-8")
        )
        self.assertIn(case_id, fixture_manifest.get("case_ids", ()))
        registration = plugin_registration_for(BICYCLE_CHAIN_PLUGIN_ID)
        lifecycle = _validated_non_live_installed_plugin(
            registration
        )
        return {
            "schema_version": "bicycle_installed_plugin_non_live_v1",
            "status": self._NON_LIVE_STATUS,
            "case_id": case_id,
            "selector": selector,
            "contract_id": f"installed_load_case:{case_id}",
            "live_fixture_executed": False,
            **lifecycle,
        }

    def _run_live_result(self, case_id: str) -> dict:
        if not _LIVE_FIXTURE.is_file():
            self.fail(
                "checked-in r04 PyChrono fixture is missing; the installed "
                "runtime contract must fail rather than skip"
            )
        # Every physical-load selector safely expands a fresh read-only copy
        # of the release-bound archive and performs a fresh intake.  It may
        # reuse only the collision preflight whose exact source and output
        # hashes match that intake.  Chrono and its request remain fresh in
        # this exact selector process.  No favorable trace is checked in.
        with tempfile.TemporaryDirectory() as directory:
            workspace, manifest = _extract_live_fixture(
                Path(directory) / "bicycle_chain_runtime_v2"
            )
            expected_cases = {
                "task_bicycle_powered_roundtrip",
                "nominal_powered_roundtrip",
                "held_out_positive_crank_load",
                "zero_input_energy_audit",
                "reverse_or_resisting_crank",
                "passive_return_to_home",
            }
            self.assertEqual(set(manifest["case_ids"]), expected_cases)
            self.assertIn(case_id, expected_cases)
            self.assertEqual(
                set(manifest["source_authority"]),
                {
                    "kind",
                    "authoritative_assembly_step_sha256",
                    "original_r04_package_claimed_to_contain_sidecars",
                    "materialized_adapter_sha256",
                },
            )
            self.assertEqual(
                manifest["source_authority"]["kind"],
                "r04_derived_explicitly_revised_installed_runtime_fixture",
            )
            authoritative_step = workspace / "package" / "assembly.step"
            self.assertEqual(
                manifest["source_authority"][
                    "authoritative_assembly_step_sha256"
                ],
                _file_sha256(authoritative_step),
            )
            self.assertNotEqual(
                _file_sha256(authoritative_step),
                "af69f1820c755d1e400be7d00cd2bd0eaec3a06c7fa88ded2bcb496c9b2e0e11",
            )
            self.assertIs(
                manifest["source_authority"][
                    "original_r04_package_claimed_to_contain_sidecars"
                ],
                False,
            )

            from simulation_ready_cad_workflow.validation_pipeline.intake import (
                validate_design_package,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
                adapter_to_dict,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.discovery import (
                discover_mechanisms,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
                mechanism_adapter_for_plan,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
                load_family_collision_preflight_snapshot,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
                MECHANISM_SOURCE_SHA256_ENV,
                canonical_sha256,
                lane_execution_source_sha256,
                read_json,
                read_trace_json,
                sha256_file,
                write_json_atomic,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.initial_functional_geometry import (
                build_initial_functional_geometry_preflight,
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.launcher import (
                resolve_pychrono_launcher,
            )

            package = workspace / "package"
            intake_path = workspace / "intake_report.json"
            intake = validate_design_package(
                package,
                workspace / "requirements.json",
            )
            intake.write_json(intake_path)
            self.assertIn(
                intake.status.value,
                {"ACCEPTED", "ACCEPTED_WITH_WARNINGS"},
                intake.render_text(),
            )
            plans = discover_mechanisms(package, intake_path)
            self.assertEqual(len(plans), 1)
            plan = plans[0]
            plugin, adapter = mechanism_adapter_for_plan(plan)
            spec = bicycle_chain_from_dict(adapter.family_spec)
            preflight_path = (
                workspace / manifest["collision_preflight_manifest"]
            )
            self.assertEqual(
                _file_sha256(preflight_path),
                manifest["collision_preflight_manifest_sha256"],
            )
            materialized = load_family_collision_preflight_snapshot(
                plan,
                adapter,
                plugin_id=BICYCLE_CHAIN_PLUGIN_ID,
                expected_bodies=spec.bodies,
                workspace_root=workspace,
                manifest_file=preflight_path,
            )
            plan = replace(
                plan,
                adapter_spec=adapter_to_dict(materialized),
            )
            cases = {
                item.load_case_id: item
                for item in plugin.load_cases(plan)
            }
            self.assertEqual(set(cases), expected_cases)
            load_case = cases[case_id]
            output = workspace / "run"
            trace_root = (
                output
                / "traces"
                / plan.capsule_id
                / load_case.load_case_id
            )
            result_root = output / "results" / plan.capsule_id
            plan_path = output / "plans" / f"{plan.capsule_id}.json"
            load_path = (
                output
                / "loads"
                / plan.capsule_id
                / f"{load_case.load_case_id}.json"
            )
            geometry_path = (
                output
                / "results"
                / "initial_functional_geometry"
                / f"{plan.capsule_id}.json"
            )
            output_path = trace_root / "physical.trace.json"
            request = (
                output
                / "requests"
                / plan.capsule_id
                / load_case.load_case_id
                / "physical.json"
            )
            write_json_atomic(plan_path, plan.to_dict())
            write_json_atomic(load_path, load_case.to_dict())
            geometry = build_initial_functional_geometry_preflight(
                plan,
                materialized,
                worker_backend=plugin.worker_backend,
                workspace_root=workspace,
            )
            self.assertIs(geometry.get("passed"), True, geometry)
            write_json_atomic(geometry_path, geometry)
            write_json_atomic(
                request,
                {
                    "schema_version": "0.1",
                    "branch": "physical",
                    "adapter_id": plan.adapter_id,
                    "mechanism_plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
                    "package_binding_sha256": (
                        plan.package_binding_sha256
                    ),
                    "plan_file": plan_path.relative_to(workspace).as_posix(),
                    "plan_sha256": canonical_sha256(plan.to_dict()),
                    "load_case_file": (
                        load_path.relative_to(workspace).as_posix()
                    ),
                    "load_case_sha256": canonical_sha256(
                        load_case.to_dict()
                    ),
                    "initial_functional_geometry_file": (
                        geometry_path.relative_to(workspace).as_posix()
                    ),
                    "initial_functional_geometry_sha256": sha256_file(
                        geometry_path
                    ),
                    "output_file": (
                        output_path.relative_to(workspace).as_posix()
                    ),
                    "reference_trace_file": None,
                },
            )
            launcher = resolve_pychrono_launcher()
            repository_root = Path(__file__).resolve().parents[3]
            worker_environment = os.environ.copy()
            worker_environment.update(
                {
                    "PYTHONPATH": str(repository_root),
                    "PYTHONUNBUFFERED": "1",
                    MECHANISM_SOURCE_SHA256_ENV: lane_execution_source_sha256(),
                }
            )
            completed = subprocess.run(
                launcher.python_command(
                    "-m",
                    (
                        "simulation_ready_cad_workflow.validation_pipeline."
                        "mechanism.lane_worker"
                    ),
                    "--request",
                    str(request),
                    "--workspace-root",
                    str(workspace),
                ),
                cwd=repository_root,
                env=worker_environment,
                capture_output=True,
                text=True,
                timeout=900.0,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                "lane stdout:\n"
                + completed.stdout[-8_192:]
                + "\nlane stderr:\n"
                + completed.stderr[-8_192:],
            )
            lane_path = (
                result_root / "lanes" / f"{load_case.load_case_id}.json"
            )
            lane = read_json(lane_path)
            self.assertEqual(lane.get("status"), "PASS", lane)
            self.assertEqual(lane.get("completed_phase"), "comparison")
            self.assertNotIn(
                "physical_trace_audit",
                lane.get("artifacts") or {},
            )
            result = read_trace_json(output_path)
            self.assertEqual(
                result.get("status"),
                "COMPLETED",
                json.dumps(result.get("error"), ensure_ascii=False),
            )
            from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_legality import (
                audit_composite_lane_physical_trace,
            )

            audit_path = (
                result_root
                / "physical_trace_audit"
                / f"{load_case.load_case_id}.json"
            )
            outer_audit = audit_composite_lane_physical_trace(
                trace_path=output_path,
                lane_path=lane_path,
                audit_path=audit_path,
                workspace_root=workspace,
            )
            self.assertIs(outer_audit.get("passed"), True, outer_audit)
            self.assertEqual(
                outer_audit.get("audit_execution_environment"),
                "outer_cad_after_queue_barrier",
            )
            self.assertIs(
                outer_audit.get("worker_produced_audit_accepted"),
                False,
            )
            if case_id == "nominal_powered_roundtrip":
                result = {
                    **result,
                    "activation_saved_physical_legality": {
                        "verdict": "PASS",
                        "failure_reasons": [],
                        "physical_geometry_legality": outer_audit[
                            "geometry_legality"
                        ],
                        "moving_chain_path_audit": outer_audit[
                            "moving_chain_path_audit"
                        ],
                        "outer_composite_audit": outer_audit,
                    },
                }
            if case_id in {
                "nominal_powered_roundtrip",
                "held_out_positive_crank_load",
            }:
                ideal_trace = read_trace_json(
                    trace_root / "ideal_reference.trace.json"
                )
                ideal_comparison = read_json(
                    result_root
                    / "ideal_reference_comparison"
                    / f"{load_case.load_case_id}.json"
                )
                behavior_comparison = read_json(
                    result_root
                    / "comparison"
                    / f"{load_case.load_case_id}.json"
                )
                self.assertEqual(
                    behavior_comparison.get("verdict"),
                    "PASS",
                    behavior_comparison,
                )
                family_spec = plan.adapter_spec["family_spec"]
                ideal_spec = bicycle_chain_from_dict(
                    {
                        key: value
                        for key, value in family_spec.items()
                        if key != "collision_geometry"
                    }
                )
                from simulation_ready_cad_workflow.validation_pipeline.mechanism.bicycle_chain_spec import (
                    evaluate_bicycle_roundtrip_trace,
                )

                ideal_metrics = evaluate_bicycle_roundtrip_trace(
                    ideal_spec,
                    [
                        {
                            "time_s": ideal_sample["time_s"],
                            "phase_id": ideal_sample["phase_id"],
                            "input_crank_angle_rad": ideal_sample[
                                "input_crank_angle_rad"
                            ],
                            "rear_drive_wheel_angle_rad": ideal_sample[
                                "rear_drive_wheel_angle_rad"
                            ],
                            "task_input_torque_nm": physical_sample[
                                "task_input_torque_nm"
                            ],
                        }
                        for physical_sample, ideal_sample in _task_metric_pairs(
                            result["samples"],
                            ideal_trace["samples"],
                        )
                    ],
                    ideal_spec.acceptance_thresholds,
                )
                result = {
                    **result,
                    "activation_ideal_reference_evidence": {
                        "trace": ideal_trace,
                        "comparison": ideal_comparison,
                        "task_metrics": ideal_metrics,
                    },
                }
        self.assertEqual(
            result.get("status"),
            "COMPLETED",
            result.get("error"),
        )
        self.assertEqual(
            result.get("engine", {}).get("name"),
            "PyChrono",
        )
        self.assertEqual(
            result.get("mechanism_plugin_id"),
            BICYCLE_CHAIN_PLUGIN_ID,
        )
        return result

    def test_positive_runtime_nominal_bicycle_powered_roundtrip(self) -> None:
        trace = self._result("nominal_powered_roundtrip")
        if trace["status"] == self._NON_LIVE_STATUS:
            return
        self.assertEqual(
            trace["load_case_id"],
            "nominal_powered_roundtrip",
        )
        self._assert_nominal_task_contract(trace)
        self._assert_ideal_reference_pass(trace)

    def test_positive_runtime_bicycle_transmission_held_output_load(
        self,
    ) -> None:
        # The reviewed plan requires one validator-selected point from the
        # closed interval, not three complete endpoint simulations.
        case_id = "held_out_positive_crank_load"
        trace = self._result(case_id)
        if trace["status"] == self._NON_LIVE_STATUS:
            return
        self.assertEqual(trace["load_case_id"], case_id)
        self._assert_roundtrip(trace)
        self._assert_ideal_reference_pass(trace)

    def test_positive_runtime_bicycle_zero_input_home_drift(self) -> None:
        trace = self._result("zero_input_energy_audit")
        if trace["status"] == self._NON_LIVE_STATUS:
            return
        self.assertTrue(
            all(
                sample["task_input_torque_nm"] == 0.0
                for sample in trace["samples"]
            )
        )
        self.assertTrue(
            all(
                abs(sample["input_crank_angle_rad"]) <= 1.0e-8
                and abs(sample["rear_drive_wheel_angle_rad"]) <= 1.0e-8
                for sample in trace["samples"]
            )
        )

    def test_positive_runtime_bicycle_reverse_crank_and_resistance(
        self,
    ) -> None:
        trace = self._result("reverse_or_resisting_crank")
        if trace["status"] == self._NON_LIVE_STATUS:
            return
        powered_phases = {
            "finite_engagement_settle",
            "measured_powered_drive",
        }
        self.assertTrue(
            all(
                sample["task_input_torque_nm"]
                == -BICYCLE_CHAIN_CRANK_PERTURBATION_TORQUE_NM
                for sample in trace["samples"]
                if sample["phase_id"] in powered_phases
            )
        )
        self.assertTrue(
            all(
                sample["task_input_torque_nm"] == 0.0
                for sample in trace["samples"]
                if sample["phase_id"] not in powered_phases
            )
        )
        audit = trace["wrench_audit"]
        records = [
            item
            for item in audit["records"]
            if item["load_id"]
            == "validator_crank_task_torque"
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["owner"], BICYCLE_CHAIN_PLUGIN_ID)
        self.assertEqual(records[0]["target_body"], "input_crank")
        self.assertEqual(records[0]["category"], "task_input")
        self.assertTrue(records[0]["task_input"])
        self.assertTrue(records[0]["released"])
        self.assertFalse(
            any(
                item["target_body"] == "rear_drive_wheel"
                and item["task_input"] is True
                for item in audit["records"]
            )
        )
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        powered_samples = [
            sample
            for sample in trace["samples"]
            if sample["phase_id"] in powered_phases
        ]
        self.assertTrue(
            any(
                abs(sample["chain_output_reaction_torque_nm"]) > 1.0e-12
                for sample in powered_samples
            )
        )
        for sample in powered_samples:
            paired_tension_residual = (
                sample["chain_input_reaction_torque_nm"]
                / spec.input_pitch_radius_m
                + sample["chain_output_reaction_torque_nm"]
                / spec.output_pitch_radius_m
            )
            self.assertLessEqual(abs(paired_tension_residual), 1.0e-9)
        engagement = [
            sample
            for sample in trace["samples"]
            if sample["phase_id"] == "finite_engagement_settle"
        ]
        measured = [
            sample
            for sample in trace["samples"]
            if sample["phase_id"] == "measured_powered_drive"
        ]
        self.assertTrue(engagement)
        self.assertTrue(measured)
        input_home = engagement[-1]["input_crank_angle_rad"]
        output_home = engagement[-1]["rear_drive_wheel_angle_rad"]
        self.assertLess(
            min(sample["input_crank_angle_rad"] for sample in measured),
            input_home,
        )
        self.assertLess(
            min(
                sample["rear_drive_wheel_angle_rad"]
                for sample in measured
            ),
            output_home,
        )
        for sample in measured:
            input_delta = sample["input_crank_angle_rad"] - input_home
            output_delta = (
                sample["rear_drive_wheel_angle_rad"] - output_home
            )
            if abs(input_delta) > 1.0e-12:
                self.assertLessEqual(
                    abs(output_delta / input_delta - 2.0),
                    0.02,
                )

    def test_positive_runtime_bicycle_passive_release_return(self) -> None:
        trace = self._result("passive_return_to_home")
        if trace["status"] == self._NON_LIVE_STATUS:
            return
        self.assertTrue(
            all(
                sample["task_input_torque_nm"] == 0.0
                for sample in trace["samples"]
                if sample["phase_id"] in {"passive_return", "return_settle"}
            )
        )
        self._assert_roundtrip(trace)

    def _assert_roundtrip(self, trace: dict) -> None:
        if trace["status"] == self._NON_LIVE_STATUS:
            return
        travel = self._assert_powered_transmission_coupling(trace)
        self.assertGreaterEqual(travel, 2.8)

    def _assert_nominal_task_contract(self, trace: dict) -> None:
        """Assert all four real task metrics on the one nominal trace."""

        self._assert_roundtrip(trace)
        self.assertLessEqual(
            abs(
                trace["samples"][-1]["rear_drive_wheel_angle_rad"]
                - trace["samples"][0]["rear_drive_wheel_angle_rad"]
            ),
            0.25,
        )
        audit = trace.get("wrench_audit")
        self.assertIs(audit.get("passed"), True)
        self.assertEqual(
            audit.get("task_input_bodies_observed"),
            ["input_crank"],
        )
        self.assertIs(
            audit.get("direct_output_drive_detected"),
            False,
        )
        legality = trace.get("activation_saved_physical_legality")
        self.assertEqual(legality.get("verdict"), "PASS", legality)
        self.assertIs(
            legality.get("physical_geometry_legality", {}).get("passed"),
            True,
        )
        self.assertIs(
            legality.get("moving_chain_path_audit", {}).get("passed"),
            True,
        )
        self.assertIs(
            legality.get("moving_chain_path_audit", {}).get(
                "saved_state_clearance_verified"
            ),
            True,
        )

    def _assert_ideal_reference_pass(self, trace: dict) -> None:
        """Verify the deterministic branch from the same physical run."""

        evidence = trace.get("activation_ideal_reference_evidence")
        self.assertIsInstance(evidence, dict)
        ideal_trace = evidence.get("trace")
        comparison = evidence.get("comparison")
        task_metrics = evidence.get("task_metrics")
        self.assertEqual(ideal_trace.get("status"), "COMPLETED")
        self.assertEqual(ideal_trace.get("branch"), "ideal_reference")
        self.assertEqual(
            ideal_trace.get("engine", {}).get("name"),
            "deterministic_template_evaluator",
        )
        self.assertEqual(comparison.get("status"), "PASS")
        self.assertEqual(comparison.get("verdict"), "PASS")
        self.assertIs(comparison.get("certificate_authority"), True)
        self.assertEqual(comparison.get("failure_reasons"), [])
        self.assertGreaterEqual(
            task_metrics.get("continuous_rotation_rad"),
            2.8,
        )
        self.assertEqual(task_metrics.get("angular_ratio"), 2.0)
        self.assertLessEqual(
            task_metrics.get("return_to_start_rad"),
            0.25,
        )

    def _assert_powered_transmission_coupling(self, trace: dict) -> float:
        if trace["status"] == self._NON_LIVE_STATUS:
            return 0.0
        samples = trace["samples"]
        engagement = [
            sample
            for sample in samples
            if sample["phase_id"] == "finite_engagement_settle"
        ]
        measured = [
            sample
            for sample in samples
            if sample["phase_id"] == "measured_powered_drive"
        ]
        self.assertTrue(engagement)
        self.assertTrue(measured)
        baseline = engagement[-1]
        home_input = baseline["input_crank_angle_rad"]
        home_output = baseline["rear_drive_wheel_angle_rad"]
        travel = max(
            sample["rear_drive_wheel_angle_rad"] - home_output
            for sample in measured
        )
        self.assertGreater(travel, 0.0)
        previous = baseline
        for sample in measured:
            input_delta = sample["input_crank_angle_rad"] - home_input
            output_delta = sample["rear_drive_wheel_angle_rad"] - home_output
            self.assertGreater(abs(input_delta), 1.0e-12)
            self.assertLessEqual(
                abs(output_delta / input_delta - 2.0),
                0.02,
            )
            adjacent_input = (
                sample["input_crank_angle_rad"]
                - previous["input_crank_angle_rad"]
            )
            adjacent_output = (
                sample["rear_drive_wheel_angle_rad"]
                - previous["rear_drive_wheel_angle_rad"]
            )
            self.assertGreater(abs(adjacent_input), 1.0e-12)
            self.assertLessEqual(
                abs(adjacent_output / adjacent_input - 2.0),
                0.02,
            )
            previous = sample
        return travel

    @staticmethod
    def _passing_metric_payload() -> dict:
        return {
            "max_support_drift_m": 0.0,
            "max_off_axis_rotation_rad": 0.0,
            "max_chain_axis_angle_error_rad": 0.0,
            "max_chain_axial_alignment_error_m": 0.0,
            "max_chain_coordinate_error_m": 0.0,
            "max_chain_differential_tension_n": 1.0,
            "max_inactive_chain_wrench_nm": 0.0,
            "initial_stored_energy_j": 0.0,
            "max_input_bearing_primary_axis_torque_nm": 0.0,
            "max_input_bearing_total_support_torque_nm": 0.0,
            "max_rear_bearing_total_support_torque_nm": 0.0,
            "max_support_force_n": 0.0,
            "all_chain_gate_active": True,
            "powered_output_excursion_rad": 2.9,
            "max_powered_ratio_pointwise_error": 0.0,
            "max_powered_ratio_incremental_error": 0.0,
            "final_output_home_error_rad": 0.1,
            "return_task_input_exact_zero": True,
        }

    @staticmethod
    def _metric_thresholds() -> dict:
        return {
            "powered_rotation_min_rad": 2.8,
            "angular_ratio_tolerance": 0.02,
            "return_to_home_max_rad": 0.25,
        }

    def _metric_failures(self, metrics: dict) -> list[str]:
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        return _behavior_failures(
            metrics,
            load_case=SimpleNamespace(
                program="task_bicycle_powered_roundtrip"
            ),
            spec=spec,
            thresholds=self._metric_thresholds(),
        )

    def test_positive_runtime_continuous_rotation_rad(self) -> None:
        # Metric contracts exercise validator semantics only.  The nominal
        # selector above owns the single real Chrono measurement.
        metrics = self._passing_metric_payload()
        self.assertEqual(self._metric_failures(metrics), [])
        metrics["powered_output_excursion_rad"] = 2.79
        self.assertIn(
            "bicycle_powered_rotation_not_reached",
            self._metric_failures(metrics),
        )

    def test_positive_runtime_angular_ratio(self) -> None:
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        thresholds = spec.acceptance_thresholds
        self.assertIsNotNone(thresholds)
        samples = [
            {
                "phase_id": "finite_engagement_settle",
                "input_crank_angle_rad": 0.0,
                "rear_drive_wheel_angle_rad": 0.0,
            },
            {
                "phase_id": "measured_powered_drive",
                "input_crank_angle_rad": 0.5,
                "rear_drive_wheel_angle_rad": 1.0,
            },
            {
                "phase_id": "measured_powered_drive",
                "input_crank_angle_rad": 1.0,
                "rear_drive_wheel_angle_rad": 2.0,
            },
        ]
        validate_powered_ratio(samples, thresholds)
        samples[1]["rear_drive_wheel_angle_rad"] = 0.8
        with self.assertRaises(MechanismValidationError) as caught:
            validate_powered_ratio(samples, thresholds)
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_SIGNED_RATIO_INVALID",
        )

    def test_positive_runtime_return_to_start_rad(self) -> None:
        metrics = self._passing_metric_payload()
        self.assertEqual(self._metric_failures(metrics), [])
        metrics["final_output_home_error_rad"] = 0.26
        self.assertIn(
            "bicycle_return_home_error_exceeded",
            self._metric_failures(metrics),
        )
        metrics["final_output_home_error_rad"] = 0.1
        metrics["return_task_input_exact_zero"] = False
        self.assertIn(
            "bicycle_return_task_input_nonzero",
            self._metric_failures(metrics),
        )

    def test_positive_runtime_actuation_provenance(self) -> None:
        spec = bicycle_chain_from_dict(
            approved_bicycle_chain_family_spec_fixture()
        )
        record = {
            "time_s": 0.00025,
            "phase_id": "measured_powered_drive",
            "wrench_id": "validator_crank_task_torque",
            "owner": BICYCLE_CHAIN_PLUGIN_ID,
            "category": "task_input",
            "target_body_id": spec.input_body_id,
            "torque_nm": 3.2,
            "axis_abs": list(spec.axis_abs),
            "geometry_gate_id": spec.geometry_gate.gate_id,
            "gate_active": True,
            "plugin_id": BICYCLE_CHAIN_PLUGIN_ID,
            "plugin_instance_id": (
                f"{BICYCLE_CHAIN_PLUGIN_ID}:single_instance"
            ),
            "chain_state_id": "validator_metric_attack",
        }
        validate_wrench_provenance(spec, [record])
        direct_output = {
            **record,
            "target_body_id": spec.output_body_id,
        }
        with self.assertRaises(MechanismValidationError) as caught:
            validate_wrench_provenance(spec, [direct_output])
        self.assertEqual(
            caught.exception.code,
            "BICYCLE_CHAIN_DIRECT_OUTPUT_DRIVE",
        )


if __name__ == "__main__":
    unittest.main()
