from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.intake import (
    IntakeReport,
    IntakeStatus,
    Severity,
    validate_design_package,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.cli import main
from simulation_ready_cad_workflow.validation_pipeline.intake.context import (
    ValidationContext,
)
from simulation_ready_cad_workflow.validation_pipeline.intake.step_geometry import (
    DEFAULT_MAX_KERNEL_RSS_GIB,
    StepGeometryInspection,
    StepSolid,
    inspect_step_geometry,
    inspect_step_geometry_isolated,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


TEST_STEP_FIXTURE = (
    Path(__file__).with_name("fixtures") / "intake_two_solid.step"
)


def _write_test_step(path: Path) -> None:
    path.write_bytes(TEST_STEP_FIXTURE.read_bytes())


def _refresh_artifact_hashes(package: Path) -> None:
    manifest_path = package / "package.json"
    manifest = _read_json(manifest_path)
    for artifact in manifest["artifacts"]:
        artifact["sha256"] = _sha256(package / artifact["path"])
    _write_json(manifest_path, manifest)


def _write_native_fixture(path: Path) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Document.xml", "<Document SchemaVersion='4'/>")


def _known_test_step_inspection() -> StepGeometryInspection:
    return StepGeometryInspection(
        kernel_available=True,
        transferred=True,
        solids=(
            StepSolid(
                solid_id="solid_1",
                valid_brep=True,
                bounds_mm={
                    "min": [-50.0, -30.0, -30.0],
                    "max": [50.0, 30.0, 30.0],
                },
                size_mm=[100.0, 60.0, 60.0],
            ),
            StepSolid(
                solid_id="solid_2",
                valid_brep=True,
                bounds_mm={
                    "min": [-60.0, -10.0, -10.0],
                    "max": [60.0, 10.0, 10.0],
                },
                size_mm=[120.0, 20.0, 20.0],
            ),
        ),
    )


def _make_package(root: Path, *, warnings: bool = False) -> tuple[Path, Path]:
    package = root / "package"
    package.mkdir()
    requirements = root / "requirements.json"
    _write_json(
        requirements,
        {
            "schema_version": "0.1",
            "requirements_id": "linear_stage_requirements",
            "tasks": [
                {
                    "task_id": "move_payload",
                    "description": "Move the payload along the guide axis.",
                    "success_criteria": [
                        "Payload translates without detaching from the guide."
                    ],
                }
            ],
        },
    )
    (package / "assembly.step").write_text(
        """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('INTAKE TEST ASSEMBLY'),'2;1');
FILE_NAME('assembly.step','2026-07-15T00:00:00',('test'),('test'),'','','');
FILE_SCHEMA(('AUTOMOTIVE_DESIGN_CC2'));
ENDSEC;
DATA;
#1=PRODUCT('base','base','',());
#2=PRODUCT('carriage','carriage','',());
ENDSEC;
END-ISO-10303-21;
""",
        encoding="ascii",
    )
    _write_test_step(package / "assembly.step")
    assembly = {
        "schema_version": "0.1",
        "units": {"length": "m", "mass": "kg", "angle": "rad"},
        "world_frame": {
            "right_handed": True,
            "up_axis": "+Z",
            "gravity_m_s2": [0.0, 0.0, -9.81],
        },
        "parts": [
            {
                "part_id": "base",
                "occurrence_id": "occ_base",
                "role": "fixed",
                "rigid_group_id": "group_base",
                "source_solid_ids": ["solid_1"],
                "declared_bounds_m": {
                    "min": [-0.05, -0.03, -0.03],
                    "max": [0.05, 0.03, 0.03],
                },
                "material": {"name": "steel", "density_kg_m3": 7850},
            },
            {
                "part_id": "carriage",
                "occurrence_id": "occ_carriage",
                "role": "dynamic",
                "rigid_group_id": "group_carriage",
                "source_solid_ids": ["solid_2"],
                "declared_bounds_m": {
                    "min": [-0.06, -0.01, -0.01],
                    "max": [0.06, 0.01, 0.01],
                },
                "material": {"name": "aluminum", "density_kg_m3": 2700},
            },
        ],
        "rigid_groups": [
            {"rigid_group_id": "group_base", "part_ids": ["base"]},
            {
                "rigid_group_id": "group_carriage",
                "part_ids": ["carriage"],
            },
        ],
        "grounded_rigid_group_ids": ["group_base"],
        "connections": [
            {
                "connection_id": "guide_contact",
                "between_rigid_group_ids": [
                    "group_base",
                    "group_carriage",
                ],
                "connection_type": "contact",
                "evidence": "Coaxial guide and carriage surfaces in CAD.",
            }
        ],
        "frames": [
            {
                "frame_id": "joint_frame",
                "rigid_group_id": "group_base",
                "origin_m": [0, 0, 0],
                "orientation_xyzw": [0, 0, 0, 1],
            },
            {
                "frame_id": "input_frame",
                "rigid_group_id": "group_carriage",
                "origin_m": [0.05, 0, 0],
                "orientation_xyzw": [0, 0, 0, 1],
            },
            {
                "frame_id": "output_frame",
                "rigid_group_id": "group_carriage",
                "origin_m": [0, 0, 0],
                "orientation_xyzw": [0, 0, 0, 1],
            },
        ],
    }
    if warnings:
        assembly["parts"][1].pop("material")
    _write_json(package / "assembly_contract.json", assembly)

    capsule = {
        "schema_version": "0.1",
        "capsule_id": "linear_stage",
        "name": "Linear stage contact mechanism",
        "member_rigid_group_ids": ["group_carriage"],
        "boundary_rigid_group_ids": ["group_base"],
        "input_ports": [
            {
                "port_id": "drive_force",
                "rigid_group_id": "group_carriage",
                "frame_id": "input_frame",
                "wrench_type": "force",
                "direction_abs": [1, 0, 0],
                "declared_range": {"min": -50, "max": 50, "unit": "N"},
            }
        ],
        "observables": [
            {
                "observable_id": "carriage_pose",
                "rigid_group_id": "group_carriage",
                "frame_id": "output_frame",
                "quantity": "pose_6d",
            }
        ],
        "expected_behavior": {
            "description": "Carriage translates along the guide axis.",
            "dof": [{"motion": "translation_along_joint_axis"}],
            "failure_conditions": [
                "carriage detaches",
                "carriage jams under nominal input",
            ],
        },
        "reduced_model_claim": {
            "verification_status": "UNVERIFIED",
            "family": "prismatic",
            "fixed_group_ids": ["group_base"],
            "moving_group_ids": ["group_carriage"],
            "joint_frame_id": "joint_frame",
            "axis_abs": [1, 0, 0],
        },
        "dependencies": [],
        "declared_nominal_tests": [
            {
                "input_port_id": "drive_force",
                "schedule": "positive then negative pulse",
            }
        ],
    }
    if warnings:
        capsule["input_ports"][0].pop("declared_range")
        capsule.pop("reduced_model_claim")
        capsule.pop("declared_nominal_tests")
    capsule_path = package / "mechanism_capsules" / "linear_stage.json"
    _write_json(capsule_path, capsule)

    _write_json(
        package / "mechanism_coverage.json",
        {
            "schema_version": "0.1",
            "coverage": [
                {
                    "part_id": "base",
                    "classification": "static",
                    "capsule_ids": [],
                },
                {
                    "part_id": "carriage",
                    "classification": "capsule_member",
                    "capsule_ids": ["linear_stage"],
                },
            ],
        },
    )
    _write_json(
        package / "machine_task_mapping.json",
        {
            "schema_version": "0.1",
            "requirements_id": "linear_stage_requirements",
            "task_mappings": [
                {
                    "task_id": "move_payload",
                    "capsule_ids": ["linear_stage"],
                    "inputs": [
                        {
                            "capsule_id": "linear_stage",
                            "port_id": "drive_force",
                        }
                    ],
                    "observables": [
                        {
                            "capsule_id": "linear_stage",
                            "observable_id": "carriage_pose",
                        }
                    ],
                    "implementation_claim": (
                        "The carriage carries the payload and is driven at "
                        "the declared force port."
                    ),
                }
            ],
        },
    )
    if not warnings:
        _write_native_fixture(package / "assembly.FCStd")

    artifact_names = [
        "assembly.step",
        "assembly_contract.json",
        "mechanism_capsules/linear_stage.json",
        "mechanism_coverage.json",
        "machine_task_mapping.json",
    ]
    if not warnings:
        artifact_names.append("assembly.FCStd")
    manifest = {
        "schema_version": "0.1",
        "package_id": "linear_stage_package",
        "design_version": "1.0.0",
        "created_by_ai": True,
        "assembly_file": "assembly.step",
        "assembly_contract_file": "assembly_contract.json",
        "mechanism_coverage_file": "mechanism_coverage.json",
        "machine_task_mapping_file": "machine_task_mapping.json",
        "mechanism_capsule_files": [
            "mechanism_capsules/linear_stage.json"
        ],
        "requirements_ref": {
            "requirements_id": "linear_stage_requirements",
            "sha256": _sha256(requirements),
        },
        "artifacts": [
            {"path": name, "sha256": _sha256(package / name)}
            for name in artifact_names
        ],
    }
    if not warnings:
        manifest["source_native_assembly_file"] = "assembly.FCStd"
    _write_json(package / "package.json", manifest)
    return package, requirements


class IntakeTests(unittest.TestCase):
    def setUp(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime.system_probe import (
            SystemProbe as RealSystemProbe,
        )

        class StableSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class StableProbe:
            def __init__(self, **_kwargs: object) -> None:
                self.real = RealSystemProbe()

            def sample(self, interval_seconds: float = 0.0) -> object:
                return StableSnapshot()

            def process_tree_rss_bytes(self, pid: int) -> int:
                return self.real.process_tree_rss_bytes(pid)

            def lower_process_tree_priority(self, pid: int) -> None:
                self.real.lower_process_tree_priority(pid)

        self._probe_patcher = patch(
            "simulation_ready_cad_workflow.validation_pipeline.runtime."
            "system_probe.SystemProbe",
            StableProbe,
        )
        self._probe_patcher.start()
        self.addCleanup(self._probe_patcher.stop)
        self._geometry_patcher = patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "rules_manifest.inspect_step_geometry",
            return_value=_known_test_step_inspection(),
        )
        self._geometry_patcher.start()
        self.addCleanup(self._geometry_patcher.stop)

    def test_complete_package_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.ACCEPTED)
        self.assertEqual(report.errors, [])
        self.assertEqual(report.warnings, [])
        self.assertTrue(report.capabilities["can_enter_physical_preflight"])
        self.assertTrue(
            report.capabilities["can_attempt_reduced_certification"]
        )
        self.assertTrue(
            report.capabilities["can_run_whole_machine_task_test"]
        )

    def test_typed_acceptance_contract_is_bound_during_intake(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            document = _read_json(requirements)
            document["schema_version"] = "0.2"
            document["prompt"] = (
                "Build a linear stage whose carriage moves at least 40 mm "
                "along the guide axis."
            )
            document["tasks"][0].pop("success_criteria")
            document["tasks"][0]["acceptance_requirements"] = [
                {
                    "requirement_id": "carriage_travel",
                    "prompt_clause": "carriage moves at least 40 mm",
                    "target_role": "carriage",
                    "input_role": "carriage drive force",
                    "metric_id": "signed_displacement_m",
                    "test_program": "task_range_roundtrip",
                    "test_load_case_id": "task_envelope_roundtrip",
                    "target": {
                        "comparator": "at_least",
                        "value": 0.04,
                        "unit": "m",
                    },
                    "allow_direct_output_actuation": True,
                    "required_branches": [
                        "physical",
                        "ideal_reference",
                    ],
                }
            ]
            _write_json(requirements, document)
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["requirements_ref"]["sha256"] = _sha256(requirements)
            _write_json(manifest_path, manifest)

            report = validate_design_package(package, requirements)

        self.assertEqual(report.status, IntakeStatus.ACCEPTED)
        self.assertTrue(
            report.capabilities["can_issue_typed_task_certificate"]
        )
        summary = report.package_summary["acceptance_contract"]
        self.assertTrue(summary["certificate_authority"])
        self.assertEqual(summary["contract"]["status"], "PASS")
        self.assertEqual(
            summary["semantic_binding"]["bindings"][0]["target"][
                "observable_id"
            ],
            "carriage_pose",
        )

    def test_schema_02_without_typed_acceptance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            document = _read_json(requirements)
            document["schema_version"] = "0.2"
            document["tasks"][0].pop("success_criteria")
            _write_json(requirements, document)
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["requirements_ref"]["sha256"] = _sha256(requirements)
            _write_json(manifest_path, manifest)

            report = validate_design_package(package, requirements)

        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertIn(
            "TYPED_ACCEPTANCE_REQUIRED",
            {issue.code for issue in report.errors},
        )
        self.assertFalse(
            report.capabilities["can_issue_typed_task_certificate"]
        )

    def test_single_source_cad_declaration_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["single_source_cad"] = {
                "policy": "required_v1",
                "authoritative_artifact": "assembly.step",
                "derived_assets": "hash_bound_and_regenerated",
            }
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.ACCEPTED)
        self.assertNotIn(
            "FIELD_UNRECOGNIZED", {issue.code for issue in report.warnings}
        )

    def test_invalid_single_source_derivation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["single_source_cad"] = {
                "policy": "required_v1",
                "authoritative_artifact": "assembly.step",
                "derived_assets": "maintained_separately",
            }
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertIn(
            "SINGLE_SOURCE_CAD_DERIVATION_INVALID",
            {issue.code for issue in report.errors},
        )

    def test_complete_package_real_kernel_path_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.inspect_step_geometry",
                side_effect=inspect_step_geometry_isolated,
            ):
                report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.ACCEPTED)
        self.assertEqual(report.errors, [])
        self.assertTrue(report.capabilities["can_enter_physical_preflight"])

    def test_optional_omissions_return_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(
                Path(directory), warnings=True
            )
            report = validate_design_package(package, requirements)
        self.assertEqual(
            report.status, IntakeStatus.ACCEPTED_WITH_WARNINGS
        )
        self.assertEqual(report.errors, [])
        codes = {issue.code for issue in report.warnings}
        self.assertTrue(
            {
                "NATIVE_ASSEMBLY_MISSING",
                "PART_MATERIAL_MISSING",
                "INPUT_RANGE_MISSING",
                "REDUCED_MODEL_CLAIM_MISSING",
                "DECLARED_NOMINAL_TESTS_MISSING",
            }.issubset(codes)
        )

    def test_ai_cannot_disable_world_gravity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            contract = _read_json(assembly_path)
            contract["world_frame"]["gravity_m_s2"] = [0.0, 0.0, 0.0]
            _write_json(assembly_path, contract)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertIn(
            "WORLD_GRAVITY_INVALID",
            {issue.code for issue in report.errors},
        )

    def test_unrecognized_ai_field_returns_actionable_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["assembly_flie"] = "assembly.step"
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertEqual(
            report.status,
            IntakeStatus.ACCEPTED_WITH_WARNINGS,
        )
        self.assertEqual(report.errors, [])
        issues = [
            issue
            for issue in report.warnings
            if issue.code == "FIELD_UNRECOGNIZED"
        ]
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].json_path, "$.assembly_flie")
        self.assertEqual(issues[0].actual, "assembly_flie")

    def test_nested_unrecognized_ai_fields_report_exact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))

            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["requirements_ref"]["sha265"] = "typo"
            _write_json(manifest_path, manifest)

            assembly_path = package / "assembly_contract.json"
            assembly = _read_json(assembly_path)
            assembly["parts"][0]["rigd_group_id"] = "group_base"
            assembly["parts"][0]["material"]["densitty_kg_m3"] = 1
            _write_json(assembly_path, assembly)

            capsule_path = (
                package / "mechanism_capsules" / "linear_stage.json"
            )
            capsule = _read_json(capsule_path)
            capsule["input_ports"][0]["frameid"] = "slider_input"
            capsule["reduced_model_claim"]["axix_abs"] = [1, 0, 0]
            _write_json(capsule_path, capsule)

            coverage_path = package / "mechanism_coverage.json"
            coverage = _read_json(coverage_path)
            coverage["coverage"][0]["classfication"] = "static_context"
            _write_json(coverage_path, coverage)

            mapping_path = package / "machine_task_mapping.json"
            mapping = _read_json(mapping_path)
            mapping["task_mappings"][0]["implementation_cliam"] = "typo"
            _write_json(mapping_path, mapping)
            _refresh_artifact_hashes(package)

            report = validate_design_package(package, requirements)

        paths = {
            issue.json_path
            for issue in report.warnings
            if issue.code == "FIELD_UNRECOGNIZED"
        }
        self.assertEqual(report.status, IntakeStatus.ACCEPTED_WITH_WARNINGS)
        self.assertEqual(report.errors, [])
        self.assertTrue(
            {
                "$.requirements_ref.sha265",
                "$.parts[0].rigd_group_id",
                "$.parts[0].material.densitty_kg_m3",
                "$.input_ports[0].frameid",
                "$.reduced_model_claim.axix_abs",
                "$.coverage[0].classfication",
                "$.task_mappings[0].implementation_cliam",
            }.issubset(paths)
        )

    def test_authority_fields_cannot_hide_behind_key_style(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = (
                package / "mechanism_capsules" / "linear_stage.json"
            )
            capsule = _read_json(capsule_path)
            capsule["FinalStatus"] = "PASS"
            capsule["verificationStatus"] = "VERIFIED"
            capsule["expected_behavior"]["passFail"] = "PASS"
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        codes = [issue.code for issue in report.errors]
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertGreaterEqual(
            codes.count("AI_AUTHORITY_FIELD_FORBIDDEN"),
            2,
        )
        self.assertIn("AI_VERIFICATION_CLAIM_FORBIDDEN", codes)

    def test_rejection_aggregates_independent_problems(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = (
                package / "mechanism_capsules" / "linear_stage.json"
            )
            capsule = _read_json(capsule_path)
            capsule["verdict"] = "PASS"
            capsule["input_ports"][0]["frame_id"] = "missing_frame"
            _write_json(capsule_path, capsule)
            coverage_path = package / "mechanism_coverage.json"
            coverage = _read_json(coverage_path)
            coverage["coverage"][1]["classification"] = "static"
            coverage["coverage"][1]["capsule_ids"] = []
            _write_json(coverage_path, coverage)
            mapping_path = package / "machine_task_mapping.json"
            mapping = _read_json(mapping_path)
            mapping["task_mappings"][0]["inputs"][0]["port_id"] = "bad"
            _write_json(mapping_path, mapping)
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["artifacts"][0]["sha256"] = "0" * 64
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertGreaterEqual(len(report.errors), 7)
        codes = {issue.code for issue in report.errors}
        self.assertTrue(
            {
                "AI_AUTHORITY_FIELD_FORBIDDEN",
                "CAPSULE_FRAME_UNKNOWN",
                "DYNAMIC_PART_NOT_COVERED",
                "CAPSULE_MEMBERSHIP_NOT_COVERED",
                "TASK_INPUT_UNKNOWN",
                "ARTIFACT_HASH_MISMATCH",
            }.issubset(codes)
        )
        self.assertNotIn("INTERNAL_VALIDATOR_ERROR", codes)
        self.assertTrue(all(issue.hint for issue in report.issues))

    def test_unhashed_extra_package_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            (package / "debug.log").write_text("scratch", encoding="utf-8")
            report = validate_design_package(package, requirements)
        self.assertIn(
            "PACKAGE_FILE_UNDECLARED",
            {issue.code for issue in report.errors},
        )

    def test_step_assembly_must_be_in_artifact_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["artifacts"] = [
                artifact
                for artifact in manifest["artifacts"]
                if artifact["path"] != "assembly.step"
            ]
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertIn("REQUIRED_ARTIFACT_UNHASHED", codes)
        self.assertIn("PACKAGE_FILE_UNDECLARED", codes)

    def test_package_manifest_cannot_hash_itself(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["artifacts"].append(
                {
                    "path": "package.json",
                    "sha256": "0" * 64,
                }
            )
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "PACKAGE_MANIFEST_SELF_HASH_FORBIDDEN",
            {issue.code for issue in report.errors},
        )

    def test_package_hardlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, requirements = _make_package(root)
            external = root / "external.bin"
            external.write_bytes(b"shared")
            linked = package / "hardlinked.bin"
            try:
                os.link(external, linked)
            except OSError as exc:
                self.skipTest(f"hard links unavailable: {exc}")
            report = validate_design_package(package, requirements)
        self.assertIn(
            "PACKAGE_HARDLINK_FORBIDDEN",
            {issue.code for issue in report.errors},
        )

    def test_package_junction_entry_is_rejected_without_traversal(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake.rules_manifest import (
            _inventory_package,
        )

        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            junction = package / "junction-like"
            junction.mkdir(parents=True)
            hidden = junction / "outside-like.bin"
            hidden.write_bytes(b"must not be inventoried")
            ctx = ValidationContext(package, None)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest._is_path_junction",
                side_effect=lambda candidate: candidate == junction,
            ):
                _inventory_package(ctx)

        self.assertIn(
            "PACKAGE_JUNCTION_FORBIDDEN",
            {issue.code for issue in ctx.report.errors},
        )
        self.assertNotIn("junction-like/outside-like.bin", ctx.package_file_paths)

    def test_manifest_path_cannot_traverse_directory_junction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            junction = package / "junction-like"
            junction.mkdir(parents=True)
            target = junction / "assembly.step"
            target.write_bytes(b"placeholder")
            ctx = ValidationContext(package, None)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "context._is_path_junction",
                side_effect=lambda candidate: candidate == junction,
            ):
                resolved = ctx.safe_package_path(
                    "junction-like/assembly.step",
                    manifest_field="assembly_file",
                )

        self.assertIsNone(resolved)
        self.assertIn(
            "PACKAGE_JUNCTION_FORBIDDEN",
            {issue.code for issue in ctx.report.errors},
        )

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["assembly_file"] = "../outside.step"
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertIn(
            "PACKAGE_PATH_TRAVERSAL",
            {issue.code for issue in report.errors},
        )

    def test_fake_step_is_rejected_by_structure_and_kernel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            (package / "assembly.step").write_text(
                """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('PLACEHOLDER'),'2;1');
ENDSEC;
DATA;
#1=PRODUCT('fake','fake','',());
ENDSEC;
END-ISO-10303-21;
""",
                encoding="ascii",
            )
            _refresh_artifact_hashes(package)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.inspect_step_geometry",
                side_effect=inspect_step_geometry_isolated,
            ):
                report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertTrue(
            {
                "STEP_ENTITY_COUNT_TOO_LOW",
                "STEP_GEOMETRY_REPRESENTATION_MISSING",
                "STEP_KERNEL_IMPORT_FAILED",
            }.issubset(codes)
        )

    def test_solid_mapping_and_rigid_role_errors_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            path = package / "assembly_contract.json"
            assembly = _read_json(path)
            assembly["parts"][1]["source_solid_ids"] = [
                "solid_1",
                "solid_99",
            ]
            assembly["parts"][1]["rigid_group_id"] = "group_base"
            assembly["rigid_groups"] = [
                {
                    "rigid_group_id": "group_base",
                    "part_ids": ["base", "carriage"],
                    "connection_evidence": "incorrectly claimed fusion",
                }
            ]
            _write_json(path, assembly)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertTrue(
            {
                "SOURCE_SOLID_ASSIGNED_MORE_THAN_ONCE",
                "SOURCE_SOLID_UNKNOWN",
                "STEP_SOLID_UNASSIGNED",
                "RIGID_GROUP_ROLE_CONFLICT",
            }.issubset(codes)
        )

    def test_non_unit_frames_and_axes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            assembly = _read_json(assembly_path)
            assembly["frames"][0]["orientation_xyzw"] = [0, 0, 0, 2]
            _write_json(assembly_path, assembly)
            capsule_path = (
                package / "mechanism_capsules" / "linear_stage.json"
            )
            capsule = _read_json(capsule_path)
            capsule["input_ports"][0]["direction_abs"] = [2, 0, 0]
            capsule["reduced_model_claim"]["axis_abs"] = [3, 0, 0]
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertTrue(
            {
                "QUATERNION_NOT_UNIT",
                "DIRECTION_NOT_UNIT",
                "AXIS_NOT_UNIT",
            }.issubset(codes)
        )

    def test_part_bounds_mismatch_detects_scale_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            assembly = _read_json(assembly_path)
            assembly["parts"][0]["declared_bounds_m"] = {
                "min": [-50, -30, -30],
                "max": [50, 30, 30],
            }
            _write_json(assembly_path, assembly)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "PART_BOUNDS_MISMATCH",
            {issue.code for issue in report.errors},
        )

    def test_generator_cannot_masquerade_as_native_cad(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            generator = package / "generate.py"
            generator.write_text("# reproducible CAD source\n", encoding="utf-8")
            manifest_path = package / "package.json"
            manifest = _read_json(manifest_path)
            manifest["source_native_assembly_file"] = "generate.py"
            manifest["artifacts"].append(
                {"path": "generate.py", "sha256": _sha256(generator)}
            )
            _write_json(manifest_path, manifest)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "NATIVE_ASSEMBLY_EXTENSION_INVALID",
            {issue.code for issue in report.errors},
        )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            text = assembly_path.read_text(encoding="utf-8")
            text = text.replace(
                '"density_kg_m3": 7850',
                '"density_kg_m3": 7850, "density_kg_m3": 7850',
                1,
            )
            assembly_path.write_text(text, encoding="utf-8")
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "JSON_DUPLICATE_KEY",
            {issue.code for issue in report.errors},
        )

    def test_nonfinite_json_constant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            text = assembly_path.read_text(encoding="utf-8")
            text = text.replace('"density_kg_m3": 7850', '"density_kg_m3": NaN', 1)
            assembly_path.write_text(text, encoding="utf-8")
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "JSON_NUMBER_NONFINITE",
            {issue.code for issue in report.errors},
        )

    def test_overflowed_json_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            text = assembly_path.read_text(encoding="utf-8")
            text = text.replace(
                '"density_kg_m3": 7850',
                '"density_kg_m3": 1e999',
                1,
            )
            assembly_path.write_text(text, encoding="utf-8")
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "JSON_NUMBER_NONFINITE",
            {issue.code for issue in report.errors},
        )

    def test_input_range_unit_must_match_wrench_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = _read_json(capsule_path)
            capsule["input_ports"][0]["declared_range"]["unit"] = "kg"
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "INPUT_RANGE_UNIT_MISMATCH",
            {issue.code for issue in report.errors},
        )

    def test_combined_wrench_is_not_supported_by_v01_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = _read_json(capsule_path)
            capsule["input_ports"][0]["wrench_type"] = "wrench"
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "WRENCH_TYPE_INVALID",
            {issue.code for issue in report.errors},
        )

    def test_renamed_text_file_cannot_masquerade_as_fcstd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            native_path = package / "assembly.FCStd"
            native_path.write_bytes(b"not a cad archive" * 20)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "NATIVE_ASSEMBLY_CONTAINER_INVALID",
            {issue.code for issue in report.errors},
        )

    def test_reduced_claim_requires_unverified_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = _read_json(capsule_path)
            capsule["reduced_model_claim"].pop("verification_status")
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "REDUCED_MODEL_VERIFICATION_STATUS_REQUIRED",
            {issue.code for issue in report.errors},
        )

    def test_reduced_claim_family_spec_must_stay_inside_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = _read_json(capsule_path)
            capsule["reduced_model_claim"]["family"] = "compound"
            capsule["reduced_model_claim"]["family_spec_file"] = (
                "../../outside.family.json"
            )
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)

        self.assertIn(
            "PACKAGE_PATH_TRAVERSAL",
            {issue.code for issue in report.errors},
        )

    def test_spherical_reduced_claim_does_not_require_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = _read_json(capsule_path)
            claim = capsule["reduced_model_claim"]
            claim["family"] = "spherical"
            claim.pop("axis_abs")
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.ACCEPTED)
        self.assertTrue(
            report.capabilities["can_attempt_reduced_certification"]
        )

    def test_nominal_test_must_reference_known_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = _read_json(capsule_path)
            capsule["declared_nominal_tests"][0]["input_port_id"] = "missing"
            _write_json(capsule_path, capsule)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "NOMINAL_TEST_INPUT_UNKNOWN",
            {issue.code for issue in report.errors},
        )

    def test_connection_type_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            assembly_path = package / "assembly_contract.json"
            assembly = _read_json(assembly_path)
            assembly["connections"][0].pop("connection_type")
            _write_json(assembly_path, assembly)
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "CONNECTION_TYPE_REQUIRED",
            {issue.code for issue in report.errors},
        )

    def test_kernel_warning_blocks_physical_preflight_capability(self) -> None:
        inspection = StepGeometryInspection(
            kernel_available=False,
            transferred=False,
            error="OCP intentionally unavailable in test",
        )
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.inspect_step_geometry",
                return_value=inspection,
            ):
                report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.ACCEPTED_WITH_WARNINGS)
        self.assertFalse(
            report.capabilities["can_enter_physical_preflight"]
        )
        self.assertIn(
            "STEP_GEOMETRY_NOT_KERNEL_VERIFIED",
            report.capabilities["physical_preflight_blockers"],
        )

    def test_step_kernel_snapshot_must_match_declared_artifact_hash(self) -> None:
        known = _known_test_step_inspection()
        mismatched = StepGeometryInspection(
            kernel_available=known.kernel_available,
            transferred=known.transferred,
            solids=known.solids,
            source_sha256="0" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.inspect_step_geometry",
                return_value=mismatched,
            ):
                report = validate_design_package(package, requirements)

        self.assertIn(
            "STEP_KERNEL_SNAPSHOT_HASH_MISMATCH",
            {issue.code for issue in report.errors},
        )
        self.assertFalse(report.capabilities["can_enter_physical_preflight"])

    def test_invalid_brep_blocks_physical_preflight_capability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            inspection = inspect_step_geometry(package / "assembly.step")
            self.assertTrue(inspection.transferred)
            solids = tuple(
                StepSolid(
                    solid_id=solid.solid_id,
                    valid_brep=index != 0,
                    bounds_mm=solid.bounds_mm,
                    size_mm=solid.size_mm,
                )
                for index, solid in enumerate(inspection.solids)
            )
            invalid_inspection = StepGeometryInspection(
                kernel_available=True,
                transferred=True,
                solids=solids,
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.inspect_step_geometry",
                return_value=invalid_inspection,
            ):
                report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.ACCEPTED_WITH_WARNINGS)
        self.assertFalse(
            report.capabilities["can_enter_physical_preflight"]
        )
        self.assertIn(
            "STEP_BREP_INVALID",
            report.capabilities["physical_preflight_blockers"],
        )

    def test_oversized_json_is_rejected_before_parsing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "large.json"
            path.write_text('{"payload":"' + ("x" * 128) + '"}', encoding="utf-8")
            ctx = ValidationContext(root, None)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "context.MAX_JSON_BYTES",
                32,
            ):
                value = ctx.load_json(
                    path,
                    required=True,
                    label="test JSON",
                )
        self.assertIsNone(value)
        self.assertIn(
            "JSON_FILE_TOO_LARGE",
            {issue.code for issue in ctx.report.errors},
        )

    def test_deep_json_is_rejected_without_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "deep.json"
            depth = 130
            path.write_text(
                '{"nested":' * depth + "0" + "}" * depth,
                encoding="utf-8",
            )
            ctx = ValidationContext(root, None)
            value = ctx.load_json(path, required=True, label="deep JSON")
        self.assertIsNone(value)
        codes = {issue.code for issue in ctx.report.errors}
        self.assertIn("JSON_NESTING_TOO_DEEP", codes)
        self.assertNotIn("INTERNAL_VALIDATOR_ERROR", codes)

    def test_collection_limit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = ValidationContext(Path(directory), None)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "context.MAX_LIST_ITEMS",
                2,
            ):
                value = ctx.require_list(
                    {"items": [1, 2, 3]},
                    "items",
                    file="test.json",
                )
        self.assertIsNone(value)
        self.assertIn(
            "LIST_TOO_LARGE",
            {issue.code for issue in ctx.report.errors},
        )

    def test_package_file_count_limit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.MAX_PACKAGE_FILES",
                1,
            ):
                report = validate_design_package(package, requirements)
        self.assertIn(
            "PACKAGE_FILE_COUNT_EXCEEDED",
            {issue.code for issue in report.errors},
        )
        self.assertIn(
            "ARTIFACT_INTEGRITY_SKIPPED",
            {issue.code for issue in report.warnings},
        )

    def test_oversized_step_is_rejected_before_kernel_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.MAX_STEP_BYTES",
                100,
            ):
                report = validate_design_package(package, requirements)
        self.assertIn(
            "ASSEMBLY_FILE_TOO_LARGE",
            {issue.code for issue in report.errors},
        )
        self.assertNotIn(
            "INTERNAL_VALIDATOR_ERROR",
            {issue.code for issue in report.errors},
        )

    def test_fcstd_archive_entry_limit_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            native_path = package / "assembly.FCStd"
            with zipfile.ZipFile(native_path, "a", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("extra.txt", "extra")
            _refresh_artifact_hashes(package)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.MAX_FCSTD_ENTRIES",
                1,
            ):
                report = validate_design_package(package, requirements)
        self.assertIn(
            "NATIVE_ASSEMBLY_ARCHIVE_ENTRY_LIMIT",
            {issue.code for issue in report.errors},
        )

    def test_fcstd_directory_limit_blocks_zip_parser(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            rules_manifest,
        )

        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with (
                patch.object(
                    rules_manifest,
                    "MAX_FCSTD_CENTRAL_DIRECTORY_BYTES",
                    1,
                ),
                patch.object(
                    rules_manifest.zipfile,
                    "ZipFile",
                    side_effect=AssertionError(
                        "ZipFile must not run past the directory guard"
                    ),
                ),
            ):
                report = validate_design_package(package, requirements)
        self.assertIn(
            "NATIVE_ASSEMBLY_ARCHIVE_DIRECTORY_LIMIT",
            {issue.code for issue in report.errors},
        )
        self.assertNotIn(
            "INTERNAL_VALIDATOR_ERROR",
            {issue.code for issue in report.errors},
        )

    def test_fcstd_unsafe_archive_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            native_path = package / "assembly.FCStd"
            with zipfile.ZipFile(native_path, "a", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("../outside.txt", "unsafe")
            _refresh_artifact_hashes(package)
            report = validate_design_package(package, requirements)
        self.assertIn(
            "NATIVE_ASSEMBLY_ARCHIVE_PATH_UNSAFE",
            {issue.code for issue in report.errors},
        )

    def test_requirements_must_be_outside_ai_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            inside = package / "requirements.json"
            inside.write_bytes(requirements.read_bytes())
            original = ValidationContext.load_json

            def guarded_load(context, path, **kwargs):
                if Path(path).resolve() == inside.resolve():
                    raise AssertionError(
                        "in-package requirements must not be parsed"
                    )
                return original(context, path, **kwargs)

            with patch.object(ValidationContext, "load_json", guarded_load):
                report = validate_design_package(package, inside)
        self.assertIn(
            "REQUIREMENTS_FILE_NOT_EXTERNAL",
            {issue.code for issue in report.errors},
        )

    def test_cli_refuses_report_inside_package(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            output = package / "intake_report.json"
            stream = StringIO()
            with redirect_stderr(stream):
                code = main(
                    [
                        "--package",
                        str(package),
                        "--requirements",
                        str(requirements),
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertFalse(output.exists())
        self.assertIn("must be outside", stream.getvalue())

    def test_package_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            target = package / "real-note.txt"
            target.write_text("note", encoding="utf-8")
            link = package / "linked-note.txt"
            try:
                link.symlink_to(target)
            except OSError as exc:
                self.skipTest(f"symbolic links unavailable: {exc}")
            report = validate_design_package(package, requirements)
        self.assertIn(
            "PACKAGE_SYMLINK_FORBIDDEN",
            {issue.code for issue in report.errors},
        )

    def test_hash_read_error_is_reported_without_internal_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "artifact.bin"
            path.write_bytes(b"data")
            ctx = ValidationContext(root, None)
            with patch.object(
                Path,
                "open",
                side_effect=PermissionError("denied for test"),
            ):
                digest = ctx.sha256_file(path)
        self.assertIsNone(digest)
        codes = {issue.code for issue in ctx.report.errors}
        self.assertIn("FILE_HASH_READ_ERROR", codes)
        self.assertNotIn("INTERNAL_VALIDATOR_ERROR", codes)

    def test_isolated_step_inspection_matches_direct_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            path = package / "assembly.step"
            source_digest = _sha256(path)
            direct = inspect_step_geometry(path)
            isolated = inspect_step_geometry_isolated(
                path,
                timeout_seconds=30.0,
            )
        self.assertTrue(isolated.kernel_available)
        self.assertTrue(isolated.transferred)
        self.assertEqual(isolated.source_sha256, source_digest)
        self.assertEqual(
            [solid.to_dict() for solid in direct.solids],
            [solid.to_dict() for solid in isolated.solids],
        )

    def test_isolated_step_inspection_timeout_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            inspection = inspect_step_geometry_isolated(
                package / "assembly.step",
                timeout_seconds=0.000001,
            )
        self.assertFalse(inspection.kernel_available)
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("timed out", inspection.error or "")

    def test_isolated_step_temp_failure_is_structured(self) -> None:
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "step_geometry.tempfile.TemporaryDirectory",
            side_effect=OSError("temp unavailable"),
        ):
            inspection = inspect_step_geometry_isolated(Path("unused.step"))
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("worker directory", inspection.error or "")

    def test_isolated_step_worker_environment_removes_case_collisions(
        self,
    ) -> None:
        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class SafeProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return SafeSnapshot()

        captured_environment: dict[str, str] = {}

        def fail_launch(*args: object, **kwargs: object) -> object:
            captured_environment.update(kwargs["env"])
            raise OSError("launch stopped for test")

        inherited_environment = {
            "omp_num_threads": "99",
            "temp": "outside",
            "pythonpath": "inherited",
            "UNCHANGED": "yes",
        }
        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=SafeProbe(),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry._snapshot_step_source",
                return_value=("a" * 64, None),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.os.environ",
                inherited_environment,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen",
                side_effect=fail_launch,
            ),
        ):
            inspection = inspect_step_geometry_isolated(Path("unused.step"))

        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertEqual(
            [
                name
                for name in captured_environment
                if name.casefold() == "omp_num_threads"
            ],
            ["OMP_NUM_THREADS"],
        )
        self.assertEqual(
            [
                name
                for name in captured_environment
                if name.casefold() == "temp"
            ],
            ["TEMP"],
        )
        self.assertEqual(captured_environment["OMP_NUM_THREADS"], "1")
        self.assertNotEqual(captured_environment["TEMP"], "outside")
        self.assertTrue(
            captured_environment["PYTHONPATH"].endswith(
                os.pathsep + "inherited"
            )
        )
        self.assertEqual(captured_environment["UNCHANGED"], "yes")

    def test_isolated_step_rejects_oversized_windows_environment(
        self,
    ) -> None:
        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class SafeProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return SafeSnapshot()

        inherited_environment = {"BIG": "x" * 32_768}
        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=SafeProbe(),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry._snapshot_step_source",
                return_value=("a" * 64, None),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.os.environ",
                inherited_environment,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry._is_windows_platform",
                return_value=True,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen"
            ) as popen,
        ):
            inspection = inspect_step_geometry_isolated(Path("unused.step"))

        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn(
            "Windows OCCT worker environment block exceeds 32767",
            inspection.error or "",
        )
        popen.assert_not_called()

    def test_occt_cleanup_reports_live_descendant(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            step_geometry,
        )

        class CompletedProcess:
            pid = 1

            def poll(self) -> int:
                return 0

        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.terminate_process_tree",
                return_value=(99,),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.process_exists",
                return_value=True,
            ),
            patch.object(step_geometry.os, "name", "posix"),
            patch.object(step_geometry.os, "killpg", create=True) as killpg,
            self.assertRaisesRegex(
                RuntimeError,
                r"surviving PIDs: \[99\]",
            ),
        ):
            step_geometry._terminate_process_tree(CompletedProcess())

        killpg.assert_called_once_with(1, step_geometry.signal.SIGTERM)

    def test_isolated_step_monitor_failure_reaps_worker(self) -> None:
        class FailingProbe:
            def __init__(self) -> None:
                self.samples = 0

            def lower_process_tree_priority(self, pid: int) -> None:
                return None

            def sample(self, interval_seconds: float = 0.0) -> object:
                self.samples += 1
                if self.samples == 1:
                    class SafeSnapshot:
                        cpu_percent = 0.0
                        memory_percent = 50.0
                        available_memory_gib = 8.0
                        free_disk_gib = 100.0

                    return SafeSnapshot()
                raise RuntimeError("monitor failed for test")

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=FailingProbe(),
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("monitoring failed", inspection.error or "")

    def test_isolated_step_uses_native_process_containment(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.runtime.process_containment import (
            ProcessContainment as RealProcessContainment,
        )

        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class MeasuredProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return SafeSnapshot()

            def process_tree_rss_bytes(self, pid: int) -> int:
                return 1024

            def lower_process_tree_priority(self, pid: int) -> None:
                return None

        attached_limits: list[int | None] = []
        close_count = 0

        class RecordingContainment(RealProcessContainment):
            def attach_and_resume(
                self,
                process: subprocess.Popen[bytes],
                *,
                max_job_memory_bytes: int | None,
            ) -> None:
                attached_limits.append(max_job_memory_bytes)
                super().attach_and_resume(
                    process,
                    max_job_memory_bytes=max_job_memory_bytes,
                )

            def close(self) -> None:
                nonlocal close_count
                try:
                    super().close()
                finally:
                    close_count += 1

        captured_creationflags: list[int] = []
        real_popen = subprocess.Popen

        def launch_short_worker(*args: object, **kwargs: object) -> object:
            captured_creationflags.append(int(kwargs.get("creationflags", 0)))
            return real_popen(
                [sys.executable, "-c", "pass"],
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with (
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline.runtime."
                    "system_probe.SystemProbe",
                    return_value=MeasuredProbe(),
                ),
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline.intake."
                    "step_geometry.ProcessContainment",
                    RecordingContainment,
                ),
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline.intake."
                    "step_geometry.subprocess.Popen",
                    side_effect=launch_short_worker,
                ),
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                )

        self.assertFalse(inspection.transferred)
        self.assertEqual(
            attached_limits,
            [int(DEFAULT_MAX_KERNEL_RSS_GIB * 1024**3)],
        )
        self.assertGreaterEqual(close_count, 1)
        self.assertEqual(len(captured_creationflags), 1)
        if os.name == "nt":
            self.assertTrue(captured_creationflags[0] & 0x00000004)

    def test_isolated_step_containment_failure_reaps_suspended_worker(self) -> None:
        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class MeasuredProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return SafeSnapshot()

            def process_tree_rss_bytes(self, pid: int) -> int:
                return 1024

            def lower_process_tree_priority(self, pid: int) -> None:
                return None

        class FailingContainment:
            def __enter__(self) -> object:
                return self

            def __exit__(
                self,
                exc_type: object,
                exc: object,
                traceback: object,
            ) -> None:
                return None

            def attach_and_resume(
                self,
                process: subprocess.Popen[bytes],
                *,
                max_job_memory_bytes: int | None,
            ) -> None:
                raise OSError("containment attach failed for test")

        launched: list[subprocess.Popen[object]] = []
        real_popen = subprocess.Popen

        def launch_sleeping_worker(*args: object, **kwargs: object) -> object:
            process = real_popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                **kwargs,
            )
            launched.append(process)
            return process

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with (
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline.runtime."
                    "system_probe.SystemProbe",
                    return_value=MeasuredProbe(),
                ),
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline.intake."
                    "step_geometry.ProcessContainment",
                    FailingContainment,
                ),
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline.intake."
                    "step_geometry.subprocess.Popen",
                    side_effect=launch_sleeping_worker,
                ),
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                )

        self.assertEqual(len(launched), 1)
        self.assertIsNotNone(launched[0].poll())
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("containment attach failed", inspection.error or "")

    def test_isolated_step_resource_preflight_defers_without_launch(self) -> None:
        class LowSnapshot:
            cpu_percent = 0.0
            memory_percent = 92.0
            available_memory_gib = 1.0
            free_disk_gib = 100.0

        class LowProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return LowSnapshot()

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.runtime."
            "system_probe.SystemProbe",
            return_value=LowProbe(),
        ), patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "step_geometry.subprocess.Popen"
        ) as popen:
            inspection = inspect_step_geometry_isolated(Path("unused.step"))
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("resource preflight", inspection.error or "")
        popen.assert_not_called()

    def test_isolated_step_cpu_preflight_defers_without_launch(self) -> None:
        class BusySnapshot:
            cpu_percent = 99.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class BusyProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return BusySnapshot()

        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=BusyProbe(),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen"
            ) as popen,
        ):
            inspection = inspect_step_geometry_isolated(Path("unused.step"))
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("99.0% CPU", inspection.error or "")
        popen.assert_not_called()

    def test_isolated_step_start_memory_covers_worker_limit(self) -> None:
        class TightSnapshot:
            cpu_percent = 0.0
            memory_percent = 70.0
            available_memory_gib = 1.4
            free_disk_gib = 100.0

        class TightProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return TightSnapshot()

        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=TightProbe(),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen"
            ) as popen,
        ):
            inspection = inspect_step_geometry_isolated(Path("unused.step"))
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("at least 1.52 GiB", inspection.error or "")
        popen.assert_not_called()

    def test_isolated_step_custom_worker_limit_keeps_emergency_reserve(self) -> None:
        class TightSnapshot:
            cpu_percent = 0.0
            memory_percent = 70.0
            available_memory_gib = 4.4
            free_disk_gib = 100.0

        class TightProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return TightSnapshot()

        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=TightProbe(),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen"
            ) as popen,
        ):
            inspection = inspect_step_geometry_isolated(
                Path("unused.step"),
                max_process_tree_rss_gib=3.0,
                critical_available_memory_gib=1.5,
                preflight_available_memory_gib=1.5,
            )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("at least 4.52 GiB", inspection.error or "")
        popen.assert_not_called()

    def test_isolated_step_monitors_actual_worker_temp_disk(self) -> None:
        captured: dict[str, object] = {}

        class LowDiskSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 0.75

        class RecordingProbe:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            def sample(self, interval_seconds: float = 0.0) -> object:
                return LowDiskSnapshot()

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.runtime."
            "system_probe.SystemProbe",
            RecordingProbe,
        ), patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "step_geometry.subprocess.Popen"
        ) as popen:
            inspection = inspect_step_geometry_isolated(
                Path("unused.step"),
                preflight_free_disk_gib=1.0,
                critical_free_disk_gib=0.5,
            )

        self.assertEqual(
            Path(captured["disk_path"]).resolve(),
            Path(tempfile.gettempdir()).resolve(),
        )
        self.assertEqual(inspection.failure_category, "infrastructure")
        popen.assert_not_called()

    def test_isolated_step_low_disk_preflight_defers_without_launch(self) -> None:
        class LowDiskSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 0.75

        class LowDiskProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return LowDiskSnapshot()

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.runtime."
            "system_probe.SystemProbe",
            return_value=LowDiskProbe(),
        ), patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "step_geometry.subprocess.Popen"
        ) as popen:
            inspection = inspect_step_geometry_isolated(
                Path("unused.step"),
                preflight_free_disk_gib=1.0,
                critical_free_disk_gib=0.5,
            )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("disk free", inspection.error or "")
        popen.assert_not_called()

    def test_isolated_step_critical_disk_space_stops_worker(self) -> None:
        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class CriticalDiskSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 0.25

        class DiskPressureProbe:
            def __init__(self) -> None:
                self.samples = 0

            def sample(self, interval_seconds: float = 0.0) -> object:
                self.samples += 1
                return (
                    SafeSnapshot()
                    if self.samples == 1
                    else CriticalDiskSnapshot()
                )

            def process_tree_rss_bytes(self, pid: int) -> int:
                return 1024

            def lower_process_tree_priority(self, pid: int) -> None:
                return None

        real_popen = subprocess.Popen

        def launch_sleeping_worker(*args: object, **kwargs: object) -> object:
            return real_popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=DiskPressureProbe(),
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen",
                side_effect=launch_sleeping_worker,
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                    critical_free_disk_gib=0.5,
                    critical_samples=1,
                )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("disk free", inspection.error or "")

    def test_isolated_step_unmeasurable_rss_stops_worker(self) -> None:
        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class BlindProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return SafeSnapshot()

            def process_tree_rss_bytes(self, pid: int) -> int:
                return 0

            def lower_process_tree_priority(self, pid: int) -> None:
                return None

        real_popen = subprocess.Popen

        def launch_sleeping_worker(*args: object, **kwargs: object) -> object:
            return real_popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=BlindProbe(),
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen",
                side_effect=launch_sleeping_worker,
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                    critical_samples=1,
                )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("could not be measured", inspection.error or "")

    def test_isolated_step_oversized_log_stops_worker(self) -> None:
        class SafeSnapshot:
            cpu_percent = 0.0
            memory_percent = 50.0
            available_memory_gib = 8.0
            free_disk_gib = 100.0

        class MeasuredProbe:
            def sample(self, interval_seconds: float = 0.0) -> object:
                return SafeSnapshot()

            def process_tree_rss_bytes(self, pid: int) -> int:
                return 1024

            def lower_process_tree_priority(self, pid: int) -> None:
                return None

        real_popen = subprocess.Popen
        child_code = (
            "import sys; "
            "sys.stdout.write('x'*2048); sys.stdout.flush()"
        )

        def launch_noisy_worker(*args: object, **kwargs: object) -> object:
            return real_popen(
                [sys.executable, "-c", child_code],
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.runtime."
                "system_probe.SystemProbe",
                return_value=MeasuredProbe(),
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.MAX_WORKER_LOG_BYTES",
                1024,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen",
                side_effect=launch_noisy_worker,
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("log limit", inspection.error or "")

    def test_isolated_step_rejects_unsafe_resource_overrides(self) -> None:
        cases = (
            {"timeout_seconds": 901.0},
            {"max_process_tree_rss_gib": 6.1},
            {"preflight_cpu_percent": 95.1},
            {
                "preflight_memory_percent": 92.1,
                "critical_memory_percent": 94.0,
            },
            {"critical_memory_percent": 96.1},
            {"preflight_available_memory_gib": 0.9},
            {"critical_available_memory_gib": 0.4},
            {"preflight_free_disk_gib": 0.9},
            {"critical_free_disk_gib": 0.4},
            {"preflight_free_disk_gib": 1.0, "critical_free_disk_gib": 2.0},
            {"critical_samples": 31},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    inspect_step_geometry_isolated(
                        Path("unused.step"),
                        **kwargs,
                    )

    def test_infrastructure_deferred_kernel_check_is_not_geometry_error(
        self,
    ) -> None:
        deferred = StepGeometryInspection(
            kernel_available=True,
            transferred=False,
            error="synthetic resource guard",
            failure_category="infrastructure",
        )
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "rules_manifest.inspect_step_geometry",
                return_value=deferred,
            ):
                report = validate_design_package(package, requirements)
        warning_codes = {issue.code for issue in report.warnings}
        error_codes = {issue.code for issue in report.errors}
        self.assertIn("STEP_KERNEL_VALIDATION_DEFERRED", warning_codes)
        self.assertNotIn("STEP_KERNEL_IMPORT_FAILED", error_codes)
        self.assertFalse(report.capabilities["can_enter_physical_preflight"])
        self.assertIn(
            "STEP_GEOMETRY_NOT_KERNEL_VERIFIED",
            report.capabilities["physical_preflight_blockers"],
        )

    def test_isolated_step_rejects_duplicate_worker_result_keys(
        self,
    ) -> None:
        real_popen = subprocess.Popen
        payload = (
            '{"kernel_available":true,'
            '"kernel_available":false,'
            '"transferred":false,'
            '"solids":[],'
            '"error":"ambiguous",'
            '"failure_category":"infrastructure"}'
        )

        def launch_ambiguous_worker(
            command: list[str],
            **kwargs: object,
        ) -> object:
            output = Path(command[command.index("--output") + 1])
            child_code = (
                "from pathlib import Path; "
                f"Path({str(output)!r}).write_text("
                f"{payload!r}, encoding='utf-8')"
            )
            return real_popen(
                [sys.executable, "-c", child_code],
                **kwargs,
            )

        with tempfile.TemporaryDirectory() as directory:
            package, _ = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "step_geometry.subprocess.Popen",
                side_effect=launch_ambiguous_worker,
            ):
                inspection = inspect_step_geometry_isolated(
                    package / "assembly.step",
                    timeout_seconds=30.0,
                )
        self.assertFalse(inspection.transferred)
        self.assertEqual(inspection.failure_category, "infrastructure")
        self.assertIn("duplicate OCCT worker result key", inspection.error or "")

    def test_worker_result_rejects_inconsistent_solid_inventory(self) -> None:
        valid_solid = {
            "solid_id": "solid_1",
            "valid_brep": True,
            "bounds_mm": {
                "min": [0.0, 0.0, 0.0],
                "max": [1.0, 2.0, 3.0],
            },
            "size_mm": [1.0, 2.0, 3.0],
        }
        bad_size = dict(valid_solid)
        bad_size["size_mm"] = [1.0, 2.0, 4.0]
        bad_id = dict(valid_solid)
        bad_id["solid_id"] = "solid_2"
        cases = (
            {
                "kernel_available": True,
                "transferred": True,
                "solids": [bad_size],
                "error": None,
            },
            {
                "kernel_available": True,
                "transferred": True,
                "solids": [bad_id],
                "error": None,
            },
            {
                "kernel_available": False,
                "transferred": True,
                "solids": [valid_solid],
                "error": None,
            },
            {
                "kernel_available": True,
                "transferred": True,
                "solids": [valid_solid],
                "error": "contradictory",
                "failure_category": "infrastructure",
            },
            {
                "kernel_available": True,
                "transferred": True,
                "solids": [valid_solid],
                "error": None,
                "unexpected": True,
            },
            {
                "kernel_available": True,
                "transferred": False,
                "solids": [],
                "error": None,
            },
            {
                "kernel_available": True,
                "transferred": True,
                "solids": [
                    {
                        **valid_solid,
                        "unexpected": True,
                    }
                ],
                "error": None,
            },
        )
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    StepGeometryInspection.from_dict(payload)

    def test_report_write_is_atomic_handles_partial_write_and_fsyncs(self) -> None:
        report = IntakeReport(
            package_dir="package",
            requirements_file="requirements.json",
        )
        original_write = os.write
        first_write = True

        def partial_once(descriptor: int, data: bytes) -> int:
            nonlocal first_write
            if first_write:
                first_write = False
                return original_write(descriptor, data[:1])
            return original_write(descriptor, data)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "intake_report.json"
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "models.os.write",
                side_effect=partial_once,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "models.os.fsync",
                wraps=os.fsync,
            ) as fsync_mock:
                report.write_json(output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            leftovers = list(output.parent.glob(".*.tmp"))

        self.assertEqual(payload["status"], "ACCEPTED")
        self.assertGreaterEqual(fsync_mock.call_count, 1)
        self.assertEqual(leftovers, [])

    def test_worker_result_read_is_bounded_and_stable(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            step_geometry,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.json"
            path.write_bytes(b"{}")
            before = path.stat()

            class ChangedStat:
                st_size = before.st_size
                st_mtime_ns = before.st_mtime_ns + 1
                st_ino = getattr(before, "st_ino", None)

            with patch.object(
                Path,
                "stat",
                side_effect=(before, ChangedStat()),
            ), self.assertRaisesRegex(ValueError, "changed while being read"):
                step_geometry._read_bounded_worker_result(path)

            path.write_bytes(b"x" * 17)
            with patch.object(
                step_geometry,
                "MAX_WORKER_RESULT_BYTES",
                16,
            ), self.assertRaisesRegex(ValueError, "exceeds size limit"):
                step_geometry._read_bounded_worker_result(path)

    def test_worker_result_rejects_excessive_json_nesting(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            step_geometry,
        )

        depth = step_geometry.MAX_WORKER_RESULT_NESTING + 1
        payload = '{"nested":' * depth + "0" + "}" * depth
        with self.assertRaisesRegex(ValueError, "nesting"):
            step_geometry._strict_worker_result_json(payload)

    def test_report_bounds_adversarial_evidence(self) -> None:
        report = IntakeReport(
            package_dir="package",
            requirements_file="requirements.json",
        )
        report.add(
            "BOUNDED_EVIDENCE",
            Severity.ERROR,
            "test evidence",
            actual={
                "long": "x" * 10_000,
                "items": list(range(250)),
                "not_finite": float("nan"),
            },
        )
        rendered = report.to_json()
        payload = json.loads(rendered)
        actual = payload["issues"][0]["actual"]
        self.assertLess(len(actual["long"]), 5_000)
        self.assertEqual(actual["items"][-1]["_truncated_items"], 150)
        self.assertEqual(actual["not_finite"], "nan")
        self.assertNotIn(": NaN", rendered)

    def test_report_issue_limit_fails_closed_and_counts_suppressed(self) -> None:
        report = IntakeReport(
            package_dir="package",
            requirements_file="requirements.json",
        )
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "models.MAX_REPORTED_ISSUES",
            2,
        ):
            for index in range(4):
                report.add(
                    f"WARNING_{index}",
                    Severity.WARNING,
                    "synthetic warning",
                )
            report.add("LATE_ERROR", Severity.ERROR, "synthetic error")

        payload = report.to_dict()
        summary = payload["summary"]
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertEqual(summary["reported_issue_count"], 3)
        self.assertEqual(summary["suppressed_issue_count"], 3)
        self.assertEqual(summary["warning_count"], 4)
        self.assertEqual(summary["error_count"], 2)
        self.assertEqual(
            summary["suppressed_issue_counts"],
            {"ERROR": 1, "WARNING": 2},
        )
        self.assertIn(
            "REPORT_ISSUE_LIMIT_EXCEEDED",
            {issue["code"] for issue in payload["issues"]},
        )

    def test_report_issue_byte_budget_fails_closed(self) -> None:
        report = IntakeReport(
            package_dir="package",
            requirements_file="requirements.json",
        )
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.intake."
            "models.MAX_REPORTED_ISSUE_BYTES",
            256,
        ):
            report.add(
                "LARGE_WARNING",
                Severity.WARNING,
                "x" * 2_000,
                actual={"payload": "y" * 2_000},
            )

        payload = report.to_dict()
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertEqual(payload["summary"]["suppressed_issue_count"], 1)
        self.assertEqual(
            payload["summary"]["suppressed_issue_counts"],
            {"WARNING": 1},
        )
        self.assertIn(
            "REPORT_ISSUE_LIMIT_EXCEEDED",
            {issue["code"] for issue in payload["issues"]},
        )
        self.assertIn("1 suppressed", report.render_text())

    def test_nul_package_path_is_rejected_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ctx = ValidationContext(Path(directory), None)
            result = ctx.safe_package_path(
                "bad\x00name.json",
                manifest_field="assembly_contract_file",
            )
        self.assertIsNone(result)
        self.assertIn(
            "PACKAGE_PATH_INVALID",
            {issue.code for issue in ctx.report.errors},
        )

    def test_windows_unsafe_package_paths_are_rejected(self) -> None:
        unsafe_paths = (
            "assembly.step:hidden",
            "logs/CON.txt",
            "trailing-dot.",
            "trailing-space ",
        )
        with tempfile.TemporaryDirectory() as directory:
            ctx = ValidationContext(Path(directory), None)
            for value in unsafe_paths:
                with self.subTest(value=value):
                    result = ctx.safe_package_path(
                        value,
                        manifest_field="assembly_file",
                    )
                    self.assertIsNone(result)
        codes = {issue.code for issue in ctx.report.errors}
        self.assertIn("PACKAGE_PATH_WINDOWS_UNSAFE", codes)

    def test_capability_derivation_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.intake."
                "api.derive_capabilities",
                side_effect=RuntimeError("capability failure for test"),
            ):
                report = validate_design_package(package, requirements)
        self.assertEqual(report.status, IntakeStatus.REJECTED)
        self.assertIn(
            "INTERNAL_VALIDATOR_ERROR",
            {issue.code for issue in report.errors},
        )
        self.assertEqual(
            report.checks[-1]["name"],
            "capability_derivation",
        )

    def test_json_change_during_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "document.json"
            path.write_text('{"value":1}', encoding="utf-8")
            before = path.stat()
            after = SimpleNamespace(
                st_size=before.st_size,
                st_mtime_ns=before.st_mtime_ns + 1,
                st_ino=before.st_ino,
            )
            ctx = ValidationContext(root, None)
            with (
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "stat", side_effect=[before, after]),
            ):
                value = ctx.load_json(
                    path,
                    required=True,
                    label="mutable document",
                )
        self.assertIsNone(value)
        self.assertIn(
            "FILE_CHANGED_DURING_VALIDATION",
            {issue.code for issue in ctx.report.errors},
        )

    def test_json_growth_past_limit_during_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "document.json"
            path.write_text("{}", encoding="utf-8")
            before = path.stat()
            after = SimpleNamespace(
                st_size=17,
                st_mtime_ns=before.st_mtime_ns + 1,
                st_ino=before.st_ino,
            )
            ctx = ValidationContext(root, None)
            with (
                patch.object(Path, "is_file", return_value=True),
                patch.object(Path, "stat", side_effect=[before, after]),
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "intake.context.MAX_JSON_BYTES",
                    16,
                ),
            ):
                value = ctx.load_json(
                    path,
                    required=True,
                    label="growing document",
                )
        self.assertIsNone(value)
        self.assertIn(
            "JSON_FILE_TOO_LARGE",
            {issue.code for issue in ctx.report.errors},
        )

    def test_step_structure_scan_handles_cross_chunk_tokens_without_lines(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            rules_manifest,
        )

        payload = (
            b"ISO-10303-21;DATA;"
            b"#12 = MANIFOLD_SOLID_BREP('x',());"
            b"#13=SHAPE_REPRESENTATION('y',());"
            b"ENDSEC;END-ISO-10303-21;"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "single-line.step"
            path.write_bytes(payload)
            with (
                patch.object(rules_manifest, "STEP_SCAN_CHUNK_BYTES", 7),
                patch.object(rules_manifest, "STEP_SCAN_OVERLAP_CHARS", 64),
            ):
                entity_count, markers = rules_manifest._scan_step_structure(
                    path
                )
        self.assertEqual(entity_count, 2)
        self.assertIn("MANIFOLD_SOLID_BREP", markers)
        self.assertIn("SHAPE_REPRESENTATION", markers)

    def test_document_change_before_completion_blocks_preflight(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            api as intake_api,
        )

        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            original_validate_artifacts = intake_api.validate_artifacts

            def validate_then_mutate(ctx: ValidationContext) -> None:
                original_validate_artifacts(ctx)
                manifest = package / "package.json"
                manifest.write_text(
                    manifest.read_text(encoding="utf-8") + " ",
                    encoding="utf-8",
                )

            with patch.object(
                intake_api,
                "validate_artifacts",
                side_effect=validate_then_mutate,
            ):
                report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertIn("DOCUMENT_CHANGED_DURING_VALIDATION", codes)
        self.assertFalse(
            report.capabilities["can_enter_physical_preflight"]
        )
        self.assertIn(
            "phase:document_stability",
            report.capabilities["physical_preflight_blockers"],
        )

    def test_package_layout_change_before_completion_is_rejected(
        self,
    ) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.intake import (
            api as intake_api,
        )

        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            original_validate_task_mapping = intake_api.validate_task_mapping

            def validate_then_add_files(ctx: ValidationContext) -> None:
                original_validate_task_mapping(ctx)
                (package / "late-output").mkdir()
                (package / "late-output" / "debug.log").write_text(
                    "late writer",
                    encoding="utf-8",
                )

            with patch.object(
                intake_api,
                "validate_task_mapping",
                side_effect=validate_then_add_files,
            ):
                report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertIn("PACKAGE_LAYOUT_CHANGED_DURING_VALIDATION", codes)
        self.assertFalse(
            report.capabilities["can_enter_physical_preflight"]
        )
        self.assertIn(
            "phase:package_stability",
            report.capabilities["physical_preflight_blockers"],
        )

    def test_hashed_step_change_after_hashing_is_rejected(self) -> None:
        original_verify = (
            ValidationContext.verify_loaded_documents_unchanged
        )

        def verify_then_mutate(ctx: ValidationContext) -> None:
            original_verify(ctx)
            step = ctx.package_dir / "assembly.step"
            step.write_bytes(step.read_bytes() + b" ")

        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch.object(
                ValidationContext,
                "verify_loaded_documents_unchanged",
                verify_then_mutate,
            ):
                report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertIn("HASHED_FILE_CHANGED_DURING_VALIDATION", codes)
        self.assertFalse(
            report.capabilities["can_enter_physical_preflight"]
        )
        self.assertIn(
            "phase:package_stability",
            report.capabilities["physical_preflight_blockers"],
        )

    def test_hashed_step_digest_detects_same_metadata_rewrite(self) -> None:
        original_verify = (
            ValidationContext.verify_loaded_documents_unchanged
        )

        def verify_then_mutate(ctx: ValidationContext) -> None:
            original_verify(ctx)
            step = ctx.package_dir / "assembly.step"
            before = step.stat()
            content = bytearray(step.read_bytes())
            content[-1] = 33 if content[-1] != 33 else 32
            step.write_bytes(content)
            os.utime(
                step,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )

        with tempfile.TemporaryDirectory() as directory:
            package, requirements = _make_package(Path(directory))
            with patch.object(
                ValidationContext,
                "verify_loaded_documents_unchanged",
                verify_then_mutate,
            ):
                report = validate_design_package(package, requirements)
        codes = {issue.code for issue in report.errors}
        self.assertIn("HASHED_FILE_CHANGED_DURING_VALIDATION", codes)
        self.assertFalse(
            report.capabilities["can_enter_physical_preflight"]
        )

    def test_document_digest_detects_same_metadata_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "document.json"
            path.write_text('{"value":1}', encoding="utf-8")
            ctx = ValidationContext(root, None)
            value = ctx.load_json(
                path,
                required=True,
                label="stable document",
            )
            before = path.stat()
            path.write_text('{"value":2}', encoding="utf-8")
            os.utime(
                path,
                ns=(before.st_atime_ns, before.st_mtime_ns),
            )
            ctx.verify_loaded_documents_unchanged()
        self.assertEqual(value, {"value": 1})
        self.assertIn(
            "DOCUMENT_CHANGED_DURING_VALIDATION",
            {issue.code for issue in ctx.report.errors},
        )

    def test_cli_refuses_to_overwrite_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            requirements = root / "requirements.json"
            requirements.write_text("immutable", encoding="utf-8")
            stream = StringIO()
            with (
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "intake.cli.validate_design_package"
                ) as validator,
                redirect_stderr(stream),
            ):
                code = main(
                    [
                        "--package",
                        str(package),
                        "--requirements",
                        str(requirements),
                        "--output",
                        str(requirements),
                    ]
                )
            preserved_requirements = requirements.read_text(encoding="utf-8")
        self.assertEqual(code, 2)
        validator.assert_not_called()
        self.assertEqual(preserved_requirements, "immutable")
        self.assertIn("must not overwrite", stream.getvalue())

    def test_cli_report_write_error_returns_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "package"
            package.mkdir()
            blocker = root / "not-a-directory"
            blocker.write_text("block", encoding="utf-8")
            output = blocker / "report.json"
            report = IntakeReport(
                package_dir=str(package),
                requirements_file=str(root / "requirements.json"),
            )
            stream = StringIO()
            with (
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "intake.cli.validate_design_package",
                    return_value=report,
                ),
                redirect_stderr(stream),
            ):
                code = main(
                    [
                        "--package",
                        str(package),
                        "--requirements",
                        str(root / "requirements.json"),
                        "--output",
                        str(output),
                    ]
                )
        self.assertEqual(code, 2)
        self.assertIn("could not write report", stream.getvalue())

    def test_cli_writes_machine_readable_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package, requirements = _make_package(root)
            output = root / "intake_report.json"
            stream = StringIO()
            with redirect_stdout(stream):
                code = main(
                    [
                        "--package",
                        str(package),
                        "--requirements",
                        str(requirements),
                        "--output",
                        str(output),
                    ]
                )
            payload = _read_json(output)
        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "ACCEPTED")
        self.assertIn("Design package intake: ACCEPTED", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
