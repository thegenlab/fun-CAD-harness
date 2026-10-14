"""Tests for the formal requirements / design package input schemas."""

from __future__ import annotations

import unittest
from typing import Any

from simulation_ready_cad_workflow.validation_pipeline.contracts.input_schemas import (
    SchemaCheckError,
    check_design_package_manifest,
    check_requirements_input,
    design_package_input_schema,
    requirements_input_schema,
)


def _typed_requirement(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "requirement_id": "gate_rises",
        "prompt_clause": "raise the gate by at least 176 mm",
        "metric_id": "signed_displacement_m",
        "observable_quantity": "pose_6d",
        "direction": "positive",
        "target": {"comparator": "at_least", "value": 0.176, "unit": "m"},
        "target_role": "rack and gate body",
        "input_role": "manual crank input",
        "test_program": "task_rack_roundtrip",
        "test_load_case_id": "task_rack_roundtrip",
        "allow_direct_output_actuation": False,
        "required_branches": ["physical", "ideal_reference"],
    }
    entry.update(overrides)
    return entry


def _requirements(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "0.2",
        "requirements_id": "demo_requirements_v1",
        "prompt": "Build a machine that can raise the gate by at least 176 mm.",
        "tasks": [
            {
                "task_id": "raise_gate",
                "description": "Raise the gate.",
                "acceptance_requirements": [_typed_requirement()],
            }
        ],
    }
    document.update(overrides)
    return document


def _manifest(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": "0.1",
        "package_id": "demo_package_v1",
        "design_version": "1.0.0",
        "created_by_ai": True,
        "assembly_file": "assembly.step",
        "assembly_contract_file": "assembly_contract.json",
        "mechanism_coverage_file": "mechanism_coverage.json",
        "machine_task_mapping_file": "machine_task_mapping.json",
        "mechanism_capsule_files": ["mechanism_capsules/drive.json"],
        "requirements_ref": {
            "requirements_id": "demo_requirements_v1",
            "sha256": "a" * 64,
        },
        "artifacts": [{"path": "assembly.step", "sha256": "b" * 64}],
    }
    document.update(overrides)
    return document


class RequirementsInputSchemaTest(unittest.TestCase):
    def test_valid_typed_document_passes(self) -> None:
        check_requirements_input(_requirements())

    def test_valid_legacy_document_passes(self) -> None:
        check_requirements_input(
            {
                "schema_version": "0.1",
                "requirements_id": "legacy_requirements_v1",
                "tasks": [
                    {
                        "task_id": "open_lid",
                        "description": "Open the lid.",
                        "success_criteria": ["lid opens"],
                    }
                ],
            }
        )

    def test_unknown_schema_version_rejected(self) -> None:
        with self.assertRaises(SchemaCheckError):
            check_requirements_input(_requirements(schema_version="0.3"))

    def test_missing_prompt_rejected_for_typed_documents(self) -> None:
        document = _requirements()
        del document["prompt"]
        with self.assertRaises(SchemaCheckError):
            check_requirements_input(document)

    def test_legacy_document_does_not_require_prompt(self) -> None:
        check_requirements_input(
            {
                "schema_version": "0.1",
                "requirements_id": "legacy_requirements_v1",
                "tasks": [{"task_id": "t", "description": "d"}],
            }
        )

    def test_missing_target_rejected(self) -> None:
        entry = _typed_requirement()
        del entry["target"]
        document = _requirements()
        document["tasks"][0]["acceptance_requirements"] = [entry]
        with self.assertRaises(SchemaCheckError):
            check_requirements_input(document)

    def test_unknown_comparator_rejected(self) -> None:
        entry = _typed_requirement(
            target={"comparator": "roughly", "value": 1, "unit": "m"}
        )
        document = _requirements()
        document["tasks"][0]["acceptance_requirements"] = [entry]
        with self.assertRaises(SchemaCheckError):
            check_requirements_input(document)

    def test_unknown_branch_rejected(self) -> None:
        entry = _typed_requirement(required_branches=["physical", "vibes"])
        document = _requirements()
        document["tasks"][0]["acceptance_requirements"] = [entry]
        with self.assertRaises(SchemaCheckError):
            check_requirements_input(document)

    def test_non_boolean_actuation_policy_rejected(self) -> None:
        entry = _typed_requirement(allow_direct_output_actuation="no")
        document = _requirements()
        document["tasks"][0]["acceptance_requirements"] = [entry]
        with self.assertRaises(SchemaCheckError):
            check_requirements_input(document)

    def test_extra_descriptive_fields_are_tolerated(self) -> None:
        check_requirements_input(
            _requirements(
                non_numeric_engineering_constraints=["complete detail"],
                visual_evidence_checklist=["front view"],
            )
        )


class DesignPackageInputSchemaTest(unittest.TestCase):
    def test_valid_manifest_passes(self) -> None:
        check_design_package_manifest(_manifest())

    def test_missing_required_field_rejected(self) -> None:
        document = _manifest()
        del document["requirements_ref"]
        with self.assertRaises(SchemaCheckError):
            check_design_package_manifest(document)

    def test_created_by_ai_must_be_true(self) -> None:
        with self.assertRaises(SchemaCheckError):
            check_design_package_manifest(_manifest(created_by_ai=False))

    def test_artifact_entries_require_path_and_hash(self) -> None:
        with self.assertRaises(SchemaCheckError):
            check_design_package_manifest(
                _manifest(artifacts=[{"path": "assembly.step"}])
            )

    def test_wrong_schema_version_rejected(self) -> None:
        with self.assertRaises(SchemaCheckError):
            check_design_package_manifest(_manifest(schema_version="0.2"))

    def test_single_source_cad_policy_shape(self) -> None:
        check_design_package_manifest(
            _manifest(
                single_source_cad={
                    "policy": "required_v1",
                    "authoritative_artifact": "assembly.step",
                    "derived_assets": "hash_bound_and_regenerated",
                }
            )
        )
        with self.assertRaises(SchemaCheckError):
            check_design_package_manifest(
                _manifest(single_source_cad={"policy": "optional"})
            )


class SchemaDocumentIntegrityTest(unittest.TestCase):
    def test_schema_files_use_only_supported_keywords(self) -> None:
        """schema_check fails closed on unknown keywords; a passing check of
        any document proves the schema files stay inside the supported
        keyword subset."""

        check_requirements_input(_requirements())
        check_design_package_manifest(_manifest())
        self.assertEqual(
            requirements_input_schema()["$id"],
            "simulation_ready_cad_requirements_input.v1",
        )
        self.assertEqual(
            design_package_input_schema()["$id"],
            "simulation_ready_cad_design_package_input.v1",
        )


if __name__ == "__main__":
    unittest.main()
