from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    adapter_to_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.discovery import (
    discover_mechanisms,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.dispatch import (
    build_dispatch_bundle,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.finalize import (
    _scoped_load_cases,
    finalize_dispatch,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    generate_ideal_reference_trace,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
    read_json,
    sha256_file,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.validation_scope import (
    FINAL_FULL_SELECTOR_AUTHORITY,
    ITERATION_TASK_SELECTOR_AUTHORITY,
    VALIDATION_MODE_FINAL_FULL,
    VALIDATION_MODE_ITERATION_TASK,
    VALIDATION_SCOPE_SCHEMA,
    resolve_validation_scope,
)
from simulation_ready_cad_workflow.validation_pipeline.runtime import (
    QueueManifest,
)
from simulation_ready_cad_workflow.validation_pipeline.tests.test_mechanism import (
    _build_package,
    _TEST_ADAPTER,
    _TEST_PYCHRONO_LAUNCHER,
    _trace,
    _with_test_fidelity,
)


def _plan(
    *,
    capsule_id: str = "capsule",
    contract: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        capsule_id=capsule_id,
        acceptance_contract=contract,
    )


def _cases(*case_ids: str) -> tuple[SimpleNamespace, ...]:
    return tuple(
        SimpleNamespace(
            load_case_id=case_id,
            canonical_test_load_case_id=case_id,
            program=case_id,
        )
        for case_id in case_ids
    )


class ValidationScopeResolverTests(unittest.TestCase):
    def test_finalize_iteration_compares_only_scope_selected_cases(self) -> None:
        plan = _plan()
        cases = _cases("nominal", "held_out")
        scope = {
            "mode": VALIDATION_MODE_ITERATION_TASK,
            "selected": [
                {
                    "capsule_id": "capsule",
                    "load_case_id": "nominal",
                }
            ],
        }
        self.assertEqual(
            tuple(
                item.load_case_id
                for item in _scoped_load_cases(plan, cases, scope)
            ),
            ("nominal",),
        )

    def test_final_full_selects_complete_plugin_inventory(self) -> None:
        plan = _plan()
        resolved = resolve_validation_scope(
            (plan,),
            {"capsule": _cases("nominal", "held_out")},
            mode=VALIDATION_MODE_FINAL_FULL,
        )
        scope = resolved.document
        hash_input = dict(scope)
        digest = hash_input.pop("scope_sha256")

        self.assertEqual(scope["schema"], VALIDATION_SCOPE_SCHEMA)
        self.assertEqual(scope["mode"], VALIDATION_MODE_FINAL_FULL)
        self.assertEqual(
            scope["selector_authority"],
            FINAL_FULL_SELECTOR_AUTHORITY,
        )
        self.assertEqual(scope["all"], scope["selected"])
        self.assertEqual(scope["omitted"], [])
        self.assertFalse(scope["certificate_authority"])
        self.assertTrue(scope["terminal_evidence_eligible"])
        self.assertEqual(digest, canonical_sha256(hash_input))
        self.assertEqual(
            resolved.required_branches("capsule", "nominal"),
            ("physical",),
        )

    def test_iteration_derives_deduplicated_contract_cases_and_branches(
        self,
    ) -> None:
        plan = _plan(
            contract={
                "status": "PASS",
                "certificate_authority": True,
                "requirements": [
                    {
                        "test_load_case_id": "task_b",
                        "test_program": "task_b",
                        "required_branches": ["physical"],
                    },
                    {
                        "test_load_case_id": "task_a",
                        "test_program": "task_a",
                        "required_branches": [
                            "physical",
                            "ideal_dynamics",
                        ],
                    },
                    {
                        "test_load_case_id": "task_b",
                        "test_program": "task_b",
                        "required_branches": [
                            "physical",
                            "ideal_reference",
                        ],
                    },
                ],
            }
        )
        resolved = resolve_validation_scope(
            (plan,),
            {"capsule": _cases("task_a", "task_b", "held_out")},
            mode=VALIDATION_MODE_ITERATION_TASK,
        )
        scope = resolved.document

        self.assertEqual(
            scope["selector_authority"],
            ITERATION_TASK_SELECTOR_AUTHORITY,
        )
        self.assertEqual(
            [
                item["load_case_id"]
                for item in scope["selected"]
            ],
            ["task_b", "task_a"],
        )
        self.assertEqual(
            [item["load_case_id"] for item in scope["omitted"]],
            ["held_out"],
        )
        self.assertEqual(
            resolved.required_branches("capsule", "task_b"),
            ("physical",),
        )
        self.assertEqual(
            resolved.required_branches("capsule", "task_a"),
            ("physical", "ideal_dynamics"),
        )
        self.assertFalse(scope["certificate_authority"])
        self.assertFalse(scope["terminal_evidence_eligible"])

    def test_iteration_resolves_shared_test_load_by_program(self) -> None:
        plan = _plan(
            contract={
                "status": "PASS",
                "certificate_authority": True,
                "requirements": [
                    {
                        "test_load_case_id": "rated_resistance",
                        "test_program": "roundtrip",
                        "required_branches": ["physical"],
                    },
                    {
                        "test_load_case_id": "rated_resistance",
                        "test_program": "multi_cycle",
                        "required_branches": ["physical"],
                    },
                ],
            }
        )
        cases = (
            SimpleNamespace(
                load_case_id="rated_roundtrip",
                canonical_test_load_case_id="rated_resistance",
                program="roundtrip",
            ),
            SimpleNamespace(
                load_case_id="rated_multi_cycle",
                canonical_test_load_case_id="rated_resistance",
                program="multi_cycle",
            ),
        )

        resolved = resolve_validation_scope(
            (plan,),
            {"capsule": cases},
            mode=VALIDATION_MODE_ITERATION_TASK,
        )

        self.assertEqual(
            resolved.selected_case_ids("capsule"),
            ("rated_roundtrip", "rated_multi_cycle"),
        )

    def test_iteration_rejects_missing_or_unknown_contract_case(self) -> None:
        for requirements, expected_code in (
            (
                [
                    {
                        "test_program": "trusted",
                        "required_branches": ["physical"],
                    }
                ],
                "VALIDATION_SCOPE_CONTRACT_CASE_MISSING",
            ),
            (
                [
                    {
                        "test_load_case_id": "invented",
                        "test_program": "invented",
                        "required_branches": ["physical"],
                    }
                ],
                "VALIDATION_SCOPE_CONTRACT_CASE_UNKNOWN",
            ),
        ):
            with self.subTest(expected_code=expected_code):
                with self.assertRaises(MechanismValidationError) as caught:
                    resolve_validation_scope(
                        (
                            _plan(
                                contract={
                                    "status": "PASS",
                                    "certificate_authority": True,
                                    "requirements": requirements,
                                }
                            ),
                        ),
                        {"capsule": _cases("trusted")},
                        mode=VALIDATION_MODE_ITERATION_TASK,
                    )
                self.assertEqual(caught.exception.code, expected_code)

    def test_unknown_mode_and_caller_selector_are_rejected(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            resolve_validation_scope(
                (_plan(),),
                {"capsule": _cases("trusted")},
                mode="quick",
            )
        self.assertEqual(
            caught.exception.code,
            "VALIDATION_SCOPE_MODE_UNKNOWN",
        )
        with self.assertRaises(TypeError):
            resolve_validation_scope(
                (_plan(),),
                {"capsule": _cases("trusted")},
                mode=VALIDATION_MODE_ITERATION_TASK,
                selected_case_ids=("trusted",),
            )


class ValidationScopeDispatchTests(unittest.TestCase):
    def _dispatch_plan(self, root: Path) -> tuple[object, object]:
        package, report = _build_package(root)
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "discovery.select_adapter",
            return_value=_TEST_ADAPTER,
        ):
            plan = discover_mechanisms(package, report)[0]
        adapter = _with_test_fidelity(
            root,
            _TEST_ADAPTER,
            plan.package_binding_sha256,
        )
        return (
            replace(plan, adapter_spec=adapter_to_dict(adapter)),
            adapter,
        )

    def test_required_v1_without_plan_or_v2_packet_reaches_dispatch(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, adapter = self._dispatch_plan(root)
            design = read_json(Path(plan.package_dir).parent / "design_input.json")
            review = read_json(
                Path(plan.package_dir).parent / "visual_review" / "summary.json"
            )
            package = read_json(Path(plan.package_dir) / "package.json")
            self.assertEqual(package["single_source_cad"]["policy"], "required_v1")
            self.assertIsNone(design.get("appearance_plan"))
            self.assertNotEqual(review.get("static_visual_gate_version"), 2)

            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=adapter,
            ):
                bundle = build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )

            evaluation = read_json(
                root / "run" / "design_quality" / "evaluation.json"
            )
            self.assertEqual(evaluation["status"], "evidence_complete")
            self.assertNotIn(
                "APPEARANCE_PLAN_INVALID",
                {item["code"] for item in evaluation["issues"]},
            )
            self.assertTrue(bundle.initial_functional_geometry_reports)
            self.assertGreater(bundle.job_count, 0)

    def test_default_dispatch_preserves_all_cases_and_binds_scope_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, adapter = self._dispatch_plan(root)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=adapter,
            ):
                bundle = build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            scope_path = root / bundle.validation_scope_manifest
            scope = read_json(scope_path)
            manifest = QueueManifest.from_file(root / bundle.queue_manifest)
            scope_relative = scope_path.relative_to(root).as_posix()

            self.assertEqual(bundle.job_count, 5)
            self.assertEqual(scope["mode"], VALIDATION_MODE_FINAL_FULL)
            self.assertEqual(len(scope["selected"]), 5)
            self.assertEqual(scope["omitted"], [])
            self.assertTrue(
                all(
                    job.input_files.get(scope_relative)
                    == sha256_file(scope_path)
                    for job in manifest.jobs
                )
            )
            self.assertEqual(
                read_json(root / "run" / "dispatch_summary.json")[
                    "validation_scope"
                ],
                scope,
            )

    def test_iteration_dispatch_runs_only_contract_case(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, adapter = self._dispatch_plan(root)
            plan = replace(
                plan,
                acceptance_contract={
                    "status": "PASS",
                    "certificate_authority": True,
                    "requirements": [
                        {
                            "test_load_case_id": "nominal_loaded",
                            "test_program": "bidirectional_axis",
                            "required_branches": ["physical"],
                        }
                    ],
                },
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=adapter,
            ):
                bundle = build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    validation_mode=VALIDATION_MODE_ITERATION_TASK,
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )

            self.assertEqual(bundle.job_count, 1)
            self.assertEqual(
                [
                    item["load_case_id"]
                    for item in bundle.validation_scope["selected"]
                ],
                ["nominal_loaded"],
            )
            self.assertEqual(
                {
                    item["load_case_id"]
                    for item in bundle.validation_scope["omitted"]
                },
                {
                    "nominal_light",
                    "zero_input_gravity",
                    "transverse_probe",
                    "held_out_mid",
                },
            )
            self.assertTrue(
                (
                    root
                    / "run"
                    / "requests"
                    / plan.capsule_id
                    / "nominal_loaded"
                    / "physical.json"
                ).is_file()
            )
            self.assertFalse(
                (root / "run" / "requests" / plan.capsule_id / "held_out_mid")
                .exists()
            )

    def test_iteration_finalize_emits_diagnostics_without_authority_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, adapter = self._dispatch_plan(root)
            plan = replace(
                plan,
                acceptance_contract={
                    "status": "PASS",
                    "certificate_authority": True,
                    "requirements": [
                        {
                            "test_load_case_id": "nominal_loaded",
                            "test_program": "bidirectional_axis",
                            "required_branches": ["physical"],
                        }
                    ],
                },
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=adapter,
            ):
                build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    validation_mode=VALIDATION_MODE_ITERATION_TASK,
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )

            dispatch_path = root / "run" / "dispatch_summary.json"
            dispatch = read_json(dispatch_path)
            dispatch["execution"] = "legacy_per_branch_v0"
            write_json_atomic(dispatch_path, dispatch)
            load_case = LoadCase.from_dict(
                read_json(
                    root
                    / "run"
                    / "loads"
                    / plan.capsule_id
                    / "nominal_loaded.json"
                )
            )
            trace_root = (
                root
                / "run"
                / "traces"
                / plan.capsule_id
                / load_case.load_case_id
            )
            physical_trace = _trace(plan, load_case, branch="physical")
            write_json_atomic(
                trace_root / "physical.trace.json", physical_trace
            )
            write_json_atomic(
                trace_root / "reduced.trace.json",
                _trace(plan, load_case, branch="reduced"),
            )
            write_json_atomic(
                trace_root / "ideal_reference.trace.json",
                generate_ideal_reference_trace(
                    plan,
                    load_case,
                    physical_trace,
                ),
            )

            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "finalize.write_certificate_artifacts"
            ) as write_certificate, patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "finalize.build_whole_machine_dispatch"
            ) as build_whole, patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "finalize.default_load_cases",
                side_effect=AssertionError("modern finalize regenerated loads"),
            ) as regenerated, patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "finalize.audit_physical_trace_evidence",
                return_value={
                    "status": "PHYSICAL_PREFLIGHT_PASSED",
                    "passed": True,
                    "failure_reasons": [],
                    "reduced_branch_authorized": True,
                    "geometry_legality": {"passed": True},
                    "component_contact_intent": {
                        "passed": True,
                        "authoritative": True,
                        "outcomes": [],
                    },
                },
            ):
                result = finalize_dispatch(
                    workspace_root=root,
                    output_dir=root / "run",
                )

            result_root = root / "run" / "results" / plan.capsule_id
            mechanism = result["mechanisms"][0]
            self.assertEqual(result["status"], "ITERATION_DIAGNOSTIC_ONLY")
            self.assertEqual(
                mechanism["overall_verdict"],
                "ITERATION_DIAGNOSTIC_ONLY",
            )
            self.assertIsNone(mechanism["certificate"])
            self.assertIsNone(mechanism["replacement"])
            self.assertIsNone(mechanism["whole_machine_dispatch"])
            self.assertFalse((result_root / "certificate.json").exists())
            self.assertFalse((result_root / "replacement.json").exists())
            self.assertFalse((result_root / "whole_machine").exists())
            write_certificate.assert_not_called()
            build_whole.assert_not_called()
            regenerated.assert_not_called()


if __name__ == "__main__":
    unittest.main()
