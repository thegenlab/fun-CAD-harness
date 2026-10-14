"""Focused ownership tests for certificate and finalize boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.contracts.failure_envelope import (
    EvidenceValidity,
    FailureOwner,
    make_failure_envelope,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism import certificate
from simulation_ready_cad_workflow.validation_pipeline.mechanism.finalize import (
    _certificate_physical_trace_audits,
    _raise_finalize_evidence_error,
    build_run_status_document,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class CertificateOwnershipTest(unittest.TestCase):
    _registration = {
        "plugin_id": "reviewer.certificate_test",
        "registration_source": "reviewer_mechanism_adapter",
    }

    def _call(self, callback):
        plugin = SimpleNamespace(
            plugin_id="reviewer.certificate_test",
            load_envelope=callback,
        )
        with mock.patch.object(
            certificate,
            "_plugin_registration",
            return_value=self._registration,
        ):
            return certificate._certificate_plugin_hook(
                plugin,
                (),
                (),
                hook="load_envelope",
            )

    def test_existing_failure_envelope_passes_through_unchanged(self) -> None:
        failure = make_failure_envelope(
            "DESIGN_INPUT_REJECTED",
            FailureOwner.DESIGN_PACKAGE,
            source_component="test.design_producer",
            message="the plugin correctly rejected design input",
        )
        original = MechanismValidationError(
            "DESIGN_INPUT_REJECTED",
            "the plugin correctly rejected design input",
            failure=failure,
        )

        def reject(_loads, _comparisons):
            raise original

        with self.assertRaises(MechanismValidationError) as caught:
            self._call(reject)

        self.assertIs(caught.exception, original)
        self.assertEqual(caught.exception.failure_envelope, failure)

    def test_typed_ownerless_failure_stays_unclassified(self) -> None:
        def reject(_loads, _comparisons):
            raise MechanismValidationError(
                "CERTIFICATE_INPUT_REJECTED",
                "the callback did not identify whether code or input is wrong",
            )

        with self.assertRaises(MechanismValidationError) as caught:
            self._call(reject)

        self.assertEqual(caught.exception.code, "CERTIFICATE_INPUT_REJECTED")
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            FailureOwner.UNCLASSIFIED.value,
        )

    def test_unexpected_and_invalid_returns_use_registration_owner(self) -> None:
        cases = (
            (lambda _loads, _comparisons: 1, "MECHANISM_ADAPTER_LOAD_ENVELOPE_INVALID"),
            (
                lambda _loads, _comparisons: (_ for _ in ()).throw(
                    RuntimeError("plugin boom")
                ),
                "MECHANISM_ADAPTER_LOAD_ENVELOPE_FAILED",
            ),
        )
        for callback, code in cases:
            with self.subTest(code=code), self.assertRaises(
                MechanismValidationError
            ) as caught:
                self._call(callback)
            self.assertEqual(caught.exception.code, code)
            self.assertEqual(
                caught.exception.failure_envelope["owner"],
                FailureOwner.REVIEWER_MECHANISM_PLUGIN.value,
            )

    def test_mechanism_plugin_resolution_failure_is_not_hidden(self) -> None:
        failure = make_failure_envelope(
            "PLUGIN_BINDING_INVALID",
            FailureOwner.HARNESS_CODE,
            source_component="test.plugin_registry",
        )
        original = MechanismValidationError(
            "PLUGIN_BINDING_INVALID",
            "plugin binding is invalid",
            failure=failure,
        )
        with mock.patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "family_plugins.mechanism_adapter_for_plan",
            side_effect=original,
        ), self.assertRaises(MechanismValidationError) as caught:
            certificate._mechanism_plugin(SimpleNamespace())

        self.assertIs(caught.exception, original)


class FinalizeOwnershipTest(unittest.TestCase):
    def test_not_certified_result_explicitly_owns_measured_failure(self) -> None:
        result = build_run_status_document(
            summaries=[
                {
                    "capsule_id": "failed_capsule",
                    "verdict": "PHYSICAL_FAILURE",
                    "overall_verdict": "NOT_CERTIFIED",
                    "typed_task_verdict": "FAIL",
                    "rigid_attachment": {},
                    "collider_fidelity": {},
                    "whole_machine_dispatch": None,
                }
            ],
            ideal_reference_summary={},
            iteration_diagnostic_only=False,
            replacement_authorized=False,
            replacement_policy="reference_only_single_mechanism",
        )

        self.assertEqual(result["status"], "NOT_CERTIFIED")
        self.assertEqual(result["failure_reason"], "CERTIFICATE_NOT_CERTIFIED")
        self.assertEqual(
            result["failure"]["owner"], FailureOwner.DESIGN_PACKAGE.value
        )
        self.assertEqual(
            result["failure"]["evidence_validity"],
            EvidenceValidity.VALID_MEASUREMENT.value,
        )

    def test_task_policy_evidence_failure_routes_to_reviewer_task_plugin(self) -> None:
        reason = (
            "phone_stand_hold:task_phone_stand_loaded_hold:"
            "WRENCH_POWERED_PHASE_INPUT_COVERAGE_MISSING"
        )
        result = build_run_status_document(
            summaries=[
                {
                    "capsule_id": "phone_stand_adjustment",
                    "verdict": "KINEMATIC_CERTIFIED",
                    "overall_verdict": "NOT_CERTIFIED",
                    "typed_task_verdict": "INCONCLUSIVE",
                    "task_acceptance_failure_reasons": [reason],
                    "rigid_attachment": {},
                    "collider_fidelity": {},
                    "whole_machine_dispatch": None,
                }
            ],
            ideal_reference_summary={},
            iteration_diagnostic_only=False,
            replacement_authorized=False,
            replacement_policy="reference_only_single_mechanism",
        )

        self.assertEqual(result["status"], "NOT_CERTIFIED")
        self.assertEqual(
            result["failure"]["owner"],
            FailureOwner.REVIEWER_TASK_PLUGIN.value,
        )
        self.assertEqual(
            result["failure"]["details"][
                "task_acceptance_failure_reasons"
            ],
            [reason],
        )

    def test_finalize_catch_preserves_existing_owner(self) -> None:
        failure = make_failure_envelope(
            "REVIEWER_CERTIFICATE_FAILED",
            FailureOwner.REVIEWER_MECHANISM_PLUGIN,
            source_component="test.reviewer_plugin",
            plugin_id="reviewer.certificate_test",
            plugin_kind="mechanism",
        )
        original = MechanismValidationError(
            "REVIEWER_CERTIFICATE_FAILED",
            "reviewer callback failed",
            failure=failure,
        )
        with self.assertRaises(MechanismValidationError) as caught:
            _raise_finalize_evidence_error(
                original,
                code="FINALIZE_EVIDENCE_INVALID",
                message="finalize could not consume evidence",
            )

        self.assertIs(caught.exception, original)
        self.assertEqual(caught.exception.failure_envelope, failure)

    def test_missing_persisted_physical_audit_is_harness_owned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "run"
            output.mkdir()
            plan = SimpleNamespace(capsule_id="capsule")
            load_case = SimpleNamespace(load_case_id="case")

            with self.assertRaises(MechanismValidationError) as caught:
                _certificate_physical_trace_audits(
                    workspace=root,
                    output=output,
                    result_root=output / "results" / "capsule",
                    plan=plan,
                    load_cases=(load_case,),
                    traces_by_load_case={},
                )

        self.assertEqual(
            caught.exception.code,
            "FINALIZE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            FailureOwner.HARNESS_CODE.value,
        )


if __name__ == "__main__":
    unittest.main()
