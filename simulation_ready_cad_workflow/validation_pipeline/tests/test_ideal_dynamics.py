from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.dispatch import (
    _requires_ideal_dynamics,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
    ExecutionRequest,
    run_request,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    set_current_wrench_phase,
    transition_current_wrench_phase,
)


def _real_plugin(plugin_id: str):
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
        registered_mechanism_adapters,
    )

    return next(
        plugin
        for plugin in registered_mechanism_adapters()
        if plugin.plugin_id == plugin_id
    )


def _no_input_load_case() -> LoadCase:
    return LoadCase(
        load_case_id="case",
        family="plain_prismatic",
        program="branch_artifact_test",
        purpose="nominal",
        moving_mass_kg=1.0,
        phases=(
            LoadPhase(
                phase_id="no_input",
                wrench_kind="none",
                direction_abs=None,
                magnitude_si=0.0,
                duration_s=0.1,
            ),
        ),
    )


def _simulated_trace(branch: str) -> dict:
    load_case = _no_input_load_case()
    schedule = [
        {
            "phase_id": "no_input",
            "wrench_kind": "none",
            "direction_abs": None,
            "magnitude_si": 0.0,
            "duration_s": 0.1,
        }
    ]
    transition_current_wrench_phase("no_input")
    set_current_wrench_phase(None)
    return {
        "schema_version": "0.1",
        "status": "COMPLETED",
        "branch": branch,
        "capsule_id": "capsule",
        "load_case_id": load_case.load_case_id,
        "executed_phases": schedule,
        "executed_load_sha256": canonical_sha256(schedule),
    }


class IdealDynamicsPolicyTests(unittest.TestCase):
    def test_physical_request_preserves_external_branch_for_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "physical.trace.json"
            execution = ExecutionRequest(
                request_file=root / "request.json",
                output_file=output,
                branch="physical",
                artifact_branch="physical",
                adapter=SimpleNamespace(family_spec={}),
                mechanism_plugin=_real_plugin("plain_prismatic_v1"),
                plan=SimpleNamespace(task_channel_registry=None),
                load_case=_no_input_load_case(),
                reference_trace_file=None,
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "worker.load_execution_request",
                return_value=execution,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "worker._simulate",
                side_effect=lambda *_: _simulated_trace("physical"),
            ):
                result = run_request(
                    root / "request.json",
                    workspace_root=root,
                )

            self.assertEqual(result["branch"], "physical")
            self.assertTrue(output.is_file())

    def test_branch_is_absent_without_explicit_typed_claim(self) -> None:
        legacy = SimpleNamespace(acceptance_contract=None)
        typed_without_dynamics = SimpleNamespace(
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "requirements": [
                    {"required_branches": ["physical", "ideal_reference"]}
                ],
            }
        )

        self.assertFalse(_requires_ideal_dynamics(legacy))
        self.assertFalse(_requires_ideal_dynamics(typed_without_dynamics))

    def test_explicit_typed_claim_enables_branch(self) -> None:
        plan = SimpleNamespace(
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "requirements": [
                    {"required_branches": ["physical", "ideal_dynamics"]}
                ],
            }
        )

        self.assertTrue(_requires_ideal_dynamics(plan))

    def test_worker_writes_new_branch_name_not_legacy_reduced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "ideal_dynamics.trace.json"
            execution = ExecutionRequest(
                request_file=root / "request.json",
                output_file=output,
                branch="reduced",
                artifact_branch="ideal_dynamics",
                adapter=SimpleNamespace(family_spec={}),
                mechanism_plugin=_real_plugin("plain_prismatic_v1"),
                plan=SimpleNamespace(task_channel_registry=None),
                load_case=_no_input_load_case(),
                reference_trace_file=None,
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "worker.load_execution_request",
                return_value=execution,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "worker._simulate",
                side_effect=lambda *_: _simulated_trace("reduced"),
            ):
                result = run_request(
                    root / "request.json",
                    workspace_root=root,
                )

        self.assertEqual(result["branch"], "ideal_dynamics")
        self.assertEqual(
            result["runtime_compatibility"]["internal_branch_name"],
            "reduced",
        )
        self.assertFalse(
            result["runtime_compatibility"]["contact_evidence_authority"]
        )


if __name__ == "__main__":
    unittest.main()
