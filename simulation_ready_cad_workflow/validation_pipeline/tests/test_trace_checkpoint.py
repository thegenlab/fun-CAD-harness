from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_runtime import (
    _compound_joint_drift_failure,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism import worker
from simulation_ready_cad_workflow.validation_pipeline.mechanism.clock_gear_runtime import (
    _clock_checkpoint_interface_ids,
    _clock_checkpoint_resume,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    registered_mechanism_adapters,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    current_wrench_ledger,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)


def _execution(output_file: Path) -> SimpleNamespace:
    family_spec = {
        "collision_geometry": {
            "schema": "family_collision_geometry_v1",
            "component_count": 1,
            "bodies": [{"body_id": "moving", "fixed": False, "components": []}],
        }
    }
    plan = SimpleNamespace(
        adapter_spec={"family_spec": family_spec},
        package_id="checkpoint-machine",
        design_version="1.0",
        package_binding_sha256="b" * 64,
        capsule_id="checkpoint_capsule",
        axis_abs=(1.0, 0.0, 0.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        input_point_abs_m=(0.0, 0.0, 0.0),
        observable_point_abs_m=(0.1, 0.0, 0.0),
        characteristic_length_m=0.1,
        task_channel_registry=None,
        to_dict=lambda: {"capsule_id": "checkpoint_capsule"},
    )
    phase = SimpleNamespace(
        phase_id="advance",
        duration_s=1.0,
        to_dict=lambda: {
            "phase_id": "advance",
            "wrench_kind": "force",
            "direction_abs": [1.0, 0.0, 0.0],
            "magnitude_si": 1.0,
            "duration_s": 1.0,
        },
    )
    load_case = SimpleNamespace(
        load_case_id="nominal",
        step_s=0.001,
        sample_interval_s=0.1,
        phases=(phase,),
        to_dict=lambda: {"load_case_id": "nominal"},
    )
    return SimpleNamespace(
        output_file=output_file,
        branch="physical",
        artifact_branch="physical",
        adapter=SimpleNamespace(adapter_id="checkpoint_adapter", family_spec=family_spec),
        mechanism_plugin=next(
            plugin
            for plugin in registered_mechanism_adapters()
            if plugin.plugin_id == "plain_prismatic_v1"
        ),
        plan=plan,
        load_case=load_case,
    )


class TraceCheckpointTests(unittest.TestCase):
    def test_clock_checkpoint_current_state_includes_support_interfaces(
        self,
    ) -> None:
        self.assertEqual(
            _clock_checkpoint_interface_ids(
                ("setting-clutch", "alarm-clutch"),
                ("support-seconds", "support-alarm"),
            ),
            {
                "setting-clutch",
                "alarm-clutch",
                "support-seconds",
                "support-alarm",
            },
        )

    def test_clock_checkpoint_selects_complete_resume_state(self) -> None:
        prior = {
            "checkpoint": {
                "resume_state": {
                    "schema_version": "clock_gear_resume_v1",
                    "time_s": 42.0,
                    "wrench_ledger_state": {
                        "schema_version": (
                            "simulation_ready_cad_wrench_checkpoint_v1"
                        )
                    },
                }
            }
        }
        selected, evidence = _clock_checkpoint_resume(prior)
        self.assertIs(selected, prior)
        self.assertEqual(evidence["execution_origin"], "checkpoint")
        self.assertFalse(evidence["checkpoint_discarded"])
        self.assertTrue(evidence["dynamics_state_restored"])
        self.assertTrue(evidence["wrench_ledger_state_restored"])

    def test_clock_without_checkpoint_starts_at_time_zero(self) -> None:
        selected, evidence = _clock_checkpoint_resume(None)
        self.assertIsNone(selected)
        self.assertFalse(evidence["checkpoint_detected"])
        self.assertEqual(evidence["execution_origin"], "time_zero")
        self.assertEqual(
            evidence["policy"],
            "resume_complete_clock_state_v1",
        )

    def test_clock_checkpoint_requires_wrench_state(self) -> None:
        prior = {
            "checkpoint": {
                "resume_state": {
                    "schema_version": "clock_gear_resume_v1",
                    "time_s": 42.0,
                }
            }
        }
        with self.assertRaises(worker.MechanismValidationError) as raised:
            _clock_checkpoint_resume(prior)
        self.assertEqual(
            raised.exception.code,
            "CLOCK_CHECKPOINT_STATE_INVALID",
        )

    def test_checkpoint_atomically_retains_sampled_body_poses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "physical.trace.json"
            execution = _execution(output)
            samples = [
                {
                    "time_s": 0.0,
                    "phase_id": "initial",
                    "body_poses": {
                        "moving": {
                            "position_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                },
                {
                    "time_s": 0.5,
                    "phase_id": "advance",
                    "body_poses": {
                        "moving": {
                            "position_abs_m": [0.5, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        }
                    },
                },
            ]
            with patch.dict(
                os.environ,
                {worker.MECHANISM_SOURCE_SHA256_ENV: "a" * 64},
            ):
                worker._write_trace_checkpoint(execution, samples)

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "PHYSICS_PARTIAL")
            self.assertEqual(payload["sample_count"], 2)
            self.assertEqual(
                payload["samples"][-1]["body_poses"]["moving"]["position_abs_m"],
                [0.5, 0.0, 0.0],
            )
            self.assertEqual(
                payload["checkpoint"]["status"],
                "IN_PROGRESS_AT_LAST_ATOMIC_WRITE",
            )
            self.assertEqual(payload["checkpoint"]["completion_fraction"], 0.5)
            self.assertTrue(payload["checkpoint"]["atomic_write"])
            self.assertFalse(payload["checkpoint"]["resumable"])
            self.assertIsNone(payload["checkpoint"]["resume_state"])
            self.assertFalse(
                payload["visual_replay_semantics"]["certification_authority"]
            )
            self.assertTrue(payload["collision_geometry"]["contact_enabled"])
            self.assertFalse(any(output.parent.glob(".*.tmp")))

    def test_resumable_checkpoint_requires_identical_execution_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "physical.trace.json"
            execution = _execution(output)
            samples = [
                {
                    "time_s": 0.5,
                    "phase_id": "advance",
                    "body_poses": {},
                }
            ]
            resume_state = {
                "schema_version": "clock_gear_resume_v1",
                "time_s": 0.5,
            }
            with patch.dict(
                os.environ,
                {worker.MECHANISM_SOURCE_SHA256_ENV: "a" * 64},
            ):
                worker._write_trace_checkpoint(
                    execution,
                    samples,
                    resume_state=resume_state,
                )
                loaded = worker._load_trace_checkpoint_for_resume(execution)

                self.assertIsNotNone(loaded)
                self.assertTrue(loaded["checkpoint"]["resumable"])
                self.assertEqual(
                    loaded["checkpoint"]["resume_state"],
                    resume_state,
                )

                execution.plan.package_binding_sha256 = "c" * 64
                with self.assertRaisesRegex(
                    worker.MechanismValidationError,
                    "package_binding_sha256",
                ) as raised:
                    worker._load_trace_checkpoint_for_resume(execution)
            self.assertEqual(
                raised.exception.code,
                "TRACE_CHECKPOINT_BINDING_MISMATCH",
            )

    def test_partial_visual_trace_without_dynamics_state_is_not_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "physical.trace.json"
            execution = _execution(output)
            samples = [
                {
                    "time_s": 0.5,
                    "phase_id": "advance",
                    "body_poses": {},
                }
            ]
            with patch.dict(
                os.environ,
                {worker.MECHANISM_SOURCE_SHA256_ENV: "a" * 64},
            ):
                worker._write_trace_checkpoint(execution, samples)
                with self.assertRaises(
                    worker.MechanismValidationError
                ) as raised:
                    worker._load_trace_checkpoint_for_resume(execution)
            self.assertEqual(
                raised.exception.code,
                "TRACE_CHECKPOINT_RESUME_STATE_INVALID",
            )

    def test_simulation_exception_preserves_checkpoint_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            output = workspace / "physical.trace.json"
            execution = _execution(output)
            samples = [{"time_s": 0.2, "phase_id": "advance", "body_poses": {}}]
            with patch.dict(
                os.environ,
                {worker.MECHANISM_SOURCE_SHA256_ENV: "a" * 64},
            ):
                worker._write_trace_checkpoint(execution, samples)
                with (
                    patch.object(
                        worker,
                        "load_execution_request",
                        return_value=execution,
                    ),
                    patch.object(
                        worker,
                        "_simulate",
                        side_effect=RuntimeError("solver stopped"),
                    ),
                ):
                    payload = worker.run_request(
                        workspace / "request.json",
                        workspace_root=workspace,
                    )

            self.assertEqual(payload["status"], "PHYSICS_INCONCLUSIVE")
            self.assertEqual(payload["sample_count"], 1)
            self.assertEqual(payload["samples"][0]["time_s"], 0.2)
            self.assertEqual(
                payload["checkpoint"]["status"],
                "TERMINATED_BY_SIMULATION_EXCEPTION",
            )
            self.assertEqual(
                payload["checkpoint"]["final_termination_reason"],
                "SIMULATION_EXCEPTION",
            )
            self.assertFalse(
                payload["visual_replay_semantics"]["certification_authority"]
            )

    def test_measured_physical_failure_preserves_sample_and_clean_input_release(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            output = workspace / "physical.trace.json"
            execution = _execution(output)
            samples = [
                {
                    "time_s": 0.2,
                    "phase_id": "advance",
                    "max_joint_drift_m": 0.004,
                    "body_poses": {},
                }
            ]

            def stop_after_recording(*_args: object) -> None:
                ledger = current_wrench_ledger()
                load_id = "declared-input-force"
                ledger.register(
                    standard_wrench_declaration(
                        plugin_id=ledger.plugin_id,
                        profile="task_input",
                        wrench_kind="force",
                        load_id=load_id,
                        owner=ledger.plugin_id,
                        target_body=next(iter(ledger.declared_input_body_ids)),
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="load_case_phase_and_input_frame",
                        active_interval="declared_load_case_phases",
                        release_required=True,
                    ),
                    wrench_kind="force",
                    initial_vector=(0.0, 0.0, 0.0),
                )
                transition_current_wrench_phase("advance")
                ledger.record_update(load_id, (1.0, 0.0, 0.0))
                worker._write_trace_checkpoint(execution, samples)
                ledger.record_update(load_id, (0.0, 0.0, 0.0))
                set_current_wrench_phase(None)
                failure = _compound_joint_drift_failure(
                    branch="physical",
                    joint_drifts_m={"drive-pin": 0.004},
                    declared_max_joint_drift_m=0.003,
                    mechanism_plugin_id="generic.compound_linkage_v1",
                    mechanism_registration_source="installed_harness_code",
                    load_case_id="rated",
                    phase_index=0,
                    phase_id="advance",
                    time_s=0.2,
                    step_s=0.00002,
                    sample_interval_s=0.01,
                )
                assert failure is not None
                raise failure

            with patch.dict(
                os.environ,
                {worker.MECHANISM_SOURCE_SHA256_ENV: "a" * 64},
            ):
                with (
                    patch.object(
                        worker,
                        "load_execution_request",
                        return_value=execution,
                    ),
                    patch.object(
                        worker,
                        "_simulate",
                        side_effect=stop_after_recording,
                    ),
                ):
                    payload = worker.run_request(
                        workspace / "request.json",
                        workspace_root=workspace,
                    )

            self.assertEqual(payload["status"], "PHYSICAL_FAILURE")
            self.assertEqual(payload["sample_count"], 1)
            self.assertEqual(payload["samples"][0]["max_joint_drift_m"], 0.004)
            self.assertEqual(
                payload["checkpoint"]["status"],
                "TERMINATED_BY_CERTIFICATION_FAILURE",
            )
            self.assertEqual(
                payload["checkpoint"]["final_termination_reason"],
                "PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED",
            )
            self.assertEqual(payload["failure"]["owner"], "design_package")
            self.assertEqual(
                payload["failure"]["evidence_validity"],
                "VALID_MEASUREMENT",
            )
            self.assertTrue(
                payload["visual_replay_semantics"][
                    "failure_evidence_authority"
                ]
            )
            self.assertEqual(payload["wrench_audit"]["release_residual_count"], 0)
            self.assertNotIn(
                "WRENCH_PHASE_CONTEXT_STALE",
                {
                    item["code"]
                    for item in payload["wrench_audit"]["violations"]
                },
            )

    def test_simulation_exception_is_not_masked_by_wrench_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            output = workspace / "physical.trace.json"
            execution = _execution(output)

            def fail_with_active_task_input(*_args: object) -> None:
                ledger = current_wrench_ledger()
                load_id = "declared-input-force"
                ledger.register(
                    standard_wrench_declaration(
                        plugin_id=ledger.plugin_id,
                        profile="task_input",
                        wrench_kind="force",
                        load_id=load_id,
                        owner=ledger.plugin_id,
                        target_body=next(iter(ledger.declared_input_body_ids)),
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="load_case_phase_and_input_frame",
                        active_interval="declared_load_case_phases",
                        release_required=True,
                    ),
                    wrench_kind="force",
                    initial_vector=(0.0, 0.0, 0.0),
                )
                transition_current_wrench_phase("advance")
                ledger.record_update(load_id, (1.0, 0.0, 0.0))
                raise RuntimeError("solver stopped with active input")

            with (
                patch.object(
                    worker,
                    "load_execution_request",
                    return_value=execution,
                ),
                patch.object(
                    worker,
                    "_simulate",
                    side_effect=fail_with_active_task_input,
                ),
            ):
                payload = worker.run_request(
                    workspace / "request.json",
                    workspace_root=workspace,
                )

            self.assertEqual(payload["status"], "PHYSICS_INCONCLUSIVE")
            self.assertEqual(payload["error"]["code"], "SIMULATION_EXCEPTION")
            self.assertEqual(payload["failure"]["code"], "RuntimeError")
            self.assertEqual(
                payload["error"]["message"],
                "solver stopped with active input",
            )


if __name__ == "__main__":
    unittest.main()
