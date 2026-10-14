from __future__ import annotations

import os
import unittest
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    bicycle_chain_whole_machine,
    whole_machine,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine_worker import (
    _attach_whole_machine_wrench_audits,
    _plain_whole_machine_executed_schedule,
    _plain_whole_machine_execution,
    _run_registered_whole_machine_load_case,
    _validate_whole_machine_wrench_audits,
    _wrench_context_for_execution,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    current_wrench_phase,
    set_current_wrench_phase,
    transition_current_wrench_phase,
)


_BACKENDS = {
    "compound": (
        "revolute_linkage_v1",
        "revolute_linkage_v1",
        {
            "input_body_id": "pedal",
            "output_body_id": "lid",
        },
    ),
    "rack": (
        "rack_pinion_v1",
        "rack_pinion_v1",
        {
            "pinion_body_id": "crank",
            "rack_body_id": "rack",
        },
    ),
    "scissor": (
        "hydraulic_scissor_lift_v1",
        "hydraulic_scissor_lift_v1",
        {
            "output_body_id": "platform",
            "actuator": {
                "barrel_body_id": "cylinder_barrel",
                "rod_body_id": "cylinder_rod",
            },
        },
    ),
    "clock": (
        "spur_gear_clock_v1",
        "spur_gear_clock_v1",
        {
            "input_body_id": "time_barrel",
            "setting_input_body_id": "setting_crown",
            "alarm_input_body_id": "alarm_barrel",
            "output_body_id": "minute_hand",
        },
    ),
}


def _execution(kind: str, load_case_id: str) -> SimpleNamespace:
    plugin_id, backend, family_spec = _BACKENDS[kind]
    load_case = LoadCase(
        load_case_id=load_case_id,
        family="compound",
        program=f"{kind}_whole_machine_program",
        purpose="nominal",
        moving_mass_kg=1.0,
        step_s=0.01,
        sample_interval_s=0.01,
        phases=(
            LoadPhase(
                phase_id="registered_phase",
                wrench_kind="none",
                direction_abs=None,
                magnitude_si=0.0,
                duration_s=0.1,
            ),
        ),
    )
    from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
        registered_mechanism_adapters,
    )

    plugin = next(
        item
        for item in registered_mechanism_adapters()
        if item.plugin_id == plugin_id
    )
    assert plugin.worker_backend == backend
    return SimpleNamespace(
        mechanism_plugin=plugin,
        load_case=load_case,
        artifact_branch="reduced",
        plan=SimpleNamespace(task_channel_registry=None),
        adapter=SimpleNamespace(family_spec=deepcopy(family_spec)),
    )


def _executed_schedule(execution: SimpleNamespace) -> list[dict]:
    result = []
    for phase in execution.load_case.phases:
        record = {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None
                if phase.direction_abs is None
                else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
            "duration_s": phase.duration_s,
        }
        if execution.mechanism_plugin.plugin_id == "spur_gear_clock_v1":
            active_body = execution.adapter.family_spec["input_body_id"]
            record.update(
                {
                    "active_input_body_id": active_body,
                    "active_input_point_abs_m": [0.0, 0.0, 0.0],
                    "termination_reason": "duration_limit",
                    "setting_clutch_engaged": False,
                    "actuation_channels": [
                        {
                            "body_id": active_body,
                            "kind": phase.wrench_kind,
                            "magnitude_si": phase.magnitude_si,
                        }
                    ],
                }
            )
        result.append(record)
    return result


def _run(execution: SimpleNamespace) -> tuple[dict, dict, dict]:
    schedule = _executed_schedule(execution)

    def simulation() -> dict:
        transition_current_wrench_phase("registered_phase")
        set_current_wrench_phase(None)
        return {
            "status": "COMPLETED",
            "executed_phases": schedule,
            "executed_load_sha256": canonical_sha256(schedule),
        }

    context = _wrench_context_for_execution(execution)
    trace, entry = _run_registered_whole_machine_load_case(
        execution,
        simulation,
    )
    return context, trace, entry


class WholeMachineWrenchAuditTests(unittest.TestCase):
    def test_plain_backend_uses_original_active_plugin_and_common_audit_set(
        self,
    ) -> None:
        plan = SimpleNamespace(
            mechanism_plugin_id="plain_prismatic_v1",
            worker_backend="plain_prismatic_v1",
            axis_abs=(1.0, 0.0, 0.0),
            home_progress_m=0.0,
            outbound_progress_m=0.5,
            max_control_force_n=10.0,
            max_stage_duration_s=1.0,
            moving_mass_kg=1.0,
            step_s=0.01,
            sample_interval_s=0.01,
        )
        execution = _plain_whole_machine_execution(plan)
        context = _wrench_context_for_execution(execution)
        stages = [
            {
                "stage": "outbound",
                "duration_s": 0.5,
                "reached": True,
                "final_coordinate_si": 0.5,
            },
            {
                "stage": "return",
                "duration_s": 0.5,
                "reached": True,
                "final_coordinate_si": 0.0,
            },
        ]
        schedule = _plain_whole_machine_executed_schedule(
            execution.load_case,
            stages,
        )

        def simulation() -> dict:
            transition_current_wrench_phase("outbound")
            set_current_wrench_phase(None)
            transition_current_wrench_phase("return")
            set_current_wrench_phase(None)
            return {
                "status": "COMPLETED",
                "stages": stages,
                "executed_phases": schedule,
                "executed_load_sha256": canonical_sha256(schedule),
            }

        trace, entry = _run_registered_whole_machine_load_case(
            execution,
            simulation,
        )
        payload = _attach_whole_machine_wrench_audits(
            trace,
            expected_contexts=(context,),
            entries=(entry,),
        )
        self.assertEqual(
            payload["wrench_audit"]["plugin_id"],
            "plain_prismatic_v1",
        )
        self.assertNotIn(
            "whole_machine_replacement_v1",
            payload["wrench_audit_summary"]["plugin_ids"],
        )
        self.assertTrue(payload["wrench_audit_summary"]["passed"])

    def test_plain_stage_cannot_claim_tolerance_only_registered_stop(
        self,
    ) -> None:
        plan = SimpleNamespace(
            mechanism_plugin_id="plain_prismatic_v1",
            worker_backend="plain_prismatic_v1",
            axis_abs=(1.0, 0.0, 0.0),
            home_progress_m=0.0,
            outbound_progress_m=0.5,
            max_control_force_n=10.0,
            max_stage_duration_s=1.0,
            moving_mass_kg=1.0,
            step_s=0.01,
            sample_interval_s=0.01,
        )
        execution = _plain_whole_machine_execution(plan)
        with self.assertRaises(MechanismValidationError) as raised:
            _plain_whole_machine_executed_schedule(
                execution.load_case,
                [
                    {
                        "stage": "outbound",
                        "duration_s": 0.5,
                        "reached": True,
                        "final_coordinate_si": 0.5 - 1.0e-12,
                    },
                    {
                        "stage": "return",
                        "duration_s": 0.5,
                        "reached": True,
                        "final_coordinate_si": 0.0,
                    },
                ],
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
        )

    def test_each_nested_backend_uses_its_original_active_plugin_ledger(
        self,
    ) -> None:
        for kind in _BACKENDS:
            with self.subTest(kind=kind):
                context, trace, entry = _run(
                    _execution(kind, f"{kind}-case")
                )
                summary = _validate_whole_machine_wrench_audits(
                    (context,),
                    (entry,),
                )
                self.assertTrue(summary["passed"])
                self.assertEqual(summary["audit_count"], 1)
                self.assertEqual(
                    trace["wrench_audit"]["plugin_id"],
                    _BACKENDS[kind][0],
                )
                self.assertEqual(
                    trace["wrench_audit"]["load_case_id"],
                    f"{kind}-case",
                )

    def test_missing_executed_schedule_digest_is_rejected(self) -> None:
        execution = _execution("rack", "rack-missing-digest")
        schedule = _executed_schedule(execution)

        def simulation() -> dict:
            transition_current_wrench_phase("registered_phase")
            set_current_wrench_phase(None)
            return {
                "status": "COMPLETED",
                "executed_phases": schedule,
            }

        with self.assertRaises(MechanismValidationError) as raised:
            _run_registered_whole_machine_load_case(
                execution,
                simulation,
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_SCHEDULE_INVALID",
        )

    def test_simulator_exception_clears_phase_context(self) -> None:
        execution = _execution("rack", "rack-simulator-error")
        clear_calls = []

        def record_clear(value: str | None) -> None:
            clear_calls.append(value)
            set_current_wrench_phase(value)

        def simulation() -> dict:
            transition_current_wrench_phase("registered_phase")
            raise RuntimeError("simulator failed after selecting a phase")

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "whole_machine_worker.set_current_wrench_phase",
            side_effect=record_clear,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "simulator failed",
            ):
                _run_registered_whole_machine_load_case(
                    execution,
                    simulation,
                )
        self.assertEqual(clear_calls, [None])
        self.assertIsNone(current_wrench_phase())

    def test_phase_record_load_mutation_is_rejected(self) -> None:
        execution = _execution("rack", "rack-changed-load")
        schedule = _executed_schedule(execution)
        schedule[0]["magnitude_si"] = 1.0

        def simulation() -> dict:
            transition_current_wrench_phase("registered_phase")
            set_current_wrench_phase(None)
            return {
                "status": "COMPLETED",
                "executed_phases": schedule,
                "executed_load_sha256": canonical_sha256(schedule),
            }

        with self.assertRaises(MechanismValidationError) as raised:
            _run_registered_whole_machine_load_case(
                execution,
                simulation,
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
        )

    def test_phase_record_unknown_field_is_rejected(self) -> None:
        execution = _execution("rack", "rack-extra-phase-field")
        schedule = _executed_schedule(execution)
        schedule[0]["design_ai_label"] = "looks powered"

        def simulation() -> dict:
            transition_current_wrench_phase("registered_phase")
            set_current_wrench_phase(None)
            return {
                "status": "COMPLETED",
                "executed_phases": schedule,
                "executed_load_sha256": canonical_sha256(schedule),
            }

        with self.assertRaises(MechanismValidationError) as raised:
            _run_registered_whole_machine_load_case(
                execution,
                simulation,
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_PHASE_RECORD_INVALID",
        )

    def test_non_object_schedule_member_is_rejected(self) -> None:
        context, _, entry = _run(
            _execution("rack", "rack-malformed-schedule")
        )
        changed = deepcopy(entry)
        changed["executed_phases"].append("not-a-phase")
        changed["executed_load_sha256"] = canonical_sha256(
            changed["executed_phases"]
        )
        changed["entry_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in changed.items()
                if key != "entry_sha256"
            }
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validate_whole_machine_wrench_audits(
                (context,),
                (changed,),
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_SCHEDULE_MISMATCH",
        )

    def test_wrong_plugin_audit_is_rejected(self) -> None:
        expected, _, _ = _run(_execution("rack", "shared-case"))
        _, _, wrong_entry = _run(
            _execution("compound", "shared-case")
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validate_whole_machine_wrench_audits(
                (expected,),
                (wrong_entry,),
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_CONTEXT_MISMATCH",
        )

    def test_cross_load_case_audit_replay_is_rejected(self) -> None:
        first_context, _, first_entry = _run(
            _execution("clock", "clock-case-a")
        )
        second_context, _, _ = _run(
            _execution("clock", "clock-case-b")
        )
        replay = deepcopy(first_entry)
        for field in (
            "plugin_id",
            "branch",
            "load_case_id",
            "program",
            "phase_ids",
            "input_body_ids",
            "output_body_ids",
            "context_sha256",
        ):
            replay[field] = deepcopy(second_context[field])
        replay["entry_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in replay.items()
                if key != "entry_sha256"
            }
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validate_whole_machine_wrench_audits(
                (first_context, second_context),
                (first_entry, replay),
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_CONTEXT_MISMATCH",
        )

    def test_rehashed_nonadaptive_duration_attack_is_rejected(self) -> None:
        context, _, entry = _run(
            _execution("rack", "rack-duration-attack")
        )
        changed = deepcopy(entry)
        changed["executed_phases"][0]["duration_s"] = 0.05
        changed["executed_load_sha256"] = canonical_sha256(
            changed["executed_phases"]
        )
        changed["entry_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in changed.items()
                if key != "entry_sha256"
            }
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validate_whole_machine_wrench_audits(
                (context,),
                (changed,),
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
        )

    def test_rehashed_one_step_duration_deletion_is_rejected(self) -> None:
        execution = _execution("rack", "rack-one-step-duration-attack")
        context, _, entry = _run(execution)
        changed = deepcopy(entry)
        changed["executed_phases"][0]["duration_s"] = (
            execution.load_case.phases[0].duration_s
            - execution.load_case.step_s
        )
        changed["executed_load_sha256"] = canonical_sha256(
            changed["executed_phases"]
        )
        changed["entry_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in changed.items()
                if key != "entry_sha256"
            }
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validate_whole_machine_wrench_audits(
                (context,),
                (changed,),
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_PHASE_RECORD_MISMATCH",
        )

    def test_context_count_or_extra_field_attack_is_rejected(self) -> None:
        context, _, entry = _run(
            _execution("scissor", "scissor-context-attack")
        )
        for mutation in ("count", "extra"):
            with self.subTest(mutation=mutation):
                changed_context = deepcopy(context)
                changed_entry = deepcopy(entry)
                if mutation == "count":
                    changed_context["phase_count"] += 1
                    changed_entry["phase_count"] += 1
                else:
                    changed_context["design_ai_passed"] = True
                    changed_entry["design_ai_passed"] = True
                changed_context["context_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in changed_context.items()
                        if key != "context_sha256"
                    }
                )
                changed_entry["context_sha256"] = changed_context[
                    "context_sha256"
                ]
                changed_entry["entry_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in changed_entry.items()
                        if key != "entry_sha256"
                    }
                )
                with self.assertRaises(
                    MechanismValidationError
                ) as raised:
                    _validate_whole_machine_wrench_audits(
                        (changed_context,),
                        (changed_entry,),
                    )
                self.assertIn(
                    raised.exception.code,
                    {
                        "WHOLE_MACHINE_WRENCH_CONTEXT_INVALID",
                        "WHOLE_MACHINE_WRENCH_AUDIT_ENTRY_INVALID",
                    },
                )

    def test_clock_multi_case_missing_one_audit_is_rejected(self) -> None:
        contexts = []
        entries = []
        for suffix in ("normal", "setting", "alarm"):
            context, _, entry = _run(
                _execution("clock", f"clock-{suffix}")
            )
            contexts.append(context)
            entries.append(entry)
        with self.assertRaises(MechanismValidationError) as raised:
            _validate_whole_machine_wrench_audits(
                contexts,
                entries[:-1],
            )
        self.assertEqual(
            raised.exception.code,
            "WHOLE_MACHINE_WRENCH_AUDIT_SET_MISMATCH",
        )


class TestWholeMachineMemoryPolicy(unittest.TestCase):
    def test_adaptive_cap_scales_with_total_ram(self):
        import psutil

        with patch.dict(os.environ):
            os.environ.pop("SIMREADY_WHOLE_MACHINE_MAX_MEM_GIB", None)
            expected = max(
                1.0,
                round(0.25 * psutil.virtual_memory().total / (1024**3), 2),
            )
            self.assertEqual(whole_machine._whole_machine_max_memory_gib(), expected)
            self.assertGreaterEqual(whole_machine._whole_machine_max_memory_gib(), 1.0)

    def test_env_override_pins_cap(self):
        with patch.dict(os.environ, {"SIMREADY_WHOLE_MACHINE_MAX_MEM_GIB": "3.5"}):
            self.assertEqual(whole_machine._whole_machine_max_memory_gib(), 3.5)


class TestWholeMachineSourceGuards(unittest.TestCase):
    def test_chain_consumer_reads_branch_semantics(self):
        src = Path(bicycle_chain_whole_machine.__file__).read_text(encoding="utf-8")
        self.assertIn('reference_trace.get("branch_semantics")', src)
        self.assertNotIn('reference_trace.get("semantics")', src)

    def test_dispatch_cap_is_adaptive_not_hardcoded(self):
        src = Path(whole_machine.__file__).read_text(encoding="utf-8")
        self.assertIn(
            "max_process_tree_memory_gib=_whole_machine_max_memory_gib()",
            src,
        )
        self.assertNotIn("max_process_tree_memory_gib=1.0", src)

    def test_chain_whole_machine_has_no_shadow_ledger(self):
        src = Path(bicycle_chain_whole_machine.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_wrench_ledger_for_execution", src)
        self.assertNotIn("activate_wrench_ledger", src)
        self.assertIn("current_wrench_ledger()", src)

    def test_chain_result_payload_stays_slim(self):
        src = Path(bicycle_chain_whole_machine.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn('"samples": samples,', src)
        self.assertIn("write_json_atomic(execution.output_file", src)


if __name__ == "__main__":
    unittest.main()
