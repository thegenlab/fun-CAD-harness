from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    mechanism_adapter_inventory,
    plugin_registration_for,
    registered_mechanism_adapters,
    wrench_policies_for_plugin,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    WRENCH_AUDIT_SCHEMA,
    WrenchLedger,
    activate_wrench_ledger,
    add_ledgered_load,
    create_body_torque,
    current_wrench_ledger,
    current_wrench_phase,
    restore_current_wrench_checkpoint,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
    verify_saved_wrench_audit,
)


class _Vector:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z


class _NativeTorque:
    def __init__(self, body: object, value: _Vector, local: bool) -> None:
        self.body = body
        self.value = value
        self.local = local

    def SetTorque(self, value: _Vector, local: bool) -> None:
        self.value = value
        self.local = local


class _Chrono:
    ChVector3d = _Vector
    ChLoadBodyTorque = _NativeTorque


class _Container:
    def __init__(self) -> None:
        self.loads: list[object] = []

    def Add(self, value: object) -> None:
        self.loads.append(value)


class WrenchLedgerTests(unittest.TestCase):
    plugin_id = "revolute_linkage_v1"

    def _ledger(
        self,
        *,
        registration: dict[str, object] | None = None,
    ) -> WrenchLedger:
        program_phases = [
            {
                "phase_id": "powered_phase",
                "wrench_kind": "torque",
                "direction_abs": [0.0, 1.0, 0.0],
                "magnitude_si": 1.0,
                "duration_s": 1.0,
            },
            {
                "phase_id": "return_phase",
                "wrench_kind": "none",
                "direction_abs": None,
                "magnitude_si": 0.0,
                "duration_s": 1.0,
            },
        ]
        return WrenchLedger(
            plugin_id=self.plugin_id,
            capability_registration=(
                registration
                if registration is not None
                else plugin_registration_for(self.plugin_id)
            ),
            policies=wrench_policies_for_plugin(self.plugin_id),
            declared_input_body_ids=("group_input_crank",),
            declared_output_body_ids=("group_rear_drive_wheel",),
            branch="physical",
            load_case_id="powered-and-return",
            program_phases=program_phases,
        )

    def _phase_bound_input_audit(
        self,
        *,
        return_nonzero: bool = False,
    ) -> dict[str, object]:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            crank = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="phase-bound-crank",
                    owner=self.plugin_id,
                    target_body="group_input_crank",
                    source_role="declared_input_port",
                    target_role="declared_input",
                    geometry_gate="validator_phase_and_axis",
                    active_interval="validator_program",
                    release_required=True,
                ),
            )
            transition_current_wrench_phase("powered_phase")
            crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
            crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            transition_current_wrench_phase("return_phase")
            crank.SetTorque(
                _Vector(0.0, 0.5 if return_nonzero else 0.0, 0.0),
                False,
            )
            crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
            ledger.bind_executed_schedule(
                [dict(item) for item in ledger.program_phases]
            )
        return ledger.verify_for_certification()

    @staticmethod
    def _rehash(audit: dict[str, object]) -> None:
        audit["ledger_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in audit.items()
                if key != "ledger_sha256"
            }
        )

    def test_checkpoint_restores_open_phase_and_mutable_load_state(self) -> None:
        def make_input() -> object:
            return create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="resumable-crank",
                    owner=self.plugin_id,
                    target_body="group_input_crank",
                    source_role="declared_input_port",
                    target_role="declared_input",
                    geometry_gate="validator_phase_and_axis",
                    active_interval="validator_program",
                    release_required=True,
                ),
            )

        first = self._ledger()
        with activate_wrench_ledger(first):
            first_input = make_input()
            transition_current_wrench_phase("powered_phase")
            first_input.SetTorque(_Vector(0.0, 1.0, 0.0), False)
            checkpoint = first.checkpoint_state()
            first_input.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            transition_current_wrench_phase("return_phase")
            first_input.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
            first.bind_executed_schedule(
                [dict(item) for item in first.program_phases]
            )

        resumed = self._ledger()
        with activate_wrench_ledger(resumed):
            resumed_input = make_input()
            restored_phase = restore_current_wrench_checkpoint(checkpoint)
            self.assertEqual(restored_phase, "powered_phase")
            self.assertEqual(current_wrench_phase(), "powered_phase")
            resumed_input.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            transition_current_wrench_phase("return_phase")
            resumed_input.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
            resumed.bind_executed_schedule(
                [dict(item) for item in resumed.program_phases]
            )

        audit = resumed.verify_for_certification()
        self.assertTrue(audit["passed"])
        record = next(
            item
            for item in audit["records"]
            if item["load_id"] == "resumable-crank"
        )
        self.assertEqual(record["nonzero_update_count"], 1)
        self.assertGreaterEqual(record["zero_update_count"], 2)

    def test_two_axis_input_and_geometry_gated_output_transmission_pass(
        self,
    ) -> None:
        ledger = self._ledger()
        container = _Container()
        with activate_wrench_ledger(ledger):
            crank = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="crank-input",
                    owner=self.plugin_id,
                    target_body="group_input_crank",
                    source_role="declared_input_port",
                    target_role="declared_input",
                    geometry_gate="powered_phase_and_crank_axis",
                    active_interval="powered_phase",
                    release_required=True,
                ),
            )
            rear = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="transmission",
                    wrench_kind="torque",
                    load_id="rear-transmission",
                    owner=self.plugin_id,
                    target_body="group_rear_drive_wheel",
                    source_role="crank_transmission",
                    target_role="task_output",
                    geometry_gate="approved_finite_transmission_geometry_gate",
                    active_interval="geometry_gated_powered_phase",
                    release_required=True,
                ),
            )
            add_ledgered_load(container, crank)
            add_ledgered_load(container, rear)
            transition_current_wrench_phase("powered_phase")
            crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
            rear.SetTorque(_Vector(0.0, 2.0, 0.0), False)
            crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            transition_current_wrench_phase("return_phase")
            crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            rear.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
            ledger.bind_executed_schedule(
                [dict(item) for item in ledger.program_phases]
            )

        audit = ledger.verify_for_certification()
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["schema_version"], WRENCH_AUDIT_SCHEMA)
        self.assertEqual(audit["task_input_bodies_observed"], ["group_input_crank"])
        self.assertEqual(audit["transmission_load_count"], 1)
        self.assertEqual(len(container.loads), 2)
        self.assertTrue(
            all(
                record["owner"]
                and record["target_body"]
                and record["law_id"]
                and record["geometry_gate"]
                and record["active_interval"]
                for record in audit["records"]
            )
        )
        self.assertEqual(
            verify_saved_wrench_audit(audit)["ledger_sha256"],
            audit["ledger_sha256"],
        )

    def test_non_task_wrench_updates_are_counted_by_active_phase(self) -> None:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            crank = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="phase-counted-crank",
                    owner=self.plugin_id,
                    target_body="group_input_crank",
                    source_role="declared_input_port",
                    target_role="declared_input",
                    geometry_gate="validator_phase_and_axis",
                    active_interval="validator_program",
                    release_required=True,
                ),
            )
            resistance = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="transmission",
                    wrench_kind="torque",
                    load_id="phase-counted-resistance",
                    owner=self.plugin_id,
                    target_body="group_rear_drive_wheel",
                    source_role="crank_transmission",
                    target_role="task_output",
                    geometry_gate="validator_phase_and_axis",
                    active_interval="validator_program",
                    release_required=True,
                ),
            )
            transition_current_wrench_phase("powered_phase")
            crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
            resistance.SetTorque(_Vector(0.0, -0.25, 0.0), False)
            crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            transition_current_wrench_phase("return_phase")
            crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            resistance.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
            resistance.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            ledger.bind_executed_schedule(
                [dict(item) for item in ledger.program_phases]
            )

        audit = ledger.verify_for_certification()
        record = next(
            item
            for item in audit["records"]
            if item["load_id"] == "phase-counted-resistance"
        )
        self.assertEqual(
            record["phase_update_counts"],
            {"powered_phase": 1, "return_phase": 1},
        )
        self.assertEqual(
            record["phase_nonzero_update_counts"],
            {"powered_phase": 1},
        )
        self.assertEqual(
            record["phase_zero_update_counts"],
            {"return_phase": 1},
        )
        self.assertEqual(record["update_count"], 3)
        self.assertEqual(
            verify_saved_wrench_audit(audit)["ledger_sha256"],
            audit["ledger_sha256"],
        )

    def test_direct_output_actuation_allowed_only_when_contract_permits(
        self,
    ) -> None:
        audit = self._phase_bound_input_audit()
        with self.assertRaises(MechanismValidationError) as default_case:
            verify_saved_wrench_audit(
                audit,
                expected_task_output_body_id="group_input_crank",
            )
        self.assertEqual(
            default_case.exception.code,
            "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
        )
        with self.assertRaises(MechanismValidationError) as forbidden_case:
            verify_saved_wrench_audit(
                audit,
                expected_task_output_body_id="group_input_crank",
                allow_direct_output_actuation=False,
            )
        self.assertEqual(
            forbidden_case.exception.code,
            "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
        )
        verified = verify_saved_wrench_audit(
            audit,
            expected_task_output_body_id="group_input_crank",
            allow_direct_output_actuation=True,
        )
        self.assertEqual(verified["ledger_sha256"], audit["ledger_sha256"])

    def test_composite_program_allows_registered_auxiliary_input_body(self) -> None:
        base = self._ledger()
        ledger = WrenchLedger(
            plugin_id=base.plugin_id,
            capability_registration=base.capability_registration,
            policies=wrench_policies_for_plugin(self.plugin_id),
            declared_input_body_ids=("group_input_crank", "group_auxiliary"),
            declared_output_body_ids=("group_rear_drive_wheel",),
            branch="physical",
            load_case_id="powered-and-return",
            program_phases=[dict(item) for item in base.program_phases],
        )
        with activate_wrench_ledger(ledger):
            inputs = []
            for load_id, body_id in (
                ("primary-input", "group_input_crank"),
                ("auxiliary-input", "group_auxiliary"),
            ):
                inputs.append(
                    create_body_torque(
                        _Chrono,
                        object(),
                        _Vector(0.0, 0.0, 0.0),
                        False,
                        declaration=standard_wrench_declaration(
                            plugin_id=self.plugin_id,
                            profile="task_input",
                            wrench_kind="torque",
                            load_id=load_id,
                            owner=self.plugin_id,
                            target_body=body_id,
                            source_role="declared_input_port",
                            target_role="declared_input",
                            geometry_gate="validator_phase_and_axis",
                            active_interval="validator_program",
                            release_required=True,
                        ),
                    )
                )
            transition_current_wrench_phase("powered_phase")
            for item in inputs:
                item.SetTorque(_Vector(0.0, 1.0, 0.0), False)
                item.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            transition_current_wrench_phase("return_phase")
            for item in inputs:
                item.SetTorque(_Vector(0.0, 0.0, 0.0), False)
            set_current_wrench_phase(None)
            ledger.bind_executed_schedule(
                [dict(item) for item in ledger.program_phases]
            )
        audit = ledger.verify_for_certification()
        verified = verify_saved_wrench_audit(
            audit,
            expected_input_body_ids=("group_input_crank", "group_auxiliary"),
            expected_task_input_body_ids=("group_input_crank",),
            allow_multiple_task_input_channels=True,
        )
        self.assertEqual(verified["ledger_sha256"], audit["ledger_sha256"])

    def test_two_axis_direct_output_drive_is_rejected_at_construction(
        self,
    ) -> None:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            with self.assertRaises(MechanismValidationError) as raised:
                create_body_torque(
                    _Chrono,
                    object(),
                    _Vector(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=self.plugin_id,
                        profile="task_input",
                        wrench_kind="torque",
                        load_id="hidden-rear-motor",
                        owner=self.plugin_id,
                        target_body="group_rear_drive_wheel",
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="forged_powered_phase",
                        active_interval="powered_phase",
                        release_required=True,
                    ),
                )
        self.assertEqual(
            raised.exception.code,
            "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
        )

    def test_aggregated_output_drive_attack_is_rejected(self) -> None:
        ledger = self._ledger()
        with self.assertRaises(MechanismValidationError) as raised:
            ledger.record_task_input_event(
                target_body="group_rear_drive_wheel",
                source_role="declared_input_port",
                active_interval="powered_phase",
                vector_si=(0.0, 1.0, 0.0),
            )
        self.assertEqual(
            raised.exception.code,
            "WRENCH_DIRECT_OUTPUT_DRIVE_FORBIDDEN",
        )

    def test_inactive_capability_cannot_apply_analytical_transmission(
        self,
    ) -> None:
        registration = plugin_registration_for(self.plugin_id)
        registration["status"] = "INACTIVE"
        ledger = self._ledger(registration=registration)
        with activate_wrench_ledger(ledger):
            with self.assertRaises(MechanismValidationError) as raised:
                create_body_torque(
                    _Chrono,
                    object(),
                    _Vector(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=self.plugin_id,
                        profile="transmission",
                        wrench_kind="torque",
                        load_id="inactive-transmission",
                        owner=self.plugin_id,
                        target_body="group_rear_drive_wheel",
                        source_role="crank_transmission",
                        target_role="task_output",
                        geometry_gate="finite_geometry_gate",
                        active_interval="geometry_gated_interval",
                    ),
                )
        self.assertEqual(raised.exception.code, "WRENCH_CAPABILITY_NOT_ACTIVE")

    def test_release_residual_blocks_certification(self) -> None:
        ledger = self._ledger()
        with self.assertRaises(MechanismValidationError) as raised:
            with activate_wrench_ledger(ledger):
                crank = create_body_torque(
                    _Chrono,
                    object(),
                    _Vector(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=self.plugin_id,
                        profile="task_input",
                        wrench_kind="torque",
                        load_id="unreleased-crank-input",
                        owner=self.plugin_id,
                        target_body="group_input_crank",
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="powered_phase_and_crank_axis",
                        active_interval="powered_phase",
                        release_required=True,
                    ),
                )
                transition_current_wrench_phase("powered_phase")
                crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
                transition_current_wrench_phase("return_phase")
        self.assertEqual(
            raised.exception.code,
            "WRENCH_PHASE_ENTRY_TASK_INPUT_NONZERO",
        )

    def test_saved_audit_tampering_is_rejected(self) -> None:
        audit = self._phase_bound_input_audit()
        audit["plugin_id"] = "forged_plugin"
        with self.assertRaises(MechanismValidationError) as raised:
            verify_saved_wrench_audit(audit)
        self.assertEqual(raised.exception.code, "WRENCH_AUDIT_HASH_INVALID")

    def test_nonzero_task_input_without_phase_fails_before_native_mutation(
        self,
    ) -> None:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            crank = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="unset-phase-crank",
                    owner=self.plugin_id,
                    target_body="group_input_crank",
                    source_role="declared_input_port",
                    target_role="declared_input",
                    geometry_gate="validator_phase_and_axis",
                    active_interval="validator_program",
                ),
            )
            with self.assertRaises(MechanismValidationError) as raised:
                crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
        self.assertEqual(raised.exception.code, "WRENCH_TASK_INPUT_PHASE_UNSET")
        self.assertEqual(crank.native_load.value.y, 0.0)

    def test_wrong_task_input_vector_fails_before_native_mutation(self) -> None:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            crank = create_body_torque(
                _Chrono,
                object(),
                _Vector(0.0, 0.0, 0.0),
                False,
                declaration=standard_wrench_declaration(
                    plugin_id=self.plugin_id,
                    profile="task_input",
                    wrench_kind="torque",
                    load_id="wrong-law-crank",
                    owner=self.plugin_id,
                    target_body="group_input_crank",
                    source_role="declared_input_port",
                    target_role="declared_input",
                    geometry_gate="validator_phase_and_axis",
                    active_interval="validator_program",
                ),
            )
            transition_current_wrench_phase("powered_phase")
            with self.assertRaises(MechanismValidationError) as raised:
                crank.SetTorque(_Vector(999.0, 0.0, 0.0), False)
        self.assertEqual(
            raised.exception.code,
            "WRENCH_TASK_INPUT_LAW_MISMATCH",
        )
        self.assertEqual(crank.native_load.value.x, 0.0)

    def test_stale_phase_context_is_rejected(self) -> None:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            set_current_wrench_phase("powered_phase")
            with self.assertRaises(MechanismValidationError) as raised:
                set_current_wrench_phase("return_phase")
        self.assertEqual(raised.exception.code, "WRENCH_PHASE_CONTEXT_STALE")

    def test_reviewer_wrench_audit_failure_carries_owner_and_causes(self) -> None:
        ledger = self._ledger(
            registration={
                "registration_source": "reviewer_mechanism_adapter",
            }
        )
        ledger.begin_phase("powered_phase")
        ledger.bind_executed_schedule(
            [dict(item) for item in ledger.program_phases]
        )

        with self.assertRaises(MechanismValidationError) as raised:
            ledger.verify_for_certification()

        failure = raised.exception.failure_envelope
        self.assertIsNotNone(failure)
        self.assertEqual(failure["code"], "WRENCH_AUDIT_FAILED")
        self.assertEqual(failure["owner"], "reviewer_mechanism_plugin")
        self.assertEqual(failure["source"]["plugin_kind"], "mechanism")
        self.assertEqual(
            {item["code"] for item in failure["causes"]},
            {
                "WRENCH_PHASE_CONTEXT_STALE",
                "WRENCH_PHASE_BARRIER_COVERAGE_INCOMPLETE",
            },
        )

    def test_unknown_phase_context_is_rejected(self) -> None:
        ledger = self._ledger()
        with activate_wrench_ledger(ledger):
            with self.assertRaises(MechanismValidationError) as raised:
                set_current_wrench_phase("unregistered_phase")
        self.assertEqual(
            raised.exception.code,
            "WRENCH_TASK_INPUT_PHASE_UNKNOWN",
        )

    def test_passive_return_rejects_any_nonzero_task_input(self) -> None:
        ledger = self._ledger()
        with self.assertRaises(MechanismValidationError) as raised:
            with activate_wrench_ledger(ledger):
                crank = create_body_torque(
                    _Chrono,
                    object(),
                    _Vector(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=self.plugin_id,
                        profile="task_input",
                        wrench_kind="torque",
                        load_id="passive-return-attack",
                        owner=self.plugin_id,
                        target_body="group_input_crank",
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="validator_phase_and_axis",
                        active_interval="validator_program",
                    ),
                )
                transition_current_wrench_phase("powered_phase")
                crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
                crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
                transition_current_wrench_phase("return_phase")
                crank.SetTorque(_Vector(0.0, 0.5, 0.0), False)
        self.assertEqual(
            raised.exception.code,
            "WRENCH_NONE_PHASE_TASK_INPUT_NONZERO",
        )

    def test_passive_return_rejects_sub_tolerance_nonzero_input(self) -> None:
        ledger = self._ledger()
        with self.assertRaises(MechanismValidationError) as raised:
            with activate_wrench_ledger(ledger):
                crank = create_body_torque(
                    _Chrono,
                    object(),
                    _Vector(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=self.plugin_id,
                        profile="task_input",
                        wrench_kind="torque",
                        load_id="tiny-passive-return-attack",
                        owner=self.plugin_id,
                        target_body="group_input_crank",
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="validator_phase_and_axis",
                        active_interval="validator_program",
                    ),
                )
                transition_current_wrench_phase("powered_phase")
                crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
                crank.SetTorque(_Vector(0.0, 0.0, 0.0), False)
                transition_current_wrench_phase("return_phase")
                crank.SetTorque(_Vector(0.0, 1.0e-15, 0.0), False)
        self.assertEqual(
            raised.exception.code,
            "WRENCH_NONE_PHASE_TASK_INPUT_NONZERO",
        )

    def test_wrench_audit_trace_context_mismatch_is_rejected(self) -> None:
        audit = self._phase_bound_input_audit()
        with self.assertRaises(MechanismValidationError) as raised:
            verify_saved_wrench_audit(
                audit,
                expected_plugin_id=self.plugin_id,
                expected_branch="physical",
                expected_load_case_id="different-load-case",
            )
        self.assertEqual(
            raised.exception.code,
            "WRENCH_AUDIT_CONTEXT_MISMATCH",
        )

    def test_phase_barrier_rehashed_counter_attack_is_rejected(self) -> None:
        audit = copy.deepcopy(self._phase_bound_input_audit())
        barrier = audit["phase_barriers"][0]
        barrier["task_input_update_count"] += 1
        barrier["barrier_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in barrier.items()
                if key != "barrier_sha256"
            }
        )
        self._rehash(audit)
        with self.assertRaises(MechanismValidationError) as raised:
            verify_saved_wrench_audit(audit)
        self.assertEqual(
            raised.exception.code,
            "WRENCH_PHASE_BARRIER_INVALID",
        )

    def test_phase_barrier_missing_or_extra_key_is_rejected(self) -> None:
        for mutation in ("missing", "extra"):
            with self.subTest(mutation=mutation):
                audit = copy.deepcopy(self._phase_bound_input_audit())
                barrier = audit["phase_barriers"][0]
                if mutation == "missing":
                    del barrier["entry_all_zero"]
                else:
                    barrier["forged_state"] = "zero"
                barrier["barrier_sha256"] = canonical_sha256(
                    {
                        key: value
                        for key, value in barrier.items()
                        if key != "barrier_sha256"
                    }
                )
                self._rehash(audit)
                with self.assertRaises(MechanismValidationError) as raised:
                    verify_saved_wrench_audit(audit)
                self.assertEqual(
                    raised.exception.code,
                    "WRENCH_PHASE_BARRIER_INVALID",
                )

    def test_phase_bucket_and_total_counter_attacks_are_rejected(self) -> None:
        mutations = (
            "extra_phase",
            "missing_bucket",
            "wrong_total",
            "wrong_global_pair",
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                audit = copy.deepcopy(self._phase_bound_input_audit())
                record = audit["records"][0]
                if mutation == "extra_phase":
                    record["phase_update_counts"]["forged_phase"] = 1
                    record["phase_zero_update_counts"]["forged_phase"] = 1
                elif mutation == "missing_bucket":
                    del record["phase_zero_update_counts"]["return_phase"]
                elif mutation == "wrong_global_pair":
                    record["update_count"] += 1
                    record["zero_update_count"] += 1
                else:
                    record["update_count"] += 1
                self._rehash(audit)
                with self.assertRaises(MechanismValidationError) as raised:
                    verify_saved_wrench_audit(audit)
                self.assertIn(
                    raised.exception.code,
                    {
                        "WRENCH_PHASE_AUDIT_INVALID",
                        "WRENCH_AUDIT_COUNTER_INVALID",
                    },
                )

    def test_rehashed_task_input_source_attack_is_rejected(self) -> None:
        audit = copy.deepcopy(self._phase_bound_input_audit())
        audit["records"][0]["source_role"] = "hidden_controller"
        audit["task_input_sources_observed"] = ["hidden_controller"]
        self._rehash(audit)
        with self.assertRaises(MechanismValidationError) as raised:
            verify_saved_wrench_audit(
                audit,
                expected_task_input_body_id="group_input_crank",
                return_phase_ids=("return_phase",),
                return_input_policy="selected_input_only",
            )
        self.assertEqual(raised.exception.code, "WRENCH_INPUT_SOURCE_INVALID")

    def test_rehashed_program_law_attack_is_rejected(self) -> None:
        audit = copy.deepcopy(self._phase_bound_input_audit())
        expected_program = copy.deepcopy(audit["program_phases"])
        audit["program_phases"][0]["direction_abs"] = [1.0, 0.0, 0.0]
        audit["program_phases"][0]["magnitude_si"] = 999.0
        self._rehash(audit)
        with self.assertRaises(MechanismValidationError) as raised:
            verify_saved_wrench_audit(
                audit,
                expected_phase_ids=("powered_phase", "return_phase"),
                expected_program_phases=expected_program,
                expected_executed_load_sha256=audit[
                    "executed_load_sha256"
                ],
            )
        self.assertEqual(
            raised.exception.code,
            "WRENCH_AUDIT_PROGRAM_LAW_MISMATCH",
        )

    def test_audit_extra_top_level_key_is_rejected_even_if_rehashed(self) -> None:
        audit = copy.deepcopy(self._phase_bound_input_audit())
        audit["forged_pass_reason"] = "looks_good"
        self._rehash(audit)
        with self.assertRaises(MechanismValidationError) as raised:
            verify_saved_wrench_audit(audit)
        self.assertEqual(raised.exception.code, "WRENCH_AUDIT_SCHEMA_INVALID")

    def test_cleanup_rejects_residual_and_resets_context(self) -> None:
        ledger = self._ledger()
        with self.assertRaises(MechanismValidationError) as raised:
            with activate_wrench_ledger(ledger):
                crank = create_body_torque(
                    _Chrono,
                    object(),
                    _Vector(0.0, 0.0, 0.0),
                    False,
                    declaration=standard_wrench_declaration(
                        plugin_id=self.plugin_id,
                        profile="task_input",
                        wrench_kind="torque",
                        load_id="cleanup-residual",
                        owner=self.plugin_id,
                        target_body="group_input_crank",
                        source_role="declared_input_port",
                        target_role="declared_input",
                        geometry_gate="validator_phase_and_axis",
                        active_interval="validator_program",
                    ),
                )
                transition_current_wrench_phase("powered_phase")
                crank.SetTorque(_Vector(0.0, 1.0, 0.0), False)
        self.assertEqual(
            raised.exception.code,
            "WRENCH_LEDGER_CLEANUP_TASK_INPUT_NONZERO",
        )
        self.assertIsNone(current_wrench_phase())
        with self.assertRaises(MechanismValidationError) as missing:
            current_wrench_ledger()
        self.assertEqual(missing.exception.code, "WRENCH_LEDGER_REQUIRED")

    def test_every_registered_plugin_has_an_explicit_active_policy_profile(
        self,
    ) -> None:
        for plugin in registered_mechanism_adapters():
            registration = plugin_registration_for(plugin.plugin_id)
            policies = wrench_policies_for_plugin(plugin.plugin_id)
            self.assertEqual(registration["status"], "ACTIVE")
            self.assertTrue(policies, plugin.plugin_id)
            self.assertEqual(
                len({item.policy_id for item in policies}),
                len(policies),
            )
        inventory = mechanism_adapter_inventory()
        self.assertTrue(
            all(item["allowed_wrench_policies"] for item in inventory)
        )

    def test_production_ast_forbids_direct_native_body_wrench_constructors(
        self,
    ) -> None:
        pipeline_root = Path(__file__).resolve().parents[1]
        violations: list[str] = []
        for path in sorted(pipeline_root.rglob("*.py")):
            if "tests" in path.parts or path.name == "wrench_ledger.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "native_load"
                ):
                    violations.append(f"{path.name}:{node.lineno}:native_load")
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if (
                    isinstance(function, ast.Attribute)
                    and function.attr
                    in {"ChLoadBodyForce", "ChLoadBodyTorque"}
                ):
                    violations.append(f"{path.name}:{node.lineno}")
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
