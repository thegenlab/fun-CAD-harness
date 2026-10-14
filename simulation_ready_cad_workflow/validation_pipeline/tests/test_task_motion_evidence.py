from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.mechanism.acceptance_metrics import (
    MetricEvaluationError,
    evaluate_registered_metric,
    evaluate_task_acceptance,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.test_load_profiles import (
    binding_for_test_load_profile,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    plugin_registration_for,
    wrench_policies_for_plugin,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.task_motion_evidence import (
    _runtime_channel_layout,
    ResolvedTaskChannel,
    TASK_WRENCH_COMPONENTS_SCHEMA,
    TaskChannelRegistry,
    registered_adaptive_termination_evidence,
    resolve_and_validate_task_channels,
    task_channel_registry_for_plan,
    task_program_spec_for_load_case,
    validate_trace_program,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.wrench_ledger import (
    WrenchLedger,
    activate_wrench_ledger,
    set_current_wrench_phase,
    standard_wrench_declaration,
    transition_current_wrench_phase,
)


def _load_case() -> LoadCase:
    return LoadCase(
        load_case_id="task_output_angle_roundtrip",
        family="compound",
        program="task_output_angle_roundtrip",
        purpose="nominal",
        moving_mass_kg=2.0,
        step_s=0.01,
        sample_interval_s=0.25,
        phases=(
            LoadPhase(
                "press_pedal",
                "torque",
                (0.0, 1.0, 0.0),
                1.0,
                1.0,
            ),
            LoadPhase("open_settle", "none", None, 0.0, 0.5),
            LoadPhase(
                "release_pedal",
                "torque",
                (0.0, -1.0, 0.0),
                0.4,
                0.5,
            ),
            LoadPhase("closed_settle", "none", None, 0.0, 0.5),
        ),
    )


def _schedule(load_case: LoadCase) -> list[dict]:
    return [
        {
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
        for phase in load_case.phases
    ]


def _samples(
    *,
    negative: bool = False,
    powered_output: tuple[float, float, float, float] | None = None,
    powered_input: tuple[float, float, float, float] | None = None,
    return_peak: float | None = None,
) -> list[dict]:
    sign = -1.0 if negative else 1.0
    outputs = powered_output or (0.5, 1.0, 2.0, 3.0)
    inputs = powered_input or (0.25, 0.5, 1.0, 1.5)
    values = [
        (0.0, "initial", 0.0, 0.0),
        (0.25, "press_pedal", sign * outputs[0], sign * inputs[0]),
        (0.50, "press_pedal", sign * outputs[1], sign * inputs[1]),
        (0.75, "press_pedal", sign * outputs[2], sign * inputs[2]),
        (1.00, "press_pedal", sign * outputs[3], sign * inputs[3]),
        (1.25, "open_settle", sign * outputs[3], sign * inputs[3]),
        (1.50, "open_settle", sign * outputs[3], sign * inputs[3]),
        (
            1.75,
            "release_pedal",
            sign * (return_peak if return_peak is not None else 1.5),
            sign * 0.75,
        ),
        (2.00, "release_pedal", sign * 0.2, sign * 0.1),
        (2.25, "closed_settle", sign * 0.1, 0.0),
        (2.50, "closed_settle", sign * 0.1, 0.0),
    ]
    return [
        {
            "time_s": time_s,
            "phase_id": phase_id,
            "compound_output_angle_rad": output,
            "compound_input_angle_rad": reference,
        }
        for time_s, phase_id, output, reference in values
    ]


def _trace(samples: list[dict]) -> dict:
    load_case = _load_case()
    schedule = _schedule(load_case)
    return {
        "status": "COMPLETED",
        "branch": "physical",
        "load_case_id": load_case.load_case_id,
        "sample_count": len(samples),
        "executed_phases": schedule,
        "executed_load_sha256": canonical_sha256(schedule),
        "samples": samples,
    }


def _registry() -> TaskChannelRegistry:
    input_channel = ResolvedTaskChannel(
        semantic_id="input_crank_torque",
        rigid_group_id="group_input_crank",
        body_id="crank_body",
        frame_id="frame_crank",
        canonical_field="compound_input_angle_rad",
        axis_abs=(0.0, 1.0, 0.0),
        wrench_type="torque",
    )
    crank_observable = ResolvedTaskChannel(
        semantic_id="input_crank_pose",
        rigid_group_id="group_input_crank",
        body_id="crank_body",
        frame_id="frame_crank",
        canonical_field="compound_input_angle_rad",
        axis_abs=(0.0, 1.0, 0.0),
        quantity="pose_6d",
    )
    rear_observable = ResolvedTaskChannel(
        semantic_id="output_wheel_pose",
        rigid_group_id="group_rear_drive_wheel",
        body_id="rear_body",
        frame_id="frame_rear",
        canonical_field="compound_output_angle_rad",
        axis_abs=(0.0, 1.0, 0.0),
        quantity="pose_6d",
    )
    return TaskChannelRegistry(
        plugin_id="revolute_linkage_v1",
        input_body_ids=("crank_body",),
        output_body_ids=("rear_body",),
        inputs={"input_crank_torque": input_channel},
        observables={
            "input_crank_pose": crank_observable,
            "output_wheel_pose": rear_observable,
        },
    )


def _binding(*, reference: bool = False) -> dict:
    result = {
        "input": {
            "port_id": "input_crank_torque",
            "rigid_group_id": "group_input_crank",
            "frame_id": "frame_crank",
            "wrench_type": "torque",
            "direction_abs": [0.0, 1.0, 0.0],
        },
        "target": {
            "observable_id": "output_wheel_pose",
            "rigid_group_id": "group_rear_drive_wheel",
            "frame_id": "frame_rear",
            "quantity": "pose_6d",
        },
    }
    if reference:
        result["reference"] = {
            "observable_id": "input_crank_pose",
            "rigid_group_id": "group_input_crank",
            "frame_id": "frame_crank",
            "quantity": "pose_6d",
        }
    return result


def _requirement(metric_id: str, *, direction: str = "either") -> dict:
    target = {
        "comparator": "within",
        "value": 2.0,
        "tolerance": 0.01,
        "unit": "ratio",
    }
    if metric_id == "continuous_rotation_rad":
        target = {"comparator": "at_least", "value": 2.8, "unit": "rad"}
    if metric_id == "return_to_start_rad":
        target = {"comparator": "at_most", "value": 0.25, "unit": "rad"}
    if metric_id == "output_torque_nm":
        target = {"comparator": "at_most", "value": 0.05, "unit": "N·m"}
    return {
        "requirement_id": f"req_{metric_id}",
        "task_id": "task_two_axis_drivetrain",
        "metric_id": metric_id,
        "direction": direction,
        "target": target,
        "allow_direct_output_actuation": False,
    }


def _validated(trace: dict):
    load_case = _load_case()
    spec = task_program_spec_for_load_case(
        load_case,
        plugin_id="revolute_linkage_v1",
    )
    return validate_trace_program(trace, load_case, program_spec=spec)


def _wrench_audit(
    *,
    include_output_torque_inventory: bool = False,
    transmission_target_body: str = "rear_body",
) -> dict:
    load_case = _load_case()
    ledger = WrenchLedger(
        plugin_id="revolute_linkage_v1",
        capability_registration=plugin_registration_for(
            "revolute_linkage_v1"
        ),
        policies=wrench_policies_for_plugin("revolute_linkage_v1"),
        declared_input_body_ids=("crank_body",),
        declared_output_body_ids=("rear_body",),
        branch="physical",
        load_case_id=load_case.load_case_id,
        program_phases=(
            phase.to_dict() for phase in load_case.phases
        ),
    )
    declaration = standard_wrench_declaration(
        plugin_id="revolute_linkage_v1",
        profile="task_input",
        wrench_kind="torque",
        load_id="crank-task-input",
        owner="revolute_linkage_v1",
        target_body="crank_body",
        source_role="declared_input_port",
        target_role="declared_input",
        geometry_gate="validator_program_and_crank_axis",
        active_interval="validator_program",
        release_required=True,
    )
    transmission = standard_wrench_declaration(
        plugin_id="revolute_linkage_v1",
        profile="transmission",
        wrench_kind="torque",
        load_id="output-transmission",
        owner="revolute_linkage_v1",
        target_body=transmission_target_body,
        source_role="finite_internal_transmission",
        target_role=(
            "task_output"
            if transmission_target_body == "rear_body"
            else "declared_input"
        ),
        geometry_gate="approved_finite_transmission_geometry_gate",
        active_interval="validator_program",
        release_required=True,
    )
    support = standard_wrench_declaration(
        plugin_id="revolute_linkage_v1",
        profile="support",
        wrench_kind="torque",
        load_id="output-passive-support",
        owner="revolute_linkage_v1",
        target_body="rear_body",
        source_role="finite_passive_support",
        target_role="task_output",
        geometry_gate="finite_support_geometry",
        active_interval="validator_program",
        release_required=True,
    )
    with activate_wrench_ledger(ledger):
        ledger.register(
            declaration,
            wrench_kind="torque",
            initial_vector=(0.0, 0.0, 0.0),
        )
        if include_output_torque_inventory:
            ledger.register(
                transmission,
                wrench_kind="torque",
                initial_vector=(0.0, 0.0, 0.0),
            )
            ledger.register(
                support,
                wrench_kind="torque",
                initial_vector=(0.0, 0.0, 0.0),
            )
        transition_current_wrench_phase("press_pedal")
        ledger.record_update("crank-task-input", (0.0, 1.0, 0.0))
        if include_output_torque_inventory:
            ledger.record_update("output-transmission", (0.0, 8.0, 0.0))
            ledger.record_update("output-passive-support", (0.0, 25.0, 0.0))
        ledger.record_update("crank-task-input", (0.0, 0.0, 0.0))
        transition_current_wrench_phase("open_settle")
        ledger.record_update("crank-task-input", (0.0, 0.0, 0.0))
        if include_output_torque_inventory:
            ledger.record_update("output-transmission", (0.0, 0.0, 0.0))
            ledger.record_update("output-passive-support", (0.0, 0.0, 0.0))
        transition_current_wrench_phase("release_pedal")
        ledger.record_update("crank-task-input", (0.0, -0.4, 0.0))
        if include_output_torque_inventory:
            ledger.record_update("output-transmission", (0.0, 0.04, 0.0))
            ledger.record_update("output-passive-support", (0.0, 25.0, 0.0))
        ledger.record_update("crank-task-input", (0.0, 0.0, 0.0))
        transition_current_wrench_phase("closed_settle")
        ledger.record_update("crank-task-input", (0.0, 0.0, 0.0))
        if include_output_torque_inventory:
            ledger.record_update("output-transmission", (0.0, 0.0, 0.0))
            ledger.record_update("output-passive-support", (0.0, 0.0, 0.0))
        set_current_wrench_phase(None)
        ledger.bind_executed_schedule(_schedule(load_case))
    return ledger.verify_for_certification()


def _output_torque_trace(
    *,
    return_torque_nm: float = 0.04,
    powered_torque_nm: float = 8.0,
) -> dict:
    trace = _trace(_samples())
    trace["wrench_components_schema_version"] = (
        TASK_WRENCH_COMPONENTS_SCHEMA
    )
    for sample in trace["samples"]:
        phase_id = sample["phase_id"]
        transmission_torque = (
            powered_torque_nm
            if phase_id == "press_pedal"
            else return_torque_nm
        )
        sample["wrench_components"] = [
            {
                "load_id": "output-transmission",
                "target_body": "rear_body",
                "wrench_kind": "torque",
                "vector_si": [0.0, transmission_torque, 0.0],
                "axis_abs": [0.0, 1.0, 0.0],
            },
            {
                "load_id": "output-passive-support",
                "target_body": "rear_body",
                "wrench_kind": "torque",
                "vector_si": [0.0, 25.0, 0.0],
                "axis_abs": [0.0, 1.0, 0.0],
            },
            {
                "load_id": "crank-task-input",
                "target_body": "crank_body",
                "wrench_kind": "torque",
                "vector_si": [0.0, 100.0, 0.0],
                "axis_abs": [0.0, 1.0, 0.0],
            },
        ]
    return trace


class TaskMotionEvidenceTests(unittest.TestCase):
    def test_bodyless_two_body_layout_maps_fixed_reference_groups(self) -> None:
        plugin = SimpleNamespace(
            plugin_id="bodyless_prismatic_test",
            coordinate=SimpleNamespace(position_field="generalized_progress_m"),
            task_input_actuation_spans_all_inputs=False,
            task_channel_layout=lambda _plan, _family_spec, _body_ids: None,
        )
        adapter = SimpleNamespace(family_spec={})
        plan = SimpleNamespace(
            capsule_id="bodyless_two_body_test",
            fixed_group_ids=("group_housing",),
            moving_group_ids=("group_carriage",),
            axis_abs=(0.0, 0.0, 1.0),
        )
        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins.mechanism_adapter_for_plan",
                return_value=(plugin, adapter),
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins.plugin_registration_for",
                return_value={"status": "ACTIVE"},
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins.registration_allows_runtime_execution",
                return_value=True,
            ),
        ):
            _plugin, group_to_body, inputs, outputs, fields, axes = (
                _runtime_channel_layout(plan)
            )
        self.assertEqual(group_to_body["group_housing"], "fixed_group")
        self.assertEqual(group_to_body["group_carriage"], "moving_group")
        self.assertEqual(inputs, {"moving_group"})
        self.assertEqual(outputs, {"moving_group"})
        self.assertEqual(fields["moving_group"], "generalized_progress_m")
        self.assertEqual(axes["moving_group"], (0.0, 0.0, 1.0))

    def test_registered_stop_normalizes_only_machine_scale_boundary_error(
        self,
    ) -> None:
        phase = SimpleNamespace(
            phase_id="hard_limit_stop",
            stop_condition="position_at_least",
            target_coordinate=0.2,
            branch_termination_policy="each_branch_target",
        )
        evidence = registered_adaptive_termination_evidence(
            phase,
            actual_coordinate_si=0.2 - math.ulp(0.2),
            reached=True,
        )
        self.assertEqual(evidence["actual_coordinate_si"], 0.2)
        with self.assertRaises(MechanismValidationError):
            registered_adaptive_termination_evidence(
                phase,
                actual_coordinate_si=0.2 - 1.0e-9,
                reached=True,
            )

    def test_positive_and_negative_direction_powered_roundtrip_pass(self) -> None:
        for negative in (False, True):
            with self.subTest(negative=negative):
                trace = _trace(_samples(negative=negative))
                validated = _validated(trace)
                channels = resolve_and_validate_task_channels(
                    _requirement("angular_ratio"),
                    _binding(reference=True),
                    _registry(),
                )
                ratio = evaluate_registered_metric(
                    _requirement("angular_ratio"),
                    _binding(reference=True),
                    trace,
                    validated_program=validated,
                    resolved_channels=channels,
                )
                rotation_channels = resolve_and_validate_task_channels(
                    _requirement("continuous_rotation_rad"),
                    _binding(),
                    _registry(),
                )
                rotation = evaluate_registered_metric(
                    _requirement("continuous_rotation_rad"),
                    _binding(),
                    trace,
                    validated_program=validated,
                    resolved_channels=rotation_channels,
                )
                returned = evaluate_registered_metric(
                    _requirement("return_to_start_rad"),
                    _binding(),
                    trace,
                    validated_program=validated,
                    resolved_channels=rotation_channels,
                )
                self.assertTrue(ratio["passed"])
                self.assertTrue(rotation["passed"])
                self.assertTrue(returned["passed"])

    def test_post_release_excursion_is_not_counted_as_powered(self) -> None:
        trace = _trace(
            _samples(
                powered_output=(0.05, 0.1, 0.15, 0.2),
                powered_input=(0.025, 0.05, 0.075, 0.1),
                return_peak=100.0,
            )
        )
        requirement = _requirement("continuous_rotation_rad")
        channels = resolve_and_validate_task_channels(
            requirement, _binding(), _registry()
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            validated_program=_validated(trace),
            resolved_channels=channels,
        )
        self.assertFalse(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.2)

    def test_output_torque_uses_only_audited_return_transmission_components(
        self,
    ) -> None:
        trace = _output_torque_trace(
            powered_torque_nm=80.0,
            return_torque_nm=0.04,
        )
        requirement = _requirement("output_torque_nm")
        channels = resolve_and_validate_task_channels(
            requirement,
            _binding(),
            _registry(),
        )

        result = evaluate_registered_metric(
            requirement,
            _binding(),
            trace,
            validated_program=_validated(trace),
            resolved_channels=channels,
            verified_wrench_audit=_wrench_audit(
                include_output_torque_inventory=True
            ),
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["measured"], 0.04)
        self.assertEqual(
            result["evidence"]["output_torque_load_ids"],
            ["output-transmission"],
        )
        self.assertEqual(
            result["evidence"]["output_torque_axis_abs"],
            [0.0, 1.0, 0.0],
        )

    def test_output_torque_rejects_missing_program_window_and_components(
        self,
    ) -> None:
        trace = _output_torque_trace()
        requirement = _requirement("output_torque_nm")
        channels = resolve_and_validate_task_channels(
            requirement,
            _binding(),
            _registry(),
        )
        audit = _wrench_audit(include_output_torque_inventory=True)
        validated = _validated(trace)
        attacks = []
        attacks.append(("TASK_OUTPUT_TORQUE_PROGRAM_MISSING", trace, None))
        attacks.append(
            (
                "TASK_OUTPUT_TORQUE_RETURN_WINDOW_MISSING",
                trace,
                replace(validated, return_sample_indices=()),
            )
        )
        schema_missing = deepcopy(trace)
        del schema_missing["wrench_components_schema_version"]
        attacks.append(
            (
                "TASK_OUTPUT_TORQUE_COMPONENT_SCHEMA_INVALID",
                schema_missing,
                validated,
            )
        )
        components_missing = deepcopy(trace)
        del components_missing["samples"][
            validated.return_sample_indices[0]
        ]["wrench_components"]
        attacks.append(
            (
                "TASK_OUTPUT_TORQUE_COMPONENT_EVIDENCE_MISSING",
                components_missing,
                validated,
            )
        )
        for code, attacked_trace, attacked_program in attacks:
            with self.subTest(code=code):
                with self.assertRaises(MetricEvaluationError) as raised:
                    evaluate_registered_metric(
                        requirement,
                        _binding(),
                        attacked_trace,
                        validated_program=attacked_program,
                        resolved_channels=channels,
                        verified_wrench_audit=audit,
                    )
                self.assertEqual(raised.exception.code, code)

        with self.assertRaises(MetricEvaluationError) as raised:
            evaluate_registered_metric(
                requirement,
                _binding(),
                trace,
                validated_program=validated,
                resolved_channels=channels,
                verified_wrench_audit=None,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_OUTPUT_TORQUE_AUDIT_MISSING",
        )
        with self.assertRaises(MetricEvaluationError) as raised:
            evaluate_registered_metric(
                requirement,
                _binding(),
                trace,
                validated_program=validated,
                resolved_channels=None,
                verified_wrench_audit=audit,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_OUTPUT_TORQUE_CHANNEL_UNRESOLVED",
        )

    def test_output_torque_rejects_channel_and_component_provenance_attacks(
        self,
    ) -> None:
        trace = _output_torque_trace()
        requirement = _requirement("output_torque_nm")
        channels = resolve_and_validate_task_channels(
            requirement,
            _binding(),
            _registry(),
        )
        audit = _wrench_audit(include_output_torque_inventory=True)
        validated = _validated(trace)
        sample_index = validated.return_sample_indices[0]

        wrong_body = deepcopy(trace)
        wrong_body["samples"][sample_index]["wrench_components"][0][
            "target_body"
        ] = "crank_body"
        wrong_kind = deepcopy(trace)
        wrong_kind["samples"][sample_index]["wrench_components"][0][
            "wrench_kind"
        ] = "force"
        wrong_axis = deepcopy(trace)
        wrong_axis["samples"][sample_index]["wrench_components"][0][
            "axis_abs"
        ] = [1.0, 0.0, 0.0]
        wrong_load_id = deepcopy(trace)
        wrong_load_id["samples"][sample_index]["wrench_components"][0][
            "load_id"
        ] = "unregistered-output-load"
        missing_selected = deepcopy(trace)
        missing_selected["samples"][sample_index]["wrench_components"] = (
            missing_selected["samples"][sample_index]["wrench_components"][1:]
        )
        attacks = (
            ("TASK_OUTPUT_TORQUE_COMPONENT_PROVENANCE_MISMATCH", wrong_body),
            ("TASK_OUTPUT_TORQUE_COMPONENT_PROVENANCE_MISMATCH", wrong_kind),
            ("TASK_OUTPUT_TORQUE_AXIS_MISMATCH", wrong_axis),
            ("TASK_OUTPUT_TORQUE_COMPONENT_INVENTORY_MISMATCH", wrong_load_id),
            ("TASK_OUTPUT_TORQUE_COMPONENT_INVENTORY_MISMATCH", missing_selected),
        )
        for code, attacked_trace in attacks:
            with self.subTest(code=code):
                with self.assertRaises(MetricEvaluationError) as raised:
                    evaluate_registered_metric(
                        requirement,
                        _binding(),
                        attacked_trace,
                        validated_program=validated,
                        resolved_channels=channels,
                        verified_wrench_audit=audit,
                    )
                self.assertEqual(raised.exception.code, code)

        with self.assertRaises(MetricEvaluationError) as raised:
            evaluate_registered_metric(
                requirement,
                _binding(),
                trace,
                validated_program=validated,
                resolved_channels=channels,
                verified_wrench_audit=_wrench_audit(
                    include_output_torque_inventory=True,
                    transmission_target_body="crank_body",
                ),
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_OUTPUT_TORQUE_TRANSMISSION_TARGET_MISMATCH",
        )

    def test_output_torque_rejects_tampered_passive_load_classification(
        self,
    ) -> None:
        trace = _output_torque_trace()
        requirement = _requirement("output_torque_nm")
        channels = resolve_and_validate_task_channels(
            requirement,
            _binding(),
            _registry(),
        )
        audit = _wrench_audit(include_output_torque_inventory=True)
        passive_record = next(
            record
            for record in audit["records"]
            if record["load_id"] == "output-passive-support"
        )
        passive_record["transmission"] = True
        passive_record["support"] = False
        audit["transmission_load_count"] = 2
        audit["support_load_count"] = 0
        audit["ledger_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in audit.items()
                if key != "ledger_sha256"
            }
        )

        with self.assertRaises(MetricEvaluationError) as raised:
            evaluate_registered_metric(
                requirement,
                _binding(),
                trace,
                validated_program=_validated(trace),
                resolved_channels=channels,
                verified_wrench_audit=audit,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_OUTPUT_TORQUE_AUDIT_INVALID",
        )

    def test_powered_ratio_checks_every_sample_and_increment(self) -> None:
        trace = _trace(
            _samples(
                powered_output=(0.5, 5.0, 1.5, 2.0),
                powered_input=(0.25, 0.5, 0.75, 1.0),
            )
        )
        requirement = _requirement("angular_ratio")
        channels = resolve_and_validate_task_channels(
            requirement,
            _binding(reference=True),
            _registry(),
        )
        result = evaluate_registered_metric(
            requirement,
            _binding(reference=True),
            trace,
            validated_program=_validated(trace),
            resolved_channels=channels,
        )
        self.assertEqual(result["measured"], 2.0)
        self.assertFalse(result["passed"])
        self.assertFalse(result["evidence"]["ratio_consistency_passed"])

    def test_phase_coverage_and_boundaries_are_required(self) -> None:
        trace = _trace(_samples())
        trace["samples"][6]["phase_id"] = "release_pedal"
        with self.assertRaises(MechanismValidationError) as raised:
            _validated(trace)
        self.assertEqual(
            raised.exception.code,
            "TASK_TRACE_PHASE_TIME_MISMATCH",
        )

    def test_rehashed_nonadaptive_duration_change_is_rejected(self) -> None:
        trace = _trace(_samples())
        trace["executed_phases"][1]["duration_s"] = 0.25
        trace["executed_load_sha256"] = canonical_sha256(
            trace["executed_phases"]
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validated(trace)
        self.assertEqual(
            raised.exception.code,
            "TASK_EXECUTED_SCHEDULE_DURATION_MISMATCH",
        )

    def test_rehashed_one_step_duration_deletion_is_rejected(self) -> None:
        load_case = _load_case()
        trace = _trace(_samples())
        trace["executed_phases"][1]["duration_s"] = (
            load_case.phases[1].duration_s - load_case.step_s
        )
        trace["executed_load_sha256"] = canonical_sha256(
            trace["executed_phases"]
        )
        with self.assertRaises(MechanismValidationError) as raised:
            _validated(trace)
        self.assertEqual(
            raised.exception.code,
            "TASK_EXECUTED_SCHEDULE_DURATION_MISMATCH",
        )

    def test_registered_minimum_powered_sample_count_is_enforced(self) -> None:
        load_case = replace(_load_case(), sample_interval_s=0.5)
        schedule = _schedule(load_case)
        samples = [
            {
                "time_s": 0.0,
                "phase_id": "initial",
                "compound_output_angle_rad": 0.0,
                "compound_input_angle_rad": 0.0,
            },
            {
                "time_s": 0.5,
                "phase_id": "press_pedal",
                "compound_output_angle_rad": 1.0,
                "compound_input_angle_rad": 0.5,
            },
            {
                "time_s": 1.0,
                "phase_id": "press_pedal",
                "compound_output_angle_rad": 2.0,
                "compound_input_angle_rad": 1.0,
            },
            {
                "time_s": 1.5,
                "phase_id": "open_settle",
                "compound_output_angle_rad": 2.0,
                "compound_input_angle_rad": 1.0,
            },
            {
                "time_s": 2.0,
                "phase_id": "release_pedal",
                "compound_output_angle_rad": 0.2,
                "compound_input_angle_rad": 0.1,
            },
            {
                "time_s": 2.5,
                "phase_id": "closed_settle",
                "compound_output_angle_rad": 0.1,
                "compound_input_angle_rad": 0.0,
            },
        ]
        trace = {
            "status": "COMPLETED",
            "sample_count": len(samples),
            "executed_phases": schedule,
            "executed_load_sha256": canonical_sha256(schedule),
            "samples": samples,
        }
        spec = task_program_spec_for_load_case(
            load_case,
            plugin_id="revolute_linkage_v1",
        )
        with self.assertRaises(MechanismValidationError) as raised:
            validate_trace_program(trace, load_case, program_spec=spec)
        self.assertEqual(
            raised.exception.code,
            "TASK_TRACE_POWERED_SAMPLE_COUNT_INSUFFICIENT",
        )

    def test_sparse_duplicate_and_count_mismatch_are_inconclusive(self) -> None:
        attacks = []
        sparse = _trace(_samples())
        del sparse["samples"][2]
        sparse["sample_count"] = len(sparse["samples"])
        attacks.append(("TASK_TRACE_SAMPLING_SPARSE", sparse))
        duplicate = _trace(_samples())
        duplicate["samples"][2]["time_s"] = duplicate["samples"][1]["time_s"]
        attacks.append(("TASK_TRACE_TIME_NOT_STRICT", duplicate))
        count = _trace(_samples())
        count["sample_count"] += 1
        attacks.append(("TASK_TRACE_SAMPLE_COUNT_MISMATCH", count))
        for code, attack in attacks:
            with self.subTest(code=code):
                with self.assertRaises(MechanismValidationError) as raised:
                    _validated(attack)
                self.assertEqual(raised.exception.code, code)

    def test_unregistered_program_has_no_inferred_certification_window(
        self,
    ) -> None:
        load_case = _load_case()
        raw = load_case.to_dict()
        raw["program"] = "unregistered_rotational_program"
        unknown = LoadCase.from_dict(raw)
        with self.assertRaises(MechanismValidationError) as raised:
            task_program_spec_for_load_case(
                unknown,
                plugin_id="revolute_linkage_v1",
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_PROGRAM_POLICY_UNREGISTERED",
        )

    def test_semantic_runtime_body_swap_is_rejected(self) -> None:
        registry = _registry()
        swapped_input = deepcopy(registry.inputs["input_crank_torque"])
        object.__setattr__(swapped_input, "body_id", "rear_body")
        forged = TaskChannelRegistry(
            plugin_id=registry.plugin_id,
            input_body_ids=registry.input_body_ids,
            output_body_ids=registry.output_body_ids,
            inputs={"input_crank_torque": swapped_input},
            observables=registry.observables,
        )
        with self.assertRaises(MechanismValidationError) as raised:
            resolve_and_validate_task_channels(
                _requirement("continuous_rotation_rad"),
                _binding(),
                forged,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_SEMANTIC_RUNTIME_INPUT_MISMATCH",
        )

    def test_same_body_ratio_alias_is_rejected(self) -> None:
        registry = _registry()
        aliased_target = ResolvedTaskChannel(
            semantic_id="output_wheel_pose",
            rigid_group_id="group_rear_drive_wheel",
            body_id="crank_body",
            frame_id="frame_rear",
            canonical_field="compound_input_angle_rad",
            axis_abs=(0.0, 1.0, 0.0),
            quantity="pose_6d",
        )
        forged = TaskChannelRegistry(
            plugin_id=registry.plugin_id,
            input_body_ids=registry.input_body_ids,
            output_body_ids=("crank_body",),
            inputs=registry.inputs,
            observables={
                **registry.observables,
                "output_wheel_pose": aliased_target,
            },
        )
        requirement = _requirement("angular_ratio")
        requirement["allow_direct_output_actuation"] = True
        with self.assertRaises(MechanismValidationError) as raised:
            resolve_and_validate_task_channels(
                requirement,
                _binding(reference=True),
                forged,
            )
        self.assertEqual(raised.exception.code, "TASK_RATIO_CHANNEL_ALIAS")

    def test_ratio_reference_must_be_exact_selected_input(self) -> None:
        registry = _registry()
        alternate_reference = ResolvedTaskChannel(
            semantic_id="alternate_input_pose",
            rigid_group_id="group_alternate_input",
            body_id="alternate_input_body",
            frame_id="frame_alternate_input",
            canonical_field="alternate_input_angle_rad",
            axis_abs=(0.0, 1.0, 0.0),
            quantity="pose_6d",
        )
        forged = TaskChannelRegistry(
            plugin_id=registry.plugin_id,
            input_body_ids=(
                *registry.input_body_ids,
                "alternate_input_body",
            ),
            output_body_ids=registry.output_body_ids,
            inputs=registry.inputs,
            observables={
                **registry.observables,
                "alternate_input_pose": alternate_reference,
            },
        )
        binding = _binding(reference=True)
        binding["reference"] = {
            "observable_id": "alternate_input_pose",
            "rigid_group_id": "group_alternate_input",
            "frame_id": "frame_alternate_input",
            "quantity": "pose_6d",
        }
        with self.assertRaises(MechanismValidationError) as raised:
            resolve_and_validate_task_channels(
                _requirement("angular_ratio"),
                binding,
                forged,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_RATIO_REFERENCE_INPUT_MISMATCH",
        )

    def test_saved_channel_axis_and_actuation_vector_are_revalidated(
        self,
    ) -> None:
        registry = _registry()
        runtime_layout = (
            SimpleNamespace(
                plugin_id=registry.plugin_id,
                task_input_actuation_spans_all_inputs=False,
            ),
            {
                "group_input_crank": "crank_body",
                "group_rear_drive_wheel": "rear_body",
            },
            {"crank_body"},
            {"rear_body"},
            {
                "crank_body": "compound_input_angle_rad",
                "rear_body": "compound_output_angle_rad",
            },
            {
                "crank_body": (0.0, 1.0, 0.0),
                "rear_body": (0.0, 1.0, 0.0),
            },
        )
        for field, value in (
            ("axis_abs", [1.0, 0.0, 0.0]),
            ("actuation_body_ids", ["rear_body"]),
        ):
            with self.subTest(field=field):
                raw = registry.to_dict()
                raw["inputs"]["input_crank_torque"][field] = value
                payload = {
                    key: item
                    for key, item in raw.items()
                    if key != "channel_registry_sha256"
                }
                raw["channel_registry_sha256"] = canonical_sha256(payload)
                plan = SimpleNamespace(task_channel_registry=raw)
                with patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "mechanism.task_motion_evidence._runtime_channel_layout",
                    return_value=runtime_layout,
                ):
                    with self.assertRaises(
                        MechanismValidationError
                    ) as raised:
                        task_channel_registry_for_plan(plan)
                self.assertEqual(
                    raised.exception.code,
                    "TASK_CHANNEL_REGISTRY_RUNTIME_MISMATCH",
                )

    def test_saved_channel_group_alias_survives_materialized_body_compaction(
        self,
    ) -> None:
        registry = _registry()
        raw = registry.to_dict()
        raw["observables"]["output_wheel_pose"]["rigid_group_id"] = (
            "group_rear_reference_alias"
        )
        raw["inputs"]["input_crank_torque"]["actuation_body_ids"] = [
            "crank_body"
        ]
        payload = {
            key: item
            for key, item in raw.items()
            if key != "channel_registry_sha256"
        }
        raw["channel_registry_sha256"] = canonical_sha256(payload)
        runtime_layout = (
            SimpleNamespace(
                plugin_id=registry.plugin_id,
                task_input_actuation_spans_all_inputs=False,
            ),
            {
                "group_input_crank": "crank_body",
                "group_rear_drive_wheel": "rear_body",
            },
            {"crank_body"},
            {"rear_body"},
            {
                "crank_body": "compound_input_angle_rad",
                "rear_body": "compound_output_angle_rad",
            },
            {
                "crank_body": (0.0, 1.0, 0.0),
                "rear_body": (0.0, 1.0, 0.0),
            },
        )
        plan = SimpleNamespace(task_channel_registry=raw)

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline."
            "mechanism.task_motion_evidence._runtime_channel_layout",
            return_value=runtime_layout,
        ):
            resolved = task_channel_registry_for_plan(plan)

        self.assertEqual(
            resolved.observables["output_wheel_pose"].body_id,
            "rear_body",
        )

    def test_channel_body_frame_and_axis_changes_are_rejected(self) -> None:
        attacks = []
        wrong_frame = _binding()
        wrong_frame["target"]["frame_id"] = "forged_frame"
        attacks.append(wrong_frame)
        wrong_axis = _binding()
        wrong_axis["input"]["direction_abs"] = [1.0, 0.0, 0.0]
        attacks.append(wrong_axis)
        nonfinite_axis = _binding()
        nonfinite_axis["input"]["direction_abs"] = [
            float("nan"),
            1.0,
            0.0,
        ]
        attacks.append(nonfinite_axis)
        wrong_group = _binding()
        wrong_group["target"]["rigid_group_id"] = "group_input_crank"
        attacks.append(wrong_group)
        for attack in attacks:
            with self.subTest(attack=attack):
                with self.assertRaises(MechanismValidationError):
                    resolve_and_validate_task_channels(
                        _requirement("continuous_rotation_rad"),
                        attack,
                        _registry(),
                    )

    def test_unrelated_direct_false_label_is_not_actuation_provenance(
        self,
    ) -> None:
        trace = _trace(_samples())
        trace["branch_semantics"] = {
            "direct_unrelated_actuation": False,
            "direct_output_actuation": False,
        }
        with self.assertRaises(MetricEvaluationError) as raised:
            evaluate_registered_metric(
                {
                    **_requirement("continuous_rotation_rad"),
                    "metric_id": "actuation_provenance",
                    "target": {
                        "comparator": "equals",
                        "value": "indirect_only",
                        "unit": "enum",
                    },
                },
                _binding(),
                trace,
            )
        self.assertEqual(
            raised.exception.code,
            "TASK_ACTUATION_EVIDENCE_MISSING",
        )

    def test_complete_public_motion_evidence_path_passes(self) -> None:
        requirements = []
        bindings = []
        for metric_id in (
            "continuous_rotation_rad",
            "angular_ratio",
            "return_to_start_rad",
            "actuation_provenance",
        ):
            requirement = _requirement(metric_id)
            if metric_id == "actuation_provenance":
                requirement["target"] = {
                    "comparator": "equals",
                    "value": "indirect_only",
                    "unit": "enum",
                }
            requirement.update(
                {
                    "test_program": "task_output_angle_roundtrip",
                    "test_load_case_id": "task_output_angle_roundtrip",
                    "required_branches": ["physical"],
                }
            )
            requirements.append(requirement)
            bindings.append(
                {
                    "requirement_id": requirement["requirement_id"],
                    **_binding(reference=metric_id == "angular_ratio"),
                }
            )
        plan = SimpleNamespace(
            observable_id="output_wheel_pose",
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "contract_sha256": "a" * 64,
                "task_contract_sha256": "b" * 64,
                "requirements": requirements,
            },
            semantic_binding={
                "status": "PASS",
                "certificate_authority": True,
                "task_binding_sha256": "c" * 64,
                "bindings": bindings,
            },
        )
        physical = _trace(_samples())
        physical["wrench_audit"] = _wrench_audit()
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "acceptance_metrics.task_channel_registry_for_plan",
            return_value=_registry(),
        ):
            result = evaluate_task_acceptance(
                plan,
                {
                    "task_output_angle_roundtrip": {
                        "program": "task_output_angle_roundtrip",
                        "load_case": _load_case().to_dict(),
                        "physical": physical,
                        "physical_trace_audit": {"passed": True},
                        "comparison": {"verdict": "PASS"},
                    }
                },
            )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["certificate_authority"])
        self.assertEqual(result["evaluation_count"], 4)

    def test_acceptance_resolves_shared_test_load_by_program(self) -> None:
        profile_id = "rated_resistive_torque_3nm"
        profile = binding_for_test_load_profile(profile_id)
        roundtrip = replace(
            _load_case(),
            load_case_id="rated_roundtrip",
            test_load_case_id=profile_id,
            load_profile_binding=profile,
        )
        multi_cycle = replace(
            roundtrip,
            load_case_id="rated_multi_cycle",
            program="task_output_angle_multi_cycle",
        )
        requirements = []
        bindings = []
        for requirement_id, load_case in (
            ("rated_roundtrip", roundtrip),
            ("rated_multi_cycle", multi_cycle),
        ):
            requirement = _requirement("continuous_rotation_rad")
            requirement.update(
                {
                    "requirement_id": requirement_id,
                    "test_program": load_case.program,
                    "test_load_case_id": profile_id,
                    "required_branches": ["physical"],
                }
            )
            requirements.append(requirement)
            bindings.append(
                {
                    "requirement_id": requirement_id,
                    **_binding(),
                }
            )
        plan = SimpleNamespace(
            observable_id="output_wheel_pose",
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "requirements": requirements,
            },
            semantic_binding={
                "status": "PASS",
                "certificate_authority": True,
                "bindings": bindings,
            },
        )
        traces = {
            load_case.load_case_id: {
                "program": load_case.program,
                "load_case": load_case.to_dict(),
            }
            for load_case in (roundtrip, multi_cycle)
        }

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "acceptance_metrics.task_channel_registry_for_plan",
            return_value=_registry(),
        ):
            result = evaluate_task_acceptance(plan, traces)

        self.assertNotIn("program_mismatch", "\n".join(result["failure_reasons"]))
        self.assertEqual(
            result["failure_reasons"],
            [
                f"rated_roundtrip:{profile_id}:PHYSICAL_TRACE_EVIDENCE_MISSING",
                f"rated_multi_cycle:{profile_id}:PHYSICAL_TRACE_EVIDENCE_MISSING",
            ],
        )

    def test_failed_physical_audit_cannot_yield_task_pass(self) -> None:
        requirement = {
            **_requirement("continuous_rotation_rad"),
            "test_program": "task_output_angle_roundtrip",
            "test_load_case_id": "task_output_angle_roundtrip",
            "required_branches": ["physical"],
        }
        plan = SimpleNamespace(
            observable_id="output_wheel_pose",
            acceptance_contract={
                "status": "PASS",
                "certificate_authority": True,
                "contract_sha256": "a" * 64,
                "task_contract_sha256": "b" * 64,
                "requirements": [requirement],
            },
            semantic_binding={
                "status": "PASS",
                "certificate_authority": True,
                "task_binding_sha256": "c" * 64,
                "bindings": [
                    {
                        "requirement_id": requirement["requirement_id"],
                        **_binding(),
                    }
                ],
            },
        )
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "acceptance_metrics.task_channel_registry_for_plan",
            return_value=_registry(),
        ):
            result = evaluate_task_acceptance(
                plan,
                {
                    "task_output_angle_roundtrip": {
                        "program": "task_output_angle_roundtrip",
                        "load_case": _load_case().to_dict(),
                        "physical": _trace(_samples()),
                        "physical_trace_audit": {
                            "passed": False,
                            "failure_reasons": ["forged_physics"],
                        },
                        "comparison": {"verdict": "PASS"},
                    }
                },
            )
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertFalse(result["certificate_authority"])
        self.assertTrue(
            any(
                "TASK_PHYSICAL_GATE_NOT_PASSING" in item
                for item in result["failure_reasons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
