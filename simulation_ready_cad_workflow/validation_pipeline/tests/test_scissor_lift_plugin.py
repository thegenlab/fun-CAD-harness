"""Tests for the ``hydraulic_scissor_lift`` mechanism family plugin.

Mirrors ``test_clock_gear_plugin.py``: the fixture binds the captured
scissor-lift family declaration onto the derived
compound adapter, then exercises plugin resolution, load-case generation,
spec parsing, synthetic trace comparison, certificate/replacement routing,
and whole-machine plan validation.  Physics itself never runs here — the
PyChrono worker is covered by the harness physics environment.
"""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    adapter_from_dict,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.certificate import (
    build_certificate,
    build_replacement,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
    compare_trace_pair,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    mechanism_adapter_for_plan,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.finite_linear_capture import (
    build_finite_linear_capture_report,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import (
    default_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadPhase,
    MechanismPlan,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_comparison import (
    EXPECTED_REDUCED_IDEAL_LINKS,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_runtime import (
    _support_alignment_axis_abs,
    _unsigned_axis_misalignment_rad,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_spec import (
    expected_ideal_links,
    scissor_lift_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_whole_machine import (
    WholeMachineScissorLiftExecutionPlan,
    build_scissor_lift_execution_plan,
)


FAMILY_JSON = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "scissor_lift_drive.family.json"
)


def _family_spec() -> dict:
    return json.loads(FAMILY_JSON.read_text(encoding="utf-8"))


def _adapter(family_spec: dict | None = None):
    return replace(
        get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
        family_spec=_family_spec() if family_spec is None else family_spec,
    )


def _plan(family_spec: dict | None = None) -> MechanismPlan:
    adapter = _adapter(family_spec)
    return MechanismPlan(
        package_dir="package",
        intake_report="report.json",
        intake_report_sha256="0" * 64,
        package_id="scissor_lift_v1",
        design_version="1.0.0",
        requirements_sha256="1" * 64,
        package_binding_sha256="2" * 64,
        capsule_id="scissor_lift_drive",
        capsule_file="scissor_lift_drive.json",
        family="compound",
        protocol_id="hydraulic_scissor_lift",
        fixed_group_ids=("group_base_frame",),
        moving_group_ids=(
            "group_outer_scissor_frame",
            "group_inner_scissor_frame",
            "group_lift_platform",
            "group_cylinder_barrel",
            "group_cylinder_rod",
        ),
        axis_abs=(0.0, 0.0, 1.0),
        joint_origin_abs_m=(0.0, 0.0, 0.0),
        gravity_abs_m_s2=(0.0, 0.0, -9.81),
        joint_frame_id="frame_outer_base",
        input_port_id="cylinder_force",
        input_frame_id="frame_cylinder",
        input_point_abs_m=(0.36, 0.0, 0.045),
        observable_id="platform_height",
        observable_frame_id="frame_platform",
        observable_point_abs_m=(0.21, 0.0, 0.387303),
        task_ids=("raise_platform", "lower_platform"),
        characteristic_length_m=0.6,
        moving_extents_m=(0.6, 0.3, 0.4),
        adapter_id=adapter.adapter_id,
        adapter_spec=adapter_to_dict(adapter),
    )


def _synthetic_traces(plan: MechanismPlan, load_case, *, physical_ideal_links: int = 0):
    spec = scissor_lift_from_dict(adapter_from_dict(plan.adapter_spec).family_spec)
    joint_ids = list(spec.joint_ids)
    home_length_m = math.dist(
        spec.actuator.barrel_anchor_abs_m, spec.actuator.rod_anchor_abs_m
    )
    common_semantics = {
        "same_mass_kg": 73.2599,
        "same_inertia_xx_kg_m2": {},
        "same_initial_pose": {},
        "same_external_load_sha256": "a" * 64,
        "family_spec_sha256": canonical_sha256(spec.to_dict()),
        "actuator_model": {
            "kind": "hydraulic_cylinder_force_pair",
            "nominal_input_force_n": spec.nominal_input_force_n,
            "return_input_force_n": spec.return_input_force_n,
            "free_axial_rotation": True,
            "length_limits_m": [
                spec.actuator.min_length_m,
                spec.actuator.max_length_m,
            ],
            "source": "validator_hydraulic_cylinder_force_pair_v1",
        },
    }

    def trace(branch: str) -> dict:
        physical = branch == "physical"

        def sample(time_s: float, phase_id: str) -> dict:
            return {
                "time_s": time_s,
                "phase_id": phase_id,
                "platform_height_m": 0.387303,
                "platform_speed_m_s": 0.0,
                "platform_progress_m": 0.0,
                "cylinder_length_m": home_length_m,
                "finite_capture_overlaps_m": {
                    item.capture_id: item.minimum_overlap_m + 0.01
                    for item in spec.finite_linear_captures
                },
                "minimum_finite_capture_overlap_m": min(
                    item.minimum_overlap_m + 0.01
                    for item in spec.finite_linear_captures
                ),
                "joint_drifts_m": {item: 0.0 for item in joint_ids},
                "max_joint_drift_m": 0.0,
                "bushing_reaction_forces_n": {
                    item: (1.0 if physical else 0.0) for item in joint_ids
                },
                "max_bushing_reaction_force_n": 1.0 if physical else 0.0,
                "off_plane_rotation_rad": 0.0,
                "platform_level_error_rad": 0.0,
                "contact_count": 0,
            }

        expected_links = list(expected_ideal_links(spec))
        return {
            "status": "COMPLETED",
            "branch": branch,
            "package_binding_sha256": plan.package_binding_sha256,
            "capsule_id": plan.capsule_id,
            "load_case_id": load_case.load_case_id,
            "mechanism_plugin_id": "hydraulic_scissor_lift_v1",
            "engine": {
                "chrono_link_count": physical_ideal_links if physical else len(expected_links),
                "ideal_links": [] if physical else expected_links,
                "internal_ideal_joint_count": (
                    physical_ideal_links if physical else len(expected_links)
                ),
            },
            "collision_geometry": {
                "contact_enabled": physical,
                "whole_body_box_proxy_used": False,
                "whole_body_convex_hull_used": False,
                "convex_hull_used": False,
                "joint_count": spec.joint_count,
                "finite_linear_capture": {
                    "status": "PASS",
                    "passed": True,
                },
            },
            "branch_semantics": common_semantics,
            "samples": [
                sample(0.0, "initial"),
                sample(0.35, load_case.phases[0].phase_id),
            ],
        }

    return trace("physical"), trace("reduced")


class ScissorLiftPluginTests(unittest.TestCase):
    def test_roller_track_capture_uses_hinge_axis_not_rail_direction(
        self,
    ) -> None:
        rail_axis = (1.0, 0.0, 0.0)
        hinge_axis = (0.0, -1.0, 0.0)

        self.assertEqual(
            _support_alignment_axis_abs(
                "prismatic",
                rail_axis,
                hinge_axis,
            ),
            hinge_axis,
        )
        self.assertEqual(
            _support_alignment_axis_abs(
                "revolute",
                rail_axis,
                hinge_axis,
            ),
            rail_axis,
        )
        self.assertEqual(
            _support_alignment_axis_abs(
                "cylindrical",
                rail_axis,
                hinge_axis,
            ),
            rail_axis,
        )

    def test_support_alignment_rejects_true_hinge_axis_skew(self) -> None:
        five_degrees = math.radians(5.0)
        tilted_hinge_axis = (
            math.sin(five_degrees),
            -math.cos(five_degrees),
            0.0,
        )

        self.assertAlmostEqual(
            _unsigned_axis_misalignment_rad(
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            0.0,
        )
        self.assertAlmostEqual(
            _unsigned_axis_misalignment_rad(
                (0.0, -1.0, 0.0),
                tilted_hinge_axis,
            ),
            five_degrees,
        )

    def test_finite_capture_is_bound_to_step_length_and_rejects_overtravel(
        self,
    ) -> None:
        declaration = scissor_lift_from_dict(
            _family_spec()
        ).finite_linear_captures[0]
        axis = declaration.axis_abs

        def point(
            origin: tuple[float, float, float],
            distance: float,
        ) -> tuple[float, float, float]:
            return tuple(
                origin[index] + axis[index] * distance
                for index in range(3)
            )

        short = build_finite_linear_capture_report(
            (declaration,),
            raw_meshes={
                53: SimpleNamespace(
                    vertices=[
                        declaration.parent_anchor_abs_m,
                        point(declaration.parent_anchor_abs_m, 0.080),
                    ]
                ),
                54: SimpleNamespace(
                    vertices=[point(declaration.parent_anchor_abs_m, 0.080)]
                ),
                55: SimpleNamespace(
                    vertices=[point(declaration.parent_anchor_abs_m, 0.080)]
                ),
                56: SimpleNamespace(
                    vertices=[point(declaration.parent_anchor_abs_m, 0.080)]
                ),
                61: SimpleNamespace(
                    vertices=[
                        declaration.child_anchor_abs_m,
                        point(declaration.child_anchor_abs_m, -0.082),
                    ]
                ),
            },
            body_solid_indices={
                "cylinder_barrel": (53, 54, 55, 56),
                "cylinder_rod": (61,),
            },
            package_binding_sha256="a" * 64,
            mesh_tolerance_m=0.0002,
        )
        self.assertFalse(short["passed"])
        self.assertIn(
            "capture_declared_travel_exceeds_step_geometry",
            short["captures"][0]["failure_reasons"],
        )

        long = build_finite_linear_capture_report(
            (declaration,),
            raw_meshes={
                53: SimpleNamespace(
                    vertices=[
                        declaration.parent_anchor_abs_m,
                        point(declaration.parent_anchor_abs_m, 0.123),
                    ]
                ),
                54: SimpleNamespace(
                    vertices=[point(declaration.parent_anchor_abs_m, 0.123)]
                ),
                55: SimpleNamespace(
                    vertices=[point(declaration.parent_anchor_abs_m, 0.123)]
                ),
                56: SimpleNamespace(
                    vertices=[point(declaration.parent_anchor_abs_m, 0.123)]
                ),
                61: SimpleNamespace(
                    vertices=[
                        declaration.child_anchor_abs_m,
                        point(declaration.child_anchor_abs_m, -0.126),
                    ]
                ),
            },
            body_solid_indices={
                "cylinder_barrel": (53, 54, 55, 56),
                "cylinder_rod": (61,),
            },
            package_binding_sha256="a" * 64,
            mesh_tolerance_m=0.0002,
        )
        self.assertTrue(long["passed"])
        self.assertGreater(
            long["captures"][0]["maximum_supported_separation_m"],
            declaration.maximum_separation_m,
        )

    def test_target_policy_is_explicit_and_fixed_phases_still_replay(self) -> None:
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.scissor_lift_runtime import (
            _branch_owns_stop_time,
            _external_load_schedule_sha256,
            _recorded_phase_duration,
        )

        reference = [{"duration_s": 0.42}]
        strict_target = LoadPhase(
            "strict_target",
            "force",
            (1.0, 0.0, 0.0),
            100.0,
            1.0,
            stop_condition="progress_at_least",
            target_progress_m=0.1,
        )
        independent_target = LoadPhase(
            "independent_target",
            "force",
            (1.0, 0.0, 0.0),
            100.0,
            1.0,
            stop_condition="progress_at_least",
            target_progress_m=0.1,
            branch_termination_policy="each_branch_target",
        )
        self.assertFalse(_branch_owns_stop_time(reference, strict_target))
        self.assertTrue(_branch_owns_stop_time(None, strict_target))
        self.assertTrue(_branch_owns_stop_time(reference, independent_target))
        self.assertEqual(
            _recorded_phase_duration(
                reference,
                phase=strict_target,
                phase_limit_s=0.42,
                elapsed_s=0.39,
            ),
            0.42,
        )
        self.assertEqual(
            _recorded_phase_duration(
                reference,
                phase=independent_target,
                phase_limit_s=1.0,
                elapsed_s=0.39,
            ),
            0.39,
        )
        self.assertNotIn(
            "branch_termination_policy",
            strict_target.to_dict(),
        )
        self.assertEqual(
            LoadPhase.from_dict(independent_target.to_dict()),
            independent_target,
        )
        with self.assertRaises(MechanismValidationError):
            LoadPhase(
                "invalid_policy",
                "force",
                (1.0, 0.0, 0.0),
                100.0,
                1.0,
                branch_termination_policy="each_branch_target",
            )
        phases = [
            {
                "phase_id": "extend",
                "wrench_kind": "force",
                "direction_abs": [1.0, 0.0, 0.0],
                "magnitude_si": 100.0,
                "duration_s": 0.42,
            }
        ]
        changed = [dict(phases[0], duration_s=0.39)]
        self.assertNotEqual(
            _external_load_schedule_sha256(phases),
            _external_load_schedule_sha256(changed),
        )

    def test_scissor_lift_plugin_owns_topology_and_six_load_cases(self) -> None:
        plan = _plan()
        plugin, adapter = mechanism_adapter_for_plan(plan)
        cases = default_load_cases(plan)

        self.assertEqual(plugin.plugin_id, "hydraulic_scissor_lift_v1")
        self.assertEqual(plugin.worker_backend, "hydraulic_scissor_lift_v1")
        self.assertEqual(plugin.whole_machine_backend, "hydraulic_scissor_lift_v1")
        self.assertEqual(
            tuple(plugin.expected_reduced_ideal_links(plan, adapter)),
            EXPECTED_REDUCED_IDEAL_LINKS,
        )
        self.assertEqual(len(cases), 6)
        self.assertEqual(
            [case.load_case_id for case in cases],
            [
                "nominal_light",
                "nominal_loaded",
                "zero_input_gravity",
                "transverse_scissor_probe",
                "held_out_mid",
                "task_platform_roundtrip",
            ],
        )
        self.assertEqual(
            [case.program for case in cases],
            [
                "bidirectional_scissor_lift",
                "bidirectional_scissor_lift",
                "zero_input_gravity",
                "transverse_scissor_probe",
                "bidirectional_scissor_lift",
                "task_platform_roundtrip",
            ],
        )
        self.assertTrue(
            {case.program for case in cases}.issubset(set(adapter.supported_programs))
        )
        task_case = cases[-1]
        self.assertEqual(task_case.phases[0].stop_condition, "position_at_least")
        self.assertAlmostEqual(task_case.phases[0].target_position_si, 0.194119)
        self.assertEqual(
            task_case.phases[0].branch_termination_policy,
            "each_branch_target",
        )
        self.assertEqual(task_case.phases[2].stop_condition, "position_at_most")
        self.assertEqual(
            task_case.phases[2].branch_termination_policy,
            "each_branch_target",
        )
        self.assertIsNone(task_case.phases[1].branch_termination_policy)
        self.assertAlmostEqual(task_case.phases[2].target_position_si, 0.003)
        self.assertAlmostEqual(cases[0].phases[0].magnitude_si, 0.82 * 2400.0)
        self.assertAlmostEqual(cases[1].phases[0].magnitude_si, 1.18 * 2400.0)
        # The transverse probe pushes along the scissor-plane normal (+Y for the
        # canonical capsule) so it excites the stiff bearing-seat directions.
        probe = cases[3]
        self.assertEqual(probe.phases[0].phase_id, "transverse_force")
        for actual, expected in zip(probe.phases[0].direction_abs, (0.0, 1.0, 0.0)):
            self.assertAlmostEqual(actual, expected)

    def test_family_spec_parses_real_capsule_json(self) -> None:
        spec = scissor_lift_from_dict(_family_spec())

        self.assertEqual(spec.mechanism_kind, "hydraulic_scissor_lift")
        self.assertEqual(len(spec.bodies), 6)
        self.assertEqual(len(spec.revolute_joints), 5)
        self.assertEqual(len(spec.prismatic_joints), 2)
        self.assertEqual(spec.cylindrical_joint.joint_id, "cylinder_barrel_rod")
        self.assertEqual(spec.joint_count, 8)
        self.assertEqual(spec.loop_count, 3)
        self.assertEqual(spec.fixed_body_id, "base_frame")
        self.assertEqual(spec.output_body_id, "lift_platform")
        self.assertEqual(expected_ideal_links(spec), EXPECTED_REDUCED_IDEAL_LINKS)
        self.assertEqual(scissor_lift_from_dict(spec.to_dict()), spec)

    def test_wrong_mechanism_kind_rejected_and_falls_back(self) -> None:
        family_spec = _family_spec()
        family_spec["mechanism_kind"] = "revolute_linkage"

        with self.assertRaisesRegex(
            MechanismValidationError, "SCISSOR_LIFT_KIND_INVALID"
        ):
            scissor_lift_from_dict(family_spec)
        plugin, _adapter = mechanism_adapter_for_plan(_plan(family_spec))
        self.assertEqual(plugin.plugin_id, "revolute_linkage_v1")

    def test_comparison_passes_on_synthetic_zero_input_traces(self) -> None:
        plan = _plan()
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "zero_input_gravity"
        )
        physical, reduced = _synthetic_traces(plan, load_case)

        comparison = compare_trace_pair(plan, load_case, physical, reduced)

        self.assertEqual(comparison["verdict"], "PASS")
        self.assertEqual(comparison["failure_reasons"], [])
        self.assertEqual(
            comparison["thresholds"]["source"], "family_spec_hydraulic_scissor_lift_v1"
        )

    def test_physical_branch_with_ideal_links_is_inconclusive(self) -> None:
        plan = _plan()
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "zero_input_gravity"
        )
        physical, reduced = _synthetic_traces(plan, load_case, physical_ideal_links=1)

        comparison = compare_trace_pair(plan, load_case, physical, reduced)

        self.assertEqual(comparison["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "physical_scissor_lift_contains_ideal_links",
            comparison["failure_reasons"],
        )

    def test_missing_samples_are_inconclusive(self) -> None:
        plan = _plan()
        load_case = next(
            item for item in default_load_cases(plan)
            if item.program == "zero_input_gravity"
        )
        physical, reduced = _synthetic_traces(plan, load_case)
        physical["samples"] = []

        comparison = compare_trace_pair(plan, load_case, physical, reduced)

        self.assertEqual(comparison["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn("TRACE_SAMPLES_MISSING", comparison["failure_reasons"])

    def test_certificate_and_replacement_carry_family_identity(self) -> None:
        plan = _plan()
        spec = scissor_lift_from_dict(adapter_from_dict(plan.adapter_spec).family_spec)
        cases = default_load_cases(plan)
        comparisons = tuple(
            {
                "load_case_id": case.load_case_id,
                "verdict": "PASS",
                "required_for_certificate": True,
                "failure_reasons": [],
                "thresholds": {"source": "family_spec_hydraulic_scissor_lift_v1"},
                "physical_metrics": {
                    "min_progress_m": 0.0,
                    "max_progress_m": 0.195,
                    "max_joint_drift_m": 1e-4,
                    "max_off_plane_rotation_rad": 1e-3,
                    "max_platform_level_error_rad": 1e-3,
                },
            }
            for case in cases
        )

        certificate = build_certificate(plan, cases, comparisons)

        self.assertEqual(certificate["verdict"], "KINEMATIC_CERTIFIED")
        model = certificate["compound_mechanism_model"]
        self.assertEqual(model["mechanism_kind"], "hydraulic_scissor_lift")
        self.assertEqual(
            model["family_spec_sha256"], canonical_sha256(spec.to_dict())
        )
        self.assertEqual(model["body_count"], 6)
        self.assertEqual(model["joint_count"], 8)
        self.assertEqual(
            model["reduced_support"],
            "5xChLinkLockRevolute_2xChLinkLockRevolutePrismatic_1xChLinkLockCylindrical",
        )
        envelope = certificate["tested_envelope"]
        self.assertAlmostEqual(envelope["applied_force_n"]["max"], 2832.0)
        self.assertAlmostEqual(envelope["applied_force_n"]["min_nonzero"], 480.0)
        self.assertAlmostEqual(envelope["moving_mass_kg"]["min"], 73.2599)
        self.assertEqual(
            envelope["generalized_platform_progress_m"]["max"], 0.195
        )
        reference_only = build_certificate(
            plan,
            cases,
            comparisons,
            replacement_authorized=False,
            replacement_policy="reference_only_single_mechanism",
        )
        self.assertFalse(reference_only["replacement_eligible"])
        self.assertEqual(
            reference_only["replacement_policy"],
            "reference_only_single_mechanism",
        )

        replacement = build_replacement(
            plan, certificate, certificate_sha256="f" * 64
        )
        topology = replacement["replacement_model"]["compound_topology"]
        self.assertEqual(topology["mechanism_kind"], "hydraulic_scissor_lift")
        self.assertEqual(
            topology["family_spec_sha256"], canonical_sha256(spec.to_dict())
        )
        self.assertEqual(topology["joint_ids"], list(spec.joint_ids))
        self.assertEqual(len(topology["joint_ids"]), 8)
        self.assertEqual(topology["ideal_links"], list(EXPECTED_REDUCED_IDEAL_LINKS))

    def _whole_machine_kwargs(self, certificate: dict, replacement_model: dict):
        plan = _plan()
        adapter = adapter_from_dict(plan.adapter_spec)
        plugin, _ = mechanism_adapter_for_plan(plan)
        return {
            "plan": plan,
            "workspace": Path("."),
            "output_dir": Path("results/scissor_lift_drive"),
            "certificate_path": Path("certificate.json"),
            "certificate_sha256": "c" * 64,
            "replacement_path": Path("replacement.json"),
            "replacement_sha256": "r" * 64,
            "certificate": certificate,
            "replacement_model": replacement_model,
            "adapter": adapter,
            "plugin": plugin,
        }

    def test_whole_machine_plan_rejects_mismatched_replacement_topology(self) -> None:
        kwargs = self._whole_machine_kwargs(
            certificate={"tested_envelope": {}},
            replacement_model={"compound_topology": {"mechanism_kind": "rack_pinion"}},
        )

        with self.assertRaisesRegex(
            MechanismValidationError, "WHOLE_MACHINE_REPLACEMENT_MODEL_MISMATCH"
        ):
            build_scissor_lift_execution_plan(**kwargs)

    def test_whole_machine_plan_rejects_tasks_outside_certified_envelope(self) -> None:
        plan = _plan()
        spec = scissor_lift_from_dict(adapter_from_dict(plan.adapter_spec).family_spec)
        kwargs = self._whole_machine_kwargs(
            certificate={
                "tested_envelope": {
                    "moving_mass_kg": {"min": 0.0, "max": 1000.0},
                    "applied_force_n": {"min_nonzero": 0.0, "max": 2000.0},
                    "generalized_platform_progress_m": {"min": 0.0, "max": 0.10},
                }
            },
            replacement_model={
                "compound_topology": {
                    "mechanism_kind": "hydraulic_scissor_lift",
                    "family_spec_sha256": canonical_sha256(spec.to_dict()),
                }
            },
        )

        with self.assertRaisesRegex(
            MechanismValidationError, "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE"
        ):
            build_scissor_lift_execution_plan(**kwargs)

    def test_whole_machine_plan_dataclass_validates_backend_and_round_trips(self) -> None:
        plan = _plan()
        adapter = adapter_from_dict(plan.adapter_spec)
        task_case = next(
            item for item in default_load_cases(plan)
            if item.program == "task_platform_roundtrip"
        )
        execution_plan = WholeMachineScissorLiftExecutionPlan(
            schema_version="0.1",
            coordinate_kind="scissor_lift_platform_progress",
            package_binding_sha256=plan.package_binding_sha256,
            capsule_id=plan.capsule_id,
            certificate_file="certificate.json",
            certificate_sha256="c" * 64,
            replacement_file="replacement.json",
            replacement_sha256="r" * 64,
            adapter_id=adapter.adapter_id,
            mechanism_plugin_id="hydraulic_scissor_lift_v1",
            worker_backend="hydraulic_scissor_lift_v1",
            task_ids=("raise_platform", "lower_platform"),
            mechanism_plan=plan.to_dict(),
            adapter_spec=plan.adapter_spec,
            task_load_case=task_case.to_dict(),
            reference_trace_file="trace.json",
            reference_trace_sha256="t" * 64,
            output_trace_file="results/scissor_lift_drive/whole_machine/whole_machine.trace.json",
            outbound_progress_m=0.195719,
            home_progress_m=0.0,
            max_control_force_n=2400.0,
            progress_tolerance_m=0.004,
        )
        execution_plan.validate()

        restored = WholeMachineScissorLiftExecutionPlan.from_dict(
            execution_plan.to_dict()
        )
        self.assertEqual(restored.coordinate_kind, "scissor_lift_platform_progress")
        self.assertEqual(restored.task_ids, ("raise_platform", "lower_platform"))

        payload = execution_plan.to_dict()
        payload["worker_backend"] = "rack_pinion_v1"
        with self.assertRaisesRegex(
            MechanismValidationError, "WHOLE_MACHINE_PLAN_VALUE_INVALID"
        ):
            WholeMachineScissorLiftExecutionPlan.from_dict(payload)

        payload = execution_plan.to_dict()
        payload["coordinate_kind"] = "rack_pinion_progress"
        with self.assertRaisesRegex(
            MechanismValidationError, "WHOLE_MACHINE_PLAN_SCHEMA_INVALID"
        ):
            WholeMachineScissorLiftExecutionPlan.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
