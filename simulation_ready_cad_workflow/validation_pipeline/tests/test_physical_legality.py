from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import trimesh

from simulation_ready_cad_workflow.validation_pipeline.mechanism.collision_policy import (
    ANALYTICAL_TRANSMISSION,
    CONTACT_PHYSICAL,
    collision_policy_disclosure,
    interface_declaration,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.conditional_interface import (
    build_functional_interface_geometry_binding,
    conditional_transmission_gate,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.contact_intent import (
    EXPECTED_REAL_CONTACT,
    build_component_contact_intent_policy,
    contact_intent_declaration,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    canonical_sha256,
    sha256_file,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
    trace_collision_thresholds,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
    _apply_executed_load_schedule_gate,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_legality import (
    apply_saved_physical_legality_gate,
    audit_dispatch_home_interpenetration,
    audit_family_collision_geometry,
    audit_prescribed_swept_envelope,
    audit_physical_trace_evidence,
    normalized_physical_audit_inputs,
    disclosed_broad_collision_exclusions,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class PhysicalLegalityTests(unittest.TestCase):
    def test_trace_collision_thresholds_preserve_runtime_budget(self) -> None:
        result = trace_collision_thresholds(
            {
                "mesh_linear_deflection_m": 200.0e-6,
                "collision_envelope_m": 12.0e-6,
                "collision_margin_m": 2.0e-6,
                "interpenetration_tolerance_m": 350.0e-6,
                "runtime_interpenetration_tolerance_m": 50.0e-6,
            }
        )

        self.assertEqual(result["mesh_linear_deflection_m"], 200.0e-6)
        self.assertEqual(result["collision_envelope_m"], 12.0e-6)
        self.assertEqual(
            result["runtime_interpenetration_tolerance_m"],
            50.0e-6,
        )

    def test_trace_collision_thresholds_fail_closed_when_budget_missing(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            MechanismValidationError,
            "runtime_interpenetration_tolerance_m",
        ):
            trace_collision_thresholds(
                {
                    "mesh_linear_deflection_m": 200.0e-6,
                    "collision_envelope_m": 12.0e-6,
                    "collision_margin_m": 2.0e-6,
                    "interpenetration_tolerance_m": 350.0e-6,
                }
            )

    def test_dispatch_home_audit_exempts_only_bound_flexible_pair(
        self,
    ) -> None:
        geometry = {
            "collision_envelope_m": 12.0e-6,
            "runtime_interpenetration_tolerance_m": 50.0e-6,
            "home_interpenetration_audit": {
                "method": "unit_test",
                "contact_events": [
                    {
                        "solid_pair": [
                            "gear_a/solid_tooth",
                            "gear_b/solid_tooth",
                        ],
                        "penetration_depth_m": 0.0002,
                    },
                    {
                        "solid_pair": [
                            "frame/solid_marker",
                            "gear_b/solid_hand",
                        ],
                        "penetration_depth_m": 0.0002,
                    },
                ],
            },
        }
        initial = {
            "interfaces": [
                {
                    "classification": "ANALYTICAL_TRANSMISSION",
                    "endpoint_a": {
                        "component_refs": ["gear_a/solid_tooth"]
                    },
                    "endpoint_b": {
                        "component_refs": ["gear_b/solid_tooth"]
                    },
                }
            ]
        }

        result = audit_dispatch_home_interpenetration(
            geometry,
            initial,
        )

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["flexible_pair_event_exemption_count"],
            1,
        )
        self.assertEqual(result["illegal_event_count"], 1)
        self.assertEqual(
            result["worst_events"][0]["solid_pair"],
            ["frame/solid_marker", "gear_b/solid_hand"],
        )

    def _fixture(self, root: Path) -> dict:
        bodies = []
        for body_id, fixed in (("fixed", True), ("moving", False)):
            path = root / f"{body_id}.obj"
            path.write_text(
                trimesh.exchange.obj.export_obj(trimesh.creation.box()),
                encoding="ascii",
            )
            bodies.append(
                {
                    "body_id": body_id,
                    "fixed": fixed,
                    "components": [
                        {
                            "solid_id": f"{body_id}_solid",
                            "mesh": path.relative_to(root).as_posix(),
                            "sha256": sha256_file(path),
                        }
                    ],
                }
            )
        return {
            "schema": "step_per_solid_body_local_v1",
            "bodies": bodies,
            "interpenetration_tolerance_m": 0.00035,
        }

    @staticmethod
    def _sample(moving_x: float) -> dict:
        return {
            "time_s": 0.0,
            "phase_id": "initial",
            "body_poses": {
                "fixed": {
                    "position_abs_m": [0.0, 0.0, 0.0],
                    "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "moving": {
                    "position_abs_m": [moving_x, 0.0, 0.0],
                    "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            },
        }

    def test_separated_and_touching_bodies_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = self._fixture(root)
            separated = audit_family_collision_geometry(
                geometry,
                [self._sample(2.0)],
                workspace_root=root,
            )
            touching = audit_family_collision_geometry(
                geometry,
                [self._sample(1.0)],
                workspace_root=root,
            )

        self.assertTrue(separated["passed"])
        self.assertTrue(touching["passed"])
        self.assertEqual(separated["tested_body_pair_count"], 1)
        self.assertLessEqual(touching["max_penetration_m"], touching["tolerance_m"])

    def test_penetrating_body_fails_with_depth(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audit = audit_family_collision_geometry(
                self._fixture(root),
                [self._sample(0.8)],
                workspace_root=root,
            )

        self.assertFalse(audit["passed"])
        self.assertGreater(audit["max_penetration_m"], 0.19)
        self.assertEqual(audit["illegal_body_pair_count"], 1)
        self.assertEqual(audit["worst_events"][0]["body_pair"], ["fixed", "moving"])
        self.assertEqual(audit["worst_events"][0]["audit_sample_index"], 0)
        self.assertEqual(audit["worst_events"][0]["source_sample_index"], 0)
        self.assertEqual(len(audit["worst_events"][0]["contact_position_abs_m"]), 3)
        self.assertEqual(len(audit["worst_events"][0]["contact_normal_abs"]), 3)
        self.assertIn(
            audit["worst_events"][0]["contact_position_source"],
            {"fcl_contact", "world_aabb_overlap_center"},
        )

    def _expected_contact_policy(
        self,
        geometry: dict,
        *,
        max_penetration_m: float | None,
        phases: tuple[str, ...] = ("initial",),
    ) -> dict:
        declaration = contact_intent_declaration(
            intent_id="expected-fixture-contact",
            kind=EXPECTED_REAL_CONTACT,
            component_a_refs=("fixed/fixed_solid",),
            component_b_refs=("moving/moving_solid",),
            active_phase_ids=tuple(phases),
            max_penetration_m=max_penetration_m,
            source="test_contract",
        )
        return build_component_contact_intent_policy(
            (),
            geometry,
            phase_ids=("initial",),
            explicit_declarations=(declaration,),
        )

    def test_declared_expected_contact_penetration_is_exempted(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = self._fixture(root)
            policy = self._expected_contact_policy(
                geometry,
                max_penetration_m=0.00035,
            )
            audit = audit_family_collision_geometry(
                geometry,
                [self._sample(0.9999)],
                workspace_root=root,
                tolerance_m=0.00005,
                contact_intent_policy=policy,
            )

        self.assertTrue(audit["passed"])
        self.assertEqual(audit["illegal_event_count"], 0)
        self.assertEqual(
            audit["declared_expected_contact_exempted_event_count"],
            1,
        )
        self.assertGreater(audit["max_penetration_m"], 0.00005)

    def test_undeclared_pair_penetration_uses_global_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = self._fixture(root)
            audit = audit_family_collision_geometry(
                geometry,
                [self._sample(0.9999)],
                workspace_root=root,
                tolerance_m=0.00005,
            )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["illegal_event_count"], 1)
        self.assertEqual(
            audit["declared_expected_contact_exempted_event_count"],
            0,
        )

    def test_declared_contact_beyond_its_own_budget_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = self._fixture(root)
            policy = self._expected_contact_policy(
                geometry,
                max_penetration_m=0.00006,
            )
            audit = audit_family_collision_geometry(
                geometry,
                [self._sample(0.9999)],
                workspace_root=root,
                tolerance_m=0.00005,
                contact_intent_policy=policy,
            )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["illegal_event_count"], 1)
        self.assertEqual(
            audit["declared_expected_contact_exempted_event_count"],
            0,
        )

    def test_declared_contact_outside_active_phase_still_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = self._fixture(root)
            policy = self._expected_contact_policy(
                geometry,
                max_penetration_m=0.00035,
                phases=("powered",),
            )
            audit = audit_family_collision_geometry(
                geometry,
                [self._sample(0.9999)],
                workspace_root=root,
                tolerance_m=0.00005,
                contact_intent_policy=policy,
            )

        self.assertFalse(audit["passed"])
        self.assertEqual(audit["illegal_event_count"], 1)
        self.assertEqual(
            audit["declared_expected_contact_exempted_event_count"],
            0,
        )

    def test_queries_only_deterministic_cross_rigid_body_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mesh_path = root / "box.obj"
            mesh_path.write_text(
                trimesh.exchange.obj.export_obj(trimesh.creation.box()),
                encoding="ascii",
            )
            component_sha256 = sha256_file(mesh_path)
            geometry = {
                "schema": "step_per_solid_body_local_v1",
                "bodies": [
                    {
                        "body_id": "fixed_a",
                        "fixed": True,
                        "components": [
                            {
                                "solid_id": "fixed_a_solid",
                                "mesh": "box.obj",
                                "sha256": component_sha256,
                            }
                        ],
                    },
                    {
                        "body_id": "fixed_b",
                        "fixed": True,
                        "components": [
                            {
                                "solid_id": "fixed_b_solid",
                                "mesh": "box.obj",
                                "sha256": component_sha256,
                            }
                        ],
                    },
                    {
                        "body_id": "moving",
                        "fixed": False,
                        "components": [
                            {
                                "solid_id": "moving_solid_a",
                                "mesh": "box.obj",
                                "sha256": component_sha256,
                            },
                            {
                                "solid_id": "moving_solid_b",
                                "mesh": "box.obj",
                                "sha256": component_sha256,
                            },
                        ],
                    },
                ],
                "interpenetration_tolerance_m": 0.00035,
            }
            sample = {
                "time_s": 0.0,
                "phase_id": "cross_body_only",
                "body_poses": {
                    "fixed_a": {
                        "position_abs_m": [0.0, 0.0, 0.0],
                        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "fixed_b": {
                        "position_abs_m": [0.0, 0.0, 0.0],
                        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                    "moving": {
                        "position_abs_m": [0.8, 0.0, 0.0],
                        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            }
            with mock.patch.object(
                trimesh.collision.CollisionManager,
                "in_collision_internal",
                side_effect=AssertionError(
                    "same-manager collision query is forbidden"
                ),
            ):
                first = audit_family_collision_geometry(
                    geometry,
                    [sample],
                    workspace_root=root,
                )
                second = audit_family_collision_geometry(
                    geometry,
                    [sample],
                    workspace_root=root,
                )

        self.assertFalse(first["passed"])
        self.assertFalse(
            first["same_rigid_body_components_submitted_to_backend"]
        )
        self.assertEqual(
            first["collision_query_partition"],
            "all_cross_rigid_body_pairs_with_at_least_one_moving_body",
        )
        self.assertEqual(
            {
                tuple(event["body_pair"])
                for event in first["contact_events"]
            },
            {
                ("fixed_a", "moving"),
                ("fixed_b", "moving"),
            },
        )
        self.assertNotIn(
            ("fixed_a", "fixed_b"),
            {
                tuple(event["body_pair"])
                for event in first["contact_events"]
            },
        )
        self.assertEqual(first["contact_events"], second["contact_events"])
        self.assertEqual(
            first["first_illegal_event"],
            second["first_illegal_event"],
        )

    def test_prescribed_sweep_reports_dynamic_first_contact_and_clear_pass(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixed_path = root / "fixed.obj"
            moving_path = root / "moving.obj"
            fixed_path.write_text(
                trimesh.exchange.obj.export_obj(
                    trimesh.creation.box(extents=(1.0, 0.2, 0.2))
                ),
                encoding="ascii",
            )
            moving_path.write_text(
                trimesh.exchange.obj.export_obj(
                    trimesh.creation.box(extents=(0.2, 0.2, 1.0))
                ),
                encoding="ascii",
            )
            geometry = {
                "schema": "step_per_solid_body_local_v1",
                "collision_envelope_m": 12.0e-6,
                "runtime_interpenetration_tolerance_m": 50.0e-6,
                "bodies": [
                    {
                        "body_id": "fixed",
                        "fixed": True,
                        "components": [
                            {
                                "solid_id": "fixed_solid",
                                "mesh": "fixed.obj",
                                "sha256": sha256_file(fixed_path),
                            }
                        ],
                    },
                    {
                        "body_id": "moving",
                        "fixed": False,
                        "components": [
                            {
                                "solid_id": "moving_solid",
                                "mesh": "moving.obj",
                                "sha256": sha256_file(moving_path),
                            }
                        ],
                    },
                ],
            }

            def sample(
                x_value: float,
                quaternion: list[float],
                index: int,
            ) -> dict:
                return {
                    "time_s": float(index),
                    "phase_id": "prescribed_sweep",
                    "body_poses": {
                        "fixed": {
                            "position_abs_m": [0.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                        "moving": {
                            "position_abs_m": [x_value, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": quaternion,
                        },
                    },
                }

            home = sample(0.9, [1.0, 0.0, 0.0, 0.0], 0)
            quarter_turn = sample(
                0.9,
                [2.0**-0.5, 0.0, 2.0**-0.5, 0.0],
                1,
            )
            collision = audit_prescribed_swept_envelope(
                geometry,
                [home, quarter_turn],
                workspace_root=root,
            )
            clear = audit_prescribed_swept_envelope(
                geometry,
                [
                    sample(1.2, [1.0, 0.0, 0.0, 0.0], 0),
                    sample(
                        1.2,
                        [2.0**-0.5, 0.0, 2.0**-0.5, 0.0],
                        1,
                    ),
                ],
                workspace_root=root,
            )

        self.assertFalse(collision["passed"])
        self.assertEqual(collision["first_illegal_sample_index"], 1)
        self.assertGreater(
            collision["first_illegal_event"]["penetration_depth_m"],
            collision["tolerance_m"],
        )
        self.assertTrue(clear["passed"])
        self.assertIsNone(clear["first_illegal_sample_index"])

    def test_broad_exclusion_is_deduplicated_and_forbidden(self) -> None:
        exclusion = {
            "joint_id": "joint",
            "parent_body_id": "fixed",
            "child_body_id": "moving",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = {
                "collision_geometry": self._fixture(root)
                | {"excluded_joint_collision_pairs": [exclusion]},
                "branch_semantics": {
                    "excluded_joint_collision_pairs": [exclusion],
                },
                "samples": [self._sample(2.0)],
            }
            result = apply_saved_physical_legality_gate(
                {"verdict": "PASS", "failure_reasons": [], "physical_metrics": {}},
                trace,
                workspace_root=root,
            )

        self.assertEqual(len(disclosed_broad_collision_exclusions(trace)), 1)
        self.assertEqual(result["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "physical_broad_body_pair_collision_exclusion_forbidden",
            result["failure_reasons"],
        )

    def test_saved_penetration_overrides_prior_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            trace = {
                "collision_geometry": self._fixture(root),
                "branch_semantics": {},
                "samples": [self._sample(0.8)],
            }
            result = apply_saved_physical_legality_gate(
                {"verdict": "PASS", "failure_reasons": [], "physical_metrics": {}},
                trace,
                workspace_root=root,
            )

        self.assertEqual(result["verdict"], "PHYSICAL_FAILURE")
        self.assertIn("physical_illegal_interpenetration", result["failure_reasons"])
        self.assertGreater(
            result["physical_metrics"]["max_illegal_interpenetration_m"],
            0.19,
        )

    def test_explicit_zero_gap_contact_requires_geometry_and_runtime_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            declaration = interface_declaration(
                interface_id="contact-a",
                classification=CONTACT_PHYSICAL,
                body_a_id="fixed",
                body_b_id="moving",
                model="step_contact",
                source="trusted_test_runtime",
                parameters={
                    "contact_component_a_refs": ["fixed/fixed_solid"],
                    "contact_component_b_refs": ["moving/moving_solid"],
                    "contact_evidence_max_separation_m": 10.0e-6,
                },
            )
            sample = self._sample(1.000005)
            sample["contact_pair_counts"] = {"fixed|moving": 1}
            trace = {
                "branch": "physical",
                "branch_semantics": {
                    "physical_interface_policy": collision_policy_disclosure(
                        (declaration,),
                        branch="physical",
                    )
                },
                "collision_geometry": self._fixture(root),
                "samples": [sample],
            }
            result = apply_saved_physical_legality_gate(
                {
                    "verdict": "PASS",
                    "failure_reasons": [],
                    "physical_metrics": {},
                },
                trace,
                workspace_root=root,
            )
            preflight = audit_physical_trace_evidence(
                trace,
                workspace_root=root,
                require_wrench_audit=False,
            )

        self.assertEqual(result["verdict"], "PASS")
        self.assertTrue(result["component_contact_intent"]["passed"])
        self.assertEqual(
            result["physical_geometry_legality"][
                "runtime_confirmed_proximity_contact_event_count"
            ],
            1,
        )
        self.assertTrue(preflight["passed"])
        self.assertEqual(
            preflight["geometry_legality"][
                "runtime_confirmed_proximity_contact_event_count"
            ],
            1,
        )

    def test_explicit_zero_gap_contact_rejects_missing_runtime_contact(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            declaration = interface_declaration(
                interface_id="contact-a",
                classification=CONTACT_PHYSICAL,
                body_a_id="fixed",
                body_b_id="moving",
                model="step_contact",
                source="trusted_test_runtime",
                parameters={
                    "contact_component_a_refs": ["fixed/fixed_solid"],
                    "contact_component_b_refs": ["moving/moving_solid"],
                    "contact_evidence_max_separation_m": 10.0e-6,
                },
            )
            trace = {
                "branch": "physical",
                "branch_semantics": {
                    "physical_interface_policy": collision_policy_disclosure(
                        (declaration,),
                        branch="physical",
                    )
                },
                "collision_geometry": self._fixture(root),
                "samples": [self._sample(1.000005)],
            }
            result = apply_saved_physical_legality_gate(
                {
                    "verdict": "PASS",
                    "failure_reasons": [],
                    "physical_metrics": {},
                },
                trace,
                workspace_root=root,
            )

        self.assertEqual(result["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "expected_component_contact_missing",
            result["failure_reasons"],
        )

    def test_physical_only_preflight_authorizes_reduced_after_geometry_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            declaration = interface_declaration(
                interface_id="contact-a",
                classification=CONTACT_PHYSICAL,
                body_a_id="fixed",
                body_b_id="moving",
                model="step_contact",
                source="trusted_test_runtime",
            )
            trace = {
                "branch": "physical",
                "branch_semantics": {
                    "physical_interface_policy": collision_policy_disclosure(
                        (declaration,), branch="physical"
                    )
                },
                "collision_geometry": self._fixture(root),
                "samples": [self._sample(0.9999)],
            }
            result = audit_physical_trace_evidence(
                trace,
                workspace_root=root,
                require_wrench_audit=False,
            )

        self.assertTrue(result["passed"])
        self.assertTrue(result["reduced_branch_authorized"])

    def test_conditional_interface_preflight_requires_bound_runtime_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            geometry = self._fixture(root)
            gate = conditional_transmission_gate(
                geometry_plugin_id="generic_overlap_gate_v1",
                body_a_id="fixed",
                body_b_id="moving",
                endpoint_a_component_refs=("fixed/fixed_solid",),
                endpoint_b_component_refs=("moving/moving_solid",),
                activation_coordinate="overlap_m",
                enter_threshold_si=0.001,
                release_threshold_si=0.0005,
                proof_source="test_step_geometry",
            )
            declaration = interface_declaration(
                interface_id="conditional-a",
                classification=ANALYTICAL_TRANSMISSION,
                body_a_id="fixed",
                body_b_id="moving",
                model="finite_conditional_penalty",
                source="trusted_test_runtime",
                transmission_gate=gate,
            )
            sample = self._sample(2.0)
            sample["conditional_interface_states"] = [
                {
                    "interface_id": "conditional-a",
                    "state": "INACTIVE",
                    "transmission_active": False,
                    "transmitted_effort_si": 0.0,
                    "activation_coordinate_si": -0.1,
                    "constraints_satisfied": True,
                }
            ]
            trace = {
                "branch": "physical",
                "branch_semantics": {
                    "physical_interface_policy": collision_policy_disclosure(
                        (declaration,), branch="physical"
                    ),
                },
                "collision_geometry": geometry,
                "samples": [sample],
            }
            missing = audit_physical_trace_evidence(
                trace,
                workspace_root=root,
                require_wrench_audit=False,
            )
            trace["branch_semantics"][
                "functional_interface_geometry_binding"
            ] = build_functional_interface_geometry_binding(
                (declaration,), geometry
            )
            trace["branch_semantics"]["conditional_interface_runtime"] = {
                "schema_version": "conditional_interface_runtime_v1",
                "evaluated_every_physics_step": True,
                "physics_step_count": 1,
                "interfaces": [
                    {
                        "interface_id": "conditional-a",
                        "evaluation_count": 1,
                        "max_inactive_abs_transmitted_effort_si": 0.0,
                        "state_path": ["INACTIVE"],
                    }
                ],
            }
            complete = audit_physical_trace_evidence(
                trace,
                workspace_root=root,
                require_wrench_audit=False,
            )

        self.assertFalse(missing["passed"])
        self.assertIn(
            "conditional_interface_geometry_binding_invalid",
            missing["failure_reasons"],
        )
        self.assertFalse(missing["reduced_branch_authorized"])
        self.assertTrue(complete["passed"], complete["failure_reasons"])
        self.assertTrue(complete["reduced_branch_authorized"])

    def test_ungated_analytical_transmission_cannot_be_declared(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            declaration = interface_declaration(
                interface_id="through-air-law",
                classification=ANALYTICAL_TRANSMISSION,
                body_a_id="fixed",
                body_b_id="moving",
                model="finite_penalty_without_geometry",
                source="trusted_test_runtime",
            )

        self.assertEqual(
            caught.exception.code,
            "ANALYTICAL_TRANSMISSION_GEOMETRY_GATE_REQUIRED",
        )

    def test_plain_trace_normalizes_to_two_body_audit_schema(self) -> None:
        geometry = {
            "fixed": {"body_origin_abs_m": [0.0, 0.0, 0.0]},
            "moving": {"body_origin_abs_m": [2.0, 0.0, 0.0]},
            "fixed_components": [{"source": "fixed.obj"}],
            "moving_components": [{"source": "moving.obj"}],
        }
        normalized, samples = normalized_physical_audit_inputs(
            {
                "collision_geometry": geometry,
                "samples": [
                    {
                        "time_s": 0.0,
                        "moving_pose": {
                            "position_abs_m": [2.0, 0.0, 0.0],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                ],
            }
        )

        self.assertEqual(
            [item["body_id"] for item in normalized["bodies"]],
            ["fixed_boundary", "moving_group"],
        )
        self.assertIn("body_poses", samples[0])

    def test_plain_trace_uses_saved_moving_fixed_body_pose(self) -> None:
        fixed_pose = {
            "position_abs_m": [0.05, 0.0, 0.03],
            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        _, samples = normalized_physical_audit_inputs(
            {
                "collision_geometry": {
                    "fixed": {"body_origin_abs_m": [0.0, 0.0, 0.0]},
                    "moving": {"body_origin_abs_m": [0.0, 0.0, 0.0]},
                    "fixed_components": [{"source": "fixed.obj"}],
                    "moving_components": [{"source": "moving.obj"}],
                },
                "samples": [
                    {
                        "time_s": 0.0,
                        "fixed_pose": fixed_pose,
                        "moving_pose": {
                            "position_abs_m": [0.05, 0.0, 0.03],
                            "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        },
                    }
                ],
            }
        )
        self.assertEqual(
            samples[0]["body_poses"]["fixed_boundary"],
            fixed_pose,
        )

    def test_schedule_gate_recomputes_hash_and_checks_duration(self) -> None:
        physical_schedule = [
            {
                "phase_id": "drive",
                "wrench_kind": "force",
                "direction_abs": [1.0, 0.0, 0.0],
                "magnitude_si": 100.0,
                "duration_s": 0.42,
            }
        ]
        reduced_schedule = [dict(physical_schedule[0], duration_s=0.39)]
        program_only_hash = canonical_sha256(
            [{key: value for key, value in physical_schedule[0].items() if key != "duration_s"}]
        )
        result = _apply_executed_load_schedule_gate(
            {"verdict": "PASS", "failure_reasons": []},
            {
                "executed_phases": physical_schedule,
                "branch_semantics": {"same_external_load_sha256": program_only_hash},
            },
            {
                "executed_phases": reduced_schedule,
                "branch_semantics": {"same_external_load_sha256": program_only_hash},
            },
        )

        self.assertEqual(result["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "branch_executed_external_load_schedule_mismatch",
            result["failure_reasons"],
        )
        self.assertIn(
            "physical_external_load_schedule_hash_invalid",
            result["failure_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
