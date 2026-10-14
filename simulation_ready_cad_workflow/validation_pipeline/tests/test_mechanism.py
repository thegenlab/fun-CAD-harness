from __future__ import annotations

import hashlib
import math
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.contracts.failure_envelope import (
    FailureOwner,
)

from simulation_ready_cad_workflow.validation_pipeline.mechanism import (
    family_plugins,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.adapters import (
    DERIVED_COMPOUND_ADAPTER_ID,
    DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
    AdapterSpec,
    MeshAssetSpec,
    TwoBodyGeometrySpec,
    WholeMachineAngularTaskSpec,
    adapter_from_dict,
    adapter_to_dict,
    get_adapter,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_spec import (
    CompoundBodySpec,
    CompoundLinkageSpec,
    CompoundRevoluteJointSpec,
    compound_linkage_from_dict,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.compound_runtime import (
    PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED,
    _compound_joint_drift_failure,
    _compound_physical_solver,
    _compound_torsion_support_torques,
    _compound_worker_backend_matches,
    _require_compound_task_actuation_backend,
    _registered_compound_torsion_support,
    _soft_limit_torque,
    _registered_passive_fixture,
    _validated_compliant_surface_contacts,
    compound_bushing_parameters,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.certificate import (
    build_certificate,
    build_replacement,
    write_certificate_artifacts,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.comparison import (
    CompoundKinematicThresholds,
    KinematicThresholds,
    _behavior_failures,
    _compound_behavior_failures,
    _compound_phase_directional_travel,
    _physical_scope_evaluation,
    _validated_external_fixture_body_id,
    compare_composite_lane_ideal_dynamics_trace_files,
    compare_ideal_dynamics_trace_files,
    compare_trace_pair,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.derived_adapter import (
    _angular_task_spec,
    _disconnected_rigid_groups,
    _initial_interpenetrations,
    _task_spec,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.discovery import (
    _assert_registered_family_fields,
    _serialized_family_spec,
    discover_mechanisms,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.dispatch import (
    _dispatch_physical_task_budget,
    _job_runtime_budget,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.certificate import (
    _load_envelope,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.dispatch import (
    _bind_final_static_preflight,
    _mechanism_load_cases,
    _reviewer_collider_fidelity,
    _static_preflight_failure,
    build_dispatch_bundle,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.finalize import (
    _certificate_physical_trace_audits,
    _load_composite_lane_evidence,
    finalize_dispatch,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
    CompliantSupportSpec,
    MechanismAdapter,
    mechanism_adapter_for_plan,
    mechanism_adapter_inventory,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_collision import (
    _validated_component_representation,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.loads import (
    default_load_cases,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    JsonDocumentError,
    MAX_JSON_BYTES,
    MAX_TRACE_JSON_BYTES,
    canonical_sha256,
    lane_execution_source_sha256,
    read_json,
    read_trace_json,
    relative_to_root,
    sha256_file,
    validator_import_root,
    source_tree_sha256,
    write_json_atomic,
    write_trace_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.ideal_reference import (
    generate_ideal_reference_trace,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.launcher import (
    PyChronoLauncher,
    PyChronoLauncherKind,
    use_pychrono_launcher,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_ideal_comparison import (
    compare_composite_lane_physical_behavior_to_ideal_reference,
    compare_physical_behavior_to_ideal_reference,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    LoadCase,
    LoadPhase,
    MechanismValidationError,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.multi_output_shaft_comparison import (
    _metrics as _multi_output_shaft_metrics,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.worker import (
    _axis_aligned_box_spec,
    _axis_aligned_y_cylinder_spec,
    _configure_prismatic_travel_limit,
    _make_nsc_system,
    _oriented_box_spec,
    _rolling_element_assets,
    _simulate,
    _sphere_spec,
    _support_response_is_active,
    _validate_plain_runtime_body_origins,
    _write_body_local_mesh,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine import (
    WholeMachineRevoluteExecutionPlan,
    _whole_machine_max_memory_gib,
    build_whole_machine_dispatch,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.whole_machine_worker import (
    _snapshot_vector3,
)
from simulation_ready_cad_workflow.validation_pipeline.runtime import QueueManifest


_TEST_ADAPTER = AdapterSpec(
    adapter_id="test_prismatic_adapter",
    family="prismatic",
    required_source_hashes=frozenset(),
    supported_programs=frozenset(
        {"bidirectional_axis", "zero_input_gravity", "transverse_probe"}
    ),
)
_TEST_PYCHRONO_LAUNCHER = PyChronoLauncher(
    PyChronoLauncherKind.PYTHON,
    "/opt/test-physics/bin/python",
)


def _wide_mcm_adapter() -> AdapterSpec:
    adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
    assert adapter.geometry is not None
    return replace(
        adapter,
        geometry=replace(
            adapter.geometry,
            certifiable_progress_limit_m=0.05,
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _with_test_fidelity(
    root: Path,
    adapter: AdapterSpec,
    package_binding_sha256: str,
) -> AdapterSpec:
    report_path = root / "test_evidence" / f"{adapter.adapter_id}.collider_fidelity.json"
    attachment_path = (
        root / "test_evidence" / f"{adapter.adapter_id}.rigid_attachment.json"
    )
    _write_json(
        attachment_path,
        {
            "schema_version": "rigid_attachment_proof_v1",
            "status": "PASS",
            "passed": True,
            "package_binding_sha256": package_binding_sha256,
            "body_count": 1,
            "solid_count": 1,
            "bodies": [{"body_id": "test_fixture", "passed": True}],
            "test_fixture_only": True,
        },
    )
    attachment_binding = {
        "schema_version": "rigid_attachment_proof_v1",
        "status": "PASS",
        "passed": True,
        "report": attachment_path.relative_to(root).as_posix(),
        "report_sha256": _sha256(attachment_path),
        "package_binding_sha256": package_binding_sha256,
    }
    _write_json(
        report_path,
        {
            "schema_version": "collider_fidelity_v1",
            "status": "PASS",
            "passed": True,
            "package_binding_sha256": package_binding_sha256,
            "test_fixture_only": True,
        },
    )
    return replace(
        adapter,
        collider_fidelity={
            "schema_version": "collider_fidelity_v1",
            "status": "PASS",
            "passed": True,
            "report": report_path.relative_to(root).as_posix(),
            "report_sha256": _sha256(report_path),
            "package_binding_sha256": package_binding_sha256,
            "rigid_attachment_proof": attachment_binding,
        },
    )


def _build_package(root: Path, *, capabilities_ready: bool = True) -> tuple[Path, Path]:
    package = root / "package"
    package.mkdir(parents=True)
    contract = {
        "world_frame": {"gravity_m_s2": [0.0, 0.0, -9.81]},
        "parts": [
            {
                "part_id": "frame",
                "rigid_group_id": "fixed_group",
                "declared_bounds_m": {
                    "min": [-0.10, -0.02, -0.01],
                    "max": [0.10, 0.02, 0.01],
                },
            },
            {
                "part_id": "carriage",
                "rigid_group_id": "moving_group",
                "declared_bounds_m": {
                    "min": [-0.02, -0.03, -0.015],
                    "max": [0.02, 0.03, 0.015],
                },
            },
        ],
        "rigid_groups": [
            {"rigid_group_id": "fixed_group", "part_ids": ["frame"]},
            {"rigid_group_id": "moving_group", "part_ids": ["carriage"]},
        ],
        "frames": [
            {"frame_id": "joint_axis", "origin_m": [0.0, 0.0, 0.0]},
            {"frame_id": "force_input", "origin_m": [0.0, 0.03, 0.0]},
            {"frame_id": "output", "origin_m": [0.0, 0.0, 0.0]},
        ],
    }
    capsule = {
        "capsule_id": "linear_stage",
        "input_ports": [
            {
                "port_id": "input_force",
                "frame_id": "force_input",
                "wrench_type": "force",
            }
        ],
        "observables": [
            {
                "observable_id": "carriage_pose",
                "frame_id": "output",
                "quantity": "pose_6d",
            }
        ],
        "reduced_model_claim": {
            "verification_status": "UNVERIFIED",
            "family": "prismatic",
            "fixed_group_ids": ["fixed_group"],
            "moving_group_ids": ["moving_group"],
            "joint_frame_id": "joint_axis",
            "axis_abs": [1.0, 0.0, 0.0],
        },
    }
    mapping = {
        "task_mappings": [
            {"task_id": "transfer_payload", "capsule_ids": ["linear_stage"]}
        ]
    }
    documents = {
        "assembly.step": "ISO-10303-21;\nEND-ISO-10303-21;\n",
        "assembly_contract.json": contract,
        "mechanism_capsules/linear_stage.json": capsule,
        "machine_task_mapping.json": mapping,
        "source_components/rail.step": "test rail\n",
        "source_components/carriage.step": "test carriage\n",
    }
    for relative, value in documents.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, str):
            path.write_text(value, encoding="ascii")
        else:
            _write_json(path, value)
    artifacts = [
        {"path": relative, "sha256": _sha256(package / relative)}
        for relative in documents
    ]
    manifest = {
        "package_id": "test_machine",
        "design_version": "1.0.0",
        "assembly_file": "assembly.step",
        "single_source_cad": {
            "policy": "required_v1",
            "authoritative_artifact": "assembly.step",
            "derived_assets": "hash_bound_and_regenerated",
        },
        "assembly_contract_file": "assembly_contract.json",
        "machine_task_mapping_file": "machine_task_mapping.json",
        "mechanism_capsule_files": ["mechanism_capsules/linear_stage.json"],
        "requirements_ref": {
            "requirements_id": "external_requirements",
            "sha256": "0" * 64,
        },
        "artifacts": artifacts,
    }
    _write_json(package / "package.json", manifest)
    review_asset = root / "visual_review" / "review.png"
    review_asset.parent.mkdir(parents=True, exist_ok=True)
    review_asset.write_bytes(b"test-render")
    _write_json(
        root / "design_input.json",
        {
            "schema_version": "0.1",
            "original_prompt": "Build a test linear-stage machine.",
            "plain_language_goal": "A carriage moves on a fixed rail.",
            "engineering_brief": {"mechanism": "linear stage"},
            "visual_acceptance_checklist": ["rail and carriage are recognizable"],
            "motion_review_roles": ["travel-pose"],
        },
    )
    _write_json(
        root / "visual_review" / "summary.json",
        {
            "schema_version": "0.1",
            "status": "PASSED",
            "assembly_step_sha256": _sha256(package / "assembly.step"),
            "rounds": [
                {
                    "round": 1,
                    "render_assets": [
                        {
                            "role": role,
                            "path": "visual_review/review.png",
                            "sha256": _sha256(review_asset),
                        }
                        for role in ("iso", "front", "side", "top", "travel-pose")
                    ],
                    "findings": [],
                }
            ],
        },
    )
    report = root / "intake_report.json"
    _write_json(
        report,
        {
            "status": "ACCEPTED_WITH_WARNINGS",
            "package_dir": str(package.resolve()),
            "capabilities": {
                "can_enter_physical_preflight": capabilities_ready,
                "can_attempt_reduced_certification": capabilities_ready,
                "physical_preflight_blockers": (
                    [] if capabilities_ready else ["STEP_GEOMETRY_NOT_VERIFIED"]
                ),
            },
            "package_summary": {
                "package_id": "test_machine",
                "design_version": "1.0.0",
                "part_count": 2,
                "rigid_group_count": 2,
                "frame_count": 3,
                "capsule_count": 1,
                "mapped_task_count": 1,
            },
        },
    )
    return package, report


def _trace(
    plan: object,
    load_case: LoadCase,
    *,
    branch: str,
    wrong_axis: bool = False,
    wrong_link_semantics: bool = False,
) -> dict[str, object]:
    axis_progress = 0.001
    sample_specs: list[tuple[str, float]] = [("initial", 0.0)]
    if load_case.program == "bidirectional_axis":
        sample_specs.extend(
            [
                (load_case.phases[0].phase_id, axis_progress),
                (load_case.phases[1].phase_id, axis_progress),
                (load_case.phases[2].phase_id, 0.0),
                (load_case.phases[3].phase_id, 0.0),
            ]
        )
    elif load_case.program == "task_range_roundtrip":
        sample_specs.extend(
            [
                (load_case.phases[0].phase_id, 0.04),
                (load_case.phases[1].phase_id, 0.04),
                (load_case.phases[2].phase_id, 0.0),
                (load_case.phases[3].phase_id, 0.0),
            ]
        )
    else:
        sample_specs.extend((phase.phase_id, 0.0) for phase in load_case.phases)
    samples = []
    for index, (phase_id, progress) in enumerate(sample_specs):
        displacement = [progress, 0.0, 0.0]
        lateral = 0.0
        if wrong_axis and index > 0:
            displacement = [0.0, 0.002, 0.0]
            progress = 0.0
            lateral = 0.002
        samples.append(
            {
                "time_s": index * 0.01,
                "phase_id": phase_id,
                "generalized_progress_m": progress,
                "axis_speed_m_s": 0.0,
                "lateral_shift_m": lateral,
                "rotation_from_start_rad": 0.0,
                "observable_displacement_abs_m": displacement,
                "contact_count": 1 if branch == "physical" else 0,
            }
        )
    initial_pose = {
        "position_abs_m": [0.0, 0.0, 0.0],
        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    internal_link_count = 0 if branch == "physical" else 1
    ideal_links = [] if branch == "physical" else ["ChLinkLockPrismatic"]
    if wrong_link_semantics and branch == "physical":
        internal_link_count = 1
        ideal_links = ["ChLinkLockPrismatic"]
    executed_phases = [
        {
            "phase_id": phase.phase_id,
            "wrench_kind": phase.wrench_kind,
            "direction_abs": (
                None if phase.direction_abs is None else list(phase.direction_abs)
            ),
            "magnitude_si": phase.magnitude_si,
            "duration_s": phase.duration_s,
        }
        for phase in load_case.phases
    ]
    payload = {
        "status": "COMPLETED",
        "branch": branch,
        "package_binding_sha256": plan.package_binding_sha256,
        "capsule_id": plan.capsule_id,
        "load_case_id": load_case.load_case_id,
        "plan_sha256": hashlib.sha256(
            json.dumps(
                plan.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest(),
        "load_case_sha256": hashlib.sha256(
            json.dumps(
                load_case.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("ascii")
        ).hexdigest(),
        "axis_abs": list(plan.axis_abs),
        "engine": {
            "internal_ideal_joint_count": internal_link_count,
            "ideal_links": ideal_links,
        },
        "branch_semantics": {
            "same_mass_kg": load_case.moving_mass_kg,
            "same_inertia_xx_kg_m2": [1e-5, 2e-5, 3e-5],
            "same_initial_pose": initial_pose,
            "same_external_load_sha256": canonical_sha256(executed_phases),
            "replaced_contacts_disabled": branch == "reduced",
        },
        "executed_phases": executed_phases,
        "samples": samples,
    }
    if plan.adapter_id == "mcmaster_6725k41_8438k4_v1":
        adapter = get_adapter(plan.adapter_id)
        assert adapter.geometry is not None
        assert adapter.geometry.rolling_elements is not None
        rolling_count = len(adapter.geometry.rolling_elements.elements)
        payload["collision_geometry"] = {
            "rolling_elements": {
                "runtime_enabled": branch == "physical",
                "count": rolling_count,
            }
        }
        payload["rolling_element_validation"] = (
            {
                "required": True,
                "passed": True,
                "rolling_response_count": rolling_count,
                "retained_count": rolling_count,
                "total_count": rolling_count,
            }
            if branch == "physical" and load_case.program == "bidirectional_axis"
            else None
        )
    return payload


class DerivedAdapterTests(unittest.TestCase):
    def test_schema_02_positive_prismatic_calibration_derives_public_travel(self) -> None:
        task = _task_spec(
            SimpleNamespace(
                task_ids=(
                    "calibrate_three_mass_points",
                    "repeat_one_kilogram_reading",
                ),
                characteristic_length_m=0.24,
            ),
            {
                "schema_version": "0.2",
                "tasks": [
                    {
                        "task_id": "calibrate_three_mass_points",
                        "description": "Measure positive-axis spring-scale travel.",
                        "acceptance_requirements": [
                            {
                                "metric_id": "signed_displacement_m",
                                "target": {
                                    "comparator": "within",
                                    "value": value,
                                    "tolerance": 0.002,
                                    "unit": "m",
                                },
                            }
                            for value in (0.01, 0.02, 0.03)
                        ],
                    },
                    {
                        "task_id": "repeat_one_kilogram_reading",
                        "description": "Repeat the same positive-axis reading.",
                        "acceptance_requirements": [
                            {
                                "metric_id": "return_to_start_m",
                                "target": {
                                    "comparator": "at_most",
                                    "value": 0.002,
                                    "unit": "m",
                                },
                            }
                        ],
                    },
                ],
            },
        )

        self.assertEqual(task.outbound_task_id, "calibrate_three_mass_points")
        self.assertEqual(task.return_task_id, "calibrate_three_mass_points")
        self.assertAlmostEqual(task.outbound_progress_m, 0.03)
        self.assertEqual(task.home_progress_m, 0.0)

    def test_task_pair_ignores_home_mention_in_open_task(self) -> None:
        class Plan:
            task_ids = ("open_drawer", "close_drawer")
            characteristic_length_m = 0.10

        requirements = {
            "tasks": [
                {
                    "task_id": "open_drawer",
                    "description": "Pull from home along global +X.",
                    "success_criteria": [
                        "Move at least 0.020 m toward +X from home."
                    ],
                },
                {
                    "task_id": "close_drawer",
                    "description": "Reverse and return the drawer.",
                    "success_criteria": [
                        "Apply force along -X and return to 0.000 m."
                    ],
                },
            ]
        }
        task = _task_spec(Plan(), requirements)
        self.assertEqual(task.outbound_task_id, "open_drawer")
        self.assertEqual(task.return_task_id, "close_drawer")
        self.assertAlmostEqual(task.outbound_progress_m, 0.020)

    def test_home_pose_interference_detects_volume_but_allows_touching(self) -> None:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.gp import gp_Pnt

        fixed = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        overlapping = BRepPrimAPI_MakeBox(
            gp_Pnt(9.0, 0.0, 0.0), 5.0, 5.0, 5.0
        ).Shape()
        touching = BRepPrimAPI_MakeBox(
            gp_Pnt(10.0, 0.0, 0.0), 5.0, 5.0, 5.0
        ).Shape()
        hits = _initial_interpenetrations([fixed, overlapping], (1,), (2,))
        self.assertEqual(len(hits), 1)
        self.assertAlmostEqual(hits[0]["overlap_volume_mm3"], 25.0)
        self.assertEqual(
            _initial_interpenetrations([fixed, touching], (1,), (2,)), ()
        )

    def test_rigid_group_connectivity_requires_real_contact_path(self) -> None:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.gp import gp_Pnt

        first = BRepPrimAPI_MakeBox(10.0, 10.0, 10.0).Shape()
        touching = BRepPrimAPI_MakeBox(
            gp_Pnt(10.0, 0.0, 0.0), 10.0, 10.0, 10.0
        ).Shape()
        separated = BRepPrimAPI_MakeBox(
            gp_Pnt(25.0, 0.0, 0.0), 5.0, 5.0, 5.0
        ).Shape()
        self.assertEqual(
            _disconnected_rigid_groups(
                [first, touching], {"group_moving": (1, 2)}
            ),
            (),
        )
        failures = _disconnected_rigid_groups(
            [first, touching, separated], {"group_moving": (1, 2, 3)}
        )
        self.assertEqual(failures[0]["rigid_group_id"], "group_moving")
        self.assertEqual(
            failures[0]["components"],
            [["solid_1", "solid_2"], ["solid_3"]],
        )

    def test_component_collision_assets_round_trip(self) -> None:
        identity = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
        component = MeshAssetSpec(
            workspace_relative_path="component.obj",
            sha256="0" * 64,
            units_to_m=1.0,
            rotation_abs_from_source=identity,
            convex=True,
        )
        adapter = AdapterSpec(
            adapter_id="component_adapter",
            family="prismatic",
            required_source_hashes=frozenset(),
            supported_programs=frozenset({"bidirectional_axis"}),
            geometry=TwoBodyGeometrySpec(
                fixed_mesh=component,
                moving_mesh=component,
                fixed_origin_abs_m=(0.0, 0.0, 0.0),
                moving_origin_abs_m=(0.0, 0.0, 0.0),
                fixed_collision_meshes=(component,),
                moving_collision_meshes=(component,),
            ),
        )
        restored = adapter_from_dict(adapter_to_dict(adapter))
        self.assertTrue(restored.geometry.fixed_collision_meshes[0].convex)
        self.assertEqual(
            restored.geometry.moving_collision_meshes[0].workspace_relative_path,
            "component.obj",
        )

class CompliantSurfaceContactTests(unittest.TestCase):
    @staticmethod
    def geometry() -> dict:
        return {
            "interpenetration_tolerance_m": 0.00035,
            "bodies": [
                {
                    "body_id": "ground",
                    "components": [{"solid_id": "glass"}],
                },
                {
                    "body_id": "rocker",
                    "components": [{"solid_id": "rubber"}],
                },
            ],
        }

    @staticmethod
    def contact() -> dict:
        return {
            "contact_id": "rubber_glass",
            "model": "validator_compliant_surface_drag_v1",
            "fixed_body_id": "ground",
            "moving_body_id": "rocker",
            "fixed_component_refs": ["ground/glass"],
            "moving_component_refs": ["rocker/rubber"],
            "contact_point_abs_m": [0.2, 0.1, 0.074],
            "normal_abs": [0.0, 0.0, 1.0],
            "normal_load_n": 12.0,
            "friction_coefficient": 0.55,
            "tangential_damping_n_s_m": 8.0,
        }

    def test_contact_is_bound_to_exact_components(self) -> None:
        result = _validated_compliant_surface_contacts(
            {"compliant_surface_contacts": [self.contact()]},
            self.geometry(),
        )
        self.assertEqual(result[0]["moving_component_refs"], ("rocker/rubber",))
        self.assertEqual(result[0]["max_penetration_m"], 0.00035)

    def test_discovery_preserves_only_supported_contact_extension(self) -> None:
        parsed = type(
            "Parsed",
            (),
            {"to_dict": lambda self: {"linkage_id": "typed"}},
        )()
        from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import (
            family_spec_plugin_for_kind,
        )

        payload = _serialized_family_spec(
            {
                "linkage_id": "typed",
                "compliant_surface_contacts": [self.contact()],
                "untrusted_unknown": "discard-me",
            },
            parsed,
            family_spec_plugin_for_kind(None),
        )
        self.assertIn("compliant_surface_contacts", payload)
        self.assertNotIn("untrusted_unknown", payload)

    def test_unknown_component_is_rejected(self) -> None:
        contact = self.contact()
        contact["moving_component_refs"] = ["rocker/not_present"]
        with self.assertRaises(MechanismValidationError) as caught:
            _validated_compliant_surface_contacts(
                {"compliant_surface_contacts": [contact]},
                self.geometry(),
            )
        self.assertEqual(
            caught.exception.code,
            "COMPLIANT_SURFACE_COMPONENT_BINDING_INVALID",
        )


class MechanismIOTests(unittest.TestCase):
    def test_compliant_support_can_apply_force_at_body_center(self) -> None:
        support = CompliantSupportSpec(
            support_id="centered_spring",
            chrono_type="ExplicitChLoadBodyForceTorque",
            translational_stiffness_n_m=(0.0, 0.0, 10.0),
            translational_damping_n_s_m=(0.0, 0.0, 1.0),
            rotational_stiffness_n_m_rad=(0.0, 0.0, 0.0),
            rotational_damping_n_m_s_rad=(0.0, 0.0, 0.0),
            free_coordinate="none",
            source="test",
            force_application_point="body_center",
        )

        self.assertEqual(
            support.to_dict()["application_frame"],
            "moving_body_center",
        )

    def test_constitutive_support_stays_active_when_guide_gate_releases(
        self,
    ) -> None:
        physical_support = object()
        constitutive_support = object()

        self.assertFalse(
            _support_response_is_active(
                physical_support,
                physical_support_specs=(physical_support,),
                guide_gate_active=False,
            )
        )
        self.assertTrue(
            _support_response_is_active(
                constitutive_support,
                physical_support_specs=(physical_support,),
                guide_gate_active=False,
            )
        )

    def test_plain_runtime_rejects_body_origin_drift_from_collision_manifest(
        self,
    ) -> None:
        execution = SimpleNamespace(
            adapter=SimpleNamespace(
                family_spec={
                    "collision_geometry": {
                        "bodies": [
                            {
                                "body_id": "fixed_group",
                                "initial_center_abs_m": [0.0, -0.012, 0.097],
                            },
                            {
                                "body_id": "moving_group",
                                "initial_center_abs_m": [0.00075, -0.026, 0.040875],
                            },
                        ]
                    }
                }
            )
        )
        matching = SimpleNamespace(
            fixed_origin_abs_m=(0.0, -0.012, 0.097),
            moving_origin_abs_m=(0.00075, -0.026, 0.040875),
        )
        _validate_plain_runtime_body_origins(execution, matching)

        with self.assertRaises(MechanismValidationError) as caught:
            _validate_plain_runtime_body_origins(
                execution,
                SimpleNamespace(
                    fixed_origin_abs_m=(0.0, 0.0, 0.0),
                    moving_origin_abs_m=(0.0, 0.0, 0.0),
                ),
            )

        self.assertEqual(
            caught.exception.code,
            "MECHANISM_RUNTIME_BODY_ORIGIN_MISMATCH",
        )

    def test_compound_physical_solver_uses_admm_for_contact_and_bushings(
        self,
    ) -> None:
        events: list[tuple[str, object]] = []

        class Solver:
            def SetMaxIterations(self, value: int) -> None:
                events.append(("iterations", value))

            def SetTolerance(self, value: float) -> None:
                events.append(("tolerance", value))

        solver = Solver()

        class Chrono:
            @staticmethod
            def ChSolverADMM() -> Solver:
                events.append(("solver", "ADMM"))
                return solver

        self.assertIs(_compound_physical_solver(Chrono), solver)
        self.assertEqual(
            events,
            [
                ("solver", "ADMM"),
                ("iterations", 500),
                ("tolerance", 1.0e-10),
            ],
        )

    def test_compound_joint_drift_failure_is_strict_and_provenance_owned(self) -> None:
        at_limit = _compound_joint_drift_failure(
            branch="physical",
            joint_drifts_m={"drive-pin": 0.003},
            declared_max_joint_drift_m=0.003,
            mechanism_plugin_id="generic.compound_linkage_v1",
            mechanism_registration_source="installed_harness_code",
            load_case_id="rated",
            phase_index=0,
            phase_id="cycle-1",
            time_s=0.125,
            step_s=0.00002,
            sample_interval_s=0.01,
        )
        reduced = _compound_joint_drift_failure(
            branch="reduced",
            joint_drifts_m={"drive-pin": 0.004},
            declared_max_joint_drift_m=0.003,
            mechanism_plugin_id="generic.compound_linkage_v1",
            mechanism_registration_source="installed_harness_code",
            load_case_id="rated",
            phase_index=0,
            phase_id="cycle-1",
            time_s=0.125,
            step_s=0.00002,
            sample_interval_s=0.01,
        )
        failure = _compound_joint_drift_failure(
            branch="physical",
            joint_drifts_m={"drive-pin": 0.004, "output-pin": 0.002},
            declared_max_joint_drift_m=0.003,
            mechanism_plugin_id="generic.compound_linkage_v1",
            mechanism_registration_source="installed_harness_code",
            load_case_id="rated",
            phase_index=0,
            phase_id="cycle-1",
            time_s=0.125,
            step_s=0.00002,
            sample_interval_s=0.01,
        )

        self.assertIsNone(at_limit)
        self.assertIsNone(reduced)
        self.assertIsNotNone(failure)
        self.assertEqual(
            failure.code,
            PHYSICAL_COMPOUND_JOINT_DRIFT_EXCEEDED,
        )
        self.assertEqual(failure.failure_envelope["owner"], "design_package")
        self.assertEqual(
            failure.failure_envelope["evidence_validity"],
            "VALID_MEASUREMENT",
        )
        self.assertEqual(failure.details["joint_id"], "drive-pin")
        self.assertEqual(failure.details["observed_joint_drift_m"], 0.004)
        self.assertEqual(failure.details["load_case_id"], "rated")
        self.assertEqual(failure.details["phase_index"], 0)
        self.assertEqual(failure.details["step_s"], 0.00002)
        self.assertEqual(failure.details["sample_interval_s"], 0.01)

        for registration_source in (
            "installed_harness_code",
            "project_design_package",
        ):
            with self.subTest(registration_source=registration_source):
                design_failure = _compound_joint_drift_failure(
                    branch="physical",
                    joint_drifts_m={"drive-pin": 0.004},
                    declared_max_joint_drift_m=0.003,
                    mechanism_plugin_id="project.compound_linkage_v1",
                    mechanism_registration_source=registration_source,
                    load_case_id="rated",
                    phase_index=0,
                    phase_id="cycle-1",
                    time_s=0.125,
                    step_s=0.00002,
                    sample_interval_s=0.01,
                )
                self.assertIsNotNone(design_failure)
                self.assertEqual(
                    design_failure.failure_envelope["owner"],
                    "design_package",
                )

        reviewer_failure = _compound_joint_drift_failure(
            branch="physical",
            joint_drifts_m={"drive-pin": 0.004},
            declared_max_joint_drift_m=0.003,
            mechanism_plugin_id="reviewer.compound_linkage_v1",
            mechanism_registration_source="reviewer_mechanism_adapter",
            load_case_id="rated",
            phase_index=0,
            phase_id="cycle-1",
            time_s=0.125,
            step_s=0.00002,
            sample_interval_s=0.01,
        )
        self.assertIsNotNone(reviewer_failure)
        self.assertEqual(
            reviewer_failure.failure_envelope["owner"], "unclassified"
        )
        self.assertEqual(
            reviewer_failure.failure_envelope["evidence_validity"],
            "VALID_MEASUREMENT",
        )
        self.assertEqual(
            reviewer_failure.failure_envelope["source"]["plugin_id"],
            "reviewer.compound_linkage_v1",
        )

    def test_compound_bushings_are_finite_engineering_bearings(self) -> None:
        parameters = compound_bushing_parameters()
        self.assertEqual(parameters["translational_stiffness_n_m"], 5_000_000.0)
        self.assertEqual(parameters["translational_damping_n_s_m"], 1_500.0)
        self.assertEqual(parameters["axis_stiffness_n_m_rad"], 5_000.0)
        self.assertGreater(parameters["free_axis_damping_n_m_s_rad"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in parameters.values()))

    def test_high_component_physical_lane_receives_extended_budget(self) -> None:
        ordinary = SimpleNamespace(
            family_spec={"collision_geometry": {"component_count": 80}}
        )
        decomposed = SimpleNamespace(
            family_spec={"collision_geometry": {"component_count": 94}}
        )

        self.assertEqual(
            _job_runtime_budget(
                ordinary, branch="physical", estimated_steps=10_000
            )[0],
            300.0,
        )
        self.assertEqual(
            _job_runtime_budget(
                decomposed, branch="physical", estimated_steps=10_000
            )[0],
            900.0,
        )

    def test_task_rich_mechanisms_receive_audited_physical_lane_budgets(self) -> None:
        ordinary = SimpleNamespace(adapter_spec={"family_spec": {}})
        deadbolt = SimpleNamespace(
            adapter_spec={
                "family_spec": {"mechanism_kind": "deadbolt_door_lock"}
            }
        )
        spring_scale = SimpleNamespace(
            adapter_spec={
                "family_spec": {
                    "mechanism_kind": "spring_scale_prismatic_spring"
                }
            }
        )
        spring_scale_v1 = SimpleNamespace(
            adapter_spec={
                "family_spec": {
                    "mechanism_kind": "spring_scale_prismatic_spring_v1"
                }
            }
        )
        reviewer_spring_scale = SimpleNamespace(
            adapter_spec={
                "family_spec": {
                    "mechanism_kind": "spring_scale_guided_carriage"
                }
            }
        )
        calibrated_spring_scale = SimpleNamespace(
            adapter_spec={
                "family_spec": {
                    "mechanism_kind": "spring_scale_calibrated_prismatic"
                }
            }
        )
        folding_knife = SimpleNamespace(
            adapter_spec={
                "family_spec": {
                    "mechanism_kind": "folding_knife_liner_lock"
                }
            }
        )

        self.assertEqual(_dispatch_physical_task_budget((ordinary,)), 8)
        self.assertEqual(_dispatch_physical_task_budget((deadbolt,)), 9)
        self.assertEqual(_dispatch_physical_task_budget((spring_scale,)), 10)
        self.assertEqual(_dispatch_physical_task_budget((spring_scale_v1,)), 10)
        self.assertEqual(
            _dispatch_physical_task_budget((reviewer_spring_scale,)), 10
        )
        self.assertEqual(
            _dispatch_physical_task_budget((calibrated_spring_scale,)), 10
        )
        self.assertEqual(_dispatch_physical_task_budget((folding_knife,)), 9)

    def test_phone_hold_selects_validator_owned_passive_fixture(self) -> None:
        execution = SimpleNamespace(
            adapter=SimpleNamespace(
                family_spec={"mechanism_kind": "rotary_detent_phone_stand"}
            ),
            load_case=SimpleNamespace(program="task_phone_stand_loaded_hold"),
        )
        linkage = SimpleNamespace(
            output_body_id="support_plate",
            bodies=(
                SimpleNamespace(body_id="ground", fixed=True),
                SimpleNamespace(body_id="support_plate", fixed=False),
            ),
        )

        fixture = _registered_passive_fixture(execution, linkage)

        self.assertIsNotNone(fixture)
        self.assertEqual(fixture["fixture_id"], "harness_phone_fixture")
        self.assertEqual(fixture["shape"], "passive_phone_box")
        self.assertAlmostEqual(fixture["mass_kg"], 0.200)
        self.assertEqual(fixture["target_body_id"], "support_plate")
        execution.load_case.program = "task_phone_stand_fold_return"
        self.assertIsNone(_registered_passive_fixture(execution, linkage))

    def test_clothespin_hold_selects_passive_wood_rod_fixture(self) -> None:
        execution = SimpleNamespace(
            adapter=SimpleNamespace(
                family_spec={
                    "mechanism_kind": "clothespin_dual_jaw_torsion"
                }
            ),
            load_case=SimpleNamespace(program="task_clothespin_hold"),
        )
        linkage = SimpleNamespace(
            input_body_id="left_jaw",
            output_body_id="right_jaw",
            bodies=(
                SimpleNamespace(body_id="hinge_carrier", fixed=True),
                SimpleNamespace(body_id="left_jaw", fixed=False),
                SimpleNamespace(body_id="right_jaw", fixed=False),
            ),
        )

        fixture = _registered_passive_fixture(execution, linkage)

        self.assertIsNotNone(fixture)
        self.assertEqual(fixture["fixture_id"], "harness_wood_rod_fixture")
        self.assertEqual(fixture["shape"], "passive_wood_rod_cylinder")
        self.assertAlmostEqual(fixture["radius_m"], 0.004)
        self.assertAlmostEqual(fixture["length_m"], 0.080)
        self.assertEqual(
            set(fixture["required_contact_body_ids"]),
            {"left_jaw", "right_jaw"},
        )
        self.assertEqual(
            fixture["disturbance_phase_ids"], ("tong_hold_verify",)
        )
        self.assertEqual(
            fixture["release_phase_ids"], ("jaw_release_spring_close",)
        )
        self.assertEqual(
            fixture["centering_phase_ids"],
            ("jaw_release_spring_close", "bilateral_grip_settle"),
        )
        self.assertEqual(fixture["contact_normal_stiffness_n_m"], 10_000.0)
        self.assertEqual(fixture["contact_normal_damping_n_s_m"], 8.0)
        execution.load_case.program = "task_clothespin_ten_cycle_return"
        self.assertIsNone(_registered_passive_fixture(execution, linkage))

    def test_clothespin_selects_finite_grounded_coupling_spring(self) -> None:
        execution = SimpleNamespace(
            adapter=SimpleNamespace(
                family_spec={
                    "mechanism_kind": "clothespin_dual_jaw_torsion",
                    "torsion_spring_parameters": {
                        "body_a_id": "left_jaw",
                        "body_b_id": "right_jaw",
                        "anchor_body_id": "hinge_carrier",
                        "axis_abs": [0.0, 1.0, 0.0],
                        "stiffness_n_m_rad": 0.65,
                        "damping_n_m_s_rad": 0.03,
                        "coupling_stiffness_n_m_rad": 12.0,
                        "coupling_damping_n_m_s_rad": 0.20,
                        "preload_torque_nm": 0.08,
                        "rest_relative_angle_rad": 0.0,
                    },
                    "jaw_stop_geometry": {"closed_angle_rad": 0.0},
                }
            )
        )
        linkage = SimpleNamespace(
            bodies=(
                SimpleNamespace(body_id="hinge_carrier", fixed=True),
                SimpleNamespace(body_id="left_jaw", fixed=False),
                SimpleNamespace(body_id="right_jaw", fixed=False),
            ),
            revolute_joints=(
                SimpleNamespace(
                    child_body_id="left_jaw", axis_abs=(0.0, 1.0, 0.0)
                ),
                SimpleNamespace(
                    child_body_id="right_jaw", axis_abs=(0.0, -1.0, 0.0)
                ),
            ),
        )

        support = _registered_compound_torsion_support(execution, linkage)

        self.assertIsNotNone(support)
        self.assertEqual(support["anchor_body_id"], "hinge_carrier")
        self.assertEqual(support["axis_abs"], (0.0, 1.0, 0.0))
        self.assertEqual(support["body_a_axis_abs"], (0.0, 1.0, 0.0))
        self.assertEqual(support["body_b_axis_abs"], (0.0, -1.0, 0.0))
        equal_torques = _compound_torsion_support_torques(
            support,
            angle_a_rad=-0.1,
            speed_a_rad_s=0.0,
            angle_b_rad=-0.1,
            speed_b_rad_s=0.0,
        )
        self.assertAlmostEqual(equal_torques[0], -0.015)
        self.assertAlmostEqual(equal_torques[1], -0.015)
        coupled_torques = _compound_torsion_support_torques(
            support,
            angle_a_rad=-0.35,
            speed_a_rad_s=0.0,
            angle_b_rad=-0.33,
            speed_b_rad_s=0.0,
        )
        self.assertGreater(coupled_torques[0], equal_torques[0])
        self.assertLess(coupled_torques[1], equal_torques[1])

    def test_compound_trace_allows_only_validated_external_fixture_body(self) -> None:
        fixture = {
            "schema_version": "validator_passive_fixture_v1",
            "fixture_id": "harness_phone_fixture",
            "shape": "passive_phone_box",
            "source": "installed_validator_phone_fixture_v1",
            "runtime_enabled": True,
            "supported": True,
            "direct_actuation": False,
            "mass_kg": 0.200,
        }
        trace = {
            "collision_geometry": {
                "external_contact_fixture": dict(fixture)
            },
            "branch_semantics": {
                "external_passive_fixture": dict(fixture),
                "direct_passive_fixture_actuation": False,
            },
        }

        self.assertEqual(
            _validated_external_fixture_body_id(trace, branch="physical"),
            "harness_phone_fixture",
        )

    def test_compound_trace_allows_validated_wood_rod_fixture(self) -> None:
        fixture = {
            "schema_version": "validator_passive_fixture_v1",
            "fixture_id": "harness_wood_rod_fixture",
            "shape": "passive_wood_rod_cylinder",
            "source": "installed_validator_wood_rod_fixture_v1",
            "runtime_enabled": True,
            "supported": True,
            "direct_actuation": False,
            "radius_m": 0.004,
            "length_m": 0.080,
            "mass_kg": math.pi * 0.004 * 0.004 * 0.080 * 700.0,
            "required_contact_body_ids": ["left_jaw", "right_jaw"],
        }
        trace = {
            "collision_geometry": {
                "external_contact_fixture": dict(fixture)
            },
            "branch_semantics": {
                "external_passive_fixture": dict(fixture),
                "direct_passive_fixture_actuation": False,
            },
        }

        self.assertEqual(
            _validated_external_fixture_body_id(trace, branch="physical"),
            "harness_wood_rod_fixture",
        )
        self.assertIsNone(
            _validated_external_fixture_body_id(trace, branch="reduced")
        )
        trace["collision_geometry"]["external_contact_fixture"][
            "supported"
        ] = False
        with self.assertRaises(MechanismValidationError) as caught:
            _validated_external_fixture_body_id(trace, branch="physical")
        self.assertEqual(
            caught.exception.code, "TRACE_EXTERNAL_FIXTURE_INVALID"
        )

    def test_bounded_json_accepts_utf8_bom_like_intake(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "document.json"
            path.write_bytes(b"\xef\xbb\xbf" + b'{"ok": true}\n')
            self.assertEqual(read_json(path), {"ok": True})

    def test_full_cadence_trace_reader_uses_separate_bounded_budget(self) -> None:
        self.assertGreater(MAX_TRACE_JSON_BYTES, MAX_JSON_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "full-cadence.trace.json"
            path.write_text(
                '{"samples":[],"padding":"'
                + ("x" * (MAX_JSON_BYTES + 1))
                + '"}',
                encoding="ascii",
            )
            self.assertGreater(path.stat().st_size, MAX_JSON_BYTES)
            self.assertLess(path.stat().st_size, MAX_TRACE_JSON_BYTES)
            with self.assertRaises(JsonDocumentError):
                read_json(path)
            trace = read_trace_json(path)
        self.assertEqual(trace["samples"], [])

    def test_full_cadence_trace_writer_streams_without_json_dumps(self) -> None:
        payload = {
            "status": "COMPLETED",
            "samples": [
                {"step": index, "position": [index / 10.0, 0.0, 0.0]}
                for index in range(100)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "streamed.trace.json"
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline."
                "mechanism.io.json.dumps",
                side_effect=AssertionError("monolithic JSON encoding used"),
            ):
                write_trace_json_atomic(path, payload)
            persisted = read_trace_json(path)
        self.assertEqual(persisted, payload)

    def test_streamed_trace_limit_preserves_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "bounded.trace.json"
            previous = b'{"status":"PREVIOUS"}\n'
            path.write_bytes(previous)
            with (
                patch(
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "mechanism.io.MAX_TRACE_JSON_BYTES",
                    64,
                ),
                self.assertRaises(JsonDocumentError),
            ):
                write_trace_json_atomic(path, {"padding": "x" * 128})

            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(root.iterdir()), [path])

    def test_source_tree_hash_changes_with_validator_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "worker.py").write_text("value = 1\n", encoding="ascii")
            original = source_tree_sha256(root)
            (root / "worker.py").write_text("value = 2\n", encoding="ascii")
            changed = source_tree_sha256(root)
        self.assertEqual(len(original), 64)
        self.assertNotEqual(original, changed)

class MechanismDiscoveryTests(unittest.TestCase):
    def test_registered_spec_accepts_common_family_spec_schema_metadata(self) -> None:
        parsed = SimpleNamespace(
            to_dict=lambda: {"mechanism_kind": "installed_kind", "bodies": []}
        )
        plugin = SimpleNamespace(extra_family_spec_fields=frozenset())

        _assert_registered_family_fields(
            {
                "schema_version": "0.1",
                "mechanism_kind": "installed_kind",
                "bodies": [],
            },
            parsed,
            plugin,
        )

    def test_registered_spec_rejects_unsupported_family_spec_schema(self) -> None:
        parsed = SimpleNamespace(
            to_dict=lambda: {"mechanism_kind": "installed_kind", "bodies": []}
        )
        plugin = SimpleNamespace(extra_family_spec_fields=frozenset())

        with self.assertRaises(MechanismValidationError) as caught:
            _assert_registered_family_fields(
                {
                    "schema_version": "9.9",
                    "mechanism_kind": "installed_kind",
                    "bodies": [],
                },
                parsed,
                plugin,
            )

        self.assertEqual(
            caught.exception.code,
            "MECHANISM_FAMILY_SPEC_SCHEMA_UNSUPPORTED",
        )

    def test_missing_reduced_claim_axis_is_owned_by_design_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            capsule_path = package / "mechanism_capsules" / "linear_stage.json"
            capsule = read_json(capsule_path)
            capsule["reduced_model_claim"].pop("axis_abs")
            _write_json(capsule_path, capsule)
            manifest_path = package / "package.json"
            manifest = read_json(manifest_path)
            for artifact in manifest["artifacts"]:
                if artifact["path"] == "mechanism_capsules/linear_stage.json":
                    artifact["sha256"] = _sha256(capsule_path)
            _write_json(manifest_path, manifest)

            with self.assertRaises(MechanismValidationError) as caught:
                discover_mechanisms(package, report)

        self.assertEqual(caught.exception.code, "REDUCED_CLAIM_AXIS_INVALID")
        self.assertEqual(
            caught.exception.failure_envelope["owner"], "design_package"
        )

    def test_registered_spec_drops_legacy_project_adapter_pointer(self) -> None:
        parsed = SimpleNamespace(
            to_dict=lambda: {
                "mechanism_kind": "installed_kind",
                "bodies": [],
            }
        )
        plugin = SimpleNamespace(
            plugin_id="installed.discovery_test_v1",
            extra_family_spec_fields=frozenset(),
        )

        serialized = _serialized_family_spec(
            {
                "mechanism_kind": "installed_kind",
                "project_adapter_file": "mechanism_templates/legacy.py",
                "bodies": [],
            },
            parsed,
            plugin,
        )

        self.assertNotIn("project_adapter_file", serialized)

    def test_compound_certificate_tolerates_task_specific_comparison_metrics(
        self,
    ) -> None:
        plan = SimpleNamespace(family="compound")
        load_case = SimpleNamespace(
            load_case_id="task-roundtrip",
            purpose="nominal",
            moving_mass_kg=0.2,
            phases=(),
        )
        comparisons = (
            {
                "physical_metrics": {
                    "peak_output_angle_rad": 0.25,
                    "terminal_output_angle_rad": 0.20,
                }
            },
        )

        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "certificate._mechanism_plugin",
            return_value=None,
        ):
            envelope = _load_envelope(plan, (load_case,), comparisons)

        self.assertIsNone(envelope["generalized_output_angle_rad"])
        self.assertIsNone(envelope["max_observed_joint_drift_m"])
        self.assertIsNone(envelope["max_observed_off_plane_rotation_rad"])

    def test_project_adapter_requires_exact_plan_approval_and_then_executes(
        self,
    ) -> None:
        source = """\
from types import SimpleNamespace
from simulation_ready_cad_workflow.validation_pipeline.mechanism.family_plugins import MechanismAdapter, GeneralizedCoordinateSpec

class ParsedSpec:
    bodies = (
        SimpleNamespace(fixed=True, group_ids=("fixed_group",)),
        SimpleNamespace(fixed=False, group_ids=("moving_group",)),
    )
    def to_dict(self):
        return {"bodies": [
            {"body_id": "fixed", "fixed": True, "group_ids": ["fixed_group"]},
            {"body_id": "moving", "fixed": False, "group_ids": ["moving_group"]},
        ]}

class ProjectAdapter(MechanismAdapter):
    plugin_id = "project.discovery_test_v1"
    families = frozenset({"compound"})
    coordinate = GeneralizedCoordinateSpec("angle", "speed", "rad", "rad/s", "torque", "N*m")
    worker_backend = "project_discovery_test_v1"
    whole_machine_backend = "project_discovery_test_v1"
    wrench_policy_profiles = frozenset({"task_input", "transmission"})
    def load_cases(self, plan): return ()
    def expected_reduced_ideal_links(self, plan, adapter): return ()
    def parse_family_spec(self, raw): return ParsedSpec()

MECHANISM_ADAPTER = ProjectAdapter()
"""
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            adapter_path = package / "mechanism_templates/project_adapter.py"
            adapter_path.parent.mkdir()
            adapter_path.write_text(source, encoding="utf-8")
            family_path = package / "mechanism_templates/project.family.json"
            _write_json(
                family_path,
                {
                    "mechanism_kind": "project_test",
                    "project_adapter_file": (
                        "mechanism_templates/project_adapter.py"
                    ),
                    "bodies": [],
                },
            )
            capsule_path = package / "mechanism_capsules/linear_stage.json"
            capsule = read_json(capsule_path)
            capsule["reduced_model_claim"]["family"] = "compound"
            capsule["reduced_model_claim"]["family_spec_file"] = (
                "mechanism_templates/project.family.json"
            )
            _write_json(capsule_path, capsule)
            manifest_path = package / "package.json"
            manifest = read_json(manifest_path)
            for item in manifest["artifacts"]:
                if item["path"] == "mechanism_capsules/linear_stage.json":
                    item["sha256"] = _sha256(capsule_path)
            for relative in (
                "mechanism_templates/project_adapter.py",
                "mechanism_templates/project.family.json",
            ):
                manifest["artifacts"].append(
                    {"path": relative, "sha256": _sha256(package / relative)}
                )
            _write_json(manifest_path, manifest)
            intake = read_json(report)
            intake["capabilities"] = {
                "can_enter_physical_preflight": True,
                "can_attempt_reduced_certification": False,
                "physical_preflight_blockers": [],
                "capsules_requiring_contact_model_only": ["linear_stage"],
            }
            _write_json(report, intake)
            approved = {
                "mechanism_templates/project_adapter.py": _sha256(
                    adapter_path
                )
            }
            try:
                with self.assertRaises(MechanismValidationError) as caught:
                    discover_mechanisms(package, report)
                self.assertEqual(
                    caught.exception.code,
                    "INTAKE_REDUCED_CAPABILITY_MISSING",
                )
                with self.assertRaises(MechanismValidationError) as caught:
                    discover_mechanisms(
                        package,
                        report,
                        approved_project_adapters={
                            "mechanism_templates/project_adapter.py": "0" * 64
                        },
                    )
                self.assertEqual(
                    caught.exception.code,
                    "PROJECT_MECHANISM_ADAPTER_NOT_APPROVED",
                )
                plan = discover_mechanisms(
                    package,
                    report,
                    approved_project_adapters=approved,
                )[0]
                plugin, _adapter = mechanism_adapter_for_plan(plan)
                self.assertEqual(
                    plugin.plugin_id, "project.discovery_test_v1"
                )
                self.assertEqual(
                    plan.adapter_spec["family_spec"]["project_adapter_file"],
                    "mechanism_templates/project_adapter.py",
                )
                intake["capabilities"]["can_attempt_reduced_certification"] = True
                intake["capabilities"][
                    "capsules_requiring_contact_model_only"
                ] = []
                _write_json(report, intake)
                parsed = SimpleNamespace(
                    bodies=(
                        SimpleNamespace(fixed=True, group_ids=("fixed_group",)),
                        SimpleNamespace(fixed=False, group_ids=("moving_group",)),
                    ),
                    to_dict=lambda: {
                        "bodies": [
                            {
                                "body_id": "fixed",
                                "fixed": True,
                                "group_ids": ["fixed_group"],
                            },
                            {
                                "body_id": "moving",
                                "fixed": False,
                                "group_ids": ["moving_group"],
                            },
                        ]
                    },
                )
                parsed_inputs: list[dict[str, object]] = []

                def parse_installed(raw):
                    parsed_inputs.append(dict(raw))
                    return parsed

                installed = SimpleNamespace(
                    plugin_id="installed.discovery_test_v1",
                    extra_family_spec_fields=frozenset(),
                    parse_family_spec=parse_installed,
                )
                with patch.object(
                    family_plugins,
                    "family_spec_plugin_for_kind",
                    return_value=installed,
                ), patch.object(
                    family_plugins,
                    "plugin_registration_for",
                    return_value={
                        "schema_version": "simulation_ready_cad_installed_plugin_v2",
                        "plugin_id": installed.plugin_id,
                        "status": "ACTIVE",
                        "registration_source": "installed_harness_code",
                    },
                ), patch.object(
                    family_plugins, "load_project_mechanism_adapter"
                ) as legacy_loader:
                    installed_plan = discover_mechanisms(package, report)[0]
                legacy_loader.assert_not_called()
                self.assertNotIn("project_adapter_file", parsed_inputs[0])
                self.assertNotIn(
                    "project_adapter_file",
                    installed_plan.adapter_spec["family_spec"],
                )
            finally:
                family_plugins._PROJECT_ADAPTERS.clear()
                family_plugins._PROJECT_ADAPTER_CACHE.clear()

    def test_discovers_accepted_capsule_and_binds_documents(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plans = discover_mechanisms(package, report)

        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.capsule_id, "linear_stage")
        self.assertEqual(plan.family, "prismatic")
        self.assertEqual(plan.axis_abs, (1.0, 0.0, 0.0))
        self.assertEqual(plan.characteristic_length_m, 0.20)
        self.assertEqual(plan.task_ids, ("transfer_payload",))
        self.assertTrue(plan.dispatchable)
        self.assertEqual(plan.adapter_spec["adapter_id"], _TEST_ADAPTER.adapter_id)
        self.assertEqual(len(plan.package_binding_sha256), 64)

    def test_plain_prismatic_plugin_is_explicit_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plan = discover_mechanisms(package, report)[0]

        plugin, adapter = mechanism_adapter_for_plan(plan)
        inventory = mechanism_adapter_inventory()
        self.assertEqual(plugin.plugin_id, "plain_prismatic_v1")
        self.assertEqual(adapter.adapter_id, _TEST_ADAPTER.adapter_id)
        self.assertEqual(inventory[0]["families"], ["prismatic"])
        self.assertEqual(
            inventory[0]["coordinate"]["position_field"],
            "generalized_progress_m",
        )

    def test_adapter_spec_identity_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plan = discover_mechanisms(package, report)[0]
        bad_spec = dict(plan.adapter_spec)
        bad_spec["adapter_id"] = "different_adapter"
        with self.assertRaises(MechanismValidationError) as caught:
            replace(plan, adapter_spec=bad_spec)
        self.assertEqual(caught.exception.code, "ADAPTER_SPEC_MISMATCH")

    def test_plain_revolute_plugin_uses_angular_units_and_torque(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                base_plan = discover_mechanisms(package, report)[0]
        adapter = get_adapter(DERIVED_PLAIN_REVOLUTE_ADAPTER_ID)
        plan = replace(
            base_plan,
            family="revolute",
            protocol_id="revolute_pivot_hinge",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        plugin, _ = mechanism_adapter_for_plan(plan)
        cases = default_load_cases(plan)
        supports = plugin.physical_supports(plan, adapter)

        self.assertEqual(plugin.plugin_id, "plain_revolute_v1")
        self.assertEqual(plugin.coordinate.position_unit, "rad")
        self.assertEqual(plugin.coordinate.input_wrench_type, "torque")
        self.assertEqual(len(cases), 2)
        self.assertEqual(
            {case.program for case in cases},
            {"zero_input_gravity", "transverse_torque_probe"},
        )
        self.assertTrue(
            all(
                phase.wrench_kind in {"none", "torque"}
                for case in cases for phase in case.phases
            )
        )
        self.assertEqual(len(supports), 1)
        self.assertEqual(supports[0].chrono_type, "ExplicitChLoadBodyForceTorque")
        self.assertEqual(supports[0].free_coordinate, "joint_frame_rz")
        self.assertEqual(
            supports[0].translational_stiffness_n_m,
            (110_000.0, 110_000.0, 110_000.0),
        )
        self.assertEqual(supports[0].rotational_stiffness_n_m_rad[2], 0.0)
        self.assertGreater(supports[0].rotational_damping_n_m_s_rad[2], 0.0)
        self.assertFalse(supports[0].to_dict()["ideal_constraint"])

    def test_compound_runtime_accepts_bound_plugin_owned_backend(self) -> None:
        plugin_id = "reviewer.folding_knife_liner_lock_firstwave_v1"
        plugin = SimpleNamespace(
            plugin_id=plugin_id,
            worker_backend=plugin_id,
        )

        self.assertTrue(_compound_worker_backend_matches(plugin))
        self.assertFalse(
            _compound_worker_backend_matches(
                SimpleNamespace(
                    plugin_id=plugin_id,
                    worker_backend="unbound.compound.backend",
                )
            )
        )

    def test_shared_compound_runtime_rejects_unimplemented_multi_input_task(
        self,
    ) -> None:
        execution = SimpleNamespace(
            mechanism_plugin=SimpleNamespace(
                plugin_id="reviewer.multi_input_compound_v1"
            ),
            load_case=SimpleNamespace(program="task_multi_input_cycle"),
            plan=SimpleNamespace(
                task_channel_registry={
                    "input_body_ids": ["blade_body", "release_body"]
                }
            ),
        )
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "task_plugins.claimed_reviewer_task_programs",
            return_value=frozenset({"task_multi_input_cycle"}),
        ):
            with self.assertRaises(MechanismValidationError) as caught:
                _require_compound_task_actuation_backend(execution)

        self.assertEqual(
            caught.exception.code,
            "REVIEWER_MECHANISM_ADAPTER_MULTI_INPUT_TASK_RUNTIME_REQUIRED",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            FailureOwner.REVIEWER_MECHANISM_PLUGIN.value,
        )

    def test_compound_plugin_owns_multibody_topology_and_loads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                base_plan = discover_mechanisms(package, report)[0]

        bodies = (
            CompoundBodySpec("ground", ("group_fixed",), True, 5.0, (0.0, 0.0, 0.25), (0.3, 0.2, 0.5)),
            CompoundBodySpec("pedal", ("group_pedal",), False, 0.35, (0.0, 0.0, 0.05), (0.22, 0.04, 0.03)),
            CompoundBodySpec("rod", ("group_rod",), False, 0.18, (0.08, 0.0, 0.25), (0.03, 0.03, 0.38)),
            CompoundBodySpec("lid", ("group_lid",), False, 0.55, (0.0, 0.0, 0.52), (0.28, 0.22, 0.035)),
        )
        joints = (
            CompoundRevoluteJointSpec(
                "pedal_ground",
                "ground",
                "pedal",
                (-0.08, 0.0, 0.06),
                (0.0, 1.0, 0.0),
                free_axis_damping_n_m_s_rad=0.4,
                limit_stiffness_n_m_rad=12_000.0,
                limit_damping_n_m_s_rad=80.0,
                limit_approach_zone_rad=0.12,
                limit_approach_damping_n_m_s_rad=18.0,
            ),
            CompoundRevoluteJointSpec("pedal_rod", "pedal", "rod", (0.10, 0.0, 0.05), (0.0, 1.0, 0.0)),
            CompoundRevoluteJointSpec("rod_lid", "rod", "lid", (0.10, 0.0, 0.50), (0.0, 1.0, 0.0)),
            CompoundRevoluteJointSpec("lid_ground", "ground", "lid", (-0.10, 0.0, 0.50), (0.0, 1.0, 0.0)),
        )
        linkage = CompoundLinkageSpec(
            linkage_id="pedal_lid_linkage",
            bodies=bodies,
            revolute_joints=joints,
            input_body_id="pedal",
            output_body_id="lid",
            input_joint_id="pedal_ground",
            output_joint_id="lid_ground",
            input_point_abs_m=(0.10, 0.0, 0.04),
            input_force_direction_abs=(0.0, 0.0, -1.0),
            output_observable_point_abs_m=(0.12, 0.0, 0.52),
            home_output_angle_rad=0.0,
            outbound_output_angle_rad=1.25,
            nominal_input_force_n=55.0,
            return_input_force_n=4.0,
            max_force_phase_s=1.6,
            max_return_phase_s=2.0,
            settle_phase_s=0.12,
            output_angle_tolerance_rad=0.02,
            max_joint_drift_m=0.001,
            max_off_plane_rotation_rad=0.04,
        )
        adapter = replace(
            get_adapter(DERIVED_COMPOUND_ADAPTER_ID),
            family_spec=linkage.to_dict(),
        )
        plan = replace(
            base_plan,
            family="compound",
            protocol_id="revolute_linkage",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        plugin, restored_adapter = mechanism_adapter_for_plan(plan)
        restored_linkage = compound_linkage_from_dict(restored_adapter.family_spec)
        cases = default_load_cases(plan)

        self.assertEqual(plugin.plugin_id, "revolute_linkage_v1")
        self.assertEqual(plugin.coordinate.component_count, 2)
        self.assertEqual(
            plugin.coordinate.input_wrench_type,
            "declared_force_or_torque",
        )
        self.assertEqual(restored_linkage.loop_count, 1)
        self.assertEqual(
            restored_linkage.revolute_joints[0].free_axis_damping_n_m_s_rad,
            0.4,
        )
        self.assertEqual(
            restored_linkage.revolute_joints[0].limit_stiffness_n_m_rad,
            12_000.0,
        )
        self.assertEqual(
            restored_linkage.revolute_joints[0].limit_approach_zone_rad,
            0.12,
        )
        self.assertEqual(
            restored_linkage.revolute_joints[0].limit_approach_damping_n_m_s_rad,
            18.0,
        )
        self.assertLess(
            _soft_limit_torque(
                joint_angle_rad=0.95,
                free_speed_rad_s=2.0,
                minimum_rad=0.0,
                maximum_rad=1.0,
                stiffness_n_m_rad=650.0,
                damping_n_m_s_rad=5.0,
                approach_zone_rad=0.10,
                approach_damping_n_m_s_rad=20.0,
            ),
            0.0,
        )
        self.assertEqual(
            _soft_limit_torque(
                joint_angle_rad=0.95,
                free_speed_rad_s=-2.0,
                minimum_rad=0.0,
                maximum_rad=1.0,
                stiffness_n_m_rad=650.0,
                damping_n_m_s_rad=5.0,
                approach_zone_rad=0.10,
                approach_damping_n_m_s_rad=20.0,
            ),
            0.0,
        )
        invalid_linkage = linkage.to_dict()
        invalid_linkage["revolute_joints"][0]["limit_approach_zone_rad"] = 0.0
        invalid_linkage["revolute_joints"][0][
            "limit_approach_damping_n_m_s_rad"
        ] = 1.0
        with self.assertRaises(MechanismValidationError) as caught:
            compound_linkage_from_dict(invalid_linkage)
        self.assertEqual(
            caught.exception.code,
            "COMPOUND_JOINT_PHYSICAL_PARAMETER_INVALID",
        )
        self.assertEqual(len(plugin.expected_reduced_ideal_links(plan, restored_adapter)), 4)
        self.assertEqual(len(cases), 6)
        self.assertEqual(cases[-1].program, "task_output_angle_roundtrip")
        self.assertEqual(cases[-1].phases[0].target_position_si, 1.25)
        self.assertTrue(all(case.family == "compound" for case in cases))

        torque_linkage = replace(
            linkage,
            input_actuation_kind="joint_torque",
            nominal_input_torque_nm=3.2,
            return_input_torque_nm=2.4,
        )
        torque_adapter = replace(
            adapter,
            family_spec=torque_linkage.to_dict(),
        )
        torque_plan = replace(
            plan,
            adapter_spec=adapter_to_dict(torque_adapter),
        )
        torque_cases = default_load_cases(torque_plan)
        for case in torque_cases:
            loaded = [phase for phase in case.phases if phase.wrench_kind != "none"]
            if case.load_case_id == "transverse_linkage_probe":
                self.assertEqual([phase.wrench_kind for phase in loaded], ["force"])
            elif case.load_case_id == "zero_input_gravity":
                self.assertEqual(loaded, [])
            else:
                self.assertEqual(
                    [phase.wrench_kind for phase in loaded],
                    ["torque", "torque"],
                )
        self.assertEqual(
            torque_cases[0].phases[0].direction_abs,
            joints[0].axis_abs,
        )
        self.assertEqual(
            torque_cases[0].phases[2].direction_abs,
            tuple(-value for value in joints[0].axis_abs),
        )

    def test_revolute_task_roundtrip_keeps_angular_target_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                base_plan = discover_mechanisms(package, report)[0]
        adapter = replace(
            get_adapter(DERIVED_PLAIN_REVOLUTE_ADAPTER_ID),
            whole_machine_angular_task=WholeMachineAngularTaskSpec(
                outbound_task_id="open_drawer",
                return_task_id="close_drawer",
                outbound_angle_rad=1.2,
                home_angle_rad=0.0,
                moving_mass_kg=0.4,
                outbound_torque_nm=0.08,
                return_torque_nm=0.09,
                max_torque_phase_s=1.5,
                settle_phase_s=0.15,
            ),
        )
        plan = replace(
            base_plan,
            family="revolute",
            protocol_id="revolute_pivot_hinge",
            adapter_id=adapter.adapter_id,
            adapter_spec=adapter_to_dict(adapter),
        )
        cases = default_load_cases(plan)
        task_case = cases[-1]

        self.assertEqual(task_case.program, "task_angle_roundtrip")
        self.assertEqual(task_case.phases[0].target_position_si, 1.2)
        self.assertIsNone(task_case.phases[0].target_progress_m)
        self.assertEqual(task_case.phases[0].target_coordinate, 1.2)

    def test_schema_02_single_steering_task_derives_public_angle(self) -> None:
        task = _angular_task_spec(
            SimpleNamespace(
                task_ids=("steering_sweep",),
                moving_extents_m=(0.65, 0.55, 0.8),
            ),
            {
                "schema_version": "0.2",
                "tasks": [
                    {
                        "task_id": "steering_sweep",
                        "description": "Turn the handlebar through the public target.",
                        "acceptance_requirements": [
                            {
                                "metric_id": "continuous_rotation_rad",
                                "target": {
                                    "comparator": "at_least",
                                    "value": 0.35,
                                    "unit": "rad",
                                },
                            }
                        ],
                    }
                ],
            },
        )

        self.assertEqual(task.outbound_task_id, "steering_sweep")
        self.assertEqual(task.return_task_id, "steering_sweep")
        self.assertEqual(task.outbound_angle_rad, 0.35)
        self.assertEqual(task.home_angle_rad, 0.0)

    def test_schema_02_single_steering_task_rejects_unusable_target(self) -> None:
        plan = SimpleNamespace(
            task_ids=("steering_sweep",),
            moving_extents_m=(0.65, 0.55, 0.8),
        )
        base = {
            "schema_version": "0.2",
            "tasks": [
                {
                    "task_id": "steering_sweep",
                    "acceptance_requirements": [
                        {
                            "metric_id": "continuous_rotation_rad",
                            "target": {
                                "comparator": "at_most",
                                "value": 0.35,
                                "unit": "deg",
                            },
                        }
                    ],
                }
            ],
        }
        with self.assertRaises(MechanismValidationError) as unusable:
            _angular_task_spec(plan, base)
        self.assertEqual(unusable.exception.code, "DERIVED_TASK_PAIR_REQUIRED")

        base["tasks"][0]["acceptance_requirements"][0]["target"] = {
            "comparator": "at_least",
            "value": math.radians(200.0),
            "unit": "rad",
        }
        with self.assertRaises(MechanismValidationError) as out_of_range:
            _angular_task_spec(plan, base)
        self.assertEqual(
            out_of_range.exception.code,
            "DERIVED_TASK_ANGLE_UNAVAILABLE",
        )

    def test_revolute_whole_machine_plan_roundtrip_preserves_angular_units(self) -> None:
        plan = WholeMachineRevoluteExecutionPlan(
            schema_version="0.1",
            coordinate_kind="rotation",
            package_binding_sha256="a" * 64,
            capsule_id="lid_hinge",
            certificate_file="run/certificate.json",
            certificate_sha256="b" * 64,
            replacement_file="run/replacement.json",
            replacement_sha256="c" * 64,
            adapter_id=DERIVED_PLAIN_REVOLUTE_ADAPTER_ID,
            mechanism_plugin_id="plain_revolute_v1",
            worker_backend="plain_revolute_v1",
            task_ids=("open_lid", "close_lid"),
            axis_abs=(0.0, -1.0, 0.0),
            joint_origin_abs_m=(-0.145, 0.0, 0.15),
            gravity_abs_m_s2=(0.0, 0.0, -9.81),
            body_origin_abs_m=(0.0, 0.0, 0.162),
            input_point_abs_m=(0.1475, 0.0, 0.1735),
            moving_extents_m=(0.304, 0.17, 0.045),
            moving_mass_kg=0.45,
            outbound_angle_rad=1.4835,
            home_angle_rad=0.0,
            max_control_torque_nm=0.82,
        )

        restored = WholeMachineRevoluteExecutionPlan.from_dict(plan.to_dict())

        self.assertEqual(restored.kinematic_pivot_tolerance_m, 1e-5)
        self.assertEqual(restored.kinematic_off_axis_tolerance_rad, 1e-6)
        self.assertEqual(restored.coordinate_kind, "rotation")
        self.assertEqual(restored.worker_backend, "plain_revolute_v1")
        self.assertAlmostEqual(restored.outbound_angle_rad, 1.4835)
        self.assertAlmostEqual(restored.max_control_torque_nm, 0.82)
        self.assertEqual(restored.axis_abs, (0.0, -1.0, 0.0))

    def test_physical_preflight_gate_blocks_dispatch_before_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(
                Path(temporary), capabilities_ready=False
            )
            with self.assertRaises(MechanismValidationError) as caught:
                discover_mechanisms(package, report)

        self.assertEqual(caught.exception.code, "INTAKE_PREFLIGHT_BLOCKED")

    def test_artifact_mutation_after_intake_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            (package / "source_components" / "rail.step").write_text(
                "mutated\n", encoding="ascii"
            )
            with self.assertRaises(MechanismValidationError) as caught:
                discover_mechanisms(package, report)

        self.assertEqual(caught.exception.code, "ARTIFACT_CHANGED_AFTER_INTAKE")

    def test_validator_load_suite_covers_nominal_and_adversarial_cases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package, report = _build_package(Path(temporary))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plan = discover_mechanisms(package, report)[0]

        cases = default_load_cases(plan)
        self.assertEqual(len(cases), 5)
        self.assertEqual(
            {case.program for case in cases},
            {"bidirectional_axis", "zero_input_gravity", "transverse_probe"},
        )
        self.assertIn("held_out", {case.purpose for case in cases})
        self.assertTrue(all(case.required_for_certificate for case in cases))

    def test_missing_product_adapter_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, report = _build_package(root)
            plans = discover_mechanisms(package, report)
            self.assertFalse(plans[0].dispatchable)
            with self.assertRaises(MechanismValidationError) as caught:
                build_dispatch_bundle(
                    plans,
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )

        self.assertEqual(caught.exception.code, "MECHANISM_ADAPTER_UNAVAILABLE")

    def test_dispatch_skips_ideal_dynamics_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, report = _build_package(root)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plans = discover_mechanisms(package, report)
            test_adapter = _with_test_fidelity(
                root, _TEST_ADAPTER, plans[0].package_binding_sha256
            )
            plans = tuple(
                replace(plan, adapter_spec=adapter_to_dict(test_adapter))
                for plan in plans
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=test_adapter,
            ):
                bundle = build_dispatch_bundle(
                    plans,
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            manifest = QueueManifest.from_file(root / bundle.queue_manifest)
            physical_manifest = QueueManifest.from_file(root / bundle.physical_queue_manifest)
            ideal_manifest = bundle.ideal_dynamics_queue_manifest
            request = json.loads(
                (root / bundle.request_files[0]).read_text(encoding="utf-8")
            )

        self.assertEqual(bundle.job_count, 5)
        self.assertEqual(len(physical_manifest.jobs), 5)
        self.assertIsNone(ideal_manifest)
        self.assertTrue(
            all(len(job.expected_outputs) == 5 for job in physical_manifest.jobs)
        )
        self.assertEqual(
            {
                Path(output).name
                for job in physical_manifest.jobs
                for output in job.expected_outputs
            },
            {
                "physical.trace.json",
                "ideal_reference.trace.json",
                *(f"{load.load_case_id}.json" for load in default_load_cases(plans[0])),
            },
        )
        self.assertEqual(request["mechanism_plugin_id"], "plain_prismatic_v1")
        self.assertEqual(len(manifest.jobs), 5)
        self.assertEqual(
            manifest.resource_policy.min_available_memory_gib, 1.5
        )
        self.assertEqual(manifest.queue_policy.stop_on_failure, True)
        self.assertEqual(manifest.queue_policy.max_timeout_resume_attempts, 3)
        self.assertEqual(
            manifest.queue_policy.max_parallel_jobs,
            len(manifest.jobs),
        )
        self.assertTrue(
            all(job.timeout_seconds == 360.0 for job in manifest.jobs)
        )
        self.assertTrue(all(job.cpu_threads == 2 for job in manifest.jobs))
        self.assertTrue(
            all(
                job.estimated_peak_memory_gib == 1.0
                and job.max_process_tree_memory_gib == 1.5
                for job in manifest.jobs
            )
        )
        self.assertTrue(
            all(
                job.command[:3]
                == (
                    "/opt/test-physics/bin/python",
                    "-m",
                    (
                        "simulation_ready_cad_workflow.validation_pipeline."
                        "mechanism.lane_worker"
                    ),
                )
                for job in manifest.jobs
            )
        )
        source_hashes = {
            job.environment.get("SIMREADY_MECHANISM_SOURCE_SHA256")
            for job in manifest.jobs
        }
        self.assertEqual(len(source_hashes), 1)
        self.assertEqual(
            next(iter(source_hashes)), lane_execution_source_sha256()
        )
        self.assertTrue(
            all(
                job.environment.get("PYTHONPATH")
                == str(validator_import_root())
                for job in manifest.jobs
            )
        )
        self.assertEqual(
            {Path(job.expected_outputs[0]).name for job in manifest.jobs},
            {"physical.trace.json"},
        )

    def test_initial_functional_geometry_failure_precedes_queue_creation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, report = _build_package(root)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plans = discover_mechanisms(package, report)
            test_adapter = _with_test_fidelity(
                root,
                _TEST_ADAPTER,
                plans[0].package_binding_sha256,
            )
            plans = tuple(
                replace(plan, adapter_spec=adapter_to_dict(test_adapter))
                for plan in plans
            )
            failed_preflight = {
                "schema_version": (
                    "initial_functional_geometry_preflight_v1"
                ),
                "status": "FAIL",
                "passed": False,
                "capsule_id": plans[0].capsule_id,
                "mechanism_kind": "fixture",
                "package_binding_sha256": (
                    plans[0].package_binding_sha256
                ),
                "analytical_interface_count": 1,
                "registered_geometry_gate_count": 0,
                "ungated_analytical_interface_count": 1,
                "initially_nontransmitting_interface_count": 1,
                "initially_nontransmitting_interfaces": [
                    "fixture-interface"
                ],
                "warning_count": 1,
                "failure_reasons": [
                    "analytical_transmission_geometry_gate_missing"
                ],
                "interfaces": [],
                "physical_simulation_started": False,
            }
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=test_adapter,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.build_initial_functional_geometry_preflight",
                return_value=failed_preflight,
            ), self.assertRaises(MechanismValidationError) as caught:
                build_dispatch_bundle(
                    plans,
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )

            report_path = (
                root
                / "run"
                / "results"
                / "initial_functional_geometry_preflight.json"
            )
            recorded = read_json(report_path)
            static_preflight = read_json(
                root
                / "run"
                / "results"
                / "mechanism_static_preflight.json"
            )
            queue_manifest_exists = (
                root / "run" / "queue_manifest.json"
            ).exists()
            requests_exist = (root / "run" / "requests").exists()

        self.assertEqual(
            caught.exception.code,
            "INITIAL_FUNCTIONAL_GEOMETRY_PREFLIGHT_FAILED",
        )
        self.assertFalse(recorded["physical_simulation_started"])
        self.assertEqual(
            recorded["ungated_analytical_interface_count"],
            1,
        )
        self.assertTrue(
            static_preflight["all_independent_checks_executed"]
        )
        self.assertEqual(
            {
                item["check_id"]: item["status"]
                for item in static_preflight["checks"]
            },
            {
                "collider_fidelity": "PASS",
                "initial_functional_geometry": "FAIL",
            },
        )
        self.assertFalse(
            static_preflight["physical_simulation_started"]
        )
        self.assertFalse(queue_manifest_exists)
        self.assertFalse(requests_exist)

    def test_missing_collider_fidelity_does_not_claim_budget_failure(
        self,
    ) -> None:
        with self.assertRaises(MechanismValidationError) as missing_box:
            _static_preflight_failure(
                [
                    {
                        "capsule_id": "clipper",
                        "check_id": "collider_fidelity",
                        "status": "FAIL",
                        "passed": False,
                        "failure_reason": "collider_fidelity_evidence_missing",
                    }
                ],
                capsule_id="clipper",
                collision_geometry=None,
            )
        missing = missing_box.exception
        self.assertEqual(
            missing.code, "COLLIDER_FIDELITY_PREFLIGHT_FAILED"
        )
        self.assertIn("missing", missing.message)
        self.assertNotIn("trusted budgets", missing.message)
        self.assertEqual(missing.failure_envelope["owner"], "design_package")
        self.assertEqual(
            missing.failure_envelope["evidence_validity"], "NOT_APPLICABLE"
        )

        with self.assertRaises(MechanismValidationError) as budget_box:
            _static_preflight_failure(
                [
                    {
                        "capsule_id": "clipper",
                        "check_id": "collider_fidelity",
                        "status": "FAIL",
                        "passed": False,
                        "report": (
                            "run/derived_adapters/clipper/collider_fidelity.json"
                        ),
                    }
                ],
                capsule_id="clipper",
                collision_geometry=None,
            )
        self.assertIn("trusted budgets", budget_box.exception.message)
        self.assertEqual(
            budget_box.exception.failure_envelope["owner"], "design_package"
        )
        self.assertEqual(
            budget_box.exception.failure_envelope["evidence_validity"],
            "VALID_MEASUREMENT",
        )

        with self.assertRaises(MechanismValidationError) as reviewer_box:
            _static_preflight_failure(
                [
                    {
                        "capsule_id": "clipper",
                        "check_id": "collider_fidelity",
                        "status": "FAIL",
                        "passed": False,
                        "failure_reason": (
                            "reviewer_mechanism_adapter_collider_evidence_missing"
                        ),
                    }
                ],
                capsule_id="clipper",
                collision_geometry=None,
            )
        self.assertEqual(
            reviewer_box.exception.code,
            "REVIEWER_MECHANISM_ADAPTER_COLLIDER_EVIDENCE_MISSING",
        )
        self.assertEqual(
            reviewer_box.exception.failure_envelope["owner"],
            "reviewer_mechanism_plugin",
        )
        self.assertEqual(
            reviewer_box.exception.failure_envelope["evidence_validity"],
            "INVALID_MEASUREMENT",
        )
        with self.assertRaises(MechanismValidationError) as combined_box:
            _static_preflight_failure(
                [
                    {
                        "capsule_id": "clipper",
                        "check_id": "collider_fidelity",
                        "status": "FAIL",
                        "passed": False,
                        "failure_reason": (
                            "reviewer_mechanism_adapter_collider_evidence_missing"
                        ),
                        "plugin_id": "reviewer.clipper",
                    },
                    {
                        "capsule_id": "clipper",
                        "check_id": "initial_functional_geometry",
                        "status": "FAIL",
                        "passed": False,
                    },
                ],
                capsule_id="clipper",
                collision_geometry=None,
            )
        self.assertEqual(
            combined_box.exception.code,
            "REVIEWER_MECHANISM_ADAPTER_COLLIDER_EVIDENCE_MISSING",
        )

    def test_home_interpenetration_carries_design_owner(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            _static_preflight_failure(
                [
                    {
                        "capsule_id": "clipper",
                        "check_id": "home_interpenetration",
                        "status": "FAIL",
                        "passed": False,
                        "worst_events": [
                            {
                                "solid_pair": ["bit_shaft/a", "handle/b"],
                                "penetration_depth_m": 0.00038,
                            }
                        ],
                    }
                ],
                capsule_id="clipper",
                collision_geometry=None,
            )

        self.assertEqual(caught.exception.code, "FAMILY_HOME_INTERPENETRATION")
        self.assertEqual(
            caught.exception.failure_envelope["owner"], "design_package"
        )
        self.assertEqual(
            caught.exception.failure_envelope["evidence_validity"],
            "VALID_MEASUREMENT",
        )

    def test_multiple_known_static_failures_remain_design_owned(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            _static_preflight_failure(
                [
                    {"check_id": "rigid_attachment", "passed": False},
                    {"check_id": "finite_linear_capture", "passed": False},
                ],
                capsule_id="clipper",
                collision_geometry=None,
            )

        self.assertEqual(
            caught.exception.code, "MECHANISM_STATIC_PREFLIGHT_FAILED"
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"], "design_package"
        )

    def test_unknown_static_check_remains_unclassified(self) -> None:
        with self.assertRaises(MechanismValidationError) as caught:
            _static_preflight_failure(
                [{"check_id": "future_static_gate", "passed": False}],
                capsule_id="clipper",
                collision_geometry=None,
            )

        self.assertEqual(
            caught.exception.failure_envelope["owner"], "unclassified"
        )
        self.assertEqual(
            caught.exception.failure_envelope["evidence_validity"],
            "NOT_APPLICABLE",
        )

    def test_missing_static_evidence_uses_materializer_owner(self) -> None:
        with self.assertRaises(MechanismValidationError) as installed:
            _static_preflight_failure(
                [
                    {
                        "check_id": "collider_fidelity",
                        "passed": False,
                        "failure_reason": "collider_fidelity_evidence_missing",
                    }
                ],
                capsule_id="clipper",
                collision_geometry=None,
                plugin_owner=FailureOwner.HARNESS_CODE,
                plugin_id="installed.clipper",
            )
        self.assertEqual(
            installed.exception.failure_envelope["owner"], "harness_code"
        )

        with self.assertRaises(MechanismValidationError) as reviewer:
            _static_preflight_failure(
                [
                    {
                        "check_id": "family_static_geometry",
                        "passed": False,
                        "failure_reason": "aggregate_evidence_missing",
                    }
                ],
                capsule_id="clipper",
                collision_geometry=None,
                plugin_owner=FailureOwner.REVIEWER_MECHANISM_PLUGIN,
                plugin_id="reviewer.clipper",
            )
        self.assertEqual(
            reviewer.exception.failure_envelope["owner"],
            "reviewer_mechanism_plugin",
        )

    def test_dispatch_routes_missing_reviewer_fidelity_to_reviewer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, report = _build_package(root)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                plans = discover_mechanisms(package, report)
            reviewer = SimpleNamespace(
                plugin_id="reviewer.missing_fidelity",
                worker_backend="plain_prismatic_v1",
                materialize=(
                    lambda plan, adapter, *, workspace_root, output_dir: adapter
                ),
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=_TEST_ADAPTER,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.mechanism_adapter_for",
                return_value=reviewer,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.plugin_registration_for",
                return_value={
                    "registration_source": "reviewer_mechanism_adapter"
                },
            ), self.assertRaises(MechanismValidationError) as caught:
                build_dispatch_bundle(
                    plans,
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            static_preflight = read_json(
                root / "run" / "results" / "mechanism_static_preflight.json"
            )
        self.assertEqual(
            caught.exception.code,
            "REVIEWER_MECHANISM_ADAPTER_COLLIDER_EVIDENCE_MISSING",
        )
        collider = next(
            item
            for item in static_preflight["checks"]
            if item["check_id"] == "collider_fidelity"
        )
        self.assertEqual(
            collider["failure_reason"],
            "reviewer_mechanism_adapter_collider_evidence_missing",
        )

    def test_reviewer_collider_fidelity_requires_real_package_bound_report(
        self,
    ) -> None:
        binding = "0" * 64
        plugin = SimpleNamespace(plugin_id="reviewer.test")
        plan = SimpleNamespace(package_binding_sha256=binding)
        missing = AdapterSpec(
            adapter_id="reviewer_test",
            family="compound",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
        )
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary).resolve()
            self.assertIsNone(
                _reviewer_collider_fidelity(
                    plugin,
                    plan,
                    missing,
                    workspace_root=workspace,
                    plugin_owner=FailureOwner.REVIEWER_MECHANISM_PLUGIN,
                )
            )
            report = workspace / "run" / "collider_fidelity.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}\n", encoding="utf-8")
            invalid_report = replace(
                missing,
                collider_fidelity={
                    "status": "FAIL",
                    "passed": False,
                    "report": "run/collider_fidelity.json",
                    "report_sha256": sha256_file(report),
                    "package_binding_sha256": binding,
                },
            )
            self.assertIsNone(
                _reviewer_collider_fidelity(
                    plugin,
                    plan,
                    invalid_report,
                    workspace_root=workspace,
                    plugin_owner=FailureOwner.REVIEWER_MECHANISM_PLUGIN,
                )
            )
            report.write_text(
                json.dumps(
                    {
                        "schema_version": "collider_fidelity_v1",
                        "status": "FAIL",
                        "passed": False,
                        "package_binding_sha256": binding,
                        "thresholds": {},
                        "component_count": 1,
                        "rigid_body_count": 1,
                        "components": [{"status": "FAIL", "passed": False}],
                        "rigid_bodies": [{"status": "FAIL", "passed": False}],
                        "failure_reasons": ["collider_fidelity_budget_exceeded"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            failed_budget = replace(
                invalid_report,
                collider_fidelity={
                    **invalid_report.collider_fidelity,
                    "report_sha256": sha256_file(report),
                },
            )
            fidelity = _reviewer_collider_fidelity(
                plugin,
                plan,
                failed_budget,
                workspace_root=workspace,
                plugin_owner=FailureOwner.REVIEWER_MECHANISM_PLUGIN,
            )
        self.assertIsNotNone(fidelity)
        self.assertFalse(fidelity["passed"])

    def test_untyped_reviewer_simulation_error_stays_with_reviewer(self) -> None:
        def fail_simulation(execution, workspace):
            raise RuntimeError("plugin bug")

        execution = SimpleNamespace(
            mechanism_plugin=SimpleNamespace(
                plugin_id="reviewer.test",
                validate_execution_geometry_proofs=lambda execution: None,
                simulate=fail_simulation,
            )
        )
        registration = {
            "plugin_id": "reviewer.test",
            "registration_source": "reviewer_mechanism_adapter",
        }
        with patch.object(
            MechanismAdapter,
            "validate_execution_geometry_proofs",
            return_value=None,
        ), patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism.worker."
            "plugin_registration_for",
            return_value=registration,
        ), self.assertRaises(MechanismValidationError) as caught:
            _simulate(execution, Path("."))
        self.assertEqual(
            caught.exception.code,
            "MECHANISM_ADAPTER_SIMULATE_FAILED",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            "reviewer_mechanism_plugin",
        )

    def test_invalid_reviewer_simulation_result_stays_with_reviewer(self) -> None:
        plugin = SimpleNamespace(
            plugin_id="reviewer.test",
            validate_execution_geometry_proofs=lambda execution: None,
            simulate=lambda execution, workspace: {"status": "COMPLETED"},
        )
        execution = SimpleNamespace(
            mechanism_plugin=plugin,
            artifact_branch="physical",
            adapter=SimpleNamespace(adapter_id="reviewer-adapter"),
            plan=SimpleNamespace(
                package_binding_sha256="package-binding",
                capsule_id="reviewer-capsule",
                to_dict=lambda: {"capsule_id": "reviewer-capsule"},
            ),
            load_case=SimpleNamespace(
                load_case_id="reviewer-load",
                to_dict=lambda: {"load_case_id": "reviewer-load"},
            ),
        )
        registration = {
            "plugin_id": "reviewer.test",
            "registration_source": "reviewer_mechanism_adapter",
        }

        with patch.object(
            MechanismAdapter,
            "validate_execution_geometry_proofs",
            return_value=None,
        ), patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism.worker."
            "plugin_registration_for",
            return_value=registration,
        ), self.assertRaises(MechanismValidationError) as caught:
            _simulate(execution, Path("."))

        self.assertEqual(
            caught.exception.code,
            "MECHANISM_ADAPTER_SIMULATE_RESULT_INVALID",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            "reviewer_mechanism_plugin",
        )

    def test_reviewer_mechanism_load_cases_have_typed_plugin_failures(self) -> None:
        plan = SimpleNamespace(capsule_id="reviewer_capsule")
        adapter = AdapterSpec(
            adapter_id="reviewer_test",
            family="compound",
            required_source_hashes=frozenset(),
            supported_programs=frozenset(),
        )

        def fail_load_cases(plan):
            raise RuntimeError("plugin bug")

        failed = SimpleNamespace(
            plugin_id="reviewer.test",
            load_cases=fail_load_cases,
        )
        registration = {
            "plugin_id": "reviewer.test",
            "registration_source": "reviewer_mechanism_adapter",
        }
        with self.assertRaises(MechanismValidationError) as failed_box:
            _mechanism_load_cases(
                failed,
                plan,
                adapter,
                plugin_registration=registration,
            )
        self.assertEqual(
            failed_box.exception.code,
            "MECHANISM_ADAPTER_LOAD_CASE_FAILED",
        )
        self.assertEqual(
            failed_box.exception.failure_envelope["owner"],
            "reviewer_mechanism_plugin",
        )

        invalid = SimpleNamespace(
            plugin_id="reviewer.test",
            load_cases=lambda plan: None,
        )
        with self.assertRaises(MechanismValidationError) as invalid_box:
            _mechanism_load_cases(
                invalid,
                plan,
                adapter,
                plugin_registration=registration,
            )
        self.assertEqual(
            invalid_box.exception.code,
            "MECHANISM_ADAPTER_LOAD_CASE_INVALID",
        )

        unsupported = SimpleNamespace(
            plugin_id="reviewer.test",
            load_cases=lambda plan: (
                LoadCase(
                    load_case_id="unsupported",
                    family="compound",
                    program="task_unclaimed",
                    purpose="nominal",
                    phases=(
                        LoadPhase(
                            "powered",
                            "torque",
                            (0.0, 0.0, 1.0),
                            1.0,
                            0.1,
                        ),
                    ),
                    moving_mass_kg=0.2,
                ),
            ),
        )
        with self.assertRaises(MechanismValidationError) as unsupported_box:
            _mechanism_load_cases(
                unsupported,
                plan,
                adapter,
                plugin_registration=registration,
            )
        self.assertEqual(
            unsupported_box.exception.code,
            "MECHANISM_ADAPTER_LOAD_CASE_PROGRAM_UNSUPPORTED",
        )

        multi_input_plan = SimpleNamespace(
            capsule_id="reviewer_capsule",
            family="compound",
            task_channel_registry={
                "input_body_ids": ["blade_body", "release_body"]
            },
        )
        multi_input = SimpleNamespace(
            plugin_id="reviewer.multi_input",
            worker_backend="reviewer.multi_input",
            load_cases=lambda plan: (
                LoadCase(
                    load_case_id="task_multi_input",
                    family="compound",
                    program="task_multi_input_cycle",
                    purpose="nominal",
                    phases=(
                        LoadPhase(
                            "powered",
                            "torque",
                            (0.0, 0.0, 1.0),
                            1.0,
                            0.1,
                        ),
                    ),
                    moving_mass_kg=0.2,
                ),
            ),
        )
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "dispatch.claimed_reviewer_task_programs",
            return_value=frozenset({"task_multi_input_cycle"}),
        ):
            with self.assertRaises(MechanismValidationError) as multi_box:
                _mechanism_load_cases(
                    multi_input,
                    multi_input_plan,
                    adapter,
                    plugin_registration={
                        "plugin_id": "reviewer.multi_input",
                        "registration_source": (
                            "reviewer_mechanism_adapter"
                        ),
                    },
                )
        self.assertEqual(
            multi_box.exception.code,
            "REVIEWER_MECHANISM_ADAPTER_MULTI_INPUT_TASK_RUNTIME_REQUIRED",
        )
        self.assertEqual(
            multi_box.exception.failure_envelope["owner"],
            FailureOwner.REVIEWER_MECHANISM_PLUGIN.value,
        )

    def test_final_static_preflight_replaces_provisional_family_binding(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_path = root / "run" / "results" / (
                "mechanism_static_preflight.json"
            )
            report = {
                "schema_version": "mechanism_static_preflight_v1",
                "status": "PASS",
                "passed": True,
                "package_binding_sha256": "package-binding",
                "checks": [
                    {
                        "check_id": "home_interpenetration",
                        "status": "PASS",
                        "passed": True,
                    }
                ],
            }
            write_json_atomic(report_path, report)
            adapter = replace(
                _TEST_ADAPTER,
                family_spec={
                    "collision_geometry": {
                        "schema": "family_step_collision_v1",
                        "package_binding_sha256": "package-binding",
                        "static_geometry_preflight": {
                            "status": "FAIL",
                            "passed": False,
                            "report": "provisional.json",
                        },
                    }
                },
            )

            rebound = _bind_final_static_preflight(
                adapter,
                workspace_root=root,
                report_path=report_path,
                report=report,
                package_binding_sha256="package-binding",
            )
            binding = rebound.family_spec["collision_geometry"][
                "static_geometry_preflight"
            ]

        self.assertEqual(binding["status"], "PASS")
        self.assertTrue(binding["passed"])
        self.assertEqual(
            binding["report"],
            "run/results/mechanism_static_preflight.json",
        )
        self.assertEqual(
            binding["report_sha256"],
            hashlib.sha256(
                json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=False,
                ).encode("utf-8")
                + b"\n"
            ).hexdigest(),
        )
        self.assertEqual(
            adapter.family_spec["collision_geometry"][
                "static_geometry_preflight"
            ]["status"],
            "FAIL",
        )

    def test_task_outside_contact_scope_stays_out_of_local_queue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, report = _build_package(root)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                base_plan = discover_mechanisms(package, report)[0]
            adapter = _with_test_fidelity(
                root,
                get_adapter("mcmaster_6725k41_8438k4_v1"),
                base_plan.package_binding_sha256,
            )
            plan = replace(
                base_plan,
                adapter_id="mcmaster_6725k41_8438k4_v1",
                adapter_spec=adapter_to_dict(adapter),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
            )
            load_cases = default_load_cases(plan)
            bundle = build_dispatch_bundle(
                (plan,),
                workspace_root=root,
                output_dir=root / "run",
                pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
            )

        self.assertEqual(len(load_cases), 5)
        self.assertEqual(bundle.job_count, 5)
        self.assertNotIn(
            "task_range_roundtrip", {item.program for item in load_cases}
        )
        self.assertFalse(
            (
                root
                / "run"
                / "requests"
                / plan.capsule_id
                / "task_envelope_roundtrip"
            ).exists()
        )

    def test_explicit_dynamic_claim_dispatches_optional_branch(self) -> None:
        wide_adapter = _wide_mcm_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package, report = _build_package(root)
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "discovery.select_adapter",
                return_value=_TEST_ADAPTER,
            ):
                base_plan = discover_mechanisms(package, report)[0]
            wide_adapter = _with_test_fidelity(
                root, wide_adapter, base_plan.package_binding_sha256
            )
            plan = replace(
                base_plan,
                adapter_id=wide_adapter.adapter_id,
                adapter_spec=adapter_to_dict(wide_adapter),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
                acceptance_contract={
                    "status": "PASS",
                    "certificate_authority": True,
                    "requirements": [
                        {
                            "required_branches": [
                                "physical",
                                "ideal_dynamics",
                            ]
                        }
                    ],
                },
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "loads.get_adapter",
                return_value=wide_adapter,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=wide_adapter,
            ):
                load_cases = default_load_cases(plan)
                bundle = build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            ideal_request = json.loads(
                (
                    root
                    / "run"
                    / "requests"
                    / plan.capsule_id
                    / "task_envelope_roundtrip"
                    / "ideal_dynamics.json"
                ).read_text(encoding="utf-8")
            )

        self.assertEqual(len(load_cases), 6)
        self.assertEqual(
            bundle.job_count,
            len(load_cases),
            "one queue job owns the complete physical/ideal pipeline for a load",
        )
        self.assertEqual(load_cases[-1].program, "task_range_roundtrip")
        self.assertEqual(load_cases[-1].phases[0].target_progress_m, 0.04)
        self.assertTrue(
            ideal_request["reference_trace_file"].endswith(
                "task_envelope_roundtrip/physical.trace.json"
            )
        )


class MechanismComparisonTests(unittest.TestCase):
    def test_bound_reviewer_task_program_defers_to_task_acceptance(self) -> None:
        load_case = LoadCase(
            load_case_id="task_plugin_case",
            family="compound",
            program="task_reviewer_owned_compound_program",
            purpose="nominal",
            phases=(
                LoadPhase("task_phase", "none", None, 0.0, 0.1),
            ),
            moving_mass_kg=0.1,
        )
        thresholds = CompoundKinematicThresholds(
            min_directional_angle_rad=0.01,
            max_joint_drift_m=0.001,
            max_off_plane_rotation_rad=0.05,
            max_zero_input_output_drift_rad=0.05,
            max_transverse_output_angle_rad=0.05,
            output_angle_tolerance_rad=0.05,
            home_output_angle_rad=0.0,
            outbound_output_angle_rad=math.pi,
        )
        metrics = {
            "max_joint_drift_m": 0.0,
            "max_off_plane_rotation_rad": 0.0,
            "max_bushing_reaction_force_n": 1.0,
            "min_output_angle_rad": 0.0,
            "max_output_angle_rad": 0.0,
            "max_abs_output_delta_rad": 0.0,
            "final_output_angle_rad": 0.0,
            "phase_directional_travel_rad": {},
            "phase_output_bounds_rad": {},
        }

        self.assertEqual(
            _compound_behavior_failures(
                metrics,
                branch="physical",
                load_case=load_case,
                thresholds=thresholds,
                reviewer_task_program=True,
            ),
            [],
        )
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "task_plugins.claimed_reviewer_task_programs",
            return_value=frozenset({load_case.program}),
        ):
            self.assertEqual(
                _compound_behavior_failures(
                    metrics,
                    branch="physical",
                    load_case=load_case,
                    thresholds=thresholds,
                ),
                [],
            )

    def test_compound_physical_behavior_propagates_task_program_owner(
        self,
    ) -> None:
        plugin = family_plugins.CompoundMechanismAdapter()
        plugin.family_spec_kinds = frozenset({"reviewer_compound_kind"})
        load_case = LoadCase(
            load_case_id="task_plugin_case",
            family="compound",
            program="task_reviewer_owned_compound_program",
            purpose="nominal",
            phases=(LoadPhase("task_phase", "none", None, 0.0, 0.1),),
            moving_mass_kg=0.1,
        )
        thresholds = CompoundKinematicThresholds(
            min_directional_angle_rad=0.01,
            max_joint_drift_m=0.001,
            max_off_plane_rotation_rad=0.05,
            max_zero_input_output_drift_rad=0.05,
            max_transverse_output_angle_rad=0.05,
            output_angle_tolerance_rad=0.05,
            home_output_angle_rad=0.0,
            outbound_output_angle_rad=math.pi,
        )
        metrics = {
            "max_joint_drift_m": 0.0,
            "max_off_plane_rotation_rad": 0.0,
            "max_bushing_reaction_force_n": 1.0,
            "min_output_angle_rad": 0.0,
            "max_output_angle_rad": 0.0,
            "max_abs_output_delta_rad": 0.0,
            "final_output_angle_rad": 0.0,
            "phase_directional_travel_rad": {},
            "phase_output_bounds_rad": {},
        }
        plan = SimpleNamespace(
            family="compound",
            axis_abs=(0.0, 0.0, 1.0),
        )
        with (
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "comparison.CompoundKinematicThresholds.for_plan",
                return_value=thresholds,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "comparison._validate_compound_trace",
                return_value=[],
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "comparison._compound_metrics",
                return_value=metrics,
            ),
            patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "task_plugins.reviewer_task_program_policy",
                return_value=object(),
            ),
        ):
            behavior = plugin.physical_behavior(
                plan,
                load_case,
                {},
                {"mechanism_kind": "reviewer_compound_kind"},
            )

        assert behavior is not None
        self.assertEqual(behavior[2], [])

    def test_zero_input_multi_output_ratios_are_json_null(self) -> None:
        load_case = LoadCase(
            load_case_id="zero_input",
            family="multi_output_shaft",
            program="zero_input_multi_output_shaft",
            purpose="adversarial",
            phases=(
                LoadPhase(
                    phase_id="zero_input",
                    wrench_kind="none",
                    direction_abs=None,
                    magnitude_si=0.0,
                    duration_s=0.1,
                ),
            ),
            moving_mass_kg=0.1,
        )
        spec = SimpleNamespace(
            input_body_id="input",
            output_body_ids=("left", "right"),
            shafts=(
                SimpleNamespace(body_id="input"),
                SimpleNamespace(body_id="left"),
                SimpleNamespace(body_id="right"),
            ),
        )
        samples = [
            {
                "phase_id": "zero_input",
                "shaft_angles_rad": {
                    "input": 0.0,
                    "left": 0.0,
                    "right": 0.0,
                },
                "max_mesh_error_rad": 0.0,
                "max_support_drift_m": 0.0,
                "max_off_axis_rotation_rad": 0.0,
                "max_bushing_reaction_force_n": 0.0,
            },
            {
                "phase_id": "zero_input",
                "shaft_angles_rad": {
                    "input": 0.0,
                    "left": 0.0,
                    "right": 0.0,
                },
                "max_mesh_error_rad": 0.0,
                "max_support_drift_m": 0.0,
                "max_off_axis_rotation_rad": 0.0,
                "max_bushing_reaction_force_n": 0.0,
            },
        ]

        metrics = _multi_output_shaft_metrics(
            samples,
            load_case,
            spec,
        )

        self.assertIsNone(
            metrics["output_to_input_ratio"]["left"]
        )
        self.assertIsNone(
            metrics["output_to_input_ratio"]["right"]
        )
        self.assertIsNone(metrics["output_pair_ratio"])
        with tempfile.TemporaryDirectory() as temporary:
            write_json_atomic(
                Path(temporary) / "metrics.json",
                {"physical_metrics": metrics},
            )

    def _plan_and_loads(self, root: Path) -> tuple[object, tuple[LoadCase, ...]]:
        package, report = _build_package(root)
        with patch(
            "simulation_ready_cad_workflow.validation_pipeline.mechanism."
            "discovery.select_adapter",
            return_value=_TEST_ADAPTER,
        ):
            plan = discover_mechanisms(package, report)[0]
        return plan, default_load_cases(plan)

    def _composite_evidence_fixture(
        self,
        root: Path,
        *,
        include_ideal_dynamics: bool,
    ) -> tuple[object, LoadCase, dict[str, Path]]:
        plan, load_cases = self._plan_and_loads(root)
        load_case = load_cases[0]
        output = root / "run"
        trace_root = (
            output / "traces" / plan.capsule_id / load_case.load_case_id
        )
        result_root = output / "results" / plan.capsule_id
        paths = {
            "load": (
                output
                / "loads"
                / plan.capsule_id
                / f"{load_case.load_case_id}.json"
            ),
            "physical_trace": trace_root / "physical.trace.json",
            "ideal_reference_trace": (
                trace_root / "ideal_reference.trace.json"
            ),
            "ideal_dynamics_trace": (
                trace_root / "ideal_dynamics.trace.json"
            ),
            "physical_trace_audit": (
                result_root
                / "physical_trace_audit"
                / f"{load_case.load_case_id}.json"
            ),
            "ideal_reference_comparison": (
                result_root
                / "ideal_reference_comparison"
                / f"{load_case.load_case_id}.json"
            ),
            "ideal_dynamics_comparison": (
                result_root
                / "ideal_dynamics_comparison"
                / f"{load_case.load_case_id}.json"
            ),
            "comparison": (
                result_root
                / "comparison"
                / f"{load_case.load_case_id}.json"
            ),
            "lane": (
                result_root / "lanes" / f"{load_case.load_case_id}.json"
            ),
            "physical_preflight": (
                output / "results" / "physical_preflight.json"
            ),
        }
        write_json_atomic(paths["load"], load_case.to_dict())
        validator_source_sha256 = "v" * 64
        physical = _trace(plan, load_case, branch="physical")
        physical["validator_source_sha256"] = validator_source_sha256
        ideal_reference = _trace(plan, load_case, branch="reduced")
        ideal_reference["branch"] = "ideal_reference"
        ideal_reference["physical_trace_sha256"] = canonical_sha256(
            physical
        )
        write_json_atomic(paths["physical_trace"], physical)
        write_json_atomic(paths["ideal_reference_trace"], ideal_reference)
        physical_sha256 = sha256_file(paths["physical_trace"])
        ideal_reference_sha256 = sha256_file(
            paths["ideal_reference_trace"]
        )
        deferred = {
            "status": "DEFERRED",
            "required_environment": "outer_cad_after_queue_barrier",
            "worker_audit_authority": False,
        }
        reference_bindings = {
            "physical_trace_sha256": physical_sha256,
            "ideal_reference_trace_sha256": ideal_reference_sha256,
        }
        write_json_atomic(
            paths["ideal_reference_comparison"],
            {
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
                "verdict": "PASS",
                "evidence_bindings": reference_bindings,
            },
        )
        write_json_atomic(
            paths["comparison"],
            {
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
                "verdict": "PASS",
                "saved_pose_geometry_audit": deferred,
                "evidence_bindings": reference_bindings,
            },
        )
        artifact_paths = {
            label: paths[label]
            for label in (
                "physical_trace",
                "ideal_reference_trace",
                "ideal_reference_comparison",
                "comparison",
            )
        }
        if include_ideal_dynamics:
            ideal_dynamics = _trace(plan, load_case, branch="reduced")
            ideal_dynamics["branch"] = "ideal_dynamics"
            ideal_dynamics["validator_source_sha256"] = (
                validator_source_sha256
            )
            write_json_atomic(
                paths["ideal_dynamics_trace"],
                ideal_dynamics,
            )
            dynamics_bindings = {
                "physical_trace_sha256": physical_sha256,
                "ideal_dynamics_trace_sha256": sha256_file(
                    paths["ideal_dynamics_trace"]
                ),
            }
            write_json_atomic(
                paths["ideal_dynamics_comparison"],
                {
                    "capsule_id": plan.capsule_id,
                    "load_case_id": load_case.load_case_id,
                    "verdict": "PASS",
                    "saved_pose_geometry_audit": deferred,
                    "evidence_bindings": dynamics_bindings,
                },
            )
            artifact_paths.update(
                {
                    "ideal_dynamics_trace": paths[
                        "ideal_dynamics_trace"
                    ],
                    "ideal_dynamics_comparison": paths[
                        "ideal_dynamics_comparison"
                    ],
                }
            )
        artifacts = {
            label: {
                "file": relative_to_root(root, path),
                "sha256": sha256_file(path),
            }
            for label, path in artifact_paths.items()
        }
        write_json_atomic(
            paths["lane"],
            {
                "schema_version": (
                    "simulation_ready_cad_mechanism_load_lane_v1"
                ),
                "status": "PASS",
                "completed_phase": "comparison",
                "package_binding_sha256": plan.package_binding_sha256,
                "capsule_id": plan.capsule_id,
                "load_case_id": load_case.load_case_id,
                "plan_sha256": canonical_sha256(plan.to_dict()),
                "load_case_sha256": canonical_sha256(load_case.to_dict()),
                "validator_source_sha256": validator_source_sha256,
                "artifacts": artifacts,
            },
        )
        write_json_atomic(
            paths["physical_trace_audit"],
            {
                "status": "PHYSICAL_PREFLIGHT_PASSED",
                "passed": True,
                "failure_reasons": [],
                "reduced_branch_authorized": True,
                "geometry_legality": {
                    "passed": True,
                    "sample_count": 2,
                    "max_penetration_m": 0.0,
                },
                "component_contact_intent": {
                    "passed": True,
                    "authoritative": True,
                    "outcomes": [],
                },
                "physical_trace_sha256": physical_sha256,
                "audit_execution_environment": (
                    "outer_cad_after_queue_barrier"
                ),
                "worker_produced_audit_accepted": False,
            },
        )
        audit_sha256 = sha256_file(paths["physical_trace_audit"])
        write_json_atomic(
            paths["physical_preflight"],
            {
                "passed": True,
                "traces": [
                    {
                        "passed": True,
                        "capsule_id": plan.capsule_id,
                        "load_case_id": load_case.load_case_id,
                        "trace": relative_to_root(
                            root, paths["physical_trace"]
                        ),
                        "trace_sha256": physical_sha256,
                        "physical_trace_audit": relative_to_root(
                            root, paths["physical_trace_audit"]
                        ),
                        "physical_trace_audit_sha256": audit_sha256,
                    }
                ],
            },
        )
        return plan, load_case, paths

    def _load_composite_fixture(
        self,
        root: Path,
        plan: object,
        load_case: LoadCase,
        paths: dict[str, Path],
    ) -> dict[str, object]:
        return _load_composite_lane_evidence(
            workspace=root,
            output=root / "run",
            plan=plan,
            load_case=load_case,
            load_path=paths["load"],
            physical_path=paths["physical_trace"],
            ideal_reference_path=paths["ideal_reference_trace"],
        )

    def test_finalize_loads_bound_optional_ideal_dynamics_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_case, paths = self._composite_evidence_fixture(
                root,
                include_ideal_dynamics=True,
            )

            result = self._load_composite_fixture(
                root, plan, load_case, paths
            )

        self.assertIn("ideal_dynamics", result["load_evidence"])
        self.assertIn(
            "ideal_dynamics_comparison",
            result["load_evidence"],
        )
        self.assertEqual(len(result["trace_files"]), 3)

    def test_certificate_rechecks_outer_audit_after_materialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_case, paths = self._composite_evidence_fixture(
                root,
                include_ideal_dynamics=False,
            )
            materialized = self._load_composite_fixture(
                root, plan, load_case, paths
            )
            changed_audit = read_json(paths["physical_trace_audit"])
            changed_audit["changed_after_materialization"] = True
            write_json_atomic(paths["physical_trace_audit"], changed_audit)

            with self.assertRaises(MechanismValidationError) as caught:
                _certificate_physical_trace_audits(
                    workspace=root,
                    output=root / "run",
                    result_root=root / "run" / "results" / plan.capsule_id,
                    plan=plan,
                    load_cases=(load_case,),
                    traces_by_load_case={
                        load_case.load_case_id: materialized["load_evidence"]
                    },
                )

        self.assertEqual(
            caught.exception.code,
            "FINALIZE_PHYSICAL_AUDIT_EVIDENCE_INVALID",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            "harness_code",
        )

    def test_finalize_rejects_half_bound_ideal_dynamics_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_case, paths = self._composite_evidence_fixture(
                root,
                include_ideal_dynamics=True,
            )
            lane = read_json(paths["lane"])
            lane["artifacts"].pop("ideal_dynamics_comparison")
            write_json_atomic(paths["lane"], lane)

            with self.assertRaisesRegex(
                MechanismValidationError,
                "must be bound together",
            ):
                self._load_composite_fixture(root, plan, load_case, paths)

    def test_finalize_rejects_extra_composite_lane_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_case, paths = self._composite_evidence_fixture(
                root,
                include_ideal_dynamics=False,
            )
            lane = read_json(paths["lane"])
            lane["artifacts"]["unexpected"] = lane["artifacts"][
                "physical_trace"
            ]
            write_json_atomic(paths["lane"], lane)

            with self.assertRaisesRegex(
                MechanismValidationError,
                "open or incomplete",
            ):
                self._load_composite_fixture(root, plan, load_case, paths)

    def test_finalize_rejects_changed_ideal_dynamics_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_case, paths = self._composite_evidence_fixture(
                root,
                include_ideal_dynamics=True,
            )
            comparison = read_json(paths["ideal_dynamics_comparison"])
            comparison["verdict"] = "FAIL"
            write_json_atomic(paths["ideal_dynamics_comparison"], comparison)

            with self.assertRaisesRegex(
                MechanismValidationError,
                "artifact changed",
            ):
                self._load_composite_fixture(root, plan, load_case, paths)

    def test_valid_synthetic_pair_passes_kinematic_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            comparison = compare_trace_pair(
                plan,
                load_case,
                _trace(plan, load_case, branch="physical"),
                _trace(plan, load_case, branch="reduced"),
            )

        self.assertEqual(comparison["verdict"], "PASS")
        self.assertEqual(comparison["failure_reasons"], [])

    def test_compound_force_response_is_not_signed_by_output_axis(self) -> None:
        load_case = LoadCase(
            load_case_id="pedal_roundtrip",
            family="compound",
            program="bidirectional_linkage",
            purpose="nominal",
            phases=(
                LoadPhase("press", "force", (0.0, 0.0, -1.0), 60.0, 1.0),
                LoadPhase("open_settle", "none", None, 0.0, 0.1),
                LoadPhase("release", "force", (0.0, 0.0, 1.0), 6.0, 1.0),
                LoadPhase("closed_settle", "none", None, 0.0, 0.1),
            ),
            moving_mass_kg=1.0,
        )
        samples = [
            {"phase_id": "press", "compound_output_angle_rad": 0.0},
            {"phase_id": "press", "compound_output_angle_rad": 1.0},
            {"phase_id": "open_settle", "compound_output_angle_rad": 1.0},
            {"phase_id": "release", "compound_output_angle_rad": 0.6},
            {"phase_id": "release", "compound_output_angle_rad": 0.0},
            {"phase_id": "closed_settle", "compound_output_angle_rad": 0.0},
        ]

        travel = _compound_phase_directional_travel(
            samples,
            load_case,
            axis=(0.0, 1.0, 0.0),
        )

        self.assertAlmostEqual(travel["press"], 1.0)
        self.assertAlmostEqual(travel["release"], 1.0)

    def test_reduced_limit_must_match_physical_contact_envelope(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            physical = _trace(plan, load_case, branch="physical")
            reduced = _trace(plan, load_case, branch="reduced")
            physical["collision_geometry"] = {
                "certifiable_progress_limit_m": 0.02
            }
            reduced["engine"]["ideal_travel_limit"] = {
                "coordinate": "joint_frame_z",
                "min_m": 0.0,
                "max_m": 0.03,
                "source": "physical_contact_certifiable_progress_limit",
            }
            comparison = compare_trace_pair(
                plan,
                load_case,
                physical,
                reduced,
            )

        self.assertEqual(comparison["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "reduced_prismatic_limit_does_not_match_physical_envelope",
            comparison["failure_reasons"],
        )

    def test_scope_tolerance_is_limited_to_one_integrator_step(self) -> None:
        wide_adapter = _wide_mcm_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            plan = replace(
                plan,
                adapter_id=wide_adapter.adapter_id,
                adapter_spec=adapter_to_dict(wide_adapter),
            )
            load_case = load_cases[0]
            metrics = {
                "max_abs_progress_m": 0.05027,
                "max_abs_axis_speed_m_s": 0.30,
            }
            failures, tolerance = _physical_scope_evaluation(
                plan, metrics, load_case
            )
            metrics["max_abs_progress_m"] = 0.051
            excessive, _ = _physical_scope_evaluation(
                plan, metrics, load_case
            )

        self.assertEqual(failures, [])
        self.assertGreaterEqual(tolerance, 0.0003)
        self.assertLess(tolerance, 0.0005)
        self.assertEqual(
            excessive,
            ["physical_contact_model_progress_scope_exceeded"],
        )

    def test_force_response_may_complete_during_following_settle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            physical = _trace(plan, load_case, branch="physical")
            reduced = _trace(plan, load_case, branch="reduced")
            reverse_force_id = load_case.phases[2].phase_id
            reverse_settle_id = load_case.phases[3].phase_id
            for sample in reduced["samples"]:
                if sample["phase_id"] == reverse_force_id:
                    sample["generalized_progress_m"] = 0.001
                    sample["observable_displacement_abs_m"] = [0.001, 0.0, 0.0]
                if sample["phase_id"] == reverse_settle_id:
                    sample["generalized_progress_m"] = 0.0
                    sample["observable_displacement_abs_m"] = [0.0, 0.0, 0.0]
            comparison = compare_trace_pair(
                plan,
                load_case,
                physical,
                reduced,
            )

        self.assertEqual(comparison["verdict"], "PASS")
        self.assertNotIn(
            "reduced_reverse_force_did_not_move_as_loaded",
            comparison["failure_reasons"],
        )

    def test_unidirectional_axis_checks_only_declared_force_phases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            base = load_cases[0]
            load_case = replace(
                base,
                program="unidirectional_axis",
                phases=base.phases[:2],
            )
            failures = _behavior_failures(
                {
                    "max_lateral_shift_m": 0.0,
                    "max_rotation_rad": 0.0,
                    "axis_purity": 1.0,
                    "phase_directional_travel_m": {
                        "forward_force": 0.001,
                    },
                },
                branch="physical",
                load_case=load_case,
                thresholds=KinematicThresholds.for_plan(plan),
            )

        self.assertEqual(failures, [])

    def test_unidirectional_axis_allows_a_bounded_low_amplitude_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            base = load_cases[0]
            load_case = replace(
                base,
                program="unidirectional_axis",
                phases=base.phases[:2],
            )
            failures = _behavior_failures(
                {
                    "max_lateral_shift_m": 0.0,
                    "max_rotation_rad": 0.0,
                    "axis_purity": 1.0,
                    "phase_directional_travel_m": {
                        "forward_force": 0.0,
                    },
                },
                branch="physical",
                load_case=load_case,
                thresholds=KinematicThresholds.for_plan(plan),
            )

        self.assertNotIn(
            "physical_forward_force_did_not_move_as_loaded",
            failures,
        )

    def test_new_ideal_dynamics_name_uses_legacy_solver_compatibility(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_cases = self._plan_and_loads(root)
            load_case = load_cases[0]
            physical_path = root / "physical.trace.json"
            ideal_path = root / "ideal_dynamics.trace.json"
            physical = _trace(plan, load_case, branch="physical")
            ideal = _trace(plan, load_case, branch="reduced")
            ideal["branch"] = "ideal_dynamics"
            _write_json(physical_path, physical)
            _write_json(ideal_path, ideal)

            result = compare_ideal_dynamics_trace_files(
                plan,
                load_case,
                physical_path,
                ideal_path,
                workspace_root=root,
            )

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(
            result["comparison_mode"],
            "physical_vs_ideal_dynamics",
        )
        self.assertFalse(result["contact_evidence_authority"])

    def test_composite_ideal_dynamics_defers_saved_pose_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_cases = self._plan_and_loads(root)
            load_case = load_cases[0]
            physical_path = root / "physical.trace.json"
            ideal_path = root / "ideal_dynamics.trace.json"
            physical = _trace(plan, load_case, branch="physical")
            ideal = _trace(plan, load_case, branch="reduced")
            ideal["branch"] = "ideal_dynamics"
            _write_json(physical_path, physical)
            _write_json(ideal_path, ideal)

            result = compare_composite_lane_ideal_dynamics_trace_files(
                plan,
                load_case,
                physical_path,
                ideal_path,
                workspace_root=root,
            )

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(
            result["saved_pose_geometry_audit"],
            {
                "status": "DEFERRED",
                "required_environment": "outer_cad_after_queue_barrier",
                "worker_audit_authority": False,
            },
        )
    def test_physical_behavior_can_certify_against_light_ideal_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            result = compare_physical_behavior_to_ideal_reference(
                plan,
                load_case,
                _trace(plan, load_case, branch="physical"),
                {
                    "verdict": "PASS",
                    "metrics": {"max_absolute_error": 0.0},
                    "threshold": {"max_absolute_error": 0.001},
                    "failure_reasons": [],
                },
                workspace_root=temporary,
            )

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(
            result["comparison_mode"],
            "physical_vs_ideal_reference",
        )
        self.assertFalse(result["contact_evidence_authority"])

    def test_composite_behavior_defers_saved_pose_audit_to_outer_cad(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            result = (
                compare_composite_lane_physical_behavior_to_ideal_reference(
                    plan,
                    load_case,
                    _trace(plan, load_case, branch="physical"),
                    {
                        "verdict": "PASS",
                        "metrics": {"max_absolute_error": 0.0},
                        "threshold": {"max_absolute_error": 0.001},
                        "failure_reasons": [],
                    },
                    workspace_root=temporary,
                )
            )

        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(
            result["saved_pose_geometry_audit"],
            {
                "status": "DEFERRED",
                "required_environment": "outer_cad_after_queue_barrier",
                "worker_audit_authority": False,
            },
        )

    def test_range_roundtrip_certifies_only_observed_forty_mm_envelope(self) -> None:
        wide_adapter = _wide_mcm_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            plan, _ = self._plan_and_loads(Path(temporary))
            plan = replace(
                plan,
                adapter_id=wide_adapter.adapter_id,
                adapter_spec=adapter_to_dict(wide_adapter),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "loads.get_adapter",
                return_value=wide_adapter,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "comparison.get_adapter",
                return_value=wide_adapter,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "certificate.get_adapter",
                return_value=wide_adapter,
            ):
                load_cases = default_load_cases(plan)
                comparisons = tuple(
                    compare_trace_pair(
                        plan,
                        load_case,
                        _trace(plan, load_case, branch="physical"),
                        _trace(plan, load_case, branch="reduced"),
                    )
                    for load_case in load_cases
                )
                certificate = build_certificate(plan, load_cases, comparisons)

        self.assertEqual(certificate["verdict"], "KINEMATIC_CERTIFIED")
        self.assertEqual(
            certificate["tested_envelope"]["generalized_progress_m"]["max"],
            0.04,
        )

    def test_certified_range_builds_one_guarded_whole_machine_job(self) -> None:
        wide_adapter = _wide_mcm_adapter()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_plan, _ = self._plan_and_loads(root)
            plan = replace(
                base_plan,
                adapter_id=wide_adapter.adapter_id,
                adapter_spec=adapter_to_dict(wide_adapter),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
                task_ids=("return_payload_home", "transport_payload_outbound"),
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "loads.get_adapter",
                return_value=wide_adapter,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "comparison.get_adapter",
                return_value=wide_adapter,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "certificate.get_adapter",
                return_value=wide_adapter,
            ):
                load_cases = default_load_cases(plan)
                comparisons = tuple(
                    compare_trace_pair(
                        plan,
                        load_case,
                        _trace(plan, load_case, branch="physical"),
                        _trace(plan, load_case, branch="reduced"),
                    )
                    for load_case in load_cases
                )
                certificate_path = root / "certificate.json"
                replacement_path = root / "replacement.json"
                trace_path = root / "run" / "traces" / "fixture.trace.json"
                write_json_atomic(trace_path, {"status": "PHYSICAL_TRACE"})
                trace_sha256 = _sha256(trace_path)
                write_certificate_artifacts(
                    plan,
                    load_cases,
                    comparisons,
                    certificate_path=certificate_path,
                    replacement_path=replacement_path,
                    trace_files=(trace_path,),
                    workspace_root=root,
                )
                certificate = read_json(certificate_path)
            summary = build_whole_machine_dispatch(
                plan,
                workspace_root=root,
                output_dir=root / "run",
                certificate_file=certificate_path,
                replacement_file=replacement_path,
                pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
            )
            manifest = QueueManifest.from_file(root / summary["queue_manifest"])
            mechanism_plan_argument = manifest.jobs[0].command.index(
                "--mechanism-plan"
            )
            mechanism_plan_file = root / manifest.jobs[0].command[
                mechanism_plan_argument + 1
            ]
            dispatched_mechanism_plan = read_json(mechanism_plan_file)

        self.assertEqual(len(manifest.jobs), 1)
        self.assertEqual(
            certificate["trace_file_hashes"],
            {"run/traces/fixture.trace.json": trace_sha256},
        )
        self.assertEqual(manifest.jobs[0].kind.value, "pychrono")
        # 2026-08-10: cap is adaptive (25% of total RAM, floor 1.0 GiB),
        # not the old hard-coded 1.0 that false-blocked r29.
        self.assertEqual(
            manifest.jobs[0].max_process_tree_memory_gib,
            _whole_machine_max_memory_gib(),
        )
        self.assertEqual(manifest.resource_policy.min_available_memory_gib, 1.5)
        self.assertEqual(
            manifest.jobs[0].command[:3],
            (
                "/opt/test-physics/bin/python",
                "-m",
                (
                    "simulation_ready_cad_workflow.validation_pipeline."
                    "mechanism.whole_machine_worker"
                ),
            ),
        )
        self.assertEqual(
            dispatched_mechanism_plan,
            json.loads(json.dumps(plan.to_dict())),
        )
        self.assertEqual(
            manifest.jobs[0].environment["PYTHONPATH"],
            str(validator_import_root()),
        )

    def test_short_range_certificate_cannot_authorize_forty_mm_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_plan, short_load_cases = self._plan_and_loads(root)
            plan = replace(
                base_plan,
                adapter_id="mcmaster_6725k41_8438k4_v1",
                adapter_spec=adapter_to_dict(
                    get_adapter("mcmaster_6725k41_8438k4_v1")
                ),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
                task_ids=("return_payload_home", "transport_payload_outbound"),
            )
            comparisons = tuple(
                compare_trace_pair(
                    plan,
                    load_case,
                    _trace(plan, load_case, branch="physical"),
                    _trace(plan, load_case, branch="reduced"),
                )
                for load_case in short_load_cases
            )
            certificate_path = root / "certificate.json"
            replacement_path = root / "replacement.json"
            write_certificate_artifacts(
                plan,
                short_load_cases,
                comparisons,
                certificate_path=certificate_path,
                replacement_path=replacement_path,
            )
            with self.assertRaises(MechanismValidationError) as caught:
                build_whole_machine_dispatch(
                    plan,
                    workspace_root=root,
                    output_dir=root / "run",
                    certificate_file=certificate_path,
                    replacement_file=replacement_path,
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )

        self.assertEqual(
            caught.exception.code,
            "WHOLE_MACHINE_TASK_OUTSIDE_CERTIFIED_ENVELOPE",
        )

    def test_wrong_axis_in_physical_branch_is_a_physical_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            comparison = compare_trace_pair(
                plan,
                load_case,
                _trace(plan, load_case, branch="physical", wrong_axis=True),
                _trace(plan, load_case, branch="reduced"),
            )

        self.assertEqual(comparison["verdict"], "PHYSICAL_FAILURE")
        self.assertIn(
            "physical_wrong_motion_subspace", comparison["failure_reasons"]
        )

    def test_hidden_ideal_joint_in_physical_branch_is_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            load_case = load_cases[0]
            comparison = compare_trace_pair(
                plan,
                load_case,
                _trace(
                    plan,
                    load_case,
                    branch="physical",
                    wrong_link_semantics=True,
                ),
                _trace(plan, load_case, branch="reduced"),
            )

        self.assertEqual(comparison["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "physical_branch_contains_internal_ideal_joint",
            comparison["failure_reasons"],
        )

    def test_bound_rolling_model_requires_explicit_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base_plan, load_cases = self._plan_and_loads(Path(temporary))
            plan = replace(
                base_plan,
                adapter_id="mcmaster_6725k41_8438k4_v1",
                adapter_spec=adapter_to_dict(
                    get_adapter("mcmaster_6725k41_8438k4_v1")
                ),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
            )
            physical = _trace(plan, load_cases[0], branch="physical")
            physical["rolling_element_validation"] = None
            comparison = compare_trace_pair(
                plan,
                load_cases[0],
                physical,
                _trace(plan, load_cases[0], branch="reduced"),
            )

        self.assertEqual(comparison["verdict"], "PHYSICS_INCONCLUSIVE")
        self.assertIn(
            "physical_rolling_element_validation_missing",
            comparison["failure_reasons"],
        )

    def test_contact_model_scope_is_a_hard_physical_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base_plan, load_cases = self._plan_and_loads(Path(temporary))
            plan = replace(
                base_plan,
                adapter_id="mcmaster_6725k41_8438k4_v1",
                adapter_spec=adapter_to_dict(
                    get_adapter("mcmaster_6725k41_8438k4_v1")
                ),
                requirements_sha256=(
                    "e962c417aa97ba4fbc4c219f745a0467eb6dd5f37bab95c2f5865ae14985b325"
                ),
            )
            physical = _trace(plan, load_cases[0], branch="physical")
            physical["samples"][1]["generalized_progress_m"] = 0.005
            physical["samples"][1]["observable_displacement_abs_m"] = [
                0.005,
                0.0,
                0.0,
            ]
            comparison = compare_trace_pair(
                plan,
                load_cases[0],
                physical,
                _trace(plan, load_cases[0], branch="reduced"),
            )

        self.assertEqual(comparison["verdict"], "PHYSICAL_FAILURE")
        self.assertIn(
            "physical_contact_model_progress_scope_exceeded",
            comparison["failure_reasons"],
        )

    def test_complete_suite_issues_certificate_and_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_cases = self._plan_and_loads(root)
            comparisons = tuple(
                compare_trace_pair(
                    plan,
                    load_case,
                    _trace(plan, load_case, branch="physical"),
                    _trace(plan, load_case, branch="reduced"),
                )
                for load_case in load_cases
            )
            certificate_path = root / "certificate.json"
            replacement_path = root / "replacement.json"
            certificate, replacement = write_certificate_artifacts(
                plan,
                load_cases,
                comparisons,
                certificate_path=certificate_path,
                replacement_path=replacement_path,
            )

        self.assertEqual(certificate["verdict"], "KINEMATIC_CERTIFIED")
        self.assertIsNotNone(replacement)
        self.assertEqual(replacement["status"], "ACTIVE")
        self.assertEqual(
            replacement["replacement_model"]["family"], "prismatic"
        )

    def test_typed_task_requires_bound_ideal_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base_plan, load_cases = self._plan_and_loads(Path(temporary))
            target_case = load_cases[0]
            plan = replace(
                base_plan,
                acceptance_contract={
                    "status": "PASS",
                    "certificate_authority": True,
                    "requirements": [
                        {
                            "test_load_case_id": target_case.load_case_id,
                            "required_branches": [
                                "physical",
                                "ideal_reference",
                            ],
                        }
                    ],
                },
                semantic_binding={
                    "status": "PASS",
                    "certificate_authority": True,
                },
            )
            comparisons = tuple(
                compare_trace_pair(
                    plan,
                    load_case,
                    _trace(plan, load_case, branch="physical"),
                    _trace(plan, load_case, branch="reduced"),
                )
                for load_case in load_cases
            )
            task_evidence = {
                "status": "PASS",
                "certificate_authority": True,
            }
            without_ideal = build_certificate(
                plan,
                load_cases,
                comparisons,
                task_acceptance=task_evidence,
            )
            with_ideal = build_certificate(
                plan,
                load_cases,
                comparisons,
                task_acceptance=task_evidence,
                ideal_reference_comparisons=(
                    {
                        "load_case_id": target_case.load_case_id,
                        "verdict": "PASS",
                    },
                ),
            )

        self.assertEqual(
            without_ideal["typed_task_verdict"],
            "INCONCLUSIVE",
        )
        self.assertFalse(without_ideal["replacement_eligible"])
        self.assertEqual(with_ideal["overall_verdict"], "TASK_CERTIFIED")
        self.assertTrue(with_ideal["replacement_eligible"])
    def test_failed_certificate_cannot_authorize_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan, load_cases = self._plan_and_loads(Path(temporary))
            comparisons = []
            for index, load_case in enumerate(load_cases):
                comparisons.append(
                    compare_trace_pair(
                        plan,
                        load_case,
                        _trace(
                            plan,
                            load_case,
                            branch="physical",
                            wrong_axis=index == 0,
                        ),
                        _trace(plan, load_case, branch="reduced"),
                    )
                )
            certificate = build_certificate(
                plan, load_cases, tuple(comparisons)
            )
            with self.assertRaises(MechanismValidationError) as caught:
                build_replacement(
                    plan,
                    certificate,
                    certificate_sha256="0" * 64,
                )

        self.assertEqual(certificate["verdict"], "PHYSICAL_FAILURE")
        self.assertEqual(caught.exception.code, "REPLACEMENT_NOT_AUTHORIZED")

    def test_failed_rerun_revokes_stale_active_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_cases = self._plan_and_loads(root)
            passing = tuple(
                compare_trace_pair(
                    plan,
                    load_case,
                    _trace(plan, load_case, branch="physical"),
                    _trace(plan, load_case, branch="reduced"),
                )
                for load_case in load_cases
            )
            certificate_path = root / "certificate.json"
            replacement_path = root / "replacement.json"
            write_certificate_artifacts(
                plan,
                load_cases,
                passing,
                certificate_path=certificate_path,
                replacement_path=replacement_path,
            )
            failing = list(passing)
            failing[0] = compare_trace_pair(
                plan,
                load_cases[0],
                _trace(
                    plan,
                    load_cases[0],
                    branch="physical",
                    wrong_axis=True,
                ),
                _trace(plan, load_cases[0], branch="reduced"),
            )
            _, replacement = write_certificate_artifacts(
                plan,
                load_cases,
                tuple(failing),
                certificate_path=certificate_path,
                replacement_path=replacement_path,
            )
            marker = json.loads(replacement_path.read_text(encoding="utf-8"))

        self.assertIsNone(replacement)
        self.assertEqual(marker["status"], "INACTIVE")
        self.assertEqual(marker["certificate_verdict"], "PHYSICAL_FAILURE")

    def test_nsc_system_sets_collision_defaults_after_construction(self) -> None:
        events: list[object] = []

        class CollisionModel:
            @staticmethod
            def SetDefaultSuggestedEnvelope(value: float) -> None:
                events.append(("envelope", value))

            @staticmethod
            def SetDefaultSuggestedMargin(value: float) -> None:
                events.append(("margin", value))

        class CollisionSystem:
            Type_BULLET = "bullet"

        class IterativeSolver:
            def SetMaxIterations(self, value: int) -> None:
                events.append(("solver_iterations", value))

            def EnableWarmStart(self, value: bool) -> None:
                events.append(("solver_warm_start", value))

        class Solver:
            def AsIterative(self) -> IterativeSolver:
                events.append("iterative_solver")
                return IterativeSolver()

        class System:
            def SetCollisionSystemType(self, value: object) -> None:
                events.append(("backend", value))

            def GetSolver(self) -> Solver:
                return Solver()

        class Chrono:
            ChCollisionModel = CollisionModel
            ChCollisionSystem = CollisionSystem

            @staticmethod
            def ChSystemNSC() -> System:
                events.append("system")
                return System()

        _make_nsc_system(
            Chrono,
            collision_envelope_m=10e-6,
            collision_margin_m=2e-6,
        )
        self.assertEqual(
            events,
            [
                "system",
                ("envelope", 10e-6),
                ("margin", 2e-6),
                ("backend", "bullet"),
                "iterative_solver",
                ("solver_iterations", 200),
                ("solver_warm_start", True),
            ],
        )

    def test_reduced_prismatic_limit_uses_joint_free_axis(self) -> None:
        events: list[object] = []

        class Limit:
            def SetActive(self, value: bool) -> None:
                events.append(("active", value))

            def SetMin(self, value: float) -> None:
                events.append(("min", value))

            def SetMax(self, value: float) -> None:
                events.append(("max", value))

        limit = Limit()

        class Prismatic:
            def LimitZ(self) -> Limit:
                events.append("limit_z")
                return limit

        disclosure = _configure_prismatic_travel_limit(
            Prismatic(), 0.02, min_progress_m=-0.001
        )

        self.assertEqual(
            events,
            ["limit_z", ("active", True), ("min", -0.001), ("max", 0.02)],
        )
        self.assertEqual(disclosure["coordinate"], "joint_frame_z")
        self.assertEqual(disclosure["min_m"], -0.001)
        self.assertEqual(disclosure["max_m"], 0.02)

    def test_vector_snapshot_does_not_alias_mutable_chrono_proxy(self) -> None:
        class Vector:
            x = 1.0
            y = 2.0
            z = 3.0

        vector = Vector()
        snapshot = _snapshot_vector3(vector)
        vector.x = 4.0

        self.assertEqual(snapshot, (1.0, 2.0, 3.0))

    def test_exact_box_recognition_requires_complete_surface(self) -> None:
        box_lines = [
            "v 0 0 0",
            "v 1 0 0",
            "v 1 2 0",
            "v 0 2 0",
            "v 0 0 3",
            "v 1 0 3",
            "v 1 2 3",
            "v 0 2 3",
            "f 1 3 2",
            "f 1 4 3",
            "f 5 6 7",
            "f 5 7 8",
            "f 1 2 6",
            "f 1 6 5",
            "f 2 3 7",
            "f 2 7 6",
            "f 3 4 8",
            "f 3 8 7",
            "f 4 1 5",
            "f 4 5 8",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "complete.obj"
            complete.write_text("\n".join(box_lines) + "\n", encoding="ascii")
            incomplete = root / "incomplete.obj"
            incomplete.write_text(
                "\n".join(box_lines[:-1]) + "\n", encoding="ascii"
            )
            box = _axis_aligned_box_spec(complete)
            incomplete_box = _axis_aligned_box_spec(incomplete)

        self.assertEqual(box, ((1.0, 2.0, 3.0), (0.5, 1.0, 1.5)))
        self.assertIsNone(incomplete_box)

    def test_exact_y_axis_cylinder_is_recognized_as_a_primitive(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pin.obj"
            mesh = trimesh.creation.cylinder(radius=0.005, height=0.18, sections=32)
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    0.5 * 3.141592653589793, (1.0, 0.0, 0.0)
                )
            )
            path.write_text(
                trimesh.exchange.obj.export_obj(mesh), encoding="ascii"
            )
            specification = _axis_aligned_y_cylinder_spec(path)

        self.assertIsNotNone(specification)
        radius, height, center = specification
        self.assertAlmostEqual(radius, 0.005, places=6)
        self.assertAlmostEqual(height, 0.18, places=6)
        self.assertTrue(all(abs(value) < 1e-9 for value in center))

    def test_family_component_representation_accepts_exact_primitive_and_mesh(
        self,
    ) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cylinder_path = root / "pin.obj"
            cylinder = trimesh.creation.cylinder(
                radius=0.005,
                height=0.18,
                sections=32,
            )
            cylinder.apply_transform(
                trimesh.transformations.rotation_matrix(
                    0.5 * 3.141592653589793,
                    (1.0, 0.0, 0.0),
                )
            )
            cylinder_path.write_text(
                trimesh.exchange.obj.export_obj(cylinder),
                encoding="ascii",
            )
            tetrahedron_path = root / "tetrahedron.obj"
            tetrahedron_path.write_text(
                "\n".join(
                    (
                        "v 0 0 0",
                        "v 1 0 0",
                        "v 0 2 0",
                        "v 0 0 3",
                        "f 1 3 2",
                        "f 1 2 4",
                        "f 2 3 4",
                        "f 3 1 4",
                    )
                )
                + "\n",
                encoding="ascii",
            )

            primitive = _validated_component_representation(
                {
                    "component_id": "generic_pin",
                    "convex": True,
                    "representation": "ChCollisionShapeCylinder",
                },
                cylinder_path,
                fixed=False,
            )
            mesh = _validated_component_representation(
                {
                    "component_id": "generic_tetrahedron",
                    "convex": True,
                    "representation": (
                        "ChCollisionShapeTriangleMeshDynamicConvex"
                    ),
                },
                tetrahedron_path,
                fixed=False,
            )

        self.assertEqual(primitive, "ChCollisionShapeCylinder")
        self.assertEqual(
            mesh,
            "ChCollisionShapeTriangleMeshDynamicConvex",
        )

    def test_family_component_representation_rejects_wrong_declaration(
        self,
    ) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pin.obj"
            mesh = trimesh.creation.cylinder(
                radius=0.005,
                height=0.18,
                sections=32,
            )
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    0.5 * 3.141592653589793,
                    (1.0, 0.0, 0.0),
                )
            )
            path.write_text(
                trimesh.exchange.obj.export_obj(mesh),
                encoding="ascii",
            )
            with self.assertRaisesRegex(
                MechanismValidationError,
                "FAMILY_COLLIDER_REPRESENTATION_CHANGED",
            ):
                _validated_component_representation(
                    {
                        "component_id": "generic_pin",
                        "convex": True,
                        "representation": (
                            "ChCollisionShapeTriangleMeshDynamicConvex"
                        ),
                    },
                    path,
                    fixed=False,
                )

    def test_exact_sphere_is_recognized_as_a_primitive(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "retainer.obj"
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.0054)
            # Mimic an OCCT tessellation without a vertex at one exact bound.
            # A bounds-midpoint recognizer would misplace this sphere centre.
            mesh.update_vertices(mesh.vertices[:, 0] > -0.00535)
            mesh.apply_translation((0.1, -0.2, 0.3))
            path.write_text(
                trimesh.exchange.obj.export_obj(mesh), encoding="ascii"
            )
            specification = _sphere_spec(path)

        self.assertIsNotNone(specification)
        radius, center = specification
        self.assertAlmostEqual(radius, 0.0054, places=6)
        self.assertEqual(
            [round(value, 6) for value in center], [0.1, -0.2, 0.3]
        )

    def test_rotated_exact_box_is_recognized_without_a_convex_hull(self) -> None:
        import trimesh

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rotated_box.obj"
            mesh = trimesh.creation.box(extents=(1.0, 2.0, 3.0))
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    0.43, (0.0, 1.0, 0.0)
                )
            )
            path.write_text(
                trimesh.exchange.obj.export_obj(mesh), encoding="ascii"
            )
            specification = _oriented_box_spec(path)

        self.assertIsNotNone(specification)
        lengths, center, rotation = specification
        self.assertEqual(
            [round(value, 6) for value in sorted(lengths)],
            [1.0, 2.0, 3.0],
        )
        self.assertTrue(all(abs(value) < 1e-9 for value in center))
        self.assertEqual(len(rotation), 3)

    def test_mesh_adapter_rotates_and_localizes_without_convex_hull(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "triangle.obj"
            source.write_text(
                "v 1000 0 0\nv 0 1000 0\nv 0 0 1000\nf 1 2 3\n",
                encoding="ascii",
            )
            asset = MeshAssetSpec(
                workspace_relative_path="triangle.obj",
                sha256=_sha256(source),
                units_to_m=0.001,
                rotation_abs_from_source=(
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, -1.0, 0.0),
                ),
            )
            destination = root / "derived" / "triangle.local.obj"
            metadata = _write_body_local_mesh(
                asset,
                workspace=root,
                destination=destination,
                body_origin_abs_m=(0.0, 0.0, 0.0),
            )
            text = destination.read_text(encoding="ascii")

        self.assertIn("v 1 0 0", text)
        self.assertIn("v 0 0 -1", text)
        self.assertFalse(metadata["convex_hull_used"])
        self.assertEqual(metadata["triangle_count"], 1)

    @unittest.skipUnless(
        (Path(__file__).resolve().parents[3] / "simulation_ready_cad_workflow/joint_template_catalog/unit_tests").is_dir(),
        "The vendor rolling-element evidence meshes are not part of this distribution")
    def test_mcmaster_rolling_element_evidence_is_hash_bound(self) -> None:
        workspace = Path(__file__).resolve().parents[3]
        adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
        assert adapter.geometry is not None
        rolling = adapter.geometry.rolling_elements
        assert rolling is not None
        assets = _rolling_element_assets(rolling, workspace=workspace)

        self.assertEqual(len(assets), 18)
        self.assertTrue(all(item["source_triangle_count"] > 0 for item in assets))
        self.assertTrue(all(item["primitive_proxy_used"] is False for item in assets))
        self.assertAlmostEqual(assets[0]["center_abs_m"][1], 0.006195244)
        self.assertAlmostEqual(assets[0]["center_abs_m"][2], -0.00114)

    def test_mcmaster_adapter_preserves_the_source_down_direction(self) -> None:
        adapter = get_adapter("mcmaster_6725k41_8438k4_v1")
        assert adapter.geometry is not None
        rotation = adapter.geometry.fixed_mesh.rotation_abs_from_source
        source_down = (0.0, -1.0, 0.0)
        world_down = tuple(
            sum(row[index] * source_down[index] for index in range(3))
            for row in rotation
        )

        self.assertEqual(world_down, (0.0, 0.0, -1.0))
        self.assertLess(
            adapter.geometry.fixed_origin_abs_m[2],
            adapter.geometry.moving_origin_abs_m[2],
        )

    def test_finalize_pairs_all_traces_and_writes_active_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_cases = self._plan_and_loads(root)
            test_adapter = _with_test_fidelity(
                root, _TEST_ADAPTER, plan.package_binding_sha256
            )
            plan = replace(plan, adapter_spec=adapter_to_dict(test_adapter))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=test_adapter,
            ):
                build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            dispatch_path = root / "run" / "dispatch_summary.json"
            dispatch = read_json(dispatch_path)
            dispatch["execution"] = "legacy_per_branch_v0"
            _write_json(dispatch_path, dispatch)
            for load_case in load_cases:
                trace_root = root / "run" / "traces" / plan.capsule_id / load_case.load_case_id
                physical_trace = _trace(plan, load_case, branch="physical")
                _write_json(
                    trace_root / "physical.trace.json",
                    physical_trace,
                )
                _write_json(
                    trace_root / "reduced.trace.json",
                    _trace(plan, load_case, branch="reduced"),
                )
                _write_json(
                    trace_root / "ideal_reference.trace.json",
                    generate_ideal_reference_trace(
                        plan,
                        load_case,
                        physical_trace,
                    ),
                )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "finalize.audit_physical_trace_evidence",
                return_value={
                    "status": "PHYSICAL_PREFLIGHT_PASSED",
                    "passed": True,
                    "failure_reasons": [],
                    "reduced_branch_authorized": True,
                    "geometry_legality": {
                        "passed": True,
                        "sample_count": 2,
                        "max_penetration_m": 0.0,
                    },
                    "component_contact_intent": {
                        "passed": True,
                        "authoritative": True,
                        "outcomes": [],
                    },
                },
            ), use_pychrono_launcher(_TEST_PYCHRONO_LAUNCHER):
                status = finalize_dispatch(
                    workspace_root=root,
                    output_dir=root / "run",
                )
            replacement = (
                root / "run" / "results" / plan.capsule_id / "replacement.json"
            )
            replacement_exists = replacement.exists()
            certificate = read_json(
                root / "run" / "results" / plan.capsule_id / "certificate.json"
            )
            fidelity_binding = test_adapter.collider_fidelity
            assert fidelity_binding is not None

        self.assertEqual(status["status"], "KINEMATIC_CERTIFIED")
        self.assertTrue(replacement_exists)
        self.assertEqual(status["mechanisms"][0]["verdict"], "KINEMATIC_CERTIFIED")
        self.assertEqual(
            certificate["collider_fidelity_evidence"]["report_sha256"],
            fidelity_binding["report_sha256"],
        )
        physical_audits = certificate["physical_trace_audit_evidence"]
        self.assertEqual(physical_audits["gate_status"], "PASS")
        self.assertEqual(
            physical_audits["record_count"], len(load_cases)
        )
        self.assertTrue(physical_audits["all_records_passed"])
        self.assertTrue(
            all(
                item["report_binding_verified"]
                and item["physical_trace_binding_verified"]
                for item in physical_audits["records"]
            )
        )
        self.assertEqual(
            status["collider_fidelity_reports"][0]["status"], "PASS"
        )

    def test_finalize_cannot_bypass_failed_physical_trace_audit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, load_cases = self._plan_and_loads(root)
            test_adapter = _with_test_fidelity(
                root,
                _TEST_ADAPTER,
                plan.package_binding_sha256,
            )
            plan = replace(
                plan,
                adapter_spec=adapter_to_dict(test_adapter),
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=test_adapter,
            ):
                build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            dispatch_path = root / "run" / "dispatch_summary.json"
            dispatch = read_json(dispatch_path)
            dispatch["execution"] = "legacy_per_branch_v0"
            _write_json(dispatch_path, dispatch)
            for load_case in load_cases:
                trace_root = (
                    root
                    / "run"
                    / "traces"
                    / plan.capsule_id
                    / load_case.load_case_id
                )
                _write_json(
                    trace_root / "physical.trace.json",
                    _trace(plan, load_case, branch="physical"),
                )
                _write_json(
                    trace_root / "reduced.trace.json",
                    _trace(plan, load_case, branch="reduced"),
                )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "finalize.audit_physical_trace_evidence",
                return_value={
                    "status": "PHYSICAL_PREFLIGHT_FAILED",
                    "passed": False,
                    "failure_reasons": [
                        "analytical_transmission_geometry_gate_missing"
                    ],
                },
            ), use_pychrono_launcher(_TEST_PYCHRONO_LAUNCHER):
                status = finalize_dispatch(
                    workspace_root=root,
                    output_dir=root / "run",
                )
            replacement_path = (
                root
                / "run"
                / "results"
                / plan.capsule_id
                / "replacement.json"
            )
            replacement_payload = (
                read_json(replacement_path)
                if replacement_path.is_file()
                else None
            )
            comparison_verdicts = [
                read_json(path)["verdict"]
                for path in (
                    root
                    / "run"
                    / "results"
                    / plan.capsule_id
                    / "comparison"
                ).glob("*.json")
                if path.name != "aggregate.json"
            ]

        self.assertEqual(status["status"], "NOT_CERTIFIED")
        self.assertIsNotNone(replacement_payload)
        self.assertEqual(replacement_payload["status"], "INACTIVE")
        self.assertNotIn("replacement_model", replacement_payload)
        self.assertTrue(comparison_verdicts)
        self.assertTrue(
            all(
                verdict == "PHYSICS_INCONCLUSIVE"
                for verdict in comparison_verdicts
            )
        )

    def test_finalize_without_traces_fails_closed_with_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, _ = self._plan_and_loads(root)
            test_adapter = _with_test_fidelity(
                root, _TEST_ADAPTER, plan.package_binding_sha256
            )
            plan = replace(plan, adapter_spec=adapter_to_dict(test_adapter))
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.mechanism."
                "dispatch.get_adapter",
                return_value=test_adapter,
            ):
                build_dispatch_bundle(
                    (plan,),
                    workspace_root=root,
                    output_dir=root / "run",
                    pychrono_launcher=_TEST_PYCHRONO_LAUNCHER,
                )
            with use_pychrono_launcher(_TEST_PYCHRONO_LAUNCHER):
                with self.assertRaises(MechanismValidationError) as caught:
                    finalize_dispatch(
                        workspace_root=root,
                        output_dir=root / "run",
                    )

        self.assertEqual(
            caught.exception.code,
            "FINALIZE_COMPOSITE_LANE_EVIDENCE_INVALID",
        )
        self.assertEqual(
            caught.exception.failure_envelope["owner"],
            "harness_code",
        )


if __name__ == "__main__":
    unittest.main()
