from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import trimesh

from simulation_ready_cad_workflow.validation_pipeline.dashboard.physics_evidence import (
    _branch_payload,
    _clock_transmission_visuals,
    _landmarks,
    _scene_manifest_is_current,
    export_physics_evidence_scene,
    generate_physics_evidence,
    select_frame_indices,
    upgrade_replay_for_presentation,
)


class PhysicsEvidenceTests(unittest.TestCase):
    def test_clock_replay_declares_only_transmission_wheels_for_arrows(self) -> None:
        family = {
            "mechanism_kind": "spur_gear_clock",
            "bodies": [
                {
                    "body_id": body_id,
                    "fixed": False,
                    "center_abs_m": [index * 0.02, 0.0, 0.0],
                }
                for index, body_id in enumerate(
                    ("barrel", "center", "balance", "hammer", "setting")
                )
            ],
            "shafts": [
                {
                    "shaft_id": f"shaft_{body_id}",
                    "body_id": body_id,
                    "origin_abs_m": [index * 0.02, 0.0, 0.0],
                    "axis_abs": [0.0, 1.0, 0.0],
                }
                for index, body_id in enumerate(
                    ("barrel", "center", "balance", "hammer", "setting")
                )
            ],
            "gear_meshes": [
                {
                    "mesh_id": "barrel_center",
                    "driver_body_id": "barrel",
                    "driven_body_id": "center",
                }
            ],
            "setting_clutch": {
                "clutch_id": "setting_center",
                "input_body_id": "setting",
                "driven_body_id": "center",
            },
        }

        annotations, edges = _clock_transmission_visuals(family)

        self.assertEqual(
            {item["body_id"] for item in annotations},
            {"barrel", "center"},
        )
        self.assertNotIn("balance", {item["body_id"] for item in annotations})
        self.assertNotIn("hammer", {item["body_id"] for item in annotations})
        self.assertEqual(
            {(item["kind"], item["first_body_id"], item["second_body_id"]) for item in edges},
            {
                ("gear_mesh", "barrel", "center"),
                ("clutch", "setting", "center"),
            },
        )
        clutch = next(item for item in edges if item["kind"] == "clutch")
        self.assertEqual(clutch["engaged_field"], "setting_clutch_engaged")

    def test_clock_shafts_become_spatial_reaction_landmarks(self) -> None:
        family = {
            "mechanism_kind": "spur_gear_clock",
            "bodies": [
                {"body_id": "ground", "fixed": True, "center_abs_m": [0.0, 0.004, 0.125]},
                {"body_id": "barrel", "fixed": False, "center_abs_m": [-0.06, 0.008, 0.07]},
            ],
            "shafts": [
                {
                    "shaft_id": "shaft_barrel",
                    "body_id": "barrel",
                    "origin_abs_m": [-0.06, 0.008, 0.07],
                    "axis_abs": [0.0, 1.0, 0.0],
                }
            ],
        }
        self.assertEqual(
            _landmarks(family),
            [
                {
                    "id": "shaft_barrel",
                    "kind": "shaft_support",
                    "parent_body_id": "ground",
                    "child_body_id": "barrel",
                    "origin_abs_m": [-0.06, 0.008, 0.07],
                    "origin_parent_local_m": [-0.06, 0.004, -0.055],
                    "axis_abs": [0.0, 1.0, 0.0],
                }
            ],
        )

    def test_frame_selection_keeps_boundaries_and_physical_extrema(self) -> None:
        samples = []
        for index in range(400):
            phase = "a" if index < 100 else "b" if index < 250 else "c"
            samples.append(
                {
                    "time_s": index / 100.0,
                    "phase_id": phase,
                    "contact_count": 20 if index == 345 else 0,
                    "max_bushing_reaction_force_n": 90.0 if index == 123 else 0.0,
                    "joint_drifts_m": {"j": 0.02 if index == 111 else 0.0},
                    "applied_force_abs_n": [80.0 if index == 234 else 0.0, 0.0, 0.0],
                }
            )

        selected = select_frame_indices(
            samples,
            max_frames=180,
            required_source_indices=(77, 333),
        )

        self.assertLessEqual(len(selected), 180)
        self.assertTrue({77, 333}.issubset(selected))
        self.assertTrue({0, 99, 100, 111, 123, 234, 249, 250, 345, 399}.issubset(selected))

    def test_branch_payload_marks_per_joint_reactions_without_changing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            trace_path = workspace / "physical.trace.json"
            trace = {
                "branch": "physical",
                "status": "COMPLETED",
                "samples": [
                    {
                        "time_s": 0.0,
                        "phase_id": "load",
                        "body_poses": {},
                        "applied_force_abs_n": [1.0, 0.0, 0.0],
                        "bushing_reaction_forces_n": {"hinge": 4.5},
                        "max_bushing_reaction_force_n": 4.5,
                    }
                ],
            }
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            payload = _branch_payload(workspace, trace_path, trace)

            self.assertTrue(payload["field_availability"]["per_joint_bushing_reaction"])
            self.assertEqual(payload["frames"][0]["bushing_reaction_forces_n"], {"hinge": 4.5})
            self.assertEqual(payload["source_sample_count"], 1)

    def test_clock_frame_keeps_dynamic_inputs_and_gear_angular_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            trace_path = workspace / "physical.trace.json"
            trace = {
                "branch": "physical",
                "status": "COMPLETED",
                "samples": [
                    {
                        "time_s": 0.1,
                        "phase_id": "ring",
                        "body_poses": {},
                        "applied_torque_abs_nm": [0.0, 0.08, 0.0],
                        "active_input_body_id": "barrel",
                        "active_input_point_abs_m": [-0.06, 0.008, 0.07],
                        "additional_alarm_torque_nm": -0.03,
                        "body_angular_velocities_abs_rad_s": {
                            "barrel": [0.0, 2.0, 0.0],
                            "alarm_barrel": [0.0, -4.0, 0.0],
                        },
                    }
                ],
            }
            family = {
                "input_body_id": "barrel",
                "alarm_input_body_id": "alarm_barrel",
                "alarm_input_point_abs_m": [-0.071, 0.008, 0.14],
                "alarm_input_axis_abs": [0.0, 1.0, 0.0],
            }
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            payload = _branch_payload(
                workspace,
                trace_path,
                trace,
                family=family,
            )
            frame = payload["frames"][0]

            self.assertEqual(frame["active_input_body_id"], "barrel")
            self.assertEqual(len(frame["input_loads"]), 2)
            self.assertEqual(
                {item["body_id"] for item in frame["input_loads"]},
                {"barrel", "alarm_barrel"},
            )
            self.assertEqual(
                frame["body_angular_velocities_abs_rad_s"]["alarm_barrel"],
                [0.0, -4.0, 0.0],
            )
            self.assertTrue(
                payload["field_availability"]["per_body_angular_velocity"]
            )

    def test_branch_payload_keeps_hash_bound_moving_chain_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            trace_path = workspace / "physical.trace.json"
            contract = {
                "schema_version": "bicycle_moving_chain_path_contract_v1",
                "contract_sha256": "a" * 64,
                "centerline_points_abs_m": [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "edge_region_ids": [
                    "rigid.tangent_span",
                    "flex.input_sprocket",
                ],
                "cumulative_lengths_m": [0.0, 0.1, 0.2],
                "total_length_m": 0.2,
                "visual_marker_count": 2,
                "visual_marker_spacing_m": 0.1,
                "chain_collision_radius_m": 0.001,
            }
            state = {
                "schema_version": "bicycle_moving_chain_path_state_v1",
                "contract_sha256": "a" * 64,
                "state_sha256": "b" * 64,
                "input_angle_rad": 1.0,
                "output_angle_rad": 2.0,
                "input_surface_displacement_m": 0.04,
                "output_surface_displacement_m": 0.04,
                "force_coordinate_error_m": 0.0,
                "phase_unwrapped_m": 0.04,
                "phase_modulo_m": 0.04,
                "input_phase_residual_m": 0.0,
                "output_phase_residual_m": 0.0,
            }
            trace = {
                "branch": "physical",
                "status": "COMPLETED",
                "moving_chain_path_contract": contract,
                "samples": [
                    {
                        "time_s": 0.1,
                        "phase_id": "powered",
                        "body_poses": {},
                        "moving_chain_path_state": state,
                    }
                ],
            }
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            payload = _branch_payload(workspace, trace_path, trace)

            self.assertEqual(payload["moving_chain_path_contract"], contract)
            self.assertEqual(
                payload["frames"][0]["moving_chain_path_state"], state
            )
            self.assertTrue(
                payload["field_availability"]["moving_chain_path_state"]
            )

    def test_saved_v2_clock_replay_is_upgraded_in_memory_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            trace_path = workspace / "run" / "traces" / "clock" / "task" / "physical.trace.json"
            trace = {
                "branch": "physical",
                "status": "COMPLETED",
                "samples": [
                    {
                        "time_s": 0.1,
                        "phase_id": "setting",
                        "body_poses": {},
                        "applied_torque_abs_nm": [0.0, 0.08, 0.0],
                        "active_input_body_id": "setting",
                        "active_input_point_abs_m": [0.0, 0.05, 0.12],
                        "setting_clutch_engaged": True,
                        "body_angular_velocities_abs_rad_s": {
                            "setting": [0.0, 1.0, 0.0],
                            "minute": [0.0, 1.0, 0.0],
                        },
                    }
                ],
            }
            trace_path.parent.mkdir(parents=True)
            trace_path.write_text(json.dumps(trace), encoding="utf-8")
            trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
            replay = {
                "schema_version": "physics_evidence_replay_v2",
                "branches": {
                    "physical": {
                        "source_trace": trace_path.relative_to(workspace).as_posix(),
                        "source_trace_sha256": trace_sha256,
                        "field_availability": {},
                        "frames": [
                            {
                                "source_sample_index": 0,
                                "body_poses": {},
                                "external_load": {
                                    "kind": "torque",
                                    "vector_abs": [0.0, 0.08, 0.0],
                                    "magnitude": 0.08,
                                    "unit": "N·m",
                                },
                            }
                        ],
                    }
                },
                "claims": {"simulation_was_rerun_for_replay": False},
            }
            family = {
                "mechanism_kind": "spur_gear_clock",
                "bodies": [
                    {"body_id": "setting", "center_abs_m": [0.0, 0.05, 0.12]},
                    {"body_id": "minute", "center_abs_m": [0.0, 0.0, 0.12]},
                ],
                "shafts": [
                    {
                        "shaft_id": "shaft_minute",
                        "body_id": "minute",
                        "origin_abs_m": [0.0, 0.0, 0.12],
                        "axis_abs": [0.0, 1.0, 0.0],
                    }
                ],
                "gear_meshes": [
                    {
                        "mesh_id": "minute_hour",
                        "driver_body_id": "minute",
                        "driven_body_id": "hour",
                    }
                ],
                "setting_clutch": {
                    "clutch_id": "setting_minute",
                    "input_body_id": "setting",
                    "driven_body_id": "minute",
                },
            }
            upgraded = upgrade_replay_for_presentation(
                replay,
                workspace_root=workspace,
                plan={"adapter_spec": {"family_spec": family}},
            )

            self.assertEqual(replay["schema_version"], "physics_evidence_replay_v2")
            self.assertEqual(upgraded["schema_version"], "physics_evidence_replay_v4")
            self.assertTrue(
                upgraded["presentation_overlay"]["derived_from_saved_trace"]
            )
            self.assertFalse(
                upgraded["presentation_overlay"]["simulation_was_rerun"]
            )
            frame = upgraded["branches"]["physical"]["frames"][0]
            self.assertEqual(frame["active_input_body_id"], "setting")
            self.assertTrue(frame["setting_clutch_engaged"])
            self.assertEqual(
                frame["body_angular_velocities_abs_rad_s"]["minute"],
                [0.0, 1.0, 0.0],
            )

    def test_scene_is_bound_to_original_per_component_meshes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / "run"
            mesh_path = workspace / "collision" / "solid_1.stl"
            mesh_path.parent.mkdir(parents=True)
            trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(mesh_path)
            collision = {
                "source": "original_step_solids",
                "component_count": 1,
                "convex_hull_used": False,
                "whole_body_convex_hull_used": False,
                "whole_body_box_proxy_used": False,
                "bodies": [
                    {
                        "body_id": "moving",
                        "fixed": False,
                        "initial_center_abs_m": [0.0, 0.0, 0.0],
                        "initial_rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "components": [
                            {
                                "solid_id": "solid_1",
                                "mesh": "collision/solid_1.stl",
                                "representation": "ChCollisionShapeTriangleMeshDynamicConvex",
                                "convex": True,
                            }
                        ],
                    }
                ],
            }
            output = run / "visual" / "physics.glb"

            manifest = export_physics_evidence_scene(
                workspace_root=workspace,
                run_dir=run,
                collision=collision,
                output=output,
                capsule_id="demo",
            )

            self.assertTrue(_scene_manifest_is_current(workspace, output, manifest, collision))
            self.assertFalse(manifest["convex_hull_used"])
            self.assertEqual(manifest["component_count"], 1)
            mesh_path.write_bytes(mesh_path.read_bytes() + b"changed")
            self.assertFalse(_scene_manifest_is_current(workspace, output, manifest, collision))

    def test_scene_keeps_every_component_of_one_decomposed_solid_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / "run"
            collision_dir = workspace / "collision"
            collision_dir.mkdir(parents=True)
            first = collision_dir / "segment.001.stl"
            second = collision_dir / "segment_002.stl"
            trimesh.creation.box(extents=[0.1, 0.02, 0.03]).export(first)
            trimesh.creation.box(extents=[0.08, 0.02, 0.04]).export(second)
            collision = {
                "source": "original_step_solids",
                "component_count": 2,
                "convex_hull_used": False,
                "whole_body_convex_hull_used": False,
                "whole_body_box_proxy_used": False,
                "bodies": [
                    {
                        "body_id": "moving",
                        "fixed": False,
                        "initial_center_abs_m": [0.0, 0.0, 0.0],
                        "initial_rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                        "components": [
                            {
                                "solid_id": "solid.1",
                                "mesh": "collision/segment.001.stl",
                                "representation": "convex",
                                "convex": True,
                            },
                            {
                                "solid_id": "solid_1",
                                "mesh": "collision/segment_002.stl",
                                "representation": "convex",
                                "convex": True,
                            },
                        ],
                    }
                ],
            }
            output = run / "visual" / "physics.glb"

            manifest = export_physics_evidence_scene(
                workspace_root=workspace,
                run_dir=run,
                collision=collision,
                output=output,
                capsule_id="demo",
            )

            nodes = [
                item["node"]
                for item in manifest["bodies"][0]["components"]
            ]
            exported = trimesh.load(output, force="scene", process=False)
            self.assertEqual(len(nodes), 2)
            self.assertEqual(len(set(nodes)), 2)
            self.assertTrue(all(not set("[].:/") & set(node) for node in nodes))
            self.assertEqual(len(exported.graph.nodes_geometry), 2)
            self.assertEqual(
                manifest["component_node_binding"],
                "unique_per_collision_component_v3",
            )


    def test_generate_accepts_legacy_two_body_trace_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / "run"
            trace_dir = run / "traces" / "legacy_demo" / "nominal"
            collision_dir = workspace / "collision"
            trace_dir.mkdir(parents=True)
            collision_dir.mkdir(parents=True)
            fixed_path = collision_dir / "fixed.obj"
            moving_path = collision_dir / "moving.obj"
            trimesh.creation.box(extents=[0.2, 0.2, 0.02]).export(fixed_path)
            trimesh.creation.box(extents=[0.04, 0.04, 0.04]).export(moving_path)

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            collision = {
                "fixed": {"body_origin_abs_m": [0.0, 0.0, 0.1]},
                "fixed_components": [
                    {
                        "derived": "collision/fixed.obj",
                        "derived_sha256": digest(fixed_path),
                        "collision_representation": "ChCollisionShapeTriangleMesh(static)",
                    }
                ],
                "moving_components": [
                    {
                        "derived": "collision/moving.obj",
                        "derived_sha256": digest(moving_path),
                        "collision_representation": "ChCollisionShapeBox",
                    }
                ],
            }
            samples = [
                {
                    "time_s": 0.0,
                    "phase_id": "start",
                    "moving_pose": {
                        "position_abs_m": [0.0, 0.0, 0.2],
                        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
                {
                    "time_s": 0.1,
                    "phase_id": "move",
                    "moving_pose": {
                        "position_abs_m": [0.01, 0.0, 0.2],
                        "rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            ]
            for branch in ("physical", "reduced"):
                trace = {
                    "branch": branch,
                    "status": "COMPLETED",
                    "engine": {"name": "saved-test-trace"},
                    "collision_geometry": collision,
                    "input_point_abs_m": [0.0, 0.0, 0.2],
                    "samples": samples,
                }
                (trace_dir / f"{branch}.trace.json").write_text(
                    json.dumps(trace), encoding="utf-8"
                )
            plan = {
                "capsule_id": "legacy_demo",
                "adapter_spec": {"family_spec": None},
            }
            penetration_path = run / "results" / "penetration_evidence_v1.json"
            penetration_path.parent.mkdir(parents=True)
            penetration_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "load_case_id": "nominal",
                                "source_sample_index": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = generate_physics_evidence(
                workspace_root=workspace,
                run_dir=run,
                plan=plan,
                force=True,
            )

            self.assertEqual(result["status"], "GENERATED")
            self.assertEqual(set(result["replays"]), {"nominal"})
            namespaced = generate_physics_evidence(
                workspace_root=workspace,
                run_dir=run,
                plan=plan,
                force=True,
                namespace_assets=True,
            )
            self.assertTrue(
                (
                    run
                    / "visual"
                    / "stage5_physics_evidence__legacy_demo.glb"
                ).is_file()
            )
            self.assertTrue(
                (
                    run
                    / "visual"
                    / "stage5_physics_evidence__legacy_demo__nominal.json"
                ).is_file()
            )
            self.assertIn("legacy_demo__nominal", namespaced["replays"]["nominal"]["path"])
            replay = json.loads(
                (run / "visual" / "stage5_physics_evidence__nominal.json").read_text(
                    encoding="utf-8"
                )
            )
            first = replay["branches"]["physical"]["frames"][0]
            self.assertEqual(
                set(first["body_poses"]), {"fixed_boundary", "moving_group"}
            )
            self.assertEqual(
                replay["collision"]["representation_counts"]["ChCollisionShapeBox"],
                1,
            )
            original_penetration_sha = replay["provenance"][
                "penetration_evidence_sources"
            ][0]["sha256"]
            penetration_path.write_text(
                json.dumps({"events": []}), encoding="utf-8"
            )
            generate_physics_evidence(
                workspace_root=workspace,
                run_dir=run,
                plan=plan,
                force=False,
            )
            replay = json.loads(
                (run / "visual" / "stage5_physics_evidence__nominal.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotEqual(
                replay["provenance"]["penetration_evidence_sources"][0]["sha256"],
                original_penetration_sha,
            )

            (trace_dir / "reduced.trace.json").unlink()
            physical_only = generate_physics_evidence(
                workspace_root=workspace,
                run_dir=run,
                plan=plan,
                force=True,
            )
            replay = json.loads(
                (run / "visual" / "stage5_physics_evidence__nominal.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(set(replay["branches"]), {"physical"})
            self.assertTrue(
                physical_only["replays"]["nominal"]["physical_only_replay"]
            )


if __name__ == "__main__":
    unittest.main()
