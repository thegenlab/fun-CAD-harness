from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import (
    ReadContext,
    _comparison_checks,
    _load_case,
    _react_verdict,
    _react_loop_state_path,
    _task_execution_coverage,
    _visual_metadata,
    compare_revision_geometry,
    group_runs_by_project,
    normalize_run,
    revision_geometry_inventory,
    scan_runs,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.path_resolution import (
    resolve_workspace_reference,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.render import (
    render_snapshot,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ReactLoopStatePathTests(unittest.TestCase):
    def test_repeated_error_pause_is_blocked(self) -> None:
        verdict = _react_verdict(
            {"status": "REPEATED_ERROR_PAUSED"},
            {"status": "running", "final": False},
        )

        self.assertEqual(verdict["status"], "blocked")
        self.assertFalse(verdict["final"])
        self.assertEqual(verdict["reason_code"], "REPEATED_ERROR_PAUSED")

    def _project(self, root: Path, manifest: dict[str, object]) -> Path:
        project = root / ".runtime_runs" / "projects" / "machine"
        _write_json(
            project / "react_project.json",
            {
                "schema_version": "simulation_ready_cad_react_project_v2",
                "workflow_profile": "single_conversation_react_v2",
                **manifest,
            },
        )
        return project

    def test_dashboard_prefers_loop_state_manifest_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self._project(
                root,
                {
                    "loop_state": (
                        ".runtime_runs/projects/machine/custom/state.json"
                    ),
                    "controller_state": (
                        ".runtime_runs/projects/machine/legacy/state.json"
                    ),
                },
            )
            self.assertEqual(
                _react_loop_state_path(root, project),
                (project / "custom" / "state.json").resolve(),
            )

    def test_dashboard_reads_legacy_controller_state_key(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = self._project(
                root,
                {
                    "controller_state": (
                        ".runtime_runs/projects/machine/legacy/state.json"
                    )
                },
            )
            self.assertEqual(
                _react_loop_state_path(root, project),
                (project / "legacy" / "state.json").resolve(),
            )


def _minimal_workspace(root: Path) -> tuple[Path, Path, Path, Path]:
    package = root / "package"
    package.mkdir()
    _write_json(package / "package.json", {"package_id": "fixture", "design_version": "1"})
    (package / "assembly.step").write_text("ISO-10303-21;END-ISO-10303-21;", encoding="ascii")
    _write_json(package / "assembly_contract.json", {"schema_version": "0.1"})
    intake = root / "intake.json"
    _write_json(
        intake,
        {
            "status": "ACCEPTED",
            "summary": {"error_count": 0, "warning_count": 0},
            "capabilities": {
                "can_enter_physical_preflight": True,
                "can_attempt_reduced_certification": True,
                "can_run_whole_machine_task_test": True,
            },
            "package_summary": {"step_solid_count": 1, "package_file_count": 3},
            "issues": [],
        },
    )
    run = root / "runs" / "fixture-run"
    capsule = "fixture_slide"
    plan = {
        "capsule_id": capsule,
        "family": "prismatic",
        "package_dir": str(package),
        "intake_report": str(intake),
        "fixed_group_ids": ["fixed"],
        "moving_group_ids": ["moving"],
        "task_ids": ["open", "close"],
    }
    _write_json(run / "plans" / "plan.json", plan)
    _write_json(
        run / "derived_adapters" / capsule / "provenance.json",
        {
            "fixed": {"solid_ids": ["fixed"]},
            "moving": {"solid_ids": ["moving"]},
            "convex_hull_used": False,
        },
    )
    _write_json(run / "queue_run" / "queue_state.json", {"status": "SUCCEEDED", "summary": {"total": 0, "SUCCEEDED": 0}})
    result = run / "results" / capsule
    certificate = result / "certificate.json"
    replacement = result / "replacement.json"
    whole = result / "whole_machine" / "result.json"
    return run, certificate, replacement, whole


class DashboardModelTests(unittest.TestCase):
    def test_task_execution_coverage_is_scoped_by_capsule(self) -> None:
        planning = {
            "test_plan": [
                {
                    "test_id": "steering",
                    "goal": "turn the handlebar",
                    "required_branches": ["physical", "ideal_reference"],
                    "target": {
                        "capsule_id": "steering_capsule",
                        "load_case_id": "task_roundtrip",
                        "metric_id": "angle_span_rad",
                    },
                }
            ]
        }
        load_cases = [
            {
                "capsule_id": "drivetrain_capsule",
                "id": "task_roundtrip",
                "status": "passed",
                "comparison_status": "passed",
                "ideal_kind": "ideal_reference",
                "physical": {"status": "passed"},
                "reduced": {"status": "passed"},
            },
            {
                "capsule_id": "steering_capsule",
                "id": "task_roundtrip",
                "status": "failed",
                "comparison_status": "failed",
                "ideal_kind": "ideal_reference",
                "physical": {"status": "failed"},
                "reduced": {"status": "passed"},
            },
        ]

        coverage = _task_execution_coverage(planning, load_cases)

        self.assertEqual(len(coverage), 1)
        self.assertEqual(coverage[0]["capsule_id"], "steering_capsule")
        self.assertEqual(coverage[0]["status"], "failed")

    def test_task_execution_coverage_requires_the_exact_ideal_branch(self) -> None:
        coverage = _task_execution_coverage(
            {
                "test_plan": [
                    {
                        "test_id": "reference_required",
                        "goal": "compare with a deterministic reference",
                        "required_branches": ["physical", "ideal_reference"],
                        "target": {
                            "capsule_id": "drive",
                            "load_case_id": "task",
                        },
                    }
                ]
            },
            [
                {
                    "capsule_id": "drive",
                    "id": "task",
                    "status": "passed",
                    "comparison_status": "passed",
                    "ideal_kind": "ideal_dynamics",
                    "physical": {"status": "passed"},
                    "reduced": {"status": "passed"},
                }
            ],
        )

        self.assertEqual(coverage[0]["status"], "evidence_incomplete")
        self.assertEqual(
            coverage[0]["branch_status"]["ideal_reference"],
            "not_started",
        )

    def test_multi_capsule_visual_names_retain_scope(self) -> None:
        physics = _visual_metadata(
            Path("stage5_physics_evidence__steering__task_roundtrip.json")
        )
        chart = _visual_metadata(
            Path("stage6_compare_steering__task_roundtrip.png")
        )
        whole = _visual_metadata(
            Path("stage8_whole_machine__steering.mp4")
        )
        clearance = _visual_metadata(
            Path("stage4_side_clearance__drawer_one.png")
        )

        self.assertEqual(physics["capsule_id"], "steering")
        self.assertEqual(physics["load_case_id"], "task_roundtrip")
        self.assertEqual(chart["capsule_id"], "steering")
        self.assertEqual(chart["load_case_id"], "task_roundtrip")
        self.assertEqual(whole["capsule_id"], "steering")
        self.assertEqual(clearance["capsule_id"], "drawer_one")

    def test_revision_inventory_distinguishes_same_solid_sidecar_components(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "r01"
            _write_json(
                run
                / "derived_adapters"
                / "bicycle"
                / "family_collision"
                / "provenance.json",
                {
                    "schema": "step_per_solid_body_local_v1",
                    "assembly_sha256": "assembly",
                    "bodies": [
                        {
                            "body_id": "input_crank",
                            "fixed": False,
                            "components": [
                                {
                                    "component_id": "input.shaft",
                                    "solid_id": "solid_2",
                                    "sha256": "shaft",
                                },
                                {
                                    "component_id": "input.chainring_backing",
                                    "solid_id": "solid_2",
                                    "sha256": "chainring",
                                },
                            ],
                        }
                    ],
                },
            )

            inventory = revision_geometry_inventory(run)

            self.assertTrue(inventory["available"])
            self.assertEqual(inventory["duplicate_keys"], [])
            self.assertEqual(
                [item["key"] for item in inventory["components"]],
                [
                    "bicycle/input_crank/input.chainring_backing",
                    "bicycle/input_crank/input.shaft",
                ],
            )
            self.assertEqual(
                {item["solid_id"] for item in inventory["components"]},
                {"solid_2"},
            )

    def test_revision_geometry_comparison_uses_component_keys_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "r01"
            target = root / "r02"

            def write_provenance(
                run: Path,
                components: list[tuple[str, str]],
            ) -> None:
                _write_json(
                    run
                    / "derived_adapters"
                    / "fixture_slide"
                    / "family_collision"
                    / "provenance.json",
                    {
                        "schema": "step_per_solid_body_local_v1",
                        "assembly_sha256": f"assembly-{run.name}",
                        "bodies": [
                            {
                                "body_id": "moving",
                                "fixed": False,
                                "components": [
                                    {
                                        "solid_id": solid_id,
                                        "sha256": sha256,
                                        "representation": "ChCollisionShapeBox",
                                    }
                                    for solid_id, sha256 in components
                                ],
                            }
                        ],
                    },
                )

            write_provenance(
                base,
                [("solid_1", "same"), ("solid_2", "before"), ("solid_3", "removed")],
            )
            write_provenance(
                target,
                [("solid_1", "same"), ("solid_2", "after"), ("solid_4", "added")],
            )

            inventory = revision_geometry_inventory(base)
            comparison = compare_revision_geometry(base, target)

            self.assertTrue(inventory["available"])
            self.assertEqual(inventory["component_count"], 3)
            self.assertEqual(
                comparison["summary"],
                {"added": 1, "removed": 1, "modified": 1, "unchanged": 1},
            )
            self.assertEqual(
                comparison["added"][0]["key"],
                "fixture_slide/moving/solid_4",
            )
            self.assertEqual(
                comparison["removed"][0]["key"],
                "fixture_slide/moving/solid_3",
            )
            self.assertEqual(
                comparison["modified"][0]["key"],
                "fixture_slide/moving/solid_2",
            )

    def test_revision_geometry_comparison_includes_placement_and_mesh_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "r01"
            target = root / "r02"

            def write_provenance(
                run: Path,
                *,
                center: list[float],
                linear_deflection: float,
            ) -> None:
                _write_json(
                    run
                    / "derived_adapters"
                    / "fixture_slide"
                    / "family_collision"
                    / "provenance.json",
                    {
                        "schema": "step_per_solid_body_local_v1",
                        "mesh_linear_deflection_m": linear_deflection,
                        "mesh_angular_deflection_rad": 0.5,
                        "bodies": [
                            {
                                "body_id": "moving",
                                "fixed": False,
                                "initial_center_abs_m": center,
                                "initial_rotation_abs_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
                                "components": [
                                    {
                                        "solid_id": "solid_1",
                                        "sha256": "identical-mesh",
                                        "representation": "ChCollisionShapeBox",
                                        "body_local": True,
                                    }
                                ],
                            }
                        ],
                    },
                )

            write_provenance(base, center=[0.0, 0.0, 0.0], linear_deflection=0.0002)
            write_provenance(target, center=[0.1, 0.0, 0.0], linear_deflection=0.0002)

            placement_comparison = compare_revision_geometry(base, target)

            self.assertTrue(placement_comparison["available"])
            self.assertEqual(placement_comparison["summary"]["modified"], 1)

            write_provenance(target, center=[0.1, 0.0, 0.0], linear_deflection=0.001)
            profile_mismatch = compare_revision_geometry(base, target)

            self.assertFalse(profile_mismatch["available"])

    def test_scan_does_not_publish_quarantined_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visible = root / ".runtime_runs" / "visible"
            quarantined = (
                root
                / ".runtime_runs"
                / "quarantine"
                / "review"
                / "removed-pass"
            )
            _write_json(
                visible / "plans" / "plan.json",
                {"package_id": "visible", "design_version": "1"},
            )
            _write_json(visible / "run_status.json", {"status": "FAILED"})
            _write_json(
                quarantined / "plans" / "plan.json",
                {"package_id": "removed", "design_version": "1"},
            )
            _write_json(
                quarantined / "run_status.json",
                {"status": "WHOLE_MACHINE_TASK_PASSED"},
            )

            rows = scan_runs(
                root / ".runtime_runs",
                workspace_root=root,
                include_history=True,
            )

            self.assertEqual([item["package_id"] for item in rows], ["visible"])

    def test_scan_prunes_candidate_regression_fixture_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visible = root / ".runtime_runs" / "visible"
            _write_json(
                visible / "plans" / "plan.json",
                {"package_id": "visible", "design_version": "1"},
            )
            _write_json(visible / "run_status.json", {"status": "FAILED"})
            for internal_root in (
                "candidate_release_regression",
                "abandoned_candidate_release_regression",
            ):
                fixture = (
                    root
                    / ".runtime_runs"
                    / internal_root
                    / "candidate"
                    / "runner"
                    / "attempt.tmp"
                    / ".runtime_runs"
                    / "projects"
                    / "fake"
                    / "runs"
                    / "r01"
                )
                _write_json(
                    fixture / "plans" / "plan.json",
                    {"package_id": "fake", "design_version": "1"},
                )
                _write_json(
                    fixture / "run_status.json",
                    {"status": "WHOLE_MACHINE_TASK_PASSED"},
                )

            rows = scan_runs(
                root / ".runtime_runs",
                workspace_root=root,
                include_history=True,
            )

            self.assertEqual([item["package_id"] for item in rows], ["visible"])

    def test_scan_does_not_walk_hf_sync_or_preview_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            visible = root / ".runtime_runs" / "visible"
            imported = (
                root
                / ".runtime_runs"
                / "hf_sync"
                / "imported"
                / "owner"
                / "machine"
                / "snap"
                / ".runtime_runs"
                / "projects"
                / "machine"
                / "runs"
                / "r01"
            )
            preview = (
                root
                / ".runtime_runs"
                / "dashboard_previews"
                / "machine"
                / "r01"
            )
            _write_json(
                visible / "plans" / "plan.json",
                {"package_id": "visible", "design_version": "1"},
            )
            _write_json(visible / "run_status.json", {"status": "FAILED"})
            for hidden in (imported, preview):
                _write_json(
                    hidden / "plans" / "plan.json",
                    {"package_id": "hidden", "design_version": "1"},
                )
                _write_json(hidden / "run_status.json", {"status": "WHOLE_MACHINE_TASK_PASSED"})

            rows = scan_runs(
                root / ".runtime_runs",
                workspace_root=root,
                include_history=True,
            )

            self.assertEqual([item["package_id"] for item in rows], ["visible"])

    def test_superseded_revision_is_stale_in_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = (
                root
                / ".runtime_runs"
                / "projects"
                / "clock"
                / "runs"
                / "r01"
            )
            _write_json(
                run / "run_manifest.json",
                {
                    "schema_version": "simulation_ready_cad_run_v1",
                    "project_id": "clock",
                    "revision": "r01",
                    "run_id": "clock-r01",
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "schema_version": "0.1",
                    "status": "SUPERSEDED",
                    "superseded_by": "clock-r02",
                },
            )

            rows = scan_runs(
                root / ".runtime_runs",
                workspace_root=root,
                include_history=True,
                include_active=True,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["revision"], "r01")
            self.assertEqual(rows[0]["status"], "stale")
            self.assertEqual(
                rows[0]["effective_status_raw"],
                "SUPERSEDED",
            )

    def test_scan_records_revision_preview_without_guessing_missing_media(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = (
                root
                / ".runtime_runs"
                / "projects"
                / "preview_machine"
                / "runs"
            )
            for revision in ("r01", "r02", "r03"):
                run = runs_root / revision
                _write_json(
                    run / "run_manifest.json",
                    {
                        "schema_version": "simulation_ready_cad_run_v1",
                        "project_id": "preview_machine",
                        "revision": revision,
                        "run_id": f"preview_machine-{revision}",
                    },
                )
                _write_json(
                    run / "run_status.json",
                    {
                        "status": "SUPERSEDED",
                        "superseded_by": "preview_machine-r04",
                    },
                )
                _write_json(
                    run / "plans" / "plan.json",
                    {
                        "capsule_id": "preview_fixture",
                        "package_id": "preview_machine",
                        "design_version": "1",
                    },
                )
            iso = runs_root / "r01" / "design_quality" / "assets" / "01_iso.png"
            iso.parent.mkdir(parents=True)
            iso.write_bytes(b"iso")
            fallback = runs_root / "r02" / "visual" / "stage1_design_package.png"
            fallback.parent.mkdir(parents=True)
            fallback.write_bytes(b"fallback")

            rows = scan_runs(
                root / ".runtime_runs",
                workspace_root=root,
                include_history=True,
                include_active=True,
            )
            previews = {
                row["revision"]: row["revision_preview_path"]
                for row in rows
            }
            preview_hashes = {
                row["revision"]: row["revision_preview_sha256"]
                for row in rows
            }

            self.assertEqual(
                previews["r01"],
                ".runtime_runs/projects/preview_machine/runs/r01/design_quality/assets/01_iso.png",
            )
            self.assertEqual(
                previews["r02"],
                ".runtime_runs/projects/preview_machine/runs/r02/visual/stage1_design_package.png",
            )
            self.assertEqual(previews["r03"], "")
            self.assertEqual(
                preview_hashes["r01"],
                hashlib.sha256(b"iso").hexdigest(),
            )
            self.assertEqual(
                preview_hashes["r02"],
                hashlib.sha256(b"fallback").hexdigest(),
            )
            self.assertEqual(preview_hashes["r03"], "")

            previous_mtime_ns = iso.stat().st_mtime_ns
            iso.write_bytes(b"ISO")
            os.utime(
                iso,
                ns=(previous_mtime_ns + 1_000_000, previous_mtime_ns + 1_000_000),
            )
            refreshed = scan_runs(
                root / ".runtime_runs",
                workspace_root=root,
                include_history=True,
                include_active=True,
            )
            refreshed_r01 = next(
                row for row in refreshed if row["revision"] == "r01"
            )
            self.assertEqual(refreshed_r01["revision_preview_path"], previews["r01"])
            self.assertEqual(
                refreshed_r01["revision_preview_sha256"],
                hashlib.sha256(b"ISO").hexdigest(),
            )

    def test_groups_revision_history_under_one_machine_project(self) -> None:
        rows = [
            {
                "id": ".runtime_runs/projects/clock/runs/r02",
                "project_id": "clock",
                "package_id": "clock",
                "revision": "r02",
                "package_version": "1.1.2",
                "status": "passed",
                "status_label": "通过",
                "updated_at": "2026-07-23T02:00:00+00:00",
            },
            {
                "id": ".runtime_runs/projects/wiper/runs/r01",
                "project_id": "wiper",
                "package_id": "wiper",
                "revision": "r01",
                "package_version": "2.0.0",
                "status": "passed",
                "status_label": "通过",
                "updated_at": "2026-07-23T03:00:00+00:00",
            },
            {
                "id": ".runtime_runs/projects/clock/runs/r10",
                "project_id": "clock",
                "package_id": "clock",
                "revision": "r10",
                "package_version": "1.1.10",
                "status": "running",
                "status_label": "进行中",
                "healthy_running": True,
                "updated_at": "2026-07-23T04:00:00+00:00",
            },
        ]

        projects = group_runs_by_project(rows)

        self.assertEqual([item["id"] for item in projects], ["clock", "wiper"])
        self.assertEqual(projects[0]["version_count"], 2)
        self.assertEqual(projects[0]["current_run_id"], rows[2]["id"])
        self.assertEqual(
            [item["revision"] for item in projects[0]["runs"]],
            ["r10", "r02"],
        )
        self.assertEqual(projects[0]["published_run_id"], rows[0]["id"])
        self.assertEqual(projects[0]["last_pass_run_id"], rows[0]["id"])
        self.assertEqual(projects[0]["last_pass"]["status"], "passed")
        self.assertEqual(projects[0]["active_run_id"], rows[2]["id"])
        self.assertEqual(projects[0]["default_reason"], "active_newer_than_published")

    def test_project_keeps_published_pass_separate_from_newer_failed_attempt(self) -> None:
        rows = [
            {
                "id": ".runtime_runs/projects/tool/runs/r31",
                "project_id": "tool",
                "package_id": "tool",
                "revision": "r31",
                "status": "failed",
                "status_label": "失败",
                "verdict": {"authority": "terminal_run_status"},
                "updated_at": "2026-07-23T04:00:00+00:00",
            },
            {
                "id": ".runtime_runs/projects/tool/runs/r30",
                "project_id": "tool",
                "package_id": "tool",
                "revision": "r30",
                "status": "passed",
                "status_label": "通过",
                "verdict": {"authority": "terminal_run_status"},
                "updated_at": "2026-07-23T03:00:00+00:00",
            },
            {
                "id": ".runtime_runs/projects/tool/runs/r29",
                "project_id": "tool",
                "package_id": "tool",
                "revision": "r29",
                "status": "running",
                "status_label": "进行中",
                "verdict": {"authority": "active_run_status"},
                "healthy_running": True,
                "updated_at": "2026-07-23T05:00:00+00:00",
            },
        ]

        project = group_runs_by_project(rows)[0]

        self.assertEqual(project["published_run_id"], rows[1]["id"])
        self.assertEqual(project["last_pass_run_id"], rows[1]["id"])
        self.assertEqual(project["last_pass"]["status"], "passed")
        self.assertEqual(project["latest_run_id"], rows[0]["id"])
        self.assertEqual(project["default_run_id"], rows[1]["id"])
        self.assertEqual(project["status"], "passed")
        self.assertEqual(project["latest_status"], "failed")
        self.assertEqual(project["active_run_id"], rows[2]["id"])

    def test_scan_limit_protects_each_project_latest_and_published_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / ".runtime_runs"
            legacy_runs: list[Path] = []
            for revision in range(1, 252):
                run = runtime / f"busy-r{revision:03d}"
                legacy_runs.append(run)
                _write_json(
                    run / "plans" / "plan.json",
                    {"package_id": "busy", "design_version": str(revision)},
                )
                _write_json(run / "run_status.json", {"status": "FAILED"})
            cold = runtime / "cold-project-pass"
            _write_json(
                cold / "plans" / "plan.json",
                {"package_id": "cold", "design_version": "1"},
            )
            _write_json(
                cold / "run_status.json",
                {"status": "TASK_CERTIFIED"},
            )
            legacy_runs.append(cold)

            rows = scan_runs(
                runtime,
                workspace_root=root,
                include_history=True,
                limit=250,
            )
            projects = group_runs_by_project(rows)

            self.assertGreaterEqual(len(rows), 250)
            self.assertIn("cold", {item["id"] for item in projects})
            cold_project = next(item for item in projects if item["id"] == "cold")
            self.assertEqual(cold_project["status"], "passed")

    def test_whole_machine_continuation_inherits_registered_upstream_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream, certificate, replacement, _ = _minimal_workspace(root)
            _write_json(
                certificate,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                },
            )
            _write_json(replacement, {"status": "ACTIVE"})
            _write_json(
                upstream / "run_status.json",
                {
                    "status": "WHOLE_MACHINE_TASK_FAILED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate),
                            "replacement": str(replacement),
                        }
                    ],
                },
            )

            continuation = upstream.parent / "fixture-run-continuation"
            whole = (
                continuation
                / "results"
                / "fixture_slide"
                / "whole_machine"
                / "result.json"
            )
            _write_json(
                whole,
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "task_results": [{"task_id": "open", "passed": True}],
                },
            )
            _write_json(whole.parent / "execution_plan.json", {})
            _write_json(
                continuation / "run_status.json",
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "continuation": {
                        "mode": "cross_run_whole_machine_continuation_v1",
                        "upstream_run": upstream.relative_to(root).as_posix(),
                    },
                    "mechanisms": [{"capsule_id": "fixture_slide"}],
                    "whole_machine_results": [{"result_file": str(whole)}],
                },
            )

            data = normalize_run(continuation, workspace_root=root)
            stages = {item["id"]: item for item in data["stages"]}

            self.assertEqual(data["run"]["name"], continuation.name)
            self.assertEqual(stages["mechanism-plan"]["status"], "passed")
            self.assertEqual(data["overview"]["certificate"]["status"], "passed")
            self.assertEqual(stages["replacement"]["status"], "passed")
            self.assertEqual(stages["whole-machine"]["status"], "passed")
            self.assertEqual(data["whole_machine"]["task_passed"], 1)
            self.assertEqual(data["diagnostics"], [])

    def test_single_capsule_missing_certificate_remains_visible_in_stage_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate, replacement, _ = _minimal_workspace(root)
            _write_json(replacement, {"status": "ACTIVE"})
            _write_json(
                run / "run_status.json",
                {
                    "status": "TASK_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate),
                            "replacement": str(replacement),
                        }
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)

            stages = {item["id"]: item for item in data["stages"]}
            self.assertNotEqual(stages["certificate"]["status"], "passed")
            self.assertFalse(
                any(
                    item["role"] == "dynamic stage aggregation"
                    for item in data["diagnostics"]
                )
            )

    def test_multiple_whole_machine_results_are_scoped_and_aggregated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, first_certificate, first_replacement, first_whole = (
                _minimal_workspace(root)
            )
            second_capsule = "second_capsule"
            _write_json(
                run / "plans" / "second.json",
                {"capsule_id": second_capsule, "family": "revolute"},
            )
            for capsule in ("fixture_slide", second_capsule):
                _write_json(
                    run / "traces" / capsule / "task" / "physical.trace.json",
                    {"branch": "physical"},
                )
            second_certificate = (
                run / "results" / second_capsule / "certificate.json"
            )
            second_replacement = (
                run / "results" / second_capsule / "replacement.json"
            )
            second_whole = (
                run
                / "results"
                / second_capsule
                / "whole_machine"
                / "result.json"
            )
            for certificate in (first_certificate, second_certificate):
                _write_json(
                    certificate,
                    {
                        "verdict": "KINEMATIC_CERTIFIED",
                        "package": {"package_binding_sha256": "binding"},
                    },
                )
            for replacement in (first_replacement, second_replacement):
                _write_json(replacement, {"status": "ACTIVE"})
            for capsule, result in (
                ("fixture_slide", first_whole),
                (second_capsule, second_whole),
            ):
                _write_json(
                    result,
                    {
                        "status": "WHOLE_MACHINE_TASK_PASSED",
                        "task_results": [
                            {
                                "task_id": f"task_{capsule}",
                                "passed": True,
                                "summary": "passed",
                            }
                        ],
                    },
                )
                _write_json(
                    result.parent / "execution_plan.json",
                    {"capsule_id": capsule, "coordinate_kind": "task"},
                )
            _write_json(
                run / "run_status.json",
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(first_certificate),
                            "replacement": str(first_replacement),
                        },
                        {
                            "capsule_id": second_capsule,
                            "certificate": str(second_certificate),
                            "replacement": str(second_replacement),
                        },
                    ],
                    "whole_machine_results": [
                        {
                            "capsule_id": "fixture_slide",
                            "result_file": str(first_whole),
                        },
                        {
                            "capsule_id": second_capsule,
                            "result_file": str(second_whole),
                        },
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)

            self.assertEqual(data["workflow"]["aggregation_status"], "complete")
            self.assertEqual(data["workflow"]["reason_codes"], [])
            self.assertEqual(
                [item["capsule_id"] for item in data["whole_machines"]],
                ["fixture_slide", second_capsule],
            )
            self.assertEqual(data["whole_machine"]["task_passed"], 2)
            self.assertEqual(data["whole_machine"]["task_total"], 2)
            self.assertEqual(
                data["whole_machine"]["canonical_field"],
                "whole_machines",
            )
            self.assertEqual(
                data["whole_machine"]["aggregation_kind"],
                "multi_capsule_summary",
            )
            self.assertEqual(
                data["whole_machine"]["coordinate"]["kind"],
                "multi_capsule",
            )
            self.assertFalse(
                any(
                    item["role"] == "dynamic stage aggregation"
                    for item in data["diagnostics"]
                )
            )
            for kind in ("certificate", "replacement", "whole-machine"):
                self.assertEqual(
                    {
                        item.get("details", {}).get("capsule_id")
                        for item in data["evidence"]
                        if item.get("kind") == kind
                    },
                    {"fixture_slide", second_capsule},
                )

            second_replacement.unlink()
            missing_binding = normalize_run(run, workspace_root=root)
            second_view = missing_binding["whole_machines"][1]
            self.assertEqual(second_view["status"], "stale")
            self.assertIn(
                "整机结果对应的 Replacement 缺失或未启用。",
                second_view["stale_reasons"],
            )

    def test_duplicate_whole_machine_capsule_is_aggregation_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate, replacement, whole = _minimal_workspace(root)
            _write_json(certificate, {"verdict": "KINEMATIC_CERTIFIED"})
            _write_json(replacement, {"status": "ACTIVE"})
            _write_json(whole, {"status": "WHOLE_MACHINE_TASK_PASSED"})
            _write_json(whole.parent / "execution_plan.json", {})
            _write_json(
                run / "run_status.json",
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate),
                            "replacement": str(replacement),
                        }
                    ],
                    "whole_machine_results": [
                        {
                            "capsule_id": "fixture_slide",
                            "result_file": str(whole),
                        },
                        {
                            "capsule_id": "fixture_slide",
                            "result_file": str(whole),
                        },
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)

            self.assertIn(
                "WHOLE_MACHINE_RESULT_CAPSULE_DUPLICATE:fixture_slide",
                data["workflow"]["reason_codes"],
            )

    def test_relocates_remote_package_and_run_paths_without_rewriting_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "simulation_ready_cad_workflow"
            package = (
                workspace
                / "demo_machines"
                / "hinged_toolbox_v1"
                / "revision_engineering"
                / "package"
            )
            package.mkdir(parents=True)
            (package / "package.json").write_text("{}", encoding="utf-8")
            intake = (
                workspace
                / ".runtime_runs"
                / "ubuntu-hinged-toolbox-r03"
                / "intake_report.json"
            )
            intake.parent.mkdir(parents=True)
            intake.write_text("{}", encoding="utf-8")

            remote_package = (
                "/remote-host/simulation-ready-cad-harness/"
                "simulation_ready_cad_workflow/demo_machines/"
                "hinged_toolbox_v1/revision_engineering/package"
            )
            remote_intake = (
                "/remote-host/simulation-ready-cad-harness/.runtime_runs/"
                "ubuntu-hinged-toolbox-r03/intake_report.json"
            )

            self.assertEqual(
                resolve_workspace_reference(workspace, remote_package),
                package.resolve(),
            )
            self.assertEqual(
                resolve_workspace_reference(workspace, remote_intake),
                intake.resolve(),
            )

    def test_scan_appends_dashboard_only_run_annotation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / ".runtime_runs" / "clock-r13"
            _write_json(
                run / "plans" / "clock.json",
                {"package_id": "clock", "design_version": "1.0"},
            )
            _write_json(run / "run_status.json", {"status": "WHOLE_MACHINE_TASK_PASSED"})
            _write_json(
                run / "dashboard_annotation.json",
                {
                    "schema_version": "0.1",
                    "display_suffix": "已知漏检：简化分支打铃联动错误，harness 未识别",
                },
            )

            rows = scan_runs(root, workspace_root=root)

            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["display_name"].endswith("harness 未识别"))

    def test_scan_groups_same_machine_across_runtime_roots_and_prefers_latest_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_run(
                relative: str,
                *,
                version: str,
                status: str,
                stamp: int,
                has_certificate: bool = True,
            ) -> Path:
                run = root / relative
                _write_json(
                    run / "plans" / "hinge.json",
                    {
                        "package_id": "hinged_toolbox_v1",
                        "design_version": version,
                    },
                )
                if has_certificate:
                    _write_json(
                        run / "results" / "hinge" / "certificate.json",
                        {
                            "package": {
                                "package_id": "hinged_toolbox_v1",
                                "design_version": version,
                            }
                        },
                    )
                status_path = run / "run_status.json"
                _write_json(status_path, {"status": status})
                os.utime(status_path, (stamp, stamp))
                return run

            first = write_run(
                ".runtime_runs/hinged-toolbox-v1/closed-loop-r33",
                version="1.0.18",
                status="WHOLE_MACHINE_TASK_PASSED",
                stamp=1_700_000_100,
            )
            second = write_run(
                "simulation_ready_cad_workflow/.runtime_runs/hinged-toolbox-engineering-r9",
                version="2.0.0",
                status="WHOLE_MACHINE_TASK_PASSED",
                stamp=1_700_000_200,
            )
            third = write_run(
                "simulation_ready_cad_workflow/.runtime_runs/hinged-toolbox-engineering-r10",
                version="2.0.1",
                status="NOT_CERTIFIED",
                stamp=1_700_000_300,
                has_certificate=False,
            )

            rows = scan_runs(root, workspace_root=root)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["name"], "hinged-toolbox-engineering-r9")
            self.assertEqual(rows[0]["package_id"], "hinged_toolbox_v1")
            self.assertEqual(rows[0]["package_version"], "2.0.0")
            self.assertEqual(
                rows[0]["display_name"],
                "hinged_toolbox_v1 v2.0.0 · hinged-toolbox-engineering-r9",
            )

    def test_comparison_checks_explain_independent_behavior_gates(self) -> None:
        load = {
            "program": "bidirectional_axis",
            "phases": [
                {"phase_id": "forward", "wrench_kind": "force"},
                {"phase_id": "reverse", "wrench_kind": "force"},
            ],
        }
        physical = {
            "max_lateral_shift_m": 0.00001,
            "max_rotation_rad": 0.0,
            "axis_purity": 0.99,
            "contact_observed": True,
            "phase_directional_travel_m": {"forward": 0.001, "reverse": 0.0012},
        }
        reduced = {
            "max_lateral_shift_m": 0.0,
            "max_rotation_rad": 0.0,
            "axis_purity": 1.0,
            "phase_directional_travel_m": {"forward": 0.004, "reverse": 0.004},
        }
        thresholds = {
            "max_lateral_shift_m": 0.0005,
            "max_rotation_rad": 0.05,
            "min_axis_purity": 0.95,
            "min_directional_travel_m": 0.0001,
        }
        rows = _comparison_checks(load, physical, reduced, thresholds)
        travel = next(row for row in rows if row["label"] == "最小顺载运动")
        self.assertTrue(travel["physical_pass"])
        self.assertTrue(travel["reduced_pass"])
        self.assertAlmostEqual(travel["physical_margin"], 0.9)
        self.assertEqual(travel["note"], "")
        self.assertTrue(next(row for row in rows if row["label"] == "真实接触证据")["physical_pass"])

    def test_comparison_checks_omit_contact_gate_when_not_published(self) -> None:
        rows = _comparison_checks(
            {
                "program": "task_angle_roundtrip",
                "phases": [],
            },
            {
                "max_pivot_drift_m": 0.0,
                "max_off_axis_rotation_rad": 0.0,
            },
            {
                "max_pivot_drift_m": 0.0,
                "max_off_axis_rotation_rad": 0.0,
            },
            {
                "max_pivot_drift_m": 0.001,
                "max_off_axis_rotation_rad": 0.01,
            },
        )

        self.assertNotIn("真实接触证据", {row["label"] for row in rows})

    def test_task_range_uses_one_sided_target_boundaries(self) -> None:
        load = {
            "program": "task_range_roundtrip",
            "phases": [
                {
                    "phase_id": "outbound",
                    "wrench_kind": "force",
                    "stop_condition": "progress_at_least",
                    "target_progress_m": 0.004,
                },
                {
                    "phase_id": "return",
                    "wrench_kind": "force",
                    "stop_condition": "progress_at_most",
                    "target_progress_m": 0.0,
                },
            ],
        }
        physical = {
            "max_lateral_shift_m": 0.0,
            "max_rotation_rad": 0.0,
            "axis_purity": 1.0,
            "contact_observed": True,
            "phase_progress_bounds_m": {
                "outbound": {"max_m": 0.0042709},
                "return": {"min_m": -0.0001131},
            },
        }
        reduced = {
            "max_lateral_shift_m": 0.0,
            "max_rotation_rad": 0.0,
            "axis_purity": 1.0,
            "phase_directional_travel_m": {},
        }
        thresholds = {
            "max_lateral_shift_m": 0.0005,
            "max_rotation_rad": 0.05,
            "min_axis_purity": 0.95,
            "min_directional_travel_m": 0.0001,
        }
        rows = _comparison_checks(load, physical, reduced, thresholds)
        outbound = next(row for row in rows if row["label"].startswith("outbound "))
        returning = next(row for row in rows if row["label"].startswith("return "))

        self.assertTrue(outbound["physical_pass"])
        self.assertAlmostEqual(outbound["limit"], 3.9)
        self.assertAlmostEqual(outbound["physical"], 4.2709)
        self.assertAlmostEqual(outbound["physical_margin"], 0.3709)
        self.assertTrue(returning["physical_pass"])
        self.assertAlmostEqual(returning["limit"], 0.1)
        self.assertAlmostEqual(returning["physical"], -0.1131)
        self.assertAlmostEqual(returning["physical_margin"], 0.2131)

    def test_failed_certificate_is_not_presented_as_certified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            _write_json(certificate_path, {"verdict": "NOT_CERTIFIED", "certificate_level": "NOT_CERTIFIED"})
            _write_json(replacement_path, {})
            _write_json(whole_path, {})
            _write_json(
                run / "run_status.json",
                {
                    "status": "FAILED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                        }
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)
            certificate_item = next(item for item in data["evidence"] if item["kind"] == "certificate")
            certificate_stage = next(item for item in data["stages"] if item["id"] == "certificate")
            self.assertEqual(certificate_item["details"]["status"], "failed")
            self.assertEqual(certificate_stage["status"], "failed")
            self.assertIn("未签发", certificate_stage["why"])
            self.assertIn("不构成可用认证包络", certificate_item["details"]["boundary"])

            replacement_item = next(item for item in data["evidence"] if item["kind"] == "replacement")
            whole_item = next(item for item in data["evidence"] if item["kind"] == "whole-machine")
            replacement_stage = next(item for item in data["stages"] if item["id"] == "replacement")
            self.assertFalse(replacement_item["details"]["certificate_hash_matches"])
            self.assertFalse(replacement_item["details"]["package_hash_matches"])
            self.assertIn("当前未具备", " ".join(replacement_stage["inputs"]))
            self.assertNotIn("ACTIVE replacement", " ".join(replacement_stage["outputs"]))
            self.assertFalse(any(item["id"] == "whole-machine" for item in data["stages"]))
            self.assertNotIn("已认证简化替换", whole_item["details"]["boundary"])
            self.assertIn("尚未生成通过", data["whole_machine"]["boundary"])
            self.assertNotIn("为什么仍然 PASS", render_snapshot(data))

    def test_stale_hash_bindings_preserve_verdict_and_mark_evidence_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                    "replacement_eligible": True,
                    "package": {"package_id": "fixture", "design_version": "1", "package_binding_sha256": "package-hash"},
                    "tested_envelope": {},
                },
            )
            _write_json(
                replacement_path,
                {
                    "status": "ACTIVE",
                    "certificate_sha256": "wrong-certificate-hash",
                    "package_binding_sha256": "package-hash",
                    "replacement_model": {"family": "prismatic"},
                    "validity_envelope": {},
                },
            )
            _write_json(
                whole_path,
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "certificate_sha256": "wrong-certificate-hash",
                    "replacement_sha256": "wrong-replacement-hash",
                    "package_binding_sha256": "package-hash",
                    "task_ids": [],
                    "stages": [],
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                        }
                    ],
                    "whole_machine_results": [{"result_file": str(whole_path)}],
                },
            )
            data = normalize_run(run, workspace_root=root)
            stages = {item["id"]: item for item in data["stages"]}
            self.assertEqual(stages["replacement"]["status"], "stale")
            self.assertEqual(stages["whole-machine"]["status"], "stale")
            self.assertEqual(data["run"]["status"], "passed")
            self.assertEqual(data["run"]["evidence_health"]["status"], "conflict")
            self.assertTrue(stages["replacement"]["issues"])
            self.assertTrue(stages["whole-machine"]["issues"])

    def test_known_legacy_checkpoint_overwrite_is_an_audit_warning_not_mechanical_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                    "replacement_eligible": True,
                    "package": {
                        "package_id": "fixture",
                        "design_version": "1",
                        "package_binding_sha256": "package-hash",
                    },
                    "mechanism": {"family": "four_bar"},
                    "tested_envelope": {},
                },
            )
            certificate_sha256 = hashlib.sha256(
                certificate_path.read_bytes()
            ).hexdigest()
            _write_json(
                replacement_path,
                {
                    "schema_version": "0.1",
                    "status": "PHYSICS_PARTIAL",
                    "branch": "reduced",
                    "package_binding_sha256": "package-hash",
                    "load_case_id": "task_roundtrip",
                    "samples": [{"time_s": 0.0}],
                },
            )
            _write_json(
                whole_path,
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "passed": True,
                    "certificate_sha256": certificate_sha256,
                    "replacement_sha256": "historical-active-replacement-hash",
                    "package_binding_sha256": "package-hash",
                    "task_ids": [],
                    "stages": [],
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                        }
                    ],
                    "whole_machine_results": [{"result_file": str(whole_path)}],
                },
            )

            data = normalize_run(run, workspace_root=root)
            stages = {item["id"]: item for item in data["stages"]}
            self.assertEqual(stages["replacement"]["status"], "warning")
            self.assertEqual(stages["whole-machine"]["status"], "passed")
            self.assertTrue(stages["replacement"]["issues"])
            self.assertTrue(stages["whole-machine"]["issues"])
            self.assertTrue(
                data["overview"]["replacement"]["historical_terminal_authority"]
            )
            self.assertTrue(
                any(
                    item["role"] == "legacy replacement record reconciliation"
                    for item in data["diagnostics"]
                )
            )

    def test_unregistered_whole_machine_file_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            _write_json(certificate_path, {"verdict": "KINEMATIC_CERTIFIED"})
            _write_json(replacement_path, {"status": "ACTIVE"})
            _write_json(
                whole_path,
                {
                    "status": "WHOLE_MACHINE_TASK_PASSED",
                    "stages": [
                        {
                            "stage": "outbound",
                            "target_progress_m": 0.02,
                            "max_progress_m": 0.02,
                            "duration_s": 0.4,
                            "reached": True,
                        }
                    ],
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "KINEMATIC_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                        }
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)
            whole_artifact = next(item for item in data["evidence"] if item["kind"] == "whole-machine")
            self.assertFalse(any(item["id"] == "whole-machine" for item in data["stages"]))
            self.assertEqual(data["whole_machine"]["task_total"], 0)
            self.assertFalse(whole_artifact["exists"])
            self.assertNotEqual(data["run"]["status"], "passed")

    def test_physical_preflight_failure_explains_reduced_branch_was_not_started(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            (run / "queue_run" / "queue_state.json").unlink()
            _write_json(certificate_path, {})
            _write_json(replacement_path, {})
            _write_json(whole_path, {})
            _write_json(
                run / "runtime_queue" / "physical" / "queue_state.json",
                {"status": "SUCCEEDED", "summary": {"total": 1, "SUCCEEDED": 1}},
            )
            preflight = {
                "status": "PHYSICAL_DISPATCH_PREFLIGHT_FAILED",
                "passed": False,
                "trace_count": 1,
                "passed_trace_count": 0,
                "failed_trace_count": 1,
                "reduced_branch_authorized": False,
                "traces": [
                    {
                        "passed": False,
                        "geometry_legality": {
                            "sample_count": 21,
                            "max_penetration_m": 0.0042,
                        },
                    }
                ],
            }
            _write_json(run / "results" / "physical_preflight.json", preflight)
            _write_json(
                run / "run_status.json",
                {
                    "status": "PHYSICAL_PREFLIGHT_FAILED",
                    "last_completed_stage": "physical_geometry_preflight",
                    "physical_queue_status": "SUCCEEDED",
                    "physical_queue_summary": {"total": 1, "SUCCEEDED": 1},
                    "physical_preflight": preflight,
                    "reduced_branch_started": False,
                },
            )

            data = normalize_run(run, workspace_root=root)
            stage = next(item for item in data["stages"] if item["id"] == "branch-simulation")

            self.assertEqual(stage["status"], "failed")
            self.assertEqual(stage["title"], "\u771f\u5b9e\u7269\u7406\u4e0e\u7406\u60f3\u53c2\u8003")
            self.assertIn("\u7406\u60f3\u53c2\u8003\u4e0d\u5f97\u7528\u6765\u8986\u76d6\u8fd9\u4e2a\u5931\u8d25", stage["why"])
            self.assertEqual(data["physical_audit"]["failed_trace_count"], 1)
            self.assertEqual(data["physical_audit"]["selected_pose_count"], 21)
            self.assertEqual(data["physical_audit"]["max_penetration_mm"], 4.2)
            self.assertFalse(data["physical_audit"]["reduced_branch_started"])
            self.assertTrue(
                any(item["kind"] == "physical-preflight" for item in data["evidence"])
            )

    def test_legacy_audit_failure_overrides_dashboard_without_rewriting_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                    "replacement_eligible": True,
                },
            )
            _write_json(replacement_path, {"status": "ACTIVE"})
            _write_json(whole_path, {})
            _write_json(
                run / "run_status.json",
                {
                    "status": "KINEMATIC_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                        }
                    ],
                },
            )
            _write_json(
                run / "migration" / "legacy_sampled_physical_audit_v1.json",
                {
                    "status": "LEGACY_SAMPLED_PHYSICAL_AUDIT_FAILED",
                    "passed": False,
                    "trace_count": 1,
                    "passed_trace_count": 0,
                    "failed_trace_count": 1,
                    "policy": {"historical_certificate_modified": False},
                    "traces": [
                        {
                            "passed": False,
                            "sampling": {"selected_sample_count": 12},
                            "geometry_legality": {"max_penetration_m": 0.002},
                        }
                    ],
                },
            )
            closeup = run / "migration" / "penetration_evidence_v1" / "event.png"
            closeup.parent.mkdir(parents=True)
            closeup.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            _write_json(
                run / "migration" / "penetration_evidence_v1.json",
                {
                    "status": "PENETRATION_VISUAL_EVIDENCE_GENERATED",
                    "events": [
                        {
                            "event_id": "event-1",
                            "load_case_id": "task",
                            "time_s": 1.25,
                            "source_sample_index": 42,
                            "audit_sample_index": 7,
                            "penetration_depth_m": 0.002,
                            "solid_pair": ["fixed/a", "moving/b"],
                            "snapshot": str(closeup.relative_to(root)).replace("\\", "/"),
                            "snapshot_sha256": "fixture-sha",
                            "dashboard_focus": {
                                "load_case_id": "task",
                                "branch": "physical",
                                "frame": 42,
                                "primary_node": "physics__fixed__a",
                                "secondary_node": "physics__moving__b",
                            },
                            "simulation_rerun_for_evidence": False,
                            "red_region_claim": "detected_contact_neighborhood_not_exact_boolean_intersection",
                        }
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)
            stage = next(item for item in data["stages"] if item["id"] == "branch-simulation")

            self.assertEqual(stage["status"], "failed")
            self.assertEqual(data["overview"]["certificate"]["status"], "passed")
            self.assertFalse(data["physical_audit"]["historical_certificate_modified"])
            self.assertNotEqual(data["run"]["status"], "passed")
            self.assertIn("\u65e7\u8bc1\u4e66\u4fdd\u6301\u539f\u6837", stage["why"])
            penetration = next(
                item for item in data["evidence"] if item["kind"] == "penetration-visual"
            )
            self.assertEqual(penetration["details"]["source_sample_index"], 42)
            self.assertEqual(penetration["details"]["dashboard_focus"]["frame"], 42)
            self.assertEqual(penetration["preview_type"], "image")

    def test_load_case_prefers_lightweight_ideal_reference_over_legacy_reduced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            capsule = "fixture_slide"
            load_id = "nominal"
            load_path = run / "loads" / capsule / f"{load_id}.json"
            _write_json(
                load_path,
                {
                    "load_case_id": load_id,
                    "program": "bidirectional_axis",
                    "purpose": "nominal motion",
                    "family": "prismatic",
                    "required_for_certificate": True,
                    "moving_mass_kg": 1.0,
                    "phases": [
                        {
                            "phase_id": "forward",
                            "wrench_kind": "force",
                            "magnitude_si": 10.0,
                        }
                    ],
                },
            )
            trace_root = run / "traces" / capsule / load_id
            _write_json(
                trace_root / "physical.trace.json",
                {
                    "status": "COMPLETED",
                    "branch": "physical",
                    "sample_count": 2,
                },
            )
            _write_json(
                trace_root / "ideal_reference.trace.json",
                {
                    "status": "COMPLETED",
                    "branch": "ideal_reference",
                    "sample_count": 2,
                    "samples": [
                        {
                            "time_s": 0.0,
                            "phase_id": "forward",
                            "output_coordinate": 0.0,
                            "output_coordinate_unit": "m",
                        },
                        {
                            "time_s": 0.1,
                            "phase_id": "forward",
                            "output_coordinate": 0.01,
                            "output_coordinate_unit": "m",
                        },
                    ],
                },
            )
            _write_json(
                run / "results" / capsule / "comparison" / f"{load_id}.json",
                {
                    "verdict": "PASS",
                    "comparison_mode": "physical_vs_ideal_reference",
                    "physical_metrics": {
                        "max_progress_m": 0.0098,
                        "max_lateral_shift_m": 0.0,
                        "max_rotation_rad": 0.0,
                        "axis_purity": 1.0,
                        "contact_observed": True,
                        "phase_directional_travel_m": {"forward": 0.0098},
                    },
                    "thresholds": {
                        "max_lateral_shift_m": 0.001,
                        "max_rotation_rad": 0.01,
                        "minimum_axis_purity": 0.95,
                    },
                    "ideal_reference_metrics": {
                        "max_absolute_error": 0.0002,
                    },
                    "ideal_reference_threshold": {
                        "max_absolute_error": 0.0005,
                        "unit": "m",
                    },
                },
            )
            evidence: list[dict] = []
            result = _load_case(
                ReadContext(root, []),
                run,
                capsule,
                load_path,
                evidence,
            )

        self.assertEqual(result["ideal_kind"], "ideal_reference")
        self.assertEqual(
            result["ideal_label"], "简化动力学分支（方程计算）"
        )
        self.assertFalse(result["ideal_dynamic_response"])
        self.assertEqual(result["reduced"]["max_progress_mm"], 10.0)
        ideal_check = next(
            item
            for item in result["checks"]
            if item["label"] == "相对理想机构的最大输出偏差"
        )
        self.assertEqual(ideal_check["physical"], 0.2)
        self.assertTrue(ideal_check["physical_pass"])
        self.assertTrue(any(item["kind"] == "trace-ideal-reference" for item in evidence))

    def test_dashboard_uses_latest_attempt_and_compact_trace_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = (
                _minimal_workspace(root)
            )
            (run / "queue_run" / "queue_state.json").unlink()
            capsule = "fixture_slide"
            load_id = "analytical_transmission"
            load_path = run / "loads" / capsule / f"{load_id}.json"
            _write_json(
                load_path,
                {
                    "load_case_id": load_id,
                    "program": "analytical_transmission_roundtrip",
                    "purpose": "exercise attempt-scoped queue evidence",
                    "family": "revolute",
                    "required_for_certificate": True,
                    "phases": [],
                },
            )
            trace_root = run / "traces" / capsule / load_id
            trace_root.mkdir(parents=True)
            (trace_root / "physical.trace.json").write_text(
                "{not-read-because-compact-summary-is-authoritative",
                encoding="utf-8",
            )
            (trace_root / "ideal_reference.trace.json").write_text(
                "{not-read-because-compact-summary-is-authoritative",
                encoding="utf-8",
            )
            _write_json(
                run / "results" / capsule / "comparison" / f"{load_id}.json",
                {
                    "verdict": "PASS",
                    "comparison_mode": "physical_vs_ideal_reference",
                    "physical_metrics": {
                        "max_pivot_drift_m": 0.0,
                        "max_off_axis_rotation_rad": 0.0,
                    },
                    "thresholds": {
                        "max_pivot_drift_m": 0.001,
                        "max_off_axis_rotation_rad": 0.01,
                    },
                    "ideal_reference_metrics": {"max_absolute_error": 0.0},
                    "ideal_reference_threshold": {
                        "max_absolute_error": 0.01,
                        "unit": "rad",
                    },
                },
            )
            job_id = "fixture-physical-job"
            _write_json(
                run / "physical_queue_manifest.json",
                {
                    "jobs": [
                        {
                            "job_id": job_id,
                            "expected_outputs": [
                                f"traces/{capsule}/{load_id}/physical.trace.json"
                            ],
                        }
                    ]
                },
            )
            _write_json(
                run
                / "runtime_queue"
                / "attempt-001"
                / "physical"
                / "queue_state.json",
                {
                    "status": "FAILED",
                    "summary": {"total": 1, "FAILED": 1},
                    "jobs": {job_id: {"status": "FAILED"}},
                },
            )
            _write_json(
                run
                / "runtime_queue"
                / "attempt-002"
                / "physical"
                / "queue_state.json",
                {
                    "status": "SUCCEEDED",
                    "summary": {"total": 1, "SUCCEEDED": 1},
                    "jobs": {
                        job_id: {
                            "status": "SUCCEEDED",
                            "lane_phase": "SUCCEEDED",
                        }
                    },
                },
            )
            preflight = {
                "status": "PASS",
                "passed": True,
                "trace_count": 1,
                "passed_trace_count": 1,
                "failed_trace_count": 0,
                "traces": [
                    {
                        "capsule_id": capsule,
                        "load_case_id": load_id,
                        "status": "PHYSICAL_PREFLIGHT_PASSED",
                        "passed": True,
                        "physical_trace_sha256": "physical-sha",
                        "geometry_legality": {
                            "passed": True,
                            "sample_count": 48801,
                            "max_penetration_m": 0.0,
                        },
                    }
                ],
            }
            _write_json(run / "results" / "physical_preflight.json", preflight)
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                },
            )
            _write_json(replacement_path, {"status": "ACTIVE"})
            _write_json(whole_path, {})
            _write_json(
                run / "run_status.json",
                {
                    "status": "KINEMATIC_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": capsule,
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                        }
                    ],
                    "branch_queues": {
                        "physical": {
                            "status": "FAILED",
                            "summary": {"total": 1, "FAILED": 1},
                        }
                    },
                    "ideal_reference": {
                        "status": "PASS",
                        "trace_count": 1,
                        "traces": [
                            {
                                "capsule_id": capsule,
                                "load_case_id": load_id,
                                "sample_count": 48801,
                                "trace_sha256": "ideal-sha",
                            }
                        ],
                    },
                },
            )

            data = normalize_run(run, workspace_root=root)

        load_case = data["load_cases"][0]
        simulation = next(
            stage for stage in data["stages"]
            if stage["id"] == "branch-simulation"
        )
        self.assertEqual(simulation["status"], "passed")
        self.assertEqual(load_case["branch_status"], "passed")
        self.assertEqual(load_case["physical"]["sample_count"], 48801)
        self.assertEqual(load_case["execution_lane"]["status"], "passed")
        self.assertNotIn(
            "真实接触证据",
            {row["label"] for row in load_case["checks"]},
        )
        trace_diagnostics = [
            item
            for item in data["diagnostics"]
            if " trace" in str(item.get("role") or "")
        ]
        self.assertEqual(trace_diagnostics, [])
        trace_artifacts = {
            item["kind"]: item
            for item in data["evidence"]
            if item["kind"] in {"trace-physical", "trace-ideal-reference"}
        }
        self.assertEqual(
            trace_artifacts["trace-physical"]["details"]["status_source"],
            "physical_preflight",
        )
        self.assertEqual(
            trace_artifacts["trace-ideal-reference"]["details"][
                "status_source"
            ],
            "ideal_reference_summary",
        )

    def test_snapshot_embeds_visual_media_without_external_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, whole_path = _minimal_workspace(root)
            _write_json(certificate_path, {})
            _write_json(replacement_path, {})
            _write_json(whole_path, {})
            _write_json(run / "run_status.json", {"status": "NOT_CERTIFIED", "mechanisms": [{"capsule_id": "fixture_slide"}]})
            visual = run / "visual"
            visual.mkdir()
            (visual / "contact_sheet.png").write_bytes(b"\x89PNG\r\n\x1a\nfixture")
            (visual / "stage1_design_package.glb").write_bytes(b"glTF-fixture")
            data = normalize_run(run, workspace_root=root, mode="snapshot")
            image = next(item for item in data["evidence"] if item["kind"] == "visual-image")
            model = next(item for item in data["evidence"] if item["kind"] == "interactive-model")
            self.assertTrue(image["preview_data_uri"].startswith("data:image/png;base64,"))
            self.assertTrue(model["preview_data_uri"].startswith("data:model/gltf-binary;base64,"))
            html = render_snapshot(data)
            self.assertIn("data:image/png;base64,", html)
            self.assertIn("data:model/gltf-binary;base64,", html)
            self.assertNotIn('src="/static/viewer3d.js"', html)
            self.assertNotIn('src="/static/app.js"', html)
            self.assertNotIn('href="/static/app.css"', html)
            self.assertNotIn('"samples":', html)
            self.assertLess(
                html.index('id="app"'),
                html.index('const app = document.getElementById("app");'),
            )
            self.assertLess(
                html.index("window.QUEST_BOOT="),
                html.index('const boot = window.QUEST_BOOT'),
            )

    def test_scan_marks_certificate_only_run_as_not_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "certificate-only"
            _write_json(run / "run_status.json", {"status": "KINEMATIC_CERTIFIED"})
            rows = scan_runs(
                root / "runs",
                workspace_root=root,
                include_history=True,
            )
            self.assertEqual(rows[0]["status"], "not_started")

    def test_scan_does_not_promote_physical_preflight_to_project_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "runs" / "physical-only"
            _write_json(
                run / "run_status.json",
                {"status": "PHYSICAL_DISPATCH_PREFLIGHT_PASSED"},
            )
            _write_json(
                run / "results" / "physical_preflight.json",
                {"status": "PHYSICAL_DISPATCH_PREFLIGHT_PASSED"},
            )

            rows = scan_runs(
                root / "runs",
                workspace_root=root,
                include_history=True,
            )

            self.assertEqual(rows[0]["status"], "not_started")
            self.assertEqual(
                rows[0]["status_raw"],
                "PHYSICAL_DISPATCH_PREFLIGHT_PASSED",
            )
            self.assertEqual(
                rows[0]["physical_audit_status_raw"],
                "PHYSICAL_DISPATCH_PREFLIGHT_PASSED",
            )

    def test_reference_only_task_uses_only_applicable_six_stage_route(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, _ = _minimal_workspace(root)
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                },
            )
            _write_json(
                replacement_path,
                {
                    "status": "INACTIVE",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "TASK_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                            "replacement_policy": "reference_only_single_mechanism",
                        }
                    ],
                },
            )

            data = normalize_run(run, workspace_root=root)

            self.assertEqual(data["run"]["status"], "passed")
            self.assertEqual(
                [item["kind"] for item in data["stages"]],
                [
                    "design-package",
                    "intake",
                    "mechanism-plan",
                    "geometry-preflight",
                    "branch-simulation",
                    "certificate",
                ],
            )
            self.assertEqual(data["run"]["progress"]["total"], 6)
            self.assertTrue(all("scope" in item for item in data["stages"]))

    def test_multi_scope_run_cannot_silently_claim_complete_stage_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, _ = _minimal_workspace(root)
            _write_json(
                run / "plans" / "second.json",
                {
                    "capsule_id": "second_capsule",
                    "family": "revolute",
                },
            )
            _write_json(
                certificate_path,
                {"verdict": "KINEMATIC_CERTIFIED"},
            )
            _write_json(
                replacement_path,
                {
                    "status": "INACTIVE",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "TASK_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                            "replacement_policy": "reference_only_single_mechanism",
                        },
                        {"capsule_id": "second_capsule"},
                    ],
                },
            )

            row = scan_runs(
                root / "runs",
                workspace_root=root,
                include_history=True,
            )[0]
            data = normalize_run(run, workspace_root=root)

            self.assertEqual(data["run"]["status"], "passed")
            self.assertEqual(data["workflow"]["aggregation_status"], "partial")
            self.assertIn(
                "CAPSULE_TRACE_EVIDENCE_MISSING:second_capsule",
                data["workflow"]["reason_codes"],
            )
            self.assertIn(
                "CAPSULE_CERTIFICATE_EVIDENCE_MISSING:second_capsule",
                data["workflow"]["reason_codes"],
            )
            self.assertEqual(data["run"]["evidence_health"]["status"], "conflict")
            self.assertEqual(row["evidence_health"]["status"], "conflict")
            self.assertTrue(
                any(
                    item["role"] == "dynamic stage aggregation"
                    for item in data["diagnostics"]
                )
            )

    def test_final_pass_with_failed_physical_audit_is_pass_with_evidence_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, certificate_path, replacement_path, _ = _minimal_workspace(root)
            _write_json(
                certificate_path,
                {"verdict": "KINEMATIC_CERTIFIED"},
            )
            _write_json(
                replacement_path,
                {
                    "status": "INACTIVE",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                run / "migration" / "legacy_sampled_physical_audit_v1.json",
                {
                    "status": "LEGACY_SAMPLED_PHYSICAL_AUDIT_FAILED",
                    "failed_trace_count": 1,
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "TASK_CERTIFIED",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(certificate_path),
                            "replacement": str(replacement_path),
                            "replacement_policy": "reference_only_single_mechanism",
                        }
                    ],
                },
            )

            row = scan_runs(
                root / "runs",
                workspace_root=root,
                include_history=True,
            )[0]
            data = normalize_run(run, workspace_root=root)

            self.assertEqual(row["status"], "passed")
            self.assertEqual(row["evidence_health"]["status"], "conflict")
            self.assertEqual(data["run"]["status"], "passed")
            self.assertEqual(data["run"]["evidence_health"]["status"], "conflict")

    def test_reference_only_certificate_lists_post_run_pose_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = "fixture_proj"
            run = root / ".runtime_runs" / "projects" / project / "runs" / "r01"
            legacy_run, certificate_path, replacement_path, _ = _minimal_workspace(root)
            shutil.copytree(legacy_run, run)
            certificate_path = (
                run / "results" / "fixture_slide" / "certificate.json"
            )
            replacement_path = (
                run / "results" / "fixture_slide" / "replacement.json"
            )
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                replacement_path,
                {
                    "status": "INACTIVE",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "TASK_CERTIFIED",
                    "replacement_policy": "reference_only_single_mechanism",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(
                                certificate_path.relative_to(root).as_posix()
                            ),
                            "replacement": str(
                                replacement_path.relative_to(root).as_posix()
                            ),
                            "replacement_policy": "reference_only_single_mechanism",
                        }
                    ],
                },
            )
            video = (
                root
                / ".runtime_runs"
                / "diagnostics"
                / "fixture_proj-r01-ideal-pose"
                / "fixture_slide"
                / "physical_vs_ideal_roundtrip.mp4"
            )
            video.parent.mkdir(parents=True)
            video.write_bytes(b"fake-mp4")
            _write_json(
                video.with_suffix(".sidecar.json"),
                {
                    "status": "ENCODED",
                    "trace_sha256": {
                        "physical": "a" * 64,
                        "reduced": "b" * 64,
                    },
                    "load_case_id": "task_angle_roundtrip",
                },
            )

            data = normalize_run(run, workspace_root=root)
            replay = next(
                item
                for item in data["evidence"]
                if (item.get("details") or {}).get("visual_role")
                == "reference-comparison-video"
            )
            certificate_stage = next(
                item for item in data["stages"] if item["id"] == "certificate"
            )

            self.assertEqual(
                [item["kind"] for item in data["stages"]],
                [
                    "design-package",
                    "intake",
                    "mechanism-plan",
                    "geometry-preflight",
                    "branch-simulation",
                    "certificate",
                ],
            )
            self.assertIn(replay["id"], certificate_stage["evidence_ids"])
            self.assertTrue(replay["exists"])

    def test_certificate_lists_diagnostic_physics_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = "fixture_proj"
            run = root / ".runtime_runs" / "projects" / project / "runs" / "r01"
            legacy_run, certificate_path, replacement_path, _ = _minimal_workspace(root)
            shutil.copytree(legacy_run, run)
            certificate_path = (
                run / "results" / "fixture_slide" / "certificate.json"
            )
            replacement_path = (
                run / "results" / "fixture_slide" / "replacement.json"
            )
            _write_json(
                certificate_path,
                {
                    "verdict": "KINEMATIC_CERTIFIED",
                    "certificate_level": "KINEMATIC_CERTIFIED",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                replacement_path,
                {
                    "status": "INACTIVE",
                    "replacement_policy": "reference_only_single_mechanism",
                },
            )
            _write_json(
                run / "run_status.json",
                {
                    "status": "TASK_CERTIFIED",
                    "replacement_policy": "reference_only_single_mechanism",
                    "mechanisms": [
                        {
                            "capsule_id": "fixture_slide",
                            "certificate": str(
                                certificate_path.relative_to(root).as_posix()
                            ),
                            "replacement": str(
                                replacement_path.relative_to(root).as_posix()
                            ),
                            "replacement_policy": "reference_only_single_mechanism",
                        }
                    ],
                },
            )
            visual_dir = (
                root
                / ".runtime_runs"
                / "diagnostics"
                / "fixture_proj-r01-physics-evidence"
                / "fixture_slide"
            )
            visual_dir.mkdir(parents=True)
            glb = visual_dir / "stage5_physics_evidence.glb"
            glb.write_bytes(b"fake-glb")
            _write_json(
                visual_dir / "stage5_physics_evidence.scene.json",
                {"output_sha256": hashlib.sha256(b"fake-glb").hexdigest()},
            )
            _write_json(
                visual_dir / "stage5_physics_evidence__task_angle_roundtrip.json",
                {
                    "capsule_id": "fixture_slide",
                    "load_case_id": "task_angle_roundtrip",
                    "branches": {"physical": {"status": "COMPLETED"}},
                },
            )

            data = normalize_run(run, workspace_root=root)
            roles = {
                (item.get("details") or {}).get("visual_role")
                for item in data["evidence"]
            }
            scene = next(
                item
                for item in data["evidence"]
                if (item.get("details") or {}).get("visual_role")
                == "physics-evidence-scene"
            )
            replay = next(
                item
                for item in data["evidence"]
                if (item.get("details") or {}).get("visual_role")
                == "physics-evidence-replay"
            )
            certificate_stage = next(
                item for item in data["stages"] if item["id"] == "certificate"
            )

            self.assertIn("physics-evidence-scene", roles)
            self.assertIn("physics-evidence-scene-manifest", roles)
            self.assertIn("physics-evidence-replay", roles)
            self.assertTrue(scene["exists"])
            self.assertEqual(
                replay["details"]["load_case_id"],
                "task_angle_roundtrip",
            )
            self.assertIn(scene["id"], certificate_stage["evidence_ids"])
            self.assertIn(replay["id"], certificate_stage["evidence_ids"])


if __name__ == "__main__":
    unittest.main()
