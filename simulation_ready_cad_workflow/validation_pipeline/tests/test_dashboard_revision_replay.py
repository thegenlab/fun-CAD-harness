from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import trimesh

from simulation_ready_cad_workflow.validation_pipeline.dashboard.scene_assets import (
    SceneMeshes,
    interactive_glb_bytes,
    load_revision_scene_meshes,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.server import (
    DashboardServer,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class DashboardRevisionReplayTests(unittest.TestCase):
    def test_interactive_glb_bytes_exports_fixed_and_moving_meshes(self) -> None:
        fixed = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        moving = trimesh.creation.box(extents=(0.5, 0.5, 0.5))

        payload = interactive_glb_bytes(
            SceneMeshes(
                fixed=[fixed],
                moving=[moving],
                source_kind="fixture",
                source_paths=[],
            ),
            capsule_id="fixture",
        )

        self.assertEqual(payload[:4], b"glTF")
        self.assertGreater(len(payload), 128)

    def test_revision_scene_falls_back_to_frozen_submission_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / ".runtime_runs/projects/demo/runs/r02"
            package = run / "design_submissions/t0002/package"
            package.mkdir(parents=True)
            (package / "assembly.step").write_text("STEP", encoding="ascii")
            _write_json(package / "assembly_contract.json", {"parts": []})
            _write_json(
                package.parent / "submission_inventory.json",
                {
                    "schema_version": "simulation_ready_cad_submission_inventory_v2",
                    "package_dir": package.relative_to(workspace).as_posix(),
                },
            )
            expected = SceneMeshes([], [], "fixture", [])
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.dashboard."
                "scene_assets._load_package_step_directory",
                return_value=expected,
            ) as loader:
                result = load_revision_scene_meshes(
                    workspace_root=workspace,
                    run_dir=run,
            )

            self.assertIs(result, expected)
            self.assertEqual(loader.call_args.args[2], package)

    def test_revision_frame_is_bound_to_catalog_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / ".runtime_runs/projects/demo/runs/r01"
            preview = run / "visual/stage1_design_package.png"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"png-frame")
            run_id = run.relative_to(workspace).as_posix()
            relative = preview.relative_to(workspace).as_posix()
            server = object.__new__(DashboardServer)
            server.base_path = ""
            server.workspace_root = workspace
            server.runs_root = workspace / ".runtime_runs"
            server.resolve_run = lambda value: run if value == run_id else None
            server.runs = lambda: [
                {
                    "id": run_id,
                    "revision_preview_path": relative,
                    "revision_preview_sha256": hashlib.sha256(
                        b"png-frame"
                    ).hexdigest(),
                }
            ]

            path, payload = server.read_revision_frame(run_id)

            self.assertEqual(path, preview)
            self.assertEqual(payload, b"png-frame")

            preview.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                server.read_revision_frame(run_id)
