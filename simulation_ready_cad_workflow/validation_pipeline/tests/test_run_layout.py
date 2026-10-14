from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulation_ready_cad_workflow.validation_pipeline.dashboard.scene_assets import (
    SceneMeshes,
    load_scene_meshes,
)
from simulation_ready_cad_workflow.validation_pipeline import run_layout


class CanonicalRunLayoutTests(unittest.TestCase):
    def test_preparing_is_nonterminal_and_cannot_be_terminal_authority(
        self,
    ) -> None:
        self.assertIn("PREPARING", run_layout.NON_TERMINAL_STATUSES)
        self.assertIn("PREPARING", run_layout.CANONICAL_RUN_STATUSES)
        self.assertEqual(
            run_layout.normalize_canonical_status(" preparing "),
            "PREPARING",
        )
        self.assertFalse(run_layout.is_terminal_status("PREPARING"))
        self.assertNotIn(
            "PREPARING",
            run_layout.CANONICAL_TERMINAL_STATUSES,
        )
        self.assertIn(
            "ITERATION_SCOPE_COMPLETE",
            run_layout.NON_TERMINAL_STATUSES,
        )
        self.assertFalse(
            run_layout.is_terminal_status("ITERATION_SCOPE_COMPLETE")
        )

    def test_engineering_scene_prefers_canonical_step_over_collider_meshes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / "run"
            run.mkdir()
            expected = SceneMeshes(
                fixed=[],
                moving=[],
                source_kind="package-step-per-solid-tessellation",
                source_paths=[],
            )
            with patch(
                "simulation_ready_cad_workflow.validation_pipeline.dashboard.scene_assets._load_package_step",
                return_value=expected,
            ), patch(
                "simulation_ready_cad_workflow.validation_pipeline.dashboard.scene_assets._load_per_solid"
            ) as collider_fallback:
                result = load_scene_meshes(
                    workspace_root=workspace,
                    run_dir=run,
                    capsule_id="fixture",
                )

            self.assertIs(result, expected)
            collider_fallback.assert_not_called()


if __name__ == "__main__":
    unittest.main()
