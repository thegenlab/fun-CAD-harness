from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import trimesh

from simulation_ready_cad_workflow.validation_pipeline.mechanism.io import (
    sha256_file,
    write_json_atomic,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.penetration_visuals import (
    build_penetration_visual_evidence,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.physical_legality import (
    audit_family_collision_geometry,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.sampling import (
    select_saved_trace_samples,
)


class PenetrationVisualEvidenceTests(unittest.TestCase):
    def test_renders_source_frame_closeup_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            run = workspace / "demo-run"
            trace_root = run / "traces" / "capsule" / "task"
            trace_root.mkdir(parents=True)
            bodies = []
            for body_id, fixed in (("fixed", True), ("moving", False)):
                mesh_path = workspace / f"{body_id}.obj"
                mesh_path.write_text(
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
                                "mesh": mesh_path.relative_to(workspace).as_posix(),
                                "sha256": sha256_file(mesh_path),
                            }
                        ],
                    }
                )
            geometry = {
                "schema": "step_per_solid_body_local_v1",
                "bodies": bodies,
                "interpenetration_tolerance_m": 0.00035,
            }

            def sample(time_s: float, moving_x: float) -> dict:
                return {
                    "time_s": time_s,
                    "phase_id": "drive",
                    "contact_count": 0,
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

            samples = [sample(0.0, 2.0), sample(0.02, 2.0), sample(0.10, 0.8)]
            trace_path = trace_root / "physical.trace.json"
            write_json_atomic(
                trace_path,
                {
                    "branch": "physical",
                    "capsule_id": "capsule",
                    "load_case_id": "task",
                    "collision_geometry": geometry,
                    "samples": samples,
                },
            )
            selected, sampling = select_saved_trace_samples(samples)
            audit = audit_family_collision_geometry(
                geometry,
                selected,
                workspace_root=workspace,
            )
            self.assertFalse(audit["passed"])
            self.assertEqual(audit["worst_events"][0]["source_sample_index"], 2)
            report_path = run / "migration" / "legacy_sampled_physical_audit_v1.json"
            write_json_atomic(
                report_path,
                {
                    "schema_version": "0.1",
                    "status": "LEGACY_SAMPLED_PHYSICAL_AUDIT_FAILED",
                    "passed": False,
                    "traces": [
                        {
                            "passed": False,
                            "trace": trace_path.relative_to(workspace).as_posix(),
                            "trace_sha256": sha256_file(trace_path),
                            "capsule_id": "capsule",
                            "load_case_id": "task",
                            "sampling": sampling,
                            "geometry_legality": audit,
                        }
                    ],
                },
            )

            summary = build_penetration_visual_evidence(
                workspace_root=workspace,
                run_dir=run,
            )
            self.assertEqual(summary["event_count"], 1)
            self.assertEqual(summary["generation_failure_count"], 0)
            event = summary["events"][0]
            self.assertEqual(event["source_sample_index"], 2)
            self.assertEqual(event["dashboard_focus"]["frame"], 2)
            self.assertEqual(
                event["dashboard_focus"]["primary_node"],
                "physics__fixed__fixed_solid",
            )
            snapshot = workspace / event["snapshot"]
            sidecar = workspace / event["sidecar"]
            self.assertTrue(snapshot.is_file())
            self.assertTrue(sidecar.is_file())
            self.assertEqual(event["snapshot_sha256"], sha256_file(snapshot))
            saved = json.loads(sidecar.read_text(encoding="utf-8"))
            self.assertFalse(saved["simulation_rerun_for_evidence"])
            image = mpimg.imread(snapshot)
            self.assertGreater(image.shape[0], 600)
            self.assertGreater(image.shape[1], 1200)
            rgb = image[..., :3]
            red = (rgb[..., 0] > 0.65) & (rgb[..., 1] < 0.42) & (rgb[..., 2] < 0.42)
            blue = (rgb[..., 0] < 0.35) & (rgb[..., 1] > 0.25) & (rgb[..., 2] > 0.50)
            orange = (rgb[..., 0] > 0.65) & (rgb[..., 1] > 0.35) & (rgb[..., 2] < 0.35)
            self.assertGreater(int(np.count_nonzero(red)), 50)
            self.assertGreater(int(np.count_nonzero(blue)), 50)
            self.assertGreater(int(np.count_nonzero(orange)), 50)


if __name__ == "__main__":
    unittest.main()
