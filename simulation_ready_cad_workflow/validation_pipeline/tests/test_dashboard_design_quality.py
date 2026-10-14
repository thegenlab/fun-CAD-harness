from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import normalize_run
from simulation_ready_cad_workflow.validation_pipeline.dashboard.render import render_snapshot
from simulation_ready_cad_workflow.validation_pipeline.design_quality import snapshot_design_quality


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DashboardDesignQualityTests(unittest.TestCase):
    def test_stage_one_exposes_prompt_goal_and_review_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            revision = root / "revision1"
            package = revision / "package"
            package.mkdir(parents=True)
            step = package / "assembly.step"
            step.write_bytes(b"step")
            _write_json(package / "package.json", {"package_id": "toolbox", "design_version": "2"})
            _write_json(package / "assembly_contract.json", {"schema_version": "0.1"})
            prompt = "Build a recognizable sheet-metal toolbox with a retained hinge."
            _write_json(
                revision / "design_input.json",
                {
                    "schema_version": "0.1",
                    "original_prompt": prompt,
                    "plain_language_goal": "A portable metal toolbox with a working lid.",
                    "engineering_brief": {
                        "mechanism": "retained hinge",
                        "major_components": ["base", "lid", "handle"],
                    },
                    "visual_acceptance_checklist": ["recognizable toolbox"],
                },
            )
            image = revision / "visual_review" / "round1" / "iso.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            _write_json(
                revision / "visual_review" / "summary.json",
                {
                    "schema_version": "0.1",
                    "status": "PASSED",
                    "assembly_step_sha256": _sha(step),
                    "rounds": [
                        {
                            "round": 1,
                            "render_assets": [
                                {
                                    "role": "iso",
                                    "label": "等轴总览",
                                    "path": "visual_review/round1/iso.png",
                                    "sha256": _sha(image),
                                }
                            ],
                            "findings": [],
                        }
                    ],
                },
            )
            intake = root / "intake.json"
            _write_json(
                intake,
                {
                    "status": "ACCEPTED",
                    "summary": {"error_count": 0, "warning_count": 0},
                    "capabilities": {"can_enter_physical_preflight": True},
                    "package_summary": {"step_solid_count": 3, "package_file_count": 3},
                },
            )
            run = root / "runs" / "toolbox-run"
            _write_json(
                run / "plans" / "hinge.json",
                {
                    "capsule_id": "hinge",
                    "family": "revolute",
                    "package_dir": str(package),
                    "intake_report": str(intake),
                },
            )
            _write_json(run / "run_status.json", {"status": "RUNNING"})
            snapshot_design_quality(package, run)

            data = normalize_run(run, workspace_root=root, mode="snapshot")
            self.assertEqual(data["design_quality"]["status"], "evidence_complete")
            self.assertEqual(data["design_quality"]["original_prompt"], prompt)
            self.assertEqual(data["design_quality"]["render_asset_count"], 1)
            self.assertEqual(data["stages"][0]["status"], "evidence_complete")
            visual = next(
                item
                for item in data["evidence"]
                if (item.get("details") or {}).get("visual_role") == "design-review-iso"
            )
            self.assertTrue(visual["preview_data_uri"].startswith("data:image/png;base64,"))
            self.assertEqual(
                visual["details"]["render_authority"], "worker_supplemental"
            )
            self.assertIs(visual["details"]["review_authority"], False)
            self.assertIn("Worker 补充图（worker_supplemental）", visual["label"])
            self.assertNotIn("权威 STEP", visual["summary"])
            self.assertEqual(
                data["design_quality"]["authority"]["scope"],
                "design_input_package_completeness",
            )
            self.assertEqual(data["design_quality"]["supplemental_issues"], [])
            html = render_snapshot(data)
            self.assertIn(prompt, html)
            self.assertIn("我们在造什么", html)
            self.assertIn("原始生成 Prompt", html)
            self.assertIn("EVIDENCE_COMPLETE", html)
            self.assertIn("设计输入与包装完整性", html)
            self.assertIn("Worker 补充图（worker_supplemental）", html)
            self.assertIn("review_authority=false", html)
            self.assertNotIn("来自同一权威 STEP 的视觉审查图", html)
            self.assertNotIn("视觉证据完整性", html)

            evaluation_path = run / "design_quality" / "evaluation.json"
            legacy = json.loads(evaluation_path.read_text(encoding="utf-8"))
            legacy["status"] = "failed"
            legacy["issues"] = []
            legacy.pop("authority")
            legacy.pop("evidence_complete")
            legacy.pop("reported_findings")
            _write_json(evaluation_path, legacy)
            migrated = normalize_run(run, workspace_root=root, mode="snapshot")
            self.assertEqual(
                migrated["design_quality"]["status"],
                "evidence_complete",
            )
            self.assertEqual(migrated["stages"][0]["status"], "evidence_complete")

            legacy["status"] = "evidence_complete"
            legacy["issues"] = [
                {
                    "code": "VISUAL_REVIEW_REQUIRED_ROLE_MISSING",
                    "field": "render_assets",
                    "message": "required view missing",
                }
            ]
            _write_json(evaluation_path, legacy)
            supplemental = normalize_run(run, workspace_root=root, mode="snapshot")
            self.assertEqual(
                supplemental["design_quality"]["status"],
                "evidence_complete",
            )
            self.assertEqual(supplemental["stages"][0]["status"], "evidence_complete")
            self.assertEqual(supplemental["design_quality"]["issues"], [])
            self.assertEqual(
                supplemental["design_quality"]["supplemental_issues"][0]["code"],
                "VISUAL_REVIEW_REQUIRED_ROLE_MISSING",
            )

            legacy["issues"] = [
                {
                    "code": "DESIGN_INPUT_FIELD_MISSING",
                    "field": "original_prompt",
                    "message": "required design-input field is empty: original_prompt",
                }
            ]
            _write_json(evaluation_path, legacy)
            incomplete = normalize_run(run, workspace_root=root, mode="snapshot")
            self.assertEqual(
                incomplete["design_quality"]["status"],
                "evidence_incomplete",
            )
            self.assertEqual(incomplete["stages"][0]["status"], "evidence_incomplete")

            legacy["status"] = "stale"
            legacy["issues"] = [
                {
                    "code": "VISUAL_REVIEW_STEP_STALE",
                    "field": "assembly_step_sha256",
                    "message": "visual review does not match the authoritative STEP hash",
                }
            ]
            _write_json(evaluation_path, legacy)
            stale_visual = normalize_run(run, workspace_root=root, mode="snapshot")
            self.assertEqual(
                stale_visual["design_quality"]["status"],
                "evidence_complete",
            )
            self.assertEqual(stale_visual["stages"][0]["status"], "evidence_complete")
            self.assertEqual(stale_visual["design_quality"]["issues"], [])
            self.assertEqual(
                stale_visual["design_quality"]["supplemental_issues"][0]["code"],
                "VISUAL_REVIEW_STEP_STALE",
            )

            legacy["issues"] = [
                {
                    "code": "DESIGN_INPUT_FIELD_MISSING",
                    "field": "original_prompt",
                    "message": "required design-input field is empty: original_prompt",
                }
            ]
            _write_json(evaluation_path, legacy)
            stale_blocking = normalize_run(run, workspace_root=root, mode="snapshot")
            self.assertEqual(
                stale_blocking["design_quality"]["status"],
                "evidence_incomplete",
            )
            self.assertEqual(stale_blocking["stages"][0]["status"], "evidence_incomplete")

            legacy["status"] = "evidence_incomplete"
            legacy["issues"] = [
                {
                    "code": "VISUAL_REVIEW_STEP_STALE",
                    "field": "assembly_step_sha256",
                    "message": "visual review does not match the authoritative STEP hash",
                }
            ]
            _write_json(evaluation_path, legacy)
            incomplete_visual = normalize_run(
                run, workspace_root=root, mode="snapshot"
            )
            self.assertEqual(
                incomplete_visual["design_quality"]["status"],
                "evidence_complete",
            )
            self.assertEqual(
                incomplete_visual["stages"][0]["status"], "evidence_complete"
            )
            self.assertEqual(incomplete_visual["design_quality"]["issues"], [])
            self.assertEqual(
                incomplete_visual["design_quality"]["supplemental_issues"][0]["code"],
                "VISUAL_REVIEW_STEP_STALE",
            )

            legacy["issues"] = []
            _write_json(evaluation_path, legacy)
            incomplete_empty = normalize_run(
                run, workspace_root=root, mode="snapshot"
            )
            self.assertEqual(
                incomplete_empty["design_quality"]["status"],
                "evidence_incomplete",
            )
            self.assertEqual(
                incomplete_empty["stages"][0]["status"], "evidence_incomplete"
            )

            legacy["issues"] = [
                {
                    "code": "DESIGN_INPUT_FIELD_MISSING",
                    "field": "original_prompt",
                    "message": "required design-input field is empty: original_prompt",
                }
            ]
            _write_json(evaluation_path, legacy)
            incomplete_blocking = normalize_run(
                run, workspace_root=root, mode="snapshot"
            )
            self.assertEqual(
                incomplete_blocking["design_quality"]["status"],
                "evidence_incomplete",
            )
            self.assertEqual(
                incomplete_blocking["stages"][0]["status"],
                "evidence_incomplete",
            )

    def test_legacy_run_marks_prompt_as_missing_instead_of_inventing_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            package.mkdir()
            _write_json(package / "package.json", {"package_id": "legacy", "design_version": "1"})
            (package / "assembly.step").write_bytes(b"step")
            _write_json(package / "assembly_contract.json", {})
            intake = root / "intake.json"
            _write_json(intake, {"status": "ACCEPTED", "summary": {}, "package_summary": {}})
            run = root / "runs" / "legacy"
            _write_json(
                run / "plans" / "legacy.json",
                {
                    "capsule_id": "legacy",
                    "package_dir": str(package),
                    "intake_report": str(intake),
                },
            )
            _write_json(run / "run_status.json", {"status": "RUNNING"})
            data = normalize_run(run, workspace_root=root)
            self.assertEqual(data["stages"][0]["status"], "warning")
            self.assertFalse(data["design_quality"]["recorded"])
            self.assertEqual(data["design_quality"]["original_prompt"], "")


if __name__ == "__main__":
    unittest.main()
