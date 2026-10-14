from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.design_quality import (
    DesignQualityError,
    evaluate_design_quality,
    snapshot_design_quality,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _codes(items) -> set[str]:
    return {item["code"] for item in items}


def _appearance_plan() -> dict:
    return {
        "schema_version": "simulation_ready_cad_appearance_plan_v1",
        "product_identity": {
            "recognizable_as": "hinged engineering toolbox",
            "must_show": ["lid", "handle", "hinge"],
            "must_not_look_like": ["anonymous box"],
        },
        "components": [
            {
                "component_id": "lid",
                "name": "retained hinged lid",
                "owner_rigid_group": "lid_group",
                "visible_features": ["hinge knuckles"],
                "interfaces": ["hinge pin"],
                "positive_checks": ["lid is retained"],
                "negative_checks": ["not a floating slab"],
            }
        ],
        "view_requirements": [
            {
                "role": "open-pose",
                "kind": "pose",
                "purpose": "show the open lid",
                "elevation_deg": 20.0,
                "azimuth_deg": -45.0,
                "source_step": "review_poses/open.step",
            },
        ],
    }


def _fixture(root: Path) -> Path:
    revision = root / "revision1"
    package = revision / "package"
    package.mkdir(parents=True)
    step = package / "assembly.step"
    step.write_bytes(b"authoritative-step")
    _write_json(
        package / "package.json",
        {
            "assembly_file": "assembly.step",
            "single_source_cad": {"policy": "required_v1"},
        },
    )
    image = revision / "visual_review" / "round1" / "iso.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"png-render")
    _write_json(
        revision / "design_input.json",
        {
            "schema_version": "0.1",
            "original_prompt": "Build a recognizable hinged engineering toolbox.",
            "plain_language_goal": "A portable metal toolbox with a working lid.",
            "engineering_brief": {"mechanism": "hinged lid"},
            "visual_acceptance_checklist": ["recognizable silhouette"],
            "motion_review_roles": ["open-pose"],
            "appearance_plan": _appearance_plan(),
        },
    )
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
                            "role": role,
                            "label": role,
                            "path": "visual_review/round1/iso.png",
                            "sha256": _sha(image),
                        }
                        for role in ("iso", "front", "side", "top", "open-pose")
                    ],
                    "findings": [],
                }
            ],
        },
    )
    return package


class DesignQualityTests(unittest.TestCase):
    def test_static_visual_v2_requires_appearance_plan_iso_and_planned_views(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            design_path = package.parent / "design_input.json"
            design = json.loads(design_path.read_text(encoding="utf-8"))
            design["appearance_plan"] = {
                "schema_version": "simulation_ready_cad_appearance_plan_v1",
                "product_identity": {
                    "recognizable_as": "hinged engineering toolbox",
                    "must_show": ["lid", "handle", "hinge"],
                    "must_not_look_like": ["anonymous box"],
                },
                "components": [
                    {
                        "component_id": "lid",
                        "name": "retained hinged lid",
                        "owner_rigid_group": "lid_group",
                        "visible_features": ["hinge knuckles"],
                        "interfaces": ["hinge pin"],
                        "positive_checks": ["lid is retained"],
                        "negative_checks": ["not a floating slab"],
                    }
                ],
                "view_requirements": [
                    {
                        "role": "hinge_section",
                        "kind": "section",
                        "purpose": "show hinge pin capture",
                        "elevation_deg": 20.0,
                        "azimuth_deg": -45.0,
                        "section_axis": "y",
                        "section_position_mm": 0.0,
                        "section_keep": "below",
                    },
                    {
                        "role": "open-pose",
                        "kind": "pose",
                        "purpose": "show the open lid",
                        "elevation_deg": 20.0,
                        "azimuth_deg": -45.0,
                        "source_step": "review_poses/open.step",
                    },
                ],
            }
            _write_json(design_path, design)
            review = package.parent / "visual_review/summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["static_visual_gate_version"] = 2
            image = package.parent / "visual_review/round1/iso.png"
            payload["rounds"][0]["render_assets"] = [
                {
                    "role": role,
                    "path": "visual_review/round1/iso.png",
                    "sha256": _sha(image),
                }
                for role in (
                    "front",
                    "rear",
                    "left",
                    "right",
                    "top",
                    "bottom",
                    "iso",
                    "hinge_section",
                    "open-pose",
                )
            ]
            _write_json(review, payload)

            result = evaluate_design_quality(package)

            self.assertEqual(result.status, "evidence_complete", result.issues)

    def test_complete_hash_bound_evidence_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            result = evaluate_design_quality(package)
            self.assertEqual(result.status, "evidence_complete")
            self.assertTrue(result.ready)
            self.assertTrue(result.evidence_complete)
            self.assertEqual(
                result.to_dict()["authority"],
                {
                    "scope": "design_input_package_completeness",
                    "certificate_authority": False,
                    "submitter_review_status_used": False,
                    "submitter_finding_status_used": False,
                },
            )
            self.assertEqual(result.reported_findings, {"P0": 0, "P1": 0, "P2": 0})
            self.assertEqual(result.open_findings, {"P0": 0, "P1": 0, "P2": 0})
            self.assertEqual(result.supplemental_issues, ())
            self.assertEqual(len(result.render_assets), 5)
            for asset in result.render_assets:
                self.assertEqual(asset["render_authority"], "worker_supplemental")
                self.assertIs(asset["review_authority"], False)
            self.assertEqual(result.to_dict()["supplemental_issues"], [])

    def test_static_visual_v1_uses_six_orthographic_base_views(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["static_visual_gate_version"] = 1
            image = package.parent / "visual_review" / "round1" / "iso.png"
            payload["rounds"][0]["render_assets"] = [
                {
                    "role": role,
                    "label": role,
                    "path": "visual_review/round1/iso.png",
                    "sha256": _sha(image),
                }
                for role in (
                    "front",
                    "rear",
                    "left",
                    "right",
                    "top",
                    "bottom",
                    "open-pose",
                )
            ]
            _write_json(review, payload)

            result = evaluate_design_quality(package)

            self.assertEqual(result.status, "evidence_complete")
            self.assertNotIn(
                "VISUAL_REVIEW_REQUIRED_ROLE_MISSING",
                {item["code"] for item in result.issues},
            )

    def test_missing_prompt_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            design = package.parent / "design_input.json"
            payload = json.loads(design.read_text(encoding="utf-8"))
            payload["original_prompt"] = ""
            _write_json(design, payload)
            result = evaluate_design_quality(package)
            self.assertEqual(result.status, "evidence_incomplete")
            self.assertIn("DESIGN_INPUT_FIELD_MISSING", {item["code"] for item in result.issues})

    def test_base_view_cannot_masquerade_as_motion_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            design = package.parent / "design_input.json"
            payload = json.loads(design.read_text(encoding="utf-8"))
            payload["motion_review_roles"] = ["iso"]
            _write_json(design, payload)
            result = evaluate_design_quality(package)
            codes = _codes(result.issues)
            self.assertIn("DESIGN_INPUT_FIELD_MISSING", codes)

    def test_absent_or_stale_submitter_visual_review_is_supplemental(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            review_root = package.parent / "visual_review"
            shutil.rmtree(review_root)

            absent = evaluate_design_quality(package)

            self.assertEqual(absent.status, "evidence_complete")
            self.assertTrue(absent.evidence_complete)
            self.assertEqual(absent.issues, ())
            self.assertIn("VISUAL_REVIEW_MISSING", _codes(absent.supplemental_issues))
            self.assertEqual(absent.render_assets, ())
            self.assertEqual(
                absent.to_dict()["supplemental_issues"][0]["code"],
                "VISUAL_REVIEW_MISSING",
            )
            self.assertNotIn(
                "render_static_visual_packet",
                absent.supplemental_issues[0]["message"],
            )

        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            (package / "assembly.step").write_bytes(b"changed-step")
            stale = evaluate_design_quality(package)
            self.assertEqual(stale.status, "evidence_complete")
            self.assertTrue(stale.ready)
            self.assertEqual(stale.issues, ())
            self.assertIn("VISUAL_REVIEW_STEP_STALE", _codes(stale.supplemental_issues))
            self.assertEqual(len(stale.render_assets), 5)
            for asset in stale.render_assets:
                self.assertEqual(asset["render_authority"], "worker_supplemental")
                self.assertIs(asset["review_authority"], False)

        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            (package / "assembly.step").unlink()
            missing_step = evaluate_design_quality(package)
            self.assertEqual(missing_step.status, "evidence_incomplete")
            self.assertIn("AUTHORITATIVE_STEP_MISSING", _codes(missing_step.issues))

    def test_invalid_appearance_plan_without_worker_packet_is_reported(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            design = package.parent / "design_input.json"
            payload = json.loads(design.read_text(encoding="utf-8"))
            payload["appearance_plan"] = {"schema_version": "not-a-plan"}
            _write_json(design, payload)
            shutil.rmtree(package.parent / "visual_review")

            result = evaluate_design_quality(package)

            self.assertEqual(result.status, "evidence_incomplete")
            self.assertIn("APPEARANCE_PLAN_INVALID", _codes(result.issues))
            self.assertIn("VISUAL_REVIEW_MISSING", _codes(result.supplemental_issues))

    def test_missing_required_review_view_is_supplemental(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["rounds"][0]["render_assets"] = [
                item
                for item in payload["rounds"][0]["render_assets"]
                if item["role"] != "top"
            ]
            _write_json(review, payload)
            result = evaluate_design_quality(package)
            self.assertEqual(result.status, "evidence_complete")
            self.assertEqual(result.issues, ())
            self.assertIn(
                "VISUAL_REVIEW_REQUIRED_ROLE_MISSING",
                _codes(result.supplemental_issues),
            )

    def test_top_level_render_assets_report_the_exact_required_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["render_assets"] = payload["rounds"][0].pop(
                "render_assets"
            )
            _write_json(review, payload)

            result = evaluate_design_quality(package)

            self.assertEqual(result.status, "evidence_complete")
            issue = next(
                item
                for item in result.supplemental_issues
                if item["code"] == "VISUAL_REVIEW_REQUIRED_ROLE_MISSING"
            )
            self.assertEqual(issue["field"], "rounds[].render_assets[]")
            self.assertIn("rounds[].render_assets[]", issue["message"])
            self.assertIn("top-level render_assets", issue["message"])

    def test_latest_round_cannot_rely_on_an_old_required_view(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            latest = json.loads(json.dumps(payload["rounds"][0]))
            latest["round"] = 2
            latest["render_assets"] = [
                item
                for item in latest["render_assets"]
                if item["role"] != "top"
            ]
            payload["rounds"].append(latest)
            _write_json(review, payload)
            result = evaluate_design_quality(package)
            self.assertEqual(result.status, "evidence_complete")
            self.assertIn(
                "VISUAL_REVIEW_REQUIRED_ROLE_MISSING",
                _codes(result.supplemental_issues),
            )

    def test_step_change_records_stale_review_as_supplemental(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            (package / "assembly.step").write_bytes(b"changed-step")
            result = evaluate_design_quality(package)
            self.assertEqual(result.status, "evidence_complete")
            self.assertTrue(result.ready)
            self.assertIn("VISUAL_REVIEW_STEP_STALE", _codes(result.supplemental_issues))

    def test_submitter_verdict_and_finding_state_have_no_pass_authority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = _fixture(Path(temporary))
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["status"] = "FAILED"
            payload["rounds"][0]["findings"] = [
                {
                    "finding_id": "missing-hinge-retainer",
                    "severity": "P1",
                    "status": "OPEN",
                    "summary": "The hinge pin has no visible retainer.",
                }
            ]
            _write_json(review, payload)
            result = evaluate_design_quality(package)
            self.assertEqual(result.status, "evidence_complete")
            self.assertTrue(result.ready)
            self.assertEqual(result.reported_findings["P1"], 1)
            self.assertEqual(result.open_findings["P1"], 1)

            payload["status"] = "PASSED"
            payload["rounds"][0]["findings"][0]["status"] = "RESOLVED"
            _write_json(review, payload)
            resolved_claim = evaluate_design_quality(package)
            self.assertEqual(resolved_claim.status, "evidence_complete")
            self.assertEqual(resolved_claim.reported_findings["P1"], 1)
            self.assertEqual(resolved_claim.open_findings["P1"], 0)
            self.assertFalse(
                resolved_claim.to_dict()["authority"]["submitter_finding_status_used"]
            )

            payload.pop("status")
            _write_json(review, payload)
            no_submitter_verdict = evaluate_design_quality(package)
            self.assertEqual(no_submitter_verdict.status, "evidence_complete")

    def test_snapshot_copies_only_declared_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _fixture(root)
            output = root / "run"
            result = snapshot_design_quality(package, output)
            self.assertTrue(result.ready)
            self.assertTrue((output / "design_quality" / "design_input.json").is_file())
            self.assertTrue((output / "design_quality" / "visual_review.json").is_file())
            assets = list((output / "design_quality" / "assets").glob("*"))
            self.assertEqual(len(assets), 5)
            self.assertEqual(_sha(assets[0]), _sha(package.parent / "visual_review" / "round1" / "iso.png"))
            evaluation = json.loads(
                (
                    output / "design_quality" / "evaluation.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(evaluation["render_assets"]), 5)
            self.assertEqual(evaluation["supplemental_issues"], [])
            for asset in evaluation["render_assets"]:
                self.assertFalse(Path(asset["path"]).is_absolute())
                self.assertTrue((output / asset["path"]).is_file())
                self.assertEqual(asset["render_authority"], "worker_supplemental")
                self.assertIs(asset["review_authority"], False)

    def test_snapshot_copies_only_hash_valid_optional_worker_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _fixture(root)
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["rounds"][0]["render_assets"][0]["sha256"] = "0" * 64
            _write_json(review, payload)
            output = root / "run"

            result = snapshot_design_quality(package, output)

            self.assertEqual(result.status, "evidence_complete")
            self.assertIn("VISUAL_REVIEW_ASSET_STALE", _codes(result.supplemental_issues))
            assets = list((output / "design_quality" / "assets").glob("*"))
            self.assertEqual(len(assets), 4)
            evaluation = json.loads(
                (output / "design_quality" / "evaluation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(len(evaluation["render_assets"]), 4)
            for asset in evaluation["render_assets"]:
                self.assertEqual(asset["render_authority"], "worker_supplemental")
                self.assertIs(asset["review_authority"], False)
                self.assertTrue((output / asset["path"]).is_file())

    def test_snapshot_rejects_asset_role_path_traversal_before_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = _fixture(root)
            review = package.parent / "visual_review" / "summary.json"
            payload = json.loads(review.read_text(encoding="utf-8"))
            payload["rounds"][0]["render_assets"][0]["role"] = (
                "../../../escaped"
            )
            _write_json(review, payload)

            output = root / "run"
            escaped = root / "escaped.png"
            with self.assertRaises(DesignQualityError) as raised:
                snapshot_design_quality(package, output)

            self.assertEqual(
                raised.exception.code,
                "VISUAL_REVIEW_ASSET_ROLE_INVALID",
            )
            self.assertFalse(escaped.exists())


if __name__ == "__main__":
    unittest.main()
