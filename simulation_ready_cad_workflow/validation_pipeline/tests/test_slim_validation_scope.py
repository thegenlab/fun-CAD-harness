"""Candidate-only geometry/scope regressions; no clock physics is simulated."""
from __future__ import annotations
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

WORKSPACE = Path.cwd()
ROOT = WORKSPACE / ".runtime_runs/slim_harness_20260907/equivalent_scope_tests"

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.audit import audit_motion
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.geometry import prepare_geometry
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.publication import read_result
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.review import review_passes, review_submission
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.validation_scope import (
    ValidationScopeError, make_validation_scope, validate_validation_scope,
    point_in_flexible_contact_region,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(position, quaternion=(1, 0, 0, 0)):
    return {"position_abs_m": list(position), "rotation_abs_quat_wxyz": list(quaternion)}


def candidate_setup(self):
    fixtures = ROOT / "test_artifacts"
    fixtures.mkdir(parents=True,exist_ok=True)
    self.root = Path(tempfile.mkdtemp(prefix=self._testMethodName + "_", dir=fixtures))
    self.workspace = WORKSPACE
    environment = mock.patch.dict(os.environ, {"SIMREADY_STEP_SHAPE_CACHE_DIR": str(self.root / "step_cache")})
    environment.start()
    self.addCleanup(environment.stop)


class ScopeCandidateTests(unittest.TestCase):
    setUp = candidate_setup

    def prepare(self, *, third=False, spring=False):
        import xml.parsers.expat
        from build123d import Box, Pos, export_step
        export_step(Pos(100, 200, 300) * Box(20, 20, 6), self.root / "mixed.step")
        export_step(Pos(100, 200, 340) * Box(1, 1, 1), self.root / "arbor.step")
        bodies = [{"id": key, "source": key + ".step", "fixed": False} for key in ("mixed", "arbor")]
        for enabled, key in ((third, "other"), (spring, "spring")):
            if enabled:
                export_step(Pos(150, 200, 300) * Box(1, 1, 1), self.root / (key + ".step"))
                bodies.append({"id": key, "source": key + ".step", "fixed": False})
        source = self.root / "scene.json"
        source.write_text(json.dumps({"schema_version": "slim_scene_v1", "units": "mm", "bodies": bodies}))
        geometry = prepare_geometry(source, self.root / "geometry")
        self.assertEqual([], geometry["errors"])
        return geometry

    def scope(self, geometry, *, refs=(), regions=True):
        source = self.root / "region_source.txt"
        source.write_text("Synthetic test declaration: only mixed/arbor, radius 2 mm, source Z +/- 1.5 mm.\n")
        rows = [{"id": "local_sleeve", "body_pair": ["mixed", "arbor"], "region_owner_body": "mixed",
                 "shape": "cylinder", "axis_origin_source_m": [.1, .2, .3],
                 "axis_direction_source": [0, 0, 1], "radius_m": .002,
                 "axial_interval_m": [-.0015, .0015],
                 "source_refs": [{"path": str(source), "sha256": digest(source), "locator": "line 1"}]}] if regions else []
        return make_validation_scope(geometry["bodies"], elastic_reference_body_ids=list(refs),
            flexible_contact_regions=rows, source_bindings={str(source): digest(source)},
            model_binding={"test_fixture": "declared passive equivalent model, not a clock result"})

    def test_region_is_pair_local_radially_and_axially_bounded_after_rotation(self):
        geometry = self.prepare(third=True)
        q = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        scoped = {**geometry, "validation_scope": self.scope(geometry)}
        def sample(arbor=(1, 2, 3), other=(1.05, 2, 3)):
            return {"time_s": 0., "body_poses": {"mixed": pose([1, 2, 3], q),
                    "arbor": pose(arbor, q), "other": pose(other, q)}}
        raw = audit_motion(geometry, [sample()], self.root / "raw_overlap.json")
        self.assertFalse(raw["passed"])
        allowed = audit_motion(scoped, [sample()], self.root / "within_local_region.json")
        self.assertTrue(allowed["passed"])
        self.assertEqual("slim_geometry_audit_v2", allowed["schema_version"])
        self.assertGreater(allowed["equivalent_elastic_exclusions"]["excluded_intersection_volume_m3_sum"], 9e-10)
        for name, current in (
            ("outside_radius", sample(arbor=(1.006, 2, 3))),
            ("partial_outside_radius", sample(arbor=(1.0018, 2, 3))),
            ("outside_axial_interval", sample(arbor=(1, 1.9976, 3))),
            ("different_body_pair", sample(arbor=(1.04, 2, 3), other=(1, 2, 3))),
        ):
            with self.subTest(name=name):
                result = audit_motion(scoped, [current], self.root / (name + ".json"))
                self.assertFalse(result["passed"])
                self.assertGreater(result["max_overlap_m3"], 1e-12)

    def test_released_support_restores_original_geometry_audit(self):
        geometry = self.prepare()
        original_scope = self.scope(geometry)
        regions = copy.deepcopy(original_scope["flexible_contact_regions"])
        regions[0]["active_element_id"] = "shaft_support"
        scope = make_validation_scope(geometry["bodies"], flexible_contact_regions=regions,
            source_bindings=original_scope["source_bindings"], model_binding=original_scope["model_binding"])
        prepared = {**geometry, "validation_scope": scope}
        def sample(t, active):
            return {"time_s": t, "body_poses": {"mixed": pose([1,2,3]), "arbor": pose([1,2,3])},
                    "equivalent_elasticity": {"elements": {"shaft_support": {"active": active}}}}
        accepted = audit_motion(prepared, [sample(0., True)], self.root / "support_active.json")
        self.assertTrue(accepted["passed"])
        released = audit_motion(prepared, [sample(0., True), sample(.001, False)], self.root / "support_released.json")
        self.assertFalse(released["passed"])
        self.assertGreater(released["max_overlap_m3"], 1e-12)
        missing = sample(0., True)
        del missing["equivalent_elasticity"]
        with self.assertRaisesRegex(RuntimeError, "recorded support activity"):
            audit_motion(prepared, [missing], self.root / "support_missing_state.json")

    def test_reference_bodies_have_no_claim_of_deformation_and_initial_audit_remains_full(self):
        geometry = self.prepare(spring=True)
        current = {"time_s": 0., "body_poses": {"mixed": pose([1, 2, 3]),
            "arbor": pose([1.04, 2, 3]), "spring": pose([1, 2, 3])}}
        original = audit_motion(geometry, [current], self.root / "full_original.json")
        self.assertFalse(original["passed"])
        scope = self.scope(geometry, refs=["spring"], regions=False)
        result = audit_motion({**geometry, "validation_scope": scope}, [current], self.root / "equivalent_reference.json")
        self.assertTrue(result["passed"])
        self.assertEqual(["spring"], result["validation_scope"]["elastic_reference_body_ids"])
        self.assertNotIn("spring", result["containment"]["source_part_counts"])
        self.assertFalse(result["equivalent_elastic_exclusions"]["contact_records_modified"])
        self.assertTrue(scope["excluded_claims"])
        omitted = copy.deepcopy(current)
        del omitted["body_poses"]["spring"]
        with self.assertRaisesRegex(RuntimeError, "omitted"):
            audit_motion({**geometry, "validation_scope": scope}, [omitted], self.root / "missing_pose.json")

    def test_contact_region_helper_uses_owner_source_frame_and_never_other_pair(self):
        geometry = self.prepare()
        scope = self.scope(geometry)
        q = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        poses = {"mixed": pose([1, 2, 3], q), "arbor": pose([1, 2, 3], q)}
        centers = {body["id"]: body["source_position_m"] for body in geometry["bodies"]}
        self.assertTrue(point_in_flexible_contact_region(scope, ["arbor", "mixed"], [1, 2, 3], poses, centers))
        self.assertFalse(point_in_flexible_contact_region(scope, ["arbor", "mixed"], [1, 1.997, 3], poses, centers))
        self.assertFalse(point_in_flexible_contact_region(scope, ["arbor", "mixed"], [1.004, 2, 3], poses, centers))
        self.assertFalse(point_in_flexible_contact_region(scope, ["other", "mixed"], [1, 2, 3], poses, centers))

    def test_scope_is_complete_and_source_bound(self):
        geometry = self.prepare()
        scope = self.scope(geometry)
        self.assertIs(scope, validate_validation_scope(scope, geometry["bodies"]))
        changed = copy.deepcopy(scope)
        changed["flexible_contact_regions"][0]["radius_m"] *= 10
        with self.assertRaises(ValidationScopeError):
            validate_validation_scope(changed)
        changed_bodies = copy.deepcopy(geometry["bodies"])
        changed_bodies[0]["source_solid_ids"].append("solid_999")
        with self.assertRaises(ValidationScopeError):
            validate_validation_scope(scope, changed_bodies)
        with self.assertRaises(ValidationScopeError):
            make_validation_scope(geometry["bodies"], elastic_reference_body_ids=["mixed", "arbor"], model_binding={"model": 1})
        with self.assertRaises(ValidationScopeError):
            make_validation_scope(geometry["bodies"], elastic_reference_body_ids=["arbor"])

    def publication_fixture(self):
        geometry = self.prepare()
        scope = self.scope(geometry)
        project = self.root / "synthetic_publication_fixture"
        run = project / "runs/r01"
        design = run / "design"
        design.mkdir(parents=True)
        bodies = [{"id": body["id"], "source": body["source_file"]} for body in geometry["bodies"]]
        scene = {"bodies": bodies, "tests": [{"id": "fixture_test"}], "equivalent_elasticity": {"fixture": True}}
        scene_file = design / "scene.json"
        scene_file.write_text(json.dumps(scene))
        task = "Synthetic publication gate fixture, not a mechanical acceptance result."
        (project / "task.md").write_text(task)
        criteria = {"task_text": task, "criteria": [{"id": "F1", "kind": "function", "statement": task}]}
        (project / "criteria.json").write_text(json.dumps(criteria))
        review = {"status": "ready", "validation_scope_signature": scope["signature"],
                  "observations": [{"criterion_id": "F1", "verdict": "met", "observed_fact": "Fixture", "evidence_paths": ["fixture"]}],
                  "test_coverage": [{"criterion_id": "F1", "test_ids": ["fixture_test"], "appropriate": True}],
                  "static_coverage": []}
        value = {"schema_version": "slim_harness_result_v1", "revision": 1, "status": "PASS",
                 "validation_scope": scope, "criteria_sha256": digest(project / "criteria.json"),
                 "source_bindings": {str(scene_file): digest(scene_file), **scope["source_bindings"]},
                 "tests": [{"test_id": "fixture_test", "physical_pass": True, "validation_scope": scope,
                    "geometry_audit": {"schema_version": "slim_geometry_audit_v2", "passed": True, "validation_scope": scope}}],
                 "review": review}
        return project, run, scene, criteria, value

    def test_publication_rejects_missing_or_mislabeled_equivalent_evidence(self):
        _, run, _, _, value = self.publication_fixture()
        target = run / "result.json"
        target.write_text(json.dumps(value))
        self.assertTrue(read_result(run)["eligible_pass"])
        for name in ("missing_result_scope", "missing_test_scope", "old_full_geometry_schema", "missing_review_scope"):
            changed = copy.deepcopy(value)
            if name == "missing_result_scope": del changed["validation_scope"]
            if name == "missing_test_scope": del changed["tests"][0]["validation_scope"]
            if name == "old_full_geometry_schema": changed["tests"][0]["geometry_audit"]["schema_version"] = "slim_geometry_audit_v1"
            if name == "missing_review_scope": del changed["review"]["validation_scope_signature"]
            target.write_text(json.dumps(changed))
            with self.subTest(name=name):
                self.assertFalse(read_result(run)["eligible_pass"])
                self.assertEqual("INCOMPLETE_EVIDENCE", read_result(run)["status"])

    def test_policy_changes_and_model_sources_invalidate_publication(self):
        project, run, _, _, value = self.publication_fixture()
        policy = project / "modeling_policy.json"
        policy.write_text(json.dumps({"selected_mode": "equivalent_elastic", "explicit_scope": "User-selected limits."}))
        value["source_bindings"][str(policy)] = digest(policy)
        (run / "result.json").write_text(json.dumps(value))
        self.assertTrue(read_result(run)["eligible_pass"])
        policy.write_text(json.dumps({"selected_mode": "equivalent_elastic", "explicit_scope": "Changed limits."}))
        self.assertEqual("STALE_EVIDENCE", read_result(run)["status"])
        policy.write_text(json.dumps({"selected_mode": "equivalent_elastic", "explicit_scope": "User-selected limits."}))
        (self.root / "region_source.txt").write_text("Changed model source\n")
        self.assertEqual("STALE_EVIDENCE", read_result(run)["status"])

    def test_preflight_prompt_distinguishes_author_omissions_from_actual_api_gaps(self):
        _, run, scene, criteria, value = self.publication_fixture()
        class Capture:
            def run_turn(self, workspace, prompt, turn_dir, timeout, **kwargs):
                self.prompt, self.schema = prompt, kwargs["response_schema"]
                return {}
        worker = Capture()
        review_submission(worker, self.root, criteria, run / "design/scene.json", [], self.root / "turn",
            validation_scope=value["validation_scope"],
            modeling_context={"selected_mode": "equivalent_elastic", "equivalent_elasticity_implemented": True})
        self.assertIn("return needs_revision with concrete author-supplied omissions", worker.prompt)
        self.assertIn("actual Harness gap", worker.prompt)
        self.assertIn("Future physics outcomes are still", worker.prompt)
        self.assertIn("validation_scope_signature", worker.schema["required"])
        self.assertTrue(review_passes(value["review"], criteria, scene, validation_scope=value["validation_scope"]))
        self.assertFalse(review_passes(value["review"], criteria, scene))


if __name__ == "__main__": unittest.main()
