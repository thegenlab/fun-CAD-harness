import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import trimesh

from simulation_ready_cad_workflow.validation_pipeline.slim_harness.print_delivery import (
    PrintStaticEvidence, canonical, document_images,
)
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.print_delivery_worker import (
    compare_mesh, quantity_rows, transform,
)
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.review import review_passes
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.static_delivery import F7, VerifiedStaticEvidence


class PrintDeliveryReviewTests(unittest.TestCase):
    def setUp(self):
        self.static = {"id": "AC8", "kind": "function", "statement": "Deliver source, STEP, STLs and an illustrated booklet.",
                       "source_quote": "Deliver source, STEP, STLs and an illustrated booklet.", "verification": "Inspect executed exports and documentation."}
        self.mixed = {"id": "AC2", "kind": "function", "statement": "Use free rigid beads and identify bead-only dosing.",
                      "source_quote": "Use free rigid beads and identify bead-only dosing.", "verification": "Inspect bodies, operation and labels."}
        self.criteria = {"criteria": [self.static, self.mixed]}
        self.scene = {"tests": [{"id": "dosing"}]}
        self.evidence = [self.receipt(self.static, "static_only"), self.receipt(self.mixed, "supplemental")]
        self.result = {"status": "ready", "observations": [
            {"criterion_id": row["id"], "verdict": "met", "observed_fact": "Independent static inspection; motion result considered in final phase.",
             "evidence_paths": ["/receipt.json", "/guide.html"]} for row in (self.static, self.mixed)],
            "static_coverage": [{"criterion_id": row["id"], "receipt_ids": ["receipt:" + row["id"]]} for row in (self.static, self.mixed)],
            "test_coverage": [{"criterion_id": "AC2", "test_ids": ["dosing"], "appropriate": True, "reason": "Actual free-bead operation."}]}

    @staticmethod
    def receipt(criterion, mode):
        return PrintStaticEvidence("receipt:" + criterion["id"], "/receipt.json", "sha", "/scene.json", "criteria-sha",
                                   ("/receipt.json", "/inspection.json", "/guide.html"), canonical(criterion), mode)

    def test_executed_static_and_mixed_coverage_can_be_reviewed(self):
        self.assertTrue(review_passes(self.result, self.criteria, self.scene, static_evidence=self.evidence))
        self.assertTrue(review_passes(self.result, self.criteria, self.scene, final=True, static_evidence=self.evidence))

    def test_supplemental_receipt_never_replaces_motion_test(self):
        self.result["test_coverage"] = []
        self.assertFalse(review_passes(self.result, self.criteria, self.scene, static_evidence=self.evidence))

    def test_static_receipt_never_replaces_independent_observation(self):
        self.result["observations"][0]["verdict"] = "unobservable"
        self.assertFalse(review_passes(self.result, self.criteria, self.scene, static_evidence=self.evidence))

    def test_receipt_for_other_criterion_cannot_be_cited(self):
        self.result["static_coverage"][0]["receipt_ids"] = ["receipt:AC2"]
        self.assertFalse(review_passes(self.result, self.criteria, self.scene, static_evidence=self.evidence))

    def test_changed_frozen_wording_does_not_keep_static_exemption(self):
        changed = copy.deepcopy(self.criteria)
        changed["criteria"][0]["statement"] += " and dispense exactly three beads."
        self.assertFalse(review_passes(self.result, changed, self.scene, static_evidence=self.evidence))

    def test_removed_stale_receipt_invalidates_previous_review(self):
        self.assertFalse(review_passes(self.result, self.criteria, self.scene, static_evidence=[]))

    def test_unrelated_observation_path_is_not_executed_evidence(self):
        self.result["observations"][0]["evidence_paths"] = ["/guide.html"]
        self.assertFalse(review_passes(self.result, self.criteria, self.scene, static_evidence=self.evidence))

    def test_legacy_f7_keeps_its_receipt_and_observation_contract(self):
        evidence = VerifiedStaticEvidence("old-receipt", "/old.json", "sha", "/scene.json", "c", ("/old.json", "/old-inspection.json"))
        result = {"status": "ready", "test_coverage": [], "static_coverage": [{"criterion_id": "F7", "receipt_ids": ["old-receipt"]}],
                  "observations": [{"criterion_id": "F7", "verdict": "met", "observed_fact": "Inspected four operations and source correspondence.", "evidence_paths": ["/old.json"]}]}
        self.assertTrue(review_passes(result, {"criteria": [F7]}, self.scene, static_evidence=[evidence]))
        self.assertFalse(review_passes(result, {"criteria": [F7]}, self.scene))


class PrintDeliveryGeometryTests(unittest.TestCase):
    def test_rigid_transform_rejects_scaled_stl_placement(self):
        matrix = np.eye(4)
        matrix[0, 0] = 1.01
        with self.assertRaisesRegex(ValueError, "rigid transform"):
            transform(matrix)

    def test_export_comparison_rejects_filled_cavity(self):
        outer = trimesh.creation.box([10, 10, 10])
        inner = trimesh.creation.box([4, 4, 4])
        inner.invert()
        hollow = trimesh.util.concatenate([outer, inner])
        with self.assertRaisesRegex(ValueError, "Export/source mismatch"):
            compare_mesh(hollow, outer, hollow.volume)

    def test_export_comparison_accepts_unchanged_shape(self):
        mesh = trimesh.creation.box([20, 10, 4])
        report = compare_mesh(mesh, mesh.copy(), mesh.volume)
        self.assertEqual(report["relative_volume_error"], 0)

    def test_actual_inventory_structures_and_duplicate_rejection(self):
        for payload in ([{"part": "x"}], {"parts": [{"id": "x"}]}, {"items": [{"part": "x"}]}):
            self.assertEqual(set(quantity_rows(payload)), {"x"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            quantity_rows([{"part": "x"}, {"part": "x"}])

    def test_booklet_image_dependencies_are_part_of_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "booklet").mkdir()
            (root / "booklet/guide.html").write_text('<img src="images/step_01.png"><img src="data:image/png;base64,AA==">', encoding="utf-8")
            self.assertEqual(document_images(root, ["booklet/guide.html"]), {"booklet/images/step_01.png"})


if __name__ == "__main__":
    unittest.main()
