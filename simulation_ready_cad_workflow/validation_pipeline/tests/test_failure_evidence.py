from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.failure_evidence import (
    carry_forward_failure_snapshots,
    informative_sample_index,
    select_representative_failures,
)


class FailureEvidenceSelectionTests(unittest.TestCase):
    def test_selects_first_worst_and_closest_without_duplicates(self) -> None:
        comparisons = (
            {
                "load_case_id": "first",
                "verdict": "PHYSICS_INCONCLUSIVE",
                "failure_reasons": ["missing"],
            },
            {
                "load_case_id": "worst",
                "verdict": "PHYSICAL_FAILURE",
                "failure_reasons": ["drift", "coupling"],
            },
            {
                "load_case_id": "closest",
                "verdict": "CLAIM_MISMATCH",
                "failure_reasons": ["range"],
            },
            {"load_case_id": "pass", "verdict": "PASS", "failure_reasons": []},
        )

        selected = select_representative_failures(comparisons)

        self.assertEqual(
            [(role, item["load_case_id"]) for role, item in selected],
            [
                ("first_failure", "first"),
                ("worst_failure", "worst"),
                ("closest_to_pass", "closest"),
            ],
        )

    def test_informative_sample_favors_combined_normalized_violation(self) -> None:
        samples = (
            {"time_s": 0.0, "max_joint_drift_m": 0.0, "coupling_error_m": 0.0},
            {"time_s": 0.1, "max_joint_drift_m": 0.5, "coupling_error_m": 0.7},
            {"time_s": 0.2, "max_joint_drift_m": 1.0, "coupling_error_m": 0.1},
        )

        self.assertEqual(informative_sample_index(samples), 1)

    def test_carries_diverse_failure_snapshots_into_new_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            old_run = workspace / "runs" / "old"
            new_run = workspace / "runs" / "new"
            (old_run / "plans").mkdir(parents=True)
            (new_run / "plans").mkdir(parents=True)
            (old_run / "plans" / "plan.json").write_text(
                json.dumps({"package_id": "machine", "capsule_id": "drive"}),
                encoding="utf-8",
            )
            cases = []
            for index, (load_id, reason) in enumerate(
                (
                    ("light", "physical_joint_drift_exceeded"),
                    ("loaded", "physical_target_not_reached"),
                    ("held", "physical_runtime_timeout"),
                    ("extra", "physical_dwell_not_held"),
                )
            ):
                snapshot = old_run / "snapshots" / f"{index}.png"
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(b"png" + bytes([index]))
                cases.append(
                    {
                        "capsule_id": "drive",
                        "load_case_id": load_id,
                        "verdict": "PHYSICAL_FAILURE",
                        "failure_reasons": [reason],
                        "snapshot": snapshot.relative_to(workspace).as_posix(),
                    }
                )
            summary_path = (
                old_run / "results" / "drive" / "failure_evidence" / "summary.json"
            )
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")

            summary, generated = carry_forward_failure_snapshots(
                SimpleNamespace(package_id="machine", capsule_id="drive"),
                workspace_root=workspace,
                output_dir=new_run,
            )

            self.assertEqual(summary["case_count"], 3)
            self.assertTrue(summary["snapshots_copied_into_current_run"])
            self.assertEqual(
                len({item["load_case_id"] for item in summary["cases"]}), 3
            )
            self.assertTrue(
                all(item["missing_requirement"] for item in summary["cases"])
            )
            self.assertTrue(
                all(
                    (workspace / item["snapshot"]).is_file()
                    for item in summary["cases"]
                )
            )
            self.assertIn(new_run / "historical_failure_evidence.json", generated)


if __name__ == "__main__":
    unittest.main()
