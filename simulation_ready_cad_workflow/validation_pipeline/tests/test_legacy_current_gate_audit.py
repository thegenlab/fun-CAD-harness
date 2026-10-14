from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.mechanism.legacy_current_gate_audit import (
    audit_run,
    build_report,
    render_markdown,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
        newline="",
    )


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class LegacyCurrentGateAuditTests(unittest.TestCase):
    def _fixture(self, root: Path, *, geometry_passed: bool = True) -> Path:
        run = root / "runs" / "legacy-machine-r1"
        _write_json(
            run / "run_status.json",
            {"status": "WHOLE_MACHINE_TASK_PASSED"},
        )
        _write_json(
            run / "plans" / "capsule.json",
            {
                "family": "revolute",
                "adapter_spec": {"adapter_id": "legacy"},
            },
        )
        _write_json(
            run / "results" / "capsule" / "certificate.json",
            {"verdict": "KINEMATIC_CERTIFIED"},
        )
        _write_json(
            run / "traces" / "capsule" / "load" / "physical.trace.json",
            {
                "branch": "physical",
                "branch_semantics": {
                    "physical_compliant_supports": [{"support_id": "legacy"}]
                },
            },
        )
        mesh = run / "derived_adapters" / "capsule" / "body.obj"
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_text("v 0 0 0\n", encoding="utf-8", newline="")
        _write_json(
            run / "migration" / "legacy_sampled_physical_audit_v1.json",
            {
                "status": (
                    "LEGACY_SAMPLED_PHYSICAL_AUDIT_PASSED"
                    if geometry_passed
                    else "LEGACY_SAMPLED_PHYSICAL_AUDIT_FAILED"
                ),
                "passed": geometry_passed,
                "trace_count": 1,
                "passed_trace_count": int(geometry_passed),
                "failed_trace_count": int(not geometry_passed),
                "traces": [
                    {
                        "sampling": {"selected_sample_count": 4},
                        "geometry_legality": {
                            "passed": geometry_passed,
                            "sample_count": 4,
                            "max_penetration_m": (
                                0.0001 if geometry_passed else 0.004
                            ),
                        },
                    }
                ],
            },
        )
        return run

    def test_read_only_audit_preserves_historical_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._fixture(root)
            before = _hashes(run)
            result = audit_run(root, run)
            after = _hashes(run)

        self.assertEqual(before, after)
        self.assertFalse(result["historical_artifacts_modified"])
        self.assertFalse(result["pychrono_rerun"])
        self.assertFalse(
            result[
                "geometry_repair_and_physics_rerun_required_for_current_certificate"
            ]
        )
        self.assertIn(
            "collider_fidelity_not_recorded_or_not_passing",
            result["current_gate_gaps"],
        )
        self.assertIn(
            "typed_acceptance_contract_not_authoritative",
            result["current_gate_gaps"],
        )
        self.assertIn(
            "component_contact_intent_not_authoritative",
            result["current_gate_gaps"],
        )
        self.assertEqual(result["recommended_action"], "DERIVE_MISSING_EVIDENCE_BEFORE_RERUN")

    def test_failed_saved_pose_audit_requires_new_physical_run_for_recognition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = self._fixture(root, geometry_passed=False)
            result = audit_run(root, run)

        self.assertTrue(
            result[
                "geometry_repair_and_physics_rerun_required_for_current_certificate"
            ]
        )
        self.assertEqual(
            result["recommended_action"],
            "REPAIR_AUTHORITATIVE_CAD_THEN_NEW_FULL_RUN",
        )

    def test_skip_prefix_does_not_read_case_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = build_report(
                root,
                [root / "runs" / "scissor-lift-v1-r10"],
                skip_prefixes=["scissor-lift"],
            )

        case = report["cases"][0]
        self.assertEqual(case["status"], "SKIPPED_BY_SCOPE")
        self.assertFalse(case["run_directory_read"])
        self.assertEqual(report["skipped_case_count"], 1)
        self.assertIn("未读取", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
