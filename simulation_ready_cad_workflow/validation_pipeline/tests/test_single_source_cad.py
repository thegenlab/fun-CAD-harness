from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from simulation_ready_cad_workflow.validation_pipeline.mechanism.dispatch import (
    _enforce_single_source_cad_policy,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class SingleSourceCadPolicyTests(unittest.TestCase):
    def _package(self, root: Path, *, assembly_hash: str = "a" * 64) -> Path:
        package = root / "package"
        package.mkdir()
        (package / "package.json").write_text(
            json.dumps(
                {
                    "assembly_file": "assembly.step",
                    "single_source_cad": {
                        "policy": "required_v1",
                        "authoritative_artifact": "assembly.step",
                        "derived_assets": "hash_bound_and_regenerated",
                    },
                    "artifacts": [
                        {"path": "assembly.step", "sha256": assembly_hash}
                    ],
                }
            ),
            encoding="utf-8",
        )
        return package

    def test_policy_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary))
            manifest_path = package / "package.json"
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            payload.pop("single_source_cad")
            manifest_path.write_text(json.dumps(payload), encoding="utf-8")
            evaluation = SimpleNamespace(
                evidence_complete=True,
                status="evidence_complete",
                assembly_step_sha256="a" * 64,
                issues=(),
            )
            with self.assertRaises(MechanismValidationError) as caught:
                _enforce_single_source_cad_policy(package, evaluation)
            self.assertEqual(caught.exception.code, "SINGLE_SOURCE_CAD_POLICY_REQUIRED")

    def test_ready_hash_bound_authority_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary))
            evaluation = SimpleNamespace(
                evidence_complete=True,
                status="evidence_complete",
                assembly_step_sha256="a" * 64,
                issues=(),
            )
            _enforce_single_source_cad_policy(package, evaluation)

    def test_visual_review_for_an_old_step_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary))
            evaluation = SimpleNamespace(
                evidence_complete=True,
                status="evidence_complete",
                assembly_step_sha256="b" * 64,
                issues=(),
            )
            with self.assertRaises(MechanismValidationError) as caught:
                _enforce_single_source_cad_policy(package, evaluation)
            self.assertEqual(caught.exception.code, "SINGLE_SOURCE_CAD_HASH_MISMATCH")

    def test_incomplete_visual_evidence_blocks_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = self._package(Path(temporary))
            evaluation = SimpleNamespace(
                evidence_complete=False,
                status="evidence_incomplete",
                assembly_step_sha256="a" * 64,
                issues=({"code": "VISUAL_REVIEW_REQUIRED_ROLE_MISSING"},),
            )
            with self.assertRaises(MechanismValidationError) as caught:
                _enforce_single_source_cad_policy(package, evaluation)
            self.assertEqual(caught.exception.code, "SINGLE_SOURCE_CAD_REVIEW_BLOCKED")


if __name__ == "__main__":
    unittest.main()
