from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.mechanism.legacy_sampled_audit import (
    audit_legacy_run,
)
from simulation_ready_cad_workflow.validation_pipeline.mechanism.models import (
    MechanismValidationError,
)


class LegacySampledAuditTests(unittest.TestCase):
    def test_scissor_lift_run_is_explicitly_excluded_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "scissor-lift-engineering-r11"
            run.mkdir()

            with self.assertRaises(MechanismValidationError) as caught:
                audit_legacy_run(workspace_root=root, run_dir=run)

            self.assertEqual(caught.exception.code, "LEGACY_AUDIT_RUN_EXCLUDED")
            self.assertFalse((run / "migration").exists())


if __name__ == "__main__":
    unittest.main()
