"""User credit changes the allowance, never measured history or the light tier."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ..slim_harness import cli
from ..slim_harness.budget import ProjectBudget,apply_user_budget_grant,GRANT_LEDGER_NAME,get_tier,check_task_count
from ..slim_harness.physical_parts import check_physical_part_inventory
from . import test_budget_entrypoints as fixtures


class UserBudgetGrantTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.workspace=Path(self.temporary.name)
        self.project=cli.create_project(self.workspace,"grant_fixture","A slider must move.")
        budget=ProjectBudget(self.project,"light")
        budget.tracker._active_seconds=3600.25
        budget.tracker._wall_offset=4000
        budget.save()
        self.checkpoint=self.project/"budget_checkpoint.json"
        self.before=json.loads(self.checkpoint.read_text(encoding="utf-8"))

    def grant(self,identifier="recording-recovery",seconds=600):
        return apply_user_budget_grant(self.project,grant_id=identifier,seconds=seconds,
            authorization="Explicit user authorization fixture; not a real project grant",
            reason="Restore allowance for the evidenced recording failure",evidence_paths=["fixture-evidence.json"])

    def test_grant_preserves_original_checkpoint_and_restores_same_light_contract(self):
        checkpoint_bytes=self.checkpoint.read_bytes();state_bytes=(self.project/"workflow.json").read_bytes()
        self.grant()
        self.assertEqual(checkpoint_bytes,self.checkpoint.read_bytes())
        self.assertEqual(state_bytes,(self.project/"workflow.json").read_bytes())
        restored=ProjectBudget(self.project,"light")
        snapshot=restored.tracker.snapshot()
        self.assertEqual(self.before["active_seconds"],snapshot["active_seconds"])
        self.assertEqual(self.before["started_at"],snapshot["started_at"])
        self.assertGreaterEqual(snapshot["wall_seconds"],self.before["wall_seconds"])
        self.assertEqual(4200,snapshot["budget_limit_seconds"])
        self.assertEqual(3600,snapshot["base_budget_limit_seconds"])
        self.assertEqual(600,snapshot["granted_active_seconds"])
        self.assertAlmostEqual(599.75,snapshot["remaining_active_seconds"])
        self.assertEqual(get_tier("light"),restored.tier)
        self.assertEqual(60,restored.tier.simulation_timeout_seconds)
        self.assertTrue(check_task_count([1,2,3],restored.tier))
        self.assertTrue(check_physical_part_inventory({"status":"ready","machine_part_count":11},restored.tier))

    def test_same_grant_is_idempotent_and_conflicting_terms_are_rejected(self):
        first=self.grant();path=self.project/GRANT_LEDGER_NAME
        original=path.read_bytes()
        self.assertEqual(first,self.grant())
        self.assertEqual(original,path.read_bytes())
        self.assertEqual(1,len(json.loads(original)["grants"]))
        with self.assertRaisesRegex(ValueError,"different terms"):self.grant(seconds=601)
        self.assertEqual(original,path.read_bytes())

    def test_one_hour_cap_applies_to_single_and_cumulative_credit(self):
        with self.assertRaisesRegex(ValueError,"one hour"):self.grant(seconds=3600.1)
        self.assertFalse((self.project/GRANT_LEDGER_NAME).exists())
        self.grant("first",2000)
        with self.assertRaisesRegex(ValueError,"additional hour"):self.grant("second",2000)
        self.assertEqual(1,len(json.loads((self.project/GRANT_LEDGER_NAME).read_text())["grants"]))
        self.assertEqual(3600.25,json.loads(self.checkpoint.read_text())["active_seconds"])

    def test_public_resume_light_spends_new_work_from_credit_without_reset(self):
        self.grant()
        with patch.object(cli,"create_worker",return_value=fixtures.FakeWorker(author_status="needs_revision")),redirect_stdout(io.StringIO()) as output:
            status=cli.main(["--workspace",str(self.workspace),"--tier","light","resume","grant_fixture"])
        result=json.loads(output.getvalue())
        self.assertEqual(2,status)
        self.assertEqual("needs_revision",result["status"])
        snapshot=json.loads(self.checkpoint.read_text(encoding="utf-8"))
        self.assertGreater(snapshot["active_seconds"],self.before["active_seconds"])
        self.assertEqual(self.before["started_at"],snapshot["started_at"])
        self.assertEqual(4200,snapshot["budget_limit_seconds"])
        self.assertLess(snapshot["remaining_active_seconds"],599.75)
        self.assertEqual(self.before["tier_config"],snapshot["tier_config"])


if __name__=="__main__":unittest.main()
