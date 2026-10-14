"""Standalone task-adaptation helpers; fixtures are not mechanical evidence."""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from ..cold_start.worker import RESPONSE_SCHEMA
from ..mechanism.io import write_json_atomic
from ..slim_harness.task_adaptation import (
    BUDGET_ACTIONS,
    CRITERIA_LOCK_NAME,
    FEEDBACK_INLINE_CHARS,
    adjustment_allowed,
    decision_context,
    decision_prompt,
    record_decision,
    review_and_apply_adjustment,
    revision_response_schema,
    review_task_change, task_definition,
)
from ..slim_harness.trace_storage import file_digest


ORIGINAL_TASK = (
    "Build a recognizable desk clock that keeps time for 24 hours and chimes."
)
ADJUSTED_TASK = (
    "Build a recognizable desk clock that keeps time for 1 hour."
)
ORIGINAL_CRITERIA = [
    {
        "id": "looks_like_clock",
        "kind": "appearance",
        "statement": "The machine is a recognizable desk clock",
        "source_quote": "recognizable desk clock",
        "verification": "Inspect original-geometry views",
    },
    {
        "id": "keeps_time",
        "kind": "function",
        "statement": "The clock keeps time for 24 hours",
        "source_quote": "keeps time for 24 hours",
        "verification": "Run the named timing test",
    },
]
APPROVED_CRITERIA = [
    {
        "id": "looks_like_clock",
        "kind": "appearance",
        "statement": "The machine is a recognizable desk clock",
        "source_quote": "recognizable desk clock",
        "verification": "Inspect original-geometry views",
    },
    {
        "id": "keeps_time",
        "kind": "function",
        "statement": "The clock keeps time for 1 hour",
        "source_quote": "keeps time for 1 hour",
        "verification": "Run the named timing test",
    },
]


class FakeReviewer:
    def __init__(self, result, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def run_turn(self, workspace, prompt, directory, timeout_seconds, **kwargs):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.calls.append({
            "workspace": workspace,
            "prompt": prompt,
            "directory": directory,
            "timeout_seconds": timeout_seconds,
            **kwargs,
        })
        write_json_atomic(directory / "response.json", self.result)
        (directory / "prompt.txt").write_text(prompt, encoding="utf-8")
        if self.error is not None:
            raise self.error
        return self.result


class TaskAdaptationTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[3] / ".runtime_runs" / "four_gaps_pair_20260912" / "task_adaptation"
        root.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.project = self.workspace / "project"
        self.design = self.project / "runs" / "r01" / "design"
        self.design.mkdir(parents=True)
        self.turn = self.project / "worker_turns" / "t001_adjust_task"
        self.result_path = self.project / "runs" / "r01" / "result.json"
        self.geometry_path = self.design / "part.step"
        self._write_project(task_adjustable=True)
        write_json_atomic(self.design / "scene.json", {"tests": [{"id": "clock", "duration_s": 1}]})
        self.author = FakeReviewer({"status": "ready", "task_change_decision": {"action": "accept", "reason": "Meaningful reduced target"}})
        self.advice = {"status": "timed_out", "raw_output": "The shorter target may hide failure; incomplete advice."}
        self.timer = patch("simulation_ready_cad_workflow.validation_pipeline.slim_harness.task_change_review.run_change_review", return_value=self.advice).start()
        self.addCleanup(patch.stopall)

    def _write_project(self, *, task_adjustable=None, frozen_sha=None, extra_request=None):
        (self.project / "task.md").write_text(ORIGINAL_TASK, encoding="utf-8")
        frozen = {
            "schema_version": "slim_frozen_criteria_v1",
            "task_file": str(self.project / "task.md"),
            "task_text": ORIGINAL_TASK,
            "criteria": ORIGINAL_CRITERIA,
            "reviewer_summary": "Fixture original criteria",
        }
        write_json_atomic(self.project / "criteria.json", frozen)
        write_json_atomic(
            self.project / CRITERIA_LOCK_NAME,
            {
                "schema_version": "slim_criteria_lock_v1",
                "criteria_sha256": file_digest(self.project / "criteria.json"),
                "task_sha256": file_digest(self.project / "task.md"),
            },
        )
        request = {"project_id": "adapt_fixture", "title": "adapt_fixture", "brief": ORIGINAL_TASK}
        if extra_request:
            request.update(extra_request)
        if task_adjustable is not None:
            request["task_adjustable"] = task_adjustable
        if frozen_sha is not None:
            request["frozen_experiment_sha256"] = frozen_sha
        write_json_atomic(
            self.project / "workflow.json",
            {
                "schema_version": "slim_harness_workflow_v1",
                "request": request,
                "revision": 1,
                "phase": "author_followup",
                "events": [
                    {
                        "kind": "criteria",
                        "result": {
                            "status": "ready",
                            "summary": frozen["reviewer_summary"],
                            "criteria": ORIGINAL_CRITERIA,
                        },
                    }
                ],
            },
        )
        self.geometry_path.write_text("fixture source; no geometry claim", encoding="utf-8")
        write_json_atomic(
            self.result_path,
            {"status": "PASS", "label": "historical-do-not-rewrite", "assertions": ["keep"]},
        )
        write_json_atomic(
            self.design / "task_adjustment.json",
            {"task_text": ADJUSTED_TASK, "reason": "Keep the clock; shorten the timed run.", "criteria": APPROVED_CRITERIA},
        )

    def _fingerprints(self):
        return {
            "task": (self.project / "task.md").read_bytes(),
            "criteria": (self.project / "criteria.json").read_bytes(),
            "lock": (self.project / CRITERIA_LOCK_NAME).read_bytes(),
            "result": self.result_path.read_bytes(),
            "workflow": (self.project / "workflow.json").read_bytes(),
            "geometry": self.geometry_path.read_bytes(),
            "proposal": (self.design / "task_adjustment.json").read_bytes(),
        }

    def _context(self, state=None, remaining=1847.25, allowed=None):
        state = state if state is not None else json.loads((self.project / "workflow.json").read_text(encoding="utf-8"))
        snapshot = {
            "remaining_active_seconds": remaining,
            "wall_seconds": 2201.5,
            "remaining_wall_seconds": 3998.5,
            "accounting_basis": "worker_elapsed_upper_estimate_plus_physics_elapsed",
        }
        context = decision_context(state, snapshot, {"summary": "timing test did not finish"})
        if allowed is not None:
            context["adjustment_allowed"] = allowed
        return context

    def _approved_worker(self, criteria=None):
        return FakeReviewer({
            "status": "ready",
            "summary": "Meaningful shorter timing run; clock function remains.",
            "criteria": criteria if criteria is not None else APPROVED_CRITERIA,
        })

    def test_adjustment_allowed_explicit_true_only(self):
        self.assertFalse(adjustment_allowed({}))
        self.assertFalse(adjustment_allowed({"request": {}}))
        self.assertFalse(adjustment_allowed({"request": {"task_adjustable": False}}))
        self.assertFalse(adjustment_allowed({"request": {"task_adjustable": "true"}}))
        self.assertFalse(adjustment_allowed({
            "request": {"task_adjustable": True, "frozen_experiment_sha256": "abc"},
        }))
        self.assertTrue(adjustment_allowed({"request": {"task_adjustable": True}}))

    def test_revision_response_schema_keeps_existing_fields(self):
        schema = revision_response_schema()
        self.assertEqual(RESPONSE_SCHEMA["required"], ["status", "summary", "evidence_paths", "issues"])
        for key in RESPONSE_SCHEMA["required"]:
            self.assertIn(key, schema["required"])
            self.assertEqual(schema["properties"][key], RESPONSE_SCHEMA["properties"][key])
        self.assertIn("budget_decision", schema["required"])
        self.assertEqual(
            schema["properties"]["budget_decision"]["properties"]["action"]["enum"],
            list(BUDGET_ACTIONS),
        )
        schema["properties"]["status"]["enum"] = ["changed"]
        self.assertNotEqual(RESPONSE_SCHEMA["properties"]["status"]["enum"], ["changed"])

    def test_decision_context_uses_actual_remaining_and_does_not_choose(self):
        state = {"request": {"task_adjustable": True}}
        remaining = 12.0
        snapshot = {
            "remaining_active_seconds": remaining,
            "wall_seconds": 4000.0,
            "remaining_wall_seconds": 11.5,
            "accounting_basis": "worker_elapsed_upper_estimate_plus_physics_elapsed",
        }
        context = decision_context(state, snapshot, "physics failed fixture")
        self.assertEqual(remaining, context["remaining_active_seconds"])
        self.assertEqual(11.5, context["remaining_wall_seconds"])
        self.assertEqual(4000.0, context["wall_seconds"])
        self.assertEqual(
            "worker_elapsed_upper_estimate_plus_physics_elapsed",
            context["accounting_basis"],
        )
        self.assertEqual(list(BUDGET_ACTIONS), context["allowed_actions"])
        self.assertNotIn("recommended_action", context)
        self.assertNotIn("chosen_action", context)
        prompt = decision_prompt(context)
        self.assertEqual(prompt, decision_prompt(context, self.workspace / "decision-context-00.json"))
        large = dict(context, feedback={"trace": "x" * (FEEDBACK_INLINE_CHARS * 3)})
        bounded = decision_prompt(large, self.workspace / "decision-context-00.json")
        self.assertLess(len(bounded), FEEDBACK_INLINE_CHARS + 2000)
        self.assertIn(str(self.workspace / "decision-context-00.json"), bounded)
        self.assertIn("12.0", prompt)
        self.assertIn("revise_mechanism", prompt)
        self.assertIn("adjust_task", prompt)
        self.assertNotIn("percent", prompt.lower())
        path = self.workspace / "decision.json"
        recorded = record_decision(path, {
            "status": "needs_revision",
            "summary": "keep revising the machine",
            "evidence_paths": [],
            "issues": [],
            "budget_decision": {
                "action": "revise_mechanism",
                "reason": "The mechanism is still the right next edit.",
                "tradeoff": "Spend remaining active time on the machine, not the task.",
            },
        }, context)
        self.assertEqual("revise_mechanism", recorded["budget_decision"]["action"])
        self.assertEqual(remaining, recorded["pre_turn_budget"]["remaining_active_seconds"])
        self.assertFalse(recorded["mechanical_certification"])
        plenty = decision_context(state, {**snapshot, "remaining_active_seconds": 10000.0}, "same")
        finish = record_decision(self.workspace / "finish.json", {
            "status": "ready",
            "summary": "stop with what we have",
            "evidence_paths": [],
            "issues": [],
            "budget_decision": {
                "action": "finish_partial",
                "reason": "Author chose to stop.",
                "tradeoff": "Accept incomplete evidence.",
            },
        }, plenty)
        self.assertEqual("finish_partial", finish["budget_decision"]["action"])
        self.assertEqual(10000.0, finish["pre_turn_budget"]["remaining_active_seconds"])

    def test_record_decision_rejects_disallowed_and_missing_choice(self):
        context = decision_context({"request": {}}, {"remaining_active_seconds": 50.0}, "fail")
        self.assertNotIn("adjust_task", context["allowed_actions"])
        with self.assertRaisesRegex(ValueError, "does not choose"):
            record_decision(self.workspace / "missing.json", {
                "status": "needs_revision",
                "summary": "no decision",
                "evidence_paths": [],
                "issues": [],
            }, context)
        with self.assertRaisesRegex(ValueError, "not allowed"):
            record_decision(self.workspace / "blocked.json", {
                "status": "needs_revision",
                "summary": "try to relax the task",
                "evidence_paths": [],
                "issues": [],
                "budget_decision": {
                    "action": "adjust_task",
                    "reason": "want a shorter task",
                    "tradeoff": "drop the long run",
                },
            }, context)
        self.assertFalse((self.workspace / "missing.json").exists())
        self.assertFalse((self.workspace / "blocked.json").exists())

    def apply(self):
        return review_and_apply_adjustment(object(), self.workspace, self.project, self.design,
                                           self.turn, self._context(), author=self.author, tier_name="light")

    def test_frozen_and_nonadjustable_changes_refused_before_review(self):
        for request in ({"task_adjustable": True, "frozen_sha": "deadbeef"}, {}):
            self._write_project(**request)
            before = self._fingerprints()
            with self.assertRaisesRegex(ValueError, "task_adjustable|frozen_experiment"):
                self.apply()
            self.assertEqual(before, self._fingerprints())
        self.timer.assert_not_called()

    def test_author_can_accept_incomplete_negative_advice_and_history_is_preserved(self):
        before = self._fingerprints()
        result = self.apply()
        self.assertTrue(result["applied"])
        history = Path(result["adjustment_history_dir"])
        self.assertEqual(before["task"], (history / "task.md").read_bytes())
        self.assertEqual(before["criteria"], (history / "criteria.json").read_bytes())
        self.assertEqual(before["lock"], (history / CRITERIA_LOCK_NAME).read_bytes())
        self.assertEqual(before["result"], self.result_path.read_bytes())
        self.assertEqual(before["workflow"], (self.project / "workflow.json").read_bytes())
        self.assertEqual(before["geometry"], self.geometry_path.read_bytes())
        self.assertEqual(ADJUSTED_TASK, (self.project / "task.md").read_text())
        self.assertEqual(APPROVED_CRITERIA, result["criteria"])
        decision = json.loads((history / "task_change_decision.json").read_text())
        self.assertEqual(self.advice, decision["advice"])
        self.assertIn(self.advice["raw_output"], self.author.calls[0]["prompt"])
        lock = json.loads((self.project / CRITERIA_LOCK_NAME).read_text())
        self.assertEqual(file_digest(self.project / "criteria.json"), lock["criteria_sha256"])
        self.assertEqual(file_digest(self.project / "task.md"), lock["task_sha256"])
        self.assertEqual(1, self.timer.call_count)

    def test_author_rejection_preserves_task_and_identical_proposal_reuses_advice(self):
        self.author.result["task_change_decision"]["action"] = "retain"
        before = self._fingerprints()
        self.assertFalse(self.apply()["applied"])
        self.assertFalse(self.apply()["applied"])
        self.assertEqual(before, self._fingerprints())
        self.assertEqual(1, self.timer.call_count)
        self.assertEqual(2, len(self.author.calls))

    def test_bad_source_quotes_are_code_rejected_without_model_call(self):
        proposal = json.loads((self.design / "task_adjustment.json").read_text())
        proposal["criteria"][1]["source_quote"] = "keeps time for 24 hours"
        write_json_atomic(self.design / "task_adjustment.json", proposal)
        before = self._fingerprints()
        with self.assertRaisesRegex(ValueError, "source quotes"):
            self.apply()
        self.timer.assert_not_called()
        self.assertEqual(before, self._fingerprints())

    def test_first_definition_and_geometry_only_edits_do_not_review(self):
        scene = {"tests": [{"id": "clock", "duration_s": 1}], "bodies": [{"id": "gear", "source": "a.step"}]}
        baseline = task_definition(ORIGINAL_TASK, scene)
        def check(definition):
            return review_task_change(object(), self.author, self.workspace, self.project, self.design,
                                      self.turn, definition, tier_name="medium")
        self.assertFalse(check(baseline)["reviewed"])
        scene["bodies"][0]["source"] = "fixed.step"
        self.assertFalse(check(task_definition(ORIGINAL_TASK, scene))["reviewed"])
        self.timer.assert_not_called()
        self.assertFalse(self.author.calls)
        scene["tests"][0]["duration_s"] = 0.5
        decision = check(task_definition(ORIGINAL_TASK, scene))
        self.assertTrue(decision["reviewed"])
        self.assertTrue(decision["accepted"])
        self.assertEqual("medium", self.timer.call_args.kwargs["tier_name"])
        self.assertIn('"duration_s": 0.5', self.timer.call_args.args[2])
        self.assertIn('"duration_s": 1', self.timer.call_args.args[2])
        self.assertEqual(self.advice, decision["advice"])

    def test_changed_default_gravity_is_a_simulation_task_change(self):
        first = task_definition(ORIGINAL_TASK, {"tests": []})
        second = task_definition(ORIGINAL_TASK, {"tests": [], "gravity_m_s2": [0, 0, 0]})
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
