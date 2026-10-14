from __future__ import annotations

import unittest

from simulation_ready_cad_workflow.validation_pipeline.runtime.timebox import (
    TIMEBOX_COMPLETED,
    TIMEBOX_NOT_ACCEPTED,
    TIMEBOX_POLICY,
    TIMEBOX_POLICY_ENV,
    derive_timebox_outcome,
    timebox_outcome_allows_continuation,
)


def _derive(**overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "job_environment": {
            TIMEBOX_POLICY_ENV: TIMEBOX_POLICY,
            "SIMREADY_CAPABILITY_EXECUTION_CLASS": "physical",
        },
        "exact_candidate_worker_bound": True,
        "dispatch_context_published": True,
        "full_declared_timebox_allocated": True,
        "deadline_reached_with_process_alive": True,
        "process_tree_stopped_after_cleanup": True,
        "input_files_unchanged_after": True,
        "finalization_errors": [],
        "observed_duration_seconds": 291.0,
        "total_wall_limit_seconds": 300.0,
    }
    arguments.update(overrides)
    return derive_timebox_outcome(**arguments)  # type: ignore[arg-type]


class TimeboxOutcomeTests(unittest.TestCase):
    def test_supervisor_deadline_is_noncertifying_completion(
        self,
    ) -> None:
        outcome = _derive()

        self.assertEqual(outcome["status"], TIMEBOX_COMPLETED)
        self.assertTrue(timebox_outcome_allows_continuation(outcome))
        for field in (
            "raw_job_success",
            "physical_pass",
            "contract_pass",
            "activation_authority",
            "terminal_authority",
            "certificate_authority",
            "semantic_progress_proven",
            "goal_alignment_evaluated",
        ):
            with self.subTest(field=field):
                self.assertFalse(outcome[field])
        self.assertFalse(
            outcome["evidence"]["child_reported_progress_used"]
        )

    def test_forged_outcome_or_child_claim_cannot_continue(
        self,
    ) -> None:
        outcome = _derive()
        for field in (
            "raw_job_success",
            "physical_pass",
            "contract_pass",
            "activation_authority",
            "terminal_authority",
            "certificate_authority",
            "semantic_progress_proven",
            "goal_alignment_evaluated",
        ):
            with self.subTest(field=field):
                forged = dict(outcome)
                forged[field] = True
                self.assertFalse(
                    timebox_outcome_allows_continuation(forged)
                )

        evidence_forgery = {
            "source": "child_trace",
            "child_reported_progress_used": True,
            "observed_duration_seconds": 301.0,
        }
        for field, value in evidence_forgery.items():
            with self.subTest(evidence=field):
                forged = dict(outcome)
                forged["evidence"] = dict(outcome["evidence"])
                forged["evidence"][field] = value
                self.assertFalse(
                    timebox_outcome_allows_continuation(forged)
                )

    def test_early_exit_and_supervisor_anomalies_are_not_accepted(
        self,
    ) -> None:
        cases = {
            "setup_or_early_exit": {
                "deadline_reached_with_process_alive": False,
            },
            "unbound_worker": {
                "exact_candidate_worker_bound": False,
            },
            "dispatch_missing": {
                "dispatch_context_published": False,
            },
            "short_queue_budget": {
                "full_declared_timebox_allocated": False,
            },
            "cleanup_survivor": {
                "process_tree_stopped_after_cleanup": False,
            },
            "input_changed": {
                "input_files_unchanged_after": False,
            },
            "finalization_error": {
                "finalization_errors": ["log drain failed"],
            },
            "total_limit_exceeded": {
                "observed_duration_seconds": 300.001,
            },
            "policy_missing": {
                "job_environment": {},
            },
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                outcome = _derive(**overrides)
                self.assertEqual(
                    outcome["status"],
                    TIMEBOX_NOT_ACCEPTED,
                )
                self.assertFalse(
                    timebox_outcome_allows_continuation(outcome)
                )


if __name__ == "__main__":
    unittest.main()
