"""Recommended run limits: total construction time and per-simulation cap."""
from __future__ import annotations

import unittest
from unittest import mock

from simulation_ready_cad_workflow.validation_pipeline.dashboard import launcher
from simulation_ready_cad_workflow.validation_pipeline.slim_harness import execution_profile
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.budget import recommended_limits
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_budget import BudgetConfig

from .test_dashboard_launch import AUTH, _call, _launch_server, _popen


def _profile(hours, seconds):
    return mock.patch.multiple(execution_profile, DEFAULT_BUDGET_HOURS=hours, DEFAULT_SIMULATION_TIMEOUT_SECONDS=seconds)


class RecommendedLimitsTest(unittest.TestCase):
    def test_the_distribution_profile_sets_high_tier_defaults_only(self) -> None:
        with _profile(None, None):
            self.assertEqual([recommended_limits(name) for name in ("light", "medium", "high")], [(1, 60), (2, 180), (3, 300)])
        with _profile(5, 600):
            self.assertEqual([recommended_limits(name) for name in ("light", "medium", "high")], [(1, 60), (2, 180), (5, 600)])

    def test_longer_budgets_are_accepted(self) -> None:
        self.assertEqual([BudgetConfig(hours=hours).total_seconds for hours in (8, 12)], [28800, 43200])
        with self.assertRaises(ValueError):
            BudgetConfig(hours=4)


class LaunchLimitsTest(unittest.TestCase):
    def test_options_offer_the_recommended_limits(self) -> None:
        with _profile(5, 600), _launch_server() as (_root, server), \
                mock.patch.object(launcher, "_checks", return_value={"ready": False, "items": []}):
            options = _call(server, "GET", "/api/launch/options")[1]
        high = next(tier for tier in options["tiers"] if tier["id"] == "high")
        self.assertEqual((high["hours"], high["simulation_timeout_seconds"]), (5, 600))
        self.assertIn(12, options["budget_hours"])
        self.assertIn(600, options["simulation_timeouts"])

    def test_chosen_limits_reach_the_harness_command(self) -> None:
        body = {"mode": "free", "project": "limits", "task": "Build it.", "provider": "codex", "tier": "high",
                "budget_hours": 8, "simulation_timeout_seconds": 900}
        with _launch_server() as (_root, server), _popen() as calls:
            self.assertEqual(_call(server, "POST", "/api/launch", body, AUTH)[0], 200)
            argv = calls[0].argv
            self.assertEqual(argv[argv.index("--budget-hours") + 1], "8")
            self.assertEqual(argv[argv.index("--simulation-timeout") + 1], "900")
            for field, value in (("budget_hours", 4), ("simulation_timeout_seconds", 450)):
                status, error = _call(server, "POST", "/api/launch", {**body, "project": "other", field: value}, AUTH)
                self.assertEqual((status, error["error"]), (400, "LAUNCH_FIELD_INVALID"))


if __name__ == "__main__":
    unittest.main()
