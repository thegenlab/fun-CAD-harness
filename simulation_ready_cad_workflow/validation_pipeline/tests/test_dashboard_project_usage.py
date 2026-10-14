from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from simulation_ready_cad_workflow.validation_pipeline.dashboard.model import (
    ReadContext,
    normalize_run,
)
from simulation_ready_cad_workflow.validation_pipeline.dashboard.project_usage import (
    load_project_usage,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _usage_record(project_id: str = "fixture") -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "project_id": project_id,
        "status": "CURRENT",
        "completed_session_count": 2,
        "running_session_count": 1,
        "total_wall_time_seconds": 30.5,
        "total_tokens": 300,
        "run_ids": ["run-1", "run-2", "run-3"],
        "completed_sessions": [
            {
                "session_id": "session-1",
                "run_id": "run-1",
                "wall_time_seconds": 10.0,
                "total_tokens": 100,
                "completed_at": "2026-07-24T01:00:00+00:00",
                "token_counter_source": "codex_goal_counter",
            },
            {
                "session_id": "session-2",
                "run_id": "run-2",
                "wall_time_seconds": 20.5,
                "total_tokens": 200,
                "completed_at": "2026-07-24T02:00:00+00:00",
                "token_counter_source": "codex_goal_counter",
            },
        ],
        "running_sessions": [
            {
                "session_id": "session-3",
                "run_id": "run-3",
                "started_at": "2026-07-24T03:00:00+00:00",
                "token_counter_source": "codex_goal_counter",
            }
        ],
        "summary_artifact": ".runtime_runs/project_usage/projects/fixture.json",
    }


class DashboardProjectUsageTests(unittest.TestCase):
    def _context(self, root: Path) -> ReadContext:
        return ReadContext(root.resolve(), [])

    def test_loads_completed_totals_and_keeps_running_session_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_json(
                root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json",
                _usage_record(),
            )

            usage = load_project_usage(self._context(root), "fixture")

            self.assertEqual(usage["status"], "recorded")
            self.assertEqual(usage["total_wall_time_seconds"], 30.5)
            self.assertEqual(usage["total_tokens"], 300)
            self.assertEqual(usage["completed_session_count"], 2)
            self.assertEqual(usage["running_session_count"], 1)
            self.assertEqual(usage["completed_sessions"][0]["run_id"], "run-2")
            self.assertEqual(usage["running_sessions"][0]["run_id"], "run-3")
            self.assertEqual(usage["current_revision"], "")
            self.assertEqual(usage["benchmark_total_tokens"], 300)
            self.assertEqual(
                usage["benchmark_token_counter_source"],
                "codex_goal_counter",
            )

    def test_current_canonical_revision_is_separate_from_cost_record_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_json(
                root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json",
                _usage_record(),
            )

            usage = load_project_usage(self._context(root), "fixture", "r56")

            self.assertEqual(usage["current_revision"], "r56")
            self.assertEqual(usage["completed_session_count"], 2)

    def test_revision_inventory_exposes_missing_cost_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _usage_record()
            record["completed_session_count"] = 1
            record["running_session_count"] = 0
            record["total_wall_time_seconds"] = 10.0
            record["total_tokens"] = 100
            record["completed_sessions"] = [record["completed_sessions"][0]]
            record["running_sessions"] = []
            record["completed_sessions"][0]["run_id"] = "fixture-r02"
            for revision in ("r01", "r02"):
                run = (
                    root
                    / ".runtime_runs"
                    / "projects"
                    / "fixture"
                    / "runs"
                    / revision
                )
                _write_json(run / "run_status.json", {"status": "FAILED"})
            _write_json(
                root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json",
                record,
            )

            usage = load_project_usage(self._context(root), "fixture", "r02")

            self.assertEqual(usage["revision_count"], 2)
            self.assertEqual(usage["recorded_revision_count"], 1)
            self.assertEqual(usage["missing_revision_count"], 1)
            self.assertEqual(usage["revision_records"][0]["revision"], "r02")
            self.assertEqual(usage["revision_records"][0]["usage_status"], "recorded")
            self.assertEqual(usage["revision_records"][1]["usage_status"], "missing")

    def test_missing_record_is_not_reported_as_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            usage = load_project_usage(self._context(root), "fixture")
            self.assertEqual(usage["status"], "unrecorded")
            self.assertIsNone(usage["total_tokens"])
            self.assertIsNone(usage["total_wall_time_seconds"])
            self.assertIsNone(usage["completed_session_count"])

    def test_project_id_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_json(
                root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json",
                _usage_record("other-project"),
            )
            ctx = self._context(root)
            usage = load_project_usage(ctx, "fixture")
            self.assertEqual(usage["status"], "unavailable")
            self.assertFalse(usage["recorded"])
            self.assertTrue(any("project_id" in issue for issue in usage["issues"]))
            self.assertTrue(ctx.diagnostics)

    def test_token_summary_detail_mismatch_does_not_block_dashboard(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _usage_record()
            record["total_tokens"] = 301
            _write_json(
                root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json",
                record,
            )
            usage = load_project_usage(self._context(root), "fixture")
            self.assertEqual(usage["status"], "recorded")
            self.assertEqual(usage["benchmark_total_tokens"], 300)
            self.assertFalse(usage["issues"])

    def test_malformed_json_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json"
            path.parent.mkdir(parents=True)
            path.write_text("{not-json", encoding="utf-8")
            ctx = self._context(root)
            usage = load_project_usage(ctx, "fixture")
            self.assertEqual(usage["status"], "unavailable")
            self.assertTrue(any(item["role"] == "project usage" for item in ctx.diagnostics))

    def test_normalize_run_binds_usage_to_package_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "package"
            _write_json(package / "package.json", {"package_id": "fixture", "design_version": "1"})
            (package / "assembly.step").write_text(
                "ISO-10303-21;END-ISO-10303-21;", encoding="ascii"
            )
            _write_json(package / "assembly_contract.json", {"schema_version": "0.1"})
            intake = root / "intake.json"
            _write_json(
                intake,
                {
                    "status": "ACCEPTED",
                    "summary": {"error_count": 0, "warning_count": 0},
                    "capabilities": {},
                    "issues": [],
                },
            )
            run = root / "runs" / "fixture-run"
            _write_json(
                run / "plans" / "fixture.json",
                {
                    "capsule_id": "fixture_joint",
                    "family": "revolute",
                    "package_dir": str(package),
                    "intake_report": str(intake),
                },
            )
            _write_json(run / "run_status.json", {"status": "NOT_CERTIFIED"})
            record = _usage_record()
            record["running_session_count"] = 0
            record["running_sessions"] = []
            _write_json(
                root / ".runtime_runs" / "project_usage" / "projects" / "fixture.json",
                record,
            )

            data = normalize_run(run, workspace_root=root)

            self.assertEqual(data["overview"]["package"]["id"], "fixture")
            self.assertEqual(data["project_usage"]["project_id"], "fixture")
            self.assertEqual(data["project_usage"]["total_tokens"], 300)

    def test_canonical_react_run_uses_project_id_for_usage_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / ".runtime_runs" / "projects" / "machine"
            _write_json(
                project / "project.json",
                {"project_id": "machine", "package_id": "machine"},
            )
            run = project / "runs" / "r01"
            _write_json(run / "run_status.json", {"status": "DESIGN_REQUIRED"})
            record = _usage_record("machine")
            record["running_session_count"] = 0
            record["running_sessions"] = []
            _write_json(
                root
                / ".runtime_runs"
                / "project_usage"
                / "projects"
                / "machine.json",
                record,
            )

            data = normalize_run(run, workspace_root=root)

            self.assertEqual(data["overview"]["package"]["id"], "machine")
            self.assertEqual(data["project_usage"]["project_id"], "machine")
            self.assertEqual(data["project_usage"]["total_tokens"], 300)

    def test_unique_run_id_binds_a_recorder_project_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = _usage_record("short-project-id")
            _write_json(
                root
                / ".runtime_runs"
                / "project_usage"
                / "projects"
                / "short-project-id.json",
                record,
            )

            usage = load_project_usage(
                self._context(root), "long_engineering_package_id", "run-2"
            )

            self.assertEqual(usage["status"], "recorded")
            self.assertEqual(usage["binding"], "run_id")
            self.assertEqual(usage["package_id"], "long_engineering_package_id")
            self.assertEqual(usage["project_id"], "short-project-id")
            self.assertEqual(usage["total_tokens"], 300)

    def test_frontend_contains_project_usage_summary_and_details(self) -> None:
        pipeline = Path(__file__).resolve().parents[1]
        app = (pipeline / "dashboard" / "static" / "app.js").read_text(encoding="utf-8")
        css = (pipeline / "dashboard" / "static" / "app.css").read_text(encoding="utf-8")
        self.assertIn("整个机械项目累计", app)
        self.assertIn("每条记录绑定具体 rxx", app)
        self.assertIn("Token 只汇总同一计数来源", app)
        self.assertIn("尚未记录", app)
        self.assertIn(".project-usage-popover", css)
        self.assertIn(".usage-session-scroll", css)


if __name__ == "__main__":
    unittest.main()
