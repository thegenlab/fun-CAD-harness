"""Exercise public construction routing, durable summaries and process isolation.

No CAD generator, model provider, or physical simulation is launched. The one
real process test creates its own disposable process tree and an unrelated peer.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ..mechanism.io import write_json_atomic
from ..slim_harness.cli import create_project, start, resume_project
from ..slim_harness.construction import budget_for, policy_prompt, revision_choices
from ..slim_harness.construction_budget import BudgetConfig, initialize_budget
from ..slim_harness.construction_supervisor import execute_project, supervise_project
from ..slim_harness.frozen_experiment import bind_scene, load_experiment
from ..slim_harness.workflow import SlimWorkflow


PREFIX = "simulation_ready_cad_workflow.validation_pipeline.slim_harness."


class ConstructionWorkflowTests(unittest.TestCase):
    def test_author_receives_own_project_and_draft_tool_output_roots(self):
        from ..cold_start.worker import CodexCliWorker
        from ..slim_harness.construction import worker_for
        project,_=self.fixture('permissions')
        worker=worker_for(project,CodexCliWorker(model='gpt-6-astra',reasoning_effort='max'))
        with patch('simulation_ready_cad_workflow.validation_pipeline.cold_start.worker.subprocess.Popen',side_effect=OSError('launch sentinel')) as launch:
            result=worker.run_turn(project/'runs/r01/design','fixture',project/'worker_turns/t001_author',1)
        command=launch.call_args.args[0]
        self.assertIn('--skip-git-repo-check',command)
        roots=[command[i+1] for i,arg in enumerate(command[:-1]) if arg=='--add-dir']
        self.assertEqual(roots,[str(project),str(self.workspace/'.runtime_runs/author_tools')])
        self.assertEqual(result['provider_error']['kind'],'launch')

    def test_reviewer_can_start_in_a_workspace_without_git(self):
        from ..cold_start.worker import CodexCliWorker
        self.assertFalse((self.workspace/'.git').exists())
        reviewer=CodexCliWorker(model='gpt-6-astra',reasoning_effort='medium')
        with patch('simulation_ready_cad_workflow.validation_pipeline.cold_start.worker.subprocess.Popen',side_effect=OSError('launch sentinel')) as launch:
            reviewer.run_turn(self.workspace,'fixture',self.workspace/'review',1,readonly=True,fresh=True)
        command=launch.call_args.args[0]
        self.assertIn('--skip-git-repo-check',command)
        self.assertEqual(command[command.index('--sandbox')+1],'read-only')
        self.assertIn('--ephemeral',command)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.task = "The mechanical asset must move, return, and retain its parts."

    def fixture(self, project_id="fixture", *, functions=3, tests=3, construction=True):
        project = create_project(self.workspace, project_id, self.task)
        criteria = [{"id": f"function_{i}", "kind": "function", "statement": f"Core function {i}",
                     "source_quote": "mechanical asset", "verification": "Real motion and geometry"}
                    for i in range(functions)]
        frozen = {"schema_version": "slim_frozen_criteria_v1", "task_file": str(project / "task.md"),
                  "task_text": self.task, "criteria": criteria, "reviewer_summary": "Fixture core requirements"}
        write_json_atomic(project / "criteria.json", frozen)
        state = json.loads((project / "workflow.json").read_text())
        state["request"]["delivery_mode"] = "submit"
        if construction:
            state["request"]["construction"] = True
        state["events"] = [{"kind": "criteria", "result": {"status": "ready", "criteria": criteria,
                                                             "summary": frozen["reviewer_summary"]}}]
        write_json_atomic(project / "workflow.json", state)
        design = project / "runs/r01/design"
        (design / "part.step").write_text("source fixture, never imported as geometry", encoding="utf-8")
        scene = {"schema_version": "slim_scene_v1", "units": "mm",
                 "bodies": [{"id": "moving", "source": "part.step", "fixed": False}], "joints": [],
                 "tests": [{"id": f"test_{i}", "purpose": f"Test core {i}", "duration_s": 1,
                            "assertions": [{"id": f"observed_{i}", "purpose": "Actual movement",
                                            "criterion_ids": [f"function_{i % max(functions, 1)}"],
                                            "expression": "last(signal('time_s')) >= 1"}]}
                           for i in range(tests)]}
        write_json_atomic(design / "scene.json", scene)
        if construction:
            initialize_budget(project, BudgetConfig(hours=2, max_tasks=3, max_tests=3))
        return project, scene

    def experiment(self):
        value = {"schema_version": "slim_frozen_experiment_v1", "task_text": self.task, "units": "mm",
                 "gravity_m_s2": [0, -9.81, 0], "reviewer_summary": "Frozen experiment fixture",
                 "criteria": [{"id": "motion", "kind": "function", "statement": "Must move",
                               "source_quote": "must move", "verification": "Measure travel"}],
                 "tests": [{"id": "motion", "purpose": "Frozen movement", "duration_s": 2,
                            "assertions": [{"id": "time", "criterion_ids": ["motion"], "purpose": "Full duration",
                                            "expression": "last(signal('time_s')) >= 2"}]}]}
        path = self.workspace / "experiment.json"
        write_json_atomic(path, value)
        return path, value

    def test_create_defaults_to_budgeted_supervision_and_persists_worker_settings(self):
        with patch(PREFIX + "construction_supervisor.supervise_project", return_value={"status": "fixture"}) as supervisor, \
             patch(PREFIX + "cli.initialize_project") as unsupervised:
            start(self.workspace, "new_asset", self.task, budget_config=BudgetConfig(hours=2),
                  model="gpt-6-astra", reasoning_effort="max", chrono_python="/fixture/chrono",
                  construction=True)
        project = self.workspace / ".runtime_runs/projects/new_asset"
        self.assertEqual(7200, budget_for(project).snapshot()["total_seconds"])
        self.assertEqual(180, budget_for(project).snapshot()["config"]["simulation_timeout_seconds"])
        self.assertEqual(600, budget_for(project).snapshot()["config"]["audit_timeout_seconds"])
        request = json.loads((project / "workflow.json").read_text())["request"]
        self.assertEqual(("gpt-6-astra", "max", "/fixture/chrono"),
                         (request["model"], request["reasoning_effort"], request["chrono_python"]))
        supervisor.assert_called_once()
        unsupervised.assert_not_called()

    def test_cli_audit_timeout_is_separate_from_native_timeout(self):
        from ..slim_harness.cli import main
        import io
        for additional,expected in [([],600),(['--audit-timeout','180'],180)]:
            with self.subTest(additional=additional), \
                 patch(PREFIX+'cli.start',return_value={'status':'PASS'}) as launch, \
                 patch('sys.stdout',new_callable=io.StringIO):
                main(['--workspace',str(self.workspace),'create','--project','separate_caps',
                      '--brief',self.task,'--simulation-timeout','60',*additional])
            config=launch.call_args.kwargs['budget_config']
            self.assertEqual(config.audit_timeout_seconds,expected)
            self.assertEqual(config.simulation_timeout_seconds,60)

    def test_light_tier_emits_construction_ledger_without_replacing_existing_budget(self):
        with patch(PREFIX + "cli.initialize_project", return_value={"status": "fixture"}) as evaluate, \
             patch(PREFIX + "construction_supervisor.supervise_project") as build:
            start(self.workspace, "light_closeout", self.task, tier="light")
        project = self.workspace / ".runtime_runs/projects/light_closeout"
        self.assertIsNotNone(budget_for(project))
        self.assertTrue((project / "construction_budget.json").is_file())
        self.assertEqual(3600, budget_for(project).snapshot()["total_seconds"])
        self.assertFalse(json.loads((project / "workflow.json").read_text())["request"].get("construction"))
        evaluate.assert_called_once()
        build.assert_not_called()

    def test_light_run_writes_completion_summary_without_forging_pass(self):
        project, _ = self.fixture("light_run", construction=False)
        initialize_budget(project, BudgetConfig(hours=1))
        with patch.object(SlimWorkflow, "_run", return_value={"status": "NEEDS_REVISION", "issues": ["fixture"]}) as run:
            result = SlimWorkflow(self.workspace, project).run()
        self.assertEqual("NEEDS_REVISION", result["status"], result)
        self.assertNotIn("eligible_pass", result)
        self.assertTrue((project / "construction_budget.json").is_file())
        summary = json.loads((project / "completion_summary.json").read_text(encoding="utf-8"))
        self.assertTrue((project / "completion_summary.md").is_file())
        self.assertFalse(summary["complete_instance_pass"])
        self.assertNotEqual("COMPLETED", summary["status"])
        self.assertEqual("NEEDS_REVISION", summary["stop_reason"])
        run.assert_called_once()

    def test_light_interrupt_still_writes_completion_summary(self):
        project, _ = self.fixture("light_interrupt", construction=False)
        initialize_budget(project, BudgetConfig(hours=1))
        with patch.object(SlimWorkflow, "_run", side_effect=RuntimeError("fixture interrupt")):
            with self.assertRaisesRegex(RuntimeError, "fixture interrupt"):
                SlimWorkflow(self.workspace, project).run()
        summary = json.loads((project / "completion_summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["complete_instance_pass"])
        self.assertNotEqual("COMPLETED", summary["status"])
        self.assertEqual("RuntimeError", summary["stop_reason"])

    def test_light_initialize_interrupt_still_writes_completion_summary(self):
        from ..slim_harness.cli import initialize_project
        project, _ = self.fixture("light_init_interrupt", construction=False)
        initialize_budget(project, BudgetConfig(hours=1))
        with patch(PREFIX + "cli._initialize_project", side_effect=RuntimeError("init interrupt")):
            with self.assertRaisesRegex(RuntimeError, "init interrupt"):
                initialize_project(self.workspace, project)
        self.assertTrue((project / "completion_summary.json").is_file())
        self.assertFalse(json.loads((project / "completion_summary.json").read_text(encoding="utf-8"))["complete_instance_pass"])

    def test_submit_keeps_evaluation_behavior_unless_construction_requested(self):
        source_project, _ = self.fixture(construction=False)
        source = source_project / "runs/r01/design/scene.json"
        with patch(PREFIX + "cli.initialize_project", return_value={"status": "fixture"}) as evaluate, \
             patch(PREFIX + "construction_supervisor.supervise_project", return_value={"status": "fixture"}) as build:
            start(self.workspace, "submitted", self.task, scene_path=source)
            start(self.workspace, "construction_submit", self.task, scene_path=source, construction=True)
        self.assertIsNone(budget_for(self.workspace / ".runtime_runs/projects/submitted"))
        self.assertIsNotNone(budget_for(self.workspace / ".runtime_runs/projects/construction_submit"))
        self.assertEqual(1, evaluate.call_count)
        self.assertEqual(1, build.call_count)

    def test_frozen_experiment_preserves_original_tests_and_rejects_construction_permissions(self):
        path, original = self.experiment()
        with patch(PREFIX + "cli.initialize_project", return_value={"status": "fixture"}), \
             patch(PREFIX + "construction_supervisor.supervise_project") as supervisor:
            start(self.workspace, "frozen", self.task, experiment_path=path, budget_config=BudgetConfig(hours=1))
        project = self.workspace / ".runtime_runs/projects/frozen"
        self.assertIsNone(budget_for(project))
        self.assertEqual(original, load_experiment(project))
        supervisor.assert_not_called()
        changed = copy.deepcopy(original["tests"])
        changed[0]["duration_s"] = .1
        with self.assertRaisesRegex(ValueError, "tests cannot be changed"):
            bind_scene({"tests": changed}, load_experiment(project))
        with self.assertRaisesRegex(ValueError, "Frozen experiments"):
            start(self.workspace, "invalid_frozen_build", self.task, experiment_path=path, construction=True)

    def test_repair_strategies_are_construction_only_and_leave_evidence_untouched(self):
        evaluation, _ = self.fixture("evaluation", construction=False)
        construction, _ = self.fixture("construction")
        failure = {"status": "NEEDS_REVISION", "issues": ["Motion did not finish"], "measured_travel_m": .002}
        before = copy.deepcopy(failure)
        self.assertIs(failure, revision_choices(evaluation, failure))
        self.assertEqual("", policy_prompt(evaluation))
        result = revision_choices(construction, failure)
        self.assertEqual("not_uniquely_determined", result["failure_cause"])
        self.assertEqual(3, len(result["available_repair_strategies"]))
        self.assertEqual(before, failure)
        self.assertEqual(.002, result["measured_travel_m"])
        self.assertIn("unchanged core semantics", policy_prompt(construction))

    def test_more_than_three_scene_tests_stop_before_geometry(self):
        project, _ = self.fixture(tests=4)
        with patch(PREFIX + "workflow.prepare_geometry") as geometry:
            with self.assertRaisesRegex((ValueError, RuntimeError), "CONSTRUCTION_TASK_TEST_LIMIT"):
                SlimWorkflow(self.workspace, project).run_revision()
        geometry.assert_not_called()

    def test_more_than_three_core_function_groups_stop_before_geometry(self):
        project, _ = self.fixture(functions=4)
        with patch(PREFIX + "workflow.prepare_geometry") as geometry:
            with self.assertRaisesRegex((ValueError, RuntimeError), "CONSTRUCTION_TASK_TEST_LIMIT"):
                SlimWorkflow(self.workspace, project).run_revision()
        geometry.assert_not_called()

    def test_exhaustion_publishes_partial_assets_and_per_task_summary_without_another_agent(self):
        project, _ = self.fixture()
        budget_for(project)._transaction(lambda state, now: state.update(active_seconds=7200))
        with patch(PREFIX + "cli.resume_project") as resume:
            result = execute_project(self.workspace, project)
        resume.assert_not_called()
        self.assertEqual("BUDGET_EXHAUSTED", result["status"])
        summary = json.loads((project / "completion_summary.json").read_text())
        self.assertEqual("BUDGET_EXHAUSTED_PARTIAL", summary["status"])
        self.assertEqual(3, summary["task_count"])
        self.assertFalse(summary["complete_instance_pass"])
        self.assertEqual(0, summary["verified_completed_tasks"])
        self.assertIn("runs/r01/design/part.step", summary["assets"])
        for task in summary["tasks"]:
            self.assertFalse(task["verified_complete"])
            self.assertEqual([], task["executed_tests"])
            self.assertTrue(task["reason"])
            self.assertTrue(task["recommendations"])
        self.assertTrue((project / "completion_summary.md").is_file())

    def test_resume_retains_requested_model_effort_and_physics_interpreter(self):
        project, _ = self.fixture(construction=False)
        state = json.loads((project / "workflow.json").read_text())
        state["request"].update(model="gpt-6-astra", reasoning_effort="max", chrono_python="/fixture/chrono")
        write_json_atomic(project / "workflow.json", state)
        with patch.object(SlimWorkflow, "run", autospec=True, return_value={"status": "fixture"}) as run:
            resume_project(self.workspace, project, in_process=True)
        workflow = run.call_args.args[0]
        self.assertEqual("gpt-6-astra", workflow.author.model)
        self.assertEqual("max", workflow.author.reasoning_effort)
        self.assertEqual("/fixture/chrono", workflow.chrono_python)

    def test_historical_workflow_cli_is_supervised_for_construction(self):
        from ..slim_harness.workflow import main
        project,_=self.fixture()
        with patch(PREFIX+'construction_supervisor.supervise_project',return_value={'status':'fixture'}) as supervised, \
             patch.object(SlimWorkflow,'run') as direct:
            main([str(project),'--workspace',str(self.workspace),'--one-revision'])
        supervised.assert_called_once_with(self.workspace,project,executable='codex',one_revision=True)
        direct.assert_not_called()

    def test_historical_workflow_cli_keeps_evaluation_single_revision(self):
        from ..slim_harness.workflow import main
        project,_=self.fixture(construction=False)
        with patch(PREFIX+'construction_supervisor.supervise_project') as supervised, \
             patch.object(SlimWorkflow,'run',return_value={'status':'fixture'}) as direct:
            main([str(project),'--workspace',str(self.workspace),'--one-revision'])
        direct.assert_called_once_with(one_revision=True)
        supervised.assert_not_called()

    def test_supervised_single_revision_does_not_start_repair_round(self):
        project,_=self.fixture()
        with patch.object(SlimWorkflow,'run',return_value={'status':'NEEDS_REVISION','issues':['fixture motion failed']}) as run, \
             patch(PREFIX+'cli.resume_project') as resume:
            result=execute_project(self.workspace,project,one_revision=True)
        run.assert_called_once_with(one_revision=True)
        resume.assert_not_called()
        self.assertEqual(result['status'],'NEEDS_REVISION')
        self.assertTrue((project/'completion_summary.json').is_file())

    def test_construction_author_uses_remaining_project_budget_not_thirty_minute_restart(self):
        from ..slim_harness.construction import author_turn_timeout
        from ..slim_harness.construction_budget import BudgetExhausted
        project,_=self.fixture()
        self.assertEqual(author_turn_timeout(project),7080-3*180)
        budget_for(project)._transaction(lambda state,now:state.update(active_seconds=6000))
        self.assertEqual(author_turn_timeout(project),1080-3*180)
        budget_for(project)._transaction(lambda state,now:state.update(active_seconds=7080))
        with self.assertRaises(BudgetExhausted):author_turn_timeout(project)
        evaluation,_=self.fixture('evaluation',construction=False)
        self.assertEqual(author_turn_timeout(evaluation),1800)

    @unittest.skipIf(os.name == "nt", "Disposable process-tree regression uses POSIX process sessions")
    def test_supervisor_deadline_stops_its_tree_and_preserves_unrelated_process(self):
        import psutil

        project, _ = self.fixture()
        budget = budget_for(project)
        snapshot = budget.snapshot()
        budget._transaction(lambda state, now: state.update(
            active_seconds=snapshot["total_seconds"] - snapshot["config"]["summary_reserve_seconds"] - .9))
        child_pid_file = project / "fixture-child.pid"
        repository = Path(__file__).resolve().parents[3]
        payload = (
            "import sys,time,subprocess,pathlib; "
            f"sys.path.insert(0,{str(repository)!r}); "
            "from simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_budget import ProjectBudget; "
            "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            f"pathlib.Path({str(child_pid_file)!r}).write_text(str(child.pid)); "
            f"activity=ProjectBudget(pathlib.Path({str(project)!r})).activity('workflow'); "
            "activity.__enter__(); time.sleep(30)"
        )
        native_popen = subprocess.Popen
        unrelated = native_popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
        spawned = []

        def launch_fixture(_command, **kwargs):
            process = native_popen([sys.executable, "-c", payload], **kwargs)
            spawned.append(process)
            return process

        try:
            with patch(PREFIX + "construction_supervisor.subprocess.Popen", side_effect=launch_fixture):
                result = supervise_project(self.workspace, project)
            self.assertEqual(1, len(spawned))
            self.assertIsNotNone(spawned[0].poll())
            self.assertIsNone(unrelated.poll())
            self.assertTrue(child_pid_file.exists())
            child_pid = int(child_pid_file.read_text())
            if psutil.pid_exists(child_pid):
                self.assertEqual(psutil.STATUS_ZOMBIE, psutil.Process(child_pid).status())
            self.assertEqual("BUDGET_EXHAUSTED_PARTIAL", result["completion"]["status"])
            record = json.loads((project / "construction_supervisor.json").read_text())
            self.assertEqual("budget_exhausted", record["status"])
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)
            for process in spawned:
                if process.poll() is None:
                    from ..runtime.system_probe import terminate_process_tree
                    terminate_process_tree(process.pid, 1)
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
