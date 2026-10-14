"""Focused resource/time recovery checks; fixtures do not certify mechanics."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

import psutil

from . import test_budget_entrypoints as fixtures
from . import test_physics_resource_entrypoints as physics_fixtures
from ..cold_start.worker import CodexCliWorker
from ..mechanism.io import write_json_atomic
from ..runtime.bounded_subprocess import BoundedSubprocessError, BoundedSubprocessResult
from ..slim_harness.author_tools import run_tool
from ..slim_harness.budget import ProjectBudget, NestedPhysicsBudget, _observed_activity_totals
from ..slim_harness.physics_invocation import run_physics_attempts
from ..slim_harness.resource_wait import wait_for_observed_resources
from ..slim_harness.trace_storage import TraceWriter, open_trace_prefix, release_trace_cache
from ..slim_harness.workflow import SlimWorkflow


PREFIX = fixtures.PREFIX
WORKER = "simulation_ready_cad_workflow.validation_pipeline.cold_start.worker."
REPO = Path(__file__).resolve().parents[3]
PYTHON = r"C:\Tools\miniconda3\python.exe"


class BudgetResourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        directory = os.environ.get("RESOURCE_RECOVERY_EVIDENCE_DIR")
        if not directory:
            fixtures.BudgetEntrypointTests.setUp(self)
            return
        # Report runs keep inspectable attempts and observations; ordinary unit
        # runs retain the existing temporary-fixture lifecycle.
        self.workspace = Path(directory) / "workspaces" / self._testMethodName
        self.workspace.mkdir(parents=True)
        self.project = fixtures.cli.create_project(self.workspace, "budget_fixture", "A slider must move.")
        self.tier = fixtures.TierConfig("fixture", 5, 0.3, 2, 0, 10)

    initialize = fixtures.BudgetEntrypointTests.initialize
    seed_scene = fixtures.BudgetEntrypointTests.seed_scene
    checkpoint = fixtures.BudgetEntrypointTests.checkpoint
    geometry = physics_fixtures.PhysicsResourceEntrypointTests.geometry

    def test_observation_transient_permission_error_retries_then_reads_closed_interval(self):
        directory=self.workspace/"observations";directory.mkdir()
        path=directory/"activity.json"
        value={"activity":"resource_admission","started_monotonic":10.,"ended_monotonic":12.}
        write_json_atomic(path,value)
        encoded=path.read_text(encoding="utf-8")
        with patch.object(Path,"read_text",side_effect=[PermissionError("temporary sharing conflict"),PermissionError("temporary sharing conflict"),encoded]) as reads,patch(PREFIX+"budget.time.sleep") as sleep:
            waits,work,issues=_observed_activity_totals(directory,0.,20.)
        self.assertEqual((2.,0.,[]),(waits,work,issues))
        self.assertEqual(3,reads.call_count)
        self.assertEqual([.015,.015],[call.args[0] for call in sleep.call_args_list])
        self.assertEqual(value,json.loads(path.read_text(encoding="utf-8")))

    def test_observation_persistent_permission_error_stays_an_error_after_bounded_retry(self):
        directory=self.workspace/"observations";directory.mkdir()
        path=directory/"activity.json";write_json_atomic(path,{"activity":"resource_admission"})
        with patch.object(Path,"read_text",side_effect=PermissionError("persistent access denied")) as reads,patch(PREFIX+"budget.time.sleep") as sleep:
            waits,work,issues=_observed_activity_totals(directory,0.,20.)
        self.assertEqual((0.,0.),(waits,work))
        self.assertEqual(1,len(issues));self.assertIn("persistent access denied",issues[0])
        self.assertEqual(3,reads.call_count)
        self.assertAlmostEqual(.03,sum(call.args[0] for call in sleep.call_args_list))
        # Parsing errors are not sharing violations and must not be retried.
        with patch.object(Path,"read_text",return_value="{bad json") as reads,patch(PREFIX+"budget.time.sleep") as sleep:
            _,_,issues=_observed_activity_totals(directory,0.,20.)
        self.assertTrue(issues);self.assertEqual(1,reads.call_count);sleep.assert_not_called()

    def evidence(self, **value):
        directory = os.environ.get("RESOURCE_RECOVERY_EVIDENCE_DIR")
        if directory:
            write_json_atomic(Path(directory) / (self._testMethodName + ".json"), value)

    def policy(self, **overrides):
        value = {"required_available_memory_gib": 0.01,
                 "max_resource_wait_seconds": 0.06, "resource_poll_seconds": 0.01,
                 "max_process_tree_memory_gib": 5}
        value.update(overrides)
        write_json_atomic(self.project / "resource_policy.json", value)

    def draft_mocks(self, scene):
        stack = ExitStack()
        stack.enter_context(patch(PREFIX + "author_tools.prepare_geometry", return_value=self.geometry()))
        stack.enter_context(patch(PREFIX + "author_tools.compile_scene", return_value=scene))
        stack.enter_context(patch(PREFIX + "author_tools.audit_motion", return_value={"passed": True}))
        def audit(prepared, samples, path):
            result = {"passed": True, "violations": [], "scope": "synthetic resource plumbing fixture"}
            write_json_atomic(path, result)
            return result
        stack.enter_context(patch(PREFIX + "author_tools.audit_saved_motion", side_effect=audit))
        return stack

    def owner(self):
        write_json_atomic(self.project / "workflow_runtime.json", {
            "status": "running", "pid": os.getpid(),
            "process_started_at_epoch": psutil.Process().create_time()})

    def prefix(self, command):
        request = json.loads(Path(command[-1]).read_text())
        writer = TraceWriter(Path(request["output"]), 10, chunk_rows=1)
        writer.append({"time_s": 0}, {"time_s": 0})
        writer.handles.close()

    def complete(self, command):
        request = json.loads(Path(command[-1]).read_text())
        poses = {body["id"]: {"position_abs_m": [0, 0, 0], "rotation_abs_quat_wxyz": [1, 0, 0, 0]}
                 for body in request["scene"].get("bodies", [])}
        write_json_atomic(Path(request["output"]), {
            "test_id": "motion", "physical_pass": True,
            "samples": [{"time_s": 0, "body_poses": poses}], "signals": {"time_s": [0]}})
        return BoundedSubprocessResult(0, b"synthetic solver fixture", b"")

    def test_invalid_resource_settings_stop_before_launch_and_tiny_positive_is_valid(self):
        probe = Mock()
        invalid = [0, -1, None, True, float("nan"), float("inf")]
        for threshold in invalid:
            with self.subTest(threshold=threshold):
                result = wait_for_observed_resources(probe, threshold, max_wait_seconds=0.1)
                self.assertEqual("RESOURCE_POLICY_REQUIRED", result["status"])
        for durations in ({"max_wait_seconds": float("nan")},
                          {"max_wait_seconds": float("inf")},
                          {"max_wait_seconds": -1},
                          {"max_wait_seconds": 0.1, "poll_seconds": 0}):
            with self.subTest(durations=durations):
                result = wait_for_observed_resources(probe, 0.01, **durations)
                self.assertEqual("RESOURCE_POLICY_REQUIRED", result["status"])
        probe.sample.assert_not_called()
        probe.sample.return_value = {"available_memory_gib": 0.02}
        self.assertEqual("ready", wait_for_observed_resources(
            probe, 0.01, max_wait_seconds=0)["status"])
        runner = Mock()
        with ProjectBudget(self.project, self.tier) as budget:
            settings = [{"required_available_memory_gib": v} for v in [0, -1, None, True, "nan", "inf"]]
            settings += [{"resource_admission": {}}, {"resource_admission": "bad"},
                         {"max_rss_bytes": 0.5}, {"max_rss_bytes": "inf"},
                         {"required_available_memory_gib": 0.01, "max_resource_wait_seconds": "nan"}]
            for value in settings:
                with self.subTest(settings=value):
                    write_json_atomic(self.project / "resource_policy.json", value)
                    result = run_physics_attempts(workspace=self.workspace, project=self.project,
                        folder=self.workspace / "invalid", scene={}, test={"id": "motion"},
                        chrono_python=PYTHON, budget=budget, timeout_seconds=0.3, runner=runner)
                    self.assertEqual("RESOURCE_POLICY_REQUIRED", result["status"], result)
                    self.assertTrue(result["needs_external_action"])
        runner.assert_not_called()
        self.evidence(invalid_declarations_rejected=len(settings), tiny_positive_gib=0.01,
                      solver_launches=runner.call_count)

    def test_actual_standalone_draft_wait_is_excluded_from_active(self):
        scene = self.seed_scene()
        self.policy()
        before = self.checkpoint()
        probe = Mock()
        probe.sample.return_value = {"available_memory_gib": 0.0}
        with self.draft_mocks(scene), patch(PREFIX + "physics_invocation.SystemProbe", return_value=probe), \
                patch(PREFIX + "author_tools.run_bounded_subprocess") as runner:
            result = run_tool("simulate", self.project / "runs/r01/design/scene.json", workspace=self.workspace)
        after = self.checkpoint()
        self.assertEqual("RESOURCE_POLICY_REQUIRED", result["status"], result)
        runner.assert_not_called()
        self.assertLess(after["active_seconds"] - before["active_seconds"], 0.01)
        self.assertEqual(before["observed_work_seconds"], after["observed_work_seconds"])
        self.assertGreater(after["observed_wait_seconds"] - before["observed_wait_seconds"], 0.05)
        self.assertGreaterEqual(probe.sample.call_count, 2)
        self.evidence(result=result, active_delta=after["active_seconds"] - before["active_seconds"],
                      observed_wait_delta=after["observed_wait_seconds"] - before["observed_wait_seconds"],
                      observed_work_delta=after["observed_work_seconds"] - before["observed_work_seconds"])

    def test_nested_retries_use_current_budget_between_parent_checkpoints(self):
        self.policy()
        seen = []
        probe = Mock()
        probe.sample.return_value = {"available_memory_gib": 1.0}
        with ProjectBudget(self.project, self.tier) as budget:
            budget.tracker._active_seconds = 4.5
            budget.start_worker_observer(self.workspace / "turn")
            budget.worker_observer_pid = os.getpid()
            def runner(command, **limits):
                budget.refresh_worker_observations()
                seen.append({"timeout": limits["timeout_seconds"],
                             "remaining_at_launch": budget.tracker.remaining_active,
                             "rss": limits["max_rss_bytes"], "request": command[-1]})
                if len(seen) == 1:
                    time.sleep(0.26)
                    raise BoundedSubprocessError("BOUNDED_SUBPROCESS_RSS", "synthetic RSS")
                return self.complete(command)
            with budget.active("worker_elapsed_upper_estimate"), \
                    patch(PREFIX + "physics_invocation.SystemProbe", return_value=probe):
                result = run_physics_attempts(workspace=self.workspace, project=self.project,
                    folder=self.workspace / "nested-retry", scene={}, test={"id": "motion"},
                    chrono_python=PYTHON, budget=NestedPhysicsBudget(self.project),
                    timeout_seconds=0.3, runner=runner)
        self.assertEqual("ready", result["status"], result)
        self.assertEqual(2, len(seen))
        self.assertLess(seen[1]["timeout"], 0.28)
        self.assertLessEqual(seen[1]["timeout"], seen[1]["remaining_at_launch"] + 0.01)
        self.assertEqual(5 * 1024**3, seen[1]["rss"])
        self.assertEqual("retry-1", Path(result["trace_file"]).parent.name)
        self.assertTrue((Path(seen[0]["request"]).parent / "execution-failure.json").is_file())
        self.evidence(seen_launches=seen, result=result)

    def test_rss_budget_exhaustion_retains_the_failed_prefix(self):
        self.policy()
        probe = Mock()
        probe.sample.return_value = {"available_memory_gib": 1.0}
        launches = []
        with ProjectBudget(self.project, self.tier) as budget:
            budget.tracker._active_seconds = 4.96
            def runner(command, **limits):
                launches.append(command[-1])
                self.prefix(command)
                time.sleep(0.05)
                raise BoundedSubprocessError("BOUNDED_SUBPROCESS_RSS", "synthetic RSS")
            with patch(PREFIX + "physics_invocation.SystemProbe", return_value=probe):
                result = run_physics_attempts(workspace=self.workspace, project=self.project,
                    folder=self.workspace / "exhaustion", scene={}, test={"id": "motion"},
                    chrono_python=PYTHON, budget=budget, timeout_seconds=0.3, runner=runner)
        self.assertEqual("ACTIVE_BUDGET_EXHAUSTED", result["status"], result)
        self.assertEqual(1, len(launches))
        prefix = result["readable_prefixes"][0]
        trace = open_trace_prefix(Path(prefix["path"]))
        try:
            self.assertEqual(1, len(trace["samples"]))
        finally:
            release_trace_cache(trace)
        self.evidence(result=result, solver_launches=len(launches))

    def test_draft_rss_success_binds_result_and_cache_input_to_successful_folder(self):
        scene = self.seed_scene()
        self.policy()
        calls = []
        probe = Mock()
        probe.sample.return_value = {"available_memory_gib": 1.0}
        def runner(command, **limits):
            calls.append(command[-1])
            if len(calls) == 1:
                self.prefix(command)
                raise BoundedSubprocessError("BOUNDED_SUBPROCESS_RSS", "synthetic RSS")
            return self.complete(command)
        with self.draft_mocks(scene), patch(PREFIX + "physics_invocation.SystemProbe", return_value=probe), \
                patch(PREFIX + "author_tools.run_bounded_subprocess", side_effect=runner):
            result = run_tool("simulate", self.project / "runs/r01/design/scene.json", workspace=self.workspace)
        self.assertEqual("DEVELOPMENT_PHYSICS_PASSED", result["status"], result)
        trace_path = Path(result["tests"][0]["trace_file"])
        self.assertEqual(Path(calls[1]).parent, trace_path.parent)
        self.assertEqual("retry-1", trace_path.parent.name)
        self.assertFalse((trace_path.parent / "geometry-audit.json").exists())
        self.assertEqual("deferred_to_formal_validation",result["tests"][0]["geometry_audit"]["status"])
        self.assertTrue((Path(calls[0]).parent / "execution-failure.json").is_file())
        self.assertTrue(list(Path(calls[0]).parent.rglob("prefix.json")))
        self.evidence(result=result, requests=calls)

    def launch_fixture(self, worker, budget, code, *, timeout=0.55):
        helper = self.workspace / "observed_child.py"
        helper.write_text(code, encoding="utf-8")
        popen = subprocess.Popen
        commands = []
        def launch(command, **kwargs):
            response = command[command.index("-o") + 1]
            child_command = [PYTHON, "-B", str(helper), str(self.project), response]
            commands.append(child_command)
            kwargs["cwd"] = str(REPO)
            kwargs["env"] = {**os.environ, "PYTHONPATH": str(REPO)}
            return popen(child_command, **kwargs)
        with patch(WORKER + "subprocess.Popen", side_effect=launch), \
                patch(PREFIX + "budget.should_retry", return_value=False):
            result = budget.worker(worker).run_turn(self.workspace, "fixture", self.workspace / "turn", timeout)
        files = {}
        for path in (self.workspace / "turn").rglob("*"):
            if path.is_file() and path.suffix in {".json", ".log", ".jsonl"}:
                files[str(path.relative_to(self.workspace / "turn"))] = path.read_text(encoding="utf-8")
        self.evidence(launch_result=result, commands=commands, files=files)
        return result, commands

    def test_formal_retry_binds_validated_results_and_audit_to_successful_folder(self):
        scene = self.seed_scene()
        self.policy()
        seen = []
        probe = Mock()
        probe.sample.return_value = {"available_memory_gib": 1.0}
        def runner(command, **limits):
            seen.append({"timeout": limits["timeout_seconds"], "request": command[-1],
                         "remaining_at_launch": workflow.budget.tracker.remaining_active})
            if len(seen) == 1:
                time.sleep(0.26)
                raise BoundedSubprocessError("BOUNDED_SUBPROCESS_RSS", "synthetic RSS")
            return self.complete(command)
        def audit(prepared, samples, path):
            result = {"passed": True, "violations": [], "scope": "synthetic resource plumbing fixture"}
            write_json_atomic(path, result)
            return result
        with ExitStack() as stack:
            for name, value in {"prepare_geometry": self.geometry(), "compile_scene": scene,
                                "audit_motion": {"passed": True, "violations": []},
                                "render_views": self.workspace / "fixture.png"}.items():
                stack.enter_context(patch(PREFIX + "workflow." + name, return_value=value))
            stack.enter_context(patch(PREFIX + "workflow.audit_saved_motion", side_effect=audit))
            stack.enter_context(patch(PREFIX + "physical_parts.resolve_physical_part_inventory",
                                     return_value={"status": "ready", "count": 1, "machine_part_count": 1}))
            stack.enter_context(patch.object(SlimWorkflow, "_preview"))
            stack.enter_context(patch.object(SlimWorkflow, "_motion", return_value=({}, {})))
            finalizer=stack.enter_context(patch.object(SlimWorkflow, "_finish_validation", return_value={
                "status": "NEEDS_REVISION", "issues": ["Synthetic fixture; no mechanical acceptance"]}))
            stack.enter_context(patch(PREFIX + "physics_invocation.SystemProbe", return_value=probe))
            stack.enter_context(patch(PREFIX + "workflow.run_bounded_subprocess", side_effect=runner))
            workflow = SlimWorkflow(self.workspace, self.project)
            workflow.budget_tracker._active_seconds = 4.5
            result = workflow.run(one_revision=True)
        self.assertEqual("NEEDS_REVISION", result["status"], result)
        finalizer.assert_called_once()
        self.assertEqual(2, len(seen))
        self.assertLess(seen[1]["timeout"], seen[0]["timeout"])
        self.assertLessEqual(seen[1]["timeout"], seen[1]["remaining_at_launch"] + 0.01)
        attempt = self.project / "runs/r01/attempts/a001"
        bundle = json.loads((attempt / "validated-physics.json").read_text())
        successful = Path(seen[1]["request"]).parent
        self.assertEqual(successful / "physical.trace.json", Path(bundle["tests"][0]["trace_file"]))
        self.assertIn(str(successful / "geometry-audit.json"), bundle["evidence_bindings"])
        self.assertTrue((Path(seen[0]["request"]).parent / "execution-failure.json").is_file())
        self.evidence(result=result, seen_launches=seen, validated_bundle=bundle)

    def test_real_nested_wait_pauses_parent_budget_and_deadline_and_keeps_unknown_time(self):
        self.policy(required_available_memory_gib=1e9, max_resource_wait_seconds=2.4)
        self.owner()
        code = '''import json,sys,time
from pathlib import Path
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.budget import physics_budget_session
from simulation_ready_cad_workflow.validation_pipeline.slim_harness.physics_invocation import run_physics_attempts
project,response=map(Path,sys.argv[1:]);sys.stdin.read();time.sleep(.06)
with physics_budget_session(project) as budget:
    result=run_physics_attempts(workspace=Path.cwd(),project=project,folder=response.parent/'physics',
        scene={},test={'id':'motion'},chrono_python=sys.executable,budget=budget,timeout_seconds=.3)
(response.parent/'nested-result.json').write_text(json.dumps(result))
response.write_text(json.dumps({'status':'ready','summary':'fixture','issues':[],'evidence_paths':[]}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':1}}))
'''
        with ProjectBudget(self.project, self.tier) as budget:
            # Cold child imports measured 0.93–1.06s on this host. Keep the wait
            # longer than the entire 2s active allowance, so failing to exclude
            # it still necessarily times out; retain 0.6s first-pass reserve.
            budget.tracker._active_seconds = 2.2
            before = budget.tracker.elapsed_active
            result, commands = self.launch_fixture(CodexCliWorker(), budget, code, timeout=2.0)
        checkpoint = self.checkpoint()
        self.assertEqual("ready", result["status"], result)
        nested = json.loads((self.workspace / "turn/nested-result.json").read_text())
        segments = json.loads((self.workspace / "turn/timing.segments.json").read_text())
        self.assertEqual("ready", result["status"], result)
        self.assertEqual("RESOURCE_POLICY_REQUIRED", nested["status"], nested)
        self.assertGreater(result["elapsed_seconds"], 2.4)
        self.assertLess(checkpoint["active_seconds"] - before, 2.0)
        self.assertGreater(checkpoint["observed_wait_seconds"], 2.35)
        self.assertGreater(checkpoint["unclassified_provider_seconds"], 0.05)
        self.assertEqual(0, checkpoint["observed_work_seconds"])
        self.assertAlmostEqual(checkpoint["wall_seconds"],
            checkpoint["active_seconds"] + checkpoint["total_pause_seconds"] - before, delta=0.05)
        self.assertTrue(segments["overlap"]["unclassified_is_not_net_thinking"])
        self.evidence(result=result, nested_result=nested, checkpoint=checkpoint,
                      active_delta=checkpoint["active_seconds"] - before, segments=segments, commands=commands)

    def test_quiet_child_is_active_and_timeout_does_not_kill_unrelated_process(self):
        unrelated = subprocess.Popen([PYTHON, "-c", "import time; time.sleep(20)"],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        def cleanup():
            if unrelated.poll() is None:
                unrelated.kill()
            unrelated.wait(timeout=2)
        self.addCleanup(cleanup)
        with ProjectBudget(self.project, self.tier) as budget:
            result, commands = self.launch_fixture(CodexCliWorker(), budget,
                "import sys,time\nsys.stdin.read();time.sleep(20)\n", timeout=0.15)
        checkpoint = self.checkpoint()
        self.assertEqual("timeout", result["provider_error"]["kind"], result)
        self.assertEqual(0, checkpoint["observed_wait_seconds"])
        self.assertGreater(checkpoint["unclassified_provider_seconds"], 0.13)
        self.assertGreater(checkpoint["active_seconds"], 0.13)
        self.assertIsNone(unrelated.poll())
        self.evidence(result=result, checkpoint=checkpoint, unrelated_pid=unrelated.pid,
                      unrelated_alive=True, commands=commands)

    def test_unfinished_wait_requires_external_recovery_without_fabricating_a_pause(self):
        code = '''import json,os,sys,time
from pathlib import Path
project,response=map(Path,sys.argv[1:]);sys.stdin.read()
directory=Path(json.loads((project/'budget_checkpoint.json').read_text())['worker_wait_directory'])
row={'activity':'resource_admission','started_monotonic':time.monotonic(),'issuer_pid':os.getpid()}
temporary=directory/'activity.tmp';temporary.write_text(json.dumps(row));temporary.replace(directory/'activity.json')
time.sleep(.08);os._exit(0)
'''
        with ProjectBudget(self.project, self.tier) as budget:
            result, commands = self.launch_fixture(CodexCliWorker(), budget, code)
        self.assertEqual("provider_error", result["status"], result)
        self.assertTrue(result["needs_external_action"])
        self.assertEqual(0, result["recovery_attempts"])
        self.assertTrue(self.checkpoint()["worker_observation_issues"])
        self.assertLess(self.checkpoint()["observed_wait_seconds"], 0.15)
        self.assertTrue(list((self.workspace / "turn/observed-activities").rglob("activity.json")))
        self.evidence(result=result, checkpoint=self.checkpoint(), commands=commands)

    def test_unsupported_parent_observer_returns_external_recovery_before_wait(self):
        scene = self.seed_scene()
        self.policy()
        self.owner()
        with ProjectBudget(self.project) as budget, budget.active("worker_elapsed_upper_estimate"):
            before = (self.project / "budget_checkpoint.json").read_bytes()
            with self.draft_mocks(scene), patch(PREFIX + "physics_invocation.wait_for_observed_resources") as wait, \
                    patch(PREFIX + "author_tools.run_bounded_subprocess") as runner:
                result = run_tool("simulate", self.project / "runs/r01/design/scene.json", workspace=self.workspace)
            self.assertEqual(before, (self.project / "budget_checkpoint.json").read_bytes())
        self.assertEqual("RESOURCE_POLICY_REQUIRED", result["status"], result)
        self.assertTrue(result["needs_external_action"])
        wait.assert_not_called()
        runner.assert_not_called()
        self.evidence(result=result, waits=wait.call_count, solver_launches=runner.call_count)


if __name__ == "__main__":
    unittest.main()
