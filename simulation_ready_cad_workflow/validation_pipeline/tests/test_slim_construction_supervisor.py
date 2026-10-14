"""Budget-end regressions use only mocked owned children, never production PIDs."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from ..mechanism.io import write_json_atomic
from ..slim_harness import workflow as _workflow
from ..slim_harness.construction_budget import BudgetConfig, BudgetExhausted, initialize_budget
from ..slim_harness.construction_supervisor import _active_sandbox, execute_project, supervise_project
from ..benchmark_sandbox.formal_limits import POST_MODEL_CAPTURE_SECONDS


PREFIX='simulation_ready_cad_workflow.validation_pipeline.slim_harness.construction_supervisor.'


class ConstructionSupervisorDeadlineTests(unittest.TestCase):
    def test_active_record_requires_this_workflow_identity_and_unique_stop_path(self):
        with tempfile.TemporaryDirectory() as directory:
            stop=Path(directory)/'sandbox-stop-owned.json'
            path=Path(str(stop)+'.active.json')
            valid={'status':'running','workflow_pid':123,'workflow_started_at_epoch':456.,
                   'stop_file':str(stop),'controller_pid':789,'output':'public-test-path'}
            write_json_atomic(path,valid)
            self.assertEqual(_active_sandbox(stop,123,456.),valid)
            for change in ({'status':'stopped'},{'workflow_pid':124},{'workflow_started_at_epoch':457.},
                           {'workflow_started_at_epoch':float('nan')},{'stop_file':str(stop)+'-other'},
                           {'controller_pid':0}):
                path.write_text(json.dumps({**valid,**change}),encoding='utf-8')
                self.assertIsNone(_active_sandbox(stop,123,456.))
            write_json_atomic(path,valid)
            self.assertIsNone(_active_sandbox(stop,123,None))

    def cooperative_stop(self,*,grace_expires=False,trigger='budget'):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            project=workspace/'.runtime_runs/projects/cooperative';project.mkdir(parents=True)
            write_json_atomic(project/'workflow.json',{'revision':1})
            (project/'task.md').write_text('Retain partial authored work.')
            real_budget=initialize_budget(project)
            charged_before=real_budget.snapshot()['active_seconds']
            active_state={'open':False}
            lease=Mock()
            if trigger=='keyboard':lease.check_available.side_effect=KeyboardInterrupt
            @contextmanager
            def activity(_kind):
                active_state['open']=True
                try:yield lease
                finally:active_state['open']=False
            budget=Mock()
            budget.snapshot.side_effect=[{'work_remaining_seconds':1},{'work_remaining_seconds':0 if trigger=='budget' else 1}]
            budget.activity.side_effect=activity
            process=Mock(pid=123,args=['owned-workflow'],returncode=None)
            process.poll.side_effect=lambda:process.returncode
            stop_path=[];calls=[];clock=[0.]
            def launch(_command,**kwargs):
                stop=Path(kwargs['env']['CADHARNESS_SANDBOX_STOP_FILE'])
                self.assertEqual(stop.parent,project)
                self.assertFalse(stop.exists())
                from ..slim_harness import construction_supervisor
                self.assertEqual(kwargs['env']['PYTHONPATH'].split(os.pathsep)[0],
                                 str(Path(construction_supervisor.__file__).resolve().parents[3]))
                self.assertEqual(kwargs['cwd'],Path(construction_supervisor.__file__).resolve().parents[3])
                stop_path.append(stop)
                write_json_atomic(Path(str(stop)+'.active.json'),{'status':'running',
                    'workflow_pid':123,'workflow_started_at_epoch':456.,'stop_file':str(stop),
                    'controller_pid':789,'output':str(project/'worker_turns/owned')})
                if trigger=='marker':stop.touch()
                if trigger in ('sigint','sigterm'):
                    number=signal.SIGINT if trigger=='sigint' else signal.SIGTERM
                    signal.getsignal(number)(number,None)
                return process
            def wait(timeout):
                self.assertFalse(active_state['open'])
                calls.append(('wait',timeout))
                if process.returncode is None:
                    self.assertEqual(json.loads(stop_path[0].read_text())['workflow_pid'],123)
                    save_seconds=POST_MODEL_CAPTURE_SECONDS+301
                    clock[0]+=min(timeout,save_seconds)
                    if grace_expires or timeout<save_seconds:
                        raise subprocess.TimeoutExpired(process.args,timeout)
                    design=project/'runs/r01/design';design.mkdir(parents=True)
                    (design/'model.py').write_text('PRESERVED_PARTIAL_SOURCE')
                    write_json_atomic(project/'construction_result.json',{'status':'INTERRUPTED','issues':[]})
                    process.returncode=0
                return process.returncode
            def terminate(pid,seconds):
                self.assertFalse(active_state['open'])
                calls.append(('terminate',pid,seconds));process.returncode=-15
                return ()
            process.wait.side_effect=wait
            original_handlers={number:signal.getsignal(number) for number in (signal.SIGINT,signal.SIGTERM)}
            with patch(PREFIX+'budget_for',return_value=budget), \
                 patch(PREFIX+'subprocess.Popen',side_effect=launch), \
                 patch(PREFIX+'time.monotonic',side_effect=lambda:clock[0]), \
                 patch(PREFIX+'_process_started_at',return_value=456.), \
                 patch(PREFIX+'terminate_process_tree',side_effect=terminate) as stop_tree:
                result=supervise_project(workspace,project)
            receipt=json.loads((project/'construction_supervisor.json').read_text())
            self.assertEqual({number:signal.getsignal(number) for number in original_handlers},original_handlers)
            self.assertEqual(real_budget.snapshot()['active_seconds'],charged_before)
            if trigger!='budget':
                self.assertEqual(result['status'],'INTERRUPTED')
                self.assertEqual(result['stop_reason'],'operator_stop')
                self.assertEqual(receipt['status'],'operator_stop')
                self.assertIsNone(receipt['budget_decision_elapsed_wall_seconds'])
                self.assertIsNotNone(receipt['stop_decision_elapsed_wall_seconds'])
                self.assertTrue((project/'completion_summary.json').is_file())
            cooperative=receipt['cooperative_sandbox_stop']
            self.assertTrue(cooperative['requested'])
            self.assertEqual(cooperative['fallback_used'],grace_expires)
            self.assertEqual(cooperative['owned_workflow_exit_confirmed'],not grace_expires)
            self.assertGreater(calls[0][1],POST_MODEL_CAPTURE_SECONDS+300)
            self.assertEqual(cooperative['wait_seconds'],calls[0][1])
            self.assertEqual(bool(receipt['shutdown_errors']),grace_expires)
            if grace_expires:
                stop_tree.assert_called_once_with(123,2.)
                self.assertTrue(cooperative['wait_expired'])
            else:
                stop_tree.assert_not_called()
                self.assertEqual((project/'runs/r01/design/model.py').read_text(),'PRESERVED_PARTIAL_SOURCE')
                self.assertIn('runs/r01/design/model.py',result['completion']['assets'])

    def test_matching_sandbox_gets_grace_after_budget_activity_closes(self):
        self.cooperative_stop()

    def test_expired_sandbox_grace_uses_owned_fallback_and_records_uncertainty(self):
        self.cooperative_stop(grace_expires=True)

    def test_operator_marker_stops_without_consuming_or_resetting_budget(self):
        self.cooperative_stop(trigger='marker')

    def test_ctrl_c_and_sigterm_use_cooperative_cleanup_and_restore_handlers(self):
        for trigger in ('keyboard','sigint','sigterm'):
            with self.subTest(trigger=trigger):self.cooperative_stop(trigger=trigger)

    def test_owned_stop_never_retries_provider_failure_or_experiment_arm(self):
        for arm in (None,'A'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                project=Path(directory)/'.runtime_runs/projects/stopped';project.mkdir(parents=True)
                write_json_atomic(project/'workflow.json',{'request':{'experiment_arm':arm}})
                marker=project/'sandbox-stop-test.json'
                budget=Mock();budget.activity.return_value=nullcontext()
                budget.snapshot.return_value={'remaining_seconds':100}
                previous={'status':'provider_error','provider_error':{'kind':'response'}}
                def resume(*_args,**_kwargs):
                    marker.touch();return previous
                with patch.dict(os.environ,{'CADHARNESS_SANDBOX_STOP_FILE':str(marker)}), \
                        patch(PREFIX+'budget_for',return_value=budget), patch(PREFIX+'summary') as summarize, \
                        patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.resume_project',side_effect=resume) as resumed:
                    result=execute_project(Path(directory),project)
                resumed.assert_called_once()
                self.assertEqual(result['status'],'INTERRUPTED')
                self.assertEqual(result['last_result'],previous)
                self.assertEqual(result['stop_reason'],'operator_stop')
                self.assertEqual(json.loads((project/'construction_result.json').read_text()),result)
                summarize.assert_called_once()
                state=json.loads((project/'workflow.json').read_text())
                self.assertNotIn('last_construction_feedback',state)

    def test_provider_outage_waits_outside_the_budget_then_resumes_a_bounded_number_of_times(self):
        # Recorded from Codex CLI 0.162.1; the same model answered again a minute later.
        outage={'status':'provider_error','provider_error':{'kind':'exit','message':
            '{"type": "error", "message": "Selected model is at capacity. Please try a different model."}'}}
        for recovers in (True,False):
            with self.subTest(recovers=recovers), tempfile.TemporaryDirectory() as directory:
                project=Path(directory)/'.runtime_runs/projects/outage';project.mkdir(parents=True)
                write_json_atomic(project/'workflow.json',{'request':{},'phase':'author_followup','stop_reason':'provider_error'})
                budget=initialize_budget(project,BudgetConfig(hours=1))
                phases=[];waiting=[];sleep=time.sleep
                def resume(*_args,**_kwargs):
                    phases.append(json.loads((project/'workflow.json').read_text())['phase'])
                    return {'status':'PASS'} if recovers and len(phases)==3 else outage
                def wait(seconds):
                    # The runner keeps owning the project and shows the wait.
                    runtime=json.loads((project/'workflow_runtime.json').read_text())
                    waiting.append((runtime['status'],runtime['phase'],runtime['pid']))
                    with self.assertRaises(_workflow.ActiveWorkflowError):_workflow.assert_inactive(project)
                    sleep(seconds)
                environment={key:value for key,value in os.environ.items() if key!='CADHARNESS_SANDBOX_STOP_FILE'}
                with patch.dict(os.environ,environment,clear=True), patch(PREFIX+'PROVIDER_RECOVERY_RETRIES',2), \
                        patch(PREFIX+'recovery_delay',return_value=.3) as delay, patch(PREFIX+'summary'), \
                        patch(PREFIX+'time.sleep',side_effect=wait), \
                        patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.resume_project',side_effect=resume):
                    result=execute_project(Path(directory),project)
                self.assertEqual('PASS' if recovers else 'provider_error',result['status'])
                self.assertEqual(['author_followup']*3,phases)
                self.assertEqual([(0,300.),(1,300.)],[call.args for call in delay.call_args_list])
                self.assertTrue(waiting)
                self.assertEqual({('running','recovery_wait',os.getpid())},set(waiting))
                state=json.loads((project/'workflow.json').read_text())
                self.assertEqual(('author_followup',2),(state['phase'],state['recovery_attempt']))
                self.assertNotIn('recovery_from_phase',state)
                self.assertNotIn('last_construction_feedback',state)
                self.assertGreater(budget.snapshot()['excluded_by_reason'].get('provider_wait',0),.5)

    def test_operator_stop_ends_a_provider_outage_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            project=Path(directory)/'.runtime_runs/projects/outage';project.mkdir(parents=True)
            write_json_atomic(project/'workflow.json',{'request':{},'phase':'author_followup'})
            initialize_budget(project,BudgetConfig(hours=1))
            marker=project/'sandbox-stop-test.json'
            outage={'status':'provider_error','provider_error':{'kind':'exit','message':'API Error: 503 Service Unavailable'}}
            def resume(*_args,**_kwargs):
                threading.Timer(.5,marker.touch).start();return outage
            started=time.monotonic()
            with patch.dict(os.environ,{'CADHARNESS_SANDBOX_STOP_FILE':str(marker)}), patch(PREFIX+'summary'), \
                    patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.resume_project',side_effect=resume) as resumed:
                result=execute_project(Path(directory),project)
            self.assertLess(time.monotonic()-started,4)
            resumed.assert_called_once()
            self.assertEqual(('INTERRUPTED','operator_stop'),(result['status'],result['stop_reason']))
            self.assertEqual('stopped',json.loads((project/'workflow.json').read_text())['phase'])

    def test_stop_before_resume_does_not_start_worker_and_other_project_marker_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);project=root/'.runtime_runs/projects/current';project.mkdir(parents=True)
            write_json_atomic(project/'workflow.json',{'request':{}})
            marker=project/'sandbox-stop-test.json';marker.touch()
            budget=Mock();budget.activity.return_value=nullcontext()
            with patch.dict(os.environ,{'CADHARNESS_SANDBOX_STOP_FILE':str(marker)}), \
                    patch(PREFIX+'budget_for',return_value=budget), patch(PREFIX+'summary'), \
                    patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.resume_project') as resumed:
                self.assertEqual(execute_project(root,project)['stop_reason'],'operator_stop')
                resumed.assert_not_called();budget.check_available.assert_not_called()
            foreign=root/'sandbox-stop-foreign.json';foreign.touch()
            with patch.dict(os.environ,{'CADHARNESS_SANDBOX_STOP_FILE':str(foreign)}), \
                    patch(PREFIX+'budget_for',return_value=budget), patch(PREFIX+'summary'), \
                    patch('simulation_ready_cad_workflow.validation_pipeline.slim_harness.cli.resume_project',return_value={'status':'PASS'}) as resumed:
                self.assertEqual(execute_project(root,project)['status'],'PASS')
                resumed.assert_called_once()

    def test_supervised_child_import_uses_code_root_not_stale_data_workspace(self):
        from ..slim_harness import construction_supervisor
        original_popen=subprocess.Popen
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory).resolve();project=workspace/'.runtime_runs/projects/provenance';project.mkdir(parents=True)
            write_json_atomic(project/'workflow.json',{'request':{}})
            stale=workspace/'simulation_ready_cad_workflow';stale.mkdir()
            (stale/'__init__.py').write_text("raise RuntimeError('STALE_DATA_WORKSPACE_IMPORTED')\n")
            budget=Mock();budget.snapshot.return_value={'work_remaining_seconds':100};budget.activity.return_value=nullcontext(Mock())
            process=Mock(pid=987654321,returncode=0);process.poll.return_value=0
            def launch(command,**options):
                self.assertEqual(command[command.index('--workspace')+1],str(workspace))
                script=('import json,sys;from simulation_ready_cad_workflow.validation_pipeline.slim_harness '
                        'import construction_supervisor as m;print(json.dumps({"source":m.__file__,"workspace":sys.argv[1]}))')
                child=original_popen([sys.executable,'-c',script,str(workspace)],cwd=options['cwd'],env=options['env'],
                                     stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                stdout,stderr=child.communicate(timeout=20)
                self.assertEqual(child.returncode,0,stderr.decode(errors='replace'))
                observed=json.loads(stdout)
                self.assertEqual(Path(observed['source']).resolve(),Path(construction_supervisor.__file__).resolve())
                self.assertEqual(observed['workspace'],str(workspace))
                return process
            with patch(PREFIX+'budget_for',return_value=budget), patch(PREFIX+'summary',return_value={}), \
                    patch(PREFIX+'_process_started_at',return_value=None), patch(PREFIX+'subprocess.Popen',side_effect=launch):
                supervise_project(workspace,project)

    def exercise_deadline(self,*,activity_raises,wait_times_out=False,needs_kill=False):
        with tempfile.TemporaryDirectory() as directory:
            workspace=Path(directory)
            project=workspace/'.runtime_runs/projects/deadline_fixture';project.mkdir(parents=True)
            write_json_atomic(project/'workflow.json',{'revision':1})
            (project/'task.md').write_text('Build and test a mechanical asset.')
            real_budget=initialize_budget(project)
            real_budget._transaction(lambda state,now:state.update(active_seconds=7080))
            exhausted=real_budget.snapshot()
            active=Mock()
            if activity_raises:active.check_available.side_effect=BudgetExhausted(exhausted)
            budget=Mock()
            budget.snapshot.side_effect=[{'work_remaining_seconds':1},
                                         {'work_remaining_seconds':1 if activity_raises else 0}]
            budget.activity.return_value=nullcontext(active)
            calls=[]
            process=Mock(pid=987654321,args=['owned-fixture'],returncode=None)
            process.poll.side_effect=lambda:process.returncode

            def terminate(pid,grace):
                self.assertEqual(pid,process.pid)
                calls.append('terminate_tree')
                if not needs_kill:process.returncode=-15
                return ()

            def kill():
                calls.append('kill_child');process.returncode=-9

            def wait(timeout):
                calls.append('wait')
                # This reproduces the original bug if either branch waits
                # before terminating its child. Cleanup then cannot hide it.
                if process.returncode is None or wait_times_out:
                    raise subprocess.TimeoutExpired(process.args,timeout)
                return process.returncode

            process.kill.side_effect=kill
            process.wait.side_effect=wait
            with patch(PREFIX+'budget_for',return_value=budget), \
                 patch(PREFIX+'subprocess.Popen',return_value=process), \
                 patch(PREFIX+'terminate_process_tree',side_effect=terminate) as stop_tree:
                result=supervise_project(workspace,project)
            stop_tree.assert_called_once_with(process.pid,2.)
            self.assertEqual(calls[0],'terminate_tree')
            self.assertEqual(calls[-1],'wait')
            if needs_kill:self.assertEqual(calls,['terminate_tree','kill_child','wait'])
            if not activity_raises:active.check_available.assert_not_called()
            self.assertIsNotNone(process.returncode)
            self.assertEqual(result['completion']['status'],'BUDGET_EXHAUSTED_PARTIAL')
            self.assertTrue((project/'completion_summary.json').is_file())
            self.assertTrue((project/'completion_summary.md').is_file())
            state=json.loads((project/'workflow.json').read_text())
            self.assertEqual(state['phase'],'budget_exhausted')
            record=json.loads((project/'construction_supervisor.json').read_text())
            self.assertEqual(record['status'],'budget_exhausted')
            self.assertEqual(bool(record['shutdown_errors']),wait_times_out)
            if wait_times_out:
                self.assertIn('bounded shutdown wait',' '.join(result['issues']))

    def test_activity_deadline_terminates_before_wait_and_still_summarizes(self):
        self.exercise_deadline(activity_raises=True,needs_kill=True)

    def test_snapshot_deadline_uses_same_cleanup_and_summary(self):
        self.exercise_deadline(activity_raises=False)

    def test_shutdown_wait_timeout_is_reported_without_losing_completion_summary(self):
        self.exercise_deadline(activity_raises=True,wait_times_out=True)


if __name__=='__main__':unittest.main()
